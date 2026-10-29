/* SPDX-License-Identifier: MIT */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_cmd_buffer.h"

static uint64_t checks;
#define CHECK(condition)                                                       \
   do {                                                                        \
      ++checks;                                                                \
      if (!(condition)) {                                                      \
         fprintf(stderr, "line %d: %s\n", __LINE__, #condition);               \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

/* Independent reference: select by comparing the actual published roots. */
static uint64_t
fresh_select(struct hk_graphics_state *gfx, uint8_t reads)
{
   const uint8_t hidden = HK_ROOT_READS_ADDRESS | HK_ROOT_READS_VERTEX_INPUT |
                          HK_ROOT_READS_DYNAMIC_BUFFERS;
   if (gfx->fs_root && gfx->fs_root != gfx->root && gfx->root_cpu &&
       !(reads & hidden) &&
       hk_root_matches_for_fragment(gfx->fs_root_cpu, gfx->root_cpu))
      return gfx->fs_root;
   gfx->fs_root = gfx->root;
   gfx->fs_root_cpu = gfx->root_cpu;
   return gfx->root;
}

static void
select_and_check(struct hk_graphics_state *fresh,
                 struct hk_graphics_state *cached, uint8_t reads, bool memo,
                 bool shadow)
{
   CHECK(fresh_select(fresh, reads) ==
         hk_m5_fragment_root_select(cached, reads, memo, shadow));
   CHECK(fresh->fs_root == cached->fs_root);
   CHECK(fresh->fs_root_cpu == cached->fs_root_cpu);
}

/* Publication uses a mutable cached CPU source and an immutable GPU copy.
 * Poison skipped dynamic bytes so accidental observation cannot pass. */
static void
publish(struct hk_graphics_state *fresh, struct hk_graphics_state *cached,
        struct hk_root_descriptor_table *gpu,
        const struct hk_root_descriptor_table *source, unsigned dynamic_end,
        uint64_t va, bool shadow)
{
   const size_t head = hk_root_descriptor_offset(dynamic_buffers) +
                       dynamic_end * sizeof(source->dynamic_buffers[0]);
   const size_t tail = hk_root_descriptor_offset(set_dynamic_buffer_start);
   memset(gpu, 0xa7, sizeof(*gpu));
   if (shadow)
      hk_root_fragment_snapshot_publish(cached, gpu, source);
   memcpy(gpu, source, head);
   memcpy((uint8_t *)gpu + tail, (const uint8_t *)source + tail,
          sizeof(*gpu) - tail);
   fresh->root = cached->root = va;
   fresh->root_cpu = cached->root_cpu = gpu;
   if (shadow) {
      size_t out = 0;
      for (size_t b = 0; b < sizeof(*gpu); ++b) {
         if (!hk_root_reads_of(b, b + 1)) {
            CHECK(cached->root_fragment_snapshot.bytes[out] ==
                  ((const uint8_t *)gpu)[b]);
            ++out;
         }
      }
      CHECK(out == HK_ROOT_FRAGMENT_VISIBLE_SIZE);
   }
}

int
main(void)
{
   struct hk_graphics_state *fresh = calloc(1, sizeof(*fresh));
   struct hk_graphics_state *cached = calloc(1, sizeof(*cached));
   struct hk_root_descriptor_table source, gpu[35];
   CHECK(fresh && cached);

   for (unsigned mode = 0; mode < 4; ++mode) {
      const bool memo = mode & 1, shadow = mode & 2;
      for (unsigned end = 0; end <= HK_MAX_DYNAMIC_BUFFERS; ++end) {
         for (unsigned byte = 0; byte < sizeof(source); ++byte) {
            memset(fresh, 0, sizeof(*fresh));
            memset(cached, 0, sizeof(*cached));
            memset(&source, 0, sizeof(source));
            publish(fresh, cached, &gpu[0], &source, end, 0x1000, shadow);
            select_and_check(fresh, cached, 0, memo, shadow);
            const struct hk_root_fragment_snapshot prior =
               cached->fs_root_fragment_snapshot;
            const struct hk_root_descriptor_table *prior_cpu =
               cached->fs_root_fragment_snapshot_cpu;
            ((uint8_t *)&source)[byte] ^= 0x5a;
            publish(fresh, cached, &gpu[1], &source, end, 0x2000, shadow);
            for (unsigned reads = 0; reads < 8; ++reads) {
               /* Each mask must decide between the old and current roots.
                * A previous hidden-read selection must not pre-adopt it. */
               fresh->fs_root = cached->fs_root = 0x1000;
               fresh->fs_root_cpu = cached->fs_root_cpu = &gpu[0];
               cached->fs_root_checked_root = 0;
               cached->fs_root_fragment_snapshot = prior;
               cached->fs_root_fragment_snapshot_cpu = prior_cpu;
               select_and_check(fresh, cached, reads, memo, shadow);
               select_and_check(fresh, cached, reads, memo, shadow);
            }
         }
      }

      /* Every publication-time global and the complete tail are visible. */
      memset(fresh, 0, sizeof(*fresh));
      memset(cached, 0, sizeof(*cached));
      memset(&source, 0, sizeof(source));
      for (unsigned globals = 0; globals < 32; ++globals) {
         source.root_desc_addr = 0x100000000ull + globals;
         source.printf_buffer_address = globals & 1 ? 0x12340000 : 0;
         source.sampler_heap_address = globals & 2 ? 0x23450000 : 0;
         source.physical_sparse_map = globals & 4 ? 0x34560000 : 0;
         source.physical_sparse_base = globals & 8 ? 0x45670000 : 0;
         source.physical_sparse_size = globals & 16 ? 0x100000 : 0;
         publish(fresh, cached, &gpu[globals], &source, 0,
                 0x3000 + globals * 0x1000, shadow);
         select_and_check(fresh, cached, 0, memo, shadow);
      }

      /* Mutable descriptor restoration cannot change a published snapshot. */
      memset(fresh, 0, sizeof(*fresh));
      memset(cached, 0, sizeof(*cached));
      memset(&source, 0, sizeof(source));
      publish(fresh, cached, &gpu[0], &source, 0, 0x1000, shadow);
      select_and_check(fresh, cached, 0, memo, shadow);
      publish(fresh, cached, &gpu[1], &source, 0, 0x2000, shadow);
      memset(&source, 0x6d, sizeof(source));
      select_and_check(fresh, cached, 0, memo, shadow);
      select_and_check(fresh, cached, 7, memo, shadow);

      /* Failed publication retains the old CPU copy while its GPU VA is 0. */
      source.push[0] ^= 1;
      publish(fresh, cached, &gpu[2], &source, 64, 0x3000, shadow);
      fresh->root = cached->root = 0;
      select_and_check(fresh, cached, 0, memo, shadow);
      select_and_check(fresh, cached, 7, memo, shadow);
      cached->root_fragment_snapshot_cpu = NULL;
      select_and_check(fresh, cached, 0, memo, shadow);

      /* Missing current/prior metadata must use the fresh comparison. */
      memset(fresh, 0, sizeof(*fresh));
      memset(cached, 0, sizeof(*cached));
      memset(&source, 0, sizeof(source));
      publish(fresh, cached, &gpu[0], &source, 0, 0x1000, shadow);
      select_and_check(fresh, cached, 0, memo, shadow);
      source.push[0] = 1;
      publish(fresh, cached, &gpu[1], &source, 0, 0x2000, shadow);
      cached->root_fragment_snapshot_cpu = NULL;
      select_and_check(fresh, cached, 0, memo, shadow);
      publish(fresh, cached, &gpu[2], &source, 0, 0x3000, shadow);
      cached->fs_root_fragment_snapshot_cpu = NULL;
      select_and_check(fresh, cached, 0, memo, shadow);

      /* Command-buffer reset permits both pointer and VA reuse. */
      memset(fresh, 0, sizeof(*fresh));
      memset(cached, 0, sizeof(*cached));
      publish(fresh, cached, &gpu[0], &source, 0, 0x1000, shadow);
      select_and_check(fresh, cached, 0, memo, shadow);
      fresh->root_cpu = cached->root_cpu = NULL;
      fresh->root = cached->root = 0;
      select_and_check(fresh, cached, 0, memo, shadow);
   }
   printf("root fragment: %" PRIu64 " checks passed\n", checks);
   free(fresh);
   free(cached);
   return 0;
}
