/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include "asahi/compiler/agx_compile.h"
#include "libagx_shaders.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;
static unsigned failures;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);         \
         ++failures;                                                           \
      }                                                                        \
   } while (0)

static uint64_t
fnv1a64(const uint8_t *bytes, size_t size)
{
   uint64_t hash = UINT64_C(14695981039346656037);
   for (size_t i = 0; i < size; ++i) {
      hash ^= bytes[i];
      hash *= UINT64_C(1099511628211);
   }
   return hash;
}

int
main(void)
{
   /* Fresh metal-nt applegpu_g17p ground truth for the equivalent MSL kernel
    * is 54 bytes: two grid SR reads, one integer multiply-add, four value
    * moves, one 128-bit stdev and end.  Native artifact SHA-256:
    * 3474e39a65f602e913af23900d0205bd436c8c930dc4fa6f3571e539ae9437cd.
    *
    * This is the exact Mesa-ABI form of the same witnessed operations.  It
    * differs in register assignment and has one extra uniform move because
    * libagx receives its six arguments in contiguous uniform halves. */
   static const uint8_t expected[] = {
      0x0c, 0xa0, 0x10, 0x06, 0x1c, 0xa1, 0x10, 0x26,
      0x2b, 0x04, 0x01, 0x08, 0x9f, 0x30, 0x54, 0x00,
      0x02, 0x04, 0x10, 0x00, 0xd0, 0x2f, 0x2a, 0x00,
      0x4b, 0x06, 0x01, 0x08, 0x5b, 0x08, 0x01, 0x08,
      0x6b, 0x0a, 0x01, 0x08, 0x7b, 0x0c, 0x01, 0x08,
      0xe7, 0x00, 0x54, 0x08, 0x00, 0x00, 0x20, 0x00,
      0x17, 0x00, 0x00, 0x10, 0x10, 0x00, 0x0e, 0x00,
      0x00, 0x00,
   };

   for (unsigned i = 0; i < LIBAGX_NUM_PROGRAMS; ++i) {
      if (i == LIBAGX_FILL_UINT4 ||
          i == LIBAGX_COPY_IMAGE ||
          i == LIBAGX_MIPMAP_UNORM8X4 ||
          i == LIBAGX_MIPMAP_UNORM8X4_ODD ||
          i == LIBAGX_DRAW_INDIRECT_BYTE_COUNT ||
          i == LIBAGX_COPY_XFB_COUNTERS ||
          i == LIBAGX_DRAW_ROBUST_INDEX_0 ||
          i == LIBAGX_DRAW_ROBUST_INDEX_1 ||
          i == LIBAGX_DRAW_ROBUST_INDEX_2 ||
          i == LIBAGX_INCREMENT_IA ||
          i == LIBAGX_INCREMENT_IA_RESTART_NONNULL ||
          i == LIBAGX_COPY_QUERY ||
          i == LIBAGX_INCREMENT_CS_INVOCATIONS ||
          i == LIBAGX_PREDICATE_INDIRECT_0 ||
          i == LIBAGX_PREDICATE_INDIRECT_1 ||
          i == LIBAGX_CONDITIONAL_RENDER)
         CHECK(libagx_g17p[i] != NULL);
      else
         CHECK(libagx_g17p[i] == NULL);
   }

   /* G17P compressed and ordinary R8/RGBA8/BGRA8 uploads use one raw
    * copy-image launch per slice.  If this precompiled entry disappears,
    * Honeykrisp falls back to a command recording failure.  Pin its launch ABI
    * and complete generated binary so the fast path cannot silently drift
    * while the inventory test stays green. */
   const uint32_t *copy_image_blob =
      libagx_g17p[LIBAGX_COPY_IMAGE];
   CHECK(copy_image_blob != NULL);
   if (!copy_image_blob)
      return 1;
   const struct agx_precompiled_kernel_info *copy_image_info =
      (const void *)copy_image_blob;
   const uint8_t *copy_image_binary =
      (const uint8_t *)copy_image_blob + sizeof(*copy_image_info);
   static const uint8_t copy_image_end[] = {0x0e, 0x00, 0x00, 0x00};
   CHECK(copy_image_info->preamble_offset == 0);
   CHECK(copy_image_info->main_offset == 0);
   CHECK(copy_image_info->main_size == 4056);
   CHECK(copy_image_info->binary_size == 4056);
   CHECK(copy_image_info->nr_gprs == 28);
   CHECK(copy_image_info->nr_preamble_gprs == 0);
   CHECK(copy_image_info->push_count == 42);
   CHECK(copy_image_info->workgroup_size[0] == 32);
   CHECK(copy_image_info->workgroup_size[1] == 1);
   CHECK(copy_image_info->workgroup_size[2] == 1);
   CHECK(copy_image_info->local_size == 0);
   CHECK(copy_image_info->imageblock_stride == 0);
   CHECK(!copy_image_info->uses_txf);
   CHECK(copy_image_info->agx3_staging_known);
   CHECK(copy_image_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(copy_image_binary, copy_image_info->binary_size) ==
         UINT64_C(0x3c9abc0bfdcf59e5));
   CHECK(memcmp(copy_image_binary + copy_image_info->binary_size -
                   sizeof(copy_image_end),
                copy_image_end, sizeof(copy_image_end)) == 0);

   /* The G17P UNORM8 mip kernel replaces one firmware render job per level
    * with a launch in the command buffer's merged compute stream. Pin the
    * complete binary and argument ABI: losing this table entry silently puts
    * STK's thousands of startup mip renders back on the graphics fallback. */
   const uint32_t *mipmap_blob =
      libagx_g17p[LIBAGX_MIPMAP_UNORM8X4];
   CHECK(mipmap_blob != NULL);
   if (!mipmap_blob)
      return 1;
   const struct agx_precompiled_kernel_info *mipmap_info =
      (const void *)mipmap_blob;
   const uint8_t *mipmap_binary =
      (const uint8_t *)mipmap_blob + sizeof(*mipmap_info);
   CHECK(mipmap_info->preamble_offset == 0);
   CHECK(mipmap_info->main_offset == 0);
   CHECK(mipmap_info->main_size == 6506);
   CHECK(mipmap_info->binary_size == 6506);
   CHECK(mipmap_info->nr_gprs == 52);
   CHECK(mipmap_info->nr_preamble_gprs == 0);
   CHECK(mipmap_info->push_count == 28);
   CHECK(mipmap_info->workgroup_size[0] == 32);
   CHECK(mipmap_info->workgroup_size[1] == 1);
   CHECK(mipmap_info->workgroup_size[2] == 1);
   CHECK(mipmap_info->local_size == 0);
   CHECK(mipmap_info->imageblock_stride == 0);
   CHECK(!mipmap_info->uses_txf);
   CHECK(mipmap_info->agx3_staging_known);
   CHECK(mipmap_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(mipmap_binary, mipmap_info->binary_size) ==
         UINT64_C(0x21df3d4244950e1c));
   CHECK(memcmp(mipmap_binary + mipmap_info->binary_size -
                   sizeof(copy_image_end),
                copy_image_end, sizeof(copy_image_end)) == 0);

   /* The odd-extent companion consumes CPU-precomputed Q8 axes.  Its bounded
    * 32-bit kernel is substantially smaller than the discarded divide-based
    * prototype (6,430 bytes / 52 GPRs versus 14,854 / 66) while retaining the
    * same 56-byte push ABI as the even kernel. */
   const uint32_t *odd_mipmap_blob =
      libagx_g17p[LIBAGX_MIPMAP_UNORM8X4_ODD];
   CHECK(odd_mipmap_blob != NULL);
   if (!odd_mipmap_blob)
      return 1;
   const struct agx_precompiled_kernel_info *odd_mipmap_info =
      (const void *)odd_mipmap_blob;
   const uint8_t *odd_mipmap_binary =
      (const uint8_t *)odd_mipmap_blob + sizeof(*odd_mipmap_info);
   CHECK(odd_mipmap_info->preamble_offset == 0);
   CHECK(odd_mipmap_info->main_offset == 0);
   CHECK(odd_mipmap_info->main_size == 6430);
   CHECK(odd_mipmap_info->binary_size == 6430);
   CHECK(odd_mipmap_info->nr_gprs == 52);
   CHECK(odd_mipmap_info->nr_preamble_gprs == 0);
   CHECK(odd_mipmap_info->push_count == 28);
   CHECK(odd_mipmap_info->workgroup_size[0] == 32);
   CHECK(odd_mipmap_info->workgroup_size[1] == 1);
   CHECK(odd_mipmap_info->workgroup_size[2] == 1);
   CHECK(odd_mipmap_info->local_size == 0);
   CHECK(odd_mipmap_info->imageblock_stride == 0);
   CHECK(!odd_mipmap_info->uses_txf);
   CHECK(odd_mipmap_info->agx3_staging_known);
   CHECK(odd_mipmap_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(odd_mipmap_binary, odd_mipmap_info->binary_size) ==
         UINT64_C(0x61721ca114c8f636));
   CHECK(memcmp(odd_mipmap_binary + odd_mipmap_info->binary_size -
                   sizeof(copy_image_end),
                copy_image_end, sizeof(copy_image_end)) == 0);

   const uint32_t *byte_count_blob =
      libagx_g17p[LIBAGX_DRAW_INDIRECT_BYTE_COUNT];
   CHECK(byte_count_blob != NULL);
   if (!byte_count_blob)
      return 1;
   const struct agx_precompiled_kernel_info *byte_count_info =
      (const void *)byte_count_blob;
   const uint8_t *byte_count_binary =
      (const uint8_t *)byte_count_blob + sizeof(*byte_count_info);
   static const uint8_t byte_count_end[] = {0x0e, 0x00, 0x00, 0x00};
   CHECK(byte_count_info->preamble_offset == 0);
   CHECK(byte_count_info->main_offset == 0);
   CHECK(byte_count_info->main_size == 332);
   CHECK(byte_count_info->binary_size == 332);
   CHECK(byte_count_info->nr_gprs == 12);
   CHECK(byte_count_info->nr_preamble_gprs == 0);
   CHECK(byte_count_info->push_count == 16);
   CHECK(byte_count_info->workgroup_size[0] == 1);
   CHECK(byte_count_info->workgroup_size[1] == 1);
   CHECK(byte_count_info->workgroup_size[2] == 1);
   CHECK(byte_count_info->local_size == 0);
   CHECK(byte_count_info->imageblock_stride == 0);
   CHECK(!byte_count_info->uses_txf);
   CHECK(byte_count_info->agx3_staging_known);
   CHECK(byte_count_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(byte_count_binary, byte_count_info->binary_size) ==
         UINT64_C(0x54af89fd897c5ccd));
   CHECK(memcmp(byte_count_binary + byte_count_info->binary_size -
                   sizeof(byte_count_end),
                byte_count_end, sizeof(byte_count_end)) == 0);

   /* Robust indexed indirect draws are generated by these three G17P kernels,
    * specialized by U8/U16/U32 index size. Pin the complete generated programs
    * and launch ABI: a missing table entry silently turns a valid robust draw
    * into VK_ERROR_FEATURE_NOT_PRESENT at command-buffer finalization. */
   const struct {
      unsigned index;
      uint32_t size;
      uint64_t hash;
   } robust_index[] = {
      {LIBAGX_DRAW_ROBUST_INDEX_0, 3030,
       UINT64_C(0x2682982277209dfc)},
      {LIBAGX_DRAW_ROBUST_INDEX_1, 3082,
       UINT64_C(0x673af284dd4e3784)},
      {LIBAGX_DRAW_ROBUST_INDEX_2, 3080,
       UINT64_C(0x68ae07cd3c59a05a)},
   };
   static const uint8_t robust_end[] = {0x0e, 0x00, 0x00, 0x00};
   for (unsigned i = 0; i < ARRAY_SIZE(robust_index); ++i) {
      const uint32_t *robust_blob =
         libagx_g17p[robust_index[i].index];
      CHECK(robust_blob != NULL);
      if (!robust_blob)
         continue;
      const struct agx_precompiled_kernel_info *robust_info =
         (const void *)robust_blob;
      const uint8_t *robust_binary =
         (const uint8_t *)robust_blob + sizeof(*robust_info);
      CHECK(robust_info->preamble_offset == 0);
      CHECK(robust_info->main_offset == 0);
      CHECK(robust_info->main_size == robust_index[i].size);
      CHECK(robust_info->binary_size == robust_index[i].size);
      CHECK(robust_info->nr_gprs == 56);
      CHECK(robust_info->nr_preamble_gprs == 0);
      CHECK(robust_info->push_count == 28);
      CHECK(robust_info->workgroup_size[0] == 32);
      CHECK(robust_info->workgroup_size[1] == 1);
      CHECK(robust_info->workgroup_size[2] == 1);
      CHECK(robust_info->local_size == 0);
      CHECK(robust_info->imageblock_stride == 0);
      CHECK(!robust_info->uses_txf);
      CHECK(robust_info->agx3_staging_known);
      CHECK(robust_info->agx3_staging_halfs == 0);
      CHECK(fnv1a64(robust_binary, robust_info->binary_size) ==
            robust_index[i].hash);
      CHECK(memcmp(robust_binary + robust_info->binary_size -
                      sizeof(robust_end),
                   robust_end, sizeof(robust_end)) == 0);
   }

   const uint32_t *blob = libagx_g17p[LIBAGX_FILL_UINT4];
   if (!blob)
      return 1;
   const struct agx_precompiled_kernel_info *info = (const void *)blob;
   const uint8_t *binary = (const uint8_t *)blob + sizeof(*info);
   CHECK(info->preamble_offset == 0);
   CHECK(info->main_offset == 0);
   CHECK(info->main_size == sizeof(expected));
   CHECK(info->binary_size == sizeof(expected));
   CHECK(info->nr_gprs == 16);
   CHECK(info->nr_preamble_gprs == 0);
   CHECK(info->push_count == 14);
   CHECK(info->workgroup_size[0] == 32);
   CHECK(info->workgroup_size[1] == 1);
   CHECK(info->workgroup_size[2] == 1);
   CHECK(info->local_size == 0);
   CHECK(info->imageblock_stride == 0);
   CHECK(!info->uses_txf);
   CHECK(info->agx3_staging_known);
   CHECK(info->agx3_staging_halfs == 0);
   CHECK(memcmp(binary, expected, sizeof(expected)) == 0);

   /* Transform feedback now uses its dedicated native compute counter-copy
    * kernel. Its byte-exact store behavior is covered by the XFB oracle; keep
    * this table test responsible for the generated launch contract. */
   const uint32_t *xfb_blob = libagx_g17p[LIBAGX_COPY_XFB_COUNTERS];
   CHECK(xfb_blob != NULL);
   if (!xfb_blob)
      return 1;
   const struct agx_precompiled_kernel_info *xfb_info =
      (const void *)xfb_blob;
   const uint8_t *xfb_binary =
      (const uint8_t *)xfb_blob + sizeof(*xfb_info);
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
   CHECK(xfb_info->preamble_offset == 0);
   CHECK(xfb_info->main_offset == 0);
   CHECK(xfb_info->main_size == 176);
   CHECK(xfb_info->binary_size == 176);
   CHECK(xfb_info->nr_gprs == 12);
   CHECK(xfb_info->push_count == 4);
   CHECK(xfb_info->workgroup_size[0] == 4);
   CHECK(xfb_info->workgroup_size[1] == 1);
   CHECK(xfb_info->workgroup_size[2] == 1);
   CHECK(xfb_info->local_size == 0);
   CHECK(!xfb_info->uses_txf);
   CHECK(xfb_info->agx3_staging_known);
   CHECK(xfb_info->agx3_staging_halfs == 0);
   CHECK(memcmp(xfb_binary + xfb_info->binary_size - sizeof(end), end,
                sizeof(end)) == 0);

   /* Pipeline-statistics IA accounting uses this exact generated kernel. Its
    * register-register-plus-immediate MAD is independently pinned against the
    * fresh Apple G17P witness in test-agx3-iadd-core.c. Keep the whole Mesa ABI
    * program stable as well: it is launched before the render and updates only
    * the non-null counter addresses supplied by the query root. */
   const uint32_t *ia_blob = libagx_g17p[LIBAGX_INCREMENT_IA];
   CHECK(ia_blob != NULL);
   if (!ia_blob)
      return 1;
   const struct agx_precompiled_kernel_info *ia_info = (const void *)ia_blob;
   const uint8_t *ia_binary =
      (const uint8_t *)ia_blob + sizeof(*ia_info);
   CHECK(ia_info->preamble_offset == 0);
   CHECK(ia_info->main_offset == 0);
   CHECK(ia_info->main_size == 1360);
   CHECK(ia_info->binary_size == 1360);
   CHECK(ia_info->nr_gprs == 26);
   CHECK(ia_info->nr_preamble_gprs == 0);
   CHECK(ia_info->push_count == 28);
   CHECK(ia_info->workgroup_size[0] == 1);
   CHECK(ia_info->workgroup_size[1] == 1);
   CHECK(ia_info->workgroup_size[2] == 1);
   CHECK(ia_info->local_size == 0);
   CHECK(ia_info->imageblock_stride == 0);
   CHECK(!ia_info->uses_txf);
   CHECK(ia_info->agx3_staging_known);
   CHECK(ia_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(ia_binary, ia_info->binary_size) ==
         UINT64_C(0x3d4ea3afaa2e9e27));
   CHECK(memcmp(ia_binary + ia_info->binary_size - sizeof(end), end,
                sizeof(end)) == 0);

   /* These are the remaining helpers needed by the public G17P pipeline-
    * statistics promise: restart-aware graphics accounting, Vulkan result
    * copies, and compute-invocation accounting. Pin each complete generated
    * program and its launch ABI, not merely the non-null dispatch-table slot. */
   const struct {
      unsigned index;
      uint32_t main_offset;
      uint32_t main_size;
      uint32_t binary_size;
      uint16_t nr_gprs;
      uint16_t push_count;
      uint16_t workgroup_size[3];
      uint16_t local_size;
      uint64_t hash;
   } statistics[] = {
      /* Two bytes longer than it used to be, and one hash apart: the restart
       * kernel breaks out of its loop from inside an enclosing block, and
       * that break moved off the four-byte CNDLT (which spends only the
       * loop's own two exec-mask levels) onto the six-byte CNDCF that carries
       * a level count. The single added instruction is
       * `8f 05 54 03 00 09` at +0x2c8 -- slot 1, inverted, three levels --
       * which is byte-for-byte one of Apple's own applegpu_g17p encodings
       * (see agx3_cf_cndcf()). Nothing else in the program moved. */
      {LIBAGX_INCREMENT_IA_RESTART_NONNULL, 128, 2940, 3068, 48, 40,
       {1024, 1, 1}, 4, UINT64_C(0x25b99a6f3e0f6323)},
      {LIBAGX_COPY_QUERY, 0, 890, 890, 28, 25, {32, 1, 1}, 0,
       UINT64_C(0x3b6988993fef71a1)},
      {LIBAGX_INCREMENT_CS_INVOCATIONS, 0, 86, 86, 8, 10, {1, 1, 1}, 0,
       UINT64_C(0xd292b1db166dd1e9)},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(statistics); ++i) {
      const uint32_t *statistics_blob = libagx_g17p[statistics[i].index];
      CHECK(statistics_blob != NULL);
      if (!statistics_blob)
         continue;
      const struct agx_precompiled_kernel_info *statistics_info =
         (const void *)statistics_blob;
      const uint8_t *statistics_binary =
         (const uint8_t *)statistics_blob + sizeof(*statistics_info);
      const uint64_t hash =
         fnv1a64(statistics_binary, statistics_info->binary_size);
      CHECK(statistics_info->preamble_offset == 0);
      CHECK(statistics_info->main_offset == statistics[i].main_offset);
      CHECK(statistics_info->main_size == statistics[i].main_size);
      CHECK(statistics_info->binary_size == statistics[i].binary_size);
      CHECK(statistics_info->nr_gprs == statistics[i].nr_gprs);
      CHECK(statistics_info->nr_preamble_gprs == 0);
      CHECK(statistics_info->push_count == statistics[i].push_count);
      CHECK(statistics_info->workgroup_size[0] ==
            statistics[i].workgroup_size[0]);
      CHECK(statistics_info->workgroup_size[1] ==
            statistics[i].workgroup_size[1]);
      CHECK(statistics_info->workgroup_size[2] ==
            statistics[i].workgroup_size[2]);
      CHECK(statistics_info->local_size == statistics[i].local_size);
      CHECK(statistics_info->imageblock_stride == 0);
      CHECK(!statistics_info->uses_txf);
      CHECK(statistics_info->agx3_staging_known);
      CHECK(statistics_info->agx3_staging_halfs == 0);
      CHECK(hash == statistics[i].hash);
      CHECK(memcmp(statistics_binary + statistics_info->binary_size -
                      sizeof(end),
                   end, sizeof(end)) == 0);
   }

   /* Apple G17P compiles the equivalent four- and five-word predicate/copy
    * kernels to the same control, address and device-store instruction
    * families used here. Fresh AIR/native SHA-256 pairs:
    *   ab91101e4182bd1656ac7602ad258174d1576451f2de55f770c64ad851ecc606
    *   b30f7801c83790963138415bf16dc9770faa22a00bb5bf08add355fd7a127fa3
    * Keep both generated variants byte-exact: the count path must never
    * silently fall back to an AGX2 precompiled binary on HAL200. */
   const struct {
      unsigned index;
      uint32_t size;
      uint64_t hash;
   } predicate[] = {
      {LIBAGX_PREDICATE_INDIRECT_0, 668, UINT64_C(0x3b6d443be271fbfb)},
      {LIBAGX_PREDICATE_INDIRECT_1, 814, UINT64_C(0x33746e6fd2961f9f)},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(predicate); ++i) {
      const uint32_t *predicate_blob = libagx_g17p[predicate[i].index];
      CHECK(predicate_blob != NULL);
      if (!predicate_blob)
         continue;
      const struct agx_precompiled_kernel_info *predicate_info =
         (const void *)predicate_blob;
      const uint8_t *predicate_binary =
         (const uint8_t *)predicate_blob + sizeof(*predicate_info);
      CHECK(predicate_info->preamble_offset == 0);
      CHECK(predicate_info->main_offset == 0);
      CHECK(predicate_info->main_size == predicate[i].size);
      CHECK(predicate_info->binary_size == predicate[i].size);
      CHECK(predicate_info->nr_gprs == 24);
      CHECK(predicate_info->nr_preamble_gprs == 0);
      CHECK(predicate_info->push_count == 14);
      CHECK(predicate_info->workgroup_size[0] == 32);
      CHECK(predicate_info->workgroup_size[1] == 1);
      CHECK(predicate_info->workgroup_size[2] == 1);
      CHECK(predicate_info->local_size == 0);
      CHECK(predicate_info->imageblock_stride == 0);
      CHECK(!predicate_info->uses_txf);
      CHECK(predicate_info->agx3_staging_known);
      CHECK(predicate_info->agx3_staging_halfs == 0);
      CHECK(fnv1a64(predicate_binary, predicate_info->binary_size) ==
            predicate[i].hash);
      CHECK(memcmp(predicate_binary + predicate_info->binary_size -
                      sizeof(end),
                   end, sizeof(end)) == 0);
   }

   /* Vulkan conditional rendering has no Metal API witness and no native VDM
    * predicate packet. Its required G17P implementation is this exact
    * compiler-produced 32-bit predicate/copy kernel followed by an already-
    * qualified native indirect draw or dispatch. Pin the complete generated
    * shader and launch ABI so it can never fall back to an AGX2 binary. */
   const uint32_t *conditional_blob =
      libagx_g17p[LIBAGX_CONDITIONAL_RENDER];
   CHECK(conditional_blob != NULL);
   if (!conditional_blob)
      return 1;
   const struct agx_precompiled_kernel_info *conditional_info =
      (const void *)conditional_blob;
   const uint8_t *conditional_binary =
      (const uint8_t *)conditional_blob + sizeof(*conditional_info);
   CHECK(conditional_info->preamble_offset == 0);
   CHECK(conditional_info->main_offset == 0);
   CHECK(conditional_info->main_size == 170);
   CHECK(conditional_info->binary_size == 170);
   CHECK(conditional_info->nr_gprs == 6);
   CHECK(conditional_info->nr_preamble_gprs == 0);
   CHECK(conditional_info->push_count == 16);
   CHECK(conditional_info->workgroup_size[0] == 1);
   CHECK(conditional_info->workgroup_size[1] == 1);
   CHECK(conditional_info->workgroup_size[2] == 1);
   CHECK(conditional_info->local_size == 0);
   CHECK(conditional_info->imageblock_stride == 0);
   CHECK(!conditional_info->uses_txf);
   CHECK(conditional_info->agx3_staging_known);
   CHECK(conditional_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(conditional_binary, conditional_info->binary_size) ==
         UINT64_C(0xdd78584e246b702d));
   CHECK(memcmp(conditional_binary + conditional_info->binary_size -
                   sizeof(end),
                end, sizeof(end)) == 0);

   if (failures) {
      fprintf(stderr, "g17p-precompiled-fill: %u/%u checks failed\n",
              failures, checks);
      return 1;
   }
   printf("g17p-precompiled-fill: %u checks passed\n", checks);
   return 0;
}
