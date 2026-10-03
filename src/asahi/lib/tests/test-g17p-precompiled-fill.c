/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include "asahi/compiler/agx_compile.h"
#include "libagx_shaders.h"

#include <inttypes.h>
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
    * differs in register assignment and argument placement. Native uniform
    * IMAD avoids the old multiplicand copy. This fingerprint uses the current
    * Mesa ABI; the encoder's independent native-byte tests cover that form. */
   static const uint8_t expected[] = {
      0x0c, 0xa0, 0x10, 0x06, 0x1c, 0xa1, 0x10, 0x26,
      0x9f, 0x30, 0x54, 0x00, 0x02, 0x20, 0x08, 0x00,
      0xd0, 0x2f, 0x28, 0x00, 0x4b, 0x12, 0x01, 0x08,
      0x5b, 0x14, 0x01, 0x08, 0x6b, 0x16, 0x01, 0x08,
      0x7b, 0x18, 0x01, 0x08, 0xe7, 0x00, 0x54, 0x08,
      0x03, 0x00, 0x20, 0x00, 0x17, 0x00, 0x00, 0x10,
      0x10, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   for (unsigned i = 0; i < LIBAGX_NUM_PROGRAMS; ++i) {
      if (i == LIBAGX_FILL_UINT4 ||
          /* Qualified with G17P lossless colour compression (2a747b85c24):
           * compressed render-pass clears dispatch it for metadata/body. */
          (i >= LIBAGX_FAST_CLEAR_0 && i <= LIBAGX_FAST_CLEAR_14) ||
          /* Adjacency index materialisation, qualified in
           * target_supports_program() with its host oracle
           * (test-poly-adj-map). */
          i == LIBAGX_WRITE_ADJ_INDICES ||
          i == LIBAGX_WRITE_ADJ_INDICES_INDIRECT ||
          /* Qualified in agx_supports_precompiled() (src/asahi/clc/
           * asahi_clc.c) after this list was first written.  This list is a
           * deliberate second copy: it is a tripwire, so a kernel that starts
           * being precompiled for G17P without anyone deciding to qualify it
           * fails here.  Keep the two in step by hand. */
          i == LIBAGX_FILL ||
          i == LIBAGX_COPY_UINT4 ||
          i == LIBAGX_COPY_UCHAR ||
          i == LIBAGX_COPY_QUERY_GL ||
          i == LIBAGX_INCREMENT_IA_RESTART ||
          i == LIBAGX_SETUP_INDIRECT_DISPATCH ||
          /* Qualified for its arithmetic oracle only; its machine code has
           * never executed, and the ray-tracing-pipeline path it serves is
           * behind HK_G17P_UNVERIFIED_FEATURES.  Presence here is not a claim
           * that it runs -- see the comment on it in agx_supports_precompiled(). */
          i == LIBAGX_TRACE_RAYS_INDIRECT_SETUP ||
          i == LIBAGX_COPY_IMAGE ||
          i == LIBAGX_MIPMAP_UNORM8X4 ||
          i == LIBAGX_MIPMAP_UNORM8X4_ODD ||
          i == LIBAGX_RESOLVE_UNORM8X4 ||
          i == LIBAGX_DRAW_INDIRECT_BYTE_COUNT ||
          i == LIBAGX_COPY_XFB_COUNTERS ||
          i == LIBAGX_DRAW_ROBUST_INDEX_0 ||
          i == LIBAGX_DRAW_ROBUST_INDEX_1 ||
          i == LIBAGX_DRAW_ROBUST_INDEX_2 ||
          i == LIBAGX_INCREMENT_IA ||
          i == LIBAGX_INCREMENT_IA_RESTART_NONNULL ||
          i == LIBAGX_COPY_QUERY ||
          i == LIBAGX_INCREMENT_CS_INVOCATIONS ||
          i == LIBAGX_COPY_TIMESTAMP ||
          i == LIBAGX_PREDICATE_INDIRECT_0 ||
          i == LIBAGX_PREDICATE_INDIRECT_1 ||
          i == LIBAGX_CONDITIONAL_RENDER ||
          i == LIBAGX_DRAW_WITHOUT_ADJ ||
          (i >= LIBAGX_UNROLL_RESTART_0 &&
           i <= LIBAGX_UNROLL_RESTART_10) ||
          i == LIBAGX_GS_SETUP_INDIRECT ||
          i == LIBAGX_PREFIX_SUM_GEOM ||
          i == LIBAGX_G17P_LOCAL_PROBE ||
          i == LIBAGX_G17P_LOCAL_SG_PROBE ||
          (i >= LIBAGX_PREFIX_SUM_TESS_0 &&
           i <= LIBAGX_PREFIX_SUM_TESS_1) ||
          i == LIBAGX_INCREMENT_STATISTIC ||
          i == LIBAGX_TESS_SETUP_INDIRECT ||
          i == LIBAGX_TESS_SETUP_INDIRECT_HW ||
          i == LIBAGX_TESS_FACTORS_HALF ||
          (i >= LIBAGX_TESS_ISOLINE_0 && i <= LIBAGX_TESS_QUAD_1))
         CHECK(libagx_g17p[i] != NULL);
      else
         CHECK(libagx_g17p[i] == NULL);

      if (libagx_g17p[i]) {
         const struct agx_precompiled_kernel_info *info =
            (const void *)libagx_g17p[i];
         CHECK(info->preamble_size == 0);
      }
   }

   /* Every primitive-restart variant is one 1024-thread workgroup and uses
    * the same 128-byte local ballot array.  The G17P precompiled dispatcher
    * consumes this metadata to allocate threadgroup memory in its Ioto
    * program, so pin it independently of the generated machine code. */
   for (unsigned i = LIBAGX_UNROLL_RESTART_0;
        i <= LIBAGX_UNROLL_RESTART_10; ++i) {
      const struct agx_precompiled_kernel_info *unroll_info =
         (const void *)libagx_g17p[i];
      CHECK(unroll_info->workgroup_size[0] == 1024);
      CHECK(unroll_info->workgroup_size[1] == 1);
      CHECK(unroll_info->workgroup_size[2] == 1);
      CHECK(unroll_info->local_size == 128);
   }

   /* Tessellation's two prefix variants exchange 32 subgroup totals through
    * the same shared allocation before generating the indirect draw. */
   for (unsigned i = LIBAGX_PREFIX_SUM_TESS_0;
        i <= LIBAGX_PREFIX_SUM_TESS_1; ++i) {
      const struct agx_precompiled_kernel_info *info =
         (const void *)libagx_g17p[i];
      CHECK(info->workgroup_size[0] == 1024);
      CHECK(info->local_size == 128);
   }

   /* The two threadgroup-memory probes carry the same launch shape as the
    * tessellation prefix sum they exist to compare against: one 1024-thread
    * workgroup, and a shared allocation the precompiled dispatcher must
    * publish (4096 B for the 1024-slot array, 128 B for the 32-subgroup
    * exchange). They are never dispatched by the driver; see geometry.cl. */
   {
      const struct agx_precompiled_kernel_info *probe =
         (const void *)libagx_g17p[LIBAGX_G17P_LOCAL_PROBE];
      CHECK(probe != NULL);
      if (probe) {
         CHECK(probe->workgroup_size[0] == 1024);
         CHECK(probe->local_size == 4096);
      }
      const struct agx_precompiled_kernel_info *sg_probe =
         (const void *)libagx_g17p[LIBAGX_G17P_LOCAL_SG_PROBE];
      CHECK(sg_probe != NULL);
      if (sg_probe) {
         CHECK(sg_probe->workgroup_size[0] == 1024);
         CHECK(sg_probe->local_size == 128);
      }
   }

   /* The fingerprints below were re-pinned against agx3 92a8b2d13e3, whose
    * generated libagx_shaders.c is byte-identical to the int4fix desktop
    * build that passed the Neo broad CTS gate (2026-10-01). Since the
    * previous pins the default G17P code generation (253b68945fd v56h,
    * 2b2e972d60b "Mesa W") promotes integer constants into a 128-byte block
    * ahead of main (main_offset 128, larger push_count) and the internal
    * kernels' argument uniforms moved up 12 halves; sizes and register counts
    * changed with them. Byte counts quoted in the comments describe the
    * kernels as first pinned. */

   /* G17P compressed and ordinary R8/RGBA8/BGRA8 uploads use one raw
    * copy-image launch per slice. Its native 2D grid and CPU-reduced tile
    * geometry eliminate per-element division, trimming the binary from 4,056
    * to 1,670 bytes and register use from 28 to 22 GPRs. If this precompiled
    * entry disappears, Honeykrisp falls back to a command recording failure.
    * Pin its launch ABI and complete generated binary so the fast path cannot
    * silently drift while the inventory test stays green. */
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
   CHECK(copy_image_info->main_size == 1414);
   CHECK(copy_image_info->binary_size == 1414);
   CHECK(copy_image_info->nr_gprs == 22);
   CHECK(copy_image_info->nr_preamble_gprs == 0);
   CHECK(copy_image_info->push_count == 54);
   CHECK(copy_image_info->workgroup_size[0] == 32);
   CHECK(copy_image_info->workgroup_size[1] == 1);
   CHECK(copy_image_info->workgroup_size[2] == 1);
   CHECK(copy_image_info->local_size == 0);
   CHECK(copy_image_info->imageblock_stride == 0);
   CHECK(!copy_image_info->uses_txf);
   CHECK(copy_image_info->agx3_staging_known);
   CHECK(copy_image_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(copy_image_binary, copy_image_info->binary_size) ==
         UINT64_C(0x75d08f991b654176));
   CHECK(memcmp(copy_image_binary + copy_image_info->binary_size -
                   sizeof(copy_image_end),
                copy_image_end, sizeof(copy_image_end)) == 0);

   /* The G17P Vulkan resolve path dispatches this raw RGBA8 kernel for 2x and
    * 4x images. Pin the complete binary and its 84-byte argument ABI so a
    * compiler change cannot silently alter the hardware-tested program. */
   const uint32_t *resolve_blob =
      libagx_g17p[LIBAGX_RESOLVE_UNORM8X4];
   CHECK(resolve_blob != NULL);
   if (!resolve_blob)
      return 1;
   const struct agx_precompiled_kernel_info *resolve_info =
      (const void *)resolve_blob;
   const uint8_t *resolve_binary =
      (const uint8_t *)resolve_blob + sizeof(*resolve_info);
   static const uint8_t resolve_end[] = {0x0e, 0x00, 0x00, 0x00};
   CHECK(resolve_info->preamble_offset == 0);
   CHECK(resolve_info->main_offset == 128);
   CHECK(resolve_info->main_size == 3854);
   CHECK(resolve_info->binary_size == 3982);
   CHECK(resolve_info->nr_gprs == 36);
   CHECK(resolve_info->nr_preamble_gprs == 0);
   CHECK(resolve_info->push_count == 70);
   CHECK(resolve_info->workgroup_size[0] == 32);
   CHECK(resolve_info->workgroup_size[1] == 1);
   CHECK(resolve_info->workgroup_size[2] == 1);
   CHECK(resolve_info->local_size == 0);
   CHECK(resolve_info->imageblock_stride == 0);
   CHECK(!resolve_info->uses_txf);
   CHECK(resolve_info->agx3_staging_known);
   CHECK(resolve_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(resolve_binary, resolve_info->binary_size) ==
         UINT64_C(0xf89623416b66406d));
   CHECK(memcmp(resolve_binary + resolve_info->binary_size -
                   sizeof(resolve_end),
                resolve_end, sizeof(resolve_end)) == 0);

   /* The G17P UNORM8 mip kernel replaces one firmware render job per level
    * with a launch in the command buffer's merged compute stream. Its native
    * 2D grid and CPU-reduced tile geometry avoid all per-pixel division,
    * trimming the binary from 6,506 to 3,234 bytes and register use from 52 to
    * 38 GPRs.
    * Pin the complete binary and argument ABI: losing this table entry
    * silently puts STK's thousands of startup mip renders back on the
    * graphics fallback. */
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
   CHECK(mipmap_info->main_size == 2830);
   CHECK(mipmap_info->binary_size == 2830);
   CHECK(mipmap_info->nr_gprs == 34);
   CHECK(mipmap_info->nr_preamble_gprs == 0);
   CHECK(mipmap_info->push_count == 40);
   CHECK(mipmap_info->workgroup_size[0] == 32);
   CHECK(mipmap_info->workgroup_size[1] == 1);
   CHECK(mipmap_info->workgroup_size[2] == 1);
   CHECK(mipmap_info->local_size == 0);
   CHECK(mipmap_info->imageblock_stride == 0);
   CHECK(!mipmap_info->uses_txf);
   CHECK(mipmap_info->agx3_staging_known);
   CHECK(mipmap_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(mipmap_binary, mipmap_info->binary_size) ==
         UINT64_C(0x116791239c583bfd));
   CHECK(memcmp(mipmap_binary + mipmap_info->binary_size -
                   sizeof(copy_image_end),
                copy_image_end, sizeof(copy_image_end)) == 0);

   /* The odd-extent companion consumes CPU-precomputed Q8 axes and reduced
    * tile geometry. Its bounded 32-bit kernel is substantially smaller than
    * the discarded divide-based prototype (3,474 bytes / 36 GPRs versus
    * 14,854 / 66) while retaining the same 56-byte push ABI as the even
    * kernel. */
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
   CHECK(odd_mipmap_info->main_offset == 128);
   CHECK(odd_mipmap_info->main_size == 3084);
   CHECK(odd_mipmap_info->binary_size == 3212);
   CHECK(odd_mipmap_info->nr_gprs == 36);
   CHECK(odd_mipmap_info->nr_preamble_gprs == 0);
   CHECK(odd_mipmap_info->push_count == 44);
   CHECK(odd_mipmap_info->workgroup_size[0] == 32);
   CHECK(odd_mipmap_info->workgroup_size[1] == 1);
   CHECK(odd_mipmap_info->workgroup_size[2] == 1);
   CHECK(odd_mipmap_info->local_size == 0);
   CHECK(odd_mipmap_info->imageblock_stride == 0);
   CHECK(!odd_mipmap_info->uses_txf);
   CHECK(odd_mipmap_info->agx3_staging_known);
   CHECK(odd_mipmap_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(odd_mipmap_binary, odd_mipmap_info->binary_size) ==
         UINT64_C(0xb399b4b4e1828d33));
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
   CHECK(byte_count_info->main_offset == 128);
   CHECK(byte_count_info->main_size == 304);
   CHECK(byte_count_info->binary_size == 432);
   CHECK(byte_count_info->nr_gprs == 12);
   CHECK(byte_count_info->nr_preamble_gprs == 0);
   CHECK(byte_count_info->push_count == 32);
   CHECK(byte_count_info->workgroup_size[0] == 1);
   CHECK(byte_count_info->workgroup_size[1] == 1);
   CHECK(byte_count_info->workgroup_size[2] == 1);
   CHECK(byte_count_info->local_size == 0);
   CHECK(byte_count_info->imageblock_stride == 0);
   CHECK(!byte_count_info->uses_txf);
   CHECK(byte_count_info->agx3_staging_known);
   CHECK(byte_count_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(byte_count_binary, byte_count_info->binary_size) ==
         UINT64_C(0x70b42dd787c3d837));
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
      uint16_t push_count;
      uint64_t hash;
   } robust_index[] = {
      {LIBAGX_DRAW_ROBUST_INDEX_0, 3332, 54,
       UINT64_C(0x8781d3fa33d42e7e)},
      {LIBAGX_DRAW_ROBUST_INDEX_1, 3394, 58,
       UINT64_C(0x862e57e86dbac1c8)},
      {LIBAGX_DRAW_ROBUST_INDEX_2, 3394, 56,
       UINT64_C(0xa79a191219121586)},
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
      const uint64_t robust_hash =
         fnv1a64(robust_binary, robust_info->binary_size);
      if (robust_info->binary_size != robust_index[i].size ||
          robust_hash != robust_index[i].hash)
         fprintf(stderr,
                 "robust-index[%u]: size=%u hash=%#" PRIx64 "\n", i,
                 robust_info->binary_size, robust_hash);
      CHECK(robust_info->preamble_offset == 0);
      CHECK(robust_info->main_offset == 128);
      CHECK(robust_info->main_size == robust_index[i].size - 128);
      CHECK(robust_info->binary_size == robust_index[i].size);
      CHECK(robust_info->nr_gprs == 56);
      CHECK(robust_info->nr_preamble_gprs == 0);
      CHECK(robust_info->push_count == robust_index[i].push_count);
      CHECK(robust_info->workgroup_size[0] == 32);
      CHECK(robust_info->workgroup_size[1] == 1);
      CHECK(robust_info->workgroup_size[2] == 1);
      CHECK(robust_info->local_size == 0);
      CHECK(robust_info->imageblock_stride == 0);
      CHECK(!robust_info->uses_txf);
      CHECK(robust_info->agx3_staging_known);
      CHECK(robust_info->agx3_staging_halfs == 0);
      CHECK(robust_hash == robust_index[i].hash);
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
   CHECK(info->push_count == 26);
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
   CHECK(xfb_info->main_size == 162);
   CHECK(xfb_info->binary_size == 162);
   CHECK(xfb_info->nr_gprs == 12);
   CHECK(xfb_info->push_count == 16);
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
   CHECK(ia_info->main_offset == 128);
   CHECK(ia_info->main_size == 1252);
   CHECK(ia_info->binary_size == 1380);
   CHECK(ia_info->nr_gprs == 28);
   CHECK(ia_info->nr_preamble_gprs == 0);
   CHECK(ia_info->push_count == 44);
   CHECK(ia_info->workgroup_size[0] == 1);
   CHECK(ia_info->workgroup_size[1] == 1);
   CHECK(ia_info->workgroup_size[2] == 1);
   CHECK(ia_info->local_size == 0);
   CHECK(ia_info->imageblock_stride == 0);
   CHECK(!ia_info->uses_txf);
   CHECK(ia_info->agx3_staging_known);
   CHECK(ia_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(ia_binary, ia_info->binary_size) ==
         UINT64_C(0xa7601b6b0a9eacb4));
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
      /* The top-level loop has the native guard/entry/exit scopes. Native
       * descriptor scheduling assigns independent slots and uses per-slot
       * dependent wait carriers while keeping adjacent producers overlapped. */
      {LIBAGX_INCREMENT_IA_RESTART_NONNULL, 128, 2638, 2766, 44, 66,
       {1024, 1, 1}, 4, UINT64_C(0x135b087e7f3ae651)},
      {LIBAGX_COPY_QUERY, 0, 802, 802, 24, 37, {32, 1, 1}, 0,
       UINT64_C(0xb93e707e6841381f)},
      {LIBAGX_INCREMENT_CS_INVOCATIONS, 0, 82, 82, 8, 22, {1, 1, 1}, 0,
       UINT64_C(0xa0c268972f000612)},
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
      {LIBAGX_PREDICATE_INDIRECT_0, 558, UINT64_C(0x694da09230b03781)},
      {LIBAGX_PREDICATE_INDIRECT_1, 686, UINT64_C(0xc4c1e02f4a270e85)},
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
      CHECK(predicate_info->nr_gprs == 18);
      CHECK(predicate_info->nr_preamble_gprs == 0);
      CHECK(predicate_info->push_count == 26);
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
   CHECK(conditional_info->main_size == 180);
   CHECK(conditional_info->binary_size == 180);
   CHECK(conditional_info->nr_gprs == 6);
   CHECK(conditional_info->nr_preamble_gprs == 0);
   CHECK(conditional_info->push_count == 28);
   CHECK(conditional_info->workgroup_size[0] == 1);
   CHECK(conditional_info->workgroup_size[1] == 1);
   CHECK(conditional_info->workgroup_size[2] == 1);
   CHECK(conditional_info->local_size == 0);
   CHECK(conditional_info->imageblock_stride == 0);
   CHECK(!conditional_info->uses_txf);
   CHECK(conditional_info->agx3_staging_known);
   CHECK(conditional_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(conditional_binary, conditional_info->binary_size) ==
         UINT64_C(0xc0f57f69909964bb));
   CHECK(memcmp(conditional_binary + conditional_info->binary_size -
                   sizeof(end),
                end, sizeof(end)) == 0);

   /* A timestamp written inside a render pass after another timestamp copies
    * the graphics control stream's completion time. Fresh metal-nt
    * applegpu_g17p ground truth for the equivalent MSL kernel, dest[0] =
    * src[0] over two ulong device buffers, performs one 64-bit lddev and one
    * 64-bit stdev (native __compute text SHA-256
    * 682dde00553bdaf5abe66ee9c2d7bb9f1ce9e3cb50db1dbd79f0289094885441).
    * The Mesa ABI form addresses both pointers from its uniforms, leaving
    * exactly that load, store and end. */
   static const uint8_t timestamp_expected[] = {
      0x67, 0x00, 0x54, 0x00, 0x04, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00,
      0x80, 0x02, 0x00, 0xe7, 0x10, 0x54, 0x00, 0x03, 0x00, 0x00, 0x00,
      0x19, 0x00, 0x00, 0x90, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   const uint32_t *timestamp_blob = libagx_g17p[LIBAGX_COPY_TIMESTAMP];
   CHECK(timestamp_blob != NULL);
   if (!timestamp_blob)
      return 1;
   const struct agx_precompiled_kernel_info *timestamp_info =
      (const void *)timestamp_blob;
   CHECK(timestamp_info->preamble_offset == 0);
   CHECK(timestamp_info->main_offset == 0);
   CHECK(timestamp_info->main_size == sizeof(timestamp_expected));
   CHECK(timestamp_info->binary_size == sizeof(timestamp_expected));
   CHECK(timestamp_info->nr_gprs == 4);
   CHECK(timestamp_info->nr_preamble_gprs == 0);
   CHECK(timestamp_info->push_count == 20);
   CHECK(timestamp_info->workgroup_size[0] == 1);
   CHECK(timestamp_info->workgroup_size[1] == 1);
   CHECK(timestamp_info->workgroup_size[2] == 1);
   CHECK(timestamp_info->local_size == 0);
   CHECK(timestamp_info->imageblock_stride == 0);
   CHECK(!timestamp_info->uses_txf);
   CHECK(timestamp_info->agx3_staging_known);
   CHECK(timestamp_info->agx3_staging_halfs == 0);
   CHECK(memcmp((const uint8_t *)timestamp_blob + sizeof(*timestamp_info),
                timestamp_expected, sizeof(timestamp_expected)) == 0);

   /* Hardware tessellation factor records: one KERNEL(32) invocation per
    * patch reading the TCS levels as floats and as bits through the
    * tessellation parameters, and writing IEEE halves from the draw's first
    * patch index. Pin the whole generated program and its launch ABI. */
   const uint32_t *factors_blob = libagx_g17p[LIBAGX_TESS_FACTORS_HALF];
   CHECK(factors_blob != NULL);
   if (!factors_blob)
      return 1;
   const struct agx_precompiled_kernel_info *factors_info =
      (const void *)factors_blob;
   const uint8_t *factors_binary =
      (const uint8_t *)factors_blob + sizeof(*factors_info);
   CHECK(factors_info->preamble_offset == 0);
   CHECK(factors_info->main_offset == 128);
   CHECK(factors_info->main_size == 1282);
   CHECK(factors_info->binary_size == 1410);
   CHECK(factors_info->nr_gprs == 28);
   CHECK(factors_info->nr_preamble_gprs == 0);
   CHECK(factors_info->push_count == 30);
   CHECK(factors_info->workgroup_size[0] == 32);
   CHECK(factors_info->workgroup_size[1] == 1);
   CHECK(factors_info->workgroup_size[2] == 1);
   CHECK(factors_info->local_size == 0);
   CHECK(factors_info->imageblock_stride == 0);
   CHECK(!factors_info->uses_txf);
   CHECK(factors_info->agx3_staging_known);
   CHECK(factors_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(factors_binary, factors_info->binary_size) ==
         UINT64_C(0xc35a55da13cfe766));
   CHECK(memcmp(factors_binary + factors_info->binary_size - sizeof(end),
                end, sizeof(end)) == 0);

   /* Indirect hardware tessellation setup: one invocation that places the
    * TCS outputs, vertex outputs and factor records in the heap without the
    * AGX2 printf/abort overflow ABI, and writes the launch grids and the
    * patch draw. Pin the whole generated program and its launch ABI. */
   const uint32_t *setup_blob = libagx_g17p[LIBAGX_TESS_SETUP_INDIRECT_HW];
   CHECK(setup_blob != NULL);
   if (!setup_blob)
      return 1;
   const struct agx_precompiled_kernel_info *setup_info =
      (const void *)setup_blob;
   const uint8_t *setup_binary =
      (const uint8_t *)setup_blob + sizeof(*setup_info);
   CHECK(setup_info->preamble_offset == 0);
   CHECK(setup_info->main_offset == 128);
   CHECK(setup_info->main_size == 2138);
   CHECK(setup_info->binary_size == 2266);
   CHECK(setup_info->nr_gprs == 40);
   CHECK(setup_info->nr_preamble_gprs == 0);
   CHECK(setup_info->push_count == 58);
   CHECK(setup_info->workgroup_size[0] == 1);
   CHECK(setup_info->workgroup_size[1] == 1);
   CHECK(setup_info->workgroup_size[2] == 1);
   CHECK(setup_info->local_size == 0);
   CHECK(setup_info->imageblock_stride == 0);
   CHECK(!setup_info->uses_txf);
   CHECK(setup_info->agx3_staging_known);
   CHECK(setup_info->agx3_staging_halfs == 0);
   CHECK(fnv1a64(setup_binary, setup_info->binary_size) ==
         UINT64_C(0xaf4799f3590fa587));
   CHECK(memcmp(setup_binary + setup_info->binary_size - sizeof(end),
                end, sizeof(end)) == 0);

   if (failures) {
      fprintf(stderr, "g17p-precompiled-fill: %u/%u checks failed\n",
              failures, checks);
      return 1;
   }
   printf("g17p-precompiled-fill: %u checks passed\n", checks);
   return 0;
}
