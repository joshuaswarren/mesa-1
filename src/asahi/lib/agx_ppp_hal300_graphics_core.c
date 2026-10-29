/* SPDX-License-Identifier: MIT */
#include "agx_ppp_hal300_graphics_core.h"
#include "agx_vdm_state_core.h"
#include <string.h>

bool
agx_ppp_hal300_graphics_core_pack(
   enum agx_chip chip, const struct agx_ppp_hal300_graphics_fields *fields,
   struct agx_ppp_hal300_graphics_core *out)
{

   if (!agx_chip_g17_m5(chip) || !fields || !out ||
       !fields->fragment_program || fields->cull_mode > 3 ||
       fields->clip_distance_words > 8 || fields->viewport_count > 16 ||
       fields->fragment_pass > 3 || fields->flat_shading_vertex > 2)
      return false;

   const uint64_t ti_spill_quads =
      ((uint64_t)fields->fragment_ti_spill_B + 3) >> 2;
   const uint64_t ti_staging_quads =
      ((((uint64_t)fields->fragment_ti_staging_B + 3) >> 2) + 1) &
      ~UINT64_C(1);
   if (ti_spill_quads > 0x3ffff || ti_staging_quads > 0xffe ||
       ((ti_spill_quads || ti_staging_quads) && !agx_chip_g17_m5(chip)))
      return false;

   if (fields->general_varyings) {
      /* HAL300's native VSStateConfig and VIT producer retain the shared
       * all-f32 UVS/CF ABI: 128 total UVS words and 127 coefficients. */
      const unsigned smooth = fields->varying_counts_32 & 0xff;
      const unsigned flat = (fields->varying_counts_32 >> 8) & 0xff;
      const unsigned linear = (fields->varying_counts_32 >> 16) & 0xff;
      const uint32_t expected_output_select =
         (fields->point_size_output ? 0x40000 : 0) |
         (fields->layer_output ? 0x180000 : 0) |
         ((fields->vertex_varying_components || fields->cf_binding_count)
             ? 0x10000 : 0) |
         (fields->fragment_reads_depth ? 0x200000 : 0) |
         ((UINT32_C(1) << fields->clip_distance_words) - 1);
      if (fields->vertex_varying_components > 124 ||
          fields->fragment_varying_components >
             fields->vertex_varying_components ||
          fields->varying_counts_16 || (fields->varying_counts_32 >> 24) ||
          smooth + flat + linear != fields->vertex_varying_components ||
          fields->uvs_words > 128 ||
          fields->uvs_words != 4 + fields->vertex_varying_components +
             fields->point_size_output + fields->layer_output +
             fields->clip_distance_words ||
          fields->output_select != expected_output_select ||
          !fields->cf_binding_count || fields->cf_binding_count > 127 ||
          fields->cf_register_count < fields->cf_binding_count ||
          fields->cf_register_count > 127 ||
          !fields->varying_table || (fields->varying_table & 63) ||
          fields->varying_table >= (UINT64_C(1) << 48))
         return false;
   } else {
      if (fields->vertex_varying_components >
             ((fields->rasterizer_discard ||
               !fields->fragment_varying_components) ? 124 : 8) ||
          4 + fields->point_size_output + fields->layer_output +
             fields->clip_distance_words + fields->vertex_varying_components > 128 ||
          fields->fragment_varying_components >
             fields->vertex_varying_components || fields->needs_primitive_id)
         return false;

      if (fields->fragment_varying_components || fields->fragment_reads_depth ||
          fields->fragment_reads_reciprocal_w) {
         if (!fields->varying_table ||
             (fields->varying_table & 63) ||
             fields->varying_table >= (UINT64_C(1) << 48) ||
             !fields->cf_binding_count || fields->cf_binding_count > 3 ||
             (fields->interpolation != AGX_PPP_HAL300_SMOOTH &&
              fields->cf_binding_count > 2 &&
              !fields->fragment_reads_reciprocal_w))
            return false;
      } else if (fields->varying_table || fields->cf_binding_count ||
                 fields->interpolation != AGX_PPP_HAL300_SMOOTH) {
         return false;
      }
      if (fields->interpolation > AGX_PPP_HAL300_LINEAR)
         return false;

      if (fields->no_color_attachment &&
          fields->interpolation == AGX_PPP_HAL300_FLAT &&
          fields->fragment_varying_components)
         return false;
   }
   const bool psiz = fields->general_varyings
                        ? (fields->output_select & 0x40000) != 0
                        : fields->point_size_output;

   if (fields->object_type > AGX_PPP_HAL300_LINES ||
       (fields->object_type == AGX_PPP_HAL300_POINTS && !psiz &&
        !fields->rasterizer_discard))
      return false;

   if ((fields->visibility_mode != 0 && fields->visibility_mode != 2 &&
        fields->visibility_mode != 3) ||
       fields->visibility_index >= (1u << 15))
      return false;

   if (fields->line_width > 0xff || fields->polygon_mode > 2 ||
       fields->fragment_depth_mode > 3 || fields->depth_compare > 7 ||
       (!fields->depth_active && (fields->depth_compare || fields->depth_write)))
      return false;

   const uint8_t compare = fields->depth_active ? fields->depth_compare : 7;
   const bool write = fields->depth_active && fields->depth_write;
   const bool depth_inactive = !fields->depth_active;
   const bool points = fields->object_type == AGX_PPP_HAL300_POINTS;
   const bool lines = fields->object_type == AGX_PPP_HAL300_LINES;
   const uint32_t control = 0x00000200 | (points || lines ? 0x04000000 : 0) |
                            (fields->scissor_enable ? 0x00010000 : 0) |
                            (fields->depth_bias_enable ? 0x00020000 : 0) |
                            ((depth_inactive || fields->stencil_active) ? 0x00040000 : 0) |
                            (fields->stencil_active ? 0x00080000 : 0) |
                            /* The native visibility-result-mode setter
                             * sets bits 15:14 of this word (0, 2 counting,
                             * 3 boolean) and offset / 8 into bits 31:17 of
                             * the word 0x14 bytes after it. */
                            ((uint32_t)fields->visibility_mode << 14);
   const uint32_t visibility_query =
      fields->visibility_mode ? (uint32_t)fields->visibility_index << 17 : 0;
   const uint32_t face = ((uint32_t)compare << 24) |
                         (write ? 0 : 0x00200000) | (fields->line_width << 8) |
                         (fields->polygon_mode << 18);

   /* HAL300 setFragmentVariant(nullptr) disables tag writes, clears its
    * fragment output mask and selects depth mode 3. Rasterizer discard
    * selects that inert fragment state while the VDM still runs the VS. */
   const uint32_t early_test =
      0x07200000 |
      ((fields->rasterizer_discard ? 3 : fields->fragment_depth_mode) << 22) |
      (points ? 0x40000000 : 0) | (lines ? 0x10000000 : 0);

   struct agx_vdm_state_vertex_pipeline_pair program;

   const uint32_t program_template = agx_chip_g17_m5(chip)
      ? (uint32_t)(ti_spill_quads << 3)
      : fields->fragment_writes_depth ? 0x40 : 0;
   if (!agx_vdm_state_vertex_pipeline_pair_pack(
          chip, fields->fragment_program, program_template, &program))
      return false;

   const uint32_t counts_32 =
      fields->general_varyings
         ? fields->varying_counts_32
         : fields->fragment_varying_components <<
              (fields->no_color_attachment
                  ? 16
                  : fields->interpolation == AGX_PPP_HAL300_FLAT ? 8 : 0);
   const uint32_t output_select =
      fields->general_varyings
         ? fields->output_select
         : (fields->point_size_output ? 0x40000 : 0) |
              (fields->layer_output ? 0x180000 : 0) |
              /* System Z/1-W bindings also require coefficient production.
               * Keep the shared linker's OUTPUT_SELECT.varyings contract. */
              ((fields->vertex_varying_components || fields->cf_binding_count)
                  ? 0x10000 : 0) |
              (fields->fragment_reads_depth ? 0x200000 : 0) |
              ((UINT32_C(1) << fields->clip_distance_words) - 1);
   const uint32_t uvs_words =
      fields->general_varyings
         ? fields->uvs_words
         : 4 + (fields->point_size_output ? 1 : 0) +
              (fields->layer_output ? 1 : 0) +
              fields->clip_distance_words + fields->vertex_varying_components;
   const uint32_t records[][7] = {
      {0x10100000, counts_32,
       fields->general_varyings ? fields->varying_counts_16 : 0, 0, 0},
      {0x01000000, (uint32_t)(fields->varying_table >> 32) |
                      (fields->cf_binding_count << 16),
       (uint32_t)fields->varying_table, (uint32_t)ti_staging_quads, 0, 0, 0},
      {0x00000001, 0x000007aa, 0xe2e00000, 0xf0c00000,
       0x00023020, 0x0001ffff},
      {0x0c280000, output_select, 0,
       fields->viewport_count ? fields->viewport_count - 1 : 0, uvs_words},
      {0x020000e0, control,
       face | fields->front_stencil_reference,
       face | fields->back_stencil_reference,
       fields->stencil_active ? fields->front_stencil : 0x0e000000,
       fields->stencil_active ? fields->back_stencil : 0x0e000000,
       visibility_query},

      /* Bit 21 is the tag-write disable that setFragmentVariant(nullptr)
       * selects; a linked fragment that writes no tile needs it as well. */
      {0x02000062, 0x00000200 |
                      (fields->rasterizer_discard ? 0x00200000
                                                : fields->fragment_pass << 29) |
                      (fields->tag_write_disable ? 0x00200000 : 0) |
                      (fields->disable_tri_merging || fields->polygon_mode || points || lines
                          ? 0x04000000
                          : 0),
       early_test, early_test,
       (fields->no_color_attachment || fields->no_color_writes ||
        fields->rasterizer_discard)
          ? 0 : 0x0001ffff},

      {0x00840002, 0,
       (fields->rasterizer_discard ? 0x00020080 : 0x80) |
          (fields->needs_primitive_id ? 0x1000 : 0)},

      {0x00800000, (fields->depth_clip_disable ? 0x800 : 0x400) |
                      ((fields->flat_shading_vertex + 1) << 7) |
                      fields->cull_mode |
                      ((uint32_t)fields->front_face_ccw << 16) |
                      (fields->primitive_msaa ? 0x8000 : 0)},
      {0x00000400, 0, 0},
   };
   const uint8_t lengths[] = {20, 28, 24, 20, 28, 20, 12, 8, 12};
   struct agx_ppp_hal300_graphics_core staged = {0};
   unsigned offset = 0;
   for (unsigned i = 0; i < AGX_PPP_HAL300_GRAPHICS_RECORDS; ++i) {
      staged.offsets[i] = offset;
      staged.lengths[i] = lengths[i];
      for (unsigned j = 0; j < lengths[i]; ++j)
         staged.bytes[offset + j] = records[i][j / 4] >> (8 * (j % 4));
      offset += lengths[i];
   }

   memcpy(staged.bytes + staged.offsets[1] + 16, program.bytes, 8);
   *out = staged;
   return true;
}
