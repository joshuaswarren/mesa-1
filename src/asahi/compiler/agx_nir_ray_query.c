/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir_ray_query.h"

#include <string.h>

#include "asahi/lib/agx_rt_result.h"

static bool
committed_result_layout(nir_ray_query_value value, uint32_t column,
                        struct agx_nir_ray_query_result_layout *out)
{
   if (column != 0)
      return false;

   switch (value) {
   case nir_ray_query_value_intersection_type:
      *out = (struct agx_nir_ray_query_result_layout){
         .byte_offset = AGX_RT_QUERY_OFF_COMMITTED_TYPE,
         .components = 1,
         .storage_bit_size = 8,
         .kind = AGX_NIR_RAY_QUERY_RESULT_U32,
      };
      return true;
   case nir_ray_query_value_intersection_t:
      *out = (struct agx_nir_ray_query_result_layout){
         .byte_offset = AGX_RT_QUERY_OFF_COMMITTED_T,
         .components = 1,
         .storage_bit_size = 32,
         .kind = AGX_NIR_RAY_QUERY_RESULT_F32,
      };
      return true;
   case nir_ray_query_value_intersection_geometry_index:
      *out = (struct agx_nir_ray_query_result_layout){
         .byte_offset = AGX_RT_QUERY_OFF_COMMITTED_GEOMETRY_ID,
         .components = 1,
         .storage_bit_size = 32,
         .kind = AGX_NIR_RAY_QUERY_RESULT_U32,
      };
      return true;
   case nir_ray_query_value_intersection_primitive_index:
      *out = (struct agx_nir_ray_query_result_layout){
         .byte_offset = AGX_RT_QUERY_OFF_COMMITTED_PRIMITIVE_ID,
         .components = 1,
         .storage_bit_size = 32,
         .kind = AGX_NIR_RAY_QUERY_RESULT_U32,
      };
      return true;
   case nir_ray_query_value_intersection_barycentrics:
      *out = (struct agx_nir_ray_query_result_layout){
         .byte_offset = AGX_RT_QUERY_OFF_COMMITTED_BARYCENTRICS,
         .components = 2,
         .storage_bit_size = 32,
         .kind = AGX_NIR_RAY_QUERY_RESULT_F32,
      };
      return true;
   case nir_ray_query_value_intersection_front_face:
      *out = (struct agx_nir_ray_query_result_layout){
         .byte_offset = AGX_RT_QUERY_OFF_COMMITTED_FRONT_FACE,
         .components = 1,
         .storage_bit_size = 8,
         .kind = AGX_NIR_RAY_QUERY_RESULT_BOOL,
      };
      return true;
   default:
      /* Everything else -- including committed instance id and user instance
       * id, whose accessors resolve through the candidate record rather than
       * a single committed field -- has no exact ray-query offset. */
      return false;
   }
}

static bool
committed_prerequisites_for_value(
   enum agx_chip chip, nir_ray_query_value value, uint32_t column,
   struct agx_nir_ray_query_committed_prerequisites *out)
{
   if (out == NULL)
      return false;

   memset(out, 0, sizeof(*out));

   struct agx_nir_ray_query_result_layout result = {0};
   if (!committed_result_layout(value, column, &result))
      return false;

   const enum agx_isa_encoding intersect_encoding =
      agx3_intersect_ray_witnessed_chip_encoding(chip);
   const enum agx_isa_encoding continue_encoding =
      agx3_continue_ray_witnessed_chip_encoding(chip);
   const enum agx_isa_encoding inc_stk_encoding =
      agx3_inc_stk_witnessed_chip_encoding(chip);

   if (intersect_encoding == AGX_ISA_ENCODING_UNKNOWN ||
       intersect_encoding != continue_encoding ||
       intersect_encoding != inc_stk_encoding)
      return false;

   struct agx3_encoded_intersect_ray intersect;
   struct agx3_encoded_continue_ray_fixed_first_site_witness continue_first;
   struct agx3_encoded_inc_stk_fixed_witness inc_stk;

   if (!agx3_pack_intersect_ray(intersect_encoding, 0, &intersect) ||
       !agx3_pack_continue_ray_fixed_first_site_witness(continue_encoding,
                                                        &continue_first) ||
       !agx3_pack_inc_stk_fixed_witness(inc_stk_encoding, &inc_stk))
      return false;

   *out = (struct agx_nir_ray_query_committed_prerequisites){
      .encoding = intersect_encoding,
      .query_state_size = AGX_RT_RAY_QUERY_STATE_SIZE,
      .intersect_ray_witness_size = intersect.length,
      .continue_ray_first_site_witness_size = continue_first.length,
      .inc_stk_witness_size = inc_stk.length,
      .result = result,
   };
   return true;
}

bool
agx_nir_ray_query_committed_load_prerequisites(
   enum agx_chip chip, const nir_intrinsic_instr *load,
   struct agx_nir_ray_query_committed_prerequisites *out)
{
   if (out == NULL)
      return false;

   memset(out, 0, sizeof(*out));

   if (load == NULL || load->intrinsic != nir_intrinsic_rq_load ||
       !nir_intrinsic_committed(load))
      return false;

   return committed_prerequisites_for_value(chip,
                                            nir_intrinsic_ray_query_value(load),
                                            nir_intrinsic_column(load), out);
}
