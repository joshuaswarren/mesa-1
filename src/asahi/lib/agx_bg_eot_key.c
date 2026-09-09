/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_bg_eot_key.h"

void
agx_g17p_configure_rt0_eot_key(enum pipe_format format,
                               struct agx_shader_key *key)
{
   if (!key)
      return;

   key->g17p_rt0_eot_words = 0;
   key->g17p_rt0_eot_rgb565 = false;
   key->g17p_rt0_eot_norm16_selector_95 = false;
   key->g17p_rt0_eot_i16_signed_two_words = false;
   key->g17p_rt0_eot_i32 = false;
   key->g17p_rt0_eot_i32_four_words = false;
   key->g17p_rt0_eot_i32_signed_four_words = false;
   key->g17p_rt0_eot_i32_selector_15 = false;
   key->g17p_rt0_eot_i32_selector_75 = false;

   key->g17p_rt0_eot_rgb565 = format == PIPE_FORMAT_R5G6B5_UNORM ||
                              format == PIPE_FORMAT_B5G6R5_UNORM;
   key->g17p_rt0_eot_i32 =
      format == PIPE_FORMAT_R32_FLOAT ||
      format == PIPE_FORMAT_R32G32_FLOAT || format == PIPE_FORMAT_R32_UINT ||
      format == PIPE_FORMAT_R32_SINT || format == PIPE_FORMAT_R32G32_SINT ||
      format == PIPE_FORMAT_R32G32_UINT ||
      format == PIPE_FORMAT_R32G32B32A32_FLOAT ||
      format == PIPE_FORMAT_R32G32B32A32_UINT ||
      format == PIPE_FORMAT_R32G32B32A32_SINT;
   key->g17p_rt0_eot_i32_four_words =
      format == PIPE_FORMAT_R32G32B32A32_FLOAT ||
      format == PIPE_FORMAT_R32G32B32A32_UINT;
   key->g17p_rt0_eot_i32_signed_four_words =
      format == PIPE_FORMAT_R32G32B32A32_SINT;
   key->g17p_rt0_eot_i32_selector_15 =
      format == PIPE_FORMAT_R32G32_SINT;
   key->g17p_rt0_eot_i32_selector_75 =
      format == PIPE_FORMAT_R32G32_UINT;
   key->g17p_rt0_eot_norm16_selector_95 =
      format == PIPE_FORMAT_R16_UNORM || format == PIPE_FORMAT_R16_SNORM ||
      format == PIPE_FORMAT_R16G16_UNORM ||
      format == PIPE_FORMAT_R16G16_SNORM ||
      format == PIPE_FORMAT_R16G16B16A16_SNORM;
   key->g17p_rt0_eot_i16_signed_two_words =
      format == PIPE_FORMAT_R16G16B16A16_SINT;

   switch (format) {
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8_SINT:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8G8_SINT:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R10G10B10A2_UINT:
      key->g17p_rt0_eot_words = 1;
      break;
   case PIPE_FORMAT_R5G6B5_UNORM:
   case PIPE_FORMAT_B5G6R5_UNORM:
   /* The packed 5551/4444 formats expand to four F16 tile lanes. The PBE
    * performs the final memory packing, so their EOT shape is two words just
    * like RGBA16F rather than the distinct three-lane RGB565 program. */
   case PIPE_FORMAT_R5G5B5A1_UNORM:
   case PIPE_FORMAT_B5G5R5A1_UNORM:
   case PIPE_FORMAT_R5G5B5X1_UNORM:
   case PIPE_FORMAT_B5G5R5X1_UNORM:
   case PIPE_FORMAT_R4G4B4A4_UNORM:
   case PIPE_FORMAT_B4G4R4A4_UNORM:
   case PIPE_FORMAT_A4B4G4R4_UNORM:
   case PIPE_FORMAT_A4R4G4B4_UNORM:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32_UINT:
      key->g17p_rt0_eot_words = 2;
      break;
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      key->g17p_rt0_eot_words = 4;
      break;
   default:
      break;
   }
}
