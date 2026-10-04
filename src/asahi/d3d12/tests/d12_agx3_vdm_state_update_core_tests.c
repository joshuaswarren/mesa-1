/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "../d12_agx3_vdm_state_update_core.h"

#include "asahi/lib/agx_vdm_state_update_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      checks++;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static const enum agx_chip supported_chips[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
};

static const enum agx_chip unsupported_chips[] = {
   AGX_CHIP_G13G, AGX_CHIP_G13X, AGX_CHIP_G14G, AGX_CHIP_G14X,
   AGX_CHIP_G15P, AGX_CHIP_G16P, AGX_CHIP_G17A, AGX_CHIP_UNKNOWN,
   (enum agx_chip)999,
};

static const struct d12_agx3_vdm_state_update_info witness_info = {
   .raw_vertex_shader_word0 = UINT32_C(0x11223344),
   .pipeline = UINT64_C(0x123456789a80),
   .raw_pipeline_template = UINT32_C(0x2abcde),
   .raw_vertex_outputs = {UINT32_C(0x55667788), UINT32_C(0x99aabbcc)},
   .raw_vertex_unknown = UINT32_C(0xddeeff00),
};

static const uint8_t witness[D12_AGX3_VDM_STATE_UPDATE_LENGTH] = {
   0x2e, 0x00, 0x00, 0x40, 0x44, 0x33, 0x22, 0x11,
   0xde, 0xbc, 0x2a, 0x12, 0x6a, 0xe2, 0x59, 0xd1,
   0x88, 0x77, 0x66, 0x55, 0xcc, 0xbb, 0xaa, 0x99,
   0x00, 0xff, 0xee, 0xdd,
};

static const struct d12_agx3_vdm_state_update_template_info
   template_witness_info = {
      .pipeline = UINT64_C(0x123456789a80),
      .raw_state_load_count_12 = UINT16_C(0x0abc),
      .raw_state_load_count_18 = UINT32_C(0x00023456),
      .raw_vs_output_count_0 = UINT8_C(0x12),
      .raw_vs_output_count_2 = UINT8_C(0x34),
      .raw_vs_output_count_1_extra = true,
      .raw_vs_output_selector = UINT8_C(0x03),
      .raw_vs_output_flag_6 = true,
      .pre_hal300_flag_group_nonzero = true,
      .pre_hal300_flag_bit_4 = true,
      .variant = D12_AGX3_VDM_STATE_UPDATE_TEMPLATE_00,
      .raw_active_context_selector = 5,
   };

static const uint8_t pre_hal300_template_witness
   [D12_AGX3_VDM_STATE_UPDATE_LENGTH] = {
      0x2e, 0x00, 0x00, 0x40, 0x00, 0xbc, 0x0a, 0x80,
      0xb0, 0xa2, 0x11, 0x12, 0x6a, 0xe2, 0x59, 0xd1,
      0x12, 0x47, 0x34, 0x00, 0x7c, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00,
   };

static const uint8_t hal300_template_witness
   [D12_AGX3_VDM_STATE_UPDATE_LENGTH] = {
      0x2e, 0x00, 0x00, 0x40, 0x00, 0xbc, 0x0a, 0x00,
      0xb0, 0xa2, 0x11, 0x12, 0x6a, 0xe2, 0x59, 0xd1,
      0x12, 0x47, 0x34, 0x00, 0x6c, 0x00, 0x00, 0x00,
      0x04, 0x00, 0x00, 0x00,
   };

static int
check_info_equal(const struct d12_agx3_vdm_state_update_info *actual,
                 const struct d12_agx3_vdm_state_update_info *expected)
{
   CHECK(actual->raw_vertex_shader_word0 ==
         expected->raw_vertex_shader_word0);
   CHECK(actual->pipeline == expected->pipeline);
   CHECK(actual->raw_pipeline_template == expected->raw_pipeline_template);
   CHECK(actual->raw_vertex_outputs[0] == expected->raw_vertex_outputs[0]);
   CHECK(actual->raw_vertex_outputs[1] == expected->raw_vertex_outputs[1]);
   CHECK(actual->raw_vertex_unknown == expected->raw_vertex_unknown);
   return 0;
}

static int
check_zero_info(const struct d12_agx3_vdm_state_update_info *info)
{
   const struct d12_agx3_vdm_state_update_info zero = {0};
   return check_info_equal(info, &zero);
}

static int
check_zero_encoding(const struct d12_agx3_vdm_state_update *encoded)
{
   for (unsigned i = 0; i < sizeof(encoded->bytes); ++i)
      CHECK(encoded->bytes[i] == 0);
   CHECK(encoded->length == 0);
   return 0;
}

static int
check_targets_and_shared_equality(void)
{
   const struct agx_vdm_state_update_info shared_info = {
      .raw_vertex_shader_word0 = witness_info.raw_vertex_shader_word0,
      .pipeline = witness_info.pipeline,
      .raw_pipeline_template = witness_info.raw_pipeline_template,
      .raw_vertex_outputs = {witness_info.raw_vertex_outputs[0],
                             witness_info.raw_vertex_outputs[1]},
      .raw_vertex_unknown = witness_info.raw_vertex_unknown,
   };

   for (unsigned i = 0; i < sizeof(supported_chips) / sizeof(supported_chips[0]);
        ++i) {
      struct d12_agx3_vdm_state_update encoded;
      struct d12_agx3_vdm_state_update_info decoded;
      struct agx_vdm_state_update shared;

      CHECK(d12_agx3_encode_vdm_state_update(supported_chips[i], &witness_info,
                                             &encoded));
      CHECK(agx_vdm_state_update_pack(supported_chips[i], &shared_info,
                                      &shared));
      CHECK(encoded.length == sizeof(witness));
      CHECK(encoded.length == shared.length);
      CHECK(memcmp(encoded.bytes, witness, sizeof(witness)) == 0);
      CHECK(memcmp(encoded.bytes, shared.bytes, sizeof(witness)) == 0);
      CHECK(d12_agx3_decode_vdm_state_update(
         supported_chips[i], encoded.bytes, encoded.length, &decoded));
      if (check_info_equal(&decoded, &witness_info))
         return 1;
   }

   return 0;
}

static int
check_domains_and_corruptions(void)
{
   for (unsigned bit = 6; bit < 48; ++bit) {
      struct d12_agx3_vdm_state_update_info info = {0};
      struct d12_agx3_vdm_state_update_info decoded;
      struct d12_agx3_vdm_state_update encoded;
      info.pipeline = UINT64_C(1) << bit;
      CHECK(d12_agx3_encode_vdm_state_update(AGX_CHIP_G17S, &info,
                                             &encoded));
      CHECK(d12_agx3_decode_vdm_state_update(
         AGX_CHIP_G17S, encoded.bytes, encoded.length, &decoded));
      if (check_info_equal(&decoded, &info))
         return 1;
   }

   for (unsigned bit = 0; bit < 22; ++bit) {
      struct d12_agx3_vdm_state_update_info info = {0};
      struct d12_agx3_vdm_state_update_info decoded;
      struct d12_agx3_vdm_state_update encoded;
      info.raw_pipeline_template = UINT32_C(1) << bit;
      CHECK(d12_agx3_encode_vdm_state_update(AGX_CHIP_G15G, &info,
                                             &encoded));
      CHECK(d12_agx3_decode_vdm_state_update(
         AGX_CHIP_G15G, encoded.bytes, encoded.length, &decoded));
      if (check_info_equal(&decoded, &info))
         return 1;
   }

   for (unsigned bit = 0; bit < sizeof(witness) * 8u; ++bit) {
      uint8_t changed[D12_AGX3_VDM_STATE_UPDATE_LENGTH];
      struct d12_agx3_vdm_state_update_info decoded;
      struct d12_agx3_vdm_state_update repacked;
      memcpy(changed, witness, sizeof(changed));
      changed[bit / 8] ^= (uint8_t)(1u << (bit % 8));
      memset(&decoded, 0xa5, sizeof(decoded));
      const bool accepted = d12_agx3_decode_vdm_state_update(
         AGX_CHIP_G17S, changed, sizeof(changed), &decoded);
      if (bit < 32) {
         CHECK(!accepted);
         if (check_zero_info(&decoded))
            return 1;
      } else {
         CHECK(accepted);
         CHECK(d12_agx3_encode_vdm_state_update(AGX_CHIP_G17S, &decoded,
                                                &repacked));
         CHECK(memcmp(changed, repacked.bytes, sizeof(changed)) == 0);
      }
   }

   return 0;
}

static int
check_template_bridge(void)
{
   for (unsigned i = 0; i < sizeof(supported_chips) / sizeof(supported_chips[0]);
        ++i) {
      struct d12_agx3_vdm_state_update_template_info info =
         template_witness_info;
      const bool hal300 = supported_chips[i] == AGX_CHIP_G17G ||
                          supported_chips[i] == AGX_CHIP_G17S ||
                          supported_chips[i] == AGX_CHIP_G17C;
      if (hal300) {
         info.pre_hal300_flag_group_nonzero = false;
         info.pre_hal300_flag_bit_4 = false;
      }

      const struct agx_vdm_state_update_template_info shared_info = {
         .pipeline = info.pipeline,
         .raw_state_load_count_12 = info.raw_state_load_count_12,
         .raw_state_load_count_18 = info.raw_state_load_count_18,
         .raw_vs_output_count_0 = info.raw_vs_output_count_0,
         .raw_vs_output_count_2 = info.raw_vs_output_count_2,
         .raw_vs_output_count_1_extra = info.raw_vs_output_count_1_extra,
         .raw_vs_output_selector = info.raw_vs_output_selector,
         .raw_vs_output_flag_6 = info.raw_vs_output_flag_6,
         .pre_hal300_flag_group_nonzero =
            info.pre_hal300_flag_group_nonzero,
         .pre_hal300_flag_bit_4 = info.pre_hal300_flag_bit_4,
         .variant = (enum agx_vdm_state_update_template_variant)info.variant,
         .raw_active_context_selector = info.raw_active_context_selector,
      };
      struct d12_agx3_vdm_state_update encoded;
      struct agx_vdm_state_update shared;
      CHECK(d12_agx3_encode_vdm_state_update_template(supported_chips[i],
                                                       &info, &encoded));
      CHECK(agx_vdm_state_update_template_pack(supported_chips[i],
                                                &shared_info, &shared));
      CHECK(encoded.length == D12_AGX3_VDM_STATE_UPDATE_LENGTH);
      CHECK(encoded.length == shared.length);
      CHECK(memcmp(encoded.bytes, shared.bytes, sizeof(encoded.bytes)) == 0);
      CHECK(memcmp(encoded.bytes,
                   hal300 ? hal300_template_witness
                          : pre_hal300_template_witness,
                   sizeof(encoded.bytes)) == 0);
   }

   {
      struct d12_agx3_vdm_state_update_template_info info =
         template_witness_info;
      struct d12_agx3_vdm_state_update encoded;
      static const uint32_t raw_context_selectors[] = {
         UINT32_MAX, 0, 1, 8, 9, UINT32_C(0x12345678), UINT32_MAX - 1,
      };
      for (unsigned i = 0;
           i < sizeof(raw_context_selectors) /
                  sizeof(raw_context_selectors[0]);
           ++i) {
         info.raw_active_context_selector = raw_context_selectors[i];
         CHECK(d12_agx3_encode_vdm_state_update_template(AGX_CHIP_G15G,
                                                          &info, &encoded));
         const uint8_t expected =
            raw_context_selectors[i] == UINT32_MAX
               ? 0
               : (raw_context_selectors[i] - 1) & UINT32_C(0x7);
         CHECK(encoded.bytes[24] == expected && encoded.bytes[25] == 0 &&
               encoded.bytes[26] == 0 && encoded.bytes[27] == 0);
      }
      info = template_witness_info;
      info.variant = (enum d12_agx3_vdm_state_update_template_variant)6;
      CHECK(!d12_agx3_encode_vdm_state_update_template(AGX_CHIP_G15G, &info,
                                                        &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      CHECK(!d12_agx3_encode_vdm_state_update_template(AGX_CHIP_G15G, NULL,
                                                        &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      CHECK(!d12_agx3_encode_vdm_state_update_template(
         AGX_CHIP_G15G, &template_witness_info, NULL));
   }

   return 0;
}

static int
check_rejections(void)
{
   for (unsigned i = 0;
        i < sizeof(unsupported_chips) / sizeof(unsupported_chips[0]); ++i) {
      struct d12_agx3_vdm_state_update encoded;
      struct d12_agx3_vdm_state_update_info decoded;
      memset(&encoded, 0xa5, sizeof(encoded));
      CHECK(!d12_agx3_encode_vdm_state_update(
         unsupported_chips[i], &witness_info, &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      CHECK(!d12_agx3_decode_vdm_state_update(
         unsupported_chips[i], witness, sizeof(witness), &decoded));
      if (check_zero_info(&decoded))
         return 1;
   }

   static const uint64_t invalid_pipelines[] = {
      1, 63, UINT64_C(0x123456789a81), UINT64_C(1) << 48, UINT64_MAX,
   };
   for (unsigned i = 0;
        i < sizeof(invalid_pipelines) / sizeof(invalid_pipelines[0]); ++i) {
      struct d12_agx3_vdm_state_update_info info = witness_info;
      struct d12_agx3_vdm_state_update encoded;
      info.pipeline = invalid_pipelines[i];
      CHECK(!d12_agx3_encode_vdm_state_update(AGX_CHIP_G17S, &info,
                                              &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
   }

   {
      struct d12_agx3_vdm_state_update encoded;
      struct d12_agx3_vdm_state_update_info info = witness_info;
      info.raw_pipeline_template = UINT32_C(1) << 22;
      CHECK(!d12_agx3_encode_vdm_state_update(AGX_CHIP_G15G, &info,
                                              &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      CHECK(!d12_agx3_encode_vdm_state_update(AGX_CHIP_G15G, NULL,
                                              &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      CHECK(!d12_agx3_encode_vdm_state_update(AGX_CHIP_G15G, &witness_info,
                                              NULL));
   }

   for (uint32_t size = 0; size < sizeof(witness); ++size) {
      struct d12_agx3_vdm_state_update_info decoded;
      CHECK(!d12_agx3_decode_vdm_state_update(AGX_CHIP_G17S, witness, size,
                                              &decoded));
      if (check_zero_info(&decoded))
         return 1;
   }

   {
      struct d12_agx3_vdm_state_update_info decoded;
      uint8_t unaligned[D12_AGX3_VDM_STATE_UPDATE_LENGTH + 1];
      memcpy(&unaligned[1], witness, sizeof(witness));
      CHECK(d12_agx3_decode_vdm_state_update(
         AGX_CHIP_G17S, &unaligned[1], sizeof(witness), &decoded));
      if (check_info_equal(&decoded, &witness_info))
         return 1;
      CHECK(!d12_agx3_decode_vdm_state_update(
         AGX_CHIP_G17S, witness, sizeof(witness) + 1, &decoded));
      if (check_zero_info(&decoded))
         return 1;
      CHECK(!d12_agx3_decode_vdm_state_update(
         AGX_CHIP_G17S, NULL, sizeof(witness), &decoded));
      if (check_zero_info(&decoded))
         return 1;
      CHECK(!d12_agx3_decode_vdm_state_update(
         AGX_CHIP_G17S, witness, sizeof(witness), NULL));
   }

   return 0;
}

static int
check_native_draw_template_bridge(void)
{
   CHECK(d12_agx3_vdm_state_update_classify_draw(false, false) ==
         D12_AGX3_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED);
   CHECK(d12_agx3_vdm_state_update_classify_draw(false, true) ==
         D12_AGX3_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED);
   CHECK(d12_agx3_vdm_state_update_classify_draw(true, false) ==
         D12_AGX3_VDM_STATE_UPDATE_DRAW_INDIRECT_NONINDEXED);
   CHECK(d12_agx3_vdm_state_update_classify_draw(true, true) ==
         D12_AGX3_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED);

   for (unsigned target = 0;
        target < sizeof(supported_chips) / sizeof(supported_chips[0]);
        ++target) {
      const enum agx_chip chip = supported_chips[target];
      const bool hal300 = chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
                          chip == AGX_CHIP_G17C;

      for (unsigned selector = 0; selector < 6; ++selector) {
         struct d12_agx3_vdm_state_update first_path;
         for (unsigned draw = 0; draw < 4; ++draw) {
            struct d12_agx3_vdm_state_update_draw_info info = {
               .source_layout =
                  D12_AGX3_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
               .draw_path =
                  (enum d12_agx3_vdm_state_update_draw_path)draw,
               .rasterization_enabled = (selector & 1) != 0,
               .raw_provoking_selector = selector >> 1,
               .template_info = template_witness_info,
            };
            struct d12_agx3_vdm_state_update actual;
            struct d12_agx3_vdm_state_update expected;

            if (hal300) {
               info.template_info.pre_hal300_flag_group_nonzero = false;
               info.template_info.pre_hal300_flag_bit_4 = false;
            }
            info.template_info.variant =
               (enum d12_agx3_vdm_state_update_template_variant)6;

            struct d12_agx3_vdm_state_update_template_info selected =
               info.template_info;
            selected.variant =
               (enum d12_agx3_vdm_state_update_template_variant)selector;
            CHECK(d12_agx3_encode_vdm_state_update_template(chip, &selected,
                                                             &expected));
            CHECK(d12_agx3_encode_vdm_state_update_draw(chip, &info,
                                                         &actual));
            CHECK(actual.length == expected.length);
            CHECK(memcmp(actual.bytes, expected.bytes, sizeof(actual.bytes)) ==
                  0);

            if (draw == 0)
               first_path = actual;
            else
               CHECK(memcmp(actual.bytes, first_path.bytes,
                            sizeof(actual.bytes)) == 0);
         }
      }
   }

   {
      struct d12_agx3_vdm_state_update_draw_info info = {
         .source_layout =
            D12_AGX3_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
         .draw_path = D12_AGX3_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED,
         .template_info = template_witness_info,
      };
      struct d12_agx3_vdm_state_update encoded;
      static const enum d12_agx3_vdm_state_update_source_layout
         rejected_layouts[] = {
            D12_AGX3_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED,
            D12_AGX3_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX2_LEGACY,
            (enum d12_agx3_vdm_state_update_source_layout)-1,
            (enum d12_agx3_vdm_state_update_source_layout)3,
         };

      for (unsigned i = 0;
           i < sizeof(rejected_layouts) / sizeof(rejected_layouts[0]); ++i) {
         info.source_layout = rejected_layouts[i];
         memset(&encoded, 0xa5, sizeof(encoded));
         CHECK(!d12_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17S, &info,
                                                       &encoded));
         if (check_zero_encoding(&encoded))
            return 1;
      }

      info.source_layout =
         D12_AGX3_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;
      for (int draw = -4; draw <= 7; ++draw) {
         if (draw >= D12_AGX3_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED &&
             draw <= D12_AGX3_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED)
            continue;
         info.draw_path =
            (enum d12_agx3_vdm_state_update_draw_path)draw;
         memset(&encoded, 0xa5, sizeof(encoded));
         CHECK(!d12_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17S, &info,
                                                       &encoded));
         if (check_zero_encoding(&encoded))
            return 1;
      }

      info.draw_path = D12_AGX3_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED;
      info.raw_provoking_selector = 3;
      CHECK(!d12_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17S, &info,
                                                    &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      info.raw_provoking_selector = 0;

      CHECK(!d12_agx3_encode_vdm_state_update_draw(AGX_CHIP_G16P, &info,
                                                    &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      CHECK(!d12_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17S, NULL,
                                                    &encoded));
      if (check_zero_encoding(&encoded))
         return 1;
      CHECK(!d12_agx3_encode_vdm_state_update_draw(AGX_CHIP_G17S, &info,
                                                    NULL));
   }

   return 0;
}

static int
check_final_vertex_uvs_producer(void)
{
   struct agx_g17p_vertex_compile_info info = {
      .hardware_vertex = true,
      .position_off = 0, .varyings_off = 4,
      .psiz_off = 4, .layer_viewport_off = 4, .clip_off = 4,
      .uvs_size = 4,
   };
   struct d12_agx3_vdm_state_update_program_info d12;
   struct agx_vdm_state_update_program_info shared;
   CHECK(d12_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &info, &d12));
   CHECK(agx_g17p_vertex_program_info(AGX_CHIP_G17P, &info, &shared));
   CHECK(d12.source_layout ==
         D12_AGX3_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT);
   CHECK(d12.template_info.raw_vs_output_count_0 == 4);
   CHECK(d12.template_info.raw_state_load_count_12 == 0);
   CHECK(d12.template_info.raw_state_load_count_18 == 0);
   CHECK(d12.template_info.raw_active_context_selector == UINT32_MAX);
   CHECK(d12.template_info.raw_vs_output_count_0 ==
         shared.template_info.raw_vs_output_count_0);

   /* The default D3D vertex path appends a one-word point size. */
   info.layer_viewport_off = info.clip_off = info.uvs_size = 5;
   CHECK(d12_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &info, &d12));
   CHECK(d12.template_info.raw_vs_output_count_0 == 5);

   /* Each corrupted final-layout axis must keep the caller unqualified. */
   const struct agx_g17p_vertex_compile_info good = info;
   for (unsigned bad = 0; bad < 9; ++bad) {
      info = good;
      switch (bad) {
      case 0: info.hardware_vertex = false; break;
      case 1: info.position_off = 1; break;
      case 2: info.varyings_off = 3; break;
      case 3: info.psiz_off = 6; break;
      case 4: info.layer_viewport_off = 7; break;
      case 5: info.clip_off = 8; break;
      case 6: info.uvs_size = 4; break;
      case 7: info.has_preamble = true; break;
      case 8: info.nr_preamble_gprs = 1; break;
      }
      memset(&d12, 0xa5, sizeof(d12));
      CHECK(!d12_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &info, &d12));
      CHECK(d12.source_layout == D12_AGX3_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED);
   }
   info = good;
   CHECK(!d12_agx3_g17p_vertex_program_info(AGX_CHIP_G17S, &info, &d12));
   CHECK(!d12_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, NULL, &d12));
   CHECK(!d12_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &info, NULL));

   /* Apple's compiler bounds all-f32 outputs at position + 31 float4s. */
   info = (struct agx_g17p_vertex_compile_info){
      .hardware_vertex = true, .varyings_off = 4,
      .psiz_off = 128, .layer_viewport_off = 128, .clip_off = 128,
      .uvs_size = 128, .user_size = 124,
   };
   CHECK(d12_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &info, &d12));
   CHECK(d12.template_info.raw_vs_output_count_0 == 128);
   info.user_size = 125;
   info.psiz_off = info.layer_viewport_off = info.clip_off = info.uvs_size = 129;
   CHECK(!d12_agx3_g17p_vertex_program_info(AGX_CHIP_G17P, &info, &d12));
   return 0;
}

int
main(void)
{
   if (check_targets_and_shared_equality() ||
       check_domains_and_corruptions() || check_template_bridge() ||
       check_native_draw_template_bridge() || check_rejections() ||
       check_final_vertex_uvs_producer())
      return 1;

   printf("d12_agx3_vdm_state_update_core: %u checks passed\n", checks);
   return 0;
}
