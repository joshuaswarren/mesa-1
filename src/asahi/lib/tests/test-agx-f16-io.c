/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_abi.h"
#include "agx_compile.h"
#include "agx_linker.h"
#include "agx_nir_lower_vbo.h"
#include "agx_uvs.h"
#include "nir_builder.h"
#include "util/bitset.h"
#include "util/format/u_formats.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Exact G17P binary after the f16 logical output has become an f32 UVS
 * export. The fifth UVSW is the user varying; the first four are position.
 *
 * Re-pinned against the agx3 92a8b2d13e3 compiler, which now issues the
 * vertex-id read and its int->float conversion before the constants. Same
 * instructions and size, renamed registers: h0 = float(vertex id), h2 = -1.0,
 * h4 = 0, h6 = 1.0 (was h6/h4/h2/h0), and the five staging copies still feed
 * slots 1..5 with float(vid), -1.0, 0, 1.0, float(vid); the five UVSWs and the
 * end are byte-identical to the previous pin. */
static const uint8_t g17p_f16_output_vs[] = {
   0x0c,0xdd,0x10,0x06,0xa7,0x17,0x54,0x00,0x02,0x00,0xac,0x20,
   0x1c,0x80,0x02,0xbe,0x00,0x00,0x00,0x0c,0x2b,0x00,0x00,0x00,
   0x3c,0x80,0x02,0x3e,0x00,0x00,0x00,0x0c,0x0b,0x00,0x21,0x04,
   0x1b,0x03,0x27,0x00,0x42,0x00,0x00,0x04,0x00,0x00,0x2b,0x05,
   0x27,0x00,0x42,0x00,0x00,0x08,0x00,0x00,0x3b,0x07,0x27,0x00,
   0x42,0x00,0x00,0x0c,0x00,0x00,0x4b,0x01,0x27,0x00,0x42,0x00,
   0x00,0x10,0x00,0x00,0x57,0x16,0x54,0x00,0x00,0x40,0x48,0x00,
   0x57,0x26,0x54,0x02,0x20,0x40,0x49,0x00,0x57,0x46,0x54,0x04,
   0x40,0x40,0x4a,0x00,0x57,0x86,0x54,0x06,0x60,0x40,0x4b,0x00,
   0x57,0x06,0x55,0x08,0x80,0x40,0x4c,0x00,0x0e,0x00,0x00,0x00,
};

static nir_shader *
build_f16_output(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17P f16 Vulkan output");

   nir_def *id = nir_u2f32(&b, nir_load_vertex_id(&b));

   nir_store_output(&b,
                    nir_vec4(&b, id, nir_imm_float(&b, -1.0f),
                             nir_imm_float(&b, 0.0f),
                             nir_imm_float(&b, 1.0f)),
                    nir_imm_int(&b, 0), .write_mask = 0xf,
                    .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS,
                    .io_semantics.num_slots = 1);

   nir_store_output(&b, nir_f2f16(&b, id),
                    nir_imm_int(&b, 0), .write_mask = 0x1,
                    .src_type = nir_type_float16,
                    .io_semantics.location = VARYING_SLOT_VAR0,
                    .io_semantics.num_slots = 1);

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return b.shader;
}

/* A separately compiled VS has no interpolation contract yet. In particular,
 * adding the default flat Layer must not move an eventual smooth colour from
 * slot 4 to slot 5 while the fragment linkage still reads slot 4. */
static bool
check_unlinked_interpolation_contract(void)
{
   nir_shader *nir = build_f16_output();
   nir->info.io_lowered = true;
   nir_opt_varyings_bulk(&nir, 1, true, UINT32_MAX, UINT32_MAX, NULL, NULL);
   bool valid = !nir->info.known_interpolation_qualifiers;

   agx_nir_lower_default_layer(nir);
   struct agx_unlinked_uvs_layout layout = {0};
   agx_nir_lower_uvs(nir, &layout, true);

   unsigned dynamic_indices = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_load_uvs_index_agx)
               dynamic_indices++;
         }
      }
   }
   valid &= dynamic_indices == 2;

   struct agx_varyings_vs smooth, flat;
   agx_assign_uvs(&smooth, &layout, VARYING_BIT_LAYER, 0);
   agx_assign_uvs(&flat, &layout,
                  VARYING_BIT_LAYER | BITFIELD64_BIT(VARYING_SLOT_VAR0), 0);
   valid &= smooth.slots[VARYING_SLOT_VAR0] == 4 &&
            smooth.slots[VARYING_SLOT_LAYER] == 5 &&
            flat.slots[VARYING_SLOT_LAYER] == 4 &&
            flat.slots[VARYING_SLOT_VAR0] == 5;

   if (!valid)
      fprintf(stderr, "unlinked VS lost its draw-time interpolation contract\n");
   ralloc_free(nir);
   return valid;
}

static bool
has_type_correct_varying_store(nir_shader *nir)
{
   unsigned stores = 0;
   unsigned converted = 0;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_uvs_agx)
               continue;

            ++stores;
            if (intr->src[0].ssa->bit_size != 32)
               return false;

            converted += nir_def_is_alu(intr->src[0].ssa) &&
                         nir_def_as_alu(intr->src[0].ssa)->op == nir_op_f2f32;
         }
      }
   }

   return stores == 5 && converted == 1;
}

static unsigned
count_native_uvsw(const struct agx_shader_part *part)
{
   const uint8_t *binary = part->binary;
   unsigned count = 0;

   for (unsigned i = 0; binary && i + 8 <= part->info.binary_size; ++i) {
      if (binary[i] == 0x57 && (binary[i + 2] & 0xfc) == 0x54) {
         ++count;
         i += 7;
      }
   }

   return count;
}

static nir_shader *
build_layer_viewport_output(bool write_layer, uint32_t layer,
                            bool write_viewport, uint32_t viewport,
                            bool viewport_varying)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "layer/viewport UVS packing");

   if (write_layer) {
      nir_store_output(&b, nir_imm_int(&b, layer), nir_imm_int(&b, 0),
                       .write_mask = 0x1, .src_type = nir_type_uint32,
                       .io_semantics.location = VARYING_SLOT_LAYER,
                       .io_semantics.num_slots = 1);
      b.shader->info.outputs_written |= VARYING_BIT_LAYER;
   }

   if (write_viewport) {
      nir_store_output(&b, nir_imm_int(&b, viewport), nir_imm_int(&b, 0),
                       .write_mask = 0x1, .src_type = nir_type_uint32,
                       .io_semantics.location = VARYING_SLOT_VIEWPORT,
                       .io_semantics.num_slots = 1,
                       .io_semantics.no_varying = !viewport_varying);
      b.shader->info.outputs_written |= VARYING_BIT_VIEWPORT;
   }

   return b.shader;
}

static bool
lowered_layer_viewport_value(bool write_layer,
                             uint32_t layer, bool write_viewport,
                             uint32_t viewport, bool viewport_varying,
                             uint32_t expected, unsigned expected_size,
                             unsigned expected_stores)
{
   nir_shader *nir = build_layer_viewport_output(
      write_layer, layer, write_viewport, viewport, viewport_varying);
   struct agx_unlinked_uvs_layout layout = {0};
   bool valid = agx_nir_lower_uvs(nir, &layout, false);

   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
   } while (progress);

   unsigned stores = 0, matching_stores = 0, viewport_stores = 0;
   uint32_t actual = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_uvs_agx)
               continue;

            stores++;
            if (!nir_src_is_const(intr->src[1])) {
               valid = false;
               continue;
            }

            const unsigned index = nir_src_as_uint(intr->src[1]);
            if (index == layout.group_offs[UVS_LAYER_VIEWPORT]) {
               matching_stores++;
               if (nir_src_is_const(intr->src[0]))
                  actual = nir_src_as_uint(intr->src[0]);
               else
                  valid = false;
            } else if (viewport_varying &&
                       index == layout.group_offs[UVS_VARYINGS]) {
               viewport_stores++;
               valid &= nir_src_is_const(intr->src[0]) &&
                        nir_src_as_uint(intr->src[0]) == viewport;
            }
         }
      }
   }

   const uint64_t expected_written =
      viewport_varying ? VARYING_BIT_VIEWPORT : 0;
   valid &= layout.size == expected_size &&
            layout.user_size == (viewport_varying ? 1 : 0) &&
            layout.written == expected_written &&
            layout.components[VARYING_SLOT_VIEWPORT] ==
               (viewport_varying ? 1 : 0) &&
            stores == expected_stores && matching_stores == 1 &&
            viewport_stores == (viewport_varying ? 1 : 0) &&
            actual == expected;
   if (!valid) {
      fprintf(stderr,
              "layer/viewport packing mismatch: layer=%#x/%u "
              "viewport=%#x/%u/%u actual=%#x expected=%#x stores=%u/%u "
              "size=%u user=%u written=%#" PRIx64 "\n",
              layer, write_layer, viewport, write_viewport,
              viewport_varying, actual, expected, stores, matching_stores,
              layout.size, layout.user_size, layout.written);
   }

   ralloc_free(nir);
   return valid;
}

static bool
lowered_g17p_layer_varying(bool explicit_layer)
{
   nir_shader *nir = build_layer_viewport_output(explicit_layer, 7, false, 0, false);
   bool added = false, repeated = false;
   NIR_PASS(added, nir, agx_nir_lower_default_layer);
   NIR_PASS(repeated, nir, agx_nir_lower_default_layer);
   bool valid = added == !explicit_layer && !repeated;
   valid &= (nir->info.outputs_written & VARYING_BIT_LAYER) != 0;
   const unsigned expected_layer = explicit_layer ? 7 : 0;
   struct agx_unlinked_uvs_layout layout = {0};
   valid &= agx_nir_lower_uvs(nir, &layout, true);

   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
   } while (progress);

   unsigned layer_stores = 0, packed_stores = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_uvs_agx ||
                !nir_src_is_const(intr->src[1]) ||
                !nir_src_is_const(intr->src[0]))
               continue;

            const unsigned index = nir_src_as_uint(intr->src[1]);
            const unsigned value = nir_src_as_uint(intr->src[0]);
            if (index == layout.group_offs[UVS_VARYINGS] && value == expected_layer)
               layer_stores++;
            if (index == layout.group_offs[UVS_LAYER_VIEWPORT] && value == expected_layer)
               packed_stores++;
         }
      }
   }

   struct agx_varyings_vs varyings;
   agx_assign_uvs(&varyings, &layout, VARYING_BIT_LAYER, 0);

   struct agx_varyings_fs fs = {
      .nr_cf = 1,
      .nr_bindings = 1,
      .bindings[0] = {
         .slot = VARYING_SLOT_LAYER,
         .count = 1,
      },
   };
   uint32_t table[2] = {0};
   bool generate_primitive_id = false;
   agx_link_varyings_vs_fs(table, &varyings, layout.user_size, &fs,
                           0, 0, &generate_primitive_id);

   valid &= layout.size == 6 && layout.user_size == 1 &&
            layout.written == VARYING_BIT_LAYER &&
            layout.components[VARYING_SLOT_LAYER] == 1 &&
            varyings.slots[VARYING_SLOT_LAYER] == 4 && layer_stores == 1 &&
            packed_stores == 1 && table[0] == UINT32_C(0x00000102) &&
            table[1] == UINT32_C(0x00000100) &&
            !generate_primitive_id;

   if (!valid) {
      fprintf(stderr,
              "G17P layer was not duplicated into fixed and flat UVS words: "
              "size=%u user=%u written=%#" PRIx64 " slot=%u stores=%u/%u\n",
              layout.size, layout.user_size, layout.written,
              varyings.slots[VARYING_SLOT_LAYER], layer_stores,
              packed_stores);
   }

   ralloc_free(nir);
   return valid;
}

/*
 * VS input side of the same 16-bit story.
 *
 * The vertex prolog is a separate shader part that publishes every attribute
 * component as a FULL 32-BIT value in its own pair of half-registers:
 * agx_nir_vs_prolog_impl() fetches with nir_load_input(b, 4, 32, ...) and
 * exports each channel at AGX_ABI_VIN_ATTRIB(4 * attrib + comp), which is
 * 2 * (8 + i) -- two halves apart. struct agx_vs_prolog_key carries no bit
 * sizes at all, so the prolog cannot publish anything narrower even in
 * principle.
 *
 * A main shader may still declare a narrower input once
 * storageInputOutput16 is advertised. The main shader must therefore load the
 * 32-bit ABI slot and narrow, never load 16 bits: agx_load_exported_to()
 * strides by the DESTINATION size, so a 16-bit load at the same base would
 * read component 1 out of component 0's high half.
 */
struct vs_input_case {
   const char *name;
   unsigned attrib;    /* attribute index (load_input base) */
   unsigned component; /* 32-bit slot component, 0..3 */
   unsigned comps;     /* number of components loaded */
   unsigned bit_size;
   nir_alu_type dest_type;
   nir_op narrow; /* expected narrowing op, nir_op_mov if none */
};

static nir_def *
widen_to_f32(nir_builder *b, nir_def *x, nir_alu_type dest_type)
{
   switch (nir_alu_type_get_base_type(dest_type)) {
   case nir_type_float:
      return nir_f2f32(b, x);
   case nir_type_int:
      return nir_i2f32(b, x);
   default:
      return nir_u2f32(b, x);
   }
}

static nir_shader *
build_vs_input(const struct vs_input_case *c)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "VS input");

   nir_def *attr = nir_load_input(
      &b, c->comps, c->bit_size, nir_imm_int(&b, 0), .base = c->attrib,
      .component = c->component, .dest_type = c->dest_type,
      .io_semantics.location = VERT_ATTRIB_GENERIC0 + c->attrib,
      .io_semantics.num_slots = 1);

   /* Consume every component so nir_def_components_read() sees them all. */
   nir_def *acc = nir_imm_float(&b, 0.0f);
   for (unsigned i = 0; i < c->comps; ++i)
      acc = nir_fadd(&b, acc, widen_to_f32(&b, nir_channel(&b, attr, i),
                                           c->dest_type));

   nir_store_output(&b,
                    nir_vec4(&b, acc, nir_imm_float(&b, 0.0f),
                             nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f)),
                    nir_imm_int(&b, 0), .write_mask = 0xf,
                    .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS,
                    .io_semantics.num_slots = 1);

   b.shader->info.inputs_read =
      BITFIELD64_BIT(VERT_ATTRIB_GENERIC0 + c->attrib);
   b.shader->info.outputs_written = VARYING_BIT_POS;
   return b.shader;
}

static nir_intrinsic_instr *
find_sole_intrinsic(nir_shader *nir, nir_intrinsic_op op, unsigned *count)
{
   nir_intrinsic_instr *found = NULL;
   *count = 0;

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != op)
               continue;

            ++(*count);
            if (!found)
               found = intr;
         }
      }
   }

   return found;
}

/* The single ALU consumer of def, or NULL if there is not exactly one. */
static nir_alu_instr *
sole_alu_use(nir_def *def)
{
   nir_alu_instr *alu = NULL;

   nir_foreach_use(use, def) {
      if (nir_src_parent_instr(use)->type != nir_instr_type_alu)
         return NULL;

      nir_alu_instr *this = nir_instr_as_alu(nir_src_parent_instr(use));
      if (alu && alu != this)
         return NULL;

      alu = this;
   }

   return alu;
}

static bool
lowered_vs_input(const struct vs_input_case *c)
{
   nir_shader *nir = build_vs_input(c);
   const unsigned slot = (4 * c->attrib) + c->component;
   bool ok = false;

   /* The gather reports attribute components in 32-BIT slot units, which is
    * the unit the prolog's component_mask and BITSET_FOREACH_SET use. A
    * 16-bit input still occupies one whole slot per component. */
   BITSET_DECLARE(mask, AGX_MAX_ATTRIBS * 4);
   BITSET_ZERO(mask);
   agx_nir_gather_vs_inputs(nir, mask);

   BITSET_DECLARE(expected, AGX_MAX_ATTRIBS * 4);
   BITSET_ZERO(expected);
   BITSET_SET_RANGE(expected, slot, slot + c->comps - 1);

   if (!BITSET_EQUAL(mask, expected)) {
      fprintf(stderr, "%s: gathered components %08x, expected %08x\n", c->name,
              mask[0], expected[0]);
      goto done;
   }

   agx_nir_lower_vs_input_to_prolog(nir);

   unsigned nr_inputs = 0, nr_loads = 0;
   find_sole_intrinsic(nir, nir_intrinsic_load_input, &nr_inputs);
   nir_intrinsic_instr *load =
      find_sole_intrinsic(nir, nir_intrinsic_load_exported_agx, &nr_loads);

   if (nr_inputs != 0 || nr_loads != 1) {
      fprintf(stderr, "%s: %u load_input and %u load_exported_agx left\n",
              c->name, nr_inputs, nr_loads);
      goto done;
   }

   /* The ABI slot is 32-bit wide whatever the shader asked for. */
   if (load->def.bit_size != 32 ||
       load->def.num_components != c->comps ||
       nir_intrinsic_base(load) != AGX_ABI_VIN_ATTRIB(slot)) {
      fprintf(stderr,
              "%s: load_exported_agx %ux%ub base %u, expected %ux32b base %u\n",
              c->name, load->def.num_components, load->def.bit_size,
              nir_intrinsic_base(load), c->comps, AGX_ABI_VIN_ATTRIB(slot));
      goto done;
   }

   nir_alu_instr *narrow = sole_alu_use(&load->def);

   if (c->narrow == nir_op_mov) {
      /* 32-bit: nothing may be inserted between the ABI and the shader. */
      ok = narrow == NULL || (narrow->op != nir_op_f2f16 &&
                              narrow->op != nir_op_i2i16 &&
                              narrow->op != nir_op_u2u16);
      if (!ok) {
         fprintf(stderr, "%s: 32-bit input grew a %s\n", c->name,
                 nir_op_infos[narrow->op].name);
      }
   } else {
      ok = narrow != NULL && narrow->op == c->narrow &&
           narrow->def.bit_size == c->bit_size &&
           narrow->def.num_components == c->comps;
      if (!ok) {
         fprintf(stderr, "%s: expected a %s of the ABI slot, got %s\n", c->name,
                 nir_op_infos[c->narrow].name,
                 narrow ? nir_op_infos[narrow->op].name : "nothing");
      }
   }

done:
   ralloc_free(nir);
   return ok;
}

/*
 * Round-trip the ABI: build the prolog from the mask the gather produced and
 * check it exports 32-bit values at exactly the bases the main shader loads.
 * This is what makes the "narrow after the load" rule ground truth rather
 * than a guess -- if the prolog ever learned to publish 16-bit halves, this
 * is the test that would notice.
 */
static bool
prolog_publishes_32bit_slots(unsigned attrib, unsigned component,
                             unsigned comps, enum pipe_format format)
{
   struct agx_vs_prolog_key key = {
      .hw = true,
      .robustness = {.level = AGX_ROBUSTNESS_GL},
   };

   const unsigned slot = (4 * attrib) + component;
   BITSET_SET_RANGE(key.component_mask, slot, slot + comps - 1);

   key.attribs[attrib] = (struct agx_velem_key){
      .format = format,
      .stride = 16,
   };

   nir_builder b =
      nir_builder_init_simple_shader(0, &agx_nir_options, "VS prolog");
   agx_nir_vs_prolog(&b, &key);

   BITSET_DECLARE(exported, AGX_MAX_ATTRIBS * 4);
   BITSET_ZERO(exported);
   bool ok = true;

   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_export_agx)
               continue;

            unsigned base = nir_intrinsic_base(intr);
            if (base < AGX_ABI_VIN_ATTRIB(0))
               continue; /* vertex/instance ID */

            unsigned i = (base - AGX_ABI_VIN_ATTRIB(0)) / 2;
            if (AGX_ABI_VIN_ATTRIB(i) != base ||
                intr->src[0].ssa->bit_size != 32) {
               fprintf(stderr,
                       "prolog exported %ub at base %u, not a 32-bit slot\n",
                       intr->src[0].ssa->bit_size, base);
               ok = false;
            } else {
               BITSET_SET(exported, i);
            }
         }
      }
   }

   if (ok && !BITSET_EQUAL(exported, key.component_mask)) {
      fprintf(stderr, "prolog exported %08x, mask asked for %08x\n",
              exported[0], key.component_mask[0]);
      ok = false;
   }

   ralloc_free(b.shader);
   return ok;
}

/* 32-bit is the regression guard: it must still lower to a bare 32-bit ABI
 * load with nothing inserted. The 16-bit rows are the storageInputOutput16
 * cases that used to trip the bit_size == 32 assert in gather_inputs(). */
static const struct vs_input_case vs_input_cases[] = {
   {"vec4 f32 input", 0, 0, 4, 32, nir_type_float32, nir_op_mov},
   {"ivec2 i32 input", 3, 1, 2, 32, nir_type_int32, nir_op_mov},
   {"f16vec2 input", 1, 0, 2, 16, nir_type_float16, nir_op_f2f16},
   {"f16 input at component 3", 2, 3, 1, 16, nir_type_float16, nir_op_f2f16},
   {"i16vec4 input", 4, 0, 4, 16, nir_type_int16, nir_op_i2i16},
   {"u16vec3 input at component 1", 5, 1, 3, 16, nir_type_uint16, nir_op_u2u16},
};

int
main(void)
{
   if (!check_unlinked_interpolation_contract())
      return 1;

   for (unsigned i = 0; i < ARRAY_SIZE(vs_input_cases); ++i) {
      if (!lowered_vs_input(&vs_input_cases[i]))
         return 1;
   }

   if (!prolog_publishes_32bit_slots(1, 0, 2, PIPE_FORMAT_R16G16_FLOAT) ||
       !prolog_publishes_32bit_slots(4, 0, 4, PIPE_FORMAT_R16G16B16A16_SINT) ||
       !prolog_publishes_32bit_slots(0, 0, 4, PIPE_FORMAT_R32G32B32A32_FLOAT)) {
      fprintf(stderr, "vertex prolog ABI is not 32-bit per component\n");
      return 1;
   }

   /* The generation-neutral path keeps layer in the packed fixed-function
    * word. G17P additionally duplicates a fragment-consumed layer into one
    * ordinary flat UVS word, matching the six-UVSW Apple vertex witness. */
   if (!lowered_g17p_layer_varying(false) || !lowered_g17p_layer_varying(true))
      return 1;

   /* Fresh applegpu_g17p witnesses are exact on both sides of no_varying:
    * viewport-only with an unread FS output has five UVSWs and a five-word
    * layout, while a fragment read adds exactly one ordinary UVSW/word. Layer
    * and viewport share one packed scalar in both cases: the layer is its low
    * half and the viewport its high half, as on AGX2. */
   if (!lowered_layer_viewport_value(true, 1, false, 0, false,
                                     UINT32_C(0x00000001), 5, 1) ||
       !lowered_layer_viewport_value(false, 0, true, 2, false,
                                     UINT32_C(0x00020000), 5, 1) ||
       !lowered_layer_viewport_value(false, 0, true, 2, true,
                                     UINT32_C(0x00020000), 6, 2) ||
       !lowered_layer_viewport_value(true, 1, true, 2, false,
                                     UINT32_C(0x00020001), 5, 1) ||
       !lowered_layer_viewport_value(true, 1, true, 2, true,
                                     UINT32_C(0x00020001), 6, 2))
      return 1;

   nir_shader *nir = build_f16_output();
   struct agx_unlinked_uvs_layout layout = {0};

   if (!agx_nir_lower_uvs(nir, &layout, false) || layout.size != 5 ||
       layout.user_size != 1 || !has_type_correct_varying_store(nir)) {
      fprintf(stderr, "f16 output was not widened to f32 before ST_VARY\n");
      ralloc_free(nir);
      return 1;
   }

   agx_preprocess_nir(nir);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(nir, &key, &part);
   const bool native_ok =
      compiled && part.binary &&
      part.info.binary_size == sizeof(g17p_f16_output_vs) &&
      memcmp(part.binary, g17p_f16_output_vs,
             sizeof(g17p_f16_output_vs)) == 0 &&
      count_native_uvsw(&part) == 5;

   if (!native_ok) {
      fprintf(stderr,
              "type-correct f16 output native mismatch: compiled=%u size=%u "
              "uvsw=%u\n",
              compiled, part.info.binary_size, count_native_uvsw(&part));
      for (unsigned i = 0; i < part.info.binary_size; ++i)
         fprintf(stderr, "%02x", ((const uint8_t *)part.binary)[i]);
      fputc('\n', stderr);
   }

   free(part.binary);
   ralloc_free(nir);
   return native_ok ? 0 : 1;
}
