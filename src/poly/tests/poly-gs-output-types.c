/* SPDX-License-Identifier: MIT */
#include "nir_builder.h"
#include "poly_nir.h"

#include <stdio.h>

static bool
check_types(nir_alu_type first, nir_alu_type second, bool overwrite)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &options, "mixed GS outputs");
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.io_lowered = true;

   nir_store_output(&b, nir_imm_int(&b, 0xdeadbeef), nir_imm_int(&b, 0),
                    .component = 0, .src_type = first,
                    .io_semantics.location = VARYING_SLOT_VAR0);
   nir_store_output(&b, nir_imm_int(&b, 0x80000001), nir_imm_int(&b, 0),
                    .component = 1, .src_type = second,
                    .io_semantics.location = VARYING_SLOT_VAR0);
   /* One packed component can also be assigned different source types. */
   if (overwrite) {
      nir_store_output(&b, nir_imm_int(&b, 0x3f800000), nir_imm_int(&b, 0),
                       .component = 0, .src_type = second,
                       .io_semantics.location = VARYING_SLOT_VAR0);
   }
   nir_emit_vertex(&b, 0);
   nir_end_primitive(&b, 0);

   nir_shader *count = NULL, *copy = NULL, *pre = NULL;
   struct poly_gs_info info;
   poly_nir_lower_gs(b.shader, &count, &copy, &pre, &info);
   nir_validate_shader(copy, "GS output types");

   nir_alu_type found[2] = {nir_type_invalid, nir_type_invalid};
   nir_foreach_block(block, nir_shader_get_entrypoint(copy)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output ||
             nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_VAR0)
            continue;
         unsigned mask = nir_intrinsic_write_mask(intr);
         for (unsigned c = 0; c < 2; ++c) {
            if (mask & BITFIELD_BIT(c))
               found[c] = nir_intrinsic_src_type(intr);
         }
      }
   }
   nir_alu_type expected = overwrite && first != second ? nir_type_uint32 : first;
   bool valid = found[0] == expected && found[1] == second;
   if (!valid)
      fprintf(stderr, "GS types %u/%u became %u/%u\n",
              first, second, found[0], found[1]);
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(copy);
   ralloc_free(pre);
   return valid;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool valid = true;
   const nir_alu_type types[] = {nir_type_float32, nir_type_uint32,
                                 nir_type_int32};
   for (unsigned a = 0; a < ARRAY_SIZE(types); ++a) {
      for (unsigned b = 0; b < ARRAY_SIZE(types); ++b) {
         valid &= check_types(types[a], types[b], false);
         valid &= check_types(types[a], types[b], true);
      }
   }
   glsl_type_singleton_decref();
   return valid ? 0 : 1;
}
