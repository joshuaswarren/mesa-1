/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Offset interpolation needs the pixel's position within the tile.  On G17P
 * the pixel-coordinate special register disagrees with the interpolated
 * fragment position once the framebuffer is multisampled, which put every
 * dEQP-VK.pipeline.*.multisample_interpolation.offset_interpolate_at_pixel_center
 * case -- 28 of them, 2x and 4x -- on the wrong pixel.  The native path must
 * therefore take the coordinate from the position varying.
 *
 * This pins which source each path uses.  Against the previous lowering, the
 * native case fails: it found load_pixel_coord and no load_frag_coord.  AGX2
 * and the other callers keep the register, so the cheap path is not lost.
 */

#include "agx_compile.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>

static void
build_at_offset_fs(nir_builder *b)
{
   nir_def *offset = nir_imm_vec2(b, 0.0f, 0.0f);
   nir_def *bary = nir_load_barycentric_at_offset(b, 32, offset,
                                                  .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *v = nir_load_interpolated_input(b, 1, 32, bary, nir_imm_int(b, 0),
                                            .base = 0, .component = 0,
                                            .dest_type = nir_type_float32,
                                            .io_semantics.location = VARYING_SLOT_VAR0,
                                            .io_semantics.num_slots = 1);
   nir_store_output(b, v, nir_imm_int(b, 0), .base = 0, .write_mask = 1,
                    .src_type = nir_type_float32,
                    .io_semantics.location = FRAG_RESULT_DATA0,
                    .io_semantics.num_slots = 1);
}

static bool
uses(nir_shader *s, nir_intrinsic_op op)
{
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               return true;
         }
      }
   }
   return false;
}

static bool
check(bool native, bool want_frag_coord)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  &agx_nir_options, "interp");
   build_at_offset_fs(&b);
   agx_nir_lower_interpolation(b.shader, native);

   const bool fc = uses(b.shader, nir_intrinsic_load_frag_coord);
   const bool pc = uses(b.shader, nir_intrinsic_load_pixel_coord);
   const bool ok = (fc == want_frag_coord) && (pc == !want_frag_coord);

   if (!ok) {
      fprintf(stderr,
              "native=%d: expected %s, got load_frag_coord=%d load_pixel_coord=%d\n",
              native, want_frag_coord ? "the position varying" : "the register",
              fc, pc);
   }
   ralloc_free(b.shader);
   return ok;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool ok = true;
   /* G17P native coefficients: the register is unreliable under MSAA. */
   ok &= check(true, true);
   /* Everyone else keeps the cheap special register. */
   ok &= check(false, false);
   glsl_type_singleton_decref();

   if (ok)
      printf("PASS offset interpolation takes its pixel position from the "
             "position varying on the native path and the register elsewhere\n");
   return ok ? 0 : 1;
}
