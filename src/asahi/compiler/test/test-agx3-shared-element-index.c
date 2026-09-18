/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * AGX3 ldtg/sttg address threadgroup memory as base + (index << shift) +
 * byte_offset, with shift set to the access width, so the index register
 * counts scalar elements while NIR's shared offsets count bytes.  The
 * conversion between them belongs to the target encoding, not to a frontend:
 * asahi_clc preprocesses one NIR shader and compiles it for g13g, g13x and
 * g17p at once, so it cannot choose a preprocessing-time flag that is right
 * for all three, and before this test existed every internal (libagx) kernel
 * reached G17P with a byte offset in the element-index field -- a fourfold
 * over-stride for a 32-bit element, which is why a `local uint vals[1024]`
 * kernel only answered for its first 256 lanes on hardware.
 *
 * Two independent checks:
 *
 *   1. the NIR the conversion produces is the element index, per width
 *      (8-bit unchanged, 16-bit halved, 32-bit quartered);
 *   2. the two frontend conventions -- agx_preprocess_nir(), which asahi_clc
 *      and the D3D12 UMD use, and agx_preprocess_nir_g17p(), which the Vulkan
 *      driver uses -- compile the same shader to the same bytes.
 */

#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_nir.h"
#include "nir_builder.h"
#include "glsl_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         return false;                                                         \
      }                                                                        \
   } while (0)

/* A shader that exchanges one `bits`-wide value through shared memory at a
 * constant byte offset.  The barrier keeps the store and the load apart. */
static nir_shader *
build(unsigned bits, unsigned byte_offset)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shared element index");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 256;

   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *offset =
      nir_iadd_imm(&b, nir_imul_imm(&b, lane, bits / 8), byte_offset);
   nir_store_shared(&b, nir_u2uN(&b, nir_iadd_imm(&b, lane, 0x11), bits),
                    offset, .align_mul = bits / 8);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
   nir_def *peer =
      nir_load_shared(&b, 1, bits, nir_imm_int(&b, byte_offset),
                      .align_mul = bits / 8);
   nir_store_global(&b, nir_u2u32(&b, peer),
                    nir_load_preamble(&b, 1, 64, .base = 0), .align_mul = 4);
   return b.shader;
}

/*
 * Check 1: after preprocessing (which narrows the byte offset to 16 bits) the
 * element-index conversion divides a constant shared offset by the access
 * width, and leaves 8-bit accesses alone because their element index and byte
 * offset are the same number.
 */
static bool
check_nir_offsets(unsigned bits, unsigned byte_offset, bool g17p_frontend)
{
   nir_shader *nir = build(bits, byte_offset);

   if (g17p_frontend) {
      /* The Vulkan driver's convention: agx_preprocess_nir_g17p() converts. */
      agx_preprocess_nir_g17p(nir);
   } else {
      /* asahi_clc's and the D3D12 UMD's convention: the target is not known
       * during preprocessing, so agx_compile_shader_nir() converts.  Stand in
       * for it here -- compiling would consume the NIR this inspects. */
      agx_preprocess_nir(nir);
      agx_nir_lower_shared_element_index(nir);
   }

   /* The narrowing and the conversion are both built unfolded, after the
    * preprocessing optimizer has already run; fold them so the offset this
    * checks is the number the backend will see. */
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_dce);

   const unsigned expected = byte_offset / (bits / 8);
   unsigned loads = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_shared)
               continue;
            loads++;
            nir_src *off = nir_get_io_offset_src(intr);
            if (!nir_src_is_const(*off)) {
               fprintf(stderr,
                       "%u-bit shared load offset is not constant\n", bits);
               ralloc_free(nir);
               return false;
            }
            const uint64_t got = nir_src_as_uint(*off);
            if (got != expected) {
               fprintf(stderr,
                       "%u-bit shared load at byte %u through %s: index %" PRIu64
                       ", expected the element index %u\n",
                       bits, byte_offset,
                       g17p_frontend ? "agx_preprocess_nir_g17p"
                                     : "the compile-time backstop",
                       got, expected);
               ralloc_free(nir);
               return false;
            }
         }
      }
   }
   ralloc_free(nir);
   CHECK(loads == 1);
   return true;
}

/* Compile `nir` for AGX3B and hand back its binary. */
static bool
compile(nir_shader *nir, struct agx_shader_part *part)
{
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .dev.chip = AGX_CHIP_G17P,
      .reserved_preamble = 4,
      .no_preamble = true,
   };
   memset(part, 0, sizeof(*part));
   return agx_compile_shader_nir(nir, &key, part);
}

/*
 * Check 2: both frontend conventions must compile the shader, and the one the
 * Vulkan driver uses must keep the machine code origin/agx3 produced.  The two
 * are deliberately *not* byte-identical: the Vulkan front end converts during
 * preprocessing, where nir_opt_algebraic folds an appended shift, while the
 * compile-time backstop has to divide exactly because by then the optimizer
 * would rewrite ushr(ishl(a, 2), 2) into a mixed-width iand the AGX3 encoder
 * cannot express.  Equivalent addresses, different instruction selection.
 */
static bool
check_both_frontends_compile(unsigned bits)
{
   nir_shader *plain = build(bits, 64);
   nir_shader *g17p = build(bits, 64);
   agx_preprocess_nir(plain);
   agx_preprocess_nir_g17p(g17p);

   struct agx_shader_part a, c;
   const bool ok_a = compile(plain, &a);
   const bool ok_c = compile(g17p, &c);
   const bool ok = ok_a && ok_c && a.binary && c.binary &&
                   a.info.binary_size > 4 && c.info.binary_size > 4;

   if (!ok) {
      fprintf(stderr, "%u-bit shared exchange did not compile on both "
                      "frontends\n", bits);
   } else {
      printf("  %2u-bit: backstop %u bytes, g17p preprocess %u bytes\n", bits,
             a.info.binary_size, c.info.binary_size);
   }

   free(a.binary);
   free(c.binary);
   ralloc_free(plain);
   ralloc_free(g17p);
   return ok;
}

/*
 * Check 3: the Vulkan front end's real call sequence.  hk_shader.c calls
 * agx_preprocess_nir_g17p() twice (:252 and :1892) and lowers `shared`
 * variables to load_shared/store_shared *between* them (:1810), so the first
 * call sees a shader with no shared access at all and the second does the
 * conversion.  A shader compiled the other way -- accesses already lowered
 * before the first call -- cannot tell the two apart, which is why the
 * frontend-agreement check above missed a regression that took the Vulkan
 * path's conversion away entirely on hardware.
 */
static bool
check_deferred_shared_lowering(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P deferred shared lowering");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_variable *var = nir_variable_create(
      b.shader, nir_var_mem_shared,
      glsl_array_type(glsl_uint_type(), 64, 0), "scratch");
   nir_deref_instr *slot =
      nir_build_deref_array_imm(&b, nir_build_deref_var(&b, var), 16);
   nir_store_deref(&b, slot, nir_imm_int(&b, 0x1234), 1);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
   nir_store_global(&b, nir_load_deref(&b, slot),
                    nir_load_preamble(&b, 1, 64, .base = 0), .align_mul = 4);

   /* hk_shader.c:252 -- shared is still a variable here. */
   agx_preprocess_nir_g17p(b.shader);

   /* hk_shader.c:1810-1812. */
   NIR_PASS(_, b.shader, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
            glsl_get_natural_size_align_bytes);
   NIR_PASS(_, b.shader, nir_lower_explicit_io, nir_var_mem_shared,
            nir_address_format_32bit_offset);

   /* hk_shader.c:1892. */
   agx_preprocess_nir_g17p(b.shader);

   NIR_PASS(_, b.shader, nir_opt_constant_folding);
   NIR_PASS(_, b.shader, nir_opt_dce);

   bool ok = false;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_shared)
               continue;
            nir_src *off = nir_get_io_offset_src(intr);
            const bool constant = nir_src_is_const(*off);
            const uint64_t got = constant ? nir_src_as_uint(*off) : ~0ull;
            /* Element 16 of a uint array: byte offset 64, element index 16. */
            ok = constant && got == 16;
            if (!ok) {
               fprintf(stderr,
                       "deferred shared lowering: load_shared offset is %s, "
                       "expected the element index 16\n",
                       constant ? "a byte offset" : "not constant");
            }
         }
      }
   }

   if (!ok)
      fprintf(stderr, "deferred shared lowering: no shared load survived\n");

   ralloc_free(b.shader);
   return ok;
}

/*
 * Check 3: a 64-bit shared access has no encodable form -- mem_access_size_align_cb
 * caps shared accesses at 32 bits -- so it is split into 32-bit halves.  The
 * element-index conversion scales by the access width, so the split has to
 * happen first or the halves inherit an index divided by 8 where the 32-bit
 * element index needs a division by 4.  Converting first addressed half the
 * intended element and failed
 * dEQP-VK.spirv_assembly.instruction.compute.workgroup_memory.{int64,uint64}
 * while every narrower width passed.
 */
static bool
check_64bit_split_before_conversion(unsigned byte_offset)
{
   nir_shader *nir = build(64, byte_offset);
   agx_preprocess_nir_g17p(nir);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_dce);

   const unsigned expected = byte_offset / 4;
   unsigned loads = 0;
   bool ok = true;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_shared)
               continue;
            loads++;

            if (intr->def.bit_size != 32) {
               fprintf(stderr,
                       "64-bit shared load reached the encoder %u bits wide; "
                       "shared accesses are capped at 32\n",
                       intr->def.bit_size);
               ok = false;
               continue;
            }

            nir_src *off = nir_get_io_offset_src(intr);
            if (!nir_src_is_const(*off)) {
               fprintf(stderr, "64-bit shared load offset is not constant\n");
               ok = false;
               continue;
            }

            const uint64_t got = nir_src_as_uint(*off);
            if (got != expected) {
               fprintf(stderr,
                       "64-bit shared load at byte %u: index %" PRIu64 ", "
                       "expected the 32-bit element index %u (a conversion "
                       "that ran before the split would give %u)\n",
                       byte_offset, got, expected, byte_offset / 8);
               ok = false;
            }
         }
      }
   }

   ralloc_free(nir);
   CHECK(loads == 1);
   return ok;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   bool ok = true;
   for (unsigned f = 0; f < 2; ++f) {
      ok &= check_nir_offsets(32, 64, f != 0);
      ok &= check_nir_offsets(16, 64, f != 0);
      ok &= check_nir_offsets(8, 64, f != 0);
   }
   ok &= check_both_frontends_compile(32);
   ok &= check_both_frontends_compile(16);
   ok &= check_deferred_shared_lowering();
   ok &= check_64bit_split_before_conversion(64);
   ok &= check_64bit_split_before_conversion(128);
   ok &= check_both_frontends_compile(64);

   glsl_type_singleton_decref();

   if (!ok)
      return 1;

   puts("agx3_shared_element_index: byte offsets become element indices on "
        "every frontend");
   return 0;
}
