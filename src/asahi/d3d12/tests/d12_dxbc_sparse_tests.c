/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Independently assembled public SM5 token grammar. No external bytecode.
 */

#include "../d12_dxbc_wrap.h"
#include "../dxil/d12_dxil.h"
#include "compiler/glsl_types.h"
#include "nir.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

struct tokens {
   uint32_t words[128];
   unsigned count;
};

static void
word(struct tokens *t, uint32_t v)
{
   assert(t->count < 128);
   t->words[t->count++] = v;
}

static void
reg(struct tokens *t, uint32_t token, unsigned index)
{
   word(t, token);
   word(t, index);
}

static void
immediate(struct tokens *t, uint32_t v)
{
   word(t, 0x00004002);
   for (unsigned i = 0; i < 4; i++)
      word(t, v);
}

static void
check(unsigned opcode, bool null_clamp, bool null_feedback, bool buffer)
{
   struct tokens t = {0};
   bool load = opcode >= 223 && opcode <= 227;
   bool compare = opcode == 220 || opcode == 222 || opcode == 229 || opcode == 233;
   bool offset = opcode == 221 || opcode == 222;
   bool clamp = opcode >= 230;
   word(&t, 0x50); /* fragment, SM5.0 container grammar */
   word(&t, 0);
   word(&t, 0x02000068); word(&t, 2); /* two temporaries */
   if (opcode == 226 || opcode == 227) {
      word(&t, ((opcode == 227 ? 4u : 3u) << 24) | (opcode == 227 ? 162u : 161u));
      reg(&t, 0x00107000, 0);
      if (opcode == 227)
         word(&t, 16);
   } else {
      word(&t, (4u << 24) | ((buffer ? 1u : opcode == 224 ? 4u : 3u) << 11) |
               (opcode == 225 ? 156u : 88u));
      reg(&t, opcode == 225 ? 0x0011e000 : 0x00107000, 0);
      /* UNORM/SNORM declarations produce floating-point values too. */
      word(&t, opcode == 225 && !buffer
                  ? (null_feedback ? 0x2222 : null_clamp ? 0x1111 : 0x5555)
                  : 0x5555);
   }
   word(&t, 0x0300005a); reg(&t, 0x00106000, 0); /* s0 */
   unsigned start = t.count;
   word(&t, opcode);
   reg(&t, 0x001000f2, 0); /* data destination r0.xyzw */
   if (null_feedback)
      word(&t, 0x0000d000);
   else
      reg(&t, 0x00100012, 1); /* residency destination r1.x */
   immediate(&t, load ? 0 : 0x3f000000); /* coordinate */
   if (offset)
      immediate(&t, 0);
   if (opcode == 227)
      immediate(&t, 0); /* member byte offset */
   reg(&t, opcode == 225 ? 0x0011ee46 : 0x00107e46, 0); /* u0/t0.xyzw */
   if (!load)
      reg(&t, 0x0010600a, 0); /* s0.x (gather component) */
   if (compare || opcode == 228 || opcode == 224 || opcode == 231)
      immediate(&t, opcode == 228 ? 0x3fa00000 : 0);
   if (opcode == 232) {
      immediate(&t, 0x3c800000);
      immediate(&t, 0x3c800000);
   }
   if (clamp) {
      if (null_clamp)
         word(&t, 0x0000d000);
      else
         immediate(&t, 0x3fa00000); /* explicit minimum LOD 1.25 */
   }
   if (opcode == 228 && null_clamp)
      word(&t, 0); /* fxc's length includes this padding word. */
   t.words[start] |= (t.count - start) << 24;
   if (!null_feedback) {
      word(&t, 0x050000ea); /* check_access_fully_mapped o1, r1.x */
      reg(&t, 0x001020f2, 1);
      reg(&t, 0x0010000a, 1);
   }
   word(&t, 0x05000036); /* mov o0, r0 */
   reg(&t, 0x001020f2, 0); reg(&t, 0x00100e46, 0);
   word(&t, 0x0100003e);
   t.words[1] = t.count;
   void *container;
   size_t size;
   assert(d12_dxbc_wrap_tokens(t.words, NULL, &container, &size));
   const char *error = NULL;
   nir_shader *shader = d12_dxbc_to_nir(container, size, MESA_SHADER_FRAGMENT,
                                      NULL, 7, NULL, &error);
   if (!shader)
      fprintf(stderr, "opcode %u: %s\n", opcode, error);
   assert(shader);
   while (nir_opt_constant_folding(shader)) {}
   unsigned textures = 0, residency_checks = 0, min_lods = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               assert(tex->is_sparse == (!null_feedback && !buffer));
               assert(tex->def.num_components == (null_feedback || buffer ? 4 : 5));
               textures++;
               min_lods += nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0;
               if (opcode == 228) {
                  int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
                  assert(lod >= 0 && nir_src_is_const(tex->src[lod].src));
                  assert(nir_src_as_float(tex->src[lod].src) == 1.25f);
               }
            } else if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               residency_checks += intr->intrinsic == nir_intrinsic_is_sparse_texels_resident;
            }
         }
      }
   }
   assert(textures == (opcode == 226 || opcode == 227 ? 0 : 1));
   assert(residency_checks == !null_feedback);
   if (clamp)
      assert(min_lods == 1);
   nir_validate_shader(shader, "DXBC sparse test");
   ralloc_free(shader);
   free(container);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   for (unsigned op = 219; op <= 233; op++) {
      if (op == 226 || op == 227)
         continue; /* Raw/structured buffer feedback remains refused. */
      check(op, true, false, false);
      check(op, false, false, false);
      check(op, false, true, false);
   }
   const unsigned buffer_ops[] = {223, 225, 226, 227};
   for (unsigned i = 0; i < 4; i++) {
      check(buffer_ops[i], true, false, true);
      check(buffer_ops[i], false, false, true);
      check(buffer_ops[i], false, true, true);
   }
   glsl_type_singleton_decref();
   puts("d12_dxbc_sparse_tests: PASS");
   return 0;
}
