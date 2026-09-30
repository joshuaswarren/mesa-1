/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "nir_builder.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int
main(void)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
   static const uint8_t outer_entry[] = {0x0f, 0x05, 0x54, 0x01};
   static const uint8_t loop_entry[] = {0x0f, 0x05, 0x54, 0x1a};
   static const uint8_t loop_exit[] = {0x0f, 0x06, 0x04, 0x02, 0x00, 0x00};
   static const uint8_t outer_exit[] = {0x0f, 0x06, 0x04, 0x01, 0x00, 0x00};
   static const uint8_t expected_binary[] = {
      0x1b, 0x04, 0x01, 0x08, 0x2b, 0x06, 0x01, 0x08,
      0x0a, 0x05, 0x23, 0x03, 0x02, 0x00, 0x05, 0x00, 0x00, 0x00,
      0x0f, 0x05, 0x54, 0x01,
      0x0f, 0x05, 0x54, 0x1a,
      0x1b, 0x04, 0x01, 0x08,
      0x67, 0x00, 0x54, 0x04, 0x00, 0x00, 0x00, 0x00,
      /* lddev f77 is the coherent flag (agx3_mem_core.c, KV(77, d->coherent)),
       * whose encoder LUT maps coherent=0 to raw 2 and coherent=1 to raw 4.
       * This byte was pinned as 0x40 (coherent=0) while the compiler still
       * dropped the bit -- i.e. the golden recorded a bug: the poll below is
       * built with ACCESS_COHERENT | ACCESS_VOLATILE, and a non-coherent load
       * may spin on a stale cached value forever. 1d7521a2e20 ("asahi: fix
       * G17P Vulkan memory model images") made it honour the access flags, so
       * the correct byte is 0x80 (coherent=1). */
      0x11, 0x00, 0x00, 0x80, 0x02, 0x00,
      0x0b, 0x01, 0x07, 0x01, 0x00, 0x20, 0x00, 0x80, 0x00, 0x00,
      0x0a, 0x05, 0x23, 0x03, 0x06, 0x00, 0x07, 0x00, 0x00, 0x00,
      0x8f, 0x04, 0x54, 0x22,
      0x0f, 0x00, 0x54, 0xd6, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00,
      0x0f, 0x06, 0x04, 0x02, 0x00, 0x00,
      0x0f, 0x06, 0x04, 0x01, 0x00, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };

   /* v53-c2: with AGX_G17P_UNIFORM_COMPARE (default on) both compares read
    * the preamble uniforms directly (Apple's reg/uniform icmp, f18/f24=3), so
    * the two uniform copies disappear. Same control structure, flag slot 0
    * and inverted CNDLT; decoded: copy h2<-u6, icmp ^h2<u4 (guard), cndst,
    * cndst, coherent lddev h2, slot-retire carrier, icmp h2==u4, cndlt inv,
    * b, cndend(2), cndend(1), end. */
   static const uint8_t expected_uniform_binary[] = {
      0x1b, 0x06, 0x01, 0x08,
      /* f13: the copied zero's last use (the loop's lddev reuses h2). */
      0x0a, 0x03, 0x2b, 0x05, 0x02, 0x00, 0x05, 0x02, 0x00, 0x00,
      0x0f, 0x05, 0x54, 0x01,
      0x0f, 0x05, 0x54, 0x1a,
      0x67, 0x00, 0x54, 0x02, 0x00, 0x00, 0x00, 0x00,
      0x11, 0x00, 0x00, 0x80, 0x02, 0x00,
      0x0b, 0x01, 0x07, 0x01, 0x00, 0x20, 0x00, 0x80, 0x00, 0x00,
      0x0a, 0x03, 0x23, 0x05, 0x06, 0x00, 0x07, 0x02, 0x00, 0x00,
      0x8f, 0x04, 0x54, 0x22,
      0x0f, 0x00, 0x54, 0xda, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00,
      0x0f, 0x06, 0x04, 0x02, 0x00, 0x00,
      0x0f, 0x06, 0x04, 0x01, 0x00, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   const bool uniform_compare =
      agx_get_alu_flags() & AGX_ALU_G17P_UNIFORM_COMPARE;
   const uint8_t *golden =
      uniform_compare ? expected_uniform_binary : expected_binary;
   const size_t golden_size = uniform_compare ? sizeof(expected_uniform_binary)
                                              : sizeof(expected_binary);
   /* Offsets of the loop compare and its CNDLT in each form. */
   const unsigned compare_at = uniform_compare ? 46 : 54;
   const unsigned cndlt_at = uniform_compare ? 56 : 64;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 event wait");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *address = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *expected = nir_load_preamble(&b, 1, 32, .base = 4);
   nir_def *zero = nir_load_preamble(&b, 1, 32, .base = 6);
   nir_if *guard = nir_push_if(&b, nir_ult(&b, zero, expected));
   nir_loop *loop = nir_push_loop(&b);
   nir_def *value = nir_load_global(
      &b, 1, 32, address, .align_mul = sizeof(uint32_t),
      .access = ACCESS_COHERENT | ACCESS_VOLATILE);
   nir_break_if(&b, nir_ieq(&b, value, expected));
   nir_pop_loop(&b, loop);
   nir_pop_if(&b, guard);

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.reserved_preamble = 8;
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   key.no_preamble = true;
   struct agx_shader_part part = {0};

   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid =
      compiled && part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 4 &&
      part.info.binary_size == golden_size &&
      memcmp(part.binary, golden, golden_size) == 0 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   /* BREAK_IF names the lanes that leave the loop, while CNDLT retains the
    * lanes selected by its predicate. The native nested-slot lowering must
    * therefore invert this CNDLT. This is the first loop, even though it is
    * inside an if, so it uses flag0. Reject both the old flag1 comparison and
    * the non-inverted CNDLT that keeps waiting after the event reaches SET. */
   static const uint8_t wrong_break_reg[] = {
      0x2a, 0x05, 0x23, 0x03, 0x06, 0x00, 0x07, 0x00, 0x00, 0x00,
   };
   static const uint8_t wrong_break_uniform[] = {
      0x2a, 0x03, 0x23, 0x05, 0x06, 0x00, 0x07, 0x02, 0x00, 0x00,
   };
   const uint8_t *wrong_break =
      uniform_compare ? wrong_break_uniform : wrong_break_reg;
   static const uint8_t wrong_cndlt[] = {0x8f, 0x04, 0x54, 0x02};
   const bool rejects_wrong_break =
      part.binary &&
      part.info.binary_size >= compare_at + sizeof(wrong_break_reg) &&
      memcmp((const uint8_t *)part.binary + compare_at, wrong_break,
             sizeof(wrong_break_reg)) != 0 &&
      part.info.binary_size >= cndlt_at + sizeof(wrong_cndlt) &&
      memcmp((const uint8_t *)part.binary + cndlt_at, wrong_cndlt,
             sizeof(wrong_cndlt)) != 0;

   unsigned control = 0;
   const uint8_t *need[] = {outer_entry, loop_entry, loop_exit, outer_exit};
   const size_t need_size[] = {sizeof(outer_entry), sizeof(loop_entry),
                               sizeof(loop_exit), sizeof(outer_exit)};
   for (unsigned i = 0; part.binary && i < part.info.binary_size && control < 4;
        ++i) {
      if (i + need_size[control] <= part.info.binary_size &&
          memcmp((const uint8_t *)part.binary + i, need[control],
                 need_size[control]) == 0) {
         ++control;
      }
   }

   if (!valid || !rejects_wrong_break || control != 4) {
      fprintf(stderr, "AGX3b event-wait poll was not encoded safely\n");
      for (unsigned i = 0; part.binary && i < part.info.binary_size; ++i)
         fprintf(stderr, "%02x", ((const uint8_t *)part.binary)[i]);
      fputc('\n', stderr);
   }

   free(part.binary);
   ralloc_free(b.shader);
   return valid && rejects_wrong_break && control == 4 ? 0 : 1;
}
