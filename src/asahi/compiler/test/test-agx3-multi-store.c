/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Multiple stores in one AGX3 shader.
 *
 * A single store encodes in every addressing form, but two stores together
 * were refused, which blocked every real compute or fragment workload. This
 * pins the working cases so the limit cannot come back.
 */

#include "agx3_iadd_core.h"
#include "agx3_select_core.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* AGX3b stdev is 14 bytes. */
#define AGX3_STDEV_LEN 14u

static unsigned checks, failures;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);          \
         ++failures;                                                           \
      }                                                                        \
   } while (0)

/* n global stores through one base pointer, at distinct dword offsets. */
static bool
compile_n_global_stores(enum agx_isa_encoding encoding, unsigned n)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 multi-store");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);

   for (unsigned i = 0; i < n; ++i) {
      nir_def *addr =
         nir_iadd(&b, base,
                  nir_u2u64(&b, nir_imul_imm(&b, index, sizeof(uint32_t))));
      if (i)
         addr = nir_iadd_imm(&b, addr, i * sizeof(uint32_t));

      nir_store_global(&b, nir_iadd_imm(&b, index, i), addr, .align_mul = 4);
   }

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   key.reserved_preamble = 4 * (1 + 1);
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   key.no_preamble = true;

   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/*
 * Two stores through two independent base pointers. NIR vectorises adjacent
 * stores through one pointer into a single wide store, so that shape does not
 * actually exercise multiple stores; this one cannot be merged.
 */
static bool
compile_two_buffers(enum agx_isa_encoding encoding)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 two-buffer store");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *a = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *c = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *byte =
      nir_u2u64(&b, nir_imul_imm(&b, index, sizeof(uint32_t)));

   nir_store_global(&b, index, nir_iadd(&b, a, byte), .align_mul = 4);
   nir_store_global(&b, nir_iadd_imm(&b, index, 7), nir_iadd(&b, c, byte),
                    .align_mul = 4);

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   key.reserved_preamble = 4 * (1 + 1);
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   key.no_preamble = true;

   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/*
 * The constant-addend form must not merely be accepted -- it has to encode as
 * Apple's own iadd/imm8. This vector is lifted from the hash-pinned corpus
 * specimens in test-agx3-pack.c (dest 0, src 2, immediate 1).
 *
 * The bug this guards against was worse than a refusal: an immediate reached
 * the wait-carrying composer, which deposits sources into register fields, so
 * `iadd d, r0, #1` encoded as `iadd d, r0, r1` and silently read a register.
 */
static void
check_immediate_iadd_bytes(void)
{
   static const uint8_t expected[] = {0x9f, 0x11, 0x54, 0x00, 0x02,
                                      0x02, 0x08, 0x88, 0x11, 0x04};
   struct agx3_sel_instr I;
   memset(&I, 0, sizeof(I));
   I.op = AGX3_SEL_OP_IADD;
   I.has_dest = true;
   I.dest = 0;
   I.dest_size = AGX3_SEL_SIZE_32;
   I.nr_srcs = 2;
   I.src[0].reg = 2;
   I.src[0].size = AGX3_SEL_SIZE_32;
   I.src[0].is_reg = true;
   I.src[1].reg = 1;      /* the immediate value, not a register index */
   I.src[1].is_reg = false;

   uint8_t out[AGX3_SEL_MAX_LENGTH];
   uint8_t len = 0;
   CHECK(agx3_select_encode(AGX_ISA_ENCODING_AGX3B, &I, out, &len));
   CHECK(len == sizeof(expected));
   CHECK(len == sizeof(expected) && memcmp(out, expected, len) == 0);

   /* An immediate wider than the witnessed byte form stays refused. */
   I.src[1].reg = 256;
   CHECK(!agx3_select_encode(AGX_ISA_ENCODING_AGX3B, &I, out, &len));
}

/*
 * An index-derived SSBO store, built the way the Vulkan path actually reaches
 * the backend.
 *
 * This shape is why a green unit-test suite was not evidence that real
 * shaders compile. Using nir_load_global_invocation_id() directly leaves the
 * backend emitting a `get_sr`, and the address arithmetic collapses to a plain
 * iadd. A SPIR-V compute shader instead arrives with the id already expanded
 * to `workgroup_id * workgroup_size + local_id`, which becomes an `imad` with
 * a constant multiplier -- a different opcode, encoded by different code.
 *
 * Spelling the expansion out here covers the shape end to end. Without it,
 * `values[gid] = k + gid` compiled in every unit test and failed in every real
 * dispatch.
 */
static bool
compile_index_derived_store(enum agx_isa_encoding encoding,
                            uint8_t *out_store, bool *out_found,
                            uint8_t *out_add, bool *out_add_found)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 index-derived store");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);

   /* gid = workgroup_id.x * 32 + local_id.x, as the Vulkan path delivers it */
   nir_def *wg = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *gid = nir_iadd(&b, nir_imul_imm(&b, wg, 32), lid);

   nir_def *addr =
      nir_iadd(&b, base,
               nir_u2u64(&b, nir_imul_imm(&b, gid, sizeof(uint32_t))));
   nir_store_global(&b, nir_iadd_imm(&b, gid, 0x12340000), addr,
                    .align_mul = 4);

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   key.reserved_preamble = 4 * (1 + 1);
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   key.no_preamble = true;

   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   /*
    * Capture the emitted stdev so the caller can assert its addressing.
    * A register index is scaled ONLY by the instruction's shift -- the
    * hardware does not also scale by the access format -- so a uint[] store
    * must carry shift 2. Emitting shift 0 here addressed base + i BYTES,
    * folding 32 lanes onto 8 elements on real hardware.
    */
   if (ok && out_store != NULL) {
      const uint8_t *bin = part.binary;
      for (unsigned i = 0; i + AGX3_STDEV_LEN <= part.info.binary_size; ++i) {
         if (bin[i] != 0xe7)
            continue;
         memcpy(out_store, bin + i, AGX3_STDEV_LEN);
         *out_found = true;
         break;
      }
   }

   /*
    * Also pin the `K + gid` add -- the instruction that feeds the store, so
    * take the ten bytes immediately ahead of the stdev rather than hunting
    * for a byte pattern. This is the shape the compute probes run, and it is
    * where a codegen change silently altered the encoding of an instruction
    * that already worked: re-routing an unshifted register add onto a newly
    * grounded path changed two liveness bits and cost two probes on
    * hardware, while every host-side test stayed green because they only
    * check that shaders COMPILE. Byte-pinning an instruction the hardware is
    * known to accept is what catches that class.
    */
   if (ok && out_add != NULL) {
      const uint8_t *bin = part.binary;
      for (unsigned i = 10; i + AGX3_STDEV_LEN <= part.info.binary_size; ++i) {
         if (bin[i] != 0xe7 || bin[i - 10] != 0x9f)
            continue;
         memcpy(out_add, bin + i - 10, 10);
         *out_add_found = true;
         break;
      }
   }
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/*
 * A CONSTANT element offset must be scaled by the access format too. The
 * index counts elements and agx_nir_lower_address() has already subtracted
 * the format shift, so a byte offset must be re-scaled by log2(element
 * bytes) before it is encoded. Emitting the raw element number addressed
 * `base + 9` instead of `base + 36`; every probe whose only immediate index
 * was 0 hid it, and it corrupted the constant-data (rodata) pointer.
 *
 * Returns the emitted lddev so the caller can assert the offset.
 */
static bool
compile_constant_offset_load(enum agx_isa_encoding encoding, uint8_t *out_load,
                             bool *out_found)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 constant offset load");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);

   /* Same gid shape the Vulkan path delivers, so the index folds into an
    * imad instead of leaving a standalone widening move. */
   nir_def *wg = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *gid = nir_iadd(&b, nir_imul_imm(&b, wg, 32), lid);

   /* base[9], a 32-bit element -- byte offset 36. */
   nir_def *v = nir_load_global_constant(
      &b, 1, 32, nir_iadd_imm(&b, base, 9 * sizeof(uint32_t)),
      .align_mul = 4);

   nir_def *addr =
      nir_iadd(&b, base, nir_u2u64(&b, nir_imul_imm(&b, gid, sizeof(uint32_t))));
   nir_store_global(&b, v, addr, .align_mul = 4);

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   key.reserved_preamble = 4 * (1 + 1);
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   key.no_preamble = true;

   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   if (ok && out_load != NULL) {
      const uint8_t *bin = part.binary;
      for (unsigned i = 0; i + AGX3_STDEV_LEN <= part.info.binary_size; ++i) {
         if (bin[i] != 0x67)
            continue;
         memcpy(out_load, bin + i, AGX3_STDEV_LEN);
         *out_found = true;
         break;
      }
   }
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/*
 * A load whose result is consumed ONLY inside a branch -- the shape of every
 * bounds-checked read, and of the atomic lowering before it was disabled on
 * AGX3.
 *
 * The load is still in flight at the divergence, so its wait would land in
 * one arm only, and agx3_slot_assign()'s linear "waited once = satisfied
 * globally" rule would then let the other arm read it unwaited. The packer
 * used to refuse this outright; it now retires the scoreboard ahead of the
 * boundary so every path enters with an empty one.
 */
static bool
compile_load_in_branch(enum agx_isa_encoding encoding)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 load consumed in branch");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *wg = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *gid = nir_iadd(&b, nir_imul_imm(&b, wg, 32), lid);

   nir_def *addr =
      nir_iadd(&b, base, nir_u2u64(&b, nir_imul_imm(&b, gid, sizeof(uint32_t))));
   nir_def *v = nir_load_global(&b, 1, 32, addr, .align_mul = 4);

   nir_push_if(&b, nir_ult_imm(&b, gid, 16));
   nir_store_global(&b, nir_iadd_imm(&b, v, 1), addr, .align_mul = 4);
   nir_pop_if(&b, NULL);

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   key.reserved_preamble = 4 * (1 + 1);
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   key.no_preamble = true;

   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/*
 * An atomic at a UNIFORM address. nir_opt_uniform_atomics() would rewrite this
 * into "one elected lane performs the whole subgroup's operation", which needs
 * ACTIVE_THREAD_INDEX_IN_SUBGROUP and TOTAL_ACTIVE_THREADS_IN_SUBGROUP (AGX3
 * SR 34/36, witnessed 2026-09-16). That pass is still skipped on AGX3 pending
 * a CTS campaign, so this must compile as a plain per-lane atomic.
 */
static bool
compile_uniform_atomic(enum agx_isa_encoding encoding)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 uniform atomic");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);

   /* Same address in every lane -- what triggers the optimization. */
   nir_global_atomic(&b, 32, base, nir_imm_int(&b, 1),
                     .atomic_op = nir_atomic_op_iadd);

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   key.reserved_preamble = 4 * (1 + 1);
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   key.no_preamble = true;

   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/*
 * `imad dst, srcA, #imm, srcC` must encode as Apple's own bytes. This vector
 * is `o[i] = a[i] * 255u + b[i]` compiled by metal-nt; it is byte-identical
 * for applegpu_g16g and applegpu_g17p.
 *
 * Both register operands are DEAD after the multiply-add in that kernel, and
 * Apple's instruction says so: f27 and f57 (slot A's and slot C's last-use
 * bits) are set, which is the 0x2b in byte 9. Leaving them out of the
 * descriptor encoded 0x21 and made this vector unreachable -- the operand
 * hints are liveness, not constants, so a witness has to state them.
 */
static void
check_imad_bytes(void)
{
   static const uint8_t expected[] = {0x9f, 0x00, 0x56, 0x00, 0x02, 0x00,
                                      0xfc, 0x23, 0x50, 0x2b, 0x22, 0x00};
   const struct agx3_imad_desc d = {
      .dest = 0, .src_a = 0, .src_a_kill = true, .multiplier = 255,
      .src_c = 4, .src_c_kill = true, .wait_mask = 1u << 5,
   };
   uint8_t out[AGX3_IADD_MAX_LENGTH];
   uint8_t len = 0;
   CHECK(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   CHECK(len == sizeof(expected));
   CHECK(len == sizeof(expected) && memcmp(out, expected, len) == 0);
}

int
main(void)
{
   check_immediate_iadd_bytes();
   check_imad_bytes();

   /*
    * The index-derived store must not only compile, it must ADDRESS
    * correctly. Golden bytes for `values[wg*32 + lid] = 0x12340000 + gid`
    * on AGX3b; byte 12 carries the index shift, and 0x11 here is shift 2
    * (a uint element stride). With the format shift dropped this byte was
    * 0x10 -- shift 0 -- and every fourth lane collided on one element.
    */
   {
      uint8_t store[AGX3_STDEV_LEN];
      bool found = false;
      static const uint8_t want[AGX3_STDEV_LEN] = {
         0xe7, 0x00, 0x54, 0x02, 0x00, 0x00, 0x20,
         0x00, 0x11, 0x00, 0x00, 0x90, 0x11, 0x00};
      uint8_t add[10];
      bool add_found = false;
      /*
       * The `K + gid` add. With AGX_G17P_PROMOTE_INTEGER_CONSTANTS (default
       * on) the constant is promoted to uniform u8 and post-RA this shader is
       *
       *   r0 = imad ^r0, #32, ^r1, #0      <- the index
       *   r1 = iadd u8, r0                 <- THIS (0x12340000 in u8)
       *   device_store ^r1, u0:u1, ^r0.abs
       *
       * in the native uniform-first form (agx3_uniform_integer_encode()),
       * whose register operand carries its own cache (f40) and discard (f42)
       * hints. r0 must stay live, because the store addresses with it:
       * re-encoding these bytes gives dest r1, register r0 with cache = 0 and
       * discard = 0, uniform u8, no destination cache; discard = 1 encodes
       * differently. Byte 2 is 0x54 -- no scoreboard wait -- because the only
       * async producers here are the two get_sr reads, and the imad above
       * already waited them.
       *
       * With the promotion off this is the register form
       * `r1 = iadd r0, ^r1` after `r1 = mov_imm #0x12340000`, pinned as
       * 9f 01 54 02 02 00 08 a8 15 05 (agx3_iadd_encode()'s kill flags
       * 0x11/0x13/0x15/0x17 in byte 8 for (A,B) = (0,0)/(1,0)/(0,1)/(1,1)).
       */
      static const uint8_t want_add[10] = {0x9f, 0x01, 0x54, 0x02, 0x02,
                                           0x10, 0x00, 0xa8, 0x13, 0x04};
      CHECK(compile_index_derived_store(AGX_ISA_ENCODING_AGX3B, store, &found,
                                        add, &add_found));
      CHECK(found);
      CHECK(add_found);
      if (add_found && memcmp(add, want_add, sizeof(want_add)) != 0) {
         fprintf(stderr, "index-derived add bytes:");
         for (unsigned i = 0; i < 10; ++i)
            fprintf(stderr, " %02x", add[i]);
         fprintf(stderr, "\n");
      }
      CHECK(add_found && memcmp(add, want_add, sizeof(want_add)) == 0);
      if (found && memcmp(store, want, sizeof(want)) != 0) {
         fprintf(stderr, "index-derived store bytes:");
         for (unsigned i = 0; i < AGX3_STDEV_LEN; ++i)
            fprintf(stderr, " %02x", store[i]);
         fprintf(stderr, "\n");
      }
      CHECK(found && memcmp(store, want, sizeof(want)) == 0);
   }

   /*
    * A constant element offset must carry the format scale. Golden bytes for
    * a 32-bit load of base[9]; the encoded byte offset is 36. Without the
    * format scale this instruction encoded offset 9, which is what corrupted
    * the constant-data pointer -- and it is invisible to every probe whose
    * only immediate index is 0.
    */
   {
      uint8_t load[AGX3_STDEV_LEN];
      bool found = false;
      static const uint8_t want[AGX3_STDEV_LEN] = {
         0x67, 0x00, 0x54, 0x02, 0x00, 0x00, 0x00,
         0x00, 0x91, 0x80, 0x04, 0x40, 0x02, 0x00};
      CHECK(compile_constant_offset_load(AGX_ISA_ENCODING_AGX3B, load, &found));
      CHECK(found);
      if (found && memcmp(load, want, sizeof(want)) != 0) {
         fprintf(stderr, "constant-offset lddev bytes:");
         for (unsigned i = 0; i < AGX3_STDEV_LEN; ++i)
            fprintf(stderr, " %02x", load[i]);
         fprintf(stderr, "\n");
      }
      CHECK(found && memcmp(load, want, sizeof(want)) == 0);
   }

   /* A load consumed only inside a branch must still compile. */
   CHECK(compile_load_in_branch(AGX_ISA_ENCODING_AGX3B));

   /* Atomics: uniform address (the optimized shape) must compile. */
   CHECK(compile_uniform_atomic(AGX_ISA_ENCODING_AGX3B));

   CHECK(compile_two_buffers(AGX_ISA_ENCODING_AGX3B));

   for (unsigned n = 1; n <= 4; ++n) {
      const bool ok = compile_n_global_stores(AGX_ISA_ENCODING_AGX3B, n);
      fprintf(stderr, "%u global store(s): %s\n", n, ok ? "ok" : "REFUSED");
      CHECK(ok);
   }

   printf("%u checks, %u failures\n", checks, failures);
   return failures ? 1 : 0;
}
