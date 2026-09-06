/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "agx_compiler.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool
check_empty_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t expected[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "empty AGX3 compute");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "empty AGX3 compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid = part.binary != NULL &&
                      part.info.binary_size == sizeof(expected) &&
                      part.info.main_offset == 0 &&
                      part.info.main_size == sizeof(expected) &&
                      part.info.stats.instrs == 1 &&
                      part.info.workgroup_size[0] == 1 &&
                      part.info.workgroup_size[1] == 1 &&
                      part.info.workgroup_size[2] == 1 &&
                      memcmp(part.binary, expected, sizeof(expected)) == 0;

   if (!valid)
      fprintf(stderr, "empty AGX3 compute shader metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_shared_store_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 shared store");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 4;
   nir_store_shared(&b, nir_imm_int(&b, 0x12345678), nir_imm_int(&b, 0),
                    .align_mul = 4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 shared-store compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs > 1 && part.info.local_size == 4 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   if (!valid)
      fprintf(stderr, "AGX3 shared-store metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_unlowered_printf_address_rejected(enum agx_isa_encoding encoding)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "unlowered printf address");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *printf_address = nir_load_printf_buffer_address(&b, 64);
   nir_store_global(&b, printf_address,
                    nir_imm_int64(&b, UINT64_C(0x2000000000)),
                    .align_mul = 8);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part;
   memset(&part, 0xa5, sizeof(part));

   if (agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 unlowered printf address was accepted\n");
      free(part.binary);
      ralloc_free(b.shader);
      return false;
   }

   const uint8_t zero[sizeof(part)] = {0};
   const bool cleared = memcmp(&part, zero, sizeof(part)) == 0;
   if (!cleared)
      fprintf(stderr, "AGX3 printf rejection did not clear compiler output\n");

   ralloc_free(b.shader);
   return cleared;
}

static bool
check_agx2_legacy_printf_address_accepted(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "legacy printf address");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *printf_address = nir_load_printf_buffer_address(&b, 64);
   nir_store_global(&b, printf_address,
                    nir_imm_int64(&b, UINT64_C(0x2000000000)),
                    .align_mul = 8);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   if (!compiled)
      fprintf(stderr, "AGX2 legacy printf address was rejected\n");

   free(part.binary);
   ralloc_free(b.shader);
   return compiled;
}

static bool
check_shared_rmw_compute(enum agx_isa_encoding encoding)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 shared RMW");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 32 * sizeof(uint32_t);

   nir_def *offset = nir_imul_imm(
      &b, nir_load_local_invocation_index(&b), sizeof(uint32_t));
   nir_def *value = nir_load_shared(&b, 1, 32, offset, .align_mul = 4);
   nir_store_shared(&b, nir_iadd_imm(&b, value, 1), offset, .align_mul = 4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   /*
    * This shader's address arithmetic is `iadd dst32, #0, zext(src16) << 2`.
    * It briefly regressed to a refusal during the immediate/shift audit:
    * before that it "compiled", but the constant was deposited into a
    * register field and the shift was dropped, so it computed a wrong address
    * every time. Both fields are now probe-witnessed and encoded
    * (agx3_iadd_core.h), so it compiles again -- correctly this time.
    */
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 shared-RMW compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > 4 &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs > 3 &&
      part.info.local_size == 32 * sizeof(uint32_t);

   if (!valid)
      fprintf(stderr, "AGX3 shared-RMW metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_local_id_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 local invocation ID");
   b.shader->info.workgroup_size[0] = 4;
   b.shader->info.workgroup_size[1] = 2;
   b.shader->info.workgroup_size[2] = 2;
   b.shader->info.shared_size = sizeof(uint32_t);

   nir_def *id = nir_load_local_invocation_id(&b);
   nir_def *sum = nir_iadd(
      &b, nir_iadd(&b, nir_channel(&b, id, 0), nir_channel(&b, id, 1)),
      nir_channel(&b, id, 2));
   nir_store_shared(&b, sum, nir_imm_int(&b, 0), .align_mul = 4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 local-ID compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 7 && part.info.local_size == sizeof(uint32_t) &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   if (!valid)
      fprintf(stderr, "AGX3 local-ID metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_subgroup_id_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 subgroup IDs");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = sizeof(uint32_t);

   nir_def *sum = nir_iadd(&b, nir_load_subgroup_invocation(&b),
                           nir_load_subgroup_id(&b));
   nir_store_shared(&b, sum, nir_imm_int(&b, 0), .align_mul = 4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 subgroup-ID compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 5 && part.info.local_size == sizeof(uint32_t) &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   if (!valid)
      fprintf(stderr, "AGX3 subgroup-ID metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_subgroup_multiply_compute(void)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P subgroup multiply");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input_address = nir_imm_int64(&b, 0x1000);
   nir_def *output_address = nir_imm_int64(&b, 0x2000);
   nir_def *value = nir_load_global(
      &b, 1, 32, input_address, .align_mul = sizeof(uint32_t));
   nir_def *product =
      nir_reduce(&b, value, .reduction_op = nir_op_imul);
   nir_store_global(&b, product, output_address,
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "G17P subgroup-multiply shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs > 5 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;
   if (!valid)
      fprintf(stderr, "G17P subgroup-multiply metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_device_ballot_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 device-load ballot");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input_address = nir_imm_int64(&b, 0x1000);
   nir_def *output_address = nir_imm_int64(&b, 0x2000);
   nir_def *value = nir_load_global(
      &b, 1, 32, input_address, .align_mul = sizeof(uint32_t));
   nir_def *mask = nir_ballot(&b, 1, 32, nir_ine_imm(&b, value, 0));
   nir_store_global(&b, mask, output_address,
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 device-load ballot shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 5 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;
   if (!valid)
      fprintf(stderr, "AGX3 device-load ballot metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_device_shuffle_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 device-load shuffle");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *value = nir_load_global(
      &b, 1, 32, nir_imm_int64(&b, 0x1000),
      .align_mul = sizeof(uint32_t));
   nir_def *lane = nir_iand_imm(
      &b, nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1004),
                          .align_mul = sizeof(uint32_t)),
      31);
   nir_def *shuffled = nir_read_invocation(&b, value, lane);
   nir_store_global(&b, shuffled, nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 device-load shuffle shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 7 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;
   if (!valid)
      fprintf(stderr, "AGX3 device-load shuffle metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_device_rmw_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 device RMW");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *address = nir_imm_int64(&b, 0x1000);
   nir_def *value =
      nir_load_global(&b, 1, 32, address, .align_mul = sizeof(uint32_t));
   nir_store_global(&b, nir_iadd_imm(&b, value, 1), address,
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 device-RMW compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 4 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   if (!valid)
      fprintf(stderr, "AGX3 device-RMW metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_device_atomic_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
   static const struct {
      nir_atomic_op op;
      const char *name;
   } cases[] = {
      {nir_atomic_op_iadd, "add"},   {nir_atomic_op_imin, "imin"},
      {nir_atomic_op_umin, "umin"}, {nir_atomic_op_imax, "imax"},
      {nir_atomic_op_umax, "umax"}, {nir_atomic_op_xchg, "xchg"},
      {nir_atomic_op_cmpxchg, "cmpxchg"},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 device atomic family");
      b.shader->info.workgroup_size[0] = 1;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *address = nir_imm_int64(&b, 0x1000);
      nir_def *value_address = nir_imm_int64(&b, 0x2000);
      nir_def *value = nir_load_global(&b, 1, 32, value_address,
                                       .align_mul = sizeof(uint32_t));
      nir_def *old;
      if (cases[i].op == nir_atomic_op_cmpxchg) {
         nir_def *replacement = nir_load_global(
            &b, 1, 32,
            nir_iadd_imm(&b, value_address, sizeof(uint32_t)),
            .align_mul = sizeof(uint32_t));
         old = nir_global_atomic_swap(&b, 32, address, value, replacement,
                                      .atomic_op = nir_atomic_op_cmpxchg);
      } else {
         old = nir_global_atomic(&b, 32, address, value,
                                 .atomic_op = cases[i].op);
      }

      /* Keep both the returned-old value and the input live.  AGX3b must copy
       * a live scalar input to a private register before using its in-place
       * return form; cmpxchg must preserve [replacement, compare] ordering.
       */
      nir_store_global(&b, old, nir_imm_int64(&b, 0x3000),
                       .align_mul = sizeof(uint32_t));
      nir_store_global(&b, value, nir_imm_int64(&b, 0x3004),
                       .align_mul = sizeof(uint32_t));
      agx_preprocess_nir(b.shader);

      struct agx_shader_key key = {0};
      key.dev.encoding = encoding;
      struct agx_shader_part part = {0};

      if (!agx_compile_shader_nir(b.shader, &key, &part)) {
         fprintf(stderr, "AGX3 device atomic %s was rejected\n",
                 cases[i].name);
         ralloc_free(b.shader);
         return false;
      }

      const bool valid =
         part.binary != NULL && part.info.binary_size > sizeof(end) &&
         part.info.main_offset == 0 &&
         part.info.main_size == part.info.binary_size &&
         part.info.stats.instrs >= 6 &&
         memcmp((const uint8_t *)part.binary +
                   part.info.binary_size - sizeof(end),
                end, sizeof(end)) == 0;

      if (!valid)
         fprintf(stderr, "AGX3 device atomic %s metadata/binary mismatch\n",
                 cases[i].name);

      free(part.binary);
      ralloc_free(b.shader);
      if (!valid)
         return false;
   }

   return true;
}

static bool
check_shared_atomic_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 shared atomic");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = sizeof(uint32_t);

   nir_def *old = nir_shared_atomic(&b, 32, nir_imm_int(&b, 0),
                                    nir_imm_int(&b, 1),
                                    .atomic_op = nir_atomic_op_iadd);
   nir_store_shared(&b, old, nir_imm_int(&b, 0),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 shared-atomic compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 4 &&
      part.info.local_size == sizeof(uint32_t) &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   if (!valid)
      fprintf(stderr, "AGX3 shared-atomic metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/*
 * Mirrors src/asahi/vulkan/tests/g17p_vulkan_compute.comp -- the G17P
 * milestone probe shader -- in the shape the AGX3 backend actually receives
 * it, i.e. after Honeykrisp lowering:
 *
 *   - the deliberately-oversized 65-entry constant array becomes
 *     nir->constant_data plus a load_constant, and hk_nir_lower_descriptors.c
 *     (lower_load_constant, :66-93) rewrites that into a speculatable global
 *     load through the root descriptor's cs.constant_data_addr;
 *   - the SSBO store becomes a global store through a descriptor-set pointer;
 *   - hk_shader.c lower_uniforms (:1026-1057) turns load_root_agx and
 *     load_descriptor_set_agx into load_preamble uniform reads.
 *
 * No other AGX3 test compiles a compute shader carrying NIR constant data,
 * and hk_dispatch_g17p_native() has no fallback: if this shape does not
 * compile, or if the promoted-constant range that falls out of it is not
 * 32-bit aligned in both base and size, hk_g17p_compute_uniform_ranges()
 * (hk_cmd_dispatch.c:168-196) rejects the dispatch with
 * VK_ERROR_FEATURE_NOT_PRESENT.  Both facts are asserted here.
 */
static bool
check_constant_data_compute(enum agx_isa_encoding encoding, bool soft_fault)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   /* Byte offset of cs.constant_data_addr inside the Honeykrisp root
    * descriptor table.  The exact value is immaterial to the encoding; what
    * matters is that the address is loaded out of the root rather than being
    * an immediate.
    */
   const unsigned root_constant_data_offset = 16;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 constant data");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   /* Exactly the payload nir_opt_large_constants leaves behind for
    * g17p_vulkan_compute.comp's deltas[65].
    */
   uint32_t deltas[65];
   for (unsigned i = 0; i < ARRAY_SIZE(deltas); ++i)
      deltas[i] = 3 + i;

   b.shader->constant_data = ralloc_size(b.shader, sizeof(deltas));
   memcpy(b.shader->constant_data, deltas, sizeof(deltas));
   b.shader->constant_data_size = sizeof(deltas);

   nir_def *root = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *set0 = nir_load_preamble(&b, 1, 64, .base = 4);

   nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *offset = nir_imul_imm(&b, index, sizeof(uint32_t));

   nir_def *constant_data_addr = nir_load_global_constant(
      &b, 1, 64, nir_iadd_imm(&b, root, root_constant_data_offset),
      .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
   nir_def *delta = nir_load_global_constant(
      &b, 1, 32, nir_iadd(&b, constant_data_addr, nir_u2u64(&b, offset)),
      .align_mul = 4, .access = ACCESS_CAN_SPECULATE);

   nir_def *output = nir_load_global_constant(&b, 1, 64, set0, .align_mul = 8,
                                              .access = ACCESS_CAN_SPECULATE);
   nir_def *value =
      nir_iadd(&b, nir_iadd_imm(&b, index, 0x12340000), delta);

   nir_store_global(&b, value,
                    nir_iadd(&b, output, nir_u2u64(&b, offset)),
                    .align_mul = 4);

   agx_preprocess_nir(b.shader);

   /* Exactly the backend key hk_compile_shader() builds for a G17P compute
    * shader with one descriptor set (hk_shader.c:1220-1250): root == 0 for
    * compute, so reserved_preamble == 4 * (1 + set_count), and G17P forces
    * no_preamble because hk_g17p_compute_shader_supported() refuses a
    * preamble.  soft_fault is exercised both
    * ways: the kernel advertises DRM_ASAHI_FEATURE_SOFT_FAULTS whenever its
    * fault_control module parameter is 0xb, which is the default, but
    * AGX_MESA_DEBUG=nosoft and a hard-fault boot both clear it, and it changes
    * how the speculatable descriptor loads above are lowered.
    */
   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   key.dev.soft_fault = soft_fault;
   key.reserved_preamble = 4 * (1 + 1);
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   key.no_preamble = true;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr,
              "AGX3 constant-data compute shader was rejected "
              "(soft_fault %u)\n",
              soft_fault);
      ralloc_free(b.shader);
      return false;
   }

   /* main_offset is deliberately not required to be zero: when the backend
    * emits a rodata blob it is uploaded *before* the code and the shader body
    * is then padded up to AGX_CODE_ALIGN (agx_compile.c:3684-3697), so a
    * constant-carrying shader legitimately starts past the top of its binary.
    */
   bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      (uint64_t)part.info.main_offset + part.info.main_size <=
         part.info.binary_size &&
      part.info.main_size > sizeof(end) && part.info.stats.instrs >= 6 &&
      part.info.nr_gprs != 0 && part.info.local_size == 0 &&
      part.info.scratch_size == 0 && part.info.preamble_scratch_size == 0 &&
      !part.info.has_preamble && part.info.texture_state_count == 0 &&
      part.info.sampler_state_count == 0 && !part.info.uses_sampler_heap &&
      !part.info.uses_txf;

   if (!valid)
      fprintf(stderr,
              "AGX3 constant-data shader does not satisfy "
              "hk_g17p_compute_shader_supported(): binary_size %u "
              "main_offset %u main_size %u instrs %u local_size %u "
              "scratch %u preamble_scratch %u has_preamble %u nr_gprs %u "
              "push_count %u tex %u smp %u heap %u txf %u "
              "rodata{base %u size_16 %u offset %u}\n",
              part.info.binary_size, part.info.main_offset,
              part.info.main_size, part.info.stats.instrs,
              part.info.local_size, part.info.scratch_size,
              part.info.preamble_scratch_size, part.info.has_preamble,
              part.info.nr_gprs, part.info.push_count,
              part.info.texture_state_count, part.info.sampler_state_count,
              part.info.uses_sampler_heap, part.info.uses_txf,
              part.info.rodata.base_uniform, part.info.rodata.size_16,
              part.info.rodata.offset);

   /* The rodata blob, when the backend emits one, is appended after main, so
    * main_size no longer covers the whole binary.  Whatever the split, the
    * shader body must still terminate.
    */
   const struct agx_rodata *rodata = &part.info.rodata;
   if (valid && rodata->size_16 != 0) {
      if (rodata->offset > part.info.binary_size ||
          (uint64_t)rodata->size_16 * 2 >
             (uint64_t)part.info.binary_size - rodata->offset) {
         fprintf(stderr, "AGX3 constant-data rodata blob is out of bounds\n");
         valid = false;
      }

      /* hk_g17p_compute_uniform_ranges() refuses an odd base or size. */
      if (valid && ((rodata->base_uniform & 1) != 0 ||
                    (rodata->size_16 & 1) != 0 || rodata->size_16 > 512)) {
         fprintf(stderr,
                 "AGX3 constant-data rodata range (base %u, size_16 %u) is "
                 "not bindable by hk_g17p_compute_uniform_ranges()\n",
                 rodata->base_uniform, rodata->size_16);
         valid = false;
      }

      /* The rodata range must sit above the root/descriptor-set uniforms that
       * hk_agx3_g17p_root_range() already claims, or the two Ioto uniform
       * ranges would overlap.
       */
      if (valid && rodata->base_uniform < 4 * (1 + 1)) {
         fprintf(stderr,
                 "AGX3 constant-data rodata base %u collides with the root "
                 "uniform range\n",
                 rodata->base_uniform);
         valid = false;
      }
   }

   if (valid &&
       memcmp((const uint8_t *)part.binary + part.info.main_offset +
                 part.info.main_size - sizeof(end),
              end, sizeof(end)) != 0) {
      fprintf(stderr, "AGX3 constant-data shader is not terminated\n");
      valid = false;
   }

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/*
 * Fail-closed companion to check_constant_data_compute(): the same shader with
 * key.no_preamble cleared must be *refused*, not miscompiled.
 *
 * agx_nir_opt_preamble hoists the uniform-invariant root and descriptor-set
 * loads into a generated preamble, whose results are written back with
 * uniform_store -- and agx3_pack_binary has no grounded AGX3 encoding for that
 * opcode, so it takes the AGX3_REFUSE fallthrough.  hk_compile_shader() avoids
 * this by setting .no_preamble for AGX_CHIP_G17P (hk_shader.c:1249); that one
 * line is load-bearing for every G17P shader that reads a descriptor, so lock
 * the coupling down here.
 */
static bool
check_preamble_refused_compute(enum agx_isa_encoding encoding)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 preamble refusal");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *root = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *output = nir_load_global_constant(&b, 1, 64, root, .align_mul = 8,
                                              .access = ACCESS_CAN_SPECULATE);
   nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_store_global(
      &b, index,
      nir_iadd(&b, output,
               nir_u2u64(&b, nir_imul_imm(&b, index, sizeof(uint32_t)))),
      .align_mul = 4);

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   key.reserved_preamble = 4 * (1 + 1);
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   /* Deliberately NOT set: key.no_preamble. */
   struct agx_shader_part part;
   memset(&part, 0xa5, sizeof(part));

   if (agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr,
              "AGX3 accepted a generated preamble; uniform_store is now "
              "grounded, so hk_shader.c may stop forcing no_preamble on "
              "G17P -- update both together\n");
      free(part.binary);
      ralloc_free(b.shader);
      return false;
   }

   const uint8_t zero[sizeof(part)] = {0};
   if (memcmp(&part, zero, sizeof(part)) != 0) {
      fprintf(stderr, "AGX3 preamble rejection did not clear compiler output\n");
      ralloc_free(b.shader);
      return false;
   }

   ralloc_free(b.shader);
   return true;
}

/*
 * hk_nir_lower_descriptors.c lower_load_constant (:66-93), reproduced so this
 * test can run the same ordering Honeykrisp does without linking libhk.
 */
static bool
lower_load_constant_like_hk(nir_builder *b, nir_intrinsic_instr *intr,
                            UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_constant)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *root = nir_load_preamble(b, 1, 64, .base = 0);
   nir_def *blob = nir_load_global_constant(b, 1, 64, nir_iadd_imm(b, root, 16),
                                            .align_mul = 8,
                                            .access = ACCESS_CAN_SPECULATE);
   nir_def *offset =
      nir_iadd_imm(b, intr->src[0].ssa, nir_intrinsic_base(intr));
   nir_def *data_def = nir_load_global_constant(
      b, intr->def.num_components, intr->def.bit_size,
      nir_iadd(b, blob, nir_u2u64(b, offset)),
      .align_mul = nir_intrinsic_align_mul(intr),
      .access = ACCESS_CAN_SPECULATE);

   nir_def_rewrite_uses(&intr->def, data_def);
   return true;
}

/*
 * g17p_vulkan_compute.comp's 65-entry deltas[] in the shape glslang and
 * nir_lower_variable_initializers hand to the driver: a function_temp array
 * filled by constant stores in one block, then read at a dynamic index.
 * 65 * 4 == 260 bytes, deliberately just over the 256-byte threshold of the
 * nir_lower_vars_to_scratch call inside agx_preprocess_nir
 * (agx_compile.c:3852) -- which is the whole point of this test.
 */
static nir_builder
build_constant_array_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 constant array");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   const unsigned n = 65;
   nir_variable *var = nir_local_variable_create(
      b.impl, glsl_array_type(glsl_uint_type(), n, 4), "deltas");
   nir_deref_instr *array = nir_build_deref_var(&b, var);

   for (unsigned i = 0; i < n; ++i) {
      nir_store_deref(&b, nir_build_deref_array_imm(&b, array, i),
                      nir_imm_int(&b, 3 + i), 0x1);
   }

   nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *delta =
      nir_load_deref(&b, nir_build_deref_array(&b, array, index));
   nir_def *output = nir_load_global_constant(
      &b, 1, 64, nir_load_preamble(&b, 1, 64, .base = 4), .align_mul = 8,
      .access = ACCESS_CAN_SPECULATE);

   nir_store_global(
      &b, nir_iadd(&b, nir_iadd_imm(&b, index, 0x12340000), delta),
      nir_iadd(&b, output,
               nir_u2u64(&b, nir_imul_imm(&b, index, sizeof(uint32_t)))),
      .align_mul = 4);

   return b;
}

static void
g17p_backend_key(enum agx_isa_encoding encoding, struct agx_shader_key *key)
{
   *key = (struct agx_shader_key){0};
   key->dev.encoding = encoding;
   key->reserved_preamble = 4 * (1 + 1);
   key->has_scratch = true;
   key->promote_constants = true;
   key->promote_textures = true;
   key->no_preamble = true;
}

/*
 * The pass-ordering contract for G17P constant data, pinned in both
 * directions.
 *
 * Honeykrisp must run nir_opt_large_constants BEFORE agx_preprocess_nir
 * (hk_shader.c hk_preprocess_nir_internal).  Run it after and the array is
 * already scratch, nir_opt_large_constants has no function_temp variable left
 * to move, and the shader reaches the backend as stack_store/stack_load
 * traffic which AGX3 cannot encode and which
 * hk_g17p_compute_shader_supported() would refuse anyway.
 *
 * `ordered` reproduces the correct order and must compile with no scratch and
 * a non-empty constant blob.  `!ordered` reproduces the regression and must
 * FAIL CLOSED -- never silently produce a scratch-using shader.
 */
static bool
check_constant_array_ordering(enum agx_isa_encoding encoding, bool ordered)
{
   nir_builder b = build_constant_array_compute();

   if (ordered)
      NIR_PASS(_, b.shader, nir_opt_large_constants, NULL, 32);

   agx_preprocess_nir(b.shader);

   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_load_constant_like_hk, nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);

   const bool has_constants = b.shader->constant_data_size != 0;
   const bool has_scratch = b.shader->scratch_size != 0;

   struct agx_shader_key key;
   g17p_backend_key(encoding, &key);
   struct agx_shader_part part;
   memset(&part, 0xa5, sizeof(part));

   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);

   bool valid;
   if (ordered) {
      valid = has_constants && !has_scratch && compiled &&
              part.info.scratch_size == 0 &&
              part.info.preamble_scratch_size == 0 && !part.info.has_preamble &&
              part.binary != NULL && part.info.binary_size != 0;

      if (!valid) {
         fprintf(stderr,
                 "AGX3 constant array did not survive as constant data: "
                 "constant_data_size %u scratch %u compiled %d "
                 "info.scratch %u\n",
                 b.shader->constant_data_size, b.shader->scratch_size, compiled,
                 compiled ? part.info.scratch_size : 0);
      }
   } else {
      /* Without the pass the array must land in scratch... */
      valid = has_scratch && !has_constants;
      if (!valid) {
         fprintf(stderr,
                 "AGX3 constant array test is no longer exercising the "
                 "scratch path (constant_data_size %u scratch %u); retune the "
                 "array size against agx_compile.c's nir_lower_vars_to_scratch "
                 "threshold\n",
                 b.shader->constant_data_size, b.shader->scratch_size);
      }

      /* ...and the backend must refuse it rather than emit something. */
      if (valid && compiled) {
         fprintf(stderr,
                 "AGX3 accepted a scratch-using compute shader; if stack "
                 "access is now grounded, widen "
                 "hk_g17p_compute_shader_supported() in the same change\n");
         valid = false;
      }

      if (valid) {
         const uint8_t zero[sizeof(part)] = {0};
         if (memcmp(&part, zero, sizeof(part)) != 0) {
            fprintf(stderr, "AGX3 scratch rejection did not clear output\n");
            valid = false;
         }
      }
   }

   if (compiled)
      free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/*
 * AGX3 ALU operands are register-or-immediate only; the uniform file is
 * reachable only from the memory-op bases, the texture heap and the subgroup
 * sources.  agx_instr_accepts_uniform() is what keeps a promoted constant out
 * of an AGX3 ALU source, and getting it wrong costs a whole-shader refusal
 * with `no grounded AGX3 encoding for iadd`.
 */
static bool
check_mixed_fma_neg_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P mixed FMA fneg");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 4;

   nir_def *lane = nir_u2f32(&b, nir_load_local_invocation_index(&b));
   nir_def *half = nir_f2f16(&b, nir_fadd_imm(&b, lane, 0.5f));
   nir_def *half32 = nir_f2f32(&b, half);
   nir_def *result = nir_ffma(&b, nir_fneg(&b, lane), half32,
                              nir_imm_float(&b, 1.0f));
   nir_store_shared(&b, result, nir_imm_int(&b, 0),
                    .align_mul = 4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary != NULL &&
                      part.info.binary_size > 4 && part.info.stats.instrs >= 7;
   if (!valid)
      fprintf(stderr, "G17P mixed-source negative FMA lowering failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_add_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shaderFloat16 add");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input = nir_load_global(
      &b, 2, 32, nir_imm_int64(&b, 0x1000),
      .align_mul = sizeof(uint32_t));
   nir_def *left = nir_f2f16(&b, nir_channel(&b, input, 0));
   nir_def *right = nir_f2f16(&b, nir_channel(&b, input, 1));
   nir_def *sum = nir_fadd(&b, left, right);
   nir_store_global(&b, nir_f2f32(&b, sum), nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary != NULL &&
                      part.info.binary_size > 4 && part.info.stats.instrs >= 7;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 add lowering failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_uniform_alu_policy(void)
{
   const struct {
      enum agx_isa_encoding encoding;
      bool alu;
   } cases[] = {
      {AGX_ISA_ENCODING_AGX2, true},
      {AGX_ISA_ENCODING_AGX3B, false},
      {AGX_ISA_ENCODING_AGX3C, false},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      const enum agx_isa_encoding e = cases[i].encoding;

      if (agx_instr_accepts_uniform(e, AGX_OPCODE_IADD, 0, 4, AGX_SIZE_32) !=
             cases[i].alu ||
          agx_instr_accepts_uniform(e, AGX_OPCODE_IMAD, 1, 4, AGX_SIZE_32) !=
             cases[i].alu) {
         fprintf(stderr, "AGX ALU uniform policy wrong for encoding %u\n", e);
         return false;
      }

      /* 64-bit ALU never takes a uniform on any encoding. */
      if (agx_instr_accepts_uniform(e, AGX_OPCODE_IADD, 0, 4, AGX_SIZE_64)) {
         fprintf(stderr, "64-bit ALU uniform accepted for encoding %u\n", e);
         return false;
      }

      /* Memory bases keep their uniform, on every encoding: the AGX3 packer
       * depends on it (agx3_pack_binary.c:961 requires a uniform base). */
      if (!agx_instr_accepts_uniform(e, AGX_OPCODE_DEVICE_LOAD, 0, 4,
                                     AGX_SIZE_64) ||
          !agx_instr_accepts_uniform(e, AGX_OPCODE_DEVICE_STORE, 1, 4,
                                     AGX_SIZE_64)) {
         fprintf(stderr, "memory-base uniform refused for encoding %u\n", e);
         return false;
      }
   }

   return true;
}

int
main(int argc, char **argv)
{
   if (argc == 2 && !strcmp(argv[1], "--subgroup-multiply-only")) {
      const bool valid = check_subgroup_multiply_compute();
      fprintf(stderr, "G17P subgroup multiply compute: %s\n",
               valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }

   if (argc == 2 && !strcmp(argv[1], "--half-add-only")) {
      const bool valid = check_half_add_compute();
      fprintf(stderr, "G17P shaderFloat16 add: %s\n",
               valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }

   if (argc == 2 && !strcmp(argv[1], "--device-ballot-only")) {
      const bool valid =
         check_device_ballot_compute(AGX_ISA_ENCODING_AGX3B) &&
         check_device_ballot_compute(AGX_ISA_ENCODING_AGX3C);
      fprintf(stderr, "AGX3 device-load ballot compute: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--device-shuffle-only")) {
      const bool valid =
         check_device_shuffle_compute(AGX_ISA_ENCODING_AGX3B) &&
         check_device_shuffle_compute(AGX_ISA_ENCODING_AGX3C);
      fprintf(stderr, "AGX3 device-load shuffle compute: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }

   const enum agx_isa_encoding encodings[] = {
      AGX_ISA_ENCODING_AGX3B,
      AGX_ISA_ENCODING_AGX3C,
   };

   /* glsl_array_type() in the constant-array case needs the type singleton. */
   glsl_type_singleton_init_or_ref();

   if (argc == 2 && strcmp(argv[1], "--device-atomics-only") == 0) {
      for (unsigned i = 0; i < ARRAY_SIZE(encodings); ++i) {
         if (!check_device_atomic_compute(encodings[i]))
            return 1;
      }
      return 0;
   }
   if (argc != 1) {
      fprintf(stderr, "usage: %s [--device-atomics-only]\n", argv[0]);
      return 1;
   }

   if (!check_uniform_alu_policy())
      return 1;
   if (!check_agx2_legacy_printf_address_accepted())
      return 1;
   if (!check_mixed_fma_neg_compute())
      return 1;
   if (!check_subgroup_multiply_compute())
      return 1;
   if (!check_half_add_compute())
      return 1;

   for (unsigned i = 0; i < sizeof(encodings) / sizeof(encodings[0]); ++i) {
      if (!check_empty_compute(encodings[i]))
         return 1;
      if (!check_unlowered_printf_address_rejected(encodings[i]))
         return 1;
      if (!check_shared_store_compute(encodings[i]))
         return 1;
      if (!check_shared_rmw_compute(encodings[i]))
         return 1;
      if (!check_local_id_compute(encodings[i]))
         return 1;
      if (!check_subgroup_id_compute(encodings[i]))
         return 1;
      if (!check_device_ballot_compute(encodings[i]))
         return 1;
      if (!check_device_shuffle_compute(encodings[i]))
         return 1;
      if (!check_device_rmw_compute(encodings[i]))
         return 1;
      if (!check_device_atomic_compute(encodings[i]))
         return 1;
      if (!check_shared_atomic_compute(encodings[i]))
         return 1;
      if (!check_constant_data_compute(encodings[i], false) ||
          !check_constant_data_compute(encodings[i], true))
         return 1;
      if (!check_preamble_refused_compute(encodings[i]))
         return 1;
      if (!check_constant_array_ordering(encodings[i], true) ||
          !check_constant_array_ordering(encodings[i], false))
         return 1;

      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "unlowered ray query");
      b.shader->info.io_lowered = true;
      b.shader->info.ray_queries = 1;

      struct agx_shader_key key = {0};
      key.dev.encoding = encodings[i];
      struct agx_shader_part part;
      memset(&part, 0xa5, sizeof(part));

      if (agx_compile_shader_nir(b.shader, &key, &part)) {
         fprintf(stderr, "AGX3 unlowered ray query was accepted\n");
         ralloc_free(b.shader);
         return 1;
      }

      const uint8_t zero[sizeof(part)] = {0};
      if (memcmp(&part, zero, sizeof(part)) != 0) {
         fprintf(stderr, "AGX3 rejection did not clear compiler output\n");
         ralloc_free(b.shader);
         return 1;
      }

      ralloc_free(b.shader);
   }

   glsl_type_singleton_decref();

   puts("agx3_compile: compute SR/shared-memory/constant-data paths emitted; "
        "constant arrays kept out of scratch; unlowered ray queries and "
        "printf addresses rejected");
   return 0;
}
