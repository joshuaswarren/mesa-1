/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Multiple SSBO accesses and device atomics in one AGX3 shader.
 *
 * src/asahi/isa/README-agx3-packer-limits.md recorded a measured limit of ONE
 * SSBO access per shader, with atomics unencodable in any position. Every row
 * of that matrix is exercised here at the level the limit was stated -- does
 * agx_compile_shader_nir() accept the shader -- so the limit cannot come back
 * silently, and so the README's matrix has an in-tree owner.
 *
 * The store encoding itself is pinned against Apple's own two-store kernel,
 * because "two stores compile" is not the same claim as "two stores encode
 * the way the hardware expects".
 */

#include "agx3_mem_core.h"
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

static const uint8_t agx3_stop[] = {0x0e, 0x00, 0x00, 0x00};

/*
 * Count device stores. A stdev is 14 bytes led by 0xe7, whose byte 2 is the
 * top of the scoreboard wait mask (0x54/0x55/0x56) and whose byte 11 is a
 * fixed 0x90 in every witnessed 32-bit form, so this is specific enough not
 * to match the interior of another instruction.
 */
static unsigned
count_stores(const struct agx_shader_part *part)
{
   const uint8_t *b = part->binary;
   unsigned n = 0;
   /* Only the code: promoted constants precede main (see run()). */
   for (unsigned i = part->info.main_offset;
        i + AGX3_STDEV_LEN <= part->info.binary_size; ++i) {
      if (b[i] == 0xe7 && (b[i + 2] & 0xfc) == 0x54 && b[i + 11] == 0x90)
         ++n;
   }
   return n;
}

static bool
compile_shape(nir_builder *b, struct agx_shader_part *part)
{
   agx_preprocess_nir(b->shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.reserved_preamble = 8;
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   key.no_preamble = true;

   return agx_compile_shader_nir(b->shader, &key, part);
}

static nir_builder
start(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static nir_builder
start_graphics(enum mesa_shader_stage stage, const char *name)
{
   nir_builder b =
      nir_builder_init_simple_shader(stage, &agx_nir_options, "%s", name);
   return b;
}

/* The global invocation id in the shape the Vulkan path delivers it. */
static nir_def *
gid(nir_builder *b)
{
   nir_def *wg = nir_channel(b, nir_load_workgroup_id(b), 0);
   nir_def *lid = nir_channel(b, nir_load_local_invocation_id(b), 0);
   return nir_iadd(b, nir_imul_imm(b, wg, 32), lid);
}

/*
 * Run one shape. `want_stores` is the number of device stores the binary must
 * contain, or 0 to skip the count (atomics are not stores).
 */
static void
run(const char *name, void (*body)(nir_builder *b, nir_def *base, nir_def *g),
    unsigned want_stores)
{
   nir_builder b = start(name);
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   body(&b, base, gid(&b));

   struct agx_shader_part part = {0};
   const bool ok = compile_shape(&b, &part);
   ++checks;
   if (!ok) {
      fprintf(stderr, "FAIL %s: refused by agx_compile_shader_nir\n", name);
      ++failures;
      ralloc_free(b.shader);
      return;
   }

   CHECK(part.binary != NULL && part.info.binary_size > sizeof(agx3_stop));
   /* Integer constants the native uniform forms can read are promoted to
    * uniforms (AGX_G17P_PROMOTE_INTEGER_CONSTANTS, default on); their data is
    * uploaded before the code, so main starts after it. Main still runs to
    * the end of the binary. */
   CHECK(part.info.rodata.size_16
            ? part.info.rodata.offset + 2u * part.info.rodata.size_16 <=
                 part.info.main_offset
            : part.info.main_offset == 0);
   CHECK(part.info.main_offset + part.info.main_size == part.info.binary_size);
   CHECK(part.info.binary_size >= sizeof(agx3_stop) &&
         memcmp((const uint8_t *)part.binary + part.info.binary_size -
                   sizeof(agx3_stop),
                agx3_stop, sizeof(agx3_stop)) == 0);

   if (want_stores) {
      const unsigned got = count_stores(&part);
      ++checks;
      if (got != want_stores) {
         fprintf(stderr, "FAIL %s: %u device stores, expected %u\n", name, got,
                 want_stores);
         ++failures;
      }
   }

   free(part.binary);
   ralloc_free(b.shader);
}

/* Graphics stages use the same native device-memory instructions, but travel
 * through different backend epilogues and pipeline keys.  Compile one
 * returned atomic plus a following store in each advertised stage so a
 * compute-only success cannot accidentally qualify the Vulkan feature bits. */
static void
run_graphics_stage(enum mesa_shader_stage stage, const char *name)
{
   nir_builder b = start_graphics(stage, name);
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *old = nir_global_atomic(&b, 32, base, nir_imm_int(&b, 1),
                                    .atomic_op = nir_atomic_op_iadd);
   nir_store_global(&b, old, nir_iadd_imm(&b, base, 4), .align_mul = 4);

   if (stage == MESA_SHADER_VERTEX) {
      nir_def *pos[4] = {
         nir_imm_float(&b, 0.0f), nir_imm_float(&b, 0.0f),
         nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f),
      };
      for (unsigned c = 0; c < 4; ++c)
         nir_store_uvs_agx(&b, pos[c], nir_imm_intN_t(&b, c, 16));
      b.shader->info.outputs_written = VARYING_BIT_POS;
   }

   struct agx_shader_part part = {0};
   const bool ok = compile_shape(&b, &part);
   if (!ok) {
      ++checks;
      ++failures;
      fprintf(stderr, "FAIL %s: refused by agx_compile_shader_nir\n", name);
   } else {
      ++checks;
   }
   if (ok) {
      CHECK(part.binary != NULL && part.info.binary_size > sizeof(agx3_stop));
      CHECK(count_stores(&part) == 1);
      CHECK(memcmp((const uint8_t *)part.binary + part.info.binary_size -
                      sizeof(agx3_stop),
                   agx3_stop, sizeof(agx3_stop)) == 0);
   }

   free(part.binary);
   ralloc_free(b.shader);
}

static nir_def *
elem_addr(nir_builder *b, nir_def *base, nir_def *index)
{
   return nir_iadd(b, base, nir_u2u64(b, nir_imul_imm(b, index, 4)));
}

/* README row: `values[0] = A;` and `values[32u + g] = B;` */
static void
two_stores_const_and_indexed(nir_builder *b, nir_def *base, nir_def *g)
{
   nir_store_global(b, nir_iadd_imm(b, g, 0xaaaa0000),
                    elem_addr(b, base, nir_imm_int(b, 0)), .align_mul = 4);
   nir_store_global(b, nir_iadd_imm(b, g, 0xbbbb0000),
                    elem_addr(b, base, nir_iadd_imm(b, g, 32)), .align_mul = 4);
}

/* README row: `values[w] = A;` and `values[32u + g] = B;` */
static void
two_stores_wg_and_indexed(nir_builder *b, nir_def *base, nir_def *g)
{
   nir_def *wg = nir_channel(b, nir_load_workgroup_id(b), 0);
   nir_store_global(b, nir_iadd_imm(b, g, 0xaaaa0000), elem_addr(b, base, wg),
                    .align_mul = 4);
   nir_store_global(b, nir_iadd_imm(b, g, 0xbbbb0000),
                    elem_addr(b, base, nir_iadd_imm(b, g, 32)), .align_mul = 4);
}

/* README row: `x = values[31u - g];` then `values[g] = K + x;` */
static void
load_then_store(nir_builder *b, nir_def *base, nir_def *g)
{
   nir_def *x = nir_load_global(
      b, 1, 32, elem_addr(b, base, nir_isub_imm(b, 31, g)), .align_mul = 4);
   nir_store_global(b, nir_iadd_imm(b, x, 0x12340000), elem_addr(b, base, g),
                    .align_mul = 4);
}

/* Two loads feeding one store: neither README row, but the same class. */
static void
two_loads_one_store(nir_builder *b, nir_def *base, nir_def *g)
{
   nir_def *x = nir_load_global(b, 1, 32, elem_addr(b, base, g), .align_mul = 4);
   nir_def *y = nir_load_global(
      b, 1, 32, elem_addr(b, base, nir_isub_imm(b, 63, g)), .align_mul = 4);
   nir_store_global(b, nir_iadd(b, x, y),
                    elem_addr(b, base, nir_iadd_imm(b, g, 128)), .align_mul = 4);
}

/* Read-modify-write of one element: a load and a store at the same address. */
static void
load_add_store_same(nir_builder *b, nir_def *base, nir_def *g)
{
   nir_def *addr = elem_addr(b, base, g);
   nir_def *x = nir_load_global(b, 1, 32, addr, .align_mul = 4);
   nir_store_global(b, nir_iadd_imm(b, x, 1), addr, .align_mul = 4);
}

static void
four_stores(nir_builder *b, nir_def *base, nir_def *g)
{
   for (unsigned i = 0; i < 4; ++i) {
      nir_store_global(b, nir_iadd_imm(b, g, (int)(i + 1)),
                       elem_addr(b, base, nir_iadd_imm(b, g, (int)(64 * i))),
                       .align_mul = 4);
   }
}

/* README row: any atomicAdd / atomicMax / atomicOr on the SSBO. */
static void
atomic_add_uniform(nir_builder *b, nir_def *base, nir_def *g)
{
   (void)g;
   nir_global_atomic(b, 32, base, nir_imm_int(b, 1),
                     .atomic_op = nir_atomic_op_iadd);
}

static void
atomic_add_indexed(nir_builder *b, nir_def *base, nir_def *g)
{
   nir_global_atomic(b, 32, elem_addr(b, base, g), nir_imm_int(b, 1),
                     .atomic_op = nir_atomic_op_iadd);
}

static void
atomic_umax(nir_builder *b, nir_def *base, nir_def *g)
{
   nir_global_atomic(b, 32, base, g, .atomic_op = nir_atomic_op_umax);
}

static void
atomic_or(nir_builder *b, nir_def *base, nir_def *g)
{
   nir_global_atomic(b, 32, base, g, .atomic_op = nir_atomic_op_ior);
}

static void
atomic_and(nir_builder *b, nir_def *base, nir_def *g)
{
   nir_global_atomic(b, 32, base, g, .atomic_op = nir_atomic_op_iand);
}

static void
atomic_xchg(nir_builder *b, nir_def *base, nir_def *g)
{
   nir_global_atomic(b, 32, base, g, .atomic_op = nir_atomic_op_xchg);
}

/* An atomic whose RESULT is used, plus a store: two accesses and a return. */
static void
atomic_return_store(nir_builder *b, nir_def *base, nir_def *g)
{
   nir_def *old = nir_global_atomic(b, 32, base, nir_imm_int(b, 1),
                                    .atomic_op = nir_atomic_op_iadd);
   nir_store_global(b, old, elem_addr(b, base, nir_iadd_imm(b, g, 1)),
                    .align_mul = 4);
}

static void
check_atomic_return_spectpr_metadata(enum mesa_shader_stage stage)
{
   struct agx_shader_part returned = {0}, discarded = {0};

   nir_builder a = stage == MESA_SHADER_COMPUTE
      ? start("returned-atomic-spectpr")
      : start_graphics(stage, "returned-atomic-spectpr");
   nir_def *a_base = nir_load_preamble(&a, 1, 64, .base = 0);
   nir_def *old = nir_global_atomic(&a, 32, a_base, nir_imm_int(&a, 1),
                                    .atomic_op = nir_atomic_op_iadd);
   nir_store_global(&a, old, nir_iadd_imm(&a, a_base, 4), .align_mul = 4);
   CHECK(compile_shape(&a, &returned));

   nir_builder b = stage == MESA_SHADER_COMPUTE
      ? start("discarded-atomic-spectpr")
      : start_graphics(stage, "discarded-atomic-spectpr");
   nir_def *b_base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_global_atomic(&b, 32, b_base, nir_imm_int(&b, 1),
                     .atomic_op = nir_atomic_op_iadd);
   CHECK(compile_shape(&b, &discarded));

   CHECK(returned.info.scratch_size == 0);
   CHECK(discarded.info.scratch_size == 0);
   CHECK(returned.info.agx3_spectpr_scratch_size == 16);
   CHECK(discarded.info.agx3_spectpr_scratch_size == 0);

   free(returned.binary);
   free(discarded.binary);
   ralloc_free(a.shader);
   ralloc_free(b.shader);
}

/*
 * Apple's own two consecutive device stores, from
 * `out[gid] = A; out[gid + 64] = B;` compiled by metal-nt for
 * applegpu_g17p:
 *
 *   e7 00 54 04 00 03 21 00 11 00 00 90 11 00
 *   e7 00 54 00 00 01 21 00 11 00 00 90 11 00
 *
 * Byte 1 is zero and byte 2 is the bare 0x54 in BOTH: a store signals no
 * scoreboard slot and a second store does not wait on the first. That is the
 * fact the "one SSBO access per shader" limit contradicted.
 */
static void
check_apple_two_store_bytes(void)
{
   static const uint8_t first[] = {0xe7, 0x00, 0x54, 0x04, 0x00, 0x03, 0x21,
                                   0x00, 0x11, 0x00, 0x00, 0x90, 0x11, 0x00};
   static const uint8_t second[] = {0xe7, 0x00, 0x54, 0x00, 0x00, 0x01, 0x21,
                                    0x00, 0x11, 0x00, 0x00, 0x90, 0x11, 0x00};
   struct agx3_mem_desc st = {0};
   st.width = AGX3_MEM_W32;
   st.mask = 1;
   st.shift = 2;
   st.base_is_uniform = true;
   st.base = 0;
   st.idx_kind = AGX3_MEM_IDX_REG;
   st.idx_kill = true;
   st.wait_mask = 0;

   uint8_t out[AGX3_MEM_MAX_LENGTH];
   uint8_t len = 0;

   st.data = 4;
   st.idx_reg = 6;
   CHECK(agx3_mem_encode_store(AGX_ISA_ENCODING_AGX3B, &st, out, &len));
   CHECK(len == sizeof(first) && memcmp(out, first, len) == 0);

   st.data = 0;
   st.idx_reg = 2;
   CHECK(agx3_mem_encode_store(AGX_ISA_ENCODING_AGX3B, &st, out, &len));
   CHECK(len == sizeof(second) && memcmp(out, second, len) == 0);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   check_apple_two_store_bytes();

   run("two-stores-const-and-indexed", two_stores_const_and_indexed, 2);
   run("two-stores-wg-and-indexed", two_stores_wg_and_indexed, 2);
   run("load-then-store", load_then_store, 1);
   run("two-loads-one-store", two_loads_one_store, 1);
   run("load-add-store-same", load_add_store_same, 1);
   run("four-stores", four_stores, 4);

   run("atomic-add-uniform", atomic_add_uniform, 0);
   run("atomic-add-indexed", atomic_add_indexed, 0);
   run("atomic-umax", atomic_umax, 0);
   run("atomic-or", atomic_or, 0);
   run("atomic-and", atomic_and, 0);
   run("atomic-xchg", atomic_xchg, 0);
   run("atomic-return-store", atomic_return_store, 1);

   check_atomic_return_spectpr_metadata(MESA_SHADER_COMPUTE);
   check_atomic_return_spectpr_metadata(MESA_SHADER_VERTEX);
   check_atomic_return_spectpr_metadata(MESA_SHADER_FRAGMENT);

   run_graphics_stage(MESA_SHADER_VERTEX, "vertex-atomic-return-store");
   run_graphics_stage(MESA_SHADER_FRAGMENT, "fragment-atomic-return-store");

   glsl_type_singleton_decref();

   printf("%u checks, %u failures\n", checks, failures);
   return failures ? 1 : 0;
}
