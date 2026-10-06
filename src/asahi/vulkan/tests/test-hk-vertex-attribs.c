/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT
 *
 * Vertex attribute count limits.
 *
 * DXVK's D3D9 fixed-function vertex shader reads 18 attributes (every entry
 * of its fixed-function input signature is an attribute, unbound ones on a
 * null stream). With AGX_MAX_ATTRIBS at 16, hk_compile_shader's per-attribute
 * key array was indexed past its end by the vertex fetch lowering (slot 16
 * read a garbage format/stride from the stack) and the input gather wrote
 * past its component bitset; Fallout: New Vegas crashed inside
 * vkCreateGraphicsPipelines on New Game. This checks the capacity, the
 * device limit derived from the prolog register ABI, and that a vertex
 * shader at that limit still fetches, exports and compiles on G17P.
 */
#include "asahi/compiler/agx_compile.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_linker.h"
#include "asahi/lib/agx_nir_lower_vbo.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x)                                                              \
   do {                                                                       \
      if (!(x)) {                                                             \
         fprintf(stderr, "ATTRIBS FAIL line %d: %s\n", __LINE__, #x);         \
         abort();                                                             \
      }                                                                       \
   } while (0)

/* Mirrors agx_compiler.h, which is private to the compiler. */
#define AGX2_HALVES  256
#define AGX3B_HALVES 192

static unsigned
count_intrinsics(nir_shader *s, nir_intrinsic_op op, unsigned *max_base)
{
   unsigned n = 0;
   *max_base = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != op)
               continue;
            ++n;
            if (nir_intrinsic_infos[op].index_map[NIR_INTRINSIC_BASE])
               *max_base = MAX2(*max_base, nir_intrinsic_base(intr));
         }
      }
   }
   return n;
}

/* A vertex shader summing nr_attribs vec4 inputs into the position. */
static nir_shader *
build_vs(unsigned nr_attribs)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "attribute limit VS");
   nir_def *sum = nir_imm_zero(&b, 4, 32);
   for (unsigned i = 0; i < nr_attribs; ++i) {
      nir_def *input = nir_load_input(&b, 4, 32, nir_imm_int(&b, 0),
                                      .base = i, .dest_type = nir_type_float32);
      sum = nir_fadd(&b, sum, input);
   }
   nir_store_output(&b, sum, nir_imm_int(&b, 0), .base = 0, .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS);
   b.shader->info.inputs_read = BITFIELD64_RANGE(VERT_ATTRIB_GENERIC0, nr_attribs);
   b.shader->info.outputs_written = VARYING_BIT_POS;
   b.shader->info.io_lowered = true;
   return b.shader;
}

static void
test_limits(void)
{
   const struct agx_device_key agx2 = {.encoding = AGX_ISA_ENCODING_AGX2};
   const struct agx_device_key g17p = {.chip = AGX_CHIP_G17P,
                                       .encoding = AGX_ISA_ENCODING_AGX3B};

   const unsigned agx2_max = agx_max_vertex_attribs(&agx2);
   const unsigned g17p_max = agx_max_vertex_attribs(&g17p);

   /* compiler/README.md says the ABI "accommodates 30 attributes"; the
    * allocator's export accounting keeps one half above the last export, so
    * the file holds one attribute fewer than the pure ABI arithmetic. */
   CHECK(agx2_max == 29);
   CHECK(g17p_max == 21);
   CHECK(agx2_max <= AGX_MAX_ATTRIBS && g17p_max <= AGX_MAX_ATTRIBS);

   /* DXVK fixed-function needs 18; the Vulkan minimum is 16. */
   CHECK(g17p_max >= 18);

   /* The last component of the last attribute, plus the allocator's spare
    * half, ends inside the register file, and one more attribute would
    * not. */
   CHECK(AGX_ABI_VIN_ATTRIB(agx2_max * 4 - 1) + 3 <= AGX2_HALVES);
   CHECK(AGX_ABI_VIN_ATTRIB((agx2_max + 1) * 4 - 1) + 3 > AGX2_HALVES);
   CHECK(AGX_ABI_VIN_ATTRIB(g17p_max * 4 - 1) + 3 <= AGX3B_HALVES);
   CHECK(AGX_ABI_VIN_ATTRIB((g17p_max + 1) * 4 - 1) + 3 > AGX3B_HALVES);

   /* The mask types that index attributes by slot cover the capacity. */
   CHECK(AGX_MAX_ATTRIBS <= 32);
   CHECK(AGX_MAX_VBUFS <= 32);
}

/* Monolithic path (static vertex input, no prolog): every slot up to the
 * capacity is fetched with the device's own key entry, dynamic and static
 * strides alike. */
static void
test_lower_vbo_capacity(bool dynamic_strides)
{
   struct agx_velem_key attribs[AGX_MAX_ATTRIBS] = {0};
   for (unsigned i = 0; i < AGX_MAX_ATTRIBS; ++i) {
      attribs[i] = (struct agx_velem_key){
         .format = (i & 1) ? PIPE_FORMAT_B8G8R8A8_UNORM
                           : PIPE_FORMAT_R32G32B32A32_FLOAT,
         .stride = dynamic_strides ? 0 : 16 * AGX_MAX_ATTRIBS,
      };
   }

   nir_shader *s = build_vs(AGX_MAX_ATTRIBS);
   CHECK(agx_nir_lower_vbo_raw(s, attribs,
                               (struct agx_robustness){
                                  .level = AGX_ROBUSTNESS_D3D,
                                  .soft_fault = true,
                               },
                               dynamic_strides));

   unsigned max_base;
   CHECK(count_intrinsics(s, nir_intrinsic_load_input, &max_base) == 0);
   CHECK(count_intrinsics(s, nir_intrinsic_load_vbo_base_agx, &max_base) ==
         AGX_MAX_ATTRIBS);
   ralloc_free(s);
}

/* Prolog path (dynamic vertex input): a G17P vertex shader reading the
 * device limit gathers every component, loads them from ABI slots inside the
 * 192-half register file, and its prolog exports to those slots and
 * compiles. */
static void
test_prolog_at_limit(void)
{
   const struct agx_device_key g17p = {.chip = AGX_CHIP_G17P,
                                       .encoding = AGX_ISA_ENCODING_AGX3B};
   const unsigned nr = agx_max_vertex_attribs(&g17p);

   nir_shader *main_shader = build_vs(nr);
   BITSET_DECLARE(mask, AGX_MAX_ATTRIBS * 4);
   BITSET_ZERO(mask);
   agx_nir_gather_vs_inputs(main_shader, mask);
   CHECK(BITSET_LAST_BIT(mask) == nr * 4);
   CHECK(BITSET_COUNT(mask) == nr * 4);

   CHECK(agx_nir_lower_vs_input_to_prolog(main_shader));
   unsigned max_base;
   CHECK(count_intrinsics(main_shader, nir_intrinsic_load_input, &max_base) == 0);
   CHECK(count_intrinsics(main_shader, nir_intrinsic_load_exported_agx,
                          &max_base) == nr);
   /* A vec4 load at the last slot's first component. */
   CHECK(max_base == AGX_ABI_VIN_ATTRIB((nr - 1) * 4));
   CHECK(max_base + 2 * 4 + 1 <= AGX3B_HALVES);

   /* The main shader's outputs still need hk's UVS lowering before the
    * backend accepts them; its end-to-end compile at 18 and 22 attributes is
    * covered by the host pipeline harness. The prolog is compiled below
    * exactly as hk compiles it (a secondary part). */
   ralloc_free(main_shader);

   struct agx_shader_key key = {
      .dev = g17p,
      .secondary = true,
      .no_stop = true,
   };

   struct agx_vs_prolog_key prolog = {
      .hw = true,
      .robustness = {.level = AGX_ROBUSTNESS_D3D, .soft_fault = true},
   };
   BITSET_COPY(prolog.component_mask, mask);
   for (unsigned i = 0; i < nr; ++i) {
      prolog.attribs[i] = (struct agx_velem_key){
         .format = PIPE_FORMAT_R32G32B32A32_FLOAT, .stride = 16 * nr};
   }

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "attribute limit prolog");
   agx_nir_vs_prolog_raw(&b, &prolog);
   CHECK(count_intrinsics(b.shader, nir_intrinsic_export_agx, &max_base) >=
         nr * 4);
   CHECK(max_base == AGX_ABI_VIN_ATTRIB(nr * 4 - 1));
   CHECK(max_base + 3 <= AGX3B_HALVES);

   struct agx_shader_part prolog_part = {0};
   CHECK(agx_compile_shader_nir(b.shader, &key, &prolog_part));
   CHECK(prolog_part.binary != NULL && prolog_part.info.binary_size > 4);
   fprintf(stderr, "prolog at %u attributes: nr_gprs=%u binary=%u bytes\n", nr,
           prolog_part.info.nr_gprs, prolog_part.info.binary_size);
   CHECK(prolog_part.info.nr_gprs <= AGX3B_HALVES);
   free(prolog_part.binary);
   ralloc_free(b.shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_limits();
   test_lower_vbo_capacity(true);
   test_lower_vbo_capacity(false);
   test_prolog_at_limit();
   glsl_type_singleton_decref();
   printf("vertex attribute limits: capacity %u, G17P %u, AGX2 %u PASS\n",
          AGX_MAX_ATTRIBS,
          agx_max_vertex_attribs(&(struct agx_device_key){
             .chip = AGX_CHIP_G17P, .encoding = AGX_ISA_ENCODING_AGX3B}),
          agx_max_vertex_attribs(
             &(struct agx_device_key){.encoding = AGX_ISA_ENCODING_AGX2}));
   return 0;
}
