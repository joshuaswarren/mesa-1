/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * DXIL parser capacity test. The parser used to keep every table in a
 * fixed array (4096 instructions, 8192 values, 256 basic blocks, 1024
 * record operands) and refused anything larger at PSO creation, which is
 * where real uber-shaders land. Build a module past the instruction,
 * value and block limits with Mesa's own DXIL writer, parse and translate
 * it; then walk a hand-built bitstream whose one record carries more
 * operands than the old scratch held.
 *
 * CPU only: no device, no GPU.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "compiler/glsl_types.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_to_dxil.h"
#include "util/blob.h"

#include "d12_native_descriptor_core.h"
#include "dxil/d12_dxil.h"

#define INSTS  20000 /* > the old 4096 instruction / 8192 value limits */
#define BLOCKS 600   /* > the old 256 block limit */
#define RECORD_OPS 5000 /* > the old 1024 record operand limit */

static int failures;

#define CHECK(cond, ...)                                                    \
   do {                                                                     \
      if (cond) {                                                           \
         printf("ok   - " __VA_ARGS__);                                     \
         printf("\n");                                                      \
      } else {                                                              \
         printf("FAIL - " __VA_ARGS__);                                     \
         printf("\n");                                                      \
         failures++;                                                        \
      }                                                                     \
   } while (0)

static int
probe_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/*
 * A fragment shader: one long dependent chain of adds with distinct
 * constants (every constant is a value of its own), then a ladder of ifs
 * keyed on an input varying so the function has many basic blocks and phis,
 * written to SV_Target so nothing is dead.
 */
static nir_shader *
build_big_ps(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, opts,
                                                  "big_ps");

   nir_variable *in = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec4_type(), "var0");
   in->data.location = VARYING_SLOT_VAR0;
   in->data.driver_location = 0;

   nir_def *x = nir_channel(&b, nir_load_var(&b, in), 0);
   nir_def *id = nir_f2i32(&b, x);
   nir_def *v = x;

   /* sin(v) + c: the transcendental keeps nir_opt_algebraic from folding
    * the chain into one add. */
   for (unsigned i = 0; i < INSTS / 2; i++)
      v = nir_fadd(&b, nir_fsin(&b, v), nir_imm_float(&b, 1.0f + i * 0.25f));

   /* Bodies past the DXIL writer's peephole-select limit stay real
    * blocks instead of becoming bcsel. */
   for (unsigned i = 0; i < BLOCKS / 2; i++) {
      nir_push_if(&b, nir_ieq_imm(&b, id, i));
      nir_def *t = v;
      for (unsigned j = 0; j < 16; j++)
         t = nir_fsin(&b, t);
      nir_pop_if(&b, NULL);
      v = nir_if_phi(&b, t, v);
   }

   nir_variable *out = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "SV_Target");
   out->data.location = FRAG_RESULT_DATA0;
   out->data.driver_location = 0;
   nir_store_var(&b, out,
                 nir_vec4(&b, v, nir_imm_float(&b, 0.0f),
                          nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f)),
                 0xf);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   return b.shader;
}

/*
 * Part two: a hand-built bitstream. Long strings and constant data
 * tables arrive as ONE record with one operand per element, so the record
 * scratch has to grow past any fixed size. Write MODULE { UNABBREV_RECORD
 * with RECORD_OPS operands } with the LLVM bit encoding and walk it.
 */
struct bitwriter {
   uint8_t *data;
   size_t size, capacity;
   uint64_t pos_bits;
};

static void
bw_emit(struct bitwriter *w, unsigned bits, uint64_t value)
{
   for (unsigned i = 0; i < bits; i++, w->pos_bits++) {
      size_t byte = w->pos_bits / 8;
      if (byte >= w->capacity) {
         w->capacity = MAX2(w->capacity * 2, (size_t)4096);
         w->data = realloc(w->data, w->capacity);
         memset(w->data + byte, 0, w->capacity - byte);
      }
      if ((value >> i) & 1)
         w->data[byte] |= 1u << (w->pos_bits % 8);
      w->size = byte + 1;
   }
}

static void
bw_vbr(struct bitwriter *w, unsigned bits, uint64_t value)
{
   uint64_t threshold = 1ull << (bits - 1);
   while (value >= threshold) {
      bw_emit(w, bits, (value & (threshold - 1)) | threshold);
      value >>= bits - 1;
   }
   bw_emit(w, bits, value);
}

static void
bw_align32(struct bitwriter *w)
{
   while (w->pos_bits % 32)
      bw_emit(w, 1, 0);
}

static unsigned seen_ops;
static unsigned seen_records;

static bool
count_record(void *user, unsigned block_id, unsigned depth,
             const struct d12_record *rec, unsigned abbrev_id)
{
   seen_records++;
   seen_ops = rec->num_ops;
   for (unsigned i = 0; i < rec->num_ops; i++) {
      if (rec->ops[i] != (i & 31))
         return false;
   }
   return block_id == D12_BLOCK_MODULE;
}

static void
test_wide_record(void)
{
   struct bitwriter w = {0};
   bw_emit(&w, 32, D12_BITCODE_MAGIC);
   bw_emit(&w, 2, D12_ABBREV_ENTER_SUBBLOCK);
   bw_vbr(&w, 8, D12_BLOCK_MODULE);
   bw_vbr(&w, 4, 3); /* abbreviation width inside the block */
   bw_align32(&w);
   size_t len_pos = w.pos_bits / 8;
   bw_emit(&w, 32, 0); /* patched below */
   uint64_t body_start = w.pos_bits;

   bw_emit(&w, 3, D12_ABBREV_UNABBREV_RECORD);
   bw_vbr(&w, 6, 1); /* record code */
   bw_vbr(&w, 6, RECORD_OPS);
   for (unsigned i = 0; i < RECORD_OPS; i++)
      bw_vbr(&w, 6, i & 31);
   bw_emit(&w, 3, D12_ABBREV_END_BLOCK);
   bw_align32(&w);

   uint32_t len_words = (uint32_t)((w.pos_bits - body_start) / 32);
   memcpy(w.data + len_pos, &len_words, sizeof(len_words));

   struct d12_bitstream bs;
   d12_bitstream_init(&bs, w.data, w.size);
   struct d12_block_visitor visitor = {.record = count_record};
   bool ok = d12_dxil_walk(&bs, &visitor);
   CHECK(ok && seen_records == 1 && seen_ops == RECORD_OPS,
         "a %u-operand record decodes intact (%s, %u records, %u ops)",
         RECORD_OPS, ok ? "walked" : bs.error_msg, seen_records, seen_ops);
   CHECK(bs.record_ops == NULL, "the walk released its record scratch");
   free(w.data);
}

static double
now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int
main(int argc, char **argv)
{
   glsl_type_singleton_init_or_ref();

   test_wide_record();

   nir_shader_compiler_options options;
   dxil_get_nir_compiler_options(&options, SHADER_MODEL_6_2, 1 << 5, 1 << 5);

   nir_shader *nir = build_big_ps(&options);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            probe_type_size, 0);

   struct nir_to_dxil_options opts = {
      .environment = DXIL_ENVIRONMENT_VULKAN,
      .shader_model_max = SHADER_MODEL_6_2,
      .validator_version_max = DXIL_VALIDATOR_1_6,
   };

   struct blob blob;
   blob_init(&blob);
   if (!nir_to_dxil(nir, &opts, NULL, &blob)) {
      printf("FAIL - nir_to_dxil\n");
      return 1;
   }
   printf("ok   - synthetic DXIL: %zu bytes\n", blob.size);

   const uint8_t *bc;
   size_t bc_size;
   const char *why = NULL;
   CHECK(d12_dxil_find_bitcode(blob.data, blob.size, &bc, &bc_size, &why),
         "bitcode located (%s)", why ? why : "container");

   struct d12_module *mod = d12_dxil_module_alloc();
   double t0 = now_ms();
   bool parsed = d12_dxil_module_parse(mod, bc, bc_size);
   double t1 = now_ms();
   CHECK(parsed, "module parses: %s", mod->error ? mod->error : "no error");
   if (!parsed)
      return 1;

   printf("info - %u types, %u values, %u functions, %u metadata, "
          "parse %.2f ms\n", mod->num_types, mod->num_values,
          mod->num_functions, mod->num_metadata, t1 - t0);

   const struct d12_function *entry = NULL;
   for (unsigned i = 0; i < mod->num_functions; i++) {
      if (!mod->functions[i].is_declaration)
         entry = &mod->functions[i];
   }
   CHECK(entry != NULL, "entry function body found");
   if (!entry)
      return 1;

   CHECK(mod->num_values > 8192, "values past the old 8192 limit: %u",
         mod->num_values);
   CHECK(entry->num_insts > 4096, "instructions past the old 4096 limit: %u",
         entry->num_insts);

   unsigned blocks = 0;
   unsigned phis = 0;
   unsigned max_phi_srcs = 0;
   for (unsigned i = 0; i < entry->num_insts; i++) {
      const struct d12_inst *inst = &entry->insts[i];
      blocks = MAX2(blocks, inst->block + 1);
      if (inst->code == D12_FUNC_INST_PHI) {
         phis++;
         max_phi_srcs = MAX2(max_phi_srcs, inst->num_phi_srcs);
      }
   }
   CHECK(blocks > 256, "basic blocks past the old 256 limit: %u", blocks);
   CHECK(phis >= BLOCKS / 2 && max_phi_srcs == 2, "%u phis, %u sources each",
         phis, max_phi_srcs);

   /* Every instruction operand must resolve or be ~0; nothing may index
    * past the finished value table. */
   bool ops_sane = true;
   for (unsigned i = 0; i < entry->num_insts; i++) {
      const struct d12_inst *inst = &entry->insts[i];
      for (unsigned o = 0; o < inst->num_ops; o++) {
         if (inst->ops[o] != ~0u && inst->ops[o] >= mod->num_values)
            ops_sane = false;
      }
   }
   CHECK(ops_sane, "every resolved operand indexes the value table");

   /* The translator consumes the dynamically sized block table. */
   struct d12_binding_map map;
   d12_binding_map_from_module(mod, &map);
   const char *error = NULL;
   nir_shader *out = d12_dxil_to_nir(mod, MESA_SHADER_FRAGMENT, &map,
                                     D12_NATIVE_DESCRIPTOR_GEN7, &error);
   CHECK(out != NULL, "DXIL -> NIR: %s", error ? error : "ok");
   if (out) {
      nir_validate_shader(out, "d12_dxil_limits");
      ralloc_free(out);
   }

   /* Reparse to time it without first-touch page faults. */
   d12_dxil_module_finish(mod);
   t0 = now_ms();
   parsed = d12_dxil_module_parse(mod, bc, bc_size);
   t1 = now_ms();
   CHECK(parsed, "reparse into a finished module: %.2f ms", t1 - t0);

   d12_dxil_module_free(mod);
   ralloc_free(nir);
   blob_finish(&blob);
   glsl_type_singleton_decref();

   printf(failures ? "DXIL LIMITS: %d FAILURES\n" : "DXIL LIMITS: ALL PASS\n",
          failures);
   return failures ? 1 : 0;
}
