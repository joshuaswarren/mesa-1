/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * CPU NIR/ISA compilation only: no device, queue, submission or pixel verdict. */
#include "../d12_nir_g17p_txf.h"
#include "../dxil/d12_dxil.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "nir_builder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;
#define CHECK(c) do { ++checks; if (!(c)) { \
   fprintf(stderr, "FAIL:%u %s\n", (unsigned)__LINE__, #c); exit(1); } } while (0)

static nir_tex_instr *
make_fetch(nir_builder *b, nir_def *offset)
{
   nir_def *handle = nir_bindless_image_agx(b, offset,
                                           .desc_set = D12_UNIFORM_TEXTURES);
   nir_tex_instr *t = nir_tex_instr_create(b->shader, 3);
   t->op = nir_texop_txf;
   t->sampler_dim = GLSL_SAMPLER_DIM_2D;
   t->dest_type = nir_type_float32;
   t->coord_components = 2;
   t->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
   t->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord,
      nir_vec2(b, nir_imm_int(b, 1), nir_imm_int(b, 2)));
   t->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   nir_def_init(&t->instr, &t->def, 4, 32);
   nir_builder_instr_insert(b, &t->instr);
   nir_store_global(b, &t->def, nir_load_root_agx(b),
                    .align_mul=16, .write_mask=15);
   return t;
}

static nir_builder
builder(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "D12 G17P native table txf CPU test");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static char *
snapshot(nir_shader *s)
{
   FILE *f = tmpfile(); CHECK(f);
   nir_print_shader(s, f); CHECK(!ferror(f));
   long n = ftell(f); CHECK(n >= 0); rewind(f);
   char *bytes = malloc((size_t)n + 1); CHECK(bytes);
   CHECK(fread(bytes, 1, (size_t)n, f) == (size_t)n);
   bytes[n] = 0; CHECK(fclose(f) == 0); return bytes;
}

static bool
lower_test_root(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   (void)data;
   if (intr->intrinsic != nir_intrinsic_load_root_agx) return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_load_preamble(b, 1, 64, .base = D12_UNIFORM_ROOT));
   return true;
}

int main(void)
{
   /* Deadzone's first PSO blends two Texture.Load results from t0 and t1.
    * This is the same array/index contract as compute, without an API sampler. */
   {
      nir_builder b = builder();
      b.shader->info.stage = MESA_SHADER_FRAGMENT;
      nir_tex_instr *a = make_fetch(&b, nir_imm_int(&b, 0));
      nir_tex_instr *z = make_fetch(&b, nir_imm_int(&b, 32));
      struct d12_nir_g17p_fragment_sample binding = {0};
      CHECK(d12_nir_lower_g17p_fragment_sample(b.shader, 9, &binding));
      CHECK(binding.indexed_fetch);
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      CHECK(a->texture_index == 0 && z->texture_index == 1);
      CHECK(nir_tex_instr_src_index(a, nir_tex_src_texture_handle) < 0);
      CHECK(nir_tex_instr_src_index(z, nir_tex_src_texture_handle) < 0);
      nir_validate_shader(b.shader, "G17P fragment array fetches");
      ralloc_free(b.shader);
   }

   /* Buffer access has already become a 2D native fetch, while
    * coord_components retains the API's one-dimensional metadata. Include
    * index 16, just beyond the packer's four-bit immediate index. */
   {
      nir_builder v = builder();
      v.shader->info.stage = MESA_SHADER_VERTEX;
      nir_tex_instr *t = make_fetch(&v, nir_imm_int(&v, 16 * 32));
      nir_instr *store = nir_block_last_instr(t->instr.block);
      CHECK(store->type == nir_instr_type_intrinsic);
      CHECK(nir_instr_as_intrinsic(store)->intrinsic == nir_intrinsic_store_global);
      nir_instr_remove(store);
      v.cursor = nir_after_instr(&t->instr);
      int coord = nir_tex_instr_src_index(t, nir_tex_src_coord);
      t->src[coord].src_type = nir_tex_src_backend1;
      t->coord_components = 1;
      for (unsigned i = 0; i < 4; ++i)
         nir_store_uvs_agx(&v, nir_channel(&v, &t->def, i),
                           nir_imm_intN_t(&v, i, 16));
      nir_store_uvs_agx(&v, nir_imm_float(&v, 1.0f),
                        nir_imm_intN_t(&v, 4, 16));
      bool found = false;
      CHECK(d12_nir_g17p_txf_supported(v.shader, &found) && found);
      CHECK(d12_nir_lower_g17p_txf(v.shader, 9));
      NIR_PASS(_, v.shader, nir_opt_constant_folding);
      CHECK(t->texture_index == 16);
      v.shader->info.outputs_written = VARYING_BIT_POS | VARYING_BIT_PSIZ;
      agx_preprocess_nir_g17p(v.shader);
      struct agx_shader_key key = {.dev.encoding = AGX_ISA_ENCODING_AGX3B,
                                  .no_preamble = true};
      struct agx_shader_part part = {0};
      CHECK(agx_compile_shader_nir(v.shader, &key, &part));
      CHECK(part.binary && part.info.uses_txf);
      free(part.binary);
      ralloc_free(v.shader);
   }

   /* A vertex can mix a buffer Load with an explicit-level sample. Reuse
    * the same native texture array while retaining sampler entry one. */
   {
      nir_builder b = builder();
      b.shader->info.stage = MESA_SHADER_VERTEX;
      nir_tex_instr *fetch = make_fetch(&b, nir_imm_int(&b, 0));
      nir_tex_instr *sample = nir_tex_instr_create(b.shader, 4);
      sample->op = nir_texop_txl;
      sample->sampler_dim = GLSL_SAMPLER_DIM_2D;
      sample->dest_type = nir_type_float32;
      sample->coord_components = 2;
      sample->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1,
         nir_imm_vec2(&b, 0.25f, 0.75f));
      sample->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle,
         nir_bindless_image_agx(&b, nir_imm_int(&b, 32),
                                .desc_set = D12_UNIFORM_TEXTURES));
      sample->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle,
         nir_imm_intN_t(&b, 1, 16));
      sample->src[3] = nir_tex_src_for_ssa(nir_tex_src_lod,
         nir_imm_floatN_t(&b, 0.0f, 16));
      nir_def_init(&sample->instr, &sample->def, 4, 32);
      nir_builder_instr_insert(&b, &sample->instr);
      nir_store_global(&b, &sample->def, nir_load_root_agx(&b),
                       .align_mul = 16, .write_mask = 15);
      struct d12_nir_g17p_fragment_sample binding = {0};
      CHECK(d12_nir_lower_g17p_graphics_resources(b.shader, 9, &binding));
      CHECK(binding.indexed_fetch && binding.indexed_samplers);
      CHECK(nir_tex_instr_src_index(fetch, nir_tex_src_texture_handle) < 0);
      CHECK(nir_tex_instr_src_index(sample, nir_tex_src_texture_handle) < 0);
      int sampler = nir_tex_instr_src_index(sample, nir_tex_src_sampler_handle);
      CHECK(sampler >= 0 && nir_src_bit_size(sample->src[sampler].src) == 64);
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      CHECK(sample->texture_index == 1);
      sampler = nir_tex_instr_src_index(sample, nir_tex_src_sampler_handle);
      CHECK(sampler >= 0);
      CHECK(nir_src_as_uint(sample->src[sampler].src) == 1);
      nir_validate_shader(b.shader, "mixed native32 fetch/sample");
      ralloc_free(b.shader);
   }

   /* Constant byte offsets become exactly one zero-extended native index. */
   const uint32_t offsets[] = {0,32,96,8192,UINT32_MAX & ~31u};
   for (unsigned i=0; i<sizeof(offsets)/sizeof(offsets[0]); i++) {
      nir_builder b = builder();
      nir_tex_instr *t = make_fetch(&b, nir_imm_int(&b, offsets[i]));
      bool found = false;
      CHECK(d12_nir_g17p_txf_supported(b.shader, &found) && found);
      CHECK(d12_nir_lower_g17p_txf(b.shader, 9));
      int idx = nir_tex_instr_src_index(t, nir_tex_src_texture_offset);
      CHECK(idx>=0 && nir_src_bit_size(t->src[idx].src)==64);
      /* Constant folding canonically moves a constant texture_offset into
       * texture_index and removes that source (try_fold_tex_offset). Keep a
       * separate actual SSA use so its folded arithmetic still has to be64. */
      nir_intrinsic_instr *index_sink = nir_store_global(
         &b, t->src[idx].src.ssa, nir_load_root_agx(&b),
         .align_mul=8, .write_mask=1);
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      CHECK(nir_src_bit_size(index_sink->src[0])==64);
      CHECK(nir_src_is_const(index_sink->src[0]));
      CHECK(nir_src_as_uint(index_sink->src[0])==offsets[i]/32);
      CHECK(nir_tex_instr_src_index(t,nir_tex_src_texture_offset)<0);
      CHECK(t->texture_index==offsets[i]/32);
      CHECK(nir_tex_instr_src_index(t,nir_tex_src_texture_handle)<0);
      CHECK(t->coord_components==2);
      int coord=nir_tex_instr_src_index(t,nir_tex_src_coord);
      CHECK(coord>=0 && nir_src_num_components(t->src[coord].src)==2);
      CHECK(nir_src_comp_as_uint(t->src[coord].src,0)==1);
      CHECK(nir_src_comp_as_uint(t->src[coord].src,1)==2);
      CHECK(nir_src_as_uint(t->src[nir_tex_instr_src_index(t,nir_tex_src_lod)].src)==0);
      nir_validate_shader(b.shader,"native exact index test");ralloc_free(b.shader);
   }
   /* A late unsupported fetch refuses before ANY earlier fetch is changed. */
   for (unsigned bad=0; bad<8; bad++) {
      nir_builder b=builder();make_fetch(&b,nir_imm_int(&b,32));
      nir_tex_instr*t=make_fetch(&b,nir_imm_int(&b,bad==6?1:64));
      int h=nir_tex_instr_src_index(t,nir_tex_src_texture_handle);
      nir_intrinsic_instr*handle=nir_src_as_intrinsic(t->src[h].src);
      switch(bad) {
      case 0:t->op=nir_texop_txl;break;
      case 1:t->sampler_dim=GLSL_SAMPLER_DIM_BUF;break;
      case 2:t->is_array=true;break;
      case 3:t->is_shadow=true;break;
      case 4:t->coord_components=3;break;
      case 5:nir_intrinsic_set_desc_set(handle,D12_UNIFORM_HEAP_TEXTURES);break;
      case 6:break; /* Unaligned input built above with intact SSA use lists. */
      case 7:b.shader->info.stage=MESA_SHADER_GEOMETRY;break;
      }
      char *before=snapshot(b.shader);bool found=false;
      CHECK(!d12_nir_g17p_txf_supported(b.shader,&found));
      CHECK(!d12_nir_lower_g17p_txf(b.shader,9));
      char*after=snapshot(b.shader);CHECK(!strcmp(before,after));
      free(before);free(after);ralloc_free(b.shader);
   }
   /* AGX2/generation mismatch is a read-only refusal. */
   nir_builder b=builder();make_fetch(&b,nir_imm_int(&b,32));
   char*before=snapshot(b.shader);CHECK(!d12_nir_lower_g17p_txf(b.shader,0));
   char*after=snapshot(b.shader);CHECK(!strcmp(before,after));free(before);free(after);ralloc_free(b.shader);
   /* Real gen9 lowering -> adapter -> G17P preprocess -> ISA compiler. */
   b=builder();nir_def*lane=nir_channel(&b,nir_load_local_invocation_id(&b),0);
   nir_tex_instr *dynamic = make_fetch(&b,nir_imul_imm(&b,lane,32));
   bool found=false;CHECK(d12_nir_g17p_txf_supported(b.shader,&found)&&found);
   NIR_PASS(_,b.shader,agx_nir_lower_texture_for_generation,9);
   CHECK(d12_nir_lower_g17p_txf(b.shader,9));
   int native_index=nir_tex_instr_src_index(dynamic,nir_tex_src_texture_offset);
   CHECK(native_index>=0 && nir_src_bit_size(dynamic->src[native_index].src)==64);
   CHECK(!nir_src_is_const(dynamic->src[native_index].src));
   /* Model the driver's root placeholder publication at its actual phase. */
   agx_preprocess_nir_g17p(b.shader);
   NIR_PASS(_,b.shader,nir_shader_intrinsics_pass,lower_test_root,
            nir_metadata_control_flow,NULL);
   CHECK(d12_nir_shift_g17p_txf_root(b.shader));
   nir_validate_shader(b.shader,"G17P table txf compiler input");
   struct agx_shader_key key={.dev.encoding=AGX_ISA_ENCODING_AGX3B,
      .no_preamble=true,.reserved_preamble=D12_UNIFORM_RESERVED};
   struct agx_shader_part part={0};
   CHECK(agx_compile_shader_nir(b.shader,&key,&part));
   CHECK(part.binary&&part.info.binary_size&&part.info.uses_txf);
   CHECK(!part.info.has_preamble&&part.info.agx3_staging_known);
   free(part.binary);ralloc_free(b.shader);
   printf("PASS %u CPU NIR/index/transaction/compiler checks; NO_QUEUE_NO_GPU_VERDICT\n",checks);
   return 0;
}
