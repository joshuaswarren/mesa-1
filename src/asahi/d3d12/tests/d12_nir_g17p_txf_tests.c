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
   fprintf(stderr, "FAIL:%u %s\n", __LINE__, #c); exit(1); } } while (0)

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
   /* Constant byte offsets become exactly one zero-extended native index. */
   const uint32_t offsets[] = {0,32,96,8192,UINT32_MAX & ~31u};
   for (unsigned i=0; i<sizeof(offsets)/sizeof(offsets[0]); i++) {
      nir_builder b = builder();
      nir_tex_instr *t = make_fetch(&b, nir_imm_int(b, offsets[i]));
      bool found = false;
      CHECK(d12_nir_g17p_txf_supported(b.shader, &found) && found);
      CHECK(d12_nir_lower_g17p_txf(b.shader, 9));
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      int idx = nir_tex_instr_src_index(t, nir_tex_src_texture_offset);
      CHECK(idx>=0 && nir_src_bit_size(t->src[idx].src)==64);
      CHECK(nir_src_is_const(t->src[idx].src));
      CHECK(nir_src_as_uint(t->src[idx].src)==offsets[i]/32);
      CHECK(nir_tex_instr_src_index(t,nir_tex_src_texture_handle)<0);
      CHECK(nir_src_as_uint(t->src[nir_tex_instr_src_index(t,nir_tex_src_lod)].src)==0);
      nir_validate_shader(b.shader,"native exact index test");ralloc_free(b.shader);
   }
   /* A late unsupported fetch refuses before ANY earlier fetch is changed. */
   for (unsigned bad=0; bad<8; bad++) {
      nir_builder b=builder();make_fetch(&b,nir_imm_int(b,32));
      nir_tex_instr*t=make_fetch(&b,nir_imm_int(b,64));
      int h=nir_tex_instr_src_index(t,nir_tex_src_texture_handle);
      nir_intrinsic_instr*handle=nir_src_as_intrinsic(t->src[h].src);
      switch(bad) {
      case 0:t->op=nir_texop_txl;break;
      case 1:t->sampler_dim=GLSL_SAMPLER_DIM_BUF;break;
      case 2:t->is_array=true;break;
      case 3:t->is_shadow=true;break;
      case 4:t->coord_components=3;break;
      case 5:nir_intrinsic_set_desc_set(handle,D12_UNIFORM_HEAP_TEXTURES);break;
      case 6:handle->src[0]=nir_src_for_ssa(nir_imm_int(&b,1));break;
      case 7:b.shader->info.stage=MESA_SHADER_FRAGMENT;break;
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
   make_fetch(&b,nir_imul_imm(&b,lane,32));
   bool found=false;CHECK(d12_nir_g17p_txf_supported(b.shader,&found)&&found);
   NIR_PASS(_,b.shader,agx_nir_lower_texture_for_generation,9);
   CHECK(d12_nir_lower_g17p_txf(b.shader,9));
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
