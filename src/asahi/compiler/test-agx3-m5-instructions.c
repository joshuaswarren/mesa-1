/* SPDX-License-Identifier: MIT */

#include "agx_compile.h"
#include "agx3_copy_core.h"
#include "agx3_iadd_core.h"
#include "agx3_mem_core.h"
#include "agx3_isel_core.h"
#include "agx3_cf_core.h"
#include "agx3_stage_core.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
   static const unsigned char expected[] = {
      0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
      0x0c,0x9d,0x10,0x26,0x24,0x81,0x01,0x40,
      0x28,0x01,0xa0,0x0a,0x27,0x06,0x04,0x0a,
      0x21,0x00,0xa1,0x02,0x68,0x01,0x20,0x00,
      0x01,0x02,0x37,0x08,0x04,0x0a,0x21,0x00,
      0xa1,0x02,0x28,0x01,0x20,0x00,0x01,0x02,
      0x27,0x00,0x20,0x2a,0x21,0x00,0xab,0x0a,
      0x00,0x00,0x4a,0x30,0x18,0x00,0x25,0x04,
      0x15,0x00,0x24,0x00,0xa0,0x22,0x00,0xe0,
      0x00,0x55,0x0e,0x00,0x00,0x00,
   };
   const struct {unsigned d,s,w; bool discard; unsigned char bytes[4];} copies[] = {
      {16,1,1,false,{0x83,0x01,0x21,0x20}},
      {0,1,0,true,{0x03,0x01,0x29,0x00}},
      {1,2,0,true,{0x03,0x02,0x29,0x01}},
   };
   for (unsigned i=0;i<3;i++) {
      unsigned char out[AGX3_COPY_MAX_LENGTH]; uint8_t len=0;
      if (!agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3C,copies[i].d,
             copies[i].s,true,false,copies[i].discard,copies[i].w,out,&len) ||
          len!=4 || memcmp(out,copies[i].bytes,4)) return 2;
   }
   unsigned char out[AGX3_STAGE_MAX_LENGTH]; uint8_t len=0;
   const unsigned char clear[]={0x13,0,0,0};
   if (!agx3_clr(AGX_ISA_ENCODING_AGX3C,2,true,false,out,&len) ||
       len!=4 || memcmp(out,clear,4)) return 3;
   puts("M5_COPY_CLEAR_PASS");
   const struct agx3_imad_desc mad = {
      .dest=2, .src_a=8, .multiplier=32, .src_c=2,
      .src_a_kill=true, .src_c_kill=true, .wait_mask=9,
   };
   const unsigned char mad_bytes[] = {
      0x37,0x24,0x04,0x5a,0x21,0x00,0xa1,0x02,0x28,0x08,0xca,0x00,0x02,0x00,
   };
   unsigned char mad_out[AGX3_IADD_MAX_LENGTH]; len=0;
   if (!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C,&mad,mad_out,&len) ||
       len!=sizeof(mad_bytes) || memcmp(mad_out,mad_bytes,sizeof(mad_bytes))) return 4;
   puts("M5_IMAD_PASS");
   const struct agx3_iadd_desc add = {
      .dest=24, .src_a_is_register=true, .src_a_reg=2, .src_a_cache=true,
      .src=16, .src_kill=true, .wait_mask=16,
   };
   const unsigned char add_bytes[] = {
      0xa7,0xc2,0x04,0x1a,0x21,0x00,0xa3,0x12,0x08,0x84,0x21,0x00,
   };
   len=0;
   if (!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3C,&add,mad_out,&len) ||
       len!=sizeof(add_bytes) || memcmp(mad_out,add_bytes,sizeof(add_bytes))) return 5;
   struct agx3_mem_desc load = {
      .data=16, .width=AGX3_MEM_W32, .mask=7, .base_is_uniform=true,
      .base=8, .idx_kind=AGX3_MEM_IDX_IMM, .byte_off=776, .slot=5,
   };
   const unsigned char load_bytes[] = {
      0x8f,0x08,0x03,0x00,0x40,0x0c,0x10,0xa1,0x01,0,0,0,0,3,
   };
   unsigned char memory[AGX3_MEM_MAX_LENGTH];len=0;
   if (!agx3_mem_encode_load(AGX_ISA_ENCODING_AGX3C,&load,memory,&len) ||
       len!=sizeof(load_bytes) || memcmp(memory,load_bytes,sizeof(load_bytes))) return 6;
   const struct agx3_isel_desc sel = {
      .dest=1,.dest_16=true,.dest_cache=true,.cmp0=2,.cmp1=12,
      .cond=AGX3_ISEL_ICOND_ULT,.true_kind=AGX3_ISEL_IMMEDIATE,
      .false_kind=AGX3_ISEL_IMMEDIATE,.sel_true=1,.sel_false=0,
      .discard={true,true,false,false},.wait_mask=64,
   };
   const unsigned char sel_bytes[] = {
      0x22,0x02,0x45,0x0b,0x2f,0x00,0xac,0x02,0x00,0x83,0x64,0x98,0x00,0x28,
   };
   unsigned char select[AGX3_ISEL_MAX_LENGTH];len=0;
   if (!agx3_isel_encode(AGX_ISA_ENCODING_AGX3C,&sel,select,&len) ||
       len!=sizeof(sel_bytes) || memcmp(select,sel_bytes,sizeof(sel_bytes))) return 7;
   puts("M5_MEMORY_SELECT_LIVENESS_PASS");
   glsl_type_singleton_init_or_ref();
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                 &agx_nir_options, "M5 EOT");
   nir_image_store_block_agx(&b, nir_imm_intN_t(&b,0,16), nir_imm_int(&b,0),
      nir_trim_vector(&b, nir_load_global_invocation_id(&b,32), 2),
      .format=PIPE_FORMAT_R8_UNORM, .image_dim=GLSL_SAMPLER_DIM_2D,
      .image_array=false, .explicit_coord=true);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {
      .dev.chip=AGX_CHIP_G17S, .dev.encoding=AGX_ISA_ENCODING_AGX3C,
      .no_preamble=true, .m5_rt0_eot=true,
   };
   struct agx_shader_part part={0};
   bool compiled=agx_compile_shader_nir(b.shader,&key,&part);
   bool ok=compiled && part.info.binary_size==sizeof(expected) &&
      part.info.main_offset==0 && part.info.main_size==sizeof(expected) &&
      part.info.nr_gprs==8 && part.info.agx3_staging_known &&
      part.info.agx3_staging_halfs==8 &&
      !memcmp(part.binary,expected,sizeof(expected));
   printf("M5_EOT_%s compiled=%u size=%u gprs=%u staging=%u\n",
      ok ? "PASS":"FAIL",compiled,part.info.binary_size,part.info.nr_gprs,
      part.info.agx3_staging_halfs);
   free(part.binary);
   ralloc_free(b.shader);
   glsl_type_singleton_decref();
   return !ok;
}
