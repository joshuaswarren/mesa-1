/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The AGX3 memory encoders must reproduce real Apple lddev/stdev instructions.
 * Each vector is a device load/store decoded from the G16G/G17S corpus; the
 * encoder rebuilds its bytes from the normalized descriptor.
 */

#include "agx3_mem_core.h"
#include "agx3_mem_core_vectors.c"

#include <stdio.h>
#include <string.h>

int
main(void)
{
   unsigned pass = 0, fail = 0;

   /* Native M5 m5_copy_indexed_buffers: the first store keeps its value
    * and base live for the next store, discards its index, and waits slot8. */
   {
      const uint8_t expected[] = {
         0x0f,0x08,0x03,0x0e,0xe1,0x0e,0x00,0x80,0x00,0x20,
      };
      const struct agx3_mem_desc d = {
         .data = 0, .width = AGX3_MEM_W32, .mask = 15,
         .base = 8, .idx_kind = AGX3_MEM_IDX_REG,
         .idx_reg = 14, .idx_kill = true, .shift = 4, .wait_mask = 128,
      };
      uint8_t out[AGX3_MEM_MAX_LENGTH], len = 0;
      if (!agx3_mem_encode_store(AGX_ISA_ENCODING_AGX3C, &d, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, sizeof(expected))) {
         fprintf(stderr, "FAIL M5 register-base store extended wait\n");
         fail++;
      } else {
         pass++;
      }
   }

   /* AGX3c table composition of a descriptor load at uniform12+244,
    * preserving four words and both producer waits1+4. The compact ep0
    * cannot represent that dependency set; native form order selects ep1. */
   {
      const uint8_t expected[] = {0x4f,0x0c,0x03,0x00,0x78,0xac,0x80,0xbe,0x00,0x20};
      const struct agx3_mem_desc d = {
         .data = 8, .width = AGX3_MEM_W32, .mask = 15,
         .base_is_uniform = true, .base = 12,
         .idx_kind = AGX3_MEM_IDX_IMM, .byte_off = 244,
         .slot = 4, .wait_mask = 9,
      };
      uint8_t out[AGX3_MEM_MAX_LENGTH], len = 0;
      if (agx3_mem_encode_load_form(AGX_ISA_ENCODING_AGX3C, &d, 0, out, &len) ||
          !agx3_mem_encode_load(AGX_ISA_ENCODING_AGX3C, &d, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, sizeof(expected))) {
         fprintf(stderr, "FAIL M5 descriptor load extended waits\n");
         fail++;
      } else {
         pass++;
      }
   }

   for (size_t i = 0; i < mem_vec_count; ++i) {
      const struct mem_vec *v = &mem_vecs[i];
      const enum agx_isa_encoding e =
         v->enc == 0 ? AGX_ISA_ENCODING_AGX3B : AGX_ISA_ENCODING_AGX3C;

      struct agx3_mem_desc d = {
         .data = (uint16_t)v->data,
         .width = (enum agx3_mem_width)v->width,
         .mask = (uint8_t)v->mask,
         .shift = (uint8_t)v->shift,
         .base_is_uniform = v->base_uni != 0,
         .base = (uint16_t)v->base,
         .base_kill = v->base_kill != 0,
         .idx_kind = v->idx_reg ? AGX3_MEM_IDX_REG : AGX3_MEM_IDX_IMM,
         .idx_scaled = v->scaled != 0,
         .byte_off = v->byte_off,
         .elem_idx = v->elem_idx,
         .idx_reg = (uint16_t)v->idx_reg_v,
         .idx_signed = v->idx_signed != 0,
         .idx_kill = v->idx_kill != 0,
         .slot = (uint8_t)v->slot,
         .wait_mask = (uint16_t)v->wait,
      };

      uint8_t out[AGX3_MEM_MAX_LENGTH];
      uint8_t len = 0;
      bool ok = v->is_load ? agx3_mem_encode_load(e, &d, out, &len)
                           : agx3_mem_encode_store(e, &d, out, &len);

      if (!ok || len != v->nbytes || memcmp(out, v->bytes, v->nbytes) != 0) {
         if (fail < 10)
            fprintf(stderr, "FAIL %s i=%zu enc=%d\n",
                    v->is_load ? "lddev" : "stdev", i, v->enc);
         fail++;
      } else {
         pass++;
      }
   }

   /* Apple's complete G17P float2-position VS: its sole/final LDDEV uses
    * normalized f79=2, signals slot 6, and waits for vertex_id in slot 1. */
   {
      static const uint8_t expected[] = {
         0x67, 0x10, 0x44, 0x00, 0x00, 0x80, 0x20,
         0x00, 0x59, 0x01, 0x00, 0x40, 0x48, 0x00,
      };
      struct agx3_mem_desc d = {
         .data = 0,
         .width = AGX3_MEM_W32,
         .mask = 3,
         .shift = 3,
         .base_is_uniform = true,
         .base = 0,
         .idx_kind = AGX3_MEM_IDX_REG,
         .idx_reg = 0,
         .idx_kill = true,
         .slot = 6,
         .wait_mask = 1,
         .agx3b_last_load = true,
      };
      uint8_t out[AGX3_MEM_MAX_LENGTH];
      uint8_t len = 0;
      const bool ok = agx3_mem_encode_load(AGX_ISA_ENCODING_AGX3B, &d, out,
                                           &len);
      if (!ok || len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "FAIL G17P final vertex LDDEV\n");
         fail++;
      } else {
         pass++;
      }

      if (agx3_mem_encode_load(AGX_ISA_ENCODING_AGX3C, &d, out, &len) ||
          agx3_mem_encode_store(AGX_ISA_ENCODING_AGX3B, &d, out, &len)) {
         fprintf(stderr, "FAIL G17P final-load guard\n");
         fail++;
      } else {
         pass += 2;
      }
   }

   /* Fresh G17P: *(device uint *)(byte_base + offsets[id]). The
    * 32-bit access uses an unscaled byte index, not an element index. */
   {
      const uint8_t expected[]={0x67,0x00,0x46,0x00,0x01,0x80,0x20,
                                0x00,0x51,0x01,0x00,0x40,0x42,0x00};
      struct agx3_mem_desc d={.data=0,.width=AGX3_MEM_W32,.mask=1,
         .base_is_uniform=true,.base=4,.idx_kind=AGX3_MEM_IDX_REG,
         .idx_reg=0,.idx_kill=true,.shift=0,.slot=6,.wait_mask=32,
         .agx3b_last_load=true};
      uint8_t out[AGX3_MEM_MAX_LENGTH],len=0;
      if(!agx3_mem_encode_load(AGX_ISA_ENCODING_AGX3B,&d,out,&len)||
         len!=sizeof(expected)||memcmp(out,expected,len)){
         fprintf(stderr,"FAIL G17P byte-indexed u32 load\n");fail++;
      }else pass++;
   }

   /* Apple Metal 32023.917, applegpu_g17p, exact `device uchar *` scalar
    * load from buffer(1). The byte result occupies the low half of an
    * aligned GPR; the sole/final load carries f79=2. */
   {
      static const uint8_t expected[] = {
         0x67, 0x10, 0x44, 0x02, 0x01, 0x00, 0x20,
         0x00, 0x61, 0x01, 0x00, 0x40, 0x42, 0x00,
      };
      struct agx3_mem_desc d = {
         .data = 2,
         .width = AGX3_MEM_W8,
         .mask = 1,
         .shift = 0,
         .base_is_uniform = true,
         .base = 4,
         .idx_kind = AGX3_MEM_IDX_REG,
         .idx_reg = 0,
         .slot = 6,
         .wait_mask = 1,
         .agx3b_last_load = true,
      };
      uint8_t out[AGX3_MEM_MAX_LENGTH], len = 0;
      if (!agx3_mem_encode_load(AGX_ISA_ENCODING_AGX3B, &d, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "FAIL G17P scalar-byte LDDEV\n");
         fail++;
      } else {
         pass++;
      }
   }

   if (fail) {
      fprintf(stderr, "agx3_mem_core: %u of %zu failed\n", fail, mem_vec_count);
      return 1;
   }
   printf("agx3_mem_core: all %u corpus memory ops reproduced\n", pass);
   return 0;
}
