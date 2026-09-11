#include "agx3_tile_core.h"
#include "agx3_tile_core_vectors.c"
#include <stdio.h>
#include <string.h>
int main(void){
   unsigned pass=0,fail=0;
   for(size_t i=0;i<tile_vec_count;i++){
      const struct tile_vec *v=&tile_vecs[i];
      enum agx_isa_encoding e=v->enc==0?AGX_ISA_ENCODING_AGX3B:AGX_ISA_ENCODING_AGX3C;
      uint8_t out[AGX3_TILE_MAX_LENGTH]; uint8_t len=0; bool ok;
      if(!strcmp(v->op,"store")) ok=agx3_tile_store(e,v->data,v->w,v->mask,v->off,agx3_tile_sample_imm(v->cov),v->wait,out,&len);
      else ok=agx3_tile_load(e,v->data,v->w,(uint8_t)v->slot,v->mask,v->off,agx3_tile_sample_imm(v->cov),v->wait,v->first,v->last,out,&len);
      if(!ok||len!=v->nbytes||memcmp(out,v->bytes,v->nbytes)){ if(fail<8) fprintf(stderr,"FAIL %s enc=%d i=%zu\n",v->op,v->enc,i); fail++; } else pass++;
   }
   {
      static const uint8_t scalar_w16[] = {
         0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0x82,0x02,0x00,0x00,0x00,
      };
      static const uint8_t scalar_w8[] = {
         0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0xc2,0x02,0x00,0x00,0x00,
      };
      const struct { enum agx3_tile_width width; const uint8_t *bytes; } v[] = {
         {AGX3_TILE_W16, scalar_w16}, {AGX3_TILE_W8, scalar_w8},
      };
      for (unsigned i = 0; i < sizeof(v) / sizeof(v[0]); ++i) {
         uint8_t out[AGX3_TILE_MAX_LENGTH], len = 0;
         if (!agx3_tile_load(AGX_ISA_ENCODING_AGX3B, 0, v[i].width, 6,
                             1, 0, AGX3_TILE_SAMPLE_1X, 0, true, true,
                             out, &len) || len != sizeof(scalar_w8) ||
             memcmp(out, v[i].bytes, len))
            fail++;
         else
            pass++;
      }
   }
   {
      uint8_t out[AGX3_TILE_MAX_LENGTH], len = 0;
      static const struct {
         enum agx3_tile_width width;
         uint8_t bytes[12];
      } pairs[] = {
         {AGX3_TILE_W8,
          {0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0xd2,0x02,0x00,0x00,0x00}},
         {AGX3_TILE_W16,
          {0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0x92,0x02,0x00,0x00,0x00}},
         {AGX3_TILE_W32,
          {0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0xb2,0x02,0x00,0x00,0x00}},
      };
      for (unsigned i = 0; i < sizeof(pairs) / sizeof(pairs[0]); ++i) {
         if (!agx3_tile_load(AGX_ISA_ENCODING_AGX3B, 0, pairs[i].width, 6,
                             3, 0, AGX3_TILE_SAMPLE_1X, 0, true, true, out,
                             &len) ||
             len != sizeof(pairs[i].bytes) ||
             memcmp(out, pairs[i].bytes, sizeof(pairs[i].bytes)))
            fail++;
         else
            pass++;
      }
   }
   {
      /* Fresh Apple B5G6R5Unorm framebuffer-fetch witness. */
      static const uint8_t rgb565[] = {
         0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0x9a,0x02,0x00,0x00,0x00,
      };
      uint8_t out[AGX3_TILE_MAX_LENGTH], len = 0;
      if (!agx3_tile_load(AGX_ISA_ENCODING_AGX3B, 0, AGX3_TILE_W16, 6,
                          7, 0, AGX3_TILE_SAMPLE_1X, 0, true, true, out,
                          &len) ||
          len != sizeof(rgb565) || memcmp(out, rgb565, len))
         fail++;
      else
         pass++;
   }
   {
      /* Apple's own applegpu_g17p multisample colour stores. Every one is the
       * register form of the sample-mask operand; there is no immediate
       * all-sample encoding anywhere in the corpus.
       *
       * .tmp/g17p-msaa-grounding, metal-nt MetalToolchain v27.1.5194.15:
       *   msaa_f_plain_s2 / _s4  (per-pixel, RGBA8Unorm)  -- one store reading
       *     SR39 straight through, waiting on the slot that read it.
       *   msaa_f_sampleid_s4     (per-sample, unrolled)   -- four stores, each
       *     masking that same SR39 word with 1 / 2 / 4 / 8 into its own
       *     register, so a register here is a plain sample bitmask.
       */
      static const struct {
         uint16_t data, reg;
         bool kill;
         uint16_t wait;
         uint8_t bytes[12];
      } stores[] = {
         /* per-pixel 2x and 4x are the same instruction */
         {0, 10, true, 1,
          {0xe7,0x16,0x54,0x00,0x00,0x00,0x0a,0x4e,0x08,0x00,0x00,0x08}},
         {6, 11, true, 0,
          {0xe7,0x06,0x54,0x06,0x00,0x00,0x0b,0x4e,0x08,0x00,0x00,0x08}},
         {4,  9, true, 0,
          {0xe7,0x06,0x54,0x04,0x00,0x00,0x09,0x4e,0x08,0x00,0x00,0x08}},
         {2, 10, true, 0,
          {0xe7,0x06,0x54,0x02,0x00,0x00,0x0a,0x4e,0x08,0x00,0x00,0x08}},
         {0,  8, true, 0,
          {0xe7,0x06,0x54,0x00,0x00,0x00,0x08,0x4e,0x08,0x00,0x00,0x08}},
      };
      for (unsigned i = 0; i < sizeof(stores) / sizeof(stores[0]); ++i) {
         uint8_t out[AGX3_TILE_MAX_LENGTH], len = 0;
         if (!agx3_tile_store(AGX_ISA_ENCODING_AGX3B, stores[i].data,
                              AGX3_TILE_W8, 15, 0,
                              agx3_tile_sample_reg(stores[i].reg,
                                                   stores[i].kill),
                              stores[i].wait, out, &len) ||
             len != sizeof(stores[i].bytes) ||
             memcmp(out, stores[i].bytes, len)) {
            fprintf(stderr, "FAIL native MS store %u len=%u bytes=", i, len);
            for (unsigned j = 0; j < len; ++j) fprintf(stderr, "%02x", out[j]);
            fprintf(stderr, "\n");
            fail++;
         } else
            pass++;
      }

      /* msaa_f_fetch_s2 / _s4: the framebuffer-fetch load takes the same
       * register operand, with its last-use bit clear because the mask
       * register is read again by the second branch. */
      static const uint8_t fetch[] = {
         0x67,0x06,0x54,0x00,0x00,0x00,0x13,0xce,0x02,0x00,0x00,0x20,
      };
      uint8_t out[AGX3_TILE_MAX_LENGTH], len = 0;
      if (!agx3_tile_load(AGX_ISA_ENCODING_AGX3B, 0, AGX3_TILE_W8, 6, 15, 0,
                          agx3_tile_sample_reg(19, false), 0, false, false,
                          out, &len) ||
          len != sizeof(fetch) || memcmp(out, fetch, len)) {
         fprintf(stderr, "FAIL native MS fetch len=%u bytes=", len);
         for (unsigned j = 0; j < len; ++j) fprintf(stderr, "%02x", out[j]);
         fprintf(stderr, "\n");
         fail++;
      } else
         pass++;

      /* The register form is AGX3b-only: no AGX3c multisample witness. */
      if (agx3_tile_store(AGX_ISA_ENCODING_AGX3C, 0, AGX3_TILE_W8, 15, 0,
                          agx3_tile_sample_reg(10, true), 0, out, &len))
         fail++;
      else
         pass++;
   }
   {
      uint8_t out[AGX3_TILE_MAX_LENGTH]; uint8_t len=0;
      if(agx3_tile_store(AGX_ISA_ENCODING_AGX3B,0,AGX3_TILE_W32,15,0,
                         AGX3_TILE_SAMPLE_1X,0x40,out,&len)) fail++; else pass++;
      if(agx3_tile_load(AGX_ISA_ENCODING_AGX3C,1,AGX3_TILE_W32,1,15,0,
                        AGX3_TILE_SAMPLE_1X,0,true,true,out,&len)) fail++; else pass++;
      if(agx3_tile_store(AGX_ISA_ENCODING_AGX3B,0,AGX3_TILE_W32,15,0,
                         AGX3_TILE_SAMPLE_1X,3,out,&len)) fail++; else pass++;
      if(agx3_tile_load(AGX_ISA_ENCODING_AGX3C,0,AGX3_TILE_W32,1,7,0,
                        AGX3_TILE_SAMPLE_1X,0,true,true,out,&len)) fail++; else pass++;
      if(agx3_tile_load(AGX_ISA_ENCODING_AGX3C,0,AGX3_TILE_W32,1,3,0,
                        AGX3_TILE_SAMPLE_1X,0,true,true,out,&len)) fail++; else pass++;
      if(agx3_tile_load(AGX_ISA_ENCODING_AGX3B,0,AGX3_TILE_W16,1,5,0,
                        AGX3_TILE_SAMPLE_1X,0,true,true,out,&len)) fail++; else pass++;
      if(agx3_tile_load(AGX_ISA_ENCODING_AGX3B,0,AGX3_TILE_W8,1,7,0,
                        AGX3_TILE_SAMPLE_1X,0,true,true,out,&len)) fail++; else pass++;
      /* The register sample operand is AGX3b-only, and no half register
       * above the form's 191 limit encodes. */
      if(agx3_tile_load(AGX_ISA_ENCODING_AGX3C,0,AGX3_TILE_W32,1,15,0,
                        agx3_tile_sample_reg(10,false),0,true,true,out,&len)) fail++; else pass++;
      if(agx3_tile_load(AGX_ISA_ENCODING_AGX3B,0,AGX3_TILE_W8,1,15,0,
                        agx3_tile_sample_reg(192,false),0,true,true,out,&len)) fail++; else pass++;
   }
   if(fail){ fprintf(stderr,"agx3_tile_core: %u of %zu failed\n",fail,tile_vec_count); return 1; }
   printf("agx3_tile_core: all %u tile ops reproduced\n",pass); return 0;
}
