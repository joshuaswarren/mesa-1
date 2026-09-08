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
      if(!strcmp(v->op,"store")) ok=agx3_tile_store(e,v->data,v->w,v->mask,v->off,v->cov,v->wait,out,&len);
      else ok=agx3_tile_load(e,v->data,v->w,(uint8_t)v->slot,v->mask,v->off,v->cov,v->wait,v->first,v->last,out,&len);
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
                             1, 0, AGX3_TILE_COV_1X, 0, true, true,
                             out, &len) || len != sizeof(scalar_w8) ||
             memcmp(out, v[i].bytes, len))
            fail++;
         else
            pass++;
      }
   }
   {
      uint8_t out[AGX3_TILE_MAX_LENGTH], len = 0;
      static const uint8_t rg8[] = {
         0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0xd2,0x02,0x00,0x00,0x00,
      };
      if (!agx3_tile_load(AGX_ISA_ENCODING_AGX3B, 0, AGX3_TILE_W8, 6,
                          3, 0, AGX3_TILE_COV_1X, 0, true, true, out, &len) ||
          len != sizeof(rg8) || memcmp(out, rg8, sizeof(rg8)))
         fail++;
      else
         pass++;
   }
   {
      uint8_t out[AGX3_TILE_MAX_LENGTH]; uint8_t len=0;
      if(agx3_tile_store(AGX_ISA_ENCODING_AGX3B,0,AGX3_TILE_W32,15,0,
                         AGX3_TILE_COV_1X,0x40,out,&len)) fail++; else pass++;
      if(agx3_tile_load(AGX_ISA_ENCODING_AGX3C,1,AGX3_TILE_W32,1,15,0,
                        AGX3_TILE_COV_1X,0,true,true,out,&len)) fail++; else pass++;
      if(agx3_tile_store(AGX_ISA_ENCODING_AGX3B,0,AGX3_TILE_W32,15,0,
                         AGX3_TILE_COV_1X,3,out,&len)) fail++; else pass++;
      if(agx3_tile_load(AGX_ISA_ENCODING_AGX3C,0,AGX3_TILE_W32,1,7,0,
                        AGX3_TILE_COV_1X,0,true,true,out,&len)) fail++; else pass++;
      if(agx3_tile_load(AGX_ISA_ENCODING_AGX3C,0,AGX3_TILE_W32,1,15,0,
                        AGX3_TILE_COV_PER_SAMPLE,0,true,true,out,&len)) fail++; else pass++;
   }
   if(fail){ fprintf(stderr,"agx3_tile_core: %u of %zu failed\n",fail,tile_vec_count); return 1; }
   printf("agx3_tile_core: all %u tile ops reproduced\n",pass); return 0;
}
