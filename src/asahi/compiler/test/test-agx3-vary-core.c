/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The AGX3 varying encoders must reproduce real Apple f32interp/ldcfr/uvsw
 * instructions (clean center-location, marker-free forms) from the varyops
 * corpus.
 */
#include "agx3_vary_core.h"
#include "agx3_vary_core_vectors.c"
#include <stdio.h>
#include <string.h>
int main(void){
   unsigned pass=0,fail=0;
   for(size_t i=0;i<vary_vec_count;i++){
      const struct vary_vec *v=&vary_vecs[i];
      enum agx_isa_encoding e=v->enc==0?AGX_ISA_ENCODING_AGX3B:AGX_ISA_ENCODING_AGX3C;
      uint8_t out[AGX3_VARY_MAX_LENGTH]; uint8_t len=0; bool ok;
      if(!strcmp(v->op,"interp")) ok=agx3_vary_interp(e,v->a,v->b,v->c!=0,out,&len);
      else if(!strcmp(v->op,"ldcf")) ok=agx3_vary_ldcf(e,v->a,v->b,(uint8_t)v->c,out,&len);
      else ok=agx3_vary_uvsw(e,v->a,v->b,(uint8_t)v->c,v->d!=0,(uint16_t)v->wait,out,&len);
      if(!ok||len!=v->nbytes||memcmp(out,v->bytes,v->nbytes)){ if(fail<10) fprintf(stderr,"FAIL %s enc=%d i=%zu\n",v->op,v->enc,i); fail++; }
      else pass++;
   }
   {
      static const uint8_t centroid_c[] =
         {0x27,0x01,0x04,0x0a,0x2a,0x20,0xaa,0x02,0x00,0x80};
      uint8_t out[AGX3_VARY_MAX_LENGTH]; uint8_t len=0;
      if(!agx3_vary_interp_at(AGX_ISA_ENCODING_AGX3C,0,1,true,
                              AGX3_VARY_CENTROID,out,&len) ||
         len!=sizeof(centroid_c) || memcmp(out,centroid_c,sizeof(centroid_c)))
         fail++;
      else pass++;

      if(agx3_vary_interp(AGX_ISA_ENCODING_AGX3B,1,0,false,out,&len)) fail++; else pass++;
      if(agx3_vary_ldcf(AGX_ISA_ENCODING_AGX3C,0,0,9,out,&len)) fail++; else pass++;
      if(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B,0,0,0,false,0,out,&len)) fail++; else pass++;
      if(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B,0,0,1,false,0x40,out,&len)) fail++; else pass++;
   }
   {
      const uint8_t native[] = {0x2f,0x0d,0x54,0,3,0,4,2,0x10,0};
      uint8_t out[AGX3_VARY_MAX_LENGTH]; uint8_t len=0;
      if (!agx3_vary_projective_rhw(AGX_ISA_ENCODING_AGX3B,0,0,true,out,&len) ||
          len != sizeof(native) || memcmp(out,native,sizeof(native))) fail++;
      else pass++;
      if (agx3_vary_projective_rhw(AGX_ISA_ENCODING_AGX3C,0,0,false,out,&len)) fail++;
      else pass++;
      if (agx3_vary_projective_rhw(AGX_ISA_ENCODING_AGX3B,0,128,false,out,&len)) fail++;
      else pass++;
      if (agx3_vary_projective_rhw(AGX_ISA_ENCODING_AGX3B,1,0,false,out,&len)) fail++;
      else pass++;
   }
   {
      /* Fresh Apple Metal 32023.917 applegpu_g17p witnesses from
       * explicit_sample.metal (source SHA-256 2f0835ef...). These isolate
       * ITRCOORDS, perspective/projective f32interp, no-perspective
       * f32interp, and the coordinate token's live/final-use alternatives. */
      static const uint8_t coords_terminal[] = {0xaf, 0x04, 0x54, 0x0a,
                                                0x03, 0x06, 0x0a, 0x03};
      static const uint8_t coords_unmarked[] = {0xaf, 0x04, 0x54, 0x00,
                                                0x03, 0x10, 0x0a, 0x02};
      static const uint8_t projective_live[] = {0x2f, 0x05, 0x54, 0x0c, 0x03,
                                                0x00, 0x05, 0x02, 0x09, 0x00};
      static const uint8_t perspective_live[] = {0x2f, 0x05, 0x54, 0x0c, 0x03,
                                                 0x06, 0x05, 0x02, 0x08, 0x00};
      static const uint8_t linear_terminal[] = {0x2f, 0x05, 0x54, 0x10, 0x02,
                                                0x08, 0x08, 0x02, 0x20, 0x00};
      uint8_t out[AGX3_VARY_MAX_LENGTH];
      uint8_t len = 0;

      if (!agx3_vary_sample_coords(AGX_ISA_ENCODING_AGX3B, 10, 3, true, 0, out,
                                   &len) ||
          len != sizeof(coords_terminal) ||
          memcmp(out, coords_terminal, sizeof(coords_terminal)))
         fail++;
      else
         pass++;
      if (!agx3_vary_sample_coords(AGX_ISA_ENCODING_AGX3B, 0, 8, false, 0, out,
                                   &len) ||
          len != sizeof(coords_unmarked) ||
          memcmp(out, coords_unmarked, sizeof(coords_unmarked)))
         fail++;
      else
         pass++;
      if (!agx3_vary_projective_rhw_sample(AGX_ISA_ENCODING_AGX3B, 12, 0, 10,
                                           false, out, &len) ||
          len != sizeof(projective_live) ||
          memcmp(out, projective_live, sizeof(projective_live)))
         fail++;
      else
         pass++;
      if (!agx3_vary_interp_sample(AGX_ISA_ENCODING_AGX3B, 12, 3, true, 10,
                                   false, out, &len) ||
          len != sizeof(perspective_live) ||
          memcmp(out, perspective_live, sizeof(perspective_live)))
         fail++;
      else
         pass++;
      if (!agx3_vary_interp_sample(AGX_ISA_ENCODING_AGX3B, 16, 4, false, 16,
                                   true, out, &len) ||
          len != sizeof(linear_terminal) ||
          memcmp(out, linear_terminal, sizeof(linear_terminal)))
         fail++;
      else
         pass++;

      if (agx3_vary_sample_coords(AGX_ISA_ENCODING_AGX3C, 0, 0, true, 0, out,
                                  &len))
         fail++;
      else
         pass++;
      if (agx3_vary_sample_coords(AGX_ISA_ENCODING_AGX3B, 1, 0, true, 0, out,
                                  &len))
         fail++;
      else
         pass++;
      if (agx3_vary_sample_coords(AGX_ISA_ENCODING_AGX3B, 0, 256, true, 0, out,
                                  &len))
         fail++;
      else
         pass++;
      if (agx3_vary_sample_coords(AGX_ISA_ENCODING_AGX3B, 0, 0, true, 0x40, out,
                                  &len))
         fail++;
      else
         pass++;
      if (agx3_vary_interp_sample(AGX_ISA_ENCODING_AGX3B, 0, 0, false, 3, false,
                                  out, &len))
         fail++;
      else
         pass++;
      if (agx3_vary_projective_rhw_sample(AGX_ISA_ENCODING_AGX3B, 0, 128, 0,
                                          false, out, &len))
         fail++;
      else
         pass++;
   }
   {
      /* Fresh Apple flat float3 fragment, 2026-09-05: only its first
       * coefficient load carries f48. Later loads preserve the old codec. */
      const uint8_t first[] = {0x1f,0x0b,0x54,0x07,0x00,0x05};
      const uint8_t next[] = {0x1f,0x03,0x54,0x83,0x00,0x05};
      uint8_t out[AGX3_VARY_MAX_LENGTH]; uint8_t len=0;
      if(!agx3_vary_ldcf_at_start(AGX_ISA_ENCODING_AGX3B,14,0,6,true,out,&len) ||
         len!=sizeof(first) || memcmp(out,first,sizeof(first))) fail++; else pass++;
      if(!agx3_vary_ldcf_at_start(AGX_ISA_ENCODING_AGX3B,6,1,6,false,out,&len) ||
         len!=sizeof(next) || memcmp(out,next,sizeof(next))) fail++; else pass++;
      if(agx3_vary_ldcf_at_start(AGX_ISA_ENCODING_AGX3C,14,0,6,true,out,&len)) fail++; else pass++;
   }
   if(fail){ fprintf(stderr,"agx3_vary_core: %u of %zu failed\n",fail,vary_vec_count); return 1; }
   printf("agx3_vary_core: all %u varying ops reproduced\n",pass); return 0;
}
