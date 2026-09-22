/* SPDX-License-Identifier: MIT */
#include "agx3_uniform_integer.h"
#include <stdio.h>
#include <stdlib.h>

/* Descriptor transport for replay-native.mjs. Fields: op,half,dest,dest_cache,
 * wait,shift, then kind,value,cache,discard for each of the three operands. */
int main(int argc, char **argv)
{
   if (argc != 19) return 2;
   unsigned v[18];
   for (unsigned i = 0; i < 18; ++i) {
      char *end;
      unsigned long parsed = strtoul(argv[i + 1], &end, 10);
      if (!argv[i + 1][0] || *end || parsed > UINT16_MAX) return 2;
      v[i] = parsed;
   }
   if (v[0] > AGX3_UI_ANDC || v[1] > 1 || v[3] > 1 ||
       v[4] > UINT8_MAX || v[5] > UINT8_MAX) return 2;
   struct agx3_uniform_integer_desc d = {
      .op = v[0], .half = v[1], .dest = v[2], .dest_cache = v[3],
      .wait_mask = v[4], .shift = v[5],
   };
   for (unsigned i = 0; i < 3; ++i) {
      if (v[6 + i * 4 + 2] > 1 || v[6 + i * 4 + 3] > 1) return 2;
      d.src[i] = (struct agx3_uniform_integer_src){
         v[6 + i * 4], v[6 + i * 4 + 1],
         v[6 + i * 4 + 2], v[6 + i * 4 + 3],
      };
   }
   uint8_t bytes[16], size;
   if (!agx3_uniform_integer_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &size))
      return 1;
   for (unsigned i = 0; i < size; ++i) printf("%02x", bytes[i]);
   puts("");
   return 0;
}
