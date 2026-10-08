/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Replay AGX_SHADER_DUMP_PATH inputs using the same compiler build. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "nir_serialize.h"
#ifdef AGX_REPLAY_D3D12
#include "asahi/d3d12/d12_nir_g17p_txf.h"
#endif

int
main(int argc, char **argv)
{
   unsigned failed = 0, total = 0;
   bool print_nir = false;
   bool lower_native32 = false;
#ifdef AGX_REPLAY_D3D12
   bool lower_d3d12_fetch = false;
   bool lower_d3d12_vertex = false;
#endif
   for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--lower-native32-samplers")) {
         lower_native32 = true;
         continue;
      }
      if (!strcmp(argv[i], "--print-nir")) {
         print_nir = true;
         continue;
      }
#ifdef AGX_REPLAY_D3D12
      if (!strcmp(argv[i], "--lower-d3d12-vertex-fetch")) {
         lower_d3d12_vertex = true;
         continue;
      }
      if (!strcmp(argv[i], "--lower-d3d12-fragment-fetch")) {
         lower_d3d12_fetch = true;
         continue;
      }
#endif
      total++;
      FILE *f = fopen(argv[i], "rb");
      if (!f)
         return 2;
      if (fseek(f, 0, SEEK_END))
         return 2;
      long size = ftell(f);
      if (size < (long)(sizeof(uint32_t) + sizeof(struct agx_shader_key)))
         return 2;
      rewind(f);
      void *data = malloc(size);
      if (!data || fread(data, 1, size, f) != (size_t)size)
         return 2;
      fclose(f);
      struct blob_reader blob;
      blob_reader_init(&blob, data, size);
      if (blob_read_uint32(&blob) != sizeof(struct agx_shader_key))
         return 2;
      struct agx_shader_key key;
      blob_copy_bytes(&blob, &key, sizeof(key));
      nir_shader *nir = nir_deserialize(NULL, &agx_nir_options, &blob);
      if (!nir || blob.overrun)
         return 2;
      fprintf(stderr, "REPLAY %s name=%s\n", argv[i],
              nir->info.name ? nir->info.name : "?");
      if (print_nir)
         nir_print_shader(nir, stderr);
#ifdef AGX_REPLAY_D3D12
      if (lower_d3d12_vertex && nir->info.stage == MESA_SHADER_VERTEX &&
          key.dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          d12_nir_lower_g17p_txf(nir, 9))
         d12_nir_shift_g17p_txf_root(nir);
      if (lower_d3d12_fetch && nir->info.stage == MESA_SHADER_FRAGMENT &&
          key.dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         struct d12_nir_g17p_fragment_sample binding;
         if (!d12_nir_lower_g17p_fragment_sample(nir, 9, &binding) ||
             !binding.indexed_fetch) {
            fprintf(stderr, "D3D12 fragment fetch input declined\n");
            return 2;
         }
      }
#endif
      if (lower_native32 && key.dev.encoding == AGX_ISA_ENCODING_AGX3B)
         agx_nir_lower_g17p_native32_samplers(nir, 16);
      struct agx_shader_part part;
      bool ok = agx_compile_shader_nir(nir, &key, &part);
      printf("%s\t%s\t%s\n", argv[i], ok ? "PASS" : "REFUSED",
             agx_last_compile_refusal());
      failed += !ok;
      free(part.binary);
      ralloc_free(nir);
      free(data);
   }
   fprintf(stderr, "REPLAY TOTAL %u refused %u\n", total, failed);
   return failed ? 1 : 0;
}
