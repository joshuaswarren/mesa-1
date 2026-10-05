/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_linker.h"
#include "nir_builder.h"
#include "util/format/u_formats.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned
count_intrinsic(nir_shader *s, nir_intrinsic_op op)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               count++;
         }
      }
   }
   return count;
}

static bool
test_configuration(enum pipe_advanced_blend_mode op,
                   enum pipe_blend_overlap_mode overlap,
                   bool src_premultiplied, bool dst_premultiplied,
                   enum pipe_format format)
{
   const unsigned mode = agx_pack_blend_advanced(
      op, overlap, src_premultiplied, dst_premultiplied, false);
   const struct agx_blend_advanced unpacked = agx_unpack_blend_advanced(mode);
   if (unpacked.op != op || unpacked.overlap != overlap ||
       unpacked.src_premultiplied != src_premultiplied ||
       unpacked.dst_premultiplied != dst_premultiplied ||
       unpacked.clamp_results) {
      fprintf(stderr,
              "advanced blend state did not round-trip "
              "(op=%u overlap=%u src_premul=%u dst_premul=%u format=%u)\n",
              op, overlap, src_premultiplied, dst_premultiplied, format);
      return false;
   }

   struct agx_fs_epilog_key key = {
      .link = {
         .size_32 = BITFIELD_BIT(0),
         .loc_written = BITFIELD_BIT(0),
      },
      .nr_samples = 1,
   };
   memset(key.remap, -1, sizeof(key.remap));
   key.remap[0] = 0;
   key.rt_formats[0] = format;
   key.blend.rt[0].advanced_blend = true;
   key.blend.rt[0].colormask = 0xf;
   key.blend.rt[0].mode = mode;

   nir_builder b =
      nir_builder_init_simple_shader(0, &agx_nir_options,
                                     "G17P advanced blend epilog");
   agx_nir_fs_epilog(&b, &key);
   nir_validate_shader(b.shader, "advanced blend epilog");

   const unsigned loads =
      count_intrinsic(b.shader, nir_intrinsic_load_local_pixel_agx);
   const unsigned stores =
      count_intrinsic(b.shader, nir_intrinsic_store_local_pixel_agx);
   if (loads != 1 || stores != 1 ||
       !b.shader->info.fs.uses_fbfetch_output) {
      fprintf(stderr,
              "advanced blend did not retain one tile load/store (%u/%u) "
              "(op=%u overlap=%u src_premul=%u dst_premul=%u format=%u)\n",
              loads, stores, op, overlap, src_premultiplied,
              dst_premultiplied, format);
      ralloc_free(b.shader);
      return false;
   }

   agx_preprocess_nir(b.shader);
   struct agx_shader_key backend_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled =
      agx_compile_shader_nir(b.shader, &backend_key, &part);
   static const uint8_t entry[] = {0x87,0x02,0x54,0x00,0x06,0x00};
   static const uint8_t read_wait[] = {0x87,0x02,0x54,0x08,0x08,0x00};
   static const uint8_t tile_load[] = {
      0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0xce,0x02,0x00,0x00,0x00,
   };
   /* The FLTNORM pair is packed straight into the store registers (r1), as
    * Apple's fragments do, so no payload copy into r0 is needed. */
   static const uint8_t tail[] = {
      0x87,0x02,0x54,0x04,0x08,0x00,
      0xe7,0x06,0x54,0x02,0x00,0x00,0x01,0x4e,0x00,0x00,0x00,0x00,
      0x07,0x02,0x54,0x0c,0x02,0x00,
      0x0e,0x00,0x00,0x00,
   };
   const size_t tail_size = sizeof(tail);
   const uint8_t *program = part.binary ?
      (const uint8_t *)part.binary + part.info.main_offset : NULL;
   const bool protocol = compiled && program && part.info.main_size >=
      sizeof(entry) + sizeof(read_wait) + sizeof(tile_load) + tail_size &&
      !memcmp(program, entry, sizeof(entry)) &&
      !memcmp(program + sizeof(entry), read_wait,
              sizeof(read_wait)) &&
      !memcmp(program + sizeof(entry) + sizeof(read_wait),
              tile_load, sizeof(tile_load)) &&
      !memcmp(program + part.info.main_size - tail_size,
              tail, tail_size);
   /* Keep the hardware-witnessed byte sequence as an anchor. Other modes
    * legitimately change register allocation or eliminate the tile load.
    */
   const bool exact_protocol = op == PIPE_ADVANCED_BLEND_MULTIPLY &&
                               overlap == PIPE_BLEND_OVERLAP_CONJOINT &&
                               !src_premultiplied && !dst_premultiplied &&
                               format == PIPE_FORMAT_R8G8B8A8_UNORM;
   const bool ok = compiled && part.binary && (!exact_protocol || protocol);
   if (!ok)
      fprintf(stderr,
              "G17P advanced blend epilog/protocol failed "
              "(op=%u overlap=%u src_premul=%u dst_premul=%u format=%u)\n",
              op, overlap, src_premultiplied, dst_premultiplied, format);

   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

int
main(void)
{
   unsigned tested = 0;
   static const enum pipe_format formats[] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
   };

   for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
      for (enum pipe_advanced_blend_mode op = PIPE_ADVANCED_BLEND_NONE;
           op <= PIPE_ADVANCED_BLEND_BLUE; ++op) {
         for (enum pipe_blend_overlap_mode overlap =
                 PIPE_BLEND_OVERLAP_UNCORRELATED;
              overlap <= PIPE_BLEND_OVERLAP_DISJOINT; ++overlap) {
            for (unsigned premultiplied = 0; premultiplied < 4;
                 ++premultiplied) {
               const bool src_premultiplied = premultiplied & BITFIELD_BIT(0);
               const bool dst_premultiplied = premultiplied & BITFIELD_BIT(1);

               if (!test_configuration(op, overlap, src_premultiplied,
                                       dst_premultiplied, formats[f]))
                  return 1;

               tested++;
            }
         }
      }
   }

   printf("G17P advanced blend: %u operation/state/format epilogs compiled\n",
          tested);
   return 0;
}
