/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_code_container_core.h"

#include <string.h>

size_t
agx3_code_block_size(size_t main_span)
{
   if (main_span == 0 || main_span > UINT32_MAX - 0xbf)
      return 0;

   return AGX3_CODE_BLOCK_ENTRY_OFFSET + ((main_span + 0x3f) & ~(size_t)0x3f);
}

bool
agx3_build_code_block(void *image, size_t capacity, const void *main,
                      size_t main_size, size_t main_span,
                      const uint8_t *constant_program,
                      bool fragment)
{
   const size_t block_size = agx3_code_block_size(main_span);
   if (!image || !main || !main_size || main_size > main_span || !block_size ||
       capacity < block_size)
      return false;

   uint8_t *bytes = image;
   memset(bytes, 0, block_size);
   for (unsigned i = 0; i < sizeof(uint32_t); ++i)
      bytes[i] = (uint8_t)(block_size >> (8 * i));

   uint8_t *constant = bytes + AGX3_CODE_BLOCK_HEADER_SIZE;
   if (constant_program)
      memcpy(constant, constant_program, AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE);
   else
      agx3_build_container_constant_program(constant, fragment, false);

   memcpy(bytes + AGX3_CODE_BLOCK_ENTRY_OFFSET, main, main_size);
   return true;
}

void
agx3_build_container_constant_program(
   uint8_t image[AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE], bool fragment,
   bool sampled_fragment)
{
   static const uint8_t fragment_stop[] = {
      0x0e, 0x00, 0x00, 0x00,
   };
   static const uint8_t sampled_fragment_program[] = {
      0x03, 0x00, 0x07, 0x00, 0x02, 0x00, 0x00,
      0x00, 0x20, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   static const uint8_t vertex_compute[] = {
      0x03, 0x00, 0x07, 0x00, 0x02, 0x00, 0x00,
      0x00, 0x60, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   const uint8_t *program =
      !fragment ? vertex_compute
                : sampled_fragment ? sampled_fragment_program : fragment_stop;
   const size_t program_size =
      !fragment ? sizeof(vertex_compute)
                : sampled_fragment ? sizeof(sampled_fragment_program)
                                   : sizeof(fragment_stop);

   memcpy(image, program, program_size);
   for (size_t offset = program_size;
        offset < AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE; offset += 2) {
      image[offset] = 0x06;
      image[offset + 1] = 0x00;
   }
}
