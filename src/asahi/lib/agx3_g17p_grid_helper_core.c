/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_g17p_grid_helper_core.h"

#include <string.h>

/*
 * `xcrun metal-nt -arch applegpu_g17p` output for agx3_g17p_grid_helper_source().
 * SHA-256 a7b47ac674d55766863a0b762cc23576155c25aec5e9f69673b4e29700d8c9c7.
 * Byte-identical to the tail of m1n1's captured G17P GRID_SETUP image.
 */
static const uint8_t agx3_g17p_grid_helper_code[
   AGX3_G17P_GRID_HELPER_SHADER_SIZE] = {
   0x2c, 0x8c, 0x09, 0x27, 0x60, 0x00, 0x00, 0x80,
   0x3c, 0x8e, 0x08, 0x27, 0x0c, 0x88, 0x08, 0x47,
   0x1c, 0x8a, 0x08, 0x47, 0x67, 0x20, 0x54, 0x08,
   0x02, 0x00, 0x00, 0x00, 0x1d, 0x04, 0x00, 0x40,
   0x22, 0x00, 0x67, 0x40, 0x44, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x5d, 0x04, 0x00, 0x40, 0x22, 0x00,
   0x6b, 0x88, 0x01, 0x24, 0x9f, 0x20, 0x54, 0x12,
   0x00, 0x10, 0x00, 0x00, 0xd0, 0x26, 0x0a, 0x00,
   0x7b, 0x8a, 0x01, 0x04, 0x9f, 0x00, 0x54, 0x14,
   0x00, 0x14, 0x08, 0x00, 0xd0, 0x26, 0x0a, 0x00,
   0x8b, 0x8c, 0x01, 0x04, 0x9f, 0x00, 0x54, 0x16,
   0x00, 0x18, 0x10, 0x00, 0xd0, 0x26, 0x0a, 0x00,
   0x0e,
};

static const char agx3_g17p_grid_helper_msl[] =
   "#include <metal_stdlib>\n"
   "using namespace metal;\n"
   "\n"
   "kernel void g17p_grid_setup(device const uint *threadgroups [[buffer(0)]],\n"
   "                            device const uint *threads_per_group [[buffer(1)]],\n"
   "                            device uint *geometry [[buffer(2)]],\n"
   "                            uint index [[thread_position_in_grid]])\n"
   "{\n"
   "    if (index != 0)\n"
   "        return;\n"
   "\n"
   "    geometry[0] = threadgroups[0] * threads_per_group[0];\n"
   "    geometry[1] = threadgroups[1] * threads_per_group[1];\n"
   "    geometry[2] = threadgroups[2] * threads_per_group[2];\n"
   "    geometry[3] = threads_per_group[0];\n"
   "    geometry[4] = threads_per_group[1];\n"
   "    geometry[5] = threads_per_group[2];\n"
   "}\n";

const uint8_t *
agx3_g17p_grid_helper_shader_image(void)
{
   return agx3_g17p_grid_helper_code;
}

const char *
agx3_g17p_grid_helper_source(void)
{
   return agx3_g17p_grid_helper_msl;
}

bool
agx3_g17p_indirect_geometry_from_groups(
   const uint32_t groups[3], const uint32_t local_size[3],
   struct agx3_g17p_indirect_geometry *out)
{
   if (!groups || !local_size || !out)
      return false;

   struct agx3_g17p_indirect_geometry staged;
   for (unsigned axis = 0; axis < 3; ++axis) {
      if (groups[axis] == 0 || local_size[axis] == 0)
         return false;

      /* The helper's multiply is a 32-bit wrapping multiply.  Refuse rather
       * than reproduce a wrapped product: the driver cannot check the GPU's
       * answer, so an overflowing dispatch must never be recorded. */
      const uint64_t global =
         (uint64_t)groups[axis] * (uint64_t)local_size[axis];
      if (global > UINT32_MAX)
         return false;

      staged.global_size[axis] = (uint32_t)global;
      staged.local_size[axis] = local_size[axis];
   }

   *out = staged;
   return true;
}

static void
agx3_g17p_store_le32(uint8_t *out, uint32_t value)
{
   out[0] = (uint8_t)(value & 0xff);
   out[1] = (uint8_t)((value >> 8) & 0xff);
   out[2] = (uint8_t)((value >> 16) & 0xff);
   out[3] = (uint8_t)((value >> 24) & 0xff);
}

static uint32_t
agx3_g17p_load_le32(const uint8_t *in)
{
   return (uint32_t)in[0] | ((uint32_t)in[1] << 8) |
          ((uint32_t)in[2] << 16) | ((uint32_t)in[3] << 24);
}

bool
agx3_g17p_indirect_geometry_pack(
   const struct agx3_g17p_indirect_geometry *geometry,
   uint8_t out[AGX3_G17P_INDIRECT_GEOMETRY_SIZE])
{
   if (!geometry || !out)
      return false;

   for (unsigned axis = 0; axis < 3; ++axis) {
      if (geometry->global_size[axis] == 0 || geometry->local_size[axis] == 0)
         return false;
   }

   uint8_t staged[AGX3_G17P_INDIRECT_GEOMETRY_SIZE];
   for (unsigned axis = 0; axis < 3; ++axis) {
      agx3_g17p_store_le32(staged + axis * 4, geometry->global_size[axis]);
      agx3_g17p_store_le32(staged + 12 + axis * 4, geometry->local_size[axis]);
   }

   memcpy(out, staged, sizeof(staged));
   return true;
}

bool
agx3_g17p_indirect_geometry_unpack(
   const uint8_t in[AGX3_G17P_INDIRECT_GEOMETRY_SIZE],
   struct agx3_g17p_indirect_geometry *out)
{
   if (!in || !out)
      return false;

   struct agx3_g17p_indirect_geometry staged;
   for (unsigned axis = 0; axis < 3; ++axis) {
      staged.global_size[axis] = agx3_g17p_load_le32(in + axis * 4);
      staged.local_size[axis] = agx3_g17p_load_le32(in + 12 + axis * 4);
      if (staged.global_size[axis] == 0 || staged.local_size[axis] == 0)
         return false;
   }

   *out = staged;
   return true;
}
