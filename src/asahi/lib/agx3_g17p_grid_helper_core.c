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

const char *
agx3_g17p_grid_helper_source(void)
{
   return agx3_g17p_grid_helper_msl;
}
