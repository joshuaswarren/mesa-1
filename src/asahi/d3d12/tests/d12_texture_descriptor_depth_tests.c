/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Pack production descriptors entirely on the CPU. No device is opened.
 */

#include "../d12_private.h"
#include <assert.h>
#include <stdio.h>

static void
check(D3D12_SRV_DIMENSION dimension, unsigned view_layers,
      unsigned expected_depth)
{
   struct d12_device device = {0};
   device.dev.chip = AGX_CHIP_G13X;
   struct agx_bo bo = {0};
   struct d12_resource resource = {0};
   resource.device = &device;
   resource.bo = &bo;
   resource.desc.Dimension = dimension == D3D12_SRV_DIMENSION_TEXTURE3D
      ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
   resource.layout = (struct ail_layout){
      .width_px = 64, .height_px = 64, .depth_px = 16,
      .levels = 4, .sample_count_sa = 1, .size_B = 65536,
      .tiling = AIL_TILING_TWIDDLED,
   };
   struct d12_descriptor descriptor = {0};
   descriptor.kind = D12_DESC_SRV;
   descriptor.resource = &resource;
   descriptor.address = 0x100000;
   descriptor.format = DXGI_FORMAT_R8_UNORM;
   descriptor.texture_srv_dimension = dimension;
   descriptor.texture_srv_layers = view_layers;
   descriptor.texture_first_level = 1;
   descriptor.texture_level_count = 3;
   descriptor.texture_component_mapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

   struct agx_texture_packed packed;
   assert(d12_refresh_texture_descriptor(&device, &packed, &descriptor));
   struct AGX_TEXTURE unpacked;
   assert(AGX_TEXTURE_unpack(NULL, (uint8_t *)&packed, &unpacked));
   assert(unpacked.depth == expected_depth);
   assert(unpacked.first_level == 1);
   assert(unpacked.last_level == 3);
}

int
main(void)
{
   check(D3D12_SRV_DIMENSION_TEXTURE3D, 1, 16);
   check(D3D12_SRV_DIMENSION_TEXTURE2DARRAY, 3, 3);
   check(D3D12_SRV_DIMENSION_TEXTURECUBEARRAY, 12, 2);
   puts("d12_texture_descriptor_depth_tests: PASS");
   return 0;
}
