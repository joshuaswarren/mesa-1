/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "d12_resolve_core.h"

#include "../libagx/resolve.h"

#include <stddef.h>

#define D12_RESOLVE_FLOAT_MODES                                           \
   (D12_RESOLVE_CORE_SUPPORT_MIN | D12_RESOLVE_CORE_SUPPORT_MAX |         \
    D12_RESOLVE_CORE_SUPPORT_AVERAGE)
#define D12_RESOLVE_INTEGER_MODES                                         \
   (D12_RESOLVE_CORE_SUPPORT_MIN | D12_RESOLVE_CORE_SUPPORT_MAX)

#define D12_RESOLVE_INFO(shader, size, modes, value_type)                 \
   ((struct d12_resolve_core_format_info){                                \
      .shader_format = (shader),                                           \
      .pixel_size_B = (size),                                              \
      .mode_mask = (modes),                                                \
      .value_class = (value_type),                                         \
   })

bool
d12_resolve_core_get_format(uint32_t format,
                            struct d12_resolve_core_format_info *info)
{
   struct d12_resolve_core_format_info result;

   switch (format) {
   /* DXGI_FORMAT_R32G32B32A32_{FLOAT,UINT,SINT}. */
   case D12_RESOLVE_CORE_FORMAT_R32G32B32A32_FLOAT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_FLOAT32_4, 16,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R32G32B32A32_UINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UINT32_4, 16,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_UINT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R32G32B32A32_SINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SINT32_4, 16,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_SINT);
      break;

   /* DXGI_FORMAT_R16G16B16A16_{FLOAT,UNORM,UINT,SNORM,SINT}. */
   case D12_RESOLVE_CORE_FORMAT_R16G16B16A16_FLOAT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_FLOAT16_4, 8,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R16G16B16A16_UNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UNORM16_4, 8,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R16G16B16A16_UINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UINT16_4, 8,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_UINT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R16G16B16A16_SNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SNORM16_4, 8,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R16G16B16A16_SINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SINT16_4, 8,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_SINT);
      break;

   /* DXGI_FORMAT_R32G32_{FLOAT,UINT,SINT}. */
   case D12_RESOLVE_CORE_FORMAT_R32G32_FLOAT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_FLOAT32_2, 8,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R32G32_UINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UINT32_2, 8,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_UINT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R32G32_SINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SINT32_2, 8,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_SINT);
      break;

   /* Plane-view formats used to resolve the native split depth/stencil
    * allocations of D32_FLOAT_S8X24_UINT resources. */
   case D12_RESOLVE_CORE_FORMAT_R32_FLOAT_X8X24_TYPELESS:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_FLOAT32_1, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_X32_TYPELESS_G8X24_UINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UINT8_1, 1,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_UINT);
      break;

   /* DXGI_FORMAT_{R10G10B10A2_UNORM,R10G10B10A2_UINT,R11G11B10_FLOAT}. */
   case D12_RESOLVE_CORE_FORMAT_R10G10B10A2_UNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_RGB10A2_UNORM, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R10G10B10A2_UINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_RGB10A2_UINT, 4,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_UINT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R11G11B10_FLOAT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_R11G11B10_FLOAT, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;

   /* DXGI_FORMAT_R8G8B8A8_{UNORM,UNORM_SRGB,UINT,SNORM,SINT}. */
   case D12_RESOLVE_CORE_FORMAT_R8G8B8A8_UNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UNORM8_4, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R8G8B8A8_UNORM_SRGB:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_RGBA8_SRGB, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R8G8B8A8_UINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UINT8_4, 4,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_UINT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R8G8B8A8_SNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SNORM8_4, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R8G8B8A8_SINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SINT8_4, 4,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_SINT);
      break;

   /* DXGI_FORMAT_R16G16_{FLOAT,UNORM,UINT,SNORM,SINT}. */
   case D12_RESOLVE_CORE_FORMAT_R16G16_FLOAT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_FLOAT16_2, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R16G16_UNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UNORM16_2, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R16G16_UINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UINT16_2, 4,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_UINT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R16G16_SNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SNORM16_2, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R16G16_SINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SINT16_2, 4,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_SINT);
      break;

   /* D32_FLOAT and R32_FLOAT share the native scalar-float path. */
   case D12_RESOLVE_CORE_FORMAT_D32_FLOAT:
   case D12_RESOLVE_CORE_FORMAT_R32_FLOAT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_FLOAT32_1, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R32_UINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UINT32_1, 4,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_UINT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R32_SINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SINT32_1, 4,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_SINT);
      break;

   /* The D24 compatibility allocation is physically float32 depth plus a
    * separate uint8 stencil plane, so its plane views use those kernels. */
   case D12_RESOLVE_CORE_FORMAT_R24_UNORM_X8_TYPELESS:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_FLOAT32_1, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_X24_TYPELESS_G8_UINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UINT8_1, 1,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_UINT);
      break;

   /* DXGI_FORMAT_R8G8_{UNORM,UINT,SNORM,SINT}. */
   case D12_RESOLVE_CORE_FORMAT_R8G8_UNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UNORM8_2, 2,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R8G8_UINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UINT8_2, 2,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_UINT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R8G8_SNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SNORM8_2, 2,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R8G8_SINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SINT8_2, 2,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_SINT);
      break;

   /* DXGI_FORMAT_R16_{FLOAT,D16_UNORM,UNORM,UINT,SNORM,SINT}. */
   case D12_RESOLVE_CORE_FORMAT_R16_FLOAT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_FLOAT16_1, 2,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_D16_UNORM:
   case D12_RESOLVE_CORE_FORMAT_R16_UNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UNORM16_1, 2,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R16_UINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UINT16_1, 2,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_UINT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R16_SNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SNORM16_1, 2,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R16_SINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SINT16_1, 2,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_SINT);
      break;

   /* DXGI_FORMAT_R8_{UNORM,UINT,SNORM,SINT} and A8_UNORM. */
   case D12_RESOLVE_CORE_FORMAT_R8_UNORM:
   case D12_RESOLVE_CORE_FORMAT_A8_UNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UNORM8_1, 1,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R8_UINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_UINT8_1, 1,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_UINT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R8_SNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SNORM8_1, 1,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_R8_SINT:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_SINT8_1, 1,
                                D12_RESOLVE_INTEGER_MODES,
                                D12_RESOLVE_CORE_VALUE_SINT);
      break;

   /* Display render-target formats. */
   case D12_RESOLVE_CORE_FORMAT_B5G6R5_UNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_BGR565_UNORM, 2,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_B5G5R5A1_UNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_BGRA5551_UNORM, 2,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_B8G8R8A8_UNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_BGRA8_UNORM, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_B8G8R8X8_UNORM:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_BGRX8_UNORM, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   case D12_RESOLVE_CORE_FORMAT_B8G8R8A8_UNORM_SRGB:
      result = D12_RESOLVE_INFO(LIBAGX_RESOLVE_FORMAT_BGRA8_SRGB, 4,
                                D12_RESOLVE_FLOAT_MODES,
                                D12_RESOLVE_CORE_VALUE_FLOAT);
      break;
   default:
      return false;
   }

   if (info)
      *info = result;
   return true;
}

bool
d12_resolve_core_supports_mode(uint32_t format, uint32_t mode)
{
   struct d12_resolve_core_format_info info;
   uint32_t bit;

   if (!d12_resolve_core_get_format(format, &info))
      return false;

   switch (mode) {
   case D12_RESOLVE_CORE_MODE_MIN:
      bit = D12_RESOLVE_CORE_SUPPORT_MIN;
      break;
   case D12_RESOLVE_CORE_MODE_MAX:
      bit = D12_RESOLVE_CORE_SUPPORT_MAX;
      break;
   case D12_RESOLVE_CORE_MODE_AVERAGE:
      bit = D12_RESOLVE_CORE_SUPPORT_AVERAGE;
      break;
   default:
      return false;
   }

   return (info.mode_mask & bit) != 0;
}

bool
d12_resolve_core_supports_average(uint32_t format)
{
   return d12_resolve_core_supports_mode(format,
                                         D12_RESOLVE_CORE_MODE_AVERAGE);
}

bool
d12_resolve_core_resource_compatible(uint32_t resource_format,
                                     uint32_t resolve_format)
{
   if (!d12_resolve_core_get_format(resolve_format, NULL))
      return false;
   if (resource_format == resolve_format)
      return true;

   /* The native runtime accepts casting inside a format family. The
    * resolve format supplies the interpretation, including SRGB decode and
    * encode. A typed allocation does not override that explicit view. */
   static const struct {
      uint32_t first, last;
   } families[] = {
      {D12_RESOLVE_CORE_FORMAT_R32G32B32A32_TYPELESS,
       D12_RESOLVE_CORE_FORMAT_R32G32B32A32_SINT},
      {D12_RESOLVE_CORE_FORMAT_R16G16B16A16_TYPELESS,
       D12_RESOLVE_CORE_FORMAT_R16G16B16A16_SINT},
      {D12_RESOLVE_CORE_FORMAT_R32G32_TYPELESS,
       D12_RESOLVE_CORE_FORMAT_R32G32_SINT},
      {D12_RESOLVE_CORE_FORMAT_R10G10B10A2_TYPELESS,
       D12_RESOLVE_CORE_FORMAT_R10G10B10A2_UINT},
      {D12_RESOLVE_CORE_FORMAT_R8G8B8A8_TYPELESS,
       D12_RESOLVE_CORE_FORMAT_R8G8B8A8_SINT},
      {D12_RESOLVE_CORE_FORMAT_R16G16_TYPELESS,
       D12_RESOLVE_CORE_FORMAT_R16G16_SINT},
      {D12_RESOLVE_CORE_FORMAT_R32_TYPELESS, D12_RESOLVE_CORE_FORMAT_R32_SINT},
      {D12_RESOLVE_CORE_FORMAT_R8G8_TYPELESS,
       D12_RESOLVE_CORE_FORMAT_R8G8_SINT},
      {D12_RESOLVE_CORE_FORMAT_R16_TYPELESS, D12_RESOLVE_CORE_FORMAT_R16_SINT},
      {D12_RESOLVE_CORE_FORMAT_R8_TYPELESS, D12_RESOLVE_CORE_FORMAT_R8_SINT},
   };
   for (unsigned i = 0; i < sizeof(families) / sizeof(families[0]); ++i) {
      if (resource_format >= families[i].first &&
          resource_format <= families[i].last) {
         resource_format = families[i].first;
         break;
      }
   }
   if (resource_format == D12_RESOLVE_CORE_FORMAT_B8G8R8A8_UNORM ||
       resource_format == D12_RESOLVE_CORE_FORMAT_B8G8R8A8_UNORM_SRGB)
      resource_format = D12_RESOLVE_CORE_FORMAT_B8G8R8A8_TYPELESS;
   if (resource_format == D12_RESOLVE_CORE_FORMAT_B8G8R8X8_UNORM)
      resource_format = D12_RESOLVE_CORE_FORMAT_B8G8R8X8_TYPELESS;

   switch (resource_format) {
   case D12_RESOLVE_CORE_FORMAT_R32G32B32A32_TYPELESS:
      return resolve_format >= D12_RESOLVE_CORE_FORMAT_R32G32B32A32_FLOAT &&
             resolve_format <= D12_RESOLVE_CORE_FORMAT_R32G32B32A32_SINT;
   case D12_RESOLVE_CORE_FORMAT_R16G16B16A16_TYPELESS:
      return resolve_format >= D12_RESOLVE_CORE_FORMAT_R16G16B16A16_FLOAT &&
             resolve_format <= D12_RESOLVE_CORE_FORMAT_R16G16B16A16_SINT;
   case D12_RESOLVE_CORE_FORMAT_R32G32_TYPELESS:
      return resolve_format >= D12_RESOLVE_CORE_FORMAT_R32G32_FLOAT &&
             resolve_format <= D12_RESOLVE_CORE_FORMAT_R32G32_SINT;
   case D12_RESOLVE_CORE_FORMAT_R32G8X24_TYPELESS:
   case D12_RESOLVE_CORE_FORMAT_D32_FLOAT_S8X24_UINT:
      return resolve_format ==
                D12_RESOLVE_CORE_FORMAT_R32_FLOAT_X8X24_TYPELESS ||
             resolve_format ==
                D12_RESOLVE_CORE_FORMAT_X32_TYPELESS_G8X24_UINT;
   case D12_RESOLVE_CORE_FORMAT_R10G10B10A2_TYPELESS:
      return resolve_format == D12_RESOLVE_CORE_FORMAT_R10G10B10A2_UNORM ||
             resolve_format == D12_RESOLVE_CORE_FORMAT_R10G10B10A2_UINT;
   case D12_RESOLVE_CORE_FORMAT_R8G8B8A8_TYPELESS:
      return resolve_format >= D12_RESOLVE_CORE_FORMAT_R8G8B8A8_UNORM &&
             resolve_format <= D12_RESOLVE_CORE_FORMAT_R8G8B8A8_SINT;
   case D12_RESOLVE_CORE_FORMAT_R16G16_TYPELESS:
      return resolve_format >= D12_RESOLVE_CORE_FORMAT_R16G16_FLOAT &&
             resolve_format <= D12_RESOLVE_CORE_FORMAT_R16G16_SINT;
   case D12_RESOLVE_CORE_FORMAT_R32_TYPELESS:
      return resolve_format >= D12_RESOLVE_CORE_FORMAT_D32_FLOAT &&
             resolve_format <= D12_RESOLVE_CORE_FORMAT_R32_SINT;
   case D12_RESOLVE_CORE_FORMAT_R24G8_TYPELESS:
   case D12_RESOLVE_CORE_FORMAT_D24_UNORM_S8_UINT:
      return resolve_format ==
                D12_RESOLVE_CORE_FORMAT_R24_UNORM_X8_TYPELESS ||
             resolve_format == D12_RESOLVE_CORE_FORMAT_X24_TYPELESS_G8_UINT;
   case D12_RESOLVE_CORE_FORMAT_R8G8_TYPELESS:
      return resolve_format >= D12_RESOLVE_CORE_FORMAT_R8G8_UNORM &&
             resolve_format <= D12_RESOLVE_CORE_FORMAT_R8G8_SINT;
   case D12_RESOLVE_CORE_FORMAT_R16_TYPELESS:
      return resolve_format >= D12_RESOLVE_CORE_FORMAT_R16_FLOAT &&
             resolve_format <= D12_RESOLVE_CORE_FORMAT_R16_SINT;
   case D12_RESOLVE_CORE_FORMAT_R8_TYPELESS:
      return resolve_format >= D12_RESOLVE_CORE_FORMAT_R8_UNORM &&
             resolve_format <= D12_RESOLVE_CORE_FORMAT_R8_SINT;
   case D12_RESOLVE_CORE_FORMAT_B8G8R8A8_TYPELESS:
      return resolve_format == D12_RESOLVE_CORE_FORMAT_B8G8R8A8_UNORM ||
             resolve_format ==
                D12_RESOLVE_CORE_FORMAT_B8G8R8A8_UNORM_SRGB;
   case D12_RESOLVE_CORE_FORMAT_B8G8R8X8_TYPELESS:
      return resolve_format == D12_RESOLVE_CORE_FORMAT_B8G8R8X8_UNORM;
   default:
      return false;
   }
}
