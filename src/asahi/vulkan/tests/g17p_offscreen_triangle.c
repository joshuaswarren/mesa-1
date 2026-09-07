/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Display-independent Vulkan graphics proof for T8140/G17P. Render one
 * full-screen triangle into a one-tile image and verify every byte.
 *
 * The default build uses vkCmdCopyImageToBuffer and intentionally exercises
 * Honeykrisp's GPU meta-copy path.  g17p-hostcopy-triangle compiles this same
 * render proof with G17P_HOST_IMAGE_COPY set: it keeps the optimal tiled render
 * target but reads it back on the CPU through VK_EXT_host_image_copy.  Keeping
 * the paths as separate binaries lets render bring-up prove TA/3D/PBE output
 * without hiding regressions in the eventual GPU copy implementation.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef G17P_F16_IO
#define G17P_F16_IO 0
#endif

#ifndef G17P_RG16_UNORM
#define G17P_RG16_UNORM 0
#endif

#ifndef G17P_RGBA16_UNORM
#define G17P_RGBA16_UNORM 0
#endif

#ifndef G17P_RG16_SNORM
#define G17P_RG16_SNORM 0
#endif

#ifndef G17P_R32_UINT
#define G17P_R32_UINT 0
#endif

#ifndef G17P_R32_SFLOAT
#define G17P_R32_SFLOAT 0
#endif

#ifndef G17P_R32_SINT
#define G17P_R32_SINT 0
#endif

#ifndef G17P_RG16_UINT
#define G17P_RG16_UINT 0
#endif

#ifndef G17P_RG16_SINT
#define G17P_RG16_SINT 0
#endif

#ifndef G17P_RG32_SINT
#define G17P_RG32_SINT 0
#endif

#ifndef G17P_RG32_SFLOAT
#define G17P_RG32_SFLOAT 0
#endif

#ifndef G17P_RG8_UINT
#define G17P_RG8_UINT 0
#endif

#ifndef G17P_RG8_SINT
#define G17P_RG8_SINT 0
#endif

#ifndef G17P_RG32_UINT
#define G17P_RG32_UINT 0
#endif

#ifndef G17P_RGBA8_UINT
#define G17P_RGBA8_UINT 0
#endif

#ifndef G17P_RGBA8_SINT
#define G17P_RGBA8_SINT 0
#endif

#ifndef G17P_RGBA8_SNORM
#define G17P_RGBA8_SNORM 0
#endif

#ifndef G17P_RG8_SNORM
#define G17P_RG8_SNORM 0
#endif

#ifndef G17P_R8_UINT
#define G17P_R8_UINT 0
#endif

#ifndef G17P_R8_SNORM
#define G17P_R8_SNORM 0
#endif

#ifndef G17P_R8_SINT
#define G17P_R8_SINT 0
#endif

#ifndef G17P_RGBA16_SNORM
#define G17P_RGBA16_SNORM 0
#endif

#ifndef G17P_R16_SNORM
#define G17P_R16_SNORM 0
#endif

#ifndef G17P_R16_UNORM
#define G17P_R16_UNORM 0
#endif

#if (G17P_F16_IO + G17P_RG16_UNORM + G17P_RGBA16_UNORM +              \
     G17P_RG16_SNORM + G17P_R32_UINT + G17P_R32_SFLOAT +              \
     G17P_R32_SINT + G17P_RG16_UINT + G17P_RG16_SINT +                \
     G17P_RG32_SINT + G17P_RG32_SFLOAT + G17P_RG8_UINT +              \
     G17P_RG8_SINT + G17P_RG32_UINT + G17P_RGBA8_UINT +               \
     G17P_RGBA8_SINT + G17P_RGBA8_SNORM + G17P_RG8_SNORM +             \
     G17P_R8_UINT + G17P_R8_SNORM + G17P_R8_SINT + G17P_RGBA16_SNORM + \
     G17P_R16_SNORM + G17P_R16_UNORM) > 1
#error "select exactly one guarded G17P oracle"
#endif

#if G17P_R16_UNORM
#include "g17p_vulkan_r16_unorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r16_unorm_frag_spv
#elif G17P_R16_SNORM
#include "g17p_vulkan_r16_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r16_snorm_frag_spv
#elif G17P_RGBA16_SNORM
#include "g17p_vulkan_rgba16_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba16_snorm_frag_spv
#elif G17P_R8_SINT
#include "g17p_vulkan_r8_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r8_sint_frag_spv
#elif G17P_R8_SNORM
#include "g17p_vulkan_r8_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r8_snorm_frag_spv
#elif G17P_R8_UINT
#include "g17p_vulkan_r8_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r8_uint_frag_spv
#elif G17P_RG8_SNORM
#include "g17p_vulkan_rg8_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg8_snorm_frag_spv
#elif G17P_RGBA8_SNORM
#include "g17p_vulkan_rgba8_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba8_snorm_frag_spv
#elif G17P_RGBA8_SINT
#include "g17p_vulkan_rgba8_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba8_sint_frag_spv
#elif G17P_RGBA8_UINT
#include "g17p_vulkan_rgba8_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rgba8_uint_frag_spv
#elif G17P_RG8_SINT
#include "g17p_vulkan_rg8_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg8_sint_frag_spv
#elif G17P_RG8_UINT
#include "g17p_vulkan_rg8_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg8_uint_frag_spv
#elif G17P_RG32_UINT
#include "g17p_vulkan_rg32_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg32_uint_frag_spv
#elif G17P_R32_UINT
#include "g17p_vulkan_r32_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r32_uint_frag_spv
#elif G17P_R32_SINT
#include "g17p_vulkan_r32_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r32_sint_frag_spv
#elif G17P_RG32_SINT
#include "g17p_vulkan_rg32_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg32_sint_frag_spv
#elif G17P_RG32_SFLOAT
#include "g17p_vulkan_rg32_sfloat_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg32_sfloat_frag_spv
#elif G17P_RG16_UINT
#include "g17p_vulkan_rg16_uint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg16_uint_frag_spv
#elif G17P_RG16_SINT
#include "g17p_vulkan_rg16_sint_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg16_sint_frag_spv
#elif G17P_F16_IO
#include "g17p_vulkan_f16_io_frag.h"
#include "g17p_vulkan_f16_io_vert.h"
#define G17P_VERT_SPV g17p_vulkan_f16_io_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_f16_io_frag_spv
#elif G17P_R32_SFLOAT
#include "g17p_vulkan_r32_sfloat_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_r32_sfloat_frag_spv
#elif G17P_RG16_SNORM
#include "g17p_vulkan_rg16_snorm_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_vulkan_rg16_snorm_frag_spv
#else
#include "g17p_wsi_triangle_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERT_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAG_SPV g17p_wsi_triangle_frag_spv
#endif

#define T8140_DEVICE_ID 0x8140
#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32

#if G17P_R16_UNORM
#define IMAGE_FORMAT VK_FORMAT_R16_UNORM
#define IMAGE_PIXEL_BYTES 2
#elif G17P_R16_SNORM
#define IMAGE_FORMAT VK_FORMAT_R16_SNORM
#define IMAGE_PIXEL_BYTES 2
#elif G17P_RGBA16_SNORM
#define IMAGE_FORMAT VK_FORMAT_R16G16B16A16_SNORM
#define IMAGE_PIXEL_BYTES 8
#elif G17P_R8_SINT
#define IMAGE_FORMAT VK_FORMAT_R8_SINT
#define IMAGE_PIXEL_BYTES 1
#elif G17P_R8_SNORM
#define IMAGE_FORMAT VK_FORMAT_R8_SNORM
#define IMAGE_PIXEL_BYTES 1
#define READBACK_GUARD_BYTES 64
#elif G17P_R8_UINT
#define IMAGE_FORMAT VK_FORMAT_R8_UINT
#define IMAGE_PIXEL_BYTES 1
#elif G17P_RG8_SNORM
#define IMAGE_FORMAT VK_FORMAT_R8G8_SNORM
#define IMAGE_PIXEL_BYTES 2
#define READBACK_GUARD_BYTES 64
#elif G17P_RGBA8_SNORM
#define IMAGE_FORMAT VK_FORMAT_R8G8B8A8_SNORM
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RGBA8_SINT
#define IMAGE_FORMAT VK_FORMAT_R8G8B8A8_SINT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RGBA8_UINT
#define IMAGE_FORMAT VK_FORMAT_R8G8B8A8_UINT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RG8_SINT
#define IMAGE_FORMAT VK_FORMAT_R8G8_SINT
#define IMAGE_PIXEL_BYTES 2
#elif G17P_RG8_UINT
#define IMAGE_FORMAT VK_FORMAT_R8G8_UINT
#define IMAGE_PIXEL_BYTES 2
#elif G17P_RGBA16_UNORM
#define IMAGE_FORMAT VK_FORMAT_R16G16B16A16_UNORM
#define IMAGE_PIXEL_BYTES 8
#elif G17P_RG16_SNORM
#define IMAGE_FORMAT VK_FORMAT_R16G16_SNORM
#define IMAGE_PIXEL_BYTES 4
#elif G17P_R32_SINT
#define IMAGE_FORMAT VK_FORMAT_R32_SINT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RG32_SINT
#define IMAGE_FORMAT VK_FORMAT_R32G32_SINT
#define IMAGE_PIXEL_BYTES 8
#elif G17P_RG32_SFLOAT
#define IMAGE_FORMAT VK_FORMAT_R32G32_SFLOAT
#define IMAGE_PIXEL_BYTES 8
#elif G17P_R32_UINT
#define IMAGE_FORMAT VK_FORMAT_R32_UINT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RG32_UINT
#define IMAGE_FORMAT VK_FORMAT_R32G32_UINT
#define IMAGE_PIXEL_BYTES 8
#elif G17P_R32_SFLOAT
#define IMAGE_FORMAT VK_FORMAT_R32_SFLOAT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RG16_UINT
#define IMAGE_FORMAT VK_FORMAT_R16G16_UINT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RG16_SINT
#define IMAGE_FORMAT VK_FORMAT_R16G16_SINT
#define IMAGE_PIXEL_BYTES 4
#elif G17P_RG16_UNORM
#define IMAGE_FORMAT VK_FORMAT_R16G16_UNORM
#define IMAGE_PIXEL_BYTES 4
#else
#define IMAGE_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define IMAGE_PIXEL_BYTES 4
#endif

#define IMAGE_BYTES (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_PIXEL_BYTES)

#ifndef G17P_HOST_IMAGE_COPY
#define G17P_HOST_IMAGE_COPY 0
#endif

#ifndef G17P_MSAA_SAMPLES
#define G17P_MSAA_SAMPLES 1
#endif

#if G17P_MSAA_SAMPLES != 1 && G17P_MSAA_SAMPLES != 2
#error "the G17P qualification fixture covers only 1x and native-default 2x"
#endif

#if G17P_HOST_IMAGE_COPY && G17P_MSAA_SAMPLES != 1
#error "the MSAA fixture validates the GPU resolve and copy path"
#endif

#if G17P_F16_IO && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the f16 I/O fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG16_UNORM && (G17P_F16_IO || G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG16 fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RGBA16_UNORM &&                                               \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RG16_SNORM ||              \
    G17P_R32_UINT || G17P_R32_SINT || G17P_RG32_SFLOAT ||             \
    G17P_HOST_IMAGE_COPY ||                                             \
    G17P_MSAA_SAMPLES != 1)
#error "the RGBA16 fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG16_SNORM &&                                                   \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_R32_UINT || G17P_R32_SINT || G17P_RG32_SFLOAT ||              \
    G17P_HOST_IMAGE_COPY ||                                              \
    G17P_MSAA_SAMPLES != 1)
#error "the RG16 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R32_UINT &&                                                     \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_RG16_SNORM || G17P_R32_SINT || G17P_RG32_SFLOAT ||            \
    G17P_HOST_IMAGE_COPY ||                                              \
    G17P_MSAA_SAMPLES != 1)
#error "the R32 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R32_SFLOAT &&                                                   \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_RG16_SNORM || G17P_R32_UINT || G17P_R32_SINT ||               \
    G17P_RG32_SFLOAT || G17P_HOST_IMAGE_COPY ||                         \
    G17P_MSAA_SAMPLES != 1)
#error "the R32 SFLOAT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R32_SINT &&                                                     \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_RG16_SNORM || G17P_R32_UINT || G17P_R32_SFLOAT ||             \
    G17P_RG32_SFLOAT || G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R32 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG16_UINT &&                                                    \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_RG16_SNORM || G17P_R32_UINT || G17P_R32_SFLOAT ||             \
    G17P_R32_SINT || G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG16 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG16_SINT &&                                                    \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_RG16_SNORM || G17P_R32_UINT || G17P_R32_SFLOAT ||             \
    G17P_R32_SINT || G17P_RG16_UINT || G17P_HOST_IMAGE_COPY ||         \
    G17P_MSAA_SAMPLES != 1)
#error "the RG16 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG32_SINT &&                                                    \
   (G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||             \
    G17P_RG16_SNORM || G17P_R32_UINT || G17P_R32_SFLOAT ||             \
    G17P_R32_SINT || G17P_RG16_UINT || G17P_RG16_SINT ||               \
    G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG32 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG32_SFLOAT &&                                                  \
   (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG32 SFLOAT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG8_UINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG8 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG32_UINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG32 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG8_SINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG8 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if (G17P_RGBA8_UINT || G17P_RGBA8_SINT) &&                           \
   (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RGBA8 integer fixtures are guarded single-sample GPU-copy oracles"
#endif

#if G17P_RGBA8_SNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RGBA8 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RG8_SNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RG8 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R8_UINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R8 UINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R8_SNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R8 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R8_SINT && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R8 SINT fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_RGBA16_SNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the RGBA16 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R16_SNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R16 SNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R16_UNORM && (G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES != 1)
#error "the R16 UNORM fixture is a guarded single-sample GPU-copy oracle"
#endif

#if G17P_R16_UNORM
#define PROBE_NAME "g17p-vulkan-r16-unorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R16_SNORM
#define PROBE_NAME "g17p-vulkan-r16-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA16_SNORM
#define PROBE_NAME "g17p-vulkan-rgba16-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R8_SINT
#define PROBE_NAME "g17p-vulkan-r8-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R8_SNORM
#define PROBE_NAME "g17p-vulkan-r8-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R8_UINT
#define PROBE_NAME "g17p-vulkan-r8-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG8_SNORM
#define PROBE_NAME "g17p-vulkan-rg8-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA8_SNORM
#define PROBE_NAME "g17p-vulkan-rgba8-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA8_SINT
#define PROBE_NAME "g17p-vulkan-rgba8-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA8_UINT
#define PROBE_NAME "g17p-vulkan-rgba8-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG8_SINT
#define PROBE_NAME "g17p-vulkan-rg8-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG8_UINT
#define PROBE_NAME "g17p-vulkan-rg8-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RGBA16_UNORM
#define PROBE_NAME "g17p-vulkan-rgba16-unorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG16_SNORM
#define PROBE_NAME "g17p-vulkan-rg16-snorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R32_SINT
#define PROBE_NAME "g17p-vulkan-r32-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG32_SINT
#define PROBE_NAME "g17p-vulkan-rg32-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG32_SFLOAT
#define PROBE_NAME "g17p-vulkan-rg32-sfloat"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R32_UINT
#define PROBE_NAME "g17p-vulkan-r32-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG32_UINT
#define PROBE_NAME "g17p-vulkan-rg32-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_R32_SFLOAT
#define PROBE_NAME "g17p-vulkan-r32-sfloat"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG16_UINT
#define PROBE_NAME "g17p-vulkan-rg16-uint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG16_SINT
#define PROBE_NAME "g17p-vulkan-rg16-sint"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_RG16_UNORM
#define PROBE_NAME "g17p-vulkan-rg16-unorm"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_F16_IO
#define PROBE_NAME "g17p-vulkan-f16-io"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_MSAA_SAMPLES == 2
#define PROBE_NAME "g17p-msaa2-triangle"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 64
#define READBACK_BYTES (IMAGE_BYTES + 2 * READBACK_GUARD_BYTES)
#elif G17P_HOST_IMAGE_COPY
#define PROBE_NAME "g17p-hostcopy-triangle"
#define READBACK_IMAGE_USAGE VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT
#define READBACK_GUARD_BYTES 0
#define READBACK_BYTES IMAGE_BYTES
#else
#define PROBE_NAME "g17p-offscreen-triangle"
#define READBACK_IMAGE_USAGE 0
#define READBACK_GUARD_BYTES 0
#define READBACK_BYTES IMAGE_BYTES
#endif

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkEnumeratePhysicalDevices);
#if G17P_F16_IO
VK_FN(vkGetPhysicalDeviceFeatures2);
#endif
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkGetPhysicalDeviceFormatProperties);
VK_FN(vkCreateDevice);
VK_FN(vkGetDeviceQueue);
VK_FN(vkCreateBuffer);
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkCreateImage);
VK_FN(vkGetImageMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkBindBufferMemory);
VK_FN(vkBindImageMemory);
VK_FN(vkMapMemory);
VK_FN(vkCreateImageView);
VK_FN(vkCreateRenderPass);
VK_FN(vkCreateShaderModule);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkCreateGraphicsPipelines);
VK_FN(vkCreateFramebuffer);
VK_FN(vkCreateCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdBeginRenderPass);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindVertexBuffers);
VK_FN(vkCmdDraw);
VK_FN(vkCmdEndRenderPass);
#if G17P_HOST_IMAGE_COPY
VK_FN(vkCopyImageToMemoryEXT);
#else
VK_FN(vkCmdCopyImageToBuffer);
#endif
VK_FN(vkEndCommandBuffer);
VK_FN(vkCreateFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);

static void
die(const char *what)
{
   fprintf(stderr, PROBE_NAME ": %s\n", what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, PROBE_NAME ": %s returned %d\n", what, result);
      exit(EXIT_FAILURE);
   }
}

#define LOAD_GLOBAL(name)                                                      \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);         \
      if (!name)                                                               \
         die("loader is missing " #name);                                     \
   } while (0)

#define LOAD_INSTANCE(instance, name)                                          \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);               \
      if (!name)                                                               \
         die("instance is missing " #name);                                   \
   } while (0)

#define LOAD_DEVICE(device, name)                                              \
   do {                                                                        \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                   \
      if (!name)                                                               \
         die("device is missing " #name);                                     \
   } while (0)

static uint32_t
find_memory_type(const VkPhysicalDeviceMemoryProperties *properties,
                 uint32_t allowed, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((allowed & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }

   die("required Vulkan memory type is unavailable");
   return UINT32_MAX;
}

static uint64_t
fnv1a64(const uint8_t *bytes, size_t size)
{
   uint64_t hash = UINT64_C(14695981039346656037);
   for (size_t i = 0; i < size; ++i) {
      hash ^= bytes[i];
      hash *= UINT64_C(1099511628211);
   }
   return hash;
}

static bool
verify_readback(const uint8_t *map)
{
   for (size_t i = 0; i < READBACK_GUARD_BYTES; ++i) {
      if (map[i] != 0xa5 ||
          map[READBACK_GUARD_BYTES + IMAGE_BYTES + i] != 0xa5) {
         fprintf(stderr, PROBE_NAME ": readback guard %zu was modified\n", i);
         return false;
      }
   }

   const uint8_t *pixels = map + READBACK_GUARD_BYTES;
#if G17P_R16_UNORM
   static const uint8_t expected[2] = {0xef, 0xbe};
#elif G17P_R16_SNORM
   static const uint8_t expected[2] = {0x00, 0xc0};
#elif G17P_RGBA16_SNORM
   static const uint8_t expected[8] = {
      0x00, 0xc0, 0x01, 0x10, 0x00, 0x60, 0xff, 0xdf,
   };
#elif G17P_R8_SINT
   static const uint8_t expected[1] = {0xa3};
#elif G17P_R8_SNORM
   static const uint8_t expected[1] = {0xc0};
#elif G17P_R8_UINT
   static const uint8_t expected[1] = {0xa7};
#elif G17P_RG8_SNORM
   static const uint8_t expected[2] = {0xc0, 0x11};
#elif G17P_RGBA8_SNORM
   static const uint8_t expected[4] = {0xc0, 0x11, 0x60, 0xdf};
#elif G17P_RGBA8_SINT
   static const uint8_t expected[4] = {0x93, 0x64, 0xcc, 0x21};
#elif G17P_RGBA8_UINT
   static const uint8_t expected[4] = {0x13, 0x57, 0xac, 0xe1};
#elif G17P_RG8_SINT
   static const uint8_t expected[2] = {0xdb, 0x64};
#elif G17P_RG8_UINT
   static const uint8_t expected[2] = {0x57, 0xac};
#elif G17P_R32_SINT
   /* Exact signed value -19088743 is 0xfedcba99 in two's complement. */
   static const uint8_t expected[4] = {0x99, 0xba, 0xdc, 0xfe};
#elif G17P_RG32_SINT
   /* Exact signed pair -19088743 and 19088743. */
   static const uint8_t expected[8] = {
      0x99, 0xba, 0xdc, 0xfe, 0x67, 0x45, 0x23, 0x01,
   };
#elif G17P_RG16_UINT
   static const uint8_t expected[4] = {0x57, 0x13, 0xe1, 0xac};
#elif G17P_RG16_SINT
   static const uint8_t expected[4] = {0xd7, 0xf6, 0x67, 0x45};
#elif G17P_R32_UINT
   static const uint8_t expected[4] = {0xef, 0xcd, 0xab, 0x89};
#elif G17P_RG32_UINT
   static const uint8_t expected[8] = {
      0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01,
   };
#elif G17P_R32_SFLOAT
   /* -0.15625 is exactly 0xbe200000 as binary32. */
   static const uint8_t expected[4] = {0x00, 0x00, 0x20, 0xbe};
#elif G17P_RG32_SFLOAT
   /* Both lanes are exact binary32: -0.15625 and 13.625. */
   static const uint8_t expected[8] = {
      0x00, 0x00, 0x20, 0xbe, 0x00, 0x00, 0x5a, 0x41,
   };
#elif G17P_RG16_SNORM
   /* Fragment (-0.5, 0.28) maps to signed-normalized -16384 (0xc000) and
    * 9175 (0x23d7). The negative first lane distinguishes SNORM from UNORM. */
   static const uint8_t expected[4] = {0x00, 0xc0, 0xd7, 0x23};
#elif G17P_RG16_UNORM
   static const uint8_t expected[4] = {0xff, 0xff, 0xae, 0x47};
#elif G17P_RGBA16_UNORM
   static const uint8_t expected[8] = {
      0xff, 0xff, 0xae, 0x47, 0xcd, 0x0c, 0xff, 0xff,
   };
#else
   static const uint8_t expected[4] = {255, 71, 13, 255};
#endif
   for (size_t pixel = 0; pixel < IMAGE_WIDTH * IMAGE_HEIGHT; ++pixel) {
      const uint8_t *actual = &pixels[pixel * sizeof(expected)];
      if (memcmp(actual, expected, sizeof(expected)) != 0) {
         fprintf(stderr, PROBE_NAME ": pixel %zu differs at byte", pixel);
         for (size_t byte = 0; byte < sizeof(expected); ++byte) {
            if (actual[byte] != expected[byte]) {
               fprintf(stderr, " %zu: got 0x%02x, expected 0x%02x\n", byte,
                       actual[byte], expected[byte]);
               break;
            }
         }
         return false;
      }
   }

   return true;
}

int
main(void)
{
   const bool record_only =
      (G17P_R16_UNORM && getenv("G17P_R16_UNORM_RECORD_ONLY") != NULL) ||
      (G17P_R16_SNORM && getenv("G17P_R16_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_RGBA16_SNORM &&
       getenv("G17P_RGBA16_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_R8_SNORM && getenv("G17P_R8_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_RG8_SNORM && getenv("G17P_RG8_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_F16_IO && getenv("G17P_F16_IO_RECORD_ONLY") != NULL) ||
      (G17P_RG16_UNORM && getenv("G17P_RG16_UNORM_RECORD_ONLY") != NULL) ||
      (G17P_RGBA16_UNORM &&
       getenv("G17P_RGBA16_UNORM_RECORD_ONLY") != NULL) ||
      (G17P_RG16_SNORM && getenv("G17P_RG16_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_R32_UINT && getenv("G17P_R32_UINT_RECORD_ONLY") != NULL) ||
      (G17P_R32_SFLOAT && getenv("G17P_R32_SFLOAT_RECORD_ONLY") != NULL) ||
      (G17P_R32_SINT && getenv("G17P_R32_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RG16_UINT && getenv("G17P_RG16_UINT_RECORD_ONLY") != NULL) ||
      (G17P_RG16_SINT && getenv("G17P_RG16_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RG32_SINT && getenv("G17P_RG32_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RG32_SFLOAT &&
       getenv("G17P_RG32_SFLOAT_RECORD_ONLY") != NULL) ||
      (G17P_RG8_UINT && getenv("G17P_RG8_UINT_RECORD_ONLY") != NULL) ||
      (G17P_RG8_SINT && getenv("G17P_RG8_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RGBA8_UINT && getenv("G17P_RGBA8_UINT_RECORD_ONLY") != NULL) ||
      (G17P_RGBA8_SINT && getenv("G17P_RGBA8_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RGBA8_SNORM && getenv("G17P_RGBA8_SNORM_RECORD_ONLY") != NULL) ||
      (G17P_R8_UINT && getenv("G17P_R8_UINT_RECORD_ONLY") != NULL) ||
      (G17P_R8_SINT && getenv("G17P_R8_SINT_RECORD_ONLY") != NULL) ||
      (G17P_RG32_UINT && getenv("G17P_RG32_UINT_RECORD_ONLY") != NULL);

   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());

   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("libvulkan.so.1 has no vkGetInstanceProcAddr");
   LOAD_GLOBAL(vkCreateInstance);

   const VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = PROBE_NAME,
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
#if G17P_F16_IO
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures2);
#endif
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFormatProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   if (!physical_count)
      die("Vulkan loader found no physical device");

   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      die("out of host memory");
   check(vkEnumeratePhysicalDevices(instance, &physical_count,
                                    physical_devices),
         "vkEnumeratePhysicalDevices(list)");

   VkPhysicalDevice physical = VK_NULL_HANDLE;
   uint32_t queue_family = UINT32_MAX;
   VkPhysicalDeviceProperties properties = {0};
   for (uint32_t p = 0; p < physical_count && !physical; ++p) {
      VkPhysicalDeviceProperties candidate;
      vkGetPhysicalDeviceProperties(physical_devices[p], &candidate);
      if (candidate.deviceID != T8140_DEVICE_ID)
         continue;

      uint32_t queue_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[p],
                                                &queue_count, NULL);
      VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
      if (!queues)
         die("out of host memory");
      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[p],
                                                &queue_count, queues);
      for (uint32_t q = 0; q < queue_count; ++q) {
         const VkQueueFlags required =
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT;
         if ((queues[q].queueFlags & required) == required) {
            physical = physical_devices[p];
            queue_family = q;
            properties = candidate;
            break;
         }
      }
      free(queues);
   }
   free(physical_devices);
   if (!physical)
      die("no T8140 device has a graphics-and-transfer queue");

#if G17P_F16_IO
   VkPhysicalDeviceVulkan11Features available11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
   };
   VkPhysicalDeviceFeatures2 available = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &available11,
   };
   vkGetPhysicalDeviceFeatures2(physical, &available);
   if (!available11.storageInputOutput16)
      die("T8140/G17P does not advertise storageInputOutput16");
#endif

   if (!(properties.limits.framebufferColorSampleCounts &
         (VkSampleCountFlags)G17P_MSAA_SAMPLES))
      die("requested color sample count is not advertised");

   VkFormatProperties format_properties;
   vkGetPhysicalDeviceFormatProperties(physical, IMAGE_FORMAT,
                                       &format_properties);
   const VkFormatFeatureFlags required_format =
      VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
   if ((format_properties.optimalTilingFeatures & required_format) !=
       required_format)
      die("target optimal images lack color-attachment or transfer-src support");

   fprintf(stderr, PROBE_NAME ": device %s, queue %u\n",
           properties.deviceName, queue_family);

   const float queue_priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority,
   };
#if G17P_HOST_IMAGE_COPY
   const VkPhysicalDeviceHostImageCopyFeaturesEXT host_copy_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT,
      .hostImageCopy = VK_TRUE,
   };
   const char *device_extensions[] = {
      VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME,
   };
#endif
#if G17P_F16_IO
   const VkPhysicalDeviceVulkan11Features enabled11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .storageInputOutput16 = VK_TRUE,
   };
#endif
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
#if G17P_F16_IO
      .pNext = &enabled11,
#elif G17P_HOST_IMAGE_COPY
      .pNext = &host_copy_features,
#endif
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
#if G17P_HOST_IMAGE_COPY
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = device_extensions,
#endif
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCreateBuffer);
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkCreateImage);
   LOAD_DEVICE(device, vkGetImageMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkBindImageMemory);
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkCreateImageView);
   LOAD_DEVICE(device, vkCreateRenderPass);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkCreateGraphicsPipelines);
   LOAD_DEVICE(device, vkCreateFramebuffer);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdBeginRenderPass);
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdBindVertexBuffers);
   LOAD_DEVICE(device, vkCmdDraw);
   LOAD_DEVICE(device, vkCmdEndRenderPass);
#if G17P_HOST_IMAGE_COPY
   LOAD_DEVICE(device, vkCopyImageToMemoryEXT);
#else
   LOAD_DEVICE(device, vkCmdCopyImageToBuffer);
#endif
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);

   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);

   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   static const float vertices[][2] = {
      {-1.0f, -1.0f},
      {3.0f, -1.0f},
      {-1.0f, 3.0f},
   };
   const VkBufferCreateInfo vertex_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(vertices),
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer vertex_buffer;
   check(vkCreateBuffer(device, &vertex_info, NULL, &vertex_buffer),
         "vkCreateBuffer(vertex)");
   VkMemoryRequirements vertex_requirements;
   vkGetBufferMemoryRequirements(device, vertex_buffer, &vertex_requirements);
   const uint32_t vertex_memory_type = find_memory_type(
      &memory_properties, vertex_requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   const VkMemoryAllocateInfo vertex_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = vertex_requirements.size,
      .memoryTypeIndex = vertex_memory_type,
   };
   VkDeviceMemory vertex_memory;
   check(vkAllocateMemory(device, &vertex_memory_info, NULL, &vertex_memory),
         "vkAllocateMemory(vertex)");
   check(vkBindBufferMemory(device, vertex_buffer, vertex_memory, 0),
         "vkBindBufferMemory(vertex)");
   void *vertex_map = NULL;
   check(vkMapMemory(device, vertex_memory, 0, sizeof(vertices), 0,
                     &vertex_map),
         "vkMapMemory(vertex)");
   memcpy(vertex_map, vertices, sizeof(vertices));

   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = IMAGE_FORMAT,
      .extent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | READBACK_IMAGE_USAGE,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   check(vkCreateImage(device, &image_info, NULL, &image), "vkCreateImage");
   VkMemoryRequirements image_requirements;
   vkGetImageMemoryRequirements(device, image, &image_requirements);
   const uint32_t image_memory_type =
      find_memory_type(&memory_properties, image_requirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   const VkMemoryAllocateInfo image_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = image_requirements.size,
      .memoryTypeIndex = image_memory_type,
   };
   VkDeviceMemory image_memory;
   check(vkAllocateMemory(device, &image_memory_info, NULL, &image_memory),
         "vkAllocateMemory(image)");
   check(vkBindImageMemory(device, image, image_memory, 0),
         "vkBindImageMemory");

#if G17P_MSAA_SAMPLES == 2
   VkImageCreateInfo msaa_image_info = image_info;
   msaa_image_info.samples = VK_SAMPLE_COUNT_2_BIT;
   msaa_image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   VkImage msaa_image;
   check(vkCreateImage(device, &msaa_image_info, NULL, &msaa_image),
         "vkCreateImage(msaa2)");
   VkMemoryRequirements msaa_image_requirements;
   vkGetImageMemoryRequirements(device, msaa_image, &msaa_image_requirements);
   const uint32_t msaa_image_memory_type =
      find_memory_type(&memory_properties,
                       msaa_image_requirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   const VkMemoryAllocateInfo msaa_image_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = msaa_image_requirements.size,
      .memoryTypeIndex = msaa_image_memory_type,
   };
   VkDeviceMemory msaa_image_memory;
   check(vkAllocateMemory(device, &msaa_image_memory_info, NULL,
                          &msaa_image_memory),
         "vkAllocateMemory(msaa2)");
   check(vkBindImageMemory(device, msaa_image, msaa_image_memory, 0),
         "vkBindImageMemory(msaa2)");
#endif

#if G17P_HOST_IMAGE_COPY
   uint8_t readback_storage[READBACK_BYTES];
   uint8_t *readback_map = readback_storage;
#else
   const VkBufferCreateInfo readback_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = READBACK_BYTES,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer readback;
   check(vkCreateBuffer(device, &readback_info, NULL, &readback),
         "vkCreateBuffer(readback)");
   VkMemoryRequirements readback_requirements;
   vkGetBufferMemoryRequirements(device, readback, &readback_requirements);
   const uint32_t readback_memory_type = find_memory_type(
      &memory_properties, readback_requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   const VkMemoryAllocateInfo readback_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = readback_requirements.size,
      .memoryTypeIndex = readback_memory_type,
   };
   VkDeviceMemory readback_memory;
   check(vkAllocateMemory(device, &readback_memory_info, NULL,
                          &readback_memory),
         "vkAllocateMemory(readback)");
   check(vkBindBufferMemory(device, readback, readback_memory, 0),
         "vkBindBufferMemory(readback)");
   uint8_t *readback_map = NULL;
   check(vkMapMemory(device, readback_memory, 0, READBACK_BYTES, 0,
                     (void **)&readback_map),
         "vkMapMemory(readback)");
#endif
   memset(readback_map, 0xa5, READBACK_BYTES);

   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = IMAGE_FORMAT,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1,
         .layerCount = 1,
      },
   };
   VkImageView view;
   check(vkCreateImageView(device, &view_info, NULL, &view),
         "vkCreateImageView");

#if G17P_MSAA_SAMPLES == 2
   VkImageViewCreateInfo msaa_view_info = view_info;
   msaa_view_info.image = msaa_image;
   VkImageView msaa_view;
   check(vkCreateImageView(device, &msaa_view_info, NULL, &msaa_view),
         "vkCreateImageView(msaa2)");
#endif

   const VkAttachmentDescription attachments[] = {{
      .format = IMAGE_FORMAT,
      .samples = (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
#if G17P_MSAA_SAMPLES == 2
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
#else
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
#endif
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
#if G17P_MSAA_SAMPLES == 2
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
#endif
   },
#if G17P_MSAA_SAMPLES == 2
   {
      .format = IMAGE_FORMAT,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
   },
#endif
   };
   const VkAttachmentReference color_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
#if G17P_MSAA_SAMPLES == 2
   const VkAttachmentReference resolve_ref = {
      .attachment = 1,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
#endif
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
#if G17P_MSAA_SAMPLES == 2
      .pResolveAttachments = &resolve_ref,
#endif
   };
   const VkSubpassDependency dependencies[] = {
      {
         .srcSubpass = VK_SUBPASS_EXTERNAL,
         .dstSubpass = 0,
         .srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      },
      {
         .srcSubpass = 0,
         .dstSubpass = VK_SUBPASS_EXTERNAL,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      },
   };
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = sizeof(attachments) / sizeof(attachments[0]),
      .pAttachments = attachments,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 2,
      .pDependencies = dependencies,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &render_pass_info, NULL, &render_pass),
         "vkCreateRenderPass");

   const VkShaderModuleCreateInfo vert_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(G17P_VERT_SPV),
      .pCode = G17P_VERT_SPV,
   };
   const VkShaderModuleCreateInfo frag_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(G17P_FRAG_SPV),
      .pCode = G17P_FRAG_SPV,
   };
   VkShaderModule vert_module, frag_module;
   check(vkCreateShaderModule(device, &vert_module_info, NULL, &vert_module),
         "vkCreateShaderModule(vertex)");
   check(vkCreateShaderModule(device, &frag_module_info, NULL, &frag_module),
         "vkCreateShaderModule(fragment)");

   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pipeline_layout;
   check(vkCreatePipelineLayout(device, &layout_info, NULL, &pipeline_layout),
         "vkCreatePipelineLayout");

   const VkPipelineShaderStageCreateInfo stages[] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vert_module,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = frag_module,
         .pName = "main",
      },
   };
   const VkVertexInputBindingDescription vertex_binding = {
      .binding = 0,
      .stride = sizeof(vertices[0]),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription vertex_attribute = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &vertex_attribute,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {
      .width = IMAGE_WIDTH,
      .height = IMAGE_HEIGHT,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };
   const VkRect2D scissor = {.extent = {IMAGE_WIDTH, IMAGE_HEIGHT}};
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
   };
   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
   };
   const VkPipelineColorBlendAttachmentState blend_attachment = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blend_attachment,
   };
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pColorBlendState = &blend,
      .layout = pipeline_layout,
      .renderPass = render_pass,
   };
   VkPipeline pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipeline),
         "vkCreateGraphicsPipelines");

#if G17P_MSAA_SAMPLES == 2
   const VkImageView framebuffer_views[] = {msaa_view, view};
#else
   const VkImageView framebuffer_views[] = {view};
#endif
   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount =
         sizeof(framebuffer_views) / sizeof(framebuffer_views[0]),
      .pAttachments = framebuffer_views,
      .width = IMAGE_WIDTH,
      .height = IMAGE_HEIGHT,
      .layers = 1,
   };
   VkFramebuffer framebuffer;
   check(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer),
         "vkCreateFramebuffer");

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool command_pool;
   check(vkCreateCommandPool(device, &pool_info, NULL, &command_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_allocate = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command_buffer;
   check(vkAllocateCommandBuffers(device, &command_allocate, &command_buffer),
         "vkAllocateCommandBuffers");
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command_buffer, &begin_info),
         "vkBeginCommandBuffer");
#if G17P_R32_SINT || G17P_RG16_SINT || G17P_RG32_SINT ||              \
   G17P_RG8_SINT || G17P_RGBA8_SINT || G17P_R8_SINT
   const VkClearValue clear = {.color = {.int32 = {0, 0, 0, 0}}};
#elif G17P_R32_UINT || G17P_RG16_UINT || G17P_RG8_UINT ||             \
   G17P_RG32_UINT || G17P_RGBA8_UINT || G17P_R8_UINT
   const VkClearValue clear = {.color = {.uint32 = {0, 0, 0, 0}}};
#else
   const VkClearValue clear = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};
#endif
   const VkRenderPassBeginInfo render_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {.extent = {IMAGE_WIDTH, IMAGE_HEIGHT}},
      .clearValueCount = 1,
      .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(command_buffer, &render_begin,
                        VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                     pipeline);
   const VkDeviceSize vertex_offset = 0;
   vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer,
                          &vertex_offset);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
   vkCmdEndRenderPass(command_buffer);

#if !G17P_HOST_IMAGE_COPY
   const VkBufferImageCopy copy = {
      .bufferOffset = READBACK_GUARD_BYTES,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .layerCount = 1,
      },
      .imageExtent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
   };
   if (!record_only) {
      vkCmdCopyImageToBuffer(command_buffer, image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                             &copy);
   }
#endif
   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

   if (record_only) {
      printf(PROBE_NAME ": PASS — feature advertised, shaders compiled, "
                        "vkEndCommandBuffer succeeded\n");
      return EXIT_SUCCESS;
   }

   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence fence;
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
   };
   check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
#if G17P_F16_IO || G17P_RG16_UNORM || G17P_RGBA16_UNORM ||            \
   G17P_RG16_SNORM || G17P_R32_UINT || G17P_R32_SFLOAT ||             \
   G17P_R32_SINT || G17P_RG16_UINT || G17P_RG16_SINT ||               \
   G17P_RG32_SINT || G17P_RG32_SFLOAT || G17P_RG8_UINT ||             \
   G17P_RG8_SINT || G17P_RG32_UINT || G17P_RGBA8_UINT ||              \
   G17P_RGBA8_SINT || G17P_RGBA8_SNORM || G17P_R8_UINT || G17P_R8_SINT
   const VkResult waited =
      vkWaitForFences(device, 1, &fence, true, UINT64_C(3000000000));
   if (waited == VK_TIMEOUT)
      die("vkWaitForFences timed out after three seconds");
   check(waited, "vkWaitForFences");
#elif G17P_HOST_IMAGE_COPY || G17P_MSAA_SAMPLES == 2
   const VkResult waited =
      vkWaitForFences(device, 1, &fence, true, UINT64_C(5000000000));
   if (waited == VK_TIMEOUT)
      die("vkWaitForFences timed out after five seconds");
   check(waited, "vkWaitForFences");
#else
   check(vkWaitForFences(device, 1, &fence, true, UINT64_MAX),
         "vkWaitForFences");
#endif

#if G17P_HOST_IMAGE_COPY
   const VkImageToMemoryCopyEXT host_copy_region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY_EXT,
      .pHostPointer = readback_map,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .layerCount = 1,
      },
      .imageExtent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
   };
   const VkCopyImageToMemoryInfoEXT host_copy = {
      .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO_EXT,
      .srcImage = image,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .regionCount = 1,
      .pRegions = &host_copy_region,
   };
   check(vkCopyImageToMemoryEXT(device, &host_copy),
         "vkCopyImageToMemoryEXT");
#endif

   if (!verify_readback(readback_map))
      return EXIT_FAILURE;

#if G17P_MSAA_SAMPLES == 2 || G17P_F16_IO || G17P_RG16_UNORM ||         \
   G17P_RGBA16_UNORM || G17P_RG16_SNORM || G17P_R32_UINT ||             \
   G17P_R32_SFLOAT || G17P_R32_SINT || G17P_RG16_UINT ||                \
   G17P_RG16_SINT || G17P_RG32_SINT || G17P_RG32_SFLOAT ||            \
   G17P_RG8_UINT || G17P_RG8_SINT || G17P_RG32_UINT ||                \
   G17P_RGBA8_UINT || G17P_RGBA8_SINT || G17P_RGBA8_SNORM ||           \
   G17P_RG8_SNORM || G17P_R8_UINT || G17P_R8_SNORM || G17P_R8_SINT || \
   G17P_RGBA16_SNORM || G17P_R16_SNORM || G17P_R16_UNORM
   sleep(3);
   if (!verify_readback(readback_map))
      return EXIT_FAILURE;
#endif
   const uint64_t checksum =
      fnv1a64(readback_map + READBACK_GUARD_BYTES, IMAGE_BYTES);
#if G17P_R16_UNORM
   printf(PROBE_NAME ": PASS — %ux%u R16 UNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R16_SNORM
   printf(PROBE_NAME ": PASS — %ux%u R16 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA16_SNORM
   printf(PROBE_NAME ": PASS — %ux%u RGBA16 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R8_SINT
   printf(PROBE_NAME ": PASS — %ux%u R8 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R8_SNORM
   printf(PROBE_NAME ": PASS — %ux%u R8 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R8_UINT
   printf(PROBE_NAME ": PASS — %ux%u R8 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG8_SNORM
   printf(PROBE_NAME ": PASS — %ux%u RG8 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA8_SNORM
   printf(PROBE_NAME ": PASS — %ux%u RGBA8 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA8_SINT
   printf(PROBE_NAME ": PASS — %ux%u RGBA8 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA8_UINT
   printf(PROBE_NAME ": PASS — %ux%u RGBA8 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG8_SINT
   printf(PROBE_NAME ": PASS — %ux%u RG8 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG8_UINT
   printf(PROBE_NAME ": PASS — %ux%u RG8 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R32_SINT
   printf(PROBE_NAME ": PASS — %ux%u R32 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG32_SINT
   printf(PROBE_NAME ": PASS — %ux%u RG32 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG16_UINT
   printf(PROBE_NAME ": PASS — %ux%u RG16 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG16_SINT
   printf(PROBE_NAME ": PASS — %ux%u RG16 SINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R32_UINT
   printf(PROBE_NAME ": PASS — %ux%u R32 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG32_UINT
   printf(PROBE_NAME ": PASS — %ux%u RG32 UINT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_R32_SFLOAT
   printf(PROBE_NAME ": PASS — %ux%u R32 SFLOAT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG32_SFLOAT
   printf(PROBE_NAME ": PASS — %ux%u RG32 SFLOAT exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG16_SNORM
   printf(PROBE_NAME ": PASS — %ux%u RG16 SNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RG16_UNORM
   printf(PROBE_NAME ": PASS — %ux%u RG16 UNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#elif G17P_RGBA16_UNORM
   printf(PROBE_NAME ": PASS — %ux%u RGBA16 UNORM exact immediately/3s, "
                     "128 guard bytes unchanged, checksum %016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#else
   printf(PROBE_NAME ": PASS — %ux%u RGBA8, checksum "
          "%016" PRIx64 "\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, checksum);
#endif
   return EXIT_SUCCESS;
}
