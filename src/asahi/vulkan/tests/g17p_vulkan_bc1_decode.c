/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
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
#include <time.h>

#ifdef G17P_COMPRESSED_3D
#include "g17p_vulkan_bc1_decode_3d_spv.h"
#else
#include "g17p_vulkan_bc1_decode_spv.h"
#endif

#define WIDTH 8
#define HEIGHT 8
#if defined(G17P_BC1_SRGB_DECODE)
#define BLOCK_BYTES 8
#define BLOCK_DATA_BYTES 32
#define TEST_FORMAT VK_FORMAT_BC1_RGBA_SRGB_BLOCK
#define NAME "g17p-vulkan-bc1-srgb-decode"
#elif defined(G17P_BC2_SRGB_DECODE)
#define BLOCK_BYTES 16
#define BLOCK_DATA_BYTES 64
#define TEST_FORMAT VK_FORMAT_BC2_SRGB_BLOCK
#define NAME "g17p-vulkan-bc2-srgb-decode"
#elif defined(G17P_BC3_SRGB_DECODE)
#define BLOCK_BYTES 16
#define BLOCK_DATA_BYTES 64
#define TEST_FORMAT VK_FORMAT_BC3_SRGB_BLOCK
#define NAME "g17p-vulkan-bc3-srgb-decode"
#elif defined(G17P_BC4_SNORM_DECODE)
#define BLOCK_BYTES 8
#define BLOCK_DATA_BYTES 32
#define TEST_FORMAT VK_FORMAT_BC4_SNORM_BLOCK
#define NAME "g17p-vulkan-bc4-snorm-decode"
#elif defined(G17P_BC5_SNORM_DECODE)
#define BLOCK_BYTES 16
#define BLOCK_DATA_BYTES 64
#define TEST_FORMAT VK_FORMAT_BC5_SNORM_BLOCK
#define NAME "g17p-vulkan-bc5-snorm-decode"
#elif defined(G17P_BC7_SRGB_DECODE)
#define BLOCK_BYTES 16
#define BLOCK_DATA_BYTES 64
#define TEST_FORMAT VK_FORMAT_BC7_SRGB_BLOCK
#define NAME "g17p-vulkan-bc7-srgb-decode"
#elif defined(G17P_BC6H_SFLOAT_DECODE)
#define BLOCK_BYTES 16
#define BLOCK_DATA_BYTES 64
#define TEST_FORMAT VK_FORMAT_BC6H_SFLOAT_BLOCK
#define NAME "g17p-vulkan-bc6h-sfloat-decode"
#elif defined(G17P_BC6H_UFLOAT_DECODE)
#define BLOCK_BYTES 16
#define BLOCK_DATA_BYTES 64
#define TEST_FORMAT VK_FORMAT_BC6H_UFLOAT_BLOCK
#define NAME "g17p-vulkan-bc6h-ufloat-decode"
#elif defined(G17P_BC7_DECODE)
#define BLOCK_BYTES 16
#define BLOCK_DATA_BYTES 64
#define TEST_FORMAT VK_FORMAT_BC7_UNORM_BLOCK
#define NAME "g17p-vulkan-bc7-decode"
#elif defined(G17P_BC2_DECODE)
#define BLOCK_BYTES 16
#define BLOCK_DATA_BYTES 64
#define TEST_FORMAT VK_FORMAT_BC2_UNORM_BLOCK
#define NAME "g17p-vulkan-bc2-decode"
#elif defined(G17P_BC4_DECODE)
#define BLOCK_BYTES 8
#define BLOCK_DATA_BYTES 32
#define TEST_FORMAT VK_FORMAT_BC4_UNORM_BLOCK
#define NAME "g17p-vulkan-bc4-decode"
#elif defined(G17P_BC5_DECODE)
#define BLOCK_BYTES 16
#define BLOCK_DATA_BYTES 64
#define TEST_FORMAT VK_FORMAT_BC5_UNORM_BLOCK
#define NAME "g17p-vulkan-bc5-decode"
#elif defined(G17P_BC3_DECODE)
#define BLOCK_BYTES 16
#define BLOCK_DATA_BYTES 64
#define TEST_FORMAT VK_FORMAT_BC3_UNORM_BLOCK
#define NAME "g17p-vulkan-bc3-decode"
#elif defined(G17P_ETC2_RGBA8_DECODE)
#define BLOCK_BYTES 16
#define BLOCK_DATA_BYTES 64
#define NAME "g17p-vulkan-etc2-eac-decode"
#elif defined(G17P_ASTC_DECODE) || defined(G17P_ASTC4X4_DECODE)
#define BLOCK_BYTES 16
#define BLOCK_DATA_BYTES 64
#define NAME "g17p-vulkan-astc-decode"
#else
#define BLOCK_BYTES 8
#define BLOCK_DATA_BYTES 32
#define TEST_FORMAT VK_FORMAT_BC1_RGBA_UNORM_BLOCK
#if defined(G17P_COMPRESSED_READBACK)
#define NAME "g17p-vulkan-bc1-3d-roundtrip"
#elif defined(G17P_COMPRESSED_3D)
#define NAME "g17p-vulkan-bc1-3d-decode"
#else
#define NAME "g17p-vulkan-bc1-decode"
#endif
#endif
#ifdef G17P_COMPRESSED_3D
#undef BLOCK_DATA_BYTES
#define BLOCK_DATA_BYTES 112
#define COPY_DEPTH 2
#define IMAGE_DEPTH 4
#define BUFFER_ROW_LENGTH 12
#define BUFFER_IMAGE_HEIGHT 12
#define SOURCE_ROW_BYTES 24
#define SOURCE_SLICE_BYTES 72
#else
#define COPY_DEPTH 1
#define IMAGE_DEPTH 1
#define BUFFER_ROW_LENGTH 0
#define BUFFER_IMAGE_HEIGHT 0
#endif
#define PIXEL_BYTES (WIDTH * HEIGHT * 4)
#if COPY_DEPTH > 1
#undef PIXEL_BYTES
#define PIXEL_BYTES (WIDTH * HEIGHT * COPY_DEPTH * 4)
#endif
#define GUARD_BYTES 64
#define DEVICE_ID 0x8140
#ifdef G17P_COMPRESSED_READBACK
#define BUFFER_COUNT 3
#define READBACK_FORMAT_FEATURE VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
#define READBACK_IMAGE_USAGE VK_IMAGE_USAGE_TRANSFER_SRC_BIT
#else
#define BUFFER_COUNT 2
#define READBACK_FORMAT_FEATURE 0
#define READBACK_IMAGE_USAGE 0
#endif

static const char *test_name = NAME;

#define FN(name) static PFN_##name name
FN(vkGetInstanceProcAddr);
FN(vkGetDeviceProcAddr);
FN(vkCreateInstance);
FN(vkEnumeratePhysicalDevices);
FN(vkGetPhysicalDeviceFeatures);
FN(vkGetPhysicalDeviceProperties);
FN(vkGetPhysicalDeviceMemoryProperties);
FN(vkGetPhysicalDeviceQueueFamilyProperties);
FN(vkGetPhysicalDeviceFormatProperties);
FN(vkGetPhysicalDeviceImageFormatProperties);
FN(vkCreateDevice);
FN(vkGetDeviceQueue);
FN(vkCreateBuffer);
FN(vkGetBufferMemoryRequirements);
FN(vkAllocateMemory);
FN(vkBindBufferMemory);
FN(vkMapMemory);
FN(vkCreateImage);
FN(vkGetImageMemoryRequirements);
FN(vkBindImageMemory);
FN(vkCreateImageView);
FN(vkCreateSampler);
FN(vkCreateDescriptorSetLayout);
FN(vkCreateDescriptorPool);
FN(vkAllocateDescriptorSets);
FN(vkUpdateDescriptorSets);
FN(vkCreateShaderModule);
FN(vkCreatePipelineLayout);
FN(vkCreateComputePipelines);
FN(vkCreateCommandPool);
FN(vkAllocateCommandBuffers);
FN(vkBeginCommandBuffer);
FN(vkCmdPipelineBarrier);
FN(vkCmdCopyBufferToImage);
FN(vkCmdCopyImageToBuffer);
FN(vkCmdBindPipeline);
FN(vkCmdBindDescriptorSets);
FN(vkCmdDispatch);
FN(vkEndCommandBuffer);
FN(vkCreateFence);
FN(vkQueueSubmit);
FN(vkWaitForFences);

static void
die(const char *message)
{
   fprintf(stderr, "%s: %s\n", test_name, message);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *call)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "%s: %s returned %d\n", test_name, call, result);
      exit(EXIT_FAILURE);
   }
}

#define LOAD_GLOBAL(n)                                                        \
   do {                                                                       \
      n = (PFN_##n)vkGetInstanceProcAddr(VK_NULL_HANDLE, #n);                 \
      if (!n)                                                                 \
         die("missing " #n);                                                 \
   } while (0)
#define LOAD_INSTANCE(i, n)                                                   \
   do {                                                                       \
      n = (PFN_##n)vkGetInstanceProcAddr(i, #n);                              \
      if (!n)                                                                 \
         die("missing " #n);                                                 \
   } while (0)
#define LOAD_DEVICE(d, n)                                                     \
   do {                                                                       \
      n = (PFN_##n)vkGetDeviceProcAddr(d, #n);                                \
      if (!n)                                                                 \
         die("missing " #n);                                                 \
   } while (0)

static uint32_t
memory_type(const VkPhysicalDeviceMemoryProperties *props, uint32_t allowed,
            VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < props->memoryTypeCount; ++i) {
      if ((allowed & (1u << i)) &&
          (props->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }
   die("required memory type is unavailable");
   return 0;
}

static uint64_t
fnv1a64(const uint8_t *data, size_t size)
{
   uint64_t hash = UINT64_C(14695981039346656037);
   for (size_t i = 0; i < size; ++i) {
      hash ^= data[i];
      hash *= UINT64_C(1099511628211);
   }
   return hash;
}

#if !defined(G17P_ETC2_RGBA8_DECODE) && !defined(G17P_ASTC_DECODE) &&         \
   !defined(G17P_ASTC4X4_DECODE)
#if !defined(G17P_BC4_DECODE) && !defined(G17P_BC5_DECODE) &&               \
   !defined(G17P_BC4_SNORM_DECODE) && !defined(G17P_BC5_SNORM_DECODE) &&    \
   !defined(G17P_BC6H_SFLOAT_DECODE) &&                                \
   !defined(G17P_BC6H_UFLOAT_DECODE) && !defined(G17P_BC7_DECODE) &&       \
   !defined(G17P_BC7_SRGB_DECODE)
static uint8_t
expand5(unsigned value)
{
   return (value << 3) | (value >> 2);
}

static uint8_t
expand6(unsigned value)
{
   return (value << 2) | (value >> 4);
}

static void
decode_bc1(const uint8_t *block, uint8_t pixels[16][4])
{
   const uint16_t c0 = block[0] | ((uint16_t)block[1] << 8);
   const uint16_t c1 = block[2] | ((uint16_t)block[3] << 8);
   uint8_t palette[4][4] = {
      {expand5(c0 >> 11), expand6((c0 >> 5) & 63), expand5(c0 & 31), 255},
      {expand5(c1 >> 11), expand6((c1 >> 5) & 63), expand5(c1 & 31), 255},
   };

   if (c0 > c1) {
      for (unsigned c = 0; c < 3; ++c) {
         palette[2][c] = (2 * palette[0][c] + palette[1][c]) / 3;
         palette[3][c] = (palette[0][c] + 2 * palette[1][c]) / 3;
      }
      palette[2][3] = palette[3][3] = 255;
   } else {
      for (unsigned c = 0; c < 3; ++c)
         palette[2][c] = (palette[0][c] + palette[1][c]) / 2;
      palette[2][3] = 255;
      memset(palette[3], 0, sizeof(palette[3]));
   }

   const uint32_t indices = block[4] | ((uint32_t)block[5] << 8) |
                            ((uint32_t)block[6] << 16) |
                            ((uint32_t)block[7] << 24);
   for (unsigned i = 0; i < 16; ++i)
      memcpy(pixels[i], palette[(indices >> (2 * i)) & 3], 4);
}

#if defined(G17P_BC1_SRGB_DECODE) || defined(G17P_BC2_SRGB_DECODE) ||     \
   defined(G17P_BC3_SRGB_DECODE)
static uint8_t
srgb_endpoint_to_linear_unorm8(uint8_t endpoint)
{
   /* Exact round-to-nearest results of the Vulkan sRGB EOTF for every
    * RGB565-expanded endpoint used by this fixture. */
   switch (endpoint) {
   case 0:   return 0;
   case 65:  return 13;
   case 66:  return 14;
   case 130: return 57;
   case 132: return 59;
   case 195: return 139;
   case 198: return 144;
   case 255: return 255;
   default: die("unexpected sRGB endpoint in reference decoder");
   }

   return 0;
}
#endif

static void
put_bc1_payload(uint8_t *out, uint16_t c0, uint16_t c1, uint32_t indices)
{
   out[0] = c0;
   out[1] = c0 >> 8;
   out[2] = c1;
   out[3] = c1 >> 8;
   out[4] = indices;
   out[5] = indices >> 8;
   out[6] = indices >> 16;
   out[7] = indices >> 24;
}
#endif

static size_t
block_offset(unsigned block)
{
#ifdef G17P_COMPRESSED_3D
   const unsigned z = block / 4;
   const unsigned in_slice = block % 4;
   return z * SOURCE_SLICE_BYTES + (in_slice / 2) * SOURCE_ROW_BYTES +
          (in_slice % 2) * BLOCK_BYTES;
#else
   return block * BLOCK_BYTES;
#endif
}

static void
put_block(uint8_t *data, unsigned block, uint16_t c0, uint16_t c1,
          uint32_t indices)
{
   uint8_t *out = data + block_offset(block);
#if defined(G17P_BC4_SNORM_DECODE) || defined(G17P_BC5_SNORM_DECODE)
   static const int8_t rg[4][2] = {
      {-127, 127}, {127, -127}, {-128, 0}, {0, 127},
   };
   (void)c0;
   (void)c1;
   (void)indices;
   memset(out, 0, BLOCK_BYTES);
   out[0] = (uint8_t)rg[block][0];
#if defined(G17P_BC5_SNORM_DECODE)
   out[8] = (uint8_t)rg[block][1];
#endif
#elif defined(G17P_BC6H_SFLOAT_DECODE)
   static const uint8_t encoded[4][16] = {
      {0xc3,0x1a,0x9d,0xbd,0xf1,0xe6,0x1d,0x6b,0x40,0xf8,0x40,0xf8,0x40,0xf8,0x40,0xf8},
      {0xa3,0x9d,0x5a,0xb4,0xd1,0xa6,0x9e,0x76,0x10,0x9f,0x10,0x9f,0x10,0x9f,0x10,0x9f},
      {0x23,0x9c,0x73,0xd0,0xb9,0xe7,0x9d,0x75,0xf0,0x02,0xf0,0x02,0xf0,0x02,0xf0,0x02},
      {0x83,0xe4,0x7b,0x48,0xbe,0x87,0xe4,0x7b,0x70,0xf0,0x70,0xf0,0x70,0xf0,0x70,0xf0},
   };
   (void)c0;
   (void)c1;
   (void)indices;
   memcpy(out, encoded[block], BLOCK_BYTES);
#elif defined(G17P_BC6H_UFLOAT_DECODE)
   static const uint8_t encoded[4][16] = {
      {0x43,0x3a,0xce,0x7a,0xeb,0xcd,0x3b,0xe9,0x60,0xfd,0x60,0xfd,0x60,0xfd,0x60,0xfd},
      {0x43,0xbb,0xd2,0x94,0x53,0x6e,0x3d,0xed,0x10,0xbf,0x10,0xbf,0x10,0xbf,0x10,0xbf},
      {0x43,0x3b,0xe1,0xa8,0xa3,0xee,0x3c,0xed,0xf0,0x48,0xf0,0x48,0xf0,0x48,0xf0,0x48},
      {0x83,0x3b,0xee,0xc0,0x7b,0xef,0xbd,0xf1,0xf0,0x03,0xf0,0x03,0xf0,0x03,0xf0,0x03},
   };
   (void)c0;
   (void)c1;
   (void)indices;
   memcpy(out, encoded[block], BLOCK_BYTES);
#elif defined(G17P_BC7_DECODE) || defined(G17P_BC7_SRGB_DECODE)
   static const uint8_t encoded[4][16] = {
      {0x10,0x84,0x20,0x04,0x21,0x86,0x01,0,0,0,0,0,0,0,0,0},
      {0x10,0xce,0x09,0xc1,0x39,0x8a,0x02,0,0,0,0,0,0,0,0,0},
      {0x10,0x5a,0x4b,0x69,0x0c,0x82,0x00,0,0,0,0,0,0,0,0,0},
      {0x10,0xde,0x63,0xac,0xd4,0xff,0x03,0,0,0,0,0,0,0,0,0},
   };
   (void)c0;
   (void)c1;
   (void)indices;
   memcpy(out, encoded[block], BLOCK_BYTES);
#elif defined(G17P_BC4_DECODE) || defined(G17P_BC5_DECODE)
   static const uint8_t rg[4][2] = {
      {0x20, 0xe0}, {0x60, 0xa0}, {0xa0, 0x60}, {0xe0, 0x20},
   };
   (void)c0;
   (void)c1;
   (void)indices;
   memset(out, 0, BLOCK_BYTES);
   out[0] = rg[block][0];
#if defined(G17P_BC5_DECODE)
   out[8] = rg[block][1];
#endif
#else
#if defined(G17P_BC2_DECODE) || defined(G17P_BC2_SRGB_DECODE)
   static const uint8_t alpha4[4] = {1, 5, 10, 15};
   memset(out, alpha4[block] | (alpha4[block] << 4), 8);
   out += 8;
#elif defined(G17P_BC3_DECODE) || defined(G17P_BC3_SRGB_DECODE)
   static const uint8_t alpha[4] = {0x20, 0x60, 0xa0, 0xff};
   out[0] = alpha[block];
   out[1] = 0;
   memset(out + 2, 0, 6);
   out += 8;
#endif
   put_bc1_payload(out, c0, c1, indices);
#endif
}

static void
build_reference(const uint8_t *blocks, uint8_t *reference)
{
#if defined(G17P_BC4_SNORM_DECODE) || defined(G17P_BC5_SNORM_DECODE)
   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const unsigned block = (y / 4) * 2 + x / 4;
         const uint8_t *encoded = blocks + block * BLOCK_BYTES;
         uint8_t *pixel = reference + (y * WIDTH + x) * 4;
         pixel[0] = encoded[0] == 127 ? 255 : 0;
#if defined(G17P_BC5_SNORM_DECODE)
         pixel[1] = encoded[8] == 127 ? 255 : 0;
#else
         pixel[1] = 0;
#endif
         pixel[2] = 0;
         pixel[3] = 255;
      }
   }
#elif defined(G17P_BC6H_SFLOAT_DECODE)
   (void)blocks;
   static const uint8_t decoded[4][4][4] = {
      {{0x3f,0x00,0x5e,0xff}, {0x47,0x00,0x56,0xff},
       {0x4f,0x00,0x4d,0xff}, {0x5e,0xbf,0x3f,0xff}},
      {{0xb0,0x10,0x4e,0xff}, {0xa7,0x14,0x53,0xff},
       {0x4e,0xee,0xb0,0xff}, {0x6c,0x52,0x7a,0xff}},
      {{0x69,0x82,0x89,0xff}, {0xfd,0xbf,0xa0,0xff},
       {0x75,0x8a,0x8d,0xff}, {0x69,0x82,0x89,0xff}},
      {{0x00,0xfd,0x00,0xff}, {0x00,0x00,0x00,0xff},
       {0x00,0xfd,0x00,0xff}, {0xfd,0x00,0xfd,0xff}},
   };
   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const unsigned block = (y / 4) * 2 + x / 4;
         memcpy(reference + (y * WIDTH + x) * 4,
                decoded[block][x % 4], 4);
      }
   }
#elif defined(G17P_BC6H_UFLOAT_DECODE)
   (void)blocks;
   static const uint8_t decoded[4][4][4] = {
      {{0x8f,0x2f,0x5f,0xff}, {0x77,0x53,0x6f,0xff},
       {0x64,0x9a,0x84,0xff}, {0x5f,0xbd,0x8f,0xff}},
      {{0xae,0x38,0x78,0xff}, {0xaa,0x3c,0x7a,0xff},
       {0x78,0xf0,0xae,0xff}, {0x81,0xa8,0x9e,0xff}},
      {{0xae,0x68,0x97,0xff}, {0x97,0xe0,0xae,0xff},
       {0xa2,0x9d,0xa3,0xff}, {0xa8,0x7b,0x9d,0xff}},
      {{0xb6,0xb6,0xc5,0xff}, {0xff,0xff,0xd1,0xff},
       {0xc5,0xc5,0xc7,0xff}, {0xb6,0xb6,0xc5,0xff}},
   };
   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const unsigned block = (y / 4) * 2 + x / 4;
         memcpy(reference + (y * WIDTH + x) * 4,
                decoded[block][x % 4], 4);
      }
   }
#elif defined(G17P_BC7_DECODE) || defined(G17P_BC7_SRGB_DECODE)
   (void)blocks;
#if defined(G17P_BC7_SRGB_DECODE)
   static const uint8_t decoded[4][4] = {
      {0x04, 0x0e, 0x3b, 0x61},
      {0x2c, 0x01, 0xcc, 0xa2},
      {0xab, 0x4c, 0x08, 0x20},
      {0xed, 0x90, 0x16, 0xff},
   };
#else
   static const uint8_t decoded[4][4] = {
      {0x21, 0x42, 0x84, 0x61},
      {0x73, 0x10, 0xe7, 0xa2},
      {0xd6, 0x94, 0x31, 0x20},
      {0xf7, 0xc6, 0x52, 0xff},
   };
#endif
   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const unsigned block = (y / 4) * 2 + x / 4;
         memcpy(reference + (y * WIDTH + x) * 4, decoded[block], 4);
      }
   }
#elif defined(G17P_BC1_SRGB_DECODE) || defined(G17P_BC2_SRGB_DECODE) ||    \
   defined(G17P_BC3_SRGB_DECODE)
   uint8_t decoded[4][16][4];
   for (unsigned block = 0; block < 4; ++block) {
      const uint8_t *encoded = blocks + block * BLOCK_BYTES;
#if defined(G17P_BC2_SRGB_DECODE)
      decode_bc1(encoded + 8, decoded[block]);
      for (unsigned texel = 0; texel < 16; ++texel)
         decoded[block][texel][3] = (encoded[0] & 0xf) * 17;
#elif defined(G17P_BC3_SRGB_DECODE)
      decode_bc1(encoded + 8, decoded[block]);
      for (unsigned texel = 0; texel < 16; ++texel)
         decoded[block][texel][3] = encoded[0];
#else
      decode_bc1(encoded, decoded[block]);
#endif
   }

   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const unsigned block = (y / 4) * 2 + x / 4;
         const unsigned texel = (y % 4) * 4 + x % 4;
         uint8_t *pixel = reference + (y * WIDTH + x) * 4;
         for (unsigned c = 0; c < 3; ++c) {
            pixel[c] =
               srgb_endpoint_to_linear_unorm8(decoded[block][texel][c]);
         }
         pixel[3] = decoded[block][texel][3];
      }
   }
#elif defined(G17P_BC4_DECODE) || defined(G17P_BC5_DECODE)
   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const unsigned block = (y / 4) * 2 + x / 4;
         const uint8_t *encoded = blocks + block * BLOCK_BYTES;
         uint8_t *pixel = reference + (y * WIDTH + x) * 4;
         pixel[0] = encoded[0];
#if defined(G17P_BC5_DECODE)
         pixel[1] = encoded[8];
#else
         pixel[1] = 0;
#endif
         pixel[2] = 0;
         pixel[3] = 255;
      }
   }
#else
   uint8_t decoded[4 * COPY_DEPTH][16][4];
   for (unsigned block = 0; block < 4 * COPY_DEPTH; ++block) {
      const uint8_t *encoded = blocks + block_offset(block);
#if defined(G17P_BC2_DECODE)
      decode_bc1(encoded + 8, decoded[block]);
      for (unsigned texel = 0; texel < 16; ++texel)
         decoded[block][texel][3] = (encoded[0] & 0xf) * 17;
#elif defined(G17P_BC3_DECODE)
      decode_bc1(encoded + 8, decoded[block]);
      for (unsigned texel = 0; texel < 16; ++texel)
         decoded[block][texel][3] = encoded[0];
#else
      decode_bc1(encoded, decoded[block]);
#endif
   }

   for (unsigned z = 0; z < COPY_DEPTH; ++z) {
      for (unsigned y = 0; y < HEIGHT; ++y) {
         for (unsigned x = 0; x < WIDTH; ++x) {
            const unsigned block = z * 4 + (y / 4) * 2 + x / 4;
            const unsigned texel = (y % 4) * 4 + x % 4;
            const unsigned pixel = z * WIDTH * HEIGHT + y * WIDTH + x;
            memcpy(reference + pixel * 4, decoded[block][texel], 4);
         }
      }
   }
#endif
}
#elif defined(G17P_ETC2_RGBA8_DECODE)
enum etc2_variant_kind {
   ETC2_RGB8,
   ETC2_SRGB8,
   ETC2_RGB8A1,
   ETC2_SRGB8A1,
   ETC2_RGBA8,
   ETC2_SRGBA8,
   EAC_R11,
   EAC_R11_SIGNED,
   EAC_RG11,
   EAC_RG11_SIGNED,
};

struct etc2_variant {
   const char *name;
   VkFormat format;
   enum etc2_variant_kind kind;
};

static const struct etc2_variant etc2_variants[] = {
   {"rgb8-unorm", VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, ETC2_RGB8},
   {"rgb8-srgb", VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK, ETC2_SRGB8},
   {"rgb8a1-unorm", VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, ETC2_RGB8A1},
   {"rgb8a1-srgb", VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK, ETC2_SRGB8A1},
   {"rgba8-unorm", VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, ETC2_RGBA8},
   {"rgba8-srgb", VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, ETC2_SRGBA8},
   {"r11-unorm", VK_FORMAT_EAC_R11_UNORM_BLOCK, EAC_R11},
   {"r11-snorm", VK_FORMAT_EAC_R11_SNORM_BLOCK, EAC_R11_SIGNED},
   {"rg11-unorm", VK_FORMAT_EAC_R11G11_UNORM_BLOCK, EAC_RG11},
   {"rg11-snorm", VK_FORMAT_EAC_R11G11_SNORM_BLOCK, EAC_RG11_SIGNED},
};

static const struct etc2_variant *
select_etc2_variant(void)
{
   const char *requested = getenv("G17P_ETC2_VARIANT");
   if (!requested || !requested[0])
      requested = "rgba8-unorm";

   for (unsigned i = 0; i < sizeof(etc2_variants) / sizeof(etc2_variants[0]);
        ++i) {
      if (strcmp(requested, etc2_variants[i].name) == 0)
         return &etc2_variants[i];
   }

   fprintf(stderr, "%s: unknown G17P_ETC2_VARIANT '%s'\n", NAME, requested);
   exit(EXIT_FAILURE);
}

static void
build_blocks_and_reference(const struct etc2_variant *variant,
                           uint8_t *blocks, uint8_t *reference)
{
   memset(blocks, 0, BLOCK_DATA_BYTES);

   const bool signed_eac = variant->kind == EAC_R11_SIGNED ||
                           variant->kind == EAC_RG11_SIGNED;
   const bool rg_eac = variant->kind == EAC_RG11 ||
                       variant->kind == EAC_RG11_SIGNED;
   const bool eac = variant->kind >= EAC_R11;

   /* With multiplier/table/index zero, EAC R11 decodes an unsigned base B
    * as clamp(B*8 + 4 - 3), then bit-replicates 11->16.  Signed R11 uses
    * clamp(B*8 - 3) and signed 11->16 replication.  These are the exact
    * packUnorm4x8 results after that specified normalization. */
   static const uint8_t unsigned_base[4] = {0, 64, 128, 255};
   static const uint8_t unsigned_value[4] = {0, 64, 128, 254};
   static const uint8_t signed_base[4] = {1, 32, 64, 127};
   static const uint8_t signed_value[4] = {1, 63, 127, 253};
   static const uint8_t etc_base[4] = {0x00, 0x44, 0x88, 0xcc};
   static const uint8_t etc_value[4] = {2, 70, 138, 206};
   static const uint8_t etc_srgb_value[4] = {0, 16, 65, 157};

   if (eac) {
      const unsigned block_bytes = rg_eac ? 16 : 8;
      const uint8_t *base = signed_eac ? signed_base : unsigned_base;
      for (unsigned block = 0; block < 4; ++block) {
         blocks[block * block_bytes] = base[block];
         if (rg_eac)
            blocks[block * block_bytes + 8] = base[3 - block];
      }
   } else if (variant->kind == ETC2_RGB8 ||
              variant->kind == ETC2_SRGB8 ||
              variant->kind == ETC2_RGBA8 ||
              variant->kind == ETC2_SRGBA8) {
      const bool rgba = variant->kind == ETC2_RGBA8 ||
                        variant->kind == ETC2_SRGBA8;
      const unsigned block_bytes = rgba ? 16 : 8;
      const unsigned color_offset = rgba ? 8 : 0;
      for (unsigned block = 0; block < 4; ++block) {
         uint8_t *color = blocks + block * block_bytes + color_offset;
         color[0] = color[1] = color[2] = etc_base[block];
      }
   } else {
      /* Zero RGB8A1 is nonopaque differential mode.  Select index 2 in the
       * odd blocks to prove punch-through alpha independently of RGB. */
      for (unsigned block = 1; block < 4; block += 2) {
         uint8_t *color = blocks + block * 8;
         color[4] = color[5] = 0xff;
      }
   }

   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const unsigned block = (y / 4) * 2 + x / 4;
         uint8_t *pixel = reference + (y * WIDTH + x) * 4;
         pixel[0] = 0;
         pixel[1] = 0;
         pixel[2] = 0;
         pixel[3] = 255;

         if (eac) {
            const uint8_t *value =
               signed_eac ? signed_value : unsigned_value;
            pixel[0] = value[block];
            if (rg_eac)
               pixel[1] = value[3 - block];
         } else if (variant->kind == ETC2_RGB8 ||
                    variant->kind == ETC2_RGBA8) {
            pixel[0] = pixel[1] = pixel[2] = etc_value[block];
            if (variant->kind == ETC2_RGBA8)
               pixel[3] = 0;
         } else if (variant->kind == ETC2_SRGB8 ||
                    variant->kind == ETC2_SRGBA8) {
            pixel[0] = pixel[1] = pixel[2] = etc_srgb_value[block];
            if (variant->kind == ETC2_SRGBA8)
               pixel[3] = 0;
         } else if ((variant->kind == ETC2_RGB8A1 ||
                     variant->kind == ETC2_SRGB8A1) && (block & 1)) {
            pixel[3] = 0;
         }
      }
   }
}
#else
struct astc_variant {
   const char *name;
   VkFormat format;
   uint8_t block_width;
   uint8_t block_height;
};

#define ASTC_VARIANT(w, h)                                                    \
   {#w "x" #h "-unorm", VK_FORMAT_ASTC_##w##x##h##_UNORM_BLOCK, w, h},     \
   {#w "x" #h "-srgb", VK_FORMAT_ASTC_##w##x##h##_SRGB_BLOCK, w, h}

static const struct astc_variant astc_variants[] = {
   ASTC_VARIANT(4, 4),   ASTC_VARIANT(5, 4),
   ASTC_VARIANT(5, 5),   ASTC_VARIANT(6, 5),
   ASTC_VARIANT(6, 6),   ASTC_VARIANT(8, 5),
   ASTC_VARIANT(8, 6),   ASTC_VARIANT(8, 8),
   ASTC_VARIANT(10, 5),  ASTC_VARIANT(10, 6),
   ASTC_VARIANT(10, 8),  ASTC_VARIANT(10, 10),
   ASTC_VARIANT(12, 10), ASTC_VARIANT(12, 12),
};

#undef ASTC_VARIANT

static const struct astc_variant *
select_astc_variant(void)
{
   const char *requested = getenv("G17P_ASTC_VARIANT");
   if (!requested || !requested[0])
      return &astc_variants[0];

   for (unsigned i = 0; i < sizeof(astc_variants) / sizeof(astc_variants[0]);
        ++i) {
      if (strcmp(requested, astc_variants[i].name) == 0)
         return &astc_variants[i];
   }

   fprintf(stderr, NAME ": unknown G17P_ASTC_VARIANT '%s'\n", requested);
   exit(EXIT_FAILURE);
}

static void
put_astc_void_extent(uint8_t *block, uint16_t r, uint16_t g, uint16_t b,
                     uint16_t a)
{
   /* 0xdfc is the 2D LDR void-extent mode.  All four 13-bit extent
    * coordinates must be either a valid increasing rectangle or 0x1fff;
    * use the all-ones no-extent form so every texel takes the constant. */
   const uint16_t words[8] = {
      0xfdfc, 0xffff, 0xffff, 0xffff, r, g, b, a,
   };
   for (unsigned i = 0; i < 8; ++i) {
      block[2 * i] = words[i];
      block[2 * i + 1] = words[i] >> 8;
   }
}

static void
build_blocks_and_reference(uint8_t *blocks, uint8_t *reference,
                           unsigned block_width, unsigned block_height)
{
   static const uint16_t colours[4][4] = {
      {0xffff, 0, 0, 0xffff},
      {0, 0xffff, 0, 0xffff},
      {0, 0, 0xffff, 0xffff},
      {0xffff, 0xffff, 0xffff, 0xffff},
   };
   const unsigned blocks_x = (WIDTH + block_width - 1) / block_width;
   const unsigned blocks_y = (HEIGHT + block_height - 1) / block_height;
   const unsigned block_count = blocks_x * blocks_y;
   if (block_count > BLOCK_DATA_BYTES / BLOCK_BYTES)
      die("ASTC block upload exceeds fixture capacity");

   memset(blocks, 0, BLOCK_DATA_BYTES);
   for (unsigned block = 0; block < block_count; ++block)
      put_astc_void_extent(blocks + block * BLOCK_BYTES,
                           colours[block % 4][0],
                           colours[block % 4][1], colours[block % 4][2],
                           colours[block % 4][3]);
   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const unsigned block =
            (y / block_height) * blocks_x + x / block_width;
         uint8_t *pixel = reference + (y * WIDTH + x) * 4;
         for (unsigned c = 0; c < 4; ++c)
            pixel[c] = colours[block % 4][c] >> 8;
      }
   }
}
#endif

static void
verify(const uint8_t *upload, const uint8_t *output,
       const uint8_t *reference, uint64_t upload_hash, const char *when)
{
   for (size_t i = 0; i < GUARD_BYTES; ++i) {
      if (upload[i] != 0xa5 ||
          upload[GUARD_BYTES + BLOCK_DATA_BYTES + i] != 0xa5)
         die("upload guard changed");
      if (output[i] != 0xa5 || output[GUARD_BYTES + PIXEL_BYTES + i] != 0xa5)
         die("output guard changed");
   }

   if (fnv1a64(upload + GUARD_BYTES, BLOCK_DATA_BYTES) != upload_hash)
      die("upload payload or internal padding changed");

   const uint8_t *actual = output + GUARD_BYTES;
   if (memcmp(actual, reference, PIXEL_BYTES) != 0) {
      for (size_t i = 0; i < PIXEL_BYTES; ++i) {
         if (actual[i] != reference[i]) {
            fprintf(stderr, "%s: %s pixel %zu channel %zu is %02x, "
                            "expected %02x\n",
                    test_name, when, i / 4, i % 4, actual[i], reference[i]);
            exit(EXIT_FAILURE);
         }
      }
   }
}

#ifdef G17P_COMPRESSED_READBACK
static void
verify_readback(const uint8_t *upload, const uint8_t *readback,
                const char *when)
{
   if (memcmp(readback, upload, BLOCK_DATA_BYTES + 2 * GUARD_BYTES) != 0) {
      for (size_t i = 0; i < BLOCK_DATA_BYTES + 2 * GUARD_BYTES; ++i) {
         if (readback[i] != upload[i]) {
            fprintf(stderr, "%s: %s raw byte %zu is %02x, expected %02x\n",
                    test_name, when, i, readback[i], upload[i]);
            exit(EXIT_FAILURE);
         }
      }
   }
}
#endif

int
main(void)
{
#if defined(G17P_ASTC_DECODE) || defined(G17P_ASTC4X4_DECODE)
   const struct astc_variant *astc = select_astc_variant();
   const VkFormat test_format = astc->format;
#elif defined(G17P_ETC2_RGBA8_DECODE)
   const struct etc2_variant *etc2 = select_etc2_variant();
   const VkFormat test_format = etc2->format;
   test_name = etc2->name;
#else
   const VkFormat test_format = TEST_FORMAT;
#endif
   const bool record_only =
      getenv("G17P_COMPRESSED_DECODE_RECORD_ONLY") != NULL ||
      getenv("G17P_BC1_DECODE_RECORD_ONLY") != NULL;
   const bool submit_only =
      getenv("G17P_COMPRESSED_DECODE_SUBMIT_ONLY") != NULL;
   const bool expect_submit_failure =
      getenv("G17P_COMPRESSED_DECODE_EXPECT_SUBMIT_FAILURE") != NULL;
   if (record_only && submit_only)
      die("record-only and submit-only modes are mutually exclusive");
   if (expect_submit_failure &&
       (!submit_only ||
        getenv("ASAHI_SHIM_FAIL_SUBMIT_ONCE") == NULL))
      die("negative submit mode requires submit-only plus an injected shim "
          "submission failure");
   if (submit_only) {
      const char *validate = getenv("ASAHI_SHIM_VALIDATE_G17P_STREAM");
      if (!validate || !strcmp(validate, "0"))
         die("submit-only mode requires the G17P DRM-shim stream validator");
      if (!expect_submit_failure) {
         if (setenv("ASAHI_SHIM_G17P_EXPECT_COMPUTE", "1", 1))
            die("failed to enable exact G17P compute-stream validation");
         /* This count is also the compressed-upload performance contract.
          * Upload and readback must emit one launch per slice, never one per
          * compressed block.  The old nested block copy recorded nine
          * launches for this tiny image and 65,536 for one 1024x1024 slice. */
         /* The trailing strong-barrier counts are cache flushes that
          * strengthen a launch's own tail barrier rather than adding a
          * record, so the launch contract above is what the first two
          * fields still pin. One is the host-read flush
          * hk_CmdPipelineBarrier2() emits for the readback (6aae441547b).
          * The other is the TRANSFER -> COMPUTE_SHADER barrier between the
          * upload and the sample: with recorded dependencies (hk_sync.h)
          * that barrier no longer splits the compute stream, so the
          * upload's last launch carries the strong flush the sample's
          * reads need. */
         if (setenv("ASAHI_SHIM_G17P_EXPECT_COMPUTE_COUNTS",
#ifdef G17P_COMPRESSED_READBACK
                    /* Two-slice upload + sample + two-slice readback. */
                    "5,5,0,3",
#else
                    /* Two-slice upload + sample. */
                    "3,3,0,2",
#endif
                    1))
            die("failed to select exact compressed 3D stream counts");
      }
   }
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());
   vkGetInstanceProcAddr = dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("missing vkGetInstanceProcAddr");
   LOAD_GLOBAL(vkCreateInstance);

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = test_name,
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   check(vkCreateInstance(&ici, NULL, &instance), "vkCreateInstance");

#define I(n) LOAD_INSTANCE(instance, n)
   I(vkGetDeviceProcAddr);
   I(vkEnumeratePhysicalDevices);
   I(vkGetPhysicalDeviceFeatures);
   I(vkGetPhysicalDeviceProperties);
   I(vkGetPhysicalDeviceMemoryProperties);
   I(vkGetPhysicalDeviceQueueFamilyProperties);
   I(vkGetPhysicalDeviceFormatProperties);
   I(vkGetPhysicalDeviceImageFormatProperties);
   I(vkCreateDevice);
#undef I

   uint32_t count = 0;
   check(vkEnumeratePhysicalDevices(instance, &count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
   if (!devices)
      die("physical-device allocation failed");
   check(vkEnumeratePhysicalDevices(instance, &count, devices),
         "vkEnumeratePhysicalDevices(list)");
   VkPhysicalDevice physical = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < count; ++i) {
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(devices[i], &properties);
      if (properties.deviceID == DEVICE_ID)
         physical = devices[i];
   }
   free(devices);
   if (!physical)
      die("T8140/G17P device not found");

   VkPhysicalDeviceFeatures features;
   vkGetPhysicalDeviceFeatures(physical, &features);
   if (!features.textureCompressionBC || !features.textureCompressionETC2 ||
       !features.textureCompressionASTC_LDR)
      die("compressed-family feature gating is incorrect");

   static const VkFormat bc_formats[] = {
      VK_FORMAT_BC1_RGB_UNORM_BLOCK,  VK_FORMAT_BC1_RGB_SRGB_BLOCK,
      VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
      VK_FORMAT_BC2_UNORM_BLOCK,      VK_FORMAT_BC2_SRGB_BLOCK,
      VK_FORMAT_BC3_UNORM_BLOCK,      VK_FORMAT_BC3_SRGB_BLOCK,
      VK_FORMAT_BC4_UNORM_BLOCK,      VK_FORMAT_BC4_SNORM_BLOCK,
      VK_FORMAT_BC5_UNORM_BLOCK,      VK_FORMAT_BC5_SNORM_BLOCK,
      VK_FORMAT_BC6H_UFLOAT_BLOCK,    VK_FORMAT_BC6H_SFLOAT_BLOCK,
      VK_FORMAT_BC7_UNORM_BLOCK,      VK_FORMAT_BC7_SRGB_BLOCK,
   };
   const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                         VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                         READBACK_FORMAT_FEATURE;
   for (unsigned i = 0; i < sizeof(bc_formats) / sizeof(bc_formats[0]); ++i) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(physical, bc_formats[i], &props);
      if ((props.optimalTilingFeatures & required) != required)
         die("a BC family format lacks sampled/transfer support");
   }
   static const VkFormat etc2_formats[] = {
      VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
      VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
      VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
      VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
      VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
      VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
      VK_FORMAT_EAC_R11_UNORM_BLOCK,
      VK_FORMAT_EAC_R11_SNORM_BLOCK,
      VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
      VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
   };
   for (unsigned i = 0; i < sizeof(etc2_formats) / sizeof(etc2_formats[0]);
        ++i) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(physical, etc2_formats[i], &props);
      if ((props.optimalTilingFeatures & required) != required)
         die("an ETC2/EAC family format lacks sampled/transfer support");
   }
   static const VkFormat astc_formats[] = {
#define ASTC_FORMATS(w, h)                                                     \
      VK_FORMAT_ASTC_##w##x##h##_UNORM_BLOCK,                                 \
      VK_FORMAT_ASTC_##w##x##h##_SRGB_BLOCK
      ASTC_FORMATS(4, 4),   ASTC_FORMATS(5, 4),
      ASTC_FORMATS(5, 5),   ASTC_FORMATS(6, 5),
      ASTC_FORMATS(6, 6),   ASTC_FORMATS(8, 5),
      ASTC_FORMATS(8, 6),   ASTC_FORMATS(8, 8),
      ASTC_FORMATS(10, 5),  ASTC_FORMATS(10, 6),
      ASTC_FORMATS(10, 8),  ASTC_FORMATS(10, 10),
      ASTC_FORMATS(12, 10), ASTC_FORMATS(12, 12),
#undef ASTC_FORMATS
   };
   for (unsigned i = 0; i < sizeof(astc_formats) / sizeof(astc_formats[0]);
        ++i) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(physical, astc_formats[i], &props);
      if ((props.optimalTilingFeatures & required) != required)
         die("an ASTC LDR family format lacks sampled/transfer support");
   }

   VkFormatProperties target_props;
   vkGetPhysicalDeviceFormatProperties(physical, test_format, &target_props);
   if ((target_props.optimalTilingFeatures & required) != required)
      die("target compressed format lacks sampled/transfer support");
#ifdef G17P_COMPRESSED_3D
   VkImageFormatProperties image_props;
   check(vkGetPhysicalDeviceImageFormatProperties(
            physical, test_format, VK_IMAGE_TYPE_3D,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               READBACK_IMAGE_USAGE,
            0,
            &image_props),
         "vkGetPhysicalDeviceImageFormatProperties(3D compressed)");
   if (image_props.maxExtent.width < WIDTH ||
       image_props.maxExtent.height < HEIGHT ||
       image_props.maxExtent.depth < IMAGE_DEPTH)
      die("3D compressed image extent was advertised too small");
#endif

   uint32_t qcount = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &qcount, NULL);
   VkQueueFamilyProperties *qprops = calloc(qcount, sizeof(*qprops));
   if (!qprops)
      die("queue-family allocation failed");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &qcount, qprops);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < qcount; ++i) {
      if ((qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qprops[i].queueCount) {
         qf = i;
         break;
      }
   }
   free(qprops);
   if (qf == UINT32_MAX)
      die("compute queue unavailable");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &dci, NULL, &device), "vkCreateDevice");

#define D(n) LOAD_DEVICE(device, n)
   D(vkGetDeviceQueue);
   D(vkCreateBuffer);
   D(vkGetBufferMemoryRequirements);
   D(vkAllocateMemory);
   D(vkBindBufferMemory);
   D(vkMapMemory);
   D(vkCreateImage);
   D(vkGetImageMemoryRequirements);
   D(vkBindImageMemory);
   D(vkCreateImageView);
   D(vkCreateSampler);
   D(vkCreateDescriptorSetLayout);
   D(vkCreateDescriptorPool);
   D(vkAllocateDescriptorSets);
   D(vkUpdateDescriptorSets);
   D(vkCreateShaderModule);
   D(vkCreatePipelineLayout);
   D(vkCreateComputePipelines);
   D(vkCreateCommandPool);
   D(vkAllocateCommandBuffers);
   D(vkBeginCommandBuffer);
   D(vkCmdPipelineBarrier);
   D(vkCmdCopyBufferToImage);
   D(vkCmdCopyImageToBuffer);
   D(vkCmdBindPipeline);
   D(vkCmdBindDescriptorSets);
   D(vkCmdDispatch);
   D(vkEndCommandBuffer);
   D(vkCreateFence);
   D(vkQueueSubmit);
   D(vkWaitForFences);
#undef D

   VkQueue queue;
   vkGetDeviceQueue(device, qf, 0, &queue);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(physical, &mp);

   const VkDeviceSize buffer_sizes[] = {
      BLOCK_DATA_BYTES + 2 * GUARD_BYTES,
      PIXEL_BYTES + 2 * GUARD_BYTES,
#ifdef G17P_COMPRESSED_READBACK
      BLOCK_DATA_BYTES + 2 * GUARD_BYTES,
#endif
   };
   const VkBufferUsageFlags buffer_usages[] = {
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
#ifdef G17P_COMPRESSED_READBACK
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
#endif
   };
   _Static_assert(sizeof(buffer_sizes) / sizeof(buffer_sizes[0]) ==
                     BUFFER_COUNT,
                  "buffer count");
   _Static_assert(sizeof(buffer_usages) / sizeof(buffer_usages[0]) ==
                     BUFFER_COUNT,
                  "buffer usage count");
   VkBuffer buffers[BUFFER_COUNT];
   VkDeviceMemory memories[BUFFER_COUNT];
   uint8_t *maps[BUFFER_COUNT];
   for (unsigned i = 0; i < BUFFER_COUNT; ++i) {
      const VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = buffer_sizes[i],
         .usage = buffer_usages[i],
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      };
      check(vkCreateBuffer(device, &bci, NULL, &buffers[i]),
            "vkCreateBuffer");
      VkMemoryRequirements requirements;
      vkGetBufferMemoryRequirements(device, buffers[i], &requirements);
      const VkMemoryAllocateInfo allocation = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = requirements.size,
         .memoryTypeIndex = memory_type(
            &mp, requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      };
      check(vkAllocateMemory(device, &allocation, NULL, &memories[i]),
            "vkAllocateMemory(buffer)");
      check(vkBindBufferMemory(device, buffers[i], memories[i], 0),
            "vkBindBufferMemory");
      check(vkMapMemory(device, memories[i], 0, buffer_sizes[i], 0,
                        (void **)&maps[i]),
            "vkMapMemory");
      memset(maps[i], 0xa5, buffer_sizes[i]);
   }

   uint8_t *blocks = maps[0] + GUARD_BYTES;
#if defined(G17P_ETC2_RGBA8_DECODE)
   uint8_t reference[PIXEL_BYTES];
   build_blocks_and_reference(etc2, blocks, reference);
#elif defined(G17P_ASTC_DECODE) || defined(G17P_ASTC4X4_DECODE)
   uint8_t reference[PIXEL_BYTES];
   build_blocks_and_reference(blocks, reference, astc->block_width,
                              astc->block_height);
#else
#if defined(G17P_BC1_SRGB_DECODE) || defined(G17P_BC2_SRGB_DECODE) ||     \
   defined(G17P_BC3_SRGB_DECODE)
   /* Each block selects a nontrivial RGB565 endpoint.  The reference above
    * independently applies the Vulkan sRGB decode before packUnorm4x8. */
   put_block(blocks, 0, 0x8410, 0x0000, 0);
   put_block(blocks, 1, 0x4208, 0x0000, 0);
   put_block(blocks, 2, 0xc618, 0x0000, 0);
   put_block(blocks, 3, 0xffff, 0x0000, 0);
#else
   put_block(blocks, 0, 0xf800, 0x0000, 0);
   put_block(blocks, 1, 0x07e0, 0x0000, 0);
   put_block(blocks, 2, 0x001f, 0x0000, 0);
#if defined(G17P_BC2_DECODE) || defined(G17P_BC3_DECODE)
   put_block(blocks, 3, 0xffff, 0x0000, 0);
#else
   /* Keep the exact oracle independent of the implementation's permitted
    * interpolation rounding: c0 <= c1 selects BC1 alpha mode and 0/1/3 use
    * only black, white and transparent black. */
   put_block(blocks, 3, 0x0000, 0xffff, UINT32_C(0x74747474));
#endif
#ifdef G17P_COMPRESSED_3D
   put_block(blocks, 4, 0x07ff, 0x0000, 0);
   put_block(blocks, 5, 0xf81f, 0x0000, 0);
   put_block(blocks, 6, 0xffe0, 0x0000, 0);
   put_block(blocks, 7, 0x0000, 0x0000, 0);
#endif
#endif
   uint8_t reference[PIXEL_BYTES];
   build_reference(blocks, reference);
#endif
   const uint64_t upload_hash = fnv1a64(blocks, BLOCK_DATA_BYTES);

   const VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
#ifdef G17P_COMPRESSED_3D
      .imageType = VK_IMAGE_TYPE_3D,
#else
      .imageType = VK_IMAGE_TYPE_2D,
#endif
      .format = test_format,
      .extent = {WIDTH, HEIGHT, IMAGE_DEPTH},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               READBACK_IMAGE_USAGE,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkImage image;
   check(vkCreateImage(device, &imci, NULL, &image), "vkCreateImage");
   VkMemoryRequirements image_requirements;
   vkGetImageMemoryRequirements(device, image, &image_requirements);
   const VkMemoryAllocateInfo image_allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = image_requirements.size,
      .memoryTypeIndex = memory_type(&mp, image_requirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory image_memory;
   check(vkAllocateMemory(device, &image_allocation, NULL, &image_memory),
         "vkAllocateMemory(image)");
   check(vkBindImageMemory(device, image, image_memory, 0),
         "vkBindImageMemory");

   const VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
#ifdef G17P_COMPRESSED_3D
      .viewType = VK_IMAGE_VIEW_TYPE_3D,
#else
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
#endif
      .format = test_format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkImageView view;
   check(vkCreateImageView(device, &ivci, NULL, &view),
         "vkCreateImageView");
   const VkSamplerCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = 0.0f,
   };
   VkSampler sampler;
   check(vkCreateSampler(device, &sci, NULL, &sampler), "vkCreateSampler");

   const VkDescriptorSetLayoutBinding bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
       VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
   };
   const VkDescriptorSetLayoutCreateInfo dslci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout set_layout;
   check(vkCreateDescriptorSetLayout(device, &dslci, NULL, &set_layout),
         "vkCreateDescriptorSetLayout");
   const VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkPipelineLayout pipeline_layout;
   check(vkCreatePipelineLayout(device, &plci, NULL, &pipeline_layout),
         "vkCreatePipelineLayout");

   const VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
#ifdef G17P_COMPRESSED_3D
      .codeSize = sizeof(g17p_vulkan_bc1_decode_3d_spv),
      .pCode = g17p_vulkan_bc1_decode_3d_spv,
#else
      .codeSize = sizeof(g17p_vulkan_bc1_decode_spv),
      .pCode = g17p_vulkan_bc1_decode_spv,
#endif
   };
   VkShaderModule shader;
   check(vkCreateShaderModule(device, &smci, NULL, &shader),
         "vkCreateShaderModule");
   const VkComputePipelineCreateInfo cpipeline = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = shader,
         .pName = "main",
      },
      .layout = pipeline_layout,
   };
   VkPipeline pipeline;
   check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpipeline, NULL,
                                  &pipeline),
         "vkCreateComputePipelines");

   const VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
   };
   const VkDescriptorPoolCreateInfo dpci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool descriptor_pool;
   check(vkCreateDescriptorPool(device, &dpci, NULL, &descriptor_pool),
         "vkCreateDescriptorPool");
   const VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkDescriptorSet descriptor_set;
   check(vkAllocateDescriptorSets(device, &dsai, &descriptor_set),
         "vkAllocateDescriptorSets");
   const VkDescriptorImageInfo image_desc = {
      .sampler = sampler,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   const VkDescriptorBufferInfo output_desc = {
      .buffer = buffers[1],
      .offset = GUARD_BYTES,
      .range = PIXEL_BYTES,
   };
   const VkWriteDescriptorSet writes[] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &image_desc,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &output_desc,
      },
   };
   vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

   const VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = qf,
   };
   VkCommandPool command_pool;
   check(vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command_buffer;
   check(vkAllocateCommandBuffers(device, &cbai, &command_buffer),
         "vkAllocateCommandBuffers");
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   check(vkBeginCommandBuffer(command_buffer, &begin),
         "vkBeginCommandBuffer");

   VkImageMemoryBarrier image_barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &image_barrier);
   const VkBufferImageCopy copy = {
      .bufferOffset = GUARD_BYTES,
      .bufferRowLength = BUFFER_ROW_LENGTH,
      .bufferImageHeight = BUFFER_IMAGE_HEIGHT,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
#ifdef G17P_COMPRESSED_3D
      .imageOffset = {0, 0, 1},
#endif
      .imageExtent = {WIDTH, HEIGHT, COPY_DEPTH},
   };
   vkCmdCopyBufferToImage(command_buffer, buffers[0], image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

   image_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
   image_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0,
                        NULL, 1, &image_barrier);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
   vkCmdDispatch(command_buffer, 1, 1, COPY_DEPTH);

   const VkBufferMemoryBarrier output_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buffers[1],
      .offset = GUARD_BYTES,
      .size = PIXEL_BYTES,
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1,
                        &output_barrier, 0, NULL);
#ifdef G17P_COMPRESSED_READBACK
   image_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
   image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   image_barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &image_barrier);
   vkCmdCopyImageToBuffer(command_buffer, image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffers[2], 1,
                          &copy);
   const VkBufferMemoryBarrier readback_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buffers[2],
      .offset = GUARD_BYTES,
      .size = BLOCK_DATA_BYTES,
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1,
                        &readback_barrier, 0, NULL);
#endif
   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

   if (record_only) {
#ifdef G17P_COMPRESSED_READBACK
      printf("%s: PASS - compressed upload/sample/raw-readback dispatch "
             "recorded\n", test_name);
#else
      printf("%s: PASS - compressed upload/sample dispatch recorded\n",
             test_name);
#endif
      return EXIT_SUCCESS;
   }

   const VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   check(vkCreateFence(device, &fci, NULL, &fence), "vkCreateFence");
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
   };
   const VkResult submitted = vkQueueSubmit(queue, 1, &submit, fence);
   if (expect_submit_failure) {
      if (submitted == VK_SUCCESS)
         die("injected compressed 3D submission failure was ignored");
      printf("%s: PASS - compressed 3D submission failed closed (%d)\n",
             test_name, submitted);
      return EXIT_SUCCESS;
   }
   check(submitted, "vkQueueSubmit");
   VkResult waited =
      vkWaitForFences(device, 1, &fence, true, UINT64_C(3000000000));
   if (waited == VK_TIMEOUT)
      die("fence timed out after three seconds");
   check(waited, "vkWaitForFences");

   if (submit_only) {
#ifdef G17P_COMPRESSED_READBACK
      printf("%s: PASS - compressed 3D upload/sample/raw-readback stream "
             "submitted to the strict G17P DRM-shim validator\n",
             test_name);
#else
      printf("%s: PASS - compressed 3D upload/sample stream submitted to "
             "the strict G17P DRM-shim validator\n",
             test_name);
#endif
      return EXIT_SUCCESS;
   }

   verify(maps[0], maps[1], reference, upload_hash, "immediate");
#ifdef G17P_COMPRESSED_READBACK
   verify_readback(maps[0], maps[2], "immediate");
#endif
   const struct timespec stable = {.tv_sec = 3};
   if (nanosleep(&stable, NULL))
      die("stability sleep interrupted");
   verify(maps[0], maps[1], reference, upload_hash, "after three seconds");
#ifdef G17P_COMPRESSED_READBACK
   verify_readback(maps[0], maps[2], "after three seconds");
#endif
   printf("%s: PASS - %u decoded pixels exact immediately/3s, "
#ifdef G17P_COMPRESSED_READBACK
          "%u raw compressed bytes/padding exact, 384 guard bytes unchanged, "
#else
          "256 guard bytes unchanged, "
#endif
          "checksum %016" PRIx64 "\n",
          test_name, WIDTH * HEIGHT * COPY_DEPTH,
#ifdef G17P_COMPRESSED_READBACK
          BLOCK_DATA_BYTES,
#endif
          fnv1a64(maps[1] + GUARD_BYTES, PIXEL_BYTES));
   return EXIT_SUCCESS;
}
