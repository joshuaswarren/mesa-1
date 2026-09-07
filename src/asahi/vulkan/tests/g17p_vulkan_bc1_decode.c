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

#include "g17p_vulkan_bc1_decode_spv.h"

#define WIDTH 8
#define HEIGHT 8
#define BLOCK_BYTES 8
#define BLOCK_DATA_BYTES 32
#define PIXEL_BYTES (WIDTH * HEIGHT * 4)
#define GUARD_BYTES 64
#define DEVICE_ID 0x8140
#define NAME "g17p-vulkan-bc1-decode"

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
   fprintf(stderr, NAME ": %s\n", message);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *call)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, NAME ": %s returned %d\n", call, result);
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

static void
put_block(uint8_t *data, unsigned block, uint16_t c0, uint16_t c1,
          uint32_t indices)
{
   uint8_t *out = data + block * BLOCK_BYTES;
   out[0] = c0;
   out[1] = c0 >> 8;
   out[2] = c1;
   out[3] = c1 >> 8;
   out[4] = indices;
   out[5] = indices >> 8;
   out[6] = indices >> 16;
   out[7] = indices >> 24;
}

static void
build_reference(const uint8_t *blocks, uint8_t *reference)
{
   uint8_t decoded[4][16][4];
   for (unsigned block = 0; block < 4; ++block)
      decode_bc1(blocks + block * BLOCK_BYTES, decoded[block]);

   for (unsigned y = 0; y < HEIGHT; ++y) {
      for (unsigned x = 0; x < WIDTH; ++x) {
         const unsigned block = (y / 4) * 2 + x / 4;
         const unsigned texel = (y % 4) * 4 + x % 4;
         memcpy(reference + (y * WIDTH + x) * 4, decoded[block][texel], 4);
      }
   }
}

static void
verify(const uint8_t *upload, const uint8_t *output,
       const uint8_t *reference, const char *when)
{
   for (size_t i = 0; i < GUARD_BYTES; ++i) {
      if (upload[i] != 0xa5 ||
          upload[GUARD_BYTES + BLOCK_DATA_BYTES + i] != 0xa5)
         die("upload guard changed");
      if (output[i] != 0xa5 || output[GUARD_BYTES + PIXEL_BYTES + i] != 0xa5)
         die("output guard changed");
   }

   const uint8_t *actual = output + GUARD_BYTES;
   if (memcmp(actual, reference, PIXEL_BYTES) != 0) {
      for (size_t i = 0; i < PIXEL_BYTES; ++i) {
         if (actual[i] != reference[i]) {
            fprintf(stderr,
                    NAME ": %s byte %zu is %02x, expected %02x\n",
                    when, i, actual[i], reference[i]);
            exit(EXIT_FAILURE);
         }
      }
   }
}

int
main(void)
{
   const bool record_only = getenv("G17P_BC1_DECODE_RECORD_ONLY") != NULL;
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());
   vkGetInstanceProcAddr = dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("missing vkGetInstanceProcAddr");
   LOAD_GLOBAL(vkCreateInstance);

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = NAME,
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
   if (!features.textureCompressionBC || features.textureCompressionETC2 ||
       features.textureCompressionASTC_LDR)
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
                                         VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   for (unsigned i = 0; i < sizeof(bc_formats) / sizeof(bc_formats[0]); ++i) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(physical, bc_formats[i], &props);
      if ((props.optimalTilingFeatures & required) != required)
         die("a BC family format lacks sampled/upload support");
   }
   static const VkFormat closed[] = {
      VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
      VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
   };
   for (unsigned i = 0; i < sizeof(closed) / sizeof(closed[0]); ++i) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(physical, closed[i], &props);
      if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
         die("an unqualified compressed family is advertised");
   }

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
   };
   VkBuffer buffers[2];
   VkDeviceMemory memories[2];
   uint8_t *maps[2];
   for (unsigned i = 0; i < 2; ++i) {
      const VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = buffer_sizes[i],
         .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
   put_block(blocks, 0, 0xf800, 0x0000, 0);
   put_block(blocks, 1, 0x07e0, 0x0000, 0);
   put_block(blocks, 2, 0x001f, 0x0000, 0);
   put_block(blocks, 3, 0xffff, 0x0000, UINT32_C(0xe4e4e4e4));
   uint8_t reference[PIXEL_BYTES];
   build_reference(blocks, reference);

   const VkImageCreateInfo imci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
      .extent = {WIDTH, HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
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
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
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
      .codeSize = sizeof(g17p_vulkan_bc1_decode_spv),
      .pCode = g17p_vulkan_bc1_decode_spv,
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
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {WIDTH, HEIGHT, 1},
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
   vkCmdDispatch(command_buffer, 1, 1, 1);

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
   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

   if (record_only) {
      puts(NAME ": PASS - complete BC family advertised, BC1 decode recorded");
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
   check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
   VkResult waited =
      vkWaitForFences(device, 1, &fence, true, UINT64_C(3000000000));
   if (waited == VK_TIMEOUT)
      die("fence timed out after three seconds");
   check(waited, "vkWaitForFences");

   verify(maps[0], maps[1], reference, "immediate");
   const struct timespec stable = {.tv_sec = 3};
   if (nanosleep(&stable, NULL))
      die("stability sleep interrupted");
   verify(maps[0], maps[1], reference, "after three seconds");
   printf(NAME ": PASS - 64 decoded pixels exact immediately/3s, "
               "256 guard bytes unchanged, checksum %016" PRIx64 "\n",
          fnv1a64(maps[1] + GUARD_BYTES, PIXEL_BYTES));
   return EXIT_SUCCESS;
}
