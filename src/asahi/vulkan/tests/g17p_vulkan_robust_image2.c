/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact robustImageAccess2 oracle. The storage image is only 4x4, while its
 * allocation remains valid, so an out-of-view alias is observable without
 * relying on a GPU fault.
 */

#include <vulkan/vulkan.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "g17p_vulkan_robust_image2_spv.h"

#define TEST_NAME           "g17p-vulkan-robust-image2"
#define T8140_DEVICE_ID     0x8140
#define IMAGE_WIDTH         4
#define IMAGE_HEIGHT        4
#define IMAGE_PIXELS        (IMAGE_WIDTH * IMAGE_HEIGHT)
#define IMAGE_BYTES         (IMAGE_PIXELS * 4)
#define OUTPUT_OFFSET       128
#define OUTPUT_VECTORS      8
#define OUTPUT_BYTES        (OUTPUT_VECTORS * 4 * sizeof(uint32_t))
#define READBACK_OFFSET     256
#define BUFFER_SIZE         512
#define BUFFER_SENTINEL     0xa5
#define WAIT_TIMEOUT_NS     UINT64_C(10000000000)

static void
die(const char *what)
{
   fprintf(stderr, TEST_NAME ": %s\n", what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, TEST_NAME ": %s returned %d\n", what, result);
      exit(EXIT_FAILURE);
   }
}

static uint32_t
find_memory_type(const VkPhysicalDeviceMemoryProperties *properties,
                 uint32_t allowed, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((allowed & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }

   die("required memory type is unavailable");
   return UINT32_MAX;
}

static void
source_pixel(uint32_t index, uint8_t rgba[4])
{
   rgba[0] = 0x10 + index;
   rgba[1] = 0x40 + index;
   rgba[2] = 0x80 + index;
   rgba[3] = 0xc0 + index;
}

static void
initialize_buffer(uint8_t *mapped)
{
   memset(mapped, BUFFER_SENTINEL, BUFFER_SIZE);
   for (uint32_t i = 0; i < IMAGE_PIXELS; ++i)
      source_pixel(i, mapped + (4 * i));
}

static void
verify_result(const uint8_t *mapped, const char *suffix)
{
   for (uint32_t i = 0; i < IMAGE_PIXELS; ++i) {
      uint8_t expected[4];
      source_pixel(i, expected);
      if (memcmp(mapped + (4 * i), expected, sizeof(expected))) {
         fprintf(stderr, TEST_NAME ": upload pixel %u changed%s\n", i,
                 suffix);
         exit(EXIT_FAILURE);
      }
   }

   for (uint32_t i = IMAGE_BYTES; i < OUTPUT_OFFSET; ++i) {
      if (mapped[i] != BUFFER_SENTINEL)
         die("prefix guard changed");
   }

   const uint32_t *output =
      (const uint32_t *)(mapped + OUTPUT_OFFSET);
   uint8_t pixel0[4], pixel14[4];
   source_pixel(0, pixel0);
   source_pixel(14, pixel14);
   for (uint32_t c = 0; c < 4; ++c) {
      if (output[c] != pixel0[c]) {
         fprintf(stderr,
                 TEST_NAME ": in-range load[0].%u is %#x, expected %#x%s\n",
                 c, output[c], pixel0[c], suffix);
         exit(EXIT_FAILURE);
      }
      if (output[4 + c] != pixel14[c]) {
         fprintf(stderr,
                 TEST_NAME ": in-range load[14].%u is %#x, expected %#x%s\n",
                 c, output[4 + c], pixel14[c], suffix);
         exit(EXIT_FAILURE);
      }
   }

   for (uint32_t i = 8; i < 24; ++i) {
      if (output[i] != 0) {
         fprintf(stderr,
                 TEST_NAME ": OOB load word %u leaked %#x, expected zero%s\n",
                 i, output[i], suffix);
         exit(EXIT_FAILURE);
      }
   }
   for (uint32_t i = 24; i < 32; ++i) {
      if (output[i] != UINT32_C(0xa5a5a5a5))
         die("output guard changed");
   }

   const uint8_t *readback = mapped + READBACK_OFFSET;
   for (uint32_t i = 0; i < IMAGE_PIXELS; ++i) {
      uint8_t expected[4];
      source_pixel(i, expected);
      if (i == 15) {
         expected[0] = 9;
         expected[1] = 8;
         expected[2] = 7;
         expected[3] = 6;
      }
      if (memcmp(readback + (4 * i), expected, sizeof(expected))) {
         fprintf(stderr,
                 TEST_NAME ": image pixel %u changed through an OOB store%s\n",
                 i, suffix);
         exit(EXIT_FAILURE);
      }
   }

   for (uint32_t i = READBACK_OFFSET + IMAGE_BYTES; i < BUFFER_SIZE; ++i) {
      if (mapped[i] != BUFFER_SENTINEL)
         die("readback guard changed");
   }
}

int
main(int argc, char **argv)
{
   const bool record_only = argc == 2 && !strcmp(argv[1], "--record-only");
   if (argc > 2 || (argc == 2 && !record_only))
      die("usage: " TEST_NAME " [--record-only]");

   const VkApplicationInfo application_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = TEST_NAME,
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &application_info,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

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
   for (uint32_t i = 0; i < physical_count; ++i) {
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(physical_devices[i], &properties);
      if (properties.deviceID == T8140_DEVICE_ID) {
         physical = physical_devices[i];
         break;
      }
   }
   free(physical_devices);
   if (physical == VK_NULL_HANDLE)
      die("T8140/G17P Vulkan device was not found");

   VkPhysicalDeviceRobustness2FeaturesEXT available_robustness2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 available_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &available_robustness2,
   };
   vkGetPhysicalDeviceFeatures2(physical, &available_features);
   if (!available_robustness2.robustImageAccess2)
      die("robustImageAccess2 is not advertised");

   VkFormatProperties format_properties;
   vkGetPhysicalDeviceFormatProperties(physical, VK_FORMAT_R8G8B8A8_UINT,
                                       &format_properties);
   const VkFormatFeatureFlags required_format =
      VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
      VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   if ((format_properties.optimalTilingFeatures & required_format) !=
       required_format)
      die("RGBA8_UINT storage/copy format support is unavailable");

   uint32_t queue_family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_family_count,
                                            NULL);
   VkQueueFamilyProperties *queue_families =
      calloc(queue_family_count, sizeof(*queue_families));
   if (!queue_families)
      die("out of host memory");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_family_count,
                                            queue_families);
   uint32_t queue_family = UINT32_MAX;
   for (uint32_t i = 0; i < queue_family_count; ++i) {
      if (queue_families[i].queueCount &&
          (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
         queue_family = i;
         break;
      }
   }
   if (queue_family == UINT32_MAX)
      die("device has no compute queue family");
   const VkQueueFlags queue_flags = queue_families[queue_family].queueFlags;
   free(queue_families);

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   VkPhysicalDeviceRobustness2FeaturesEXT enabled_robustness2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
      .robustImageAccess2 = VK_TRUE,
   };
   VkPhysicalDeviceFeatures2 enabled_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &enabled_robustness2,
   };
   const char *const enabled_extensions[] = {
      VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enabled_features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = enabled_extensions,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");
   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);

   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = BUFFER_SIZE,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer buffer;
   check(vkCreateBuffer(device, &buffer_info, NULL, &buffer),
         "vkCreateBuffer");
   VkMemoryRequirements buffer_requirements;
   vkGetBufferMemoryRequirements(device, buffer, &buffer_requirements);
   const VkMemoryAllocateInfo buffer_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = buffer_requirements.size,
      .memoryTypeIndex = find_memory_type(
         &memory_properties, buffer_requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory buffer_memory;
   check(vkAllocateMemory(device, &buffer_memory_info, NULL, &buffer_memory),
         "vkAllocateMemory(buffer)");
   check(vkBindBufferMemory(device, buffer, buffer_memory, 0),
         "vkBindBufferMemory");
   uint8_t *mapped;
   check(vkMapMemory(device, buffer_memory, 0, BUFFER_SIZE, 0,
                     (void **)&mapped),
         "vkMapMemory");
   initialize_buffer(mapped);

   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UINT,
      .extent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   check(vkCreateImage(device, &image_info, NULL, &image), "vkCreateImage");
   VkMemoryRequirements image_requirements;
   vkGetImageMemoryRequirements(device, image, &image_requirements);
   const VkMemoryAllocateInfo image_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = image_requirements.size,
      .memoryTypeIndex = find_memory_type(
         &memory_properties, image_requirements.memoryTypeBits, 0),
   };
   VkDeviceMemory image_memory;
   check(vkAllocateMemory(device, &image_memory_info, NULL, &image_memory),
         "vkAllocateMemory(image)");
   check(vkBindImageMemory(device, image, image_memory, 0),
         "vkBindImageMemory");

   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UINT,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1,
         .layerCount = 1,
      },
   };
   VkImageView image_view;
   check(vkCreateImageView(device, &view_info, NULL, &image_view),
         "vkCreateImageView");

   const VkDescriptorSetLayoutBinding bindings[] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
   };
   const VkDescriptorSetLayoutCreateInfo set_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout set_layout;
   check(vkCreateDescriptorSetLayout(device, &set_layout_info, NULL,
                                     &set_layout),
         "vkCreateDescriptorSetLayout");
   const VkDescriptorPoolSize pool_sizes[] = {
      {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1},
      {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1},
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool descriptor_pool;
   check(vkCreateDescriptorPool(device, &pool_info, NULL, &descriptor_pool),
         "vkCreateDescriptorPool");
   const VkDescriptorSetAllocateInfo set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkDescriptorSet descriptor_set;
   check(vkAllocateDescriptorSets(device, &set_info, &descriptor_set),
         "vkAllocateDescriptorSets");
   const VkDescriptorImageInfo descriptor_image = {
      .imageView = image_view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   const VkDescriptorBufferInfo descriptor_buffer = {
      .buffer = buffer,
      .offset = OUTPUT_OFFSET,
      .range = OUTPUT_BYTES,
   };
   const VkWriteDescriptorSet descriptor_writes[] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .pImageInfo = &descriptor_image,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &descriptor_buffer,
      },
   };
   vkUpdateDescriptorSets(device, 2, descriptor_writes, 0, NULL);

   const VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_robust_image2_spv),
      .pCode = g17p_vulkan_robust_image2_spv,
   };
   VkShaderModule shader;
   check(vkCreateShaderModule(device, &shader_info, NULL, &shader),
         "vkCreateShaderModule");
   const VkPipelineLayoutCreateInfo pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkPipelineLayout pipeline_layout;
   check(vkCreatePipelineLayout(device, &pipeline_layout_info, NULL,
                                &pipeline_layout),
         "vkCreatePipelineLayout");
   const VkComputePipelineCreateInfo pipeline_info = {
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
   check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                  NULL, &pipeline),
         "vkCreateComputePipelines");

   const VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool command_pool;
   check(vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command_buffer;
   check(vkAllocateCommandBuffers(device, &command_buffer_info,
                                  &command_buffer),
         "vkAllocateCommandBuffers");
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command_buffer, &begin_info),
         "vkBeginCommandBuffer");

   VkImageMemoryBarrier image_barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1,
         .layerCount = 1,
      },
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &image_barrier);
   const VkBufferImageCopy upload_region = {
      .bufferOffset = 0,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .layerCount = 1,
      },
      .imageExtent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
   };
   vkCmdCopyBufferToImage(command_buffer, buffer, image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                          &upload_region);
   image_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   image_barrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
   image_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   image_barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0,
                        NULL, 1, &image_barrier);

   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
   vkCmdDispatch(command_buffer, 6, 1, 1);

   image_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
   image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   image_barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
   image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &image_barrier);
   const VkBufferImageCopy readback_region = {
      .bufferOffset = READBACK_OFFSET,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .layerCount = 1,
      },
      .imageExtent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
   };
   vkCmdCopyImageToBuffer(command_buffer, image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1,
                          &readback_region);
   const VkBufferMemoryBarrier host_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buffer,
      .offset = 0,
      .size = BUFFER_SIZE,
   };
   vkCmdPipelineBarrier(command_buffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                           VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1,
                        &host_barrier, 0, NULL);
   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

   VkFence fence = VK_NULL_HANDLE;
   if (record_only) {
      printf(TEST_NAME ": RECORD PASS — robust RGBA8UI image load/store, "
                       "upload/readback and EndCB accepted on queue %u "
                       "(flags %#x)\n",
             queue_family, queue_flags);
   } else {
      const VkFenceCreateInfo fence_info = {
         .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      };
      check(vkCreateFence(device, &fence_info, NULL, &fence),
            "vkCreateFence");
      const VkSubmitInfo submit_info = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &command_buffer,
      };
      check(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
      check(vkWaitForFences(device, 1, &fence, VK_TRUE, WAIT_TIMEOUT_NS),
            "vkWaitForFences");
      verify_result(mapped, " immediately after the fence");
      sleep(3);
      verify_result(mapped, " after three seconds");
      puts(TEST_NAME ": PASS — in-range loads/store exact, four OOB loads "
                     "zero, four OOB stores discarded, all guards stable");
   }

   if (fence != VK_NULL_HANDLE)
      vkDestroyFence(device, fence, NULL);
   vkDestroyCommandPool(device, command_pool, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, pipeline_layout, NULL);
   vkDestroyShaderModule(device, shader, NULL);
   vkDestroyDescriptorPool(device, descriptor_pool, NULL);
   vkDestroyDescriptorSetLayout(device, set_layout, NULL);
   vkDestroyImageView(device, image_view, NULL);
   vkDestroyImage(device, image, NULL);
   vkFreeMemory(device, image_memory, NULL);
   vkUnmapMemory(device, buffer_memory);
   vkDestroyBuffer(device, buffer, NULL);
   vkFreeMemory(device, buffer_memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return 0;
}
