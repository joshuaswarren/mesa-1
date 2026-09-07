/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public Vulkan oracle for robustBufferAccess2 UBO and SSBO range checks.
 * Mapped poison follows both descriptor ranges, distinguishing descriptor
 * bounds from allocation/VM bounds and detecting stray stores.
 */

#include <vulkan/vulkan.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "g17p_vulkan_robust_buffer2_spv.h"

#define TEST_NAME            "g17p-vulkan-robust-buffer2"
#define T8140_DEVICE_ID      0x8140
#define UBO_RANGE            64
#define SSBO_OFFSET          256
#define SSBO_RANGE           64
#define OUTPUT_OFFSET        512
#define OUTPUT_WORDS         32
#define OUTPUT_GUARD_WORDS   8
#define BUFFER_SIZE          1024
#define INPUT_WORD(i)        (UINT32_C(0x41000000) + (i))
#define POISON_WORD(i)       (UINT32_C(0xbad00000) + (i))
#define SSBO_INPUT_WORD(i)   (UINT32_C(0x52000000) + (i))
#define SSBO_POISON_WORD(i)  (UINT32_C(0xced00000) + (i))
#define SSBO_STORE_WORD(i)   (UINT32_C(0x63000000) + (i))
#define OUTPUT_SENTINEL(i)   (UINT32_C(0xa5a50000) | (i))

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

   die("host-visible coherent memory is unavailable");
   return UINT32_MAX;
}

static void
verify_result(const uint32_t *mapped, const char *suffix)
{
   for (uint32_t i = 0; i < 4; ++i) {
      if (mapped[(OUTPUT_OFFSET / 4) + i] != INPUT_WORD(i)) {
         fprintf(stderr,
                 TEST_NAME ": in-range UBO word %u is %#x, expected %#x%s\n",
                 i, mapped[(OUTPUT_OFFSET / 4) + i], INPUT_WORD(i), suffix);
         exit(EXIT_FAILURE);
      }
   }

   for (uint32_t i = 0; i < 4; ++i) {
      if (mapped[(OUTPUT_OFFSET / 4) + 4 + i] != INPUT_WORD(12 + i)) {
         fprintf(stderr,
                 TEST_NAME ": exact-end UBO word %u is %#x, expected %#x%s\n",
                 i, mapped[(OUTPUT_OFFSET / 4) + 4 + i],
                 INPUT_WORD(12 + i), suffix);
         exit(EXIT_FAILURE);
      }
   }

   for (uint32_t i = 8; i < 16; ++i) {
      if (mapped[(OUTPUT_OFFSET / 4) + i] != 0) {
         fprintf(stderr,
                 TEST_NAME ": OOB UBO word %u leaked %#x, expected zero%s\n",
                 i, mapped[(OUTPUT_OFFSET / 4) + i], suffix);
         exit(EXIT_FAILURE);
      }
   }

   for (uint32_t i = 0; i < 4; ++i) {
      if (mapped[(OUTPUT_OFFSET / 4) + 16 + i] != SSBO_INPUT_WORD(i)) {
         fprintf(stderr,
                 TEST_NAME ": in-range SSBO word %u is %#x, expected %#x%s\n",
                 i, mapped[(OUTPUT_OFFSET / 4) + 16 + i],
                 SSBO_INPUT_WORD(i), suffix);
         exit(EXIT_FAILURE);
      }
   }

   for (uint32_t i = 0; i < 4; ++i) {
      if (mapped[(OUTPUT_OFFSET / 4) + 20 + i] !=
          SSBO_INPUT_WORD(12 + i)) {
         fprintf(stderr,
                 TEST_NAME ": exact-end SSBO word %u is %#x, expected %#x%s\n",
                 i, mapped[(OUTPUT_OFFSET / 4) + 20 + i],
                 SSBO_INPUT_WORD(12 + i), suffix);
         exit(EXIT_FAILURE);
      }
   }

   for (uint32_t i = 24; i < OUTPUT_WORDS; ++i) {
      if (mapped[(OUTPUT_OFFSET / 4) + i] != 0) {
         fprintf(stderr,
                 TEST_NAME ": OOB SSBO word %u leaked %#x, expected zero%s\n",
                 i, mapped[(OUTPUT_OFFSET / 4) + i], suffix);
         exit(EXIT_FAILURE);
      }
   }

   for (uint32_t i = 16; i < 32; ++i) {
      if (mapped[i] != POISON_WORD(i))
         die("mapped poison after the UBO descriptor range changed");
   }

   for (uint32_t i = 0; i < 12; ++i) {
      if (mapped[(SSBO_OFFSET / 4) + i] != SSBO_INPUT_WORD(i))
         die("in-range SSBO prefix changed");
   }

   for (uint32_t i = 12; i < 16; ++i) {
      if (mapped[(SSBO_OFFSET / 4) + i] != SSBO_STORE_WORD(i - 12))
         die("exact-end SSBO store was discarded");
   }

   for (uint32_t i = 16; i < 32; ++i) {
      if (mapped[(SSBO_OFFSET / 4) + i] != SSBO_POISON_WORD(i))
         die("OOB SSBO store changed mapped poison");
   }

   for (uint32_t i = OUTPUT_WORDS;
        i < OUTPUT_WORDS + OUTPUT_GUARD_WORDS; ++i) {
      if (mapped[(OUTPUT_OFFSET / 4) + i] != OUTPUT_SENTINEL(i))
         die("output guard changed");
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
   if (!available_features.features.robustBufferAccess ||
       !available_robustness2.robustBufferAccess2)
      die("robustBufferAccess2 is not advertised");

   VkPhysicalDeviceRobustness2PropertiesEXT robustness2_properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT,
   };
   VkPhysicalDeviceProperties2 properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &robustness2_properties,
   };
   vkGetPhysicalDeviceProperties2(physical, &properties2);
   const VkDeviceSize ubo_alignment =
      robustness2_properties.robustUniformBufferAccessSizeAlignment;
   const VkDeviceSize ssbo_alignment =
      robustness2_properties.robustStorageBufferAccessSizeAlignment;
   if (!ubo_alignment || ubo_alignment > UBO_RANGE ||
       (UBO_RANGE % ubo_alignment) != 0)
      die("oracle UBO range is not aligned to the advertised robustness2 limit");
   if (!ssbo_alignment || ssbo_alignment > SSBO_RANGE ||
       (SSBO_RANGE % ssbo_alignment) != 0)
      die("oracle SSBO range is not aligned to the advertised robustness2 limit");

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
      if (queue_families[i].queueCount) {
         queue_family = i;
         break;
      }
   }
   if (queue_family == UINT32_MAX)
      die("device has no usable queue family");
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
      .robustBufferAccess2 = VK_TRUE,
   };
   const VkPhysicalDeviceFeatures2 enabled_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &enabled_robustness2,
      .features = {.robustBufferAccess = VK_TRUE},
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

   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = BUFFER_SIZE,
      .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer buffer;
   check(vkCreateBuffer(device, &buffer_info, NULL, &buffer),
         "vkCreateBuffer");

   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, buffer, &requirements);
   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
   const uint32_t memory_type = find_memory_type(
      &memory_properties, requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = memory_type,
   };
   VkDeviceMemory memory;
   check(vkAllocateMemory(device, &memory_info, NULL, &memory),
         "vkAllocateMemory");
   check(vkBindBufferMemory(device, buffer, memory, 0),
         "vkBindBufferMemory");

   uint32_t *mapped;
   check(vkMapMemory(device, memory, 0, BUFFER_SIZE, 0, (void **)&mapped),
         "vkMapMemory");
   for (uint32_t i = 0; i < 16; ++i)
      mapped[i] = INPUT_WORD(i);
   for (uint32_t i = 16; i < 32; ++i)
      mapped[i] = POISON_WORD(i);
   for (uint32_t i = 0; i < 16; ++i)
      mapped[(SSBO_OFFSET / 4) + i] = SSBO_INPUT_WORD(i);
   for (uint32_t i = 16; i < 32; ++i)
      mapped[(SSBO_OFFSET / 4) + i] = SSBO_POISON_WORD(i);
   for (uint32_t i = 0; i < OUTPUT_WORDS + OUTPUT_GUARD_WORDS; ++i)
      mapped[(OUTPUT_OFFSET / 4) + i] = OUTPUT_SENTINEL(i);

   const VkDescriptorSetLayoutBinding bindings[] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 2,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
   };
   const VkDescriptorSetLayoutCreateInfo set_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 3,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout set_layout;
   check(vkCreateDescriptorSetLayout(device, &set_layout_info, NULL,
                                     &set_layout),
         "vkCreateDescriptorSetLayout");

   const VkDescriptorPoolSize pool_sizes[] = {
      {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1},
      {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2},
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

   const VkDescriptorBufferInfo descriptor_buffers[] = {
      {.buffer = buffer, .offset = 0, .range = UBO_RANGE},
      {.buffer = buffer, .offset = SSBO_OFFSET, .range = SSBO_RANGE},
      {.buffer = buffer, .offset = OUTPUT_OFFSET,
       .range = OUTPUT_WORDS * sizeof(uint32_t)},
   };
   const VkWriteDescriptorSet descriptor_writes[] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pBufferInfo = &descriptor_buffers[0],
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &descriptor_buffers[1],
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &descriptor_buffers[2],
      },
   };
   vkUpdateDescriptorSets(device, 3, descriptor_writes, 0, NULL);

   const VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_robust_buffer2_spv),
      .pCode = g17p_vulkan_robust_buffer2_spv,
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
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
   vkCmdDispatch(command_buffer, 1, 1, 1);
   const VkBufferMemoryBarrier host_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = buffer,
      .offset = 0,
      .size = BUFFER_SIZE,
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1,
                        &host_barrier, 0, NULL);
   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

   VkFence fence = VK_NULL_HANDLE;
   if (!record_only) {
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
      check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences");
      verify_result(mapped, "");
      sleep(3);
      verify_result(mapped, " after three seconds");
      printf(TEST_NAME ": PASS — UBO/SSBO exact-end data preserved, "
                       "constant/dynamic OOB reads zero, OOB stores "
                       "discarded, guards stable\n");
   } else {
      printf(TEST_NAME ": RECORD PASS — robustness2 feature/property query, "
                       "constant/dynamic UBO/SSBO boundary pipeline, "
                       "stores, dispatch and EndCB accepted on queue %u "
                       "(flags %#x)\n",
             queue_family, queue_flags);
   }

   if (fence != VK_NULL_HANDLE)
      vkDestroyFence(device, fence, NULL);
   vkDestroyCommandPool(device, command_pool, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, pipeline_layout, NULL);
   vkDestroyShaderModule(device, shader, NULL);
   vkDestroyDescriptorPool(device, descriptor_pool, NULL);
   vkDestroyDescriptorSetLayout(device, set_layout, NULL);
   vkUnmapMemory(device, memory);
   vkDestroyBuffer(device, buffer, NULL);
   vkFreeMemory(device, memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return EXIT_SUCCESS;
}
