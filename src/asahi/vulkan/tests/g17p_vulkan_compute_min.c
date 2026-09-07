/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Private G17P Honeykrisp compute smoke.  This deliberately uses the first
 * usable queue while public COMPUTE advertisement remains disabled.  It is a
 * bring-up probe, not an application example.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef G17P_NULL_DESCRIPTOR
#include "g17p_vulkan_null_descriptor_spv.h"
#define COMPUTE_SPV g17p_vulkan_null_descriptor_spv
#define TEST_NAME "g17p-vulkan-null-descriptor"
#else
#include "g17p_vulkan_compute_min_spv.h"
#define COMPUTE_SPV g17p_vulkan_compute_min_spv
#define TEST_NAME "g17p-vulkan-compute-min"
#endif

#define T8140_DEVICE_ID 0x8140
#define ELEMENT_COUNT   32
#ifdef G17P_NULL_DESCRIPTOR
#define GUARD_COUNT 32
#else
#define GUARD_COUNT 0
#endif

/* Distinct per element and distinguishable from both 0 and any plausible
 * shader result. */
#define SENTINEL(i) (UINT32_C(0xA5A50000) | (uint32_t)(i))

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceFeatures2);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkGetDeviceQueue);
VK_FN(vkCreateBuffer);
VK_FN(vkDestroyBuffer);
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkFreeMemory);
VK_FN(vkBindBufferMemory);
VK_FN(vkMapMemory);
VK_FN(vkUnmapMemory);
VK_FN(vkCreateDescriptorSetLayout);
VK_FN(vkDestroyDescriptorSetLayout);
VK_FN(vkCreateDescriptorPool);
VK_FN(vkDestroyDescriptorPool);
VK_FN(vkAllocateDescriptorSets);
VK_FN(vkUpdateDescriptorSets);
VK_FN(vkCreateShaderModule);
VK_FN(vkDestroyShaderModule);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkDestroyPipelineLayout);
VK_FN(vkCreateComputePipelines);
VK_FN(vkDestroyPipeline);
VK_FN(vkCreateCommandPool);
VK_FN(vkDestroyCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindDescriptorSets);
VK_FN(vkCmdDispatch);
VK_FN(vkCmdPipelineBarrier);
VK_FN(vkEndCommandBuffer);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);

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

   die("host-visible coherent memory is unavailable");
   return UINT32_MAX;
}

int
main(int argc, char **argv)
{
   const bool record_only = argc == 2 && !strcmp(argv[1], "--record-only");
   if (argc > 2 || (argc == 2 && !record_only))
      die("usage: " TEST_NAME " [--record-only]");

   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());

   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("libvulkan.so.1 has no vkGetInstanceProcAddr");
   LOAD_GLOBAL(vkCreateInstance);

   const VkApplicationInfo application_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "g17p-vulkan-compute",
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

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures2);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
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
   VkPhysicalDeviceProperties properties;
   for (uint32_t i = 0; i < physical_count; ++i) {
      vkGetPhysicalDeviceProperties(physical_devices[i], &properties);
      if (properties.deviceID == T8140_DEVICE_ID) {
         physical = physical_devices[i];
         break;
      }
   }
   free(physical_devices);
   if (physical == VK_NULL_HANDLE)
      die("T8140/G17P Vulkan device was not found");

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
      if (queue_families[i].queueCount != 0) {
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
#ifdef G17P_NULL_DESCRIPTOR
   VkPhysicalDeviceRobustness2FeaturesEXT available_robustness2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 available_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &available_robustness2,
   };
   vkGetPhysicalDeviceFeatures2(physical, &available_features);
   if (!available_robustness2.nullDescriptor)
      die("nullDescriptor is not advertised");

   const VkPhysicalDeviceRobustness2FeaturesEXT enabled_robustness2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
      .nullDescriptor = VK_TRUE,
   };
   const char *const enabled_extensions[] = {
      VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
   };
#endif
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
#ifdef G17P_NULL_DESCRIPTOR
      .pNext = &enabled_robustness2,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = enabled_extensions,
#endif
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCreateBuffer);
   LOAD_DEVICE(device, vkDestroyBuffer);
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkFreeMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkUnmapMemory);
   LOAD_DEVICE(device, vkCreateDescriptorSetLayout);
   LOAD_DEVICE(device, vkDestroyDescriptorSetLayout);
   LOAD_DEVICE(device, vkCreateDescriptorPool);
   LOAD_DEVICE(device, vkDestroyDescriptorPool);
   LOAD_DEVICE(device, vkAllocateDescriptorSets);
   LOAD_DEVICE(device, vkUpdateDescriptorSets);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkDestroyShaderModule);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkDestroyPipelineLayout);
   LOAD_DEVICE(device, vkCreateComputePipelines);
   LOAD_DEVICE(device, vkDestroyPipeline);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkDestroyCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdBindDescriptorSets);
   LOAD_DEVICE(device, vkCmdDispatch);
   LOAD_DEVICE(device, vkCmdPipelineBarrier);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkDestroyFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);

   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);

   const VkDeviceSize payload_size = ELEMENT_COUNT * sizeof(uint32_t);
   const VkDeviceSize buffer_size =
      (ELEMENT_COUNT + GUARD_COUNT) * sizeof(uint32_t);
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = buffer_size,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
   check(vkMapMemory(device, memory, 0, buffer_size, 0, (void **)&mapped),
         "vkMapMemory");
   /* Prefill with a per-element sentinel instead of zero. A zero prefill
    * cannot distinguish "the dispatch never ran" from "the store ran and wrote
    * zero", and that ambiguity is exactly what stalled the last round: every
    * element read back as 0, which is also what an untouched buffer looks
    * like. With the sentinel the readback classifies itself. */
   for (uint32_t i = 0; i < ELEMENT_COUNT + GUARD_COUNT; ++i)
      mapped[i] = SENTINEL(i);

   const VkDescriptorSetLayoutBinding bindings[] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
#ifdef G17P_NULL_DESCRIPTOR
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 2,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
#endif
   };
   const VkDescriptorSetLayoutCreateInfo set_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
      .pBindings = bindings,
   };
   VkDescriptorSetLayout set_layout;
   check(vkCreateDescriptorSetLayout(device, &set_layout_info, NULL,
                                     &set_layout),
         "vkCreateDescriptorSetLayout");

   const VkDescriptorPoolSize pool_sizes[] = {
      {
         .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
#ifdef G17P_NULL_DESCRIPTOR
         .descriptorCount = 2,
#else
         .descriptorCount = 1,
#endif
      },
#ifdef G17P_NULL_DESCRIPTOR
      {
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
      },
#endif
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
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
#ifdef G17P_NULL_DESCRIPTOR
      {
         .buffer = VK_NULL_HANDLE,
         .offset = 0,
         .range = VK_WHOLE_SIZE,
      },
#endif
      {
         .buffer = buffer,
         .offset = 0,
         .range = payload_size,
      },
#ifdef G17P_NULL_DESCRIPTOR
      {
         .buffer = VK_NULL_HANDLE,
         .offset = 0,
         .range = VK_WHOLE_SIZE,
      },
#endif
   };
   VkWriteDescriptorSet descriptor_writes[
      sizeof(descriptor_buffers) / sizeof(descriptor_buffers[0])];
   for (uint32_t i = 0;
        i < sizeof(descriptor_buffers) / sizeof(descriptor_buffers[0]); ++i) {
      descriptor_writes[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = i,
         .descriptorCount = 1,
#ifdef G17P_NULL_DESCRIPTOR
         .descriptorType = i == 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                  : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
#else
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
#endif
         .pBufferInfo = &descriptor_buffers[i],
      };
   }
   vkUpdateDescriptorSets(
      device, sizeof(descriptor_writes) / sizeof(descriptor_writes[0]),
      descriptor_writes, 0, NULL);

   const VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(COMPUTE_SPV),
      .pCode = COMPUTE_SPV,
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
      .stage =
         {
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
      .size = buffer_size,
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1,
                        &host_barrier, 0, NULL);
   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

   if (record_only) {
      printf(TEST_NAME ": RECORD PASS — null SSBO load/store and UBO load "
                       "descriptor update, pipeline, dispatch and EndCB "
                       "accepted on queue %u (flags %#x)\n",
             queue_family, queue_flags);
      goto cleanup;
   }

   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence fence;
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");
   const VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
   };
   check(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
   check(vkWaitForFences(device, 1, &fence, true, UINT64_MAX),
         "vkWaitForFences");

   unsigned untouched = 0, zeroed = 0, wrong = 0, first_bad = UINT32_MAX;
   for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
#ifdef G17P_NULL_DESCRIPTOR
      const uint32_t expected = UINT32_C(0x6d000000) + i;
#else
      const uint32_t expected = UINT32_C(0x12340000) + i;
#endif
      if (mapped[i] == expected)
         continue;
      if (first_bad == UINT32_MAX)
         first_bad = i;
      if (mapped[i] == SENTINEL(i))
         untouched++;
      else if (mapped[i] == 0)
         zeroed++;
      else
         wrong++;
   }

   if (untouched || zeroed || wrong) {
      fprintf(stderr,
              TEST_NAME ": FAIL untouched=%u zeroed=%u wrong=%u of %u\n"
              TEST_NAME ":   value[%u] is %#x, expected %#x "
              "(sentinel would be %#x)\n",
              untouched, zeroed, wrong, ELEMENT_COUNT, first_bad,
              mapped[first_bad],
#ifdef G17P_NULL_DESCRIPTOR
              UINT32_C(0x6d000000) + first_bad,
#else
              UINT32_C(0x12340000) + first_bad,
#endif
              SENTINEL(first_bad));
      /* The classification is the whole point, so spell it out. */
      if (untouched == ELEMENT_COUNT)
         fprintf(stderr,
                 TEST_NAME ":   -> the store never executed: every "
                 "element still holds its prefill sentinel. The dispatch did "
                 "not run (or ran and was discarded); this is NOT a shader "
                 "arithmetic bug.\n");
      else if (zeroed == ELEMENT_COUNT)
         fprintf(stderr,
                 TEST_NAME ":   -> the buffer was zeroed by something "
                 "other than this shader; the store wrote 0 or the memory was "
                 "clobbered.\n");
      else if (wrong)
         fprintf(stderr,
                 TEST_NAME ":   -> the store DID execute; the value "
                 "is wrong, so the arithmetic or the constant-data load is at "
                 "fault.\n");
      /* "Wrote nothing" and "wrote somewhere else" have completely different
       * causes, and reporting only element 0 cannot tell them apart. Dump the
       * whole buffer so a store landing at a wrong offset is visible. */
      fprintf(stderr, TEST_NAME ":   buffer:\n");
      for (uint32_t i = 0; i < ELEMENT_COUNT; i += 8) {
         fprintf(stderr, TEST_NAME ":     [%2u] ", i);
         for (uint32_t j = 0; j < 8 && i + j < ELEMENT_COUNT; ++j)
            fprintf(stderr, "%08x ", mapped[i + j]);
         fprintf(stderr, "\n");
      }
      return EXIT_FAILURE;
   }

#ifdef G17P_NULL_DESCRIPTOR
   for (uint32_t i = ELEMENT_COUNT; i < ELEMENT_COUNT + GUARD_COUNT; ++i) {
      if (mapped[i] != SENTINEL(i))
         die("output guard changed");
   }

   sleep(3);
   for (uint32_t i = 0; i < ELEMENT_COUNT; ++i) {
      if (mapped[i] != UINT32_C(0x6d000000) + i)
         die("null-load result changed after three seconds");
   }
   for (uint32_t i = ELEMENT_COUNT; i < ELEMENT_COUNT + GUARD_COUNT; ++i) {
      if (mapped[i] != SENTINEL(i))
         die("output guard changed after three seconds");
   }

   printf(TEST_NAME ": PASS — 32 null SSBO/UBO loads returned zero, null "
                    "SSBO stores were discarded, exact output and 32 guards "
                    "stable for three seconds\n");
#else
   printf("g17p-vulkan-compute-min: PASS — 32 buffer writes on queue family %u "
          "(flags %#x, private G17P compute route)\n",
          queue_family, queue_flags);
#endif

   vkDestroyFence(device, fence, NULL);
cleanup:
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
   dlclose(loader);
   return EXIT_SUCCESS;
}
