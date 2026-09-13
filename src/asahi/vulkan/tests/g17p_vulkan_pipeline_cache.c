/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public Vulkan pipeline-cache persistence oracle for G17P.  No command is
 * submitted: this isolates cache header identity, shader serialization,
 * import, merge, and compile-required behavior from GPU execution.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "g17p_vulkan_compute_noroot_spv.h"

#define TEST_NAME        "g17p-vulkan-pipeline-cache"
#define T8140_DEVICE_ID  0x8140
#define CACHE_HEADER_SIZE 32

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceFeatures2);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkEnumerateDeviceExtensionProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkCreatePipelineCache);
VK_FN(vkDestroyPipelineCache);
VK_FN(vkGetPipelineCacheData);
VK_FN(vkMergePipelineCaches);
VK_FN(vkCreateShaderModule);
VK_FN(vkDestroyShaderModule);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkDestroyPipelineLayout);
VK_FN(vkCreateComputePipelines);
VK_FN(vkDestroyPipeline);

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

static bool
has_extension(VkPhysicalDevice physical, const char *name)
{
   uint32_t count = 0;
   check(vkEnumerateDeviceExtensionProperties(physical, NULL, &count, NULL),
         "vkEnumerateDeviceExtensionProperties(count)");

   VkExtensionProperties *extensions = calloc(count, sizeof(*extensions));
   if (!extensions)
      die("out of host memory");

   check(vkEnumerateDeviceExtensionProperties(physical, NULL, &count,
                                               extensions),
         "vkEnumerateDeviceExtensionProperties(list)");
   bool found = false;
   for (uint32_t i = 0; i < count; ++i)
      found |= !strcmp(extensions[i].extensionName, name);
   free(extensions);
   return found;
}

static VkPipelineCache
create_cache(VkDevice device, const void *data, size_t size)
{
   const VkPipelineCacheCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      .initialDataSize = size,
      .pInitialData = data,
   };
   VkPipelineCache cache;
   check(vkCreatePipelineCache(device, &info, NULL, &cache),
         "vkCreatePipelineCache");
   return cache;
}

static VkResult
create_pipeline(VkDevice device, VkPipelineCache cache, VkPipelineLayout layout,
                VkShaderModule shader, VkPipelineCreateFlags flags,
                VkPipeline *pipeline)
{
   const VkPipelineShaderStageCreateInfo stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shader,
      .pName = "main",
   };
   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .flags = flags,
      .stage = stage,
      .layout = layout,
   };
   return vkCreateComputePipelines(device, cache, 1, &info, NULL, pipeline);
}

static uint32_t
read_u32(const uint8_t *data, size_t offset)
{
   uint32_t value;
   memcpy(&value, data + offset, sizeof(value));
   return value;
}

int
main(void)
{
   /* Explicit-cache behavior must not be hidden by a process-global disk
    * cache populated by an earlier test run. */
   setenv("MESA_SHADER_CACHE_DISABLE", "true", 1);

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
      .pApplicationName = TEST_NAME,
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_3,
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
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkEnumerateDeviceExtensionProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      die("out of host memory");
   check(vkEnumeratePhysicalDevices(instance, &physical_count,
                                    physical_devices),
         "vkEnumeratePhysicalDevices(list)");

   VkPhysicalDevice physical = VK_NULL_HANDLE;
   VkPhysicalDeviceProperties properties = {0};
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

   if (!has_extension(physical,
                      VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME))
      die("VK_EXT_pipeline_creation_cache_control is not advertised");

   VkPhysicalDevicePipelineCreationCacheControlFeatures cache_features = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_CREATION_CACHE_CONTROL_FEATURES,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &cache_features,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   if (!cache_features.pipelineCreationCacheControl)
      die("pipelineCreationCacheControl is not supported");

   uint32_t queue_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, NULL);
   VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
   if (!queues)
      die("out of host memory");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, queues);
   uint32_t queue_family = UINT32_MAX;
   for (uint32_t i = 0; i < queue_count; ++i) {
      if (queues[i].queueCount) {
         queue_family = i;
         break;
      }
   }
   free(queues);
   if (queue_family == UINT32_MAX)
      die("device has no usable queue family");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   cache_features.pipelineCreationCacheControl = VK_TRUE;
   const char *extensions[] = {
      VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &cache_features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkCreatePipelineCache);
   LOAD_DEVICE(device, vkDestroyPipelineCache);
   LOAD_DEVICE(device, vkGetPipelineCacheData);
   LOAD_DEVICE(device, vkMergePipelineCaches);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkDestroyShaderModule);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkDestroyPipelineLayout);
   LOAD_DEVICE(device, vkCreateComputePipelines);
   LOAD_DEVICE(device, vkDestroyPipeline);

   const VkShaderModuleCreateInfo shader_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_compute_noroot_spv),
      .pCode = g17p_vulkan_compute_noroot_spv,
   };
   VkShaderModule shader;
   check(vkCreateShaderModule(device, &shader_info, NULL, &shader),
         "vkCreateShaderModule");

   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout layout;
   check(vkCreatePipelineLayout(device, &layout_info, NULL, &layout),
         "vkCreatePipelineLayout");

   VkPipelineCache source = create_cache(device, NULL, 0);
   VkPipeline compiled = VK_NULL_HANDLE;
   check(create_pipeline(device, source, layout, shader, 0, &compiled),
         "initial vkCreateComputePipelines");

   size_t cache_size = 0;
   check(vkGetPipelineCacheData(device, source, &cache_size, NULL),
         "vkGetPipelineCacheData(size)");
   if (cache_size <= CACHE_HEADER_SIZE)
      die("compiled cache contains no serialized shader object");
   uint8_t *cache_data = malloc(cache_size);
   if (!cache_data)
      die("out of host memory");
   size_t written = cache_size;
   check(vkGetPipelineCacheData(device, source, &written, cache_data),
         "vkGetPipelineCacheData(data)");
   if (written != cache_size || read_u32(cache_data, 0) != CACHE_HEADER_SIZE ||
       read_u32(cache_data, 4) != VK_PIPELINE_CACHE_HEADER_VERSION_ONE ||
       read_u32(cache_data, 8) != properties.vendorID ||
       read_u32(cache_data, 12) != properties.deviceID ||
       memcmp(cache_data + 16, properties.pipelineCacheUUID, VK_UUID_SIZE))
      die("pipeline cache header does not match physical-device identity");

   vkDestroyPipeline(device, compiled, NULL);
   vkDestroyPipelineCache(device, source, NULL);

   VkPipelineCache imported = create_cache(device, cache_data, cache_size);
   VkPipeline imported_pipeline = VK_NULL_HANDLE;
   check(create_pipeline(
            device, imported, layout, shader,
            VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT,
            &imported_pipeline),
         "imported-cache vkCreateComputePipelines");
   vkDestroyPipeline(device, imported_pipeline, NULL);

   VkPipelineCache merged = create_cache(device, NULL, 0);
   check(vkMergePipelineCaches(device, merged, 1, &imported),
         "vkMergePipelineCaches");
   VkPipeline merged_pipeline = VK_NULL_HANDLE;
   check(create_pipeline(
            device, merged, layout, shader,
            VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT,
            &merged_pipeline),
         "merged-cache vkCreateComputePipelines");
   vkDestroyPipeline(device, merged_pipeline, NULL);

   uint8_t *wrong_uuid = malloc(cache_size);
   if (!wrong_uuid)
      die("out of host memory");
   memcpy(wrong_uuid, cache_data, cache_size);
   wrong_uuid[16] ^= 0x80;
   VkPipelineCache invalid = create_cache(device, wrong_uuid, cache_size);
   VkPipeline rejected_pipeline = VK_NULL_HANDLE;
   VkResult invalid_result = create_pipeline(
      device, invalid, layout, shader,
      VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT,
      &rejected_pipeline);
   if (invalid_result != VK_PIPELINE_COMPILE_REQUIRED ||
       rejected_pipeline != VK_NULL_HANDLE)
      die("mismatched cache UUID was not ignored");

   printf(TEST_NAME ": PASS size=%zu uuid=", cache_size);
   for (unsigned i = 0; i < VK_UUID_SIZE; ++i)
      printf("%02x", properties.pipelineCacheUUID[i]);
   puts(" import-hit merge-hit wrong-uuid-miss");

   vkDestroyPipelineCache(device, invalid, NULL);
   vkDestroyPipelineCache(device, merged, NULL);
   vkDestroyPipelineCache(device, imported, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyShaderModule(device, shader, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   free(wrong_uuid);
   free(cache_data);
   dlclose(loader);
   return EXIT_SUCCESS;
}
