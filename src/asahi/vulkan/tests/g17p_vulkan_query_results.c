/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public no-submit query-result oracle for G17P. It checks host-reset query
 * pools through the Vulkan ABI, including partial 32/64-bit results,
 * availability placement, nontrivial stride, and guarded output bounds.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_NAME       "g17p-vulkan-query-results"
#define T8140_DEVICE_ID 0x8140
#define QUERY_COUNT     3
#define GUARD_BYTE      0xa5

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceFeatures2);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkCreateQueryPool);
VK_FN(vkDestroyQueryPool);
VK_FN(vkResetQueryPool);
VK_FN(vkGetQueryPoolResults);

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

static void
store_integer(uint8_t *dst, uint32_t width, uint64_t value)
{
   if (width == sizeof(uint64_t)) {
      memcpy(dst, &value, sizeof(value));
   } else {
      const uint32_t value32 = value;
      memcpy(dst, &value32, sizeof(value32));
   }
}

static void
verify_partial(VkDevice device, VkQueryPool pool, uint32_t reports,
               bool result64, const char *name)
{
   const uint32_t width = result64 ? sizeof(uint64_t) : sizeof(uint32_t);
   const uint32_t payload = (reports + 1) * width;
   const uint32_t stride = payload + (2 * width);
   const size_t data_size = ((QUERY_COUNT - 1) * stride) + payload;
   union {
      uint64_t aligned;
      uint8_t bytes[256];
   } actual, expected;

   if (sizeof(actual.bytes) < data_size + 16)
      die("internal result buffer is too small");

   memset(actual.bytes, GUARD_BYTE, sizeof(actual.bytes));
   memset(expected.bytes, GUARD_BYTE, sizeof(expected.bytes));
   uint8_t *const actual_data = actual.bytes + 8;
   uint8_t *const expected_data = expected.bytes + 8;

   VkQueryResultFlags flags = VK_QUERY_RESULT_PARTIAL_BIT |
                              VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;
   if (result64)
      flags |= VK_QUERY_RESULT_64_BIT;

   const VkResult result = vkGetQueryPoolResults(
      device, pool, 0, QUERY_COUNT, data_size, actual_data, stride, flags);
   if (result != VK_SUCCESS) {
      fprintf(stderr, TEST_NAME ": partial %s/%u returned %d\n", name,
              result64 ? 64 : 32, result);
      exit(EXIT_FAILURE);
   }

   for (uint32_t query = 0; query < QUERY_COUNT; ++query) {
      uint8_t *record = expected_data + (query * stride);
      for (uint32_t report = 0; report < reports; ++report)
         store_integer(record + (report * width), width, 0);
      store_integer(record + (reports * width), width, 0);
   }

   if (memcmp(actual.bytes, expected.bytes, sizeof(actual.bytes)))
      die("partial query values, availability, stride, or guards differ");
}

static void
verify_unavailable(VkDevice device, VkQueryPool pool, uint32_t reports,
                   bool result64, const char *name)
{
   const uint32_t width = result64 ? sizeof(uint64_t) : sizeof(uint32_t);
   const uint32_t payload = (reports + 1) * width;
   const uint32_t stride = payload + (2 * width);
   const size_t data_size = ((QUERY_COUNT - 1) * stride) + payload;
   union {
      uint64_t aligned;
      uint8_t bytes[256];
   } actual, expected;

   memset(actual.bytes, GUARD_BYTE, sizeof(actual.bytes));
   memset(expected.bytes, GUARD_BYTE, sizeof(expected.bytes));
   uint8_t *const actual_data = actual.bytes + 8;
   uint8_t *const expected_data = expected.bytes + 8;
   VkQueryResultFlags flags = VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;
   if (result64)
      flags |= VK_QUERY_RESULT_64_BIT;

   const VkResult result = vkGetQueryPoolResults(
      device, pool, 0, QUERY_COUNT, data_size, actual_data, stride, flags);
   if (result != VK_NOT_READY) {
      fprintf(stderr, TEST_NAME ": unavailable %s/%u returned %d\n", name,
              result64 ? 64 : 32, result);
      exit(EXIT_FAILURE);
   }

   for (uint32_t query = 0; query < QUERY_COUNT; ++query) {
      uint8_t *record = expected_data + (query * stride);
      store_integer(record + (reports * width), width, 0);
   }

   if (memcmp(actual.bytes, expected.bytes, sizeof(actual.bytes)))
      die("unavailable query result, availability, stride, or guards differ");
}

static void
run_pool(VkDevice device, VkQueryType type,
         VkQueryPipelineStatisticFlags statistics, uint32_t reports,
         const char *name)
{
   const VkQueryPoolCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = type,
      .queryCount = QUERY_COUNT,
      .pipelineStatistics = statistics,
   };
   VkQueryPool pool;
   check(vkCreateQueryPool(device, &info, NULL, &pool),
         "vkCreateQueryPool");

   /* A reset initializes all queries and makes their zeroed reports
    * unavailable. Repeat a strict subrange reset between width variants. */
   vkResetQueryPool(device, pool, 0, QUERY_COUNT);
   verify_unavailable(device, pool, reports, false, name);
   verify_partial(device, pool, reports, false, name);
   vkResetQueryPool(device, pool, 1, QUERY_COUNT - 1);
   verify_unavailable(device, pool, reports, true, name);
   verify_partial(device, pool, reports, true, name);

   vkDestroyQueryPool(device, pool, NULL);
   printf(TEST_NAME ": %s 32/64-bit partial+availability PASS\n", name);
}

int
main(void)
{
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

   VkPhysicalDeviceVulkan12Features features12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &features12,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features2);
   if (!features12.hostQueryReset)
      die("hostQueryReset is not advertised");
   if (!features2.features.pipelineStatisticsQuery)
      die("pipelineStatisticsQuery is not advertised");

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
   const VkPhysicalDeviceVulkan12Features enabled12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .hostQueryReset = VK_TRUE,
   };
   const VkPhysicalDeviceFeatures enabled = {
      .pipelineStatisticsQuery = VK_TRUE,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enabled12,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .pEnabledFeatures = &enabled,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkCreateQueryPool);
   LOAD_DEVICE(device, vkDestroyQueryPool);
   LOAD_DEVICE(device, vkResetQueryPool);
   LOAD_DEVICE(device, vkGetQueryPoolResults);

   run_pool(device, VK_QUERY_TYPE_OCCLUSION, 0, 1, "occlusion");
   run_pool(device, VK_QUERY_TYPE_PIPELINE_STATISTICS,
            VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
               VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT,
            2, "pipeline-statistics(2)");

   printf(TEST_NAME ": PASS (no GPU submissions)\n");
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return EXIT_SUCCESS;
}
