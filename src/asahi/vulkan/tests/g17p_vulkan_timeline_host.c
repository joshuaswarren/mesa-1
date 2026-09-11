/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public no-submit timeline-semaphore oracle for G17P. It checks core/KHR
 * host entrypoints, WAIT_ANY/WAIT_ALL behavior, monotonic counter visibility,
 * and repeated timeline/binary semaphore object lifetimes.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_NAME       "g17p-vulkan-timeline-host"
#define T8140_DEVICE_ID 0x8140
#define ITERATIONS      64

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
VK_FN(vkCreateSemaphore);
VK_FN(vkDestroySemaphore);
VK_FN(vkGetSemaphoreCounterValue);
VK_FN(vkGetSemaphoreCounterValueKHR);
VK_FN(vkSignalSemaphore);
VK_FN(vkSignalSemaphoreKHR);
VK_FN(vkWaitSemaphores);
VK_FN(vkWaitSemaphoresKHR);

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

static void
expect(VkResult result, VkResult expected, const char *what)
{
   if (result != expected) {
      fprintf(stderr, TEST_NAME ": %s returned %d, expected %d\n", what,
              result, expected);
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

static VkSemaphore
create_semaphore(VkDevice device, VkSemaphoreType type, uint64_t initial)
{
   const VkSemaphoreTypeCreateInfo type_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = type,
      .initialValue = initial,
   };
   const VkSemaphoreCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = type == VK_SEMAPHORE_TYPE_TIMELINE ? &type_info : NULL,
   };
   VkSemaphore semaphore;
   check(vkCreateSemaphore(device, &info, NULL, &semaphore),
         type == VK_SEMAPHORE_TYPE_TIMELINE
            ? "vkCreateSemaphore(timeline)"
            : "vkCreateSemaphore(binary)");
   return semaphore;
}

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

static void
check_counter(VkDevice device, VkSemaphore semaphore, uint64_t expected,
              bool khr)
{
   uint64_t value = UINT64_MAX;
   VkResult result = khr
      ? vkGetSemaphoreCounterValueKHR(device, semaphore, &value)
      : vkGetSemaphoreCounterValue(device, semaphore, &value);
   check(result, khr ? "vkGetSemaphoreCounterValueKHR"
                     : "vkGetSemaphoreCounterValue");
   if (value != expected) {
      fprintf(stderr, TEST_NAME ": %s counter=%llu expected=%llu\n",
              khr ? "KHR" : "core", (unsigned long long)value,
              (unsigned long long)expected);
      exit(EXIT_FAILURE);
   }
}

static void
signal_counter(VkDevice device, VkSemaphore semaphore, uint64_t value,
               bool khr)
{
   const VkSemaphoreSignalInfo info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
      .semaphore = semaphore,
      .value = value,
   };
   check(khr ? vkSignalSemaphoreKHR(device, &info)
             : vkSignalSemaphore(device, &info),
         khr ? "vkSignalSemaphoreKHR" : "vkSignalSemaphore");
}

static VkResult
wait_counters(VkDevice device, const VkSemaphore *semaphores,
              const uint64_t *values, uint32_t count,
              VkSemaphoreWaitFlags flags, bool khr)
{
   const VkSemaphoreWaitInfo info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .flags = flags,
      .semaphoreCount = count,
      .pSemaphores = semaphores,
      .pValues = values,
   };
   return khr ? vkWaitSemaphoresKHR(device, &info, 0)
              : vkWaitSemaphores(device, &info, 0);
}

static void
run_iteration(VkDevice device, uint32_t iteration)
{
   const uint64_t initial[] = {
      1 + ((uint64_t)iteration * 16),
      5 + ((uint64_t)iteration * 16),
   };
   VkSemaphore timeline[] = {
      create_semaphore(device, VK_SEMAPHORE_TYPE_TIMELINE, initial[0]),
      create_semaphore(device, VK_SEMAPHORE_TYPE_TIMELINE, initial[1]),
   };
   VkSemaphore binary =
      create_semaphore(device, VK_SEMAPHORE_TYPE_BINARY, 0);

   check_counter(device, timeline[0], initial[0], false);
   check_counter(device, timeline[1], initial[1], true);

   expect(wait_counters(device, timeline, initial, 2, 0, false), VK_SUCCESS,
          "core WAIT_ALL(initial)");

   const uint64_t any_ready[] = {initial[0] + 1, initial[1]};
   expect(wait_counters(device, timeline, any_ready, 2,
                        VK_SEMAPHORE_WAIT_ANY_BIT, true),
          VK_SUCCESS, "KHR WAIT_ANY(one ready)");

   const uint64_t future[] = {initial[0] + 1, initial[1] + 1};
   expect(wait_counters(device, timeline, future, 2,
                        VK_SEMAPHORE_WAIT_ANY_BIT, false),
          VK_TIMEOUT, "core WAIT_ANY(none ready)");
   expect(wait_counters(device, timeline, future, 2, 0, true), VK_TIMEOUT,
          "KHR WAIT_ALL(future)");

   const uint64_t signaled[] = {initial[0] + 1, initial[1] + 1};
   signal_counter(device, timeline[0], signaled[0], true);
   signal_counter(device, timeline[1], signaled[1], false);
   check_counter(device, timeline[0], signaled[0], false);
   check_counter(device, timeline[1], signaled[1], true);
   expect(wait_counters(device, timeline, signaled, 2, 0, false), VK_SUCCESS,
          "core WAIT_ALL(signaled)");

   vkDestroySemaphore(device, binary, NULL);
   vkDestroySemaphore(device, timeline[1], NULL);
   vkDestroySemaphore(device, timeline[0], NULL);
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
   if (!features12.timelineSemaphore)
      die("timelineSemaphore is not advertised");
   if (!has_extension(physical, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME))
      die("VK_KHR_timeline_semaphore is not advertised");

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
      .timelineSemaphore = VK_TRUE,
   };
   const char *extensions[] = {
      VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enabled12,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkCreateSemaphore);
   LOAD_DEVICE(device, vkDestroySemaphore);
   LOAD_DEVICE(device, vkGetSemaphoreCounterValue);
   LOAD_DEVICE(device, vkGetSemaphoreCounterValueKHR);
   LOAD_DEVICE(device, vkSignalSemaphore);
   LOAD_DEVICE(device, vkSignalSemaphoreKHR);
   LOAD_DEVICE(device, vkWaitSemaphores);
   LOAD_DEVICE(device, vkWaitSemaphoresKHR);

   const VkSemaphoreWaitInfo empty_wait = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
   };
   check(vkWaitSemaphores(device, &empty_wait, 0),
         "vkWaitSemaphores(empty)");

   for (uint32_t i = 0; i < ITERATIONS; ++i)
      run_iteration(device, i);

   printf(TEST_NAME ": PASS iterations=%u timelines=%u binaries=%u "
                    "(no GPU submissions)\n",
          ITERATIONS, ITERATIONS * 2, ITERATIONS);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return EXIT_SUCCESS;
}
