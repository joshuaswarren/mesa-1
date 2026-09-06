/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Loader-level T8140/G17P enumeration probe. It creates only a Vulkan
 * instance and reads physical-device properties. It does not create a logical
 * device, queue, command buffer, or any GPU work.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define T8140_DEVICE_ID 0x8140

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceFeatures);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);

static void
die(const char *what)
{
   fprintf(stderr, "asahi-vulkan-g17p-enumerate: %s\n", what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "asahi-vulkan-g17p-enumerate: %s returned %d\n", what,
              result);
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

static bool
has_bounded_g17p_queue(VkPhysicalDevice physical)
{
   uint32_t count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
   if (!count)
      return false;

   VkQueueFamilyProperties *queues = calloc(count, sizeof(*queues));
   if (!queues)
      die("queue-family allocation failed");

   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues);
   bool found = false;
   for (uint32_t i = 0; i < count; ++i) {
      const VkQueueFlags required = VK_QUEUE_GRAPHICS_BIT |
                                    VK_QUEUE_TRANSFER_BIT;
      const VkQueueFlags deferred = VK_QUEUE_COMPUTE_BIT |
                                    VK_QUEUE_SPARSE_BINDING_BIT;
      if ((queues[i].queueFlags & required) == required &&
          !(queues[i].queueFlags & deferred) && queues[i].queueCount == 1) {
         if (queues[i].timestampValidBits != 64)
            continue;
         found = true;
         break;
      }
   }

   free(queues);
   return found;
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

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "asahi-vulkan-g17p-enumerate",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "none",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_0,
   };
   const VkInstanceCreateInfo create = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance = VK_NULL_HANDLE;
   check(vkCreateInstance(&create, NULL, &instance), "vkCreateInstance");

   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);

   uint32_t count = 0;
   check(vkEnumeratePhysicalDevices(instance, &count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   if (!count)
      die("no Vulkan physical devices found");

   VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
   if (!devices)
      die("physical-device allocation failed");
   check(vkEnumeratePhysicalDevices(instance, &count, devices),
         "vkEnumeratePhysicalDevices(list)");

   bool found = false;
   for (uint32_t i = 0; i < count; ++i) {
      VkPhysicalDeviceProperties properties;
      VkPhysicalDeviceFeatures features;
      vkGetPhysicalDeviceProperties(devices[i], &properties);
      if (properties.deviceID != T8140_DEVICE_ID)
         continue;
      vkGetPhysicalDeviceFeatures(devices[i], &features);

      if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
          !strstr(properties.deviceName, "A18 Pro") ||
          !strstr(properties.deviceName, "G17P") ||
          properties.limits.framebufferColorSampleCounts !=
             VK_SAMPLE_COUNT_1_BIT ||
          !properties.limits.timestampComputeAndGraphics ||
          properties.limits.timestampPeriod != 1.0f ||
          features.geometryShader || features.tessellationShader ||
          !features.samplerAnisotropy || features.sparseBinding ||
          !features.occlusionQueryPrecise || features.pipelineStatisticsQuery ||
          !has_bounded_g17p_queue(devices[i])) {
         die("T8140/G17P properties do not match the bounded profile");
      }

      printf("asahi-vulkan-g17p-enumerate: PASS — %s (device %#x)\n",
             properties.deviceName, properties.deviceID);
      found = true;
      break;
   }

   free(devices);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);

   if (!found)
      die("Honeykrisp did not enumerate T8140/G17P");
   return EXIT_SUCCESS;
}
