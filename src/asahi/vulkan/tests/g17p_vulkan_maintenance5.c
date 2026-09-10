/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public no-submit VK_KHR_maintenance5 oracle for G17P. It checks usage2
 * buffer requirement overrides and virtual/live image subresource layouts.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_NAME       "g17p-vulkan-maintenance5"
#define T8140_DEVICE_ID 0x8140

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
VK_FN(vkCreateBuffer);
VK_FN(vkDestroyBuffer);
VK_FN(vkGetBufferMemoryRequirements2);
VK_FN(vkGetDeviceBufferMemoryRequirements);
VK_FN(vkCreateImage);
VK_FN(vkDestroyImage);
VK_FN(vkGetImageSubresourceLayout2KHR);
VK_FN(vkGetDeviceImageSubresourceLayoutKHR);

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

static void
check_usage2_case(VkDevice device, VkBufferUsageFlags legacy_usage,
                  VkBufferUsageFlags2KHR usage2, uint64_t expected_alignment,
                  const char *name)
{
   const VkBufferUsageFlags2CreateInfoKHR usage2_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR,
      .usage = usage2,
   };
   const VkBufferCreateInfo create = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = &usage2_info,
      .size = 65,
      .usage = legacy_usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   const VkDeviceBufferMemoryRequirements device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
      .pCreateInfo = &create,
   };
   VkMemoryRequirements2 device_requirements = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
   };
   vkGetDeviceBufferMemoryRequirements(device, &device_info,
                                        &device_requirements);

   VkBuffer buffer;
   check(vkCreateBuffer(device, &create, NULL, &buffer), "vkCreateBuffer");
   const VkBufferMemoryRequirementsInfo2 object_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
      .buffer = buffer,
   };
   VkMemoryRequirements2 object_requirements = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
   };
   vkGetBufferMemoryRequirements2(device, &object_info, &object_requirements);

   const VkMemoryRequirements *a = &device_requirements.memoryRequirements;
   const VkMemoryRequirements *b = &object_requirements.memoryRequirements;
   if (a->size != b->size || a->alignment != b->alignment ||
       a->memoryTypeBits != b->memoryTypeBits ||
       a->alignment != expected_alignment) {
      fprintf(stderr, TEST_NAME ": %s device=%llu/%llu/%#x "
                                "live=%llu/%llu/%#x expected-align=%llu\n",
              name, (unsigned long long)a->size,
              (unsigned long long)a->alignment, a->memoryTypeBits,
              (unsigned long long)b->size,
              (unsigned long long)b->alignment, b->memoryTypeBits,
              (unsigned long long)expected_alignment);
      exit(EXIT_FAILURE);
   }

   vkDestroyBuffer(device, buffer, NULL);
}

static void
check_subresource_layout(VkDevice device)
{
   const VkImageCreateInfo create = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {37, 19, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   const VkImageSubresource2KHR subresource = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2_KHR,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0,
         .arrayLayer = 0,
      },
   };
   const VkDeviceImageSubresourceInfoKHR device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_SUBRESOURCE_INFO_KHR,
      .pCreateInfo = &create,
      .pSubresource = &subresource,
   };
   VkSubresourceLayout2KHR device_layout = {
      .sType = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_KHR,
   };
   vkGetDeviceImageSubresourceLayoutKHR(device, &device_info, &device_layout);

   VkImage image;
   check(vkCreateImage(device, &create, NULL, &image), "vkCreateImage");
   VkSubresourceLayout2KHR object_layout = {
      .sType = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2_KHR,
   };
   vkGetImageSubresourceLayout2KHR(device, image, &subresource,
                                   &object_layout);

   const VkSubresourceLayout *a = &device_layout.subresourceLayout;
   const VkSubresourceLayout *b = &object_layout.subresourceLayout;
   if (a->offset != b->offset || a->size != b->size ||
       a->rowPitch != b->rowPitch || a->arrayPitch != b->arrayPitch ||
       a->depthPitch != b->depthPitch || !a->size || !a->rowPitch)
      die("virtual/live linear image subresource layouts differ");

   vkDestroyImage(device, image, NULL);
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
   if (!has_extension(physical, VK_KHR_MAINTENANCE_5_EXTENSION_NAME))
      die("VK_KHR_maintenance5 is not advertised");

   VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
   };
   VkPhysicalDeviceFeatures2 features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &maintenance5,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features2);
   if (!maintenance5.maintenance5)
      die("maintenance5 feature is not advertised");

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
   const VkPhysicalDeviceMaintenance5FeaturesKHR enabled5 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR,
      .maintenance5 = VK_TRUE,
   };
   const char *extensions[] = {VK_KHR_MAINTENANCE_5_EXTENSION_NAME};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enabled5,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkCreateBuffer);
   LOAD_DEVICE(device, vkDestroyBuffer);
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements2);
   LOAD_DEVICE(device, vkGetDeviceBufferMemoryRequirements);
   LOAD_DEVICE(device, vkCreateImage);
   LOAD_DEVICE(device, vkDestroyImage);
   LOAD_DEVICE(device, vkGetImageSubresourceLayout2KHR);
   LOAD_DEVICE(device, vkGetDeviceImageSubresourceLayoutKHR);

   check_usage2_case(device, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT_KHR, 64,
                     "usage2 adds uniform alignment");
   check_usage2_case(device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT_KHR, 16,
                     "usage2 replaces uniform alignment");
   check_subresource_layout(device);

   printf(TEST_NAME ": PASS usage2 override=64/16 linear-layout exact "
                    "(no GPU submissions)\n");
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return EXIT_SUCCESS;
}
