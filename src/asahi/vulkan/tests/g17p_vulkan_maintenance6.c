/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public no-submit VK_KHR_maintenance6 oracle for G17P. It checks all three
 * implementation properties (including the multilayer property Zink consumes)
 * and per-resource status for successful batched buffer and image memory binds.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_NAME       "g17p-vulkan-maintenance6"
#define T8140_DEVICE_ID 0x8140

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceProperties2);
VK_FN(vkGetPhysicalDeviceFeatures2);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkEnumerateDeviceExtensionProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkCreateBuffer);
VK_FN(vkDestroyBuffer);
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkCreateImage);
VK_FN(vkDestroyImage);
VK_FN(vkGetImageMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkFreeMemory);
VK_FN(vkBindBufferMemory2);
VK_FN(vkBindImageMemory2);

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

static uint64_t
align_u64(uint64_t value, uint64_t alignment)
{
   return (value + alignment - 1) & ~(alignment - 1);
}

static uint32_t
choose_memory_type(VkPhysicalDevice physical, uint32_t memory_type_bits)
{
   VkPhysicalDeviceMemoryProperties properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &properties);
   for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      if (memory_type_bits & (1u << i))
         return i;
   }
   die("resources have no common memory type");
   return UINT32_MAX;
}

static VkDeviceMemory
allocate_memory(VkPhysicalDevice physical, VkDevice device, uint64_t size,
                uint32_t memory_type_bits)
{
   const VkMemoryAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = size,
      .memoryTypeIndex = choose_memory_type(physical, memory_type_bits),
   };
   VkDeviceMemory memory;
   check(vkAllocateMemory(device, &info, NULL, &memory), "vkAllocateMemory");
   return memory;
}

static void
check_buffer_bind_status(VkPhysicalDevice physical, VkDevice device)
{
   const VkBufferCreateInfo create_infos[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = 65,
         .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      },
      {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = 4097,
         .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      },
   };
   VkBuffer buffers[2];
   VkMemoryRequirements requirements[2];
   uint32_t memory_type_bits = UINT32_MAX;

   for (uint32_t i = 0; i < 2; ++i) {
      check(vkCreateBuffer(device, &create_infos[i], NULL, &buffers[i]),
            "vkCreateBuffer");
      vkGetBufferMemoryRequirements(device, buffers[i], &requirements[i]);
      memory_type_bits &= requirements[i].memoryTypeBits;
   }

   const uint64_t second_offset =
      align_u64(requirements[0].size, requirements[1].alignment);
   VkDeviceMemory memory =
      allocate_memory(physical, device, second_offset + requirements[1].size,
                      memory_type_bits);
   VkResult results[2] = {VK_NOT_READY, VK_TIMEOUT};
   const VkBindMemoryStatusKHR statuses[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_BIND_MEMORY_STATUS_KHR,
         .pResult = &results[0],
      },
      {
         .sType = VK_STRUCTURE_TYPE_BIND_MEMORY_STATUS_KHR,
         .pResult = &results[1],
      },
   };
   const VkBindBufferMemoryInfo bind_infos[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
         .pNext = &statuses[0],
         .buffer = buffers[0],
         .memory = memory,
         .memoryOffset = 0,
      },
      {
         .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
         .pNext = &statuses[1],
         .buffer = buffers[1],
         .memory = memory,
         .memoryOffset = second_offset,
      },
   };

   check(vkBindBufferMemory2(device, 2, bind_infos),
         "vkBindBufferMemory2(batch)");
   if (results[0] != VK_SUCCESS || results[1] != VK_SUCCESS)
      die("batched buffer bind did not populate both success statuses");

   for (uint32_t i = 0; i < 2; ++i)
      vkDestroyBuffer(device, buffers[i], NULL);
   vkFreeMemory(device, memory, NULL);
}

static void
check_image_bind_status(VkPhysicalDevice physical, VkDevice device)
{
   const VkImageCreateInfo create_infos[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D,
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .extent = {37, 19, 1},
         .mipLevels = 1,
         .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      },
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D,
         .format = VK_FORMAT_B8G8R8A8_UNORM,
         .extent = {23, 11, 1},
         .mipLevels = 1,
         .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      },
   };
   VkImage images[2];
   VkMemoryRequirements requirements[2];
   uint32_t memory_type_bits = UINT32_MAX;

   for (uint32_t i = 0; i < 2; ++i) {
      check(vkCreateImage(device, &create_infos[i], NULL, &images[i]),
            "vkCreateImage");
      vkGetImageMemoryRequirements(device, images[i], &requirements[i]);
      memory_type_bits &= requirements[i].memoryTypeBits;
   }

   const uint64_t second_offset =
      align_u64(requirements[0].size, requirements[1].alignment);
   VkDeviceMemory memory =
      allocate_memory(physical, device, second_offset + requirements[1].size,
                      memory_type_bits);
   VkResult results[2] = {VK_EVENT_SET, VK_INCOMPLETE};
   const VkBindMemoryStatusKHR statuses[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_BIND_MEMORY_STATUS_KHR,
         .pResult = &results[0],
      },
      {
         .sType = VK_STRUCTURE_TYPE_BIND_MEMORY_STATUS_KHR,
         .pResult = &results[1],
      },
   };
   const VkBindImageMemoryInfo bind_infos[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
         .pNext = &statuses[0],
         .image = images[0],
         .memory = memory,
         .memoryOffset = 0,
      },
      {
         .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
         .pNext = &statuses[1],
         .image = images[1],
         .memory = memory,
         .memoryOffset = second_offset,
      },
   };

   check(vkBindImageMemory2(device, 2, bind_infos),
         "vkBindImageMemory2(batch)");
   if (results[0] != VK_SUCCESS || results[1] != VK_SUCCESS)
      die("batched image bind did not populate both success statuses");

   for (uint32_t i = 0; i < 2; ++i)
      vkDestroyImage(device, images[i], NULL);
   vkFreeMemory(device, memory, NULL);
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
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties2);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures2);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
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
   if (!has_extension(physical, VK_KHR_MAINTENANCE_6_EXTENSION_NAME))
      die("VK_KHR_maintenance6 is not advertised");

   VkPhysicalDeviceMaintenance6FeaturesKHR maintenance6 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES_KHR,
   };
   VkPhysicalDeviceFeatures2 features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &maintenance6,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features2);
   if (!maintenance6.maintenance6)
      die("maintenance6 feature is not advertised");

   VkPhysicalDeviceMaintenance6PropertiesKHR maintenance6_properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_PROPERTIES_KHR,
   };
   VkPhysicalDeviceProperties2 properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &maintenance6_properties,
   };
   vkGetPhysicalDeviceProperties2(physical, &properties2);
   if (maintenance6_properties.blockTexelViewCompatibleMultipleLayers ||
       maintenance6_properties.maxCombinedImageSamplerDescriptorCount != 3 ||
       maintenance6_properties.fragmentShadingRateClampCombinerInputs)
      die("unexpected G17P maintenance6 properties");

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
   const VkPhysicalDeviceMaintenance6FeaturesKHR enabled6 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES_KHR,
      .maintenance6 = VK_TRUE,
   };
   const char *extensions[] = {VK_KHR_MAINTENANCE_6_EXTENSION_NAME};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enabled6,
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
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkCreateImage);
   LOAD_DEVICE(device, vkDestroyImage);
   LOAD_DEVICE(device, vkGetImageMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkFreeMemory);
   LOAD_DEVICE(device, vkBindBufferMemory2);
   LOAD_DEVICE(device, vkBindImageMemory2);

   check_buffer_bind_status(physical, device);
   check_image_bind_status(physical, device);

   printf(TEST_NAME ": PASS props=0/3/0 buffer-status=2 image-status=2 "
                    "(no GPU submissions)\n");
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return EXIT_SUCCESS;
}
