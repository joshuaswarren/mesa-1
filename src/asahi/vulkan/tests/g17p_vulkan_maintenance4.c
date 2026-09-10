/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public no-submit VK_KHR_maintenance4 oracle for G17P. It compares device
 * memory-requirement queries with requirements from equivalent live objects.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_NAME       "g17p-vulkan-maintenance4"
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
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkEnumerateDeviceExtensionProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkCreateBuffer);
VK_FN(vkDestroyBuffer);
VK_FN(vkGetBufferMemoryRequirements2);
VK_FN(vkGetDeviceBufferMemoryRequirements);
VK_FN(vkGetDeviceBufferMemoryRequirementsKHR);
VK_FN(vkCreateImage);
VK_FN(vkDestroyImage);
VK_FN(vkGetImageMemoryRequirements2);
VK_FN(vkGetDeviceImageMemoryRequirements);
VK_FN(vkGetDeviceImageMemoryRequirementsKHR);
VK_FN(vkGetImageSparseMemoryRequirements2);
VK_FN(vkGetDeviceImageSparseMemoryRequirements);
VK_FN(vkGetDeviceImageSparseMemoryRequirementsKHR);

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
compare_requirements(const VkMemoryRequirements2 *device,
                     const VkMemoryDedicatedRequirements *device_dedicated,
                     const VkMemoryRequirements2 *object,
                     const VkMemoryDedicatedRequirements *object_dedicated,
                     const char *name)
{
   const VkMemoryRequirements *a = &device->memoryRequirements;
   const VkMemoryRequirements *b = &object->memoryRequirements;
   if (a->size != b->size || a->alignment != b->alignment ||
       a->memoryTypeBits != b->memoryTypeBits ||
       device_dedicated->prefersDedicatedAllocation !=
          object_dedicated->prefersDedicatedAllocation ||
       device_dedicated->requiresDedicatedAllocation !=
          object_dedicated->requiresDedicatedAllocation) {
      fprintf(stderr, TEST_NAME ": %s requirement mismatch\n", name);
      exit(EXIT_FAILURE);
   }
   if (!a->size || !a->alignment || !a->memoryTypeBits)
      die("memory requirements contain a zero required field");
}

static void
check_buffer_case(VkDevice device, const VkBufferCreateInfo *create,
                  bool khr, const char *name)
{
   VkMemoryDedicatedRequirements device_dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
      .prefersDedicatedAllocation = VK_TRUE,
      .requiresDedicatedAllocation = VK_TRUE,
   };
   VkMemoryRequirements2 device_requirements = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
      .pNext = &device_dedicated,
   };
   const VkDeviceBufferMemoryRequirements info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS,
      .pCreateInfo = create,
   };
   if (khr) {
      vkGetDeviceBufferMemoryRequirementsKHR(device, &info,
                                              &device_requirements);
   } else {
      vkGetDeviceBufferMemoryRequirements(device, &info,
                                           &device_requirements);
   }

   VkBuffer buffer;
   check(vkCreateBuffer(device, create, NULL, &buffer), "vkCreateBuffer");
   VkMemoryDedicatedRequirements object_dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
      .prefersDedicatedAllocation = VK_TRUE,
      .requiresDedicatedAllocation = VK_TRUE,
   };
   VkMemoryRequirements2 object_requirements = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
      .pNext = &object_dedicated,
   };
   const VkBufferMemoryRequirementsInfo2 object_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
      .buffer = buffer,
   };
   vkGetBufferMemoryRequirements2(device, &object_info, &object_requirements);
   compare_requirements(&device_requirements, &device_dedicated,
                        &object_requirements, &object_dedicated, name);
   vkDestroyBuffer(device, buffer, NULL);
}

static void
check_image_case(VkDevice device, const VkImageCreateInfo *create,
                 bool khr, const char *name)
{
   VkMemoryDedicatedRequirements device_dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
      .prefersDedicatedAllocation = VK_TRUE,
      .requiresDedicatedAllocation = VK_TRUE,
   };
   VkMemoryRequirements2 device_requirements = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
      .pNext = &device_dedicated,
   };
   const VkDeviceImageMemoryRequirements info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS,
      .pCreateInfo = create,
      .planeAspect = VK_IMAGE_ASPECT_NONE,
   };
   if (khr) {
      vkGetDeviceImageMemoryRequirementsKHR(device, &info,
                                             &device_requirements);
   } else {
      vkGetDeviceImageMemoryRequirements(device, &info,
                                          &device_requirements);
   }

   uint32_t device_sparse_count = UINT32_MAX;
   if (khr) {
      vkGetDeviceImageSparseMemoryRequirementsKHR(
         device, &info, &device_sparse_count, NULL);
   } else {
      vkGetDeviceImageSparseMemoryRequirements(
         device, &info, &device_sparse_count, NULL);
   }
   if (device_sparse_count != 0)
      die("non-sparse device image returned sparse requirements");

   VkImage image;
   check(vkCreateImage(device, create, NULL, &image), "vkCreateImage");
   VkMemoryDedicatedRequirements object_dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
      .prefersDedicatedAllocation = VK_TRUE,
      .requiresDedicatedAllocation = VK_TRUE,
   };
   VkMemoryRequirements2 object_requirements = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
      .pNext = &object_dedicated,
   };
   const VkImageMemoryRequirementsInfo2 object_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
      .image = image,
   };
   vkGetImageMemoryRequirements2(device, &object_info, &object_requirements);

   const VkImageSparseMemoryRequirementsInfo2 sparse_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2,
      .image = image,
   };
   uint32_t object_sparse_count = UINT32_MAX;
   vkGetImageSparseMemoryRequirements2(device, &sparse_info,
                                        &object_sparse_count, NULL);
   if (object_sparse_count != device_sparse_count)
      die("device/live sparse requirement counts differ");

   compare_requirements(&device_requirements, &device_dedicated,
                        &object_requirements, &object_dedicated, name);
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
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties2);
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
   if (!has_extension(physical, VK_KHR_MAINTENANCE_4_EXTENSION_NAME))
      die("VK_KHR_maintenance4 is not advertised");

   VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
   };
   VkPhysicalDeviceFeatures2 features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &features13,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features2);
   if (!features13.maintenance4)
      die("maintenance4 feature is not advertised");

   VkPhysicalDeviceMaintenance4Properties maintenance4 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES,
   };
   VkPhysicalDeviceProperties2 properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &maintenance4,
   };
   vkGetPhysicalDeviceProperties2(physical, &properties2);
   if (maintenance4.maxBufferSize < (UINT64_C(1) << 20))
      die("maxBufferSize is below the oracle range");

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
   const VkPhysicalDeviceVulkan13Features enabled13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .maintenance4 = VK_TRUE,
   };
   const char *extensions[] = {VK_KHR_MAINTENANCE_4_EXTENSION_NAME};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enabled13,
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
   LOAD_DEVICE(device, vkGetDeviceBufferMemoryRequirementsKHR);
   LOAD_DEVICE(device, vkCreateImage);
   LOAD_DEVICE(device, vkDestroyImage);
   LOAD_DEVICE(device, vkGetImageMemoryRequirements2);
   LOAD_DEVICE(device, vkGetDeviceImageMemoryRequirements);
   LOAD_DEVICE(device, vkGetDeviceImageMemoryRequirementsKHR);
   LOAD_DEVICE(device, vkGetImageSparseMemoryRequirements2);
   LOAD_DEVICE(device, vkGetDeviceImageSparseMemoryRequirements);
   LOAD_DEVICE(device, vkGetDeviceImageSparseMemoryRequirementsKHR);

   const VkBufferCreateInfo buffers[] = {
      {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = 1,
         .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      },
      {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = 65537,
         .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      },
      {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = (UINT64_C(1) << 20) - 1,
         .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      },
   };
   for (uint32_t i = 0; i < 3; ++i)
      check_buffer_case(device, &buffers[i], i & 1, "buffer");

   const VkImageCreateInfo images[] = {
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_1D,
         .format = VK_FORMAT_R8_UNORM,
         .extent = {37, 1, 1},
         .mipLevels = 1,
         .arrayLayers = 3,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      },
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D,
         .format = VK_FORMAT_B8G8R8A8_UNORM,
         .extent = {65, 33, 1},
         .mipLevels = 4,
         .arrayLayers = 2,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      },
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_3D,
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .extent = {17, 9, 5},
         .mipLevels = 3,
         .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      },
   };
   for (uint32_t i = 0; i < 3; ++i)
      check_image_case(device, &images[i], i & 1, "image");

   printf(TEST_NAME ": PASS buffers=3 images=3 core+KHR exact "
                    "maxBufferSize=%llu (no GPU submissions)\n",
          (unsigned long long)maintenance4.maxBufferSize);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return EXIT_SUCCESS;
}
