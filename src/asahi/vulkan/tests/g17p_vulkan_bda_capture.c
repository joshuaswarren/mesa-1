/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public no-submit buffer-device-address capture/replay oracle for G17P.
 * It checks the core and KHR entrypoints, both opaque capture-address
 * objects, and an exact destroy/recreate replay roundtrip.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_NAME       "g17p-vulkan-bda-capture"
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
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkFreeMemory);
VK_FN(vkBindBufferMemory);
VK_FN(vkGetBufferDeviceAddress);
VK_FN(vkGetBufferDeviceAddressKHR);
VK_FN(vkGetBufferOpaqueCaptureAddress);
VK_FN(vkGetBufferOpaqueCaptureAddressKHR);
VK_FN(vkGetDeviceMemoryOpaqueCaptureAddress);
VK_FN(vkGetDeviceMemoryOpaqueCaptureAddressKHR);

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

static VkBuffer
create_capture_buffer(VkDevice device, uint64_t replay_address)
{
   const VkBufferOpaqueCaptureAddressCreateInfo capture = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO,
      .opaqueCaptureAddress = replay_address,
   };
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = replay_address ? &capture : NULL,
      .flags = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
      .size = 4096,
      .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer buffer;
   check(vkCreateBuffer(device, &info, NULL, &buffer), "vkCreateBuffer");
   return buffer;
}

static VkDeviceMemory
allocate_capture_memory(VkDevice device, VkDeviceSize size,
                        uint32_t memory_type, uint64_t replay_address)
{
   const VkMemoryOpaqueCaptureAddressAllocateInfo capture = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO,
      .opaqueCaptureAddress = replay_address,
   };
   const VkMemoryAllocateFlagsInfo flags = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .pNext = replay_address ? &capture : NULL,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT |
               VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
   };
   const VkMemoryAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &flags,
      .allocationSize = size,
      .memoryTypeIndex = memory_type,
   };
   VkDeviceMemory memory;
   check(vkAllocateMemory(device, &info, NULL, &memory), "vkAllocateMemory");
   return memory;
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
      .pApplicationName = TEST_NAME,
      .apiVersion = VK_API_VERSION_1_2,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
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
   if (!has_extension(physical, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME))
      die("VK_KHR_buffer_device_address is not advertised");

   VkPhysicalDeviceBufferDeviceAddressFeatures bda_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &bda_features,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   if (!bda_features.bufferDeviceAddress ||
       !bda_features.bufferDeviceAddressCaptureReplay)
      die("buffer device-address capture/replay is not advertised");

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
   const char *extensions[] = {VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &bda_features,
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
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkFreeMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkGetBufferDeviceAddress);
   LOAD_DEVICE(device, vkGetBufferDeviceAddressKHR);
   LOAD_DEVICE(device, vkGetBufferOpaqueCaptureAddress);
   LOAD_DEVICE(device, vkGetBufferOpaqueCaptureAddressKHR);
   LOAD_DEVICE(device, vkGetDeviceMemoryOpaqueCaptureAddress);
   LOAD_DEVICE(device, vkGetDeviceMemoryOpaqueCaptureAddressKHR);

   VkBuffer buffer = create_capture_buffer(device, 0);
   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, buffer, &requirements);
   if (!requirements.memoryTypeBits)
      die("capture buffer has no compatible memory type");
   const uint32_t memory_type = __builtin_ctz(requirements.memoryTypeBits);
   VkDeviceMemory memory = allocate_capture_memory(
      device, requirements.size, memory_type, 0);
   check(vkBindBufferMemory(device, buffer, memory, 0),
         "vkBindBufferMemory(capture)");

   const VkBufferDeviceAddressInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = buffer,
   };
   const VkDeviceMemoryOpaqueCaptureAddressInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO,
      .memory = memory,
   };
   const uint64_t address = vkGetBufferDeviceAddress(device, &buffer_info);
   const uint64_t address_khr =
      vkGetBufferDeviceAddressKHR(device, &buffer_info);
   const uint64_t buffer_capture =
      vkGetBufferOpaqueCaptureAddress(device, &buffer_info);
   const uint64_t buffer_capture_khr =
      vkGetBufferOpaqueCaptureAddressKHR(device, &buffer_info);
   const uint64_t memory_capture =
      vkGetDeviceMemoryOpaqueCaptureAddress(device, &memory_info);
   const uint64_t memory_capture_khr =
      vkGetDeviceMemoryOpaqueCaptureAddressKHR(device, &memory_info);
   if (!address || address != address_khr || !buffer_capture ||
       buffer_capture != buffer_capture_khr || address != buffer_capture)
      die("core/KHR buffer address or opaque capture address mismatch");
   if (!memory_capture || memory_capture != memory_capture_khr)
      die("core/KHR memory opaque capture address is zero or mismatched");

   vkDestroyBuffer(device, buffer, NULL);
   vkFreeMemory(device, memory, NULL);

   buffer = create_capture_buffer(device, buffer_capture);
   memory = allocate_capture_memory(device, requirements.size, memory_type,
                                    memory_capture);
   check(vkBindBufferMemory(device, buffer, memory, 0),
         "vkBindBufferMemory(replay)");

   VkBufferDeviceAddressInfo replay_buffer_info = buffer_info;
   replay_buffer_info.buffer = buffer;
   VkDeviceMemoryOpaqueCaptureAddressInfo replay_memory_info = memory_info;
   replay_memory_info.memory = memory;
   if (vkGetBufferDeviceAddress(device, &replay_buffer_info) != address ||
       vkGetBufferOpaqueCaptureAddress(device, &replay_buffer_info) !=
          buffer_capture ||
       vkGetDeviceMemoryOpaqueCaptureAddress(device, &replay_memory_info) !=
          memory_capture)
      die("capture/replay address roundtrip changed");

   printf(TEST_NAME ": PASS address=%#" PRIx64
                    " buffer_capture=%#" PRIx64
                    " memory_capture=%#" PRIx64
                    " alignment=%" PRIu64 "\n",
          address, buffer_capture, memory_capture, requirements.alignment);

   vkDestroyBuffer(device, buffer, NULL);
   vkFreeMemory(device, memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return 0;
}
