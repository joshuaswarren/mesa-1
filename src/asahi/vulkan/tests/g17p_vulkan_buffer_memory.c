/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public no-submit buffer-memory oracle for G17P.  It checks every advertised
 * host-visible coherent memory type, nonzero buffer binding offsets, coherent
 * flush/invalidate, and data persistence across core and KHR map lifetimes.
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

#define TEST_NAME       "g17p-vulkan-buffer-memory"
#define T8140_DEVICE_ID 0x8140

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
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
VK_FN(vkMapMemory);
VK_FN(vkUnmapMemory);
VK_FN(vkMapMemory2KHR);
VK_FN(vkUnmapMemory2KHR);
VK_FN(vkFlushMappedMemoryRanges);
VK_FN(vkInvalidateMappedMemoryRanges);

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

static uint64_t
align_u64(uint64_t value, uint64_t alignment)
{
   return (value + alignment - 1) & ~(alignment - 1);
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

static VkBuffer
create_buffer(VkDevice device, VkDeviceSize size)
{
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer buffer;
   check(vkCreateBuffer(device, &info, NULL, &buffer), "vkCreateBuffer");
   return buffer;
}

static uint8_t
initial_byte(size_t offset, uint32_t memory_type)
{
   return (uint8_t)(0x5a ^ (offset * 37u) ^ (memory_type * 0x63u));
}

static void
run_memory_type(VkDevice device, const VkPhysicalDeviceProperties *properties,
                uint32_t memory_type, VkMemoryPropertyFlags flags)
{
   const VkDeviceSize size_a = 4096;
   const VkDeviceSize size_b = 8192;
   VkBuffer buffer_a = create_buffer(device, size_a);
   VkBuffer buffer_b = create_buffer(device, size_b);
   VkMemoryRequirements req_a, req_b;
   vkGetBufferMemoryRequirements(device, buffer_a, &req_a);
   vkGetBufferMemoryRequirements(device, buffer_b, &req_b);

   if (!(req_a.memoryTypeBits & (1u << memory_type)) ||
       !(req_b.memoryTypeBits & (1u << memory_type)))
      die("host-visible memory type is incompatible with an ordinary buffer");

   const VkDeviceSize atom = properties->limits.nonCoherentAtomSize;
   const VkDeviceSize map_alignment = properties->limits.minMemoryMapAlignment;
   const VkDeviceSize range_alignment = atom > map_alignment ? atom : map_alignment;
   const VkDeviceSize alignment_a =
      req_a.alignment > range_alignment ? req_a.alignment : range_alignment;
   const VkDeviceSize alignment_b =
      req_b.alignment > range_alignment ? req_b.alignment : range_alignment;
   const VkDeviceSize offset_a = alignment_a;
   const VkDeviceSize offset_b =
      align_u64(offset_a + req_a.size + range_alignment, alignment_b);
   const VkDeviceSize allocation_size =
      align_u64(offset_b + req_b.size + range_alignment, range_alignment);

   const VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = allocation_size,
      .memoryTypeIndex = memory_type,
   };
   VkDeviceMemory memory;
   check(vkAllocateMemory(device, &alloc_info, NULL, &memory),
         "vkAllocateMemory");
   check(vkBindBufferMemory(device, buffer_a, memory, offset_a),
         "vkBindBufferMemory(A)");
   check(vkBindBufferMemory(device, buffer_b, memory, offset_b),
         "vkBindBufferMemory(B)");

   uint8_t *expected = malloc(allocation_size);
   if (!expected)
      die("out of host memory");
   for (size_t i = 0; i < allocation_size; ++i)
      expected[i] = initial_byte(i, memory_type);
   memset(expected + offset_a, 0xa1, size_a);
   memset(expected + offset_b, 0xb2, size_b);

   void *map = NULL;
   check(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &map),
         "vkMapMemory(full)");
   if (!map || ((uintptr_t)map % properties->limits.minMemoryMapAlignment))
      die("full memory map is null or incorrectly aligned");
   memcpy(map, expected, allocation_size);

   const VkMappedMemoryRange ranges[] = {
      {
         .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = memory,
         .offset = offset_a,
         .size = align_u64(size_a, atom),
      },
      {
         .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = memory,
         .offset = offset_b,
         .size = VK_WHOLE_SIZE,
      },
   };
   check(vkFlushMappedMemoryRanges(device, 2, ranges),
         "vkFlushMappedMemoryRanges");
   check(vkInvalidateMappedMemoryRanges(device, 2, ranges),
         "vkInvalidateMappedMemoryRanges");
   if (memcmp(map, expected, allocation_size))
      die("coherent bytes changed across flush/invalidate");
   vkUnmapMemory(device, memory);

   VkMemoryMapInfoKHR map_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR,
      .memory = memory,
      .offset = offset_a,
      .size = VK_WHOLE_SIZE,
   };
   map = NULL;
   check(vkMapMemory2KHR(device, &map_info, &map),
         "vkMapMemory2KHR(nonzero offset)");
   if (!map || memcmp(map, expected + offset_a, allocation_size - offset_a))
      die("bytes did not persist across core unmap/KHR remap");
   const size_t relative_b = offset_b - offset_a;
   ((uint8_t *)map)[relative_b + 17] ^= 0xff;
   expected[offset_b + 17] ^= 0xff;

   const VkMemoryUnmapInfoKHR unmap_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR,
      .memory = memory,
   };
   check(vkUnmapMemory2KHR(device, &unmap_info), "vkUnmapMemory2KHR");

   map = NULL;
   check(vkMapMemory(device, memory, 0, allocation_size, 0, &map),
         "vkMapMemory(remap)");
   if (!map || memcmp(map, expected, allocation_size))
      die("bytes did not persist across KHR unmap/core remap");
   vkUnmapMemory(device, memory);

   printf(TEST_NAME ": type=%u flags=%#x allocation=%" PRIu64
                    " offsets=%" PRIu64 "/%" PRIu64 " PASS\n",
          memory_type, flags, allocation_size, offset_a, offset_b);

   free(expected);
   vkDestroyBuffer(device, buffer_b, NULL);
   vkDestroyBuffer(device, buffer_a, NULL);
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
   if (!has_extension(physical, VK_KHR_MAP_MEMORY_2_EXTENSION_NAME))
      die("VK_KHR_map_memory2 is not advertised");

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
   const char *extensions[] = {VK_KHR_MAP_MEMORY_2_EXTENSION_NAME};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
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
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkUnmapMemory);
   LOAD_DEVICE(device, vkMapMemory2KHR);
   LOAD_DEVICE(device, vkUnmapMemory2KHR);
   LOAD_DEVICE(device, vkFlushMappedMemoryRanges);
   LOAD_DEVICE(device, vkInvalidateMappedMemoryRanges);

   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
   unsigned tested = 0;
   bool saw_cached = false, saw_write_combined = false;
   for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      const VkMemoryPropertyFlags flags =
         memory_properties.memoryTypes[i].propertyFlags;
      const VkMemoryPropertyFlags required =
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      if ((flags & required) != required)
         continue;
      run_memory_type(device, &properties, i, flags);
      saw_cached |= flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
      saw_write_combined |= !(flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
      tested++;
   }
   if (!tested || !saw_cached || !saw_write_combined)
      die("cached and write-combined coherent memory types were not both tested");

   printf(TEST_NAME ": PASS types=%u atom=%" PRIu64 " map-align=%zu\n",
          tested, properties.limits.nonCoherentAtomSize,
          properties.limits.minMemoryMapAlignment);

   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return EXIT_SUCCESS;
}
