/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public no-submit VK_EXT_memory_budget oracle for G17P. The drm-shim test
 * fixes HK_SYSMEM so allocation, free, and reuse deltas are byte-exact.
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

#define TEST_NAME       "g17p-vulkan-memory-budget"
#define T8140_DEVICE_ID 0x8140
#define FIXED_HEAP_SIZE (UINT64_C(1) << 30)
#define PAGE_SIZE       16384
#define BUDGET_GRANULE  (UINT64_C(1) << 20)

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceMemoryProperties2);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkEnumerateDeviceExtensionProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkAllocateMemory);
VK_FN(vkFreeMemory);

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

static uint64_t
expected_budget(uint64_t heap_size, uint64_t used)
{
   const uint64_t available = heap_size - used;
   uint64_t allocatable = available - (available / 10);
   allocatable &= ~(BUDGET_GRANULE - 1);
   if (allocatable > heap_size - used)
      allocatable = heap_size - used;
   return used + allocatable;
}

static VkPhysicalDeviceMemoryProperties
check_budget(VkPhysicalDevice physical, uint64_t expected_usage,
             const char *phase)
{
   VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
   };
   VkPhysicalDeviceMemoryProperties2 properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
      .pNext = &budget,
   };
   vkGetPhysicalDeviceMemoryProperties2(physical, &properties);

   if (properties.memoryProperties.memoryHeapCount != 1 ||
       properties.memoryProperties.memoryHeaps[0].size != FIXED_HEAP_SIZE)
      die("fixed drm-shim heap shape is not exact");
   if (budget.heapUsage[0] != expected_usage) {
      fprintf(stderr, TEST_NAME ": %s usage=%" PRIu64 " expected=%" PRIu64
                                "\n",
              phase, budget.heapUsage[0], expected_usage);
      exit(EXIT_FAILURE);
   }

   const uint64_t expected = expected_budget(FIXED_HEAP_SIZE, expected_usage);
   if (budget.heapBudget[0] != expected) {
      fprintf(stderr, TEST_NAME ": %s budget=%" PRIu64 " expected=%" PRIu64
                                "\n",
              phase, budget.heapBudget[0], expected);
      exit(EXIT_FAILURE);
   }

   for (uint32_t i = 1; i < VK_MAX_MEMORY_HEAPS; ++i) {
      if (budget.heapUsage[i] || budget.heapBudget[i])
         die("unused memory-budget array entries are nonzero");
   }

   return properties.memoryProperties;
}

static VkDeviceMemory
allocate(VkDevice device, uint32_t memory_type, VkDeviceSize size)
{
   const VkMemoryAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
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
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties2);
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
   if (!has_extension(physical, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
      die("VK_EXT_memory_budget is not advertised");

   const VkPhysicalDeviceMemoryProperties memory_properties =
      check_budget(physical, 0, "initial");
   if (memory_properties.memoryTypeCount < 2 ||
       memory_properties.memoryTypes[0].heapIndex != 0 ||
       memory_properties.memoryTypes[1].heapIndex != 0)
      die("G17P cached/write-combined memory types are not on heap zero");

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
   const char *extensions[] = {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME};
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
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkFreeMemory);

   const VkDeviceSize requested[] = {
      1,
      PAGE_SIZE,
      PAGE_SIZE + 1,
      (UINT64_C(1) << 20) + 1,
   };
   VkDeviceMemory memory[4];
   uint64_t expected_usage = 0;
   for (uint32_t i = 0; i < 4; ++i) {
      memory[i] = allocate(device, i & 1, requested[i]);
      expected_usage += align_u64(requested[i], PAGE_SIZE);
      check_budget(physical, expected_usage, "allocation growth");
   }

   vkFreeMemory(device, memory[1], NULL);
   expected_usage -= align_u64(requested[1], PAGE_SIZE);
   check_budget(physical, expected_usage, "free cached/WC middle");
   vkFreeMemory(device, memory[2], NULL);
   expected_usage -= align_u64(requested[2], PAGE_SIZE);
   check_budget(physical, expected_usage, "free second middle");

   VkDeviceMemory reused = allocate(device, 1, requested[2]);
   expected_usage += align_u64(requested[2], PAGE_SIZE);
   check_budget(physical, expected_usage, "reuse allocation");
   vkFreeMemory(device, reused, NULL);
   expected_usage -= align_u64(requested[2], PAGE_SIZE);
   check_budget(physical, expected_usage, "reuse free");

   vkFreeMemory(device, memory[3], NULL);
   expected_usage -= align_u64(requested[3], PAGE_SIZE);
   vkFreeMemory(device, memory[0], NULL);
   expected_usage -= align_u64(requested[0], PAGE_SIZE);
   check_budget(physical, expected_usage, "all freed");

   printf(TEST_NAME ": PASS heap=%" PRIu64
                    " peak=%" PRIu64 " cached+WC reuse exact "
                    "(no GPU submissions)\n",
          FIXED_HEAP_SIZE,
          align_u64(requested[0], PAGE_SIZE) +
             align_u64(requested[1], PAGE_SIZE) +
             align_u64(requested[2], PAGE_SIZE) +
             align_u64(requested[3], PAGE_SIZE));

   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return EXIT_SUCCESS;
}
