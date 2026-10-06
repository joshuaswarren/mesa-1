/* SPDX-License-Identifier: MIT */
/* Allocation-only probe. No GPU submissions. A new device per repetition
 * prevents BO-cache hits. Device setup/teardown are outside allocation timing.
 * Build: cc -O2 -Wall -Wextra vk-alloc.c -lvulkan -o vk-alloc */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(clockid_t clock)
{
   struct timespec t;
   clock_gettime(clock, &t);
   return t.tv_sec + t.tv_nsec * 1e-9;
}

#define CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { \
   fprintf(stderr, "%s: %d\n", #x, r); return 1; } } while (0)

int main(int argc, char **argv)
{
   unsigned reps = argc > 1 ? strtoul(argv[1], NULL, 0) : 1;
   unsigned type = argc > 2 ? strtoul(argv[2], NULL, 0) : 0;
   VkDeviceSize size = argc > 3 ? strtoull(argv[3], NULL, 0) : 268435456;
   bool fresh = argc <= 4 || strcmp(argv[4], "pressure") != 0;
   VkInstance instance;
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 0;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
   if (!count) return 1;
   VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
   if (!devices) return 1;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, devices));
   VkPhysicalDevice physical = devices[0];
   free(devices);
   VkPhysicalDeviceProperties identity;
   vkGetPhysicalDeviceProperties(physical, &identity);
   fprintf(stderr, "device=%s driver=%u\n", identity.deviceName, identity.driverVersion);
   VkPhysicalDeviceMemoryProperties props;
   vkGetPhysicalDeviceMemoryProperties(physical, &props);
   if (type >= props.memoryTypeCount) return 1;
   fprintf(stderr, "type=%u flags=%#x bytes=%llu fresh_devices=%u\n", type,
      props.memoryTypes[type].propertyFlags, (unsigned long long)size, fresh);
   float priority = 1;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &priority};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
   VkDevice device = VK_NULL_HANDLE;
   if (!fresh) {
      CHECK(vkCreateDevice(physical, &dci, NULL, &device));
      VkMemoryAllocateInfo seed = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = 16384, .memoryTypeIndex = type};
      VkDeviceMemory small;
      CHECK(vkAllocateMemory(device, &seed, NULL, &small));
      vkFreeMemory(device, small, NULL);
   }
   printf("iteration,bytes,type,allocate_wall_s,allocate_cpu_s,free_wall_s,free_cpu_s\n");
   fflush(stdout);
   for (unsigned i = 0; i < reps; ++i) {
      if (fresh)
         CHECK(vkCreateDevice(physical, &dci, NULL, &device));
      VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = size, .memoryTypeIndex = type};
      VkDeviceMemory mem;
      double wall = now(CLOCK_MONOTONIC), cpu = now(CLOCK_PROCESS_CPUTIME_ID);
      CHECK(vkAllocateMemory(device, &ai, NULL, &mem));
      double acpu = now(CLOCK_PROCESS_CPUTIME_ID) - cpu;
      double awall = now(CLOCK_MONOTONIC) - wall;
      wall = now(CLOCK_MONOTONIC); cpu = now(CLOCK_PROCESS_CPUTIME_ID);
      vkFreeMemory(device, mem, NULL);
      double fcpu = now(CLOCK_PROCESS_CPUTIME_ID) - cpu;
      double fwall = now(CLOCK_MONOTONIC) - wall;
      printf("%u,%llu,%u,%.9f,%.9f,%.9f,%.9f\n", i,
         (unsigned long long)size, type, awall, acpu, fwall, fcpu);
      fflush(stdout);
      if (fresh)
         vkDestroyDevice(device, NULL);
   }
   if (!fresh)
      vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return 0;
}
