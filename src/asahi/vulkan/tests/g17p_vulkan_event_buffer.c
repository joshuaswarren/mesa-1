/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_NAME "g17p-vulkan-event-buffer"
#define T8140_DEVICE_ID 0x8140
#define GUARD_WORDS 8
#define PAYLOAD_WORDS 64
#define TOTAL_WORDS (2 * GUARD_WORDS + PAYLOAD_WORDS)
#define SRC_GUARD UINT32_C(0x13579bdf)
#define DST_GUARD UINT32_C(0x2468ace0)
#define SRC_SENTINEL UINT32_C(0xa5a5a5a5)
#define DST_SENTINEL UINT32_C(0x5a5a5a5a)
#define PAYLOAD_VALUE UINT32_C(0xc001d00d)

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceFeatures2);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkGetDeviceQueue);
VK_FN(vkCreateBuffer);
VK_FN(vkDestroyBuffer);
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkFreeMemory);
VK_FN(vkBindBufferMemory);
VK_FN(vkMapMemory);
VK_FN(vkUnmapMemory);
VK_FN(vkCreateEvent);
VK_FN(vkDestroyEvent);
VK_FN(vkGetEventStatus);
VK_FN(vkSetEvent);
VK_FN(vkResetEvent);
VK_FN(vkCreateCommandPool);
VK_FN(vkDestroyCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdFillBuffer);
VK_FN(vkCmdSetEvent2);
VK_FN(vkCmdWaitEvents2);
VK_FN(vkCmdCopyBuffer);
VK_FN(vkCmdResetEvent2);
VK_FN(vkCmdPipelineBarrier2);
VK_FN(vkEndCommandBuffer);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);

struct test_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint32_t *map;
};

static void
die(const char *message)
{
   fprintf(stderr, TEST_NAME ": %s\n", message);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *operation)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, TEST_NAME ": %s returned %d\n", operation, result);
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

static uint32_t
memory_type(const VkPhysicalDeviceMemoryProperties *properties, uint32_t mask)
{
   const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((mask & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }

   die("host-visible coherent memory is unavailable");
   return UINT32_MAX;
}

static struct test_buffer
create_buffer(VkDevice device,
              const VkPhysicalDeviceMemoryProperties *properties)
{
   const VkDeviceSize size = TOTAL_WORDS * sizeof(uint32_t);
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   struct test_buffer result = {0};
   check(vkCreateBuffer(device, &buffer_info, NULL, &result.buffer),
         "vkCreateBuffer");

   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, result.buffer, &requirements);
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = memory_type(properties, requirements.memoryTypeBits),
   };
   check(vkAllocateMemory(device, &memory_info, NULL, &result.memory),
         "vkAllocateMemory");
   check(vkBindBufferMemory(device, result.buffer, result.memory, 0),
         "vkBindBufferMemory");
   check(vkMapMemory(device, result.memory, 0, size, 0,
                     (void **)&result.map),
         "vkMapMemory");
   return result;
}

static void
destroy_buffer(VkDevice device, struct test_buffer *buffer)
{
   vkUnmapMemory(device, buffer->memory);
   vkDestroyBuffer(device, buffer->buffer, NULL);
   vkFreeMemory(device, buffer->memory, NULL);
}

static void
seed_buffer(struct test_buffer *buffer, uint32_t guard, uint32_t sentinel)
{
   for (uint32_t i = 0; i < TOTAL_WORDS; ++i) {
      const bool payload = i >= GUARD_WORDS &&
                           i < GUARD_WORDS + PAYLOAD_WORDS;
      buffer->map[i] = payload ? sentinel : guard;
   }
}

static bool
verify_buffer(const struct test_buffer *buffer, uint32_t guard,
              uint32_t payload)
{
   for (uint32_t i = 0; i < TOTAL_WORDS; ++i) {
      const bool in_payload = i >= GUARD_WORDS &&
                              i < GUARD_WORDS + PAYLOAD_WORDS;
      const uint32_t expected = in_payload ? payload : guard;
      if (buffer->map[i] != expected) {
         fprintf(stderr, TEST_NAME ": word %u is %#x, expected %#x\n", i,
                 buffer->map[i], expected);
         return false;
      }
   }
   return true;
}

int
main(int argc, char **argv)
{
   const bool record_only = argc == 2 && !strcmp(argv[1], "--record-only");
   if (argc > 2 || (argc == 2 && !record_only))
      die("usage: " TEST_NAME " [--record-only]");

   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());
   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("libvulkan.so.1 has no vkGetInstanceProcAddr");
   LOAD_GLOBAL(vkCreateInstance);

   const VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = TEST_NAME,
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures2);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      die("physical-device allocation failed");
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

   VkPhysicalDeviceSynchronization2Features available_sync2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 available = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &available_sync2,
   };
   vkGetPhysicalDeviceFeatures2(physical, &available);
   if (!available_sync2.synchronization2)
      die("synchronization2 is not advertised");

   uint32_t family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, NULL);
   VkQueueFamilyProperties *families = calloc(family_count, sizeof(*families));
   if (!families)
      die("queue-family allocation failed");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families);
   uint32_t family = UINT32_MAX;
   for (uint32_t i = 0; i < family_count; ++i) {
      if (families[i].queueCount &&
          (families[i].queueFlags & VK_QUEUE_TRANSFER_BIT)) {
         family = i;
         break;
      }
   }
   free(families);
   if (family == UINT32_MAX)
      die("device has no transfer queue");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkPhysicalDeviceSynchronization2Features enable_sync2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
      .synchronization2 = VK_TRUE,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enable_sync2,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCreateBuffer);
   LOAD_DEVICE(device, vkDestroyBuffer);
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkFreeMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkUnmapMemory);
   LOAD_DEVICE(device, vkCreateEvent);
   LOAD_DEVICE(device, vkDestroyEvent);
   LOAD_DEVICE(device, vkGetEventStatus);
   LOAD_DEVICE(device, vkSetEvent);
   LOAD_DEVICE(device, vkResetEvent);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkDestroyCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdFillBuffer);
   LOAD_DEVICE(device, vkCmdSetEvent2);
   LOAD_DEVICE(device, vkCmdWaitEvents2);
   LOAD_DEVICE(device, vkCmdCopyBuffer);
   LOAD_DEVICE(device, vkCmdResetEvent2);
   LOAD_DEVICE(device, vkCmdPipelineBarrier2);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkDestroyFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);

   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
   struct test_buffer source = create_buffer(device, &memory_properties);
   struct test_buffer destination = create_buffer(device, &memory_properties);
   seed_buffer(&source, SRC_GUARD, SRC_SENTINEL);
   seed_buffer(&destination, DST_GUARD, DST_SENTINEL);

   const VkEventCreateInfo event_info = {
      .sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO,
   };
   VkEvent event;
   check(vkCreateEvent(device, &event_info, NULL, &event), "vkCreateEvent");
   if (vkGetEventStatus(device, event) != VK_EVENT_RESET)
      die("new event was not RESET");
   check(vkSetEvent(device, event), "vkSetEvent");
   if (vkGetEventStatus(device, event) != VK_EVENT_SET)
      die("host SET was not visible");
   check(vkResetEvent(device, event), "vkResetEvent");
   if (vkGetEventStatus(device, event) != VK_EVENT_RESET)
      die("host RESET was not visible");

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VkCommandPool pool;
   check(vkCreateCommandPool(device, &pool_info, NULL, &pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command;
   check(vkAllocateCommandBuffers(device, &command_info, &command),
         "vkAllocateCommandBuffers");
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command, &begin_info), "vkBeginCommandBuffer");

   const VkDeviceSize payload_offset = GUARD_WORDS * sizeof(uint32_t);
   const VkDeviceSize payload_size = PAYLOAD_WORDS * sizeof(uint32_t);
   vkCmdFillBuffer(command, source.buffer, payload_offset, payload_size,
                   PAYLOAD_VALUE);

   const VkMemoryBarrier2 set_barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
   };
   const VkDependencyInfo set_dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &set_barrier,
   };
   vkCmdSetEvent2(command, event, &set_dependency);

   const VkBufferMemoryBarrier2 wait_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = source.buffer,
      .offset = payload_offset,
      .size = payload_size,
   };
   const VkDependencyInfo wait_dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &wait_barrier,
   };
   vkCmdWaitEvents2(command, 1, &event, &wait_dependency);

   const VkBufferCopy copy = {
      .srcOffset = payload_offset,
      .dstOffset = payload_offset,
      .size = payload_size,
   };
   vkCmdCopyBuffer(command, source.buffer, destination.buffer, 1, &copy);
   vkCmdResetEvent2(command, event, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

   const VkBufferMemoryBarrier2 host_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
      .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = destination.buffer,
      .offset = payload_offset,
      .size = payload_size,
   };
   const VkDependencyInfo host_dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &host_barrier,
   };
   vkCmdPipelineBarrier2(command, &host_dependency);
   check(vkEndCommandBuffer(command), "vkEndCommandBuffer");

   if (record_only) {
      if (!verify_buffer(&source, SRC_GUARD, SRC_SENTINEL) ||
          !verify_buffer(&destination, DST_GUARD, DST_SENTINEL) ||
          vkGetEventStatus(device, event) != VK_EVENT_RESET)
         die("recording mutated host-visible state");
      printf(TEST_NAME ": RECORD PASS — synchronization2 advertised; "
             "fill/set/wait(buffer)/copy/reset recorded with guards intact\n");
   } else {
      const VkFenceCreateInfo fence_info = {
         .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      };
      VkFence fence;
      check(vkCreateFence(device, &fence_info, NULL, &fence),
            "vkCreateFence");
      const VkSubmitInfo submit_info = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &command,
      };
      check(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
      check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences");

      if (!verify_buffer(&source, SRC_GUARD, PAYLOAD_VALUE) ||
          !verify_buffer(&destination, DST_GUARD, PAYLOAD_VALUE) ||
          vkGetEventStatus(device, event) != VK_EVENT_RESET)
         die("event buffer dependency did not produce the exact result");
      printf(TEST_NAME ": PASS — 64 payload words exact, 32 guard words "
             "intact, event RESET\n");
      vkDestroyFence(device, fence, NULL);
   }

   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyEvent(device, event, NULL);
   destroy_buffer(device, &destination);
   destroy_buffer(device, &source);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return EXIT_SUCCESS;
}
