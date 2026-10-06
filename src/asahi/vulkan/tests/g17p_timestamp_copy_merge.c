/* SPDX-License-Identifier: MIT
 * Preserve an earlier timestamp dependency when two later CDMs merge.
 * Default: CPU recording through the G17P no-op DRM shim, no readback claim.
 * Hardware: build with G17P_TIMESTAMP_COPY_MERGE_HARDWARE and pass --hardware.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define BULK_BYTES 1261568u
#define QUERY_BYTES 8u
#define GUARD_BYTES 16u
#define QUERY_STRIDE (QUERY_BYTES + GUARD_BYTES)
#define QUERY_FLAGS (VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
#define CHECK(call) do { \
   VkResult result = (call); \
   if (result != VK_SUCCESS) { \
      fprintf(stderr, "%s: VkResult %d\n", #call, result); \
      exit(EXIT_FAILURE); \
   } \
} while (0)

static void
fail(const char *message)
{
   fprintf(stderr, "timestamp-copy-merge: %s\n", message);
   exit(EXIT_FAILURE);
}

static void
check_mode(bool hardware)
{
   if (hardware) {
#ifndef G17P_TIMESTAMP_COPY_MERGE_HARDWARE
      fail("hardware mode requires an explicitly enabled build");
#else
      if (getenv("LD_PRELOAD"))
         fail("hardware mode rejects LD_PRELOAD");
#endif
   } else {
      const char *chip = getenv("ASAHI_SHIM_CHIP");
      Dl_info info = {0};
      void *entry = dlsym(RTLD_DEFAULT, "ioctl");
      if (!chip || strcmp(chip, "g17p") || !entry || !dladdr(entry, &info) ||
          !info.dli_fname ||
          strcmp(strrchr(info.dli_fname, '/') ?
                 strrchr(info.dli_fname, '/') + 1 : info.dli_fname,
                 "libasahi_noop_drm_shim.so"))
         fail("record mode requires the loaded G17P no-op DRM shim");
   }
}

struct buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   void *map;
};

static struct buffer
create_buffer(VkDevice device, const VkPhysicalDeviceMemoryProperties *memory,
              VkDeviceSize size, VkBufferUsageFlags usage)
{
   struct buffer buffer = {0};
   VkBufferCreateInfo create = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
   };
   CHECK(vkCreateBuffer(device, &create, NULL, &buffer.buffer));
   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, buffer.buffer, &requirements);
   const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   uint32_t type;
   for (type = 0; type < memory->memoryTypeCount; type++) {
      if ((requirements.memoryTypeBits & (1u << type)) &&
          (memory->memoryTypes[type].propertyFlags & required) == required)
         break;
   }
   if (type == memory->memoryTypeCount)
      fail("host-visible coherent memory unavailable");
   VkMemoryAllocateInfo allocate = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = type,
   };
   CHECK(vkAllocateMemory(device, &allocate, NULL, &buffer.memory));
   CHECK(vkBindBufferMemory(device, buffer.buffer, buffer.memory, 0));
   CHECK(vkMapMemory(device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.map));
   return buffer;
}

static void
destroy_buffer(VkDevice device, struct buffer buffer)
{
   vkUnmapMemory(device, buffer.memory);
   vkDestroyBuffer(device, buffer.buffer, NULL);
   vkFreeMemory(device, buffer.memory, NULL);
}

int
main(int argc, char **argv)
{
   bool hardware = argc == 2 && !strcmp(argv[1], "--hardware");
   if (argc != 1 && !hardware)
      fail("usage: g17p_timestamp_copy_merge [--hardware]");
   check_mode(hardware);

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "g17p-timestamp-copy-merge",
      .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));
   uint32_t count = 0;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
   if (count != 1)
      fail("select exactly one G17P device through VK_DRIVER_FILES");
   VkPhysicalDevice physical;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   VkPhysicalDeviceProperties properties;
   vkGetPhysicalDeviceProperties(physical, &properties);
   if (properties.deviceID != 0x8140)
      fail("requires G17P");
   VkPhysicalDeviceVulkan12Features features12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &features12,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   if (!features12.hostQueryReset)
      fail("hostQueryReset unsupported");
   VkQueueFamilyProperties queues[32];
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
   if (!count || count > 32)
      fail("unsupported queue-family count");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues);
   uint32_t family;
   for (family = 0; family < count; family++) {
      if ((queues[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
          queues[family].timestampValidBits)
         break;
   }
   if (family == count)
      fail("graphics queue with timestamps unavailable");
   float priority = 1.0f;
   VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   features12 = (VkPhysicalDeviceVulkan12Features) {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .hostQueryReset = VK_TRUE,
   };
   VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features12,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   VkDevice device;
   CHECK(vkCreateDevice(physical, &device_info, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   VkPhysicalDeviceMemoryProperties memory;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory);
   VkQueryPoolCreateInfo query_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = 1,
   };
   VkQueryPool query_pool;
   CHECK(vkCreateQueryPool(device, &query_info, NULL, &query_pool));
   /* GPU CmdResetQueryPool would add ordering that masks this regression. */
   vkResetQueryPool(device, query_pool, 0, 1);
   struct buffer results = create_buffer(device, &memory, QUERY_STRIDE,
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   struct buffer source = create_buffer(device, &memory, BULK_BYTES,
                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
   struct buffer destination = create_buffer(device, &memory, BULK_BYTES,
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   for (unsigned i = 0; i < BULK_BYTES; i++)
      ((unsigned char *)source.map)[i] = (unsigned char)(i * 131u + 17u);
   memset(destination.map, 0xcc, BULK_BYTES);
   memset(results.map, 0xff, QUERY_STRIDE);

   VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(device, &pool_info, NULL, &pool));
   VkCommandBufferAllocateInfo allocate = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command;
   CHECK(vkAllocateCommandBuffers(device, &allocate, &command));
   VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   CHECK(vkBeginCommandBuffer(command, &begin));
   vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                       query_pool, 0);
   VkBufferCopy copy = {.size = BULK_BYTES};
   vkCmdCopyBuffer(command, source.buffer, destination.buffer, 1, &copy);
   vkCmdCopyQueryPoolResults(command, query_pool, 0, 1, results.buffer,
                            0, QUERY_STRIDE, QUERY_FLAGS);
   CHECK(vkEndCommandBuffer(command));
   VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command,
   };
   CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   /* Keep this in a SECOND submission, after the first wait. Adding HOST_READ
    * to the first command buffer can mask the lost timestamp dependency. */
   CHECK(vkResetCommandPool(device, pool, 0));
   CHECK(vkBeginCommandBuffer(command, &begin));
   VkMemoryBarrier host = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(command));
   CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   if (hardware) {
      uint64_t timestamp = 0, copied = 0;
      CHECK(vkGetQueryPoolResults(device, query_pool, 0, 1, sizeof(timestamp),
                                 &timestamp, sizeof(timestamp),
                                 VK_QUERY_RESULT_64_BIT));
      memcpy(&copied, results.map, sizeof(copied));
      if (!timestamp || timestamp == UINT64_MAX || copied != timestamp)
         fail("timestamp copy readback mismatch");
      if (memcmp(source.map, destination.map, BULK_BYTES))
         fail("bulk copy readback mismatch");
      for (unsigned i = QUERY_BYTES; i < QUERY_STRIDE; i++) {
         if (((unsigned char *)results.map)[i] != 0xff)
            fail("query guard overwritten");
      }
      printf("TIMESTAMP_COPY_MERGE_READBACK_PASS timestamp=%" PRIu64
             " bytes=%u guards=%u\n", timestamp, BULK_BYTES, GUARD_BYTES);
   } else {
      puts("TIMESTAMP_COPY_MERGE_RECORDED results_not_checked=1");
   }

   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyQueryPool(device, query_pool, NULL);
   destroy_buffer(device, results);
   destroy_buffer(device, source);
   destroy_buffer(device, destination);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return EXIT_SUCCESS;
}
