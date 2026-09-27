/* SPDX-License-Identifier: MIT */
/* Empty graphics query: reset -> end -> waiting GPU copy in one command buffer.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r = (call);                                                     \
      if (r != VK_SUCCESS) {                                                   \
         fprintf(stderr, "%s: %d\n", #call, r);                                \
         return 1;                                                             \
      }                                                                        \
   } while (0)

int main(void)
{
   VkInstance instance;
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice gpu;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &gpu));
   if (count == 0)
      return 77;
   VkPhysicalDeviceProperties properties;
   vkGetPhysicalDeviceProperties(gpu, &properties);
   printf("device=%s\n", properties.deviceName);
   fflush(stdout);
   VkPhysicalDeviceFeatures features;
   vkGetPhysicalDeviceFeatures(gpu, &features);
   if (!features.pipelineStatisticsQuery)
      return 77;
   memset(&features, 0, sizeof(features));
   features.pipelineStatisticsQuery = VK_TRUE;
   uint32_t families = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(gpu, &families, NULL);
   VkQueueFamilyProperties *props = calloc(families, sizeof(*props));
   if (!props)
      return 1;
   vkGetPhysicalDeviceQueueFamilyProperties(gpu, &families, props);
   uint32_t family = 0;
   while (family < families &&
          !(props[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
      family++;
   if (family == families)
      return 3;
   float priority = 1.0f;
   VkDeviceQueueCreateInfo qci = {
       .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
       .queueFamilyIndex = family,
       .queueCount = 1,
       .pQueuePriorities = &priority};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci,
                             .pEnabledFeatures = &features};
   VkDevice device;
   CHECK(vkCreateDevice(gpu, &dci, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .size = 16,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buffer;
   CHECK(vkCreateBuffer(device, &bci, NULL, &buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(device, buffer, &req);
   VkPhysicalDeviceMemoryProperties memory;
   vkGetPhysicalDeviceMemoryProperties(gpu, &memory);
   uint32_t type = 0;
   VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   while (type < memory.memoryTypeCount &&
          (!(req.memoryTypeBits & (1u << type)) ||
           (memory.memoryTypes[type].propertyFlags & required) != required))
      type++;
   if (type == memory.memoryTypeCount)
      return 4;
   VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .allocationSize = req.size,
                               .memoryTypeIndex = type};
   VkDeviceMemory allocation;
   CHECK(vkAllocateMemory(device, &mai, NULL, &allocation));
   CHECK(vkBindBufferMemory(device, buffer, allocation, 0));
   uint64_t *data;
   CHECK(vkMapMemory(device, allocation, 0, VK_WHOLE_SIZE, 0, (void **)&data));
   data[0] = data[1] = UINT64_MAX;
   VkQueryPoolCreateInfo qpi = {
       .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
       .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS,
       .queryCount = 1,
       .pipelineStatistics =
           VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT};
   VkQueryPool pool;
   CHECK(vkCreateQueryPool(device, &qpi, NULL, &pool));
   VkCommandPoolCreateInfo cpi = {
       .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
       .queueFamilyIndex = family};
   VkCommandPool commands;
   CHECK(vkCreateCommandPool(device, &cpi, NULL, &commands));
   VkCommandBufferAllocateInfo cai = {
       .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
       .commandPool = commands,
       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
       .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(device, &cai, &cmd));
   VkCommandBufferBeginInfo begin = {
       .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cmd, &begin));
   vkCmdResetQueryPool(cmd, pool, 0, 1);
   vkCmdBeginQuery(cmd, pool, 0, 0);
   vkCmdEndQuery(cmd, pool, 0);
   vkCmdCopyQueryPoolResults(cmd, pool, 0, 1, buffer, 0, 16,
                             VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT |
                                 VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
   VkMemoryBarrier host_barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                   .srcAccessMask =
                                       VK_ACCESS_TRANSFER_WRITE_BIT,
                                   .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier, 0,
                        NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(device, &fci, NULL, &fence));
   VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                          .commandBufferCount = 1,
                          .pCommandBuffers = &cmd};
   CHECK(vkQueueSubmit(queue, 1, &submit, fence));
   CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ULL));
   printf("result=%" PRIu64 " available=%" PRIu64 "\n", data[0], data[1]);
   int failed = data[0] != 0 || data[1] != 1;
   vkDestroyFence(device, fence, NULL);
   vkDestroyCommandPool(device, commands, NULL);
   vkDestroyQueryPool(device, pool, NULL);
   vkUnmapMemory(device, allocation);
   vkDestroyBuffer(device, buffer, NULL);
   vkFreeMemory(device, allocation, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   free(props);
   return failed;
}
