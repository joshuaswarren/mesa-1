/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Sparse multiview clear/load/store preservation. Two passes address separate
 * API layers with no dependency between them; inactive layers must survive.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#define CHECK(call)                                                            \
  do {                                                                         \
    VkResult r = (call);                                                       \
    if (r != VK_SUCCESS) {                                                     \
      fprintf(stderr, "%s: %d\n", #call, r);                                   \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)
static uint32_t memory_type(VkPhysicalDevice p, uint32_t bits,
                            VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties m;
  vkGetPhysicalDeviceMemoryProperties(p, &m);
  for (uint32_t i = 0; i < m.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  fprintf(stderr, "no memory type\n");
  exit(2);
}
int main(int argc, char **argv) {
  unsigned loops = argc > 1 ? strtoul(argv[1], NULL, 10) : 200;
  unsigned width = argc > 2 ? strtoul(argv[2], NULL, 10) : 128;
  VkInstance inst;
  VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                          .apiVersion = VK_API_VERSION_1_3};
  VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                              .pApplicationInfo = &ai};
  CHECK(vkCreateInstance(&ici, NULL, &inst));
  uint32_t n = 0;
  CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
  if (!n)
    return 2;
  VkPhysicalDevice *physical = calloc(n, sizeof(*physical));
  CHECK(vkEnumeratePhysicalDevices(inst, &n, physical));
  VkPhysicalDevice p = physical[0];
  for (uint32_t j = 0; j < n; j++) {
    VkPhysicalDeviceProperties candidate;
    vkGetPhysicalDeviceProperties(physical[j], &candidate);
    if (candidate.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
      p = physical[j];
      break;
    }
  }
  free(physical);
  VkPhysicalDeviceProperties prop;
  vkGetPhysicalDeviceProperties(p, &prop);
  printf("device=%s loops=%u width=%u\n", prop.deviceName, loops, width);
  uint32_t families = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(p, &families, NULL);
  VkQueueFamilyProperties *fp = calloc(families, sizeof(*fp));
  vkGetPhysicalDeviceQueueFamilyProperties(p, &families, fp);
  uint32_t family = 0;
  while (family < families && !(fp[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
    family++;
  free(fp);
  if (family == families)
    return 2;
  float priority = 1;
  VkDeviceQueueCreateInfo qi = {.sType =
                                    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                .queueFamilyIndex = family,
                                .queueCount = 1,
                                .pQueuePriorities = &priority};
  VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE};
  VkPhysicalDeviceVulkan11Features f11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .pNext = &f13,
      .multiview = VK_TRUE};
  VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                           .pNext = &f11,
                           .queueCreateInfoCount = 1,
                           .pQueueCreateInfos = &qi};
  VkDevice dev;
  CHECK(vkCreateDevice(p, &di, NULL, &dev));
  VkQueue queue;
  vkGetDeviceQueue(dev, family, 0, &queue);
  VkImageCreateInfo imci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                            .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UINT,
                            .extent = {width, width, 1},
                            .mipLevels = 1,
                            .arrayLayers = 2,
                            .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  VkImage image;
  CHECK(vkCreateImage(dev, &imci, NULL, &image));
  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(dev, image, &mr);
  VkMemoryAllocateInfo mi = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = memory_type(p, mr.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
  VkDeviceMemory imem;
  CHECK(vkAllocateMemory(dev, &mi, NULL, &imem));
  CHECK(vkBindImageMemory(dev, image, imem, 0));
  VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
      .format = imci.format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2}};
  VkImageView view;
  CHECK(vkCreateImageView(dev, &vi, NULL, &view));
  VkDeviceSize bytes = (VkDeviceSize)width * width * 8;
  VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                           .size = bytes,
                           .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  VkBuffer buffer;
  CHECK(vkCreateBuffer(dev, &bi, NULL, &buffer));
  vkGetBufferMemoryRequirements(dev, buffer, &mr);
  mi.allocationSize = mr.size;
  mi.memoryTypeIndex =
      memory_type(p, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  VkDeviceMemory bmem;
  CHECK(vkAllocateMemory(dev, &mi, NULL, &bmem));
  CHECK(vkBindBufferMemory(dev, buffer, bmem, 0));
  uint8_t *map;
  CHECK(vkMapMemory(dev, bmem, 0, VK_WHOLE_SIZE, 0, (void **)&map));
  VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = family};
  VkCommandPool pool;
  CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
  VkCommandBufferAllocateInfo ca = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
  VkCommandBuffer cmd;
  CHECK(vkAllocateCommandBuffers(dev, &ca, &cmd));
  VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  CHECK(vkCreateFence(dev, &fi, NULL, &fence));
  unsigned fails = 0;
  for (unsigned i = 0; i < loops; i++) {
    CHECK(vkResetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo cb = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cb));
    VkImageMemoryBarrier bar = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = vi.subresourceRange};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                         NULL, 0, NULL, 1, &bar);
    uint8_t expect[2][4] = {{(i * 13 + 3) & 255, 255, 246, 255},
                            {(i * 17 + 29) & 255, 42, 117, 193}};
    for (unsigned layer = 0; layer < 2; layer++) {
      VkRenderingAttachmentInfo att = {
          .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
          .imageView = view,
          .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
      for (unsigned c = 0; c < 4; c++)
        att.clearValue.color.uint32[c] = expect[layer][c];
      VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                            .renderArea = {{0, 0}, {width, width}},
                            .layerCount = 1,
                            .viewMask = 1u << layer,
                            .colorAttachmentCount = 1,
                            .pColorAttachments = &att};
      vkCmdBeginRendering(cmd, &ri);
      vkCmdEndRendering(cmd);
    }
    bar.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bar.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
                         &bar);
    VkBufferImageCopy copy = {
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 2},
        .imageExtent = {width, width, 1}};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           buffer, 1, &copy);
    VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                            .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0,
                         NULL);
    CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1,
                       .pCommandBuffers = &cmd};
    CHECK(vkQueueSubmit(queue, 1, &si, fence));
    CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
    CHECK(vkResetFences(dev, 1, &fence));
    VkMappedMemoryRange range = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                 .memory = bmem,
                                 .size = VK_WHOLE_SIZE};
    CHECK(vkInvalidateMappedMemoryRanges(dev, 1, &range));
    for (unsigned layer = 0; layer < 2; layer++) {
      unsigned mismatches = 0;
      for (unsigned px = 0; px < width * width; px++)
        if (memcmp(map + (layer * width * width + px) * 4, expect[layer], 4))
          mismatches++;
      if (mismatches) {
        fails++;
        printf("FAIL iteration=%u layer=%u mismatch=%u/%u\n", i, layer,
               mismatches, width * width);
      }
    }
  }
  printf("RESULT iterations=%u layer_failures=%u\n", loops, fails);
  vkUnmapMemory(dev, bmem);
  vkDestroyFence(dev, fence, NULL);
  vkDestroyCommandPool(dev, pool, NULL);
  vkDestroyImageView(dev, view, NULL);
  vkDestroyBuffer(dev, buffer, NULL);
  vkFreeMemory(dev, bmem, NULL);
  vkDestroyImage(dev, image, NULL);
  vkFreeMemory(dev, imem, NULL);
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(inst, NULL);
  return fails ? 1 : 0;
}
