/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Verify partial draws and clears in every mip and array layer of a 1D image.
 * Distinct signed integer colors identify omitted draws and crossed layers.
 * Submit and wait for each mip, then check all levels in one readback.
 */
#include "vulkan_mipmap_layers_shaders.h"
#include <stdbool.h>
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

struct buffer {
  VkBuffer buffer;
  VkDeviceMemory memory;
  uint32_t *map;
};

static struct buffer buffer_create(VkPhysicalDevice physical, VkDevice device,
                                   VkDeviceSize size,
                                   VkBufferUsageFlags usage) {
  struct buffer b;
  VkBufferCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
  };
  CHECK(vkCreateBuffer(device, &ci, NULL, &b.buffer));
  VkMemoryRequirements req;
  VkPhysicalDeviceMemoryProperties props;
  vkGetBufferMemoryRequirements(device, b.buffer, &req);
  vkGetPhysicalDeviceMemoryProperties(physical, &props);
  unsigned type;
  const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for (type = 0; type < props.memoryTypeCount; ++type) {
    if ((req.memoryTypeBits & (1u << type)) &&
        (props.memoryTypes[type].propertyFlags & flags) == flags)
      break;
  }
  if (type == props.memoryTypeCount) {
    fprintf(stderr, "No coherent host memory\n");
    exit(2);
  }
  VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = type,
  };
  CHECK(vkAllocateMemory(device, &alloc, NULL, &b.memory));
  CHECK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
  CHECK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, (void **)&b.map));
  return b;
}

static void buffer_destroy(VkDevice device, struct buffer *b) {
  vkUnmapMemory(device, b->memory);
  vkDestroyBuffer(device, b->buffer, NULL);
  vkFreeMemory(device, b->memory, NULL);
}

static unsigned memory_type(VkPhysicalDevice p, uint32_t bits) {
  VkPhysicalDeviceMemoryProperties props;
  vkGetPhysicalDeviceMemoryProperties(p, &props);
  for (unsigned i = 0; i < props.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags &
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
      return i;
  fprintf(stderr, "No image memory type\n");
  exit(2);
}

static void submit(VkDevice device, VkQueue queue, VkCommandBuffer cmd,
                   VkFence fence) {
  CHECK(vkEndCommandBuffer(cmd));
  VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                     .commandBufferCount = 1,
                     .pCommandBuffers = &cmd};
  CHECK(vkQueueSubmit(queue, 1, &si, fence));
  CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_C(10000000000)));
  CHECK(vkResetFences(device, 1, &fence));
}

static void begin(VkCommandBuffer cmd) {
  CHECK(vkResetCommandBuffer(cmd, 0));
  VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  CHECK(vkBeginCommandBuffer(cmd, &bi));
}

int main(int argc, char **argv) {
  unsigned loops = argc > 1 ? strtoul(argv[1], NULL, 10) : 200;
  unsigned width = argc > 2 ? strtoul(argv[2], NULL, 10) : 128;
  const unsigned layers = 4;
  if (!loops || loops > 2000 || width < 2 || width > 4096 ||
      (width & (width - 1)))
    return 2;
  unsigned levels = 0;
  for (unsigned w = width; w; w >>= 1)
    ++levels;
  VkInstance instance;
  VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                           .apiVersion = VK_API_VERSION_1_3};
  VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                              .pApplicationInfo = &app};
  CHECK(vkCreateInstance(&ici, NULL, &instance));
  unsigned n = 0;
  CHECK(vkEnumeratePhysicalDevices(instance, &n, NULL));
  if (!n)
    return 2;
  VkPhysicalDevice *devices = calloc(n, sizeof(*devices));
  CHECK(vkEnumeratePhysicalDevices(instance, &n, devices));
  VkPhysicalDevice physical = devices[0];
  free(devices);
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physical, &props);
  printf("device=%s loops=%u width=%u layers=%u levels=%u\n", props.deviceName,
         loops, width, layers, levels);
  unsigned families = 0, family = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, NULL);
  VkQueueFamilyProperties *fp = calloc(families, sizeof(*fp));
  vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, fp);
  while (family < families && !(fp[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
    ++family;
  free(fp);
  if (family == families)
    return 2;
  float priority = 1;
  VkDeviceQueueCreateInfo qci = {.sType =
                                     VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                 .queueFamilyIndex = family,
                                 .queueCount = 1,
                                 .pQueuePriorities = &priority};
  VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f13};
  vkGetPhysicalDeviceFeatures2(physical, &features);
  if (!f13.dynamicRendering)
    return 77;
  f13 = (VkPhysicalDeviceVulkan13Features){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE};
  VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                            .pNext = &f13,
                            .queueCreateInfoCount = 1,
                            .pQueueCreateInfos = &qci};
  VkDevice device;
  CHECK(vkCreateDevice(physical, &dci, NULL, &device));
  VkQueue queue;
  vkGetDeviceQueue(device, family, 0, &queue);
  const VkFormat format = VK_FORMAT_R16G16_SINT;
  VkImageCreateInfo imci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                            .imageType = VK_IMAGE_TYPE_1D,
                            .format = format,
                            .extent = {width, 1, 1},
                            .mipLevels = levels,
                            .arrayLayers = layers,
                            .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
  VkImage image;
  CHECK(vkCreateImage(device, &imci, NULL, &image));
  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(device, image, &req);
  VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = req.size,
                                .memoryTypeIndex =
                                    memory_type(physical, req.memoryTypeBits)};
  VkDeviceMemory memory;
  CHECK(vkAllocateMemory(device, &alloc, NULL, &memory));
  CHECK(vkBindImageMemory(device, image, memory, 0));
  VkImageView *views = calloc(levels * layers, sizeof(*views));
  for (unsigned m = 0; m < levels; ++m)
    for (unsigned l = 0; l < layers; ++l) {
      VkImageViewCreateInfo vi = {
          .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
          .image = image,
          .viewType = VK_IMAGE_VIEW_TYPE_1D,
          .format = format,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, m, 1, l, 1}};
      CHECK(vkCreateImageView(device, &vi, NULL, &views[m * layers + l]));
    }
  VkShaderModule shaders[2];
  VkShaderModuleCreateInfo sci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(mipmap_vert),
      .pCode = mipmap_vert};
  CHECK(vkCreateShaderModule(device, &sci, NULL, &shaders[0]));
  sci.codeSize = sizeof(mipmap_frag);
  sci.pCode = mipmap_frag;
  CHECK(vkCreateShaderModule(device, &sci, NULL, &shaders[1]));
  VkPipelineShaderStageCreateInfo stages[] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = shaders[0],
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = shaders[1],
       .pName = "main"}};
  VkPushConstantRange pc = {.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                            .size = 16};
  VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pc};
  VkPipelineLayout layout;
  CHECK(vkCreatePipelineLayout(device, &lci, NULL, &layout));
  VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
  VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1};
  VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .lineWidth = 1};
  VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
  VkPipelineColorBlendAttachmentState attachment = {.colorWriteMask = 15};
  VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &attachment};
  VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT,
                              VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dynamic};
  VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &format};
  VkGraphicsPipelineCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pColorBlendState = &cb,
      .pDynamicState = &ds,
      .layout = layout};
  VkPipeline pipeline;
  CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pci, NULL,
                                  &pipeline));
  VkBufferImageCopy *copies = calloc(levels, sizeof(*copies));
  VkDeviceSize bytes = 0;
  for (unsigned m = 0; m < levels; ++m) {
    copies[m] = (VkBufferImageCopy){
        .bufferOffset = bytes,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, layers},
        .imageExtent = {width >> m, 1, 1}};
    bytes += (width >> m) * layers * 4;
  }
  struct buffer readback =
      buffer_create(physical, device, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  VkCommandPoolCreateInfo pool_ci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = family};
  VkCommandPool pool;
  CHECK(vkCreateCommandPool(device, &pool_ci, NULL, &pool));
  VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
  VkCommandBuffer cmd;
  CHECK(vkAllocateCommandBuffers(device, &ai, &cmd));
  VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  CHECK(vkCreateFence(device, &fci, NULL, &fence));
  for (unsigned iteration = 0; iteration < loops; ++iteration) {
    for (unsigned m = 0; m < levels; ++m) {
      begin(cmd);
      if (m == 0) {
        VkImageMemoryBarrier bar = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0,
                                 layers}};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                             0, NULL, 0, NULL, 1, &bar);
      }
      unsigned w = width >> m;
      VkViewport viewport = {0, 0, w, 1, 0, 1};
      VkRect2D scissor = {{w / 2, 0}, {w - w / 2, 1}};
      for (unsigned l = 0; l < layers; ++l) {
        int32_t value[4] = {(int32_t)(iteration * 13 + l * 17 + m + 1),
                            -(int32_t)(iteration + l * 19 + m + 3), 0, 1};
        VkRenderingAttachmentInfo att = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = views[m * layers + l],
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue.color.int32 = {-101, 107, 0, 1}};
        VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                              .renderArea = {{0, 0}, {w, 1}},
                              .layerCount = 1,
                              .colorAttachmentCount = 1,
                              .pColorAttachments = &att};
        vkCmdBeginRendering(cmd, &ri);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16,
                           value);
        vkCmdDraw(cmd, 3, 1, l * 3, 0);
        vkCmdEndRendering(cmd);
      }
      submit(device, queue, cmd, fence);
    }
    begin(cmd);
    VkImageMemoryBarrier bar = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers}};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
                         &bar);
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.buffer, levels, copies);
    VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                            .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0,
                         NULL);
    submit(device, queue, cmd, fence);
    for (unsigned m = 0; m < levels; ++m) {
      unsigned w = width >> m;
      int16_t *data =
          (int16_t *)((uint8_t *)readback.map + copies[m].bufferOffset);
      for (unsigned l = 0; l < layers; ++l)
        for (unsigned x = 0; x < w; ++x) {
          int16_t expected[2] = {-101, 107};
          if (x >= w / 2) {
            expected[0] = iteration * 13 + l * 17 + m + 1;
            expected[1] = -(int)(iteration + l * 19 + m + 3);
          }
          if (memcmp(data + (l * w + x) * 2, expected, sizeof(expected))) {
            fprintf(stderr,
                    "FAIL iteration=%u mip=%u layer=%u x=%u expected=(%d,%d) "
                    "actual=(%d,%d)\n",
                    iteration, m, l, x, expected[0], expected[1],
                    data[(l * w + x) * 2], data[(l * w + x) * 2 + 1]);
            return 1;
          }
        }
    }
  }
  printf("PASS: %u iterations, every pixel of %u mips and %u layers checked\n",
         loops, levels, layers);
  vkDestroyFence(device, fence, NULL);
  vkDestroyCommandPool(device, pool, NULL);
  buffer_destroy(device, &readback);
  vkDestroyPipeline(device, pipeline, NULL);
  vkDestroyPipelineLayout(device, layout, NULL);
  vkDestroyShaderModule(device, shaders[0], NULL);
  vkDestroyShaderModule(device, shaders[1], NULL);
  for (unsigned i = 0; i < levels * layers; ++i)
    vkDestroyImageView(device, views[i], NULL);
  free(views);
  free(copies);
  vkDestroyImage(device, image, NULL);
  vkFreeMemory(device, memory, NULL);
  vkDestroyDevice(device, NULL);
  vkDestroyInstance(instance, NULL);
  return 0;
}
