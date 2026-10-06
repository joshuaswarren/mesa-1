/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT
 * Reproduce full-area NONE/NONE Z/S preservation across color-only draws.
 * A full-area NONE/NONE color pass used to store uninitialized tile depth
 * and stencil. The same partial-area test passed because it forced a reload.
 * Check exact Z/S preservation and a depth-tested color oracle for primary
 * and secondary draws, with active depth, stencil and clear controls.
 * Run explicitly; this hardware test is not registered with meson test.
 */
#define VK_NO_PROTOTYPES
#include "g17p_vulkan_none_preserve_frag.h"
#include "g17p_vulkan_none_preserve_vert.h"
#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#define W 32u
#define H 32u
#define N (W * H)
#define Z_OFF 64u
#define S_OFF (Z_OFF + N * 4)
#define C_OFF (S_OFF + N + 64)
#define BYTES (C_OFF + N * 4 + 64)
#define VK(call)                                                               \
  do {                                                                         \
    VkResult r_ = (call);                                                      \
    if (r_ != VK_SUCCESS) {                                                    \
      fprintf(stderr, "%s failed: %d, line %d\n", #call, r_, __LINE__);        \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)
static void *vulkan_loader;
static PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
static PFN_vkAllocateMemory vkAllocateMemory;
static PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
static PFN_vkBindBufferMemory vkBindBufferMemory;
static PFN_vkBindImageMemory vkBindImageMemory;
static PFN_vkCmdBeginRendering vkCmdBeginRendering;
static PFN_vkCmdBindPipeline vkCmdBindPipeline;
static PFN_vkCmdCopyImageToBuffer vkCmdCopyImageToBuffer;
static PFN_vkCmdDraw vkCmdDraw;
static PFN_vkCmdEndRendering vkCmdEndRendering;
static PFN_vkCmdExecuteCommands vkCmdExecuteCommands;
static PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
static PFN_vkCmdPushConstants vkCmdPushConstants;
static PFN_vkCmdSetScissor vkCmdSetScissor;
static PFN_vkCreateBuffer vkCreateBuffer;
static PFN_vkCreateCommandPool vkCreateCommandPool;
static PFN_vkCreateDevice vkCreateDevice;
static PFN_vkCreateFence vkCreateFence;
static PFN_vkCreateGraphicsPipelines vkCreateGraphicsPipelines;
static PFN_vkCreateImage vkCreateImage;
static PFN_vkCreateImageView vkCreateImageView;
static PFN_vkCreateInstance vkCreateInstance;
static PFN_vkCreatePipelineLayout vkCreatePipelineLayout;
static PFN_vkCreateShaderModule vkCreateShaderModule;
static PFN_vkDestroyBuffer vkDestroyBuffer;
static PFN_vkDestroyCommandPool vkDestroyCommandPool;
static PFN_vkDestroyDevice vkDestroyDevice;
static PFN_vkDestroyFence vkDestroyFence;
static PFN_vkDestroyImage vkDestroyImage;
static PFN_vkDestroyImageView vkDestroyImageView;
static PFN_vkDestroyInstance vkDestroyInstance;
static PFN_vkDestroyPipeline vkDestroyPipeline;
static PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout;
static PFN_vkDestroyShaderModule vkDestroyShaderModule;
static PFN_vkEndCommandBuffer vkEndCommandBuffer;
static PFN_vkEnumerateDeviceExtensionProperties
    vkEnumerateDeviceExtensionProperties;
static PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;
static PFN_vkFreeMemory vkFreeMemory;
static PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
static PFN_vkGetDeviceQueue vkGetDeviceQueue;
static PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements;
static PFN_vkGetPhysicalDeviceMemoryProperties
    vkGetPhysicalDeviceMemoryProperties;
static PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
static PFN_vkGetPhysicalDeviceQueueFamilyProperties
    vkGetPhysicalDeviceQueueFamilyProperties;
static PFN_vkMapMemory vkMapMemory;
static PFN_vkQueueSubmit vkQueueSubmit;
static PFN_vkResetCommandPool vkResetCommandPool;
static PFN_vkResetFences vkResetFences;
static PFN_vkUnmapMemory vkUnmapMemory;
static PFN_vkWaitForFences vkWaitForFences;

static void load_vulkan(void) {
  vulkan_loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!vulkan_loader) {
    fprintf(stderr, "Cannot load Vulkan: %s\n", dlerror());
    exit(2);
  }
#define LOAD(name)                                                             \
  do {                                                                         \
    name = (PFN_##name)dlsym(vulkan_loader, #name);                            \
    if (!name) {                                                               \
      fprintf(stderr, "Missing loader entry point %s\n", #name);               \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)
  LOAD(vkAllocateCommandBuffers);
  LOAD(vkAllocateMemory);
  LOAD(vkBeginCommandBuffer);
  LOAD(vkBindBufferMemory);
  LOAD(vkBindImageMemory);
  LOAD(vkCmdBeginRendering);
  LOAD(vkCmdBindPipeline);
  LOAD(vkCmdCopyImageToBuffer);
  LOAD(vkCmdDraw);
  LOAD(vkCmdEndRendering);
  LOAD(vkCmdExecuteCommands);
  LOAD(vkCmdPipelineBarrier);
  LOAD(vkCmdPushConstants);
  LOAD(vkCmdSetScissor);
  LOAD(vkCreateBuffer);
  LOAD(vkCreateCommandPool);
  LOAD(vkCreateDevice);
  LOAD(vkCreateFence);
  LOAD(vkCreateGraphicsPipelines);
  LOAD(vkCreateImage);
  LOAD(vkCreateImageView);
  LOAD(vkCreateInstance);
  LOAD(vkCreatePipelineLayout);
  LOAD(vkCreateShaderModule);
  LOAD(vkDestroyBuffer);
  LOAD(vkDestroyCommandPool);
  LOAD(vkDestroyDevice);
  LOAD(vkDestroyFence);
  LOAD(vkDestroyImage);
  LOAD(vkDestroyImageView);
  LOAD(vkDestroyInstance);
  LOAD(vkDestroyPipeline);
  LOAD(vkDestroyPipelineLayout);
  LOAD(vkDestroyShaderModule);
  LOAD(vkEndCommandBuffer);
  LOAD(vkEnumerateDeviceExtensionProperties);
  LOAD(vkEnumeratePhysicalDevices);
  LOAD(vkFreeMemory);
  LOAD(vkGetBufferMemoryRequirements);
  LOAD(vkGetDeviceQueue);
  LOAD(vkGetImageMemoryRequirements);
  LOAD(vkGetPhysicalDeviceMemoryProperties);
  LOAD(vkGetPhysicalDeviceProperties);
  LOAD(vkGetPhysicalDeviceQueueFamilyProperties);
  LOAD(vkMapMemory);
  LOAD(vkQueueSubmit);
  LOAD(vkResetCommandPool);
  LOAD(vkResetFences);
  LOAD(vkUnmapMemory);
  LOAD(vkWaitForFences);
#undef LOAD
}

static VkDevice dev;
static VkPhysicalDeviceMemoryProperties mem;
struct image {
  VkImage image;
  VkDeviceMemory memory;
  VkImageView view;
};
static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
  for (uint32_t i = 0; i < mem.memoryTypeCount; i++)
    if ((bits & (1u << i)) &&
        (mem.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  fprintf(stderr, "No compatible memory type flags=%x\n", flags);
  exit(2);
}
static struct image make_image(VkFormat format, VkImageUsageFlags usage,
                               VkImageAspectFlags aspects) {
  struct image o = {0};
  VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                          .imageType = VK_IMAGE_TYPE_2D,
                          .format = format,
                          .extent = {W, H, 1},
                          .mipLevels = 1,
                          .arrayLayers = 1,
                          .samples = VK_SAMPLE_COUNT_1_BIT,
                          .tiling = VK_IMAGE_TILING_OPTIMAL,
                          .usage = usage,
                          .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                          .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
  VK(vkCreateImage(dev, &ci, NULL, &o.image));
  VkMemoryRequirements req;
  vkGetImageMemoryRequirements(dev, o.image, &req);
  VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                             .allocationSize = req.size,
                             .memoryTypeIndex =
                                 memory_type(req.memoryTypeBits, 0)};
  VK(vkAllocateMemory(dev, &ai, NULL, &o.memory));
  VK(vkBindImageMemory(dev, o.image, o.memory, 0));
  VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                              .image = o.image,
                              .viewType = VK_IMAGE_VIEW_TYPE_2D,
                              .format = format,
                              .subresourceRange = {aspects, 0, 1, 0, 1}};
  VK(vkCreateImageView(dev, &vi, NULL, &o.view));
  return o;
}
static void destroy_image(struct image *o) {
  vkDestroyImageView(dev, o->view, NULL);
  vkDestroyImage(dev, o->image, NULL);
  vkFreeMemory(dev, o->memory, NULL);
}
static void barrier(VkCommandBuffer cb) {
  VkMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT};
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, NULL, 0,
                       NULL);
}
static void initialize_layout(VkCommandBuffer cb, struct image *o,
                              VkImageAspectFlags aspects) {
  VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT |
                                             VK_ACCESS_MEMORY_WRITE_BIT,
                            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .image = o->image,
                            .subresourceRange = {aspects, 0, 1, 0, 1}};
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                       1, &b);
}
static VkPipeline make_pipeline(VkPipelineLayout layout, VkShaderModule vs,
                                VkShaderModule fs, bool depth_test,
                                bool depth_write, VkCompareOp cmp, bool stencil,
                                uint32_t reference) {
  VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = vs,
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = fs,
       .pName = "main"}};
  VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
  VkViewport viewport = {0, 0, W, H, 0, 1};
  VkRect2D rect = {{0, 0}, {W, H}};
  VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &rect};
  VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1};
  VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
  VkStencilOpState ss = {.failOp = VK_STENCIL_OP_KEEP,
                         .passOp = VK_STENCIL_OP_REPLACE,
                         .depthFailOp = VK_STENCIL_OP_KEEP,
                         .compareOp = VK_COMPARE_OP_ALWAYS,
                         .compareMask = 255,
                         .writeMask = 255,
                         .reference = reference};
  VkPipelineDepthStencilStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = depth_test,
      .depthWriteEnable = depth_write,
      .depthCompareOp = cmp,
      .stencilTestEnable = stencil,
      .front = ss,
      .back = ss,
      .minDepthBounds = 0,
      .maxDepthBounds = 1};
  VkPipelineColorBlendAttachmentState ba = {.colorWriteMask = 15};
  VkPipelineColorBlendStateCreateInfo bs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &ba};
  VkDynamicState dyn = VK_DYNAMIC_STATE_SCISSOR;
  VkPipelineDynamicStateCreateInfo dy = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 1,
      .pDynamicStates = &dyn};
  VkFormat cf = VK_FORMAT_R8G8B8A8_UNORM;
  VkPipelineRenderingCreateInfo ri = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &cf,
      .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT_S8_UINT,
      .stencilAttachmentFormat = VK_FORMAT_D32_SFLOAT_S8_UINT};
  VkGraphicsPipelineCreateInfo pi = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &ri,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pDepthStencilState = &ds,
      .pColorBlendState = &bs,
      .pDynamicState = &dy,
      .layout = layout};
  VkPipeline out;
  VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi, NULL, &out));
  return out;
}
static void begin_render(VkCommandBuffer cb, struct image *color,
                         struct image *depth, VkAttachmentLoadOp load,
                         VkAttachmentStoreOp store, float clear_z,
                         uint32_t clear_s, VkRect2D area, bool secondary) {
  VkRenderingAttachmentInfo c = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = color->view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color = {{0, 0, 0, 1}}};
  VkRenderingAttachmentInfo d = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = depth->view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = load,
      .storeOp = store,
      .clearValue.depthStencil = {clear_z, clear_s}};
  VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .flags =
          secondary ? VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT : 0,
      .renderArea = area,
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &c,
      .pDepthAttachment = &d,
      .pStencilAttachment = &d};
  vkCmdBeginRendering(cb, &ri);
}
static void draw(VkCommandBuffer cb, VkPipeline p, VkPipelineLayout l, float z,
                 VkRect2D rect, unsigned repeats) {
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
  vkCmdSetScissor(cb, 0, 1, &rect);
  vkCmdPushConstants(cb, l, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &z);
  for (unsigned n = 0; n < repeats; n++)
    vkCmdDraw(cb, 3, 1, 0, 0);
}
int main(int argc, char **argv) {
  load_vulkan();
  unsigned repeats = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 0) : 1;
  if (!repeats || repeats > 4096) {
    fprintf(stderr, "usage: %s [middle_draw_repetitions 1..4096]\n", argv[0]);
    return 2;
  }
  VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                           .pApplicationName = "none-preserve",
                           .apiVersion = VK_API_VERSION_1_3};
  VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                              .pApplicationInfo = &app};
  VkInstance instance;
  VK(vkCreateInstance(&ici, NULL, &instance));
  uint32_t count = 0;
  VK(vkEnumeratePhysicalDevices(instance, &count, NULL));
  if (!count) {
    fprintf(stderr, "No Vulkan device\n");
    return 2;
  }
  VkPhysicalDevice *gpus = calloc(count, sizeof(*gpus));
  VK(vkEnumeratePhysicalDevices(instance, &count, gpus));
  VkPhysicalDevice gpu = gpus[0];
  free(gpus);
  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(gpu, &properties);
  printf("device=%s\n", properties.deviceName);
  vkGetPhysicalDeviceMemoryProperties(gpu, &mem);
  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(gpu, &qn, NULL);
  VkQueueFamilyProperties *qp = calloc(qn, sizeof(*qp));
  vkGetPhysicalDeviceQueueFamilyProperties(gpu, &qn, qp);
  uint32_t family = ~0u;
  for (uint32_t i = 0; i < qn; i++)
    if (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      family = i;
      break;
    }
  free(qp);
  if (family == ~0u)
    return 2;
  uint32_t en = 0;
  VK(vkEnumerateDeviceExtensionProperties(gpu, NULL, &en, NULL));
  VkExtensionProperties *ep = calloc(en, sizeof(*ep));
  VK(vkEnumerateDeviceExtensionProperties(gpu, NULL, &en, ep));
  const char *ext = NULL;
  for (uint32_t i = 0; i < en; i++)
    if (!strcmp(ep[i].extensionName, "VK_KHR_load_store_op_none"))
      ext = "VK_KHR_load_store_op_none";
  for (uint32_t i = 0; i < en && !ext; i++)
    if (!strcmp(ep[i].extensionName, "VK_EXT_load_store_op_none"))
      ext = "VK_EXT_load_store_op_none";
  free(ep);
  if (!ext) {
    fprintf(stderr, "load_store_op_none unavailable\n");
    return 2;
  }
  VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .dynamicRendering = VK_TRUE};
  float priority = 1;
  VkDeviceQueueCreateInfo qi = {.sType =
                                    VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                .queueFamilyIndex = family,
                                .queueCount = 1,
                                .pQueuePriorities = &priority};
  VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                            .pNext = &f13,
                            .queueCreateInfoCount = 1,
                            .pQueueCreateInfos = &qi,
                            .enabledExtensionCount = 1,
                            .ppEnabledExtensionNames = &ext};
  VK(vkCreateDevice(gpu, &dci, NULL, &dev));
  VkQueue queue;
  vkGetDeviceQueue(dev, family, 0, &queue);
  struct image color = make_image(VK_FORMAT_R8G8B8A8_UNORM,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                  VK_IMAGE_ASPECT_COLOR_BIT);
  struct image depth =
      make_image(VK_FORMAT_D32_SFLOAT_S8_UINT,
                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                 VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
  VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                           .size = BYTES,
                           .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
  VkBuffer buffer;
  VK(vkCreateBuffer(dev, &bi, NULL, &buffer));
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(dev, buffer, &req);
  VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = memory_type(req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
  VkDeviceMemory memory;
  VK(vkAllocateMemory(dev, &ai, NULL, &memory));
  VK(vkBindBufferMemory(dev, buffer, memory, 0));
  uint8_t *mapped;
  VK(vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, (void **)&mapped));
  VkShaderModuleCreateInfo sm = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(none_vert),
      .pCode = none_vert};
  VkShaderModule vs, fs;
  VK(vkCreateShaderModule(dev, &sm, NULL, &vs));
  sm.codeSize = sizeof(none_frag);
  sm.pCode = none_frag;
  VK(vkCreateShaderModule(dev, &sm, NULL, &fs));
  VkPushConstantRange push = {VK_SHADER_STAGE_VERTEX_BIT, 0, 4};
  VkPipelineLayoutCreateInfo li = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push};
  VkPipelineLayout layout;
  VK(vkCreatePipelineLayout(dev, &li, NULL, &layout));
  VkPipeline initial =
      make_pipeline(layout, vs, fs, true, true, VK_COMPARE_OP_ALWAYS, true, 42);
  VkPipeline untouched = make_pipeline(layout, vs, fs, false, false,
                                       VK_COMPARE_OP_ALWAYS, false, 0);
  VkPipeline write_z =
      make_pipeline(layout, vs, fs, true, true, VK_COMPARE_OP_ALWAYS, false, 0);
  VkPipeline write_s = make_pipeline(layout, vs, fs, false, false,
                                     VK_COMPARE_OP_ALWAYS, true, 91);
  VkPipeline probe = make_pipeline(layout, vs, fs, true, false,
                                   VK_COMPARE_OP_GREATER, false, 0);
  VkCommandPoolCreateInfo pci = {.sType =
                                     VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                 .queueFamilyIndex = family};
  VkCommandPool pool;
  VK(vkCreateCommandPool(dev, &pci, NULL, &pool));
  VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
  VkCommandBuffer primary, secondary;
  VK(vkAllocateCommandBuffers(dev, &cai, &primary));
  cai.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
  VK(vkAllocateCommandBuffers(dev, &cai, &secondary));
  VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  VK(vkCreateFence(dev, &fi, NULL, &fence));
  VkRect2D full = {{0, 0}, {W, H}}, left = {{0, 0}, {W / 2, H}};
  unsigned failed = 0, passed = 0;
  for (unsigned mode = 0; mode < 4; mode++)
    for (unsigned partial = 0; partial < 2; partial++)
      for (unsigned sec = 0; sec < 2; sec++) {
        memset(mapped, 0xa5, BYTES);
        VK(vkResetCommandPool(dev, pool, 0));
        VkCommandBufferBeginInfo begin = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        VK(vkBeginCommandBuffer(primary, &begin));
        initialize_layout(primary, &color, VK_IMAGE_ASPECT_COLOR_BIT);
        initialize_layout(primary, &depth,
                          VK_IMAGE_ASPECT_DEPTH_BIT |
                              VK_IMAGE_ASPECT_STENCIL_BIT);
        begin_render(primary, &color, &depth, VK_ATTACHMENT_LOAD_OP_CLEAR,
                     VK_ATTACHMENT_STORE_OP_STORE, .75f, 17, full, false);
        draw(primary, initial, layout, .25f, left, 1);
        vkCmdEndRendering(primary);
        barrier(primary);
        VkRect2D area = {{0, 0}, {partial ? 27 : W, partial ? 19 : H}};
        VkPipeline middle = mode == 1   ? write_z
                            : mode == 2 ? write_s
                                        : untouched;
        if (sec) {
          VkFormat cf = VK_FORMAT_R8G8B8A8_UNORM;
          VkCommandBufferInheritanceRenderingInfo ri = {
              .sType =
                  VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO,
              .colorAttachmentCount = 1,
              .pColorAttachmentFormats = &cf,
              .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT_S8_UINT,
              .stencilAttachmentFormat = VK_FORMAT_D32_SFLOAT_S8_UINT,
              .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
          VkCommandBufferInheritanceInfo inh = {
              .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
              .pNext = &ri};
          VkCommandBufferBeginInfo sb = {
              .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
              .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT |
                       VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
              .pInheritanceInfo = &inh};
          VK(vkBeginCommandBuffer(secondary, &sb));
          draw(secondary, middle, layout, .5f, area, repeats);
          VK(vkEndCommandBuffer(secondary));
        }
        begin_render(primary, &color, &depth,
                     mode == 0   ? VK_ATTACHMENT_LOAD_OP_NONE
                     : mode == 3 ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                 : VK_ATTACHMENT_LOAD_OP_LOAD,
                     mode == 0 ? VK_ATTACHMENT_STORE_OP_NONE
                               : VK_ATTACHMENT_STORE_OP_STORE,
                     .625f, 51, area, sec);
        if (sec)
          vkCmdExecuteCommands(primary, 1, &secondary);
        else
          draw(primary, middle, layout, .5f, area, repeats);
        vkCmdEndRendering(primary);
        barrier(primary);
        VkBufferImageCopy copy = {
            .bufferOffset = Z_OFF,
            .imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1},
            .imageExtent = {W, H, 1}};
        vkCmdCopyImageToBuffer(primary, depth.image, VK_IMAGE_LAYOUT_GENERAL,
                               buffer, 1, &copy);
        copy.bufferOffset = S_OFF;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
        vkCmdCopyImageToBuffer(primary, depth.image, VK_IMAGE_LAYOUT_GENERAL,
                               buffer, 1, &copy);
        barrier(primary);
        begin_render(primary, &color, &depth, VK_ATTACHMENT_LOAD_OP_LOAD,
                     VK_ATTACHMENT_STORE_OP_STORE, 0, 0, full, false);
        draw(primary, probe, layout, .5f, full, 1);
        vkCmdEndRendering(primary);
        barrier(primary);
        copy.bufferOffset = C_OFF;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vkCmdCopyImageToBuffer(primary, color.image, VK_IMAGE_LAYOUT_GENERAL,
                               buffer, 1, &copy);
        VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL,
                             0, NULL);
        VK(vkEndCommandBuffer(primary));
        VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                               .commandBufferCount = 1,
                               .pCommandBuffers = &primary};
        VK(vkQueueSubmit(queue, 1, &submit, fence));
        VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 30000000000ull));
        VK(vkResetFences(dev, 1, &fence));
        unsigned bad_z = 0, bad_s = 0, bad_c = 0, bad_guard = 0;
        float first_z = 0;
        unsigned first_x = 0, first_y = 0;
        for (unsigned y = 0; y < H; y++)
          for (unsigned x = 0; x < W; x++) {
            bool inside = x < area.extent.width && y < area.extent.height;
            float expected_z = x < W / 2 ? .25f : .75f;
            uint8_t expected_s = x < W / 2 ? 42 : 17;
            if (inside && mode == 1)
              expected_z = .5f;
            if (inside && mode == 2)
              expected_s = 91;
            if (inside && mode == 3) {
              expected_z = .625f;
              expected_s = 51;
            }
            float got;
            memcpy(&got, mapped + Z_OFF + 4 * (y * W + x), 4);
            if (got != expected_z) {
              if (!bad_z) {
                first_z = got;
                first_x = x;
                first_y = y;
              }
              bad_z++;
            }
            bad_s += mapped[S_OFF + y * W + x] != expected_s;
            uint8_t expected_c[4] = {expected_z < .5f ? 255 : 0, 0, 0, 255};
            bad_c +=
                memcmp(mapped + C_OFF + 4 * (y * W + x), expected_c, 4) != 0;
          }
        for (unsigned i = 0; i < BYTES; i++)
          if (i < Z_OFF || (i >= S_OFF + N && i < C_OFF) || i >= C_OFF + N * 4)
            bad_guard += mapped[i] != 0xa5;
        bool ok = !bad_z && !bad_s && !bad_c && !bad_guard;
        printf(
            "%s mode=%s area=%s commands=%s repeat=%u bad_depth=%u "
            "bad_stencil=%u bad_color=%u bad_guard=%u first_depth=%g@%u,%u\n",
            ok ? "PASS" : "FAIL",
            mode == 0   ? "none-none"
            : mode == 1 ? "depth-write"
            : mode == 2 ? "stencil-write"
                        : "clear",
            partial ? "27x19" : "32x32", sec ? "secondary" : "primary", repeats,
            bad_z, bad_s, bad_c, bad_guard, first_z, first_x, first_y);
        fflush(stdout);
        passed += ok;
        failed += !ok;
      }
  printf("RESULT %u pass %u fail\n", passed, failed);
  vkDestroyFence(dev, fence, NULL);
  vkDestroyCommandPool(dev, pool, NULL);
  vkDestroyPipeline(dev, initial, NULL);
  vkDestroyPipeline(dev, untouched, NULL);
  vkDestroyPipeline(dev, write_z, NULL);
  vkDestroyPipeline(dev, write_s, NULL);
  vkDestroyPipeline(dev, probe, NULL);
  vkDestroyPipelineLayout(dev, layout, NULL);
  vkDestroyShaderModule(dev, vs, NULL);
  vkDestroyShaderModule(dev, fs, NULL);
  vkUnmapMemory(dev, memory);
  vkDestroyBuffer(dev, buffer, NULL);
  vkFreeMemory(dev, memory, NULL);
  destroy_image(&color);
  destroy_image(&depth);
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(instance, NULL);
  dlclose(vulkan_loader);
  return failed ? 1 : 0;
}
