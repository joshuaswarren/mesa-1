/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Records a SuperTuxKart-shaped frame: many small render passes, each one
 * clearing and storing its attachment around a single draw, all into one
 * command buffer.
 *
 * Every one of those passes composes a background program, an end-of-tile
 * program and their partial-render twins, and every draw composes a vertex and
 * a fragment Ioto program. Each is a couple of hundred bytes. Given a BO of its
 * own, each also costs a 16KiB page plus a 16KiB guard page of the 4GiB USC
 * exec window -- over a hundred times the program -- and a frame of this shape
 * exhausts the window:
 *
 *   MESA: warning: ../src/asahi/vulkan/hk_cmd_draw.c:325:
 *     VK_ERROR_OUT_OF_DEVICE_MEMORY
 *   supertuxkart: vk_command_buffer.c:105: vk_command_buffer_end:
 *     Assertion `command_buffer->state == MESA_VK_COMMAND_BUFFER_STATE_RECORDING'
 *     failed.
 *
 * Recording alone reaches every allocation, so this runs against drm-shim with
 * no GPU. The shim counts the executable-window mappings the recording
 * establishes and holds them to ASAHI_SHIM_G17P_MAX_USC_BINDS; the budget, not
 * this file, is what fails when programs go back to one BO each.
 * --execute additionally submits the recording on real hardware.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "g17p_wsi_triangle_frag.h"
#include "g17p_wsi_triangle_vert.h"
#include "g17p_constant_data_frag.h"

#define T8140_DEVICE_ID 0x8140

/* Enough passes that a per-program BO consumes tens of megabytes of the
 * window, while the whole recording still takes well under a second. */
#define RENDER_PASSES 512
#define DRAWS_PER_PASS 2
#define VERTICES 3
#define VERTEX_STRIDE (2 * sizeof(float))

#define VK_FN(name) static PFN_##name name
VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
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
VK_FN(vkCreateImage);
VK_FN(vkDestroyImage);
VK_FN(vkGetImageMemoryRequirements);
VK_FN(vkBindImageMemory);
VK_FN(vkCreateImageView);
VK_FN(vkDestroyImageView);
VK_FN(vkCreateShaderModule);
VK_FN(vkDestroyShaderModule);
VK_FN(vkCreateRenderPass);
VK_FN(vkDestroyRenderPass);
VK_FN(vkCreateFramebuffer);
VK_FN(vkDestroyFramebuffer);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkDestroyPipelineLayout);
VK_FN(vkCreateGraphicsPipelines);
VK_FN(vkDestroyPipeline);
VK_FN(vkCreateCommandPool);
VK_FN(vkDestroyCommandPool);
VK_FN(vkResetCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkEndCommandBuffer);
VK_FN(vkCmdBeginRenderPass);
VK_FN(vkCmdEndRenderPass);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindVertexBuffers);
VK_FN(vkCmdSetViewport);
VK_FN(vkCmdSetScissor);
VK_FN(vkCmdDraw);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);

static const char *const NAME = "g17p-vulkan-ioto-program-pressure";

static uint64_t
monotonic_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static void
die(const char *message)
{
   fprintf(stderr, "%s: %s\n", NAME, message);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "%s: %s returned %d\n", NAME, what, result);
      exit(EXIT_FAILURE);
   }
}

#define LOAD_GLOBAL(name)                                                      \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);         \
      if (!name)                                                               \
         die("loader is missing " #name);                                      \
   } while (0)

#define LOAD_INSTANCE(instance, name)                                          \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);               \
      if (!name)                                                               \
         die("instance is missing " #name);                                    \
   } while (0)

static uint32_t
memory_type(const VkPhysicalDeviceMemoryProperties *properties, uint32_t bits,
            VkMemoryPropertyFlags need)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & need) == need)
         return i;
   }
   die("no suitable memory type");
   return 0;
}

int
main(int argc, char **argv)
{
   const bool execute = argc == 2 && !strcmp(argv[1], "--execute");
   if (argc > 2 || (argc == 2 && !execute))
      die("usage: g17p-vulkan-ioto-program-pressure [--execute]");

   unsigned record_rounds = 1;
   const char *rounds_env = getenv("G17P_RECORD_ROUNDS");
   if (rounds_env) {
      char *end = NULL;
      const unsigned long parsed = strtoul(rounds_env, &end, 10);
      if (!rounds_env[0] || *end || parsed == 0 || parsed > UINT_MAX)
         die("G17P_RECORD_ROUNDS must be an integer in 1..UINT_MAX");
      record_rounds = parsed;
   }
   if (execute && record_rounds != 1)
      die("G17P_RECORD_ROUNDS is recording-only; omit --execute");

   unsigned redundant_pipeline_binds = 0;
   const char *binds_env = getenv("G17P_REDUNDANT_PIPELINE_BINDS");
   if (binds_env) {
      char *end = NULL;
      const unsigned long parsed = strtoul(binds_env, &end, 10);
      if (!binds_env[0] || *end || parsed > UINT_MAX)
         die("G17P_REDUNDANT_PIPELINE_BINDS must be an integer in 0..UINT_MAX");
      redundant_pipeline_binds = parsed;
   }
   const bool alternate_pipelines =
      getenv("G17P_ALTERNATE_PIPELINES") != NULL;
   const bool alternate_pipelines_per_draw =
      getenv("G17P_ALTERNATE_PIPELINES_PER_DRAW") != NULL;
   unsigned draws_per_pass = DRAWS_PER_PASS;
   const char *draws_env = getenv("G17P_DRAWS_PER_PASS");
   if (draws_env) {
      char *end = NULL;
      const unsigned long parsed = strtoul(draws_env, &end, 10);
      if (!draws_env[0] || *end || parsed == 0 || parsed > UINT_MAX)
         die("G17P_DRAWS_PER_PASS must be an integer in 1..UINT_MAX");
      draws_per_pass = parsed;
   }
   const unsigned color_attachment_count =
      getenv("G17P_SPILL_ATTACHMENTS") != NULL ? 8 : 1;
   const VkFormat color_format = color_attachment_count > 1
      ? VK_FORMAT_R32G32B32A32_SFLOAT
      : VK_FORMAT_R8G8B8A8_UNORM;
   const bool constant_data_shader =
      getenv("G17P_CONSTANT_DATA_SHADER") != NULL;

   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());
   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("libvulkan.so.1 has no vkGetInstanceProcAddr");
   LOAD_GLOBAL(vkCreateInstance);

   const VkApplicationInfo application = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = NAME,
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &application,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance), "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t device_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &device_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   VkPhysicalDevice *devices = calloc(device_count, sizeof(*devices));
   if (!devices)
      die("out of host memory");
   check(vkEnumeratePhysicalDevices(instance, &device_count, devices),
         "vkEnumeratePhysicalDevices(list)");

   VkPhysicalDevice physical = VK_NULL_HANDLE;
   uint32_t queue_family = UINT32_MAX;
   for (uint32_t d = 0; d < device_count && !physical; ++d) {
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(devices[d], &properties);
      if (properties.deviceID != T8140_DEVICE_ID)
         continue;
      uint32_t count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(devices[d], &count, NULL);
      VkQueueFamilyProperties *queues = calloc(count, sizeof(*queues));
      if (!queues)
         die("out of host memory");
      vkGetPhysicalDeviceQueueFamilyProperties(devices[d], &count, queues);
      for (uint32_t q = 0; q < count; ++q) {
         if (queues[q].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            physical = devices[d];
            queue_family = q;
            break;
         }
      }
      free(queues);
   }
   free(devices);
   if (!physical)
      die("no T8140/G17P graphics device");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

#define LOAD(name)                                                             \
   do {                                                                        \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                   \
      if (!name)                                                               \
         die("device is missing " #name);                                      \
   } while (0)
   LOAD(vkDestroyDevice); LOAD(vkGetDeviceQueue);
   LOAD(vkCreateBuffer); LOAD(vkDestroyBuffer);
   LOAD(vkGetBufferMemoryRequirements); LOAD(vkAllocateMemory);
   LOAD(vkFreeMemory); LOAD(vkBindBufferMemory); LOAD(vkMapMemory);
   LOAD(vkUnmapMemory); LOAD(vkCreateImage); LOAD(vkDestroyImage);
   LOAD(vkGetImageMemoryRequirements); LOAD(vkBindImageMemory);
   LOAD(vkCreateImageView); LOAD(vkDestroyImageView);
   LOAD(vkCreateShaderModule); LOAD(vkDestroyShaderModule);
   LOAD(vkCreateRenderPass); LOAD(vkDestroyRenderPass);
   LOAD(vkCreateFramebuffer); LOAD(vkDestroyFramebuffer);
   LOAD(vkCreatePipelineLayout); LOAD(vkDestroyPipelineLayout);
   LOAD(vkCreateGraphicsPipelines); LOAD(vkDestroyPipeline);
   LOAD(vkCreateCommandPool); LOAD(vkDestroyCommandPool);
   LOAD(vkResetCommandPool);
   LOAD(vkAllocateCommandBuffers); LOAD(vkBeginCommandBuffer);
   LOAD(vkEndCommandBuffer); LOAD(vkCmdBeginRenderPass);
   LOAD(vkCmdEndRenderPass); LOAD(vkCmdBindPipeline);
   LOAD(vkCmdBindVertexBuffers); LOAD(vkCmdSetViewport);
   LOAD(vkCmdSetScissor); LOAD(vkCmdDraw);
   LOAD(vkCreateFence); LOAD(vkDestroyFence);
   LOAD(vkQueueSubmit); LOAD(vkWaitForFences);
#undef LOAD

   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);
   VkPhysicalDeviceMemoryProperties memory;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory);

   const VkDeviceSize vertex_bytes = VERTICES * VERTEX_STRIDE;
   const VkBufferCreateInfo vertex_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = vertex_bytes,
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
   };
   VkBuffer vertex_buffer;
   check(vkCreateBuffer(device, &vertex_buffer_info, NULL, &vertex_buffer),
         "vkCreateBuffer");
   VkMemoryRequirements vertex_requirements;
   vkGetBufferMemoryRequirements(device, vertex_buffer, &vertex_requirements);
   const VkMemoryAllocateInfo vertex_alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = vertex_requirements.size,
      .memoryTypeIndex =
         memory_type(&memory, vertex_requirements.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory vertex_memory;
   check(vkAllocateMemory(device, &vertex_alloc, NULL, &vertex_memory),
         "vkAllocateMemory(vertices)");
   check(vkBindBufferMemory(device, vertex_buffer, vertex_memory, 0),
         "vkBindBufferMemory");
   void *vertex_map;
   check(vkMapMemory(device, vertex_memory, 0, vertex_bytes, 0, &vertex_map),
         "vkMapMemory");
   static const float positions[VERTICES][2] = {
      {-1.0f, -1.0f}, {3.0f, -1.0f}, {-1.0f, 3.0f},
   };
   memcpy(vertex_map, positions, sizeof(positions));
   vkUnmapMemory(device, vertex_memory);

   const VkImageCreateInfo colour_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = color_format,
      .extent = {64, 64, 1},
      .mipLevels = 1, .arrayLayers = color_attachment_count,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage colour_image;
   check(vkCreateImage(device, &colour_info, NULL, &colour_image),
         "vkCreateImage(colour)");
   VkMemoryRequirements colour_requirements;
   vkGetImageMemoryRequirements(device, colour_image, &colour_requirements);
   const VkMemoryAllocateInfo colour_alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = colour_requirements.size,
      .memoryTypeIndex = memory_type(&memory,
                                     colour_requirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory colour_memory;
   check(vkAllocateMemory(device, &colour_alloc, NULL, &colour_memory),
         "vkAllocateMemory(colour)");
   check(vkBindImageMemory(device, colour_image, colour_memory, 0),
         "vkBindImageMemory(colour)");
   VkImageView colour_views[8] = {0};
   for (unsigned i = 0; i < color_attachment_count; ++i) {
      const VkImageViewCreateInfo colour_view_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = colour_image,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = color_format,
         .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseArrayLayer = i,
            .levelCount = 1, .layerCount = 1,
         },
      };
      check(vkCreateImageView(device, &colour_view_info, NULL,
                              &colour_views[i]),
            "vkCreateImageView(colour)");
   }

   /* Clear on entry and store on exit, so each pass composes a background
    * clear program and an end-of-tile store program rather than the empty
    * pair a DONT_CARE attachment gets. */
   VkAttachmentDescription attachments[8];
   VkAttachmentReference colour_references[8];
   for (unsigned i = 0; i < color_attachment_count; ++i) {
      attachments[i] = (VkAttachmentDescription){
         .format = color_format,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      };
      colour_references[i] = (VkAttachmentReference){
         .attachment = i,
         .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      };
   }
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = color_attachment_count,
      .pColorAttachments = colour_references,
   };
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .subpassCount = 1, .pSubpasses = &subpass,
      .attachmentCount = color_attachment_count, .pAttachments = attachments,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &render_pass_info, NULL, &render_pass),
         "vkCreateRenderPass");
   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = color_attachment_count, .pAttachments = colour_views,
      .width = 64, .height = 64, .layers = 1,
   };
   VkFramebuffer framebuffer;
   check(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer),
         "vkCreateFramebuffer");

   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout layout;
   check(vkCreatePipelineLayout(device, &layout_info, NULL, &layout),
         "vkCreatePipelineLayout");

   const VkShaderModuleCreateInfo vertex_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_wsi_triangle_vert_spv),
      .pCode = g17p_wsi_triangle_vert_spv,
   };
   VkShaderModule vertex_module;
   check(vkCreateShaderModule(device, &vertex_module_info, NULL,
                              &vertex_module),
         "vkCreateShaderModule(vertex)");
   const VkShaderModuleCreateInfo fragment_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = constant_data_shader ? sizeof(g17p_constant_data_frag_spv)
                                       : sizeof(g17p_wsi_triangle_frag_spv),
      .pCode = constant_data_shader ? g17p_constant_data_frag_spv
                                    : g17p_wsi_triangle_frag_spv,
   };
   VkShaderModule fragment_module;
   check(vkCreateShaderModule(device, &fragment_module_info, NULL,
                              &fragment_module),
         "vkCreateShaderModule(fragment)");

   const VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vertex_module, .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fragment_module, .pName = "main",
      },
   };
   const VkVertexInputBindingDescription vertex_binding = {
      .binding = 0, .stride = VERTEX_STRIDE,
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription attribute = {
      .location = 0, .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
      .offset = 0,
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &attribute,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkPipelineViewportStateCreateInfo viewport = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .scissorCount = 1,
   };
   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState blend_attachments[8] = {0};
   for (unsigned i = 0; i < color_attachment_count; ++i) {
      blend_attachments[i].colorWriteMask =
         VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
   }
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = color_attachment_count,
      .pAttachments = blend_attachments,
   };
   const VkDynamicState dynamic_states[2] = {
      VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
   };
   const VkPipelineDynamicStateCreateInfo dynamic = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dynamic_states,
   };
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pColorBlendState = &blend,
      .pDynamicState = &dynamic,
      .layout = layout, .renderPass = render_pass,
   };
   VkPipeline pipelines[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipelines[0]),
         "vkCreateGraphicsPipelines");
   if (alternate_pipelines || alternate_pipelines_per_draw) {
      check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL, &pipelines[1]),
            "vkCreateGraphicsPipelines(alternate)");
   }

   const VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool command_pool;
   check(vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_alloc = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command;
   check(vkAllocateCommandBuffers(device, &command_alloc, &command),
         "vkAllocateCommandBuffers");

   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   const uint64_t record_start_ns = monotonic_ns();
   VkClearValue clears[8] = {0};
   for (unsigned i = 0; i < color_attachment_count; ++i)
      clears[i].color.float32[3] = 1.0f;
   const VkDeviceSize vertex_offset = 0;
   for (unsigned round = 0; round < record_rounds; ++round) {
      check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");

      for (unsigned pass = 0; pass < RENDER_PASSES; ++pass) {
         const VkRenderPassBeginInfo pass_begin = {
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = render_pass,
            .framebuffer = framebuffer,
            .renderArea = {.extent = {64, 64}},
            .clearValueCount = color_attachment_count,
            .pClearValues = clears,
         };
         vkCmdBeginRenderPass(command, &pass_begin, VK_SUBPASS_CONTENTS_INLINE);
         VkPipeline pipeline =
            pipelines[alternate_pipelines ? (pass & 1) : 0];
         vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
         for (unsigned bind = 0; bind < redundant_pipeline_binds; ++bind)
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
         vkCmdBindVertexBuffers(command, 0, 1, &vertex_buffer, &vertex_offset);

         for (unsigned draw = 0; draw < draws_per_pass; ++draw) {
            if (alternate_pipelines_per_draw) {
               pipeline = pipelines[draw & 1];
               vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline);
            }
            /* Move the viewport every draw. A frame that binds identical state
             * for every draw would let the driver skip the per-draw program
             * upload entirely, which is not the shape that ran out of window. */
            const VkViewport view = {
               .x = (float)(draw & 1),
               .width = 64.0f, .height = 64.0f, .maxDepth = 1.0f,
            };
            const VkRect2D scissor = {.extent = {64, 64}};
            vkCmdSetViewport(command, 0, 1, &view);
            vkCmdSetScissor(command, 0, 1, &scissor);
            vkCmdDraw(command, VERTICES, 1, 0, 0);
         }

         vkCmdEndRenderPass(command);
      }

      const VkResult recorded = vkEndCommandBuffer(command);
      if (recorded != VK_SUCCESS) {
         fprintf(stderr,
                 "%s: vkEndCommandBuffer returned %d after %u render passes "
                 "and %u draws in round %u/%u\n",
                 NAME, recorded, RENDER_PASSES,
                 RENDER_PASSES * draws_per_pass, round + 1, record_rounds);
         return EXIT_FAILURE;
      }

      if (round + 1 != record_rounds)
         check(vkResetCommandPool(device, command_pool, 0),
               "vkResetCommandPool");
   }
   const uint64_t record_end_ns = monotonic_ns();

   if (execute) {
      const VkFenceCreateInfo fence_info = {
         .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      };
      VkFence fence;
      check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");
      const VkSubmitInfo submit = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &command,
      };
      check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
      check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences");
      vkDestroyFence(device, fence, NULL);
   }

   vkDestroyCommandPool(device, command_pool, NULL);
   if (pipelines[1] != VK_NULL_HANDLE)
      vkDestroyPipeline(device, pipelines[1], NULL);
   vkDestroyPipeline(device, pipelines[0], NULL);
   vkDestroyShaderModule(device, fragment_module, NULL);
   vkDestroyShaderModule(device, vertex_module, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyFramebuffer(device, framebuffer, NULL);
   vkDestroyRenderPass(device, render_pass, NULL);
   for (unsigned i = 0; i < color_attachment_count; ++i)
      vkDestroyImageView(device, colour_views[i], NULL);
   vkDestroyImage(device, colour_image, NULL);
   vkFreeMemory(device, colour_memory, NULL);
   vkDestroyBuffer(device, vertex_buffer, NULL);
   vkFreeMemory(device, vertex_memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);

   printf("G17P IOTO PROGRAM PRESSURE PASS: %u rounds, %u render passes and "
          "%u draws per command buffer; total_record_us=%" PRIu64
          ", record_us_per_round=%" PRIu64 "\n",
          record_rounds, RENDER_PASSES, RENDER_PASSES * draws_per_pass,
          (record_end_ns - record_start_ns) / 1000,
          (record_end_ns - record_start_ns) / 1000 / record_rounds);
   return EXIT_SUCCESS;
}
