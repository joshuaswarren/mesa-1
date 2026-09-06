/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Render-only Vulkan submit for T8140/G17P: clear, draw one full-screen
 * triangle into a one-tile RGBA8 image, store at end of tile. No readback.
 *
 * This exists to exercise the 3D/TA launch path on its own. The image->buffer
 * copy of g17p-render-only is a *compute* dispatch, which is blocked
 * on its own separate problem, so removing it isolates whether a render
 * submit reaches the GPU and completes.
 *
 * Every wait is bounded: a launch that faults never signals its fence, and a
 * hung wait would cost a reboot. VK_TIMEOUT is reported as its own outcome
 * rather than blocking.
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
#include <unistd.h>

#include "g17p_wsi_triangle_frag.h"
#include "g17p_wsi_triangle_vert.h"

#define T8140_DEVICE_ID 0x8140
#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32
#define IMAGE_BYTES (IMAGE_WIDTH * IMAGE_HEIGHT * 4)

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
#ifdef G17P_OCCLUSION_QUERY
VK_FN(vkGetPhysicalDeviceFeatures);
#endif
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkGetPhysicalDeviceFormatProperties);
VK_FN(vkCreateDevice);
VK_FN(vkGetDeviceQueue);
VK_FN(vkCreateBuffer);
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkCreateImage);
VK_FN(vkGetImageMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkBindBufferMemory);
VK_FN(vkBindImageMemory);
VK_FN(vkMapMemory);
VK_FN(vkCreateImageView);
VK_FN(vkCreateRenderPass);
VK_FN(vkCreateShaderModule);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkCreateGraphicsPipelines);
VK_FN(vkCreateFramebuffer);
VK_FN(vkCreateCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdBeginRenderPass);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindVertexBuffers);
VK_FN(vkCmdDraw);
VK_FN(vkCmdEndRenderPass);
VK_FN(vkEndCommandBuffer);
VK_FN(vkCreateFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);
#ifdef G17P_OCCLUSION_QUERY
VK_FN(vkCreateQueryPool);
VK_FN(vkCmdResetQueryPool);
VK_FN(vkCmdBeginQuery);
VK_FN(vkCmdEndQuery);
VK_FN(vkCmdWriteTimestamp);
VK_FN(vkGetQueryPoolResults);
#endif

static void
die(const char *what)
{
   fprintf(stderr, "g17p-render-only: %s\n", what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "g17p-render-only: %s returned %d\n", what,
              result);
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

/* Non-fatal variant: returns UINT32_MAX instead of dying, so an optional
 * memory class can be probed. */
static uint32_t
try_memory_type(const VkPhysicalDeviceMemoryProperties *properties,
                uint32_t allowed, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((allowed & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }
   return UINT32_MAX;
}

static uint32_t
find_memory_type(const VkPhysicalDeviceMemoryProperties *properties,
                 uint32_t allowed, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((allowed & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }

   die("required Vulkan memory type is unavailable");
   return UINT32_MAX;
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

   const VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "g17p-render-only",
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
#ifdef G17P_OCCLUSION_QUERY
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures);
#endif
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFormatProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   if (!physical_count)
      die("Vulkan loader found no physical device");

   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      die("out of host memory");
   check(vkEnumeratePhysicalDevices(instance, &physical_count,
                                    physical_devices),
         "vkEnumeratePhysicalDevices(list)");

   VkPhysicalDevice physical = VK_NULL_HANDLE;
   uint32_t queue_family = UINT32_MAX;
   VkPhysicalDeviceProperties properties = {0};
   for (uint32_t p = 0; p < physical_count && !physical; ++p) {
      VkPhysicalDeviceProperties candidate;
      vkGetPhysicalDeviceProperties(physical_devices[p], &candidate);
      if (candidate.deviceID != T8140_DEVICE_ID)
         continue;

      uint32_t queue_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[p],
                                                &queue_count, NULL);
      VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
      if (!queues)
         die("out of host memory");
      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[p],
                                                &queue_count, queues);
      for (uint32_t q = 0; q < queue_count; ++q) {
         const VkQueueFlags required =
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT;
         if ((queues[q].queueFlags & required) == required) {
            physical = physical_devices[p];
            queue_family = q;
            properties = candidate;
            break;
         }
      }
      free(queues);
   }
   free(physical_devices);
   if (!physical)
      die("no T8140 device has a graphics-and-transfer queue");

   VkFormatProperties format_properties;
   vkGetPhysicalDeviceFormatProperties(physical, VK_FORMAT_R8G8B8A8_UNORM,
                                       &format_properties);
   const VkFormatFeatureFlags required_format =
      VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
   if ((format_properties.optimalTilingFeatures & required_format) !=
       required_format)
      die("RGBA8 optimal images lack color-attachment or transfer-src support");

   fprintf(stderr, "g17p-render-only: device %s, queue %u\n",
           properties.deviceName, queue_family);

#ifdef G17P_OCCLUSION_QUERY
   VkPhysicalDeviceFeatures physical_features;
   vkGetPhysicalDeviceFeatures(physical, &physical_features);
   if (!physical_features.occlusionQueryPrecise)
      die("T8140 does not advertise precise occlusion queries");
   if (physical_features.pipelineStatisticsQuery)
      die("T8140 unexpectedly advertises ungrounded pipeline statistics");
   if (!properties.limits.timestampComputeAndGraphics ||
       properties.limits.timestampPeriod != 1.0f)
      die("T8140 timestamp properties differ from the bounded profile");

   uint32_t query_queue_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &query_queue_count,
                                             NULL);
   VkQueueFamilyProperties *query_queues =
      calloc(query_queue_count, sizeof(*query_queues));
   if (!query_queues)
      die("out of host memory");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &query_queue_count,
                                             query_queues);
   if (queue_family >= query_queue_count ||
       query_queues[queue_family].timestampValidBits != 64)
      die("T8140 graphics queue does not expose exact 64-bit timestamps");
   free(query_queues);
#endif

   const float queue_priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
#ifdef G17P_OCCLUSION_QUERY
      .pEnabledFeatures = &(const VkPhysicalDeviceFeatures){
         .occlusionQueryPrecise = VK_TRUE,
      },
#endif
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCreateBuffer);
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkCreateImage);
   LOAD_DEVICE(device, vkGetImageMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkBindImageMemory);
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkCreateImageView);
   LOAD_DEVICE(device, vkCreateRenderPass);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkCreateGraphicsPipelines);
   LOAD_DEVICE(device, vkCreateFramebuffer);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdBeginRenderPass);
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdBindVertexBuffers);
   LOAD_DEVICE(device, vkCmdDraw);
   LOAD_DEVICE(device, vkCmdEndRenderPass);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);
#ifdef G17P_OCCLUSION_QUERY
   LOAD_DEVICE(device, vkCreateQueryPool);
   LOAD_DEVICE(device, vkCmdResetQueryPool);
   LOAD_DEVICE(device, vkCmdBeginQuery);
   LOAD_DEVICE(device, vkCmdEndQuery);
   LOAD_DEVICE(device, vkCmdWriteTimestamp);
   LOAD_DEVICE(device, vkGetQueryPoolResults);
#endif

   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);

   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   static const float vertices[][2] = {
      {-1.0f, -1.0f},
      {3.0f, -1.0f},
      {-1.0f, 3.0f},
   };
   const VkBufferCreateInfo vertex_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(vertices),
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer vertex_buffer;
   check(vkCreateBuffer(device, &vertex_info, NULL, &vertex_buffer),
         "vkCreateBuffer(vertex)");
   VkMemoryRequirements vertex_requirements;
   vkGetBufferMemoryRequirements(device, vertex_buffer, &vertex_requirements);
   const uint32_t vertex_memory_type = find_memory_type(
      &memory_properties, vertex_requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   const VkMemoryAllocateInfo vertex_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = vertex_requirements.size,
      .memoryTypeIndex = vertex_memory_type,
   };
   VkDeviceMemory vertex_memory;
   check(vkAllocateMemory(device, &vertex_memory_info, NULL, &vertex_memory),
         "vkAllocateMemory(vertex)");
   check(vkBindBufferMemory(device, vertex_buffer, vertex_memory, 0),
         "vkBindBufferMemory(vertex)");
   void *vertex_map = NULL;
   check(vkMapMemory(device, vertex_memory, 0, sizeof(vertices), 0,
                     &vertex_map),
         "vkMapMemory(vertex)");
   memcpy(vertex_map, vertices, sizeof(vertices));


   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   check(vkCreateImage(device, &image_info, NULL, &image), "vkCreateImage");
   VkMemoryRequirements image_requirements;
   vkGetImageMemoryRequirements(device, image, &image_requirements);
   /* Host-visible so the render target can be inspected WITHOUT a copy
    * command: this ICD does not implement vkCmdCopyImageToBuffer and fails
    * vkEndCommandBuffer with -8 if one is recorded. Falls back to device-local
    * if no host-visible type accepts the image. */
   uint32_t image_memory_type =
      try_memory_type(&memory_properties, image_requirements.memoryTypeBits,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   bool image_host_visible = image_memory_type != UINT32_MAX;
   if (!image_host_visible)
      image_memory_type =
         find_memory_type(&memory_properties, image_requirements.memoryTypeBits,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   fprintf(stderr, "g17p-render-only: image memory host-visible=%d type=%u\n",
           (int)image_host_visible, image_memory_type);
   const VkMemoryAllocateInfo image_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = image_requirements.size,
      .memoryTypeIndex = image_memory_type,
   };
   VkDeviceMemory image_memory;
   check(vkAllocateMemory(device, &image_memory_info, NULL, &image_memory),
         "vkAllocateMemory(image)");
   check(vkBindImageMemory(device, image, image_memory, 0),
         "vkBindImageMemory");


   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1,
         .layerCount = 1,
      },
   };
   VkImageView view;
   check(vkCreateImageView(device, &view_info, NULL, &view),
         "vkCreateImageView");

   const VkAttachmentDescription attachment = {
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
   };
   const VkAttachmentReference color_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
   };
   const VkSubpassDependency dependencies[] = {
      {
         .srcSubpass = VK_SUBPASS_EXTERNAL,
         .dstSubpass = 0,
         .srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      },
      {
         .srcSubpass = 0,
         .dstSubpass = VK_SUBPASS_EXTERNAL,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      },
   };
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 2,
      .pDependencies = dependencies,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &render_pass_info, NULL, &render_pass),
         "vkCreateRenderPass");

   const VkShaderModuleCreateInfo vert_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_wsi_triangle_vert_spv),
      .pCode = g17p_wsi_triangle_vert_spv,
   };
   const VkShaderModuleCreateInfo frag_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_wsi_triangle_frag_spv),
      .pCode = g17p_wsi_triangle_frag_spv,
   };
   VkShaderModule vert_module, frag_module;
   check(vkCreateShaderModule(device, &vert_module_info, NULL, &vert_module),
         "vkCreateShaderModule(vertex)");
   check(vkCreateShaderModule(device, &frag_module_info, NULL, &frag_module),
         "vkCreateShaderModule(fragment)");

   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pipeline_layout;
   check(vkCreatePipelineLayout(device, &layout_info, NULL, &pipeline_layout),
         "vkCreatePipelineLayout");

   const VkPipelineShaderStageCreateInfo stages[] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vert_module,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = frag_module,
         .pName = "main",
      },
   };
   const VkVertexInputBindingDescription vertex_binding = {
      .binding = 0,
      .stride = sizeof(vertices[0]),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription vertex_attribute = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &vertex_attribute,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {
      .width = IMAGE_WIDTH,
      .height = IMAGE_HEIGHT,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };
   const VkRect2D scissor = {.extent = {IMAGE_WIDTH, IMAGE_HEIGHT}};
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
   };
   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkPipelineColorBlendAttachmentState blend_attachment = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blend_attachment,
   };
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pColorBlendState = &blend,
      .layout = pipeline_layout,
      .renderPass = render_pass,
   };
   VkPipeline pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipeline),
         "vkCreateGraphicsPipelines");
   fprintf(stderr, "g17p-render-only: graphics pipeline created\n");

   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = 1,
      .pAttachments = &view,
      .width = IMAGE_WIDTH,
      .height = IMAGE_HEIGHT,
      .layers = 1,
   };
   VkFramebuffer framebuffer;
   check(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer),
         "vkCreateFramebuffer");

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool command_pool;
   check(vkCreateCommandPool(device, &pool_info, NULL, &command_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_allocate = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command_buffer;
   check(vkAllocateCommandBuffers(device, &command_allocate, &command_buffer),
         "vkAllocateCommandBuffers");
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command_buffer, &begin_info),
         "vkBeginCommandBuffer");
#ifdef G17P_OCCLUSION_QUERY
   const VkQueryPoolCreateInfo query_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_OCCLUSION,
      .queryCount = 2,
   };
   VkQueryPool occlusion_pool;
   check(vkCreateQueryPool(device, &query_info, NULL, &occlusion_pool),
         "vkCreateQueryPool(occlusion)");
   const VkQueryPoolCreateInfo timestamp_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = 2,
   };
   VkQueryPool timestamp_pool;
   check(vkCreateQueryPool(device, &timestamp_info, NULL, &timestamp_pool),
         "vkCreateQueryPool(timestamp)");
   vkCmdResetQueryPool(command_buffer, occlusion_pool, 0, 2);
   vkCmdResetQueryPool(command_buffer, timestamp_pool, 0, 2);
   vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       timestamp_pool, 0);
#endif
   const VkClearValue clear = {.color = {{0.0f, 0.0f, 0.0f, 1.0f}}};
   const VkRenderPassBeginInfo render_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {.extent = {IMAGE_WIDTH, IMAGE_HEIGHT}},
      .clearValueCount = 1,
      .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(command_buffer, &render_begin,
                        VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                     pipeline);
   const VkDeviceSize vertex_offset = 0;
   vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer,
                          &vertex_offset);
#ifdef G17P_OCCLUSION_QUERY
   vkCmdBeginQuery(command_buffer, occlusion_pool, 0,
                   VK_QUERY_CONTROL_PRECISE_BIT);
#endif
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
#ifdef G17P_OCCLUSION_QUERY
   vkCmdEndQuery(command_buffer, occlusion_pool, 0);
   /* A second, empty precise interval proves reset and post-graphics
    * availability independently of the nonzero hardware counter. */
   vkCmdBeginQuery(command_buffer, occlusion_pool, 1,
                   VK_QUERY_CONTROL_PRECISE_BIT);
   vkCmdEndQuery(command_buffer, occlusion_pool, 1);
#endif
   vkCmdEndRenderPass(command_buffer);

#ifdef G17P_OCCLUSION_QUERY
   vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                       timestamp_pool, 1);
#endif


   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
   fprintf(stderr, "g17p-render-only: command buffer recorded\n");

#ifdef G17P_OCCLUSION_QUERY
   if (getenv("G17P_QUERY_RECORD_ONLY")) {
      puts("g17p-occlusion-query: record-only PASS");
      return EXIT_SUCCESS;
   }
#endif

   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence fence;
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
   };
   check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");

   fprintf(stderr, "g17p-render-only: submitted\n");

   /* Five seconds: long enough for any real 3D kick, short enough that a
    * launch which never signals is reported rather than hanging the box. */
   const VkResult waited =
      vkWaitForFences(device, 1, &fence, true, UINT64_C(5000000000));
   if (waited == VK_TIMEOUT) {
      fprintf(stderr, "g17p-render-only: TIMEOUT - fence never signalled\n");
      return EXIT_FAILURE;
   }
   if (waited != VK_SUCCESS) {
      fprintf(stderr, "g17p-render-only: vkWaitForFences returned %d\n",
              waited);
      return EXIT_FAILURE;
   }

#ifdef G17P_OCCLUSION_QUERY
   struct query_result {
      uint64_t samples;
      uint64_t available;
   } query_results[2] = {{0}};
   check(vkGetQueryPoolResults(device, occlusion_pool, 0, 2,
                               sizeof(query_results), query_results,
                               sizeof(query_results[0]),
                               VK_QUERY_RESULT_64_BIT |
                                  VK_QUERY_RESULT_WAIT_BIT |
                                  VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
         "vkGetQueryPoolResults(occlusion)");
   fprintf(stderr,
           "g17p-render-only: occlusion samples=[%" PRIu64 ",%" PRIu64
           "] available=[%" PRIu64 ",%" PRIu64 "] expected=[%u,0]\n",
           query_results[0].samples, query_results[1].samples,
           query_results[0].available, query_results[1].available,
           IMAGE_WIDTH * IMAGE_HEIGHT);
   if (query_results[0].available != 1 || query_results[1].available != 1 ||
       query_results[0].samples != IMAGE_WIDTH * IMAGE_HEIGHT ||
       query_results[1].samples != 0)
      die("precise occlusion results differ from full and empty intervals");

   struct timestamp_result {
      uint64_t value;
      uint64_t available;
   } timestamp_results[2] = {{0}};
   check(vkGetQueryPoolResults(device, timestamp_pool, 0, 2,
                               sizeof(timestamp_results), timestamp_results,
                               sizeof(timestamp_results[0]),
                               VK_QUERY_RESULT_64_BIT |
                                  VK_QUERY_RESULT_WAIT_BIT |
                                  VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
         "vkGetQueryPoolResults(timestamp)");
   fprintf(stderr,
           "g17p-render-only: timestamps=[%" PRIu64 ",%" PRIu64
           "] available=[%" PRIu64 ",%" PRIu64 "]\n",
           timestamp_results[0].value, timestamp_results[1].value,
           timestamp_results[0].available, timestamp_results[1].available);
   if (timestamp_results[0].available != 1 ||
       timestamp_results[1].available != 1 ||
       timestamp_results[0].value == 0 ||
       timestamp_results[1].value <= timestamp_results[0].value)
      die("timestamp interval is unavailable, zero, or non-increasing");
#endif

   /* Verify PIXELS, not just the fence. The clear is opaque black and the
    * vertex data is a full-viewport triangle, so a correct render leaves NO
    * cleared pixel behind. */
   if (!image_host_visible) {
      printf("g17p-render-only: INCONCLUSIVE - fence signalled but the render "
             "target is not host-visible, so no pixel was verified\n");
      return EXIT_FAILURE;
   }
   uint8_t *pixels = NULL;
   check(vkMapMemory(device, image_memory, 0, IMAGE_BYTES, 0,
                     (void **)&pixels),
         "vkMapMemory(image)");
   unsigned exact = 0;
   uint32_t first = 0;
   for (unsigned i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; i++) {
      const uint8_t r = pixels[i * 4 + 0], g = pixels[i * 4 + 1];
      const uint8_t b = pixels[i * 4 + 2], a = pixels[i * 4 + 3];
      if (i == 0)
         first = ((uint32_t)r << 24) | ((uint32_t)g << 16) |
                 ((uint32_t)b << 8) | a;
      if (r == 0xff && g == 0x47 && b == 0x0d && a == 0xff)
         exact++;
   }
   fprintf(stderr,
           "g17p-render-only: pixels exact=%u/%u first=%#010x\n",
           exact, IMAGE_WIDTH * IMAGE_HEIGHT, first);
   for (unsigned y = 0; y < IMAGE_HEIGHT; y += 4) {
      char row[IMAGE_WIDTH / 2 + 1];
      for (unsigned x = 0; x < IMAGE_WIDTH; x += 2) {
         const uint8_t *p = &pixels[(y * IMAGE_WIDTH + x) * 4];
         row[x / 2] = (p[0] | p[1] | p[2]) ? '#' : '.';
      }
      row[IMAGE_WIDTH / 2] = 0;
      fprintf(stderr, "g17p-render-only: |%s|\n", row);
   }
   if (exact != IMAGE_WIDTH * IMAGE_HEIGHT)
      die("render target differs from exact RGBA8 draw colour");
   if (memcmp(vertex_map, vertices, sizeof(vertices)) != 0)
      die("vertex source changed during query render");

#ifdef G17P_OCCLUSION_QUERY
   sleep(3);
   struct query_result stable_queries[2] = {{0}};
   struct timestamp_result stable_timestamps[2] = {{0}};
   check(vkGetQueryPoolResults(device, occlusion_pool, 0, 2,
                               sizeof(stable_queries), stable_queries,
                               sizeof(stable_queries[0]),
                               VK_QUERY_RESULT_64_BIT |
                                  VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
         "vkGetQueryPoolResults(occlusion stable)");
   check(vkGetQueryPoolResults(device, timestamp_pool, 0, 2,
                               sizeof(stable_timestamps), stable_timestamps,
                               sizeof(stable_timestamps[0]),
                               VK_QUERY_RESULT_64_BIT |
                                  VK_QUERY_RESULT_WITH_AVAILABILITY_BIT),
         "vkGetQueryPoolResults(timestamp stable)");
   if (memcmp(stable_queries, query_results, sizeof(query_results)) != 0 ||
       memcmp(stable_timestamps, timestamp_results,
              sizeof(timestamp_results)) != 0 ||
       memcmp(vertex_map, vertices, sizeof(vertices)) != 0)
      die("query results or source changed after three seconds");
   for (unsigned i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
      const uint8_t *p = &pixels[i * 4];
      if (p[0] != 0xff || p[1] != 0x47 || p[2] != 0x0d || p[3] != 0xff)
         die("render target changed after three seconds");
   }
#endif

   printf("g17p-render-only: PASS - %ux%u RGBA8, %u/%u exact pixels%s\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, exact, IMAGE_WIDTH * IMAGE_HEIGHT,
#ifdef G17P_OCCLUSION_QUERY
          ", precise occlusion + timestamp interval stable"
#else
          ""
#endif
   );
   return EXIT_SUCCESS;
}
