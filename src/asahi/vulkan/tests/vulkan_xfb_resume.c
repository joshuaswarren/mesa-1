/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Verify vertex data and XFB counter resumption across pipeline changes.
 * Four line topologies exercise both software vertex and geometry stages;
 * alternating direct/indirect draws also exercise the internal grid helper.
 */
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "vulkan_xfb_resume_shaders.h"

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r = (call);                                                     \
      if (r != VK_SUCCESS) {                                                   \
         fprintf(stderr, "%s: %d\n", #call, r);                                \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

struct buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint32_t *map;
};

static struct buffer
buffer_create(VkPhysicalDevice physical, VkDevice device, VkDeviceSize size,
              VkBufferUsageFlags usage)
{
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

static void
buffer_destroy(VkDevice device, struct buffer *b)
{
   vkUnmapMemory(device, b->memory);
   vkDestroyBuffer(device, b->buffer, NULL);
   vkFreeMemory(device, b->memory, NULL);
}

static unsigned
verify(uint32_t *map, unsigned offset, VkPrimitiveTopology topology,
       unsigned count, unsigned first, uint32_t tag)
{
   const bool adjacency =
       topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY ||
       topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
   const bool strip =
       topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP ||
       topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
   unsigned primitives =
       strip ? count - (adjacency ? 3 : 1) : count / (adjacency ? 4 : 2);
   for (unsigned p = 0; p < primitives; ++p) {
      unsigned base = first + (strip ? p : p * (adjacency ? 4 : 2)) + adjacency;
      for (unsigned v = 0; v < 2; ++v) {
         uint32_t id = base + v;
         uint32_t expected[4] = {id, id ^ tag, (id * 7) + tag, ~id};
         for (unsigned c = 0; c < 4; ++c) {
            unsigned index = (offset + p * 2 + v) * 4 + c;
            if (map[index] != expected[c]) {
               fprintf(
                   stderr,
                   "topology=%u vertex=%u component=%u expected=%u actual=%u\n",
                   topology, offset + p * 2 + v, c, expected[c], map[index]);
               exit(1);
            }
         }
      }
   }
   return primitives * 2;
}

int
main(int argc, char **argv)
{
   unsigned loops = argc > 1 ? strtoul(argv[1], NULL, 10) : 200;
   unsigned vertices_per_draw = argc > 2 ? strtoul(argv[2], NULL, 10) : 128;
   if (!loops || loops > 2000 || vertices_per_draw < 4 ||
       vertices_per_draw > 65536 || vertices_per_draw % 4) {
      fprintf(
          stderr,
          "Expected 1..2000 loops and 4..65536 vertices (multiple of four)\n");
      return 2;
   }
   unsigned second_count = vertices_per_draw * 3 / 4;
   const VkDeviceSize output_size =
       (VkDeviceSize)(vertices_per_draw + second_count) * 32 + 64;
   VkInstance instance;
   VkApplicationInfo app = {
       .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
       .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo ici = {
       .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
       .pApplicationInfo = &app,
   };
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
   printf("device=%s loops=%u vertices=%u\n", props.deviceName, loops,
          vertices_per_draw);

   VkPhysicalDeviceProvokingVertexFeaturesEXT provoking = {
       .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT,
   };
   VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb = {
       .sType =
           VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
       .pNext = &provoking,
   };
   VkPhysicalDeviceFeatures2 features = {
       .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
       .pNext = &xfb,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   if (!xfb.transformFeedback ||
       !provoking.provokingVertexLast ||
       !provoking.transformFeedbackPreservesProvokingVertex) {
      puts("SKIP: provoking-vertex XFB unavailable");
      vkDestroyInstance(instance, NULL);
      return 77;
   }
   features.features = (VkPhysicalDeviceFeatures){
       .geometryShader = features.features.geometryShader};
   xfb.geometryStreams = VK_FALSE;
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
   VkDeviceQueueCreateInfo qci = {
       .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
       .queueFamilyIndex = family,
       .queueCount = 1,
       .pQueuePriorities = &priority,
   };
   const char *extensions[] = {VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME,
                               VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME};
   VkDeviceCreateInfo dci = {
       .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
       .pNext = &features,
       .queueCreateInfoCount = 1,
       .pQueueCreateInfos = &qci,
       .enabledExtensionCount = 2,
       .ppEnabledExtensionNames = extensions,
   };
   VkDevice device;
   CHECK(vkCreateDevice(physical, &dci, NULL, &device));
   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   PFN_vkCmdBindTransformFeedbackBuffersEXT bind_xfb =
       (void *)vkGetDeviceProcAddr(device,
                                   "vkCmdBindTransformFeedbackBuffersEXT");
   PFN_vkCmdBeginTransformFeedbackEXT begin_xfb =
       (void *)vkGetDeviceProcAddr(device, "vkCmdBeginTransformFeedbackEXT");
   PFN_vkCmdEndTransformFeedbackEXT end_xfb =
       (void *)vkGetDeviceProcAddr(device, "vkCmdEndTransformFeedbackEXT");
   if (!bind_xfb || !begin_xfb || !end_xfb)
      return 2;
   VkShaderModule shaders[2];
   VkShaderModuleCreateInfo sci = {
       .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
       .codeSize = sizeof(xfb_vert),
       .pCode = xfb_vert,
   };
   CHECK(vkCreateShaderModule(device, &sci, NULL, &shaders[0]));
   sci.codeSize = sizeof(xfb_frag);
   sci.pCode = xfb_frag;
   CHECK(vkCreateShaderModule(device, &sci, NULL, &shaders[1]));
   VkPipelineShaderStageCreateInfo stages[] = {
       {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = shaders[0],
        .pName = "main"},
       {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = shaders[1],
        .pName = "main"},
   };
   VkPushConstantRange pc = {.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                             .size = 4};
   VkPipelineLayoutCreateInfo lci = {
       .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
       .pushConstantRangeCount = 1,
       .pPushConstantRanges = &pc};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(device, &lci, NULL, &layout));
   VkVertexInputBindingDescription binding = {
       .binding = 0, .stride = 16, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
   VkVertexInputAttributeDescription attribute = {
       .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32A32_UINT};
   VkPipelineVertexInputStateCreateInfo vi = {
       .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
       .vertexBindingDescriptionCount = 1,
       .pVertexBindingDescriptions = &binding,
       .vertexAttributeDescriptionCount = 1,
       .pVertexAttributeDescriptions = &attribute};
   VkPipelineInputAssemblyStateCreateInfo ia = {
       .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
   VkViewport viewport = {0, 0, 1, 1, 0, 1};
   VkRect2D scissor = {{0, 0}, {1, 1}};
   VkPipelineViewportStateCreateInfo vp = {
       .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
       .viewportCount = 1,
       .pViewports = &viewport,
       .scissorCount = 1,
       .pScissors = &scissor};
   VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pv = {
       .sType =
           VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT};
   VkPipelineRasterizationStateCreateInfo rs = {
       .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
       .pNext = &pv,
       .rasterizerDiscardEnable = VK_FALSE,
       .polygonMode = VK_POLYGON_MODE_FILL,
       .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {
       .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
       .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
   VkPipelineColorBlendStateCreateInfo cb = {
       .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
   VkSubpassDescription subpass = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS};
   VkSubpassDependency dependency = {
       .srcSubpass = 0, .dstSubpass = 0,
       .srcStageMask = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
       .dstStageMask = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
       .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
       .dstAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT};
   VkRenderPassCreateInfo render_pass_info = {
       .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
       .subpassCount = 1, .pSubpasses = &subpass,
       .dependencyCount = 1, .pDependencies = &dependency};
   VkRenderPass render_pass;
   CHECK(vkCreateRenderPass(device, &render_pass_info, NULL, &render_pass));
   VkFramebufferCreateInfo framebuffer_info = {
       .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
       .renderPass = render_pass, .width = 1, .height = 1, .layers = 1};
   VkFramebuffer framebuffer;
   CHECK(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer));
   VkGraphicsPipelineCreateInfo pci = {
       .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
       .renderPass = render_pass,
       .stageCount = 2,
       .pStages = stages,
       .pVertexInputState = &vi,
       .pInputAssemblyState = &ia,
       .pViewportState = &vp,
       .pRasterizationState = &rs,
       .pMultisampleState = &ms,
       .pColorBlendState = &cb,
       .layout = layout,
   };
   VkPrimitiveTopology topologies[] = {
       VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
       VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
       VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY};
   VkPipeline pipelines[4][2];
   for (unsigned t = 0; t < 4; ++t) {
      ia.topology = topologies[t];
      for (unsigned d = 0; d < 2; ++d) {
         pv.provokingVertexMode = d ? VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT
                                    : VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT;
         CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pci, NULL,
                                         &pipelines[t][d]));
      }
   }
   struct buffer input =
       buffer_create(physical, device, (VkDeviceSize)vertices_per_draw * 3 * 16,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
   for (unsigned v = 0; v < vertices_per_draw * 3; ++v) {
      input.map[v * 4] = v;
      input.map[v * 4 + 1] = v * 7;
      input.map[v * 4 + 2] = ~v;
      input.map[v * 4 + 3] = UINT32_C(0xbadc0ffe);
   }
   struct buffer output =
       buffer_create(physical, device, output_size,
                     VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT);
   struct buffer counter =
       buffer_create(physical, device, 16,
                     VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT);
   struct buffer indirect =
       buffer_create(physical, device, 2 * sizeof(VkDrawIndirectCommand),
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
   const VkDrawIndirectCommand draws[] = {
       {vertices_per_draw, 1, 0, 0},
       {second_count, 1, vertices_per_draw * 2, 0}};
   memcpy(indirect.map, draws, sizeof(draws));
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
   for (unsigned i = 0; i < loops; ++i) {
      for (unsigned t = 0; t < 4; ++t) {
         memset(output.map, 0xcd, output_size);
         memset(counter.map, 0, 16);
         CHECK(vkResetCommandBuffer(cmd, 0));
         VkCommandBufferBeginInfo bi = {
             .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
         CHECK(vkBeginCommandBuffer(cmd, &bi));
         VkRenderPassBeginInfo ri = {
             .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
             .renderPass = render_pass, .framebuffer = framebuffer,
             .renderArea = {{0, 0}, {1, 1}}};
         vkCmdBeginRenderPass(cmd, &ri, VK_SUBPASS_CONTENTS_INLINE);
         VkDeviceSize offset = 0, size = output_size;
         vkCmdBindVertexBuffers(cmd, 0, 1, &input.buffer, &offset);
         bind_xfb(cmd, 0, 1, &output.buffer, &offset, &size);
         uint32_t tags[2] = {i * 17 + 3, i * 17 + 11};
         for (unsigned d = 0; d < 2; ++d) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipelines[t][d]);
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4,
                               &tags[d]);
            begin_xfb(cmd, 0, 1, &counter.buffer, &offset);
            if ((i + d) & 1)
               vkCmdDrawIndirect(cmd, indirect.buffer, d * sizeof(draws[0]), 1,
                                 sizeof(draws[0]));
            else
               vkCmdDraw(cmd, draws[d].vertexCount, 1, draws[d].firstVertex, 0);
            end_xfb(cmd, 0, 1, &counter.buffer, &offset);
            VkMemoryBarrier barrier = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask =
                    VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
                .dstAccessMask =
                    VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT};
            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                 VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                 0, 1, &barrier, 0, NULL, 0, NULL);
         }
         vkCmdEndRenderPass(cmd);
         VkMemoryBarrier host = {
             .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
             .srcAccessMask =
                 VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT |
                 VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
             .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
         vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                              VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL,
                              0, NULL);
         CHECK(vkEndCommandBuffer(cmd));
         VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                .commandBufferCount = 1,
                                .pCommandBuffers = &cmd};
         CHECK(vkQueueSubmit(queue, 1, &submit, fence));
         CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE,
                               UINT64_C(10000000000)));
         CHECK(vkResetFences(device, 1, &fence));
         unsigned vertices =
             verify(output.map, 0, topologies[t], draws[0].vertexCount,
                    draws[0].firstVertex, tags[0]);
         vertices +=
             verify(output.map, vertices, topologies[t], draws[1].vertexCount,
                    draws[1].firstVertex, tags[1]);
         if (counter.map[0] != vertices * 16) {
            fprintf(stderr, "Counter expected=%u actual=%u\n", vertices * 16,
                    counter.map[0]);
            return 1;
         }
         if (output.map[vertices * 4] != UINT32_C(0xcdcdcdcd)) {
            fprintf(stderr, "Write beyond XFB count\n");
            return 1;
         }
      }
   }
   printf("PASS: %u topology/counter-resume checks\n", loops * 4);
   vkDestroyFence(device, fence, NULL);
   vkDestroyCommandPool(device, pool, NULL);
   buffer_destroy(device, &indirect);
   buffer_destroy(device, &input);
   buffer_destroy(device, &counter);
   buffer_destroy(device, &output);
   for (unsigned t = 0; t < 4; ++t)
      for (unsigned d = 0; d < 2; ++d)
         vkDestroyPipeline(device, pipelines[t][d], NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyFramebuffer(device, framebuffer, NULL);
   vkDestroyRenderPass(device, render_pass, NULL);
   for (unsigned j = 0; j < 2; ++j)
      vkDestroyShaderModule(device, shaders[j], NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return 0;
}
