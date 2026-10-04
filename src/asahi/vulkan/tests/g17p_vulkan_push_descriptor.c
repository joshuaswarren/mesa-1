/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public record-only VK_KHR_push_descriptor oracle for G17P. It covers
 * allocated and pushed descriptor sets with combined and separate sampled
 * image/sampler bindings, including command-buffer reset and replacement.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "g17p_vulkan_array_lod_explicit_frag.h"
#include "g17p_vulkan_array_lod_implicit_frag.h"
#include "g17p_vulkan_push_descriptor_array_combined_frag.h"
#include "g17p_vulkan_push_descriptor_array_separate_frag.h"
#include "g17p_vulkan_push_descriptor_combined_frag.h"
#include "g17p_vulkan_push_descriptor_pixels_vert.h"
#include "g17p_vulkan_push_descriptor_separate_frag.h"

#ifndef G17P_ARRAY_CONTROL
#define G17P_ARRAY_CONTROL 0
#endif
#ifndef G17P_ARRAY_LOD_CONTROL
#define G17P_ARRAY_LOD_CONTROL 0
#endif

#if G17P_ARRAY_LOD_CONTROL
#define TEST_NAME       "g17p-vulkan-array-lod-control"
#define OUTPUT_CASES    4
#elif G17P_ARRAY_CONTROL
#define TEST_NAME       "g17p-vulkan-push-descriptor-array-control"
#define OUTPUT_CASES    2
#else
#define TEST_NAME       "g17p-vulkan-push-descriptor"
#define OUTPUT_CASES    5
#endif
#define T8140_DEVICE_ID 0x8140
#define SOURCE_WIDTH    4
#define SOURCE_HEIGHT   4
#define OUTPUT_WIDTH    32
#define OUTPUT_HEIGHT   32
#define GUARD_SIZE      64
#define OUTPUT_BYTES    (OUTPUT_WIDTH * OUTPUT_HEIGHT * 4)
#define OUTPUT_STRIDE   (GUARD_SIZE + OUTPUT_BYTES + GUARD_SIZE)
#define READBACK_BYTES  (OUTPUT_CASES * OUTPUT_STRIDE)

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceProperties2);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkEnumerateDeviceExtensionProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkCreateImage);
VK_FN(vkDestroyImage);
VK_FN(vkGetImageMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkFreeMemory);
VK_FN(vkBindImageMemory);
VK_FN(vkBindBufferMemory);
VK_FN(vkCreateImageView);
VK_FN(vkDestroyImageView);
VK_FN(vkCreateSampler);
VK_FN(vkDestroySampler);
VK_FN(vkCreateDescriptorSetLayout);
VK_FN(vkDestroyDescriptorSetLayout);
VK_FN(vkGetDescriptorSetLayoutSupport);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkDestroyPipelineLayout);
VK_FN(vkCreateDescriptorPool);
VK_FN(vkDestroyDescriptorPool);
VK_FN(vkAllocateDescriptorSets);
VK_FN(vkUpdateDescriptorSets);
VK_FN(vkCreateCommandPool);
VK_FN(vkDestroyCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkEndCommandBuffer);
VK_FN(vkResetCommandBuffer);
VK_FN(vkCmdBindDescriptorSets);
VK_FN(vkCmdPushDescriptorSetKHR);
VK_FN(vkCmdPipelineBarrier);
VK_FN(vkCmdBlitImage);
VK_FN(vkCreateBuffer);
VK_FN(vkDestroyBuffer);
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkMapMemory);
VK_FN(vkUnmapMemory);
VK_FN(vkFlushMappedMemoryRanges);
VK_FN(vkInvalidateMappedMemoryRanges);
VK_FN(vkCreateShaderModule);
VK_FN(vkDestroyShaderModule);
VK_FN(vkCreateGraphicsPipelines);
VK_FN(vkDestroyPipeline);
VK_FN(vkGetDeviceQueue);
VK_FN(vkCmdCopyBufferToImage);
VK_FN(vkCmdCopyImageToBuffer);
VK_FN(vkCmdBeginRendering);
VK_FN(vkCmdEndRendering);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdSetViewport);
VK_FN(vkCmdSetScissor);
VK_FN(vkCmdDraw);
VK_FN(vkCmdPushConstants);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);

struct image_resource {
   VkImage image;
   VkDeviceMemory memory;
   VkImageView view;
};

struct buffer_resource {
   VkBuffer buffer;
   VkDeviceMemory memory;
   void *map;
};

static void
die(const char *what)
{
   fprintf(stderr, TEST_NAME ": %s\n", what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, TEST_NAME ": %s returned %d\n", what, result);
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

static bool
has_extension(VkPhysicalDevice physical, const char *name)
{
   uint32_t count = 0;
   check(vkEnumerateDeviceExtensionProperties(physical, NULL, &count, NULL),
         "vkEnumerateDeviceExtensionProperties(count)");
   VkExtensionProperties *extensions = calloc(count, sizeof(*extensions));
   if (!extensions)
      die("out of host memory");
   check(vkEnumerateDeviceExtensionProperties(physical, NULL, &count,
                                               extensions),
         "vkEnumerateDeviceExtensionProperties(list)");
   bool found = false;
   for (uint32_t i = 0; i < count; ++i)
      found |= !strcmp(extensions[i].extensionName, name);
   free(extensions);
   return found;
}

static uint32_t
choose_memory_type(VkPhysicalDevice physical, uint32_t memory_type_bits)
{
   VkPhysicalDeviceMemoryProperties properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &properties);
   for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      if (memory_type_bits & (1u << i))
         return i;
   }
   die("image has no compatible memory type");
   return UINT32_MAX;
}

static uint32_t
choose_host_memory_type(VkPhysicalDevice physical, uint32_t memory_type_bits)
{
   VkPhysicalDeviceMemoryProperties properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &properties);
   const VkMemoryPropertyFlags required =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
      if ((memory_type_bits & (1u << i)) &&
          (properties.memoryTypes[i].propertyFlags & required) == required)
         return i;
   }
   die("buffer has no coherent host-visible memory type");
   return UINT32_MAX;
}

static struct image_resource
create_image_resource(VkPhysicalDevice physical, VkDevice device,
                      uint32_t width, uint32_t height, VkImageUsageFlags usage,
                      bool create_view)
{
   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {width, height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   struct image_resource resource = {0};
   check(vkCreateImage(device, &image_info, NULL, &resource.image),
         "vkCreateImage(resource)");

   VkMemoryRequirements requirements;
   vkGetImageMemoryRequirements(device, resource.image, &requirements);
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex =
         choose_memory_type(physical, requirements.memoryTypeBits),
   };
   check(vkAllocateMemory(device, &memory_info, NULL, &resource.memory),
         "vkAllocateMemory(image resource)");
   check(vkBindImageMemory(device, resource.image, resource.memory, 0),
         "vkBindImageMemory(resource)");

   if (create_view) {
      const VkImageViewCreateInfo view_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = resource.image,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
         },
      };
      check(vkCreateImageView(device, &view_info, NULL, &resource.view),
            "vkCreateImageView(resource)");
   }
   return resource;
}

static void
destroy_image_resource(VkDevice device, struct image_resource *resource)
{
   if (resource->view)
      vkDestroyImageView(device, resource->view, NULL);
   vkDestroyImage(device, resource->image, NULL);
   vkFreeMemory(device, resource->memory, NULL);
}

static struct buffer_resource
create_buffer_resource(VkPhysicalDevice physical, VkDevice device,
                       VkDeviceSize size, VkBufferUsageFlags usage)
{
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   struct buffer_resource resource = {0};
   check(vkCreateBuffer(device, &buffer_info, NULL, &resource.buffer),
         "vkCreateBuffer(resource)");

   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, resource.buffer, &requirements);
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex =
         choose_host_memory_type(physical, requirements.memoryTypeBits),
   };
   check(vkAllocateMemory(device, &memory_info, NULL, &resource.memory),
         "vkAllocateMemory(buffer resource)");
   check(vkBindBufferMemory(device, resource.buffer, resource.memory, 0),
         "vkBindBufferMemory(resource)");
   check(vkMapMemory(device, resource.memory, 0, VK_WHOLE_SIZE, 0,
                     &resource.map),
         "vkMapMemory(resource)");
   return resource;
}

static void
destroy_buffer_resource(VkDevice device, struct buffer_resource *resource)
{
   vkUnmapMemory(device, resource->memory);
   vkDestroyBuffer(device, resource->buffer, NULL);
   vkFreeMemory(device, resource->memory, NULL);
}

static VkShaderModule
create_shader(VkDevice device, const uint32_t *code, size_t code_size)
{
   const VkShaderModuleCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code_size,
      .pCode = code,
   };
   VkShaderModule shader;
   check(vkCreateShaderModule(device, &info, NULL, &shader),
         "vkCreateShaderModule");
   return shader;
}

static VkPipeline
create_graphics_pipeline(VkDevice device, VkPipelineLayout layout,
                         VkShaderModule vertex, VkShaderModule fragment)
{
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vertex,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fragment,
         .pName = "main",
      },
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkPipelineViewportStateCreateInfo viewport = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
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
   const VkPipelineColorBlendAttachmentState attachment = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &attachment,
   };
   const VkDynamicState dynamic_states[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
   };
   const VkPipelineDynamicStateCreateInfo dynamic = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2,
      .pDynamicStates = dynamic_states,
   };
   const VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats =
         (VkFormat[]){VK_FORMAT_R8G8B8A8_UNORM},
   };
   const VkGraphicsPipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pColorBlendState = &blend,
      .pDynamicState = &dynamic,
      .layout = layout,
   };
   VkPipeline pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, NULL,
                                   &pipeline),
         "vkCreateGraphicsPipelines");
   return pipeline;
}

static VkDescriptorSetLayout
create_layout(VkDevice device, VkDescriptorSetLayoutCreateFlags flags,
              const VkDescriptorSetLayoutBinding *bindings,
              uint32_t binding_count, const char *what)
{
   const VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = flags,
      .bindingCount = binding_count,
      .pBindings = bindings,
   };
   VkDescriptorSetLayoutSupport support = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT,
   };
   vkGetDescriptorSetLayoutSupport(device, &info, &support);
   if (!support.supported)
      die(what);

   VkDescriptorSetLayout layout;
   check(vkCreateDescriptorSetLayout(device, &info, NULL, &layout),
         "vkCreateDescriptorSetLayout");
   return layout;
}

static VkPipelineLayout
create_pipeline_layout_ext(VkDevice device, VkDescriptorSetLayout set_layout,
                           bool layer_push_constant)
{
   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      .offset = 0,
      .size = sizeof(int32_t),
   };
   const VkPipelineLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
      .pushConstantRangeCount = layer_push_constant ? 1 : 0,
      .pPushConstantRanges = layer_push_constant ? &push_range : NULL,
   };
   VkPipelineLayout layout;
   check(vkCreatePipelineLayout(device, &info, NULL, &layout),
         "vkCreatePipelineLayout");
   return layout;
}

static VkPipelineLayout
create_pipeline_layout(VkDevice device, VkDescriptorSetLayout set_layout)
{
   return create_pipeline_layout_ext(device, set_layout, false);
}

static void
record_matrix(VkCommandBuffer command_buffer,
              VkPipelineLayout allocated_combined_pipeline_layout,
              VkPipelineLayout allocated_separate_pipeline_layout,
              VkPipelineLayout pushed_combined_pipeline_layout,
              VkPipelineLayout pushed_separate_pipeline_layout,
              VkDescriptorSet combined_set, VkDescriptorSet separate_set,
              const VkWriteDescriptorSet *combined_write,
              const VkWriteDescriptorSet separate_writes[2], bool reverse)
{
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command_buffer, &begin),
         "vkBeginCommandBuffer");

   if (!reverse) {
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              allocated_combined_pipeline_layout, 0, 1,
                              &combined_set, 0, NULL);
      vkCmdPushDescriptorSetKHR(command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pushed_combined_pipeline_layout, 0, 1,
                                combined_write);
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              allocated_separate_pipeline_layout, 0, 1,
                              &separate_set, 0, NULL);
      vkCmdPushDescriptorSetKHR(command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pushed_separate_pipeline_layout, 0, 2,
                                separate_writes);
   } else {
      vkCmdPushDescriptorSetKHR(command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pushed_separate_pipeline_layout, 0, 2,
                                separate_writes);
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              allocated_separate_pipeline_layout, 0, 1,
                              &separate_set, 0, NULL);
      vkCmdPushDescriptorSetKHR(command_buffer,
                                VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pushed_combined_pipeline_layout, 0, 1,
                                combined_write);
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              allocated_combined_pipeline_layout, 0, 1,
                              &combined_set, 0, NULL);
   }

   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
}

static void
record_meta_blit(VkCommandBuffer command_buffer, VkImage source,
                 VkImage destination)
{
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command_buffer, &begin),
         "vkBeginCommandBuffer(meta blit)");

   const VkImageMemoryBarrier barriers[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = source,
         .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
         },
      },
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = destination,
         .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
         },
      },
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        2, barriers);

   const VkImageBlit region = {
      .srcSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
      .srcOffsets = {{0, 0, 0}, {4, 4, 1}},
      .dstSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
      .dstOffsets = {{0, 0, 0}, {OUTPUT_WIDTH, OUTPUT_HEIGHT, 1}},
   };
   vkCmdBlitImage(command_buffer, source,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region,
                  VK_FILTER_NEAREST);
   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer(meta blit)");
}

static void
image_barrier(VkCommandBuffer command_buffer, VkImage image,
              VkAccessFlags src_access, VkAccessFlags dst_access,
              VkImageLayout old_layout, VkImageLayout new_layout,
              VkPipelineStageFlags src_stage,
              VkPipelineStageFlags dst_stage)
{
   const VkImageMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = src_access,
      .dstAccessMask = dst_access,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .baseMipLevel = 0,
         .levelCount = 1,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
   };
   vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, NULL, 0,
                        NULL, 1, &barrier);
}

static void
record_pixel_matrix(VkCommandBuffer command_buffer,
                    struct buffer_resource *upload,
                    struct buffer_resource *readback,
                    struct image_resource *source,
                    struct image_resource outputs[OUTPUT_CASES],
                    const VkPipeline pipelines[4],
                    const VkPipelineLayout pipeline_layouts[4],
                    const VkDescriptorSet descriptor_sets[2],
                    const VkWriteDescriptorSet *pushed_combined_write,
                    const VkWriteDescriptorSet pushed_separate_writes[2])
{
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command_buffer, &begin),
         "vkBeginCommandBuffer(pixel matrix)");

   image_barrier(command_buffer, source->image, 0,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
   const VkBufferImageCopy upload_region = {
      .bufferOffset = GUARD_SIZE,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
      .imageOffset = {0, 0, 0},
      .imageExtent = {SOURCE_WIDTH, SOURCE_HEIGHT, 1},
   };
   vkCmdCopyBufferToImage(command_buffer, upload->buffer, source->image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                          &upload_region);
   image_barrier(command_buffer, source->image,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

   const VkViewport viewport = {
      .x = 0,
      .y = 0,
      .width = OUTPUT_WIDTH,
      .height = OUTPUT_HEIGHT,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };
   const VkRect2D scissor = {
      .offset = {0, 0},
      .extent = {OUTPUT_WIDTH, OUTPUT_HEIGHT},
   };
   const VkClearValue clear = {
      .color = {.float32 = {165.0f / 255.0f, 165.0f / 255.0f,
                           165.0f / 255.0f, 165.0f / 255.0f}},
   };

   const bool push_before_pipeline =
      getenv("G17P_PUSH_DESCRIPTOR_BEFORE_PIPELINE") != NULL;
   const uint32_t render_case_count = G17P_ARRAY_CONTROL ? 2 : 4;
   for (uint32_t i = 0; i < render_case_count; ++i) {
      image_barrier(command_buffer, outputs[i].image, 0,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
      const VkRenderingAttachmentInfo color_attachment = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = outputs[i].view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue = clear,
      };
      const VkRenderingInfo rendering = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = scissor,
         .layerCount = 1,
         .colorAttachmentCount = 1,
         .pColorAttachments = &color_attachment,
      };
      vkCmdBeginRendering(command_buffer, &rendering);
      if (!push_before_pipeline || i < 2)
         vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipelines[i]);
      if (i == 0) {
         vkCmdBindDescriptorSets(command_buffer,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline_layouts[i], 0, 1,
                                 &descriptor_sets[0], 0, NULL);
      } else if (i == 1) {
         vkCmdBindDescriptorSets(command_buffer,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 pipeline_layouts[i], 0, 1,
                                 &descriptor_sets[1], 0, NULL);
      } else if (i == 2) {
         vkCmdPushDescriptorSetKHR(command_buffer,
                                   VK_PIPELINE_BIND_POINT_GRAPHICS,
                                   pipeline_layouts[i], 0, 1,
                                   pushed_combined_write);
      } else {
         vkCmdPushDescriptorSetKHR(command_buffer,
                                   VK_PIPELINE_BIND_POINT_GRAPHICS,
                                   pipeline_layouts[i], 0, 2,
                                   pushed_separate_writes);
      }
      /* A pushed set may precede the pipeline using it. */
      if (push_before_pipeline && i >= 2)
         vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipelines[i]);
      vkCmdSetViewport(command_buffer, 0, 1, &viewport);
      vkCmdSetScissor(command_buffer, 0, 1, &scissor);
      vkCmdDraw(command_buffer, 3, 1, 0, 0);
      vkCmdEndRendering(command_buffer);

      image_barrier(command_buffer, outputs[i].image,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT);
      const VkBufferImageCopy readback_region = {
         .bufferOffset = i * OUTPUT_STRIDE + GUARD_SIZE,
         .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
         },
         .imageOffset = {0, 0, 0},
         .imageExtent = {OUTPUT_WIDTH, OUTPUT_HEIGHT, 1},
      };
      vkCmdCopyImageToBuffer(command_buffer, outputs[i].image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readback->buffer, 1, &readback_region);
   }

#if !G17P_ARRAY_CONTROL && !G17P_ARRAY_LOD_CONTROL
   image_barrier(command_buffer, source->image, VK_ACCESS_SHADER_READ_BIT,
                 VK_ACCESS_TRANSFER_READ_BIT,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
   image_barrier(command_buffer, outputs[4].image, 0,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
   const VkImageBlit blit = {
      .srcSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
      .srcOffsets = {{0, 0, 0}, {SOURCE_WIDTH, SOURCE_HEIGHT, 1}},
      .dstSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
      .dstOffsets = {{0, 0, 0}, {OUTPUT_WIDTH, OUTPUT_HEIGHT, 1}},
   };
   vkCmdBlitImage(command_buffer, source->image,
                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  outputs[4].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  1, &blit, VK_FILTER_NEAREST);
   image_barrier(command_buffer, outputs[4].image,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
   const VkBufferImageCopy blit_readback = {
      .bufferOffset = 4 * OUTPUT_STRIDE + GUARD_SIZE,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
      .imageOffset = {0, 0, 0},
      .imageExtent = {OUTPUT_WIDTH, OUTPUT_HEIGHT, 1},
   };
   vkCmdCopyImageToBuffer(command_buffer, outputs[4].image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback->buffer, 1, &blit_readback);
#endif

   const VkBufferMemoryBarrier host_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = readback->buffer,
      .offset = 0,
      .size = VK_WHOLE_SIZE,
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1,
                        &host_barrier, 0, NULL);
   check(vkEndCommandBuffer(command_buffer),
         "vkEndCommandBuffer(pixel matrix)");
}

static void
source_pixel(uint32_t x, uint32_t y, uint8_t out[4])
{
   out[0] = 10 + x * 53;
   out[1] = 20 + y * 47;
   out[2] = 30 + (x + y) * 29;
   out[3] = 255;
}

static void
array_lod_color(uint32_t mip, uint32_t layer, uint8_t out[4])
{
   static const uint8_t colors[2][2][4] = {
      {{31, 67, 109, 255}, {149, 43, 211, 255}},
      {{227, 173, 19, 255}, {17, 199, 137, 255}},
   };
   memcpy(out, colors[mip][layer], 4);
}

static void
record_array_lod_matrix(VkCommandBuffer command_buffer,
                        struct buffer_resource *upload,
                        struct buffer_resource *readback,
                        struct image_resource *source,
                        struct image_resource outputs[4],
                        const VkPipeline pipelines[4],
                        VkPipelineLayout pipeline_layout,
                        VkDescriptorSet descriptor_set)
{
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command_buffer, &begin),
         "vkBeginCommandBuffer(array LOD)");

   const VkImageMemoryBarrier to_transfer = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = source->image,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .baseMipLevel = 0,
         .levelCount = 2,
         .baseArrayLayer = 0,
         .layerCount = 2,
      },
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &to_transfer);

   const VkBufferImageCopy uploads[4] = {
      {
         .bufferOffset = GUARD_SIZE,
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {4, 4, 1},
      },
      {
         .bufferOffset = GUARD_SIZE + 64,
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 1},
         .imageExtent = {4, 4, 1},
      },
      {
         .bufferOffset = GUARD_SIZE + 128,
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1},
         .imageExtent = {2, 2, 1},
      },
      {
         .bufferOffset = GUARD_SIZE + 144,
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 1},
         .imageExtent = {2, 2, 1},
      },
   };
   vkCmdCopyBufferToImage(command_buffer, upload->buffer, source->image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 4, uploads);

   VkImageMemoryBarrier to_sample = to_transfer;
   to_sample.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
   to_sample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   to_sample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0,
                        NULL, 1, &to_sample);

   const VkViewport viewport = {
      .width = OUTPUT_WIDTH,
      .height = OUTPUT_HEIGHT,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };
   const VkRect2D scissor = {
      .extent = {OUTPUT_WIDTH, OUTPUT_HEIGHT},
   };
   const VkClearValue clear = {
      .color = {.float32 = {165.0f / 255.0f, 165.0f / 255.0f,
                           165.0f / 255.0f, 165.0f / 255.0f}},
   };
   for (uint32_t i = 0; i < 4; ++i) {
      image_barrier(command_buffer, outputs[i].image, 0,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
      const VkRenderingAttachmentInfo attachment = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = outputs[i].view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue = clear,
      };
      const VkRenderingInfo rendering = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = scissor,
         .layerCount = 1,
         .colorAttachmentCount = 1,
         .pColorAttachments = &attachment,
      };
      vkCmdBeginRendering(command_buffer, &rendering);
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipelines[i]);
      vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
      const int32_t layer = i & 1;
      vkCmdPushConstants(command_buffer, pipeline_layout,
                         VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(layer),
                         &layer);
      vkCmdSetViewport(command_buffer, 0, 1, &viewport);
      vkCmdSetScissor(command_buffer, 0, 1, &scissor);
      vkCmdDraw(command_buffer, 3, 1, 0, 0);
      vkCmdEndRendering(command_buffer);

      image_barrier(command_buffer, outputs[i].image,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT);
      const VkBufferImageCopy copy = {
         .bufferOffset = i * OUTPUT_STRIDE + GUARD_SIZE,
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {OUTPUT_WIDTH, OUTPUT_HEIGHT, 1},
      };
      vkCmdCopyImageToBuffer(command_buffer, outputs[i].image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readback->buffer, 1, &copy);
   }

   const VkBufferMemoryBarrier host_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = readback->buffer,
      .size = VK_WHOLE_SIZE,
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1,
                        &host_barrier, 0, NULL);
   check(vkEndCommandBuffer(command_buffer),
         "vkEndCommandBuffer(array LOD)");
}

static bool
validate_array_lod(const struct buffer_resource *upload,
                   const struct buffer_resource *readback, const char *phase)
{
   static const char *names[4] = {
      "implicit-lod-layer0", "implicit-lod-layer1",
      "explicit-lod0-layer0", "explicit-lod0-layer1",
   };
   bool all_ok = true;
   const uint8_t *upload_bytes = upload->map;
   const uint8_t *readback_bytes = readback->map;
   for (uint32_t i = 0; i < GUARD_SIZE; ++i) {
      if (upload_bytes[i] != 0xa5 || upload_bytes[GUARD_SIZE + 160 + i] != 0xa5)
         all_ok = false;
   }
   for (uint32_t c = 0; c < 4; ++c) {
      const uint8_t *slot = readback_bytes + c * OUTPUT_STRIDE;
      uint8_t expected[4];
      array_lod_color(0, c & 1, expected);
      uint32_t pixel_wrong = 0, guard_wrong = 0;
      uint8_t first[4] = {0};
      for (uint32_t i = 0; i < GUARD_SIZE; ++i) {
         if (slot[i] != 0xa5 || slot[GUARD_SIZE + OUTPUT_BYTES + i] != 0xa5)
            guard_wrong++;
      }
      for (uint32_t p = 0; p < OUTPUT_WIDTH * OUTPUT_HEIGHT; ++p) {
         const uint8_t *actual = slot + GUARD_SIZE + p * 4;
         if (memcmp(actual, expected, 4)) {
            if (pixel_wrong == 0)
               memcpy(first, actual, 4);
            pixel_wrong++;
         }
      }
      if (pixel_wrong || guard_wrong) {
         fprintf(stderr, TEST_NAME
                         ": CASE %s FAIL pixels=%u guards=%u first=%02x%02x%02x%02x "
                         "expected=%02x%02x%02x%02x phase=%s\n",
                 names[c], pixel_wrong, guard_wrong, first[0], first[1],
                 first[2], first[3], expected[0], expected[1], expected[2],
                 expected[3], phase);
         all_ok = false;
      } else {
         printf(TEST_NAME ": CASE %s PASS pixels=1024 guards=128 phase=%s\n",
                names[c], phase);
      }
   }
   return all_ok;
}

static bool
validate_pixels(const struct buffer_resource *upload,
                const struct buffer_resource *readback, const char *phase)
{
   const uint8_t *upload_bytes = upload->map;
   const uint8_t *readback_bytes = readback->map;
   bool all_ok = true;
   uint32_t upload_guard_wrong = 0;
   for (uint32_t i = 0; i < GUARD_SIZE; ++i) {
      if (upload_bytes[i] != 0xa5 ||
          upload_bytes[GUARD_SIZE + SOURCE_WIDTH * SOURCE_HEIGHT * 4 + i] !=
             0xa5)
         upload_guard_wrong++;
   }
   if (upload_guard_wrong) {
      fprintf(stderr, TEST_NAME ": source guards FAIL wrong=%u during %s\n",
              upload_guard_wrong, phase);
      all_ok = false;
   }

   static const char *case_names[OUTPUT_CASES] = {
#if G17P_ARRAY_LOD_CONTROL
      "implicit-lod-layer0", "implicit-lod-layer1",
      "explicit-lod0-layer0", "explicit-lod0-layer1",
#elif G17P_ARRAY_CONTROL
      "array-allocated-combined", "array-allocated-separate",
#else
      "allocated-combined", "allocated-separate", "pushed-combined",
      "pushed-separate", "public-blit",
#endif
   };
   for (uint32_t c = 0; c < OUTPUT_CASES; ++c) {
      const uint8_t *slot = readback_bytes + c * OUTPUT_STRIDE;
      uint32_t guard_wrong = 0;
      for (uint32_t i = 0; i < GUARD_SIZE; ++i) {
         if (slot[i] != 0xa5 || slot[GUARD_SIZE + OUTPUT_BYTES + i] != 0xa5)
            guard_wrong++;
      }
      uint32_t pixel_wrong = 0;
      uint32_t first_x = 0, first_y = 0;
      uint8_t first_actual[4] = {0}, first_expected[4] = {0};
      for (uint32_t y = 0; y < OUTPUT_HEIGHT; ++y) {
         for (uint32_t x = 0; x < OUTPUT_WIDTH; ++x) {
            uint8_t expected[4];
            source_pixel((x * SOURCE_WIDTH) / OUTPUT_WIDTH,
                         (y * SOURCE_HEIGHT) / OUTPUT_HEIGHT, expected);
            const uint8_t *actual =
               slot + GUARD_SIZE + (y * OUTPUT_WIDTH + x) * 4;
            if (memcmp(actual, expected, 4)) {
               if (pixel_wrong == 0) {
                  first_x = x;
                  first_y = y;
                  memcpy(first_actual, actual, 4);
                  memcpy(first_expected, expected, 4);
               }
               pixel_wrong++;
            }
         }
      }
      if (pixel_wrong || guard_wrong) {
         fprintf(stderr, TEST_NAME
                         ": CASE %s FAIL pixels=%u guards=%u "
                         "first=%u,%u got=%02x%02x%02x%02x "
                         "expected=%02x%02x%02x%02x phase=%s\n",
                 case_names[c], pixel_wrong, guard_wrong, first_x, first_y,
                 first_actual[0], first_actual[1], first_actual[2],
                 first_actual[3], first_expected[0], first_expected[1],
                 first_expected[2], first_expected[3], phase);
         all_ok = false;
      } else {
         printf(TEST_NAME ": CASE %s PASS pixels=1024 guards=128 phase=%s\n",
                case_names[c], phase);
      }
   }
   return all_ok;
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

   const VkApplicationInfo application_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = TEST_NAME,
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &application_info,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties2);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkEnumerateDeviceExtensionProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      die("out of host memory");
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
   if (!has_extension(physical, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME))
      die("VK_KHR_push_descriptor is not advertised");

   VkPhysicalDevicePushDescriptorPropertiesKHR push_properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR,
   };
   VkPhysicalDeviceProperties2 properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &push_properties,
   };
   vkGetPhysicalDeviceProperties2(physical, &properties2);
   if (push_properties.maxPushDescriptors != 32)
      die("unexpected G17P maxPushDescriptors");

   uint32_t queue_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, NULL);
   VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
   if (!queues)
      die("out of host memory");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_count, queues);
   uint32_t queue_family = UINT32_MAX;
   for (uint32_t i = 0; i < queue_count; ++i) {
      if (queues[i].queueCount) {
         queue_family = i;
         break;
      }
   }
   free(queues);
   if (queue_family == UINT32_MAX)
      die("device has no usable queue family");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const char *extensions[] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkCreateImage);
   LOAD_DEVICE(device, vkDestroyImage);
   LOAD_DEVICE(device, vkGetImageMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkFreeMemory);
   LOAD_DEVICE(device, vkBindImageMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkCreateImageView);
   LOAD_DEVICE(device, vkDestroyImageView);
   LOAD_DEVICE(device, vkCreateSampler);
   LOAD_DEVICE(device, vkDestroySampler);
   LOAD_DEVICE(device, vkCreateDescriptorSetLayout);
   LOAD_DEVICE(device, vkDestroyDescriptorSetLayout);
   LOAD_DEVICE(device, vkGetDescriptorSetLayoutSupport);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkDestroyPipelineLayout);
   LOAD_DEVICE(device, vkCreateDescriptorPool);
   LOAD_DEVICE(device, vkDestroyDescriptorPool);
   LOAD_DEVICE(device, vkAllocateDescriptorSets);
   LOAD_DEVICE(device, vkUpdateDescriptorSets);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkDestroyCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkResetCommandBuffer);
   LOAD_DEVICE(device, vkCmdBindDescriptorSets);
   LOAD_DEVICE(device, vkCmdPushDescriptorSetKHR);
   LOAD_DEVICE(device, vkCmdPipelineBarrier);
   LOAD_DEVICE(device, vkCmdBlitImage);
   LOAD_DEVICE(device, vkCreateBuffer);
   LOAD_DEVICE(device, vkDestroyBuffer);
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkUnmapMemory);
   LOAD_DEVICE(device, vkFlushMappedMemoryRanges);
   LOAD_DEVICE(device, vkInvalidateMappedMemoryRanges);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkDestroyShaderModule);
   LOAD_DEVICE(device, vkCreateGraphicsPipelines);
   LOAD_DEVICE(device, vkDestroyPipeline);
   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCmdCopyBufferToImage);
   LOAD_DEVICE(device, vkCmdCopyImageToBuffer);
   LOAD_DEVICE(device, vkCmdBeginRendering);
   LOAD_DEVICE(device, vkCmdEndRendering);
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdSetViewport);
   LOAD_DEVICE(device, vkCmdSetScissor);
   LOAD_DEVICE(device, vkCmdDraw);
   LOAD_DEVICE(device, vkCmdPushConstants);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkDestroyFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);

   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {4, 4, 1},
      .mipLevels = G17P_ARRAY_LOD_CONTROL ? 2 : 1,
      .arrayLayers = G17P_ARRAY_LOD_CONTROL ? 2 : 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   check(vkCreateImage(device, &image_info, NULL, &image), "vkCreateImage");
   VkMemoryRequirements memory_requirements;
   vkGetImageMemoryRequirements(device, image, &memory_requirements);
   const VkMemoryAllocateInfo memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memory_requirements.size,
      .memoryTypeIndex =
         choose_memory_type(physical, memory_requirements.memoryTypeBits),
   };
   VkDeviceMemory memory;
   check(vkAllocateMemory(device, &memory_info, NULL, &memory),
         "vkAllocateMemory");
   check(vkBindImageMemory(device, image, memory, 0), "vkBindImageMemory");

   VkImage destination;
   VkImageCreateInfo destination_info = image_info;
   destination_info.extent.width = OUTPUT_WIDTH;
   destination_info.extent.height = OUTPUT_HEIGHT;
   destination_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   check(vkCreateImage(device, &destination_info, NULL, &destination),
         "vkCreateImage(destination)");
   VkMemoryRequirements destination_requirements;
   vkGetImageMemoryRequirements(device, destination,
                                &destination_requirements);
   const VkMemoryAllocateInfo destination_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = destination_requirements.size,
      .memoryTypeIndex = choose_memory_type(
         physical, destination_requirements.memoryTypeBits),
   };
   VkDeviceMemory destination_memory;
   check(vkAllocateMemory(device, &destination_memory_info, NULL,
                          &destination_memory),
         "vkAllocateMemory(destination)");
   check(vkBindImageMemory(device, destination, destination_memory, 0),
         "vkBindImageMemory(destination)");

   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = (G17P_ARRAY_CONTROL || G17P_ARRAY_LOD_CONTROL)
                     ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                     : VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .baseMipLevel = 0,
         .levelCount = G17P_ARRAY_LOD_CONTROL ? 2 : 1,
         .baseArrayLayer = 0,
         .layerCount = G17P_ARRAY_LOD_CONTROL ? 2 : 1,
      },
   };
   VkImageView view;
   check(vkCreateImageView(device, &view_info, NULL, &view),
         "vkCreateImageView");

   const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = 1.0f,
   };
   VkSampler sampler;
   check(vkCreateSampler(device, &sampler_info, NULL, &sampler),
         "vkCreateSampler");

   const VkDescriptorSetLayoutBinding combined_binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   const VkDescriptorSetLayoutBinding separate_bindings[2] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
   };
   VkDescriptorSetLayout allocated_combined_layout = create_layout(
      device, 0, &combined_binding, 1, "allocated combined layout unsupported");
   VkDescriptorSetLayout allocated_separate_layout = create_layout(
      device, 0, separate_bindings, 2, "allocated separate layout unsupported");
   VkDescriptorSetLayout pushed_combined_layout = create_layout(
      device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
      &combined_binding, 1, "pushed combined layout unsupported");
   VkDescriptorSetLayout pushed_separate_layout = create_layout(
      device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
      separate_bindings, 2, "pushed separate layout unsupported");

   VkPipelineLayout allocated_combined_pipeline_layout =
      create_pipeline_layout_ext(device, allocated_combined_layout,
                                 G17P_ARRAY_LOD_CONTROL);
   VkPipelineLayout allocated_separate_pipeline_layout =
      create_pipeline_layout(device, allocated_separate_layout);
   VkPipelineLayout pushed_combined_pipeline_layout =
      create_pipeline_layout(device, pushed_combined_layout);
   VkPipelineLayout pushed_separate_pipeline_layout =
      create_pipeline_layout(device, pushed_separate_layout);

   VkShaderModule vertex_shader = create_shader(
      device, g17p_vulkan_push_descriptor_pixels_vert_spv,
      sizeof(g17p_vulkan_push_descriptor_pixels_vert_spv));
#if G17P_ARRAY_LOD_CONTROL
   VkShaderModule combined_fragment_shader = create_shader(
      device, g17p_vulkan_array_lod_implicit_frag_spv,
      sizeof(g17p_vulkan_array_lod_implicit_frag_spv));
   VkShaderModule separate_fragment_shader = create_shader(
      device, g17p_vulkan_array_lod_explicit_frag_spv,
      sizeof(g17p_vulkan_array_lod_explicit_frag_spv));
#elif G17P_ARRAY_CONTROL
   VkShaderModule combined_fragment_shader = create_shader(
      device, g17p_vulkan_push_descriptor_array_combined_frag_spv,
      sizeof(g17p_vulkan_push_descriptor_array_combined_frag_spv));
   VkShaderModule separate_fragment_shader = create_shader(
      device, g17p_vulkan_push_descriptor_array_separate_frag_spv,
      sizeof(g17p_vulkan_push_descriptor_array_separate_frag_spv));
#else
   VkShaderModule combined_fragment_shader = create_shader(
      device, g17p_vulkan_push_descriptor_combined_frag_spv,
      sizeof(g17p_vulkan_push_descriptor_combined_frag_spv));
   VkShaderModule separate_fragment_shader = create_shader(
      device, g17p_vulkan_push_descriptor_separate_frag_spv,
      sizeof(g17p_vulkan_push_descriptor_separate_frag_spv));
#endif
#if G17P_ARRAY_LOD_CONTROL
   const VkPipelineLayout pixel_pipeline_layouts[4] = {
      allocated_combined_pipeline_layout,
      allocated_combined_pipeline_layout,
      allocated_combined_pipeline_layout,
      allocated_combined_pipeline_layout,
   };
   VkPipeline pixel_pipelines[4] = {
      create_graphics_pipeline(device, pixel_pipeline_layouts[0],
                               vertex_shader, combined_fragment_shader),
      create_graphics_pipeline(device, pixel_pipeline_layouts[1],
                               vertex_shader, combined_fragment_shader),
      create_graphics_pipeline(device, pixel_pipeline_layouts[2],
                               vertex_shader, separate_fragment_shader),
      create_graphics_pipeline(device, pixel_pipeline_layouts[3],
                               vertex_shader, separate_fragment_shader),
   };
#else
   const VkPipelineLayout pixel_pipeline_layouts[4] = {
      allocated_combined_pipeline_layout,
      allocated_separate_pipeline_layout,
      pushed_combined_pipeline_layout,
      pushed_separate_pipeline_layout,
   };
   VkPipeline pixel_pipelines[4] = {
      create_graphics_pipeline(device, pixel_pipeline_layouts[0],
                               vertex_shader, combined_fragment_shader),
      create_graphics_pipeline(device, pixel_pipeline_layouts[1],
                               vertex_shader, separate_fragment_shader),
      create_graphics_pipeline(device, pixel_pipeline_layouts[2],
                               vertex_shader, combined_fragment_shader),
      create_graphics_pipeline(device, pixel_pipeline_layouts[3],
                               vertex_shader, separate_fragment_shader),
   };
#endif

   const VkDescriptorPoolSize pool_sizes[2] = {
      {
         .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
      },
      {
         .type = VK_DESCRIPTOR_TYPE_SAMPLER,
         .descriptorCount = 1,
      },
   };
   const VkDescriptorPoolSize sampled_image_pool_size = {
      .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .descriptorCount = 1,
   };
   VkDescriptorPoolSize all_pool_sizes[3] = {
      pool_sizes[0], pool_sizes[1], sampled_image_pool_size,
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2,
      .poolSizeCount = 3,
      .pPoolSizes = all_pool_sizes,
   };
   VkDescriptorPool descriptor_pool;
   check(vkCreateDescriptorPool(device, &pool_info, NULL, &descriptor_pool),
         "vkCreateDescriptorPool");
   const VkDescriptorSetLayout allocated_layouts[2] = {
      allocated_combined_layout, allocated_separate_layout,
   };
   const VkDescriptorSetAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 2,
      .pSetLayouts = allocated_layouts,
   };
   VkDescriptorSet descriptor_sets[2];
   check(vkAllocateDescriptorSets(device, &allocate_info, descriptor_sets),
         "vkAllocateDescriptorSets");

   const VkDescriptorImageInfo combined_info = {
      .sampler = sampler,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   const VkDescriptorImageInfo separate_infos[2] = {
      {
         .sampler = sampler,
      },
      {
         .imageView = view,
         .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      },
   };
   VkWriteDescriptorSet allocated_writes[3] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_sets[0],
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &combined_info,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_sets[1],
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
         .pImageInfo = &separate_infos[0],
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_sets[1],
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .pImageInfo = &separate_infos[1],
      },
   };
   vkUpdateDescriptorSets(device, 3, allocated_writes, 0, NULL);

#if !G17P_ARRAY_LOD_CONTROL
   const VkWriteDescriptorSet pushed_combined_write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &combined_info,
   };
   const VkWriteDescriptorSet pushed_separate_writes[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
         .pImageInfo = &separate_infos[0],
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .pImageInfo = &separate_infos[1],
      },
   };
#endif

   const VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool command_pool;
   check(vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command_buffer;
   check(vkAllocateCommandBuffers(device, &command_buffer_info,
                                  &command_buffer),
         "vkAllocateCommandBuffers");

#if !G17P_ARRAY_CONTROL && !G17P_ARRAY_LOD_CONTROL
   record_matrix(command_buffer, allocated_combined_pipeline_layout,
                 allocated_separate_pipeline_layout,
                 pushed_combined_pipeline_layout,
                 pushed_separate_pipeline_layout, descriptor_sets[0],
                 descriptor_sets[1], &pushed_combined_write,
                 pushed_separate_writes, false);
   check(vkResetCommandBuffer(command_buffer, 0), "vkResetCommandBuffer");
   record_matrix(command_buffer, allocated_combined_pipeline_layout,
                 allocated_separate_pipeline_layout,
                 pushed_combined_pipeline_layout,
                 pushed_separate_pipeline_layout, descriptor_sets[0],
                 descriptor_sets[1], &pushed_combined_write,
                 pushed_separate_writes, true);
#endif

#if G17P_ARRAY_LOD_CONTROL
   struct buffer_resource upload = create_buffer_resource(
      physical, device, GUARD_SIZE + 160 + GUARD_SIZE,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
#else
   struct buffer_resource upload = create_buffer_resource(
      physical, device, GUARD_SIZE + SOURCE_WIDTH * SOURCE_HEIGHT * 4 +
                           GUARD_SIZE,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
#endif
   struct buffer_resource readback = create_buffer_resource(
      physical, device, READBACK_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
#if G17P_ARRAY_LOD_CONTROL
   memset(upload.map, 0xa5, GUARD_SIZE + 160 + GUARD_SIZE);
   const uint32_t lod_offsets[2][2] = {{0, 64}, {128, 144}};
   const uint32_t lod_pixels[2] = {16, 4};
   for (uint32_t mip = 0; mip < 2; ++mip) {
      for (uint32_t layer = 0; layer < 2; ++layer) {
         uint8_t color[4];
         array_lod_color(mip, layer, color);
         uint8_t *data = (uint8_t *)upload.map + GUARD_SIZE +
                         lod_offsets[mip][layer];
         for (uint32_t p = 0; p < lod_pixels[mip]; ++p)
            memcpy(data + p * 4, color, 4);
      }
   }
#else
   memset(upload.map, 0xa5,
          GUARD_SIZE + SOURCE_WIDTH * SOURCE_HEIGHT * 4 + GUARD_SIZE);
   for (uint32_t y = 0; y < SOURCE_HEIGHT; ++y) {
      for (uint32_t x = 0; x < SOURCE_WIDTH; ++x) {
         uint8_t *pixel = (uint8_t *)upload.map + GUARD_SIZE +
                          (y * SOURCE_WIDTH + x) * 4;
         source_pixel(x, y, pixel);
      }
   }
#endif
   memset(readback.map, 0xa5, READBACK_BYTES);
   const VkMappedMemoryRange flush_ranges[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = upload.memory,
         .offset = 0,
         .size = VK_WHOLE_SIZE,
      },
      {
         .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = readback.memory,
         .offset = 0,
         .size = VK_WHOLE_SIZE,
      },
   };
   check(vkFlushMappedMemoryRanges(device, 2, flush_ranges),
         "vkFlushMappedMemoryRanges");

   struct image_resource pixel_source = {
      .image = image,
      .memory = memory,
      .view = view,
   };
   struct image_resource pixel_outputs[OUTPUT_CASES];
   for (uint32_t i = 0; i < OUTPUT_CASES; ++i) {
      pixel_outputs[i] = create_image_resource(
         physical, device, OUTPUT_WIDTH, OUTPUT_HEIGHT,
         (i < 4 ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0) |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            (i == 4 ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0),
         i < 4);
   }

   check(vkResetCommandBuffer(command_buffer, 0),
         "vkResetCommandBuffer(pixel matrix)");
#if G17P_ARRAY_LOD_CONTROL
   record_array_lod_matrix(command_buffer, &upload, &readback, &pixel_source,
                           pixel_outputs, pixel_pipelines,
                           allocated_combined_pipeline_layout,
                           descriptor_sets[0]);
#else
   record_pixel_matrix(command_buffer, &upload, &readback, &pixel_source,
                       pixel_outputs, pixel_pipelines,
                       pixel_pipeline_layouts, descriptor_sets,
                       &pushed_combined_write, pushed_separate_writes);
#endif

   const bool record_only =
      getenv("G17P_PUSH_DESCRIPTOR_RECORD_ONLY") != NULL;
   if (record_only) {
#if G17P_ARRAY_LOD_CONTROL
      puts(TEST_NAME ": RECORD PASS dimension=2D_ARRAY layers=2 mips=2 "
                     "allocated=combined implicit+explicit-lod0 cases=4 "
                     "(no GPU submission)");
#elif G17P_ARRAY_CONTROL
      puts(TEST_NAME ": RECORD PASS dimension=2D_ARRAY layers=1 filter=NEAREST "
                     "allocated=combined+separate cases=2 "
                     "(no GPU submission)");
#else
      puts(TEST_NAME ": RECORD PASS max=32 allocated=combined+separate "
                     "pushed=combined+separate meta-blit=separate-push "
                     "cases=5 (no GPU submission)");
#endif
   } else {
      VkQueue queue;
      vkGetDeviceQueue(device, queue_family, 0, &queue);
      const VkFenceCreateInfo fence_info = {
         .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      };
      VkFence fence;
      check(vkCreateFence(device, &fence_info, NULL, &fence),
            "vkCreateFence");
      const VkSubmitInfo submit = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &command_buffer,
      };
      check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
      check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences");
      const VkMappedMemoryRange invalidate_ranges[2] = {
         {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = upload.memory,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
         },
         {
            .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
            .memory = readback.memory,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
         },
      };
      check(vkInvalidateMappedMemoryRanges(device, 2, invalidate_ranges),
            "vkInvalidateMappedMemoryRanges(immediate)");
#if G17P_ARRAY_LOD_CONTROL
      const bool immediate_ok =
         validate_array_lod(&upload, &readback, "immediate");
#else
      const bool immediate_ok =
         validate_pixels(&upload, &readback, "immediate");
#endif
      sleep(3);
      check(vkInvalidateMappedMemoryRanges(device, 2, invalidate_ranges),
            "vkInvalidateMappedMemoryRanges(stable)");
#if G17P_ARRAY_LOD_CONTROL
      const bool stable_ok =
         validate_array_lod(&upload, &readback, "stable");
#else
      const bool stable_ok = validate_pixels(&upload, &readback, "stable");
#endif
      if (!immediate_ok || !stable_ok)
         die("pixel matrix mismatch");
#if G17P_ARRAY_LOD_CONTROL
      puts(TEST_NAME ": PIXEL PASS dimension=2D_ARRAY layers=2 mips=2 "
                     "allocated=combined implicit+explicit-lod0 cases=4 "
                     "pixels=4096 guards=512 immediate+3s");
#elif G17P_ARRAY_CONTROL
      puts(TEST_NAME ": PIXEL PASS dimension=2D_ARRAY layers=1 "
                     "filter=NEAREST cases=2 pixels=2048 guards=256 "
                     "immediate+3s");
#else
      puts(TEST_NAME ": PIXEL PASS dimension=2D filter=NEAREST cases=5 "
                     "pixels=5120 guards=640 immediate+3s");
#endif
      vkDestroyFence(device, fence, NULL);
   }

   for (uint32_t i = 0; i < OUTPUT_CASES; ++i)
      destroy_image_resource(device, &pixel_outputs[i]);
   destroy_buffer_resource(device, &readback);
   destroy_buffer_resource(device, &upload);
   vkDestroyCommandPool(device, command_pool, NULL);
   vkDestroyDescriptorPool(device, descriptor_pool, NULL);
   for (uint32_t i = 0; i < 4; ++i)
      vkDestroyPipeline(device, pixel_pipelines[i], NULL);
   vkDestroyShaderModule(device, separate_fragment_shader, NULL);
   vkDestroyShaderModule(device, combined_fragment_shader, NULL);
   vkDestroyShaderModule(device, vertex_shader, NULL);
   vkDestroyPipelineLayout(device, pushed_separate_pipeline_layout, NULL);
   vkDestroyPipelineLayout(device, pushed_combined_pipeline_layout, NULL);
   vkDestroyPipelineLayout(device, allocated_separate_pipeline_layout, NULL);
   vkDestroyPipelineLayout(device, allocated_combined_pipeline_layout, NULL);
   vkDestroyDescriptorSetLayout(device, pushed_separate_layout, NULL);
   vkDestroyDescriptorSetLayout(device, pushed_combined_layout, NULL);
   vkDestroyDescriptorSetLayout(device, allocated_separate_layout, NULL);
   vkDestroyDescriptorSetLayout(device, allocated_combined_layout, NULL);
   vkDestroySampler(device, sampler, NULL);
   vkDestroyImageView(device, view, NULL);
   vkDestroyImage(device, destination, NULL);
   vkFreeMemory(device, destination_memory, NULL);
   vkDestroyImage(device, image, NULL);
   vkFreeMemory(device, memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return EXIT_SUCCESS;
}
