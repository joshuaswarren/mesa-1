/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Records the exact G17P native-graphics uniform-range pressure a Zink or
 * Firefox WebGL vertex program produces: six vertex attributes, a draw
 * parameter, a sampled texture, three descriptor sets, and a promoted
 * constant. hk_g17p_graphics_uniform_ranges() must publish six independently
 * addressed uniform ranges for it.
 *
 * When the graphics range budget is smaller than the number of ranges the
 * collector can emit, the driver refuses fail-closed: the promoted-constant
 * range is dropped, hk_upload_usc_words() sets VK_ERROR_FEATURE_NOT_PRESENT,
 * and the error surfaces at vkEndCommandBuffer -- the same signature Firefox
 * produces on hardware:
 *
 *   MESA: error: G17P graphics uniform ranges refused: stage=0 ... rodata=2@72
 *   MESA: error: ZINK: vkEndCommandBuffer failed (VK_ERROR_FEATURE_NOT_PRESENT)
 *
 * Recording is enough to reach the refusal, so this runs against drm-shim
 * with no GPU. --execute additionally submits it on real hardware.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "g17p_vulkan_vertex_range_pressure_vert.h"
#include "g17p_wsi_triangle_frag.h"

#define T8140_DEVICE_ID 0x8140
#define ATTRIBS         6
#define VERTEX_STRIDE   (ATTRIBS * 16)
#define VERTICES        3
#define SETS            3

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
VK_FN(vkCreateImage);
VK_FN(vkDestroyImage);
VK_FN(vkGetImageMemoryRequirements);
VK_FN(vkBindImageMemory);
VK_FN(vkCreateImageView);
VK_FN(vkDestroyImageView);
VK_FN(vkCreateSampler);
VK_FN(vkDestroySampler);
VK_FN(vkCreateDescriptorSetLayout);
VK_FN(vkDestroyDescriptorSetLayout);
VK_FN(vkCreateDescriptorPool);
VK_FN(vkDestroyDescriptorPool);
VK_FN(vkAllocateDescriptorSets);
VK_FN(vkUpdateDescriptorSets);
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
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkEndCommandBuffer);
VK_FN(vkCmdBeginRenderPass);
VK_FN(vkCmdEndRenderPass);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindVertexBuffers);
VK_FN(vkCmdBindDescriptorSets);
VK_FN(vkCmdDraw);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);

static const char *const NAME = "g17p-vulkan-vertex-range-pressure";

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
      die("usage: g17p-vulkan-vertex-range-pressure [--execute]");

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
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures2);
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

   VkPhysicalDeviceShaderDrawParametersFeatures draw_parameters = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &draw_parameters,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   if (!draw_parameters.shaderDrawParameters)
      die("shaderDrawParameters not advertised");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   VkPhysicalDeviceShaderDrawParametersFeatures enable_draw_parameters = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES,
      .shaderDrawParameters = VK_TRUE,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enable_draw_parameters,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

#define LOAD(name)                                                             \
   do {                                                                        \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                    \
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
   LOAD(vkCreateSampler); LOAD(vkDestroySampler);
   LOAD(vkCreateDescriptorSetLayout); LOAD(vkDestroyDescriptorSetLayout);
   LOAD(vkCreateDescriptorPool); LOAD(vkDestroyDescriptorPool);
   LOAD(vkAllocateDescriptorSets); LOAD(vkUpdateDescriptorSets);
   LOAD(vkCreateShaderModule); LOAD(vkDestroyShaderModule);
   LOAD(vkCreateRenderPass); LOAD(vkDestroyRenderPass);
   LOAD(vkCreateFramebuffer); LOAD(vkDestroyFramebuffer);
   LOAD(vkCreatePipelineLayout); LOAD(vkDestroyPipelineLayout);
   LOAD(vkCreateGraphicsPipelines); LOAD(vkDestroyPipeline);
   LOAD(vkCreateCommandPool); LOAD(vkDestroyCommandPool);
   LOAD(vkAllocateCommandBuffers); LOAD(vkBeginCommandBuffer);
   LOAD(vkEndCommandBuffer); LOAD(vkCmdBeginRenderPass);
   LOAD(vkCmdEndRenderPass); LOAD(vkCmdBindPipeline);
   LOAD(vkCmdBindVertexBuffers); LOAD(vkCmdBindDescriptorSets);
   LOAD(vkCmdDraw); LOAD(vkCreateFence); LOAD(vkDestroyFence);
   LOAD(vkQueueSubmit); LOAD(vkWaitForFences);
#undef LOAD

   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);
   VkPhysicalDeviceMemoryProperties memory;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory);

   /* Vertex buffer: six vec4 attributes per vertex. */
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
   memset(vertex_map, 0, vertex_bytes);
   vkUnmapMemory(device, vertex_memory);

   /* One 1x1 sampled image, shared by all three descriptor sets. */
   const VkImageCreateInfo sampled_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {1, 1, 1},
      .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage sampled_image;
   check(vkCreateImage(device, &sampled_info, NULL, &sampled_image),
         "vkCreateImage(sampled)");
   VkMemoryRequirements sampled_requirements;
   vkGetImageMemoryRequirements(device, sampled_image, &sampled_requirements);
   const VkMemoryAllocateInfo sampled_alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = sampled_requirements.size,
      .memoryTypeIndex = memory_type(
         &memory, sampled_requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory sampled_memory;
   check(vkAllocateMemory(device, &sampled_alloc, NULL, &sampled_memory),
         "vkAllocateMemory(sampled)");
   check(vkBindImageMemory(device, sampled_image, sampled_memory, 0),
         "vkBindImageMemory(sampled)");
   const VkImageViewCreateInfo sampled_view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = sampled_image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VkImageView sampled_view;
   check(vkCreateImageView(device, &sampled_view_info, NULL, &sampled_view),
         "vkCreateImageView(sampled)");
   const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = VK_LOD_CLAMP_NONE,
      .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
   };
   VkSampler sampler;
   check(vkCreateSampler(device, &sampler_info, NULL, &sampler),
         "vkCreateSampler");

   /* Three descriptor sets, so the root range covers three sets exactly as
    * the observed hardware refusal reports (sets=3). */
   const VkDescriptorSetLayoutBinding bindings[2] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
      },
   };
   const VkDescriptorSetLayoutCreateInfo first_set_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings,
   };
   const VkDescriptorSetLayoutCreateInfo set_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout set_layouts[SETS];
   for (unsigned i = 0; i < SETS; ++i)
      check(vkCreateDescriptorSetLayout(
               device, i == 0 ? &first_set_layout_info : &set_layout_info, NULL,
               &set_layouts[i]),
            "vkCreateDescriptorSetLayout");

   const VkDescriptorPoolSize pool_sizes[2] = {
      {
         .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = SETS,
      },
      {
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
      },
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = SETS,
      .poolSizeCount = 2,
      .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool pool;
   check(vkCreateDescriptorPool(device, &pool_info, NULL, &pool),
         "vkCreateDescriptorPool");
   const VkDescriptorSetAllocateInfo set_alloc = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool,
      .descriptorSetCount = SETS,
      .pSetLayouts = set_layouts,
   };
   VkDescriptorSet sets[SETS];
   check(vkAllocateDescriptorSets(device, &set_alloc, sets),
         "vkAllocateDescriptorSets");

   const VkDescriptorImageInfo image_binding = {
      .sampler = sampler,
      .imageView = sampled_view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   const VkDeviceSize block_bytes = 32 * 16;
   const VkBufferCreateInfo block_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = block_bytes,
      .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
   };
   VkBuffer block_buffer;
   check(vkCreateBuffer(device, &block_buffer_info, NULL, &block_buffer),
         "vkCreateBuffer(block)");
   VkMemoryRequirements block_requirements;
   vkGetBufferMemoryRequirements(device, block_buffer, &block_requirements);
   const VkMemoryAllocateInfo block_alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = block_requirements.size,
      .memoryTypeIndex =
         memory_type(&memory, block_requirements.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory block_memory;
   check(vkAllocateMemory(device, &block_alloc, NULL, &block_memory),
         "vkAllocateMemory(block)");
   check(vkBindBufferMemory(device, block_buffer, block_memory, 0),
         "vkBindBufferMemory(block)");
   void *block_map;
   check(vkMapMemory(device, block_memory, 0, block_bytes, 0, &block_map),
         "vkMapMemory(block)");
   memset(block_map, 0, block_bytes);
   vkUnmapMemory(device, block_memory);

   const VkDescriptorBufferInfo block_binding = {
      .buffer = block_buffer, .offset = 0, .range = block_bytes,
   };
   VkWriteDescriptorSet writes[SETS + 1];
   for (unsigned i = 0; i < SETS; ++i) {
      writes[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = sets[i],
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &image_binding,
      };
   }
   writes[SETS] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = sets[0],
      .dstBinding = 1,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .pBufferInfo = &block_binding,
   };
   vkUpdateDescriptorSets(device, SETS + 1, writes, 0, NULL);

   /* 1x1 colour attachment. */
   const VkImageCreateInfo colour_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {1, 1, 1},
      .mipLevels = 1, .arrayLayers = 1,
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
   const VkImageViewCreateInfo colour_view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = colour_image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };
   VkImageView colour_view;
   check(vkCreateImageView(device, &colour_view_info, NULL, &colour_view),
         "vkCreateImageView(colour)");

   const VkAttachmentDescription attachment = {
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkAttachmentReference colour_reference = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &colour_reference,
   };
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .subpassCount = 1, .pSubpasses = &subpass,
      .attachmentCount = 1, .pAttachments = &attachment,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &render_pass_info, NULL, &render_pass),
         "vkCreateRenderPass");
   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = 1, .pAttachments = &colour_view,
      .width = 1, .height = 1, .layers = 1,
   };
   VkFramebuffer framebuffer;
   check(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer),
         "vkCreateFramebuffer");

   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = SETS,
      .pSetLayouts = set_layouts,
   };
   VkPipelineLayout layout;
   check(vkCreatePipelineLayout(device, &layout_info, NULL, &layout),
         "vkCreatePipelineLayout");

   const VkShaderModuleCreateInfo vertex_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_vertex_range_pressure_vert_spv),
      .pCode = g17p_vulkan_vertex_range_pressure_vert_spv,
   };
   VkShaderModule vertex_module;
   check(vkCreateShaderModule(device, &vertex_module_info, NULL,
                              &vertex_module),
         "vkCreateShaderModule(vertex)");
   const VkShaderModuleCreateInfo fragment_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_wsi_triangle_frag_spv),
      .pCode = g17p_wsi_triangle_frag_spv,
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
   VkVertexInputAttributeDescription attributes[ATTRIBS];
   for (unsigned i = 0; i < ATTRIBS; ++i) {
      attributes[i] = (VkVertexInputAttributeDescription){
         .location = i, .binding = 0,
         .format = VK_FORMAT_R32G32B32A32_SFLOAT,
         .offset = i * 16,
      };
   }
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = ATTRIBS,
      .pVertexAttributeDescriptions = attributes,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport static_viewport = {
      .width = 1.0f, .height = 1.0f, .maxDepth = 1.0f,
   };
   const VkRect2D static_scissor = {.extent = {1, 1}};
   const VkPipelineViewportStateCreateInfo viewport = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &static_viewport,
      .scissorCount = 1, .pScissors = &static_scissor,
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
   const VkPipelineColorBlendAttachmentState blend_attachment = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &blend_attachment,
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
      .layout = layout, .renderPass = render_pass,
   };
   VkPipeline pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipeline),
         "vkCreateGraphicsPipelines");

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
   check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");
   const VkRenderPassBeginInfo pass_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {.extent = {1, 1}},
   };
   vkCmdBeginRenderPass(command, &pass_begin, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0,
                           SETS, sets, 0, NULL);
   const VkDeviceSize vertex_offset = 0;
   vkCmdBindVertexBuffers(command, 0, 1, &vertex_buffer, &vertex_offset);

   /* The draw-parameter range is published because the vertex shader reads
    * gl_BaseVertexARB, not because firstVertex is nonzero; a nonzero
    * firstVertex simply keeps the recorded draw realistic. */
   vkCmdDraw(command, VERTICES, 1, 1, 0);
   vkCmdEndRenderPass(command);

   const VkResult recorded = vkEndCommandBuffer(command);
   if (recorded != VK_SUCCESS) {
      fprintf(stderr,
              "%s: vkEndCommandBuffer returned %d; the native G17P graphics "
              "path refused this vertex program's uniform ranges\n",
              NAME, recorded);
      return EXIT_FAILURE;
   }

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
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyShaderModule(device, fragment_module, NULL);
   vkDestroyShaderModule(device, vertex_module, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyFramebuffer(device, framebuffer, NULL);
   vkDestroyRenderPass(device, render_pass, NULL);
   vkDestroyImageView(device, colour_view, NULL);
   vkDestroyImage(device, colour_image, NULL);
   vkFreeMemory(device, colour_memory, NULL);
   vkDestroyDescriptorPool(device, pool, NULL);
   for (unsigned i = 0; i < SETS; ++i)
      vkDestroyDescriptorSetLayout(device, set_layouts[i], NULL);
   vkDestroySampler(device, sampler, NULL);
   vkDestroyImageView(device, sampled_view, NULL);
   vkDestroyImage(device, sampled_image, NULL);
   vkFreeMemory(device, sampled_memory, NULL);
   vkDestroyBuffer(device, block_buffer, NULL);
   vkFreeMemory(device, block_memory, NULL);
   vkDestroyBuffer(device, vertex_buffer, NULL);
   vkFreeMemory(device, vertex_memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);

   puts("G17P VERTEX RANGE PRESSURE PASS: six-range native vertex program "
        "reached vkEndCommandBuffer");
   return EXIT_SUCCESS;
}
