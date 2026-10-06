/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Strict public VK_EXT_transform_feedback probe for T8140/G17P.  The first
 * run appends three distinct vec4 values to a guarded buffer; its counter is
 * then used to resume a second run, which must append the same three values.
 * Nonzero Vulkan binding, vertex-attribute, XFB-output, and counter offsets
 * make zero-based or stride-confused implementations fail unambiguously.
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

#include "g17p_vulkan_transform_feedback_indexed.h"
#include "g17p_vulkan_transform_feedback_vert.h"
#include "g17p_wsi_triangle_frag.h"
#include "g17p_wsi_triangle_vert.h"

#define T8140_DEVICE_ID 0x8140
#define GUARD           64
#define VERTEX_STRIDE   32
#define XFB_STRIDE      32
#define VERTICES        3
#define ROUNDS          2
#define XFB_BYTES       (VERTICES * ROUNDS * XFB_STRIDE)
#define SENTINEL        0xa5

#define VK_FN(name) static PFN_##name name
VK_FN(vkGetInstanceProcAddr);
VK_FN(vkCreateBufferView);
VK_FN(vkDestroyBufferView);
VK_FN(vkCreateDescriptorSetLayout);
VK_FN(vkDestroyDescriptorSetLayout);
VK_FN(vkCreateDescriptorPool);
VK_FN(vkDestroyDescriptorPool);
VK_FN(vkAllocateDescriptorSets);
VK_FN(vkUpdateDescriptorSets);
VK_FN(vkCmdBindDescriptorSets);

VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceProperties2);
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
VK_FN(vkCmdBindTransformFeedbackBuffersEXT);
VK_FN(vkCmdBeginTransformFeedbackEXT);
VK_FN(vkCmdEndTransformFeedbackEXT);
VK_FN(vkCmdDrawIndirectByteCountEXT);
VK_FN(vkCmdDraw);
VK_FN(vkCmdPipelineBarrier);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);

static const char *const NAME = "g17p-vulkan-transform-feedback";

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

#define LOAD_DEVICE(device, name)                                              \
   do {                                                                        \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                   \
      if (!name)                                                               \
         die("device is missing " #name);                                      \
   } while (0)

struct buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint8_t *map;
   VkDeviceSize size;
};

static uint32_t
memory_type(const VkPhysicalDeviceMemoryProperties *properties, uint32_t bits)
{
   const VkMemoryPropertyFlags need = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & need) == need)
         return i;
   }
   die("no host-visible coherent memory type");
   return 0;
}

static struct buffer
make_buffer(VkDevice device, const VkPhysicalDeviceMemoryProperties *memory,
            VkDeviceSize size, VkBufferUsageFlags usage)
{
   struct buffer out = {.size = size};
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
   };
   check(vkCreateBuffer(device, &info, NULL, &out.buffer), "vkCreateBuffer");

   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, out.buffer, &requirements);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = memory_type(memory, requirements.memoryTypeBits),
   };
   check(vkAllocateMemory(device, &alloc, NULL, &out.memory),
         "vkAllocateMemory");
   check(vkBindBufferMemory(device, out.buffer, out.memory, 0),
         "vkBindBufferMemory");
   check(vkMapMemory(device, out.memory, 0, size, 0, (void **)&out.map),
         "vkMapMemory");
   return out;
}

static void
destroy_buffer(VkDevice device, struct buffer *buffer)
{
   vkUnmapMemory(device, buffer->memory);
   vkDestroyBuffer(device, buffer->buffer, NULL);
   vkFreeMemory(device, buffer->memory, NULL);
}

static unsigned
verify_bytes(const struct buffer *vertices, const struct buffer *xfb,
             const struct buffer *counter, bool indexed)
{
   static const float values[VERTICES][4] = {
      {1.0f, 2.0f, 3.0f, 4.0f},
      {10.0f, 20.0f, 30.0f, 40.0f},
      {-1.0f, -2.0f, -3.0f, -4.0f},
   };
   unsigned failures = 0;

   for (size_t i = 0; i < vertices->size; ++i) {
      uint8_t expected = SENTINEL;
      for (unsigned v = 0; v < VERTICES; ++v) {
         const size_t payload = GUARD + v * VERTEX_STRIDE + 8;
         if (i >= payload && i < payload + sizeof(values[v]))
            expected = ((const uint8_t *)values[v])[i - payload];
      }
      if (vertices->map[i] != expected) {
         fprintf(stderr,
                 "%s: vertex source changed at byte %zu: %02x != %02x\n", NAME,
                 i, vertices->map[i], expected);
         failures++;
         break;
      }
   }

   float expected_values[VERTICES][4];
   for (unsigned v = 0; v < VERTICES; v++)
      for (unsigned c = 0; c < 4; c++)
         expected_values[v][c] =
            values[v][c] + (indexed ? (float)(100 * (v + 1) + c + 1) : 0.0f);
   for (size_t i = 0; i < xfb->size; ++i) {
      uint8_t expected = SENTINEL;
      for (unsigned round = 0; round < ROUNDS; ++round) {
         for (unsigned v = 0; v < VERTICES; ++v) {
            const size_t payload =
               GUARD + round * VERTICES * XFB_STRIDE + v * XFB_STRIDE + 16;
            if (i >= payload && i < payload + sizeof(expected_values[v]))
               expected = ((const uint8_t *)expected_values[v])[i - payload];
         }
      }
      if (xfb->map[i] != expected) {
         fprintf(stderr, "%s: XFB byte %zu: %02x != %02x\n", NAME, i,
                 xfb->map[i], expected);

         size_t changed = 0;
         size_t first = SIZE_MAX;
         size_t last = 0;
         for (size_t j = 0; j < xfb->size; ++j) {
            if (xfb->map[j] == SENTINEL)
               continue;
            if (first == SIZE_MAX)
               first = j;
            last = j;
            changed++;
         }
         if (first == SIZE_MAX) {
            fprintf(stderr, "%s: XFB buffer remained entirely sentinel\n",
                    NAME);
         } else {
            fprintf(stderr,
                    "%s: XFB has %zu non-sentinel bytes spanning %zu..%zu\n",
                    NAME, changed, first, last);
         }
         failures++;
         break;
      }
   }

   for (size_t i = 0; i < counter->size; ++i) {
      uint8_t expected = SENTINEL;
      const uint32_t final_offset = XFB_BYTES;
      if (i >= GUARD && i < GUARD + sizeof(final_offset))
         expected = ((const uint8_t *)&final_offset)[i - GUARD];
      if (counter->map[i] != expected) {
         fprintf(stderr, "%s: counter/guard byte %zu: %02x != %02x\n", NAME, i,
                 counter->map[i], expected);
         failures++;
         break;
      }
   }
   if (failures) {
      uint32_t observed_offset;
      memcpy(&observed_offset, counter->map + GUARD, sizeof(observed_offset));
      fprintf(stderr, "%s: observed XFB counter %#x\n", NAME, observed_offset);
   }
   return failures;
}

int
main(int argc, char **argv)
{
   const bool rasterized = getenv("G17P_XFB_RASTERIZED_DIAGNOSTIC") != NULL;
   const bool geometry =
      argc == 2 && (!strcmp(argv[1], "--indexed-gs-execute") ||
                    !strcmp(argv[1], "--indexed-gs-record"));
   const bool indexed =
      geometry || (argc == 2 && (!strcmp(argv[1], "--indexed-execute") ||
                                 !strcmp(argv[1], "--indexed-record")));
   const bool execute = argc == 2 && (!strcmp(argv[1], "--execute") ||
                                      !strcmp(argv[1], "--indexed-execute") ||
                                      !strcmp(argv[1], "--indexed-gs-execute"));
   const bool submit_only = argc == 2 && !strcmp(argv[1], "--submit-only");
   if (argc > 2 || (argc == 2 && !execute && !submit_only && !indexed))
      die(
         "usage: g17p-vulkan-transform-feedback [--submit-only|--execute|--indexed-record|--indexed-execute|--indexed-gs-record|--indexed-gs-execute]");
   if (submit_only && !getenv("ASAHI_SHIM_G17P_EXPECT_XFB"))
      die("--submit-only is a drm-shim oracle; use --execute on hardware");

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
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties2);
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

   VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb_feature = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &xfb_feature,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   if (!xfb_feature.transformFeedback)
      die("transformFeedback not advertised");

   VkPhysicalDeviceTransformFeedbackPropertiesEXT xfb_properties = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT,
   };
   VkPhysicalDeviceProperties2 properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &xfb_properties,
   };
   vkGetPhysicalDeviceProperties2(physical, &properties2);
   if (xfb_properties.maxTransformFeedbackBuffers < 1 ||
       xfb_properties.maxTransformFeedbackBufferSize < XFB_BYTES ||
       xfb_properties.maxTransformFeedbackBufferDataStride < XFB_STRIDE ||
       !xfb_properties.transformFeedbackDraw)
      die("transform-feedback properties cannot represent the oracle");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   xfb_feature.transformFeedback = VK_TRUE;
   xfb_feature.geometryStreams = VK_FALSE;
   const char *extensions[] = {VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME};
   if (geometry && !features.features.geometryShader)
      die("geometryShader not supported");
   const VkPhysicalDeviceFeatures core_features = {.geometryShader = geometry};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &xfb_feature,
      .pEnabledFeatures = &core_features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

#define LOAD(name) LOAD_DEVICE(device, name)
   LOAD(vkCreateBufferView);
   LOAD(vkDestroyBufferView);
   LOAD(vkCreateDescriptorSetLayout);
   LOAD(vkDestroyDescriptorSetLayout);
   LOAD(vkCreateDescriptorPool);
   LOAD(vkDestroyDescriptorPool);
   LOAD(vkAllocateDescriptorSets);
   LOAD(vkUpdateDescriptorSets);
   LOAD(vkCmdBindDescriptorSets);
   LOAD(vkDestroyDevice);
   LOAD(vkGetDeviceQueue);
   LOAD(vkCreateBuffer);
   LOAD(vkDestroyBuffer);
   LOAD(vkGetBufferMemoryRequirements);
   LOAD(vkAllocateMemory);
   LOAD(vkFreeMemory);
   LOAD(vkBindBufferMemory);
   LOAD(vkMapMemory);
   LOAD(vkUnmapMemory);
   LOAD(vkCreateImage);
   LOAD(vkDestroyImage);
   LOAD(vkGetImageMemoryRequirements);
   LOAD(vkBindImageMemory);
   LOAD(vkCreateImageView);
   LOAD(vkDestroyImageView);
   LOAD(vkCreateShaderModule);
   LOAD(vkDestroyShaderModule);
   LOAD(vkCreateRenderPass);
   LOAD(vkDestroyRenderPass);
   LOAD(vkCreateFramebuffer);
   LOAD(vkDestroyFramebuffer);
   LOAD(vkCreatePipelineLayout);
   LOAD(vkDestroyPipelineLayout);
   LOAD(vkCreateGraphicsPipelines);
   LOAD(vkDestroyPipeline);
   LOAD(vkCreateCommandPool);
   LOAD(vkDestroyCommandPool);
   LOAD(vkAllocateCommandBuffers);
   LOAD(vkBeginCommandBuffer);
   LOAD(vkEndCommandBuffer);
   LOAD(vkCmdBeginRenderPass);
   LOAD(vkCmdEndRenderPass);
   LOAD(vkCmdBindPipeline);
   LOAD(vkCmdBindVertexBuffers);
   LOAD(vkCmdBindTransformFeedbackBuffersEXT);
   LOAD(vkCmdBeginTransformFeedbackEXT);
   LOAD(vkCmdEndTransformFeedbackEXT);
   LOAD(vkCmdDrawIndirectByteCountEXT);
   LOAD(vkCmdDraw);
   LOAD(vkCmdPipelineBarrier);
   LOAD(vkCreateFence);
   LOAD(vkDestroyFence);
   LOAD(vkQueueSubmit);
   LOAD(vkWaitForFences);
#undef LOAD

   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);
   VkPhysicalDeviceMemoryProperties memory;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory);

   struct buffer vertices =
      make_buffer(device, &memory, GUARD + VERTICES * VERTEX_STRIDE + GUARD,
                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
   struct buffer xfb =
      make_buffer(device, &memory, GUARD + XFB_BYTES + GUARD,
                  VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT);
   struct buffer counter =
      make_buffer(device, &memory, GUARD + sizeof(uint32_t) + GUARD,
                  VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT);
   memset(vertices.map, SENTINEL, vertices.size);
   memset(xfb.map, SENTINEL, xfb.size);
   memset(counter.map, SENTINEL, counter.size);
   static const float values[VERTICES][4] = {
      {1.0f, 2.0f, 3.0f, 4.0f},
      {10.0f, 20.0f, 30.0f, 40.0f},
      {-1.0f, -2.0f, -3.0f, -4.0f},
   };
   for (unsigned v = 0; v < VERTICES; ++v)
      memcpy(vertices.map + GUARD + v * VERTEX_STRIDE + 8, values[v],
             sizeof(values[v]));

   struct buffer texture = make_buffer(
      device, &memory, VERTICES * 16, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT);
   for (unsigned v = 0; v < VERTICES; v++)
      for (unsigned c = 0; c < 4; c++)
         ((float *)texture.map)[v * 4 + c] = (float)(100 * (v + 1) + c + 1);
   VkBufferView texture_view;
   check(
      vkCreateBufferView(device,
                         &(VkBufferViewCreateInfo){
                            .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
                            .buffer = texture.buffer,
                            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                            .range = VERTICES * 16},
                         NULL, &texture_view),
      "texture view");
   VkDescriptorSetLayoutBinding tb = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT};
   VkDescriptorSetLayout dlayout;
   check(vkCreateDescriptorSetLayout(
            device,
            &(VkDescriptorSetLayoutCreateInfo){
               .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
               .bindingCount = 1,
               .pBindings = &tb},
            NULL, &dlayout),
         "descriptor layout");
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1};
   VkDescriptorPool dp;
   check(vkCreateDescriptorPool(
            device,
            &(VkDescriptorPoolCreateInfo){
               .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
               .maxSets = 1,
               .poolSizeCount = 1,
               .pPoolSizes = &ps},
            NULL, &dp),
         "descriptor pool");
   VkDescriptorSet ds;
   check(vkAllocateDescriptorSets(
            device,
            &(VkDescriptorSetAllocateInfo){
               .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
               .descriptorPool = dp,
               .descriptorSetCount = 1,
               .pSetLayouts = &dlayout},
            &ds),
         "descriptor allocate");
   VkWriteDescriptorSet wr = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = ds,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
      .pTexelBufferView = &texture_view};
   vkUpdateDescriptorSets(device, 1, &wr, 0, NULL);
   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {1, 1, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   check(vkCreateImage(device, &image_info, NULL, &image), "vkCreateImage");
   VkMemoryRequirements image_requirements;
   vkGetImageMemoryRequirements(device, image, &image_requirements);
   const VkMemoryAllocateInfo image_alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = image_requirements.size,
      .memoryTypeIndex =
         memory_type(&memory, image_requirements.memoryTypeBits),
   };
   VkDeviceMemory image_memory;
   check(vkAllocateMemory(device, &image_alloc, NULL, &image_memory),
         "vkAllocateMemory(image)");
   check(vkBindImageMemory(device, image, image_memory, 0),
         "vkBindImageMemory");
   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange =
         {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
         },
   };
   VkImageView image_view;
   check(vkCreateImageView(device, &view_info, NULL, &image_view),
         "vkCreateImageView");

   const VkAttachmentDescription attachment = {
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkAttachmentReference color = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color,
   };
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .attachmentCount = 1,
      .pAttachments = &attachment,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &render_pass_info, NULL, &render_pass),
         "vkCreateRenderPass");
   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = 1,
      .pAttachments = &image_view,
      .width = 1,
      .height = 1,
      .layers = 1,
   };
   VkFramebuffer framebuffer;
   check(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer),
         "vkCreateFramebuffer");
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = indexed ? 1 : 0,
      .pSetLayouts = &dlayout,
   };
   VkPipelineLayout layout;
   check(vkCreatePipelineLayout(device, &layout_info, NULL, &layout),
         "vkCreatePipelineLayout");
   const VkShaderModuleCreateInfo vertex_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = indexed ? (geometry ? sizeof(g17p_xfb_indexed_vert_spv)
                                      : sizeof(g17p_xfb_indexed_auto_vert_spv))
                          : sizeof(g17p_vulkan_transform_feedback_vert_spv),
      .pCode = indexed ? (geometry ? g17p_xfb_indexed_vert_spv
                                   : g17p_xfb_indexed_auto_vert_spv)
                       : g17p_vulkan_transform_feedback_vert_spv,
   };
   VkShaderModule vertex_module;
   check(
      vkCreateShaderModule(device, &vertex_module_info, NULL, &vertex_module),
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
   const VkShaderModuleCreateInfo draw_vertex_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_wsi_triangle_vert_spv),
      .pCode = g17p_wsi_triangle_vert_spv,
   };
   VkShaderModule draw_vertex_module;
   check(vkCreateShaderModule(device, &draw_vertex_module_info, NULL,
                              &draw_vertex_module),
         "vkCreateShaderModule(draw vertex)");

   VkShaderModule geometry_module;
   check(vkCreateShaderModule(
            device,
            &(VkShaderModuleCreateInfo){
               .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
               .codeSize = sizeof(g17p_xfb_passthrough_geom_spv),
               .pCode = g17p_xfb_passthrough_geom_spv},
            NULL, &geometry_module),
         "geometry module");
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vertex_module,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = geometry ? VK_SHADER_STAGE_GEOMETRY_BIT
                           : VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = geometry ? geometry_module : fragment_module,
         .pName = "main",
      },
   };
   const VkVertexInputBindingDescription binding = {
      .binding = 0,
      .stride = VERTEX_STRIDE,
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription attribute = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32B32A32_SFLOAT,
      .offset = 8,
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &binding,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &attribute,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = geometry ? VK_PRIMITIVE_TOPOLOGY_POINT_LIST
                           : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport static_viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = 1.0f,
      .height = 1.0f,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };
   const VkRect2D static_scissor = {.extent = {1, 1}};
   const VkPipelineViewportStateCreateInfo viewport = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &static_viewport,
      .scissorCount = 1,
      .pScissors = &static_scissor,
   };
   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .rasterizerDiscardEnable = !rasterized,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = geometry || rasterized ? 2 : 1,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .layout = layout,
      .renderPass = render_pass,
   };
   VkPipeline pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipeline),
         "vkCreateGraphicsPipelines");

   /* Consume the XFB counter with a plain rasterizer-discard pipeline. The
    * source stream remains the transform-feedback pipeline above; using a
    * separate consumer is the normal draw-auto shape and keeps this oracle
    * focused on byte-count conversion rather than indirect geometry lowering.
    */
   const VkPipelineShaderStageCreateInfo draw_stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = draw_vertex_module,
      .pName = "main",
   };
   const VkPipelineVertexInputStateCreateInfo draw_vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkGraphicsPipelineCreateInfo draw_pipeline_info = pipeline_info;
   draw_pipeline_info.stageCount = 1;
   draw_pipeline_info.pStages = &draw_stage;
   draw_pipeline_info.pVertexInputState = &draw_vertex_input;
   VkPipeline draw_pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                   &draw_pipeline_info, NULL, &draw_pipeline),
         "vkCreateGraphicsPipelines(draw byte count)");

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool pool;
   check(vkCreateCommandPool(device, &pool_info, NULL, &pool),
         "vkCreateCommandPool");
   VkCommandBuffer command;
   const VkCommandBufferAllocateInfo command_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   check(vkAllocateCommandBuffers(device, &command_info, &command),
         "vkAllocateCommandBuffers");
   check(vkBeginCommandBuffer(
            command,
            &(VkCommandBufferBeginInfo){
               .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            }),
         "vkBeginCommandBuffer");
   const VkRenderPassBeginInfo begin_render = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {.extent = {1, 1}},
   };
   vkCmdBeginRenderPass(command, &begin_render, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   if (indexed)
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                              0, 1, &ds, 0, NULL);
   const VkDeviceSize vertex_offset = GUARD;
   vkCmdBindVertexBuffers(command, 0, 1, &vertices.buffer, &vertex_offset);
   const VkDeviceSize xfb_offset = GUARD;
   const VkDeviceSize xfb_size = XFB_BYTES;
   vkCmdBindTransformFeedbackBuffersEXT(command, 0, 1, &xfb.buffer, &xfb_offset,
                                        &xfb_size);
   const VkDeviceSize counter_offset = GUARD;
   const VkBuffer null_counter = VK_NULL_HANDLE;
   vkCmdBeginTransformFeedbackEXT(command, 0, 1, &null_counter,
                                  &counter_offset);
   vkCmdDraw(command, VERTICES, 1, 0, 0);
   vkCmdEndTransformFeedbackEXT(command, 0, 1, &counter.buffer,
                                &counter_offset);
   vkCmdEndRenderPass(command);
   /* Resume consumes the previous EndTransformFeedback counter write.
    * Put the dependency outside the render pass so it needs no subpass
    * self-dependency. VK_EXT_transform_feedback requires this ordering. */
   VkBufferMemoryBarrier resume_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
      .dstAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = counter.buffer,
      .offset = counter_offset,
      .size = sizeof(uint32_t)};
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                        VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT, 0, 0,
                        NULL, 1, &resume_barrier, 0, NULL);
   vkCmdBeginRenderPass(command, &begin_render, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   if (indexed)
      vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                              0, 1, &ds, 0, NULL);
   vkCmdBindVertexBuffers(command, 0, 1, &vertices.buffer, &vertex_offset);
   vkCmdBindTransformFeedbackBuffersEXT(command, 0, 1, &xfb.buffer, &xfb_offset,
                                        &xfb_size);
   vkCmdBeginTransformFeedbackEXT(command, 0, 1, &counter.buffer,
                                  &counter_offset);
   vkCmdDraw(command, VERTICES, 1, 0, 0);
   vkCmdEndTransformFeedbackEXT(command, 0, 1, &counter.buffer,
                                &counter_offset);
   vkCmdEndRenderPass(command);

   const VkBufferMemoryBarrier counter_barrier = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
      .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = counter.buffer,
      .offset = counter_offset,
      .size = sizeof(uint32_t),
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, NULL, 1,
                        &counter_barrier, 0, NULL);

   /* The final counter is six XFB records. Subtract the first round's bytes,
    * then divide by the nontrivial XFB stride: the generated draw must contain
    * exactly three vertices without reading the counter on the CPU. */
   vkCmdBeginRenderPass(command, &begin_render, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, draw_pipeline);
   vkCmdDrawIndirectByteCountEXT(command, 1, 0, counter.buffer, counter_offset,
                                 VERTICES * XFB_STRIDE, XFB_STRIDE);
   vkCmdEndRenderPass(command);
   const VkMemoryBarrier host_barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier, 0,
                        NULL, 0, NULL);
   check(vkEndCommandBuffer(command), "vkEndCommandBuffer");

   if (execute || submit_only) {
      VkFence fence;
      check(vkCreateFence(device,
                          &(VkFenceCreateInfo){
                             .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                          },
                          NULL, &fence),
            "vkCreateFence");
      const VkSubmitInfo submit = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &command,
      };
      check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
      check(vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ull),
            "vkWaitForFences");
      if (execute) {
         unsigned failures = verify_bytes(&vertices, &xfb, &counter, indexed);
         sleep(3);
         failures += verify_bytes(&vertices, &xfb, &counter, indexed);
         if (failures)
            die("exact output, counter, or guard check failed");
      }
      vkDestroyFence(device, fence, NULL);
   }

   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyPipeline(device, draw_pipeline, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyShaderModule(device, fragment_module, NULL);
   vkDestroyShaderModule(device, draw_vertex_module, NULL);
   vkDestroyShaderModule(device, vertex_module, NULL);
   vkDestroyShaderModule(device, geometry_module, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyDescriptorPool(device, dp, NULL);
   vkDestroyDescriptorSetLayout(device, dlayout, NULL);
   vkDestroyBufferView(device, texture_view, NULL);
   destroy_buffer(device, &texture);
   vkDestroyFramebuffer(device, framebuffer, NULL);
   vkDestroyRenderPass(device, render_pass, NULL);
   vkDestroyImageView(device, image_view, NULL);
   vkDestroyImage(device, image, NULL);
   vkFreeMemory(device, image_memory, NULL);
   destroy_buffer(device, &counter);
   destroy_buffer(device, &xfb);
   destroy_buffer(device, &vertices);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);

   if (execute)
      printf(
         "G17P TRANSFORM FEEDBACK %s PASS: two exact guarded captures "
         "and resumed counter stayed exact immediately and after 3 seconds\n",
         rasterized ? "RASTERIZED DIAGNOSTIC" : "");
   else if (submit_only)
      puts(
         "G17P TRANSFORM FEEDBACK SUBMIT PASS: public XFB path reached the UAPI stream validator");
   else
      puts(
         "G17P TRANSFORM FEEDBACK RECORD PASS: public feature and complete native XFB path reached vkEndCommandBuffer");
   return EXIT_SUCCESS;
}
