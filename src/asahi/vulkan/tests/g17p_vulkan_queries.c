/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Strict public Vulkan query fixture for T8140/G17P.  It uses only the query
 * scope advertised by the public device profile: graphics/compute timestamps
 * and precise occlusion.  This graphics fixture does not create a compute
 * pipeline, and pipeline-statistics queries are deliberately not created
 * because the G17P profile does not advertise them.
 *
 * --record-only records and destroys a command buffer without submission.
 * --execute additionally requires G17P_QUERY_EXECUTION_ACK=YES, bounds the
 * fence wait, and checks availability, timestamp ordering, and output guards.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "g17p_wsi_triangle_frag.h"
#include "g17p_wsi_triangle_vert.h"

#define T8140_DEVICE_ID 0x8140
#define IMAGE_WIDTH 512
#define IMAGE_HEIGHT 512
#define EXPECTED_SAMPLES ((uint64_t)IMAGE_WIDTH * IMAGE_HEIGHT)
#define EXECUTION_ACK "G17P_QUERY_EXECUTION_ACK"
#define EXECUTION_ACK_VALUE "YES"
#define FENCE_TIMEOUT_NS UINT64_C(2000000000)
#define OUTPUT_GUARD UINT64_C(0xd15ea5e5c0decafe)

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceFeatures);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceFormatProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkGetDeviceQueue);
VK_FN(vkCreateBuffer);
VK_FN(vkDestroyBuffer);
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkCreateImage);
VK_FN(vkDestroyImage);
VK_FN(vkGetImageMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkFreeMemory);
VK_FN(vkBindBufferMemory);
VK_FN(vkBindImageMemory);
VK_FN(vkMapMemory);
VK_FN(vkUnmapMemory);
VK_FN(vkCreateImageView);
VK_FN(vkDestroyImageView);
VK_FN(vkCreateRenderPass);
VK_FN(vkDestroyRenderPass);
VK_FN(vkCreateShaderModule);
VK_FN(vkDestroyShaderModule);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkDestroyPipelineLayout);
VK_FN(vkCreateGraphicsPipelines);
VK_FN(vkDestroyPipeline);
VK_FN(vkCreateFramebuffer);
VK_FN(vkDestroyFramebuffer);
VK_FN(vkCreateQueryPool);
VK_FN(vkDestroyQueryPool);
VK_FN(vkCreateCommandPool);
VK_FN(vkDestroyCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkEndCommandBuffer);
VK_FN(vkCmdResetQueryPool);
VK_FN(vkCmdWriteTimestamp);
VK_FN(vkCmdBeginRenderPass);
VK_FN(vkCmdEndRenderPass);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindVertexBuffers);
VK_FN(vkCmdDraw);
VK_FN(vkCmdBeginQuery);
VK_FN(vkCmdEndQuery);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);
VK_FN(vkGetQueryPoolResults);

struct output_slot {
   uint64_t value;
   uint64_t availability;
   uint64_t guard;
};

struct timestamp_output {
   uint64_t prefix_guard;
   struct output_slot slots[2];
   uint64_t suffix_guard;
};

struct occlusion_output {
   uint64_t prefix_guard;
   struct output_slot slot;
   uint64_t suffix_guard;
};

static void
die(const char *what)
{
   fprintf(stderr, "g17p-vulkan-queries: %s\n", what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "g17p-vulkan-queries: %s returned %d\n", what, result);
      exit(EXIT_FAILURE);
   }
}

#define LOAD_GLOBAL(name)                                                      \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);         \
      if (!name)                                                               \
         die("loader is missing " #name);                                    \
   } while (0)

#define LOAD_INSTANCE(instance, name)                                          \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);               \
      if (!name)                                                               \
         die("instance is missing " #name);                                  \
   } while (0)

#define LOAD_DEVICE(device, name)                                              \
   do {                                                                        \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                   \
      if (!name)                                                               \
         die("device is missing " #name);                                    \
   } while (0)

static bool
guards_intact(const struct output_slot *slots, uint32_t count,
              uint64_t prefix, uint64_t suffix)
{
   if (prefix != OUTPUT_GUARD || suffix != OUTPUT_GUARD)
      return false;

   for (uint32_t i = 0; i < count; ++i) {
      if (slots[i].guard != OUTPUT_GUARD)
         return false;
   }

   return true;
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

static void
check_unavailable(VkDevice device, VkQueryPool pool, uint32_t query,
                  const char *name)
{
   struct output_slot slot = {
      .value = OUTPUT_GUARD,
      .availability = OUTPUT_GUARD,
      .guard = OUTPUT_GUARD,
   };
   const VkResult result = vkGetQueryPoolResults(
      device, pool, query, 1, sizeof(slot), &slot.value, sizeof(slot),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
   if (result != VK_NOT_READY) {
      fprintf(stderr, "g17p-vulkan-queries: unavailable %s returned %d\n",
              name, result);
      exit(EXIT_FAILURE);
   }
   if (slot.value != OUTPUT_GUARD || slot.availability != 0 ||
       slot.guard != OUTPUT_GUARD)
      die("unavailable query modified a result or guard");
}

static void
check_results(VkDevice device, VkQueryPool timestamps, VkQueryPool occlusion,
              unsigned pass)
{
   struct timestamp_output ts = {
      .prefix_guard = OUTPUT_GUARD,
      .slots = {
         {.value = OUTPUT_GUARD, .availability = OUTPUT_GUARD,
          .guard = OUTPUT_GUARD},
         {.value = OUTPUT_GUARD, .availability = OUTPUT_GUARD,
          .guard = OUTPUT_GUARD},
      },
      .suffix_guard = OUTPUT_GUARD,
   };
   struct occlusion_output oq = {
      .prefix_guard = OUTPUT_GUARD,
      .slot = {.value = OUTPUT_GUARD, .availability = OUTPUT_GUARD,
               .guard = OUTPUT_GUARD},
      .suffix_guard = OUTPUT_GUARD,
   };
   const VkQueryResultFlags flags = VK_QUERY_RESULT_64_BIT |
                                    VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;

   check(vkGetQueryPoolResults(device, timestamps, 0, 2,
                               sizeof(ts) - offsetof(struct timestamp_output,
                                                     slots),
                               &ts.slots[0].value, sizeof(ts.slots[0]), flags),
         "vkGetQueryPoolResults(timestamps)");
   check(vkGetQueryPoolResults(device, occlusion, 0, 1,
                               sizeof(oq) - offsetof(struct occlusion_output,
                                                     slot),
                               &oq.slot.value, sizeof(oq.slot), flags),
         "vkGetQueryPoolResults(occlusion)");

   if (!guards_intact(ts.slots, 2, ts.prefix_guard, ts.suffix_guard) ||
       !guards_intact(&oq.slot, 1, oq.prefix_guard, oq.suffix_guard))
      die("query result write crossed a guarded output boundary");
   if (ts.slots[0].availability != 1 || ts.slots[1].availability != 1 ||
       oq.slot.availability != 1)
      die("submitted query did not become available");
   if (ts.slots[0].value == 0 || ts.slots[1].value <= ts.slots[0].value)
      die("reused timestamp interval is zero or not strictly increasing");
   if (oq.slot.value != EXPECTED_SAMPLES)
      die("reused precise occlusion result is not exact");

   printf("g17p-vulkan-queries: pass %u — timestamps %llu < %llu, "
          "occlusion=%llu; availability and guards exact\n",
          pass,
          (unsigned long long)ts.slots[0].value,
          (unsigned long long)ts.slots[1].value,
          (unsigned long long)oq.slot.value);
}

int
main(int argc, char **argv)
{
   if (argc != 2 || (strcmp(argv[1], "--record-only") != 0 &&
                     strcmp(argv[1], "--execute") != 0)) {
      fprintf(stderr, "usage: %s <--record-only|--execute>\n", argv[0]);
      return EXIT_FAILURE;
   }

   const bool execute = strcmp(argv[1], "--execute") == 0;
   if (execute) {
      const char *ack = getenv(EXECUTION_ACK);
      if (!ack || strcmp(ack, EXECUTION_ACK_VALUE) != 0)
         die("--execute requires " EXECUTION_ACK "=" EXECUTION_ACK_VALUE);
   }

   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());

   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("libvulkan.so.1 has no vkGetInstanceProcAddr");
   LOAD_GLOBAL(vkCreateInstance);

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "g17p-vulkan-queries",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "none",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = VK_API_VERSION_1_0,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance = VK_NULL_HANDLE;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFormatProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   if (!physical_count)
      die("no Vulkan physical devices found");

   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      die("physical-device allocation failed");
   check(vkEnumeratePhysicalDevices(instance, &physical_count,
                                    physical_devices),
         "vkEnumeratePhysicalDevices(list)");

   VkPhysicalDevice physical = VK_NULL_HANDLE;
   VkPhysicalDeviceProperties properties = {0};
   VkPhysicalDeviceFeatures features = {0};
   uint32_t queue_family = UINT32_MAX;
   for (uint32_t p = 0; p < physical_count && !physical; ++p) {
      VkPhysicalDeviceProperties candidate;
      VkPhysicalDeviceFeatures candidate_features;
      vkGetPhysicalDeviceProperties(physical_devices[p], &candidate);
      if (candidate.deviceID != T8140_DEVICE_ID)
         continue;
      vkGetPhysicalDeviceFeatures(physical_devices[p], &candidate_features);

      if (!candidate.limits.timestampComputeAndGraphics ||
          candidate.limits.timestampPeriod != 1.0f ||
          !candidate_features.occlusionQueryPrecise ||
          candidate_features.pipelineStatisticsQuery)
         die("T8140 query advertisement differs from the bounded profile");

      uint32_t queue_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[p],
                                                &queue_count, NULL);
      VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
      if (!queues)
         die("queue-family allocation failed");
      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[p],
                                                &queue_count, queues);
      for (uint32_t q = 0; q < queue_count; ++q) {
         const VkQueueFlags required = VK_QUEUE_GRAPHICS_BIT |
                                       VK_QUEUE_COMPUTE_BIT |
                                       VK_QUEUE_TRANSFER_BIT;
         if (queues[q].queueCount > 0 &&
             (queues[q].queueFlags & required) == required &&
             !(queues[q].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) &&
             queues[q].timestampValidBits == 64) {
            physical = physical_devices[p];
            properties = candidate;
            features = candidate_features;
            queue_family = q;
            break;
         }
      }
      free(queues);
   }
   free(physical_devices);
   if (!physical)
      die("no T8140/G17P queue matches the public query profile");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkPhysicalDeviceFeatures enabled_features = {
      .occlusionQueryPrecise = features.occlusionQueryPrecise,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .pEnabledFeatures = &enabled_features,
   };
   VkDevice device = VK_NULL_HANDLE;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCreateBuffer);
   LOAD_DEVICE(device, vkDestroyBuffer);
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkCreateImage);
   LOAD_DEVICE(device, vkDestroyImage);
   LOAD_DEVICE(device, vkGetImageMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkFreeMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkBindImageMemory);
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkUnmapMemory);
   LOAD_DEVICE(device, vkCreateImageView);
   LOAD_DEVICE(device, vkDestroyImageView);
   LOAD_DEVICE(device, vkCreateRenderPass);
   LOAD_DEVICE(device, vkDestroyRenderPass);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkDestroyShaderModule);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkDestroyPipelineLayout);
   LOAD_DEVICE(device, vkCreateGraphicsPipelines);
   LOAD_DEVICE(device, vkDestroyPipeline);
   LOAD_DEVICE(device, vkCreateFramebuffer);
   LOAD_DEVICE(device, vkDestroyFramebuffer);
   LOAD_DEVICE(device, vkCreateQueryPool);
   LOAD_DEVICE(device, vkDestroyQueryPool);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkDestroyCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCmdResetQueryPool);
   LOAD_DEVICE(device, vkCmdWriteTimestamp);
   LOAD_DEVICE(device, vkCmdBeginRenderPass);
   LOAD_DEVICE(device, vkCmdEndRenderPass);
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdBindVertexBuffers);
   LOAD_DEVICE(device, vkCmdDraw);
   LOAD_DEVICE(device, vkCmdBeginQuery);
   LOAD_DEVICE(device, vkCmdEndQuery);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkDestroyFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);
   LOAD_DEVICE(device, vkGetQueryPoolResults);

   const VkQueryPoolCreateInfo timestamp_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = 3,
   };
   const VkQueryPoolCreateInfo occlusion_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_OCCLUSION,
      .queryCount = 2,
   };
   VkQueryPool timestamps = VK_NULL_HANDLE;
   VkQueryPool occlusion = VK_NULL_HANDLE;
   check(vkCreateQueryPool(device, &timestamp_info, NULL, &timestamps),
         "vkCreateQueryPool(timestamp)");
   check(vkCreateQueryPool(device, &occlusion_info, NULL, &occlusion),
         "vkCreateQueryPool(occlusion)");

   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   VkFormatProperties format_properties;
   vkGetPhysicalDeviceFormatProperties(physical, VK_FORMAT_R8G8B8A8_UNORM,
                                       &format_properties);
   if (!(format_properties.optimalTilingFeatures &
         VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
      die("RGBA8 optimal images lack color-attachment support");

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
   VkBuffer vertex_buffer = VK_NULL_HANDLE;
   check(vkCreateBuffer(device, &vertex_info, NULL, &vertex_buffer),
         "vkCreateBuffer(vertex)");
   VkMemoryRequirements vertex_requirements;
   vkGetBufferMemoryRequirements(device, vertex_buffer, &vertex_requirements);
   const VkMemoryAllocateInfo vertex_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = vertex_requirements.size,
      .memoryTypeIndex = find_memory_type(
         &memory_properties, vertex_requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory vertex_memory = VK_NULL_HANDLE;
   check(vkAllocateMemory(device, &vertex_memory_info, NULL, &vertex_memory),
         "vkAllocateMemory(vertex)");
   check(vkBindBufferMemory(device, vertex_buffer, vertex_memory, 0),
         "vkBindBufferMemory(vertex)");
   void *vertex_map = NULL;
   check(vkMapMemory(device, vertex_memory, 0, sizeof(vertices), 0,
                     &vertex_map),
         "vkMapMemory(vertex)");
   memcpy(vertex_map, vertices, sizeof(vertices));
   vkUnmapMemory(device, vertex_memory);

   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image = VK_NULL_HANDLE;
   check(vkCreateImage(device, &image_info, NULL, &image), "vkCreateImage");
   VkMemoryRequirements image_requirements;
   vkGetImageMemoryRequirements(device, image, &image_requirements);
   const VkMemoryAllocateInfo image_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = image_requirements.size,
      .memoryTypeIndex = find_memory_type(
         &memory_properties, image_requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory image_memory = VK_NULL_HANDLE;
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
   VkImageView view = VK_NULL_HANDLE;
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
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
   };
   VkRenderPass render_pass = VK_NULL_HANDLE;
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
   VkShaderModule vert_module = VK_NULL_HANDLE;
   VkShaderModule frag_module = VK_NULL_HANDLE;
   check(vkCreateShaderModule(device, &vert_module_info, NULL, &vert_module),
         "vkCreateShaderModule(vertex)");
   check(vkCreateShaderModule(device, &frag_module_info, NULL, &frag_module),
         "vkCreateShaderModule(fragment)");

   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
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
   VkPipeline pipeline = VK_NULL_HANDLE;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipeline),
         "vkCreateGraphicsPipelines");

   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = 1,
      .pAttachments = &view,
      .width = IMAGE_WIDTH,
      .height = IMAGE_HEIGHT,
      .layers = 1,
   };
   VkFramebuffer framebuffer = VK_NULL_HANDLE;
   check(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer),
         "vkCreateFramebuffer");

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool command_pool = VK_NULL_HANDLE;
   check(vkCreateCommandPool(device, &pool_info, NULL, &command_pool),
         "vkCreateCommandPool");

   const VkCommandBufferAllocateInfo allocate_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 2,
   };
   VkCommandBuffer command_buffers[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
   check(vkAllocateCommandBuffers(device, &allocate_info, command_buffers),
         "vkAllocateCommandBuffers");

   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   const VkClearValue clear = {.color = {{0.0f, 0.0f, 0.0f, 1.0f}}};
   const VkRenderPassBeginInfo render_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {.extent = {IMAGE_WIDTH, IMAGE_HEIGHT}},
      .clearValueCount = 1,
      .pClearValues = &clear,
   };
   const VkDeviceSize vertex_offset = 0;
   for (unsigned pass = 0; pass < 2; ++pass) {
      VkCommandBuffer command_buffer = command_buffers[pass];
      check(vkBeginCommandBuffer(command_buffer, &begin_info),
            "vkBeginCommandBuffer");
      vkCmdResetQueryPool(command_buffer, timestamps, 0, 3);
      vkCmdResetQueryPool(command_buffer, occlusion, 0, 2);
      vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          timestamps, 0);
      vkCmdBeginRenderPass(command_buffer, &render_begin,
                           VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipeline);
      vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer,
                             &vertex_offset);
      vkCmdBeginQuery(command_buffer, occlusion, 0,
                      VK_QUERY_CONTROL_PRECISE_BIT);
      vkCmdDraw(command_buffer, 3, 1, 0, 0);
      vkCmdEndQuery(command_buffer, occlusion, 0);
      vkCmdEndRenderPass(command_buffer);
      vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                          timestamps, 1);
      check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
   }

   printf("g17p-vulkan-queries: %s (device %#x), queue %u: "
          "timestamps=graphics+compute/64-bit/1ns, precise-occlusion=yes, "
          "pipeline-statistics=no\n",
          properties.deviceName, properties.deviceID, queue_family);

   if (execute) {
      VkQueue queue = VK_NULL_HANDLE;
      vkGetDeviceQueue(device, queue_family, 0, &queue);
      const VkFenceCreateInfo fence_info = {
         .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      };
      for (unsigned pass = 0; pass < 2; ++pass) {
         VkFence fence = VK_NULL_HANDLE;
         check(vkCreateFence(device, &fence_info, NULL, &fence),
               "vkCreateFence");
         const VkSubmitInfo submit_info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffers[pass],
         };
         check(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
         VkResult wait = vkWaitForFences(device, 1, &fence, VK_TRUE,
                                         FENCE_TIMEOUT_NS);
         if (wait == VK_TIMEOUT)
            die("bounded two-second fence wait timed out");
         check(wait, "vkWaitForFences");
         check_results(device, timestamps, occlusion, pass);
         vkDestroyFence(device, fence, NULL);
      }
      check_unavailable(device, timestamps, 2, "timestamp");
      check_unavailable(device, occlusion, 1, "occlusion");
      puts("g17p-vulkan-queries: reset/reuse PASS; unavailable results and "
           "availability exact");
   } else {
      puts("g17p-vulkan-queries: record-only PASS — no queue submission or "
           "query-result read performed");
   }

   vkDestroyCommandPool(device, command_pool, NULL);
   vkDestroyFramebuffer(device, framebuffer, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, pipeline_layout, NULL);
   vkDestroyShaderModule(device, frag_module, NULL);
   vkDestroyShaderModule(device, vert_module, NULL);
   vkDestroyRenderPass(device, render_pass, NULL);
   vkDestroyImageView(device, view, NULL);
   vkDestroyImage(device, image, NULL);
   vkFreeMemory(device, image_memory, NULL);
   vkDestroyBuffer(device, vertex_buffer, NULL);
   vkFreeMemory(device, vertex_memory, NULL);
   vkDestroyQueryPool(device, occlusion, NULL);
   vkDestroyQueryPool(device, timestamps, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return EXIT_SUCCESS;
}
