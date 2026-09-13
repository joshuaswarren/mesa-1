/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * vkQuake's world is produced entirely by compute.  One primary command
 * buffer (PCBX_UPDATE_LIGHTMAPS) dispatches two kernels per frame:
 *
 *   1. the brush-culling kernel, which atomically accumulates each indirect
 *      command's indexCount and writes the triangle indices the world draws
 *      from.  Every world surface -- and the sky -- is drawn by
 *      vkCmdDrawIndexedIndirect from that buffer, so a zero indexCount means
 *      no world geometry is submitted at all;
 *   2. the lightmap kernel, which imageStores an rgba8 lightmap that
 *      world.frag multiplies the base texture by.  A zero lightmap renders
 *      every world surface black.
 *
 * Alias models, particles, the view model and the HUD are recorded into
 * different command buffers and carry their own light, so either failure
 * leaves them visible while the world goes black.  That is the signature
 * this oracle is built to attribute, and the two kernels are reproduced at
 * their real resource shape rather than as smoke tests: the G17P native
 * compute admission gate in hk_cmd_dispatch.c measures preamble, scratch,
 * texture and sampler state and root set count, and a two-buffer kernel
 * records where a ten-binding one may not.
 *
 * Under the DRM shim with G17P_VKQUAKE_COMPUTE_RECORD_ONLY=1 this qualifies
 * pipeline creation and the record-time admission gate.  On hardware it also
 * submits, then compares the generated indirect commands, the generated
 * index buffer and every lightmap texel against exact expected values.
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

/* 1 = the brush-culling kernel alone, 2 = the lightmap kernel alone,
 * 3 = both in one command buffer, as vkQuake records them. */
#ifndef G17P_VKQ_STAGE
#define G17P_VKQ_STAGE 3
#endif

#if G17P_VKQ_STAGE < 1 || G17P_VKQ_STAGE > 3
#error "select a valid vkQuake world compute stage"
#endif

#define G17P_VKQ_GEN      (G17P_VKQ_STAGE == 1 || G17P_VKQ_STAGE == 3)
#define G17P_VKQ_LIGHTMAP (G17P_VKQ_STAGE == 2 || G17P_VKQ_STAGE == 3)

#if G17P_VKQ_STAGE == 1
#define PROBE_NAME "g17p-vulkan-vkquake-world-gen"
#elif G17P_VKQ_STAGE == 2
#define PROBE_NAME "g17p-vulkan-vkquake-lightmap"
#else
#define PROBE_NAME "g17p-vulkan-vkquake-world-compute"
#endif

#if G17P_VKQ_GEN
#include "g17p_vulkan_vkquake_world_gen_comp.h"
#endif
#if G17P_VKQ_LIGHTMAP
#include "g17p_vulkan_vkquake_lightmap_comp.h"
#endif

#define T8140_DEVICE_ID 0x8140

/* Distinguishable from zero and from any value either kernel produces. */
#define SENTINEL UINT32_C(0xA5A5A5A5)

#define LIGHTMAP_DIM   8
#define LIGHTMAP_BYTES (LIGHTMAP_DIM * LIGHTMAP_DIM * 4)

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
VK_FN(vkCreateDescriptorSetLayout);
VK_FN(vkDestroyDescriptorSetLayout);
VK_FN(vkCreateDescriptorPool);
VK_FN(vkDestroyDescriptorPool);
VK_FN(vkAllocateDescriptorSets);
VK_FN(vkUpdateDescriptorSets);
VK_FN(vkCreateShaderModule);
VK_FN(vkDestroyShaderModule);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkDestroyPipelineLayout);
VK_FN(vkCreateComputePipelines);
VK_FN(vkDestroyPipeline);
VK_FN(vkCreateCommandPool);
VK_FN(vkDestroyCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindDescriptorSets);
VK_FN(vkCmdPushConstants);
VK_FN(vkCmdDispatch);
VK_FN(vkCmdPipelineBarrier);
VK_FN(vkEndCommandBuffer);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);
#if G17P_VKQ_LIGHTMAP
VK_FN(vkCreateImage);
VK_FN(vkDestroyImage);
VK_FN(vkGetImageMemoryRequirements);
VK_FN(vkBindImageMemory);
VK_FN(vkCreateImageView);
VK_FN(vkDestroyImageView);
VK_FN(vkCmdCopyBufferToImage);
VK_FN(vkCmdCopyImageToBuffer);
#endif

static void
die(const char *what)
{
   fprintf(stderr, PROBE_NAME ": %s\n", what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, PROBE_NAME ": %s returned %d\n", what, (int)result);
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

static VkPhysicalDeviceMemoryProperties memory_properties;
static VkDevice device;

static uint32_t
find_memory_type(uint32_t allowed, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
      if ((allowed & (1u << i)) &&
          (memory_properties.memoryTypes[i].propertyFlags & required) ==
             required)
         return i;
   }

   die("host-visible coherent memory is unavailable");
   return UINT32_MAX;
}

struct mapped_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   void *map;
   VkDeviceSize size;
};

static struct mapped_buffer
create_mapped_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                     const char *what)
{
   struct mapped_buffer out = {.size = size};
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   check(vkCreateBuffer(device, &info, NULL, &out.buffer), what);

   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, out.buffer, &requirements);
   const VkMemoryAllocateInfo allocate = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex =
         find_memory_type(requirements.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   check(vkAllocateMemory(device, &allocate, NULL, &out.memory), what);
   check(vkBindBufferMemory(device, out.buffer, out.memory, 0), what);
   check(vkMapMemory(device, out.memory, 0, VK_WHOLE_SIZE, 0, &out.map),
         what);
   memset(out.map, 0, size);
   return out;
}

/* ------------------------------------------------------------------ */
/* The brush-culling kernel's inputs, and what it must produce.        */
/* ------------------------------------------------------------------ */
#if G17P_VKQ_GEN

struct draw_indirect_command {
   uint32_t indexCount;
   uint32_t instanceCount;
   uint32_t firstIndex;
   int32_t vertexOffset;
   uint32_t firstInstance;
};

struct gen_surface {
   float normal_x, normal_y, normal_z, dist;
   uint32_t packed_tex_edgecount;
   uint32_t vbo_offset;
   uint32_t packed_lightstyles;
   uint32_t packed_light_st;
};

struct gen_push_constants {
   uint32_t num_draws;
   uint32_t first_draw;
   uint32_t visibility_offset;
   float vieworg_x, vieworg_y, vieworg_z;
   uint32_t instance_base;
};

#define GEN_SURFACE_COUNT 4
#define GEN_DRAW_COUNT    2
#define GEN_INDEX_WORDS   64
#define GEN_BACKFACE_BIT  UINT32_C(0x8000)

/* firstIndex of the two commands.  Both are non-zero so a kernel that
 * ignores firstIndex writes its indices in the wrong place instead of
 * accidentally landing on the expected window. */
#define GEN_FIRST_INDEX_0 16
#define GEN_FIRST_INDEX_1 40

static uint32_t
gen_pack_surface(uint32_t tex_idx, bool backface, uint32_t numedges)
{
   return (tex_idx & 0x7FFFu) | (backface ? GEN_BACKFACE_BIT : 0) |
          (numedges << 16);
}

/* The triangle fan the kernel writes for one surface. */
static unsigned
gen_expected_fan(uint32_t firstvert, uint32_t numedges, uint32_t *out)
{
   unsigned n = 0;
   for (uint32_t i = 2; i < numedges; ++i) {
      out[n++] = firstvert;
      out[n++] = firstvert + i - 1;
      out[n++] = firstvert + i;
   }
   return n;
}
#endif

/* ------------------------------------------------------------------ */
/* The lightmap kernel's inputs, and what it must produce.             */
/* ------------------------------------------------------------------ */
#if G17P_VKQ_LIGHTMAP

struct lightmap_push_constants {
   uint32_t num_dlights;
   uint32_t lightmap_width;
   uint32_t x_workgroup_offset;
   uint32_t y_workgroup_offset;
   uint32_t dlights_only;
   uint32_t num_cached_dlights;
   uint32_t submodel_transforms_offset;
   uint32_t prev_submodel_transforms_offset;
   float view_pos_x, view_pos_y, view_pos_z;
};

/* Two lightstyles contribute, at scales 1/2 and 1/4.  Every product is an
 * exact multiple of 1/255, so the expected texel is exact in UNORM8 and no
 * tolerance is needed. */
#define LM_STYLE0_TEXEL_R 0x40
#define LM_STYLE0_TEXEL_G 0x80
#define LM_STYLE0_TEXEL_B 0xC0
#define LM_STYLE1_TEXEL_R 0x10
#define LM_STYLE1_TEXEL_G 0x20
#define LM_STYLE1_TEXEL_B 0x30

#define LM_EXPECTED_R (LM_STYLE0_TEXEL_R / 2 + LM_STYLE1_TEXEL_R / 4)
#define LM_EXPECTED_G (LM_STYLE0_TEXEL_G / 2 + LM_STYLE1_TEXEL_G / 4)
#define LM_EXPECTED_B (LM_STYLE0_TEXEL_B / 2 + LM_STYLE1_TEXEL_B / 4)
#endif

int
main(void)
{
   const bool record_only =
      getenv("G17P_VKQUAKE_COMPUTE_RECORD_ONLY") != NULL;

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
      .pApplicationName = PROBE_NAME,
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &application_info,
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
   VkPhysicalDeviceProperties properties;
   for (uint32_t i = 0; i < physical_count; ++i) {
      vkGetPhysicalDeviceProperties(physical_devices[i], &properties);
      if (properties.deviceID == T8140_DEVICE_ID) {
         physical = physical_devices[i];
         break;
      }
   }
   free(physical_devices);
   if (physical == VK_NULL_HANDLE)
      die("T8140/G17P Vulkan device was not found");

   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   uint32_t queue_family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_family_count,
                                            NULL);
   VkQueueFamilyProperties *queue_families =
      calloc(queue_family_count, sizeof(*queue_families));
   if (!queue_families)
      die("out of host memory");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &queue_family_count,
                                            queue_families);
   uint32_t queue_family = UINT32_MAX;
   for (uint32_t i = 0; i < queue_family_count; ++i) {
      if (queue_families[i].queueCount != 0) {
         queue_family = i;
         break;
      }
   }
   free(queue_families);
   if (queue_family == UINT32_MAX)
      die("device has no usable queue family");

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
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCreateBuffer);
   LOAD_DEVICE(device, vkDestroyBuffer);
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkFreeMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkCreateDescriptorSetLayout);
   LOAD_DEVICE(device, vkDestroyDescriptorSetLayout);
   LOAD_DEVICE(device, vkCreateDescriptorPool);
   LOAD_DEVICE(device, vkDestroyDescriptorPool);
   LOAD_DEVICE(device, vkAllocateDescriptorSets);
   LOAD_DEVICE(device, vkUpdateDescriptorSets);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkDestroyShaderModule);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkDestroyPipelineLayout);
   LOAD_DEVICE(device, vkCreateComputePipelines);
   LOAD_DEVICE(device, vkDestroyPipeline);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkDestroyCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdBindDescriptorSets);
   LOAD_DEVICE(device, vkCmdPushConstants);
   LOAD_DEVICE(device, vkCmdDispatch);
   LOAD_DEVICE(device, vkCmdPipelineBarrier);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkDestroyFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);
#if G17P_VKQ_LIGHTMAP
   LOAD_DEVICE(device, vkCreateImage);
   LOAD_DEVICE(device, vkDestroyImage);
   LOAD_DEVICE(device, vkGetImageMemoryRequirements);
   LOAD_DEVICE(device, vkBindImageMemory);
   LOAD_DEVICE(device, vkCreateImageView);
   LOAD_DEVICE(device, vkDestroyImageView);
   LOAD_DEVICE(device, vkCmdCopyBufferToImage);
   LOAD_DEVICE(device, vkCmdCopyImageToBuffer);
#endif

   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);

   printf(PROBE_NAME ": device %s, queue %u\n", properties.deviceName,
          queue_family);

   VkDescriptorPoolSize pool_sizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 2,
      .poolSizeCount = 4,
      .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool descriptor_pool;
   check(vkCreateDescriptorPool(device, &pool_info, NULL, &descriptor_pool),
         "vkCreateDescriptorPool");

   /* ---------------------------------------------------------------- */
#if G17P_VKQ_GEN
   VkDescriptorSetLayoutBinding gen_bindings[6];
   for (unsigned i = 0; i < 6; ++i) {
      gen_bindings[i] = (VkDescriptorSetLayoutBinding){
         .binding = i,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      };
   }
   const VkDescriptorSetLayoutCreateInfo gen_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 6,
      .pBindings = gen_bindings,
   };
   VkDescriptorSetLayout gen_set_layout;
   check(vkCreateDescriptorSetLayout(device, &gen_layout_info, NULL,
                                     &gen_set_layout),
         "vkCreateDescriptorSetLayout(brush culling)");

   const VkPushConstantRange gen_push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(struct gen_push_constants),
   };
   const VkPipelineLayoutCreateInfo gen_pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &gen_set_layout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &gen_push_range,
   };
   VkPipelineLayout gen_pipeline_layout;
   check(vkCreatePipelineLayout(device, &gen_pipeline_layout_info, NULL,
                                &gen_pipeline_layout),
         "vkCreatePipelineLayout(brush culling)");

   const VkShaderModuleCreateInfo gen_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_vkquake_world_gen_comp_spv),
      .pCode = g17p_vulkan_vkquake_world_gen_comp_spv,
   };
   VkShaderModule gen_module;
   check(vkCreateShaderModule(device, &gen_module_info, NULL, &gen_module),
         "vkCreateShaderModule(brush culling)");

   const VkComputePipelineCreateInfo gen_pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = gen_module,
         .pName = "main",
      },
      .layout = gen_pipeline_layout,
   };
   VkPipeline gen_pipeline;
   check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                  &gen_pipeline_info, NULL, &gen_pipeline),
         "vkCreateComputePipelines(brush culling)");

   /* Inputs.  Two of the four surfaces survive: surface 2 fails the
    * visibility bitmask and surface 3 fails the backface test.  The two
    * survivors share one indirect command, so the atomicAdd is contended
    * and its return value decides where each fan lands. */
   struct mapped_buffer indirect = create_mapped_buffer(
      GEN_DRAW_COUNT * sizeof(struct draw_indirect_command),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "indirect command buffer");
   struct draw_indirect_command *commands = indirect.map;
   commands[0] = (struct draw_indirect_command){
      .indexCount = 0,
      .instanceCount = 1,
      .firstIndex = GEN_FIRST_INDEX_0,
      .vertexOffset = 0,
      .firstInstance = 0,
   };
   commands[1] = (struct draw_indirect_command){
      .indexCount = 0,
      .instanceCount = 1,
      .firstIndex = GEN_FIRST_INDEX_1,
      .vertexOffset = 0,
      .firstInstance = 0,
   };

   struct mapped_buffer surfaces = create_mapped_buffer(
      GEN_SURFACE_COUNT * sizeof(struct gen_surface),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "surfaces buffer");
   struct gen_surface *surface = surfaces.map;
   for (unsigned i = 0; i < GEN_SURFACE_COUNT; ++i) {
      surface[i] = (struct gen_surface){
         .normal_x = 0.0f,
         .normal_y = 0.0f,
         .normal_z = 1.0f,
         /* vieworg is the origin, so dot(normal, vieworg) == 0 > dist and a
          * front-facing surface is kept. */
         .dist = -1.0f,
      };
   }
   surface[0].packed_tex_edgecount = gen_pack_surface(0, false, 4);
   surface[0].vbo_offset = 100;
   surface[1].packed_tex_edgecount = gen_pack_surface(0, false, 5);
   surface[1].vbo_offset = 200;
   surface[2].packed_tex_edgecount = gen_pack_surface(1, false, 4);
   surface[2].vbo_offset = 300;
   surface[3].packed_tex_edgecount = gen_pack_surface(1, true, 4);
   surface[3].vbo_offset = 400;

   struct mapped_buffer visibility = create_mapped_buffer(
      sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      "visibility buffer");
   ((uint32_t *)visibility.map)[0] = 0xBu; /* surfaces 0, 1 and 3 */

   struct mapped_buffer indices = create_mapped_buffer(
      GEN_INDEX_WORDS * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      "index buffer");
   for (unsigned i = 0; i < GEN_INDEX_WORDS; ++i)
      ((uint32_t *)indices.map)[i] = SENTINEL;

   struct mapped_buffer submodels = create_mapped_buffer(
      GEN_SURFACE_COUNT * sizeof(uint32_t),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "surface submodels buffer");
   struct mapped_buffer instances = create_mapped_buffer(
      4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      "bmodel instances buffer");

   const VkDescriptorSetAllocateInfo gen_set_allocate = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &gen_set_layout,
   };
   VkDescriptorSet gen_set;
   check(vkAllocateDescriptorSets(device, &gen_set_allocate, &gen_set),
         "vkAllocateDescriptorSets(brush culling)");

   const VkBuffer gen_buffers[6] = {
      indirect.buffer, surfaces.buffer, visibility.buffer,
      indices.buffer,  submodels.buffer, instances.buffer,
   };
   VkDescriptorBufferInfo gen_buffer_infos[6];
   VkWriteDescriptorSet gen_writes[6];
   for (unsigned i = 0; i < 6; ++i) {
      gen_buffer_infos[i] = (VkDescriptorBufferInfo){
         .buffer = gen_buffers[i],
         .range = VK_WHOLE_SIZE,
      };
      gen_writes[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = gen_set,
         .dstBinding = i,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &gen_buffer_infos[i],
      };
   }
   vkUpdateDescriptorSets(device, 6, gen_writes, 0, NULL);
#endif /* G17P_VKQ_GEN */

   /* ---------------------------------------------------------------- */
#if G17P_VKQ_LIGHTMAP
   const VkDescriptorSetLayoutBinding lm_bindings[] = {
      {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
      {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
      {2, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
      {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
      {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
      {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
      {6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
      {7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
      {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
      {9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
   };
   const VkDescriptorSetLayoutCreateInfo lm_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 10,
      .pBindings = lm_bindings,
   };
   VkDescriptorSetLayout lm_set_layout;
   check(vkCreateDescriptorSetLayout(device, &lm_layout_info, NULL,
                                     &lm_set_layout),
         "vkCreateDescriptorSetLayout(lightmap)");

   const VkPushConstantRange lm_push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(struct lightmap_push_constants),
   };
   const VkPipelineLayoutCreateInfo lm_pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &lm_set_layout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &lm_push_range,
   };
   VkPipelineLayout lm_pipeline_layout;
   check(vkCreatePipelineLayout(device, &lm_pipeline_layout_info, NULL,
                                &lm_pipeline_layout),
         "vkCreatePipelineLayout(lightmap)");

   const VkShaderModuleCreateInfo lm_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_vkquake_lightmap_comp_spv),
      .pCode = g17p_vulkan_vkquake_lightmap_comp_spv,
   };
   VkShaderModule lm_module;
   check(vkCreateShaderModule(device, &lm_module_info, NULL, &lm_module),
         "vkCreateShaderModule(lightmap)");

   const VkBool32 scaled_lm = VK_FALSE;
   const VkSpecializationMapEntry lm_spec_entry = {
      .constantID = 0,
      .offset = 0,
      .size = sizeof(VkBool32),
   };
   const VkSpecializationInfo lm_spec = {
      .mapEntryCount = 1,
      .pMapEntries = &lm_spec_entry,
      .dataSize = sizeof(scaled_lm),
      .pData = &scaled_lm,
   };
   const VkComputePipelineCreateInfo lm_pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = lm_module,
         .pName = "main",
         .pSpecializationInfo = &lm_spec,
      },
      .layout = lm_pipeline_layout,
   };
   VkPipeline lm_pipeline;
   check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1,
                                  &lm_pipeline_info, NULL, &lm_pipeline),
         "vkCreateComputePipelines(lightmap)");

   /* The lightmap the world is modulated by. */
   const VkImageCreateInfo lm_image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {LIGHTMAP_DIM, LIGHTMAP_DIM, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage lm_image;
   check(vkCreateImage(device, &lm_image_info, NULL, &lm_image),
         "vkCreateImage(lightmap)");
   VkMemoryRequirements lm_requirements;
   vkGetImageMemoryRequirements(device, lm_image, &lm_requirements);
   const VkMemoryAllocateInfo lm_allocate = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = lm_requirements.size,
      .memoryTypeIndex = find_memory_type(lm_requirements.memoryTypeBits, 0),
   };
   VkDeviceMemory lm_memory;
   check(vkAllocateMemory(device, &lm_allocate, NULL, &lm_memory),
         "vkAllocateMemory(lightmap)");
   check(vkBindImageMemory(device, lm_image, lm_memory, 0),
         "vkBindImageMemory(lightmap)");
   const VkImageViewCreateInfo lm_view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = lm_image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkImageView lm_view;
   check(vkCreateImageView(device, &lm_view_info, NULL, &lm_view),
         "vkCreateImageView(lightmap)");

   /* The static lightmap inputs: one R32_UINT surface index image and three
    * RGBA8 lightstyle images. */
   struct source_image {
      VkImage image;
      VkDeviceMemory memory;
      VkImageView view;
   };
   struct source_image sources[4];
   const VkFormat source_formats[4] = {
      VK_FORMAT_R32_UINT,
      VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_R8G8B8A8_UNORM,
   };
   for (unsigned i = 0; i < 4; ++i) {
      VkImageCreateInfo info = lm_image_info;
      info.format = source_formats[i];
      info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      check(vkCreateImage(device, &info, NULL, &sources[i].image),
            "vkCreateImage(lightmap source)");
      VkMemoryRequirements requirements;
      vkGetImageMemoryRequirements(device, sources[i].image, &requirements);
      const VkMemoryAllocateInfo allocate = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = requirements.size,
         .memoryTypeIndex = find_memory_type(requirements.memoryTypeBits, 0),
      };
      check(vkAllocateMemory(device, &allocate, NULL, &sources[i].memory),
            "vkAllocateMemory(lightmap source)");
      check(vkBindImageMemory(device, sources[i].image, sources[i].memory, 0),
            "vkBindImageMemory(lightmap source)");
      VkImageViewCreateInfo view_info = lm_view_info;
      view_info.image = sources[i].image;
      view_info.format = source_formats[i];
      check(vkCreateImageView(device, &view_info, NULL, &sources[i].view),
            "vkCreateImageView(lightmap source)");
   }

   struct mapped_buffer upload = create_mapped_buffer(
      4 * LIGHTMAP_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "upload buffer");
   uint32_t *surface_index_texels = upload.map;
   for (unsigned i = 0; i < LIGHTMAP_DIM * LIGHTMAP_DIM; ++i)
      surface_index_texels[i] = 0;
   uint8_t *style_texels = (uint8_t *)upload.map + LIGHTMAP_BYTES;
   for (unsigned i = 0; i < LIGHTMAP_DIM * LIGHTMAP_DIM; ++i) {
      style_texels[4 * i + 0] = LM_STYLE0_TEXEL_R;
      style_texels[4 * i + 1] = LM_STYLE0_TEXEL_G;
      style_texels[4 * i + 2] = LM_STYLE0_TEXEL_B;
      style_texels[4 * i + 3] = 0;
   }
   uint8_t *style1_texels = (uint8_t *)upload.map + 2 * LIGHTMAP_BYTES;
   for (unsigned i = 0; i < LIGHTMAP_DIM * LIGHTMAP_DIM; ++i) {
      style1_texels[4 * i + 0] = LM_STYLE1_TEXEL_R;
      style1_texels[4 * i + 1] = LM_STYLE1_TEXEL_G;
      style1_texels[4 * i + 2] = LM_STYLE1_TEXEL_B;
      style1_texels[4 * i + 3] = 0;
   }
   memset((uint8_t *)upload.map + 3 * LIGHTMAP_BYTES, 0, LIGHTMAP_BYTES);

   /* surfaces[0] names lightstyles 0 and 1 and terminates with 255. */
   struct mapped_buffer lm_surfaces = create_mapped_buffer(
      8 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      "lightmap surfaces buffer");
   uint32_t *lm_surface = lm_surfaces.map;
   lm_surface[0] = 0;                     /* normal_x */
   lm_surface[1] = 0;                     /* normal_y */
   lm_surface[2] = 0x3F800000;            /* normal_z = 1.0f */
   lm_surface[3] = 0;                     /* dist */
   lm_surface[4] = 0;                     /* packed_tex_edgecount */
   lm_surface[5] = 0;                     /* vbo_offset */
   lm_surface[6] = 0xFFFF0100u;           /* styles 0, 1, then 255 */
   lm_surface[7] = 0;                     /* packed_light_st */

   struct mapped_buffer lm_bounds = create_mapped_buffer(
      8 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      "workgroup bounds buffer");
   ((uint32_t *)lm_bounds.map)[6] = 0xFFFFFFFFu;

   struct mapped_buffer lm_scales = create_mapped_buffer(
      16 * 4 * sizeof(float), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      "lightstyle scales buffer");
   float *scales = lm_scales.map;
   scales[0] = 0.5f;  /* style 0 */
   scales[1] = 0.25f; /* style 1 */

   struct mapped_buffer lm_lights = create_mapped_buffer(
      64 * 8 * sizeof(float), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      "lights buffer");
   struct mapped_buffer lm_verts = create_mapped_buffer(
      4 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      "world verts buffer");
   struct mapped_buffer lm_submodels = create_mapped_buffer(
      4 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      "lightmap submodels buffer");
   struct mapped_buffer lm_transforms = create_mapped_buffer(
      12 * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      "submodel transforms buffer");

   struct mapped_buffer readback = create_mapped_buffer(
      LIGHTMAP_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT, "readback buffer");
   memset(readback.map, 0x5A, LIGHTMAP_BYTES);

   const VkDescriptorSetAllocateInfo lm_set_allocate = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &lm_set_layout,
   };
   VkDescriptorSet lm_set;
   check(vkAllocateDescriptorSets(device, &lm_set_allocate, &lm_set),
         "vkAllocateDescriptorSets(lightmap)");

   const VkDescriptorImageInfo lm_storage_info = {
      .imageView = lm_view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   const VkDescriptorImageInfo lm_surface_index_info = {
      .imageView = sources[0].view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   const VkDescriptorImageInfo lm_style_infos[3] = {
      {.imageView = sources[1].view,
       .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
      {.imageView = sources[2].view,
       .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
      {.imageView = sources[3].view,
       .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
   };
   const VkBuffer lm_buffers[] = {
      lm_surfaces.buffer,  lm_bounds.buffer,  lm_scales.buffer,
      lm_lights.buffer,    lm_verts.buffer,   lm_submodels.buffer,
      lm_transforms.buffer,
   };
   const uint32_t lm_buffer_bindings[] = {3, 4, 5, 6, 7, 8, 9};
   const VkDescriptorType lm_buffer_types[] = {
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
   };
   VkDescriptorBufferInfo lm_buffer_infos[7];
   VkWriteDescriptorSet lm_writes[10];
   lm_writes[0] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = lm_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &lm_storage_info,
   };
   lm_writes[1] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = lm_set,
      .dstBinding = 1,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .pImageInfo = &lm_surface_index_info,
   };
   lm_writes[2] = (VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = lm_set,
      .dstBinding = 2,
      .descriptorCount = 3,
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .pImageInfo = lm_style_infos,
   };
   for (unsigned i = 0; i < 7; ++i) {
      lm_buffer_infos[i] = (VkDescriptorBufferInfo){
         .buffer = lm_buffers[i],
         .range = VK_WHOLE_SIZE,
      };
      lm_writes[3 + i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = lm_set,
         .dstBinding = lm_buffer_bindings[i],
         .descriptorCount = 1,
         .descriptorType = lm_buffer_types[i],
         .pBufferInfo = &lm_buffer_infos[i],
      };
   }
   vkUpdateDescriptorSets(device, 10, lm_writes, 0, NULL);
#endif /* G17P_VKQ_LIGHTMAP */

   /* ---------------------------------------------------------------- */
   const VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool command_pool;
   check(vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_buffer_allocate = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 2,
   };
   VkCommandBuffer command_buffers[2];
   check(vkAllocateCommandBuffers(device, &command_buffer_allocate,
                                  command_buffers),
         "vkAllocateCommandBuffers");
   /* vkQuake uploads its static textures on a staging command buffer and
    * records the per-frame dispatches on PCBX_UPDATE_LIGHTMAPS.  Keeping
    * them apart here means a refusal in one is attributed to that one
    * instead of poisoning the whole measurement: a command buffer records
    * no further error once it is INVALID. */
   VkCommandBuffer upload_cmd = command_buffers[0];
   VkCommandBuffer cmd = command_buffers[1];

   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(upload_cmd, &begin),
         "vkBeginCommandBuffer(upload)");
   bool upload_failed = false;

#if G17P_VKQ_LIGHTMAP
   /* Upload the static lightmap inputs and open the storage image. */
   VkImageMemoryBarrier to_transfer[4];
   for (unsigned i = 0; i < 4; ++i) {
      to_transfer[i] = (VkImageMemoryBarrier){
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = 0,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = sources[i].image,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
   }
   const VkImageMemoryBarrier storage_to_general = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = lm_image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   vkCmdPipelineBarrier(upload_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        4, to_transfer);
   vkCmdPipelineBarrier(upload_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0,
                        NULL, 1, &storage_to_general);

   for (unsigned i = 0; i < 4; ++i) {
      const VkBufferImageCopy copy = {
         .bufferOffset = i * LIGHTMAP_BYTES,
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {LIGHTMAP_DIM, LIGHTMAP_DIM, 1},
      };
      vkCmdCopyBufferToImage(upload_cmd, upload.buffer, sources[i].image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
   }

   VkImageMemoryBarrier to_read[4];
   for (unsigned i = 0; i < 4; ++i) {
      to_read[i] = to_transfer[i];
      to_read[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      to_read[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      to_read[i].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      to_read[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   }
   vkCmdPipelineBarrier(upload_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0,
                        NULL, 4, to_read);
#endif

   const VkResult upload_record = vkEndCommandBuffer(upload_cmd);
   if (upload_record != VK_SUCCESS) {
      fprintf(stderr,
              PROBE_NAME
              ": FAIL(upload) — vkEndCommandBuffer returned %d.  A texture "
              "upload was refused at record time, so the lightmap inputs "
              "never reach the GPU.\n",
              (int)upload_record);
      upload_failed = true;
   }

   check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer(work)");

#if G17P_VKQ_GEN
   const struct gen_push_constants gen_push = {
      .num_draws = GEN_SURFACE_COUNT,
      .first_draw = 0,
      .visibility_offset = 0,
      .vieworg_x = 0.0f,
      .vieworg_y = 0.0f,
      .vieworg_z = 0.0f,
      .instance_base = 0,
   };
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, gen_pipeline);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           gen_pipeline_layout, 0, 1, &gen_set, 0, NULL);
   vkCmdPushConstants(cmd, gen_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(gen_push), &gen_push);
   vkCmdDispatch(cmd, 1, 1, 1);
#endif

#if G17P_VKQ_LIGHTMAP
   const struct lightmap_push_constants lm_push = {
      .num_dlights = 0,
      .lightmap_width = LIGHTMAP_DIM,
      .x_workgroup_offset = 0,
      .y_workgroup_offset = 0,
      .dlights_only = 0,
      .num_cached_dlights = 0,
      .submodel_transforms_offset = 0,
      .prev_submodel_transforms_offset = 0,
      .view_pos_x = 0.0f,
      .view_pos_y = 0.0f,
      .view_pos_z = 0.0f,
   };
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lm_pipeline);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           lm_pipeline_layout, 0, 1, &lm_set, 0, NULL);
   vkCmdPushConstants(cmd, lm_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                      sizeof(lm_push), &lm_push);
   vkCmdDispatch(cmd, 1, 1, 1);

   const VkImageMemoryBarrier storage_to_src = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = lm_image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &storage_to_src);
   const VkBufferImageCopy readback_copy = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {LIGHTMAP_DIM, LIGHTMAP_DIM, 1},
   };
   vkCmdCopyImageToBuffer(cmd, lm_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback.buffer, 1, &readback_copy);
#endif

   const VkResult record = vkEndCommandBuffer(cmd);
   if (record != VK_SUCCESS) {
      fprintf(stderr,
              PROBE_NAME
              ": FAIL — vkEndCommandBuffer returned %d.  A recording call was "
              "refused, so this command buffer is INVALID and the work it "
              "carries never reaches the GPU.\n",
              (int)record);
      return EXIT_FAILURE;
   }

   if (record_only) {
      if (upload_failed) {
         fprintf(stderr, PROBE_NAME
                 ": FAIL — the dispatches record, but the texture upload "
                 "does not\n");
         return EXIT_FAILURE;
      }
      printf(PROBE_NAME
             ": PASS — pipelines compiled and the dispatches recorded; "
             "vkEndCommandBuffer succeeded\n");
      return EXIT_SUCCESS;
   }

   if (upload_failed)
      return EXIT_FAILURE;

   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence fence;
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 2,
      .pCommandBuffers = command_buffers,
   };
   check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
   check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
         "vkWaitForFences");

   bool failed = false;

#if G17P_VKQ_GEN
   uint32_t fan0[16], fan1[16];
   const unsigned fan0_words = gen_expected_fan(100, 4, fan0);
   const unsigned fan1_words = gen_expected_fan(200, 5, fan1);
   const uint32_t expected_index_count = fan0_words + fan1_words;

   if (commands[0].indexCount != expected_index_count) {
      fprintf(stderr,
              PROBE_NAME ": FAIL — indirect command 0 indexCount %u, "
                         "expected %u.  A zero here is a world that is never "
                         "drawn.\n",
              commands[0].indexCount, expected_index_count);
      failed = true;
   }
   if (commands[1].indexCount != 0) {
      fprintf(stderr,
              PROBE_NAME ": FAIL — indirect command 1 indexCount %u, "
                         "expected 0 (both its surfaces are culled)\n",
              commands[1].indexCount);
      failed = true;
   }

   /* The two survivors race on one atomicAdd, so either fan may land first,
    * but each fan is contiguous and starts at the command's firstIndex. */
   const uint32_t *written = (const uint32_t *)indices.map;
   const uint32_t *window = written + GEN_FIRST_INDEX_0;
   const bool fan0_first =
      memcmp(window, fan0, fan0_words * sizeof(uint32_t)) == 0 &&
      memcmp(window + fan0_words, fan1, fan1_words * sizeof(uint32_t)) == 0;
   const bool fan1_first =
      memcmp(window, fan1, fan1_words * sizeof(uint32_t)) == 0 &&
      memcmp(window + fan1_words, fan0, fan0_words * sizeof(uint32_t)) == 0;
   if (!fan0_first && !fan1_first) {
      fprintf(stderr, PROBE_NAME
              ": FAIL — generated indices at %u are wrong; got",
              GEN_FIRST_INDEX_0);
      for (unsigned i = 0; i < expected_index_count; ++i)
         fprintf(stderr, " %u", window[i]);
      fputc('\n', stderr);
      failed = true;
   }
   for (unsigned i = 0; i < GEN_INDEX_WORDS; ++i) {
      const bool inside = i >= GEN_FIRST_INDEX_0 &&
                          i < GEN_FIRST_INDEX_0 + expected_index_count;
      if (!inside && written[i] != SENTINEL) {
         fprintf(stderr,
                 PROBE_NAME ": FAIL — index word %u outside the command's "
                            "range was overwritten with 0x%08" PRIx32 "\n",
                 i, written[i]);
         failed = true;
         break;
      }
   }
#endif

#if G17P_VKQ_LIGHTMAP
   const uint8_t *texels = readback.map;
   for (unsigned i = 0; i < LIGHTMAP_DIM * LIGHTMAP_DIM; ++i) {
      const uint8_t r = texels[4 * i + 0], g = texels[4 * i + 1];
      const uint8_t b = texels[4 * i + 2], a = texels[4 * i + 3];
      if (r != LM_EXPECTED_R || g != LM_EXPECTED_G || b != LM_EXPECTED_B ||
          a != 0) {
         fprintf(stderr,
                 PROBE_NAME ": FAIL — lightmap texel %u is %02x%02x%02x%02x, "
                            "expected %02x%02x%02x00.  An all-zero lightmap "
                            "renders every world surface black.\n",
                 i, r, g, b, a, LM_EXPECTED_R, LM_EXPECTED_G, LM_EXPECTED_B);
         failed = true;
         break;
      }
   }
#endif

   if (failed)
      return EXIT_FAILURE;

   printf(PROBE_NAME ": PASS — vkQuake's world compute produced exactly the "
                     "expected output\n");
   return EXIT_SUCCESS;
}
