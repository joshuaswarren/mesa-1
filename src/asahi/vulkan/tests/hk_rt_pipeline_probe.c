/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Host-side probe for VK_KHR_ray_tracing_pipeline on G17P.
 *
 * It runs under the no-op drm-shim, which executes nothing, so it can only say
 * that the driver ACCEPTS and RECORDS the work -- that every stage compiles
 * into the megakernel, that the shader groups get distinct handles, that a
 * pipeline library imports, and that vkCmdTraceRaysKHR and both indirect forms
 * record a command stream the shim's validator accepts.  It cannot say that a
 * ray hits anything.  Everything it prints is labelled accordingly, and the
 * decision rule for the hardware run it precedes is in
 * docs/G17P_RAYTRACING_PIPELINE_ABI.md.
 *
 * The configuration is printed BEFORE any submit, so a run that takes a device
 * down still records what it was asked to do.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hk_rt_pipeline_probe_spv.h"
#include "hk_rt_pipeline_probe_features_spv.h"

#define T8140_DEVICE_ID 0x8140

#define LAUNCH_WIDTH  8
#define LAUNCH_HEIGHT 4
#define LAUNCH_DEPTH  1

#define HANDLE_SIZE 32

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
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
VK_FN(vkGetBufferDeviceAddress);
VK_FN(vkCreateDescriptorSetLayout);
VK_FN(vkDestroyDescriptorSetLayout);
VK_FN(vkCreateShaderModule);
VK_FN(vkDestroyShaderModule);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkDestroyPipelineLayout);
VK_FN(vkCreateComputePipelines);
VK_FN(vkCreateRayTracingPipelinesKHR);
VK_FN(vkGetRayTracingShaderGroupHandlesKHR);
VK_FN(vkGetRayTracingShaderGroupStackSizeKHR);
VK_FN(vkDestroyPipeline);
VK_FN(vkCreateCommandPool);
VK_FN(vkDestroyCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdTraceRaysKHR);
VK_FN(vkCmdTraceRaysIndirectKHR);
VK_FN(vkCmdTraceRaysIndirect2KHR);
VK_FN(vkEndCommandBuffer);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);

static const char *prog = "hk-rt-pipeline-probe";
static unsigned failures;

static void
die(const char *what)
{
   fprintf(stderr, "%s: %s\n", prog, what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "%s: %s returned %d\n", prog, what, result);
      exit(EXIT_FAILURE);
   }
}

static void
expect(bool condition, const char *what)
{
   printf("%s: check %-48s %s\n", prog, what, condition ? "PASS" : "FAIL");
   if (!condition)
      failures++;
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

static uint32_t
find_memory_type(const VkPhysicalDeviceMemoryProperties *properties,
                 uint32_t allowed, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((allowed & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }

   die("host-visible coherent memory is unavailable");
   return UINT32_MAX;
}

struct scratch_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   VkDeviceAddress address;
   void *map;
};

static struct scratch_buffer
make_buffer(VkDevice device, const VkPhysicalDeviceMemoryProperties *memprops,
            VkDeviceSize size, VkBufferUsageFlags usage)
{
   struct scratch_buffer out = {0};

   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   check(vkCreateBuffer(device, &buffer_info, NULL, &out.buffer),
         "vkCreateBuffer");

   VkMemoryRequirements reqs;
   vkGetBufferMemoryRequirements(device, out.buffer, &reqs);

   const VkMemoryAllocateFlagsInfo flags_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
   };
   const VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &flags_info,
      .allocationSize = reqs.size,
      .memoryTypeIndex = find_memory_type(
         memprops, reqs.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   check(vkAllocateMemory(device, &alloc_info, NULL, &out.memory),
         "vkAllocateMemory");
   check(vkBindBufferMemory(device, out.buffer, out.memory, 0),
         "vkBindBufferMemory");
   check(vkMapMemory(device, out.memory, 0, VK_WHOLE_SIZE, 0, &out.map),
         "vkMapMemory");
   memset(out.map, 0, size);

   const VkBufferDeviceAddressInfo address_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = out.buffer,
   };
   out.address = vkGetBufferDeviceAddress(device, &address_info);

   return out;
}

static VkShaderModule
make_module(VkDevice device, const uint32_t *spv, size_t bytes)
{
   const VkShaderModuleCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = bytes,
      .pCode = spv,
   };
   VkShaderModule module;
   check(vkCreateShaderModule(device, &info, NULL, &module),
         "vkCreateShaderModule");
   return module;
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
      .pApplicationName = prog,
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
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures2);
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

   /* Prefer G17P, but take whatever Asahi device the shim exposes: the AGX2
    * control arm (ASAHI_SHIM_CHIP unset or g13g) is not optional -- a refusal
    * that reproduces on both arms is not an AGX3 bug.  The device is printed
    * with the configuration below, so a log always says which arm it is. */
   VkPhysicalDevice physical = VK_NULL_HANDLE;
   VkPhysicalDeviceProperties properties;
   for (uint32_t i = 0; i < physical_count; ++i) {
      vkGetPhysicalDeviceProperties(physical_devices[i], &properties);
      if (properties.deviceID == T8140_DEVICE_ID) {
         physical = physical_devices[i];
         break;
      }
   }
   if (physical == VK_NULL_HANDLE && physical_count > 0) {
      physical = physical_devices[0];
      vkGetPhysicalDeviceProperties(physical, &properties);
   }
   free(physical_devices);
   if (physical == VK_NULL_HANDLE)
      die("no Vulkan device was found");

   VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_features = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
   };
   VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
      .pNext = &rt_features,
   };
   VkPhysicalDeviceBufferDeviceAddressFeatures bda_features = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
      .pNext = &as_features,
   };
   VkPhysicalDeviceFeatures2 features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &bda_features,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features2);

   VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &rt_props,
   };
   vkGetPhysicalDeviceProperties2(physical, &props2);

   /* Configuration first, before anything can fail or hang. */
   printf("%s: device=%s deviceID=0x%x launch=%ux%ux%u\n", prog,
          properties.deviceName, properties.deviceID, LAUNCH_WIDTH,
          LAUNCH_HEIGHT, LAUNCH_DEPTH);
   printf("%s: rayTracingPipeline=%u traceRaysIndirect=%u "
          "handleSize=%u baseAlignment=%u handleAlignment=%u "
          "maxRecursion=%u maxHitAttribute=%u\n",
          prog, rt_features.rayTracingPipeline,
          rt_features.rayTracingPipelineTraceRaysIndirect,
          rt_props.shaderGroupHandleSize, rt_props.shaderGroupBaseAlignment,
          rt_props.shaderGroupHandleAlignment, rt_props.maxRayRecursionDepth,
          rt_props.maxRayHitAttributeSize);
   fflush(stdout);

   if (!rt_features.rayTracingPipeline) {
      printf("%s: rayTracingPipeline is not advertised -- set "
             "HK_G17P_UNVERIFIED_FEATURES=ray_query,ray_tracing_pipeline. "
             "NOT MEASURED.\n",
             prog);
      vkDestroyInstance(instance, NULL);
      return 77;
   }

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

   VkPhysicalDeviceMemoryProperties memprops;
   vkGetPhysicalDeviceMemoryProperties(physical, &memprops);

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const char *device_extensions[] = {
      VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
      VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
      VK_KHR_RAY_TRACING_MAINTENANCE_1_EXTENSION_NAME,
      VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
      VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features2,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount =
         (uint32_t)(sizeof(device_extensions) / sizeof(device_extensions[0])),
      .ppEnabledExtensionNames = device_extensions,
   };
   VkDevice device;
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
   LOAD_DEVICE(device, vkUnmapMemory);
   LOAD_DEVICE(device, vkGetBufferDeviceAddress);
   LOAD_DEVICE(device, vkCreateDescriptorSetLayout);
   LOAD_DEVICE(device, vkDestroyDescriptorSetLayout);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkDestroyShaderModule);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkDestroyPipelineLayout);
   LOAD_DEVICE(device, vkCreateComputePipelines);
   LOAD_DEVICE(device, vkCreateRayTracingPipelinesKHR);
   LOAD_DEVICE(device, vkGetRayTracingShaderGroupHandlesKHR);
   LOAD_DEVICE(device, vkGetRayTracingShaderGroupStackSizeKHR);
   LOAD_DEVICE(device, vkDestroyPipeline);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkDestroyCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdTraceRaysKHR);
   LOAD_DEVICE(device, vkCmdTraceRaysIndirectKHR);
   LOAD_DEVICE(device, vkCmdTraceRaysIndirect2KHR);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkDestroyFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);

   const VkDescriptorSetLayoutBinding bindings[] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_ALL,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_ALL,
      },
   };
   const VkDescriptorSetLayoutCreateInfo set_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout set_layout;
   check(vkCreateDescriptorSetLayout(device, &set_layout_info, NULL,
                                     &set_layout),
         "vkCreateDescriptorSetLayout");

   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkPipelineLayout layout;
   check(vkCreatePipelineLayout(device, &layout_info, NULL, &layout),
         "vkCreatePipelineLayout");

   VkShaderModule rgen = make_module(device, hk_rt_pipeline_probe_rgen_spv,
                                     sizeof(hk_rt_pipeline_probe_rgen_spv));
   VkShaderModule rmiss = make_module(device, hk_rt_pipeline_probe_rmiss_spv,
                                      sizeof(hk_rt_pipeline_probe_rmiss_spv));
   VkShaderModule rchit = make_module(device, hk_rt_pipeline_probe_rchit_spv,
                                      sizeof(hk_rt_pipeline_probe_rchit_spv));
   VkShaderModule rahit = make_module(device, hk_rt_pipeline_probe_rahit_spv,
                                      sizeof(hk_rt_pipeline_probe_rahit_spv));
   VkShaderModule rint = make_module(device, hk_rt_pipeline_probe_rint_spv,
                                     sizeof(hk_rt_pipeline_probe_rint_spv));
   VkShaderModule rcall = make_module(device, hk_rt_pipeline_probe_rcall_spv,
                                      sizeof(hk_rt_pipeline_probe_rcall_spv));

   /* Six stages, five groups: one of every kind the extension defines, so a
    * stage that fails to inline fails here rather than on hardware. */
   const VkPipelineShaderStageCreateInfo stages[] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
       .module = rgen,
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
       .module = rmiss,
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
       .module = rchit,
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
       .module = rahit,
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
       .module = rint,
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_CALLABLE_BIT_KHR,
       .module = rcall,
       .pName = "main"},
   };

#define UNUSED_SHADER VK_SHADER_UNUSED_KHR
   const VkRayTracingShaderGroupCreateInfoKHR groups[] = {
      /* 0: raygen */
      {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
       .generalShader = 0,
       .closestHitShader = UNUSED_SHADER,
       .anyHitShader = UNUSED_SHADER,
       .intersectionShader = UNUSED_SHADER},
      /* 1: miss */
      {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
       .generalShader = 1,
       .closestHitShader = UNUSED_SHADER,
       .anyHitShader = UNUSED_SHADER,
       .intersectionShader = UNUSED_SHADER},
      /* 2: triangle hit group with an any-hit shader */
      {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
       .generalShader = UNUSED_SHADER,
       .closestHitShader = 2,
       .anyHitShader = 3,
       .intersectionShader = UNUSED_SHADER},
      /* 3: procedural hit group */
      {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
       .generalShader = UNUSED_SHADER,
       .closestHitShader = 2,
       .anyHitShader = 3,
       .intersectionShader = 4},
      /* 4: callable */
      {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
       .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
       .generalShader = 5,
       .closestHitShader = UNUSED_SHADER,
       .anyHitShader = UNUSED_SHADER,
       .intersectionShader = UNUSED_SHADER},
   };
   const uint32_t stage_count = 6;
   const uint32_t group_count = 5;

   /* Bisect first, whole pipeline second.  The megakernel inlines every stage
    * into one binary, so a compiler gap in any one stage fails the whole
    * create with no indication of which stage it was.  Build up one stage set
    * at a time and print each, so a failure names the smallest set that
    * reproduces it before the full pipeline is ever attempted. */
   static const struct {
      const char *name;
      uint32_t stages;
      uint32_t groups;
   } ladder[] = {
      {"raygen only", 1, 1},
      {"raygen+miss", 2, 2},
      {"raygen+miss+chit+ahit", 4, 3},
      {"all six stages", 6, 5},
   };

   for (unsigned i = 0; i < sizeof(ladder) / sizeof(ladder[0]); ++i) {
      const VkRayTracingPipelineCreateInfoKHR rung_info = {
         .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
         .stageCount = ladder[i].stages,
         .pStages = stages,
         .groupCount = ladder[i].groups,
         .pGroups = groups,
         .maxPipelineRayRecursionDepth = 1,
         .layout = layout,
      };
      VkPipeline rung = VK_NULL_HANDLE;
      printf("%s: rung %-24s creating...\n", prog, ladder[i].name);
      fflush(stdout);
      const VkResult rung_result = vkCreateRayTracingPipelinesKHR(
         device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rung_info, NULL, &rung);
      printf("%s: rung %-24s %s (%d)\n", prog, ladder[i].name,
             rung_result == VK_SUCCESS ? "PASS" : "FAIL", rung_result);
      fflush(stdout);
      if (rung_result != VK_SUCCESS)
         failures++;
      if (rung != VK_NULL_HANDLE)
         vkDestroyPipeline(device, rung, NULL);
   }

   /* Second ladder: the constructs dEQP-VK.ray_tracing_pipeline exercises that
    * the six-stage ladder above does not reach.  The shaders above declare a
    * callable stage but never call it, never read a hit system value, never
    * terminate a ray from an any-hit shader, never declare a shaderRecordEXT
    * block and never read gl_CullMaskEXT.  Each rung is its own pipeline and
    * is printed BEFORE it is created, so a compiler gap names the construct. */
   VkShaderModule call_rgen =
      make_module(device, hk_rt_probe_call_rgen_spv,
                  sizeof(hk_rt_probe_call_rgen_spv));
   VkShaderModule sysval_rchit =
      make_module(device, hk_rt_probe_sysval_rchit_spv,
                  sizeof(hk_rt_probe_sysval_rchit_spv));
   VkShaderModule terminate_rahit =
      make_module(device, hk_rt_probe_terminate_rahit_spv,
                  sizeof(hk_rt_probe_terminate_rahit_spv));
   VkShaderModule record_rgen =
      make_module(device, hk_rt_probe_record_rgen_spv,
                  sizeof(hk_rt_probe_record_rgen_spv));
   VkShaderModule record_rmiss =
      make_module(device, hk_rt_probe_record_rmiss_spv,
                  sizeof(hk_rt_probe_record_rmiss_spv));
   VkShaderModule cullmask_rgen =
      make_module(device, hk_rt_probe_cullmask_rgen_spv,
                  sizeof(hk_rt_probe_cullmask_rgen_spv));
   VkShaderModule cullmask_rmiss =
      make_module(device, hk_rt_probe_cullmask_rmiss_spv,
                  sizeof(hk_rt_probe_cullmask_rmiss_spv));
   VkShaderModule rq_rchit =
      make_module(device, hk_rt_probe_rq_rchit_spv,
                  sizeof(hk_rt_probe_rq_rchit_spv));
   VkShaderModule rqarray_rchit =
      make_module(device, hk_rt_probe_rqarray_rchit_spv,
                  sizeof(hk_rt_probe_rqarray_rchit_spv));

#define RT_STAGE(bit, mod)                                                     \
   {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,              \
    .stage = (bit),                                                            \
    .module = (mod),                                                           \
    .pName = "main"}
#define RT_GENERAL(idx)                                                        \
   {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,       \
    .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,                      \
    .generalShader = (idx),                                                    \
    .closestHitShader = UNUSED_SHADER,                                         \
    .anyHitShader = UNUSED_SHADER,                                             \
    .intersectionShader = UNUSED_SHADER}
#define RT_TRIANGLES(chit, ahit)                                               \
   {.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,       \
    .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,          \
    .generalShader = UNUSED_SHADER,                                            \
    .closestHitShader = (chit),                                                \
    .anyHitShader = (ahit),                                                    \
    .intersectionShader = UNUSED_SHADER}

   const VkPipelineShaderStageCreateInfo call_stages[] = {
      RT_STAGE(VK_SHADER_STAGE_RAYGEN_BIT_KHR, call_rgen),
      RT_STAGE(VK_SHADER_STAGE_CALLABLE_BIT_KHR, rcall),
   };
   const VkRayTracingShaderGroupCreateInfoKHR call_groups[] = {
      RT_GENERAL(0),
      RT_GENERAL(1),
   };

   const VkPipelineShaderStageCreateInfo sysval_stages[] = {
      RT_STAGE(VK_SHADER_STAGE_RAYGEN_BIT_KHR, rgen),
      RT_STAGE(VK_SHADER_STAGE_MISS_BIT_KHR, rmiss),
      RT_STAGE(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, sysval_rchit),
   };
   const VkRayTracingShaderGroupCreateInfoKHR sysval_groups[] = {
      RT_GENERAL(0),
      RT_GENERAL(1),
      RT_TRIANGLES(2, UNUSED_SHADER),
   };

   const VkPipelineShaderStageCreateInfo terminate_stages[] = {
      RT_STAGE(VK_SHADER_STAGE_RAYGEN_BIT_KHR, rgen),
      RT_STAGE(VK_SHADER_STAGE_MISS_BIT_KHR, rmiss),
      RT_STAGE(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, rchit),
      RT_STAGE(VK_SHADER_STAGE_ANY_HIT_BIT_KHR, terminate_rahit),
   };
   const VkRayTracingShaderGroupCreateInfoKHR terminate_groups[] = {
      RT_GENERAL(0),
      RT_GENERAL(1),
      RT_TRIANGLES(2, 3),
   };

   const VkPipelineShaderStageCreateInfo record_stages[] = {
      RT_STAGE(VK_SHADER_STAGE_RAYGEN_BIT_KHR, record_rgen),
      RT_STAGE(VK_SHADER_STAGE_MISS_BIT_KHR, record_rmiss),
   };
   const VkRayTracingShaderGroupCreateInfoKHR record_groups[] = {
      RT_GENERAL(0),
      RT_GENERAL(1),
   };

   /* A ray query INSIDE a ray-tracing-pipeline stage.  dEQP-VK.ray_query has a
    * variant of nearly every case for each ray-tracing stage, so this is the
    * larger half of what enabling this extension turns on. */
   const VkPipelineShaderStageCreateInfo rq_stages[] = {
      RT_STAGE(VK_SHADER_STAGE_RAYGEN_BIT_KHR, rgen),
      RT_STAGE(VK_SHADER_STAGE_MISS_BIT_KHR, rmiss),
      RT_STAGE(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, rq_rchit),
   };
   const VkRayTracingShaderGroupCreateInfoKHR rq_groups[] = {
      RT_GENERAL(0),
      RT_GENERAL(1),
      RT_TRIANGLES(2, UNUSED_SHADER),
   };

   /* The shape the rayqueryterminate CTS cases compile: an ARRAY of ray
    * queries indexed in a loop, plus rayQueryTerminateEXT. */
   const VkPipelineShaderStageCreateInfo rqarray_stages[] = {
      RT_STAGE(VK_SHADER_STAGE_RAYGEN_BIT_KHR, rgen),
      RT_STAGE(VK_SHADER_STAGE_MISS_BIT_KHR, rmiss),
      RT_STAGE(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, rqarray_rchit),
   };
   const VkRayTracingShaderGroupCreateInfoKHR rqarray_groups[] = {
      RT_GENERAL(0),
      RT_GENERAL(1),
      RT_TRIANGLES(2, UNUSED_SHADER),
   };

   const VkPipelineShaderStageCreateInfo cullmask_stages[] = {
      RT_STAGE(VK_SHADER_STAGE_RAYGEN_BIT_KHR, cullmask_rgen),
      RT_STAGE(VK_SHADER_STAGE_MISS_BIT_KHR, cullmask_rmiss),
   };
   const VkRayTracingShaderGroupCreateInfoKHR cullmask_groups[] = {
      RT_GENERAL(0),
      RT_GENERAL(1),
   };

   const struct {
      const char *name;
      const VkPipelineShaderStageCreateInfo *stages;
      uint32_t stage_count;
      const VkRayTracingShaderGroupCreateInfoKHR *groups;
      uint32_t group_count;
   } features[] = {
      {"executeCallableEXT", call_stages, 2, call_groups, 2},
      {"hit system values", sysval_stages, 3, sysval_groups, 3},
      {"terminateRayEXT in any-hit", terminate_stages, 4, terminate_groups, 3},
      {"shaderRecordEXT block", record_stages, 2, record_groups, 2},
      {"gl_CullMaskEXT", cullmask_stages, 2, cullmask_groups, 2},
      {"ray query in closest-hit", rq_stages, 3, rq_groups, 3},
      {"ray query array + terminate", rqarray_stages, 3, rqarray_groups, 3},
   };

   /* HK_RT_PROBE_ONLY=<substring> runs one feature rung.  A rung that takes
    * the process down takes the rest of the ladder with it, so a bisected rerun
    * has to be able to ask for exactly one. */
   const char *only = getenv("HK_RT_PROBE_ONLY");

   for (unsigned i = 0; i < sizeof(features) / sizeof(features[0]); ++i) {
      if (only && !strstr(features[i].name, only)) {
         printf("%s: feature %-28s SKIPPED (HK_RT_PROBE_ONLY)\n", prog,
                features[i].name);
         continue;
      }

      const VkRayTracingPipelineCreateInfoKHR feature_info = {
         .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
         .stageCount = features[i].stage_count,
         .pStages = features[i].stages,
         .groupCount = features[i].group_count,
         .pGroups = features[i].groups,
         .maxPipelineRayRecursionDepth = 1,
         .layout = layout,
      };
      VkPipeline feature_pipeline = VK_NULL_HANDLE;
      printf("%s: feature %-28s creating...\n", prog, features[i].name);
      fflush(stdout);
      const VkResult feature_result = vkCreateRayTracingPipelinesKHR(
         device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &feature_info, NULL,
         &feature_pipeline);
      printf("%s: feature %-28s %s (%d)\n", prog, features[i].name,
             feature_result == VK_SUCCESS ? "PASS" : "FAIL", feature_result);
      fflush(stdout);
      if (feature_result != VK_SUCCESS)
         failures++;
      if (feature_pipeline != VK_NULL_HANDLE)
         vkDestroyPipeline(device, feature_pipeline, NULL);
   }

   /* Control, and it is the load-bearing one for the rung above: the SAME ray
    * query in an ORDINARY compute pipeline, which is the path
    * dEQP-VK.ray_query.*.comp.* takes and which measured 43/45 on Neo.  If
    * this passes and the closest-hit rung does not, the difference is the
    * megakernel path, not ray queries on G17P. */
   {
      VkShaderModule rq_comp = make_module(device, hk_rt_probe_rq_comp_spv,
                                           sizeof(hk_rt_probe_rq_comp_spv));
      const VkComputePipelineCreateInfo compute_info = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = RT_STAGE(VK_SHADER_STAGE_COMPUTE_BIT, rq_comp),
         .layout = layout,
      };
      VkPipeline compute_pipeline = VK_NULL_HANDLE;
      printf("%s: control %-28s creating...\n", prog, "ray query in compute");
      fflush(stdout);
      const VkResult compute_result = vkCreateComputePipelines(
         device, VK_NULL_HANDLE, 1, &compute_info, NULL, &compute_pipeline);
      printf("%s: control %-28s %s (%d)\n", prog, "ray query in compute",
             compute_result == VK_SUCCESS ? "PASS" : "FAIL", compute_result);
      fflush(stdout);
      if (compute_result != VK_SUCCESS)
         failures++;
      if (compute_pipeline != VK_NULL_HANDLE)
         vkDestroyPipeline(device, compute_pipeline, NULL);
      vkDestroyShaderModule(device, rq_comp, NULL);
   }

   /* rayTracingPositionFetch is compiled for
    * (hk_ray_tracing_pipeline.c:1422) but VK_KHR_ray_tracing_position_fetch is
    * not in hk_physical_device.c's supported set, so the feature can never be
    * enabled and gl_HitTriangleVertexPositionsEXT is unreachable.  NOT
    * MEASURED, deliberately, rather than tested through a disabled feature. */
   printf("%s: feature %-28s NOT MEASURED (extension unadvertised)\n", prog,
          "position fetch");

   const VkRayTracingPipelineCreateInfoKHR pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .stageCount = stage_count,
      .pStages = stages,
      .groupCount = group_count,
      .pGroups = groups,
      .maxPipelineRayRecursionDepth = 1,
      .layout = layout,
   };

   VkPipeline pipeline = VK_NULL_HANDLE;
   VkResult create_result = vkCreateRayTracingPipelinesKHR(
      device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipeline_info, NULL,
      &pipeline);
   expect(create_result == VK_SUCCESS && pipeline != VK_NULL_HANDLE,
          "vkCreateRayTracingPipelinesKHR");
   if (create_result != VK_SUCCESS) {
      fprintf(stderr, "%s: vkCreateRayTracingPipelinesKHR returned %d\n", prog,
              create_result);
      return EXIT_FAILURE;
   }

   /* A pipeline library holds stages and groups but compiles no megakernel,
    * and importing it must renumber the imported groups' stage references
    * without renumbering their handles. */
   /* A library's group indices are relative to the LIBRARY's own stage array,
    * so the miss group has to be restated with generalShader 0 rather than
    * reusing the one that names index 1 of the full pipeline. */
   const VkRayTracingShaderGroupCreateInfoKHR library_group = {
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
      .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
      .generalShader = 0,
      .closestHitShader = UNUSED_SHADER,
      .anyHitShader = UNUSED_SHADER,
      .intersectionShader = UNUSED_SHADER,
   };
   const VkRayTracingPipelineCreateInfoKHR library_info = {
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR,
      .stageCount = 1,
      .pStages = &stages[1], /* the miss shader */
      .groupCount = 1,
      .pGroups = &library_group,
      .maxPipelineRayRecursionDepth = 1,
      .layout = layout,
   };
   VkPipeline library = VK_NULL_HANDLE;
   expect(vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE,
                                         VK_NULL_HANDLE, 1, &library_info,
                                         NULL, &library) == VK_SUCCESS &&
             library != VK_NULL_HANDLE,
          "create a pipeline library");

   uint8_t library_handle[HANDLE_SIZE] = {0};
   if (library != VK_NULL_HANDLE) {
      check(vkGetRayTracingShaderGroupHandlesKHR(device, library, 0, 1,
                                                 sizeof(library_handle),
                                                 library_handle),
            "vkGetRayTracingShaderGroupHandlesKHR(library)");
   }

   VkPipeline imported = VK_NULL_HANDLE;
   if (library != VK_NULL_HANDLE) {
      const VkPipelineLibraryCreateInfoKHR libraries = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
         .libraryCount = 1,
         .pLibraries = &library,
      };
      const VkRayTracingPipelineCreateInfoKHR import_info = {
         .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
         .stageCount = 1,
         .pStages = &stages[0], /* the raygen shader */
         .groupCount = 1,
         .pGroups = &groups[0],
         .pLibraryInfo = &libraries,
         .maxPipelineRayRecursionDepth = 1,
         .layout = layout,
      };
      expect(vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE,
                                            VK_NULL_HANDLE, 1, &import_info,
                                            NULL, &imported) == VK_SUCCESS &&
                imported != VK_NULL_HANDLE,
             "import a pipeline library");
   }

   if (imported != VK_NULL_HANDLE) {
      uint8_t imported_handles[2 * HANDLE_SIZE] = {0};
      check(vkGetRayTracingShaderGroupHandlesKHR(device, imported, 0, 2,
                                                 sizeof(imported_handles),
                                                 imported_handles),
            "vkGetRayTracingShaderGroupHandlesKHR(imported)");
      /* The imported group keeps the handle it had in the library: an
       * application may have captured it before importing. */
      expect(memcmp(imported_handles + HANDLE_SIZE, library_handle,
                    HANDLE_SIZE) == 0,
             "an imported group keeps its library handle");
   }

   uint8_t handles[5 * HANDLE_SIZE];
   memset(handles, 0xCD, sizeof(handles));
   check(vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, group_count,
                                              sizeof(handles), handles),
         "vkGetRayTracingShaderGroupHandlesKHR");

   bool distinct = true, nonzero = true, padded = true;
   for (uint32_t i = 0; i < group_count; ++i) {
      const uint8_t *h = handles + i * HANDLE_SIZE;
      uint32_t index;
      memcpy(&index, h, 4);
      if (index == 0)
         nonzero = false;
      for (unsigned b = 4; b < HANDLE_SIZE; ++b)
         if (h[b] != 0)
            padded = false;
      for (uint32_t j = 0; j < i; ++j)
         if (memcmp(h, handles + j * HANDLE_SIZE, HANDLE_SIZE) == 0)
            distinct = false;
   }
   expect(distinct, "group handles are distinct");
   expect(nonzero, "group handles are non-zero");
   expect(padded, "group handles are zero-padded to 32 bytes");

   /* A group reports zero for a shader it does not have, and non-zero for one
    * it does whenever that shader has private storage. */
   expect(vkGetRayTracingShaderGroupStackSizeKHR(
             device, pipeline, 0, VK_SHADER_GROUP_SHADER_INTERSECTION_KHR) == 0,
          "a raygen group has no intersection stack");
   expect(vkGetRayTracingShaderGroupStackSizeKHR(
             device, pipeline, 3, VK_SHADER_GROUP_SHADER_INTERSECTION_KHR) != 0,
          "a procedural group has an intersection stack");
   expect(vkGetRayTracingShaderGroupStackSizeKHR(
             device, pipeline, 2, VK_SHADER_GROUP_SHADER_INTERSECTION_KHR) == 0,
          "a triangle group has no intersection stack");

   /* Build a shader binding table: one raygen record, one miss record, two hit
    * records, one callable record, each shaderGroupHandleSize aligned up to
    * shaderGroupBaseAlignment for the region base. */
   const uint32_t base_align = rt_props.shaderGroupBaseAlignment;
   const uint32_t stride = rt_props.shaderGroupHandleSize;
   const VkDeviceSize sbt_size = 4 * base_align;

   struct scratch_buffer sbt = make_buffer(
      device, &memprops, sbt_size,
      VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
         VK_BUFFER_USAGE_TRANSFER_DST_BIT);

   uint8_t *sbt_map = sbt.map;
   memcpy(sbt_map + 0 * base_align, handles + 0 * HANDLE_SIZE, HANDLE_SIZE);
   memcpy(sbt_map + 1 * base_align, handles + 1 * HANDLE_SIZE, HANDLE_SIZE);
   memcpy(sbt_map + 2 * base_align, handles + 2 * HANDLE_SIZE, HANDLE_SIZE);
   memcpy(sbt_map + 2 * base_align + stride, handles + 3 * HANDLE_SIZE,
          HANDLE_SIZE);
   memcpy(sbt_map + 3 * base_align, handles + 4 * HANDLE_SIZE, HANDLE_SIZE);

   const VkStridedDeviceAddressRegionKHR raygen_region = {
      .deviceAddress = sbt.address + 0 * base_align,
      .stride = base_align,
      .size = base_align,
   };
   const VkStridedDeviceAddressRegionKHR miss_region = {
      .deviceAddress = sbt.address + 1 * base_align,
      .stride = stride,
      .size = stride,
   };
   const VkStridedDeviceAddressRegionKHR hit_region = {
      .deviceAddress = sbt.address + 2 * base_align,
      .stride = stride,
      .size = 2 * stride,
   };
   const VkStridedDeviceAddressRegionKHR callable_region = {
      .deviceAddress = sbt.address + 3 * base_align,
      .stride = stride,
      .size = stride,
   };

   struct scratch_buffer indirect =
      make_buffer(device, &memprops, sizeof(VkTraceRaysIndirectCommand2KHR),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
   {
      VkTraceRaysIndirectCommand2KHR *cmd2 = indirect.map;
      *cmd2 = (VkTraceRaysIndirectCommand2KHR){
         .raygenShaderRecordAddress = raygen_region.deviceAddress,
         .raygenShaderRecordSize = raygen_region.size,
         .missShaderBindingTableAddress = miss_region.deviceAddress,
         .missShaderBindingTableSize = miss_region.size,
         .missShaderBindingTableStride = miss_region.stride,
         .hitShaderBindingTableAddress = hit_region.deviceAddress,
         .hitShaderBindingTableSize = hit_region.size,
         .hitShaderBindingTableStride = hit_region.stride,
         .callableShaderBindingTableAddress = callable_region.deviceAddress,
         .callableShaderBindingTableSize = callable_region.size,
         .callableShaderBindingTableStride = callable_region.stride,
         .width = LAUNCH_WIDTH,
         .height = LAUNCH_HEIGHT,
         .depth = LAUNCH_DEPTH,
      };
   }

   struct scratch_buffer dims =
      make_buffer(device, &memprops, sizeof(VkTraceRaysIndirectCommandKHR),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
   {
      VkTraceRaysIndirectCommandKHR *cmd1 = dims.map;
      *cmd1 = (VkTraceRaysIndirectCommandKHR){
         .width = LAUNCH_WIDTH,
         .height = LAUNCH_HEIGHT,
         .depth = LAUNCH_DEPTH,
      };
   }

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool pool;
   check(vkCreateCommandPool(device, &pool_info, NULL, &pool),
         "vkCreateCommandPool");

   const VkCommandBufferAllocateInfo cmd_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   check(vkAllocateCommandBuffers(device, &cmd_info, &cmd),
         "vkAllocateCommandBuffers");

   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(cmd, &begin_info), "vkBeginCommandBuffer");

   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
   vkCmdTraceRaysKHR(cmd, &raygen_region, &miss_region, &hit_region,
                     &callable_region, LAUNCH_WIDTH, LAUNCH_HEIGHT,
                     LAUNCH_DEPTH);
   vkCmdTraceRaysIndirectKHR(cmd, &raygen_region, &miss_region, &hit_region,
                             &callable_region, dims.address);
   vkCmdTraceRaysIndirect2KHR(cmd, indirect.address);

   check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
   expect(true, "record three trace-rays forms");

   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);

   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence fence;
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");

   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
   };
   const VkResult submit_result = vkQueueSubmit(queue, 1, &submit, fence);
   expect(submit_result == VK_SUCCESS, "vkQueueSubmit accepts the stream");
   if (submit_result == VK_SUCCESS)
      vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

   printf("%s: NOTE the drm-shim executes nothing. This says the driver "
          "records ray tracing pipelines and trace-rays commands, and says "
          "NOTHING about what a ray hits.\n",
          prog);

   vkDestroyFence(device, fence, NULL);
   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyBuffer(device, dims.buffer, NULL);
   vkFreeMemory(device, dims.memory, NULL);
   vkDestroyBuffer(device, indirect.buffer, NULL);
   vkFreeMemory(device, indirect.memory, NULL);
   vkDestroyBuffer(device, sbt.buffer, NULL);
   vkFreeMemory(device, sbt.memory, NULL);
   if (imported != VK_NULL_HANDLE)
      vkDestroyPipeline(device, imported, NULL);
   if (library != VK_NULL_HANDLE)
      vkDestroyPipeline(device, library, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyShaderModule(device, rcall, NULL);
   vkDestroyShaderModule(device, rint, NULL);
   vkDestroyShaderModule(device, rahit, NULL);
   vkDestroyShaderModule(device, rchit, NULL);
   vkDestroyShaderModule(device, rmiss, NULL);
   vkDestroyShaderModule(device, rgen, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyDescriptorSetLayout(device, set_layout, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);

   if (failures) {
      printf("%s: %u checks FAILED\n", prog, failures);
      return EXIT_FAILURE;
   }

   printf("%s: all checks PASS\n", prog);
   return EXIT_SUCCESS;
}
