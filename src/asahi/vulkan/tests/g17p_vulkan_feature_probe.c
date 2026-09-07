/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G17P game-feature probe: does an ADVERTISED feature survive both halves of
 * the driver?
 *
 * `g17p-vulkan-pipeline-probe` answers "does the AGX3 packer encode this
 * shader", which is the compile half.  It creates its device with **no**
 * features enabled, so it cannot express a shader that needs one, and it
 * never records a command, so it never reaches the G17P native admission
 * gates in hk_cmd_dispatch.c / hk_cmd_buffer.c.  Those gates run at RECORD
 * time and refuse with VK_ERROR_FEATURE_NOT_PRESENT, which surfaces at
 * vkEndCommandBuffer.  A feature that compiles but cannot be recorded is
 * exactly the failure mode this probe exists to find: the application has
 * already committed to the driver by then, so the refusal lands mid-frame
 * rather than at device selection.
 *
 * So this probe:
 *   1. queries every Vulkan 1.1/1.2/1.3 feature the device advertises and
 *      enables all of them, the way DXVK and VKD3D-Proton do;
 *   2. creates a compute pipeline from each SPIR-V module given;
 *   3. binds it and records one vkCmdDispatch, then reports the result of
 *      vkEndCommandBuffer.
 *
 * Needs no hardware, only the G17P drm-shim.  Exit status is the number of
 * modules that failed either stage.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_NAME "g17p-vulkan-feature-probe"
#define T8140_DEVICE_ID 0x8140

/* Descriptor set 0 is a superset layout so every probe shader binds against
 * one pipeline layout:
 *
 *   binding 0  storage buffer            [1]   every shader's output
 *   binding 1  storage buffer            [1]   scalar/typed input
 *   binding 2  storage buffer            [4]   descriptor-indexed SSBO array
 *   binding 3  combined image sampler    [4]   descriptor-indexed texture array
 *   binding 4  storage image             [1]   UAV write
 *
 * Bindings 2 and 3 carry UPDATE_AFTER_BIND | PARTIALLY_BOUND |
 * UPDATE_UNUSED_WHILE_PENDING so the descriptor-indexing probes exercise the
 * lifetimes DXVK and VKD3D-Proton actually use, not just the array shape.
 */
#define PROBE_BINDING_COUNT 5

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceFeatures2);
VK_FN(vkGetPhysicalDeviceProperties2);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkCreateDescriptorSetLayout);
VK_FN(vkDestroyDescriptorSetLayout);
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
VK_FN(vkEndCommandBuffer);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindDescriptorSets);
VK_FN(vkCmdDispatch);
VK_FN(vkResetCommandPool);

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

static uint32_t *
read_spv(const char *path, size_t *size_out)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      fprintf(stderr, TEST_NAME ": cannot open %s\n", path);
      exit(EXIT_FAILURE);
   }
   if (fseek(f, 0, SEEK_END) != 0)
      die("seek failed");
   long size = ftell(f);
   if (fseek(f, 0, SEEK_SET) != 0)
      die("seek failed");
   if (size <= 0 || (size % 4) != 0) {
      fprintf(stderr, TEST_NAME ": %s is not a SPIR-V module\n", path);
      exit(EXIT_FAILURE);
   }
   uint32_t *words = malloc((size_t)size);
   if (!words || fread(words, 1, (size_t)size, f) != (size_t)size)
      die("short read");
   fclose(f);
   *size_out = (size_t)size;
   return words;
}

/* Strip directories and the .spv suffix so the report reads as a case name. */
static void
case_name(const char *path, char *out, size_t out_size)
{
   const char *base = strrchr(path, '/');
   base = base ? base + 1 : path;
   snprintf(out, out_size, "%s", base);
   char *dot = strrchr(out, '.');
   if (dot && strcmp(dot, ".spv") == 0)
      *dot = '\0';
}

int
main(int argc, char **argv)
{
   /* `--no-update-after-bind` clears the descriptorBinding*UpdateAfterBind
    * features before vkCreateDevice, and drops the matching binding flags and
    * the UPDATE_AFTER_BIND_POOL layout flag.  Several G17P bounded descriptor
    * lowerings in hk_compile_nir() are gated on
    * `!enabled_features.descriptorBindingSampledImageUpdateAfterBind`, so the
    * same shader can compile with update-after-bind off and be refused with it
    * on.  Translation layers enable it unconditionally, so the difference
    * between the two runs of this probe is the part of the driver a real game
    * would never see working. */
   bool update_after_bind = true;
   int first_module = 1;
   for (; first_module < argc; ++first_module) {
      if (strcmp(argv[first_module], "--no-update-after-bind") == 0)
         update_after_bind = false;
      else
         break;
   }

   if (first_module >= argc) {
      fprintf(stderr, "usage: " TEST_NAME
                      " [--no-update-after-bind] <shader.spv> [more.spv ...]\n");
      return EXIT_FAILURE;
   }

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
   check(vkCreateInstance(&instance_info, NULL, &instance), "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures2);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties2);
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

   /* Enable everything the device says it supports, exactly as a translation
    * layer would.  Querying into the same structs we then pass to
    * vkCreateDevice is what makes this an advertisement test: if a feature is
    * VK_TRUE here it is enabled below, and any probe that then fails is a
    * feature the driver advertises but cannot execute. */
   VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
   };
   VkPhysicalDeviceVulkan12Features features12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .pNext = &features13,
   };
   VkPhysicalDeviceVulkan11Features features11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .pNext = &features12,
   };
   VkPhysicalDeviceFeatures2 features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &features11,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features2);

   if (!update_after_bind) {
      features12.descriptorBindingUniformBufferUpdateAfterBind = VK_FALSE;
      features12.descriptorBindingSampledImageUpdateAfterBind = VK_FALSE;
      features12.descriptorBindingStorageImageUpdateAfterBind = VK_FALSE;
      features12.descriptorBindingStorageBufferUpdateAfterBind = VK_FALSE;
      features12.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_FALSE;
      features12.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_FALSE;
      features13.descriptorBindingInlineUniformBlockUpdateAfterBind = VK_FALSE;
   }

   printf("# device %s (deviceID 0x%x)\n", properties.deviceName,
          properties.deviceID);
   printf("# update-after-bind: %s\n",
          update_after_bind ? "enabled (translation-layer default)"
                            : "DISABLED (--no-update-after-bind)");
   printf("# advertised: shaderInt64=%u shaderInt16=%u shaderFloat64=%u\n",
          features2.features.shaderInt64, features2.features.shaderInt16,
          features2.features.shaderFloat64);
   printf("# advertised: shaderInt8=%u shaderFloat16=%u bufferDeviceAddress=%u\n",
          features12.shaderInt8, features12.shaderFloat16,
          features12.bufferDeviceAddress);
   printf("# advertised: storageBuffer8Bit=%u storageBuffer16Bit=%u\n",
          features12.storageBuffer8BitAccess,
          features11.storageBuffer16BitAccess);
   printf("# advertised: descriptorIndexing=%u ssboNonUniform=%u "
          "sampledImageNonUniform=%u partiallyBound=%u variableCount=%u\n",
          features12.descriptorIndexing,
          features12.shaderStorageBufferArrayNonUniformIndexing,
          features12.shaderSampledImageArrayNonUniformIndexing,
          features12.descriptorBindingPartiallyBound,
          features12.descriptorBindingVariableDescriptorCount);
   printf("# advertised: fragmentStoresAndAtomics=%u "
          "vertexPipelineStoresAndAtomics=%u geometryShader=%u "
          "tessellationShader=%u\n",
          features2.features.fragmentStoresAndAtomics,
          features2.features.vertexPipelineStoresAndAtomics,
          features2.features.geometryShader,
          features2.features.tessellationShader);
   printf("# advertised: multiViewport=%u maxViewports=%u sparseBinding=%u "
          "multiview=%u shaderOutputViewportIndex=%u shaderOutputLayer=%u\n",
          features2.features.multiViewport, properties.limits.maxViewports,
          features2.features.sparseBinding, features11.multiview,
          features12.shaderOutputViewportIndex, features12.shaderOutputLayer);
   printf("# advertised: maxFramebufferLayers=%u maxColorAttachments=%u\n",
          properties.limits.maxFramebufferLayers,
          properties.limits.maxColorAttachments);
   printf("# advertised: multiDrawIndirect=%u drawIndirectFirstInstance=%u "
          "drawIndirectCount=%u timelineSemaphore=%u synchronization2=%u\n",
          features2.features.multiDrawIndirect,
          features2.features.drawIndirectFirstInstance,
          features12.drawIndirectCount, features12.timelineSemaphore,
          features13.synchronization2);
   printf("# advertised: robustBufferAccess=%u robustImageAccess=%u "
          "bufferDeviceAddressCaptureReplay=%u graphicsPipelineLibrary(ext)\n",
          features2.features.robustBufferAccess, features13.robustImageAccess,
          features12.bufferDeviceAddressCaptureReplay);

   VkPhysicalDeviceVulkan11Properties properties11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
   };
   VkPhysicalDeviceProperties2 properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &properties11,
   };
   vkGetPhysicalDeviceProperties2(physical, &properties2);

   static const struct {
      VkSubgroupFeatureFlagBits bit;
      const char *name;
   } subgroup_ops[] = {
      {VK_SUBGROUP_FEATURE_BASIC_BIT, "BASIC"},
      {VK_SUBGROUP_FEATURE_VOTE_BIT, "VOTE"},
      {VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, "ARITHMETIC"},
      {VK_SUBGROUP_FEATURE_BALLOT_BIT, "BALLOT"},
      {VK_SUBGROUP_FEATURE_SHUFFLE_BIT, "SHUFFLE"},
      {VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT, "SHUFFLE_RELATIVE"},
      {VK_SUBGROUP_FEATURE_CLUSTERED_BIT, "CLUSTERED"},
      {VK_SUBGROUP_FEATURE_QUAD_BIT, "QUAD"},
      {VK_SUBGROUP_FEATURE_ROTATE_BIT, "ROTATE"},
      {VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT, "ROTATE_CLUSTERED"},
   };
   printf("# advertised: subgroupSize=%u subgroupOps =",
          properties11.subgroupSize);
   for (unsigned i = 0; i < sizeof(subgroup_ops) / sizeof(subgroup_ops[0]); ++i) {
      if (properties11.subgroupSupportedOperations & subgroup_ops[i].bit)
         printf(" %s", subgroup_ops[i].name);
   }
   printf("\n");
   fflush(stdout);

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
      if (queue_families[i].queueCount != 0 &&
          (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
         queue_family = i;
         break;
      }
   }
   free(queue_families);
   if (queue_family == UINT32_MAX)
      die("device has no compute queue family");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features2,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice with every advertised feature enabled");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkCreateDescriptorSetLayout);
   LOAD_DEVICE(device, vkDestroyDescriptorSetLayout);
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
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdBindDescriptorSets);
   LOAD_DEVICE(device, vkCmdDispatch);
   LOAD_DEVICE(device, vkResetCommandPool);

   const VkDescriptorSetLayoutBinding bindings[PROBE_BINDING_COUNT] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 2,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 4,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 3,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 4,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 4,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
   };

   const VkDescriptorBindingFlags indexing_flags =
      (update_after_bind ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT : 0) |
      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
      VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
   const VkDescriptorBindingFlags binding_flags[PROBE_BINDING_COUNT] = {
      0, 0, indexing_flags, indexing_flags, 0,
   };
   const VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info = {
      .sType =
         VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .bindingCount = PROBE_BINDING_COUNT,
      .pBindingFlags = binding_flags,
   };
   const VkDescriptorSetLayoutCreateInfo set_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &flags_info,
      .flags = update_after_bind
                  ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT
                  : 0,
      .bindingCount = PROBE_BINDING_COUNT,
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

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool cmd_pool;
   check(vkCreateCommandPool(device, &pool_info, NULL, &cmd_pool),
         "vkCreateCommandPool");

   const VkCommandBufferAllocateInfo cmd_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cmd_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   check(vkAllocateCommandBuffers(device, &cmd_info, &cmd),
         "vkAllocateCommandBuffers");

   int failures = 0;
   for (int i = first_module; i < argc; ++i) {
      char name[128];
      case_name(argv[i], name, sizeof(name));

      size_t size = 0;
      uint32_t *words = read_spv(argv[i], &size);

      const VkShaderModuleCreateInfo module_info = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = size,
         .pCode = words,
      };
      VkShaderModule module;
      check(vkCreateShaderModule(device, &module_info, NULL, &module),
            "vkCreateShaderModule");

      const VkComputePipelineCreateInfo pipeline_info = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = module,
            .pName = "main",
         },
         .layout = layout,
      };
      VkPipeline pipeline = VK_NULL_HANDLE;
      const VkResult compile = vkCreateComputePipelines(
         device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipeline);

      VkResult record = VK_SUCCESS;
      bool recorded = false;
      if (compile == VK_SUCCESS && pipeline != VK_NULL_HANDLE) {
         check(vkResetCommandPool(device, cmd_pool, 0), "vkResetCommandPool");

         const VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
         };
         check(vkBeginCommandBuffer(cmd, &begin_info), "vkBeginCommandBuffer");
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
         vkCmdDispatch(cmd, 1, 1, 1);
         record = vkEndCommandBuffer(cmd);
         recorded = true;
      }

      const bool compile_ok = compile == VK_SUCCESS &&
                              pipeline != VK_NULL_HANDLE;
      const bool record_ok = recorded && record == VK_SUCCESS;

      printf("%-28s COMPILE=%-8s RECORD=%-8s (compile %d, record %d)\n", name,
             compile_ok ? "ok" : "REFUSED",
             !recorded ? "-" : record_ok ? "ok" : "REFUSED", compile, record);
      fflush(stdout);

      if (!compile_ok || !record_ok)
         ++failures;

      if (pipeline != VK_NULL_HANDLE)
         vkDestroyPipeline(device, pipeline, NULL);
      vkDestroyShaderModule(device, module, NULL);
      free(words);
   }

   vkDestroyCommandPool(device, cmd_pool, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyDescriptorSetLayout(device, set_layout, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return failures;
}
