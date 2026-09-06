/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "g17p_wsi_triangle_frag.h"
#include "g17p_wsi_triangle_vert.h"

#define T8140_DEVICE_ID 0x8140

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
VK_FN(vkCreateShaderModule);
VK_FN(vkDestroyShaderModule);
VK_FN(vkCreateRenderPass);
VK_FN(vkDestroyRenderPass);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkDestroyPipelineLayout);
VK_FN(vkCreateGraphicsPipelines);
VK_FN(vkDestroyPipeline);

static void
die(const char *message)
{
   fprintf(stderr, "g17p-graphics-pipeline-library: %s\n", message);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "g17p-graphics-pipeline-library: %s returned %d\n",
              what, result);
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

static VkPipeline
create_library(VkDevice device, VkGraphicsPipelineLibraryFlagsEXT part,
               const VkGraphicsPipelineCreateInfo *template,
               const char *name)
{
   VkGraphicsPipelineLibraryCreateInfoEXT library = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
      .flags = part,
   };
   VkGraphicsPipelineCreateInfo info = *template;
   info.pNext = &library;
   info.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;

   VkPipeline pipeline = VK_NULL_HANDLE;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, NULL,
                                   &pipeline),
         name);
   if (!pipeline)
      die("library creation returned a null handle");
   return pipeline;
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

   const VkApplicationInfo application = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "g17p-graphics-pipeline-library",
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &application,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures2);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties2);
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

      uint32_t queue_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(devices[d], &queue_count, NULL);
      VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
      if (!queues)
         die("out of host memory");
      vkGetPhysicalDeviceQueueFamilyProperties(devices[d], &queue_count,
                                                queues);
      for (uint32_t q = 0; q < queue_count; ++q) {
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
      die("no T8140 graphics device");

   VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT library_feature = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &library_feature,
   };
   vkGetPhysicalDeviceFeatures2(physical, &features);
   if (!library_feature.graphicsPipelineLibrary)
      die("graphicsPipelineLibrary is not advertised");

   VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT library_properties = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT,
   };
   VkPhysicalDeviceProperties2 properties = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &library_properties,
   };
   vkGetPhysicalDeviceProperties2(physical, &properties);
   if (!library_properties.graphicsPipelineLibraryFastLinking ||
       !library_properties
           .graphicsPipelineLibraryIndependentInterpolationDecoration)
      die("graphics pipeline library properties are incomplete");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   library_feature.graphicsPipelineLibrary = VK_TRUE;
   const char *extensions[] = {
      VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
      VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &library_feature,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 2,
      .ppEnabledExtensionNames = extensions,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkDestroyShaderModule);
   LOAD_DEVICE(device, vkCreateRenderPass);
   LOAD_DEVICE(device, vkDestroyRenderPass);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkDestroyPipelineLayout);
   LOAD_DEVICE(device, vkCreateGraphicsPipelines);
   LOAD_DEVICE(device, vkDestroyPipeline);

   const VkAttachmentDescription attachment = {
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
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
      .attachmentCount = 1,
      .pAttachments = &attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &render_pass_info, NULL, &render_pass),
         "vkCreateRenderPass");

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
   const VkShaderModuleCreateInfo fragment_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_wsi_triangle_frag_spv),
      .pCode = g17p_wsi_triangle_frag_spv,
   };
   VkShaderModule vertex_module, fragment_module;
   check(vkCreateShaderModule(device, &vertex_module_info, NULL,
                              &vertex_module),
         "vkCreateShaderModule(vertex)");
   check(vkCreateShaderModule(device, &fragment_module_info, NULL,
                              &fragment_module),
         "vkCreateShaderModule(fragment)");

   const VkPipelineShaderStageCreateInfo vertex_stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = vertex_module,
      .pName = "main",
   };
   const VkPipelineShaderStageCreateInfo fragment_stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = fragment_module,
      .pName = "main",
   };
   const VkVertexInputBindingDescription binding = {
      .binding = 0,
      .stride = 8,
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription attribute = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
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
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {
      .width = 32.0f,
      .height = 32.0f,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };
   const VkRect2D scissor = {.extent = {32, 32}};
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
   const VkPipelineColorBlendAttachmentState color_attachment = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &color_attachment,
   };

   VkGraphicsPipelineCreateInfo library_template = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .layout = layout,
      .renderPass = render_pass,
   };
   library_template.pVertexInputState = &vertex_input;
   library_template.pInputAssemblyState = &input_assembly;
   VkPipeline libraries[4];
   libraries[0] = create_library(
      device, VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT,
      &library_template, "vertex-input library");

   library_template.pVertexInputState = NULL;
   library_template.pInputAssemblyState = NULL;
   library_template.stageCount = 1;
   library_template.pStages = &vertex_stage;
   library_template.pViewportState = &viewport_state;
   library_template.pRasterizationState = &raster;
   libraries[1] = create_library(
      device, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
      &library_template, "pre-raster library");

   library_template.pViewportState = NULL;
   library_template.pRasterizationState = NULL;
   library_template.pStages = &fragment_stage;
   library_template.pMultisampleState = &multisample;
   libraries[2] = create_library(
      device, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT,
      &library_template, "fragment-shader library");

   library_template.stageCount = 0;
   library_template.pStages = NULL;
   library_template.pColorBlendState = &blend;
   libraries[3] = create_library(
      device, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT,
      &library_template, "fragment-output library");

   const VkPipelineLibraryCreateInfoKHR link_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
      .libraryCount = 4,
      .pLibraries = libraries,
   };
   const VkGraphicsPipelineCreateInfo final_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &link_info,
      .layout = layout,
      .renderPass = render_pass,
   };
   VkPipeline final_pipeline = VK_NULL_HANDLE;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &final_info,
                                   NULL, &final_pipeline),
         "four-part graphics pipeline link");
   if (!final_pipeline)
      die("final link returned a null handle");

   for (uint32_t i = 0; i < 4; ++i)
      vkDestroyPipeline(device, libraries[i], NULL);
   vkDestroyShaderModule(device, fragment_module, NULL);
   vkDestroyShaderModule(device, vertex_module, NULL);

   /* A linked pipeline owns its imported state and must remain valid after
    * every source library and shader module has been destroyed. */
   vkDestroyPipeline(device, final_pipeline, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyRenderPass(device, render_pass, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);

   puts("G17P GRAPHICS PIPELINE LIBRARY PASS: four parts linked and independently owned");
   return EXIT_SUCCESS;
}
