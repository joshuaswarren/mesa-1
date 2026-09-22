/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Record empty attachmentless passes through both Vulkan rendering APIs.
 * Zink can emit these before any draw supplies rasterizationSamples. They
 * must not poison the command buffer. No work is submitted to the GPU.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#define FUNCTIONS(F) \
   F(vkCreateInstance) \
   F(vkDestroyInstance) \
   F(vkEnumeratePhysicalDevices) \
   F(vkGetPhysicalDeviceProperties) \
   F(vkGetPhysicalDeviceQueueFamilyProperties) \
   F(vkCreateDevice) \
   F(vkDestroyDevice) \
   F(vkCreateCommandPool) \
   F(vkDestroyCommandPool) \
   F(vkAllocateCommandBuffers) \
   F(vkBeginCommandBuffer) \
   F(vkEndCommandBuffer) \
   F(vkCmdBeginRendering) \
   F(vkCmdEndRendering) \
   F(vkCreateRenderPass) \
   F(vkDestroyRenderPass) \
   F(vkCreateFramebuffer) \
   F(vkDestroyFramebuffer) \
   F(vkCmdBeginRenderPass) \
   F(vkCmdEndRenderPass)

#define DECLARE(name) static PFN_##name name;
FUNCTIONS(DECLARE)
#undef DECLARE

static void
check(VkResult result, const char *operation)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "%s: %d\n", operation, result);
      exit(EXIT_FAILURE);
   }
}

int
main(void)
{
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader) {
      fprintf(stderr, "Vulkan loader: %s\n", dlerror());
      return EXIT_FAILURE;
   }
#define LOAD(name) \
   name = (PFN_##name)dlsym(loader, #name); \
   if (!name) { fprintf(stderr, "missing %s\n", #name); return EXIT_FAILURE; }
   FUNCTIONS(LOAD)
#undef LOAD

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance), "create instance");

   uint32_t count = 0;
   check(vkEnumeratePhysicalDevices(instance, &count, NULL), "count devices");
   VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
   if (!devices || !count)
      return EXIT_FAILURE;
   check(vkEnumeratePhysicalDevices(instance, &count, devices), "enumerate devices");
   VkPhysicalDevice physical = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < count; ++i) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(devices[i], &props);
      if (props.deviceID == 0x8140)
         physical = devices[i];
   }
   free(devices);
   if (physical == VK_NULL_HANDLE) {
      fprintf(stderr, "T8140 device required\n");
      return EXIT_FAILURE;
   }

   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
   VkQueueFamilyProperties *families = calloc(count, sizeof(*families));
   if (!families || !count)
      return EXIT_FAILURE;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families);
   uint32_t family = 0;
   while (family < count && !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
      ++family;
   free(families);
   if (family == count)
      return EXIT_FAILURE;

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkPhysicalDeviceDynamicRenderingFeatures dynamic = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
      .dynamicRendering = VK_TRUE,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &dynamic,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device), "create device");
   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VkCommandPool pool;
   check(vkCreateCommandPool(device, &pool_info, NULL, &pool), "create pool");
   const VkCommandBufferAllocateInfo allocate = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 2,
   };
   VkCommandBuffer commands[2];
   check(vkAllocateCommandBuffers(device, &allocate, commands), "allocate commands");
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   const VkRenderingInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea.extent = {32, 32},
      .layerCount = 1,
   };
   check(vkBeginCommandBuffer(commands[0], &begin), "begin dynamic commands");
   vkCmdBeginRendering(commands[0], &rendering);
   vkCmdEndRendering(commands[0]);
   check(vkEndCommandBuffer(commands[0]), "end attachmentless dynamic commands");

   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
   };
   const VkRenderPassCreateInfo pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .subpassCount = 1,
      .pSubpasses = &subpass,
   };
   VkRenderPass pass;
   check(vkCreateRenderPass(device, &pass_info, NULL, &pass), "create render pass");
   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = pass,
      .width = 32, .height = 32, .layers = 1,
   };
   VkFramebuffer framebuffer;
   check(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer),
         "create framebuffer");
   const VkRenderPassBeginInfo pass_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = pass,
      .framebuffer = framebuffer,
      .renderArea.extent = {32, 32},
   };
   check(vkBeginCommandBuffer(commands[1], &begin), "begin legacy commands");
   vkCmdBeginRenderPass(commands[1], &pass_begin, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdEndRenderPass(commands[1]);
   check(vkEndCommandBuffer(commands[1]), "end attachmentless legacy commands");

   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyFramebuffer(device, framebuffer, NULL);
   vkDestroyRenderPass(device, pass, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   puts("PASS: attachmentless dynamic and legacy rendering recorded (no submission)");
   return EXIT_SUCCESS;
}
