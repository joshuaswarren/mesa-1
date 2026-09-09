/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Public Vulkan oracle for WSI synchronization failure recovery.  A headless
 * swapchain keeps this runnable under the G17P DRM shim while still exercising
 * the common acquire, queue submit, present, and teardown paths.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_NAME "g17p-vulkan-wsi-rollback"
#define T8140_DEVICE_ID 0x8140

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkCreateHeadlessSurfaceEXT);
VK_FN(vkDestroySurfaceKHR);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkGetPhysicalDeviceSurfaceSupportKHR);
VK_FN(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
VK_FN(vkGetPhysicalDeviceSurfaceFormatsKHR);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkGetDeviceQueue);
VK_FN(vkCreateSwapchainKHR);
VK_FN(vkDestroySwapchainKHR);
VK_FN(vkCreateSemaphore);
VK_FN(vkDestroySemaphore);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkAcquireNextImageKHR);
VK_FN(vkQueueSubmit);
VK_FN(vkQueuePresentKHR);
VK_FN(vkQueueWaitIdle);

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

static VkResult
submit_and_present(VkQueue queue, VkSwapchainKHR swapchain,
                   VkSemaphore acquired, VkSemaphore render_done,
                   uint32_t image_index, VkResult *per_swapchain)
{
   const VkPipelineStageFlags wait_stage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
   const VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &acquired,
      .pWaitDstStageMask = &wait_stage,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &render_done,
   };
   check(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE),
         "vkQueueSubmit");

   *per_swapchain = VK_SUCCESS;
   const VkPresentInfoKHR present_info = {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &render_done,
      .swapchainCount = 1,
      .pSwapchains = &swapchain,
      .pImageIndices = &image_index,
      .pResults = per_swapchain,
   };
   return vkQueuePresentKHR(queue, &present_info);
}

int
main(int argc, char **argv)
{
   if (argc != 2 || (strcmp(argv[1], "--semaphore-failure") &&
                     strcmp(argv[1], "--fence-failure") &&
                     strcmp(argv[1], "--present-import-failure") &&
                     strcmp(argv[1], "--present-export-failure")))
      die("usage: " TEST_NAME " <--semaphore-failure|--fence-failure|"
           "--present-import-failure|--present-export-failure>");

   const char *test_case = argv[1] + 2;
   const bool present_failure =
      !strcmp(test_case, "present-import-failure") ||
      !strcmp(test_case, "present-export-failure");
   const char *configured = getenv(present_failure
      ? "ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK"
      : "ASAHI_SHIM_EXPECT_ACQUIRE_ROLLBACK");
   if (!configured || strcmp(configured, test_case))
      die("DRM-shim WSI rollback oracle is not configured for this case");

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
      .apiVersion = VK_API_VERSION_1_1,
   };
   const char *const instance_extensions[] = {
      VK_KHR_SURFACE_EXTENSION_NAME,
      VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &application_info,
      .enabledExtensionCount = 2,
      .ppEnabledExtensionNames = instance_extensions,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkCreateHeadlessSurfaceEXT);
   LOAD_INSTANCE(instance, vkDestroySurfaceKHR);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceSurfaceSupportKHR);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceSurfaceFormatsKHR);
   LOAD_INSTANCE(instance, vkCreateDevice);

   const VkHeadlessSurfaceCreateInfoEXT surface_info = {
      .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
   };
   VkSurfaceKHR surface;
   check(vkCreateHeadlessSurfaceEXT(instance, &surface_info, NULL, &surface),
         "vkCreateHeadlessSurfaceEXT");

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
      VkBool32 supported = VK_FALSE;
      check(vkGetPhysicalDeviceSurfaceSupportKHR(physical, i, surface,
                                                 &supported),
            "vkGetPhysicalDeviceSurfaceSupportKHR");
      if (queue_families[i].queueCount && supported) {
         queue_family = i;
         break;
      }
   }
   free(queue_families);
   if (queue_family == UINT32_MAX)
      die("device has no present-capable queue family");

   VkSurfaceCapabilitiesKHR capabilities;
   check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface,
                                                    &capabilities),
         "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
   uint32_t format_count = 0;
   check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface,
                                              &format_count, NULL),
         "vkGetPhysicalDeviceSurfaceFormatsKHR(count)");
   if (!format_count)
      die("headless surface exposes no format");
   VkSurfaceFormatKHR *formats = calloc(format_count, sizeof(*formats));
   if (!formats)
      die("out of host memory");
   check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface,
                                              &format_count, formats),
         "vkGetPhysicalDeviceSurfaceFormatsKHR(list)");
   const VkSurfaceFormatKHR format = formats[0];
   free(formats);

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const char *const device_extensions[] = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = device_extensions,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCreateSwapchainKHR);
   LOAD_DEVICE(device, vkDestroySwapchainKHR);
   LOAD_DEVICE(device, vkCreateSemaphore);
   LOAD_DEVICE(device, vkDestroySemaphore);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkDestroyFence);
   LOAD_DEVICE(device, vkAcquireNextImageKHR);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkQueuePresentKHR);
   LOAD_DEVICE(device, vkQueueWaitIdle);

   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);

   const VkSwapchainCreateInfoKHR swapchain_info = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = surface,
      .minImageCount = capabilities.minImageCount,
      .imageFormat = format.format,
      .imageColorSpace = format.colorSpace,
      .imageExtent = {64, 64},
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .preTransform = capabilities.currentTransform,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = VK_PRESENT_MODE_FIFO_KHR,
      .clipped = VK_TRUE,
   };
   VkSwapchainKHR swapchain;
   check(vkCreateSwapchainKHR(device, &swapchain_info, NULL, &swapchain),
         "vkCreateSwapchainKHR");

   const VkSemaphoreCreateInfo semaphore_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
   };
   VkSemaphore acquired, render_done, held_acquired = VK_NULL_HANDLE;
   check(vkCreateSemaphore(device, &semaphore_info, NULL, &acquired),
         "vkCreateSemaphore(acquired)");
   check(vkCreateSemaphore(device, &semaphore_info, NULL, &render_done),
         "vkCreateSemaphore(render_done)");
   if (present_failure) {
      check(vkCreateSemaphore(device, &semaphore_info, NULL, &held_acquired),
            "vkCreateSemaphore(held acquired)");
   }
   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence acquired_fence = VK_NULL_HANDLE;
   if (!present_failure) {
      check(vkCreateFence(device, &fence_info, NULL, &acquired_fence),
            "vkCreateFence(acquired)");

      uint32_t ignored_index = UINT32_MAX;
      VkResult result =
         vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired,
                               acquired_fence, &ignored_index);
      if (result != VK_ERROR_UNKNOWN) {
         fprintf(stderr,
                 TEST_NAME ": injected acquire returned %d, expected %d\n",
                 result, VK_ERROR_UNKNOWN);
         return EXIT_FAILURE;
      }

      /* This creates a new syncobj immediately after AcquireNextImage
       * returns.  The shim uses it as a marker and requires every temporary
       * payload from the failed acquire to have been destroyed first. */
      VkFence marker_fence;
      check(vkCreateFence(device, &fence_info, NULL, &marker_fence),
            "vkCreateFence(cleanup marker)");
      vkDestroyFence(device, marker_fence, NULL);
   } else {
      uint32_t first_image = UINT32_MAX;
      check(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired,
                                  VK_NULL_HANDLE, &first_image),
            "vkAcquireNextImageKHR(first present)");
      if (first_image != 0)
         die("first acquire did not select image zero");

      VkResult per_swapchain = VK_SUCCESS;
      VkResult result = submit_and_present(queue, swapchain, acquired,
                                           render_done, first_image,
                                           &per_swapchain);
      if (result != VK_SUCCESS || per_swapchain != VK_SUCCESS) {
         fprintf(stderr,
                 TEST_NAME ": dma-buf sync-file fallback returned %d/%d\n",
                 result, per_swapchain);
         return EXIT_FAILURE;
      }
   }

   uint32_t held_image = UINT32_MAX;
   uint32_t image_index = UINT32_MAX;
   if (present_failure) {
      check(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, held_acquired,
                                  VK_NULL_HANDLE, &held_image),
            "vkAcquireNextImageKHR(held image)");
      if (held_image == 0)
         die("headless present did not defer its just-presented image");

      check(vkAcquireNextImageKHR(device, swapchain, 0, acquired,
                                  VK_NULL_HANDLE, &image_index),
            "vkAcquireNextImageKHR(reacquire presented image)");
   } else {
      check(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired,
                                  acquired_fence, &image_index),
            "vkAcquireNextImageKHR(retry)");
   }
   if (image_index != 0)
      die("failed acquire leaked image zero from the available pool");

   VkResult per_swapchain = VK_SUCCESS;
   check(submit_and_present(queue, swapchain, acquired, render_done,
                            image_index, &per_swapchain),
         "vkQueuePresentKHR(recovered image)");
   check(per_swapchain, "VkPresentInfoKHR::pResults[0]");

   if (present_failure) {
      check(submit_and_present(queue, swapchain, held_acquired, render_done,
                               held_image, &per_swapchain),
            "vkQueuePresentKHR(held image)");
      check(per_swapchain, "held VkPresentInfoKHR::pResults[0]");
   }
   check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");

   if (acquired_fence != VK_NULL_HANDLE)
      vkDestroyFence(device, acquired_fence, NULL);
   vkDestroySemaphore(device, render_done, NULL);
   vkDestroySemaphore(device, acquired, NULL);
   if (held_acquired != VK_NULL_HANDLE)
      vkDestroySemaphore(device, held_acquired, NULL);
   vkDestroySwapchainKHR(device, swapchain, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroySurfaceKHR(instance, surface, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);

   if (present_failure) {
      printf(TEST_NAME ": PASS — dma-buf sync-file %s failure fell back to "
                       "a CPU wait; image zero was presented and reacquired\n",
             strstr(test_case, "import") ? "import" : "export");
   } else {
      printf(TEST_NAME ": PASS — %s rolled back image zero and temporary "
                       "payloads; retry submit/present/teardown completed\n",
             test_case);
   }
   return EXIT_SUCCESS;
}
