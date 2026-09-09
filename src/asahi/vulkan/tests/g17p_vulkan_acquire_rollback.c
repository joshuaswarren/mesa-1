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

#include "vulkan/wsi/wsi_common_private.h"

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
VK_FN(vkReleaseSwapchainImagesKHR);

static VkResult (*backend_release_images)(struct wsi_swapchain *, uint32_t,
                                          const uint32_t *);
static struct wsi_swapchain *expected_release_chain;
static uint32_t expected_release_index;
static unsigned release_calls;
static bool release_shape_valid;

struct present_override {
   struct wsi_swapchain *chain;
   VkResult (*queue_present)(struct wsi_swapchain *, uint32_t, uint64_t,
                             const VkPresentRegionKHR *);
   VkResult result;
   unsigned calls;
};

static struct present_override present_overrides[2];
static bool present_shape_valid;

static VkResult
override_present_result(struct wsi_swapchain *chain, uint32_t image_index,
                        uint64_t present_id,
                        const VkPresentRegionKHR *damage)
{
   for (uint32_t i = 0; i < 2; i++) {
      struct present_override *override = &present_overrides[i];
      if (override->chain != chain)
         continue;

      override->calls++;
      present_shape_valid &= image_index < chain->image_count &&
                             present_id == 0 && damage == NULL;
      VkResult result = override->queue_present(
         chain, image_index, present_id, damage);
      return result == VK_SUCCESS ? override->result : result;
   }

   present_shape_valid = false;
   return VK_ERROR_UNKNOWN;
}

static VkResult
fail_first_release(struct wsi_swapchain *chain, uint32_t count,
                   const uint32_t *indices)
{
   release_calls++;
   release_shape_valid &= chain == expected_release_chain && count == 1 &&
                          indices && indices[0] == expected_release_index;
   if (release_calls == 1)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   return backend_release_images(chain, count, indices);
}

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
present_waiting(VkQueue queue, VkSwapchainKHR swapchain,
                VkSemaphore render_done, uint32_t image_index,
                VkResult *per_swapchain)
{
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

static VkResult
submit_and_present_pair(VkQueue queue, const VkSwapchainKHR swapchains[2],
                        const VkSemaphore acquired[2],
                        VkSemaphore render_done, const uint32_t image_indices[2],
                        VkResult results[2])
{
   const VkPipelineStageFlags wait_stages[2] = {
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
   };
   const VkSubmitInfo submit_info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .waitSemaphoreCount = 2,
      .pWaitSemaphores = acquired,
      .pWaitDstStageMask = wait_stages,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &render_done,
   };
   check(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE),
         "vkQueueSubmit(pair)");

   results[0] = results[1] = VK_SUCCESS;
   const VkPresentInfoKHR present_info = {
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &render_done,
      .swapchainCount = 2,
      .pSwapchains = swapchains,
      .pImageIndices = image_indices,
      .pResults = results,
   };
   return vkQueuePresentKHR(queue, &present_info);
}

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

   return present_waiting(queue, swapchain, render_done, image_index,
                          per_swapchain);
}

int
main(int argc, char **argv)
{
   if (argc != 2 || (strcmp(argv[1], "--semaphore-failure") &&
                     strcmp(argv[1], "--fence-failure") &&
                     strcmp(argv[1], "--present-import-failure") &&
                     strcmp(argv[1], "--present-export-failure") &&
                     strcmp(argv[1], "--present-submit-failure") &&
                     strcmp(argv[1], "--empty-submit-failure") &&
                     strcmp(argv[1], "--release-failure") &&
                     strcmp(argv[1], "--present-priority")))
      die("usage: " TEST_NAME " <--semaphore-failure|--fence-failure|"
           "--present-import-failure|--present-export-failure|"
           "--present-submit-failure|--empty-submit-failure|"
           "--release-failure|--present-priority>");

   const char *test_case = argv[1] + 2;
   const bool empty_submit_failure =
      !strcmp(test_case, "empty-submit-failure");
   const bool release_failure = !strcmp(test_case, "release-failure");
   const bool present_priority = !strcmp(test_case, "present-priority");
   const bool present_submit_failure =
      !strcmp(test_case, "present-submit-failure");
   const bool present_failure =
      !strcmp(test_case, "present-import-failure") ||
      !strcmp(test_case, "present-export-failure") ||
      present_submit_failure;
   const char *configured = getenv((present_failure || empty_submit_failure ||
                                     release_failure || present_priority)
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
      VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME,
   };
   const VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR maintenance1 = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR,
      .swapchainMaintenance1 = VK_TRUE,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = release_failure ? &maintenance1 : NULL,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = release_failure ? 2 : 1,
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
   if (release_failure)
      LOAD_DEVICE(device, vkReleaseSwapchainImagesKHR);

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

   if (present_priority) {
      VkHeadlessSurfaceCreateInfoEXT second_surface_info = {
         .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
      };
      VkSurfaceKHR second_surface;
      check(vkCreateHeadlessSurfaceEXT(instance, &second_surface_info, NULL,
                                       &second_surface),
            "vkCreateHeadlessSurfaceEXT(second)");

      VkSwapchainCreateInfoKHR second_swapchain_info = swapchain_info;
      second_swapchain_info.surface = second_surface;
      VkSwapchainKHR second_swapchain;
      check(vkCreateSwapchainKHR(device, &second_swapchain_info, NULL,
                                 &second_swapchain),
            "vkCreateSwapchainKHR(second)");

      VkSemaphore second_acquired;
      check(vkCreateSemaphore(device, &semaphore_info, NULL,
                              &second_acquired),
            "vkCreateSemaphore(second acquired)");
      const VkSemaphore acquire_semaphores[2] = {acquired, second_acquired};

      uint32_t image_indices[2] = {UINT32_MAX, UINT32_MAX};
      check(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                  acquire_semaphores[0], VK_NULL_HANDLE,
                                  &image_indices[0]),
            "vkAcquireNextImageKHR(priority first)");
      check(vkAcquireNextImageKHR(device, second_swapchain, UINT64_MAX,
                                  acquire_semaphores[1], VK_NULL_HANDLE,
                                  &image_indices[1]),
            "vkAcquireNextImageKHR(priority second)");

      present_overrides[0] = (struct present_override) {
         .chain = wsi_swapchain_from_handle(swapchain),
      };
      present_overrides[1] = (struct present_override) {
         .chain = wsi_swapchain_from_handle(second_swapchain),
      };
      for (uint32_t i = 0; i < 2; i++) {
         present_overrides[i].queue_present =
            present_overrides[i].chain->queue_present;
         present_overrides[i].chain->queue_present = override_present_result;
      }
      present_shape_valid = true;

      const VkSwapchainKHR swapchains[2] = {
         swapchain,
         second_swapchain,
      };
      const VkResult priority_order[] = {
         VK_SUCCESS,
         VK_SUBOPTIMAL_KHR,
         VK_ERROR_OUT_OF_DATE_KHR,
         VK_ERROR_SURFACE_LOST_KHR,
         VK_ERROR_DEVICE_LOST,
      };
      VkResult per_swapchain[2];
      unsigned iteration = 0;
      for (unsigned high = 1;
           high < sizeof(priority_order) / sizeof(priority_order[0]); high++) {
         present_overrides[0].result = priority_order[high - 1];
         present_overrides[1].result = priority_order[high];

         for (unsigned reverse = 0; reverse < 2; reverse++, iteration++) {
            const unsigned first = reverse ? 1 : 0;
            const unsigned second = reverse ? 0 : 1;
            const VkSwapchainKHR ordered_swapchains[2] = {
               swapchains[first],
               swapchains[second],
            };
            const uint32_t ordered_indices[2] = {
               image_indices[first],
               image_indices[second],
            };
            const VkResult expected_results[2] = {
               present_overrides[first].result,
               present_overrides[second].result,
            };

            VkResult result = submit_and_present_pair(
               queue, ordered_swapchains, acquire_semaphores, render_done,
               ordered_indices, per_swapchain);
            if (result != priority_order[high] ||
                per_swapchain[0] != expected_results[0] ||
                per_swapchain[1] != expected_results[1]) {
               fprintf(stderr,
                       TEST_NAME ": priority present %u returned %d [%d, %d], "
                                 "expected %d [%d, %d]\n",
                       iteration, result, per_swapchain[0], per_swapchain[1],
                       priority_order[high], expected_results[0],
                       expected_results[1]);
               return EXIT_FAILURE;
            }

            const bool final_iteration =
               high + 1 == sizeof(priority_order) / sizeof(priority_order[0]) &&
               reverse == 1;
            if (!final_iteration) {
               check(vkAcquireNextImageKHR(
                        device, swapchain, UINT64_MAX, acquire_semaphores[0],
                        VK_NULL_HANDLE, &image_indices[0]),
                     "vkAcquireNextImageKHR(priority repeat first)");
               check(vkAcquireNextImageKHR(
                        device, second_swapchain, UINT64_MAX,
                        acquire_semaphores[1], VK_NULL_HANDLE,
                        &image_indices[1]),
                     "vkAcquireNextImageKHR(priority repeat second)");
            }
         }
      }
      if (!present_shape_valid || present_overrides[0].calls != iteration ||
          present_overrides[1].calls != iteration)
         die("present backends did not receive every exact request");

      for (uint32_t i = 0; i < 2; i++) {
         present_overrides[i].chain->queue_present =
            present_overrides[i].queue_present;
      }
      check(vkQueueWaitIdle(queue), "vkQueueWaitIdle(priority)");
      vkDestroySemaphore(device, second_acquired, NULL);
      vkDestroySemaphore(device, render_done, NULL);
      vkDestroySemaphore(device, acquired, NULL);
      vkDestroySwapchainKHR(device, second_swapchain, NULL);
      vkDestroySwapchainKHR(device, swapchain, NULL);
      vkDestroySurfaceKHR(instance, second_surface, NULL);
      vkDestroyDevice(device, NULL);
      vkDestroySurfaceKHR(instance, surface, NULL);
      vkDestroyInstance(instance, NULL);
      dlclose(loader);
      printf(TEST_NAME ": PASS — device/surface/out-of-date/suboptimal "
                       "priorities were order-independent across two "
                       "swapchains\n");
      return EXIT_SUCCESS;
   }

   if (release_failure) {
      uint32_t image_index = UINT32_MAX;
      check(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired,
                                  VK_NULL_HANDLE, &image_index),
            "vkAcquireNextImageKHR(release)");
      if (image_index != 0)
         die("first release acquire did not select image zero");

      struct wsi_swapchain *internal_chain =
         wsi_swapchain_from_handle(swapchain);
      struct wsi_image *internal_image =
         internal_chain->get_wsi_image(internal_chain, image_index);
      if (!internal_image->acquired)
         die("acquired image is not owned by the application");

      backend_release_images = internal_chain->release_images;
      expected_release_chain = internal_chain;
      expected_release_index = image_index;
      release_shape_valid = true;
      internal_chain->release_images = fail_first_release;

      const VkReleaseSwapchainImagesInfoKHR release_info = {
         .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
         .swapchain = swapchain,
         .imageIndexCount = 1,
         .pImageIndices = &image_index,
      };
      VkResult result = vkReleaseSwapchainImagesKHR(device, &release_info);
      if (result != VK_ERROR_OUT_OF_HOST_MEMORY) {
         fprintf(stderr,
                 TEST_NAME ": injected image release returned %d, expected %d\n",
                 result, VK_ERROR_OUT_OF_HOST_MEMORY);
         return EXIT_FAILURE;
      }
      if (!internal_image->acquired)
         die("failed backend release discarded common image ownership");

      check(vkReleaseSwapchainImagesKHR(device, &release_info),
            "vkReleaseSwapchainImagesKHR(retry)");
      if (internal_image->acquired)
         die("successful release retained common image ownership");
      if (release_calls != 2 || !release_shape_valid)
         die("release backend did not receive two exact image-zero requests");
      internal_chain->release_images = backend_release_images;

      uint32_t reacquired_index = UINT32_MAX;
      check(vkAcquireNextImageKHR(device, swapchain, 0, render_done,
                                  VK_NULL_HANDLE, &reacquired_index),
            "vkAcquireNextImageKHR(released image)");
      if (reacquired_index != image_index)
         die("released image was not returned to the available pool");

      const VkReleaseSwapchainImagesInfoKHR final_release_info = {
         .sType = VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR,
         .swapchain = swapchain,
         .imageIndexCount = 1,
         .pImageIndices = &reacquired_index,
      };
      check(vkReleaseSwapchainImagesKHR(device, &final_release_info),
            "vkReleaseSwapchainImagesKHR(final)");

      vkDestroySemaphore(device, render_done, NULL);
      vkDestroySemaphore(device, acquired, NULL);
      vkDestroySwapchainKHR(device, swapchain, NULL);
      vkDestroyDevice(device, NULL);
      vkDestroySurfaceKHR(instance, surface, NULL);
      vkDestroyInstance(instance, NULL);
      dlclose(loader);
      printf(TEST_NAME ": PASS — failed image release preserved ownership; "
                       "retry released and reacquired image zero\n");
      return EXIT_SUCCESS;
   }

   if (empty_submit_failure) {
      uint32_t image_index = UINT32_MAX;
      check(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired,
                                  VK_NULL_HANDLE, &image_index),
            "vkAcquireNextImageKHR(empty submit)");

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
      VkResult result =
         vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
      if (result != VK_ERROR_DEVICE_LOST) {
         fprintf(stderr,
                 TEST_NAME ": injected empty submit returned %d, expected %d\n",
                 result, VK_ERROR_DEVICE_LOST);
         return EXIT_FAILURE;
      }

      vkDestroySemaphore(device, render_done, NULL);
      vkDestroySemaphore(device, acquired, NULL);
      vkDestroySwapchainKHR(device, swapchain, NULL);
      vkDestroyDevice(device, NULL);
      vkDestroySurfaceKHR(instance, surface, NULL);
      vkDestroyInstance(instance, NULL);
      dlclose(loader);
      printf(TEST_NAME ": PASS — failed empty-submit sync transfer was "
                       "propagated as device loss; teardown completed\n");
      return EXIT_SUCCESS;
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
      if (present_submit_failure) {
         if (result != VK_ERROR_UNKNOWN ||
             per_swapchain != VK_ERROR_UNKNOWN) {
            fprintf(stderr,
                    TEST_NAME ": injected pre-present submit returned %d/%d, "
                              "expected %d\n",
                    result, per_swapchain, VK_ERROR_UNKNOWN);
            return EXIT_FAILURE;
         }

         result = present_waiting(queue, swapchain, render_done, first_image,
                                  &per_swapchain);
      }
      if (result != VK_SUCCESS || per_swapchain != VK_SUCCESS) {
         fprintf(stderr,
                 TEST_NAME ": WSI present recovery returned %d/%d\n",
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
      if (present_submit_failure) {
         printf(TEST_NAME ": PASS — failed pre-present sync submission was "
                          "cleaned; retry presented and reacquired image zero\n");
      } else {
         printf(TEST_NAME ": PASS — dma-buf sync-file %s failure fell back "
                          "to a CPU wait; image zero was presented and "
                          "reacquired\n",
                strstr(test_case, "import") ? "import" : "export");
      }
   } else {
      printf(TEST_NAME ": PASS — %s rolled back image zero and temporary "
                       "payloads; retry submit/present/teardown completed\n",
             test_case);
   }
   return EXIT_SUCCESS;
}
