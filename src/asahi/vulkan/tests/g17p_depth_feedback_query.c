/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT
 *
 * Exercise the complete image-format/host-copy performance query used by
 * Zink. The pure G17P policy test cannot detect a contradictory generic
 * compression check. No logical device or GPU submission is required.
 *
 * Set VK_DRIVER_FILES to the tested ICD. Argument: plain or compressed.
 * Both arms require color feedback images to retain their plain layout.
 * Loader setup follows hk_sparse_format_properties.c.
 */
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define CHECK(c) do { \
   if (!(c)) { fprintf(stderr, "depth feedback query: %s failed at %u\n", \
                       #c, __LINE__); return 1; } \
} while (0)
#define LOAD(name) \
   PFN_##name name = (PFN_##name)get_proc(instance, #name); \
   CHECK(name != NULL)

int
main(int argc, char **argv)
{
   CHECK(argc == 2);
   const bool compressed = strcmp(argv[1], "compressed") == 0;
   CHECK(compressed || strcmp(argv[1], "plain") == 0);
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   CHECK(loader != NULL);
   PFN_vkGetInstanceProcAddr get_proc =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   CHECK(get_proc != NULL);
   VkInstance instance = VK_NULL_HANDLE;
   LOAD(vkCreateInstance);
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo create = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   CHECK(vkCreateInstance(&create, NULL, &instance) == VK_SUCCESS);
   LOAD(vkDestroyInstance);
   LOAD(vkEnumeratePhysicalDevices);
   LOAD(vkGetPhysicalDeviceProperties);
   LOAD(vkGetPhysicalDeviceImageFormatProperties2);
   uint32_t count = 1;
   VkPhysicalDevice device;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &device) == VK_SUCCESS);
   CHECK(count == 1);
   VkPhysicalDeviceProperties properties;
   vkGetPhysicalDeviceProperties(device, &properties);
   CHECK(strstr(properties.deviceName, "G17P") != NULL);

   const VkFormat formats[] = {
      VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT,
      VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_R8G8B8A8_UNORM,
   };
   for (unsigned i = 0; i < sizeof(formats) / sizeof(formats[0]); ++i) {
      const bool depth = i < 3;
      VkPhysicalDeviceImageFormatInfo2 info = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
         .format = formats[i],
         .type = VK_IMAGE_TYPE_2D,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .flags = depth ?
            VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT : 0,
         .usage = VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT |
                  VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                  (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT :
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
      };
      VkHostImageCopyDevicePerformanceQueryEXT performance = {
         .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY_EXT,
      };
      VkImageFormatProperties2 image = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
         .pNext = &performance,
      };
      CHECK(vkGetPhysicalDeviceImageFormatProperties2(device, &info, &image)
            == VK_SUCCESS);
      printf("format=%u usage=%#x flags=%#x optimal=%u identical=%u\n",
             info.format, info.usage, info.flags,
             performance.optimalDeviceAccess, performance.identicalMemoryLayout);
      const bool plain = !(depth && compressed);
      CHECK(!!performance.optimalDeviceAccess == plain);
      CHECK(!!performance.identicalMemoryLayout == plain);
   }
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   puts("DEPTH_FEEDBACK_QUERY_PASS");
   return 0;
}
