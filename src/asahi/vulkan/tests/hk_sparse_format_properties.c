/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Physical-device queries only: exercise Vulkan 1.0 and the KHR properties2
 * entrypoints without creating a logical device or submitting GPU work.
 */
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { \
   if (!(c)) { fprintf(stderr, "sparse properties: %s failed at %u\n", \
                       #c, __LINE__); return 1; } \
} while (0)
#define LOAD(name) \
   PFN_##name name = (PFN_##name)get_proc(instance, #name); \
   CHECK(name != NULL)

int main(int argc, char **argv)
{
   CHECK(argc == 2);
   const bool expect_sparse = strcmp(argv[1], "sparse") == 0;
   CHECK(expect_sparse || strcmp(argv[1], "no-sparse") == 0);
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   CHECK(loader != NULL);
   PFN_vkGetInstanceProcAddr get_proc =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   CHECK(get_proc != NULL);
   VkInstance instance = VK_NULL_HANDLE;
   LOAD(vkCreateInstance);
   const char *extension = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_0,
   };
   const VkInstanceCreateInfo create = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = &extension,
   };
   CHECK(vkCreateInstance(&create, NULL, &instance) == VK_SUCCESS);
   LOAD(vkDestroyInstance);
   LOAD(vkEnumeratePhysicalDevices);
   LOAD(vkGetPhysicalDeviceFeatures);
   LOAD(vkGetPhysicalDeviceProperties);
   LOAD(vkGetPhysicalDeviceImageFormatProperties2KHR);
   LOAD(vkGetPhysicalDeviceSparseImageFormatProperties);
   LOAD(vkGetPhysicalDeviceSparseImageFormatProperties2KHR);
   uint32_t count = 1;
   VkPhysicalDevice device;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &device) == VK_SUCCESS);
   CHECK(count == 1);
   VkPhysicalDeviceFeatures features;
   vkGetPhysicalDeviceFeatures(device, &features);
   CHECK(!!features.sparseBinding == expect_sparse);
   VkPhysicalDeviceProperties properties;
   vkGetPhysicalDeviceProperties(device, &properties);
   if (!expect_sparse)
      /* "Apple A18 Pro (G17P <revision>)" since the exact product name
       * gained the GPU codename and revision (a682a30b929). */
      CHECK(strncmp(properties.deviceName, "Apple A18 Pro (G17P ",
                    strlen("Apple A18 Pro (G17P ")) == 0);

   VkPhysicalDeviceImageFormatInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
   };
   VkImageFormatProperties2 image = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
   };
   CHECK(vkGetPhysicalDeviceImageFormatProperties2KHR(device, &info, &image)
         == VK_SUCCESS);
   CHECK(image.imageFormatProperties.maxExtent.width > 0);

   const VkImageCreateFlags flags[] = {
      VK_IMAGE_CREATE_SPARSE_BINDING_BIT,
      VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
      VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
         VK_IMAGE_CREATE_SPARSE_ALIASED_BIT,
   };
   for (unsigned i = 0; i < sizeof(flags) / sizeof(flags[0]); ++i) {
      info.flags = flags[i];
      memset(&image.imageFormatProperties, 0xa5, sizeof(image.imageFormatProperties));
      CHECK(vkGetPhysicalDeviceImageFormatProperties2KHR(device, &info, &image)
            == (expect_sparse ? VK_SUCCESS : VK_ERROR_FORMAT_NOT_SUPPORTED));
      if (!expect_sparse) {
         const VkImageFormatProperties zero = {0};
         CHECK(memcmp(&image.imageFormatProperties, &zero, sizeof(zero)) == 0);
      }
   }

   const VkPhysicalDeviceSparseImageFormatInfo2 sparse = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2,
      .format = info.format, .type = info.type, .tiling = info.tiling,
      .usage = info.usage, .samples = VK_SAMPLE_COUNT_1_BIT,
   };
   uint32_t legacy_count = 0, count2 = 0;
   vkGetPhysicalDeviceSparseImageFormatProperties(device, sparse.format,
      sparse.type, sparse.samples, sparse.usage, sparse.tiling, &legacy_count, NULL);
   vkGetPhysicalDeviceSparseImageFormatProperties2KHR(device, &sparse, &count2, NULL);
   CHECK(legacy_count == (expect_sparse ? 1u : 0u));
   CHECK(count2 == legacy_count);

   VkSparseImageFormatProperties legacy[2], before[2];
   memset(legacy, 0xa5, sizeof(legacy));
   memcpy(before, legacy, sizeof(legacy));
   VkSparseImageFormatProperties2 result[2], before2[2];
   memset(result, 0xa5, sizeof(result));
   for (unsigned i = 0; i < 2; ++i) {
      result[i].sType = VK_STRUCTURE_TYPE_SPARSE_IMAGE_FORMAT_PROPERTIES_2;
      result[i].pNext = NULL;
   }
   memcpy(before2, result, sizeof(result));
   legacy_count = count2 = 2;
   vkGetPhysicalDeviceSparseImageFormatProperties(device, sparse.format,
      sparse.type, sparse.samples, sparse.usage, sparse.tiling, &legacy_count, legacy);
   vkGetPhysicalDeviceSparseImageFormatProperties2KHR(device, &sparse, &count2, result);
   CHECK(legacy_count == (expect_sparse ? 1u : 0u));
   CHECK(count2 == legacy_count);
   const unsigned untouched = expect_sparse ? 1 : 0;
   CHECK(memcmp(legacy + untouched, before + untouched,
                (2 - untouched) * sizeof(legacy[0])) == 0);
   CHECK(memcmp(result + untouched, before2 + untouched,
                (2 - untouched) * sizeof(result[0])) == 0);
   if (expect_sparse) {
      CHECK(memcmp(&legacy[0], &result[0].properties, sizeof(legacy[0])) == 0);
      CHECK(legacy[0].imageGranularity.width > 0);
   }
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   printf("PASS: sparse binding=%u, image and sparse queries agree, unused output untouched\n",
          expect_sparse);
   return 0;
}
