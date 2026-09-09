/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef G17P_BLIT_IMAGE
#define G17P_BLIT_IMAGE 0
#endif

#ifndef G17P_RESOLVE_IMAGE
#define G17P_RESOLVE_IMAGE 0
#endif

#ifndef G17P_COPY_IMAGE_3D
#define G17P_COPY_IMAGE_3D 0
#endif

#ifndef G17P_STENCIL_BLIT
#define G17P_STENCIL_BLIT 0
#endif

#if ((G17P_BLIT_IMAGE != 0) + (G17P_RESOLVE_IMAGE != 0) + \
     (G17P_COPY_IMAGE_3D != 0) + (G17P_STENCIL_BLIT != 0)) > 1
#error "select exactly one image operation oracle"
#endif

#ifdef G17P_CLEAR_IMAGE
#define TEST_NAME "g17p-vulkan-clear-image"
#elif G17P_STENCIL_BLIT
#define TEST_NAME "g17p-vulkan-stencil-blit"
#elif G17P_BLIT_IMAGE == 1
#define TEST_NAME "g17p-vulkan-blit-image-nearest-flip"
#elif G17P_BLIT_IMAGE == 2
#define TEST_NAME "g17p-vulkan-blit-image-linear-flip"
#elif G17P_BLIT_IMAGE == 3
#define TEST_NAME "g17p-vulkan-blit-cube-layers-mip"
#elif G17P_RESOLVE_IMAGE
#define TEST_NAME "g17p-vulkan-resolve-image4"
#elif G17P_COPY_IMAGE_3D
#define TEST_NAME "g17p-vulkan-image-copy-3d"
#else
#define TEST_NAME "g17p-vulkan-image-copy-subregion"
#endif
#define T8140_DEVICE_ID 0x8140
#define WIDTH 32u
#if G17P_BLIT_IMAGE == 3
#define HEIGHT 32u
#define IMAGE_BASE_WIDTH 64u
#define IMAGE_BASE_HEIGHT 64u
#define IMAGE_DEPTH 1u
#define IMAGE_LAYERS 12u
#define IMAGE_MIP_LEVELS 2u
#define OUTPUT_SLICES 6u
#define TEST_LAYER_COUNT OUTPUT_SLICES
#define SOURCE_BASE_LAYER 6u
#define DESTINATION_BASE_LAYER 0u
#define SOURCE_MIP_LEVEL 0u
#define DESTINATION_MIP_LEVEL 1u
#else
#define HEIGHT 24u
#define IMAGE_BASE_WIDTH WIDTH
#define IMAGE_BASE_HEIGHT HEIGHT
#if G17P_COPY_IMAGE_3D
#define IMAGE_DEPTH 5u
#define OUTPUT_SLICES IMAGE_DEPTH
#else
#define IMAGE_DEPTH 1u
#define OUTPUT_SLICES 1u
#endif
#define TEST_LAYER_COUNT 1u
#define IMAGE_LAYERS 1u
#define IMAGE_MIP_LEVELS 1u
#define SOURCE_BASE_LAYER 0u
#define DESTINATION_BASE_LAYER 0u
#define SOURCE_MIP_LEVEL 0u
#define DESTINATION_MIP_LEVEL 0u
#endif
#define ROW_LENGTH 40u
#define GUARD_BYTES 64u
#if G17P_STENCIL_BLIT
#define IMAGE_FORMAT VK_FORMAT_S8_UINT
#define IMAGE_ASPECT VK_IMAGE_ASPECT_STENCIL_BIT
#define PIXEL_BYTES 1u
#else
#define IMAGE_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define IMAGE_ASPECT VK_IMAGE_ASPECT_COLOR_BIT
#define PIXEL_BYTES 4u
#endif
#define IMAGE_BYTES (WIDTH * HEIGHT * OUTPUT_SLICES * PIXEL_BYTES)
#define UPLOAD_BYTES                                                          \
   (GUARD_BYTES + ROW_LENGTH * HEIGHT * OUTPUT_SLICES * PIXEL_BYTES +        \
    GUARD_BYTES)
#define DESTINATION_UPLOAD_BYTES (GUARD_BYTES + IMAGE_BYTES + GUARD_BYTES)
#define READBACK_BYTES (GUARD_BYTES + IMAGE_BYTES + GUARD_BYTES)

#define VK_FN(name) static PFN_##name name
VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkGetPhysicalDeviceFormatProperties);
VK_FN(vkGetPhysicalDeviceImageFormatProperties);
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
VK_FN(vkCreateCommandPool);
VK_FN(vkDestroyCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdPipelineBarrier);
VK_FN(vkCmdCopyBufferToImage);
VK_FN(vkCmdClearColorImage);
VK_FN(vkCmdClearDepthStencilImage);
VK_FN(vkCmdCopyImage);
VK_FN(vkCmdBlitImage);
VK_FN(vkCmdResolveImage);
VK_FN(vkCmdCopyImageToBuffer);
VK_FN(vkEndCommandBuffer);
VK_FN(vkCreateFence);
VK_FN(vkDestroyFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);

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

#define LOAD_GLOBAL(name)                                                     \
   do {                                                                       \
      name = (PFN_##name)vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);        \
      if (!name)                                                              \
         die("loader is missing " #name);                                    \
   } while (0)
#define LOAD_INSTANCE(instance, name)                                         \
   do {                                                                       \
      name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);              \
      if (!name)                                                              \
         die("instance is missing " #name);                                  \
   } while (0)
#define LOAD_DEVICE(device, name)                                             \
   do {                                                                       \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                  \
      if (!name)                                                              \
         die("device is missing " #name);                                    \
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
   die("required memory type is unavailable");
   return UINT32_MAX;
}

struct mapped_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint8_t *map;
   size_t size;
};

static struct mapped_buffer
create_buffer(VkDevice device,
              const VkPhysicalDeviceMemoryProperties *memory_properties,
              size_t size, VkBufferUsageFlags usage)
{
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   struct mapped_buffer out = {.size = size};
   check(vkCreateBuffer(device, &info, NULL, &out.buffer), "vkCreateBuffer");
   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, out.buffer, &requirements);
   const VkMemoryAllocateInfo allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = find_memory_type(
         memory_properties, requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   check(vkAllocateMemory(device, &allocation, NULL, &out.memory),
         "vkAllocateMemory(buffer)");
   check(vkBindBufferMemory(device, out.buffer, out.memory, 0),
         "vkBindBufferMemory");
   check(vkMapMemory(device, out.memory, 0, size, 0, (void **)&out.map),
         "vkMapMemory");
   return out;
}

struct image {
   VkImage image;
   VkDeviceMemory memory;
};

static struct image
create_image(VkDevice device,
             const VkPhysicalDeviceMemoryProperties *memory_properties,
             VkSampleCountFlagBits samples)
{
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = G17P_BLIT_IMAGE == 3 ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
      .imageType = G17P_COPY_IMAGE_3D ? VK_IMAGE_TYPE_3D
                                      : VK_IMAGE_TYPE_2D,
      .format = IMAGE_FORMAT,
      .extent = {IMAGE_BASE_WIDTH, IMAGE_BASE_HEIGHT, IMAGE_DEPTH},
      .mipLevels = IMAGE_MIP_LEVELS,
      .arrayLayers = IMAGE_LAYERS,
      .samples = samples,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   struct image out;
   check(vkCreateImage(device, &info, NULL, &out.image), "vkCreateImage");
   VkMemoryRequirements requirements;
   vkGetImageMemoryRequirements(device, out.image, &requirements);
   const VkMemoryAllocateInfo allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = find_memory_type(memory_properties,
                                          requirements.memoryTypeBits, 0),
   };
   check(vkAllocateMemory(device, &allocation, NULL, &out.memory),
         "vkAllocateMemory(image)");
   check(vkBindImageMemory(device, out.image, out.memory, 0),
         "vkBindImageMemory");
   return out;
}

static void
source_pixel(uint32_t x, uint32_t y, uint32_t z, uint8_t pixel[4])
{
#if G17P_STENCIL_BLIT
   pixel[0] = 0x5a;
#else
   pixel[0] = (uint8_t)(x * 7u + z * 31u + 3u);
   pixel[1] = (uint8_t)(y * 9u + z * 17u + 5u);
   pixel[2] = (uint8_t)((x * 5u) ^ (y * 3u) ^ (z * 29u));
   pixel[3] = 255;
#endif
}

static bool
verify(const struct mapped_buffer *upload,
       const uint8_t upload_expected[UPLOAD_BYTES],
       const struct mapped_buffer *destination_upload,
       const uint8_t destination_upload_expected[DESTINATION_UPLOAD_BYTES],
       const struct mapped_buffer *readback, bool clear_image, bool blit_image,
       bool copy_3d)
{
   if (memcmp(upload->map, upload_expected, UPLOAD_BYTES) != 0) {
      for (size_t i = 0; i < UPLOAD_BYTES; ++i) {
         if (upload->map[i] != upload_expected[i]) {
            fprintf(stderr, TEST_NAME ": upload byte %zu changed\n", i);
            break;
         }
      }
      return false;
   }
   if (memcmp(destination_upload->map, destination_upload_expected,
              DESTINATION_UPLOAD_BYTES) != 0) {
      for (size_t i = 0; i < DESTINATION_UPLOAD_BYTES; ++i) {
         if (destination_upload->map[i] != destination_upload_expected[i]) {
            fprintf(stderr, TEST_NAME ": destination upload byte %zu changed\n",
                    i);
            break;
         }
      }
      return false;
   }
   for (size_t i = 0; i < GUARD_BYTES; ++i) {
      if (readback->map[i] != 0xa5 ||
          readback->map[GUARD_BYTES + IMAGE_BYTES + i] != 0xa5) {
         fprintf(stderr, TEST_NAME ": readback guard byte %zu changed\n", i);
         return false;
      }
   }

   const uint8_t *pixels = readback->map + GUARD_BYTES;
   for (uint32_t z = 0; z < OUTPUT_SLICES; ++z) {
      for (uint32_t y = 0; y < HEIGHT; ++y) {
         for (uint32_t x = 0; x < WIDTH; ++x) {
            uint8_t expected[4] = {0};
#if !G17P_STENCIL_BLIT
            expected[3] = 255;
#endif
            if (clear_image) {
               expected[0] = 255;
               expected[2] = 255;
            } else if (x >= 11 && x < 24 && y >= 4 && y < 19) {
               if ((copy_3d && z >= 2 && z < 4) || G17P_RESOLVE_IMAGE) {
                  if (copy_3d)
                     source_pixel(5 + x - 11, 3 + y - 4, z - 1, expected);
                  else {
                     expected[0] = 255;
                     expected[2] = 255;
                  }
               } else if (!copy_3d && blit_image) {
                  source_pixel(23 - (x - 11), 18 - (y - 4),
                               G17P_BLIT_IMAGE == 3 ? SOURCE_BASE_LAYER + z : 0,
                               expected);
               } else if (!copy_3d) {
                  source_pixel(5 + x - 11, 3 + y - 4, 0, expected);
               }
            }
            const uint8_t *actual =
               pixels + ((z * HEIGHT + y) * WIDTH + x) * PIXEL_BYTES;
            if (memcmp(actual, expected, PIXEL_BYTES) != 0) {
#if G17P_STENCIL_BLIT
               fprintf(stderr,
                       TEST_NAME ": stencil (%u,%u) got %02x, expected %02x\n",
                       x, y, actual[0], expected[0]);
#else
               fprintf(stderr,
                       TEST_NAME ": pixel (%u,%u,%u) got %02x%02x%02x%02x, "
                                 "expected %02x%02x%02x%02x\n",
                       x, y, z, actual[0], actual[1], actual[2], actual[3],
                       expected[0], expected[1], expected[2], expected[3]);
#endif
               return false;
            }
         }
      }
   }
   return true;
}

int
main(int argc, char **argv)
{
#ifdef G17P_CLEAR_IMAGE
   const bool clear_image = true;
#else
   const bool clear_image = false;
#endif
   const bool blit_image = G17P_BLIT_IMAGE != 0 || G17P_STENCIL_BLIT;
   const bool linear_blit = G17P_BLIT_IMAGE == 2;
   const bool layered_mip_blit = G17P_BLIT_IMAGE == 3;
   const bool resolve_image = G17P_RESOLVE_IMAGE != 0;
   const bool copy_3d = G17P_COPY_IMAGE_3D != 0;
   const bool record_only = argc == 2 && !strcmp(argv[1], "--record-only");
   const bool submit_only = argc == 2 && !strcmp(argv[1], "--submit-only");
   const bool expect_submit_failure =
      argc == 2 && !strcmp(argv[1], "--expect-submit-failure");
   if (argc > 2 ||
       (argc == 2 && !record_only && !submit_only && !expect_submit_failure))
      die("usage: " TEST_NAME
          " [--record-only|--submit-only|--expect-submit-failure]");

   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());
   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("libvulkan has no vkGetInstanceProcAddr");
   LOAD_GLOBAL(vkCreateInstance);

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = TEST_NAME,
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");
   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFormatProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceImageFormatProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   if (physical_count == 0)
      die("no Vulkan physical devices found");
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
      die("T8140/G17P device was not found");

   VkFormatProperties format_properties;
   vkGetPhysicalDeviceFormatProperties(physical, IMAGE_FORMAT,
                                       &format_properties);
   const VkFormatFeatureFlags required =
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
      VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
      (blit_image ? VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                       VK_FORMAT_FEATURE_BLIT_DST_BIT
                  : 0) |
      (linear_blit ? VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT : 0);
   if ((format_properties.optimalTilingFeatures & required) != required)
      die("required optimal transfer/blit support is not advertised");
   if (resolve_image) {
      VkImageFormatProperties image_properties;
      check(vkGetPhysicalDeviceImageFormatProperties(
               physical, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D,
               VK_IMAGE_TILING_OPTIMAL,
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
               0, &image_properties),
            "vkGetPhysicalDeviceImageFormatProperties(4x resolve source)");
      if (!(image_properties.sampleCounts & VK_SAMPLE_COUNT_4_BIT))
         die("RGBA8 optimal transfer images do not advertise four samples");
   }
   if (copy_3d) {
      VkImageFormatProperties image_properties;
      check(vkGetPhysicalDeviceImageFormatProperties(
               physical, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_3D,
               VK_IMAGE_TILING_OPTIMAL,
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
               0, &image_properties),
            "vkGetPhysicalDeviceImageFormatProperties(3D copy)");
      if (image_properties.maxExtent.width < WIDTH ||
          image_properties.maxExtent.height < HEIGHT ||
          image_properties.maxExtent.depth < IMAGE_DEPTH)
         die("RGBA8 optimal transfer images do not advertise the test volume");
   }
   if (layered_mip_blit) {
      VkImageFormatProperties image_properties;
      check(vkGetPhysicalDeviceImageFormatProperties(
               physical, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D,
               VK_IMAGE_TILING_OPTIMAL,
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
               VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, &image_properties),
            "vkGetPhysicalDeviceImageFormatProperties(cube mip blit)");
      if (image_properties.maxExtent.width < IMAGE_BASE_WIDTH ||
          image_properties.maxExtent.height < IMAGE_BASE_HEIGHT ||
          image_properties.maxMipLevels < IMAGE_MIP_LEVELS ||
          image_properties.maxArrayLayers < IMAGE_LAYERS)
         die("RGBA8 cube-compatible transfer images do not advertise the "
             "test geometry");
   }

   uint32_t family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, NULL);
   if (family_count == 0)
      die("physical device has no queue families");
   VkQueueFamilyProperties *families = calloc(family_count, sizeof(*families));
   if (!families)
      die("out of host memory");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families);
   uint32_t family = UINT32_MAX;
   for (uint32_t i = 0; i < family_count; ++i) {
      if (families[i].queueCount &&
          (families[i].queueFlags & VK_QUEUE_TRANSFER_BIT)) {
         family = i;
         break;
      }
   }
   free(families);
   if (family == UINT32_MAX)
      die("no transfer-capable queue family");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   VkDevice device;
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
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkDestroyCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdPipelineBarrier);
   LOAD_DEVICE(device, vkCmdCopyBufferToImage);
   LOAD_DEVICE(device, vkCmdClearColorImage);
   LOAD_DEVICE(device, vkCmdClearDepthStencilImage);
   LOAD_DEVICE(device, vkCmdCopyImage);
   LOAD_DEVICE(device, vkCmdBlitImage);
   LOAD_DEVICE(device, vkCmdResolveImage);
   LOAD_DEVICE(device, vkCmdCopyImageToBuffer);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkDestroyFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);

   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
   struct mapped_buffer upload = create_buffer(
      device, &memory_properties, UPLOAD_BYTES,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
   struct mapped_buffer destination_upload = create_buffer(
      device, &memory_properties, DESTINATION_UPLOAD_BYTES,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
   struct mapped_buffer readback = create_buffer(
      device, &memory_properties, READBACK_BYTES,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   uint8_t upload_expected[UPLOAD_BYTES];
   memset(upload_expected, 0xa5, sizeof(upload_expected));
   for (uint32_t z = 0; z < OUTPUT_SLICES; ++z) {
      for (uint32_t y = 0; y < HEIGHT; ++y) {
         for (uint32_t x = 0; x < WIDTH; ++x) {
            source_pixel(
               x, y, layered_mip_blit ? SOURCE_BASE_LAYER + z : z,
               upload_expected + GUARD_BYTES +
                  ((z * HEIGHT + y) * ROW_LENGTH + x) * PIXEL_BYTES);
         }
      }
   }
   memcpy(upload.map, upload_expected, sizeof(upload_expected));
   uint8_t destination_upload_expected[DESTINATION_UPLOAD_BYTES];
   memset(destination_upload_expected, 0xa5,
          sizeof(destination_upload_expected));
   memset(destination_upload_expected + GUARD_BYTES, 0, IMAGE_BYTES);
#if !G17P_STENCIL_BLIT
   for (uint32_t i = 0; i < IMAGE_BYTES; i += 4)
      destination_upload_expected[GUARD_BYTES + i + 3] = 255;
#endif
   memcpy(destination_upload.map, destination_upload_expected,
          sizeof(destination_upload_expected));
   memset(readback.map, 0xa5, READBACK_BYTES);
   struct image source = create_image(
      device, &memory_properties,
      resolve_image ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_1_BIT);
   struct image destination = create_image(
      device, &memory_properties, VK_SAMPLE_COUNT_1_BIT);

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VkCommandPool pool;
   check(vkCreateCommandPool(device, &pool_info, NULL, &pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command;
   check(vkAllocateCommandBuffers(device, &command_info, &command),
         "vkAllocateCommandBuffers");
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");

   const VkImageSubresourceRange source_range = {
      .aspectMask = IMAGE_ASPECT,
      .baseMipLevel = SOURCE_MIP_LEVEL,
      .levelCount = 1,
      .baseArrayLayer = SOURCE_BASE_LAYER,
      .layerCount = TEST_LAYER_COUNT,
   };
   const VkImageSubresourceRange destination_range = {
      .aspectMask = IMAGE_ASPECT,
      .baseMipLevel = DESTINATION_MIP_LEVEL,
      .levelCount = 1,
      .baseArrayLayer = DESTINATION_BASE_LAYER,
      .layerCount = TEST_LAYER_COUNT,
   };
   VkImageMemoryBarrier initial[2];
   memset(initial, 0, sizeof(initial));
   for (uint32_t i = 0; i < 2; ++i) {
      initial[i] = (VkImageMemoryBarrier){
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = i ? destination.image : source.image,
         .subresourceRange = i ? destination_range : source_range,
      };
   }
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        2, initial);

   const VkBufferImageCopy upload_region = {
      .bufferOffset = GUARD_BYTES,
      .bufferRowLength = ROW_LENGTH,
      .bufferImageHeight = HEIGHT,
      .imageSubresource = {
         .aspectMask = IMAGE_ASPECT,
         .mipLevel = SOURCE_MIP_LEVEL,
         .baseArrayLayer = SOURCE_BASE_LAYER,
         .layerCount = TEST_LAYER_COUNT,
      },
      .imageExtent = {WIDTH, HEIGHT, IMAGE_DEPTH},
   };
   if (G17P_STENCIL_BLIT) {
      const VkImageSubresourceRange stencil_range = {
         .aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT,
         .levelCount = 1,
         .layerCount = 1,
      };
      const VkClearDepthStencilValue source_clear = {
         .depth = 1.0f,
         .stencil = 0x5a,
      };
      const VkClearDepthStencilValue destination_clear = {
         .depth = 1.0f,
         .stencil = 0,
      };
      vkCmdClearDepthStencilImage(command, source.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  &source_clear, 1, &stencil_range);
      vkCmdClearDepthStencilImage(command, destination.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  &destination_clear, 1, &stencil_range);
   } else if (resolve_image) {
      const VkClearColorValue magenta = {
         .float32 = {1.0f, 0.0f, 1.0f, 1.0f},
      };
      vkCmdClearColorImage(command, source.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &magenta, 1,
                           &source_range);
   } else {
      vkCmdCopyBufferToImage(command, upload.buffer, source.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                             &upload_region);
   }
   const VkBufferImageCopy destination_upload_region = {
      .bufferOffset = GUARD_BYTES,
      .imageSubresource = {
         .aspectMask = IMAGE_ASPECT,
         .mipLevel = DESTINATION_MIP_LEVEL,
         .baseArrayLayer = DESTINATION_BASE_LAYER,
         .layerCount = TEST_LAYER_COUNT,
      },
      .imageExtent = {WIDTH, HEIGHT, 1},
   };
   if (copy_3d) {
      const VkClearColorValue black = {
         .float32 = {0.0f, 0.0f, 0.0f, 1.0f},
      };
      vkCmdClearColorImage(command, destination.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1,
                           &destination_range);
   } else if (!G17P_STENCIL_BLIT) {
      vkCmdCopyBufferToImage(command, destination_upload.buffer,
                             destination.image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                             &destination_upload_region);
   }

   const VkImageMemoryBarrier uploads_ready[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = source.image,
         .subresourceRange = source_range,
      },
      {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = destination.image,
         .subresourceRange = destination_range,
      },
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        2, uploads_ready);
   const VkImageCopy image_copy = {
      .srcSubresource = {
         .aspectMask = IMAGE_ASPECT,
         .mipLevel = SOURCE_MIP_LEVEL,
         .baseArrayLayer = SOURCE_BASE_LAYER,
         .layerCount = TEST_LAYER_COUNT,
      },
#if G17P_COPY_IMAGE_3D
      .srcOffset = {5, 3, 1},
#else
      .srcOffset = {5, 3, 0},
#endif
      .dstSubresource = {
         .aspectMask = IMAGE_ASPECT,
         .mipLevel = DESTINATION_MIP_LEVEL,
         .baseArrayLayer = DESTINATION_BASE_LAYER,
         .layerCount = TEST_LAYER_COUNT,
      },
#if G17P_COPY_IMAGE_3D
      .dstOffset = {11, 4, 2},
      .extent = {13, 15, 2},
#else
      .dstOffset = {11, 4, 0},
      .extent = {13, 15, 1},
#endif
   };
   if (clear_image) {
      const VkClearColorValue magenta = {
         .float32 = {1.0f, 0.0f, 1.0f, 1.0f},
      };
      vkCmdClearColorImage(command, destination.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &magenta, 1,
                           &destination_range);
   } else if (resolve_image) {
      const VkImageResolve image_resolve = {
         .srcSubresource = {
            .aspectMask = IMAGE_ASPECT,
            .mipLevel = SOURCE_MIP_LEVEL,
            .baseArrayLayer = SOURCE_BASE_LAYER,
            .layerCount = TEST_LAYER_COUNT,
         },
         .srcOffset = {5, 3, 0},
         .dstSubresource = {
            .aspectMask = IMAGE_ASPECT,
            .mipLevel = DESTINATION_MIP_LEVEL,
            .baseArrayLayer = DESTINATION_BASE_LAYER,
            .layerCount = TEST_LAYER_COUNT,
         },
         .dstOffset = {11, 4, 0},
         .extent = {13, 15, 1},
      };
      vkCmdResolveImage(command, source.image,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        destination.image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                        &image_resolve);
   } else if (blit_image) {
      const VkImageBlit image_blit = {
         .srcSubresource = {
            .aspectMask = IMAGE_ASPECT,
            .mipLevel = SOURCE_MIP_LEVEL,
            .baseArrayLayer = SOURCE_BASE_LAYER,
            .layerCount = TEST_LAYER_COUNT,
         },
         .srcOffsets = {{24, 19, 0}, {11, 4, 1}},
         .dstSubresource = {
            .aspectMask = IMAGE_ASPECT,
            .mipLevel = DESTINATION_MIP_LEVEL,
            .baseArrayLayer = DESTINATION_BASE_LAYER,
            .layerCount = TEST_LAYER_COUNT,
         },
         .dstOffsets = {{11, 4, 0}, {24, 19, 1}},
      };
      vkCmdBlitImage(command, source.image,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination.image,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &image_blit,
                     linear_blit ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
   } else {
      vkCmdCopyImage(command, source.image,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination.image,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &image_copy);
   }

   const VkImageMemoryBarrier destination_ready = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = destination.image,
      .subresourceRange = destination_range,
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        1, &destination_ready);
   const VkBufferImageCopy readback_region = {
      .bufferOffset = GUARD_BYTES,
      .imageSubresource = {
         .aspectMask = IMAGE_ASPECT,
         .mipLevel = DESTINATION_MIP_LEVEL,
         .baseArrayLayer = DESTINATION_BASE_LAYER,
         .layerCount = TEST_LAYER_COUNT,
      },
      .imageExtent = {WIDTH, HEIGHT, IMAGE_DEPTH},
   };
   vkCmdCopyImageToBuffer(command, destination.image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback.buffer, 1, &readback_region);
   const VkBufferMemoryBarrier host_ready = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = readback.buffer,
      .offset = GUARD_BYTES,
      .size = IMAGE_BYTES,
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 0, NULL, 1,
                        &host_ready, 0, NULL);
   check(vkEndCommandBuffer(command), "vkEndCommandBuffer");

   VkFence fence = VK_NULL_HANDLE;
   if (record_only) {
      if (clear_image)
         printf(TEST_NAME ": RECORD PASS — RGBA8 optimal image clear and "
                          "guarded readback recorded\n");
      else if (G17P_STENCIL_BLIT)
         printf(TEST_NAME ": RECORD PASS — reversed 13x15 S8 image blit "
                          "and guarded readback recorded\n");
      else if (blit_image)
         printf(TEST_NAME ": RECORD PASS — reversed 13x15 %s RGBA8 %s"
                          "blit and guarded readback recorded\n",
                linear_blit ? "linear" : "nearest",
                layered_mip_blit ? "six-layer cube mip0-to-mip1 "
                                 : "image ");
      else if (resolve_image)
         printf(TEST_NAME ": RECORD PASS — 13x15 offset RGBA8 4x-to-1x "
                          "image resolve and guarded readback recorded\n");
      else if (copy_3d)
         printf(TEST_NAME ": RECORD PASS — two 3D slices copied between "
                          "nonzero source/destination Z offsets\n");
      else
         printf(TEST_NAME ": RECORD PASS — padded upload, nonzero image "
                          "subregion copy and guarded readback recorded\n");
      goto cleanup;
   }

   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command,
   };
   const VkResult submitted = vkQueueSubmit(queue, 1, &submit, fence);
   if (expect_submit_failure) {
      if (submitted != VK_ERROR_DEVICE_LOST) {
         fprintf(stderr,
                 TEST_NAME ": vkQueueSubmit returned %d, expected %d\n",
                 submitted, VK_ERROR_DEVICE_LOST);
         return EXIT_FAILURE;
      }
      printf(TEST_NAME ": EXPECTED SUBMIT FAILURE PASS\n");
      goto cleanup;
   }
   check(submitted, "vkQueueSubmit");
   const VkResult waited =
      vkWaitForFences(device, 1, &fence, true, UINT64_C(3000000000));
   if (waited == VK_TIMEOUT)
      die("vkWaitForFences timed out after three seconds");
   check(waited, "vkWaitForFences");
   if (submit_only) {
      printf(TEST_NAME ": SUBMIT PASS — command stream retired\n");
      goto cleanup;
   }
   if (!verify(&upload, upload_expected, &destination_upload,
               destination_upload_expected, &readback, clear_image,
               blit_image, copy_3d))
      return EXIT_FAILURE;
   sleep(3);
   if (!verify(&upload, upload_expected, &destination_upload,
               destination_upload_expected, &readback, clear_image,
               blit_image, copy_3d))
      return EXIT_FAILURE;

   uint64_t checksum = UINT64_C(1469598103934665603);
   for (size_t i = 0; i < IMAGE_BYTES; ++i) {
      checksum ^= readback.map[GUARD_BYTES + i];
      checksum *= UINT64_C(1099511628211);
   }
   if (clear_image)
      printf(TEST_NAME ": PASS — exact magenta RGBA8 clear immediately/3s; "
                       "both uploads and 384 guard bytes unchanged, checksum "
                       "%016" PRIx64 "\n",
             checksum);
   else if (G17P_STENCIL_BLIT)
      printf(TEST_NAME ": PASS — reversed 13x15 S8 blit into "
                       "destination(11,4) exact immediately/3s; both uploads "
                       "and 384 guard bytes unchanged, checksum %016" PRIx64
                       "\n", checksum);
   else if (blit_image)
      printf(TEST_NAME ": PASS — reversed 13x15 %s %sblit into "
                       "destination(11,4) exact immediately/3s; both uploads "
                       "and 384 guard bytes unchanged, checksum %016" PRIx64
                       "\n",
             linear_blit ? "linear" : "nearest",
             layered_mip_blit ? "six-layer cube mip0-to-mip1 " : "",
             checksum);
   else if (resolve_image)
      printf(TEST_NAME ": PASS — exact magenta 13x15 RGBA8 4x-to-1x resolve "
                       "into destination(11,4) immediately/3s; both uploads "
                       "and 384 guard bytes unchanged, checksum %016" PRIx64
                       "\n", checksum);
   else if (copy_3d)
      printf(TEST_NAME ": PASS — two exact 13x15 RGBA8 slices copied from "
                       "distinct source z=1,2 to destination z=2,3 "
                       "immediately/3s; "
                       "384 guard bytes unchanged, checksum %016" PRIx64
                       "\n", checksum);
   else
      printf(TEST_NAME ": PASS — 13x15 source(5,3) to destination(11,4) "
                       "exact immediately/3s; both uploads and 384 guard "
                       "bytes unchanged, checksum %016" PRIx64 "\n",
             checksum);

cleanup:
   if (fence != VK_NULL_HANDLE)
      vkDestroyFence(device, fence, NULL);
   vkDestroyCommandPool(device, pool, NULL);
   vkDestroyImage(device, destination.image, NULL);
   vkFreeMemory(device, destination.memory, NULL);
   vkDestroyImage(device, source.image, NULL);
   vkFreeMemory(device, source.memory, NULL);
   vkUnmapMemory(device, readback.memory);
   vkDestroyBuffer(device, readback.buffer, NULL);
   vkFreeMemory(device, readback.memory, NULL);
   vkUnmapMemory(device, destination_upload.memory);
   vkDestroyBuffer(device, destination_upload.buffer, NULL);
   vkFreeMemory(device, destination_upload.memory, NULL);
   vkUnmapMemory(device, upload.memory);
   vkDestroyBuffer(device, upload.buffer, NULL);
   vkFreeMemory(device, upload.memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return EXIT_SUCCESS;
}
