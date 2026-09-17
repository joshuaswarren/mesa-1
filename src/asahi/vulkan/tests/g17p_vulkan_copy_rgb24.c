/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */

/* Ground truth for vkCmdCopyImage on a three-component format, whose physical
 * G17P storage is four components.  Uploads a known byte pattern, reads the
 * source back, copies image to image, reads the destination back, and prints
 * both.  The source readback separates an upload/readback defect from a copy
 * defect: if the source comes back intact and the destination does not, the
 * image-to-image copy is the only step left.
 *
 *   g17p-vulkan-copy-rgb24 <VkFormat> <bytes per texel>
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_NAME "g17p-vulkan-copy-rgb24"
#define T8140_DEVICE_ID 0x8140
static uint32_t g_width = 8u;
static uint32_t g_height = 4u;
#define WIDTH g_width
#define HEIGHT g_height

static VkFormat g_format = VK_FORMAT_B8G8R8_UNORM;
static uint32_t g_texel_B = 3u;

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
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkGetDeviceQueue);
VK_FN(vkCreateBuffer);
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkCreateImage);
VK_FN(vkGetImageMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkBindBufferMemory);
VK_FN(vkBindImageMemory);
VK_FN(vkMapMemory);
VK_FN(vkCreateCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdPipelineBarrier);
VK_FN(vkCmdCopyBufferToImage);
VK_FN(vkCmdCopyImage);
VK_FN(vkCmdCopyImageToBuffer);
VK_FN(vkEndCommandBuffer);
VK_FN(vkCreateFence);
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
find_memory_type(const VkPhysicalDeviceMemoryProperties *props, uint32_t allowed,
                 VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < props->memoryTypeCount; ++i) {
      if ((allowed & (1u << i)) &&
          (props->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }
   die("required memory type is unavailable");
   return UINT32_MAX;
}

struct buf {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint8_t *map;
};

static struct buf
make_buffer(VkDevice device, const VkPhysicalDeviceMemoryProperties *mem,
            VkDeviceSize size, VkBufferUsageFlags usage)
{
   struct buf b = {0};
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   check(vkCreateBuffer(device, &info, NULL, &b.buffer), "vkCreateBuffer");

   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(device, b.buffer, &req);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex =
         find_memory_type(mem, req.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   check(vkAllocateMemory(device, &alloc, NULL, &b.memory), "vkAllocateMemory");
   check(vkBindBufferMemory(device, b.buffer, b.memory, 0),
         "vkBindBufferMemory");
   check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, (void **)&b.map),
         "vkMapMemory");
   memset(b.map, 0xcd, size);
   return b;
}

static VkImage
make_image(VkDevice device, const VkPhysicalDeviceMemoryProperties *mem)
{
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = g_format,
      .extent = {WIDTH, HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   check(vkCreateImage(device, &info, NULL, &image), "vkCreateImage");

   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(device, image, &req);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = find_memory_type(mem, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory memory;
   check(vkAllocateMemory(device, &alloc, NULL, &memory), "vkAllocateMemory");
   check(vkBindImageMemory(device, image, memory, 0), "vkBindImageMemory");
   return image;
}

static void
barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to,
        VkAccessFlags src_access, VkAccessFlags dst_access)
{
   const VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = src_access,
      .dstAccessMask = dst_access,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                        1, &b);
}

static void
dump(const char *what, const uint8_t *p, uint32_t bytes)
{
   printf("%s:", what);
   for (uint32_t i = 0; i < bytes; ++i) {
      if (i % g_texel_B == 0)
         printf(" ");
      printf("%02x", p[i]);
   }
   printf("\n");
}

int
main(int argc, char **argv)
{
   if (argc >= 2)
      g_format = (VkFormat)strtoul(argv[1], NULL, 0);
   if (argc >= 3)
      g_texel_B = (uint32_t)strtoul(argv[2], NULL, 0);
   if (argc >= 4)
      g_width = (uint32_t)strtoul(argv[3], NULL, 0);
   if (argc >= 5)
      g_height = (uint32_t)strtoul(argv[4], NULL, 0);
   if (argc > 5 || g_texel_B == 0 || g_texel_B > 16 || !g_width || !g_height) {
      fprintf(stderr,
              "usage: " TEST_NAME " <VkFormat> <bytes per texel> [w] [h]\n");
      return EXIT_FAILURE;
   }
   const uint32_t image_bytes = WIDTH * HEIGHT * g_texel_B;

   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      loader = dlopen("libvulkan_asahi.so", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die("no Vulkan loader");
   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("loader has no vkGetInstanceProcAddr");
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
   check(vkCreateInstance(&instance_info, NULL, &instance), "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFormatProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t count = 0;
   check(vkEnumeratePhysicalDevices(instance, &count, NULL),
         "vkEnumeratePhysicalDevices");
   if (count == 0)
      die("no physical devices");
   VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
   check(vkEnumeratePhysicalDevices(instance, &count, devices),
         "vkEnumeratePhysicalDevices");

   VkPhysicalDevice physical = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < count; ++i) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(devices[i], &props);
      if (props.deviceID == T8140_DEVICE_ID) {
         physical = devices[i];
         break;
      }
   }
   if (physical == VK_NULL_HANDLE)
      physical = devices[0];

   VkFormatProperties fprops;
   vkGetPhysicalDeviceFormatProperties(physical, g_format, &fprops);
   printf(TEST_NAME ": format=%u texel_B=%u optimal_features=0x%08x\n",
          (unsigned)g_format, g_texel_B, fprops.optimalTilingFeatures);
   const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                     VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   if ((fprops.optimalTilingFeatures & need) != need)
      die("format cannot be transferred in optimal tiling");

   VkPhysicalDeviceMemoryProperties mem;
   vkGetPhysicalDeviceMemoryProperties(physical, &mem);

   uint32_t family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, NULL);
   VkQueueFamilyProperties *families = calloc(family_count, sizeof(*families));
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families);
   uint32_t family = UINT32_MAX;
   for (uint32_t i = 0; i < family_count; ++i) {
      if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
         family = i;
         break;
      }
   }
   if (family == UINT32_MAX)
      die("no graphics queue family");

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
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkCreateImage);
   LOAD_DEVICE(device, vkGetImageMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkBindImageMemory);
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdPipelineBarrier);
   LOAD_DEVICE(device, vkCmdCopyBufferToImage);
   LOAD_DEVICE(device, vkCmdCopyImage);
   LOAD_DEVICE(device, vkCmdCopyImageToBuffer);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);

   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);

   struct buf upload = make_buffer(device, &mem, image_bytes,
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
   struct buf src_back = make_buffer(device, &mem, image_bytes,
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   struct buf dst_back = make_buffer(device, &mem, image_bytes,
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   for (uint32_t i = 0; i < image_bytes; ++i)
      upload.map[i] = (uint8_t)(i * 7u + 11u);

   VkImage src = make_image(device, &mem);
   VkImage dst = make_image(device, &mem);

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VkCommandPool pool;
   check(vkCreateCommandPool(device, &pool_info, NULL, &pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo cb_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   check(vkAllocateCommandBuffers(device, &cb_info, &cb),
         "vkAllocateCommandBuffers");

   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(cb, &begin), "vkBeginCommandBuffer");

   barrier(cb, src, VK_IMAGE_LAYOUT_UNDEFINED,
           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
           VK_ACCESS_TRANSFER_WRITE_BIT);
   barrier(cb, dst, VK_IMAGE_LAYOUT_UNDEFINED,
           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
           VK_ACCESS_TRANSFER_WRITE_BIT);

   const VkBufferImageCopy region = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {WIDTH, HEIGHT, 1},
   };
   vkCmdCopyBufferToImage(cb, upload.buffer, src,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

   barrier(cb, src, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
           VK_ACCESS_TRANSFER_READ_BIT);

   vkCmdCopyImageToBuffer(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          src_back.buffer, 1, &region);

   const VkImageCopy copy = {
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .extent = {WIDTH, HEIGHT, 1},
   };
   vkCmdCopyImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

   barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
           VK_ACCESS_TRANSFER_READ_BIT);

   vkCmdCopyImageToBuffer(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          dst_back.buffer, 1, &region);

   check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");

   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence fence;
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cb,
   };
   check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
   check(vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ull),
         "vkWaitForFences");

   const uint32_t show = image_bytes < 192u ? image_bytes : 192u;
   dump("uploaded ", upload.map, show);
   dump("src back ", src_back.map, show);
   dump("dst back ", dst_back.map, show);

   uint32_t good = 0;
   while (good < WIDTH * HEIGHT &&
          memcmp(dst_back.map + good * g_texel_B,
                 upload.map + good * g_texel_B, g_texel_B) == 0)
      good++;
   printf(TEST_NAME ": %ux%u %u-byte texels, %u of %u texels copied "
                    "correctly before the first wrong one\n",
          WIDTH, HEIGHT, g_texel_B, good, WIDTH * HEIGHT);

   const bool src_ok = memcmp(src_back.map, upload.map, image_bytes) == 0;
   const bool dst_ok = memcmp(dst_back.map, upload.map, image_bytes) == 0;
   printf(TEST_NAME ": upload+readback %s, image-to-image copy %s\n",
          src_ok ? "INTACT" : "WRONG", dst_ok ? "INTACT" : "WRONG");
   if (!src_ok || !dst_ok) {
      for (uint32_t i = 0; i < image_bytes; ++i) {
         if (dst_back.map[i] != upload.map[i]) {
            printf(TEST_NAME ": first copy mismatch at byte %u (texel %u "
                             "component %u): got %02x want %02x\n",
                   i, i / g_texel_B, i % g_texel_B, dst_back.map[i],
                   upload.map[i]);
            break;
         }
      }
      return EXIT_FAILURE;
   }
   printf(TEST_NAME ": PASS\n");
   return EXIT_SUCCESS;
}
