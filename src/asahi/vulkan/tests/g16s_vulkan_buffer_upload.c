/* SPDX-License-Identifier: MIT
 * Buffer uploads to single-slice graphics attachment views. Nonzero array/Z
 * origins, mip levels, row/image padding and unaligned rectangle edges must
 * preserve the source bytes and every untouched destination texel.
 * --record-only uses drm-shim and never submits; it cannot check GPU results.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(call) do { VkResult r = (call); if (r != VK_SUCCESS) { \
   fprintf(stderr, "%s: %d\n", #call, r); exit(2); } } while (0)

static VkDevice dev;
static VkPhysicalDevice physical;
static VkQueue queue;
static VkCommandPool pool;
static bool record_only;

static VkDeviceMemory
memory(VkMemoryRequirements req, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties props;
   vkGetPhysicalDeviceMemoryProperties(physical, &props);
   unsigned i;
   for (i = 0; i < props.memoryTypeCount; ++i)
      if ((req.memoryTypeBits & (1u << i)) &&
          (props.memoryTypes[i].propertyFlags & flags) == flags) break;
   if (i == props.memoryTypeCount) exit(3);
   VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size, .memoryTypeIndex = i };
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &ai, NULL, &mem));
   return mem;
}

static VkBuffer
buffer(VkDeviceSize size, VkDeviceMemory *mem, void **map)
{
   VkBufferCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT };
   VkBuffer buf;
   CHECK(vkCreateBuffer(dev, &ci, NULL, &buf));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, buf, &req);
   *mem = memory(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   CHECK(vkBindBufferMemory(dev, buf, *mem, 0));
   CHECK(vkMapMemory(dev, *mem, 0, VK_WHOLE_SIZE, 0, map));
   return buf;
}

static unsigned
check_case(VkFormat format, unsigned bytes, unsigned shape, unsigned level)
{
   const bool volume = shape == 3, one_d = shape == 4;
   unsigned width = 129, height = one_d ? 1 : 127;
   unsigned depth = volume ? 4 : 1;
   unsigned layers = shape == 1 || one_d ? 3 : shape == 2 ? 6 : 1;
   if (shape == 2) height = width;
   VkImageCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = shape == 2 ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
      .imageType = volume ? VK_IMAGE_TYPE_3D : one_d ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D,
      .format = format, .extent = {width, height, depth}, .mipLevels = 2,
      .arrayLayers = layers, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT };
   VkImageFormatProperties props;
   VkResult supported = vkGetPhysicalDeviceImageFormatProperties(physical, format,
      ci.imageType, ci.tiling, ci.usage, ci.flags, &props);
   if (supported == VK_ERROR_FORMAT_NOT_SUPPORTED) {
      printf("SKIP format=%u shape=%u level=%u unsupported\n", format, shape, level);
      return 0;
   }
   CHECK(supported);
   VkImage image;
   CHECK(vkCreateImage(dev, &ci, NULL, &image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, image, &req);
   VkDeviceMemory imem = memory(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   CHECK(vkBindImageMemory(dev, image, imem, 0));
   width >>= level; height = one_d ? 1 : height >> level;
   depth = volume ? depth >> level : layers;
   size_t full_size = (size_t)width * height * depth * bytes;
   const unsigned px = 1, py = one_d ? 0 : 3;
   const unsigned pw = width - 3, ph = one_d ? 1 : height - 5;
   const unsigned first = depth > 1 ? 1 : 0, count = depth - first;
   const unsigned row = pw + 2, rows = ph + 3;
   const size_t patch_offset = (full_size + 63) & ~(size_t)63;
   const size_t patch_size = (size_t)row * rows * count * bytes;
   VkDeviceMemory smem, rmem;
   uint8_t *src, *readback;
   VkBuffer source = buffer(patch_offset + patch_size, &smem, (void **)&src);
   VkBuffer result = buffer(full_size + 64, &rmem, (void **)&readback);
   uint8_t *expected = malloc(full_size);
   if (!expected) exit(3);
   for (size_t i = 0; i < patch_offset + patch_size; ++i)
      src[i] = (uint8_t)((i * 37 + (i >> 8) * 19 + shape * 23 + level * 11) ^ 0xa5);
   memcpy(expected, src, full_size);
   memset(readback, 0xcd, full_size + 64);
   for (unsigned z = 0; z < count; ++z)
      for (unsigned y = 0; y < ph; ++y)
         memcpy(expected + (((size_t)(z + first) * height + y + py) * width + px) * bytes,
                src + patch_offset + ((size_t)z * rows + y) * row * bytes, pw * bytes);

   VkCommandBufferAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &ai, &cmd));
   VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
   CHECK(vkBeginCommandBuffer(cmd, &begin));
   VkImageMemoryBarrier ib = { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, layers} };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      0, 0, NULL, 0, NULL, 1, &ib);
   VkBufferImageCopy all = { .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, volume ? 1 : layers},
      .imageExtent = {width, height, volume ? depth : 1} };
   vkCmdCopyBufferToImage(cmd, source, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &all);
   ib.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   ib.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      0, 0, NULL, 0, NULL, 1, &ib);
   VkBufferImageCopy patch = { .bufferOffset = patch_offset,
      .bufferRowLength = row, .bufferImageHeight = one_d ? 0 : rows,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, volume ? 0 : first, volume ? 1 : count},
      .imageOffset = {px, py, volume ? first : 0}, .imageExtent = {pw, ph, volume ? count : 1} };
   /* For 1D arrays Vulkan's source layer stride is bufferRowLength, since
    * there is only one row per image. */
   if (one_d) {
      patch.bufferImageHeight = 0;
      for (unsigned z = 0; z < count; ++z)
         memcpy(expected + ((size_t)(z + first) * width + px) * bytes,
                src + patch_offset + (size_t)z * row * bytes, pw * bytes);
   }
   vkCmdCopyBufferToImage(cmd, source, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &patch);
   ib.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   ib.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      0, 0, NULL, 0, NULL, 1, &ib);
   vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, result, 1, &all);
   VkBufferMemoryBarrier host = { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = result, .size = VK_WHOLE_SIZE };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
      0, 0, NULL, 1, &host, 0, NULL);
   CHECK(vkEndCommandBuffer(cmd));
   unsigned failure = 0;
   if (!record_only) {
      VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
      VkFence fence;
      CHECK(vkCreateFence(dev, &fci, NULL, &fence));
      VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1, .pCommandBuffers = &cmd };
      CHECK(vkQueueSubmit(queue, 1, &submit, fence));
      CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 30ull * 1000000000));
      vkDestroyFence(dev, fence, NULL);
      for (size_t i = 0; i < full_size + 64; ++i) {
         uint8_t want = i < full_size ? expected[i] : 0xcd;
         if (readback[i] != want) {
            fprintf(stderr, "MISMATCH format=%u shape=%u level=%u byte=%zu got=%02x expected=%02x\n",
               format, shape, level, i, readback[i], want);
            failure = 1;
            break;
         }
      }
   }
   printf("%s format=%u bytes=%u shape=%u level=%u\n",
      record_only ? "RECORDED" : failure ? "FAIL" : "PASS", format, bytes, shape, level);
   fflush(stdout);
   free(expected);
   vkFreeCommandBuffers(dev, pool, 1, &cmd);
   vkDestroyImage(dev, image, NULL); vkFreeMemory(dev, imem, NULL);
   vkUnmapMemory(dev, smem); vkUnmapMemory(dev, rmem);
   vkDestroyBuffer(dev, source, NULL); vkDestroyBuffer(dev, result, NULL);
   vkFreeMemory(dev, smem, NULL); vkFreeMemory(dev, rmem, NULL);
   return failure;
}

int main(int argc, char **argv)
{
   if (argc > 2 || (argc == 2 && strcmp(argv[1], "--record-only"))) return 2;
   record_only = argc == 2;
   VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 };
   VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app };
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(physical, &props);
   fprintf(stderr, "Device: %s; mode: %s\n", props.deviceName, record_only ? "record only, no GPU submit" : "GPU readback");
   float priority = 1;
   VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &priority };
   VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
   CHECK(vkCreateDevice(physical, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue);
   VkCommandPoolCreateInfo pci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0 };
   CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
   const struct { VkFormat format; unsigned bytes; } formats[] = {
      {VK_FORMAT_R8G8B8A8_UNORM, 4}, {VK_FORMAT_R8G8B8A8_SRGB, 4},
      {VK_FORMAT_B8G8R8A8_UNORM, 4}, {VK_FORMAT_R8G8B8A8_UINT, 4},
      {VK_FORMAT_R32_UINT, 4}, {VK_FORMAT_R32_SFLOAT, 4},
      {VK_FORMAT_R16G16_SFLOAT, 4}, {VK_FORMAT_R8_UINT, 1},
      {VK_FORMAT_R16_UINT, 2}, {VK_FORMAT_R32G32_UINT, 8},
      {VK_FORMAT_R32G32B32A32_UINT, 16},
   };
   unsigned failures = 0;
   for (unsigned f = 0; f < sizeof(formats)/sizeof(formats[0]); ++f)
      for (unsigned shape = 0; shape < 5; ++shape)
         for (unsigned level = 0; level < 2; ++level)
            failures += check_case(formats[f].format, formats[f].bytes, shape, level);
   vkDestroyCommandPool(dev, pool, NULL); vkDestroyDevice(dev, NULL); vkDestroyInstance(instance, NULL);
   return failures ? 1 : 0;
}
