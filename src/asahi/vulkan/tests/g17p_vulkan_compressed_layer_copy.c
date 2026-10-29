/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * vkCmdCopyImage from one layer of a multi-layer block-compressed image
 * into a losslessly compressed colour image: R11 EAC cube faces and 3D
 * slices into an RG32F mip chain, the shape of the GLES 3.1
 * copy_image.mixed.viewclass_64_bits_mixed.*_eac_*.cubemap_to_* cases. On
 * G17P a compressed destination takes the graphics copy path, whose source
 * view must address the copied layer.
 *
 * Every source block holds a distinct pattern (level, layer, block), and the
 * copy is bitwise, so each destination texel must hold exactly the 8 bytes
 * of its source block.
 *
 * usage: g17p-vulkan-compressed-layer-copy [--shim] [--expect-compressed]
 *   --shim               no-op DRM shim: nothing executes, skip readback
 *   --expect-compressed  require the driver's "compression enabled" message
 *                        for the destination (needs ASAHI_MESA_DEBUG=perf)
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #call, r_);                               \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

#define SRC_SIZE   132
#define CUBE_LEVELS 8
#define VOL_DEPTH  4
#define VOL_LEVELS 3
#define DST_W      129
#define DST_H      127
#define DST_LEVELS 8

static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;

static FILE *driver_log;
static int saved_stderr = -1;

static void
restore_stderr(void)
{
   if (saved_stderr < 0)
      return;
   fflush(stderr);
   dup2(saved_stderr, STDERR_FILENO);
   close(saved_stderr);
   saved_stderr = -1;
}

/* CHECK() exits during setup too. Keep the loader error and the failing
 * Vulkan call visible even when the normal log checks are never reached. */
static void
replay_driver_log(void)
{
   restore_stderr();
   if (!driver_log)
      return;
   rewind(driver_log);
   char line[1024];
   while (fgets(line, sizeof(line), driver_log))
      fputs(line, stderr);
   fclose(driver_log);
   driver_log = NULL;
}

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   }
   fprintf(stderr, "no memory type for %#x\n", flags);
   exit(2);
}

static VkImage
image_create(VkImageType type, VkFormat format, VkImageCreateFlags flags,
             VkExtent3D extent, uint32_t levels, uint32_t layers,
             VkImageUsageFlags usage)
{
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = flags,
      .imageType = type,
      .format = format,
      .extent = extent,
      .mipLevels = levels,
      .arrayLayers = layers,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
   };
   VkImage image;
   CHECK(vkCreateImage(dev, &info, NULL, &image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, image, &req);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex =
         memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &memory));
   CHECK(vkBindImageMemory(dev, image, memory, 0));
   return image;
}

static VkBuffer
buffer_create(VkDeviceSize size, VkBufferUsageFlags usage, uint32_t **map)
{
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
   };
   VkBuffer buffer;
   CHECK(vkCreateBuffer(dev, &info, NULL, &buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, buffer, &req);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex =
         memory_type(req.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &memory));
   CHECK(vkBindBufferMemory(dev, buffer, memory, 0));
   CHECK(vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, (void **)map));
   return buffer;
}

static VkCommandBuffer
begin(void)
{
   const VkCommandBufferAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &info, &cmd));
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &begin_info));
   return cmd;
}

static void
barrier(VkCommandBuffer cmd)
{
   const VkMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, NULL,
                        0, NULL);
}

static void
to_general(VkCommandBuffer cmd, VkImage image)
{
   const VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                           VK_REMAINING_MIP_LEVELS, 0,
                           VK_REMAINING_ARRAY_LAYERS},
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                        1, &b);
}

static void
submit(VkCommandBuffer cmd)
{
   CHECK(vkEndCommandBuffer(cmd));
   const VkSubmitInfo info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   vkFreeCommandBuffers(dev, pool, 1, &cmd);
}

static uint32_t
minify(uint32_t size, uint32_t level)
{
   return size >> level ? size >> level : 1;
}

static uint32_t
min_u32(uint32_t a, uint32_t b)
{
   return a < b ? a : b;
}

static uint32_t
blocks(uint32_t size, uint32_t level)
{
   return (minify(size, level) + 3) / 4;
}

/* Two normal floats per block, so no float path can alter the bits. */
static void
pattern(bool volume, uint32_t level, uint32_t layer, uint32_t bx, uint32_t by,
        uint32_t out[2])
{
   out[0] = 0x3f0000a5u | level << 20 | layer << 16 | by << 8;
   out[1] = 0x40000000u | volume << 16 | bx << 8 | level << 4 | layer;
}

struct source {
   VkImage image;
   bool volume;
   uint32_t levels, layers; /* layers: array layers, or level-0 depth */
};

static uint32_t
source_layers(const struct source *s, uint32_t level)
{
   return s->volume ? minify(s->layers, level) : s->layers;
}

static void
fill(const struct source *s)
{
   VkDeviceSize size = 0;
   for (uint32_t l = 0; l < s->levels; ++l)
      size += 8ull * blocks(SRC_SIZE, l) * blocks(SRC_SIZE, l) *
              source_layers(s, l);

   uint32_t *map;
   VkBuffer staging =
      buffer_create(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &map);
   VkCommandBuffer cmd = begin();
   to_general(cmd, s->image);
   VkDeviceSize offset = 0;
   for (uint32_t l = 0; l < s->levels; ++l) {
      const uint32_t bw = blocks(SRC_SIZE, l), bh = blocks(SRC_SIZE, l);
      for (uint32_t layer = 0; layer < source_layers(s, l); ++layer) {
         for (uint32_t by = 0; by < bh; ++by) {
            for (uint32_t bx = 0; bx < bw; ++bx)
               pattern(s->volume, l, layer, bx, by,
                       &map[(offset + 8ull * (by * bw + bx)) / 4]);
         }
         const VkBufferImageCopy region = {
            .bufferOffset = offset,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l,
                                 s->volume ? 0 : layer, 1},
            .imageOffset = {0, 0, s->volume ? (int32_t)layer : 0},
            .imageExtent = {minify(SRC_SIZE, l), minify(SRC_SIZE, l), 1},
         };
         vkCmdCopyBufferToImage(cmd, staging, s->image,
                                VK_IMAGE_LAYOUT_GENERAL, 1, &region);
         offset += 8ull * bw * bh;
      }
   }
   barrier(cmd);
   submit(cmd);
   vkDestroyBuffer(dev, staging, NULL);
}

struct copy {
   uint32_t level, dst_level;
   uint32_t sbx, sby, dx, dy, bw, bh; /* blocks; destination texels */
};

/* The copies of the GLES shape: level 0, level 1, a small level and the
 * last block of the chain into the last destination level. */
static const struct copy cube_copies[] = {
   {0, 0, 1, 2, 0, 0, 31, 30},
   {1, 1, 0, 0, 1, 1, 15, 15},
   {3, 3, 0, 0, 0, 0, 4, 4},
   {5, DST_LEVELS - 1, 0, 0, 0, 0, 1, 1},
};

static const struct copy volume_copies[] = {
   {0, 0, 1, 2, 0, 0, 31, 30},
   {1, 1, 0, 0, 1, 1, 15, 15},
};

static unsigned failures;

static void
copy_and_check(const struct source *s, VkImage dst, const struct copy *c,
               uint32_t layer, bool readback)
{
   uint32_t *map;
   VkBuffer out = buffer_create(8ull * c->bw * c->bh,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT, &map);
   memset(map, 0, 8ull * c->bw * c->bh);

   VkCommandBuffer cmd = begin();
   const VkImageCopy region = {
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, c->level,
                         s->volume ? 0 : layer, 1},
      .srcOffset = {4 * c->sbx, 4 * c->sby, s->volume ? (int32_t)layer : 0},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, c->dst_level, 0, 1},
      .dstOffset = {c->dx, c->dy, 0},
      .extent = {min_u32(4 * c->bw, minify(SRC_SIZE, c->level) - 4 * c->sbx),
                 min_u32(4 * c->bh, minify(SRC_SIZE, c->level) - 4 * c->sby),
                 1},
   };
   vkCmdCopyImage(cmd, s->image, VK_IMAGE_LAYOUT_GENERAL, dst,
                  VK_IMAGE_LAYOUT_GENERAL, 1, &region);
   barrier(cmd);
   const VkBufferImageCopy back = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, c->dst_level, 0, 1},
      .imageOffset = {c->dx, c->dy, 0},
      .imageExtent = {c->bw, c->bh, 1},
   };
   vkCmdCopyImageToBuffer(cmd, dst, VK_IMAGE_LAYOUT_GENERAL, out, 1, &back);
   barrier(cmd);
   submit(cmd);

   if (readback) {
      unsigned bad = 0;
      for (uint32_t y = 0; y < c->bh; ++y) {
         for (uint32_t x = 0; x < c->bw; ++x) {
            uint32_t want[2];
            pattern(s->volume, c->level, layer, c->sbx + x, c->sby + y, want);
            const uint32_t *got = &map[2 * (y * c->bw + x)];
            if (got[0] != want[0] || got[1] != want[1]) {
               if (!bad)
                  fprintf(stderr,
                          "%s level %u layer %u -> level %u: block (%u,%u) "
                          "%08x %08x, want %08x %08x\n",
                          s->volume ? "3D" : "cube", c->level, layer,
                          c->dst_level, c->sbx + x, c->sby + y, got[0],
                          got[1], want[0], want[1]);
               bad++;
            }
         }
      }
      if (bad) {
         fprintf(stderr, "  %u of %u blocks wrong\n", bad, c->bw * c->bh);
         failures++;
      }
   }
   vkDestroyBuffer(dev, out, NULL);
}

int
main(int argc, char **argv)
{
   bool shim = false, expect_compressed = false;
   for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--shim"))
         shim = true;
      else if (!strcmp(argv[i], "--expect-compressed"))
         expect_compressed = true;
   }

   /* Capture the driver's log to check it below. */
   fflush(stderr);
   saved_stderr = dup(STDERR_FILENO);
   driver_log = tmpfile();
   if (saved_stderr < 0 || !driver_log ||
       atexit(replay_driver_log) != 0 ||
       dup2(fileno(driver_log), STDERR_FILENO) < 0) {
      replay_driver_log();
      fprintf(stderr, "could not capture driver log\n");
      return 2;
   }

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));
   uint32_t count = 1;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &pdev));
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   CHECK(vkCreateDevice(pdev, &device_info, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue);
   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
   };
   CHECK(vkCreateCommandPool(dev, &pool_info, NULL, &pool));

   const VkImageUsageFlags transfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                      VK_IMAGE_USAGE_SAMPLED_BIT;
   const struct source cube = {
      .image = image_create(VK_IMAGE_TYPE_2D, VK_FORMAT_EAC_R11_UNORM_BLOCK,
                            VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,
                            (VkExtent3D){SRC_SIZE, SRC_SIZE, 1}, CUBE_LEVELS,
                            6, transfer),
      .levels = CUBE_LEVELS,
      .layers = 6,
   };
   const struct source volume = {
      .image = image_create(VK_IMAGE_TYPE_3D, VK_FORMAT_EAC_R11_UNORM_BLOCK, 0,
                            (VkExtent3D){SRC_SIZE, SRC_SIZE, VOL_DEPTH},
                            VOL_LEVELS, 1, transfer),
      .volume = true,
      .levels = VOL_LEVELS,
      .layers = VOL_DEPTH,
   };
   VkImage dst = image_create(VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32_SFLOAT, 0,
                              (VkExtent3D){DST_W, DST_H, 1}, DST_LEVELS, 1,
                              transfer | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
   fill(&cube);
   fill(&volume);
   VkCommandBuffer cmd = begin();
   to_general(cmd, dst);
   submit(cmd);

   unsigned copies = 0;
   for (unsigned i = 0; i < sizeof(cube_copies) / sizeof(cube_copies[0]); ++i) {
      for (uint32_t face = 0; face < 6; ++face, ++copies)
         copy_and_check(&cube, dst, &cube_copies[i], face, !shim);
   }
   for (unsigned i = 0; i < sizeof(volume_copies) / sizeof(volume_copies[0]);
        ++i) {
      const uint32_t slices = minify(VOL_DEPTH, volume_copies[i].level);
      for (uint32_t slice = 0; slice < slices; ++slice, ++copies)
         copy_and_check(&volume, dst, &volume_copies[i], slice, !shim);
   }
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);

   restore_stderr();
   rewind(driver_log);
   bool partial_view = false, dst_compressed = false;
   char line[1024];
   while (fgets(line, sizeof(line), driver_log)) {
      fputs(line, stderr);
      partial_view |= strstr(line, "addresses only the first") != NULL;
      dst_compressed |=
         strstr(line, "compression enabled: r32g32_float 129x127") != NULL;
   }

   fclose(driver_log);
   driver_log = NULL;

   if (partial_view) {
      fprintf(stderr, "FAIL: a copy viewed several layers through an "
                      "uncompressed view of the block-compressed source\n");
      failures++;
   }
   if (expect_compressed && !dst_compressed) {
      fprintf(stderr, "FAIL: the RG32F destination was not compressed, so "
                      "the graphics copy path was not exercised\n");
      failures++;
   }
   printf("%u copies%s: %s\n", copies, shim ? " (no readback)" : "",
          failures ? "FAIL" : "PASS");
   return failures ? 1 : 0;
}
