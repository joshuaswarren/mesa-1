/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact G17P oracle for the way Hyprland actually draws a window border.
 *
 * CHyprOpenGLImpl::renderBorder (Hyprland 0.56.1 src/render/OpenGL.cpp:2226)
 * does not draw the border once. It builds
 *
 *     borderRegion = damage & (window + borderSize)
 *     borderRegion -= window.expand(-borderSize - round)
 *
 * and then issues one glScissor + glDrawArrays(GL_TRIANGLE_STRIP, 0, 4) per
 * rectangle of that region, with alpha blending on. pixman decomposes a ring
 * into horizontal bands, so an ordinary window border is four adjacent
 * scissor rectangles that tile the ring exactly and share their edges. The
 * only state that changes between those draws is the scissor.
 *
 * That shape is what "the border shows weird disconnects" looks like from the
 * driver side, and it is not covered. Every G17P scissor assertion in this
 * tree is a full-frame rectangle (test-ppp-hal200-raster-core.c) or a crop
 * that still starts at x=0,y=0 (test-hk-g17p-full-frame-scissor.c), and both
 * only compare packed bytes -- which cannot tell an inclusive max_x from an
 * exclusive one, because a full-frame rectangle has no pixel at x == width
 * either way. The one test whose expected values would pin those edges,
 * ../../gl-tests/test-g17p-gl-premult-pingpong.c, draws two *overlapping*
 * scissors at x=0 and x=2 on an 8x4 target with a +/-1 tolerance, and there
 * is no recorded hardware run of it: run it with --hardware alongside this
 * oracle, because it reaches the same state through GL and Zink instead of
 * through Vulkan directly.
 *
 * What nothing covers at all is the case Hyprland actually creates:
 * *adjacent* rectangles that tile a region and share an edge, where an
 * inclusive/exclusive error is a doubly blended line rather than a
 * mis-coloured column, and where the seam is inside an AGX tile.
 *
 * On G17P the exact scissor does not travel through hk_flush_vp_state() at
 * all: hk_g17p_emit_native_bind_tail() carries it in the HAL200 raster record
 * (hk_cmd_draw.c:3416) and hk_cmd_draw.c:4910 skips the HAL300 viewport
 * emitter whenever that bind tail was emitted. So the ring is drawn through a
 * code path whose seam behaviour has no witness.
 *
 * The oracle makes the framebuffer a literal coverage count. The fragment
 * shader writes a constant 16/255 in every channel and the attachment blends
 * ONE/ONE, so after the ladder each pixel reads back as 16 * (number of
 * scissored draws that covered it):
 *
 *      0  a hole      -- a scissor that clipped a row or column short
 *     16  correct     -- covered exactly once
 *     32  a seam      -- two adjacent rectangles both wrote the shared edge
 *
 * Modes, each moving exactly one variable from the previous one:
 *
 *   whole    one draw, scissor = the whole attachment       must PASS
 *   tile     four rectangles tiling the whole attachment    must PASS
 *   ring     Hyprland's four-rectangle border ring          must PASS
 *   damage   the ring again over a loaded previous frame    must PASS
 *   gap      the ring with one rectangle one row short      must FAIL
 *   overlap  the ring with one rectangle one row long       must FAIL
 *
 * The two must-FAIL modes are what make the must-PASS modes mean anything: a
 * comparison that accepts any image, or a run where nothing executed, reports
 * success on the PASS modes too. They draw a deliberately wrong rectangle
 * list and are checked against the correct reference.
 *
 * Nothing here is hardware-validated. Under the no-op drm-shim nothing
 * executes, so only --record-only (recording and encoding) is meaningful
 * there; every pixel claim needs a real G17P.
 */

#include <vulkan/vulkan.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "g17p_vulkan_scissor_ring_frag.h"
#include "g17p_vulkan_scissor_ring_vert.h"

#define TEST_NAME "f55-hypr-border-scissor-ring"
#define T8140_DEVICE_ID 0x8140

#define WIDTH 96
#define HEIGHT 96
#define GUARD 64
#define PIXEL_BYTES 4
#define IMAGE_BYTES (WIDTH * HEIGHT * PIXEL_BYTES)
#define READBACK_BYTES (IMAGE_BYTES + 2 * GUARD)

/* One draw's contribution, in unorm8 units. */
#define STEP 16

enum mode {
   MODE_WHOLE,
   MODE_TILE,
   MODE_RING,
   MODE_DAMAGE,
   MODE_GAP,
   MODE_OVERLAP,
};

struct rect {
   int32_t x, y;
   uint32_t w, h;
};

/*
 * Hyprland's own geometry, at a size small enough to verify by hand: a 64x64
 * window at (16,16) with a 5px border. newBox is [11,85), the subtracted
 * inner box is window.expand(-5) = [21,75), and pixman bands the difference
 * into a top strip, two side strips and a bottom strip. The band edges at
 * y=21 and y=75 are deliberately not multiples of 32, so they fall inside an
 * AGX tile rather than on a tile boundary where a coarse region clip would
 * accidentally give the right answer.
 */
static const struct rect RING[] = {
   {11, 11, 74, 10},
   {11, 21, 10, 54},
   {75, 21, 10, 54},
   {11, 75, 74, 10},
};

/* The same ring with the left band one row short: row y=74 of x in [11,21)
 * is covered by nothing. This is what a max_y that is one too small does. */
static const struct rect RING_GAP[] = {
   {11, 11, 74, 10},
   {11, 21, 10, 53},
   {75, 21, 10, 54},
   {11, 75, 74, 10},
};

/* The same ring with the top strip one row long: row y=21 is covered twice.
 * This is what a max_y that is one too large does. */
static const struct rect RING_OVERLAP[] = {
   {11, 11, 74, 11},
   {11, 21, 10, 54},
   {75, 21, 10, 54},
   {11, 75, 74, 10},
};

/* Four rectangles tiling the whole attachment. The split is at 37 and 59 so
 * that neither seam is tile-aligned. */
static const struct rect TILE[] = {
   {0, 0, 37, 59},
   {37, 0, 59, 59},
   {0, 59, 37, 37},
   {37, 59, 59, 37},
};

static const struct rect WHOLE[] = {
   {0, 0, WIDTH, HEIGHT},
};

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

static uint32_t
memory_type(const VkPhysicalDeviceMemoryProperties *properties, uint32_t bits,
            VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }
   die("no compatible memory type");
   return 0;
}

/* The rectangles a mode actually draws. */
static const struct rect *
drawn_rects(enum mode mode, unsigned *count)
{
   switch (mode) {
   case MODE_WHOLE:
      *count = 1;
      return WHOLE;
   case MODE_TILE:
      *count = 4;
      return TILE;
   case MODE_GAP:
      *count = 4;
      return RING_GAP;
   case MODE_OVERLAP:
      *count = 4;
      return RING_OVERLAP;
   case MODE_RING:
   case MODE_DAMAGE:
   default:
      *count = 4;
      return RING;
   }
}

/* The rectangles a correct driver would have covered. For the two negative
 * modes this is deliberately not what was drawn. */
static const struct rect *
expected_rects(enum mode mode, unsigned *count)
{
   switch (mode) {
   case MODE_WHOLE:
      *count = 1;
      return WHOLE;
   case MODE_TILE:
      *count = 4;
      return TILE;
   default:
      *count = 4;
      return RING;
   }
}

static bool
inside(const struct rect *r, int px, int py)
{
   return px >= r->x && px < r->x + (int)r->w && py >= r->y &&
          py < r->y + (int)r->h;
}

/* MODE_DAMAGE draws a whole-attachment pass first and then the ring over it
 * with loadOp LOAD, exactly as a compositor redraws one damaged window. */
static unsigned
expected_coverage(enum mode mode, int px, int py)
{
   unsigned count = 0, n = 0;
   const struct rect *rects = expected_rects(mode, &n);
   for (unsigned i = 0; i < n; ++i)
      count += inside(&rects[i], px, py) ? 1 : 0;
   if (mode == MODE_DAMAGE)
      ++count;
   return count;
}

static bool
verify(const uint8_t *map, enum mode mode, const char *stage)
{
   for (size_t i = 0; i < GUARD; ++i) {
      if (map[i] != 0xa5 || map[GUARD + IMAGE_BYTES + i] != 0xa5) {
         fprintf(stderr, TEST_NAME ": %s readback guard changed at byte %zu\n",
                 stage, i);
         return false;
      }
   }

   unsigned holes = 0, seams = 0, other = 0, reported = 0;
   for (int py = 0; py < HEIGHT; ++py) {
      for (int px = 0; px < WIDTH; ++px) {
         const unsigned want = expected_coverage(mode, px, py) * STEP;
         const uint8_t *actual =
            map + GUARD + ((size_t)py * WIDTH + px) * PIXEL_BYTES;
         bool bad = false;
         for (int c = 0; c < 4; ++c) {
            const int delta = (int)actual[c] - (int)want;
            if (delta > 1 || delta < -1)
               bad = true;
         }
         if (!bad)
            continue;

         if (actual[0] < want)
            ++holes;
         else if (actual[0] > want)
            ++seams;
         else
            ++other;

         if (reported < 8) {
            fprintf(stderr,
                    TEST_NAME ": %s (%d,%d) covered %u time(s), expected %u "
                              "(rgba %u,%u,%u,%u)\n",
                    stage, px, py, actual[0] / STEP, want / STEP, actual[0],
                    actual[1], actual[2], actual[3]);
            ++reported;
         }
      }
   }

   if (holes || seams || other) {
      fprintf(stderr,
              TEST_NAME ": %s %u under-covered, %u over-covered, %u "
                        "inconsistent of %d pixels\n",
              stage, holes, seams, other, WIDTH * HEIGHT);
      return false;
   }
   return true;
}

/* A reference that is uniform, or that is mostly the clear colour, would let
 * an unwritten attachment pass. Refuse to run rather than report a hollow
 * PASS. */
static bool
reference_is_discriminating(enum mode mode, unsigned *covered_out)
{
   const unsigned base = mode == MODE_DAMAGE ? 1 : 0;
   unsigned covered = 0, uncovered = 0;
   for (int py = 0; py < HEIGHT; ++py) {
      for (int px = 0; px < WIDTH; ++px) {
         const unsigned n = expected_coverage(mode, px, py);
         /* A typo in a rectangle table would otherwise show up as a
          * reference that expects a doubly blended seam, and the oracle
          * would then demand the very artefact it exists to detect. */
         if (n > base + 1)
            die("the reference rectangles overlap; the table is wrong");
         if (n > base)
            ++covered;
         else
            ++uncovered;
      }
   }
   *covered_out = covered;
   /* MODE_WHOLE and MODE_TILE cover every pixel exactly once, so their
    * discrimination is the count itself: a cleared attachment reads 0.
    * Every other mode additionally has to have both kinds of pixel, or a
    * uniform image would satisfy it. */
   if (mode == MODE_WHOLE || mode == MODE_TILE)
      return covered == WIDTH * HEIGHT && uncovered == 0;
   return covered > 0 && uncovered > 0;
}

int
main(int argc, char **argv)
{
   if (argc < 2)
      die("usage: " TEST_NAME
          " <whole|tile|ring|damage|gap|overlap> [--record-only]");

   enum mode mode;
   if (strcmp(argv[1], "whole") == 0)
      mode = MODE_WHOLE;
   else if (strcmp(argv[1], "tile") == 0)
      mode = MODE_TILE;
   else if (strcmp(argv[1], "ring") == 0)
      mode = MODE_RING;
   else if (strcmp(argv[1], "damage") == 0)
      mode = MODE_DAMAGE;
   else if (strcmp(argv[1], "gap") == 0)
      mode = MODE_GAP;
   else if (strcmp(argv[1], "overlap") == 0)
      mode = MODE_OVERLAP;
   else
      die("mode must be whole, tile, ring, damage, gap or overlap");

   const bool negative = mode == MODE_GAP || mode == MODE_OVERLAP;
   bool record_only = false;
   for (int i = 2; i < argc; ++i) {
      if (strcmp(argv[i], "--record-only") == 0)
         record_only = true;
      else
         die("unknown option");
   }

   unsigned covered = 0;
   if (!reference_is_discriminating(mode, &covered))
      die("the reference for this mode cannot distinguish an unwritten "
          "attachment");

   unsigned draw_count = 0;
   const struct rect *rects = drawn_rects(mode, &draw_count);
   printf(TEST_NAME ": mode=%s %u scissored draw(s), %u of %d pixels covered "
                    "by the reference\n",
          argv[1], draw_count + (mode == MODE_DAMAGE ? 1 : 0), covered,
          WIDTH * HEIGHT);

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

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      die("out of host memory");
   check(vkEnumeratePhysicalDevices(instance, &physical_count,
                                    physical_devices),
         "vkEnumeratePhysicalDevices(list)");
   VkPhysicalDevice physical = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < physical_count; ++i) {
      VkPhysicalDeviceProperties candidate;
      vkGetPhysicalDeviceProperties(physical_devices[i], &candidate);
      if (candidate.deviceID == T8140_DEVICE_ID) {
         physical = physical_devices[i];
         break;
      }
   }
   free(physical_devices);
   if (physical == VK_NULL_HANDLE)
      die("T8140/G17P Vulkan device was not found");

   uint32_t family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, NULL);
   VkQueueFamilyProperties *families = calloc(family_count, sizeof(*families));
   if (!families)
      die("out of host memory");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families);
   uint32_t family = UINT32_MAX;
   for (uint32_t i = 0; i < family_count; ++i) {
      if (families[i].queueCount &&
          (families[i].queueFlags &
           (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT)) ==
             (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT)) {
         family = i;
         break;
      }
   }
   free(families);
   if (family == UINT32_MAX)
      die("device has no graphics-and-transfer queue");

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
   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   VkImage target;
   const VkImageCreateInfo target_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {WIDTH, HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   check(vkCreateImage(device, &target_info, NULL, &target), "vkCreateImage");
   VkMemoryRequirements target_requirements;
   vkGetImageMemoryRequirements(device, target, &target_requirements);
   const VkMemoryAllocateInfo target_allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = target_requirements.size,
      .memoryTypeIndex = memory_type(&memory_properties,
                                     target_requirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory target_memory;
   check(vkAllocateMemory(device, &target_allocation, NULL, &target_memory),
         "vkAllocateMemory(target)");
   check(vkBindImageMemory(device, target, target_memory, 0),
         "vkBindImageMemory(target)");

   const VkImageViewCreateInfo target_view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = target,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1,
         .layerCount = 1,
      },
   };
   VkImageView target_view;
   check(vkCreateImageView(device, &target_view_info, NULL, &target_view),
         "vkCreateImageView(target)");

   VkBuffer readback;
   const VkBufferCreateInfo readback_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = READBACK_BYTES,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   check(vkCreateBuffer(device, &readback_info, NULL, &readback),
         "vkCreateBuffer(readback)");
   VkMemoryRequirements readback_requirements;
   vkGetBufferMemoryRequirements(device, readback, &readback_requirements);
   const VkMemoryAllocateInfo readback_allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = readback_requirements.size,
      .memoryTypeIndex =
         memory_type(&memory_properties, readback_requirements.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory readback_memory;
   check(vkAllocateMemory(device, &readback_allocation, NULL,
                          &readback_memory),
         "vkAllocateMemory(readback)");
   check(vkBindBufferMemory(device, readback, readback_memory, 0),
         "vkBindBufferMemory(readback)");
   void *mapped = NULL;
   check(vkMapMemory(device, readback_memory, 0, READBACK_BYTES, 0, &mapped),
         "vkMapMemory(readback)");
   uint8_t *map = mapped;
   memset(map, 0xa5, READBACK_BYTES);

   /* Two render passes: the first clears, the second loads. MODE_DAMAGE uses
    * both, every other mode uses only the clearing one, so a mode difference
    * never changes anything but the number of passes and rectangles. */
   const VkAttachmentDescription attachments[2] = {
      {
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      },
      {
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      },
   };
   const VkAttachmentReference reference = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &reference,
   };
   const VkSubpassDependency dependencies[2] = {
      {
         .srcSubpass = VK_SUBPASS_EXTERNAL,
         .dstSubpass = 0,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                          VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      },
      {
         .srcSubpass = 0,
         .dstSubpass = VK_SUBPASS_EXTERNAL,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT |
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                          VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
      },
   };

   VkRenderPass render_pass[2];
   for (unsigned i = 0; i < 2; ++i) {
      VkAttachmentDescription description = attachments[i];
      /* Whichever pass runs last has to hand the image to the copy. */
      if ((mode == MODE_DAMAGE) == (i == 1))
         description.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      else
         description.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      const VkRenderPassCreateInfo pass_info = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
         .attachmentCount = 1,
         .pAttachments = &description,
         .subpassCount = 1,
         .pSubpasses = &subpass,
         .dependencyCount = 2,
         .pDependencies = dependencies,
      };
      check(vkCreateRenderPass(device, &pass_info, NULL, &render_pass[i]),
            "vkCreateRenderPass");
   }

   VkFramebuffer framebuffer[2];
   for (unsigned i = 0; i < 2; ++i) {
      const VkFramebufferCreateInfo framebuffer_info = {
         .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
         .renderPass = render_pass[i],
         .attachmentCount = 1,
         .pAttachments = &target_view,
         .width = WIDTH,
         .height = HEIGHT,
         .layers = 1,
      };
      check(vkCreateFramebuffer(device, &framebuffer_info, NULL,
                                &framebuffer[i]),
            "vkCreateFramebuffer");
   }

   VkShaderModule vert_module, frag_module;
   const VkShaderModuleCreateInfo vert_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_scissor_ring_vert_spv),
      .pCode = g17p_vulkan_scissor_ring_vert_spv,
   };
   const VkShaderModuleCreateInfo frag_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_scissor_ring_frag_spv),
      .pCode = g17p_vulkan_scissor_ring_frag_spv,
   };
   check(vkCreateShaderModule(device, &vert_module_info, NULL, &vert_module),
         "vkCreateShaderModule(vert)");
   check(vkCreateShaderModule(device, &frag_module_info, NULL, &frag_module),
         "vkCreateShaderModule(frag)");

   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pipeline_layout;
   check(vkCreatePipelineLayout(device, &layout_info, NULL, &pipeline_layout),
         "vkCreatePipelineLayout");

   const VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vert_module,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = frag_module,
         .pName = "main",
      },
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {
      .width = WIDTH, .height = HEIGHT, .maxDepth = 1.0f};
   const VkRect2D full_scissor = {.extent = {WIDTH, HEIGHT}};
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &full_scissor,
   };
   /* The scissor is the one thing Hyprland changes between border draws, so
    * it is the one thing this pipeline leaves dynamic. */
   const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_SCISSOR};
   const VkPipelineDynamicStateCreateInfo dynamic_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 1,
      .pDynamicStates = dynamic_states,
   };
   const VkPipelineRasterizationStateCreateInfo rasterization = {
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
   const VkPipelineColorBlendAttachmentState blend_attachment = {
      .blendEnable = VK_TRUE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blend_attachment,
   };

   VkPipeline pipeline[2];
   for (unsigned i = 0; i < 2; ++i) {
      const VkGraphicsPipelineCreateInfo pipeline_info = {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .stageCount = 2,
         .pStages = stages,
         .pVertexInputState = &vertex_input,
         .pInputAssemblyState = &input_assembly,
         .pViewportState = &viewport_state,
         .pRasterizationState = &rasterization,
         .pMultisampleState = &multisample,
         .pColorBlendState = &blend,
         .pDynamicState = &dynamic_state,
         .layout = pipeline_layout,
         .renderPass = render_pass[i],
         .subpass = 0,
      };
      check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL, &pipeline[i]),
            "vkCreateGraphicsPipelines");
   }
   printf(TEST_NAME ": pipelines created (blend one/one, dynamic scissor)\n");

   const VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VkCommandPool command_pool;
   check(vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command_buffer;
   check(vkAllocateCommandBuffers(device, &command_info, &command_buffer),
         "vkAllocateCommandBuffers");

   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command_buffer, &begin), "vkBeginCommandBuffer");

   VkClearValue clear = {0};
   const VkRenderPassBeginInfo clear_pass = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass[0],
      .framebuffer = framebuffer[0],
      .renderArea = {.extent = {WIDTH, HEIGHT}},
      .clearValueCount = 1,
      .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(command_buffer, &clear_pass,
                        VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                     pipeline[0]);
   if (mode == MODE_DAMAGE) {
      /* The previous frame: one unscissored draw over the whole attachment. */
      vkCmdSetScissor(command_buffer, 0, 1, &full_scissor);
      vkCmdDraw(command_buffer, 3, 1, 0, 0);
   } else {
      for (unsigned i = 0; i < draw_count; ++i) {
         const VkRect2D scissor = {
            .offset = {rects[i].x, rects[i].y},
            .extent = {rects[i].w, rects[i].h},
         };
         vkCmdSetScissor(command_buffer, 0, 1, &scissor);
         vkCmdDraw(command_buffer, 3, 1, 0, 0);
      }
   }
   vkCmdEndRenderPass(command_buffer);

   if (mode == MODE_DAMAGE) {
      const VkRenderPassBeginInfo load_pass = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = render_pass[1],
         .framebuffer = framebuffer[1],
         .renderArea = {.extent = {WIDTH, HEIGHT}},
      };
      vkCmdBeginRenderPass(command_buffer, &load_pass,
                           VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipeline[1]);
      for (unsigned i = 0; i < draw_count; ++i) {
         const VkRect2D scissor = {
            .offset = {rects[i].x, rects[i].y},
            .extent = {rects[i].w, rects[i].h},
         };
         vkCmdSetScissor(command_buffer, 0, 1, &scissor);
         vkCmdDraw(command_buffer, 3, 1, 0, 0);
      }
      vkCmdEndRenderPass(command_buffer);
   }

   const VkBufferImageCopy readback_copy = {
      .bufferOffset = GUARD,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {WIDTH, HEIGHT, 1},
   };
   vkCmdCopyImageToBuffer(command_buffer, target,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                          &readback_copy);
   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
   };
   check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
   check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");

   /* Under the no-op drm-shim the submission is walked by the G17P stream
    * validator and then discarded, so the command stream is checked but no
    * pixel was ever written. Stop before the comparison rather than report a
    * verdict about an attachment nothing touched. */
   if (record_only) {
      printf(TEST_NAME ": RECORDS (VkResult 0) mode=%s -- encoded and "
                       "submitted, nothing executed, no pixel claim\n",
             argv[1]);
      return EXIT_SUCCESS;
   }

   const bool ok = verify(map, mode, argv[1]);

   if (negative) {
      if (ok) {
         fprintf(stderr,
                 TEST_NAME ": negative control %s PASSED the correct "
                           "reference; the comparison does not see a "
                           "one-row scissor error, so every PASS above is "
                           "meaningless\n",
                 argv[1]);
         return EXIT_FAILURE;
      }
      printf(TEST_NAME ": negative control %s failed as required -- the "
                       "comparison detects a one-row scissor error\n",
             argv[1]);
      return EXIT_SUCCESS;
   }

   if (!ok)
      return EXIT_FAILURE;

   printf(TEST_NAME ": PASS -- %s matched the host coverage reference "
                    "exactly; readback guards unchanged\n",
          argv[1]);
   return EXIT_SUCCESS;
}
