/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact G17P oracle for Qt 6's HiQ subpixel distance-field text material.
 *
 * Omarchy's QuickShell surfaces render their glyphs with this material, and on
 * Neo its output is wrong in a very specific way: a glyph run comes out as a
 * near-uniformly covered rectangle with a per-channel fringe, instead of sharp
 * glyphs.  Reading the two captures that show it
 * (neo-gpu-harness/logs/omarchy-about-zink-sync-blur-off-3ff6094b.png and
 * qs-full-sdf-capture-20260910-0737.png) back through the material's own blend
 * -- CONSTANT_COLOR / ONE_MINUS_SRC_COLOR, so the destination contributes
 * (1 - coverage) per channel -- recovers coverage triples of the form
 * (a, a + d, a + 2d) with d quantised to +/- a/3.  That is exactly
 * (n.x + n.y + c)/3, (n.y + c + n.z)/3, (c + n.z + n.w)/3 with the taps
 * saturated, i.e. the five taps disagree wildly where they should agree.
 *
 * The material is the only thing in the desktop that issues four
 * OpImageSampleProjImplicitLod of one sampler alongside a fifth plain
 * implicit-LOD sample of the same sampler, reading an 8-bit atlas through the
 * alpha component of a zero/zero/zero/red swizzle.  This oracle reproduces that
 * shape with taps that land on exact texel centres, so the expected image is
 * computable on the host, and checks it three ways:
 *
 *   center   the plain implicit-LOD tap alone, blending disabled
 *   blend    the same shader with Qt's blend turned on, isolating blend state
 *   taps     the four offset taps alone, blending disabled
 *   hiq      the whole material including both smoothsteps and the blend
 *
 * Three candidate mechanisms were already eliminated on the host, without
 * hardware, before this oracle was written:
 *
 *   - textureProj is not it.  nir_lower_tex sets lower_txp = ~0 for AGX
 *     (agx_nir_lower_texture.c:801), and the AGX IR for the projective and the
 *     hand-divided shader is byte-identical apart from the source hash.
 *   - The five-member f156 fragment sample group is not unique to this shader:
 *     the hardware-passing Hyprland blur oracle (f34) emits five samples too.
 *   - The single-component w write mask is not mis-encoded.  Apple's own
 *     applegpu_g17p output for a .w-only, a .x-only and a four-component
 *     texture2d sample differs in exactly the mask field, and
 *     agx3_tex_encode_sample reproduces all three byte-for-byte.
 *
 * What is left that no passing oracle covers is the pair this ladder isolates:
 * an 8-bit single-channel image sampled through a zero/zero/zero/red swizzle,
 * and Qt's CONSTANT_COLOR / ONE_MINUS_SRC_COLOR blend.
 *
 * Nothing here is hardware-validated.  Under drm-shim nothing executes, so only
 * the build/encode gate in run-f41-qt-subpixel-text.sh is meaningful there.
 */

#include <vulkan/vulkan.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TEST_NAME "f41-qt-subpixel-text"
#define T8140_DEVICE_ID 0x8140
#define WIDTH 32
#define HEIGHT 32
#define ATLAS_WIDTH 32
#define ATLAS_HEIGHT 32
#define GUARD 64
#define PIXEL_BYTES 4
#define IMAGE_BYTES (WIDTH * HEIGHT * PIXEL_BYTES)
#define READBACK_BYTES (IMAGE_BYTES + 2 * GUARD)
#define ATLAS_BYTES (ATLAS_WIDTH * ATLAS_HEIGHT)

/* Exactly representable in binary32 so the host reference and the shader agree
 * on the inputs; only the arithmetic between them is under test. */
#define ALPHA_MIN 0.40625f
#define ALPHA_MAX 0.59375f
#define COLOR_W 0.75f

static const float BLEND_CONSTANTS[4] = {0.25f, 0.5f, 0.75f, 1.0f};
static const uint8_t CLEAR_BYTES[4] = {16, 32, 48, 255};

enum mode {
   MODE_CENTER,
   MODE_BLEND,
   MODE_TAPS,
   MODE_HIQ,
};

struct params {
   float alpha_min;
   float alpha_max;
   float color_w;
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

static uint32_t *
read_spv(const char *path, size_t *size_out)
{
   FILE *file = fopen(path, "rb");
   if (!file)
      die("cannot open SPIR-V module");
   if (fseek(file, 0, SEEK_END) != 0)
      die("cannot size SPIR-V module");
   long size = ftell(file);
   if (size <= 0 || (size & 3) != 0)
      die("invalid SPIR-V module size");
   rewind(file);
   uint32_t *words = malloc((size_t)size);
   if (!words || fread(words, 1, (size_t)size, file) != (size_t)size)
      die("cannot read SPIR-V module");
   fclose(file);
   *size_out = (size_t)size;
   return words;
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

/* The atlas deliberately changes fast along x: every tap of a pixel sees a
 * different value, so a driver that collapses the taps, drops the projective
 * divide, or leaves the destination untouched cannot land on the reference. */
static uint8_t
atlas_texel(int x, int y)
{
   if (x < 0)
      x = 0;
   if (x >= ATLAS_WIDTH)
      x = ATLAS_WIDTH - 1;
   if (y < 0)
      y = 0;
   if (y >= ATLAS_HEIGHT)
      y = ATLAS_HEIGHT - 1;
   return (uint8_t)((x * 17 + y * 5) & 0xff);
}

static float
clampf(float v, float lo, float hi)
{
   return v < lo ? lo : (v > hi ? hi : v);
}

static float
mixf(float a, float b, float t)
{
   return a * (1.0f - t) + b * t;
}

static float
smoothstepf(float lo, float hi, float x)
{
   float t = clampf((x - lo) / (hi - lo), 0.0f, 1.0f);
   return t * t * (3.0f - 2.0f * t);
}

static uint8_t
unorm8(float v)
{
   float scaled = clampf(v, 0.0f, 1.0f) * 255.0f;
   return (uint8_t)(scaled + 0.5f);
}

/* shift is the negative control: it moves every tap one texel, which is what a
 * dropped projective divide or a mis-encoded coordinate would do. */
static void
reference_pixel(enum mode mode, int px, int py, int shift, uint8_t out[4])
{
   const float far_left = atlas_texel(px - 2 + shift, py) / 255.0f;
   const float near_left = atlas_texel(px - 1 + shift, py) / 255.0f;
   const float centre = atlas_texel(px + shift, py) / 255.0f;
   const float near_right = atlas_texel(px + 1 + shift, py) / 255.0f;
   const float far_right = atlas_texel(px + 2 + shift, py) / 255.0f;

   float src[4];
   bool blended;

   if (mode == MODE_CENTER || mode == MODE_BLEND) {
      src[0] = src[1] = src[2] = src[3] = centre;
      blended = mode == MODE_BLEND;
   } else if (mode == MODE_TAPS) {
      src[0] = far_left;
      src[1] = near_left;
      src[2] = near_right;
      src[3] = far_right;
      blended = false;
   } else {
      float n[4] = {far_left, near_left, near_right, far_right};
      float c = centre;

      const float dx = fminf(fabsf(n[1] - n[0]) * 2.0f, 0.67f);
      const float dy = fminf(fabsf(n[3] - n[2]) * 2.0f, 0.67f);
      const float lo_x = mixf(ALPHA_MIN, 0.5f, dx);
      const float lo_y = mixf(ALPHA_MIN, 0.5f, dy);
      const float hi_x = mixf(ALPHA_MAX, 0.5f, dx);
      const float hi_y = mixf(ALPHA_MAX, 0.5f, dy);

      n[0] = smoothstepf(lo_x, hi_x, n[0]);
      n[1] = smoothstepf(lo_x, hi_x, n[1]);
      n[2] = smoothstepf(lo_y, hi_y, n[2]);
      n[3] = smoothstepf(lo_y, hi_y, n[3]);
      c = smoothstepf(lo_x + lo_y, hi_x + hi_y, 2.0f * c);

      src[0] = 0.333f * (n[0] + n[1] + c) * COLOR_W;
      src[1] = 0.333f * (n[1] + c + n[2]) * COLOR_W;
      src[2] = 0.333f * (c + n[2] + n[3]) * COLOR_W;
      src[3] = c * COLOR_W;
      blended = true;
   }

   for (int i = 0; i < 4; ++i) {
      float value = src[i];
      if (blended) {
         const float dst = CLEAR_BYTES[i] / 255.0f;
         const float factor = clampf(value, 0.0f, 1.0f);
         value = BLEND_CONSTANTS[i] * factor + dst * (1.0f - factor);
      }
      out[i] = unorm8(value);
   }
}

static bool
verify(const uint8_t *map, enum mode mode, int shift, const char *stage)
{
   for (size_t i = 0; i < GUARD; ++i) {
      if (map[i] != 0xa5 || map[GUARD + IMAGE_BYTES + i] != 0xa5) {
         fprintf(stderr, TEST_NAME ": %s readback guard changed at byte %zu\n",
                 stage, i);
         return false;
      }
   }

   unsigned mismatches = 0;
   for (int py = 0; py < HEIGHT; ++py) {
      for (int px = 0; px < WIDTH; ++px) {
         uint8_t expected[4];
         reference_pixel(mode, px, py, shift, expected);
         const uint8_t *actual =
            map + GUARD + ((size_t)py * WIDTH + px) * PIXEL_BYTES;
         for (int i = 0; i < 4; ++i) {
            const int delta = (int)actual[i] - (int)expected[i];
            if (delta > 1 || delta < -1) {
               if (mismatches < 8) {
                  fprintf(stderr,
                          TEST_NAME ": %s pixel (%d,%d) got "
                                    "%02x%02x%02x%02x expected "
                                    "%02x%02x%02x%02x\n",
                          stage, px, py, actual[0], actual[1], actual[2],
                          actual[3], expected[0], expected[1], expected[2],
                          expected[3]);
               }
               ++mismatches;
               break;
            }
         }
      }
   }
   if (mismatches) {
      fprintf(stderr, TEST_NAME ": %s %u of %d pixels differ\n", stage,
              mismatches, WIDTH * HEIGHT);
      return false;
   }
   return true;
}

/* A reference that is nearly constant, or that mostly reproduces the clear
 * colour, would let an unwritten or merely cleared attachment pass.  Refuse to
 * run in that case rather than report a hollow PASS. */
static unsigned
reference_clear_coloured(enum mode mode)
{
   unsigned count = 0;
   for (int py = 0; py < HEIGHT; ++py) {
      for (int px = 0; px < WIDTH; ++px) {
         uint8_t pixel[4];
         reference_pixel(mode, px, py, 0, pixel);
         if (memcmp(pixel, CLEAR_BYTES, sizeof(pixel)) == 0)
            ++count;
      }
   }
   return count;
}

static unsigned
reference_distinct_values(enum mode mode)
{
   uint32_t seen[WIDTH * HEIGHT];
   unsigned count = 0;
   for (int py = 0; py < HEIGHT; ++py) {
      for (int px = 0; px < WIDTH; ++px) {
         uint8_t pixel[4];
         reference_pixel(mode, px, py, 0, pixel);
         uint32_t packed = (uint32_t)pixel[0] | ((uint32_t)pixel[1] << 8) |
                           ((uint32_t)pixel[2] << 16) |
                           ((uint32_t)pixel[3] << 24);
         bool known = false;
         for (unsigned i = 0; i < count; ++i) {
            if (seen[i] == packed) {
               known = true;
               break;
            }
         }
         if (!known)
            seen[count++] = packed;
      }
   }
   return count;
}

int
main(int argc, char **argv)
{
   if (argc < 4)
      die("usage: f41-qt-subpixel-text <vert.spv> <frag.spv> "
          "<center|blend|taps|hiq> [--negative] [--identity-swizzle]");

   enum mode mode;
   if (strcmp(argv[3], "center") == 0)
      mode = MODE_CENTER;
   else if (strcmp(argv[3], "blend") == 0)
      mode = MODE_BLEND;
   else if (strcmp(argv[3], "taps") == 0)
      mode = MODE_TAPS;
   else if (strcmp(argv[3], "hiq") == 0)
      mode = MODE_HIQ;
   else
      die("mode must be center, blend, taps or hiq");

   bool negative = false;
   bool identity_swizzle = false;
   for (int i = 4; i < argc; ++i) {
      if (strcmp(argv[i], "--negative") == 0)
         negative = true;
      else if (strcmp(argv[i], "--identity-swizzle") == 0)
         identity_swizzle = true;
      else
         die("unknown option");
   }

   const unsigned distinct = reference_distinct_values(mode);
   if (distinct < 16) {
      fprintf(stderr,
              TEST_NAME ": reference for %s has only %u distinct pixels; a "
                        "cleared buffer could pass it\n",
              argv[3], distinct);
      return EXIT_FAILURE;
   }
   const unsigned clear_coloured = reference_clear_coloured(mode);
   if (clear_coloured * 3 >= WIDTH * HEIGHT) {
      fprintf(stderr,
              TEST_NAME ": %u of %d reference pixels for %s are the clear "
                        "colour; a merely cleared attachment could pass it\n",
              clear_coloured, WIDTH * HEIGHT, argv[3]);
      return EXIT_FAILURE;
   }

   size_t vert_size = 0, frag_size = 0;
   uint32_t *vert_spv = read_spv(argv[1], &vert_size);
   uint32_t *frag_spv = read_spv(argv[2], &frag_size);
   printf(TEST_NAME ": mode=%s vert=%zu words frag=%zu words reference has %u "
                    "distinct pixels, %u of %d equal to the clear colour\n",
          argv[3], vert_size / 4, frag_size / 4, distinct, clear_coloured,
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

   /* The atlas: R8_UNORM, uploaded as a sub-region buffer-to-image copy, and
    * viewed through the zero/zero/zero/red swizzle Zink gives GL_ALPHA8. */
   VkImage atlas;
   const VkImageCreateInfo atlas_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8_UNORM,
      .extent = {ATLAS_WIDTH, ATLAS_HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   check(vkCreateImage(device, &atlas_info, NULL, &atlas), "vkCreateImage(atlas)");
   VkMemoryRequirements atlas_requirements;
   vkGetImageMemoryRequirements(device, atlas, &atlas_requirements);
   const VkMemoryAllocateInfo atlas_allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = atlas_requirements.size,
      .memoryTypeIndex = memory_type(&memory_properties,
                                     atlas_requirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory atlas_memory;
   check(vkAllocateMemory(device, &atlas_allocation, NULL, &atlas_memory),
         "vkAllocateMemory(atlas)");
   check(vkBindImageMemory(device, atlas, atlas_memory, 0),
         "vkBindImageMemory(atlas)");

   const VkComponentMapping alpha_swizzle = {
      .r = VK_COMPONENT_SWIZZLE_ZERO,
      .g = VK_COMPONENT_SWIZZLE_ZERO,
      .b = VK_COMPONENT_SWIZZLE_ZERO,
      .a = VK_COMPONENT_SWIZZLE_R,
   };
   const VkComponentMapping identity = {
      .r = VK_COMPONENT_SWIZZLE_IDENTITY,
      .g = VK_COMPONENT_SWIZZLE_IDENTITY,
      .b = VK_COMPONENT_SWIZZLE_IDENTITY,
      .a = VK_COMPONENT_SWIZZLE_IDENTITY,
   };
   const VkImageViewCreateInfo atlas_view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = atlas,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8_UNORM,
      .components = identity_swizzle ? identity : alpha_swizzle,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1,
         .layerCount = 1,
      },
   };
   VkImageView atlas_view;
   check(vkCreateImageView(device, &atlas_view_info, NULL, &atlas_view),
         "vkCreateImageView(atlas)");

   /* Destination colour attachment. */
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
   check(vkCreateImage(device, &target_info, NULL, &target),
         "vkCreateImage(target)");
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

   /* Staging buffer holds the atlas bytes; readback buffer holds the result. */
   VkBuffer staging;
   const VkBufferCreateInfo staging_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = ATLAS_BYTES,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   check(vkCreateBuffer(device, &staging_info, NULL, &staging),
         "vkCreateBuffer(staging)");
   VkMemoryRequirements staging_requirements;
   vkGetBufferMemoryRequirements(device, staging, &staging_requirements);
   const VkMemoryAllocateInfo staging_allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = staging_requirements.size,
      .memoryTypeIndex = memory_type(&memory_properties,
                                     staging_requirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory staging_memory;
   check(vkAllocateMemory(device, &staging_allocation, NULL, &staging_memory),
         "vkAllocateMemory(staging)");
   check(vkBindBufferMemory(device, staging, staging_memory, 0),
         "vkBindBufferMemory(staging)");
   uint8_t *staging_map;
   check(vkMapMemory(device, staging_memory, 0, ATLAS_BYTES, 0,
                     (void **)&staging_map),
         "vkMapMemory(staging)");
   for (int y = 0; y < ATLAS_HEIGHT; ++y)
      for (int x = 0; x < ATLAS_WIDTH; ++x)
         staging_map[y * ATLAS_WIDTH + x] = atlas_texel(x, y);
   vkUnmapMemory(device, staging_memory);

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
      .memoryTypeIndex = memory_type(&memory_properties,
                                     readback_requirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory readback_memory;
   check(vkAllocateMemory(device, &readback_allocation, NULL, &readback_memory),
         "vkAllocateMemory(readback)");
   check(vkBindBufferMemory(device, readback, readback_memory, 0),
         "vkBindBufferMemory(readback)");
   uint8_t *map;
   check(vkMapMemory(device, readback_memory, 0, READBACK_BYTES, 0,
                     (void **)&map),
         "vkMapMemory(readback)");
   memset(map, 0xa5, READBACK_BYTES);

   VkBuffer uniform;
   const VkBufferCreateInfo uniform_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(struct params),
      .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   check(vkCreateBuffer(device, &uniform_info, NULL, &uniform),
         "vkCreateBuffer(uniform)");
   VkMemoryRequirements uniform_requirements;
   vkGetBufferMemoryRequirements(device, uniform, &uniform_requirements);
   const VkMemoryAllocateInfo uniform_allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = uniform_requirements.size,
      .memoryTypeIndex = memory_type(&memory_properties,
                                     uniform_requirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory uniform_memory;
   check(vkAllocateMemory(device, &uniform_allocation, NULL, &uniform_memory),
         "vkAllocateMemory(uniform)");
   check(vkBindBufferMemory(device, uniform, uniform_memory, 0),
         "vkBindBufferMemory(uniform)");
   struct params *uniform_map;
   check(vkMapMemory(device, uniform_memory, 0, sizeof(struct params), 0,
                     (void **)&uniform_map),
         "vkMapMemory(uniform)");
   uniform_map->alpha_min = ALPHA_MIN;
   uniform_map->alpha_max = ALPHA_MAX;
   uniform_map->color_w = COLOR_W;
   vkUnmapMemory(device, uniform_memory);

   const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = 0.0f,
   };
   VkSampler sampler;
   check(vkCreateSampler(device, &sampler_info, NULL, &sampler),
         "vkCreateSampler");

   /* Binding 0 is the material's uniform block and binding 1 the atlas, the
    * same layout g17p-vulkan-graphics-probe uses, so the shim encode gate and
    * this oracle consume one identical SPIR-V module. */
   const VkDescriptorSetLayoutBinding bindings[2] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
   };
   const VkDescriptorSetLayoutCreateInfo set_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout set_layout;
   check(vkCreateDescriptorSetLayout(device, &set_layout_info, NULL,
                                     &set_layout),
         "vkCreateDescriptorSetLayout");
   const VkDescriptorPoolSize pool_sizes[2] = {
      {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1},
      {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1},
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool descriptor_pool;
   check(vkCreateDescriptorPool(device, &pool_info, NULL, &descriptor_pool),
         "vkCreateDescriptorPool");
   const VkDescriptorSetAllocateInfo set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkDescriptorSet descriptor_set;
   check(vkAllocateDescriptorSets(device, &set_info, &descriptor_set),
         "vkAllocateDescriptorSets");
   const VkDescriptorImageInfo descriptor_image = {
      .sampler = sampler,
      .imageView = atlas_view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   const VkDescriptorBufferInfo descriptor_uniform = {
      .buffer = uniform,
      .offset = 0,
      .range = sizeof(struct params),
   };
   const VkWriteDescriptorSet writes[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pBufferInfo = &descriptor_uniform,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &descriptor_image,
      },
   };
   vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

   const VkAttachmentDescription attachment = {
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
   };
   const VkAttachmentReference color = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color,
   };
   const VkSubpassDependency dependencies[2] = {
      {
         .srcSubpass = VK_SUBPASS_EXTERNAL,
         .dstSubpass = 0,
         .srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_SHADER_READ_BIT,
      },
      {
         .srcSubpass = 0,
         .dstSubpass = VK_SUBPASS_EXTERNAL,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      },
   };
   const VkRenderPassCreateInfo pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 2,
      .pDependencies = dependencies,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &pass_info, NULL, &render_pass),
         "vkCreateRenderPass");
   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = 1,
      .pAttachments = &target_view,
      .width = WIDTH,
      .height = HEIGHT,
      .layers = 1,
   };
   VkFramebuffer framebuffer;
   check(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer),
         "vkCreateFramebuffer");

   VkShaderModule vert_module, frag_module;
   const VkShaderModuleCreateInfo vert_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = vert_size,
      .pCode = vert_spv,
   };
   const VkShaderModuleCreateInfo frag_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = frag_size,
      .pCode = frag_spv,
   };
   check(vkCreateShaderModule(device, &vert_module_info, NULL, &vert_module),
         "vkCreateShaderModule(vert)");
   check(vkCreateShaderModule(device, &frag_module_info, NULL, &frag_module),
         "vkCreateShaderModule(frag)");

   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
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
   const VkRect2D scissor = {.extent = {WIDTH, HEIGHT}};
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
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
   /* Qt's own state for this material: the text colour is the blend constant,
    * and the destination is scaled by one minus the per-channel coverage. */
   const VkPipelineColorBlendAttachmentState blend_attachment = {
      .blendEnable = (mode == MODE_HIQ || mode == MODE_BLEND) ? VK_TRUE
                                                              : VK_FALSE,
      .srcColorBlendFactor = VK_BLEND_FACTOR_CONSTANT_COLOR,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_CONSTANT_ALPHA,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blend_attachment,
   };
   memcpy(blend.blendConstants, BLEND_CONSTANTS, sizeof(BLEND_CONSTANTS));

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
      .layout = pipeline_layout,
      .renderPass = render_pass,
      .subpass = 0,
   };
   VkPipeline pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipeline),
         "vkCreateGraphicsPipelines");
   printf(TEST_NAME ": pipeline created, blend=%s swizzle=%s\n",
          (mode == MODE_HIQ || mode == MODE_BLEND)
             ? "constant-color/one-minus-src-color"
             : "disabled",
          identity_swizzle ? "identity(.r)" : "zzz-r(.a)");

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

   VkImageMemoryBarrier to_transfer = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = atlas,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
                        &to_transfer);

   const VkBufferImageCopy copy = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {ATLAS_WIDTH, ATLAS_HEIGHT, 1},
   };
   vkCmdCopyBufferToImage(command_buffer, staging, atlas,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

   VkImageMemoryBarrier to_sampled = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = atlas,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0,
                        NULL, 1, &to_sampled);

   VkClearValue clear;
   for (int i = 0; i < 4; ++i)
      clear.color.float32[i] = CLEAR_BYTES[i] / 255.0f;
   const VkRenderPassBeginInfo pass_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {.extent = {WIDTH, HEIGHT}},
      .clearValueCount = 1,
      .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(command_buffer, &pass_begin,
                        VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
   vkCmdEndRenderPass(command_buffer);

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

   const int shift = negative ? 1 : 0;
   bool ok = verify(map, mode, shift, "immediate");
   if (ok) {
      sleep(3);
      ok = verify(map, mode, shift, "+3s");
   }

   if (negative) {
      if (ok) {
         fprintf(stderr,
                 TEST_NAME ": negative control PASSED the shifted reference; "
                           "the comparison is not sensitive to tap position\n");
         return EXIT_FAILURE;
      }
      printf(TEST_NAME ": negative control failed as required — the "
                       "comparison detects a one-texel tap shift\n");
      return EXIT_SUCCESS;
   }

   if (!ok)
      return EXIT_FAILURE;

   printf(TEST_NAME
          ": PASS — %s matched the host reference exactly (+/-1 LSB) "
          "immediately and after three seconds; readback guards unchanged\n",
          argv[3]);
   return EXIT_SUCCESS;
}
