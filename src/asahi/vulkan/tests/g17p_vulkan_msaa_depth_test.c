/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
/*
 * Qualify a multisample DEPTH attachment on G17P: per-sample depth testing,
 * and the ZLS store/load round trip that carries the depth buffer across two
 * render passes.
 *
 * Pass 1 draws a slanted triangle at depth 0.25 into a multisample colour +
 * multisample depth pair, then draws a FULL-SCREEN quad at depth 0.75 with
 * LESS. The quad covers every pixel completely, so coverage cannot vary
 * within a pixel; the only thing that can vary per sample is the depth test.
 * Both draws write depth. The colour attachment resolves to R1.
 *
 * Pass 2 reloads the same depth attachment (loadOp = LOAD) and draws one
 * full-screen quad at depth 0.5 with LESS and no depth write. Samples the
 * triangle owns (0.25) fail; samples the first quad owns (0.75) pass. The
 * colour attachment resolves to R2.
 *
 * Let k be the number of samples of a pixel that the triangle won in pass 1.
 * With B the triangle colour, A the quad colour, and pass 2 clearing to A and
 * drawing B:
 *
 *     R1 = (k*B + (N-k)*A) / N          R2 = ((N-k)*B + k*A) / N
 *
 * so for every pixel and every channel, exactly:
 *
 *     R1 + R2 == A + B
 *
 * That invariant only holds if the per-sample depth values survived the store
 * and the reload. Both colours are chosen so every one of the N+1 possible
 * means is an exact integer at 2x and 4x, so there is no rounding model and no
 * model of where the hardware puts its sample positions.
 *
 * The load-bearing check is that some pixel has 0 < k < N. A partial value can
 * only come from the per-sample depth test, because the second draw covers
 * every pixel fully -- with a per-pixel or resolved depth buffer every pixel
 * would be entirely A or entirely B and no intermediate value could exist.
 *
 * Needs HK_G17P_MSAA=1 and HK_G17P_UNVERIFIED_FEATURES=msaa_depth. The test
 * says so and exits non-zero rather than silently skipping.
 */
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "g17p_vulkan_msaa_depth_frag.h"
#include "g17p_vulkan_msaa_depth_vert.h"

#ifndef G17P_MSAA_SAMPLES
#define G17P_MSAA_SAMPLES 4
#endif

#if G17P_MSAA_SAMPLES != 2 && G17P_MSAA_SAMPLES != 4
#error "G17P_MSAA_SAMPLES must be 2 or 4"
#endif

#if G17P_MSAA_SAMPLES == 2
#define TEST_NAME "g17p-vulkan-msaa-depth2"
#else
#define TEST_NAME "g17p-vulkan-msaa-depth4"
#endif

#define WIDTH 64u
#define HEIGHT 64u
#define PIXEL_BYTES 4u
#define IMAGE_BYTES (WIDTH * HEIGHT * PIXEL_BYTES)
#define GUARD_BYTES 64u
#define READBACK_BYTES (GUARD_BYTES + IMAGE_BYTES + GUARD_BYTES)
#define GUARD_PATTERN 0xa5
#define COLOR_FORMAT VK_FORMAT_R8G8B8A8_UNORM
#define DEPTH_FORMAT VK_FORMAT_D32_SFLOAT

/* Must match g17p_vulkan_msaa_depth.frag. */
static const uint8_t colour_b[4] = {128, 64, 32, 16};
static const uint8_t colour_a[4] = {16, 32, 64, 128};
/* Distinct from every admissible mean of A and B, so a pixel nothing drew to
 * is detectable rather than aliasing onto a legal answer. */
static const uint8_t clear_rgba[4] = {255, 255, 255, 255};

#define VK_FN(name) static PFN_##name name
VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
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
VK_FN(vkCreateImageView);
VK_FN(vkDestroyImageView);
VK_FN(vkCreateShaderModule);
VK_FN(vkDestroyShaderModule);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkDestroyPipelineLayout);
VK_FN(vkCreateRenderPass);
VK_FN(vkDestroyRenderPass);
VK_FN(vkCreateFramebuffer);
VK_FN(vkDestroyFramebuffer);
VK_FN(vkCreateGraphicsPipelines);
VK_FN(vkDestroyPipeline);
VK_FN(vkCreateCommandPool);
VK_FN(vkDestroyCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdBeginRenderPass);
VK_FN(vkCmdEndRenderPass);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindVertexBuffers);
VK_FN(vkCmdDraw);
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
         die("loader is missing " #name);                                     \
   } while (0)
#define LOAD_INSTANCE(instance, name)                                         \
   do {                                                                       \
      name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);              \
      if (!name)                                                              \
         die("instance is missing " #name);                                   \
   } while (0)
#define LOAD_DEVICE(device, name)                                             \
   do {                                                                       \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                  \
      if (!name)                                                              \
         die("device is missing " #name);                                     \
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
   die("no suitable memory type");
   return 0;
}

struct image {
   VkImage image;
   VkDeviceMemory memory;
   VkImageView view;
};

static struct image
create_image(VkDevice device,
             const VkPhysicalDeviceMemoryProperties *memory_properties,
             VkFormat format, VkImageAspectFlags aspect,
             VkSampleCountFlagBits samples, VkImageUsageFlags usage,
             const char *what)
{
   struct image out = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {WIDTH, HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = samples,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   check(vkCreateImage(device, &info, NULL, &out.image), what);

   VkMemoryRequirements requirements;
   vkGetImageMemoryRequirements(device, out.image, &requirements);
   const VkMemoryAllocateInfo allocate = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex =
         find_memory_type(memory_properties, requirements.memoryTypeBits,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   check(vkAllocateMemory(device, &allocate, NULL, &out.memory), what);
   check(vkBindImageMemory(device, out.image, out.memory, 0), what);

   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = out.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = {aspect, 0, 1, 0, 1},
   };
   check(vkCreateImageView(device, &view_info, NULL, &out.view), what);
   return out;
}

static void
destroy_image(VkDevice device, struct image *image)
{
   vkDestroyImageView(device, image->view, NULL);
   vkDestroyImage(device, image->image, NULL);
   vkFreeMemory(device, image->memory, NULL);
}

/* Return k in 0..N if the pixel is exactly the k-sample mean of B and A, or
 * -1 if it is not an admissible answer at all. */
static int
coverage_of(const uint8_t *pixel)
{
   for (unsigned k = 0; k <= G17P_MSAA_SAMPLES; ++k) {
      bool match = true;
      for (unsigned c = 0; c < 4; ++c) {
         const unsigned sum =
            k * colour_b[c] + (G17P_MSAA_SAMPLES - k) * colour_a[c];
         if (sum % G17P_MSAA_SAMPLES || pixel[c] != sum / G17P_MSAA_SAMPLES) {
            match = false;
            break;
         }
      }
      if (match)
         return (int)k;
   }
   return -1;
}

int
main(void)
{
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      loader = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die("cannot load the Vulkan loader");
   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("loader is missing vkGetInstanceProcAddr");

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
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t device_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &device_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   if (!device_count)
      die("no physical device");
   VkPhysicalDevice physical_devices[8];
   if (device_count > 8)
      device_count = 8;
   check(vkEnumeratePhysicalDevices(instance, &device_count, physical_devices),
         "vkEnumeratePhysicalDevices");
   VkPhysicalDevice physical = physical_devices[0];

   VkPhysicalDeviceProperties properties;
   vkGetPhysicalDeviceProperties(physical, &properties);
   fprintf(stderr, TEST_NAME ": device %s\n", properties.deviceName);

   const VkSampleCountFlags want = (VkSampleCountFlags)G17P_MSAA_SAMPLES;
   if ((properties.limits.framebufferColorSampleCounts & want) == 0) {
      fprintf(stderr,
              TEST_NAME ": framebufferColorSampleCounts lacks %ux "
                        "(set HK_G17P_MSAA=1)\n",
              G17P_MSAA_SAMPLES);
      return EXIT_FAILURE;
   }
   if ((properties.limits.framebufferDepthSampleCounts & want) == 0) {
      fprintf(stderr,
              TEST_NAME ": framebufferDepthSampleCounts lacks %ux (set "
                        "HK_G17P_UNVERIFIED_FEATURES=msaa_depth)\n",
              G17P_MSAA_SAMPLES);
      return EXIT_FAILURE;
   }

   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   uint32_t family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, NULL);
   if (!family_count)
      die("no queue family");
   VkQueueFamilyProperties families[8];
   if (family_count > 8)
      family_count = 8;
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
   LOAD_DEVICE(device, vkCreateImageView);
   LOAD_DEVICE(device, vkDestroyImageView);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkDestroyShaderModule);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkDestroyPipelineLayout);
   LOAD_DEVICE(device, vkCreateRenderPass);
   LOAD_DEVICE(device, vkDestroyRenderPass);
   LOAD_DEVICE(device, vkCreateFramebuffer);
   LOAD_DEVICE(device, vkDestroyFramebuffer);
   LOAD_DEVICE(device, vkCreateGraphicsPipelines);
   LOAD_DEVICE(device, vkDestroyPipeline);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkDestroyCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdBeginRenderPass);
   LOAD_DEVICE(device, vkCmdEndRenderPass);
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdBindVertexBuffers);
   LOAD_DEVICE(device, vkCmdDraw);
   LOAD_DEVICE(device, vkCmdCopyImageToBuffer);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkDestroyFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);

   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);

   /* Slanted triangle first, then a full-screen triangle. */
   static const float vertices[] = {
      -0.90f, -0.85f, 0.88f, -0.80f, -0.05f, 0.90f,
      -1.0f,  -1.0f,  3.0f,  -1.0f,  -1.0f,  3.0f,
   };
   VkBuffer vertex_buffer;
   const VkBufferCreateInfo vertex_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sizeof(vertices),
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   check(vkCreateBuffer(device, &vertex_buffer_info, NULL, &vertex_buffer),
         "vkCreateBuffer(vertices)");
   VkMemoryRequirements vertex_requirements;
   vkGetBufferMemoryRequirements(device, vertex_buffer, &vertex_requirements);
   const VkMemoryAllocateInfo vertex_allocate = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = vertex_requirements.size,
      .memoryTypeIndex = find_memory_type(
         &memory_properties, vertex_requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory vertex_memory;
   check(vkAllocateMemory(device, &vertex_allocate, NULL, &vertex_memory),
         "vkAllocateMemory(vertices)");
   check(vkBindBufferMemory(device, vertex_buffer, vertex_memory, 0),
         "vkBindBufferMemory(vertices)");
   void *vertex_map = NULL;
   check(vkMapMemory(device, vertex_memory, 0, sizeof(vertices), 0,
                     &vertex_map),
         "vkMapMemory(vertices)");
   memcpy(vertex_map, vertices, sizeof(vertices));
   vkUnmapMemory(device, vertex_memory);

   VkBuffer readback_buffer[2];
   VkDeviceMemory readback_memory[2];
   uint8_t *readback_map[2];
   for (unsigned i = 0; i < 2; ++i) {
      const VkBufferCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = READBACK_BYTES,
         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      };
      check(vkCreateBuffer(device, &info, NULL, &readback_buffer[i]),
            "vkCreateBuffer(readback)");
      VkMemoryRequirements requirements;
      vkGetBufferMemoryRequirements(device, readback_buffer[i], &requirements);
      const VkMemoryAllocateInfo allocate = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = requirements.size,
         .memoryTypeIndex = find_memory_type(
            &memory_properties, requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      };
      check(vkAllocateMemory(device, &allocate, NULL, &readback_memory[i]),
            "vkAllocateMemory(readback)");
      check(vkBindBufferMemory(device, readback_buffer[i], readback_memory[i],
                               0),
            "vkBindBufferMemory(readback)");
      void *map = NULL;
      check(vkMapMemory(device, readback_memory[i], 0, READBACK_BYTES, 0,
                        &map),
            "vkMapMemory(readback)");
      readback_map[i] = map;
      memset(readback_map[i], GUARD_PATTERN, READBACK_BYTES);
   }

   struct image msaa_color[2];
   struct image resolved[2];
   for (unsigned i = 0; i < 2; ++i) {
      msaa_color[i] = create_image(
         device, &memory_properties, COLOR_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT,
         (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, "multisample colour");
      resolved[i] = create_image(
         device, &memory_properties, COLOR_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT,
         VK_SAMPLE_COUNT_1_BIT,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
         "resolve target");
   }
   /* One depth attachment, shared by both passes: pass 2 reloads what pass 1
    * stored, which is the whole point. */
   struct image depth = create_image(
      device, &memory_properties, DEPTH_FORMAT, VK_IMAGE_ASPECT_DEPTH_BIT,
      (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, "multisample depth");

   const VkAttachmentReference color_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkAttachmentReference resolve_ref = {
      .attachment = 1,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkAttachmentReference depth_ref = {
      .attachment = 2,
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
      .pResolveAttachments = &resolve_ref,
      .pDepthStencilAttachment = &depth_ref,
   };

   VkRenderPass render_pass[2];
   for (unsigned pass = 0; pass < 2; ++pass) {
      const VkAttachmentDescription attachments[] = {
         {
            .format = COLOR_FORMAT,
            .samples = (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         },
         {
            .format = COLOR_FORMAT,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         },
         {
            .format = DEPTH_FORMAT,
            .samples = (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
            .loadOp = pass == 0 ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                : VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = pass == 0
                                ? VK_IMAGE_LAYOUT_UNDEFINED
                                : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         },
      };
      const VkRenderPassCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
         .attachmentCount = 3,
         .pAttachments = attachments,
         .subpassCount = 1,
         .pSubpasses = &subpass,
      };
      check(vkCreateRenderPass(device, &info, NULL, &render_pass[pass]),
            "vkCreateRenderPass");
   }

   VkFramebuffer framebuffer[2];
   for (unsigned pass = 0; pass < 2; ++pass) {
      const VkImageView views[] = {msaa_color[pass].view, resolved[pass].view,
                                   depth.view};
      const VkFramebufferCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
         .renderPass = render_pass[pass],
         .attachmentCount = 3,
         .pAttachments = views,
         .width = WIDTH,
         .height = HEIGHT,
         .layers = 1,
      };
      check(vkCreateFramebuffer(device, &info, NULL, &framebuffer[pass]),
            "vkCreateFramebuffer");
   }

   VkShaderModule vert_module, frag_module;
   const VkShaderModuleCreateInfo vert_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_msaa_depth_vert_spv),
      .pCode = g17p_vulkan_msaa_depth_vert_spv,
   };
   const VkShaderModuleCreateInfo frag_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_msaa_depth_frag_spv),
      .pCode = g17p_vulkan_msaa_depth_frag_spv,
   };
   check(vkCreateShaderModule(device, &vert_module_info, NULL, &vert_module),
         "vkCreateShaderModule(vertex)");
   check(vkCreateShaderModule(device, &frag_module_info, NULL, &frag_module),
         "vkCreateShaderModule(fragment)");

   VkPipelineLayout layout;
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   check(vkCreatePipelineLayout(device, &layout_info, NULL, &layout),
         "vkCreatePipelineLayout");

   const VkVertexInputBindingDescription vertex_binding = {
      .binding = 0,
      .stride = 2 * sizeof(float),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription vertex_attribute = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
      .offset = 0,
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &vertex_attribute,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {0.0f, 0.0f, (float)WIDTH, (float)HEIGHT,
                                0.0f, 1.0f};
   const VkRect2D scissor = {{0, 0}, {WIDTH, HEIGHT}};
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
   const VkPipelineColorBlendAttachmentState blend_attachment = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blend_attachment,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
      .minSampleShading = 1.0f,
   };

   /* Three pipelines: the triangle at 0.25 writing depth, the pass-1
    * full-screen quad at 0.75 writing depth, and the pass-2 full-screen quad
    * at 0.5 that only reads it. */
   struct draw_config {
      float depth;
      int colour;
      VkBool32 depth_write;
      unsigned pass;
   };
   const struct draw_config draws[] = {
      {0.25f, 0, VK_TRUE, 0},
      {0.75f, 1, VK_TRUE, 0},
      {0.50f, 0, VK_FALSE, 1},
   };
   VkPipeline pipeline[3];
   for (unsigned d = 0; d < 3; ++d) {
      const VkSpecializationMapEntry vert_entry = {
         .constantID = 0, .offset = 0, .size = sizeof(float)};
      const VkSpecializationInfo vert_spec = {
         .mapEntryCount = 1,
         .pMapEntries = &vert_entry,
         .dataSize = sizeof(float),
         .pData = &draws[d].depth,
      };
      const VkSpecializationMapEntry frag_entry = {
         .constantID = 0, .offset = 0, .size = sizeof(int)};
      const VkSpecializationInfo frag_spec = {
         .mapEntryCount = 1,
         .pMapEntries = &frag_entry,
         .dataSize = sizeof(int),
         .pData = &draws[d].colour,
      };
      const VkPipelineShaderStageCreateInfo stages[] = {
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_module,
            .pName = "main",
            .pSpecializationInfo = &vert_spec,
         },
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag_module,
            .pName = "main",
            .pSpecializationInfo = &frag_spec,
         },
      };
      const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
         .depthTestEnable = VK_TRUE,
         .depthWriteEnable = draws[d].depth_write,
         .depthCompareOp = VK_COMPARE_OP_LESS,
         .maxDepthBounds = 1.0f,
      };
      const VkGraphicsPipelineCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .stageCount = 2,
         .pStages = stages,
         .pVertexInputState = &vertex_input,
         .pInputAssemblyState = &input_assembly,
         .pViewportState = &viewport_state,
         .pRasterizationState = &rasterization,
         .pMultisampleState = &multisample,
         .pDepthStencilState = &depth_stencil,
         .pColorBlendState = &blend,
         .layout = layout,
         .renderPass = render_pass[draws[d].pass],
      };
      check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, NULL,
                                      &pipeline[d]),
            "vkCreateGraphicsPipelines");
   }

   const VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VkCommandPool command_pool;
   check(vkCreateCommandPool(device, &command_pool_info, NULL, &command_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo command_buffer_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command_buffer;
   check(vkAllocateCommandBuffers(device, &command_buffer_info,
                                  &command_buffer),
         "vkAllocateCommandBuffers");
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command_buffer, &begin),
         "vkBeginCommandBuffer");

   VkClearValue clears[3];
   memset(clears, 0, sizeof(clears));
   for (unsigned c = 0; c < 4; ++c)
      clears[0].color.float32[c] = (float)clear_rgba[c] / 255.0f;
   clears[1] = clears[0];
   clears[2].depthStencil.depth = 1.0f;

   const VkDeviceSize triangle_offset = 0;
   const VkDeviceSize fullscreen_offset = 3 * 2 * sizeof(float);

   /* Record only the first N of the three draws. Two failures can stack here
    * -- a per-sample depth test that never varies, and a pass whose colour
    * never reaches its resolve target -- and they look the same in the final
    * invariant. Dropping draws separates them: with N=0 both passes are
    * clear-only, so pass 1 must resolve to the clear and nothing about depth
    * is involved yet. Diagnostic only; the default records all three and the
    * invariant below is unchanged. */
   unsigned max_draws = 3;
   {
      const char *limit = getenv("G17P_MSAA_DEPTH_MAX_DRAWS");
      if (limit) {
         max_draws = (unsigned)atoi(limit);
         if (max_draws > 3)
            max_draws = 3;
         fprintf(stderr, TEST_NAME ": recording %u of 3 draws\n", max_draws);
      }
   }

   for (unsigned pass = 0; pass < 2; ++pass) {
      if (pass == 1) {
         /* Pass 2 clears colour to A, so a sample whose depth test fails
          * keeps A and the invariant below stays exact. */
         for (unsigned c = 0; c < 4; ++c)
            clears[0].color.float32[c] = (float)colour_a[c] / 255.0f;
         clears[1] = clears[0];
      }
      const VkRenderPassBeginInfo pass_begin = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = render_pass[pass],
         .framebuffer = framebuffer[pass],
         .renderArea = {{0, 0}, {WIDTH, HEIGHT}},
         .clearValueCount = 3,
         .pClearValues = clears,
      };
      vkCmdBeginRenderPass(command_buffer, &pass_begin,
                           VK_SUBPASS_CONTENTS_INLINE);
      for (unsigned d = 0; d < max_draws; ++d) {
         if (draws[d].pass != pass)
            continue;
         const VkDeviceSize offset =
            (d == 0) ? triangle_offset : fullscreen_offset;
         vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline[d]);
         vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &offset);
         vkCmdDraw(command_buffer, 3, 1, 0, 0);
      }
      vkCmdEndRenderPass(command_buffer);
   }

   const VkBufferImageCopy copy = {
      .bufferOffset = GUARD_BYTES,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {WIDTH, HEIGHT, 1},
   };
   for (unsigned i = 0; i < 2; ++i) {
      vkCmdCopyImageToBuffer(command_buffer, resolved[i].image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readback_buffer[i], 1, &copy);
   }
   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence fence;
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
   };
   check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
   check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
         "vkWaitForFences");

   unsigned failures = 0;
   unsigned partial_pixels = 0;
   unsigned histogram[G17P_MSAA_SAMPLES + 1];
   memset(histogram, 0, sizeof(histogram));

   for (unsigned i = 0; i < 2; ++i) {
      for (unsigned g = 0; g < GUARD_BYTES; ++g) {
         if (readback_map[i][g] != GUARD_PATTERN ||
             readback_map[i][GUARD_BYTES + IMAGE_BYTES + g] !=
                GUARD_PATTERN) {
            fprintf(stderr, TEST_NAME ": readback %u guard byte %u changed\n",
                    i, g);
            failures++;
            break;
         }
      }
   }

   for (unsigned p = 0; p < WIDTH * HEIGHT; ++p) {
      const uint8_t *r1 = readback_map[0] + GUARD_BYTES + p * PIXEL_BYTES;
      const uint8_t *r2 = readback_map[1] + GUARD_BYTES + p * PIXEL_BYTES;
      const int k1 = coverage_of(r1);
      const int k2 = coverage_of(r2);

      if (k1 < 0 || k2 < 0) {
         if (failures < 8) {
            fprintf(stderr,
                    TEST_NAME ": pixel %u is not an N-sample mean of the two "
                              "colours: pass1 %02x %02x %02x %02x, pass2 "
                              "%02x %02x %02x %02x\n",
                    p, r1[0], r1[1], r1[2], r1[3], r2[0], r2[1], r2[2],
                    r2[3]);
         }
         failures++;
         continue;
      }

      histogram[k1]++;
      if (k1 > 0 && k1 < G17P_MSAA_SAMPLES)
         partial_pixels++;

      /* The store/load invariant: pass 2 must see exactly the complement of
       * the samples pass 1's triangle won. */
      if (k1 + k2 != G17P_MSAA_SAMPLES) {
         if (failures < 8) {
            fprintf(stderr,
                    TEST_NAME ": pixel %u: pass 1 kept %d samples and pass 2 "
                              "kept %d, which do not sum to %u -- the "
                              "multisample depth buffer did not survive the "
                              "store and reload\n",
                    p, k1, G17P_MSAA_SAMPLES - k2, G17P_MSAA_SAMPLES);
         }
         failures++;
      }

      for (unsigned c = 0; c < 4; ++c) {
         const unsigned expect = colour_a[c] + colour_b[c];
         if ((unsigned)r1[c] + (unsigned)r2[c] != expect) {
            if (failures < 8) {
               fprintf(stderr,
                       TEST_NAME ": pixel %u channel %u: %u + %u != %u\n", p,
                       c, r1[c], r2[c], expect);
            }
            failures++;
         }
      }
   }

   if (!partial_pixels) {
      fprintf(stderr,
              TEST_NAME ": no pixel has a partial sample count. The second "
                        "draw covers every pixel completely, so a partial "
                        "result can only come from a per-sample depth test -- "
                        "without one, nothing here was qualified\n");
      failures++;
   }
   if (!histogram[0] || !histogram[G17P_MSAA_SAMPLES]) {
      fprintf(stderr,
              TEST_NAME ": the triangle covered every pixel or none of them, "
                        "so the geometry missed its own edges\n");
      failures++;
   }

   fprintf(stderr, TEST_NAME ": %ux, coverage histogram", G17P_MSAA_SAMPLES);
   for (unsigned k = 0; k <= G17P_MSAA_SAMPLES; ++k)
      fprintf(stderr, " k=%u:%u", k, histogram[k]);
   fprintf(stderr, ", %u partial pixels, %u failures\n", partial_pixels,
           failures);

   vkDestroyFence(device, fence, NULL);
   vkDestroyCommandPool(device, command_pool, NULL);
   for (unsigned d = 0; d < 3; ++d)
      vkDestroyPipeline(device, pipeline[d], NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyShaderModule(device, frag_module, NULL);
   vkDestroyShaderModule(device, vert_module, NULL);
   for (unsigned i = 0; i < 2; ++i) {
      vkDestroyFramebuffer(device, framebuffer[i], NULL);
      vkDestroyRenderPass(device, render_pass[i], NULL);
      destroy_image(device, &resolved[i]);
      destroy_image(device, &msaa_color[i]);
      vkUnmapMemory(device, readback_memory[i]);
      vkDestroyBuffer(device, readback_buffer[i], NULL);
      vkFreeMemory(device, readback_memory[i], NULL);
   }
   destroy_image(device, &depth);
   vkDestroyBuffer(device, vertex_buffer, NULL);
   vkFreeMemory(device, vertex_memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);

   if (failures) {
      printf(TEST_NAME ": FAIL\n");
      return EXIT_FAILURE;
   }
   printf(TEST_NAME ": PASS\n");
   return EXIT_SUCCESS;
}
