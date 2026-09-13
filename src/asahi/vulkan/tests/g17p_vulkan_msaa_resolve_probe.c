/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
/*
 * Isolate the G17P multisample resolve from everything that feeds it.
 *
 * On hardware, 2x and 4x colour both resolve to exactly half brightness in
 * every channel. Two hypotheses survive that arithmetic at BOTH sample counts:
 *
 *   H-half  half the samples carry the fragment result and the other half read
 *           ZERO (not the clear -- the clear alpha would show), resolve
 *           divides by N;
 *   H-2N    every sample carries the result and the resolve divides by 2N.
 *
 * They are indistinguishable by any measurement that varies coverage, because
 * they are the same arithmetic. What separates the draw path from the resolve
 * path is a pass with NO DRAW AT ALL.
 *
 * Three cases, no geometry ambiguity anywhere -- every draw is a full-screen
 * triangle, so every pixel is completely covered and coverage cannot vary
 * within a pixel:
 *
 *   case 0  one sample, quad          -> must be exactly the draw colour.
 *                                        Proves the shader and the harness.
 *   case 1  N samples, CLEAR ONLY     -> must be exactly the clear colour.
 *                                        No fragment shader runs. If this
 *                                        halves, the entire draw path --
 *                                        rasterisation, coverage, sample mask,
 *                                        the fragment epilogue -- is exonerated
 *                                        in one measurement, and only the
 *                                        clear-to-tile-to-resolve chain is
 *                                        left.
 *   case 2  N samples, quad           -> must be exactly the draw colour.
 *
 * Each case prints its observed value next to what each named hypothesis
 * predicts, so the answer is read off rather than argued. Pair it with
 * AGX_SPECLM_SAMPLE_CODE: case 1 is the cheapest thing to sweep, because it
 * has no geometry, no shader and one correct answer.
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
#define TEST_NAME "g17p-vulkan-msaa-resolve2"
#else
#define TEST_NAME "g17p-vulkan-msaa-resolve4"
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

/* Must match g17p_vulkan_msaa_depth.frag: colour_index 0 is the draw colour. */
static const uint8_t draw_rgba[4] = {128, 64, 32, 16};
/* A clear that is non-zero in EVERY channel, including alpha, and distinct
 * from the draw colour in every channel. Both are multiples of four, so every
 * k-of-N mean is an exact integer at 2x and 4x. The alpha matters: it is what
 * separates "the untouched samples hold the clear" from "they hold zero". */
static const uint8_t clear_rgba[4] = {200, 160, 120, 80};

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


/* Name what a pixel value means, so the log reads as a verdict rather than as
 * four numbers. */
static void
describe(const char *label, const uint8_t *observed, unsigned samples,
         bool drew)
{
   const uint8_t *base = drew ? draw_rgba : clear_rgba;
   uint8_t exact[4], halved[4], with_clear[4];
   for (unsigned c = 0; c < 4; ++c) {
      exact[c] = base[c];
      halved[c] = (uint8_t)((base[c] * (samples / 2)) / samples);
      /* half written, the rest holding the clear */
      with_clear[c] = (uint8_t)(((unsigned)base[c] * (samples / 2) +
                                 (unsigned)clear_rgba[c] *
                                    (samples - samples / 2)) /
                                samples);
   }

   const char *verdict = "UNCLASSIFIED";
   if (!memcmp(observed, exact, 4))
      verdict = "correct";
   else if (samples == 1)
      verdict = "WRONG at one sample -- the harness itself is suspect";
   else if (!memcmp(observed, halved, 4))
      verdict = drew ? "HALVED (H-half with zeros, or H-2N)"
                     : "HALVED -- the resolve is wrong on its own";
   else if (drew && !memcmp(observed, with_clear, 4))
      verdict = "half written, untouched samples hold the CLEAR";

   if (samples == 1) {
      fprintf(stderr,
              TEST_NAME ": %-22s observed %02x %02x %02x %02x | correct "
                        "%02x %02x %02x %02x -> %s\n",
              label, observed[0], observed[1], observed[2], observed[3],
              exact[0], exact[1], exact[2], exact[3], verdict);
      return;
   }

   fprintf(stderr,
           TEST_NAME ": %-22s observed %02x %02x %02x %02x | correct "
                     "%02x %02x %02x %02x | halved %02x %02x %02x %02x -> %s\n",
           label, observed[0], observed[1], observed[2], observed[3], exact[0],
           exact[1], exact[2], exact[3], halved[0], halved[1], halved[2],
           halved[3], verdict);
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

   if ((properties.limits.framebufferColorSampleCounts &
        (VkSampleCountFlags)G17P_MSAA_SAMPLES) == 0) {
      fprintf(stderr,
              TEST_NAME ": framebufferColorSampleCounts lacks %ux "
                        "(set HK_G17P_MSAA=1)\n",
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

   /* One full-screen triangle. There is deliberately no partially covered
    * pixel anywhere in this test. */
   static const float vertices[] = {
      -1.0f, -1.0f, 3.0f, -1.0f, -1.0f, 3.0f,
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
   const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .maxDepthBounds = 1.0f,
   };

   /* case 0 is one sample with no resolve; cases 1 and 2 are N samples with a
    * resolve attachment. Case 1 issues no draw at all. */
   struct probe_case {
      const char *label;
      unsigned samples;
      bool draw;
   };
   const struct probe_case cases[] = {
      {"1x, quad", 1, true},
      {"Nx, CLEAR ONLY", G17P_MSAA_SAMPLES, false},
      {"Nx, quad", G17P_MSAA_SAMPLES, true},
   };
   const unsigned case_count = 3;

   VkBuffer readback_buffer[3];
   VkDeviceMemory readback_memory[3];
   uint8_t *readback_map[3];
   for (unsigned i = 0; i < case_count; ++i) {
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

   struct image msaa_image[3], target[3];
   VkRenderPass render_pass[3];
   VkFramebuffer framebuffer[3];
   VkPipeline pipeline[3];

   VkClearValue clears[2];
   memset(clears, 0, sizeof(clears));
   for (unsigned c = 0; c < 4; ++c)
      clears[0].color.float32[c] = (float)clear_rgba[c] / 255.0f;
   clears[1] = clears[0];

   for (unsigned i = 0; i < case_count; ++i) {
      const bool ms = cases[i].samples > 1;
      const VkSampleCountFlagBits s = (VkSampleCountFlagBits)cases[i].samples;

      msaa_image[i] = create_image(
         device, &memory_properties, COLOR_FORMAT, VK_IMAGE_ASPECT_COLOR_BIT,
         s,
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            (ms ? 0u : (unsigned)VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
         "colour attachment");
      if (ms) {
         target[i] = create_image(device, &memory_properties, COLOR_FORMAT,
                                  VK_IMAGE_ASPECT_COLOR_BIT,
                                  VK_SAMPLE_COUNT_1_BIT,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                  "resolve target");
      } else {
         target[i] = msaa_image[i];
      }

      const VkAttachmentDescription attachments[] = {
         {
            .format = COLOR_FORMAT,
            .samples = s,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = ms ? VK_ATTACHMENT_STORE_OP_DONT_CARE
                          : VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = ms ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                              : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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
      };
      const VkAttachmentReference colour_ref = {
         .attachment = 0,
         .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      };
      const VkAttachmentReference resolve_ref = {
         .attachment = 1,
         .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      };
      const VkSubpassDescription subpass = {
         .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
         .colorAttachmentCount = 1,
         .pColorAttachments = &colour_ref,
         .pResolveAttachments = ms ? &resolve_ref : NULL,
      };
      const VkRenderPassCreateInfo pass_info = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
         .attachmentCount = ms ? 2u : 1u,
         .pAttachments = attachments,
         .subpassCount = 1,
         .pSubpasses = &subpass,
      };
      check(vkCreateRenderPass(device, &pass_info, NULL, &render_pass[i]),
            "vkCreateRenderPass");

      const VkImageView views[] = {msaa_image[i].view, target[i].view};
      const VkFramebufferCreateInfo fb_info = {
         .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
         .renderPass = render_pass[i],
         .attachmentCount = ms ? 2u : 1u,
         .pAttachments = views,
         .width = WIDTH,
         .height = HEIGHT,
         .layers = 1,
      };
      check(vkCreateFramebuffer(device, &fb_info, NULL, &framebuffer[i]),
            "vkCreateFramebuffer");

      pipeline[i] = VK_NULL_HANDLE;
      if (cases[i].draw) {
         const float draw_depth = 0.0f;
         const int colour_index = 0;
         const VkSpecializationMapEntry vert_entry = {
            .constantID = 0, .offset = 0, .size = sizeof(float)};
         const VkSpecializationInfo vert_spec = {
            .mapEntryCount = 1,
            .pMapEntries = &vert_entry,
            .dataSize = sizeof(float),
            .pData = &draw_depth,
         };
         const VkSpecializationMapEntry frag_entry = {
            .constantID = 0, .offset = 0, .size = sizeof(int)};
         const VkSpecializationInfo frag_spec = {
            .mapEntryCount = 1,
            .pMapEntries = &frag_entry,
            .dataSize = sizeof(int),
            .pData = &colour_index,
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
         const VkPipelineMultisampleStateCreateInfo multisample = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = s,
            .minSampleShading = 1.0f,
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
            .renderPass = render_pass[i],
         };
         check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info,
                                         NULL, &pipeline[i]),
               "vkCreateGraphicsPipelines");
      }

      const VkRenderPassBeginInfo pass_begin = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = render_pass[i],
         .framebuffer = framebuffer[i],
         .renderArea = {{0, 0}, {WIDTH, HEIGHT}},
         .clearValueCount = ms ? 2u : 1u,
         .pClearValues = clears,
      };
      vkCmdBeginRenderPass(command_buffer, &pass_begin,
                           VK_SUBPASS_CONTENTS_INLINE);
      if (cases[i].draw) {
         const VkDeviceSize offset = 0;
         vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline[i]);
         vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer, &offset);
         vkCmdDraw(command_buffer, 3, 1, 0, 0);
      }
      vkCmdEndRenderPass(command_buffer);

      const VkBufferImageCopy copy = {
         .bufferOffset = GUARD_BYTES,
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {WIDTH, HEIGHT, 1},
      };
      vkCmdCopyImageToBuffer(command_buffer, target[i].image,
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
   for (unsigned i = 0; i < case_count; ++i) {
      for (unsigned g = 0; g < GUARD_BYTES; ++g) {
         if (readback_map[i][g] != GUARD_PATTERN ||
             readback_map[i][GUARD_BYTES + IMAGE_BYTES + g] !=
                GUARD_PATTERN) {
            fprintf(stderr, TEST_NAME ": case %u guard byte %u changed\n", i,
                    g);
            failures++;
            break;
         }
      }

      /* Every pixel is fully covered, so one value should occupy the whole
       * image. Say so if it does not, rather than reporting pixel 0 and
       * hiding the rest. */
      const uint8_t *pixels = readback_map[i] + GUARD_BYTES;
      unsigned distinct = 1;
      for (unsigned p = 1; p < WIDTH * HEIGHT; ++p) {
         if (memcmp(pixels + p * PIXEL_BYTES, pixels, PIXEL_BYTES)) {
            distinct++;
            if (distinct == 2) {
               fprintf(stderr,
                       TEST_NAME ": case %u (%s) is not uniform: pixel %u is "
                                 "%02x %02x %02x %02x against pixel 0 "
                                 "%02x %02x %02x %02x\n",
                       i, cases[i].label, p, pixels[p * PIXEL_BYTES],
                       pixels[p * PIXEL_BYTES + 1],
                       pixels[p * PIXEL_BYTES + 2],
                       pixels[p * PIXEL_BYTES + 3], pixels[0], pixels[1],
                       pixels[2], pixels[3]);
               failures++;
            }
         }
      }

      describe(cases[i].label, pixels, cases[i].samples, cases[i].draw);

      const uint8_t *want = cases[i].draw ? draw_rgba : clear_rgba;
      if (memcmp(pixels, want, PIXEL_BYTES))
         failures++;
   }

   fprintf(stderr, TEST_NAME ": %u failures\n", failures);

   vkDestroyFence(device, fence, NULL);
   vkDestroyCommandPool(device, command_pool, NULL);
   for (unsigned i = 0; i < case_count; ++i) {
      if (pipeline[i] != VK_NULL_HANDLE)
         vkDestroyPipeline(device, pipeline[i], NULL);
      vkDestroyFramebuffer(device, framebuffer[i], NULL);
      vkDestroyRenderPass(device, render_pass[i], NULL);
      if (target[i].image != msaa_image[i].image)
         destroy_image(device, &target[i]);
      destroy_image(device, &msaa_image[i]);
      vkUnmapMemory(device, readback_memory[i]);
      vkDestroyBuffer(device, readback_buffer[i], NULL);
      vkFreeMemory(device, readback_memory[i], NULL);
   }
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyShaderModule(device, frag_module, NULL);
   vkDestroyShaderModule(device, vert_module, NULL);
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
