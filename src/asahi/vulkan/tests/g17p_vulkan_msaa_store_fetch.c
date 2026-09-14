/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
/*
 * Qualify storing a multisample colour attachment UNRESOLVED and reading the
 * individual samples back, on G17P.
 *
 * Two render passes draw the same triangle into two identical multisample
 * images. Pass A uses the already-hardware-qualified direct-resolve shape
 * (storeOp = DONT_CARE plus an AVERAGE resolve attachment) and keeps only the
 * resolved single-sample result. Pass B uses storeOp = STORE with no resolve
 * attachment, which is the path this oracle exists to qualify. N further
 * single-sample passes then texelFetch one sample index each out of pass B's
 * image.
 *
 * The checks are exact and need no model of where the hardware puts its
 * sample positions:
 *
 *   1. Every fetched sample is bit-exactly either the clear colour or the
 *      triangle colour. An implementation that silently resolved on store
 *      would hand back the averaged edge pixels instead, and fail here.
 *   2. The already-qualified resolved image equals the integer mean of the
 *      fetched samples, per channel, per pixel. The triangle colour is
 *      128/64/32/16 and the clear is 0, so every mean is exact at 2x and 4x.
 *   3. At least one pixel has samples that disagree, and the resolved image
 *      contains at least one value that is neither clear nor triangle. Without
 *      these the first two checks would pass vacuously on a 1x render.
 *   4. Guard bytes around both readbacks are untouched.
 *
 * A device without multisample colour attachments and sampled images fails
 * the test rather than silently skipping it.
 */
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "g17p_vulkan_msaa_fetch.h"
#include "g17p_vulkan_msaa_store.h"
#include "g17p_wsi_triangle_vert.h"

#ifndef G17P_MSAA_SAMPLES
#define G17P_MSAA_SAMPLES 4
#endif

#if G17P_MSAA_SAMPLES != 2 && G17P_MSAA_SAMPLES != 4
#error "G17P_MSAA_SAMPLES must be 2 or 4"
#endif

#if G17P_MSAA_SAMPLES == 2
#define TEST_NAME "g17p-vulkan-msaa-store-fetch2"
#else
#define TEST_NAME "g17p-vulkan-msaa-store-fetch4"
#endif

#define WIDTH 64u
#define HEIGHT 64u
#define PIXEL_BYTES 4u
#define IMAGE_BYTES (WIDTH * HEIGHT * PIXEL_BYTES)
#define GUARD_BYTES 64u
#define READBACK_BYTES (GUARD_BYTES + IMAGE_BYTES + GUARD_BYTES)
#define GUARD_PATTERN 0xa5
#define IMAGE_FORMAT VK_FORMAT_R8G8B8A8_UNORM

/* Must match g17p_vulkan_msaa_store.frag. */
static const uint8_t triangle_rgba[4] = {128, 64, 32, 16};
static const uint8_t clear_rgba[4] = {0, 0, 0, 0};

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
VK_FN(vkCreateSampler);
VK_FN(vkDestroySampler);
VK_FN(vkCreateDescriptorSetLayout);
VK_FN(vkDestroyDescriptorSetLayout);
VK_FN(vkCreateDescriptorPool);
VK_FN(vkDestroyDescriptorPool);
VK_FN(vkAllocateDescriptorSets);
VK_FN(vkUpdateDescriptorSets);
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
VK_FN(vkCmdBindDescriptorSets);
VK_FN(vkCmdDraw);
VK_FN(vkCmdPipelineBarrier);
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
             VkSampleCountFlagBits samples, VkImageUsageFlags usage,
             const char *what)
{
   struct image out = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = IMAGE_FORMAT,
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
      .format = IMAGE_FORMAT,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
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

/* Colours the oracle admits, and how it names them in a failure. */
static int
classify(const uint8_t *pixel)
{
   if (!memcmp(pixel, clear_rgba, 4))
      return 0;
   if (!memcmp(pixel, triangle_rgba, 4))
      return 1;
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

   const VkSampleCountFlags wanted =
      (VkSampleCountFlags)VK_SAMPLE_COUNT_1_BIT |
      (VkSampleCountFlags)G17P_MSAA_SAMPLES;
   if ((properties.limits.framebufferColorSampleCounts &
        (VkSampleCountFlags)G17P_MSAA_SAMPLES) == 0) {
      fprintf(stderr,
              TEST_NAME ": framebufferColorSampleCounts lacks %ux\n",
              G17P_MSAA_SAMPLES);
      return EXIT_FAILURE;
   }
   if ((properties.limits.sampledImageColorSampleCounts &
        (VkSampleCountFlags)G17P_MSAA_SAMPLES) == 0) {
      fprintf(stderr,
              TEST_NAME ": sampledImageColorSampleCounts lacks %ux (set "
                        "HK_G17P_UNVERIFIED_FEATURES=msaa_store)\n",
              G17P_MSAA_SAMPLES);
      return EXIT_FAILURE;
   }
   (void)wanted;

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
   LOAD_DEVICE(device, vkCreateSampler);
   LOAD_DEVICE(device, vkDestroySampler);
   LOAD_DEVICE(device, vkCreateDescriptorSetLayout);
   LOAD_DEVICE(device, vkDestroyDescriptorSetLayout);
   LOAD_DEVICE(device, vkCreateDescriptorPool);
   LOAD_DEVICE(device, vkDestroyDescriptorPool);
   LOAD_DEVICE(device, vkAllocateDescriptorSets);
   LOAD_DEVICE(device, vkUpdateDescriptorSets);
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
   LOAD_DEVICE(device, vkCmdBindDescriptorSets);
   LOAD_DEVICE(device, vkCmdDraw);
   LOAD_DEVICE(device, vkCmdPipelineBarrier);
   LOAD_DEVICE(device, vkCmdCopyImageToBuffer);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkDestroyFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);

   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);

   /* Vertices: a slanted triangle for pass A/B, then a full-screen triangle
    * for the fetch passes. Slanted edges guarantee partially covered pixels,
    * which is the only place an unresolved store differs from a resolved one.
    */
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

   /* One readback buffer per output image: the resolved control and one per
    * fetched sample index. */
   const unsigned readbacks = 1 + G17P_MSAA_SAMPLES;
   VkBuffer readback_buffer[1 + G17P_MSAA_SAMPLES];
   VkDeviceMemory readback_memory[1 + G17P_MSAA_SAMPLES];
   uint8_t *readback_map[1 + G17P_MSAA_SAMPLES];
   for (unsigned i = 0; i < readbacks; ++i) {
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

   struct image control_msaa = create_image(
      device, &memory_properties, (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, "control multisample image");
   struct image resolved = create_image(
      device, &memory_properties, VK_SAMPLE_COUNT_1_BIT,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      "resolve target");
   struct image stored_msaa = create_image(
      device, &memory_properties, (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      "stored multisample image");
   struct image fetched[G17P_MSAA_SAMPLES];
   for (unsigned s = 0; s < G17P_MSAA_SAMPLES; ++s) {
      fetched[s] = create_image(device, &memory_properties,
                                VK_SAMPLE_COUNT_1_BIT,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                "fetch target");
   }

   /* Pass A: multisample colour, DONT_CARE, with an AVERAGE resolve. */
   const VkAttachmentDescription resolve_attachments[] = {
      {
         .format = IMAGE_FORMAT,
         .samples = (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      },
      {
         .format = IMAGE_FORMAT,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      },
   };
   /* Pass B: multisample colour, STORE, no resolve. */
   const VkAttachmentDescription store_attachment = {
      .format = IMAGE_FORMAT,
      .samples = (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   /* Fetch passes: ordinary single-sample colour. */
   const VkAttachmentDescription fetch_attachment = {
      .format = IMAGE_FORMAT,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
   };

   const VkAttachmentReference color_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkAttachmentReference resolve_ref = {
      .attachment = 1,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };

   VkRenderPass render_pass[3];
   {
      const VkSubpassDescription subpass = {
         .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
         .colorAttachmentCount = 1,
         .pColorAttachments = &color_ref,
         .pResolveAttachments = &resolve_ref,
      };
      const VkRenderPassCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
         .attachmentCount = 2,
         .pAttachments = resolve_attachments,
         .subpassCount = 1,
         .pSubpasses = &subpass,
      };
      check(vkCreateRenderPass(device, &info, NULL, &render_pass[0]),
            "vkCreateRenderPass(resolve)");
   }
   {
      const VkSubpassDescription subpass = {
         .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
         .colorAttachmentCount = 1,
         .pColorAttachments = &color_ref,
      };
      const VkRenderPassCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
         .attachmentCount = 1,
         .pAttachments = &store_attachment,
         .subpassCount = 1,
         .pSubpasses = &subpass,
      };
      check(vkCreateRenderPass(device, &info, NULL, &render_pass[1]),
            "vkCreateRenderPass(store)");
      const VkRenderPassCreateInfo fetch_info = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
         .attachmentCount = 1,
         .pAttachments = &fetch_attachment,
         .subpassCount = 1,
         .pSubpasses = &subpass,
      };
      check(vkCreateRenderPass(device, &fetch_info, NULL, &render_pass[2]),
            "vkCreateRenderPass(fetch)");
   }

   VkFramebuffer framebuffer[2 + G17P_MSAA_SAMPLES];
   {
      const VkImageView resolve_views[] = {control_msaa.view, resolved.view};
      const VkFramebufferCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
         .renderPass = render_pass[0],
         .attachmentCount = 2,
         .pAttachments = resolve_views,
         .width = WIDTH,
         .height = HEIGHT,
         .layers = 1,
      };
      check(vkCreateFramebuffer(device, &info, NULL, &framebuffer[0]),
            "vkCreateFramebuffer(resolve)");
      const VkFramebufferCreateInfo store_fb = {
         .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
         .renderPass = render_pass[1],
         .attachmentCount = 1,
         .pAttachments = &stored_msaa.view,
         .width = WIDTH,
         .height = HEIGHT,
         .layers = 1,
      };
      check(vkCreateFramebuffer(device, &store_fb, NULL, &framebuffer[1]),
            "vkCreateFramebuffer(store)");
      for (unsigned s = 0; s < G17P_MSAA_SAMPLES; ++s) {
         const VkFramebufferCreateInfo fetch_fb = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass[2],
            .attachmentCount = 1,
            .pAttachments = &fetched[s].view,
            .width = WIDTH,
            .height = HEIGHT,
            .layers = 1,
         };
         check(vkCreateFramebuffer(device, &fetch_fb, NULL,
                                   &framebuffer[2 + s]),
               "vkCreateFramebuffer(fetch)");
      }
   }

   VkSampler sampler;
   const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = 0.0f,
      .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
   };
   check(vkCreateSampler(device, &sampler_info, NULL, &sampler),
         "vkCreateSampler");

   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo set_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   VkDescriptorSetLayout set_layout;
   check(vkCreateDescriptorSetLayout(device, &set_layout_info, NULL,
                                     &set_layout),
         "vkCreateDescriptorSetLayout");
   const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .descriptorCount = 1,
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
   };
   VkDescriptorPool descriptor_pool;
   check(vkCreateDescriptorPool(device, &pool_info, NULL, &descriptor_pool),
         "vkCreateDescriptorPool");
   const VkDescriptorSetAllocateInfo set_allocate = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkDescriptorSet descriptor_set;
   check(vkAllocateDescriptorSets(device, &set_allocate, &descriptor_set),
         "vkAllocateDescriptorSets");
   const VkDescriptorImageInfo descriptor_image = {
      .sampler = sampler,
      .imageView = stored_msaa.view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   const VkWriteDescriptorSet descriptor_write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &descriptor_image,
   };
   vkUpdateDescriptorSets(device, 1, &descriptor_write, 0, NULL);

   VkShaderModule vert_module, store_module, fetch_module;
   const VkShaderModuleCreateInfo vert_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_wsi_triangle_vert_spv),
      .pCode = g17p_wsi_triangle_vert_spv,
   };
   const VkShaderModuleCreateInfo store_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_msaa_store_spv),
      .pCode = g17p_vulkan_msaa_store_spv,
   };
   const VkShaderModuleCreateInfo fetch_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(g17p_vulkan_msaa_fetch_spv),
      .pCode = g17p_vulkan_msaa_fetch_spv,
   };
   check(vkCreateShaderModule(device, &vert_module_info, NULL, &vert_module),
         "vkCreateShaderModule(vertex)");
   check(vkCreateShaderModule(device, &store_module_info, NULL,
                              &store_module),
         "vkCreateShaderModule(store)");
   check(vkCreateShaderModule(device, &fetch_module_info, NULL,
                              &fetch_module),
         "vkCreateShaderModule(fetch)");

   VkPipelineLayout empty_layout, fetch_layout;
   const VkPipelineLayoutCreateInfo empty_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   check(vkCreatePipelineLayout(device, &empty_layout_info, NULL,
                                &empty_layout),
         "vkCreatePipelineLayout(empty)");
   const VkPipelineLayoutCreateInfo fetch_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
   };
   check(vkCreatePipelineLayout(device, &fetch_layout_info, NULL,
                                &fetch_layout),
         "vkCreatePipelineLayout(fetch)");

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
   };
   const VkPipelineMultisampleStateCreateInfo multisample_msaa = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = (VkSampleCountFlagBits)G17P_MSAA_SAMPLES,
      .minSampleShading = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo multisample_1x = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .minSampleShading = 1.0f,
   };

   VkPipeline pipeline[2 + G17P_MSAA_SAMPLES];
   {
      const VkPipelineShaderStageCreateInfo stages[] = {
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_module,
            .pName = "main",
         },
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = store_module,
            .pName = "main",
         },
      };
      VkGraphicsPipelineCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .stageCount = 2,
         .pStages = stages,
         .pVertexInputState = &vertex_input,
         .pInputAssemblyState = &input_assembly,
         .pViewportState = &viewport_state,
         .pRasterizationState = &rasterization,
         .pMultisampleState = &multisample_msaa,
         .pDepthStencilState = &depth_stencil,
         .pColorBlendState = &blend,
         .layout = empty_layout,
         .renderPass = render_pass[0],
      };
      check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, NULL,
                                      &pipeline[0]),
            "vkCreateGraphicsPipelines(resolve)");
      info.renderPass = render_pass[1];
      check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, NULL,
                                      &pipeline[1]),
            "vkCreateGraphicsPipelines(store)");
   }
   for (unsigned s = 0; s < G17P_MSAA_SAMPLES; ++s) {
      const uint32_t sample_index = s;
      const VkSpecializationMapEntry entry = {
         .constantID = 0,
         .offset = 0,
         .size = sizeof(sample_index),
      };
      const VkSpecializationInfo specialization = {
         .mapEntryCount = 1,
         .pMapEntries = &entry,
         .dataSize = sizeof(sample_index),
         .pData = &sample_index,
      };
      const VkPipelineShaderStageCreateInfo stages[] = {
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_module,
            .pName = "main",
         },
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fetch_module,
            .pName = "main",
            .pSpecializationInfo = &specialization,
         },
      };
      const VkGraphicsPipelineCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .stageCount = 2,
         .pStages = stages,
         .pVertexInputState = &vertex_input,
         .pInputAssemblyState = &input_assembly,
         .pViewportState = &viewport_state,
         .pRasterizationState = &rasterization,
         .pMultisampleState = &multisample_1x,
         .pDepthStencilState = &depth_stencil,
         .pColorBlendState = &blend,
         .layout = fetch_layout,
         .renderPass = render_pass[2],
      };
      check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, NULL,
                                      &pipeline[2 + s]),
            "vkCreateGraphicsPipelines(fetch)");
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

   VkClearValue clears[2];
   memset(clears, 0, sizeof(clears));
   for (unsigned c = 0; c < 4; ++c)
      clears[0].color.float32[c] = (float)clear_rgba[c] / 255.0f;
   clears[1] = clears[0];

   const VkDeviceSize vertex_offset = 0;
   VkRenderPassBeginInfo pass_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass[0],
      .framebuffer = framebuffer[0],
      .renderArea = {{0, 0}, {WIDTH, HEIGHT}},
      .clearValueCount = 2,
      .pClearValues = clears,
   };
   vkCmdBeginRenderPass(command_buffer, &pass_begin,
                        VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                     pipeline[0]);
   vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer,
                          &vertex_offset);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
   vkCmdEndRenderPass(command_buffer);

   pass_begin.renderPass = render_pass[1];
   pass_begin.framebuffer = framebuffer[1];
   pass_begin.clearValueCount = 1;
   vkCmdBeginRenderPass(command_buffer, &pass_begin,
                        VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                     pipeline[1]);
   vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer,
                          &vertex_offset);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
   vkCmdEndRenderPass(command_buffer);

   const VkImageMemoryBarrier sample_barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = stored_msaa.image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   vkCmdPipelineBarrier(command_buffer,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0,
                        NULL, 1, &sample_barrier);

   const VkDeviceSize fullscreen_offset = 3 * 2 * sizeof(float);
   for (unsigned s = 0; s < G17P_MSAA_SAMPLES; ++s) {
      pass_begin.renderPass = render_pass[2];
      pass_begin.framebuffer = framebuffer[2 + s];
      vkCmdBeginRenderPass(command_buffer, &pass_begin,
                           VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pipeline[2 + s]);
      vkCmdBindDescriptorSets(command_buffer,
                              VK_PIPELINE_BIND_POINT_GRAPHICS, fetch_layout,
                              0, 1, &descriptor_set, 0, NULL);
      vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer,
                             &fullscreen_offset);
      vkCmdDraw(command_buffer, 3, 1, 0, 0);
      vkCmdEndRenderPass(command_buffer);
   }

   const VkBufferImageCopy copy = {
      .bufferOffset = GUARD_BYTES,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {WIDTH, HEIGHT, 1},
   };
   vkCmdCopyImageToBuffer(command_buffer, resolved.image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback_buffer[0], 1, &copy);
   for (unsigned s = 0; s < G17P_MSAA_SAMPLES; ++s) {
      vkCmdCopyImageToBuffer(command_buffer, fetched[s].image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readback_buffer[1 + s], 1, &copy);
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
   unsigned disagreeing_pixels = 0;
   unsigned intermediate_resolved = 0;

   for (unsigned i = 0; i < readbacks; ++i) {
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

   const uint8_t *resolved_pixels = readback_map[0] + GUARD_BYTES;
   for (unsigned p = 0; p < WIDTH * HEIGHT; ++p) {
      unsigned sum[4] = {0, 0, 0, 0};
      bool disagree = false;
      for (unsigned s = 0; s < G17P_MSAA_SAMPLES; ++s) {
         const uint8_t *pixel =
            readback_map[1 + s] + GUARD_BYTES + p * PIXEL_BYTES;
         const int kind = classify(pixel);
         if (kind < 0) {
            if (failures < 8) {
               fprintf(stderr,
                       TEST_NAME ": pixel %u sample %u is %02x %02x %02x %02x,"
                                 " neither clear nor triangle -- the store"
                                 " resolved instead of keeping samples\n",
                       p, s, pixel[0], pixel[1], pixel[2], pixel[3]);
            }
            failures++;
         }
         for (unsigned c = 0; c < 4; ++c)
            sum[c] += pixel[c];
         if (s && memcmp(pixel, readback_map[1] + GUARD_BYTES +
                                   p * PIXEL_BYTES,
                         PIXEL_BYTES))
            disagree = true;
      }
      if (disagree)
         disagreeing_pixels++;

      const uint8_t *reference = resolved_pixels + p * PIXEL_BYTES;
      if (classify(reference) < 0)
         intermediate_resolved++;
      for (unsigned c = 0; c < 4; ++c) {
         const unsigned mean = sum[c] / G17P_MSAA_SAMPLES;
         if (sum[c] % G17P_MSAA_SAMPLES) {
            fprintf(stderr,
                    TEST_NAME ": pixel %u channel %u sample sum %u is not a"
                              " multiple of %u -- oracle colours are wrong\n",
                    p, c, sum[c], G17P_MSAA_SAMPLES);
            failures++;
         } else if (reference[c] != mean) {
            if (failures < 8) {
               fprintf(stderr,
                       TEST_NAME ": pixel %u channel %u: resolved %u, mean of"
                                 " fetched samples %u\n",
                       p, c, reference[c], mean);
            }
            failures++;
         }
      }
   }

   if (!disagreeing_pixels) {
      fprintf(stderr, TEST_NAME
              ": no pixel has samples that disagree -- the render was"
              " effectively single-sampled, so nothing was qualified\n");
      failures++;
   }
   if (!intermediate_resolved) {
      fprintf(stderr, TEST_NAME
              ": the resolved control image has no partially covered pixel --"
              " the geometry missed every edge, so nothing was qualified\n");
      failures++;
   }

   fprintf(stderr,
           TEST_NAME ": %ux, %u pixels with disagreeing samples, %u partially"
                     " covered resolved pixels, %u failures\n",
           G17P_MSAA_SAMPLES, disagreeing_pixels, intermediate_resolved,
           failures);

   vkDestroyFence(device, fence, NULL);
   vkDestroyCommandPool(device, command_pool, NULL);
   for (unsigned i = 0; i < 2 + G17P_MSAA_SAMPLES; ++i) {
      vkDestroyPipeline(device, pipeline[i], NULL);
      vkDestroyFramebuffer(device, framebuffer[i], NULL);
   }
   vkDestroyPipelineLayout(device, fetch_layout, NULL);
   vkDestroyPipelineLayout(device, empty_layout, NULL);
   vkDestroyShaderModule(device, fetch_module, NULL);
   vkDestroyShaderModule(device, store_module, NULL);
   vkDestroyShaderModule(device, vert_module, NULL);
   vkDestroyDescriptorPool(device, descriptor_pool, NULL);
   vkDestroyDescriptorSetLayout(device, set_layout, NULL);
   vkDestroySampler(device, sampler, NULL);
   for (unsigned i = 0; i < 3; ++i)
      vkDestroyRenderPass(device, render_pass[i], NULL);
   for (unsigned s = 0; s < G17P_MSAA_SAMPLES; ++s)
      destroy_image(device, &fetched[s]);
   destroy_image(device, &stored_msaa);
   destroy_image(device, &resolved);
   destroy_image(device, &control_msaa);
   for (unsigned i = 0; i < readbacks; ++i) {
      vkUnmapMemory(device, readback_memory[i]);
      vkDestroyBuffer(device, readback_buffer[i], NULL);
      vkFreeMemory(device, readback_memory[i], NULL);
   }
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
