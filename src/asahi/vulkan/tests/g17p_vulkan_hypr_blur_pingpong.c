/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact G17P render-attachment -> sampled-image blur discriminator.
 *
 * The first render pass clears an RGBA8 attachment to opaque green and stores
 * it. An explicit COLOR_ATTACHMENT_WRITE -> SHADER_READ dependency transitions
 * that same image for sampling. The supplied Hyprland blur shader renders it
 * into a second RGBA8 attachment with blending disabled. Every readback pixel
 * must remain exactly 00 ff 00 ff, including after a three-second residency
 * check. This isolates the boundary not covered by upload-backed texture tests.
 */

#include <vulkan/vulkan.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef TEST_NAME
#define TEST_NAME "f35-hyprland-blur-pingpong"
#endif
#define T8140_DEVICE_ID 0x8140
#define WIDTH 32
#define HEIGHT 32
#define GUARD 64
#define PIXEL_BYTES 4
#define IMAGE_BYTES (WIDTH * HEIGHT * PIXEL_BYTES)
#define READBACK_BYTES (IMAGE_BYTES + 2 * GUARD)

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
memory_type(const VkPhysicalDeviceMemoryProperties *properties,
            uint32_t bits, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }
   die("no compatible memory type");
   return 0;
}

struct image {
   VkImage image;
   VkDeviceMemory memory;
   VkImageView view;
};

static struct image
create_image(VkDevice device,
             const VkPhysicalDeviceMemoryProperties *properties,
             VkImageUsageFlags usage)
{
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {WIDTH, HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   struct image result = {0};
   check(vkCreateImage(device, &info, NULL, &result.image), "vkCreateImage");

   VkMemoryRequirements requirements;
   vkGetImageMemoryRequirements(device, result.image, &requirements);
   const VkMemoryAllocateInfo allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = memory_type(properties, requirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   check(vkAllocateMemory(device, &allocation, NULL, &result.memory),
         "vkAllocateMemory(image)");
   check(vkBindImageMemory(device, result.image, result.memory, 0),
         "vkBindImageMemory");

   const VkImageViewCreateInfo view = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = result.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1,
         .layerCount = 1,
      },
   };
   check(vkCreateImageView(device, &view, NULL, &result.view),
         "vkCreateImageView");
   return result;
}

static VkRenderPass
create_render_pass(VkDevice device, VkImageLayout final_layout,
                   VkPipelineStageFlags external_stage,
                   VkAccessFlags external_access)
{
   const VkAttachmentDescription attachment = {
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = final_layout,
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
         .srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      },
      {
         .srcSubpass = 0,
         .dstSubpass = VK_SUBPASS_EXTERNAL,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = external_stage,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = external_access,
      },
   };
   const VkRenderPassCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 2,
      .pDependencies = dependencies,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &info, NULL, &render_pass),
         "vkCreateRenderPass");
   return render_pass;
}

static VkFramebuffer
create_framebuffer(VkDevice device, VkRenderPass render_pass, VkImageView view)
{
   const VkFramebufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = 1,
      .pAttachments = &view,
      .width = WIDTH,
      .height = HEIGHT,
      .layers = 1,
   };
   VkFramebuffer framebuffer;
   check(vkCreateFramebuffer(device, &info, NULL, &framebuffer),
         "vkCreateFramebuffer");
   return framebuffer;
}

static bool
verify(const uint8_t *map, const char *region_mode)
{
   for (size_t i = 0; i < GUARD; ++i) {
      if (map[i] != 0xa5 || map[GUARD + IMAGE_BYTES + i] != 0xa5) {
         fprintf(stderr, TEST_NAME ": readback guard changed at byte %zu\n", i);
         return false;
      }
   }
#if F36_REGION
   struct probe {
      unsigned x, y;
      uint8_t expected[4];
   } probes[3];
   if (strcmp(region_mode, "down") == 0) {
      const struct probe down[3] = {
         {2, 8, {0xff, 0xff, 0xff, 0xff}},
         {13, 8, {0x00, 0x00, 0x00, 0xff}},
         {20, 8, {0xff, 0x00, 0xff, 0xff}},
      };
      memcpy(probes, down, sizeof(probes));
   } else {
      const struct probe up[3] = {
         {4, 8, {0xff, 0xff, 0xff, 0xff}},
         {20, 8, {0x00, 0x00, 0x00, 0xff}},
         {26, 8, {0x00, 0x00, 0x00, 0xff}},
      };
      memcpy(probes, up, sizeof(probes));
   }
   for (unsigned i = 0; i < 3; ++i) {
      const size_t pixel = probes[i].y * WIDTH + probes[i].x;
      const uint8_t *actual = map + GUARD + pixel * PIXEL_BYTES;
      if (memcmp(actual, probes[i].expected, PIXEL_BYTES) != 0) {
         fprintf(stderr, TEST_NAME ": %s probe (%u,%u) got %02x%02x%02x%02x, expected %02x%02x%02x%02x\n",
                 region_mode, probes[i].x, probes[i].y, actual[0], actual[1],
                 actual[2], actual[3], probes[i].expected[0],
                 probes[i].expected[1], probes[i].expected[2],
                 probes[i].expected[3]);
         return false;
      }
   }
#else
   (void)region_mode;
   const uint8_t expected[4] = {0x00, 0xff, 0x00, 0xff};
   for (size_t pixel = 0; pixel < WIDTH * HEIGHT; ++pixel) {
      const uint8_t *actual = map + GUARD + pixel * PIXEL_BYTES;
      if (memcmp(actual, expected, sizeof(expected)) != 0) {
         fprintf(stderr,
                 TEST_NAME ": pixel %zu differs: got %02x%02x%02x%02x, "
                           "expected 00ff00ff\n",
                 pixel, actual[0], actual[1], actual[2], actual[3]);
         return false;
      }
   }
#endif
   return true;
}

int
main(int argc, char **argv)
{
#if F36_REGION
   if (argc != 4 || (strcmp(argv[3], "down") != 0 &&
                     strcmp(argv[3], "up") != 0))
      die("usage: f36-hyprland-blur-region <vert.spv> <frag.spv> <down|up>");
   const char *region_mode = argv[3];
#else
   if (argc != 3)
      die("usage: f35-hyprland-blur-pingpong <vert.spv> <frag.spv>");
   const char *region_mode = NULL;
#endif

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
          (families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT |
                                     VK_QUEUE_TRANSFER_BIT)) ==
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

   struct image source = create_image(
      device, &memory_properties,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
   struct image destination = create_image(
      device, &memory_properties,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = READBACK_BYTES,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer readback;
   check(vkCreateBuffer(device, &buffer_info, NULL, &readback),
         "vkCreateBuffer");
   VkMemoryRequirements buffer_requirements;
   vkGetBufferMemoryRequirements(device, readback, &buffer_requirements);
   const VkMemoryAllocateInfo buffer_allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = buffer_requirements.size,
      .memoryTypeIndex = memory_type(
         &memory_properties, buffer_requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory readback_memory;
   check(vkAllocateMemory(device, &buffer_allocation, NULL, &readback_memory),
         "vkAllocateMemory(readback)");
   check(vkBindBufferMemory(device, readback, readback_memory, 0),
         "vkBindBufferMemory");
   uint8_t *map;
   check(vkMapMemory(device, readback_memory, 0, READBACK_BYTES, 0,
                     (void **)&map),
         "vkMapMemory");
   memset(map, 0xa5, READBACK_BYTES);

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
      .imageView = source.view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   const VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptor_set,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &descriptor_image,
   };
   vkUpdateDescriptorSets(device, 1, &write, 0, NULL);

   VkRenderPass source_pass = create_render_pass(
      device, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
   VkRenderPass destination_pass = create_render_pass(
      device, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
   VkFramebuffer source_fb =
      create_framebuffer(device, source_pass, source.view);
   VkFramebuffer destination_fb =
      create_framebuffer(device, destination_pass, destination.view);

   size_t vert_size, frag_size;
   uint32_t *vert_words = read_spv(argv[1], &vert_size);
   uint32_t *frag_words = read_spv(argv[2], &frag_size);
   const VkShaderModuleCreateInfo vert_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = vert_size,
      .pCode = vert_words,
   };
   const VkShaderModuleCreateInfo frag_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = frag_size,
      .pCode = frag_words,
   };
   VkShaderModule vert, frag;
   check(vkCreateShaderModule(device, &vert_info, NULL, &vert),
         "vkCreateShaderModule(vertex)");
   check(vkCreateShaderModule(device, &frag_info, NULL, &frag),
         "vkCreateShaderModule(fragment)");
   free(vert_words);
   free(frag_words);
   const VkPipelineLayoutCreateInfo pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkPipelineLayout pipeline_layout;
   check(vkCreatePipelineLayout(device, &pipeline_layout_info, NULL,
                                &pipeline_layout),
         "vkCreatePipelineLayout");
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vert,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = frag,
         .pName = "main",
      },
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   const VkPipelineInputAssemblyStateCreateInfo assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {
      .width = WIDTH,
      .height = HEIGHT,
      .maxDepth = 1.0f,
   };
   const VkRect2D scissor = {
      .extent = {
#if F36_REGION
         strcmp(region_mode, "down") == 0 ? WIDTH / 2 : WIDTH,
         strcmp(region_mode, "down") == 0 ? HEIGHT / 2 : HEIGHT,
#else
         WIDTH, HEIGHT,
#endif
      },
   };
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
   };
   const VkPipelineRasterizationStateCreateInfo raster = {
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
      .colorWriteMask = 0xf,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blend_attachment,
   };
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pColorBlendState = &blend,
      .layout = pipeline_layout,
      .renderPass = destination_pass,
   };
   VkPipeline pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipeline),
         "vkCreateGraphicsPipelines");

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
   VkCommandBuffer command;
   check(vkAllocateCommandBuffers(device, &command_info, &command),
         "vkAllocateCommandBuffers");
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer");

#if F36_REGION
   const VkClearValue source_clear = {
      .color = {.float32 = {1.0f, 0.0f, 0.0f, 1.0f}},
   };
#else
   const VkClearValue source_clear = {
      .color = {.float32 = {0.0f, 1.0f, 0.0f, 1.0f}},
   };
#endif
   const VkRenderPassBeginInfo source_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = source_pass,
      .framebuffer = source_fb,
      .renderArea = {.extent = {WIDTH, HEIGHT}},
      .clearValueCount = 1,
      .pClearValues = &source_clear,
   };
   vkCmdBeginRenderPass(command, &source_begin, VK_SUBPASS_CONTENTS_INLINE);
#if F36_REGION
   const VkClearAttachment clear_attachment = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .colorAttachment = 0,
      .clearValue = {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}},
   };
   VkClearRect black_rect = {
      .rect = {.extent = {WIDTH, HEIGHT}},
      .layerCount = 1,
   };
   if (strcmp(region_mode, "up") == 0) {
      black_rect.rect.extent.width = WIDTH / 2;
      black_rect.rect.extent.height = HEIGHT / 2;
   }
   vkCmdClearAttachments(command, 1, &clear_attachment, 1, &black_rect);
   VkClearAttachment white_attachment = clear_attachment;
   white_attachment.clearValue.color.float32[0] = 1.0f;
   white_attachment.clearValue.color.float32[1] = 1.0f;
   white_attachment.clearValue.color.float32[2] = 1.0f;
   VkClearRect white_rect = black_rect;
   white_rect.rect.extent.width = strcmp(region_mode, "down") == 0
                                    ? WIDTH / 2 : WIDTH / 4;
   vkCmdClearAttachments(command, 1, &white_attachment, 1, &white_rect);
#endif
   vkCmdEndRenderPass(command);

   const VkImageMemoryBarrier sampled = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = source.image,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1,
         .layerCount = 1,
      },
   };
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0,
                        NULL, 1, &sampled);

#if F36_REGION
   const VkClearValue black = {
      .color = {.float32 = {1.0f, 0.0f, 1.0f, 1.0f}},
   };
#else
   const VkClearValue black = {
      .color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}},
   };
#endif
   const VkRenderPassBeginInfo destination_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = destination_pass,
      .framebuffer = destination_fb,
      .renderArea = {.extent = {WIDTH, HEIGHT}},
      .clearValueCount = 1,
      .pClearValues = &black,
   };
   vkCmdBeginRenderPass(command, &destination_begin,
                        VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
   vkCmdDraw(command, 3, 1, 0, 0);
   vkCmdEndRenderPass(command);

   const VkBufferImageCopy copy = {
      .bufferOffset = GUARD,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .layerCount = 1,
      },
      .imageExtent = {WIDTH, HEIGHT, 1},
   };
   vkCmdCopyImageToBuffer(command, destination.image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
                          &copy);
   check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command,
   };
   check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
   check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
   if (!verify(map, region_mode))
      return EXIT_FAILURE;
   sleep(3);
   if (!verify(map, region_mode))
      return EXIT_FAILURE;

#if F36_REGION
   printf(TEST_NAME ": PASS — %s viewport/scissor/UV region probes exact immediately/3s; readback guards unchanged\n", region_mode);
#else
   printf(TEST_NAME ": PASS — stored RGBA8 attachment transitioned to "
                    "LINEAR sampled input and remained exact green "
                    "immediately/3s; readback guards unchanged\n");
#endif

   vkDestroyCommandPool(device, command_pool, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, pipeline_layout, NULL);
   vkDestroyShaderModule(device, frag, NULL);
   vkDestroyShaderModule(device, vert, NULL);
   vkDestroyFramebuffer(device, destination_fb, NULL);
   vkDestroyFramebuffer(device, source_fb, NULL);
   vkDestroyRenderPass(device, destination_pass, NULL);
   vkDestroyRenderPass(device, source_pass, NULL);
   vkDestroyDescriptorPool(device, descriptor_pool, NULL);
   vkDestroyDescriptorSetLayout(device, set_layout, NULL);
   vkDestroySampler(device, sampler, NULL);
   vkUnmapMemory(device, readback_memory);
   vkDestroyBuffer(device, readback, NULL);
   vkFreeMemory(device, readback_memory, NULL);
   vkDestroyImageView(device, destination.view, NULL);
   vkDestroyImage(device, destination.image, NULL);
   vkFreeMemory(device, destination.memory, NULL);
   vkDestroyImageView(device, source.view, NULL);
   vkDestroyImage(device, source.image, NULL);
   vkFreeMemory(device, source.memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return EXIT_SUCCESS;
}
