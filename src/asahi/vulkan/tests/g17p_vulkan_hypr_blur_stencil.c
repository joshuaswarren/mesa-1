/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Hyprland popup/layer blur stencil oracle for T8140/G17P.
 */

#define TEST_NAME "f37-hyprland-blur-stencil"
#define main f35_unused_main
#include "g17p_vulkan_hypr_blur_pingpong.c"
#undef main

static VkShaderModule
shader_module(VkDevice device, const char *path)
{
   size_t size;
   uint32_t *words = read_spv(path, &size);
   const VkShaderModuleCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = words,
   };
   VkShaderModule module;
   check(vkCreateShaderModule(device, &info, NULL, &module),
         "vkCreateShaderModule");
   free(words);
   return module;
}

static VkPipeline
make_pipeline(VkDevice device, VkRenderPass render_pass,
              VkPipelineLayout layout, VkShaderModule vertex,
              VkShaderModule fragment, bool color_write, bool stencil_write,
              bool stencil_equal, bool blend_enable)
{
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = vertex,
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = fragment,
       .pName = "main"},
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   const VkPipelineInputAssemblyStateCreateInfo assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {
      .width = WIDTH, .height = HEIGHT, .maxDepth = 1.0f,
   };
   const VkRect2D scissor = {.extent = {WIDTH, HEIGHT}};
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
   const VkStencilOpState stencil = {
      .failOp = VK_STENCIL_OP_KEEP,
      .passOp = stencil_write ? VK_STENCIL_OP_REPLACE : VK_STENCIL_OP_KEEP,
      .depthFailOp = VK_STENCIL_OP_KEEP,
      .compareOp = stencil_equal ? VK_COMPARE_OP_EQUAL : VK_COMPARE_OP_ALWAYS,
      .compareMask = 0xff,
      .writeMask = stencil_write ? 0xff : 0,
      .reference = 1,
   };
   const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .stencilTestEnable = (stencil_write || stencil_equal) ? VK_TRUE : VK_FALSE,
      .front = stencil,
      .back = stencil,
      .maxDepthBounds = 1.0f,
   };
   const VkPipelineColorBlendAttachmentState blend_attachment = {
      .blendEnable = blend_enable,
      .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      .alphaBlendOp = VK_BLEND_OP_ADD,
      .colorWriteMask = color_write ? 0xf : 0,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blend_attachment,
   };
   const VkGraphicsPipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pDepthStencilState = &depth_stencil,
      .pColorBlendState = &blend,
      .layout = layout,
      .renderPass = render_pass,
   };
   VkPipeline pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, NULL,
                                   &pipeline),
         "vkCreateGraphicsPipelines");
   return pipeline;
}

int
main(int argc, char **argv)
{
   if (argc != 5)
      die("usage: f37-hyprland-blur-stencil <vert.spv> <mask.spv> <bg.spv> <client.spv>");

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
   VkPhysicalDevice *devices = calloc(physical_count, sizeof(*devices));
   if (!devices)
      die("out of host memory");
   check(vkEnumeratePhysicalDevices(instance, &physical_count, devices),
         "vkEnumeratePhysicalDevices(list)");
   VkPhysicalDevice physical = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < physical_count; ++i) {
      VkPhysicalDeviceProperties candidate;
      vkGetPhysicalDeviceProperties(devices[i], &candidate);
      if (candidate.deviceID == T8140_DEVICE_ID) {
         physical = devices[i];
         break;
      }
   }
   free(devices);
   if (!physical)
      die("T8140/G17P Vulkan device was not found");

   uint32_t family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, NULL);
   VkQueueFamilyProperties *families = calloc(family_count, sizeof(*families));
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
      die("no graphics-and-transfer queue");
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
   VkPhysicalDeviceMemoryProperties properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &properties);

   struct image color = create_image(
      device, &properties,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
   const VkImageCreateInfo stencil_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      /* Zink maps Hyprland's GL_DEPTH24_STENCIL8 workbuffer attachment to
       * Honeykrisp's supported packed depth/stencil format. */
      .format = VK_FORMAT_D32_SFLOAT_S8_UINT,
      .extent = {WIDTH, HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage stencil_image;
   check(vkCreateImage(device, &stencil_info, NULL, &stencil_image),
         "vkCreateImage(stencil)");
   VkMemoryRequirements stencil_requirements;
   vkGetImageMemoryRequirements(device, stencil_image, &stencil_requirements);
   const VkMemoryAllocateInfo stencil_allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = stencil_requirements.size,
      .memoryTypeIndex = memory_type(&properties,
                                     stencil_requirements.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory stencil_memory;
   check(vkAllocateMemory(device, &stencil_allocation, NULL, &stencil_memory),
         "vkAllocateMemory(stencil)");
   check(vkBindImageMemory(device, stencil_image, stencil_memory, 0),
         "vkBindImageMemory(stencil)");
   const VkImageViewCreateInfo stencil_view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = stencil_image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_D32_SFLOAT_S8_UINT,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
         .levelCount = 1,
         .layerCount = 1,
      },
   };
   VkImageView stencil_view;
   check(vkCreateImageView(device, &stencil_view_info, NULL, &stencil_view),
         "vkCreateImageView(stencil)");

   const VkAttachmentDescription attachments[2] = {
      {
         .format = VK_FORMAT_R8G8B8A8_UNORM,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      },
      {
         .format = VK_FORMAT_D32_SFLOAT_S8_UINT,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      },
   };
   const VkAttachmentReference color_ref = {
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkAttachmentReference stencil_ref = {
      .attachment = 1,
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
      .pDepthStencilAttachment = &stencil_ref,
   };
   const VkSubpassDependency dependencies[2] = {
      {.srcSubpass = VK_SUBPASS_EXTERNAL,
       .dstSubpass = 0,
       .srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
       .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
       .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT},
      {.srcSubpass = 0,
       .dstSubpass = VK_SUBPASS_EXTERNAL,
       .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
       .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
       .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
       .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT},
   };
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 2,
      .pAttachments = attachments,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 2,
      .pDependencies = dependencies,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &render_pass_info, NULL, &render_pass),
         "vkCreateRenderPass");
   const VkImageView framebuffer_views[2] = {color.view, stencil_view};
   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = 2,
      .pAttachments = framebuffer_views,
      .width = WIDTH,
      .height = HEIGHT,
      .layers = 1,
   };
   VkFramebuffer framebuffer;
   check(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer),
         "vkCreateFramebuffer");

   VkShaderModule vertex = shader_module(device, argv[1]);
   VkShaderModule mask_shader = shader_module(device, argv[2]);
   VkShaderModule bg_shader = shader_module(device, argv[3]);
   VkShaderModule client_shader = shader_module(device, argv[4]);
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout layout;
   check(vkCreatePipelineLayout(device, &layout_info, NULL, &layout),
         "vkCreatePipelineLayout");
   VkPipeline mask = make_pipeline(device, render_pass, layout, vertex,
                                   mask_shader, false, true, false, false);
   VkPipeline background = make_pipeline(device, render_pass, layout, vertex,
                                         bg_shader, true, false, true, false);
   VkPipeline client = make_pipeline(device, render_pass, layout, vertex,
                                     client_shader, true, false, false, true);

   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = READBACK_BYTES,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
         &properties, buffer_requirements.memoryTypeBits,
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
   const VkClearValue clears[2] = {
      {.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}},
      {.depthStencil = {.depth = 1.0f, .stencil = 0}},
   };
   const VkRenderPassBeginInfo render_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {.extent = {WIDTH, HEIGHT}},
      .clearValueCount = 2,
      .pClearValues = clears,
   };
   vkCmdBeginRenderPass(command, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, mask);
   vkCmdDraw(command, 3, 1, 0, 0);
   vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, background);
   vkCmdDraw(command, 3, 1, 0, 0);
   vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, client);
   vkCmdDraw(command, 3, 1, 0, 0);
   vkCmdEndRenderPass(command);
   const VkBufferImageCopy copy = {
      .bufferOffset = GUARD,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1,
      },
      .imageExtent = {WIDTH, HEIGHT, 1},
   };
   vkCmdCopyImageToBuffer(command, color.image,
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

   for (unsigned iteration = 0; iteration < 2; ++iteration) {
      for (size_t i = 0; i < GUARD; ++i) {
         if (map[i] != 0xa5 || map[GUARD + IMAGE_BYTES + i] != 0xa5)
            die("readback guard changed");
      }
      for (unsigned y = 0; y < HEIGHT; ++y) {
         for (unsigned x = 0; x < WIDTH; ++x) {
            const uint8_t expected[4] = {
               x < WIDTH / 2 ? 0x00 : 0x80,
               x < WIDTH / 2 ? 0x00 : 0x80,
               x < WIDTH / 2 ? 0x20 : 0x00,
               0xff,
            };
            const uint8_t *actual = map + GUARD +
                                    (y * WIDTH + x) * PIXEL_BYTES;
            if (memcmp(actual, expected, PIXEL_BYTES) != 0) {
               fprintf(stderr, TEST_NAME ": pixel (%u,%u) got %02x%02x%02x%02x, expected %02x%02x%02x%02x\n",
                       x, y, actual[0], actual[1], actual[2], actual[3],
                       expected[0], expected[1], expected[2], expected[3]);
               return EXIT_FAILURE;
            }
         }
      }
      if (iteration == 0)
         sleep(3);
   }
   printf(TEST_NAME ": PASS — discard-alpha stencil REPLACE, EQUAL-masked "
                    "blurred background, and premultiplied client composite "
                    "exact immediately/3s; guards unchanged\n");
   return EXIT_SUCCESS;
}
