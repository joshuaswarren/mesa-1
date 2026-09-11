/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Minimal Vulkan reproduction of the Omarchy About fragment-program contract.
 * This is deliberately not registered in Meson: it is an isolated hardware
 * oracle while the HAL200 FragmentProgram final scalar remains under study.
 */

#define TEST_NAME "f39-about-vf-cpf"
#define main g17p_about_unused_pingpong_main
#include "g17p_vulkan_hypr_blur_pingpong.c"
#undef main

#include "g17p_vulkan_about_vf_cpf_frag.h"
#include "g17p_vulkan_about_vf_cpf_vert.h"

#include <stddef.h>

struct composite_params {
   uint32_t source_index;
   float opacity;
   float texel_offset[2];
};

_Static_assert(offsetof(struct composite_params, source_index) == 0,
               "std140 source_index offset");
_Static_assert(offsetof(struct composite_params, opacity) == 4,
               "std140 opacity offset");
_Static_assert(offsetof(struct composite_params, texel_offset) == 8,
               "std140 texel_offset offset");
_Static_assert(sizeof(struct composite_params) == 16, "std140 UBO size");

struct host_buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint8_t *map;
   VkDeviceSize size;
};

static struct host_buffer
create_host_buffer(VkDevice device,
                   const VkPhysicalDeviceMemoryProperties *properties,
                   VkDeviceSize size, VkBufferUsageFlags usage)
{
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   struct host_buffer result = {.size = size};
   check(vkCreateBuffer(device, &info, NULL, &result.buffer),
         "vkCreateBuffer");

   VkMemoryRequirements requirements;
   vkGetBufferMemoryRequirements(device, result.buffer, &requirements);
   const VkMemoryAllocateInfo allocation = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = requirements.size,
      .memoryTypeIndex = memory_type(
         properties, requirements.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   check(vkAllocateMemory(device, &allocation, NULL, &result.memory),
         "vkAllocateMemory(buffer)");
   check(vkBindBufferMemory(device, result.buffer, result.memory, 0),
         "vkBindBufferMemory");
   check(vkMapMemory(device, result.memory, 0, size, 0, (void **)&result.map),
         "vkMapMemory");
   return result;
}

static void
destroy_host_buffer(VkDevice device, struct host_buffer *buffer)
{
   vkUnmapMemory(device, buffer->memory);
   vkDestroyBuffer(device, buffer->buffer, NULL);
   vkFreeMemory(device, buffer->memory, NULL);
}

static VkShaderModule
embedded_module(VkDevice device, const uint32_t *words, size_t bytes,
                const char *what)
{
   const VkShaderModuleCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = bytes,
      .pCode = words,
   };
   VkShaderModule module;
   check(vkCreateShaderModule(device, &info, NULL, &module), what);
   return module;
}

enum {
   SPV_OP_TYPE_ARRAY = 28,
   SPV_OP_CONSTANT = 43,
   SPV_OP_DECORATE = 71,
   SPV_OP_MEMBER_DECORATE = 72,
   SPV_OP_IMAGE_SAMPLE_IMPLICIT_LOD = 87,
   SPV_OP_SELECTION_MERGE = 247,
   SPV_OP_BRANCH_CONDITIONAL = 250,
   SPV_DECORATION_BUILTIN = 11,
   SPV_DECORATION_LOCATION = 30,
   SPV_DECORATION_BINDING = 33,
   SPV_DECORATION_DESCRIPTOR_SET = 34,
   SPV_BUILTIN_POINT_SIZE = 1,
};

struct spv_shape {
   unsigned location0;
   unsigned set0_binding0;
   unsigned set0_binding1;
   unsigned array_length2;
   unsigned point_size;
   unsigned implicit_lod;
   unsigned selection_merge;
   unsigned branch_conditional;
};

static void
inspect_spirv(const uint32_t *words, size_t bytes, struct spv_shape *shape)
{
   if (bytes < 5 * sizeof(uint32_t) || words[0] != UINT32_C(0x07230203) ||
       (bytes & 3) != 0)
      die("embedded SPIR-V header is malformed");

   const size_t count = bytes / sizeof(uint32_t);
   const uint32_t bound = words[3];
   if (bound == 0 || bound > 4096)
      die("unexpected SPIR-V ID bound");

   int32_t *sets = malloc(bound * sizeof(*sets));
   int32_t *bindings = malloc(bound * sizeof(*bindings));
   uint32_t *constants = calloc(bound, sizeof(*constants));
   if (!sets || !bindings || !constants)
      die("out of host memory");
   for (uint32_t i = 0; i < bound; ++i)
      sets[i] = bindings[i] = -1;

   for (size_t i = 5; i < count;) {
      const uint16_t length = words[i] >> 16;
      const uint16_t opcode = words[i] & 0xffff;
      if (length == 0 || i + length > count)
         die("invalid SPIR-V instruction length");

      if (opcode == SPV_OP_DECORATE && length >= 4) {
         const uint32_t target = words[i + 1];
         const uint32_t decoration = words[i + 2];
         const uint32_t value = words[i + 3];
         if (target < bound && decoration == SPV_DECORATION_DESCRIPTOR_SET)
            sets[target] = value;
         if (target < bound && decoration == SPV_DECORATION_BINDING)
            bindings[target] = value;
         if (decoration == SPV_DECORATION_LOCATION && value == 0)
            shape->location0++;
      } else if (opcode == SPV_OP_MEMBER_DECORATE && length >= 5 &&
                 words[i + 3] == SPV_DECORATION_BUILTIN &&
                 words[i + 4] == SPV_BUILTIN_POINT_SIZE) {
         shape->point_size++;
      } else if (opcode == SPV_OP_CONSTANT && length >= 4 &&
                 words[i + 2] < bound) {
         constants[words[i + 2]] = words[i + 3];
      } else if (opcode == SPV_OP_TYPE_ARRAY && length >= 4 &&
                 words[i + 3] < bound && constants[words[i + 3]] == 2) {
         shape->array_length2++;
      } else if (opcode == SPV_OP_IMAGE_SAMPLE_IMPLICIT_LOD) {
         shape->implicit_lod++;
      } else if (opcode == SPV_OP_SELECTION_MERGE) {
         shape->selection_merge++;
      } else if (opcode == SPV_OP_BRANCH_CONDITIONAL) {
         shape->branch_conditional++;
      }
      i += length;
   }

   for (uint32_t i = 0; i < bound; ++i) {
      if (sets[i] == 0 && bindings[i] == 0)
         shape->set0_binding0++;
      if (sets[i] == 0 && bindings[i] == 1)
         shape->set0_binding1++;
   }
   free(constants);
   free(bindings);
   free(sets);
}

static void
verify_embedded_shape(void)
{
   struct spv_shape vertex = {0}, fragment = {0};
   inspect_spirv(g17p_vulkan_about_vf_cpf_vert_spv,
                 sizeof(g17p_vulkan_about_vf_cpf_vert_spv), &vertex);
   inspect_spirv(g17p_vulkan_about_vf_cpf_frag_spv,
                 sizeof(g17p_vulkan_about_vf_cpf_frag_spv), &fragment);

   if (vertex.location0 < 2 || vertex.point_size != 1)
      die("vertex SPIR-V lacks position/UV location 0 or PointSize");
   if (fragment.location0 < 2 || fragment.set0_binding0 != 1 ||
       fragment.set0_binding1 != 1 || fragment.array_length2 == 0 ||
       fragment.implicit_lod < 2 || fragment.selection_merge != 1 ||
       fragment.branch_conditional != 1)
      die("fragment SPIR-V does not match the About contract");
}

static void
compare_spirv_file(const char *path, const uint32_t *expected,
                   size_t expected_size)
{
   size_t size;
   uint32_t *actual = read_spv(path, &size);
   if (size != expected_size || memcmp(actual, expected, size) != 0) {
      free(actual);
      die("compiled SPIR-V differs from the embedded fixture");
   }
   free(actual);
}

static bool
verify_pixels_and_guards(const struct host_buffer *readback,
                         const struct host_buffer *vertices,
                         const struct host_buffer *uniform)
{
   for (size_t i = 0; i < GUARD; ++i) {
      if (readback->map[i] != 0xa5 ||
          readback->map[GUARD + IMAGE_BYTES + i] != 0xa5 ||
          vertices->map[i] != 0xa5 || vertices->map[GUARD + 24 + i] != 0xa5 ||
          uniform->map[i] != 0xa5 || uniform->map[GUARD + 16 + i] != 0xa5) {
         fprintf(stderr, TEST_NAME ": guard byte %zu changed\n", i);
         return false;
      }
   }

   static const uint8_t expected[4] = {0x00, 0xff, 0x00, 0xff};
   for (size_t pixel = 0; pixel < WIDTH * HEIGHT; ++pixel) {
      const uint8_t *actual = readback->map + GUARD + pixel * 4;
      if (memcmp(actual, expected, sizeof(expected)) != 0) {
         fprintf(stderr, TEST_NAME ": pixel %zu got %02x%02x%02x%02x, "
                                   "expected 00ff00ff\n",
                 pixel, actual[0], actual[1], actual[2], actual[3]);
         return false;
      }
   }
   return true;
}

int
main(int argc, char **argv)
{
   uint32_t draw_count = 1;
   bool record_only = false;

   verify_embedded_shape();
   if (argc == 4 && strcmp(argv[1], "--check-spv") == 0) {
      compare_spirv_file(argv[2], g17p_vulkan_about_vf_cpf_vert_spv,
                         sizeof(g17p_vulkan_about_vf_cpf_vert_spv));
      compare_spirv_file(argv[3], g17p_vulkan_about_vf_cpf_frag_spv,
                         sizeof(g17p_vulkan_about_vf_cpf_frag_spv));
      puts(TEST_NAME ": SPIR-V shape and embedded fixtures PASS");
      return EXIT_SUCCESS;
   }
   if (argc == 3 &&
       (!strcmp(argv[1], "--draw-count") ||
        !strcmp(argv[1], "--record-only"))) {
      char *end = NULL;
      unsigned long parsed = strtoul(argv[2], &end, 0);
      if (!argv[2][0] || !end || *end || parsed == 0 || parsed > 65536)
         die("draw count must be in 1..65536");
      draw_count = parsed;
      record_only = !strcmp(argv[1], "--record-only");
   } else if (argc != 1) {
      die("usage: f39-about-vf-cpf [--draw-count|--record-only 1..65536] "
          "[--check-spv vert.spv frag.spv]");
   }

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
   if (!physical)
      die("T8140/G17P Vulkan device was not found");

   VkPhysicalDeviceFeatures features;
   vkGetPhysicalDeviceFeatures(physical, &features);
   if (!features.shaderSampledImageArrayDynamicIndexing)
      die("sampled-image array dynamic indexing is unavailable");
   const VkPhysicalDeviceFeatures enabled_features = {
      .shaderSampledImageArrayDynamicIndexing = VK_TRUE,
   };

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
      .pNext = NULL,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .pEnabledFeatures = &enabled_features,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");
   VkQueue queue;
   vkGetDeviceQueue(device, family, 0, &queue);
   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   struct image sources[2];
   for (unsigned i = 0; i < 2; ++i) {
      sources[i] = create_image(
         device, &memory_properties,
         VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
   }
   struct image output = create_image(
      device, &memory_properties,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

   struct host_buffer vertices = create_host_buffer(
      device, &memory_properties, 2 * GUARD + 24,
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
   struct host_buffer uniform = create_host_buffer(
      device, &memory_properties, 2 * GUARD + 16,
      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
   struct host_buffer readback = create_host_buffer(
      device, &memory_properties, READBACK_BYTES,
      VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   memset(vertices.map, 0xa5, vertices.size);
   memset(uniform.map, 0xa5, uniform.size);
   memset(readback.map, 0xa5, readback.size);
   static const float triangle[6] = {-1.0f, -1.0f, 3.0f,
                                     -1.0f, -1.0f, 3.0f};
   memcpy(vertices.map + GUARD, triangle, sizeof(triangle));
   const struct composite_params params = {
      .source_index = 1,
      .opacity = 1.0f,
      .texel_offset = {0.0f, 0.0f},
   };
   memcpy(uniform.map + GUARD, &params, sizeof(params));

   const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = 0.0f,
   };
   VkSampler sampler;
   check(vkCreateSampler(device, &sampler_info, NULL, &sampler),
         "vkCreateSampler");

   const VkDescriptorSetLayoutBinding bindings[2] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 2,
         .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
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
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
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
   const VkDescriptorImageInfo image_info[2] = {
      {sampler, sources[0].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
      {sampler, sources[1].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
   };
   const VkDescriptorBufferInfo uniform_info = {
      .buffer = uniform.buffer,
      .offset = GUARD,
      .range = sizeof(params),
   };
   const VkWriteDescriptorSet writes[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 0,
         .descriptorCount = 2,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = image_info,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = descriptor_set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pBufferInfo = &uniform_info,
      },
   };
   vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

   VkRenderPass render_pass = create_render_pass(
      device, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
   VkFramebuffer framebuffer =
      create_framebuffer(device, render_pass, output.view);
   VkShaderModule vertex = embedded_module(
      device, g17p_vulkan_about_vf_cpf_vert_spv,
      sizeof(g17p_vulkan_about_vf_cpf_vert_spv),
      "vkCreateShaderModule(vertex)");
   VkShaderModule fragment = embedded_module(
      device, g17p_vulkan_about_vf_cpf_frag_spv,
      sizeof(g17p_vulkan_about_vf_cpf_frag_spv),
      "vkCreateShaderModule(fragment)");
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
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
       VK_SHADER_STAGE_VERTEX_BIT, vertex, "main", NULL},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
       VK_SHADER_STAGE_FRAGMENT_BIT, fragment, "main", NULL},
   };
   const VkVertexInputBindingDescription vertex_binding = {
      .binding = 0,
      .stride = 2 * sizeof(float),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription vertex_attribute = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &vertex_attribute,
   };
   const VkPipelineInputAssemblyStateCreateInfo assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {.width = WIDTH, .height = HEIGHT,
                                .maxDepth = 1.0f};
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
      .renderPass = render_pass,
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

   VkImageMemoryBarrier to_clear[2];
   for (unsigned i = 0; i < 2; ++i) {
      to_clear[i] = (VkImageMemoryBarrier){
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = sources[i].image,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
      };
   }
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                        2, to_clear);
   const VkClearColorValue source_colours[2] = {
      {.float32 = {1.0f, 0.0f, 0.0f, 1.0f}},
      {.float32 = {0.0f, 1.0f, 0.0f, 1.0f}},
   };
   const VkImageSubresourceRange colour_range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1,
   };
   for (unsigned i = 0; i < 2; ++i)
      vkCmdClearColorImage(command, sources[i].image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           &source_colours[i], 1, &colour_range);
   VkImageMemoryBarrier to_sample[2];
   for (unsigned i = 0; i < 2; ++i) {
      to_sample[i] = (VkImageMemoryBarrier){
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = sources[i].image,
         .subresourceRange = colour_range,
      };
   }
   vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0,
                        NULL, 2, to_sample);

   const VkClearValue black = {
      .color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}},
   };
   const VkRenderPassBeginInfo render_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {.extent = {WIDTH, HEIGHT}},
      .clearValueCount = 1,
      .pClearValues = &black,
   };
   vkCmdBeginRenderPass(command, &render_begin, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           pipeline_layout, 0, 1, &descriptor_set, 0, NULL);
   const VkDeviceSize vertex_offset = GUARD;
   vkCmdBindVertexBuffers(command, 0, 1, &vertices.buffer, &vertex_offset);
   /* A single draw proves the shader and descriptor contract, but it cannot
    * expose a parameter-buffer packet-size mismatch: that kind of mismatch
    * accumulates as VF_CPF consumes successive transformed primitives.  Keep
    * every other state bit fixed and repeat only the exact size-7 About draw.
    * This is deliberately selectable so the one-draw control remains in the
    * same executable and the stress cell has no shader or pipeline delta. */
   for (uint32_t i = 0; i < draw_count; ++i)
      vkCmdDraw(command, 3, 1, 0, 0);
   vkCmdEndRenderPass(command);
   const VkBufferImageCopy copy = {
      .bufferOffset = GUARD,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {WIDTH, HEIGHT, 1},
   };
   vkCmdCopyImageToBuffer(command, output.image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback.buffer, 1, &copy);
   check(vkEndCommandBuffer(command), "vkEndCommandBuffer");
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command,
   };
   check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit");
   check(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
   if (record_only) {
      printf(TEST_NAME ": PASS — recorded/submitted %u exact size-7 "
                       "draw(s)\n",
             draw_count);
   } else {
      if (!verify_pixels_and_guards(&readback, &vertices, &uniform))
         return EXIT_FAILURE;
      sleep(3);
      if (!verify_pixels_and_guards(&readback, &vertices, &uniform))
         return EXIT_FAILURE;

      printf(TEST_NAME ": PASS — %u exact size-7 draw(s), set0 binding0 "
                       "sampler2D[2], binding1 std140 UBO, smooth vec2 + "
                       "PointSize, dynamic source[1], implicit LOD and guarded "
                       "RGBA8 output immediately/3s\n",
             draw_count);
   }

   vkDestroyCommandPool(device, command_pool, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   vkDestroyPipelineLayout(device, pipeline_layout, NULL);
   vkDestroyShaderModule(device, fragment, NULL);
   vkDestroyShaderModule(device, vertex, NULL);
   vkDestroyFramebuffer(device, framebuffer, NULL);
   vkDestroyRenderPass(device, render_pass, NULL);
   vkDestroyDescriptorPool(device, descriptor_pool, NULL);
   vkDestroyDescriptorSetLayout(device, set_layout, NULL);
   vkDestroySampler(device, sampler, NULL);
   destroy_host_buffer(device, &readback);
   destroy_host_buffer(device, &uniform);
   destroy_host_buffer(device, &vertices);
   vkDestroyImageView(device, output.view, NULL);
   vkDestroyImage(device, output.image, NULL);
   vkFreeMemory(device, output.memory, NULL);
   for (unsigned i = 0; i < 2; ++i) {
      vkDestroyImageView(device, sources[i].view, NULL);
      vkDestroyImage(device, sources[i].image, NULL);
      vkFreeMemory(device, sources[i].memory, NULL);
   }
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return EXIT_SUCCESS;
}
