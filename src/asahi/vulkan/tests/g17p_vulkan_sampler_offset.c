/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Texture-state alignment oracle for G17P sampled-texture pointers.
 *
 * The G17P combined-sampler and texel-fetch ABIs hand the texture unit a
 * pointer to the 32-byte texture state, and the hardware ignores that
 * pointer's low five bits.  Sampled-image descriptors are only 8-byte aligned
 * (72-byte stride), so any binding after a 16-byte buffer descriptor puts the
 * texture word at 16 mod 32.  vkcube's set 0 is exactly that shape -- a
 * uniform buffer at binding 0 and its one combined image sampler at binding 1
 * -- and on drivers that published set address + binding offset its first
 * frame faulted the fragment unit and poisoned the VM (t-1411).
 *
 * Modes (one per run, so a fault in one cannot hide a result of another):
 *   graphics          UBO at binding 0, combined image sampler at binding 1:
 *                     texture word at set offset 16.
 *   graphics-control  the same shaders with UNIFORM_BUFFER_DYNAMIC at binding
 *                     0, which takes no set space: texture word at offset 0.
 *   compute           SSBO at binding 0, sampled image read with texelFetch
 *                     at binding 1: texture word at set offset 16.
 *   compute-control   STORAGE_BUFFER_DYNAMIC at binding 0: offset 0.
 *
 * Every output texel is compared exactly against a 4x4 RGBA8 source read with
 * NEAREST filtering.  Exit status: 0 exact, 1 wrong texels or device loss,
 * 2 setup failure.  A failing run on hardware may also GPU-fault, so never
 * loop it against a driver that is known to fail.
 */

#include <vulkan/vulkan.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "g17p_vulkan_sampler_offset_comp.h"
#include "g17p_vulkan_sampler_offset_frag.h"
#include "g17p_vulkan_sampler_offset_vert.h"

#define TEST_NAME       "g17p-vulkan-sampler-offset"
#define T8140_DEVICE_ID 0x8140
#define TEX_DIM         4
#define TEX_TEXELS      (TEX_DIM * TEX_DIM)
#define TEX_BYTES       (TEX_TEXELS * 4)
#define RT_DIM_MAX      64
#define RT_BYTES        (RT_DIM_MAX * RT_DIM_MAX * 4)
#define WAIT_TIMEOUT_NS UINT64_C(5000000000)

static void
die(const char *what)
{
   fprintf(stderr, TEST_NAME ": %s\n", what);
   exit(2);
}

static void
check(VkResult result, const char *what)
{
   if (result == VK_ERROR_DEVICE_LOST) {
      fprintf(stderr, TEST_NAME ": FAIL %s: VK_ERROR_DEVICE_LOST\n", what);
      exit(1);
   }
   if (result != VK_SUCCESS) {
      fprintf(stderr, TEST_NAME ": %s returned %d\n", what, result);
      exit(2);
   }
}

static uint32_t
find_memory_type(const VkPhysicalDeviceMemoryProperties *properties,
                 uint32_t allowed, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if ((allowed & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }
   die("required memory type is unavailable");
   return UINT32_MAX;
}

/* Distinct, non-zero bytes per texel and channel. */
static void
texel(uint32_t index, uint8_t rgba[4])
{
   rgba[0] = 0x11 * (index + 1);
   rgba[1] = 0xf0 - 7 * index;
   rgba[2] = 0x20 + 13 * index;
   rgba[3] = 0x80 | index;
}

struct buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   uint8_t *map;
};

static struct buffer
create_buffer(VkDevice device, const VkPhysicalDeviceMemoryProperties *mem,
              VkDeviceSize size, VkBufferUsageFlags usage)
{
   struct buffer b;
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   check(vkCreateBuffer(device, &info, NULL, &b.buffer), "vkCreateBuffer");
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(device, b.buffer, &req);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = find_memory_type(
         mem, req.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   check(vkAllocateMemory(device, &alloc, NULL, &b.memory),
         "vkAllocateMemory(buffer)");
   check(vkBindBufferMemory(device, b.buffer, b.memory, 0),
         "vkBindBufferMemory");
   check(vkMapMemory(device, b.memory, 0, size, 0, (void **)&b.map),
         "vkMapMemory");
   return b;
}

static VkImage
create_image(VkDevice device, const VkPhysicalDeviceMemoryProperties *mem,
             VkFormat format, uint32_t dim, VkImageUsageFlags usage,
             VkDeviceMemory *memory)
{
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {dim, dim, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   check(vkCreateImage(device, &info, NULL, &image), "vkCreateImage");
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(device, image, &req);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = find_memory_type(mem, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   check(vkAllocateMemory(device, &alloc, NULL, memory),
         "vkAllocateMemory(image)");
   check(vkBindImageMemory(device, image, *memory, 0), "vkBindImageMemory");
   return image;
}

static void
barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from,
        VkImageLayout to, VkAccessFlags src_access, VkAccessFlags dst_access,
        VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage)
{
   const VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = src_access,
      .dstAccessMask = dst_access,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1,
                        &b);
}

static VkShaderModule
shader(VkDevice device, const uint32_t *code, size_t size)
{
   const VkShaderModuleCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code,
   };
   VkShaderModule module;
   check(vkCreateShaderModule(device, &info, NULL, &module),
         "vkCreateShaderModule");
   return module;
}

int
main(int argc, char **argv)
{
   if (argc != 2)
      die("usage: " TEST_NAME
          " graphics|graphics-control|compute|compute-control");
   const char *mode = argv[1];
   const bool compute = !strncmp(mode, "compute", 7);
   const bool control = strstr(mode, "-control") != NULL;
   if (strcmp(mode, "graphics") && strcmp(mode, "graphics-control") &&
       strcmp(mode, "compute") && strcmp(mode, "compute-control"))
      die("unknown mode");

   /* Indexed-heap lowering normally wins for this shader.  Force the narrower
    * direct txf ABI so this probe covers its set-relative texture pointer. */
   if (compute && setenv("HK_PERFTEST", "txf", 1))
      die("setenv(HK_PERFTEST=txf) failed");
   /* G17P_PROBE_RT_DIM (4..64, a multiple of 4) sizes the render target;
    * below 16x16 hk does not compress it. */
   const char *rt_env = getenv("G17P_PROBE_RT_DIM");
   const uint32_t rt_dim = rt_env ? (uint32_t)atoi(rt_env) : RT_DIM_MAX;
   if (rt_dim < TEX_DIM || rt_dim > RT_DIM_MAX || rt_dim % TEX_DIM)
      die("G17P_PROBE_RT_DIM must be a multiple of 4 in 4..64");
   /* vkcube sets its viewport and scissor dynamically, as does this probe by
    * default; G17P_PROBE_STATIC_VIEWPORT=1 bakes them into the pipeline. */
   const bool static_viewport = getenv("G17P_PROBE_STATIC_VIEWPORT") != NULL;
   const VkViewport viewport = {0, 0, rt_dim, rt_dim, 0, 1};
   const VkRect2D scissor = {{0, 0}, {rt_dim, rt_dim}};

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

   uint32_t count = 0;
   check(vkEnumeratePhysicalDevices(instance, &count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   VkPhysicalDevice physicals[8];
   if (!count)
      die("Vulkan loader found no physical device");
   if (count > 8)
      count = 8;
   check(vkEnumeratePhysicalDevices(instance, &count, physicals),
         "vkEnumeratePhysicalDevices(list)");
   VkPhysicalDevice physical = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < count; ++i) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(physicals[i], &props);
      if (props.deviceID == T8140_DEVICE_ID)
         physical = physicals[i];
   }
   /* G17P_PROBE_ANY_DEVICE=1 checks the oracle itself on another GPU. */
   if (physical == VK_NULL_HANDLE && getenv("G17P_PROBE_ANY_DEVICE"))
      physical = physicals[0];
   if (physical == VK_NULL_HANDLE)
      die("T8140/G17P Vulkan device was not found");

   uint32_t family_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, NULL);
   VkQueueFamilyProperties families[8];
   if (family_count > 8)
      family_count = 8;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count,
                                            families);
   const VkQueueFlags wanted = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
   uint32_t family = UINT32_MAX;
   for (uint32_t i = 0; i < family_count && family == UINT32_MAX; ++i) {
      if (families[i].queueCount &&
          (families[i].queueFlags & wanted) == wanted)
         family = i;
   }
   if (family == UINT32_MAX)
      die("no graphics+compute queue family");

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
   VkPhysicalDeviceMemoryProperties mem;
   vkGetPhysicalDeviceMemoryProperties(physical, &mem);

   /* Staging holds the texture source, then the readback. */
   struct buffer staging =
      create_buffer(device, &mem, TEX_BYTES + RT_BYTES,
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT);
   for (uint32_t i = 0; i < TEX_TEXELS; ++i)
      texel(i, staging.map + 4 * i);
   memset(staging.map + TEX_BYTES, 0xa5, RT_BYTES);

   /* The binding-0 buffer: an unread uniform buffer for graphics, the
    * output words for compute.  Its descriptor's bytes precede the texture
    * word in the set and are never zero, which is what made the misread
    * state fault. */
   const VkDeviceSize b0_size = compute ? TEX_BYTES : 16;
   struct buffer b0 = create_buffer(
      device, &mem, b0_size,
      compute ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
              : VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
   if (compute) {
      memset(b0.map, 0xa5, TEX_BYTES);
   } else {
      const float scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
      memcpy(b0.map, scale, sizeof(scale));
   }

   VkDeviceMemory tex_memory;
   const VkFormat tex_format = compute ? VK_FORMAT_R8G8B8A8_UINT
                                       : VK_FORMAT_R8G8B8A8_UNORM;
   VkImage tex = create_image(device, &mem, tex_format, TEX_DIM,
                              VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                              &tex_memory);
   const VkImageViewCreateInfo tex_view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = tex,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = tex_format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkImageView tex_view;
   check(vkCreateImageView(device, &tex_view_info, NULL, &tex_view),
         "vkCreateImageView(texture)");
   const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   VkSampler sampler;
   check(vkCreateSampler(device, &sampler_info, NULL, &sampler),
         "vkCreateSampler");

   const VkDescriptorType b0_type =
      compute ? (control ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                         : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
              : (control ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                         : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
   const VkDescriptorType b1_type =
      compute ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
              : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   const VkDescriptorSetLayoutBinding bindings[2] = {
      {
         .binding = 0,
         .descriptorType = b0_type,
         .descriptorCount = 1,
         .stageFlags = compute ? VK_SHADER_STAGE_COMPUTE_BIT
                               : VK_SHADER_STAGE_VERTEX_BIT,
      },
      {
         .binding = 1,
         .descriptorType = b1_type,
         .descriptorCount = 1,
         .stageFlags = compute ? VK_SHADER_STAGE_COMPUTE_BIT
                               : VK_SHADER_STAGE_FRAGMENT_BIT,
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
      {b0_type, 1},
      {b1_type, 1},
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool pool;
   check(vkCreateDescriptorPool(device, &pool_info, NULL, &pool),
         "vkCreateDescriptorPool");
   const VkDescriptorSetAllocateInfo set_alloc = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkDescriptorSet set;
   check(vkAllocateDescriptorSets(device, &set_alloc, &set),
         "vkAllocateDescriptorSets");
   const VkDescriptorBufferInfo b0_desc = {b0.buffer, 0, b0_size};
   const VkDescriptorImageInfo b1_desc = {
      .sampler = compute ? VK_NULL_HANDLE : sampler,
      .imageView = tex_view,
      .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   const VkWriteDescriptorSet writes[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = b0_type,
         .pBufferInfo = &b0_desc,
      },
      {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = set,
         .dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = b1_type,
         .pImageInfo = &b1_desc,
      },
   };
   vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

   const VkPipelineLayoutCreateInfo pipeline_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkPipelineLayout pipeline_layout;
   check(vkCreatePipelineLayout(device, &pipeline_layout_info, NULL,
                                &pipeline_layout),
         "vkCreatePipelineLayout");

   VkPipeline pipeline;
   VkRenderPass render_pass = VK_NULL_HANDLE;
   VkFramebuffer framebuffer = VK_NULL_HANDLE;
   VkImage rt = VK_NULL_HANDLE;
   VkDeviceMemory rt_memory = VK_NULL_HANDLE;
   VkImageView rt_view = VK_NULL_HANDLE;
   if (compute) {
      VkShaderModule cs = shader(device, g17p_vulkan_sampler_offset_comp,
                                 sizeof(g17p_vulkan_sampler_offset_comp));
      const VkComputePipelineCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = cs,
            .pName = "main",
         },
         .layout = pipeline_layout,
      };
      check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &info, NULL,
                                     &pipeline),
            "vkCreateComputePipelines");
      vkDestroyShaderModule(device, cs, NULL);
   } else {
      rt = create_image(device, &mem, VK_FORMAT_R8G8B8A8_UNORM, rt_dim,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                        &rt_memory);
      VkImageViewCreateInfo rt_view_info = tex_view_info;
      rt_view_info.image = rt;
      rt_view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
      check(vkCreateImageView(device, &rt_view_info, NULL, &rt_view),
            "vkCreateImageView(target)");
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
      const VkAttachmentReference color_ref = {
         0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
      const VkSubpassDescription subpass = {
         .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
         .colorAttachmentCount = 1,
         .pColorAttachments = &color_ref,
      };
      /* The readback copy must wait for the colour writes; the implicit
       * external dependency only reaches BOTTOM_OF_PIPE. */
      const VkSubpassDependency to_copy = {
         .srcSubpass = 0,
         .dstSubpass = VK_SUBPASS_EXTERNAL,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      };
      const VkRenderPassCreateInfo rp_info = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
         .attachmentCount = 1,
         .pAttachments = &attachment,
         .subpassCount = 1,
         .pSubpasses = &subpass,
         .dependencyCount = 1,
         .pDependencies = &to_copy,
      };
      check(vkCreateRenderPass(device, &rp_info, NULL, &render_pass),
            "vkCreateRenderPass");
      const VkFramebufferCreateInfo fb_info = {
         .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
         .renderPass = render_pass,
         .attachmentCount = 1,
         .pAttachments = &rt_view,
         .width = rt_dim,
         .height = rt_dim,
         .layers = 1,
      };
      check(vkCreateFramebuffer(device, &fb_info, NULL, &framebuffer),
            "vkCreateFramebuffer");

      VkShaderModule vs = shader(device, g17p_vulkan_sampler_offset_vert,
                                 sizeof(g17p_vulkan_sampler_offset_vert));
      VkShaderModule fs = shader(device, g17p_vulkan_sampler_offset_frag,
                                 sizeof(g17p_vulkan_sampler_offset_frag));
      const VkPipelineShaderStageCreateInfo stages[2] = {
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vs,
            .pName = "main",
         },
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fs,
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
      const VkPipelineViewportStateCreateInfo viewport_state = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
         .viewportCount = 1,
         .pViewports = static_viewport ? &viewport : NULL,
         .scissorCount = 1,
         .pScissors = static_viewport ? &scissor : NULL,
      };
      const VkDynamicState dynamic_states[] = {
         VK_DYNAMIC_STATE_VIEWPORT,
         VK_DYNAMIC_STATE_SCISSOR,
      };
      const VkPipelineDynamicStateCreateInfo dynamic = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
         .dynamicStateCount = 2,
         .pDynamicStates = dynamic_states,
      };
      const VkPipelineRasterizationStateCreateInfo raster = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
         .polygonMode = VK_POLYGON_MODE_FILL,
         .cullMode = VK_CULL_MODE_NONE,
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
      const VkGraphicsPipelineCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .stageCount = 2,
         .pStages = stages,
         .pVertexInputState = &vertex_input,
         .pInputAssemblyState = &input_assembly,
         .pViewportState = &viewport_state,
         .pRasterizationState = &raster,
         .pMultisampleState = &multisample,
         .pColorBlendState = &blend,
         .pDynamicState = static_viewport ? NULL : &dynamic,
         .layout = pipeline_layout,
         .renderPass = render_pass,
      };
      check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, NULL,
                                      &pipeline),
            "vkCreateGraphicsPipelines");
      vkDestroyShaderModule(device, vs, NULL);
      vkDestroyShaderModule(device, fs, NULL);
   }

   const VkCommandPoolCreateInfo cmd_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   VkCommandPool cmd_pool;
   check(vkCreateCommandPool(device, &cmd_pool_info, NULL, &cmd_pool),
         "vkCreateCommandPool");
   const VkCommandBufferAllocateInfo cmd_alloc = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cmd_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   check(vkAllocateCommandBuffers(device, &cmd_alloc, &cmd),
         "vkAllocateCommandBuffers");
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");

   barrier(cmd, tex, VK_IMAGE_LAYOUT_UNDEFINED,
           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
           VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
           VK_PIPELINE_STAGE_TRANSFER_BIT);
   const VkBufferImageCopy upload = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {TEX_DIM, TEX_DIM, 1},
   };
   vkCmdCopyBufferToImage(cmd, staging.buffer, tex,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &upload);
   barrier(cmd, tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
           VK_PIPELINE_STAGE_TRANSFER_BIT,
           compute ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                   : VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

   const uint32_t dynamic_offset = 0;
   const uint32_t dynamic_count = control ? 1 : 0;
   if (compute) {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              pipeline_layout, 0, 1, &set, dynamic_count,
                              &dynamic_offset);
      vkCmdDispatch(cmd, 1, 1, 1);
      const VkMemoryBarrier host = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL,
                           0, NULL);
   } else {
      const VkClearValue clear = {.color = {.float32 = {0, 0, 0, 0}}};
      const VkRenderPassBeginInfo rp_begin = {
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
         .renderPass = render_pass,
         .framebuffer = framebuffer,
         .renderArea = {{0, 0}, {rt_dim, rt_dim}},
         .clearValueCount = 1,
         .pClearValues = &clear,
      };
      vkCmdBeginRenderPass(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      if (!static_viewport) {
         vkCmdSetViewport(cmd, 0, 1, &viewport);
         vkCmdSetScissor(cmd, 0, 1, &scissor);
      }
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              pipeline_layout, 0, 1, &set, dynamic_count,
                              &dynamic_offset);
      vkCmdDraw(cmd, 3, 1, 0, 0);
      vkCmdEndRenderPass(cmd);
      const VkBufferImageCopy readback = {
         .bufferOffset = TEX_BYTES,
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .imageExtent = {rt_dim, rt_dim, 1},
      };
      vkCmdCopyImageToBuffer(cmd, rt, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             staging.buffer, 1, &readback);
      const VkMemoryBarrier host = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL,
                           0, NULL);
   }
   check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence fence;
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
   };
   check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
   check(vkWaitForFences(device, 1, &fence, VK_TRUE, WAIT_TIMEOUT_NS),
         "vkWaitForFences");

   unsigned bad = 0;
   if (compute) {
      for (uint32_t i = 0; i < TEX_TEXELS; ++i) {
         uint8_t want[4];
         texel(i, want);
         const uint8_t *got = b0.map + 4 * i;
         if (memcmp(got, want, 4) && bad++ < 4)
            fprintf(stderr,
                    TEST_NAME ": texel %u is %02x%02x%02x%02x, expected "
                    "%02x%02x%02x%02x\n",
                    i, got[0], got[1], got[2], got[3], want[0], want[1],
                    want[2], want[3]);
      }
   } else {
      const uint32_t cell = rt_dim / TEX_DIM;
      for (uint32_t y = 0; y < rt_dim; ++y) {
         for (uint32_t x = 0; x < rt_dim; ++x) {
            uint8_t want[4];
            texel((y / cell) * TEX_DIM + x / cell, want);
            const uint8_t *got = staging.map + TEX_BYTES + 4 * (y * rt_dim + x);
            if (memcmp(got, want, 4) && bad++ < 4)
               fprintf(stderr,
                       TEST_NAME ": pixel (%u,%u) is %02x%02x%02x%02x, "
                       "expected %02x%02x%02x%02x\n",
                       x, y, got[0], got[1], got[2], got[3], want[0], want[1],
                       want[2], want[3]);
         }
      }
   }

   printf(TEST_NAME ": %s: texture word at set offset %u, %ux%u target%s: "
          "%s (%u wrong)\n",
          mode, control ? 0u : 16u, rt_dim, rt_dim,
          compute ? "" : static_viewport ? ", static viewport" : "",
          bad ? "FAIL" : "PASS", bad);

   vkDestroyFence(device, fence, NULL);
   vkDestroyCommandPool(device, cmd_pool, NULL);
   vkDestroyPipeline(device, pipeline, NULL);
   if (!compute) {
      vkDestroyFramebuffer(device, framebuffer, NULL);
      vkDestroyRenderPass(device, render_pass, NULL);
      vkDestroyImageView(device, rt_view, NULL);
      vkDestroyImage(device, rt, NULL);
      vkFreeMemory(device, rt_memory, NULL);
   }
   vkDestroyPipelineLayout(device, pipeline_layout, NULL);
   vkDestroyDescriptorPool(device, pool, NULL);
   vkDestroyDescriptorSetLayout(device, set_layout, NULL);
   vkDestroySampler(device, sampler, NULL);
   vkDestroyImageView(device, tex_view, NULL);
   vkDestroyImage(device, tex, NULL);
   vkFreeMemory(device, tex_memory, NULL);
   vkDestroyBuffer(device, b0.buffer, NULL);
   vkFreeMemory(device, b0.memory, NULL);
   vkDestroyBuffer(device, staging.buffer, NULL);
   vkFreeMemory(device, staging.memory, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return bad ? 1 : 0;
}
