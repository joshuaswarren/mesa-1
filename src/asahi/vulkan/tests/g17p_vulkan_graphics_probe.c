/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Generic G17P graphics-pipeline encodability probe.
 *
 * Takes a vertex SPIR-V and one or more fragment SPIR-V modules, creates a
 * graphics pipeline from each pair, and prints the VkResult. This is the
 * fragment-side counterpart of g17p_vulkan_pipeline_probe and needs no
 * hardware, only the G17P drm-shim -- which is how the fragment refusals
 * listed in src/asahi/isa/README-agx3-packer-limits.md were measured.
 *
 * COLOR_FORMAT and DEPTH_FORMAT (numeric VkFormat values) override the
 * attachment formats; DEPTH_FORMAT=0 drops the depth attachment.
 *
 * PROBE_BINDINGS replaces the default descriptor set 0 with a comma-separated
 * list of <binding>:<kind> entries (ubo, ssbo, cis, image, sampler), each
 * visible to every graphics stage.  The default -- a vertex uniform buffer at
 * 0 and a fragment combined image sampler at 1 -- is what a hand-written probe
 * shader uses, but a real application's shader declares its own set, and a
 * pipeline layout that does not contain a binding the shader uses does not
 * fail cleanly: it aborts inside hk_nir_lower_descriptors.c, upstream of the
 * packer this probe exists to measure.
 *
 * PROBE_PAIRS=1 reads argv as alternating <vert.spv> <frag.spv> pairs instead
 * of one vertex module shared by every fragment module, so a corpus captured
 * from an application -- where each program is a linked pair -- can be replayed
 * as the pairs it actually was.
 *
 * Exit status is the number of pipelines that failed.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_NAME "g17p-vulkan-graphics-probe"
#define T8140_DEVICE_ID 0x8140

#define VK_FN(name) static PFN_##name name
VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkCreateDevice);
VK_FN(vkDestroyDevice);
VK_FN(vkCreateShaderModule);
VK_FN(vkDestroyShaderModule);
VK_FN(vkCreateRenderPass);
VK_FN(vkDestroyRenderPass);
VK_FN(vkCreateDescriptorSetLayout);
VK_FN(vkDestroyDescriptorSetLayout);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkDestroyPipelineLayout);
VK_FN(vkCreateGraphicsPipelines);
VK_FN(vkDestroyPipeline);

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

#define LOAD_GLOBAL(name)                                                      \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);         \
      if (!name)                                                               \
         die("loader is missing " #name);                                      \
   } while (0)
#define LOAD_INSTANCE(instance, name)                                          \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);               \
      if (!name)                                                               \
         die("instance is missing " #name);                                    \
   } while (0)
#define LOAD_DEVICE(device, name)                                              \
   do {                                                                        \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                   \
      if (!name)                                                               \
         die("device is missing " #name);                                      \
   } while (0)

static uint32_t *
read_spv(const char *path, size_t *size_out)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      fprintf(stderr, TEST_NAME ": cannot open %s\n", path);
      exit(EXIT_FAILURE);
   }
   fseek(f, 0, SEEK_END);
   long size = ftell(f);
   fseek(f, 0, SEEK_SET);
   if (size <= 0 || (size % 4) != 0) {
      fprintf(stderr, TEST_NAME ": %s is not a SPIR-V module\n", path);
      exit(EXIT_FAILURE);
   }
   uint32_t *words = malloc((size_t)size);
   if (!words || fread(words, 1, (size_t)size, f) != (size_t)size)
      die("short read");
   fclose(f);
   *size_out = (size_t)size;
   return words;
}

static VkFormat
env_format(const char *name, VkFormat fallback)
{
   const char *v = getenv(name);
   return v ? (VkFormat)atoi(v) : fallback;
}

#define PROBE_MAX_BINDINGS 64

/* Parse PROBE_BINDINGS, returning the number of bindings written.  Returns 2
 * -- leaving the caller's default in place -- when the variable is unset, so
 * every existing caller of this probe keeps the layout it was measured with. */
static uint32_t
parse_bindings(VkDescriptorSetLayoutBinding *out)
{
   const char *spec = getenv("PROBE_BINDINGS");
   if (!spec || !*spec)
      return 2;

   uint32_t n = 0;
   for (const char *p = spec; *p && n < PROBE_MAX_BINDINGS;) {
      char *end = NULL;
      const unsigned long binding = strtoul(p, &end, 10);
      if (end == p || *end != ':')
         die("PROBE_BINDINGS: expected <binding>:<kind>");
      p = end + 1;

      const char *kind = p;
      while (*p && *p != ',')
         ++p;
      const size_t len = (size_t)(p - kind);
      if (*p == ',')
         ++p;

      VkDescriptorType type;
      if (!strncmp(kind, "ubo", len) && len == 3)
         type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      else if (!strncmp(kind, "ssbo", len) && len == 4)
         type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      else if (!strncmp(kind, "cis", len) && len == 3)
         type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      else if (!strncmp(kind, "image", len) && len == 5)
         type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      else if (!strncmp(kind, "sampled", len) && len == 7)
         type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      else if (!strncmp(kind, "sampler", len) && len == 7)
         type = VK_DESCRIPTOR_TYPE_SAMPLER;
      else
         die("PROBE_BINDINGS: unknown descriptor kind");

      out[n++] = (VkDescriptorSetLayoutBinding){
         .binding = (uint32_t)binding,
         .descriptorType = type,
         .descriptorCount = 1,
         /* Every stage: which stage uses a binding is the shader's business,
          * and a stage flag that is too narrow fails the pipeline for a reason
          * that has nothing to do with what is being measured. */
         .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
      };
   }
   return n;
}

int
main(int argc, char **argv)
{
   if (argc < 3) {
      fprintf(stderr, "usage: " TEST_NAME " <vert.spv> <frag.spv> [more.spv]\n");
      return EXIT_FAILURE;
   }

   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());
   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("libvulkan.so.1 has no vkGetInstanceProcAddr");
   LOAD_GLOBAL(vkCreateInstance);

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = TEST_NAME,
      .apiVersion = VK_API_VERSION_1_1,
   };
   const VkInstanceCreateInfo ii = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   check(vkCreateInstance(&ii, NULL, &instance), "vkCreateInstance");

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t count = 0;
   check(vkEnumeratePhysicalDevices(instance, &count, NULL), "enumerate");
   VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
   if (!devices)
      die("out of host memory");
   check(vkEnumeratePhysicalDevices(instance, &count, devices), "enumerate");
   VkPhysicalDevice physical = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < count; ++i) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(devices[i], &props);
      if (props.deviceID == T8140_DEVICE_ID) {
         physical = devices[i];
         break;
      }
   }
   free(devices);
   if (physical == VK_NULL_HANDLE)
      die("T8140/G17P Vulkan device was not found");

   uint32_t families = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, NULL);
   VkQueueFamilyProperties *props = calloc(families, sizeof(*props));
   if (!props)
      die("out of host memory");
   vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, props);
   uint32_t family = UINT32_MAX;
   for (uint32_t i = 0; i < families; ++i) {
      if (props[i].queueCount != 0) {
         family = i;
         break;
      }
   }
   free(props);
   if (family == UINT32_MAX)
      die("device has no usable queue family");

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qi,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &di, NULL, &device), "vkCreateDevice");

   LOAD_DEVICE(device, vkDestroyDevice);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkDestroyShaderModule);
   LOAD_DEVICE(device, vkCreateRenderPass);
   LOAD_DEVICE(device, vkDestroyRenderPass);
   LOAD_DEVICE(device, vkCreateDescriptorSetLayout);
   LOAD_DEVICE(device, vkDestroyDescriptorSetLayout);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkDestroyPipelineLayout);
   LOAD_DEVICE(device, vkCreateGraphicsPipelines);
   LOAD_DEVICE(device, vkDestroyPipeline);

   /* A uniform buffer and a combined image sampler: the two descriptors a
    * textured draw needs, which is what vkcube binds.  PROBE_BINDINGS
    * replaces this wholesale for a captured application shader. */
   VkDescriptorSetLayoutBinding binds[PROBE_MAX_BINDINGS] = {
      {.binding = 0,
       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_VERTEX_BIT},
      {.binding = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
   };
   uint32_t nr_binds = parse_bindings(binds);
   const VkDescriptorSetLayoutCreateInfo dsli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = nr_binds,
      .pBindings = binds,
   };
   VkDescriptorSetLayout set_layout;
   check(vkCreateDescriptorSetLayout(device, &dsli, NULL, &set_layout), "dsl");
   const VkPushConstantRange push = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .size = 64};
   const VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push,
   };
   VkPipelineLayout layout;
   check(vkCreatePipelineLayout(device, &pli, NULL, &layout), "layout");

   const VkFormat color = env_format("COLOR_FORMAT", VK_FORMAT_B8G8R8A8_UNORM);
   const VkFormat depth = env_format("DEPTH_FORMAT", VK_FORMAT_D16_UNORM);
   const VkAttachmentDescription atts[2] = {
      {.format = color,
       .samples = VK_SAMPLE_COUNT_1_BIT,
       .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
       .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
       .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
       .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
      {.format = depth,
       .samples = VK_SAMPLE_COUNT_1_BIT,
       .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
       .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
       .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
       .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
       .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
       .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
   };
   const VkAttachmentReference cref = {
      .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
   const VkAttachmentReference zref = {
      .attachment = 1,
      .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
   const VkSubpassDescription sub = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &cref,
      .pDepthStencilAttachment = depth ? &zref : NULL,
   };
   const VkRenderPassCreateInfo rpi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = depth ? 2u : 1u,
      .pAttachments = atts,
      .subpassCount = 1,
      .pSubpasses = &sub,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &rpi, NULL, &render_pass), "render pass");

   const char *pairs_env = getenv("PROBE_PAIRS");
   const bool pairs = pairs_env && strcmp(pairs_env, "0");
   if (pairs && ((argc - 1) % 2) != 0)
      die("PROBE_PAIRS=1 needs an even number of SPIR-V arguments");

   size_t vsize = 0;
   uint32_t *vwords = NULL;
   VkShaderModule vertex = VK_NULL_HANDLE;
   if (!pairs) {
      vwords = read_spv(argv[1], &vsize);
      const VkShaderModuleCreateInfo vmi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = vsize,
         .pCode = vwords,
      };
      check(vkCreateShaderModule(device, &vmi, NULL, &vertex), "vertex module");
   }

   int failures = 0;
   for (int i = 2; i < argc; i += pairs ? 2 : 1) {
      if (pairs) {
         size_t psize = 0;
         uint32_t *pwords = read_spv(argv[i - 1], &psize);
         const VkShaderModuleCreateInfo pvmi = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = psize,
            .pCode = pwords,
         };
         check(vkCreateShaderModule(device, &pvmi, NULL, &vertex),
               "vertex module");
         free(pwords);
      }
      size_t fsize = 0;
      uint32_t *fwords = read_spv(argv[i], &fsize);
      const VkShaderModuleCreateInfo fmi = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = fsize,
         .pCode = fwords,
      };
      VkShaderModule fragment;
      check(vkCreateShaderModule(device, &fmi, NULL, &fragment),
            "fragment module");

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
      const VkPipelineVertexInputStateCreateInfo vi = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      const VkPipelineInputAssemblyStateCreateInfo ia = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
      const VkViewport vp = {.width = 32, .height = 32, .maxDepth = 1.0f};
      const VkRect2D scissor = {.extent = {32, 32}};
      const VkPipelineViewportStateCreateInfo vps = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
         .viewportCount = 1,
         .pViewports = &vp,
         .scissorCount = 1,
         .pScissors = &scissor};
      const VkPipelineRasterizationStateCreateInfo rs = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
         .polygonMode = VK_POLYGON_MODE_FILL,
         .cullMode = VK_CULL_MODE_BACK_BIT,
         .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
         .lineWidth = 1.0f};
      const VkPipelineMultisampleStateCreateInfo ms = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
         .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
      const VkPipelineDepthStencilStateCreateInfo ds = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
         .depthTestEnable = depth ? VK_TRUE : VK_FALSE,
         .depthWriteEnable = depth ? VK_TRUE : VK_FALSE,
         .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
         .maxDepthBounds = 1.0f};
      const VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xf};
      const VkPipelineColorBlendStateCreateInfo cb = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
         .attachmentCount = 1,
         .pAttachments = &cba};
      const VkGraphicsPipelineCreateInfo gpi = {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .stageCount = 2,
         .pStages = stages,
         .pVertexInputState = &vi,
         .pInputAssemblyState = &ia,
         .pViewportState = &vps,
         .pRasterizationState = &rs,
         .pMultisampleState = &ms,
         .pDepthStencilState = &ds,
         .pColorBlendState = &cb,
         .layout = layout,
         .renderPass = render_pass,
      };

      VkPipeline pipeline = VK_NULL_HANDLE;
      const VkResult r = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                                   &gpi, NULL, &pipeline);
      printf("%-52s %s (VkResult %d)\n", argv[i],
             r == VK_SUCCESS ? "ENCODES" : "REFUSED", r);
      fflush(stdout);
      if (r != VK_SUCCESS || pipeline == VK_NULL_HANDLE)
         ++failures;
      if (pipeline != VK_NULL_HANDLE)
         vkDestroyPipeline(device, pipeline, NULL);
      vkDestroyShaderModule(device, fragment, NULL);
      free(fwords);
      if (pairs) {
         vkDestroyShaderModule(device, vertex, NULL);
         vertex = VK_NULL_HANDLE;
      }
   }

   if (vertex != VK_NULL_HANDLE)
      vkDestroyShaderModule(device, vertex, NULL);
   free(vwords);
   vkDestroyRenderPass(device, render_pass, NULL);
   vkDestroyPipelineLayout(device, layout, NULL);
   vkDestroyDescriptorSetLayout(device, set_layout, NULL);
   vkDestroyDevice(device, NULL);
   vkDestroyInstance(instance, NULL);
   return failures;
}
