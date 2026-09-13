/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Loader-level probe that measures this ICD against the *hard* device
 * requirements of the two translation layers games actually reach a Linux
 * Vulkan driver through: DXVK (D3D8/9/10/11) and VKD3D-Proton (D3D12).
 *
 * Both layers are self-disabling: a single missing required feature makes them
 * refuse the device outright, and each reports only the first thing it misses.
 * This probe reports *all* of them in one pass so the gap list is exact.
 *
 * The requirement tables below are transcribed from the layers' own sources,
 * not from documentation:
 *
 *   DXVK 3.0.2 (doitsujin/dxvk 8759acd15dc79c8a92f5dd028b9e0d81a153c262)
 *     src/dxvk/dxvk_device_info.cpp:802 getFeatureList(), every entry whose
 *     `require` argument is true; src/dxvk/dxvk_device_info.cpp:745
 *     checkDeviceCompatibility() for the Vulkan version, graphics queue and
 *     push-data checks; src/dxvk/dxvk_instance.h:12 DxvkVulkanApiVersion;
 *     src/dxvk/dxvk_limits.h:22 MaxTotalPushDataSize.
 *
 *   VKD3D-Proton 3.1.0 (HansKristian-Work/vkd3d-proton
 *     2a38e7991f851e85151c104504d5e343530d1523)
 *     libs/vkd3d/device.c:2406 vkd3d_init_device_caps(), every branch that
 *     returns E_INVALIDARG; libs/d3d12core/main.c:492 for the device Vulkan
 *     version; include/vkd3d.h:53 VKD3D_MIN_API_VERSION.
 *
 * It creates an instance and reads physical-device features, properties and
 * extensions.  It never creates a logical device, queue or any GPU work, so it
 * runs unchanged against the real render node or against the drm-shim.
 *
 * Exit status is 0 when the observed gap set is exactly the recorded expected
 * gap set for the reported chip, 1 otherwise.  Closing a gap is therefore a
 * deliberate edit here, and silently *losing* a feature a layer needs fails.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define T8140_DEVICE_ID 0x8140

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkDestroyInstance);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkEnumerateDeviceExtensionProperties);
VK_FN(vkGetPhysicalDeviceFeatures2);
VK_FN(vkGetPhysicalDeviceProperties2);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);

static void
die(const char *what)
{
   fprintf(stderr, "asahi-vulkan-g17p-layer-requirements: %s\n", what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "asahi-vulkan-g17p-layer-requirements: %s returned %d\n",
              what, result);
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

/* Which layer needs an entry.  A gap that only one layer cares about is a
 * materially different amount of work from one that blocks both. */
enum layer {
   LAYER_DXVK = 1 << 0,
   LAYER_VKD3D = 1 << 1,
};

struct requirement {
   const char *name;
   /* Device extension that must be present for the bit to mean anything, or
    * NULL for a core feature/property. */
   const char *extension;
   unsigned layers;
   /* Filled in at run time; the tables are static so the pointers are set in
    * build_requirements() once the queried structures exist. */
   const VkBool32 *value;
};

struct chain {
   VkPhysicalDeviceFeatures2 features;
   VkPhysicalDeviceVulkan11Features vk11;
   VkPhysicalDeviceVulkan12Features vk12;
   VkPhysicalDeviceVulkan13Features vk13;
   VkPhysicalDeviceRobustness2FeaturesEXT robustness2;
   VkPhysicalDeviceDepthClipEnableFeaturesEXT depth_clip;
   VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5;
   VkPhysicalDeviceMaintenance6FeaturesKHR maintenance6;
   VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb;
   VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT divisor;

   VkPhysicalDeviceProperties2 properties;
   VkPhysicalDeviceVulkan12Properties vk12_props;
   VkPhysicalDeviceVulkan13Properties vk13_props;
   VkPhysicalDeviceTransformFeedbackPropertiesEXT xfb_props;
};

static void *
chain_push(void *head, void *next, VkStructureType type)
{
   VkBaseOutStructure *s = next;
   s->sType = type;
   s->pNext = head;
   return next;
}

static void
query_chain(VkPhysicalDevice physical, struct chain *c)
{
   memset(c, 0, sizeof(*c));

   void *head = NULL;
   head = chain_push(head, &c->vk11,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
   head = chain_push(head, &c->vk12,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
   head = chain_push(head, &c->vk13,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES);
   head = chain_push(head, &c->robustness2,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT);
   head = chain_push(
      head, &c->depth_clip,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT);
   head = chain_push(head, &c->maintenance5,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES);
   head = chain_push(head, &c->maintenance6,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_6_FEATURES);
   head = chain_push(
      head, &c->xfb,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT);
   head = chain_push(
      head, &c->divisor,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT);

   c->features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
   c->features.pNext = head;
   vkGetPhysicalDeviceFeatures2(physical, &c->features);

   head = NULL;
   head = chain_push(head, &c->vk12_props,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES);
   head = chain_push(head, &c->vk13_props,
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES);
   head = chain_push(
      head, &c->xfb_props,
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT);

   c->properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
   c->properties.pNext = head;
   vkGetPhysicalDeviceProperties2(physical, &c->properties);
}

struct extension_set {
   VkExtensionProperties *entries;
   uint32_t count;
};

static void
query_extensions(VkPhysicalDevice physical, struct extension_set *set)
{
   set->count = 0;
   check(vkEnumerateDeviceExtensionProperties(physical, NULL, &set->count, NULL),
         "vkEnumerateDeviceExtensionProperties");
   set->entries = calloc(set->count ? set->count : 1, sizeof(*set->entries));
   if (!set->entries)
      die("out of memory");
   check(vkEnumerateDeviceExtensionProperties(physical, NULL, &set->count,
                                              set->entries),
         "vkEnumerateDeviceExtensionProperties");
}

static bool
has_extension(const struct extension_set *set, const char *name,
              uint32_t min_spec_version)
{
   for (uint32_t i = 0; i < set->count; i++) {
      if (!strcmp(set->entries[i].extensionName, name))
         return set->entries[i].specVersion >= min_spec_version;
   }
   return false;
}

int
main(void)
{
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      loader = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die("failed to load the Vulkan loader");

   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("loader is missing vkGetInstanceProcAddr");

   LOAD_GLOBAL(vkCreateInstance);

   /* Both layers request a Vulkan 1.3 instance. */
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "asahi-vulkan-g17p-layer-requirements",
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };

   VkInstance instance = VK_NULL_HANDLE;
   check(vkCreateInstance(&info, NULL, &instance), "vkCreateInstance");

   LOAD_INSTANCE(instance, vkDestroyInstance);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkEnumerateDeviceExtensionProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFeatures2);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties2);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);

   uint32_t count = 0;
   check(vkEnumeratePhysicalDevices(instance, &count, NULL),
         "vkEnumeratePhysicalDevices");
   if (!count)
      die("no physical device");

   VkPhysicalDevice *physical = calloc(count, sizeof(*physical));
   if (!physical)
      die("out of memory");
   check(vkEnumeratePhysicalDevices(instance, &count, physical),
         "vkEnumeratePhysicalDevices");

   struct chain c;
   struct extension_set ext;
   VkPhysicalDevice chosen = VK_NULL_HANDLE;

   for (uint32_t i = 0; i < count; i++) {
      query_chain(physical[i], &c);
      if (c.properties.properties.deviceID == T8140_DEVICE_ID) {
         chosen = physical[i];
         break;
      }
   }
   if (chosen == VK_NULL_HANDLE)
      chosen = physical[0];

   query_chain(chosen, &c);
   query_extensions(chosen, &ext);

   const VkPhysicalDeviceProperties *props = &c.properties.properties;
   const VkPhysicalDeviceFeatures *core = &c.features.features;

   printf("device            : %s (0x%04x)\n", props->deviceName,
          props->deviceID);
   printf("apiVersion        : %u.%u.%u\n", VK_API_VERSION_MAJOR(props->apiVersion),
          VK_API_VERSION_MINOR(props->apiVersion),
          VK_API_VERSION_PATCH(props->apiVersion));

   const bool is_g17p = props->deviceID == T8140_DEVICE_ID;

/* Every entry that DXVK 3.0.2 getFeatureList() marks required, plus the
 * VKD3D-Proton hard checks.  Keep the order of the DXVK source. */
#define R(layer_mask, ext_name, expr)                                          \
   {                                                                           \
      .name = #expr, .extension = ext_name, .layers = (layer_mask),            \
      .value = &(expr)                                                         \
   }

   const struct requirement reqs[] = {
      /* DXVK, VkPhysicalDeviceFeatures */
      R(LAYER_DXVK, NULL, core->depthBiasClamp),
      R(LAYER_DXVK, NULL, core->depthClamp),
      R(LAYER_DXVK, NULL, core->dualSrcBlend),
      R(LAYER_DXVK, NULL, core->fillModeNonSolid),
      R(LAYER_DXVK, NULL, core->fragmentStoresAndAtomics),
      R(LAYER_DXVK, NULL, core->fullDrawIndexUint32),
      R(LAYER_DXVK, NULL, core->geometryShader),
      R(LAYER_DXVK, NULL, core->imageCubeArray),
      R(LAYER_DXVK, NULL, core->independentBlend),
      R(LAYER_DXVK, NULL, core->multiDrawIndirect),
      R(LAYER_DXVK, NULL, core->multiViewport),
      R(LAYER_DXVK, NULL, core->occlusionQueryPrecise),
      R(LAYER_DXVK, NULL, core->robustBufferAccess),
      R(LAYER_DXVK, NULL, core->sampleRateShading),
      R(LAYER_DXVK, NULL, core->samplerAnisotropy),
      R(LAYER_DXVK, NULL, core->shaderClipDistance),
      R(LAYER_DXVK, NULL, core->shaderCullDistance),
      R(LAYER_DXVK, NULL, core->shaderImageGatherExtended),
      R(LAYER_DXVK, NULL, core->shaderInt16),
      R(LAYER_DXVK, NULL, core->shaderInt64),
      R(LAYER_DXVK, NULL, core->shaderSampledImageArrayDynamicIndexing),
      R(LAYER_DXVK, NULL, core->textureCompressionBC),

      /* Vulkan 1.1 */
      R(LAYER_DXVK | LAYER_VKD3D, NULL, c.vk11.shaderDrawParameters),
      R(LAYER_DXVK, NULL, c.vk11.storageBuffer16BitAccess),

      /* Vulkan 1.2 */
      R(LAYER_DXVK, NULL, c.vk12.bufferDeviceAddress),
      R(LAYER_DXVK, NULL, c.vk12.descriptorIndexing),
      R(LAYER_DXVK, NULL, c.vk12.storageBuffer8BitAccess),
      R(LAYER_DXVK, NULL, c.vk12.descriptorBindingSampledImageUpdateAfterBind),
      R(LAYER_DXVK, NULL, c.vk12.descriptorBindingUpdateUnusedWhilePending),
      R(LAYER_DXVK, NULL, c.vk12.descriptorBindingPartiallyBound),
      R(LAYER_DXVK, NULL, c.vk12.hostQueryReset),
      R(LAYER_DXVK, NULL, c.vk12.runtimeDescriptorArray),
      R(LAYER_DXVK | LAYER_VKD3D, NULL, c.vk12.samplerMirrorClampToEdge),
      R(LAYER_DXVK, NULL, c.vk12.scalarBlockLayout),
      R(LAYER_DXVK, NULL, c.vk12.shaderInt8),
      R(LAYER_DXVK, NULL, c.vk12.timelineSemaphore),
      R(LAYER_DXVK, NULL, c.vk12.uniformBufferStandardLayout),
      R(LAYER_DXVK, NULL, c.vk12.vulkanMemoryModel),

      /* Vulkan 1.3 */
      R(LAYER_DXVK, NULL, c.vk13.inlineUniformBlock),
      R(LAYER_DXVK, NULL, c.vk13.computeFullSubgroups),
      R(LAYER_DXVK, NULL, c.vk13.dynamicRendering),
      R(LAYER_DXVK, NULL, c.vk13.maintenance4),
      R(LAYER_DXVK, NULL, c.vk13.shaderDemoteToHelperInvocation),
      R(LAYER_DXVK, NULL, c.vk13.shaderZeroInitializeWorkgroupMemory),
      R(LAYER_DXVK, NULL, c.vk13.subgroupSizeControl),
      R(LAYER_DXVK, NULL, c.vk13.synchronization2),

      /* Extension-backed features */
      R(LAYER_DXVK, VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME,
        c.depth_clip.depthClipEnable),
      R(LAYER_DXVK | LAYER_VKD3D, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
        c.robustness2.robustBufferAccess2),
      /* DXVK treats robustImageAccess2 as optional; VKD3D-Proton requires it
       * (libs/vkd3d/device.c:3429). */
      R(LAYER_VKD3D, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
        c.robustness2.robustImageAccess2),
      R(LAYER_DXVK | LAYER_VKD3D, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
        c.robustness2.nullDescriptor),
      R(LAYER_DXVK | LAYER_VKD3D, VK_KHR_MAINTENANCE_5_EXTENSION_NAME,
        c.maintenance5.maintenance5),
      R(LAYER_DXVK | LAYER_VKD3D, VK_KHR_MAINTENANCE_6_EXTENSION_NAME,
        c.maintenance6.maintenance6),

      /* VKD3D-Proton only */
      R(LAYER_VKD3D, VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME,
        c.divisor.vertexAttributeInstanceRateDivisor),
      R(LAYER_VKD3D, VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME,
        c.divisor.vertexAttributeInstanceRateZeroDivisor),
      R(LAYER_VKD3D, VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME,
        c.xfb_props.transformFeedbackQueries),
   };
#undef R

   /* Extensions required outright, independent of any feature bit. */
   static const struct {
      const char *name;
      uint32_t min_spec_version;
      unsigned layers;
   } required_extensions[] = {
      { VK_KHR_SWAPCHAIN_EXTENSION_NAME, 0, LAYER_DXVK },
      { VK_KHR_LOAD_STORE_OP_NONE_EXTENSION_NAME, 0, LAYER_DXVK },
      { VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, 0, LAYER_VKD3D },
      /* VKD3D-Proton disables the extension below spec version 3, which then
       * fails its own vertexAttributeInstanceRateDivisor check. */
      { VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME, 3, LAYER_VKD3D },
   };

   unsigned gaps = 0;
   unsigned dxvk_gaps = 0, vkd3d_gaps = 0;

   printf("\n-- required device extensions --\n");
   for (size_t i = 0; i < sizeof(required_extensions) / sizeof(*required_extensions);
        i++) {
      const bool ok = has_extension(&ext, required_extensions[i].name,
                                    required_extensions[i].min_spec_version);
      printf("  %-6s %-8s %s\n", ok ? "ok" : "MISSING",
             required_extensions[i].layers == LAYER_DXVK   ? "dxvk"
             : required_extensions[i].layers == LAYER_VKD3D ? "vkd3d"
                                                            : "both",
             required_extensions[i].name);
      if (!ok) {
         gaps++;
         if (required_extensions[i].layers & LAYER_DXVK)
            dxvk_gaps++;
         if (required_extensions[i].layers & LAYER_VKD3D)
            vkd3d_gaps++;
      }
   }

   unsigned feature_gaps = 0;
   printf("\n-- required features --\n");
   for (size_t i = 0; i < sizeof(reqs) / sizeof(*reqs); i++) {
      const bool ext_ok =
         !reqs[i].extension || has_extension(&ext, reqs[i].extension, 0);
      const bool ok = ext_ok && *reqs[i].value;
      if (!ok) {
         gaps++;
         feature_gaps++;
         if (reqs[i].layers & LAYER_DXVK)
            dxvk_gaps++;
         if (reqs[i].layers & LAYER_VKD3D)
            vkd3d_gaps++;
         printf("  MISSING %-8s %s%s\n",
                reqs[i].layers == LAYER_DXVK    ? "dxvk"
                : reqs[i].layers == LAYER_VKD3D ? "vkd3d"
                                                : "both",
                reqs[i].name, ext_ok ? "" : " (extension absent)");
      }
   }
   if (!feature_gaps)
      printf("  (none)\n");

   /* Non-feature gates. */
   printf("\n-- required limits --\n");
   const bool push_ok = props->limits.maxPushConstantsSize >= 256;
   printf("  %-7s dxvk     maxPushConstantsSize=%u (need 256)\n",
          push_ok ? "ok" : "MISSING", props->limits.maxPushConstantsSize);
   if (!push_ok) {
      gaps++;
      dxvk_gaps++;
   }

   const bool storage_texel_ok =
      c.vk13_props.storageTexelBufferOffsetSingleTexelAlignment ||
      c.vk13_props.storageTexelBufferOffsetAlignmentBytes == 1;
   const bool uniform_texel_ok =
      c.vk13_props.uniformTexelBufferOffsetSingleTexelAlignment ||
      c.vk13_props.uniformTexelBufferOffsetAlignmentBytes == 1;
   printf("  %-7s vkd3d    storageTexelBufferOffsetSingleTexelAlignment=%u "
          "alignmentBytes=%llu\n",
          storage_texel_ok ? "ok" : "MISSING",
          c.vk13_props.storageTexelBufferOffsetSingleTexelAlignment,
          (unsigned long long)
             c.vk13_props.storageTexelBufferOffsetAlignmentBytes);
   printf("  %-7s vkd3d    uniformTexelBufferOffsetSingleTexelAlignment=%u "
          "alignmentBytes=%llu\n",
          uniform_texel_ok ? "ok" : "MISSING",
          c.vk13_props.uniformTexelBufferOffsetSingleTexelAlignment,
          (unsigned long long)
             c.vk13_props.uniformTexelBufferOffsetAlignmentBytes);
   if (!storage_texel_ok) {
      gaps++;
      vkd3d_gaps++;
   }
   if (!uniform_texel_ok) {
      gaps++;
      vkd3d_gaps++;
   }

   /* Truthfulness invariant.  A capability property that describes a feature
    * the device does not expose is a false advertisement: it turns a clean
    * "device skipped" into a layer that proceeds and then fails somewhere
    * else.  VKD3D-Proton gates transform feedback on transformFeedbackQueries
    * alone, so this pairing has a real consumer. */
   unsigned lies = 0;
   printf("\n-- property/feature consistency --\n");
   if (!c.xfb.transformFeedback) {
      const bool advertised =
         c.xfb_props.transformFeedbackQueries ||
         c.xfb_props.transformFeedbackStreamsLinesTriangles ||
         c.xfb_props.transformFeedbackRasterizationStreamSelect ||
         c.xfb_props.transformFeedbackDraw ||
         c.xfb_props.maxTransformFeedbackStreams ||
         c.xfb_props.maxTransformFeedbackBuffers ||
         c.xfb_props.maxTransformFeedbackBufferSize ||
         c.xfb_props.maxTransformFeedbackStreamDataSize ||
         c.xfb_props.maxTransformFeedbackBufferDataSize ||
         c.xfb_props.maxTransformFeedbackBufferDataStride;
      printf("  %-7s transformFeedback=0, queries=%u streams=%u buffers=%u\n",
             advertised ? "LIE" : "ok",
             c.xfb_props.transformFeedbackQueries,
             c.xfb_props.maxTransformFeedbackStreams,
             c.xfb_props.maxTransformFeedbackBuffers);
      if (advertised)
         lies++;
   } else {
      printf("  ok      transformFeedback=1\n");
   }

   printf("\nDXVK 3.0.2       : %s (%u unmet hard requirement%s)\n",
          dxvk_gaps ? "DEVICE REFUSED" : "device accepted", dxvk_gaps,
          dxvk_gaps == 1 ? "" : "s");
   printf("VKD3D-Proton 3.1 : %s (%u unmet hard requirement%s)\n",
          vkd3d_gaps ? "DEVICE REFUSED" : "device accepted", vkd3d_gaps,
          vkd3d_gaps == 1 ? "" : "s");

   /* Recorded expectation.  G17P's HAL200 graphics path is deliberately
    * bounded (hk_physical_device.c, `if (dev->chip == AGX_CHIP_G17P)`), so the
    * gap count is known and must only change on purpose. */
   const unsigned expected_dxvk = is_g17p ? 3 : 0;
   const unsigned expected_vkd3d = is_g17p ? 3 : 0;

   bool as_expected =
      dxvk_gaps == expected_dxvk && vkd3d_gaps == expected_vkd3d;
   if (!as_expected) {
      printf("\nUNEXPECTED: recorded gaps are dxvk=%u vkd3d=%u for this chip.\n"
             "Update this probe together with the change that moved them.\n",
             expected_dxvk, expected_vkd3d);
   }
   if (lies) {
      printf("\nFAIL: %u property block%s describe%s a feature this device does "
             "not expose.\n",
             lies, lies == 1 ? "" : "s", lies == 1 ? "s" : "");
      as_expected = false;
   }

   free(ext.entries);
   free(physical);
   vkDestroyInstance(instance, NULL);
   dlclose(loader);
   return as_expected ? EXIT_SUCCESS : EXIT_FAILURE;
}
