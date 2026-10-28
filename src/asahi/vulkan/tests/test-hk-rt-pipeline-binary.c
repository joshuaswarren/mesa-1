/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* CPU control of the production RT creation entry point.  Translation and
 * megakernel construction are real; only GPU shader compilation/upload is
 * replaced.  These checks do not execute shaders or establish conformance.
 */
#include "spirv/spirv.h"
#include "util/blob.h"
#include "util/set.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_ray_tracing_pipeline.h"
#include "hk_shader.h"
#include "nir.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_instance.h"
#include "vk_nir.h"
#include "vk_pipeline_cache.h"
#include "vk_pipeline_layout.h"
#include "vk_shader_module.h"

#include <stdio.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __func__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)
#define FAIL  VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT
#define LIB   VK_PIPELINE_CREATE_LIBRARY_BIT_KHR
#define EARLY VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT

static unsigned translations, backends, shader_destroys, live_nir;
static nir_shader *tracked_nir[64];
static VkResult backend_result = VK_SUCCESS;
static unsigned fail_translation;

VkResult __real_vk_pipeline_shader_stage_to_nir(
   struct vk_device *, VkPipelineCreateFlags2KHR,
   const VkPipelineShaderStageCreateInfo *, const struct spirv_to_nir_options *,
   const nir_shader_compiler_options *, void *, nir_shader **);

VkResult __wrap_vk_pipeline_shader_stage_to_nir(
   struct vk_device *dev, VkPipelineCreateFlags2KHR flags,
   const VkPipelineShaderStageCreateInfo *stage,
   const struct spirv_to_nir_options *spirv_options,
   const nir_shader_compiler_options *nir_options, void *ctx, nir_shader **out);

VkResult
__wrap_vk_pipeline_shader_stage_to_nir(
   struct vk_device *dev, VkPipelineCreateFlags2KHR flags,
   const VkPipelineShaderStageCreateInfo *stage,
   const struct spirv_to_nir_options *spirv_options,
   const nir_shader_compiler_options *nir_options, void *ctx, nir_shader **out)
{
   translations++;
   if (fail_translation == translations)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = __real_vk_pipeline_shader_stage_to_nir(
      dev, flags, stage, spirv_options, nir_options, ctx, out);
   if (result == VK_SUCCESS) {
      unsigned i;
      for (i = 0; i < ARRAY_SIZE(tracked_nir); i++) {
         if (!tracked_nir[i]) {
            tracked_nir[i] = *out;
            live_nir++;
            break;
         }
      }
      CHECK(i < ARRAY_SIZE(tracked_nir));
   }
   return result;
}

void __real_ralloc_free(void *);
void __wrap_ralloc_free(void *ptr);

void
__wrap_ralloc_free(void *ptr)
{
   if (ptr) {
      for (unsigned i = 0; i < ARRAY_SIZE(tracked_nir); i++) {
         if (tracked_nir[i] == ptr) {
            tracked_nir[i] = NULL;
            live_nir--;
         }
      }
   }
   __real_ralloc_free(ptr);
}

VkResult __wrap_hk_compile_shader(
   struct hk_device *dev, struct vk_shader_compile_info *info,
   const struct vk_graphics_pipeline_state *state,
   const struct vk_features *features, const VkAllocationCallbacks *alloc,
   struct hk_api_shader **out);

VkResult
__wrap_hk_compile_shader(struct hk_device *dev,
                         struct vk_shader_compile_info *info,
                         const struct vk_graphics_pipeline_state *state,
                         const struct vk_features *features,
                         const VkAllocationCallbacks *alloc,
                         struct hk_api_shader **out)
{
   backends++;
   ralloc_free(info->nir);
   if (backend_result != VK_SUCCESS)
      return backend_result;
   *out = calloc(1, sizeof(**out));
   CHECK(*out);
   return VK_SUCCESS;
}

void __wrap_hk_api_shader_destroy(struct vk_device *dev,
                                  struct vk_shader *shader,
                                  const VkAllocationCallbacks *alloc);

void
__wrap_hk_api_shader_destroy(struct vk_device *dev, struct vk_shader *shader,
                             const VkAllocationCallbacks *alloc)
{
   shader_destroys++;
   free(container_of(shader, struct hk_api_shader, vk));
}

/* Empty raygen SPIR-V, assembled from the public SPIR-V opcodes. */
static const uint32_t raygen[] = {
   SpvMagicNumber,
   0x00010400,
   0,
   5,
   0,
   (2 << 16) | SpvOpCapability,
   SpvCapabilityShader,
   (2 << 16) | SpvOpCapability,
   SpvCapabilityRayTracingKHR,
   (6 << 16) | SpvOpExtension,
   0x5f565053,
   0x5f52484b,
   0x5f796172,
   0x63617274,
   0x00676e69,
   (3 << 16) | SpvOpMemoryModel,
   SpvAddressingModelLogical,
   SpvMemoryModelGLSL450,
   (5 << 16) | SpvOpEntryPoint,
   SpvExecutionModelRayGenerationKHR,
   3,
   0x6e69616d,
   0,
   (2 << 16) | SpvOpTypeVoid,
   1,
   (3 << 16) | SpvOpTypeFunction,
   2,
   1,
   (5 << 16) | SpvOpFunction,
   1,
   3,
   SpvFunctionControlMaskNone,
   2,
   (2 << 16) | SpvOpLabel,
   4,
   (1 << 16) | SpvOpReturn,
   (1 << 16) | SpvOpFunctionEnd,
};

static struct vk_instance instance;
static struct hk_device dev;
static struct hk_physical_device pdev;
static VkDevice device;
static struct vk_pipeline_layout *layout;
static VkShaderModule module;
static VkPipelineShaderStageCreateInfo stage;
static VkRayTracingShaderGroupCreateInfoKHR group = {
   .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
   .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
   .generalShader = 0,
   .closestHitShader = VK_SHADER_UNUSED_KHR,
   .anyHitShader = VK_SHADER_UNUSED_KHR,
   .intersectionShader = VK_SHADER_UNUSED_KHR,
};
static unsigned pipeline_allocs, pipeline_frees;

static void *VKAPI_CALL
alloc_pipeline(void *data, size_t size, size_t alignment,
               VkSystemAllocationScope scope)
{
   void *ptr = NULL;
   if (posix_memalign(&ptr, MAX2(alignment, sizeof(void *)), size) != 0)
      return NULL;
   pipeline_allocs++;
   return ptr;
}
static void VKAPI_CALL
free_pipeline(void *data, void *ptr)
{
   if (ptr)
      pipeline_frees++;
   free(ptr);
}
static const VkAllocationCallbacks alloc = {
   .pfnAllocation = alloc_pipeline,
   .pfnFree = free_pipeline,
};

static void VKAPI_CALL
get_properties(VkPhysicalDevice physical, VkPhysicalDeviceProperties *props)
{
   memset(props, 0, sizeof(*props));
}

static VkRayTracingPipelineCreateInfoKHR
info(VkPipelineCreateFlags flags)
{
   return (VkRayTracingPipelineCreateInfoKHR){
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .flags = flags,
      .stageCount = 1,
      .pStages = &stage,
      .groupCount = 1,
      .pGroups = &group,
      .maxPipelineRayRecursionDepth = 1,
      .layout = vk_pipeline_layout_to_handle(layout),
   };
}

static void
destroy(VkPipeline pipeline)
{
   if (pipeline) {
      struct hk_ray_tracing_pipeline *p =
         hk_ray_tracing_pipeline_from_handle(pipeline);
      p->base.ops->destroy(&dev.vk, &p->base, &alloc);
   }
}

static VkResult
create(VkPipelineCache cache, const VkRayTracingPipelineCreateInfoKHR *ci,
       VkPipeline *pipeline)
{
   *pipeline = (VkPipeline)(uintptr_t)0xbad;
   return hk_CreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, cache, 1, ci,
                                          &alloc, pipeline);
}
static unsigned serialized, restored;
static bool
fixture_serialize(struct vk_device *device, const struct vk_shader *shader,
                  struct blob *blob)
{
   ++serialized;
   blob_write_uint32(blob, 0x1234abcd);
   return !blob->out_of_memory;
}
static const struct vk_shader_ops fixture_shader_ops = {.serialize =
                                                           fixture_serialize};
static VkResult
fixture_deserialize(struct vk_device *device, struct blob_reader *blob,
                    uint32_t version, const VkAllocationCallbacks *alloc,
                    struct vk_shader **out)
{
   CHECK(blob_read_uint32(blob) == 0x1234abcd);
   ++restored;
   struct hk_api_shader *shader = calloc(1, sizeof(*shader));
   CHECK(shader);
   shader->vk.ops = &fixture_shader_ops;
   *out = &shader->vk;
   return VK_SUCCESS;
}
static const struct vk_device_shader_ops fixture_device_ops = {
   .deserialize = fixture_deserialize};

static VkPipelineBinaryKHR
capture(VkPipeline pipeline)
{
   VkPipelineBinaryCreateInfoKHR ci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
      .pipeline = pipeline};
   VkPipelineBinaryHandlesInfoKHR out = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
   CHECK(hk_CreatePipelineBinariesKHR(device, &ci, &alloc, &out) == VK_SUCCESS);
   CHECK(out.pipelineBinaryCount == 1);
   VkPipelineBinaryKHR binary = VK_NULL_HANDLE;
   out.pPipelineBinaries = &binary;
   out.pipelineBinaryCount = 0;
   CHECK(hk_CreatePipelineBinariesKHR(device, &ci, &alloc, &out) ==
         VK_INCOMPLETE);
   out.pipelineBinaryCount = 1;
   CHECK(hk_CreatePipelineBinariesKHR(device, &ci, &alloc, &out) == VK_SUCCESS);
   CHECK(out.pipelineBinaryCount == 1 && binary);
   return binary;
}

/* Intercept the common entrypoints for negative-chip controls. A sentinel
 * proves delegation without asking the common runtime to consume an HK RT
 * object (the very object-layout bug this G17P port repairs). */
static unsigned classic_delegations;
VkResult __real_vk_common_CreatePipelineBinariesKHR(
   VkDevice, const VkPipelineBinaryCreateInfoKHR *,
   const VkAllocationCallbacks *, VkPipelineBinaryHandlesInfoKHR *);
VkResult __wrap_vk_common_CreatePipelineBinariesKHR(
   VkDevice d, const VkPipelineBinaryCreateInfoKHR *ci,
   const VkAllocationCallbacks *a, VkPipelineBinaryHandlesInfoKHR *out);
VkResult
__wrap_vk_common_CreatePipelineBinariesKHR(
   VkDevice d, const VkPipelineBinaryCreateInfoKHR *ci,
   const VkAllocationCallbacks *a, VkPipelineBinaryHandlesInfoKHR *out)
{
   if (dev.dev.chip != AGX_CHIP_G17P && !agx_chip_g17_m5(dev.dev.chip)) {
      classic_delegations++;
      return VK_NOT_READY;
   }
   return __real_vk_common_CreatePipelineBinariesKHR(d, ci, a, out);
}

VkResult __real_vk_common_GetPipelineKeyKHR(
   VkDevice, const VkPipelineCreateInfoKHR *, VkPipelineBinaryKeyKHR *);
VkResult __wrap_vk_common_GetPipelineKeyKHR(
   VkDevice d, const VkPipelineCreateInfoKHR *ci, VkPipelineBinaryKeyKHR *key);
VkResult
__wrap_vk_common_GetPipelineKeyKHR(
   VkDevice d, const VkPipelineCreateInfoKHR *ci, VkPipelineBinaryKeyKHR *key)
{
   if (dev.dev.chip != AGX_CHIP_G17P && !agx_chip_g17_m5(dev.dev.chip)) {
      classic_delegations++;
      return VK_NOT_READY;
   }
   return __real_vk_common_GetPipelineKeyKHR(d, ci, key);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   vk_object_base_init(&dev.vk, &dev.vk.base, VK_OBJECT_TYPE_DEVICE);
   vk_object_base_init(NULL, &pdev.vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   instance.app_info.api_version = VK_API_VERSION_1_4;
   pdev.vk.instance = &instance;
   dev.vk.physical = &pdev.vk;
   dev.dev.chip = AGX_CHIP_G17P;
   dev.vk.alloc = *vk_default_allocator();
   dev.vk.robustness_state = (struct vk_pipeline_robustness_state){
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
   };
   dev.vk.enabled_features.rayTracingPipeline = true;
   dev.vk.enabled_features.shaderModuleIdentifier = true;
   dev.vk.enabled_features.shaderInt64 = true;
   pdev.vk.supported_features = dev.vk.enabled_features;
   pdev.vk.dispatch_table.GetPhysicalDeviceProperties = get_properties;
   pdev.vk.properties.subgroupSize = 32;
   pdev.vk.properties.minSubgroupSize = 32;
   pdev.vk.properties.maxSubgroupSize = 32;
   device = hk_device_to_handle(&dev);
   for (enum agx_chip chip = AGX_CHIP_G13G; chip < AGX_CHIP_UNKNOWN; ++chip) {
      if (chip == AGX_CHIP_G17P || agx_chip_g17_m5(chip))
         continue;
      dev.dev.chip = chip;
      VkRayTracingPipelineCreateInfoKHR rt = {
         .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
      VkPipelineCreateInfoKHR ci = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR, .pNext = &rt};
      VkPipelineBinaryCreateInfoKHR bi = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
         .pPipelineCreateInfo = &ci};
      VkPipelineBinaryHandlesInfoKHR out = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
      VkPipelineBinaryKeyKHR key = {.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
      unsigned before = classic_delegations;
      CHECK(hk_CreatePipelineBinariesKHR(device, &bi, NULL, &out) == VK_NOT_READY);
      CHECK(hk_GetPipelineKeyKHR(device, &ci, &key) == VK_NOT_READY);
      CHECK(classic_delegations == before + 2);
   }
   dev.dev.chip = AGX_CHIP_G17P;
   layout = vk_pipeline_layout_zalloc(
      &dev.vk, sizeof(*layout),
      &(VkPipelineLayoutCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      });
   CHECK(layout);
   VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(raygen),
      .pCode = raygen,
   };
   CHECK(vk_common_CreateShaderModule(device, &module_info, NULL, &module) ==
         VK_SUCCESS);
   stage = (VkPipelineShaderStageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
      .module = module,
      .pName = "main",
   };

   dev.vk.shader_ops = &fixture_device_ops;
   VkPipelineBinaryKHR binaries[2];
   const VkShaderStageFlagBits stages[] = {
      VK_SHADER_STAGE_RAYGEN_BIT_KHR,       VK_SHADER_STAGE_MISS_BIT_KHR,
      VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,  VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
      VK_SHADER_STAGE_INTERSECTION_BIT_KHR, VK_SHADER_STAGE_CALLABLE_BIT_KHR,
   };
   const SpvExecutionModel models[] = {
      SpvExecutionModelRayGenerationKHR, SpvExecutionModelMissKHR,
      SpvExecutionModelClosestHitKHR,    SpvExecutionModelAnyHitKHR,
      SpvExecutionModelIntersectionKHR,  SpvExecutionModelCallableKHR,
   };
   for (unsigned st = 0; st < ARRAY_SIZE(stages); ++st) {
      uint32_t code[ARRAY_SIZE(raygen)];
      memcpy(code, raygen, sizeof(code));
      for (unsigned i = 5; i < ARRAY_SIZE(code); i += code[i] >> 16)
         if ((code[i] & 0xffff) == SpvOpEntryPoint)
            code[i + 1] = models[st];
      vk_common_DestroyShaderModule(device, module, NULL);
      module_info.pCode = code;
      CHECK(vk_common_CreateShaderModule(device, &module_info, NULL, &module) ==
            VK_SUCCESS);
      stage.module = module;
      stage.stage = stages[st];
      group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
      group.generalShader = 0;
      group.closestHitShader = group.anyHitShader = group.intersectionShader =
         VK_SHADER_UNUSED_KHR;
      if (st >= 2 && st <= 4) {
         group.type =
            st == 4 ? VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR
                    : VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
         group.generalShader = VK_SHADER_UNUSED_KHR;
         if (st == 2)
            group.closestHitShader = 0;
         if (st == 3)
            group.anyHitShader = 0;
         if (st == 4)
            group.intersectionShader = 0;
      }
      for (unsigned library = 0; library < 2; ++library) {
         VkRayTracingPipelineCreateInfoKHR ci = info(library ? LIB : 0);
         VkPipeline pipeline;
         CHECK(create(VK_NULL_HANDLE, &ci, &pipeline) == VK_SUCCESS);
         struct hk_ray_tracing_pipeline *p =
            hk_ray_tracing_pipeline_from_handle(pipeline);
         if (p->shader)
            p->shader->vk.ops = &fixture_shader_ops;
         VkPipelineBinaryKeyKHR key = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
         VkPipelineCreateInfoKHR wrapper = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR,
            .pNext = &ci};
         CHECK(hk_GetPipelineKeyKHR(device, &wrapper, &key) == VK_SUCCESS);
         CHECK(key.keySize == 32 && !memcmp(key.key, p->binary_key, 32));
         binaries[library] = capture(pipeline);
         VkPipelineBinaryDataInfoKHR data_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR,
            .pipelineBinary = binaries[library]};
         size_t size = 0;
         CHECK(vk_common_GetPipelineBinaryDataKHR(device, &data_info, &key,
                                                  &size, NULL) == VK_SUCCESS);
         void *bytes = malloc(size);
         CHECK(bytes && size > 32);
         CHECK(vk_common_GetPipelineBinaryDataKHR(device, &data_info, &key,
                                                  &size, bytes) == VK_SUCCESS);
         VkPipelineBinaryDataKHR data = {.dataSize = size, .pData = bytes};
         VkPipelineBinaryKeysAndDataKHR keys = {.binaryCount = 1,
                                                .pPipelineBinaryKeys = &key,
                                                .pPipelineBinaryData = &data};
         VkPipelineBinaryCreateInfoKHR imported = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
            .pKeysAndDataInfo = &keys};
         VkPipelineBinaryKHR copy;
         VkPipelineBinaryHandlesInfoKHR out = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR,
            .pipelineBinaryCount = 1,
            .pPipelineBinaries = &copy};
         CHECK(hk_CreatePipelineBinariesKHR(device, &imported, &alloc, &out) ==
               VK_SUCCESS);
         free(bytes);
         unsigned old_backends = backends, old_translations = translations;
         VkPipelineBinaryInfoKHR binary_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR,
            .binaryCount = 1,
            .pPipelineBinaries = &copy};
         ci.pNext = &binary_info;
         ci.pStages = NULL; /* Binaries restore stage NIR without touching
                               module inputs. */
         VkPipeline roundtrip;
         CHECK(create(VK_NULL_HANDLE, &ci, &roundtrip) == VK_SUCCESS);
         CHECK(backends == old_backends && translations == old_translations);
         struct hk_ray_tracing_pipeline *r =
            hk_ray_tracing_pipeline_from_handle(roundtrip);
         CHECK(r->stage_count == p->stage_count &&
               r->group_count == p->group_count);
         CHECK(!!r->shader == !library && r->base.stages == p->base.stages);
         CHECK(!memcmp(r->groups, p->groups,
                       p->group_count * sizeof(p->groups[0])));
         CHECK(r->stages[0]->nir->info.stage ==
               vk_to_mesa_shader_stage(stages[st]));
         /* A restored library can still supply NIR to a subsequent executable. */
         if (library) {
            VkPipelineLibraryCreateInfoKHR libs = {
               .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
               .libraryCount = 1,
               .pLibraries = &roundtrip};
            VkRayTracingPipelineCreateInfoKHR link = info(0);
            link.pLibraryInfo = &libs;
            link.stageCount = link.groupCount = 0;
            VkPipeline linked;
            CHECK(create(VK_NULL_HANDLE, &link, &linked) == VK_SUCCESS);
            destroy(linked);
            VkPipelineCreateInfoKHR w = {
               .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR,
               .pNext = &link};
            CHECK(hk_GetPipelineKeyKHR(device, &w, &key) == VK_SUCCESS);
            VkPipelineBinaryCreateInfoKHR miss = {
               .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
               .pPipelineCreateInfo = &w};
            out.pipelineBinaryCount = 99;
            CHECK(hk_CreatePipelineBinariesKHR(device, &miss, &alloc, &out) ==
                  VK_PIPELINE_BINARY_MISSING_KHR);
            CHECK(out.pipelineBinaryCount == 0);
         }
         destroy(roundtrip);
         destroy(pipeline);
         vk_common_DestroyPipelineBinaryKHR(device, copy, &alloc);
         vk_common_DestroyPipelineBinaryKHR(device, binaries[library], &alloc);
      }
   }
   CHECK(serialized == 6 && restored == 6);
   CHECK(live_nir == 0 && layout->ref_cnt == 1);
   vk_common_DestroyShaderModule(device, module, NULL);
   vk_pipeline_layout_unref(&dev.vk, layout);
   CHECK(pipeline_allocs == pipeline_frees);
   vk_object_base_finish(&dev.vk.base);
   vk_object_base_finish(&pdev.vk.base);
   glsl_type_singleton_decref();
   puts(
      "PASS: RT executable/library binary transport and recreation; no translation or backend on import");
   return 0;
}
