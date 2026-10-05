/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "d12_core_caps_query.h"
#include "d12_private.h"
#include "d12_raytracing_core.h"
#include "util/log.h"
#include "util/macros.h"
#include "agx_device.h"
#include <string.h>

void
d12_core_caps_init_agx(struct d12_core_caps *caps, const struct agx_device *dev)
{
   const bool legacy = dev->params.gpu_generation == 13 ||
                       dev->params.gpu_generation == 14;
   const struct d12_core_caps_device_info info = {
      .generation = legacy ? dev->params.gpu_generation :
                    dev->chip == AGX_CHIP_G17P ? 17 : 0,
      .active_cores = agx_get_num_cores(dev),
      .vm_start = dev->params.vm_start,
      .vm_end = dev->params.vm_end,
      .isolated_mmu = dev->vm_id != 0,
#ifndef _WIN32
      /* The native DRM winsys maps coherent shmem; its cache maintenance
       * hook is a no-op. Virtio and the Windows winsys need their own proof. */
      .cache_coherent_uma = !dev->is_virtio,
#endif
      /* The AGX2 backend lowers int64, preserves f16 and lowers wave ops.
       * The G17P backend has not qualified their complete DXIL contracts. */
      .compiler_wave_ops = legacy,
      .compiler_int64 = legacy,
      .compiler_native_16bit = legacy,
   };
   d12_core_caps_init(caps, &info);
}

HRESULT
d12_core_caps_query(const struct d12_core_caps *caps, D3D12_FEATURE feature,
                               void *data, UINT data_size)
{
   if (!data)
      return E_INVALIDARG;

   switch (feature) {
   case D3D12_FEATURE_D3D12_OPTIONS21: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS21 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;
      memset(opts, 0, sizeof(*opts));
      opts->ExecuteIndirectTier = D3D12_EXECUTE_INDIRECT_TIER_1_0;
      return S_OK;
   }

   case D3D12_FEATURE_DISPLAYABLE: {
      D3D12_FEATURE_DATA_DISPLAYABLE *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;
      memset(opts, 0, sizeof(*opts));
      return S_OK;
   }

   case D3D12_FEATURE_COMMAND_QUEUE_PRIORITY: {
      D3D12_FEATURE_DATA_COMMAND_QUEUE_PRIORITY *q = data;
      if (data_size != sizeof(*q))
         return E_INVALIDARG;
      if (q->CommandListType != D3D12_COMMAND_LIST_TYPE_DIRECT &&
          q->CommandListType != D3D12_COMMAND_LIST_TYPE_COMPUTE &&
          q->CommandListType != D3D12_COMMAND_LIST_TYPE_COPY)
         return E_INVALIDARG;
      q->PriorityForTypeIsSupported = q->Priority == D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
      return S_OK;
   }

   case D3D12_FEATURE_SHADER_CACHE: {
      D3D12_FEATURE_DATA_SHADER_CACHE *q = data;
      if (data_size != sizeof(*q))
         return E_INVALIDARG;
      q->SupportFlags = D3D12_SHADER_CACHE_SUPPORT_SINGLE_PSO;
      return S_OK;
   }

   case D3D12_FEATURE_EXISTING_HEAPS: {
      D3D12_FEATURE_DATA_EXISTING_HEAPS *q = data;
      if (data_size != sizeof(*q))
         return E_INVALIDARG;
      q->Supported = FALSE;
      return S_OK;
   }

   case D3D12_FEATURE_SERIALIZATION: {
      D3D12_FEATURE_DATA_SERIALIZATION *q = data;
      if (data_size != sizeof(*q) || q->NodeIndex >= caps->node_count)
         return E_INVALIDARG;
      q->HeapSerializationTier = D3D12_HEAP_SERIALIZATION_TIER_0;
      return S_OK;
   }

   case D3D12_FEATURE_CROSS_NODE: {
      D3D12_FEATURE_DATA_CROSS_NODE *q = data;
      if (data_size != sizeof(*q))
         return E_INVALIDARG;
      memset(q, 0, sizeof(*q));
      return S_OK;
   }

   case D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_SUPPORT: {
      D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_SUPPORT *q = data;
      if (data_size != sizeof(*q) || q->NodeIndex >= caps->node_count)
         return E_INVALIDARG;
      q->Support = D3D12_PROTECTED_RESOURCE_SESSION_SUPPORT_FLAG_NONE;
      return S_OK;
   }

   case D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPE_COUNT: {
      D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPE_COUNT *q = data;
      if (data_size != sizeof(*q) || q->NodeIndex >= caps->node_count)
         return E_INVALIDARG;
      q->Count = 0;
      return S_OK;
   }

   case D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPES: {
      D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPES *q = data;
      if (data_size != sizeof(*q) || q->NodeIndex >= caps->node_count || q->Count)
         return E_INVALIDARG;
      return S_OK;
   }

   case D3D12_FEATURE_QUERY_META_COMMAND:
      if (data_size != sizeof(D3D12_FEATURE_DATA_QUERY_META_COMMAND))
         return E_INVALIDARG;
      return E_INVALIDARG; /* no meta commands are enumerated */

   case D3D12_FEATURE_PLACED_RESOURCE_SUPPORT_INFO: {
      D3D12_FEATURE_DATA_PLACED_RESOURCE_SUPPORT_INFO *q = data;
      if (data_size != sizeof(*q))
         return E_INVALIDARG;
      q->Supported = FALSE; /* tight placement contract not implemented */
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_TIGHT_ALIGNMENT: {
      D3D12_FEATURE_DATA_TIGHT_ALIGNMENT *q = data;
      if (data_size != sizeof(*q))
         return E_INVALIDARG;
      memset(q, 0, sizeof(*q));
      return S_OK;
   }


   case D3D12_FEATURE_D3D12_OPTIONS: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      memset(opts, 0, sizeof(*opts));
      /* Honest conservative shape; raised only when the backing exists. */
      opts->MaxGPUVirtualAddressBitsPerResource = caps->gpu_virtual_address_bits;
      opts->ResourceBindingTier =
         (D3D12_RESOURCE_BINDING_TIER)caps->resource_binding_tier;
      opts->TiledResourcesTier =
         (D3D12_TILED_RESOURCES_TIER)caps->tiled_resources_tier;
      opts->ResourceHeapTier =
         (D3D12_RESOURCE_HEAP_TIER)caps->resource_heap_tier;
      opts->OutputMergerLogicOp = caps->output_merger_logic_op;
      opts->TypedUAVLoadAdditionalFormats =
         caps->typed_uav_load_additional_formats;
      return S_OK;
   }

   /* A feature-query structure being understood is distinct from the
    * optional feature being implemented. Return a valid, zero-capability
    * answer for the structures this runtime exposes instead of E_INVALIDARG:
    * applications use the fields to choose their fallback paths.
    */
   case D3D12_FEATURE_D3D12_OPTIONS1: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS1 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;
      memset(opts, 0, sizeof(*opts));
      opts->WaveOps = caps->wave_ops;
      opts->WaveLaneCountMin = caps->wave_lane_count_min;
      opts->WaveLaneCountMax = caps->wave_lane_count_max;
      opts->TotalLaneCount = caps->total_lane_count;
      opts->Int64ShaderOps = caps->int64_shader_ops;
      return S_OK;
   }

   /* These optional capability structures contain only output fields whose
    * zero spelling means unsupported/tier 0.  Understanding the query is not
    * a capability claim: returning E_INVALIDARG here prevents applications
    * from taking their specified fallback paths.  Keep the answers
    * conservative until the corresponding command/shader path is complete.
    */
#define D12_ZERO_OPTIONS(N)                                                   \
   case D3D12_FEATURE_D3D12_OPTIONS##N: {                                    \
      D3D12_FEATURE_DATA_D3D12_OPTIONS##N *opts = data;                      \
      if (data_size != sizeof(*opts))                                        \
         return E_INVALIDARG;                                                \
      memset(opts, 0, sizeof(*opts));                                        \
      return S_OK;                                                           \
   }

   D12_ZERO_OPTIONS(6)
   D12_ZERO_OPTIONS(8)
   D12_ZERO_OPTIONS(10)
   D12_ZERO_OPTIONS(20)

#undef D12_ZERO_OPTIONS

   case D3D12_FEATURE_D3D12_OPTIONS2: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS2 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      memset(opts, 0, sizeof(*opts));
      /* DepthBoundsTestSupported stays FALSE. HAL300 has native depth-bounds
       * state, but its Gen7 PPP selector/block writer is not connected; the
       * runnable AGX2 path has no such state and Honeykrisp reports none. */
      opts->ProgrammableSamplePositionsTier =
         (D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER)
            caps->programmable_sample_positions_tier;
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS3: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS3 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      memset(opts, 0, sizeof(*opts));
      opts->CopyQueueTimestampQueriesSupported =
         caps->copy_queue_timestamp_queries;
      opts->WriteBufferImmediateSupportFlags =
         (D3D12_COMMAND_LIST_SUPPORT_FLAGS)
            caps->write_buffer_immediate_support_flags;
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS4: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS4 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      memset(opts, 0, sizeof(*opts));
      /* Apple AGX has native 16-bit floating-point ALU support and both the
       * DXIL front end and AGX compiler retain f16 arithmetic. */
      opts->Native16BitShaderOpsSupported = caps->native_16bit_shader_ops;
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS12: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS12 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      memset(opts, 0, sizeof(*opts));
      /* This counter convention is not known because mesh shaders and their
       * pipeline statistics are not exposed.  UNKNOWN is the conservative
       * public spelling, unlike zero/FALSE which would assert a convention.
       */
      opts->MSPrimitivesPipelineStatisticIncludesCulledPrimitives =
         D3D12_TRI_STATE_UNKNOWN;
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS7: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS7 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;
      memset(opts, 0, sizeof(*opts));
      opts->MeshShaderTier =
         (D3D12_MESH_SHADER_TIER)caps->mesh_shader_tier;
      opts->SamplerFeedbackTier = D3D12_SAMPLER_FEEDBACK_TIER_NOT_SUPPORTED;
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS9: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS9 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;
      memset(opts, 0, sizeof(*opts));
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS11: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS11 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;
      memset(opts, 0, sizeof(*opts));
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS13: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS13 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;
      memset(opts, 0, sizeof(*opts));
      opts->TextureCopyBetweenDimensionsSupported =
         caps->texture_copy_between_dimensions;
      opts->AlphaBlendFactorSupported = caps->alpha_blend_factor;
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS16: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS16 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      memset(opts, 0, sizeof(*opts));
      opts->DynamicDepthBiasSupported = caps->dynamic_depth_bias;
      opts->GPUUploadHeapSupported = caps->gpu_upload_heap;
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS14: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS14 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      memset(opts, 0, sizeof(*opts));
      /* AdvancedTextureOpsSupported needs sampler feedback's companion
       * operations, which AGX has no equivalent for.
       * WriteableMSAATexturesSupported needs a TEXTURE2DMS unordered-access
       * view, which CreateUnorderedAccessView does not build; the shader-side
       * lowering exists but the descriptor side does not. */
      opts->IndependentFrontAndBackStencilRefMaskSupported =
         caps->independent_front_and_back_stencil;
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS15: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS15 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      memset(opts, 0, sizeof(*opts));
      opts->TriangleFanSupported = caps->triangle_fan;
      opts->DynamicIndexBufferStripCutSupported =
         caps->dynamic_index_buffer_strip_cut;
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS17: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS17 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      memset(opts, 0, sizeof(*opts));
      opts->NonNormalizedCoordinateSamplersSupported =
         caps->non_normalized_coordinate_samplers;
      /* ManualWriteTrackingResourceSupported is a WriteWatch contract this
       * driver does not implement. */
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS18: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS18 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      memset(opts, 0, sizeof(*opts));
      /* The render-pass implementation honours the declared per-attachment
       * access types rather than emulating them, so its results are the
       * results the application asked for. */
      opts->RenderPassesValid = caps->render_passes_tier != 0;
      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS19: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS19 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      /* Heap limits are shared with the descriptor path. Sampling/rasterizer
       * promises stay off until their exact optional contracts pass. */
      memset(opts, 0, sizeof(*opts));
      opts->PointSamplingAddressesNeverRoundUp = FALSE;
      /* No-output multisampling has a separate contract from multisampled
       * attachments. Withhold it until a no-output coverage witness passes.
       * The public mask uses log2(sample_count) as the bit index. */
      opts->SupportedSampleCountsWithNoOutputs = 1;
      opts->MaxSamplerDescriptorHeapSize = caps->max_sampler_heap_size;
      opts->MaxSamplerDescriptorHeapSizeWithStaticSamplers = caps->max_sampler_heap_size;
      opts->MaxViewDescriptorHeapSize = caps->max_view_heap_size;
      return S_OK;
   }

   case D3D12_FEATURE_PREDICATION: {
      D3D12_FEATURE_DATA_PREDICATION *predication = data;
      if (data_size != sizeof(D3D12_FEATURE_DATA_PREDICATION))
         return E_INVALIDARG;
      predication->Supported = caps->predication ? TRUE : FALSE;
      return S_OK;
   }

   case D3D12_FEATURE_HARDWARE_COPY: {
      D3D12_FEATURE_DATA_HARDWARE_COPY *hardware_copy = data;
      if (data_size != sizeof(D3D12_FEATURE_DATA_HARDWARE_COPY))
         return E_INVALIDARG;
      hardware_copy->Supported = caps->hardware_copy ? TRUE : FALSE;
      return S_OK;
   }

   case D3D12_FEATURE_APPLICATION_SPECIFIC_DRIVER_STATE: {
      D3D12_FEATURE_DATA_APPLICATION_SPECIFIC_DRIVER_STATE *state = data;
      if (data_size != sizeof(*state))
         return E_INVALIDARG;
      state->Supported = FALSE;
      return S_OK;
   }

   case D3D12_FEATURE_BYTECODE_BYPASS_HASH_SUPPORTED: {
      D3D12_FEATURE_DATA_BYTECODE_BYPASS_HASH_SUPPORTED *bypass = data;
      if (data_size != sizeof(*bypass))
         return E_INVALIDARG;
      bypass->Supported = FALSE;
      return S_OK;
   }

   case D3D12_FEATURE_SHADER_CACHE_ABI_SUPPORT:
      /* The Windows ABI's WCHAR is 16-bit, while the Linux DirectX-Headers
       * stub currently typedefs it to 32-bit wchar_t. This query returns no
       * data when unsupported, so accept either public structure spelling.
       */
      if (data_size != sizeof(D3D12_FEATURE_DATA_SHADERCACHE_ABI_SUPPORT) &&
          data_size != 128 * sizeof(uint16_t) + 4 * sizeof(uint64_t))
         return E_INVALIDARG;
      return E_FAIL;

   case D3D12_FEATURE_D3D12_OPTIONS22: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS22 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;
      memset(opts, 0, sizeof(*opts));
      opts->Max1DDispatchSize =
         D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
      return S_OK;
   }

   case D3D12_FEATURE_BARRIER_LAYOUT: {
      D3D12_FEATURE_DATA_BARRIER_LAYOUT *layout = data;
      if (data_size != sizeof(*layout))
         return E_INVALIDARG;

      layout->Supported = FALSE;

      return S_OK;
   }

   case D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT: {
      D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT *va = data;
      if (data_size != sizeof(*va))
         return E_INVALIDARG;

      va->MaxGPUVirtualAddressBitsPerResource =
         caps->gpu_virtual_address_bits;
      va->MaxGPUVirtualAddressBitsPerProcess =
         caps->gpu_virtual_address_bits;
      return S_OK;
   }

   case D3D12_FEATURE_ROOT_SIGNATURE: {
      D3D12_FEATURE_DATA_ROOT_SIGNATURE *rs = data;
      if (data_size != sizeof(*rs))
         return E_INVALIDARG;

      if (rs->HighestVersion < D3D_ROOT_SIGNATURE_VERSION_1_0 ||
          rs->HighestVersion > D3D_ROOT_SIGNATURE_VERSION_1_2)
         return E_INVALIDARG;
      if ((uint32_t)rs->HighestVersion > caps->max_root_signature_version)
         rs->HighestVersion = (D3D_ROOT_SIGNATURE_VERSION)caps->max_root_signature_version;

      return S_OK;
   }

   case D3D12_FEATURE_FORMAT_SUPPORT: {
      D3D12_FEATURE_DATA_FORMAT_SUPPORT *support = data;
      uint64_t native_support;
      if (data_size != sizeof(*support))
         return E_INVALIDARG;

      support->Support1 = D3D12_FORMAT_SUPPORT1_NONE;
      support->Support2 = D3D12_FORMAT_SUPPORT2_NONE;

      /* Processing the query successfully is distinct from supporting any
       * particular use of the format.  Reserved/forward-compatible DXGI
       * spellings are valid negative queries; truly out-of-range values are
       * malformed. */
      if ((UINT)support->Format > (UINT)DXGI_FORMAT_A4B4G4R4_UNORM ||
          (support->Format > DXGI_FORMAT_B4G4R4A4_UNORM &&
           support->Format < DXGI_FORMAT_P208))
         return E_INVALIDARG;

      native_support = d12_format_frontend_support(support->Format);
#define D12_MAP_FORMAT1(frontend_bit, public_bit)                           \
   do {                                                                      \
      if (native_support & (frontend_bit))                                   \
         support->Support1 =                                                 \
            (D3D12_FORMAT_SUPPORT1)(support->Support1 | (public_bit));        \
   } while (0)
#define D12_MAP_FORMAT2(frontend_bit, public_bit)                           \
   do {                                                                      \
      if (native_support & (frontend_bit))                                   \
         support->Support2 =                                                 \
            (D3D12_FORMAT_SUPPORT2)(support->Support2 | (public_bit));        \
   } while (0)
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_BUFFER,
                      D3D12_FORMAT_SUPPORT1_BUFFER);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_VERTEX_BUFFER,
                      D3D12_FORMAT_SUPPORT1_IA_VERTEX_BUFFER);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_INDEX_BUFFER,
                      D3D12_FORMAT_SUPPORT1_IA_INDEX_BUFFER);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_TEXTURE_1D,
                      D3D12_FORMAT_SUPPORT1_TEXTURE1D);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_TEXTURE_2D,
                      D3D12_FORMAT_SUPPORT1_TEXTURE2D);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_TEXTURE_3D,
                      D3D12_FORMAT_SUPPORT1_TEXTURE3D);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_TEXTURE_CUBE,
                      D3D12_FORMAT_SUPPORT1_TEXTURECUBE);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_SHADER_LOAD,
                      D3D12_FORMAT_SUPPORT1_SHADER_LOAD);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_SHADER_SAMPLE,
                      D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_SHADER_SAMPLE_COMPARISON,
                      D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE_COMPARISON);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_MIP,
                      D3D12_FORMAT_SUPPORT1_MIP);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_RENDER_TARGET,
                      D3D12_FORMAT_SUPPORT1_RENDER_TARGET);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_BLENDABLE,
                      D3D12_FORMAT_SUPPORT1_BLENDABLE);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_DEPTH_STENCIL,
                      D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_MULTISAMPLE_RESOLVE,
                      D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_MULTISAMPLE_RENDER_TARGET,
                      D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_MULTISAMPLE_LOAD,
                      D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_DISPLAY,
                      D3D12_FORMAT_SUPPORT1_DISPLAY);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_CAST_WITHIN_BIT_LAYOUT,
                      D3D12_FORMAT_SUPPORT1_CAST_WITHIN_BIT_LAYOUT);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_TYPED_UAV,
                      D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_SHADER_GATHER,
                      D3D12_FORMAT_SUPPORT1_SHADER_GATHER);
      D12_MAP_FORMAT1(D12_FRONTEND_FORMAT_SHADER_GATHER_COMPARISON,
                      D3D12_FORMAT_SUPPORT1_SHADER_GATHER_COMPARISON);
      D12_MAP_FORMAT2(D12_FRONTEND_FORMAT_UAV_TYPED_LOAD,
                      D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD);
      D12_MAP_FORMAT2(D12_FRONTEND_FORMAT_UAV_TYPED_STORE,
                      D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
#undef D12_MAP_FORMAT1
#undef D12_MAP_FORMAT2

      if (!caps->full_texture_sampling) {
         support->Support1 = (D3D12_FORMAT_SUPPORT1)(support->Support1 &
            ~(D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE |
              D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE_COMPARISON |
              D3D12_FORMAT_SUPPORT1_SHADER_GATHER |
              D3D12_FORMAT_SUPPORT1_SHADER_GATHER_COMPARISON));
      }
      if (!(caps->sample_counts & (2u | 4u))) {
         support->Support1 = (D3D12_FORMAT_SUPPORT1)(support->Support1 &
            ~(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE |
              D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET |
              D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD));
      }
      if (!caps->typed_uav_load_additional_formats &&
          support->Format != DXGI_FORMAT_R32_FLOAT &&
          support->Format != DXGI_FORMAT_R32_UINT &&
          support->Format != DXGI_FORMAT_R32_SINT) {
         support->Support2 = (D3D12_FORMAT_SUPPORT2)(support->Support2 &
            ~D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD);
      }

      return S_OK;
   }

   case D3D12_FEATURE_FORMAT_INFO: {
      D3D12_FEATURE_DATA_FORMAT_INFO *info = data;
      if (data_size != sizeof(*info))
         return E_INVALIDARG;

      /*
       * PLANE COUNT. Every format is one plane except a depth format that
       * also carries stencil -- on AGX those are physically separate
       * allocations, which is precisely why d12_format_stencil_plane
       * exists, so the answer is derived from the same source the resource
       * path uses rather than from a second table.
       *
       * DXGI_FORMAT_UNKNOWN is legal here and answers 1; a format we cannot
       * express at all is E_INVALIDARG, since a plane count for it would be
       * an invention.
       */
      if (info->Format == DXGI_FORMAT_UNKNOWN) {
         info->PlaneCount = 1;
         return S_OK;
      }

      /* 66 is an unassigned hole in the DXGI enum -- not a format with no
       * planes, but no format at all, so it is refused rather than
       * answered. */
      if (info->Format == (DXGI_FORMAT)66)
         return E_INVALIDARG;

      /*
       * The PLANE-VIEW formats of a depth/stencil pair. These name one
       * plane of a two-plane resource -- R32_FLOAT_X8X24_TYPELESS is its
       * depth half, X32_TYPELESS_G8X24_UINT its stencil half -- so they
       * describe a 2-plane format even though neither is itself a
       * depth/stencil format this driver can allocate. They are not in
       * d12_format_stencil_plane's table for exactly that reason, so they
       * are listed here rather than by widening a table used for
       * allocation.
       */
      switch (info->Format) {
      case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:
      case DXGI_FORMAT_X32_TYPELESS_G8X24_UINT:
      case DXGI_FORMAT_R24_UNORM_X8_TYPELESS:
      case DXGI_FORMAT_X24_TYPELESS_G8_UINT:
         info->PlaneCount = 2;
         return S_OK;
      default:
         break;
      }

      /*
       * Everything else is single-plane, INCLUDING formats this driver
       * cannot allocate. Plane count is a property of the FORMAT, not of
       * our support for it -- refusing the query for a format we happen not
       * to implement was the wrong model, and the suite caught it by asking
       * about formats we do not map and expecting 1.
       */
      if (d12_format_stencil_plane(info->Format) != PIPE_FORMAT_NONE)
         info->PlaneCount = 2;
      else
         info->PlaneCount = 1;

      return S_OK;
   }

   case D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS: {
      D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS *levels = data;
      if (data_size != sizeof(*levels))
         return E_INVALIDARG;

      levels->NumQualityLevels = 0;

      /* A zero sample count is a malformed query.  One sample is the
       * ordinary non-MSAA layout and therefore has one quality level for
       * every DXGI format spelling, independent of image-format support.
       * AGX and the resource path currently expose the hardware's 2x and
       * 4x modes; larger powers of two remain conservatively unsupported.
       */
      if (!levels->SampleCount)
         return E_FAIL;

      if (levels->SampleCount == 1) {
         levels->NumQualityLevels = 1;
      } else if (levels->Format != DXGI_FORMAT_UNKNOWN &&
                 (levels->SampleCount == 2 || levels->SampleCount == 4) &&
                 (caps->sample_counts & levels->SampleCount) &&
                 (d12_format_to_pipe(levels->Format) != PIPE_FORMAT_NONE ||
                  d12_format_depth_plane(levels->Format) != PIPE_FORMAT_NONE)) {
         levels->NumQualityLevels = 1;
      }

      return S_OK;
   }

   case D3D12_FEATURE_D3D12_OPTIONS5: {
      /* d12_raytracing_core.h reproduces these so it can stay free of
       * Windows headers. Pin them to the real values here. */
      static_assert(D12_RT_SHADER_IDENTIFIER_SIZE ==
                          D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES &&
                       D12_RT_SHADER_RECORD_BYTE_ALIGNMENT ==
                          D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT &&
                       D12_RT_SHADER_TABLE_BYTE_ALIGNMENT ==
                          D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT &&
                       D12_RT_MAX_SHADER_RECORD_STRIDE ==
                          D3D12_RAYTRACING_MAX_SHADER_RECORD_STRIDE &&
                       D12_RT_AS_BYTE_ALIGNMENT ==
                          D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT &&
                       D12_RT_MAX_ATTRIBUTE_SIZE ==
                          D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES &&
                       D12_RT_MAX_DECLARABLE_RECURSION ==
                          D3D12_RAYTRACING_MAX_DECLARABLE_TRACE_RECURSION_DEPTH,
                    "DXR host ABI constants");

      D3D12_FEATURE_DATA_D3D12_OPTIONS5 *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      memset(opts, 0, sizeof(*opts));
      /* No connected acceleration-structure traversal/DispatchRays path. */
      opts->RaytracingTier = D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
      opts->RenderPassesTier =
         (D3D12_RENDER_PASS_TIER)caps->render_passes_tier;
      opts->SRVOnlyTiledResourceTier3 = FALSE;
      return S_OK;
   }

   case D3D12_FEATURE_ARCHITECTURE: {
      D3D12_FEATURE_DATA_ARCHITECTURE *arch = data;
      if (data_size != sizeof(*arch))
         return E_INVALIDARG;
      if (arch->NodeIndex != 0)
         return E_INVALIDARG;

      arch->TileBasedRenderer = caps->tile_based_deferred_renderer;
      arch->UMA = caps->uma;
      arch->CacheCoherentUMA = caps->cache_coherent_uma;
      return S_OK;
   }

   case D3D12_FEATURE_ARCHITECTURE1: {
      D3D12_FEATURE_DATA_ARCHITECTURE1 *arch = data;
      if (data_size != sizeof(*arch))
         return E_INVALIDARG;
      if (arch->NodeIndex != 0)
         return E_INVALIDARG;

      arch->TileBasedRenderer = caps->tile_based_deferred_renderer;
      arch->UMA = caps->uma;
      arch->CacheCoherentUMA = caps->cache_coherent_uma;
      arch->IsolatedMMU = caps->isolated_mmu;
      return S_OK;
   }

   case D3D12_FEATURE_SHADER_MODEL: {
      D3D12_FEATURE_DATA_SHADER_MODEL *sm = data;
      if (data_size != sizeof(*sm))
         return E_INVALIDARG;

      /* Negotiate against the complete shader-model contract, independent
       * of whether an individual f16 instruction can be compiled. */
      if (sm->HighestShaderModel < D3D_SHADER_MODEL_5_1 ||
          sm->HighestShaderModel > D3D_SHADER_MODEL_6_8)
         return E_INVALIDARG;
      const D3D_SHADER_MODEL maximum =
         (D3D_SHADER_MODEL)caps->max_shader_model;
      if (sm->HighestShaderModel > maximum)
         sm->HighestShaderModel = maximum;
      return S_OK;
   }

   case D3D12_FEATURE_FEATURE_LEVELS: {
      D3D12_FEATURE_DATA_FEATURE_LEVELS *levels = data;
      if (data_size != sizeof(*levels))
         return E_INVALIDARG;
      if (!levels->pFeatureLevelsRequested || !levels->NumFeatureLevels)
         return E_INVALIDARG;

      D3D_FEATURE_LEVEL best = (D3D_FEATURE_LEVEL)d12_core_caps_choose_feature_level(caps,
         (const uint32_t *)levels->pFeatureLevelsRequested,
         levels->NumFeatureLevels);

      if (!best)
         return E_INVALIDARG;

      levels->MaxSupportedFeatureLevel = best;
      return S_OK;
   }

   default:
      mesa_logw("ID3D12Device::CheckFeatureSupport: feature %d not "
                "implemented",
                (int)feature);
      return E_INVALIDARG;
   }
}

