/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * ID3D12Device on a real agx_device, plus the D3D12CreateDevice entry point.
 * Platform discovery and winsys lifetime live behind d12_platform_*.
 */

#include "d12_private.h"
#include "d12_attachment_view_core.h"
#include "d12_allocation_math.h"
#include "d12_buffer_view_core.h"
#include "d12_core_caps.h"
#include "d12_query_core.h"

#include <math.h>
#include <string.h>

#include "util/hash_table.h"
#define XXH_INLINE_ALL
#include "util/xxhash.h"

/*
 * Report an unimplemented COM method. Logged once per method so a test
 * suite run produces a readable list of what is missing rather than a
 * flood, and so an unimplemented call is a recordable failure instead of
 * a jump to NULL.
 */
void
d12_stub_report(const char *what)
{
   static const char *seen[256];
   static unsigned num_seen;

   for (unsigned i = 0; i < num_seen; i++) {
      if (seen[i] == what)
         return;
   }

   if (num_seen < ARRAY_SIZE(seen))
      seen[num_seen++] = what;

   mesa_logw("unimplemented: %s", what);
}

void
d12_log_unknown_riid(const char *where, REFIID riid)
{
   const GUID *g = riid;
   mesa_loge("%s: unsupported riid "
             "{%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x}",
             where, (unsigned)g->Data1, g->Data2, g->Data3, g->Data4[0],
             g->Data4[1],
             g->Data4[2], g->Data4[3], g->Data4[4], g->Data4[5], g->Data4[6],
             g->Data4[7]);
}

/*
 * IUnknown + ID3D12Object
 */

static HRESULT STDMETHODCALLTYPE
d12_device_QueryInterface(ID3D12Device2 *iface, REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);

   if (!out)
      return E_POINTER;

   static const IID *const device_iids[] = {
      &IID_IUnknown,        &IID_ID3D12Object,   &IID_ID3D12Device,
      &IID_ID3D12Device1,   &IID_ID3D12Device2,  &IID_ID3D12Device3,
      &IID_ID3D12Device4,   &IID_ID3D12Device5,  &IID_ID3D12Device6,
      &IID_ID3D12Device7,   &IID_ID3D12Device8,  &IID_ID3D12Device9,
      &IID_ID3D12Device10,  &IID_ID3D12Device11, &IID_ID3D12Device12,
      &IID_ID3D12Device13,  &IID_ID3D12Device14, &IID_ID3D12Device15,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(device_iids); i++) {
      if (d12_guid_equal(riid, device_iids[i])) {
         /* One object, one vtable: every ID3D12DeviceN table is a strict
          * prefix of the ID3D12Device15 table this object carries, so the
          * newer IIDs return the same pointer. */
         d12_device_add_ref(device);
         *out = iface;
         return S_OK;
      }
   }

   if (d12_guid_equal(riid, &IID_ID3D12DeviceConfiguration) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceConfiguration1)) {
      d12_device_add_ref(device);
      *out = &device->iface_config1;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&device->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12Device::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_device_AddRef(ID3D12Device2 *iface)
{
   struct d12_device *device = d12_device_from_iface(iface);
   return p_atomic_inc_return(&device->refcount);
}

static ULONG STDMETHODCALLTYPE
d12_device_Release(ID3D12Device2 *iface)
{
   struct d12_device *device = d12_device_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&device->refcount);

   if (refs == 0) {
      mesa_logi("closing device %s", device->dev.name);
      d12_device_backend_cleanup(device);
      d12_platform_close_device(device);
      free(device);
   }

   return refs;
}

void
d12_device_add_ref(struct d12_device *device)
{
   d12_device_AddRef((ID3D12Device2 *)&device->iface);
}

void
d12_device_release(struct d12_device *device)
{
   d12_device_Release((ID3D12Device2 *)&device->iface);
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_device, ID3D12Device2,
                                d12_device_from_iface)

/*
 * ID3D12Device
 */

static UINT STDMETHODCALLTYPE
d12_device_GetNodeCount(ID3D12Device2 *iface)
{
   return 1;
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreateCommandQueue(ID3D12Device2 *iface,
                              const D3D12_COMMAND_QUEUE_DESC *desc,
                              REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);

   /* Nothing may be created on a removed device. */
   if (d12_device_is_removed(device))
      return DXGI_ERROR_DEVICE_REMOVED;

   return d12_queue_create(device, desc, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreateCommandAllocator(ID3D12Device2 *iface,
                                  D3D12_COMMAND_LIST_TYPE type, REFIID riid,
                                  void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);
   return d12_cmd_allocator_create(device, type, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreateCommandList(ID3D12Device2 *iface, UINT node_mask,
                             D3D12_COMMAND_LIST_TYPE type,
                             ID3D12CommandAllocator *allocator,
                             ID3D12PipelineState *initial_state, REFIID riid,
                             void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);

   /* COM CONTRACT: the out-pointer is set to NULL BEFORE any validation.
    * A failing Create that leaves it untouched hands the caller back an
    * uninitialised stack variable, and a well-written application then
    * passes that garbage to the next call -- which is how a refusal here
    * turned into a segfault three calls later, in CreateRenderTargetView
    * and in Draw, with a resource pointer full of ASCII. Nulling first
    * turns every one of those into an ordinary failure.
    */
   if (out)
      *out = NULL;

   if (node_mask > 1)
      return E_INVALIDARG;
   if (!allocator)
      return E_INVALIDARG;

   struct d12_cmd_allocator *d12_allocator =
      d12_cmd_allocator_from_iface(allocator);
   if (d12_allocator->type != type)
      return E_INVALIDARG;
   /*
    * The INITIAL PIPELINE STATE. A list created with one is already bound
    * to it and may draw without any SetPipelineState call, which is
    * exactly what test_command_list_initial_pipeline_state does. This was
    * dropped with a note saying PSOs did not exist yet -- a comment that
    * went stale the moment they did, leaving a legal draw failing with
    * "Draw without a graphics pipeline state".
    *
    * d12_list_Reset already honours its own initial_state argument, so the
    * two entry points now agree.
    */
   return d12_cmd_list_create(
      device, type, d12_allocator,
      initial_state ? d12_pipeline_state_from_iface(initial_state) : NULL,
      riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device_CheckFeatureSupport(ID3D12Device2 *iface, D3D12_FEATURE feature,
                               void *data, UINT data_size)
{
   const struct d12_core_caps *caps = d12_core_get_caps();
   (void)iface;

   switch (feature) {
   case D3D12_FEATURE_D3D12_OPTIONS: {
      D3D12_FEATURE_DATA_D3D12_OPTIONS *opts = data;
      if (data_size != sizeof(*opts))
         return E_INVALIDARG;

      memset(opts, 0, sizeof(*opts));
      /* Honest conservative shape; raised only when the backing exists. */
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
      opts->DynamicDepthBiasSupported = TRUE;
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

      /* We understand the query but do not yet expose RasterizerDesc2.
       * A zero capability answer makes applications take the supported
       * legacy rasterizer path instead of mistaking E_INVALIDARG for an
       * unknown runtime interface. AGX point sampling was measured across
       * the 8-bit subtexel boundary: addresses below the boundary never
       * round up, while linear sampling and gather retain their normal
       * fixed-point rules.
       */
      memset(opts, 0, sizeof(*opts));
      opts->PointSamplingAddressesNeverRoundUp = TRUE;
      /* Target-independent rasterization: with nothing attached the render
       * command's sample count is purely the rasterization rate, so every
       * count AGX can rasterize is available. */
      opts->SupportedSampleCountsWithNoOutputs =
         (1u << 0) | (1u << 1) | (1u << 3);
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

      bool direct = layout->CommandListType == D3D12_COMMAND_LIST_TYPE_DIRECT;
      bool compute =
         layout->CommandListType == D3D12_COMMAND_LIST_TYPE_COMPUTE;

      /* D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON (value 30) was added after
       * the DirectX-Headers version this build uses. Keep the public value
       * here so callers built against the current header get the right
       * queue/layout validation answer.
       */
      if ((unsigned)layout->Layout == 30) {
         layout->Supported =
            layout->CommandListType == D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE ||
            layout->CommandListType == D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS ||
            layout->CommandListType == D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE;
         return S_OK;
      }

      switch (layout->Layout) {
      case D3D12_BARRIER_LAYOUT_COMMON:
      case D3D12_BARRIER_LAYOUT_UNDEFINED:
         layout->Supported = TRUE;
         break;

      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE:
      case D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST:
      case D3D12_BARRIER_LAYOUT_RENDER_TARGET:
      case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ:
      case D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE:
      case D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE:
      case D3D12_BARRIER_LAYOUT_RESOLVE_DEST:
      case D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE:
         layout->Supported = direct;
         break;

      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COMMON:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_GENERIC_READ:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_UNORDERED_ACCESS:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_SOURCE:
      case D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_DEST:
         layout->Supported = compute;
         break;

      case D3D12_BARRIER_LAYOUT_GENERIC_READ:
      case D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS:
      case D3D12_BARRIER_LAYOUT_SHADER_RESOURCE:
      case D3D12_BARRIER_LAYOUT_COPY_SOURCE:
      case D3D12_BARRIER_LAYOUT_COPY_DEST:
         layout->Supported = direct || compute;
         break;

      case D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_READ:
      case D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_WRITE:
         layout->Supported = layout->CommandListType ==
                             D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS;
         break;

      case D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_READ:
      case D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_WRITE:
         layout->Supported = layout->CommandListType ==
                             D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE;
         break;

      case D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ:
      case D3D12_BARRIER_LAYOUT_VIDEO_DECODE_WRITE:
         layout->Supported = layout->CommandListType ==
                             D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE;
         break;

      default:
         layout->Supported = FALSE;
         break;
      }

      return S_OK;
   }

   case D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT: {
      D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT *va = data;
      if (data_size != sizeof(*va))
         return E_INVALIDARG;

      va->MaxGPUVirtualAddressBitsPerResource =
         d12_core_get_caps()->gpu_virtual_address_bits;
      va->MaxGPUVirtualAddressBitsPerProcess =
         d12_core_get_caps()->gpu_virtual_address_bits;
      return S_OK;
   }

   case D3D12_FEATURE_ROOT_SIGNATURE: {
      D3D12_FEATURE_DATA_ROOT_SIGNATURE *rs = data;
      if (data_size != sizeof(*rs))
         return E_INVALIDARG;

      /*
       * This is a REQUEST/REPLY field: the caller writes the highest
       * version it can use and the runtime writes back the highest it
       * supports, capped by that. d12_root_sig parses both 1.0 and 1.1
       * blobs (the version-2 range offset is handled there), so 1.1 is
       * honest.
       */
      if (rs->HighestVersion != D3D_ROOT_SIGNATURE_VERSION_1_0 &&
          rs->HighestVersion != D3D_ROOT_SIGNATURE_VERSION_1_1) {
         /* An unrecognised version is refused and the field is LEFT
          * UNTOUCHED -- the suite checks that 0 and 0xdeadbeef both come
          * back exactly as passed. Clamping them would look like a
          * successful negotiation of a version the caller never named. */
         return E_INVALIDARG;
      }

      /* Both 1.0 and 1.1 parse (d12_root_sig handles the version-2 range
       * layout), so whatever the caller can use, we support. */
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
      /*
       * Stated rather than left absent. The DXIL translator implements
       * ray queries as a traversal that visits nothing, which is the
       * correct answer only while no acceleration structure can exist --
       * and this is what guarantees that.
       */
      /*
       * Ray tracing is not supported and will report so until a traversal
       * path exists. The host-side half of DXR -- shader tables, shader
       * identifiers, DispatchRays bounds, acceleration-structure sizing --
       * is implemented and tested in d12_raytracing_core.h, because none of
       * it depends on how rays are traced. What is missing is tracing them:
       * Apple hardware ray tracing starts at G15 (M3), this driver runs on
       * G13/G14 (M1/M2) which has none, Mesa supports no G15/G16 part, and
       * on G17 (M5) the traversal core of the ISA is 96.8% ungrounded so no
       * traversal code can be emitted.
       */
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

      /* The DXIL front end and AGX compiler preserve native 16-bit SM 6.2
       * arithmetic. The query clamps DOWN to the shared renderer policy,
       * per the D3D12 contract. */
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

      D3D_FEATURE_LEVEL best = (D3D_FEATURE_LEVEL)d12_core_choose_feature_level(
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

/*
 * The binding assignment a pipeline is compiled against. It has to be the
 * one stored on the root signature: the command list fills the root table
 * through the same assignment at bind time, and two sides deriving their
 * own would only agree by luck.
 */
static const struct d12_binding_map *
d12_pso_binding_map(ID3D12RootSignature *rs)
{
   /* No signature is legal only for a shader that binds nothing. The
    * translator reports it by name if the shader does declare a resource,
    * rather than silently resolving it to slot 0.
    */
   static const struct d12_binding_map empty;

   return rs ? &((struct d12_root_signature *)rs)->map : &empty;
}

/* The root-signature parser writes a canonical, zero-initialized layout, so
 * comparing the populated arrays is both padding-safe and stricter than just
 * comparing the binding map. The latter would miss flags and static-sampler
 * state that participate in pipeline compatibility.
 */
static bool
d12_root_layout_equal(const struct d12_root_layout *a,
                      const struct d12_root_layout *b)
{
   if (a->num_params != b->num_params ||
       a->num_ranges != b->num_ranges ||
       a->num_static_samplers != b->num_static_samplers ||
       a->flags != b->flags ||
       memcmp(a->params, b->params, sizeof(a->params)))
      return false;

   if (a->num_ranges &&
       memcmp(a->ranges, b->ranges,
              a->num_ranges * sizeof(*a->ranges)))
      return false;

   return !a->num_static_samplers ||
          !memcmp(a->static_samplers, b->static_samplers,
                  a->num_static_samplers * sizeof(*a->static_samplers));
}

/* When the caller leaves pRootSignature NULL, D3D12 derives the pipeline
 * layout from a root signature embedded in any participating shader. More
 * than one shader may carry the signature, but all embedded copies must
 * describe the same layout. Stages without an RTS0 part are allowed, and if
 * no stage embeds one the existing empty-map behavior remains available to
 * resource-free shaders.
 */
static HRESULT
d12_embedded_pso_binding_map(const D3D12_SHADER_BYTECODE *const *stages,
                             unsigned stage_count,
                             struct d12_root_layout *owned_layout,
                             struct d12_binding_map *owned_map,
                             const struct d12_binding_map **out_map)
{
   memset(owned_layout, 0, sizeof(*owned_layout));
   memset(owned_map, 0, sizeof(*owned_map));
   bool found = false;

   for (unsigned i = 0; i < stage_count; i++) {
      const D3D12_SHADER_BYTECODE *stage = stages[i];
      if (!stage || !stage->pShaderBytecode || !stage->BytecodeLength ||
          !d12_root_signature_has_rts0(stage->pShaderBytecode,
                                       stage->BytecodeLength))
         continue;

      struct d12_root_layout parsed;
      if (!d12_root_signature_parse(stage->pShaderBytecode,
                                    stage->BytecodeLength, &parsed)) {
         d12_root_layout_cleanup(owned_layout);
         return E_INVALIDARG;
      }

      if (!found) {
         *owned_layout = parsed;
         found = true;
      } else {
         bool matches = d12_root_layout_equal(owned_layout, &parsed);
         d12_root_layout_cleanup(&parsed);
         if (!matches) {
            mesa_loge("pipeline shaders contain incompatible embedded root "
                      "signatures");
            d12_root_layout_cleanup(owned_layout);
            return E_INVALIDARG;
         }
      }
   }

   if (found) {
      if (!d12_root_layout_build_map(owned_layout, owned_map)) {
         d12_root_layout_cleanup(owned_layout);
         return E_INVALIDARG;
      }
      *out_map = owned_map;
   } else {
      *out_map = d12_pso_binding_map(NULL);
   }

   return S_OK;
}

/* Pipeline-library blobs must not contain process pointers. Build a stable
 * descriptor key from the value fields, shader bytes, canonical root layout,
 * and the pointed-to input/SO declarations. This is validation metadata, not
 * executable code: a deserialized library recompiles a matching description
 * on first use (through the device shader cache when the stages were
 * compiled in this process) and then caches that PSO object.
 *
 * Each piece is folded in as XXH64(piece, previous lane) so the key is a
 * chain over the pieces in order; the former byte-at-a-time FNV loop over
 * the whole bytecode was the largest remaining cost of a cached PSO. The
 * serialized-library format version covers the key definition. */
static void
d12_library_key_add(struct d12_pipeline_library_key *key, const void *data,
                    size_t size)
{
   key->lo = XXH64(data, size, key->lo);
   key->hi = XXH64(data, size, key->hi ^ UINT64_C(0x9e3779b97f4a7c15));
}

static void
d12_library_key_add_string(struct d12_pipeline_library_key *key,
                           const char *string)
{
   size_t length = string ? strlen(string) : 0;
   d12_library_key_add(key, &length, sizeof(length));
   if (length)
      d12_library_key_add(key, string, length);
}

static struct d12_pipeline_library_key
d12_library_key_begin(void)
{
   return (struct d12_pipeline_library_key){
      .lo = UINT64_C(14695981039346656037),
      .hi = UINT64_C(7809847782465536322),
   };
}

static void
d12_library_key_add_root(struct d12_pipeline_library_key *key,
                         const struct d12_root_layout *layout)
{
   d12_library_key_add(key, &layout->num_params,
                       sizeof(layout->num_params));
   d12_library_key_add(key, layout->params,
                       layout->num_params * sizeof(layout->params[0]));
   d12_library_key_add(key, &layout->num_ranges,
                       sizeof(layout->num_ranges));
   if (layout->num_ranges)
      d12_library_key_add(key, layout->ranges,
                          layout->num_ranges * sizeof(layout->ranges[0]));
   d12_library_key_add(key, &layout->num_static_samplers,
                       sizeof(layout->num_static_samplers));
   if (layout->num_static_samplers)
      d12_library_key_add(key, layout->static_samplers,
                          layout->num_static_samplers *
                             sizeof(layout->static_samplers[0]));
   d12_library_key_add(key, &layout->flags, sizeof(layout->flags));
}

static struct d12_pipeline_library_key
d12_compute_library_key(const D3D12_COMPUTE_PIPELINE_STATE_DESC *desc,
                        const struct d12_root_layout *layout)
{
   struct d12_pipeline_library_key key = d12_library_key_begin();
   D3D12_COMPUTE_PIPELINE_STATE_DESC canonical = *desc;
   canonical.pRootSignature = NULL;
   canonical.CS.pShaderBytecode = NULL;
   canonical.CachedPSO.pCachedBlob = NULL;
   canonical.CachedPSO.CachedBlobSizeInBytes = 0;

   d12_library_key_add(&key, &canonical, sizeof(canonical));
   d12_library_key_add_root(&key, layout);
   d12_library_key_add(&key, desc->CS.pShaderBytecode,
                       desc->CS.BytecodeLength);
   return key;
}

static struct d12_pipeline_library_key
d12_graphics_library_key(const D3D12_GRAPHICS_PIPELINE_STATE_DESC *desc,
                         const struct d12_root_layout *layout)
{
   struct d12_pipeline_library_key key = d12_library_key_begin();
   D3D12_GRAPHICS_PIPELINE_STATE_DESC canonical = *desc;
   canonical.pRootSignature = NULL;
   canonical.VS.pShaderBytecode = NULL;
   canonical.PS.pShaderBytecode = NULL;
   canonical.DS.pShaderBytecode = NULL;
   canonical.HS.pShaderBytecode = NULL;
   canonical.GS.pShaderBytecode = NULL;
   canonical.InputLayout.pInputElementDescs = NULL;
   canonical.StreamOutput.pSODeclaration = NULL;
   canonical.StreamOutput.pBufferStrides = NULL;
   canonical.CachedPSO.pCachedBlob = NULL;
   canonical.CachedPSO.CachedBlobSizeInBytes = 0;

   d12_library_key_add(&key, &canonical, sizeof(canonical));
   d12_library_key_add_root(&key, layout);

   const D3D12_SHADER_BYTECODE *stages[] = {
      &desc->VS, &desc->PS, &desc->DS, &desc->HS, &desc->GS,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(stages); i++) {
      if (stages[i]->pShaderBytecode && stages[i]->BytecodeLength)
         d12_library_key_add(&key, stages[i]->pShaderBytecode,
                             stages[i]->BytecodeLength);
   }

   for (unsigned i = 0; i < desc->InputLayout.NumElements; i++) {
      D3D12_INPUT_ELEMENT_DESC element =
         desc->InputLayout.pInputElementDescs[i];
      const char *semantic = element.SemanticName;
      element.SemanticName = NULL;
      d12_library_key_add(&key, &element, sizeof(element));
      d12_library_key_add_string(&key, semantic);
   }

   for (unsigned i = 0; i < desc->StreamOutput.NumEntries; i++) {
      D3D12_SO_DECLARATION_ENTRY entry =
         desc->StreamOutput.pSODeclaration[i];
      const char *semantic = entry.SemanticName;
      entry.SemanticName = NULL;
      d12_library_key_add(&key, &entry, sizeof(entry));
      d12_library_key_add_string(&key, semantic);
   }
   if (desc->StreamOutput.NumStrides)
      d12_library_key_add(&key, desc->StreamOutput.pBufferStrides,
                          desc->StreamOutput.NumStrides * sizeof(UINT));

   return key;
}

static bool
d12_library_key_equal(struct d12_pipeline_library_key a,
                      struct d12_pipeline_library_key b)
{
   return a.lo == b.lo && a.hi == b.hi;
}

/*
 * The graphics counterpart of the compute cached-PSO blob: an identifying
 * header, the five stage bytecodes and the canonical root layout, with the
 * fixed-function state, input layout and stream-output declaration covered
 * by the library key. Like the compute blob it is a correctness contract
 * (exact byte comparison on CreateGraphicsPipelineState) rather than
 * executable code; the device shader cache is what makes the recreation
 * cheap in-process.
 */
struct d12_graphics_cache_header {
   char magic[8];
   uint32_t version;
   uint32_t num_params;
   uint32_t num_ranges;
   uint32_t num_static_samplers;
   uint32_t flags;
   uint32_t stage_sizes[5]; /* VS, PS, DS, HS, GS */
   uint64_t key_lo, key_hi;
};

static uint8_t *
d12_graphics_cache_blob(const D3D12_GRAPHICS_PIPELINE_STATE_DESC *desc,
                        const struct d12_root_layout *layout,
                        struct d12_pipeline_library_key key, size_t *size_out)
{
   const D3D12_SHADER_BYTECODE *stages[] = {
      &desc->VS, &desc->PS, &desc->DS, &desc->HS, &desc->GS,
   };
   struct d12_graphics_cache_header header = {
      .magic = {'D', '1', '2', 'A', 'G', 'X', 'G', 'P'},
      .version = 1,
      .num_params = layout->num_params,
      .num_ranges = layout->num_ranges,
      .num_static_samplers = layout->num_static_samplers,
      .flags = layout->flags,
      .key_lo = key.lo,
      .key_hi = key.hi,
   };

   size_t size = sizeof(header);
   for (unsigned i = 0; i < ARRAY_SIZE(stages); i++) {
      const size_t stage_size =
         stages[i]->pShaderBytecode ? stages[i]->BytecodeLength : 0;
      if (stage_size > UINT32_MAX || stage_size > SIZE_MAX / 2 - size)
         return NULL;
      header.stage_sizes[i] = stage_size;
      size += stage_size;
   }
   if (layout->num_params > D12_MAX_ROOT_PARAMS)
      return NULL;
   const size_t params_size = layout->num_params * sizeof(layout->params[0]);
   const size_t ranges_size = layout->num_ranges * sizeof(layout->ranges[0]);
   const size_t samplers_size =
      layout->num_static_samplers * sizeof(layout->static_samplers[0]);
   size += params_size + ranges_size + samplers_size;

   uint8_t *blob = malloc(size);
   if (!blob)
      return NULL;
   uint8_t *cursor = blob;
   memcpy(cursor, &header, sizeof(header));
   cursor += sizeof(header);
   for (unsigned i = 0; i < ARRAY_SIZE(stages); i++) {
      if (header.stage_sizes[i]) {
         memcpy(cursor, stages[i]->pShaderBytecode, header.stage_sizes[i]);
         cursor += header.stage_sizes[i];
      }
   }
   memcpy(cursor, layout->params, params_size);
   cursor += params_size;
   if (ranges_size) {
      memcpy(cursor, layout->ranges, ranges_size);
      cursor += ranges_size;
   }
   if (samplers_size)
      memcpy(cursor, layout->static_samplers, samplers_size);

   *size_out = size;
   return blob;
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreateComputePipelineState(
   ID3D12Device2 *iface, const D3D12_COMPUTE_PIPELINE_STATE_DESC *desc,
   REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);

   /* COM CONTRACT: the out-pointer is set to NULL BEFORE any validation.
    * A failing Create that leaves it untouched hands the caller back an
    * uninitialised stack variable, and a well-written application then
    * passes that garbage to the next call -- which is how a refusal here
    * turned into a segfault three calls later, in CreateRenderTargetView
    * and in Draw, with a resource pointer full of ASCII. Nulling first
    * turns every one of those into an ordinary failure.
    */
   if (out)
      *out = NULL;

   if (!desc || !desc->CS.pShaderBytecode || !desc->CS.BytecodeLength)
      return E_INVALIDARG;

   /* A pipeline is compiled against its root signature's binding
    * assignment. Without one there is nothing to bind resources through,
    * so a shader that declares any would resolve to no slot at all.
    */
   const struct d12_binding_map *map;
   struct d12_root_layout embedded_layout = {0};
   struct d12_binding_map embedded_map = {0};

   const struct d12_root_layout *layout;
   if (desc->pRootSignature) {
      map = d12_pso_binding_map(desc->pRootSignature);
      layout = &((struct d12_root_signature *)desc->pRootSignature)->layout;
   } else {
      const D3D12_SHADER_BYTECODE *stages[] = {&desc->CS};
      HRESULT hr = d12_embedded_pso_binding_map(
         stages, ARRAY_SIZE(stages), &embedded_layout, &embedded_map, &map);
      if (FAILED(hr))
         return hr;
      layout = &embedded_layout;
   }

   /*
    * A cached PSO blob is driver-private.  Keep it deliberately transparent:
    * an identifying header followed by the original shader bytes and the
    * canonical parsed root layout.  Exact byte comparison rejects a cache
    * paired with different shader code or a different root signature without
    * relying on collision-prone hashes.  We still recompile after validation;
    * the blob is a correctness contract first, an optimization later.
    */
   struct d12_compute_cache_header {
      char magic[8];
      uint32_t version;
      uint32_t shader_size;
      uint32_t num_params;
      uint32_t num_ranges;
      uint32_t num_static_samplers;
      uint32_t flags;
   } header = {
      .magic = {'D', '1', '2', 'A', 'G', 'X', 'C', 'S'},
      .version = 2,
      .shader_size = desc->CS.BytecodeLength,
      .num_params = layout->num_params,
      .num_ranges = layout->num_ranges,
      .num_static_samplers = layout->num_static_samplers,
      .flags = layout->flags,
   };

   if (desc->CS.BytecodeLength > UINT32_MAX ||
       layout->num_params > D12_MAX_ROOT_PARAMS) {
      d12_root_layout_cleanup(&embedded_layout);
      return E_INVALIDARG;
   }

   size_t params_size = layout->num_params * sizeof(layout->params[0]);
   size_t ranges_size = layout->num_ranges * sizeof(layout->ranges[0]);
   size_t samplers_size =
      layout->num_static_samplers * sizeof(layout->static_samplers[0]);
   if (desc->CS.BytecodeLength >
       SIZE_MAX - sizeof(header) - params_size - ranges_size - samplers_size) {
      d12_root_layout_cleanup(&embedded_layout);
      return E_OUTOFMEMORY;
   }

   size_t cache_size = sizeof(header) + desc->CS.BytecodeLength +
                       params_size + ranges_size + samplers_size;
   uint8_t *cache = malloc(cache_size);
   if (!cache) {
      d12_root_layout_cleanup(&embedded_layout);
      return E_OUTOFMEMORY;
   }

   uint8_t *cursor = cache;
   memcpy(cursor, &header, sizeof(header));
   cursor += sizeof(header);
   memcpy(cursor, desc->CS.pShaderBytecode, desc->CS.BytecodeLength);
   cursor += desc->CS.BytecodeLength;
   memcpy(cursor, layout->params, params_size);
   cursor += params_size;
   if (ranges_size) {
      memcpy(cursor, layout->ranges, ranges_size);
      cursor += ranges_size;
   }
   if (samplers_size)
      memcpy(cursor, layout->static_samplers, samplers_size);

   if (desc->CachedPSO.CachedBlobSizeInBytes &&
       (!desc->CachedPSO.pCachedBlob ||
        desc->CachedPSO.CachedBlobSizeInBytes != cache_size ||
        memcmp(desc->CachedPSO.pCachedBlob, cache, cache_size))) {
      free(cache);
      d12_root_layout_cleanup(&embedded_layout);
      return E_INVALIDARG;
   }

   HRESULT hr = d12_pipeline_state_create_compute(
      device, desc->CS.pShaderBytecode, desc->CS.BytecodeLength, map, riid,
      out);
   if (SUCCEEDED(hr) && out && *out) {
      struct d12_pipeline_state *pso = *(void **)out;
      pso->cached_blob = cache;
      pso->cached_blob_size = cache_size;
      pso->library_key = d12_compute_library_key(desc, layout);
      pso->library_key_valid = true;
      cache = NULL;
   }
   free(cache);
   d12_root_layout_cleanup(&embedded_layout);
   return hr;
}

/*
 * ID3D12Device2::CreatePipelineState — the STREAM descriptor.
 *
 * Instead of a fixed struct, the caller hands over a byte stream of
 * tagged subobjects: each is a D3D12_PIPELINE_STATE_SUBOBJECT_TYPE
 * followed by its payload, and the whole entry is aligned to void*.
 * Mesh pipelines exist only in this form -- there is no
 * D3D12_MESH_PIPELINE_STATE_DESC.
 *
 * Unknown subobject types are refused rather than skipped: the stream
 * gives no length for an entry, so a type this driver does not know is
 * also a type it cannot step over. Guessing the size would silently
 * misparse everything after it.
 */
struct d12_pso_stream {
   const D3D12_SHADER_BYTECODE *vs, *ps, *ms, *as, *cs, *gs, *hs, *ds;
   ID3D12RootSignature *root_sig;
   const D3D12_RASTERIZER_DESC *raster;
   const D3D12_BLEND_DESC *blend;
   const D3D12_INPUT_LAYOUT_DESC *input_layout;
   const D3D12_STREAM_OUTPUT_DESC *stream_output;
   const D3D12_DEPTH_STENCIL_DESC *ds_state;
   struct D3D12_RT_FORMAT_ARRAY rt_formats;
   DXGI_SAMPLE_DESC sample_desc;
   UINT sample_mask;
   DXGI_FORMAT dsv_format;
   D3D12_PRIMITIVE_TOPOLOGY_TYPE primitive_topology;
   D3D12_INDEX_BUFFER_STRIP_CUT_VALUE strip_cut;
   D3D12_PIPELINE_STATE_FLAGS flags;
   bool have_rt_formats;
   bool have_sample_desc;
   bool have_sample_mask;
};

static bool
d12_parse_pso_stream(const D3D12_PIPELINE_STATE_STREAM_DESC *desc,
                     struct d12_pso_stream *out)
{
   const uint8_t *p = (const uint8_t *)desc->pPipelineStateSubobjectStream;
   const uint8_t *end = p + desc->SizeInBytes;
   uint64_t seen = 0;

   memset(out, 0, sizeof(*out));

   while (p + sizeof(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE) <= end) {
      D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type;
      memcpy(&type, p, sizeof(type));

      if ((unsigned)type >= 64 || (seen & BITFIELD64_BIT((unsigned)type))) {
         mesa_loge("CreatePipelineState: duplicate or invalid stream "
                   "subobject %d", (int)type);
         return false;
      }
      seen |= BITFIELD64_BIT((unsigned)type);

      const uint8_t *body = NULL;
      size_t body_offset = 0;
      size_t body_size;

#define BODY(ctype)                                                            \
   do {                                                                        \
      body_offset = ALIGN_POT(sizeof(type), _Alignof(ctype));                  \
      body = p + body_offset;                                                  \
      body_size = sizeof(ctype);                                               \
      if (body > end || (size_t)(end - body) < body_size)                     \
         return false;                                                         \
   } while (0)
#define SUBOBJ(TY, field, ctype)                                                \
   case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_##TY:                              \
      BODY(ctype);                                                             \
      out->field = (const ctype *)body;                                        \
      break

      switch (type) {
         SUBOBJ(VS, vs, D3D12_SHADER_BYTECODE);
         SUBOBJ(PS, ps, D3D12_SHADER_BYTECODE);
         SUBOBJ(MS, ms, D3D12_SHADER_BYTECODE);
         SUBOBJ(AS, as, D3D12_SHADER_BYTECODE);
         SUBOBJ(CS, cs, D3D12_SHADER_BYTECODE);
         SUBOBJ(GS, gs, D3D12_SHADER_BYTECODE);
         SUBOBJ(HS, hs, D3D12_SHADER_BYTECODE);
         SUBOBJ(DS, ds, D3D12_SHADER_BYTECODE);
         SUBOBJ(RASTERIZER, raster, D3D12_RASTERIZER_DESC);
         SUBOBJ(INPUT_LAYOUT, input_layout, D3D12_INPUT_LAYOUT_DESC);
         SUBOBJ(DEPTH_STENCIL, ds_state, D3D12_DEPTH_STENCIL_DESC);

      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1:
         /* DESC1 appends DepthBoundsTestEnable; the prefix consumed by the
          * current raster path is byte-for-byte DESC. */
         BODY(D3D12_DEPTH_STENCIL_DESC1);
         out->ds_state = (const D3D12_DEPTH_STENCIL_DESC *)body;
         break;

      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE:
         BODY(ID3D12RootSignature *);
         memcpy(&out->root_sig, body, sizeof(out->root_sig));
         break;

      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS:
         BODY(struct D3D12_RT_FORMAT_ARRAY);
         memcpy(&out->rt_formats, body, sizeof(out->rt_formats));
         out->have_rt_formats = true;
         break;

      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT:
         BODY(DXGI_FORMAT);
         memcpy(&out->dsv_format, body, sizeof(out->dsv_format));
         break;

      /* Accepted and ignored: none of these change what we build. */
      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK:
         BODY(UINT);
         break;
      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS:
         BODY(D3D12_PIPELINE_STATE_FLAGS);
         memcpy(&out->flags, body, sizeof(out->flags));
         break;
      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK:
         BODY(UINT);
         memcpy(&out->sample_mask, body, sizeof(out->sample_mask));
         out->have_sample_mask = true;
         break;
      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC:
         BODY(DXGI_SAMPLE_DESC);
         memcpy(&out->sample_desc, body, sizeof(out->sample_desc));
         out->have_sample_desc = true;
         break;
      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY:
         BODY(D3D12_PRIMITIVE_TOPOLOGY_TYPE);
         memcpy(&out->primitive_topology, body,
                sizeof(out->primitive_topology));
         break;
      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT:
         BODY(D3D12_STREAM_OUTPUT_DESC);
         out->stream_output = (const D3D12_STREAM_OUTPUT_DESC *)body;
         break;
      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE:
         BODY(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE);
         memcpy(&out->strip_cut, body, sizeof(out->strip_cut));
         break;
      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING:
         BODY(D3D12_VIEW_INSTANCING_DESC);
         break;
      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND:
         BODY(D3D12_BLEND_DESC);
         out->blend = (const D3D12_BLEND_DESC *)body;
         break;
      case D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO:
         BODY(D3D12_CACHED_PIPELINE_STATE);
         break;

      default:
         mesa_loge("CreatePipelineState: unknown stream subobject %d; the "
                   "stream carries no length so it cannot be stepped over",
                   (int)type);
         return false;
      }
#undef SUBOBJ
#undef BODY

      p += ALIGN_POT(body_offset + body_size, sizeof(void *));
   }

   return true;
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreatePipelineState(ID3D12Device2 *iface,
                               const D3D12_PIPELINE_STATE_STREAM_DESC *desc,
                               REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);

   if (!out)
      return E_POINTER;

   /* See the COM contract note above: null before validating. */
   *out = NULL;

   if (!desc || !desc->pPipelineStateSubobjectStream || !desc->SizeInBytes)
      return E_INVALIDARG;

   struct d12_pso_stream st;
   if (!d12_parse_pso_stream(desc, &st))
      return E_INVALIDARG;

   const struct d12_binding_map *map = d12_pso_binding_map(st.root_sig);

   /* A compute stream may carry ignored graphics defaults, but a vertex
    * shader selects the graphics pipeline class and cannot coexist with CS. */
   if (st.cs && st.cs->pShaderBytecode &&
       st.vs && st.vs->pShaderBytecode)
      return E_INVALIDARG;

   enum pipe_format rt_format = PIPE_FORMAT_NONE;
   if (st.have_rt_formats && st.rt_formats.NumRenderTargets >= 1 &&
       d12_format_is_renderable(st.rt_formats.RTFormats[0]))
      rt_format = d12_format_to_pipe(st.rt_formats.RTFormats[0]);

   DXGI_FORMAT dsv_format = st.dsv_format;
   if (dsv_format != DXGI_FORMAT_UNKNOWN &&
       !d12_format_is_depth_stencil(dsv_format))
      dsv_format = DXGI_FORMAT_UNKNOWN;

   unsigned nr_samples = st.have_sample_desc ? st.sample_desc.Count : 1;
   UINT sample_mask = st.have_sample_mask ? st.sample_mask : ~0u;
   if ((nr_samples != 1 && nr_samples != 2 && nr_samples != 4) ||
       (st.have_sample_desc && st.sample_desc.Quality != 0))
      return E_INVALIDARG;

   /* A compute pipeline expressed as a stream. */
   if (st.cs && st.cs->pShaderBytecode) {
      return d12_pipeline_state_create_compute(
         device, st.cs->pShaderBytecode, st.cs->BytecodeLength, map, riid,
         out);
   }

   if (st.ms && st.ms->pShaderBytecode) {
      if (!st.ps || !st.ps->pShaderBytecode) {
         mesa_loge("CreatePipelineState: a mesh pipeline needs a pixel "
                   "shader");
         return E_INVALIDARG;
      }
      return d12_pipeline_state_create_mesh(
         device, st.ms->pShaderBytecode, st.ms->BytecodeLength,
         st.as ? st.as->pShaderBytecode : NULL,
         st.as ? st.as->BytecodeLength : 0, st.ps->pShaderBytecode,
         st.ps->BytecodeLength, rt_format,
         st.have_rt_formats ? st.rt_formats.NumRenderTargets : 0,
         st.rt_formats.RTFormats, nr_samples, sample_mask, st.ds_state,
         dsv_format,
         st.raster,
         st.blend, map, riid, out);
   }

   if (st.vs && st.vs->pShaderBytecode) {
      return d12_pipeline_state_create_graphics(
         device, st.vs->pShaderBytecode, st.vs->BytecodeLength,
         st.ps ? st.ps->pShaderBytecode : NULL,
         st.ps ? st.ps->BytecodeLength : 0,
         st.gs ? st.gs->pShaderBytecode : NULL,
         st.gs ? st.gs->BytecodeLength : 0,
         st.hs ? st.hs->pShaderBytecode : NULL,
         st.hs ? st.hs->BytecodeLength : 0,
         st.ds ? st.ds->pShaderBytecode : NULL,
         st.ds ? st.ds->BytecodeLength : 0, rt_format,
         st.have_rt_formats ? st.rt_formats.NumRenderTargets : 0,
         st.rt_formats.RTFormats, nr_samples, sample_mask, st.ds_state,
         dsv_format,
         st.raster, st.raster ? (float)st.raster->DepthBias : 0.0f,
         true, true,
         st.ds_state ? st.ds_state->StencilReadMask : UINT8_MAX,
         st.ds_state ? st.ds_state->StencilWriteMask : UINT8_MAX,
         st.ds_state ? st.ds_state->StencilReadMask : UINT8_MAX,
         st.ds_state ? st.ds_state->StencilWriteMask : UINT8_MAX,
         st.blend, st.input_layout, st.stream_output, st.primitive_topology,
         st.strip_cut, st.flags, map, riid, out);
   }

   mesa_loge("CreatePipelineState: stream declares no usable pipeline");
   return E_INVALIDARG;
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreateGraphicsPipelineState(
   ID3D12Device2 *iface, const D3D12_GRAPHICS_PIPELINE_STATE_DESC *desc,
   REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);

   /* COM CONTRACT: the out-pointer is set to NULL BEFORE any validation.
    * A failing Create that leaves it untouched hands the caller back an
    * uninitialised stack variable, and a well-written application then
    * passes that garbage to the next call -- which is how a refusal here
    * turned into a segfault three calls later, in CreateRenderTargetView
    * and in Draw, with a resource pointer full of ASCII. Nulling first
    * turns every one of those into an ordinary failure.
    */
   if (out)
      *out = NULL;

   if (!desc || !desc->VS.pShaderBytecode)
      return E_INVALIDARG;

   /* Tessellation needs BOTH stages. One without the other would fall
    * through the tessellation path and draw the control points
    * untessellated -- which draws something, and something reads as a
    * pass. Refuse by name.
    */
   bool has_hs = desc->HS.pShaderBytecode && desc->HS.BytecodeLength;
   bool has_ds = desc->DS.pShaderBytecode && desc->DS.BytecodeLength;
   if (has_hs != has_ds) {
      mesa_loge("CreateGraphicsPipelineState: a %s shader is bound without "
                "a %s shader; tessellation needs both.",
                has_hs ? "hull" : "domain", has_hs ? "domain" : "hull");
      return E_INVALIDARG;
   }

   if (desc->NumRenderTargets > ARRAY_SIZE(desc->RTVFormats))
      return E_INVALIDARG;

   if ((desc->SampleDesc.Count != 1 && desc->SampleDesc.Count != 2 &&
        desc->SampleDesc.Count != 4) || desc->SampleDesc.Quality != 0)
      return E_INVALIDARG;

   /* Logic operations use one common output-merger operation. They cannot be
    * combined with independent per-target blend state on an active target.
    */
   if (desc->BlendState.IndependentBlendEnable) {
      for (unsigned i = 0; i < desc->NumRenderTargets; i++) {
         if (desc->BlendState.RenderTarget[i].LogicOpEnable)
            return E_INVALIDARG;
      }
   }

   /* Inactive RTV slots are not spare metadata storage: D3D12 requires them
    * to retain the UNKNOWN spelling beyond NumRenderTargets.
    */
   for (unsigned i = desc->NumRenderTargets;
        i < ARRAY_SIZE(desc->RTVFormats); i++) {
      if (desc->RTVFormats[i] != DXGI_FORMAT_UNKNOWN)
         return E_INVALIDARG;
   }

   /* Stream-output declarations are legal only when the root signature opts
    * in. The execution path remains explicitly unsupported, but accepting a
    * PSO that the signature itself forbids is an API validation bug.
    */
   if (desc->StreamOutput.NumEntries &&
       (!desc->pRootSignature ||
        !(((struct d12_root_signature *)desc->pRootSignature)->layout.flags &
          D3D12_ROOT_SIGNATURE_FLAG_ALLOW_STREAM_OUTPUT)))
      return E_INVALIDARG;

   /* The pipeline's declared render-target format now drives the
    * tilebuffer layout instead of an assumed RGBA8.
   */
   enum pipe_format rt_format = PIPE_FORMAT_NONE;
   if (desc->NumRenderTargets >= 1 &&
       d12_format_is_renderable(desc->RTVFormats[0]))
      rt_format = d12_format_to_pipe(desc->RTVFormats[0]);

   DXGI_FORMAT dsv_format = desc->DSVFormat;
   if (dsv_format != DXGI_FORMAT_UNKNOWN &&
       !d12_format_is_depth_stencil(dsv_format))
      dsv_format = DXGI_FORMAT_UNKNOWN;

   const struct d12_binding_map *map;
   const struct d12_root_layout *layout;
   struct d12_root_layout embedded_layout = {0};
   struct d12_binding_map embedded_map = {0};

   if (desc->pRootSignature) {
      map = d12_pso_binding_map(desc->pRootSignature);
      layout = &((struct d12_root_signature *)desc->pRootSignature)->layout;
   } else {
      const D3D12_SHADER_BYTECODE *stages[] = {
         &desc->VS, &desc->PS, &desc->GS, &desc->HS, &desc->DS,
      };
      HRESULT hr = d12_embedded_pso_binding_map(
         stages, ARRAY_SIZE(stages), &embedded_layout, &embedded_map, &map);
      if (FAILED(hr))
         return hr;
      layout = &embedded_layout;
   }

   /* The cached blob GetCachedBlob returns, and the contract a supplied
    * CachedPSO has to meet byte for byte (same rule as compute). */
   const struct d12_pipeline_library_key key =
      d12_graphics_library_key(desc, layout);
   size_t cache_size = 0;
   uint8_t *cache = d12_graphics_cache_blob(desc, layout, key, &cache_size);
   if (!cache) {
      d12_root_layout_cleanup(&embedded_layout);
      return E_OUTOFMEMORY;
   }
   if (desc->CachedPSO.CachedBlobSizeInBytes &&
       (!desc->CachedPSO.pCachedBlob ||
        desc->CachedPSO.CachedBlobSizeInBytes != cache_size ||
        memcmp(desc->CachedPSO.pCachedBlob, cache, cache_size))) {
      free(cache);
      d12_root_layout_cleanup(&embedded_layout);
      return E_INVALIDARG;
   }

   HRESULT hr = d12_pipeline_state_create_graphics(
      device, desc->VS.pShaderBytecode, desc->VS.BytecodeLength,
      desc->PS.pShaderBytecode, desc->PS.BytecodeLength,
      desc->GS.pShaderBytecode, desc->GS.BytecodeLength,
      desc->HS.pShaderBytecode, desc->HS.BytecodeLength,
      desc->DS.pShaderBytecode, desc->DS.BytecodeLength, rt_format,
      desc->NumRenderTargets, desc->RTVFormats, desc->SampleDesc.Count,
      desc->SampleMask, &desc->DepthStencilState,
      dsv_format, &desc->RasterizerState,
      (float)desc->RasterizerState.DepthBias,
      true, true,
      desc->DepthStencilState.StencilReadMask,
      desc->DepthStencilState.StencilWriteMask,
      desc->DepthStencilState.StencilReadMask,
      desc->DepthStencilState.StencilWriteMask, &desc->BlendState,
      &desc->InputLayout, &desc->StreamOutput, desc->PrimitiveTopologyType,
      desc->IBStripCutValue, desc->Flags, map, riid, out);
   if (SUCCEEDED(hr) && out && *out) {
      struct d12_pipeline_state *pso = *(void **)out;
      pso->cached_blob = cache;
      pso->cached_blob_size = cache_size;
      pso->library_key = key;
      pso->library_key_valid = true;
      cache = NULL;
   }
   free(cache);
   d12_root_layout_cleanup(&embedded_layout);
   return hr;
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreateRootSignature(ID3D12Device2 *iface, UINT node_mask,
                               const void *blob, SIZE_T blob_size,
                               REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);

   /* COM CONTRACT: the out-pointer is set to NULL BEFORE any validation.
    * A failing Create that leaves it untouched hands the caller back an
    * uninitialised stack variable, and a well-written application then
    * passes that garbage to the next call -- which is how a refusal here
    * turned into a segfault three calls later, in CreateRenderTargetView
    * and in Draw, with a resource pointer full of ASCII. Nulling first
    * turns every one of those into an ordinary failure.
    */
   if (out)
      *out = NULL;

   if (node_mask > 1)
      return E_INVALIDARG;

   return d12_root_signature_create(device, blob, blob_size, riid, out);
}

/*
 * D3D12_RESOURCE_ALLOCATION_INFO for a set of resource descriptions.
 *
 * This is a pure query -- no allocation happens -- but it has to agree with
 * what CreateCommittedResource would actually allocate, so it runs the same
 * ail layout maths rather than a second estimate that could drift.
 *
 * It was the single largest one-blocker cluster in the suite: 11 tests where
 * an unimplemented GetResourceAllocationInfo was the ONLY thing standing
 * between them and a verdict. The stub returned {0, 0}, and a zero size is
 * not a refusal an application can see -- it silently produces a heap too
 * small for the resource it was sized for.
 *
 * Alignment follows D3D12's documented defaults: 4 MB for multisampled
 * resources, 64 KB otherwise. Buffers are their own byte length, aligned to
 * 64 KB. Sizes accumulate with each resource aligned to its own requirement,
 * which is what a caller placing them consecutively in one heap needs.
 */
/*
 * DirectX-Headers C interfaces use an explicit aggregate return pointer on
 * Windows and a by-value return on Linux. Keep this method and its internal
 * caller on the same platform ABI as the vtable, using the existing helpers.
 */
/*
 * Resolve an abstract heap type into the concrete properties this adapter
 * would use for it.  Every answer here is a UMA answer: there is one memory
 * pool, so the pool preference is always L0 and the only thing that varies is
 * how the CPU sees the memory.
 *
 * GPU_UPLOAD is write-combined because that is what it is for -- the CPU
 * streams into it and never reads back, and write-combining is what makes
 * those writes cheap.  UPLOAD is write-combined for the same reason;
 * READBACK is write-back because the CPU reads it.
 */
static D12_AGGREGATE_RETURN(D3D12_HEAP_PROPERTIES) STDMETHODCALLTYPE
d12_device_GetCustomHeapProperties(ID3D12Device2 *iface
                                      D12_AGGREGATE_OUT(D3D12_HEAP_PROPERTIES),
                                   UINT node_mask, D3D12_HEAP_TYPE heap_type)
{
   D3D12_HEAP_PROPERTIES props = {
      .Type = D3D12_HEAP_TYPE_CUSTOM,
      .MemoryPoolPreference = D3D12_MEMORY_POOL_L0,
      .CreationNodeMask = node_mask ? node_mask : 1,
      .VisibleNodeMask = node_mask ? node_mask : 1,
   };

   (void)iface;

   switch (heap_type) {
   case D3D12_HEAP_TYPE_DEFAULT:
      props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE;
      break;
   case D3D12_HEAP_TYPE_UPLOAD:
   case D3D12_HEAP_TYPE_GPU_UPLOAD:
      props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE;
      break;
   case D3D12_HEAP_TYPE_READBACK:
      props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
      break;
   default:
      /* CUSTOM has no abstract properties to resolve, and an unknown type
       * has none either. D3D12 answers both with a zeroed structure. */
      props = (D3D12_HEAP_PROPERTIES){0};
      break;
   }

   D12_RETURN_AGGREGATE(props);
}

static D3D12_RESOURCE_ALLOCATION_INFO
d12_device_compute_allocation_info(
   ID3D12Device2 *iface, UINT visible_mask, UINT count,
   const D3D12_RESOURCE_DESC *descs, D3D12_RESOURCE_ALLOCATION_INFO1 *info1)
{
   struct d12_device *device = d12_device_from_iface(iface);
   D3D12_RESOURCE_ALLOCATION_INFO info = {
      UINT64_MAX, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
   };

   if (visible_mask > 1)
      return info;

   /*
    * An EMPTY query is not an invalid one. D3D12 answers count == 0 with
    * {SizeInBytes = 0, Alignment = 1}, which test_resource_allocation_info
    * asserts directly. UINT64_MAX is the answer for a description that
    * cannot be allocated, not for asking about nothing -- I had those two
    * cases conflated.
    */
   if (!count) {
      info.SizeInBytes = 0;
      info.Alignment = 1;
      return info;
   }

   if (!descs) {
      info.SizeInBytes = UINT64_MAX;
      info.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
      return info;
   }

   uint64_t total = 0;
   /* Each resource below contributes its actual placement requirement.
    * Starting at the ordinary 64 KiB default prevents a single qualifying
    * small texture from ever reporting its requested 4 KiB alignment. */
   uint64_t max_align = 1;

   for (UINT i = 0; i < count; i++) {
      const D3D12_RESOURCE_DESC *d = &descs[i];
      uint64_t align = d->SampleDesc.Count > 1
                          ? D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT
                          : D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
      uint64_t size;
      uint64_t slice0_B = 0;

      /*
       * desc->Alignment is an INPUT: the caller may REQUEST an alignment,
       * and D3D12 refuses one it cannot honour by answering
       * {UINT64_MAX, default}. 0 means "pick for me".
       *
       * test_resource_allocation_info asserts this directly, including that
       * a small-alignment (4 KB) request is refused for a texture too large
       * to be placed that way. Ignoring the field -- which is what this did
       * at first -- turns every one of those refusals into a plausible
       * number the caller then trusts.
       */
      bool want_small =
         d->Alignment == D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT &&
         d->Dimension != D3D12_RESOURCE_DIMENSION_BUFFER &&
         d->SampleDesc.Count <= 1 &&
         !(d->Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                       D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));

      /*
       * MSAA defaults to 4 MB placement, but an application MAY explicitly
       * ask for the ordinary 64 KB and D3D12 honours it -- that is the
       * "workaround" test_placed_msaa_alignment_workaround checks, and it
       * asserts both halves: Alignment == 0 gives 4 MB, an explicit 64 KB
       * request gives 64 KB. Treating 4 MB as mandatory refuses a legal
       * query.
       */
      if (d->SampleDesc.Count > 1 &&
          d->Alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT)
         align = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

      /* Any other explicit request we cannot honour is refused outright.
       * The small-alignment request is decided AFTER the size is known,
       * because whether it qualifies depends on the size (below).
       */
      if (d->Alignment != 0 && d->Alignment != align && !want_small) {
         info.SizeInBytes = UINT64_MAX;
         info.Alignment = align;
         return info;
      }

      if (d->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
         if (!d->Width || d->Height != 1 || d->DepthOrArraySize != 1 ||
             d->MipLevels != 1 || d->Format != DXGI_FORMAT_UNKNOWN ||
             d->SampleDesc.Count != 1 || d->SampleDesc.Quality != 0 ||
             d->Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR)
            return info;
         size = d->Width;
      } else {
         /* The resource constructor and this reflection query share one
          * layout calculation. A format or concrete layout we cannot express
          * has no honest size, so fail rather than returning a plausible heap
          * that creation would later overrun or reject. */
         enum pipe_format pipe = d12_format_is_depth_stencil(d->Format)
                                    ? d12_format_depth_plane(d->Format)
                                    : d12_format_to_pipe(d->Format);
         if (pipe == PIPE_FORMAT_NONE ||
             FAILED(d12_resource_get_allocation_size(
                device, d, 0, 0, false, &size))) {
            /* A refusal still has to report a LEGAL alignment: the caller
             * feeds it straight into align(), which asserts on zero. The
             * test checks `Alignment >= D3D12_DEFAULT_RESOURCE_PLACEMENT_
             * ALIGNMENT` even on the UINT64_MAX path. */
            info.SizeInBytes = UINT64_MAX;
            info.Alignment = align;
            return info;
         }

         /* Block-aware size of ONE SLICE OF LEVEL 0; the small-placement
          * test below is on this, not on the total. */
         slice0_B =
            (uint64_t)util_format_get_nblocksx(pipe, (uint32_t)d->Width) *
                    util_format_get_nblocksy(pipe, d->Height) *
                    util_format_get_blocksize(pipe);
      }

      /*
       * SMALL RESOURCE PLACEMENT. 4 KB alignment is only legal when the
       * resource actually fits the small-resource budget; D3D12 refuses the
       * request otherwise rather than quietly upgrading it. Deciding this
       * on the real layout size is what the guess-by-dimensions version got
       * wrong -- the suite checks a texture that is small in width and
       * height but too large once its format and mip chain are counted.
       */
      if (want_small) {
         /*
          * The 4 KB threshold is on ONE SLICE OF LEVEL 0, block-aware --
          * not on the total. The suite pins this precisely: 256x512 BC1 is
          * exactly 64 KB per slice and qualifies even with 10 mips or 64
          * array layers, while 260x512 BC1 (66.5 KB) and 1024x1024 RGBA8
          * do not. Testing the total size instead refuses the 64-layer and
          * 1024-layer cases, which is what the first version did.
          */
         if (slice0_B <= D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT) {
            align = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT;
         } else {
            info.SizeInBytes = UINT64_MAX;
            info.Alignment = align;
            return info;
         }
      }

      uint64_t offset, padded_size;
      if (!d12_allocation_append(size, align, &total, &max_align,
                                 &offset, &padded_size)) {
         info.Alignment = MAX2(max_align, align);
         return info;
      }
      if (info1) {
         info1[i] = (D3D12_RESOURCE_ALLOCATION_INFO1){
            .Offset = offset,
            .Alignment = align,
            .SizeInBytes = padded_size,
         };
      }
   }

   if (!d12_allocation_align(total, max_align, &info.SizeInBytes)) {
      info.SizeInBytes = UINT64_MAX;
      info.Alignment = max_align;
      return info;
   }
   info.Alignment = max_align;
   return info;
}

static D12_AGGREGATE_RETURN(D3D12_RESOURCE_ALLOCATION_INFO) STDMETHODCALLTYPE
d12_device_GetResourceAllocationInfo(
   ID3D12Device2 *iface D12_AGGREGATE_OUT(D3D12_RESOURCE_ALLOCATION_INFO),
   UINT visible_mask, UINT count, const D3D12_RESOURCE_DESC *descs)
{
   D3D12_RESOURCE_ALLOCATION_INFO info = d12_device_compute_allocation_info(
      iface, visible_mask, count, descs, NULL);
   D12_RETURN_AGGREGATE(info);
}

void
d12_device_get_resource_allocation_info_runtime(
   struct d12_device *device, UINT visible_mask,
   const D3D12_RESOURCE_DESC *desc,
   D3D12_RESOURCE_ALLOCATION_INFO *info)
{
   if (!info)
      return;
   *info = (D3D12_RESOURCE_ALLOCATION_INFO){
      .SizeInBytes = UINT64_MAX,
      .Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
   };
   if (!device || !desc)
      return;
#ifdef _WIN32
   (void)d12_device_GetResourceAllocationInfo(
      (ID3D12Device2 *)&device->iface, info, visible_mask, 1, desc);
#else
   *info = d12_device_GetResourceAllocationInfo(
      (ID3D12Device2 *)&device->iface, visible_mask, 1, desc);
#endif
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreateHeap(ID3D12Device2 *iface, const D3D12_HEAP_DESC *desc,
                      REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);

   if (out)
      *out = NULL;

   return d12_heap_create(device, desc, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreatePlacedResource(ID3D12Device2 *iface, ID3D12Heap *heap_iface,
                                UINT64 heap_offset,
                                const D3D12_RESOURCE_DESC *desc,
                                D3D12_RESOURCE_STATES initial_state,
                                const D3D12_CLEAR_VALUE *optimized_clear,
                                REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);

   if (out)
      *out = NULL;

   if (!d12_heap_iface_is_ours(heap_iface) || !desc)
      return E_INVALIDARG;

   return d12_resource_create_placed(device, d12_memheap_from_iface(heap_iface),
                                     heap_offset, desc, 0, 0, initial_state,
                                     optimized_clear, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreateReservedResource(ID3D12Device2 *iface,
                                  const D3D12_RESOURCE_DESC *desc,
                                  D3D12_RESOURCE_STATES initial_state,
                                  const D3D12_CLEAR_VALUE *optimized_clear,
                                  REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);

   return d12_resource_create_reserved(device, desc, initial_state,
                                       optimized_clear, riid, out);
}

static void STDMETHODCALLTYPE
d12_device_GetResourceTiling(
   ID3D12Device2 *iface, ID3D12Resource *resource_iface,
   UINT *num_tiles_for_resource, D3D12_PACKED_MIP_INFO *packed_mips,
   D3D12_TILE_SHAPE *tile_shape, UINT *num_subresource_tilings,
   UINT first_subresource, D3D12_SUBRESOURCE_TILING *subresource_tilings)
{
   if (!resource_iface)
      return;

   struct d12_resource *res = d12_resource_from_iface(resource_iface);
   if (!res->reserved)
      return;

   if (num_tiles_for_resource)
      *num_tiles_for_resource = res->sparse_tile_count;
   if (packed_mips)
      *packed_mips = res->sparse_packed_mips;
   if (tile_shape)
      *tile_shape = res->sparse_tile_shape;

   if (!num_subresource_tilings)
      return;

   UINT available = first_subresource < res->sparse_subresource_count
                       ? res->sparse_subresource_count - first_subresource
                       : 0;
   UINT count = MIN2(*num_subresource_tilings, available);
   if (subresource_tilings && count) {
      memcpy(subresource_tilings, res->sparse_tilings + first_subresource,
             (size_t)count * sizeof(*subresource_tilings));
   }
   *num_subresource_tilings = count;
}

static struct d12_command_signature *
d12_command_signature_from_iface(ID3D12CommandSignature *iface)
{
   return (struct d12_command_signature *)iface;
}

struct d12_pipeline_library {
   ID3D12PipelineLibrary iface; /* must be first */
   uint32_t refcount;
   struct d12_private_data_store private_data;
   struct d12_device *device; /* holds a device ref */
   /* Games store and load from several threads at once. The lock covers
    * the entry array and the name index; pipeline creation on a load runs
    * outside it and the first creation to finish is the one kept. */
   simple_mtx_t lock;
   struct d12_pipeline_library_entry {
      uint16_t *name; /* owned, NUL-terminated */
      uint32_t name_length;
      bool is_compute;
      struct d12_pipeline_library_key key;
      struct d12_pipeline_state *pipeline; /* hidden library ownership */
   } *entries;
   uint32_t entry_count;
   uint32_t entry_capacity;
   /* entry->name -> entry index, so lookups do not scan the array. */
   struct hash_table *by_name;
};

static uint32_t
d12_library_name_hash(const void *key)
{
   const uint16_t *name = key;
   size_t length = 0;
   while (name[length])
      length++;
   return _mesa_hash_data(name, length * sizeof(*name));
}

static bool
d12_library_name_equal(const void *a_, const void *b_)
{
   const uint16_t *a = a_, *b = b_;
   size_t i = 0;
   while (a[i] && a[i] == b[i])
      i++;
   return a[i] == b[i];
}

struct d12_pipeline_library_blob_header {
   char magic[8];
   uint32_t version;
   uint32_t entry_count;
   uint64_t total_size;
};

struct d12_pipeline_library_blob_entry {
   uint32_t name_length;
   uint32_t is_compute;
   struct d12_pipeline_library_key key;
};

static const char d12_pipeline_library_magic[8] = {
   'D', '1', '2', 'A', 'G', 'X', 'P', 'L'
};

/* Version 2: the entry key became the XXH64 chain in d12_library_key_add.
 * Older blobs are refused, so the application creates a fresh library. */
#define D12_PIPELINE_LIBRARY_BLOB_VERSION 2

static inline struct d12_pipeline_library *
d12_pipeline_library_from_iface(ID3D12PipelineLibrary *iface)
{
   return (struct d12_pipeline_library *)iface;
}

static HRESULT STDMETHODCALLTYPE
d12_pipeline_library_QueryInterface(ID3D12PipelineLibrary *iface,
                                    REFIID riid, void **out)
{
   struct d12_pipeline_library *library =
      d12_pipeline_library_from_iface(iface);

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D12Object) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceChild) ||
       d12_guid_equal(riid, &IID_ID3D12PipelineLibrary)) {
      p_atomic_inc(&library->refcount);
      *out = iface;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&library->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12PipelineLibrary::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_pipeline_library_AddRef(ID3D12PipelineLibrary *iface)
{
   return p_atomic_inc_return(
      &d12_pipeline_library_from_iface(iface)->refcount);
}

static ULONG STDMETHODCALLTYPE
d12_pipeline_library_Release(ID3D12PipelineLibrary *iface)
{
   struct d12_pipeline_library *library =
      d12_pipeline_library_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&library->refcount);

   if (!refs) {
      for (uint32_t i = 0; i < library->entry_count; i++) {
         if (library->entries[i].pipeline)
            d12_pipeline_state_library_unref(library->entries[i].pipeline);
         free(library->entries[i].name);
      }
      free(library->entries);
      if (library->by_name)
         _mesa_hash_table_destroy(library->by_name, NULL);
      simple_mtx_destroy(&library->lock);
      d12_private_data_store_destroy(&library->private_data);
      d12_device_release(library->device);
      free(library);
   }

   return refs;
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_pipeline_library, ID3D12PipelineLibrary,
                                d12_pipeline_library_from_iface)

static HRESULT STDMETHODCALLTYPE
d12_pipeline_library_GetDevice(ID3D12PipelineLibrary *iface, REFIID riid,
                               void **out)
{
   struct d12_pipeline_library *library =
      d12_pipeline_library_from_iface(iface);
   return ID3D12Device_QueryInterface(&library->device->iface, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_pipeline_library_StorePipeline(ID3D12PipelineLibrary *iface, LPCWSTR name,
                                   ID3D12PipelineState *pipeline)
{
   struct d12_pipeline_library *library =
      d12_pipeline_library_from_iface(iface);
   if (!name || !pipeline)
      return E_INVALIDARG;

   const uint16_t *wire_name = (const uint16_t *)name;
   size_t name_length = 0;
   while (wire_name[name_length])
      name_length++;
   if (!name_length || name_length > UINT32_MAX)
      return E_INVALIDARG;

   struct d12_pipeline_state *pso = (struct d12_pipeline_state *)pipeline;
   if (pso->device != library->device || !pso->library_key_valid)
      return E_INVALIDARG;

   uint16_t *owned_name = malloc((name_length + 1) * sizeof(uint16_t));
   if (!owned_name)
      return E_OUTOFMEMORY;
   memcpy(owned_name, wire_name, name_length * sizeof(uint16_t));
   owned_name[name_length] = 0;

   simple_mtx_lock(&library->lock);
   HRESULT hr = S_OK;
   if (_mesa_hash_table_search(library->by_name, owned_name)) {
      hr = E_INVALIDARG;
      goto out;
   }

   if (library->entry_count == library->entry_capacity) {
      uint32_t capacity = library->entry_capacity
                             ? library->entry_capacity * 2
                             : 4;
      if (capacity < library->entry_capacity) {
         hr = E_OUTOFMEMORY;
         goto out;
      }
      void *entries = realloc(library->entries,
                              capacity * sizeof(library->entries[0]));
      if (!entries) {
         hr = E_OUTOFMEMORY;
         goto out;
      }
      library->entries = entries;
      library->entry_capacity = capacity;
   }

   const uint32_t index = library->entry_count;
   if (!_mesa_hash_table_insert(library->by_name, owned_name,
                                (void *)(uintptr_t)index)) {
      hr = E_OUTOFMEMORY;
      goto out;
   }

   library->entries[library->entry_count++] =
      (struct d12_pipeline_library_entry){
         .name = owned_name,
         .name_length = name_length,
         .is_compute = pso->is_compute,
         .key = pso->library_key,
         .pipeline = pso,
      };
   owned_name = NULL;
   d12_pipeline_state_library_ref(pso);

out:
   simple_mtx_unlock(&library->lock);
   free(owned_name);
   return hr;
}

/* The index of the named entry, or UINT32_MAX; the lock is held. */
static uint32_t
d12_pipeline_library_find_locked(struct d12_pipeline_library *library,
                                 LPCWSTR name)
{
   if (!name || !name[0])
      return UINT32_MAX;

   struct hash_entry *ent =
      _mesa_hash_table_search(library->by_name, (const uint16_t *)name);
   return ent ? (uint32_t)(uintptr_t)ent->data : UINT32_MAX;
}

/*
 * Load: an entry that already holds its pipeline (stored in this library
 * object, or loaded earlier) hands it back. A deserialized entry has only
 * its validation key, so the description is created through the device,
 * whose shader cache reuses the compiled stages when the same bytecode was
 * compiled in this process. Creation runs outside the library lock; when
 * two threads load the same name at once the first result is kept.
 */
static HRESULT
d12_pipeline_library_load(struct d12_pipeline_library *library, LPCWSTR name,
                          bool compute,
                          const struct d12_pipeline_library_key *key,
                          HRESULT (*create)(struct d12_pipeline_library *,
                                            const void *, REFIID, void **),
                          const void *desc, REFIID riid, void **out)
{
   simple_mtx_lock(&library->lock);
   const uint32_t index = d12_pipeline_library_find_locked(library, name);
   if (index == UINT32_MAX || library->entries[index].is_compute != compute ||
       !d12_library_key_equal(*key, library->entries[index].key)) {
      simple_mtx_unlock(&library->lock);
      return E_INVALIDARG;
   }
   struct d12_pipeline_state *stored = library->entries[index].pipeline;
   if (stored)
      d12_pipeline_state_library_ref(stored);
   simple_mtx_unlock(&library->lock);

   if (stored) {
      HRESULT hr = ID3D12PipelineState_QueryInterface(&stored->iface, riid,
                                                      out);
      d12_pipeline_state_library_unref(stored);
      return hr;
   }

   HRESULT hr = create(library, desc, riid, out);
   if (FAILED(hr))
      return hr;

   struct d12_pipeline_state *pso = *out;
   if (!pso->library_key_valid || !d12_library_key_equal(pso->library_key, *key)) {
      ID3D12PipelineState_Release(&pso->iface);
      *out = NULL;
      return E_INVALIDARG;
   }

   simple_mtx_lock(&library->lock);
   struct d12_pipeline_library_entry *entry = &library->entries[index];
   if (entry->pipeline) {
      /* Another thread loaded this name meanwhile: return its pipeline. */
      stored = entry->pipeline;
      d12_pipeline_state_library_ref(stored);
      simple_mtx_unlock(&library->lock);
      ID3D12PipelineState_Release(&pso->iface);
      hr = ID3D12PipelineState_QueryInterface(&stored->iface, riid, out);
      d12_pipeline_state_library_unref(stored);
      return hr;
   }
   entry->pipeline = pso;
   d12_pipeline_state_library_ref(pso);
   simple_mtx_unlock(&library->lock);
   return S_OK;
}

static HRESULT
d12_pipeline_library_create_graphics(struct d12_pipeline_library *library,
                                     const void *desc, REFIID riid,
                                     void **out)
{
   return ID3D12Device_CreateGraphicsPipelineState(&library->device->iface,
                                                   desc, riid, out);
}

static HRESULT
d12_pipeline_library_create_compute(struct d12_pipeline_library *library,
                                    const void *desc, REFIID riid, void **out)
{
   return ID3D12Device_CreateComputePipelineState(&library->device->iface,
                                                  desc, riid, out);
}

static HRESULT
d12_compute_desc_library_key(const D3D12_COMPUTE_PIPELINE_STATE_DESC *desc,
                             struct d12_pipeline_library_key *key)
{
   if (!desc || !desc->CS.pShaderBytecode || !desc->CS.BytecodeLength)
      return E_INVALIDARG;

   struct d12_root_layout embedded_layout = {0};
   struct d12_binding_map embedded_map = {0};
   const struct d12_binding_map *unused_map;
   const struct d12_root_layout *layout;

   if (desc->pRootSignature) {
      layout = &((struct d12_root_signature *)desc->pRootSignature)->layout;
   } else {
      const D3D12_SHADER_BYTECODE *stages[] = {&desc->CS};
      HRESULT hr = d12_embedded_pso_binding_map(
         stages, ARRAY_SIZE(stages), &embedded_layout, &embedded_map,
         &unused_map);
      if (FAILED(hr))
         return hr;
      layout = &embedded_layout;
   }

   *key = d12_compute_library_key(desc, layout);
   d12_root_layout_cleanup(&embedded_layout);
   return S_OK;
}

static HRESULT
d12_graphics_desc_library_key(const D3D12_GRAPHICS_PIPELINE_STATE_DESC *desc,
                              struct d12_pipeline_library_key *key)
{
   if (!desc || !desc->VS.pShaderBytecode || !desc->VS.BytecodeLength)
      return E_INVALIDARG;

   struct d12_root_layout embedded_layout = {0};
   struct d12_binding_map embedded_map = {0};
   const struct d12_binding_map *unused_map;
   const struct d12_root_layout *layout;

   if (desc->pRootSignature) {
      layout = &((struct d12_root_signature *)desc->pRootSignature)->layout;
   } else {
      const D3D12_SHADER_BYTECODE *stages[] = {
         &desc->VS, &desc->PS, &desc->GS, &desc->HS, &desc->DS,
      };
      HRESULT hr = d12_embedded_pso_binding_map(
         stages, ARRAY_SIZE(stages), &embedded_layout, &embedded_map,
         &unused_map);
      if (FAILED(hr))
         return hr;
      layout = &embedded_layout;
   }

   *key = d12_graphics_library_key(desc, layout);
   d12_root_layout_cleanup(&embedded_layout);
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_pipeline_library_LoadGraphicsPipeline(
   ID3D12PipelineLibrary *iface, LPCWSTR name,
   const D3D12_GRAPHICS_PIPELINE_STATE_DESC *desc, REFIID riid, void **out)
{
   if (out)
      *out = NULL;
   if (!out)
      return E_POINTER;

   struct d12_pipeline_library *library =
      d12_pipeline_library_from_iface(iface);

   struct d12_pipeline_library_key key;
   HRESULT hr = d12_graphics_desc_library_key(desc, &key);
   if (FAILED(hr))
      return E_INVALIDARG;

   return d12_pipeline_library_load(library, name, false, &key,
                                    d12_pipeline_library_create_graphics,
                                    desc, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_pipeline_library_LoadComputePipeline(
   ID3D12PipelineLibrary *iface, LPCWSTR name,
   const D3D12_COMPUTE_PIPELINE_STATE_DESC *desc, REFIID riid, void **out)
{
   if (out)
      *out = NULL;
   if (!out)
      return E_POINTER;

   struct d12_pipeline_library *library =
      d12_pipeline_library_from_iface(iface);

   struct d12_pipeline_library_key key;
   HRESULT hr = d12_compute_desc_library_key(desc, &key);
   if (FAILED(hr))
      return E_INVALIDARG;

   return d12_pipeline_library_load(library, name, true, &key,
                                    d12_pipeline_library_create_compute,
                                    desc, riid, out);
}

static SIZE_T STDMETHODCALLTYPE
d12_pipeline_library_GetSerializedSize(ID3D12PipelineLibrary *iface)
{
   struct d12_pipeline_library *library =
      d12_pipeline_library_from_iface(iface);
   size_t size = sizeof(struct d12_pipeline_library_blob_header);
   simple_mtx_lock(&library->lock);
   for (uint32_t i = 0; i < library->entry_count; i++) {
      const struct d12_pipeline_library_entry *entry = &library->entries[i];
      size_t name_size = (size_t)entry->name_length * sizeof(uint16_t);
      if (name_size > SIZE_MAX - size -
                         sizeof(struct d12_pipeline_library_blob_entry)) {
         size = 0;
         break;
      }
      size += sizeof(struct d12_pipeline_library_blob_entry) + name_size;
   }
   simple_mtx_unlock(&library->lock);
   return size;
}

static HRESULT STDMETHODCALLTYPE
d12_pipeline_library_Serialize(ID3D12PipelineLibrary *iface, void *data,
                               SIZE_T size)
{
   struct d12_pipeline_library *library =
      d12_pipeline_library_from_iface(iface);
   if (!data)
      return E_POINTER;

   simple_mtx_lock(&library->lock);
   size_t required = sizeof(struct d12_pipeline_library_blob_header);
   for (uint32_t i = 0; i < library->entry_count; i++) {
      required += sizeof(struct d12_pipeline_library_blob_entry) +
                  (size_t)library->entries[i].name_length * sizeof(uint16_t);
   }
   if (size < required) {
      simple_mtx_unlock(&library->lock);
      return E_INVALIDARG;
   }

   struct d12_pipeline_library_blob_header header = {
      .version = D12_PIPELINE_LIBRARY_BLOB_VERSION,
      .entry_count = library->entry_count,
      .total_size = required,
   };
   memcpy(header.magic, d12_pipeline_library_magic, sizeof(header.magic));

   uint8_t *cursor = data;
   memcpy(cursor, &header, sizeof(header));
   cursor += sizeof(header);
   for (uint32_t i = 0; i < library->entry_count; i++) {
      const struct d12_pipeline_library_entry *entry = &library->entries[i];
      struct d12_pipeline_library_blob_entry serialized = {
         .name_length = entry->name_length,
         .is_compute = entry->is_compute,
         .key = entry->key,
      };
      memcpy(cursor, &serialized, sizeof(serialized));
      cursor += sizeof(serialized);
      size_t name_size = (size_t)entry->name_length * sizeof(uint16_t);
      memcpy(cursor, entry->name, name_size);
      cursor += name_size;
   }
   simple_mtx_unlock(&library->lock);

   return S_OK;
}

static const ID3D12PipelineLibraryVtbl d12_pipeline_library_vtbl = {
   .QueryInterface = d12_pipeline_library_QueryInterface,
   .AddRef = d12_pipeline_library_AddRef,
   .Release = d12_pipeline_library_Release,
   .GetPrivateData = d12_pipeline_library_GetPrivateData,
   .SetPrivateData = d12_pipeline_library_SetPrivateData,
   .SetPrivateDataInterface = d12_pipeline_library_SetPrivateDataInterface,
   .SetName = d12_pipeline_library_SetName,
   .GetDevice = d12_pipeline_library_GetDevice,
   .StorePipeline = d12_pipeline_library_StorePipeline,
   .LoadGraphicsPipeline = d12_pipeline_library_LoadGraphicsPipeline,
   .LoadComputePipeline = d12_pipeline_library_LoadComputePipeline,
   .GetSerializedSize = d12_pipeline_library_GetSerializedSize,
   .Serialize = d12_pipeline_library_Serialize,
};

static HRESULT
d12_pipeline_library_deserialize(struct d12_pipeline_library *library,
                                 const void *blob, size_t blob_size)
{
   if (blob_size < sizeof(struct d12_pipeline_library_blob_header))
      return E_INVALIDARG;

   struct d12_pipeline_library_blob_header header;
   memcpy(&header, blob, sizeof(header));
   if (memcmp(header.magic, d12_pipeline_library_magic,
              sizeof(header.magic)) ||
       header.version != D12_PIPELINE_LIBRARY_BLOB_VERSION ||
       header.total_size != blob_size ||
       header.entry_count > (blob_size - sizeof(header)) /
                               sizeof(struct d12_pipeline_library_blob_entry))
      return E_INVALIDARG;

   if (header.entry_count) {
      library->entries = calloc(header.entry_count,
                                sizeof(library->entries[0]));
      if (!library->entries)
         return E_OUTOFMEMORY;
      library->entry_capacity = header.entry_count;
   }

   const uint8_t *cursor = (const uint8_t *)blob + sizeof(header);
   const uint8_t *end = (const uint8_t *)blob + blob_size;
   for (uint32_t i = 0; i < header.entry_count; i++) {
      if ((size_t)(end - cursor) <
          sizeof(struct d12_pipeline_library_blob_entry))
         return E_INVALIDARG;

      struct d12_pipeline_library_blob_entry serialized;
      memcpy(&serialized, cursor, sizeof(serialized));
      cursor += sizeof(serialized);
      if (!serialized.name_length || serialized.is_compute > 1 ||
          serialized.name_length >
             (size_t)(end - cursor) / sizeof(uint16_t))
         return E_INVALIDARG;

      size_t name_size =
         (size_t)serialized.name_length * sizeof(uint16_t);
      uint16_t *name = malloc(name_size + sizeof(uint16_t));
      if (!name)
         return E_OUTOFMEMORY;
      memcpy(name, cursor, name_size);
      name[serialized.name_length] = 0;
      cursor += name_size;

      /* Embedded NULs would make lookup ambiguous; duplicate names make
       * Store/Load semantics ambiguous as well. Reject both on import. */
      bool invalid = false;
      for (uint32_t j = 0; j < serialized.name_length; j++)
         invalid |= name[j] == 0;
      invalid |= _mesa_hash_table_search(library->by_name, name) != NULL;
      if (invalid) {
         free(name);
         return E_INVALIDARG;
      }
      if (!_mesa_hash_table_insert(library->by_name, name,
                                   (void *)(uintptr_t)library->entry_count)) {
         free(name);
         return E_OUTOFMEMORY;
      }

      library->entries[library->entry_count++] =
         (struct d12_pipeline_library_entry){
            .name = name,
            .name_length = serialized.name_length,
            .is_compute = serialized.is_compute,
            .key = serialized.key,
         };
   }

   return cursor == end ? S_OK : E_INVALIDARG;
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreatePipelineLibrary(ID3D12Device2 *iface,
                                 const void *library_blob,
                                 SIZE_T library_blob_length, REFIID riid,
                                 void **out)
{
   if (!riid && !out && !library_blob && !library_blob_length)
      return S_FALSE;

   if (!out)
      return E_POINTER;
   *out = NULL;

   if (!riid || (!!library_blob != !!library_blob_length))
      return E_INVALIDARG;

   struct d12_pipeline_library *library = calloc(1, sizeof(*library));
   if (!library)
      return E_OUTOFMEMORY;

   library->iface.lpVtbl =
      (ID3D12PipelineLibraryVtbl *)&d12_pipeline_library_vtbl;
   library->refcount = 1;
   library->device = d12_device_from_iface(iface);
   d12_device_add_ref(library->device);
   d12_private_data_store_init(&library->private_data,
                               (IUnknown *)&library->iface);
   simple_mtx_init(&library->lock, mtx_plain);
   library->by_name = _mesa_hash_table_create(NULL, d12_library_name_hash,
                                              d12_library_name_equal);

   HRESULT hr = library->by_name ? S_OK : E_OUTOFMEMORY;
   if (SUCCEEDED(hr) && library_blob)
      hr = d12_pipeline_library_deserialize(library, library_blob,
                                            library_blob_length);
   if (FAILED(hr)) {
      d12_pipeline_library_Release(&library->iface);
      return hr;
   }

   hr = d12_pipeline_library_QueryInterface(&library->iface, riid, out);
   d12_pipeline_library_Release(&library->iface);
   return hr;
}

static HRESULT STDMETHODCALLTYPE
d12_command_signature_QueryInterface(ID3D12CommandSignature *iface,
                                     REFIID riid, void **out)
{
   struct d12_command_signature *signature =
      d12_command_signature_from_iface(iface);

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D12Object) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceChild) ||
       d12_guid_equal(riid, &IID_ID3D12Pageable) ||
       d12_guid_equal(riid, &IID_ID3D12CommandSignature)) {
      p_atomic_inc(&signature->refcount);
      *out = iface;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&signature->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12CommandSignature::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_command_signature_AddRef(ID3D12CommandSignature *iface)
{
   return p_atomic_inc_return(
      &d12_command_signature_from_iface(iface)->refcount);
}

static ULONG STDMETHODCALLTYPE
d12_command_signature_Release(ID3D12CommandSignature *iface)
{
   struct d12_command_signature *signature =
      d12_command_signature_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&signature->refcount);

   if (!refs) {
      d12_private_data_store_destroy(&signature->private_data);
      d12_device_release(signature->device);
      free(signature->arguments);
      free(signature);
   }

   return refs;
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_command_signature,
                                ID3D12CommandSignature,
                                d12_command_signature_from_iface)

static HRESULT STDMETHODCALLTYPE
d12_command_signature_GetDevice(ID3D12CommandSignature *iface, REFIID riid,
                                void **out)
{
   struct d12_command_signature *signature =
      d12_command_signature_from_iface(iface);
   return ID3D12Device_QueryInterface(&signature->device->iface, riid, out);
}

static const ID3D12CommandSignatureVtbl d12_command_signature_vtbl = {
   .QueryInterface = d12_command_signature_QueryInterface,
   .AddRef = d12_command_signature_AddRef,
   .Release = d12_command_signature_Release,
   .GetPrivateData = d12_command_signature_GetPrivateData,
   .SetPrivateData = d12_command_signature_SetPrivateData,
   .SetPrivateDataInterface = d12_command_signature_SetPrivateDataInterface,
   .SetName = d12_command_signature_SetName,
   .GetDevice = d12_command_signature_GetDevice,
};

/*
 * Command signatures retain their validated public description for the
 * native ExecuteIndirect recorder. This is a real pageable D3D12 object, not
 * a success-only token: COM identity, device ownership, private data, and
 * argument lifetime all follow the ordinary object contract.
 *
 * DRAW, DRAW_INDEXED and DISPATCH are terminal.  A signature may place root
 * arguments before the one terminal command, but the terminal command must be
 * last and unique.
 */
static HRESULT STDMETHODCALLTYPE
d12_device_CreateCommandSignature(ID3D12Device2 *iface,
                                  const D3D12_COMMAND_SIGNATURE_DESC *desc,
                                  ID3D12RootSignature *root_signature,
                                  REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);

   if (!out)
      return E_POINTER;

   *out = NULL;

   if (!desc || !desc->NumArgumentDescs || desc->NumArgumentDescs > 65 ||
       !desc->pArgumentDescs || desc->NodeMask > 1)
      return E_INVALIDARG;

   struct d12_root_signature *rs =
      (struct d12_root_signature *)root_signature;
   if (rs && rs->device != device)
      return E_INVALIDARG;

   bool saw_terminal = false;
   D3D12_INDIRECT_ARGUMENT_TYPE terminal_type =
      (D3D12_INDIRECT_ARGUMENT_TYPE)~0u;
   for (UINT i = 0; i < desc->NumArgumentDescs; i++) {
      D3D12_INDIRECT_ARGUMENT_TYPE type = desc->pArgumentDescs[i].Type;
      bool terminal = type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW ||
                      type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED ||
                      type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
      if (terminal) {
         if (saw_terminal || i + 1 != desc->NumArgumentDescs)
            return E_INVALIDARG;
         saw_terminal = true;
         terminal_type = type;
      }
   }

   if (!saw_terminal)
      return E_INVALIDARG;

   size_t packed_size = 0;
   bool changes_root_arguments = false;
   bool has_root_reference = false;
   bool last_root_was_constant = false;
   bool has_index_buffer = false;
   UINT last_root_parameter = 0;
   UINT last_constant_end = 0;
   UINT vertex_buffer_mask = 0;
   for (UINT i = 0; i < desc->NumArgumentDescs; i++) {
      const D3D12_INDIRECT_ARGUMENT_DESC *argument =
         &desc->pArgumentDescs[i];
      D3D12_INDIRECT_ARGUMENT_TYPE type = argument->Type;

      bool terminal = type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW ||
                      type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED ||
                      type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;

      if (terminal) {
         packed_size += type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW
                           ? sizeof(D3D12_DRAW_ARGUMENTS)
                        : type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED
                           ? sizeof(D3D12_DRAW_INDEXED_ARGUMENTS)
                           : sizeof(D3D12_DISPATCH_ARGUMENTS);
      } else if (type == D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT) {
         if (!rs || !argument->Constant.Num32BitValuesToSet ||
             argument->Constant.RootParameterIndex >= rs->layout.num_params)
            return E_INVALIDARG;
         const struct d12_root_param *parameter =
            &rs->layout.params[argument->Constant.RootParameterIndex];
         if (parameter->type != D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS ||
             argument->Constant.DestOffsetIn32BitValues >
                parameter->num_constants ||
             argument->Constant.Num32BitValuesToSet >
                parameter->num_constants -
                   argument->Constant.DestOffsetIn32BitValues)
            return E_INVALIDARG;
         if (argument->Constant.Num32BitValuesToSet >
             (SIZE_MAX - packed_size) / 4)
            return E_INVALIDARG;

         UINT root_parameter = argument->Constant.RootParameterIndex;
         UINT constant_start =
            argument->Constant.DestOffsetIn32BitValues;
         if ((has_root_reference && root_parameter < last_root_parameter) ||
             (has_root_reference && root_parameter == last_root_parameter &&
              (!last_root_was_constant ||
               constant_start < last_constant_end)))
            return E_INVALIDARG;

         has_root_reference = true;
         last_root_was_constant = true;
         last_root_parameter = root_parameter;
         last_constant_end =
            constant_start + argument->Constant.Num32BitValuesToSet;
         packed_size +=
            (size_t)argument->Constant.Num32BitValuesToSet * 4;
         changes_root_arguments = true;
      } else if (type ==
                    D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW ||
                 type == D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW ||
                 type ==
                    D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW) {
         UINT root_parameter_index;
         D3D12_ROOT_PARAMETER_TYPE expected;
         if (type == D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW) {
            root_parameter_index =
               argument->ConstantBufferView.RootParameterIndex;
            expected = D3D12_ROOT_PARAMETER_TYPE_CBV;
         } else if (type == D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW) {
            root_parameter_index =
               argument->ShaderResourceView.RootParameterIndex;
            expected = D3D12_ROOT_PARAMETER_TYPE_SRV;
         } else {
            root_parameter_index =
               argument->UnorderedAccessView.RootParameterIndex;
            expected = D3D12_ROOT_PARAMETER_TYPE_UAV;
         }
         if (!rs || root_parameter_index >= rs->layout.num_params ||
             rs->layout.params[root_parameter_index].type != expected ||
             packed_size > SIZE_MAX - sizeof(uint64_t))
            return E_INVALIDARG;
         if (has_root_reference &&
             root_parameter_index <= last_root_parameter)
            return E_INVALIDARG;
         has_root_reference = true;
         last_root_was_constant = false;
         last_root_parameter = root_parameter_index;
         packed_size += sizeof(uint64_t);
         changes_root_arguments = true;
      } else if (type == D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW ||
                 type == D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW) {
         if (type == D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW) {
            UINT slot = argument->VertexBuffer.Slot;
            if (slot >= D12_MAX_VB_SLOTS ||
                (vertex_buffer_mask & BITFIELD_BIT(slot)) ||
                packed_size >
                   SIZE_MAX - sizeof(D3D12_VERTEX_BUFFER_VIEW))
               return E_INVALIDARG;
            vertex_buffer_mask |= BITFIELD_BIT(slot);
            packed_size += sizeof(D3D12_VERTEX_BUFFER_VIEW);
         } else {
            if (has_index_buffer ||
                packed_size > SIZE_MAX - sizeof(D3D12_INDEX_BUFFER_VIEW))
               return E_INVALIDARG;
            has_index_buffer = true;
            packed_size += sizeof(D3D12_INDEX_BUFFER_VIEW);
         }
      } else {
         return E_INVALIDARG;
      }
   }

   if ((terminal_type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH &&
        (vertex_buffer_mask || has_index_buffer)) ||
       (has_index_buffer &&
        terminal_type != D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED))
      return E_INVALIDARG;

   /* Microsoft requires a root signature exactly when the command signature
    * mutates root arguments. A terminal-only or IA-only signature supplies
    * NULL; a root constant/view signature supplies its validating layout. */
   if ((rs != NULL) != changes_root_arguments)
      return E_INVALIDARG;

   if (desc->ByteStride < packed_size || (desc->ByteStride & 3))
      return E_INVALIDARG;

   struct d12_command_signature *signature = calloc(1, sizeof(*signature));
   if (!signature)
      return E_OUTOFMEMORY;

   signature->arguments =
      malloc(sizeof(*signature->arguments) * desc->NumArgumentDescs);
   if (!signature->arguments) {
      free(signature);
      return E_OUTOFMEMORY;
   }

   memcpy(signature->arguments, desc->pArgumentDescs,
          sizeof(*signature->arguments) * desc->NumArgumentDescs);
   signature->iface.lpVtbl =
      (ID3D12CommandSignatureVtbl *)&d12_command_signature_vtbl;
   signature->refcount = 1;
   signature->device = device;
   signature->desc = *desc;
   signature->desc.pArgumentDescs = signature->arguments;
   d12_private_data_store_init(&signature->private_data,
                               (IUnknown *)&signature->iface);
   d12_device_add_ref(device);

   HRESULT hr = d12_command_signature_QueryInterface(&signature->iface, riid,
                                                      out);
   d12_command_signature_Release(&signature->iface);
   return hr;
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreateDescriptorHeap(ID3D12Device2 *iface,
                                const D3D12_DESCRIPTOR_HEAP_DESC *desc,
                                REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);
   return d12_descriptor_heap_create(device, desc, riid, out);
}

static void STDMETHODCALLTYPE
d12_device_CreateRenderTargetView(ID3D12Device2 *iface,
                                  ID3D12Resource *resource,
                                  const D3D12_RENDER_TARGET_VIEW_DESC *desc,
                                  D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   struct d12_rtv *slot = (struct d12_rtv *)dest.ptr;

   if (!slot)
      return;

   memset(slot, 0, sizeof(*slot));

   if (!resource) {
      /* Null RTVs are legal placeholders; record an empty slot. */
      return;
   }

   struct d12_resource *res = d12_resource_from_iface(resource);

   /*
    * The VIEW's format wins over the resource's. That is not a preference:
    * a TYPELESS resource carries no interpretation at all, and D3D12
    * requires the view to supply one, so checking the resource format
    * rejects every typeless render target no matter what the application
    * asked for. DXGI_FORMAT_UNKNOWN in the desc means "inherit", and a
    * NULL desc means the same.
    */
   DXGI_FORMAT view_format = (desc && desc->Format != DXGI_FORMAT_UNKNOWN)
                                ? desc->Format
                                : res->desc.Format;

   /* Report WHICH half failed. Reporting the format for a dimension
    * mismatch sent a debugging session after a format table that was
    * already correct.
    */
   const bool is_1d =
      res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D;
   const bool is_2d =
      res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D;
   const bool is_3d =
      res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;

   if (!is_1d && !is_2d && !is_3d) {
      mesa_loge("CreateRenderTargetView: dimension %d unsupported (need 1D, "
                "2D or 3D), format %d", (int)res->desc.Dimension,
                (int)view_format);
      slot->resource = NULL;
      return;
   }

   if (!d12_format_is_renderable(view_format)) {
      mesa_loge("CreateRenderTargetView: format %d is not renderable",
                (int)view_format);
      slot->resource = NULL;
      return;
   }

   const unsigned samples = res->desc.SampleDesc.Count;
   if (((is_1d || is_3d) && samples != 1) ||
       (is_2d && samples != 1 && samples != 2 && samples != 4)) {
      mesa_loge("CreateRenderTargetView: %u samples unsupported", samples);
      slot->resource = NULL;
      return;
   }

   /*
    * ARRAY SLICES. A TEXTURE2DARRAY view selects a consecutive layer range;
    * ail lays layers out contiguously at layer_stride_B, so the first slice
    * is just an offset and the view retains the range count. Refusing the
    * view dimension made every array test fail at the view even once the
    * resource itself was legal.
    */
   uint32_t first_layer = 0;
   uint32_t view_level = 0;
   uint32_t view_layers = MAX2(res->desc.DepthOrArraySize, 1);

   if (desc) {
      if (is_3d) {
         if (desc->ViewDimension != D3D12_RTV_DIMENSION_TEXTURE3D) {
            mesa_loge("CreateRenderTargetView: 3D view dimension %d "
                      "unsupported", (int)desc->ViewDimension);
            slot->resource = NULL;
            return;
         }

         view_level = desc->Texture3D.MipSlice;
         first_layer = desc->Texture3D.FirstWSlice;
         view_layers = desc->Texture3D.WSize;
      } else if (is_1d) {
         if (desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE1D) {
            view_level = desc->Texture1D.MipSlice;
            view_layers = 1;
         } else if (desc->ViewDimension ==
                    D3D12_RTV_DIMENSION_TEXTURE1DARRAY) {
            view_level = desc->Texture1DArray.MipSlice;
            first_layer = desc->Texture1DArray.FirstArraySlice;
            view_layers = desc->Texture1DArray.ArraySize;
         } else {
            mesa_loge("CreateRenderTargetView: 1D view dimension %d "
                      "unsupported", (int)desc->ViewDimension);
            slot->resource = NULL;
            return;
         }
      } else if (samples > 1) {
         /* MSAA resources are necessarily single-level, but may expose a
          * bounded array-slice range just like ordinary 2D arrays. */
         if (desc->ViewDimension ==
             D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY) {
            first_layer = desc->Texture2DMSArray.FirstArraySlice;
            view_layers = desc->Texture2DMSArray.ArraySize;
         } else if (desc->ViewDimension !=
                    D3D12_RTV_DIMENSION_TEXTURE2DMS) {
            mesa_loge("CreateRenderTargetView: multisample view dimension %d "
                      "unsupported", (int)desc->ViewDimension);
            slot->resource = NULL;
            return;
         } else {
            view_layers = 1;
         }
      } else if (desc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2DARRAY) {
         if (desc->Texture2DArray.PlaneSlice != 0) {
            mesa_loge("CreateRenderTargetView: plane slice %u unsupported",
                      desc->Texture2DArray.PlaneSlice);
            slot->resource = NULL;
            return;
         }
         view_level = desc->Texture2DArray.MipSlice;
         first_layer = desc->Texture2DArray.FirstArraySlice;
         view_layers = desc->Texture2DArray.ArraySize;
      } else if (desc->ViewDimension != D3D12_RTV_DIMENSION_TEXTURE2D) {
         mesa_loge("CreateRenderTargetView: view dimension %d unsupported",
                   (int)desc->ViewDimension);
         slot->resource = NULL;
         return;
      } else {
         if (desc->Texture2D.PlaneSlice != 0) {
            mesa_loge("CreateRenderTargetView: plane slice %u unsupported",
                      desc->Texture2D.PlaneSlice);
            slot->resource = NULL;
            return;
         }
         view_level = desc->Texture2D.MipSlice;
         view_layers = 1;
      }
   }

   struct d12_attachment_view view;
   const struct d12_attachment_view_request request = {
      .width = res->layout.width_px,
      .height = res->layout.height_px,
      .levels = res->layout.levels,
      .layers = res->layout.depth_px,
      .mip = view_level,
      .first_layer = first_layer,
      .layer_count = view_layers,
      .mipmapped_layers = is_3d,
   };
   if (!d12_attachment_view_resolve(&request, &view)) {
      mesa_loge("CreateRenderTargetView: layer range %u+%u at mip %u is out "
                "of range", first_layer, view_layers, view_level);
      slot->resource = NULL;
      return;
   }
   first_layer = view.first_layer;
   view_level = view.mip;
   view_layers = view.layer_count;

   slot->resource = res;
   /* The interpretation the shader and the clear path must both use --
    * storing the resource's typeless format here would push the same
    * "no interpretation" problem downstream to every consumer.
    */
   slot->format = view_format;
   slot->kind = D12_DESC_RTV;
   /* Keep the attachment at the first selected layer's miptree base. The AGX
    * texture/PBE descriptors carry view_level explicitly; pre-offsetting both
    * the address and the level would apply the mip offset twice. */
   slot->address = res->addr +
                   (uint64_t)first_layer * res->layout.layer_stride_B;
   /* RTV descriptors are CPU-only. Reuse their texture metadata words for
    * the selected layer count and mip without growing the descriptor ABI. */
   slot->texture_layers = view_layers;
   slot->texture_level = view_level;
}

static void STDMETHODCALLTYPE
d12_device_CreateDepthStencilView(ID3D12Device2 *iface,
                                  ID3D12Resource *resource,
                                  const D3D12_DEPTH_STENCIL_VIEW_DESC *desc,
                                  D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   struct d12_rtv *slot = (struct d12_rtv *)dest.ptr;

   if (!slot)
      return;

   memset(slot, 0, sizeof(*slot));

   if (!resource) {
      /* Null DSVs are legal placeholders; record an empty slot. */
      slot->resource = NULL;
      return;
   }

   struct d12_resource *res = d12_resource_from_iface(resource);

   /* Fixed-function ZLS uses one address for both loads and stores. Sparse
    * holes use distinct zero-read and discard-write aliases, so a reserved
    * depth resource cannot satisfy D3D12 null-tile semantics through ZLS. */
   if (res->reserved) {
      mesa_loge("CreateDepthStencilView: reserved depth resources are "
                "unsupported");
      slot->resource = NULL;
      return;
   }

   const bool is_1d =
      res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D;
   const bool is_2d =
      res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D;
   const unsigned samples = MAX2(res->desc.SampleDesc.Count, 1);
   if ((!is_1d && !is_2d) || (is_1d && samples != 1) ||
       (is_2d && samples != 1 && samples != 2 && samples != 4)) {
      mesa_loge("CreateDepthStencilView: dimension %d/sample count %u is "
                "unsupported", (int)res->desc.Dimension, samples);
      slot->resource = NULL;
      return;
   }

   if (!res->is_depth_stencil) {
      mesa_loge("CreateDepthStencilView: resource format %d is not a "
                "depth/stencil format", (int)res->desc.Format);
      slot->resource = NULL;
      return;
   }

   uint32_t first_layer = 0;
   uint32_t mip_level = 0;
   /* A NULL descriptor requests the default view over every array slice. */
   uint32_t view_layers =
      desc ? 1 : MAX2(res->desc.DepthOrArraySize, 1);
   if (desc) {
      if (is_1d) {
         if (desc->ViewDimension == D3D12_DSV_DIMENSION_TEXTURE1D) {
            mip_level = desc->Texture1D.MipSlice;
         } else if (desc->ViewDimension ==
                    D3D12_DSV_DIMENSION_TEXTURE1DARRAY) {
            first_layer = desc->Texture1DArray.FirstArraySlice;
            view_layers = desc->Texture1DArray.ArraySize;
            mip_level = desc->Texture1DArray.MipSlice;
         } else {
            mesa_loge("CreateDepthStencilView: view dimension %d does not "
                      "match 1D resource", (int)desc->ViewDimension);
            slot->resource = NULL;
            return;
         }
      } else if (samples > 1) {
         if (desc->ViewDimension ==
             D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY) {
            first_layer = desc->Texture2DMSArray.FirstArraySlice;
            view_layers = desc->Texture2DMSArray.ArraySize;
         } else if (desc->ViewDimension !=
                    D3D12_DSV_DIMENSION_TEXTURE2DMS) {
            mesa_loge("CreateDepthStencilView: view dimension %d does not "
                      "match %u-sample resource",
                      (int)desc->ViewDimension, samples);
            slot->resource = NULL;
            return;
         }
      } else if (desc->ViewDimension ==
                 D3D12_DSV_DIMENSION_TEXTURE2DARRAY) {
         first_layer = desc->Texture2DArray.FirstArraySlice;
         view_layers = desc->Texture2DArray.ArraySize;
         mip_level = desc->Texture2DArray.MipSlice;
      } else if (desc->ViewDimension != D3D12_DSV_DIMENSION_TEXTURE2D) {
         mesa_loge("CreateDepthStencilView: view dimension %d does not match "
                   "single-sample resource", (int)desc->ViewDimension);
         slot->resource = NULL;
         return;
      } else {
         mip_level = desc->Texture2D.MipSlice;
      }
   }

   struct d12_attachment_view view;
   const struct d12_attachment_view_request request = {
      .width = res->layout.width_px,
      .height = res->layout.height_px,
      .levels = res->layout.levels,
      .layers = res->layout.depth_px,
      .mip = mip_level,
      .first_layer = first_layer,
      .layer_count = view_layers,
   };
   if (!d12_attachment_view_resolve(&request, &view)) {
      mesa_loge("CreateDepthStencilView: layer range %u+%u at mip %u is out "
                "of range", first_layer, view_layers, mip_level);
      slot->resource = NULL;
      return;
   }
   first_layer = view.first_layer;
   mip_level = view.mip;
   view_layers = view.layer_count;

   const D3D12_DSV_FLAGS flags = desc ? desc->Flags : D3D12_DSV_FLAG_NONE;
   const D3D12_DSV_FLAGS valid = D3D12_DSV_FLAG_READ_ONLY_DEPTH |
                                 D3D12_DSV_FLAG_READ_ONLY_STENCIL;
   if (flags & ~valid) {
      mesa_loge("CreateDepthStencilView: invalid flags 0x%x",
                (unsigned)flags);
      slot->resource = NULL;
      return;
   }
   if ((flags & D3D12_DSV_FLAG_READ_ONLY_STENCIL) && !res->stencil_bo) {
      mesa_loge("CreateDepthStencilView: READ_ONLY_STENCIL on a format with "
                "no stencil plane");
      slot->resource = NULL;
      return;
   }

   slot->resource = res;
   slot->format = desc && desc->Format != DXGI_FORMAT_UNKNOWN
                     ? desc->Format
                     : res->desc.Format;
   slot->kind = D12_DESC_DSV;
   slot->dsv_flags = flags;
   slot->address = res->addr +
                   ail_get_layer_level_B(&res->layout, first_layer,
                                         mip_level);
   slot->counter_address = res->stencil_bo
      ? res->stencil_addr + ail_get_layer_level_B(
           &res->stencil_layout, first_layer, mip_level)
      : 0;
   /* CPU-only DSV descriptors reuse size_B for their selected layer count,
    * matching the established RTV representation. */
   slot->size_B = view_layers;
   slot->texture_plane = mip_level;
}

static enum agx_wrap
d12_sampler_wrap(D3D12_TEXTURE_ADDRESS_MODE mode)
{
   switch (mode) {
   case D3D12_TEXTURE_ADDRESS_MODE_WRAP:        return AGX_WRAP_REPEAT;
   case D3D12_TEXTURE_ADDRESS_MODE_MIRROR:      return AGX_WRAP_MIRRORED_REPEAT;
   case D3D12_TEXTURE_ADDRESS_MODE_CLAMP:       return AGX_WRAP_CLAMP_TO_EDGE;
   case D3D12_TEXTURE_ADDRESS_MODE_BORDER:      return AGX_WRAP_CLAMP_TO_BORDER;
   case D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE: return AGX_WRAP_MIRRORED_CLAMP_TO_EDGE;
   default:                                     return AGX_WRAP_CLAMP_TO_EDGE;
   }
}

static enum agx_compare_func
d12_sampler_compare(D3D12_COMPARISON_FUNC func)
{
   switch (func) {
   case D3D12_COMPARISON_FUNC_NEVER:         return AGX_COMPARE_FUNC_NEVER;
   case D3D12_COMPARISON_FUNC_LESS:          return AGX_COMPARE_FUNC_LESS;
   case D3D12_COMPARISON_FUNC_EQUAL:         return AGX_COMPARE_FUNC_EQUAL;
   case D3D12_COMPARISON_FUNC_LESS_EQUAL:    return AGX_COMPARE_FUNC_LEQUAL;
   case D3D12_COMPARISON_FUNC_GREATER:       return AGX_COMPARE_FUNC_GREATER;
   case D3D12_COMPARISON_FUNC_NOT_EQUAL:     return AGX_COMPARE_FUNC_NOT_EQUAL;
   case D3D12_COMPARISON_FUNC_GREATER_EQUAL: return AGX_COMPARE_FUNC_GEQUAL;
   case D3D12_COMPARISON_FUNC_ALWAYS:        return AGX_COMPARE_FUNC_ALWAYS;
   default:                                  return AGX_COMPARE_FUNC_NEVER;
   }
}

static enum agx_border_colour
d12_sampler_border(const D3D12_SAMPLER_DESC2 *desc)
{
   if (desc->Flags & D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR) {
      const UINT *color = desc->UintBorderColor;
      if (color[0] == 0 && color[1] == 0 && color[2] == 0)
         return color[3] == 1 ? AGX_BORDER_COLOUR_OPAQUE_BLACK
                              : AGX_BORDER_COLOUR_TRANSPARENT_BLACK;
      if (color[0] == 1 && color[1] == 1 && color[2] == 1 && color[3] == 1)
         return AGX_BORDER_COLOUR_OPAQUE_WHITE;
      return AGX_BORDER_COLOUR_TRANSPARENT_BLACK;
   }

   const FLOAT *color = desc->FloatBorderColor;
   if (color[0] == 0.0f && color[1] == 0.0f && color[2] == 0.0f) {
      return color[3] == 1.0f ? AGX_BORDER_COLOUR_OPAQUE_BLACK
                              : AGX_BORDER_COLOUR_TRANSPARENT_BLACK;
   }

   if (color[0] == 1.0f && color[1] == 1.0f && color[2] == 1.0f &&
       color[3] == 1.0f)
      return AGX_BORDER_COLOUR_OPAQUE_WHITE;

   /* The exact value remains in sampler_border for a future custom-border
    * table.  Until that table is bound, use a deterministic safe fallback. */
   return AGX_BORDER_COLOUR_TRANSPARENT_BLACK;
}

static uint32_t
d12_sampler_custom_border_flags(const D3D12_SAMPLER_DESC2 *desc)
{
   /* The shader fallback compares normalized coordinates against [0, 1).
    * Leave D3D12's non-normalized-coordinate extension on the native path. */
   if (desc->Flags & D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES)
      return 0;

   uint32_t flags = 0;
   if (desc->AddressU == D3D12_TEXTURE_ADDRESS_MODE_BORDER)
      flags |= D12_CUSTOM_BORDER_U;
   if (desc->AddressV == D3D12_TEXTURE_ADDRESS_MODE_BORDER)
      flags |= D12_CUSTOM_BORDER_V;
   if (desc->AddressW == D3D12_TEXTURE_ADDRESS_MODE_BORDER)
      flags |= D12_CUSTOM_BORDER_W;

   if (!flags)
      return 0;

   bool standard;
   if (desc->Flags & D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR) {
      const UINT *c = desc->UintBorderColor;
      standard = (c[0] == 0 && c[1] == 0 && c[2] == 0 &&
                  (c[3] == 0 || c[3] == 1)) ||
                 (c[0] == 1 && c[1] == 1 && c[2] == 1 && c[3] == 1);
   } else {
      const FLOAT *c = desc->FloatBorderColor;
      standard = (c[0] == 0.0f && c[1] == 0.0f && c[2] == 0.0f &&
                  (c[3] == 0.0f || c[3] == 1.0f)) ||
                 (c[0] == 1.0f && c[1] == 1.0f && c[2] == 1.0f &&
                  c[3] == 1.0f);
   }

   if (standard)
      return 0;

   flags |= D12_CUSTOM_BORDER_ENABLE;
   if (D3D12_DECODE_MIN_FILTER(desc->Filter) == D3D12_FILTER_TYPE_POINT &&
       D3D12_DECODE_MAG_FILTER(desc->Filter) == D3D12_FILTER_TYPE_POINT)
      flags |= D12_CUSTOM_BORDER_POINT;

   return flags;
}

void
d12_pack_sampler_descriptor(struct d12_device *device,
                            struct d12_descriptor *slot,
                            const D3D12_SAMPLER_DESC2 *desc)
{
   if (!device || !slot || !desc)
      return;

   memset(slot, 0, sizeof(*slot));

   unsigned anisotropy = 1;
   if (D3D12_DECODE_IS_ANISOTROPIC_FILTER(desc->Filter)) {
      unsigned requested = CLAMP(desc->MaxAnisotropy, 1, 16);
      while (anisotropy < requested)
         anisotropy <<= 1;
   }

   const bool comparison = D3D12_DECODE_IS_COMPARISON_FILTER(desc->Filter);
   /*
    * The currently runnable AGX2 sampler has no MINIMUM/MAXIMUM field, so it
    * is reduced in the shader; see d12_sampler_reduction_core.h for which
    * filters actually need that and d12_nir_lower_sampler_reduction for the
    * lowering. SamplerGen5 performs the all-linear subset natively; mixed
    * filters stay on the same shader fallback. Forcing nearest mip filtering
    * here is what lets that fallback's single 2x2 gather land on the level
    * the hardware selected.
    */
   const enum agx_sampler_generation sampler_generation =
      agx_chip_sampler_generation(device->dev.chip);
   const struct d12_sampler_reduction_policy reduction_policy =
      sampler_generation == AGX_SAMPLER_GENERATION_GEN5
         ? d12_sampler_reduction_for_filter_gen5((uint32_t)desc->Filter)
         : d12_sampler_reduction_for_filter((uint32_t)desc->Filter);
   const bool reduces =
      reduction_policy.reduction != D12_SAMPLER_REDUCE_NONE;

   const enum agx_filter magnify =
      D3D12_DECODE_MAG_FILTER(desc->Filter) == D3D12_FILTER_TYPE_LINEAR
         ? AGX_FILTER_LINEAR
         : AGX_FILTER_NEAREST;
   const enum agx_filter minify =
      D3D12_DECODE_MIN_FILTER(desc->Filter) == D3D12_FILTER_TYPE_LINEAR
         ? AGX_FILTER_LINEAR
         : AGX_FILTER_NEAREST;
   const enum agx_mip_filter mip_filter =
      !reduction_policy.force_mip_nearest &&
            D3D12_DECODE_MIP_FILTER(desc->Filter) == D3D12_FILTER_TYPE_LINEAR
         ? AGX_MIP_FILTER_LINEAR
         : AGX_MIP_FILTER_NEAREST;
   const enum agx_wrap wrap_s = d12_sampler_wrap(desc->AddressU);
   const enum agx_wrap wrap_t = d12_sampler_wrap(desc->AddressV);
   const enum agx_wrap wrap_r = d12_sampler_wrap(desc->AddressW);
   const enum agx_border_colour border = d12_sampler_border(desc);
   const enum agx_compare_func compare_func =
      d12_sampler_compare(desc->ComparisonFunc);
   const bool pixel_coordinates =
      desc->Flags & D3D12_SAMPLER_FLAG_NON_NORMALIZED_COORDINATES;

   switch (sampler_generation) {
   case AGX_SAMPLER_GENERATION_GEN4_HAL200:
      agx_pack(&slot->sampler, SAMPLER_HAL200, cfg) {
         cfg.minimum_lod = desc->MinLOD;
         cfg.maximum_lod = desc->MaxLOD;
         cfg.maximum_anisotropy = anisotropy;
         cfg.magnify = magnify;
         cfg.minify = minify;
         cfg.mip_filter = mip_filter;
         cfg.wrap_s = wrap_s;
         cfg.wrap_t = wrap_t;
         cfg.wrap_r = wrap_r;
         cfg.pixel_coordinates = pixel_coordinates;
         cfg.compare_func = compare_func;
         cfg.compare_enable = comparison;
         cfg.lod_bias = desc->MipLODBias;
         cfg.border_colour = border;
      }
      break;

   case AGX_SAMPLER_GENERATION_GEN5:
      agx_pack(&slot->sampler, SAMPLER_GEN5, cfg) {
         cfg.minimum_lod = desc->MinLOD;
         cfg.maximum_lod = desc->MaxLOD;
         cfg.maximum_anisotropy = anisotropy;
         cfg.magnify = magnify;
         cfg.minify = minify;
         cfg.mip_filter = mip_filter;
         cfg.wrap_s = wrap_s;
         cfg.wrap_t = wrap_t;
         cfg.wrap_r = wrap_r;
         cfg.pixel_coordinates = pixel_coordinates;
         cfg.lod_bias = desc->MipLODBias;
         cfg.border_colour = border;

         if (reduction_policy.native_gen5_mode !=
             D12_SAMPLER_GEN5_REDUCTION_WEIGHTED_AVERAGE) {
            cfg.reduction_mode =
               (enum agx_sampler_reduction_mode)
                  reduction_policy.native_gen5_mode;
         } else {
            /* On Gen5 bit 42 selects compare-function interpretation of
             * bits 39:41. It is set even for a non-comparison sample. */
            cfg.compare_mode = true;
            cfg.compare_func = compare_func;
         }
      }
      break;

   case AGX_SAMPLER_GENERATION_GEN4:
      agx_pack(&slot->sampler, SAMPLER, cfg) {
         cfg.minimum_lod = desc->MinLOD;
         cfg.maximum_lod = desc->MaxLOD;
         cfg.maximum_anisotropy = anisotropy;
         cfg.magnify = magnify;
         cfg.minify = minify;
         cfg.mip_filter = mip_filter;
         cfg.wrap_s = wrap_s;
         cfg.wrap_t = wrap_t;
         cfg.wrap_r = wrap_r;
         cfg.pixel_coordinates = pixel_coordinates;
         cfg.compare_func = compare_func;
         cfg.compare_enable = comparison;
         cfg.border_colour = border;
      }
      break;

   case AGX_SAMPLER_GENERATION_UNKNOWN:
   default:
      /* Device open rejects unknown sampler ABIs. Leave a null state for
       * defensive callers that reached this helper without a live device. */
      break;
   }

   memcpy(slot->sampler_border, desc->FloatBorderColor,
          sizeof(slot->sampler_border));
   slot->sampler_lod_bias =
      sampler_generation == AGX_SAMPLER_GENERATION_GEN4 ? desc->MipLODBias
                                                       : 0.0f;
   slot->sampler_flags = desc->Flags;
   slot->sampler_reserved = d12_sampler_custom_border_flags(desc);
   if (D3D12_DECODE_MIP_FILTER(desc->Filter) == D3D12_FILTER_TYPE_POINT ||
       reduction_policy.force_mip_nearest)
      slot->sampler_reserved |= D12_SAMPLER_MIP_POINT;
   if (reduces) {
      slot->sampler_reserved |=
         reduction_policy.reduction == D12_SAMPLER_REDUCE_MINIMUM
            ? D12_SAMPLER_REDUCTION_MIN
            : D12_SAMPLER_REDUCTION_MAX;
   }
   slot->sampler_kind = D12_DESC_SAMPLER;
}

static void STDMETHODCALLTYPE
d12_device_CreateSampler(ID3D12Device2 *iface,
                         const D3D12_SAMPLER_DESC *desc,
                         D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   struct d12_device *device = d12_device_from_iface(iface);
   struct d12_descriptor *slot = (struct d12_descriptor *)dest.ptr;
   if (!slot || !desc)
      return;

   D3D12_SAMPLER_DESC2 desc2 = {
      .Filter = desc->Filter,
      .AddressU = desc->AddressU,
      .AddressV = desc->AddressV,
      .AddressW = desc->AddressW,
      .MipLODBias = desc->MipLODBias,
      .MaxAnisotropy = desc->MaxAnisotropy,
      .ComparisonFunc = desc->ComparisonFunc,
      .MinLOD = desc->MinLOD,
      .MaxLOD = desc->MaxLOD,
      .Flags = D3D12_SAMPLER_FLAG_NONE,
   };
   memcpy(desc2.FloatBorderColor, desc->BorderColor,
          sizeof(desc2.FloatBorderColor));
   d12_pack_sampler_descriptor(device, slot, &desc2);
   d12_descriptor_heap_update_slot(device, slot);
}

/*
 * Buffer views. A descriptor here is just the address the shader will
 * reach the resource through, resolved once at view creation.
 *
 * Texture SRVs are recorded separately below; this helper is deliberately
 * buffer-only so a tiled resource can never be interpreted as linear data.
 */
static bool
d12_fill_buffer_descriptor(struct d12_descriptor *slot,
                           struct d12_resource *res, uint64_t first_element,
                           uint32_t num_elements, uint32_t stride_B,
                           const char *what)
{
   if (res->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
      mesa_loge("%s: only buffer views are supported so far", what);
      slot->kind = D12_DESC_NONE;
      slot->resource = NULL;
      return false;
   }

   if (!d12_buffer_view_element_count_valid(num_elements)) {
      mesa_loge("%s: NumElements exceeds the D3D12 buffer-view limit", what);
      slot->kind = D12_DESC_NONE;
      slot->resource = NULL;
      return false;
   }

   stride_B = MAX2(stride_B, 1);
   if (stride_B > UINT16_MAX) {
      mesa_loge("%s: buffer stride exceeds root metadata ABI", what);
      slot->kind = D12_DESC_NONE;
      slot->resource = NULL;
      return false;
   }

   uint64_t offset;
   if (!d12_buffer_view_range(res->desc.Width, first_element, num_elements,
                              stride_B, &offset, NULL)) {
      mesa_loge("%s: view spans past the end of the resource", what);
      slot->kind = D12_DESC_NONE;
      slot->resource = NULL;
      return false;
   }

   slot->resource = res;
   slot->address = res->addr + offset;
   slot->buffer_count = num_elements;
   slot->buffer_stride_B = stride_B;
   return true;
}

/*
 * View creation builds the descriptor in a zeroed local and commits it to
 * the heap slot once, so the native mirror is refreshed exactly once per
 * call. Every early return below leaves the local zeroed, which commits a
 * D12_DESC_NONE descriptor for the rejected view.
 */
static void
d12_device_fill_srv(struct d12_device *device, ID3D12Resource *resource,
                    const D3D12_SHADER_RESOURCE_VIEW_DESC *desc,
                    struct d12_descriptor *slot)
{
   if (!resource) {
      /*
       * A null descriptor must READ AS ZERO, not fault. Leaving address 0
       * here (which zeroing the slot did) makes the shader dereference
       * NULL: the GPU reports `address: 0x0, reason: Unmapped, read: true`
       * and times out the queue -- and because the compositor shares this
       * GPU, that freezes the desktop too.
       *
       * Bisecting the suite localised the timeout burst to exactly the
       * test_null_* cluster: names.txt 1..278 gave +0, 279..418 gave +69.
       */
      slot->address = device->null_va;
      slot->size_B = D12_NULL_REGION_SIZE;
      return;
   }

   struct d12_resource *res = d12_resource_from_iface(resource);

   /* Texture views record a validated mip/layer subset; the native hardware
    * descriptor is packed at draw time, when the heap it lives in is built. */
   if ((desc && (desc->ViewDimension == D3D12_SRV_DIMENSION_TEXTURE1D ||
                 desc->ViewDimension == D3D12_SRV_DIMENSION_TEXTURE1DARRAY ||
                 desc->ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D ||
                 desc->ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY ||
                 desc->ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DMS ||
                 desc->ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY ||
                 desc->ViewDimension == D3D12_SRV_DIMENSION_TEXTURECUBE ||
                 desc->ViewDimension == D3D12_SRV_DIMENSION_TEXTURECUBEARRAY ||
                 desc->ViewDimension == D3D12_SRV_DIMENSION_TEXTURE3D)) ||
       (!desc && (res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D ||
                  res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
                  res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D))) {
      const bool is_3d =
         res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
      const bool is_1d =
         res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D;
      if (!is_1d && res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
          !is_3d) {
         mesa_loge("CreateShaderResourceView: texture view of a non-texture "
                   "resource");
         return;
      }

      const unsigned samples = MAX2(res->desc.SampleDesc.Count, 1);
      const bool msaa = samples > 1;
      const unsigned resource_layers =
         is_3d ? 1 : MAX2(res->desc.DepthOrArraySize, 1);
      D3D12_SRV_DIMENSION view_dim =
         desc    ? desc->ViewDimension
         : is_3d ? D3D12_SRV_DIMENSION_TEXTURE3D
         : is_1d ? (resource_layers > 1 ? D3D12_SRV_DIMENSION_TEXTURE1DARRAY
                                        : D3D12_SRV_DIMENSION_TEXTURE1D)
         : msaa  ? (resource_layers > 1 ? D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY
                                        : D3D12_SRV_DIMENSION_TEXTURE2DMS)
                 : (resource_layers > 1 ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY
                                        : D3D12_SRV_DIMENSION_TEXTURE2D);
      bool view_msaa = view_dim == D3D12_SRV_DIMENSION_TEXTURE2DMS ||
                       view_dim == D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
      bool view_1d = view_dim == D3D12_SRV_DIMENSION_TEXTURE1D ||
                     view_dim == D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
      if ((view_dim == D3D12_SRV_DIMENSION_TEXTURE3D) != is_3d ||
          view_1d != is_1d || (samples != 1 && samples != 2 && samples != 4) ||
          view_msaa != msaa) {
         mesa_loge("CreateShaderResourceView: implicit texture view shape "
                   "unsupported (mips %u, layers %u, samples %u)",
                   res->desc.MipLevels, res->desc.DepthOrArraySize,
                   res->desc.SampleDesc.Count);
         return;
      }

      unsigned first_level = 0;
      unsigned level_count = 1;
      if (!msaa) {
         switch (view_dim) {
         case D3D12_SRV_DIMENSION_TEXTURE1D:
            first_level = desc ? desc->Texture1D.MostDetailedMip : 0;
            level_count = desc ? desc->Texture1D.MipLevels : UINT32_MAX;
            break;
         case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
            first_level = desc ? desc->Texture1DArray.MostDetailedMip : 0;
            level_count = desc ? desc->Texture1DArray.MipLevels : UINT32_MAX;
            break;
         case D3D12_SRV_DIMENSION_TEXTURE2D:
            first_level = desc ? desc->Texture2D.MostDetailedMip : 0;
            level_count = desc ? desc->Texture2D.MipLevels : UINT32_MAX;
            break;
         case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
            first_level = desc ? desc->Texture2DArray.MostDetailedMip : 0;
            level_count = desc ? desc->Texture2DArray.MipLevels : UINT32_MAX;
            break;
         case D3D12_SRV_DIMENSION_TEXTURECUBE:
            first_level = desc->TextureCube.MostDetailedMip;
            level_count = desc->TextureCube.MipLevels;
            break;
         case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
            first_level = desc->TextureCubeArray.MostDetailedMip;
            level_count = desc->TextureCubeArray.MipLevels;
            break;
         case D3D12_SRV_DIMENSION_TEXTURE3D:
            first_level = desc ? desc->Texture3D.MostDetailedMip : 0;
            level_count = desc ? desc->Texture3D.MipLevels : UINT32_MAX;
            break;
         default:
            mesa_loge("CreateShaderResourceView: invalid non-MSAA view %u",
                      view_dim);
            return;
         }
         if (first_level >= res->desc.MipLevels) {
            mesa_loge("CreateShaderResourceView: first mip %u is outside "
                      "%u resource levels", first_level,
                      res->desc.MipLevels);
            return;
         }

         unsigned remaining = res->desc.MipLevels - first_level;
         if (level_count == UINT32_MAX)
            level_count = remaining;
         if (!level_count || level_count > remaining ||
             first_level > 0xf || level_count > 0x1f) {
            mesa_loge("CreateShaderResourceView: mip range %u+%u is outside "
                      "%u resource levels", first_level, level_count,
                      res->desc.MipLevels);
            return;
         }
      }

      unsigned first_layer = 0;
      unsigned view_layers = 1;
      switch (view_dim) {
      case D3D12_SRV_DIMENSION_TEXTURE1D:
      case D3D12_SRV_DIMENSION_TEXTURE2D:
      case D3D12_SRV_DIMENSION_TEXTURE2DMS:
         break;
      case D3D12_SRV_DIMENSION_TEXTURE3D:
         break;
      case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
         first_layer = desc ? desc->Texture1DArray.FirstArraySlice : 0;
         view_layers = desc ? desc->Texture1DArray.ArraySize : resource_layers;
         break;
      case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
         first_layer = desc ? desc->Texture2DArray.FirstArraySlice : 0;
         view_layers = desc ? desc->Texture2DArray.ArraySize : resource_layers;
         break;
      case D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY:
         first_layer = desc ? desc->Texture2DMSArray.FirstArraySlice : 0;
         view_layers = desc ? desc->Texture2DMSArray.ArraySize : resource_layers;
         break;
      case D3D12_SRV_DIMENSION_TEXTURECUBE:
         view_layers = 6;
         break;
      case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: {
         first_layer = desc->TextureCubeArray.First2DArrayFace;
         unsigned cubes = desc->TextureCubeArray.NumCubes;
         if (cubes == UINT32_MAX)
            cubes = (resource_layers - MIN2(first_layer, resource_layers)) / 6;
         if (cubes > UINT32_MAX / 6) {
            mesa_loge("CreateShaderResourceView: cube count overflow");
            return;
         }
         view_layers = cubes * 6;
         break;
      }
      default:
         mesa_loge("CreateShaderResourceView: invalid texture view %u",
                   view_dim);
         return;
      }

      if (view_layers == UINT32_MAX)
         view_layers = resource_layers - MIN2(first_layer, resource_layers);
      if (first_layer >= resource_layers || !view_layers ||
          view_layers > resource_layers - first_layer || view_layers > UINT8_MAX ||
          ((view_dim == D3D12_SRV_DIMENSION_TEXTURECUBE ||
            view_dim == D3D12_SRV_DIMENSION_TEXTURECUBEARRAY) &&
           ((first_layer % 6) || (view_layers % 6)))) {
         mesa_loge("CreateShaderResourceView: layer range %u+%u/%u invalid",
                   first_layer, view_layers, resource_layers);
         return;
      }

      unsigned plane = 0;
      if (desc && !msaa) {
         if (view_dim == D3D12_SRV_DIMENSION_TEXTURE2D)
            plane = desc->Texture2D.PlaneSlice;
         else if (view_dim == D3D12_SRV_DIMENSION_TEXTURE2DARRAY)
            plane = desc->Texture2DArray.PlaneSlice;
      }
      if (plane > 1 || (plane == 1 && !res->stencil_bo)) {
         mesa_loge("CreateShaderResourceView: texture plane %u unsupported",
                   plane);
         return;
      }

      float api_min_lod = 0.0f;
      if (desc && !msaa) {
         switch (view_dim) {
         case D3D12_SRV_DIMENSION_TEXTURE1D:
            api_min_lod = desc->Texture1D.ResourceMinLODClamp;
            break;
         case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
            api_min_lod = desc->Texture1DArray.ResourceMinLODClamp;
            break;
         case D3D12_SRV_DIMENSION_TEXTURE2D:
            api_min_lod = desc->Texture2D.ResourceMinLODClamp;
            break;
         case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
            api_min_lod = desc->Texture2DArray.ResourceMinLODClamp;
            break;
         case D3D12_SRV_DIMENSION_TEXTURECUBE:
            api_min_lod = desc->TextureCube.ResourceMinLODClamp;
            break;
         case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
            api_min_lod = desc->TextureCubeArray.ResourceMinLODClamp;
            break;
         case D3D12_SRV_DIMENSION_TEXTURE3D:
            api_min_lod = desc->Texture3D.ResourceMinLODClamp;
            break;
         default:
            break;
         }
      }
      api_min_lod = MAX2(api_min_lod, 0.0f);

      const struct ail_layout *view_layout =
         plane ? &res->stencil_layout : &res->layout;

      slot->kind = D12_DESC_SRV;
      slot->resource = res;
      slot->address = (plane ? res->stencil_addr : res->addr) +
                      ail_get_layer_offset_B(view_layout, first_layer);
      if (res->reserved)
         slot->address += device->dev.sparse_ro_offset;
      slot->format = desc && desc->Format != DXGI_FORMAT_UNKNOWN
                        ? desc->Format
                        : res->desc.Format;
      slot->texture_component_mapping =
         desc ? desc->Shader4ComponentMapping
              : D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      slot->texture_first_level = first_level;
      slot->texture_level_count = level_count;
      float quantized_min_lod =
         floorf(api_min_lod * 256.0f + 0.5f) / 256.0f;
      unsigned last_level = first_level + level_count - 1;
      slot->texture_min_lod = MAX2(api_min_lod - first_level, 0.0f);
      slot->texture_min_lod_oob = quantized_min_lod > last_level;
      slot->texture_txf_min_level = slot->texture_min_lod_oob
                                       ? 0x1f
                                       : MAX2((int)floorf(quantized_min_lod) -
                                                 (int)first_level,
                                              0);
      slot->texture_srv_layers = view_layers;
      slot->texture_srv_dimension = view_dim;
      slot->texture_plane = plane;
      return;
   }

   if (desc && desc->ViewDimension != D3D12_SRV_DIMENSION_BUFFER) {
      mesa_loge("CreateShaderResourceView: view dimension %d unsupported",
                (int)desc->ViewDimension);
      return;
   }

   uint64_t first = desc ? desc->Buffer.FirstElement : 0;
   uint64_t default_count = res->desc.Width / 4;
   if (!desc && default_count > UINT32_MAX) {
      mesa_loge("CreateShaderResourceView: default buffer view has too many "
                "elements");
      return;
   }
   uint32_t count = desc ? desc->Buffer.NumElements : (uint32_t)default_count;
   uint32_t stride = desc ? desc->Buffer.StructureByteStride : 4;

   /* A raw (ByteAddressBuffer) view addresses 4-byte words. */
   if (desc && (desc->Buffer.Flags & D3D12_BUFFER_SRV_FLAG_RAW)) {
      stride = 4;
   } else if (!stride && desc && desc->Format != DXGI_FORMAT_UNKNOWN) {
      /* A typed Buffer<T> has StructureByteStride == 0. Its element size
       * comes from the view format, not from the structured-buffer field.
       */
      stride = d12_format_bytes_per_pixel(desc->Format);
      if (!stride) {
         mesa_loge("CreateShaderResourceView: typed buffer format %u "
                   "has no supported element size", (unsigned)desc->Format);
         return;
      }
   }
   if (!stride)
      stride = 4;

   if (d12_fill_buffer_descriptor(slot, res, first, count, stride,
                                  "CreateShaderResourceView")) {
      if (res->reserved)
         slot->address += device->dev.sparse_ro_offset;
      slot->kind = D12_DESC_SRV;
      slot->format = desc ? desc->Format : DXGI_FORMAT_UNKNOWN;
      slot->buffer_kind =
         desc && (desc->Buffer.Flags & D3D12_BUFFER_SRV_FLAG_RAW)
            ? D12_BUFFER_VIEW_RAW
         : desc && desc->Buffer.StructureByteStride
            ? D12_BUFFER_VIEW_STRUCTURED
            : D12_BUFFER_VIEW_TYPED;
   }
}

static void STDMETHODCALLTYPE
d12_device_CreateShaderResourceView(ID3D12Device2 *iface,
                                    ID3D12Resource *resource,
                                    const D3D12_SHADER_RESOURCE_VIEW_DESC *desc,
                                    D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   struct d12_device *device = d12_device_from_iface(iface);
   struct d12_descriptor *slot = (struct d12_descriptor *)dest.ptr;

   if (!slot)
      return;

   struct d12_descriptor view = {0};
   d12_device_fill_srv(device, resource, desc, &view);
   *slot = view;
   d12_descriptor_heap_update_slot(device, slot);
}

static void
d12_device_fill_uav(struct d12_device *device, ID3D12Resource *resource,
                    ID3D12Resource *counter,
                    const D3D12_UNORDERED_ACCESS_VIEW_DESC *desc,
                    struct d12_descriptor *slot)
{
   if (!resource) {
      /* Same reasoning as the null SRV above.  Leave counter_address zero:
       * command recording replaces absent counters with a distinct mapped
       * zeroed cell per binding slot, separate from this read-as-zero region.
       */
      slot->address = device->null_va;
      slot->size_B = D12_NULL_REGION_SIZE;
      return;
   }

   struct d12_resource *res = d12_resource_from_iface(resource);

   D3D12_UAV_DIMENSION dimension = desc ? desc->ViewDimension
                                        : D3D12_UAV_DIMENSION_UNKNOWN;

   if (res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D ||
       res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
      const unsigned samples = MAX2(res->desc.SampleDesc.Count, 1);
      uint32_t level = 0, first_layer = 0, layers = 1;
      const bool is_3d =
         res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
      const bool is_1d =
         res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D;

      if (samples != 1) {
         mesa_loge("CreateUnorderedAccessView: multisample textures cannot "
                   "have UAVs");
         return;
      }

      if (!desc) {
         layers = MAX2(res->desc.DepthOrArraySize, 1);
         dimension = is_3d   ? D3D12_UAV_DIMENSION_TEXTURE3D
                     : is_1d ? (layers > 1 ? D3D12_UAV_DIMENSION_TEXTURE1DARRAY
                                           : D3D12_UAV_DIMENSION_TEXTURE1D)
                     : layers > 1 ? D3D12_UAV_DIMENSION_TEXTURE2DARRAY
                                  : D3D12_UAV_DIMENSION_TEXTURE2D;
      } else if (is_1d && dimension == D3D12_UAV_DIMENSION_TEXTURE1D) {
         level = desc->Texture1D.MipSlice;
      } else if (is_1d && dimension == D3D12_UAV_DIMENSION_TEXTURE1DARRAY) {
         level = desc->Texture1DArray.MipSlice;
         first_layer = desc->Texture1DArray.FirstArraySlice;
         layers = desc->Texture1DArray.ArraySize;
      } else if (!is_1d && !is_3d &&
                 dimension == D3D12_UAV_DIMENSION_TEXTURE2D) {
         if (desc->Texture2D.PlaneSlice != 0) {
            mesa_loge("CreateUnorderedAccessView: plane slice %u "
                      "unsupported", desc->Texture2D.PlaneSlice);
            return;
         }
         level = desc->Texture2D.MipSlice;
      } else if (!is_1d && !is_3d &&
                 dimension == D3D12_UAV_DIMENSION_TEXTURE2DARRAY) {
         if (desc->Texture2DArray.PlaneSlice != 0) {
            mesa_loge("CreateUnorderedAccessView: plane slice %u "
                      "unsupported", desc->Texture2DArray.PlaneSlice);
            return;
         }
         level = desc->Texture2DArray.MipSlice;
         first_layer = desc->Texture2DArray.FirstArraySlice;
         layers = desc->Texture2DArray.ArraySize;
      } else if (is_3d && dimension == D3D12_UAV_DIMENSION_TEXTURE3D) {
         level = desc->Texture3D.MipSlice;
         first_layer = desc->Texture3D.FirstWSlice;
         layers = desc->Texture3D.WSize;
      } else {
         mesa_loge("CreateUnorderedAccessView: texture view dimension %d "
                   "unsupported", (int)dimension);
         return;
      }

      struct d12_attachment_view view;
      const struct d12_attachment_view_request request = {
         .width = res->layout.width_px,
         .height = res->layout.height_px,
         .levels = res->layout.levels,
         .layers = res->layout.depth_px,
         .mip = level,
         .first_layer = first_layer,
         .layer_count = layers,
         .mipmapped_layers = is_3d,
      };
      if (!d12_attachment_view_resolve(&request, &view)) {
         mesa_loge("CreateUnorderedAccessView: texture range level %u "
                   "layers %u+%u is out of range", level, first_layer,
                   layers);
         return;
      }
      level = view.mip;
      first_layer = view.first_layer;
      layers = view.layer_count;

      DXGI_FORMAT view_format =
         desc && desc->Format != DXGI_FORMAT_UNKNOWN ? desc->Format
                                                     : res->desc.Format;
      enum pipe_format pf = d12_format_to_pipe(view_format);
      if (pf == PIPE_FORMAT_NONE || !ail_is_valid_pixel_format(pf) ||
          !ail_pixel_format[pf].texturable ||
          util_format_is_depth_or_stencil(pf)) {
         mesa_loge("CreateUnorderedAccessView: texture format %u is not "
                   "readable by AGX", (unsigned)view_format);
         return;
      }

      slot->resource = res;
      slot->address = res->addr +
                      (uint64_t)first_layer * res->layout.layer_stride_B;
      /* A sliced 3D UAV is easiest and safest to describe as a view-local
       * miptree: fold its selected mip into the base, then expose level 0
       * with exactly WSize layers.  This makes both GetDimensions and PBE
       * robustness relative to the D3D view instead of to the full image. */
      if (is_3d)
         slot->address += ail_get_level_offset_B(&res->layout, level);

      if (is_3d) {
         uint64_t end = (slot->address - res->addr) +
                        (uint64_t)(layers - 1) *
                           res->layout.layer_stride_B +
                        ail_get_level_size_B(&res->layout, level);
         if (!res->bo || end > res->bo->size) {
            mesa_loge("CreateUnorderedAccessView: 3D slice span "
                      "0x%llx..0x%llx exceeds its 0x%zx-byte BO",
                      (unsigned long long)(slot->address - res->addr),
                      (unsigned long long)end,
                      res->bo ? res->bo->size : 0);
            memset(slot, 0, sizeof(*slot));
            return;
         }
      }
      slot->texture_layers = layers;
      slot->texture_level = level;
      slot->format = view_format;
      slot->kind = D12_DESC_UAV;
      slot->texture_uav_dimension = dimension;
      return;
   }

   if (res->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
       (desc && dimension != D3D12_UAV_DIMENSION_BUFFER)) {
      mesa_loge("CreateUnorderedAccessView: view dimension %d unsupported",
                (int)dimension);
      return;
   }

   uint64_t first = desc ? desc->Buffer.FirstElement : 0;
   uint64_t default_count = res->desc.Width / 4;
   if (!desc && default_count > UINT32_MAX) {
      mesa_loge("CreateUnorderedAccessView: default buffer view has too many "
                "elements");
      return;
   }
   uint32_t count = desc ? desc->Buffer.NumElements : (uint32_t)default_count;
   uint32_t stride = desc ? desc->Buffer.StructureByteStride : 4;

   if (desc && (desc->Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW)) {
      stride = 4;
   } else if (!stride && desc && desc->Format != DXGI_FORMAT_UNKNOWN) {
      stride = d12_format_bytes_per_pixel(desc->Format);
      if (!stride) {
         mesa_loge("CreateUnorderedAccessView: typed buffer format %u "
                   "has no supported element size", (unsigned)desc->Format);
         return;
      }
   }
   if (!stride)
      stride = 4;

   /* A UAV counter is a separate dense buffer allocation reached through
    * bufferUpdateCounter. Reserved counters need residency-gated atomic RMW
    * semantics; until that sideband exists, reject them instead of letting
    * unmapped counters share the writable sparse scratch page. */
   if (counter) {
      struct d12_resource *cres = d12_resource_from_iface(counter);
      const uint64_t off = desc ? desc->Buffer.CounterOffsetInBytes : 0;
      if (!desc || !desc->Buffer.StructureByteStride ||
          cres->device != device || cres->reserved || !cres->bo ||
          cres->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
          (off % D3D12_UAV_COUNTER_PLACEMENT_ALIGNMENT) != 0 ||
          off > cres->desc.Width ||
          sizeof(uint32_t) > cres->desc.Width - off) {
         mesa_loge("CreateUnorderedAccessView: invalid or sparse UAV counter");
         return;
      }
      slot->counter_address = cres->addr + off;
   }

   if (d12_fill_buffer_descriptor(slot, res, first, count, stride,
                                  "CreateUnorderedAccessView")) {
      slot->kind = D12_DESC_UAV;
      slot->format = desc ? desc->Format : DXGI_FORMAT_UNKNOWN;
      slot->buffer_kind =
         desc && (desc->Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW)
            ? D12_BUFFER_VIEW_RAW
         : desc && desc->Buffer.StructureByteStride
            ? D12_BUFFER_VIEW_STRUCTURED
            : D12_BUFFER_VIEW_TYPED;
   }
}

static void STDMETHODCALLTYPE
d12_device_CreateUnorderedAccessView(
   ID3D12Device2 *iface, ID3D12Resource *resource, ID3D12Resource *counter,
   const D3D12_UNORDERED_ACCESS_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   struct d12_device *device = d12_device_from_iface(iface);
   struct d12_descriptor *slot = (struct d12_descriptor *)dest.ptr;

   if (!slot)
      return;

   struct d12_descriptor view = {0};
   d12_device_fill_uav(device, resource, counter, desc, &view);
   *slot = view;
   d12_descriptor_heap_update_slot(device, slot);
}

static void STDMETHODCALLTYPE
d12_device_CreateConstantBufferView(
   ID3D12Device2 *iface, const D3D12_CONSTANT_BUFFER_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   struct d12_device *device = d12_device_from_iface(iface);
   struct d12_descriptor *slot = (struct d12_descriptor *)dest.ptr;

   if (!slot)
      return;

   struct d12_descriptor view = {0};

   if (!desc || !desc->BufferLocation) {
      /* Null CBVs need the same mapped read-as-zero backing as null SRVs
       * and UAVs.  Oversized descriptor tables are dereferenced directly
       * by the shader, so leaving this field zero turns a legal null CBV
       * load (or a type-mismatched null descriptor load) into an unmapped
       * GPU read instead of returning zero.
       */
      view.address = device->null_va;
      view.size_B = D12_NULL_REGION_SIZE;
   } else {
      /* A CBV names a GPU address directly, but retain a reserved-buffer
       * owner when one exists. The read-only sparse alias supplies D3D12's
       * zero-fill behavior for unmapped tiles, while the resource pointer
       * publishes its residency map and current backing allocations to the
       * exact read set. */
      view.kind = D12_DESC_CBV;
      view.address = d12_sparse_resolve_cbv_address(
         device, desc->BufferLocation, &view.resource);
      view.size_B = desc->SizeInBytes;
   }

   *slot = view;
   d12_descriptor_heap_update_slot(device, slot);
}

static UINT STDMETHODCALLTYPE
d12_device_GetDescriptorHandleIncrementSize(
   ID3D12Device2 *iface, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
   /* sizeof(struct d12_descriptor). Apps must treat this as opaque; the
    * value only has to be stable and truthful about our layout.
    */
   switch (type) {
   case D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV:
   case D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER:
   case D3D12_DESCRIPTOR_HEAP_TYPE_RTV:
   case D3D12_DESCRIPTOR_HEAP_TYPE_DSV:
      return sizeof(struct d12_descriptor);
   default:
      return 0;
   }
}

void
d12_device_create_srv_runtime(
   struct d12_device *device, ID3D12Resource *resource,
   const D3D12_SHADER_RESOURCE_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE destination)
{
   d12_device_CreateShaderResourceView(
      (ID3D12Device2 *)&device->iface, resource, desc, destination);
}

void
d12_device_create_uav_runtime(
   struct d12_device *device, ID3D12Resource *resource,
   ID3D12Resource *counter, const D3D12_UNORDERED_ACCESS_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE destination)
{
   d12_device_CreateUnorderedAccessView(
      (ID3D12Device2 *)&device->iface, resource, counter, desc, destination);
}

void
d12_device_create_rtv_runtime(
   struct d12_device *device, ID3D12Resource *resource,
   const D3D12_RENDER_TARGET_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE destination)
{
   d12_device_CreateRenderTargetView(
      (ID3D12Device2 *)&device->iface, resource, desc, destination);
}

void
d12_device_create_dsv_runtime(
   struct d12_device *device, ID3D12Resource *resource,
   const D3D12_DEPTH_STENCIL_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE destination)
{
   d12_device_CreateDepthStencilView(
      (ID3D12Device2 *)&device->iface, resource, desc, destination);
}

void
d12_device_create_cbv_runtime(
   struct d12_device *device, const D3D12_CONSTANT_BUFFER_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE destination)
{
   d12_device_CreateConstantBufferView(
      (ID3D12Device2 *)&device->iface, desc, destination);
}

void
d12_device_create_sampler_runtime(
   struct d12_device *device, const D3D12_SAMPLER_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE destination)
{
   d12_device_CreateSampler(
      (ID3D12Device2 *)&device->iface, desc, destination);
}

static void STDMETHODCALLTYPE
d12_device_CopyDescriptorsSimple(
   ID3D12Device2 *iface, UINT count, D3D12_CPU_DESCRIPTOR_HANDLE dest,
   D3D12_CPU_DESCRIPTOR_HANDLE src, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
   struct d12_device *device = d12_device_from_iface(iface);
   UINT stride = d12_device_GetDescriptorHandleIncrementSize(iface, type);

   if (!count || !stride || !dest.ptr || !src.ptr)
      return;

   /* D3D12 forbids overlap between source and destination ranges. */
   memcpy((void *)(uintptr_t)dest.ptr, (const void *)(uintptr_t)src.ptr,
          (size_t)count * stride);
   d12_descriptor_heap_update_range(
      device, (struct d12_descriptor *)(uintptr_t)dest.ptr, count);
}

void
d12_device_copy_descriptors_simple_runtime(
   struct d12_device *device, UINT count,
   D3D12_CPU_DESCRIPTOR_HANDLE destination,
   D3D12_CPU_DESCRIPTOR_HANDLE source, D3D12_DESCRIPTOR_HEAP_TYPE type)
{
   d12_device_CopyDescriptorsSimple(
      (ID3D12Device2 *)&device->iface, count, destination, source, type);
}

static void STDMETHODCALLTYPE
d12_device_CopyDescriptors(
   ID3D12Device2 *iface, UINT num_dest_ranges,
   const D3D12_CPU_DESCRIPTOR_HANDLE *dest_starts,
   const UINT *dest_sizes, UINT num_src_ranges,
   const D3D12_CPU_DESCRIPTOR_HANDLE *src_starts, const UINT *src_sizes,
   D3D12_DESCRIPTOR_HEAP_TYPE type)
{
   struct d12_device *device = d12_device_from_iface(iface);
   UINT stride = d12_device_GetDescriptorHandleIncrementSize(iface, type);

   if (!stride || !dest_starts || !src_starts)
      return;

   UINT dest_range = 0, src_range = 0;
   UINT dest_offset = 0, src_offset = 0;

   /* The API presents two flattened descriptor streams split across
    * independently-sized ranges. Walk both streams together, skipping empty
    * ranges and copying the largest contiguous run available on each side.
    * A NULL size array means one descriptor per range.
    */
   while (dest_range < num_dest_ranges && src_range < num_src_ranges) {
      UINT dest_count = dest_sizes ? dest_sizes[dest_range] : 1;
      UINT src_count = src_sizes ? src_sizes[src_range] : 1;

      if (dest_offset >= dest_count) {
         dest_range++;
         dest_offset = 0;
         continue;
      }
      if (src_offset >= src_count) {
         src_range++;
         src_offset = 0;
         continue;
      }

      UINT count = MIN2(dest_count - dest_offset, src_count - src_offset);
      uintptr_t dest = dest_starts[dest_range].ptr +
                       (uintptr_t)dest_offset * stride;
      uintptr_t src = src_starts[src_range].ptr +
                      (uintptr_t)src_offset * stride;

      if (!dest || !src)
         return;

      memcpy((void *)dest, (const void *)src, (size_t)count * stride);
      d12_descriptor_heap_update_range(device, (struct d12_descriptor *)dest,
                                       count);
      dest_offset += count;
      src_offset += count;
   }
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreateCommittedResource(ID3D12Device2 *iface,
                                   const D3D12_HEAP_PROPERTIES *heap_props,
                                   D3D12_HEAP_FLAGS heap_flags,
                                   const D3D12_RESOURCE_DESC *desc,
                                   D3D12_RESOURCE_STATES initial_state,
                                   const D3D12_CLEAR_VALUE *clear_value,
                                   REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);

   /* On failure the out pointer must be NULLED. The suite deliberately
    * seeds it with 0xdeadbeef and checks it comes back NULL. */
   if (out)
      *out = NULL;

   return d12_resource_create_committed(device, heap_props, heap_flags, desc,
                                        initial_state, clear_value, riid, out);
}

static unsigned
d12_copyable_plane_count(DXGI_FORMAT format)
{
   return d12_format_stencil_plane(format) != PIPE_FORMAT_NONE ? 2 : 1;
}

static enum pipe_format
d12_copyable_plane_pipe(DXGI_FORMAT format, unsigned plane)
{
   if (d12_format_is_depth_stencil(format)) {
      return plane == 0 ? d12_format_depth_plane(format)
                        : d12_format_stencil_plane(format);
   }

   return plane == 0 ? d12_format_to_pipe(format) : PIPE_FORMAT_NONE;
}

static DXGI_FORMAT
d12_copyable_plane_format(DXGI_FORMAT format, unsigned plane,
                          unsigned plane_count)
{
   /* D3D exposes depth/stencil copies as independent typeless planes, even
    * though the resource itself has a packed or depth-typed DXGI format.
    */
   if (plane_count == 2)
      return plane == 0 ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_R8_TYPELESS;

   return format;
}

/*
 * Describe how a resource's subresources sit in a buffer.  Offsets are
 * relative to BaseOffset, while TotalBytes is the exact span of the selected
 * footprints and deliberately excludes that base.  Rows and depth slices
 * include padding only between neighbours; the final row/slice ends at its
 * real byte size.
 */
static void STDMETHODCALLTYPE
d12_device_GetCopyableFootprints(ID3D12Device2 *iface,
                                 const D3D12_RESOURCE_DESC *desc,
                                 UINT first_subresource,
                                 UINT num_subresources, UINT64 base_offset,
                                 D3D12_PLACED_SUBRESOURCE_FOOTPRINT *layouts,
                                 UINT *num_rows, UINT64 *row_size_bytes,
                                 UINT64 *total_bytes)
{
   (void)iface;

   if (layouts)
      memset(layouts, 0xff, (size_t)num_subresources * sizeof(*layouts));
   if (num_rows)
      memset(num_rows, 0xff, (size_t)num_subresources * sizeof(*num_rows));
   if (row_size_bytes)
      memset(row_size_bytes, 0xff,
             (size_t)num_subresources * sizeof(*row_size_bytes));

   UINT64 total = UINT64_MAX;
   bool buffer;
   unsigned planes, layers, mips;

   if (!desc)
      goto out;

   buffer = desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER;
   mips = desc->MipLevels;
   layers = desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
               ? 1
               : desc->DepthOrArraySize;
   planes = buffer ? 1 : d12_copyable_plane_count(desc->Format);

   /* Reject malformed resource descriptions without partially populating
    * the caller's arrays.  D3D specifies all-ones sentinels for this void
    * API's invalid-input result.
    */
   if (!desc->Width || !desc->Height || !desc->DepthOrArraySize || !mips ||
       !desc->SampleDesc.Count ||
       desc->Dimension < D3D12_RESOURCE_DIMENSION_BUFFER ||
       desc->Dimension > D3D12_RESOURCE_DIMENSION_TEXTURE3D)
      goto out;

   if (buffer) {
      if (desc->Height != 1 || desc->DepthOrArraySize != 1 || mips != 1 ||
          desc->Format != DXGI_FORMAT_UNKNOWN ||
          desc->SampleDesc.Count != 1 ||
          desc->Layout != D3D12_TEXTURE_LAYOUT_ROW_MAJOR)
         goto out;
   } else {
      if (desc->Format == DXGI_FORMAT_UNKNOWN ||
          (desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D &&
           (desc->Height != 1 || desc->SampleDesc.Count != 1)))
         goto out;

      for (unsigned plane = 0; plane < planes; plane++) {
         if (d12_copyable_plane_pipe(desc->Format, plane) == PIPE_FORMAT_NONE)
            goto out;
      }
   }

   uint64_t subresource_count = (uint64_t)mips * layers * planes;
   if (first_subresource >= subresource_count ||
       num_subresources > subresource_count - first_subresource)
      goto out;

   UINT64 offset = 0;
   total = 0;

   for (UINT i = 0; i < num_subresources; i++) {
      UINT sub = first_subresource + i;
      unsigned mip = sub % mips;
      unsigned plane = sub / ((uint64_t)mips * layers);
      enum pipe_format pipe = buffer ? PIPE_FORMAT_R8_UINT
                                     : d12_copyable_plane_pipe(desc->Format,
                                                               plane);
      unsigned block_w = util_format_get_blockwidth(pipe);
      unsigned block_h = util_format_get_blockheight(pipe);
      unsigned block_size = util_format_get_blocksize(pipe);
      uint64_t width64 = buffer ? desc->Width : MAX2(desc->Width >> mip, 1);
      UINT height = buffer ? 1 : MAX2(desc->Height >> mip, 1);
      UINT depth = desc->Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
                      ? MAX2(desc->DepthOrArraySize >> mip, 1)
                      : 1;

      width64 = ALIGN_POT(width64, block_w);
      height = ALIGN_POT(height, block_h);
      if (width64 > UINT32_MAX)
         goto invalid;

      UINT width = width64;
      UINT rows = height / block_h;
      UINT64 row_size = (width64 / block_w) * block_size;
      UINT row_alignment =
         D3D12_TEXTURE_DATA_PITCH_ALIGNMENT * planes;
      UINT64 pitch64 = ALIGN_POT(row_size, row_alignment);
      if (pitch64 > UINT32_MAX)
         goto invalid;
      UINT pitch = pitch64;

      if (layouts) {
         layouts[i].Offset = base_offset + offset;
         layouts[i].Footprint.Format =
            buffer ? DXGI_FORMAT_UNKNOWN
                   : d12_copyable_plane_format(desc->Format, plane, planes);
         layouts[i].Footprint.Width = width;
         layouts[i].Footprint.Height = height;
         layouts[i].Footprint.Depth = depth;
         layouts[i].Footprint.RowPitch = pitch;
      }
      if (num_rows)
         num_rows[i] = rows;
      if (row_size_bytes)
         row_size_bytes[i] = row_size;

      UINT64 slice_size = (uint64_t)(rows - 1) * pitch + row_size;
      UINT64 slice_stride = ALIGN_POT(slice_size, row_alignment);
      UINT64 sub_size = (uint64_t)(depth - 1) * slice_stride + slice_size;
      total = offset + sub_size;
      offset = ALIGN_POT(total, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
   }

out:
   if (total_bytes)
      *total_bytes = total;
   return;

invalid:
   /* A late arithmetic/range failure has the same all-ones contract as an
    * invalid descriptor; erase any earlier partial results first.
    */
   if (layouts)
      memset(layouts, 0xff, (size_t)num_subresources * sizeof(*layouts));
   if (num_rows)
      memset(num_rows, 0xff, (size_t)num_subresources * sizeof(*num_rows));
   if (row_size_bytes)
      memset(row_size_bytes, 0xff,
             (size_t)num_subresources * sizeof(*row_size_bytes));
   total = UINT64_MAX;
   goto out;
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreateFence(ID3D12Device2 *iface, UINT64 initial_value,
                       D3D12_FENCE_FLAGS flags, REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);
   return d12_fence_create(device, initial_value, flags, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device_CreateSharedHandle(ID3D12Device2 *iface,
                              ID3D12DeviceChild *object,
                              const SECURITY_ATTRIBUTES *attributes,
                              DWORD access, LPCWSTR name, HANDLE *handle)
{
   struct d12_device *device = d12_device_from_iface(iface);
   HRESULT hr = d12_resource_create_shared_handle(
      device, object, attributes, access, name, handle);
   if (hr == E_NOTIMPL)
      hr = d12_fence_create_shared_handle(device, object, attributes, access,
                                          name, handle);
   return hr;
}

static HRESULT STDMETHODCALLTYPE
d12_device_OpenSharedHandle(ID3D12Device2 *iface, HANDLE handle, REFIID riid,
                            void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);
   if (d12_guid_equal(riid, &IID_ID3D12Fence))
      return d12_fence_open_shared_handle(device, handle, riid, out);
   if (d12_guid_equal(riid, &IID_ID3D12Heap))
      return d12_heap_open_shared_handle(device, handle, riid, out);
   return d12_resource_open_shared_handle(device, handle, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device_OpenSharedHandleByName(ID3D12Device2 *iface, LPCWSTR name,
                                  DWORD access, HANDLE *handle)
{
   return d12_platform_open_shared_handle_by_name(
      d12_device_from_iface(iface), name, access, handle);
}

static HRESULT
d12_device_set_pageable_resident(struct d12_device *device,
                                 ID3D12Pageable *object, bool resident)
{
   if (!object)
      return E_INVALIDARG;

   ID3D12Resource *resource_iface = NULL;
   if (SUCCEEDED(ID3D12Pageable_QueryInterface(
          object, &IID_ID3D12Resource, (void **)&resource_iface))) {
      if (!d12_resource_iface_is_ours(resource_iface)) {
         ID3D12Resource_Release(resource_iface);
         return E_INVALIDARG;
      }
      struct d12_resource *resource =
         d12_resource_from_iface(resource_iface);
      HRESULT hr = resource->device == device
                      ? d12_platform_set_bo_resident(
                           device, resource->bo, resident)
                      : E_INVALIDARG;
      if (SUCCEEDED(hr) && resource->stencil_bo &&
          resource->stencil_bo != resource->bo) {
         hr = d12_platform_set_bo_resident(device, resource->stencil_bo,
                                           resident);
      }
      ID3D12Resource_Release(resource_iface);
      return hr;
   }

   ID3D12Heap *heap_iface = NULL;
   if (SUCCEEDED(ID3D12Pageable_QueryInterface(
          object, &IID_ID3D12Heap, (void **)&heap_iface))) {
      if (!d12_heap_iface_is_ours(heap_iface)) {
         ID3D12Heap_Release(heap_iface);
         return E_INVALIDARG;
      }
      struct d12_heap *heap = d12_memheap_from_iface(heap_iface);
      HRESULT hr = heap->device == device
                      ? d12_platform_set_bo_resident(device, heap->bo,
                                                     resident)
                      : E_INVALIDARG;
      ID3D12Heap_Release(heap_iface);
      return hr;
   }

   /* Command queues, descriptor heaps and pipeline state objects are also
    * ID3D12Pageable, but their backing is either permanently resident or
    * charged through resources referenced by submitted work. */
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_device_MakeResident(ID3D12Device2 *iface, UINT count,
                        ID3D12Pageable *const *objects)
{
   if (count && !objects)
      return E_INVALIDARG;
   struct d12_device *device = d12_device_from_iface(iface);
   for (UINT i = 0; i < count; i++) {
      HRESULT hr = d12_device_set_pageable_resident(device, objects[i], true);
      if (FAILED(hr))
         return hr;
   }
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_device_Evict(ID3D12Device2 *iface, UINT count,
                 ID3D12Pageable *const *objects)
{
   if (count && !objects)
      return E_INVALIDARG;
   struct d12_device *device = d12_device_from_iface(iface);
   for (UINT i = 0; i < count; i++) {
      HRESULT hr = d12_device_set_pageable_resident(device, objects[i], false);
      if (FAILED(hr))
         return hr;
   }
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_device_SetEventOnMultipleFenceCompletion(
   ID3D12Device2 *iface, ID3D12Fence *const *fences,
   const UINT64 *values, UINT count,
   D3D12_MULTIPLE_FENCE_WAIT_FLAGS flags, HANDLE event)
{
   struct d12_device *device = d12_device_from_iface(iface);
   return d12_fence_set_event_multiple(device, fences, values, count, flags,
                                       event);
}

static HRESULT STDMETHODCALLTYPE
d12_device_GetDeviceRemovedReason(ID3D12Device2 *iface)
{
   struct d12_device *device = d12_device_from_iface(iface);

   /* The recorded cause, not a blanket DEVICE_REMOVED: an application that
    * submitted a recording command list needs to see INVALID_CALL to know
    * the fault was its own. */
   return d12_device_removed_reason(device);
}

static D12_AGGREGATE_RETURN(LUID) STDMETHODCALLTYPE
d12_device_GetAdapterLuid(ID3D12Device2 *iface D12_AGGREGATE_OUT(LUID))
{
   D12_RETURN_AGGREGATE(d12_device_from_iface(iface)->adapter_luid);
}

/* ID3D12Device2's vtable is a superset of ID3D12Device's, so the object
 * carries one vtable and QueryInterface hands out the newer IIDs against
 * the same pointer. Mesh pipelines are created through
 * ID3D12Device2::CreatePipelineState, which takes a STREAM descriptor
 * rather than the fixed GRAPHICS/COMPUTE structs.
 */
static HRESULT STDMETHODCALLTYPE
d12_device_CreateQueryHeap(ID3D12Device2 *iface,
                           const D3D12_QUERY_HEAP_DESC *desc, REFIID riid,
                           void **out)
{
   struct d12_device *device = d12_device_from_iface(iface);
   return d12_query_heap_create(device, desc, riid, out);
}

/* ID3D12Device3..15 methods. The device object is one pointer for every
 * interface version, so these see the same `iface` as the Device2 methods. */
static struct d12_device *
d12_device_from_iface15(ID3D12Device15 *iface)
{
   return d12_device_from_iface((ID3D12Device2 *)iface);
}

static inline ID3D12Device2 *
d12_device2(ID3D12Device15 *iface)
{
   return (ID3D12Device2 *)iface;
}

/* D3D12_RESOURCE_DESC1 is D3D12_RESOURCE_DESC plus a sampler-feedback mip
 * region, which this driver does not implement; the common prefix is what
 * every resource path consumes. */
static D3D12_RESOURCE_DESC
d12_resource_desc_from_desc1(const D3D12_RESOURCE_DESC1 *desc1)
{
   D3D12_RESOURCE_DESC desc = {
      .Dimension = desc1->Dimension,
      .Alignment = desc1->Alignment,
      .Width = desc1->Width,
      .Height = desc1->Height,
      .DepthOrArraySize = desc1->DepthOrArraySize,
      .MipLevels = desc1->MipLevels,
      .Format = desc1->Format,
      .SampleDesc = desc1->SampleDesc,
      .Layout = desc1->Layout,
      .Flags = desc1->Flags,
   };
   return desc;
}

static bool
d12_resource_desc1_supported(const D3D12_RESOURCE_DESC1 *desc1)
{
   return desc1 && desc1->SamplerFeedbackMipRegion.Width == 0 &&
          desc1->SamplerFeedbackMipRegion.Height == 0 &&
          desc1->SamplerFeedbackMipRegion.Depth == 0;
}

/* GetResourceAllocationInfo1/2/3 add a per-resource breakdown of the
 * aggregate answer: each resource's offset inside one heap sized by the
 * aggregate, laid out in order with each resource's own alignment. */
static D3D12_RESOURCE_ALLOCATION_INFO
d12_device_allocation_info1(ID3D12Device2 *iface, UINT visible_mask,
                            UINT count, const D3D12_RESOURCE_DESC *descs,
                            D3D12_RESOURCE_ALLOCATION_INFO1 *info1)
{
   return d12_device_compute_allocation_info(iface, visible_mask, count,
                                             descs, info1);
}

static HRESULT STDMETHODCALLTYPE
d12_device3_EnqueueMakeResident(ID3D12Device15 *iface,
                                D3D12_RESIDENCY_FLAGS flags, UINT count,
                                ID3D12Pageable *const *objects,
                                ID3D12Fence *fence, UINT64 fence_value)
{
   /* UMA residency is immediate; the fence may be signalled right away. */
   HRESULT hr = d12_device_MakeResident(d12_device2(iface), count, objects);
   if (SUCCEEDED(hr) && fence)
      hr = ID3D12Fence_Signal(fence, fence_value);
   return hr;
}

static HRESULT STDMETHODCALLTYPE
d12_device4_CreateCommandList1(ID3D12Device15 *iface, UINT node_mask,
                               D3D12_COMMAND_LIST_TYPE type,
                               D3D12_COMMAND_LIST_FLAGS flags, REFIID riid,
                               void **out)
{
   struct d12_device *device = d12_device_from_iface15(iface);
   if (out)
      *out = NULL;
   if (node_mask > 1 || flags != D3D12_COMMAND_LIST_FLAG_NONE)
      return E_INVALIDARG;
   return d12_cmd_list_create(device, type, NULL, NULL, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device4_CreateCommittedResource1(
   ID3D12Device15 *iface, const D3D12_HEAP_PROPERTIES *heap_props,
   D3D12_HEAP_FLAGS heap_flags, const D3D12_RESOURCE_DESC *desc,
   D3D12_RESOURCE_STATES initial_state, const D3D12_CLEAR_VALUE *clear_value,
   ID3D12ProtectedResourceSession *session, REFIID riid, void **out)
{
   if (out)
      *out = NULL;
   if (session)
      return E_INVALIDARG;
   return d12_device_CreateCommittedResource(d12_device2(iface), heap_props,
                                             heap_flags, desc, initial_state,
                                             clear_value, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device4_CreateHeap1(ID3D12Device15 *iface, const D3D12_HEAP_DESC *desc,
                        ID3D12ProtectedResourceSession *session, REFIID riid,
                        void **out)
{
   if (out)
      *out = NULL;
   if (session)
      return E_INVALIDARG;
   return d12_device_CreateHeap(d12_device2(iface), desc, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device4_CreateReservedResource1(ID3D12Device15 *iface,
                                    const D3D12_RESOURCE_DESC *desc,
                                    D3D12_RESOURCE_STATES initial_state,
                                    const D3D12_CLEAR_VALUE *clear_value,
                                    ID3D12ProtectedResourceSession *session,
                                    REFIID riid, void **out)
{
   if (out)
      *out = NULL;
   if (session)
      return E_INVALIDARG;
   return d12_device_CreateReservedResource(d12_device2(iface), desc,
                                            initial_state, clear_value, riid,
                                            out);
}

static D12_AGGREGATE_RETURN(D3D12_RESOURCE_ALLOCATION_INFO) STDMETHODCALLTYPE
d12_device4_GetResourceAllocationInfo1(
   ID3D12Device15 *iface D12_AGGREGATE_OUT(D3D12_RESOURCE_ALLOCATION_INFO),
   UINT visible_mask, UINT count, const D3D12_RESOURCE_DESC *descs,
   D3D12_RESOURCE_ALLOCATION_INFO1 *info1)
{
   D3D12_RESOURCE_ALLOCATION_INFO info = d12_device_allocation_info1(
      d12_device2(iface), visible_mask, count, descs, info1);
#ifdef _WIN32
   *ret = info;
   return ret;
#else
   return info;
#endif
}

static void STDMETHODCALLTYPE
d12_device5_RemoveDevice(ID3D12Device15 *iface)
{
   d12_device_mark_removed(d12_device_from_iface15(iface),
                           DXGI_ERROR_DEVICE_REMOVED);
}

static HRESULT STDMETHODCALLTYPE
d12_device5_EnumerateMetaCommands(ID3D12Device15 *iface, UINT *count,
                                  D3D12_META_COMMAND_DESC *descs)
{
   if (!count)
      return E_INVALIDARG;
   *count = 0;
   return S_OK;
}

static D3D12_DRIVER_MATCHING_IDENTIFIER_STATUS STDMETHODCALLTYPE
d12_device5_CheckDriverMatchingIdentifier(
   ID3D12Device15 *iface, D3D12_SERIALIZED_DATA_TYPE type,
   const D3D12_SERIALIZED_DATA_DRIVER_MATCHING_IDENTIFIER *identifier)
{
   return D3D12_DRIVER_MATCHING_IDENTIFIER_UNSUPPORTED_TYPE;
}

static HRESULT STDMETHODCALLTYPE
d12_device6_SetBackgroundProcessingMode(ID3D12Device15 *iface,
                                        D3D12_BACKGROUND_PROCESSING_MODE mode,
                                        D3D12_MEASUREMENTS_ACTION action,
                                        HANDLE event, BOOL *more_measurements)
{
   /* No background shader recompilation exists; nothing to measure. */
   if (more_measurements)
      *more_measurements = FALSE;
   return S_OK;
}

static D12_AGGREGATE_RETURN(D3D12_RESOURCE_ALLOCATION_INFO) STDMETHODCALLTYPE
d12_device8_GetResourceAllocationInfo2(
   ID3D12Device15 *iface D12_AGGREGATE_OUT(D3D12_RESOURCE_ALLOCATION_INFO),
   UINT visible_mask, UINT count, const D3D12_RESOURCE_DESC1 *descs1,
   D3D12_RESOURCE_ALLOCATION_INFO1 *info1)
{
   D3D12_RESOURCE_ALLOCATION_INFO info = {
      UINT64_MAX, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
   };
   D3D12_RESOURCE_DESC *descs =
      count ? calloc(count, sizeof(*descs)) : NULL;
   if (!count || (descs && descs1)) {
      bool ok = true;
      for (UINT i = 0; i < count; i++) {
         ok = ok && d12_resource_desc1_supported(&descs1[i]);
         if (ok)
            descs[i] = d12_resource_desc_from_desc1(&descs1[i]);
      }
      if (ok)
         info = d12_device_allocation_info1(d12_device2(iface), visible_mask,
                                            count, descs, info1);
   }
   free(descs);
#ifdef _WIN32
   *ret = info;
   return ret;
#else
   return info;
#endif
}

static HRESULT STDMETHODCALLTYPE
d12_device8_CreateCommittedResource2(
   ID3D12Device15 *iface, const D3D12_HEAP_PROPERTIES *heap_props,
   D3D12_HEAP_FLAGS heap_flags, const D3D12_RESOURCE_DESC1 *desc1,
   D3D12_RESOURCE_STATES initial_state, const D3D12_CLEAR_VALUE *clear_value,
   ID3D12ProtectedResourceSession *session, REFIID riid, void **out)
{
   if (out)
      *out = NULL;
   if (session || !d12_resource_desc1_supported(desc1))
      return E_INVALIDARG;
   D3D12_RESOURCE_DESC desc = d12_resource_desc_from_desc1(desc1);
   return d12_device_CreateCommittedResource(d12_device2(iface), heap_props,
                                             heap_flags, &desc, initial_state,
                                             clear_value, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device8_CreatePlacedResource1(ID3D12Device15 *iface, ID3D12Heap *heap,
                                  UINT64 heap_offset,
                                  const D3D12_RESOURCE_DESC1 *desc1,
                                  D3D12_RESOURCE_STATES initial_state,
                                  const D3D12_CLEAR_VALUE *clear_value,
                                  REFIID riid, void **out)
{
   if (out)
      *out = NULL;
   if (!d12_resource_desc1_supported(desc1))
      return E_INVALIDARG;
   D3D12_RESOURCE_DESC desc = d12_resource_desc_from_desc1(desc1);
   return d12_device_CreatePlacedResource(d12_device2(iface), heap,
                                          heap_offset, &desc, initial_state,
                                          clear_value, riid, out);
}

static void STDMETHODCALLTYPE
d12_device8_GetCopyableFootprints1(
   ID3D12Device15 *iface, const D3D12_RESOURCE_DESC1 *desc1,
   UINT first_subresource, UINT num_subresources, UINT64 base_offset,
   D3D12_PLACED_SUBRESOURCE_FOOTPRINT *layouts, UINT *num_rows,
   UINT64 *row_size_bytes, UINT64 *total_bytes)
{
   if (!desc1) {
      if (total_bytes)
         *total_bytes = UINT64_MAX;
      return;
   }
   D3D12_RESOURCE_DESC desc = d12_resource_desc_from_desc1(desc1);
   d12_device_GetCopyableFootprints(d12_device2(iface), &desc,
                                    first_subresource, num_subresources,
                                    base_offset, layouts, num_rows,
                                    row_size_bytes, total_bytes);
}

static HRESULT STDMETHODCALLTYPE
d12_device9_ShaderCacheControl(ID3D12Device15 *iface,
                               D3D12_SHADER_CACHE_KIND_FLAGS kinds,
                               D3D12_SHADER_CACHE_CONTROL_FLAGS control)
{
   /* There is no persistent shader cache yet; every control is a no-op. */
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_device9_CreateCommandQueue1(ID3D12Device15 *iface,
                                const D3D12_COMMAND_QUEUE_DESC *desc,
                                REFIID creator, REFIID riid, void **out)
{
   /* The creator ID only tags the queue for tooling. */
   return d12_device_CreateCommandQueue(d12_device2(iface), desc, riid, out);
}

/* Enhanced-barrier creation paths: resource states are not tracked, so the
 * initial layout only needs to be a legal one and castable formats, which
 * typeless-equivalent views already permit on UMA, are recorded nowhere. */
static HRESULT STDMETHODCALLTYPE
d12_device10_CreateCommittedResource3(
   ID3D12Device15 *iface, const D3D12_HEAP_PROPERTIES *heap_props,
   D3D12_HEAP_FLAGS heap_flags, const D3D12_RESOURCE_DESC1 *desc1,
   D3D12_BARRIER_LAYOUT initial_layout, const D3D12_CLEAR_VALUE *clear_value,
   ID3D12ProtectedResourceSession *session, UINT32 num_castable_formats,
   const DXGI_FORMAT *castable_formats, REFIID riid, void **out)
{
   if (out)
      *out = NULL;
   if (session || !d12_resource_desc1_supported(desc1) ||
       (num_castable_formats && !castable_formats))
      return E_INVALIDARG;
   D3D12_RESOURCE_DESC desc = d12_resource_desc_from_desc1(desc1);
   return d12_device_CreateCommittedResource(
      d12_device2(iface), heap_props, heap_flags, &desc,
      D3D12_RESOURCE_STATE_COMMON, clear_value, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device10_CreatePlacedResource2(
   ID3D12Device15 *iface, ID3D12Heap *heap, UINT64 heap_offset,
   const D3D12_RESOURCE_DESC1 *desc1, D3D12_BARRIER_LAYOUT initial_layout,
   const D3D12_CLEAR_VALUE *clear_value, UINT32 num_castable_formats,
   const DXGI_FORMAT *castable_formats, REFIID riid, void **out)
{
   if (out)
      *out = NULL;
   if (!d12_resource_desc1_supported(desc1) ||
       (num_castable_formats && !castable_formats))
      return E_INVALIDARG;
   D3D12_RESOURCE_DESC desc = d12_resource_desc_from_desc1(desc1);
   return d12_device_CreatePlacedResource(d12_device2(iface), heap,
                                          heap_offset, &desc,
                                          D3D12_RESOURCE_STATE_COMMON,
                                          clear_value, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_device10_CreateReservedResource2(
   ID3D12Device15 *iface, const D3D12_RESOURCE_DESC *desc,
   D3D12_BARRIER_LAYOUT initial_layout, const D3D12_CLEAR_VALUE *clear_value,
   ID3D12ProtectedResourceSession *session, UINT32 num_castable_formats,
   const DXGI_FORMAT *castable_formats, REFIID riid, void **out)
{
   if (out)
      *out = NULL;
   if (session || (num_castable_formats && !castable_formats))
      return E_INVALIDARG;
   return d12_device_CreateReservedResource(d12_device2(iface), desc,
                                            D3D12_RESOURCE_STATE_COMMON,
                                            clear_value, riid, out);
}

static bool
d12_sampler_desc_from_desc2(const D3D12_SAMPLER_DESC2 *desc2,
                            D3D12_SAMPLER_DESC *desc)
{
   if (!desc2)
      return false;
   /* Integer border colours have no AGX custom-border encoding here. */
   if (desc2->Flags & D3D12_SAMPLER_FLAG_UINT_BORDER_COLOR)
      return false;
   *desc = (D3D12_SAMPLER_DESC){
      .Filter = desc2->Filter,
      .AddressU = desc2->AddressU,
      .AddressV = desc2->AddressV,
      .AddressW = desc2->AddressW,
      .MipLODBias = desc2->MipLODBias,
      .MaxAnisotropy = desc2->MaxAnisotropy,
      .ComparisonFunc = desc2->ComparisonFunc,
      .MinLOD = desc2->MinLOD,
      .MaxLOD = desc2->MaxLOD,
   };
   memcpy(desc->BorderColor, desc2->FloatBorderColor,
          sizeof(desc->BorderColor));
   return true;
}

static void STDMETHODCALLTYPE
d12_device11_CreateSampler2(ID3D12Device15 *iface,
                            const D3D12_SAMPLER_DESC2 *desc2,
                            D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   D3D12_SAMPLER_DESC desc;
   if (!d12_sampler_desc_from_desc2(desc2, &desc)) {
      d12_stub_report("ID3D12Device11::CreateSampler2 (unsupported desc)");
      return;
   }
   d12_device_CreateSampler(d12_device2(iface), &desc, dest);
}

static D12_AGGREGATE_RETURN(D3D12_RESOURCE_ALLOCATION_INFO) STDMETHODCALLTYPE
d12_device12_GetResourceAllocationInfo3(
   ID3D12Device15 *iface D12_AGGREGATE_OUT(D3D12_RESOURCE_ALLOCATION_INFO),
   UINT visible_mask, UINT count, const D3D12_RESOURCE_DESC1 *descs1,
   const UINT32 *num_castable_formats,
   const DXGI_FORMAT *const *castable_formats,
   D3D12_RESOURCE_ALLOCATION_INFO1 *info1)
{
   /* Castable formats never change the allocation on UMA. */
#ifdef _WIN32
   return d12_device8_GetResourceAllocationInfo2(iface, ret, visible_mask,
                                                 count, descs1, info1);
#else
   return d12_device8_GetResourceAllocationInfo2(iface, visible_mask, count,
                                                 descs1, info1);
#endif
}

static HRESULT STDMETHODCALLTYPE
d12_device15_TryCreateShaderResourceView(
   ID3D12Device15 *iface, ID3D12Resource *resource,
   const D3D12_SHADER_RESOURCE_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   if (!dest.ptr || (!resource && !desc))
      return E_INVALIDARG;
   d12_device_CreateShaderResourceView(d12_device2(iface), resource, desc,
                                       dest);
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_device15_TryCreateUnorderedAccessView(
   ID3D12Device15 *iface, ID3D12Resource *resource, ID3D12Resource *counter,
   const D3D12_UNORDERED_ACCESS_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   if (!dest.ptr || (!resource && !desc))
      return E_INVALIDARG;
   d12_device_CreateUnorderedAccessView(d12_device2(iface), resource, counter,
                                        desc, dest);
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_device15_TryCreateConstantBufferView(
   ID3D12Device15 *iface, const D3D12_CONSTANT_BUFFER_VIEW_DESC *desc,
   D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   if (!dest.ptr)
      return E_INVALIDARG;
   d12_device_CreateConstantBufferView(d12_device2(iface), desc, dest);
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_device15_TryCreateSampler2(ID3D12Device15 *iface,
                               const D3D12_SAMPLER_DESC2 *desc2,
                               D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   D3D12_SAMPLER_DESC desc;
   if (!dest.ptr || !d12_sampler_desc_from_desc2(desc2, &desc))
      return E_INVALIDARG;
   d12_device_CreateSampler(d12_device2(iface), &desc, dest);
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_device15_TryCreateRenderTargetView(
   ID3D12Device15 *iface, ID3D12Resource *resource,
   const D3D12_RENDER_TARGET_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   if (!dest.ptr || (!resource && !desc))
      return E_INVALIDARG;
   d12_device_CreateRenderTargetView(d12_device2(iface), resource, desc, dest);
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_device15_TryCreateDepthStencilView(
   ID3D12Device15 *iface, ID3D12Resource *resource,
   const D3D12_DEPTH_STENCIL_VIEW_DESC *desc, D3D12_CPU_DESCRIPTOR_HANDLE dest)
{
   if (!dest.ptr || (!resource && !desc))
      return E_INVALIDARG;
   d12_device_CreateDepthStencilView(d12_device2(iface), resource, desc, dest);
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_device15_CreateQueryHeap1(ID3D12Device15 *iface,
                              const D3D12_QUERY_HEAP_DESC *desc,
                              D3D12_QUERY_HEAP_FLAGS flags, REFIID riid,
                              void **out)
{
   return d12_query_heap_create_flags(d12_device_from_iface15(iface), desc,
                                      flags, riid, out);
}

static bool
d12_cpu_query_type_matches(const struct d12_query_heap *heap,
                           D3D12_QUERY_TYPE type)
{
   return d12_query_core_type_matches((uint32_t)heap->desc.Type,
                                      (uint32_t)type) &&
          (heap->desc.Type != D3D12_QUERY_HEAP_TYPE_SO_STATISTICS ||
           type == D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0);
}

static HRESULT STDMETHODCALLTYPE
d12_device15_ResolveQueryData(ID3D12Device15 *iface,
                              ID3D12QueryHeap *query_heap,
                              D3D12_QUERY_TYPE type, UINT start,
                              UINT count, void *resolved)
{
   struct d12_device *device = d12_device_from_iface15(iface);
   struct d12_query_heap *heap = (struct d12_query_heap *)query_heap;

   if (!heap || !resolved || heap->device != device ||
       !(heap->flags & D3D12_QUERY_HEAP_FLAG_CPU_RESOLVE) ||
       !d12_cpu_query_type_matches(heap, type) ||
       start > heap->desc.Count || count > heap->desc.Count - start)
      return E_INVALIDARG;

   const uint8_t *src = agx_bo_map(heap->bo);
   if (!src)
      return E_OUTOFMEMORY;

   uint8_t *dst = resolved;
   for (UINT i = 0; i < count; i++) {
      const uint8_t *value = src + (size_t)(start + i) * heap->stride;
      if (type == D3D12_QUERY_TYPE_BINARY_OCCLUSION) {
         uint64_t sample;
         memcpy(&sample, value, sizeof(sample));
         sample = sample != 0;
         memcpy(dst + (size_t)i * sizeof(sample), &sample, sizeof(sample));
      } else {
         memcpy(dst + (size_t)i * heap->stride, value, heap->stride);
      }
   }

   return S_OK;
}


static struct d12_device *
d12_device_from_config1(ID3D12DeviceConfiguration1 *iface)
{
   return container_of(iface, struct d12_device, iface_config1);
}

static HRESULT STDMETHODCALLTYPE
d12_device_config1_QueryInterface(ID3D12DeviceConfiguration1 *iface,
                                  REFIID riid, void **out)
{
   struct d12_device *device = d12_device_from_config1(iface);
   return d12_device_QueryInterface((ID3D12Device2 *)&device->iface, riid,
                                    out);
}

static ULONG STDMETHODCALLTYPE
d12_device_config1_AddRef(ID3D12DeviceConfiguration1 *iface)
{
   return d12_device_AddRef(
      (ID3D12Device2 *)&d12_device_from_config1(iface)->iface);
}

static ULONG STDMETHODCALLTYPE
d12_device_config1_Release(ID3D12DeviceConfiguration1 *iface)
{
   return d12_device_Release(
      (ID3D12Device2 *)&d12_device_from_config1(iface)->iface);
}

static D12_AGGREGATE_RETURN(D3D12_DEVICE_CONFIGURATION_DESC) STDMETHODCALLTYPE
d12_device_config1_GetDesc(
   ID3D12DeviceConfiguration1 *iface
   D12_AGGREGATE_OUT(D3D12_DEVICE_CONFIGURATION_DESC))
{
   (void)iface;
   D3D12_DEVICE_CONFIGURATION_DESC desc = {
      .Flags = D3D12_DEVICE_FLAG_NONE,
      .GpuBasedValidationFlags = 0,
      .SDKVersion = D3D12_SDK_VERSION,
      .NumEnabledExperimentalFeatures = 0,
   };
   D12_RETURN_AGGREGATE(desc);
}

static HRESULT STDMETHODCALLTYPE
d12_device_config1_GetEnabledExperimentalFeatures(
   ID3D12DeviceConfiguration1 *iface, GUID *guids, UINT count)
{
   (void)iface;
   (void)guids;
   return count == 0 ? S_OK : E_INVALIDARG;
}

static HRESULT STDMETHODCALLTYPE
d12_device_config1_SerializeVersionedRootSignature(
   ID3D12DeviceConfiguration1 *iface,
   const D3D12_VERSIONED_ROOT_SIGNATURE_DESC *desc, ID3DBlob **blob,
   ID3DBlob **error_blob)
{
   (void)iface;
   if (error_blob)
      *error_blob = NULL;
   if (!blob)
      return E_POINTER;
   *blob = NULL;

   void *data = NULL;
   size_t size = 0;
   HRESULT hr = d12_serialize_versioned_root_signature(desc, &data, &size);
   if (FAILED(hr))
      return hr;

   *blob = d12_blob_create(data, size);
   free(data);
   return *blob ? S_OK : E_OUTOFMEMORY;
}

static HRESULT STDMETHODCALLTYPE
d12_device_config1_CreateVersionedRootSignatureDeserializer(
   ID3D12DeviceConfiguration1 *iface, const void *src, SIZE_T size,
   REFIID riid, void **out)
{
   (void)iface;
   return d12_versioned_root_signature_deserializer_create(src, size, riid,
                                                           out);
}

static HRESULT STDMETHODCALLTYPE
d12_device_config1_CreateVersionedRootSignatureDeserializerFromSubobjectInLibrary(
   ID3D12DeviceConfiguration1 *iface, const void *src, SIZE_T size,
   LPCWSTR subobject_name, REFIID riid, void **out)
{
   (void)iface;
   return d12_versioned_root_signature_deserializer_create_from_library(
      src, size, subobject_name, riid, out);
}

static const ID3D12DeviceConfiguration1Vtbl d12_device_config1_vtbl = {
   .QueryInterface = d12_device_config1_QueryInterface,
   .AddRef = d12_device_config1_AddRef,
   .Release = d12_device_config1_Release,
   .GetDesc = d12_device_config1_GetDesc,
   .GetEnabledExperimentalFeatures =
      d12_device_config1_GetEnabledExperimentalFeatures,
   .SerializeVersionedRootSignature =
      d12_device_config1_SerializeVersionedRootSignature,
   .CreateVersionedRootSignatureDeserializer =
      d12_device_config1_CreateVersionedRootSignatureDeserializer,
   .CreateVersionedRootSignatureDeserializerFromSubobjectInLibrary =
      d12_device_config1_CreateVersionedRootSignatureDeserializerFromSubobjectInLibrary,
};

/* The generated C interfaces repeat every inherited function pointer with a
 * newer This type. Keep the type-checked ID3D12Device2 table as an exact
 * prefix and append the ID3D12Device3..15 slots, so COM callers see one flat
 * ID3D12Device15 vtable while every callback stays checked against the
 * interface it was written for (the same shape as d12_list9_vtbl). */
struct d12_device_vtbl {
   ID3D12Device2Vtbl v2;
   /* ID3D12Device3 */
   HRESULT (STDMETHODCALLTYPE *OpenExistingHeapFromAddress)(
      ID3D12Device15 *, const void *, REFIID, void **);
   HRESULT (STDMETHODCALLTYPE *OpenExistingHeapFromFileMapping)(
      ID3D12Device15 *, HANDLE, REFIID, void **);
   HRESULT (STDMETHODCALLTYPE *EnqueueMakeResident)(
      ID3D12Device15 *, D3D12_RESIDENCY_FLAGS, UINT, ID3D12Pageable *const *,
      ID3D12Fence *, UINT64);
   /* ID3D12Device4 */
   HRESULT (STDMETHODCALLTYPE *CreateCommandList1)(
      ID3D12Device15 *, UINT, D3D12_COMMAND_LIST_TYPE,
      D3D12_COMMAND_LIST_FLAGS, REFIID, void **);
   HRESULT (STDMETHODCALLTYPE *CreateProtectedResourceSession)(
      ID3D12Device15 *, const D3D12_PROTECTED_RESOURCE_SESSION_DESC *, REFIID,
      void **);
   HRESULT (STDMETHODCALLTYPE *CreateCommittedResource1)(
      ID3D12Device15 *, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS,
      const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES,
      const D3D12_CLEAR_VALUE *, ID3D12ProtectedResourceSession *, REFIID,
      void **);
   HRESULT (STDMETHODCALLTYPE *CreateHeap1)(
      ID3D12Device15 *, const D3D12_HEAP_DESC *,
      ID3D12ProtectedResourceSession *, REFIID, void **);
   HRESULT (STDMETHODCALLTYPE *CreateReservedResource1)(
      ID3D12Device15 *, const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES,
      const D3D12_CLEAR_VALUE *, ID3D12ProtectedResourceSession *, REFIID,
      void **);
#ifdef _WIN32
   D3D12_RESOURCE_ALLOCATION_INFO *(STDMETHODCALLTYPE *GetResourceAllocationInfo1)(
      ID3D12Device15 *, D3D12_RESOURCE_ALLOCATION_INFO *, UINT, UINT,
      const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_ALLOCATION_INFO1 *);
#else
   D3D12_RESOURCE_ALLOCATION_INFO (STDMETHODCALLTYPE *GetResourceAllocationInfo1)(
      ID3D12Device15 *, UINT, UINT, const D3D12_RESOURCE_DESC *,
      D3D12_RESOURCE_ALLOCATION_INFO1 *);
#endif
   /* ID3D12Device5 */
   HRESULT (STDMETHODCALLTYPE *CreateLifetimeTracker)(
      ID3D12Device15 *, ID3D12LifetimeOwner *, REFIID, void **);
   void (STDMETHODCALLTYPE *RemoveDevice)(ID3D12Device15 *);
   HRESULT (STDMETHODCALLTYPE *EnumerateMetaCommands)(
      ID3D12Device15 *, UINT *, D3D12_META_COMMAND_DESC *);
   HRESULT (STDMETHODCALLTYPE *EnumerateMetaCommandParameters)(
      ID3D12Device15 *, REFGUID, D3D12_META_COMMAND_PARAMETER_STAGE, UINT *,
      UINT *, D3D12_META_COMMAND_PARAMETER_DESC *);
   HRESULT (STDMETHODCALLTYPE *CreateMetaCommand)(
      ID3D12Device15 *, REFGUID, UINT, const void *, SIZE_T, REFIID, void **);
   HRESULT (STDMETHODCALLTYPE *CreateStateObject)(
      ID3D12Device15 *, const D3D12_STATE_OBJECT_DESC *, REFIID, void **);
   void (STDMETHODCALLTYPE *GetRaytracingAccelerationStructurePrebuildInfo)(
      ID3D12Device15 *,
      const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS *,
      D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO *);
   D3D12_DRIVER_MATCHING_IDENTIFIER_STATUS (
      STDMETHODCALLTYPE *CheckDriverMatchingIdentifier)(
      ID3D12Device15 *, D3D12_SERIALIZED_DATA_TYPE,
      const D3D12_SERIALIZED_DATA_DRIVER_MATCHING_IDENTIFIER *);
   /* ID3D12Device6 */
   HRESULT (STDMETHODCALLTYPE *SetBackgroundProcessingMode)(
      ID3D12Device15 *, D3D12_BACKGROUND_PROCESSING_MODE,
      D3D12_MEASUREMENTS_ACTION, HANDLE, BOOL *);
   /* ID3D12Device7 */
   HRESULT (STDMETHODCALLTYPE *AddToStateObject)(
      ID3D12Device15 *, const D3D12_STATE_OBJECT_DESC *, ID3D12StateObject *,
      REFIID, void **);
   HRESULT (STDMETHODCALLTYPE *CreateProtectedResourceSession1)(
      ID3D12Device15 *, const D3D12_PROTECTED_RESOURCE_SESSION_DESC1 *,
      REFIID, void **);
   /* ID3D12Device8 */
#ifdef _WIN32
   D3D12_RESOURCE_ALLOCATION_INFO *(STDMETHODCALLTYPE *GetResourceAllocationInfo2)(
      ID3D12Device15 *, D3D12_RESOURCE_ALLOCATION_INFO *, UINT, UINT,
      const D3D12_RESOURCE_DESC1 *, D3D12_RESOURCE_ALLOCATION_INFO1 *);
#else
   D3D12_RESOURCE_ALLOCATION_INFO (STDMETHODCALLTYPE *GetResourceAllocationInfo2)(
      ID3D12Device15 *, UINT, UINT, const D3D12_RESOURCE_DESC1 *,
      D3D12_RESOURCE_ALLOCATION_INFO1 *);
#endif
   HRESULT (STDMETHODCALLTYPE *CreateCommittedResource2)(
      ID3D12Device15 *, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS,
      const D3D12_RESOURCE_DESC1 *, D3D12_RESOURCE_STATES,
      const D3D12_CLEAR_VALUE *, ID3D12ProtectedResourceSession *, REFIID,
      void **);
   HRESULT (STDMETHODCALLTYPE *CreatePlacedResource1)(
      ID3D12Device15 *, ID3D12Heap *, UINT64, const D3D12_RESOURCE_DESC1 *,
      D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **);
   void (STDMETHODCALLTYPE *CreateSamplerFeedbackUnorderedAccessView)(
      ID3D12Device15 *, ID3D12Resource *, ID3D12Resource *,
      D3D12_CPU_DESCRIPTOR_HANDLE);
   void (STDMETHODCALLTYPE *GetCopyableFootprints1)(
      ID3D12Device15 *, const D3D12_RESOURCE_DESC1 *, UINT, UINT, UINT64,
      D3D12_PLACED_SUBRESOURCE_FOOTPRINT *, UINT *, UINT64 *, UINT64 *);
   /* ID3D12Device9 */
   HRESULT (STDMETHODCALLTYPE *CreateShaderCacheSession)(
      ID3D12Device15 *, const D3D12_SHADER_CACHE_SESSION_DESC *, REFIID,
      void **);
   HRESULT (STDMETHODCALLTYPE *ShaderCacheControl)(
      ID3D12Device15 *, D3D12_SHADER_CACHE_KIND_FLAGS,
      D3D12_SHADER_CACHE_CONTROL_FLAGS);
   HRESULT (STDMETHODCALLTYPE *CreateCommandQueue1)(
      ID3D12Device15 *, const D3D12_COMMAND_QUEUE_DESC *, REFIID, REFIID,
      void **);
   /* ID3D12Device10 */
   HRESULT (STDMETHODCALLTYPE *CreateCommittedResource3)(
      ID3D12Device15 *, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS,
      const D3D12_RESOURCE_DESC1 *, D3D12_BARRIER_LAYOUT,
      const D3D12_CLEAR_VALUE *, ID3D12ProtectedResourceSession *, UINT32,
      const DXGI_FORMAT *, REFIID, void **);
   HRESULT (STDMETHODCALLTYPE *CreatePlacedResource2)(
      ID3D12Device15 *, ID3D12Heap *, UINT64, const D3D12_RESOURCE_DESC1 *,
      D3D12_BARRIER_LAYOUT, const D3D12_CLEAR_VALUE *, UINT32,
      const DXGI_FORMAT *, REFIID, void **);
   HRESULT (STDMETHODCALLTYPE *CreateReservedResource2)(
      ID3D12Device15 *, const D3D12_RESOURCE_DESC *, D3D12_BARRIER_LAYOUT,
      const D3D12_CLEAR_VALUE *, ID3D12ProtectedResourceSession *, UINT32,
      const DXGI_FORMAT *, REFIID, void **);
   /* ID3D12Device11 */
   void (STDMETHODCALLTYPE *CreateSampler2)(
      ID3D12Device15 *, const D3D12_SAMPLER_DESC2 *,
      D3D12_CPU_DESCRIPTOR_HANDLE);
   /* ID3D12Device12 */
#ifdef _WIN32
   D3D12_RESOURCE_ALLOCATION_INFO *(STDMETHODCALLTYPE *GetResourceAllocationInfo3)(
      ID3D12Device15 *, D3D12_RESOURCE_ALLOCATION_INFO *, UINT, UINT,
      const D3D12_RESOURCE_DESC1 *, const UINT32 *,
      const DXGI_FORMAT *const *, D3D12_RESOURCE_ALLOCATION_INFO1 *);
#else
   D3D12_RESOURCE_ALLOCATION_INFO (STDMETHODCALLTYPE *GetResourceAllocationInfo3)(
      ID3D12Device15 *, UINT, UINT, const D3D12_RESOURCE_DESC1 *,
      const UINT32 *, const DXGI_FORMAT *const *,
      D3D12_RESOURCE_ALLOCATION_INFO1 *);
#endif
   /* ID3D12Device13 */
   HRESULT (STDMETHODCALLTYPE *OpenExistingHeapFromAddress1)(
      ID3D12Device15 *, const void *, SIZE_T, REFIID, void **);
   /* ID3D12Device14 */
   HRESULT (STDMETHODCALLTYPE *CreateRootSignatureFromSubobjectInLibrary)(
      ID3D12Device15 *, UINT, const void *, SIZE_T, LPCWSTR, REFIID, void **);
   /* ID3D12Device15 */
   HRESULT (STDMETHODCALLTYPE *RegisterTrimNotificationCallback)(
      ID3D12Device15 *, D3D12_REGISTER_TRIM_NOTIFICATION *);
   HRESULT (STDMETHODCALLTYPE *UnregisterTrimNotificationCallback)(
      ID3D12Device15 *, DWORD);
   HRESULT (STDMETHODCALLTYPE *TryCreateShaderResourceView)(
      ID3D12Device15 *, ID3D12Resource *,
      const D3D12_SHADER_RESOURCE_VIEW_DESC *, D3D12_CPU_DESCRIPTOR_HANDLE);
   HRESULT (STDMETHODCALLTYPE *TryCreateUnorderedAccessView)(
      ID3D12Device15 *, ID3D12Resource *, ID3D12Resource *,
      const D3D12_UNORDERED_ACCESS_VIEW_DESC *, D3D12_CPU_DESCRIPTOR_HANDLE);
   HRESULT (STDMETHODCALLTYPE *TryCreateConstantBufferView)(
      ID3D12Device15 *, const D3D12_CONSTANT_BUFFER_VIEW_DESC *,
      D3D12_CPU_DESCRIPTOR_HANDLE);
   HRESULT (STDMETHODCALLTYPE *TryCreateSampler2)(
      ID3D12Device15 *, const D3D12_SAMPLER_DESC2 *,
      D3D12_CPU_DESCRIPTOR_HANDLE);
   HRESULT (STDMETHODCALLTYPE *TryCreateRenderTargetView)(
      ID3D12Device15 *, ID3D12Resource *, const D3D12_RENDER_TARGET_VIEW_DESC *,
      D3D12_CPU_DESCRIPTOR_HANDLE);
   HRESULT (STDMETHODCALLTYPE *TryCreateDepthStencilView)(
      ID3D12Device15 *, ID3D12Resource *, const D3D12_DEPTH_STENCIL_VIEW_DESC *,
      D3D12_CPU_DESCRIPTOR_HANDLE);
   HRESULT (STDMETHODCALLTYPE *TryCreateSamplerFeedbackUnorderedAccessView)(
      ID3D12Device15 *, ID3D12Resource *, ID3D12Resource *,
      D3D12_CPU_DESCRIPTOR_HANDLE);
   HRESULT (STDMETHODCALLTYPE *CreateQueryHeap1)(
      ID3D12Device15 *, const D3D12_QUERY_HEAP_DESC *, D3D12_QUERY_HEAP_FLAGS,
      REFIID, void **);
   HRESULT (STDMETHODCALLTYPE *ResolveQueryData)(
      ID3D12Device15 *, ID3D12QueryHeap *, D3D12_QUERY_TYPE, UINT, UINT,
      void *);
};

static_assert(offsetof(ID3D12Device15Vtbl, OpenExistingHeapFromAddress) ==
                 sizeof(ID3D12Device2Vtbl),
              "device 2 must be an exact vtable prefix of device 15");
static_assert(offsetof(struct d12_device_vtbl, CreateQueryHeap1) ==
                 offsetof(ID3D12Device15Vtbl, CreateQueryHeap1),
              "private device-15 vtable must match the public ABI");
static_assert(sizeof(struct d12_device_vtbl) == sizeof(ID3D12Device15Vtbl),
              "private device-15 vtable must match the public ABI");

#define D12_DEVICE15_TAIL(name) .name = d12_stub_ID3D12Device15_##name

static const struct d12_device_vtbl d12_device_vtbl = {
   /* every unimplemented slot fails safely */
   D12_DEVICE15_TAIL(OpenExistingHeapFromAddress),
   D12_DEVICE15_TAIL(OpenExistingHeapFromFileMapping),
   .EnqueueMakeResident = d12_device3_EnqueueMakeResident,
   .CreateCommandList1 = d12_device4_CreateCommandList1,
   D12_DEVICE15_TAIL(CreateProtectedResourceSession),
   .CreateCommittedResource1 = d12_device4_CreateCommittedResource1,
   .CreateHeap1 = d12_device4_CreateHeap1,
   .CreateReservedResource1 = d12_device4_CreateReservedResource1,
   .GetResourceAllocationInfo1 = d12_device4_GetResourceAllocationInfo1,
   D12_DEVICE15_TAIL(CreateLifetimeTracker),
   .RemoveDevice = d12_device5_RemoveDevice,
   .EnumerateMetaCommands = d12_device5_EnumerateMetaCommands,
   D12_DEVICE15_TAIL(EnumerateMetaCommandParameters),
   D12_DEVICE15_TAIL(CreateMetaCommand),
   D12_DEVICE15_TAIL(CreateStateObject),
   D12_DEVICE15_TAIL(GetRaytracingAccelerationStructurePrebuildInfo),
   .CheckDriverMatchingIdentifier = d12_device5_CheckDriverMatchingIdentifier,
   .SetBackgroundProcessingMode = d12_device6_SetBackgroundProcessingMode,
   D12_DEVICE15_TAIL(AddToStateObject),
   D12_DEVICE15_TAIL(CreateProtectedResourceSession1),
   .GetResourceAllocationInfo2 = d12_device8_GetResourceAllocationInfo2,
   .CreateCommittedResource2 = d12_device8_CreateCommittedResource2,
   .CreatePlacedResource1 = d12_device8_CreatePlacedResource1,
   D12_DEVICE15_TAIL(CreateSamplerFeedbackUnorderedAccessView),
   .GetCopyableFootprints1 = d12_device8_GetCopyableFootprints1,
   D12_DEVICE15_TAIL(CreateShaderCacheSession),
   .ShaderCacheControl = d12_device9_ShaderCacheControl,
   .CreateCommandQueue1 = d12_device9_CreateCommandQueue1,
   .CreateCommittedResource3 = d12_device10_CreateCommittedResource3,
   .CreatePlacedResource2 = d12_device10_CreatePlacedResource2,
   .CreateReservedResource2 = d12_device10_CreateReservedResource2,
   .CreateSampler2 = d12_device11_CreateSampler2,
   .GetResourceAllocationInfo3 = d12_device12_GetResourceAllocationInfo3,
   D12_DEVICE15_TAIL(OpenExistingHeapFromAddress1),
   D12_DEVICE15_TAIL(CreateRootSignatureFromSubobjectInLibrary),
   D12_DEVICE15_TAIL(RegisterTrimNotificationCallback),
   D12_DEVICE15_TAIL(UnregisterTrimNotificationCallback),
   .TryCreateShaderResourceView = d12_device15_TryCreateShaderResourceView,
   .TryCreateUnorderedAccessView = d12_device15_TryCreateUnorderedAccessView,
   .TryCreateConstantBufferView = d12_device15_TryCreateConstantBufferView,
   .TryCreateSampler2 = d12_device15_TryCreateSampler2,
   .TryCreateRenderTargetView = d12_device15_TryCreateRenderTargetView,
   .TryCreateDepthStencilView = d12_device15_TryCreateDepthStencilView,
   D12_DEVICE15_TAIL(TryCreateSamplerFeedbackUnorderedAccessView),
   .CreateQueryHeap1 = d12_device15_CreateQueryHeap1,
   .ResolveQueryData = d12_device15_ResolveQueryData,

   .v2 = {
   D12_ID3D12Device2_DEFAULTS,
   .QueryInterface = d12_device_QueryInterface,
   .AddRef = d12_device_AddRef,
   .Release = d12_device_Release,
   .GetPrivateData = d12_device_GetPrivateData,
   .SetPrivateData = d12_device_SetPrivateData,
   .SetPrivateDataInterface = d12_device_SetPrivateDataInterface,
   .SetName = d12_device_SetName,
   .GetNodeCount = d12_device_GetNodeCount,
   .CreateCommandQueue = d12_device_CreateCommandQueue,
   .CreateCommandAllocator = d12_device_CreateCommandAllocator,
   .CreateCommandList = d12_device_CreateCommandList,
   .CheckFeatureSupport = d12_device_CheckFeatureSupport,
   .CreateGraphicsPipelineState = d12_device_CreateGraphicsPipelineState,
   .CreatePipelineState = d12_device_CreatePipelineState,
   .CreateComputePipelineState = d12_device_CreateComputePipelineState,
   .CreatePipelineLibrary = d12_device_CreatePipelineLibrary,
   .CreateRootSignature = d12_device_CreateRootSignature,
   .CreateDescriptorHeap = d12_device_CreateDescriptorHeap,
   .CreateCommandSignature = d12_device_CreateCommandSignature,
   .CreateHeap = d12_device_CreateHeap,
   .CreatePlacedResource = d12_device_CreatePlacedResource,
   .CreateReservedResource = d12_device_CreateReservedResource,
   .GetResourceTiling = d12_device_GetResourceTiling,
   .GetCustomHeapProperties = d12_device_GetCustomHeapProperties,
   .GetResourceAllocationInfo = d12_device_GetResourceAllocationInfo,
   .CreateRenderTargetView = d12_device_CreateRenderTargetView,
   .CreateDepthStencilView = d12_device_CreateDepthStencilView,
   .CreateSampler = d12_device_CreateSampler,
   .CreateShaderResourceView = d12_device_CreateShaderResourceView,
   .CreateUnorderedAccessView = d12_device_CreateUnorderedAccessView,
   .CreateConstantBufferView = d12_device_CreateConstantBufferView,
   .GetDescriptorHandleIncrementSize =
      d12_device_GetDescriptorHandleIncrementSize,
   .CopyDescriptors = d12_device_CopyDescriptors,
   .CopyDescriptorsSimple = d12_device_CopyDescriptorsSimple,
   .CreateCommittedResource = d12_device_CreateCommittedResource,
   .GetCopyableFootprints = d12_device_GetCopyableFootprints,
   .CreateQueryHeap = d12_device_CreateQueryHeap,
   .CreateFence = d12_device_CreateFence,
   .CreateSharedHandle = d12_device_CreateSharedHandle,
   .OpenSharedHandle = d12_device_OpenSharedHandle,
   .OpenSharedHandleByName = d12_device_OpenSharedHandleByName,
   .MakeResident = d12_device_MakeResident,
   .Evict = d12_device_Evict,
   .SetEventOnMultipleFenceCompletion =
      d12_device_SetEventOnMultipleFenceCompletion,
   .GetDeviceRemovedReason = d12_device_GetDeviceRemovedReason,
   .GetAdapterLuid = d12_device_GetAdapterLuid,
   /* Everything unset keeps its generated fail-safe stub. */
   },
};

/*
 * Entry points
 */

/*
 * Frontend-neutral initialization of the native AGX D3D12 backend.  The
 * caller owns discovery and winsys construction and must provide an already
 * opened struct agx_device.  This is the seam used by both the public COM
 * factory and the WDDM DDI adapter: the latter injects runtime callbacks and
 * therefore must never call d12_platform_open_device().
 */
HRESULT
d12_device_backend_init(struct d12_device *device,
                        D3D_FEATURE_LEVEL feature_level)
{
   d12_device_removal_state_init(&device->removal);
   agx_bg_eot_init(&device->bg_eot, &device->dev);
   if (!device->bg_eot.ht)
      return E_FAIL;

   device->oclqry_bo =
      agx_bo_create(&device->dev, 0x4000, 0, 0, "d3d12 oclqry scratch");
   if (!device->oclqry_bo)
      goto fail_bg;

   /* Sampler 0, tessellation descriptor, then the zero-backed NULL region. */
   const unsigned sampler_size =
      agx_chip_sampler_heap_element_size(device->dev.chip);
   if (!sampler_size)
      goto fail_oclqry;
   device->rodata_bo =
      agx_bo_create(&device->dev,
                    ALIGN_POT(sampler_size, 16) +
                       sizeof(struct poly_heap) + D12_NULL_REGION_SIZE,
                    0, 0, "d3d12 rodata");
   if (!device->rodata_bo)
      goto fail_oclqry;

   /* Lock2 can refuse an allocation dxgkrnl just handed us.  Everything below
    * writes through this mapping, so treat a failed map as a failed device
    * rather than faulting during device creation. */
   uint8_t *rodata = agx_bo_map(device->rodata_bo);
   if (!rodata)
      goto fail_rodata;

   memset(rodata, 0, sampler_size);
   if (device->dev.chip == AGX_CHIP_G17P) {
      if (sampler_size != 32 || agx_chip_sampler_generation(device->dev.chip) !=
                               AGX_SAMPLER_GENERATION_GEN4_HAL200)
         goto fail_rodata;
      agx_pack(rodata, SAMPLER_HAL200, cfg) {
         cfg.minimum_lod = 0.0f;
         cfg.maximum_lod = INFINITY;
         cfg.maximum_anisotropy = 1;
         cfg.mip_filter = AGX_MIP_FILTER_NEAREST;
         cfg.wrap_s = cfg.wrap_t = cfg.wrap_r = AGX_WRAP_CLAMP_TO_BORDER;
         cfg.border_colour = AGX_BORDER_COLOUR_TRANSPARENT_BLACK;
      }
      memset(&device->dev.txf_sampler, 0, sizeof(device->dev.txf_sampler));
   } else {
      agx_pack_txf_sampler((struct agx_sampler_packed *)rodata);
      agx_pack(&device->dev.txf_sampler, USC_SAMPLER, cfg) {
         cfg.start = 0;
         cfg.count = 1;
         cfg.buffer = device->rodata_bo->va->addr;
      }
   }

   /* Match Honeykrisp's 128 MiB geometry/tessellation heap. */
   device->heap_bo = agx_bo_create(&device->dev, 128 * 1024 * 1024, 0, 0,
                                   "d3d12 tessellation heap");
   if (!device->heap_bo)
      goto fail_rodata;

   const size_t heap_off = ALIGN_POT(sampler_size, 16);
   const size_t null_off = heap_off + sizeof(struct poly_heap);
   device->heap_desc_va = device->rodata_bo->va->addr + heap_off;
   device->null_va = device->rodata_bo->va->addr + null_off;
   memset(rodata + null_off, 0, D12_NULL_REGION_SIZE);
   device->heap_desc_cpu = (struct poly_heap *)(rodata + heap_off);
   *device->heap_desc_cpu = (struct poly_heap){
      .base = (DEVICE(uchar))device->heap_bo->va->addr,
      .bottom = 0,
      .size = (uint32_t)device->heap_bo->size,
   };

   /* The advertised typed-UAV-load promise is checked against the table that
    * answers CheckFormatSupport, once, where a mismatch is still attributable
    * to the driver rather than to an application reading zeroes. */
   {
      DXGI_FORMAT missing = DXGI_FORMAT_UNKNOWN;
      if (!d12_format_typed_uav_load_complete(
             d12_core_get_caps()->typed_uav_load_additional_formats,
             &missing)) {
         mesa_loge("capability policy advertises typed UAV loads that DXGI "
                   "format %u does not report", (unsigned)missing);
      }
   }

   simple_mtx_init(&device->scratch_lock, mtx_plain);
   simple_mtx_init(&device->descriptor_heap_lock, mtx_plain);
   simple_mtx_init(&device->sparse_resource_lock, mtx_plain);
   d12_device_g17p_vdm_pool_init(device);
   list_inithead(&device->sparse_resources);
   d12_shader_cache_init(device);
   agx_scratch_init(&device->dev, &device->scratch_vs);
   agx_scratch_init(&device->dev, &device->scratch_fs);
   agx_scratch_init(&device->dev, &device->scratch_cs);

   device->iface.lpVtbl = (ID3D12DeviceVtbl *)&d12_device_vtbl;
   device->iface_config1.lpVtbl =
      (ID3D12DeviceConfiguration1Vtbl *)&d12_device_config1_vtbl;
   device->refcount = 1;
   device->feature_level = feature_level;
   d12_private_data_store_init(&device->private_data,
                               (IUnknown *)&device->iface);
   return S_OK;

fail_rodata:
   agx_bo_unreference(&device->dev, device->rodata_bo);
   device->rodata_bo = NULL;
fail_oclqry:
   agx_bo_unreference(&device->dev, device->oclqry_bo);
   device->oclqry_bo = NULL;
fail_bg:
   agx_bg_eot_cleanup(&device->bg_eot);
   return E_OUTOFMEMORY;
}

void
d12_device_backend_cleanup(struct d12_device *device)
{
   d12_private_data_store_destroy(&device->private_data);
   agx_bo_unreference(&device->dev, device->oclqry_bo);
   agx_bo_unreference(&device->dev, device->rodata_bo);
   if (device->heap_bo)
      agx_bo_unreference(&device->dev, device->heap_bo);
   agx_scratch_fini(&device->scratch_vs);
   agx_scratch_fini(&device->scratch_fs);
   agx_scratch_fini(&device->scratch_cs);
   d12_device_g17p_vdm_pool_fini(device);
   d12_shader_cache_finish(device);
   simple_mtx_destroy(&device->scratch_lock);
   d12_descriptor_registry_destroy(device);
   simple_mtx_destroy(&device->descriptor_heap_lock);
   simple_mtx_destroy(&device->sparse_resource_lock);
   agx_bg_eot_cleanup(&device->bg_eot);
}

PUBLIC HRESULT WINAPI
D3D12CreateDevice(IUnknown *adapter, D3D_FEATURE_LEVEL minimum_feature_level,
                  REFIID riid, void **out)
{
   const D3D_FEATURE_LEVEL max_fl =
      (D3D_FEATURE_LEVEL)d12_core_get_caps()->max_feature_level;

   /* D3D12 has no levels below 11_0; accepting 9_x/10_x would claim
    * support for a contract that does not exist here.
    */
   if (!d12_core_supports_feature_level((uint32_t)minimum_feature_level)) {
      mesa_loge("D3D12CreateDevice: feature level 0x%x unsupported (this "
                "driver is 0x%x only)",
                minimum_feature_level, max_fl);
      return E_INVALIDARG;
   }

   /* Answer the capability probe before opening anything, so the result
    * does not depend on whether a GPU node happens to be present.
    */
   if (!out)
      return S_FALSE;

   struct d12_device *device = calloc(1, sizeof(*device));
   if (!device)
      return E_OUTOFMEMORY;

   if (!d12_platform_open_device(device, adapter)) {
      mesa_loge("D3D12CreateDevice: no compatible AGX device found");
      free(device);
      return E_FAIL;
   }

#ifdef _WIN32
   device->submit_read_set = true;
#else
   device->submit_read_set = device->dev.is_virtio;
#endif

   HRESULT hr = d12_device_backend_init(device, max_fl);
   if (FAILED(hr)) {
      mesa_loge("D3D12CreateDevice: native backend initialization failed");
      d12_platform_close_device(device);
      free(device);
      return hr;
   }

   mesa_logi("opened %s (chip 0x%x) as D3D12 device, feature level 0x%x",
             device->dev.name, device->dev.params.chip_id,
             device->feature_level);

   hr = d12_device_QueryInterface((ID3D12Device2 *)&device->iface, riid, out);
   /* Drop the construction ref; QueryInterface holds the caller's. */
   d12_device_Release((ID3D12Device2 *)&device->iface);
   return hr;
}

PUBLIC HRESULT WINAPI
D3D12GetDebugInterface(REFIID riid, void **out)
{
   /* No debug layer. Callers treat this as optional. */
   if (out)
      *out = NULL;
   return E_NOINTERFACE;
}

/*
 * Remaining module-level entry points.
 *
 * These are exported so a conformance suite can LINK and RUN against us:
 * a missing symbol fails to resolve and kills the process before a single
 * test reports, whereas an honest failure here is a recordable result.
 * They are deliberately not faked — each returns a real failure until the
 * corresponding feature exists.
 */

PUBLIC HRESULT WINAPI
D3D12SerializeRootSignature(const D3D12_ROOT_SIGNATURE_DESC *desc,
                            D3D_ROOT_SIGNATURE_VERSION version,
                            ID3DBlob **blob, ID3DBlob **error_blob)
{
   if (error_blob)
      *error_blob = NULL;
   if (!blob)
      return E_POINTER;

   /* The legacy entry point only accepts the legacy description.  Native
    * also leaves the caller's output pointer untouched for this version
    * error, which applications use to distinguish argument validation.
    */
   if (version != D3D_ROOT_SIGNATURE_VERSION_1_0) {
      mesa_loge("D3D12SerializeRootSignature: version 0x%x unsupported",
                (unsigned)version);
      return E_INVALIDARG;
   }

   *blob = NULL;

   void *data = NULL;
   size_t size = 0;
   HRESULT hr = d12_serialize_root_signature(desc, &data, &size);
   if (FAILED(hr))
      return hr;

   *blob = d12_blob_create(data, size);
   free(data);

   return *blob ? S_OK : E_OUTOFMEMORY;
}

PUBLIC HRESULT WINAPI
D3D12SerializeVersionedRootSignature(
   const D3D12_VERSIONED_ROOT_SIGNATURE_DESC *desc, ID3DBlob **blob,
   ID3DBlob **error_blob)
{
   if (error_blob)
      *error_blob = NULL;
   if (!blob)
      return E_POINTER;

   *blob = NULL;

   void *data = NULL;
   size_t size = 0;
   HRESULT hr =
      d12_serialize_versioned_root_signature(desc, &data, &size);
   if (FAILED(hr))
      return hr;

   *blob = d12_blob_create(data, size);
   free(data);

   return *blob ? S_OK : E_OUTOFMEMORY;
}

PUBLIC HRESULT WINAPI
D3D12CreateRootSignatureDeserializer(const void *src, SIZE_T size,
                                     REFIID riid, void **out)
{
   return d12_root_signature_deserializer_create(src, size, riid, out);
}

PUBLIC HRESULT WINAPI
D3D12CreateVersionedRootSignatureDeserializer(const void *src, SIZE_T size,
                                              REFIID riid, void **out)
{
   return d12_versioned_root_signature_deserializer_create(src, size, riid,
                                                           out);
}

PUBLIC HRESULT WINAPI
D3D12GetInterface(REFCLSID rclsid, REFIID riid, void **out)
{
   d12_stub_report("D3D12GetInterface");
   if (out)
      *out = NULL;
   return E_NOINTERFACE;
}

PUBLIC HRESULT WINAPI
D3D12EnableExperimentalFeatures(UINT num_features, const IID *iids,
                                void *configs, UINT *configs_sizes)
{
   /* No experimental feature is implemented; enabling none is a success,
    * which is what runtimes probe before creating a device. */
   if (num_features == 0)
      return S_OK;
   d12_stub_report("D3D12EnableExperimentalFeatures");
   return E_NOINTERFACE;
}
