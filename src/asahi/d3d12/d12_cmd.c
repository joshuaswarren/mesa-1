/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * ID3D12CommandAllocator and ID3D12GraphicsCommandList.
 *
 * A command list is an ordered sequence of hardware segments (see
 * d12_private.h): CDM compute streams for copies/dispatches, and RENDER
 * segments for clears. Encoding happens at record time, mirroring
 * Honeykrisp's internal-dispatch (hk_dispatch_precomp) and bg/eot
 * (hk_build_bg_eot) recipes. No CPU-side execution lies anywhere: every
 * recorded operation becomes real GPU work at ExecuteCommandLists.
 */

#include "d12_g17p_txf_prefix_core.h"
#include "util/mesa-blake3.h"
#include "d12_private.h"
#include "d12_buffer_view_core.h"
#include "d12_agx3_cmdstream_core.h"
#include "d12_agx3_graphics_bind_core.h"
#include "asahi/lib/agx_vdm_hal200_direct_draw_core.h"
#include "d12_agx3_ppp_hal300_depth_bias_scissor_core.h"
#include "d12_agx3_ppp_hal300_depth_stencil_core.h"
#include "d12_agx3_ppp_viewport_payload_core.h"
#include "d12_agx3_ppp_w_clamp_cull2_core.h"
#include "d12_agx3_vdm_barrier_core.h"
#include "d12_agx3_vdm_stream_terminate_core.h"
#include "d12_command_validation_core.h"
#include "d12_discard_core.h"
#include "d12_present_format_core.h"
#include "d12_query_core.h"
#include "d12_resolve_core.h"
#include "d12_sparse_residency_core.h"
#include "d12_texture_copy_math.h"

#include "agx_helpers.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "asahi/compiler/agx3_graphics_program.h"
#include "agx_abi.h"
#include "agx_nir_texture.h"
#include "agx_usc.h"
#include "agx_vdm_state_update_core.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "present_blt.h"
#include "resolve.h"
#include "util/format/format_utils.h"
#include "util/format/u_format.h"
#include "util/u_math.h"

static bool d12_segment_ensure_viewport_arrays(struct d12_cmd_list *list,
                                                struct d12_segment *seg);

/* A legal ExecuteIndirect count call may retain thousands of software-
 * geometry placeholders until the count is known on the GPU.  Keep one
 * bounded command-list allocation large enough for that worst case; reset
 * still recycles subsequent recordings toward their measured high-water
 * mark in d12_list_take_fresh_storage(). */
#define D12_STREAM_BO_SIZE (32 * 1024 * 1024)
#define D12_USC_BO_SIZE    (32 * 1024 * 1024)
#define D12_RECYCLED_STREAM_MIN (128 * 1024)
#define D12_RECYCLED_USC_MIN    (64 * 1024)
/*
 * Control-stream space accounting. Every emission into the stream checks
 * for the worst case of what it is about to write, sized from the packet
 * lengths it uses, plus room for the terminator the close paths emit
 * without a check of their own. D12_DISPATCH_STREAM_MARGIN is the AGX3
 * dispatch sink capacity and the largest single emission; the data
 * uploads keep that much plus a terminator free between the data floor
 * and the stream so a check made before an upload still holds after it.
 */
#define D12_DISPATCH_STREAM_MARGIN 0x300

#define D12_STREAM_TERMINATE_BYTES                                          \
   MAX2(AGX_VDM_STREAM_TERMINATE_LENGTH, AGX_CDM_STREAM_TERMINATE_LENGTH)
#define D12_STREAM_RESERVE_BYTES                                            \
   (D12_DISPATCH_STREAM_MARGIN + D12_STREAM_TERMINATE_BYTES)

/* Legacy CDM dispatch: launch words, the G14X word, both grid forms (one
 * is emitted, size for either), then the cache-flush barrier. */
#define D12_CDM_DISPATCH_BYTES                                              \
   (AGX_CDM_LAUNCH_WORD_0_LENGTH + AGX_CDM_LAUNCH_WORD_1_LENGTH +           \
    AGX_CDM_UNK_G14X_LENGTH + AGX_CDM_INDIRECT_LENGTH +                     \
    AGX_CDM_GLOBAL_SIZE_LENGTH + AGX_CDM_LOCAL_SIZE_LENGTH +                \
    AGX_CDM_BARRIER_LENGTH)

/* Initial VDM barrier of a render segment, either generation. */
#define D12_VDM_BARRIER_BYTES                                               \
   MAX2(AGX_VDM_BARRIER_LENGTH, D12_AGX3_VDM_BARRIER_LENGTH)

/* One draw: restart state, vertex shader state (AGX2 words or the AGX3
 * state update), then every Index List word agx_vdm_draw can emit. */
#define D12_VDM_DRAW_BYTES                                                  \
   (2 * AGX_VDM_STATE_LENGTH + AGX_VDM_STATE_RESTART_INDEX_LENGTH +         \
    AGX_VDM_STATE_VERTEX_SHADER_WORD_1_LENGTH +                             \
    AGX_VDM_STATE_VERTEX_UNKNOWN_LENGTH + 4 +                               \
    D12_AGX3_VDM_STATE_UPDATE_LENGTH + AGX_INDEX_LIST_LENGTH +              \
    AGX_INDEX_LIST_BUFFER_LO_LENGTH + AGX_INDEX_LIST_INDIRECT_BUFFER_LENGTH + \
    AGX_INDEX_LIST_COUNT_LENGTH + AGX_INDEX_LIST_INSTANCES_LENGTH +         \
    AGX_INDEX_LIST_START_LENGTH + AGX_INDEX_LIST_BUFFER_SIZE_LENGTH)

static_assert(D12_CDM_DISPATCH_BYTES <= D12_DISPATCH_STREAM_MARGIN,
              "the upload reserve must cover a CDM dispatch");
static_assert(D12_VDM_DRAW_BYTES <= D12_DISPATCH_STREAM_MARGIN,
              "the upload reserve must cover a VDM draw");
static_assert(D12_VDM_BARRIER_BYTES <= D12_DISPATCH_STREAM_MARGIN,
              "the upload reserve must cover a VDM barrier");

/*
 * Append one entry to a bundle's command log (see struct d12_bundle_cmd).
 * Returns the zeroed header; the payload follows it. NULL, with the list
 * in error, when out of memory.
 */
static struct d12_bundle_cmd *
d12_bundle_record(struct d12_cmd_list *list, enum d12_bundle_op op,
                  size_t payload_size)
{
   size_t size = ALIGN_POT(sizeof(struct d12_bundle_cmd) + payload_size, 8);
   struct d12_bundle_cmd *cmd =
      util_dynarray_grow_bytes(&list->bundle_cmds, 1, size);
   if (!cmd) {
      mesa_loge("bundle: out of memory recording a command");
      list->state = D12_LIST_ERROR;
      list->record_error = E_OUTOFMEMORY;
      return NULL;
   }

   memset(cmd, 0, size);
   cmd->op = op;
   cmd->size_B = size;
   cmd->pso_set = list->pso != NULL;
   cmd->topology_set = list->bundle_topology_set;
   return cmd;
}

static inline void *
d12_bundle_payload(struct d12_bundle_cmd *cmd)
{
   return cmd + 1;
}

static void d12_list_replay_bundle(struct d12_cmd_list *list,
                                   struct d12_cmd_list *bundle);
/* Keep a single kernel command bounded even for command lists containing
 * hundreds of thousands of tiny meta dispatches. */
#define D12_MAX_DISPATCHES_PER_SEGMENT 8192

/* Default rendering stays closed. The registered noop-fd callback enables
 * metadata tests; a separate exact opt-in plus new-KMD capability enables only
 * the bounded physical baseline. No unsupported producer guard is bypassed. */
#ifndef _WIN32
extern bool asahi_noop_d3d12_graphics_qualification_allowed(int fd)
   __attribute__((weak, visibility("default")));
#endif

static bool
d12_g17p_baseline_graphics_enabled(const struct agx_device *dev)
{
#ifndef _WIN32
   if (asahi_noop_d3d12_graphics_qualification_allowed &&
       asahi_noop_d3d12_graphics_qualification_allowed(dev->fd))
      return true;
   /* Explicit sacrificial baseline opt-in; the controller must additionally
    * verify the matched kernel boot. The capability excludes the old KMD.
    * Every existing narrow producer/resource/draw guard remains mandatory. */
   return d12_debug()->physical_baseline && dev->chip == AGX_CHIP_G17P &&
          (dev->params.features & DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS) &&
          dev->g17p_render_context_bo;
#else
   (void)dev;
   return false;
#endif
}

bool
d12_g17p_depth_enabled(const struct agx_device *dev)
{
   if (!dev) return false;
#ifndef _WIN32
   if (d12_debug()->shim_depth_metadata && dev->chip == AGX_CHIP_G17P &&
       dev->g17p_render_context_bo &&
       asahi_noop_d3d12_graphics_qualification_allowed &&
       asahi_noop_d3d12_graphics_qualification_allowed(dev->fd))
      return true;
   /* Separate bounded physical admission. The controller must verify the
    * exact matched kernel image/boot before setting both flags. A flag alone
    * cannot admit an arbitrary fd: require an actual Asahi render node and
    * the native context plus kernel capability on this initialized device. */
   return d12_debug()->physical_d32 && d12_debug()->physical_baseline &&
          dev->chip == AGX_CHIP_G17P && dev->g17p_render_context_bo &&
          (dev->params.features & DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS) &&
          d12_platform_is_real_asahi_render_fd(dev->fd);
#else
   (void)dev;
   return false;
#endif
}

/* First native ZLS subset: D32 only, no compression, stencil, arrays/MSAA,
 * mip or read-only view. Bounds/stride are checked before UAPI construction. */
bool
d12_g17p_depth_segment_supported(const struct d12_segment *seg)
{
   const struct d12_resource *ds = seg ? seg->ds : NULL;
   const uint64_t limit = UINT64_C(1) << 48;
   if (!ds || !ds->bo || !ds->bo->va || !seg->rt[0] ||
       !ds->bo->va->addr || ds->bo->va->addr >= limit ||
       ds->bo->size > limit - ds->bo->va->addr ||
       ds->reserved || ds->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       ds->desc.Format != DXGI_FORMAT_D32_FLOAT ||
       !(ds->desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) ||
       ds->layout.format != PIPE_FORMAT_Z32_FLOAT || ds->stencil_bo ||
       ds->layout.compressed || ds->layout.levels != 1 ||
       !ds->layout.width_px || !ds->layout.height_px ||
       ds->layout.width_px > 16384 || ds->layout.height_px > 16384 ||
       ds->layout.depth_px != 1 || ds->layout.sample_count_sa != 1 ||
       (ds->layout.tiling != AIL_TILING_GPU &&
        ds->layout.tiling != AIL_TILING_TWIDDLED) ||
       seg->ds_level || seg->dsv_flags || seg->stencil_address ||
       seg->layers != 1 || !seg->width_px || !seg->height_px ||
       seg->width_px > 16384 || seg->height_px > 16384 ||
       seg->width_px > ds->layout.width_px ||
       seg->height_px > ds->layout.height_px ||
       !ds->layout.size_B || ds->layout.layer_stride_B < AIL_PAGESIZE ||
       ds->layout.layer_stride_B % AIL_PAGESIZE ||
       ds->layout.layer_stride_B / AIL_PAGESIZE >
          (UINT32_MAX >> 14) + UINT64_C(1) ||
       (seg->ds_address & (AIL_PAGESIZE - 1)) ||
       seg->ds_address < ds->bo->va->addr)
      return false;
   const uint64_t offset = seg->ds_address - ds->bo->va->addr;
   return offset <= ds->bo->size &&
          ds->layout.size_B <= ds->bo->size - offset &&
          ds->layout.layer_stride_B <= ds->bo->size - offset;
}

_Static_assert(offsetof(struct d12_cmd_list, iface) == 0,
               "Command-list COM interface must remain at object offset zero");

static bool d12_pack_texture_generation(
   enum agx_descriptor_generation generation, void *out,
   const struct AGX_TEXTURE *legacy, unsigned semantic_sample_count);
static bool d12_pack_pbe_generation(
   enum agx_descriptor_generation generation, void *out,
   const struct AGX_PBE *legacy);

/*
 * ID3D12CommandAllocator. The real allocations live on the list for now;
 * the allocator is lifetime bookkeeping to honor the API shape.
 */

static HRESULT STDMETHODCALLTYPE
d12_alloc_QueryInterface(ID3D12CommandAllocator *iface, REFIID riid,
                         void **out)
{
   struct d12_cmd_allocator *alloc = d12_cmd_allocator_from_iface(iface);

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D12Object) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceChild) ||
       d12_guid_equal(riid, &IID_ID3D12Pageable) ||
       d12_guid_equal(riid, &IID_ID3D12CommandAllocator)) {
      p_atomic_inc(&alloc->refcount);
      *out = iface;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&alloc->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12CommandAllocator::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_alloc_AddRef(ID3D12CommandAllocator *iface)
{
   struct d12_cmd_allocator *alloc = d12_cmd_allocator_from_iface(iface);
   return p_atomic_inc_return(&alloc->refcount);
}

/* Unreference every BO in a pool beyond `keep` entries. */
static void
d12_alloc_pool_drain(struct d12_cmd_allocator *alloc,
                     struct util_dynarray *pool, unsigned keep)
{
   while (util_dynarray_num_elements(pool, struct agx_bo *) > keep) {
      struct agx_bo *bo = util_dynarray_pop(pool, struct agx_bo *);
      agx_bo_unreference(&alloc->device->dev, bo);
   }
}

static void
d12_alloc_pool_recycle(struct d12_cmd_allocator *alloc,
                       struct util_dynarray *parked,
                       struct util_dynarray *free_pool)
{
   util_dynarray_foreach(parked, struct agx_bo *, bo)
      util_dynarray_append_typed(free_pool, struct agx_bo *, *bo);
   util_dynarray_clear(parked);
   d12_alloc_pool_drain(alloc, free_pool, D12_ALLOCATOR_POOL_MAX);
}

/* Take the first pooled BO large enough for `size`, or NULL. */
static struct agx_bo *
d12_alloc_pool_take(struct util_dynarray *pool, size_t size)
{
   struct agx_bo **bos = util_dynarray_begin(pool);
   const unsigned count = util_dynarray_num_elements(pool, struct agx_bo *);
   for (unsigned i = 0; i < count; i++) {
      if (bos[i]->size < size)
         continue;
      struct agx_bo *bo = bos[i];
      bos[i] = bos[count - 1];
      (void)util_dynarray_pop(pool, struct agx_bo *);
      return bo;
   }
   return NULL;
}

static ULONG STDMETHODCALLTYPE
d12_alloc_Release(ID3D12CommandAllocator *iface)
{
   struct d12_cmd_allocator *alloc = d12_cmd_allocator_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&alloc->refcount);

   if (refs == 0) {
      d12_alloc_pool_drain(alloc, &alloc->parked_stream_bos, 0);
      d12_alloc_pool_drain(alloc, &alloc->parked_usc_bos, 0);
      d12_alloc_pool_drain(alloc, &alloc->free_stream_bos, 0);
      d12_alloc_pool_drain(alloc, &alloc->free_usc_bos, 0);
      util_dynarray_fini(&alloc->parked_stream_bos);
      util_dynarray_fini(&alloc->parked_usc_bos);
      util_dynarray_fini(&alloc->free_stream_bos);
      util_dynarray_fini(&alloc->free_usc_bos);
      d12_private_data_store_destroy(&alloc->private_data);
      d12_device_release(alloc->device);
      free(alloc);
   }

   return refs;
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_alloc, ID3D12CommandAllocator,
                                d12_cmd_allocator_from_iface)

static HRESULT STDMETHODCALLTYPE
d12_alloc_GetDevice(ID3D12CommandAllocator *iface, REFIID riid, void **out)
{
   struct d12_cmd_allocator *alloc = d12_cmd_allocator_from_iface(iface);
   return ID3D12Device_QueryInterface(&alloc->device->iface, riid, out);
}

static HRESULT STDMETHODCALLTYPE
d12_alloc_Reset(ID3D12CommandAllocator *iface)
{
   struct d12_cmd_allocator *alloc = d12_cmd_allocator_from_iface(iface);

   /* D3D12 requires this to FAIL while any list built from the allocator
    * is still recording. Returning S_OK regardless hides a real
    * application bug and corrupts whatever that list goes on to emit.
    */
   if (alloc->recording_lists) {
      mesa_loge("CommandAllocator::Reset with %u list(s) still recording",
                alloc->recording_lists);
      return E_FAIL;
   }

   /* Every submission recorded from this allocator has completed (the
    * application's contract for calling Reset), so parked storage may be
    * recorded over again. */
   d12_alloc_pool_recycle(alloc, &alloc->parked_stream_bos,
                          &alloc->free_stream_bos);
   d12_alloc_pool_recycle(alloc, &alloc->parked_usc_bos,
                          &alloc->free_usc_bos);
   return S_OK;
}

static const ID3D12CommandAllocatorVtbl d12_alloc_vtbl = {
   /* every unimplemented slot fails safely */
   D12_ID3D12CommandAllocator_DEFAULTS,
   .QueryInterface = d12_alloc_QueryInterface,
   .AddRef = d12_alloc_AddRef,
   .Release = d12_alloc_Release,
   .GetPrivateData = d12_alloc_GetPrivateData,
   .SetPrivateData = d12_alloc_SetPrivateData,
   .SetPrivateDataInterface = d12_alloc_SetPrivateDataInterface,
   .SetName = d12_alloc_SetName,
   .GetDevice = d12_alloc_GetDevice,
   .Reset = d12_alloc_Reset,
};

HRESULT
d12_cmd_allocator_create(struct d12_device *device,
                         D3D12_COMMAND_LIST_TYPE type, REFIID riid,
                         void **out)
{
   if (!out)
      return E_POINTER;
   *out = NULL;

   switch (type) {
   case D3D12_COMMAND_LIST_TYPE_DIRECT:
   case D3D12_COMMAND_LIST_TYPE_BUNDLE:
   case D3D12_COMMAND_LIST_TYPE_COMPUTE:
   case D3D12_COMMAND_LIST_TYPE_COPY:
      break;
   default:
      mesa_loge("CreateCommandAllocator: invalid type %d", (int)type);
      return E_INVALIDARG;
   }

   struct d12_cmd_allocator *alloc = calloc(1, sizeof(*alloc));
   if (!alloc)
      return E_OUTOFMEMORY;

   alloc->iface.lpVtbl = (ID3D12CommandAllocatorVtbl *)&d12_alloc_vtbl;
   alloc->refcount = 1;
   alloc->device = device;
   alloc->type = type;
   d12_private_data_store_init(&alloc->private_data, (IUnknown *)&alloc->iface);
   util_dynarray_init(&alloc->parked_stream_bos, NULL);
   util_dynarray_init(&alloc->parked_usc_bos, NULL);
   util_dynarray_init(&alloc->free_stream_bos, NULL);
   util_dynarray_init(&alloc->free_usc_bos, NULL);
   d12_device_add_ref(device);

   HRESULT hr = d12_alloc_QueryInterface(&alloc->iface, riid, out);
   d12_alloc_Release(&alloc->iface);
   return hr;
}

/*
 * ID3D12GraphicsCommandList
 */

static uint8_t *
d12_list_stream_map(struct d12_cmd_list *list)
{
   return agx_bo_map(list->stream_bo);
}

enum d12_scratch_stage {
   D12_SCRATCH_CS,
   D12_SCRATCH_VS,
   D12_SCRATCH_FS,
};

/*
 * Reserve the device-wide scratch pool and retain the exact resulting BO in
 * the segment. agx_scratch_alloc may grow the pool by replacing its BO; a
 * segment reference keeps the old mapping alive if another command list grows
 * the pool after this one has been recorded.
 */
static bool
d12_segment_reserve_scratch(struct d12_cmd_list *list,
                            struct d12_segment *seg,
                            enum d12_scratch_stage stage,
                            const struct agx_shader_info *info)
{
   unsigned size = MAX2(info->scratch_size, info->preamble_scratch_size);
   if (!size)
      return true;

   struct d12_device *device = list->device;
   struct agx_scratch *scratch;
   struct agx_bo **segment_bo;
   bool *segment_preamble;

   switch (stage) {
   case D12_SCRATCH_CS:
      scratch = &device->scratch_cs;
      segment_bo = &seg->scratch_cs_bo;
      segment_preamble = &seg->scratch_cs_preamble;
      break;
   case D12_SCRATCH_VS:
      scratch = &device->scratch_vs;
      segment_bo = &seg->scratch_vs_bo;
      segment_preamble = &seg->scratch_vs_preamble;
      break;
   case D12_SCRATCH_FS:
      scratch = &device->scratch_fs;
      segment_bo = &seg->scratch_fs_bo;
      segment_preamble = &seg->scratch_fs_preamble;
      break;
   default:
      UNREACHABLE("invalid d3d12 scratch stage");
   }

   simple_mtx_lock(&device->scratch_lock);
   agx_scratch_alloc(scratch, size, 0);

   if (!scratch->buf) {
      simple_mtx_unlock(&device->scratch_lock);
      mesa_loge("failed to allocate %u dwords of shader scratch", size);
      list->state = D12_LIST_ERROR;
      return false;
   }

   if (*segment_bo != scratch->buf) {
      agx_bo_reference(scratch->buf);
      if (*segment_bo)
         agx_bo_unreference(&device->dev, *segment_bo);
      *segment_bo = scratch->buf;
   }
   *segment_preamble |= info->preamble_scratch_size > 0;
   simple_mtx_unlock(&device->scratch_lock);

   if (d12_debug()->dump_scratch)
      mesa_logi("scratch stage=%u size=%u preamble=%u bo=0x%llx (%zu B)",
                stage, info->scratch_size, info->preamble_scratch_size,
                (unsigned long long)(*segment_bo)->va->addr,
                (*segment_bo)->size);

   return true;
}

static void
d12_list_release_segment_scratch(struct d12_cmd_list *list)
{
   for (unsigned i = 0; i < list->segment_count; i++) {
      struct d12_segment *seg = &list->segments[i];
      struct agx_bo **bos[] = {
         &seg->scratch_cs_bo,
         &seg->scratch_vs_bo,
         &seg->scratch_fs_bo,
      };

      for (unsigned j = 0; j < ARRAY_SIZE(bos); j++) {
         if (!*bos[j])
            continue;

         agx_bo_unreference(&list->device->dev, *bos[j]);
         *bos[j] = NULL;
      }

      for (unsigned pipe = 0; pipe < D12_WRITE_PIPE_COUNT; pipe++) {
         struct d12_write_tracker *tracker = &seg->writes[pipe];
         if (tracker->initialized)
            util_dynarray_fini(&tracker->refs);
         memset(tracker, 0, sizeof(*tracker));
      }
   }
}

static struct d12_write_tracker *
d12_segment_write_tracker(struct d12_segment *seg,
                          enum d12_write_pipe pipe)
{
   assert(pipe < D12_WRITE_PIPE_COUNT);
   struct d12_write_tracker *tracker = &seg->writes[pipe];
   if (!tracker->initialized) {
      util_dynarray_init(&tracker->refs, NULL);
      tracker->initialized = true;
   }
   return tracker;
}

static void
d12_segment_track_write_range(struct d12_segment *seg,
                              enum d12_write_pipe pipe,
                              uint64_t address, uint64_t size)
{
   struct d12_write_tracker *tracker =
      d12_segment_write_tracker(seg, pipe);

   if (!address || (size && address > UINT64_MAX - size)) {
      tracker->incomplete = true;
      return;
   }

   util_dynarray_foreach(&tracker->refs, struct d12_write_ref, ref) {
      if (ref->kind == D12_WRITE_REF_RANGE &&
          ref->range.address == address && ref->range.size == size)
         return;
   }

   util_dynarray_append_typed(
      &tracker->refs, struct d12_write_ref,
      ((struct d12_write_ref){
         .kind = D12_WRITE_REF_RANGE,
         .range = {.address = address, .size = size},
      }));
}

static void
d12_segment_track_write_descriptor(struct d12_segment *seg,
                                   enum d12_write_pipe pipe,
                                   const struct d12_descriptor *descriptor)
{
   if (!descriptor)
      return;

   struct d12_write_tracker *tracker =
      d12_segment_write_tracker(seg, pipe);
   util_dynarray_foreach(&tracker->refs, struct d12_write_ref, ref) {
      if (ref->kind == D12_WRITE_REF_DESCRIPTOR &&
          ref->descriptor == descriptor)
         return;
   }

   util_dynarray_append_typed(
      &tracker->refs, struct d12_write_ref,
      ((struct d12_write_ref){
         .kind = D12_WRITE_REF_DESCRIPTOR,
         .descriptor = descriptor,
      }));
}

static void
d12_segment_track_write_resource(struct d12_segment *seg,
                                 enum d12_write_pipe pipe,
                                 struct d12_resource *resource)
{
   if (!resource)
      return;

   struct d12_write_tracker *tracker =
      d12_segment_write_tracker(seg, pipe);
   util_dynarray_foreach(&tracker->refs, struct d12_write_ref, ref) {
      if (ref->kind == D12_WRITE_REF_RESOURCE &&
          ref->resource == resource)
         return;
   }

   util_dynarray_append_typed(
      &tracker->refs, struct d12_write_ref,
      ((struct d12_write_ref){
         .kind = D12_WRITE_REF_RESOURCE,
         .resource = resource,
      }));
}

static void
d12_segment_track_write_heap(struct d12_segment *seg,
                             enum d12_write_pipe pipe,
                             const struct d12_descriptor_heap *heap,
                             uint32_t first, uint32_t count)
{
   struct d12_write_tracker *tracker =
      d12_segment_write_tracker(seg, pipe);
   if (!heap || first >= heap->desc.NumDescriptors) {
      tracker->incomplete = true;
      return;
   }

   /* Clamp rather than refuse: a sized range that runs past the bound heap
    * is the application's problem, and the descriptors that do exist are
    * still exactly the ones this submission can write. */
   if (count > heap->desc.NumDescriptors - first)
      count = heap->desc.NumDescriptors - first;

   util_dynarray_foreach(&tracker->refs, struct d12_write_ref, ref) {
      if (ref->kind == D12_WRITE_REF_HEAP && ref->heap.heap == heap &&
          ref->heap.first == first && ref->heap.count == count)
         return;
   }

   util_dynarray_append_typed(
      &tracker->refs, struct d12_write_ref,
      ((struct d12_write_ref){
         .kind = D12_WRITE_REF_HEAP,
         .heap = {.heap = heap, .first = first, .count = count},
      }));
}

static void
d12_segment_mark_write_set_incomplete(struct d12_segment *seg,
                                      enum d12_write_pipe pipe)
{
   d12_segment_write_tracker(seg, pipe)->incomplete = true;
}

static struct d12_read_tracker *
d12_list_read_tracker(struct d12_cmd_list *list)
{
   struct d12_read_tracker *tracker = &list->reads;
   if (!tracker->initialized) {
      util_dynarray_init(&tracker->refs, NULL);
      tracker->initialized = true;
   }
   return tracker;
}

static void
d12_list_track_read_bo(struct d12_cmd_list *list, struct agx_bo *bo)
{
   if (!bo || !bo->va || !bo->size) {
      d12_list_read_tracker(list)->incomplete = true;
      return;
   }

   struct d12_read_tracker *tracker = d12_list_read_tracker(list);
   util_dynarray_foreach(&tracker->refs, struct d12_read_ref, ref) {
      if (ref->kind == D12_READ_REF_BO && ref->bo == bo)
         return;
   }

   agx_bo_reference(bo);
   util_dynarray_append_typed(
      &tracker->refs, struct d12_read_ref,
      ((struct d12_read_ref){.kind = D12_READ_REF_BO, .bo = bo}));
}

static void
d12_list_track_read_range(struct d12_cmd_list *list, uint64_t address,
                          uint64_t size)
{
   if (!list->device->submit_read_set)
      return;
   struct d12_read_tracker *tracker = d12_list_read_tracker(list);
   if (!address || (size && address > UINT64_MAX - size)) {
      tracker->incomplete = true;
      return;
   }

   util_dynarray_foreach(&tracker->refs, struct d12_read_ref, ref) {
      if (ref->kind == D12_READ_REF_RANGE &&
          ref->range.address == address && ref->range.size == size)
         return;
   }

   util_dynarray_append_typed(
      &tracker->refs, struct d12_read_ref,
      ((struct d12_read_ref){
         .kind = D12_READ_REF_RANGE,
         .range = {.address = address, .size = size},
      }));
}

static void
d12_list_track_read_descriptor(struct d12_cmd_list *list,
                               const struct d12_descriptor *descriptor)
{
   if (!descriptor || !list->device->submit_read_set)
      return;

   struct d12_read_tracker *tracker = d12_list_read_tracker(list);
   util_dynarray_foreach(&tracker->refs, struct d12_read_ref, ref) {
      if (ref->kind == D12_READ_REF_DESCRIPTOR &&
          ref->descriptor == descriptor)
         return;
   }

   util_dynarray_append_typed(
      &tracker->refs, struct d12_read_ref,
      ((struct d12_read_ref){
         .kind = D12_READ_REF_DESCRIPTOR,
         .descriptor = descriptor,
      }));
}

static void
d12_list_track_read_heap(struct d12_cmd_list *list,
                         const struct d12_descriptor_heap *heap,
                         uint32_t first, uint32_t count, bool resources)
{
   if (!list->device->submit_read_set)
      return;
   struct d12_read_tracker *tracker = d12_list_read_tracker(list);
   if (!heap || first >= heap->desc.NumDescriptors) {
      tracker->incomplete = true;
      return;
   }

   /* Clamp rather than refuse; see d12_segment_track_write_heap. */
   if (count > heap->desc.NumDescriptors - first)
      count = heap->desc.NumDescriptors - first;

   util_dynarray_foreach(&tracker->refs, struct d12_read_ref, ref) {
      if (ref->kind == D12_READ_REF_HEAP && ref->heap.heap == heap &&
          ref->heap.first == first && ref->heap.count == count &&
          ref->heap.resources == resources)
         return;
   }

   util_dynarray_append_typed(
      &tracker->refs, struct d12_read_ref,
      ((struct d12_read_ref){
         .kind = D12_READ_REF_HEAP,
         .heap = {.heap = heap,
                  .first = first,
                  .count = count,
                  .resources = resources},
      }));
}

static void
d12_list_track_read_resource(struct d12_cmd_list *list,
                             struct d12_resource *resource)
{
   /* Application resources outlive their submissions by API contract; only
    * a submit that publishes a read set needs them listed. */
   if (!resource || !list->device->submit_read_set)
      return;

   if (resource->reserved) {
      struct d12_read_tracker *tracker = d12_list_read_tracker(list);
      util_dynarray_foreach(&tracker->refs, struct d12_read_ref, ref) {
         if (ref->kind == D12_READ_REF_RESOURCE &&
             ref->resource == resource)
            return;
      }
      util_dynarray_append_typed(
         &tracker->refs, struct d12_read_ref,
         ((struct d12_read_ref){
            .kind = D12_READ_REF_RESOURCE,
            .resource = resource,
         }));
      return;
   }

   if (!resource->bo) {
      d12_list_read_tracker(list)->incomplete = true;
      return;
   }

   d12_list_track_read_bo(list, resource->bo);
   if (resource->stencil_bo)
      d12_list_track_read_bo(list, resource->stencil_bo);
}

static void
d12_list_mark_read_set_incomplete(struct d12_cmd_list *list)
{
   d12_list_read_tracker(list)->incomplete = true;
}

static void
d12_list_read_tracker_reset(struct d12_cmd_list *list)
{
   struct d12_read_tracker *tracker = &list->reads;
   if (tracker->initialized) {
      util_dynarray_foreach(&tracker->refs, struct d12_read_ref, ref) {
         if (ref->kind == D12_READ_REF_BO && ref->bo)
            agx_bo_unreference(&list->device->dev, ref->bo);
      }
      util_dynarray_fini(&tracker->refs);
   }

   memset(tracker, 0, sizeof(*tracker));
   util_dynarray_init(&tracker->refs, NULL);
   tracker->initialized = true;
   list->tracked_pso = NULL;
}

static void
d12_list_read_tracker_fini(struct d12_cmd_list *list)
{
   struct d12_read_tracker *tracker = &list->reads;
   if (!tracker->initialized)
      return;

   util_dynarray_foreach(&tracker->refs, struct d12_read_ref, ref) {
      if (ref->kind == D12_READ_REF_BO && ref->bo)
         agx_bo_unreference(&list->device->dev, ref->bo);
   }
   util_dynarray_fini(&tracker->refs);
   memset(tracker, 0, sizeof(*tracker));
}

static void
d12_list_track_pso_reads(struct d12_cmd_list *list,
                         const struct d12_pipeline_state *pso)
{
   if (!pso || pso == list->tracked_pso)
      return;
   list->tracked_pso = pso;

   struct agx_bo *const bos[] = {
      pso->code_bo,
      pso->vs.linked.bo,
      pso->vs.code_bo,
      pso->fs.linked.bo,
      pso->fs.code_bo,
      pso->sw_vs_bo,
      pso->sw_tes_bo,
      pso->gs_count_bo,
      pso->gs_pre_bo,
      pso->gs_main_bo,
      pso->gs_rast.linked.bo,
      pso->gs_rast.code_bo,
      pso->tcs_bo,
      pso->tes.linked.bo,
      pso->tes.code_bo,
      pso->ms_bo,
      pso->ms_setup_bo,
      pso->ms_rast.linked.bo,
      pso->ms_rast.code_bo,
      pso->as_bo,
      pso->as_setup_bo,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(bos); ++i) {
      if (bos[i])
         d12_list_track_read_bo(list, bos[i]);
   }
}

static struct d12_dynamic_state_values
d12_list_dynamic_state(const struct d12_cmd_list *list)
{
   struct d12_dynamic_state_values state = {
      .stencil_ref_front = list->stencil_ref_front,
      .stencil_ref_back = list->stencil_ref_back,
      .depth_bias = list->depth_bias,
      .depth_bias_clamp = list->depth_bias_clamp,
      .slope_scaled_depth_bias = list->slope_scaled_depth_bias,
      .strip_cut = (uint32_t)list->strip_cut,
   };
   memcpy(state.blend_factor, list->blend_factor,
          sizeof(state.blend_factor));
   return state;
}

static void
d12_list_set_dynamic_state(struct d12_cmd_list *list,
                           const struct d12_dynamic_state_values *state)
{
   list->stencil_ref_front = state->stencil_ref_front;
   list->stencil_ref_back = state->stencil_ref_back;
   memcpy(list->blend_factor, state->blend_factor,
          sizeof(list->blend_factor));
   list->depth_bias = state->depth_bias;
   list->depth_bias_clamp = state->depth_bias_clamp;
   list->slope_scaled_depth_bias = state->slope_scaled_depth_bias;
   list->strip_cut = (D3D12_INDEX_BUFFER_STRIP_CUT_VALUE)state->strip_cut;
}

/* Reset the API-visible state, without rewinding command storage or removing
 * commands already recorded. This is shared by command-list Reset and
 * ClearState so neither path can leak bindings from a previous recording. */
static void
d12_list_reset_api_state(struct d12_cmd_list *list)
{
   const bool has_graphics_pso = list->pso && !list->pso->is_compute;
   const struct d12_dynamic_state_values dynamic = d12_dynamic_state_reset(
      has_graphics_pso,
      has_graphics_pso ? list->pso->raster.depth_bias : 0.0f,
      has_graphics_pso ? list->pso->raster.depth_bias_clamp : 0.0f,
      has_graphics_pso ? list->pso->raster.slope_scaled_depth_bias : 0.0f,
      has_graphics_pso ? (uint32_t)list->pso->strip_cut : 0);

   list->topology = MESA_PRIM_UNKNOWN;
   list->patch_control_points = 0;
   list->predicate_gpu = 0;
   list->predicate_op = D3D12_PREDICATION_OP_EQUAL_ZERO;
   memset(list->so, 0, sizeof(list->so));

   list->resource_descriptor_heap = NULL;
   list->sampler_descriptor_heap = NULL;
   list->compute_rs = NULL;
   list->graphics_rs = NULL;
   memset(list->compute_root_table, 0, sizeof(list->compute_root_table));
   memset(list->graphics_root_table, 0, sizeof(list->graphics_root_table));
   memset(list->compute_root_source, 0, sizeof(list->compute_root_source));
   memset(list->graphics_root_source, 0,
          sizeof(list->graphics_root_source));
   memset(list->compute_sampler_table, 0,
          sizeof(list->compute_sampler_table));
   memset(list->graphics_sampler_table, 0,
          sizeof(list->graphics_sampler_table));
   list->root_generation[0]++;
   list->root_generation[1]++;

   memset(list->vb, 0, sizeof(list->vb));
   memset(&list->ib, 0, sizeof(list->ib));
   memset(list->viewports, 0, sizeof(list->viewports));
   memset(list->scissors, 0, sizeof(list->scissors));
   list->viewport_count = 0;
   list->scissor_count = 0;
   memset(list->bound_rtv, 0, sizeof(list->bound_rtv));
   memset(&list->bound_dsv, 0, sizeof(list->bound_dsv));
   list->bound_rtv_count = 0;
   list->has_bound_dsv = false;
   d12_list_set_dynamic_state(list, &dynamic);
}

static void
d12_list_begin_recording(struct d12_cmd_list *list)
{
   d12_list_release_segment_scratch(list);
   d12_list_read_tracker_reset(list);
   d12_list_g17p_vdm_storage_reset(list);
   list->stream_cur = d12_list_stream_map(list);
   list->data_floor = list->stream_bo->size;
   list->usc_offset = 0;
   list->segment_count = 0;
   list->cdm_open = false;
   list->render_open = false;
   list->pregfx_open = false;
   list->pregfx_cur = NULL;
   d12_list_reset_api_state(list);

   list->uses_heap = false;
   list->active_occlusion_queries = 0;
   list->active_pipeline_stats_heap = NULL;
   memset(&list->pipeline_stats, 0, sizeof(list->pipeline_stats));
   memset(list->active_so_stats_heap, 0,
          sizeof(list->active_so_stats_heap));

   /* The real free is the GPU write Close appends (see there): a closed
    * list can be executed many times without ever being Reset, so a reset
    * hook cannot be the only thing that frees the heap. This CPU zero
    * stays as a floor for the first recording, when nothing has been
    * submitted yet and no GPU write is in flight to race with.
    */
   if (list->device->heap_desc_cpu)
      list->device->heap_desc_cpu->bottom = 0;

   list->state = D12_LIST_RECORDING;

   util_dynarray_clear(&list->descriptor_fixups);
   util_dynarray_clear(&list->indirect_constant_fixups);
   util_dynarray_clear(&list->indirect_argument_fixups);
   util_dynarray_clear(&list->bundle_cmds);
   /* Invalidate rather than memset: the caches are several hundred KiB. */
   for (unsigned t = 0; t < 2; t++) {
      for (unsigned w = 0; w < D12_BINDING_CACHE_WAYS; w++)
         list->binding_upload_cache[t][w].valid = false;
      list->binding_cache_next[t] = 0;
   }
   for (unsigned i = 0; i < D12_GFX_UPLOAD_CACHE_SIZE; i++)
      list->gfx_upload_cache[i].valid = false;
   for (unsigned i = 0; i < D12_FRAGMENT_STATE_CACHE_SIZE; i++)
      list->fragment_state_cache[i].valid = false;
   list->direct_draw_params_valid = false;
   list->direct_draw_params_gpu = 0;
   list->blend_upload_valid = false;
   list->blend_upload_gpu = 0;
   list->bundle_topology_set = false;
   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE && list->pso) {
      /* The initial pipeline state is the bundle's first command. */
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_SET_PIPELINE_STATE, 0);
      if (cmd)
         cmd->pso = &list->pso->iface;
   }

   /* Points into the stream BO, whose data floor has just been rewound, so
    * the old address is about to be handed out to somebody else.
    */
   list->null_buffer = 0;
   list->discard_buffer = 0;
   list->null_counters = 0;
   list->sparse_registry_view_gpu = 0;
   list->uses_sparse_registry = false;
}

/* Upload a data blob into the top of the stream BO; returns its GPU VA. */
static uint64_t
d12_list_upload(struct d12_cmd_list *list, const void *data, size_t size,
                size_t align)
{
   /*
    * Check BEFORE subtracting. data_floor and size are size_t, so an
    * oversized upload makes `data_floor - size` wrap to a value near
    * SIZE_MAX -- which then sails past the `floor <= used + MARGIN` guard
    * (a huge floor looks like plenty of room) and memcpys far outside the
    * BO. The guard was load-bearing and could be skipped by the very case
    * it existed to catch.
    */
   if (size > list->data_floor) {
      mesa_loge("command list upload of %zu bytes exceeds data space",
                size);
      list->state = D12_LIST_ERROR;
      return 0;
   }

   size_t floor = (list->data_floor - size) & ~(align - 1);
   size_t used = list->stream_cur - d12_list_stream_map(list);

   if (floor <= used + D12_STREAM_RESERVE_BYTES) {
      mesa_loge("command list out of stream/data space: control=%zu "
                "data=%zu capacity=%zu request=%zu usc=%zu/%zu",
                used, list->stream_bo->size - list->data_floor,
                list->stream_bo->size, size, list->usc_offset,
                list->usc_bo->size);
      list->state = D12_LIST_ERROR;
      return 0;
   }

   memcpy(d12_list_stream_map(list) + floor, data, size);
   list->data_floor = floor;
   return list->stream_bo->va->addr + floor;
}

/* Reserve zeroed data space; returns its GPU address. */
static uint64_t
d12_list_upload_zero(struct d12_cmd_list *list, size_t size, size_t align)
{
   /* Same underflow as d12_list_upload -- check before subtracting. */
   if (size > list->data_floor) {
      mesa_loge("command list zero-reserve of %zu bytes exceeds data space",
                size);
      list->state = D12_LIST_ERROR;
      return 0;
   }

   size_t floor = (list->data_floor - size) & ~(align - 1);
   size_t used = list->stream_cur - d12_list_stream_map(list);

   if (floor <= used + D12_STREAM_RESERVE_BYTES) {
      mesa_loge("command list out of stream/data space: control=%zu "
                "data=%zu capacity=%zu request=%zu usc=%zu/%zu",
                used, list->stream_bo->size - list->data_floor,
                list->stream_bo->size, size, list->usc_offset,
                list->usc_bo->size);
      list->state = D12_LIST_ERROR;
      return 0;
   }

   memset(d12_list_stream_map(list) + floor, 0, size);
   list->data_floor = floor;
   return list->stream_bo->va->addr + floor;
}

/*
 * Zeroed memory that an unbound root slot can point at instead of NULL,
 * allocated once per recording and shared by every slot that needs it.
 *
 * 64 KiB is D3D12's own maximum constant-buffer size, so for a null CBV the
 * emulation is exact: every offset the API can legally address is covered.
 * A structured SRV/UAV indexed far enough past the end can still walk off
 * it -- covering that needs per-access bounds checking, not a bigger buffer
 * -- but it turns the common case from a guaranteed queue timeout into a
 * spec-correct read of zero.
 */
static uint64_t
d12_list_null_buffer(struct d12_cmd_list *list)
{
   if (!list->null_buffer)
      list->null_buffer = d12_list_upload_zero(list, 64 * 1024, 16);

   return list->null_buffer;
}

static uint64_t
d12_list_null_counters(struct d12_cmd_list *list)
{
   if (!list->null_counters) {
      list->null_counters = d12_list_upload_zero(
         list, D12_MAX_RESOURCE_BINDINGS * sizeof(uint32_t), 16);
   }

   return list->null_counters;
}

static uint64_t
d12_list_discard_buffer(struct d12_cmd_list *list)
{
   if (!list->discard_buffer)
      list->discard_buffer = d12_list_upload_zero(list, 16, 16);

   return list->discard_buffer;
}

/* CPU pointer for a GPU address previously returned by an upload. */
static uint8_t *
d12_list_data_cpu(struct d12_cmd_list *list, uint64_t gpu)
{
   if (!gpu)
      return NULL;
   return d12_list_stream_map(list) + (gpu - list->stream_bo->va->addr);
}

static bool
d12_list_stream_contains(const struct d12_cmd_list *list, uint64_t gpu,
                         size_t size)
{
   uint64_t base = list->stream_bo->va->addr;
   return gpu >= base && size <= list->stream_bo->size &&
          gpu - base <= list->stream_bo->size - size;
}

/* Allocate USC words; returns CPU pointer and the usc-space address. */
static uint8_t *
d12_list_alloc_usc(struct d12_cmd_list *list, size_t size, uint32_t *out_usc)
{
   size_t off = ALIGN_POT(list->usc_offset, 64);
   if (size > list->usc_bo->size || off > list->usc_bo->size - size) {
      mesa_loge("command list out of USC space");
      list->state = D12_LIST_ERROR;
      return NULL;
   }
   list->usc_offset = off + size;

   *out_usc =
      agx_usc_addr(&list->device->dev, list->usc_bo->va->addr + off);
   return (uint8_t *)agx_bo_map(list->usc_bo) + off;
}

/*
 * Room for `need` bytes of control stream plus the terminator the close
 * path will append. `need` is the caller's worst case for what it emits
 * next, never larger than the upload reserve (asserted) so that uploads
 * made between this check and the emission cannot invalidate it.
 */
static bool
d12_list_check_stream_space(struct d12_cmd_list *list, size_t need)
{
   assert(need <= D12_DISPATCH_STREAM_MARGIN);
   size_t used = list->stream_cur - d12_list_stream_map(list);
   if (used + need + D12_STREAM_TERMINATE_BYTES > list->data_floor) {
      mesa_loge("command list out of control-stream space (%zu + %zu)",
                used, need);
      list->state = D12_LIST_ERROR;
      return false;
   }
   return true;
}

/* The same for the pre-graphics stream, which is closed with a CDM
 * terminator of its own. */
static bool
d12_list_check_pregfx_space(struct d12_cmd_list *list, size_t need)
{
   if (list->pregfx_cur + need + D12_STREAM_TERMINATE_BYTES >
       list->pregfx_end) {
      mesa_loge("pre-graphics stream full (%zu needed)", need);
      list->state = D12_LIST_ERROR;
      return false;
   }
   return true;
}

/* Begin a graphics stream with the cache-invalidating barrier selected by
 * the physical command-stream generation.  The AGX2 packet is intentionally
 * kept byte-for-byte equivalent to the old direct pack, while AGX3 consumes
 * the four-byte CommandEncodingNext witness shared with Honeykrisp. */
static bool
d12_list_emit_initial_vdm_barrier(struct d12_cmd_list *list)
{
   switch (agx_chip_command_stream_generation(list->device->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      agx_pack(list->stream_cur, VDM_BARRIER, cfg) {
         cfg.usc_cache_inval = true;
      }
      list->stream_cur += AGX_VDM_BARRIER_LENGTH;
      return true;

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      struct d12_agx3_vdm_barrier barrier;
      if (!d12_agx3_encode_vdm_barrier(list->device->dev.chip, &barrier))
         break;

      memcpy(list->stream_cur, barrier.bytes, barrier.length);
      list->stream_cur += barrier.length;
      return true;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      break;
   }

   mesa_loge("no grounded initial VDM barrier for %s",
             agx_chip_name(list->device->dev.chip));
   list->state = D12_LIST_ERROR;
   return false;
}

/* Emit the conservative post-dispatch cache flush in the command-stream ISA
 * selected by the physical chip.  CommandEncodingNext emits flags=5 as its
 * strong form on every grounded G15--G17 identity; using it here preserves
 * the existing flush-after-every-launch policy without feeding an AGX2 packet
 * to an AGX3 parser. */
static bool
d12_list_emit_cdm_cache_flush(struct d12_cmd_list *list, uint8_t **cursor)
{
   switch (agx_chip_command_stream_generation(list->device->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      *cursor = (uint8_t *)agx_cdm_barrier((uint32_t *)*cursor,
                                           list->device->dev.chip);
      return true;

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      const struct agx3_cdm_barrier_info info = {
         .chip = list->device->dev.chip,
         .flags = 0x5,
      };
      struct d12_agx3_cdm_barrier_encoding encoded;
      if (d12_agx3_encode_cdm_barrier(&info, &encoded)) {
         memcpy(*cursor, encoded.bytes, encoded.length);
         *cursor += encoded.length;
         return true;
      }
      break;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      break;
   }

   mesa_loge("no grounded CDM cache flush for %s",
             agx_chip_name(list->device->dev.chip));
   list->state = D12_LIST_ERROR;
   return false;
}

/*
 * Direct3D orders two dispatches' memory accesses only across a barrier
 * (a UAV barrier between dependent dispatches, a transition before the
 * data is read by anything else), so an API dispatch owes its cache flush
 * to the next barrier rather than to the next launch: a run of
 * independent dispatches is one flush instead of one per launch, which
 * on G13 costs about 5 us of GPU time each. The flush is emitted at every
 * point whose consumer the driver cannot see the barrier for: an API
 * ResourceBarrier, a driver-generated launch (copies, clears, queries,
 * indirect patching, which the application never barriers against), and
 * the segment's terminator, so a hardware command always ends flushed and
 * the next command's barrier field covers the rest. The native G17P
 * dispatch record carries its own barrier and is unchanged.
 */
static bool
d12_list_flush_cdm_pending(struct d12_cmd_list *list)
{
   if (!list->cdm_open)
      return true;
   struct d12_segment *seg = &list->segments[list->segment_count - 1];
   if (!seg->cdm_flush_pending)
      return true;
   /* The dispatch reserved these bytes with its own space check. */
   if (!d12_list_check_stream_space(list, AGX_CDM_BARRIER_LENGTH))
      return false;
   if (!d12_list_emit_cdm_cache_flush(list, &list->stream_cur))
      return false;
   seg->cdm_flush_pending = false;
   return true;
}

static bool
d12_list_require_legacy_compute(struct d12_cmd_list *list)
{
   if (agx_chip_command_stream_generation(list->device->dev.chip) !=
       AGX_CS_GENERATION_AGX2) {
      mesa_loge("legacy USC compute launch rejected for %s",
                agx_chip_name(list->device->dev.chip));
      list->state = D12_LIST_ERROR;
      return false;
   }

   return true;
}

/* The existing launch inputs describe AGX2 USC state.  Keep every caller on
 * one generation gate so a latent AGX3 device can never receive a legacy CDM
 * packet while the dispatch-local Ioto program path is being brought up. */
static bool
d12_list_emit_legacy_cdm_launch(
   struct d12_cmd_list *list, uint8_t **cursor, struct agx_grid grid,
   struct agx_workgroup workgroup,
   struct agx_cdm_launch_word_0_packed launch, uint32_t usc)
{
   if (!d12_list_require_legacy_compute(list))
      return false;

   *cursor = (uint8_t *)agx_cdm_launch(
      (uint32_t *)*cursor, list->device->dev.chip, grid, workgroup, launch, usc);
   return true;
}

static void d12_list_close_pregfx(struct d12_cmd_list *list);

static bool
d12_list_emit_cdm_terminate(struct d12_cmd_list *list)
{
   switch (agx_chip_command_stream_generation(list->device->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      list->stream_cur =
         (uint8_t *)agx_cdm_terminate((uint32_t *)list->stream_cur);
      return true;

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      struct d12_agx3_cdm_stream_terminate_encoding terminate;
      if (!d12_agx3_encode_cdm_stream_terminate(list->device->dev.chip,
                                                &terminate))
         break;

      memcpy(list->stream_cur, terminate.bytes, terminate.length);
      list->stream_cur += terminate.length;
      return true;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      break;
   }

   mesa_loge("no grounded CDM stream terminator for %s",
             agx_chip_name(list->device->dev.chip));
   list->state = D12_LIST_ERROR;
   return false;
}

static bool
d12_list_emit_vdm_terminate(struct d12_cmd_list *list)
{
   switch (agx_chip_command_stream_generation(list->device->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      list->stream_cur =
         (uint8_t *)agx_vdm_terminate((uint32_t *)list->stream_cur);
      return true;

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      struct d12_agx3_vdm_stream_terminate terminate;
      if (!d12_agx3_encode_vdm_stream_terminate(list->device->dev.chip,
                                                &terminate))
         break;

      memcpy(list->stream_cur, terminate.bytes, terminate.length);
      list->stream_cur += terminate.length;
      return true;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      break;
   }

   mesa_loge("no grounded VDM stream terminator for %s",
             agx_chip_name(list->device->dev.chip));
   list->state = D12_LIST_ERROR;
   return false;
}

/* Close whichever segment is open, terminating its control stream. */
static void
d12_list_close_cdm(struct d12_cmd_list *list)
{
   if (list->cdm_open) {
      struct d12_segment *seg = &list->segments[list->segment_count - 1];
      d12_list_flush_cdm_pending(list);
      d12_list_emit_cdm_terminate(list);
      seg->stream_end = list->stream_cur - d12_list_stream_map(list);
      list->cdm_open = false;
   }

   if (list->render_open) {
      /* The pre-graphics stream belongs to this pass; it is meaningless
       * once the pass it precedes has ended.
       */
      d12_list_close_pregfx(list);

      struct d12_segment *seg = &list->segments[list->segment_count - 1];
      d12_list_emit_vdm_terminate(list);
      seg->stream_end = list->stream_cur - d12_list_stream_map(list);
      if (list->device->dev.chip == AGX_CHIP_G17P &&
          d12_g17p_baseline_graphics_enabled(&list->device->dev) && list->state != D12_LIST_ERROR) {
         const size_t size = seg->stream_end - seg->stream_start;
         uint64_t compact;
         void *copy = d12_list_alloc_g17p_vdm(list, size, 64, &compact);
         if (copy) {
            memcpy(copy, d12_list_stream_map(list) + seg->stream_start, size);
            seg->g17p_vdm_control_base = AGX_G17P_VDM_USER_BASE + compact;
            if (agx_bo_sync_for_device(list->stream_bo, 0, list->stream_bo->size))
               list->state = D12_LIST_ERROR;
         }
      }
      list->render_open = false;
   }
}

/* Size of one render pass's pre-graphics compute stream. ExecuteIndirect
 * count-buffer calls retain zero-count placeholders up to MaxCommandCount so
 * their GPU-late count remains correct. The conformance state test legally
 * records 2,052 software-geometry draws in one pass; allow that bounded case
 * within the bounded command-list BO.
 */
#define D12_PREGFX_STREAM_SIZE (512 * 1024)

/*
 * Terminate the pre-graphics stream, if one is open. Called when the
 * render pass it belongs to ends, because the pair is what makes the
 * ordering meaningful: the compute segment sits immediately before the
 * render segment and nothing may be appended to it afterwards.
 */
static void
d12_list_close_pregfx(struct d12_cmd_list *list)
{
   if (!list->pregfx_open)
      return;

   struct d12_segment *seg = &list->segments[list->pregfx_seg];
   uint8_t *saved_stream_cur = list->stream_cur;
   list->stream_cur = list->pregfx_cur;
   d12_list_emit_cdm_terminate(list);
   seg->stream_end = list->stream_cur - d12_list_stream_map(list);
   list->stream_cur = saved_stream_cur;
   list->pregfx_open = false;
   list->pregfx_cur = NULL;
}

/*
 * Get the pre-graphics compute segment for the open render pass, opening
 * it on first use. Unlike d12_list_get_cdm this does NOT end the render
 * pass -- that is the entire point. See the comment on pregfx_cur in
 * d12_private.h.
 *
 * The segment's slot was reserved immediately before the render segment
 * when the render pass began, so opening it here moves nothing: a
 * `struct d12_segment *` taken earlier in the draw (the render segment
 * with its sampler heap and scratch already bound) stays valid across
 * this call. Inserting the slot here instead used to shift the render
 * segment up by one and silently retarget every such pointer at the new
 * compute segment.
 *
 * Returns NULL if there is no open render pass to precede, which is a
 * caller bug rather than a resource limit: without a render pass the
 * ordinary CDM path is correct and cheaper.
 */
static struct d12_segment *
d12_list_get_pregfx(struct d12_cmd_list *list)
{
   if (list->pregfx_open)
      return &list->segments[list->pregfx_seg];

   if (!list->render_open || list->segment_count < 2) {
      mesa_loge("pre-graphics compute requested with no open render pass");
      list->state = D12_LIST_ERROR;
      return NULL;
   }

   /* Carve the stream out of the data area. It is a control stream living
    * in data space, which the submit path does not care about: it reads
    * stream_start/stream_end and nothing else.
    */
   uint64_t gpu = d12_list_upload_zero(list, D12_PREGFX_STREAM_SIZE, 64);
   if (!gpu)
      return NULL;

   size_t offset = gpu - list->stream_bo->va->addr;

   /* The reserved prelude: an empty CDM segment, which the submit path
    * skips while it has no dispatches. */
   unsigned at = list->segment_count - 2;
   struct d12_segment *seg = &list->segments[at];
   if (seg->type != D12_SEG_CDM || seg->dispatches || seg->stream_end) {
      mesa_loge("render pass has no reserved pre-graphics segment");
      list->state = D12_LIST_ERROR;
      return NULL;
   }
   seg->stream_start = offset;

   list->pregfx_cur = d12_list_stream_map(list) + offset;
   list->pregfx_end = list->pregfx_cur + D12_PREGFX_STREAM_SIZE;
   list->pregfx_seg = at;
   list->pregfx_open = true;
   return seg;
}

/* Get an open CDM segment to append dispatches to, starting one if needed. */
static struct d12_segment *
d12_list_get_cdm(struct d12_cmd_list *list)
{
   if (list->cdm_open)
      return &list->segments[list->segment_count - 1];

   /* A render pass may still be open; compute work ends it. */
   if (list->render_open)
      d12_list_close_cdm(list);

   if (list->segment_count >= D12_MAX_SEGMENTS) {
      mesa_loge("command list segment limit reached");
      list->state = D12_LIST_ERROR;
      return NULL;
   }

   struct d12_segment *seg = &list->segments[list->segment_count++];
   memset(seg, 0, sizeof(*seg));
   seg->type = D12_SEG_CDM;
   seg->stream_start = list->stream_cur - d12_list_stream_map(list);
   list->cdm_open = true;
   return seg;
}

/* All D3D12 libagx argument blocks begin with a GPU pointer.  For copy,
 * fill, resolve, immediate-write and indirect-patch programs it is the
 * destination; for the poly/tess helpers it is an internal parameter block.
 * Treating the latter as writable is a valid conservative superset and
 * normally coalesces into the command-list stream BO at submit time. */
static void
d12_segment_track_precompiled_write(struct d12_segment *seg,
                                    enum libagx_program idx,
                                    const void *args, size_t args_size)
{
   if (args_size >= sizeof(uint64_t)) {
      uint64_t first;
      memcpy(&first, args, sizeof(first));
      if (first)
         d12_segment_track_write_range(seg, D12_WRITE_COMPUTE, first, 0);
   }

   /* These query helpers put their destination after one or two read-only
    * pointers instead of in the common first slot. */
   if (idx == LIBAGX_COPY_QUERY_GL &&
       args_size == sizeof(struct libagx_copy_query_gl_args)) {
      const struct libagx_copy_query_gl_args *query = args;
      d12_segment_track_write_range(seg, D12_WRITE_COMPUTE, query->dest, 0);
   } else if (idx == LIBAGX_QUERY_DELTA &&
              args_size == sizeof(struct libagx_query_delta_args)) {
      const struct libagx_query_delta_args *query = args;
      d12_segment_track_write_range(seg, D12_WRITE_COMPUTE, query->result,
                                    sizeof(uint64_t));
   }
}

/* Decode the stable pointer fields of libagx kernels used for ordinary
 * copies, clears, resolves, queries and predication. These programs are
 * common enough that forcing the whole-live-BO fallback for every invocation
 * would erase the read-set optimization on copy queues and clear-heavy
 * frames. Programs with GPU-generated or evolving pointer ABIs deliberately
 * return false and retain the conservative path. */
static bool
d12_list_track_precompiled_reads(struct d12_cmd_list *list,
                                 enum libagx_program idx, const void *args,
                                 size_t args_size)
{
#define D12_PRECOMP_ARGS(type)                                                \
   const struct type *a = args;                                               \
   if (args_size != sizeof(*a))                                               \
      return false
#define D12_TRACK_ADDRESS(field)                                              \
   do {                                                                       \
      if (a->field)                                                           \
         d12_list_track_read_range(list, a->field, 0);                        \
   } while (0)

   switch (idx) {
   case LIBAGX_COPY_UCHAR: {
      D12_PRECOMP_ARGS(libagx_copy_uchar_args);
      D12_TRACK_ADDRESS(dest);
      D12_TRACK_ADDRESS(src);
      return true;
   }
   case LIBAGX_WRITE_U32: {
      D12_PRECOMP_ARGS(libagx_write_u32_args);
      if (a->address)
         d12_list_track_read_range(list, a->address, sizeof(uint32_t));
      return true;
   }
   case LIBAGX_FILL: {
      D12_PRECOMP_ARGS(libagx_fill_args);
      D12_TRACK_ADDRESS(address);
      return true;
   }
   case LIBAGX_FILL_IMAGE: {
      D12_PRECOMP_ARGS(libagx_fill_image_args);
      D12_TRACK_ADDRESS(dest);
      return true;
   }
   case LIBAGX_COPY_IMAGE: {
      D12_PRECOMP_ARGS(libagx_copy_image_args);
      D12_TRACK_ADDRESS(dest);
      D12_TRACK_ADDRESS(src);
      return true;
   }
   case LIBAGX_PRESENT_BLT: {
      D12_PRECOMP_ARGS(libagx_present_blt_args);
      D12_TRACK_ADDRESS(dest);
      D12_TRACK_ADDRESS(src);
      return true;
   }
   case LIBAGX_COPY_QUERY_GL: {
      D12_PRECOMP_ARGS(libagx_copy_query_gl_args);
      D12_TRACK_ADDRESS(query);
      D12_TRACK_ADDRESS(dest);
      return true;
   }
   case LIBAGX_QUERY_DELTA: {
      D12_PRECOMP_ARGS(libagx_query_delta_args);
      D12_TRACK_ADDRESS(counter);
      D12_TRACK_ADDRESS(start);
      D12_TRACK_ADDRESS(result);
      return true;
   }
   case LIBAGX_D3D12_PREDICATE: {
      D12_PRECOMP_ARGS(libagx_d3d12_predicate_args);
      D12_TRACK_ADDRESS(out);
      D12_TRACK_ADDRESS(in);
      D12_TRACK_ADDRESS(predicate);
      return true;
   }
   case LIBAGX_D3D12_INDIRECT_PATCH: {
      D12_PRECOMP_ARGS(libagx_d3d12_indirect_patch_args);
      D12_TRACK_ADDRESS(out);
      D12_TRACK_ADDRESS(in);
      D12_TRACK_ADDRESS(draw_count);
      return true;
   }
   case LIBAGX_D3D12_INDIRECT_CONSTANT: {
      D12_PRECOMP_ARGS(libagx_d3d12_indirect_constant_args);
      D12_TRACK_ADDRESS(out);
      D12_TRACK_ADDRESS(previous);
      D12_TRACK_ADDRESS(in);
      D12_TRACK_ADDRESS(draw_count);
      return true;
   }
   case LIBAGX_D3D12_INDIRECT_DISPATCH: {
      D12_PRECOMP_ARGS(libagx_d3d12_indirect_dispatch_args);
      D12_TRACK_ADDRESS(out);
      D12_TRACK_ADDRESS(in);
      return true;
   }
   case LIBAGX_RESOLVE_FLOAT32_MIN:
   case LIBAGX_RESOLVE_FLOAT32_AVERAGE:
   case LIBAGX_RESOLVE_UNORM8X4:
   case LIBAGX_RESOLVE_FLOAT32X4: {
      D12_PRECOMP_ARGS(libagx_resolve_unorm8x4_args);
      D12_TRACK_ADDRESS(dest);
      D12_TRACK_ADDRESS(src);
      return true;
   }
   case LIBAGX_RESOLVE_UNORM8X4_TYPED: {
      D12_PRECOMP_ARGS(libagx_resolve_unorm8x4_typed_args);
      D12_TRACK_ADDRESS(dest);
      D12_TRACK_ADDRESS(src);
      return true;
   }
   case LIBAGX_RESOLVE_SCALAR_EXTREMA: {
      D12_PRECOMP_ARGS(libagx_resolve_scalar_extrema_args);
      D12_TRACK_ADDRESS(dest);
      D12_TRACK_ADDRESS(src);
      return true;
   }
   case LIBAGX_RESOLVE_TYPED: {
      D12_PRECOMP_ARGS(libagx_resolve_typed_args);
      D12_TRACK_ADDRESS(dest);
      D12_TRACK_ADDRESS(src);
      return true;
   }
   default:
      return false;
   }

#undef D12_TRACK_ADDRESS
#undef D12_PRECOMP_ARGS
}

/* Native direct precompiled compute uses the same metadata/argument/Ioto
 * join as Gallium. Each recording owns its uploaded arguments and program;
 * the existing command-list storage and read-set references retain them
 * until the actual submitted timeline point retires. */
static bool
d12_list_append_g17p_precomp(struct d12_cmd_list *list,
                            const struct agx_precompiled_shader *prog,
                            const void *args, size_t args_size,
                            struct agx_grid grid, uint8_t **cursor)
{
   struct agx_device *dev = &list->device->dev;
   if (dev->chip != AGX_CHIP_G17P || !prog || !prog->bo || !args ||
       !args_size || args_size > UINT16_MAX * 2u || agx_is_indirect(grid) ||
       !prog->info.main_size || !prog->info.nr_gprs ||
       prog->info.nr_preamble_gprs || !prog->info.agx3_staging_known ||
       (prog->info.rodata.size_16 && !prog->rodata_offset) ||
       (prog->info.printf_uniform &&
        (prog->info.printf_uniform < AGX_ABI_AGX3_PRECOMP_ARG_HALFS ||
         !dev->printf_buffer_address)) ||
       prog->entry_offset > UINT64_MAX - prog->ptr ||
       prog->rodata_offset > UINT64_MAX - prog->ptr ||
       !prog->b.workgroup.x || !prog->b.workgroup.y || !prog->b.workgroup.z)
      goto fail;

   const size_t printf_offset = prog->info.printf_uniform
      ? (size_t)(prog->info.printf_uniform - AGX_ABI_AGX3_PRECOMP_ARG_HALFS) * 2
      : 0;
   const size_t uploaded_size = ALIGN_POT(
      prog->info.printf_uniform ? MAX2(args_size, printf_offset + 8) : args_size,
      4);
   if (uploaded_size / 2 > UINT16_MAX)
      goto fail;
   const uint64_t data_gpu = d12_list_upload_zero(list, uploaded_size, 4);
   if (!data_gpu)
      return false;
   uint8_t *data_cpu = d12_list_stream_map(list) +
      (data_gpu - list->stream_bo->va->addr);
   memcpy(data_cpu, args, args_size);
   if (prog->info.printf_uniform)
      memcpy(data_cpu + printf_offset, &dev->printf_buffer_address,
             sizeof(uint64_t));

   struct agx3_ioto_uniform_range ranges[2] = {{
      .start_halfs = AGX_ABI_AGX3_PRECOMP_ARG_HALFS,
      .size_halfs = uploaded_size / 2,
      .address = data_gpu,
   }};
   unsigned range_count = 1;
   if (prog->info.rodata.size_16) {
      ranges[range_count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = prog->info.rodata.base_uniform,
         .size_halfs = prog->info.rodata.size_16,
         .address = prog->ptr + prog->rodata_offset,
      };
   }
   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(false, false, false,
                                                  &launch_sources))
      goto fail;
   uint32_t relative_program = 0;
   uint8_t *program_cpu = d12_list_alloc_usc(
      list, AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE, &relative_program);
   const uint64_t program_va = dev->shader_base + relative_program;
   const struct agx3_g17p_compute_program_desc desc = {
      .program_va = program_va,
      .shader_va = prog->ptr + prog->entry_offset,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .shader_binary_size = prog->info.main_size,
      .scratch_size = prog->info.agx3_tps_size,
      .nr_gprs = prog->info.nr_gprs,
      .compiler_d_valid = prog->info.agx3_staging_known,
      .compiler_d = (uint32_t)prog->info.agx3_staging_halfs * 2,
      .threadgroup_memory_size_B = prog->info.local_size
         ? MAX2(256u, ALIGN_POT(prog->info.local_size, 64u)) : 0,
      .launch_sources = launch_sources,
      .ldshdr_f1 = AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
   };
   struct agx3_g17p_compute_program generated;
   if (!program_cpu || !agx_va_in_usc_window(dev, program_va) ||
       !agx_va_in_usc_window(dev, desc.shader_va) ||
       !agx3_g17p_compute_program_generate(dev->chip, &desc, &generated))
      goto fail;
   memcpy(program_cpu, generated.bytes, generated.length);
   const size_t program_offset = program_va - list->usc_bo->va->addr;
   /* This was the last allocation from the program BO; retain only the
    * actual generated span so repeated small copies do not spend 6 KiB each. */
   list->usc_offset = program_offset + generated.length;
   if (agx_bo_sync_for_device(list->usc_bo, program_offset, generated.length) ||
       agx_bo_sync_for_device(list->stream_bo,
                              data_gpu - list->stream_bo->va->addr,
                              uploaded_size))
      goto fail;

   const struct agx3_compute_direct_grid direct = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {grid.count[0], grid.count[1], grid.count[2]},
      .local_size = {prog->b.workgroup.x, prog->b.workgroup.y,
                     prog->b.workgroup.z},
      .dimensions = grid.count[2] != 1 || prog->b.workgroup.z != 1 ? 3
         : grid.count[1] != 1 || prog->b.workgroup.y != 1 ? 2 : 1,
   };
   struct agx3_compute_dispatch_sink sink = {
      .data = *cursor,
      .capacity = D12_DISPATCH_STREAM_MARGIN,
   };
   if (!agx3_g17p_native_compute_dispatch_append_with_barrier(
          &generated.launch, &direct, 0x5, &sink))
      goto fail;
   *cursor += sink.length;
   return true;

fail:
   mesa_loge("native G17P precompiled compute producer rejected metadata/resources");
   list->state = D12_LIST_ERROR;
   return false;
}

/*
 * Dispatch a precompiled libagx kernel, exactly as hk_dispatch_precomp does.
 */
static void
d12_list_dispatch_precomp_to(struct d12_cmd_list *list,
                             enum libagx_program idx, const void *args,
                             size_t args_size, struct agx_grid grid,
                             bool pre_gfx)
{
   struct d12_device *device = list->device;

   if (list->state != D12_LIST_RECORDING) {
      mesa_loge("recording into a non-recording command list");
      return;
   }

   const bool native = device->dev.chip == AGX_CHIP_G17P;
   if (native) {
      /* Native graphics/indirect integration is a separate producer gate. */
      if (pre_gfx || agx_is_indirect(grid)) {
         mesa_loge("native precompiled graphics/indirect launch unavailable");
         list->state = D12_LIST_ERROR;
         return;
      }
      if (!grid.count[0] || !grid.count[1] || !grid.count[2])
         return;
   } else if (!d12_list_require_legacy_compute(list)) {
      return;
   }

   /* Tessellation runs its libagx kernels ahead of the draw that consumes
    * them, so they go in the pre-graphics stream rather than ending the
    * render pass.
    */
   struct d12_segment *seg =
      pre_gfx ? d12_list_get_pregfx(list) : d12_list_get_cdm(list);
   if (!seg)
      return;

   if (!pre_gfx && seg->dispatches >= D12_MAX_DISPATCHES_PER_SEGMENT) {
      d12_list_close_cdm(list);
      seg = d12_list_get_cdm(list);
      if (!seg)
         return;
   }

   struct agx_precompiled_shader *prog =
      agx_get_precompiled(&device->bg_eot, idx);
   if (!prog) {
      mesa_loge("failed to get precompiled program %d", (int)idx);
      list->state = D12_LIST_ERROR;
      return;
   }

   d12_list_track_read_bo(list, prog->bo);

   if (!d12_list_track_precompiled_reads(list, idx, args, args_size))
      d12_list_mark_read_set_incomplete(list);

   uint8_t **cur = pre_gfx ? &list->pregfx_cur : &list->stream_cur;

   /* A helper consumes what the application's dispatches wrote without
    * an API barrier in between. */
   if (!pre_gfx && !d12_list_flush_cdm_pending(list))
      return;

   /* The native sink bounds its own emission by its capacity. */
   size_t need = native ? D12_DISPATCH_STREAM_MARGIN : D12_CDM_DISPATCH_BYTES;
   if (pre_gfx ? !d12_list_check_pregfx_space(list, need)
               : !d12_list_check_stream_space(list, need))
      return;

   if (native) {
      if (!d12_list_append_g17p_precomp(list, prog, args, args_size, grid, cur))
         return;
   } else {
      uint64_t data_gpu = d12_list_upload(list, args, args_size, 4);
      if (!data_gpu)
         return;

      uint32_t usc;
      uint8_t *usc_cpu = d12_list_alloc_usc(list, agx_usc_size(15), &usc);
      if (!usc_cpu)
         return;

      agx_usc_words_precomp((uint32_t *)usc_cpu, &prog->b, data_gpu, args_size);

      if (d12_debug()->dump_usc) {
         fprintf(stderr, "precomp USC @%u:", usc);
         for (unsigned i = 0; i < 32; i++)
            fprintf(stderr, " %02x", usc_cpu[i]);
         fprintf(stderr, "\n  data_gpu=0x%llx size=%zu\n",
                 (unsigned long long)data_gpu, args_size);
      }

      if (!d12_list_emit_legacy_cdm_launch(list, cur, grid, prog->b.workgroup,
                                           prog->b.launch, usc))
         return;
      if (!d12_list_emit_cdm_cache_flush(list, cur))
         return;
   }
   seg->dispatches++;
   d12_segment_track_precompiled_write(seg, idx, args, args_size);
}

static void
d12_list_dispatch_precomp(struct d12_cmd_list *list, enum libagx_program idx,
                          const void *args, size_t args_size,
                          struct agx_grid grid)
{
   d12_list_dispatch_precomp_to(list, idx, args, args_size, grid, false);
}

/* Return an indirect argument record which is either an exact copy of `input`
 * or all zeroes according to the SetPredication snapshot. The pre-graphics
 * form is inserted before an already-open render segment, while compute uses
 * the ordinary ordered CDM stream. */
static uint64_t
d12_list_apply_predication(struct d12_cmd_list *list, uint64_t input,
                           unsigned words, bool pre_gfx)
{
   if (!list->predicate_gpu)
      return input;

   if (!input || !words || words > 5) {
      mesa_loge("invalid predicated argument record");
      list->state = D12_LIST_ERROR;
      return 0;
   }

   uint64_t output = d12_list_upload_zero(list, words * sizeof(uint32_t), 4);
   if (!output)
      return 0;

   struct libagx_d3d12_predicate_args args = {
      .out = output,
      .in = input,
      .predicate = list->predicate_gpu,
      .words = words,
      .skip_if_nonzero =
         list->predicate_op == D3D12_PREDICATION_OP_NOT_EQUAL_ZERO,
   };

   d12_list_dispatch_precomp_to(list, LIBAGX_D3D12_PREDICATE, &args,
                                sizeof(args), agx_1d(1), pre_gfx);
   if (d12_debug()->dump_predication)
      mesa_logi("predication patch: snapshot=0x%llx input=0x%llx "
                "output=0x%llx words=%u skip_if_nonzero=%u pregfx=%u",
                (unsigned long long)args.predicate,
                (unsigned long long)args.in,
                (unsigned long long)args.out, args.words,
                args.skip_if_nonzero, pre_gfx);
   return list->state == D12_LIST_RECORDING ? output : 0;
}

struct d12_indirect_root_patch {
   uint64_t in;
   uint64_t draw_count;
   uint64_t previous_tables;
   uint64_t sparse_ro_offset;
   uint32_t previous_table_count;
   uint32_t command_index;
   uint32_t slot;
   bool publish_unbounded;
   bool read_only;
};

static bool
d12_indirect_root_has_uav(const struct d12_indirect_root_patch *patches,
                          unsigned patch_count)
{
   for (unsigned i = 0; i < patch_count; i++) {
      if (!patches[i].read_only)
         return true;
   }
   return false;
}

struct d12_indirect_vb_patch {
   uint64_t in;
   uint64_t draw_count;
   uint64_t previous_params;
   uint32_t command_index;
   uint32_t slot;
};

/* Emit execution-time argument patches immediately before their consumer.
 * Graphics uses the segment's pre-graphics stream; compute uses the ordinary
 * ordered CDM stream.  Root descriptors target a unique compact root table,
 * whose address is known only after binding upload. */
static bool
d12_list_dispatch_indirect_patches(
   struct d12_cmd_list *list,
   const struct libagx_d3d12_indirect_patch_args *patches,
   unsigned patch_count,
   const struct libagx_d3d12_indirect_constant_args *constant_patches,
   unsigned constant_patch_count,
   const struct d12_indirect_root_patch *root_patches,
   unsigned root_patch_count, uint64_t tables_gpu, uint32_t table_count,
   bool pre_gfx)
{
   for (unsigned i = 0; i < patch_count; ++i) {
      d12_list_dispatch_precomp_to(list, LIBAGX_D3D12_INDIRECT_PATCH,
                                   &patches[i], sizeof(patches[i]), agx_1d(1),
                                   pre_gfx);
      if (list->state != D12_LIST_RECORDING)
         return false;
   }

   for (unsigned i = 0; i < constant_patch_count; ++i) {
      d12_list_dispatch_precomp_to(
         list, LIBAGX_D3D12_INDIRECT_CONSTANT, &constant_patches[i],
         sizeof(constant_patches[i]), agx_1d(1), pre_gfx);
      if (list->state != D12_LIST_RECORDING)
         return false;
   }

   if (root_patch_count && (!tables_gpu || !table_count)) {
      mesa_loge("ExecuteIndirect: no compact root table for descriptor patch");
      list->state = D12_LIST_ERROR;
      return false;
   }

   for (unsigned i = 0; i < root_patch_count; ++i) {
      const struct d12_indirect_root_patch *patch = &root_patches[i];
      if (patch->slot >= table_count) {
         mesa_loge("ExecuteIndirect: root slot %u exceeds table size %u",
                   patch->slot, table_count);
         list->state = D12_LIST_ERROR;
         return false;
      }

      if (patch->previous_tables &&
          patch->slot >= patch->previous_table_count) {
         mesa_loge("ExecuteIndirect: previous root slot %u exceeds table "
                   "size %u", patch->slot, patch->previous_table_count);
         list->state = D12_LIST_ERROR;
         return false;
      }

      uint64_t null_address = d12_list_null_buffer(list);
      if (!null_address)
         return false;

      const struct libagx_d3d12_indirect_root_args args = {
         .out_address = tables_gpu + patch->slot * sizeof(uint64_t),
         .out_bufinfo = tables_gpu +
                        (2 * table_count + patch->slot) * sizeof(uint64_t),
         .out_residency = tables_gpu +
                          (3 * table_count + patch->slot) * sizeof(uint64_t),
         .in = patch->in,
         .draw_count = patch->draw_count,
         .previous_address = patch->previous_tables
            ? patch->previous_tables + patch->slot * sizeof(uint64_t) : 0,
         .previous_bufinfo = patch->previous_tables
            ? patch->previous_tables +
                 (2 * patch->previous_table_count + patch->slot) *
                    sizeof(uint64_t)
            : 0,
         .previous_residency = patch->previous_tables
            ? patch->previous_tables +
                 (3 * patch->previous_table_count + patch->slot) *
                    sizeof(uint64_t)
            : 0,
         .sparse_registry = list->sparse_registry_view_gpu,
         .null_address = null_address,
         .sparse_ro_offset = patch->sparse_ro_offset,
         .command_index = patch->command_index,
         .publish_unbounded = patch->publish_unbounded,
         .translate_read = patch->read_only,
      };
      d12_list_dispatch_precomp_to(list, LIBAGX_D3D12_INDIRECT_ROOT, &args,
                                   sizeof(args), agx_1d(1), pre_gfx);
      if (list->state != D12_LIST_RECORDING)
         return false;
   }

   return true;
}

/* Native one-layer, one-sample tiled RGBA8/BGRA8 tile traffic. The legacy
 * builder remains guarded until the renderer wires this producer together
 * with native PPP/VDM lifetime and render-command qualification. */
uint32_t
d12_build_g17p_bg_eot(struct d12_cmd_list *list, const struct d12_segment *seg,
                     const float legacy_color[4],
                     const enum agx_bg_eot_op ops[AGX_MAX_RENDER_TARGETS],
                     const struct agx_tilebuffer_layout *tib,
                     uint64_t *out_resources)
{
   struct d12_device *device = list->device;
   struct agx_device *dev = &device->dev;
   struct d12_resource *rt = seg->rt[0];
   const enum agx_bg_eot_op op = ops[0];
   const char *failure_stage = "attachment-shape";
   struct agx_bg_eot_shader *shader = NULL;

   if (dev->chip != AGX_CHIP_G17P || !out_resources || !rt || !rt->bo ||
       (seg->ds && (!d12_g17p_depth_enabled(dev) ||
                    !d12_g17p_depth_segment_supported(seg))) ||
       rt->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       rt->layout.sample_count_sa != 1 ||
       seg->layers != 1 || tib->nr_samples != 1 || seg->rt_level[0] != 0 ||
       (rt->layout.tiling != AIL_TILING_GPU &&
        rt->layout.tiling != AIL_TILING_TWIDDLED) || rt->layout.levels != 1 ||
       (seg->rt_view_format[0] != PIPE_FORMAT_R8G8B8A8_UNORM &&
        seg->rt_view_format[0] != PIPE_FORMAT_B8G8R8A8_UNORM) ||
       (op != AGX_BG_CLEAR && op != AGX_BG_LOAD && op != AGX_EOT_STORE &&
        op != AGX_BG_EOT_NONE))
      goto fail;
   failure_stage = "extra-attachment";
   for (unsigned i = 1; i < AGX_MAX_RENDER_TARGETS; i++)
      if (seg->rt[i] || ops[i] != AGX_BG_EOT_NONE)
         goto fail;
   failure_stage = "attachment-address";
   if (!seg->rt_address[0])
      goto fail;
   struct agx_bg_eot_key key = {.tib = *tib};
   key.op[0] = op;
   if (op == AGX_BG_CLEAR)
      key.reserved_preamble = AGX3_G17P_RT0_CLEAR_PUSH_HALFS;
   failure_stage = "aux-shader-metadata";
   shader = agx_get_bg_eot_shader(&device->bg_eot, &key);
   if (!shader || !shader->bo || !shader->info.main_size ||
       !shader->info.nr_gprs || shader->info.has_preamble ||
       shader->info.rodata.size_16 || shader->info.scratch_size ||
       !shader->info.agx3_staging_known ||
       shader->ptr > UINT64_MAX - shader->info.main_offset)
      goto fail;
   failure_stage = "aux-shader-usc-window";
   const uint64_t shader_va = shader->ptr + shader->info.main_offset;
   if (!agx_va_in_usc_window(dev, shader_va))
      goto fail;
   failure_stage = "program-allocation";
   uint32_t relative = 0;
   uint8_t *cpu = d12_list_alloc_usc(list, AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE,
                                   &relative);
   if (!cpu)
      goto fail;
   const uint64_t program_va = dev->shader_base + relative;
   uint64_t resource_spec = 0;
   uint8_t program[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   size_t length = 0;
   const struct agx_ioto_g17p_imageblock imageblock = {
      .sample_size_B = tib->sample_size_B, .nr_samples = tib->nr_samples,
      .tile_size_px = tib->tile_size,
   };
   if (op == AGX_EOT_STORE) {
      const enum pipe_format pf = seg->rt_view_format[0];
      const struct util_format_description *fmt = util_format_description(pf);
      struct AGX_PBE semantic = {AGX_PBE_header};
      semantic.dimension = AGX_TEXTURE_DIMENSION_2D;
      semantic.layout = agx_translate_layout(rt->layout.tiling);
      semantic.channels = ail_pixel_format[pf].channels;
      semantic.type = ail_pixel_format[pf].type;
      semantic.buffer = seg->rt_address[0];
      semantic.width = rt->layout.width_px;
      semantic.height = rt->layout.height_px;
      semantic.levels = semantic.layers = 1;
      semantic.page_aligned_layers = rt->layout.page_aligned_layers;
      semantic.swizzle_r = semantic.swizzle_g = semantic.swizzle_b = semantic.swizzle_a = 0;
      for (unsigned c = 0; c < 4; c++) {
         if (fmt->swizzle[c] == PIPE_SWIZZLE_X) semantic.swizzle_r = c;
         if (fmt->swizzle[c] == PIPE_SWIZZLE_Y) semantic.swizzle_g = c;
         if (fmt->swizzle[c] == PIPE_SWIZZLE_Z) semantic.swizzle_b = c;
         if (fmt->swizzle[c] == PIPE_SWIZZLE_W) semantic.swizzle_a = c;
      }
      failure_stage = "eot-pbe-pack";
      uint8_t pbe[32];
      if (!d12_pack_pbe_generation(agx_chip_descriptor_generation(dev->chip),
                                  pbe, &semantic))
         goto fail;
      failure_stage = "eot-pbe-upload";
      const uint64_t argument = d12_list_upload(list, pbe, sizeof(pbe), 32);
      if (!argument)
         goto fail;
      const struct agx3_g17p_rt0_eot_program_desc desc = {
         .program_va = program_va, .shader_va = shader_va,
         .pbe_argument_va = argument, .attachment_count = 1,
         .imageblock_offset_B = agx_tilebuffer_offset_B(tib, 0),
         .shader_binary_size = shader->info.binary_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_halfs,
         .imageblock = imageblock,
      };
      failure_stage = "eot-program-generate";
      struct agx3_g17p_rt0_eot_program generated;
      if (!agx3_g17p_rt0_eot_program_generate(dev->chip, &desc, &generated))
         goto fail;
      memcpy(program, generated.bytes, generated.length);
      length = generated.length;
      resource_spec = AGX_IOTO_G17P_EOT_RESOURCE_SPEC;
   } else if (op == AGX_BG_EOT_NONE) {
      failure_stage = "empty-shader-resources";
      if (shader->info.push_count || shader->info.uses_txf)
         goto fail;
      const struct agx3_g17p_graphics_program_desc desc = {
         .program_va = program_va, .shader_va = shader_va,
         .shader_binary_size = shader->info.binary_size,
         .nr_gprs = shader->info.nr_gprs, .compiler_d_valid = true,
         .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
         .fragment = true, .imageblock = imageblock,
      };
      failure_stage = "empty-program-generate";
      struct agx3_g17p_graphics_program generated;
      if (!agx3_g17p_graphics_program_generate(dev->chip, &desc, &generated) ||
          !agx3_g17p_rt0_bg_resource_spec(dev->chip, AGX3_G17P_RT0_BG_NONE,
                                         &resource_spec))
         goto fail;
      memcpy(program, generated.bytes, generated.length);
      length = generated.length;
   } else {
      struct agx3_g17p_rt0_bg_program_desc desc = {
         .program_va = program_va, .shader_va = shader_va,
         .shader_binary_size = shader->info.binary_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_halfs,
         .clear_push_halfs = shader->info.push_count, .staging_known = true,
         .op = op == AGX_BG_CLEAR ? AGX3_G17P_RT0_BG_CLEAR : AGX3_G17P_RT0_BG_RELOAD,
         .imageblock = imageblock,
      };
      failure_stage = "background-shader-resources";
      if (!agx3_g17p_rt0_bg_shader_resources_match(desc.op,
             shader->info.push_count, shader->info.uses_txf))
         goto fail;
      if (op == AGX_BG_CLEAR) {
         failure_stage = "clear-argument";
         const float *color = seg->render_pass ? seg->rt_clear[0] : legacy_color;
         if (!color)
            goto fail;
         desc.clear_value_va = d12_list_upload(list, color, 16, 16);
         if (!desc.clear_value_va)
            goto fail;
      } else {
         const enum pipe_format pf = seg->rt_view_format[0];
         const struct util_format_description *fmt = util_format_description(pf);
         struct AGX_TEXTURE texture = {AGX_TEXTURE_header};
         texture.dimension = AGX_TEXTURE_DIMENSION_2D;
         texture.layout = agx_translate_layout(rt->layout.tiling);
         texture.channels = ail_pixel_format[pf].channels;
         texture.type = ail_pixel_format[pf].type;
         texture.swizzle_r = agx_channel_from_pipe(fmt->swizzle[0]);
         texture.swizzle_g = agx_channel_from_pipe(fmt->swizzle[1]);
         texture.swizzle_b = agx_channel_from_pipe(fmt->swizzle[2]);
         texture.swizzle_a = agx_channel_from_pipe(fmt->swizzle[3]);
         texture.address = d12_sparse_read_address(device, seg->rt_address[0]);
         texture.width = rt->layout.width_px;
         texture.height = rt->layout.height_px;
         texture.depth = 1;
         texture.page_aligned_layers = rt->layout.page_aligned_layers;
         failure_stage = "reload-texture-pack";
         uint8_t packed[32];
         if (!device->rodata_bo || !device->rodata_bo->va ||
             !d12_pack_texture_generation(agx_chip_descriptor_generation(dev->chip),
                                          packed, &texture, 1))
            goto fail;
         failure_stage = "reload-texture-upload";
         const uint64_t table_va = d12_list_upload(list, packed, sizeof(packed), 32);
         if (!table_va)
            goto fail;
         const uint64_t pointers[2] = {table_va, device->rodata_bo->va->addr};
         failure_stage = "reload-pointer-upload";
         const uint64_t pointers_va = d12_list_upload(list, pointers, sizeof(pointers), 8);
         if (!pointers_va)
            goto fail;
         desc.texture_pointer_va = pointers_va;
         desc.sampler_pointer_va = pointers_va + 8;
      }
      failure_stage = "background-program-generate";
      struct agx3_g17p_rt0_bg_program generated;
      if (!agx3_g17p_rt0_bg_program_generate(dev->chip, &desc, &generated) ||
          !agx3_g17p_rt0_bg_resource_spec(dev->chip, desc.op, &resource_spec))
         goto fail;
      memcpy(program, generated.bytes, generated.length);
      length = generated.length;
   }
   failure_stage = "generated-program-length";
   if (!length || length > AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE)
      goto fail;
   memcpy(cpu, program, length);
   const size_t offset = program_va - list->usc_bo->va->addr;
   list->usc_offset = offset + length;
   failure_stage = "program-arguments-publish";
   if (agx_bo_sync_for_device(list->usc_bo, offset, length) ||
       agx_bo_sync_for_device(list->stream_bo, list->data_floor,
                              list->stream_bo->size - list->data_floor))
      goto fail;
   *out_resources = resource_spec;
   d12_list_track_read_bo(list, shader->bo);
   return relative;
fail:
   mesa_loge("native BG/EOT rejected stage=%s op=%u rt=%u fmt=%u tiling=%u "
             "levels=%u samples=%u/%u layers=%u address=%llx "
             "shader=%u main=%u@%u binary=%u gprs=%u pre=%u rodata=%u "
             "scratch=%u staging=%u/%u push=%u txf=%u tile=%u sample_bytes=%u",
             failure_stage, (unsigned)op, rt != NULL,
             (unsigned)seg->rt_view_format[0], rt ? (unsigned)rt->layout.tiling : 0,
             rt ? rt->layout.levels : 0, rt ? rt->layout.sample_count_sa : 0,
             tib->nr_samples, seg->layers,
             (unsigned long long)seg->rt_address[0], shader != NULL,
             shader ? shader->info.main_size : 0, shader ? shader->info.main_offset : 0,
             shader ? shader->info.binary_size : 0, shader ? shader->info.nr_gprs : 0,
             shader ? shader->info.has_preamble : 0, shader ? shader->info.rodata.size_16 : 0,
             shader ? shader->info.scratch_size : 0,
             shader ? shader->info.agx3_staging_known : 0,
             shader ? shader->info.agx3_staging_halfs : 0,
             shader ? shader->info.push_count : 0, shader ? shader->info.uses_txf : 0,
             tib->tile_size, tib->sample_size_B);
   list->state = D12_LIST_ERROR;
   return 0;
}

/*
 * Build the USC words for a background (clear) or end-of-tile (store)
 * program targeting the render segment's selected subresources. Mirrors
 * hk_build_bg_eot for the bounded RT/layer/sample subset used here.
 */
static uint32_t
d12_build_bg_eot_usc(struct d12_cmd_list *list, const struct d12_segment *seg,
                     const float legacy_color[4],
                     const enum agx_bg_eot_op ops[AGX_MAX_RENDER_TARGETS],
                     const struct agx_tilebuffer_layout *tib,
                     struct agx_counts_packed *out_counts, uint64_t *out_native_resources)
{
   struct d12_device *device = list->device;

   if (device->dev.chip == AGX_CHIP_G17P &&
       d12_g17p_baseline_graphics_enabled(&list->device->dev))
      return d12_build_g17p_bg_eot(list, seg, legacy_color, ops, tib, out_native_resources);

   /* This builder emits AGX2 USC resource/state words. A clear-only list
    * never reaches the vertex-state guard, so reject here until native
    * BG/EOT Ioto programs and render resource specifications are supplied. */
   if (agx_chip_command_stream_generation(device->dev.chip) !=
       AGX_CS_GENERATION_AGX2) {
      mesa_loge("native graphics background/EOT producer unavailable");
      list->state = D12_LIST_ERROR;
      return 0;
   }

   uint32_t usc;
   size_t usc_size = agx_usc_size(3 + 8);
   uint8_t *cpu = d12_list_alloc_usc(list, usc_size, &usc);
   if (!cpu)
      return 0;

   struct agx_usc_builder b = agx_usc_builder(cpu, usc_size);
   struct agx_bg_eot_key key = {.tib = *tib};
   key.tib.layered = seg->layers > 1;
   unsigned uniforms = 0, nr_tex = 0;
   bool uses_txf = false;

   for (unsigned rt_index = 0; rt_index < AGX_MAX_RENDER_TARGETS; rt_index++) {
      enum agx_bg_eot_op op = ops[rt_index];
      struct d12_resource *rt = seg->rt[rt_index];
      uint64_t rt_address = seg->rt_address[rt_index];
      unsigned rt_level = seg->rt_level[rt_index];
      enum pipe_format type_format = seg->rt_view_format[rt_index];
      key.op[rt_index] = op;

      if (op == AGX_BG_EOT_NONE)
         continue;

      if (!rt) {
         mesa_loge("background/EOT operation requested for unbound RT%u",
                   rt_index);
         list->state = D12_LIST_ERROR;
         return 0;
      }

      if (op == AGX_BG_CLEAR) {

         /*
          * ClearRenderTargetView always hands over FLOAT[4], whatever the
          * target's type. For a PURE INTEGER target those floats are integer
          * VALUES, not normalized intensities, so they must be converted
          * rather than forwarded: clearing a UINT target to 1 has to write
          * the integer 1, where passing the float straight through wrote 255
          * ("Got 0xff00ff00, expected 0x01000100" in
          * test_clear_render_target_view).
          *
          * The format consulted here is the RESOURCE's, which is safe because
          * it is only being asked "is this integer-typed?". An earlier attempt
          * also drove the tilebuffer and PBE from a view format; those describe
          * MEMORY, so the store wrote the wrong size and produced 75 GPU
          * timeouts. Type interpretation and memory layout are different
          * questions -- do not merge them again.
          */
         /* The VIEW's format answers "what type is this", which the resource
          * format cannot for a TYPELESS target -- typeless maps to a float
          * sibling purely to fix the block size, so a UINT clear otherwise
          * looked like a float one.
          *
          * Measured, not assumed: view format 8 failures vs resource format 15
          * in test_clear_render_target_view.
          *
          * Used for THAT QUESTION ONLY. Memory layout still comes from the
          * resource; merging the two wrote the wrong size and cost 75 GPU
          * timeouts.
          */
         enum pipe_format cf =
            type_format != PIPE_FORMAT_NONE ? type_format : rt->layout.format;
         union {
            float f[4];
            uint32_t u[4];
            int32_t i[4];
         } cv;

         /* A render pass clears each attachment to its own value; every
          * other path clears one target at a time and carries one colour. */
         const float *color = seg->render_pass ? seg->rt_clear[rt_index]
                                               : legacy_color;
         if (!color) {
            mesa_loge("background clear requested for RT%u with no clear "
                      "value", rt_index);
            list->state = D12_LIST_ERROR;
            return 0;
         }

         if (util_format_is_pure_uint(cf)) {
            for (unsigned c = 0; c < 4; c++)
               cv.u[c] = (uint32_t)color[c];
         } else if (util_format_is_pure_sint(cf)) {
            for (unsigned c = 0; c < 4; c++)
               cv.i[c] = (int32_t)color[c];
         } else {
            memcpy(cv.f, color, sizeof(cv.f));
         }

         /* 16 bytes of clear color, register base 4 — hk's fixed ABI */
         uint64_t colour = d12_list_upload(list, cv.f, 16, 16);
         if (!colour)
            return 0;

         agx_usc_uniform(&b, 4 + (8 * rt_index), 8, colour);
         uniforms = MAX2(uniforms, 4 + (8 * rt_index) + 8);
      } else if (op == AGX_BG_LOAD) {

         enum pipe_format pf =
            type_format != PIPE_FORMAT_NONE ? type_format : rt->layout.format;
         const struct util_format_description *fdesc =
            util_format_description(pf);
         const unsigned samples = MAX2(rt->layout.sample_count_sa, 1);
         struct AGX_TEXTURE semantic = {AGX_TEXTURE_header};
         {
            semantic.dimension =
               seg->layers > 1
                  ? (samples > 1
                        ? AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED
                        : AGX_TEXTURE_DIMENSION_2D_ARRAY)
                  : (samples > 1 ? AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED
                                 : AGX_TEXTURE_DIMENSION_2D);
            semantic.layout = agx_translate_layout(rt->layout.tiling);
            semantic.channels = ail_pixel_format[pf].channels;
            semantic.type = ail_pixel_format[pf].type;
            semantic.srgb = util_format_is_srgb(pf);
            semantic.swizzle_r = agx_channel_from_pipe(fdesc->swizzle[0]);
            semantic.swizzle_g = agx_channel_from_pipe(fdesc->swizzle[1]);
            semantic.swizzle_b = agx_channel_from_pipe(fdesc->swizzle[2]);
            semantic.swizzle_a = agx_channel_from_pipe(fdesc->swizzle[3]);
            semantic.address =
               d12_sparse_read_address(list->device, rt_address);
            semantic.width = rt->layout.width_px;
            semantic.height = rt->layout.height_px;
            semantic.first_level = rt_level;
            semantic.last_level = rt_level;
            semantic.unk_mipmapped = rt->layout.levels > 1;

            if (samples > 1)
               semantic.samples = agx_translate_sample_count(samples);

            if (rt->layout.tiling == AIL_TILING_LINEAR) {
               semantic.stride =
                  ail_get_linear_stride_B(&rt->layout, 0) - 16;
               if (seg->layers > 1) {
                  semantic.depth_linear = seg->layers;
                  semantic.layer_stride_linear =
                     rt->layout.layer_stride_B - 0x80;
                  semantic.extended = true;
               }
            } else {
               semantic.depth = seg->layers;
               semantic.page_aligned_layers =
                  rt->layout.page_aligned_layers;
            }
         }

         struct d12_native_descriptor_layout native_layout;
         const enum agx_descriptor_generation generation =
            agx_chip_descriptor_generation(device->dev.chip);
         uint8_t desc[32];
         if (!d12_native_descriptor_layout(generation, &native_layout) ||
             !d12_pack_texture_generation(generation, desc, &semantic,
                                          samples)) {
            mesa_loge("background load needs an unproven native texture axis");
            list->state = D12_LIST_ERROR;
            return 0;
         }

         uint64_t desc_ptr = d12_list_upload(
            list, desc, native_layout.texture_stride_B, 8);
         if (!desc_ptr)
            return 0;

         if (d12_debug()->validate_mrt)
            mesa_logi("MRT validate: BG_LOAD RT%u texture_slot=%u "
                      "desc=0x%llx",
                      rt_index, rt_index * 2,
                      (unsigned long long)desc_ptr);

         agx_usc_pack(&b, TEXTURE, cfg) {
            /* Background texture handles use the even half of eMRT's
             * two-slots-per-target ABI, matching agx_bg_eot.c.
             */
            cfg.start = rt_index * 2;
            cfg.count = 1;
            cfg.buffer = desc_ptr;
         }
         nr_tex = MAX2(nr_tex, (rt_index * 2) + 1);
         uses_txf = true;
      } else if (op == AGX_EOT_STORE) {

         /* EOT uses the view's channel interpretation but the resource's actual
          * memory layout. This is the bounded attachment subset of
          * hk_image_view.c::pack_pbe.
          */
         const unsigned samples = MAX2(rt->layout.sample_count_sa, 1);
         struct AGX_PBE semantic = {AGX_PBE_header};
         {
            semantic.dimension =
               seg->layers > 1
                  ? (samples > 1
                        ? AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED
                        : AGX_TEXTURE_DIMENSION_2D_ARRAY)
                  : (samples > 1 ? AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED
                                 : AGX_TEXTURE_DIMENSION_2D);
            semantic.layout = agx_translate_layout(rt->layout.tiling);
            enum pipe_format pf = type_format != PIPE_FORMAT_NONE
                                     ? util_format_linear(type_format)
                                     : rt->layout.format;
            const struct util_format_description *fdesc =
               util_format_description(pf);

            semantic.channels = ail_pixel_format[pf].channels;
            semantic.type = ail_pixel_format[pf].type;
            semantic.srgb = util_format_is_srgb(pf);

            /* Honour the format's channel order, so BGRA does not come out
             * with red and blue swapped.
             */
            semantic.swizzle_r = 0;
            semantic.swizzle_g = 1;
            semantic.swizzle_b = 2;
            semantic.swizzle_a = 3;
            for (unsigned c = 0; c < fdesc->nr_channels && c < 4; c++) {
               switch (fdesc->swizzle[c]) {
               case PIPE_SWIZZLE_X:
                  semantic.swizzle_r = c;
                  break;
               case PIPE_SWIZZLE_Y:
                  semantic.swizzle_g = c;
                  break;
               case PIPE_SWIZZLE_Z:
                  semantic.swizzle_b = c;
                  break;
               case PIPE_SWIZZLE_W:
                  semantic.swizzle_a = c;
                  break;
               default:
                  break;
               }
            }
            semantic.buffer = rt_address;
            semantic.width = rt->layout.width_px;
            semantic.height = rt->layout.height_px;
            semantic.level = rt_level;
            semantic.levels = rt->layout.levels;
            semantic.unk_mipmapped = rt->layout.levels > 1;
            if (samples > 1)
               semantic.samples = agx_translate_sample_count(samples);
            if (rt->layout.tiling == AIL_TILING_LINEAR) {
               semantic.stride =
                  ail_get_linear_stride_B(&rt->layout, 0) - 4;
               if (seg->layers > 1) {
                  semantic.depth_linear = seg->layers;
                  semantic.layer_stride_linear =
                     rt->layout.layer_stride_B - 0x80;
                  semantic.extended = true;
               } else {
                  semantic.layers = 1;
               }
            } else {
               semantic.layers = seg->layers;
               semantic.page_aligned_layers =
                  rt->layout.page_aligned_layers;
            }
         }

         struct d12_native_descriptor_layout native_layout;
         const enum agx_descriptor_generation generation =
            agx_chip_descriptor_generation(device->dev.chip);
         uint8_t desc[32];
         if (!d12_native_descriptor_layout(generation, &native_layout) ||
             !d12_pack_pbe_generation(generation, desc, &semantic)) {
            mesa_loge("color store needs an unproven native PBE axis");
            list->state = D12_LIST_ERROR;
            return 0;
         }

         uint64_t desc_ptr =
            d12_list_upload(list, desc, native_layout.pbe_stride_B, 8);
         if (!desc_ptr)
            return 0;

         if (d12_debug()->validate_mrt)
            mesa_logi("MRT validate: EOT_STORE RT%u pbe_slot=%u desc=0x%llx",
                      rt_index, rt_index,
                      (unsigned long long)desc_ptr);

         agx_usc_pack(&b, TEXTURE, cfg) {
            cfg.start = rt_index;
            cfg.count = 1;
            cfg.buffer = desc_ptr;
         }
         nr_tex = MAX2(nr_tex, rt_index + 1);
      }
   }

   if (uses_txf)
      agx_usc_push_packed(&b, SAMPLER, device->dev.txf_sampler);

   agx_usc_push_packed(&b, SHARED, &key.tib.usc);

   key.reserved_preamble = uniforms;
   struct agx_bg_eot_shader *shader =
      agx_get_bg_eot_shader(&device->bg_eot, &key);
   if (!shader) {
      mesa_loge("failed to build bg/eot shader");
      list->state = D12_LIST_ERROR;
      return 0;
   }
   d12_list_track_read_bo(list, shader->bo);

   agx_usc_pack(&b, SHADER, cfg) {
      cfg.code = agx_usc_addr(&device->dev,
                              shader->ptr + shader->info.main_offset);
      cfg.unk_2 = 0;
   }

   agx_usc_pack(&b, REGISTERS, cfg)
      cfg.register_count = shader->info.nr_gprs;

   if (shader->info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code = agx_usc_addr(&device->dev,
                                 shader->ptr + shader->info.preamble_offset);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   agx_pack(out_counts, COUNTS, cfg) {
      cfg.uniform_register_count = shader->info.push_count;
      cfg.preshader_register_count = shader->info.nr_preamble_gprs;
      cfg.texture_state_register_count = nr_tex;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(uses_txf ? 1 : 0, false);
   }

   if (d12_debug()->validate_mrt)
      mesa_logi("MRT validate: bg/eot ops ready nr_tex=%u uniforms=%u "
                "uses_txf=%u",
                nr_tex, uniforms, uses_txf);

   return usc;
}

static bool
d12_segment_matches_bindings(const struct d12_cmd_list *list,
                             const struct d12_segment *seg, bool use_rtv)
{
   const unsigned rtv_count = use_rtv ? list->bound_rtv_count : 0;
   const struct d12_descriptor *dsv =
      list->has_bound_dsv ? &list->bound_dsv : NULL;

   if (seg->rt_count != rtv_count ||
       seg->ds != (dsv ? dsv->resource : NULL))
      return false;

   for (unsigned i = 0; i < rtv_count; i++) {
      const struct d12_descriptor *rtv = &list->bound_rtv[i];
      if (seg->rt[i] != rtv->resource)
         return false;

      if (rtv->resource &&
          (seg->rt_address[i] != rtv->address ||
           seg->rt_view_format[i] != d12_format_to_pipe(rtv->format) ||
           seg->rt_level[i] != rtv->texture_level ||
           seg->layers != MAX2(rtv->size_B, 1)))
         return false;
   }

   if (dsv &&
       (seg->dsv_flags != dsv->dsv_flags ||
        seg->ds_address != dsv->address ||
        seg->stencil_address != dsv->counter_address ||
        seg->layers != MAX2(dsv->size_B, 1)))
      return false;

   return true;
}

/* Return the common format of all non-null RTVs, or NONE for an empty or
 * mixed-format framebuffer.  A common real format is the proof needed to
 * materialize UNKNOWN PSO slots without inventing their representation. */
static enum pipe_format
d12_segment_uniform_rt_format(const struct d12_segment *seg)
{
   enum pipe_format format = PIPE_FORMAT_NONE;

   for (unsigned i = 0; i < seg->rt_count; i++) {
      if (!seg->rt[i])
         continue;

      if (format == PIPE_FORMAT_NONE)
         format = seg->rt_view_format[i];
      else if (seg->rt_view_format[i] != format)
         return PIPE_FORMAT_NONE;
   }

   return format;
}

static bool
d12_segment_matches_unused_rt_layout(const struct d12_segment *seg,
                                     const struct d12_pipeline_state *pso)
{
   if (seg->tib.nr_samples != pso->tib.nr_samples)
      return false;

   /* No colour can reach the tilebuffer, so only the sample ABI matters. */
   if (!pso->rt_write_mask)
      return true;

   return pso->stable_unused_rt_layout &&
          memcmp(&seg->tib, &pso->tib, sizeof(seg->tib)) == 0;
}

/*
 * Open a render segment from authoritative OM descriptors. D3D12 does not
 * require a clear before drawing, so AGX's main background program must LOAD
 * the target in that case. Partial renders always use their own LOAD/STORE
 * pair: reusing a CLEAR program after a tiler spill destroys earlier draws.
 */
static struct d12_segment *
d12_list_begin_render(struct d12_cmd_list *list,
                      const struct d12_descriptor *rtvs, unsigned rtv_count,
                      const struct d12_descriptor *dsv,
                      struct d12_pipeline_state *pso,
                      enum agx_bg_eot_op bg_op, const float color[4],
                      bool process_empty_tiles)
{
   d12_list_close_cdm(list);

   /* Two slots: the pre-graphics prelude and the render segment itself. */
   if (list->segment_count + 2 > D12_MAX_SEGMENTS) {
      mesa_loge("command list segment limit reached");
      list->state = D12_LIST_ERROR;
      return NULL;
   }

   if (rtv_count > AGX_MAX_RENDER_TARGETS || (rtv_count && !rtvs)) {
      mesa_loge("render pass received %u RTV descriptors", rtv_count);
      list->state = D12_LIST_ERROR;
      return NULL;
   }

   /* Native depth has no qualified producer yet, including clear-only
    * passes which never reach the triangle draw admission guard. */
   if (dsv && list->device->dev.chip == AGX_CHIP_G17P &&
       !d12_g17p_depth_enabled(&list->device->dev)) {
      mesa_loge("native depth render producer remains closed");
      list->state = D12_LIST_ERROR;
      return NULL;
   }

   if (dsv && (!dsv->resource || dsv->kind != D12_DESC_DSV)) {
      mesa_loge("render pass received an invalid DSV descriptor");
      list->state = D12_LIST_ERROR;
      return NULL;
   }

   /* Reserve the pre-graphics prelude in the slot before the render
    * segment, so that d12_list_get_pregfx can open it later without
    * moving the render segment (see there). An unused prelude is an empty
    * CDM segment, which submission skips. Neither slot is committed until
    * the render segment is complete. */
   struct d12_segment *prelude = &list->segments[list->segment_count];
   memset(prelude, 0, sizeof(*prelude));
   prelude->type = D12_SEG_CDM;

   struct d12_segment *seg = &list->segments[list->segment_count + 1];
   memset(seg, 0, sizeof(*seg));
   seg->type = D12_SEG_RENDER;
   seg->process_empty_tiles = process_empty_tiles;
   seg->layers = 1;

   /* Sample positions are fixed for a whole hardware render, so snapshot
    * them when the segment opens rather than reading them at submit. */
   seg->sample_positions =
      list->sample_positions_set ? list->sample_positions : 0;

   /* A declared render pass states its tile traffic per attachment. Carry it
    * from the list, where BeginRenderPass recorded it before any draw
    * existed to open this segment. */
   if (list->render_pass_active) {
      seg->render_pass = true;
      memcpy(seg->rt_plan, list->render_pass_rt_plan,
             sizeof(seg->rt_plan));
      memcpy(seg->rt_clear, list->render_pass_rt_clear,
             sizeof(seg->rt_clear));
      seg->depth_plan = list->render_pass_depth_plan;
      seg->stencil_plan = list->render_pass_stencil_plan;
      if (seg->depth_plan.load == D12_RP_LOAD_CLEAR)
         seg->depth_cleared = true;
      if (seg->stencil_plan.load == D12_RP_LOAD_CLEAR) {
         seg->stencil_cleared = true;
         seg->stencil_clear = list->render_pass_stencil_clear;
      }
   }

   seg->rt_count = rtv_count;
   unsigned bound_rts = 0;
   uint8_t bound_rt_mask = 0;
   unsigned rt_samples = 1;
   for (unsigned i = 0; i < rtv_count; i++) {
      const struct d12_descriptor *rtv = &rtvs[i];
      if (!rtv->resource)
         continue;

      if (rtv->kind != D12_DESC_RTV) {
         mesa_loge("render pass received an invalid RTV%u descriptor", i);
         list->state = D12_LIST_ERROR;
         return NULL;
      }

      enum pipe_format view_pf = d12_format_to_pipe(rtv->format);
      if (view_pf == PIPE_FORMAT_NONE || !ail_is_valid_pixel_format(view_pf) ||
          util_format_get_blocksize(view_pf) !=
             util_format_get_blocksize(rtv->resource->layout.format)) {
         mesa_loge("RTV%u format %u is incompatible with resource format %u",
                   i, (unsigned)rtv->format,
                   (unsigned)rtv->resource->layout.format);
         list->state = D12_LIST_ERROR;
         return NULL;
      }

      unsigned samples = MAX2(rtv->resource->layout.sample_count_sa, 1);
      if (bound_rts && samples != rt_samples) {
         mesa_loge("render targets have mismatched sample counts %u and %u",
                   rt_samples, samples);
         list->state = D12_LIST_ERROR;
         return NULL;
      }

      const unsigned layers = MAX2(rtv->size_B, 1);
      if (bound_rts && layers != seg->layers) {
         mesa_loge("render targets have mismatched layer counts %u and %u",
                   seg->layers, layers);
         list->state = D12_LIST_ERROR;
         return NULL;
      }

      seg->rt[i] = rtv->resource;
      d12_list_track_read_resource(list, rtv->resource);
      bound_rt_mask |= BITFIELD_BIT(i);
      seg->rt_address[i] = rtv->address ? rtv->address : rtv->resource->addr;
      seg->rt_view_format[i] = view_pf;
      seg->rt_level[i] = (uint8_t)rtv->texture_level;

      if (d12_debug()->validate_mrt) {
         const uint64_t bo_start = rtv->resource->reserved
                                      ? rtv->resource->addr
                                      : rtv->resource->bo->va->addr;
         const uint64_t bo_size = rtv->resource->reserved
                                     ? rtv->resource->sparse_size_B
                                     : rtv->resource->bo->size;
         const uint64_t bo_end = bo_start + bo_size;
         const uint64_t span =
            (uint64_t)rtv->resource->layout.layer_stride_B * (layers - 1) +
            ail_get_level_offset_B(&rtv->resource->layout,
                                   rtv->texture_level) +
            ail_get_level_size_B(&rtv->resource->layout,
                                 rtv->texture_level);
         if (seg->rt_address[i] < bo_start || seg->rt_address[i] >= bo_end ||
             span > bo_end - seg->rt_address[i]) {
            mesa_loge("MRT validate: RT%u address 0x%llx span 0x%llx is "
                      "outside BO [0x%llx, 0x%llx)",
                      i, (unsigned long long)seg->rt_address[i],
                      (unsigned long long)span, (unsigned long long)bo_start,
                      (unsigned long long)bo_end);
            list->state = D12_LIST_ERROR;
            return NULL;
         }

         mesa_logi("MRT validate: RT%u addr=0x%llx span=0x%llx "
                   "bo=[0x%llx,0x%llx) format=%u samples=%u",
                   i, (unsigned long long)seg->rt_address[i],
                   (unsigned long long)span, (unsigned long long)bo_start,
                   (unsigned long long)bo_end, view_pf, samples);
      }
      rt_samples = samples;
      seg->layers = layers;
      if (!bound_rts) {
         seg->width_px = u_minify(rtv->resource->layout.width_px,
                                  rtv->texture_level);
         seg->height_px = u_minify(rtv->resource->layout.height_px,
                                   rtv->texture_level);
      } else {
         seg->width_px =
            MIN2(seg->width_px,
                 u_minify(rtv->resource->layout.width_px,
                           rtv->texture_level));
         seg->height_px = MIN2(seg->height_px,
                               u_minify(rtv->resource->layout.height_px,
                                         rtv->texture_level));
      }
      bound_rts++;
   }

   if (dsv) {
      seg->ds = dsv->resource;
      d12_list_track_read_resource(list, dsv->resource);
      seg->ds_address = dsv->address;
      seg->stencil_address = dsv->counter_address;
      seg->ds_level = dsv->texture_plane;
      seg->dsv_flags = dsv->dsv_flags;

      /* The hardware seeds each tile's depth from ISP_BGOBJDEPTH in the
       * depth plane's own encoding, so a pass-declared clear can only be
       * encoded once the attachment's format is known. */
      if (seg->render_pass && seg->depth_plan.load == D12_RP_LOAD_CLEAR) {
         seg->isp_bgobjdepth =
            dsv->resource->layout.format == PIPE_FORMAT_Z16_UNORM
               ? _mesa_float_to_unorm(list->render_pass_depth_clear, 16)
               : fui(list->render_pass_depth_clear);
      }
      const unsigned ds_layers = MAX2(dsv->size_B, 1);
      const unsigned ds_samples =
         MAX2(dsv->resource->layout.sample_count_sa, 1);
      if (bound_rts && ds_samples != rt_samples) {
         mesa_loge("render targets and depth/stencil have mismatched sample "
                   "counts %u and %u", rt_samples, ds_samples);
         list->state = D12_LIST_ERROR;
         return NULL;
      }
      if (bound_rts && ds_layers != seg->layers) {
         mesa_loge("render targets and depth/stencil have mismatched layer "
                   "counts %u and %u", seg->layers, ds_layers);
         list->state = D12_LIST_ERROR;
         return NULL;
      }
      if (!bound_rts)
         seg->layers = ds_layers;

      const uint64_t z_start = dsv->resource->reserved
                                  ? dsv->resource->addr
                                  : dsv->resource->bo->va->addr;
      const uint64_t z_size = dsv->resource->reserved
                                 ? dsv->resource->sparse_size_B
                                 : dsv->resource->bo->size;
      const uint64_t z_end = z_start + z_size;
      const uint64_t z_span =
         (uint64_t)dsv->resource->layout.layer_stride_B * (ds_layers - 1) +
         ail_get_level_size_B(&dsv->resource->layout, seg->ds_level);
      if (seg->ds_address < z_start || seg->ds_address >= z_end ||
          z_span > z_end - seg->ds_address) {
         mesa_loge("DSV depth address 0x%llx span 0x%llx is outside BO "
                   "[0x%llx, 0x%llx)",
                   (unsigned long long)seg->ds_address,
                   (unsigned long long)z_span,
                   (unsigned long long)z_start,
                   (unsigned long long)z_end);
         list->state = D12_LIST_ERROR;
         return NULL;
      }
      if (dsv->resource->stencil_bo) {
         const uint64_t s_start = dsv->resource->stencil_bo->va->addr;
         const uint64_t s_end = s_start + dsv->resource->stencil_bo->size;
         const uint64_t s_span =
            (uint64_t)dsv->resource->stencil_layout.layer_stride_B *
               (ds_layers - 1) +
            ail_get_level_size_B(&dsv->resource->stencil_layout,
                                 seg->ds_level);
         if (seg->stencil_address < s_start ||
             seg->stencil_address >= s_end ||
             s_span > s_end - seg->stencil_address) {
            mesa_loge("DSV stencil address 0x%llx span 0x%llx is outside BO "
                      "[0x%llx, 0x%llx)",
                      (unsigned long long)seg->stencil_address,
                      (unsigned long long)s_span,
                      (unsigned long long)s_start,
                      (unsigned long long)s_end);
            list->state = D12_LIST_ERROR;
            return NULL;
         }
      }
      rt_samples = ds_samples;
      const unsigned ds_width =
         u_minify(dsv->resource->layout.width_px, seg->ds_level);
      const unsigned ds_height =
         u_minify(dsv->resource->layout.height_px, seg->ds_level);
      if (!seg->width_px) {
         seg->width_px = ds_width;
         seg->height_px = ds_height;
      } else {
         /* D3D12 permits mismatched RTV/DSV extents. Restrict the hardware
          * render area to their intersection, matching the minimum already
          * selected for multiple colour attachments and preventing either
          * ZLS or PBE from walking beyond its smaller image. */
         seg->width_px = MIN2(seg->width_px, ds_width);
         seg->height_px = MIN2(seg->height_px, ds_height);
      }
   }

   if (!seg->width_px) {
      uint32_t viewport_width;
      uint32_t viewport_height;

      if (!d12_viewport_render_extent(
             list->viewport_count, list->viewports,
             &viewport_width, &viewport_height)) {
         mesa_loge("render pass has invalid attachment-less viewport state");
         list->state = D12_LIST_ERROR;
         return NULL;
      }

      seg->width_px = (uint16_t)viewport_width;
      seg->height_px = (uint16_t)viewport_height;
   }

   /* Target-independent rasterization: with nothing attached, the render
    * command's sample count is purely the rasterization rate, so a forced
    * count is exactly expressible there.  d12_raster_rasterization_samples
    * keeps the attachment's own count whenever anything is attached, which
    * is the only shape AGX can encode. */
   if (pso && pso->raster.forced_sample_count) {
      rt_samples = d12_raster_rasterization_samples(
         pso->raster.forced_sample_count, rt_samples,
         bound_rts != 0 || dsv != NULL);
   }

   if (pso) {
      const enum pipe_format uniform_bound_format =
         d12_segment_uniform_rt_format(seg);
      const bool stable_pso_matches =
         pso->stable_unused_rt_layout &&
         (!bound_rts || uniform_bound_format == pso->tib.logical_format[0]);
      const bool seed_from_bound =
         !pso->rt_write_mask && uniform_bound_format != PIPE_FORMAT_NONE;

      seg->merge_unused_rts = stable_pso_matches || seed_from_bound;
      seg->rt_write_mask =
         seg->merge_unused_rts ? bound_rt_mask : pso->rt_write_mask;
      if (seed_from_bound) {
         enum pipe_format dense[AGX_MAX_RENDER_TARGETS];
         for (unsigned i = 0; i < ARRAY_SIZE(dense); i++)
            dense[i] = uniform_bound_format;
         seg->tib = agx_build_tilebuffer_layout(dense, ARRAY_SIZE(dense),
                                                rt_samples, seg->layers > 1);
      } else if ((bound_rts || dsv) && pso->tib.nr_samples != rt_samples) {
         /* Native D3D12 drivers are robust to a multisampled PSO on a
          * single-sampled attachment (only sample zero is live), and the
          * conformance suite also exercises the inverse out-of-spec case.
          * The attachment owns the actual tile/PBE allocation. Rebuild only
          * that layout at its sample count; the compiled shader keeps the
          * same formats and per-sample stride, so the overlapping sample-zero
          * ABI is unchanged and no descriptor address is invented. */
         seg->tib = agx_build_tilebuffer_layout(
            pso->tib.logical_format, rtv_count, rt_samples,
            seg->layers > 1);
         mesa_logw("render pass sample mismatch: PSO %u, attachments %u; "
                   "using attachment sample allocation",
                   pso->tib.nr_samples, rt_samples);
      } else {
         seg->tib = pso->tib;
      }

      if (d12_debug()->validate_mrt) {
         for (unsigned i = 0; i < rtv_count; i++) {
            if (seg->rt[i] &&
                seg->tib.logical_format[i] != seg->rt_view_format[i]) {
               mesa_loge("MRT validate: RT%u PSO format %u != view format "
                         "%u", i, seg->tib.logical_format[i],
                         seg->rt_view_format[i]);
               list->state = D12_LIST_ERROR;
               return NULL;
            }
         }
      }
   } else {
      seg->rt_write_mask = (uint8_t)BITFIELD_MASK(rtv_count);
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
      for (unsigned i = 0; i < rtv_count; i++)
         formats[i] = seg->rt_view_format[i];
      seg->tib = agx_build_tilebuffer_layout(formats, rtv_count, rt_samples,
                                             seg->layers > 1);
   }

   enum agx_bg_eot_op main_bg[AGX_MAX_RENDER_TARGETS] = {0};
   enum agx_bg_eot_op main_eot[AGX_MAX_RENDER_TARGETS] = {0};
   enum agx_bg_eot_op partial_bg[AGX_MAX_RENDER_TARGETS] = {0};
   enum agx_bg_eot_op partial_eot[AGX_MAX_RENDER_TARGETS] = {0};
   for (unsigned i = 0; i < rtv_count; i++) {
      if (!seg->rt[i])
         continue;
      if (!(seg->rt_write_mask & BITFIELD_BIT(i)))
         continue;

      /* A render pass states the tile traffic per attachment; outside one
       * the caller's single bg_op applies to all of them and every
       * attachment is written back. */
      const struct d12_render_pass_plan plan =
         seg->render_pass ? seg->rt_plan[i]
                          : d12_render_pass_plan_legacy(bg_op ==
                                                        AGX_BG_CLEAR);

      switch (plan.load) {
      case D12_RP_LOAD_CLEAR:
         main_bg[i] = AGX_BG_CLEAR;
         break;
      case D12_RP_LOAD_PRESERVE:
         main_bg[i] = AGX_BG_LOAD;
         break;
      case D12_RP_LOAD_NONE:
         main_bg[i] = AGX_BG_EOT_NONE;
         break;
      }
      main_eot[i] = plan.store == D12_RP_STORE_PRESERVE ? AGX_EOT_STORE
                                                        : AGX_BG_EOT_NONE;

      /* The partial pair is not the application's to choose. A tile that
       * spills mid-pass is stored and reloaded purely to free tile memory,
       * so both halves must move it whatever the pass asked for at its
       * boundaries -- otherwise the spill loses the work already done. */
      partial_bg[i] = AGX_BG_LOAD;
      partial_eot[i] = AGX_EOT_STORE;
   }

   seg->bg_usc = d12_build_bg_eot_usc(
      list, seg, color, main_bg, &seg->tib, &seg->bg_counts, &seg->g17p_bg_resources);
   seg->eot_usc = d12_build_bg_eot_usc(
      list, seg, NULL, main_eot, &seg->tib, &seg->eot_counts, &seg->g17p_eot_resources);
   seg->partial_bg_usc = d12_build_bg_eot_usc(
      list, seg, NULL, partial_bg, &seg->tib, &seg->partial_bg_counts, &seg->g17p_partial_bg_resources);
   seg->partial_eot_usc = d12_build_bg_eot_usc(
      list, seg, NULL, partial_eot, &seg->tib, &seg->partial_eot_counts, &seg->g17p_partial_eot_resources);
   if (!seg->bg_usc || !seg->eot_usc || !seg->partial_bg_usc ||
       !seg->partial_eot_usc)
      return NULL;

   if (!d12_list_check_stream_space(list, D12_VDM_BARRIER_BYTES))
      return NULL;

   seg->stream_start = list->stream_cur - d12_list_stream_map(list);
   if (!d12_list_emit_initial_vdm_barrier(list))
      return NULL;

   if (list->device->dev.chip == AGX_CHIP_G17P &&
       d12_g17p_baseline_graphics_enabled(&list->device->dev)) {
      if (!list->device->dev.g17p_render_context_bo) {
         list->state = D12_LIST_ERROR;
         return NULL;
      }
      /* The native render UAPI requires a resident, nonzero scissor base
       * even for PROCESS_EMPTY_TILES clears with no draw. Draws previously
       * allocated these arrays lazily, leaving clear-only segments at zero.
       * Use the same owned, zeroed, aligned arrays without consuming indices. */
      if (!d12_segment_ensure_viewport_arrays(list, seg))
         return NULL;
      struct d12_agx3_ppp_w_clamp_cull2 initial;
      const struct d12_agx3_ppp_w_clamp_cull2_fields fields = {.cull2_raw = 0x80};
      uint64_t compact;
      void *payload = d12_list_alloc_g17p_vdm(list, sizeof(initial.bytes), 4, &compact);
      if (!payload || !d12_agx3_encode_ppp_w_clamp_cull2(
             list->device->dev.chip, &fields, &initial)) {
         list->state = D12_LIST_ERROR;
         return NULL;
      }
      memcpy(payload, initial.bytes, initial.length);
      if (!agx_ppp_emit_state(list->device->dev.chip, &list->stream_cur,
                              compact, initial.length)) {
         list->state = D12_LIST_ERROR;
         return NULL;
      }
      /* Same direct-render sentinel as HK CS initialization: no guessed
       * context cycling and no mutable device-global PPP payload. */
      seg->agx3_active_context_selector_valid = true;
      seg->agx3_raw_active_context_selector = UINT32_MAX;
   } else    if (agx_chip_command_stream_generation(list->device->dev.chip) ==
       AGX_CS_GENERATION_HAL300) {
      struct d12_agx3_ppp_w_clamp_cull2 initial;
      const struct d12_agx3_ppp_w_clamp_cull2_fields fields = {
         /* Exact G17G/G17X default: W clamp 0.0, clamp-W enabled. */
         .w_clamp_raw = 0,
         .cull2_raw = UINT32_C(0x80),
      };
      struct d12_agx3_ppp_hal300_depth_stencil depth_stencil;
      const struct d12_agx3_ppp_hal300_depth_stencil_fields
         depth_stencil_fields = {
            /* Exact G17G/G17X fixed-layout initial render-state witness. */
            .fragment_control_raw = UINT32_C(0x000007aa),
            .fragment_front_face_raw = UINT32_C(0xe2e00000),
            .fragment_back_face_raw = UINT32_C(0xf0c00000),
            .fragment_front_stencil_raw = UINT32_C(0x00023020),
            .fragment_back_stencil_raw = UINT32_C(0x0001ffff),
         };
      if (!d12_agx3_encode_ppp_hal300_depth_stencil(
             list->device->dev.chip, &depth_stencil_fields,
             &depth_stencil) ||
          !d12_agx3_encode_ppp_w_clamp_cull2(list->device->dev.chip,
                                              &fields, &initial)) {
         mesa_loge("no grounded HAL300 initial PPP payload for %s",
                   agx_chip_name(list->device->dev.chip));
         list->state = D12_LIST_ERROR;
         return NULL;
      }

      const uint64_t depth_stencil_gpu = d12_list_upload(
         list, depth_stencil.bytes, depth_stencil.length,
         AGX_PPP_HEADER_ALIGN);
      const uint64_t gpu = d12_list_upload(
         list, initial.bytes, initial.length, AGX_PPP_HEADER_ALIGN);
      uint8_t records[2 * AGX_PPP_STATE_LENGTH];
      uint8_t *record = records;
      if (!depth_stencil_gpu || !gpu ||
          !agx_ppp_emit_state(list->device->dev.chip, &record,
                              depth_stencil_gpu, depth_stencil.length) ||
          !agx_ppp_emit_state(list->device->dev.chip, &record, gpu,
                              initial.length) ||
          record != records + sizeof(records)) {
         mesa_loge("failed to emit HAL300 initial PPP state for %s",
                   agx_chip_name(list->device->dev.chip));
         list->state = D12_LIST_ERROR;
         return NULL;
      }

      memcpy(list->stream_cur, records, sizeof(records));
      list->stream_cur += sizeof(records);
   } else {
      struct AGX_PPP_HEADER present = {
         .w_clamp = true,
         .occlusion_query_2 = true,
         .output_unknown = true,
         .varying_word_2 = true,
         .viewport_count = 1,
      };
      size_t size = agx_ppp_update_size(&present);
      uint64_t gpu = d12_list_upload_zero(list, size, AGX_PPP_HEADER_ALIGN);
      uint8_t *cpu = d12_list_data_cpu(list, gpu);
      if (!cpu)
         return NULL;

      struct agx_ppp_update ppp = agx_new_ppp_update(
         (struct agx_ptr){.cpu = cpu, .gpu = gpu}, size, &present);
      agx_ppp_push(&ppp, W_CLAMP, cfg)
         cfg.w_clamp = 1e-10;
      agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY_2, cfg);
      agx_ppp_push(&ppp, OUTPUT_UNKNOWN, cfg);
      agx_ppp_push(&ppp, VARYING_2, cfg);
      if (!agx_ppp_legacy_payload_supported(list->device->dev.chip) ||
          !agx_ppp_fini(list->device->dev.chip, &list->stream_cur, &ppp)) {
         mesa_loge("no grounded legacy PPP payload/finalizer for %s",
                   agx_chip_name(list->device->dev.chip));
         list->state = D12_LIST_ERROR;
         return NULL;
      }
   }

   list->segment_count += 2;
   list->render_open = true;
   return seg;
   }

static bool
d12_pack_texture_generation(enum agx_descriptor_generation generation,
                            void *out, const struct AGX_TEXTURE *legacy,
                            unsigned semantic_sample_count)
{
   if (!out || !legacy)
      return false;

   if (generation == AGX_DESCRIPTOR_GENERATION_7 ||
       generation == AGX_DESCRIPTOR_GENERATION_8) {
      AGX_TEXTURE_pack(out, legacy);
      return true;
   }

   const bool has_levels = legacy->first_level || legacy->last_level ||
                           legacy->unk_mipmapped;
   const bool extended = legacy->extended || has_levels;
   if (legacy->address >> 48 ||
       (legacy->compressed && legacy->acceleration_buffer >> 48))
      return false;

   if (generation == AGX_DESCRIPTOR_GENERATION_9) {
      const bool multisampled =
         legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
         legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
      uint64_t sample_field;
      if (!d12_native_texture_gen9_sample_field(
             multisampled, semantic_sample_count, legacy->samples,
             &sample_field))
         return false;

      const unsigned sample_selector =
         (sample_field >> D12_NATIVE_TEXTURE_GEN9_SAMPLES_SHIFT) &
         D12_NATIVE_TEXTURE_GEN9_SAMPLES_MASK;

      struct AGX_TEXTURE_GEN9 cfg = {
         .dimension = legacy->dimension,
         .layout = legacy->layout,
         .channels = legacy->channels,
         .type = legacy->type,
         .swizzle_r = legacy->swizzle_r,
         .swizzle_g = legacy->swizzle_g,
         .swizzle_b = legacy->swizzle_b,
         .swizzle_a = legacy->swizzle_a,
         .width = legacy->width,
         .height = legacy->height,
         .samples = sample_selector,
         .has_levels = has_levels,
         /* RE proves mode/compression moved together to word 0[60:63]. */
         .state_control = legacy->mode | (legacy->compression << 2),
         .address = legacy->address,
         .srgb = legacy->srgb,
         .srgb_2_channel = legacy->srgb_2_channel,
         .stride = legacy->layout == AGX_LAYOUT_LINEAR ? legacy->stride : 0,
         .depth = legacy->layout == AGX_LAYOUT_LINEAR ? 1 : legacy->depth,
         .page_aligned_layers = legacy->page_aligned_layers,
         .extended = extended,
         .acceleration_buffer =
            legacy->compressed && legacy->extended
               ? legacy->acceleration_buffer : 0,
         .depth_linear = legacy->layout == AGX_LAYOUT_LINEAR &&
                         legacy->extended ? legacy->depth_linear : 1,
         .layer_stride_linear = legacy->layout == AGX_LAYOUT_LINEAR &&
                                legacy->extended
                                   ? legacy->layer_stride_linear : 0,
         .first_level = legacy->first_level,
         .last_level = legacy->last_level,
      };
      if (!cfg.width || cfg.width > 16384 || !cfg.height ||
          cfg.height > 16384)
         return false;
      memset(out, 0, AGX_TEXTURE_GEN9_LENGTH);
      AGX_TEXTURE_GEN9_pack(out, &cfg);
      return true;
   }

   if (generation == AGX_DESCRIPTOR_GENERATION_10) {
      struct AGX_TEXTURE_GEN10 cfg = {
         .dimension = legacy->dimension,
         .layout = legacy->layout,
         .channels = legacy->channels,
         .type = legacy->type,
         .swizzle_r = legacy->swizzle_r,
         .swizzle_g = legacy->swizzle_g,
         .swizzle_b = legacy->swizzle_b,
         .swizzle_a = legacy->swizzle_a,
         .width = legacy->width,
         .height = legacy->height,
         .samples = legacy->samples,
         .unk_mipmapped = legacy->unk_mipmapped,
         .compressed = legacy->compressed,
         .image_mode_enable = legacy->mode != AGX_IMAGE_MODE_NORMAL,
         .image_mode = legacy->mode,
         .address = legacy->address,
         .srgb = legacy->srgb,
         .srgb_2_channel = legacy->srgb_2_channel,
         .stride = legacy->layout == AGX_LAYOUT_LINEAR ? legacy->stride : 0,
         .depth = legacy->layout == AGX_LAYOUT_LINEAR ? 1 : legacy->depth,
         .page_aligned_layers = legacy->page_aligned_layers,
         .extended = extended,
         .acceleration_buffer =
            legacy->compressed && legacy->extended
               ? legacy->acceleration_buffer : 0,
         .depth_linear = legacy->layout == AGX_LAYOUT_LINEAR &&
                         legacy->extended ? legacy->depth_linear : 1,
         .layer_stride_linear = legacy->layout == AGX_LAYOUT_LINEAR &&
                                legacy->extended
                                   ? legacy->layer_stride_linear : 0,
         .first_level = legacy->first_level,
         .last_level = legacy->last_level,
      };
      if (!cfg.width || cfg.width > 32768 || !cfg.height ||
          cfg.height > 32768)
         return false;
      memset(out, 0, AGX_TEXTURE_GEN10_LENGTH);
      AGX_TEXTURE_GEN10_pack(out, &cfg);
      return true;
   }

   return false;
}

static bool
d12_pack_pbe_generation(enum agx_descriptor_generation generation, void *out,
                        const struct AGX_PBE *legacy)
{
   if (!out || !legacy)
      return false;

   if (generation == AGX_DESCRIPTOR_GENERATION_7 ||
       generation == AGX_DESCRIPTOR_GENERATION_8) {
      AGX_PBE_pack(out, legacy);
      return true;
   }

   if (generation != AGX_DESCRIPTOR_GENERATION_9 &&
       generation != AGX_DESCRIPTOR_GENERATION_10)
      return false;

   if (legacy->dimension != AGX_TEXTURE_DIMENSION_2D ||
       (legacy->layout != AGX_LAYOUT_TWIDDLED &&
        legacy->layout != AGX_LAYOUT_GPU) ||
       legacy->unk_52 || legacy->rotate_90 || legacy->flip_vertical ||
       legacy->samples != AGX_SAMPLE_COUNT_2 || legacy->unk_mipmapped ||
       legacy->compressed || legacy->mode != AGX_IMAGE_MODE_NORMAL ||
       legacy->level != 0 || legacy->levels != 1 || legacy->layers != 1 ||
       legacy->srgb || legacy->extended || legacy->stride ||
       legacy->acceleration_buffer || legacy->depth_linear != 1 ||
       legacy->layer_stride_linear || legacy->level_offset_sw ||
       legacy->aligned_width_msaa_sw || legacy->sample_count_log2_sw ||
       legacy->tile_width_sw != 1 || legacy->tile_height_sw != 1 ||
       legacy->layer_stride_sw || legacy->buffer_offset_sw)
      return false;

   const struct agx_pbe_agx3_2d_fields fields = {
      .layout = legacy->layout,
      .channels = legacy->channels,
      .type = legacy->type,
      .swizzle_r = legacy->swizzle_r,
      .swizzle_g = legacy->swizzle_g,
      .swizzle_b = legacy->swizzle_b,
      .swizzle_a = legacy->swizzle_a,
      .width = legacy->width,
      .height = legacy->height,
      .buffer = legacy->buffer,
      .page_aligned_layers = legacy->page_aligned_layers,
   };
   struct agx_pbe_agx3_descriptor packed;
   if (!d12_native_pbe_agx3_2d_pack(generation, &fields, &packed))
      return false;

   memcpy(out, packed.bytes, sizeof(packed.bytes));
   return true;
}

static void
d12_refresh_pbe_descriptor_agx2(struct agx_pbe_packed *out,
                                const struct d12_descriptor *d)
{
   agx_set_null_pbe(out);

   if (d->kind != D12_DESC_UAV || !d->resource)
      return;

   struct d12_resource *res = d->resource;
   enum pipe_format pf = d12_format_to_pipe(d->format);
   if (pf == PIPE_FORMAT_NONE || !ail_is_valid_pixel_format(pf) ||
       !ail_pixel_format[pf].texturable ||
       util_format_is_depth_or_stencil(pf))
      return;

   const struct util_format_description *fdesc =
      util_format_description(pf);
   const unsigned blocksize = util_format_get_blocksize(pf);

   if (res->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
      if (d->buffer_kind != D12_BUFFER_VIEW_TYPED || !blocksize ||
          !d->buffer_count)
         return;

      uint64_t base = d->address & ~0xfull;
      unsigned tail_B = d->address & 0xf;
      if (tail_B % blocksize)
         return;

      unsigned elements = d->buffer_count;
      agx_pack(out, PBE, cfg) {
         cfg.dimension = AGX_TEXTURE_DIMENSION_2D;
         cfg.layout = AGX_LAYOUT_LINEAR;
         cfg.channels = ail_pixel_format[pf].channels;
         cfg.type = ail_pixel_format[pf].type;
         cfg.srgb = util_format_is_srgb(pf);

         for (unsigned i = 0; i < fdesc->nr_channels && i < 4; i++) {
            if (fdesc->swizzle[i] == PIPE_SWIZZLE_X)
               cfg.swizzle_r = i;
            else if (fdesc->swizzle[i] == PIPE_SWIZZLE_Y)
               cfg.swizzle_g = i;
            else if (fdesc->swizzle[i] == PIPE_SWIZZLE_Z)
               cfg.swizzle_b = i;
            else if (fdesc->swizzle[i] == PIPE_SWIZZLE_W)
               cfg.swizzle_a = i;
         }

         cfg.buffer = base;
         cfg.buffer_offset_sw = tail_B / blocksize;
         cfg.width = AGX_TEXTURE_BUFFER_WIDTH;
         cfg.height = DIV_ROUND_UP(elements, cfg.width);
         cfg.level = 0;
         cfg.levels = 1;
         cfg.stride = (cfg.width * blocksize) - 4;
         cfg.layers = 1;
      }
      return;
   }

   const bool is_3d =
      res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D &&
      d->texture_uav_dimension == D3D12_UAV_DIMENSION_TEXTURE3D;
   if ((res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D &&
        res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D && !is_3d) ||
       !res->layout.size_B || !res->layout.writeable_image ||
       res->layout.tiling == AIL_TILING_LINEAR)
      return;

   const struct ail_layout *layout = &res->layout;
   const unsigned level = d->texture_level;
   const unsigned layers = MAX2(d->texture_layers, 1);
   const bool array =
      d->texture_uav_dimension == D3D12_UAV_DIMENSION_TEXTURE1DARRAY ||
      d->texture_uav_dimension == D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
   const unsigned descriptor_level = is_3d ? 0 : level;
   const unsigned width = is_3d ? u_minify(layout->width_px, level)
                                : layout->width_px;
   const unsigned height = is_3d ? u_minify(layout->height_px, level)
                                 : layout->height_px;

   agx_pack(out, PBE, cfg) {
      cfg.dimension = is_3d ? AGX_TEXTURE_DIMENSION_3D
                      : array ? AGX_TEXTURE_DIMENSION_2D_ARRAY
                              : AGX_TEXTURE_DIMENSION_2D;
      cfg.layout = agx_translate_layout(layout->tiling);
      cfg.channels = ail_pixel_format[pf].channels;
      cfg.type = ail_pixel_format[pf].type;
      cfg.srgb = util_format_is_srgb(pf);

      for (unsigned i = 0; i < fdesc->nr_channels && i < 4; i++) {
         if (fdesc->swizzle[i] == PIPE_SWIZZLE_X)
            cfg.swizzle_r = i;
         else if (fdesc->swizzle[i] == PIPE_SWIZZLE_Y)
            cfg.swizzle_g = i;
         else if (fdesc->swizzle[i] == PIPE_SWIZZLE_Z)
            cfg.swizzle_b = i;
         else if (fdesc->swizzle[i] == PIPE_SWIZZLE_W)
            cfg.swizzle_a = i;
      }

      cfg.buffer = d->address;
      cfg.width = width;
      cfg.height = height;
      cfg.level = descriptor_level;
      cfg.levels = is_3d ? 1 : layout->levels;
      cfg.layers = layers;
      cfg.page_aligned_layers = layout->page_aligned_layers;

      /* PBE words 4/5 are software sideband when the architectural
       * descriptor is not extended. libagx_image_texel_address consumes
       * these fields for tiled image atomics. */
      if (!cfg.extended) {
         cfg.level_offset_sw =
            is_3d ? 0 : ail_get_level_offset_B(layout, level);
         cfg.tile_width_sw = layout->tilesize_el[level].width_el;
         cfg.tile_height_sw = layout->tilesize_el[level].height_el;
         cfg.layer_stride_sw = layout->layer_stride_B;
      }
   }
}

bool
d12_refresh_pbe_descriptor(struct d12_device *device, void *out,
                           const struct d12_descriptor *d)
{
   if (!device || !out || !d)
      return false;

   const enum agx_descriptor_generation generation =
      agx_chip_descriptor_generation(device->dev.chip);
   if (generation == AGX_DESCRIPTOR_GENERATION_7 ||
       generation == AGX_DESCRIPTOR_GENERATION_8) {
      d12_refresh_pbe_descriptor_agx2(out, d);
      return true;
   }

   struct d12_descriptor shadow = *d;
   if (generation == AGX_DESCRIPTOR_GENERATION_9 ||
       generation == AGX_DESCRIPTOR_GENERATION_10) {
      /* The legacy state builder is retained as the single semantic source,
       * but its packer has only a 36-bit shifted address field. Feed it the
       * low 40 address bits, then restore the proven 44-bit Gen9/10 address
       * below before the generation packer runs. */
      shadow.address &= (UINT64_C(1) << 40) - 1;
   }

   struct agx_pbe_packed legacy;
   d12_refresh_pbe_descriptor_agx2(&legacy, &shadow);
   struct agx_pbe_packed null_legacy;
   agx_set_null_pbe(&null_legacy);
   struct AGX_PBE unpacked = {0};
   if (!AGX_PBE_unpack(NULL, (const uint8_t *)&legacy, &unpacked))
      return false;

   if (memcmp(&legacy, &null_legacy, sizeof(legacy)) == 0)
      unpacked.dimension = AGX_TEXTURE_DIMENSION_2D;

   if ((generation == AGX_DESCRIPTOR_GENERATION_9 ||
       generation == AGX_DESCRIPTOR_GENERATION_10) &&
       d->kind == D12_DESC_UAV && d->resource &&
       memcmp(&legacy, &null_legacy, sizeof(legacy)) != 0) {
      if (d->resource->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
          (d->address & 0xf)) {
         d12_set_null_pbe_descriptor(device, out);
         return false;
      }
      unpacked.buffer = d->address & ~UINT64_C(0xf);

      /* AGX_PBE_unpack exposes every interpretation of overlapping Gen7/8
       * words at once. Only the tiled, single-level interpretation survives
       * the bounded AGX3 gate. Canonicalize its linear/compression aliases and
       * Mesa's Gen7/8-only software-addressing sideband. AGX3 lowering gates
       * every software-sideband consumer above.
       */
      unpacked.stride = 0;
      unpacked.acceleration_buffer = 0;
      unpacked.depth_linear = 1;
      unpacked.layer_stride_linear = 0;
      unpacked.level_offset_sw = 0;
      unpacked.aligned_width_msaa_sw = 0;
      unpacked.sample_count_log2_sw = 0;
      unpacked.tile_width_sw = 1;
      unpacked.tile_height_sw = 1;
      unpacked.layer_stride_sw = 0;
      unpacked.buffer_offset_sw = 0;
   }

   if (d12_pack_pbe_generation(generation, out, &unpacked))
      return true;

   /* Descriptor writes cannot report an HRESULT. Replace an unrepresentable
    * state with the generation's robust null before reporting failure to
    * command recording/refresh callers, so no stale native state survives. */
   d12_set_null_pbe_descriptor(device, out);
   return false;
}

bool
d12_set_null_pbe_descriptor(struct d12_device *device, void *out)
{
   if (!device || !out)
      return false;

   const enum agx_descriptor_generation generation =
      agx_chip_descriptor_generation(device->dev.chip);
   if (generation == AGX_DESCRIPTOR_GENERATION_7 ||
       generation == AGX_DESCRIPTOR_GENERATION_8) {
      agx_set_null_pbe(out);
      return true;
   }

   struct agx_pbe_packed legacy;
   agx_set_null_pbe(&legacy);
   struct AGX_PBE unpacked = {0};
   if (!AGX_PBE_unpack(NULL, (const uint8_t *)&legacy, &unpacked))
      return false;
   unpacked.dimension = AGX_TEXTURE_DIMENSION_2D;

   return d12_pack_pbe_generation(generation, out, &unpacked);
}

/*
 * Build the two tables a shader reaches its resources through, and return
 * the addresses to publish in uniform registers.
 *
 * - The ROOT TABLE is 64-bit addresses indexed by binding slot: buffers,
 *   read with one indirection.
 * - The TEXTURE HEAP is hardware texture descriptors at the same slot
 *   index scaled by the descriptor size. AGX does not reach a texture by
 *   address; it reads a descriptor, so one must exist in memory even
 *   though D3D12's own descriptor never becomes memory.
 *
 * BOTH out-parameters are addresses of an 8-byte cell CONTAINING the
 * table's address, not the table itself. agx_usc_uniform fills uniform
 * registers FROM memory at the address it is given, so handing it a table
 * directly loads the table's first eight bytes into the register pair.
 * For the texture heap that is descriptor word 0, which the hardware then
 * uses as the heap base -- a fault at an address that looks like garbage
 * but is exactly the descriptor's own first dword.
 */
static enum agx_texture_dimension
d12_native_texture_dimension(const struct d12_descriptor *d,
                             unsigned samples);
static unsigned d12_native_texture_layers(const struct d12_descriptor *d);

static enum pipe_format
d12_native_texture_format(const struct d12_descriptor *d)
{
   struct d12_resource *res = d->resource;
   if (!res)
      return PIPE_FORMAT_NONE;

   if (d->texture_plane == 1)
      return PIPE_FORMAT_R8_UINT;

   /* ZLS storage formats are not legal AGX sampled-texture formats.  Depth
    * SRVs address the same plane but interpret it through the matching colour
    * format.  This is especially important for the X8/X24 view spellings,
    * which intentionally have no standalone allocation entry in the DXGI
    * table. */
   if (res->is_depth_stencil) {
      switch (res->layout.format) {
      case PIPE_FORMAT_Z32_FLOAT: return PIPE_FORMAT_R32_FLOAT;
      case PIPE_FORMAT_Z16_UNORM: return PIPE_FORMAT_R16_UNORM;
      default: return PIPE_FORMAT_NONE;
      }
   }

   if (d->kind == D12_DESC_UAV || d->format != DXGI_FORMAT_UNKNOWN)
      return d12_format_to_pipe(d->format);

   return res->layout.format;
}

static bool
d12_sparse_texture_map_address(const struct d12_descriptor *d,
                               uint64_t *address);

bool
d12_descriptor_resource_span(const struct d12_descriptor *d, uint64_t *base,
                             uint64_t *size)
{
   struct d12_resource *res = d->resource;
   if (!res)
      return false;

   if (res->reserved) {
      if (d->texture_plane == 1)
         return false;
      *base = res->addr +
              ((d->kind == D12_DESC_SRV || d->kind == D12_DESC_CBV)
                  ? res->device->dev.sparse_ro_offset
                  : 0);
      *size = res->sparse_size_B;
      return true;
   }

   struct agx_bo *bo = d->texture_plane == 1 && res->stencil_bo
                          ? res->stencil_bo
                          : res->bo;
   if (!bo)
      return false;
   *base = bo->va->addr;
   *size = bo->size;
   return true;
}

static void
d12_apply_texture_shape(struct d12_descriptor *d, unsigned shape)
{
   if (!d || !d->resource || shape == D12_TEXTURE_SHAPE_NONE ||
       (d->kind != D12_DESC_SRV && d->kind != D12_DESC_UAV))
      return;

   const bool array = shape == D12_TEXTURE_SHAPE_1D_ARRAY ||
                      shape == D12_TEXTURE_SHAPE_2D_ARRAY;
   const bool one_d =
      shape == D12_TEXTURE_SHAPE_1D || shape == D12_TEXTURE_SHAPE_1D_ARRAY;

   if (d->kind == D12_DESC_SRV) {
      d->texture_srv_dimension =
         one_d ? (array ? D3D12_SRV_DIMENSION_TEXTURE1DARRAY
                        : D3D12_SRV_DIMENSION_TEXTURE1D)
               : (array ? D3D12_SRV_DIMENSION_TEXTURE2DARRAY
                        : D3D12_SRV_DIMENSION_TEXTURE2D);
      if (!array)
         d->texture_srv_layers = 1;
   } else {
      d->texture_uav_dimension =
         one_d ? (array ? D3D12_UAV_DIMENSION_TEXTURE1DARRAY
                        : D3D12_UAV_DIMENSION_TEXTURE1D)
               : (array ? D3D12_UAV_DIMENSION_TEXTURE2DARRAY
                        : D3D12_UAV_DIMENSION_TEXTURE2D);
      if (!array)
         d->texture_layers = 1;
   }
}

/* One slot of the four dense root planes: resource address, UAV counter,
 * BUFINFO and sparse residency. Shared by the full upload and the re-root
 * fast path so both publish identical bits. */
static bool
d12_root_plane_slot(struct d12_cmd_list *list, const struct d12_descriptor *d,
                    const struct d12_descriptor *source, unsigned i,
                    uint64_t *null_buf, uint64_t *null_counters,
                    uint64_t *addrs)
{
   addrs[D12_ROOT_RESOURCE_BASE + i] = d->address;
   addrs[D12_ROOT_COUNTER_BASE + i] = d->counter_address;
   addrs[D12_ROOT_BUFINFO_BASE + i] = 0;
   addrs[D12_ROOT_RESIDENCY_BASE + i] =
      d12_descriptor_sparse_residency_address(d);

   if (d->resource &&
       d->resource->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
       (d->kind == D12_DESC_SRV || d->kind == D12_DESC_UAV)) {
      if (!source) {
         /* A bare root SRV/UAV has no declared view extent. The resource
          * pointer exists only to publish sparse residency metadata. */
         addrs[D12_ROOT_BUFINFO_BASE + i] = UINT32_MAX;
      } else {
         switch (d->buffer_kind) {
         case D12_BUFFER_VIEW_RAW:
         case D12_BUFFER_VIEW_TYPED:
         case D12_BUFFER_VIEW_STRUCTURED:
            addrs[D12_ROOT_BUFINFO_BASE + i] = d->buffer_count;
            break;
         default:
            break;
         }
         if (d->buffer_kind != D12_BUFFER_VIEW_NONE)
            addrs[D12_ROOT_BUFINFO_BASE + i] |=
               (uint64_t)d->buffer_stride_B << 32;
      }
   } else if (d->resource && d->kind == D12_DESC_SRV &&
              (d->resource->desc.Dimension ==
                  D3D12_RESOURCE_DIMENSION_TEXTURE1D ||
               d->resource->desc.Dimension ==
                  D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
               d->resource->desc.Dimension ==
                  D3D12_RESOURCE_DIMENSION_TEXTURE3D)) {
      /* Texture SRVs use the low BUFINFO word for the runtime per-view
       * minimum LOD clamp. Buffer users interpret the same plane as a
       * count, and the resource dimension keeps the meanings disjoint. */
      uint32_t info = d->texture_txf_min_level |
                      (d->texture_min_lod_oob ? 0x80000000u : 0);
      addrs[D12_ROOT_BUFINFO_BASE + i] =
         fui(d->texture_min_lod) | (uint64_t)info << 32;
   } else if (d->address && !d->resource &&
              (d->kind == D12_DESC_SRV || d->kind == D12_DESC_UAV)) {
      /* Root SRV/UAV descriptors carry only a GPU address, so D3D12
       * provides no view extent to publish.  Treat their element count
       * as unbounded instead of zero: structured-store robustness uses
       * this plane for descriptor-table views, and must not accidentally
       * discard every valid access through a bare root descriptor.  Null
       * root descriptors retain count zero because their address is zero
       * until the safe null-buffer substitution below.
       */
      addrs[D12_ROOT_BUFINFO_BASE + i] = UINT32_MAX;
   }

   if (!addrs[D12_ROOT_RESOURCE_BASE + i]) {
      if (!*null_buf && !(*null_buf = d12_list_null_buffer(list)))
         return false;
      addrs[D12_ROOT_RESOURCE_BASE + i] = *null_buf;
   }

   if (!addrs[D12_ROOT_COUNTER_BASE + i]) {
      if (!*null_counters && !(*null_counters = d12_list_null_counters(list)))
         return false;
      addrs[D12_ROOT_COUNTER_BASE + i] = *null_counters + i * sizeof(uint32_t);
   }
   return true;
}

/* A slot that only feeds the root planes: unbound, root constants, a CBV,
 * or a bare root SRV/UAV with no resource behind it. Such slots never pack
 * a texture or PBE descriptor, so replacing one leaves every uploaded heap
 * valid and only the planes need re-rooting. */
static bool
d12_descriptor_is_plane_only(const struct d12_descriptor *d)
{
   if (d->resource)
      return false;
   switch (d->kind) {
   case D12_DESC_CBV:
   case D12_DESC_SRV:
   case D12_DESC_UAV:
      return true;
   default:
      return !d->address && !d->counter_address;
   }
}

static void
d12_binding_upload_cache_outputs(const struct d12_binding_upload_cache *cache,
                                 uint64_t *out_root, uint64_t *out_textures,
                                 uint64_t *out_pbes, uint64_t *out_heap,
                                 uint64_t *out_heap_textures,
                                 uint64_t *out_heap_pbes,
                                 uint64_t *out_heap_samplers,
                                 uint64_t *out_tables,
                                 uint32_t *out_table_count)
{
   *out_root = cache->root;
   *out_textures = cache->textures;
   *out_pbes = cache->pbes;
   *out_heap = cache->heap;
   *out_heap_textures = cache->heap_textures;
   *out_heap_pbes = cache->heap_pbes;
   *out_heap_samplers = cache->heap_samplers;
   if (out_tables)
      *out_tables = cache->tables;
   if (out_table_count)
      *out_table_count = cache->table_count;
}

/* Re-root a cached binding upload after plane-only slots changed: patch
 * those slots in the cached dense planes, upload fresh planes and headers,
 * and point the root at them. The texture and PBE heaps, their home cells
 * and the bindless heap cells are reused unchanged. */
static bool
d12_list_reroot_bindings(struct d12_cmd_list *list,
                         struct d12_binding_upload_cache *cache,
                         const struct d12_descriptor *raw_table,
                         const struct d12_descriptor *const *sources,
                         const struct d12_descriptor_heap *bound_heap)
{
   const unsigned root_count = cache->table_count;
   uint64_t null_buf = 0, null_counters = 0;

   for (unsigned i = 0; i < root_count; i++) {
      if (memcmp(&cache->table[i], &raw_table[i], sizeof(raw_table[i])) == 0)
         continue;
      cache->table[i] = raw_table[i];

      uint64_t addrs[D12_ROOT_TABLE_ENTRIES];
      if (!d12_root_plane_slot(list, &raw_table[i], sources[i], i, &null_buf,
                               &null_counters, addrs))
         return false;
      cache->compact[i] = addrs[D12_ROOT_RESOURCE_BASE + i];
      cache->compact[root_count + i] = addrs[D12_ROOT_COUNTER_BASE + i];
      cache->compact[2 * root_count + i] = addrs[D12_ROOT_BUFINFO_BASE + i];
      cache->compact[3 * root_count + i] = addrs[D12_ROOT_RESIDENCY_BASE + i];
   }

   uint64_t tables_gpu = d12_list_upload(
      list, cache->compact, 4 * root_count * sizeof(uint64_t), 8);
   if (!tables_gpu)
      return false;

   if (!null_buf && !(null_buf = d12_list_null_buffer(list)))
      return false;
   uint64_t discard_buf = d12_list_discard_buffer(list);
   if (!discard_buf)
      return false;

   uint64_t metadata_header[] = {
      tables_gpu + root_count * sizeof(uint64_t),
      tables_gpu + 2 * root_count * sizeof(uint64_t),
      null_buf,
      discard_buf,
      tables_gpu + 3 * root_count * sizeof(uint64_t),
      bound_heap && bound_heap->residency_bo
         ? bound_heap->residency_bo->va->addr : 0,
      list->device->dev.sparse_ro_offset,
   };
   uint64_t metadata_gpu =
      d12_list_upload(list, metadata_header, sizeof(metadata_header), 8);
   if (!metadata_gpu)
      return false;

   /* The sampler-table plane and the PBE null offset are unchanged; read
    * them back from the cached header rather than recomputing. */
   const uint64_t *old_header_cell =
      (const uint64_t *)d12_list_data_cpu(list, cache->root);
   const uint64_t *old_header =
      old_header_cell
         ? (const uint64_t *)d12_list_data_cpu(list, *old_header_cell)
         : NULL;
   if (!old_header)
      return false;
   uint64_t header[] = {
      tables_gpu,
      metadata_gpu,
      old_header[2],
      old_header[3],
   };
   static_assert(sizeof(header) == D12_ROOT_HEADER_SIZE,
                 "root header ABI mismatch");
   uint64_t header_gpu = d12_list_upload(list, header, sizeof(header), 8);
   if (!header_gpu)
      return false;
   uint64_t root = d12_list_upload(list, &header_gpu, sizeof(header_gpu), 8);
   if (!root)
      return false;

   /* Mutable shader-visible table slots are refreshed at submission into
    * the planes of every upload that flattened them. */
   for (unsigned i = 0; i < root_count; i++) {
      if (!sources[i])
         continue;
      struct d12_descriptor_fixup fixup = {
         .source = sources[i],
         .tables_gpu = tables_gpu,
         .slot = i,
         .table_count = root_count,
         .texture_shape =
            list->pso ? list->pso->texture_shapes[i] : D12_TEXTURE_SHAPE_NONE,
      };
      util_dynarray_append_typed(&list->descriptor_fixups,
                                 struct d12_descriptor_fixup, fixup);
   }

   cache->root = root;
   cache->tables = tables_gpu;
   return true;
}

static bool
d12_list_upload_bindings(struct d12_cmd_list *list, bool compute,
                         uint64_t *out_root, uint64_t *out_textures,
                         uint64_t *out_pbes, uint64_t *out_heap,
                         uint64_t *out_heap_textures,
                         uint64_t *out_heap_pbes,
                         uint64_t *out_heap_samplers, bool force_unique,
                         uint64_t *out_tables, uint32_t *out_table_count)
{
   const struct d12_descriptor *raw_table =
      compute ? list->compute_root_table : list->graphics_root_table;
   const struct d12_descriptor *const *sources =
      compute ? list->compute_root_source : list->graphics_root_source;
   const struct d12_descriptor *sampler_table =
      compute ? list->compute_sampler_table : list->graphics_sampler_table;
   const struct d12_root_signature *rs =
      compute ? list->compute_rs : list->graphics_rs;
   struct d12_descriptor_heap *bound_heap = list->resource_descriptor_heap;
   struct d12_binding_upload_cache *ways =
      list->binding_upload_cache[compute ? 1 : 0];
   struct d12_binding_upload_cache *cache = NULL;
   const uint32_t generation = list->root_generation[compute];

   /* The common case: nothing bound changed since one of the cached
    * uploads was built or last validated. */
   for (unsigned w = 0; !force_unique && w < D12_BINDING_CACHE_WAYS; w++) {
      struct d12_binding_upload_cache *e = &ways[w];
      if (e->valid && e->generation == generation && e->rs == rs &&
          e->pso == list->pso && e->resource_heap == bound_heap &&
          e->sampler_heap == list->sampler_descriptor_heap) {
         d12_binding_upload_cache_outputs(e, out_root, out_textures,
                                          out_pbes, out_heap,
                                          out_heap_textures, out_heap_pbes,
                                          out_heap_samplers, out_tables,
                                          out_table_count);
         return true;
      }
   }

   bool validate = d12_debug()->validate_bindings;
   const enum agx_descriptor_generation descriptor_generation =
      agx_chip_descriptor_generation(list->device->dev.chip);
   struct d12_native_descriptor_layout native_layout;
   if (!d12_native_descriptor_layout((uint32_t)descriptor_generation,
                                     &native_layout)) {
      mesa_loge("binding upload: descriptor generation %u has no proven "
                "native heap layout", (unsigned)descriptor_generation);
      list->state = D12_LIST_ERROR;
      return false;
   }
   if (bound_heap &&
       (bound_heap->texture_stride_B != native_layout.texture_stride_B ||
        bound_heap->pbe_stride_B != native_layout.pbe_stride_B)) {
      mesa_loge("binding upload: bound native descriptor heap layout does "
                "not match the device generation");
      list->state = D12_LIST_ERROR;
      return false;
   }

   uint8_t texture_shapes[D12_MAX_RESOURCE_BINDINGS] = {0};
   if (list->pso)
      memcpy(texture_shapes, list->pso->texture_shapes, sizeof(texture_shapes));

   /* The generation moved but the contents may still match (a root
    * parameter rewritten with the same value), or only root constants and
    * bare root descriptors may differ. Both are decided per slot, over the
    * slots the bound root signature can actually address: a slot past its
    * binding map is unreachable by the bound shaders. */
   const unsigned compare_slots = rs ? MAX2(rs->map.resource_count, 1) : 1;
   const unsigned compare_samplers = rs ? rs->map.sampler_count : 0;
   for (unsigned w = 0; !force_unique && w < D12_BINDING_CACHE_WAYS; w++) {
      struct d12_binding_upload_cache *e = &ways[w];
      if (!e->valid || e->rs != rs || e->resource_heap != bound_heap ||
          e->sampler_heap != list->sampler_descriptor_heap ||
          memcmp(e->samplers, sampler_table,
                 compare_samplers * sizeof(*sampler_table)) != 0 ||
          memcmp(e->sources, sources, compare_slots * sizeof(*sources)) != 0 ||
          memcmp(e->texture_shapes, texture_shapes, compare_slots) != 0)
         continue;

      unsigned changed = 0;
      bool plane_only = true;
      for (unsigned i = 0; i < compare_slots && plane_only; i++) {
         if (memcmp(&e->table[i], &raw_table[i], sizeof(raw_table[i])) == 0)
            continue;
         changed++;
         plane_only = d12_descriptor_is_plane_only(&e->table[i]) &&
                      d12_descriptor_is_plane_only(&raw_table[i]) &&
                      !sources[i];
      }
      if (!plane_only)
         continue;
      if (changed &&
          !d12_list_reroot_bindings(list, e, raw_table, sources, bound_heap))
         return false;
      e->generation = generation;
      e->pso = list->pso;
      d12_binding_upload_cache_outputs(e, out_root, out_textures, out_pbes,
                                       out_heap, out_heap_textures,
                                       out_heap_pbes, out_heap_samplers,
                                       out_tables, out_table_count);
      return true;
   }

   /* An ExecuteIndirect root descriptor is patched by the GPU after command
    * recording.  Its binding snapshot must never enter the immutable upload
    * cache: a later draw with the same CPU-visible null root state would then
    * alias storage which the earlier command mutates at execution time. */
   if (force_unique) {
      for (unsigned w = 0; w < D12_BINDING_CACHE_WAYS; w++)
         ways[w].valid = false;
   } else {
      for (unsigned w = 0; w < D12_BINDING_CACHE_WAYS && !cache; w++) {
         if (!ways[w].valid)
            cache = &ways[w];
      }
      if (!cache) {
         uint8_t *next = &list->binding_cache_next[compute ? 1 : 0];
         cache = &ways[*next];
         *next = (*next + 1) % D12_BINDING_CACHE_WAYS;
      }
   }

   /* Slots past the bound root signature's binding map are unreachable by
    * its shaders: every per-slot pass below stops at root_count. */
   const unsigned root_count = rs ? MAX2(rs->map.resource_count, 1) : 1;
   if (root_count > D12_MAX_RESOURCE_BINDINGS) {
      list->state = D12_LIST_ERROR;
      return false;
   }

   struct d12_descriptor effective_table[D12_MAX_RESOURCE_BINDINGS];
   memcpy(effective_table, raw_table, root_count * sizeof(*raw_table));
   for (unsigned i = 0; i < root_count; i++)
      d12_apply_texture_shape(&effective_table[i], texture_shapes[i]);
   const struct d12_descriptor *table = effective_table;

   unsigned last_texture = root_count;
   for (unsigned i = 0; i < root_count; i++) {
      if ((table[i].kind == D12_DESC_SRV || table[i].kind == D12_DESC_UAV) &&
          table[i].resource &&
          (table[i].resource->desc.Dimension ==
              D3D12_RESOURCE_DIMENSION_TEXTURE1D ||
           table[i].resource->desc.Dimension ==
              D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
           table[i].resource->desc.Dimension ==
              D3D12_RESOURCE_DIMENSION_TEXTURE3D ||
           (table[i].resource->desc.Dimension ==
               D3D12_RESOURCE_DIMENSION_BUFFER &&
            table[i].format != DXGI_FORMAT_UNKNOWN)))
         last_texture = MAX2(last_texture, i + 1);
   }

   /* A finite root-signature range is a validation/aliasing declaration,
    * not a hardware bounds check. SM 5.1 and DXIL may dynamically index
    * beyond NumDescriptors into the rest of the bound shader-visible heap.
    * Dense root slots cannot represent that: two independently based tables
    * would overlap, which is exactly what the array-reinterpretation tests
    * exercise. A dynamically indexed texture handle therefore addresses the
    * heap's own native texture/PBE mirrors, which descriptor writes keep
    * coherent, through the heap index of the range's first descriptor,
    * published in the range's first resource slot. Nothing is copied. */
   struct d12_dynamic_texture_range {
      uint32_t root_range;
      uint32_t base_slot;
      uint32_t first;
   } dynamic_ranges[D12_MAX_RESOURCE_BINDINGS];
   unsigned dynamic_range_count = 0;

   if (rs && list->pso && bound_heap) {
      uintptr_t heap_start = (uintptr_t)bound_heap->slots;
      size_t heap_bytes =
         (size_t)bound_heap->desc.NumDescriptors * sizeof(bound_heap->slots[0]);

      for (unsigned e = 0; e < rs->map.count; e++) {
         const unsigned klass = rs->map.entries[e].klass;
         const unsigned root_range = rs->map.entries[e].root_range;
         if ((klass != D12_RES_SRV && klass != D12_RES_UAV) ||
             rs->map.entries[e].unbounded)
            continue;

         unsigned base_entry = e;
         for (unsigned j = 0; j < rs->map.count; j++) {
            if (rs->map.entries[j].root_range == root_range &&
                rs->map.entries[j].reg < rs->map.entries[base_entry].reg)
               base_entry = j;
         }

         unsigned base_slot = rs->map.entries[base_entry].slot;
         unsigned shape = list->pso->texture_shapes[base_slot];
         if (shape == D12_TEXTURE_SHAPE_NONE)
            continue;

         bool seen = false;
         for (unsigned j = 0; j < dynamic_range_count; j++)
            seen |= dynamic_ranges[j].root_range == root_range;
         if (seen)
            continue;

         const struct d12_descriptor *source = sources[base_slot];
         uintptr_t source_ptr = (uintptr_t)source;
         if (!source || source_ptr < heap_start ||
             source_ptr - heap_start >= heap_bytes ||
             (source_ptr - heap_start) % sizeof(*source)) {
            mesa_loge("dynamic texture range %u has no valid bound-heap base",
                      root_range);
            list->state = D12_LIST_ERROR;
            return false;
         }
         if (!bound_heap->texture_bo || !bound_heap->pbe_bo) {
            mesa_loge("dynamic texture range %u is bound to a heap without "
                      "native mirrors", root_range);
            list->state = D12_LIST_ERROR;
            return false;
         }

         size_t first =
            (source_ptr - heap_start) / sizeof(bound_heap->slots[0]);
         if (dynamic_range_count >= ARRAY_SIZE(dynamic_ranges)) {
            list->state = D12_LIST_ERROR;
            return false;
         }

         dynamic_ranges[dynamic_range_count++] =
            (struct d12_dynamic_texture_range){
               .root_range = root_range,
               .base_slot = base_slot,
               .first = first,
            };
      }
   }

   const size_t native_desc_count = last_texture;

   *out_heap = 0;
   *out_heap_textures = 0;
   *out_heap_pbes = 0;
   *out_heap_samplers = 0;

   /* Four arrays back to back: resource base addresses, UAV counters,
    * BUFINFO, then sparse residency map pointers. A small header
    * below points to each plane so shaders never need a global-load immediate
    * beyond the AGX encoding.
    */
   /* An UNBOUND slot is address 0, and a shader that reads one dereferences
    * it -- which the GPU reports as
    *
    *     address: 0x0  reason: Unmapped  read: true
    *
    * and which then times out the queue. Because the compositor shares this
    * GPU, that stalls the desktop too: 75 such timeouts in one suite run on
    * 2026-08-05 froze the GUI and cost a physical reboot.
    *
    * Substituting zeroed memory is not merely crash avoidance, it is the
    * specified behaviour: D3D12 defines reads through a null descriptor as
    * returning zero, and writes as discarded. Faulting was always the wrong
    * answer.
    */
   uint64_t null_buf = 0, null_counters = 0;
   uint64_t addrs[D12_ROOT_TABLE_ENTRIES];
   for (unsigned i = 0; i < root_count; i++) {
      if (!d12_root_plane_slot(list, &table[i], sources[i], i, &null_buf,
                               &null_counters, addrs))
         return false;
   }

   for (unsigned r = 0; r < dynamic_range_count; r++) {
      addrs[D12_ROOT_RESOURCE_BASE + dynamic_ranges[r].base_slot] =
         dynamic_ranges[r].first;
   }

   /* The shader ABI names the three planes through pointers in the root
    * headers; they do not need the old fixed 256-entry stride.  Upload only
    * the dense slots declared by this root signature.  Stress workloads can
    * record tens of thousands of root-descriptor changes in one list, and a
    * fixed 6 KiB upload per dispatch exhausts an otherwise-small 32 MiB
    * command arena. */
   uint64_t compact[D12_ROOT_TABLE_ENTRIES];
   memcpy(compact, addrs + D12_ROOT_RESOURCE_BASE,
          root_count * sizeof(uint64_t));
   memcpy(compact + root_count, addrs + D12_ROOT_COUNTER_BASE,
          root_count * sizeof(uint64_t));
   memcpy(compact + 2 * root_count, addrs + D12_ROOT_BUFINFO_BASE,
          root_count * sizeof(uint64_t));
   memcpy(compact + 3 * root_count, addrs + D12_ROOT_RESIDENCY_BASE,
          root_count * sizeof(uint64_t));

   uint64_t tables_gpu = d12_list_upload(
      list, compact, 4 * root_count * sizeof(uint64_t), 8);
   if (!tables_gpu)
      return false;
   if (out_tables)
      *out_tables = tables_gpu;
   if (out_table_count)
      *out_table_count = root_count;

   if (!null_buf && !(null_buf = d12_list_null_buffer(list)))
      return false;
   uint64_t discard_buf = d12_list_discard_buffer(list);
   if (!discard_buf)
      return false;

   uint64_t metadata_header[] = {
      tables_gpu + root_count * sizeof(uint64_t),
      tables_gpu + 2 * root_count * sizeof(uint64_t),
      null_buf,
      discard_buf,
      tables_gpu + 3 * root_count * sizeof(uint64_t),
      bound_heap && bound_heap->residency_bo
         ? bound_heap->residency_bo->va->addr : 0,
      list->device->dev.sparse_ro_offset,
   };
   static_assert(sizeof(metadata_header) == D12_ROOT_METADATA_HEADER_SIZE,
                 "root metadata header ABI mismatch");

   uint64_t metadata_gpu =
      d12_list_upload(list, metadata_header, sizeof(metadata_header), 8);
   if (!metadata_gpu)
      return false;

   /* Resource and sampler slots are independent namespaces. Oversized
    * sampler ranges store their first global heap index in this fourth
    * plane; finite/static samplers continue through the compact USC file. */
   bool rooted_samplers = false;
   if (rs) {
      for (unsigned i = 0; i < rs->map.count; i++) {
         rooted_samplers |= rs->map.entries[i].klass == D12_RES_SAMPLER &&
                            rs->map.entries[i].unbounded;
      }
   }

   uint64_t sampler_tables_gpu = 0;
   if (rooted_samplers) {
      uint64_t sampler_bases[D12_MAX_RESOURCE_BINDINGS];
      for (unsigned i = 0; i < D12_MAX_RESOURCE_BINDINGS; i++)
         sampler_bases[i] = sampler_table[i].address;

      sampler_tables_gpu = d12_list_upload(
         list, sampler_bases, sizeof(sampler_bases), 8);
      if (!sampler_tables_gpu)
         return false;
   }

   uint64_t heap_pbe_null_offset =
      bound_heap && bound_heap->pbe_bo
         ? (uint64_t)bound_heap->desc.NumDescriptors *
              bound_heap->pbe_stride_B
         : 0;
   uint64_t header[] = {
      tables_gpu,
      metadata_gpu,
      sampler_tables_gpu,
      heap_pbe_null_offset,
   };
   static_assert(sizeof(header) == D12_ROOT_HEADER_SIZE,
                 "root header ABI mismatch");

   uint64_t header_gpu = d12_list_upload(list, header, sizeof(header), 8);
   if (!header_gpu)
      return false;

   *out_root = d12_list_upload(list, &header_gpu, sizeof(header_gpu), 8);
   if (!*out_root)
      return false;

   /* A bindless texture handle is a byte offset from this heap. Allocate a
    * null descriptor for every slot declared by the active root signature,
    * even when no texture is currently bound. This keeps stale/uninitialised
    * root parameters undefined-but-safe instead of making an omitted uniform
    * read as VA 0 and wedging the GPU.
    */
   size_t heap_size;
   if (!d12_native_descriptor_span(native_layout.texture_stride_B,
                                   native_desc_count, 0, &heap_size)) {
      list->state = D12_LIST_ERROR;
      return false;
   }
   uint64_t heap_gpu = d12_list_upload_zero(list, heap_size, 64);
   if (!heap_gpu)
      return false;

   if (validate && (!d12_list_stream_contains(list, heap_gpu, heap_size) ||
                    (heap_gpu & 63))) {
      mesa_loge("binding validation: texture heap 0x%llx+%zu is outside "
                "the 64-byte-aligned command-list stream BO",
                (unsigned long long)heap_gpu, heap_size);
      list->state = D12_LIST_ERROR;
      return false;
   }

   uint8_t *descs = d12_list_data_cpu(list, heap_gpu);

   for (unsigned i = 0; i < native_desc_count; i++) {
      if (!d12_set_null_texture_descriptor(
             list->device, descs + (size_t)i * native_layout.texture_stride_B)) {
         list->state = D12_LIST_ERROR;
         return false;
      }
   }

   for (unsigned i = 0; i < last_texture; i++) {
      if (table[i].kind != D12_DESC_SRV && table[i].kind != D12_DESC_UAV)
         continue;

      struct d12_resource *res = table[i].resource;
      if (!res)
         continue;

      if (res->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
         if (!d12_refresh_texture_descriptor(
                list->device,
                descs + (size_t)i * native_layout.texture_stride_B,
                &table[i])) {
            list->state = D12_LIST_ERROR;
            return false;
         }
         continue;
      }

      if (res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D &&
          res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
          res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D)
         continue;

      const bool stencil = table[i].texture_plane == 1;
      const struct ail_layout *layout =
         stencil ? &res->stencil_layout : &res->layout;
      struct agx_bo *texture_bo = stencil ? res->stencil_bo : res->bo;
      if ((!texture_bo && !res->reserved) || !layout->size_B ||
          (stencil && res->reserved))
         continue;

      if (validate) {
         uint64_t bo_base, bo_size;
         if (!d12_descriptor_resource_span(&table[i], &bo_base, &bo_size)) {
            mesa_loge("binding validation: texture slot %u has no storage",
                      i);
            list->state = D12_LIST_ERROR;
            return false;
         }
         if (table[i].address < bo_base ||
             table[i].address - bo_base >= bo_size) {
            mesa_loge("binding validation: texture slot %u address 0x%llx "
                      "is outside resource BO 0x%llx..0x%llx", i,
                      (unsigned long long)table[i].address,
                      (unsigned long long)bo_base,
                      (unsigned long long)(bo_base + bo_size));
            list->state = D12_LIST_ERROR;
            return false;
         }
      }

      /* S8 is a storage/aspect format, not the colour interpretation used by
       * a Texture2D SRV. D3D12 exposes the stencil plane as an R8_UINT value;
       * keep the S8 layout/address but pack the descriptor's channels and
       * type from that view interpretation.
       */
      enum pipe_format pf = d12_native_texture_format(&table[i]);
      if (pf == PIPE_FORMAT_NONE || !ail_is_valid_pixel_format(pf) ||
          !ail_pixel_format[pf].texturable ||
          util_format_is_depth_or_stencil(pf))
         continue;

      const struct util_format_description *fdesc =
         util_format_description(pf);
      uint8_t format_swizzle[4] = {
         fdesc->swizzle[0], fdesc->swizzle[1],
         fdesc->swizzle[2], fdesc->swizzle[3],
      };

      /* The separately allocated stencil byte is physically R8 on AGX, but
       * D3D12's X24_TYPELESS_G8_UINT and X32_TYPELESS_G8X24_UINT views expose
       * it as memory component 1.  Keep that logical component placement
       * before composing Shader4ComponentMapping; treating the plane as a
       * plain R8 view makes FROM_MEMORY_COMPONENT_1 sample constant zero. */
      if (stencil &&
          (table[i].format == DXGI_FORMAT_X24_TYPELESS_G8_UINT ||
           table[i].format == DXGI_FORMAT_X32_TYPELESS_G8X24_UINT)) {
         format_swizzle[0] = PIPE_SWIZZLE_0;
         format_swizzle[1] = PIPE_SWIZZLE_X;
         format_swizzle[2] = PIPE_SWIZZLE_0;
         format_swizzle[3] = PIPE_SWIZZLE_1;
      }

      uint8_t swizzle[4];
      if (table[i].kind == D12_DESC_SRV) {
         uint8_t view_swizzle[4];
         for (unsigned c = 0; c < 4; c++) {
            view_swizzle[c] = D3D12_DECODE_SHADER_4_COMPONENT_MAPPING(
               c, table[i].texture_component_mapping);
         }
         util_format_compose_swizzles(format_swizzle, view_swizzle, swizzle);
      } else {
         memcpy(swizzle, format_swizzle, sizeof(swizzle));
      }
      const unsigned samples = MAX2(layout->sample_count_sa, 1);
      const enum agx_texture_dimension dimension =
         d12_native_texture_dimension(&table[i], samples);
      const unsigned level = table[i].kind == D12_DESC_UAV
                                ? table[i].texture_level
                                : table[i].texture_first_level;
      const unsigned last_level = table[i].kind == D12_DESC_UAV
                                     ? level
                                     : level + MAX2(
                                          table[i].texture_level_count, 1) - 1;
      const unsigned layers = dimension == AGX_TEXTURE_DIMENSION_3D
                                 ? layout->depth_px
                                 : d12_native_texture_layers(&table[i]);
      const bool layered =
         dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY ||
         dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED ||
         dimension == AGX_TEXTURE_DIMENSION_CUBE_ARRAY;
      uint64_t sparse_address = 0;
      const bool sparse =
         d12_sparse_texture_map_address(&table[i], &sparse_address);

      if (descriptor_generation == AGX_DESCRIPTOR_GENERATION_9 ||
          descriptor_generation == AGX_DESCRIPTOR_GENERATION_10) {
         if (!d12_refresh_texture_descriptor(
                list->device,
                descs + (size_t)i * native_layout.texture_stride_B,
                &table[i])) {
            mesa_loge("binding upload: texture slot %u uses fields not "
                      "proven for descriptor generation %u", i,
                      (unsigned)descriptor_generation);
            list->state = D12_LIST_ERROR;
            return false;
         }
         continue;
      }

      struct agx_texture_packed *native =
         (struct agx_texture_packed *)(descs +
            (size_t)i * native_layout.texture_stride_B);

      agx_pack(native, TEXTURE, cfg) {
         cfg.dimension = dimension;
         cfg.layout = agx_translate_layout(layout->tiling);
         cfg.channels = ail_pixel_format[pf].channels;
         cfg.type = ail_pixel_format[pf].type;
         cfg.srgb = util_format_is_srgb(pf);

         cfg.swizzle_r = agx_channel_from_pipe(swizzle[0]);
         cfg.swizzle_g = agx_channel_from_pipe(swizzle[1]);
         cfg.swizzle_b = agx_channel_from_pipe(swizzle[2]);
         cfg.swizzle_a = agx_channel_from_pipe(swizzle[3]);

         if (sparse) {
            cfg.mode = AGX_IMAGE_MODE_SPARSE;
            cfg.address = sparse_address;
         } else {
            cfg.address = table[i].address;
         }
         cfg.width = layout->width_px;
         cfg.height = layout->height_px;
         cfg.first_level = level;
         cfg.last_level = last_level;
         cfg.unk_mipmapped = layout->levels > 1;
         if (samples > 1)
            cfg.samples = agx_translate_sample_count(samples);

         /* Depth and Stride share bit 110; the hardware reads one or the
          * other by layout. Depth carries minus(1), so a single-layer
          * texture packs zero either way -- but write only the one that
          * applies, because an array does not.
          */
         if (layout->tiling == AIL_TILING_LINEAR) {
            /* TEXTURE.Stride is (bytes - 16) >> 4. Note PBE.Stride is a
             * different convention: bytes - 4, unshifted.
             */
            cfg.stride = ail_get_linear_stride_B(layout, 0) - 16;
            if (layered) {
               cfg.depth_linear = layers;
               cfg.layer_stride_linear = layout->layer_stride_B - 0x80;
               cfg.extended = true;
            }
         } else {
            cfg.depth = layers;
            cfg.page_aligned_layers = layout->page_aligned_layers;
         }
      }
   }

   if (validate || d12_debug()->dump_tex) {
      for (unsigned r = 0; r < dynamic_range_count; r++) {
         mesa_logi("dynamic texture range=%u root-slot=%u heap-first=%u "
                   "valid=1", dynamic_ranges[r].root_range,
                   dynamic_ranges[r].base_slot, dynamic_ranges[r].first);
      }
   }

   size_t pbe_heap_size;
   if (!d12_native_descriptor_span(native_layout.pbe_stride_B,
                                   native_desc_count, 0, &pbe_heap_size)) {
      list->state = D12_LIST_ERROR;
      return false;
   }
   uint64_t pbe_heap_gpu =
      d12_list_upload_zero(list, pbe_heap_size, 64);
   if (!pbe_heap_gpu)
      return false;

   if (validate &&
       (!d12_list_stream_contains(list, pbe_heap_gpu, pbe_heap_size) ||
        (pbe_heap_gpu & 63))) {
      mesa_loge("binding validation: PBE heap 0x%llx+%zu is outside "
                "the 64-byte-aligned command-list stream BO",
                (unsigned long long)pbe_heap_gpu, pbe_heap_size);
      list->state = D12_LIST_ERROR;
      return false;
   }

   uint8_t *pbes = d12_list_data_cpu(list, pbe_heap_gpu);
   for (unsigned i = 0; i < native_desc_count; i++) {
      if (!d12_set_null_pbe_descriptor(
             list->device, pbes + (size_t)i * native_layout.pbe_stride_B)) {
         list->state = D12_LIST_ERROR;
         return false;
      }
   }
   for (unsigned i = 0; i < last_texture; i++) {
      if (validate && table[i].kind == D12_DESC_UAV && table[i].resource &&
          (table[i].resource->desc.Dimension ==
              D3D12_RESOURCE_DIMENSION_TEXTURE1D ||
           table[i].resource->desc.Dimension ==
              D3D12_RESOURCE_DIMENSION_TEXTURE2D) &&
          (!table[i].resource->layout.writeable_image ||
           table[i].resource->layout.tiling == AIL_TILING_LINEAR)) {
         mesa_loge("binding validation: writable texture slot %u has no "
                   "atomic-capable tiled layout", i);
         list->state = D12_LIST_ERROR;
         return false;
      }
      if (!d12_refresh_pbe_descriptor(
             list->device, pbes + (size_t)i * native_layout.pbe_stride_B,
             &table[i])) {
         list->state = D12_LIST_ERROR;
         return false;
      }

      if (d12_debug()->dump_tex) {
         const uint8_t *raw =
            pbes + (size_t)i * native_layout.pbe_stride_B;
         fprintf(stderr, "pbe[%u]:", i);
         for (unsigned b = 0; b < native_layout.pbe_stride_B; b++)
            fprintf(stderr, " %02x", raw[b]);
         fprintf(stderr, "\n");
      }
   }

   if (d12_debug()->dump_tex) {
      for (unsigned i = 0; i < last_texture; i++) {
         struct d12_resource *res =
            (table[i].kind == D12_DESC_SRV ||
             table[i].kind == D12_DESC_UAV) ? table[i].resource : NULL;
         fprintf(stderr,
                 "tex[%u] kind=%u dimension=%u va=0x%llx size=%u "
                 "stride=%u format=%u\n", i, table[i].kind,
                 res ? (unsigned)res->desc.Dimension : 0,
                 (unsigned long long)table[i].address, table[i].size_B,
                 table[i].stride_B, (unsigned)table[i].format);
         const uint8_t *d =
            descs + (size_t)i * native_layout.texture_stride_B;
         fprintf(stderr, "  desc:");
         for (unsigned b = 0; b < native_layout.texture_stride_B; b++)
            fprintf(stderr, " %02x", d[b]);
         fprintf(stderr, "\n");
      }
   }

   /* The heap needs a home for the same reason the root table does. */
   *out_textures = d12_list_upload(list, &heap_gpu, sizeof(heap_gpu), 8);
   *out_pbes =
      d12_list_upload(list, &pbe_heap_gpu, sizeof(pbe_heap_gpu), 8);

   /* Heap-rooted and SM 6.6 handles address the application's coherent
    * 40-byte descriptor array, while AGX texture/image instructions require
    * parallel generation-sized native arrays. Publish one home cell for each
    * base; shader lowering derives the same global heap index for all three.
    */
   if (bound_heap && bound_heap->bo && bound_heap->texture_bo &&
       bound_heap->pbe_bo) {
      uint64_t base = bound_heap->bo->va->addr;
      uint64_t textures = bound_heap->texture_bo->va->addr;
      uint64_t pbe_heap_base = bound_heap->pbe_bo->va->addr;
      *out_heap = d12_list_upload(list, &base, sizeof(base), 8);
      *out_heap_textures =
         d12_list_upload(list, &textures, sizeof(textures), 8);
      *out_heap_pbes =
         d12_list_upload(list, &pbe_heap_base, sizeof(pbe_heap_base), 8);
      if (!*out_heap || !*out_heap_textures || !*out_heap_pbes)
         return false;

      if (validate) {
         uint8_t expected_null[32];
         const bool expected_valid = d12_set_null_pbe_descriptor(
            list->device, expected_null);
         const uint64_t null_offset =
            (uint64_t)bound_heap->desc.NumDescriptors *
            bound_heap->pbe_stride_B;
         const uint8_t *pbe_map = agx_bo_map(bound_heap->pbe_bo);
         const bool null_valid =
            expected_valid && pbe_map != NULL &&
            null_offset <= bound_heap->pbe_bo->size &&
            bound_heap->pbe_stride_B <=
               bound_heap->pbe_bo->size - null_offset &&
            memcmp(pbe_map + null_offset, &expected_null,
                   bound_heap->pbe_stride_B) == 0;
         mesa_logi("binding validation: bindless heap slots=0x%llx "
                   "textures=0x%llx pbes=0x%llx count=%u "
                   "null-pbe-offset=0x%llx valid=%u",
                   (unsigned long long)base,
                   (unsigned long long)textures,
                   (unsigned long long)pbes,
                   bound_heap->desc.NumDescriptors,
                   (unsigned long long)null_offset, null_valid);
         if (!null_valid) {
            list->state = D12_LIST_ERROR;
            return false;
         }
      }
   }

   struct d12_descriptor_heap *bound_samplers =
      list->sampler_descriptor_heap;
   if (bound_samplers && bound_samplers->sampler_bo) {
      uint64_t samplers = bound_samplers->sampler_bo->va->addr;
      *out_heap_samplers =
         d12_list_upload(list, &samplers, sizeof(samplers), 8);
      if (!*out_heap_samplers)
         return false;

      if (validate) {
         mesa_logi("binding validation: bindless sampler heap=0x%llx "
                   "count=%u home=0x%llx valid=1",
                   (unsigned long long)samplers,
                   bound_samplers->desc.NumDescriptors,
                   (unsigned long long)*out_heap_samplers);
      }
   }

   if (validate) {
      if (!*out_textures || !*out_pbes ||
          !d12_list_stream_contains(list, *out_textures, sizeof(heap_gpu))) {
         mesa_loge("binding validation: texture-home cell 0x%llx is outside "
                   "the command-list stream BO",
                   (unsigned long long)*out_textures);
         list->state = D12_LIST_ERROR;
         return false;
      }

      if (!d12_list_stream_contains(list, *out_pbes,
                                    sizeof(pbe_heap_gpu)) ||
          *(uint64_t *)d12_list_data_cpu(list, *out_pbes) != pbe_heap_gpu) {
         mesa_loge("binding validation: PBE-home cell 0x%llx is invalid",
                   (unsigned long long)*out_pbes);
         list->state = D12_LIST_ERROR;
         return false;
      }

      uint64_t home = *(uint64_t *)d12_list_data_cpu(list, *out_textures);
      if (home != heap_gpu) {
         mesa_loge("binding validation: texture-home cell holds 0x%llx, "
                   "expected heap 0x%llx", (unsigned long long)home,
                   (unsigned long long)heap_gpu);
         list->state = D12_LIST_ERROR;
         return false;
      }

      mesa_logi("binding validation: texture heap=0x%llx+%zu home=0x%llx "
                "PBE=0x%llx+%zu home=0x%llx resource-slots=%u valid=1",
                (unsigned long long)heap_gpu, heap_size,
                (unsigned long long)*out_textures,
                (unsigned long long)pbe_heap_gpu, pbe_heap_size,
                (unsigned long long)*out_pbes, last_texture);
   }

   if (d12_debug()->dump_tex && *out_textures) {
      uint64_t home = *(uint64_t *)d12_list_data_cpu(list, *out_textures);
      fprintf(stderr,
              "chain: textures_gpu=0x%llx -> home holds 0x%llx "
              "(heap_gpu=0x%llx) %s\n",
              (unsigned long long)*out_textures, (unsigned long long)home,
              (unsigned long long)heap_gpu,
              home == heap_gpu ? "MATCH" : "MISMATCH");
   }

   /* A shader-visible finite table remains mutable until execution.  Record
    * where this draw/dispatch flattened each source slot so the queue can
    * refresh it after Close and immediately before submission.
    */
   for (unsigned i = 0; i < root_count; i++) {
      if (!sources[i])
         continue;

      bool dynamic_base = false;
      for (unsigned r = 0; r < dynamic_range_count; r++)
         dynamic_base |= dynamic_ranges[r].base_slot == i;
      if (dynamic_base)
         continue;

      struct d12_descriptor_fixup fixup = {
         .source = sources[i],
         .tables_gpu = tables_gpu,
         .textures_gpu = i < last_texture ? heap_gpu : 0,
         .pbes_gpu = i < last_texture ? pbe_heap_gpu : 0,
         .slot = i,
         .table_count = root_count,
         .texture_shape =
            list->pso ? list->pso->texture_shapes[i] : D12_TEXTURE_SHAPE_NONE,
      };
      util_dynarray_append_typed(&list->descriptor_fixups,
                                 struct d12_descriptor_fixup, fixup);
   }

   if (!*out_textures || !*out_pbes)
      return false;

   if (force_unique)
      return true;

   cache->valid = true;
   cache->generation = generation;
   cache->rs = rs;
   cache->pso = list->pso;
   cache->resource_heap = bound_heap;
   cache->sampler_heap = list->sampler_descriptor_heap;
   memcpy(cache->compact, compact, 4 * root_count * sizeof(uint64_t));
   cache->pbe_table = pbe_heap_gpu;
   /* Hits compare compare_slots / compare_samplers entries, the same
    * bounds, so only the reachable prefix of each snapshot is kept. */
   memcpy(cache->table, raw_table, compare_slots * sizeof(*raw_table));
   memcpy(cache->samplers, sampler_table,
          compare_samplers * sizeof(*sampler_table));
   memcpy(cache->sources, sources, compare_slots * sizeof(*sources));
   memcpy(cache->texture_shapes, texture_shapes, compare_slots);
   cache->root = *out_root;
   cache->textures = *out_textures;
   cache->texture_table = heap_gpu;
   cache->texture_table_size_B = heap_size;
   cache->pbes = *out_pbes;
   cache->heap = *out_heap;
   cache->heap_textures = *out_heap_textures;
   cache->heap_pbes = *out_heap_pbes;
   cache->heap_samplers = *out_heap_samplers;
   cache->tables = tables_gpu;
   cache->table_count = root_count;
   return true;
}

enum d12_write_visibility_group {
   D12_WRITE_VIS_COMPUTE,
   D12_WRITE_VIS_GRAPHICS_VERTEX,
   D12_WRITE_VIS_GRAPHICS_FRAGMENT,
};

static bool
d12_write_visibility_matches(uint32_t visibility,
                             enum d12_write_visibility_group group)
{
   if (group == D12_WRITE_VIS_COMPUTE ||
       visibility == D3D12_SHADER_VISIBILITY_ALL)
      return true;

   if (group == D12_WRITE_VIS_GRAPHICS_FRAGMENT)
      return visibility == D3D12_SHADER_VISIBILITY_PIXEL;

   return visibility != D3D12_SHADER_VISIBILITY_PIXEL;
}

/* Record every UAV allocation a shader in this hardware pipe may write.
 * Root-signature entries are deliberately conservative: the compiler may
 * prove an entry dead, but publishing one extra owning allocation is valid
 * while omitting one used by dynamic flow violates Aurora's v3 writeback
 * contract. */
static void
d12_segment_track_bound_uavs(struct d12_cmd_list *list,
                             struct d12_segment *seg,
                             bool compute_bindings,
                             enum d12_write_visibility_group visibility,
                             enum d12_write_pipe pipe)
{
   const struct d12_root_signature *rs =
      compute_bindings ? list->compute_rs : list->graphics_rs;
   const struct d12_descriptor *table =
      compute_bindings ? list->compute_root_table : list->graphics_root_table;
   const struct d12_descriptor *const *sources =
      compute_bindings ? list->compute_root_source
                       : list->graphics_root_source;
   const struct d12_descriptor_heap *heap =
      list->resource_descriptor_heap;

   if (!rs)
      return;

   /* Shader Model 6.6 ResourceDescriptorHeap indexing does not name a root
    * range.  The current coherent heap contents are therefore the smallest
    * host-derivable conservative set. */
   if ((rs->layout.flags &
        D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED) &&
       heap && heap->desc.NumDescriptors) {
      d12_segment_track_write_heap(seg, pipe, heap, 0, 0);
   }

   for (unsigned i = 0; i < rs->map.count; i++) {
      if (rs->map.entries[i].klass != D12_RES_UAV ||
          !d12_write_visibility_matches(rs->map.entries[i].visibility,
                                        visibility))
         continue;

      const unsigned slot = rs->map.entries[i].slot;
      if (slot >= D12_MAX_RESOURCE_BINDINGS) {
         d12_segment_write_tracker(seg, pipe)->incomplete = true;
         continue;
      }

      if (rs->map.entries[i].unbounded) {
         if (!heap || !heap->bo || !table[slot].address ||
             table[slot].address < heap->bo->va->addr) {
            d12_segment_write_tracker(seg, pipe)->incomplete = true;
            continue;
         }

         uint64_t byte_offset =
            table[slot].address - heap->bo->va->addr;
         if (byte_offset % sizeof(struct d12_descriptor) ||
             byte_offset / sizeof(struct d12_descriptor) >=
                heap->desc.NumDescriptors) {
            d12_segment_write_tracker(seg, pipe)->incomplete = true;
            continue;
         }

         d12_segment_track_write_heap(
            seg, pipe, heap,
            (uint32_t)(byte_offset / sizeof(struct d12_descriptor)),
            rs->map.entries[i].range_count);
         continue;
      }

      if (sources[slot]) {
         d12_segment_track_write_descriptor(seg, pipe,
                                            sources[slot]);
      } else if (table[slot].kind == D12_DESC_UAV &&
                 table[slot].address) {
         if (table[slot].resource) {
            d12_segment_track_write_resource(seg, pipe,
                                             table[slot].resource);
         } else {
            /* A dense root UAV has no declared length. Queue packing resolves
             * the address to its complete owning allocation. */
            d12_segment_track_write_range(seg, pipe,
                                          table[slot].address, 0);
         }
      }
   }
}

/* Record every resource a shader in one visibility group may reference.
 * Mutable descriptor slots are intentionally retained by address and read at
 * ExecuteCommandLists, after the command-list descriptor refresh. This is the
 * point at which D3D12 freezes descriptor contents for the submission. */
static void
d12_list_track_bound_reads(struct d12_cmd_list *list, bool compute_bindings,
                           enum d12_write_visibility_group visibility)
{
   if (!list->device->submit_read_set)
      return;
   const struct d12_root_signature *rs =
      compute_bindings ? list->compute_rs : list->graphics_rs;
   const struct d12_descriptor *table =
      compute_bindings ? list->compute_root_table : list->graphics_root_table;
   const struct d12_descriptor *samplers =
      compute_bindings ? list->compute_sampler_table
                       : list->graphics_sampler_table;
   const struct d12_descriptor *const *sources =
      compute_bindings ? list->compute_root_source
                       : list->graphics_root_source;

   if (!rs)
      return;

   if (rs->layout.flags &
       D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED) {
      if (list->resource_descriptor_heap &&
          list->resource_descriptor_heap->desc.NumDescriptors)
         d12_list_track_read_heap(list, list->resource_descriptor_heap, 0,
                                  0, true);
      else
         d12_list_mark_read_set_incomplete(list);
   }

   if (rs->layout.flags &
       D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED) {
      if (list->sampler_descriptor_heap &&
          list->sampler_descriptor_heap->desc.NumDescriptors)
         d12_list_track_read_heap(list, list->sampler_descriptor_heap, 0,
                                  0, false);
      else
         d12_list_mark_read_set_incomplete(list);
   }

   for (unsigned i = 0; i < rs->map.count; ++i) {
      const unsigned klass = rs->map.entries[i].klass;
      if (!d12_write_visibility_matches(rs->map.entries[i].visibility,
                                        visibility))
         continue;

      const unsigned slot = rs->map.entries[i].slot;
      if (slot >= D12_MAX_RESOURCE_BINDINGS) {
         d12_list_mark_read_set_incomplete(list);
         continue;
      }

      if (klass == D12_RES_SAMPLER) {
         if (rs->map.entries[i].unbounded) {
            const struct d12_descriptor_heap *heap =
               list->sampler_descriptor_heap;
            const uint64_t first = samplers[slot].address;
            if (!heap || first >= heap->desc.NumDescriptors)
               d12_list_mark_read_set_incomplete(list);
            else
               d12_list_track_read_heap(list, heap, (uint32_t)first,
                                        rs->map.entries[i].range_count,
                                        false);
         }
         continue;
      }

      if (klass >= D12_RES_CLASS_COUNT) {
         d12_list_mark_read_set_incomplete(list);
         continue;
      }

      if (rs->map.entries[i].unbounded) {
         const struct d12_descriptor_heap *heap =
            list->resource_descriptor_heap;
         if (!heap || !heap->bo || !table[slot].address ||
             table[slot].address < heap->bo->va->addr) {
            d12_list_mark_read_set_incomplete(list);
            continue;
         }

         const uint64_t byte_offset =
            table[slot].address - heap->bo->va->addr;
         if (byte_offset % sizeof(struct d12_descriptor) ||
             byte_offset / sizeof(struct d12_descriptor) >=
                heap->desc.NumDescriptors) {
            d12_list_mark_read_set_incomplete(list);
            continue;
         }

         d12_list_track_read_heap(
            list, heap,
            (uint32_t)(byte_offset / sizeof(struct d12_descriptor)),
            rs->map.entries[i].range_count, true);
      } else if (sources[slot]) {
         d12_list_track_read_descriptor(list, sources[slot]);
      } else if (table[slot].resource) {
         d12_list_track_read_resource(list, table[slot].resource);
      } else if (table[slot].address) {
         /* Root CBV/SRV/UAV descriptors carry an address but no resource
          * object or trustworthy shader-access length. Resolve the owner at
          * Execute and publish its complete allocation. */
         d12_list_track_read_range(list, table[slot].address, 0);
      }
   }
}

static enum agx_texture_dimension
d12_native_texture_dimension(const struct d12_descriptor *d, unsigned samples)
{
   if (d->kind == D12_DESC_UAV) {
      if (d->texture_uav_dimension == D3D12_UAV_DIMENSION_TEXTURE3D)
         return AGX_TEXTURE_DIMENSION_3D;
      return (d->texture_uav_dimension == D3D12_UAV_DIMENSION_TEXTURE1DARRAY ||
              d->texture_uav_dimension == D3D12_UAV_DIMENSION_TEXTURE2DARRAY)
                ? AGX_TEXTURE_DIMENSION_2D_ARRAY
                : AGX_TEXTURE_DIMENSION_2D;
   }

   switch ((D3D12_SRV_DIMENSION)d->texture_srv_dimension) {
   case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
   case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
      return AGX_TEXTURE_DIMENSION_2D_ARRAY;
   case D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY:
      return AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
   case D3D12_SRV_DIMENSION_TEXTURECUBE:
      return AGX_TEXTURE_DIMENSION_CUBE;
   case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
      return AGX_TEXTURE_DIMENSION_CUBE_ARRAY;
   case D3D12_SRV_DIMENSION_TEXTURE3D:
      return AGX_TEXTURE_DIMENSION_3D;
   case D3D12_SRV_DIMENSION_TEXTURE2DMS:
      return AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED;
   default:
      return samples > 1 ? AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED
                         : AGX_TEXTURE_DIMENSION_2D;
   }
}

static unsigned
d12_native_texture_layers(const struct d12_descriptor *d)
{
   if (d->kind == D12_DESC_UAV)
      return MAX2(d->texture_layers, 1);

   unsigned layers = MAX2(d->texture_srv_layers, 1);
   if (d->texture_srv_dimension == D3D12_SRV_DIMENSION_TEXTURECUBE ||
       d->texture_srv_dimension == D3D12_SRV_DIMENSION_TEXTURECUBEARRAY)
      return MAX2(layers / 6, 1);

   return layers;
}

static uint64_t
d12_descriptor_bufinfo(const struct d12_descriptor *d)
{
   if (d->resource &&
       d->resource->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
       (d->kind == D12_DESC_SRV || d->kind == D12_DESC_UAV)) {
      uint64_t count = 0;
      switch (d->buffer_kind) {
      case D12_BUFFER_VIEW_RAW:
      case D12_BUFFER_VIEW_TYPED:
      case D12_BUFFER_VIEW_STRUCTURED:
         count = d->buffer_count;
         break;
      default:
         break;
      }

      if (d->buffer_kind != D12_BUFFER_VIEW_NONE)
         count |= (uint64_t)d->buffer_stride_B << 32;
      return count;
   }

   if (d->resource && d->kind == D12_DESC_SRV &&
       (d->resource->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D ||
        d->resource->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        d->resource->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)) {
      uint32_t info = d->texture_txf_min_level |
                      (d->texture_min_lod_oob ? 0x80000000u : 0);
      return fui(d->texture_min_lod) | (uint64_t)info << 32;
   }

   if (d->address && !d->resource &&
       (d->kind == D12_DESC_SRV || d->kind == D12_DESC_UAV))
      return UINT32_MAX;

   return 0;
}

static bool
d12_sparse_texture_map_address(const struct d12_descriptor *d,
                               uint64_t *address)
{
   struct d12_resource *res = d ? d->resource : NULL;
   if (!address || !res || !res->reserved || !res->sparse_map ||
       !res->sparse_map->va || d->texture_plane != 0 ||
       !res->layout.layer_stride_B)
      return false;

   const uint64_t alias_offset =
      d->kind == D12_DESC_SRV ? res->device->dev.sparse_ro_offset : 0;
   if (res->addr > UINT64_MAX - alias_offset)
      return false;
   const uint64_t base = res->addr + alias_offset;
   if (d->address < base)
      return false;

   const uint64_t layer_offset_B = d->address - base;
   if ((layer_offset_B % res->layout.layer_stride_B) != 0)
      return false;

   uint32_t index_el;
   if (!d12_sparse_residency_index(
          layer_offset_B, res->layout.layer_stride_B,
          res->layout.depth_px, res->layout.sparse_folios_per_layer,
          res->layout.sparse_table_size_B, &index_el))
      return false;

   const uint64_t byte_offset =
      (uint64_t)index_el * D12_AGX_SPARSE_ELEMENT_SIZE_B;
   if (res->sparse_map->va->addr > UINT64_MAX - byte_offset)
      return false;
   *address = res->sparse_map->va->addr + byte_offset;
   return true;
}

uint64_t
d12_descriptor_sparse_residency_address(const struct d12_descriptor *d)
{
   if (!d || !d->resource)
      return 0;

   if (d->resource->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
      return d12_sparse_buffer_residency_address(d->resource, d->address);

   uint64_t address = 0;
   return d12_sparse_texture_map_address(d, &address) ? address : 0;
}

/* Recreate one API SRV/UAV's AGX descriptor in already-allocated command-list
 * storage.  This is the same packing performed while recording; Execute uses
 * it only for finite shader-visible table slots whose CPU descriptor changed
 * after Close.
 */
static void
d12_refresh_texture_descriptor_agx2(struct agx_texture_packed *out,
                                    const struct d12_descriptor *d)
{
   agx_set_null_texture(out);

   if ((d->kind != D12_DESC_SRV && d->kind != D12_DESC_UAV) || !d->resource)
      return;

   struct d12_resource *res = d->resource;
   if (res->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
      if (d->format == DXGI_FORMAT_UNKNOWN)
         return;

      enum pipe_format pf = d12_format_to_pipe(d->format);
      if (pf == PIPE_FORMAT_NONE || !ail_is_valid_pixel_format(pf) ||
          !ail_pixel_format[pf].texturable ||
          util_format_is_depth_or_stencil(pf))
         return;

      const struct util_format_description *fdesc =
         util_format_description(pf);
      unsigned blocksize = util_format_get_blocksize(pf);
      unsigned elements = d->buffer_count;
      uint64_t read_address = d->address;
      if (res->reserved &&
          !agx_addr_is_ro(&res->device->dev, read_address))
         read_address += res->device->dev.sparse_ro_offset;
      uint64_t base = read_address & ~0xfull;
      unsigned tail_B = read_address & 0xf;
      if (!elements || tail_B % blocksize)
         return;

      agx_pack(out, TEXTURE, cfg) {
         cfg.dimension = AGX_TEXTURE_DIMENSION_2D;
         cfg.layout = AGX_LAYOUT_LINEAR;
         cfg.channels = ail_pixel_format[pf].channels;
         cfg.type = ail_pixel_format[pf].type;
         cfg.srgb = util_format_is_srgb(pf);
         cfg.srgb_2_channel =
            cfg.srgb && util_format_colormask(fdesc) == 0x3;
         cfg.swizzle_r = agx_channel_from_pipe(fdesc->swizzle[0]);
         cfg.swizzle_g = agx_channel_from_pipe(fdesc->swizzle[1]);
         cfg.swizzle_b = agx_channel_from_pipe(fdesc->swizzle[2]);
         cfg.swizzle_a = agx_channel_from_pipe(fdesc->swizzle[3]);
         /* AGX requires a 16-byte-aligned texture address. Preserve a
          * texel-aligned D3D buffer-view offset in the descriptor's software
          * element offset, as Honeykrisp does for VkBufferView. */
         cfg.address = base;
         cfg.width = AGX_TEXTURE_BUFFER_WIDTH;
         cfg.height = DIV_ROUND_UP(elements, cfg.width);
         cfg.first_level = 0;
         cfg.last_level = 0;
         cfg.buffer_size_sw = elements;
         cfg.buffer_offset_sw = tail_B / blocksize;
         cfg.depth = 1;
         cfg.stride = (cfg.width * blocksize) - 16;
      }
      return;
   }

   if (res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D &&
       res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
       res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D)
      return;

   const bool stencil = d->texture_plane == 1;
   const struct ail_layout *layout =
      stencil ? &res->stencil_layout : &res->layout;
   struct agx_bo *texture_bo = stencil ? res->stencil_bo : res->bo;
   if ((!texture_bo && !res->reserved) || !layout->size_B ||
       (stencil && res->reserved))
      return;

   enum pipe_format pf = d12_native_texture_format(d);
   if (pf == PIPE_FORMAT_NONE || !ail_is_valid_pixel_format(pf) ||
       !ail_pixel_format[pf].texturable ||
       util_format_is_depth_or_stencil(pf))
      return;

   const struct util_format_description *fdesc = util_format_description(pf);
   uint8_t format_swizzle[4] = {
      fdesc->swizzle[0], fdesc->swizzle[1],
      fdesc->swizzle[2], fdesc->swizzle[3],
   };

   /* A8 shares R8 storage on AGX, but its D3D view exposes that physical
    * byte as alpha and forces RGB to zero.  Preserve that logical format
    * swizzle before composing Shader4ComponentMapping. */
   if (d->format == DXGI_FORMAT_A8_UNORM) {
      format_swizzle[0] = PIPE_SWIZZLE_0;
      format_swizzle[1] = PIPE_SWIZZLE_0;
      format_swizzle[2] = PIPE_SWIZZLE_0;
      format_swizzle[3] = PIPE_SWIZZLE_X;
   } else if (stencil &&
              (d->format == DXGI_FORMAT_X24_TYPELESS_G8_UINT ||
               d->format == DXGI_FORMAT_X32_TYPELESS_G8X24_UINT)) {
      format_swizzle[0] = PIPE_SWIZZLE_0;
      format_swizzle[1] = PIPE_SWIZZLE_X;
      format_swizzle[2] = PIPE_SWIZZLE_0;
      format_swizzle[3] = PIPE_SWIZZLE_1;
   }

   uint8_t swizzle[4];
   if (d->kind == D12_DESC_SRV) {
      uint8_t view_swizzle[4];
      for (unsigned c = 0; c < 4; c++) {
         view_swizzle[c] = D3D12_DECODE_SHADER_4_COMPONENT_MAPPING(
            c, d->texture_component_mapping);
      }
      util_format_compose_swizzles(format_swizzle, view_swizzle, swizzle);
   } else {
      memcpy(swizzle, format_swizzle, sizeof(swizzle));
   }
   const unsigned samples = MAX2(layout->sample_count_sa, 1);
   const enum agx_texture_dimension dimension =
      d12_native_texture_dimension(d, samples);
   const bool rebased_3d_uav =
      d->kind == D12_DESC_UAV && dimension == AGX_TEXTURE_DIMENSION_3D;
   const unsigned source_level = d->kind == D12_DESC_UAV
                                    ? d->texture_level
                                    : d->texture_first_level;
   const unsigned level = rebased_3d_uav ? 0 : source_level;
   const unsigned last_level = d->kind == D12_DESC_UAV
                                  ? level
                                  : level + MAX2(d->texture_level_count, 1) - 1;
   const unsigned layers = d12_native_texture_layers(d);
   const bool layered =
      dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY ||
      dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED ||
      dimension == AGX_TEXTURE_DIMENSION_CUBE_ARRAY;
   uint64_t sparse_address = 0;
   const bool sparse = d12_sparse_texture_map_address(d, &sparse_address);

   agx_pack(out, TEXTURE, cfg) {
      cfg.dimension = dimension;
      cfg.layout = agx_translate_layout(layout->tiling);
      cfg.channels = ail_pixel_format[pf].channels;
      cfg.type = ail_pixel_format[pf].type;
      cfg.srgb = util_format_is_srgb(pf);
      cfg.swizzle_r = agx_channel_from_pipe(swizzle[0]);
      cfg.swizzle_g = agx_channel_from_pipe(swizzle[1]);
      cfg.swizzle_b = agx_channel_from_pipe(swizzle[2]);
      cfg.swizzle_a = agx_channel_from_pipe(swizzle[3]);
      if (sparse) {
         cfg.mode = AGX_IMAGE_MODE_SPARSE;
         cfg.address = sparse_address;
      } else {
         cfg.address = d->address;
      }
      cfg.width = rebased_3d_uav
                     ? u_minify(layout->width_px, source_level)
                     : layout->width_px;
      cfg.height = rebased_3d_uav
                      ? u_minify(layout->height_px, source_level)
                      : layout->height_px;
      cfg.first_level = level;
      cfg.last_level = last_level;
      cfg.unk_mipmapped = !rebased_3d_uav && layout->levels > 1;
      if (samples > 1)
         cfg.samples = agx_translate_sample_count(samples);

      if (layout->tiling == AIL_TILING_LINEAR) {
         cfg.stride = ail_get_linear_stride_B(layout, 0) - 16;
         if (layered) {
            cfg.depth_linear = layers;
            cfg.layer_stride_linear = layout->layer_stride_B - 0x80;
            cfg.extended = true;
         }
      } else {
         cfg.depth = layers;
         cfg.page_aligned_layers = layout->page_aligned_layers;
      }
   }
}

bool
d12_refresh_texture_descriptor(struct d12_device *device, void *out,
                               const struct d12_descriptor *d)
{
   if (!device || !out || !d)
      return false;

   const enum agx_descriptor_generation generation =
      agx_chip_descriptor_generation(device->dev.chip);
   if (generation == AGX_DESCRIPTOR_GENERATION_7 ||
       generation == AGX_DESCRIPTOR_GENERATION_8) {
      d12_refresh_texture_descriptor_agx2(out, d);
      return true;
   }

   struct d12_descriptor shadow = *d;
   struct d12_resource shadow_resource;
   if (generation == AGX_DESCRIPTOR_GENERATION_9 ||
       generation == AGX_DESCRIPTOR_GENERATION_10)
      shadow.address &= (UINT64_C(1) << 40) - 1;
   if ((generation == AGX_DESCRIPTOR_GENERATION_9 ||
        generation == AGX_DESCRIPTOR_GENERATION_10) &&
       d->resource && d->resource->reserved &&
       d->resource->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
      shadow_resource = *d->resource;
      shadow_resource.reserved = false;
      shadow.resource = &shadow_resource;
   }

   struct agx_texture_packed legacy;
   d12_refresh_texture_descriptor_agx2(&legacy, &shadow);
   struct agx_texture_packed null_legacy;
   agx_set_null_texture(&null_legacy);
   struct AGX_TEXTURE unpacked = {0};
   if (!AGX_TEXTURE_unpack(NULL, (const uint8_t *)&legacy, &unpacked))
      return false;

   const struct ail_layout *semantic_layout = NULL;
   if (d->resource) {
      semantic_layout =
         d->texture_plane == 1 &&
               d->resource->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
            ? &d->resource->stencil_layout
            : &d->resource->layout;
   }
   const unsigned semantic_sample_count =
      semantic_layout ? MAX2(semantic_layout->sample_count_sa, 1) : 1;

   if ((generation == AGX_DESCRIPTOR_GENERATION_9 ||
       generation == AGX_DESCRIPTOR_GENERATION_10) &&
       (d->kind == D12_DESC_SRV || d->kind == D12_DESC_UAV) && d->resource &&
       memcmp(&legacy, &null_legacy, sizeof(legacy)) != 0) {
      if (d->resource->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
         uint64_t read_address = d->address;
         if (d->resource->reserved &&
             !agx_addr_is_ro(&device->dev, read_address))
            read_address += device->dev.sparse_ro_offset;
         /* AGX2 stores a sub-16-byte typed-buffer offset in software
          * sideband. No equivalent Gen9/10 field is proven yet. */
         if (read_address & 0xf) {
            d12_set_null_texture_descriptor(device, out);
            return false;
         }
         unpacked.address = read_address & ~UINT64_C(0xf);
      } else {
         uint64_t sparse_address;
         if (d12_sparse_texture_map_address(d, &sparse_address)) {
            unpacked.mode = AGX_IMAGE_MODE_SPARSE;
            unpacked.address = sparse_address;
         } else {
            unpacked.address = d->address;
         }
      }
   }

   if (d12_pack_texture_generation(generation, out, &unpacked,
                                   semantic_sample_count))
      return true;

   d12_set_null_texture_descriptor(device, out);
   return false;
}

bool
d12_set_null_texture_descriptor(struct d12_device *device, void *out)
{
   if (!device || !out)
      return false;

   const enum agx_descriptor_generation generation =
      agx_chip_descriptor_generation(device->dev.chip);
   if (generation == AGX_DESCRIPTOR_GENERATION_7 ||
       generation == AGX_DESCRIPTOR_GENERATION_8) {
      agx_set_null_texture(out);
      return true;
   }

   struct agx_texture_packed legacy;
   agx_set_null_texture(&legacy);
   struct AGX_TEXTURE unpacked = {0};
   if (!AGX_TEXTURE_unpack(NULL, (const uint8_t *)&legacy, &unpacked))
      return false;

   return d12_pack_texture_generation(generation, out, &unpacked, 1);
}

bool
d12_cmd_list_refresh_descriptors(struct d12_cmd_list *list)
{
   bool validate = d12_debug()->validate_bindings;
   struct d12_native_descriptor_layout native_layout;
   if (!list ||
       !d12_native_descriptor_layout(
          (uint32_t)agx_chip_descriptor_generation(list->device->dev.chip),
          &native_layout))
      return false;

   util_dynarray_foreach (&list->descriptor_fixups,
                          struct d12_descriptor_fixup, fixup) {
      if (!fixup->source || fixup->slot >= D12_MAX_RESOURCE_BINDINGS ||
          fixup->slot >= fixup->table_count)
         return false;

      const struct d12_descriptor *d = fixup->source;
      struct d12_descriptor effective = *d;
      d12_apply_texture_shape(&effective, fixup->texture_shape);
      d = &effective;
      unsigned slot = fixup->slot;

      if (validate && d->resource) {
         uint64_t bo_base, bo_size;
         if (!d12_descriptor_resource_span(d, &bo_base, &bo_size) ||
             (d->address && (d->address < bo_base ||
                             d->address - bo_base >= bo_size))) {
            mesa_loge("descriptor refresh: slot %u source address 0x%llx "
                      "is outside its resource BO",
                      slot, (unsigned long long)d->address);
            return false;
         }
      }


      uint64_t *tables = (uint64_t *)d12_list_data_cpu(list, fixup->tables_gpu);
      if (!tables)
         return false;

      tables[slot] =
         d->address ? d->address : list->null_buffer;
      tables[fixup->table_count + slot] =
         d->counter_address
            ? d->counter_address
            : list->null_counters + slot * sizeof(uint32_t);
      tables[2 * fixup->table_count + slot] =
         d12_descriptor_bufinfo(d);
      tables[3 * fixup->table_count + slot] =
         d12_descriptor_sparse_residency_address(d);

      if (fixup->textures_gpu) {
         void *texture = d12_list_data_cpu(
            list, fixup->textures_gpu +
                     (uint64_t)slot * native_layout.texture_stride_B);
         if (!texture)
            return false;
         if (!d12_refresh_texture_descriptor(list->device, texture, d))
            return false;
      }

      if (fixup->pbes_gpu) {
         void *pbe = d12_list_data_cpu(
            list, fixup->pbes_gpu +
                     (uint64_t)slot * native_layout.pbe_stride_B);
         if (!pbe)
            return false;
         if (!d12_refresh_pbe_descriptor(list->device, pbe, d))
            return false;
      }

      if (d12_debug()->dump_descriptor_fixups)
         mesa_logi("descriptor refresh: slot=%u source=%p kind=%u "
                   "address=0x%llx tables=0x%llx textures=0x%llx "
                   "pbes=0x%llx valid=1",
                   slot, (const void *)d, d->kind,
                   (unsigned long long)d->address,
                   (unsigned long long)fixup->tables_gpu,
                   (unsigned long long)fixup->textures_gpu,
                   (unsigned long long)fixup->pbes_gpu);
   }

   util_dynarray_foreach (&list->indirect_constant_fixups,
                          struct d12_indirect_constant_fixup, fixup) {
      const struct d12_resource *source = fixup->source;
      if (!source || !source->bo ||
          source->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
          fixup->source_offset_B > source->desc.Width ||
          fixup->size_B > source->desc.Width - fixup->source_offset_B ||
          source->offset_B > source->bo->size ||
          fixup->source_offset_B > source->bo->size - source->offset_B ||
          fixup->size_B > source->bo->size - source->offset_B -
                              fixup->source_offset_B ||
          !d12_list_stream_contains(list, fixup->destination_gpu,
                                    fixup->size_B)) {
         mesa_loge("indirect constant refresh references invalid storage");
         return false;
      }

      const uint8_t *source_map = agx_bo_map(source->bo);
      if (!source_map) {
         mesa_loge("indirect constant refresh cannot map source storage");
         return false;
      }

      const uint8_t *src =
         source_map + source->offset_B + fixup->source_offset_B;
      uint8_t *dst = d12_list_data_cpu(list, fixup->destination_gpu);
      memcpy(dst, src, fixup->size_B);

      if (d12_debug()->dump_indirect)
         mesa_logi("indirect constants: source=%p+0x%llx destination=0x%llx "
                   "size=%u valid=1", (void *)source,
                   (unsigned long long)fixup->source_offset_B,
                   (unsigned long long)fixup->destination_gpu,
                   fixup->size_B);
   }

   util_dynarray_foreach (&list->indirect_argument_fixups,
                          struct d12_indirect_argument_fixup, fixup) {
      const struct d12_resource *source = fixup->source;
      const struct d12_resource *count = fixup->count_source;
      if (!source || !source->bo || !count || !count->bo ||
          source->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
          count->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
          fixup->source_offset_B > source->desc.Width ||
          fixup->size_B > source->desc.Width - fixup->source_offset_B ||
          fixup->count_offset_B > count->desc.Width ||
          sizeof(uint32_t) > count->desc.Width - fixup->count_offset_B ||
          source->offset_B > source->bo->size ||
          fixup->source_offset_B > source->bo->size - source->offset_B ||
          fixup->size_B > source->bo->size - source->offset_B -
                              fixup->source_offset_B ||
          count->offset_B > count->bo->size ||
          fixup->count_offset_B > count->bo->size - count->offset_B ||
          sizeof(uint32_t) > count->bo->size - count->offset_B -
                                fixup->count_offset_B ||
          !d12_list_stream_contains(list, fixup->destination_gpu,
                                    fixup->size_B)) {
         mesa_loge("indirect argument refresh references invalid storage");
         return false;
      }

      const uint8_t *count_map = agx_bo_map(count->bo);
      const uint8_t *source_map = agx_bo_map(source->bo);
      if (!count_map || !source_map) {
         mesa_loge("indirect argument refresh cannot map source storage");
         return false;
      }

      const uint8_t *count_cpu =
         count_map + count->offset_B + fixup->count_offset_B;
      uint32_t command_count;
      memcpy(&command_count, count_cpu, sizeof(command_count));

      uint8_t *dst = d12_list_data_cpu(list, fixup->destination_gpu);
      if (fixup->command_index < command_count) {
         const uint8_t *src =
            source_map + source->offset_B + fixup->source_offset_B;
         memcpy(dst, src, fixup->size_B);
      } else {
         memset(dst, 0, fixup->size_B);
      }

      if (d12_debug()->dump_indirect)
         mesa_logi("indirect args: source=%p+0x%llx count=%u index=%u "
                   "destination=0x%llx size=%u valid=1", (void *)source,
                   (unsigned long long)fixup->source_offset_B, command_count,
                   fixup->command_index,
                   (unsigned long long)fixup->destination_gpu,
                   fixup->size_B);
   }

   /* The stream BO is coherent, but order the CPU descriptor writes before
    * the queue ioctl makes the command stream visible to the GPU.
    */
   __sync_synchronize();
   return true;
}

static int
d12_sparse_registry_record_compare(const void *a, const void *b)
{
   const struct d12_sparse_registry_record *ra = a;
   const struct d12_sparse_registry_record *rb = b;
   return ra->base < rb->base ? -1 : ra->base > rb->base;
}

static bool
d12_sparse_registry_grow_capacity(uint32_t count, uint32_t *capacity)
{
   if (!count || !capacity)
      return false;

   uint32_t grown = 16;
   while (grown < count) {
      if (grown > UINT32_MAX / 2) {
         grown = UINT32_MAX;
         break;
      }
      grown *= 2;
   }

   *capacity = grown;
   return grown >= count;
}

bool
d12_cmd_list_refresh_sparse_registry(struct d12_cmd_list *list)
{
   if (!list || !list->uses_sparse_registry)
      return true;

   struct d12_sparse_registry_view *view =
      (struct d12_sparse_registry_view *)d12_list_data_cpu(
         list, list->sparse_registry_view_gpu);
   if (!view)
      return false;

   struct d12_device *device = list->device;
   for (;;) {
      struct d12_sparse_registry_record *records =
         list->sparse_registry_bo ? agx_bo_map(list->sparse_registry_bo)
                                  : NULL;
      uint64_t live = 0;

      simple_mtx_lock(&device->sparse_resource_lock);
      list_for_each_entry(struct d12_resource, resource,
                          &device->sparse_resources, sparse_link) {
         if (resource->reserved &&
             resource->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
            ++live;
      }

      if (live > UINT32_MAX ||
          (live && (!records || live > list->sparse_registry_capacity))) {
         simple_mtx_unlock(&device->sparse_resource_lock);

         if (live > UINT32_MAX)
            return false;

         uint32_t capacity;
         uint64_t size_B;
         if (!d12_sparse_registry_grow_capacity((uint32_t)live, &capacity) ||
             !d12_sparse_registry_allocation_size(capacity, AIL_PAGESIZE,
                                                  &size_B) ||
             size_B > SIZE_MAX)
            return false;

         struct agx_bo *bo = agx_bo_create(
            &device->dev, (size_t)size_B, AIL_PAGESIZE, AGX_BO_WRITEBACK,
            "d3d12 ExecuteIndirect sparse registry");
         if (!bo || !agx_bo_map(bo)) {
            if (bo)
               agx_bo_unreference(&device->dev, bo);
            return false;
         }

         struct agx_bo *old = list->sparse_registry_bo;
         list->sparse_registry_bo = bo;
         list->sparse_registry_capacity = capacity;
         if (old)
            agx_bo_unreference(&device->dev, old);
         continue;
      }

      uint32_t count = 0;
      list_for_each_entry(struct d12_resource, resource,
                          &device->sparse_resources, sparse_link) {
         if (!resource->reserved ||
             resource->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER)
            continue;

         if (!resource->addr || !resource->sparse_size_B ||
             resource->addr > UINT64_MAX - resource->sparse_size_B ||
             !resource->sparse_map || !resource->sparse_map->va) {
            simple_mtx_unlock(&device->sparse_resource_lock);
            return false;
         }

         records[count++] = (struct d12_sparse_registry_record){
            .base = resource->addr,
            .size = resource->sparse_size_B,
            .residency = resource->sparse_map->va->addr,
         };
      }
      simple_mtx_unlock(&device->sparse_resource_lock);

      if (count > 1)
         qsort(records, count, sizeof(*records),
               d12_sparse_registry_record_compare);

      uint64_t previous_end = 0;
      for (uint32_t i = 0; i < count; ++i) {
         if (i && records[i].base < previous_end)
            return false;
         previous_end = records[i].base + records[i].size;
      }

      *view = (struct d12_sparse_registry_view){
         .records = count ? list->sparse_registry_bo->va->addr : 0,
         .count = count,
         .capacity = list->sparse_registry_capacity,
      };

      /* Publish snapshot contents before the fixed stream view which names
       * them, then before the queue ioctl exposes either allocation. */
      __sync_synchronize();
      return true;
   }
}

/*
 * Dispatch a pipeline compiled from the application's DXIL. Mirrors
 * hk_dispatch_with_usc: build USC words binding the shader, its registers
 * and our resource-address uniforms, then emit a CDM launch.
 */
/*
 * The uniform registers a compute launch publishes. Split out because the
 * geometry programs need the same USC construction as an ordinary
 * dispatch plus the two poly parameter blocks, and duplicating ~60 lines
 * of USC packing to add two uniforms is how the two paths drift apart.
 *
 * A zero address means "not published", which is not the same as
 * publishing zero: the geometry programs read these as pointers.
 */
struct d12_dispatch_uniforms {
   uint64_t root;
   uint64_t textures;
   uint64_t pbes;
   uint64_t heap;
   uint64_t heap_textures;
   uint64_t heap_pbes;
   uint64_t heap_samplers;
   uint64_t draw;
   uint64_t vertex_params;
   uint64_t geom_params;
   uint64_t tess_params;
   uint64_t mesh_params;
   /* Address of a 64-bit writable sink POINTER.  Poly's pre-GS shader
    * unconditionally materialises four pipeline-statistics addresses even
    * when no such D3D12 query is active.  Like the other pointer-valued
    * preamble fields, agx_usc_uniform loads the pointer from this address.
    */
   uint64_t stat_query;
   /* Address of the VS output MASK, not the mask itself: agx_usc_uniform
    * fills registers FROM memory, and load_vs_outputs_poly reads a 64-bit
    * value. The geometry body computes every vertex address in the output
    * buffer from this mask, so a wrong one is not a wrong value, it is a
    * wrong POINTER.
    */
   uint64_t vs_outputs;

   /* Address of a POINTER to the vertex-attribute block, not the block:
    * agx_usc_uniform fills registers from memory, and the shader reads a
    * 64-bit base it then adds offsets to. Same indirection as the mesh
    * and geometry parameter blocks.
    */
   uint64_t vb_params;
};

static bool
d12_sampler_visible(D3D12_SHADER_VISIBILITY declared,
                    D3D12_SHADER_VISIBILITY stage)
{
   return declared == D3D12_SHADER_VISIBILITY_ALL || declared == stage;
}

static const struct d12_descriptor *
d12_list_bound_sampler(struct d12_cmd_list *list, bool compute_bindings,
                       D3D12_SHADER_VISIBILITY visibility, unsigned logical,
                       bool *visible_slot)
{
   const struct d12_descriptor *dynamic =
      compute_bindings ? list->compute_sampler_table
                       : list->graphics_sampler_table;
   const struct d12_root_signature *rs =
      compute_bindings ? list->compute_rs : list->graphics_rs;
   const struct d12_descriptor *chosen = NULL;

   *visible_slot = false;
   if (rs) {
      for (unsigned i = 0; i < rs->map.count; i++) {
         if (rs->map.entries[i].klass == D12_RES_SAMPLER &&
             rs->map.entries[i].slot == logical &&
             (rs->map.entries[i].visibility == D3D12_SHADER_VISIBILITY_ALL ||
              rs->map.entries[i].visibility == visibility)) {
            *visible_slot = true;
            break;
         }
      }

      for (uint32_t i = 0; i < rs->layout.num_static_samplers; i++) {
         const D3D12_STATIC_SAMPLER_DESC1 *s =
            &rs->layout.static_samplers[i];
         if (!d12_sampler_visible(s->ShaderVisibility, visibility))
            continue;
         if (d12_binding_map_find_visible(
                &rs->map, D12_RES_SAMPLER, s->RegisterSpace,
                s->ShaderRegister, visibility) == logical) {
            chosen = &rs->static_samplers[i];
            break;
         }
      }
   }

   if (!chosen && dynamic[logical].sampler_kind == D12_DESC_SAMPLER)
      chosen = &dynamic[logical];

   return chosen;
}

/* Upload sampler state which is not present in the selected AGX descriptor:
 * arbitrary point-border colours, plus D3D's floating-point LOD bias on the
 * AGX2/G15 Gen4 layout. HAL200/Gen5 publish a zero sideband bias because the
 * signed 5:6 value is already in the hardware word. */
static bool
d12_list_upload_custom_borders(struct d12_cmd_list *list,
                               const struct agx_shader_info *info,
                               bool compute_bindings,
                               D3D12_SHADER_VISIBILITY visibility,
                               uint64_t *out_gpu)
{
   bool validate = d12_debug()->validate_bindings;
   unsigned count = info->uses_txf ? MAX2(info->sampler_state_count, 1)
                                   : info->sampler_state_count;
   *out_gpu = 0;

   /* A bindless sampler handle indexes the coherent 40-byte D3D12 sampler
    * heap.  Besides the exact border/bias metadata, each slot carries the
    * deduplicated native AGX index.  Publish that heap directly so updates
    * after command-list recording remain visible at execution time. */
   if (info->uses_sampler_heap) {
      struct d12_descriptor_heap *heap = list->sampler_descriptor_heap;
      if (!heap || !heap->bo || !heap->sampler_bo ||
          heap->desc.Type != D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) {
         mesa_loge("bindless custom-sampler metadata has no shader-visible "
                   "sampler heap bound");
         list->state = D12_LIST_ERROR;
         return false;
      }

      uint64_t table_gpu = heap->bo->va->addr;
      *out_gpu = d12_list_upload(list, &table_gpu, sizeof(table_gpu), 8);
      if (!*out_gpu)
         return false;

      if (validate) {
         mesa_logi("binding validation: bindless sampler metadata="
                   "0x%llx count=%u native=%u home=0x%llx valid=1",
                   (unsigned long long)table_gpu, heap->desc.NumDescriptors,
                   heap->sampler_unique_count, (unsigned long long)*out_gpu);
      }
      return true;
   }

   if (count <= 1)
      return true;
   if (count > 17) {
      mesa_loge("shader requires %u sampler states; AGX supports 17", count);
      list->state = D12_LIST_ERROR;
      return false;
   }

   struct d12_custom_border_entry entries[16] = {{0}};
   for (unsigned logical = 0; logical < count - 1; logical++) {
      bool visible_slot;
      const struct d12_descriptor *chosen = d12_list_bound_sampler(
         list, compute_bindings, visibility, logical, &visible_slot);

      if (chosen) {
         entries[logical].flags = chosen->sampler_reserved;
         entries[logical].lod_bias = chosen->sampler_lod_bias;
         memcpy(entries[logical].color, chosen->sampler_border,
                sizeof(entries[logical].color));
      } else if (validate && visible_slot) {
         mesa_loge("binding validation: sampler s%u has no visible static "
                   "or dynamic descriptor for stage %u", logical,
                   (unsigned)visibility);
         list->state = D12_LIST_ERROR;
         return false;
      }
   }

   size_t size = (count - 1) * sizeof(entries[0]);
   uint64_t table_gpu = d12_list_upload(list, entries, size, 8);
   if (!table_gpu)
      return false;

   if (validate && ((table_gpu & 7) ||
                    !d12_list_stream_contains(list, table_gpu, size))) {
      mesa_loge("binding validation: custom-border table 0x%llx+%zu is "
                "outside the aligned command-list stream BO",
                (unsigned long long)table_gpu, size);
      list->state = D12_LIST_ERROR;
      return false;
   }

   /* USC UNIFORM records copy bytes from the supplied GPU address. The
    * shader uniform is a pointer, so publish an address-of-address home
    * cell, not the table itself (whose first eight bytes are flags/pad). */
   *out_gpu = d12_list_upload(list, &table_gpu, sizeof(table_gpu), 8);
   if (!*out_gpu)
      return false;

   if (validate && ((*out_gpu & 7) ||
                    !d12_list_stream_contains(list, *out_gpu,
                                              sizeof(table_gpu)))) {
      mesa_loge("binding validation: custom-border pointer home 0x%llx+%zu "
                "is outside the aligned command-list stream BO",
                (unsigned long long)*out_gpu, sizeof(table_gpu));
      list->state = D12_LIST_ERROR;
      return false;
   }

   if (d12_debug()->dump_samplers) {
      mesa_logi("custom borders: stage=%u compute=%u entries=%u "
                "table=0x%llx home=0x%llx stream=0x%llx..0x%llx "
                "valid=%u/%u",
                (unsigned)visibility, compute_bindings, count - 1,
                (unsigned long long)table_gpu,
                (unsigned long long)*out_gpu,
                (unsigned long long)list->stream_bo->va->addr,
                (unsigned long long)(list->stream_bo->va->addr +
                                     list->stream_bo->size),
                d12_list_stream_contains(list, table_gpu, size),
                d12_list_stream_contains(list, *out_gpu,
                                         sizeof(table_gpu)));
      for (unsigned i = 0; i < count - 1; i++) {
         mesa_logi("  sampler%u: flags=0x%x lod_bias=%f "
                   "color=%08x %08x %08x %08x",
                   i, entries[i].flags, entries[i].lod_bias, entries[i].color[0],
                   entries[i].color[1], entries[i].color[2],
                   entries[i].color[3]);
      }
   }

   return true;
}

/* Upload one contiguous sampler state file. AGX reserves hardware state 0
 * for texture Load; application sampler S therefore lives at S + 1. Using
 * one USC record for the complete file matches the proven Gallium path and
 * avoids overlapping state records with unrelated backing addresses. */
static bool
d12_list_push_samplers(struct d12_cmd_list *list,
                       struct agx_usc_builder *b,
                       const struct agx_shader_info *info,
                       bool compute_bindings,
                       D3D12_SHADER_VISIBILITY visibility)
{
   bool validate = d12_debug()->validate_bindings;
   unsigned count = info->uses_txf ? MAX2(info->sampler_state_count, 1)
                                   : info->sampler_state_count;

   if (d12_debug()->dump_samplers)
      mesa_logi("sampler requirement: stage=%u compute=%u count=%u txf=%u",
                (unsigned)visibility, compute_bindings, count,
                info->uses_txf);

   if (!count)
      return true;
   if (count > 17) {
      mesa_loge("shader requires %u sampler states; AGX supports 17", count);
      list->state = D12_LIST_ERROR;
      return false;
   }

   const unsigned stride =
      agx_chip_sampler_heap_element_size(list->device->dev.chip);
   if (!stride) {
      mesa_loge("sampler heap ABI is unknown for this AGX identity");
      list->state = D12_LIST_ERROR;
      return false;
   }

   uint32_t states[17 * (32 / sizeof(uint32_t))] = {0};
   for (unsigned i = 0; i < count; i++)
      agx_pack_txf_sampler(
         (struct agx_sampler_packed *)((uint8_t *)states +
                                       ((size_t)i * stride)));

   for (unsigned hw = 1; hw < count; hw++) {
      unsigned logical = hw - 1;
      bool visible_slot;
      const struct d12_descriptor *chosen = d12_list_bound_sampler(
         list, compute_bindings, visibility, logical, &visible_slot);

      if (chosen) {
         memcpy((uint8_t *)states + ((size_t)hw * stride), &chosen->sampler,
                sizeof(chosen->sampler));
      } else if (validate && visible_slot) {
         mesa_loge("binding validation: sampler s%u has no visible static "
                   "or dynamic descriptor for stage %u", logical,
                   (unsigned)visibility);
         list->state = D12_LIST_ERROR;
         return false;
      }
   }

   size_t state_size = count * stride;
   uint64_t gpu = d12_list_upload(list, states, state_size, stride);
   if (!gpu)
      return false;

   if (validate && ((gpu & 7) ||
                    !d12_list_stream_contains(list, gpu, state_size))) {
      mesa_loge("binding validation: sampler file 0x%llx+%zu is outside "
                "the aligned command-list stream BO",
                (unsigned long long)gpu, state_size);
      list->state = D12_LIST_ERROR;
      return false;
   }

   if (d12_debug()->dump_samplers) {
      mesa_logi("samplers: stage=%u compute=%u states=%u logical=%u "
                "gpu=0x%llx stream=0x%llx..0x%llx valid=%u",
                (unsigned)visibility, compute_bindings, count,
                (compute_bindings ? list->compute_rs : list->graphics_rs)
                   ? (compute_bindings ? list->compute_rs : list->graphics_rs)
                        ->map.sampler_count
                   : 0,
                (unsigned long long)gpu,
                (unsigned long long)list->stream_bo->va->addr,
                (unsigned long long)(list->stream_bo->va->addr +
                                     list->stream_bo->size),
                d12_list_stream_contains(list, gpu, state_size));
      for (unsigned i = 0; i < count; i++) {
         const uint8_t *raw = (const uint8_t *)&states[i];
         mesa_logi("  ss%u: %02x %02x %02x %02x %02x %02x %02x %02x",
                   i, raw[0], raw[1], raw[2], raw[3], raw[4], raw[5],
                   raw[6], raw[7]);
      }
   }

   agx_usc_pack(b, SAMPLER, cfg) {
      cfg.start = 0;
      cfg.count = count;
      cfg.buffer = gpu;
   }
   return true;
}

/* Build the USC words for one compute launch; returns the USC offset, or
 * 0 on failure.
 */
static uint32_t
d12_list_build_compute_usc(struct d12_cmd_list *list, uint64_t code_va,
                           const struct agx_shader_info *info,
                           const struct d12_dispatch_uniforms *u,
                           bool compute_bindings,
                           D3D12_SHADER_VISIBILITY visibility,
                           uint8_t **out_cpu)
{
   struct d12_device *device = list->device;

   uint32_t usc;
   size_t usc_size = agx_usc_size(19);
   uint8_t *usc_cpu = d12_list_alloc_usc(list, usc_size, &usc);
   if (!usc_cpu)
      return 0;

   if (out_cpu)
      *out_cpu = usc_cpu;

   struct agx_usc_builder b = agx_usc_builder(usc_cpu, usc_size);

   /* Four 16-bit uniform registers hold the 64-bit root pointer. */
   agx_usc_uniform(&b, D12_UNIFORM_ROOT, D12_UNIFORM_ROOT_REGS, u->root);
   if (u->textures) {
      agx_usc_uniform(&b, D12_UNIFORM_TEXTURES, D12_UNIFORM_TEXTURE_REGS,
                      u->textures);
   }

   /* The geometry programs read their parameter blocks from these two,
    * which the DXIL translator already lowers poly's parameter
    * intrinsics to. Nothing else publishes them.
    */
   /* Ascending base order: 4, 8, 20, 24, 28, 32, 36, 52. */
   if (u->draw) {
      agx_usc_uniform(&b, D12_UNIFORM_DRAW, D12_UNIFORM_DRAW_REGS, u->draw);
   }
   if (u->tess_params) {
      agx_usc_uniform(&b, D12_UNIFORM_TESS_PARAMS, D12_UNIFORM_PARAM_REGS,
                      u->tess_params);
   }
   if (u->geom_params) {
      agx_usc_uniform(&b, D12_UNIFORM_GEOM_PARAMS, D12_UNIFORM_PARAM_REGS,
                      u->geom_params);
   }
   if (u->vs_outputs) {
      agx_usc_uniform(&b, D12_UNIFORM_VS_OUTPUTS, D12_UNIFORM_PARAM_REGS,
                      u->vs_outputs);
   }
   if (u->vertex_params) {
      agx_usc_uniform(&b, D12_UNIFORM_VERTEX_PARAMS, D12_UNIFORM_PARAM_REGS,
                      u->vertex_params);
   }
   if (u->mesh_params) {
      agx_usc_uniform(&b, D12_UNIFORM_MESH_PARAMS, D12_UNIFORM_MESH_REGS,
                      u->mesh_params);
   }
   if (u->stat_query) {
      agx_usc_uniform(&b, D12_UNIFORM_STAT_QUERY, D12_UNIFORM_STAT_REGS,
                      u->stat_query);
   }

   /*
    * VERTEX ATTRIBUTE FETCH, and this is a COMPUTE builder on purpose.
    *
    * The software VS that a tessellation, geometry or mesh pipeline
    * substitutes for the hardware vertex stage is a compute shader, and
    * compile_sw_vs_stage runs agx_nir_lower_vbo over it -- so it fetches
    * its attributes through exactly this uniform. The graphics builder
    * published it and this one did not, so every such dispatch read the
    * vertex buffer through uniform registers nobody had written, i.e.
    * through whatever the previous dispatch happened to leave in them.
    *
    * It is the same class of bug as the rodata comment directly below,
    * and it is what wedged the GPU on test_quad_tessellation_dxil: a read
    * of an unmapped address in a region we never map, different every
    * run, belonging to none of our allocations. Our own g7 tessellation
    * probe never caught it because it derives its geometry from
    * SV_VertexID and so fetches no attributes at all.
    */
   if (u->vb_params) {
      agx_usc_uniform(&b, D12_UNIFORM_VB_PARAMS, D12_UNIFORM_VB_REGS,
                      u->vb_params);
   }

   if (u->heap) {
      agx_usc_uniform(&b, D12_UNIFORM_HEAP, D12_UNIFORM_HEAP_REGS,
                      u->heap);
   }
   if (u->pbes) {
      agx_usc_uniform(&b, D12_UNIFORM_PBES, D12_UNIFORM_PBE_REGS,
                      u->pbes);
   }
   if (u->heap_textures) {
      agx_usc_uniform(&b, D12_UNIFORM_HEAP_TEXTURES,
                      D12_UNIFORM_HEAP_TEXTURE_REGS, u->heap_textures);
   }
   if (u->heap_pbes) {
      agx_usc_uniform(&b, D12_UNIFORM_HEAP_PBES,
                      D12_UNIFORM_HEAP_PBE_REGS, u->heap_pbes);
   }
   if (u->heap_samplers) {
      agx_usc_uniform(&b, D12_UNIFORM_HEAP_SAMPLERS,
                      D12_UNIFORM_HEAP_SAMPLER_REGS, u->heap_samplers);
   }

   uint64_t custom_borders;
   if (!d12_list_upload_custom_borders(list, info, compute_bindings,
                                       visibility, &custom_borders))
      return 0;
   if (custom_borders) {
      agx_usc_uniform(&b, D12_UNIFORM_CUSTOM_BORDERS,
                      D12_UNIFORM_CUSTOM_BORDER_REGS, custom_borders);
   }

   /*
    * Constants the compiler promoted out of the instruction stream live in
    * the shader BO's rodata, and the shader reads them as uniforms above
    * its push range. Nothing else publishes them: skip this and those
    * uniforms hold whatever the last dispatch left there, so a shader
    * computes with stale numbers instead of its own constants.
    */
   if (info->rodata.size_16)
      agx_usc_immediates(&b, &info->rodata, code_va);

   if (!d12_list_push_samplers(list, &b, info, compute_bindings, visibility))
      return 0;

   /* The compiler publishes static TGSM usage in info->local_size. Mirror
    * Honeykrisp's non-fragment program binding so AGX allocates that shared
    * memory for the workgroup instead of advertising "no shared memory". */
   agx_usc_shared_non_fragment(&b, info, 0);

   agx_usc_pack(&b, SHADER, cfg) {
      cfg.code = agx_usc_addr(&device->dev, code_va + info->main_offset);
      cfg.unk_2 = 3; /* compute */
   }

   agx_usc_pack(&b, REGISTERS, cfg)
      cfg.register_count = info->nr_gprs;

   if (info->has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code =
            agx_usc_addr(&device->dev, code_va + info->preamble_offset);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   if (d12_debug()->dump_gs_usc) {
      mesa_logi("compute USC: code=0x%llx main=%u gprs=%u push=%u "
                "preamble=%d tex=%u txf=%d pregprs=%u "
                "root=0x%llx vp=0x%llx gp=0x%llx vso=0x%llx stat=0x%llx",
                (unsigned long long)code_va, info->main_offset, info->nr_gprs,
                info->push_count, info->has_preamble,
                info->texture_state_count, info->uses_txf, info->nr_preamble_gprs,
                (unsigned long long)u->root,
                (unsigned long long)u->vertex_params,
                (unsigned long long)u->geom_params,
                (unsigned long long)u->vs_outputs,
                (unsigned long long)u->stat_query);
      for (unsigned i = 0; i < 48; i += 16) {
         mesa_logi("  %02x: %02x %02x %02x %02x %02x %02x %02x %02x "
                   "%02x %02x %02x %02x %02x %02x %02x %02x", i,
                   usc_cpu[i], usc_cpu[i+1], usc_cpu[i+2], usc_cpu[i+3],
                   usc_cpu[i+4], usc_cpu[i+5], usc_cpu[i+6], usc_cpu[i+7],
                   usc_cpu[i+8], usc_cpu[i+9], usc_cpu[i+10], usc_cpu[i+11],
                   usc_cpu[i+12], usc_cpu[i+13], usc_cpu[i+14], usc_cpu[i+15]);
      }
   }

   return usc;
}

/* Pack the launch word for one compute program. */
static struct agx_cdm_launch_word_0_packed
d12_compute_launch_word(const struct agx_shader_info *info)
{
   struct agx_cdm_launch_word_0_packed launch;
   agx_pack(&launch, CDM_LAUNCH_WORD_0, cfg) {
      cfg.texture_state_register_count = info->texture_state_count;
      /* See the note in d12_pipeline.c: txf needs a sampler counted. */
      cfg.sampler_state_register_count = agx_translate_sampler_state_count(
         info->uses_txf ? MAX2(info->sampler_state_count, 1)
                        : info->sampler_state_count,
         false);
      cfg.uniform_register_count = info->push_count;
      cfg.preshader_register_count = info->nr_preamble_gprs;
   }
   return launch;
}

static bool
d12_segment_bind_sampler_heap(struct d12_cmd_list *list,
                              struct d12_segment *seg, bool required)
{
   if (!required)
      return true;

   struct d12_descriptor_heap *heap = list->sampler_descriptor_heap;
   if (!heap || !heap->sampler_bo ||
       heap->desc.Type != D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) {
      mesa_loge("shader uses bindless samplers with no sampler heap bound");
      list->state = D12_LIST_ERROR;
      return false;
   }
   if (heap->sampler_unique_count == UINT32_MAX ||
       heap->sampler_unique_count > 1024) {
      mesa_loge("sampler heap has more than 1024 distinct native states; "
                "AGX firmware supports 1024");
      list->state = D12_LIST_ERROR;
      return false;
   }

   d12_list_track_read_heap(list, heap, 0, 0, false);

   uint64_t address = heap->sampler_bo->va->addr;
   if (seg->sampler_heap &&
       (seg->sampler_heap != address ||
        seg->sampler_count != heap->sampler_unique_count)) {
      mesa_loge("one hardware segment references multiple sampler heaps");
      list->state = D12_LIST_ERROR;
      return false;
   }

   seg->sampler_heap = address;
   seg->sampler_count = heap->sampler_unique_count;
   if (d12_debug()->dump_samplers) {
      mesa_logi("bindless sampler compaction: logical=%u native=%u "
                "heap=0x%llx",
                heap->desc.NumDescriptors, heap->sampler_unique_count,
                (unsigned long long)address);
   }
   return true;
}

/* Metadata admission belongs only to the registered, validated noop fd. */
bool
d12_g17p_txf_metadata_enabled(const struct agx_device *dev)
{
#ifndef _WIN32
   return dev && dev->chip == AGX_CHIP_G17P && dev->g17p_render_context_bo &&
          d12_debug()->shim_texture_metadata &&
          asahi_noop_d3d12_graphics_qualification_allowed &&
          asahi_noop_d3d12_graphics_qualification_allowed(dev->fd);
#else
   (void)dev;
   return false;
#endif
}

/* The bounded native indexed txf producer uses the same physical baseline
 * and exact-prior-barrier contract as direct compute. A shim flag cannot
 * admit a real fd, and the physical baseline cannot admit a noop fd. */
bool
d12_g17p_txf_physical_enabled(const struct agx_device *dev)
{
#ifndef _WIN32
   const char *baseline = getenv("AURORA_D3D12_PHYSICAL_BASELINE");
   return dev && dev->chip == AGX_CHIP_G17P &&
          baseline && !strcmp(baseline, "1") &&
          dev->g17p_render_context_bo &&
          (dev->params.features & DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS) &&
          d12_platform_is_real_asahi_render_fd(dev->fd);
#else
   (void)dev;
   return false;
#endif
}

/* Monolithic native compute, including bounded indexed txf. Indirect,
 * API samplers, preamble and legacy scratch retain their own guards. */
static bool
d12_list_append_g17p_pso(struct d12_cmd_list *list,
                        const struct d12_pipeline_state *pso,
                        const struct d12_dispatch_uniforms *u,
                        unsigned x, unsigned y, unsigned z)
{
   struct agx_device *dev = &list->device->dev;
   const struct agx_shader_info *info = &pso->info;
   const bool native_txf = pso->g17p_native_txf;
   if (dev->chip != AGX_CHIP_G17P || !pso->g17p_compute_container ||
       (native_txf && !d12_g17p_txf_metadata_enabled(dev) &&
        !d12_g17p_txf_physical_enabled(dev)) ||
       !pso->code_bo || !pso->code_va || !u->root ||
       !info->main_size || !info->nr_gprs || !info->agx3_staging_known ||
       info->has_preamble || info->nr_preamble_gprs || info->scratch_size ||
       info->preamble_scratch_size || info->agx3_uniform_dma_valid ||
       info->imageblock_stride || info->texture_state_count ||
       info->uses_sampler_heap ||
       (native_txf ? (!info->uses_txf || info->sampler_state_count != 1)
                     : (info->sampler_state_count || info->uses_txf)) ||
       info->uses_ray_traversal ||
       (info->rodata.size_16 && !pso->g17p_rodata_va))
      goto fail;

   struct agx3_ioto_uniform_range ranges[10];
   unsigned range_count = 0;
   if (native_txf) {
      /* The compute binding upload this dispatch consumes must still be
       * cached: its texture table extent is what the prefix publishes. */
      const struct d12_binding_upload_cache *cache = NULL;
      for (unsigned w = 0; w < D12_BINDING_CACHE_WAYS; w++) {
         const struct d12_binding_upload_cache *e =
            &list->binding_upload_cache[1][w];
         if (e->valid && e->root == u->root && e->textures == u->textures)
            cache = e;
      }
      struct agx_bo *sampler_bo = list->device->rodata_bo;
      if (!u->textures || !cache ||
          !sampler_bo || !sampler_bo->va || sampler_bo->size < 32 ||
          !d12_list_stream_contains(list, u->root, sizeof(uint64_t)) ||
          !d12_list_stream_contains(list, u->textures, sizeof(uint64_t)) ||
          !d12_list_stream_contains(list, cache->texture_table,
                                    cache->texture_table_size_B))
         goto fail;
      uint8_t *stream_cpu = d12_list_stream_map(list);
      if (!stream_cpu)
         goto fail;
      const uint8_t *texture_cell_cpu =
         stream_cpu + (u->textures - list->stream_bo->va->addr);
      uint64_t texture_table;
      memcpy(&texture_table, texture_cell_cpu, sizeof(texture_table));
      if (texture_table != cache->texture_table)
         goto fail;
      const uint64_t sampler_table = sampler_bo->va->addr;
      const uint64_t sampler_cell = d12_list_upload(
         list, &sampler_table, sizeof(sampler_table), 8);
      const struct d12_g17p_txf_prefix_sources sources = {
         .root_cell = u->root, .texture_cell = u->textures,
         .sampler_cell = sampler_cell, .texture_table = texture_table,
         .texture_table_size_B = cache->texture_table_size_B,
         .sampler_table = sampler_table,
      };
      struct d12_g17p_txf_range prefix[4];
      if (!d12_g17p_txf_prefix_ranges(&sources, prefix))
         goto fail;
      for (unsigned i = 0; i < 4; ++i)
         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = prefix[i].start_halfs,
            .size_halfs = prefix[i].size_halfs,
            .address = prefix[i].address,
         };
      if (agx_bo_sync_for_device(sampler_bo, 0, 32) ||
          agx_bo_sync_for_device(list->stream_bo,
             texture_table - list->stream_bo->va->addr,
             cache->texture_table_size_B))
         goto fail;
      d12_list_track_read_bo(list, sampler_bo);
      d12_list_track_read_range(list, texture_table,
                                cache->texture_table_size_B);
      if (!pso->code_bo->va || pso->code_va < pso->code_bo->va->addr ||
          info->main_size > pso->code_bo->size ||
          pso->code_va - pso->code_bo->va->addr >
             pso->code_bo->size - info->main_size)
         goto fail;
      const uint8_t *code_cpu = agx_bo_map(pso->code_bo);
      const uint8_t *sampler_cpu = agx_bo_map(sampler_bo);
      if (!code_cpu || !sampler_cpu)
         goto fail;
      blake3_hash code_hash, table_hash, sampler_hash;
      char code_hex[BLAKE3_HEX_LEN], table_hex[BLAKE3_HEX_LEN];
      char sampler_hex[BLAKE3_HEX_LEN];
      _mesa_blake3_compute(code_cpu + (pso->code_va - pso->code_bo->va->addr),
                           info->main_size, code_hash);
      _mesa_blake3_compute(stream_cpu + (texture_table - list->stream_bo->va->addr),
                           cache->texture_table_size_B, table_hash);
      _mesa_blake3_compute(sampler_cpu, 32, sampler_hash);
      _mesa_blake3_format(code_hex, code_hash);
      _mesa_blake3_format(table_hex, table_hash);
      _mesa_blake3_format(sampler_hex, sampler_hash);
      mesa_logi("native Texture.Load CODE: va=0x%llx size=%u blake3=%s "
                "table-blake3=%s sampler32-blake3=%s",
                (unsigned long long)pso->code_va, info->main_size,
                code_hex, table_hex, sampler_hex);
      mesa_logi("native Texture.Load metadata: table=0x%llx span=%llu "
                "texture-cell=0x%llx sampler-cell=0x%llx sampler=0x%llx "
                "root-cell=0x%llx halves=0/4/8/query16 fd-policy=%s",
                (unsigned long long)texture_table,
                (unsigned long long)cache->texture_table_size_B,
                (unsigned long long)u->textures,
                (unsigned long long)sampler_cell,
                (unsigned long long)sampler_table,
                (unsigned long long)u->root,
                d12_g17p_txf_metadata_enabled(dev)
                   ? "registered-shim-only" : "physical-asahi");
   }
#define D12_G17P_RANGE(field, base, size)                                       do {                                                                             if (u->field) {                                                                  ranges[range_count++] = (struct agx3_ioto_uniform_range){                        .start_halfs = base, .size_halfs = size, .address = u->field,             };                                                                        }                                                                         } while (0)
   if (!native_txf) {
      D12_G17P_RANGE(root, D12_UNIFORM_ROOT, D12_UNIFORM_ROOT_REGS);
      D12_G17P_RANGE(textures, D12_UNIFORM_TEXTURES, D12_UNIFORM_TEXTURE_REGS);
   }
   D12_G17P_RANGE(heap, D12_UNIFORM_HEAP, D12_UNIFORM_HEAP_REGS);
   D12_G17P_RANGE(pbes, D12_UNIFORM_PBES, D12_UNIFORM_PBE_REGS);
   D12_G17P_RANGE(heap_textures, D12_UNIFORM_HEAP_TEXTURES,
                  D12_UNIFORM_HEAP_TEXTURE_REGS);
   D12_G17P_RANGE(heap_pbes, D12_UNIFORM_HEAP_PBES, D12_UNIFORM_HEAP_PBE_REGS);
   D12_G17P_RANGE(heap_samplers, D12_UNIFORM_HEAP_SAMPLERS,
                  D12_UNIFORM_HEAP_SAMPLER_REGS);
#undef D12_G17P_RANGE
   if (info->rodata.size_16) {
      ranges[range_count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = info->rodata.base_uniform,
         .size_halfs = info->rodata.size_16,
         .address = pso->g17p_rodata_va,
      };
   }
   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(
          info->uses_resource_info_query, false, false, &launch_sources))
      goto fail;
   uint32_t relative_program = 0;
   uint8_t *program_cpu = d12_list_alloc_usc(
      list, AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE, &relative_program);
   const uint64_t program_va = dev->shader_base + relative_program;
   const struct agx3_g17p_compute_program_desc desc = {
      .program_va = program_va,
      .shader_va = pso->code_va,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .shader_binary_size = info->main_size,
      .scratch_size = info->agx3_spectpr_scratch_size,
      .nr_gprs = info->nr_gprs,
      .compiler_d_valid = info->agx3_staging_known,
      .compiler_d = (uint32_t)info->agx3_staging_halfs * 2,
      .threadgroup_memory_size_B = info->local_size
         ? MAX2(256u, ALIGN_POT(info->local_size, 64u)) : 0,
      .launch_sources = launch_sources,
      .ldshdr_f1 = AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
   };
   struct agx3_g17p_compute_program generated;
   if (!program_cpu || !agx_va_in_usc_window(dev, program_va) ||
       !agx_va_in_usc_window(dev, pso->code_va) ||
       !agx3_g17p_compute_program_generate(dev->chip, &desc, &generated))
      goto fail;
   memcpy(program_cpu, generated.bytes, generated.length);
   const size_t offset = program_va - list->usc_bo->va->addr;
   list->usc_offset = offset + generated.length;
   if (agx_bo_sync_for_device(list->usc_bo, offset, generated.length))
      goto fail;
   for (unsigned i = 0; i < range_count; i++) {
      if (d12_list_stream_contains(list, ranges[i].address,
                                    (size_t)ranges[i].size_halfs * 2) &&
          agx_bo_sync_for_device(list->stream_bo,
                                 ranges[i].address - list->stream_bo->va->addr,
                                 (size_t)ranges[i].size_halfs * 2))
         goto fail;
   }

   const struct agx3_compute_direct_grid grid = {
      .source = AGX3_CDM_DIRECT_GRID_THREADGROUP_COUNTS,
      .size = {x, y, z},
      .local_size = {info->workgroup_size[0], info->workgroup_size[1],
                     info->workgroup_size[2]},
      .dimensions = z != 1 || info->workgroup_size[2] != 1 ? 3
         : y != 1 || info->workgroup_size[1] != 1 ? 2 : 1,
   };
   struct agx3_compute_dispatch_sink sink = {
      .data = list->stream_cur,
      .capacity = D12_DISPATCH_STREAM_MARGIN,
   };
   if (!d12_list_check_stream_space(list, sink.capacity) ||
       !agx3_g17p_native_compute_dispatch_append_with_barrier(
          &generated.launch, &grid, 0x5, &sink))
      goto fail;
   list->stream_cur += sink.length;
   return true;

fail:
   mesa_loge("native direct buffer-compute producer rejected metadata/resources");
   list->state = D12_LIST_ERROR;
   return false;
}

static void
d12_list_dispatch_pso(struct d12_cmd_list *list, unsigned x, unsigned y,
                      unsigned z, uint64_t indirect_args,
                      const struct libagx_d3d12_indirect_patch_args *patches,
                      unsigned patch_count,
                      const struct libagx_d3d12_indirect_constant_args
                         *constant_patches,
                      unsigned constant_patch_count,
                      const struct d12_indirect_root_patch *root_patches,
                      unsigned root_patch_count, uint64_t *out_tables,
                      uint32_t *out_table_count)
{
   struct d12_pipeline_state *pso = list->pso;

   if (list->state != D12_LIST_RECORDING)
      return;

   const bool native = list->device->dev.chip == AGX_CHIP_G17P;
   if (!native && !d12_list_require_legacy_compute(list))
      return;

   if (!d12_command_has_compute_pipeline(pso != NULL,
                                         pso ? pso->is_compute : false))
      return;
   if (native) {
      if (indirect_args || list->predicate_gpu || patch_count ||
          constant_patch_count || root_patch_count) {
         mesa_loge("native indirect/predicated compute producer unavailable");
         list->state = D12_LIST_ERROR;
         return;
      }
      if (!x || !y || !z)
         return;
   }

   d12_list_track_pso_reads(list, pso);
   d12_list_track_bound_reads(list, true, D12_WRITE_VIS_COMPUTE);
   if (root_patch_count)
      d12_list_mark_read_set_incomplete(list);

   uint64_t root_gpu = 0, textures_gpu = 0, pbes_gpu = 0;
   uint64_t heap_gpu = 0, heap_textures_gpu = 0, heap_pbes_gpu = 0;
   uint64_t heap_samplers_gpu = 0;
   uint64_t tables_gpu = 0;
   uint32_t table_count = 0;
   if (!d12_list_upload_bindings(list, true, &root_gpu, &textures_gpu,
                                 &pbes_gpu, &heap_gpu, &heap_textures_gpu,
                                 &heap_pbes_gpu, &heap_samplers_gpu,
                                 root_patch_count != 0, &tables_gpu,
                                 &table_count))
      return;

   if (!d12_list_dispatch_indirect_patches(
          list, patches, patch_count, constant_patches, constant_patch_count,
          root_patches, root_patch_count, tables_gpu, table_count, false))
      return;
   if (out_tables)
      *out_tables = tables_gpu;
   if (out_table_count)
      *out_table_count = table_count;

   struct agx_workgroup wg = agx_workgroup(pso->info.workgroup_size[0],
                                           pso->info.workgroup_size[1],
                                           pso->info.workgroup_size[2]);

   /* Predication and ExecuteIndirect both turn Dispatch into a GPU-resident
    * argument record. Keep that record in D3D12's group-count units until
    * predication is resolved, then convert it to the global-thread counts
    * consumed by AGX indirect CDM launches. */
   if (list->predicate_gpu) {
      if (!indirect_args) {
         const D3D12_DISPATCH_ARGUMENTS args = {
            .ThreadGroupCountX = x,
            .ThreadGroupCountY = y,
            .ThreadGroupCountZ = z,
         };
         indirect_args = d12_list_upload(list, &args, sizeof(args), 4);
      }

      indirect_args =
         d12_list_apply_predication(list, indirect_args, 3, false);
      if (!indirect_args)
         return;
   }

   if (indirect_args && (wg.x != 1 || wg.y != 1 || wg.z != 1)) {
      uint64_t scaled_args = d12_list_upload_zero(
         list, sizeof(D3D12_DISPATCH_ARGUMENTS), 4);
      if (!scaled_args)
         return;

      const struct libagx_d3d12_indirect_dispatch_args args = {
         .out = scaled_args,
         .in = indirect_args,
         .workgroup_x = wg.x,
         .workgroup_y = wg.y,
         .workgroup_z = wg.z,
      };
      d12_list_dispatch_precomp_to(
         list, LIBAGX_D3D12_INDIRECT_DISPATCH, &args, sizeof(args),
         agx_1d(1), false);
      if (list->state != D12_LIST_RECORDING)
         return;

      indirect_args = scaled_args;
   }

   /* A patch dispatch can roll the CDM segment at its dispatch limit. Bind
    * terminal-dispatch state only after all argument transforms so it lands
    * on the segment which actually consumes the root table. */
   struct d12_segment *seg = d12_list_get_cdm(list);
   if (!seg)
      return;

   if (!d12_segment_bind_sampler_heap(
          list, seg, pso->info.uses_sampler_heap))
      return;

   if (!d12_segment_reserve_scratch(list, seg, D12_SCRATCH_CS, &pso->info))
      return;

   d12_segment_track_bound_uavs(list, seg, true, D12_WRITE_VIS_COMPUTE,
                                D12_WRITE_COMPUTE);
   /* ExecuteIndirect root UAV values are read by the GPU from the argument
    * buffer immediately before the consumer.  The buffer may itself have
    * been generated by prior GPU work, so a CPU snapshot cannot prove the
    * destination allocation.  Keep all host-derivable UAV ranges as cache
    * hints, but deliberately withhold Aurora's complete-write declaration.
    * Direct-backed allocations remain legal; a live fallback-shadow leaf is
    * rejected by the KMD before firmware visibility, exactly as contract v3
    * requires. */
   if (d12_indirect_root_has_uav(root_patches, root_patch_count))
      d12_segment_mark_write_set_incomplete(seg, D12_WRITE_COMPUTE);

   if (native) {
      if (!d12_list_append_g17p_pso(
             list, pso,
             &(struct d12_dispatch_uniforms){
                .root = root_gpu, .textures = textures_gpu, .pbes = pbes_gpu,
                .heap = heap_gpu, .heap_textures = heap_textures_gpu,
                .heap_pbes = heap_pbes_gpu, .heap_samplers = heap_samplers_gpu,
             }, x, y, z))
         return;
      if (pso->g17p_native_txf) {
         if (d12_g17p_txf_metadata_enabled(&list->device->dev))
            seg->g17p_txf_metadata_only = true;
         else
            seg->g17p_txf_physical = true;
      }
      seg->dispatches++;
      return;
   }

   uint8_t *usc_cpu = NULL;
   uint32_t usc = d12_list_build_compute_usc(
      list, pso->code_va, &pso->info,
      &(struct d12_dispatch_uniforms){.root = root_gpu,
                                      .textures = textures_gpu,
                                      .pbes = pbes_gpu,
                                      .heap = heap_gpu,
                                      .heap_textures = heap_textures_gpu,
                                      .heap_pbes = heap_pbes_gpu,
                                      .heap_samplers = heap_samplers_gpu},
      true, D3D12_SHADER_VISIBILITY_ALL, &usc_cpu);
   if (!usc)
      return;

   if (d12_debug()->dump_usc) {
      fprintf(stderr, "pso USC @%u:", usc);
      for (unsigned i = 0; i < 32; i++)
         fprintf(stderr, " %02x", usc_cpu[i]);
      fprintf(stderr, "\n  root_gpu=0x%llx textures_gpu=0x%llx "
                      "pbes_gpu=0x%llx\n",
              (unsigned long long)root_gpu,
              (unsigned long long)textures_gpu,
              (unsigned long long)pbes_gpu);
   }

   if (d12_debug()->dump_tex) {
      fprintf(stderr,
              "cs info: uses_txf=%d sampler_state_count=%u "
              "texture_state_count=%u push_count=%u nr_preamble_gprs=%u\n",
              pso->info.uses_txf, pso->info.sampler_state_count,
              pso->info.texture_state_count, pso->info.push_count,
              pso->info.nr_preamble_gprs);
      fprintf(stderr, "   rodata: size_16=%u base_uniform=%u offset=%u\n",
              pso->info.rodata.size_16, pso->info.rodata.base_uniform,
              pso->info.rodata.offset);
   }

   struct agx_cdm_launch_word_0_packed launch =
      d12_compute_launch_word(&pso->info);

   /* D3D12 Dispatch counts thread GROUPS; AGX launches want global threads.
    * GPU-resident records were scaled above; direct dispatches can be scaled
    * while encoding the launch without allocating an intermediate record.
    */
   struct agx_grid grid;
   if (indirect_args) {
      grid = agx_grid_indirect(indirect_args);
   } else {
      grid = agx_3d(x * wg.x, y * wg.y, z * wg.z);
   }

   if (!d12_list_check_stream_space(list, D12_CDM_DISPATCH_BYTES))
      return;

   if (!d12_list_emit_legacy_cdm_launch(list, &list->stream_cur, grid, wg,
                                        launch, usc))
      return;
   if (d12_debug()->flush_each_dispatch) {
      if (!d12_list_emit_cdm_cache_flush(list, &list->stream_cur))
         return;
   } else {
      seg->cdm_flush_pending = true;
   }
   seg->dispatches++;
}

/* First bounded graphics-stage producer: monolithic native VS/FS with
 * command-owned root/draw/blend ranges. This supplies Ioto bytes only; the
 * independent VDM-context, PPP, background and EOT gates remain mandatory.
 * Texture, preamble and promoted-constant ABIs are deliberately not admitted. */
static uint32_t
d12_gfx_stage_g17p(struct d12_cmd_list *list, const struct d12_gfx_stage *st,
                  bool fragment, const struct agx_tilebuffer_layout *tib,
                  uint64_t root_gpu, uint64_t draw_gpu, uint64_t blend_gpu)
{
   struct agx_device *dev = &list->device->dev;
   const struct agx_shader_info *info = &st->info;
   const struct agx_linked_shader *linked = &st->linked;
   if (dev->chip != AGX_CHIP_G17P || !linked->bo || !linked->bo->va ||
       !linked->entry_offset || !linked->size || linked->size > UINT32_MAX ||
       linked->bo->va->addr > UINT64_MAX - linked->entry_offset ||
       linked->entry_offset > linked->bo->size ||
       linked->size > linked->bo->size - linked->entry_offset ||
       !linked->nr_gprs || !linked->g17p_compiler_d_valid ||
       info->stage != (fragment ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX) ||
       !st->g17p_root_only || info->scratch_size || linked->scratch_size ||
       info->has_preamble || info->nr_preamble_gprs ||
       info->preamble_scratch_size || info->rodata.size_16 ||
       info->agx3_uniform_dma_valid || info->texture_state_count ||
       info->sampler_state_count || info->uses_sampler_heap || linked->uses_txf ||
       !root_gpu || !tib || tib->nr_samples != 1)
      goto fail;

   struct agx3_ioto_uniform_range ranges[3];
   size_t count = 0;
   if (fragment && blend_gpu)
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = 4, .size_halfs = 8, .address = blend_gpu,
      };
   ranges[count++] = (struct agx3_ioto_uniform_range){
      .start_halfs = fragment ? D12_UNIFORM_FRAGMENT_ROOT : D12_UNIFORM_ROOT,
      .size_halfs = D12_UNIFORM_ROOT_REGS, .address = root_gpu,
   };
   if (draw_gpu)
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = D12_UNIFORM_DRAW,
         .size_halfs = D12_UNIFORM_DRAW_REGS, .address = draw_gpu,
      };
   for (size_t i = 0; i < count; i++) {
      const size_t size = (size_t)ranges[i].size_halfs * 2;
      /* This subset consumes only immutable command-list uploads. */
      if (!d12_list_stream_contains(list, ranges[i].address, size) ||
          agx_bo_sync_for_device(list->stream_bo,
             ranges[i].address - list->stream_bo->va->addr, size))
         goto fail;
   }

   const uint64_t shader_va = linked->bo->va->addr + linked->entry_offset;
   uint32_t relative_program = 0;
   uint8_t *cpu = d12_list_alloc_usc(list, AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE,
                                   &relative_program);
   const uint64_t program_va = dev->shader_base + relative_program;
   const struct agx3_g17p_graphics_program_desc desc = {
      .program_va = program_va,
      .shader_va = shader_va,
      .uniform_ranges = ranges,
      .uniform_range_count = count,
      .shader_binary_size = linked->size,
      .scratch_size = MAX2(linked->scratch_size,
                           linked->agx3_spectpr_scratch_size),
      .nr_gprs = linked->nr_gprs,
      .compiler_d_valid = linked->g17p_compiler_d_valid,
      .compiler_d = linked->g17p_compiler_d,
      .fragment = fragment,
      .writes_sample_mask = fragment &&
         ((linked->fragment_control.opaque[0] >> 29) == AGX_PASS_TYPE_PUNCH_THROUGH ||
          (linked->fragment_control.opaque[0] >> 29) ==
             AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH),
      .colour_classes = fragment ? linked->g17p_colour_classes : 0,
      .imageblock = {
         .sample_size_B = tib->sample_size_B,
         .nr_samples = tib->nr_samples,
         .tile_size_px = tib->tile_size,
      },
   };
   struct agx3_g17p_graphics_program generated;
   if (!cpu || !agx_va_in_usc_window(dev, program_va) ||
       !agx_va_in_usc_window(dev, shader_va) ||
       linked->size > AGX_USC_WINDOW_SIZE - (shader_va - dev->shader_base) ||
       !agx3_g17p_graphics_program_generate(dev->chip, &desc, &generated))
      goto fail;
   const size_t offset = program_va - list->usc_bo->va->addr;
   memcpy(cpu, generated.bytes, generated.length);
   list->usc_offset = offset + generated.length;
   if (agx_bo_sync_for_device(list->usc_bo, offset, generated.length))
      goto fail;
   d12_list_track_read_bo(list, linked->bo);
   return relative_program;

fail:
   mesa_loge("native graphics-stage Ioto producer rejected metadata/resources");
   list->state = D12_LIST_ERROR;
   return 0;
}

/*
 * Build the USC words that bind one linked graphics stage: our root
 * uniform, the stage's shared config, then the linker's packed program
 * words. Mirrors hk_upload_usc_words for the no-descriptor case.
 */
static uint32_t
d12_gfx_stage_usc(struct d12_cmd_list *list, const struct d12_gfx_stage *st,
                  bool fragment, const struct agx_tilebuffer_layout *tib,
                  uint64_t root_gpu, uint64_t textures_gpu, uint64_t pbes_gpu,
                  uint64_t heap_gpu, uint64_t heap_textures_gpu,
                  uint64_t heap_pbes_gpu, uint64_t heap_samplers_gpu,
                  uint64_t draw_gpu, uint64_t blend_gpu,
                  D3D12_SHADER_VISIBILITY visibility,
                  const struct d12_dispatch_uniforms *poly)
{
   struct d12_device *device = list->device;

   if (device->dev.chip == AGX_CHIP_G17P) {
      /* Unused null texture/PBE home cells are always uploaded by the
       * binding collector; the frontend usage proof decides admission. */
      if (!st->g17p_root_only || heap_gpu || heap_textures_gpu ||
          heap_pbes_gpu || heap_samplers_gpu || poly) {
         mesa_loge("native graphics texture/poly stage producer unavailable");
         list->state = D12_LIST_ERROR;
         return 0;
      }
      return d12_gfx_stage_g17p(list, st, fragment, tib,
                                 root_gpu, draw_gpu, blend_gpu);
   }

   uint32_t usc;
   uint8_t *cpu = d12_list_alloc_usc(list, agx_usc_size(18), &usc);
   if (!cpu)
      return 0;

   struct agx_usc_builder b = agx_usc_builder(cpu, agx_usc_size(18));

   /* USC uniform records must be emitted in ascending register order.
    * Fragment registers 4..11 carry D3D12's four blend constants, followed
    * by the fragment root at 12 and the texture-heap base at 16. Keeping the
    * texture base at the old slot 8 would alias the upper half of the blend
    * constants whenever a shader sampled and blended in the same draw. */
   if (fragment) {
      agx_usc_uniform(&b, 4, 8, blend_gpu);
      agx_usc_uniform(&b, D12_UNIFORM_FRAGMENT_ROOT, D12_UNIFORM_ROOT_REGS,
                      root_gpu);
   } else {
      agx_usc_uniform(&b, D12_UNIFORM_ROOT, D12_UNIFORM_ROOT_REGS, root_gpu);
   }
   if (textures_gpu)
      agx_usc_uniform(&b, D12_UNIFORM_TEXTURES, D12_UNIFORM_TEXTURE_REGS,
                      textures_gpu);

   if (draw_gpu) {
      agx_usc_uniform(&b, D12_UNIFORM_DRAW, D12_UNIFORM_DRAW_REGS,
                      draw_gpu);
   }

   /*
    * A geometry shader's RASTERIZATION program is a hardware vertex
    * shader that re-runs the geometry body, so it reads the same
    * parameter blocks the compute half does -- and the emitted vertices
    * through them. Publishing them only for the compute dispatches
    * leaves this program reading whatever was in those registers, which
    * measured as a point covering the ENTIRE render target while the
    * same points without a geometry shader drew four pixels.
    */
   /* Ascending base order, matching the compute builder: 24, 28, 32, 36. */
   if (poly) {
      if (poly->geom_params) {
         agx_usc_uniform(&b, D12_UNIFORM_GEOM_PARAMS, D12_UNIFORM_PARAM_REGS,
                         poly->geom_params);
      }
      if (poly->tess_params) {
         agx_usc_uniform(&b, D12_UNIFORM_TESS_PARAMS, D12_UNIFORM_PARAM_REGS,
                         poly->tess_params);
      }
      if (poly->vs_outputs) {
         agx_usc_uniform(&b, D12_UNIFORM_VS_OUTPUTS, D12_UNIFORM_PARAM_REGS,
                         poly->vs_outputs);
      }
      if (poly->vertex_params) {
         agx_usc_uniform(&b, D12_UNIFORM_VERTEX_PARAMS,
                         D12_UNIFORM_PARAM_REGS, poly->vertex_params);
      }
      if (poly->mesh_params) {
         agx_usc_uniform(&b, D12_UNIFORM_MESH_PARAMS, D12_UNIFORM_MESH_REGS,
                         poly->mesh_params);
      }
      if (poly->vb_params) {
         agx_usc_uniform(&b, D12_UNIFORM_VB_PARAMS, D12_UNIFORM_VB_REGS,
                         poly->vb_params);
      }
   }

   if (heap_gpu) {
      agx_usc_uniform(&b, D12_UNIFORM_HEAP, D12_UNIFORM_HEAP_REGS,
                      heap_gpu);
   }
   if (pbes_gpu) {
      agx_usc_uniform(&b, D12_UNIFORM_PBES, D12_UNIFORM_PBE_REGS,
                      pbes_gpu);
   }
   if (heap_textures_gpu) {
      agx_usc_uniform(&b, D12_UNIFORM_HEAP_TEXTURES,
                      D12_UNIFORM_HEAP_TEXTURE_REGS, heap_textures_gpu);
   }
   if (heap_pbes_gpu) {
      agx_usc_uniform(&b, D12_UNIFORM_HEAP_PBES,
                      D12_UNIFORM_HEAP_PBE_REGS, heap_pbes_gpu);
   }
   if (heap_samplers_gpu) {
      agx_usc_uniform(&b, D12_UNIFORM_HEAP_SAMPLERS,
                      D12_UNIFORM_HEAP_SAMPLER_REGS, heap_samplers_gpu);
   }

   uint64_t custom_borders;
   if (!d12_list_upload_custom_borders(list, &st->info, false, visibility,
                                       &custom_borders))
      return 0;
   if (custom_borders) {
      agx_usc_uniform(&b, D12_UNIFORM_CUSTOM_BORDERS,
                      D12_UNIFORM_CUSTOM_BORDER_REGS, custom_borders);
   }

   /* Promoted constants start above D12_UNIFORM_RESERVED, so bind them only
    * after every driver-owned uniform to preserve ascending register order. */
   if (st->info.rodata.size_16) {
      agx_usc_immediates(&b, &st->info.rodata, st->linked.bo->va->addr);
   }

   if (!d12_list_push_samplers(list, &b, &st->info, false, visibility))
      return 0;

   if (fragment) {
      agx_usc_push_packed(&b, SHARED, &tib->usc);
   } else {
      agx_usc_shared_none(&b);
   }

   agx_usc_push_packed(&b, SHADER, &st->linked.shader);
   agx_usc_push_packed(&b, REGISTERS, &st->linked.regs);

   if (fragment) {
      agx_usc_push_packed(&b, FRAGMENT_PROPERTIES,
                          &st->linked.fragment_props);
   }

   if (st->info.has_preamble) {
      bool preamble_valid = st->code_bo &&
         st->preamble_va >= st->code_bo->va->addr &&
         st->preamble_va - st->code_bo->va->addr < st->code_bo->size;

      if (!preamble_valid) {
         mesa_loge("binding validation: %s preamble 0x%llx is not owned "
                   "by its compiled-part BO",
                   fragment ? "fragment" : "vertex",
                   (unsigned long long)st->preamble_va);
         list->state = D12_LIST_ERROR;
         return 0;
      }

      if (d12_debug()->dump_tex)
         mesa_logi("%s preamble: va=0x%llx bo=0x%llx..0x%llx valid=1 "
                   "tex_states=%u sampler_states=%u pregprs=%u",
                   fragment ? "fragment" : "vertex",
                   (unsigned long long)st->preamble_va,
                   (unsigned long long)st->code_bo->va->addr,
                   (unsigned long long)(st->code_bo->va->addr +
                                        st->code_bo->size),
                   st->info.texture_state_count,
                   st->info.sampler_state_count,
                   st->info.nr_preamble_gprs);

      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code = agx_usc_addr(&device->dev, st->preamble_va);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   size_t usc_used = b.head - cpu;
   if (d12_debug()->validate_bindings &&
       usc_used > agx_usc_size(18)) {
      mesa_loge("binding validation: graphics USC used %zu bytes of %zu",
                usc_used, (size_t)agx_usc_size(18));
      list->state = D12_LIST_ERROR;
      return 0;
   }

   return usc;
}

/*
 * Add one contiguous hardware scissor sequence for this draw and return its
 * base index.
 *
 * These are not only the XY bounds. Each entry also carries the depth CLIP
 * range, and the hardware consults it for every fragment: with no array
 * uploaded, ISP_SCISSOR_BASE reads as zero, the clip range is [0, 0], and
 * every fragment with a nonzero window Z is discarded. That failure looks
 * exactly like a broken depth test, which is how it was found.
 *
 * A sequence is deduplicated against the previous sequence, so repeated draws
 * with unchanged viewport and scissor arrays do not consume more storage.
 */
static bool
d12_segment_ensure_viewport_arrays(struct d12_cmd_list *list,
                                   struct d12_segment *seg)
{
   struct d12_viewport_storage_plan plan;

   static_assert(AGX_SCISSOR_LENGTH ==
                 D12_AGX3_PPP_HAL300_SCISSOR_RECORD_LENGTH);
   static_assert(AGX_DEPTH_BIAS_LENGTH ==
                 D12_AGX3_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH);

   if (seg->scissor_base && seg->depth_bias_base)
      return true;
   if (seg->scissor_base || seg->depth_bias_base) {
      list->state = D12_LIST_ERROR;
      return false;
   }
   if (!d12_viewport_plan_storage(
          true, D12_SCISSORS_PER_SEGMENT, AGX_SCISSOR_LENGTH,
          D12_DEPTH_BIASES_PER_SEGMENT, AGX_DEPTH_BIAS_LENGTH, &plan) ||
       plan.scissor_bytes > SIZE_MAX || plan.depth_bias_bytes > SIZE_MAX) {
      list->state = D12_LIST_ERROR;
      return false;
   }

   const uint64_t scissor_base = d12_list_upload_zero(
      list, (size_t)plan.scissor_bytes, 64);
   const uint64_t depth_bias_base = d12_list_upload_zero(
      list, (size_t)plan.depth_bias_bytes, 64);
   if (!scissor_base || !depth_bias_base)
      return false;

   seg->scissor_base = scissor_base;
   seg->depth_bias_base = depth_bias_base;
   return true;
}

static unsigned
d12_list_push_scissors(
   struct d12_cmd_list *list, struct d12_segment *seg, unsigned count,
   const struct d12_viewport_hw_state states[D12_VIEWPORT_STATE_MAX])
{
   uint8_t packed[D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE]
                 [AGX_SCISSOR_LENGTH];

   if (!count || count > ARRAY_SIZE(packed)) {
      list->state = D12_LIST_ERROR;
      return ~0u;
   }
   if (!d12_segment_ensure_viewport_arrays(list, seg))
      return ~0u;

   const enum agx_chip chip = list->device->dev.chip;
   const bool hal300 = agx_chip_command_stream_generation(chip) ==
                       AGX_CS_GENERATION_HAL300;
   for (unsigned i = 0; i < count; ++i) {
      if (hal300) {
         const struct d12_agx3_ppp_hal300_scissor_record_fields fields = {
            .max_x = states[i].clip[2],
            .min_x = states[i].clip[0],
            .max_y = states[i].clip[3],
            .min_y = states[i].clip[1],
            /* D3D12 clips against 0 <= z <= w in clip space. MinDepth and
             * MaxDepth define the viewport mapping, not the clip range. */
            .min_z_raw = fui(0.0f),
            .max_z_raw = fui(1.0f),
         };
         struct d12_agx3_ppp_hal300_scissor_record exact;
         if (!d12_agx3_encode_ppp_hal300_scissor_record(chip, &fields,
                                                         &exact)) {
            mesa_loge("no grounded HAL300 scissor record for %s",
                      agx_chip_name(chip));
            list->state = D12_LIST_ERROR;
            return ~0u;
         }
         memcpy(packed[i], exact.bytes, exact.length);
      } else {
         agx_pack(packed[i], SCISSOR, cfg) {
            cfg.min_x = states[i].clip[0];
            cfg.min_y = states[i].clip[1];
            cfg.max_x = states[i].clip[2];
            cfg.max_y = states[i].clip[3];
            cfg.min_z = 0.0f;
            cfg.max_z = 1.0f;
         }
      }
   }

   uint8_t *array = d12_list_data_cpu(list, seg->scissor_base);
   if (!array) {
      mesa_loge("draw into a render pass with no scissor array");
      list->state = D12_LIST_ERROR;
      return ~0u;
   }

   const size_t bytes = count * AGX_SCISSOR_LENGTH;
   if (seg->scissor_count >= count) {
      unsigned previous = seg->scissor_count - count;
      uint8_t *prev = array + previous * AGX_SCISSOR_LENGTH;
      if (!memcmp(prev, packed, bytes))
         return previous;
   }

   if (count > D12_SCISSORS_PER_SEGMENT - seg->scissor_count) {
      mesa_loge("render pass exceeded %u distinct scissor/viewport states",
                D12_SCISSORS_PER_SEGMENT);
      list->state = D12_LIST_ERROR;
      return ~0u;
   }

   unsigned index = seg->scissor_count;
   seg->scissor_count += count;
   memcpy(array + index * AGX_SCISSOR_LENGTH, packed, bytes);
   return index;
}

/* Pack one D3D12 rasterizer depth-bias state into the array named by the
 * render command.  AGX's integer Z16 mode uses the same scale measured by
 * Honeykrisp. D24 is represented internally as Z32_FLOAT, so one D24 LSB is
 * 32 float ULPs at the test's exponent and the constant factor is expanded
 * accordingly. */
static unsigned
d12_list_push_depth_bias(struct d12_cmd_list *list, struct d12_segment *seg,
                         const struct d12_pipeline_state *pso)
{
   float constant = pso->raster.dynamic_depth_bias
                       ? list->depth_bias
                       : pso->raster.depth_bias;
   float clamp = pso->raster.dynamic_depth_bias
                    ? list->depth_bias_clamp
                    : pso->raster.depth_bias_clamp;
   float slope = pso->raster.dynamic_depth_bias
                    ? list->slope_scaled_depth_bias
                    : pso->raster.slope_scaled_depth_bias;
   DXGI_FORMAT format = pso->ds.dsv_format;
   if (format == DXGI_FORMAT_UNKNOWN && seg->ds)
      format = seg->ds->desc.Format;

   if (format == DXGI_FORMAT_D16_UNORM)
      constant /= (1u << 15);
   else if (format == DXGI_FORMAT_D24_UNORM_S8_UINT)
      constant *= 32.0f;

   uint8_t packed[AGX_DEPTH_BIAS_LENGTH];
   const enum agx_chip chip = list->device->dev.chip;
   if (agx_chip_command_stream_generation(chip) ==
       AGX_CS_GENERATION_HAL300) {
      const struct d12_agx3_ppp_hal300_depth_bias_record_fields fields = {
         .depth_bias_raw = fui(constant),
         .slope_scale_raw = fui(slope),
         .clamp_raw = fui(clamp),
      };
      struct d12_agx3_ppp_hal300_depth_bias_record exact;
      if (!d12_agx3_encode_ppp_hal300_depth_bias_record(chip, &fields,
                                                        &exact)) {
         mesa_loge("no grounded HAL300 depth-bias record for %s",
                   agx_chip_name(chip));
         list->state = D12_LIST_ERROR;
         return ~0u;
      }
      memcpy(packed, exact.bytes, exact.length);
   } else {
      agx_pack(packed, DEPTH_BIAS, cfg) {
         cfg.depth_bias = constant;
         cfg.slope_scale = slope;
         cfg.clamp = clamp;
      }
   }

   uint8_t *array = d12_list_data_cpu(list, seg->depth_bias_base);
   if (!array) {
      mesa_loge("draw into a render pass with no depth-bias array");
      list->state = D12_LIST_ERROR;
      return ~0u;
   }

   if (seg->depth_bias_count) {
      uint8_t *prev = array +
                      (seg->depth_bias_count - 1) * AGX_DEPTH_BIAS_LENGTH;
      if (!memcmp(prev, packed, AGX_DEPTH_BIAS_LENGTH))
         return seg->depth_bias_count - 1;
   }

   if (seg->depth_bias_count >= D12_DEPTH_BIASES_PER_SEGMENT) {
      mesa_loge("render pass exceeded %u distinct depth-bias states",
                D12_DEPTH_BIASES_PER_SEGMENT);
      list->state = D12_LIST_ERROR;
      return ~0u;
   }

   unsigned index = seg->depth_bias_count++;
   memcpy(array + index * AGX_DEPTH_BIAS_LENGTH, packed,
          AGX_DEPTH_BIAS_LENGTH);
   return index;
}

static bool
d12_emit_hal300_viewport_ppp(
   struct d12_cmd_list *list, const struct d12_segment *seg,
   unsigned viewport_count,
   const struct d12_viewport_hw_state states[D12_VIEWPORT_STATE_MAX],
   unsigned scissor_index, unsigned depth_bias_index, uint8_t **out)
{
   if (!out || !*out || viewport_count == 0 ||
       viewport_count > D12_AGX3_PPP_VIEWPORT_MAX_VIEWPORTS ||
       scissor_index > D12_SCISSORS_PER_SEGMENT - viewport_count ||
       depth_bias_index >= D12_DEPTH_BIASES_PER_SEGMENT)
      return false;

   const enum agx_chip chip = list->device->dev.chip;
   struct d12_agx3_ppp_hal300_depth_bias_scissor_bases bases;
   const struct d12_agx3_ppp_hal300_depth_bias_scissor_bases_fields
      base_fields = {
         .depth_bias_base =
            seg->depth_bias_base +
            (uint64_t)depth_bias_index * AGX_DEPTH_BIAS_LENGTH,
         .scissor_base =
            seg->scissor_base + (uint64_t)scissor_index * AGX_SCISSOR_LENGTH,
      };
   if (!d12_agx3_encode_ppp_hal300_depth_bias_scissor_bases(
          chip, &base_fields, &bases))
      return false;

   struct d12_agx3_ppp_viewport_fields viewport_fields = {
      .viewport_count = viewport_count,
   };
   for (unsigned i = 0; i < viewport_count; ++i) {
      uint8_t region_clip[AGX_REGION_CLIP_LENGTH];
      agx_pack(region_clip, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = states[i].clip[0] / 32;
         cfg.min_y = states[i].clip[1] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(states[i].clip[2], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(states[i].clip[3], 1), 32);
      }
      memcpy(&viewport_fields.region_clip_raw[i], region_clip,
             sizeof(region_clip));

      viewport_fields.viewports[i] =
         (struct d12_agx3_ppp_viewport_transform_raw){
            .translate_x = fui(states[i].transform[0]),
            .scale_x = fui(states[i].transform[3]),
            .translate_y = fui(states[i].transform[1]),
            .scale_y = fui(states[i].transform[4]),
            .translate_z = fui(states[i].transform[2]),
            .scale_z = fui(states[i].transform[5]),
         };
   }

   uint8_t viewport_control[AGX_VIEWPORT_CONTROL_LENGTH];
   agx_pack(viewport_control, VIEWPORT_CONTROL, cfg)
      ;
   memcpy(&viewport_fields.viewport_control_raw, viewport_control,
          sizeof(viewport_control));

   struct d12_agx3_ppp_viewport_payload viewport;
   if (!d12_agx3_encode_ppp_viewport_payload(chip, &viewport_fields,
                                             &viewport))
      return false;

   const uint64_t bases_gpu = d12_list_upload(
      list, bases.bytes, bases.length, AGX_PPP_HEADER_ALIGN);
   const uint64_t viewport_gpu = d12_list_upload(
      list, viewport.bytes, viewport.length, AGX_PPP_HEADER_ALIGN);
   if (!bases_gpu || !viewport_gpu)
      return false;

   uint8_t records[2 * AGX_PPP_STATE_LENGTH];
   uint8_t *record = records;
   /* Match HAL300's initial render-state order: viewport, then base update. */
   if (!agx_ppp_emit_state(chip, &record, viewport_gpu, viewport.length) ||
       !agx_ppp_emit_state(chip, &record, bases_gpu, bases.length) ||
       record != records + sizeof(records))
      return false;

   memcpy(*out, records, sizeof(records));
   *out += sizeof(records);
   return true;
}

/*
 * Push one face's stencil state. When the test is off the slot still has to
 * be written: the hardware keeps this state across the control stream, and
 * an all-zero FRAGMENT_STENCIL means compare NEVER, which discards every
 * fragment rather than doing nothing.
 */
static void
d12_ppp_push_stencil(struct agx_ppp_update *ppp,
                     const struct d12_stencil_face *face,
                     const struct d12_depth_stencil_state *ds, bool enabled,
                     bool read_only, bool back_face)
{
   agx_ppp_push(ppp, FRAGMENT_STENCIL, cfg) {
      if (enabled) {
         cfg.compare = face->compare;
         cfg.read_mask = back_face ? ds->back_stencil_read_mask
                                   : ds->front_stencil_read_mask;
         cfg.write_mask = read_only
                            ? 0
                            : back_face ? ds->back_stencil_write_mask
                                        : ds->front_stencil_write_mask;
         cfg.depth_pass = face->pass;
         cfg.depth_fail = face->depth_fail;
         cfg.stencil_fail = face->stencil_fail;
      } else {
         cfg.compare = AGX_ZS_FUNC_ALWAYS;
         cfg.read_mask = 0xFF;
         cfg.write_mask = 0xFF;
         cfg.depth_pass = AGX_STENCIL_OP_KEEP;
         cfg.depth_fail = AGX_STENCIL_OP_KEEP;
         cfg.stencil_fail = AGX_STENCIL_OP_KEEP;
      }
   }
}

/*
 * Emit one draw into the open render segment's VDM stream: vertex-shader
 * state, the per-draw PPP block, viewport/scissor, then the draw itself.
 */
/*
 * The rasterizer's object type. It is NOT cosmetic and NOT implied by the
 * VDM primitive: it tells the rasterizer what shape to fill. It was
 * hardcoded to TRIANGLE, so a point list rasterised as triangles --
 * measured as zero pixels without a geometry shader and the ENTIRE render
 * target with one. hk derives it the same way (hk_cmd_draw.c:2158).
 */
static enum agx_object_type
d12_object_type(enum mesa_prim prim)
{
   if (prim == MESA_PRIM_POINTS)
      return AGX_OBJECT_TYPE_POINT_SPRITE_UV01;
   else if (prim <= MESA_PRIM_LINE_STRIP)
      return AGX_OBJECT_TYPE_LINE;
   else
      return AGX_OBJECT_TYPE_TRIANGLE;
}

/* The hardware primitive for a mesa_prim. */
static enum agx_primitive
d12_agx_primitive(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return AGX_PRIMITIVE_POINTS;
   case MESA_PRIM_LINES:
      return AGX_PRIMITIVE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return AGX_PRIMITIVE_LINE_STRIP;
   case MESA_PRIM_TRIANGLE_STRIP:
      return AGX_PRIMITIVE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return AGX_PRIMITIVE_TRIANGLE_FAN;
   default:
      return AGX_PRIMITIVE_TRIANGLES;
   }
}

static bool
d12_list_dispatch_gs_stage(struct d12_cmd_list *list,
                           struct d12_segment *seg, uint64_t va,
                           const struct agx_shader_info *info,
                           struct agx_grid grid, struct agx_workgroup wg,
                           const struct d12_dispatch_uniforms *u,
                           D3D12_SHADER_VISIBILITY visibility)
{
   if (!d12_list_require_legacy_compute(list))
      return false;

   if (!va || !info->binary_size ||
       !d12_segment_reserve_scratch(list, seg, D12_SCRATCH_CS, info))
      return false;

   uint32_t usc = d12_list_build_compute_usc(
      list, va, info, u, false, visibility, NULL);
   if (!usc)
      return false;

   if (!d12_list_check_pregfx_space(list, D12_CDM_DISPATCH_BYTES))
      return false;

   struct agx_cdm_launch_word_0_packed launch =
      d12_compute_launch_word(info);
   if (!d12_list_emit_legacy_cdm_launch(list, &list->pregfx_cur, grid, wg,
                                        launch, usc))
      return false;
   if (!d12_list_emit_cdm_cache_flush(list, &list->pregfx_cur))
      return false;
   seg->dispatches++;
   return true;
}

/*
 * Run a geometry shader's compute half and report what the rasterization
 * draw becomes.
 *
 * For transform feedback, this includes the count, optional prefix-sum and
 * pre-GS programs before the ordinary geometry body. When XFB is absent,
 * the same path remains the two-dispatch software-VS/GS sequence.
 *
 * The dispatches go in the PRE-GRAPHICS stream so they land before the
 * open render pass rather than ending it.
 */
static bool
d12_list_gs_prerast(struct d12_cmd_list *list, struct d12_pipeline_state *pso,
                    unsigned vertex_count, unsigned instance_count,
                    bool indexed, unsigned first_index,
                    const struct agx_draw *upstream_draw,
                    uint64_t indirect_ibv,
                    bool upstream_restart_unrolled,
                    uint64_t root_gpu, uint64_t textures_gpu,
                    uint64_t pbes_gpu,
                    uint64_t draw_gpu, unsigned *out_verts,
                    unsigned *out_instances, struct agx_draw *out_draw,
                    struct d12_dispatch_uniforms *out_poly)
{
   /* 64 threads, and this value has to agree in three places: both
    * parameter blocks bake it, and the launch below uses it. Nothing
    * cross-checks them.
    */
   const uint32_t wg_size[3] = {64, 1, 1};
   struct agx_workgroup wg = agx_workgroup(wg_size[0], wg_size[1], wg_size[2]);

   /* The software VS writes one vec4 per live output per vertex; the GS
    * reads it back. Zero outputs would mean a zero-size allocation, so
    * keep at least one word rather than handing the shader a null.
    */
   const bool indirect =
      upstream_draw && agx_is_indirect(upstream_draw->b);
   const bool dynamic_ibv =
      indirect && indexed && indirect_ibv && !upstream_restart_unrolled;
   const uint64_t sw_outputs =
      pso->has_tess ? pso->sw_tes_outputs : pso->sw_vs_outputs;
   const enum mesa_prim input_prim =
      pso->has_tess
         ? (pso->tess_points
               ? MESA_PRIM_POINTS
               : pso->tess_mode == TESS_PRIMITIVE_ISOLINES
                    ? MESA_PRIM_LINES
                    : MESA_PRIM_TRIANGLES)
         : upstream_restart_unrolled ? u_decomposed_prim(list->topology)
                                      : list->topology;

   struct poly_vertex_params vp;
   poly_vertex_params_init(&vp, sw_outputs, wg_size);
   if (!indirect)
      poly_vertex_params_set_draw(&vp, vertex_count, instance_count);

   /* poly_gs_setup_indirect writes the indexed buffer/range after the
    * restart/tessellation setup kernel has run, but the index width is static
    * draw state and is consumed later by the software VS.  Poly's generic
    * helper intentionally leaves that field to the driver. */
   if (indirect && upstream_draw->indexed && !dynamic_ibv)
      vp.index_size_B = agx_index_size_to_B(upstream_draw->index_size);

   if (!indirect && indexed) {
      enum agx_index_size index_size = list->ib.addr
                                          ? list->ib.index_size
                                          : AGX_INDEX_SIZE_U16;
      unsigned index_size_B = agx_index_size_to_B(index_size);
      unsigned range_el = list->ib.size_B / index_size_B;

      vp.index_size_B = index_size_B;
      if (list->ib.addr && first_index < range_el) {
         vp.index_buffer = list->ib.addr +
                           (uint64_t)first_index * index_size_B;
         vp.index_buffer_range_el = range_el - first_index;
      } else {
         /* poly_load_index still performs one physical load before it
          * substitutes zero for an OOB element, so an empty/null view needs
          * mapped zero storage rather than VA 0.
          */
         vp.index_buffer = list->device->null_va;
         vp.index_buffer_range_el = 0;
      }
   }

   size_t out_size = 0;
   if (!indirect) {
      out_size = poly_tcs_in_size((uint64_t)vertex_count * instance_count,
                                  sw_outputs);
      vp.output_buffer = d12_list_upload_zero(list, MAX2(out_size, 16), 16);
      if (!vp.output_buffer)
         return false;
   }

   /* Remember it so the vertex buffer can be read back once the GPU is
    * idle. Whether the software vertex shader actually wrote what the
    * rasterization program then reads is the one question a command
    * stream dump cannot answer.
    */
   if (!indirect && d12_debug()->dump_gs_vb) {
      list->device->dbg_gs_vb = d12_list_data_cpu(list, vp.output_buffer);
      list->device->dbg_gs_vb_size = MAX2(out_size, 16);
      mesa_logi("gs vb: gpu=0x%llx cpu=%p size=%zu outputs=0x%llx "
                "verts=%u insts=%u",
                (unsigned long long)vp.output_buffer,
                list->device->dbg_gs_vb, list->device->dbg_gs_vb_size,
                (unsigned long long)sw_outputs, vertex_count,
                instance_count);
   }

   /* poly's software input assembly always emits the index-buffer pull,
    * predicated on index_size != 0. We draw non-indexed, so the branch is
    * never taken -- but the pointer is still materialised, and hk never
    * leaves it null: it points it at a sink address even when unused
    * (hk_index_buffer). Leaving it zero puts a null dereference in the
    * shader, and a fault aborts the whole command buffer, which presents
    * as the compute half silently doing nothing.
    */
   if (!vp.index_buffer)
      vp.index_buffer = list->device->null_va;

   struct poly_geometry_params gp;
   poly_geometry_params_init(&gp, input_prim, wg_size);
   gp.flat_outputs = pso->flat_outputs;
   if (d12_debug()->validate_bindings && pso->flat_outputs) {
      mesa_logi("GS flat transport: outputs=0x%llx topology=%u shape=%u",
                (unsigned long long)gp.flat_outputs, input_prim,
                pso->gs_info.shape);
   }
   if (!indirect) {
      poly_geometry_params_set_draw(&gp, input_prim, pso->gs_info.shape,
                                    pso->gs_info.max_indices, vertex_count,
                                    instance_count);
   }

   /* poly's pre-GS program always dereferences the query/counter pointers,
    * even when D3D12 has no corresponding query active. Give every unused
    * field its own writable word in the command-list BO rather than a null
    * address. This is also the sink for an unbound SO slot's offset.
    */
   uint64_t sink = d12_list_upload_zero(list, 128, 8);
   if (!sink)
      return false;

   /* create_pre_gs() always asks for the GS/clipping pipeline-statistics
    * addresses.  D3D12AGX accounts its D3D12 pipeline query on the CPU, so
    * these Poly-side atomics deliberately target scratch.  The USC binding
    * must contain the address of that scratch address, not the scratch address
    * itself: uniform records copy bytes from memory into preamble registers.
    * Leaving registers 48..51 unpublished made the pre-GS shader interpret
    * stale register contents as a GPU VA, producing the varying unmapped UL1C
    * reads seen in the quarantined stream-output runs.
    */
   uint64_t stat_query = d12_list_upload(list, &sink, sizeof(sink), 8);
   if (!stat_query)
      return false;

   if (d12_debug()->validate_bindings) {
      bool valid = d12_list_stream_contains(list, sink, 128) &&
                   d12_list_stream_contains(list, stat_query, sizeof(sink)) &&
                   *(uint64_t *)d12_list_data_cpu(list, stat_query) == sink;
      mesa_logi("GS stat sink: home=0x%llx target=0x%llx valid=%u",
                (unsigned long long)stat_query,
                (unsigned long long)sink, valid);
      if (!valid) {
         list->state = D12_LIST_ERROR;
         return false;
      }
   }

   unsigned sink_word = 0;
   for (unsigned i = 0; i < POLY_MAX_VERTEX_STREAMS; i++) {
      gp.prims_generated_counter[i] = sink + 4 * sink_word++;
      gp.xfb_prims_generated_counter[i] = sink + 4 * sink_word++;
      gp.xfb_overflow[i] = sink + 4 * sink_word++;

      if (i < ARRAY_SIZE(list->active_so_stats_heap) &&
          list->active_so_stats_heap[i]) {
         struct d12_query_heap *heap = list->active_so_stats_heap[i];
         uint64_t result = heap->bo->va->addr +
                           (uint64_t)list->active_so_stats_index[i] *
                              heap->stride;

         /* D3D12_QUERY_DATA_SO_STATISTICS is two uint64_t fields:
          * {written, storage-needed}. Poly increments uint32_t counters;
          * BeginQuery has already zeroed both complete fields.
          */
         gp.xfb_prims_generated_counter[i] = result;
         gp.prims_generated_counter[i] = result + sizeof(uint64_t);
      }
   }
   gp.xfb_any_overflow = sink + 4 * sink_word++;

   for (unsigned i = 0; i < POLY_MAX_SO_BUFFERS; i++) {
      const D3D12_STREAM_OUTPUT_BUFFER_VIEW *view = &list->so[i];
      bool bound = pso->gs_info.xfb && view->BufferLocation &&
                   view->SizeInBytes && view->BufferFilledSizeLocation;

      if (bound) {
         gp.xfb_base_original[i] = view->BufferLocation;
         gp.xfb_size[i] = view->SizeInBytes;
         gp.xfb_offs_ptrs[i] = view->BufferFilledSizeLocation;
      } else {
         gp.xfb_base_original[i] = sink + 4 * sink_word++;
         gp.xfb_size[i] = 0;
         gp.xfb_offs_ptrs[i] = sink + 4 * sink_word++;
      }
   }

   gp.count_buffer_stride = pso->gs_info.count_words * 4;
   if (!indirect && gp.count_buffer_stride) {
      /* With prefix sums, each input primitive owns one count record. The
       * fixed-count form only needs poly's documented 16-byte scratch.
       */
      size_t count_size = pso->gs_info.prefix_sum
                             ? (size_t)gp.input_primitives *
                                  gp.count_buffer_stride
                             : 16;
      gp.count_buffer =
         d12_list_upload_zero(list, MAX2(count_size, 16), 4);
      if (!gp.count_buffer)
         return false;
   }

   uint64_t static_index_buffer = 0;
   size_t static_index_size = 0;
   if (pso->gs_info.shape == POLY_GS_SHAPE_DYNAMIC_INDEXED && !indirect) {
      size_t index_size = (size_t)gp.draw.index_count * sizeof(uint32_t);
      if (!index_size || index_size > UINT32_MAX) {
         mesa_loge("geometry output index buffer size %zu is invalid",
                   index_size);
         list->state = D12_LIST_ERROR;
         return false;
      }

      gp.output_index_buffer = d12_list_upload_zero(list, index_size, 4);
      if (!gp.output_index_buffer)
         return false;

      *out_draw = agx_draw_indexed(
         gp.draw.index_count, gp.draw.instance_count, 0, 0, 0,
         gp.output_index_buffer, index_size, AGX_INDEX_SIZE_U32, false);
   } else if (pso->gs_info.shape == POLY_GS_SHAPE_STATIC_INDEXED) {
      /* Poly proved the emitted topology at compile time. Its compact table
       * is a byte index buffer (0xff is primitive restart), instanced once
       * per input primitive. Uploading that table avoids the dynamic index
       * writer while preserving arbitrary static strip breaks. */
      static_index_size = pso->gs_info.max_indices;
      static_index_buffer = d12_list_upload(
         list, pso->gs_info.topology, static_index_size, 1);
      if (!static_index_buffer)
         return false;

      if (!indirect) {
         *out_draw = agx_draw_indexed(
            gp.draw.index_count, gp.draw.instance_count, 0, 0, 0,
            static_index_buffer, static_index_size, AGX_INDEX_SIZE_U8, true);
      }
   } else if (!indirect) {
      /* Non-indexed Poly shapes still replace the API draw. Reusing the
       * application's original indexed/indirect arguments here makes the
       * GS raster shader index its generated vertex buffer with input indices
       * or StartVertexLocation. Zero-based cases passed only by accident. */
      *out_draw = (struct agx_draw){
         .b = agx_3d(gp.draw.index_count, gp.draw.instance_count, 1),
      };
   }

   if (d12_debug()->dump_so) {
      mesa_logi("SO draw: xfb=%u prefix=%u count_words=%u count=0x%llx "
                "stride=%u input_prims=%u grid=%ux%u shape=%u",
                pso->gs_info.xfb, pso->gs_info.prefix_sum,
                pso->gs_info.count_words,
                (unsigned long long)gp.count_buffer,
                gp.count_buffer_stride, gp.input_primitives, gp.grid[0],
                gp.grid[1], pso->gs_info.shape);
      for (unsigned i = 0; i < POLY_MAX_SO_BUFFERS; i++) {
         mesa_logi("  SO%u: base=0x%llx size=%u filled=0x%llx",
                   i, (unsigned long long)gp.xfb_base_original[i],
                   gp.xfb_size[i],
                   (unsigned long long)gp.xfb_offs_ptrs[i]);
      }
   }

   /* Both blocks are read through 64-bit constant loads whose offsets the
    * shaders bake, so the alignment is not cosmetic.
    */
   uint64_t vp_gpu = d12_list_upload(list, &vp, sizeof(vp), 8);
   uint64_t gp_gpu = d12_list_upload(list, &gp, sizeof(gp), 8);
   if (!vp_gpu || !gp_gpu)
      return false;

   if (!indirect && d12_debug()->validate_bindings) {
      size_t output_size = MAX2(out_size, 16);
      bool valid =
         d12_list_stream_contains(list, vp_gpu, sizeof(vp)) &&
         d12_list_stream_contains(list, gp_gpu, sizeof(gp)) &&
         d12_list_stream_contains(list, vp.output_buffer, output_size) &&
         (!vp.index_size_B || (vp.index_buffer && vp.index_buffer_range_el));
      mesa_logi("direct GS validate: vp=0x%llx gp=0x%llx "
                "output=[0x%llx,+0x%zx] index=[0x%llx,%u x %u] valid=%u",
                (unsigned long long)vp_gpu, (unsigned long long)gp_gpu,
                (unsigned long long)vp.output_buffer, output_size,
                (unsigned long long)vp.index_buffer,
                vp.index_buffer_range_el, vp.index_size_B, valid);
      if (!valid) {
         list->state = D12_LIST_ERROR;
         return false;
      }
   }

   if (indirect && d12_debug()->validate_bindings) {
      const uint64_t heap_base = list->device->heap_bo->va->addr;
      const uint64_t heap_size = list->device->heap_bo->size;
      bool valid =
         upstream_draw->b.ptr &&
         d12_list_stream_contains(list, vp_gpu, sizeof(vp)) &&
         d12_list_stream_contains(list, gp_gpu, sizeof(gp)) &&
         (!upstream_draw->indexed || dynamic_ibv ||
          (upstream_draw->index_buffer &&
           upstream_draw->index_buffer_range_B));

      mesa_logi("indirect GS validate: draw=0x%llx vp=0x%llx gp=0x%llx "
                "indexed=%u index=[0x%llx,+0x%x] "
                "heap=[0x%llx,+0x%llx] valid=%d",
                (unsigned long long)upstream_draw->b.ptr,
                (unsigned long long)vp_gpu, (unsigned long long)gp_gpu,
                upstream_draw->indexed,
                (unsigned long long)upstream_draw->index_buffer,
                upstream_draw->index_buffer_range_B,
                (unsigned long long)heap_base,
                (unsigned long long)heap_size, valid);
      if (!valid) {
         list->state = D12_LIST_ERROR;
         return false;
      }
   }

   /* THE INDIRECTION. agx_usc_uniform fills uniform registers FROM
    * memory, so publishing a POINTER means uploading the pointer and
    * pointing the uniform at that -- the address-of-an-address the root
    * table already uses (docs/findings.md).
    *
    * Pointing the uniform straight at the parameter block instead makes
    * the shader load the block's first eight bytes AS the block address:
    * poly_vertex_params begins with index_buffer, so the software vertex
    * shader took the index buffer pointer to be its own parameter block
    * and wrote nothing anywhere.
    *
    * VS_OUTPUTS is not like this: it is a VALUE (the output mask), so the
    * uniform correctly points at the mask itself.
    */
   uint64_t vp_ptr = d12_list_upload(list, &vp_gpu, sizeof(vp_gpu), 8);
   uint64_t gp_ptr = d12_list_upload(list, &gp_gpu, sizeof(gp_gpu), 8);
   if (!vp_ptr || !gp_ptr)
      return false;

   /* The mask lives in memory because the uniform is filled from memory. */
   uint64_t outputs_mask = sw_outputs;
   uint64_t outputs_gpu =
      d12_list_upload(list, &outputs_mask, sizeof(outputs_mask), 8);
   if (!outputs_gpu)
      return false;

   struct d12_dispatch_uniforms u = {
      .root = root_gpu,
      .textures = textures_gpu,
      .pbes = pbes_gpu,
      .heap = out_poly->heap,
      .heap_textures = out_poly->heap_textures,
      .heap_pbes = out_poly->heap_pbes,
      .heap_samplers = out_poly->heap_samplers,
      /* poly_nir_lower_sw_vs reads the base vertex from here. */
      .draw = draw_gpu,
      .vertex_params = vp_ptr,
      .geom_params = gp_ptr,
      .tess_params = out_poly->tess_params,
      .stat_query = stat_query,
      .vs_outputs = outputs_gpu,
      /* Published by the caller BEFORE this function runs: the
       * software VS fetches attributes through it. */
      .vb_params = out_poly->vb_params,
   };

   struct d12_segment *seg = d12_list_get_pregfx(list);
   if (!seg)
      return false;

   struct agx_grid sw_grid = agx_3d(vp.grid[0], vp.grid[1], 1);
   struct agx_grid gs_grid = agx_3d(gp.grid[0], gp.grid[1], 1);

   if (indirect) {
      list->uses_heap = true;
      if (dynamic_ibv) {
         const struct libagx_d3d12_gs_setup_indirect_ibv_args args = {
            .view = indirect_ibv,
            .draw = upstream_draw->b.ptr,
            .vp = vp_gpu,
            .p = gp_gpu,
            .heap = list->device->heap_desc_va,
            .vs_outputs = sw_outputs,
            .sparse_registry = list->sparse_registry_view_gpu,
            .null_address = list->device->null_va,
            .sparse_ro_offset = list->device->dev.sparse_ro_offset,
            .format_r16_uint = DXGI_FORMAT_R16_UINT,
            .format_r32_uint = DXGI_FORMAT_R32_UINT,
            .prim = input_prim,
            .is_prefix_summing = pso->gs_info.prefix_sum,
            .max_indices = pso->gs_info.max_indices,
            .shape = pso->gs_info.shape,
         };
         d12_list_dispatch_precomp_to(
            list, LIBAGX_D3D12_GS_SETUP_INDIRECT_IBV, &args, sizeof(args),
            agx_1d(1), true);
      } else {
         const struct libagx_gs_setup_indirect_args args = {
            .index_buffer = upstream_draw->indexed
                               ? upstream_draw->index_buffer
                               : list->device->null_va,
            .draw = upstream_draw->b.ptr,
            .vp = vp_gpu,
            .p = gp_gpu,
            .heap = list->device->heap_desc_va,
            .vs_outputs = sw_outputs,
            .index_size_B = upstream_draw->indexed
                               ? agx_index_size_to_B(upstream_draw->index_size)
                               : 0,
            .index_buffer_range_el = upstream_draw->indexed
               ? upstream_draw->index_buffer_range_B /
                    agx_index_size_to_B(upstream_draw->index_size)
               : 0,
            .prim = input_prim,
            .is_prefix_summing = pso->gs_info.prefix_sum,
            .max_indices = pso->gs_info.max_indices,
            .shape = pso->gs_info.shape,
         };
         d12_list_dispatch_precomp_to(list, LIBAGX_GS_SETUP_INDIRECT, &args,
                                      sizeof(args), agx_1d(1), true);
      }
      if (list->state != D12_LIST_RECORDING)
         return false;

      sw_grid = agx_grid_indirect_local(
         vp_gpu + offsetof(struct poly_vertex_params, grid));
      gs_grid = agx_grid_indirect_local(
         gp_gpu + offsetof(struct poly_geometry_params, grid));

      uint64_t indirect_draw =
         gp_gpu + offsetof(struct poly_geometry_params, draw);
      if (pso->gs_info.shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         *out_draw = agx_draw_indexed_indirect(
            indirect_draw, list->device->heap_bo->va->addr,
            list->device->heap_bo->size, AGX_INDEX_SIZE_U32, true);
      } else if (pso->gs_info.shape == POLY_GS_SHAPE_STATIC_INDEXED) {
         *out_draw = agx_draw_indexed_indirect(
            indirect_draw, static_index_buffer, static_index_size,
            AGX_INDEX_SIZE_U8, true);
      } else {
         *out_draw = agx_draw_indirect(indirect_draw);
      }
   }

   /* Software VS/TES first. Grids are in THREADS, not workgroups -- unlike the
    * API dispatch path, which multiplies. Each helper appends a full barrier
    * because every following program consumes the previous one's output.
    */
   if (!d12_list_dispatch_gs_stage(
          list, seg, pso->has_tess ? pso->sw_tes_va : pso->sw_vs_va,
          pso->has_tess ? &pso->sw_tes_info : &pso->sw_vs_info,
          sw_grid, wg, &u,
          pso->has_tess ? D3D12_SHADER_VISIBILITY_DOMAIN
                        : D3D12_SHADER_VISIBILITY_VERTEX))
      return false;

   if (pso->gs_info.xfb) {
      if (pso->gs_info.count_words &&
          !d12_list_dispatch_gs_stage(
             list, seg, pso->gs_count_va, &pso->gs_count_info,
             gs_grid, wg, &u,
             D3D12_SHADER_VISIBILITY_GEOMETRY))
         return false;

      if (pso->gs_info.prefix_sum) {
         struct libagx_prefix_sum_geom_args args = {.p = gp_gpu};
         d12_list_dispatch_precomp_to(
            list, LIBAGX_PREFIX_SUM_GEOM, &args, sizeof(args),
            agx_1d(1024 * pso->gs_info.count_words), true);
         if (list->state != D12_LIST_RECORDING)
            return false;
      }

      if (!d12_list_dispatch_gs_stage(
             list, seg, pso->gs_pre_va, &pso->gs_pre_info, agx_1d(1),
             agx_workgroup(1, 1, 1), &u,
             D3D12_SHADER_VISIBILITY_GEOMETRY))
         return false;
   }

   if (!d12_list_dispatch_gs_stage(
          list, seg, pso->gs_main_va, &pso->gs_main_info,
          gs_grid, wg, &u,
          D3D12_SHADER_VISIBILITY_GEOMETRY))
      return false;

   /* The rasterization draw's extent is closed-form in the shape, not a
    * GPU-written count -- that is only needed for indirect draws.
    */
   *out_verts = indirect ? 0 : gp.draw.index_count;
   *out_instances = indirect ? 0 : gp.draw.instance_count;
   *out_poly = u;
   return true;
}

/*
 * Run a tessellation pipeline's compute half and hand back the indexed
 * indirect draw that rasterises the result.
 *
 * Five dispatches, in this order, all into the pre-graphics stream:
 *
 *   1. the software vertex shader, writing the VS output buffer
 *   2. the hull shader, one workgroup per patch, writing the TCS buffer
 *   3. the tessellator in COUNT mode, writing only per-patch counts
 *   4. a prefix sum over those counts, which also allocates the index
 *      buffer out of the heap and writes the indirect draw descriptor
 *   5. the tessellator again in WITH_COUNTS mode, now emitting domain
 *      points and indices
 *
 * Steps 3 and 5 are the same kernel run twice on purpose: the counts are
 * prefix-summed IN PLACE between them, and step 5 reads counts[patch-1]
 * as its write offset. It cannot be fused (hk does the same).
 */

/*
 * VERTEX ATTRIBUTE FETCH parameters. The PSO resolved which signature
 * element each attribute feeds and where inside a vertex it sits; the view
 * supplies the address and the stride. agx_calculate_vbo_clamp folds the
 * attribute's byte offset into the base and returns the highest in-bounds
 * element index, and it is also what redirects an out-of-range attribute
 * at the zero page rather than at nothing.
 *
 * Returns 0 when the pipeline fetches no attributes, or
 * D12_VB_PARAMS_FAILED if an upload failed. Callers must publish this
 * BEFORE any pre-rasterisation compute dispatch -- see the call site.
 */
static uint64_t
d12_list_upload_vb_params(struct d12_cmd_list *list,
                          struct d12_pipeline_state *pso,
                          uint64_t *out_params_gpu)
{
   if (out_params_gpu)
      *out_params_gpu = 0;
   if (!pso->nr_vb_attribs)
      return 0;

   struct d12_vb_params vbp;
   memset(&vbp, 0, sizeof(vbp));

   for (unsigned i = 0; i < pso->nr_vb_attribs; i++) {
      const struct d12_vb_attrib *a = &pso->vb_attribs[i];

      if (!a->live || !list->vb[a->slot].addr) {
         vbp.base[i] = AGX_ZERO_PAGE_ADDRESS;
         continue;
      }

      vbp.clamp[i] = agx_calculate_vbo_clamp(
         list->vb[a->slot].addr, a->format, list->vb[a->slot].size_B,
         list->vb[a->slot].stride_B, a->offset_B, &vbp.base[i]);

      /* The shader multiplies the element index by this directly, so
       * it is in interchange elements rather than bytes. */
      vbp.stride_el[i] =
         a->interchange_align
            ? list->vb[a->slot].stride_B / a->interchange_align
            : 0;
   }

   if (d12_debug()->dump_vb) {
      for (unsigned i = 0; i < pso->nr_vb_attribs; i++) {
         const struct d12_vb_attrib *a = &pso->vb_attribs[i];
         mesa_logi("vb attrib %u: live=%d slot=%u off=%u ialign=%u "
                   "-> base=0x%llx clamp=%u stride_el=%u "
                   "(vb addr=0x%llx size=%u stride=%u)",
                   i, a->live, a->slot, a->offset_B, a->interchange_align,
                   (unsigned long long)vbp.base[i], vbp.clamp[i],
                   vbp.stride_el[i],
                   (unsigned long long)list->vb[a->slot].addr,
                   list->vb[a->slot].size_B, list->vb[a->slot].stride_B);
      }
   }

   uint64_t vbp_gpu = d12_list_upload(list, &vbp, sizeof(vbp), 8);
   if (!vbp_gpu)
      return D12_VB_PARAMS_FAILED;
   if (out_params_gpu)
      *out_params_gpu = vbp_gpu;

   uint64_t ptr = d12_list_upload(list, &vbp_gpu, sizeof(vbp_gpu), 8);
   return ptr ? ptr : D12_VB_PARAMS_FAILED;
}

static bool
d12_list_dispatch_indirect_vb_patches(
   struct d12_cmd_list *list, const struct d12_pipeline_state *pso,
   const struct d12_indirect_vb_patch *patches, unsigned patch_count,
   uint64_t params_gpu)
{
   if (!patch_count || !pso->nr_vb_attribs)
      return true;
   if (!params_gpu) {
      mesa_loge("ExecuteIndirect: no vertex parameter block for VBV patch");
      list->state = D12_LIST_ERROR;
      return false;
   }

   for (unsigned patch_index = 0; patch_index < patch_count; ++patch_index) {
      const struct d12_indirect_vb_patch *patch = &patches[patch_index];
      for (unsigned attribute = 0; attribute < pso->nr_vb_attribs;
           ++attribute) {
         const struct d12_vb_attrib *a = &pso->vb_attribs[attribute];
         if (!a->live || a->slot != patch->slot)
            continue;

         const struct libagx_d3d12_indirect_vbv_args args = {
            .out_base = params_gpu + offsetof(struct d12_vb_params, base) +
                        attribute * sizeof(uint64_t),
            .out_clamp = params_gpu + offsetof(struct d12_vb_params, clamp) +
                         attribute * sizeof(uint32_t),
            .out_stride = params_gpu +
                          offsetof(struct d12_vb_params, stride_el) +
                          attribute * sizeof(uint32_t),
            .in = patch->in,
            .draw_count = patch->draw_count,
            .previous_base = patch->previous_params
               ? patch->previous_params +
                    offsetof(struct d12_vb_params, base) +
                    attribute * sizeof(uint64_t)
               : 0,
            .previous_clamp = patch->previous_params
               ? patch->previous_params +
                    offsetof(struct d12_vb_params, clamp) +
                    attribute * sizeof(uint32_t)
               : 0,
            .previous_stride = patch->previous_params
               ? patch->previous_params +
                    offsetof(struct d12_vb_params, stride_el) +
                    attribute * sizeof(uint32_t)
               : 0,
            .sparse_registry = list->sparse_registry_view_gpu,
            .sparse_ro_offset = list->device->dev.sparse_ro_offset,
            .command_index = patch->command_index,
            .offset_B = a->offset_B,
            .element_size_B = util_format_get_blocksize(a->format),
            .interchange_align = a->interchange_align,
         };
         d12_list_dispatch_precomp_to(
            list, LIBAGX_D3D12_INDIRECT_VBV, &args, sizeof(args), agx_1d(1),
            true);
         if (list->state != D12_LIST_RECORDING)
            return false;
      }
   }

   return true;
}

static bool
d12_list_tess_prerast(struct d12_cmd_list *list,
                      struct d12_pipeline_state *pso, unsigned vertex_count,
                      unsigned instance_count, bool indexed,
                      unsigned first_index, uint64_t root_gpu,
                      uint64_t textures_gpu, uint64_t pbes_gpu,
                      uint64_t draw_gpu,
                      struct agx_draw *out_draw,
                      struct d12_dispatch_uniforms *out_poly)
{
   struct d12_device *device = list->device;

   if (!d12_list_require_legacy_compute(list))
      return false;

   if (!pso->tcs_output_patch_size || !pso->tcs_output_stride) {
      mesa_loge("tessellation pipeline has no patch layout");
      list->state = D12_LIST_ERROR;
      return false;
   }

   /* D3D12 carries the input patch size in the topology; the control
    * point count is the only thing API state contributes here.
    */
   unsigned input_patch_size = list->patch_control_points;
   if (!input_patch_size) {
      mesa_loge("draw with a tessellation pipeline but no patch topology; "
                "call IASetPrimitiveTopology with a control-point patch list");
      list->state = D12_LIST_ERROR;
      return false;
   }

   unsigned in_patches = vertex_count / input_patch_size;
   unsigned unrolled = in_patches * instance_count;
   if (!unrolled) {
      /* A partial patch or zero instances is an empty draw, not a command-list
       * error. Publish a safe zero-count raster descriptor and skip every
       * software tessellation stage. */
      *out_draw = agx_draw_indexed(
         0, 0, 0, 0, 0, list->device->null_va, 4,
         AGX_INDEX_SIZE_U32, false);
      return true;
   }

   const uint32_t wg_size[3] = {64, 1, 1};

   /* The VS prepass and its output buffer, exactly as the geometry path
    * builds them: the hull shader reads this.
    */
   struct poly_vertex_params vp;
   poly_vertex_params_init(&vp, pso->sw_vs_outputs, wg_size);
   poly_vertex_params_set_draw(&vp, vertex_count, instance_count);

   size_t vsout = poly_tcs_in_size((uint64_t)vertex_count * instance_count,
                                   pso->sw_vs_outputs);
   vp.output_buffer = d12_list_upload_zero(list, MAX2(vsout, 16), 16);
   if (!vp.output_buffer)
      return false;

   if (indexed) {
      enum agx_index_size index_size = list->ib.addr
                                          ? list->ib.index_size
                                          : AGX_INDEX_SIZE_U16;
      unsigned index_size_B = agx_index_size_to_B(index_size);
      unsigned range_el = list->ib.size_B / index_size_B;

      vp.index_size_B = index_size_B;
      if (list->ib.addr && first_index < range_el) {
         vp.index_buffer = list->ib.addr +
                           (uint64_t)first_index * index_size_B;
         vp.index_buffer_range_el = range_el - first_index;
      } else {
         vp.index_buffer = list->device->null_va;
         vp.index_buffer_range_el = 0;
      }
   } else {
      /* Never null: software input assembly materialises this pointer even
       * on the non-indexed path it does not take. */
      vp.index_buffer = vp.output_buffer;
   }

   /* One blob for the tessellator's scratch, carved by offset the way hk
    * does it. out_draws is the 5-word indexed draw the GPU fills in.
    */
   size_t tcs_out_B = (size_t)unrolled * pso->tcs_output_stride;
   size_t coord_B = (size_t)unrolled * 4;
   size_t count_B = (size_t)unrolled * 4;
   size_t draw_B = 5 * 4;

   uint64_t tcs_buffer = d12_list_upload_zero(list, tcs_out_B, 16);
   uint64_t coord_allocs = d12_list_upload_zero(list, coord_B, 4);
   uint64_t counts = d12_list_upload_zero(list, count_B, 4);
   uint64_t out_draws = d12_list_upload_zero(list, draw_B, 4);
   if (!tcs_buffer || !coord_allocs || !counts || !out_draws)
      return false;

   /* This list bump-allocates from the device tessellation heap, so it owes
    * the heap a GPU-side reset before it finishes. See d12_list_Close.
    */
   list->uses_heap = true;

   struct poly_tess_params tp = {
      .heap = (DEVICE(struct poly_heap))device->heap_desc_va,
      .patch_coord_buffer =
         (DEVICE(struct poly_tess_point))device->heap_bo->va->addr,
      .coord_allocs = (DEVICE(uint32_t))coord_allocs,
      .out_draws = (DEVICE(uint32_t))out_draws,
      .tcs_buffer = (DEVICE(float))tcs_buffer,
      .counts = (DEVICE(uint32_t))counts,
      /* index_buffer is GPU-written by the prefix sum. */
      /* No pipeline-statistics query is active. Pointing this at the heap
       * descriptor corrupts its 64-bit base when the tessellator performs
       * the documented atomic increment before allocating domain points. */
      .statistic = 0,
      .tcs_per_vertex_outputs = pso->tcs_per_vertex_outputs,
      .input_patch_size = input_patch_size,
      .output_patch_size = pso->tcs_output_patch_size,
      .tcs_patch_constants = pso->tcs_nr_patch_outputs,
      .patches_per_instance = in_patches,
      .tcs_stride_el = pso->tcs_output_stride / 4,
      .nr_patches = unrolled,
      .points_mode = pso->tess_points,
      .isolines = pso->tess_mode == TESS_PRIMITIVE_ISOLINES,
      .ccw = pso->tess_ccw && !pso->tess_points &&
             pso->tess_mode != TESS_PRIMITIVE_ISOLINES,
   };

   switch (pso->tess_spacing) {
   case TESS_SPACING_EQUAL:
      tp.partitioning = POLY_TESS_PARTITIONING_INTEGER;
      break;
   case TESS_SPACING_FRACTIONAL_ODD:
      tp.partitioning = POLY_TESS_PARTITIONING_FRACTIONAL_ODD;
      break;
   default:
      tp.partitioning = POLY_TESS_PARTITIONING_FRACTIONAL_EVEN;
      break;
   }

   uint64_t vp_gpu = d12_list_upload(list, &vp, sizeof(vp), 8);
   uint64_t tp_gpu = d12_list_upload(list, &tp, sizeof(tp), 8);
   if (!vp_gpu || !tp_gpu)
      return false;

   /* Published BY REFERENCE: the uniform is filled FROM memory, so the
    * pointer itself has to live somewhere. See docs/findings.md.
    */
   uint64_t outputs_mask = pso->sw_vs_outputs;
   uint64_t vp_ptr = d12_list_upload(list, &vp_gpu, sizeof(vp_gpu), 8);
   uint64_t tp_ptr = d12_list_upload(list, &tp_gpu, sizeof(tp_gpu), 8);
   uint64_t vso_ptr =
      d12_list_upload(list, &outputs_mask, sizeof(outputs_mask), 8);
   if (!vp_ptr || !tp_ptr || !vso_ptr)
      return false;

   struct d12_dispatch_uniforms u = {
      .root = root_gpu,
      .textures = textures_gpu,
      .pbes = pbes_gpu,
      .heap = out_poly->heap,
      .heap_textures = out_poly->heap_textures,
      .heap_pbes = out_poly->heap_pbes,
      .heap_samplers = out_poly->heap_samplers,
      .draw = draw_gpu,
      .vertex_params = vp_ptr,
      .tess_params = tp_ptr,
      .vs_outputs = vso_ptr,
      /* Published by the caller BEFORE this function runs: the
       * software VS fetches attributes through it. */
      .vb_params = out_poly->vb_params,
   };

   struct d12_segment *seg = d12_list_get_pregfx(list);
   if (!seg)
      return false;

   /* 1 and 2: software VS, then the hull shader. The hull shader runs one
    * WORKGROUP per patch -- its workgroup size is the output patch size,
    * not 64, because a patch has to fit in one workgroup for the barrier
    * elision poly relies on.
    */
   const struct {
      uint64_t va;
      const struct agx_shader_info *info;
      struct agx_grid grid;
      struct agx_workgroup wg;
      D3D12_SHADER_VISIBILITY visibility;
   } stages[] = {
      {pso->sw_vs_va, &pso->sw_vs_info, agx_3d(vp.grid[0], vp.grid[1], 1),
       agx_workgroup(64, 1, 1), D3D12_SHADER_VISIBILITY_VERTEX},
      {pso->tcs_va, &pso->tcs_info,
       agx_3d(in_patches * pso->tcs_info.workgroup_size[0], instance_count, 1),
       agx_workgroup(pso->tcs_info.workgroup_size[0], 1, 1),
       D3D12_SHADER_VISIBILITY_HULL},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(stages); i++) {
      if (!d12_segment_reserve_scratch(list, seg, D12_SCRATCH_CS,
                                       stages[i].info))
         return false;

      uint32_t usc = d12_list_build_compute_usc(list, stages[i].va,
                                                stages[i].info, &u, false,
                                                stages[i].visibility, NULL);
      if (!usc)
         return false;

      if (!d12_list_check_pregfx_space(list, D12_CDM_DISPATCH_BYTES))
         return false;

      struct agx_cdm_launch_word_0_packed launch =
         d12_compute_launch_word(stages[i].info);

      if (!d12_list_emit_legacy_cdm_launch(
             list, &list->pregfx_cur, stages[i].grid, stages[i].wg, launch,
             usc))
         return false;
      if (!d12_list_emit_cdm_cache_flush(list, &list->pregfx_cur))
         return false;
      seg->dispatches++;
   }

   /* 3, 4, 5: the tessellator twice with a prefix sum between. The mode
    * selects the kernel; the _0/_1 suffix is the COUNT/WITH_COUNTS
    * shader-key variant, compiled as separate binaries rather than a
    * runtime branch.
    */
   enum libagx_program count_prog, emit_prog;
   if (pso->tess_mode == TESS_PRIMITIVE_ISOLINES) {
      count_prog = LIBAGX_TESS_ISOLINE_0;
      emit_prog = LIBAGX_TESS_ISOLINE_1;
   } else if (pso->tess_mode == TESS_PRIMITIVE_QUADS) {
      count_prog = LIBAGX_TESS_QUAD_0;
      emit_prog = LIBAGX_TESS_QUAD_1;
   } else {
      count_prog = LIBAGX_TESS_TRI_0;
      emit_prog = LIBAGX_TESS_TRI_1;
   }

   struct {
      uint64_t params;
   } targs = {tp_gpu};

   d12_list_dispatch_precomp_to(list, count_prog, &targs, sizeof(targs),
                                agx_1d(unrolled), true);

   struct {
      uint64_t params;
      uint64_t c_prims;
      uint64_t c_inv;
   } sargs = {tp_gpu, 0, 0};

   d12_list_dispatch_precomp_to(list, LIBAGX_PREFIX_SUM_TESS_0, &sargs,
                                sizeof(sargs), agx_1d(1024), true);

   d12_list_dispatch_precomp_to(list, emit_prog, &targs, sizeof(targs),
                                agx_1d(unrolled), true);

   if (list->state != D12_LIST_RECORDING)
      return false;

   if (d12_debug()->dump_tess) {
      /* The GPU faults here read a real but unmapped address that changes
       * every run, so the address alone says nothing. Print the extent of
       * every allocation this draw hands the tessellator, in the same run,
       * and the faulting address can be attributed to one of them (or to
       * none, which is its own answer).
       */
      mesa_logi("tess VA map:");
      mesa_logi("  heap        0x%llx..0x%llx",
                (unsigned long long)device->heap_bo->va->addr,
                (unsigned long long)(device->heap_bo->va->addr +
                                     device->heap_bo->size));
      mesa_logi("  heap_desc   0x%llx  (bottom now %u, size %u)",
                (unsigned long long)device->heap_desc_va,
                device->heap_desc_cpu ? device->heap_desc_cpu->bottom : 0,
                device->heap_desc_cpu ? device->heap_desc_cpu->size : 0);
      mesa_logi("  tcs_buffer  0x%llx..0x%llx  (%zu B)",
                (unsigned long long)tcs_buffer,
                (unsigned long long)(tcs_buffer + tcs_out_B), tcs_out_B);
      mesa_logi("  coord_alloc 0x%llx..0x%llx  (%zu B)",
                (unsigned long long)coord_allocs,
                (unsigned long long)(coord_allocs + coord_B), coord_B);
      mesa_logi("  counts      0x%llx..0x%llx  (%zu B)",
                (unsigned long long)counts,
                (unsigned long long)(counts + count_B), count_B);
      mesa_logi("  out_draws   0x%llx..0x%llx  (%zu B)",
                (unsigned long long)out_draws,
                (unsigned long long)(out_draws + draw_B), draw_B);
      mesa_logi("  vs_out      0x%llx..0x%llx  (%zu B)",
                (unsigned long long)vp.output_buffer,
                (unsigned long long)(vp.output_buffer + vsout), vsout);
      mesa_logi("  stream_bo   0x%llx..0x%llx",
                (unsigned long long)list->stream_bo->va->addr,
                (unsigned long long)(list->stream_bo->va->addr +
                                     list->stream_bo->size));

      list->device->dbg_gs_vb = d12_list_data_cpu(list, vp.output_buffer);
      list->device->dbg_gs_vb_size = MAX2(vsout, 16);
      list->device->dbg_tess_draw = d12_list_data_cpu(list, out_draws);
      list->device->dbg_tess_counts = d12_list_data_cpu(list, counts);
      list->device->dbg_tess_counts_n = unrolled;
      list->device->dbg_tcs_buf = d12_list_data_cpu(list, tcs_buffer);
      list->device->dbg_tcs_buf_n =
         unrolled * (pso->tcs_output_stride / 4);
      mesa_logi("tess: patches=%u unrolled=%u patch_size=%u out_patch=%u "
                "hs_wg=%u stride=%u mode=%u part=%u tcs_pv_out=0x%llx",
                in_patches, unrolled, input_patch_size,
                pso->tcs_output_patch_size, pso->tcs_info.workgroup_size[0],
                pso->tcs_output_stride,
                pso->tess_mode, tp.partitioning,
                (unsigned long long)pso->tcs_per_vertex_outputs);
   }

   /* The rasterisation draw is always indexed, always indirect and always
    * U32: the index count only exists after the GPU prefix sum. The WHOLE
    * heap is the index buffer and out_draws[2] carries the element offset
    * into it.
    */
   *out_draw = agx_draw_indexed_indirect(out_draws, device->heap_bo->va->addr,
                                         device->heap_bo->size,
                                         AGX_INDEX_SIZE_U32, false);
   *out_poly = u;
   return true;
}

/*
 * Run a mesh pipeline's compute half and report the draw it becomes.
 *
 * Three dispatches into the pre-graphics stream:
 *   1. the mesh shader itself, one workgroup per DispatchMesh group
 *   2. ms_setup, turning the GPU-written counts into an indirect draw
 *   3. (nothing else -- the rasterization VS is the hardware stage)
 *
 * The mesh shader writes its vertices, primitives and indices through
 * D12_UNIFORM_MESH_PARAMS. Unlike geometry and tessellation there is no
 * poly emulation behind this: the ABI is this driver's own.
 */
struct d12_mesh_stage {
   uint64_t va;
   const struct agx_shader_info *info;
   struct agx_grid grid;
   struct agx_workgroup wg;
   D3D12_SHADER_VISIBILITY visibility;
};

static bool
d12_list_mesh_prerast(struct d12_cmd_list *list,
                      struct d12_pipeline_state *pso, unsigned tg_x,
                      unsigned tg_y, unsigned tg_z, uint64_t root_gpu,
                      uint64_t textures_gpu, uint64_t pbes_gpu,
                      uint64_t draw_gpu,
                      struct agx_draw *out_draw,
                      struct d12_dispatch_uniforms *out_poly)
{
   struct d12_device *device = list->device;

   if (!d12_list_require_legacy_compute(list))
      return false;

   unsigned groups = tg_x * tg_y * tg_z;
   if (!groups) {
      mesa_loge("DispatchMesh with an empty thread group count");
      list->state = D12_LIST_ERROR;
      return false;
   }

   /* Worst case per workgroup, from the shader's declared maxima. D3D12
    * caps a mesh shader at 256 vertices and 256 primitives per group.
    */
   const unsigned max_verts = 256, max_prims = 256;
   unsigned nr_slots = MAX2(util_last_bit64(pso->ms_outputs), 1);

   /* Fixed stride: the translator addresses slot (sigId + row) within a
    * D12_MESH_MAX_SLOTS-wide record per vertex, so the buffer has to be
    * sized that way whatever the shader actually writes.
    */
   const size_t vtx_stride = D12_MESH_MAX_SLOTS * D12_MESH_SLOT_STRIDE;
   size_t verts_B = (size_t)groups * max_verts * vtx_stride;
   size_t prims_B = (size_t)groups * max_prims * vtx_stride;
   size_t idx_B = (size_t)groups * max_prims * 3 * 4;

   /* With an amplification shader the mesh grid is GPU-written, so the
    * worst case has to cover whatever it asks for. D3D12 caps
    * DispatchMesh at 2^22 groups total but a test-sized bound is what
    * the buffers can actually hold; anything larger is refused rather
    * than silently overrun.
    */
   if (pso->has_amp)
      groups = D12_MESH_MAX_AMP_GROUPS;

   uint64_t counts = d12_list_upload_zero(list, 8, 8);
   uint64_t vertices = d12_list_upload_zero(list, verts_B, 16);
   uint64_t primitives = d12_list_upload_zero(list, prims_B, 16);
   uint64_t indices = d12_list_upload_zero(list, idx_B, 4);
   uint64_t drawdesc = d12_list_upload_zero(list, 5 * 4, 4);
   uint64_t dispatch = d12_list_upload_zero(list, 3 * 4, 4);
   uint64_t gridbuf = d12_list_upload_zero(list, 6 * 4, 4);
   uint64_t payload =
      pso->as_payload_size
         ? d12_list_upload_zero(list, pso->as_payload_size, 16)
         : 0;
   if (!counts || !vertices || !primitives || !indices || !drawdesc ||
       !dispatch || !gridbuf)
      return false;

   /* The parameter block the shader indirects through. Laid out by
    * D12_MESH_OFF_*; the payload is null because amplification shaders
    * are refused at PSO creation.
    */
   uint64_t block[8] = {counts,   vertices, primitives, indices,
                        payload,  dispatch, drawdesc,   gridbuf};
   uint64_t mp_gpu = d12_list_upload(list, block, sizeof(block), 8);
   if (!mp_gpu)
      return false;

   /* Published BY REFERENCE, like every other parameter block. */
   uint64_t mp_ptr = d12_list_upload(list, &mp_gpu, sizeof(mp_gpu), 8);
   if (!mp_ptr)
      return false;

   struct d12_dispatch_uniforms u = {
      .root = root_gpu,
      .textures = textures_gpu,
      .pbes = pbes_gpu,
      .heap = out_poly->heap,
      .heap_textures = out_poly->heap_textures,
      .heap_pbes = out_poly->heap_pbes,
      .heap_samplers = out_poly->heap_samplers,
      .draw = draw_gpu,
      .mesh_params = mp_ptr,
      /* Published by the caller BEFORE this function runs: the
       * software VS fetches attributes through it. */
      .vb_params = out_poly->vb_params,
   };

   struct d12_segment *seg = d12_list_get_pregfx(list);
   if (!seg)
      return false;

   /* With an amplification shader the chain grows at the front: AS runs
    * on the DispatchMesh grid the application gave, writes the payload
    * and its own DispatchMesh(x,y,z), then as_setup turns that thread
    * GROUP count into the six-word indirect-local grid the mesh
    * dispatch consumes. Without one, the mesh grid is known on the CPU.
    */
   struct d12_mesh_stage stages[4];
   unsigned nr_stages = 0;

   if (pso->has_amp) {
      stages[nr_stages++] = (struct d12_mesh_stage){
         pso->as_va, &pso->as_info,
         agx_3d(tg_x * pso->as_workgroup[0], tg_y * pso->as_workgroup[1],
                tg_z * pso->as_workgroup[2]),
         agx_workgroup(pso->as_workgroup[0], pso->as_workgroup[1],
                       pso->as_workgroup[2]),
         D3D12_SHADER_VISIBILITY_AMPLIFICATION};

      stages[nr_stages++] = (struct d12_mesh_stage){
         pso->as_setup_va, &pso->as_setup_info, agx_3d(1, 1, 1),
         agx_workgroup(1, 1, 1), D3D12_SHADER_VISIBILITY_ALL};

      stages[nr_stages++] = (struct d12_mesh_stage){
         pso->ms_va, &pso->ms_info, agx_grid_indirect_local(gridbuf),
         agx_workgroup(pso->ms_workgroup[0], pso->ms_workgroup[1],
                       pso->ms_workgroup[2]), D3D12_SHADER_VISIBILITY_MESH};
   } else {
      stages[nr_stages++] = (struct d12_mesh_stage){
         pso->ms_va, &pso->ms_info,
         agx_3d(tg_x * pso->ms_workgroup[0], tg_y * pso->ms_workgroup[1],
                tg_z * pso->ms_workgroup[2]),
         agx_workgroup(pso->ms_workgroup[0], pso->ms_workgroup[1],
                       pso->ms_workgroup[2]), D3D12_SHADER_VISIBILITY_MESH};
   }

   stages[nr_stages++] = (struct d12_mesh_stage){
      pso->ms_setup_va, &pso->ms_setup_info, agx_3d(1, 1, 1),
      agx_workgroup(1, 1, 1), D3D12_SHADER_VISIBILITY_ALL};

   for (unsigned i = 0; i < nr_stages; i++) {
      if (!d12_segment_reserve_scratch(list, seg, D12_SCRATCH_CS,
                                       stages[i].info))
         return false;

      uint32_t usc = d12_list_build_compute_usc(list, stages[i].va,
                                                stages[i].info, &u, false,
                                                stages[i].visibility, NULL);
      if (!usc)
         return false;

      if (!d12_list_check_pregfx_space(list, D12_CDM_DISPATCH_BYTES))
         return false;

      struct agx_cdm_launch_word_0_packed launch =
         d12_compute_launch_word(stages[i].info);

      if (!d12_list_emit_legacy_cdm_launch(
             list, &list->pregfx_cur, stages[i].grid, stages[i].wg, launch,
             usc))
         return false;
      if (!d12_list_emit_cdm_cache_flush(list, &list->pregfx_cur))
         return false;
      seg->dispatches++;
   }

   if (list->state != D12_LIST_RECORDING)
      return false;

   if (d12_debug()->dump_mesh) {
      device->dbg_tess_draw = d12_list_data_cpu(list, drawdesc);
      device->dbg_tess_counts = d12_list_data_cpu(list, counts);
      device->dbg_tess_counts_n = 2;
      device->dbg_mesh_verts = d12_list_data_cpu(list, vertices);
      device->dbg_mesh_idx = d12_list_data_cpu(list, indices);
      device->dbg_mesh_grid = d12_list_data_cpu(list, gridbuf);
      device->dbg_mesh_disp = d12_list_data_cpu(list, dispatch);
      mesa_logi("mesh: groups=%u slots=%u vtx_stride=%zu idx_B=%zu amp=%d",
                groups, nr_slots, vtx_stride, idx_B, pso->has_amp);
   }

   *out_draw = agx_draw_indexed_indirect(drawdesc, indices, idx_B,
                                         AGX_INDEX_SIZE_U32, false);
   *out_poly = u;
   return true;
}

static D3D12_INDEX_BUFFER_STRIP_CUT_VALUE
d12_effective_strip_cut(const struct d12_cmd_list *list,
                        const struct d12_pipeline_state *pso)
{
   return pso->raster.dynamic_strip_cut ? list->strip_cut : pso->strip_cut;
}

static bool
d12_strip_cut_enabled(const struct d12_cmd_list *list,
                      const struct d12_pipeline_state *pso,
                      enum mesa_prim topology)
{
   if (topology != MESA_PRIM_LINE_STRIP &&
       topology != MESA_PRIM_TRIANGLE_STRIP &&
       topology != MESA_PRIM_TRIANGLE_FAN)
      return false;

   return d12_effective_strip_cut(list, pso) !=
          D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
}

/*
 * Turn one indexed strip draw with D3D12 primitive restart into a GPU-written
 * indexed-list draw.  Geometry and transform feedback run in compute, where
 * hardware VDM restart is too late; Poly therefore needs the restart markers
 * removed before its software input assembly and primitive decomposition.
 *
 * The input is always indirect (a direct D3D12 draw is uploaded by the caller),
 * so both direct and ExecuteIndirect take this same ordered libagx path.  The
 * generated indices live in the device Poly heap and the returned descriptor
 * names that heap with restart disabled.
 */
static bool
d12_list_unroll_gs_restart(struct d12_cmd_list *list,
                           const struct d12_pipeline_state *pso,
                           enum mesa_prim topology,
                           const struct agx_draw *input,
                           uint64_t indirect_ibv,
                           struct agx_draw *output)
{
   assert(input->indexed && input->restart && agx_is_indirect(input->b));

   uint64_t out_draw =
      d12_list_upload_zero(list, sizeof(struct poly_indirect_draw), 4);
   if (!out_draw)
      return false;

   uint32_t restart_index =
      d12_effective_strip_cut(list, pso) ==
            D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF
         ? UINT16_MAX
         : UINT32_MAX;

   if (indirect_ibv) {
      const struct libagx_d3d12_unroll_restart_ibv_args args = {
         .heap = list->device->heap_desc_va,
         .view = indirect_ibv,
         .in_draw = input->b.ptr,
         .out_draw = out_draw,
         .sparse_registry = list->sparse_registry_view_gpu,
         .null_address = list->device->null_va,
         .sparse_ro_offset = list->device->dev.sparse_ro_offset,
         .format_r16_uint = DXGI_FORMAT_R16_UINT,
         .format_r32_uint = DXGI_FORMAT_R32_UINT,
         .restart_index = restart_index,
         .flatshade_first = true,
      };

      list->uses_heap = true;
      d12_list_dispatch_precomp_to(
         list,
         LIBAGX_D3D12_UNROLL_RESTART_IBV(poly_compact_prim(topology)),
         &args, sizeof(args), agx_1d(1024), true);
      if (list->state != D12_LIST_RECORDING)
         return false;

      *output = agx_draw_indexed_indirect(
         out_draw, list->device->heap_bo->va->addr,
         list->device->heap_bo->size, AGX_INDEX_SIZE_U32, false);
      return true;
   }

   struct libagx_unroll_restart_args args = {
      .heap = list->device->heap_desc_va,
      .index_buffer = input->index_buffer,
      .in_draw = input->b.ptr,
      .out_draw = out_draw,
      .max_draws = 1,
      .restart_index = restart_index,
      .index_buffer_size_el =
         input->index_buffer_range_B /
         agx_index_size_to_B(input->index_size),
      .index_size_log2 = input->index_size,
      .flatshade_first = true,
   };

   if (d12_debug()->validate_bindings) {
      bool valid = input->b.ptr && input->index_buffer &&
                   input->index_buffer_range_B &&
                   d12_list_stream_contains(list, out_draw,
                                            sizeof(struct poly_indirect_draw));
      mesa_logi("GS restart unroll: in=0x%llx index=[0x%llx,+0x%x] "
                "out=0x%llx restart=0x%x prim=%u valid=%u",
                (unsigned long long)input->b.ptr,
                (unsigned long long)input->index_buffer,
                input->index_buffer_range_B,
                (unsigned long long)out_draw, restart_index, topology, valid);
      if (!valid) {
         list->state = D12_LIST_ERROR;
         return false;
      }
   }

   list->uses_heap = true;
   d12_list_dispatch_precomp_to(
      list, LIBAGX_UNROLL_RESTART(poly_compact_prim(topology)), &args,
      sizeof(args), agx_1d(1024), true);
   if (list->state != D12_LIST_RECORDING)
      return false;

   *output = agx_draw_indexed_indirect(
      out_draw, list->device->heap_bo->va->addr,
      list->device->heap_bo->size, input->index_size, false);
   return true;
}

static struct d12_gfx_upload_cache_entry *
d12_gfx_upload_cache_lookup(struct d12_cmd_list *list,
                            const struct d12_pipeline_state *pso,
                            const struct agx_tilebuffer_layout *tib,
                            uint64_t root, uint64_t textures, uint64_t pbes,
                            uint64_t heap, uint64_t heap_textures,
                            uint64_t heap_pbes, uint64_t heap_samplers,
                            uint64_t draw, uint64_t blend)
{
   uint64_t hash = ((uintptr_t)pso >> 4) ^ root ^ textures ^ pbes ^ draw ^
                   blend ^ heap ^ heap_textures ^ heap_pbes ^ heap_samplers;
   unsigned start = hash & (D12_GFX_UPLOAD_CACHE_SIZE - 1);

   static_assert(
      (D12_GFX_UPLOAD_CACHE_SIZE & (D12_GFX_UPLOAD_CACHE_SIZE - 1)) == 0,
      "graphics upload cache must be power-of-two sized");

   for (unsigned probe = 0; probe < D12_USC_CACHE_PROBES; probe++) {
      struct d12_gfx_upload_cache_entry *entry =
         &list->gfx_upload_cache[(start + probe) &
                                 (D12_GFX_UPLOAD_CACHE_SIZE - 1)];

      if (!entry->valid)
         return entry;

      if (entry->pso == pso && entry->root == root &&
          entry->textures == textures && entry->pbes == pbes &&
          entry->heap == heap && entry->heap_textures == heap_textures &&
          entry->heap_pbes == heap_pbes &&
          entry->heap_samplers == heap_samplers && entry->draw == draw &&
          entry->blend == blend && memcmp(&entry->tib, tib, sizeof(*tib)) == 0)
         return entry;
   }

   /* Evict within the window; the caller rebuilds into the returned slot. */
   struct d12_gfx_upload_cache_entry *victim =
      &list->gfx_upload_cache[(start + (list->usc_cache_tick++ %
                                        D12_USC_CACHE_PROBES)) &
                              (D12_GFX_UPLOAD_CACHE_SIZE - 1)];
   victim->valid = false;
   return victim;
}

static struct d12_fragment_state_cache_entry *
d12_fragment_state_cache_lookup(struct d12_cmd_list *list,
                                const struct d12_pipeline_state *pso,
                                uint32_t fs_usc, uint32_t varyings,
                                enum mesa_prim topology, uint8_t dsv_flags,
                                uint8_t stencil_ref_front,
                                uint8_t stencil_ref_back, bool has_ds,
                                bool has_stencil, bool depth_bias_enable,
                                bool occlusion_counting)
{
   uint64_t hash = ((uintptr_t)pso >> 4) ^ fs_usc ^ ((uint64_t)varyings << 17) ^
                   ((uint64_t)topology << 9) ^ ((uint64_t)dsv_flags << 25) ^
                   ((uint64_t)stencil_ref_front << 33) ^
                   ((uint64_t)stencil_ref_back << 41) ^
                   ((uint64_t)has_ds << 49) ^ ((uint64_t)has_stencil << 50) ^
                   ((uint64_t)depth_bias_enable << 51) ^
                   ((uint64_t)occlusion_counting << 52);
   unsigned start = hash & (D12_FRAGMENT_STATE_CACHE_SIZE - 1);

   static_assert((D12_FRAGMENT_STATE_CACHE_SIZE &
                  (D12_FRAGMENT_STATE_CACHE_SIZE - 1)) == 0,
                 "fragment-state cache must be power-of-two sized");

   for (unsigned probe = 0; probe < D12_USC_CACHE_PROBES; probe++) {
      struct d12_fragment_state_cache_entry *entry =
         &list->fragment_state_cache[(start + probe) &
                                     (D12_FRAGMENT_STATE_CACHE_SIZE - 1)];

      if (!entry->valid)
         return entry;

      if (entry->pso == pso && entry->fs_usc == fs_usc &&
          entry->varyings == varyings && entry->topology == topology &&
          entry->dsv_flags == dsv_flags &&
          entry->stencil_ref_front == stencil_ref_front &&
          entry->stencil_ref_back == stencil_ref_back &&
          entry->has_ds == has_ds && entry->has_stencil == has_stencil &&
          entry->depth_bias_enable == depth_bias_enable &&
          entry->occlusion_counting == occlusion_counting)
         return entry;
   }

   struct d12_fragment_state_cache_entry *victim =
      &list->fragment_state_cache[(start + (list->usc_cache_tick++ %
                                            D12_USC_CACHE_PROBES)) &
                                  (D12_FRAGMENT_STATE_CACHE_SIZE - 1)];
   victim->valid = false;
   return victim;
}

static bool
d12_emit_ppp_state(struct d12_cmd_list *list, uint8_t **out, uint64_t gpu,
                   size_t size)
{
   if (agx_ppp_legacy_payload_supported(list->device->dev.chip) &&
       agx_ppp_emit_state(list->device->dev.chip, out, gpu, size))
      return true;

   mesa_loge("no grounded legacy PPP payload/finalizer for %s",
             agx_chip_name(list->device->dev.chip));
   list->state = D12_LIST_ERROR;
   return false;
}

static bool
d12_emit_g17p_triangle(struct d12_cmd_list *list, struct d12_segment *seg,
                        struct d12_pipeline_state *pso, uint32_t vs_usc,
                        uint32_t fs_usc, unsigned vertex_count,
                        unsigned instance_count, unsigned first_vertex)
{
   if (!d12_g17p_baseline_graphics_enabled(&list->device->dev) ||
       !seg->agx3_active_context_selector_valid ||
       !pso->vs.g17p_root_only || !pso->fs.g17p_root_only ||
       pso->vs.g17p_writes_memory || pso->fs.g17p_writes_memory ||
       pso->has_gs || pso->has_tess || pso->has_mesh || pso->nr_vb_attribs ||
       pso->raster.polygon_mode != AGX_POLYGON_MODE_FILL ||
       pso->raster.depth_bias || pso->raster.slope_scaled_depth_bias ||
       (seg->ds && (!d12_g17p_depth_enabled(&list->device->dev) ||
                    !d12_g17p_depth_segment_supported(seg) ||
                    pso->ds.stencil_test ||
                    pso->ds.dsv_format != DXGI_FORMAT_D32_FLOAT)) ||
       seg->tib.nr_samples != 1 || list->viewport_count != 1 ||
       list->scissor_count != 1 || list->topology != MESA_PRIM_TRIANGLES ||
       pso->fs.info.writes_sample_mask || pso->fs.info.reads_tib ||
       seg->scissor_count >= D12_SCISSORS_PER_SEGMENT ||
       seg->depth_bias_count >= D12_DEPTH_BIASES_PER_SEGMENT ||
       list->active_occlusion_queries || list->active_pipeline_stats_heap)
      goto fail;
   if (!d12_segment_ensure_viewport_arrays(list, seg))
      goto fail;
   struct d12_viewport_hw_state states[D12_VIEWPORT_STATE_MAX];
   if (d12_viewport_build_hw_state(list->viewport_count, list->viewports,
          list->scissor_count, list->scissors, seg->width_px, seg->height_px,
          states) != 1)
      goto fail;
   uint64_t compact;
   void *payload = d12_list_alloc_g17p_vdm(list, AGX_G17P_BIND_STATE_BYTES, 64, &compact);
   if (!payload)
      goto fail;
   struct d12_agx3_triangle_bind_desc desc = {
      .compact_va = compact,
      .fragment_program = list->device->dev.shader_base + fs_usc,
      .linked_fragment_control = pso->fs.linked.fragment_control.opaque[0],
      .raw_output_select = pso->uvs.osel.opaque[0],
      .uvs_size = pso->uvs.size, .user_size = pso->uvs.user_size,
      .coefficient_bindings = pso->fs.linked.cf.nr_bindings,
      .reads_z = pso->fs.linked.cf.reads_z,
      .writes_memory = pso->fs.g17p_writes_memory,
      .writes_depth = pso->fs.info.depth_layout != FRAG_DEPTH_LAYOUT_UNCHANGED,
      .needs_quad_derivatives = pso->fs.info.needs_quad_derivatives,
      .has_preamble = pso->fs.info.has_preamble,
      .cull_mode = pso->raster.cull_front | (pso->raster.cull_back << 1),
      .front_face_ccw = pso->raster.front_face_ccw,
      .depth_clip = pso->raster.depth_clip,
      .has_depth = seg->ds != NULL,
      .depth_test = pso->ds.depth_test,
      .depth_write = pso->ds.depth_write,
      .depth_compare = pso->ds.depth_func,
      .raster = {
         .scissor_enable = true,
         .min_x = states[0].clip[0], .min_y = states[0].clip[1],
         .max_x = states[0].clip[2], .max_y = states[0].clip[3],
         .max_z_raw = fui(1.0f),
      },
      .viewport.viewport_count = 1,
   };
   uint8_t region_clip[AGX_REGION_CLIP_LENGTH];
   agx_pack(region_clip, REGION_CLIP, cfg) {
      cfg.enable = true;
      cfg.min_x = states[0].clip[0] / 32;
      cfg.min_y = states[0].clip[1] / 32;
      cfg.max_x = DIV_ROUND_UP(MAX2(states[0].clip[2], 1), 32);
      cfg.max_y = DIV_ROUND_UP(MAX2(states[0].clip[3], 1), 32);
   }
   memcpy(&desc.viewport.region_clip_raw[0], region_clip, sizeof(region_clip));
   desc.viewport.viewports[0] = (struct agx_ppp_viewport_transform_raw){
      .translate_x = fui(states[0].transform[0]), .scale_x = fui(states[0].transform[3]),
      .translate_y = fui(states[0].transform[1]), .scale_y = fui(states[0].transform[4]),
      .translate_z = fui(states[0].transform[2]), .scale_z = fui(states[0].transform[5]),
   };
   uint8_t control[AGX_VIEWPORT_CONTROL_LENGTH];
   agx_pack(control, VIEWPORT_CONTROL, cfg) ;
   memcpy(&desc.viewport.viewport_control_raw, control, sizeof(control));
   struct d12_agx3_triangle_bind generated;
   if (!d12_agx3_triangle_bind_generate(list->device->dev.chip, &desc, &generated))
      goto fail;
   memcpy(payload, generated.bytes, sizeof(generated.bytes));
   uint8_t *scissor = d12_list_data_cpu(list, seg->scissor_base);
   uint8_t *depth_bias = d12_list_data_cpu(list, seg->depth_bias_base);
   if (!scissor || !depth_bias)
      goto fail;
   memcpy(scissor + seg->scissor_count * AGX_SCISSOR_LENGTH,
          generated.raster.scissor, sizeof(generated.raster.scissor));
   memcpy(depth_bias + seg->depth_bias_count * AGX_DEPTH_BIAS_LENGTH,
          generated.raster.depth_bias, sizeof(generated.raster.depth_bias));
   seg->scissor_count++;
   seg->depth_bias_count++;
   uint64_t draw_compact;
   void *draw_payload = d12_list_alloc_g17p_vdm(list, 8, 4, &draw_compact);
   uint32_t draw_offset;
   if (!draw_payload || !agx_g17p_upload_draw_indices(draw_payload, 8, draw_compact,
          seg->scissor_count * AGX_SCISSOR_LENGTH,
          seg->depth_bias_count * AGX_DEPTH_BIAS_LENGTH, 1, &draw_offset))
      goto fail;
   struct d12_agx3_vdm_state_update_draw_info info = {
      .source_layout = pso->vs.agx3_vdm_state_update.source_layout,
      .draw_path = D12_AGX3_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED,
      .rasterization_enabled = pso->rasterization_enabled,
      .template_info = pso->vs.agx3_vdm_state_update.template_info,
   };
   info.template_info.pipeline = list->device->dev.shader_base + vs_usc;
   info.template_info.raw_active_context_selector = seg->agx3_raw_active_context_selector;
   struct d12_agx3_vdm_state_update vertex;
   const struct agx_vdm_hal200_direct_draw_fields draw_fields = {
      .draw_state_context_offset = draw_offset, .primitive = AGX_PRIMITIVE_TRIANGLES,
      .vertex_count = vertex_count, .instance_count = instance_count,
      .vertex_start = first_vertex,
   };
   struct agx_vdm_hal200_direct_draw draw;
   if (!d12_agx3_encode_vdm_state_update_draw(list->device->dev.chip, &info, &vertex) ||
       !agx_vdm_hal200_direct_draw_pack(list->device->dev.chip, &draw_fields, &draw) ||
       !d12_list_check_stream_space(list, (size_t)vertex.length +
                                             generated.upload.tail.length +
                                             draw.length))
      goto fail;
   uint8_t *out = list->stream_cur;
   memcpy(out, vertex.bytes, vertex.length); out += vertex.length;
   memcpy(out, generated.upload.tail.bytes, generated.upload.tail.length);
   out += generated.upload.tail.length;
   memcpy(out, draw.bytes, draw.length); out += draw.length;
   list->stream_cur = out;
   seg->has_draws = true;
   return true;
fail:
   mesa_loge("native baseline triangle producer rejected draw/state/resources");
   list->state = D12_LIST_ERROR;
   return false;
}

static void
d12_list_draw(struct d12_cmd_list *list, unsigned vertex_count,
              unsigned instance_count, unsigned first_vertex,
              unsigned first_instance, bool indexed, unsigned first_index,
              uint64_t indirect_args,
              const struct libagx_d3d12_indirect_patch_args *indirect_patches,
              unsigned indirect_patch_count,
              const struct libagx_d3d12_indirect_constant_args
                 *indirect_constant_patches,
              unsigned indirect_constant_patch_count,
              const struct d12_indirect_root_patch *indirect_root_patches,
              unsigned indirect_root_patch_count,
              const struct d12_indirect_vb_patch *indirect_vb_patches,
              unsigned indirect_vb_patch_count, uint64_t indirect_ibv,
              uint64_t *out_tables, uint32_t *out_table_count,
              uint64_t *out_vb_params)
{
   struct d12_device *device = list->device;
   struct d12_pipeline_state *pso = list->pso;

   if (list->state != D12_LIST_RECORDING)
      return;

   /* Native D3D12 ignores a draw recorded before any PSO is bound.  This is
    * not a command-list recording error; later complete draws still execute.
    */
   if (!d12_command_has_graphics_pipeline(pso != NULL,
                                          pso ? pso->is_compute : false))
      return;

   /* Reset/ClearState and bundle entry deliberately leave IA topology
    * undefined. Undefined draw output must not silently become triangles. */
   if (list->topology == MESA_PRIM_UNKNOWN)
      return;

   d12_list_track_pso_reads(list, pso);
   d12_list_track_bound_reads(list, false,
                              D12_WRITE_VIS_GRAPHICS_VERTEX);
   d12_list_track_bound_reads(list, false,
                              D12_WRITE_VIS_GRAPHICS_FRAGMENT);

   for (unsigned slot = 0;
        list->device->submit_read_set && slot < ARRAY_SIZE(list->vb);
        ++slot) {
      if (list->vb[slot].addr && list->vb[slot].size_B) {
         struct d12_resource *resource =
            d12_sparse_buffer_for_address(list->device,
                                          list->vb[slot].addr);
         if (resource)
            d12_list_track_read_resource(list, resource);
         else
            d12_list_track_read_range(list, list->vb[slot].addr,
                                      list->vb[slot].size_B);
      }
   }
   if (list->device->submit_read_set && indexed && list->ib.addr &&
       list->ib.size_B) {
      struct d12_resource *resource =
         d12_sparse_buffer_for_address(list->device, list->ib.addr);
      if (resource)
         d12_list_track_read_resource(list, resource);
      else
         d12_list_track_read_range(list, list->ib.addr, list->ib.size_B);
   }

   for (unsigned stream = 0;
        list->device->submit_read_set && stream < ARRAY_SIZE(list->so);
        ++stream) {
      const D3D12_STREAM_OUTPUT_BUFFER_VIEW *view = &list->so[stream];
      if (view->BufferLocation && view->SizeInBytes) {
         struct d12_resource *resource = d12_sparse_buffer_for_address(
            list->device, view->BufferLocation);
         if (resource)
            d12_list_track_read_resource(list, resource);
         else
            d12_list_track_read_range(list, view->BufferLocation,
                                      view->SizeInBytes);
      }
      if (view->BufferFilledSizeLocation) {
         struct d12_resource *resource = d12_sparse_buffer_for_address(
            list->device, view->BufferFilledSizeLocation);
         if (resource)
            d12_list_track_read_resource(list, resource);
         else
            d12_list_track_read_range(list,
                                      view->BufferFilledSizeLocation,
                                      sizeof(uint32_t));
      }
   }

   /* GPU-late root/VB/IB records may name allocations which did not exist in
    * command-list state at record time. Their argument buffers are tracked,
    * but their transitive resource set cannot be proven on the host. */
   if (indirect_root_patch_count || indirect_vb_patch_count || indirect_ibv)
      d12_list_mark_read_set_incomplete(list);

   if (list->active_pipeline_stats_heap && vertex_count && instance_count) {
      uint64_t vertices = (uint64_t)vertex_count * instance_count;
      uint64_t primitives;
      switch (list->topology) {
      case MESA_PRIM_POINTS:
         primitives = vertices;
         break;
      case MESA_PRIM_LINES:
         primitives = (vertex_count / 2u) * (uint64_t)instance_count;
         break;
      case MESA_PRIM_LINE_STRIP:
         primitives = vertex_count > 1
                         ? (uint64_t)(vertex_count - 1) * instance_count
                         : 0;
         break;
      case MESA_PRIM_TRIANGLE_STRIP:
         primitives = vertex_count > 2
                         ? (uint64_t)(vertex_count - 2) * instance_count
                         : 0;
         break;
      default:
         primitives = (vertex_count / 3u) * (uint64_t)instance_count;
         break;
      }

      list->pipeline_stats.IAVertices += vertices;
      list->pipeline_stats.IAPrimitives += primitives;
      list->pipeline_stats.VSInvocations += vertices;
      list->pipeline_stats.CInvocations += primitives;
      list->pipeline_stats.CPrimitives += primitives;
      if (pso->has_pixel_shader && primitives)
         list->pipeline_stats.PSInvocations++;
   }

   if (list->render_open) {
      struct d12_segment *open = &list->segments[list->segment_count - 1];
      const bool layout_matches =
         open->merge_unused_rts
            ? d12_segment_matches_unused_rt_layout(open, pso)
            : memcmp(&open->tib, &pso->tib, sizeof(open->tib)) == 0 &&
                 open->rt_write_mask == pso->rt_write_mask;

      if (!d12_segment_matches_bindings(list, open, pso->has_pixel_shader) ||
          !layout_matches)
         d12_list_close_cdm(list);
   }

   if (!list->render_open) {
      const unsigned rtv_count =
         pso->has_pixel_shader ? list->bound_rtv_count : 0;
      const struct d12_descriptor *rtvs =
         rtv_count ? list->bound_rtv : NULL;
      const struct d12_descriptor *dsv =
         list->has_bound_dsv ? &list->bound_dsv : NULL;
      if (!d12_list_begin_render(list, rtvs, rtv_count, dsv, pso,
                                 AGX_BG_LOAD, NULL, false))
         return;
   }

   struct d12_segment *seg = &list->segments[list->segment_count - 1];

   /* Scissor and depth-bias bases are per render command. End the current
    * pass before either 16-bit hardware array can run out of reserved
    * command-list storage. The next pass loads the attachments and keeps
    * application-visible rendering continuous. */
   const unsigned required_scissors = MAX2((unsigned)list->viewport_count, 1);
   if (required_scissors >
          D12_SCISSORS_PER_SEGMENT - seg->scissor_count ||
       seg->depth_bias_count >= D12_DEPTH_BIASES_PER_SEGMENT) {
      d12_list_close_cdm(list);
      const unsigned rtv_count =
         pso->has_pixel_shader ? list->bound_rtv_count : 0;
      const struct d12_descriptor *rtvs =
         rtv_count ? list->bound_rtv : NULL;
      const struct d12_descriptor *dsv =
         list->has_bound_dsv ? &list->bound_dsv : NULL;
      seg = d12_list_begin_render(list, rtvs, rtv_count, dsv, pso,
                                  AGX_BG_LOAD, NULL, false);
      if (!seg)
         return;
   }

   /* Root table, exactly as the dispatch path builds it -- from the
    * GRAPHICS bindings, which D3D12 keeps separate from the compute ones.
    */
   uint64_t root_gpu = 0, textures_gpu = 0, pbes_gpu = 0;
   uint64_t heap_gpu = 0, heap_textures_gpu = 0, heap_pbes_gpu = 0;
   uint64_t heap_samplers_gpu = 0;
   uint64_t tables_gpu = 0;
   uint32_t table_count = 0;
   if (!d12_list_upload_bindings(list, false, &root_gpu, &textures_gpu,
                                 &pbes_gpu, &heap_gpu, &heap_textures_gpu,
                                 &heap_pbes_gpu, &heap_samplers_gpu,
                                 indirect_root_patch_count != 0, &tables_gpu,
                                 &table_count))
      return;

   /* ExecuteIndirect argument and root data may have been produced by GPU
    * work earlier in this list. Patch unique snapshots in the pre-graphics
    * stream after their table addresses are known and before the render pass
    * which consumes them. */
   if (!d12_list_dispatch_indirect_patches(
          list, indirect_patches, indirect_patch_count,
          indirect_constant_patches, indirect_constant_patch_count,
          indirect_root_patches, indirect_root_patch_count, tables_gpu,
          table_count, true))
      return;
   if (out_tables)
      *out_tables = tables_gpu;
   if (out_table_count)
      *out_table_count = table_count;

   /* SM 6.8 draw parameters. agx_usc_uniform fills registers FROM memory.
    * The two D3D12 indirect structures both end in the exact pair this ABI
    * needs, so indirect draws can point into the application's buffer and
    * observe its contents at GPU execution time.
    */
   uint64_t draw_gpu;
   if (indirect_args) {
      draw_gpu = indirect_args + (indexed ? 12 : 8);
   } else {
      /* D3D robust null-index behaviour selects vertex zero. Applying a
       * BaseVertexLocation after substituting the zero index defeats that
       * guarantee (and native implementations explicitly ignore it here).
       */
      uint32_t draw_params[2] = {
         indexed && !list->ib.addr ? 0 : first_vertex,
         first_instance,
      };
      if (list->direct_draw_params_valid &&
          memcmp(list->direct_draw_params, draw_params, sizeof(draw_params)) ==
             0) {
         draw_gpu = list->direct_draw_params_gpu;
      } else {
         draw_gpu = d12_list_upload(list, draw_params, sizeof(draw_params), 8);
         if (draw_gpu) {
            memcpy(list->direct_draw_params, draw_params, sizeof(draw_params));
            list->direct_draw_params_gpu = draw_gpu;
            list->direct_draw_params_valid = true;
         }
      }
   }
   if (!draw_gpu)
      return;

   /* GEOMETRY. The compute half runs first, into the pre-graphics stream,
    * and hands back what the rasterization draw becomes. From here on the
    * "vertex shader" is the GS rasterization program, which reads the
    * emitted vertices rather than vertex attributes.
    */
   enum mesa_prim topology = list->topology;
   const bool api_restart =
      indexed && d12_strip_cut_enabled(list, pso, list->topology);
   struct d12_dispatch_uniforms poly_u = {
      .heap = heap_gpu,
      .heap_textures = heap_textures_gpu,
      .heap_pbes = heap_pbes_gpu,
      .heap_samplers = heap_samplers_gpu,
   };

   /*
    * VERTEX ATTRIBUTE FETCH has to be published BEFORE the pre-rasterisation
    * compute half runs, not after it.
    *
    * The software VS that a tessellation, geometry or mesh pipeline
    * substitutes for the hardware vertex stage is lowered by
    * agx_nir_lower_vbo (compile_sw_vs_stage), so it FETCHES ATTRIBUTES
    * THROUGH THIS UNIFORM. But d12_list_*_prerast emit their dispatches
    * internally, and vb_params used to be filled in below them -- so the
    * software VS was dispatched with vb_params == 0, and
    * d12_list_build_compute_usc pushes that uniform only when it is
    * non-zero. The uniform registers were therefore never written and the
    * shader fetched through whatever the previous dispatch had left in
    * them: a garbage pointer, different every run, belonging to none of
    * our allocations.
    *
    * That is the fault that cost this campaign several days. It reads as
    * `unit: UL1C, read: true, reason: Unmapped` at an address in a region
    * we never map (0xbb..., against allocations at 0x2ff...), and it is
    * why test_quad_tessellation_dxil wedges the GPU while our own g7
    * tessellation probe does not -- g7 derives its geometry from
    * SV_VertexID and so never fetches an attribute at all.
    */
   uint64_t vb_params_data_gpu = 0;
   uint64_t vb_params_gpu =
      d12_list_upload_vb_params(list, pso, &vb_params_data_gpu);
   if (vb_params_gpu == D12_VB_PARAMS_FAILED)
      return;
   if (!d12_list_dispatch_indirect_vb_patches(
          list, pso, indirect_vb_patches, indirect_vb_patch_count,
          vb_params_data_gpu))
      return;
   if (out_vb_params)
      *out_vb_params = vb_params_data_gpu;
   poly_u.vb_params = vb_params_gpu;

   const struct d12_gfx_stage *vstage = &pso->vs;
   const struct agx_unlinked_uvs_layout *uvs = &pso->uvs;
   struct agx_varyings_vs *varys = &pso->varyings;
   struct agx_draw draw = {0};
   bool draw_set = false;
   struct agx_draw gs_upstream = {0};
   const struct agx_draw *gs_input = NULL;
   bool gs_restart_unrolled = false;

   /* TESSELLATION. Same shape as geometry: the compute half runs first,
    * into the pre-graphics stream, and the DOMAIN shader becomes the
    * hardware vertex stage. Unlike geometry the draw itself is replaced
    * -- indexed and indirect, because the index count only exists after
    * the GPU prefix sum.
    */
   if (pso->has_mesh) {
      /* vertex_count/instance_count carry the DispatchMesh thread group
       * counts here -- a mesh draw has no vertices of its own.
       */
      if (!d12_list_mesh_prerast(list, pso, vertex_count, instance_count,
                                 first_vertex, root_gpu, textures_gpu,
                                 pbes_gpu, draw_gpu, &draw, &poly_u))
         return;

      vstage = &pso->ms_rast;
      uvs = &pso->ms_uvs;
      varys = &pso->ms_varyings;
      topology = MESA_PRIM_TRIANGLES;
      draw_set = true;
   } else if (pso->has_tess) {
      if (!d12_list_tess_prerast(list, pso, vertex_count, instance_count,
                                 indexed, first_index, root_gpu, textures_gpu,
                                 pbes_gpu, draw_gpu, &draw,
                                 &poly_u))
         return;

      vstage = &pso->tes;
      uvs = &pso->tes_uvs;
      varys = &pso->tes_varyings;
      topology = pso->tess_points  ? MESA_PRIM_POINTS
                 : pso->tess_mode == TESS_PRIMITIVE_ISOLINES
                    ? MESA_PRIM_LINES
                    : MESA_PRIM_TRIANGLES;
      draw_set = true;
   }

   if (pso->has_gs && !pso->has_tess && (indirect_args || api_restart)) {
      uint64_t input_args = indirect_args;

      /* Poly restart-unroll and indirect-GS setup consume a five-word
       * indexed indirect descriptor.  Upload the ordinary direct arguments so
       * both entry points share the exact same GPU-side setup. */
      if (!input_args) {
         const D3D12_DRAW_INDEXED_ARGUMENTS args = {
            .IndexCountPerInstance = vertex_count,
            .InstanceCount = instance_count,
            .StartIndexLocation = first_index,
            .BaseVertexLocation = (INT)first_vertex,
            .StartInstanceLocation = first_instance,
         };
         input_args = d12_list_upload(list, &args, sizeof(args), 4);
         if (!input_args)
            return;
      }

      if (indexed) {
         uint64_t index_buffer = list->ib.addr ? list->ib.addr
                                               : device->null_va;
         uint32_t index_range = list->ib.addr ? list->ib.size_B : 4;
         enum agx_index_size index_size = list->ib.addr
                                             ? list->ib.index_size
                                             : AGX_INDEX_SIZE_U16;
         gs_upstream = agx_draw_indexed_indirect(
            input_args, index_buffer, index_range, index_size, api_restart);
      } else {
         gs_upstream = agx_draw_indirect(input_args);
      }

      if (api_restart) {
         struct agx_draw restart_input = gs_upstream;
         if (!d12_list_unroll_gs_restart(list, pso, list->topology,
                                         &restart_input, indirect_ibv,
                                         &gs_upstream))
            return;
         gs_restart_unrolled = true;
      }

      gs_input = &gs_upstream;
   }

   if (pso->has_gs) {
      if (pso->has_tess)
         gs_input = &draw;

      if (!d12_list_gs_prerast(list, pso, vertex_count, instance_count,
                               indexed, first_index,
                               gs_input, indirect_ibv,
                               gs_restart_unrolled,
                               root_gpu, textures_gpu, pbes_gpu, draw_gpu,
                               &vertex_count, &instance_count, &draw,
                               &poly_u))
         return;

      vstage = &pso->gs_rast;
      uvs = &pso->gs_uvs;
      varys = &pso->gs_varyings;
      topology = pso->gs_info.mode;
      first_vertex = 0;
      /* Every geometry pipeline consumes Poly's output draw. This includes
       * non-indexed shapes: their counts and zero-based vertex IDs are no
       * longer the application's original draw parameters. */
      draw_set = true;
   }

   if (d12_debug()->dump_mesh) {
      mesa_logi("draw: has_mesh=%d mesh_params=0x%llx vstage=%s",
                pso->has_mesh, (unsigned long long)poly_u.mesh_params,
                vstage == &pso->ms_rast ? "ms_rast" : "other");
   }

   uint64_t blend_gpu;
   if (list->blend_upload_valid &&
       memcmp(list->blend_upload, list->blend_factor,
              sizeof(list->blend_factor)) == 0) {
      blend_gpu = list->blend_upload_gpu;
   } else {
      blend_gpu = d12_list_upload(list, list->blend_factor,
                                  sizeof(list->blend_factor), 16);
      if (blend_gpu) {
         memcpy(list->blend_upload, list->blend_factor,
                sizeof(list->blend_factor));
         list->blend_upload_gpu = blend_gpu;
         list->blend_upload_valid = true;
      }
   }
   if (!blend_gpu)
      return;

   D3D12_SHADER_VISIBILITY vertex_visibility =
      vstage == &pso->tes        ? D3D12_SHADER_VISIBILITY_DOMAIN
      : vstage == &pso->gs_rast ? D3D12_SHADER_VISIBILITY_GEOMETRY
      : vstage == &pso->ms_rast ? D3D12_SHADER_VISIBILITY_MESH
                                  : D3D12_SHADER_VISIBILITY_VERTEX;

   if (!d12_segment_bind_sampler_heap(
          list, seg, vstage->info.uses_sampler_heap ||
                     pso->fs.info.uses_sampler_heap))
      return;

   if (!d12_segment_reserve_scratch(list, seg, D12_SCRATCH_VS,
                                    &vstage->info) ||
       !d12_segment_reserve_scratch(list, seg, D12_SCRATCH_FS,
                                    &pso->fs.info))
      return;

   const bool cacheable_gfx_upload = vstage == &pso->vs && !pso->has_gs &&
                                     !pso->has_tess && !pso->has_mesh &&
                                     !pso->nr_vb_attribs;
   struct d12_gfx_upload_cache_entry *gfx_cache =
      cacheable_gfx_upload
         ? d12_gfx_upload_cache_lookup(list, pso, &seg->tib, root_gpu,
                                       textures_gpu, pbes_gpu, heap_gpu,
                                       heap_textures_gpu, heap_pbes_gpu,
                                       heap_samplers_gpu, draw_gpu, blend_gpu)
         : NULL;
   const bool gfx_cache_hit = gfx_cache && gfx_cache->valid;
   uint32_t vs_usc = gfx_cache_hit ? gfx_cache->vs_usc : 0;
   uint32_t fs_usc = gfx_cache_hit ? gfx_cache->fs_usc : 0;
   uint32_t varyings = gfx_cache_hit ? gfx_cache->varyings : 0;

   if (!gfx_cache_hit) {
      vs_usc = d12_gfx_stage_usc(
         list, vstage, false, &seg->tib, root_gpu, textures_gpu, pbes_gpu,
         heap_gpu, heap_textures_gpu, heap_pbes_gpu, heap_samplers_gpu,
         draw_gpu, 0, vertex_visibility,
         (pso->has_gs || pso->has_tess || pso->has_mesh || pso->nr_vb_attribs)
            ? &poly_u
            : NULL);
      fs_usc = d12_gfx_stage_usc(
         list, &pso->fs, true, &seg->tib, root_gpu, textures_gpu, pbes_gpu,
         heap_gpu, heap_textures_gpu, heap_pbes_gpu, heap_samplers_gpu,
         draw_gpu, blend_gpu, D3D12_SHADER_VISIBILITY_PIXEL, NULL);
      if (!vs_usc || !fs_usc)
         return;

      /* Coefficient-register bindings linking VS outputs to FS inputs. */
      unsigned bindings = pso->fs.linked.cf.nr_bindings;
      if (bindings) {
         size_t size =
            AGX_CF_BINDING_HEADER_LENGTH + bindings * AGX_CF_BINDING_LENGTH;
         uint32_t link_usc;
         uint8_t *cpu = d12_list_alloc_usc(list, size, &link_usc);
         if (!cpu)
            return;

         bool generate_prim_id = false;
         agx_link_varyings_vs_fs(cpu, varys, uvs->user_size, &pso->fs.linked.cf,
                                 0, 0, &generate_prim_id);
         varyings = link_usc;
      }

      if (gfx_cache) {
         *gfx_cache = (struct d12_gfx_upload_cache_entry){
            .valid = true,
            .pso = pso,
            .tib = seg->tib,
            .root = root_gpu,
            .textures = textures_gpu,
            .pbes = pbes_gpu,
            .heap = heap_gpu,
            .heap_textures = heap_textures_gpu,
            .heap_pbes = heap_pbes_gpu,
            .heap_samplers = heap_samplers_gpu,
            .draw = draw_gpu,
            .blend = blend_gpu,
            .vs_usc = vs_usc,
            .fs_usc = fs_usc,
            .varyings = varyings,
         };
      }
   }

   if (device->dev.chip == AGX_CHIP_G17P) {
      if (indexed || indirect_args || indirect_ibv || list->predicate_gpu ||
          indirect_patch_count || indirect_constant_patch_count ||
          indirect_root_patch_count || indirect_vb_patch_count ||
          first_instance || first_vertex || first_index || vertex_count != 3 ||
          instance_count != 1) {
         mesa_loge("native baseline supports only a direct first triangle");
         list->state = D12_LIST_ERROR;
         return;
      }
      d12_emit_g17p_triangle(list, seg, pso, vs_usc, fs_usc,
                             vertex_count, instance_count, first_vertex);
      return;
   }

   if (!d12_list_check_stream_space(list, D12_VDM_DRAW_BYTES))
      return;

   uint8_t *out = list->stream_cur;

   /* Primitive restart has two independent pieces of VDM state: the draw's
    * enable bit and a persistent 32-bit sentinel register. The sentinel is
    * selected by the PSO, not inferred from the bound index width (D3D12
    * deliberately permits 0xffff with a 32-bit buffer and vice versa). */
   bool restart = api_restart;
   bool vdm_restart = draw_set ? draw.restart : restart;
   if (vdm_restart) {
      /* Poly-generated GS index buffers always carry their natural all-ones
       * restart sentinel. They are independent of the application's strip-cut
       * PSO state (and may exist for a non-indexed D3D draw), so key this from
       * the final draw rather than the original Draw[Indexed] call. Leaving
       * the persistent VDM sentinel at its reset value made generated index 0
       * look like a restart and dropped the first stream-output vertex. */
      uint32_t restart_index = draw_set
         ? BITFIELD_MASK(8 * agx_index_size_to_B(draw.index_size))
         : d12_effective_strip_cut(list, pso) ==
                 D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF
              ? UINT16_MAX
              : UINT32_MAX;

      agx_push(out, VDM_STATE, cfg) {
         cfg.restart_index_present = true;
      }
      agx_push(out, VDM_STATE_RESTART_INDEX, cfg) {
         cfg.value = restart_index;
      }
   }

   /* Vertex shader state.  AGX3 uses the exact six-way
    * VertexProgramVariant selector; AGX2 Counts/UVS are never substituted for
    * its independently produced state-load and VsStateConfig fields. */
   switch (agx_chip_command_stream_generation(list->device->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      agx_push(out, VDM_STATE, cfg) {
         cfg.vertex_shader_word_0_present = true;
         cfg.vertex_shader_word_1_present = true;
         cfg.vertex_outputs_present = true;
         cfg.vertex_unknown_present = true;
      }
      agx_push_packed(out, vstage->counts, VDM_STATE_VERTEX_SHADER_WORD_0);
      agx_push(out, VDM_STATE_VERTEX_SHADER_WORD_1, cfg) {
         cfg.pipeline = vs_usc;
      }
      agx_push_packed(out, uvs->vdm, VDM_STATE_VERTEX_OUTPUTS);
      agx_push(out, VDM_STATE_VERTEX_UNKNOWN, cfg);
      /* Pad to a multiple of 8 bytes */
      memset(out, 0, 4);
      out += 4;
      break;

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      const struct d12_agx3_vdm_state_update_program_info *program =
         &vstage->agx3_vdm_state_update;
      uint64_t pipeline_va;
      const bool pipeline_va_valid =
         agx_vdm_state_update_resolve_pipeline_va(
            list->device->dev.shader_base, vs_usc, &pipeline_va);
      const bool complete_program =
         pipeline_va_valid &&
         program->source_layout ==
            D12_AGX3_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT &&
         seg->agx3_active_context_selector_valid;
      const bool state_indexed = draw_set ? draw.indexed : indexed;
      const bool state_indirect =
         draw_set ? agx_is_indirect(draw.b)
                  : indirect_ibv ? false
                                 : indirect_args || list->predicate_gpu;
      struct d12_agx3_vdm_state_update_draw_info info = {
         .source_layout = complete_program
               ? D12_AGX3_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT
               : D12_AGX3_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED,
         .draw_path = d12_agx3_vdm_state_update_classify_draw(
            state_indirect, state_indexed),
         /* The native renderer sources this bit from
          * RenderPipeline::isRasterizationEnabled. */
         .rasterization_enabled = pso->rasterization_enabled,
         /* D3D12 is first-provoking.  Triangle fans use AGX's dedicated
          * first-fan (vertex 1) coefficient layout. */
         .raw_provoking_selector =
            topology == MESA_PRIM_TRIANGLE_FAN ? 1 : 0,
         .template_info = program->template_info,
      };
      info.template_info.pipeline = pipeline_va;
      info.template_info.raw_active_context_selector =
         seg->agx3_raw_active_context_selector;

      struct d12_agx3_vdm_state_update encoded;
      if (!d12_agx3_encode_vdm_state_update_draw(
             list->device->dev.chip, &info, &encoded)) {
         mesa_loge("no complete AGX3 VertexProgramVariant state for %s",
                   agx_chip_name(list->device->dev.chip));
         list->state = D12_LIST_ERROR;
         return;
      }

      memcpy(out, encoded.bytes, encoded.length);
      out += encoded.length;
      break;
   }

   default:
      mesa_loge("unknown command-stream generation for %s",
                agx_chip_name(list->device->dev.chip));
      list->state = D12_LIST_ERROR;
      return;
   }

   /* Viewport, scissor and depth bias */
   {
      /* AGX's viewport transform does not provide the API's exact pixel
       * clip at fractional edges. Honeykrisp therefore intersects the
       * hardware scissor with the viewport, truncating each edge to the
       * integer pixel boundary before applying the API scissor.
       */
      struct d12_viewport_hw_state states[D12_VIEWPORT_STATE_MAX];
      const unsigned viewport_count = d12_viewport_build_hw_state(
         list->viewport_count, list->viewports, list->scissor_count,
         list->scissors, seg->width_px, seg->height_px, states);
      if (!viewport_count) {
         list->state = D12_LIST_ERROR;
         return;
      }

      /* REGION_CLIP below is a coarse tile bound. The hardware scissor array
       * carries exact bounds and depth clipping for every viewport.
       */
      unsigned scissor_index =
         d12_list_push_scissors(list, seg, viewport_count, states);
      if (scissor_index == ~0u)
         return;

      unsigned depth_bias_index = d12_list_push_depth_bias(list, seg, pso);
      if (depth_bias_index == ~0u)
         return;

      bool unchanged =
         seg->viewport_ppp_valid &&
         seg->viewport_count == viewport_count &&
         seg->viewport_scissor_index == scissor_index &&
         seg->viewport_depth_bias_index == depth_bias_index;
      for (unsigned i = 0; unchanged && i < viewport_count; ++i) {
         unchanged =
            memcmp(seg->viewport_clip[i], states[i].clip,
                   sizeof(states[i].clip)) == 0 &&
            memcmp(seg->viewport_transform[i], states[i].transform,
                   sizeof(states[i].transform)) == 0;
      }

      /* PPP state persists in a render stream. Emit this complete group only
       * when one of its inputs changes; fragment-only updates below do not
       * invalidate it. */
      if (!unchanged) {
         if (agx_chip_command_stream_generation(list->device->dev.chip) ==
             AGX_CS_GENERATION_HAL300) {
            if (!d12_emit_hal300_viewport_ppp(
                   list, seg, viewport_count, states, scissor_index,
                   depth_bias_index, &out)) {
               mesa_loge("no grounded HAL300 viewport/base emitter for %s",
                         agx_chip_name(list->device->dev.chip));
               list->state = D12_LIST_ERROR;
               return;
            }
            goto viewport_ppp_emitted;
         }

         struct AGX_PPP_HEADER present = {
            .depth_bias_scissor = true,
            .region_clip = true,
            .viewport = true,
            .viewport_count = viewport_count,
         };

         size_t size = agx_ppp_update_size(&present);
         uint64_t gpu = d12_list_upload_zero(list, size, AGX_PPP_HEADER_ALIGN);
         uint8_t *cpu = d12_list_data_cpu(list, gpu);
         if (!cpu)
            return;

         struct agx_ppp_update ppp = agx_new_ppp_update(
            (struct agx_ptr){.cpu = cpu, .gpu = gpu}, size, &present);

         agx_ppp_push(&ppp, DEPTH_BIAS_SCISSOR, cfg) {
            cfg.scissor = scissor_index;
            cfg.depth_bias = depth_bias_index;
         }

         for (unsigned i = 0; i < viewport_count; ++i) {
            agx_ppp_push(&ppp, REGION_CLIP, cfg) {
               cfg.enable = true;
               cfg.min_x = states[i].clip[0] / 32;
               cfg.min_y = states[i].clip[1] / 32;
               cfg.max_x =
                  DIV_ROUND_UP(MAX2(states[i].clip[2], 1), 32);
               cfg.max_y =
                  DIV_ROUND_UP(MAX2(states[i].clip[3], 1), 32);
            }
         }

         agx_ppp_push(&ppp, VIEWPORT_CONTROL, cfg);

         for (unsigned i = 0; i < viewport_count; ++i) {
            agx_ppp_push(&ppp, VIEWPORT, cfg) {
               cfg.translate_x = states[i].transform[0];
               cfg.translate_y = states[i].transform[1];
               cfg.translate_z = states[i].transform[2];
               cfg.scale_x = states[i].transform[3];
               cfg.scale_y = states[i].transform[4];
               cfg.scale_z = states[i].transform[5];
            }
         }

         if (!agx_ppp_legacy_payload_supported(list->device->dev.chip) ||
             !agx_ppp_fini(list->device->dev.chip, &out, &ppp)) {
            mesa_loge("no grounded legacy PPP payload/finalizer for %s",
                      agx_chip_name(list->device->dev.chip));
            list->state = D12_LIST_ERROR;
            return;
         }

      viewport_ppp_emitted:
         seg->viewport_ppp_valid = true;
         seg->viewport_count = (uint8_t)viewport_count;
         seg->viewport_scissor_index = (uint16_t)scissor_index;
         seg->viewport_depth_bias_index = (uint16_t)depth_bias_index;
         for (unsigned i = 0; i < viewport_count; ++i) {
            memcpy(seg->viewport_clip[i], states[i].clip,
                   sizeof(states[i].clip));
            memcpy(seg->viewport_transform[i], states[i].transform,
                   sizeof(states[i].transform));
         }
      }
   }

   /* Fragment state */
   {
      /* Depth and stencil only test against something that exists. A PSO
       * asking for a depth test with no depth attached would otherwise
       * compare against whatever the ZLS unit last had pointed at it.
       */
      const struct d12_depth_stencil_state *ds = &pso->ds;
      const bool has_ds = seg->ds != NULL;
      const bool z_test = has_ds && ds->depth_test;
      const bool z_read_only =
         seg->dsv_flags & D3D12_DSV_FLAG_READ_ONLY_DEPTH;
      const bool s_read_only =
         seg->dsv_flags & D3D12_DSV_FLAG_READ_ONLY_STENCIL;
      const bool z_write = has_ds && ds->depth_write && !z_read_only;
      const bool has_stencil = has_ds && seg->ds->stencil_bo != NULL;
      const bool s_test = has_stencil && ds->stencil_test;
      const float effective_depth_bias = pso->raster.dynamic_depth_bias
                                            ? list->depth_bias
                                            : pso->raster.depth_bias;
      const float effective_slope_bias =
         pso->raster.dynamic_depth_bias
            ? list->slope_scaled_depth_bias
            : pso->raster.slope_scaled_depth_bias;

      const bool depth_bias_enable =
         (effective_depth_bias != 0.0f || effective_slope_bias != 0.0f) &&
         d12_object_type(topology) == AGX_OBJECT_TYPE_TRIANGLE;
      const bool occlusion_counting = list->active_occlusion_queries != 0;
      struct d12_fragment_state_cache_entry *fragment_cache =
         cacheable_gfx_upload
            ? d12_fragment_state_cache_lookup(
                 list, pso, fs_usc, varyings, topology, seg->dsv_flags,
                 list->stencil_ref_front, list->stencil_ref_back, has_ds,
                 has_stencil, depth_bias_enable, occlusion_counting)
            : NULL;

      if (fragment_cache && fragment_cache->valid) {
         if (!d12_emit_ppp_state(list, &out, fragment_cache->gpu,
                                 fragment_cache->size))
            return;
      } else {

         /* Field order is fixed by AGX_PPP_HEADER and the pushes below must
          * follow it exactly, or agx_ppp_fini catches a size mismatch.
          *
          * Both faces are pushed. Measured on this hardware (M2, G14G): with
          * two-sided state off, back-facing primitives use the FRONT slot --
          * a deliberately back-wound triangle depth-tests correctly even with
          * the back half left unwritten. The back half becomes load-bearing
          * the moment two_sided_stencil is set below, which is exactly when
          * D3D12 gives the two faces different stencil ops, so it is
          * programmed unconditionally rather than left to that coupling.
          */
         struct AGX_PPP_HEADER present = {
            .fragment_control = true,
            .fragment_control_2 = true,
            .fragment_front_face = true,
            .fragment_front_face_2 = true,
            .fragment_front_stencil = true,
            .fragment_back_face = true,
            .fragment_back_face_2 = true,
            .fragment_back_stencil = true,
            .output_select = true,
            .varying_counts_32 = true,
            .varying_counts_16 = true,
            .cull = true,
            .fragment_shader = true,
            .occlusion_query = true,
            .output_size = true,
            .viewport_count = 1,
         };

         size_t size = agx_ppp_update_size(&present);
         uint64_t gpu = d12_list_upload_zero(list, size, AGX_PPP_HEADER_ALIGN);
         uint8_t *cpu = d12_list_data_cpu(list, gpu);
         if (!cpu)
            return;

         struct agx_ppp_update ppp = agx_new_ppp_update(
            (struct agx_ptr){.cpu = cpu, .gpu = gpu}, size, &present);

         agx_ppp_push(&ppp, FRAGMENT_CONTROL, cfg) {
            cfg.visibility_mode = occlusion_counting
                                     ? AGX_VISIBILITY_MODE_COUNTING
                                     : AGX_VISIBILITY_MODE_NONE;
            /* Always on: the scissor entry the draw indexes carries the depth
             * clip range as well as the XY bounds, so this cannot be switched
             * off just because the application set no scissor rect.
             */
            cfg.scissor_enable = true;
            cfg.depth_bias_enable = depth_bias_enable;
            cfg.stencil_test_enable = s_test;
            cfg.two_sided_stencil = s_test;
         }

         /* The linker's own fragment control (tag writes etc.). */
         agx_ppp_push_packed(&ppp, &pso->fs.linked.fragment_control,
                             FRAGMENT_CONTROL);

         /* Depth function. AGX_ZS_FUNC_NEVER is 0, so leaving this unset
          * rejects every fragment — geometry rasterizes and nothing is
          * written. With no depth attachment the test must be ALWAYS.
          */
         struct agx_fragment_face_packed face;
         agx_pack(&face, FRAGMENT_FACE, cfg) {
            cfg.line_width =
               agx_pack_line_width(pso->raster.smooth_line ? 2.0f : 1.0f);
            cfg.polygon_mode = pso->raster.polygon_mode;
            cfg.disable_depth_write = !z_write;
            cfg.depth_function = z_test ? ds->depth_func : AGX_ZS_FUNC_ALWAYS;
         }

         agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, face) {
            cfg.stencil_reference = list->stencil_ref_front;
         }

         struct agx_fragment_face_2_packed face_2;
         agx_pack(&face_2, FRAGMENT_FACE_2, cfg) {
            cfg.object_type = d12_object_type(topology);
         }
         agx_merge(face_2, pso->fs.frag_face, FRAGMENT_FACE_2);
         agx_ppp_push_packed(&ppp, &face_2, FRAGMENT_FACE_2);

         d12_ppp_push_stencil(&ppp, &ds->front, ds, s_test, s_read_only,
                              false);

         /* Back face: same depth state, its own stencil ops. */
         agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, face) {
            cfg.stencil_reference = list->stencil_ref_back;
         }
         agx_ppp_push_packed(&ppp, &face_2, FRAGMENT_FACE_2);
         d12_ppp_push_stencil(&ppp, &ds->back, ds, s_test, s_read_only,
                              true);

         /* Output select is the MERGE of the vertex side (which says the
          * position is written) and the fragment side. Pushing only the
          * fragment half rasterizes nothing.
          */
         struct agx_output_select_packed osel = uvs->osel;
         agx_ppp_push_merged_blobs(&ppp, AGX_OUTPUT_SELECT_LENGTH, &osel,
                                   &pso->fs.linked.osel);

         agx_ppp_push_packed(&ppp, &varys->counts_32, VARYING_COUNTS);
         agx_ppp_push_packed(&ppp, &varys->counts_16, VARYING_COUNTS);

         agx_ppp_push(&ppp, CULL, cfg) {
            cfg.front_face_ccw = pso->raster.front_face_ccw;
            /* Direct hardware TES needs the tessellator's declared winding
             * folded into culling, as in Honeykrisp. Once any geometry stage
             * follows, however, Poly presents an explicit list of fully ordered
             * output triangles; applying the tessellation XOR again reverses
             * them. This includes the XFB passthrough synthesized by this
             * driver. D3D12 has no selectable domain origin, so there is no
             * lower-left XOR. */
            if (pso->has_tess && !pso->has_gs)
               cfg.front_face_ccw ^= pso->tess_ccw;
            cfg.cull_front = pso->raster.cull_front;
            cfg.cull_back = pso->raster.cull_back;
            cfg.depth_clip = pso->raster.depth_clip;
         }

         agx_ppp_push_packed(&ppp, &pso->fs.fs_counts, FRAGMENT_SHADER_WORD_0);

         agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_1, cfg) {
            cfg.pipeline = fs_usc;
         }

         agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_2, cfg) {
            cfg.cf_bindings = varyings;
         }

         agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_3, cfg);

         agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY, cfg);

         agx_ppp_push(&ppp, OUTPUT_SIZE, cfg) {
            cfg.count = uvs->size;
         }

         if (!agx_ppp_legacy_payload_supported(list->device->dev.chip) ||
             !agx_ppp_fini(list->device->dev.chip, &out, &ppp)) {
            mesa_loge("no grounded legacy PPP payload/finalizer for %s",
                      agx_chip_name(list->device->dev.chip));
            list->state = D12_LIST_ERROR;
            return;
         }
         if (fragment_cache) {
            *fragment_cache = (struct d12_fragment_state_cache_entry){
               .valid = true,
               .pso = pso,
               .fs_usc = fs_usc,
               .varyings = varyings,
               .topology = topology,
               .dsv_flags = seg->dsv_flags,
               .stencil_ref_front = list->stencil_ref_front,
               .stencil_ref_back = list->stencil_ref_back,
               .has_ds = has_ds,
               .has_stencil = has_stencil,
               .depth_bias_enable = depth_bias_enable,
               .occlusion_counting = occlusion_counting,
               .gpu = gpu,
               .size = size,
            };
         }
      }
   }

   /* The draw. Tessellation and mesh already built an indexed indirect
    * one, so only the ordinary path constructs anything here.
    *
    * For an indexed draw `first_vertex` is D3D12's BaseVertexLocation --
    * added to every fetched index -- and is NOT the same thing as the
    * `start` of a non-indexed draw, which is a vertex offset. Passing it
    * as `start` would skip indices instead of biasing vertices.
   */
   if (!draw_set) {
      uint64_t index_buffer = list->ib.addr ? list->ib.addr
                                            : device->null_va;
      uint32_t index_range = list->ib.addr ? list->ib.size_B : 4;
      enum agx_index_size index_size = list->ib.addr
                                          ? list->ib.index_size
                                          : AGX_INDEX_SIZE_U16;

      if (list->predicate_gpu) {
         if (!indirect_args) {
            if (indexed) {
               const D3D12_DRAW_INDEXED_ARGUMENTS args = {
                  .IndexCountPerInstance = vertex_count,
                  .InstanceCount = instance_count,
                  .StartIndexLocation = first_index,
                  .BaseVertexLocation = (INT)first_vertex,
                  .StartInstanceLocation = first_instance,
               };
               indirect_args =
                  d12_list_upload(list, &args, sizeof(args), 4);
            } else {
               const D3D12_DRAW_ARGUMENTS args = {
                  .VertexCountPerInstance = vertex_count,
                  .InstanceCount = instance_count,
                  .StartVertexLocation = first_vertex,
                  .StartInstanceLocation = first_instance,
               };
               indirect_args =
                  d12_list_upload(list, &args, sizeof(args), 4);
            }
         }

         indirect_args = d12_list_apply_predication(
            list, indirect_args, indexed ? 5 : 4, true);
         if (!indirect_args)
            return;
      }

      if (indirect_args && indexed) {
         draw = agx_draw_indexed_indirect(
            indirect_args, index_buffer, index_range, index_size, restart);
      } else if (indirect_args) {
         draw = agx_draw_indirect(indirect_args);
      } else if (indexed) {
         draw = agx_draw_indexed(vertex_count, instance_count, first_index,
                                 first_vertex, first_instance, index_buffer,
                                 index_range, index_size, restart);
      } else {
         draw = (struct agx_draw){
            .b = agx_3d(vertex_count, instance_count, 1),
            .start = first_vertex,
            .start_instance = first_instance,
         };
      }
   }

   if (indexed && indirect_ibv && !pso->has_gs) {
      if (!indirect_args) {
         mesa_loge("ExecuteIndirect: indirect IBV has no terminal draw");
         list->state = D12_LIST_ERROR;
         return;
      }

      const uint64_t vdm_gpu = list->stream_bo->va->addr +
         (uint64_t)(out - d12_list_stream_map(list));
      const struct libagx_d3d12_indirect_indexed_draw_args args = {
         .out = vdm_gpu,
         .command = indirect_args,
         .view = indirect_ibv,
         .sparse_registry = list->sparse_registry_view_gpu,
         .null_address = device->null_va,
         .sparse_ro_offset = device->dev.sparse_ro_offset,
         .format_r16_uint = DXGI_FORMAT_R16_UINT,
         .format_r32_uint = DXGI_FORMAT_R32_UINT,
         .restart = restart,
         .topology = d12_agx_primitive(topology),
      };
      d12_list_dispatch_precomp_to(
         list, LIBAGX_D3D12_INDIRECT_INDEXED_DRAW, &args, sizeof(args),
         agx_1d(1), true);
      if (list->state != D12_LIST_RECORDING)
         return;

      /* Reserve the exact direct-indexed command width and initialize it to
       * a safe no-op. The pre-graphics kernel overwrites these words after
       * the application's IBV/draw records are execution-visible. */
      draw = agx_draw_indexed(0, 0, 0, 0, 0, device->null_va, 4,
                              AGX_INDEX_SIZE_U16, restart);
   }

   /* Ordinary graphics UAV writes are split by firmware pipe.  ALL-visible
    * bindings are conservatively present in both sets.  Software geometry,
    * tessellation and mesh stages additionally execute in the pre-graphics
    * CDM segment and therefore need a compute ownership record as well. */
   d12_segment_track_bound_uavs(
      list, seg, false, D12_WRITE_VIS_GRAPHICS_VERTEX, D12_WRITE_VERTEX);
   d12_segment_track_bound_uavs(
      list, seg, false, D12_WRITE_VIS_GRAPHICS_FRAGMENT,
      D12_WRITE_FRAGMENT);

   if (d12_indirect_root_has_uav(indirect_root_patches,
                                 indirect_root_patch_count)) {
      /* Shader visibility could place this root UAV in either graphics pipe.
       * Conservatively withhold completeness for both; software geometry,
       * tessellation and mesh may also consume it from the pre-graphics CDM
       * stream below. */
      d12_segment_mark_write_set_incomplete(seg, D12_WRITE_VERTEX);
      d12_segment_mark_write_set_incomplete(seg, D12_WRITE_FRAGMENT);
      if (list->pregfx_open) {
         struct d12_segment *pregfx = &list->segments[list->pregfx_seg];
         d12_segment_mark_write_set_incomplete(pregfx,
                                               D12_WRITE_COMPUTE);
      }
   }

   if (list->active_occlusion_queries) {
      d12_segment_track_write_range(
         seg, D12_WRITE_FRAGMENT, device->oclqry_bo->va->addr,
         device->oclqry_bo->size);
   }

   if (list->pregfx_open) {
      struct d12_segment *pregfx = &list->segments[list->pregfx_seg];
      d12_segment_track_bound_uavs(
         list, pregfx, false, D12_WRITE_VIS_GRAPHICS_VERTEX,
         D12_WRITE_COMPUTE);

      /* Poly stream output writes the data buffer, its filled-size counter,
       * and any active SO statistics result from the pre-graphics compute
       * stream. */
      for (unsigned stream = 0; stream < ARRAY_SIZE(list->so); stream++) {
         const D3D12_STREAM_OUTPUT_BUFFER_VIEW *view = &list->so[stream];
         if (view->BufferLocation && view->SizeInBytes) {
            struct d12_resource *resource = d12_sparse_buffer_for_address(
               list->device, view->BufferLocation);
            if (resource)
               d12_segment_track_write_resource(
                  pregfx, D12_WRITE_COMPUTE, resource);
            else
               d12_segment_track_write_range(
                  pregfx, D12_WRITE_COMPUTE, view->BufferLocation,
                  view->SizeInBytes);

            resource = d12_sparse_buffer_for_address(
               list->device, view->BufferFilledSizeLocation);
            if (resource)
               d12_segment_track_write_resource(
                  pregfx, D12_WRITE_COMPUTE, resource);
            else
               d12_segment_track_write_range(
                  pregfx, D12_WRITE_COMPUTE,
                  view->BufferFilledSizeLocation, sizeof(uint32_t));
         }

         if (list->active_so_stats_heap[stream]) {
            struct d12_query_heap *heap =
               list->active_so_stats_heap[stream];
            d12_segment_track_write_range(
               pregfx, D12_WRITE_COMPUTE,
               heap->bo->va->addr +
                  (uint64_t)list->active_so_stats_index[stream] *
                     heap->stride,
               heap->stride);
         }
      }
   }

   const enum agx_primitive agx_topology = d12_agx_primitive(topology);
   if (!agx_vdm_draw_supported(device->dev.chip, draw, agx_topology)) {
      mesa_loge(
         "Draw form is not grounded for the selected AGX command stream");
      list->state = D12_LIST_ERROR;
      return;
   }

   out = (uint8_t *)agx_vdm_draw((uint32_t *)out, device->dev.chip, draw,
                                 agx_topology);

   list->stream_cur = out;
   seg->has_draws = true;
}

static void STDMETHODCALLTYPE
d12_list_DispatchMesh(ID3D12GraphicsCommandList6 *iface, UINT x, UINT y,
                      UINT z)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (!x || !y || !z)
      return;

   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_DISPATCH_MESH, 0);
      if (cmd) {
         cmd->dispatch.x = x;
         cmd->dispatch.y = y;
         cmd->dispatch.z = z;
      }
      return;
   }

   /* A mesh draw has no vertices; the thread group counts ride in the
    * same three arguments the vertex path uses, and d12_list_draw routes
    * on pso->has_mesh.
    */
   if (!list->pso || !list->pso->has_mesh) {
      mesa_loge("DispatchMesh without a mesh pipeline state");
      list->state = D12_LIST_ERROR;
      return;
   }

   d12_list_draw(list, x, y, z, 0, false, 0, 0, NULL, 0, NULL, 0, NULL, 0,
                 NULL, 0, 0, NULL, NULL, NULL);
}

static void STDMETHODCALLTYPE
d12_list_DrawInstanced(ID3D12GraphicsCommandList6 *iface,
                       UINT vertex_count_per_instance, UINT instance_count,
                       UINT start_vertex, UINT start_instance)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (!vertex_count_per_instance || !instance_count)
      return;

   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd = d12_bundle_record(list, D12_BUNDLE_DRAW, 0);
      if (cmd) {
         cmd->draw.vertex_count = vertex_count_per_instance;
         cmd->draw.instance_count = instance_count;
         cmd->draw.first_vertex = start_vertex;
         cmd->draw.first_instance = start_instance;
      }
      return;
   }

   d12_list_draw(list, vertex_count_per_instance, instance_count,
                 start_vertex, start_instance, false, 0, 0, NULL, 0, NULL, 0,
                 NULL, 0, NULL, 0, 0, NULL, NULL, NULL);
}

static void STDMETHODCALLTYPE
d12_list_DrawIndexedInstanced(ID3D12GraphicsCommandList6 *iface,
                              UINT index_count_per_instance,
                              UINT instance_count, UINT start_index,
                              INT base_vertex, UINT start_instance)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (!index_count_per_instance || !instance_count)
      return;

   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_DRAW_INDEXED, 0);
      if (cmd) {
         cmd->draw.vertex_count = index_count_per_instance;
         cmd->draw.instance_count = instance_count;
         cmd->draw.first_index = start_index;
         cmd->draw.base_vertex = base_vertex;
         cmd->draw.first_instance = start_instance;
      }
      return;
   }

   /* Match the non-indexed path: without a PSO this draw is ignored before
    * any indexed-draw state is validated.
    */
   if (list->state != D12_LIST_RECORDING || !list->pso)
      return;

   d12_list_draw(list, index_count_per_instance, instance_count,
                 (unsigned)base_vertex, start_instance, true, start_index, 0,
                 NULL, 0, NULL, 0, NULL, 0, NULL, 0, 0, NULL, NULL, NULL);
}

static uint64_t
d12_list_set_root_constants(struct d12_cmd_list *list, bool compute,
                            UINT root_param_index, UINT num_values,
                            const void *data, UINT offset_in_values);
static unsigned
d12_list_set_root_descriptor(struct d12_cmd_list *list, bool compute,
                             UINT root_param_index,
                             D3D12_GPU_VIRTUAL_ADDRESS address,
                             D3D12_ROOT_PARAMETER_TYPE expected);
static unsigned
d12_list_root_descriptor_slot(struct d12_cmd_list *list, bool compute,
                              UINT root_param_index,
                              D3D12_ROOT_PARAMETER_TYPE expected);

static bool
d12_indirect_argument_layout(const D3D12_INDIRECT_ARGUMENT_DESC *argument,
                             size_t *size, bool *terminal)
{
   *terminal = false;
   switch (argument->Type) {
   case D3D12_INDIRECT_ARGUMENT_TYPE_DRAW:
      *size = sizeof(D3D12_DRAW_ARGUMENTS);
      *terminal = true;
      return true;
   case D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED:
      *size = sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
      *terminal = true;
      return true;
   case D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH:
      *size = sizeof(D3D12_DISPATCH_ARGUMENTS);
      *terminal = true;
      return true;
   case D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT:
      if (!argument->Constant.Num32BitValuesToSet)
         return false;
      *size = (size_t)argument->Constant.Num32BitValuesToSet *
              sizeof(uint32_t);
      return true;
   case D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW:
   case D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW:
   case D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW:
      *size = sizeof(uint64_t);
      return true;
   case D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW:
      *size = sizeof(D3D12_VERTEX_BUFFER_VIEW);
      return true;
   case D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW:
      *size = sizeof(D3D12_INDEX_BUFFER_VIEW);
      return true;
   default:
      return false;
   }
}

static bool
d12_indirect_argument_has_gpu_address(
   const D3D12_INDIRECT_ARGUMENT_DESC *argument)
{
   switch (argument->Type) {
   case D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW:
   case D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW:
   case D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW:
   case D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW:
   case D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW:
      return true;
   default:
      return false;
   }
}

static bool
d12_list_reset_indirect_state(
   struct d12_cmd_list *list, bool compute,
   const struct d12_command_signature *signature)
{
   const uint32_t zero[D12_MAX_ROOT_PARAMS] = {0};
   const UINT terminal_index = signature->desc.NumArgumentDescs - 1;

   for (UINT index = 0; index < terminal_index; ++index) {
      const D3D12_INDIRECT_ARGUMENT_DESC *argument =
         &signature->arguments[index];
      switch (argument->Type) {
      case D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT:
         if (!d12_list_set_root_constants(
                list, compute, argument->Constant.RootParameterIndex,
                argument->Constant.Num32BitValuesToSet, zero,
                argument->Constant.DestOffsetIn32BitValues))
            return false;
         break;
      case D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW:
         if (d12_list_set_root_descriptor(
                list, compute,
                argument->ConstantBufferView.RootParameterIndex, 0,
                D3D12_ROOT_PARAMETER_TYPE_CBV) == ~0u)
            return false;
         break;
      case D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW:
         if (d12_list_set_root_descriptor(
                list, compute,
                argument->ShaderResourceView.RootParameterIndex, 0,
                D3D12_ROOT_PARAMETER_TYPE_SRV) == ~0u)
            return false;
         break;
      case D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW:
         if (d12_list_set_root_descriptor(
                list, compute,
                argument->UnorderedAccessView.RootParameterIndex, 0,
                D3D12_ROOT_PARAMETER_TYPE_UAV) == ~0u)
            return false;
         break;
      case D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW: {
         UINT slot = argument->VertexBuffer.Slot;
         if (slot >= D12_MAX_VB_SLOTS) {
            list->state = D12_LIST_ERROR;
            return false;
         }
         memset(&list->vb[slot], 0, sizeof(list->vb[slot]));
         break;
      }
      case D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW:
         list->ib.addr = 0;
         list->ib.size_B = 0;
         list->ib.index_size = AGX_INDEX_SIZE_U16;
         break;
      default:
         break;
      }
   }

   return list->state == D12_LIST_RECORDING;
}

static void STDMETHODCALLTYPE
d12_list_ExecuteIndirect(ID3D12GraphicsCommandList6 *iface,
                         ID3D12CommandSignature *signature_iface,
                         UINT max_command_count,
                         ID3D12Resource *argument_buffer_iface,
                         UINT64 argument_buffer_offset,
                         ID3D12Resource *count_buffer_iface,
                         UINT64 count_buffer_offset)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->state != D12_LIST_RECORDING || !max_command_count)
      return;

   if (!signature_iface || !argument_buffer_iface) {
      mesa_loge("ExecuteIndirect: null signature or argument buffer");
      list->state = D12_LIST_ERROR;
      return;
   }

   struct d12_command_signature *signature =
      (struct d12_command_signature *)signature_iface;
   struct d12_resource *argument_buffer =
      d12_resource_from_iface(argument_buffer_iface);
   struct d12_resource *count_buffer = count_buffer_iface
      ? d12_resource_from_iface(count_buffer_iface) : NULL;

   if (signature->device != list->device ||
       argument_buffer->device != list->device || !argument_buffer->bo ||
       argument_buffer->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
       (argument_buffer_offset & 3)) {
      mesa_loge("ExecuteIndirect: incompatible signature or argument buffer");
      list->state = D12_LIST_ERROR;
      return;
   }

   if (count_buffer &&
       (count_buffer->device != list->device || !count_buffer->bo ||
        count_buffer->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
        (count_buffer_offset & 3) ||
        count_buffer_offset > count_buffer->desc.Width ||
        sizeof(uint32_t) > count_buffer->desc.Width - count_buffer_offset)) {
      mesa_loge("ExecuteIndirect: invalid count buffer");
      list->state = D12_LIST_ERROR;
      return;
   }

   /* The D3D12 indirect-drawing contract permits ExecuteIndirect in bundles
    * only for a single terminal operation and a CPU-specified count. Root or
    * IA mutations and a GPU count buffer require a direct/compute list. */
   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      if (count_buffer || signature->desc.NumArgumentDescs != 1) {
         mesa_loge("ExecuteIndirect: bundle signature changes state or "
                   "uses a count buffer");
         list->state = D12_LIST_ERROR;
         return;
      }

      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_EXECUTE_INDIRECT, 0);
      if (cmd) {
         cmd->execute_indirect.signature = signature_iface;
         cmd->execute_indirect.max_count = max_command_count;
         cmd->execute_indirect.args = argument_buffer_iface;
         cmd->execute_indirect.args_offset = argument_buffer_offset;
      }
      return;
   }

   UINT terminal_index = signature->desc.NumArgumentDescs - 1;
   D3D12_INDIRECT_ARGUMENT_TYPE type =
      signature->arguments[terminal_index].Type;
   const bool dispatch = type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
   const bool indexed = type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
   const size_t terminal_size =
      dispatch ? sizeof(D3D12_DISPATCH_ARGUMENTS)
      : indexed ? sizeof(D3D12_DRAW_INDEXED_ARGUMENTS)
                : sizeof(D3D12_DRAW_ARGUMENTS);
   if (type != D3D12_INDIRECT_ARGUMENT_TYPE_DRAW && !indexed && !dispatch) {
      mesa_loge("ExecuteIndirect: argument type %u unsupported", type);
      list->state = D12_LIST_ERROR;
      return;
   }

   if (!list->pso ||
       (dispatch && !list->pso->is_compute) ||
       (!dispatch && (list->pso->is_compute || list->pso->has_tess ||
                      list->pso->has_mesh))) {
      mesa_loge("ExecuteIndirect: tessellation and mesh graphics pipelines "
                "are unsupported");
      list->state = D12_LIST_ERROR;
      return;
   }

   size_t terminal_offset = 0;
   bool has_indirect_ibv = false;
   bool has_gpu_addresses = false;
   for (UINT i = 0; i < terminal_index; i++) {
      const D3D12_INDIRECT_ARGUMENT_DESC *arg = &signature->arguments[i];
      size_t argument_size;
      bool terminal;
      if (!d12_indirect_argument_layout(arg, &argument_size, &terminal) ||
          terminal || argument_size > SIZE_MAX - terminal_offset) {
         mesa_loge("ExecuteIndirect: argument type %u before the terminal "
                   "command is unsupported", arg->Type);
         list->state = D12_LIST_ERROR;
         return;
      }
      terminal_offset += argument_size;
      has_indirect_ibv |=
         arg->Type == D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW;
      has_gpu_addresses |= d12_indirect_argument_has_gpu_address(arg);
   }

   if (dispatch && has_indirect_ibv) {
      mesa_loge("ExecuteIndirect: compute signature contains IA state");
      list->state = D12_LIST_ERROR;
      return;
   }

   if (indexed && !has_indirect_ibv && !list->ib.addr) {
      mesa_loge("ExecuteIndirect: indexed draw has no index buffer");
      list->state = D12_LIST_ERROR;
      return;
   }

   if (terminal_size > SIZE_MAX - terminal_offset) {
      list->state = D12_LIST_ERROR;
      return;
   }
   const size_t argument_size = terminal_offset + terminal_size;
   const uint64_t stride = signature->desc.ByteStride;
   const uint64_t width = argument_buffer->desc.Width;
   if (stride < argument_size || (stride & 3) ||
       argument_buffer_offset > width ||
       (uint64_t)(max_command_count - 1) >
          (width - argument_buffer_offset) / stride ||
       argument_size > width - argument_buffer_offset -
                          (uint64_t)(max_command_count - 1) * stride) {
      mesa_loge("ExecuteIndirect: argument range is out of bounds");
      list->state = D12_LIST_ERROR;
      return;
   }

   if (terminal_index > D12_MAX_ROOT_PARAMS) {
      mesa_loge("ExecuteIndirect: too many root argument records");
      list->state = D12_LIST_ERROR;
      return;
   }

   d12_list_track_read_resource(list, argument_buffer);
   d12_list_track_read_resource(list, count_buffer);
   if (has_gpu_addresses) {
      if (!list->sparse_registry_view_gpu) {
         list->sparse_registry_view_gpu = d12_list_upload_zero(
            list, sizeof(struct d12_sparse_registry_view), 8);
         if (!list->sparse_registry_view_gpu)
            return;
      }
      list->uses_sparse_registry = true;
      d12_list_mark_read_set_incomplete(list);
   }

   const uint64_t argument_read_base =
      d12_sparse_read_address(list->device, argument_buffer->addr);
   const uint64_t count_read_base = count_buffer
      ? d12_sparse_read_address(list->device, count_buffer->addr)
      : 0;

   uint64_t previous_tables = 0;
   uint64_t previous_vb_params = 0;
   uint32_t previous_table_count = 0;

   for (UINT i = 0; i < max_command_count; i++) {
      uint64_t command_offset = argument_buffer_offset + (uint64_t)i * stride;
      size_t running_offset = 0;
      struct libagx_d3d12_indirect_patch_args
         patches[D12_MAX_ROOT_PARAMS + 1];
      struct libagx_d3d12_indirect_constant_args
         constant_patches[D12_MAX_ROOT_PARAMS];
      struct d12_indirect_root_patch root_patches[D12_MAX_ROOT_PARAMS];
      struct d12_indirect_vb_patch vb_patches[D12_MAX_ROOT_PARAMS];
      unsigned patch_count = 0;
      unsigned constant_patch_count = 0;
      unsigned root_patch_count = 0;
      unsigned vb_patch_count = 0;
      uint64_t indirect_ibv = 0;

      for (UINT j = 0; j < terminal_index; j++) {
         const D3D12_INDIRECT_ARGUMENT_DESC *arg = &signature->arguments[j];
         uint64_t source_offset = command_offset + running_offset;
         size_t size_B;
         bool ignored_terminal;
         if (!d12_indirect_argument_layout(arg, &size_B,
                                           &ignored_terminal)) {
            list->state = D12_LIST_ERROR;
            return;
         }

         if (arg->Type == D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT) {
            struct d12_root_signature *rs =
               dispatch ? list->compute_rs : list->graphics_rs;
            const struct d12_root_param *root_param =
               rs && arg->Constant.RootParameterIndex < rs->layout.num_params
                  ? &rs->layout.params[arg->Constant.RootParameterIndex]
                  : NULL;
            unsigned slot = root_param
               ? d12_binding_map_find_root_range(
                    &rs->map,
                    d12_root_param_range_base(
                       &rs->layout, arg->Constant.RootParameterIndex),
                    root_param->base_register)
               : ~0u;
            struct d12_descriptor *table =
               dispatch ? list->compute_root_table
                        : list->graphics_root_table;
            uint64_t previous =
               slot != ~0u && table[slot].kind == D12_DESC_CBV
                  ? table[slot].address
                  : 0;
            const uint32_t zero[D12_MAX_ROOT_PARAMS] = {0};
            uint64_t destination = d12_list_set_root_constants(
               list, dispatch, arg->Constant.RootParameterIndex,
               arg->Constant.Num32BitValuesToSet, zero,
               arg->Constant.DestOffsetIn32BitValues);
            if (!destination || !root_param || slot == ~0u ||
                list->state != D12_LIST_RECORDING)
               return;

            constant_patches[constant_patch_count++] =
               (struct libagx_d3d12_indirect_constant_args){
                  .out = destination,
                  .previous = previous,
                  .in = argument_read_base + source_offset,
                  .draw_count = count_buffer
                     ? count_read_base + count_buffer_offset : 0,
                  .command_index = i,
                  .total_size_B = (uint32_t)ALIGN_POT(
                     (size_t)root_param->num_constants * sizeof(uint32_t),
                     16),
                  .destination_offset_B =
                     arg->Constant.DestOffsetIn32BitValues * 4,
                  .size_B = (uint32_t)size_B,
               };
         } else if (arg->Type ==
                       D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW ||
                    arg->Type ==
                       D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW ||
                    arg->Type ==
                       D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW) {
            UINT root_parameter_index;
            D3D12_ROOT_PARAMETER_TYPE expected;
            bool publish_unbounded = false;
            bool read_only = false;

            switch (arg->Type) {
            case D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT_BUFFER_VIEW:
               root_parameter_index =
                  arg->ConstantBufferView.RootParameterIndex;
               expected = D3D12_ROOT_PARAMETER_TYPE_CBV;
               read_only = true;
               break;
            case D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW:
               root_parameter_index =
                  arg->ShaderResourceView.RootParameterIndex;
               expected = D3D12_ROOT_PARAMETER_TYPE_SRV;
               publish_unbounded = true;
               read_only = true;
               break;
            case D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW:
               root_parameter_index =
                  arg->UnorderedAccessView.RootParameterIndex;
               expected = D3D12_ROOT_PARAMETER_TYPE_UAV;
               publish_unbounded = true;
               break;
            default:
               list->state = D12_LIST_ERROR;
               return;
            }

            unsigned slot = d12_list_root_descriptor_slot(
               list, dispatch, root_parameter_index, expected);
            if (slot == ~0u || root_patch_count >= D12_MAX_ROOT_PARAMS)
               return;
            root_patches[root_patch_count++] =
               (struct d12_indirect_root_patch){
                  .in = argument_read_base + source_offset,
                  .draw_count = count_buffer
                     ? count_read_base + count_buffer_offset : 0,
                  .previous_tables = previous_tables,
                  .sparse_ro_offset = list->device->dev.sparse_ro_offset,
                  .previous_table_count = previous_table_count,
                  .command_index = i,
                  .slot = slot,
                  .publish_unbounded = publish_unbounded,
                  .read_only = read_only,
               };
         } else if (arg->Type ==
                    D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW) {
            if (arg->VertexBuffer.Slot >= D12_MAX_VB_SLOTS ||
                vb_patch_count >= D12_MAX_ROOT_PARAMS) {
               list->state = D12_LIST_ERROR;
               return;
            }
            vb_patches[vb_patch_count++] = (struct d12_indirect_vb_patch){
               .in = argument_read_base + source_offset,
               .draw_count = count_buffer
                  ? count_read_base + count_buffer_offset : 0,
               .previous_params = previous_vb_params,
               .command_index = i,
               .slot = arg->VertexBuffer.Slot,
            };
         } else if (arg->Type ==
                    D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW) {
            indirect_ibv = argument_read_base + source_offset;
         } else {
            list->state = D12_LIST_ERROR;
            return;
         }
         running_offset += size_B;
      }

      uint64_t args = argument_read_base + command_offset + terminal_offset;
      if (count_buffer) {
         uint64_t destination =
            d12_list_upload_zero(list, terminal_size, 4);
         if (!destination)
            return;

         patches[patch_count++] =
            (struct libagx_d3d12_indirect_patch_args){
               .out = destination,
               .in = argument_read_base + command_offset + terminal_offset,
               .draw_count = count_read_base + count_buffer_offset,
               .command_index = i,
               .size_B = terminal_size,
            };
         args = destination;
      }
      uint64_t current_tables = 0;
      uint64_t current_vb_params = 0;
      uint32_t current_table_count = 0;
      if (dispatch) {
         d12_list_dispatch_pso(list, 0, 0, 0, args, patches, patch_count,
                               constant_patches, constant_patch_count,
                               root_patches, root_patch_count, &current_tables,
                               &current_table_count);
      } else {
         d12_list_draw(list, 0, 0, 0, 0, indexed, 0, args, patches,
                       patch_count, constant_patches, constant_patch_count,
                       root_patches, root_patch_count, vb_patches,
                       vb_patch_count, indirect_ibv, &current_tables,
                       &current_table_count, &current_vb_params);
      }
      if (list->state != D12_LIST_RECORDING)
         return;
      previous_tables = current_tables;
      previous_table_count = current_table_count;
      previous_vb_params = current_vb_params;
   }

   /* ExecuteIndirect does not leave mutable signature state behind: IA views
    * and root views become null, while every root-constant range named by the
    * signature becomes zero. */
   (void)d12_list_reset_indirect_state(list, dispatch, signature);
}

/*
 * Record the bound vertex buffers. The view carries the STRIDE, which
 * D3D12 makes draw state rather than pipeline state -- that is why the
 * PSO compiles its fetch with dynamic strides and this is published per
 * draw rather than baked.
 */
static void STDMETHODCALLTYPE
d12_list_IASetVertexBuffers(ID3D12GraphicsCommandList6 *iface, UINT start_slot,
                            UINT num_views,
                            const D3D12_VERTEX_BUFFER_VIEW *views)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   /* Unlike IASetIndexBuffer(NULL), a NULL view-array pointer does not
    * unbind anything.  The call is ignored; zero BufferLocation entries in
    * a real array are how individual vertex-buffer slots are unbound. */
   if (!views)
      return;

   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd = d12_bundle_record(
         list, D12_BUNDLE_IA_VERTEX_BUFFERS, num_views * sizeof(*views));
      if (!cmd)
         return;
      cmd->vertex_buffers.start = start_slot;
      cmd->vertex_buffers.count = num_views;
      memcpy(d12_bundle_payload(cmd), views, num_views * sizeof(*views));
   }

   for (UINT i = 0; i < num_views; i++) {
      UINT slot = start_slot + i;
      if (slot >= D12_MAX_VB_SLOTS) {
         mesa_logw("IASetVertexBuffers: slot %u beyond %u", slot,
                   D12_MAX_VB_SLOTS - 1);
         continue;
      }

      list->vb[slot].addr =
         d12_sparse_read_address(list->device, views[i].BufferLocation);
      list->vb[slot].size_B = views[i].SizeInBytes;
      list->vb[slot].stride_B = views[i].StrideInBytes;
   }
}

/*
 * Record the bound index buffer. DXGI gives the index WIDTH as a format;
 * AGX wants it as an enum, and there are only two legal values -- D3D12
 * has no 8-bit index buffers.
 */
static void STDMETHODCALLTYPE
d12_list_IASetIndexBuffer(ID3D12GraphicsCommandList6 *iface,
                          const D3D12_INDEX_BUFFER_VIEW *view)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_IA_INDEX_BUFFER, 0);
      if (!cmd)
         return;
      cmd->index_buffer.bound = view != NULL;
      if (view)
         cmd->index_buffer.view = *view;
   }

   if (!view) {
      list->ib.addr = 0;
      list->ib.size_B = 0;
      list->ib.index_size = AGX_INDEX_SIZE_U16;
      return;
   }

   switch (view->Format) {
   case DXGI_FORMAT_R16_UINT:
      list->ib.index_size = AGX_INDEX_SIZE_U16;
      break;
   case DXGI_FORMAT_R32_UINT:
      list->ib.index_size = AGX_INDEX_SIZE_U32;
      break;
   case DXGI_FORMAT_UNKNOWN:
      /* Native drivers commonly accept UNKNOWN as the historical 16-bit
       * default. vkd3d's robustness coverage relies on that behaviour.
       */
      list->ib.index_size = AGX_INDEX_SIZE_U16;
      break;
   default:
      mesa_loge("IASetIndexBuffer: format %u is not a valid index format",
                (unsigned)view->Format);
      list->state = D12_LIST_ERROR;
      return;
   }

   list->ib.addr =
      d12_sparse_read_address(list->device, view->BufferLocation);
   list->ib.size_B = view->SizeInBytes;
}

static void STDMETHODCALLTYPE
d12_list_IASetPrimitiveTopology(ID3D12GraphicsCommandList6 *iface,
                                D3D12_PRIMITIVE_TOPOLOGY topology)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_IA_PRIMITIVE_TOPOLOGY, 0);
      if (!cmd)
         return;
      cmd->topology = topology;
      list->bundle_topology_set = true;
   }

   /* Geometry shaders take points and lines as input, so the topology
    * can no longer be assumed. Anything not listed here would be drawn
    * as the wrong primitive, so it is refused rather than approximated.
    */
   switch (topology) {
   case D3D_PRIMITIVE_TOPOLOGY_POINTLIST:
      list->topology = MESA_PRIM_POINTS;
      list->patch_control_points = 0;
      break;
   case D3D_PRIMITIVE_TOPOLOGY_LINELIST:
      list->topology = MESA_PRIM_LINES;
      break;
   case D3D_PRIMITIVE_TOPOLOGY_LINESTRIP:
      list->topology = MESA_PRIM_LINE_STRIP;
      break;
   case D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST:
      list->topology = MESA_PRIM_TRIANGLES;
      break;
   case D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP:
      list->topology = MESA_PRIM_TRIANGLE_STRIP;
      break;
   case D3D_PRIMITIVE_TOPOLOGY_TRIANGLEFAN:
      list->topology = MESA_PRIM_TRIANGLE_FAN;
      break;
   default:
      /* Patch lists encode their control-point count in the topology
       * value itself. That count is the only thing API state contributes
       * to tessellation, so it is recorded rather than refused.
       */
      if (topology >= D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST &&
          topology <= D3D_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST) {
         list->patch_control_points =
            1 + (topology - D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST);
         list->topology = MESA_PRIM_PATCHES;
         break;
      }
      mesa_loge("IASetPrimitiveTopology: topology %d is not supported",
                (int)topology);
      list->state = D12_LIST_ERROR;
      break;
   }
}


static void STDMETHODCALLTYPE
d12_list_RSSetViewports(ID3D12GraphicsCommandList6 *iface, UINT count,
                        const D3D12_VIEWPORT *viewports)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (count > ARRAY_SIZE(list->viewports) || (count && !viewports)) {
      list->state = D12_LIST_ERROR;
      return;
   }

   for (UINT i = 0; i < count; ++i) {
      list->viewports[i] = (struct d12_viewport_desc){
         .x = viewports[i].TopLeftX,
         .y = viewports[i].TopLeftY,
         .width = viewports[i].Width,
         .height = viewports[i].Height,
         .min_depth = viewports[i].MinDepth,
         .max_depth = viewports[i].MaxDepth,
      };
   }
   list->viewport_count = (uint8_t)count;
}

static void STDMETHODCALLTYPE
d12_list_RSSetScissorRects(ID3D12GraphicsCommandList6 *iface, UINT count,
                           const D3D12_RECT *rects)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (count > ARRAY_SIZE(list->scissors) || (count && !rects)) {
      list->state = D12_LIST_ERROR;
      return;
   }

   for (UINT i = 0; i < count; ++i) {
      list->scissors[i] = (struct d12_scissor_desc){
         .left = rects[i].left,
         .top = rects[i].top,
         .right = rects[i].right,
         .bottom = rects[i].bottom,
      };
   }
   list->scissor_count = (uint8_t)count;
}

static void STDMETHODCALLTYPE
d12_list_SetGraphicsRootSignature(ID3D12GraphicsCommandList6 *iface,
                                  ID3D12RootSignature *root_sig)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   list->graphics_rs = (struct d12_root_signature *)root_sig;
   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_SET_GRAPHICS_ROOT_SIGNATURE, 0);
      if (cmd)
         cmd->root_sig = root_sig;
   }
}

/*
 * Bind one root descriptor: a raw GPU address for a single shader
 * register, named indirectly through a root PARAMETER index.
 *
 * The root signature is what turns one into the other. Assuming the
 * parameter index IS the register happens to hold when an application
 * declares its parameters in register order and uses one class, and
 * silently binds the wrong resource the moment it does not.
 */
static unsigned
d12_list_root_descriptor_slot(struct d12_cmd_list *list, bool compute,
                              UINT root_param_index,
                              D3D12_ROOT_PARAMETER_TYPE expected)
{
   struct d12_root_signature *rs =
      compute ? list->compute_rs : list->graphics_rs;

   if (!rs) {
      mesa_loge("root descriptor bound with no root signature set");
      list->state = D12_LIST_ERROR;
      return ~0u;
   }

   if (root_param_index >= rs->layout.num_params) {
      mesa_loge("root parameter %u is past the signature's %u parameters",
                root_param_index, rs->layout.num_params);
      list->state = D12_LIST_ERROR;
      return ~0u;
   }

   const struct d12_root_param *rp = &rs->layout.params[root_param_index];
   if (rp->type != D3D12_ROOT_PARAMETER_TYPE_CBV &&
       rp->type != D3D12_ROOT_PARAMETER_TYPE_SRV &&
       rp->type != D3D12_ROOT_PARAMETER_TYPE_UAV) {
      mesa_loge("root parameter %u is not an address descriptor (type %u)",
                root_param_index, rp->type);
      list->state = D12_LIST_ERROR;
      return ~0u;
   }

   /* The three Set*Root*View entry points all write the same 64-bit root
    * descriptor slot. The signature, not the chosen setter spelling,
    * determines how the shader interprets that address. */
   (void)expected;

   unsigned root_range =
      d12_root_param_range_base(&rs->layout, root_param_index);
   unsigned slot = d12_binding_map_find_root_range(
      &rs->map, root_range, rp->base_register);
   if (slot == ~0u) {
      mesa_loge("root parameter %u names register %u in space %u, which the "
                "signature's binding map does not cover",
                root_param_index, rp->base_register, rp->space);
      list->state = D12_LIST_ERROR;
      return ~0u;
   }

   return slot;
}

static unsigned
d12_list_set_root_descriptor(struct d12_cmd_list *list, bool compute,
                             UINT root_param_index,
                             D3D12_GPU_VIRTUAL_ADDRESS address,
                             D3D12_ROOT_PARAMETER_TYPE expected)
{
   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_SET_ROOT_DESCRIPTOR, 0);
      if (!cmd)
         return ~0u;
      cmd->descriptor.compute = compute;
      cmd->descriptor.index = root_param_index;
      cmd->descriptor.address = address;
      cmd->descriptor.type = expected;
   }

   unsigned slot = d12_list_root_descriptor_slot(
      list, compute, root_param_index, expected);
   if (slot == ~0u)
      return slot;

   struct d12_root_signature *rs =
      compute ? list->compute_rs : list->graphics_rs;
   const struct d12_root_param *rp = &rs->layout.params[root_param_index];
   struct d12_descriptor *table =
      compute ? list->compute_root_table : list->graphics_root_table;
   const struct d12_descriptor **sources =
      compute ? list->compute_root_source : list->graphics_root_source;

   /* A root descriptor is a bare address: no resource behind it, so no
    * hardware texture descriptor can be built from one.
    */
   list->root_generation[compute]++;
   memset(&table[slot], 0, sizeof(table[slot]));
   sources[slot] = NULL;
   table[slot].address = address;
   table[slot].resource =
      d12_sparse_buffer_for_address(list->device, address);
   table[slot].kind = rp->type == D3D12_ROOT_PARAMETER_TYPE_SRV
                         ? D12_DESC_SRV
                         : (rp->type == D3D12_ROOT_PARAMETER_TYPE_UAV
                               ? D12_DESC_UAV
                               : D12_DESC_CBV);
   return slot;
}

/*
 * Bind a descriptor table: a contiguous run of descriptors in a
 * shader-visible heap, covering one or more register ranges.
 *
 * Our shader ABI reaches resources through a flat table of addresses, so
 * binding a table means copying each descriptor's address into the slot
 * for the register that range covers. Descriptor indexing in the shader is
 * therefore not supported yet -- the flattening happens here, at bind
 * time, rather than in the shader.
 */
static void
d12_list_set_root_table(struct d12_cmd_list *list, bool compute,
                        UINT root_param_index,
                        D3D12_GPU_DESCRIPTOR_HANDLE base)
{
   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_SET_ROOT_TABLE, 0);
      if (!cmd)
         return;
      cmd->table.compute = compute;
      cmd->table.index = root_param_index;
      cmd->table.base = base;
   }

   struct d12_root_signature *rs =
      compute ? list->compute_rs : list->graphics_rs;

   if (!rs) {
      mesa_loge("descriptor table bound with no root signature set");
      list->state = D12_LIST_ERROR;
      return;
   }

   if (root_param_index >= rs->layout.num_params) {
      mesa_loge("root parameter %u is past the signature's %u parameters",
                root_param_index, rs->layout.num_params);
      list->state = D12_LIST_ERROR;
      return;
   }

   const struct d12_root_param *rp = &rs->layout.params[root_param_index];
   if (rp->type != D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE) {
      mesa_loge("root parameter %u is not a descriptor table",
                root_param_index);
      list->state = D12_LIST_ERROR;
      return;
   }

   /* GPU handles are opaque CPU pointers into a heap's slot array.  Finite
    * ranges are snapshotted below.  An unbounded range instead resolves the
    * pointer through SetDescriptorHeaps to the coherent GPU BO and publishes
    * that descriptor-array address in one root slot. */
   const struct d12_descriptor *slots = (const struct d12_descriptor *)base.ptr;
   if (!slots) {
      mesa_loge("descriptor table bound with a null GPU handle");
      list->state = D12_LIST_ERROR;
      return;
   }

   /*
    * A GPU descriptor handle is an opaque token into the currently bound
    * shader-visible heap.  Do not dereference it until ownership and slot
    * alignment have been proved.  Apart from avoiding a CPU fault, ignoring
    * a malformed handle preserves the previous table binding, matching the
    * robust behaviour applications rely on for obviously corrupt table VAs.
    *
    * D3D12 root tables cannot mix sampler ranges with CBV/SRV/UAV ranges, so
    * the first range selects the one compatible bound heap.  Empty tables do
    * not consume a handle.
    */
   if (!rp->num_ranges)
      return;

   const struct d12_root_range *ranges =
      d12_root_param_ranges(&rs->layout, rp);
   unsigned table_class = d12_descriptor_range_class(ranges[0].type);
   if (table_class >= D12_RES_CLASS_COUNT) {
      mesa_loge("descriptor table has unknown range type %u",
                ranges[0].type);
      list->state = D12_LIST_ERROR;
      return;
   }

   bool table_is_sampler = table_class == D12_RES_SAMPLER;
   for (unsigned r = 1; r < rp->num_ranges; r++) {
      unsigned klass = d12_descriptor_range_class(ranges[r].type);
      if (klass >= D12_RES_CLASS_COUNT ||
          (klass == D12_RES_SAMPLER) != table_is_sampler) {
         mesa_loge("descriptor table mixes incompatible heap types");
         list->state = D12_LIST_ERROR;
         return;
      }
   }

   struct d12_descriptor_heap *bound_heap =
      table_is_sampler ? list->sampler_descriptor_heap
                       : list->resource_descriptor_heap;
   uintptr_t handle = (uintptr_t)slots;
   uintptr_t heap_start = bound_heap ? (uintptr_t)bound_heap->slots : 0;
   size_t heap_bytes =
      bound_heap ? (size_t)bound_heap->desc.NumDescriptors *
                      sizeof(struct d12_descriptor)
                 : 0;

   if (!bound_heap || handle < heap_start ||
       handle - heap_start >= heap_bytes ||
       (handle - heap_start) % sizeof(struct d12_descriptor)) {
      mesa_logw("ignoring descriptor-table handle outside the bound %s heap",
                table_is_sampler ? "sampler" : "resource");
      return;
   }

   size_t heap_index =
      (handle - heap_start) / sizeof(struct d12_descriptor);

   struct d12_descriptor *resource_table =
      compute ? list->compute_root_table : list->graphics_root_table;
   const struct d12_descriptor **resource_sources =
      compute ? list->compute_root_source : list->graphics_root_source;
   struct d12_descriptor *sampler_table =
      compute ? list->compute_sampler_table : list->graphics_sampler_table;
   list->root_generation[compute]++;

   unsigned cursor = 0;
   unsigned root_range_base =
      d12_root_param_range_base(&rs->layout, root_param_index);
   for (unsigned r = 0; r < rp->num_ranges; r++) {
      const struct d12_root_range *range = &ranges[r];
      unsigned root_range = root_range_base + r;

      unsigned offset = range->offset_in_descriptors;
      if (offset == D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND)
         offset = cursor;

      unsigned klass = d12_descriptor_range_class(range->type);
      if (klass >= D12_RES_CLASS_COUNT) {
         mesa_loge("descriptor range has unknown type %u", range->type);
         list->state = D12_LIST_ERROR;
         return;
      }

      if (range->num_descriptors == UINT32_MAX ||
          d12_binding_map_root_range_unbounded(
             &rs->map, root_range, range->base_register)) {
         if (klass == D12_RES_SAMPLER) {
            struct d12_descriptor_heap *heap =
               list->sampler_descriptor_heap;
            if (!heap || !heap->sampler_bo ||
                heap->desc.Type != D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) {
               mesa_loge("heap-indexed sampler table has no shader-visible "
                         "sampler heap bound");
               list->state = D12_LIST_ERROR;
               return;
            }

            if ((size_t)offset >= heap->desc.NumDescriptors - heap_index) {
               mesa_loge("unbounded sampler range offset %u is outside a "
                         "%u-slot heap at index %zu", offset,
                         heap->desc.NumDescriptors, heap_index);
               list->state = D12_LIST_ERROR;
               return;
            }

            unsigned slot = d12_binding_map_find_root_range(
               &rs->map, root_range, range->base_register);
            if (slot == ~0u ||
                !d12_binding_map_root_range_unbounded(
                   &rs->map, root_range, range->base_register)) {
               mesa_loge("heap-indexed sampler range has no binding-map "
                         "base");
               list->state = D12_LIST_ERROR;
               return;
            }

            size_t first = heap_index + offset;
            sampler_table[slot] = (struct d12_descriptor){
               /* This plane is an index, not an address. Index zero is a
                * valid first sampler and must remain distinguishable from
                * an absent plane through the root-signature map. */
               .address = first,
            };

            if (d12_debug()->validate_bindings) {
               size_t remaining = heap->desc.NumDescriptors - first;
               /* Only a genuinely unbounded range owns the tail of the heap.
                * A sized range ends, and demanding that every later sampler
                * slot be populated would fail a legal signature.  This
                * mirrors the resource-range validator below. */
               size_t scan_limit = range->num_descriptors == UINT32_MAX
                                      ? remaining
                                      : range->num_descriptors;
               size_t scan = MIN2(remaining, scan_limit);
               unsigned populated = 0;
               unsigned invalid_native = 0;
               for (size_t i = 0; i < scan; i++) {
                  populated += heap->slots[first + i].sampler_kind ==
                               D12_DESC_SAMPLER;
                  invalid_native +=
                     heap->slots[first + i].sampler_native_index >=
                     heap->sampler_unique_count;
               }

               mesa_logi("binding validation: unbounded sampler range "
                         "space=%u reg=%u root-slot=%u heap-index=%zu "
                         "remaining=%zu scan=%zu populated=%u "
                         "native=%u invalid-native=%u valid=%u",
                         range->space, range->base_register, slot, first,
                         remaining, scan, populated, heap->sampler_unique_count,
                         invalid_native,
                         populated == scan && invalid_native == 0);
               if (populated != scan || invalid_native) {
                  list->state = D12_LIST_ERROR;
                  return;
               }
            }
            continue;
         }

         struct d12_descriptor_heap *heap =
            list->resource_descriptor_heap;
         if (!heap || !heap->bo ||
             heap->desc.Type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) {
            mesa_loge("heap-indexed descriptor table has no shader-visible "
                      "resource heap bound");
            list->state = D12_LIST_ERROR;
            return;
         }

         if ((size_t)offset >= heap->desc.NumDescriptors - heap_index) {
            mesa_loge("unbounded descriptor range offset %u is outside a "
                      "%u-slot heap at index %zu", offset,
                      heap->desc.NumDescriptors, heap_index);
            list->state = D12_LIST_ERROR;
            return;
         }

         unsigned slot = d12_binding_map_find_root_range(
            &rs->map, root_range, range->base_register);
         if (slot == ~0u ||
             !d12_binding_map_root_range_unbounded(
                &rs->map, root_range, range->base_register)) {
            mesa_loge("heap-indexed descriptor range has no binding-map base");
            list->state = D12_LIST_ERROR;
            return;
         }

         resource_table[slot] = (struct d12_descriptor){
            .address = heap->bo->va->addr +
                       (heap_index + offset) *
                          sizeof(struct d12_descriptor),
         };
         resource_sources[slot] = NULL;

         if (d12_debug()->validate_bindings) {
            size_t first = heap_index + offset;
            size_t remaining =
               (size_t)heap->desc.NumDescriptors - first;
            size_t scan_limit = range->num_descriptors == UINT32_MAX
                                   ? 1024
                                   : range->num_descriptors;
            size_t scan = MIN2(remaining, scan_limit);
            unsigned populated = 0, mapped = 0, counter_ptrs = 0;
            unsigned invalid = 0, invalid_native_null = 0;
            uint8_t null_texture[32], null_pbe[32];
            const bool null_texture_valid =
               d12_set_null_texture_descriptor(list->device, null_texture);
            const bool null_pbe_valid =
               d12_set_null_pbe_descriptor(list->device, null_pbe);
            const uint8_t *native_textures =
               heap->texture_bo ? agx_bo_map(heap->texture_bo) : NULL;
            const uint8_t *native_pbes =
               heap->pbe_bo ? agx_bo_map(heap->pbe_bo) : NULL;

            for (size_t i = 0; i < scan; i++) {
               const struct d12_descriptor *d = &heap->slots[first + i];
               populated += d->kind != D12_DESC_NONE;
               mapped += d->address != 0;
               counter_ptrs += d->counter_address != 0;

               /* Heap-indexed shaders dereference the descriptor's first
                * word directly.  A zero address is therefore unsafe even
                * for a null descriptor: it must name the resident zero
                * region.  Also prove that resource-less slots have matching
                * null AGX texture/PBE descriptors in the parallel heaps.
                */
               if (!d->address)
                  invalid++;
               if (d->counter_address && (d->counter_address & 3))
                  invalid++;

               if (!d->resource &&
                   (!null_texture_valid || !null_pbe_valid ||
                    !native_textures || !native_pbes ||
                    memcmp(native_textures +
                              (first + i) * heap->texture_stride_B,
                           null_texture, heap->texture_stride_B) != 0 ||
                    memcmp(native_pbes + (first + i) * heap->pbe_stride_B,
                           null_pbe, heap->pbe_stride_B) != 0))
                  invalid_native_null++;

               if (d->resource && d->resource->bo) {
                  uint64_t bo_base = d->resource->bo->va->addr;
                  if (!d->address || d->address < bo_base ||
                      d->address - bo_base >= d->resource->bo->size)
                     invalid++;
               }
            }

            mesa_logi("binding validation: unbounded range class=%u "
                      "space=%u reg=%u root-slot=%u heap-index=%zu "
                      "gpu=0x%llx remaining=%zu scan=%zu populated=%u "
                      "mapped=%u counters=%u invalid=%u "
                      "invalid-native-null=%u valid=%u",
                      klass, range->space, range->base_register, slot, first,
                      (unsigned long long)resource_table[slot].address,
                      remaining, scan, populated, mapped, counter_ptrs, invalid,
                      invalid_native_null,
                      invalid == 0 && invalid_native_null == 0);
            if (invalid || invalid_native_null) {
               list->state = D12_LIST_ERROR;
               return;
            }
         }
         continue;
      }

      for (unsigned i = 0; i < range->num_descriptors; i++) {
         unsigned slot = d12_binding_map_find_root_range(
            &rs->map, root_range, range->base_register + i);
         if (slot == ~0u) {
            mesa_loge("descriptor range reaches a register the signature's "
                      "binding map does not cover");
            list->state = D12_LIST_ERROR;
            return;
         }

         struct d12_descriptor *table =
            klass == D12_RES_SAMPLER ? sampler_table : resource_table;
         size_t source_index = heap_index + (size_t)offset + i;

         if (source_index < bound_heap->desc.NumDescriptors) {
            table[slot] = bound_heap->slots[source_index];
            if (klass != D12_RES_SAMPLER)
               resource_sources[slot] = &bound_heap->slots[source_index];
         } else {
            /* A finite root range may legally extend beyond the heap even
             * when the shader only addresses its in-bounds prefix.  Keep the
             * CPU snapshot bounded and represent the inaccessible suffix as
             * null descriptors. */
            memset(&table[slot], 0, sizeof(table[slot]));
            if (klass != D12_RES_SAMPLER)
               resource_sources[slot] = NULL;
         }
      }

      cursor = offset + range->num_descriptors;
   }
}

static void STDMETHODCALLTYPE
d12_list_SetGraphicsRootDescriptorTable(ID3D12GraphicsCommandList6 *iface,
                                        UINT root_param_index,
                                        D3D12_GPU_DESCRIPTOR_HANDLE base)
{
   d12_list_set_root_table(d12_cmd_list_from_iface(iface), false,
                           root_param_index, base);
}

static void STDMETHODCALLTYPE
d12_list_SetComputeRootDescriptorTable(ID3D12GraphicsCommandList6 *iface,
                                       UINT root_param_index,
                                       D3D12_GPU_DESCRIPTOR_HANDLE base)
{
   d12_list_set_root_table(d12_cmd_list_from_iface(iface), true,
                           root_param_index, base);
}

static void STDMETHODCALLTYPE
d12_list_SetGraphicsRootShaderResourceView(ID3D12GraphicsCommandList6 *iface,
                                           UINT index,
                                           D3D12_GPU_VIRTUAL_ADDRESS address)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   address = d12_sparse_read_address(list->device, address);
   d12_list_set_root_descriptor(list, false, index, address,
                                D3D12_ROOT_PARAMETER_TYPE_SRV);
}

static void STDMETHODCALLTYPE
d12_list_SetComputeRootShaderResourceView(ID3D12GraphicsCommandList6 *iface,
                                          UINT index,
                                          D3D12_GPU_VIRTUAL_ADDRESS address)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   address = d12_sparse_read_address(list->device, address);
   d12_list_set_root_descriptor(list, true, index, address,
                                D3D12_ROOT_PARAMETER_TYPE_SRV);
}

/*
 * ROOT CONSTANTS. The application hands over N 32-bit values inline; the
 * shader sees them as a constant buffer at the parameter's register. Our
 * shader ABI reaches a CBV through an address, so the values are uploaded
 * into the command list's data area and that address goes in the slot --
 * the same slot a root CBV would use, which is why the binding map
 * already assigns one (d12_root_layout_build_map).
 *
 * Uploading per call rather than accumulating means a later partial write
 * has to preserve the earlier values, so the current contents are copied
 * forward before the new ones land on top.
 */
static uint64_t
d12_list_set_root_constants(struct d12_cmd_list *list, bool compute,
                            UINT root_param_index, UINT num_values,
                            const void *data, UINT offset_in_values)
{
   struct d12_root_signature *rs =
      compute ? list->compute_rs : list->graphics_rs;

   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd = d12_bundle_record(
         list, D12_BUNDLE_SET_ROOT_CONSTANTS,
         (size_t)num_values * sizeof(uint32_t));
      if (!cmd)
         return 0;
      cmd->constants.compute = compute;
      cmd->constants.index = root_param_index;
      cmd->constants.count = num_values;
      cmd->constants.offset = offset_in_values;
      if (data)
         memcpy(d12_bundle_payload(cmd), data,
                (size_t)num_values * sizeof(uint32_t));
   }

   if (!rs) {
      mesa_loge("root constants bound with no root signature set");
      list->state = D12_LIST_ERROR;
      return 0;
   }

   if (root_param_index >= rs->layout.num_params) {
      mesa_loge("root parameter %u is past the signature's %u parameters",
                root_param_index, rs->layout.num_params);
      list->state = D12_LIST_ERROR;
      return 0;
   }

   const struct d12_root_param *rp = &rs->layout.params[root_param_index];
   if (rp->type != D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS) {
      mesa_loge("root parameter %u is type %u, bound as root constants",
                root_param_index, rp->type);
      list->state = D12_LIST_ERROR;
      return 0;
   }

   unsigned total = MAX2(rp->num_constants, offset_in_values + num_values);
   if (!total || !data)
      return 0;

   unsigned root_range =
      d12_root_param_range_base(&rs->layout, root_param_index);
   unsigned slot = d12_binding_map_find_root_range(
      &rs->map, root_range, rp->base_register);
   if (slot == ~0u) {
      mesa_loge("root constants at register %u space %u are not in the "
                "signature's binding map",
                rp->base_register, rp->space);
      list->state = D12_LIST_ERROR;
      return 0;
   }

   struct d12_descriptor *table =
      compute ? list->compute_root_table : list->graphics_root_table;
   const struct d12_descriptor **sources =
      compute ? list->compute_root_source : list->graphics_root_source;

   /* A constant buffer is read in 16-byte units, so round up rather than
    * leave the tail of the last vector reading past the allocation. */
   size_t size = ALIGN_POT((size_t)total * 4, 16);

   uint64_t gpu = d12_list_upload_zero(list, size, 16);
   if (!gpu) {
      list->state = D12_LIST_ERROR;
      return 0;
   }
   uint32_t *staging = (uint32_t *)d12_list_data_cpu(list, gpu);

   /* Preserve what a previous call to this same parameter wrote. */
   if (table[slot].kind == D12_DESC_CBV && table[slot].address) {
      const void *old = d12_list_data_cpu(list, table[slot].address);
      if (old)
         memcpy(staging, old, MIN2(size, (size_t)rp->num_constants * 4));
   }

   memcpy(staging + offset_in_values, data, (size_t)num_values * 4);

   memset(&table[slot], 0, sizeof(table[slot]));
   sources[slot] = NULL;
   table[slot].address = gpu;
   table[slot].kind = D12_DESC_CBV;
   list->root_generation[compute]++;
   return gpu;
}

static void STDMETHODCALLTYPE
d12_list_SetGraphicsRoot32BitConstants(ID3D12GraphicsCommandList6 *iface,
                                       UINT index, UINT num, const void *data,
                                       UINT offset)
{
   d12_list_set_root_constants(d12_cmd_list_from_iface(iface), false, index,
                               num, data, offset);
}

static void STDMETHODCALLTYPE
d12_list_SetComputeRoot32BitConstants(ID3D12GraphicsCommandList6 *iface,
                                      UINT index, UINT num, const void *data,
                                      UINT offset)
{
   d12_list_set_root_constants(d12_cmd_list_from_iface(iface), true, index,
                               num, data, offset);
}

static void STDMETHODCALLTYPE
d12_list_SetGraphicsRoot32BitConstant(ID3D12GraphicsCommandList6 *iface,
                                      UINT index, UINT value, UINT offset)
{
   d12_list_set_root_constants(d12_cmd_list_from_iface(iface), false, index, 1,
                               &value, offset);
}

static void STDMETHODCALLTYPE
d12_list_SetComputeRoot32BitConstant(ID3D12GraphicsCommandList6 *iface,
                                     UINT index, UINT value, UINT offset)
{
   d12_list_set_root_constants(d12_cmd_list_from_iface(iface), true, index, 1,
                               &value, offset);
}

static void STDMETHODCALLTYPE
d12_list_SetGraphicsRootConstantBufferView(ID3D12GraphicsCommandList6 *iface,
                                           UINT index,
                                           D3D12_GPU_VIRTUAL_ADDRESS address)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   address = d12_sparse_read_address(list->device, address);
   d12_list_set_root_descriptor(list, false, index, address,
                                D3D12_ROOT_PARAMETER_TYPE_CBV);
}

static void STDMETHODCALLTYPE
d12_list_SetComputeRootConstantBufferView(ID3D12GraphicsCommandList6 *iface,
                                          UINT index,
                                          D3D12_GPU_VIRTUAL_ADDRESS address)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   address = d12_sparse_read_address(list->device, address);
   d12_list_set_root_descriptor(list, true, index, address,
                                D3D12_ROOT_PARAMETER_TYPE_CBV);
}

static void STDMETHODCALLTYPE
d12_list_SetGraphicsRootUnorderedAccessView(ID3D12GraphicsCommandList6 *iface,
                                            UINT index,
                                            D3D12_GPU_VIRTUAL_ADDRESS address)
{
   d12_list_set_root_descriptor(d12_cmd_list_from_iface(iface), false, index,
                                address, D3D12_ROOT_PARAMETER_TYPE_UAV);
}

static void STDMETHODCALLTYPE
d12_list_SetDescriptorHeaps(ID3D12GraphicsCommandList6 *iface, UINT count,
                            ID3D12DescriptorHeap *const *heaps)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (count > 2 || (count && !heaps)) {
      mesa_loge("SetDescriptorHeaps: invalid heap array");
      list->state = D12_LIST_ERROR;
      return;
   }

   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_SET_DESCRIPTOR_HEAPS, 0);
      if (!cmd)
         return;
      cmd->heaps.count = count;
      for (UINT i = 0; i < count; i++)
         cmd->heaps.heaps[i] = heaps[i];
   }

   for (UINT i = 0; i < count; i++) {
      if (!heaps[i]) {
         mesa_loge("SetDescriptorHeaps: null heap");
         list->state = D12_LIST_ERROR;
         return;
      }

      struct d12_descriptor_heap *heap =
         (struct d12_descriptor_heap *)heaps[i];
      if (!(heap->desc.Flags & D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) ||
          (heap->desc.Type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV &&
           heap->desc.Type != D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER)) {
         mesa_loge("SetDescriptorHeaps: heap %u is not shader-visible", i);
         list->state = D12_LIST_ERROR;
         return;
      }

      if (heap->desc.Type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
         list->resource_descriptor_heap = heap;
      else {
         /* The DRM command, not an individual USC program, owns the
          * bindless sampler heap. End any existing command before changing
          * it so earlier dispatches/draws retain their recorded heap. */
         if (list->sampler_descriptor_heap != heap &&
             (list->cdm_open || list->render_open))
            d12_list_close_cdm(list);
         list->sampler_descriptor_heap = heap;
      }
   }
}

static void STDMETHODCALLTYPE
d12_list_SetPipelineState(ID3D12GraphicsCommandList6 *iface,
                          ID3D12PipelineState *pso)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   list->pso = pso ? d12_pipeline_state_from_iface(pso) : NULL;
   if (list->pso && !list->pso->is_compute) {
      struct d12_dynamic_state_values dynamic = d12_list_dynamic_state(list);
      dynamic.depth_bias = list->pso->raster.depth_bias;
      dynamic.depth_bias_clamp = list->pso->raster.depth_bias_clamp;
      dynamic.slope_scaled_depth_bias =
         list->pso->raster.slope_scaled_depth_bias;
      dynamic.strip_cut = (uint32_t)list->pso->strip_cut;
      d12_list_set_dynamic_state(list, &dynamic);
   }

   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_SET_PIPELINE_STATE, 0);
      if (cmd)
         cmd->pso = pso;
   }
}

static void STDMETHODCALLTYPE
d12_list_ExecuteBundle(ID3D12GraphicsCommandList6 *iface,
                       ID3D12GraphicsCommandList *bundle_iface)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   struct d12_cmd_list *bundle = d12_cmd_list_from_iface(bundle_iface);

   if (list->state != D12_LIST_RECORDING ||
       list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE)
      return;

   if (!bundle || bundle->type != D3D12_COMMAND_LIST_TYPE_BUNDLE ||
       bundle->state != D12_LIST_CLOSED || bundle->had_error) {
      mesa_loge("ExecuteBundle: invalid or unclosed bundle");
      list->state = D12_LIST_ERROR;
      return;
   }

   /* Replaying the bundle's calls against this list applies every piece
    * of state it set, in order, and leaves it set afterwards: D3D12's
    * inheritance back into the caller, including from a bundle with no
    * draw. */
   d12_list_replay_bundle(list, bundle);
}

/* Clear API state without rewinding the command stream. Commands recorded
 * before ClearState remain part of the list; only the state inherited by
 * later commands returns to defaults, with the optional PSO installed.
 */
static void STDMETHODCALLTYPE
d12_list_ClearState(ID3D12GraphicsCommandList6 *iface,
                    ID3D12PipelineState *initial_state)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->state != D12_LIST_RECORDING)
      return;

   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE ||
       list->type == D3D12_COMMAND_LIST_TYPE_COPY) {
      mesa_loge("ClearState is invalid for command list type %u",
                (unsigned)list->type);
      list->state = D12_LIST_ERROR;
      list->record_error = E_FAIL;
      return;
   }

   /* Render-target bindings are part of the cleared state. Finish an open
    * hardware segment so later work cannot inherit targets or a bindless
    * sampler heap from the pre-clear command stream.
    */
   if (list->cdm_open || list->render_open)
      d12_list_close_cdm(list);

   if (list->state != D12_LIST_RECORDING)
      return;

   list->pso = initial_state ? d12_pipeline_state_from_iface(initial_state)
                             : NULL;
   d12_list_reset_api_state(list);
}

static void STDMETHODCALLTYPE
d12_list_SetComputeRootSignature(ID3D12GraphicsCommandList6 *iface,
                                 ID3D12RootSignature *root_sig)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   list->compute_rs = (struct d12_root_signature *)root_sig;
   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_SET_COMPUTE_ROOT_SIGNATURE, 0);
      if (cmd)
         cmd->root_sig = root_sig;
   }
}

static void STDMETHODCALLTYPE
d12_list_SetComputeRootUnorderedAccessView(ID3D12GraphicsCommandList6 *iface,
                                           UINT root_param_index,
                                           D3D12_GPU_VIRTUAL_ADDRESS address)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   /* A root UAV is a raw GPU virtual address, which is exactly what the
    * shader's binding ABI wants. Which shader register it feeds comes from
    * the root signature, not from the parameter index.
    */
   d12_list_set_root_descriptor(list, true, root_param_index, address,
                                D3D12_ROOT_PARAMETER_TYPE_UAV);
}

static void STDMETHODCALLTYPE
d12_list_Dispatch(ID3D12GraphicsCommandList6 *iface, UINT x, UINT y, UINT z)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (!x || !y || !z)
      return;

   /* Native D3D12 implementations tolerate an oversized X dimension, but
    * discard dispatches whose Y or Z dimension exceeds the API limit.  Do
    * that before encoding an AGX launch: otherwise the hardware executes a
    * command D3D12 defines as out of range, potentially doing a very large
    * amount of unintended work. */
   if (y > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION ||
       z > D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION)
      return;

   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_DISPATCH, 0);
      if (cmd) {
         cmd->dispatch.x = x;
         cmd->dispatch.y = y;
         cmd->dispatch.z = z;
      }
      return;
   }

   d12_list_dispatch_pso(list, x, y, z, 0, NULL, 0, NULL, 0, NULL, 0, NULL,
                         NULL);
}

static HRESULT STDMETHODCALLTYPE
d12_list_QueryInterface(ID3D12GraphicsCommandList6 *iface, REFIID riid,
                        void **out)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D12Object) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceChild) ||
       d12_guid_equal(riid, &IID_ID3D12CommandList) ||
       d12_guid_equal(riid, &IID_ID3D12GraphicsCommandList) ||
       d12_guid_equal(riid, &IID_ID3D12GraphicsCommandList1) ||
       d12_guid_equal(riid, &IID_ID3D12GraphicsCommandList2) ||
       d12_guid_equal(riid, &IID_ID3D12GraphicsCommandList3) ||
       d12_guid_equal(riid, &IID_ID3D12GraphicsCommandList4) ||
       d12_guid_equal(riid, &IID_ID3D12GraphicsCommandList5) ||
       d12_guid_equal(riid, &IID_ID3D12GraphicsCommandList6) ||
       d12_guid_equal(riid, &IID_ID3D12GraphicsCommandList7) ||
       d12_guid_equal(riid, &IID_ID3D12GraphicsCommandList8) ||
       d12_guid_equal(riid, &IID_ID3D12GraphicsCommandList9)) {
      /* The common vtable is ABI-complete through list 9. */
      p_atomic_inc(&list->refcount);
      *out = iface;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&list->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12GraphicsCommandList::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_list_AddRef(ID3D12GraphicsCommandList6 *iface)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   return p_atomic_inc_return(&list->refcount);
}

static ULONG STDMETHODCALLTYPE
d12_list_Release(ID3D12GraphicsCommandList6 *iface)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&list->refcount);

   if (refs == 0) {
      /* A caller may release a list without closing it (or QueryInterface may
       * fail while the freshly-created list is still recording). Keep the
       * allocator occupancy invariant balanced on that path too. Close has
       * already transitioned successful/error recordings to CLOSED and
       * decremented the same counter.
       */
      if ((list->state == D12_LIST_RECORDING ||
           list->state == D12_LIST_ERROR) &&
          list->allocator && list->allocator->recording_lists)
         list->allocator->recording_lists--;

      /* Submitted queue operations own the exact BOs they reference. */
      d12_list_release_segment_scratch(list);
      d12_list_read_tracker_fini(list);
      d12_list_g17p_vdm_storage_reset(list);
      util_dynarray_fini(&list->g17p_vdm_storage);

      if (list->stream_bo)
         agx_bo_unreference(&list->device->dev, list->stream_bo);
      if (list->usc_bo)
         agx_bo_unreference(&list->device->dev, list->usc_bo);
      if (list->sparse_registry_bo)
         agx_bo_unreference(&list->device->dev, list->sparse_registry_bo);
      d12_private_data_store_destroy(&list->private_data);
      util_dynarray_fini(&list->descriptor_fixups);
      util_dynarray_fini(&list->indirect_constant_fixups);
      util_dynarray_fini(&list->indirect_argument_fixups);
      util_dynarray_fini(&list->bundle_cmds);
      if (list->allocator)
         d12_alloc_Release(&list->allocator->iface);
      d12_device_release(list->device);
      free(list);
   }

   return refs;
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_list, ID3D12GraphicsCommandList6,
                                d12_cmd_list_from_iface)

static HRESULT STDMETHODCALLTYPE
d12_list_GetDevice(ID3D12GraphicsCommandList6 *iface, REFIID riid, void **out)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   return ID3D12Device_QueryInterface(&list->device->iface, riid, out);
}

static D3D12_COMMAND_LIST_TYPE STDMETHODCALLTYPE
d12_list_GetType(ID3D12GraphicsCommandList6 *iface)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   return list->type;
}

static HRESULT STDMETHODCALLTYPE
d12_list_Close(ID3D12GraphicsCommandList6 *iface)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   /*
    * A list that hit an error during recording still CLOSES. D3D12
    * reports the recording error from Close and transitions the list to
    * closed regardless; it simply cannot be executed.
    *
    * Returning early without closing was a cascade, not a nicety. The
    * list stayed RECORDING, so the allocator's recording_lists never
    * dropped, so CommandAllocator::Reset failed for the rest of the
    * test -- and because these tests LOOP, every later iteration failed
    * too. It showed up as the two largest failure messages in the whole
    * vkd3d suite ("Failed to close command list", 43 tests; "Failed to
    * reset command allocator", 33) and none of them were about the thing
    * the test was actually checking.
    */
   if (list->state != D12_LIST_RECORDING &&
       list->state != D12_LIST_ERROR)
      return E_FAIL;

   bool errored = list->state == D12_LIST_ERROR;
   if (errored)
      list->had_error = true;

   /*
    * Free the tessellation heap from the GPU, at the end of this list's own
    * execution -- exactly hk_heap()'s queued write.
    *
    * The CPU zero in d12_list_reset_state is not enough and the comment
    * there was wrong about why. D3D12 lets a CLOSED list be handed to
    * ExecuteCommandLists as many times as the caller likes with no Reset in
    * between, and the vkd3d tests do precisely that in loops. Every one of
    * those executions bump-allocates from `bottom` and nothing ever puts it
    * back, so the heap walks off its end and the tessellator reads unmapped
    * memory -- the fault first-faulting-test.sh attributed to
    * test_quad_tessellation_dxil (line 301, unit UL1C, reason Unmapped).
    *
    * Growing the heap to 128 MiB did not move that number (+66 against
    * +69, inside the run-to-run spread), which is what a leak rather than
    * an under-size predicts: a bigger bucket takes longer to overflow but
    * still overflows.
    *
    * A compute dispatch here ends any open render pass and lands in a
    * segment that barriers on everything before it, so the reset is
    * ordered after the tessellation work that consumed the heap.
    */
   if (!errored && list->uses_heap && list->device->heap_desc_va) {
      struct libagx_write_u32_args args = {
         .address = list->device->heap_desc_va +
                    offsetof(struct poly_heap, bottom),
         .value = 0,
      };
      d12_list_dispatch_precomp(list, LIBAGX_WRITE_U32, &args, sizeof(args),
                                agx_1d(1));

      /* That dispatch can fail (segment limit, stream full) and signals it
       * by setting D12_LIST_ERROR. `errored` was sampled before it ran, so
       * without this re-read a list whose stream is now incomplete would
       * be closed as good and submitted -- the exact way this bench wedges.
       */
      if (list->state == D12_LIST_ERROR) {
         errored = true;
         list->had_error = true;
      }
   }

   if (!errored)
      d12_list_close_cdm(list);

   list->state = D12_LIST_CLOSED;

   if (list->allocator && list->allocator->recording_lists)
      list->allocator->recording_lists--;

   if (errored) {
      mesa_loge("Close on errored command list");
      return FAILED(list->record_error) ? list->record_error : E_FAIL;
   }

   return S_OK;
}

/* Swap in freshly allocated recording storage. The queue operation owns the
 * submitted set. Returns false only on allocation failure, in which case the
 * list keeps its existing storage untouched. */
static bool
d12_list_take_fresh_storage(struct d12_cmd_list *list)
{
   struct d12_device *device = list->device;

   /*
    * Size the next recording from this one's actual high-water marks.  The
    * first recording still gets the conservative 1 MiB/2 MiB capacities;
    * repeated small submissions then halve toward compact floors while
    * retaining at least twice their observed demand.  This matters for
    * legal wait-before-signal chains, where thousands of closed recordings
    * can be queued before any of them is allowed to retire.
    */
   size_t control_used =
      list->stream_cur - d12_list_stream_map(list);
   size_t data_used = list->stream_bo->size - list->data_floor;
   size_t stream_need = control_used + data_used +
                        D12_STREAM_RESERVE_BYTES;
   size_t stream_size = MAX3(D12_RECYCLED_STREAM_MIN,
                             ALIGN_POT(stream_need * 2, 16 * 1024),
                             list->stream_bo->size / 2);
   stream_size = MIN2(stream_size, (size_t)D12_STREAM_BO_SIZE);

   size_t usc_size = MAX3(D12_RECYCLED_USC_MIN,
                          ALIGN_POT(list->usc_offset * 2, 16 * 1024),
                          list->usc_bo->size / 2);
   usc_size = MIN2(usc_size, (size_t)D12_USC_BO_SIZE);

   struct d12_cmd_allocator *allocator = list->allocator;
   struct agx_bo *stream =
      allocator ? d12_alloc_pool_take(&allocator->free_stream_bos, stream_size)
                : NULL;
   struct agx_bo *usc =
      allocator ? d12_alloc_pool_take(&allocator->free_usc_bos, usc_size)
                : NULL;
   if (!stream)
      stream = agx_bo_create(&device->dev, stream_size, 0, AGX_BO_WRITEBACK,
                             "d3d12 cmdlist stream");
   if (!usc)
      usc = agx_bo_create(&device->dev, usc_size, 0,
                          AGX_BO_LOW_VA | AGX_BO_WRITEBACK,
                          "d3d12 cmdlist usc");

   /* Same admission rule as creation: the recording path maps this storage
    * without a null check, so a refused Lock2 must fail Reset here rather
    * than fault inside the application's next recording call. */
   if (!stream || !usc || !agx_bo_map(stream) || !agx_bo_map(usc)) {
      if (stream)
         agx_bo_unreference(&device->dev, stream);
      if (usc)
         agx_bo_unreference(&device->dev, usc);
      mesa_loge("command list Reset: out of memory for fresh storage");
      return false;
   }

   /* Queue operations own references to the exact submitted BOs; the
    * recording allocator keeps the storage for reuse after its Reset. */
   d12_list_release_segment_scratch(list);
   if (allocator) {
      util_dynarray_append_typed(&allocator->parked_stream_bos, struct agx_bo *,
                           list->stream_bo);
      util_dynarray_append_typed(&allocator->parked_usc_bos, struct agx_bo *,
                           list->usc_bo);
   } else {
      agx_bo_unreference(&device->dev, list->stream_bo);
      agx_bo_unreference(&device->dev, list->usc_bo);
   }
   if (list->sparse_registry_bo)
      agx_bo_unreference(&device->dev, list->sparse_registry_bo);

   list->stream_bo = stream;
   list->usc_bo = usc;
   list->sparse_registry_bo = NULL;
   list->sparse_registry_capacity = 0;
   list->submitted = false;
   return true;
}

static HRESULT STDMETHODCALLTYPE
d12_list_Reset(ID3D12GraphicsCommandList6 *iface,
               ID3D12CommandAllocator *allocator,
               ID3D12PipelineState *initial_state)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   int32_t failed_close_result;

   /* A recording failure reported by Close permanently poisons this command
    * list. Microsoft requires every later Reset to return that same error. */
   if (d12_command_list_failed_close_reset(
          list->state == D12_LIST_CLOSED, list->had_error,
          (int32_t)list->record_error, (int32_t)E_FAIL,
          &failed_close_result))
      return (HRESULT)failed_close_result;

   if (!allocator)
      return E_INVALIDARG;

   struct d12_cmd_allocator *new_allocator =
      d12_cmd_allocator_from_iface(allocator);
   if (new_allocator->type != list->type)
      return E_INVALIDARG;

   /* Reset is only valid for a closed list, and an allocator can back at most
    * one actively recording list. Test these before touching list state or
    * parking any in-flight storage so a rejected call is side-effect free.
    */
   if (list->state != D12_LIST_CLOSED)
      return E_FAIL;
   if (new_allocator->recording_lists)
      return E_INVALIDARG;

   /*
    * If this recording has already been submitted, the GPU may still be
    * reading its control stream, its USC words and its uploaded data --
    * all of which live in stream_bo/usc_bo. D3D12 explicitly allows Reset
    * immediately after ExecuteCommandLists, so we cannot rewind them:
    * ID3D12CommandAllocator::Reset is where completion is guaranteed, not
    * here. Park the old storage on the allocator that recorded it and take
    * fresh storage.
    *
    * Rewinding regardless is what corrupted test_copy_queue_buffer (the
    * readback was recorded over commands still executing) and what hung
    * the VDM in test_render_pass_suspend_resume_opts_disabled, where one
    * list was re-recorded 32 times and the hardware walked a control
    * stream being rewritten beneath it.
    */
   if (list->submitted && !d12_list_take_fresh_storage(list))
      return E_OUTOFMEMORY;

   list->record_error = S_OK;
   list->pso = initial_state ? d12_pipeline_state_from_iface(initial_state)
                             : NULL;

   /* Re-entering the recording state re-registers with the allocator. The
    * list keeps a reference on it: parked storage is returned to it on a
    * later Reset, which may come after the application dropped its own. */
   if (list->state != D12_LIST_RECORDING) {
      if (new_allocator != list->allocator) {
         d12_alloc_AddRef(&new_allocator->iface);
         if (list->allocator)
            d12_alloc_Release(&list->allocator->iface);
      }
      list->allocator = new_allocator;
      list->allocator->recording_lists++;
   }

   d12_list_begin_recording(list);
   return S_OK;
}

static bool
d12_query_type_matches(const struct d12_query_heap *heap,
                       D3D12_QUERY_TYPE type)
{
   return d12_query_core_type_matches((uint32_t)heap->desc.Type,
                                      (uint32_t)type);
}

static bool
d12_list_validate_query(struct d12_cmd_list *list,
                        struct d12_query_heap *heap,
                        D3D12_QUERY_TYPE type, UINT index)
{
   if (!heap || index >= heap->desc.Count ||
       !d12_query_type_matches(heap, type) ||
       !d12_query_core_timestamp_command_compatible(
          (uint32_t)heap->desc.Type, (uint32_t)list->type)) {
      mesa_loge("query type/index does not match its heap");
      list->state = D12_LIST_ERROR;
      return false;
   }

   return true;
}

static void
d12_list_copy_gpu_bytes(struct d12_cmd_list *list, uint64_t dst,
                        uint64_t src, size_t size_B)
{
   struct libagx_copy_uchar_args args = {
      .dest = dst,
      .src = src,
   };

   d12_list_dispatch_precomp(list, LIBAGX_COPY_UCHAR, &args, sizeof(args),
                             agx_1d(size_B));
}

static void STDMETHODCALLTYPE
d12_list_SetPredication(ID3D12GraphicsCommandList6 *iface,
                        ID3D12Resource *buffer_iface,
                        UINT64 aligned_buffer_offset,
                        D3D12_PREDICATION_OP operation)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->state != D12_LIST_RECORDING)
      return;

   if (!buffer_iface) {
      list->predicate_gpu = 0;
      return;
   }

   struct d12_resource *buffer = d12_resource_from_iface(buffer_iface);
   if (buffer->device != list->device ||
       buffer->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
       (aligned_buffer_offset & 7) ||
       aligned_buffer_offset > buffer->desc.Width ||
       sizeof(uint64_t) > buffer->desc.Width - aligned_buffer_offset ||
       (operation != D3D12_PREDICATION_OP_EQUAL_ZERO &&
        operation != D3D12_PREDICATION_OP_NOT_EQUAL_ZERO)) {
      mesa_loge("SetPredication: invalid buffer, offset, or operation");
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }

   uint64_t snapshot = d12_list_upload_zero(list, sizeof(uint64_t), 8);
   if (!snapshot)
      return;

   d12_list_track_read_resource(list, buffer);

   const uint64_t source = d12_sparse_read_address(
      list->device, buffer->addr + aligned_buffer_offset);
   d12_list_copy_gpu_bytes(list, snapshot, source,
                           sizeof(uint64_t));
   if (list->state != D12_LIST_RECORDING)
      return;

   list->predicate_gpu = snapshot;
   list->predicate_op = operation;

   if (d12_debug()->dump_predication)
      mesa_logi("SetPredication snapshot: source=0x%llx latch=0x%llx "
                "skip_if_nonzero=%u",
                (unsigned long long)source,
                (unsigned long long)snapshot,
                operation == D3D12_PREDICATION_OP_NOT_EQUAL_ZERO);
}

static void STDMETHODCALLTYPE
d12_list_BeginQuery(ID3D12GraphicsCommandList6 *iface,
                    ID3D12QueryHeap *query_heap, D3D12_QUERY_TYPE type,
                    UINT index)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   struct d12_query_heap *heap = (struct d12_query_heap *)query_heap;

   if (list->state != D12_LIST_RECORDING)
      return;

   if (!d12_list_validate_query(list, heap, type, index))
      return;

   d12_list_track_read_bo(list, heap->bo);

   if (heap->desc.Type == D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS) {
      if (list->active_pipeline_stats_heap) {
         mesa_loge("BeginQuery: pipeline statistics query already active");
         list->state = D12_LIST_ERROR;
         return;
      }
      list->active_pipeline_stats_heap = heap;
      list->active_pipeline_stats_index = index;
      memset(&list->pipeline_stats, 0, sizeof(list->pipeline_stats));
      return;
   }

   if (heap->desc.Type == D3D12_QUERY_HEAP_TYPE_SO_STATISTICS) {
      unsigned stream = type - D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0;
      if (stream >= ARRAY_SIZE(list->active_so_stats_heap) ||
          list->active_so_stats_heap[stream]) {
         mesa_loge("BeginQuery: stream-output statistics query already "
                   "active for stream %u", stream);
         list->state = D12_LIST_ERROR;
         return;
      }

      const D3D12_QUERY_DATA_SO_STATISTICS zero = {0};
      uint64_t source = d12_list_upload(list, &zero, sizeof(zero), 8);
      uint64_t result = heap->bo->va->addr + (uint64_t)index * heap->stride;
      if (!source)
         return;

      d12_list_copy_gpu_bytes(list, result, source, sizeof(zero));
      if (list->state != D12_LIST_RECORDING)
         return;

      list->active_so_stats_heap[stream] = heap;
      list->active_so_stats_index[stream] = index;
      return;
   }

   if (d12_query_core_is_timestamp_heap((uint32_t)heap->desc.Type)) {
      mesa_loge("BeginQuery is invalid for timestamp heaps");
      list->state = D12_LIST_ERROR;
      return;
   }

   if (heap->desc.Type != D3D12_QUERY_HEAP_TYPE_OCCLUSION) {
      mesa_loge("BeginQuery currently supports occlusion heaps only");
      list->state = D12_LIST_ERROR;
      return;
   }

   uint64_t start = heap->bo->va->addr + heap->start_offset +
                    (uint64_t)index * heap->stride;
   d12_list_copy_gpu_bytes(list, start, list->device->oclqry_bo->va->addr,
                           sizeof(uint64_t));
   if (list->state == D12_LIST_RECORDING)
      list->active_occlusion_queries++;
}

static void STDMETHODCALLTYPE
d12_list_EndQuery(ID3D12GraphicsCommandList6 *iface,
                  ID3D12QueryHeap *query_heap, D3D12_QUERY_TYPE type,
                  UINT index)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   struct d12_query_heap *heap = (struct d12_query_heap *)query_heap;

   if (list->state != D12_LIST_RECORDING)
      return;

   if (!d12_list_validate_query(list, heap, type, index))
      return;

   d12_list_track_read_bo(list, heap->bo);

   uint64_t result = heap->bo->va->addr + (uint64_t)index * heap->stride;

   if (heap->desc.Type == D3D12_QUERY_HEAP_TYPE_SO_STATISTICS) {
      unsigned stream = type - D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0;
      if (stream >= ARRAY_SIZE(list->active_so_stats_heap) ||
          list->active_so_stats_heap[stream] != heap ||
          list->active_so_stats_index[stream] != index) {
         mesa_loge("EndQuery has no matching stream-output statistics query");
         list->state = D12_LIST_ERROR;
         return;
      }

      list->active_so_stats_heap[stream] = NULL;
      return;
   }

   if (heap->desc.Type == D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS) {
      if (list->active_pipeline_stats_heap != heap ||
          list->active_pipeline_stats_index != index) {
         mesa_loge("EndQuery has no matching pipeline statistics query");
         list->state = D12_LIST_ERROR;
         return;
      }

      uint64_t source = d12_list_upload(
         list, &list->pipeline_stats, sizeof(list->pipeline_stats), 8);
      if (source)
         d12_list_copy_gpu_bytes(list, result, source,
                                 sizeof(list->pipeline_stats));
      list->active_pipeline_stats_heap = NULL;
      return;
   }

   if (d12_query_core_is_timestamp_heap((uint32_t)heap->desc.Type)) {
      /* The kernel-calibrated GPU clock is sampled while recording, then
       * copied by the GPU at this exact point in the command stream.  This
       * preserves Resolve-before-rewrite ordering across submissions.
       */
      uint64_t timestamp = agx_get_gpu_timestamp(&list->device->dev);
      uint64_t source =
         d12_list_upload(list, &timestamp, sizeof(timestamp), 8);
      if (source)
         d12_list_copy_gpu_bytes(list, result, source, sizeof(timestamp));
      return;
   }

   if (heap->desc.Type != D3D12_QUERY_HEAP_TYPE_OCCLUSION ||
       !list->active_occlusion_queries) {
      mesa_loge("EndQuery has no matching active occlusion query");
      list->state = D12_LIST_ERROR;
      return;
   }

   struct libagx_query_delta_args args = {
      .counter = list->device->oclqry_bo->va->addr,
      .start = heap->bo->va->addr + heap->start_offset +
               (uint64_t)index * heap->stride,
      .result = result,
   };
   d12_list_dispatch_precomp(list, LIBAGX_QUERY_DELTA, &args, sizeof(args),
                             agx_1d(1));
   list->active_occlusion_queries--;
}

static void STDMETHODCALLTYPE
d12_list_ResolveQueryData(ID3D12GraphicsCommandList6 *iface,
                          ID3D12QueryHeap *query_heap,
                          D3D12_QUERY_TYPE type, UINT first_query,
                          UINT query_count, ID3D12Resource *destination,
                          UINT64 destination_offset)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   struct d12_query_heap *heap = (struct d12_query_heap *)query_heap;
   struct d12_resource *dst = destination
                                 ? d12_resource_from_iface(destination)
                                 : NULL;

   if (list->state != D12_LIST_RECORDING || !query_count)
      return;

   if (!heap || first_query >= heap->desc.Count ||
       query_count > heap->desc.Count - first_query ||
       !d12_query_type_matches(heap, type) ||
       !d12_query_core_timestamp_command_compatible(
          (uint32_t)heap->desc.Type, (uint32_t)list->type) ||
       !dst ||
       dst->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
       destination_offset > dst->desc.Width ||
       (uint64_t)query_count * heap->stride >
          dst->desc.Width - destination_offset) {
      mesa_loge("ResolveQueryData: invalid heap, range, or destination");
      list->state = D12_LIST_ERROR;
      return;
   }

   d12_list_track_read_bo(list, heap->bo);
   d12_list_track_read_resource(list, dst);

   uint64_t src = heap->bo->va->addr +
                  (uint64_t)first_query * heap->stride;
   uint64_t out = dst->addr + destination_offset;

   if (type == D3D12_QUERY_TYPE_BINARY_OCCLUSION) {
      /* libagx_copy_query_gl's U64/bool path canonicalises every nonzero
       * sample count to the 64-bit value one required by D3D12.
       */
      for (UINT i = 0; i < query_count; i++) {
         struct libagx_copy_query_gl_args args = {
            .query = src + (uint64_t)i * heap->stride,
            .dest = out + (uint64_t)i * heap->stride,
            .value_type = 3, /* PIPE_QUERY_TYPE_U64 */
            .bool_size = 8,
         };
         d12_list_dispatch_precomp(list, LIBAGX_COPY_QUERY_GL, &args,
                                   sizeof(args), agx_1d(1));
      }
   } else {
      d12_list_copy_gpu_bytes(list, out, src,
                              (size_t)query_count * heap->stride);
   }
}

static void STDMETHODCALLTYPE
d12_list_CopyBufferRegion(ID3D12GraphicsCommandList6 *iface,
                          ID3D12Resource *dst, UINT64 dst_offset,
                          ID3D12Resource *src, UINT64 src_offset,
                          UINT64 num_bytes)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   struct d12_resource *dstr = d12_resource_from_iface(dst);
   struct d12_resource *srcr = d12_resource_from_iface(src);

   if (!dst || !src || !num_bytes) {
      mesa_loge("CopyBufferRegion: bad args");
      if (list)
         list->state = D12_LIST_ERROR;
      return;
   }

   if (dst_offset + num_bytes > dstr->desc.Width ||
       src_offset + num_bytes > srcr->desc.Width) {
      mesa_loge("CopyBufferRegion: out of bounds");
      list->state = D12_LIST_ERROR;
      return;
   }

   d12_list_track_read_resource(list, dstr);
   d12_list_track_read_resource(list, srcr);

   struct libagx_copy_uchar_args args = {
      .dest = dstr->addr + dst_offset,
      .src = d12_sparse_read_address(list->device,
                                     srcr->addr + src_offset),
   };

   d12_list_dispatch_precomp(list, LIBAGX_COPY_UCHAR, &args, sizeof(args),
                             agx_1d(num_bytes));
}

static void STDMETHODCALLTYPE
d12_list_WriteBufferImmediate(
   ID3D12GraphicsCommandList6 *iface, UINT count,
   const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER *params,
   const D3D12_WRITEBUFFERIMMEDIATE_MODE *modes)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->state != D12_LIST_RECORDING)
      return;

   if (count && !params) {
      mesa_loge("WriteBufferImmediate: null parameter array");
      list->state = D12_LIST_ERROR;
      return;
   }

   for (UINT i = 0; i < count; i++) {
      D3D12_WRITEBUFFERIMMEDIATE_MODE mode =
         modes ? modes[i] : D3D12_WRITEBUFFERIMMEDIATE_MODE_DEFAULT;
      if (mode != D3D12_WRITEBUFFERIMMEDIATE_MODE_DEFAULT &&
          mode != D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN &&
          mode != D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT) {
         mesa_loge("WriteBufferImmediate: invalid mode %u", (unsigned)mode);
         list->state = D12_LIST_ERROR;
         list->record_error = E_INVALIDARG;
         return;
      }

      if (!params[i].Dest || (params[i].Dest & 3)) {
         mesa_loge("WriteBufferImmediate: invalid destination 0x%llx",
                   (unsigned long long)params[i].Dest);
         list->state = D12_LIST_ERROR;
         return;
      }

      d12_list_track_read_range(list, params[i].Dest, sizeof(uint32_t));

      /* Marker modes differ only in observability to profiling tools. The
       * memory operation is still the same ordered 32-bit write. */
      struct libagx_write_u32_args args = {
         .address = params[i].Dest,
         .value = params[i].Value,
      };
      d12_list_dispatch_precomp(list, LIBAGX_WRITE_U32, &args, sizeof(args),
                                agx_1d(1));
      if (list->state == D12_LIST_ERROR)
         return;
   }
}

struct d12_copy_surface {
   struct agx_bo *bo;
   enum pipe_format format;
   uint64_t base;
   uint64_t storage_offset_B;
   uint64_t size_B;
   uint64_t allocation_size_B;
   uint32_t width_px;
   uint32_t height_px;
   uint32_t depth_px;
   uint32_t width_el;
   uint32_t height_el;
   uint32_t block_width_px;
   uint32_t block_height_px;
   uint32_t blocksize_B;
   uint32_t pitch_B;
   uint64_t slice_stride_B;
   uint32_t stride_el;
   uint32_t tile_width_el;
   uint32_t tile_height_el;
   uint8_t sample_count;
   bool tiled;
   bool reserved;
};

static bool
d12_copy_surface_check_storage(const struct d12_copy_surface *surface,
                               const char *name)
{
   if (surface->reserved) {
      if (surface->storage_offset_B > surface->allocation_size_B ||
          surface->size_B >
             surface->allocation_size_B - surface->storage_offset_B) {
         mesa_loge("CopyTextureRegion: %s sparse surface offset 0x%" PRIx64
                   "+0x%" PRIx64 " exceeds reserved VA size 0x%" PRIx64,
                   name, surface->storage_offset_B, surface->size_B,
                   surface->allocation_size_B);
         return false;
      }
      return true;
   }

   if (!surface->bo || surface->storage_offset_B > surface->bo->size ||
       surface->size_B > surface->bo->size - surface->storage_offset_B) {
      mesa_loge("CopyTextureRegion: %s surface [0x%" PRIx64
                ", +0x%" PRIx64 ") exceeds BO size 0x%zx",
                name, surface->storage_offset_B, surface->size_B,
                surface->bo ? (size_t)surface->bo->size : 0);
      return false;
   }

   return true;
}

static bool
d12_copy_surface_from_texture(struct d12_copy_surface *surface,
                              struct d12_resource *res, UINT subresource,
                              const char *name)
{
   const bool is_3d =
      res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
   if (res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D &&
       res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D && !is_3d) {
      mesa_loge("CopyTextureRegion: %s is not a texture", name);
      return false;
   }

   const unsigned mips = MAX2(res->desc.MipLevels, 1);
   const unsigned layers = is_3d ? 1 : MAX2(res->desc.DepthOrArraySize, 1);
   const uint64_t plane_size = (uint64_t)mips * layers;
   const unsigned plane = subresource / plane_size;
   const unsigned in_plane = subresource % plane_size;
   const unsigned mip = in_plane % mips;
   const unsigned layer = in_plane / mips;

   const struct ail_layout *layout;
   struct agx_bo *bo;
   uint64_t plane_base;
   if (plane == 0) {
      layout = &res->layout;
      bo = res->bo;
      plane_base = res->addr;
   } else if (plane == 1 && res->stencil_bo &&
              res->stencil_layout.size_B) {
      layout = &res->stencil_layout;
      bo = res->stencil_bo;
      plane_base = res->stencil_addr;
   } else {
      mesa_loge("CopyTextureRegion: %s plane %u unsupported", name, plane);
      return false;
   }

   if (mip >= layout->levels || layer >= layout->depth_px) {
      mesa_loge("CopyTextureRegion: %s subresource %u is out of range", name,
                subresource);
      return false;
   }

   const enum pipe_format format = layout->format;
   const uint32_t width_px = u_minify(layout->width_px, mip);
   const uint32_t height_px = u_minify(layout->height_px, mip);
   const uint32_t depth_px = is_3d ? u_minify(layout->depth_px, mip) : 1;
   const uint32_t samples = MAX2(layout->sample_count_sa, 1);
   const uint64_t offset_B = ail_get_layer_level_B(layout, layer, mip);
   const uint64_t slice_size_B = ail_get_level_size_B(layout, mip);
   const uint64_t size_B =
      (uint64_t)(depth_px - 1) * layout->layer_stride_B + slice_size_B;
   if ((!res->reserved && (!bo || plane_base < bo->va->addr)) ||
       (res->reserved && plane != 0)) {
      mesa_loge("CopyTextureRegion: %s plane has invalid storage", name);
      return false;
   }

   *surface = (struct d12_copy_surface){
      .bo = bo,
      .format = format,
      .base = plane_base + offset_B,
      .storage_offset_B = res->reserved ? offset_B
                                        : plane_base - bo->va->addr + offset_B,
      .size_B = size_B,
      .allocation_size_B = res->reserved ? res->sparse_size_B : 0,
      .width_px = width_px,
      .height_px = height_px,
      .depth_px = depth_px,
      .width_el = util_format_get_nblocksx(format, width_px),
      .height_el = util_format_get_nblocksy(format, height_px),
      .block_width_px = util_format_get_blockwidth(format),
      .block_height_px = util_format_get_blockheight(format),
      /* AIL tiles one logical pixel as a block containing all samples. Raw
       * texture copies therefore copy the whole sample block together. */
      .blocksize_B = util_format_get_blocksize(format) * samples,
      .sample_count = samples,
      .slice_stride_B = layout->layer_stride_B,
      .tiled = layout->tiling != AIL_TILING_LINEAR,
      .reserved = res->reserved,
   };

   if (surface->tiled) {
      if (!ail_is_level_twiddled_uncompressed(layout, mip)) {
         mesa_loge("CopyTextureRegion: %s uses compressed metadata", name);
         return false;
      }

      surface->stride_el = layout->stride_el[mip];
      surface->tile_width_el = layout->tilesize_el[mip].width_el;
      surface->tile_height_el = layout->tilesize_el[mip].height_el;
      if (!surface->stride_el || !surface->tile_width_el ||
          !surface->tile_height_el) {
         mesa_loge("CopyTextureRegion: %s has an invalid tiled layout", name);
         return false;
      }
   } else {
      surface->pitch_B = ail_get_linear_stride_B(layout, mip);
   }

   return d12_copy_surface_check_storage(surface, name);
}

HRESULT
d12_cmd_present_blt(
   struct d12_cmd_list *list, struct d12_resource *destination,
   struct d12_resource *source,
   const struct d12_frontend_present_blt_desc *desc)
{
   struct d12_copy_surface d;
   struct d12_copy_surface s;

   if (!list || !destination || !source || !desc ||
       list->state != D12_LIST_RECORDING || destination == source ||
       destination->device != list->device || source->device != list->device ||
       destination->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       source->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       desc->rotation > LIBAGX_PRESENT_ROTATION_CCW_270 ||
       !d12_copy_surface_from_texture(
          &d, destination, desc->destination_subresource,
          "presentation-blit destination") ||
       !d12_copy_surface_from_texture(
          &s, source, desc->source_subresource,
          "presentation-blit source"))
      return E_INVALIDARG;

   if (d.reserved || s.reserved || d.depth_px != 1 || s.depth_px != 1 ||
       d.block_width_px != 1 || d.block_height_px != 1 ||
       s.block_width_px != 1 || s.block_height_px != 1 ||
       d.sample_count != 1 || !s.sample_count ||
       (!!desc->resolve != (s.sample_count > 1)) ||
       desc->destination_right <= desc->destination_left ||
       desc->destination_bottom <= desc->destination_top ||
       desc->destination_right > d.width_px ||
       desc->destination_bottom > d.height_px ||
       desc->source_right <= desc->source_left ||
       desc->source_bottom <= desc->source_top ||
       desc->source_right > s.width_px ||
       desc->source_bottom > s.height_px)
      return E_INVALIDARG;

   struct d12_present_format_info destination_format;
   struct d12_present_format_info source_format;
   const uint32_t width =
      desc->destination_right - desc->destination_left;
   const uint32_t height =
      desc->destination_bottom - desc->destination_top;
   if (!d12_present_format_core_get((uint32_t)destination->desc.Format,
                                    &destination_format) ||
       !d12_present_format_core_get((uint32_t)source->desc.Format,
                                    &source_format) ||
       d.blocksize_B != destination_format.pixel_size_B ||
       s.blocksize_B != source_format.pixel_size_B * s.sample_count ||
       (uint64_t)width * height > UINT32_MAX)
      return E_NOTIMPL;

   d12_list_track_read_resource(list, destination);
   d12_list_track_read_resource(list, source);

   struct libagx_present_blt_args args = {
      .dest = d.base,
      .src = s.base,
      .dest_width_px = width,
      .dest_height_px = height,
      .dest_x_px = desc->destination_left,
      .dest_y_px = desc->destination_top,
      .src_left_px = desc->source_left,
      .src_top_px = desc->source_top,
      .src_right_px = desc->source_right,
      .src_bottom_px = desc->source_bottom,
      .dest_pitch_B = d.pitch_B,
      .src_pitch_B = s.pitch_B,
      .dest_stride_el = d.stride_el,
      .src_stride_el = s.stride_el,
      .dest_tile_width_el = d.tile_width_el,
      .dest_tile_height_el = d.tile_height_el,
      .src_tile_width_el = s.tile_width_el,
      .src_tile_height_el = s.tile_height_el,
      .dest_pixel_size_B = destination_format.pixel_size_B,
      .src_pixel_size_B = source_format.pixel_size_B,
      .source_sample_count = s.sample_count,
      .dest_tiled = d.tiled,
      .src_tiled = s.tiled,
      .dest_format = destination_format.shader_format,
      .src_format = source_format.shader_format,
      .rotation = desc->rotation,
   };

   d12_list_dispatch_precomp(
      list, LIBAGX_PRESENT_BLT, &args, sizeof(args),
      agx_1d((uint64_t)width * height));
   return list->state == D12_LIST_ERROR ? E_FAIL : S_OK;
}

static bool
d12_copy_surface_from_footprint(
   struct d12_copy_surface *surface, struct d12_resource *res,
   const D3D12_PLACED_SUBRESOURCE_FOOTPRINT *placed, const char *name)
{
   if (res->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
      mesa_loge("CopyTextureRegion: %s footprint is not in a buffer", name);
      return false;
   }

   enum pipe_format format = d12_format_to_pipe(placed->Footprint.Format);
   if (format == PIPE_FORMAT_NONE)
      format = d12_format_depth_plane(placed->Footprint.Format);

   if (format == PIPE_FORMAT_NONE || !placed->Footprint.Depth ||
       !placed->Footprint.Width || !placed->Footprint.Height) {
      mesa_loge("CopyTextureRegion: %s has an unsupported footprint", name);
      return false;
   }

   const uint32_t block_width = util_format_get_blockwidth(format);
   const uint32_t block_height = util_format_get_blockheight(format);
   const uint32_t blocksize = util_format_get_blocksize(format);
   const uint32_t width_el =
      util_format_get_nblocksx(format, placed->Footprint.Width);
   const uint32_t height_el =
      util_format_get_nblocksy(format, placed->Footprint.Height);
   const uint64_t minimum_pitch = (uint64_t)width_el * blocksize;

   if (placed->Footprint.RowPitch < minimum_pitch) {
      mesa_loge("CopyTextureRegion: %s row pitch %u is smaller than %" PRIu64,
                name, placed->Footprint.RowPitch, minimum_pitch);
      return false;
   }

   const uint64_t slice_stride_B =
      (uint64_t)height_el * placed->Footprint.RowPitch;
   const uint64_t size_B =
      (uint64_t)(placed->Footprint.Depth - 1) * slice_stride_B +
      (uint64_t)(height_el - 1) * placed->Footprint.RowPitch + minimum_pitch;
   if (placed->Offset > res->desc.Width ||
       size_B > res->desc.Width - placed->Offset) {
      mesa_loge("CopyTextureRegion: %s footprint exceeds its buffer", name);
      return false;
   }

   *surface = (struct d12_copy_surface){
      .bo = res->bo,
      .format = format,
      .base = res->addr + placed->Offset,
      .storage_offset_B = res->offset_B + placed->Offset,
      .size_B = size_B,
      .width_px = placed->Footprint.Width,
      .height_px = placed->Footprint.Height,
      .depth_px = placed->Footprint.Depth,
      .width_el = width_el,
      .height_el = height_el,
      .block_width_px = block_width,
      .block_height_px = block_height,
      .blocksize_B = blocksize,
      .pitch_B = placed->Footprint.RowPitch,
      .slice_stride_B = slice_stride_B,
      .sample_count = 1,
   };

   return d12_copy_surface_check_storage(surface, name);
}

static bool
d12_copy_surface_from_location(struct d12_copy_surface *surface,
                               const D3D12_TEXTURE_COPY_LOCATION *location,
                               const char *name)
{
   struct d12_resource *res = d12_resource_from_iface(location->pResource);

   if (location->Type == D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX)
      return d12_copy_surface_from_texture(surface, res,
                                           location->SubresourceIndex, name);
   if (location->Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT)
      return d12_copy_surface_from_footprint(surface, res,
                                             &location->PlacedFootprint, name);

   mesa_loge("CopyTextureRegion: %s location type %u unsupported", name,
             location->Type);
   return false;
}

static void
d12_copy_texture_region(struct d12_cmd_list *list,
                        const D3D12_TEXTURE_COPY_LOCATION *dst,
                        UINT dst_x, UINT dst_y, UINT dst_z,
                        const D3D12_TEXTURE_COPY_LOCATION *src,
                        const D3D12_BOX *src_box)
{
   if (!dst || !src || !dst->pResource || !src->pResource) {
      mesa_loge("CopyTextureRegion: null location");
      list->state = D12_LIST_ERROR;
      return;
   }

   struct d12_copy_surface d, s;
   if (!d12_copy_surface_from_location(&d, dst, "destination") ||
       !d12_copy_surface_from_location(&s, src, "source")) {
      list->state = D12_LIST_ERROR;
      return;
   }

   d12_list_track_read_resource(
      list, d12_resource_from_iface(dst->pResource));
   d12_list_track_read_resource(
      list, d12_resource_from_iface(src->pResource));

   /* Source reads use the reserved resource's zero-filled alias. The
    * destination retains its writable primary VA so unmapped writes are
    * discarded into the sink page. */
   s.base = d12_sparse_read_address(list->device, s.base);

   if (d.blocksize_B != s.blocksize_B) {
      mesa_loge("CopyTextureRegion: source/destination elements are %u/%u "
                "bytes", s.blocksize_B, d.blocksize_B);
      list->state = D12_LIST_ERROR;
      return;
   }

   if (d.sample_count != s.sample_count) {
      mesa_loge("CopyTextureRegion: source/destination sample counts are "
                "%u/%u", s.sample_count, d.sample_count);
      list->state = D12_LIST_ERROR;
      return;
   }

   const struct d12_texture_copy_surface_geometry destination_geometry = {
      .width_px = d.width_px,
      .height_px = d.height_px,
      .depth_px = d.depth_px,
      .width_el = d.width_el,
      .height_el = d.height_el,
      .block_width_px = d.block_width_px,
      .block_height_px = d.block_height_px,
   };
   const struct d12_texture_copy_surface_geometry source_geometry = {
      .width_px = s.width_px,
      .height_px = s.height_px,
      .depth_px = s.depth_px,
      .width_el = s.width_el,
      .height_el = s.height_el,
      .block_width_px = s.block_width_px,
      .block_height_px = s.block_height_px,
   };
   const struct d12_texture_copy_region_geometry region_geometry = {
      .destination_x_px = dst_x,
      .destination_y_px = dst_y,
      .destination_z_px = dst_z,
      .has_source_box = src_box != NULL,
      .source_left_px = src_box ? src_box->left : 0,
      .source_top_px = src_box ? src_box->top : 0,
      .source_front_px = src_box ? src_box->front : 0,
      .source_right_px = src_box ? src_box->right : 0,
      .source_bottom_px = src_box ? src_box->bottom : 0,
      .source_back_px = src_box ? src_box->back : 0,
   };
   struct d12_texture_copy_plan plan;
   const enum d12_texture_copy_plan_status status =
      d12_texture_copy_plan_region(&destination_geometry, &source_geometry,
                                   &region_geometry, &plan);
   if (status == D12_TEXTURE_COPY_PLAN_EMPTY)
      return;
   if (status != D12_TEXTURE_COPY_PLAN_OK) {
      const char *reason = "invalid geometry";
      switch (status) {
      case D12_TEXTURE_COPY_PLAN_SOURCE_X_UNALIGNED:
         reason = "source x is not block aligned";
         break;
      case D12_TEXTURE_COPY_PLAN_SOURCE_Y_UNALIGNED:
         reason = "source y is not block aligned";
         break;
      case D12_TEXTURE_COPY_PLAN_DESTINATION_X_UNALIGNED:
         reason = "destination x is not block aligned";
         break;
      case D12_TEXTURE_COPY_PLAN_DESTINATION_Y_UNALIGNED:
         reason = "destination y is not block aligned";
         break;
      case D12_TEXTURE_COPY_PLAN_OUT_OF_BOUNDS:
         reason = "copy region is out of bounds";
         break;
      default:
         break;
      }
      mesa_loge("CopyTextureRegion: %s", reason);
      list->state = D12_LIST_ERROR;
      return;
   }

   /* libagx_copy_image now uses the same CPU-reduced tile geometry and
    * two-dimensional grid as Honeykrisp's raw image-copy caller. Its Morton
    * encoder expands seven bits per axis, so refuse geometry outside that
    * contract instead of silently truncating tiled addresses. */
   const struct d12_copy_surface *surfaces[] = {&d, &s};
   for (unsigned i = 0; i < ARRAY_SIZE(surfaces); ++i) {
      const struct d12_copy_surface *surface = surfaces[i];
      if (surface->tiled &&
          (!surface->stride_el ||
           !util_is_power_of_two_nonzero(surface->tile_width_el) ||
           !util_is_power_of_two_nonzero(surface->tile_height_el) ||
           surface->tile_width_el > 128 || surface->tile_height_el > 128)) {
         mesa_loge("CopyTextureRegion: tiled geometry exceeds copy kernel contract");
         list->state = D12_LIST_ERROR;
         return;
      }
   }

   struct libagx_copy_image_args args = {
      .width_el = plan.width_el,
      .height_el = plan.height_el,
      .dest_x_el = plan.destination_x_el,
      .dest_y_el = plan.destination_y_el,
      .src_x_el = plan.source_x_el,
      .src_y_el = plan.source_y_el,
      .dest_pitch_B = d.pitch_B,
      .src_pitch_B = s.pitch_B,
      .dest_tiles_per_row =
         d.tiled ? ((uint64_t)d.stride_el + d.tile_width_el - 1) >>
                      util_logbase2(d.tile_width_el) : 0,
      .src_tiles_per_row =
         s.tiled ? ((uint64_t)s.stride_el + s.tile_width_el - 1) >>
                      util_logbase2(s.tile_width_el) : 0,
      .dest_tile_width_log2 = d.tiled ? util_logbase2(d.tile_width_el) : 0,
      .dest_tile_height_log2 = d.tiled ? util_logbase2(d.tile_height_el) : 0,
      .src_tile_width_log2 = s.tiled ? util_logbase2(s.tile_width_el) : 0,
      .src_tile_height_log2 = s.tiled ? util_logbase2(s.tile_height_el) : 0,
      .blocksize_B = s.blocksize_B,
      .dest_tiled = d.tiled,
      .src_tiled = s.tiled,
   };

   /* The proven copy kernel is two-dimensional. AIL stores a 3D mip as a
    * stack of equally-strided twiddled slices, so preserve that kernel and
    * advance only the validated source/destination bases for each Z slice.
    */
   for (uint32_t z = 0; z < plan.depth_px; ++z) {
      args.dest = d.base +
                  (uint64_t)(plan.destination_z_px + z) * d.slice_stride_B;
      args.src = s.base +
                 (uint64_t)(plan.source_front_px + z) * s.slice_stride_B;
      d12_list_dispatch_precomp(list, LIBAGX_COPY_IMAGE, &args, sizeof(args),
                                agx_2d(plan.width_el, plan.height_el));
      if (list->state == D12_LIST_ERROR)
         return;
   }
}

static void
d12_resolve(struct d12_cmd_list *list, ID3D12Resource *dst,
            UINT dst_subresource, UINT dst_x, UINT dst_y,
            ID3D12Resource *src, UINT src_subresource,
            const D3D12_RECT *src_rect, DXGI_FORMAT format,
            D3D12_RESOLVE_MODE mode)
{
   struct d12_resource *dstr = d12_resource_from_iface(dst);
   struct d12_resource *srcr = d12_resource_from_iface(src);
   const unsigned dst_samples = MAX2(dstr->layout.sample_count_sa, 1);
   const unsigned src_samples = MAX2(srcr->layout.sample_count_sa, 1);

   _Static_assert((int)D3D12_RESOLVE_MODE_MIN ==
                     (int)D12_RESOLVE_CORE_MODE_MIN,
                  "D3D12/core MIN resolve ABI drift");
   _Static_assert((int)D3D12_RESOLVE_MODE_MAX ==
                     (int)D12_RESOLVE_CORE_MODE_MAX,
                  "D3D12/core MAX resolve ABI drift");
   _Static_assert((int)D3D12_RESOLVE_MODE_AVERAGE ==
                     (int)D12_RESOLVE_CORE_MODE_AVERAGE,
                  "D3D12/core AVERAGE resolve ABI drift");

   struct d12_resolve_core_format_info resolve_info;
   if (dstr->device != list->device || srcr->device != list->device ||
       dstr->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       srcr->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       !d12_resolve_core_resource_compatible((uint32_t)dstr->desc.Format,
                                             (uint32_t)format) ||
       !d12_resolve_core_resource_compatible((uint32_t)srcr->desc.Format,
                                             (uint32_t)format) ||
       !d12_resolve_core_get_format((uint32_t)format, &resolve_info) ||
       !d12_resolve_core_supports_mode((uint32_t)format, (uint32_t)mode)) {
      mesa_loge("ResolveSubresource: incompatible resource/format/mode "
                "(source %u, destination %u, format %u, mode %u)",
                (unsigned)srcr->desc.Format, (unsigned)dstr->desc.Format,
                (unsigned)format, (unsigned)mode);
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }

   const bool minimum = mode == D3D12_RESOLVE_MODE_MIN;
   const bool maximum = mode == D3D12_RESOLVE_MODE_MAX;
   const bool average = mode == D3D12_RESOLVE_MODE_AVERAGE;
   const bool float32_scalar =
      format == DXGI_FORMAT_D32_FLOAT ||
      format == DXGI_FORMAT_R32_FLOAT ||
      format == DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
   const bool min_float32 = minimum && float32_scalar;
   const bool average_float32 = average && float32_scalar;
   const bool extrema_uint32 =
      (minimum || maximum) && format == DXGI_FORMAT_R32_UINT;
   const bool extrema_sint32 =
      (minimum || maximum) && format == DXGI_FORMAT_R32_SINT;
   const bool extrema_stencil8 =
      (minimum || maximum) && format == DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
   const bool extrema_scalar =
      (maximum && float32_scalar) || extrema_uint32 || extrema_sint32 ||
      extrema_stencil8;
   const bool unorm8x4_average =
      average && (format == DXGI_FORMAT_R8G8B8A8_UNORM ||
                  format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
   const bool unorm8x4_extrema =
      (minimum || maximum) && format == DXGI_FORMAT_R8G8B8A8_UNORM;
   const bool unorm8x4 = unorm8x4_average || unorm8x4_extrema;
   const bool float32x4 =
      average && format == DXGI_FORMAT_R32G32B32A32_FLOAT;
   const unsigned pixel_B = resolve_info.pixel_size_B;
   if (dst_samples != 1 || (src_samples != 2 && src_samples != 4) ||
       !pixel_B) {
      mesa_loge("ResolveSubresource: unsupported format/mode/sample shape "
                "(format %u, mode %u, %u -> %u samples)",
                (unsigned)format, (unsigned)mode, src_samples, dst_samples);
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }

   struct d12_copy_surface d, s;
   if (!d12_copy_surface_from_texture(&d, dstr, dst_subresource,
                                      "resolve destination") ||
       !d12_copy_surface_from_texture(&s, srcr, src_subresource,
                                      "resolve source") ||
       d.blocksize_B != pixel_B ||
       s.blocksize_B != pixel_B * src_samples || !s.tiled) {
      mesa_loge("ResolveSubresource: incompatible source/destination");
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }

   const int64_t left = src_rect ? src_rect->left : 0;
   const int64_t top = src_rect ? src_rect->top : 0;
   const int64_t right = src_rect ? src_rect->right : s.width_px;
   const int64_t bottom = src_rect ? src_rect->bottom : s.height_px;
   if (left < 0 || top < 0 || right < left || bottom < top ||
       right > s.width_px || bottom > s.height_px ||
       (uint64_t)dst_x + (right - left) > d.width_px ||
       (uint64_t)dst_y + (bottom - top) > d.height_px) {
      mesa_loge("ResolveSubresource: resolve rectangle is out of bounds");
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }

   const unsigned width_px = right - left;
   const unsigned height_px = bottom - top;
   if (!width_px || !height_px)
      return;

   d12_list_track_read_resource(list, dstr);
   d12_list_track_read_resource(list, srcr);

   s.base = d12_sparse_read_address(list->device, s.base);

   struct libagx_resolve_unorm8x4_args args = {
      .dest = d.base,
      .src = s.base,
      .width_px = width_px,
      .height_px = height_px,
      .dest_x_px = dst_x,
      .dest_y_px = dst_y,
      .src_x_px = left,
      .src_y_px = top,
      .dest_pitch_B = d.pitch_B,
      .dest_stride_el = d.stride_el,
      .src_stride_el = s.stride_el,
      .dest_tile_width_el = d.tile_width_el,
      .dest_tile_height_el = d.tile_height_el,
      .src_tile_width_el = s.tile_width_el,
      .src_tile_height_el = s.tile_height_el,
      .sample_count = src_samples,
      .dest_tiled = d.tiled,
      .src_tiled = s.tiled,
      .resolve_mode = minimum ? 1 : maximum ? 2 : 0,
   };

   const bool typed_unorm8x4 =
      unorm8x4_average &&
      (util_format_is_srgb(s.format) ||
       format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);

   if (typed_unorm8x4) {
      struct libagx_resolve_unorm8x4_typed_args typed_args = {
         .dest = d.base,
         .src = s.base,
         .width_px = width_px,
         .height_px = height_px,
         .dest_x_px = dst_x,
         .dest_y_px = dst_y,
         .src_x_px = left,
         .src_y_px = top,
         .dest_pitch_B = d.pitch_B,
         .dest_stride_el = d.stride_el,
         .src_stride_el = s.stride_el,
         .dest_tile_width_el = d.tile_width_el,
         .dest_tile_height_el = d.tile_height_el,
         .src_tile_width_el = s.tile_width_el,
         .src_tile_height_el = s.tile_height_el,
         .sample_count = src_samples,
         .dest_tiled = d.tiled,
         .src_tiled = s.tiled,
         /* ResolveSubresource's Format parameter supplies the typed
          * interpretation when either allocation is typeless. */
         .src_srgb = format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
         .dest_srgb = format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
      };
      d12_list_dispatch_precomp(list, LIBAGX_RESOLVE_UNORM8X4_TYPED,
                                &typed_args, sizeof(typed_args),
                                agx_1d((uint64_t)width_px * height_px));
   } else if (extrema_scalar) {
      struct libagx_resolve_scalar_extrema_args extrema_args = {
         .dest = d.base,
         .src = s.base,
         .width_px = width_px,
         .height_px = height_px,
         .dest_x_px = dst_x,
         .dest_y_px = dst_y,
         .src_x_px = left,
         .src_y_px = top,
         .dest_pitch_B = d.pitch_B,
         .dest_stride_el = d.stride_el,
         .src_stride_el = s.stride_el,
         .dest_tile_width_el = d.tile_width_el,
         .dest_tile_height_el = d.tile_height_el,
         .src_tile_width_el = s.tile_width_el,
         .src_tile_height_el = s.tile_height_el,
         .sample_count = src_samples,
         .dest_tiled = d.tiled,
         .src_tiled = s.tiled,
         .scalar_size_B = pixel_B,
         .operation = extrema_uint32 ? 1 : extrema_sint32 ? 2 : 0,
         .choose_max = maximum,
      };
      d12_list_dispatch_precomp(list, LIBAGX_RESOLVE_SCALAR_EXTREMA,
                                &extrema_args, sizeof(extrema_args),
                                agx_1d((uint64_t)width_px * height_px));
   } else if (min_float32 || average_float32 || unorm8x4 || float32x4) {
      const enum libagx_program shader =
         min_float32 ? LIBAGX_RESOLVE_FLOAT32_MIN
                     : average_float32 ? LIBAGX_RESOLVE_FLOAT32_AVERAGE
                     : unorm8x4 ? LIBAGX_RESOLVE_UNORM8X4
                                 : LIBAGX_RESOLVE_FLOAT32X4;
      d12_list_dispatch_precomp(list, shader, &args, sizeof(args),
                                agx_1d((uint64_t)width_px * height_px));
   } else {
      struct libagx_resolve_typed_args typed_args = {
         .dest = d.base,
         .src = s.base,
         .width_px = width_px,
         .height_px = height_px,
         .dest_x_px = dst_x,
         .dest_y_px = dst_y,
         .src_x_px = left,
         .src_y_px = top,
         .dest_pitch_B = d.pitch_B,
         .dest_stride_el = d.stride_el,
         .src_stride_el = s.stride_el,
         .dest_tile_width_el = d.tile_width_el,
         .dest_tile_height_el = d.tile_height_el,
         .src_tile_width_el = s.tile_width_el,
         .src_tile_height_el = s.tile_height_el,
         .sample_count = src_samples,
         .dest_tiled = d.tiled,
         .src_tiled = s.tiled,
         .pixel_size_B = resolve_info.pixel_size_B,
         .format = resolve_info.shader_format,
         .resolve_mode = mode,
         .value_class = resolve_info.value_class,
      };
      d12_list_dispatch_precomp(list, LIBAGX_RESOLVE_TYPED, &typed_args,
                                sizeof(typed_args),
                                agx_1d((uint64_t)width_px * height_px));
   }
}

static void STDMETHODCALLTYPE
d12_list_ResolveSubresource(ID3D12GraphicsCommandList6 *iface,
                            ID3D12Resource *dst, UINT dst_subresource,
                            ID3D12Resource *src, UINT src_subresource,
                            DXGI_FORMAT format)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   if (list->state != D12_LIST_RECORDING || !dst || !src)
      return;

   d12_resolve(list, dst, dst_subresource, 0, 0, src, src_subresource, NULL,
               format, D3D12_RESOLVE_MODE_AVERAGE);
}

static void STDMETHODCALLTYPE
d12_list_ResolveSubresourceRegion(ID3D12GraphicsCommandList6 *iface,
                                  ID3D12Resource *dst, UINT dst_subresource,
                                  UINT dst_x, UINT dst_y, ID3D12Resource *src,
                                  UINT src_subresource, D3D12_RECT *src_rect,
                                  DXGI_FORMAT format,
                                  D3D12_RESOLVE_MODE resolve_mode)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   if (list->state != D12_LIST_RECORDING || !dst || !src)
      return;

   if (resolve_mode == D3D12_RESOLVE_MODE_AVERAGE ||
       resolve_mode == D3D12_RESOLVE_MODE_MIN ||
       resolve_mode == D3D12_RESOLVE_MODE_MAX) {
      d12_resolve(list, dst, dst_subresource, dst_x, dst_y, src,
                  src_subresource, src_rect, format, resolve_mode);
      return;
   }

   if (resolve_mode == D3D12_RESOLVE_MODE_DECOMPRESS) {
      D3D12_TEXTURE_COPY_LOCATION d = {
         .pResource = dst,
         .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
         .SubresourceIndex = dst_subresource,
      };
      D3D12_TEXTURE_COPY_LOCATION s = {
         .pResource = src,
         .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
         .SubresourceIndex = src_subresource,
      };
      D3D12_BOX box;
      D3D12_BOX *box_ptr = NULL;
      if (src_rect) {
         if (src_rect->left < 0 || src_rect->top < 0 ||
             src_rect->right < src_rect->left ||
             src_rect->bottom < src_rect->top) {
            mesa_loge("ResolveSubresourceRegion: invalid source rectangle");
            list->state = D12_LIST_ERROR;
            return;
         }
         box = (D3D12_BOX){
            .left = src_rect->left,
            .top = src_rect->top,
            .front = 0,
            .right = src_rect->right,
            .bottom = src_rect->bottom,
            .back = 1,
         };
         box_ptr = &box;
      }

      d12_copy_texture_region(list, &d, dst_x, dst_y, 0, &s, box_ptr);
      return;
   }

   mesa_loge("ResolveSubresourceRegion: unsupported resolve mode %u",
             (unsigned)resolve_mode);
   list->state = D12_LIST_ERROR;
}

static void STDMETHODCALLTYPE
d12_list_CopyResource(ID3D12GraphicsCommandList6 *iface, ID3D12Resource *dst,
                      ID3D12Resource *src)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->state != D12_LIST_RECORDING || !dst || !src)
      return;

   struct d12_resource *dstr = d12_resource_from_iface(dst);
   struct d12_resource *srcr = d12_resource_from_iface(src);

   if (dstr->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER &&
       srcr->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
      if (dstr->desc.Width != srcr->desc.Width) {
         mesa_loge("CopyResource: buffer sizes differ");
         list->state = D12_LIST_ERROR;
         return;
      }

      d12_list_CopyBufferRegion(iface, dst, 0, src, 0, dstr->desc.Width);
      return;
   }

   if (dstr->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
       srcr->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
      mesa_loge("CopyResource: resource dimensions differ or are unsupported");
      list->state = D12_LIST_ERROR;
      return;
   }

   /* Invalid format-size copies are observed in shipped applications. Native
    * implementations either ignore them or perform an implementation-defined
    * bit copy; skipping is safe and matches vkd3d-proton's hang avoidance.
    */
   if (util_format_get_blocksize(dstr->layout.format) !=
       util_format_get_blocksize(srcr->layout.format)) {
      mesa_logw("CopyResource: ignoring incompatible texture element sizes");
      return;
   }

   const unsigned dst_subresources =
      MAX2(dstr->desc.MipLevels, 1) * MAX2(dstr->desc.DepthOrArraySize, 1) *
      (dstr->stencil_bo ? 2 : 1);
   const unsigned src_subresources =
      MAX2(srcr->desc.MipLevels, 1) * MAX2(srcr->desc.DepthOrArraySize, 1) *
      (srcr->stencil_bo ? 2 : 1);
   if (dst_subresources != src_subresources) {
      mesa_loge("CopyResource: texture subresource counts differ");
      list->state = D12_LIST_ERROR;
      return;
   }

   for (unsigned i = 0; i < dst_subresources; ++i) {
      D3D12_TEXTURE_COPY_LOCATION d = {
         .pResource = dst,
         .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
         .SubresourceIndex = i,
      };
      D3D12_TEXTURE_COPY_LOCATION s = {
         .pResource = src,
         .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
         .SubresourceIndex = i,
      };

      d12_copy_texture_region(list, &d, 0, 0, 0, &s, NULL);
      if (list->state == D12_LIST_ERROR)
         return;
   }
}

static void STDMETHODCALLTYPE
d12_list_ClearRenderTargetView(ID3D12GraphicsCommandList6 *iface,
                               D3D12_CPU_DESCRIPTOR_HANDLE rtv_handle,
                               const FLOAT color[4], UINT num_rects,
                               const D3D12_RECT *rects)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->state != D12_LIST_RECORDING)
      return;

   if (num_rects && !rects) {
      mesa_loge("ClearRenderTargetView: null rectangle array");
      list->state = D12_LIST_ERROR;
      return;
   }

   struct d12_rtv *rtv = (struct d12_rtv *)rtv_handle.ptr;
   if (!rtv || !rtv->resource || rtv->kind != D12_DESC_RTV) {
      mesa_loge("ClearRenderTargetView: invalid RTV handle");
      list->state = D12_LIST_ERROR;
      return;
   }

   if (num_rects != 0) {
      struct d12_resource *res = rtv->resource;
      struct d12_copy_surface surface;
      if (rtv->address < res->addr || !res->layout.layer_stride_B) {
         mesa_loge("ClearRenderTargetView: RTV address precedes its resource");
         list->state = D12_LIST_ERROR;
         return;
      }

      const uint64_t view_offset_B = rtv->address - res->addr;
      if (view_offset_B % res->layout.layer_stride_B) {
         mesa_loge("ClearRenderTargetView: RTV layer address is misaligned");
         list->state = D12_LIST_ERROR;
         return;
      }

      const unsigned level = rtv->texture_level;
      const uint64_t first_layer64 =
         view_offset_B / res->layout.layer_stride_B;
      if (level >= res->layout.levels) {
         mesa_loge("ClearRenderTargetView: RTV mip is out of range");
         list->state = D12_LIST_ERROR;
         return;
      }
      const unsigned resource_layers =
         res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D
            ? u_minify(res->layout.depth_px, level)
            : res->layout.depth_px;
      const unsigned view_layers = MAX2(rtv->texture_layers, 1);
      if (first_layer64 >= resource_layers ||
          view_layers > resource_layers - first_layer64) {
         mesa_loge("ClearRenderTargetView: RTV subresource is out of range");
         list->state = D12_LIST_ERROR;
         return;
      }

      const unsigned first_layer = (unsigned)first_layer64;
      unsigned subresource = level;
      if (res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D)
         subresource += first_layer * res->layout.levels;
      if (!d12_copy_surface_from_texture(&surface, res, subresource,
                                         "partial-clear destination")) {
         list->state = D12_LIST_ERROR;
         return;
      }
      if (res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
         surface.base +=
            (uint64_t)first_layer * res->layout.layer_stride_B;
         surface.storage_offset_B +=
            (uint64_t)first_layer * res->layout.layer_stride_B;
         surface.depth_px = view_layers;
         surface.size_B =
            (uint64_t)(view_layers - 1) * res->layout.layer_stride_B +
            ail_get_level_size_B(&res->layout, level);
      }
      if (!d12_copy_surface_check_storage(&surface,
                                          "partial-clear destination")) {
         list->state = D12_LIST_ERROR;
         return;
      }
      if (res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
         surface.depth_px = 1;
         surface.size_B = ail_get_level_size_B(&res->layout, level);
      }

      enum pipe_format format = d12_format_to_pipe(rtv->format);
      const unsigned format_blocksize =
         format != PIPE_FORMAT_NONE ? util_format_get_blocksize(format) : 0;
      if (format == PIPE_FORMAT_NONE ||
          format_blocksize * surface.sample_count != surface.blocksize_B ||
          surface.blocksize_B > 64 || surface.block_width_px != 1 ||
          surface.block_height_px != 1) {
         mesa_loge("ClearRenderTargetView: partial clear format %u "
                   "unsupported", (unsigned)rtv->format);
         list->state = D12_LIST_ERROR;
         return;
      }

      union {
         float f[4];
         uint32_t u[4];
         int32_t i[4];
      } converted;
      if (util_format_is_pure_uint(format)) {
         for (unsigned c = 0; c < 4; ++c)
            converted.u[c] = (uint32_t)color[c];
      } else if (util_format_is_pure_sint(format)) {
         for (unsigned c = 0; c < 4; ++c)
            converted.i[c] = (int32_t)color[c];
      } else {
         memcpy(converted.f, color, sizeof(converted.f));
      }

      uint8_t packed[64] = {0};
      util_format_pack_rgba(format, packed, &converted, 1);
      for (unsigned s = 1; s < surface.sample_count; ++s)
         memcpy(packed + s * format_blocksize, packed, format_blocksize);
      uint64_t value = d12_list_upload(list, packed, surface.blocksize_B, 16);
      if (!value)
         return;

      for (unsigned layer = 0; layer < view_layers; ++layer) {
         struct d12_copy_surface layer_surface = surface;
         layer_surface.base +=
            (uint64_t)layer * res->layout.layer_stride_B;
         layer_surface.storage_offset_B +=
            (uint64_t)layer * res->layout.layer_stride_B;
         if (!d12_copy_surface_check_storage(&layer_surface,
                                             "partial-clear destination")) {
            list->state = D12_LIST_ERROR;
            return;
         }

         for (unsigned i = 0; i < num_rects; ++i) {
            const int64_t left = MAX2((int64_t)rects[i].left, 0);
            const int64_t top = MAX2((int64_t)rects[i].top, 0);
            const int64_t right = MIN2((int64_t)rects[i].right,
                                       (int64_t)layer_surface.width_px);
            const int64_t bottom = MIN2((int64_t)rects[i].bottom,
                                        (int64_t)layer_surface.height_px);

            if (right <= left || bottom <= top)
               continue;

            const uint32_t width = right - left;
            const uint32_t height = bottom - top;
            struct libagx_fill_image_args args = {
               .dest = layer_surface.base,
               .value = value,
               .width_el = width,
               .height_el = height,
               .dest_x_el = left,
               .dest_y_el = top,
               .dest_pitch_B = layer_surface.pitch_B,
               .dest_stride_el = layer_surface.stride_el,
               .dest_tile_width_el = layer_surface.tile_width_el,
               .dest_tile_height_el = layer_surface.tile_height_el,
               .blocksize_B = layer_surface.blocksize_B,
               .dest_tiled = layer_surface.tiled,
            };
            d12_list_dispatch_precomp(list, LIBAGX_FILL_IMAGE, &args,
                                      sizeof(args),
                                      agx_1d((uint64_t)width * height));
            if (list->state == D12_LIST_ERROR)
               return;
         }
      }
      return;
   }

   d12_list_begin_render(list, rtv, 1, NULL, NULL, AGX_BG_CLEAR, color, true);
}

/* Raw element copies cover linear placed footprints and AIL tiled texture
 * subresources through the shared, bounds-checked path above. */
static void STDMETHODCALLTYPE
d12_list_CopyTextureRegion(ID3D12GraphicsCommandList6 *iface,
                           const D3D12_TEXTURE_COPY_LOCATION *dst,
                           UINT dst_x, UINT dst_y, UINT dst_z,
                           const D3D12_TEXTURE_COPY_LOCATION *src,
                           const D3D12_BOX *src_box)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->state != D12_LIST_RECORDING)
      return;

   d12_copy_texture_region(list, dst, dst_x, dst_y, dst_z, src, src_box);
   return;

}

static void STDMETHODCALLTYPE
d12_list_CopyTiles(ID3D12GraphicsCommandList6 *iface,
                   ID3D12Resource *tiled_resource_iface,
                   const D3D12_TILED_RESOURCE_COORDINATE *start,
                   const D3D12_TILE_REGION_SIZE *size,
                   ID3D12Resource *buffer_iface, UINT64 buffer_offset_B,
                   D3D12_TILE_COPY_FLAGS flags)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   if (list->state != D12_LIST_RECORDING)
      return;

   const D3D12_TILE_COPY_FLAGS allowed =
      D3D12_TILE_COPY_FLAG_NO_HAZARD |
      D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE |
      D3D12_TILE_COPY_FLAG_SWIZZLED_TILED_RESOURCE_TO_LINEAR_BUFFER;
   const bool to_tiled =
      flags & D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE;
   const bool explicit_to_buffer =
      flags & D3D12_TILE_COPY_FLAG_SWIZZLED_TILED_RESOURCE_TO_LINEAR_BUFFER;

   if (!tiled_resource_iface || !buffer_iface || !start || !size ||
       (flags & ~allowed) || (to_tiled && explicit_to_buffer)) {
      mesa_loge("CopyTiles: invalid arguments or flags %#x", (unsigned)flags);
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }

   /* NONE has the API-defined tiled-to-linear direction. */
   const bool to_buffer = !to_tiled;
   struct d12_resource *tiled =
      d12_resource_from_iface(tiled_resource_iface);
   struct d12_resource *buffer = d12_resource_from_iface(buffer_iface);
   if (!tiled->reserved ||
       buffer->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER ||
       buffer->reserved || !buffer->bo) {
      mesa_loge("CopyTiles: expected one reserved resource and one ordinary "
                "buffer");
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }

   uint32_t *tiles = NULL;
   uint32_t tile_count = 0;
   HRESULT hr = d12_sparse_collect_tiles(tiled, 1, start, size, &tiles,
                                         &tile_count);
   const uint64_t tile_B = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
   uint64_t copy_size_B = (uint64_t)tile_count * tile_B;
   if (FAILED(hr) || buffer_offset_B > buffer->desc.Width ||
       copy_size_B > buffer->desc.Width - buffer_offset_B) {
      mesa_loge("CopyTiles: tile region or linear buffer range is invalid");
      free(tiles);
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }

   d12_list_track_read_resource(list, tiled);
   d12_list_track_read_resource(list, buffer);

   enum pipe_format format = d12_format_to_pipe(tiled->desc.Format);
   for (uint32_t i = 0; i < tile_count; i++) {
      struct d12_sparse_tile_copy_info info;
      if (!d12_sparse_get_tile_copy_info(tiled, tiles[i], &info)) {
         list->state = D12_LIST_ERROR;
         list->record_error = E_INVALIDARG;
         break;
      }

      uint64_t linear = buffer->addr + buffer_offset_B + (uint64_t)i * tile_B;
      if (d12_debug()->dump_sparse) {
         simple_mtx_lock(&tiled->sparse_lock);
         struct d12_sparse_mapping mapping = tiled->sparse_mappings[tiles[i]];
         simple_mtx_unlock(&tiled->sparse_lock);
         mesa_logi("CopyTiles tile=%u direction=%s linear=0x%llx "
                   "mapped-bo=%u heap=0x%llx standard=%u",
                   tiles[i], to_buffer ? "tiled-to-linear" : "linear-to-tiled",
                   (unsigned long long)linear,
                   mapping.bo ? mapping.bo->uapi_handle : 0,
                   (unsigned long long)mapping.heap_offset_B, info.standard);
      }
      if (tiled->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ||
          !info.standard) {
         uint64_t remaining = tiled->sparse_size_B - info.offset_B;
         uint64_t bytes = MIN2(tile_B, remaining);
         struct libagx_copy_uchar_args args = {
            .dest = to_buffer ? linear : tiled->addr + info.offset_B,
            .src = to_buffer
                      ? d12_sparse_read_address(
                           list->device, tiled->addr + info.offset_B)
                      : linear,
         };
         d12_list_dispatch_precomp(list, LIBAGX_COPY_UCHAR, &args,
                                   sizeof(args), agx_1d(bytes));
         if (list->state == D12_LIST_ERROR)
            break;
         continue;
      }

      if (format == PIPE_FORMAT_NONE ||
          info.subresource >= tiled->sparse_subresource_count) {
         mesa_loge("CopyTiles: unsupported sparse texture format/subresource");
         list->state = D12_LIST_ERROR;
         list->record_error = E_INVALIDARG;
         break;
      }

      const unsigned level = info.subresource % tiled->sparse_mip_count;
      const uint32_t mip_width = u_minify(tiled->layout.width_px, level);
      const uint32_t mip_height = u_minify(tiled->layout.height_px, level);
      const uint32_t tile_width = tiled->sparse_tile_shape.WidthInTexels;
      const uint32_t tile_height = tiled->sparse_tile_shape.HeightInTexels;
      const uint32_t x = info.x * tile_width;
      const uint32_t y = info.y * tile_height;
      if (x >= mip_width || y >= mip_height) {
         mesa_loge("CopyTiles: sparse texture tile lies outside its mip");
         list->state = D12_LIST_ERROR;
         list->record_error = E_INVALIDARG;
         break;
      }

      const uint32_t width = MIN2(tile_width, mip_width - x);
      const uint32_t height = MIN2(tile_height, mip_height - y);
      const uint32_t row_pitch =
         util_format_get_nblocksx(format, tile_width) *
         util_format_get_blocksize(format);
      if (d12_debug()->dump_sparse) {
         mesa_logi("CopyTiles texture subresource=%u tile-xy=%u,%u "
                   "rect=%u,%u+%ux%u row-pitch=%u",
                   info.subresource, info.x, info.y, x, y, width, height,
                   row_pitch);
      }
      D3D12_TEXTURE_COPY_LOCATION texture = {
         .pResource = tiled_resource_iface,
         .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
         .SubresourceIndex = info.subresource,
      };
      D3D12_TEXTURE_COPY_LOCATION linear_tile = {
         .pResource = buffer_iface,
         .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
         .PlacedFootprint = {
            .Offset = buffer_offset_B + (uint64_t)i * tile_B,
            .Footprint = {
               .Format = tiled->desc.Format,
               .Width = width,
               .Height = height,
               .Depth = 1,
               .RowPitch = row_pitch,
            },
         },
      };

      if (to_tiled) {
         D3D12_BOX box = {0, 0, 0, width, height, 1};
         d12_copy_texture_region(list, &texture, x, y, 0, &linear_tile,
                                 &box);
      } else {
         D3D12_BOX box = {x, y, 0, x + width, y + height, 1};
         d12_copy_texture_region(list, &linear_tile, 0, 0, 0, &texture,
                                 &box);
      }
      if (list->state == D12_LIST_ERROR)
         break;
   }

   free(tiles);
}

static void STDMETHODCALLTYPE
d12_list_DiscardResource(ID3D12GraphicsCommandList6 *iface,
                         ID3D12Resource *resource_iface,
                         const D3D12_DISCARD_REGION *region)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->state != D12_LIST_RECORDING)
      return;

   if (!resource_iface) {
      mesa_loge("DiscardResource: resource is NULL");
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }

   struct d12_resource *resource = d12_resource_from_iface(resource_iface);
   enum d12_discard_dimension dimension;
   switch (resource->desc.Dimension) {
   case D3D12_RESOURCE_DIMENSION_BUFFER:
      dimension = D12_DISCARD_DIMENSION_BUFFER;
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE1D:
      dimension = D12_DISCARD_DIMENSION_TEXTURE1D;
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE2D:
      dimension = D12_DISCARD_DIMENSION_TEXTURE2D;
      break;
   case D3D12_RESOURCE_DIMENSION_TEXTURE3D:
      dimension = D12_DISCARD_DIMENSION_TEXTURE3D;
      break;
   default:
      mesa_loge("DiscardResource: invalid resource dimension");
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }

   const struct d12_discard_resource_info info = {
      .dimension = dimension,
      .width = resource->desc.Width,
      .height = resource->desc.Height,
      .mip_levels = resource->desc.MipLevels,
      .array_size = resource->desc.DepthOrArraySize,
      .plane_count = resource->stencil_bo ? 2 : 1,
      .allow_unordered_access =
         resource->desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
   };
   struct d12_discard_region_info region_info;
   const struct d12_discard_region_info *region_info_ptr = NULL;
   if (region) {
      region_info = (struct d12_discard_region_info){
         .rectangle_count = region->NumRects,
         .rectangles_present = region->pRects != NULL,
         .first_subresource = region->FirstSubresource,
         .subresource_count = region->NumSubresources,
      };
      region_info_ptr = &region_info;
   }

   if (resource->device != list->device) {
      mesa_loge("DiscardResource: resource belongs to another device");
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }

   const enum d12_discard_validation_result result = d12_discard_validate(
      list->type == D3D12_COMMAND_LIST_TYPE_DIRECT,
      list->type == D3D12_COMMAND_LIST_TYPE_COMPUTE, &info, region_info_ptr);
   if (result != D12_DISCARD_VALID) {
      mesa_loge("DiscardResource: %s",
                result == D12_DISCARD_UNSUPPORTED_COMMAND_LIST
                   ? "command-list type is unsupported"
                   : "invalid resource or region");
      list->state = D12_LIST_ERROR;
      list->record_error = result == D12_DISCARD_UNSUPPORTED_COMMAND_LIST
                              ? E_NOTIMPL
                              : E_INVALIDARG;
      return;
   }

   /* Discard makes contents undefined; it does not require them to change.
    * Keeping the allocation and its metadata is therefore the conservative
    * implementation until AGX render-pass load elimination consumes this
    * hint. No stream address is recorded, so retaining the resource here is
    * unnecessary.
    *
    * Direct3D also requires RT/DS/UAV resources to be in the corresponding
    * legacy resource state. This command core does not track legacy states;
    * that API precondition remains the application's responsibility and is
    * diagnosed by the D3D12 debug layer when enabled. */
}

static void STDMETHODCALLTYPE
d12_list_ResourceBarrier(ID3D12GraphicsCommandList6 *iface, UINT num_barriers,
                         const D3D12_RESOURCE_BARRIER *barriers)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   /* Memory is coherent UMA and segments execute in order via the
    * submit-header barrier fields, so a barrier has one effect on the
    * stream: the cache flush the open compute segment's last API dispatch
    * deferred (d12_list_flush_cdm_pending) is emitted here, whatever the
    * barrier's type. State transitions are otherwise accepted as no-ops
    * until image layout transitions matter.
    */
   if (list->state != D12_LIST_RECORDING)
      return;
   if (num_barriers && !d12_list_flush_cdm_pending(list))
      return;
   for (UINT i = 0; i < num_barriers; i++) {
      if (barriers[i].Type != D3D12_RESOURCE_BARRIER_TYPE_TRANSITION &&
          barriers[i].Type != D3D12_RESOURCE_BARRIER_TYPE_UAV) {
         mesa_logw("ResourceBarrier: type %d ignored",
                   (int)barriers[i].Type);
      }
   }
}

/*
 * D3D12's integer UAV clear does not numerically convert through the view's
 * type.  It truncates each UINT component to the corresponding storage
 * channel.  That distinction matters for views such as R16G16_FLOAT: a UINT
 * clear of {0x1234, 0xabcd} writes 0xabcd1234, whereas a float pack would
 * reinterpret those inputs as tiny floating-point values and write zeroes.
 *
 * Keep the table deliberately limited to the four-byte buffer formats the
 * fill kernel below can write exactly.  Unsupported formats fail recording
 * instead of silently producing a differently converted bit pattern.
 */
static bool
d12_pack_clear_uav_uint(DXGI_FORMAT format, const UINT values[4],
                        void *packed, unsigned *packed_size_B)
{
   uint32_t word;

   switch (format) {
   case DXGI_FORMAT_R32_TYPELESS:
   case DXGI_FORMAT_R32_FLOAT:
   case DXGI_FORMAT_R32_UINT:
   case DXGI_FORMAT_R32_SINT:
      memcpy(packed, &values[0], sizeof(uint32_t));
      *packed_size_B = sizeof(uint32_t);
      return true;

   case DXGI_FORMAT_R16G16_FLOAT:
   case DXGI_FORMAT_R16G16_UNORM:
   case DXGI_FORMAT_R16G16_UINT:
   case DXGI_FORMAT_R16G16_SNORM:
   case DXGI_FORMAT_R16G16_SINT:
      word = (values[0] & 0xffffu) | ((values[1] & 0xffffu) << 16);
      memcpy(packed, &word, sizeof(word));
      *packed_size_B = sizeof(uint32_t);
      return true;

   case DXGI_FORMAT_R16G16B16A16_FLOAT:
   case DXGI_FORMAT_R16G16B16A16_UNORM:
   case DXGI_FORMAT_R16G16B16A16_UINT:
   case DXGI_FORMAT_R16G16B16A16_SNORM:
   case DXGI_FORMAT_R16G16B16A16_SINT: {
      uint16_t words[4];
      for (unsigned i = 0; i < ARRAY_SIZE(words); i++)
         words[i] = values[i];
      memcpy(packed, words, sizeof(words));
      *packed_size_B = sizeof(words);
      return true;
   }

   case DXGI_FORMAT_R8G8B8A8_UNORM:
   case DXGI_FORMAT_R8G8B8A8_UINT:
   case DXGI_FORMAT_R8G8B8A8_SNORM:
   case DXGI_FORMAT_R8G8B8A8_SINT:
      word = (values[0] & 0xffu) | ((values[1] & 0xffu) << 8) |
             ((values[2] & 0xffu) << 16) |
             ((values[3] & 0xffu) << 24);
      memcpy(packed, &word, sizeof(word));
      *packed_size_B = sizeof(uint32_t);
      return true;

   case DXGI_FORMAT_R11G11B10_FLOAT:
      word = (values[0] & 0x7ffu) | ((values[1] & 0x7ffu) << 11) |
             ((values[2] & 0x3ffu) << 22);
      memcpy(packed, &word, sizeof(word));
      *packed_size_B = sizeof(uint32_t);
      return true;

   case DXGI_FORMAT_R10G10B10A2_UNORM:
   case DXGI_FORMAT_R10G10B10A2_UINT:
      word = (values[0] & 0x3ffu) | ((values[1] & 0x3ffu) << 10) |
             ((values[2] & 0x3ffu) << 20) |
             ((values[3] & 0x3u) << 30);
      memcpy(packed, &word, sizeof(word));
      *packed_size_B = sizeof(uint32_t);
      return true;

   default: {
      enum pipe_format pipe = d12_format_to_pipe(format);
      if (pipe == PIPE_FORMAT_NONE ||
          (!util_format_is_pure_uint(pipe) &&
           !util_format_is_pure_sint(pipe)))
         return false;

      unsigned blocksize_B = util_format_get_blocksize(pipe);
      if (!blocksize_B || blocksize_B > 16)
         return false;

      util_format_pack_rgba(pipe, packed, values, 1);
      *packed_size_B = blocksize_B;
      return true;
   }
   }
}

static bool
d12_pack_clear_uav_float(DXGI_FORMAT format, const FLOAT values[4],
                         void *packed, unsigned *packed_size_B)
{
   enum pipe_format pipe = d12_format_to_pipe(format);

   if (pipe == PIPE_FORMAT_NONE)
      return false;

   unsigned blocksize_B = util_format_get_blocksize(pipe);
   if (!blocksize_B || blocksize_B > 16)
      return false;

   util_format_pack_rgba(pipe, packed, values, 1);
   *packed_size_B = blocksize_B;
   return true;
}

static void
d12_list_clear_uav_buffer(struct d12_cmd_list *list,
                          D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle,
                          D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle,
                          ID3D12Resource *resource, const void *packed,
                          unsigned packed_size_B,
                          UINT num_rects, const D3D12_RECT *rects)
{
   if (list->state != D12_LIST_RECORDING)
      return;

   /*
    * Hardware implementations generally treat the CPU descriptor as
    * authoritative for ClearUAV; the API's GPU handle is an opaque companion
    * used to make the descriptor shader-visible.  This driver flattens GPU
    * descriptors on the CPU and never dereferences one on the GPU, so using
    * the CPU view is both the native model and avoids trusting an arbitrary
    * GPU token as a host pointer.
    */
   (void)gpu_handle;

   struct d12_descriptor *view = (struct d12_descriptor *)cpu_handle.ptr;
   struct d12_resource *res =
      resource ? d12_resource_from_iface(resource) : NULL;

   if (!view || !res || view->kind != D12_DESC_UAV ||
       view->resource != res ||
       res->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
      mesa_loge("ClearUnorderedAccessView: invalid buffer UAV");
      list->state = D12_LIST_ERROR;
      return;
   }

   if (num_rects && !rects) {
      mesa_loge("ClearUnorderedAccessView: missing buffer clear rectangles");
      list->state = D12_LIST_ERROR;
      return;
   }

   const uint64_t view_size_B = d12_buffer_view_size_B(
      view->buffer_count, view->buffer_stride_B);
   if (!packed_size_B || packed_size_B > 16 ||
       view->address % packed_size_B || view_size_B % packed_size_B) {
      mesa_loge("ClearUnorderedAccessView: buffer view has invalid element "
                "alignment");
      list->state = D12_LIST_ERROR;
      return;
   }

   if (!view->buffer_count)
      return;

   const uint32_t view_elements = view->buffer_count;
   uint64_t value = 0;
   if (packed_size_B != sizeof(uint32_t)) {
      value = d12_list_upload(list, packed, packed_size_B, 16);
      if (!value)
         return;
   }

   const unsigned count = num_rects ? num_rects : 1;
   for (unsigned i = 0; i < count; i++) {
      int64_t left = 0, right = view_elements;
      if (num_rects) {
         /* A buffer UAV is a one-row image whose X coordinate is the
          * element index. Rectangles not intersecting row zero are empty. */
         if (rects[i].top > 0 || rects[i].bottom <= 0)
            continue;
         left = MAX2((int64_t)rects[i].left, 0);
         right = MIN2((int64_t)rects[i].right, (int64_t)view_elements);
      }

      if (right <= left)
         continue;

      const uint32_t elements = right - left;
      const uint64_t address = view->address + left * packed_size_B;
      if (packed_size_B == sizeof(uint32_t)) {
         uint32_t packed_u32;
         memcpy(&packed_u32, packed, sizeof(packed_u32));
         struct libagx_fill_args args = {
            .address = address,
            .value = packed_u32,
         };
         d12_list_dispatch_precomp(list, LIBAGX_FILL, &args, sizeof(args),
                                   agx_1d(elements));
      } else {
         struct libagx_fill_image_args args = {
            .dest = address,
            .value = value,
            .width_el = elements,
            .height_el = 1,
            .dest_pitch_B = elements * packed_size_B,
            .dest_stride_el = elements,
            .dest_tile_width_el = 1,
            .dest_tile_height_el = 1,
            .blocksize_B = packed_size_B,
            .dest_tiled = false,
         };
         d12_list_dispatch_precomp(list, LIBAGX_FILL_IMAGE, &args,
                                   sizeof(args), agx_1d(elements));
      }

      if (list->state == D12_LIST_ERROR)
         return;
   }
}

static void
d12_list_clear_uav_texture(struct d12_cmd_list *list,
                           D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle,
                           D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle,
                           ID3D12Resource *resource, const void *packed,
                           unsigned packed_size_B,
                           UINT num_rects, const D3D12_RECT *rects)
{
   /* As with buffer clears, the CPU descriptor is authoritative. The GPU
    * handle is the shader-visible companion and need not be dereferenced by
    * this command-recording implementation. */
   (void)gpu_handle;

   struct d12_descriptor *view = (struct d12_descriptor *)cpu_handle.ptr;
   struct d12_resource *res =
      resource ? d12_resource_from_iface(resource) : NULL;

   const bool is_3d =
      res && res->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
   if (!view || !res || view->kind != D12_DESC_UAV ||
       view->resource != res ||
       (res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D &&
        res->desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
        !is_3d) ||
       (num_rects && !rects)) {
      mesa_loge("ClearUnorderedAccessView: invalid texture UAV");
      list->state = D12_LIST_ERROR;
      return;
   }

   const unsigned mips = res->layout.levels;
   const uint64_t layer_stride = res->layout.layer_stride_B;
   const unsigned level = view->texture_level;
   if (view->address < res->addr || !layer_stride || level >= mips) {
      mesa_loge("ClearUnorderedAccessView: invalid texture layer address");
      list->state = D12_LIST_ERROR;
      return;
   }

   uint64_t view_offset_B = view->address - res->addr;
   if (is_3d) {
      const uint64_t level_offset_B =
         ail_get_level_offset_B(&res->layout, level);
      if (view_offset_B < level_offset_B) {
         mesa_loge("ClearUnorderedAccessView: invalid 3D mip address");
         list->state = D12_LIST_ERROR;
         return;
      }
      view_offset_B -= level_offset_B;
   }
   if (view_offset_B % layer_stride) {
      mesa_loge("ClearUnorderedAccessView: misaligned texture layer address");
      list->state = D12_LIST_ERROR;
      return;
   }

   const uint64_t first_layer64 = view_offset_B / layer_stride;
   const unsigned resource_layers =
      is_3d ? u_minify(res->layout.depth_px, level)
            : res->layout.depth_px;
   const unsigned layers = MAX2(view->texture_layers, 1);
   if (first_layer64 >= resource_layers ||
       layers > resource_layers - first_layer64) {
      mesa_loge("ClearUnorderedAccessView: texture range is out of bounds");
      list->state = D12_LIST_ERROR;
      return;
   }

   enum pipe_format view_format = d12_format_to_pipe(view->format);
   if (view_format == PIPE_FORMAT_NONE ||
       util_format_get_blocksize(view_format) != packed_size_B ||
       util_format_get_blockwidth(view_format) != 1 ||
       util_format_get_blockheight(view_format) != 1) {
      mesa_loge("ClearUnorderedAccessView: texture format %u unsupported",
                (unsigned)view->format);
      list->state = D12_LIST_ERROR;
      return;
   }

   uint64_t value = d12_list_upload(list, packed, packed_size_B, 16);
   if (!value)
      return;

   for (unsigned layer = 0; layer < layers; layer++) {
      const unsigned subresource = is_3d
         ? level
         : level + ((unsigned)first_layer64 + layer) * mips;
      struct d12_copy_surface surface;
      if (!d12_copy_surface_from_texture(
             &surface, res, subresource, "UAV-clear destination")) {
         list->state = D12_LIST_ERROR;
         return;
      }
      if (is_3d) {
         const uint64_t slice = first_layer64 + layer;
         surface.base += slice * layer_stride;
         surface.storage_offset_B += slice * layer_stride;
         surface.depth_px = 1;
         surface.size_B = ail_get_level_size_B(&res->layout, level);
      }
      if (!d12_copy_surface_check_storage(&surface,
                                          "UAV-clear destination") ||
          surface.sample_count != 1 ||
          surface.blocksize_B != packed_size_B ||
          surface.block_width_px != 1 || surface.block_height_px != 1) {
         mesa_loge("ClearUnorderedAccessView: texture surface unsupported");
         list->state = D12_LIST_ERROR;
         return;
      }

      const unsigned count = num_rects ? num_rects : 1;
      for (unsigned i = 0; i < count; i++) {
         int64_t left = 0, top = 0;
         int64_t right = surface.width_px, bottom = surface.height_px;
         if (num_rects) {
            left = MAX2((int64_t)rects[i].left, 0);
            top = MAX2((int64_t)rects[i].top, 0);
            right = MIN2((int64_t)rects[i].right,
                         (int64_t)surface.width_px);
            bottom = MIN2((int64_t)rects[i].bottom,
                          (int64_t)surface.height_px);
         }

         if (right <= left || bottom <= top)
            continue;

         const uint32_t width = right - left;
         const uint32_t height = bottom - top;
         struct libagx_fill_image_args args = {
            .dest = surface.base,
            .value = value,
            .width_el = width,
            .height_el = height,
            .dest_x_el = left,
            .dest_y_el = top,
            .dest_pitch_B = surface.pitch_B,
            .dest_stride_el = surface.stride_el,
            .dest_tile_width_el = surface.tile_width_el,
            .dest_tile_height_el = surface.tile_height_el,
            .blocksize_B = surface.blocksize_B,
            .dest_tiled = surface.tiled,
         };
         d12_list_dispatch_precomp(list, LIBAGX_FILL_IMAGE, &args,
                                   sizeof(args),
                                   agx_1d((uint64_t)width * height));
         if (list->state == D12_LIST_ERROR)
            return;
      }
   }
}

static void
d12_list_clear_uav(struct d12_cmd_list *list,
                   D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle,
                   D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle,
                   ID3D12Resource *resource, const void *packed,
                   unsigned packed_size_B, UINT num_rects,
                   const D3D12_RECT *rects)
{
   struct d12_resource *res =
      resource ? d12_resource_from_iface(resource) : NULL;

   d12_list_track_read_resource(list, res);

   if (res && res->desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
      d12_list_clear_uav_texture(list, gpu_handle, cpu_handle, resource,
                                 packed, packed_size_B, num_rects, rects);
   } else {
      d12_list_clear_uav_buffer(list, gpu_handle, cpu_handle, resource,
                                packed, packed_size_B, num_rects, rects);
   }
}

static void STDMETHODCALLTYPE
d12_list_ClearUnorderedAccessViewUint(
   ID3D12GraphicsCommandList6 *iface, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle,
   D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, ID3D12Resource *resource,
   const UINT values[4], UINT num_rects, const D3D12_RECT *rects)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   struct d12_descriptor *view = (struct d12_descriptor *)cpu_handle.ptr;
   uint8_t packed[16] = {0};
   unsigned packed_size_B;

   if (list->state != D12_LIST_RECORDING)
      return;

   if (!values || !view ||
       !d12_pack_clear_uav_uint(view->format, values, packed,
                                &packed_size_B)) {
      mesa_loge("ClearUnorderedAccessViewUint: unsupported or invalid view");
      list->state = D12_LIST_ERROR;
      return;
   }

   d12_list_clear_uav(list, gpu_handle, cpu_handle, resource, packed,
                      packed_size_B,
                      num_rects, rects);
}

static void STDMETHODCALLTYPE
d12_list_ClearUnorderedAccessViewFloat(
   ID3D12GraphicsCommandList6 *iface, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle,
   D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, ID3D12Resource *resource,
   const FLOAT values[4], UINT num_rects, const D3D12_RECT *rects)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   struct d12_descriptor *view = (struct d12_descriptor *)cpu_handle.ptr;
   uint8_t packed[16] = {0};
   unsigned packed_size_B;

   if (list->state != D12_LIST_RECORDING)
      return;

   if (!values || !view ||
       !d12_pack_clear_uav_float(view->format, values, packed,
                                 &packed_size_B)) {
      mesa_loge("ClearUnorderedAccessViewFloat: unsupported or invalid view");
      list->state = D12_LIST_ERROR;
      return;
   }

   d12_list_clear_uav(list, gpu_handle, cpu_handle, resource, packed,
                      packed_size_B,
                      num_rects, rects);
}

/*
 * Attach a depth/stencil resource to the open render pass.
 *
 * Depth never goes through a background/end-of-tile USC program the way
 * colour does: the fixed-function ZLS unit loads and stores it, configured
 * from the render command. So attaching is bookkeeping here, and the actual
 * hardware setup happens in d12_fill_render_cmd.
 */
static bool
d12_list_attach_ds(struct d12_cmd_list *list,
                   const struct d12_descriptor *dsv)
{
   if (list->device->dev.chip == AGX_CHIP_G17P &&
       !d12_g17p_depth_enabled(&list->device->dev)) {
      mesa_loge("native depth attach/clear producer remains closed");
      list->state = D12_LIST_ERROR;
      return false;
   }

   struct d12_resource *ds = dsv->resource;
   if (!list->render_open) {
      mesa_loge("depth/stencil needs an open render pass; clear the colour "
                "target first (depth-only passes are not supported yet)");
      list->state = D12_LIST_ERROR;
      return false;
   }

   struct d12_segment *seg = &list->segments[list->segment_count - 1];

   if (seg->ds && seg->ds != ds) {
      mesa_loge("two different depth/stencil targets in one render pass");
      list->state = D12_LIST_ERROR;
      return false;
   }

   /* The ZLS unit addresses one buffer for the whole pass. Clamp a colour
    * clear pass to the DSV intersection before any draw is emitted, so the
    * uncovered region can never test outside the smaller allocation. */
   const unsigned level = dsv->texture_plane;
   const unsigned width = u_minify(ds->layout.width_px, level);
   const unsigned height = u_minify(ds->layout.height_px, level);
   seg->width_px = MIN2(seg->width_px, width);
   seg->height_px = MIN2(seg->height_px, height);

   seg->ds = ds;
   d12_list_track_read_resource(list, ds);
   seg->ds_address = dsv->address;
   seg->stencil_address = dsv->counter_address;
   seg->ds_level = level;
   seg->dsv_flags = dsv->dsv_flags;
   const unsigned ds_layers = MAX2(dsv->size_B, 1);
   if (seg->layers != ds_layers) {
      mesa_loge("render pass and depth/stencil have mismatched layer counts "
                "%u and %u", seg->layers, ds_layers);
      list->state = D12_LIST_ERROR;
      return false;
   }
   if (list->device->dev.chip == AGX_CHIP_G17P &&
       !d12_g17p_depth_segment_supported(seg)) {
      mesa_loge("native D32 attachment rejected storage/view/extent metadata");
      list->state = D12_LIST_ERROR;
      return false;
   }
   return true;
}

static bool
d12_list_clear_surface_rects(struct d12_cmd_list *list,
                             const struct d12_copy_surface *surface,
                             const void *packed, UINT num_rects,
                             const D3D12_RECT *rects, const char *name)
{
   if (!rects || surface->block_width_px != 1 ||
       surface->block_height_px != 1 || surface->blocksize_B > 16) {
      mesa_loge("%s: invalid rectangle or surface format", name);
      list->state = D12_LIST_ERROR;
      return false;
   }

   if (!surface->sample_count ||
       surface->blocksize_B % surface->sample_count != 0) {
      mesa_loge("%s: invalid multisample block shape", name);
      list->state = D12_LIST_ERROR;
      return false;
   }

   const unsigned sample_size_B =
      surface->blocksize_B / surface->sample_count;
   uint8_t value_pattern[16] = {0};
   if (sample_size_B > sizeof(value_pattern)) {
      mesa_loge("%s: sample value is too large", name);
      list->state = D12_LIST_ERROR;
      return false;
   }
   for (unsigned s = 0; s < surface->sample_count; ++s)
      memcpy(value_pattern + s * sample_size_B, packed, sample_size_B);

   uint64_t value = d12_list_upload(list, value_pattern,
                                    surface->blocksize_B, 16);
   if (!value)
      return false;

   for (unsigned i = 0; i < num_rects; ++i) {
      const int64_t left = MAX2((int64_t)rects[i].left, 0);
      const int64_t top = MAX2((int64_t)rects[i].top, 0);
      const int64_t right =
         MIN2((int64_t)rects[i].right, (int64_t)surface->width_px);
      const int64_t bottom =
         MIN2((int64_t)rects[i].bottom, (int64_t)surface->height_px);
      if (right <= left || bottom <= top)
         continue;

      const uint32_t width = right - left;
      const uint32_t height = bottom - top;
      struct libagx_fill_image_args args = {
         .dest = surface->base,
         .value = value,
         .width_el = width,
         .height_el = height,
         .dest_x_el = left,
         .dest_y_el = top,
         .dest_pitch_B = surface->pitch_B,
         .dest_stride_el = surface->stride_el,
         .dest_tile_width_el = surface->tile_width_el,
         .dest_tile_height_el = surface->tile_height_el,
         .blocksize_B = surface->blocksize_B,
         .dest_tiled = surface->tiled,
      };
      d12_list_dispatch_precomp(list, LIBAGX_FILL_IMAGE, &args, sizeof(args),
                                agx_1d((uint64_t)width * height));
      if (list->state == D12_LIST_ERROR)
         return false;
   }

   return true;
}

static void STDMETHODCALLTYPE
d12_list_ClearDepthStencilView(ID3D12GraphicsCommandList6 *iface,
                               D3D12_CPU_DESCRIPTOR_HANDLE dsv_handle,
                               D3D12_CLEAR_FLAGS flags, FLOAT depth,
                               UINT8 stencil, UINT num_rects,
                               const D3D12_RECT *rects)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->state != D12_LIST_RECORDING)
      return;

   struct d12_rtv *dsv = (struct d12_rtv *)dsv_handle.ptr;
   if (!dsv || !dsv->resource || dsv->kind != D12_DESC_DSV) {
      mesa_loge("ClearDepthStencilView: invalid DSV handle");
      list->state = D12_LIST_ERROR;
      return;
   }

   struct d12_resource *ds = dsv->resource;

   if (num_rects != 0) {
      if (flags & D3D12_CLEAR_FLAG_DEPTH) {
         struct d12_copy_surface surface;
         if (!d12_copy_surface_from_texture(&surface, ds,
                                            dsv->texture_plane,
                                            "partial depth clear")) {
            list->state = D12_LIST_ERROR;
            return;
         }

         uint32_t packed = ds->layout.format == PIPE_FORMAT_Z16_UNORM
                              ? _mesa_float_to_unorm(depth, 16)
                              : fui(depth);
         for (unsigned layer = 0; layer < MAX2(dsv->size_B, 1); ++layer) {
            surface.base = dsv->address +
               (uint64_t)layer * ds->layout.layer_stride_B;
            surface.storage_offset_B = surface.reserved
               ? surface.base - ds->addr
               : surface.base - surface.bo->va->addr;
            if (!d12_copy_surface_check_storage(&surface,
                                                "partial depth clear") ||
                !d12_list_clear_surface_rects(list, &surface, &packed,
                                              num_rects, rects,
                                              "depth clear"))
               return;
         }
      }

      /* A combined DEPTH|STENCIL clear is legal through a depth-only DSV;
       * the absent aspect is ignored. */
      if ((flags & D3D12_CLEAR_FLAG_STENCIL) && ds->stencil_bo) {
         struct d12_copy_surface surface;
         const unsigned plane_subresource =
            ds->layout.levels * MAX2(ds->desc.DepthOrArraySize, 1) +
            dsv->texture_plane;
         if (!d12_copy_surface_from_texture(&surface, ds,
                                            plane_subresource,
                                            "partial stencil clear")) {
            list->state = D12_LIST_ERROR;
            return;
         }
         for (unsigned layer = 0; layer < MAX2(dsv->size_B, 1); ++layer) {
            surface.base = dsv->counter_address +
               (uint64_t)layer * ds->stencil_layout.layer_stride_B;
            surface.storage_offset_B =
               surface.base - surface.bo->va->addr;
            if (!d12_copy_surface_check_storage(&surface,
                                                "partial stencil clear") ||
                !d12_list_clear_surface_rects(list, &surface, &stencil,
                                              num_rects, rects,
                                              "stencil clear"))
               return;
         }
      }
      return;
   }

   if (list->render_open) {
      struct d12_segment *open =
         &list->segments[list->segment_count - 1];
      /* ZLS clear/load state is consumed at pass start. Mutating it after a
       * draw would reorder Draw->Clear into Clear->Draw. A clear view is
       * independent of current OM bindings, so a different slice/range (or
       * an incompatible colour-pass layer count) also needs its own pass.
       */
      if (open->has_draws || open->layers != MAX2(dsv->size_B, 1) ||
          (open->ds &&
           (open->ds != ds || open->ds_address != dsv->address ||
            open->stencil_address != dsv->counter_address ||
            open->dsv_flags != dsv->dsv_flags)))
         d12_list_close_cdm(list);
   }

   if (!list->render_open) {
      if (!d12_list_begin_render(list, NULL, 0, dsv, NULL,
                                 AGX_BG_EOT_NONE, NULL, true))
         return;
   } else if (!d12_list_attach_ds(list, dsv)) {
      return;
   }

   struct d12_segment *seg = &list->segments[list->segment_count - 1];
   seg->process_empty_tiles = true;

   /* The clear is not a draw and not a program: the hardware seeds every
    * tile's depth from ISP_BGOBJDEPTH, encoded in the depth plane's own
    * format, and its stencil from the low byte of ISP_BGOBJVALS.
    */
   if (flags & D3D12_CLEAR_FLAG_DEPTH) {
      if (ds->layout.format == PIPE_FORMAT_Z16_UNORM)
         seg->isp_bgobjdepth = _mesa_float_to_unorm(depth, 16);
      else
         seg->isp_bgobjdepth = fui(depth);

      seg->depth_cleared = true;
   }

   if ((flags & D3D12_CLEAR_FLAG_STENCIL) && ds->stencil_bo) {
      seg->stencil_clear = stencil;
      seg->stencil_cleared = true;
   }
}

static void STDMETHODCALLTYPE
d12_list_OMSetStencilRef(ID3D12GraphicsCommandList6 *iface, UINT ref)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   list->stencil_ref_front = (uint8_t)ref;
   list->stencil_ref_back = (uint8_t)ref;
   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_OM_STENCIL_REF, 0);
      if (cmd) {
         cmd->stencil_ref.front = ref;
         cmd->stencil_ref.back = ref;
      }
   }
}

static void STDMETHODCALLTYPE
d12_list_Barrier(ID3D12GraphicsCommandList9 *iface, UINT32 count,
                 const D3D12_BARRIER_GROUP *groups)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   (void)count;
   (void)groups;
   mesa_loge("ID3D12GraphicsCommandList7::Barrier is unsupported");
   list->state = D12_LIST_ERROR;
   list->record_error = E_NOTIMPL;
}

static void STDMETHODCALLTYPE
d12_list_OMSetFrontAndBackStencilRef(ID3D12GraphicsCommandList9 *iface,
                                     UINT front, UINT back)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   list->stencil_ref_front = (uint8_t)front;
   list->stencil_ref_back = (uint8_t)back;
   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_OM_STENCIL_REF, 0);
      if (cmd) {
         cmd->stencil_ref.front = front;
         cmd->stencil_ref.back = back;
      }
   }
}

static void STDMETHODCALLTYPE
d12_list_RSSetDepthBias(ID3D12GraphicsCommandList9 *iface, FLOAT depth_bias,
                        FLOAT clamp, FLOAT slope)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   list->depth_bias = depth_bias;
   list->depth_bias_clamp = clamp;
   list->slope_scaled_depth_bias = slope;
   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_RS_DEPTH_BIAS, 0);
      if (cmd) {
         cmd->depth_bias.bias = depth_bias;
         cmd->depth_bias.clamp = clamp;
         cmd->depth_bias.slope = slope;
      }
   }
}

static void STDMETHODCALLTYPE
d12_list_IASetIndexBufferStripCutValue(
   ID3D12GraphicsCommandList9 *iface,
   D3D12_INDEX_BUFFER_STRIP_CUT_VALUE strip_cut)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   if (strip_cut != D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED &&
       strip_cut != D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF &&
       strip_cut != D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF) {
      mesa_loge("IASetIndexBufferStripCutValue: invalid value %u",
                (unsigned)strip_cut);
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }
   list->strip_cut = strip_cut;
   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_IA_STRIP_CUT, 0);
      if (cmd)
         cmd->strip_cut = strip_cut;
   }
}

static void STDMETHODCALLTYPE
d12_list_OMSetBlendFactor(ID3D12GraphicsCommandList6 *iface,
                          const FLOAT factor[4])
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   static const float one[4] = {1.0f, 1.0f, 1.0f, 1.0f};
   memcpy(list->blend_factor, factor ? factor : one,
          sizeof(list->blend_factor));
   if (list->type == D3D12_COMMAND_LIST_TYPE_BUNDLE) {
      struct d12_bundle_cmd *cmd =
         d12_bundle_record(list, D12_BUNDLE_OM_BLEND_FACTOR, 0);
      if (cmd)
         memcpy(cmd->blend_factor, list->blend_factor,
                sizeof(cmd->blend_factor));
   }
}

static void STDMETHODCALLTYPE
d12_list_OMSetRenderTargets(ID3D12GraphicsCommandList6 *iface, UINT num_rtvs,
                            const D3D12_CPU_DESCRIPTOR_HANDLE *rtvs,
                            BOOL single_handle,
                            const D3D12_CPU_DESCRIPTOR_HANDLE *dsv)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->state != D12_LIST_RECORDING)
      return;

   memset(list->bound_rtv, 0, sizeof(list->bound_rtv));
   list->bound_rtv_count = 0;
   if (num_rtvs > AGX_MAX_RENDER_TARGETS || (num_rtvs && !rtvs)) {
      mesa_loge("OMSetRenderTargets: invalid RTV count %u", num_rtvs);
      list->state = D12_LIST_ERROR;
      return;
   }

   for (unsigned i = 0; i < num_rtvs; i++) {
      uintptr_t handle = single_handle
                            ? rtvs[0].ptr +
                                 i * sizeof(struct d12_descriptor)
                            : rtvs[i].ptr;
      const struct d12_rtv *rtv = (const struct d12_rtv *)handle;
      if (!rtv || !rtv->resource)
         continue;

      if (rtv->kind != D12_DESC_RTV) {
         mesa_loge("OMSetRenderTargets: handle %u is not a render-target "
                   "view", i);
         list->state = D12_LIST_ERROR;
         return;
      }

      list->bound_rtv[i] = *rtv;
   }
   list->bound_rtv_count = num_rtvs;

   memset(&list->bound_dsv, 0, sizeof(list->bound_dsv));
   list->has_bound_dsv = false;
   if (dsv) {
      struct d12_rtv *view = (struct d12_rtv *)dsv->ptr;
      if (view && view->resource && view->kind == D12_DESC_DSV) {
         list->bound_dsv = *view;
         list->has_bound_dsv = true;
      } else if (view && view->resource) {
         mesa_loge("OMSetRenderTargets: DSV handle is not a depth/stencil "
                   "view");
         list->state = D12_LIST_ERROR;
         return;
      }
   }

   /* Ending here makes attachment changes authoritative even if no draw
    * follows. The next draw opens a pass with the new descriptor snapshot.
    */
   if (list->render_open &&
       !d12_segment_matches_bindings(
          list, &list->segments[list->segment_count - 1], true))
      d12_list_close_cdm(list);
}

/*
 * Programmable sample positions.
 *
 * AGX already takes the pattern as data in PPP_MULTISAMPLECTL; the driver was
 * writing a fixed default into it. The register scope is one whole render,
 * which is exactly tier 1: one pattern for every pixel in the render target.
 */
bool
d12_cmd_list_set_sample_positions(struct d12_cmd_list *list,
                                  uint32_t samples_per_pixel,
                                  uint32_t pixel_count,
                                  const struct d12_sample_position *positions)
{
   uint32_t packed = 0;

   if (!list || list->state != D12_LIST_RECORDING)
      return false;

   /* The positions apply to a whole hardware render, so a change has to
    * start a new one rather than reinterpret work already recorded. */
   d12_list_close_cdm(list);

   /* Zero of both is D3D12's spelling for "back to the standard pattern". */
   if (!samples_per_pixel && !pixel_count) {
      list->sample_positions_set = false;
      list->sample_positions = 0;
      return true;
   }

   if (!d12_sample_positions_tier1_request_valid(samples_per_pixel,
                                                 pixel_count, 0) ||
       !positions ||
       !d12_sample_positions_pack(samples_per_pixel, positions, &packed)) {
      mesa_loge("SetSamplePositions: %u samples across %u pixels is not a "
                "pattern this adapter can program",
                samples_per_pixel, pixel_count);
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return false;
   }

   list->sample_positions = packed;
   list->sample_positions_set = true;
   return true;
}

/*
 * Render passes.
 *
 * OMSetRenderTargets binds attachments and says nothing about what happens
 * to them at the edges of the work, so the driver has to assume the worst:
 * seed every tile from memory on entry and write every tile back on exit.
 * On a tile-based deferred renderer that assumption is the expensive part.
 * A render pass replaces it with the application's own statement per
 * attachment, which is why DISCARD is not a hint here -- it removes a
 * memory round trip for the whole render target.
 *
 * The bindings are recorded exactly as OMSetRenderTargets records them so
 * every draw path downstream is unchanged; only the tile traffic differs.
 */
bool
d12_cmd_list_begin_render_pass(
   struct d12_cmd_list *list, unsigned rt_count,
   const struct d12_rtv *const *rtvs,
   const struct d12_render_pass_plan *rt_plans,
   const float (*rt_clears)[4], const struct d12_rtv *dsv,
   const struct d12_render_pass_plan *depth_plan,
   const struct d12_render_pass_plan *stencil_plan, float depth_clear,
   uint8_t stencil_clear)
{
   if (!list || list->state != D12_LIST_RECORDING)
      return false;

   if (list->render_pass_active) {
      mesa_loge("BeginRenderPass inside an open render pass");
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return false;
   }

   if (rt_count > AGX_MAX_RENDER_TARGETS ||
       (rt_count && (!rtvs || !rt_plans || !rt_clears)) ||
       (dsv && (!depth_plan || !stencil_plan))) {
      mesa_loge("BeginRenderPass: invalid attachment description");
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return false;
   }

   /* Whatever was bound before is not part of this pass. */
   d12_list_close_cdm(list);
   memset(list->bound_rtv, 0, sizeof(list->bound_rtv));
   list->bound_rtv_count = 0;
   memset(&list->bound_dsv, 0, sizeof(list->bound_dsv));
   list->has_bound_dsv = false;
   memset(list->render_pass_rt_plan, 0, sizeof(list->render_pass_rt_plan));
   memset(list->render_pass_rt_clear, 0, sizeof(list->render_pass_rt_clear));

   for (unsigned i = 0; i < rt_count; i++) {
      const struct d12_rtv *rtv = rtvs[i];

      /* A NO_ACCESS attachment is declared but untouched. It still occupies
       * its slot so later slots keep their register numbers. */
      if (rt_plans[i].no_access)
         continue;

      if (!rtv || !rtv->resource || rtv->kind != D12_DESC_RTV) {
         mesa_loge("BeginRenderPass: attachment %u is not a render-target "
                   "view", i);
         list->state = D12_LIST_ERROR;
         list->record_error = E_INVALIDARG;
         return false;
      }

      list->bound_rtv[i] = *rtv;
      list->render_pass_rt_plan[i] = rt_plans[i];
      memcpy(list->render_pass_rt_clear[i], rt_clears[i],
             sizeof(list->render_pass_rt_clear[i]));
   }
   list->bound_rtv_count = (uint8_t)rt_count;

   if (dsv && !(depth_plan->no_access && stencil_plan->no_access)) {
      if (!dsv->resource || dsv->kind != D12_DESC_DSV) {
         mesa_loge("BeginRenderPass: depth-stencil handle is not a "
                   "depth/stencil view");
         list->state = D12_LIST_ERROR;
         list->record_error = E_INVALIDARG;
         return false;
      }
      list->bound_dsv = *dsv;
      list->has_bound_dsv = true;
      list->render_pass_depth_plan = *depth_plan;
      list->render_pass_stencil_plan = *stencil_plan;
   } else {
      list->render_pass_depth_plan = (struct d12_render_pass_plan){0};
      list->render_pass_stencil_plan = (struct d12_render_pass_plan){0};
   }

   list->render_pass_depth_clear = depth_clear;
   list->render_pass_stencil_clear = stencil_clear;
   list->render_pass_active = true;
   return true;
}

bool
d12_cmd_list_end_render_pass(struct d12_cmd_list *list)
{
   if (!list || list->state != D12_LIST_RECORDING)
      return false;

   if (!list->render_pass_active) {
      mesa_loge("EndRenderPass without an open render pass");
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return false;
   }

   /* Close the hardware render here rather than at the next binding change:
    * the end-of-tile programs the pass asked for are only correct for the
    * work inside it. */
   d12_list_close_cdm(list);
   list->render_pass_active = false;
   memset(list->render_pass_rt_plan, 0, sizeof(list->render_pass_rt_plan));
   memset(list->render_pass_rt_clear, 0, sizeof(list->render_pass_rt_clear));
   list->render_pass_depth_plan = (struct d12_render_pass_plan){0};
   list->render_pass_stencil_plan = (struct d12_render_pass_plan){0};
   return true;
}

static void STDMETHODCALLTYPE
d12_list_SOSetTargets(ID3D12GraphicsCommandList6 *iface, UINT start_slot,
                      UINT num_views,
                      const D3D12_STREAM_OUTPUT_BUFFER_VIEW *views)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);

   if (list->state != D12_LIST_RECORDING)
      return;
   if (start_slot > ARRAY_SIZE(list->so) ||
       num_views > ARRAY_SIZE(list->so) - start_slot) {
      mesa_loge("SOSetTargets: slot range %u+%u exceeds four buffers",
                start_slot, num_views);
      list->state = D12_LIST_ERROR;
      return;
   }

   for (unsigned i = 0; i < num_views; i++) {
      D3D12_STREAM_OUTPUT_BUFFER_VIEW view = views ? views[i] :
                                                    (D3D12_STREAM_OUTPUT_BUFFER_VIEW){0};
      if (!view.SizeInBytes) {
         list->so[start_slot + i] =
            (D3D12_STREAM_OUTPUT_BUFFER_VIEW){0};
         continue;
      }
      if (!d12_stream_output_view_valid(
             view.BufferLocation, view.SizeInBytes,
             view.BufferFilledSizeLocation)) {
         mesa_loge("SOSetTargets: incomplete, misaligned, or oversized "
                   "buffer view %u", i);
         list->state = D12_LIST_ERROR;
         return;
      }
      list->so[start_slot + i] = view;
   }
}

/* The generated C interfaces repeat every inherited function pointer with a
 * newer This type. Keep the already-checked list-6 table as an exact prefix,
 * then append the list-7/8/9 slots. COM callers see the same flat pointer
 * array, while the compiler continues checking every existing callback
 * against the interface it was written for. */
struct d12_list9_vtbl {
   ID3D12GraphicsCommandList6Vtbl v6;
   void (STDMETHODCALLTYPE *Barrier)(ID3D12GraphicsCommandList9 *, UINT32,
                                     const D3D12_BARRIER_GROUP *);
   void (STDMETHODCALLTYPE *OMSetFrontAndBackStencilRef)(
      ID3D12GraphicsCommandList9 *, UINT, UINT);
   void (STDMETHODCALLTYPE *RSSetDepthBias)(ID3D12GraphicsCommandList9 *,
                                            FLOAT, FLOAT, FLOAT);
   void (STDMETHODCALLTYPE *IASetIndexBufferStripCutValue)(
      ID3D12GraphicsCommandList9 *, D3D12_INDEX_BUFFER_STRIP_CUT_VALUE);
};

static_assert(offsetof(ID3D12GraphicsCommandList9Vtbl, Barrier) ==
                 sizeof(ID3D12GraphicsCommandList6Vtbl),
              "command-list 6 must be an exact vtable prefix of list 9");
static_assert(sizeof(struct d12_list9_vtbl) ==
                 sizeof(ID3D12GraphicsCommandList9Vtbl),
              "private list-9 vtable must match the public ABI");

/* The public API spells the access types with its own enumeration; the
 * numeric values are the same as the DDI's, which the render-pass core is
 * written against and its host tests pin. */
static_assert((uint32_t)D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_DISCARD ==
                    D12_RP_BEGIN_DISCARD &&
                 (uint32_t)D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_PRESERVE ==
                    D12_RP_BEGIN_PRESERVE &&
                 (uint32_t)D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR ==
                    D12_RP_BEGIN_CLEAR &&
                 (uint32_t)D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_NO_ACCESS ==
                    D12_RP_BEGIN_NO_ACCESS,
              "render-pass beginning access ABI");
static_assert((uint32_t)D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_DISCARD ==
                    D12_RP_END_DISCARD &&
                 (uint32_t)D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE ==
                    D12_RP_END_PRESERVE &&
                 (uint32_t)D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_RESOLVE ==
                    D12_RP_END_RESOLVE &&
                 (uint32_t)D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_NO_ACCESS ==
                    D12_RP_END_NO_ACCESS,
              "render-pass ending access ABI");

static void STDMETHODCALLTYPE
d12_list_SetSamplePositions(ID3D12GraphicsCommandList6 *iface,
                            UINT samples_per_pixel, UINT pixels,
                            D3D12_SAMPLE_POSITION *positions)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   struct d12_sample_position local[D12_SAMPLE_POSITION_MAX_SAMPLES];
   const uint32_t count = MIN2(samples_per_pixel,
                               D12_SAMPLE_POSITION_MAX_SAMPLES);

   static_assert(sizeof(D3D12_SAMPLE_POSITION) ==
                    sizeof(struct d12_sample_position),
                 "sample position ABI");

   for (uint32_t i = 0; i < count && positions; ++i) {
      local[i].x = positions[i].X;
      local[i].y = positions[i].Y;
   }

   (void)d12_cmd_list_set_sample_positions(
      list, samples_per_pixel, pixels,
      positions && samples_per_pixel <= D12_SAMPLE_POSITION_MAX_SAMPLES
         ? local
         : NULL);
}

static void STDMETHODCALLTYPE
d12_list_BeginRenderPass(
   ID3D12GraphicsCommandList6 *iface, UINT rt_count,
   const D3D12_RENDER_PASS_RENDER_TARGET_DESC *rts,
   const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC *ds,
   D3D12_RENDER_PASS_FLAGS flags)
{
   struct d12_cmd_list *list = d12_cmd_list_from_iface(iface);
   const struct d12_rtv *rtvs[AGX_MAX_RENDER_TARGETS] = {0};
   struct d12_render_pass_plan plans[AGX_MAX_RENDER_TARGETS] = {0};
   float clears[AGX_MAX_RENDER_TARGETS][4] = {{0}};
   struct d12_render_pass_plan depth_plan = {0}, stencil_plan = {0};
   float depth_clear = 0.0f;
   uint8_t stencil_clear = 0;

   if (list->state != D12_LIST_RECORDING)
      return;

   /* SUSPENDING/RESUMING split one logical pass across command lists. The
    * access types already say PRESERVE at the seam, so the tile traffic is
    * correct without recognising the flags; nothing here keeps a tile
    * resident across a submit boundary anyway. UAV writes need no special
    * handling because a fragment shader writes them exactly as it does
    * outside a pass. */
   (void)flags;

   if (rt_count > AGX_MAX_RENDER_TARGETS || (rt_count && !rts)) {
      mesa_loge("BeginRenderPass: invalid render-target count %u", rt_count);
      list->state = D12_LIST_ERROR;
      list->record_error = E_INVALIDARG;
      return;
   }

   for (unsigned i = 0; i < rt_count; i++) {
      if (!d12_render_pass_plan_for(
             (uint32_t)rts[i].BeginningAccess.Type,
             (uint32_t)rts[i].EndingAccess.Type, &plans[i])) {
         mesa_loge("BeginRenderPass: render target %u has an unsupported "
                   "access pair (%d, %d)", i,
                   (int)rts[i].BeginningAccess.Type,
                   (int)rts[i].EndingAccess.Type);
         list->state = D12_LIST_ERROR;
         list->record_error = E_INVALIDARG;
         return;
      }

      rtvs[i] = (const struct d12_rtv *)rts[i].cpuDescriptor.ptr;
      if (plans[i].load == D12_RP_LOAD_CLEAR) {
         memcpy(clears[i], rts[i].BeginningAccess.Clear.ClearValue.Color,
                sizeof(clears[i]));
      }
   }

   if (ds) {
      if (!d12_render_pass_plan_for(
             (uint32_t)ds->DepthBeginningAccess.Type,
             (uint32_t)ds->DepthEndingAccess.Type, &depth_plan) ||
          !d12_render_pass_plan_for(
             (uint32_t)ds->StencilBeginningAccess.Type,
             (uint32_t)ds->StencilEndingAccess.Type, &stencil_plan)) {
         mesa_loge("BeginRenderPass: depth-stencil has an unsupported "
                   "access pair");
         list->state = D12_LIST_ERROR;
         list->record_error = E_INVALIDARG;
         return;
      }

      if (depth_plan.load == D12_RP_LOAD_CLEAR) {
         depth_clear =
            ds->DepthBeginningAccess.Clear.ClearValue.DepthStencil.Depth;
      }
      if (stencil_plan.load == D12_RP_LOAD_CLEAR) {
         stencil_clear = (uint8_t)ds->StencilBeginningAccess.Clear.ClearValue
                            .DepthStencil.Stencil;
      }
   }

   /* A resolving attachment must still be resolved when the pass ends. That
    * is a second pass over the surface, which this driver does not yet
    * emit -- refusing is the only answer that does not silently drop the
    * application's resolve. */
   for (unsigned i = 0; i < rt_count; i++) {
      if (!plans[i].resolve)
         continue;
      mesa_loge("BeginRenderPass: ENDING_ACCESS_RESOLVE on render target %u "
                "is unsupported", i);
      list->state = D12_LIST_ERROR;
      list->record_error = E_NOTIMPL;
      return;
   }
   if (depth_plan.resolve || stencil_plan.resolve) {
      mesa_loge("BeginRenderPass: depth-stencil ENDING_ACCESS_RESOLVE is "
                "unsupported");
      list->state = D12_LIST_ERROR;
      list->record_error = E_NOTIMPL;
      return;
   }

   (void)d12_cmd_list_begin_render_pass(
      list, rt_count, rtvs, plans, clears,
      ds ? (const struct d12_rtv *)ds->cpuDescriptor.ptr : NULL,
      ds ? &depth_plan : NULL, ds ? &stencil_plan : NULL, depth_clear,
      stencil_clear);
}

static void STDMETHODCALLTYPE
d12_list_EndRenderPass(ID3D12GraphicsCommandList6 *iface)
{
   (void)d12_cmd_list_end_render_pass(d12_cmd_list_from_iface(iface));
}

/*
 * Replay a closed bundle's command log through this list's own entry
 * points, so every call is validated and recorded exactly as if the
 * application had made it here. Draws and dispatches recorded before the
 * bundle set its own pipeline (or, for draws, topology) are undefined
 * and replay as nothing rather than borrowing the caller's.
 */
static void
d12_list_replay_bundle(struct d12_cmd_list *list, struct d12_cmd_list *bundle)
{
   ID3D12GraphicsCommandList6 *iface6 =
      (ID3D12GraphicsCommandList6 *)&list->iface;
   ID3D12GraphicsCommandList9 *iface9 =
      (ID3D12GraphicsCommandList9 *)&list->iface;
   const uint8_t *p = bundle->bundle_cmds.data;
   const uint8_t *end = p + bundle->bundle_cmds.size;

   while (p < end && list->state == D12_LIST_RECORDING) {
      const struct d12_bundle_cmd *cmd = (const struct d12_bundle_cmd *)p;
      const void *payload = cmd + 1;

      switch (cmd->op) {
      case D12_BUNDLE_SET_PIPELINE_STATE:
         d12_list_SetPipelineState(iface6, cmd->pso);
         break;
      case D12_BUNDLE_SET_GRAPHICS_ROOT_SIGNATURE:
         d12_list_SetGraphicsRootSignature(iface6, cmd->root_sig);
         break;
      case D12_BUNDLE_SET_COMPUTE_ROOT_SIGNATURE:
         d12_list_SetComputeRootSignature(iface6, cmd->root_sig);
         break;
      case D12_BUNDLE_SET_DESCRIPTOR_HEAPS:
         d12_list_SetDescriptorHeaps(iface6, cmd->heaps.count,
                                     cmd->heaps.heaps);
         break;
      case D12_BUNDLE_SET_ROOT_TABLE:
         d12_list_set_root_table(list, cmd->table.compute, cmd->table.index,
                                 cmd->table.base);
         break;
      case D12_BUNDLE_SET_ROOT_DESCRIPTOR:
         d12_list_set_root_descriptor(list, cmd->descriptor.compute,
                                      cmd->descriptor.index,
                                      cmd->descriptor.address,
                                      cmd->descriptor.type);
         break;
      case D12_BUNDLE_SET_ROOT_CONSTANTS:
         d12_list_set_root_constants(list, cmd->constants.compute,
                                     cmd->constants.index,
                                     cmd->constants.count, payload,
                                     cmd->constants.offset);
         break;
      case D12_BUNDLE_IA_PRIMITIVE_TOPOLOGY:
         d12_list_IASetPrimitiveTopology(iface6, cmd->topology);
         break;
      case D12_BUNDLE_IA_VERTEX_BUFFERS:
         d12_list_IASetVertexBuffers(iface6, cmd->vertex_buffers.start,
                                     cmd->vertex_buffers.count, payload);
         break;
      case D12_BUNDLE_IA_INDEX_BUFFER:
         d12_list_IASetIndexBuffer(
            iface6, cmd->index_buffer.bound ? &cmd->index_buffer.view : NULL);
         break;
      case D12_BUNDLE_OM_BLEND_FACTOR:
         d12_list_OMSetBlendFactor(iface6, cmd->blend_factor);
         break;
      case D12_BUNDLE_OM_STENCIL_REF:
         d12_list_OMSetFrontAndBackStencilRef(iface9, cmd->stencil_ref.front,
                                              cmd->stencil_ref.back);
         break;
      case D12_BUNDLE_RS_DEPTH_BIAS:
         d12_list_RSSetDepthBias(iface9, cmd->depth_bias.bias,
                                 cmd->depth_bias.clamp, cmd->depth_bias.slope);
         break;
      case D12_BUNDLE_IA_STRIP_CUT:
         d12_list_IASetIndexBufferStripCutValue(iface9, cmd->strip_cut);
         break;
      case D12_BUNDLE_DRAW:
         if (cmd->pso_set && cmd->topology_set)
            d12_list_DrawInstanced(iface6, cmd->draw.vertex_count,
                                   cmd->draw.instance_count,
                                   cmd->draw.first_vertex,
                                   cmd->draw.first_instance);
         break;
      case D12_BUNDLE_DRAW_INDEXED:
         if (cmd->pso_set && cmd->topology_set)
            d12_list_DrawIndexedInstanced(
               iface6, cmd->draw.vertex_count, cmd->draw.instance_count,
               cmd->draw.first_index, cmd->draw.base_vertex,
               cmd->draw.first_instance);
         break;
      case D12_BUNDLE_DISPATCH:
         if (cmd->pso_set)
            d12_list_Dispatch(iface6, cmd->dispatch.x, cmd->dispatch.y,
                              cmd->dispatch.z);
         break;
      case D12_BUNDLE_DISPATCH_MESH:
         if (cmd->pso_set)
            d12_list_DispatchMesh(iface6, cmd->dispatch.x, cmd->dispatch.y,
                                  cmd->dispatch.z);
         break;
      case D12_BUNDLE_EXECUTE_INDIRECT:
         if (cmd->pso_set)
            d12_list_ExecuteIndirect(
               iface6, cmd->execute_indirect.signature,
               cmd->execute_indirect.max_count, cmd->execute_indirect.args,
               cmd->execute_indirect.args_offset,
               cmd->execute_indirect.count,
               cmd->execute_indirect.count_offset);
         break;
      }

      p += cmd->size_B;
   }
}

static const struct d12_list9_vtbl d12_list_vtbl = {
   .v6 = {
   /* every unimplemented slot fails safely */
   D12_ID3D12GraphicsCommandList6_DEFAULTS,
   .QueryInterface = d12_list_QueryInterface,
   .AddRef = d12_list_AddRef,
   .Release = d12_list_Release,
   .GetPrivateData = d12_list_GetPrivateData,
   .SetPrivateData = d12_list_SetPrivateData,
   .SetPrivateDataInterface = d12_list_SetPrivateDataInterface,
   .SetName = d12_list_SetName,
   .GetDevice = d12_list_GetDevice,
   .GetType = d12_list_GetType,
   .Close = d12_list_Close,
   .Reset = d12_list_Reset,
   .CopyBufferRegion = d12_list_CopyBufferRegion,
   .WriteBufferImmediate = d12_list_WriteBufferImmediate,
   .CopyResource = d12_list_CopyResource,
   .CopyTextureRegion = d12_list_CopyTextureRegion,
   .CopyTiles = d12_list_CopyTiles,
   .ResolveSubresource = d12_list_ResolveSubresource,
   .ResolveSubresourceRegion = d12_list_ResolveSubresourceRegion,
   .ResourceBarrier = d12_list_ResourceBarrier,
   .OMSetRenderTargets = d12_list_OMSetRenderTargets,
   .SOSetTargets = d12_list_SOSetTargets,
   .OMSetStencilRef = d12_list_OMSetStencilRef,
   .OMSetBlendFactor = d12_list_OMSetBlendFactor,
   .ClearRenderTargetView = d12_list_ClearRenderTargetView,
   .ClearDepthStencilView = d12_list_ClearDepthStencilView,
   .ClearUnorderedAccessViewUint = d12_list_ClearUnorderedAccessViewUint,
   .ClearUnorderedAccessViewFloat = d12_list_ClearUnorderedAccessViewFloat,
   .DiscardResource = d12_list_DiscardResource,
   .ClearState = d12_list_ClearState,
   .SetPipelineState = d12_list_SetPipelineState,
   .SetComputeRootSignature = d12_list_SetComputeRootSignature,
   .SetComputeRootUnorderedAccessView =
      d12_list_SetComputeRootUnorderedAccessView,
   .SetGraphicsRootUnorderedAccessView =
      d12_list_SetGraphicsRootUnorderedAccessView,
   .SetComputeRootShaderResourceView =
      d12_list_SetComputeRootShaderResourceView,
   .SetGraphicsRootShaderResourceView =
      d12_list_SetGraphicsRootShaderResourceView,
   .SetComputeRootConstantBufferView =
      d12_list_SetComputeRootConstantBufferView,
   .SetGraphicsRoot32BitConstants = d12_list_SetGraphicsRoot32BitConstants,
   .SetComputeRoot32BitConstants = d12_list_SetComputeRoot32BitConstants,
   .SetGraphicsRoot32BitConstant = d12_list_SetGraphicsRoot32BitConstant,
   .SetComputeRoot32BitConstant = d12_list_SetComputeRoot32BitConstant,
   .SetGraphicsRootConstantBufferView =
      d12_list_SetGraphicsRootConstantBufferView,
   .SetComputeRootDescriptorTable = d12_list_SetComputeRootDescriptorTable,
   .SetGraphicsRootDescriptorTable = d12_list_SetGraphicsRootDescriptorTable,
   .SetDescriptorHeaps = d12_list_SetDescriptorHeaps,
   .Dispatch = d12_list_Dispatch,
   .SetGraphicsRootSignature = d12_list_SetGraphicsRootSignature,
   .IASetPrimitiveTopology = d12_list_IASetPrimitiveTopology,
   .IASetVertexBuffers = d12_list_IASetVertexBuffers,
   .RSSetViewports = d12_list_RSSetViewports,
   .RSSetScissorRects = d12_list_RSSetScissorRects,
   .DrawInstanced = d12_list_DrawInstanced,
   .DrawIndexedInstanced = d12_list_DrawIndexedInstanced,
   .ExecuteBundle = d12_list_ExecuteBundle,
   .ExecuteIndirect = d12_list_ExecuteIndirect,
   .SetPredication = d12_list_SetPredication,
   .IASetIndexBuffer = d12_list_IASetIndexBuffer,
   .BeginQuery = d12_list_BeginQuery,
   .EndQuery = d12_list_EndQuery,
   .ResolveQueryData = d12_list_ResolveQueryData,
   .DispatchMesh = d12_list_DispatchMesh,
   .SetSamplePositions = d12_list_SetSamplePositions,
   .BeginRenderPass = d12_list_BeginRenderPass,
   .EndRenderPass = d12_list_EndRenderPass,
   /* Everything unset is NULL: loud crash over silent lie. */
   },
   .Barrier = d12_list_Barrier,
   .OMSetFrontAndBackStencilRef = d12_list_OMSetFrontAndBackStencilRef,
   .RSSetDepthBias = d12_list_RSSetDepthBias,
   .IASetIndexBufferStripCutValue = d12_list_IASetIndexBufferStripCutValue,
};

HRESULT
d12_cmd_list_create(struct d12_device *device, D3D12_COMMAND_LIST_TYPE type,
                    struct d12_cmd_allocator *allocator,
                    struct d12_pipeline_state *initial_state, REFIID riid,
                    void **out)
{
   if (!out)
      return E_POINTER;

   switch (type) {
   case D3D12_COMMAND_LIST_TYPE_DIRECT:
   case D3D12_COMMAND_LIST_TYPE_BUNDLE:
   case D3D12_COMMAND_LIST_TYPE_COMPUTE:
   case D3D12_COMMAND_LIST_TYPE_COPY:
      break;
   default:
      return E_INVALIDARG;
   }

   /* ID3D12Device4::CreateCommandList1 creates a list with no allocator; it
    * starts closed and the first Reset supplies the allocator. */
   if (allocator && (allocator->type != type || allocator->recording_lists))
      return E_INVALIDARG;

   struct d12_cmd_list *list = calloc(1, sizeof(*list));
   if (!list)
      return E_OUTOFMEMORY;

   list->stream_bo = agx_bo_create(&device->dev, D12_STREAM_BO_SIZE, 0,
                                   AGX_BO_WRITEBACK, "d3d12 cmdlist stream");
   list->usc_bo = agx_bo_create(&device->dev, D12_USC_BO_SIZE, 0,
                                AGX_BO_LOW_VA | AGX_BO_WRITEBACK,
                                "d3d12 cmdlist usc");
   /* Every later use of this storage dereferences agx_bo_map() without a
    * check, which is only safe because a failed Lock2 fails construction
    * here.  Keep the map in the same admission test as the allocation. */
   if (!list->stream_bo || !list->usc_bo || !agx_bo_map(list->stream_bo) ||
       !agx_bo_map(list->usc_bo)) {
      if (list->stream_bo)
         agx_bo_unreference(&device->dev, list->stream_bo);
      if (list->usc_bo)
         agx_bo_unreference(&device->dev, list->usc_bo);
      free(list);
      return E_OUTOFMEMORY;
   }

   list->iface.lpVtbl =
      (ID3D12GraphicsCommandListVtbl *)&d12_list_vtbl.v6;
   list->refcount = 1;
   list->device = device;
   list->type = type;
   list->allocator = allocator;
   if (allocator)
      d12_alloc_AddRef(&allocator->iface);
   list->pso = initial_state;
   d12_private_data_store_init(&list->private_data, (IUnknown *)&list->iface);
   util_dynarray_init(&list->g17p_vdm_storage, NULL);
   util_dynarray_init(&list->descriptor_fixups, NULL);
   util_dynarray_init(&list->indirect_constant_fixups, NULL);
   util_dynarray_init(&list->indirect_argument_fixups, NULL);
   util_dynarray_init(&list->bundle_cmds, NULL);
   if (allocator)
      allocator->recording_lists++;
   d12_device_add_ref(device);

   /* CreateCommandList returns a list in the recording state;
    * CreateCommandList1 returns a closed one. */
   if (allocator)
      d12_list_begin_recording(list);
   else
      list->state = D12_LIST_CLOSED;

   HRESULT hr = d12_list_QueryInterface((ID3D12GraphicsCommandList6 *)&list->iface, riid, out);
   d12_list_Release((ID3D12GraphicsCommandList6 *)&list->iface);
   return hr;
}
