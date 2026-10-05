/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * ID3D12CommandQueue on an asahi hardware queue.  DRM timeline syncobjs do
 * not support wait-before-signal: Honeykrisp therefore routes Vulkan submits
 * through Mesa's threaded-submit layer until every dependency is pending.
 * D3D12 needs the same property, plus non-monotonic fence values, so each
 * queue has a small ordered submit worker.  Wait, submit and Signal calls
 * enqueue immediately; that worker resolves dependencies and feeds AGX in
 * API order.  A separate retirement worker waits for hardware timeline
 * points, publishes fence signals, and releases command storage.  Keeping
 * completion off the submit path is what allows multiple command lists and
 * frames to remain in flight.
 */

#include "d12_private.h"
#include "d12_agx3_cmdstream_core.h"
#include "d12_native_render_resources_core.h"
#include "d12_read_set_core.h"
#include "d12_submit.h"
#include "d12_write_set_core.h"

#include "decode.h"
#include "util/bitscan.h"
#include "util/os_time.h"
#include "util/u_dynarray.h"

#include "drm-uapi/drm.h"

#include "agx_helpers.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "agx_abi.h"
#include "agx_usc.h"
#include "clc/asahi_clc.h"
#include "util/u_math.h"

#include <time.h>
#include <errno.h>

/*
 * Aurora: forward progress must not depend on a native waiter wakeup.
 *
 * On this Windows-on-Apple-Silicon ARM64 target a lost wake was captured
 * directly while bringing up the D3D11 stack: a submission producer sat
 * indefinitely in NtWaitForAlertByThreadId acquiring the submission queue
 * lock, even though the live lock word was already zero, and other workers
 * in the same process showed the identical free-lock/waiter contradiction.
 * A lost wake here stops command submission before dxgkrnl or the KMD is
 * ever called, so it presents as a GPU hang with no submission to blame.
 *
 * Mesa's Windows c11 threads map mtx_t to a CRITICAL_SECTION and cnd_t to a
 * CONDITION_VARIABLE, which are exactly the primitives observed dropping
 * wakes.  Acquire with a bounded try/yield loop, and make every condition
 * wait a legal spurious wakeup followed by a predicate recheck -- every call
 * site below already loops on its predicate, so a spurious wake costs
 * nothing.  A real cnd_signal still returns immediately, so this changes no
 * healthy-path latency; the timeout only bounds the damage of a lost wake.
 *
 * This is deliberately not #ifdef _WIN32.  The Linux build is where these
 * paths get exercised locally, and a concurrency primitive that differs
 * between the test build and the shipping one hides exactly the class of bug
 * it is here to contain.  The portable cost is one interlocked operation on
 * the uncontended path plus a bounded idle poll.
 */
#define D12_SYNC_SPINS_BEFORE_SLEEP 64u
#define D12_SYNC_FALLBACK_NS        1000000L

static uint8_t *
d12_queue_emit_cdm_cache_flush(struct agx_device *dev, uint8_t *cursor)
{
   switch (agx_chip_command_stream_generation(dev->chip)) {
   case AGX_CS_GENERATION_AGX2:
      return (uint8_t *)agx_cdm_barrier((uint32_t *)cursor, dev->chip);

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      const struct agx3_cdm_barrier_info info = {
         .chip = dev->chip,
         .flags = 0x5,
      };
      struct d12_agx3_cdm_barrier_encoding encoded;
      if (!d12_agx3_encode_cdm_barrier(&info, &encoded))
         return NULL;

      memcpy(cursor, encoded.bytes, encoded.length);
      return cursor + encoded.length;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      return NULL;
   }
}

/* A waiter that is blocking a producer polls at the floor: it is only ever
 * waiting because work is already pending, so the poll is transient and the
 * recovery must be fast enough not to cost a frame.  A worker parked on an
 * empty queue backs off instead -- an idle desktop must not burn a core at
 * 1 kHz per queue -- and still recovers from a lost wake within the ceiling.
 */
#define D12_SYNC_IDLE_CEILING_NS    (32 * D12_SYNC_FALLBACK_NS)

static void
d12_sync_nap(long nanoseconds)
{
   const struct timespec nap = {.tv_sec = 0, .tv_nsec = nanoseconds};
   thrd_sleep(&nap, NULL);
}

static void
d12_sync_lock(mtx_t *mutex)
{
   unsigned spins = 0;

   while (mtx_trylock(mutex) != thrd_success) {
      if (++spins < D12_SYNC_SPINS_BEFORE_SLEEP) {
         thrd_yield();
      } else {
         d12_sync_nap(D12_SYNC_FALLBACK_NS);
         spins = 0;
      }
   }
}

static void
d12_sync_wait_for(cnd_t *cond, mtx_t *mutex, long nanoseconds)
{
   struct timespec deadline;

   if (timespec_get(&deadline, TIME_UTC) != TIME_UTC) {
      /* No clock: a plain wait still beats a busy loop, and the caller's
       * predicate loop stays correct either way. */
      cnd_wait(cond, mutex);
      return;
   }

   deadline.tv_nsec += nanoseconds;
   while (deadline.tv_nsec >= 1000000000L) {
      deadline.tv_nsec -= 1000000000L;
      deadline.tv_sec += 1;
   }
   (void)cnd_timedwait(cond, mutex, &deadline);
}

static void
d12_sync_wait(cnd_t *cond, mtx_t *mutex)
{
   d12_sync_wait_for(cond, mutex, D12_SYNC_FALLBACK_NS);
}

/* Doubles the caller's backoff up to the ceiling.  Callers reset it to zero
 * once their predicate is satisfied, so a busy queue always re-enters at the
 * floor and only a genuinely idle one drifts out to the ceiling. */
static void
d12_sync_wait_idle(cnd_t *cond, mtx_t *mutex, long *backoff_ns)
{
   if (*backoff_ns < D12_SYNC_FALLBACK_NS)
      *backoff_ns = D12_SYNC_FALLBACK_NS;

   d12_sync_wait_for(cond, mutex, *backoff_ns);

   if (*backoff_ns < D12_SYNC_IDLE_CEILING_NS)
      *backoff_ns *= 2;
}

enum d12_queue_op_type {
   D12_QUEUE_OP_WAIT,
   D12_QUEUE_OP_SUBMIT,
   D12_QUEUE_OP_SIGNAL,
   D12_QUEUE_OP_PLACEMENT_SIGNAL,
   D12_QUEUE_OP_PLACEMENT_BARRIER,
   D12_QUEUE_OP_PLACEMENT_WAIT,
   D12_QUEUE_OP_SPARSE_BIND,
};

struct d12_sparse_completion {
   mtx_t lock;
   cnd_t event;
   bool done;
   int result;
};

struct d12_queue_op {
   struct d12_queue_op *next;
   enum d12_queue_op_type type;

   struct d12_fence *fence;
   struct d12_placement_fence *placement_fence;
   uint32_t placement_wait_count;
   struct d12_queue_placement_wait
      placement_waits[D12_QUEUE_MAX_PLACEMENT_DEPENDENCIES];
   uint64_t value;
   struct d12_fence_ticket ticket;
   uint64_t retire_value;

   struct d12_cmd_list *list;
   struct d12_device *submit_device;
   bool bos_initialized;
   bool capture_failed;
   /*
    * A command list is mutable as soon as ExecuteCommandLists returns: the
    * application may Reset it while this operation is still waiting behind
    * an earlier fence.  Keep the exact storage referenced by the packed
    * payload alive here instead of relying on the list's current BOs (which
    * Reset replaces) or retaining every old recording until allocator
    * Reset.  The latter made a long wait-before-signal chain consume 3 MiB
    * per submit and exhaust system memory before either queue could run.
    */
   struct util_dynarray bos;
   struct util_dynarray g17p_vdm_storage;
   void *payload;
   size_t payload_size;
   unsigned nr_vdm;
   unsigned nr_cdm;
   bool write_set_complete;
   bool read_set_complete;
   uint32_t exact_read_range_count;
   struct d12_read_range
      exact_read_ranges[AGX_SUBMIT_MAX_READ_RANGES];
   bool signal_slot_valid;
   uint32_t signal_slot;
   uint8_t inline_payload[2 * sizeof(struct drm_asahi_cmd_header) +
                          sizeof(struct drm_asahi_attachment) +
                          sizeof(struct drm_asahi_cmd_compute)];

   struct drm_asahi_gem_bind_op *bind_ops;
   uint32_t bind_count;
   struct d12_resource *sparse_resource;
   struct d12_sparse_map_update *sparse_map_updates;
   uint32_t sparse_map_update_count;
   struct d12_sparse_completion *completion;
};

static void d12_queue_stop_worker(struct d12_queue *queue);
static void d12_queue_signal_arena_cleanup(struct d12_queue *queue);

static HRESULT STDMETHODCALLTYPE
d12_queue_QueryInterface(ID3D12CommandQueue *iface, REFIID riid, void **out)
{
   struct d12_queue *queue = d12_queue_from_iface(iface);

   if (!out)
      return E_POINTER;

   if (d12_guid_equal(riid, &IID_IUnknown) ||
       d12_guid_equal(riid, &IID_ID3D12Object) ||
       d12_guid_equal(riid, &IID_ID3D12DeviceChild) ||
       d12_guid_equal(riid, &IID_ID3D12Pageable) ||
       d12_guid_equal(riid, &IID_ID3D12CommandQueue)) {
      p_atomic_inc(&queue->refcount);
      *out = iface;
      return S_OK;
   }

   if (d12_private_data_query_notifier(&queue->private_data, riid, out))
      return S_OK;

   d12_log_unknown_riid("ID3D12CommandQueue::QueryInterface", riid);
   *out = NULL;
   return E_NOINTERFACE;
}

static ULONG STDMETHODCALLTYPE
d12_queue_AddRef(ID3D12CommandQueue *iface)
{
   struct d12_queue *queue = d12_queue_from_iface(iface);
   return p_atomic_inc_return(&queue->refcount);
}

static ULONG STDMETHODCALLTYPE
d12_queue_Release(ID3D12CommandQueue *iface)
{
   struct d12_queue *queue = d12_queue_from_iface(iface);
   uint32_t refs = p_atomic_dec_return(&queue->refcount);

   if (refs == 0) {
      d12_queue_stop_worker(queue);
      d12_queue_signal_arena_cleanup(queue);
      queue->device->sync->destroy(queue->device->sync, queue->syncobj);
      agx_destroy_command_queue(&queue->device->dev, queue->agx_queue_id);
      cnd_destroy(&queue->retire_available);
      cnd_destroy(&queue->work_available);
      mtx_destroy(&queue->ddi_submit_lock);
      mtx_destroy(&queue->lock);
      d12_private_data_store_destroy(&queue->private_data);
      d12_device_release(queue->device);
      free(queue);
   }

   return refs;
}

D12_DEFINE_PRIVATE_DATA_METHODS(d12_queue, ID3D12CommandQueue,
                                d12_queue_from_iface)

static HRESULT STDMETHODCALLTYPE
d12_queue_GetDevice(ID3D12CommandQueue *iface, REFIID riid, void **out)
{
   struct d12_queue *queue = d12_queue_from_iface(iface);
   return ID3D12Device_QueryInterface(&queue->device->iface, riid, out);
}

/*
 * Fill a drm_asahi_cmd_render for a clear-only render segment. Mirrors
 * hk_queue.c's asahi_fill_vdm_command for the color-only, single-layer case.
 */
/*
 * The standard D3D pattern for a sample count, encoded the way AGX wants it.
 * Derived rather than tabulated so the defaults and any pattern the
 * application programs come from one encoder; d12_sample_position_core_tests
 * asserts this still produces 0x88, 0x44cc and 0xeaa26e26.
 */
uint32_t
d12_default_sample_positions(unsigned nr_samples)
{
   struct d12_sample_position positions[D12_SAMPLE_POSITION_MAX_SAMPLES];
   uint32_t packed;

   if (!d12_sample_positions_standard(nr_samples, positions) ||
       !d12_sample_positions_pack(nr_samples, positions, &packed))
      UNREACHABLE("invalid D3D12 sample count");

   return packed;
}

/* Mirror the existing native kernel address checks before any ioctl. This
 * identifies the exact UAPI field; it does not relax alignment or residency. */
static const char *
d12_g17p_render_address_error(const struct drm_asahi_cmd_render *c)
{
   if (!c->vdm_ctrl_stream_base || (c->vdm_ctrl_stream_base & 3))
      return "vdm_ctrl_stream_base";
   if (!c->isp_scissor_base || (c->isp_scissor_base & 7))
      return "isp_scissor_base";
   if (c->isp_dbias_base & 7)
      return "isp_dbias_base";
   if (c->isp_oclqry_base & 7)
      return "isp_oclqry_base";
   if ((c->sampler_heap == 0) != (c->sampler_count == 0) ||
       (c->sampler_count && (c->sampler_heap & 7)))
      return "sampler_heap/count";
   return NULL;
}

static bool
d12_fill_render_cmd(struct d12_device *device, struct d12_cmd_list *list,
                    const struct d12_segment *seg,
                    struct drm_asahi_cmd_render *c)
{
   memset(c, 0, sizeof(*c));
   if (device->dev.chip == AGX_CHIP_G17P && seg->ds &&
       (!d12_g17p_depth_enabled(&device->dev) ||
        !d12_g17p_depth_segment_supported(seg))) {
      mesa_loge("native depth submission refused before unqualified ZLS metadata");
      return false;
   }

   c->vdm_ctrl_stream_base = seg->g17p_vdm_control_base
      ? seg->g17p_vdm_control_base : list->stream_bo->va->addr + seg->stream_start;

   agx_pack(&c->ppp_ctrl, CR_PPP_CONTROL, cfg) {
      cfg.enable_w_clamp = true;
      cfg.fixed_point_format = 1;
   }

   c->width_px = seg->width_px;
   c->height_px = seg->height_px;

   /* Background object values: hk's base constant, with the stencil clear
    * value in the low byte.
    */
   c->isp_bgobjdepth = device->dev.chip == AGX_CHIP_G17P && !seg->ds
      ? fui(1.0f) : seg->isp_bgobjdepth;
   c->isp_bgobjvals = 0x300 | seg->stencil_clear;

   /*
    * Depth/stencil. The ZLS unit is fixed-function: unlike colour, there is
    * no background or end-of-tile program involved. Pointing it at the
    * buffers and setting the load/store bits is the entire configuration.
    *
    * z_load is the interesting bit. If the pass cleared depth, every tile
    * starts from isp_bgobjdepth and loading would overwrite that with stale
    * memory. If it did not, the application is continuing from depth it
    * rendered earlier, and NOT loading would silently discard it.
    */
   if (seg->ds) {
      const struct ail_layout *z = &seg->ds->layout;
      /* A render pass states depth and stencil traffic explicitly; outside
       * one it is inferred, and a cleared aspect must still be committed
       * because READ_ONLY constrains draw writes rather than the clear. */
      const struct d12_render_pass_plan depth_plan =
         seg->render_pass
            ? seg->depth_plan
            : (struct d12_render_pass_plan){
                 .load = seg->depth_cleared ? D12_RP_LOAD_CLEAR
                                            : D12_RP_LOAD_PRESERVE,
                 .store = (seg->depth_cleared ||
                           !(seg->dsv_flags &
                             D3D12_DSV_FLAG_READ_ONLY_DEPTH))
                             ? D12_RP_STORE_PRESERVE
                             : D12_RP_STORE_NONE,
              };

      struct agx_zls zls = {
         .z_load = depth_plan.load == D12_RP_LOAD_PRESERVE,
         .z_store = depth_plan.store == D12_RP_STORE_PRESERVE,
      };

      c->depth.base = seg->ds_address;

      /* Stride is in pages, biased, with the low bit set — the encoding
       * hk uses (hk_cmd_draw.c).
       */
      unsigned z_pages = z->layer_stride_B / AIL_PAGESIZE;
      c->depth.stride = ((z_pages - 1) << 14) | 1;

      const struct ail_layout *s = NULL;
      if (seg->ds->stencil_bo) {
         const struct d12_render_pass_plan stencil_plan =
            seg->render_pass
               ? seg->stencil_plan
               : (struct d12_render_pass_plan){
                    .load = seg->stencil_cleared ? D12_RP_LOAD_CLEAR
                                                 : D12_RP_LOAD_PRESERVE,
                    .store = (seg->stencil_cleared ||
                              !(seg->dsv_flags &
                                D3D12_DSV_FLAG_READ_ONLY_STENCIL))
                                ? D12_RP_STORE_PRESERVE
                                : D12_RP_STORE_NONE,
                 };

         s = &seg->ds->stencil_layout;
         zls.s_load = stencil_plan.load == D12_RP_LOAD_PRESERVE;
         zls.s_store = stencil_plan.store == D12_RP_STORE_PRESERVE;

         c->stencil.base = seg->stencil_address;
         unsigned s_pages = s->layer_stride_B / AIL_PAGESIZE;
         c->stencil.stride = ((s_pages - 1) << 14) | 1;
      }

      struct agx_zls_control_packed zls_ctrl;
      agx_pack_zls_control(&zls_ctrl, z, s, &zls);
      static_assert(sizeof(c->zls_ctrl) == sizeof(zls_ctrl),
                    "zls_ctrl width must match the packed struct");
      memcpy(&c->zls_ctrl, &zls_ctrl, sizeof(zls_ctrl));

      /* The depth buffer may be larger than the colour target. */
      agx_pack(&c->isp_zls_pixels, CR_ISP_ZLS_PIXELS, cfg) {
         cfg.x = u_minify(z->width_px, seg->ds_level);
         cfg.y = u_minify(z->height_px, seg->ds_level);
      }

      /* Integer depth bias applies to unorm depth buffers, float bias to
       * floating-point ones.
       */
      if (z->format == PIPE_FORMAT_Z16_UNORM)
         c->flags |= DRM_ASAHI_RENDER_DBIAS_IS_INT;
   }

   /* Standard D3D sample positions, in the same packed spelling Honeykrisp
    * programs. Leaving this zero for a draw is a GPU-hang class bug.
    */
   unsigned samples = MAX2(seg->tib.nr_samples, 1);
   c->ppp_multisamplectl = seg->sample_positions
                              ? seg->sample_positions
                              : d12_default_sample_positions(samples);

   agx_tilebuffer_set_drm_cmd(c, &seg->tib);
   c->samples = samples;
   c->sample_size_B = seg->tib.sample_size_B;
   c->layers = seg->layers;

   const float tan_60 = 1.732051f;
   c->isp_merge_upper_x = fui(tan_60 / seg->width_px);
   c->isp_merge_upper_y = fui(tan_60 / seg->height_px);

   c->bg.usc = seg->bg_usc | 4;
   c->eot.usc = seg->eot_usc | 4;

   c->partial_bg.usc = seg->partial_bg_usc | 4;
   c->partial_eot.usc = seg->partial_eot_usc | 4;

   c->sampler_heap = seg->sampler_heap;
   c->sampler_count = seg->sampler_count;

   if (device->dev.chip == AGX_CHIP_G17P) {
      const struct d12_native_render_resource_specs specs = {
         .bg = seg->g17p_bg_resources, .eot = seg->g17p_eot_resources,
         .partial_bg = seg->g17p_partial_bg_resources,
         .partial_eot = seg->g17p_partial_eot_resources,
      };
      d12_native_render_resource_specs_pack(&specs, c);
   } else {
      static_assert(sizeof(c->bg.rsrc_spec) == sizeof(seg->bg_counts),
                    "counts word must match rsrc_spec");
      memcpy(&c->bg.rsrc_spec, &seg->bg_counts, sizeof(seg->bg_counts));
      memcpy(&c->eot.rsrc_spec, &seg->eot_counts, sizeof(seg->eot_counts));
      memcpy(&c->partial_bg.rsrc_spec, &seg->partial_bg_counts,
             sizeof(seg->partial_bg_counts));
      memcpy(&c->partial_eot.rsrc_spec, &seg->partial_eot_counts,
             sizeof(seg->partial_eot_counts));
   }

   /* The scissor array carries both the XY bounds and the depth clip range
    * that draws index into. Occlusion scratch must be valid memory whether
    * or not queries run (mirrors hk pointing at the device query heap).
    */
   c->isp_scissor_base = seg->scissor_base;
   c->isp_dbias_base = seg->depth_bias_base;
   c->isp_oclqry_base = device->oclqry_bo->va->addr;

   if (seg->process_empty_tiles)
      c->flags |= DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES;

   if (seg->scratch_vs_bo) {
      c->flags |= DRM_ASAHI_RENDER_VERTEX_SCRATCH;
      c->vertex_helper.data = seg->scratch_vs_bo->va->addr;
      c->vertex_helper.cfg = seg->scratch_vs_preamble ? (1 << 16) : 0;
      c->vertex_helper.binary = agx_helper_program(&device->bg_eot);
   }

   if (seg->scratch_fs_bo) {
      c->fragment_helper.data = seg->scratch_fs_bo->va->addr;
      c->fragment_helper.cfg = seg->scratch_fs_preamble ? (1 << 16) : 0;
      c->fragment_helper.binary = agx_helper_program(&device->bg_eot);
   }

   /*
    * Every address this command hands the hardware, so a VDM fault can be
    * matched to the field that carried it. The fault we are chasing reads
    * 0x4000000000 -- exactly 1 << 38, a round number rather than a leaked
    * pointer, and not AGX_ZERO_PAGE_ADDRESS (1 << 32) -- from dVDM_SSD,
    * the unit that walks this control stream. Print the candidates in the
    * same run as the fault and match, exactly as the tessellation VA map
    * did for the last one.
    */
   if (device->dev.chip == AGX_CHIP_G17P) {
      const char *field = d12_g17p_render_address_error(c);
      if (field) {
         mesa_loge("G17P render submission metadata refused: field=%s "
                   "vdm=0x%llx scissor=0x%llx dbias=0x%llx "
                   "oclqry=0x%llx sampler=0x%llx/%u extent=%ux%u "
                   "draws=%u empty_tiles=%u", field,
                   (unsigned long long)c->vdm_ctrl_stream_base,
                   (unsigned long long)c->isp_scissor_base,
                   (unsigned long long)c->isp_dbias_base,
                   (unsigned long long)c->isp_oclqry_base,
                   (unsigned long long)c->sampler_heap, c->sampler_count,
                   c->width_px, c->height_px, seg->has_draws,
                   seg->process_empty_tiles);
         return false;
      }
   }

   if (getenv("D3D12AGX_DUMP_RENDER_CMD")) {
      mesa_logi("render cmd: vdm_base=0x%llx (stream_bo 0x%llx..0x%llx, "
                "seg %zu..%zu, dispatches=%u)",
                (unsigned long long)c->vdm_ctrl_stream_base,
                (unsigned long long)list->stream_bo->va->addr,
                (unsigned long long)(list->stream_bo->va->addr +
                                     list->stream_bo->size),
                seg->stream_start, seg->stream_end, seg->dispatches);
      mesa_logi("  depth=0x%llx stencil=0x%llx scissor=0x%llx "
                "dbias=0x%llx oclqry=0x%llx",
                (unsigned long long)c->depth.base,
                (unsigned long long)c->stencil.base,
                (unsigned long long)c->isp_scissor_base,
                (unsigned long long)c->isp_dbias_base,
                (unsigned long long)c->isp_oclqry_base);
      mesa_logi("  bg.usc=0x%llx eot.usc=0x%llx  %ux%ux%u samples=%u",
                (unsigned long long)c->bg.usc, (unsigned long long)c->eot.usc,
                c->width_px, c->height_px, c->layers, c->samples);
      if (seg->ds)
         mesa_logi("  native D32 metadata: zls=0x%llx pixels=0x%x "
                   "depth-stride=0x%x clear=0x%x load=%u store=%u "
                   "fd-policy=%s",
                   (unsigned long long)c->zls_ctrl, c->isp_zls_pixels,
                   c->depth.stride, c->isp_bgobjdepth,
                   (unsigned)!seg->depth_cleared,
                   (unsigned)(seg->depth_cleared ||
                              !(seg->dsv_flags & D3D12_DSV_FLAG_READ_ONLY_DEPTH)),
#ifndef _WIN32
                   d12_platform_is_real_asahi_render_fd(device->dev.fd)
                      ? "physical-asahi-render-D32-optin" : "registered-shim-only"
#else
                   "closed-windows"
#endif
                   );
   }
   return true;
}

static void
d12_queue_op_destroy(struct d12_queue_op *op)
{
   if (op->type == D12_QUEUE_OP_WAIT)
      d12_fence_ticket_fini(&op->ticket);
   if (op->fence)
      d12_fence_internal_unref(op->fence);
   if (op->placement_fence)
      d12_placement_fence_unref(op->placement_fence);
   for (uint32_t index = 0; index < op->placement_wait_count; ++index) {
      if (op->placement_waits[index].fence)
         d12_placement_fence_unref(op->placement_waits[index].fence);
   }
   if (op->type == D12_QUEUE_OP_SUBMIT && op->bos_initialized) {
      util_dynarray_foreach(&op->bos, struct agx_bo *, bo)
         agx_bo_unreference(&op->submit_device->dev, *bo);
      util_dynarray_fini(&op->bos);
      util_dynarray_foreach(&op->g17p_vdm_storage,
                            struct d12_g17p_vdm_storage *, storage)
         d12_g17p_vdm_storage_unref(*storage);
      util_dynarray_fini(&op->g17p_vdm_storage);
   }
   if (op->list)
      ID3D12GraphicsCommandList_Release(&op->list->iface);
   if (op->sparse_resource)
      ID3D12Resource_Release(&op->sparse_resource->iface);
   free(op->bind_ops);
   free(op->sparse_map_updates);
   if (op->payload && op->payload != op->inline_payload)
      free(op->payload);
   free(op);
}

#define D12_SUBMIT_MAX_ATTACHMENTS 16

static bool
d12_queue_resolve_bo_range(struct agx_device *dev, uint64_t address,
                           uint64_t *base, uint64_t *size)
{
   bool found = false;

   simple_mtx_lock(&dev->bo_map_lock);
   for (uint32_t handle = 0; handle < dev->max_handle; handle++) {
      struct agx_bo *bo = agx_lookup_bo(dev, handle);
      if (!bo->va || !bo->size)
         continue;

      /* agx_va_alloc() returns absolute addresses for both heaps. LOW_VA
       * selects the USC heap; only packed USC pointers subtract shader_base. */
      uint64_t bo_base = bo->va->addr;
      if (address < bo_base || address - bo_base >= bo->size)
         continue;

      *base = bo_base;
      *size = bo->size;
      found = true;
      break;
   }
   simple_mtx_unlock(&dev->bo_map_lock);
   return found;
}

static bool
d12_queue_attachment_add(struct agx_device *dev,
                         struct d12_write_range *attachments,
                         unsigned *count, unsigned limit,
                         uint64_t pointer, uint64_t size)
{
   if (!pointer)
      return true;

   if (!size && !d12_queue_resolve_bo_range(dev, pointer, &pointer, &size))
      return false;
   return d12_write_set_add(attachments, count, limit, pointer, size) ==
          D12_WRITE_SET_OK;
}

static bool
d12_queue_attachment_add_bo_range(struct d12_write_range *attachments,
                                  unsigned *count, unsigned limit,
                                  struct agx_bo *bo, uint64_t offset,
                                  uint64_t size)
{
   if (!bo || !bo->va || !bo->size)
      return false;

   return d12_write_set_add_allocation(
             attachments, count, limit, bo->va->addr, bo->size, offset,
             size) == D12_WRITE_SET_OK;
}

static bool
d12_queue_attachment_add_descriptor(
   struct agx_device *dev, struct d12_write_range *attachments,
   unsigned *count, unsigned limit, const struct d12_descriptor *descriptor)
{
   if (!descriptor || descriptor->kind != D12_DESC_UAV)
      return true;

   if (descriptor->resource && descriptor->resource->reserved) {
      /* Every hole in a writable sparse resource aliases the device scratch
       * page. Include that physical backing once even when no application
       * tiles are mapped, or AGX_SUBMIT_WRITE_SET_COMPLETE would omit a real
       * GPU write target. */
      bool complete = dev->scratch_bo &&
         d12_queue_attachment_add(
            dev, attachments, count, limit, AGX_SCRATCH_PAGE_ADDRESS,
            dev->scratch_bo ? dev->scratch_bo->size : 0);
      simple_mtx_lock(&descriptor->resource->sparse_lock);
      for (uint32_t i = 0; i < descriptor->resource->sparse_tile_count; ++i) {
         const struct d12_sparse_mapping mapping =
            descriptor->resource->sparse_mappings[i];
         struct agx_bo *bo = mapping.bo;
         if (bo &&
             !d12_queue_attachment_add_bo_range(
                attachments, count, limit, bo, mapping.heap_offset_B,
                D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES)) {
            complete = false;
         }
      }
      simple_mtx_unlock(&descriptor->resource->sparse_lock);
      return complete;
   }

   uint64_t base = 0, size = 0;
   if (descriptor->resource) {
      if (!d12_descriptor_resource_span(descriptor, &base, &size))
         return false;
   } else {
      base = descriptor->address;
   }

   if (base && !d12_queue_attachment_add(dev, attachments, count, limit,
                                         base, size))
      return false;
   if (descriptor->counter_address &&
       !d12_queue_attachment_add(dev, attachments, count, limit,
                                 descriptor->counter_address,
                                 sizeof(uint32_t)))
      return false;
   return true;
}

static bool
d12_queue_collect_write_tracker(
   struct agx_device *dev, const struct d12_write_tracker *tracker,
   struct d12_write_range *attachments, unsigned *count,
   unsigned limit, bool *complete)
{
   if (tracker->incomplete)
      *complete = false;
   if (!tracker->initialized)
      return true;

   util_dynarray_foreach(&tracker->refs, struct d12_write_ref, ref) {
      switch (ref->kind) {
      case D12_WRITE_REF_RANGE:
         if (!d12_queue_attachment_add(
                dev, attachments, count, limit, ref->range.address,
                ref->range.size)) {
            *complete = false;
            return true;
         }
         break;
      case D12_WRITE_REF_DESCRIPTOR:
         if (!d12_queue_attachment_add_descriptor(
                dev, attachments, count, limit, ref->descriptor)) {
            *complete = false;
            return true;
         }
         break;
      case D12_WRITE_REF_HEAP: {
         const struct d12_descriptor_heap *heap = ref->heap.heap;
         if (!heap || ref->heap.first >= heap->desc.NumDescriptors)
            return false;

         if (!heap->live_resource_words || !heap->live_resource_word_count) {
            *complete = false;
            return true;
         }

         /* A sized root range ends; only the heap-directly-indexed case
          * genuinely owns the tail. See d12_queue_read_set_add_heap. */
         uint32_t heap_limit = heap->desc.NumDescriptors;
         if (ref->heap.count &&
             ref->heap.count <= heap->desc.NumDescriptors - ref->heap.first)
            heap_limit = ref->heap.first + ref->heap.count;

         simple_mtx_lock(&heap->device->descriptor_heap_lock);
         uint32_t first_word = ref->heap.first >> 6;
         uint32_t last_word = (heap_limit - 1) >> 6;
         if (last_word >= heap->live_resource_word_count)
            last_word = heap->live_resource_word_count - 1;
         for (uint32_t w = first_word; w <= last_word; ++w) {
            uint64_t active = heap->live_resource_words[w];
            if (w == first_word)
               active &= UINT64_MAX << (ref->heap.first & 63);

            while (active) {
               uint32_t index = (w << 6) + u_bit_scan64(&active);
               if (index >= heap_limit)
                  break;
               if (!d12_queue_attachment_add_descriptor(
                      dev, attachments, count, limit,
                      &heap->slots[index])) {
                  *complete = false;
                  simple_mtx_unlock(
                     &heap->device->descriptor_heap_lock);
                  return true;
               }
            }
         }
         simple_mtx_unlock(&heap->device->descriptor_heap_lock);
         break;
      }
      case D12_WRITE_REF_RESOURCE: {
         struct d12_descriptor descriptor = {
            .resource = ref->resource,
            .kind = D12_DESC_UAV,
         };
         if (!d12_queue_attachment_add_descriptor(
                dev, attachments, count, limit, &descriptor)) {
            *complete = false;
            return true;
         }
         break;
      }
      default:
         return false;
      }
   }
   return true;
}

static bool
d12_queue_emit_write_set(struct d12_device *device,
                         const struct d12_cmd_list *list,
                         const struct d12_segment *seg,
                         enum d12_write_pipe pipe, uint8_t *payload,
                         size_t *payload_size, bool *complete)
{
   struct agx_device *dev = &device->dev;
   static_assert(sizeof(struct d12_write_range) ==
                 sizeof(struct drm_asahi_attachment),
                 "write range UAPI size");
   static_assert(offsetof(struct d12_write_range, pointer) ==
                 offsetof(struct drm_asahi_attachment, pointer),
                 "write range UAPI pointer offset");
   static_assert(offsetof(struct d12_write_range, size) ==
                 offsetof(struct drm_asahi_attachment, size),
                 "write range UAPI size offset");

   struct d12_write_range
      attachments[D12_SUBMIT_MAX_ATTACHMENTS] = {0};
   unsigned count = 0;
   const unsigned limit =
      MIN2(D12_SUBMIT_MAX_ATTACHMENTS, dev->params.max_attachments);
   bool pipe_complete = true;

   /* Internal GPU-written state is part of the ownership set too. */
   if (pipe == D12_WRITE_COMPUTE) {
      if (!d12_queue_attachment_add(
             dev, attachments, &count, limit, list->stream_bo->va->addr,
             list->stream_bo->size))
         pipe_complete = false;
      if (seg->scratch_cs_bo &&
          !d12_queue_attachment_add(
             dev, attachments, &count, limit,
             seg->scratch_cs_bo->va->addr, seg->scratch_cs_bo->size))
         pipe_complete = false;
      if (list->uses_heap && device->heap_bo &&
          !d12_queue_attachment_add(
             dev, attachments, &count, limit, device->heap_bo->va->addr,
             device->heap_bo->size))
         pipe_complete = false;
   } else if (pipe == D12_WRITE_VERTEX) {
      if (seg->scratch_vs_bo &&
          !d12_queue_attachment_add(
             dev, attachments, &count, limit,
             seg->scratch_vs_bo->va->addr, seg->scratch_vs_bo->size))
         pipe_complete = false;
   } else {
      if (seg->scratch_fs_bo &&
          !d12_queue_attachment_add(
             dev, attachments, &count, limit,
             seg->scratch_fs_bo->va->addr, seg->scratch_fs_bo->size))
         pipe_complete = false;

      for (unsigned i = 0; i < seg->rt_count; i++) {
         if (!(seg->rt_write_mask & BITFIELD_BIT(i)) || !seg->rt[i])
            continue;
         struct d12_descriptor target = {
            .resource = seg->rt[i],
            .kind = D12_DESC_UAV,
         };
         if (!d12_queue_attachment_add_descriptor(
                dev, attachments, &count, limit, &target))
            pipe_complete = false;
      }

      if (seg->ds) {
         const bool depth_store =
            seg->depth_cleared ||
            !(seg->dsv_flags & D3D12_DSV_FLAG_READ_ONLY_DEPTH);
         const bool stencil_store =
            seg->ds->stencil_bo &&
            (seg->stencil_cleared ||
             !(seg->dsv_flags & D3D12_DSV_FLAG_READ_ONLY_STENCIL));
         struct d12_descriptor target = {
            .resource = seg->ds,
            .kind = D12_DESC_UAV,
         };
         if (depth_store &&
             !d12_queue_attachment_add_descriptor(
                dev, attachments, &count, limit, &target))
            pipe_complete = false;
         if (stencil_store) {
            target.texture_plane = 1;
            if (!d12_queue_attachment_add_descriptor(
                   dev, attachments, &count, limit, &target))
               pipe_complete = false;
         }
      }
   }

   if (!d12_queue_collect_write_tracker(
          dev, &seg->writes[pipe], attachments, &count, limit,
          &pipe_complete))
      return false;
   *complete &= pipe_complete;
   if (!count)
      return true;

   const enum drm_asahi_cmd_type type =
      pipe == D12_WRITE_VERTEX ? DRM_ASAHI_SET_VERTEX_ATTACHMENTS
      : pipe == D12_WRITE_FRAGMENT
         ? DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS
         : DRM_ASAHI_SET_COMPUTE_ATTACHMENTS;
   const struct drm_asahi_cmd_header header = {
      .cmd_type = type,
      .size = count * sizeof(attachments[0]),
      .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
      .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
   };
   memcpy(payload + *payload_size, &header, sizeof(header));
   *payload_size += sizeof(header);
   memcpy(payload + *payload_size, attachments,
          count * sizeof(attachments[0]));
   *payload_size += count * sizeof(attachments[0]);
   return true;
}

static void
d12_queue_op_capture_bo(struct d12_queue_op *op, struct agx_bo *bo)
{
   if (!bo)
      return;

   util_dynarray_foreach(&op->bos, struct agx_bo *, captured) {
      if (*captured == bo)
         return;
   }

   struct agx_bo **slot = util_dynarray_grow(&op->bos, struct agx_bo *, 1);
   if (!slot) {
      op->capture_failed = true;
      return;
   }
   agx_bo_reference(bo);
   *slot = bo;
}

static void
d12_queue_op_capture_bos(struct d12_queue_op *op,
                         struct d12_cmd_list *list)
{
   util_dynarray_init(&op->bos, NULL);
   util_dynarray_init(&op->g17p_vdm_storage, NULL);
   op->bos_initialized = true;
   op->submit_device = list->device;

   d12_queue_op_capture_bo(op, list->stream_bo);
   d12_queue_op_capture_bo(op, list->usc_bo);
   d12_queue_op_capture_bo(op, list->sparse_registry_bo);
   util_dynarray_foreach(&list->g17p_vdm_storage,
                         struct d12_g17p_vdm_storage *, storage) {
      struct d12_g17p_vdm_storage **slot = util_dynarray_grow(
         &op->g17p_vdm_storage, struct d12_g17p_vdm_storage *, 1);
      if (!slot) {
         op->capture_failed = true;
         return;
      }
      d12_g17p_vdm_storage_ref(*storage);
      *slot = *storage;
      d12_queue_op_capture_bo(op, (*storage)->bo);
   }

   /* Scratch addresses are packed directly into the DRM command payload. */
   for (unsigned i = 0; i < list->segment_count; i++) {
      struct agx_bo *scratch[] = {
         list->segments[i].scratch_cs_bo,
         list->segments[i].scratch_vs_bo,
         list->segments[i].scratch_fs_bo,
      };

      for (unsigned j = 0; j < ARRAY_SIZE(scratch); j++) {
         d12_queue_op_capture_bo(op, scratch[j]);
      }
   }
}

static bool
d12_queue_read_set_add_range(struct d12_queue_op *op, uint64_t gpu_va,
                             uint64_t length)
{
   return d12_read_set_add(op->exact_read_ranges,
                           &op->exact_read_range_count,
                           ARRAY_SIZE(op->exact_read_ranges), gpu_va,
                           length) == D12_READ_SET_OK;
}

static bool
d12_queue_read_set_add_bo(struct d12_queue_op *op, struct agx_bo *bo)
{
   if (!bo || !bo->va || !bo->size)
      return false;

   /* Retention and cache ownership are one KMD contract. Keep the exact BO
    * alive in the queued operation even when the application resets its list
    * as soon as ExecuteCommandLists returns. */
   d12_queue_op_capture_bo(op, bo);
   return d12_queue_read_set_add_range(op, bo->va->addr, bo->size);
}

static bool
d12_queue_read_set_add_bo_range(struct d12_queue_op *op, struct agx_bo *bo,
                                uint64_t offset, uint64_t length)
{
   if (!bo || !bo->va || !bo->size)
      return false;

   d12_queue_op_capture_bo(op, bo);
   return d12_read_set_add_allocation(
             op->exact_read_ranges, &op->exact_read_range_count,
             ARRAY_SIZE(op->exact_read_ranges), bo->va->addr, bo->size,
             offset, length) == D12_READ_SET_OK;
}

static bool
d12_queue_read_set_resolve_range(struct d12_queue_op *op,
                                 struct agx_device *dev, uint64_t address,
                                 uint64_t length)
{
   if (!address)
      return false;

   struct agx_bo *owner = NULL;
   uint64_t base = 0;
   uint64_t size = 0;

   /* Retain under the same lock which protects BO-map membership.  Merely
    * returning a pointer and referencing it after unlock races final BO
    * release on applications which legally recycle command-list state. */
   simple_mtx_lock(&dev->bo_map_lock);
   for (uint32_t handle = 0; handle < dev->max_handle; ++handle) {
      struct agx_bo *bo = agx_lookup_bo(dev, handle);
      if (!bo->va || !bo->size)
         continue;

      const uint64_t bo_base = bo->va->addr;
      if (address < bo_base || address - bo_base >= bo->size)
         continue;

      if (length && length > bo->size - (address - bo_base))
         break;

      agx_bo_reference(bo);
      owner = bo;
      base = bo_base;
      size = bo->size;
      break;
   }
   simple_mtx_unlock(&dev->bo_map_lock);

   if (!owner)
      return false;

   /* Use the whole owner allocation.  This is still an exact retained set,
    * and it safely covers texture swizzles, placed-resource aliases and raw
    * root descriptors whose declared byte size is not shader-bounds proof. */
   const bool added = d12_queue_read_set_add_range(op, base, size);
   if (added)
      d12_queue_op_capture_bo(op, owner);
   agx_bo_unreference(dev, owner);
   return added;
}

static bool
d12_queue_read_set_add_resource(struct d12_queue_op *op,
                                struct d12_resource *resource)
{
   if (!resource)
      return true;

   if (resource->reserved) {
      bool complete = true;
      if (resource->sparse_map)
         complete &= d12_queue_read_set_add_bo(op, resource->sparse_map);

      /* Tile mappings are mutable queue state. Resolve them while the queue
       * operation is built and retain each backing allocation. Publish only
       * the mapped 64 KiB heap spans to the KMD. If 256 disjoint spans would
       * overflow the ABI, d12_read_set_add_allocation first widens an already
       * represented heap so its fragments coalesce; only an allocation that
       * still cannot fit makes the complete set fall back. */
      simple_mtx_lock(&resource->sparse_lock);
      for (uint32_t i = 0; i < resource->sparse_tile_count; ++i) {
         const struct d12_sparse_mapping mapping =
            resource->sparse_mappings[i];
         struct agx_bo *bo = mapping.bo;
         if (bo)
            complete &= d12_queue_read_set_add_bo_range(
               op, bo, mapping.heap_offset_B,
               D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES);
      }
      simple_mtx_unlock(&resource->sparse_lock);
      return complete;
   }

   if (!resource->bo)
      return false;

   bool complete = d12_queue_read_set_add_bo(op, resource->bo);
   if (resource->stencil_bo)
      complete &= d12_queue_read_set_add_bo(op, resource->stencil_bo);
   return complete;
}

static bool
d12_queue_read_set_add_descriptor(
   struct d12_queue_op *op, struct agx_device *dev,
   const struct d12_descriptor *descriptor)
{
   if (!descriptor || descriptor->kind == D12_DESC_NONE ||
       descriptor->kind == D12_DESC_SAMPLER)
      return true;

   bool complete = true;
   if (descriptor->resource) {
      complete &= d12_queue_read_set_add_resource(op, descriptor->resource);
   } else if (descriptor->address) {
      complete &= d12_queue_read_set_resolve_range(
         op, dev, descriptor->address, descriptor->size_B);
   }

   if (descriptor->counter_address) {
      complete &= d12_queue_read_set_resolve_range(
         op, dev, descriptor->counter_address, sizeof(uint32_t));
   }
   return complete;
}

static bool
d12_queue_read_set_add_heap(struct d12_queue_op *op,
                            struct agx_device *dev,
                            const struct d12_read_ref *ref)
{
   const struct d12_descriptor_heap *heap = ref->heap.heap;
   if (!heap || ref->heap.first >= heap->desc.NumDescriptors)
      return false;

   bool complete = true;
   complete &= d12_queue_read_set_add_bo(op, heap->bo);

   if (ref->heap.resources) {
      complete &= d12_queue_read_set_add_bo(op, heap->texture_bo);
      complete &= d12_queue_read_set_add_bo(op, heap->pbe_bo);
      complete &= d12_queue_read_set_add_bo(op, heap->residency_bo);
      if (!heap->live_resource_words || !heap->live_resource_word_count)
         return false;

      /* Descriptor writes and bitmap maintenance use the same device lock.
       * This gives Execute a coherent descriptor snapshot while reducing a
       * one-million-slot heap scan to the number of populated descriptors.
       */
      /* A root range the signature sized explicitly ends. Walking to the
       * end of the heap would pull every later populated descriptor's
       * allocation into this submission's read set, which is both a much
       * larger WDDM cache-maintenance cost and the fastest way to hit the
       * 256-range submit ceiling. */
      uint32_t limit = heap->desc.NumDescriptors;
      if (ref->heap.count &&
          ref->heap.count <= heap->desc.NumDescriptors - ref->heap.first)
         limit = ref->heap.first + ref->heap.count;

      simple_mtx_lock(&heap->device->descriptor_heap_lock);
      uint32_t first_word = ref->heap.first >> 6;
      uint32_t last_word = (limit - 1) >> 6;
      if (last_word >= heap->live_resource_word_count)
         last_word = heap->live_resource_word_count - 1;
      for (uint32_t w = first_word; w <= last_word; ++w) {
         uint64_t active = heap->live_resource_words[w];
         if (w == first_word)
            active &= UINT64_MAX << (ref->heap.first & 63);

         while (active) {
            uint32_t index = (w << 6) + u_bit_scan64(&active);
            if (index >= limit)
               break;
            complete &= d12_queue_read_set_add_descriptor(
               op, dev, &heap->slots[index]);
         }
      }
      simple_mtx_unlock(&heap->device->descriptor_heap_lock);
   } else {
      complete &= d12_queue_read_set_add_bo(op, heap->sampler_bo);
   }
   return complete;
}

static void
d12_queue_op_build_read_set(struct d12_queue_op *op,
                            struct d12_cmd_list *list)
{
   struct d12_device *device = list->device;
   struct agx_device *dev = &device->dev;
   bool complete = !list->reads.incomplete;

   static_assert(sizeof(struct d12_read_range) ==
                    sizeof(struct agx_submit_read_range),
                 "frontend read-range ABI width");
   static_assert(offsetof(struct d12_read_range, gpu_va) ==
                    offsetof(struct agx_submit_read_range, gpu_va) &&
                    offsetof(struct d12_read_range, length) ==
                    offsetof(struct agx_submit_read_range, length),
                 "frontend read-range ABI offsets");

   /* Storage and scratch were captured before this call. Iterate only the
    * initial vector: collecting tracker references below may grow op->bos. */
   const unsigned captured_count =
      util_dynarray_num_elements(&op->bos, struct agx_bo *);
   struct agx_bo **captured = util_dynarray_begin(&op->bos);
   for (unsigned i = 0; i < captured_count; ++i)
      complete &= d12_queue_read_set_add_bo(op, captured[i]);

   util_dynarray_foreach(&op->g17p_vdm_storage,
                         struct d12_g17p_vdm_storage *, storage)
      complete &= d12_queue_read_set_add_range(op, (*storage)->alias,
                                              (*storage)->bo->size);

   /* Every render command names the occlusion buffer even when counting is
    * disabled. Device rodata contains the null region and heap descriptor
    * consumed by ordinary shaders and libagx helpers. */
   complete &= d12_queue_read_set_add_bo(op, device->rodata_bo);
   if (op->nr_vdm)
      complete &= d12_queue_read_set_add_bo(op, device->oclqry_bo);
   if (list->uses_heap)
      complete &= d12_queue_read_set_add_bo(op, device->heap_bo);

   /* Process-root BO wrappers intentionally have no agx_va. Their fixed
    * addresses are nevertheless part of the complete retained set. */
   if (dev->zero_bo) {
      d12_queue_op_capture_bo(op, dev->zero_bo);
      complete &= d12_queue_read_set_add_range(
         op, AGX_ZERO_PAGE_ADDRESS, dev->zero_bo->size);
   }
   if (dev->scratch_bo) {
      d12_queue_op_capture_bo(op, dev->scratch_bo);
      complete &= d12_queue_read_set_add_range(
         op, AGX_SCRATCH_PAGE_ADDRESS, dev->scratch_bo->size);
   }
   struct agx_bo *printf_bo = dev->printf.bo;
   if (printf_bo) {
      d12_queue_op_capture_bo(op, printf_bo);
      complete &= printf_bo->va
                     ? d12_queue_read_set_add_bo(op, printf_bo)
                     : d12_queue_read_set_add_range(
                          op, dev->printf_buffer_address, printf_bo->size);
   }
   if (dev->g17p_render_context_bo) {
      d12_queue_op_capture_bo(op, dev->g17p_render_context_bo);
      complete &= d12_queue_read_set_add_range(
         op, AGX_G17P_RENDER_CONTEXT_BASE,
         dev->g17p_render_context_bo->size);
   }

   if (list->reads.initialized) {
      util_dynarray_foreach(&list->reads.refs, struct d12_read_ref, ref) {
         switch (ref->kind) {
         case D12_READ_REF_BO:
            complete &= d12_queue_read_set_add_bo(op, ref->bo);
            break;
         case D12_READ_REF_RANGE:
            complete &= d12_queue_read_set_resolve_range(
               op, dev, ref->range.address, ref->range.size);
            break;
         case D12_READ_REF_DESCRIPTOR:
            complete &= d12_queue_read_set_add_descriptor(
               op, dev, ref->descriptor);
            break;
         case D12_READ_REF_HEAP:
            complete &= d12_queue_read_set_add_heap(op, dev, ref);
            break;
         case D12_READ_REF_RESOURCE:
            complete &= d12_queue_read_set_add_resource(op, ref->resource);
            break;
         default:
            complete = false;
            break;
         }
      }
   }

   op->read_set_complete = complete;
   if (!complete) {
      op->exact_read_range_count = 0;
      if (getenv("D3D12AGX_DUMP_READ_SET"))
         mesa_logw("ExecuteCommandLists: exact read set incomplete; using "
                   "the conservative WDDM allocation scan");
   } else if (getenv("D3D12AGX_DUMP_READ_SET")) {
      mesa_logi("ExecuteCommandLists: exact read set contains %u canonical "
                "ranges", op->exact_read_range_count);
   }
}

static bool
d12_queue_enqueue_locked(struct d12_queue *queue, struct d12_queue_op *op)
{
   if (p_atomic_read(&queue->stop))
      return false;

   if (queue->op_tail)
      queue->op_tail->next = op;
   else
      queue->op_head = op;
   queue->op_tail = op;
   cnd_signal(&queue->work_available);
   return true;
}

static void
d12_queue_retire_enqueue(struct d12_queue *queue, struct d12_queue_op *op)
{
   op->next = NULL;
   d12_sync_lock(&queue->lock);
   if (queue->retire_tail)
      queue->retire_tail->next = op;
   else
      queue->retire_head = op;
   queue->retire_tail = op;
   cnd_signal(&queue->retire_available);
   mtx_unlock(&queue->lock);
}

_Static_assert(sizeof(struct libagx_write_u64_args) == 16,
               "queue signal arguments must fit their reserved stream slot");

static HRESULT
d12_queue_signal_arena_init(struct d12_queue *queue)
{
   struct agx_device *dev = &queue->device->dev;
   const bool native = dev->chip == AGX_CHIP_G17P;
   if (!native && agx_chip_command_stream_generation(dev->chip) !=
                     AGX_CS_GENERATION_AGX2) {
      mesa_loge("queue-signal producer unavailable for %s",
                agx_chip_name(dev->chip));
      return E_NOTIMPL;
   }

   struct agx_precompiled_shader *program =
      agx_get_precompiled(&queue->device->bg_eot, LIBAGX_WRITE_U64);
   struct agx_bo *stream = NULL;
   struct agx_bo *usc = NULL;
   struct agx_bo *scratch = NULL;
   uint8_t *stream_map;
   uint8_t *usc_map;
   uint32_t common_control_size = 0;
   const uint32_t usc_size = native ? AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE
                                    : agx_usc_size(15);
   const uint32_t usc_stride = native ? AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE
                                      : D12_QUEUE_SIGNAL_USC_STRIDE;
   const uint32_t usc_arena_size = native
      ? usc_stride * D12_QUEUE_SIGNAL_SLOT_COUNT : D12_QUEUE_SIGNAL_ARENA_SIZE;

   if (!program || !program->bo || !dev->params.max_attachments)
      return E_FAIL;

   /* Only the precompiler's native code container and fully described Ioto
    * ABI may pass the G17P producer gate. No legacy USC words are reused. */
   if (native &&
       (!program->info.main_size || !program->info.nr_gprs ||
        program->info.nr_preamble_gprs || program->info.printf_uniform ||
        !program->info.agx3_staging_known ||
        (program->info.rodata.size_16 && !program->rodata_offset) ||
        program->entry_offset > UINT64_MAX - program->ptr ||
        program->rodata_offset > UINT64_MAX - program->ptr ||
        program->b.workgroup.x != 1 || program->b.workgroup.y != 1 ||
        program->b.workgroup.z != 1))
      return E_FAIL;

   stream = agx_bo_create(dev, D12_QUEUE_SIGNAL_ARENA_SIZE, 0,
                          AGX_BO_WRITEBACK,
                          "d3d12 ddi fence signal stream arena");
   usc = agx_bo_create(dev, usc_arena_size, 0,
                       AGX_BO_LOW_VA | AGX_BO_WRITEBACK,
                       "d3d12 ddi fence signal usc arena");
   scratch = agx_bo_create(dev, AIL_PAGESIZE, 0, AGX_BO_WRITEBACK,
                           "d3d12 ddi queue dependency scratch");
   if (!stream || !usc || !scratch) {
      agx_bo_unreference(dev, stream);
      agx_bo_unreference(dev, usc);
      agx_bo_unreference(dev, scratch);
      return E_OUTOFMEMORY;
   }

   stream_map = agx_bo_map(stream);
   usc_map = agx_bo_map(usc);
   if (!stream_map || !usc_map || !agx_bo_map(scratch))
      goto fail;
   memset(stream_map, 0, stream->size);
   memset(usc_map, 0, usc->size);

   for (uint32_t slot = 0; slot < D12_QUEUE_SIGNAL_SLOT_COUNT; ++slot) {
      uint32_t control_offset;
      uint32_t data_offset;
      uint32_t usc_offset;
      /* The pack helpers have no output-size parameter.  Encode into a
       * generously sized scratch page first, then reject anything that does
       * not fit before copying into the tightly packed arena slot. */
      uint32_t control_words[4096 / sizeof(uint32_t)] = {0};

      if (!d12_queue_signal_slot_layout_program(
             slot, 0, usc_size, usc_stride, usc_arena_size,
             &control_offset, &data_offset, &usc_offset))
         goto fail;

      uint8_t *control;
      if (native) {
         struct agx3_ioto_uniform_range ranges[2] = {{
            .start_halfs = AGX_ABI_AGX3_PRECOMP_ARG_HALFS,
            .size_halfs = sizeof(struct libagx_write_u64_args) / 2,
            .address = stream->va->addr + data_offset,
         }};
         unsigned range_count = 1;
         if (program->info.rodata.size_16) {
            ranges[range_count++] = (struct agx3_ioto_uniform_range){
               .start_halfs = program->info.rodata.base_uniform,
               .size_halfs = program->info.rodata.size_16,
               .address = program->ptr + program->rodata_offset,
            };
         }
         struct agx3_cdm_direct_template_sources launch_sources;
         if (!agx3_cdm_direct_template_sources_compute(false, false, false,
                                                       &launch_sources))
            goto fail;
         const struct agx3_g17p_compute_program_desc desc = {
            .program_va = usc->va->addr + usc_offset,
            .shader_va = program->ptr + program->entry_offset,
            .uniform_ranges = ranges,
            .uniform_range_count = range_count,
            .shader_binary_size = program->info.main_size,
            .scratch_size = program->info.agx3_tps_size,
            .nr_gprs = program->info.nr_gprs,
            .compiler_d_valid = program->info.agx3_staging_known,
            .compiler_d = (uint32_t)program->info.agx3_staging_halfs * 2,
            .threadgroup_memory_size_B = program->info.local_size
               ? MAX2(256u, ALIGN_POT(program->info.local_size, 64u)) : 0,
            .launch_sources = launch_sources,
            .ldshdr_f1 = AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
         };
         struct agx3_g17p_compute_program generated;
         if (!agx_va_in_usc_window(dev, desc.program_va) ||
             !agx_va_in_usc_window(dev, desc.shader_va) ||
             !agx3_g17p_compute_program_generate(dev->chip, &desc, &generated) ||
             generated.length > usc_size)
            goto fail;
         memcpy(usc_map + usc_offset, generated.bytes, generated.length);
         if (!slot && getenv("D3D12AGX_DUMP_SYNC")) {
            mesa_logi("native queue signal producer: code=%u gprs=%u "
                      "staging_halfs=%u tps=%u rodata_halfs=%u ioto=%u "
                      "stride=%u slots=%u local=1x1x1",
                      program->info.main_size, program->info.nr_gprs,
                      program->info.agx3_staging_halfs,
                      program->info.agx3_tps_size,
                      program->info.rodata.size_16, generated.length,
                      usc_stride, D12_QUEUE_SIGNAL_SLOT_COUNT);
         }
         const struct agx3_compute_direct_grid grid = {
            .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
            .size = {1, 1, 1},
            .local_size = {program->b.workgroup.x, program->b.workgroup.y,
                           program->b.workgroup.z},
            .dimensions = program->b.workgroup.z != 1 ? 3
               : program->b.workgroup.y != 1 ? 2 : 1,
         };
         struct agx3_compute_dispatch_sink sink = {
            .data = (uint8_t *)control_words,
            .capacity = sizeof(control_words),
         };
         if (!agx3_g17p_native_compute_dispatch_append_with_barrier(
                &generated.launch, &grid, 0x5, &sink))
            goto fail;
         control = (uint8_t *)control_words + sink.length;
      } else {
         agx_usc_words_precomp((uint32_t *)(usc_map + usc_offset), &program->b,
                               stream->va->addr + data_offset,
                               sizeof(struct libagx_write_u64_args));
         control = (uint8_t *)agx_cdm_launch(
            control_words, dev->chip, agx_1d(1), program->b.workgroup,
            program->b.launch,
            agx_usc_addr(dev, usc->va->addr + usc_offset));
         control = d12_queue_emit_cdm_cache_flush(dev, control);
      }
      if (!control)
         goto fail;
      const uint32_t control_size =
         (uint32_t)(control - (uint8_t *)control_words);
      if (!d12_queue_signal_slot_layout_program(
             slot, control_size, usc_size, usc_stride, usc_arena_size,
             &control_offset, &data_offset, &usc_offset) ||
          (common_control_size && common_control_size != control_size))
         goto fail;
      common_control_size = control_size;
      memcpy(stream_map + control_offset, control_words, control_size);
   }

   if (!common_control_size ||
       agx_bo_sync_for_device(usc, 0, usc->size) ||
       agx_bo_sync_for_device(stream, 0, stream->size))
      goto fail;
   queue->signal_stream_bo = stream;
   queue->signal_usc_bo = usc;
   queue->dependency_scratch_bo = scratch;
   agx_bo_reference(program->bo);
   queue->signal_program_bo = program->bo;
   queue->signal_control_size = common_control_size;
   queue->signal_program_stride = usc_stride;
   queue->signal_program_size = usc_size;
   return S_OK;

fail:
   agx_bo_unreference(dev, stream);
   agx_bo_unreference(dev, usc);
   agx_bo_unreference(dev, scratch);
   return E_FAIL;
}

static void
d12_queue_signal_arena_cleanup(struct d12_queue *queue)
{
   struct agx_device *dev = &queue->device->dev;

   agx_bo_unreference(dev, queue->signal_stream_bo);
   agx_bo_unreference(dev, queue->signal_usc_bo);
   agx_bo_unreference(dev, queue->signal_program_bo);
   agx_bo_unreference(dev, queue->dependency_scratch_bo);
   queue->signal_stream_bo = NULL;
   queue->signal_usc_bo = NULL;
   queue->signal_program_bo = NULL;
   queue->dependency_scratch_bo = NULL;
   queue->signal_control_size = 0;
   queue->signal_program_stride = 0;
   queue->signal_program_size = 0;
   queue->signal_slots = 0;
}

static void
d12_queue_release_signal_slot(struct d12_queue *queue,
                              struct d12_queue_op *op)
{
   if (!op->signal_slot_valid)
      return;

   d12_sync_lock(&queue->lock);
   const bool released =
      d12_queue_signal_slot_release(&queue->signal_slots, op->signal_slot);
   op->signal_slot_valid = false;
   cnd_broadcast(&queue->retire_available);
   mtx_unlock(&queue->lock);
   if (!released) {
      mesa_loge("command queue signal-slot accounting failure");
      d12_device_mark_removed(queue->device, DXGI_ERROR_DEVICE_REMOVED);
   }
}

static bool
d12_queue_prepare_internal_write(struct d12_queue *queue,
                                 struct d12_queue_op *op,
                                 uint64_t destination, uint64_t value)
{
   uint32_t control_offset;
   uint32_t data_offset;
   uint32_t usc_offset;
   int slot;

   if (!destination || !queue->signal_stream_bo ||
       !queue->signal_usc_bo || !queue->signal_program_bo)
      return false;

   d12_sync_lock(&queue->lock);
   while ((slot = d12_queue_signal_slot_acquire(&queue->signal_slots)) < 0 &&
          !p_atomic_read(&queue->stop)) {
      d12_sync_wait(&queue->retire_available, &queue->lock);
   }
   mtx_unlock(&queue->lock);
   if (slot < 0)
      return false;

   op->signal_slot = (uint32_t)slot;
   op->signal_slot_valid = true;
   if (!d12_queue_signal_slot_layout_program(
          op->signal_slot, queue->signal_control_size,
          queue->signal_program_size, queue->signal_program_stride,
          queue->signal_usc_bo->size, &control_offset, &data_offset,
          &usc_offset)) {
      d12_queue_release_signal_slot(queue, op);
      return false;
   }
   (void)usc_offset;

   const struct libagx_write_u64_args shader_args = {
      .address = destination,
      .value = value,
   };
   memcpy((uint8_t *)agx_bo_map(queue->signal_stream_bo) + data_offset,
          &shader_args, sizeof(shader_args));
   if (agx_bo_sync_for_device(queue->signal_stream_bo, data_offset,
                              sizeof(shader_args))) {
      d12_queue_release_signal_slot(queue, op);
      return false;
   }

   op->submit_device = queue->device;
   op->payload_size = 2 * sizeof(struct drm_asahi_cmd_header) +
                      sizeof(struct drm_asahi_attachment) +
                      sizeof(struct drm_asahi_cmd_compute);
   if (op->payload_size > sizeof(op->inline_payload)) {
      d12_queue_release_signal_slot(queue, op);
      return false;
   }
   op->payload = op->inline_payload;
   memset(op->payload, 0, op->payload_size);

   const struct drm_asahi_cmd_header attachment_header = {
      .cmd_type = DRM_ASAHI_SET_COMPUTE_ATTACHMENTS,
      .size = sizeof(struct drm_asahi_attachment),
      .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
      .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
   };
   const struct drm_asahi_attachment attachment = {
      .pointer = destination,
      .size = sizeof(uint64_t),
   };
   const struct drm_asahi_cmd_header header = agx_cmd_header(true, 0, 0);
   const struct drm_asahi_cmd_compute command = {
      .cdm_ctrl_stream_base =
         queue->signal_stream_bo->va->addr + control_offset,
      .cdm_ctrl_stream_end = queue->signal_stream_bo->va->addr +
                             control_offset + queue->signal_control_size,
   };
   size_t payload_offset = 0;
   memcpy((uint8_t *)op->payload + payload_offset, &attachment_header,
          sizeof(attachment_header));
   payload_offset += sizeof(attachment_header);
   memcpy((uint8_t *)op->payload + payload_offset, &attachment,
          sizeof(attachment));
   payload_offset += sizeof(attachment);
   memcpy((uint8_t *)op->payload + payload_offset, &header, sizeof(header));
   payload_offset += sizeof(header);
   memcpy((uint8_t *)op->payload + payload_offset, &command,
          sizeof(command));
   op->nr_cdm = 1;
   op->write_set_complete = true;
   /* libagx_write_u64 performs one store. Its complete read footprint is the
    * control/argument stream, the USC records, and the pool BO containing
    * immutable code/rodata. The destination belongs only to the write set. */
   op->read_set_complete = true;
   if (!d12_queue_read_set_add_range(
          op, queue->signal_stream_bo->va->addr,
          queue->signal_stream_bo->size) ||
       !d12_queue_read_set_add_range(
          op, queue->signal_usc_bo->va->addr,
          queue->signal_usc_bo->size) ||
       !d12_queue_read_set_add_range(
          op, queue->signal_program_bo->va->addr,
          queue->signal_program_bo->size)) {
      op->read_set_complete = false;
      op->exact_read_range_count = 0;
      return false;
   }
   return true;
}

static bool
d12_queue_prepare_placement_signal(struct d12_queue *queue,
                                   struct d12_queue_op *op)
{
   const struct d12_placement_fence *fence = op->placement_fence;

   return fence && fence->count == 1 &&
          d12_queue_prepare_internal_write(
             queue, op, fence->entries[0].current_gpu_address, op->value);
}

static bool
d12_queue_prepare_placement_barrier(struct d12_queue *queue,
                                    struct d12_queue_op *op)
{
   return queue->dependency_scratch_bo &&
          d12_queue_prepare_internal_write(
             queue, op, queue->dependency_scratch_bo->va->addr,
             d12_queue_timeline_marker(&queue->timeline));
}

static bool
d12_queue_take_pending_placement_waits(struct d12_queue *queue,
                                       struct d12_queue_op *op)
{
   if (!queue || !op ||
       queue->pending_placement_wait_count >
          ARRAY_SIZE(op->placement_waits))
      return false;

   op->placement_wait_count = queue->pending_placement_wait_count;
   memcpy(op->placement_waits, queue->pending_placement_waits,
          (size_t)op->placement_wait_count * sizeof(op->placement_waits[0]));
   memset(queue->pending_placement_waits, 0,
          (size_t)queue->pending_placement_wait_count *
             sizeof(queue->pending_placement_waits[0]));
   queue->pending_placement_wait_count = 0;
   return true;
}

static bool
d12_queue_process_submit(struct d12_queue *queue, struct d12_queue_op *op)
{
   struct d12_device *device = queue->device;
   struct agx_device *dev = &device->dev;

   /* ASAHI_MESA_DEBUG=trace: decode the exact command stream which will be
    * submitted, not a reconstructed view of it. */
   if (dev->debug & AGX_DBG_TRACE) {
      struct util_dynarray wrapped;
      util_dynarray_init(&wrapped, NULL);
      void *copy =
         util_dynarray_grow_bytes(&wrapped, 1, op->payload_size);
      memcpy(copy, op->payload, op->payload_size);
      agxdecode_drm_cmdbuf(dev->agxdecode, &dev->params, &wrapped, true);
      util_dynarray_fini(&wrapped);
      agxdecode_next_frame();
   }

   uint64_t in = d12_queue_timeline_marker(&queue->timeline);
   if (in == UINT64_MAX) {
      mesa_loge("command queue timeline exhausted");
      d12_device_mark_removed(device, DXGI_ERROR_DEVICE_REMOVED);
      return false;
   }
   uint64_t out = in + 1;

   struct drm_asahi_sync syncs[2] = {
      {
         .sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
         .handle = queue->syncobj,
         .timeline_value = in,
      },
      {
         .sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
         .handle = queue->syncobj,
         .timeline_value = out,
      },
   };
   struct drm_asahi_submit submit = {
      .flags = dev->write_set_contract && op->write_set_complete
                  ? AGX_SUBMIT_WRITE_SET_COMPLETE
                  : 0,
      .queue_id = queue->agx_queue_id,
      .in_sync_count = 1,
      .out_sync_count = 1,
      .cmdbuf_size = op->payload_size,
      .syncs = (uint64_t)(uintptr_t)syncs,
      .cmdbuf = (uint64_t)(uintptr_t)op->payload,
   };

   /* The kernel/firmware accept at most 64 commands in one submit ioctl.
    * Follow Honeykrisp's queue_submit_looped model: slice the packed payload,
    * make barriers ioctl-relative, wait only on the first slice and signal
    * only from the last.  The queue itself preserves slice order. */
   const unsigned command_count = op->nr_vdm + op->nr_cdm;
   const unsigned max_commands = 64;
   uint8_t *cmdbuf = op->payload;
   uint32_t offs = 0;
   unsigned submitted_vdm = 0, submitted_cdm = 0;
   unsigned commands_remaining = command_count;
   int ret = 0;
   struct agx_submit_fence_dependency
      placement_dependencies[D12_QUEUE_MAX_PLACEMENT_DEPENDENCIES];
   uint32_t placement_wait_count = 0;
   bool placement_signal_described = false;

   for (uint32_t i = 0; i < op->placement_wait_count; ++i) {
      const struct d12_placement_fence *fence =
         op->placement_waits[i].fence;
      if (!fence || fence->count != 1 ||
          placement_wait_count >= ARRAY_SIZE(placement_dependencies)) {
         mesa_loge("command queue placement wait set is invalid");
         d12_device_mark_removed(device, DXGI_ERROR_INVALID_CALL);
         return false;
      }
      const struct d12_placement_fence_entry *entry = &fence->entries[0];
      placement_dependencies[placement_wait_count++] =
         (struct agx_submit_fence_dependency){
            .current_gpu_va = entry->current_gpu_address,
            .monitored_gpu_va = entry->monitored_gpu_address,
            .value = op->placement_waits[i].value,
            .flags = AGX_SUBMIT_FENCE_WAIT |
                     (entry->flags ? AGX_SUBMIT_FENCE_BOTTOM_OF_PIPE : 0),
         };
   }
   if (op->type == D12_QUEUE_OP_PLACEMENT_SIGNAL &&
       op->placement_fence && op->placement_fence->count == 1 &&
       placement_wait_count < ARRAY_SIZE(placement_dependencies)) {
      const struct d12_placement_fence_entry *entry =
         &op->placement_fence->entries[0];
      placement_dependencies[placement_wait_count] =
         (struct agx_submit_fence_dependency){
            .current_gpu_va = entry->current_gpu_address,
            .monitored_gpu_va = entry->monitored_gpu_address,
            .value = op->value,
            .flags = AGX_SUBMIT_FENCE_SIGNAL |
                     (entry->flags ? AGX_SUBMIT_FENCE_BOTTOM_OF_PIPE : 0),
         };
      placement_signal_described = true;
   }

   while (commands_remaining) {
      bool first = commands_remaining == command_count;
      unsigned count = MIN2(commands_remaining, max_commands);
      commands_remaining -= count;
      bool last = commands_remaining == 0;
      uint32_t base_offs = offs;
      unsigned vdm_count = 0, cdm_count = 0;

      unsigned hardware_count = 0;
      struct drm_asahi_cmd_header *first_hardware = NULL;
      while (hardware_count < count) {
         struct drm_asahi_cmd_header *cmd = (void *)(cmdbuf + offs);
         offs += sizeof(*cmd) + cmd->size;

         if (cmd->cmd_type == DRM_ASAHI_SET_VERTEX_ATTACHMENTS ||
             cmd->cmd_type == DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS ||
             cmd->cmd_type == DRM_ASAHI_SET_COMPUTE_ATTACHMENTS) {
            if (cmd->vdm_barrier != DRM_ASAHI_BARRIER_NONE ||
                cmd->cdm_barrier != DRM_ASAHI_BARRIER_NONE) {
               ret = -EINVAL;
               break;
            }
            continue;
         }

         if (!first_hardware)
            first_hardware = cmd;
         hardware_count++;

         if (cmd->cmd_type == DRM_ASAHI_CMD_RENDER) {
            vdm_count++;
         } else if (cmd->cmd_type == DRM_ASAHI_CMD_COMPUTE) {
            cdm_count++;
         } else {
            ret = -EINVAL;
            break;
         }

         if (cmd->vdm_barrier != DRM_ASAHI_BARRIER_NONE) {
            assert(cmd->vdm_barrier >= submitted_vdm);
            cmd->vdm_barrier -= submitted_vdm;
         }
         if (cmd->cdm_barrier != DRM_ASAHI_BARRIER_NONE) {
            assert(cmd->cdm_barrier >= submitted_cdm);
            cmd->cdm_barrier -= submitted_cdm;
         }
      }
      if (ret)
         break;

      /* If the final slice starts with work from only one engine, make it
       * wait for both engines from the preceding ioctl before it publishes
       * the command-list completion point. */
      if (last && !first) {
         assert(first_hardware != NULL);
         if (first_hardware->vdm_barrier == DRM_ASAHI_BARRIER_NONE)
            first_hardware->vdm_barrier = 0;
         if (first_hardware->cdm_barrier == DRM_ASAHI_BARRIER_NONE)
            first_hardware->cdm_barrier = 0;
      }

      struct drm_asahi_submit slice = {
         .flags = submit.flags,
         .queue_id = submit.queue_id,
         .in_sync_count = first ? submit.in_sync_count : 0,
         .out_sync_count = last ? submit.out_sync_count : 0,
         .cmdbuf_size = offs - base_offs,
         .syncs = first ? submit.syncs
                        : submit.syncs +
                             sizeof(struct drm_asahi_sync) *
                                submit.in_sync_count,
         .cmdbuf = submit.cmdbuf + base_offs,
      };

      struct agx_submit_virt virt = {
         .ring_idx = 0,
         .flags = op->read_set_complete
                     ? AGX_SUBMIT_VIRT_READ_SET_COMPLETE
                     : 0,
         .read_range_count = op->exact_read_range_count,
         .read_ranges = (const struct agx_submit_read_range *)
                           op->exact_read_ranges,
         .fence_dependency_count =
            (first ? placement_wait_count : 0) +
            (last && placement_signal_described ? 1 : 0),
         .fence_dependencies =
            first ? placement_dependencies
                  : (last && placement_signal_described
                        ? &placement_dependencies[placement_wait_count]
                        : NULL),
      };
      ret = dev->ops.submit(dev, &slice, &virt);
      if (ret)
         break;

      submitted_vdm += vdm_count;
      submitted_cdm += cdm_count;
   }

   if (ret) {
      mesa_loge("DRM_IOCTL_ASAHI_SUBMIT failed: %d -- device marked removed",
                ret);
      d12_device_mark_removed(device, DXGI_ERROR_DEVICE_REMOVED);
      return false;
   }

   uint64_t committed_in;
   uint64_t committed_out;
   d12_sync_lock(&queue->lock);
   const bool timeline_committed =
      d12_queue_timeline_next_submit(&queue->timeline, &committed_in,
                                     &committed_out);
   mtx_unlock(&queue->lock);
   if (!timeline_committed ||
       committed_in != in || committed_out != out) {
      mesa_loge("command queue timeline accounting failure");
      d12_device_mark_removed(device, DXGI_ERROR_DEVICE_REMOVED);
      return false;
   }
   op->retire_value = out;
   if (getenv("D3D12AGX_DUMP_SYNC")) {
      uint64_t stream_address = 0;
      if (op->bos_initialized && util_dynarray_num_elements(
                                    &op->bos, struct agx_bo *) > 0) {
         stream_address = (*(struct agx_bo **)
                              util_dynarray_begin(&op->bos))->va->addr;
      } else if (op->signal_slot_valid && queue->signal_stream_bo) {
         stream_address = queue->signal_stream_bo->va->addr +
                          op->signal_slot *
                             D12_QUEUE_SIGNAL_STREAM_STRIDE;
      }
      mesa_logi("SYNC submit: queue %p agxq %u  in=%llu out=%llu  "
                "list %p stream_bo 0x%llx",
                (void *)queue, queue->agx_queue_id,
                (unsigned long long)in, (unsigned long long)out,
                (void *)op->list, (unsigned long long)stream_address);
   }

   return true;
}

static bool
d12_queue_wait_timeline(struct d12_queue *queue, uint64_t point)
{
   if (point == 0)
      return true;

   int ret = queue->device->sync->timeline_wait(
      queue->device->sync, &queue->syncobj, &point, 1, INT64_MAX,
      DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT, NULL);
   if (ret) {
      mesa_loge("queue GPU completion wait failed: %d", ret);
      d12_device_mark_removed(queue->device, DXGI_ERROR_DEVICE_REMOVED);
      return false;
   }

   return true;
}

static bool
d12_queue_wait_submit_capacity(struct d12_queue *queue)
{
   d12_sync_lock(&queue->lock);
   while (!d12_queue_timeline_has_capacity(
             &queue->timeline, D12_QUEUE_MAX_INFLIGHT_SUBMITS) &&
          !p_atomic_read(&queue->stop)) {
      d12_sync_wait(&queue->retire_available, &queue->lock);
   }
   const bool available = !p_atomic_read(&queue->stop);
   mtx_unlock(&queue->lock);
   return available;
}

/* Run opt-in readback diagnostics only after the retirement worker has
 * observed completion of the exact submission whose storage it still owns. */
static void
d12_queue_dump_completion(struct d12_device *device)
{
   if (device->dbg_gs_vb) {
      const float *f = (const float *)device->dbg_gs_vb;
      const uint32_t *u = (const uint32_t *)device->dbg_gs_vb;
      unsigned n = MIN2(device->dbg_gs_vb_size / 4, 32);
      mesa_logi("geometry vertex buffer (%zu bytes):", device->dbg_gs_vb_size);
      for (unsigned i = 0; i < n; i += 4) {
         mesa_logi("  slot %u: %8.4f %8.4f %8.4f %8.4f  "
                   "[%08x %08x %08x %08x]",
                   i / 4, f[i], f[i + 1], f[i + 2], f[i + 3], u[i],
                   u[i + 1], u[i + 2], u[i + 3]);
      }
      device->dbg_gs_vb = NULL;
   }

   if (device->dbg_tess_draw) {
      const uint32_t *d = (const uint32_t *)device->dbg_tess_draw;
      mesa_logi("tess out_draws: count=%u inst=%u first=%u bias=%u "
                "start_inst=%u",
                d[0], d[1], d[2], d[3], d[4]);

      const uint32_t *c = (const uint32_t *)device->dbg_tess_counts;
      for (unsigned i = 0; i < MIN2(device->dbg_tess_counts_n, 8); i++)
         mesa_logi("  counts[%u] = %u", i, c[i]);

      const float *t = (const float *)device->dbg_tcs_buf;
      const uint32_t *ti = (const uint32_t *)device->dbg_tcs_buf;
      for (unsigned i = 0; i < MIN2(device->dbg_tcs_buf_n, 140); i += 4) {
         mesa_logi("  tcs[%2u] %8.3f %8.3f %8.3f %8.3f  "
                   "(0x%08x %08x %08x %08x)",
                   i, t[i], t[i + 1], t[i + 2], t[i + 3], ti[i],
                   ti[i + 1], ti[i + 2], ti[i + 3]);
      }

      if (device->dbg_mesh_grid) {
         const uint32_t *g = (const uint32_t *)device->dbg_mesh_grid;
         const uint32_t *disp = (const uint32_t *)device->dbg_mesh_disp;
         mesa_logi("  AS DispatchMesh grid: %u %u %u", disp[0], disp[1],
                   disp[2]);
         mesa_logi("  mesh indirect grid: threads %u %u %u local %u %u %u",
                   g[0], g[1], g[2], g[3], g[4], g[5]);
         device->dbg_mesh_grid = NULL;
      }

      if (device->dbg_mesh_verts) {
         const float *v = (const float *)device->dbg_mesh_verts;
         unsigned stride_f = (32 * 16) / 4;
         for (unsigned i = 0; i < 4; i++) {
            const float *p = v + i * stride_f;
            mesa_logi("  mesh vtx %u: %8.4f %8.4f %8.4f %8.4f", i, p[0],
                      p[1], p[2], p[3]);
         }
         const uint32_t *ix = (const uint32_t *)device->dbg_mesh_idx;
         mesa_logi("  mesh indices: %u %u %u / %u %u %u", ix[0], ix[1],
                   ix[2], ix[3], ix[4], ix[5]);
         device->dbg_mesh_verts = NULL;
      }
      device->dbg_tess_draw = NULL;
   }
}

static int
d12_queue_retire_worker(void *data)
{
   struct d12_queue *queue = data;
   long idle_backoff_ns = 0;

   for (;;) {
      d12_sync_lock(&queue->lock);
      while (!queue->retire_head && !p_atomic_read(&queue->retire_stop))
         d12_sync_wait_idle(&queue->retire_available, &queue->lock,
                            &idle_backoff_ns);
      idle_backoff_ns = 0;

      if (!queue->retire_head && p_atomic_read(&queue->retire_stop)) {
         mtx_unlock(&queue->lock);
         break;
      }

      struct d12_queue_op *op = queue->retire_head;
      queue->retire_head = op->next;
      if (!queue->retire_head)
         queue->retire_tail = NULL;
      op->next = NULL;
      mtx_unlock(&queue->lock);

      const bool completed = d12_queue_wait_timeline(queue,
                                                      op->retire_value);
      if (completed) {
         d12_sync_lock(&queue->lock);
         const bool retired =
            d12_queue_timeline_retire(&queue->timeline, op->retire_value);
         cnd_broadcast(&queue->retire_available);
         mtx_unlock(&queue->lock);
         if (!retired) {
            mesa_loge("command queue retirement accounting failure");
            d12_device_mark_removed(queue->device,
                                    DXGI_ERROR_DEVICE_REMOVED);
         } else if (op->type == D12_QUEUE_OP_SIGNAL) {
            if (getenv("D3D12AGX_DUMP_SYNC")) {
               mesa_logi("SYNC Signal retire: queue %p fence %p value %llu "
                         "point %llu", (void *)queue, (void *)op->fence,
                         (unsigned long long)op->value,
                         (unsigned long long)op->retire_value);
            }
            d12_fence_signal_value(op->fence, op->value);
         } else if (op->type == D12_QUEUE_OP_SUBMIT) {
            d12_queue_dump_completion(queue->device);
         }
      }

      if (op->type == D12_QUEUE_OP_PLACEMENT_SIGNAL ||
          op->type == D12_QUEUE_OP_PLACEMENT_BARRIER)
         d12_queue_release_signal_slot(queue, op);
      d12_queue_op_destroy(op);
   }

   return 0;
}

static int
d12_queue_worker(void *data)
{
   struct d12_queue *queue = data;
   long idle_backoff_ns = 0;

   for (;;) {
      d12_sync_lock(&queue->lock);
      while (!queue->op_head && !p_atomic_read(&queue->stop))
         d12_sync_wait_idle(&queue->work_available, &queue->lock,
                            &idle_backoff_ns);
      idle_backoff_ns = 0;

      if (p_atomic_read(&queue->stop)) {
         mtx_unlock(&queue->lock);
         break;
      }

      struct d12_queue_op *op = queue->op_head;
      queue->op_head = op->next;
      if (!queue->op_head)
         queue->op_tail = NULL;
      op->next = NULL;
      mtx_unlock(&queue->lock);

      bool retire = false;

      switch (op->type) {
      case D12_QUEUE_OP_WAIT:
         if (getenv("D3D12AGX_DUMP_SYNC"))
            mesa_logi("SYNC Wait worker: queue %p fence %p value %llu",
                      (void *)queue, (void *)op->fence,
                      (unsigned long long)op->value);
         d12_fence_ticket_wait(&op->ticket, &queue->stop);
         break;
      case D12_QUEUE_OP_SUBMIT:
         retire = d12_queue_wait_submit_capacity(queue) &&
                  d12_queue_process_submit(queue, op);
         break;
      case D12_QUEUE_OP_PLACEMENT_SIGNAL:
         retire = d12_queue_wait_submit_capacity(queue) &&
                  d12_queue_prepare_placement_signal(queue, op) &&
                  d12_queue_process_submit(queue, op);
         if (!retire)
            d12_queue_release_signal_slot(queue, op);
         break;
      case D12_QUEUE_OP_PLACEMENT_BARRIER:
         retire = d12_queue_wait_submit_capacity(queue) &&
                  d12_queue_prepare_placement_barrier(queue, op) &&
                  d12_queue_process_submit(queue, op);
         if (!retire)
            d12_queue_release_signal_slot(queue, op);
         break;
      case D12_QUEUE_OP_SIGNAL:
         if (getenv("D3D12AGX_DUMP_SYNC"))
            mesa_logi("SYNC Signal schedule: queue %p fence %p value %llu",
                      (void *)queue, (void *)op->fence,
                      (unsigned long long)op->value);
         op->retire_value = d12_queue_timeline_marker(&queue->timeline);
         retire = true;
         break;
      case D12_QUEUE_OP_PLACEMENT_WAIT:
         if (getenv("D3D12AGX_DUMP_SYNC"))
            mesa_logi("SYNC DDI Wait worker: queue %p fence %p value %llu",
                      (void *)queue, (void *)op->placement_fence,
                      (unsigned long long)op->value);
         (void)d12_placement_fence_wait(op->placement_fence, op->value,
                                        &queue->stop);
         break;
      case D12_QUEUE_OP_SPARSE_BIND: {
         if (queue->ddi_immediate_submit)
            d12_sync_lock(&queue->ddi_submit_lock);
         uint64_t point = d12_queue_timeline_marker(&queue->timeline);
         int result = -EIO;
         if (point != UINT64_MAX && d12_queue_wait_timeline(queue, point) &&
             d12_sparse_commit_map_updates(
                op->sparse_resource, op->sparse_map_updates,
                op->sparse_map_update_count)) {
            result = queue->device->dev.ops.bo_bind(
               &queue->device->dev, op->bind_ops, op->bind_count);
         }
         if (result) {
            mesa_loge("sparse VM_BIND batch failed: %d", result);
            d12_device_mark_removed(queue->device,
                                    DXGI_ERROR_DEVICE_REMOVED);
         }
         if (queue->ddi_immediate_submit)
            mtx_unlock(&queue->ddi_submit_lock);

         d12_sync_lock(&op->completion->lock);
         op->completion->result = result;
         op->completion->done = true;
         cnd_signal(&op->completion->event);
         mtx_unlock(&op->completion->lock);
         break;
      }
      }

      if (retire)
         d12_queue_retire_enqueue(queue, op);
      else
         d12_queue_op_destroy(op);
   }

   return 0;
}

static void
d12_queue_stop_worker(struct d12_queue *queue)
{
   if (!queue->worker_started && !queue->retire_worker_started)
      return;

   if (queue->worker_started) {
      p_atomic_set(&queue->stop, 1);
      d12_sync_lock(&queue->lock);
      cnd_broadcast(&queue->work_available);
      cnd_broadcast(&queue->retire_available);
      mtx_unlock(&queue->lock);
      thrd_join(queue->worker, NULL);
      queue->worker_started = false;
   }

   /* No producer remains.  Drain every successfully submitted operation
    * before the sync object, AGX queue, or retained BOs can be destroyed. */
   if (queue->retire_worker_started) {
      p_atomic_set(&queue->retire_stop, 1);
      d12_sync_lock(&queue->lock);
      cnd_broadcast(&queue->retire_available);
      mtx_unlock(&queue->lock);
      thrd_join(queue->retire_worker, NULL);
      queue->retire_worker_started = false;
   }

   struct d12_queue_op *op = queue->op_head;
   while (op) {
      struct d12_queue_op *next = op->next;
      d12_queue_op_destroy(op);
      op = next;
   }
   queue->op_head = queue->op_tail = NULL;

   op = queue->retire_head;
   while (op) {
      struct d12_queue_op *next = op->next;
      d12_queue_op_destroy(op);
      op = next;
   }
   queue->retire_head = queue->retire_tail = NULL;

   d12_sync_lock(&queue->ddi_submit_lock);
   for (uint32_t index = 0;
        index < queue->pending_placement_wait_count; ++index) {
      if (queue->pending_placement_waits[index].fence)
         d12_placement_fence_unref(
            queue->pending_placement_waits[index].fence);
   }
   memset(queue->pending_placement_waits, 0,
          sizeof(queue->pending_placement_waits));
   queue->pending_placement_wait_count = 0;
   mtx_unlock(&queue->ddi_submit_lock);
}

/* queue->ddi_submit_lock is held by the caller.  Successful submissions are
 * transferred to the retirement worker; failure consumes the operation and
 * poisons the device so no later queue callback can overtake it. */
static bool
d12_queue_submit_ddi_op_locked(struct d12_queue *queue,
                               struct d12_queue_op *op,
                               bool take_pending_waits)
{
   if (!queue || !op)
      return false;

   if (take_pending_waits &&
       !d12_queue_take_pending_placement_waits(queue, op))
      goto fail;

   if (!d12_queue_wait_submit_capacity(queue) ||
       !d12_queue_process_submit(queue, op))
      goto fail;

   d12_queue_retire_enqueue(queue, op);
   return true;

fail:
   d12_queue_release_signal_slot(queue, op);
   d12_queue_op_destroy(op);
   if (!d12_device_is_removed(queue->device))
      d12_device_mark_removed(queue->device, DXGI_ERROR_DEVICE_REMOVED);
   return false;
}

/* Materialize a group of raw D3D12 fence waits into the WDDM context before
 * accepting more waits or a queue-side page-table update.  The tiny AGX store
 * exists only to give SubmitCommandCb a normal DMA packet; KMD v4 holds it in
 * the per-context FIFO until every placement is ready. */
static bool
d12_queue_flush_pending_placement_waits_locked(struct d12_queue *queue)
{
   if (!queue->pending_placement_wait_count)
      return true;

   struct d12_queue_op *op = calloc(1, sizeof(*op));
   if (!op) {
      d12_device_mark_removed(queue->device, E_OUTOFMEMORY);
      return false;
   }
   op->type = D12_QUEUE_OP_PLACEMENT_BARRIER;
   if (!d12_queue_take_pending_placement_waits(queue, op) ||
       !d12_queue_prepare_placement_barrier(queue, op)) {
      d12_queue_release_signal_slot(queue, op);
      d12_queue_op_destroy(op);
      d12_device_mark_removed(queue->device, DXGI_ERROR_DEVICE_REMOVED);
      return false;
   }
   return d12_queue_submit_ddi_op_locked(queue, op, false);
}

static void STDMETHODCALLTYPE
d12_queue_ExecuteCommandLists(ID3D12CommandQueue *iface, UINT count,
                              ID3D12CommandList *const *lists)
{
   struct d12_queue *queue = d12_queue_from_iface(iface);
   struct d12_device *device = queue->device;

   if (queue->ddi_immediate_submit)
      d12_sync_lock(&queue->ddi_submit_lock);
   d12_sync_lock(&queue->lock);

   for (UINT i = 0; i < count; i++) {
      /* Our lists are always ID3D12GraphicsCommandList objects. */
      struct d12_cmd_list *list = (struct d12_cmd_list *)lists[i];

      if (list && list->had_error) {
         /* Closed, but its stream is incomplete: recording failed part
          * way. Submitting it is not a wrong picture, it is undefined
          * work on the GPU -- which is how this bench wedged and cost a
          * reboot. */
         mesa_loge("ExecuteCommandLists: list %u errored while recording; "
                   "dropped", i);
         continue;
      }
      if (!list || list->state != D12_LIST_CLOSED) {
         /*
          * Submitting a list still being recorded is an APPLICATION error,
          * and D3D12 answers it by removing the device with
          * DXGI_ERROR_INVALID_CALL -- not by ignoring the list. Dropping it
          * silently let the application carry on believing the work was
          * queued.
          */
         mesa_loge("ExecuteCommandLists: list %u not closed; device removed",
                   i);
         d12_device_mark_removed(queue->device, DXGI_ERROR_INVALID_CALL);
         continue;
      }

      if (list->segment_count == 0)
         continue;

      if (!d12_cmd_list_refresh_descriptors(list)) {
         mesa_loge("ExecuteCommandLists: descriptor refresh for list %u "
                   "references invalid command-list storage; dropped",
                   i);
         continue;
      }

      if (!d12_cmd_list_refresh_sparse_registry(list)) {
         mesa_loge("ExecuteCommandLists: sparse registry refresh for list %u "
                   "failed; dropped", i);
         continue;
      }

      if (getenv("D3D12AGX_DUMP_SEGS")) {
         size_t control_used =
            list->stream_cur - (uint8_t *)agx_bo_map(list->stream_bo);
         mesa_logi("list storage: control=%zu data=%zu/%zu usc=%zu/%zu "
                   "segments=%u",
                   control_used, list->stream_bo->size - list->data_floor,
                   list->stream_bo->size, list->usc_offset,
                   list->usc_bo->size, list->segment_count);
      }

      /* Pack the list as one logical payload. The worker slices it to the
       * kernel command-count limit while preserving these global barriers. */
      size_t max_payload =
         list->segment_count *
         (3 * sizeof(struct drm_asahi_cmd_header) +
          2 * D12_SUBMIT_MAX_ATTACHMENTS *
             sizeof(struct drm_asahi_attachment) +
          MAX2(sizeof(struct drm_asahi_cmd_compute),
               sizeof(struct drm_asahi_cmd_render)));
      uint8_t *payload = malloc(max_payload);
      if (!payload) {
         mesa_loge("ExecuteCommandLists: out of memory");
         if (queue->ddi_immediate_submit)
            d12_device_mark_removed(device, E_OUTOFMEMORY);
         break;
      }

      size_t payload_size = 0;
      unsigned nr_vdm = 0, nr_cdm = 0;
      bool write_set_complete = true;
      bool payload_valid = true;

      for (unsigned s = 0; s < list->segment_count; s++) {
         const struct d12_segment *seg = &list->segments[s];
         bool compute = seg->type == D12_SEG_CDM;

         if (getenv("D3D12AGX_DUMP_SEGS"))
            mesa_logi("seg %u: %s start=%zu end=%zu dispatches=%u", s,
                      compute ? "CDM" : "RENDER", seg->stream_start,
                      seg->stream_end, seg->dispatches);

         if (compute && seg->dispatches == 0)
            continue;

         if (compute) {
            if (!d12_queue_emit_write_set(
               device, list, seg, D12_WRITE_COMPUTE, payload,
               &payload_size, &write_set_complete)) {
               payload_valid = false;
               break;
            }
         } else {
            if (!d12_queue_emit_write_set(
               device, list, seg, D12_WRITE_VERTEX, payload,
               &payload_size, &write_set_complete) ||
                !d12_queue_emit_write_set(
                  device, list, seg, D12_WRITE_FRAGMENT, payload,
                  &payload_size, &write_set_complete)) {
               payload_valid = false;
               break;
            }
         }

         struct drm_asahi_cmd_header header =
            agx_cmd_header(compute, nr_vdm, nr_cdm);
         if (!compute)
            header.size = d12_render_cmd_size(device->dev.chip,
                                              device->dev.render_rsrc_spec_hi,
                                              device->dev.params.features);
         memcpy(payload + payload_size, &header, sizeof(header));
         payload_size += sizeof(header);

         if (compute) {
            if (seg->g17p_txf_metadata_only &&
                !d12_g17p_txf_metadata_enabled(&device->dev)) {
               mesa_loge("native Texture.Load submission remains registered-shim-only");
               payload_valid = false;
               break;
            }
            if (seg->g17p_txf_physical &&
                !d12_g17p_txf_physical_enabled(&device->dev)) {
               mesa_loge("native Texture.Load physical admission revoked");
               payload_valid = false;
               break;
            }
            struct drm_asahi_cmd_compute cmd = {
               .cdm_ctrl_stream_base =
                  list->stream_bo->va->addr + seg->stream_start,
               .cdm_ctrl_stream_end =
                  list->stream_bo->va->addr + seg->stream_end,
               .sampler_heap = seg->sampler_heap,
               .sampler_count = seg->sampler_count,
            };
            if (seg->scratch_cs_bo) {
               cmd.helper.data = seg->scratch_cs_bo->va->addr;
               cmd.helper.cfg =
                  seg->scratch_cs_preamble ? (1 << 16) : 0;
               cmd.helper.binary = agx_helper_program(&device->bg_eot);
            }
            memcpy(payload + payload_size, &cmd, sizeof(cmd));
            payload_size += sizeof(cmd);
            nr_cdm++;
         } else {
            struct drm_asahi_cmd_render cmd;
            if (!d12_fill_render_cmd(device, list, seg, &cmd)) {
               payload_valid = false;
               break;
            }
            memcpy(payload + payload_size, &cmd, header.size);
            payload_size += header.size;
            nr_vdm++;
         }
      }

      if (!payload_valid) {
         mesa_loge("ExecuteCommandLists: invalid GPU submission metadata or write tracker; device "
                   "removed before submission");
         free(payload);
         d12_device_mark_removed(device, DXGI_ERROR_DEVICE_REMOVED);
         break;
      }

      if (!payload_size) {
         free(payload);
         continue;
      }

      struct d12_queue_op *op = calloc(1, sizeof(*op));
      if (!op) {
         mesa_loge("ExecuteCommandLists: out of memory queuing list %u", i);
         free(payload);
         if (queue->ddi_immediate_submit)
            d12_device_mark_removed(device, E_OUTOFMEMORY);
         break;
      }

      op->type = D12_QUEUE_OP_SUBMIT;
      op->list = list;
      op->payload = payload;
      op->payload_size = payload_size;
      op->nr_vdm = nr_vdm;
      op->nr_cdm = nr_cdm;
      op->write_set_complete = write_set_complete;
      if (!write_set_complete && getenv("D3D12AGX_DUMP_WRITE_SET")) {
         mesa_logw("ExecuteCommandLists: submitting conservative attachment "
                   "hints without the Aurora complete-write declaration");
      }

      d12_queue_op_capture_bos(op, list);
      d12_queue_op_build_read_set(op, list);
      if (op->capture_failed) {
         /* AddRef below has not occurred; partial snapshots own only their
          * captured BO/alias references, not a command-list reference. */
         op->list = NULL;
         d12_queue_op_destroy(op);
         d12_device_mark_removed(device, E_OUTOFMEMORY);
         break;
      }

      /* Reset is legal as soon as ExecuteCommandLists returns.  Mark this
       * recording submitted before handing it back to the application, and
       * keep the list alive until the worker observes GPU completion. */
      ID3D12GraphicsCommandList_AddRef(&list->iface);
      list->submitted = true;
      if (queue->ddi_immediate_submit) {
         if (!d12_queue_take_pending_placement_waits(queue, op)) {
            d12_queue_op_destroy(op);
            d12_device_mark_removed(device, DXGI_ERROR_INVALID_CALL);
            break;
         }
         mtx_unlock(&queue->lock);
         const bool retire = d12_queue_wait_submit_capacity(queue) &&
                             d12_queue_process_submit(queue, op);
         if (retire) {
            d12_queue_retire_enqueue(queue, op);
         } else {
            d12_queue_op_destroy(op);
            if (!d12_device_is_removed(device))
               d12_device_mark_removed(device, DXGI_ERROR_DEVICE_REMOVED);
         }
         d12_sync_lock(&queue->lock);
         if (!retire)
            break;
      } else if (!d12_queue_enqueue_locked(queue, op)) {
         d12_queue_op_destroy(op);
         break;
      }
   }

   mtx_unlock(&queue->lock);
   if (queue->ddi_immediate_submit)
      mtx_unlock(&queue->ddi_submit_lock);
}

static void STDMETHODCALLTYPE
d12_queue_SetMarker(ID3D12CommandQueue *iface, UINT metadata,
                    const void *data, UINT size)
{
   /* PIX marker: harmless no-op. */
}

static void STDMETHODCALLTYPE
d12_queue_BeginEvent(ID3D12CommandQueue *iface, UINT metadata,
                     const void *data, UINT size)
{
   /* PIX event: harmless no-op. */
}

static void STDMETHODCALLTYPE
d12_queue_EndEvent(ID3D12CommandQueue *iface)
{
   /* PIX event: harmless no-op. */
}

static bool
d12_queue_bind_sparse_ordered(struct d12_queue *queue,
                              struct d12_resource *resource,
                              struct drm_asahi_gem_bind_op *bind_ops,
                              uint32_t bind_count,
                              struct d12_sparse_map_update *map_updates,
                              uint32_t map_update_count)
{
   /* The caller holds the lock which defines this queue's API order while
    * the resource helper mutates sparse_mappings and while this operation is
    * submitted/enqueued. DDI queues use ddi_submit_lock. Worker queues use
    * queue->lock, which this function releases immediately after enqueue so
    * it can wait for the synchronous API completion without blocking the
    * worker. */
   if (!bind_count) {
      bool valid = map_update_count == 0;
      free(bind_ops);
      free(map_updates);
      if (!queue->ddi_immediate_submit)
         mtx_unlock(&queue->lock);
      return valid;
   }
   if (!resource || (map_update_count && !map_updates)) {
      free(bind_ops);
      free(map_updates);
      if (!queue->ddi_immediate_submit)
         mtx_unlock(&queue->lock);
      return false;
   }

   if (queue->ddi_immediate_submit) {
      struct d12_queue_op *op = calloc(1, sizeof(*op));
      bool success = false;

      if (!op) {
         free(bind_ops);
         free(map_updates);
         d12_device_mark_removed(queue->device, E_OUTOFMEMORY);
         return false;
      }
      op->type = D12_QUEUE_OP_SPARSE_BIND;
      op->bind_ops = bind_ops;
      op->bind_count = bind_count;
      op->sparse_resource = resource;
      op->sparse_map_updates = map_updates;
      op->sparse_map_update_count = map_update_count;
      ID3D12Resource_AddRef(&resource->iface);

      if (!d12_queue_flush_pending_placement_waits_locked(queue) ||
          !d12_queue_wait_submit_capacity(queue) ||
          !queue->device->dev.ops.bo_bind_timeline)
         goto ddi_out;

      uint64_t in = d12_queue_timeline_marker(&queue->timeline);
      if (in == UINT64_MAX || !d12_queue_wait_timeline(queue, in) ||
          !d12_sparse_commit_map_updates(resource, map_updates,
                                         map_update_count) ||
          queue->device->dev.ops.bo_bind_timeline(
             &queue->device->dev, queue->agx_queue_id, queue->syncobj, in,
             bind_ops, bind_count))
         goto ddi_out;

      uint64_t committed_in;
      uint64_t committed_out;
      d12_sync_lock(&queue->lock);
      const bool committed = d12_queue_timeline_next_submit(
         &queue->timeline, &committed_in, &committed_out);
      mtx_unlock(&queue->lock);
      if (!committed || committed_in != in || committed_out != in + 1)
         goto ddi_out;

      free(op->bind_ops);
      op->bind_ops = NULL;
      free(op->sparse_map_updates);
      op->sparse_map_updates = NULL;
      op->sparse_map_update_count = 0;
      op->retire_value = committed_out;
      d12_queue_retire_enqueue(queue, op);
      op = NULL;
      success = true;

ddi_out:
      if (!success && !d12_device_is_removed(queue->device))
         d12_device_mark_removed(queue->device,
                                 DXGI_ERROR_DEVICE_REMOVED);
      d12_queue_op_destroy(op);
      return success;
   }

   struct d12_sparse_completion completion = {0};
   if (mtx_init(&completion.lock, mtx_plain) != thrd_success) {
      free(bind_ops);
      free(map_updates);
      mtx_unlock(&queue->lock);
      return false;
   }
   if (cnd_init(&completion.event) != thrd_success) {
      mtx_destroy(&completion.lock);
      free(bind_ops);
      free(map_updates);
      mtx_unlock(&queue->lock);
      return false;
   }

   struct d12_queue_op *op = calloc(1, sizeof(*op));
   if (!op) {
      cnd_destroy(&completion.event);
      mtx_destroy(&completion.lock);
      free(bind_ops);
      free(map_updates);
      mtx_unlock(&queue->lock);
      return false;
   }
   op->type = D12_QUEUE_OP_SPARSE_BIND;
   op->bind_ops = bind_ops;
   op->bind_count = bind_count;
   op->sparse_resource = resource;
   op->sparse_map_updates = map_updates;
   op->sparse_map_update_count = map_update_count;
   op->completion = &completion;
   ID3D12Resource_AddRef(&resource->iface);

   d12_sync_lock(&completion.lock);
   bool queued = d12_queue_enqueue_locked(queue, op);
   mtx_unlock(&queue->lock);
   if (!queued) {
      mtx_unlock(&completion.lock);
      d12_queue_op_destroy(op);
      cnd_destroy(&completion.event);
      mtx_destroy(&completion.lock);
      return false;
   }

   while (!completion.done)
      d12_sync_wait(&completion.event, &completion.lock);
   int result = completion.result;
   mtx_unlock(&completion.lock);
   cnd_destroy(&completion.event);
   mtx_destroy(&completion.lock);
   return result == 0;
}

HRESULT
d12_queue_update_tile_mappings(
   struct d12_queue *queue, struct d12_resource *res,
   UINT num_regions, const D3D12_TILED_RESOURCE_COORDINATE *starts,
   const D3D12_TILE_REGION_SIZE *sizes, struct d12_heap *heap,
   UINT num_ranges, const D3D12_TILE_RANGE_FLAGS *range_flags,
   const UINT *heap_starts, const UINT *range_counts,
   D3D12_TILE_MAPPING_FLAGS flags)
{
   if (!queue || !res || (flags & ~D3D12_TILE_MAPPING_FLAG_NO_HAZARD))
      return E_INVALIDARG;
   if (res->device != queue->device ||
       (heap && heap->device != queue->device))
      return E_INVALIDARG;

   struct drm_asahi_gem_bind_op *ops = NULL;
   uint32_t count = 0;
   struct d12_sparse_map_update *map_updates = NULL;
   uint32_t map_update_count = 0;

   /* Sparse software state and the matching queue operation are one ordered
    * transaction. Otherwise ExecuteCommandLists can snapshot the new BO set
    * before the VM_BIND which makes those BOs visible at the reserved VA. */
   if (queue->ddi_immediate_submit)
      d12_sync_lock(&queue->ddi_submit_lock);
   else
      d12_sync_lock(&queue->lock);

   HRESULT hr = d12_sparse_update_tile_mappings(
      res, num_regions, starts, sizes, heap, num_ranges, range_flags,
      heap_starts, range_counts, &ops, &count, &map_updates,
      &map_update_count);
   if (FAILED(hr)) {
      if (queue->ddi_immediate_submit)
         mtx_unlock(&queue->ddi_submit_lock);
      else
         mtx_unlock(&queue->lock);
      return hr;
   }

   const bool bound = d12_queue_bind_sparse_ordered(
      queue, res, ops, count, map_updates, map_update_count);
   if (queue->ddi_immediate_submit)
      mtx_unlock(&queue->ddi_submit_lock);
   if (!bound) {
      d12_device_mark_removed(queue->device, DXGI_ERROR_DEVICE_REMOVED);
      return d12_device_removed_reason(queue->device);
   }
   return S_OK;
}

static void STDMETHODCALLTYPE
d12_queue_UpdateTileMappings(
   ID3D12CommandQueue *iface, ID3D12Resource *resource_iface,
   UINT num_regions, const D3D12_TILED_RESOURCE_COORDINATE *starts,
   const D3D12_TILE_REGION_SIZE *sizes, ID3D12Heap *heap_iface,
   UINT num_ranges, const D3D12_TILE_RANGE_FLAGS *range_flags,
   const UINT *heap_starts, const UINT *range_counts,
   D3D12_TILE_MAPPING_FLAGS flags)
{
   HRESULT hr = d12_queue_update_tile_mappings(
      d12_queue_from_iface(iface),
      resource_iface ? d12_resource_from_iface(resource_iface) : NULL,
      num_regions, starts, sizes,
      heap_iface ? d12_memheap_from_iface(heap_iface) : NULL, num_ranges,
      range_flags, heap_starts, range_counts, flags);
   if (FAILED(hr))
      mesa_loge("UpdateTileMappings: invalid mapping description (%#x)",
                (unsigned)hr);
}

HRESULT
d12_queue_copy_tile_mappings(
   struct d12_queue *queue, struct d12_resource *dst,
   const D3D12_TILED_RESOURCE_COORDINATE *dst_start,
   struct d12_resource *src,
   const D3D12_TILED_RESOURCE_COORDINATE *src_start,
   const D3D12_TILE_REGION_SIZE *size, D3D12_TILE_MAPPING_FLAGS flags)
{
   if (!queue || !dst || !src ||
       (flags & ~D3D12_TILE_MAPPING_FLAG_NO_HAZARD))
      return E_INVALIDARG;
   if (dst->device != queue->device || src->device != queue->device)
      return E_INVALIDARG;

   struct drm_asahi_gem_bind_op *ops = NULL;
   uint32_t count = 0;
   struct d12_sparse_map_update *map_updates = NULL;
   uint32_t map_update_count = 0;

   if (queue->ddi_immediate_submit)
      d12_sync_lock(&queue->ddi_submit_lock);
   else
      d12_sync_lock(&queue->lock);

   HRESULT hr = d12_sparse_copy_tile_mappings(
      dst, dst_start, src, src_start, size, &ops, &count, &map_updates,
      &map_update_count);
   if (FAILED(hr)) {
      if (queue->ddi_immediate_submit)
         mtx_unlock(&queue->ddi_submit_lock);
      else
         mtx_unlock(&queue->lock);
      return hr;
   }

   const bool bound = d12_queue_bind_sparse_ordered(
      queue, dst, ops, count, map_updates, map_update_count);
   if (queue->ddi_immediate_submit)
      mtx_unlock(&queue->ddi_submit_lock);
   if (!bound) {
      d12_device_mark_removed(queue->device, DXGI_ERROR_DEVICE_REMOVED);
      return d12_device_removed_reason(queue->device);
   }
   return S_OK;
}

static void STDMETHODCALLTYPE
d12_queue_CopyTileMappings(
   ID3D12CommandQueue *iface, ID3D12Resource *dst_iface,
   const D3D12_TILED_RESOURCE_COORDINATE *dst_start,
   ID3D12Resource *src_iface,
   const D3D12_TILED_RESOURCE_COORDINATE *src_start,
   const D3D12_TILE_REGION_SIZE *size, D3D12_TILE_MAPPING_FLAGS flags)
{
   HRESULT hr = d12_queue_copy_tile_mappings(
      d12_queue_from_iface(iface),
      dst_iface ? d12_resource_from_iface(dst_iface) : NULL, dst_start,
      src_iface ? d12_resource_from_iface(src_iface) : NULL, src_start,
      size, flags);
   if (FAILED(hr))
      mesa_loge("CopyTileMappings: invalid mapping description (%#x)",
                (unsigned)hr);
}

static HRESULT STDMETHODCALLTYPE
d12_queue_Signal(ID3D12CommandQueue *iface, ID3D12Fence *fence_iface,
                 UINT64 value)
{
   struct d12_queue *queue = d12_queue_from_iface(iface);
   if (!fence_iface)
      return E_INVALIDARG;
   struct d12_fence *fence = d12_fence_from_iface(fence_iface);

   if (d12_device_is_removed(queue->device))
      return DXGI_ERROR_DEVICE_REMOVED;
   if (fence->device != queue->device)
      return E_INVALIDARG;

   struct d12_queue_op *op = calloc(1, sizeof(*op));
   if (!op)
      return E_OUTOFMEMORY;
   op->type = D12_QUEUE_OP_SIGNAL;
   op->fence = fence;
   op->value = value;
   d12_fence_internal_ref(fence);

   if (getenv("D3D12AGX_DUMP_SYNC")) {
      mesa_logi("SYNC Signal enqueue: queue %p agxq %u fence %p value %llu",
                (void *)queue, queue->agx_queue_id, (void *)fence,
                (unsigned long long)value);
   }

   d12_sync_lock(&queue->lock);
   bool queued = d12_queue_enqueue_locked(queue, op);
   mtx_unlock(&queue->lock);
   if (!queued) {
      d12_queue_op_destroy(op);
      return DXGI_ERROR_DEVICE_REMOVED;
   }
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_queue_Wait(ID3D12CommandQueue *iface, ID3D12Fence *fence_iface,
               UINT64 value)
{
   struct d12_queue *queue = d12_queue_from_iface(iface);
   if (!fence_iface)
      return E_INVALIDARG;
   struct d12_fence *fence = d12_fence_from_iface(fence_iface);
   if (fence->device != queue->device)
      return E_INVALIDARG;

   struct d12_queue_op *op = calloc(1, sizeof(*op));
   if (!op)
      return E_OUTOFMEMORY;
   op->type = D12_QUEUE_OP_WAIT;
   op->fence = fence;
   op->value = value;
   d12_fence_internal_ref(fence);
   d12_fence_ticket_init(fence, &op->ticket, value);

   if (getenv("D3D12AGX_DUMP_SYNC")) {
      mesa_logi("SYNC Wait enqueue: queue %p agxq %u fence %p value %llu",
                (void *)queue, queue->agx_queue_id, (void *)fence,
                (unsigned long long)value);
   }

   d12_sync_lock(&queue->lock);
   bool queued = d12_queue_enqueue_locked(queue, op);
   mtx_unlock(&queue->lock);
   if (!queued) {
      d12_queue_op_destroy(op);
      return DXGI_ERROR_DEVICE_REMOVED;
   }
   return S_OK;
}

HRESULT
d12_queue_signal_placement_fence(struct d12_queue *queue,
                                 struct d12_placement_fence *fence,
                                 uint64_t value)
{
   struct d12_queue_op *op;

   if (!queue || !fence || fence->device != queue->device)
      return E_INVALIDARG;
   if (d12_device_is_removed(queue->device))
      return DXGI_ERROR_DEVICE_REMOVED;
   if (fence->count != 1)
      return E_NOTIMPL;
   if (!queue->signal_stream_bo || !queue->signal_usc_bo ||
       !queue->signal_program_bo)
      return E_FAIL;

   op = calloc(1, sizeof(*op));
   if (!op)
      return E_OUTOFMEMORY;
   op->type = D12_QUEUE_OP_PLACEMENT_SIGNAL;
   op->placement_fence = fence;
   op->value = value;
   d12_placement_fence_ref(fence);

   if (queue->ddi_immediate_submit) {
      bool retire;

      d12_sync_lock(&queue->ddi_submit_lock);
      /* Keep the KMD v4 vector exact: a signal record must never be dropped
       * merely because the wait prefix filled the packet. */
      if (queue->pending_placement_wait_count ==
             ARRAY_SIZE(queue->pending_placement_waits) &&
          !d12_queue_flush_pending_placement_waits_locked(queue)) {
         mtx_unlock(&queue->ddi_submit_lock);
         d12_queue_op_destroy(op);
         return d12_device_removed_reason(queue->device);
      }
      if (!d12_queue_take_pending_placement_waits(queue, op)) {
         mtx_unlock(&queue->ddi_submit_lock);
         d12_queue_op_destroy(op);
         return E_INVALIDARG;
      }
      retire = d12_queue_wait_submit_capacity(queue) &&
               d12_queue_prepare_placement_signal(queue, op) &&
               d12_queue_process_submit(queue, op);
      if (retire) {
         d12_queue_retire_enqueue(queue, op);
      } else {
         d12_queue_release_signal_slot(queue, op);
         d12_queue_op_destroy(op);
         if (!d12_device_is_removed(queue->device))
            d12_device_mark_removed(queue->device,
                                    DXGI_ERROR_DEVICE_REMOVED);
      }
      mtx_unlock(&queue->ddi_submit_lock);
      return retire ? S_OK : d12_device_removed_reason(queue->device);
   }

   d12_sync_lock(&queue->lock);
   const bool queued = d12_queue_enqueue_locked(queue, op);
   mtx_unlock(&queue->lock);
   if (!queued) {
      d12_queue_op_destroy(op);
      return DXGI_ERROR_DEVICE_REMOVED;
   }
   return S_OK;
}

HRESULT
d12_queue_wait_placement_fence(struct d12_queue *queue,
                               struct d12_placement_fence *fence,
                               uint64_t value)
{
   struct d12_queue_op *op;

   if (!queue || !fence || fence->device != queue->device)
      return E_INVALIDARG;
   if (d12_device_is_removed(queue->device))
      return DXGI_ERROR_DEVICE_REMOVED;
   if (fence->count != 1)
      return E_NOTIMPL;

   if (queue->ddi_immediate_submit) {
      const uint64_t current = fence->entries[0].current_gpu_address;
      const uint64_t monitored = fence->entries[0].monitored_gpu_address;

      d12_sync_lock(&queue->ddi_submit_lock);
      for (uint32_t index = 0;
           index < queue->pending_placement_wait_count; ++index) {
         struct d12_queue_placement_wait *pending =
            &queue->pending_placement_waits[index];
         const struct d12_placement_fence_entry *entry =
            &pending->fence->entries[0];
         if (entry->current_gpu_address == current &&
             entry->monitored_gpu_address == monitored) {
            if (value > pending->value)
               pending->value = value;
            mtx_unlock(&queue->ddi_submit_lock);
            return S_OK;
         }
      }
      if (queue->pending_placement_wait_count >=
          ARRAY_SIZE(queue->pending_placement_waits)) {
         if (!d12_queue_flush_pending_placement_waits_locked(queue)) {
            mtx_unlock(&queue->ddi_submit_lock);
            return d12_device_removed_reason(queue->device);
         }
      }
      d12_placement_fence_ref(fence);
      queue->pending_placement_waits[
         queue->pending_placement_wait_count++] =
         (struct d12_queue_placement_wait){fence, value};
      mtx_unlock(&queue->ddi_submit_lock);
      return S_OK;
   }

   op = calloc(1, sizeof(*op));
   if (!op)
      return E_OUTOFMEMORY;
   op->type = D12_QUEUE_OP_PLACEMENT_WAIT;
   op->placement_fence = fence;
   op->value = value;
   d12_placement_fence_ref(fence);

   d12_sync_lock(&queue->lock);
   const bool queued = d12_queue_enqueue_locked(queue, op);
   mtx_unlock(&queue->lock);
   if (!queued) {
      d12_queue_op_destroy(op);
      return DXGI_ERROR_DEVICE_REMOVED;
   }
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_queue_GetTimestampFrequency(ID3D12CommandQueue *iface, UINT64 *frequency)
{
   struct d12_queue *queue = d12_queue_from_iface(iface);
   struct agx_device *dev = &queue->device->dev;

   if (!frequency)
      return E_POINTER;

   /* ns = ticks * num / den  =>  ticks/second = 1e9 * den / num */
   if (!dev->user_timestamp_to_ns.num) {
      mesa_loge("GetTimestampFrequency: no timestamp calibration from "
                "kernel");
      return E_FAIL;
   }

   *frequency = (UINT64)(1000000000ull * dev->user_timestamp_to_ns.den /
                         dev->user_timestamp_to_ns.num);
   return S_OK;
}

static HRESULT STDMETHODCALLTYPE
d12_queue_GetClockCalibration(ID3D12CommandQueue *iface,
                              UINT64 *gpu_timestamp, UINT64 *cpu_timestamp)
{
   struct d12_queue *queue = d12_queue_from_iface(iface);

   if (!gpu_timestamp || !cpu_timestamp)
      return E_POINTER;

   /* The GPU value is in the same calibrated domain used by EndQuery and
    * GetTimestampFrequency.  Linux has no Windows QPC ABI, so expose the
    * monotonic nanosecond clock as the paired CPU counter; callers use its
    * own frequency on this platform. */
   *cpu_timestamp = os_time_get_nano();
   *gpu_timestamp = agx_get_gpu_timestamp(&queue->device->dev);
   return S_OK;
}

static D12_AGGREGATE_RETURN(D3D12_COMMAND_QUEUE_DESC) STDMETHODCALLTYPE
d12_queue_GetDesc(ID3D12CommandQueue *iface
                  D12_AGGREGATE_OUT(D3D12_COMMAND_QUEUE_DESC))
{
   struct d12_queue *queue = d12_queue_from_iface(iface);
   D12_RETURN_AGGREGATE(queue->desc);
}

static const ID3D12CommandQueueVtbl d12_queue_vtbl = {
   /* every unimplemented slot fails safely */
   D12_ID3D12CommandQueue_DEFAULTS,
   .QueryInterface = d12_queue_QueryInterface,
   .AddRef = d12_queue_AddRef,
   .Release = d12_queue_Release,
   .GetPrivateData = d12_queue_GetPrivateData,
   .SetPrivateData = d12_queue_SetPrivateData,
   .SetPrivateDataInterface = d12_queue_SetPrivateDataInterface,
   .SetName = d12_queue_SetName,
   .GetDevice = d12_queue_GetDevice,
   .ExecuteCommandLists = d12_queue_ExecuteCommandLists,
   .SetMarker = d12_queue_SetMarker,
   .BeginEvent = d12_queue_BeginEvent,
   .EndEvent = d12_queue_EndEvent,
   .Signal = d12_queue_Signal,
   .Wait = d12_queue_Wait,
   .GetTimestampFrequency = d12_queue_GetTimestampFrequency,
   .GetClockCalibration = d12_queue_GetClockCalibration,
   .GetDesc = d12_queue_GetDesc,
   .UpdateTileMappings = d12_queue_UpdateTileMappings,
   .CopyTileMappings = d12_queue_CopyTileMappings,
};

static HRESULT
d12_queue_create_internal(struct d12_device *device,
                          const D3D12_COMMAND_QUEUE_DESC *desc, REFIID riid,
                          bool ddi_immediate_submit, void **out)
{
   if (!out)
      return E_POINTER;
   if (!desc)
      return E_INVALIDARG;

   switch (desc->Type) {
   case D3D12_COMMAND_LIST_TYPE_DIRECT:
   case D3D12_COMMAND_LIST_TYPE_COMPUTE:
   case D3D12_COMMAND_LIST_TYPE_COPY:
      /* All three run on the same asahi queue type for now; the render vs
       * compute split happens at command-buffer encoding time (G2+).
       */
      break;
   default:
      mesa_loge("d12_queue_create: queue type %d not supported",
                (int)desc->Type);
      return E_NOTIMPL;
   }

   if (desc->NodeMask > 1) {
      mesa_loge("d12_queue_create: multi-adapter NodeMask 0x%x unsupported",
                desc->NodeMask);
      return E_INVALIDARG;
   }

   struct d12_queue *queue = calloc(1, sizeof(*queue));
   if (!queue)
      return E_OUTOFMEMORY;

   struct agx_device *dev = &device->dev;

   queue->agx_queue_id =
      agx_create_command_queue(dev, DRM_ASAHI_PRIORITY_MEDIUM);
   if (queue->agx_queue_id == UINT32_MAX) {
      const int error = errno;
      free(queue);
      return (error == ENOMEM || error == ENOSPC || error == EBUSY) ?
             E_OUTOFMEMORY : E_FAIL;
   }

   if (device->sync->create(device->sync, 0, &queue->syncobj)) {
      mesa_loge("d12_queue_create: sync object creation failed");
      agx_destroy_command_queue(dev, queue->agx_queue_id);
      free(queue);
      return E_FAIL;
   }

   /* Materialize point 0 for the first real AGX submission. */
   uint64_t zero = 0;
   if (device->sync->timeline_signal(device->sync, &queue->syncobj, &zero,
                                     1)) {
      mesa_loge("d12_queue_create: initial sync signal failed");
      device->sync->destroy(device->sync, queue->syncobj);
      agx_destroy_command_queue(dev, queue->agx_queue_id);
      free(queue);
      return E_FAIL;
   }

   queue->iface.lpVtbl = (ID3D12CommandQueueVtbl *)&d12_queue_vtbl;
   queue->refcount = 1;
   queue->device = device;
   queue->desc = *desc;

   /*
    * NodeMask 0 means "the one adapter" and D3D12 REPORTS it as 1 -- GetDesc
    * returns the normalised description, not the request. Echoing the
    * caller's zero is the same mistake GetDesc on a resource made with
    * MipLevels and Alignment: a field the runtime is expected to resolve,
    * handed back unresolved.
    */
   if (!queue->desc.NodeMask)
      queue->desc.NodeMask = 1;

   d12_queue_timeline_init(&queue->timeline);
   queue->ddi_immediate_submit = ddi_immediate_submit;
   if (mtx_init(&queue->lock, mtx_plain) != thrd_success) {
      device->sync->destroy(device->sync, queue->syncobj);
      agx_destroy_command_queue(dev, queue->agx_queue_id);
      free(queue);
      return E_FAIL;
   }
   if (mtx_init(&queue->ddi_submit_lock, mtx_plain) != thrd_success) {
      mtx_destroy(&queue->lock);
      device->sync->destroy(device->sync, queue->syncobj);
      agx_destroy_command_queue(dev, queue->agx_queue_id);
      free(queue);
      return E_FAIL;
   }
   if (cnd_init(&queue->work_available) != thrd_success) {
      mtx_destroy(&queue->ddi_submit_lock);
      mtx_destroy(&queue->lock);
      device->sync->destroy(device->sync, queue->syncobj);
      agx_destroy_command_queue(dev, queue->agx_queue_id);
      free(queue);
      return E_FAIL;
   }
   if (cnd_init(&queue->retire_available) != thrd_success) {
      cnd_destroy(&queue->work_available);
      mtx_destroy(&queue->ddi_submit_lock);
      mtx_destroy(&queue->lock);
      device->sync->destroy(device->sync, queue->syncobj);
      agx_destroy_command_queue(dev, queue->agx_queue_id);
      free(queue);
      return E_FAIL;
   }
   d12_private_data_store_init(&queue->private_data,
                               (IUnknown *)&queue->iface);
   d12_device_add_ref(device);

   HRESULT signal_arena_result = d12_queue_signal_arena_init(queue);
   if (FAILED(signal_arena_result)) {
      d12_device_release(device);
      d12_private_data_store_destroy(&queue->private_data);
      cnd_destroy(&queue->retire_available);
      cnd_destroy(&queue->work_available);
      mtx_destroy(&queue->ddi_submit_lock);
      mtx_destroy(&queue->lock);
      device->sync->destroy(device->sync, queue->syncobj);
      agx_destroy_command_queue(dev, queue->agx_queue_id);
      free(queue);
      return signal_arena_result;
   }

   if (thrd_create(&queue->retire_worker, d12_queue_retire_worker, queue) !=
       thrd_success) {
      d12_queue_signal_arena_cleanup(queue);
      d12_device_release(device);
      d12_private_data_store_destroy(&queue->private_data);
      cnd_destroy(&queue->retire_available);
      cnd_destroy(&queue->work_available);
      mtx_destroy(&queue->ddi_submit_lock);
      mtx_destroy(&queue->lock);
      device->sync->destroy(device->sync, queue->syncobj);
      agx_destroy_command_queue(dev, queue->agx_queue_id);
      free(queue);
      return E_FAIL;
   }
   queue->retire_worker_started = true;

   if (thrd_create(&queue->worker, d12_queue_worker, queue) != thrd_success) {
      p_atomic_set(&queue->retire_stop, 1);
      d12_sync_lock(&queue->lock);
      cnd_signal(&queue->retire_available);
      mtx_unlock(&queue->lock);
      thrd_join(queue->retire_worker, NULL);
      queue->retire_worker_started = false;
      d12_queue_signal_arena_cleanup(queue);
      d12_device_release(device);
      d12_private_data_store_destroy(&queue->private_data);
      cnd_destroy(&queue->retire_available);
      cnd_destroy(&queue->work_available);
      mtx_destroy(&queue->ddi_submit_lock);
      mtx_destroy(&queue->lock);
      device->sync->destroy(device->sync, queue->syncobj);
      agx_destroy_command_queue(dev, queue->agx_queue_id);
      free(queue);
      return E_FAIL;
   }
   queue->worker_started = true;

   HRESULT hr = d12_queue_QueryInterface(&queue->iface, riid, out);
   d12_queue_Release(&queue->iface);
   return hr;
}

HRESULT
d12_queue_create(struct d12_device *device,
                 const D3D12_COMMAND_QUEUE_DESC *desc, REFIID riid,
                 void **out)
{
   return d12_queue_create_internal(device, desc, riid, false, out);
}

HRESULT
d12_queue_create_ddi(struct d12_device *device,
                     const D3D12_COMMAND_QUEUE_DESC *desc, REFIID riid,
                     void **out)
{
   return d12_queue_create_internal(device, desc, riid, true, out);
}
