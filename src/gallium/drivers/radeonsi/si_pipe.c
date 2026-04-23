/*
 * Copyright 2010 Jerome Glisse <glisse@freedesktop.org>
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "si_pipe.h"
#include "gfx/si_gfx.h"
#include "mm/si_mm.h"

#include "driver_ddebug/dd_util.h"
#include "si_public.h"
#include "sid.h"
#include "ac_shader_util.h"
#include "ac_shadowed_regs.h"
#include "compiler/nir/nir.h"
#include "util/disk_cache.h"
#include "util/hex.h"
#include "util/u_cpu_detect.h"
#include "util/u_memory.h"
#include "util/u_suballoc.h"
#include "util/u_tests.h"
#include "util/u_upload_mgr.h"
#include "util/xmlconfig.h"
#include "si_utrace.h"
#include "si_video.h"

#include "aco_interface.h"

#if AMD_LLVM_AVAILABLE
#include "ac_llvm_util.h"
#endif

#if HAVE_AMDGPU_VIRTIO
#include "virtio/virtio-gpu/drm_hw.h"
#endif

#include <xf86drm.h>

static struct pipe_context *si_create_context(struct pipe_screen *screen, unsigned flags);

static const struct debug_named_value radeonsi_debug_options[] = {
   /* Information logging options: */
   {"info", DBG(INFO), "Print driver information"},
   {"tex", DBG(TEX), "Print texture info"},
   {"compute", DBG(COMPUTE), "Print compute info"},
   {"vm", DBG(VM), "Print virtual addresses when creating resources"},
   {"cache_stats", DBG(CACHE_STATS), "Print shader cache statistics."},
   {"ib", DBG(IB), "Print command buffers."},
   {"elements", DBG(VERTEX_ELEMENTS), "Print vertex elements."},

   /* Driver options: */
   {"nowc", DBG(NO_WC), "Disable GTT write combining"},
   {"nowcstream", DBG(NO_WC_STREAM), "Disable GTT write combining for streaming uploads"},
   {"check_vm", DBG(CHECK_VM), "Check VM faults and dump debug info."},
   {"reserve_vmid", DBG(RESERVE_VMID), "Force VMID reservation per context."},
   {"shadowregs", DBG(SHADOW_REGS), "Enable CP register shadowing."},
   {"userqnoshadowregs", DBG(USERQ_NO_SHADOW_REGS), "Disable register shadowing in userqueue."},
   {"nofastdlist", DBG(NO_FAST_DISPLAY_LIST), "Disable fast display lists"},
   {"nodmashaders", DBG(NO_DMA_SHADERS), "Disable uploading shaders via CP DMA and map them directly."},

   /* 3D engine options: */
   {"nongg", DBG(NO_NGG), "Disable NGG and use the legacy pipeline."},
   {"nggc", DBG(ALWAYS_NGG_CULLING_ALL), "Always use NGG culling even when it can hurt."},
   {"nonggc", DBG(NO_NGG_CULLING), "Disable NGG culling."},
   {"switch_on_eop", DBG(SWITCH_ON_EOP), "Program WD/IA to switch on end-of-packet."},
   {"nooutoforder", DBG(NO_OUT_OF_ORDER), "Disable out-of-order rasterization"},
   {"nodpbb", DBG(NO_DPBB), "Disable DPBB. Overrules the dpbb enable option."},
   {"dpbb", DBG(DPBB), "Enable DPBB for gfx9 dGPU. Default enabled for gfx9 APU and >= gfx10."},
   {"nohyperz", DBG(NO_HYPERZ), "Disable Hyper-Z"},
   {"no2d", DBG(NO_2D_TILING), "Disable 2D tiling"},
   {"notiling", DBG(NO_TILING), "Disable tiling"},
   {"nodisplaytiling", DBG(NO_DISPLAY_TILING), "Disable display tiling"},
   {"nodisplaydcc", DBG(NO_DISPLAY_DCC), "Disable display DCC"},
   {"noexporteddcc", DBG(NO_EXPORTED_DCC), "Disable DCC for all exported buffers (via DMABUF, etc.)"},
   {"nodcc", DBG(NO_DCC), "Disable DCC."},
   {"nodccclear", DBG(NO_DCC_CLEAR), "Disable DCC fast clear."},
   {"nodccstore", DBG(NO_DCC_STORE), "Disable DCC stores"},
   {"dccstore", DBG(DCC_STORE), "Enable DCC stores"},
   {"nodccmsaa", DBG(NO_DCC_MSAA), "Disable DCC for MSAA"},
   {"nofmask", DBG(NO_FMASK), "Disable MSAA compression"},
   {"nodma", DBG(NO_DMA), "Disable SDMA-copy for DRI_PRIME"},

   {"forcegfxblit", DBG(FORCE_GFX_BLIT), "Force the use of fragment shaders for image clears, copies, blits, and resolve."},
   {"forcecomputeblit", DBG(FORCE_COMPUTE_BLIT), "Force the use of compute shaders for image clears, copies, blits, and resolve."},
   {"forcefastclear", DBG(FORCE_FAST_CLEAR), "Force the use of image \"fast clear\" when possible. For debug only."},

   {"extra_md", DBG(EXTRA_METADATA), "Set UMD metadata for all textures and with additional fields for umr"},

   {"tmz", DBG(TMZ), "Force allocation of scanout/depth/stencil buffer as encrypted"},
   {"sqtt", DBG(SQTT), "Enable SQTT"},
   {"export_modifier", DBG(EXPORT_MODIFIER), "Export real modifier instead of DRM_FORMAT_MOD_INVALID"},

   DEBUG_NAMED_VALUE_END /* must be last */
};

static const struct debug_named_value test_options[] = {
   /* Tests: */
   {"clearbuffer", DBG(TEST_CLEAR_BUFFER), "Test correctness of the clear_buffer compute shader"},
   {"copybuffer", DBG(TEST_COPY_BUFFER), "Test correctness of the copy_buffer compute shader"},
   {"imagecopy", DBG(TEST_IMAGE_COPY), "Invoke resource_copy_region tests with images and exit."},
   {"computeblit", DBG(TEST_COMPUTE_BLIT), "Invoke blits tests and exit."},
   {"testvmfaultcp", DBG(TEST_VMFAULT_CP), "Invoke a CP VM fault test and exit."},
   {"testvmfaultshader", DBG(TEST_VMFAULT_SHADER), "Invoke a shader VM fault test and exit."},
   {"dmaperf", DBG(TEST_DMA_PERF), "Test DMA performance"},
   {"testmemperf", DBG(TEST_MEM_PERF), "Test map + memcpy perf using the winsys."},

   DEBUG_NAMED_VALUE_END /* must be last */
};

void si_init_aux_async_compute_ctx(struct si_screen *sscreen)
{
   assert(!sscreen->async_compute_context);
   sscreen->async_compute_context =
      si_create_context(&sscreen->b,
                        SI_CONTEXT_FLAG_AUX |
                        PIPE_CONTEXT_LOSE_CONTEXT_ON_RESET |
                        (sscreen->options.aux_debug ? PIPE_CONTEXT_DEBUG : 0) |
                        PIPE_CONTEXT_COMPUTE_ONLY);

   /* Limit the numbers of waves allocated for this context. */
   if (sscreen->async_compute_context)
      ((struct si_context*)sscreen->async_compute_context)->cs_max_waves_per_sh = 2;
}

static void decref_implicit_resource(struct hash_entry *entry)
{
   pipe_resource_reference((struct pipe_resource**)&entry->data, NULL);
}

/*
 * pipe_context
 */
static void si_destroy_context(struct pipe_context *context)
{
   struct si_context *sctx = (struct si_context *)context;

   si_fini_gfx_context(sctx);
   si_fini_mm_context(sctx);

   if (sctx->ctx)
      sctx->ws->ctx_destroy(sctx->ctx);

   if (sctx->dirty_implicit_resources)
      _mesa_hash_table_destroy(sctx->dirty_implicit_resources,
                               decref_implicit_resource);

   if (sctx->b.stream_uploader)
      u_upload_destroy(sctx->b.stream_uploader);
   if (sctx->b.const_uploader && sctx->b.const_uploader != sctx->b.stream_uploader)
      u_upload_destroy(sctx->b.const_uploader);
   if (sctx->cached_gtt_allocator)
      u_upload_destroy(sctx->cached_gtt_allocator);

   slab_destroy_child(&sctx->pool_transfers);
   slab_destroy_child(&sctx->pool_transfers_unsync);

   u_suballocator_destroy(&sctx->allocator_zeroed_memory);

   _mesa_hash_table_destroy(sctx->tex_handles, NULL);
   _mesa_hash_table_destroy(sctx->img_handles, NULL);

   util_dynarray_fini(&sctx->resident_tex_handles);
   util_dynarray_fini(&sctx->resident_img_handles);
   util_dynarray_fini(&sctx->resident_tex_needs_color_decompress);
   util_dynarray_fini(&sctx->resident_img_needs_color_decompress);
   util_dynarray_fini(&sctx->resident_tex_needs_depth_decompress);

   if (!(sctx->context_flags & SI_CONTEXT_FLAG_AUX))
      p_atomic_dec(&context->screen->num_contexts);

   FREE(sctx);
}

static enum pipe_reset_status si_get_reset_status(struct pipe_context *ctx)
{
   struct si_context *sctx = (struct si_context *)ctx;
   if (sctx->context_flags & SI_CONTEXT_FLAG_AUX)
      return PIPE_NO_RESET;

   bool needs_reset, reset_completed;
   enum pipe_reset_status status = sctx->ws->ctx_query_reset_status(sctx->ctx, false,
                                                                    &needs_reset, &reset_completed);

   if (status != PIPE_NO_RESET) {
      if (sctx->has_reset_been_notified && reset_completed)
         return PIPE_NO_RESET;

      sctx->has_reset_been_notified = true;

      if (!(sctx->context_flags & SI_CONTEXT_FLAG_AUX)) {
         /* Call the gallium frontend to set a no-op API dispatch. */
         if (needs_reset && sctx->device_reset_callback.reset)
            sctx->device_reset_callback.reset(sctx->device_reset_callback.data, status);
      }
   }
   return status;
}

static void si_set_device_reset_callback(struct pipe_context *ctx,
                                         const struct pipe_device_reset_callback *cb)
{
   struct si_context *sctx = (struct si_context *)ctx;

   if (cb)
      sctx->device_reset_callback = *cb;
   else
      memset(&sctx->device_reset_callback, 0, sizeof(sctx->device_reset_callback));
}

/* Apitrace profiling:
 *   1) qapitrace : Tools -> Profile: Measure CPU & GPU times
 *   2) In the middle panel, zoom in (mouse wheel) on some bad draw call
 *      and remember its number.
 *   3) In Mesa, enable queries and performance counters around that draw
 *      call and print the results.
 *   4) glretrace --benchmark --markers ..
 */
static void si_emit_string_marker(struct pipe_context *ctx, const char *string, int len)
{
   struct si_context *sctx = (struct si_context *)ctx;

   dd_parse_apitrace_marker(string, len, &sctx->apitrace_call_number);

   if (sctx->sqtt_enabled)
      si_write_user_event(sctx, &sctx->gfx_cs, UserEventTrigger, string, len);

   if (sctx->log)
      u_log_printf(sctx->log, "\nString marker: %*s\n", len, string);
}

static void si_set_debug_callback(struct pipe_context *ctx, const struct util_debug_callback *cb)
{
   struct si_context *sctx = (struct si_context *)ctx;
   struct si_screen *screen = sctx->screen;

   util_queue_finish(&screen->shader_compiler_queue);
   util_queue_finish(&screen->shader_compiler_queue_opt_variants);

   if (cb)
      sctx->debug = *cb;
   else
      memset(&sctx->debug, 0, sizeof(sctx->debug));
}

static void si_set_log_context(struct pipe_context *ctx, struct u_log_context *log)
{
   struct si_context *sctx = (struct si_context *)ctx;
   sctx->log = log;

   if (log)
      u_log_add_auto_logger(log, si_auto_log_cs, sctx);
}

static void si_set_context_param(struct pipe_context *ctx, enum pipe_context_param param,
                                 unsigned value)
{
   struct radeon_winsys *ws = ((struct si_context *)ctx)->ws;

   switch (param) {
   case PIPE_CONTEXT_PARAM_UPDATE_THREAD_SCHEDULING:
      ws->pin_threads_to_L3_cache(ws, value);
      break;
   default:;
   }
}

static void si_set_frontend_noop(struct pipe_context *ctx, bool enable)
{
   struct si_context *sctx = (struct si_context *)ctx;

   ctx->flush(ctx, NULL, PIPE_FLUSH_ASYNC);
   sctx->is_noop = enable;
}

/* Function used by the pipe_loader to decide which driver to use when
 * the KMD is virtio_gpu.
 */
bool si_virtgpu_probe_nctx(int fd, const struct virgl_renderer_capset_drm *caps)
{
   #ifdef HAVE_AMDGPU_VIRTIO
   return caps->context_type == VIRTGPU_DRM_CONTEXT_AMDGPU;
   #else
   return false;
   #endif
}

struct pipe_context *si_create_context(struct pipe_screen *screen, unsigned flags)
{
   struct si_screen *sscreen = (struct si_screen *)screen;
   STATIC_ASSERT(DBG_COUNT <= 64);

   struct si_context *sctx = CALLOC_STRUCT(si_context);

   if (!sctx) {
      mesa_loge("can't allocate a context");
      return NULL;
   }

   sctx->b.screen = screen; /* this must be set first */
   sctx->b.priv = NULL;
   sctx->b.destroy = si_destroy_context;
   sctx->screen = sscreen; /* Easy accessing of screen/winsys. */
   sctx->is_debug = (flags & PIPE_CONTEXT_DEBUG) != 0;
   sctx->context_flags = flags;

   slab_create_child(&sctx->pool_transfers, &sscreen->pool_transfers);
   slab_create_child(&sctx->pool_transfers_unsync, &sscreen->pool_transfers);

   sctx->ws = sscreen->ws;
   sctx->family = sscreen->info.family;
   sctx->gfx_level = sscreen->info.gfx_level;
   sctx->vcn_ip_ver = sscreen->info.vcn_ip_version;

   /* Initialize the context handle and the command stream. */
   sctx->ctx = sctx->ws->ctx_create(sctx->ws, sctx->context_flags);
   if (!sctx->ctx) {
      mesa_loge("can't create radeon_winsys_ctx");
      goto fail;
   }

   /* Initialize private allocators. */
   u_suballocator_init(&sctx->allocator_zeroed_memory, &sctx->b, 128 * 1024, 0,
                       PIPE_USAGE_DEFAULT,
                       SI_RESOURCE_FLAG_CLEAR | SI_RESOURCE_FLAG_32BIT, false);

   sctx->cached_gtt_allocator = u_upload_create(&sctx->b, 16 * 1024, 0, PIPE_USAGE_STAGING, 0);
   if (!sctx->cached_gtt_allocator) {
      mesa_loge("can't create cached_gtt_allocator");
      goto fail;
   }

   /* Initialize public allocators. Unify uploaders as follows:
    * - dGPUs: The const uploader writes to VRAM and the stream uploader writes to RAM.
    * - APUs: There is only one uploader instance writing to RAM. VRAM has the same perf on APUs.
    */
   bool is_apu = !sscreen->info.has_dedicated_vram;
   sctx->b.stream_uploader =
      u_upload_create(&sctx->b, 1024 * 1024, 0,
                      sscreen->debug_flags & DBG(NO_WC_STREAM) ? PIPE_USAGE_STAGING
                                                               : PIPE_USAGE_STREAM,
                      SI_RESOURCE_FLAG_32BIT); /* same flags as const_uploader */
   if (!sctx->b.stream_uploader) {
      mesa_loge("can't create stream_uploader");
      goto fail;
   }

   if (is_apu) {
      sctx->b.const_uploader = sctx->b.stream_uploader;
   } else {
      sctx->b.const_uploader =
         u_upload_create(&sctx->b, 256 * 1024, 0, PIPE_USAGE_DEFAULT,
                         SI_RESOURCE_FLAG_32BIT);
      if (!sctx->b.const_uploader) {
         mesa_loge("can't create const_uploader");
         goto fail;
      }
   }

   sctx->b.set_debug_callback = si_set_debug_callback;
   sctx->b.set_log_context = si_set_log_context;
   sctx->b.set_context_param = si_set_context_param;
   sctx->b.get_device_reset_status = si_get_reset_status;
   sctx->b.set_device_reset_callback = si_set_device_reset_callback;
   sctx->b.set_frontend_noop = si_set_frontend_noop;

   list_inithead(&sctx->active_queries);
   si_init_buffer_functions(sctx);
   si_init_fence_functions(sctx);
   si_init_context_texture_functions(sctx);

   /* Bindless handles. */
   sctx->tex_handles = _mesa_hash_table_create(NULL, _mesa_hash_pointer, _mesa_key_pointer_equal);
   sctx->img_handles = _mesa_hash_table_create(NULL, _mesa_hash_pointer, _mesa_key_pointer_equal);

   sctx->resident_tex_handles = UTIL_DYNARRAY_INIT;
   sctx->resident_img_handles = UTIL_DYNARRAY_INIT;
   sctx->resident_tex_needs_color_decompress = UTIL_DYNARRAY_INIT;
   sctx->resident_img_needs_color_decompress = UTIL_DYNARRAY_INIT;
   sctx->resident_tex_needs_depth_decompress = UTIL_DYNARRAY_INIT;

   sctx->dirty_implicit_resources = _mesa_pointer_hash_table_create(NULL);
   if (!sctx->dirty_implicit_resources) {
      mesa_loge("can't create dirty_implicit_resources");
      goto fail;
   }

   if (!(flags & PIPE_CONTEXT_MEDIA_ONLY)) {
      if (!si_init_gfx_context(sscreen, sctx, flags))
         goto fail;
   }

   /* PIPE_CONTEXT_COMPUTE_ONLY doesn't mean no multimedia, it means no graphics so always
    * init mm but don't fail if it reports an error.
    */
   si_init_mm_context(sscreen, sctx, flags);

   if (!(flags & SI_CONTEXT_FLAG_AUX)) {
      p_atomic_inc(&screen->num_contexts);

      /* Check if the aux_context needs to be recreated */
      for (unsigned i = 0; i < ARRAY_SIZE(sscreen->aux_contexts); i++) {
         if (!sscreen->aux_contexts[i].ctx)
            continue;

         struct si_context *saux = si_get_aux_context(sscreen, &sscreen->aux_contexts[i]);
         enum pipe_reset_status status =
            sctx->ws->ctx_query_reset_status(saux->ctx, true, NULL, NULL);

         if (status != PIPE_NO_RESET) {
            /* We lost the aux_context, create a new one */
            unsigned context_flags = saux->context_flags;
            saux->b.destroy(&saux->b);

            saux = (struct si_context *)si_create_context(&sscreen->b, context_flags);
            if (sscreen->options.aux_debug)
               saux->b.set_log_context(&saux->b, &sscreen->aux_contexts[i].log);

            sscreen->aux_contexts[i].ctx = &saux->b;
         }
         si_put_aux_context_flush(&sscreen->aux_contexts[i]);
      }

      simple_mtx_lock(&sscreen->async_compute_context_lock);
      if (sscreen->async_compute_context) {
         struct si_context *compute_ctx = (struct si_context*)sscreen->async_compute_context;
         enum pipe_reset_status status =
            sctx->ws->ctx_query_reset_status(compute_ctx->ctx, true, NULL, NULL);

         if (status != PIPE_NO_RESET) {
            sscreen->async_compute_context->destroy(sscreen->async_compute_context);
            sscreen->async_compute_context = NULL;
         }
      }
      simple_mtx_unlock(&sscreen->async_compute_context_lock);

      si_reset_debug_log_buffer(sctx);
   }

   return &sctx->b;
fail:
   mesa_loge("Failed to create a context.");
   si_destroy_context(&sctx->b);
   return NULL;
}

void
si_get_scratch_tmpring_size(struct si_context *sctx, unsigned bytes_per_wave,
                            bool is_compute, unsigned *spi_tmpring_size)
{
   bytes_per_wave = ac_compute_scratch_wavesize(&sctx->screen->info, bytes_per_wave);

   if (is_compute) {
      sctx->max_seen_compute_scratch_bytes_per_wave =
         MAX2(sctx->max_seen_compute_scratch_bytes_per_wave, bytes_per_wave);
   } else {
      sctx->max_seen_scratch_bytes_per_wave =
         MAX2(sctx->max_seen_scratch_bytes_per_wave, bytes_per_wave);
   }

   /* TODO: We could decrease WAVES to make the whole buffer fit into the infinity cache. */
   ac_get_scratch_tmpring_size(&sctx->screen->info, sctx->screen->info.max_scratch_waves,
                               is_compute ? sctx->max_seen_compute_scratch_bytes_per_wave
                                          : sctx->max_seen_scratch_bytes_per_wave,
                               spi_tmpring_size);
}

static bool si_is_resource_busy(struct pipe_screen *screen, struct pipe_resource *resource,
                                unsigned usage)
{
   struct radeon_winsys *ws = ((struct si_screen *)screen)->ws;

   return !ws->buffer_wait(ws, si_resource(resource)->buf, 0,
                           /* If mapping for write, we need to wait for all reads and writes.
                            * If mapping for read, we only need to wait for writes.
                            */
                           (usage & PIPE_MAP_WRITE ? RADEON_USAGE_READWRITE : RADEON_USAGE_WRITE) |
                           RADEON_USAGE_DISALLOW_SLOW_REPLY);
}

static struct pipe_context *si_pipe_create_context(struct pipe_screen *screen, void *priv,
                                                   unsigned flags)
{
   struct si_screen *sscreen = (struct si_screen *)screen;
   struct pipe_context *ctx;
   struct si_context *sctx;

   if (sscreen->debug_flags & DBG(CHECK_VM))
      flags |= PIPE_CONTEXT_DEBUG;

   ctx = si_create_context(screen, flags);
   sctx = (struct si_context *)ctx;

   if (ctx && sscreen->info.gfx_level >= GFX9 && sscreen->debug_flags & DBG(SQTT)) {
      /* Auto-enable stable performance profile if possible. */
      if (screen->num_contexts == 1)
          sscreen->ws->cs_set_pstate(&sctx->gfx_cs, RADEON_CTX_PSTATE_PEAK);

      if (ac_check_profile_state(&sscreen->info)) {
         mesa_loge("Canceling RGP trace request as a hang condition has been "
                   "detected. Force the GPU into a profiling mode with e.g. "
                   "\"echo profile_peak  > "
                   "/sys/class/drm/card0/device/power_dpm_force_performance_level\"");
      } else {
         if (!si_init_sqtt(sctx)) {
            FREE(ctx);
            return NULL;
         }

         si_handle_sqtt(sctx, &sctx->gfx_cs);
      }
   }

   if (!(flags & PIPE_CONTEXT_PREFER_THREADED))
      return ctx;

   /* Clover (compute-only) is unsupported. */
   if (flags & PIPE_CONTEXT_COMPUTE_ONLY)
      return ctx;

   /* When shaders are logged to stderr, asynchronous compilation is
    * disabled too. */
   if (sscreen->shader_debug_flags & DBG_ALL_SHADERS)
      return ctx;

   /* Use asynchronous flushes only on amdgpu, since the radeon
    * implementation for fence_server_sync is incomplete. */
   struct pipe_context *tc =
      threaded_context_create(ctx, &sscreen->pool_transfers,
                              si_replace_buffer_storage,
                              &(struct threaded_context_options){
                                 .create_fence = sscreen->info.is_amdgpu ?
                                       si_create_fence : NULL,
                                 .is_resource_busy = si_is_resource_busy,
                                 .driver_calls_flush_notify = true,
                                 .unsynchronized_create_fence_fd = true,
                              },
                              &sctx->tc);

   if (tc && tc != ctx)
      threaded_context_init_bytes_mapped_limit((struct threaded_context *)tc, 4);

   return tc;
}

/*
 * pipe_screen
 */
void si_destroy_screen(struct pipe_screen *pscreen)
{
   struct si_screen *sscreen = (struct si_screen *)pscreen;

   if (!sscreen->ws->unref(sscreen->ws))
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(sscreen->aux_contexts); i++) {
      if (!sscreen->aux_contexts[i].ctx)
         continue;

      struct si_context *saux = si_get_aux_context(sscreen, &sscreen->aux_contexts[i]);
      struct u_log_context *aux_log = saux->log;
      if (aux_log) {
         saux->b.set_log_context(&saux->b, NULL);
         u_log_context_destroy(aux_log);
         FREE(aux_log);
      }

      saux->b.destroy(&saux->b);
      mtx_unlock(&sscreen->aux_contexts[i].lock);
      mtx_destroy(&sscreen->aux_contexts[i].lock);
   }

   si_fini_gfx_screen(sscreen);

   simple_mtx_destroy(&sscreen->print_ib_mutex);

   slab_destroy_parent(&sscreen->pool_transfers);

   util_idalloc_mt_fini(&sscreen->buffer_ids);

   sscreen->ws->destroy(sscreen->ws);
   FREE(sscreen);
}

static void si_test_vmfault(struct si_screen *sscreen, uint64_t test_flags)
{
   struct pipe_context *ctx = sscreen->aux_context.general.ctx;
   struct si_context *sctx = (struct si_context *)ctx;
   struct pipe_resource *buf = pipe_buffer_create_const0(&sscreen->b, 0, PIPE_USAGE_DEFAULT, 64);

   if (!buf) {
      puts("Buffer allocation failed.");
      exit(1);
   }

   si_resource(buf)->gpu_address = 0; /* cause a VM fault */

   if (test_flags & DBG(TEST_VMFAULT_CP)) {
      si_cp_dma_copy_buffer(sctx, buf, buf, 0, 4, 4);
      ctx->flush(ctx, NULL, 0);
      puts("VM fault test: CP - done.");
   }
   if (test_flags & DBG(TEST_VMFAULT_SHADER)) {
      util_test_constant_buffer(ctx, buf);
      puts("VM fault test: Shader - done.");
   }
   exit(0);
}

static struct pipe_screen *radeonsi_screen_create_impl(struct radeon_winsys *ws,
                                                       const struct pipe_screen_config *config)
{
   struct si_screen *sscreen = CALLOC_STRUCT(si_screen);
   uint64_t test_flags;

   if (!sscreen) {
      return NULL;
   }

   {
#define OPT_BOOL(name, dflt, description)                                                          \
   sscreen->options.name = driQueryOptionb(config->options, "radeonsi_" #name);
#define OPT_INT(name, dflt, description)                                                           \
   sscreen->options.name = driQueryOptioni(config->options, "radeonsi_" #name);
#include "si_debug_options.h"
   }

   sscreen->ws = ws;
   ws->query_info(ws, &sscreen->info);

   sscreen->debug_flags = debug_get_flags_option("R600_DEBUG", radeonsi_debug_options, 0);
   sscreen->debug_flags |= debug_get_flags_option("AMD_DEBUG", radeonsi_debug_options, 0);
   test_flags = debug_get_flags_option("AMD_TEST", test_options, 0);

   if ((sscreen->debug_flags & DBG(TMZ)) &&
       !sscreen->info.has_tmz_support) {
      fprintf(stderr, "radeonsi: requesting TMZ features but TMZ is not supported\n");
      FREE(sscreen);
      return NULL;
   }

   util_idalloc_mt_init_tc(&sscreen->buffer_ids);

   /* Set functions first. */
   sscreen->b.context_create = si_pipe_create_context;
   sscreen->b.destroy = si_destroy_screen;

   si_init_screen_buffer_functions(sscreen);
   si_init_screen_fence_functions(sscreen);
   si_init_screen_state_functions(sscreen);
   si_init_screen_texture_functions(sscreen);

   si_init_screen_get_functions(sscreen);
   si_init_screen_caps(sscreen);

   if (sscreen->debug_flags & DBG(INFO))
      ac_print_gpu_info(stdout, &sscreen->info, ws->get_fd(ws));

   slab_create_parent(&sscreen->pool_transfers, sizeof(struct si_transfer), 64);

   (void)simple_mtx_init(&sscreen->print_ib_mutex, mtx_plain);

   if (!si_init_gfx_screen(sscreen)) {
      FREE(sscreen);
      return NULL;
   }
   /* Don't fail if the multimedia support is missing. */
   si_init_mm_screen(sscreen);

   si_init_renderer_string(sscreen);

   for (unsigned i = 0; i < ARRAY_SIZE(sscreen->aux_contexts); i++)
      (void)mtx_init(&sscreen->aux_contexts[i].lock, mtx_plain | mtx_recursive);

   if (test_flags & DBG(TEST_CLEAR_BUFFER))
      si_test_clear_buffer(sscreen);

   if (test_flags & DBG(TEST_COPY_BUFFER))
      si_test_copy_buffer(sscreen);

   if (test_flags & DBG(TEST_IMAGE_COPY))
      si_test_image_copy_region(sscreen);

   if (test_flags & DBG(TEST_COMPUTE_BLIT))
      si_test_blit(sscreen, test_flags);

   if (test_flags & DBG(TEST_DMA_PERF))
      si_test_dma_perf(sscreen);

   if (test_flags & DBG(TEST_MEM_PERF))
      si_test_mem_perf(sscreen);

   if (test_flags & (DBG(TEST_VMFAULT_CP) | DBG(TEST_VMFAULT_SHADER)))
      si_test_vmfault(sscreen, test_flags);

   return &sscreen->b;
}

struct pipe_screen *radeonsi_screen_create(int fd, const struct pipe_screen_config *config)
{
   struct radeon_winsys *rw = NULL;
   drmVersionPtr version;

   version = drmGetVersion(fd);
   if (!version)
     return NULL;

#if AMD_LLVM_AVAILABLE
   /* LLVM must be initialized before util_queue because both u_queue and LLVM call atexit,
    * and LLVM must call it first because its atexit handler executes C++ destructors,
    * which must be done after our compiler threads using LLVM in u_queue are finished
    * by their atexit handler. Since atexit handlers are called in the reverse order,
    * LLVM must be initialized first, followed by u_queue.
    */
   ac_init_llvm_once();
#endif

   driParseConfigFiles(config->options, config->options_info, 0, "radeonsi",
                       NULL, NULL, NULL, 0, NULL, 0);

#ifdef HAVE_AMDGPU_VIRTIO
   if (strcmp(version->name, "virtio_gpu") == 0) {
      rw = amdgpu_winsys_create(fd, config, radeonsi_screen_create_impl, true);
   } else if (debug_get_bool_option("AMD_FORCE_VPIPE", false)) {
      rw = amdgpu_winsys_create(-1, config, radeonsi_screen_create_impl, true);
   } else
#endif
   {
      switch (version->version_major) {
      case 2:
         rw = radeon_drm_winsys_create(fd, config, radeonsi_screen_create_impl);
         break;
      case 3:
         rw = amdgpu_winsys_create(fd, config, radeonsi_screen_create_impl, false);
         break;
      }
   }

   drmFreeVersion(version);
   return rw ? rw->screen : NULL;
}

struct si_context *si_get_aux_context(struct si_screen *sscreen, struct si_aux_context *actx)
{
   mtx_lock(&actx->lock);
   /* Init aux_context on demand. */
   if (actx->ctx == NULL) {
      bool compute = !sscreen->info.has_graphics ||
                     actx == &sscreen->aux_context.compute_resource_init ||
                     actx == &sscreen->aux_context.shader_upload;
      actx->ctx =
         si_create_context(&sscreen->b,
                           SI_CONTEXT_FLAG_AUX | PIPE_CONTEXT_LOSE_CONTEXT_ON_RESET |
                           (sscreen->options.aux_debug ? PIPE_CONTEXT_DEBUG : 0) |
                           (compute ? PIPE_CONTEXT_COMPUTE_ONLY : 0));
      assert(actx->ctx);

      if (sscreen->options.aux_debug) {
         u_log_context_init(&actx->log);

         struct pipe_context *ctx = actx->ctx;
         ctx->set_log_context(ctx, &actx->log);
      }
   }
   return (struct si_context*)actx->ctx;
}

void si_put_aux_context_flush(struct si_aux_context *ctx)
{
   ctx->ctx->flush(ctx->ctx, NULL, 0);
   mtx_unlock(&ctx->lock);
}
