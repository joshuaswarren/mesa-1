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

struct ac_llvm_compiler *si_create_llvm_compiler(struct si_screen *sscreen)
{
#if AMD_LLVM_AVAILABLE
   struct ac_llvm_compiler *compiler = CALLOC_STRUCT(ac_llvm_compiler);
   if (!compiler)
      return NULL;

   if (!ac_init_llvm_compiler(compiler, sscreen->info.family,
                              sscreen->shader_debug_flags & DBG(CHECK_IR) ? AC_TM_CHECK_IR : 0))
      return NULL;

   compiler->beo = ac_create_backend_optimizer(compiler->tm);
   return compiler;
#else
   return NULL;
#endif
}

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

void si_destroy_llvm_compiler(struct ac_llvm_compiler *compiler)
{
#if AMD_LLVM_AVAILABLE
   ac_destroy_llvm_compiler(compiler);
   FREE(compiler);
#endif
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

   si_fini_mm_context(sctx);

   util_queue_finish(&sctx->screen->shader_compiler_queue);
   util_queue_finish(&sctx->screen->shader_compiler_queue_opt_variants);

   if (context->set_debug_callback)
      context->set_debug_callback(context, NULL);

   util_unreference_framebuffer_state(&sctx->framebuffer.state);
   si_release_all_descriptors(sctx);

   if (sctx->gfx_level >= GFX10 && sctx->is_gfx_queue)
      si_gfx11_destroy_query(sctx);

   if (sctx->sqtt) {
      struct si_screen *sscreen = sctx->screen;

      si_handle_sqtt(sctx, &sctx->gfx_cs);

      if (sscreen->b.num_contexts == 1 && !(sctx->context_flags & SI_CONTEXT_FLAG_AUX))
          sscreen->ws->cs_set_pstate(&sctx->gfx_cs, RADEON_CTX_PSTATE_NONE);

      si_destroy_sqtt(sctx);
   }

   si_utrace_fini(sctx);

   pipe_resource_reference(&sctx->esgs_ring, NULL);
   pipe_resource_reference(&sctx->gsvs_ring, NULL);
   pipe_resource_reference(&sctx->null_const_buf.buffer, NULL);
   si_resource_reference(&sctx->border_color_buffer, NULL);
   free(sctx->border_color_table);
   si_resource_reference(&sctx->scratch_buffer, NULL);
   si_resource_reference(&sctx->compute_scratch_buffer, NULL);
   si_resource_reference(&sctx->wait_mem_scratch, NULL);
   si_resource_reference(&sctx->wait_mem_scratch_tmz, NULL);
   si_resource_reference(&sctx->small_prim_cull_info_buf, NULL);
   si_resource_reference(&sctx->pipeline_stats_query_buf, NULL);

   if (sctx->cs_preamble_state)
      si_pm4_free_state(sctx, sctx->cs_preamble_state, ~0);
   if (sctx->cs_preamble_state_tmz)
      si_pm4_free_state(sctx, sctx->cs_preamble_state_tmz, ~0);

   if (sctx->fixed_func_tcs_shader_cache) {
      hash_table_foreach(sctx->fixed_func_tcs_shader_cache, entry) {
         sctx->b.delete_tcs_state(&sctx->b, entry->data);
      }
      _mesa_hash_table_destroy(sctx->fixed_func_tcs_shader_cache, NULL);
   }

   if (sctx->custom_dsa_flush)
      sctx->b.delete_depth_stencil_alpha_state(&sctx->b, sctx->custom_dsa_flush);
   if (sctx->custom_blend_fmask_decompress)
      sctx->b.delete_blend_state(&sctx->b, sctx->custom_blend_fmask_decompress);
   if (sctx->custom_blend_eliminate_fastclear)
      sctx->b.delete_blend_state(&sctx->b, sctx->custom_blend_eliminate_fastclear);
   if (sctx->custom_blend_dcc_decompress)
      sctx->b.delete_blend_state(&sctx->b, sctx->custom_blend_dcc_decompress);
   if (sctx->vs_blit_pos)
      sctx->b.delete_vs_state(&sctx->b, sctx->vs_blit_pos);
   if (sctx->vs_blit_pos_layered)
      sctx->b.delete_vs_state(&sctx->b, sctx->vs_blit_pos_layered);
   if (sctx->vs_blit_texcoord)
      sctx->b.delete_vs_state(&sctx->b, sctx->vs_blit_texcoord);
   if (sctx->cs_clear_buffer_rmw)
      sctx->b.delete_compute_state(&sctx->b, sctx->cs_clear_buffer_rmw);
   if (sctx->cs_ubyte_to_ushort)
      sctx->b.delete_compute_state(&sctx->b, sctx->cs_ubyte_to_ushort);
   for (unsigned i = 0; i < ARRAY_SIZE(sctx->cs_dcc_retile); i++) {
      if (sctx->cs_dcc_retile[i])
         sctx->b.delete_compute_state(&sctx->b, sctx->cs_dcc_retile[i]);
   }
   if (sctx->no_velems_state)
      sctx->b.delete_vertex_elements_state(&sctx->b, sctx->no_velems_state);

   if (sctx->global_buffers) {
      sctx->b.set_global_binding(&sctx->b, 0, sctx->max_global_buffers, NULL, NULL);
      FREE(sctx->global_buffers);
   }

   for (unsigned i = 0; i < ARRAY_SIZE(sctx->cs_fmask_expand); i++) {
      for (unsigned j = 0; j < ARRAY_SIZE(sctx->cs_fmask_expand[i]); j++) {
         if (sctx->cs_fmask_expand[i][j]) {
            sctx->b.delete_compute_state(&sctx->b, sctx->cs_fmask_expand[i][j]);
         }
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(sctx->cs_clear_image_dcc_single); i++) {
      for (unsigned j = 0; j < ARRAY_SIZE(sctx->cs_clear_image_dcc_single[i]); j++) {
         if (sctx->cs_clear_image_dcc_single[i][j]) {
            sctx->b.delete_compute_state(&sctx->b, sctx->cs_clear_image_dcc_single[i][j]);
         }
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(sctx->cs_clear_dcc_msaa); i++) {
      for (unsigned j = 0; j < ARRAY_SIZE(sctx->cs_clear_dcc_msaa[i]); j++) {
         for (unsigned k = 0; k < ARRAY_SIZE(sctx->cs_clear_dcc_msaa[i][j]); k++) {
            for (unsigned l = 0; l < ARRAY_SIZE(sctx->cs_clear_dcc_msaa[i][j][k]); l++) {
               for (unsigned m = 0; m < ARRAY_SIZE(sctx->cs_clear_dcc_msaa[i][j][k][l]); m++) {
                  if (sctx->cs_clear_dcc_msaa[i][j][k][l][m])
                     sctx->b.delete_compute_state(&sctx->b, sctx->cs_clear_dcc_msaa[i][j][k][l][m]);
               }
            }
         }
      }
   }

   if (sctx->blitter)
      util_blitter_destroy(sctx->blitter);

   if (sctx->query_result_shader)
      sctx->b.delete_compute_state(&sctx->b, sctx->query_result_shader);
   if (sctx->sh_query_result_shader)
      sctx->b.delete_compute_state(&sctx->b, sctx->sh_query_result_shader);

   if (sctx->gfx_cs.priv)
      sctx->ws->cs_destroy(&sctx->gfx_cs);
   if (sctx->ctx)
      sctx->ws->ctx_destroy(sctx->ctx);
   if (sctx->sdma_cs) {
      sctx->ws->cs_destroy(sctx->sdma_cs);
      free(sctx->sdma_cs);
   }

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

   sctx->ws->fence_reference(sctx->ws, &sctx->last_gfx_fence, NULL);
   si_resource_reference(&sctx->eop_bug_scratch, NULL);
   si_resource_reference(&sctx->eop_bug_scratch_tmz, NULL);
   si_resource_reference(&sctx->shadowing.registers, NULL);
   si_resource_reference(&sctx->shadowing.csa, NULL);

   if (sctx->compiler)
      si_destroy_llvm_compiler(sctx->compiler);

   si_saved_cs_reference(&sctx->current_saved_cs, NULL);

   _mesa_hash_table_destroy(sctx->tex_handles, NULL);
   _mesa_hash_table_destroy(sctx->img_handles, NULL);

   util_dynarray_fini(&sctx->resident_tex_handles);
   util_dynarray_fini(&sctx->resident_img_handles);
   util_dynarray_fini(&sctx->resident_tex_needs_color_decompress);
   util_dynarray_fini(&sctx->resident_img_needs_color_decompress);
   util_dynarray_fini(&sctx->resident_tex_needs_depth_decompress);

   if (!(sctx->context_flags & SI_CONTEXT_FLAG_AUX))
      p_atomic_dec(&context->screen->num_contexts);

   if (sctx->cs_dma_shaders) {
      hash_table_u64_foreach(sctx->cs_dma_shaders, entry) {
         context->delete_compute_state(context, entry.data);
      }
      _mesa_hash_table_u64_destroy(sctx->cs_dma_shaders);
   }

   if (sctx->cs_blit_shaders) {
      hash_table_u64_foreach(sctx->cs_blit_shaders, entry) {
         context->delete_compute_state(context, entry.data);
      }
      _mesa_hash_table_u64_destroy(sctx->cs_blit_shaders);
   }

   if (sctx->ps_resolve_shaders) {
      hash_table_u64_foreach(sctx->ps_resolve_shaders, entry) {
         context->delete_fs_state(context, entry.data);
      }
      _mesa_hash_table_u64_destroy(sctx->ps_resolve_shaders);
   }

   si_resource_reference(&sctx->task_wait_buf, NULL);
   si_resource_reference(&sctx->task_ring, NULL);
   si_resource_reference(&sctx->task_scratch_buffer, NULL);
   si_resource_reference(&sctx->mesh_scratch_ring, NULL);
   if (sctx->task_preamble_state)
      si_pm4_free_state(sctx, sctx->task_preamble_state, ~0);

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

static struct pipe_context *si_create_context(struct pipe_screen *screen, unsigned flags)
{
   struct si_screen *sscreen = (struct si_screen *)screen;
   STATIC_ASSERT(DBG_COUNT <= 64);

   /* Don't create a context if it's not compute-only and hw is compute-only. */
   if (!sscreen->info.has_graphics && !(flags & PIPE_CONTEXT_COMPUTE_ONLY)) {
      mesa_loge("can't create a graphics context on a compute chip");
      return NULL;
   }

   struct si_context *sctx = CALLOC_STRUCT(si_context);
   struct radeon_winsys *ws = sscreen->ws;

   if (!sctx) {
      mesa_loge("can't allocate a context");
      return NULL;
   }

   sctx->is_gfx_queue = sscreen->info.gfx_level == GFX6 ||
                        /* Compute queues hang on Raven and derivatives, see:
                         * https://gitlab.freedesktop.org/mesa/mesa/-/issues/12310 */
                        ((sscreen->info.family == CHIP_RAVEN ||
                          sscreen->info.family == CHIP_RAVEN2) &&
                         !sscreen->info.has_dedicated_vram) ||
                        !sscreen->info.ip[AMD_IP_COMPUTE].num_queues ||
                        !(flags & PIPE_CONTEXT_COMPUTE_ONLY);

   if (sctx->is_gfx_queue) {
      if (sscreen->info.userq_ip_mask & (1 << AMD_IP_GFX))
         sctx->uses_userq_reg_shadowing = !(sscreen->debug_flags & DBG(USERQ_NO_SHADOW_REGS));
      else
         sctx->uses_kernelq_reg_shadowing = sscreen->info.has_kernelq_reg_shadowing;
   }

   if (flags & PIPE_CONTEXT_DEBUG)
      sscreen->record_llvm_ir = true; /* racy but not critical */

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

   if (sctx->gfx_level == GFX7 || sctx->gfx_level == GFX8 || sctx->gfx_level == GFX9) {
      sctx->eop_bug_scratch = si_aligned_buffer_create(
         &sscreen->b, PIPE_RESOURCE_FLAG_UNMAPPABLE | SI_RESOURCE_FLAG_DRIVER_INTERNAL,
         PIPE_USAGE_DEFAULT, 16 * sscreen->info.max_render_backends, 256);
      if (!sctx->eop_bug_scratch) {
         mesa_loge("can't create eop_bug_scratch");
         goto fail;
      }
   }

   /* Initialize the context handle and the command stream. */
   sctx->ctx = sctx->ws->ctx_create(sctx->ws, sctx->context_flags);
   if (!sctx->ctx) {
      mesa_loge("can't create radeon_winsys_ctx");
      goto fail;
   }

   if (!ws->cs_create(&sctx->gfx_cs, sctx->ctx, sctx->is_gfx_queue ? AMD_IP_GFX : AMD_IP_COMPUTE,
                      (void *)si_flush_gfx_cs, sctx)) {
      mesa_loge("can't create gfx_cs");
      sctx->gfx_cs.priv = NULL;
      goto fail;
   }
   assert(sctx->gfx_cs.priv);

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

   /* Border colors. */
   if (sscreen->info.compiler_info.has_3d_cube_border_color_mipmap) {
      sctx->border_color_table = malloc(SI_MAX_BORDER_COLORS * sizeof(*sctx->border_color_table));
      if (!sctx->border_color_table) {
         mesa_loge("can't create border_color_table");
         goto fail;
      }

      sctx->border_color_buffer = si_resource(pipe_buffer_create(
         screen, 0, PIPE_USAGE_DEFAULT, SI_MAX_BORDER_COLORS * sizeof(*sctx->border_color_table)));
      if (!sctx->border_color_buffer) {
         mesa_loge("can't create border_color_buffer");
         goto fail;
      }

      sctx->border_color_map =
         ws->buffer_map(ws, sctx->border_color_buffer->buf, NULL, PIPE_MAP_WRITE);
      if (!sctx->border_color_map) {
         mesa_loge("can't map border_color_buffer");
         goto fail;
      }
   }

#if !AMD_LLVM_AVAILABLE
   sctx->shader.vs.key.ge.use_aco = 1;
   sctx->shader.gs.key.ge.use_aco = 1;
   sctx->shader.tcs.key.ge.use_aco = 1;
   sctx->shader.tes.key.ge.use_aco = 1;
#endif

   sctx->ngg = sscreen->use_ngg;

   sctx->b.emit_string_marker = si_emit_string_marker;
   sctx->b.set_debug_callback = si_set_debug_callback;
   sctx->b.set_log_context = si_set_log_context;
   sctx->b.set_context_param = si_set_context_param;
   sctx->b.get_device_reset_status = si_get_reset_status;
   sctx->b.set_device_reset_callback = si_set_device_reset_callback;
   sctx->b.set_frontend_noop = si_set_frontend_noop;

#ifdef HAVE_GFX_COMPUTE
   si_init_all_descriptors(sctx);
   si_init_barrier_functions(sctx);
   si_init_clear_functions(sctx);
   si_init_blit_functions(sctx);
   si_init_compute_functions(sctx);
   si_init_compute_blit_functions(sctx);
   si_init_debug_functions(sctx);
   si_init_query_functions(sctx);
   si_init_state_compute_functions(sctx);
#else
   list_inithead(&sctx->active_queries);
#endif
   si_init_buffer_functions(sctx);
   si_init_fence_functions(sctx);
   si_init_context_texture_functions(sctx);

   /* Initialize graphics-only context functions. */
   if (sctx->is_gfx_queue) {
#ifdef HAVE_GFX_COMPUTE
      if (sctx->gfx_level >= GFX10)
         si_gfx11_init_query(sctx);
      si_init_msaa_functions(sctx);
      si_init_shader_functions(sctx);
      si_init_state_functions(sctx);
      si_init_streamout_functions(sctx);
      si_init_viewport_functions(sctx);

      sctx->blitter = util_blitter_create(&sctx->b);
      if (sctx->blitter == NULL) {
         mesa_loge("can't create blitter");
         goto fail;
      }
      sctx->blitter->use_single_triangle = true;

      /* Some states are expected to be always non-NULL. */
      sctx->noop_blend = util_blitter_get_noop_blend_state(sctx->blitter);
      sctx->queued.named.blend = sctx->noop_blend;

      sctx->noop_dsa = util_blitter_get_noop_dsa_state(sctx->blitter);
      sctx->queued.named.dsa = sctx->noop_dsa;

      sctx->no_velems_state = sctx->b.create_vertex_elements_state(&sctx->b, 0, NULL);
      sctx->vertex_elements = sctx->no_velems_state;

      sctx->discard_rasterizer_state = util_blitter_get_discard_rasterizer_state(sctx->blitter);
      sctx->queued.named.rasterizer = sctx->discard_rasterizer_state;

      switch (sctx->gfx_level) {
      case GFX6:
         si_init_draw_functions_GFX6(sctx);
         break;
      case GFX7:
         si_init_draw_functions_GFX7(sctx);
         break;
      case GFX8:
         si_init_draw_functions_GFX8(sctx);
         break;
      case GFX9:
         si_init_draw_functions_GFX9(sctx);
         break;
      case GFX10:
         si_init_draw_functions_GFX10(sctx);
         break;
      case GFX10_3:
         si_init_draw_functions_GFX10_3(sctx);
         break;
      case GFX11:
         si_init_draw_functions_GFX11(sctx);
         break;
      case GFX11_5:
         si_init_draw_functions_GFX11_5(sctx);
         break;
      case GFX11_7:
         si_init_draw_functions_GFX11_7(sctx);
         break;
      case GFX12:
         si_init_draw_functions_GFX12(sctx);
         break;
      default:
         UNREACHABLE("unhandled gfx level");
      }
#endif
   }

   if (screen->caps.mesh_shader)
      si_init_task_mesh_shader_functions(sctx);

   sctx->sample_mask = 0xffff;

   /* GFX7 cannot unbind a constant buffer (S_BUFFER_LOAD doesn't skip loads
    * if NUM_RECORDS == 0). We need to use a dummy buffer instead. */
   if (sctx->gfx_level == GFX7) {
      sctx->null_const_buf.buffer =
         pipe_aligned_buffer_create(screen,
                                    PIPE_RESOURCE_FLAG_UNMAPPABLE | SI_RESOURCE_FLAG_32BIT |
                                    SI_RESOURCE_FLAG_DRIVER_INTERNAL,
                                    PIPE_USAGE_DEFAULT, 16,
                                    sctx->screen->info.tcc_cache_line_size);
      if (!sctx->null_const_buf.buffer) {
         mesa_loge("can't create null_const_buf");
         goto fail;
      }
      sctx->null_const_buf.buffer_size = sctx->null_const_buf.buffer->width0;

      for (unsigned shader = 0; shader < SI_NUM_SHADERS; shader++) {
         if (!sctx->is_gfx_queue && shader != MESA_SHADER_COMPUTE)
            continue;

         for (unsigned i = 0; i < SI_NUM_CONST_BUFFERS; i++) {
            sctx->b.set_constant_buffer(&sctx->b, shader, i, &sctx->null_const_buf);
         }
      }

      si_set_internal_const_buffer(sctx, SI_HS_CONST_DEFAULT_TESS_LEVELS, &sctx->null_const_buf);
      si_set_internal_const_buffer(sctx, SI_VS_CONST_INSTANCE_DIVISORS, &sctx->null_const_buf);
      si_set_internal_const_buffer(sctx, SI_VS_CONST_CLIP_PLANES, &sctx->null_const_buf);
      si_set_internal_const_buffer(sctx, SI_PS_CONST_POLY_STIPPLE, &sctx->null_const_buf);
   }

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

   /* PIPE_CONTEXT_COMPUTE_ONLY doesn't mean no multimedia, it means no graphics so always
    * init mm but don't fail if it reports an error.
    */
   si_init_mm_context(sscreen, sctx, flags);

   /* The remainder of this function initializes the gfx CS and must be last. */
   assert(sctx->gfx_cs.current.cdw == 0);

   if (!si_init_cp_reg_shadowing(sctx))
      goto fail;

   /* Set immutable fields of shader keys. */
   if (sctx->gfx_level >= GFX9) {
      /* The LS output / HS input layout can be communicated
       * directly instead of via user SGPRs for merged LS-HS.
       * This also enables jumping over the VS for HS-only waves.
       */
      sctx->shader.tcs.key.ge.opt.prefer_mono = 1;

      /* This enables jumping over the VS for GS-only waves. */
      sctx->shader.gs.key.ge.opt.prefer_mono = 1;
   }

   si_utrace_init(sctx);

   si_begin_new_gfx_cs(sctx, true);
   assert(sctx->gfx_cs.current.cdw == sctx->initial_gfx_cs_size);

   if (sctx->gfx_level >= GFX9 && sctx->gfx_level < GFX11) {
      sctx->wait_mem_scratch =
           si_aligned_buffer_create(screen,
                                    PIPE_RESOURCE_FLAG_UNMAPPABLE |
                                    SI_RESOURCE_FLAG_DRIVER_INTERNAL,
                                    PIPE_USAGE_DEFAULT, 4,
                                    sscreen->info.tcc_cache_line_size);
      if (!sctx->wait_mem_scratch) {
         mesa_loge("can't create wait_mem_scratch");
         goto fail;
      }

      si_cp_write_data(sctx, sctx->wait_mem_scratch, 0, 4, V_371_MEMORY, V_371_MICRO_ENGINE,
                       &sctx->wait_mem_number);
   }

   if (sctx->gfx_level == GFX7) {
      /* Clear the NULL constant buffer, because loads should return zeros. */
      uint32_t clear_value = 0;
      si_cp_dma_clear_buffer(sctx, &sctx->gfx_cs, sctx->null_const_buf.buffer, 0,
                             sctx->null_const_buf.buffer->width0, clear_value);
      si_barrier_after_simple_buffer_op(sctx, 0, sctx->null_const_buf.buffer, NULL);
   }

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

   sctx->initial_gfx_cs_size = sctx->gfx_cs.current.cdw;
   sctx->last_timestamp_cmd = NULL;

   sctx->cs_dma_shaders = _mesa_hash_table_u64_create(NULL);
   if (!sctx->cs_dma_shaders)
      goto fail;

   sctx->cs_blit_shaders = _mesa_hash_table_u64_create(NULL);
   if (!sctx->cs_blit_shaders)
      goto fail;

   sctx->ps_resolve_shaders = _mesa_hash_table_u64_create(NULL);
   if (!sctx->ps_resolve_shaders)
      goto fail;

   /* Initialize compute_tmpring_size. */
   si_get_scratch_tmpring_size(sctx, 0, true, &sctx->compute_tmpring_size);

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
