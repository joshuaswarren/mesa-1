/*
 * Copyright 2026 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */
#include "si_gfx.h"
#include "si_pipe.h"
#include "compiler/nir/nir.h"
#include "ac_shader_util.h"
#include "ac_shadowed_regs.h"
#include "util/disk_cache.h"
#include "aco_interface.h"
#include "util/hex.h"
#include "util/u_cpu_detect.h"

#include <sys/utsname.h>
#include <ctype.h>

#if AMD_LLVM_AVAILABLE
#include "ac_llvm_util.h"
#endif

#include <xf86drm.h>

static const struct debug_named_value radeonsi_shader_debug_options[] = {
   /* Shader logging options: */
   {"vs", DBG(VS), "Print vertex shaders"},
   {"ps", DBG(PS), "Print pixel shaders"},
   {"gs", DBG(GS), "Print geometry shaders"},
   {"tcs", DBG(TCS), "Print tessellation control shaders"},
   {"tes", DBG(TES), "Print tessellation evaluation shaders"},
   {"cs", DBG(CS), "Print compute shaders"},
   {"ts", DBG(TS), "Print task shaders"},
   {"ms", DBG(MS), "Print mesh shaders"},

   {"initnir", DBG(INIT_NIR), "Print initial input NIR when shaders are created"},
   {"nir", DBG(NIR), "Print final NIR after lowering when shader variants are created"},
   {"initllvm", DBG(INIT_LLVM), "Print initial LLVM IR before optimizations"},
   {"llvm", DBG(LLVM), "Print final LLVM IR"},
   {"initaco", DBG(INIT_ACO), "Print initial ACO IR before optimizations"},
   {"aco", DBG(ACO), "Print final ACO IR"},
   {"asm", DBG(ASM), "Print final shaders in asm"},
   {"stats", DBG(STATS), "Print shader-db stats to stderr"},

   /* Shader compiler options the shader cache should be aware of: */
   {"w32ge", DBG(W32_GE), "Use Wave32 for vertex, tessellation, and geometry shaders."},
   {"w32ps", DBG(W32_PS), "Use Wave32 for pixel shaders."},
   {"w32cs", DBG(W32_CS), "Use Wave32 for computes shaders."},
   {"w64ge", DBG(W64_GE), "Use Wave64 for vertex, tessellation, and geometry shaders."},
   {"w64ps", DBG(W64_PS), "Use Wave64 for pixel shaders."},
   {"w64cs", DBG(W64_CS), "Use Wave64 for computes shaders."},

   /* Shader compiler options (with no effect on the shader cache): */
   {"checkir", DBG(CHECK_IR), "Enable additional sanity checks on shader IR"},
   {"mono", DBG(MONOLITHIC_SHADERS), "Use old-style monolithic shaders compiled on demand"},
   {"nooptvariant", DBG(NO_OPT_VARIANT), "Disable compiling optimized shader variants."},
   {"usellvm", DBG(USE_LLVM), "Use LLVM as shader compiler when possible"},

   DEBUG_NAMED_VALUE_END /* must be last */
};

static void si_init_gs_info(struct si_screen *sscreen)
{
   sscreen->gs_table_depth = ac_get_gs_table_depth(sscreen->info.gfx_level, sscreen->info.family);
}

static void
parse_hex(char *out, const char *in, unsigned length)
{
   for (unsigned i = 0; i < length; ++i)
      out[i] = 0;

   for (unsigned i = 0; i < 2 * length; ++i) {
      unsigned v = in[i] <= '9' ? in[i] - '0' : (in[i] >= 'a' ? (in[i] - 'a' + 10) : (in[i] - 'A' + 10));
      out[i / 2] |= v << (4 * (1 - i % 2));
   }
}

static void si_disk_cache_create(struct si_screen *sscreen)
{
   /* Don't use the cache if shader dumping is enabled. */
   if (sscreen->shader_debug_flags & DBG_ALL_SHADERS)
      return;

   blake3_hasher ctx;
   unsigned char blake3[BLAKE3_KEY_LEN];
   char cache_id[BLAKE3_HEX_LEN];

   _mesa_blake3_init(&ctx);

#ifdef RADEONSI_BUILD_ID_OVERRIDE
   {
      unsigned size = strlen(RADEONSI_BUILD_ID_OVERRIDE) / 2;
      char *data = alloca(size);
      parse_hex(data, RADEONSI_BUILD_ID_OVERRIDE, size);
      _mesa_blake3_update(&ctx, data, size);
   }
#else
   if (!disk_cache_get_function_identifier(si_disk_cache_create, &ctx))
      return;
#endif

#if AMD_LLVM_AVAILABLE
   if (!disk_cache_get_function_identifier(LLVMInitializeAMDGPUTargetInfo, &ctx))
      return;
#endif

   /* NIR options depend on si_screen::use_aco, which affects all shaders, including GLSL
    * compilation.
    */
   _mesa_blake3_update(&ctx, &sscreen->use_aco, sizeof(sscreen->use_aco));

   _mesa_blake3_final(&ctx, blake3);
   mesa_bytes_to_hex(cache_id, blake3, BLAKE3_KEY_LEN);

   sscreen->disk_shader_cache = disk_cache_create(ac_get_family_name(sscreen->info.family),
                                                  cache_id, sscreen->info.address32_hi);
}

static void si_set_max_shader_compiler_threads(struct pipe_screen *screen, unsigned max_threads)
{
   struct si_screen *sscreen = (struct si_screen *)screen;

   /* This function doesn't allow a greater number of threads than
    * the queue had at its creation. */
   util_queue_adjust_num_threads(&sscreen->shader_compiler_queue, max_threads, false);
   /* Don't change the number of threads on the low priority queue. */
}

static bool si_is_parallel_shader_compilation_finished(struct pipe_screen *screen, void *shader,
                                                       mesa_shader_stage shader_type)
{
   struct si_shader_selector *sel = (struct si_shader_selector *)shader;

   return util_queue_fence_is_signalled(&sel->ready);
}

static void si_setup_force_shader_use_aco(struct si_screen *sscreen, bool support_aco)
{
   /* Usage:
    *   1. shader type: vs|tcs|tes|gs|ps|cs, specify a class of shaders to use aco
    *   2. shader blake: specify a single shader blake directly to use aco
    *   3. filename: specify a file which contains shader blakes in lines
    */

   sscreen->use_aco_shader_type = MESA_SHADER_NONE;

   if (sscreen->use_aco || !support_aco)
      return;

   const char *option = debug_get_option("AMD_FORCE_SHADER_USE_ACO", NULL);
   if (!option)
      return;

   if (!strcmp("vs", option)) {
      sscreen->use_aco_shader_type = MESA_SHADER_VERTEX;
      return;
   } else if (!strcmp("tcs", option)) {
      sscreen->use_aco_shader_type = MESA_SHADER_TESS_CTRL;
      return;
   } else if (!strcmp("tes", option)) {
      sscreen->use_aco_shader_type = MESA_SHADER_TESS_EVAL;
      return;
   } else if (!strcmp("gs", option)) {
      sscreen->use_aco_shader_type = MESA_SHADER_GEOMETRY;
      return;
   } else if (!strcmp("ps", option)) {
      sscreen->use_aco_shader_type = MESA_SHADER_FRAGMENT;
      return;
   } else if (!strcmp("cs", option)) {
      sscreen->use_aco_shader_type = MESA_SHADER_COMPUTE;
      return;
   }

   blake3_hash blake;
   if (_mesa_blake3_from_printed_string(blake, option)) {
      sscreen->use_aco_shader_blakes = MALLOC(sizeof(blake));
      memcpy(sscreen->use_aco_shader_blakes[0], blake, sizeof(blake));
      sscreen->num_use_aco_shader_blakes = 1;
      return;
   }

   FILE *f = fopen(option, "r");
   if (!f) {
      mesa_loge("invalid AMD_FORCE_SHADER_USE_ACO value");
      return;
   }

   unsigned max_size = 16 * sizeof(blake3_hash);
   sscreen->use_aco_shader_blakes = MALLOC(max_size);

   char line[1024];
   while (fgets(line, sizeof(line), f)) {
      if (sscreen->num_use_aco_shader_blakes * sizeof(blake3_hash) >= max_size) {
         sscreen->use_aco_shader_blakes = REALLOC(
            sscreen->use_aco_shader_blakes, max_size, max_size * 2);
         max_size *= 2;
      }

      if (line[BLAKE3_PRINTED_LEN] == '\n')
         line[BLAKE3_PRINTED_LEN] = 0;

      if (_mesa_blake3_from_printed_string(
             sscreen->use_aco_shader_blakes[sscreen->num_use_aco_shader_blakes], line))
         sscreen->num_use_aco_shader_blakes++;
   }

   fclose(f);
}

static bool
is_pro_graphics(struct si_screen *sscreen)
{
   return  strstr(sscreen->info.marketing_name, "Pro") ||
           strstr(sscreen->info.marketing_name, "PRO") ||
           strstr(sscreen->info.marketing_name, "Frontier");
}

static bool
si_is_compute_copy_faster(struct pipe_screen *pscreen,
                          enum pipe_format src_format,
                          enum pipe_format dst_format,
                          unsigned width,
                          unsigned height,
                          unsigned depth,
                          bool cpu)
{
   if (cpu)
      /* very basic for now */
      return (uint64_t)width * height * depth > 64 * 64;
   return false;
}

static void
si_driver_thread_add_job(struct pipe_screen *screen, void *data,
                         struct util_queue_fence *fence,
                         pipe_driver_thread_func execute,
                         pipe_driver_thread_func cleanup,
                         const size_t job_size)
{
   struct si_screen *sscreen = (struct si_screen *)screen;
   util_queue_add_job(&sscreen->shader_compiler_queue, data, fence, execute, cleanup, job_size);
}

static struct disk_cache *si_get_disk_shader_cache(struct pipe_screen *pscreen)
{
   struct si_screen *sscreen = (struct si_screen *)pscreen;

   return sscreen->disk_shader_cache;
}

bool si_init_gfx_screen(struct si_screen *sscreen) {
   unsigned hw_threads, num_comp_hi_threads, num_comp_lo_threads;
   const bool support_aco = aco_is_gpu_supported(&sscreen->info);
   bool support_llvm = false;

#if AMD_LLVM_AVAILABLE
   support_llvm = strlen(ac_get_llvm_processor_name(sscreen->info.family)) != 0;
#endif

   sscreen->has_gfx_compute = support_aco || support_llvm;

   if (!sscreen->has_gfx_compute)
      return true;

   ac_get_task_info(&sscreen->info, &sscreen->task_info);

   si_disk_cache_create(sscreen);

   if (sscreen->info.gfx_level >= GFX11) {
      sscreen->use_ngg = true;
      sscreen->use_ngg_culling = sscreen->info.max_render_backends >= 2 &&
                                 !(sscreen->debug_flags & DBG(NO_NGG_CULLING));
   } else {
      sscreen->use_ngg = !(sscreen->debug_flags & DBG(NO_NGG)) &&
                         sscreen->info.gfx_level >= GFX10 &&
                         (sscreen->info.family != CHIP_NAVI14 ||
                          is_pro_graphics(sscreen));
      sscreen->use_ngg_culling = sscreen->use_ngg &&
                                 sscreen->info.max_render_backends >= 2 &&
                                 !(sscreen->debug_flags & DBG(NO_NGG_CULLING));
   }

   sscreen->has_draw_indirect_multi =
      (sscreen->info.family >= CHIP_POLARIS10) ||
      (sscreen->info.gfx_level == GFX8 && sscreen->info.pfp_fw_version >= 121 &&
       sscreen->info.me_fw_version >= 87) ||
      (sscreen->info.gfx_level == GFX7 && sscreen->info.pfp_fw_version >= 211 &&
       sscreen->info.me_fw_version >= 173) ||
      (sscreen->info.gfx_level == GFX6 && sscreen->info.pfp_fw_version >= 79 &&
       sscreen->info.me_fw_version >= 142);

   si_driver_ds_init();

   sscreen->b.get_disk_shader_cache = si_get_disk_shader_cache;
   sscreen->b.is_compute_copy_faster = si_is_compute_copy_faster;
   sscreen->b.driver_thread_add_job = si_driver_thread_add_job;

   sscreen->context_roll_log_filename = debug_get_option("AMD_ROLLS", NULL);
   sscreen->shader_debug_flags = debug_get_flags_option("AMD_DEBUG", radeonsi_shader_debug_options, 0);

   if (sscreen->debug_flags & DBG(NO_DISPLAY_DCC)) {
      sscreen->info.use_display_dcc_unaligned = false;
      sscreen->info.use_display_dcc_with_retile_blit = false;
   }

   /* Using the environment variable doesn't enable PAIRS packets for simplicity. */
   if ((sscreen->debug_flags & DBG(SHADOW_REGS)) &&
       !(sscreen->info.userq_ip_mask & (1 << AMD_IP_GFX)))
      sscreen->info.has_kernelq_reg_shadowing = true;

#if AMD_LLVM_AVAILABLE
   sscreen->use_aco = support_aco && sscreen->info.has_image_opcodes &&
                      !(sscreen->shader_debug_flags & DBG(USE_LLVM));
#else
   sscreen->use_aco = true;
#endif

   if (sscreen->use_aco && !support_aco) {
      mesa_loge("ACO does not support this chip yet");
      return false;
   }

   si_setup_force_shader_use_aco(sscreen, support_aco);

   sscreen->b.set_max_shader_compiler_threads = si_set_max_shader_compiler_threads;
   sscreen->b.is_parallel_shader_compilation_finished = si_is_parallel_shader_compilation_finished;
   sscreen->b.finalize_nir = si_finalize_nir;

   sscreen->nir_options = CALLOC_STRUCT(nir_shader_compiler_options);

   si_init_screen_state_functions(sscreen);
   si_init_screen_query_functions(sscreen);
   si_init_screen_live_shader_cache(sscreen);

   si_init_screen_nir_options(sscreen);
   si_init_shader_caps(sscreen);
   si_init_compute_caps(sscreen);
   si_init_gfx_caps(sscreen);
   if (sscreen->b.caps.mesh_shader)
      si_init_mesh_caps(sscreen);

   sscreen->force_aniso = MIN2(16, debug_get_num_option("R600_TEX_ANISO", -1));
   if (sscreen->force_aniso == -1) {
      sscreen->force_aniso = MIN2(16, debug_get_num_option("AMD_TEX_ANISO", -1));
   }

   if (sscreen->force_aniso >= 0) {
      printf("radeonsi: Forcing anisotropy filter to %ix\n",
             /* round down to a power of two */
             1 << util_logbase2(sscreen->force_aniso));
   }

   (void)simple_mtx_init(&sscreen->async_compute_context_lock, mtx_plain);
   (void)simple_mtx_init(&sscreen->gpu_load_mutex, mtx_plain);
   (void)simple_mtx_init(&sscreen->gds_mutex, mtx_plain);
   (void)simple_mtx_init(&sscreen->tess_ring_lock, mtx_plain);

   si_init_gs_info(sscreen);
   if (!si_init_shader_cache(sscreen)) {
      FREE(sscreen->nir_options);
      return false;
   }

   if (sscreen->info.gfx_level < GFX10_3)
      sscreen->options.vrs2x2 = false;

   /* Determine the number of shader compiler threads. */
   const struct util_cpu_caps_t *caps = util_get_cpu_caps();
   hw_threads = caps->nr_cpus;

   if (hw_threads >= 12) {
      num_comp_hi_threads = hw_threads * 3 / 4;
      num_comp_lo_threads = hw_threads / 3;
   } else if (hw_threads >= 6) {
      num_comp_hi_threads = hw_threads - 2;
      num_comp_lo_threads = hw_threads / 2;
   } else if (hw_threads >= 2) {
      num_comp_hi_threads = hw_threads - 1;
      num_comp_lo_threads = hw_threads / 2;
   } else {
      num_comp_hi_threads = 1;
      num_comp_lo_threads = 1;
   }

#if !defined(NDEBUG)
   nir_process_debug_variable();

   /* Use a single compilation thread if NIR printing is enabled to avoid
    * multiple shaders being printed at the same time.
    */
   if (NIR_DEBUG(PRINT)) {
      num_comp_hi_threads = 1;
      num_comp_lo_threads = 1;
   }
#endif

   num_comp_hi_threads = MIN2(num_comp_hi_threads, ARRAY_SIZE(sscreen->compiler));
   num_comp_lo_threads = MIN2(num_comp_lo_threads, ARRAY_SIZE(sscreen->compiler_lowp));

   /* Take a reference on the glsl types for the compiler threads. */
   glsl_type_singleton_init_or_ref();

   /* Start with a single thread and a single slot.
    * Each time we'll hit the "all slots are in use" case, the number of threads and
    * slots will be increased.
    */
   int num_slots = num_comp_hi_threads == 1 ? 64 : 1;
   if (!util_queue_init(&sscreen->shader_compiler_queue, "sh", num_slots,
                        num_comp_hi_threads,
                        UTIL_QUEUE_INIT_RESIZE_IF_FULL |
                        UTIL_QUEUE_INIT_SET_FULL_THREAD_AFFINITY, NULL)) {
      si_destroy_shader_cache(sscreen);
      FREE(sscreen->nir_options);
      glsl_type_singleton_decref();
      return false;
   }

   if (!util_queue_init(&sscreen->shader_compiler_queue_opt_variants, "sh_opt", num_slots,
                        num_comp_lo_threads,
                        UTIL_QUEUE_INIT_RESIZE_IF_FULL |
                        UTIL_QUEUE_INIT_SET_FULL_THREAD_AFFINITY, NULL)) {
      si_destroy_shader_cache(sscreen);
      FREE(sscreen->nir_options);
      glsl_type_singleton_decref();
      return false;
   }

   if (!debug_get_bool_option("RADEON_DISABLE_PERFCOUNTERS", false))
      si_init_perfcounters(sscreen);

   if (sscreen->debug_flags & DBG(NO_OUT_OF_ORDER))
      sscreen->info.has_out_of_order_rast = false;

   /* Only set this for the cases that are known to work, which are:
    * - GFX9 if bpp >= 4 (in bytes)
    */
   if (sscreen->info.gfx_level >= GFX10) {
      memset(sscreen->allow_dcc_msaa_clear_to_reg_for_bpp, true,
             sizeof(sscreen->allow_dcc_msaa_clear_to_reg_for_bpp));
   } else if (sscreen->info.gfx_level == GFX9) {
      for (unsigned bpp_log2 = util_logbase2(1); bpp_log2 <= util_logbase2(16); bpp_log2++)
         sscreen->allow_dcc_msaa_clear_to_reg_for_bpp[bpp_log2] = true;
   }

   /* DCC stores have 50% performance of uncompressed stores and sometimes
    * even less than that. It's risky to enable on dGPUs.
    */
   sscreen->always_allow_dcc_stores = !(sscreen->debug_flags & DBG(NO_DCC_STORE)) &&
                                      (sscreen->debug_flags & DBG(DCC_STORE) ||
                                       sscreen->info.gfx_level >= GFX11 || /* always enabled on gfx11 */
                                       (sscreen->info.gfx_level >= GFX10_3 &&
                                        !sscreen->info.has_dedicated_vram));

   sscreen->dpbb_allowed = !(sscreen->debug_flags & DBG(NO_DPBB)) &&
                           (sscreen->info.gfx_level >= GFX10 ||
                            /* Only enable primitive binning on gfx9 APUs by default. */
                            (sscreen->info.gfx_level == GFX9 && !sscreen->info.has_dedicated_vram) ||
                            sscreen->debug_flags & DBG(DPBB));

   if (sscreen->dpbb_allowed) {
      if ((sscreen->info.has_dedicated_vram && sscreen->info.max_render_backends > 4) ||
	  sscreen->info.gfx_level >= GFX10) {
	 /* Only bin draws that have no CONTEXT and SH register changes between
	  * them because higher settings cause hangs. We've only been able to
	  * reproduce hangs on smaller chips (e.g. Navi24, Phoenix), though all
	  * chips might have them. What we see may be due to a driver bug.
	  */
         sscreen->pbb_context_states_per_bin = 1;
         sscreen->pbb_persistent_states_per_bin = 1;
      } else {
         /* This is a workaround for:
          *    https://bugs.freedesktop.org/show_bug.cgi?id=110214
          * (an alternative is to insert manual BATCH_BREAK event when
          *  a context_roll is detected). */
         sscreen->pbb_context_states_per_bin = sscreen->info.has_gfx9_scissor_bug ? 1 : 3;
         sscreen->pbb_persistent_states_per_bin = 8;
      }

      if (!sscreen->info.has_gfx9_scissor_bug)
         sscreen->pbb_context_states_per_bin =
            debug_get_num_option("AMD_DEBUG_DPBB_CS", sscreen->pbb_context_states_per_bin);
      sscreen->pbb_persistent_states_per_bin =
         debug_get_num_option("AMD_DEBUG_DPBB_PS", sscreen->pbb_persistent_states_per_bin);

      assert(sscreen->pbb_context_states_per_bin >= 1 &&
             sscreen->pbb_context_states_per_bin <= 6);
      assert(sscreen->pbb_persistent_states_per_bin >= 1 &&
             sscreen->pbb_persistent_states_per_bin <= 32);
   }

   (void)simple_mtx_init(&sscreen->shader_parts_mutex, mtx_plain);
   sscreen->use_monolithic_shaders =
      (sscreen->shader_debug_flags & DBG(MONOLITHIC_SHADERS)) != 0;

   if (debug_get_bool_option("RADEON_DUMP_SHADERS", false))
      sscreen->shader_debug_flags |= DBG_ALL_SHADERS;

   /* Syntax:
    *     EQAA=s,z,c
    * Example:
    *     EQAA=8,4,2

    * That means 8 coverage samples, 4 Z/S samples, and 2 color samples.
    * Constraints:
    *     s >= z >= c (ignoring this only wastes memory)
    *     s = [2..16]
    *     z = [2..8]
    *     c = [2..8]
    *
    * Only MSAA color and depth buffers are overridden.
    */
   if (sscreen->info.has_eqaa_surface_allocator) {
      const char *eqaa = debug_get_option("EQAA", NULL);
      unsigned s, z, f;

      if (eqaa && sscanf(eqaa, "%u,%u,%u", &s, &z, &f) == 3 && s && z && f) {
         sscreen->eqaa_force_coverage_samples = s;
         sscreen->eqaa_force_z_samples = z;
         sscreen->eqaa_force_color_samples = f;
      }
   }

   if (sscreen->info.gfx_level >= GFX11) {
      sscreen->attribute_pos_prim_ring =
         si_aligned_buffer_create(&sscreen->b,
                                  PIPE_RESOURCE_FLAG_UNMAPPABLE |
                                  SI_RESOURCE_FLAG_32BIT |
                                  SI_RESOURCE_FLAG_DRIVER_INTERNAL |
                                  SI_RESOURCE_FLAG_DISCARDABLE,
                                  PIPE_USAGE_DEFAULT,
                                  sscreen->info.total_attribute_pos_prim_ring_size,
                                  2 * 1024 * 1024);
   }

   ac_print_nonshadowed_regs(sscreen->info.gfx_level, sscreen->info.family);

   return true;
}

void si_fini_gfx_screen(struct si_screen *sscreen) {
   struct si_shader_part *parts[] = {sscreen->ps_prologs, sscreen->ps_epilogs};
   unsigned i;

   if (!sscreen->has_gfx_compute)
      return;

   if (sscreen->debug_flags & DBG(CACHE_STATS)) {
      printf("live shader cache:   hits = %u, misses = %u\n", sscreen->live_shader_cache.hits,
             sscreen->live_shader_cache.misses);
      printf("memory shader cache: hits = %u, misses = %u\n", sscreen->num_memory_shader_cache_hits,
             sscreen->num_memory_shader_cache_misses);
      printf("disk shader cache:   hits = %u, misses = %u\n", sscreen->num_disk_shader_cache_hits,
             sscreen->num_disk_shader_cache_misses);
   }

   si_resource_reference(&sscreen->attribute_pos_prim_ring, NULL);
   si_resource_reference(&sscreen->attribute_pos_prim_ring_tmz, NULL);
   pipe_resource_reference(&sscreen->tess_rings, NULL);
   pipe_resource_reference(&sscreen->tess_rings_tmz, NULL);

   util_queue_destroy(&sscreen->shader_compiler_queue);
   util_queue_destroy(&sscreen->shader_compiler_queue_opt_variants);

   simple_mtx_destroy(&sscreen->async_compute_context_lock);
   if (sscreen->async_compute_context)
      sscreen->async_compute_context->destroy(sscreen->async_compute_context);

   /* Release the reference on glsl types of the compiler threads. */
   glsl_type_singleton_decref();

   for (i = 0; i < ARRAY_SIZE(sscreen->compiler); i++) {
      if (sscreen->compiler[i])
         si_destroy_llvm_compiler(sscreen->compiler[i]);
   }

   for (i = 0; i < ARRAY_SIZE(sscreen->compiler_lowp); i++) {
      if (sscreen->compiler_lowp[i])
         si_destroy_llvm_compiler(sscreen->compiler_lowp[i]);
   }

   /* Free shader parts. */
   for (i = 0; i < ARRAY_SIZE(parts); i++) {
      while (parts[i]) {
         struct si_shader_part *part = parts[i];

         parts[i] = part->next;
         si_shader_binary_clean(&part->binary);
         FREE(part);
      }
   }
   simple_mtx_destroy(&sscreen->shader_parts_mutex);
   si_destroy_shader_cache(sscreen);

   si_destroy_perfcounters(sscreen);
   si_gpu_load_kill_thread(sscreen);

   simple_mtx_destroy(&sscreen->gpu_load_mutex);
   simple_mtx_destroy(&sscreen->gds_mutex);
   simple_mtx_destroy(&sscreen->tess_ring_lock);

   radeon_bo_reference(sscreen->ws, &sscreen->gds_oa, NULL);

   disk_cache_destroy(sscreen->disk_shader_cache);
   util_vertex_state_cache_deinit(&sscreen->vertex_state_cache);

   util_live_shader_cache_deinit(&sscreen->live_shader_cache);

   FREE(sscreen->use_aco_shader_blakes);
   FREE(sscreen->nir_options);
}
