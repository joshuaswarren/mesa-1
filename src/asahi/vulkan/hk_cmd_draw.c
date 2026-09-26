/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include <assert.h>
#include <sys/resource.h>
#include "agx_abi.h"
#include "agx_bg_eot.h"
#include "agx_bo.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_nir_lower_vbo.h"
#include "agx_ppp.h"
#include "agx_ppp_hal200_graphics_core.h"
#include "agx_vdm_state_core.h"
#include "agx_ppp_hal200_raster_core.h"
#include "agx_ppp_hal300_depth_bias_scissor_core.h"
#include "agx_ppp_viewport_payload_core.h"
#include "agx_ppp_w_clamp_cull2_core.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "agx_uvs.h"
#include "agx_vdm_barrier_core.h"
#include "agx_vdm_hal200_bind_tail_core.h"
#include "agx_vdm_hal200_direct_draw_core.h"
#include "agx_vdm_tessellate_core.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_agx3_ioto_program_core.h"
#include "hk_g17p_ppp_indices.h"
#include "hk_g17p_full_frame_scissor.h"
#include "hk_g17p_draw_state_upload.h"
#include "hk_agx3_vdm_decode.h"
#include "hk_physical_device.h"
#include "hk_private.h"
#include "hk_shader.h"
#include "asahi/lib/agx3_g17p_grid_helper_core.h"

#include "asahi/genxml/agx_pack.h"
#include "asahi/libagx/compression.h"
#include "asahi/libagx/libagx.h"
#include "asahi/libagx/query.h"
#include "asahi/libagx/tessellator.h"
#include "poly/geometry.h"
#include "util/blend.h"
#include "util/format/format_utils.h"
#include "util/format/u_formats.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/ralloc.h"
#include "util/u_prim.h"
#include "vulkan/vulkan_core.h"
#include "layout.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_intrinsics.h"
#include "nir_lower_blend.h"
#include "nir_xfb_info.h"
#include "pool.h"
#include "shader_enums.h"
#include "vk_blend.h"

#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_pipeline.h"
#include "vk_render_pass.h"
#include "vk_shader.h"
#include "vk_standard_sample_locations.h"
#include "vk_util.h"

/* Diagnostic controls are immutable for a process. Keep environment parsing
 * out of the per-draw path; all controls remain opt-in and retain their exact
 * existing behavior. */
DEBUG_GET_ONCE_BOOL_OPTION(hk_g17p_interdraw_barrier,
                           "HK_G17P_INTERDRAW_BARRIER", false)
DEBUG_GET_ONCE_BOOL_OPTION(hk_g17p_nonpoint_no_psiz,
                           "HK_G17P_NONPOINT_NO_PSIZ", false)
DEBUG_GET_ONCE_BOOL_OPTION(hk_g17p_hal200_about_flags1,
                           "HK_G17P_HAL200_ABOUT_FLAGS1", false)
DEBUG_GET_ONCE_BOOL_OPTION(hk_g17p_force_tri_merge_disable,
                           "HK_G17P_FORCE_TRI_MERGE_DISABLE", false)
DEBUG_GET_ONCE_NUM_OPTION(hk_g17p_dirty_bisect,
                           "HK_G17P_DIRTY_BISECT", 0)
/* Failure-isolation only.  The production default keeps prepared-state
 * consumption enabled; this forces the authoritative builder so the two
 * paths can be compared on identical hardware and workloads. */
DEBUG_GET_ONCE_BOOL_OPTION(hk_g17p_prepared_dynamic_disable,
                           "HK_G17P_PREPARED_DYNAMIC_DISABLE", false)
#define IS_DIRTY(bit) BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_##bit)

#define IS_SHADER_DIRTY(bit)                                                   \
   (cmd->state.gfx.shaders_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

#define IS_LINKED_DIRTY(bit)                                                   \
   (cmd->state.gfx.linked_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

enum hk_shader_perf_detail_phase {
   HK_PERF_DETAIL_FLUSH,
   HK_PERF_DETAIL_CCS,
   HK_PERF_DETAIL_TESS,
   HK_PERF_DETAIL_GEOM,
   HK_PERF_DETAIL_TESS_SETUP,
   HK_PERF_DETAIL_TESS_VS,
   HK_PERF_DETAIL_TESS_TCS,
   HK_PERF_DETAIL_TESS_FACTORS,
   HK_PERF_DETAIL_STREAM,
   HK_PERF_DETAIL_PREPROCESS,
   HK_PERF_DETAIL_SHADERS,
   HK_PERF_DETAIL_BOOKKEEPING,
   HK_PERF_DETAIL_DYNAMIC,
   HK_PERF_DETAIL_PRELINK_VS,
   HK_PERF_DETAIL_PRELINK_FS,
   HK_PERF_DETAIL_UPLOAD_PARAMS,
   HK_PERF_DETAIL_UPLOAD_VARYINGS,
   HK_PERF_DETAIL_UPLOAD_ROOT,
   HK_PERF_DETAIL_PUBLISH_PROGRAM,
   HK_PERF_DETAIL_PUBLISH_VDM,
   HK_PERF_DETAIL_PUBLISH_TAIL,
   HK_PERF_DETAIL_SW_TESS_SETUP,
   HK_PERF_DETAIL_SW_TESS_RESERVE,
   HK_PERF_DETAIL_SW_TESS_SHADERS,
   HK_PERF_DETAIL_SW_TESS_COUNT,
   HK_PERF_DETAIL_SW_TESS_PREFIX,
   HK_PERF_DETAIL_SW_TESS_FINAL,
   HK_PERF_DETAIL_COUNT,
};

static const char *const hk_shader_perf_detail_names[] = {
   "flush",       "ccs",            "tess",        "geom",
   "tess_setup",  "tess_vs",        "tess_tcs",    "tess_factors",
   "stream",      "preprocess",     "shaders",     "bookkeeping",
   "dynamic",     "prelink_vs",     "prelink_fs",  "upload_params",
   "upload_vary", "upload_root",    "publish_prog", "publish_vdm",
   "publish_tail", "sw_tess_setup", "sw_tess_reserve", "sw_tess_shaders",
   "sw_tess_count", "sw_tess_prefix", "sw_tess_final",
};

static_assert(ARRAY_SIZE(hk_shader_perf_detail_names) ==
                 HK_PERF_DETAIL_COUNT,
              "shader performance phase names");

static _Thread_local uint64_t
   hk_shader_perf_current_detail[HK_PERF_DETAIL_COUNT];

struct hk_shader_perf_draw_group {
   const void *shader;
   uint64_t total_ns;
   uint64_t min_ns;
   uint64_t max_ns;
   uint64_t first_ns;
   uint64_t first_flush_ns;
   uint64_t max_flush_ns;
   uint64_t first_ccs_ns;
   uint64_t first_tess_ns;
   uint64_t first_geom_ns;
   uint64_t first_tess_setup_ns;
   uint64_t first_tess_vs_ns;
   uint64_t first_tess_tcs_ns;
   uint64_t first_tess_factors_ns;
   uint64_t first_bo_alloc_ns;
   uint64_t max_bo_alloc_ns;
   uint32_t count;
   uint32_t max_index;
   uint32_t first_bo_allocs;
   uint32_t first_bo_creates;
   uint32_t max_bo_allocs;
   uint32_t max_bo_creates;
   long first_minor_faults;
   long first_major_faults;
   uint64_t first_detail[HK_PERF_DETAIL_COUNT];
   uint64_t total_detail[HK_PERF_DETAIL_COUNT];
   uint64_t max_detail[HK_PERF_DETAIL_COUNT];
};

static struct {
   bool initialized;
   bool enabled;
   struct hk_shader_perf_draw_group groups[32];
} hk_shader_perf_draw_trace;

static void
hk_shader_perf_draw_dump(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(hk_shader_perf_draw_trace.groups); ++i) {
      const struct hk_shader_perf_draw_group *g =
         &hk_shader_perf_draw_trace.groups[i];
      if (!g->count)
         continue;
      fprintf(stderr,
              "HK_SHADER_PERF draw-group=%u shader=%p count=%u total_ns=%" PRIu64
              " min_ns=%" PRIu64 " max_ns=%" PRIu64
              " first_ns=%" PRIu64 " max_index=%u first_flush_ns=%" PRIu64
              " max_flush_ns=%" PRIu64 " first_ccs_ns=%" PRIu64
              " first_tess_ns=%" PRIu64 " first_geom_ns=%" PRIu64
              " first_tess_setup_ns=%" PRIu64
              " first_tess_vs_ns=%" PRIu64
              " first_tess_tcs_ns=%" PRIu64
              " first_tess_factors_ns=%" PRIu64 "\n",
              i, g->shader, g->count, g->total_ns, g->min_ns, g->max_ns,
              g->first_ns, g->max_index, g->first_flush_ns,
              g->max_flush_ns, g->first_ccs_ns, g->first_tess_ns,
              g->first_geom_ns, g->first_tess_setup_ns, g->first_tess_vs_ns,
              g->first_tess_tcs_ns, g->first_tess_factors_ns);
      fprintf(stderr,
              "HK_SHADER_PERF draw-bo-group=%u first_allocs=%u first_creates=%u "
              "first_alloc_ns=%" PRIu64 " max_allocs=%u max_creates=%u "
              "max_alloc_ns=%" PRIu64 "\n",
              i, g->first_bo_allocs, g->first_bo_creates,
              g->first_bo_alloc_ns, g->max_bo_allocs, g->max_bo_creates,
              g->max_bo_alloc_ns);
      fprintf(stderr,
              "HK_SHADER_PERF draw-fault-group=%u minor=%ld major=%ld\n", i,
              g->first_minor_faults, g->first_major_faults);
      for (unsigned p = 0; p < HK_PERF_DETAIL_COUNT; ++p) {
         fprintf(stderr,
                 "HK_SHADER_PERF draw-phase-group=%u phase=%s first_ns=%" PRIu64
                 " total_ns=%" PRIu64 " count=%u max_ns=%" PRIu64 "\n",
                 i, hk_shader_perf_detail_names[p], g->first_detail[p],
                 g->total_detail[p], g->count, g->max_detail[p]);
      }
   }
}

static inline bool
hk_shader_perf_draw_enabled(const struct hk_device *dev)
{
   static bool dump_registered;
   const bool enabled = dev->g17p_shader_perf.trace ||
                        dev->g17p_shader_perf.aggregate;
   if (unlikely(enabled && !dump_registered)) {
      atexit(hk_shader_perf_draw_dump);
      dump_registered = true;
   }
   return enabled;
}

static ATTRIBUTE_NOINLINE void
hk_shader_perf_prefault_stack(void)
{
   volatile uint8_t pages[64 * 1024];
   for (size_t i = 0; i < sizeof(pages); i += 4096)
      pages[i] = 0;
}

static ATTRIBUTE_NOINLINE void
hk_shader_perf_prefault_driver(void)
{
   FILE *maps = fopen("/proc/self/maps", "r");
   if (!maps)
      return;

   char line[1024];
   while (fgets(line, sizeof(line), maps)) {
      unsigned long start, end;
      char perms[5] = {0};
      if (!strstr(line, "libvulkan_asahi.so") ||
          sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3 ||
          perms[0] != 'r')
         continue;

      for (unsigned long p = start; p < end; p += 4096)
         (void)*(const volatile uint8_t *)p;
   }
   fclose(maps);
}

static ATTRIBUTE_NOINLINE void
hk_shader_perf_prefault_main_upload(struct hk_cmd_buffer *cmd)
{
   struct hk_uploader *uploader = &cmd->uploader.main;
   if (!uploader->map)
      return;

   const size_t first = uploader->offset & ~(size_t)4095;
   for (size_t off = first; off < HK_CMD_BO_SIZE; off += 4096) {
      volatile uint8_t *p = uploader->map + off;
      const uint8_t value = *p;
      *p = value;
   }
}

static ATTRIBUTE_NOINLINE void
hk_shader_perf_prefault_ioto(struct hk_cmd_buffer *cmd)
{
   struct hk_ioto_arena *arena = &cmd->ioto;
   if (!arena->map)
      return;

   const size_t first = arena->cursor & ~(size_t)4095;
   for (size_t off = first; off < arena->capacity; off += 4096) {
      volatile uint8_t *p = arena->map + off;
      const uint8_t value = *p;
      *p = value;
   }
}

static ATTRIBUTE_NOINLINE void
hk_shader_perf_prefault_stream(struct hk_cs *cs)
{
   if (!cs || !cs->current || !cs->end)
      return;

   uintptr_t first = (uintptr_t)cs->current & ~(uintptr_t)4095;
   for (uintptr_t addr = first; addr < (uintptr_t)cs->end; addr += 4096) {
      volatile uint8_t *p = (volatile uint8_t *)addr;
      const uint8_t value = *p;
      *p = value;
   }
}

/* The recorders below write a _Thread_local array. In a shared object its
 * address is a __tls_get_addr call, which the compiler may hoist into the
 * caller when inlined; keep every access out of line so the per-draw
 * functions that call them only under a trace flag never pay it. */
static ATTRIBUTE_NOINLINE ATTRIBUTE_COLD void
hk_shader_perf_detail_reset(void)
{
   memset(hk_shader_perf_current_detail, 0,
          sizeof(hk_shader_perf_current_detail));
}

static ATTRIBUTE_NOINLINE ATTRIBUTE_COLD void
hk_shader_perf_draw_record(const void *shader, uint64_t elapsed_ns,
                           uint32_t bo_allocs, uint32_t bo_creates,
                           uint64_t bo_alloc_ns, long minor_faults,
                           long major_faults)
{
   for (unsigned i = 0; i < ARRAY_SIZE(hk_shader_perf_draw_trace.groups); ++i) {
      struct hk_shader_perf_draw_group *g =
         &hk_shader_perf_draw_trace.groups[i];
      if (!g->shader || g->shader == shader) {
         if (!g->shader)
            g->shader = shader;
         if (!g->count) {
            g->min_ns = UINT64_MAX;
            g->first_ns = elapsed_ns;
            g->first_bo_allocs = bo_allocs;
            g->first_bo_creates = bo_creates;
            g->first_bo_alloc_ns = bo_alloc_ns;
            g->first_minor_faults = minor_faults;
            g->first_major_faults = major_faults;
            memcpy(g->first_detail, hk_shader_perf_current_detail,
                   sizeof(g->first_detail));
         }
         g->count++;
         g->total_ns += elapsed_ns;
         for (unsigned p = 0; p < HK_PERF_DETAIL_COUNT; ++p)
            g->total_detail[p] += hk_shader_perf_current_detail[p];
         g->min_ns = MIN2(g->min_ns, elapsed_ns);
         if (elapsed_ns > g->max_ns) {
            g->max_ns = elapsed_ns;
            g->max_index = g->count;
            g->max_bo_allocs = bo_allocs;
            g->max_bo_creates = bo_creates;
            g->max_bo_alloc_ns = bo_alloc_ns;
            memcpy(g->max_detail, hk_shader_perf_current_detail,
                   sizeof(g->max_detail));
         }
         return;
      }
   }
}

static ATTRIBUTE_NOINLINE ATTRIBUTE_COLD void
hk_shader_perf_flush_record(const void *shader, uint64_t elapsed_ns)
{
   hk_shader_perf_current_detail[HK_PERF_DETAIL_FLUSH] = elapsed_ns;
   for (unsigned i = 0; i < ARRAY_SIZE(hk_shader_perf_draw_trace.groups); ++i) {
      struct hk_shader_perf_draw_group *g =
         &hk_shader_perf_draw_trace.groups[i];
      if (!g->shader || g->shader == shader) {
         if (!g->shader)
            g->shader = shader;
         if (!g->first_flush_ns)
            g->first_flush_ns = elapsed_ns;
         g->max_flush_ns = MAX2(g->max_flush_ns, elapsed_ns);
         return;
      }
   }
}

static ATTRIBUTE_NOINLINE ATTRIBUTE_COLD void
hk_shader_perf_phase_record(const void *shader, unsigned phase,
                            uint64_t elapsed_ns)
{
   assert(phase <= HK_PERF_DETAIL_TESS_FACTORS - HK_PERF_DETAIL_CCS);
   hk_shader_perf_current_detail[HK_PERF_DETAIL_CCS + phase] = elapsed_ns;
   for (unsigned i = 0; i < ARRAY_SIZE(hk_shader_perf_draw_trace.groups); ++i) {
      struct hk_shader_perf_draw_group *g =
         &hk_shader_perf_draw_trace.groups[i];
      if (!g->shader || g->shader == shader) {
         if (!g->shader)
            g->shader = shader;
         if (g->count == 0) {
            if (phase == 0)
               g->first_ccs_ns = elapsed_ns;
            else if (phase == 1)
               g->first_tess_ns = elapsed_ns;
            else if (phase == 2)
               g->first_geom_ns = elapsed_ns;
            else if (phase == 3)
               g->first_tess_setup_ns = elapsed_ns;
            else if (phase == 4)
               g->first_tess_vs_ns = elapsed_ns;
            else if (phase == 5)
               g->first_tess_tcs_ns = elapsed_ns;
            else if (phase == 6)
               g->first_tess_factors_ns = elapsed_ns;
         }
         return;
      }
   }
}

static ATTRIBUTE_NOINLINE ATTRIBUTE_COLD void
hk_shader_perf_flush_detail_record(uint64_t stream_ns, uint64_t preprocess_ns,
                                   uint64_t shaders_ns,
                                   uint64_t bookkeeping_ns,
                                   uint64_t dynamic_ns)
{
   hk_shader_perf_current_detail[HK_PERF_DETAIL_STREAM] = stream_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_PREPROCESS] = preprocess_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_SHADERS] = shaders_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_BOOKKEEPING] = bookkeeping_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_DYNAMIC] = dynamic_ns;
}

static ATTRIBUTE_NOINLINE ATTRIBUTE_COLD void
hk_shader_perf_dynamic_detail_record(
   uint64_t prelink_vs_ns, uint64_t prelink_fs_ns, uint64_t upload_params_ns,
   uint64_t upload_varyings_ns, uint64_t upload_root_ns,
   uint64_t publish_program_ns, uint64_t publish_vdm_ns,
   uint64_t publish_tail_ns)
{
   hk_shader_perf_current_detail[HK_PERF_DETAIL_PRELINK_VS] = prelink_vs_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_PRELINK_FS] = prelink_fs_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_UPLOAD_PARAMS] =
      upload_params_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_UPLOAD_VARYINGS] =
      upload_varyings_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_UPLOAD_ROOT] = upload_root_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_PUBLISH_PROGRAM] =
      publish_program_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_PUBLISH_VDM] = publish_vdm_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_PUBLISH_TAIL] = publish_tail_ns;
}

static ATTRIBUTE_NOINLINE ATTRIBUTE_COLD void
hk_shader_perf_sw_tess_detail_record(
   uint64_t setup_ns, uint64_t reserve_ns, uint64_t shaders_ns,
   uint64_t count_ns, uint64_t prefix_ns, uint64_t final_ns)
{
   hk_shader_perf_current_detail[HK_PERF_DETAIL_SW_TESS_SETUP] = setup_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_SW_TESS_RESERVE] = reserve_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_SW_TESS_SHADERS] = shaders_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_SW_TESS_COUNT] = count_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_SW_TESS_PREFIX] = prefix_ns;
   hk_shader_perf_current_detail[HK_PERF_DETAIL_SW_TESS_FINAL] = final_ns;
}

static void
hk_prelink_graphics(struct hk_cmd_buffer *cmd, bool indexed,
                    enum agx_index_size index_size, bool restart,
                    bool vertex_only);
static void hk_flush_shaders(struct hk_cmd_buffer *cmd);
static bool hk_g17p_tess_is_native(const struct hk_cmd_buffer *cmd);
static struct hk_shader *hk_bound_fs(struct hk_cmd_buffer *cmd);
static uint64_t hk_heap(struct hk_cmd_buffer *cmd);
static enum mesa_prim hk_gs_in_prim(struct hk_cmd_buffer *cmd);

static enum libagx_program
hk_sw_tess_program(enum tess_primitive_mode prim, enum poly_tess_mode mode)
{
   switch (prim) {
   case TESS_PRIMITIVE_QUADS:
      return LIBAGX_TESS_QUAD(mode);
   case TESS_PRIMITIVE_TRIANGLES:
      return LIBAGX_TESS_TRI(mode);
   case TESS_PRIMITIVE_ISOLINES:
      return LIBAGX_TESS_ISOLINE(mode);
   default:
      return LIBAGX_NUM_PROGRAMS;
   }
}

static const char *
hk_g17p_tg_probe_option(void)
{
   static const char *probe;
   static bool initialized;
   if (unlikely(!initialized)) {
      probe = getenv("HK_G17P_TG_PROBE");
      initialized = true;
   }
   return probe;
}

static void hk_g17p_prepare_native_bind_tail(struct hk_cmd_buffer *cmd);
static bool hk_g17p_prepare_vdm_state_update(struct hk_cmd_buffer *cmd);
static uint64_t hk_upload_geometry_params(struct hk_cmd_buffer *cmd,
                                          struct agx_draw draw);
static void hk_init_tess_params(struct hk_cmd_buffer *cmd,
                                struct poly_tess_params *out);
static void hk_flush_dynamic_state(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint32_t draw_id,
   struct agx_draw draw, bool ordinary_draw_path,
   bool *g17p_native_bind_tail_emitted_out, bool prepare_only);

void
hk_cmd_prepare_bound_graphics(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   memset(&gfx->g17p_precomp, 0, sizeof(gfx->g17p_precomp));
   gfx->g17p_sw_tess_batch = (struct hk_g17p_prepared_sw_tess_batch){0};
   gfx->prepared_vertex_params_filled = false;
   gfx->prepared_tess_params_filled = false;
   gfx->prepared_geometry_params_filled = false;
   gfx->prepared_vdm_state_update_valid = false;

   if (!cmd->current_cs.gfx ||
       !gfx->shaders[MESA_SHADER_VERTEX] ||
       !gfx->shaders[MESA_SHADER_FRAGMENT] ||
       vk_command_buffer_has_error(&cmd->vk))
      return;

   /* Shaders, dynamic state and attachment formats are now all known. Build
    * the exact non-indexed immutable variants outside vkCmdDraw*. An indexed
    * draw gets its width-specific VS variant when its index buffer is bound;
    * the draw path still performs the authoritative lookup. */
   hk_prelink_graphics(cmd, false, AGX_INDEX_SIZE_U16,
                       cmd->vk.dynamic_graphics_state.ia.primitive_restart_enable,
                       false);

   /* Preparation must consume the same complete shader/root state as the
    * authoritative draw path.  In particular, transformed pipelines use the
    * post-tessellation or geometry raster variant for TES/GS -> FS UVS
    * assignment.  Uploading either native program before this flush sees the
    * previous (often zero) linked_varyings table and can both refuse a legal
    * program and snapshot stale root indices.  Keep dirtiness intact: the
    * draw remains authoritative and merely exact-matches these prepared
    * objects when state has not changed. */
   hk_flush_shaders(cmd);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   if (!gfx->shaders[MESA_SHADER_GEOMETRY] &&
       !gfx->shaders[MESA_SHADER_TESS_EVAL]) {
      if (!hk_cmd_buffer_prepare_graphics_root(cmd))
         return;
      hk_g17p_prepare_native_bind_tail(cmd);
      return;
   }

   /* A tessellation/geometry draw necessarily records compute work before the
    * graphics stream. Allocate both ordered streams while binding state, not
    * in the first draw, preserving the established list order. */
   (void)hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx, true);
   struct hk_cs *post_gfx =
      hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.post_gfx, true);
   if (post_gfx &&
       !util_dynarray_ensure_cap(&post_gfx->imm_writes,
                                 sizeof(struct libagx_imm_write))) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }

   if (dev->dev.chip != AGX_CHIP_G17P)
      return;

   if (gfx->shaders[MESA_SHADER_TESS_CTRL] &&
       gfx->shaders[MESA_SHADER_TESS_EVAL]) {
      struct hk_shader *tes =
         hk_any_variant(gfx->shaders[MESA_SHADER_TESS_EVAL]);
      struct hk_shader *tcs =
         hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);
      gfx->tess.info =
         hk_tess_info_merge(tes->info.tess.info, tcs->info.tess.info);
      gfx->tess.prim = gfx->tess.info.points
         ? MESA_PRIM_POINTS
         : gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES
              ? MESA_PRIM_LINES
              : MESA_PRIM_TRIANGLES;
   }

   if (!gfx->prepared_vertex_params.gpu) {
      gfx->prepared_vertex_params =
         hk_pool_alloc(cmd, sizeof(struct poly_vertex_params), 8);
      if (!gfx->prepared_vertex_params.gpu)
         return;
   }
   gfx->descriptors.root.draw.vertex_params =
      gfx->prepared_vertex_params.gpu;
   gfx->descriptors.root.draw.vertex_outputs =
      hk_bound_sw_vs(gfx)->b.info.outputs;

   /* The transformed vertex-parameter block is mostly shader-owned state.
    * Initialize that immutable base while binding the shaders, leaving the
    * actual draw to patch only counts, index metadata and the output address
    * immediately before publication. */
   const uint32_t vertex_wg_size[3] = {64, 1, 1};
   poly_vertex_params_init(gfx->prepared_vertex_params.cpu,
                           gfx->descriptors.root.draw.vertex_outputs,
                           vertex_wg_size);
   gfx->prepared_vertex_params_filled = true;

   if (gfx->shaders[MESA_SHADER_TESS_EVAL] &&
       !gfx->prepared_tess_params.gpu) {
      gfx->prepared_tess_params =
         hk_pool_alloc(cmd, sizeof(struct poly_tess_params), 4);
      if (!gfx->prepared_tess_params.gpu)
         return;
      gfx->descriptors.root.draw.tess_params =
         gfx->prepared_tess_params.gpu;
   }
   if (gfx->shaders[MESA_SHADER_TESS_EVAL]) {
      hk_init_tess_params(cmd, gfx->prepared_tess_params.cpu);
      gfx->prepared_tess_params_filled = true;
   }

   /* Software tessellation always hands geometry a five-u32 indexed indirect
    * draw. Reserve that exact record while bound state is complete, so the
    * GS-setup argument block can name a stable command-buffer-owned address.
    * The record is still filled by the tessellator and is unreachable until
    * the matching draw publishes its launches. */
   if (gfx->shaders[MESA_SHADER_TESS_EVAL] &&
       !hk_g17p_tess_is_native(cmd) &&
       !gfx->prepared_tess_out_draws.gpu) {
      gfx->prepared_tess_out_draws =
         hk_pool_alloc(cmd, 5 * sizeof(uint32_t), 4);
      if (!gfx->prepared_tess_out_draws.gpu)
         return;
      gfx->tess.out_draws = gfx->prepared_tess_out_draws.gpu;
   }

   if (gfx->shaders[MESA_SHADER_GEOMETRY] &&
       !gfx->prepared_geometry_params.gpu) {
      gfx->prepared_geometry_params =
         hk_pool_alloc(cmd, sizeof(struct poly_geometry_params), 8);
      if (!gfx->prepared_geometry_params.gpu)
         return;
      gfx->descriptors.root.draw.geometry_params =
         gfx->prepared_geometry_params.gpu;
   }

   /* With tessellation feeding geometry, the GS parameter block is indirect
    * and therefore independent of the API draw counts.  Build that immutable
    * block now when transform feedback and its queries are inactive.  Keep
    * ownership of the reserved GPU record so the real draw only publishes it. */
   if (gfx->shaders[MESA_SHADER_TESS_EVAL] &&
       gfx->shaders[MESA_SHADER_GEOMETRY] && !gfx->xfb_enabled &&
       gfx->prepared_geometry_params.gpu) {
      bool no_xfb_queries = true;
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb_query); ++i)
         no_xfb_queries &= gfx->xfb_query[i] == 0;

      if (no_xfb_queries) {
         const struct agx_ptr reserved = gfx->prepared_geometry_params;
         const uint64_t params = hk_upload_geometry_params(
            cmd, (struct agx_draw){.b = agx_3d(1, 1, 1)});
         if (params == reserved.gpu &&
             !vk_command_buffer_has_error(&cmd->vk)) {
            gfx->prepared_geometry_params = reserved;
            gfx->prepared_geometry_params_filled = true;
         }
      }
   }

   struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);
   struct hk_linked_shader *sw_vs_linked =
      sw_vs->only_linked ?: gfx->linked[MESA_SHADER_VERTEX];

   if (sw_vs_linked->b.uses_base_param && !gfx->prepared_draw_params.gpu) {
      gfx->prepared_draw_params = hk_pool_alloc(cmd, 8, 4);
      if (!gfx->prepared_draw_params.gpu)
         return;
      gfx->draw_params = gfx->prepared_draw_params.gpu;
   }
   if (sw_vs->b.info.uses_draw_id && !gfx->prepared_draw_id.gpu) {
      gfx->prepared_draw_id = hk_pool_alloc(cmd, 4, 4);
      if (!gfx->prepared_draw_id.gpu)
         return;
      gfx->draw_id_ptr = gfx->prepared_draw_id.gpu;
   }

   if (!hk_cmd_buffer_prepare_graphics_root(cmd))
      return;

   /* Complete the draw-independent root and linkage while BeginRendering has
    * the exact shader/render/dynamic state.  This preparation advances no
    * stream and clears no Vulkan dirtiness; the draw remains authoritative.
    * Mutable tessellation/geometry records already have stable addresses and
    * are filled only by the real draw. */
   bool prepared_tail_emitted = false;
   hk_flush_dynamic_state(cmd, cmd->current_cs.gfx, 0,
                          (struct agx_draw){.b = agx_3d(1, 1, 1)}, true,
                          &prepared_tail_emitted, true);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_linked_shader *linked_hw_vs =
      hw_vs == sw_vs ? gfx->linked[MESA_SHADER_VERTEX] : hw_vs->only_linked;
   if (!gfx->prepared_vertex_usc) {
      gfx->prepared_vertex_usc =
         hk_prepare_usc_words(cmd, hw_vs, linked_hw_vs,
                              &gfx->prepared_vertex_compact_root);
      if (gfx->prepared_vertex_usc) {
         gfx->prepared_vertex_shader = hw_vs;
         gfx->prepared_vertex_linked = linked_hw_vs;
         gfx->prepared_vertex_root = gfx->root;
         gfx->prepared_vertex_draw_params = gfx->draw_params;
         gfx->prepared_vertex_draw_id_ptr = gfx->draw_id_ptr;
      }
   }

   struct hk_shader *fs = hk_bound_fs(cmd);
   if (!gfx->prepared_fragment_usc)
      gfx->prepared_fragment_usc =
         hk_upload_usc_words(cmd, fs, gfx->linked[MESA_SHADER_FRAGMENT]);

   hk_g17p_prepare_native_bind_tail(cmd);
   (void)hk_g17p_prepare_vdm_state_update(cmd);

   hk_g17p_prewarm_linked_compute_program(cmd, sw_vs, sw_vs_linked, true);
   hk_g17p_prepare_graphics_dispatch(
      cmd, sw_vs, sw_vs_linked, 0, agx_workgroup(1, 1, 1),
      &gfx->g17p_dispatch.tess_vs);

   struct hk_api_shader *tcs_api = gfx->shaders[MESA_SHADER_TESS_CTRL];
   if (tcs_api) {
      struct hk_shader *tcs = hk_only_variant(tcs_api);
      hk_g17p_prewarm_linked_compute_program(cmd, tcs, tcs->only_linked, true);
      hk_g17p_prepare_graphics_dispatch(
         cmd, tcs, tcs->only_linked, 0, agx_workgroup(1, 1, 1),
         &gfx->g17p_dispatch.tess_tcs);
   }

   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   if (gs) {
      struct hk_shader *pre_gs_vs = hk_bound_sw_vs_before_gs(gfx);
      struct hk_linked_shader *pre_gs_vs_linked =
         pre_gs_vs->only_linked ?: gfx->linked[MESA_SHADER_VERTEX];
      hk_g17p_prewarm_linked_compute_program(cmd, pre_gs_vs,
                                              pre_gs_vs_linked, true);
      const uint64_t vertex_grid = gfx->prepared_vertex_params.gpu +
         offsetof(struct poly_vertex_params, grid);
      const uint64_t geometry_grid = gfx->prepared_geometry_params.gpu +
         offsetof(struct poly_geometry_params, grid);
      hk_g17p_prepare_graphics_dispatch(
         cmd, pre_gs_vs, pre_gs_vs_linked, vertex_grid,
         agx_workgroup(64, 1, 1), &gfx->g17p_dispatch.gs_vs);

      struct hk_shader *variants[] = {
         &gs->variants[HK_GS_VARIANT_RAST], hk_main_gs_variant(gs),
         hk_count_gs_variant(gs), hk_pre_gs_variant(gs),
      };
      for (unsigned i = 0; i < ARRAY_SIZE(variants); ++i) {
         hk_g17p_prewarm_linked_compute_program(
            cmd, variants[i], variants[i]->only_linked, true);
      }
      hk_g17p_prepare_graphics_dispatch(
         cmd, hk_main_gs_variant(gs), hk_main_gs_variant(gs)->only_linked,
         geometry_grid,
         agx_workgroup(64, 1, 1), &gfx->g17p_dispatch.gs_main);
      hk_g17p_prepare_graphics_dispatch(
         cmd, hk_count_gs_variant(gs), hk_count_gs_variant(gs)->only_linked,
         geometry_grid, agx_workgroup(64, 1, 1),
         &gfx->g17p_dispatch.gs_count);
      hk_g17p_prepare_graphics_dispatch(
         cmd, hk_pre_gs_variant(gs), hk_pre_gs_variant(gs)->only_linked, 0,
         agx_workgroup(1, 1, 1),
         &gfx->g17p_dispatch.gs_pre);

      /* A software-tessellation draw always becomes this exact indexed
       * indirect draw before entering GS setup. Prepare the full 72-byte
       * libagx argument identity now, but publish nothing until the actual
       * draw exact-matches it. Any state change takes the ordinary fallback. */
      if (gfx->shaders[MESA_SHADER_TESS_EVAL] &&
          !hk_g17p_tess_is_native(cmd) &&
          gfx->prepared_tess_out_draws.gpu) {
         const uint64_t heap = hk_heap(cmd);
         const struct agx_draw prepared_draw = agx_draw_indexed_indirect(
            gfx->prepared_tess_out_draws.gpu, dev->heap->va->addr,
            dev->heap->size, AGX_INDEX_SIZE_U32, false);
         struct hk_shader *count = hk_count_gs_variant(gs);
         const struct libagx_gs_setup_indirect_args gsi = {
            .heap = heap,
            .index_buffer = prepared_draw.index_buffer,
            .draw = prepared_draw.b.ptr,
            .vp = gfx->descriptors.root.draw.vertex_params,
            .p = gfx->descriptors.root.draw.geometry_params,
            .vs_outputs = pre_gs_vs->b.info.outputs,
            .index_size_B = agx_index_size_to_B(prepared_draw.index_size),
            .index_buffer_range_el = agx_draw_index_range_el(prepared_draw),
            .prim = hk_gs_in_prim(cmd),
            .is_prefix_summing = count->info.gs.prefix_sum,
            .max_indices = count->info.gs.max_indices,
            .shape = count->info.gs.shape,
         };
         hk_g17p_prepare_precomp(cmd, LIBAGX_GS_SETUP_INDIRECT, &gsi,
                                 sizeof(gsi),
                                 &gfx->g17p_precomp.gs_setup);
      }
   }

   hk_g17p_prewarm_indirect_helpers(cmd);
   (void)hk_g17p_strong_dispatch_flush_enabled();
   (void)hk_g17p_tg_probe_option();

   if (dev->g17p_shader_perf.trace)
      fprintf(stderr,
              "HK_SHADER_PERF prepare-bound tess=%u native=%u gs=%u state=%" PRIx64 "\n",
              gfx->shaders[MESA_SHADER_TESS_EVAL] != NULL,
              hk_g17p_tess_is_native(cmd), gs != NULL,
              gfx->descriptors.root.draw.tess_params);

   /* Software tessellation's three libagx launches have immutable argument
    * bytes once rendering state is fixed. Publish those exact command-local
    * loaders now; the first draw then only appends their launch records. */
   if (gfx->shaders[MESA_SHADER_TESS_EVAL] &&
       !hk_g17p_tess_is_native(cmd)) {
      const uint64_t state = gfx->descriptors.root.draw.tess_params;
      const enum libagx_program count_program = hk_sw_tess_program(
         gfx->tess.info.mode, POLY_TESS_MODE_COUNT);
      const enum libagx_program final_program = hk_sw_tess_program(
         gfx->tess.info.mode, POLY_TESS_MODE_WITH_COUNTS);
      if (count_program == LIBAGX_NUM_PROGRAMS ||
          final_program == LIBAGX_NUM_PROGRAMS)
         return;

      hk_g17p_prepare_precomp(cmd, count_program, &state, sizeof(state),
                              &gfx->g17p_precomp.tess_count);

      const uint64_t c_prims = gs ? 0 : hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT);
      const uint64_t c_invs = gs ? 0 : hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);
      const struct libagx_prefix_sum_tess_args prefix = {
         .p = state,
         .c_prims = c_prims,
         .c_invs = c_invs,
      };
      hk_g17p_prepare_precomp(
         cmd, LIBAGX_PREFIX_SUM_TESS(c_prims || c_invs), &prefix,
         sizeof(prefix), &gfx->g17p_precomp.tess_prefix);
      hk_g17p_prepare_precomp(cmd, final_program, &state, sizeof(state),
                              &gfx->g17p_precomp.tess_final);

      const bool complete = gfx->g17p_dispatch.tess_vs.valid &&
         gfx->g17p_dispatch.tess_tcs.valid &&
         gfx->g17p_precomp.tess_count.valid &&
         gfx->g17p_precomp.tess_prefix.valid &&
         gfx->g17p_precomp.tess_final.valid;
      if (complete) {
         const void *templates[HK_G17P_SW_TESS_BATCH_LAUNCHES] = {
            gfx->g17p_dispatch.tess_vs.direct_template,
            gfx->g17p_dispatch.tess_tcs.direct_template,
            gfx->g17p_precomp.tess_count.direct_template,
            gfx->g17p_precomp.tess_prefix.direct_template,
            gfx->g17p_precomp.tess_final.direct_template,
         };
         for (unsigned i = 0; i < ARRAY_SIZE(templates); ++i)
            memcpy(gfx->g17p_sw_tess_batch.bytes[i], templates[i],
                   AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE);
         gfx->g17p_sw_tess_batch.vs = hk_bound_sw_vs(gfx);
         gfx->g17p_sw_tess_batch.tcs =
            hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);
         gfx->g17p_sw_tess_batch.dynamic_generation =
            cmd->vk.dynamic_graphics_state.generation;
         gfx->g17p_sw_tess_batch.descriptor_generation =
            gfx->descriptors.generation;
         gfx->g17p_sw_tess_batch.root = gfx->root;
         gfx->g17p_sw_tess_batch.draw_params = gfx->draw_params;
         gfx->g17p_sw_tess_batch.draw_id_ptr = gfx->draw_id_ptr;
         gfx->g17p_sw_tess_batch.state = state;
         gfx->g17p_sw_tess_batch.c_prims = c_prims;
         gfx->g17p_sw_tess_batch.c_invs = c_invs;
         gfx->g17p_sw_tess_batch.count_program = count_program;
         gfx->g17p_sw_tess_batch.final_program = final_program;
         if (gs) {
            struct hk_shader *main = hk_main_gs_variant(gs);
            gfx->g17p_sw_tess_batch.gs_vs =
               hk_bound_sw_vs_before_gs(gfx);
            gfx->g17p_sw_tess_batch.gs_main = main;
            gfx->g17p_sw_tess_batch.includes_gs =
               gfx->g17p_precomp.gs_setup.valid &&
               gfx->g17p_dispatch.gs_vs.indirect_local_valid &&
               gfx->g17p_dispatch.gs_main.indirect_local_valid &&
               !main->info.gs.xfb;
         }
         gfx->g17p_sw_tess_batch.valid = true;
      }

   }

}

/* G17P batches draws in one graphics stream. Explicit split policies and
 * depth caps are diagnostic controls only. The historical depth experiments
 * and their intermittent faults are retained in commit fb8fd6b881d; they do
 * not establish a hardware limit of one draw or publication per stream.
 */
enum hk_g17p_draw_split {
   /* Every draw after the first starts a new stream. */
   HK_G17P_DRAW_SPLIT_EVERY,
   /* Only a draw that republishes render state starts a new stream. */
   HK_G17P_DRAW_SPLIT_REPUBLISH,
   /* Batch draws until a real render dependency or capacity boundary. */
   HK_G17P_DRAW_SPLIT_NEVER,
};

static enum hk_g17p_draw_split
hk_g17p_draw_split_policy(const struct hk_device *dev)
{
   static int policy = -1;

   if (dev->dev.chip != AGX_CHIP_G17P)
      return HK_G17P_DRAW_SPLIT_NEVER;

   if (policy < 0) {
      const char *spelled = getenv("HK_G17P_DRAW_SPLIT");

      if (spelled && !strcmp(spelled, "every"))
         policy = HK_G17P_DRAW_SPLIT_EVERY;
      else if (spelled && !strcmp(spelled, "republish"))
         policy = HK_G17P_DRAW_SPLIT_REPUBLISH;
      else if (spelled && !strcmp(spelled, "never"))
         policy = HK_G17P_DRAW_SPLIT_NEVER;
      else if (debug_get_bool_option("HK_G17P_CONCAT_DRAWS", false))
         policy = HK_G17P_DRAW_SPLIT_NEVER;
      else
         policy = HK_G17P_DRAW_SPLIT_NEVER;

      /* The old opt-in spelling, kept working so existing A/B scripts that
       * asked for the unconditional split still get it. */
      if (debug_get_bool_option("HK_G17P_SPLIT_DIRECT_DRAWS", false))
         policy = HK_G17P_DRAW_SPLIT_EVERY;
   }

   return (enum hk_g17p_draw_split)policy;
}

/*
 * How many render-state publications one stream may carry.
 *
 * Diagnostic limit on state publications, independently of draw bodies.
 * Zero is the default and means no artificial cap.
 */
static unsigned
hk_g17p_state_publications_per_stream(const struct hk_device *dev)
{
   static int cap = -1;

   if (dev->dev.chip != AGX_CHIP_G17P)
      return 0;

   if (cap < 0)
      cap = debug_get_num_option("HK_G17P_STATE_PUBLICATIONS_PER_STREAM", 0);

   return cap > 0 ? (unsigned)cap : 0;
}

/*
 * Hard cap on draw *bodies* per graphics control stream, independent of the
 * publication cap above. Zero means no artificial cap. Historical depth
 * experiments are documented in fb8fd6b881d, including intermittent faults.
 */
static unsigned
hk_g17p_draws_per_stream(const struct hk_device *dev)
{
   static int cap = -1;

   if (dev->dev.chip != AGX_CHIP_G17P)
      return 0;

   if (cap < 0)
      cap = debug_get_num_option("HK_G17P_DRAWS_PER_STREAM", 0);

   return cap > 0 ? (unsigned)cap : 0;
}

/*
 * Number of VDM state contexts to rotate a stream's state publications
 * through.  1 (the default) pins every publication to context 0, which is what
 * m1n1's single-draw witness recorded and all this driver has ever emitted.
 * The hardware field is three bits, so 8 is the ceiling.
 */
static unsigned
hk_g17p_vdm_contexts(const struct hk_device *dev)
{
   static int contexts = -1;

   if (dev->dev.chip != AGX_CHIP_G17P)
      return 1;

   if (contexts < 0) {
      contexts = debug_get_num_option("HK_G17P_VDM_CONTEXTS", 0);

      /* The original boolean spelling asked for the full rotation. */
      if (contexts <= 0)
         contexts = debug_get_bool_option("HK_G17P_CYCLE_VDM_CONTEXT", false)
                        ? 8
                        : 1;
   }

   return CLAMP(contexts, 1, 8);
}

static bool
hk_g17p_cycle_vdm_context_enabled(const struct hk_device *dev)
{
   return hk_g17p_vdm_contexts(dev) > 1;
}

/* CTS coverage of indirect draws is pretty bad, so it's helpful to be able to
 * get some extra smoke testing.
 */
#define HK_TEST_INDIRECTS (0)

UNUSED static inline void
print_draw(struct agx_draw d, FILE *fp)
{
   if (agx_is_indirect(d.b))
      fprintf(fp, "indirect (buffer %" PRIx64 "):", d.b.ptr);
   else
      fprintf(fp, "direct (%ux%u):", d.b.count[0], d.b.count[1]);

   if (d.index_size)
      fprintf(fp, " index_size=%u", agx_index_size_to_B(d.index_size));
   else
      fprintf(fp, " non-indexed");

   if (d.restart)
      fprintf(fp, " restart");

   if (d.index_bias)
      fprintf(fp, " index_bias=%u", d.index_bias);

   if (d.start)
      fprintf(fp, " start=%u", d.start);

   if (d.start_instance)
      fprintf(fp, " start_instance=%u", d.start_instance);

   fprintf(fp, "\n");
}

static bool
hk_rast_discard(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* A non-zero rasterization stream acts as a rasterizer discard unless
    * there's a multistream geometry shader bound.
    */
   if (dyn->rs.rasterization_stream != 0) {
      struct hk_api_shader *gs = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
      return !gs || !gs->variants[HK_GS_VARIANT_COUNT].info.gs.multistream;
   }

   return dyn->rs.rasterizer_discard_enable;
}

static bool
hk_g17p_xfb_vertex_side_effects(struct hk_cmd_buffer *cmd,
                                struct hk_shader *hw_vs)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

   return hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G17P &&
          gfx->xfb_enabled && gs && gs->is_passthrough &&
          hw_vs == &gs->variants[HK_GS_VARIANT_RAST];
}

static void
hk_cmd_buffer_dirty_render_pass(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* These depend on color attachment count */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_WRITE_MASKS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ADVANCED);

   /* These depend on the depth/stencil format */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_BOUNDS_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_DEPTH_BIAS_FACTORS);

   /* This may depend on render targets for ESO */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);

   /* This may depend on render targets */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP);
}

void
hk_cmd_buffer_begin_graphics(struct hk_cmd_buffer *cmd,
                             const VkCommandBufferBeginInfo *pBeginInfo)
{
   if (cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
       (pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)) {
      char gcbiar_data[VK_GCBIARR_DATA_SIZE(HK_MAX_RTS)];
      const VkRenderingInfo *resume_info =
         vk_get_command_buffer_inheritance_as_rendering_resume(
            cmd->vk.level, pBeginInfo, gcbiar_data);
      if (resume_info) {
         hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), resume_info);
      } else {
         const VkCommandBufferInheritanceRenderingInfo *inheritance_info =
            vk_get_command_buffer_inheritance_rendering_info(cmd->vk.level,
                                                             pBeginInfo);
         assert(inheritance_info);

         struct hk_rendering_state *render = &cmd->state.gfx.render;
         render->flags = inheritance_info->flags;
         render->area = (VkRect2D){};
         render->layer_count = 0;
         render->view_mask = inheritance_info->viewMask;
         render->tilebuffer.nr_samples = inheritance_info->rasterizationSamples;

         render->color_att_count = inheritance_info->colorAttachmentCount;
         for (uint32_t i = 0; i < render->color_att_count; i++) {
            render->color_att[i].vk_format =
               inheritance_info->pColorAttachmentFormats[i];
         }
         render->depth_att.vk_format = inheritance_info->depthAttachmentFormat;
         render->stencil_att.vk_format =
            inheritance_info->stencilAttachmentFormat;

         const VkRenderingAttachmentLocationInfoKHR att_loc_info_default = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
            .colorAttachmentCount = inheritance_info->colorAttachmentCount,
         };
         const VkRenderingAttachmentLocationInfoKHR *att_loc_info =
            vk_get_command_buffer_rendering_attachment_location_info(
               cmd->vk.level, pBeginInfo);
         if (att_loc_info == NULL)
            att_loc_info = &att_loc_info_default;

         vk_cmd_set_rendering_attachment_locations(&cmd->vk, att_loc_info);

         hk_cmd_buffer_dirty_render_pass(cmd);
      }
   }

   hk_cmd_buffer_dirty_all(cmd);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   cmd->state.gfx.descriptors.root.draw.view_index = 0;
   cmd->state.gfx.descriptors.root_dirty = true;
}

void
hk_cmd_invalidate_graphics_state(struct hk_cmd_buffer *cmd)
{
   /* From the Vulkan 1.3.275 spec:
    *
    *    "...There is one exception to this rule - if the primary command
    *    buffer is inside a render pass instance, then the render pass and
    *    subpass state is not disturbed by executing secondary command
    *    buffers."
    *
    * We need to reset everything EXCEPT the render pass state.
    */
   struct hk_rendering_state render_save = cmd->state.gfx.render;
   struct hk_push_descriptor_set *push[HK_MAX_SETS];
   memcpy(push, cmd->state.gfx.descriptors.push, sizeof(push));
   memset(&cmd->state.gfx, 0, sizeof(cmd->state.gfx));
   cmd->state.gfx.render = render_save;
   memcpy(cmd->state.gfx.descriptors.push, push, sizeof(push));

   /* Dirty after clearing the cached state.  Doing this before memset loses
    * every dirty bit and lets a post-secondary rebind be elided. */
   hk_cmd_buffer_dirty_all(cmd);
}

static void
hk_attachment_init(struct hk_attachment *att,
                   const VkRenderingAttachmentInfo *info)
{
   if (info == NULL || info->imageView == VK_NULL_HANDLE) {
      *att = (struct hk_attachment){
         .iview = NULL,
      };
      return;
   }

   VK_FROM_HANDLE(hk_image_view, iview, info->imageView);
   *att = (struct hk_attachment){
      .vk_format = iview->vk.format,
      .iview = iview,
   };

   if (info->resolveMode != VK_RESOLVE_MODE_NONE) {
      VK_FROM_HANDLE(hk_image_view, res_iview, info->resolveImageView);
      att->resolve_mode = info->resolveMode;
      att->resolve_iview = res_iview;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetRenderingAreaGranularityKHR(
   VkDevice device, const VkRenderingAreaInfoKHR *pRenderingAreaInfo,
   VkExtent2D *pGranularity)
{
   *pGranularity = (VkExtent2D){.width = 1, .height = 1};
}

static bool
is_attachment_stored(const VkRenderingAttachmentInfo *att)
{
   /* When resolving, we store the intermediate multisampled image as the
    * resolve is a separate control stream. This could be optimized.
    */
   return att->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
          att->resolveMode != VK_RESOLVE_MODE_NONE;
}

/* LDSHDR's fetch span saturates from 321 bytes of text on, so a longer
 * program, such as a multi-target mixed-format reload, loads exactly like the
 * main fragment shaders of any size. */
static bool
hk_g17p_bg_eot_shader_supported(const struct agx_bg_eot_shader *shader)
{
   return shader && shader->bo && shader->bo->va && shader->ptr != 0 &&
          shader->info.binary_size > 0 &&
          shader->info.main_offset < shader->info.binary_size &&
          shader->info.main_size <=
             shader->info.binary_size - shader->info.main_offset &&
          !shader->info.has_preamble && shader->info.rodata.size_16 == 0 &&
          shader->info.nr_gprs > 0;
}

/* Build the BG or EOT program of an AGX_G17P_MRT_MIXED layout. Only the
 * targets the operation touches are numbered, each keeping its framebuffer
 * tile offset. Reloads are numbered before clears: a reload reads its texture
 * and sampler pointers from uniform halves 0..7, which the clear value of a
 * target zero would otherwise overlap at halves 4..11. */
static struct hk_bg_eot
hk_build_g17p_mixed_bg_eot(struct hk_cmd_buffer *cmd,
                           const struct agx_bg_eot_key *framebuffer_key,
                           const VkRenderingInfo *info, bool store)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct agx_bg_eot_key key = *framebuffer_key;
   memset(key.tib.logical_format, 0, sizeof(key.tib.logical_format));
   memset(key.tib.spilled, 0, sizeof(key.tib.spilled));
   memset(key.tib._offset_B, 0, sizeof(key.tib._offset_B));
   memset(key.op, 0, sizeof(key.op));
   key.mixed_mrt = true;
   /* Multisample targets are stored unresolved: a multi-target pass
    * resolves afterwards, from the multisample images. */
   key.multisample_store = store && key.tib.nr_samples > 1;

   const VkRenderingAttachmentInfo *atts[HK_MAX_RTS];
   struct hk_image_view *views[HK_MAX_RTS];
   static const enum agx_bg_eot_op order[] = {AGX_BG_LOAD, AGX_BG_CLEAR,
                                              AGX_EOT_STORE};
   unsigned count = 0, reloads = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(order); ++i) {
      for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
         if (framebuffer_key->op[rt] != order[i])
            continue;

         struct hk_image_view *view = cmd->state.gfx.render.color_att[rt].iview;
         if (count == ARRAY_SIZE(views) || rt >= info->colorAttachmentCount ||
             !view ||
             view->vk.image->samples != MAX2(key.tib.nr_samples, 1) ||
             info->pColorAttachments[rt].imageView == VK_NULL_HANDLE) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FORMAT_NOT_SUPPORTED);
            return (struct hk_bg_eot){0};
         }

         key.tib.logical_format[count] = framebuffer_key->tib.logical_format[rt];
         key.tib._offset_B[count] = framebuffer_key->tib._offset_B[rt];
         key.op[count] = order[i];
         atts[count] = &info->pColorAttachments[rt];
         views[count++] = view;
         reloads += order[i] == AGX_BG_LOAD;
      }
   }

   const unsigned descriptor_generation =
      agx_chip_descriptor_generation(dev->dev.chip);
   if (hk_pbe_descriptor_size(descriptor_generation) !=
          AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE ||
       hk_texture_descriptor_size(descriptor_generation) !=
          AGX_TEXTURE_GEN9_LENGTH ||
       agx_chip_sampler_heap_element_size(dev->dev.chip) != 32) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }

   const bool clears = !store && count > reloads;
   if (clears)
      key.reserved_preamble = 4 + 8 * count;

   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&dev->bg_eot, &key);
   if (!hk_g17p_bg_eot_shader_supported(shader) ||
       (!store && (shader->info.push_count != key.reserved_preamble ||
                   shader->info.uses_txf != (reloads > 0)))) {
      mesa_loge("G17P %s shader declined: supported=%u push=%u want=%u "
                "uses_txf=%u want=%u count=%u reloads=%u samples=%u "
                "layered=%u sample_size=%uB",
                store ? "EOT" : "background",
                hk_g17p_bg_eot_shader_supported(shader),
                shader ? shader->info.push_count : 0,
                key.reserved_preamble,
                shader ? shader->info.uses_txf : 0, reloads > 0, count,
                reloads, key.tib.nr_samples, key.tib.layered,
                key.tib.sample_size_B);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }

   const enum agx3_g17p_rt0_bg_op bg_op =
      count == 0         ? AGX3_G17P_RT0_BG_NONE
      : reloads == count ? AGX3_G17P_RT0_BG_RELOAD
      : reloads          ? AGX3_G17P_RT0_BG_CLEAR_RELOAD
                         : AGX3_G17P_RT0_BG_CLEAR;
   const size_t program_capacity =
      count == 0  ? AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE
      : store     ? (count > 4 ? AGX_IOTO_G17P_OCTA_EOT_PROGRAM_MAX_SIZE
                               : AGX_IOTO_G17P_QUAD_EOT_PROGRAM_MAX_SIZE)
      : count > 4 ? AGX_IOTO_G17P_OCTA_BACKGROUND_PROGRAM_MAX_SIZE
      : bg_op == AGX3_G17P_RT0_BG_CLEAR_RELOAD
         ? AGX_IOTO_G17P_CLEAR_RELOAD_PROGRAM_MAX_SIZE
         : AGX_IOTO_G17P_RT0_RELOAD_PROGRAM_MAX_SIZE;
   struct hk_ioto_reservation reservation;
   if (!hk_cmd_reserve_ioto_program(cmd, program_capacity,
                                    "G17P mixed MRT Ioto program",
                                    &reservation))
      return (struct hk_bg_eot){0};

   const struct agx_ioto_g17p_imageblock imageblock = {
      .sample_size_B = key.tib.sample_size_B,
      .nr_samples = key.tib.nr_samples,
      .tile_size_px = key.tib.tile_size,
   };
   const uint8_t *program_bytes;
   size_t program_length;
   uint64_t resource_spec;
   struct hk_agx3_g17p_graphics_program empty_program;
   struct agx3_g17p_rt0_eot_program eot_program;
   struct agx3_g17p_rt0_bg_program bg_program;
   if (count == 0) {
      const struct hk_agx3_g17p_graphics_program_desc desc = {
         .program_va = reservation.va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .compiler_d_valid = shader->info.agx3_staging_known,
         .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
         .fragment = true,
         .imageblock = imageblock,
      };
      if (!hk_agx3_g17p_basic_graphics_program_generate(
             dev->dev.chip, &desc, &empty_program) ||
          !agx3_g17p_rt0_bg_resource_spec(dev->dev.chip, bg_op,
                                          &resource_spec)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
      program_bytes = empty_program.bytes;
      program_length = empty_program.length;
   } else if (store) {
      static_assert(sizeof(struct hk_native_pbe_descriptor) ==
                       AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE,
                    "native PBE table stride");
      /* A layered pass stores through the 2D-ARRAY block image store, so its
       * targets need the array-shaped PBE descriptor. The plain BG/EOT
       * descriptor is deliberately flattened to one layer -- pack_pbe() forces
       * layers = 1 for HK_DESC_USAGE_BG_EOT -- and handing it to an array
       * store writes the wrong surface. This mirrors the single-target path.
       */
      struct hk_native_pbe_descriptor pbe_arguments[HK_MAX_RTS] = {0};
      for (unsigned rt = 0; rt < count; ++rt) {
         if (!(key.tib.layered ? views[rt]->planes[0].layered_eot_pbe_valid
                               : views[rt]->planes[0].eot_pbe_valid)) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FORMAT_NOT_SUPPORTED);
            return (struct hk_bg_eot){0};
         }
         pbe_arguments[rt] = key.tib.layered ? views[rt]->planes[0].layered_eot
                                             : views[rt]->planes[0].eot;
      }

      const uint64_t pbe_argument_va = hk_pool_upload(
         cmd, pbe_arguments, count * AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE, 32);
      if (!pbe_argument_va)
         return (struct hk_bg_eot){0};

      const struct agx3_g17p_rt0_eot_program_desc desc = {
         .program_va = reservation.va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .pbe_argument_va = pbe_argument_va,
         .imageblock_offset_B = key.tib._offset_B[0],
         .attachment_count = count,
         .imageblock_offset1_B = key.tib._offset_B[1],
         .imageblock_offset2_B = key.tib._offset_B[2],
         .imageblock_offset3_B = key.tib._offset_B[3],
         .imageblock_offset4_B = key.tib._offset_B[4],
         .imageblock_offset5_B = key.tib._offset_B[5],
         .imageblock_offset6_B = key.tib._offset_B[6],
         .imageblock_offset7_B = key.tib._offset_B[7],
         .mixed_mrt = count > 1,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_known
                             ? shader->info.agx3_staging_halfs
                             : 0,
         .imageblock = imageblock,
      };
      if (!agx3_g17p_rt0_eot_program_generate(dev->dev.chip, &desc,
                                               &eot_program)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
      program_bytes = eot_program.bytes;
      program_length = eot_program.length;
      resource_spec = AGX_IOTO_G17P_EOT_RESOURCE_SPEC;
   } else {
      struct agx3_g17p_rt0_bg_program_desc desc = {
         .program_va = reservation.va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_halfs,
         .clear_push_halfs = shader->info.push_count,
         .clear_start_halfs = 4 + 8 * reloads,
         .staging_known = shader->info.agx3_staging_known,
         .op = bg_op,
         .imageblock = imageblock,
      };

      if (clears) {
         static_assert(sizeof(atts[0]->clearValue.color) == 16, "fixed ABI");
         VkClearColorValue clear_values[HK_MAX_RTS];
         for (unsigned rt = reloads; rt < count; ++rt)
            clear_values[rt - reloads] = atts[rt]->clearValue.color;
         desc.clear_value_va = hk_pool_upload(
            cmd, clear_values, 16 * (count - reloads), 16);
         if (!desc.clear_value_va)
            return (struct hk_bg_eot){0};
      }

      if (reloads) {
         /* Target n reads native texture index 2n; the odd entries between
          * them stay zero. */
         static_assert(sizeof(struct hk_native_texture_descriptor) ==
                          AGX_TEXTURE_GEN9_LENGTH,
                       "native texture table stride");
         struct hk_native_texture_descriptor textures[2 * HK_MAX_RTS - 1] = {0};
         for (unsigned rt = 0; rt < reloads; ++rt)
            textures[2 * rt] = key.tib.layered
                                  ? views[rt]->planes[0].layered_background
                                  : views[rt]->planes[0].background;
         const uint64_t texture_va = hk_pool_upload(
            cmd, textures, (2 * reloads - 1) * AGX_TEXTURE_GEN9_LENGTH,
            AGX_TEXTURE_GEN9_LENGTH);
         const uint64_t sampler_va =
            dev->rodata.bo && dev->rodata.bo->va ? dev->rodata.bo->va->addr : 0;
         if (!texture_va || !sampler_va)
            return (struct hk_bg_eot){0};
         const uint64_t pointers[2] = {texture_va, sampler_va};
         const uint64_t table =
            hk_pool_upload(cmd, pointers, sizeof(pointers), 8);
         if (!table)
            return (struct hk_bg_eot){0};
         desc.texture_pointer_va = table;
         desc.sampler_pointer_va = table + sizeof(uint64_t);
      }

      if (!agx3_g17p_rt0_bg_program_generate(dev->dev.chip, &desc,
                                             &bg_program) ||
          !agx3_g17p_rt0_bg_resource_spec(dev->dev.chip, bg_op,
                                          &resource_spec)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
      program_bytes = bg_program.bytes;
      program_length = bg_program.length;
   }

   memcpy(reservation.map, program_bytes, program_length);
   const uint64_t published =
      hk_cmd_publish_ioto_program(cmd, &reservation, program_length);
   if (!published)
      return (struct hk_bg_eot){0};

   return (struct hk_bg_eot){
      .usc = published,
      .rsrc_spec = resource_spec,
   };
}

/* The native dual- and multi-target background and EOT programs carry one op
 * for every target: agx3_g17p_rt0_bg_program_generate() selects a single
 * RELOAD/CLEAR/CLEAR_RELOAD program, and the EOT builtin stores every target
 * it names. A layout whose live targets do not share one op -- an attachment
 * that is not stored beside one that is, or a clear beside a reload -- cannot
 * be expressed that way, which is what the multi/dual guards below refuse.
 *
 * The mixed builder can express it: it groups targets into LOAD, then CLEAR,
 * then STORE, drops the untouched ones, and carries each target's own tile
 * offset and PBE descriptor. Route there rather than refusing. */
static bool
hk_g17p_bg_eot_ops_uniform(const struct agx_bg_eot_key *key)
{
   bool seen = false;
   enum agx_bg_eot_op first = AGX_BG_EOT_NONE;

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->tib.logical_format); ++rt) {
      if (key->tib.logical_format[rt] == PIPE_FORMAT_NONE)
         continue;
      if (!seen) {
         first = key->op[rt];
         seen = true;
      } else if (key->op[rt] != first) {
         return false;
      }
   }
   return true;
}

static struct hk_bg_eot
hk_build_g17p_bg_eot(struct hk_cmd_buffer *cmd,
                     struct agx_bg_eot_key *key,
                     const VkRenderingInfo *info, bool store,
                     bool partial_render)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   /* Classification compacts holes internally, so it is invariant under the
    * explicit key/view compaction below. Keep the result for all subsequent
    * BG/EOT routing instead of rescanning the eight targets twice. */
   const enum agx_g17p_mrt_layout mrt = agx_g17p_classify_mrt(&key->tib);
   if (agx_tilebuffer_spills(&key->tib) ||
       mrt == AGX_G17P_MRT_MIXED ||
       !hk_g17p_bg_eot_ops_uniform(key))
      return hk_build_g17p_mixed_bg_eot(cmd, key, info, store);

   /* Vulkan keeps unused colour attachments as holes in the logical array.
    * Native BG/EOT programs number only live attachments. Compact the key and
    * matching image views together while preserving each physical tile
    * offset, then leave the render state itself in Vulkan location order. */
   struct agx_bg_eot_key compact_key = *key;
   uint8_t attachment_map[HK_MAX_RTS];
   const unsigned active_count = agx_tilebuffer_compact(
      &key->tib, &compact_key.tib, attachment_map);
   memset(compact_key.op, 0, sizeof(compact_key.op));
   struct hk_attachment color_att[HK_MAX_RTS] = {0};
   VkRenderingAttachmentInfo attachments[HK_MAX_RTS] = {0};
   for (unsigned rt = 0; rt < active_count; ++rt) {
      const unsigned logical = attachment_map[rt];
      compact_key.op[rt] = key->op[logical];
      color_att[rt] = cmd->state.gfx.render.color_att[logical];
      attachments[rt] = info->pColorAttachments[logical];
   }
   if (active_count == 0 && compact_key.tib.nr_samples == 0)
      agx_tilebuffer_set_samples(&compact_key.tib, 1);
   VkRenderingInfo compact_info = *info;
   compact_info.colorAttachmentCount = active_count;
   compact_info.pColorAttachments = attachments;
   key = &compact_key;
   info = &compact_info;
   if (!hk_agx3_g17p_rt0_tile_layout_supported(&key->tib)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }

   /* As in the generic BG/EOT path, compile an empty attachmentless program
    * as 1x. This is only the local shader key: the render state's sample count
    * remains deferred until a draw supplies it.
    */
   if (!key->tib.nr_samples)
      key->tib.nr_samples = 1;

   /* Decide from the compacted attachment and its matching view. Vulkan may
    * leave holes before the sole live color attachment; selecting from
    * logical slot0 and colorAttachmentCount before compaction loses its
    * STORE requirement and selects a single-sample EOT for an MS surface.
    * Keep samples for partial renders, explicit STORE, and resolves that
    * execute after the pass. Final direct resolves remain single-sampled. */
   if (store && active_count == 1) {
      const VkRenderingAttachmentInfo *att = &attachments[0];
      const struct hk_image_view *iview = color_att[0].iview;
      const bool store_source = att->storeOp == VK_ATTACHMENT_STORE_OP_STORE;
      const bool has_resolve = att->resolveImageView != VK_NULL_HANDLE &&
                               att->resolveMode != VK_RESOLVE_MODE_NONE;
      const bool resolved_after_pass = has_resolve &&
         !hk_agx3_g17p_direct_color_resolve_supported(
            &key->tib, active_count, has_resolve,
            att->resolveMode == VK_RESOLVE_MODE_AVERAGE_BIT, store_source);
      key->multisample_store = hk_agx3_g17p_multisample_store_supported(
         &key->tib, active_count,
         partial_render || store_source || resolved_after_pass,
         iview ? iview->vk.image->samples : 0);
   }

   const bool wboit_mrt = mrt == AGX_G17P_MRT_WBOIT;
   const bool rgba8_mrt = mrt == AGX_G17P_MRT_RGBA8;
   const bool r11g11b10_mrt = mrt == AGX_G17P_MRT_R11G11B10;
   const bool dual_mrt = wboit_mrt || (rgba8_mrt && active_count == 2) ||
                         r11g11b10_mrt;
   const bool multi_mrt = mrt == AGX_G17P_MRT_RGBA32 ||
                          mrt == AGX_G17P_MRT_R32_QUAD ||
                          (rgba8_mrt && active_count > 2);
   unsigned multi_count = 0;
   if (multi_mrt) {
      while (multi_count < HK_MAX_RTS &&
             key->tib.logical_format[multi_count] != PIPE_FORMAT_NONE)
         multi_count++;
   }

   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      const enum agx_bg_eot_op op = key->op[rt];
      const bool rt0_supported = store
                                    ? (op == AGX_EOT_STORE || op == AGX_BG_EOT_NONE)
                                    : (op == AGX_BG_EOT_NONE ||
                                       op == AGX_BG_CLEAR || op == AGX_BG_LOAD);
      const bool dual_mrt_rt1 =
         rt == 1 && dual_mrt &&
         (store ? op == AGX_EOT_STORE
                : (op == AGX_BG_CLEAR || op == AGX_BG_LOAD));
      const bool multi_mrt_rt = rt < multi_count && multi_mrt &&
         (store ? op == AGX_EOT_STORE : (op == AGX_BG_CLEAR || op == AGX_BG_LOAD));
      if ((rt == 0 && !rt0_supported) ||
          (rt != 0 && op != AGX_BG_EOT_NONE && !dual_mrt_rt1 && !multi_mrt_rt)) {
         /* The refusal is about WHICH target carries WHICH operation in a
          * layout the G17P background/EOT programs cannot express, and the
          * deferred VK_ERROR_FEATURE_NOT_PRESENT surfaces much later at
          * vkEndCommandBuffer with none of that visible. Name it. */
         mesa_loge("G17P %s program declined: rt%u op=%u, ops=%u%u%u%u%u%u%u%u "
                   "mrt=%u dual=%u multi=%u/%u samples=%u layered=%u "
                   "sample_size=%uB",
                   store ? "EOT" : "background", rt, (unsigned)op,
                   (unsigned)key->op[0], (unsigned)key->op[1],
                   (unsigned)key->op[2], (unsigned)key->op[3],
                   (unsigned)key->op[4], (unsigned)key->op[5],
                   (unsigned)key->op[6], (unsigned)key->op[7], (unsigned)mrt,
                   dual_mrt, multi_mrt, multi_count, key->tib.nr_samples,
                   key->tib.layered, key->tib.sample_size_B);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
   }

   const enum agx_bg_eot_op op = key->op[0];
   const bool dual_mrt_active = dual_mrt && key->op[1] != AGX_BG_EOT_NONE;
   const bool multi_mrt_active = multi_mrt &&
      (key->op[0] != AGX_BG_EOT_NONE || key->op[1] != AGX_BG_EOT_NONE ||
       key->op[2] != AGX_BG_EOT_NONE || key->op[3] != AGX_BG_EOT_NONE);
   const unsigned attachment_count =
      multi_mrt_active ? multi_count : dual_mrt_active ? 2 : 1;
   if (multi_mrt_active) {
      for (unsigned rt = 0; rt < attachment_count; ++rt) {
         if (info->colorAttachmentCount < attachment_count ||
             key->op[rt] != op || op == AGX_BG_EOT_NONE) {
            /* DIAGNOSTIC: name the refusal the way the rt0 guard above does.
             * These two guards refuse silently, which is why this bucket had
             * no driver-side evidence at all. */
            mesa_loge("G17P %s multi-MRT guard declined: rt%u op=%u op0=%u "
                      "ops=%u%u%u%u%u%u%u%u count=%u attachments=%u mrt=%u "
                      "samples=%u sample_size=%uB",
                      store ? "EOT" : "background", rt, (unsigned)key->op[rt],
                      (unsigned)op, (unsigned)key->op[0], (unsigned)key->op[1],
                      (unsigned)key->op[2], (unsigned)key->op[3],
                      (unsigned)key->op[4], (unsigned)key->op[5],
                      (unsigned)key->op[6], (unsigned)key->op[7],
                      attachment_count, info->colorAttachmentCount,
                      (unsigned)mrt, key->tib.nr_samples,
                      key->tib.sample_size_B);
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return (struct hk_bg_eot){0};
         }
      }
   }
   if (dual_mrt_active &&
       (key->op[1] != op || info->colorAttachmentCount < 2)) {
      mesa_loge("G17P %s dual-MRT guard declined: op0=%u op1=%u "
                "attachments=%u mrt=%u samples=%u sample_size=%uB",
                store ? "EOT" : "background", (unsigned)op,
                (unsigned)key->op[1], info->colorAttachmentCount,
                (unsigned)mrt, key->tib.nr_samples, key->tib.sample_size_B);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }
   if (!store && op == AGX_BG_CLEAR)
      key->reserved_preamble =
         multi_mrt_active ? 4 + 8 * attachment_count
         : rgba8_mrt && dual_mrt_active
            ? AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS
         : r11g11b10_mrt && dual_mrt_active
            ? AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS
         : wboit_mrt && dual_mrt_active
            ? AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS
            : AGX3_G17P_RT0_CLEAR_PUSH_HALFS;

   struct agx_bg_eot_shader *shader =
      agx_get_bg_eot_shader(&dev->bg_eot, key);
   const bool shader_resources_match =
      shader &&
      (store || (op == AGX_BG_EOT_NONE
                    ? shader->info.push_count == 0 && !shader->info.uses_txf
                 : agx3_g17p_rt0_bg_shader_resources_match(
                      op == AGX_BG_CLEAR ? AGX3_G17P_RT0_BG_CLEAR
                                         : AGX3_G17P_RT0_BG_RELOAD,
                      shader->info.push_count, shader->info.uses_txf)));
   if (!store && hk_g17p_launch_trace_enabled() && shader)
      fprintf(stderr, "G17P BG metadata: op=%u size=%u gprs=%u push=%u "
                      "staging=%u txf=%u supported=%u resources=%u\n",
              op, shader->info.binary_size, shader->info.nr_gprs,
              shader->info.push_count, shader->info.agx3_staging_halfs,
              shader->info.uses_txf, hk_g17p_bg_eot_shader_supported(shader),
              shader_resources_match);
   if (!hk_g17p_bg_eot_shader_supported(shader) || !shader_resources_match) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }

   size_t program_capacity;
   const char *program_label;
   if (op == AGX_BG_EOT_NONE) {
      program_capacity = AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE;
      program_label = "G17P empty tile Ioto program";
   } else if (store) {
      program_capacity = multi_mrt_active ? AGX_IOTO_G17P_QUAD_EOT_PROGRAM_MAX_SIZE
                                        : AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE;
      program_label = "G17P RT0 EOT Ioto program";
   } else if (op == AGX_BG_CLEAR) {
      program_capacity =
         multi_mrt_active ? AGX_IOTO_G17P_QUAD_CLEAR_PROGRAM_MAX_SIZE
         : rgba8_mrt && dual_mrt_active
            ? AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE
         : r11g11b10_mrt && dual_mrt_active
            ? AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE
         : wboit_mrt && dual_mrt_active
            ? AGX_IOTO_G17P_WBOIT_MRT_CLEAR_PROGRAM_SIZE
            : AGX_IOTO_G17P_RT0_CLEAR_PROGRAM_SIZE;
      program_label = "G17P RT0 clear Ioto program";
   } else {
      program_capacity = AGX_IOTO_G17P_RT0_RELOAD_PROGRAM_MAX_SIZE;
      program_label = "G17P RT0 reload Ioto program";
   }

   struct hk_ioto_reservation program_reservation;
   if (!hk_cmd_reserve_ioto_program(cmd, program_capacity, program_label,
                                    &program_reservation))
      return (struct hk_bg_eot){0};

   void *program_map = program_reservation.map;
   const uint64_t program_va = program_reservation.va;

   size_t program_length;
   uint64_t resource_spec;
   if (store && op == AGX_EOT_STORE) {
      struct hk_image_view *iview = color_att[0].iview;
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[0];
      /* A direct resolve is the pass's output, so only the pass's final
       * end-of-tile program may perform one. A partial render is a mid-frame
       * flush whose tile is reloaded by the draws that follow it, and a
       * resolve is not reloadable: it averages the samples away. The
       * mid-frame store therefore writes the multisample surface unresolved,
       * which is what key->multisample_store selects below.
       */
      const bool direct_resolve =
         !partial_render &&
         hk_agx3_g17p_direct_color_resolve_supported(
            &key->tib, info->colorAttachmentCount,
            att->resolveImageView != VK_NULL_HANDLE,
            att->resolveMode == VK_RESOLVE_MODE_AVERAGE_BIT,
            att->storeOp == VK_ATTACHMENT_STORE_OP_STORE);
      struct hk_image_view *target = direct_resolve
         ? color_att[0].resolve_iview : iview;
      struct hk_image_view *iview1 = dual_mrt_active
         ? color_att[1].iview : NULL;
      const bool target_pbe_valid =
         target && (key->tib.layered
                       ? target->planes[0].layered_eot_pbe_valid
                       : target->planes[0].eot_pbe_valid);
      const bool target1_pbe_valid =
         iview1 && (key->tib.layered
                       ? iview1->planes[0].layered_eot_pbe_valid
                       : iview1->planes[0].eot_pbe_valid);
      /* A G17P end-of-tile program writes either an ordinary single-sample
       * PBE target -- resolving the tile on the way out -- or, when the key
       * asked for it, the unresolved multisample surface through the separate
       * multisample IMGWBLK builtin. Those are different programs, so the
       * target's sample count has to agree with the one the key selected.
       *
       * Without this, storeOp = STORE on a multisample attachment picks
       * target = iview (direct resolve requires storeOp != STORE), skips the
       * samples != 1 check below because that is guarded by direct_resolve,
       * and writes single-sample data into a multisample surface with no
       * diagnostic. Refuse that combination instead of storing wrong pixels.
       */
      const bool multisample_store =
         dev->dev.chip == AGX_CHIP_G17P && key->multisample_store;
      if (!iview || !target ||
          (dev->dev.chip == AGX_CHIP_G17P && !direct_resolve &&
           !multisample_store && target->vk.image->samples != 1) ||
          (multisample_store &&
           (direct_resolve || target != iview ||
            target->vk.image->samples != key->tib.nr_samples)) ||
          (direct_resolve &&
           (target->vk.image->samples != 1 ||
            target->vk.format != iview->vk.format)) ||
          !target_pbe_valid ||
          info->pColorAttachments[0].imageView == VK_NULL_HANDLE ||
          (dual_mrt_active &&
           (!iview1 || !target1_pbe_valid ||
            info->pColorAttachments[1].imageView == VK_NULL_HANDLE))) {
         if (debug_get_bool_option("HK_DEBUG_RECORD", false))
            mesa_loge("EOT target mismatch: partial=%u active=%u samples=%u "
                      "target_samples=%u direct=%u ms_store=%u pbe=%u "
                      "store_op=%u resolve_mode=%u",
                      partial_render, active_count, key->tib.nr_samples,
                      target ? target->vk.image->samples : 0,
                      direct_resolve, multisample_store, target_pbe_valid,
                      att->storeOp, att->resolveMode);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
         return (struct hk_bg_eot){0};
      }

      const unsigned descriptor_generation =
         agx_chip_descriptor_generation(dev->dev.chip);
      if (hk_pbe_descriptor_size(descriptor_generation) !=
          AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
      static_assert(sizeof(struct hk_native_pbe_descriptor) ==
                       AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE,
                    "native PBE table stride");

      const struct hk_native_pbe_descriptor target_pbe =
         key->tib.layered ? target->planes[0].layered_eot
                          : target->planes[0].eot;
      struct hk_native_pbe_descriptor target1_pbe = {0};
      if (dual_mrt_active) {
         target1_pbe = key->tib.layered ? iview1->planes[0].layered_eot
                                        : iview1->planes[0].eot;
      }
      struct hk_native_pbe_descriptor pbe_arguments[4] = {target_pbe};
      if (dual_mrt_active)
         pbe_arguments[1] = target1_pbe;
      if (multi_mrt_active) {
         for (unsigned rt = 1; rt < attachment_count; ++rt) {
            struct hk_image_view *view = color_att[rt].iview;
            if (!view || !view->planes[0].eot_pbe_valid ||
                view->vk.image->samples != 1 ||
                info->pColorAttachments[rt].imageView == VK_NULL_HANDLE) {
               vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
               return (struct hk_bg_eot){0};
            }
            pbe_arguments[rt] = view->planes[0].eot;
         }
      }
      const size_t pbe_argument_size =
         multi_mrt_active ? attachment_count * AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE
         : dual_mrt_active ? AGX_IOTO_G17P_RGBA8_MRT_PBE_ARGUMENT_SIZE
                         : AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE;
      const uint64_t pbe_argument_va =
         hk_pool_upload(cmd, pbe_arguments, pbe_argument_size, 32);
      if (!pbe_argument_va)
         return (struct hk_bg_eot){0};
      if (hk_g17p_launch_trace_enabled())
         hk_g17p_trace_line("eot-pbe", pbe_argument_va, pbe_arguments,
                            pbe_argument_size);

      const struct agx3_g17p_rt0_eot_program_desc desc = {
         .program_va = program_va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .pbe_argument_va = pbe_argument_va,
         .imageblock_offset_B = agx_tilebuffer_offset_B(&key->tib, 0),
         .attachment_count = attachment_count,
         .imageblock_offset1_B =
            attachment_count > 1 ? agx_tilebuffer_offset_B(&key->tib, 1) : 0,
         .imageblock_offset2_B = attachment_count > 2 ? agx_tilebuffer_offset_B(&key->tib, 2) : 0,
         .imageblock_offset3_B = attachment_count > 3 ? agx_tilebuffer_offset_B(&key->tib, 3) : 0,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_known
                            ? shader->info.agx3_staging_halfs : 0,
         .imageblock = {
            .sample_size_B = key->tib.sample_size_B,
            .nr_samples = key->tib.nr_samples,
            .tile_size_px = key->tib.tile_size,
         },
      };
      struct agx3_g17p_rt0_eot_program program;
      if (!agx3_g17p_rt0_eot_program_generate(dev->dev.chip, &desc,
                                               &program)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      memcpy(program_map, program.bytes, program.length);
      program_length = program.length;
      resource_spec = AGX_IOTO_G17P_EOT_RESOURCE_SPEC;
      /* The address the program was composed for. An identical program
       * already published by this recording keeps its own address, which the
       * BG/EOT trace below reports. */
      if (hk_g17p_launch_trace_enabled())
         fprintf(stderr,
                 "G17P EOT composed: program=%" PRIx64 " length=%zu "
                 "pbe=%" PRIx64 " attachments=%u offsets=%u,%u,%u,%u "
                 "gprs=%u staging=%u\n",
                 program_va, program_length, pbe_argument_va,
                 desc.attachment_count, desc.imageblock_offset_B,
                 desc.imageblock_offset1_B, desc.imageblock_offset2_B,
                 desc.imageblock_offset3_B, desc.nr_gprs,
                 desc.staging_halfs);

      if (direct_resolve)
         cmd->state.gfx.render.g17p_direct_color_resolve = true;
   } else if (op == AGX_BG_EOT_NONE) {
      const struct hk_agx3_g17p_graphics_program_desc desc = {
         .program_va = program_va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .compiler_d_valid = shader->info.agx3_staging_known,
         .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
         .fragment = true,
         .imageblock = {
            .sample_size_B = key->tib.sample_size_B,
            .nr_samples = key->tib.nr_samples,
            .tile_size_px = key->tib.tile_size,
         },
      };
      struct hk_agx3_g17p_graphics_program program;
      if (!hk_agx3_g17p_basic_graphics_program_generate(dev->dev.chip, &desc,
                                                        &program)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      memcpy(program_map, program.bytes, program.length);
      program_length = program.length;
      if (!agx3_g17p_rt0_bg_resource_spec(
             dev->dev.chip, AGX3_G17P_RT0_BG_NONE, &resource_spec)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
   } else {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[0];
      struct hk_image_view *iview = color_att[0].iview;
      const VkRenderingAttachmentInfo *att_info1 =
         dual_mrt_active ? &info->pColorAttachments[1] : NULL;
      struct hk_image_view *iview1 =
         dual_mrt_active ? color_att[1].iview : NULL;
      if (!iview || att_info->imageView == VK_NULL_HANDLE ||
          (dual_mrt_active &&
           (!iview1 || att_info1->imageView == VK_NULL_HANDLE))) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
         return (struct hk_bg_eot){0};
      }
      if (multi_mrt_active) {
         for (unsigned rt = 1; rt < attachment_count; ++rt) {
            struct hk_image_view *view = color_att[rt].iview;
            if (!view || view->vk.image->samples != 1 ||
                info->pColorAttachments[rt].imageView == VK_NULL_HANDLE) {
               vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
               return (struct hk_bg_eot){0};
            }
         }
      }

      const unsigned descriptor_generation =
         agx_chip_descriptor_generation(dev->dev.chip);
      if (hk_texture_descriptor_size(descriptor_generation) != 32 ||
          agx_chip_sampler_heap_element_size(dev->dev.chip) != 32) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      struct agx3_g17p_rt0_bg_program_desc desc = {
         .program_va = program_va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_halfs,
         .clear_push_halfs = shader->info.push_count,
         .staging_known = shader->info.agx3_staging_known,
         .op = op == AGX_BG_CLEAR ? AGX3_G17P_RT0_BG_CLEAR
                                  : AGX3_G17P_RT0_BG_RELOAD,
         .imageblock = {
            .sample_size_B = key->tib.sample_size_B,
            .nr_samples = key->tib.nr_samples,
            .tile_size_px = key->tib.tile_size,
         },
      };

      if (op == AGX_BG_CLEAR) {
         static_assert(sizeof(att_info->clearValue.color) == 16, "fixed ABI");
         VkClearColorValue clear_values[4] = {
            att_info->clearValue.color,
         };
         if (dual_mrt_active)
            clear_values[1] = att_info1->clearValue.color;
         if (multi_mrt_active) {
            for (unsigned rt = 1; rt < attachment_count; ++rt)
               clear_values[rt] = info->pColorAttachments[rt].clearValue.color;
         }
         const size_t clear_size =
            multi_mrt_active ? 16 * attachment_count
            : rgba8_mrt && dual_mrt_active ? 32
            : r11g11b10_mrt && dual_mrt_active ? 28
            : wboit_mrt && dual_mrt_active ? 20 : 16;
         desc.clear_value_va =
            hk_pool_upload(cmd, clear_values, clear_size, 16);
         if (!desc.clear_value_va)
            return (struct hk_bg_eot){0};
      } else {
         /* Native texture-table entries occupy aligned 32-byte slots. RT1 is
          * native texture index2, so the exact MRT table retains index1 as a
          * zero gap. */
         static_assert(sizeof(struct hk_native_texture_descriptor) ==
                          AGX_TEXTURE_GEN9_LENGTH,
                       "native texture table stride");
         struct hk_native_texture_descriptor textures[7] = {
            key->tib.layered ? iview->planes[0].layered_background
                             : iview->planes[0].background,
         };
         if (dual_mrt_active)
            textures[2] = key->tib.layered
                             ? iview1->planes[0].layered_background
                             : iview1->planes[0].background;
         if (multi_mrt_active) {
            for (unsigned rt = 1; rt < attachment_count; ++rt)
               textures[2 * rt] = color_att[rt].iview->planes[0].background;
         }
         const size_t texture_size = multi_mrt_active
            ? (2 * attachment_count - 1) * AGX_TEXTURE_GEN9_LENGTH : dual_mrt_active
            ? 3 * AGX_TEXTURE_GEN9_LENGTH : AGX_TEXTURE_GEN9_LENGTH;
         const uint64_t texture_va =
            hk_pool_upload(cmd, textures, texture_size,
                           AGX_TEXTURE_GEN9_LENGTH);
         const uint64_t sampler_va =
            dev->rodata.bo && dev->rodata.bo->va ? dev->rodata.bo->va->addr : 0;
         if (!texture_va || !sampler_va)
            return (struct hk_bg_eot){0};
         const uint64_t pointers[2] = {texture_va, sampler_va};
         const uint64_t table = hk_pool_upload(cmd, pointers, sizeof(pointers), 8);
         if (!table)
            return (struct hk_bg_eot){0};
         desc.texture_pointer_va = table;
         desc.sampler_pointer_va = table + sizeof(uint64_t);
      }

      struct agx3_g17p_rt0_bg_program program;
      if (!agx3_g17p_rt0_bg_program_generate(dev->dev.chip, &desc,
                                             &program)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      memcpy(program_map, program.bytes, program.length);
      program_length = program.length;
      if (!agx3_g17p_rt0_bg_resource_spec(dev->dev.chip, desc.op,
                                          &resource_spec)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
   }

   const uint64_t published =
      hk_cmd_publish_ioto_program(cmd, &program_reservation, program_length);
   if (!published)
      return (struct hk_bg_eot){0};

   if (!store && hk_g17p_launch_trace_enabled()) {
      fprintf(stderr,
              "G17P BG launch: op=%s program=%" PRIx64 " resources=%" PRIx64
              " length=%zu staging_known=%u staging_halfs=%u compiler_d=%u\n",
              op == AGX_BG_CLEAR ? "clear" :
              op == AGX_BG_LOAD ? "reload" : "none",
              published, resource_spec, program_length,
              shader->info.agx3_staging_known,
              shader->info.agx3_staging_halfs,
              shader->info.agx3_staging_known
                 ? (unsigned)shader->info.agx3_staging_halfs * 2
                 : shader->info.nr_gprs);
   }

   return (struct hk_bg_eot){
      .usc = published,
      .rsrc_spec = resource_spec,
   };
}

struct hk_g17p_bg_eot_cache_key {
   struct agx_tilebuffer_layout tib;
   struct {
      uint64_t image_view;
      uint64_t resolve_image_view;
      VkClearColorValue clear;
      uint32_t load_op;
      uint32_t store_op;
      uint32_t resolve_mode;
   } attachments[HK_MAX_RTS];
   uint32_t layers;
   uint8_t color_attachment_count;
   bool incomplete_render_area;
};

static uint64_t
hk_g17p_bg_eot_cache_fingerprint(
   const struct hk_g17p_bg_eot_cache_key *key)
{
   static_assert(sizeof(*key) % sizeof(uint64_t) == 0,
                 "word-addressable BG/EOT cache key");
   const uint8_t *bytes = (const uint8_t *)key;
   uint64_t hash = UINT64_C(0x9e3779b97f4a7c15);
   for (unsigned i = 0; i < sizeof(*key) / sizeof(uint64_t); ++i) {
      uint64_t word;
      memcpy(&word, bytes + i * sizeof(word), sizeof(word));
      hash ^= word + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) +
              (hash >> 2);
   }
   return hash;
}

static struct hk_g17p_bg_eot_cache_entry *
hk_g17p_bg_eot_cache_get(struct hk_cmd_buffer *cmd,
                         const VkRenderingInfo *info,
                         bool incomplete_render_area)
{
   static_assert(sizeof(struct hk_g17p_bg_eot_cache_key) <=
                    HK_G17P_BG_EOT_CACHE_KEY_MAX,
                 "G17P BG/EOT cache key storage");

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (dev->dev.chip != AGX_CHIP_G17P ||
       dev->g17p_diagnostics.launch_trace)
      return NULL;

   struct hk_g17p_bg_eot_cache_key cache_key = {
      .tib = cmd->state.gfx.render.tilebuffer,
      .layers = cmd->state.gfx.render.cr.layers,
      .color_attachment_count = info->colorAttachmentCount,
      .incomplete_render_area = incomplete_render_area,
   };
   for (unsigned i = 0; i < info->colorAttachmentCount; ++i) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[i];
      cache_key.attachments[i].image_view = (uintptr_t)att->imageView;
      cache_key.attachments[i].load_op = att->loadOp;
      cache_key.attachments[i].store_op = att->storeOp;
      if (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
         cache_key.attachments[i].clear = att->clearValue.color;
      if (att->resolveMode != VK_RESOLVE_MODE_NONE) {
         cache_key.attachments[i].resolve_image_view =
            (uintptr_t)att->resolveImageView;
         cache_key.attachments[i].resolve_mode = att->resolveMode;
      }
   }

   const uint64_t hash = hk_g17p_bg_eot_cache_fingerprint(&cache_key);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_g17p_bg_eot_cache_entry *cached = NULL;
   for (unsigned i = 0; i < HK_G17P_BG_EOT_CACHE_ENTRIES; ++i) {
      struct hk_g17p_bg_eot_cache_entry *entry =
         &gfx->g17p_bg_eot_cache[i];
      if (!entry->valid) {
         if (!cached)
            cached = entry;
         continue;
      }
      if (entry->hash == hash && entry->key_size == sizeof(cache_key) &&
          !memcmp(entry->key, &cache_key, sizeof(cache_key))) {
         gfx->render.g17p_direct_color_resolve |=
            entry->direct_color_resolve_variants != 0;
         return entry;
      }
   }

   if (!cached)
      cached = &gfx->g17p_bg_eot_cache[
         gfx->g17p_bg_eot_cache_next++ % HK_G17P_BG_EOT_CACHE_ENTRIES];

   memcpy(cached->key, &cache_key, sizeof(cache_key));
   cached->hash = hash;
   cached->key_size = sizeof(cache_key);
   cached->valid_variants = 0;
   cached->direct_color_resolve_variants = 0;
   cached->valid = true;
   return cached;
}

static struct hk_bg_eot
hk_build_bg_eot(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                bool store, bool partial_render, bool incomplete_render_area,
                struct hk_g17p_bg_eot_cache_entry *cache)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* Construct the key */
   struct agx_bg_eot_key key = {.tib = render->tilebuffer};
   static_assert(AGX_BG_EOT_NONE == 0, "default initializer");

   key.tib.layered = (render->cr.layers > 1);

   /* Every background program that reloads a multisample tile reloads it from
    * the multisample surface, per sample. That is the only surface that
    * carries the samples, and the samples are exactly what a reload exists to
    * restore: a resolve has averaged them away and cannot be un-averaged.
    * This holds for loadOp = LOAD, for an incomplete render area, and for the
    * mid-frame reload of a partial render alike, so single_sample_reload is
    * never the right answer here and is left false.
    *
    * Gallium does set it (agx_state.c), and it swaps in the resolve target's
    * surface to match. Honeykrisp once set the flag without swapping the
    * descriptor, which asked a single-sample texture fetch for the
    * multisample attachment -- a surface a direct resolve never writes. A
    * fresh allocation reads as zero, so every sample a partial render carried
    * across a stream boundary came back zero and the tile's whole history,
    * clear included, was lost.
    */
   key.single_sample_reload = false;

   bool needs_textures_for_spilled_rts =
      agx_tilebuffer_spills(&render->tilebuffer) && !partial_render && !store;

   for (unsigned i = 0; i < info->colorAttachmentCount; ++i) {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE)
         continue;

      /* Partial render programs exist only to store/load the tilebuffer to
       * main memory. When render targets are already spilled to main memory,
       * there's nothing to do.
       */
      if (key.tib.spilled[i] &&
          (partial_render || store || dev->dev.chip == AGX_CHIP_G17P))
         continue;

      if (store) {
         bool should_store = is_attachment_stored(att_info);

         /* Partial renders always need to flush to memory. */
         should_store |= partial_render;

         if (should_store)
            key.op[i] = AGX_EOT_STORE;
      } else {
         bool load = att_info->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
         bool clear = att_info->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;

         /* The background program used for partial renders must always load
          * whatever was stored in the mid-frame end-of-tile program.
          */
         load |= partial_render;

         /* With an incomplete render area, we're forced to load back tiles and
          * then use the 3D pipe for the clear.
          */
         load |= incomplete_render_area;

         /* Don't read back spilled render targets, they're already in memory */
         load &= !key.tib.spilled[i];

         /* This is a very frustrating corner case. From the spec:
          *
          *     VK_ATTACHMENT_STORE_OP_NONE specifies the contents within the
          *     render area are not accessed by the store operation as long as
          *     no values are written to the attachment during the render pass.
          *
          * With VK_ATTACHMENT_STORE_OP_NONE, we suppress stores on the main
          * end-of-tile program. Unfortunately, that's not enough: we also need
          * to preserve the contents throughout partial renders. The easiest way
          * to do that is forcing a load in the background program, so that
          * partial stores for unused attachments will be no-op'd by writing
          * existing contents.
          *
          * Optimizing this would require nontrivial tracking. Fortunately,
          * this is all Android gunk and we don't have to care too much for
          * dekstop games. So do the simple thing.
          */
         bool no_store = (att_info->storeOp == VK_ATTACHMENT_STORE_OP_NONE);
         bool no_store_wa = no_store && !load && !clear;
         if (no_store_wa) {
            perf_debug(cmd, "STORE_OP_NONE workaround");
         }

         load |= no_store_wa;

         /* Don't apply clears for spilled render targets when we clear the
          * render area explicitly after.
          */
         if (key.tib.spilled[i] && incomplete_render_area)
            continue;

         if (load)
            key.op[i] = AGX_BG_LOAD;
         else if (clear)
            key.op[i] = AGX_BG_CLEAR;
      }
   }

   if (dev->dev.chip == AGX_CHIP_G17P) {
      const unsigned variant = (store ? 2 : 0) | (partial_render ? 1 : 0);
      if (cache && (cache->valid_variants & BITFIELD_BIT(variant))) {
         render->g17p_direct_color_resolve |=
            (cache->direct_color_resolve_variants &
             BITFIELD_BIT(variant)) != 0;
         return cache->program[variant];
      }

      const struct hk_bg_eot program =
         hk_build_g17p_bg_eot(cmd, &key, info, store, partial_render);
      if (cache && program.usc && !vk_command_buffer_has_error(&cmd->vk)) {
         cache->program[variant] = program;
         cache->valid_variants |= BITFIELD_BIT(variant);
         if (store && !partial_render && render->g17p_direct_color_resolve)
            cache->direct_color_resolve_variants |= BITFIELD_BIT(variant);
      }
      return program;
   }

   const unsigned descriptor_generation =
      agx_chip_descriptor_generation(dev->dev.chip);
   if (!hk_native_pbe_hardware_supported(descriptor_generation)) {
      if (needs_textures_for_spilled_rts) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
         if (key.op[rt] == AGX_EOT_STORE) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return (struct hk_bg_eot){0};
         }
      }
   }

   /* Begin building the pipeline */
   size_t usc_size = agx_usc_size(3 + HK_MAX_RTS);
   struct agx_ptr t = hk_pool_usc_alloc(cmd, usc_size, 64);
   if (!t.cpu)
      return (struct hk_bg_eot){.usc = t.gpu};

   struct agx_usc_builder b = agx_usc_builder(t.cpu, usc_size);

   bool uses_txf = false;
   unsigned uniforms = 0;
   unsigned nr_tex = 0;

   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[rt];
      struct hk_image_view *iview = render->color_att[rt].iview;

      if (key.op[rt] == AGX_BG_LOAD) {
         uses_txf = true;

         const struct hk_native_texture_descriptor *desc =
            key.tib.layered ? &iview->planes[0].layered_background
                            : &iview->planes[0].background;

         agx_usc_pack(&b, TEXTURE, cfg) {
            /* Shifted to match eMRT indexing, could be optimized */
            cfg.start = rt * 2;
            cfg.count = 1;
            cfg.buffer = hk_pool_upload(
               cmd, desc,
               hk_texture_descriptor_size(descriptor_generation), 8);
         }

         nr_tex = (rt * 2) + 1;
      } else if (key.op[rt] == AGX_BG_CLEAR) {
         static_assert(sizeof(att_info->clearValue.color) == 16, "fixed ABI");
         uint64_t colour =
            hk_pool_upload(cmd, &att_info->clearValue.color, 16, 16);

         agx_usc_uniform(&b, 4 + (8 * rt), 8, colour);
         uniforms = MAX2(uniforms, 4 + (8 * rt) + 8);
      } else if (key.op[rt] == AGX_EOT_STORE) {
         const bool valid = key.tib.layered
                               ? iview->planes[0].layered_eot_pbe_valid
                               : iview->planes[0].eot_pbe_valid;
         if (!valid) {
            mesa_loge("render target %u needs an unproven native PBE axis",
                      rt);
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FORMAT_NOT_SUPPORTED);
            return (struct hk_bg_eot){0};
         }

         const struct hk_native_pbe_descriptor *desc =
            key.tib.layered ? &iview->planes[0].layered_eot
                            : &iview->planes[0].eot;

         agx_usc_pack(&b, TEXTURE, cfg) {
            cfg.start = rt;
            cfg.count = 1;
            cfg.buffer = hk_pool_upload(
               cmd, desc, hk_pbe_descriptor_size(descriptor_generation), 8);
         }

         nr_tex = rt + 1;
      }
   }

   if (needs_textures_for_spilled_rts) {
      hk_usc_upload_spilled_rt_descs(&b, cmd);
      uniforms = MAX2(uniforms, 4);
   }

   if (uses_txf) {
      agx_usc_push_packed(&b, SAMPLER, dev->dev.txf_sampler);
   }

   /* For attachmentless rendering, we don't know the sample count until
    * draw-time. But we have trivial bg/eot programs in that case too.
    */
   if (key.tib.nr_samples >= 1) {
      agx_usc_push_packed(&b, SHARED, &key.tib.usc);
   } else {
      assert(key.tib.sample_size_B == 0);
      agx_usc_shared_none(&b);

      key.tib.nr_samples = 1;
   }

   /* Get the shader */
   key.reserved_preamble = uniforms;
   /* XXX: locking? */
   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&dev->bg_eot, &key);

   agx_usc_pack(&b, SHADER, cfg) {
      cfg.code = agx_usc_addr(&dev->dev, shader->ptr);
      cfg.unk_2 = 0;
   }

   agx_usc_pack(&b, REGISTERS, cfg)
      cfg.register_count = shader->info.nr_gprs;

   if (shader->info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code =
            agx_usc_addr(&dev->dev, shader->ptr + shader->info.preamble_offset);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   struct hk_bg_eot ret = {.usc = t.gpu};

   agx_pack(&ret.counts, COUNTS, cfg) {
      cfg.uniform_register_count = shader->info.push_count;
      cfg.preshader_register_count = shader->info.nr_preamble_gprs;
      cfg.texture_state_register_count = nr_tex;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(uses_txf ? 1 : 0, false);
   }

   return ret;
}

static void
hk_merge_render_iview(struct hk_rendering_state *render,
                      struct hk_image_view *iview, bool zls)
{
   if (iview) {
      unsigned samples = iview->vk.image->samples;
      /* TODO: is this right for ycbcr? */
      unsigned level = iview->vk.base_mip_level;
      unsigned width = u_minify(iview->vk.image->extent.width, level);
      unsigned height = u_minify(iview->vk.image->extent.height, level);

      assert(render->tilebuffer.nr_samples == 0 ||
             render->tilebuffer.nr_samples == samples);
      render->tilebuffer.nr_samples = samples;

      /* TODO: Is this merging logic sound? Not sure how this is supposed to
       * work conceptually.
       */
      render->cr.width = MAX2(render->cr.width, width);
      render->cr.height = MAX2(render->cr.height, height);

      if (zls) {
         render->cr.zls_width = width;
         render->cr.zls_height = height;
      }
   }
}

static void
hk_pack_zls_control(struct agx_zls_control_packed *packed,
                    struct ail_layout *z_layout, struct ail_layout *s_layout,
                    const VkRenderingAttachmentInfo *z,
                    const VkRenderingAttachmentInfo *s,
                    bool incomplete_render_area, bool partial_render)
{
   struct agx_zls zls = {0};

   if (z) {
      /* XXX: Dropping Z stores is wrong if the render pass gets split into
       * multiple control streams (can that ever happen?) We need more ZLS
       * variants. Force || true for now.
       */
      zls.z_store = z->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    z->resolveMode != VK_RESOLVE_MODE_NONE || partial_render ||
                    true;

      zls.z_load = z->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
                   incomplete_render_area;
   }

   if (s) {
      /* TODO:
       * Fail
       * dEQP-VK.renderpass.dedicated_allocation.formats.d32_sfloat_s8_uint.input.dont_care.store.self_dep_clear_draw_use_input_aspect
       * without the force
       * .. maybe a VkRenderPass emulation bug.
       */
      zls.s_store = s->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    s->resolveMode != VK_RESOLVE_MODE_NONE || partial_render ||
                    true;

      zls.s_load = s->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
                   incomplete_render_area;
   }

   agx_pack_zls_control(packed, z_layout, s_layout, &zls);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginRendering(VkCommandBuffer commandBuffer,
                     const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (dev->g17p_shader_perf.prefault_stack)
      hk_shader_perf_prefault_stack();
   if (dev->g17p_shader_perf.prefault_driver)
      hk_shader_perf_prefault_driver();
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   memset(render, 0, sizeof(*render));

   render->flags = pRenderingInfo->flags;
   render->area = pRenderingInfo->renderArea;
   render->view_mask = pRenderingInfo->viewMask;
   render->layer_count = pRenderingInfo->layerCount;
   render->tilebuffer.nr_samples = 0;

   const uint32_t layer_count = render->view_mask
                                   ? util_last_bit(render->view_mask)
                                   : render->layer_count;

   render->color_att_count = pRenderingInfo->colorAttachmentCount;
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_attachment_init(&render->color_att[i],
                         &pRenderingInfo->pColorAttachments[i]);
   }

   hk_attachment_init(&render->depth_att, pRenderingInfo->pDepthAttachment);
   hk_attachment_init(&render->stencil_att, pRenderingInfo->pStencilAttachment);

   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_merge_render_iview(render, render->color_att[i].iview, false);
   }

   hk_merge_render_iview(
      render, render->depth_att.iview ?: render->stencil_att.iview, true);

   /* Infer for attachmentless. samples is inferred at draw-time. */
   render->cr.width =
      MAX2(render->cr.width, render->area.offset.x + render->area.extent.width);

   render->cr.height = MAX2(render->cr.height,
                            render->area.offset.y + render->area.extent.height);

   /* On G17P, zero dimensions are the native inactive-ZLS encoding.  Keep
    * them zero for attachmentless and colour-only rendering; every control
    * stream, including a partial stream, inherits that state from render. */
   if (!render->cr.zls_width && dev->dev.chip != AGX_CHIP_G17P) {
      render->cr.zls_width = render->cr.width;
      render->cr.zls_height = render->cr.height;
   }

   render->cr.layers = layer_count;

   /* Choose a tilebuffer layout given the framebuffer key */
   enum pipe_format formats[HK_MAX_RTS] = {0};
   for (unsigned i = 0; i < render->color_att_count; ++i) {
      formats[i] = hk_format_to_pipe_format(render->color_att[i].vk_format);
      if (agx_native_graphics(&dev->dev)) {
         formats[i] = hk_padded_image_format(formats[i]);
         const enum pipe_format integer =
            hk_scaled_texture_format(formats[i]);
         if (integer != PIPE_FORMAT_NONE)
            formats[i] = integer;
      }
   }

   /* G17P's bounded RT0 programs use non-array image operations. Other chips
    * keep the existing always-layered layout.
    */
   const bool layered =
      dev->dev.chip != AGX_CHIP_G17P ||
      !hk_agx3_g17p_single_layer_rendering(render->view_mask, layer_count);
   /* Storage capacity and raster samples are independent limits. In
    * particular 16x16 at 4x may occupy the full32KiB, while 32x32 at 4x
    * exceeds the sample budget even for an eight-byte sample. */
   const bool native_tile_limits = dev->dev.chip == AGX_CHIP_G17P;
   /* A G17P multisample tilebuffer wider than eight bytes per sample only
    * keeps a tile's whole sample where colour-writing geometry covered that
    * tile; see AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE. A load-op clear
    * gets that guarantee from the full-area clear draw emitted below, so it
    * stays in the tilebuffer. A reloaded attachment cannot: the background
    * reload is itself the truncating program, and nothing else rewrites the
    * tiles the application does not draw to. Multiview has a second, separate
    * defect that the clear draw does not cover either. Those two spill. */
   bool g17p_wide_ms_to_memory = false;
   if (dev->dev.chip == AGX_CHIP_G17P && render->tilebuffer.nr_samples > 1) {
      g17p_wide_ms_to_memory = render->view_mask != 0;

      for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
         const VkRenderingAttachmentInfo *att =
            &pRenderingInfo->pColorAttachments[i];

         g17p_wide_ms_to_memory |= att->imageView != VK_NULL_HANDLE &&
                                   att->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD &&
                                   is_attachment_stored(att);
      }
   }
   const uint8_t max_sample_size_B =
      dev->dev.chip != AGX_CHIP_G17P ? 0
      : g17p_wide_ms_to_memory
           ? AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE
           : AGX_G17P_MAX_BYTES_PER_SAMPLE;
   render->tilebuffer = agx_build_tilebuffer_layout_with_limits(
      formats, render->color_att_count, render->tilebuffer.nr_samples, layered,
      max_sample_size_B,
      native_tile_limits ? 32768 : 0, native_tile_limits ? 2048 : 0);

   if (dev->dev.chip == AGX_CHIP_G17P)
      render->g17p_rt0_layout_supported =
         hk_agx3_g17p_rt0_tile_layout_supported(&render->tilebuffer);

   if (dev->dev.chip == AGX_CHIP_G17P &&
       !render->g17p_rt0_layout_supported) {
      char names[256] = "";
      for (unsigned i = 0, len = 0; i < render->color_att_count; ++i) {
         len += snprintf(names + len, sizeof(names) - MIN2(len, sizeof(names)),
                         "%s%s", i ? "," : "",
                         util_format_short_name(formats[i]));
         len = MIN2(len, sizeof(names));
      }
      mesa_loge_once("G17P tilebuffer layout refused: attachments=%u "
                     "samples=%u sample_size=%u tile_size=%u formats=%s",
                     render->color_att_count, render->tilebuffer.nr_samples,
                     render->tilebuffer.sample_size_B,
                     render->tilebuffer.tile_size, names);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   const VkRenderingAttachmentLocationInfoKHR ral_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
      .colorAttachmentCount = pRenderingInfo->colorAttachmentCount,
   };
   vk_cmd_set_rendering_attachment_locations(&cmd->vk, &ral_info);

   hk_cmd_buffer_dirty_render_pass(cmd);

   /* Determine whether the render area is complete, enabling us to use a
    * fast-clear.
    *
    * TODO: If it is incomplete but tile aligned, it should be possibly to fast
    * clear with the appropriate settings. This is critical for performance.
    */
   bool incomplete_render_area =
      render->area.offset.x > 0 || render->area.offset.y > 0 ||
      render->area.extent.width < render->cr.width ||
      render->area.extent.height < render->cr.height ||
      (render->view_mask &&
       render->view_mask != BITFIELD64_MASK(render->cr.layers));

   perf_debug(cmd, "Rendering %ux%ux%u@%u %s%s", render->cr.width,
              render->cr.height, render->cr.layers,
              render->tilebuffer.nr_samples,
              render->view_mask ? " multiview" : "",
              incomplete_render_area ? " incomplete" : "");

   struct hk_g17p_bg_eot_cache_entry *bg_eot_cache =
      hk_g17p_bg_eot_cache_get(cmd, pRenderingInfo,
                               incomplete_render_area);

   render->cr.bg.main = hk_build_bg_eot(cmd, pRenderingInfo, false, false,
                                        incomplete_render_area, bg_eot_cache);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   render->cr.bg.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, false, true,
                      incomplete_render_area, bg_eot_cache);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   render->cr.eot.main =
      hk_build_bg_eot(cmd, pRenderingInfo, true, false,
                      incomplete_render_area, bg_eot_cache);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   render->cr.eot.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, true, true,
                      incomplete_render_area, bg_eot_cache);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   render->cr.isp_bgobjvals = 0x300;

   const VkRenderingAttachmentInfo *attach_z = pRenderingInfo->pDepthAttachment;
   const VkRenderingAttachmentInfo *attach_s =
      pRenderingInfo->pStencilAttachment;

   struct ail_layout *z_layout = NULL, *s_layout = NULL;

   if (attach_z != NULL && attach_z != VK_NULL_HANDLE && attach_z->imageView) {
      struct hk_image_view *view = render->depth_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      z_layout = &image->planes[0].layout;

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      const struct util_format_description *desc =
         util_format_description(hk_format_to_pipe_format(view->vk.format));

      assert(desc->format == PIPE_FORMAT_Z32_FLOAT ||
             desc->format == PIPE_FORMAT_Z16_UNORM ||
             desc->format == PIPE_FORMAT_Z16_UNORM_S8_UINT ||
             desc->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT);

      render->cr.depth.buffer =
         hk_image_base_address(image, 0) +
         ail_get_layer_level_B(z_layout, first_layer, level);

      /* Main stride in pages */
      assert((z_layout->depth_px == 1 ||
              util_is_aligned(z_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned Z layers");

      unsigned stride_pages = z_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.depth.stride = ((stride_pages - 1) << 14) | 1;

      /* A compressed single-layer image is not page padded, so a small one
       * has a zero-page layer stride and the expression above underflows.
       * The stride is unused for one layer; Apple's G17P Framebuffer writes
       * 0 for a sub-cacheline layer and cacheline units below one page, a
       * form the G17P kernel's ZLS validation does not accept. Send 0. */
      if (dev->dev.chip == AGX_CHIP_G17P && stride_pages == 0) {
         assert(z_layout->depth_px == 1);
         render->cr.depth.stride = 0;
      }

      assert(z_layout->tiling != AIL_TILING_LINEAR && "must tile");

      if (z_layout->compressed) {
         render->cr.depth.meta =
            hk_image_base_address(image, 0) + z_layout->metadata_offset_B +
            (first_layer * z_layout->compression_layer_stride_B) +
            z_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(util_is_aligned(z_layout->compression_layer_stride_B,
                                AIL_CACHELINE) &&
                "Cacheline aligned Z meta layers");

         unsigned stride_lines =
            z_layout->compression_layer_stride_B / AIL_CACHELINE;
         render->cr.depth.meta_stride = (stride_lines - 1) << 14;
      }

      float clear_depth = attach_z->clearValue.depthStencil.depth;

      if (z_layout->format == PIPE_FORMAT_Z16_UNORM) {
         render->cr.isp_bgobjdepth = _mesa_float_to_unorm(clear_depth, 16);
      } else {
         render->cr.isp_bgobjdepth = fui(clear_depth);
      }
   }

   if (attach_s != NULL && attach_s != VK_NULL_HANDLE && attach_s->imageView) {
      struct hk_image_view *view = render->stencil_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      /* Stencil is always the last plane (possibly the only plane) */
      unsigned plane = image->plane_count - 1;
      s_layout = &image->planes[plane].layout;
      assert(s_layout->format == PIPE_FORMAT_S8_UINT);

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      render->cr.stencil.buffer =
         hk_image_base_address(image, plane) +
         ail_get_layer_level_B(s_layout, first_layer, level);

      /* Main stride in pages */
      assert((s_layout->depth_px == 1 ||
              util_is_aligned(s_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned S layers");
      unsigned stride_pages = s_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.stencil.stride = ((stride_pages - 1) << 14) | 1;
      if (dev->dev.chip == AGX_CHIP_G17P && stride_pages == 0) {
         assert(s_layout->depth_px == 1);
         render->cr.stencil.stride = 0;
      }

      if (s_layout->compressed) {
         render->cr.stencil.meta =
            hk_image_base_address(image, plane) + s_layout->metadata_offset_B +
            (first_layer * s_layout->compression_layer_stride_B) +
            s_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(util_is_aligned(s_layout->compression_layer_stride_B,
                                AIL_CACHELINE) &&
                "Cacheline aligned S meta layers");

         unsigned stride_lines =
            s_layout->compression_layer_stride_B / AIL_CACHELINE;

         render->cr.stencil.meta_stride = (stride_lines - 1) << 14;
      }

      render->cr.isp_bgobjvals |= attach_s->clearValue.depthStencil.stencil;
   }

   hk_pack_zls_control(&render->cr.zls_control, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, false);

   hk_pack_zls_control(&render->cr.zls_control_partial, z_layout, s_layout,
                       attach_z, attach_s, incomplete_render_area, true);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   if (cmd->state.gfx.descriptors.root.draw.view_index != 0) {
      cmd->state.gfx.descriptors.root.draw.view_index = 0;
      cmd->state.gfx.descriptors.root_dirty = true;
   }

   if (render->flags & VK_RENDERING_RESUMING_BIT)
      return;

   /* The first control stream of the render pass is special since it gets
    * the clears. Create it and swap in the clear.
    */
   assert(!cmd->current_cs.gfx && "not already in a render pass");
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   if (!cs)
      return;

   cs->cr.bg.main = render->cr.bg.main;
   cs->cr.zls_control = render->cr.zls_control;

   /* Reordering barrier for post-gfx, in case we had any. */
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);

   /* Don't reorder compute across render passes.
    *
    * TODO: Check if this is necessary if the proper PipelineBarriers are
    * handled... there may be CTS bugs...
    */
   hk_cmd_buffer_end_compute(cmd);

   /* If we spill colour attachments, we need to decompress them. This happens
    * at the start of the render; it is not re-emitted when resuming
    * secondaries. It could be hoisted to the start of the command buffer but
    * we're not that clever yet.
    */
   if (agx_tilebuffer_spills(&render->tilebuffer)) {
      const unsigned descriptor_generation =
         agx_chip_descriptor_generation(dev->dev.chip);
      if (!hk_native_pbe_hardware_supported(descriptor_generation)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      perf_debug(cmd, "eMRT render pass");

      for (unsigned i = 0; i < render->color_att_count; ++i) {
         struct hk_image_view *view = render->color_att[i].iview;
         if (view) {
            if (!view->planes[0].emrt_pbe_valid) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FORMAT_NOT_SUPPORTED);
               return;
            }

            struct hk_image *image =
               container_of(view->vk.image, struct hk_image, vk);

            /* TODO: YCbCr interaction? */
            uint8_t plane = 0;
            uint8_t image_plane = view->planes[plane].image_plane;
            struct ail_layout *layout = &image->planes[image_plane].layout;

            if (ail_is_level_logically_compressed(layout,
                                                  view->vk.base_mip_level)) {

               perf_debug(cmd, "Decompressing in-place");

               /* libagx_decompress has no G17P binary. G17P only compresses
                * 1-8 byte single-sample targets, which cannot spill (8 x 8 B
                * fit the 120 B/sample tilebuffer), so this is unreachable;
                * fail closed with a diagnostic rather than a NULL program. */
               if (dev->dev.chip == AGX_CHIP_G17P) {
                  mesa_loge_once("G17P: spilled render pass with a compressed "
                                 "attachment (%s) is unsupported",
                                 util_format_short_name(layout->format));
                  vk_command_buffer_set_error(&cmd->vk,
                                              VK_ERROR_FEATURE_NOT_PRESENT);
                  return;
               }

               unsigned level = view->vk.base_mip_level;
               unsigned layer = view->vk.base_array_layer;
               uint64_t base = hk_image_base_address(image, image_plane);

               struct libagx_decompress_images imgs;
               memcpy(&imgs.compressed, &view->planes[plane].emrt_texture,
                      sizeof(imgs.compressed));
               memcpy(&imgs.uncompressed, &view->planes[plane].emrt_pbe,
                      sizeof(imgs.uncompressed));

               struct agx_grid grid =
                  agx_3d(ail_metadata_width_tl(layout, level) * 32,
                         ail_metadata_height_tl(layout, level), layer_count);

               libagx_decompress(cmd, grid, AGX_BARRIER_ALL | AGX_PREGFX,
                                 layout, layer, level, base,
                                 hk_pool_upload(cmd, &imgs, sizeof(imgs), 64));
            }
         }
      }
   }

   uint32_t clear_count = 0;
   VkClearAttachment clear_att[HK_MAX_RTS + 1];
   bool resolved_clear = false;

   for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *att_info =
         &pRenderingInfo->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE ||
          att_info->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
         continue;

      clear_att[clear_count++] = (VkClearAttachment){
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .colorAttachment = i,
         .clearValue = att_info->clearValue,
      };

      render->color_att[i].clear = true;

      static_assert(sizeof(render->color_att[i].clear_colour) ==
                    sizeof(att_info->clearValue));
      memcpy(render->color_att[i].clear_colour, &att_info->clearValue,
             sizeof(att_info->clearValue));

      resolved_clear |= is_attachment_stored(att_info);
   }

   const uint32_t color_clear_count = clear_count;

   clear_att[clear_count] = (VkClearAttachment){
      .aspectMask = 0,
   };

   if (attach_z && attach_z->imageView != VK_NULL_HANDLE &&
       attach_z->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
      clear_att[clear_count].clearValue.depthStencil.depth =
         attach_z->clearValue.depthStencil.depth;

      render->depth_att.clear = true;
      render->depth_att.clear_colour[0] =
         fui(attach_z->clearValue.depthStencil.depth);

      resolved_clear |= is_attachment_stored(attach_z);
   }

   if (attach_s != NULL && attach_s->imageView != VK_NULL_HANDLE &&
       attach_s->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      clear_att[clear_count].clearValue.depthStencil.stencil =
         attach_s->clearValue.depthStencil.stencil;

      render->stencil_att.clear = true;
      render->stencil_att.clear_colour[1] =
         attach_s->clearValue.depthStencil.stencil;

      resolved_clear |= is_attachment_stored(attach_s);
   }

   if (clear_att[clear_count].aspectMask != 0)
      clear_count++;

   const bool clear_spills = dev->dev.chip == AGX_CHIP_G17P &&
      agx_tilebuffer_spills(&render->tilebuffer);

   /* A G17P multisample tilebuffer wider than eight bytes per sample cannot
    * survive a tile that no colour-writing geometry covered: see
    * AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE. Emit the load-op clear as
    * an ordinary full-area draw instead. That covers every tile, so the
    * background program only ever runs on a tile the hardware also
    * rasterizes, which keeps the whole sample. The target stays in the
    * tilebuffer. */
   const bool clear_wide_multisample =
      dev->dev.chip == AGX_CHIP_G17P && color_clear_count > 0 &&
      render->tilebuffer.nr_samples > 1 &&
      render->tilebuffer.sample_size_B >
         AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE;
   if (clear_spills && !incomplete_render_area && !clear_wide_multisample) {
      unsigned selected = 0;
      for (unsigned i = 0; i < clear_count; ++i) {
         if (clear_att[i].aspectMask == VK_IMAGE_ASPECT_COLOR_BIT &&
             render->tilebuffer.spilled[clear_att[i].colorAttachment])
            clear_att[selected++] = clear_att[i];
      }
      clear_count = selected;
   }
   cs->cr.process_empty_tiles = resolved_clear;
   if (clear_count > 0 &&
       (incomplete_render_area || clear_spills || clear_wide_multisample)) {
      const VkClearRect clear_rect = {
         .rect = render->area,
         .baseArrayLayer = 0,
         .layerCount = render->view_mask ? 1 : render->layer_count,
      };

      hk_CmdClearAttachments(hk_cmd_buffer_to_handle(cmd), clear_count,
                             clear_att, 1, &clear_rect);
   } else {
      /* If a tile is empty, we do not want to process it, as the redundant
       * roundtrip of memory-->tilebuffer-->memory wastes a tremendous amount of
       * memory bandwidth. Any draw marks a tile as non-empty, so we only need
       * to process empty tiles if the background+EOT programs have a side
       * effect. This is the case exactly when there is an attachment we are
       * fast clearing and then storing.
       */
      cs->cr.process_empty_tiles = resolved_clear;
   }

   hk_cmd_prepare_bound_graphics(cmd);

   if (dev->g17p_shader_perf.prefault_main)
      hk_shader_perf_prefault_main_upload(cmd);
   if (dev->g17p_shader_perf.prefault_ioto)
      hk_shader_perf_prefault_ioto(cmd);
   if (dev->g17p_shader_perf.prefault_streams) {
      hk_shader_perf_prefault_stream(cmd->current_cs.gfx);
      hk_shader_perf_prefault_stream(cmd->current_cs.pre_gfx);
      hk_shader_perf_prefault_stream(cmd->current_cs.post_gfx);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndRendering(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* The last control stream of the render pass is special since it gets its
    * stores dropped. Swap it in.
    */
   struct hk_cs *cs = cmd->current_cs.gfx;
   if (cs) {
      cs->cr.eot.main = render->cr.eot.main;
   }

   perf_debug(cmd, "End rendering");
   hk_cmd_buffer_end_graphics(cmd);

   bool need_resolve = false;

   /* Translate render state back to VK for meta */
   VkRenderingAttachmentInfo vk_color_att[HK_MAX_RTS];
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE &&
          !render->g17p_direct_color_resolve)
         need_resolve = true;

      vk_color_att[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = hk_image_view_to_handle(render->color_att[i].iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .resolveMode = render->g17p_direct_color_resolve
                           ? VK_RESOLVE_MODE_NONE
                           : render->color_att[i].resolve_mode,
         .resolveImageView =
            hk_image_view_to_handle(render->color_att[i].resolve_iview),
         .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
   }

   const VkRenderingAttachmentInfo vk_depth_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(render->depth_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->depth_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->depth_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingAttachmentInfo vk_stencil_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = hk_image_view_to_handle(render->stencil_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->stencil_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->stencil_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingInfo vk_render = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = vk_color_att,
      .pDepthAttachment = &vk_depth_att,
      .pStencilAttachment = &vk_stencil_att,
   };

   if (render->flags & VK_RENDERING_SUSPENDING_BIT)
      need_resolve = false;

   memset(render, 0, sizeof(*render));

   if (need_resolve) {
      perf_debug(cmd, "Resolving render pass, colour store op %u",
                 vk_color_att[0].storeOp);

      hk_meta_resolve_rendering(cmd, &vk_render);
   }
}

static void
hk_init_heap(const void *data)
{
   struct hk_device *dev = (struct hk_device *)data;

   perf_debug_dev(&dev->dev, "Allocating heap");

   size_t size = 128 * 1024 * 1024;
   dev->heap = agx_bo_create(&dev->dev, size, 0, 0, "Geometry heap");

   /* The geometry state buffer is initialized here and then is treated by
    * the CPU as rodata, even though the GPU uses it for scratch internally.
    */
   struct poly_heap *map = agx_bo_map(dev->rodata.heap_bo);

   *map = (struct poly_heap){
      .base = dev->heap->va->addr,
      .size = size,
   };
}

/*
 * Allocating the geometry heap is a 128 MB agx_bo_create(), measured at 15.5 ms
 * on G17P: the kernel has to find and zero the pages.  Every consumer of the
 * heap is reached from a draw, so leaving the allocation to util_call_once()
 * put those 15.5 ms inside the first vkCmdDraw() that used geometry or
 * tessellation -- a stall in the middle of a render pass, and ~150x the cost of
 * the same draw once warm (median 105 us).
 *
 * Nothing about the heap depends on the draw, only on the device, so the
 * allocation can be forced as soon as we know the client has a stage that will
 * need it.  hk_heap() keeps the util_call_once(), so correctness never depends
 * on having predicted the stage.
 */
void
hk_device_ensure_geometry_heap(struct hk_device *dev)
{
   util_call_once_data(&dev->heap_init_once, hk_init_heap, dev);
}

static uint64_t
hk_heap(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   util_call_once_data(&dev->heap_init_once, hk_init_heap, dev);

   /* We need to free all allocations after each command buffer execution */
   if (!cmd->uses_heap) {
      perf_debug(cmd, "Freeing heap");
      uint64_t addr = dev->rodata.heap;

      /* Zeroing the allocated index frees everything */
      hk_queue_write(cmd, addr + offsetof(struct poly_heap, bottom), 0,
                     true /* after gfx */);

      cmd->uses_heap = true;
   }

   return dev->rodata.heap;
}

static uint64_t
hk_index_buffer(uint64_t index_buffer, uint size_el, uint offset_el,
                uint elsize_B)
{
   if (offset_el < size_el)
      return index_buffer + (offset_el * elsize_B);
   else
      return AGX_ZERO_PAGE_ADDRESS;
}

static uint64_t
hk_upload_vertex_params(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_vertex_params params;
   poly_vertex_params_init(&params, 0, wg_size);

   /* XXX: We should deduplicate this logic */
   bool indirect = agx_is_indirect(draw.b) || draw.restart;

   if (!indirect)
      poly_vertex_params_set_draw(&params, draw.b.count[0], draw.b.count[1]);

   if (draw.indexed) {
      unsigned index_size_B = agx_index_size_to_B(draw.index_size);
      unsigned range_el = agx_draw_index_range_el(draw);

      params.index_buffer = hk_index_buffer(agx_draw_index_buffer(draw),
                                            range_el, 0, index_size_B);

      params.index_size_B = index_size_B;
      params.index_buffer_range_el = range_el;
   }

   if (gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY]) {

      struct hk_shader *vs = hk_bound_sw_vs(gfx);
      params.outputs = vs->b.info.outputs;

      if (!indirect) {
         uint32_t verts = draw.b.count[0], instances = draw.b.count[1];
         unsigned vb_size =
            poly_tcs_in_size(verts * instances, vs->b.info.outputs);

         /* Allocate if there are any outputs, or use the null sink to trap
          * reads if there aren't. Those reads are undefined but should not
          * fault. Affects:
          *
          *    dEQP-VK.pipeline.monolithic.no_position.explicit_declarations.basic.single_view.v0_g1
          */
         params.output_buffer = vb_size ? hk_pool_alloc(cmd, vb_size, 4).gpu
                                        : AGX_SCRATCH_PAGE_ADDRESS;
      }
   }

   desc->root.draw.vertex_outputs = params.outputs;

   struct agx_ptr target = gfx->prepared_vertex_params.gpu
      ? gfx->prepared_vertex_params
      : hk_pool_alloc(cmd, sizeof(params), 8);
   if (!target.cpu)
      return 0;
   memcpy(target.cpu, &params, sizeof(params));
   gfx->prepared_vertex_params = (struct agx_ptr){0};
   gfx->prepared_vertex_params_filled = false;
   return target.gpu;
}

/* Finalize the prepared transformed-VS block without rebuilding its immutable
 * shader-owned fields.  Direct draws author only the count/index tuple and an
 * output allocation; indirect/restart setup remains on the general path. */
static uint64_t
hk_g17p_fill_prepared_direct_vertex_params(struct hk_cmd_buffer *cmd,
                                           struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   if (agx_is_indirect(draw.b) || draw.restart ||
       !gfx->prepared_vertex_params_filled ||
       !gfx->prepared_vertex_params.gpu ||
       !gfx->prepared_vertex_params.cpu)
      return 0;

   struct poly_vertex_params *params = gfx->prepared_vertex_params.cpu;
   poly_vertex_params_set_draw(params, draw.b.count[0], draw.b.count[1]);

   if (draw.indexed) {
      const unsigned index_size_B = agx_index_size_to_B(draw.index_size);
      const unsigned range_el = agx_draw_index_range_el(draw);
      params->index_buffer = hk_index_buffer(agx_draw_index_buffer(draw),
                                             range_el, 0, index_size_B);
      params->index_size_B = index_size_B;
      params->index_buffer_range_el = range_el;
   }

   const unsigned vb_size = poly_tcs_in_size(
      draw.b.count[0] * draw.b.count[1], params->outputs);
   params->output_buffer = vb_size ? hk_pool_alloc(cmd, vb_size, 4).gpu
                                   : AGX_SCRATCH_PAGE_ADDRESS;

   const uint64_t target = gfx->prepared_vertex_params.gpu;
   gfx->prepared_vertex_params = (struct agx_ptr){0};
   gfx->prepared_vertex_params_filled = false;
   return target;
}

static enum mesa_prim
hk_gs_in_prim(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];

   if (tes != NULL)
      return gfx->tess.prim;
   else
      return vk_topology_to_mesa(dyn->ia.primitive_topology);
}

static enum mesa_prim
hk_rast_prim(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   if (gs != NULL) {
      return gs->variants[HK_GS_VARIANT_RAST].info.gs.mode;
   } else {
      switch (dyn->ia.primitive_topology) {
      case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
      case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
         return MESA_PRIM_LINES;
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
         return MESA_PRIM_TRIANGLES;
      default:
         return hk_gs_in_prim(cmd);
      }
   }
}

DEBUG_GET_ONCE_BOOL_OPTION(hk_g17p_ordinary_variant_trace,
                          "HK_G17P_ORDINARY_VARIANT_TRACE", false)

static struct hk_shader *
hk_bound_fs(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *obj = gfx->shaders[MESA_SHADER_FRAGMENT];
   unsigned samples = cmd->vk.dynamic_graphics_state.ms.rasterization_samples;
   if (!obj || !obj->g17p_ordinary_variants)
      return hk_fs_variant(obj, samples);

   bool changed;
   struct hk_shader *selected = hk_ordinary_descriptor_select_push(
      &gfx->ordinary_fs, obj, gfx->descriptors.sets, gfx->descriptors.root.sets,
      gfx->descriptors.push, gfx->descriptors.push_active,
      HK_MAX_SETS, gfx->descriptors.generation, samples, &changed);
   if (changed)
      gfx->shaders_dirty |= BITFIELD_BIT(MESA_SHADER_FRAGMENT);
   if (changed && debug_get_option_hk_g17p_ordinary_variant_trace())
      fprintf(stderr, "G17P ordinary variant: selected=%u generation=%" PRIu64
                      " main=%u\n",
              (unsigned)(selected - obj->variants), gfx->descriptors.generation,
              selected->b.info.main_size);
   return selected;
}

static uint64_t
hk_upload_geometry_params(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct hk_shader *fs = hk_bound_fs(cmd);
   struct hk_shader *count = hk_count_gs_variant(gs);

   /* XXX: We should deduplicate this logic */
   bool indirect = agx_is_indirect(draw.b) ||
                   gfx->shaders[MESA_SHADER_TESS_EVAL] || draw.restart;
   enum mesa_prim mode = hk_gs_in_prim(cmd);

   if (draw.restart) {
      mode = u_decomposed_prim(mode);
   }

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_geometry_params params;
   poly_geometry_params_init(&params, mode, wg_size);

   params.flat_outputs = fs->info.fs.interp.flat;

   if (gfx->xfb_enabled) {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb); ++i) {
         params.xfb_base_original[i] = gfx->xfb[i].addr;
         params.xfb_size[i] = gfx->xfb[i].range;
         params.xfb_offs_ptrs[i] = gfx->xfb_offsets + i * sizeof(uint32_t);
      }
   } else {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb); ++i) {
         params.xfb_offs_ptrs[i] = AGX_ZERO_PAGE_ADDRESS;
         params.xfb_base_original[i] = AGX_ZERO_PAGE_ADDRESS;
         params.xfb_size[i] = 0;
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb_query); ++i) {
      uint64_t q = gfx->xfb_query[i];

      if (q) {
         params.xfb_prims_generated_counter[i] = q;
         params.prims_generated_counter[i] = q + sizeof(uint64_t);
      } else {
         params.xfb_prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
         params.prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
      }

      /* TODO: Optimize out? */
      params.xfb_overflow[i] = AGX_SCRATCH_PAGE_ADDRESS;
   }

   /* TODO: Optimize out? */
   params.xfb_any_overflow = AGX_SCRATCH_PAGE_ADDRESS;

   /* Calculate input primitive count for direct draws, and allocate the vertex
    * & count buffers. GPU calculates and allocates for indirect draws.
    */
   params.count_buffer_stride = count->info.gs.count_words * 4;

   if (!count->info.gs.prefix_sum && params.count_buffer_stride) {
      struct agx_ptr T = hk_pool_alloc(cmd, 16, 4);
      memset(T.cpu, 0, 16);
      params.count_buffer = T.gpu;
   }

   struct poly_gs_info *gsi = &count->info.gs;

   if (indirect) {
      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         /* Need to allocate heap if we haven't yet */
         hk_heap(cmd);

         cmd->geom_index_buffer = dev->heap->va->addr;
         cmd->geom_index_count = dev->heap->size;
      } else {
         cmd->geom_index_count =
            poly_gs_rast_vertices(gsi->shape, gsi->max_indices, 1, 0);
      }
   } else {
      poly_geometry_params_set_draw(&params, mode, gsi->shape, gsi->max_indices,
                                    draw.b.count[0], draw.b.count[1]);

      unsigned size = params.input_primitives * params.count_buffer_stride;
      if (count->info.gs.prefix_sum && size) {
         params.count_buffer = hk_pool_alloc(cmd, size, 4).gpu;
      }

      cmd->geom_index_count = params.draw.index_count;
      cmd->geom_instance_count = params.draw.instance_count;

      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         params.output_index_buffer =
            hk_pool_alloc(cmd, cmd->geom_index_count * 4, 4).gpu;

         cmd->geom_index_buffer = params.output_index_buffer;
      }
   }

   if (gsi->shape == POLY_GS_SHAPE_STATIC_INDEXED) {
      cmd->geom_index_buffer =
         hk_pool_upload(cmd, count->info.gs.topology, gsi->max_indices * 4, 4);
   }

   struct agx_ptr target = gfx->prepared_geometry_params.gpu
      ? gfx->prepared_geometry_params
      : hk_pool_alloc(cmd, sizeof(params), 8);
   if (!target.cpu)
      return 0;
   memcpy(target.cpu, &params, sizeof(params));
   gfx->prepared_geometry_params = (struct agx_ptr){0};
   gfx->prepared_geometry_params_filled = false;
   return target.gpu;
}

static void
hk_init_tess_params(struct hk_cmd_buffer *cmd, struct poly_tess_params *out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *tcs = hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);

   enum poly_tess_partitioning partitioning =
      gfx->tess.info.spacing == TESS_SPACING_EQUAL
         ? POLY_TESS_PARTITIONING_INTEGER
      : gfx->tess.info.spacing == TESS_SPACING_FRACTIONAL_ODD
         ? POLY_TESS_PARTITIONING_FRACTIONAL_ODD
         : POLY_TESS_PARTITIONING_FRACTIONAL_EVEN;

   *out = (struct poly_tess_params){
      .heap = hk_heap(cmd),
      .tcs_stride_el = tcs->info.tess.tcs_output_stride / 4,
      .statistic = hk_pipeline_stat_addr(
         cmd,
         VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT),

      .input_patch_size = dyn->ts.patch_control_points,
      .output_patch_size = tcs->info.tess.tcs_output_patch_size,
      .tcs_patch_constants = tcs->info.tess.tcs_nr_patch_outputs,
      .tcs_per_vertex_outputs = tcs->info.tess.tcs_per_vertex_outputs,
      .partitioning = partitioning,
      .points_mode = gfx->tess.info.points,
      .isolines = gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES,
   };

   if (!out->points_mode && gfx->tess.info.mode != TESS_PRIMITIVE_ISOLINES) {
      out->ccw = gfx->tess.info.ccw;
      out->ccw ^=
         dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }

   /* heap is allocated by hk_heap */
   out->patch_coord_buffer = dev->heap->va->addr;
}

static void
hk_upload_tess_params(struct hk_cmd_buffer *cmd, struct poly_tess_params *out,
                      struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct agx_ptr prepared_out_draws = gfx->prepared_tess_out_draws;

   if (!gfx->prepared_tess_params_filled)
      hk_init_tess_params(cmd, out);

   uint32_t draw_stride_el = 5;
   size_t draw_stride_B = draw_stride_el * sizeof(uint32_t);

   if (!agx_is_indirect(draw.b)) {
      unsigned in_patches = draw.b.count[0] / out->input_patch_size;
      unsigned unrolled_patches = in_patches * draw.b.count[1];

      uint32_t alloc = 0;
      uint32_t tcs_out_offs = alloc;
      alloc += unrolled_patches * out->tcs_stride_el * sizeof(uint32_t);

      uint32_t patch_coord_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      uint32_t count_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      /* Single API draw */
      uint32_t draw_offs = alloc;
      alloc += draw_stride_B;

      struct agx_ptr blob = hk_pool_alloc(cmd, alloc, 4);
      out->tcs_buffer = blob.gpu + tcs_out_offs;
      out->patches_per_instance = in_patches;
      out->coord_allocs = blob.gpu + patch_coord_offs;
      out->nr_patches = unrolled_patches;
      out->out_draws = prepared_out_draws.gpu
         ? prepared_out_draws.gpu
         : blob.gpu + draw_offs;
      out->counts = blob.gpu + count_offs;
   } else {
      /* Allocate 3x indirect global+local grids for VS/TCS/tess */
      uint32_t grid_stride = sizeof(uint32_t) * 6;
      gfx->tess.grids = hk_pool_alloc(cmd, grid_stride * 3, 4).gpu;

      out->out_draws = prepared_out_draws.gpu
         ? prepared_out_draws.gpu
         : hk_pool_alloc(cmd, draw_stride_B, 4).gpu;
   }

   gfx->tess.out_draws = out->out_draws;
   gfx->prepared_tess_out_draws = (struct agx_ptr){0};
   gfx->prepared_tess_params_filled = false;
}

static struct hk_api_shader *
hk_build_meta_shader_locked(struct hk_device *dev, struct hk_internal_key *key,
                            hk_internal_builder_t builder)
{
   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(dev->kernels.ht, key);
   if (ent)
      return ent->data;

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, NULL);
   builder(&b, key->key);

   const struct vk_pipeline_robustness_state rs = {
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
   };

   struct vk_shader_compile_info info = {
      .stage = b.shader->info.stage,
      .nir = b.shader,
      .robustness = &rs,
   };

   hk_preprocess_nir_internal(dev->vk.physical, b.shader);

   struct hk_api_shader *s;
   if (hk_compile_shader(dev, &info, NULL, NULL, NULL, &s) != VK_SUCCESS)
      return NULL;

   /* ..and cache it before we return. The key is on the stack right now, so
    * clone it before using it as a hash table key. The clone is logically owned
    * by the hash table.
    */
   size_t total_key_size = sizeof(*key) + key->key_size;
   void *cloned_key = ralloc_memdup(dev->kernels.ht, key, total_key_size);

   _mesa_hash_table_insert(dev->kernels.ht, cloned_key, s);
   return s;
}

struct hk_api_shader *
hk_meta_shader(struct hk_device *dev, hk_internal_builder_t builder, void *data,
               size_t data_size)
{
   size_t total_key_size = sizeof(struct hk_internal_key) + data_size;

   struct hk_internal_key *key = alloca(total_key_size);
   memset(key, 0, total_key_size);
   key->builder = builder;
   key->key_size = data_size;

   if (data_size)
      memcpy(key->key, data, data_size);

   simple_mtx_lock(&dev->kernels.lock);
   struct hk_api_shader *s = hk_build_meta_shader_locked(dev, key, builder);
   simple_mtx_unlock(&dev->kernels.lock);

   return s;
}

static struct agx_draw
hk_draw_as_indexed_indirect(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   assert(draw.indexed);

   if (agx_is_indirect(draw.b))
      return draw;

   VkDrawIndexedIndirectCommand desc = {
      .indexCount = draw.b.count[0],
      .instanceCount = draw.b.count[1],
      .firstIndex = draw.start,
      .vertexOffset = draw.index_bias,
      .firstInstance = draw.start_instance,
   };

   return agx_draw_indexed_indirect(
      hk_pool_upload(cmd, &desc, sizeof(desc), 4), draw.index_buffer,
      draw.index_buffer_range_B, draw.index_size, draw.restart);
}

static struct agx_draw
hk_draw_without_restart(struct hk_cmd_buffer *cmd, struct agx_draw draw,
                        uint32_t draw_count)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   perf_debug(cmd, "Unrolling primitive restart due to GS/XFB");

   /* The unroll kernel assumes an indirect draw. Synthesize one if needed */
   draw = hk_draw_as_indexed_indirect(cmd, draw);

   /* Next, we unroll the index buffer used by the indirect draw */
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

   assert(draw_count == 1 && "TODO: multidraw");

   struct libagx_unroll_restart_args ia = {
      .heap = hk_heap(cmd),
      .index_buffer = draw.index_buffer,
      .in_draw = draw.b.ptr,
      .out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t) * draw_count, 4).gpu,
      .max_draws = 1 /* TODO: MDI */,
      .restart_index = gfx->index.restart,
      .index_buffer_size_el = agx_draw_index_range_el(draw),
      .index_size_log2 = draw.index_size,
      .flatshade_first =
         dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT,
   };

   libagx_unroll_restart_struct(cmd, agx_1d(1024 * draw_count),
                                AGX_BARRIER_ALL | AGX_PREGFX, ia,
                                poly_compact_prim(prim));

   return agx_draw_indexed_indirect(ia.out_draw, dev->heap->va->addr,
                                    dev->heap->size, draw.index_size,
                                    false /* restart */);
}

static __attribute__((hot)) struct agx_draw
hk_launch_gs_prerast(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                     struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct agx_grid grid_vs, grid_gs;

   struct hk_shader *vs = hk_bound_sw_vs_before_gs(gfx);
   struct hk_shader *main = hk_main_gs_variant(gs);
   struct hk_shader *count = hk_count_gs_variant(gs);
   struct hk_shader *pre_gs = hk_pre_gs_variant(gs);

   uint64_t vertex_params = desc->root.draw.vertex_params;
   uint64_t geometry_params = desc->root.draw.geometry_params;
   unsigned count_words = count->info.gs.count_words;
   struct agx_workgroup wg = agx_workgroup(64, 1, 1);

   /* The software-tessellation fast path may already have appended exact GS
    * setup, vertex and main launches in the same unpublished transaction.
    * Only the final VDM draw description remains to be selected here. */
   if (gfx->g17p_sw_tess_batch.gs_emitted) {
      gfx->g17p_sw_tess_batch.gs_emitted = false;
      assert(agx_is_indirect(draw.b));
      goto gs_result;
   }

   if (false /* TODO */)
      perf_debug(cmd, "Transform feedbck");
   else if (count_words)
      perf_debug(cmd, "Geometry shader with counts");
   else
      perf_debug(cmd, "Geometry shader without counts");

   enum mesa_prim mode = hk_gs_in_prim(cmd);

   if (draw.restart) {
      draw = hk_draw_without_restart(cmd, draw, 1);
      mode = u_decomposed_prim(mode);
   }

   /* Setup grids */
   if (agx_is_indirect(draw.b)) {
      struct libagx_gs_setup_indirect_args gsi = {
         .heap = hk_heap(cmd),
         .index_buffer = draw.index_buffer,
         .draw = draw.b.ptr,
         .vp = desc->root.draw.vertex_params,
         .p = desc->root.draw.geometry_params,
         .vs_outputs = vs->b.info.outputs,
         .prim = mode,
         .is_prefix_summing = count->info.gs.prefix_sum,
         .max_indices = count->info.gs.max_indices,
         .shape = count->info.gs.shape,
      };

      if (draw.indexed) {
         gsi.index_size_B = agx_index_size_to_B(draw.index_size);
         gsi.index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      if (!hk_dispatch_prepared_precomp_g17p(
             cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
             LIBAGX_GS_SETUP_INDIRECT, &gsi, sizeof(gsi),
             &gfx->g17p_precomp.gs_setup)) {
         libagx_gs_setup_indirect_struct(cmd, agx_1d(1),
                                         AGX_BARRIER_ALL | AGX_PREGFX, gsi);
      }

      grid_vs = agx_grid_indirect_local(
         vertex_params + offsetof(struct poly_vertex_params, grid));

      grid_gs = agx_grid_indirect_local(
         geometry_params + offsetof(struct poly_geometry_params, grid));
   } else {
      grid_vs = grid_gs = draw.b;
      grid_gs.count[0] = u_decomposed_prims_for_vertices(mode, draw.b.count[0]);
   }

   /* Launch the vertex shader first */
   if (dev->dev.chip == AGX_CHIP_G17P) {
      if (!hk_dispatch_prepared_g17p_native(
             cmd, cs, vs, &gfx->g17p_dispatch.gs_vs, grid_vs, wg))
         hk_dispatch_with_local_size(cmd, cs, vs, grid_vs, wg);
   } else {
      hk_reserve_scratch(cmd, cs, vs);
      hk_dispatch_with_usc(
         dev, cs, &vs->b.info,
         hk_upload_usc_words(cmd, vs,
                             vs->info.stage == MESA_SHADER_VERTEX
                                ? gfx->linked[MESA_SHADER_VERTEX]
                                : vs->only_linked),
         grid_vs, wg);
   }

   /* Transform feedback and various queries require extra dispatching,
    * determine if we need that here.
    */
   VkQueryPipelineStatisticFlagBits gs_queries =
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;

   struct hk_root_descriptor_table *root = &cmd->state.gfx.descriptors.root;
   bool xfb_or_queries =
      main->info.gs.xfb || (root->draw.pipeline_stats_flags & gs_queries);

   if (xfb_or_queries) {
      /* If we need counts, launch the count shader and prefix sum the results. */
      if (count_words) {
         perf_debug(dev, "Geometry shader count");
         if (!hk_dispatch_prepared_g17p_native(
                cmd, cs, count, &gfx->g17p_dispatch.gs_count, grid_gs, wg))
            hk_dispatch_with_local_size(cmd, cs, count, grid_gs, wg);
      }

      if (count->info.gs.prefix_sum) {
         perf_debug(dev, "Geometry shader transform feedback prefix sum");
         libagx_prefix_sum_geom(cmd, agx_1d(1024 * count_words),
                                AGX_BARRIER_ALL | AGX_PREGFX, geometry_params);
      }

      /* Transform feedback / query program */
      perf_debug(dev, "Transform feedback / geometry query");
      if (!hk_dispatch_prepared_g17p_native(
             cmd, cs, pre_gs, &gfx->g17p_dispatch.gs_pre, agx_1d(1),
             agx_workgroup(1, 1, 1)))
         hk_dispatch_with_local_size(cmd, cs, pre_gs, agx_1d(1),
                                     agx_workgroup(1, 1, 1));
   }

   /* Pre-rast geometry shader */
   if (!hk_dispatch_prepared_g17p_native(
          cmd, cs, main, &gfx->g17p_dispatch.gs_main, grid_gs, wg))
      hk_dispatch_with_local_size(cmd, cs, main, grid_gs, wg);

gs_result:
   if (poly_gs_indexed(count->info.gs.shape)) {
      enum agx_index_size index_size =
         agx_translate_index_size(poly_gs_index_size(count->info.gs.shape));

      if (agx_is_indirect(draw.b)) {
         return agx_draw_indexed_indirect(
            geometry_params + offsetof(struct poly_geometry_params, draw),
            cmd->geom_index_buffer, cmd->geom_index_count, index_size, true);
      } else {
         return agx_draw_indexed(cmd->geom_index_count,
                                 cmd->geom_instance_count, 0, 0, 0,
                                 cmd->geom_index_buffer,
                                 cmd->geom_index_count * 4, index_size, true);
      }
   } else {
      if (agx_is_indirect(draw.b)) {
         return agx_draw_indirect(geometry_params +
                                  offsetof(struct poly_geometry_params, draw));
      } else {
         return (struct agx_draw){
            .b = agx_3d(cmd->geom_index_count, cmd->geom_instance_count, 1),
         };
      }
   }
}

/*
 * G17P tessellates in hardware. The vertex and tessellation control shaders
 * still run as compute kernels, writing the control points and levels, and a
 * kernel turns each patch's levels into the factor record the tessellator
 * reads. The draw itself is then a native Tessellate command that runs the
 * evaluation shader as the post-tessellation vertex program.
 *
 * The tessellator draws every instance's patches as one run, in the order the
 * TCS dispatch unrolled them, so that the evaluation shader addresses the TCS
 * output directly by the tessellator's patch index. An indirect draw only
 * learns its patch count on the GPU: a setup kernel places its buffers in the
 * heap and writes the launch grids and the patch draw the native indirect
 * Tessellate command reads.
 */
static void
hk_launch_tess_g17p(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                    struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_shader *vs = hk_bound_sw_vs(gfx);
   struct hk_shader *tcs = hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);
   const uint64_t params = gfx->descriptors.root.draw.tess_params;
   const bool quads = gfx->tess.info.mode == TESS_PRIMITIVE_QUADS;
   const uint32_t record_B = (quads ? 6 : 4) * sizeof(uint16_t);
   struct agx_grid grid_vs, grid_tcs, grid_factors;
   const bool perf_trace = hk_shader_perf_draw_enabled(dev);
   const void *perf_shader = gfx->shaders[MESA_SHADER_VERTEX];
   static uint32_t perf_calls;
   const uint32_t perf_call = dev->g17p_shader_perf.trace
      ? p_atomic_inc_return(&perf_calls) : 0;
   uint64_t setup_ns = 0, vs_ns = 0, tcs_ns = 0, factors_ns = 0;
   uint64_t phase_start = perf_trace ? os_time_get_nano() : 0;

   /* An evaluation shader declaring isolines or point mode never reaches
    * here: hk_g17p_tess_is_native() sent it to the software tessellator and
    * its bound variant was lowered for that path. What can still reach here
    * is a *control* shader that introduces point mode for an evaluation
    * shader that did not declare it, which Vulkan permits and which the
    * evaluation shader's own lowering could not anticipate. */
   if (!gfx->tess.params || gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES ||
       gfx->tess.info.points) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   uint64_t tcs_stat = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT);

   if (agx_is_indirect(draw.b)) {
      struct agx_ptr patch_draw = hk_pool_alloc(cmd, 4 * sizeof(uint32_t), 4);
      if (!patch_draw.gpu) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }

      struct libagx_tess_setup_indirect_hw_args args = {
         .p = params,
         .grids = gfx->tess.grids,
         .vp = gfx->descriptors.root.draw.vertex_params,
         .indirect = draw.b.ptr,
         .patch_draw = patch_draw.gpu,
         .vertex_outputs = vs->b.info.outputs,
         .factor_record_B = record_B,
         .tcs_statistic = tcs_stat,
      };

      if (draw.indexed) {
         args.in_index_buffer = draw.index_buffer;
         args.in_index_size_B = agx_index_size_to_B(draw.index_size);
         args.in_index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_tess_setup_indirect_hw_struct(cmd, agx_1d(1),
                                           AGX_BARRIER_ALL | AGX_PREGFX, args);

      const uint32_t grid_stride = 6 * sizeof(uint32_t);
      grid_vs = agx_grid_indirect_local(gfx->tess.grids);
      grid_tcs = agx_grid_indirect_local(gfx->tess.grids + grid_stride);
      grid_factors =
         agx_grid_indirect_local(gfx->tess.grids + 2 * grid_stride);

      /* The records are somewhere in the heap, which the tessellator reads
       * as the factor buffer from the draw's patch start. */
      gfx->tess.factor_buffer = dev->heap->va->addr;
      gfx->tess.factor_buffer_size_B = dev->heap->size;
      gfx->tess.patch_draw = patch_draw.gpu;
      gfx->tess.patches = 0;
   } else {
      const uint32_t patches = draw.b.count[0] / dyn->ts.patch_control_points;
      const uint32_t unrolled = patches * draw.b.count[1];

      if (hk_stat_enabled(tcs_stat)) {
         libagx_increment_statistic(cmd, agx_1d(1),
                                    AGX_BARRIER_ALL | AGX_PREGFX, tcs_stat,
                                    patches);
      }

      grid_vs = grid_tcs = draw.b;
      grid_tcs.count[0] = patches * tcs->info.tess.tcs_output_patch_size;
      grid_factors = agx_1d(unrolled);

      struct agx_ptr factors = hk_pool_alloc(cmd, unrolled * record_B, 4);
      if (!factors.cpu) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }

      gfx->tess.factor_buffer = factors.gpu;
      gfx->tess.factor_buffer_size_B = unrolled * record_B;
      gfx->tess.patch_draw = 0;
      gfx->tess.patches = unrolled;

      if (hk_g17p_launch_trace_enabled()) {
         fprintf(stderr,
                 "G17P tess draw: verts=%u instances=%u cps=%u patches=%u "
                 "unrolled=%u record_B=%u factors=%" PRIx64 " size=%u "
                 "vs_grid=%u/%u/%u tcs_grid=%u/%u/%u\n",
                 draw.b.count[0], draw.b.count[1],
                 dyn->ts.patch_control_points, patches, unrolled, record_B,
                 (uint64_t)factors.gpu, unrolled * record_B, grid_vs.count[0],
                 grid_vs.count[1], grid_vs.count[2], grid_tcs.count[0],
                 grid_tcs.count[1], grid_tcs.count[2]);
      }
   }

   if (perf_trace) {
      setup_ns = os_time_get_nano() - phase_start;
      hk_shader_perf_phase_record(perf_shader, 3,
                                  setup_ns);
      phase_start = os_time_get_nano();
   }

   hk_dispatch_with_local_size(cmd, cs, vs, grid_vs, agx_workgroup(64, 1, 1));
   if (perf_trace) {
      vs_ns = os_time_get_nano() - phase_start;
      hk_shader_perf_phase_record(perf_shader, 4,
                                  vs_ns);
      phase_start = os_time_get_nano();
   }
   hk_dispatch_with_local_size(
      cmd, cs, tcs, grid_tcs,
      agx_workgroup(tcs->info.tess.tcs_output_patch_size, 1, 1));

   if (perf_trace) {
      tcs_ns = os_time_get_nano() - phase_start;
      hk_shader_perf_phase_record(perf_shader, 5,
                                  tcs_ns);
      phase_start = os_time_get_nano();
   }

   libagx_tess_factors_half(cmd, grid_factors, AGX_BARRIER_ALL | AGX_PREGFX,
                            params, gfx->tess.factor_buffer, quads);
   if (perf_trace) {
      factors_ns = os_time_get_nano() - phase_start;
      hk_shader_perf_phase_record(perf_shader, 6,
                                  factors_ns);
      if (dev->g17p_shader_perf.trace && perf_call <= 2) {
         fprintf(stderr,
                 "HK_SHADER_PERF tess-call=%u shader=%p setup_ns=%" PRIu64
                 " vs_ns=%" PRIu64 " tcs_ns=%" PRIu64
                 " factors_ns=%" PRIu64 "\n",
                 perf_call, perf_shader, setup_ns, vs_ns, tcs_ns, factors_ns);
      }
   }
}

/*
 * Draw the patches the pre-pass prepared: the draw-state selector when the
 * render state was just published, the tessellator's state record, and a
 * native Tessellate command over the factor records.
 */
static bool
hk_g17p_emit_tessellate(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        uint8_t *out, uint32_t draw_state_offset,
                        bool render_state_emitted)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct hk_tess_info info = gfx->tess.info;

   /* The tessellator orders its output in a lower-left domain: a clockwise
    * triangle there is counter-clockwise in Vulkan's default upper-left
    * domain, so only a lower-left pipeline keeps the shader's winding. */
   const bool ccw =
      info.ccw ==
      (dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT);
   const struct agx_vdm_tessellation_state_info state = {
      .partition = info.spacing == TESS_SPACING_EQUAL
                      ? AGX_VDM_TESSELLATION_PARTITION_INTEGER
                   : info.spacing == TESS_SPACING_FRACTIONAL_ODD
                      ? AGX_VDM_TESSELLATION_PARTITION_FRACTIONAL_ODD
                      : AGX_VDM_TESSELLATION_PARTITION_FRACTIONAL_EVEN,
      .counterclockwise = ccw,
      .max_factor = 64,
      .quads = info.mode == TESS_PRIMITIVE_QUADS,
      .float_factors = false,
      .step = AGX_VDM_TESSELLATION_STEP_PER_PATCH,
      .scale_half = 0x3c00,
   };
   const uint32_t factor_words_minus_one =
      DIV_ROUND_UP(gfx->tess.factor_buffer_size_B, 4) - 1;

   uint8_t state_bytes[AGX_VDM_TESSELLATION_STATE_LENGTH];
   uint8_t draw_bytes[MAX2(AGX_VDM_TESSELLATE_DIRECT_LENGTH,
                           AGX_VDM_TESSELLATE_INDIRECT_LENGTH)];
   size_t draw_B;
   bool packed;
   if (gfx->tess.patch_draw) {
      const struct agx_vdm_tessellate_indirect_info indirect = {
         .factor_buffer = gfx->tess.factor_buffer,
         .indirect_buffer = gfx->tess.patch_draw,
         .factor_buffer_size_minus_one = factor_words_minus_one,
      };
      struct agx_vdm_tessellate_indirect encoded;
      packed = agx_vdm_tessellate_indirect_pack(dev->dev.chip, &indirect,
                                                &encoded);
      memcpy(draw_bytes, encoded.bytes, encoded.length);
      draw_B = encoded.length;
   } else {
      const struct agx_vdm_tessellate_direct_info direct = {
         .factor_buffer = gfx->tess.factor_buffer,
         .patch_count = gfx->tess.patches,
         .instance_count = 1,
         .factor_buffer_size_minus_one = factor_words_minus_one,
      };
      struct agx_vdm_tessellate_direct encoded;
      packed = agx_vdm_tessellate_direct_pack(dev->dev.chip, &direct,
                                              &encoded) &&
               gfx->tess.patches;
      memcpy(draw_bytes, encoded.bytes, encoded.length);
      draw_B = encoded.length;
   }

   const size_t selector_B = render_state_emitted ? sizeof(uint32_t) : 0;
   const size_t total_B = selector_B + sizeof(state_bytes) + draw_B;
   if (!gfx->tess.factor_buffer || !packed ||
       !agx_vdm_tessellation_state_pack(dev->dev.chip, &state, state_bytes) ||
       (size_t)((uint8_t *)cs->end - out) < total_B) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   uint8_t *cursor = out;
   if (render_state_emitted) {
      memcpy(cursor, &draw_state_offset, sizeof(draw_state_offset));
      cursor += sizeof(draw_state_offset);
   }
   memcpy(cursor, state_bytes, sizeof(state_bytes));
   cursor += sizeof(state_bytes);
   memcpy(cursor, draw_bytes, draw_B);
   cursor += draw_B;
   cs->current = cursor;
   return true;
}

static inline void
hk_g17p_materialize_prepared_indirect(
   struct hk_g17p_prepared_dispatch *prepared, uint64_t grid,
   struct agx_workgroup local_size)
{
   memset(prepared->indirect_local_geometry.cpu, 0,
          AGX3_G17P_INDIRECT_GEOMETRY_SIZE);
   struct hk_root_descriptor_table *helper_root =
      prepared->indirect_local_helper_root.cpu;
   helper_root->cs.group_count_addr = grid;
   helper_root->cs.grid_helper_geometry_addr =
      prepared->indirect_local_geometry.gpu;
   helper_root->cs.grid_helper_sparse_map_addr = 0;
   helper_root->cs.grid_helper_local_size[0] = local_size.x;
   helper_root->cs.grid_helper_local_size[1] = local_size.y;
   helper_root->cs.grid_helper_local_size[2] = local_size.z;
   prepared->indirect_local_consumed = true;
}

static __attribute__((hot)) struct agx_draw
hk_g17p_emit_prepared_sw_tess_batch(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, struct hk_shader *vs,
   struct hk_shader *tcs, struct agx_grid grid_vs, struct agx_grid grid_tcs,
   struct agx_grid grid_tess, uint64_t state, uint64_t c_prims,
   uint64_t c_inv, enum libagx_program count_program,
   enum libagx_program final_program)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_g17p_prepared_sw_tess_batch *batch =
      &gfx->g17p_sw_tess_batch;

   if (!batch->valid || batch->vs != vs || batch->tcs != tcs ||
       batch->dynamic_generation !=
          cmd->vk.dynamic_graphics_state.generation ||
       batch->descriptor_generation != gfx->descriptors.generation ||
       batch->root != gfx->root || batch->draw_params != gfx->draw_params ||
       batch->draw_id_ptr != gfx->draw_id_ptr || batch->state != state ||
       batch->c_prims != c_prims || batch->c_invs != c_inv ||
       batch->count_program != count_program ||
       batch->final_program != final_program ||
       hk_cmd_buffer_device(cmd)->dev.chip != AGX_CHIP_G17P ||
       cs->type != HK_CS_CDM || hk_g17p_launch_trace_enabled() ||
       agx_is_indirect(grid_vs) || agx_is_indirect(grid_tcs) ||
       agx_is_indirect(grid_tess) || !grid_vs.count[0] ||
       !grid_vs.count[1] || !grid_vs.count[2] || !grid_tcs.count[0] ||
       !grid_tcs.count[1] || !grid_tcs.count[2] || !grid_tess.count[0] ||
       !grid_tess.count[1] || !grid_tess.count[2])
      return (struct agx_draw){0};

   const struct agx_draw tess_draw = agx_draw_indexed_indirect(
      gfx->tess.out_draws, hk_cmd_buffer_device(cmd)->heap->va->addr,
      hk_cmd_buffer_device(cmd)->heap->size, AGX_INDEX_SIZE_U32, false);
   bool emit_gs = false;
   struct hk_g17p_prepared_dispatch *pgs_vs =
      &gfx->g17p_dispatch.gs_vs;
   struct hk_g17p_prepared_dispatch *pgs_main =
      &gfx->g17p_dispatch.gs_main;
   const struct hk_g17p_prepared_precomp *pgs_setup =
      &gfx->g17p_precomp.gs_setup;
   struct hk_shader *gs_vs = NULL;
   struct hk_shader *gs_main = NULL;
   const struct agx_workgroup gs_local = agx_workgroup(64, 1, 1);
   const uint64_t vertex_grid =
      gfx->descriptors.root.draw.vertex_params +
      offsetof(struct poly_vertex_params, grid);
   const uint64_t geometry_grid =
      gfx->descriptors.root.draw.geometry_params +
      offsetof(struct poly_geometry_params, grid);

   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   if (batch->includes_gs && gs) {
      gs_vs = hk_bound_sw_vs_before_gs(gfx);
      gs_main = hk_main_gs_variant(gs);
      const VkQueryPipelineStatisticFlagBits gs_queries =
         VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
         VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
         VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
         VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;
      const bool xfb_or_queries =
         gs_main->info.gs.xfb ||
         (gfx->descriptors.root.draw.pipeline_stats_flags & gs_queries);
      emit_gs = !xfb_or_queries && batch->gs_vs == gs_vs &&
         batch->gs_main == gs_main && pgs_setup->valid &&
         pgs_setup->idx == LIBAGX_GS_SETUP_INDIRECT &&
         pgs_vs->indirect_local_valid &&
         !pgs_vs->indirect_local_consumed &&
         pgs_vs->indirect_local_grid == vertex_grid &&
         pgs_main->indirect_local_valid &&
         !pgs_main->indirect_local_consumed &&
         pgs_main->indirect_local_grid == geometry_grid;
   }

   hk_reserve_scratch(cmd, cs, vs);
   hk_reserve_scratch(cmd, cs, tcs);
   if (emit_gs) {
      hk_reserve_scratch(cmd, cs, gs_vs);
      hk_reserve_scratch(cmd, cs, gs_main);
   }
   const size_t gs_bytes = emit_gs
      ? AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE +
           2 * AGX3_G17P_NATIVE_INDIRECT_COMPUTE_DISPATCH_SIZE
      : 0;
   hk_ensure_cs_has_space(
      cmd, cs, sizeof(gfx->g17p_sw_tess_batch.bytes) + gs_bytes + 64);
   if (vk_command_buffer_has_error(&cmd->vk))
      return (struct agx_draw){0};

   uint8_t *out = cs->current;
   memcpy(out, gfx->g17p_sw_tess_batch.bytes,
          sizeof(gfx->g17p_sw_tess_batch.bytes));
   const struct agx_grid grids[HK_G17P_SW_TESS_BATCH_LAUNCHES] = {
      grid_vs, grid_tcs, grid_tess, agx_1d(1024), grid_tess,
   };
   const struct agx_workgroup locals[2] = {
      agx_workgroup(64, 1, 1),
      agx_workgroup(tcs->info.tess.tcs_output_patch_size, 1, 1),
   };
   const uint8_t graphics_barrier =
      hk_g17p_strong_dispatch_flush_enabled() ? 0x5 : 0;
   for (unsigned launch = 0;
        launch < HK_G17P_SW_TESS_BATCH_LAUNCHES; ++launch) {
      uint8_t *record =
         out + launch * AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE;
      for (unsigned axis = 0; axis < 3; ++axis)
         agx3_cmdstream_store_le32(record + 16 + axis * 4,
                                    grids[launch].count[axis]);
      if (launch < ARRAY_SIZE(locals)) {
         const uint32_t local[3] = {
            locals[launch].x, locals[launch].y, locals[launch].z,
         };
         for (unsigned axis = 0; axis < 3; ++axis)
            agx3_cmdstream_store_le32(record + 28 + axis * 4, local[axis]);
      }
      agx3_cmdstream_store_le32(
         record + AGX3_CDM_DIRECT_LAUNCH_SIZE,
         agx3_cdm_barrier_word(launch < 2 ? graphics_barrier : 0x5));
   }

   uint8_t *cursor = out + sizeof(gfx->g17p_sw_tess_batch.bytes);
   cs->strong_flush_end = cursor;
   cs->weak_barrier_end = NULL;
   cs->stats.cmds += HK_G17P_SW_TESS_BATCH_LAUNCHES;

   if (emit_gs) {
      hk_g17p_materialize_prepared_indirect(pgs_vs, vertex_grid, gs_local);
      hk_g17p_materialize_prepared_indirect(pgs_main, geometry_grid,
                                             gs_local);
      memcpy(cursor, pgs_setup->direct_template,
             AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE);
      agx3_cmdstream_store_le32(
         cursor + AGX3_CDM_DIRECT_LAUNCH_SIZE,
         agx3_cdm_barrier_word(0x5));
      cursor += AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE;
      cs->strong_flush_end = cursor;
      memcpy(cursor, pgs_vs->indirect_local_template,
             AGX3_G17P_NATIVE_INDIRECT_COMPUTE_DISPATCH_SIZE);
      cursor += AGX3_G17P_NATIVE_INDIRECT_COMPUTE_DISPATCH_SIZE;
      memcpy(cursor, pgs_main->indirect_local_template,
             AGX3_G17P_NATIVE_INDIRECT_COMPUTE_DISPATCH_SIZE);
      cursor += AGX3_G17P_NATIVE_INDIRECT_COMPUTE_DISPATCH_SIZE;
      cs->weak_barrier_end = cursor;
      cs->stats.cmds += 3;
      gfx->g17p_sw_tess_batch.gs_emitted = true;
   }

   cs->current = cursor;
   return tess_draw;
}

static __attribute__((hot)) struct agx_draw
hk_g17p_launch_prepared_sw_tess_gs(struct hk_cmd_buffer *cmd,
                                   struct hk_cs *cs,
                                   struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

   if (dev->dev.chip != AGX_CHIP_G17P || agx_is_indirect(draw.b) || !gs ||
       !gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       hk_g17p_tess_is_native(cmd) ||
       !gfx->g17p_sw_tess_batch.valid ||
       !gfx->g17p_sw_tess_batch.includes_gs)
      return (struct agx_draw){0};

   /* Pipeline statistics add a real extra dispatch which is deliberately not
    * part of the ordinary prepared transaction. */
   if (hk_stat_enabled(hk_pipeline_stat_addr(
          cmd,
          VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT)))
      return (struct agx_draw){0};

   struct hk_shader *vs = hk_bound_sw_vs(gfx);
   struct hk_shader *tcs =
      hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);
   const uint32_t input_patch_size =
      cmd->vk.dynamic_graphics_state.ts.patch_control_points;
   if (!input_patch_size)
      return (struct agx_draw){0};

   const uint32_t patches = draw.b.count[0] / input_patch_size;
   struct agx_grid grid_vs = draw.b;
   struct agx_grid grid_tcs = draw.b;
   grid_tcs.count[0] = patches * tcs->info.tess.tcs_output_patch_size;
   const struct agx_grid grid_tess = agx_1d(patches * draw.b.count[1]);
   const uint64_t state = gfx->descriptors.root.draw.tess_params;
   const enum libagx_program count_program =
      hk_sw_tess_program(gfx->tess.info.mode, POLY_TESS_MODE_COUNT);
   const enum libagx_program final_program =
      hk_sw_tess_program(gfx->tess.info.mode, POLY_TESS_MODE_WITH_COUNTS);

   return hk_g17p_emit_prepared_sw_tess_batch(
      cmd, cs, vs, tcs, grid_vs, grid_tcs, grid_tess, state, 0, 0,
      count_program, final_program);
}

static __attribute__((hot)) struct agx_draw
hk_launch_tess(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
               struct agx_draw draw, uint64_t c_prims, uint64_t c_inv)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct agx_grid grid_vs, grid_tcs, grid_tess;

   struct hk_shader *vs = hk_bound_sw_vs(gfx);
   struct hk_shader *tcs = hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint32_t input_patch_size = dyn->ts.patch_control_points;
   uint64_t state = gfx->descriptors.root.draw.tess_params;
   struct hk_tess_info info = gfx->tess.info;
   const bool perf_trace = hk_shader_perf_draw_enabled(dev);
   static uint32_t perf_calls;
   const uint32_t perf_call = dev->g17p_shader_perf.trace
      ? p_atomic_inc_return(&perf_calls) : 0;
   uint64_t phase_start = perf_trace ? os_time_get_nano() : 0;
   uint64_t setup_ns = 0, reserve_ns = 0, shader_dispatch_ns = 0;
   uint64_t count_ns = 0, prefix_ns = 0, final_ns = 0;
   bool prepared_vs = false, prepared_tcs = false;

   perf_debug(cmd, "Tessellation");

   uint64_t tcs_stat = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT);

   /* Setup grids */
   if (agx_is_indirect(draw.b)) {
      perf_debug(cmd, "Indirect tessellation");

      struct libagx_tess_setup_indirect_args args = {
         .p = state,
         .grids = gfx->tess.grids,
         .indirect = draw.b.ptr,
         .vp = gfx->descriptors.root.draw.vertex_params,
         .vertex_outputs = vs->b.info.outputs,
         .tcs_statistic = hk_pipeline_stat_addr(
            cmd,
            VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT),
      };

      if (draw.indexed) {
         args.in_index_buffer = draw.index_buffer;
         args.in_index_size_B = agx_index_size_to_B(draw.index_size);
         args.in_index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_tess_setup_indirect_struct(cmd, agx_1d(1),
                                        AGX_BARRIER_ALL | AGX_PREGFX, args);

      uint32_t grid_stride = sizeof(uint32_t) * 6;
      grid_vs = agx_grid_indirect_local(gfx->tess.grids + 0 * grid_stride);
      grid_tcs = agx_grid_indirect_local(gfx->tess.grids + 1 * grid_stride);
      grid_tess = agx_grid_indirect_local(gfx->tess.grids + 2 * grid_stride);
   } else {
      uint32_t patches = draw.b.count[0] / input_patch_size;
      grid_vs = grid_tcs = draw.b;

      grid_tcs.count[0] = patches * tcs->info.tess.tcs_output_patch_size;
      grid_tess = agx_1d(patches * draw.b.count[1]);

      if (hk_g17p_launch_trace_enabled()) {
         fprintf(stderr,
                 "G17P sw tess draw: verts=%u instances=%u cps=%u patches=%u "
                 "unrolled=%u out_draws=%" PRIx64 " heap=%u\n",
                 draw.b.count[0], draw.b.count[1], input_patch_size, patches,
                 patches * draw.b.count[1], (uint64_t)gfx->tess.out_draws,
                 (unsigned)dev->heap->size);
      }

      /* TCS invocation counter increments once per-patch */
      if (hk_stat_enabled(tcs_stat)) {
         perf_debug(cmd, "Direct TCS statistic");
         libagx_increment_statistic(
            cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, tcs_stat, patches);
      }
   }

   if (perf_trace) {
      setup_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   if (perf_trace) {
      reserve_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   const enum libagx_program count_program =
      hk_sw_tess_program(info.mode, POLY_TESS_MODE_COUNT);
   const enum libagx_program final_program =
      hk_sw_tess_program(info.mode, POLY_TESS_MODE_WITH_COUNTS);
   struct agx_draw batched = hk_g17p_emit_prepared_sw_tess_batch(
      cmd, cs, vs, tcs, grid_vs, grid_tcs, grid_tess, state, c_prims, c_inv,
      count_program, final_program);
   if (batched.b.ptr)
      return batched;

   if (dev->dev.chip == AGX_CHIP_G17P) {
      prepared_vs = hk_dispatch_prepared_g17p_native(
         cmd, cs, vs, &gfx->g17p_dispatch.tess_vs, grid_vs,
         agx_workgroup(64, 1, 1));
      if (!prepared_vs)
         hk_dispatch_with_local_size(cmd, cs, vs, grid_vs,
                                     agx_workgroup(64, 1, 1));
      prepared_tcs = hk_dispatch_prepared_g17p_native(
         cmd, cs, tcs, &gfx->g17p_dispatch.tess_tcs, grid_tcs,
         agx_workgroup(tcs->info.tess.tcs_output_patch_size, 1, 1));
      if (!prepared_tcs)
         hk_dispatch_with_local_size(
            cmd, cs, tcs, grid_tcs,
            agx_workgroup(tcs->info.tess.tcs_output_patch_size, 1, 1));
   } else {
      /* Legacy launches do not reserve scratch in hk_dispatch_with_usc(). */
      hk_reserve_scratch(cmd, cs, vs);
      hk_reserve_scratch(cmd, cs, tcs);
      hk_dispatch_with_usc(
         dev, cs, &vs->b.info,
         hk_upload_usc_words(cmd, vs, gfx->linked[MESA_SHADER_VERTEX]),
         grid_vs, agx_workgroup(64, 1, 1));

      hk_dispatch_with_usc(
         dev, cs, &tcs->b.info,
         hk_upload_usc_words(cmd, tcs, tcs->only_linked), grid_tcs,
         agx_workgroup(tcs->info.tess.tcs_output_patch_size, 1, 1));
   }

   if (perf_trace) {
      shader_dispatch_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   /* First generate counts, then prefix sum them, and then tessellate. */
   if (!hk_dispatch_prepared_precomp_g17p(
          cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX, count_program, &state,
          sizeof(state), &gfx->g17p_precomp.tess_count)) {
      libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX,
                        info.mode, POLY_TESS_MODE_COUNT, state);
   }

   if (perf_trace) {
      count_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   /* HK_G17P_TG_PROBE=local|sg runs one threadgroup-memory probe in this
    * dispatch stream, with the same internal precompiled record, immediately
    * before the scan. Inert unless the variable is set; it is how the quarter-
    * sized threadgroup window on this path was measured. */
   {
      const char *probe = hk_g17p_tg_probe_option();
      if (unlikely(probe != NULL)) {
         struct agx_ptr report = hk_pool_alloc(cmd, 8192, 4);
         if (report.gpu) {
            if (probe[0] == 's') {
               libagx_g17p_local_sg_probe(cmd, agx_1d(1024),
                                          AGX_BARRIER_ALL | AGX_PREGFX,
                                          report.gpu);
            } else {
               libagx_g17p_local_probe(cmd, agx_1d(1024),
                                       AGX_BARRIER_ALL | AGX_PREGFX,
                                       report.gpu);
            }
         }
      }
   }

   const struct libagx_prefix_sum_tess_args prefix = {
      .p = state,
      .c_prims = c_prims,
      .c_invs = c_inv,
   };
   const enum libagx_program prefix_program =
      LIBAGX_PREFIX_SUM_TESS(c_prims || c_inv);
   if (!hk_dispatch_prepared_precomp_g17p(
          cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX, prefix_program,
          &prefix, sizeof(prefix), &gfx->g17p_precomp.tess_prefix))
      libagx_prefix_sum_tess(cmd, agx_1d(1024),
                             AGX_BARRIER_ALL | AGX_PREGFX, state, c_prims,
                             c_inv, c_prims || c_inv);

   if (perf_trace) {
      prefix_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   if (!hk_dispatch_prepared_precomp_g17p(
          cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX, final_program, &state,
          sizeof(state), &gfx->g17p_precomp.tess_final)) {
      libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX,
                        info.mode, POLY_TESS_MODE_WITH_COUNTS, state);
   }

   if (perf_trace) {
      final_ns = os_time_get_nano() - phase_start;
      hk_shader_perf_sw_tess_detail_record(setup_ns, reserve_ns,
                                            shader_dispatch_ns, count_ns,
                                            prefix_ns, final_ns);
      if (dev->g17p_shader_perf.trace && perf_call <= 2) {
         fprintf(stderr,
                 "HK_SHADER_PERF sw-tess-call=%u setup_ns=%" PRIu64
                 " reserve_ns=%" PRIu64 " shader_dispatch_ns=%" PRIu64
                 " count_ns=%" PRIu64 " prefix_ns=%" PRIu64
                 " final_ns=%" PRIu64 " prepared_vs=%u prepared_tcs=%u\n",
                 perf_call, setup_ns, reserve_ns, shader_dispatch_ns, count_ns,
                 prefix_ns, final_ns, prepared_vs, prepared_tcs);
      }
   }

   return agx_draw_indexed_indirect(gfx->tess.out_draws, dev->heap->va->addr,
                                    dev->heap->size, AGX_INDEX_SIZE_U32, false);
}

__attribute__((hot)) void
hk_cmd_bind_graphics_shader(struct hk_cmd_buffer *cmd,
                            const mesa_shader_stage stage,
                            struct hk_api_shader *shader)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* Null fragment shaders are annoying to handle, because we may still need an
    * actual fragment shader to attach a prolog to. Rather than adding special
    * cases, we just bind an empty fragment shader instead of NULL to make
    * everything work correctly.
    */
   if (stage == MESA_SHADER_FRAGMENT && shader == NULL) {
      shader = dev->null_fs;
   }

   assert(stage < ARRAY_SIZE(cmd->state.gfx.shaders));
   if (cmd->state.gfx.shaders[stage] == shader)
      return;

   /* Any direct stage replacement (shader objects, meta restore, or an
    * internal passthrough GS) invalidates the common runtime's redundant
    * pipeline-bind identity.  A complete pipeline bind sets it again only
    * after every stage and static dynamic state have been restored. */
   cmd->vk.last_graphics_pipeline = NULL;

   cmd->state.gfx.shaders[stage] = shader;
   cmd->state.gfx.shaders_dirty |= BITFIELD_BIT(stage);

   if (stage == MESA_SHADER_FRAGMENT) {
      cmd->state.gfx.ordinary_fs.valid = false;
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);
   }
}

static uint64_t
hk_upload_constant_data(struct hk_cmd_buffer *cmd, struct hk_shader *s)
{
   if (!s || s->data_size == 0)
      return 0;

   uint64_t addr =
      hk_pool_upload(cmd, s->data_ptr, s->data_size, HK_MIN_UBO_ALIGNMENT);
   if (!addr)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   return addr;
}

static void
hk_flush_shaders(struct hk_cmd_buffer *cmd)
{
   struct hk_api_shader *fs_obj =
       cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT];
   /* Selection may dirty FS before the ordinary early-out below. */
   if (fs_obj->g17p_ordinary_variants)
      hk_bound_fs(cmd);
   const bool sample_variant_dirty = fs_obj->g17p_sample_variants &&
      BITSET_TEST(cmd->vk.dynamic_graphics_state.dirty,
                  MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);
   if (cmd->state.gfx.shaders_dirty == 0 && !sample_variant_dirty)
      return;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (sample_variant_dirty)
      gfx->shaders_dirty |= BITFIELD_BIT(MESA_SHADER_FRAGMENT);

   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   /* Geometry shading overrides the restart index, reemit on rebind */
   if (IS_SHADER_DIRTY(GEOMETRY)) {
      struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
      struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

      const bool api_gs = gs && !gs->is_passthrough;
      if (desc->root.draw.api_gs != api_gs) {
         desc->root.draw.api_gs = api_gs;
         desc->root_dirty = true;
      }
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE);
   }

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *fs = hk_bound_fs(cmd);

   /* A new VS/FS pair or sample-count variant may change interpolation. This
    * assignment does not depend on the fast linked shaders.
    */
   agx_assign_uvs(&gfx->linked_varyings, &hw_vs->info.uvs,
                  fs->info.fs.interp.flat,
                  fs->info.fs.interp.linear);

   if (memcmp(desc->root.draw.uvs_index, gfx->linked_varyings.slots,
              sizeof(desc->root.draw.uvs_index))) {
      memcpy(desc->root.draw.uvs_index, gfx->linked_varyings.slots,
             sizeof(desc->root.draw.uvs_index));
      desc->root_dirty = true;
   }

   /* Shader replacement changes the logical root only if the selected
    * immutable data address actually changes.  In particular, equivalent
    * shader objects must not clone the root and invalidate both native IOTO
    * programs merely because their API handles differ. */
   const uint64_t vs_constant_data_addr = hk_upload_constant_data(
      cmd, hk_any_variant(gfx->shaders[MESA_SHADER_VERTEX]));
   const uint64_t fs_constant_data_addr =
      hk_upload_constant_data(cmd, fs);
   if (desc->root.draw.vs_constant_data_addr != vs_constant_data_addr) {
      desc->root.draw.vs_constant_data_addr = vs_constant_data_addr;
      desc->root_dirty = true;
   }
   if (desc->root.draw.fs_constant_data_addr != fs_constant_data_addr) {
      desc->root.draw.fs_constant_data_addr = fs_constant_data_addr;
      desc->root_dirty = true;
   }
}

static void build_g17p_fs_epilog(nir_builder *b, const void *key_);

static struct agx_shader_part *
hk_get_prolog_epilog_locked(struct hk_device *dev, struct hk_internal_key *key,
                            hk_internal_builder_t builder, bool preprocess_nir,
                            bool stop, unsigned cf_base,
                            const BITSET_WORD *g17p_tile_partitions,
                            unsigned nr_samples)
{
   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(dev->prolog_epilog.ht, key);
   if (ent)
      return ent->data;

   nir_builder b = nir_builder_init_simple_shader(0, &agx_nir_options, NULL);
   builder(&b, key->key);

   if (preprocess_nir)
      agx_preprocess_nir(b.shader);

   struct agx_shader_key backend_key = {
      .dev = agx_gather_device_key(&dev->dev),
      .fs.nr_samples = nr_samples,
      .secondary = true,
      .no_stop = !stop,
   };
   if (builder == build_g17p_fs_epilog)
      backend_key.g17p_emrt =
         ((const struct agx_fs_epilog_key *)key->key)->native_spills;
   if (builder == build_g17p_fs_epilog) {
      /* Preprocessing can unroll any multisample epilog, not only eMRT.
       * Its induction masks then become literals. Materialize the native
       * coverage operands after that transformation, before binary packing. */
      NIR_PASS(_, b.shader, agx_nir_lower_g17p_tile_samples, nr_samples);
   }
   if (backend_key.g17p_emrt) {
      /* Only the image accesses may be conditional. Every resident tile
       * operation must execute on every path through the epilog. */
      nir_metadata_require(b.impl, nir_metadata_dominance);
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            if ((op == nir_intrinsic_store_local_pixel_agx ||
                 op == nir_intrinsic_load_local_pixel_agx) &&
                !nir_block_dominates(block, nir_impl_last_block(b.impl))) {
               backend_key.g17p_emrt = false;
               mesa_loge("G17P eMRT has conditional resident tile access");
            }
         }
      }
   }
   if (g17p_tile_partitions) {
      memcpy(backend_key.g17p_tile_partitions, g17p_tile_partitions,
             sizeof(backend_key.g17p_tile_partitions));
   }

   /* We always use dynamic sample shading in the GL driver. Indicate that. */
   if (b.shader->info.stage == MESA_SHADER_FRAGMENT) {
      backend_key.fs.cf_base = cf_base;

      if (b.shader->info.fs.uses_sample_shading)
         backend_key.fs.inside_sample_loop = true;
   }

   struct agx_shader_part *part =
      rzalloc(dev->prolog_epilog.ht, struct agx_shader_part);
   if (!part) {
      ralloc_free(b.shader);
      return NULL;
   }

   const bool compiled = agx_compile_shader_nir(b.shader, &backend_key, part);

   ralloc_free(b.shader);
   if (!compiled) {
      /* A rejected backend instruction leaves an empty part. Caching that
       * as success lets fast-link silently omit required output/blend code. */
      ralloc_free(part);
      return NULL;
   }

   /* ..and cache it before we return. The key is on the stack right now, so
    * clone it before using it as a hash table key. The clone is logically owned
    * by the hash table.
    */
   size_t total_key_size = sizeof(*key) + key->key_size;
   void *cloned_key = ralloc_memdup(dev->prolog_epilog.ht, key, total_key_size);
   if (!cloned_key) {
      free(part->binary);
      ralloc_free(part);
      return NULL;
   }

   _mesa_hash_table_insert(dev->prolog_epilog.ht, cloned_key, part);
   return part;
}

static struct agx_shader_part *
hk_get_prolog_epilog(struct hk_device *dev, void *data, size_t data_size,
                     hk_internal_builder_t builder, bool preprocess_nir,
                     bool stop, unsigned cf_base,
                     const BITSET_WORD *g17p_tile_partitions,
                     unsigned nr_samples)
{
   /* Build the meta shader key */
   size_t total_key_size = sizeof(struct hk_internal_key) + data_size;

   struct hk_internal_key *key = alloca(total_key_size);
   memset(key, 0, total_key_size);
   key->builder = builder;
   key->key_size = data_size;
   key->stop = stop;

   if (data_size)
      memcpy(key->key, data, data_size);

   simple_mtx_lock(&dev->prolog_epilog.lock);

   struct agx_shader_part *part =
      hk_get_prolog_epilog_locked(dev, key, builder, preprocess_nir, stop,
                                  cf_base, g17p_tile_partitions, nr_samples);

   simple_mtx_unlock(&dev->prolog_epilog.lock);
   return part;
}

static struct hk_linked_shader *
hk_get_fast_linked_locked_vs(struct hk_device *dev, struct hk_shader *shader,
                             struct hk_fast_link_key_vs *key)
{
   struct agx_shader_part *prolog =
      hk_get_prolog_epilog(dev, &key->prolog, sizeof(key->prolog),
                           dev->dev.chip == AGX_CHIP_G17P
                              ? agx_nir_vs_prolog_raw : agx_nir_vs_prolog,
                           false, false, 0, 0, 0);
   if (!prolog)
      return NULL;

   struct hk_fast_link_key *key_clone =
      ralloc_memdup(shader->linked.ht, key, sizeof(*key));
   if (!key_clone)
      return NULL;

   struct hk_linked_shader *linked =
      hk_fast_link(dev, false, shader, prolog, NULL, 0);
   if (!linked) {
      ralloc_free(key_clone);
      return NULL;
   }

   /* XXX: Fix this higher up the stack */
   linked->sw_indexing = !key->prolog.hw || key->prolog.adjacency;
   linked->sw_indexing_hw_vs = key->prolog.hw && key->prolog.adjacency;
   linked->b.uses_base_param |= linked->sw_indexing;

   _mesa_hash_table_insert(shader->linked.ht, key_clone, linked);
   return linked;
}

static bool
lower_fs_root(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_root_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_load_preamble(b, 1, 64, .base = AGX_ABI_FUNI_ROOT));
   return true;
}

static void
build_fs_prolog(nir_builder *b, const void *key)
{
   agx_nir_fs_prolog(b, key);

   /* Lower load_stat_query_address_agx, needed for FS statistics */
   NIR_PASS(_, b->shader, hk_lower_uvs_index, MESA_SHADER_FRAGMENT, 0, 0);
   NIR_PASS(_, b->shader, nir_shader_intrinsics_pass, lower_fs_root,
            nir_metadata_control_flow, NULL);
}

static void
build_g17p_fs_epilog(nir_builder *b, const void *key_)
{
   const struct agx_fs_epilog_key *key = key_;
   if (key->link.sample_shading) {
      /* The main fragment exported every sample's colours. Build the per-pixel
       * epilog, then store and blend each sample from its own colours. */
      struct agx_fs_epilog_key pixel = *key;
      pixel.link.sample_shading = false;
      pixel.defer_monolithic_msaa = true;
      agx_nir_fs_epilog(b, &pixel);

      struct agx_g17p_colour_layout layout;
      agx_g17p_colour_layout_init(&layout, key->link.loc_written,
                                  key->link.size_32, key->link.colour_comps,
                                  key->link.packed_colour);

      NIR_PASS(_, b->shader, agx_nir_lower_to_per_sample);
      NIR_PASS(_, b->shader, agx_nir_unroll_g17p_samples_layout,
               key->nr_samples, &layout);
      bool opt_progress;
      do {
         opt_progress = false;
         NIR_PASS(opt_progress, b->shader, nir_opt_constant_folding);
         NIR_PASS(opt_progress, b->shader, nir_opt_algebraic);
      } while (opt_progress);
      NIR_PASS(_, b->shader, agx_nir_hoist_g17p_zs_writes);
   } else {
      /* G17P's tile operations carry the live coverage register. With
       * alpha-to-coverage, Apple's 2x/4x blended fragments therefore issue
       * one masked tile load/store pair rather than cloning the blend once
       * per sample. Keep the ordinary per-sample lowering for every other
       * pixel-shaded epilog. */
      struct agx_fs_epilog_key pixel = *key;
      pixel.defer_monolithic_msaa = key->blend.alpha_to_coverage &&
                                    key->nr_samples > 1;
      agx_nir_fs_epilog(b, &pixel);
   }

   NIR_PASS(_, b->shader, agx_nir_lower_g17p_tile_samples, key->nr_samples);
}

static struct hk_linked_shader *
hk_get_fast_linked_locked_fs(struct hk_device *dev, struct hk_shader *shader,
                             struct hk_fast_link_key_fs *key)
{
   /* TODO: prolog without fs needs to work too... */
   bool needs_prolog = key->prolog.statistics ||
                       key->prolog.cull_distance_size ||
                       key->prolog.api_sample_mask != 0xff;

   struct agx_shader_part *prolog = NULL;
   if (needs_prolog) {
      prolog = hk_get_prolog_epilog(dev, &key->prolog, sizeof(key->prolog),
                                    build_fs_prolog, false, false,
                                    key->prolog.cf_base, 0,
                                    key->prolog.nr_samples);
      if (!prolog) {
         mesa_loge("fragment prolog compilation failed: stats=%u cull=%u "
                   "api-mask=%#x run-zs=%u cf-base=%u",
                   key->prolog.statistics, key->prolog.cull_distance_size,
                   key->prolog.api_sample_mask, key->prolog.run_zs_tests,
                   key->prolog.cf_base);
         return NULL;
      }
   }

   /* If sample shading is used, don't stop at the epilog, there's a
    * footer that the fast linker will insert to stop.
    */
   bool epilog_stop = (key->nr_samples_shaded == 0);

   /* A G17P epilog selects each written target's pixel classes from its
    * position among all of the framebuffer's colour targets. */
   BITSET_DECLARE(g17p_tile_partitions, AGX_G17P_MAX_BYTES_PER_SAMPLE) = {0};
   if (dev->dev.chip == AGX_CHIP_G17P) {
      const struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout_with_limits(
            key->epilog.rt_formats, ARRAY_SIZE(key->epilog.rt_formats),
            key->epilog.nr_samples, true, key->epilog.max_sample_size_B,
            key->epilog.max_bytes_per_tile, key->epilog.max_samples_per_tile);
      for (unsigned rt = 0; rt < ARRAY_SIZE(tib.logical_format); ++rt) {
         if (tib.logical_format[rt] != PIPE_FORMAT_NONE && !tib.spilled[rt])
            BITSET_SET(g17p_tile_partitions, tib._offset_B[rt]);
      }
      if (BITSET_COUNT(g17p_tile_partitions) < 2)
         BITSET_ZERO(g17p_tile_partitions);
   }

   struct agx_shader_part *epilog =
      hk_get_prolog_epilog(dev, &key->epilog, sizeof(key->epilog),
                           dev->dev.chip == AGX_CHIP_G17P
                              ? build_g17p_fs_epilog
                              : agx_nir_fs_epilog,
                           true, epilog_stop, 0, g17p_tile_partitions,
                           dev->dev.chip == AGX_CHIP_G17P
                              ? key->epilog.nr_samples
                              : 0);
   if (!epilog) {
      uint32_t link = 0;
      memcpy(&link, &key->epilog.link, sizeof(link));
      mesa_loge("fragment epilog compilation failed: link=%#010x samples=%u "
                "shaded=%u stop=%u small=%u preserve-zero=%u srgb=%u "
                "formats=%x/%x/%x/%x remap=%d/%d/%d/%d",
                link, key->epilog.nr_samples, key->nr_samples_shaded,
                epilog_stop, key->epilog.force_small_tile,
                key->epilog.preserve_zero_mask, key->epilog.lower_srgb,
                key->epilog.rt_formats[0], key->epilog.rt_formats[1],
                key->epilog.rt_formats[2], key->epilog.rt_formats[3],
                key->epilog.remap[0], key->epilog.remap[1],
                key->epilog.remap[2], key->epilog.remap[3]);
      return NULL;
   }

   struct hk_fast_link_key *key_clone =
      ralloc_memdup(shader->linked.ht, key, sizeof(*key));
   if (!key_clone) {
      mesa_loge("fragment fast-link key allocation failed");
      return NULL;
   }

   struct hk_linked_shader *linked =
      hk_fast_link(dev, true, shader, prolog, epilog, key->nr_samples_shaded);
   if (!linked) {
      uint32_t link_info = 0;
      memcpy(&link_info, &key->epilog.link, sizeof(link_info));
      mesa_loge("fragment fast-link failed: link=%#010x samples=%u shaded=%u "
                "prolog=%u main=%u epilog=%u",
                link_info, key->epilog.nr_samples, key->nr_samples_shaded,
                prolog != NULL, shader->b.binary != NULL,
                epilog->binary != NULL);
      ralloc_free(key_clone);
      return NULL;
   }
   linked->g17p_blend_constant_base = key->epilog.blend_constant_base;
   linked->g17p_emrt_heap_base = key->epilog.emrt_heap_base;
   linked->g17p_prolog_uses_root = key->prolog.statistics;

   _mesa_hash_table_insert(shader->linked.ht, key_clone, linked);
   return linked;
}

/*
 * First, look for a fully linked variant. Else, build the required shader
 * parts and link.
 */
static struct hk_linked_shader *
hk_get_fast_linked(struct hk_device *dev, struct hk_shader *shader, void *key)
{
   struct hk_linked_shader *linked;
   simple_mtx_lock(&shader->linked.lock);

   if (!hk_shader_ensure_link_ht_locked(shader)) {
      simple_mtx_unlock(&shader->linked.lock);
      return NULL;
   }

   struct hash_entry *ent = _mesa_hash_table_search(shader->linked.ht, key);

   if (ent)
      linked = ent->data;
   else if (shader->info.stage == MESA_SHADER_VERTEX)
      linked = hk_get_fast_linked_locked_vs(dev, shader, key);
   else if (shader->info.stage == MESA_SHADER_FRAGMENT)
      linked = hk_get_fast_linked_locked_fs(dev, shader, key);
   else
      UNREACHABLE("invalid stage");

   simple_mtx_unlock(&shader->linked.lock);
   return linked;
}

static __attribute__((hot)) bool
hk_update_fast_linked(struct hk_cmd_buffer *cmd, struct hk_shader *shader,
                      void *key)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   mesa_shader_stage stage = shader->info.stage;

   if (stage == MESA_SHADER_VERTEX && gfx->fast_link_vs.valid &&
       gfx->fast_link_vs.shader == shader &&
       memcmp(&gfx->fast_link_vs.key, key,
              sizeof(gfx->fast_link_vs.key)) == 0 &&
       gfx->linked[stage] != NULL)
      return true;

   if (stage == MESA_SHADER_FRAGMENT && gfx->fast_link_fs.valid &&
       gfx->fast_link_fs.shader == shader &&
       memcmp(&gfx->fast_link_fs.key, key,
              sizeof(gfx->fast_link_fs.key)) == 0 &&
       gfx->linked[stage] != NULL)
      return true;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *new = hk_get_fast_linked(dev, shader, key);
   if (!new) {
      mesa_loge("failed to compile or link required %s shader parts",
                _mesa_shader_stage_to_string(stage));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   if (gfx->linked[stage] != new) {
      gfx->linked[stage] = new;
      gfx->linked_dirty |= BITFIELD_BIT(stage);
   }

   if (stage == MESA_SHADER_VERTEX) {
      gfx->fast_link_vs.shader = shader;
      memcpy(&gfx->fast_link_vs.key, key, sizeof(gfx->fast_link_vs.key));
      gfx->fast_link_vs.valid = true;
   } else if (stage == MESA_SHADER_FRAGMENT) {
      gfx->fast_link_fs.shader = shader;
      memcpy(&gfx->fast_link_fs.key, key, sizeof(gfx->fast_link_fs.key));
      gfx->fast_link_fs.valid = true;
   }
   return true;
}

static enum agx_polygon_mode
translate_polygon_mode(VkPolygonMode vk_mode)
{
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_FILL ==
                 AGX_POLYGON_MODE_FILL);
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_LINE ==
                 AGX_POLYGON_MODE_LINE);
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_POINT ==
                 AGX_POLYGON_MODE_POINT);

   assert(vk_mode <= VK_POLYGON_MODE_POINT);
   return (enum agx_polygon_mode)vk_mode;
}

static enum agx_zs_func
translate_compare_op(VkCompareOp vk_mode)
{
   static_assert((enum agx_zs_func)VK_COMPARE_OP_NEVER == AGX_ZS_FUNC_NEVER);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_LESS == AGX_ZS_FUNC_LESS);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_EQUAL == AGX_ZS_FUNC_EQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_LESS_OR_EQUAL ==
                 AGX_ZS_FUNC_LEQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_GREATER ==
                 AGX_ZS_FUNC_GREATER);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_NOT_EQUAL ==
                 AGX_ZS_FUNC_NOT_EQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_GREATER_OR_EQUAL ==
                 AGX_ZS_FUNC_GEQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_ALWAYS == AGX_ZS_FUNC_ALWAYS);

   assert(vk_mode <= VK_COMPARE_OP_ALWAYS);
   return (enum agx_zs_func)vk_mode;
}

static enum agx_stencil_op
translate_stencil_op(VkStencilOp vk_op)
{
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_KEEP ==
                 AGX_STENCIL_OP_KEEP);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_ZERO ==
                 AGX_STENCIL_OP_ZERO);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_REPLACE ==
                 AGX_STENCIL_OP_REPLACE);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INCREMENT_AND_CLAMP ==
                 AGX_STENCIL_OP_INCR_SAT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_DECREMENT_AND_CLAMP ==
                 AGX_STENCIL_OP_DECR_SAT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INVERT ==
                 AGX_STENCIL_OP_INVERT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INCREMENT_AND_WRAP ==
                 AGX_STENCIL_OP_INCR_WRAP);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_DECREMENT_AND_WRAP ==
                 AGX_STENCIL_OP_DECR_WRAP);

   return (enum agx_stencil_op)vk_op;
}

static struct agx_fragment_stencil_packed
hk_pack_stencil_face(struct vk_stencil_test_face_state s, bool enabled)
{
   struct agx_fragment_stencil_packed packed;

   if (enabled) {
      agx_pack(&packed, FRAGMENT_STENCIL, cfg) {
         cfg.compare = translate_compare_op(s.op.compare);
         cfg.write_mask = s.write_mask;
         cfg.read_mask = s.compare_mask;

         cfg.depth_pass = translate_stencil_op(s.op.pass);
         cfg.depth_fail = translate_stencil_op(s.op.depth_fail);
         cfg.stencil_fail = translate_stencil_op(s.op.fail);
      }
   } else {
      agx_pack(&packed, FRAGMENT_STENCIL, cfg) {
         cfg.compare = AGX_ZS_FUNC_ALWAYS;
         cfg.write_mask = 0xFF;
         cfg.read_mask = 0xFF;

         cfg.depth_pass = AGX_STENCIL_OP_KEEP;
         cfg.depth_fail = AGX_STENCIL_OP_KEEP;
         cfg.stencil_fail = AGX_STENCIL_OP_KEEP;
      }
   }

   return packed;
}

static void
hk_ppp_push_stencil_face(struct agx_ppp_update *ppp,
                         struct vk_stencil_test_face_state s, bool enabled)
{
   struct agx_fragment_stencil_packed packed =
      hk_pack_stencil_face(s, enabled);
   agx_ppp_push_packed(ppp, &packed, FRAGMENT_STENCIL);
}

static bool
hk_stencil_test_enabled(struct hk_cmd_buffer *cmd)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   return dyn->ds.stencil.test_enable &&
          render->stencil_att.vk_format != VK_FORMAT_UNDEFINED;
}

static bool
hk_emit_hal300_vp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        enum agx_chip chip, unsigned viewport_count,
                        unsigned scissor_index, const unsigned minx[],
                        const unsigned miny[], const unsigned maxx[],
                        const unsigned maxy[], const VkViewport viewports[],
                        uint8_t **out)
{
   if (!out || !*out || viewport_count == 0 ||
       viewport_count > AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS ||
       !agx_ppp_hal300_depth_bias_scissor_core_supported(chip))
      return false;

   const size_t scissor_offset =
      (size_t)scissor_index * AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH;
   const size_t scissor_size =
      (size_t)viewport_count * AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH;
   if (scissor_offset > cs->scissor.size ||
       scissor_size > cs->scissor.size - scissor_offset)
      return false;

   const uint64_t scissor_base =
      hk_pool_upload(cmd, (const uint8_t *)cs->scissor.data + scissor_offset,
                     scissor_size, 64);

   static const uint8_t zero_depth_bias[AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH];
   const void *depth_bias = zero_depth_bias;
   if (cs->depth_bias.size >= AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH) {
      depth_bias = (const uint8_t *)cs->depth_bias.data + cs->depth_bias.size -
                   AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH;
   }

   const uint64_t depth_bias_base =
      hk_pool_upload(cmd, depth_bias,
                     AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH, 64);
   if (!scissor_base || !depth_bias_base)
      return false;

   struct agx_ppp_hal300_depth_bias_scissor_bases bases;
   const struct agx_ppp_hal300_depth_bias_scissor_bases_fields base_fields = {
      .depth_bias_base = depth_bias_base,
      .scissor_base = scissor_base,
   };
   if (!agx_ppp_hal300_depth_bias_scissor_bases_pack(chip, &base_fields,
                                                     &bases))
      return false;

   struct agx_ppp_viewport_payload_fields viewport_fields = {
      .viewport_count = viewport_count,
   };

   for (unsigned i = 0; i < viewport_count; ++i) {
      uint8_t region_clip[AGX_REGION_CLIP_LENGTH];
      agx_pack(region_clip, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
      memcpy(&viewport_fields.region_clip_raw[i], region_clip,
             sizeof(region_clip));

      const VkViewport *vp = &viewports[i];
      viewport_fields.viewports[i] =
         (struct agx_ppp_viewport_transform_raw){
            .translate_x = fui(vp->x + 0.5f * vp->width),
            .scale_x = fui(vp->width * 0.5f),
            .translate_y = fui(vp->y + 0.5f * vp->height),
            .scale_y = fui(vp->height * 0.5f),
            .translate_z = fui(vp->minDepth),
            .scale_z = fui(vp->maxDepth - vp->minDepth),
         };
   }

   uint8_t viewport_control[AGX_VIEWPORT_CONTROL_LENGTH];
   agx_pack(viewport_control, VIEWPORT_CONTROL, cfg)
      ;
   memcpy(&viewport_fields.viewport_control_raw, viewport_control,
          sizeof(viewport_control));

   struct agx_ppp_viewport_payload_core viewport;
   if (!agx_ppp_viewport_payload_core_pack(chip, &viewport_fields, &viewport))
      return false;

   const uint64_t bases_gpu =
      hk_pool_upload(cmd, bases.bytes, bases.length, AGX_PPP_HEADER_ALIGN);
   const uint64_t viewport_gpu =
      hk_pool_upload(cmd, viewport.bytes, viewport.length,
                     AGX_PPP_HEADER_ALIGN);
   if (!bases_gpu || !viewport_gpu)
      return false;

   /* Keep emission transactional: both exact PPP pointers are staged before
    * the caller's live control-stream cursor is advanced. HAL300's initial
    * render-state path emits the viewport update before the base update. */
   uint8_t records[2 * AGX_PPP_STATE_LENGTH];
   uint8_t *record = records;
   if (!agx_ppp_emit_state(chip, &record, viewport_gpu, viewport.length) ||
       !agx_ppp_emit_state(chip, &record, bases_gpu, bases.length) ||
       record != records + sizeof(records))
      return false;

   memcpy(*out, records, sizeof(records));
   *out += sizeof(records);
   return true;
}

static void
hk_flush_vp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const bool hal300 =
      agx_chip_command_stream_generation(dev->dev.chip) ==
      AGX_CS_GENERATION_HAL300;
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   static_assert(AGX_SCISSOR_LENGTH ==
                 AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH);

   /* We always need at least 1 viewport for the hardware. With rasterizer
    * discard the app may not supply any, but we can just program garbage.
    */
   unsigned count = MAX2(dyn->vp.viewport_count, 1);

   unsigned minx[HK_MAX_VIEWPORTS] = {0}, miny[HK_MAX_VIEWPORTS] = {0};
   unsigned maxx[HK_MAX_VIEWPORTS] = {0}, maxy[HK_MAX_VIEWPORTS] = {0};

   /* We implicitly scissor to the viewport. We need to do a min/max dance to
    * handle inverted viewports.
    */
   for (uint32_t i = 0; i < dyn->vp.viewport_count; i++) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      minx[i] = MIN2(vp->x, vp->x + vp->width);
      miny[i] = MIN2(vp->y, vp->y + vp->height);
      maxx[i] = MAX2(vp->x, vp->x + vp->width);
      maxy[i] = MAX2(vp->y, vp->y + vp->height);
   }

   /* Additionally clamp to the framebuffer so we don't rasterize
    * off-screen pixels. TODO: Is this necessary? the GL driver does this but
    * it might be cargoculted at this point.
    */
   for (unsigned i = 0; i < count; ++i) {
      minx[i] = MIN2(minx[i], cmd->state.gfx.render.cr.width);
      maxx[i] = MIN2(maxx[i], cmd->state.gfx.render.cr.width);
      miny[i] = MIN2(miny[i], cmd->state.gfx.render.cr.height);
      maxy[i] = MIN2(maxy[i], cmd->state.gfx.render.cr.height);
   }

   /* We additionally apply any API scissors */
   for (unsigned i = 0; i < dyn->vp.scissor_count; ++i) {
      const VkRect2D *s = &dyn->vp.scissors[i];

      minx[i] = MAX2(minx[i], s->offset.x);
      miny[i] = MAX2(miny[i], s->offset.y);
      maxx[i] = MIN2(maxx[i], s->offset.x + s->extent.width);
      maxy[i] = MIN2(maxy[i], s->offset.y + s->extent.height);
   }

   /* Upload a hardware scissor for each viewport, whether there's a
    * corresponding API scissor or not.
    */
   unsigned index = cs->scissor.size / AGX_SCISSOR_LENGTH;
   uint8_t *scissors =
      util_dynarray_grow_bytes(&cs->scissor, count, AGX_SCISSOR_LENGTH);

   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      /* These settings in conjunction with the PPP control depth clip/clamp
       * settings implement depth clip/clamping. Properly setting them
       * together is required for conformant depth clip enable.
       *
       * TODO: Reverse-engineer the finer interactions here.
       */
      const float min_z = dyn->rs.depth_clamp_enable
                             ? MIN2(vp->minDepth, vp->maxDepth)
                             : 0.0f;
      const float max_z = dyn->rs.depth_clamp_enable
                             ? MAX2(vp->minDepth, vp->maxDepth)
                             : 1.0f;
      uint8_t *record = scissors + i * AGX_SCISSOR_LENGTH;

      if (hal300) {
         const struct agx_ppp_hal300_scissor_record_fields fields = {
            .max_x = maxx[i],
            .min_x = minx[i],
            .max_y = maxy[i],
            .min_y = miny[i],
            .min_z_raw = fui(min_z),
            .max_z_raw = fui(max_z),
         };
         struct agx_ppp_hal300_scissor_record exact;
         if (!agx_ppp_hal300_scissor_record_pack(dev->dev.chip, &fields,
                                                  &exact)) {
            mesa_loge("no grounded HAL300 scissor record for %s",
                      agx_chip_name(dev->dev.chip));
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         memcpy(record, exact.bytes, exact.length);
      } else {
         agx_pack(record, SCISSOR, cfg) {
            cfg.min_x = minx[i];
            cfg.min_y = miny[i];
            cfg.max_x = maxx[i];
            cfg.max_y = maxy[i];
            cfg.min_z = min_z;
            cfg.max_z = max_z;
         }
      }
   }

   if (hal300) {
      if (!hk_emit_hal300_vp_state(cmd, cs, dev->dev.chip, count, index, minx,
                                   miny, maxx, maxy, dyn->vp.viewports, out)) {
         mesa_loge("no grounded HAL300 viewport/base emitter for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      }
      return;
   }

   /* Upload state */
   struct AGX_PPP_HEADER present = {
      .depth_bias_scissor = true,
      .region_clip = true,
      .viewport = true,
      .viewport_count = count,
   };

   size_t size = agx_ppp_update_size(&present);
   struct agx_ptr T = hk_pool_vdm_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &present);

   agx_ppp_push(&ppp, DEPTH_BIAS_SCISSOR, cfg) {
      cfg.scissor = index;

      /* Use the current depth bias, we allocate linearly */
      unsigned count = cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH;
      cfg.depth_bias = count ? count - 1 : 0;
   };

   for (unsigned i = 0; i < count; ++i) {
      agx_ppp_push(&ppp, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
   }

   agx_ppp_push(&ppp, VIEWPORT_CONTROL, cfg)
      ;

   /* Upload viewports */
   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      agx_ppp_push(&ppp, VIEWPORT, cfg) {
         cfg.translate_x = vp->x + 0.5f * vp->width;
         cfg.translate_y = vp->y + 0.5f * vp->height;
         cfg.translate_z = vp->minDepth;

         cfg.scale_x = vp->width * 0.5f;
         cfg.scale_y = vp->height * 0.5f;
         cfg.scale_z = vp->maxDepth - vp->minDepth;
      }
   }

   const bool ppp_ok = agx_ppp_legacy_payload_supported(dev->dev.chip) &&
                       agx_ppp_fini(dev->dev.chip, out, &ppp);
   if (!ppp_ok) {
      mesa_loge("no grounded legacy PPP payload/finalizer for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   } else if (hk_g17p_launch_trace_enabled()) {
      hk_g17p_trace_line("ppp-viewport", ppp.gpu_base, ppp.cpu_base,
                         ppp.total_size);
   }
}

static enum agx_object_type
translate_object_type(enum mesa_prim topology)
{
   static_assert(MESA_PRIM_LINES < MESA_PRIM_LINE_STRIP);
   static_assert(MESA_PRIM_TRIANGLES >= MESA_PRIM_LINE_STRIP);

   if (topology == MESA_PRIM_POINTS)
      return AGX_OBJECT_TYPE_POINT_SPRITE_UV01;
   else if (topology <= MESA_PRIM_LINE_STRIP)
      return AGX_OBJECT_TYPE_LINE;
   else
      return AGX_OBJECT_TYPE_TRIANGLE;
}

static enum agx_primitive
translate_hw_primitive_topology(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return AGX_PRIMITIVE_POINTS;
   case MESA_PRIM_LINES:
      return AGX_PRIMITIVE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return AGX_PRIMITIVE_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return AGX_PRIMITIVE_TRIANGLES;
   case MESA_PRIM_TRIANGLE_STRIP:
      return AGX_PRIMITIVE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return AGX_PRIMITIVE_TRIANGLE_FAN;
   default:
      UNREACHABLE("Invalid hardware primitive topology");
   }
}

static inline enum agx_vdm_vertex
translate_vdm_vertex(unsigned vtx)
{
   static_assert(AGX_VDM_VERTEX_0 == 0);
   static_assert(AGX_VDM_VERTEX_1 == 1);
   static_assert(AGX_VDM_VERTEX_2 == 2);

   assert(vtx <= 2);
   return vtx;
}

static inline enum agx_ppp_vertex
translate_ppp_vertex(unsigned vtx)
{
   static_assert(AGX_PPP_VERTEX_0 == 0 + 1);
   static_assert(AGX_PPP_VERTEX_1 == 1 + 1);
   static_assert(AGX_PPP_VERTEX_2 == 2 + 1);

   assert(vtx <= 2);
   return vtx + 1;
}

static void
hk_flush_index(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_api_shader *gs = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
   uint32_t index = cmd->state.gfx.index.restart;

   if (gs) {
      enum poly_gs_shape shape =
         gs->variants[HK_GS_VARIANT_COUNT].info.gs.shape;
      index = BITFIELD_MASK(8 * poly_gs_index_size(shape));
   }

   /* VDM State updates are relatively expensive, so only emit them when the
    * restart index changes. This is simpler than accurate dirty tracking.
    */
   if (cs->restart_index != index) {
      uint8_t *out = cs->current;
      agx_push(out, VDM_STATE, cfg) {
         cfg.restart_index_present = true;
      }

      agx_push(out, VDM_STATE_RESTART_INDEX, cfg) {
         cfg.value = index;
      }

      cs->current = out;
      cs->restart_index = index;
   }
}

static float hk_depth_bias_factor(VkFormat format, bool exact,
                                  bool force_unorm);

/* Every refusal below is a real capability gap that silently drops the draw.
 * Report the exact check that refused, once per site, so a failing
 * application names a specific unimplemented G17P bind-tail field
 * instead of a single opaque message. */
static __attribute__((cold, noinline)) bool
hk_g17p_bind_tail_refused(unsigned line)
{
   static uint32_t reported[64];
   static unsigned nr_reported;
   for (unsigned i = 0; i < nr_reported; ++i) {
      if (reported[i] == line)
         return false;
   }
   if (nr_reported < ARRAY_SIZE(reported))
      reported[nr_reported++] = line;
   mesa_loge("G17P native bind tail refused at hk_cmd_draw.c:%u", line);
   return false;
}

static bool
hk_g17p_nonpoint_no_psiz_diagnostic(bool enabled,
                                    const struct hk_graphics_state *gfx,
                                    const struct hk_shader *vs)
{
   if (!vs || !enabled ||
       gfx->topology == AGX_PRIMITIVE_POINTS)
      return false;

   /* A terminal point-size word can be removed without shifting any later
    * layer/viewport or clip output. */
   return vs->info.uvs.group_offs[UVS_PSIZ] + 1 ==
             vs->info.uvs.group_offs[UVS_LAYER_VIEWPORT] &&
          vs->info.uvs.group_offs[UVS_LAYER_VIEWPORT] ==
             vs->info.uvs.group_offs[UVS_CLIP_DIST] &&
          vs->info.uvs.group_offs[UVS_CLIP_DIST] == vs->info.uvs.size;
}

static bool
hk_g17p_bind_tail_state_matches(
   const struct hk_cmd_buffer *cmd,
   const struct hk_g17p_prepared_bind_tail *prepared);

static bool
hk_g17p_build_native_bind_tail(struct hk_cmd_buffer *cmd,
                               struct hk_g17p_prepared_bind_tail *prepared)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct hk_rendering_state *render = &gfx->render;
   const unsigned viewport_count = MAX2(dyn->vp.viewport_count, 1);

   /* The record is command-buffer-owned and every field consumed by a valid
    * snapshot is overwritten below.  Invalidating it is sufficient on entry;
    * clearing the complete viewport arrays here made every rebuild pay for
    * unused viewport slots as well. */
   if (prepared)
      prepared->valid = false;

   if (dev->dev.chip != AGX_CHIP_G17P || !prepared)
      return hk_g17p_bind_tail_refused(__LINE__);
   if (!hk_g17p_viewport_state_supported(dyn->vp.viewport_count,
                                         dyn->vp.scissor_count))
      return hk_g17p_bind_tail_refused(__LINE__);

   struct hk_shader *fs = hk_bound_fs(cmd);
   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   if (!fs || !linked_fs)
      return hk_g17p_bind_tail_refused(__LINE__);

   const uint64_t fragment_program = gfx->prepared_fragment_usc
      ? gfx->prepared_fragment_usc
      : hk_upload_usc_words(cmd, fs, linked_fs);
   gfx->prepared_fragment_usc = 0;
   if (!fragment_program)
      return hk_g17p_bind_tail_refused(__LINE__);
   if ((fragment_program & UINT64_C(0x3f)) != 0 ||
       fragment_program >= (UINT64_C(1) << 48))
      return hk_g17p_bind_tail_refused(__LINE__);

   /* Every input below except the fragment stage program is identical to the
    * last successful build: only the root, descriptor or vertex-buffer
    * identity moved (a descriptor push or vertex-buffer bind), and those reach
    * the tail only through fragment_program above.  Repack the fragment
    * record and upload the same bind group with the same allocation. */
   if (gfx->g17p_bind_tail_memo.valid &&
       gfx->g17p_bind_tail_memo.fs == fs &&
       gfx->g17p_bind_tail_memo.hw_vs == hk_bound_hw_vs(gfx) &&
       hk_g17p_bind_tail_state_matches(cmd,
                                       &gfx->g17p_bind_tail_memo.snapshot)) {
      struct agx_ppp_hal200_fragment_fields memo_fs_fields =
         gfx->g17p_bind_tail_memo.fs_fields;
      memo_fs_fields.stage_program = fragment_program;
      struct agx_ppp_hal200_graphics_core memo_fs_record;
      if (!agx_ppp_hal200_fragment_core_pack(dev->dev.chip, &memo_fs_fields,
                                             &memo_fs_record))
         return hk_g17p_bind_tail_refused(__LINE__);

      uint8_t memo_bind_group[0x80];
      memcpy(memo_bind_group, gfx->g17p_bind_tail_memo.bind_group,
             sizeof(memo_bind_group));
      memcpy(&memo_bind_group[0x00], memo_fs_record.bytes,
             memo_fs_record.length);

      const struct agx_ppp_viewport_payload_core *memo_viewport =
         &gfx->g17p_bind_tail_memo.viewport_record;
      const size_t memo_state_size =
         agx_g17p_bind_state_size(memo_viewport->length);
      struct agx_ptr memo_state =
         hk_pool_vdm_alloc(cmd, memo_state_size, 64);
      struct hk_g17p_bind_state_upload memo_uploaded;
      if (!memo_state_size || !memo_state.cpu ||
          !hk_g17p_upload_bind_state(
             memo_state.cpu, memo_state_size, memo_state.gpu,
             memo_bind_group, memo_viewport->bytes, memo_viewport->length,
             gfx->linked_varyings.counts_32.opaque[0],
             gfx->linked_varyings.counts_16.opaque[0], &memo_uploaded))
         return hk_g17p_bind_tail_refused(__LINE__);

      if (dev->g17p_diagnostics.launch_trace) {
         hk_g17p_trace_line("hal200-bind-group", memo_state.gpu,
                            memo_state.cpu, sizeof(memo_bind_group));
         hk_g17p_trace_line("hal200-fragment-bind", fragment_program,
                            memo_fs_record.bytes, memo_fs_record.length);
         hk_g17p_trace_line("hal200-viewport",
                            memo_state.gpu + HK_G17P_BIND_GROUP_BYTES,
                            (uint8_t *)memo_state.cpu +
                               HK_G17P_BIND_GROUP_BYTES,
                            memo_viewport->length);
      }

      *prepared = gfx->g17p_bind_tail_memo.snapshot;
      prepared->tail = memo_uploaded.tail;
      prepared->dynamic_generation = dyn->generation;
      prepared->descriptor_generation = gfx->descriptors.generation;
      prepared->vertex_buffer_generation = gfx->vertex_buffer_generation;
      prepared->root = gfx->root;
      prepared->valid = true;
      if (dev->g17p_diagnostics.launch_trace) {
         const struct agx_vdm_hal200_bind_pair *pairs = memo_uploaded.pairs;
         fprintf(stderr,
                 "G17P HAL200 bind tail: bind0=%08x fs=%08x output=%08x "
                 "primary=%08x secondary=%08x viewport=%08x wcull2=%08x "
                 "cull=%08x\n",
                 pairs[0].context_offset, pairs[1].context_offset,
                 pairs[2].context_offset, pairs[3].context_offset,
                 pairs[4].context_offset, pairs[5].context_offset,
                 pairs[6].context_offset, pairs[7].context_offset);
      }
      return true;
   }
   if (!hk_g17p_smooth_varyings_supported(cmd, linked_fs)) {
      mesa_loge_once(
         "G17P varying layout refused: bindings=%u cf=%u reads_z=%u "
         "table=%" PRIx64 " topology=%u provoking=%u",
         linked_fs->b.cf.nr_bindings, linked_fs->b.cf.nr_cf,
         linked_fs->b.cf.reads_z, gfx->varyings, gfx->topology,
         gfx->provoking);
      return hk_g17p_bind_tail_refused(__LINE__);
   }
   if ((linked_fs->b.cf.nr_bindings != 0) != (gfx->varyings != 0))
      return hk_g17p_bind_tail_refused(__LINE__);

   /* This record describes the program actually launched by the Index List.
    * Ordinary draws use the API vertex shader, while GS/XFB draws use the
    * geometry raster-copy variant. Taking the API VS unconditionally makes
    * the HAL200 output contract disagree with the live VertexProgramVariant
    * as soon as interstage optimization changes their UVS footprints. */
   const struct hk_shader *vs = hk_bound_hw_vs(gfx);
   if (!vs)
      return hk_g17p_bind_tail_refused(__LINE__);

   /* Native EXP-O2A changes both point-size enable and the output count with
    * the draw primitive: triangles do not consume the terminal point-size
    * word. Dynamic-topology pipelines currently retain the default
    * gl_PointSize write in their one HW variant, so provide an exact,
    * default-off A/B for that missing variant selection. Restrict it to a
    * terminal point-size word with no layer/viewport or clip outputs; changing
    * a middle UVS word would require rewriting the remaining output layout. */
   const bool nonpoint_no_psiz =
      hk_g17p_nonpoint_no_psiz_diagnostic(
         dev->g17p_diagnostics.nonpoint_no_psiz, gfx, vs);
   const uint32_t active_output_size =
      vs->info.uvs.size - (nonpoint_no_psiz ? 1 : 0);
   const uint32_t active_output_select =
      vs->info.uvs.osel.opaque[0] &
      ~(nonpoint_no_psiz ? BITFIELD_BIT(18) : 0);

   uint32_t output_size;
   if (!agx_ppp_hal200_output_size_pack(
          active_output_size, 0, &output_size))
      return hk_g17p_bind_tail_refused(__LINE__);

   uint32_t output_select;
   if (!agx_ppp_hal200_output_select_pack(
          dev->dev.chip, active_output_select,
          linked_fs->b.cf.nr_bindings != 0, linked_fs->b.cf.reads_z,
          &output_select))
      return hk_g17p_bind_tail_refused(__LINE__);

   if (nonpoint_no_psiz)
      mesa_logi_once("G17P non-point terminal PointSize A/B active");

   /* Trace-only falsifier for the deterministic Omarchy About VF_CPF fault.
    * The retained faulting trace has exactly this sampled-fragment contract:
    * two coefficient bindings, position + smooth float2 + point size (size 7,
    * selector 0x50000), and a zero final FragmentProgram record word.  Metal
    * copies the low nibble of compiler metadata field 32 into this word;
    * retained tex2/vary3 records carry 1.  Mesa has no demonstrated semantic
    * equivalent for that compiler-profile class, so keep this opt-in and
    * shape-specific. */
   const bool about_flags1_diagnostic =
      dev->g17p_diagnostics.hal200_about_flags1 &&
      fs->b.info.g17p_sampled_fragment &&
      linked_fs->b.cf.nr_bindings == 2 && !linked_fs->b.cf.reads_z &&
      output_size == 7 && output_select == UINT32_C(0x00050000);
   if (about_flags1_diagnostic)
      mesa_loge_once("G17P HAL200 About flags=1 diagnostic active");

   struct agx_ppp_hal200_graphics_core fs_record;
   struct agx_ppp_hal200_fragment_fields fs_fields = {
      /* HAL200 finalize 0x3f04e4..0x3f0508 puts iteration count at bit 16.
       * The packer merges the complete VIT address, including its high bits. */
      .resource_counts = linked_fs->b.cf.nr_bindings << 16,
      .varying_table = gfx->varyings,
      .unknown_0c = 0,
      .stage_program = fragment_program,
      .stage_program_template = 0,
      .flags = about_flags1_diagnostic ? BITFIELD_BIT(0) : 0,
   };
   if (fs->b.info.has_preamble &&
       (!fs->b.info.agx3_preamble_staging_known ||
        !agx_ppp_hal200_fragment_ti_resources(
          dev->dev.chip, fs->b.info.preamble_scratch_size,
          fs->b.info.agx3_preamble_staging_halfs * 2u, &fs_fields)))
      return hk_g17p_bind_tail_refused(__LINE__);

   struct agx_ppp_hal200_graphics_core output_record;
   struct agx_ppp_hal200_graphics_fields output_fields = {
      /* Native vertex update 0x481be8..0x481c2c writes zero at
       * context+0x1f90 and viewport_count-1 at +0x1f94; both are zero
       * for this single-viewport layout. VIT supplies interpolation.
       * FragmentProgramVariant 0x373eb4..0x373ee0 sets output-select bit16
       * for nonempty iteration metadata, merged by 0x4824fc..0x482518. */
      .words = {output_select,
                0, 0, output_size,
                AGX_PPP_HAL200_PRIMARY_FACE_HEADER, 0},
   };
   if (!agx_ppp_hal200_viewport_count_pack(
          dev->dev.chip, viewport_count, &output_fields.words[2]))
      return hk_g17p_bind_tail_refused(__LINE__);

   struct agx_ppp_hal200_graphics_core primary_record;
   struct agx_ppp_hal200_graphics_fields primary_fields = {
      .words = {0, UINT32_C(0x07200f00),
                UINT32_C(0x0e000000), UINT32_C(0x07200f00),
                UINT32_C(0x0e000000), 0},
   };

   /* Each draw already owns its complete PRIMARY_FACE record. Native depth
    * control only changes these two words; retain the allocation and staging
    * layout and do not add launches, cache flushes or device-global state. */
   const bool depth_test = render->depth_att.iview && dyn->ds.depth.test_enable;
   /* A shader with side effects under VK_COMPARE_OP_NEVER emulates the test
    * by exporting NaN depth (agx_nir_lower_frag_sidefx); the hardware test
    * must then pass everything, as FRAGMENT_FACE does, or the fragments
    * never run (rasterization.frag_side_effects.*.depth_never). */
   const VkCompareOp depth_compare =
      gfx->descriptors.root.draw.force_never_in_shader
         ? VK_COMPARE_OP_ALWAYS : dyn->ds.depth.compare_op;
   if (dyn->ds.depth.bounds_test.enable ||
       !agx_ppp_hal200_depth_face_pack(
          dev->dev.chip, depth_test, dyn->ds.depth.write_enable,
          depth_compare, primary_fields.words[1],
          &primary_fields.words[1]) ||
       !agx_ppp_hal200_depth_face_pack(
          dev->dev.chip, depth_test, dyn->ds.depth.write_enable,
          depth_compare, primary_fields.words[3],
          &primary_fields.words[3]))
      return hk_g17p_bind_tail_refused(__LINE__);

   const bool stencil_test = hk_stencil_test_enabled(cmd);
   const struct vk_stencil_test_face_state stencil_faces[2] = {
      dyn->ds.stencil.front, dyn->ds.stencil.back,
   };
   for (unsigned i = 0; i < 2; ++i) {
      const struct vk_stencil_test_face_state *face = &stencil_faces[i];
      const unsigned face_word = 1 + 2 * i;
      /* Native458734/458740 updates only the low reference byte. */
      primary_fields.words[face_word] =
         (primary_fields.words[face_word] & ~0xffu) | (face->reference & 0xffu);
      primary_fields.words[face_word + 1] = agx_ppp_hal200_stencil_pack(
         stencil_test, face->op.compare, face->op.fail, face->op.depth_fail,
         face->op.pass, face->compare_mask, face->write_mask);
   }

   struct agx_ppp_hal200_graphics_core secondary_record;
   struct agx_ppp_hal200_graphics_fields secondary_fields = {
      .words = {UINT32_C(0x00000200), UINT32_C(0x07e00000),
                UINT32_C(0x07e00000), UINT32_C(0x0001ffff)},
   };

   /* UNCHANGED permits early depth rejection. DEPTHF shaders must publish
    * their actual depth contract here, including the forced-ANY contract of
    * late side effects. This is independent of pass type and triangle merging. */
   const unsigned conservative_depth =
      agx_translate_depth_layout(fs->b.info.depth_layout);
   for (unsigned face = 1; face <= 2; ++face) {
      if (!agx_ppp_hal200_fragment_face_depth_pack(
             dev->dev.chip, conservative_depth, fs->b.info.early_fragment_tests,
             fs->info.fs.writes_memory, secondary_fields.words[face],
             &secondary_fields.words[face]))
         return hk_g17p_bind_tail_refused(__LINE__);
   }

   const unsigned line_width = agx_pack_line_width(dyn->rs.line.width);
   const unsigned polygon_mode = dyn->rs.polygon_mode;
   const unsigned object_type = gfx->object_type;
   if (!agx_ppp_hal200_fragment_face_raster_pack(
          dev->dev.chip, line_width, polygon_mode, primary_fields.words[1],
          &primary_fields.words[1]) ||
       !agx_ppp_hal200_fragment_face_raster_pack(
          dev->dev.chip, line_width, polygon_mode, primary_fields.words[3],
          &primary_fields.words[3]) ||
       !agx_ppp_hal200_fragment_face_object_pack(
          dev->dev.chip, object_type, secondary_fields.words[1],
          &secondary_fields.words[1]) ||
       !agx_ppp_hal200_fragment_face_object_pack(
          dev->dev.chip, object_type, secondary_fields.words[2],
          &secondary_fields.words[2]))
      return hk_g17p_bind_tail_refused(__LINE__);

   struct agx_ppp_hal200_graphics_core cull_record;
   struct agx_ppp_hal200_graphics_fields cull_fields = {
      .words = {UINT32_C(0x00000480)},
   };
   /* Software tessellation emits its own winding; fold the domain's
    * orientation into the front face as the common CULL packet does
    * (shader_object.misc.state.*.vert_tess_frag.cull.*). */
   bool front_face_ccw = dyn->rs.front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;
   if (gfx->shaders[MESA_SHADER_TESS_CTRL] &&
       !gfx->shaders[MESA_SHADER_GEOMETRY]) {
      front_face_ccw ^= gfx->tess.info.ccw;
      front_face_ccw ^= dyn->ts.domain_origin ==
                        VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }
   if (!agx_ppp_hal200_cull_mode_pack(
          dev->dev.chip, dyn->rs.cull_mode, front_face_ccw,
          cull_fields.words[0], &cull_fields.words[0]))
      return hk_g17p_bind_tail_refused(__LINE__);
   if (!agx_ppp_hal200_cull_provoking_pack(dev->dev.chip, gfx->provoking,
                                         cull_fields.words[0],
                                         &cull_fields.words[0]))
      return hk_g17p_bind_tail_refused(__LINE__);
   if (!agx_ppp_hal200_depth_clip_pack(
          dev->dev.chip,
          vk_rasterization_state_depth_clip_enable(&dyn->rs),
          cull_fields.words[0], &cull_fields.words[0]))
      return hk_g17p_bind_tail_refused(__LINE__);
   if (!agx_ppp_hal200_rasterizer_discard_pack(
          dev->dev.chip, hk_rast_discard(cmd), secondary_fields.words[0],
          cull_fields.words[0], &secondary_fields.words[0],
          &cull_fields.words[0]))
      return hk_g17p_bind_tail_refused(__LINE__);
   if (!agx_ppp_hal200_primitive_msaa_pack(
          dev->dev.chip,
          object_type == AGX_OBJECT_TYPE_LINE &&
             dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR,
          cull_fields.words[0], &cull_fields.words[0]))
      return hk_g17p_bind_tail_refused(__LINE__);

   struct agx_ppp_w_clamp_cull2_core w_cull2_record;
   const struct agx_ppp_w_clamp_cull2_fields w_cull2_fields = {
      .w_clamp_raw = 0,
      /* The generated primitive-ID coefficient needs its rasterizer input
       * enabled as well as the matching VDM producer and output slot. */
      .cull2_raw = UINT32_C(0x80) |
                   (gfx->generate_primitive_id ? UINT32_C(0x1000) : 0),
   };

   struct agx_ppp_viewport_payload_fields viewport_fields = {
      .viewport_count = viewport_count,
   };
   unsigned minx[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS] = {0};
   unsigned miny[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS] = {0};
   unsigned maxx[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS] = {0};
   unsigned maxy[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS] = {0};
   for (unsigned i = 0; i < viewport_count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];
      minx[i] = MIN2(vp->x, vp->x + vp->width);
      miny[i] = MIN2(vp->y, vp->y + vp->height);
      maxx[i] = MAX2(vp->x, vp->x + vp->width);
      maxy[i] = MAX2(vp->y, vp->y + vp->height);
      minx[i] = MIN2(minx[i], render->cr.width);
      miny[i] = MIN2(miny[i], render->cr.height);
      maxx[i] = MIN2(maxx[i], render->cr.width);
      maxy[i] = MIN2(maxy[i], render->cr.height);
      if (i < dyn->vp.scissor_count) {
         const VkRect2D *s = &dyn->vp.scissors[i];
         minx[i] = MAX2(minx[i], s->offset.x);
         miny[i] = MAX2(miny[i], s->offset.y);
         maxx[i] = MIN2(maxx[i], s->offset.x + s->extent.width);
         maxy[i] = MIN2(maxy[i], s->offset.y + s->extent.height);
      }
   }

   const VkViewport *vp = &dyn->vp.viewports[0];
   const struct hk_image_view *color = render->color_att_count == 1
                                         ? render->color_att[0].iview : NULL;
   const VkRect2D api_scissor = dyn->vp.scissor_count
                                  ? dyn->vp.scissors[0] : (VkRect2D){0};
   const struct hk_g17p_full_frame_scissor full_frame = {
      .width = render->cr.width,
      .height = render->cr.height,
      .area_x = render->area.offset.x,
      .area_y = render->area.offset.y,
      .area_width = render->area.extent.width,
      .area_height = render->area.extent.height,
      .scissor_x = api_scissor.offset.x,
      .scissor_y = api_scissor.offset.y,
      .scissor_width = api_scissor.extent.width,
      .scissor_height = api_scissor.extent.height,
      .viewport_x = vp->x,
      .viewport_y = vp->y,
      .viewport_width = vp->width,
      .viewport_height = vp->height,
      .min_depth = vp->minDepth,
      .max_depth = vp->maxDepth,
      .viewport_count = dyn->vp.viewport_count,
      .scissor_count = dyn->vp.scissor_count,
      .single_color_layer_sample =
         color && !render->depth_att.iview && !render->stencil_att.iview &&
         render->layer_count == 1 && !render->view_mask &&
         render->tilebuffer.nr_samples == 1 &&
         u_minify(color->vk.image->extent.width, color->vk.base_mip_level) ==
            render->cr.width &&
         u_minify(color->vk.image->extent.height, color->vk.base_mip_level) ==
            render->cr.height,
      .default_depth =
         vk_rasterization_state_depth_clip_enable(&dyn->rs) &&
         !dyn->rs.depth_clamp_enable && !dyn->vp.depth_clip_negative_one_to_one &&
         !dyn->rs.depth_bias.enable && !hk_rast_discard(cmd),
   };
   const bool scissor_enable = dyn->vp.viewport_count > 0 &&
                              !hk_g17p_full_frame_scissor_can_elide(&full_frame);
   const bool depth_bias_enable =
      dyn->rs.depth_bias.enable &&
      gfx->object_type == AGX_OBJECT_TYPE_TRIANGLE;
   const bool exact = dyn->rs.depth_bias.exact;
   const bool force_unorm =
      dyn->rs.depth_bias.representation ==
      VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;
   const float depth_bias =
      depth_bias_enable
         ? dyn->rs.depth_bias.constant_factor /
              hk_depth_bias_factor(render->depth_att.vk_format, exact,
                                   force_unorm)
         : 0.0f;
   struct agx_ppp_hal200_raster_state
      raster_states[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
   for (unsigned i = 0; i < viewport_count; ++i) {
      const VkViewport *current_vp = &dyn->vp.viewports[i];
      const float min_z = dyn->rs.depth_clamp_enable
                             ? MIN2(current_vp->minDepth, current_vp->maxDepth)
                             : 0.0f;
      const float max_z = dyn->rs.depth_clamp_enable
                             ? MAX2(current_vp->minDepth, current_vp->maxDepth)
                             : 1.0f;
      const struct agx_ppp_hal200_raster_fields raster_fields = {
         .scissor_enable = scissor_enable,
         .depth_bias_enable = depth_bias_enable,
         .max_x = MIN2(maxx[i], UINT16_MAX),
         .min_x = MIN2(minx[i], UINT16_MAX),
         .max_y = MIN2(maxy[i], UINT16_MAX),
         .min_y = MIN2(miny[i], UINT16_MAX),
         .min_z_raw = fui(min_z),
         .max_z_raw = fui(max_z),
         .depth_bias_raw = fui(depth_bias),
         .slope_scale_raw =
            depth_bias_enable ? fui(dyn->rs.depth_bias.slope_factor) : 0,
         .clamp_raw = depth_bias_enable ? fui(dyn->rs.depth_bias.clamp) : 0,
      };
      if (!agx_ppp_hal200_raster_state_build(
             dev->dev.chip, &raster_fields, &raster_states[i]))
         return hk_g17p_bind_tail_refused(__LINE__);
   }
   const struct agx_ppp_hal200_raster_state *raster_state = &raster_states[0];

   /* Native update 0x46c6ac..0x46c6dc stores pass type at context+0x1f78.
    * The 0x46ff78 path emits it in SECONDARY header 0x02000048's first
    * payload word, masked by 0xe6600000 and ORed with 0x200. It is not the
    * older-generation FRAGMENT_CONTROL location in our PRIMARY record.
    */
   const uint32_t linked_fragment_control =
      linked_fs->b.fragment_control.opaque[0];
   const uint32_t pass_type = linked_fragment_control >> 29;
   if (pass_type != AGX_PASS_TYPE_OPAQUE &&
       pass_type != AGX_PASS_TYPE_TRANSLUCENT &&
       pass_type != AGX_PASS_TYPE_PUNCH_THROUGH &&
       pass_type != AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH)
      return hk_g17p_bind_tail_refused(__LINE__);
   /* Native458934..45896c retains both stencil-control bits in the
    * same per-draw record as depth. No extra state allocation or launch. */
   uint32_t fragment_control = raster_state->fragment_control |
      (stencil_test ? UINT32_C(0x000c0000) : 0);
   const struct agx_ppp_hal200_fragment_control_fields control_fields = {
      .linked_control = linked_fragment_control,

      /* Apple G17P shader metadata leaves bit26 clear for texture gradients,
       * storage writes, framebuffer fetch, sample-rate execution, and shadows.
       * Multisampled depth/depth-stencil variants set it.
       *
       * A shader that takes implicit quad derivatives still needs it on this
       * stack. Triangle merging shades a 2x2 quad straddling the shared edge
       * of two primitives once, with its lanes interpolated from two different
       * plane equations, so an implicit-LOD sample on that quad derives its
       * mip level across the seam.
       *
       * Measured on G17P, 2026-09-18: without this term all 30
       * dEQP-VK.texture.mipmap.{2d,3d,cubemap}.projected.* cases fail and
       * every invalid pixel lies exactly on the shared diagonal of the test's
       * two triangles (31/31 in the decoded error mask of
       * 2d.projected.nearest_linear_clamp); with it, all 30 pass and the 18
       * basic/affine siblings are unchanged. Those siblings pass either way
       * because both of their triangles carry the same affine texcoord map, so
       * the cross-seam derivative equals the correct one.
       *
       * AGX2 never had the defect: agx_compile.c derives disable_tri_merging
       * from needs_coarse_quad_helper_invocations, which nir_gather_info sets
       * for every implicit-derivative texture op; agx_linker.c ORs it across
       * the shader parts into the linked FRAGMENT_CONTROL bit 26; and
       * hk_flush_ppp_state() merges that word into the secondary control.
       * G17P never reaches it -- the draw path calls hk_flush_ppp_state() only
       * `if (!g17p_native_bind_tail_emitted)` -- so the whole control comes
       * from this record instead, and agx_ppp_hal200_fragment_controls_pack()
       * deliberately masks bit 26 out of the original and rebuilds it from the
       * three fields below. The helper-invocation term was dropped in that
       * re-derivation and nothing else carried it.
       *
       * Whether Apple's own driver disables merging here or keeps quads
       * coherent some other way is NOT established; only that this stack needs
       * the bit. Kept narrower than the linker's bit 26, which also fires on
       * memory writes and wide subgroup intrinsics -- neither needs quad
       * coherence, and Apple's metadata leaves bit 26 clear for both. */
      .shader_tri_merge_disable =
         fs->b.info.needs_quad_derivatives ||
         (render->tilebuffer.nr_samples > 1 &&
          fs->info.fs.epilog_key.write_z) ||
         dev->g17p_diagnostics.force_tri_merge_disable,

      /* Native forces bit26 for type2 and conditionally for type3 from the
       * draw-buffer state. Keep the already-qualified conservative type3
       * policy until that draw-buffer field is decoded. */
      .coverage_tri_merge_disable =
         pass_type == AGX_PASS_TYPE_PUNCH_THROUGH ||
         pass_type == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH,

      /* encodeISPPrimitiveType selects the runtime override for points,
       * lines, and non-filled triangles, then applies it to both controls. */
      .primitive_tri_merge_disable =
         object_type != AGX_OBJECT_TYPE_TRIANGLE ||
         polygon_mode != VK_POLYGON_MODE_FILL,
   };
   if (!agx_ppp_hal200_fragment_controls_pack(
          dev->dev.chip, &control_fields, fragment_control,
          secondary_fields.words[0], &fragment_control,
          &secondary_fields.words[0]))
      return hk_g17p_bind_tail_refused(__LINE__);

   uint32_t occlusion_query;
   if (!agx_ppp_hal200_occlusion_pack(
          dev->dev.chip, gfx->occlusion.mode, gfx->occlusion.index,
          fragment_control, &fragment_control, &occlusion_query))
      return hk_g17p_bind_tail_refused(__LINE__);
   output_fields.words[5] = fragment_control;
   primary_fields.words[0] = fragment_control;
   primary_fields.words[5] = occlusion_query;
   if (!agx_ppp_hal200_fragment_core_pack(dev->dev.chip, &fs_fields,
                                          &fs_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->dev.chip, AGX_PPP_HAL200_GRAPHICS_OUTPUT, &output_fields,
          &output_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->dev.chip, AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE,
          &primary_fields, &primary_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->dev.chip, AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE,
          &secondary_fields, &secondary_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->dev.chip, AGX_PPP_HAL200_GRAPHICS_CULL, &cull_fields,
          &cull_record) ||
       !agx_ppp_w_clamp_cull2_core_pack(dev->dev.chip, &w_cull2_fields,
                                         &w_cull2_record))
      return hk_g17p_bind_tail_refused(__LINE__);

   for (unsigned i = 0; i < viewport_count; ++i) {
      static_assert(sizeof(prepared->scissors[i]) ==
                    sizeof(raster_states[i].scissor));
      static_assert(sizeof(prepared->depth_bias[i]) ==
                    sizeof(raster_states[i].depth_bias));
      memcpy(prepared->scissors[i], raster_states[i].scissor,
             sizeof(prepared->scissors[i]));
      memcpy(prepared->depth_bias[i], raster_states[i].depth_bias,
             sizeof(prepared->depth_bias[i]));
   }

   /* Preserve the physical +0x00..+0x73 layout from Metal.  The complete
    * 28-byte FS record occupies +0x00..+0x1b; in particular, +0x14 is the
    * low stage-program word and must not be overlaid by the old anonymous
    * metadata guess.  The output window at +0x1c intentionally overlaps the
    * primary header/control at +0x30/+0x34.
    */
   static_assert(AGX_PPP_HAL200_FS_LENGTH == 0x1c,
                 "HAL200 FS must end where the output record begins");
   uint8_t bind_group[0x80] = {0};
   memcpy(&bind_group[0x00], fs_record.bytes, fs_record.length);
   memcpy(&bind_group[0x1c], output_record.bytes, output_record.length);
   memcpy(&bind_group[0x30], primary_record.bytes, primary_record.length);
   memcpy(&bind_group[0x4c], secondary_record.bytes, secondary_record.length);
   memcpy(&bind_group[0x60], w_cull2_record.bytes, w_cull2_record.length);
   memcpy(&bind_group[0x6c], cull_record.bytes, cull_record.length);

   for (unsigned i = 0; i < viewport_count; ++i) {
      uint8_t region_clip[AGX_REGION_CLIP_LENGTH];
      agx_pack(region_clip, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
      memcpy(&viewport_fields.region_clip_raw[i], region_clip,
             sizeof(region_clip));
      const VkViewport *current_vp = &dyn->vp.viewports[i];
      viewport_fields.viewports[i] =
         (struct agx_ppp_viewport_transform_raw){
            .translate_x = fui(current_vp->x + 0.5f * current_vp->width),
            .scale_x = fui(current_vp->width * 0.5f),
            .translate_y = fui(current_vp->y + 0.5f * current_vp->height),
            .scale_y = fui(current_vp->height * 0.5f),
            .translate_z = fui(current_vp->minDepth),
            .scale_z = fui(current_vp->maxDepth - current_vp->minDepth),
         };
   }

   struct agx_ppp_viewport_payload_core viewport_record;
   if (!agx_ppp_viewport_payload_core_pack(dev->dev.chip, &viewport_fields,
                                            &viewport_record))
      return hk_g17p_bind_tail_refused(__LINE__);

   /* A VDM fetch happens after recording. Never point earlier draws at a
    * device-global record that another draw/CB can overwrite. This upload
    * and its architectural low alias live until command-buffer reset. */
   const size_t state_size = agx_g17p_bind_state_size(viewport_record.length);
   struct agx_ptr state = hk_pool_vdm_alloc(cmd, state_size, 64);
   struct hk_g17p_bind_state_upload uploaded;
   if (!state_size || !state.cpu ||
       !hk_g17p_upload_bind_state(state.cpu, state_size, state.gpu,
                                  bind_group, viewport_record.bytes,
                                  viewport_record.length,
                                  gfx->linked_varyings.counts_32.opaque[0],
                                  gfx->linked_varyings.counts_16.opaque[0],
                                  &uploaded))
      return hk_g17p_bind_tail_refused(__LINE__);

   if (dev->g17p_diagnostics.launch_trace) {
      hk_g17p_trace_line("hal200-bind-group", state.gpu, state.cpu,
                         sizeof(bind_group));
      hk_g17p_trace_line("hal200-fragment-bind", fragment_program,
                         fs_record.bytes, fs_record.length);
      hk_g17p_trace_line("hal200-viewport", state.gpu + HK_G17P_BIND_GROUP_BYTES,
                         (uint8_t *)state.cpu + HK_G17P_BIND_GROUP_BYTES,
                         viewport_record.length);
   }

   prepared->tail = uploaded.tail;
   prepared->dynamic_generation = dyn->generation;
   prepared->descriptor_generation = gfx->descriptors.generation;
   prepared->vertex_buffer_generation = gfx->vertex_buffer_generation;
   prepared->root = gfx->root;
   prepared->render_color0 = render->color_att_count == 1
      ? render->color_att[0].iview : NULL;
   prepared->render_depth = render->depth_att.iview;
   prepared->render_stencil = render->stencil_att.iview;
   prepared->render_area = render->area;
   prepared->render_depth_format = render->depth_att.vk_format;
   prepared->render_width = render->cr.width;
   prepared->render_height = render->cr.height;
   prepared->render_layer_count = render->layer_count;
   prepared->render_view_mask = render->view_mask;
   prepared->render_color_count = render->color_att_count;
   prepared->render_samples = render->tilebuffer.nr_samples;
   memcpy(prepared->shaders, gfx->shaders, sizeof(prepared->shaders));
   prepared->linked_vs = gfx->linked[MESA_SHADER_VERTEX];
   prepared->linked_fs = linked_fs;
   prepared->varyings = gfx->varyings;
   prepared->varying_counts_32 = gfx->linked_varyings.counts_32.opaque[0];
   prepared->varying_counts_16 = gfx->linked_varyings.counts_16.opaque[0];
   prepared->occlusion_index = gfx->occlusion.index;
   prepared->topology = gfx->topology;
   prepared->object_type = gfx->object_type;
   prepared->provoking = gfx->provoking;
   prepared->occlusion_mode = gfx->occlusion.mode;
   prepared->viewport_count = viewport_count;
   prepared->generate_primitive_id = gfx->generate_primitive_id;
   prepared->force_never_in_shader =
      gfx->descriptors.root.draw.force_never_in_shader;
   prepared->in_meta = cmd->in_meta;
   prepared->valid = true;

   gfx->g17p_bind_tail_memo.snapshot = *prepared;
   gfx->g17p_bind_tail_memo.fs_fields = fs_fields;
   gfx->g17p_bind_tail_memo.viewport_record = viewport_record;
   memcpy(gfx->g17p_bind_tail_memo.bind_group, bind_group,
          sizeof(bind_group));
   gfx->g17p_bind_tail_memo.fs = fs;
   gfx->g17p_bind_tail_memo.hw_vs = vs;
   gfx->g17p_bind_tail_memo.valid = true;

   if (dev->g17p_diagnostics.launch_trace) {
      const struct agx_vdm_hal200_bind_pair *pairs = uploaded.pairs;
      fprintf(stderr,
              "G17P HAL200 bind tail: bind0=%08x fs=%08x output=%08x "
              "primary=%08x secondary=%08x viewport=%08x wcull2=%08x "
              "cull=%08x\n",
              pairs[0].context_offset, pairs[1].context_offset,
              pairs[2].context_offset, pairs[3].context_offset,
              pairs[4].context_offset, pairs[5].context_offset,
              pairs[6].context_offset, pairs[7].context_offset);
   }

   return true;
}

/* Every builder input except the root / descriptor / vertex-buffer
 * identity, which reach the tail only through the fragment stage program. */
static bool
hk_g17p_bind_tail_state_matches(
   const struct hk_cmd_buffer *cmd,
   const struct hk_g17p_prepared_bind_tail *prepared)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   const struct hk_rendering_state *render = &gfx->render;

   return prepared->dynamic_generation == dyn->generation &&
          prepared->render_color0 ==
             (render->color_att_count == 1
                 ? render->color_att[0].iview : NULL) &&
          prepared->render_depth == render->depth_att.iview &&
          prepared->render_stencil == render->stencil_att.iview &&
          memcmp(&prepared->render_area, &render->area,
                 sizeof(prepared->render_area)) == 0 &&
          prepared->render_depth_format == render->depth_att.vk_format &&
          prepared->render_width == render->cr.width &&
          prepared->render_height == render->cr.height &&
          prepared->render_layer_count == render->layer_count &&
          prepared->render_view_mask == render->view_mask &&
          prepared->render_color_count == render->color_att_count &&
          prepared->render_samples == render->tilebuffer.nr_samples &&
          memcmp(prepared->shaders, gfx->shaders,
                 sizeof(prepared->shaders)) == 0 &&
          prepared->linked_vs == gfx->linked[MESA_SHADER_VERTEX] &&
          prepared->linked_fs == gfx->linked[MESA_SHADER_FRAGMENT] &&
          prepared->varyings == gfx->varyings &&
          prepared->varying_counts_32 ==
             gfx->linked_varyings.counts_32.opaque[0] &&
          prepared->varying_counts_16 ==
             gfx->linked_varyings.counts_16.opaque[0] &&
          prepared->occlusion_index == gfx->occlusion.index &&
          prepared->topology == gfx->topology &&
          prepared->object_type == gfx->object_type &&
          prepared->provoking == gfx->provoking &&
          prepared->occlusion_mode == gfx->occlusion.mode &&
          prepared->viewport_count == MAX2(dyn->vp.viewport_count, 1) &&
          prepared->generate_primitive_id == gfx->generate_primitive_id &&
          prepared->force_never_in_shader ==
             gfx->descriptors.root.draw.force_never_in_shader &&
          prepared->in_meta == cmd->in_meta;
}

static bool
hk_g17p_prepared_bind_tail_matches(
   const struct hk_cmd_buffer *cmd,
   const struct hk_g17p_prepared_bind_tail *prepared)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;

   return prepared->valid &&
          prepared->descriptor_generation == gfx->descriptors.generation &&
          prepared->vertex_buffer_generation ==
             gfx->vertex_buffer_generation &&
          prepared->root == gfx->root &&
          hk_g17p_bind_tail_state_matches(cmd, prepared);
}

static void __attribute__((cold, noinline))
hk_g17p_trace_prepared_bind_tail_miss(
   const struct hk_cmd_buffer *cmd,
   const struct hk_g17p_prepared_bind_tail *prepared)
{
   static uint32_t reported;
   const uint32_t report = p_atomic_inc_return(&reported);
   if (report > 8)
      return;

   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   const struct hk_rendering_state *render = &gfx->render;
   uint32_t mismatch = 0;

   mismatch |= !prepared->valid ? BITFIELD_BIT(0) : 0;
   mismatch |= prepared->dynamic_generation != dyn->generation
                  ? BITFIELD_BIT(1) : 0;
   mismatch |= prepared->descriptor_generation != gfx->descriptors.generation
                  ? BITFIELD_BIT(2) : 0;
   mismatch |= prepared->root != gfx->root ? BITFIELD_BIT(3) : 0;
   mismatch |= prepared->render_color0 !=
                  (render->color_att_count == 1
                      ? render->color_att[0].iview : NULL) ||
               prepared->render_depth != render->depth_att.iview ||
               prepared->render_stencil != render->stencil_att.iview ||
               memcmp(&prepared->render_area, &render->area,
                      sizeof(prepared->render_area)) != 0 ||
               prepared->render_depth_format != render->depth_att.vk_format ||
               prepared->render_width != render->cr.width ||
               prepared->render_height != render->cr.height ||
               prepared->render_layer_count != render->layer_count ||
               prepared->render_view_mask != render->view_mask ||
               prepared->render_color_count != render->color_att_count ||
               prepared->render_samples != render->tilebuffer.nr_samples
                  ? BITFIELD_BIT(4) : 0;
   mismatch |= memcmp(prepared->shaders, gfx->shaders,
                      sizeof(prepared->shaders)) != 0
                  ? BITFIELD_BIT(5) : 0;
   mismatch |= prepared->linked_vs != gfx->linked[MESA_SHADER_VERTEX] ||
               prepared->linked_fs != gfx->linked[MESA_SHADER_FRAGMENT]
                  ? BITFIELD_BIT(6) : 0;
   mismatch |= prepared->varyings != gfx->varyings ? BITFIELD_BIT(7) : 0;
   mismatch |= prepared->varying_counts_32 !=
                  gfx->linked_varyings.counts_32.opaque[0] ||
               prepared->varying_counts_16 !=
                  gfx->linked_varyings.counts_16.opaque[0]
                  ? BITFIELD_BIT(8) : 0;
   mismatch |= prepared->occlusion_index != gfx->occlusion.index ||
               prepared->occlusion_mode != gfx->occlusion.mode
                  ? BITFIELD_BIT(9) : 0;
   mismatch |= prepared->topology != gfx->topology ||
               prepared->object_type != gfx->object_type ||
               prepared->provoking != gfx->provoking
                  ? BITFIELD_BIT(10) : 0;
   mismatch |= prepared->generate_primitive_id !=
                  gfx->generate_primitive_id ||
               prepared->force_never_in_shader !=
                  gfx->descriptors.root.draw.force_never_in_shader ||
               prepared->in_meta != cmd->in_meta
                  ? BITFIELD_BIT(11) : 0;

   fprintf(stderr,
           "HK_SHADER_PERF bind-tail-miss=%08x valid=%u dyn=%" PRIu64
           "/%" PRIu64 " desc=%" PRIu64 "/%" PRIu64
           " root=%" PRIx64 "/%" PRIx64 " varyings=%" PRIx64
           "/%" PRIx64 "\n",
           mismatch, prepared->valid, prepared->dynamic_generation,
           dyn->generation, prepared->descriptor_generation,
           gfx->descriptors.generation, prepared->root, gfx->root,
           prepared->varyings, gfx->varyings);
}

static bool
hk_g17p_publish_native_bind_tail(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out,
   const struct hk_g17p_prepared_bind_tail *prepared)
{
   if (!prepared->valid || !out || !*out)
      return false;

   const unsigned scissor_bytes =
      prepared->viewport_count * AGX_SCISSOR_LENGTH;
   const unsigned depth_bias_bytes =
      prepared->viewport_count * AGX_DEPTH_BIAS_LENGTH;
   if (!util_dynarray_ensure_cap(&cs->scissor,
                                 cs->scissor.size + scissor_bytes) ||
       !util_dynarray_ensure_cap(&cs->depth_bias,
                                 cs->depth_bias.size + depth_bias_bytes)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return false;
   }

   uint8_t *scissors = util_dynarray_grow_bytes(
      &cs->scissor, prepared->viewport_count, AGX_SCISSOR_LENGTH);
   uint8_t *depth_bias = util_dynarray_grow_bytes(
      &cs->depth_bias, prepared->viewport_count, AGX_DEPTH_BIAS_LENGTH);
   assert(scissors && depth_bias);

   memcpy(scissors, prepared->scissors, scissor_bytes);
   memcpy(depth_bias, prepared->depth_bias, depth_bias_bytes);
   memcpy(*out, prepared->tail.bytes, prepared->tail.length);
   *out += prepared->tail.length;
   return true;
}

/* known_match: the caller evaluated hk_g17p_prepared_bind_tail_matches()
 * for this draw and nothing it reads has changed since. */
static __attribute__((hot)) bool
hk_g17p_emit_native_bind_tail(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                              uint8_t **out, bool known_match)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_g17p_prepared_bind_tail *prepared =
      &gfx->g17p_bind_tail;
   if ((known_match && prepared->valid) ||
       hk_g17p_prepared_bind_tail_matches(cmd, prepared)) {
      return hk_g17p_publish_native_bind_tail(cmd, cs, out, prepared);
   }

   return hk_g17p_build_native_bind_tail(cmd, prepared) &&
          hk_g17p_publish_native_bind_tail(cmd, cs, out, prepared);
}

/* Resolve the draw-independent inputs needed by the first HAL200 state update
 * while recording is still outside vkCmdDraw*. Nothing here advances a
 * control stream. The allocation and tail remain unreachable by firmware
 * until hk_g17p_publish_native_bind_tail() runs in the draw path. */
static void
hk_g17p_prepare_native_bind_tail(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   struct hk_g17p_prepared_bind_tail *prepared = &gfx->g17p_bind_tail;

   if (dev->dev.chip != AGX_CHIP_G17P || !cmd->current_cs.gfx ||
       !gfx->linked[MESA_SHADER_VERTEX] ||
       !gfx->linked[MESA_SHADER_FRAGMENT] ||
       vk_command_buffer_has_error(&cmd->vk))
      return;

   const enum mesa_prim prim = hk_rast_prim(cmd);
   gfx->topology = translate_hw_primitive_topology(prim);
   gfx->object_type = translate_object_type(prim);

   unsigned provoking;
   if (dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT)
      provoking = 2;
   else if (gfx->topology == AGX_PRIMITIVE_TRIANGLE_FAN)
      provoking = 1;
   else
      provoking = 0;

   if (provoking != gfx->provoking) {
      gfx->provoking = provoking;
      gfx->dirty |= HK_DIRTY_PROVOKING;
      gfx->descriptors.root.draw.provoking = provoking;
      gfx->descriptors.root_dirty = true;
   }

   if (dyn->ms.rasterization_samples && !gfx->render.tilebuffer.nr_samples) {
      agx_tilebuffer_set_samples(&gfx->render.tilebuffer,
                                 dyn->ms.rasterization_samples);
      gfx->render.g17p_rt0_layout_supported =
         hk_agx3_g17p_rt0_tile_layout_supported(&gfx->render.tilebuffer);
      if (!gfx->render.g17p_rt0_layout_supported)
         return;
      cmd->current_cs.gfx->tib = gfx->render.tilebuffer;
   }

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *fs = hk_bound_fs(cmd);
   const bool has_z =
      gfx->render.depth_att.vk_format != VK_FORMAT_UNDEFINED;
   const bool force_never_in_shader =
      has_z && dyn->ds.depth.test_enable &&
      dyn->ds.depth.compare_op == VK_COMPARE_OP_NEVER &&
      fs->info.fs.writes_memory;
   if (gfx->descriptors.root.draw.force_never_in_shader !=
       force_never_in_shader) {
      gfx->descriptors.root.draw.force_never_in_shader =
         force_never_in_shader;
      gfx->descriptors.root_dirty = true;
   }

   /* Publication only copies this command-buffer-owned immutable record and
    * its inline viewport payloads. Reuse it when every input read by the
    * builder, including the render identity, is still exact. */
   if (hk_g17p_prepared_bind_tail_matches(cmd, prepared))
      return;

   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   const unsigned bindings = linked_fs->b.cf.nr_bindings;
   if (bindings) {
      const size_t linkage_size =
         AGX_CF_BINDING_HEADER_LENGTH + bindings * AGX_CF_BINDING_LENGTH;
      struct agx_ptr t = hk_pool_usc_alloc(cmd, linkage_size, 64);
      if (!t.cpu)
         return;

      agx_link_varyings_vs_fs(
         t.cpu, &gfx->linked_varyings, hw_vs->info.uvs.user_size,
         &linked_fs->b.cf, gfx->provoking, 0, &gfx->generate_primitive_id);
      gfx->varyings = t.gpu;
   } else {
      gfx->varyings = 0;
      gfx->generate_primitive_id = false;
   }
   gfx->dirty |= HK_DIRTY_VARYINGS;

   if (!hk_g17p_build_native_bind_tail(cmd, prepared))
      return;
}

/* Encode the complete immutable first-draw VDM state record while shader and
 * render binding work is already in progress.  The prepared transaction is
 * deliberately limited to the default direct, non-indexed path; indexed,
 * indirect and diagnostic context rotation retain the authoritative encoder. */
static bool
hk_g17p_prepare_vdm_state_update(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_cs *cs = cmd->current_cs.gfx;
   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   gfx->prepared_vdm_state_update_valid = false;

   if (dev->dev.chip != AGX_CHIP_G17P || !cs || !hw_vs ||
       !gfx->prepared_vertex_usc ||
       !gfx->prepared_fast_link_vs.valid ||
       gfx->prepared_fast_link_vs.indexed ||
       gfx->prepared_fast_link_vs.restart ||
       hk_g17p_cycle_vdm_context_enabled(dev) ||
       !cs->agx3_active_context_selector_valid)
      return false;

   const struct agx_vdm_state_update_program_info *program =
      &hw_vs->agx3_vdm_state_update;
   const uint64_t pipeline_program = gfx->prepared_vertex_usc;
   const bool pipeline_va_valid =
      pipeline_program < (UINT64_C(1) << 48) &&
      (pipeline_program & UINT64_C(0x3f)) == 0;
   const bool complete_program =
      pipeline_va_valid &&
      hk_agx3_vdm_live_source_layout(
         program, cs->agx3_active_context_selector_valid, true) ==
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;
   if (!complete_program)
      return false;

   struct agx_vdm_state_update_draw_info info = {
      .source_layout =
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
      .draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED,
      .rasterization_enabled =
         !hk_rast_discard(cmd) ||
         hk_g17p_xfb_vertex_side_effects(cmd, hw_vs),
      .raw_provoking_selector = gfx->provoking,
      .template_info = program->template_info,
   };
   info.template_info.pipeline = pipeline_program;
   info.template_info.raw_vs_output_flag_6 = gfx->generate_primitive_id;
   info.template_info.raw_vs_output_count_1_extra =
      gfx->generate_primitive_id;
   info.template_info.raw_active_context_selector =
      cs->agx3_raw_active_context_selector;
   if (hk_g17p_nonpoint_no_psiz_diagnostic(
          dev->g17p_diagnostics.nonpoint_no_psiz, gfx, hw_vs)) {
      assert(info.template_info.raw_vs_output_count_0 > 0);
      --info.template_info.raw_vs_output_count_0;
   }

   gfx->prepared_vdm_state_update_valid =
      hk_agx3_encode_vdm_state_update_draw(
         dev->dev.chip, &info, &gfx->prepared_vdm_state_update);
   return gfx->prepared_vdm_state_update_valid;
}

/*
 * Return the given sample positions, packed into a 32-bit word with fixed
 * point nibbles for each x/y component of the (at most 4) samples. This is
 * suitable for programming the PPP_MULTISAMPLECTL control register.
 */
static uint32_t
hk_pack_ppp_multisamplectrl(const struct vk_sample_locations_state *sl)
{
   uint32_t ctrl = 0;

   for (int32_t i = sl->per_pixel - 1; i >= 0; i--) {
      VkSampleLocationEXT loc = sl->locations[i];

      uint32_t x = CLAMP(loc.x, 0.0f, 0.9375f) * 16.0;
      uint32_t y = CLAMP(loc.y, 0.0f, 0.9375f) * 16.0;

      assert(x <= 15);
      assert(y <= 15);

      /* Push bytes in reverse order so we can use constant shifts. */
      ctrl = (ctrl << 8) | (y << 4) | x;
   }

   return ctrl;
}

/*
 * Return the standard sample positions, prepacked as above for efficiency.
 */
uint32_t
hk_default_sample_positions(unsigned nr_samples)
{
   switch (nr_samples) {
   case 0:
   case 1:
      return 0x88;
   case 2:
      return 0x44cc;
   case 4:
      return 0xeaa26e26;
   default:
      UNREACHABLE("Invalid sample count");
   }
}

static void
hk_flush_ppp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *fs = hk_bound_fs(cmd);

   bool hw_vs_dirty = IS_SHADER_DIRTY(VERTEX) || IS_SHADER_DIRTY(TESS_EVAL) ||
                      IS_SHADER_DIRTY(GEOMETRY);
   bool fs_dirty = IS_SHADER_DIRTY(FRAGMENT);

   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   bool linked_fs_dirty = IS_LINKED_DIRTY(FRAGMENT);

   bool varyings_dirty = gfx->dirty & HK_DIRTY_VARYINGS;

   bool face_dirty =
      IS_DIRTY(DS_DEPTH_TEST_ENABLE) || IS_DIRTY(DS_DEPTH_WRITE_ENABLE) ||
      IS_DIRTY(DS_DEPTH_COMPARE_OP) || IS_DIRTY(DS_STENCIL_REFERENCE) ||
      IS_DIRTY(RS_LINE_WIDTH) || IS_DIRTY(RS_POLYGON_MODE) || fs_dirty;

   bool stencil_face_dirty =
      IS_DIRTY(DS_STENCIL_OP) || IS_DIRTY(DS_STENCIL_COMPARE_MASK) ||
      IS_DIRTY(DS_STENCIL_WRITE_MASK) || IS_DIRTY(DS_STENCIL_TEST_ENABLE);

   struct AGX_PPP_HEADER dirty = {
      .fragment_control =
         IS_DIRTY(DS_STENCIL_TEST_ENABLE) || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
         IS_DIRTY(RS_DEPTH_BIAS_ENABLE) || gfx->dirty & HK_DIRTY_OCCLUSION,

      .fragment_control_2 =
         IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) || linked_fs_dirty,

      .fragment_front_face = face_dirty,
      .fragment_front_face_2 = fs_dirty || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY),
      .fragment_front_stencil = stencil_face_dirty,
      .fragment_back_face = face_dirty,
      .fragment_back_face_2 = fs_dirty || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY),
      .fragment_back_stencil = stencil_face_dirty,
      .output_select = hw_vs_dirty || linked_fs_dirty || varyings_dirty,
      .varying_counts_32 = varyings_dirty,
      .varying_counts_16 = varyings_dirty,
      .cull = IS_DIRTY(RS_CULL_MODE) ||
              IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) ||
              IS_DIRTY(RS_FRONT_FACE) || IS_DIRTY(RS_DEPTH_CLIP_ENABLE) ||
              IS_DIRTY(RS_DEPTH_CLAMP_ENABLE) || IS_DIRTY(RS_LINE_MODE) ||
              IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
              (gfx->dirty & HK_DIRTY_PROVOKING) || IS_SHADER_DIRTY(TESS_CTRL) ||
              IS_SHADER_DIRTY(TESS_EVAL) || IS_DIRTY(TS_DOMAIN_ORIGIN),
      .cull_2 = varyings_dirty,

      .fragment_shader = fs_dirty || linked_fs_dirty || varyings_dirty ||
                         gfx->descriptors.root_dirty,

      .occlusion_query = gfx->dirty & HK_DIRTY_OCCLUSION,
      .output_size = hw_vs_dirty,
      .viewport_count = 1, /* irrelevant */
   };

   if (dev->dev.chip != AGX_CHIP_G17P)
      dirty.fragment_shader &= !linked_fs->b.no_op;

   /* The first G17P hardware fault is introduced by this aggregate dirty PPP
    * update. Keep a diagnostic-only selector that emits one independently
    * consumable legacy group at a time so the bad HAL200 block can be
    * identified without changing any of its dynamic values.
    *
    * 1: fragment control and non-stencil faces
    * 2: cull words
    * 3: fragment shader words
    * 4: output selection, varying counts, and output size
    * 5: occlusion query and output size
    */
   if (dev->dev.chip == AGX_CHIP_G17P) {
      static int g17p_dirty_bisect = -1;
      if (g17p_dirty_bisect < 0)
         g17p_dirty_bisect =
            debug_get_option_hk_g17p_dirty_bisect();

      if (g17p_dirty_bisect > 0) {
         const struct AGX_PPP_HEADER all = dirty;
         dirty = (struct AGX_PPP_HEADER){.viewport_count = 1};

         switch (g17p_dirty_bisect) {
         case 1:
            dirty.fragment_control = all.fragment_control;
            dirty.fragment_control_2 = all.fragment_control_2;
            dirty.fragment_front_face = all.fragment_front_face;
            dirty.fragment_front_face_2 = all.fragment_front_face_2;
            dirty.fragment_back_face = all.fragment_back_face;
            dirty.fragment_back_face_2 = all.fragment_back_face_2;
            break;
         case 2:
            dirty.cull = all.cull;
            dirty.cull_2 = all.cull_2;
            break;
         case 3:
            dirty.fragment_shader = all.fragment_shader;
            break;
         case 4:
            dirty.output_select = all.output_select;
            dirty.varying_counts_32 = all.varying_counts_32;
            dirty.varying_counts_16 = all.varying_counts_16;
            dirty.output_size = all.output_size;
            break;
         case 5:
            dirty.occlusion_query = all.occlusion_query;
            dirty.output_size = all.output_size;
            break;
         default:
            mesa_loge("HK_G17P_DIRTY_BISECT=%d is outside 1..5",
                      g17p_dirty_bisect);
            dirty = all;
            break;
         }

         fprintf(stderr, "G17P dirty PPP bisect: group=%d\n",
                 g17p_dirty_bisect);
      }
   }

   /* Calculate the update size. If it equals the header, there is nothing to
    * update so early-exit.
    */
   size_t size = agx_ppp_update_size(&dirty);
   if (size == AGX_PPP_HEADER_LENGTH)
      return;

   /* Otherwise, allocate enough space for the update and push it. */
   assert(size > AGX_PPP_HEADER_LENGTH);

   struct agx_ptr T = hk_pool_vdm_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &dirty);

   if (dirty.fragment_control) {
      agx_ppp_push(&ppp, FRAGMENT_CONTROL, cfg) {
         cfg.visibility_mode = gfx->occlusion.mode;
         cfg.stencil_test_enable = hk_stencil_test_enabled(cmd);

         /* TODO: Consider optimizing this? */
         cfg.two_sided_stencil = cfg.stencil_test_enable;

         cfg.depth_bias_enable = dyn->rs.depth_bias.enable &&
                                 gfx->object_type == AGX_OBJECT_TYPE_TRIANGLE;

         /* Always enable scissoring so we may scissor to the viewport (TODO:
          * optimize this out if the viewport is the default and the app does
          * not use the scissor test)
          */
         cfg.scissor_enable = true;

         /* This avoids broken derivatives along primitive edges */
         cfg.disable_tri_merging = gfx->object_type != AGX_OBJECT_TYPE_TRIANGLE;
      }
   }

   if (dirty.fragment_control_2) {
      /* Annoying, rasterizer_discard seems to be ignored (sometimes?) in the
       * main fragment control word and has to be combined into the secondary
       * word for reliable behaviour.
       */
      agx_ppp_push_merged(&ppp, FRAGMENT_CONTROL, cfg,
                          linked_fs->b.fragment_control) {

         cfg.tag_write_disable = hk_rast_discard(cmd);
      }
   }

   struct agx_fragment_face_packed fragment_face = {};
   struct agx_fragment_face_2_packed fragment_face_2 = {};

   if (dirty.fragment_front_face) {
      bool has_z = render->depth_att.vk_format != VK_FORMAT_UNDEFINED;
      bool z_test = has_z && dyn->ds.depth.test_enable;

      agx_pack(&fragment_face, FRAGMENT_FACE, cfg) {
         cfg.line_width = agx_pack_line_width(dyn->rs.line.width);
         cfg.polygon_mode = translate_polygon_mode(dyn->rs.polygon_mode);
         cfg.disable_depth_write = !(z_test && dyn->ds.depth.write_enable);

         if (z_test && !gfx->descriptors.root.draw.force_never_in_shader)
            cfg.depth_function = translate_compare_op(dyn->ds.depth.compare_op);
         else
            cfg.depth_function = AGX_ZS_FUNC_ALWAYS;
      };

      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, fragment_face) {
         cfg.stencil_reference = dyn->ds.stencil.front.reference;
      }
   }

   if (dirty.fragment_front_face_2) {
      agx_pack(&fragment_face_2, FRAGMENT_FACE_2, cfg) {
         cfg.object_type = gfx->object_type;
      }

      agx_merge(fragment_face_2, fs->frag_face, FRAGMENT_FACE_2);
      agx_ppp_push_packed(&ppp, &fragment_face_2, FRAGMENT_FACE_2);
   }

   if (dirty.fragment_front_stencil) {
      hk_ppp_push_stencil_face(&ppp, dyn->ds.stencil.front,
                               hk_stencil_test_enabled(cmd));
   }

   if (dirty.fragment_back_face) {
      assert(dirty.fragment_front_face);

      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, fragment_face) {
         cfg.stencil_reference = dyn->ds.stencil.back.reference;
      }
   }

   if (dirty.fragment_back_face_2) {
      assert(dirty.fragment_front_face_2);

      agx_ppp_push_packed(&ppp, &fragment_face_2, FRAGMENT_FACE_2);
   }

   if (dirty.fragment_back_stencil) {
      hk_ppp_push_stencil_face(&ppp, dyn->ds.stencil.back,
                               hk_stencil_test_enabled(cmd));
   }

   if (dirty.output_select) {
      struct agx_output_select_packed osel = hw_vs->info.uvs.osel;

      agx_ppp_push_merged_blobs(&ppp, AGX_OUTPUT_SELECT_LENGTH, &osel,
                                &linked_fs->b.osel);
   }

   assert(dirty.varying_counts_32 == dirty.varying_counts_16);

   if (dirty.varying_counts_32) {
      agx_ppp_push_packed(&ppp, &gfx->linked_varyings.counts_32,
                          VARYING_COUNTS);

      agx_ppp_push_packed(&ppp, &gfx->linked_varyings.counts_16,
                          VARYING_COUNTS);
   }

   if (dirty.cull) {
      agx_ppp_push(&ppp, CULL, cfg) {
         cfg.cull_front = dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT;
         cfg.cull_back = dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT;
         cfg.front_face_ccw = dyn->rs.front_face != VK_FRONT_FACE_CLOCKWISE;

         if (gfx->shaders[MESA_SHADER_TESS_CTRL] &&
             !gfx->shaders[MESA_SHADER_GEOMETRY]) {
            cfg.front_face_ccw ^= gfx->tess.info.ccw;
            cfg.front_face_ccw ^= dyn->ts.domain_origin ==
                                  VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
         }

         cfg.flat_shading_vertex = translate_ppp_vertex(gfx->provoking);
         cfg.rasterizer_discard = hk_rast_discard(cmd);

         /* We do not support unrestricted depth, so clamping is inverted from
          * clipping. This implementation seems to pass CTS without unrestricted
          * depth support.
          *
          * TODO: Make sure this is right with gl_FragDepth.
          */
         cfg.depth_clip = vk_rasterization_state_depth_clip_enable(&dyn->rs);
         cfg.depth_clamp = !cfg.depth_clip;

         cfg.primitive_msaa =
            gfx->object_type == AGX_OBJECT_TYPE_LINE &&
            dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
      }
   }

   if (dirty.cull_2) {
      agx_ppp_push(&ppp, CULL_2, cfg) {
         cfg.needs_primitive_id = gfx->generate_primitive_id;
         cfg.clamp_w = true;
      }
   }

   if (dirty.fragment_shader) {
      /* TODO: Do less often? */
      hk_reserve_scratch(cmd, cs, fs);

      const uint64_t uploaded = hk_upload_usc_words(cmd, fs, linked_fs);
      uint32_t fragment_pipeline;
      switch (agx_chip_command_stream_generation(dev->dev.chip)) {
      case AGX_CS_GENERATION_AGX2:
         if (uploaded > UINT32_MAX) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         fragment_pipeline = uploaded;
         break;
      case AGX_CS_GENERATION_HAL200:
         if (uploaded == 0 || uploaded < dev->dev.shader_base ||
             uploaded - dev->dev.shader_base > UINT32_MAX) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         fragment_pipeline = agx_usc_addr(&dev->dev, uploaded);
         break;
      case AGX_CS_GENERATION_G15:
      case AGX_CS_GENERATION_HAL300:
      case AGX_CS_GENERATION_UNKNOWN:
      default:
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      agx_ppp_push_packed(&ppp, &linked_fs->fs_counts, FRAGMENT_SHADER_WORD_0);

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_1, cfg) {
         cfg.pipeline = fragment_pipeline;
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_2, cfg) {
         cfg.cf_bindings = gfx->varyings;
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_3, cfg)
         ;
   }

   if (dirty.occlusion_query) {
      agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY, cfg) {
         cfg.index = gfx->occlusion.index;
      }
   }

   if (dirty.output_size) {
      agx_ppp_push(&ppp, OUTPUT_SIZE, cfg) {
         cfg.count = hw_vs->info.uvs.size;
      }
   }

   const bool ppp_ok = agx_ppp_legacy_payload_supported(dev->dev.chip) &&
                       agx_ppp_fini(dev->dev.chip, out, &ppp);
   if (!ppp_ok) {
      mesa_loge("no grounded legacy PPP payload/finalizer for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   } else if (hk_g17p_launch_trace_enabled()) {
      hk_g17p_trace_line("ppp-dirty", ppp.gpu_base, ppp.cpu_base,
                         ppp.total_size);
   }
}

/*
 * Based somewhat on the calculation in the PowerVR driver, and mostly trial &
 * error to pass CTS. This is a mess.
 */
static float
hk_depth_bias_factor(VkFormat format, bool exact, bool force_unorm)
{
   if (hk_depth_stencil_plane_format(format, 0) == VK_FORMAT_D16_UNORM) {
      return exact ? (1 << 16) : (1 << 15);
   } else if (force_unorm) {
      return exact ? (1ull << 24) : (1ull << 23);
   } else {
      return 1.0;
   }
}

static bool
uses_blend_constant(const struct vk_color_blend_state *cb)
{
   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 1 ==
                 VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR);

   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 2 ==
                 VK_BLEND_FACTOR_CONSTANT_ALPHA);

   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 3 ==
                 VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA);

   for (unsigned i = 0; i < cb->attachment_count; ++i) {
      unsigned factors[] = {
         cb->attachments[i].src_color_blend_factor,
         cb->attachments[i].src_alpha_blend_factor,
         cb->attachments[i].dst_color_blend_factor,
         cb->attachments[i].dst_alpha_blend_factor,
      };

      for (unsigned j = 0; j < ARRAY_SIZE(factors); ++j) {
         if (factors[j] >= VK_BLEND_FACTOR_CONSTANT_COLOR &&
             factors[j] <= VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA)
            return true;
      }
   }

   return false;
}

/* HAL200 texture and sampler heaps consume fragment uniform halves0..7. The
 * ordinary ABI's blend constants at halves4..11 therefore cannot coexist
 * with an indexed/combined texture shader. Shader parts can address any even
 * uniform half directly, so place the epilog-only constants after every main
 * shader range. The Ioto program publishes that exact relocated range.
 */
static bool
hk_g17p_relocated_blend_constant_base(const struct hk_shader *fs,
                                      uint16_t *base_out)
{
   if (!fs || !base_out)
      return false;

   unsigned base = MAX2(AGX_G17P_TXF_ROOT_HALFS, fs->b.info.push_count);
   if (fs->b.info.rodata.size_16) {
      const unsigned rodata_end = fs->b.info.rodata.base_uniform +
                                  fs->b.info.rodata.size_16;
      base = MAX2(base, rodata_end);
   }
   base = ALIGN_POT(base, 2);
   if (base > 512 - 8)
      return false;

   *base_out = base;
   return true;
}

void
agx_fill_velem_keys(const struct vk_vertex_input_state *vi,
                    uint64_t attribs_read, struct agx_velem_key *keys)
{
   u_foreach_bit(a, vi->attributes_valid) {
      struct vk_vertex_attribute_state attr = vi->attributes[a];

      assert(vi->bindings_valid & BITFIELD_BIT(attr.binding));
      struct vk_vertex_binding_state binding = vi->bindings[attr.binding];

      /* nir_assign_io_var_locations compacts vertex inputs, eliminating
       * unused inputs. We need to do the same here to match the locations.
       */
      unsigned slot = util_bitcount64(attribs_read & BITFIELD_MASK(a));

      keys[slot] = (struct agx_velem_key){
         .format = hk_format_to_pipe_format(attr.format),
         .stride = binding.stride,
         .divisor = binding.divisor,
         .instanced = binding.input_rate == VK_VERTEX_INPUT_RATE_INSTANCE,
      };
   }
}

static void
hk_fill_fast_link_key_vs(struct hk_cmd_buffer *cmd, struct hk_shader *sw_vs,
                         struct hk_shader *hw_vs, bool indexed,
                         enum agx_index_size index_size, bool restart,
                         struct hk_fast_link_key_vs *key)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   *key = (struct hk_fast_link_key_vs){
      .prolog.hw = (sw_vs == hw_vs),
      .prolog.uniform_base = sw_vs->info.agx3_indexed_sampler_valid
                                ? AGX_ABI_G17P_TEXTURE_POINTER_HALFS
                                : 0,

      /* FIXME: handle pipeline robustness "properly" */
      .prolog.robustness.level =
         (dev->vk.enabled_features.robustBufferAccess2 ||
          dev->vk.enabled_features.pipelineRobustness)
            ? AGX_ROBUSTNESS_D3D
         : dev->vk.enabled_features.robustBufferAccess
            ? AGX_ROBUSTNESS_GL
            : AGX_ROBUSTNESS_DISABLED,
      .prolog.robustness.soft_fault = agx_has_soft_fault(&dev->dev),
      .prolog.static_vi = !sw_vs->info.vs.use_prolog,
   };

   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);
   if (mesa_prim_has_adjacency(prim)) {
      if (restart)
         prim = u_decomposed_prim(prim);

      /* G17P materialises adjacency into an index buffer before the hardware
       * vertex program. Applying the remap again in the prolog is wrong. */
      if (dev->dev.chip != AGX_CHIP_G17P)
         key->prolog.adjacency = prim;
   }

   if (key->prolog.adjacency || !key->prolog.hw) {
      key->prolog.sw_index_size_B =
         indexed ? agx_index_size_to_B(index_size) : 0;
   }

   static_assert(sizeof(key->prolog.component_mask) ==
                 sizeof(sw_vs->info.vs.attrib_components_read));
   BITSET_COPY(key->prolog.component_mask,
               sw_vs->info.vs.attrib_components_read);

   if (sw_vs->info.vs.use_prolog) {
      agx_fill_velem_keys(dyn->vi, sw_vs->info.vs.attribs_read,
                          key->prolog.attribs);

      u_foreach_bit(a, dyn->vi->attributes_valid) {
         unsigned slot =
            util_bitcount64(sw_vs->info.vs.attribs_read & BITFIELD_MASK(a));
         key->prolog.attribs[slot].stride =
            dyn->vi_binding_strides[dyn->vi->attributes[a].binding];
      }
   }
}

static bool
hk_fill_fast_link_key_fs(struct hk_cmd_buffer *cmd, struct hk_shader *fs,
                         struct hk_shader *hw_vs,
                         struct hk_fast_link_key_fs *key, bool report_errors)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_rendering_state *render = &gfx->render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   unsigned samples = MAX2(dyn->ms.rasterization_samples, 1);
   unsigned tib_sample_mask = BITFIELD_MASK(dyn->ms.rasterization_samples);
   unsigned api_sample_mask = dyn->ms.sample_mask & tib_sample_mask;
   bool has_sample_mask = api_sample_mask != tib_sample_mask;
   bool nontrivial_force_early =
      fs && (fs->b.info.early_fragment_tests &&
             (fs->b.info.writes_sample_mask || fs->info.fs.writes_memory));
   bool epilog_discards = agx_fs_epilog_may_discard(
      fs ? &fs->info.fs.epilog_key : NULL,
      dyn->ms.alpha_to_coverage_enable, nontrivial_force_early);
   bool defer_api_sample_mask = dev->dev.chip == AGX_CHIP_G17P &&
                                has_sample_mask && !nontrivial_force_early;

   if (dev->dev.chip == AGX_CHIP_G17P && fs->info.fs.uses_sample_shading &&
       dyn->ms.rasterization_samples > 1) {
      if (report_errors)
         mesa_loge("G17P: per-sample fragment execution needs a fixed sample "
                   "count; refusing a %u-sample draw that requests it",
                   dyn->ms.rasterization_samples);
      return false;
   }

   if (dev->dev.chip == AGX_CHIP_G17P &&
       fs->info.fs.epilog_key.sample_shading &&
       fs->info.fs.g17p_samples != samples) {
      if (report_errors)
         mesa_loge("G17P: fragment unrolled for %u samples cannot run a "
                   "%u-sample draw",
                   fs->info.fs.g17p_samples, dyn->ms.rasterization_samples);
      return false;
   }

   if (dev->dev.chip == AGX_CHIP_G17P && fs->b.info.writes_sample_mask &&
       fs->info.fs.g17p_samples != samples) {
      if (report_errors)
         mesa_loge("G17P: fragment coverage updates compiled for %u samples "
                   "cannot run a %u-sample draw",
                   fs->info.fs.g17p_samples, dyn->ms.rasterization_samples);
      return false;
   }

   unsigned samples_shaded = 0;
   if (fs->info.fs.epilog_key.sample_shading &&
       dev->dev.chip != AGX_CHIP_G17P)
      samples_shaded = dyn->ms.rasterization_samples;

   *key = (struct hk_fast_link_key_fs){
      .prolog.statistics = hk_stat_enabled(hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT)),
      .prolog.cull_distance_size = hw_vs->info.cull_distance_array_size,
      .prolog.api_sample_mask =
         has_sample_mask && !nontrivial_force_early && !defer_api_sample_mask
            ? api_sample_mask
            : 0xff,
      .prolog.nr_samples = dev->dev.chip == AGX_CHIP_G17P ? samples : 0,
      .nr_samples_shaded = samples_shaded,
   };

   bool prolog_discards =
      (has_sample_mask && !defer_api_sample_mask) ||
      key->prolog.cull_distance_size;
   if (key->prolog.statistics || prolog_discards) {
      key->prolog.run_zs_tests = !nontrivial_force_early &&
                                 !fs->b.info.writes_sample_mask &&
                                 !epilog_discards && prolog_discards;
      if (key->prolog.cull_distance_size)
         key->prolog.cf_base = fs->b.info.varyings.fs.nr_cf;
   }

   key->epilog = (struct agx_fs_epilog_key){
      .link = fs->info.fs.epilog_key,
      .nr_samples = samples,
      .blend.alpha_to_coverage = dyn->ms.alpha_to_coverage_enable,
      .blend.alpha_to_one = dyn->ms.alpha_to_one_enable,
      .blend.logicop_enable = dyn->cb.logic_op_enable,
      .blend.logicop_func = vk_logic_op_to_pipe(dyn->cb.logic_op),
      .defer_api_sample_mask = defer_api_sample_mask,
      .api_sample_mask = api_sample_mask,
   };
   agx_fs_epilog_apply_device_policy(dev->dev.chip, &key->epilog);

   key->epilog.native_spills = dev->dev.chip == AGX_CHIP_G17P &&
                               agx_tilebuffer_spills(&render->tilebuffer);
   if (key->epilog.native_spills) {
      uint16_t base;
      if (!hk_g17p_relocated_blend_constant_base(fs, &base) ||
          ALIGN_POT(base, 4) > 512 - 12)
         return false;
      key->epilog.emrt_heap_base = ALIGN_POT(base, 4);
      key->epilog.blend_constant_base = key->epilog.emrt_heap_base + 4;
   }

   if (!key->epilog.native_spills && dev->dev.chip == AGX_CHIP_G17P &&
       uses_blend_constant(&dyn->cb) &&
       (fs->info.agx3_combined_sampler_valid ||
        fs->info.agx3_indexed_sampler_valid) &&
       !hk_g17p_relocated_blend_constant_base(
          fs, &key->epilog.blend_constant_base))
      return false;

   for (unsigned rt = 0; rt < ARRAY_SIZE(dyn->cal.color_map); ++rt) {
      int map = dyn->cal.color_map[rt];
      key->epilog.remap[rt] =
         map == MESA_VK_ATTACHMENT_UNUSED ? -1 : map;
   }

   key->epilog.link.already_ran_zs |= nontrivial_force_early;
   key->epilog.max_sample_size_B = render->tilebuffer.max_sample_size_B;
   key->epilog.max_bytes_per_tile = render->tilebuffer.max_bytes_per_tile;
   key->epilog.max_samples_per_tile = render->tilebuffer.max_samples_per_tile;

   for (uint32_t i = 0; i < render->color_att_count; i++) {
      key->epilog.rt_formats[i] =
         hk_format_to_pipe_format(render->color_att[i].vk_format);
      if (agx_native_graphics(&dev->dev)) {
         const enum pipe_format logical = key->epilog.rt_formats[i];
         key->epilog.rt_formats[i] = hk_padded_image_format(logical);
         const enum pipe_format integer =
            hk_scaled_texture_format(key->epilog.rt_formats[i]);
         if (integer != PIPE_FORMAT_NONE) {
            key->epilog.rt_formats[i] = integer;
            key->epilog.scaled_formats[i] = logical;
         } else if (key->epilog.rt_formats[i] != logical) {
            key->epilog.blend_formats[i] = logical;
         }
      }

      const struct vk_color_blend_attachment_state *cb =
         &dyn->cb.attachments[i];
      bool write_enable = dyn->cb.color_write_enables & BITFIELD_BIT(i);
      unsigned write_mask = write_enable ? cb->write_mask : 0;

      if ((dev->dev.chip == AGX_CHIP_G16G ||
           dev->dev.chip == AGX_CHIP_G17P) && !write_enable) {
         key->epilog.preserve_rt |= BITFIELD_BIT(i);
         key->epilog.blend.rt[i] = (struct agx_blend_rt_key){
            .colormask = BITFIELD_MASK(4),
            .mode = agx_pack_blend_standard(
               PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
               PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO),
         };
      } else if (!cb->blend_enable) {
         key->epilog.blend.rt[i] = (struct agx_blend_rt_key){
            .colormask = write_mask,
            .mode = agx_pack_blend_standard(
               PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
               PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO),
         };
      } else if (cb->color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
         key->epilog.blend.rt[i] = (struct agx_blend_rt_key){
            .colormask = write_mask,
            .advanced_blend = 1,
            .mode = agx_pack_blend_advanced(
               vk_advanced_blend_op_to_pipe(cb->color_blend_op),
               vk_blend_overlap_to_pipe(cb->blend_overlap),
               cb->src_premultiplied, cb->dst_premultiplied,
               cb->clamp_results),
         };
         assert(cb->clamp_results == false);
      } else {
         key->epilog.blend.rt[i] = (struct agx_blend_rt_key){
            .colormask = write_mask,
            .mode = agx_pack_blend_standard(
               vk_blend_op_to_pipe(cb->color_blend_op),
               vk_blend_factor_to_pipe(cb->src_color_blend_factor),
               vk_blend_factor_to_pipe(cb->dst_color_blend_factor),
               vk_blend_op_to_pipe(cb->alpha_blend_op),
               vk_blend_factor_to_pipe(cb->src_alpha_blend_factor),
               vk_blend_factor_to_pipe(cb->dst_alpha_blend_factor)),
         };
      }
   }

   return true;
}

static void
hk_prelink_graphics(struct hk_cmd_buffer *cmd, bool indexed,
                    enum agx_index_size index_size, bool restart,
                    bool vertex_only)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (!gfx->shaders[MESA_SHADER_VERTEX] ||
       !gfx->shaders[MESA_SHADER_FRAGMENT] ||
       vk_command_buffer_has_error(&cmd->vk))
      return;

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);
   struct hk_fast_link_key_vs vs_key;
   hk_fill_fast_link_key_vs(cmd, sw_vs, hw_vs, indexed, index_size, restart,
                            &vs_key);
   if (!hk_update_fast_linked(cmd, sw_vs, &vs_key))
      return;

   gfx->prepared_fast_link_vs = (struct hk_prepared_fast_link_vs){
      .sw_vs = sw_vs,
      .hw_vs = hw_vs,
      .dynamic_generation = cmd->vk.dynamic_graphics_state.generation,
      .index_size = index_size,
      .indexed = indexed,
      .restart = restart,
      .valid = true,
   };

   if (!vertex_only) {
      struct hk_shader *fs = hk_bound_fs(cmd);
      struct hk_fast_link_key_fs fs_key;
      if (hk_fill_fast_link_key_fs(cmd, fs, hw_vs, &fs_key, false))
         (void)hk_update_fast_linked(cmd, fs, &fs_key);
   }
}

static inline bool
hk_prelinked_vs_matches(const struct hk_cmd_buffer *cmd,
                        const struct hk_shader *sw_vs,
                        const struct hk_shader *hw_vs,
                        const struct agx_draw draw)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_prepared_fast_link_vs *prepared =
      &gfx->prepared_fast_link_vs;

   return prepared->valid && gfx->fast_link_vs.valid &&
          gfx->linked[MESA_SHADER_VERTEX] != NULL &&
          prepared->sw_vs == sw_vs && prepared->hw_vs == hw_vs &&
          prepared->dynamic_generation ==
             cmd->vk.dynamic_graphics_state.generation &&
          prepared->indexed == draw.indexed &&
          (!draw.indexed ||
           (prepared->index_size == draw.index_size &&
           prepared->restart == draw.restart));
}

/* Consume a complete unpublished G17P state transaction without entering the
 * 20 KiB general dynamic-state builder.  This is deliberately an exact fast
 * path: binding/render setup has already derived every immutable field, while
 * this function only fills draw-authored records and publishes the prepared
 * state.  A mismatch returns before mutating any single-use allocation. */
static __attribute__((hot, noinline)) bool
hk_g17p_flush_prepared_dynamic_state(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint32_t draw_id,
   struct agx_draw draw, bool ordinary_draw_path,
   bool *g17p_native_bind_tail_emitted_out)
{
   /* Nothing was prepared since the last draw consumed it: the common case
    * for every draw after the first of a binding, answered before any of the
    * state below is looked up. */
   if (!cmd->state.gfx.prepared_vertex_usc)
      return false;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);
   struct hk_linked_shader *linked_vs = gfx->linked[MESA_SHADER_VERTEX];
   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   const bool transformed = gfx->shaders[MESA_SHADER_TESS_EVAL] ||
                            gfx->shaders[MESA_SHADER_GEOMETRY] ||
                            (linked_vs && linked_vs->sw_indexing);

   if (dev->dev.chip != AGX_CHIP_G17P || !ordinary_draw_path ||
       desc->root_dirty || !gfx->prepared_vertex_usc ||
       !gfx->prepared_vdm_state_update_valid || !linked_vs ||
       !linked_fs || !hk_prelinked_vs_matches(cmd, sw_vs, hw_vs, draw) ||
       !hk_g17p_prepared_bind_tail_matches(cmd, &gfx->g17p_bind_tail) ||
       (transformed &&
        (!gfx->prepared_vertex_params.gpu ||
         desc->root.draw.vertex_params !=
            gfx->prepared_vertex_params.gpu)) ||
       (gfx->shaders[MESA_SHADER_TESS_EVAL] &&
        (!gfx->prepared_tess_params.gpu ||
         desc->root.draw.tess_params != gfx->prepared_tess_params.gpu)) ||
       (gfx->shaders[MESA_SHADER_GEOMETRY] &&
        (!gfx->prepared_geometry_params.gpu ||
         desc->root.draw.geometry_params !=
            gfx->prepared_geometry_params.gpu)) ||
       (!agx_is_indirect(draw.b) && linked_vs->b.uses_base_param &&
        !gfx->prepared_draw_params.gpu) ||
       (sw_vs->b.info.uses_draw_id && !gfx->prepared_draw_id.gpu))
      return false;

   const bool perf_trace = hk_shader_perf_draw_enabled(dev);
   uint64_t phase_start = perf_trace ? os_time_get_nano() : 0;
   uint64_t upload_params_ns = 0, publish_program_ns = 0;
   uint64_t publish_vdm_ns = 0, publish_tail_ns = 0;

   *g17p_native_bind_tail_emitted_out = false;
   cs->g17p_state_publications++;
   uint8_t *out = cs->current;

   struct agx_vdm_barrier_core interdraw_barrier;
   if (cs->stats.cmds > 0 && debug_get_option_hk_g17p_interdraw_barrier() &&
       agx_vdm_barrier_core_pack(dev->dev.chip, &interdraw_barrier)) {
      memcpy(out, interdraw_barrier.bytes, interdraw_barrier.length);
      out += interdraw_barrier.length;
   }

   if (linked_vs->b.uses_base_param) {
      if (agx_is_indirect(draw.b)) {
         gfx->draw_params = draw.b.ptr +
            (draw.indexed
                ? offsetof(VkDrawIndexedIndirectCommand, vertexOffset)
                : offsetof(VkDrawIndirectCommand, firstVertex));
         gfx->g17p_draw_params_host_valid = false;
      } else {
         const uint32_t params[2] = {
            draw.indexed ? draw.index_bias : draw.start,
            draw.start_instance,
         };
         memcpy(gfx->g17p_draw_params_value, params, sizeof(params));
         memcpy(gfx->prepared_draw_params.cpu, params, sizeof(params));
         gfx->draw_params = gfx->prepared_draw_params.gpu;
         gfx->prepared_draw_params = (struct agx_ptr){0};
         gfx->g17p_draw_params_host_valid = true;
      }
   } else {
      gfx->draw_params = 0;
      gfx->g17p_draw_params_host_valid = false;
   }

   if (sw_vs->b.info.uses_draw_id) {
      gfx->g17p_draw_id_value = draw_id;
      gfx->g17p_draw_id_host_valid = true;
      memcpy(gfx->prepared_draw_id.cpu, &draw_id, 4);
      gfx->draw_id_ptr = gfx->prepared_draw_id.gpu;
      gfx->prepared_draw_id = (struct agx_ptr){0};
   } else {
      gfx->draw_id_ptr = 0;
      gfx->g17p_draw_id_host_valid = false;
   }

   if (IS_DIRTY(MS_SAMPLE_LOCATIONS) || IS_DIRTY(MS_SAMPLE_LOCATIONS_ENABLE) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES)) {
      const uint32_t ctrl = dyn->ms.sample_locations_enable
         ? hk_pack_ppp_multisamplectrl(dyn->ms.sample_locations)
         : hk_default_sample_positions(dyn->ms.rasterization_samples);
      const bool dont_commit =
         cmd->in_meta || dyn->ms.rasterization_samples == 0;
      if (!cs->has_sample_locations) {
         cs->ppp_multisamplectl = ctrl;
         cs->has_sample_locations |= !dont_commit;
      } else {
         assert(dont_commit || cs->ppp_multisamplectl == ctrl);
      }
      assert(desc->root.draw.ppp_multisamplectl == ctrl);
   }

   if (transformed) {
      const uint64_t old_outputs = desc->root.draw.vertex_outputs;
      uint64_t vertex_params = 0;
      if (!agx_is_indirect(draw.b) && !draw.restart &&
          gfx->prepared_vertex_params_filled)
         vertex_params =
            hk_g17p_fill_prepared_direct_vertex_params(cmd, draw);
      else
         vertex_params = hk_upload_vertex_params(cmd, draw);
      if (!vertex_params || desc->root.draw.vertex_params != vertex_params ||
          desc->root.draw.vertex_outputs != old_outputs) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return true;
      }
   }

   struct agx_ptr tess_args = {0};
   if (gfx->shaders[MESA_SHADER_TESS_EVAL]) {
      tess_args = gfx->prepared_tess_params;
      gfx->prepared_tess_params = (struct agx_ptr){0};
      hk_upload_tess_params(cmd, tess_args.cpu, draw);
      gfx->tess.params = tess_args.cpu;
   }

   if (gfx->shaders[MESA_SHADER_GEOMETRY]) {
      uint64_t geometry_params;
      if (gfx->prepared_geometry_params_filled && !gfx->xfb_enabled &&
          !gfx->xfb_query[0] && !gfx->xfb_query[1] &&
          !gfx->xfb_query[2] && !gfx->xfb_query[3]) {
         geometry_params = gfx->prepared_geometry_params.gpu;
         gfx->prepared_geometry_params = (struct agx_ptr){0};
         gfx->prepared_geometry_params_filled = false;
      } else {
         geometry_params = hk_upload_geometry_params(cmd, draw);
      }
      if (!geometry_params ||
          desc->root.draw.geometry_params != geometry_params) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return true;
      }
   }

   if (perf_trace) {
      upload_params_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   assert(!desc->root_dirty);
   hk_reserve_scratch(cmd, cs, hw_vs);

   struct hk_linked_shader *linked_hw_vs =
      hw_vs == sw_vs ? linked_vs : hw_vs->only_linked;
   bool vertex_exact = gfx->prepared_vertex_shader == hw_vs &&
                       gfx->prepared_vertex_linked == linked_hw_vs &&
                       gfx->prepared_vertex_root == gfx->root;
   if (vertex_exact &&
       (gfx->prepared_vertex_draw_params != gfx->draw_params ||
        gfx->prepared_vertex_draw_id_ptr != gfx->draw_id_ptr)) {
      vertex_exact = hk_retarget_prepared_usc_root(
         cmd, hw_vs, linked_hw_vs, gfx->prepared_vertex_usc,
         gfx->prepared_vertex_compact_root);
   }
   if (!vertex_exact) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return true;
   }

   gfx->prepared_vertex_usc = 0;
   gfx->prepared_vertex_compact_root = (struct agx_ptr){0};

   if (perf_trace) {
      publish_program_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   memcpy(out, gfx->prepared_vdm_state_update.bytes,
          gfx->prepared_vdm_state_update.length);
   out += gfx->prepared_vdm_state_update.length;
   gfx->prepared_vdm_state_update_valid = false;

   if (perf_trace) {
      publish_vdm_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   /* The exact predicate above already validated the full tail identity.
    * Publish it directly instead of walking the large snapshot a second time
    * in hk_g17p_emit_native_bind_tail(). */
   if (!hk_g17p_publish_native_bind_tail(
          cmd, cs, &out, &gfx->g17p_bind_tail)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      cs->current = out;
      return true;
   }
   *g17p_native_bind_tail_emitted_out = true;
   cs->current = out;

   if (perf_trace) {
      publish_tail_ns = os_time_get_nano() - phase_start;
      hk_shader_perf_dynamic_detail_record(
         0, 0, upload_params_ns, 0, 0, publish_program_ns,
         publish_vdm_ns, publish_tail_ns);
   }

   vk_dynamic_graphics_state_clear_dirty(dyn);
   gfx->shaders_dirty = 0;
   gfx->linked_dirty = 0;
   gfx->dirty = 0;
   desc->root_dirty = false;
   return true;
}

/*
 * Does this draw republish render state, or does it reuse what the previous
 * draw in the same stream latched?
 *
 * This is the exact condition hk_flush_dynamic_state() uses to decide whether
 * it writes anything at all, hoisted so the split policy and the emitter
 * cannot drift apart.  It must be called at the point where every producer of
 * dirtiness -- the passthrough-GS bind, hk_flush_shaders(), the push-descriptor
 * flush -- has already run, otherwise it answers for a different draw than the
 * one that gets recorded.
 */
static bool
hk_gfx_republishes_render_state(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);

   return vk_dynamic_graphics_state_any_dirty(dyn) || gfx->dirty ||
          gfx->descriptors.root_dirty || gfx->shaders_dirty ||
          sw_vs->b.info.uses_draw_id || sw_vs->b.info.uses_base_param ||
          (gfx->linked[MESA_SHADER_VERTEX] &&
           gfx->linked[MESA_SHADER_VERTEX]->b.uses_base_param);
}

static bool
hk_vdm_update_inputs_equal(const struct agx_vdm_state_update_draw_info *a,
                           const struct agx_vdm_state_update_draw_info *b)
{
   const struct agx_vdm_state_update_template_info *x = &a->template_info;
   const struct agx_vdm_state_update_template_info *y = &b->template_info;

   return a->source_layout == b->source_layout &&
          a->draw_path == b->draw_path &&
          a->rasterization_enabled == b->rasterization_enabled &&
          a->raw_provoking_selector == b->raw_provoking_selector &&
          x->raw_state_load_count_12 == y->raw_state_load_count_12 &&
          x->raw_state_load_count_18 == y->raw_state_load_count_18 &&
          x->raw_vs_output_count_0 == y->raw_vs_output_count_0 &&
          x->raw_vs_output_count_2 == y->raw_vs_output_count_2 &&
          x->raw_vs_output_count_1_extra == y->raw_vs_output_count_1_extra &&
          x->raw_vs_output_selector == y->raw_vs_output_selector &&
          x->raw_vs_output_flag_6 == y->raw_vs_output_flag_6 &&
          x->pre_hal300_flag_group_nonzero ==
             y->pre_hal300_flag_group_nonzero &&
          x->pre_hal300_flag_bit_4 == y->pre_hal300_flag_bit_4 &&
          x->variant == y->variant &&
          x->raw_active_context_selector == y->raw_active_context_selector;
}

/* hk_agx3_encode_vdm_state_update_draw(), reusing the previous encoding when
 * every input except the pipeline address is unchanged.  The pipeline
 * address occupies only the two-word pipeline pair at bytes 8..15
 * (agx_vdm_state_update_pack()), and that pair is packed by the same
 * function with the same template field, so the bytes and the refusal
 * condition are exactly the encoder's. */
static bool
hk_g17p_encode_vdm_state_update(struct hk_cmd_buffer *cmd, enum agx_chip chip,
                                const struct agx_vdm_state_update_draw_info *info,
                                struct agx_vdm_state_update *encoded)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   if (gfx->g17p_vdm_update_memo.valid &&
       hk_vdm_update_inputs_equal(&gfx->g17p_vdm_update_memo.info, info)) {
      struct agx_vdm_state_vertex_pipeline_pair pair;
      *encoded = gfx->g17p_vdm_update_memo.encoded;
      if (!agx_vdm_state_vertex_pipeline_pair_pack(
             chip, info->template_info.pipeline,
             info->template_info.raw_state_load_count_18 << 3, &pair)) {
         memset(encoded, 0, sizeof(*encoded));
         return false;
      }
      static_assert(sizeof(pair.bytes) == 8, "pipeline pair is two words");
      memcpy(&encoded->bytes[8], pair.bytes, sizeof(pair.bytes));
      return true;
   }

   if (!hk_agx3_encode_vdm_state_update_draw(chip, info, encoded))
      return false;

   gfx->g17p_vdm_update_memo.info = *info;
   gfx->g17p_vdm_update_memo.encoded = *encoded;
   gfx->g17p_vdm_update_memo.valid = true;
   return true;
}

/* The AGX3 VertexProgramVariant state update for pipeline_program and, on
 * G17P, the native HAL200 bind tail: the vertex-program publication of every
 * draw that republishes render state.  Shared by hk_flush_dynamic_state_slow()
 * and hk_g17p_emit_indirect_record(), so both emit exactly the same records.
 * tail_known_match: see hk_g17p_emit_native_bind_tail().  Returns false, with
 * the command buffer error set and cs->current at the failed record, when
 * publication is refused. */
static bool
hk_agx3_publish_vertex_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                             uint8_t **out_p, struct hk_shader *hw_vs,
                             uint64_t pipeline_program, struct agx_draw draw,
                             bool ordinary_draw_path, bool tail_known_match,
                             bool *tail_emitted, bool perf_trace,
                             uint64_t *phase_start, uint64_t *publish_vdm_ns,
                             uint64_t *publish_tail_ns)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint8_t *out = *out_p;

   *tail_emitted = false;
   const struct agx_vdm_state_update_program_info *program =
      &hw_vs->agx3_vdm_state_update;
   const bool pipeline_va_valid =
      pipeline_program != 0 && pipeline_program < (UINT64_C(1) << 48) &&
      (pipeline_program & UINT64_C(0x3f)) == 0;
   const bool complete_program =
      pipeline_va_valid &&
      hk_agx3_vdm_live_source_layout(
         program, cs->agx3_active_context_selector_valid,
         ordinary_draw_path) ==
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;
   struct agx_vdm_state_update_draw_info info = {
      .source_layout = complete_program
            ? AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT
            : AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED,
      .draw_path = agx_is_indirect(draw.b)
         ? (draw.indexed
               ? AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED
               : AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_NONINDEXED)
         : (draw.indexed
               ? AGX_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED
               : AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED),
      /* RenderPipeline+0x2f0 is populated from
       * isRasterizationEnabled; the selected template carries 0x30
       * when this is false on pre-HAL300.  Software XFB stores live in
       * the raster variant's vertex program, so it must use the live
       * vertex template even when Vulkan discards rasterization.  The
       * independently packed HAL200 PPP discard bits still suppress
       * fragments. */
      .rasterization_enabled =
         !hk_rast_discard(cmd) ||
         hk_g17p_xfb_vertex_side_effects(cmd, hw_vs),
      /* hk already stores the actual hardware provoking vertex 0..2.
       * Apple's setter maps its API mode to this same value before
       * selecting one of the six templates. */
      .raw_provoking_selector = gfx->provoking,
      .template_info = program->template_info,
   };
   info.template_info.pipeline = pipeline_program;
   info.template_info.raw_vs_output_flag_6 = gfx->generate_primitive_id;
   info.template_info.raw_vs_output_count_1_extra =
      gfx->generate_primitive_id;
   if (hk_g17p_cycle_vdm_context_enabled(dev) &&
       cs->agx3_raw_active_context_selector == UINT32_MAX)
      cs->agx3_raw_active_context_selector = 1;
   info.template_info.raw_active_context_selector =
      cs->agx3_raw_active_context_selector;

   /* Keep VertexProgramVariant's UVS footprint coupled to the HAL200
    * output state selected below. Native non-point draws omit the
    * terminal PointSize word from both records. */
   if (dev->dev.chip == AGX_CHIP_G17P &&
       hk_g17p_nonpoint_no_psiz_diagnostic(
          dev->g17p_diagnostics.nonpoint_no_psiz, gfx, hw_vs)) {
      assert(info.template_info.raw_vs_output_count_0 > 0);
      --info.template_info.raw_vs_output_count_0;
   }

   struct agx_vdm_state_update encoded;
   if (!hk_g17p_encode_vdm_state_update(cmd, dev->dev.chip, &info,
                                        &encoded)) {
      const char *reason =
         !pipeline_va_valid ? "pipeline-va"
         : program->source_layout !=
              AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT
            ? "shader-layout"
         : !cs->agx3_active_context_selector_valid ? "active-context"
         : !ordinary_draw_path ? "transformed-draw"
                               : "template-fields";
      mesa_loge_once(
         "no complete AGX3 VertexProgramVariant state for %s: "
         "reason=%s pipeline=0x%" PRIx64
         " layout=%u active=%u/0x%08x ordinary=%u hw_eq_sw=%u "
         "draw=%u indexed=%u indirect=%u restart=%u raster=%u "
         "provoking=%u prim_id=%u topology=%u template="
         "%u/%u/%u/%u/%u/%u/%u/%u",
         agx_chip_name(dev->dev.chip), reason, pipeline_program,
         (unsigned)program->source_layout,
         cs->agx3_active_context_selector_valid,
         cs->agx3_raw_active_context_selector, ordinary_draw_path,
         hw_vs == hk_bound_sw_vs(gfx), (unsigned)info.draw_path, draw.indexed,
         agx_is_indirect(draw.b), draw.restart,
         info.rasterization_enabled, info.raw_provoking_selector,
         gfx->generate_primitive_id,
         (unsigned)dyn->ia.primitive_topology,
         info.template_info.raw_state_load_count_12,
         info.template_info.raw_state_load_count_18,
         info.template_info.raw_vs_output_count_0,
         info.template_info.raw_vs_output_count_2,
         info.template_info.raw_vs_output_count_1_extra,
         info.template_info.raw_vs_output_selector,
         info.template_info.raw_vs_output_flag_6,
         info.template_info.pre_hal300_flag_group_nonzero);
      vk_command_buffer_set_error(&cmd->vk,
                                  VK_ERROR_FEATURE_NOT_PRESENT);
      cs->current = out;
      return false;
   }

   /* Keep both words of the packed 48-bit vertex-pipeline pair.  The
    * Metal witness's 0x01000000/0x00004a00 pair decodes to its own
    * 0x10000128000 pipeline allocation; substituting only the second
    * word redirects Honeykrisp to an unrelated shader container.
    */

   memcpy(out, encoded.bytes, encoded.length);
   out += encoded.length;

   /* Host-checkable launch encoding: word 1 bits 8..19 and the
    * pipeline template bits 3..20 carry a vertex CProg's TI counts. */
   if (unlikely(hk_g17p_launch_trace_enabled()))
      hk_g17p_trace_line("vdm-state-update", pipeline_program,
                         encoded.bytes, encoded.length);

   if (perf_trace) {
      *publish_vdm_ns = os_time_get_nano() - *phase_start;
      *phase_start = os_time_get_nano();
   }

   if (hk_g17p_cycle_vdm_context_enabled(dev)) {
      const unsigned contexts = hk_g17p_vdm_contexts(dev);

      assert(cs->agx3_raw_active_context_selector >= 1 &&
             cs->agx3_raw_active_context_selector <= contexts);
      cs->agx3_raw_active_context_selector =
         cs->agx3_raw_active_context_selector >= contexts
            ? 1
            : cs->agx3_raw_active_context_selector + 1;
   }

   if (dev->dev.chip == AGX_CHIP_G17P) {
      if (!hk_g17p_emit_native_bind_tail(cmd, cs, &out,
                                           tail_known_match)) {
         mesa_loge("failed to emit native HAL200 bind tail for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         cs->current = out;
         return false;
      }
      *tail_emitted = true;

      if (perf_trace) {
         *publish_tail_ns = os_time_get_nano() - *phase_start;
         *phase_start = os_time_get_nano();
      }
   }

   *out_p = out;
   return true;
}

static __attribute__((noinline)) void
hk_flush_dynamic_state_slow(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            uint32_t draw_id, struct agx_draw draw,
                            bool ordinary_draw_path,
                            bool *g17p_native_bind_tail_emitted_out,
                            bool prepare_only)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const bool perf_trace = hk_shader_perf_draw_enabled(dev);
   static uint32_t perf_dynamic_calls;
   const uint32_t perf_call = dev->g17p_shader_perf.trace
      ? p_atomic_inc_return(&perf_dynamic_calls) : 0;
   uint64_t phase_start = perf_trace ? os_time_get_nano() : 0;
   uint64_t prelink_vs_ns = 0, prelink_fs_ns = 0, upload_ns = 0;
   uint64_t publish_ns = 0;
   uint64_t upload_params_ns = 0, upload_varyings_ns = 0;
   uint64_t upload_root_ns = 0, publish_program_ns = 0;
   uint64_t publish_vdm_ns = 0, publish_tail_ns = 0;

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);
   bool g17p_native_bind_tail_emitted = false;
   *g17p_native_bind_tail_emitted_out = false;

   /* The caller admits ordinary draws plus the bounded passthrough-GS XFB
    * route.  Reject every other software-transformed draw even if no state is
    * dirty, otherwise it could inherit a preceding draw's AGX3 state. */
   if (agx_chip_command_stream_generation(dev->dev.chip) !=
          AGX_CS_GENERATION_AGX2 &&
       !ordinary_draw_path) {
      mesa_loge("no grounded AGX3 VertexProgramVariant state for transformed "
                "draw path on %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   if (!hk_gfx_republishes_render_state(cmd))
      return;

   /* Past this point the draw installs a fresh vertex-program template, a
    * fresh HAL200 bind tail and a fresh draw-state selector, all into the
    * stream's one render-state context. Count it: that depth, not the draw
    * count, is what hk_g17p_state_publications_per_stream() bounds. */
   if (!prepare_only)
      cs->g17p_state_publications++;

   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   assert(cs->current + 0x1000 < cs->end && "already ensured space");
   uint8_t *out = cs->current;
   /* Native captures concatenate state publications without this cache
    * invalidation. Retain the old barrier only as a diagnostic control;
    * immutable per-draw state does not require a per-draw cache flush. */
   struct agx_vdm_barrier_core interdraw_barrier;
   if (!prepare_only && cs->stats.cmds > 0 &&
       debug_get_option_hk_g17p_interdraw_barrier() &&
       agx_vdm_barrier_core_pack(dev->dev.chip, &interdraw_barrier)) {
      memcpy(out, interdraw_barrier.bytes, interdraw_barrier.length);
      out += interdraw_barrier.length;
   }

   struct hk_shader *fs = hk_bound_fs(cmd);
   bool gt_dirty = IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL) ||
                   IS_SHADER_DIRTY(GEOMETRY);
   bool vgt_dirty = IS_SHADER_DIRTY(VERTEX) || gt_dirty;
   bool fs_dirty = IS_SHADER_DIRTY(FRAGMENT);
   struct hk_linked_shader *prepared_linked_vs =
      gfx->linked[MESA_SHADER_VERTEX];
   struct hk_linked_shader *prepared_linked_fs =
      gfx->linked[MESA_SHADER_FRAGMENT];

   /* BeginRendering / shader binding already ran this exact state through the
    * authoritative builder without publishing it.  When every identity still
    * matches the draw, skip re-deriving the immutable root/link/tail state and
    * continue at the genuinely draw-dependent parameter fill.  Indirect
    * values remain GPU-authored: the later block installs the descriptor VA,
    * and the compact-root retarget only changes that VA before publication.
    *
    * Any intervening state, descriptor, shader, root, or indexed-form change
    * misses this predicate and takes the complete path below. */
   const bool g17p_prepared_exact =
      !debug_get_option_hk_g17p_prepared_dynamic_disable() &&
      dev->dev.chip == AGX_CHIP_G17P && !prepare_only &&
      !desc->root_dirty && gfx->prepared_vertex_usc && prepared_linked_vs &&
      prepared_linked_fs &&
      hk_prelinked_vs_matches(cmd, sw_vs, hw_vs, draw) &&
      hk_g17p_prepared_bind_tail_matches(cmd, &gfx->g17p_bind_tail);
   if (g17p_prepared_exact)
      goto g17p_draw_parameters;

   if (IS_DIRTY(CB_BLEND_CONSTANTS) &&
       memcmp(desc->root.draw.blend_constant, dyn->cb.blend_constants,
              sizeof(dyn->cb.blend_constants))) {
      static_assert(sizeof(desc->root.draw.blend_constant) ==
                       sizeof(dyn->cb.blend_constants),
                    "common size");

      memcpy(desc->root.draw.blend_constant, dyn->cb.blend_constants,
             sizeof(dyn->cb.blend_constants));
      desc->root_dirty = true;
   }

   if (IS_DIRTY(CB_BLEND_EQUATIONS) || IS_DIRTY(CB_BLEND_ENABLES)) {
      gfx->uses_blend_constant = uses_blend_constant(&dyn->cb);
   }

   if (IS_DIRTY(MS_SAMPLE_MASK)) {
      const uint16_t sample_mask = dyn->ms.sample_mask;
      if (desc->root.draw.api_sample_mask != sample_mask) {
         desc->root.draw.api_sample_mask = sample_mask;
         desc->root_dirty = true;
      }
   }

   if (IS_DIRTY(RS_RASTERIZATION_STREAM)) {
      const uint16_t stream = dyn->rs.rasterization_stream;
      if (desc->root.draw.rasterization_stream != stream) {
         desc->root.draw.rasterization_stream = stream;
         desc->root_dirty = true;
      }

      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE);
   }

   if (fs_dirty || IS_DIRTY(DS_DEPTH_TEST_ENABLE) ||
       IS_DIRTY(DS_DEPTH_COMPARE_OP)) {

      const struct hk_rendering_state *render = &cmd->state.gfx.render;
      bool has_z = render->depth_att.vk_format != VK_FORMAT_UNDEFINED;
      bool z_test = has_z && dyn->ds.depth.test_enable;

      const uint16_t force_never_in_shader =
         z_test && dyn->ds.depth.compare_op == VK_COMPARE_OP_NEVER && fs &&
         fs->info.fs.writes_memory;

      if (desc->root.draw.force_never_in_shader != force_never_in_shader) {
         desc->root.draw.force_never_in_shader = force_never_in_shader;
         desc->root_dirty = true;
      }
   }

   /* The main shader must not run tests if the epilog will. */
   bool nontrivial_force_early =
      fs && (fs->b.info.early_fragment_tests &&
             (fs->b.info.writes_sample_mask || fs->info.fs.writes_memory));

   bool epilog_discards = agx_fs_epilog_may_discard(
      fs ? &fs->info.fs.epilog_key : NULL,
      dyn->ms.alpha_to_coverage_enable, nontrivial_force_early);

   if (fs_dirty || IS_DIRTY(MS_ALPHA_TO_COVERAGE_ENABLE)) {
      const uint16_t no_epilog_discard = !epilog_discards ? UINT16_MAX : 0;
      const uint16_t alpha_to_coverage = dyn->ms.alpha_to_coverage_enable;
      if (desc->root.draw.no_epilog_discard != no_epilog_discard ||
          desc->root.draw.alpha_to_coverage != alpha_to_coverage) {
         desc->root.draw.no_epilog_discard = no_epilog_discard;
         desc->root.draw.alpha_to_coverage = alpha_to_coverage;
         desc->root_dirty = true;
      }
   }

   if ((IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) ||
        IS_DIRTY(VI_BINDING_STRIDES) || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
        vgt_dirty || !gfx->fast_link_vs.valid) &&
       !hk_prelinked_vs_matches(cmd, sw_vs, hw_vs, draw)) {
      struct hk_fast_link_key_vs key;
      hk_fill_fast_link_key_vs(cmd, sw_vs, hw_vs, draw.indexed,
                               draw.index_size, draw.restart, &key);

      if (!hk_update_fast_linked(cmd, sw_vs, &key)) {
         cs->current = out;
         return;
      }

   }

   if (perf_trace) {
      prelink_vs_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   if (IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) ||
       IS_DIRTY(VI_BINDING_STRIDES) || vgt_dirty ||
       (gfx->dirty & HK_DIRTY_VB)) {

      unsigned slot = 0;
      bool vertex_root_changed = false;
      u_foreach_bit(a, sw_vs->info.vs.attribs_read) {
         uint64_t base;
         uint32_t clamp;
         uint32_t stride;
         if (dyn->vi->attributes_valid & BITFIELD_BIT(a)) {
            struct vk_vertex_attribute_state attr = dyn->vi->attributes[a];
            struct hk_addr_range vb = gfx->vb[attr.binding];
            __typeof__(gfx->vb_format_memo[0]) *memo =
               &gfx->vb_format_memo[slot];
            if (memo->vk_format != attr.format || !memo->interchange_align) {
               memo->vk_format = attr.format;
               memo->format = hk_format_to_pipe_format(attr.format);
               memo->interchange_align = util_format_get_blocksize(
                  agx_vbo_internal_format(memo->format));
            }
            enum pipe_format fmt = memo->format;
            unsigned interchange_align = memo->interchange_align;

            clamp = agx_calculate_vbo_clamp(
               vb.addr, fmt, vb.range, dyn->vi_binding_strides[attr.binding],
               attr.offset, &base);
            stride =
               dyn->vi_binding_strides[attr.binding] / interchange_align;
         } else {
            base = AGX_ZERO_PAGE_ADDRESS;
            clamp = 0;
            stride = 0;
         }

         vertex_root_changed |= desc->root.draw.attrib_base[slot] != base ||
                                desc->root.draw.attrib_clamps[slot] != clamp ||
                                desc->root.draw.attrib_strides[slot] != stride;
         desc->root.draw.attrib_base[slot] = base;
         desc->root.draw.attrib_clamps[slot] = clamp;
         desc->root.draw.attrib_strides[slot] = stride;

         ++slot;
      }

      desc->root_dirty |= vertex_root_changed;
   }

   if (vgt_dirty || IS_SHADER_DIRTY(FRAGMENT) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES) || IS_DIRTY(MS_SAMPLE_MASK) ||
       IS_DIRTY(MS_ALPHA_TO_COVERAGE_ENABLE) ||
       IS_DIRTY(MS_ALPHA_TO_ONE_ENABLE) || IS_DIRTY(CB_LOGIC_OP) ||
       IS_DIRTY(CB_LOGIC_OP_ENABLE) || IS_DIRTY(CB_WRITE_MASKS) ||
       IS_DIRTY(CB_COLOR_WRITE_ENABLES) || IS_DIRTY(CB_ATTACHMENT_COUNT) ||
       IS_DIRTY(CB_BLEND_ENABLES) || IS_DIRTY(CB_BLEND_EQUATIONS) ||
       IS_DIRTY(CB_BLEND_CONSTANTS) || IS_DIRTY(CB_BLEND_ADVANCED) ||
       desc->root_dirty /* for pipeline stats */ ||
       !gfx->fast_link_fs.valid) {

      unsigned tib_sample_mask = BITFIELD_MASK(dyn->ms.rasterization_samples);
      unsigned api_sample_mask = dyn->ms.sample_mask & tib_sample_mask;
      bool has_sample_mask = api_sample_mask != tib_sample_mask;
      bool defer_api_sample_mask =
         dev->dev.chip == AGX_CHIP_G17P && has_sample_mask &&
         !nontrivial_force_early;

      if (hw_vs->info.cull_distance_array_size) {
         perf_debug(cmd, "Emulating cull distance (size %u)",
                    hw_vs->info.cull_distance_array_size);
      }

      if (has_sample_mask) {
         perf_debug(cmd, "Emulating sample mask");
      }

      /* A fixed-count shader whose sample exports exceeded the register file
       * was canonicalized to one sample. Keep that guard; dynamic sample-rate
       * shaders instead select a native variant in hk_bound_fs().
       */
      if (dev->dev.chip == AGX_CHIP_G17P && fs->info.fs.uses_sample_shading &&
          dyn->ms.rasterization_samples > 1) {
         mesa_loge("G17P: per-sample fragment execution needs a fixed sample "
                   "count; refusing a %u-sample draw that requests it",
                   dyn->ms.rasterization_samples);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      /* A fragment unrolled per sample shades exactly its pipeline's count. */
      if (dev->dev.chip == AGX_CHIP_G17P &&
          fs->info.fs.epilog_key.sample_shading &&
          fs->info.fs.g17p_samples != MAX2(dyn->ms.rasterization_samples, 1)) {
         mesa_loge("G17P: fragment unrolled for %u samples cannot run a "
                   "%u-sample draw",
                   fs->info.fs.g17p_samples, dyn->ms.rasterization_samples);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      /* A G17P fragment's coverage updates are compiled for the sample count
       * of its pipeline; a dynamic count other than that cannot reuse them.
       */
      if (dev->dev.chip == AGX_CHIP_G17P && fs->b.info.writes_sample_mask &&
          fs->info.fs.g17p_samples != MAX2(dyn->ms.rasterization_samples, 1)) {
         mesa_loge("G17P: fragment coverage updates compiled for %u samples "
                   "cannot run a %u-sample draw",
                   fs->info.fs.g17p_samples, dyn->ms.rasterization_samples);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      /* G17P shades every sample inside the fragment rather than looping. */
      unsigned samples_shaded = 0;
      if (fs->info.fs.epilog_key.sample_shading &&
          dev->dev.chip != AGX_CHIP_G17P)
         samples_shaded = dyn->ms.rasterization_samples;

      struct hk_fast_link_key_fs key = {
         .prolog.statistics = hk_stat_enabled(hk_pipeline_stat_addr(
            cmd, VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT)),

         .prolog.cull_distance_size = hw_vs->info.cull_distance_array_size,
         .prolog.api_sample_mask =
            has_sample_mask && !nontrivial_force_early &&
                  !defer_api_sample_mask
               ? api_sample_mask
               : 0xff,
         .prolog.nr_samples = dev->dev.chip == AGX_CHIP_G17P
                                 ? MAX2(dyn->ms.rasterization_samples, 1)
                                 : 0,
         .nr_samples_shaded = samples_shaded,
      };

      bool prolog_discards =
         (has_sample_mask && !defer_api_sample_mask) ||
         key.prolog.cull_distance_size;

      bool needs_prolog = key.prolog.statistics || prolog_discards;

      if (needs_prolog) {
         /* With late main shader tests, the prolog runs tests if neither the
          * main shader nor epilog will.
          *
          * With (nontrivial) early main shader tests, the prolog does not
          * run tests, the tests will run at the start of the main shader.
          * This ensures tests are after API sample mask and cull distance
          * discards.
          */
         key.prolog.run_zs_tests = !nontrivial_force_early &&
                                   !fs->b.info.writes_sample_mask &&
                                   !epilog_discards && prolog_discards;

         if (key.prolog.cull_distance_size) {
            key.prolog.cf_base = fs->b.info.varyings.fs.nr_cf;
         }
      }

      key.epilog = (struct agx_fs_epilog_key){
         .link = fs->info.fs.epilog_key,
         .nr_samples = MAX2(dyn->ms.rasterization_samples, 1),
         .blend.alpha_to_coverage = dyn->ms.alpha_to_coverage_enable,
         .blend.alpha_to_one = dyn->ms.alpha_to_one_enable,
         .blend.logicop_enable = dyn->cb.logic_op_enable,
         .blend.logicop_func = vk_logic_op_to_pipe(dyn->cb.logic_op),
         .defer_api_sample_mask = defer_api_sample_mask,
         .api_sample_mask = api_sample_mask,
      };
      agx_fs_epilog_apply_device_policy(dev->dev.chip, &key.epilog);

      key.epilog.native_spills = dev->dev.chip == AGX_CHIP_G17P &&
         agx_tilebuffer_spills(&render->tilebuffer);
      if (key.epilog.native_spills) {
         uint16_t base;
         if (!hk_g17p_relocated_blend_constant_base(fs, &base) ||
             ALIGN_POT(base, 4) > 512 - 12) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            cs->current = out;
            return;
         }
         key.epilog.emrt_heap_base = ALIGN_POT(base, 4);
         key.epilog.blend_constant_base = key.epilog.emrt_heap_base + 4;
      }

      if (!key.epilog.native_spills && dev->dev.chip == AGX_CHIP_G17P && gfx->uses_blend_constant &&
          (fs->info.agx3_combined_sampler_valid ||
           fs->info.agx3_indexed_sampler_valid) &&
          !hk_g17p_relocated_blend_constant_base(
             fs, &key.epilog.blend_constant_base)) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         cs->current = out;
         return;
      }

      for (unsigned rt = 0; rt < ARRAY_SIZE(dyn->cal.color_map); ++rt) {
         int map = dyn->cal.color_map[rt];
         key.epilog.remap[rt] = map == MESA_VK_ATTACHMENT_UNUSED ? -1 : map;
      }

      if (dyn->ms.alpha_to_one_enable || dyn->ms.alpha_to_coverage_enable ||
          dyn->cb.logic_op_enable) {

         perf_debug(
            cmd, "Epilog with%s%s%s",
            dyn->ms.alpha_to_one_enable ? " alpha-to-one" : "",
            dyn->ms.alpha_to_coverage_enable ? " alpha-to-coverage" : "",
            dyn->cb.logic_op_enable ? " logic-op" : "");
      }

      key.epilog.link.already_ran_zs |= nontrivial_force_early;

      struct hk_rendering_state *render = &cmd->state.gfx.render;
      key.epilog.max_sample_size_B = render->tilebuffer.max_sample_size_B;
      key.epilog.max_bytes_per_tile = render->tilebuffer.max_bytes_per_tile;
      key.epilog.max_samples_per_tile = render->tilebuffer.max_samples_per_tile;
      for (uint32_t i = 0; i < render->color_att_count; i++) {
         key.epilog.rt_formats[i] =
            hk_format_to_pipe_format(render->color_att[i].vk_format);
         if (agx_native_graphics(&dev->dev)) {
            const enum pipe_format logical = key.epilog.rt_formats[i];
            key.epilog.rt_formats[i] = hk_padded_image_format(logical);
            const enum pipe_format integer =
               hk_scaled_texture_format(key.epilog.rt_formats[i]);
            if (integer != PIPE_FORMAT_NONE) {
               key.epilog.rt_formats[i] = integer;
               key.epilog.scaled_formats[i] = logical;
            } else if (key.epilog.rt_formats[i] != logical) {
               key.epilog.blend_formats[i] = logical;
            }
         }

         const struct vk_color_blend_attachment_state *cb =
            &dyn->cb.attachments[i];

         bool write_enable = dyn->cb.color_write_enables & BITFIELD_BIT(i);
         unsigned write_mask = write_enable ? cb->write_mask : 0;

         /* A zero hardware mask deletes the tile store while raster coverage
          * still makes a HAL200 tile nonempty. Materialize the unchanged tile
          * value with the already-grounded blend load/store path instead.
          * nir_lower_blend always blends, so use a default blend state when
          * blending is disabled at an API level.
          */
         if ((dev->dev.chip == AGX_CHIP_G16G ||
              dev->dev.chip == AGX_CHIP_G17P) && !write_enable) {
            /* The epilog reloads and stores the tile value itself, which
             * unlike a ZERO/ONE blend also covers pure-integer formats. */
            key.epilog.preserve_rt |= BITFIELD_BIT(i);
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = BITFIELD_MASK(4),
               .mode = agx_pack_blend_standard(
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO),
            };
         } else if (!dyn->cb.attachments[i].blend_enable) {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = write_mask,
               .mode = agx_pack_blend_standard(
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO),
            };
         } else if (cb->color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = write_mask,
               .advanced_blend = 1,
               .mode = agx_pack_blend_advanced(
                  vk_advanced_blend_op_to_pipe(cb->color_blend_op),
                  vk_blend_overlap_to_pipe(cb->blend_overlap),
                  cb->src_premultiplied, cb->dst_premultiplied,
                  cb->clamp_results),
            };

            assert(cb->clamp_results == false);
         } else {
            key.epilog.blend.rt[i] = (struct agx_blend_rt_key){
               .colormask = write_mask,
               .mode = agx_pack_blend_standard(
                  vk_blend_op_to_pipe(cb->color_blend_op),
                  vk_blend_factor_to_pipe(cb->src_color_blend_factor),
                  vk_blend_factor_to_pipe(cb->dst_color_blend_factor),
                  vk_blend_op_to_pipe(cb->alpha_blend_op),
                  vk_blend_factor_to_pipe(cb->src_alpha_blend_factor),
                  vk_blend_factor_to_pipe(cb->dst_alpha_blend_factor)),
            };
         }
      }

      if (!hk_update_fast_linked(cmd, fs, &key)) {
         cs->current = out;
         return;
      }
   }

   if (perf_trace) {
      prelink_fs_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

g17p_draw_parameters:
   /* If the vertex shader uses draw parameters, vertex uniforms are dirty every
    * draw. Fragment uniforms are unaffected.
    *
    * For a direct draw, we upload the draw parameters as-if indirect to
    * avoid keying to indirectness.
    */
   if (!prepare_only &&
       gfx->linked[MESA_SHADER_VERTEX]->b.uses_base_param) {
      if (agx_is_indirect(draw.b)) {
         gfx->draw_params = draw.b.ptr;
         gfx->g17p_draw_params_host_valid = false;

         if (draw.indexed) {
            gfx->draw_params +=
               offsetof(VkDrawIndexedIndirectCommand, vertexOffset);
         } else {
            gfx->draw_params += offsetof(VkDrawIndirectCommand, firstVertex);
         }
      } else {
         uint32_t params[] = {
            draw.indexed ? draw.index_bias : draw.start,
            draw.start_instance,
         };
         memcpy(gfx->g17p_draw_params_value, params, sizeof(params));
         gfx->g17p_draw_params_host_valid = true;

         if (gfx->prepared_draw_params.gpu) {
            memcpy(gfx->prepared_draw_params.cpu, params, sizeof(params));
            gfx->draw_params = gfx->prepared_draw_params.gpu;
            gfx->prepared_draw_params = (struct agx_ptr){0};
         } else {
            gfx->draw_params =
               hk_pool_upload(cmd, params, sizeof(params), 4);
         }
      }
   } else if (!prepare_only) {
      gfx->draw_params = 0;
      gfx->g17p_draw_params_host_valid = false;
   }

   if (!prepare_only && sw_vs->b.info.uses_draw_id) {
      gfx->g17p_draw_id_value = draw_id;
      gfx->g17p_draw_id_host_valid = true;
      /* Uploaded as a whole dword: the native G17P uniform range that
       * publishes it is loaded in 32-bit units, so a two-byte allocation
       * would leave the upper half of the loaded dword reading whatever
       * followed it in the pool. The shader only consumes the low half. */
      if (gfx->prepared_draw_id.gpu) {
         memcpy(gfx->prepared_draw_id.cpu, &draw_id, 4);
         gfx->draw_id_ptr = gfx->prepared_draw_id.gpu;
         gfx->prepared_draw_id = (struct agx_ptr){0};
      } else {
         gfx->draw_id_ptr = hk_pool_upload(cmd, &draw_id, 4, 4);
      }
   } else if (!prepare_only) {
      gfx->draw_id_ptr = 0;
      gfx->g17p_draw_id_host_valid = false;
   }

   if (!g17p_prepared_exact) {
      if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || gt_dirty) {
         enum mesa_prim prim = hk_rast_prim(cmd);

         gfx->topology = translate_hw_primitive_topology(prim);
         gfx->object_type = translate_object_type(prim);
      }

      if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || IS_DIRTY(RS_PROVOKING_VERTEX)) {
         unsigned provoking;
         if (dyn->rs.provoking_vertex ==
             VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT)
            provoking = 2;
         else if (gfx->topology == AGX_PRIMITIVE_TRIANGLE_FAN)
            provoking = 1;
         else
            provoking = 0;

         if (provoking != gfx->provoking) {
            gfx->provoking = provoking;
            gfx->dirty |= HK_DIRTY_PROVOKING;

            gfx->descriptors.root.draw.provoking = provoking;
            gfx->descriptors.root_dirty = true;
         }
      }
   }

   /* With attachmentless rendering, we don't know the sample count until draw
    * time, so we do a late tilebuffer fix up. But with rasterizer discard,
    * rasterization_samples might be 0.
    *
    * Note that we ignore dyn->ms.rasterization_samples when we do have a sample
    * count from an attachment. In Vulkan, these have to match anyway, but DX12
    * drivers are robust against this scenarios and vkd3d-proton will go out of
    * spec here. No reason we can't be robust here too.
    */
   if (dyn->ms.rasterization_samples && !gfx->render.tilebuffer.nr_samples) {
      agx_tilebuffer_set_samples(&gfx->render.tilebuffer,
                                 dyn->ms.rasterization_samples);

      /* The provisional attachmentless layout was admitted at begin time;
       * enforce the hardware contract again once the sample count is known.
       */
      if (dev->dev.chip == AGX_CHIP_G17P &&
          !(gfx->render.g17p_rt0_layout_supported =
               hk_agx3_g17p_rt0_tile_layout_supported(
                  &gfx->render.tilebuffer))) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      cs->tib = gfx->render.tilebuffer;
   }

   if (IS_DIRTY(MS_SAMPLE_LOCATIONS) || IS_DIRTY(MS_SAMPLE_LOCATIONS_ENABLE) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES)) {

      uint32_t ctrl;
      if (dyn->ms.sample_locations_enable) {
         ctrl = hk_pack_ppp_multisamplectrl(dyn->ms.sample_locations);
      } else {
         ctrl = hk_default_sample_positions(dyn->ms.rasterization_samples);
      }

      bool dont_commit = cmd->in_meta || dyn->ms.rasterization_samples == 0;

      if (!prepare_only && !cs->has_sample_locations) {
         cs->ppp_multisamplectl = ctrl;

         /* If we're in vk_meta, do not commit to the sample locations yet.
          * vk_meta doesn't care, but the app will!
          */
         cs->has_sample_locations |= !dont_commit;
      } else if (!prepare_only) {
         assert(dont_commit || cs->ppp_multisamplectl == ctrl);
      }

      if (gfx->descriptors.root.draw.ppp_multisamplectl != ctrl) {
         gfx->descriptors.root.draw.ppp_multisamplectl = ctrl;
         gfx->descriptors.root_dirty = true;
      }
   }

   if (perf_trace) {
      upload_params_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   /* Link varyings before uploading tessellation state, becuase the
    * gfx->generate_primitive_id boolean needs to be plumbed.
    */
   /* Fast-link updates above may replace either object.  Never carry the
    * admission snapshot into the authoritative path: doing so was the r246
    * correctness regression and made every later consumer see stale linkage. */
   struct hk_linked_shader *linked_vs = gfx->linked[MESA_SHADER_VERTEX];
   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   bool linked_vs_dirty = IS_LINKED_DIRTY(VERTEX);
   bool linked_fs_dirty = IS_LINKED_DIRTY(FRAGMENT);
   const bool prepared_varyings =
      dev->dev.chip == AGX_CHIP_G17P &&
      hk_g17p_prepared_bind_tail_matches(cmd, &gfx->g17p_bind_tail);
   /* Between here and the bind-tail emission below only the root can move
    * one of the tail's inputs (varyings are relinked only when this is
    * false), so the emission reuses this answer while the root is unchanged
    * instead of comparing the whole snapshot again. */
   const uint64_t prepared_varyings_root = gfx->root;
   if (unlikely(dev->g17p_shader_perf.trace && !prepared_varyings))
      hk_g17p_trace_prepared_bind_tail_miss(cmd, &gfx->g17p_bind_tail);

   if (!prepared_varyings &&
       ((gfx->dirty & HK_DIRTY_PROVOKING) || vgt_dirty || linked_fs_dirty)) {
      unsigned bindings = linked_fs->b.cf.nr_bindings;
      if (bindings) {
         size_t linkage_size =
            AGX_CF_BINDING_HEADER_LENGTH + (bindings * AGX_CF_BINDING_LENGTH);

         const bool native_vit = dev->dev.chip == AGX_CHIP_G17P;
         struct agx_ptr t =
            hk_pool_usc_alloc(cmd, linkage_size, native_vit ? 64 : 16);
         if (!t.cpu)
            return;

         agx_link_varyings_vs_fs(
            t.cpu, &gfx->linked_varyings, hw_vs->info.uvs.user_size,
            &linked_fs->b.cf, gfx->provoking, 0, &gfx->generate_primitive_id);

         gfx->varyings = native_vit ? t.gpu : agx_usc_addr(&dev->dev, t.gpu);
         if (native_vit && hk_g17p_launch_trace_enabled())
            hk_g17p_trace_line("hal200-varying-table", t.gpu, t.cpu,
                               linkage_size);
      } else {
         gfx->varyings = 0;
      }

      gfx->dirty |= HK_DIRTY_VARYINGS;
   }

   if (perf_trace) {
      upload_varyings_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   if (!prepare_only &&
       (gfx->shaders[MESA_SHADER_TESS_EVAL] ||
        gfx->shaders[MESA_SHADER_GEOMETRY] || linked_vs->sw_indexing)) {
      const uint64_t old_outputs = desc->root.draw.vertex_outputs;
      const uint64_t vertex_params = hk_upload_vertex_params(cmd, draw);
      if (desc->root.draw.vertex_params != vertex_params ||
          desc->root.draw.vertex_outputs != old_outputs) {
         desc->root.draw.vertex_params = vertex_params;
         desc->root_dirty = true;
      }
   }

   struct agx_ptr tess_args = {0};
   if (!prepare_only && gfx->shaders[MESA_SHADER_TESS_EVAL]) {
      if (gfx->prepared_tess_params.gpu) {
         tess_args = gfx->prepared_tess_params;
         gfx->prepared_tess_params = (struct agx_ptr){0};
      } else {
         tess_args = hk_pool_alloc(cmd, sizeof(struct poly_tess_params), 4);
      }
      if (gfx->descriptors.root.draw.tess_params != tess_args.gpu) {
         gfx->descriptors.root.draw.tess_params = tess_args.gpu;
         gfx->descriptors.root_dirty = true;
      }
   }

   if (!prepare_only && gfx->shaders[MESA_SHADER_GEOMETRY]) {
      const uint64_t geometry_params = hk_upload_geometry_params(cmd, draw);
      if (gfx->descriptors.root.draw.geometry_params != geometry_params) {
         gfx->descriptors.root.draw.geometry_params = geometry_params;
         gfx->descriptors.root_dirty = true;
      }
   }

   /* Root must be uploaded after the above, which touch the root. */
   if (gfx->descriptors.root_dirty) {
      gfx->root =
         hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   }

   /* The root owns only the stable pointer to this block.  Its contents are
    * per-draw state and must be populated even when the root itself is already
    * complete.  In particular, an indirect draw writes the GPU addresses of
    * the three execution-time grid records here; tying this fill to a root
    * re-upload leaves those addresses zero on the prepared-root path.
    *
    * Tess parameters consume the final root address, so allocate their stable
    * storage before root publication and fill it only after publication. */
   if (tess_args.cpu) {
      assert(gfx->root);
      hk_upload_tess_params(cmd, tess_args.cpu, draw);
      gfx->tess.params = tess_args.cpu;
   }

   if (prepare_only) {
      if (!gfx->root) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
      gfx->descriptors.root_dirty = false;
      hk_g17p_prepare_native_bind_tail(cmd);
      return;
   }

   if (perf_trace) {
      upload_root_ns = os_time_get_nano() - phase_start;
      upload_ns = upload_params_ns + upload_varyings_ns + upload_root_ns;
      phase_start = os_time_get_nano();
   }

   /* Hardware dynamic state must be deferred until after the root and fast
    * linking, since it will use the root address and the linked shaders.
    */
   /* Any dirty G17P draw must publish HAL200 state. In particular, stencil,
    * depth and line-width changes do not necessarily dirty the VS or root.
    * Letting those changes fall through to hk_flush_ppp_state() emits the
    * legacy aggregate layout into an otherwise native stream. The initial
    * draw hid this because all shader state was dirty on stream creation. */
   if (dev->dev.chip == AGX_CHIP_G17P ||
      (gfx->dirty & (HK_DIRTY_PROVOKING | HK_DIRTY_VARYINGS)) ||
      IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) || linked_vs_dirty || vgt_dirty ||
      gfx->descriptors.root_dirty || gfx->draw_id_ptr || gfx->draw_params ||
      IS_DIRTY(RS_DEPTH_BIAS_ENABLE) || IS_DIRTY(RS_DEPTH_BIAS_FACTORS) ||
      IS_DIRTY(VP_SCISSORS) || IS_DIRTY(VP_SCISSOR_COUNT) ||
      IS_DIRTY(VP_VIEWPORTS) || IS_DIRTY(VP_VIEWPORT_COUNT) ||
      IS_DIRTY(VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
      IS_DIRTY(RS_DEPTH_CLIP_ENABLE) || IS_DIRTY(RS_DEPTH_CLAMP_ENABLE)) {

      /* TODO: Do less often? */
      hk_reserve_scratch(cmd, cs, hw_vs);

      struct hk_linked_shader *linked_hw_vs =
         (hw_vs == sw_vs) ? linked_vs : hw_vs->only_linked;
      bool prepared_vertex_exact =
         gfx->prepared_vertex_usc &&
         gfx->prepared_vertex_shader == hw_vs &&
         gfx->prepared_vertex_linked == linked_hw_vs &&
         gfx->prepared_vertex_root == gfx->root;
      const bool prepared_vertex_pointers_exact =
         gfx->prepared_vertex_draw_params == gfx->draw_params &&
         gfx->prepared_vertex_draw_id_ptr == gfx->draw_id_ptr;
      if (prepared_vertex_exact && !prepared_vertex_pointers_exact) {
         prepared_vertex_exact = hk_retarget_prepared_usc_root(
            cmd, hw_vs, linked_hw_vs,
            gfx->prepared_vertex_usc,
            gfx->prepared_vertex_compact_root);
      }
      const uint64_t pipeline_program =
         prepared_vertex_exact
            ? gfx->prepared_vertex_usc
            : hk_upload_usc_words(cmd, hw_vs, linked_hw_vs);
      gfx->prepared_vertex_usc = 0;
      gfx->prepared_vertex_compact_root = (struct agx_ptr){0};

      if (perf_trace) {
         publish_program_ns = os_time_get_nano() - phase_start;
         phase_start = os_time_get_nano();
      }

      switch (agx_chip_command_stream_generation(dev->dev.chip)) {
      case AGX_CS_GENERATION_AGX2:
         if (pipeline_program > UINT32_MAX) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            cs->current = out;
            return;
         }

         agx_push(out, VDM_STATE, cfg) {
            cfg.vertex_shader_word_0_present = true;
            cfg.vertex_shader_word_1_present = true;
            cfg.vertex_outputs_present = true;
            cfg.vertex_unknown_present = true;
         }

         agx_push_packed(out, hw_vs->counts, VDM_STATE_VERTEX_SHADER_WORD_0);

         agx_push(out, VDM_STATE_VERTEX_SHADER_WORD_1, cfg) {
            cfg.pipeline = pipeline_program;
         }

         agx_push_packed(out, hw_vs->info.uvs.vdm,
                         VDM_STATE_VERTEX_OUTPUTS);

         agx_push(out, VDM_STATE_VERTEX_UNKNOWN, cfg) {
            cfg.flat_shading_control = translate_vdm_vertex(gfx->provoking);
            cfg.unknown_4 = cfg.unknown_5 = hk_rast_discard(cmd);
            cfg.generate_primitive_id = gfx->generate_primitive_id;
         }

         /* Pad up to a multiple of 8 bytes */
         memset(out, 0, 4);
         out += 4;
         break;

      case AGX_CS_GENERATION_G15:
      case AGX_CS_GENERATION_HAL200:
      case AGX_CS_GENERATION_HAL300: {
         bool tail_emitted = false;
         if (!hk_agx3_publish_vertex_state(
                cmd, cs, &out, hw_vs, pipeline_program, draw,
                ordinary_draw_path,
                prepared_varyings && gfx->root == prepared_varyings_root,
                &tail_emitted, perf_trace, &phase_start, &publish_vdm_ns,
                &publish_tail_ns))
            return;
         if (tail_emitted) {
            g17p_native_bind_tail_emitted = true;
            *g17p_native_bind_tail_emitted_out = true;
         }
         break;
      }

      default:
         mesa_loge("unknown command-stream generation for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         cs->current = out;
         return;
      }
   }

   /* A pure cull/winding update needs new immutable fragment state, but
    * leaves the vertex program and its VDM state unchanged. */
   if (dev->dev.chip == AGX_CHIP_G17P && !g17p_native_bind_tail_emitted &&
       (IS_DIRTY(RS_CULL_MODE) || IS_DIRTY(RS_FRONT_FACE))) {
      if (!hk_g17p_emit_native_bind_tail(cmd, cs, &out, false)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         cs->current = out;
         return;
      }
      g17p_native_bind_tail_emitted = true;
      *g17p_native_bind_tail_emitted_out = true;
   }

   if (!g17p_native_bind_tail_emitted && IS_DIRTY(RS_DEPTH_BIAS_FACTORS)) {
      bool exact = dyn->rs.depth_bias.exact;
      bool force_unorm =
         dyn->rs.depth_bias.representation ==
         VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;

      const float slope_scale = dyn->rs.depth_bias.slope_factor;
      const float clamp = dyn->rs.depth_bias.clamp;
      const float depth_bias =
         dyn->rs.depth_bias.constant_factor /
         hk_depth_bias_factor(render->depth_att.vk_format, exact, force_unorm);
      uint8_t record[AGX_DEPTH_BIAS_LENGTH];

      static_assert(AGX_DEPTH_BIAS_LENGTH ==
                    AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH);
      if (agx_chip_command_stream_generation(dev->dev.chip) ==
          AGX_CS_GENERATION_HAL300) {
         const struct agx_ppp_hal300_depth_bias_record_fields fields = {
            .depth_bias_raw = fui(depth_bias),
            .slope_scale_raw = fui(slope_scale),
            .clamp_raw = fui(clamp),
         };
         struct agx_ppp_hal300_depth_bias_record exact_record;
         if (!agx_ppp_hal300_depth_bias_record_pack(
                dev->dev.chip, &fields, &exact_record)) {
            mesa_loge("no grounded HAL300 depth-bias record for %s",
                      agx_chip_name(dev->dev.chip));
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            cs->current = out;
            return;
         }
         memcpy(record, exact_record.bytes, exact_record.length);
      } else {
         agx_pack(record, DEPTH_BIAS, cfg) {
            cfg.depth_bias = depth_bias;
            cfg.slope_scale = slope_scale;
            cfg.clamp = clamp;
         }
      }

      void *ptr =
         util_dynarray_grow_bytes(&cs->depth_bias, 1, sizeof(record));
      memcpy(ptr, record, sizeof(record));
   }

   /* Hardware viewport/scissor state is entangled with depth bias. */
   if (!g17p_native_bind_tail_emitted &&
       (IS_DIRTY(RS_DEPTH_BIAS_FACTORS) || IS_DIRTY(VP_SCISSORS) ||
       IS_DIRTY(VP_SCISSOR_COUNT) || IS_DIRTY(VP_VIEWPORTS) ||
       IS_DIRTY(VP_VIEWPORT_COUNT) ||
       IS_DIRTY(VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
       IS_DIRTY(RS_DEPTH_CLIP_ENABLE) || IS_DIRTY(RS_DEPTH_CLAMP_ENABLE))) {

      hk_flush_vp_state(cmd, cs, &out);
      if (vk_command_buffer_has_error(&cmd->vk)) {
         cs->current = out;
         return;
      }
   }

   if (!g17p_native_bind_tail_emitted)
      hk_flush_ppp_state(cmd, cs, &out);

   cs->current = out;

   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   vk_dynamic_graphics_state_clear_dirty(dyn);
   gfx->shaders_dirty = 0;
   gfx->linked_dirty = 0;
   gfx->dirty = 0;
   gfx->descriptors.root_dirty = false;

   if (perf_trace) {
      publish_ns = os_time_get_nano() - phase_start;
      hk_shader_perf_dynamic_detail_record(
         prelink_vs_ns, prelink_fs_ns, upload_params_ns, upload_varyings_ns,
         upload_root_ns, publish_program_ns, publish_vdm_ns,
         publish_tail_ns);
      if (dev->g17p_shader_perf.trace && perf_call <= 3) {
         fprintf(stderr,
                 "HK_SHADER_PERF dynamic-call=%u prelink_vs_ns=%" PRIu64
                 " prelink_fs_ns=%" PRIu64 " upload_ns=%" PRIu64
                 " publish_ns=%" PRIu64 " upload_params_ns=%" PRIu64
                 " upload_varyings_ns=%" PRIu64
                 " upload_root_ns=%" PRIu64
                 " publish_program_ns=%" PRIu64
                 " publish_vdm_ns=%" PRIu64
                 " publish_tail_ns=%" PRIu64 "\n",
                 perf_call, prelink_vs_ns, prelink_fs_ns, upload_ns,
                 publish_ns, upload_params_ns, upload_varyings_ns,
                 upload_root_ns, publish_program_ns, publish_vdm_ns,
                 publish_tail_ns);
      }
   }
}

static __attribute__((hot)) void
hk_flush_dynamic_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       uint32_t draw_id, struct agx_draw draw,
                       bool ordinary_draw_path,
                       bool *g17p_native_bind_tail_emitted_out,
                       bool prepare_only)
{
   if (unlikely(debug_get_option_hk_g17p_prepared_dynamic_disable())) {
      static bool reported;
      if (!reported) {
         fprintf(stderr, "G17P diagnostic: prepared dynamic consumption disabled\n");
         reported = true;
      }
   }

   if (!prepare_only &&
       !debug_get_option_hk_g17p_prepared_dynamic_disable() &&
       hk_g17p_flush_prepared_dynamic_state(
          cmd, cs, draw_id, draw, ordinary_draw_path,
          g17p_native_bind_tail_emitted_out))
      return;

   hk_flush_dynamic_state_slow(
      cmd, cs, draw_id, draw, ordinary_draw_path,
      g17p_native_bind_tail_emitted_out, prepare_only);
}

static bool
hk_needs_index_robustness(struct hk_cmd_buffer *cmd, struct agx_draw *draw,
                          const struct agx_draw *conditional_direct)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (!draw->indexed)
      return false;

   /* Geometry or tessellation use robust software index buffer fetch anyway */
   if (gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL])
      return false;

   /* Soft fault does not cover the hardware index buffer fetch, use the zero
    * page instead.
    */
   if (draw->index_buffer_range_B == 0 ||
       (dev->dev.chip == AGX_CHIP_G17P &&
        draw->index_buffer == AGX_ZERO_PAGE_ADDRESS &&
        draw->index_buffer_range_B == 4)) {
      draw->index_buffer = AGX_ZERO_PAGE_ADDRESS;
      draw->index_buffer_range_B = 4;
      draw->start = 0;
      return false;
   }

   if (!(dev->vk.enabled_features.robustBufferAccess ||
         dev->vk.enabled_features.robustBufferAccess2 ||
         dev->vk.enabled_features.pipelineRobustness))
      return false;

   /* A conditionally lowered direct draw has an immutable indirect record:
    * apply the direct draw's exact range test instead of conservatively
    * treating it as application-authored indirect data. */
   if (agx_is_indirect(draw->b))
      return conditional_direct
                ? agx_direct_draw_overreads_indices(*conditional_direct)
                : true;

   return agx_direct_draw_overreads_indices(*draw);
}

static void
hk_handle_passthrough_gs(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

   if (!IS_SHADER_DIRTY(VERTEX) && !IS_SHADER_DIRTY(GEOMETRY))
      return;

   /* If there's an application geometry shader, there's nothing to un/bind */
   if (gs && !gs->is_passthrough)
      return;

   /* Determine if we need a geometry shader to emulate XFB or adjacency */
   struct hk_shader *last_sw = hk_bound_sw_vs_before_gs(gfx);
   uint32_t xfb_outputs = last_sw->info.xfb_info.output_count;
   bool needs_gs = xfb_outputs;

   /* If we don't need a GS but we do have a passthrough, unbind it */
   if (!needs_gs) {
      if (gs != NULL) {
         assert(gs->is_passthrough);
         hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_GEOMETRY, NULL);
      }
      return;
   }

   /* Else, we need to bind a passthrough GS */
   size_t key_size = hk_passthrough_gs_key_size(xfb_outputs);
   struct hk_passthrough_gs_key *key = alloca(key_size);

   *key = (struct hk_passthrough_gs_key){
      .prim = u_decomposed_prim(hk_gs_in_prim(cmd)),
      .outputs = last_sw->b.info.outputs,
      .clip_distance_array_size = last_sw->info.clip_distance_array_size,
      .cull_distance_array_size = last_sw->info.cull_distance_array_size,
   };

   if (xfb_outputs) {
      typed_memcpy(key->xfb_stride, last_sw->info.xfb_stride,
                   ARRAY_SIZE(key->xfb_stride));

      memcpy(&key->xfb_info, &last_sw->info.xfb_info,
             nir_xfb_info_size(xfb_outputs));
   }

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   perf_debug(cmd, "Binding passthrough GS for%s\n", xfb_outputs ? " XFB" : "");

   gs = hk_meta_shader(dev, hk_nir_passthrough_gs, key, key_size);
   if (!gs) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   gs->is_passthrough = true;
   hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_GEOMETRY, gs);
}

static __attribute__((hot)) struct hk_cs *
hk_flush_gfx_state(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                   struct agx_draw *draw, bool *needs_idx_robust,
                   bool *g17p_native_bind_tail_emitted,
                   const struct agx_draw *conditional_direct)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   const bool perf_trace = hk_shader_perf_draw_enabled(dev);
   static uint32_t perf_flush_calls;
   const uint32_t perf_call = dev->g17p_shader_perf.trace
      ? p_atomic_inc_return(&perf_flush_calls) : 0;
   uint64_t phase_start = perf_trace ? os_time_get_nano() : 0;
   uint64_t stream_ns = 0, preprocess_ns = 0, shaders_ns = 0;
   uint64_t bookkeeping_ns = 0, dynamic_ns = 0;

   /* Every stage hk_flush_shaders() reads has to actually be bound.  Binding
    * VK_NULL_HANDLE for a stage is how CmdBindShadersEXT *unbinds* it, and it
    * is also what a client gets when we refuse to compile that stage's shader
    * -- so this is the point where a compiler refusal would otherwise walk
    * into the draw path.
    *
    * The fragment stage already has an answer: hk_cmd_bind_graphics_shader()
    * substitutes dev->null_fs for a null bind.  Apply it to a stage that was
    * never bound at all, so the substitution does not depend on the client
    * having called CmdBindShadersEXT for it.
    *
    * The vertex stage has no such substitute, and hk_bound_hw_vs() indexes
    * ->variants[] off the bound API shader: with none bound that is a wild
    * pointer a few bytes above zero, which hk_flush_shaders() dereferences
    * for its UVS layout and then publishes to the VDM.  Fail the draw closed
    * instead.
    */
   if (gfx->shaders[MESA_SHADER_FRAGMENT] == NULL)
      hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_FRAGMENT, NULL);

   if (gfx->shaders[MESA_SHADER_VERTEX] == NULL) {
      mesa_loge("draw with no vertex shader bound; dropping it");
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return NULL;
   }

   /* Policies that do not depend on what this draw will publish are decided
    * here, before the stream is chosen, so the whole of the per-stream setup
    * below runs on the stream the draw is actually recorded into.  Ending the
    * stream reuses the established retained-render transition, so the
    * continuation loads the tilebuffer through the partial BG program exactly
    * as a mid-render-pass split already did. */
   const enum hk_g17p_draw_split split_policy = hk_g17p_draw_split_policy(dev);
   const unsigned draws_per_stream = hk_g17p_draws_per_stream(dev);

   if (cmd->current_cs.gfx && cmd->current_cs.gfx->stats.cmds > 0) {
      const unsigned recorded = cmd->current_cs.gfx->stats.cmds;

      if (split_policy == HK_G17P_DRAW_SPLIT_EVERY) {
         hk_g17p_note_gfx_split(cmd, "one-draw-per-stream", recorded, 0);
         hk_cmd_buffer_end_graphics(cmd);
      } else if (draws_per_stream && recorded >= draws_per_stream) {
         hk_g17p_note_gfx_split(cmd, "draws-per-stream", recorded, 0);
         hk_cmd_buffer_end_graphics(cmd);
      }
   }

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   if (!cs)
      return NULL;

   /* Annoyingly,
    * VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT is
    * render pass state on Imaginapple but draw state in Vulkan. In practice,
    * Proton never changes it within a render pass, but we technically need to
    * handle the switch regardless. Do so early since `cs` will be invalidated
    * if we need to split the render pass to switch representation mid-frame.
    *
    * Note we only do this dance with depth bias is actually enabled to avoid
    * senseless control stream splits with DXVK.
    */
   /* A split later in this function may need to reapply unchanged API state
    * to a fresh render, including the opt-in publication-depth diagnostic. */
   const bool dbias_selected = dyn->rs.depth_bias.enable;
   const bool dbias_is_int =
      dbias_selected &&
      ((dyn->rs.depth_bias.representation ==
        VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT) ||
       (hk_depth_stencil_plane_format(gfx->render.depth_att.vk_format, 0) ==
        VK_FORMAT_D16_UNORM));

   if (dbias_selected) {
      /* Attempt to set dbias_is_int per the draw requirement. If this fails,
       * flush the control stream and set it on the new control stream.
       */
      bool succ = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
      if (!succ) {
         perf_debug(cmd, "Splitting control stream due to depth bias");

         hk_g17p_note_gfx_split(cmd, "depth-bias-representation",
                                cs->stats.cmds, cs->end - cs->current);
         hk_cmd_buffer_end_graphics(cmd);
         cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
         if (!cs)
            return NULL;

         succ = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(succ && "can always set tri-state on a new control stream");
      }
   }

   hk_ensure_cs_has_space(cmd, cs, HK_GFX_DRAW_RESERVATION);
   if (vk_command_buffer_has_error(&cmd->vk))
      return NULL;

   if (perf_trace) {
      stream_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

#ifndef NDEBUG
   if (unlikely(dev->dev.debug & AGX_DBG_DIRTY)) {
      hk_cmd_buffer_dirty_all(cmd);
   }
#endif

   /* Merge tess info before GS construction since that depends on
    * gfx->tess.prim
    */
   if ((IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL)) &&
       gfx->shaders[MESA_SHADER_TESS_CTRL]) {
      struct hk_api_shader *tcs = gfx->shaders[MESA_SHADER_TESS_CTRL];
      struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
      struct hk_shader *tese = hk_any_variant(tes);
      struct hk_shader *tesc = hk_only_variant(tcs);

      gfx->tess.info =
         hk_tess_info_merge(tese->info.tess.info, tesc->info.tess.info);

      /* Determine primitive based on the merged state */
      if (gfx->tess.info.points) {
         gfx->tess.prim = MESA_PRIM_POINTS;
      } else if (gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES) {
         gfx->tess.prim = MESA_PRIM_LINES;
      } else {
         gfx->tess.prim = MESA_PRIM_TRIANGLES;
      }
   }

   /* TODO: Try to reduce draw overhead of this */
   hk_handle_passthrough_gs(cmd, *draw);
   if (vk_command_buffer_has_error(&cmd->vk))
      return NULL;

   if (perf_trace) {
      preprocess_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   hk_flush_shaders(cmd);

   if (perf_trace) {
      shaders_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   const bool geom = gfx->shaders[MESA_SHADER_GEOMETRY];
   const bool tess = gfx->shaders[MESA_SHADER_TESS_EVAL];
   bool adj = mesa_prim_has_adjacency(
      vk_topology_to_mesa(dyn->ia.primitive_topology));
   adj &= !geom;
   *needs_idx_robust =
      hk_needs_index_robustness(cmd, draw, conditional_direct);
   /* The G17P robust-index producer emits the same native HAL200 indexed
    * body as an ordinary draw, only after a pre-graphics compute pass has
    * bounded the index range.  It therefore needs the ordinary render-state
    * publication path as well; the generated body carries the matching
    * context-relative selector when that publication is fresh. */
   const bool g17p_robust_index_path =
      dev->dev.chip == AGX_CHIP_G17P && *needs_idx_robust;
   const bool ordinary_draw_path =
      !geom && !tess && !adj &&
      (!*needs_idx_robust || g17p_robust_index_path);
   /* Geometry and tessellation draw their compute-generated vertices through
    * the bound hardware vertex program: poly's geometry raster variant, or
    * the evaluation shader after the software or hardware tessellator. It
    * publishes its render state exactly like an ordinary draw's. */
   /* An adjacency topology with no application geometry shader keeps the
    * hardware vertex program and only makes it fetch its own indices, so it
    * publishes render state exactly as a geometry or tessellation draw does.
    * Excluding it left hk_flush_dynamic_state() refusing the draw, and every
    * adjacency topology failed at vkEndCommandBuffer with
    * VK_ERROR_FEATURE_NOT_PRESENT. */
   const bool g17p_transformed_path =
      dev->dev.chip == AGX_CHIP_G17P && (geom || tess || adj);
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   /* HAL200's indexed draw packet carries its own native restart-state
    * record. Geometry/XFB lowers even a non-indexed API draw to that packet,
    * so emitting the legacy state here as well produces two consecutive
    * restart records before the synthetic indexed draw. */
   if ((draw->restart || geom) && dev->dev.chip != AGX_CHIP_G17P)
      hk_flush_index(cmd, cs);

   /* Everything that can dirty render state for this draw has now run, so
    * this is the first point at which "does this draw republish state?" is
    * answerable, and -- because hk_flush_index() is not on the G17P path --
    * the last point at which nothing of this draw has been written to the
    * stream yet.  Both halves are required: the republication is the hazard,
    * and a split has to happen before the record it protects. */
   const bool republishes = hk_gfx_republishes_render_state(cmd);

   const unsigned publication_cap = hk_g17p_state_publications_per_stream(dev);
   const bool publication_depth_reached =
      publication_cap && cs->g17p_state_publications >= publication_cap;

   if (republishes && cs->stats.cmds > 0 &&
       (split_policy == HK_G17P_DRAW_SPLIT_REPUBLISH ||
        publication_depth_reached)) {
      hk_g17p_note_gfx_split(cmd, "state-publication-depth", cs->stats.cmds,
                             cs->end - cs->current);
      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
      if (!cs)
         return NULL;

      if (dbias_selected) {
         ASSERTED bool succ = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(succ && "can always set tri-state on a new control stream");
      }

      hk_ensure_cs_has_space(cmd, cs, HK_GFX_DRAW_RESERVATION);
   }

   if (perf_trace) {
      bookkeeping_ns = os_time_get_nano() - phase_start;
      phase_start = os_time_get_nano();
   }

   hk_flush_dynamic_state(cmd, cs, draw_id, *draw,
                          ordinary_draw_path || g17p_transformed_path,
                          g17p_native_bind_tail_emitted, false);
   if (perf_trace) {
      dynamic_ns = os_time_get_nano() - phase_start;
      hk_shader_perf_flush_detail_record(stream_ns, preprocess_ns, shaders_ns,
                                         bookkeeping_ns, dynamic_ns);
      if (dev->g17p_shader_perf.trace && perf_call <= 3) {
         fprintf(stderr,
                 "HK_SHADER_PERF flush-call=%u stream_ns=%" PRIu64
                 " preprocess_ns=%" PRIu64 " shaders_ns=%" PRIu64
                 " bookkeeping_ns=%" PRIu64 " dynamic_ns=%" PRIu64 "\n",
                 perf_call, stream_ns, preprocess_ns, shaders_ns,
                 bookkeeping_ns, dynamic_ns);
      }
   }
   return cs;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindIndexBuffer2KHR(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, VkDeviceSize size,
                          VkIndexType indexType)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   cmd->state.gfx.index = (struct hk_index_buffer_state){
      .buffer = hk_buffer_addr_range(buffer, offset, size, true),
      .size = agx_translate_index_size(vk_index_type_to_bytes(indexType)),
      .restart = vk_index_to_restart(indexType),
   };

   /* TODO: check if necessary, blob does this */
   cmd->state.gfx.index.buffer.range =
      align(cmd->state.gfx.index.buffer.range, 4);

   /* Binding supplies the last draw-specific VS key component. This is a
    * bounded cache warm-up only when a render is active; it emits no GPU
    * state and never clears Vulkan dirtiness. */
   if (cmd->current_cs.gfx) {
      hk_prelink_graphics(
         cmd, true, cmd->state.gfx.index.size,
         cmd->vk.dynamic_graphics_state.ia.primitive_restart_enable, true);

      /* Software tessellation/geometry runs the API vertex stage as compute
       * before rasterization.  The index width is part of that stage's exact
       * fast-link identity, so the non-indexed dispatch prepared at
       * BeginRendering cannot be consumed after this bind.  Rebuild only the
       * affected immutable dispatch template here, while the application is
       * already establishing indexed state, instead of falling back to the
       * general encoder inside the first indexed draw. */
      struct hk_device *dev = hk_cmd_buffer_device(cmd);
      struct hk_graphics_state *gfx = &cmd->state.gfx;
      if (dev->dev.chip == AGX_CHIP_G17P &&
          !vk_command_buffer_has_error(&cmd->vk)) {
         /* The indexed fast link also changes the vertex identity captured by
          * the immutable HAL200 tail.  Refresh it alongside the dispatch so
          * varying linkage and tail construction stay out of the first draw. */
         hk_g17p_prepare_native_bind_tail(cmd);

         if (gfx->shaders[MESA_SHADER_TESS_EVAL]) {
            struct hk_shader *vs = hk_bound_sw_vs(gfx);
            struct hk_linked_shader *linked =
               vs->only_linked ?: gfx->linked[MESA_SHADER_VERTEX];

            hk_g17p_prepare_graphics_dispatch(
               cmd, vs, linked, 0, agx_workgroup(1, 1, 1),
               &gfx->g17p_dispatch.tess_vs);
         } else if (gfx->shaders[MESA_SHADER_GEOMETRY]) {
            struct hk_shader *vs = hk_bound_sw_vs_before_gs(gfx);
            struct hk_linked_shader *linked =
               vs->only_linked ?: gfx->linked[MESA_SHADER_VERTEX];
            const uint64_t grid = gfx->prepared_vertex_params.gpu +
               offsetof(struct poly_vertex_params, grid);
            hk_g17p_prepare_graphics_dispatch(
               cmd, vs, linked, grid, agx_workgroup(64, 1, 1),
               &gfx->g17p_dispatch.gs_vs);
         }
      }
   }

}

void
hk_cmd_bind_vertex_buffer(struct hk_cmd_buffer *cmd, uint32_t vb_idx,
                          struct hk_addr_range addr_range)
{
   cmd->state.gfx.vb[vb_idx] = addr_range;
   cmd->state.gfx.dirty |= HK_DIRTY_VB;
   cmd->state.gfx.vertex_buffer_generation++;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                         uint32_t bindingCount, const VkBuffer *pBuffers,
                         const VkDeviceSize *pOffsets,
                         const VkDeviceSize *pSizes,
                         const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pStrides) {
      vk_cmd_set_vertex_binding_strides(&cmd->vk, firstBinding, bindingCount,
                                        pStrides);
   }

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;

      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;
      const struct hk_addr_range addr_range =
         hk_buffer_addr_range(buffer, pOffsets[i], size, true);

      hk_cmd_bind_vertex_buffer(cmd, idx, addr_range);
   }
}

static bool
hk_set_view_index(struct hk_cmd_buffer *cmd, uint32_t view_idx)
{
   if (cmd->state.gfx.render.view_mask) {
      cmd->state.gfx.descriptors.root.draw.view_index = view_idx;
      cmd->state.gfx.descriptors.root_dirty = true;
   }

   return true;
}

/*
 * Iterator macro to duplicate a draw for each enabled view (when multiview is
 * enabled, else always view 0). Along with hk_lower_multiview, this forms the
 * world's worst multiview lowering.
 */
#define hk_foreach_view(cmd)                                                   \
   u_foreach_bit(view_idx, cmd->state.gfx.render.view_mask ?: 1)               \
      if (hk_set_view_index(cmd, view_idx))

static void
hk_ia_update(struct hk_cmd_buffer *cmd, struct agx_draw draw,
             uint64_t ia_vertices, uint64_t ia_prims, uint64_t vs_invocations,
             uint64_t c_prims, uint64_t c_inv)
{
   perf_debug(cmd, "Input assembly counters");

   uint64_t draw_ptr;
   if (agx_is_indirect(draw.b)) {
      draw_ptr = draw.b.ptr;
   } else {
      uint32_t desc[] = {draw.b.count[0], draw.b.count[1], 0};
      draw_ptr = hk_pool_upload(cmd, &desc, sizeof(desc), 4);
   }

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

   bool geom = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
   bool tess = cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];

   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Clipper counters depend on geom/tess outputs and must be written with the
    * geom/tess output. They are updated as IA counters only when geom/tess is
    * not used. G17P represents a counter that is not written by the writable
    * scratch page, like a disabled one.
    */
   if (geom || tess) {
      c_prims = dev->dev.chip == AGX_CHIP_G17P ? AGX_SCRATCH_PAGE_ADDRESS : 0;
      c_inv = c_prims;
   }

   uint32_t patch_size = dyn->ts.patch_control_points;

   if (draw.restart) {
      uint32_t index_size_B = agx_index_size_to_B(draw.index_size);

      if (dev->dev.chip == AGX_CHIP_G17P) {
         /* Disabled statistics are represented by the writable scratch page,
          * so all five addresses are valid. Use the native specialization
          * that does not need to reduce nullable 64-bit pointers. */
         libagx_increment_ia_restart_nonnull(
            cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX, ia_vertices,
            ia_prims, vs_invocations, c_prims, c_inv, draw_ptr,
            draw.index_buffer, agx_draw_index_range_el(draw),
            cmd->state.gfx.index.restart, index_size_B, prim, patch_size);
      } else {
         libagx_increment_ia_restart(
            cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX, ia_vertices,
            ia_prims, vs_invocations, c_prims, c_inv, draw_ptr,
            draw.index_buffer, agx_draw_index_range_el(draw),
            cmd->state.gfx.index.restart, index_size_B, prim, patch_size);
      }
   } else {
      libagx_increment_ia(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                          ia_vertices, ia_prims, vs_invocations, c_prims, c_inv,
                          draw_ptr, prim, patch_size);
   }
}

/*
 * Whether this draw's tessellation runs on the G17P hardware tessellator.
 *
 * The decision is per pipeline, not per device: Apple's state record has no
 * isoline and no point-mode encoding, so an evaluation shader declaring
 * either keeps the software tessellator even with HK_G17P_NATIVE_TESS=1, and
 * hk_compile_nir() compiled it that way. Keeping the two in one predicate is
 * what makes the fallback safe -- the bound evaluation variant and the draw
 * path always agree on which tessellator produced the coordinates.
 */
static bool
hk_g17p_tess_is_native(const struct hk_cmd_buffer *cmd)
{
   const struct hk_device *dev = hk_cmd_buffer_device((struct hk_cmd_buffer *)cmd);
   struct hk_api_shader *tes = cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];

   return dev->g17p_native_tess && tes &&
          hk_any_variant(tes)->info.tess.g17p_native_domain;
}

/* The G17P draw record after any state publication: the context-relative
 * draw-state selector record, then the HAL200 draw body (indirect, indexed,
 * direct or tessellated).  Shared by hk_draw_impl() and
 * hk_g17p_emit_indirect_record().  Returns false with the command buffer
 * error set when the draw cannot be encoded. */
static bool
hk_g17p_emit_draw(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t *out,
                  struct agx_draw draw, enum agx_primitive topology,
                  bool native_tess, bool tail_emitted)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* This context-relative PPP record selects the latest scissor
    * and depth-bias array entries. Header0x100 is PPP selector bit8,
    * not a direct-draw shape; +4 contains indices, not enable flags.
    * Native bind-state emission appended those owned records above.
    */
   struct agx_ptr draw_state = hk_pool_vdm_alloc(cmd, 8, 4);
   uint32_t draw_state_offset;
   if (!draw_state.cpu ||
       !hk_g17p_upload_draw_indices(
          draw_state.cpu, 8, draw_state.gpu, cs->scissor.size,
          cs->depth_bias.size,
          MAX2(cmd->vk.dynamic_graphics_state.vp.viewport_count, 1),
          &draw_state_offset)) {
      vk_command_buffer_set_error(&cmd->vk,
                                  VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }
   if (native_tess) {
      if (!hk_g17p_emit_tessellate(cmd, cs, out, draw_state_offset,
                                   tail_emitted))
         return false;
   } else if (agx_is_indirect(draw.b)) {
      const struct agx_vdm_hal200_indirect_draw_fields fields = {
         .draw_state_context_offset = draw_state_offset,
         .primitive = (uint8_t)topology,
         .indexed = draw.indexed,
         .index_size_B =
            draw.indexed ? agx_index_size_to_B(draw.index_size) : 0,
         .restart_enable = draw.restart,
         /* Geometry/XFB lowers a non-indexed API draw to a synthetic
          * indexed draw.  Derive the fixed Vulkan restart value from
          * that draw, not from the original bound-index state. */
         .restart_index = draw.indexed
                             ? agx_primitive_restart_index(
                                  draw.index_size)
                             : 0,
         .index_buffer = draw.index_buffer,
         .index_buffer_range_B = draw.index_buffer_range_B,
         /* Unlike PPP and stream-link pointers, the Index List
          * indirect-argument pointer is a native absolute 48-bit VA.
          * Compacting it by the render-context base makes hardware
          * read an unmapped address and silently skip the draw. */
         .indirect_buffer = draw.b.ptr,
      };
      struct agx_vdm_hal200_indirect_draw encoded;
      if (!agx_vdm_hal200_indirect_draw_pack(dev->dev.chip, &fields,
                                              &encoded)) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }
      /* The selector completes the native bind tail. With unchanged
       * state, indirect draws need the same body-only framing as
       * direct draws; a bare selector would be parsed as a header. */
      size_t emitted;
      if (!agx_vdm_hal200_draw_emit(
             dev->dev.chip, encoded.bytes, encoded.length,
             tail_emitted, out,
             (uint8_t *)cs->end - out, &emitted)) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }
      cs->current = out + emitted;
      if (hk_g17p_launch_trace_enabled())
         hk_g17p_trace_line(draw.indexed
                               ? "hal200-indexed-indirect-draw"
                               : "hal200-indirect-draw",
                            draw.b.ptr, out, emitted);
   } else if (draw.indexed) {
      const struct agx_vdm_hal200_indexed_draw_fields fields = {
         .primitive = (uint8_t)topology,
         .index_size_B = agx_index_size_to_B(draw.index_size),
         .restart_enable = draw.restart,
         .restart_index =
            agx_primitive_restart_index(draw.index_size),
         .draw_state_context_offset = draw_state_offset,
         .index_buffer = draw.index_buffer,
         .index_buffer_range_B = draw.index_buffer_range_B,
         .first_index = draw.start,
         .index_count = draw.b.count[0],
         .instance_count = draw.b.count[1],
         .base_vertex = (int32_t)draw.index_bias,
      };
      struct agx_vdm_hal200_indexed_draw encoded;
      if (!agx_vdm_hal200_indexed_draw_pack(dev->dev.chip, &fields,
                                             &encoded)) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }
      /* Metal's indexed entry point emits the 36-byte draw body
       * after encodeAndEmitRenderState.  Its leading four-byte PPP
       * selector is emitted only with a fresh native bind tail. */
      size_t emitted;
      if (!agx_vdm_hal200_draw_emit(
             dev->dev.chip, encoded.bytes, encoded.length,
             tail_emitted, out,
             (uint8_t *)cs->end - out,
             &emitted)) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }
      cs->current = out + emitted;
      if (hk_g17p_launch_trace_enabled())
         hk_g17p_trace_line("hal200-indexed-draw", draw.index_buffer,
                            out, emitted);
   } else {
      const struct agx_vdm_hal200_direct_draw_fields fields = {
         .draw_state_context_offset = draw_state_offset,
         .primitive = (uint8_t)topology,
         .vertex_count = draw.b.count[0],
         .instance_count = draw.b.count[1],
         .vertex_start = draw.start,
      };
      struct agx_vdm_hal200_direct_draw encoded;
      if (!agx_vdm_hal200_direct_draw_pack(dev->dev.chip, &fields,
                                            &encoded)) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }

      /* The native non-indexed entry point likewise emits a bare
       * 16-byte body when render state is unchanged.  A standalone
       * selector before it makes 0x61c4 part of a bogus VDM address. */
      size_t emitted;
      if (!agx_vdm_hal200_draw_emit(
             dev->dev.chip, encoded.bytes, encoded.length,
             tail_emitted, out,
             (uint8_t *)cs->end - out,
             &emitted)) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }
      cs->current = out + emitted;
      if (hk_g17p_launch_trace_enabled())
         hk_g17p_trace_line("hal200-direct-draw", 0, out, emitted);
   }
   if (hk_g17p_launch_trace_enabled())
      hk_g17p_trace_line("hal200-draw-state",
                         draw_state.gpu, draw_state.cpu, 8);

   return true;
}

static __attribute__((hot)) void
hk_draw_impl(struct hk_cmd_buffer *cmd, uint16_t draw_id,
             struct agx_draw draw_)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   /* Filter trivial draws so we don't need to worry about null index buffers */
   if (!agx_is_indirect(draw_.b) &&
       (draw_.b.count[0] == 0 || draw_.b.count[1] == 0))
      return;

   draw_.restart = dyn->ia.primitive_restart_enable && draw_.indexed;
   draw_.index_size = cmd->state.gfx.index.size;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* A tessellated draw with fewer vertices than one patch draws nothing.
    * The hardware tessellator takes whole patches, so filter it with the
    * other trivial draws before any state is published. */
   if (hk_g17p_tess_is_native(cmd) && !agx_is_indirect(draw_.b) &&
       draw_.b.count[0] < dyn->ts.patch_control_points)
      return;

   /* Null index bindings execute the requested vertices, all with index zero.
    * Normalize before native admission and conditional draw lowering, as the
    * older path does in hk_needs_index_robustness. The one-dword range bounds
    * hardware fetches to the zero page; it is not a count limit.
    */
   if (dev->dev.chip == AGX_CHIP_G17P && draw_.indexed &&
       draw_.index_buffer_range_B == 0) {
      draw_.index_buffer = AGX_ZERO_PAGE_ADDRESS;
      draw_.index_buffer_range_B = 4;
      draw_.start = 0;
   }
   const struct agx_draw api_draw = draw_;
   bool conditional_direct_indexed = false;

   /* There is no native conditional-render packet on AGX. Lower the command
    * to the API's indirect record and let an ordered compute kernel either
    * copy it or zero its counts from the execution-time predicate. This also
    * predicates IA/pipeline-stat accounting and every later geometry lowering
    * because they all consume the patched record. */
   if (hk_conditional_render_is_active(&cmd->state.conditional, cmd->in_meta,
                                        cmd->conditional_meta)) {
      uint64_t source = draw_.b.ptr;
      const uint32_t words = draw_.indexed ? 5 : 4;
      conditional_direct_indexed =
         draw_.indexed && !agx_is_indirect(draw_.b);

      if (!agx_is_indirect(draw_.b)) {
         uint32_t record[HK_CONDITIONAL_RENDER_MAX_WORDS];
         const uint32_t built = hk_conditional_render_direct_draw_record(
            record, draw_.indexed, draw_.b.count[0], draw_.b.count[1],
            draw_.start, (int32_t)draw_.index_bias, draw_.start_instance);
         assert(built == words);
         source = hk_pool_upload(cmd, record, words * sizeof(uint32_t), 4);
      }

      const uint64_t patched = hk_apply_conditional_rendering(
         cmd, source, words, AGX_BARRIER_ALL | AGX_PREGFX);
      if (!patched)
         return;

      draw_ = draw_.indexed
                 ? agx_draw_indexed_indirect(
                      patched, draw_.index_buffer,
                      draw_.index_buffer_range_B, draw_.index_size,
                      draw_.restart)
                 : agx_draw_indirect(patched);
   }

   uint64_t stat_ia_verts = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT);

   uint64_t stat_ia_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT);

   uint64_t stat_vs_inv = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT);

   uint64_t stat_c_inv = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);

   uint64_t stat_c_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT);

   bool ia_stats = hk_stat_enabled(stat_ia_verts) ||
                   hk_stat_enabled(stat_ia_prims) ||
                   hk_stat_enabled(stat_vs_inv) ||
                   hk_stat_enabled(stat_c_inv) || hk_stat_enabled(stat_c_prims);
   /* Validate the native index domain before robustness or geometry lowering
    * can introduce a pre-render compute dispatch.  Robust draws keep their
    * application range for the ordered GPU-side bounds pass, which shadows the
    * in-range indices and pads the overread with zero indices; only draws
    * that reach the native Index List unmodified prove their complete span
    * here. */
   /* Geometry and software tessellation run the vertex shader in compute,
    * which fetches indices robustly like on every other generation. */
   const bool software_index_fetch =
      cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] ||
      (cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL] &&
       !hk_g17p_tess_is_native(cmd));
   if (dev->dev.chip == AGX_CHIP_G17P && draw_.indexed &&
       !software_index_fetch) {
      uint64_t address;
      uint32_t size_word;
      const bool valid_index_size =
         draw_.index_size == AGX_INDEX_SIZE_U8 ||
         draw_.index_size == AGX_INDEX_SIZE_U16 ||
         draw_.index_size == AGX_INDEX_SIZE_U32;
      struct agx_draw robust_draw = draw_;
      const bool robust_index = hk_needs_index_robustness(
         cmd, &robust_draw, conditional_direct_indexed ? &api_draw : NULL);
      const bool conditional_direct_in_bounds =
         !conditional_direct_indexed || robust_index ||
         (valid_index_size &&
          agx_vdm_hal200_index_range(
             api_draw.index_buffer, api_draw.index_buffer_range_B,
             api_draw.start, api_draw.b.count[0],
             agx_index_size_to_B(api_draw.index_size), &address, &size_word));
      /* A tessellated draw's vertex shader fetches its indices in compute,
       * but they still have to be in range. */
      if (!valid_index_size ||
          (agx_is_indirect(draw_.b)
              ? (!draw_.b.ptr || !draw_.index_buffer ||
                 draw_.index_buffer_range_B == 0 ||
                 !conditional_direct_in_bounds)
              : !robust_index &&
                   !agx_vdm_hal200_index_range(
                      draw_.index_buffer, draw_.index_buffer_range_B,
                      draw_.start, draw_.b.count[0],
                      agx_index_size_to_B(draw_.index_size), &address,
                      &size_word))) {
         mesa_loge("native G17P indexed draw requires a valid "
                   "uint8/uint16/uint32 range");
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
   }

   hk_foreach_view(cmd) {
      struct agx_draw draw = draw_;
      bool needs_idx_robust = false;
      bool g17p_native_bind_tail_emitted = false;
      const bool perf_trace = hk_shader_perf_draw_enabled(dev);
      const uint64_t flush_start = perf_trace ? os_time_get_nano() : 0;
      struct hk_cs *cs =
         hk_flush_gfx_state(cmd, draw_id, &draw, &needs_idx_robust,
                            &g17p_native_bind_tail_emitted,
                            conditional_direct_indexed ? &api_draw : NULL);
      if (perf_trace) {
         hk_shader_perf_flush_record(
            cmd->state.gfx.shaders[MESA_SHADER_VERTEX],
            os_time_get_nano() - flush_start);
      }
      /* If we failed to allocate a control stream, we've already lost the
       * device. Just drop the draw so we don't crash.
       */
      if (!cs)
         return;
      if (vk_command_buffer_has_error(&cmd->vk))
         return;

      struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
      bool geom = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
      bool tess = cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];
      bool adj = mesa_prim_has_adjacency(
         vk_topology_to_mesa(dyn->ia.primitive_topology));
      adj &= !geom;
      needs_idx_robust &= !adj;

      struct hk_cs *ccs = NULL;
      uint8_t *out = cs->current;
      assert(cs->current + 0x1000 < cs->end);

      if (tess && HK_PERF(dev, NOTESS))
         continue;

      cs->stats.calls++;

      if (geom || tess) {
         const uint64_t phase_start = perf_trace ? os_time_get_nano() : 0;
         ccs =
            hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx, true);
         if (perf_trace) {
            hk_shader_perf_phase_record(
               cmd->state.gfx.shaders[MESA_SHADER_VERTEX], 0,
               os_time_get_nano() - phase_start);
         }
         if (!ccs)
            return;
      }

      if (ia_stats) {
         hk_ia_update(cmd, draw, stat_ia_verts, stat_ia_prims, stat_vs_inv,
                      stat_c_prims, stat_c_inv);
      }

      struct agx_draw prepared_tess_gs = {0};
      if (tess && geom)
         prepared_tess_gs =
            hk_g17p_launch_prepared_sw_tess_gs(cmd, ccs, draw);
      const bool prepared_tess_gs_emitted = prepared_tess_gs.b.ptr != 0;
      if (prepared_tess_gs_emitted)
         draw = prepared_tess_gs;

      if (tess && !prepared_tess_gs_emitted) {
         const uint64_t phase_start = perf_trace ? os_time_get_nano() : 0;
         if (hk_g17p_tess_is_native(cmd)) {
            hk_launch_tess_g17p(cmd, ccs, draw);
            if (vk_command_buffer_has_error(&cmd->vk))
               return;
         } else {
            draw = hk_launch_tess(cmd, ccs, draw, geom ? 0 : stat_c_prims,
                                  geom ? 0 : stat_c_inv);
         }
         if (perf_trace) {
            hk_shader_perf_phase_record(
               cmd->state.gfx.shaders[MESA_SHADER_VERTEX], 1,
               os_time_get_nano() - phase_start);
         }
      }

      if (geom && !prepared_tess_gs_emitted) {
         const uint64_t phase_start = perf_trace ? os_time_get_nano() : 0;
         draw = hk_launch_gs_prerast(cmd, ccs, draw);
         if (perf_trace) {
            hk_shader_perf_phase_record(
               cmd->state.gfx.shaders[MESA_SHADER_VERTEX], 2,
               os_time_get_nano() - phase_start);
         }
      }

      if (adj) {
         assert(!geom && "geometry shaders handle adj directly");
         enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

         if (hk_g17p_launch_trace_enabled()) {
            fprintf(stderr,
                    "G17P adj-enter: prim=%u restart=%u indexed=%u "
                    "indirect=%u index_size=%u count=%u\n",
                    (unsigned)prim, draw.restart, draw.indexed,
                    agx_is_indirect(draw.b), (unsigned)draw.index_size,
                    agx_is_indirect(draw.b) ? 0u : draw.b.count[0]);
         }

         if (draw.restart) {
            draw = hk_draw_without_restart(cmd, draw, 1);
            prim = u_decomposed_prim(prim);

            if (hk_g17p_launch_trace_enabled()) {
               fprintf(stderr,
                       "G17P adj-after-unroll: prim=%u indirect=%u "
                       "index_size=%u\n",
                       (unsigned)prim, agx_is_indirect(draw.b),
                       (unsigned)draw.index_size);
            }
         }

         if (agx_is_indirect(draw.b)) {
            const size_t size = sizeof(VkDrawIndexedIndirectCommand);
            static_assert(sizeof(VkDrawIndexedIndirectCommand) >
                             sizeof(VkDrawIndirectCommand),
                          "allocation size is conservative");

            uint64_t out_draw = hk_pool_alloc(cmd, size, 4).gpu;
            struct hk_device *dev = hk_cmd_buffer_device(cmd);

            if (dev->dev.chip == AGX_CHIP_G17P) {
               /* As in the direct case above, a G17P hardware vertex shader
                * never sees the prolog's remap, so the mapping has to reach
                * the input assembler as real indices.  The output count is
                * only known once the GPU has read the indirect command, so
                * the kernel allocates the index buffer from the heap and
                * names it in the indexed indirect draw it writes. */
               if (hk_g17p_launch_trace_enabled()) {
                  fprintf(stderr,
                          "G17P adj-indirect-kernel: prim=%u in_draw=%llx "
                          "out_draw=%llx ib=%llx range_el=%u elsize=%u\n",
                          (unsigned)prim,
                          (unsigned long long)draw.b.ptr,
                          (unsigned long long)out_draw,
                          (unsigned long long)(draw.indexed ? draw.index_buffer
                                                            : 0),
                          draw.indexed ? agx_draw_index_range_el(draw) : 0,
                          draw.indexed
                             ? agx_index_size_to_B(draw.index_size)
                             : 0);
               }

               libagx_write_adj_indices_indirect(
                  cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                  hk_heap(cmd), draw.b.ptr, out_draw,
                  draw.indexed ? draw.index_buffer : 0,
                  draw.indexed ? agx_draw_index_range_el(draw) : 0,
                  draw.indexed ? agx_index_size_to_B(draw.index_size) : 0,
                  prim);

               draw = agx_draw_indexed_indirect(out_draw, dev->heap->va->addr,
                                                dev->heap->size,
                                                AGX_INDEX_SIZE_U32, false);
            } else {
               struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

               libagx_draw_without_adj(
                  cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, out_draw,
                  draw.b.ptr, desc->root.draw.vertex_params, draw.index_buffer,
                  draw.indexed ? agx_draw_index_range_el(draw) : 0,
                  draw.indexed ? agx_index_size_to_B(draw.index_size) : 0,
                  prim);

               draw = agx_draw_indirect(out_draw);
            }
         } else {
            unsigned count = libagx_remap_adj_count(draw.b.count[0], prim);
            struct hk_device *dev = hk_cmd_buffer_device(cmd);

            /* Software indexing hands the remap to the VS prolog, which a G17P
             * hardware vertex shader never observes: it reads its invocation
             * index straight from SR 0xdd rather than the
             * AGX_ABI_VIN_VERTEX_ID the prolog exports, so every adjacency
             * draw fetched in[i] where it owed in[map(i)].  Materialise the
             * mapping as an index buffer instead and let the input assembler
             * apply it, which is the one path the vertex stage does see. */
            if (count > 0 && dev->dev.chip == AGX_CHIP_G17P) {
               unsigned ib_B = count * sizeof(uint32_t);
               uint64_t ib = hk_pool_alloc(cmd, ib_B, 4).gpu;

               /* Take the index buffer straight off the draw rather than out
                * of poly_vertex_params, which is only uploaded when the linked
                * vertex shader asked for software indexing. */
               libagx_write_adj_indices(
                  cmd, agx_1d(count), AGX_BARRIER_ALL | AGX_PREGFX, ib,
                  draw.indexed ? agx_draw_index_buffer(draw) : 0,
                  draw.indexed ? agx_draw_index_range_el(draw) : 0, count,
                  draw.indexed ? agx_index_size_to_B(draw.index_size) : 0,
                  draw.indexed ? draw.index_bias : draw.start, prim);

               draw = agx_draw_indexed(count, draw.b.count[1], 0, 0,
                                       draw.start_instance, ib, ib_B,
                                       AGX_INDEX_SIZE_U32, false);
            } else {
               draw = (struct agx_draw){
                  .b = agx_3d(count, draw.b.count[1], 1),
               };
            }
         }
      }

      /* Geometry, tessellation and adjacency lowering can turn a nonempty API
       * draw into a direct raster draw with no complete output primitives.
       * There is no draw packet to encode in that case. The pre-raster work,
       * including any shader side effects, has already been recorded above. */
      if (!agx_is_indirect(draw.b) &&
          (draw.b.count[0] == 0 || draw.b.count[1] == 0)) {
         /* The bind tail was already published before lowering determined
          * there were no complete primitives. Its selector word must still
          * be present, even though no Index List follows it. */
         if (dev->dev.chip == AGX_CHIP_G17P && g17p_native_bind_tail_emitted) {
            struct agx_ptr state = hk_pool_vdm_alloc(cmd, 8, 4);
            uint32_t offset;
            if (!state.cpu || !hk_g17p_upload_draw_indices(
                   state.cpu, 8, state.gpu, cs->scissor.size,
                   cs->depth_bias.size, MAX2(dyn->vp.viewport_count, 1),
                   &offset)) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }
            memcpy(out, &offset, sizeof(offset));
            out += sizeof(offset);
            cs->current = out;
         }
         continue;
      }

      enum agx_primitive topology = cmd->state.gfx.topology;
      const bool native_tess = tess && hk_g17p_tess_is_native(cmd);
      if (!native_tess &&
          !agx_vdm_draw_supported(dev->dev.chip, draw, topology)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      if (hk_g17p_draw_trace_enabled() && dev->dev.chip == AGX_CHIP_G17P) {
         cs->g17p_draw_trace.draw_id = draw_id;
         cs->g17p_draw_trace.topology = topology;
         cs->g17p_draw_trace.api_indexed = api_draw.indexed;
         cs->g17p_draw_trace.api_indirect = agx_is_indirect(api_draw.b);
         cs->g17p_draw_trace.api_count =
            agx_is_indirect(api_draw.b) ? 0 : api_draw.b.count[0];
         cs->g17p_draw_trace.api_instances =
            agx_is_indirect(api_draw.b) ? 0 : api_draw.b.count[1];
         cs->g17p_draw_trace.api_first =
            agx_is_indirect(api_draw.b) ? 0 : api_draw.start;
         cs->g17p_draw_trace.api_base_vertex =
            agx_is_indirect(api_draw.b) ? 0 : api_draw.index_bias;
         cs->g17p_draw_trace.api_first_instance =
            agx_is_indirect(api_draw.b) ? 0 : api_draw.start_instance;
         cs->g17p_draw_trace.emitted_indexed = draw.indexed;
         cs->g17p_draw_trace.emitted_indirect = agx_is_indirect(draw.b);
         cs->g17p_draw_trace.emitted_indirect_buffer =
            agx_is_indirect(draw.b) ? draw.b.ptr : 0;
      }

      if (needs_idx_robust) {
         assert(!geom && !tess && !adj);
         perf_debug(cmd, "lowering robust index buffer");

         cs->current = out;

         draw = hk_draw_as_indexed_indirect(cmd, draw);
         if (!agx_vdm_draw_supported(dev->dev.chip, draw, topology)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         if (hk_g17p_draw_trace_enabled() &&
             dev->dev.chip == AGX_CHIP_G17P) {
            cs->g17p_draw_trace.emitted_indexed = draw.indexed;
            cs->g17p_draw_trace.emitted_indirect = true;
            cs->g17p_draw_trace.emitted_indirect_buffer = draw.b.ptr;
         }

         uint32_t draw_state_offset = 0;
         if (dev->dev.chip == AGX_CHIP_G17P &&
             g17p_native_bind_tail_emitted) {
            struct agx_ptr draw_state = hk_pool_vdm_alloc(cmd, 8, 4);
            if (!draw_state.cpu ||
                !hk_g17p_upload_draw_indices(
                   draw_state.cpu, 8, draw_state.gpu, cs->scissor.size,
                   cs->depth_bias.size,
                   MAX2(cmd->vk.dynamic_graphics_state.vp.viewport_count, 1),
                   &draw_state_offset)) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }
         }

         size_t size_B = libagx_draw_robust_index_vdm_size(
            dev->dev.chip, g17p_native_bind_tail_emitted);
         uint64_t target = hk_cs_alloc_for_indirect(cs, size_B);

         uint64_t empty_draw_jump = 0;
         if (dev->dev.chip == AGX_CHIP_G17P) {
            const uint64_t compact =
               hk_g17p_vdm_compact(cmd, target + size_B);
            if (!compact || compact >= (UINT64_C(1) << 40)) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_OUT_OF_DEVICE_MEMORY);
               return;
            }
            empty_draw_jump = compact;
         }

         libagx_draw_robust_index(cmd, agx_1d(32), AGX_BARRIER_ALL | AGX_PREGFX,
                                  target, hk_heap(cmd), draw.b.ptr,
                                  draw.index_buffer, draw.index_buffer_range_B,
                                  draw.restart, dev->dev.chip, topology,
                                  draw_state_offset,
                                  g17p_native_bind_tail_emitted,
                                  empty_draw_jump,
                                  draw.index_size);
      } else {
         if (dev->dev.chip == AGX_CHIP_G17P) {
            if (!hk_g17p_emit_draw(cmd, cs, out, draw, topology, native_tess,
                                   g17p_native_bind_tail_emitted))
               return;
         } else {
            cs->current = (void *)agx_vdm_draw((uint32_t *)out,
                                               dev->dev.chip, draw, topology);
         }
      }

      cs->stats.cmds++;
   }
}

static __attribute__((hot)) void
hk_draw(struct hk_cmd_buffer *cmd, uint16_t draw_id, struct agx_draw draw)
{
   static uint32_t trace_draws;
   static bool trace_snapshot_dumped;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const bool trace = hk_shader_perf_draw_enabled(dev);
   const bool detail = dev->g17p_shader_perf.trace;
   const uint64_t start = trace ? os_time_get_nano() : 0;
   const void *shader = cmd->state.gfx.shaders[MESA_SHADER_VERTEX];
   struct rusage usage_start = {0}, usage_end = {0};

   if (trace)
      hk_shader_perf_detail_reset();

   if (trace) {
      cmd->shader_perf_draw_bo_allocs = 0;
      cmd->shader_perf_draw_bo_creates = 0;
      cmd->shader_perf_draw_bo_alloc_ns = 0;
      cmd->shader_perf_draw_active = true;
   }
   if (detail) {
      getrusage(RUSAGE_SELF, &usage_start);
   }

   hk_draw_impl(cmd, draw_id, draw);

   if (trace) {
      if (detail) {
         getrusage(RUSAGE_SELF, &usage_end);
      }
      cmd->shader_perf_draw_active = false;
      hk_shader_perf_draw_record(shader, os_time_get_nano() - start,
                                 cmd->shader_perf_draw_bo_allocs,
                                 cmd->shader_perf_draw_bo_creates,
                                 cmd->shader_perf_draw_bo_alloc_ns,
                                 detail ? usage_end.ru_minflt - usage_start.ru_minflt : 0,
                                 detail ? usage_end.ru_majflt - usage_start.ru_majflt : 0);
      if (detail && !trace_snapshot_dumped && ++trace_draws == 16384) {
         trace_snapshot_dumped = true;
         fprintf(stderr, "HK_SHADER_PERF snapshot_draws=%u\n", trace_draws);
         hk_shader_perf_draw_dump();
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount,
           uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct agx_draw draw;

   if (HK_TEST_INDIRECTS) {
      uint32_t data[] = {
         vertexCount,
         instanceCount,
         firstVertex,
         firstInstance,
      };

      draw = agx_draw_indirect(hk_pool_upload(cmd, data, sizeof(data), 4));
   } else {
      draw = (struct agx_draw){
         .b = agx_3d(vertexCount, instanceCount, 1),
         .start = firstVertex,
         .start_instance = firstInstance,
      };
   }

   hk_draw(cmd, 0, draw);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                   const VkMultiDrawInfoEXT *pVertexInfo,
                   uint32_t instanceCount, uint32_t firstInstance,
                   uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   for (unsigned i = 0; i < drawCount; ++i) {
      struct agx_draw draw = {
         .b = agx_3d(pVertexInfo->vertexCount, instanceCount, 1),
         .start = pVertexInfo->firstVertex,
         .start_instance = firstInstance,
      };

      hk_draw(cmd, i, draw);
      pVertexInfo = ((void *)pVertexInfo) + stride;
   }
}

static void
hk_draw_indexed(VkCommandBuffer commandBuffer, uint16_t draw_id,
                uint32_t indexCount, uint32_t instanceCount,
                uint32_t firstIndex, int32_t vertexOffset,
                uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct agx_draw draw;
   struct hk_addr_range buf = cmd->state.gfx.index.buffer;

   if (HK_TEST_INDIRECTS && draw_id == 0) {
      uint32_t data[] = {
         indexCount, instanceCount, firstIndex, vertexOffset, firstInstance,
      };
      uint64_t addr = hk_pool_upload(cmd, data, sizeof(data), 4);

      draw = agx_draw_indexed_indirect(addr, buf.addr, buf.range, 0, 0);
   } else {
      draw =
         agx_draw_indexed(indexCount, instanceCount, firstIndex, vertexOffset,
                          firstInstance, buf.addr, buf.range, 0, 0);
   }

   hk_draw(cmd, draw_id, draw);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                  uint32_t instanceCount, uint32_t firstIndex,
                  int32_t vertexOffset, uint32_t firstInstance)
{
   hk_draw_indexed(commandBuffer, 0, indexCount, instanceCount, firstIndex,
                   vertexOffset, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                          const VkMultiDrawIndexedInfoEXT *pIndexInfo,
                          uint32_t instanceCount, uint32_t firstInstance,
                          uint32_t stride, const int32_t *pVertexOffset)
{
   for (unsigned i = 0; i < drawCount; ++i) {
      const int32_t vertex_offset =
         pVertexOffset != NULL ? *pVertexOffset : pIndexInfo->vertexOffset;

      hk_draw_indexed(commandBuffer, i, pIndexInfo->indexCount, instanceCount,
                      pIndexInfo->firstIndex, vertex_offset, firstInstance);

      pIndexInfo = ((void *)pIndexInfo) + stride;
   }
}

/* hk_draw_impl()'s normalisation of an API indirect draw on G17P. */
static inline void
hk_g17p_normalize_indirect_draw(const struct hk_cmd_buffer *cmd,
                                struct agx_draw *draw)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   draw->restart = dyn->ia.primitive_restart_enable && draw->indexed;
   draw->index_size = cmd->state.gfx.index.size;
   if (draw->indexed && draw->index_buffer_range_B == 0) {
      draw->index_buffer = AGX_ZERO_PAGE_ADDRESS;
      draw->index_buffer_range_B = 4;
      draw->start = 0;
   }
}

/*
 * Records of an indirect (multi-)draw on G17P whose draws would find nothing
 * to do in hk_draw_impl() / hk_flush_gfx_state() /
 * hk_flush_dynamic_state_slow() except their own publication: no dirty
 * dynamic, shader, link, root, descriptor or vertex-buffer state, and none of
 * the transformed, robust, conditional, statistics, multiview, split or trace
 * paths.  That is every record after the first of an indexed-indirect
 * multi-draw.  In that state the general path reduces to exactly the sequence
 * in hk_g17p_emit_indirect_record() -- stream space, the record's draw
 * parameters, the vertex-program publication (hk_agx3_publish_vertex_state())
 * when the vertex shader reads draw parameters, then the draw record
 * (hk_g17p_emit_draw()) -- so records are emitted directly, byte for byte
 * what the general path emits.
 *
 * The check reads no per-record value except the indirect address, and
 * emitting a record changes none of what it reads, so a multi-draw checks it
 * once (draw is already normalised).  It has no side effects.
 */
static bool
hk_g17p_indirect_records_eligible(struct hk_cmd_buffer *cmd,
                                  const struct agx_draw *draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   assert(agx_is_indirect(draw->b));

   if (dev->dev.chip != AGX_CHIP_G17P || !cmd->current_cs.gfx ||
       vk_command_buffer_has_error(&cmd->vk) ||
       hk_shader_perf_draw_enabled(dev) || dev->g17p_shader_perf.trace ||
       hk_g17p_draw_trace_enabled() || (dev->dev.debug & AGX_DBG_DIRTY))
      return false;

   /* Nothing dirty: every state-derivation block of the general path is
    * then a no-op, and hk_flush_shaders() returns before doing anything. */
   if (gfx->shaders_dirty || gfx->linked_dirty || gfx->dirty ||
       desc->root_dirty || desc->push_dirty ||
       vk_dynamic_graphics_state_any_dirty(dyn) ||
       !gfx->fast_link_vs.valid || !gfx->fast_link_fs.valid ||
       gfx->prepared_vertex_usc)
      return false;

   struct hk_api_shader *fs_obj = gfx->shaders[MESA_SHADER_FRAGMENT];
   struct hk_linked_shader *linked_vs = gfx->linked[MESA_SHADER_VERTEX];
   if (!fs_obj || !gfx->shaders[MESA_SHADER_VERTEX] || !linked_vs ||
       !gfx->linked[MESA_SHADER_FRAGMENT] || linked_vs->sw_indexing ||
       gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] || gfx->render.view_mask ||
       mesa_prim_has_adjacency(
          vk_topology_to_mesa(dyn->ia.primitive_topology)))
      return false;

   /* hk_bound_fs() must answer from its cache without dirtying the FS. */
   if (fs_obj->g17p_ordinary_variants &&
       !(gfx->ordinary_fs.valid && gfx->ordinary_fs.api == fs_obj &&
         gfx->ordinary_fs.descriptor_generation == desc->generation &&
         gfx->ordinary_fs.samples == dyn->ms.rasterization_samples))
      return false;

   if (hk_conditional_render_is_active(&cmd->state.conditional, cmd->in_meta,
                                       cmd->conditional_meta) ||
       (desc->root.draw.pipeline_stats_flags &
        (VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
         VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
         VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
         VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
         VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT)) ||
       dyn->rs.depth_bias.enable ||
       (dyn->ms.rasterization_samples && !gfx->render.tilebuffer.nr_samples))
      return false;

   if (hk_g17p_draw_split_policy(dev) != HK_G17P_DRAW_SPLIT_NEVER ||
       hk_g17p_draws_per_stream(dev) ||
       hk_g17p_state_publications_per_stream(dev) ||
       debug_get_option_hk_g17p_interdraw_barrier())
      return false;

   /* hk_draw_impl()'s index validation, and the general path's refusals. */
   if (draw->indexed) {
      struct agx_draw robust_draw = *draw;
      if ((draw->index_size != AGX_INDEX_SIZE_U8 &&
           draw->index_size != AGX_INDEX_SIZE_U16 &&
           draw->index_size != AGX_INDEX_SIZE_U32) ||
          !draw->index_buffer ||
          hk_needs_index_robustness(cmd, &robust_draw, NULL))
         return false;
   }

   return agx_vdm_draw_supported(dev->dev.chip, *draw, gfx->topology);
}

/* Emit one eligible record (see hk_g17p_indirect_records_eligible()). */
static void
hk_g17p_emit_indirect_record(struct hk_cmd_buffer *cmd, uint16_t draw_id,
                             struct agx_draw draw, bool republishes)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_cs *cs = cmd->current_cs.gfx;

   /* The general path from here on, with its no-op branches removed. */
   hk_ensure_cs_has_space(cmd, cs, HK_GFX_DRAW_RESERVATION);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   bool tail_emitted = false;
   if (republishes) {
      struct hk_linked_shader *linked_vs = gfx->linked[MESA_SHADER_VERTEX];
      struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
      struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);

      cs->g17p_state_publications++;
      uint8_t *out = cs->current;

      if (linked_vs->b.uses_base_param) {
         gfx->draw_params =
            draw.b.ptr +
            (draw.indexed ? offsetof(VkDrawIndexedIndirectCommand, vertexOffset)
                          : offsetof(VkDrawIndirectCommand, firstVertex));
      } else {
         gfx->draw_params = 0;
      }
      gfx->g17p_draw_params_host_valid = false;

      if (sw_vs->b.info.uses_draw_id) {
         /* The general path uploads the whole 32-bit draw index. */
         const uint32_t draw_id_word = draw_id;
         gfx->g17p_draw_id_value = draw_id_word;
         gfx->g17p_draw_id_host_valid = true;
         if (gfx->prepared_draw_id.gpu) {
            memcpy(gfx->prepared_draw_id.cpu, &draw_id_word, 4);
            gfx->draw_id_ptr = gfx->prepared_draw_id.gpu;
            gfx->prepared_draw_id = (struct agx_ptr){0};
         } else {
            gfx->draw_id_ptr = hk_pool_upload(cmd, &draw_id_word, 4, 4);
         }
      } else {
         gfx->draw_id_ptr = 0;
         gfx->g17p_draw_id_host_valid = false;
      }

      const bool prepared_varyings =
         hk_g17p_prepared_bind_tail_matches(cmd, &gfx->g17p_bind_tail);
      const uint64_t prepared_varyings_root = gfx->root;

      hk_reserve_scratch(cmd, cs, hw_vs);
      struct hk_linked_shader *linked_hw_vs =
         (hw_vs == sw_vs) ? linked_vs : hw_vs->only_linked;
      const uint64_t pipeline_program =
         hk_upload_usc_words(cmd, hw_vs, linked_hw_vs);
      gfx->prepared_vertex_usc = 0;
      gfx->prepared_vertex_compact_root = (struct agx_ptr){0};

      uint64_t unused_ns = 0;
      if (!hk_agx3_publish_vertex_state(
             cmd, cs, &out, hw_vs, pipeline_program, draw, true,
             prepared_varyings && gfx->root == prepared_varyings_root,
             &tail_emitted, false, &unused_ns, &unused_ns, &unused_ns))
         return;

      cs->current = out;
      if (vk_command_buffer_has_error(&cmd->vk))
         return;

      vk_dynamic_graphics_state_clear_dirty(dyn);
      gfx->shaders_dirty = 0;
      gfx->linked_dirty = 0;
      gfx->dirty = 0;
      desc->root_dirty = false;
   }

   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   cs->stats.calls++;
   if (!hk_g17p_emit_draw(cmd, cs, cs->current, draw, gfx->topology, false,
                          tail_emitted))
      return;
   cs->stats.cmds++;
}

/* Record one indirect record: directly when the multi-draw is eligible
 * (checked once, on the first record that is), otherwise through hk_draw(). */
static inline void
hk_draw_indirect_record(struct hk_cmd_buffer *cmd, uint16_t draw_id,
                        struct agx_draw draw, int *eligible)
{
   if (*eligible != 0) {
      struct agx_draw normalized = draw;
      hk_g17p_normalize_indirect_draw(cmd, &normalized);
      if (*eligible < 0) {
         *eligible = hk_g17p_indirect_records_eligible(cmd, &normalized)
                        ? 1 : -1;
      }
      /* A zero indirect address is refused by the general path. */
      if (*eligible > 0 && normalized.b.ptr &&
          !vk_command_buffer_has_error(&cmd->vk)) {
         hk_g17p_emit_indirect_record(
            cmd, draw_id, normalized, hk_gfx_republishes_render_state(cmd));
         return;
      }
   }
   hk_draw(cmd, draw_id, draw);
}

static void
hk_draw_indirect_inner(VkCommandBuffer commandBuffer, uint64_t base,
                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndirect-drawCount-00476
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to sizeof(VkDrawIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndirectCommand));
   }

   int eligible = hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G17P ? -1 : 0;
   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      uint64_t addr = base + stride * draw_id;
      hk_draw_indirect_record(cmd, draw_id, agx_draw_indirect(addr),
                              &eligible);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                   VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   hk_draw_indirect_inner(commandBuffer, hk_buffer_address_ro(buffer, offset),
                          drawCount, stride);
}

static void
hk_draw_indexed_indirect_inner(VkCommandBuffer commandBuffer, uint64_t buffer,
                               uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndexedIndirect-drawCount-00528
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to
    * sizeof(VkDrawIndexedIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndexedIndirectCommand));
   }

   int eligible = hk_cmd_buffer_device(cmd)->dev.chip == AGX_CHIP_G17P ? -1 : 0;
   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      uint64_t addr = buffer + stride * draw_id;
      struct hk_addr_range buf = cmd->state.gfx.index.buffer;

      hk_draw_indirect_record(
         cmd, draw_id,
         agx_draw_indexed_indirect(addr, buf.addr, buf.range, 0, 0),
         &eligible);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, uint32_t drawCount,
                          uint32_t stride)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   hk_draw_indexed_indirect_inner(
      commandBuffer, hk_buffer_address_ro(buffer, offset), drawCount, stride);
}

/*
 * To implement drawIndirectCount generically, we dispatch a compute kernel to
 * patch the indirect buffer and then we dispatch the predicated maxDrawCount
 * indirect draws.
 */
static void
hk_draw_indirect_count(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                       VkDeviceSize offset, VkBuffer countBuffer,
                       VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                       uint32_t stride, bool indexed)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(hk_buffer, count_buffer, countBuffer);

   perf_debug(cmd, "Draw indirect count");

   assert((stride % 4) == 0 && "aligned");

   /* The Vulkan commands are defined as no-ops for a zero maximum. Avoid a
    * zero-sized pool allocation and a zero-grid internal dispatch. */
   if (maxDrawCount == 0)
      return;

   size_t out_stride = sizeof(uint32_t) * (indexed ? 5 : 4);
   uint64_t patched = hk_pool_alloc(cmd, out_stride * maxDrawCount, 4).gpu;
   uint64_t in = hk_buffer_address_ro(buffer, offset);
   uint64_t count_addr = hk_buffer_address_ro(count_buffer, countBufferOffset);

   libagx_predicate_indirect(cmd, agx_1d(maxDrawCount),
                             AGX_BARRIER_ALL | AGX_PREGFX, patched, in,
                             count_addr, stride / 4, indexed);

   if (indexed) {
      hk_draw_indexed_indirect_inner(commandBuffer, patched, maxDrawCount,
                                     out_stride);
   } else {
      hk_draw_indirect_inner(commandBuffer, patched, maxDrawCount, out_stride);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                        VkDeviceSize offset, VkBuffer countBuffer,
                        VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                        uint32_t stride)
{
   hk_draw_indirect_count(commandBuffer, _buffer, offset, countBuffer,
                          countBufferOffset, maxDrawCount, stride, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                               VkDeviceSize offset, VkBuffer countBuffer,
                               VkDeviceSize countBufferOffset,
                               uint32_t maxDrawCount, uint32_t stride)
{
   hk_draw_indirect_count(commandBuffer, _buffer, offset, countBuffer,
                          countBufferOffset, maxDrawCount, stride, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectByteCountEXT(VkCommandBuffer commandBuffer,
                               uint32_t instanceCount, uint32_t firstInstance,
                               VkBuffer counterBuffer,
                               VkDeviceSize counterBufferOffset,
                               uint32_t counterOffset, uint32_t vertexStride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, counter, counterBuffer);

   perf_debug(cmd, "Draw indirect byte count");

   const uint64_t draw =
      hk_pool_alloc(cmd, sizeof(VkDrawIndirectCommand), 4).gpu;
   const uint64_t count =
      hk_buffer_address_ro(counter, counterBufferOffset);

   libagx_draw_indirect_byte_count(
      cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, draw, count,
      counterOffset, vertexStride, instanceCount, firstInstance);

   hk_draw_indirect_inner(commandBuffer, draw, 1,
                          sizeof(VkDrawIndirectCommand));
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindTransformFeedbackBuffersEXT(VkCommandBuffer commandBuffer,
                                      uint32_t firstBinding,
                                      uint32_t bindingCount,
                                      const VkBuffer *pBuffers,
                                      const VkDeviceSize *pOffsets,
                                      const VkDeviceSize *pSizes)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;
      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;

      gfx->xfb[idx] = hk_buffer_addr_range(buffer, pOffsets[i], size, false);
   }
}

static void
hk_begin_end_xfb(VkCommandBuffer commandBuffer, uint32_t firstCounterBuffer,
                 uint32_t counterBufferCount, const VkBuffer *pCounterBuffers,
                 const VkDeviceSize *pCounterBufferOffsets, bool begin)

{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   gfx->xfb_enabled = begin;

   /* If we haven't reserved XFB offsets yet for the command buffer, do so. */
   if (!gfx->xfb_offsets) {
      gfx->xfb_offsets = hk_pool_alloc(cmd, 4 * sizeof(uint32_t), 4).gpu;
   }

   struct libagx_xfb_counter_copy params = {};
   unsigned copies = 0;

   /* For CmdBeginTransformFeedbackEXT, we need to initialize everything */
   if (begin) {
      for (copies = 0; copies < 4; ++copies) {
         params.dest[copies] = gfx->xfb_offsets + copies * sizeof(uint32_t);
      }
   }

   for (unsigned i = 0; i < counterBufferCount; ++i) {
      if (pCounterBuffers[i] == VK_NULL_HANDLE)
         continue;

      VK_FROM_HANDLE(hk_buffer, buffer, pCounterBuffers[i]);

      uint64_t offset = pCounterBufferOffsets ? pCounterBufferOffsets[i] : 0;
      uint64_t cb_addr = hk_buffer_address_rw(buffer, offset);
      uint32_t cmd_idx = firstCounterBuffer + i;

      if (begin) {
         params.src[cmd_idx] = cb_addr;
      } else {
         params.dest[copies] = cb_addr;
         params.src[copies] = gfx->xfb_offsets + cmd_idx * sizeof(uint32_t);
         ++copies;
      }
   }

   if (begin)
      copies = 4;

   if (copies > 0) {
      perf_debug(cmd, "XFB counter copy");

      libagx_copy_xfb_counters(cmd, agx_1d(copies),
                               AGX_BARRIER_ALL | AGX_PREGFX,
                               hk_pool_upload(cmd, &params, sizeof(params), 8));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                                uint32_t firstCounterBuffer,
                                uint32_t counterBufferCount,
                                const VkBuffer *pCounterBuffers,
                                const VkDeviceSize *pCounterBufferOffsets)
{
   hk_begin_end_xfb(commandBuffer, firstCounterBuffer, counterBufferCount,
                    pCounterBuffers, pCounterBufferOffsets, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                              uint32_t firstCounterBuffer,
                              uint32_t counterBufferCount,
                              const VkBuffer *pCounterBuffers,
                              const VkDeviceSize *pCounterBufferOffsets)
{
   hk_begin_end_xfb(commandBuffer, firstCounterBuffer, counterBufferCount,
                    pCounterBuffers, pCounterBufferOffsets, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginConditionalRenderingEXT(
   VkCommandBuffer commandBuffer,
   const VkConditionalRenderingBeginInfoEXT *pConditionalRenderingBegin)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, pConditionalRenderingBegin->buffer);
   struct hk_conditional_render_state *conditional =
      &cmd->state.conditional;

   assert(!conditional->active);
   conditional->address =
      hk_buffer_address_ro(buffer, pConditionalRenderingBegin->offset);
   conditional->active = true;
   conditional->inverted =
      pConditionalRenderingBegin->flags &
      VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT;

   if (!conditional->address)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndConditionalRenderingEXT(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_conditional_render_state *conditional =
      &cmd->state.conditional;

   assert(conditional->active);
   const bool inherited = conditional->inherited;
   *conditional = (struct hk_conditional_render_state){
      .inherited = inherited,
   };
}
