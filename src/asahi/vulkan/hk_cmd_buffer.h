/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "util/macros.h"

#include "util/list.h"
#include "agx_abi.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_pack.h"
#include "agx_tilebuffer.h"
#include "agx_uvs.h"
#include "libagx_dgc.h"
#include "pool.h"
#include "shader_enums.h"

#include "asahi/compiler/agx3_ioto_uniform_program.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx3_compute_program_core.h"
#include "asahi/lib/agx_ppp_hal200_graphics_core.h"
#include "asahi/lib/agx_ppp_viewport_payload_core.h"
#include "asahi/lib/agx_vdm_hal200_bind_tail_core.h"

#include "hk_private.h"
#include "hk_conditional_render_core.h"
#include "hk_shader.h"
#include "hk_sync.h"

#include "hk_cmd_pool.h"
#include "hk_cs_root_size.h"
#include "hk_descriptor_set.h"
#include "hk_drm_render_core.h"
#include "hk_ioto_arena_core.h"
#include "hk_pipeline_statistics_core.h"

#include "asahi/lib/agx_nir_lower_vbo.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

#include "vk_command_buffer.h"

#define HK_G17P_FAULTTRACE_SNAPSHOT_MAGIC UINT64_C(0x3250414e53474648)
#define HK_G17P_FAULTTRACE_MAX_DESCRIPTOR_BYTES (64 * 1024)

struct hk_g17p_faulttrace_snapshot_header {
   uint64_t magic;
   uint64_t program_va;
   uint64_t shader_va;
   uint64_t root_va;
   uint32_t ioto_size;
   uint32_t linked_size;
   uint32_t main_size;
   uint32_t root_size;
   uint64_t descriptor_va[HK_MAX_SETS];
   uint32_t descriptor_size[HK_MAX_SETS];
   uint32_t descriptor_capture_size[HK_MAX_SETS];
};

#include <stdio.h>

struct hk_buffer;
struct hk_cmd_bo;
struct hk_cmd_pool;
struct hk_image_view;
struct hk_push_descriptor_set;
struct hk_ray_tracing_pipeline;
struct hk_api_shader;
struct hk_shader;
struct hk_linked_shader;
struct agx_usc_builder;
struct vk_shader;

/* Exact launch metadata selected while graphics state is prepared. Program
 * records, indirect helper roots and geometry blocks are command-buffer-owned
 * and remain unreachable by firmware until the matching draw copies the
 * launch bytes into its live stream. Mutable indirect backing is single-use. */
struct hk_g17p_prepared_dispatch {
   struct agx3_compute_program program;
   uint8_t direct_template[AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE];
   uint8_t indirect_local_template
      [AGX3_G17P_NATIVE_INDIRECT_COMPUTE_DISPATCH_SIZE];
   struct agx_ptr indirect_local_geometry;
   struct agx_ptr indirect_local_helper_root;
   struct hk_shader *shader;
   struct hk_linked_shader *linked;
   uint64_t root;
   uint64_t draw_params;
   uint64_t draw_id_ptr;
   uint64_t indirect_local_grid;
   uint32_t indirect_local_size[3];
   bool indirect_local_valid;
   bool indirect_local_consumed;
   bool valid;
};

/* Exact command-buffer-owned precompiled launch selected while graphics state
 * is complete. Keep the key inline: the dynarray which owns the backing
 * loader may grow, so a hot-path pointer to one of its entries would be
 * unstable. The launch record itself is pointer-free. */
#define HK_G17P_PREPARED_PRECOMP_DATA_MAX 72
struct hk_g17p_prepared_precomp {
   struct agx3_compute_program program;
   struct agx_workgroup local_size;
   enum libagx_program idx;
   uint8_t data[HK_G17P_PREPARED_PRECOMP_DATA_MAX];
   uint8_t direct_template[AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE];
   uint8_t data_size;
   bool valid;
};

#define HK_G17P_SW_TESS_BATCH_LAUNCHES 5
struct hk_g17p_prepared_sw_tess_batch {
   uint8_t bytes[HK_G17P_SW_TESS_BATCH_LAUNCHES]
                [AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE];
   struct hk_shader *vs;
   struct hk_shader *tcs;
   struct hk_shader *gs_vs;
   struct hk_shader *gs_main;
   uint64_t dynamic_generation;
   uint64_t descriptor_generation;
   uint64_t root;
   uint64_t draw_params;
   uint64_t draw_id_ptr;
   uint64_t state;
   uint64_t c_prims;
   uint64_t c_invs;
   uint16_t count_program;
   uint16_t final_program;
   bool includes_gs;
   bool gs_emitted;
   bool valid;
};

/* A complete but unpublished HAL200 state update.  The referenced VDM state
 * is command-buffer-owned.  The scissor/depth records stay inline until the
 * draw chooses its live stream, so preparation cannot perturb stream-local
 * indices or make firmware-visible bytes reachable early. */
struct hk_g17p_prepared_bind_tail {
   struct agx_vdm_hal200_bind_tail tail;
   uint8_t scissors[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS]
                   [AGX_SCISSOR_LENGTH];
   uint8_t depth_bias[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS]
                    [AGX_DEPTH_BIAS_LENGTH];
   uint64_t dynamic_generation;
   uint64_t descriptor_generation;
   uint64_t vertex_buffer_generation;
   uint64_t root;
   /* The fragment shader variant the tail launches and the uniform-DMA set
    * base its program was built with (hk_g17p_dma_set_base()). */
   const struct hk_shader *fragment_shader;
   uint64_t fragment_dma_set_base;
   struct hk_image_view *render_color0;
   struct hk_image_view *render_depth;
   struct hk_image_view *render_stencil;
   VkRect2D render_area;
   VkFormat render_depth_format;
   uint32_t render_width;
   uint32_t render_height;
   uint32_t render_layer_count;
   uint32_t render_view_mask;
   uint8_t render_color_count;
   uint8_t render_samples;
   struct hk_api_shader *shaders[MESA_SHADER_MESH + 1];
   struct hk_linked_shader *linked_vs;
   struct hk_linked_shader *linked_fs;
   uint64_t varyings;
   uint32_t varying_counts_32;
   uint32_t varying_counts_16;
   uint32_t occlusion_index;
   uint8_t topology;
   uint8_t object_type;
   uint8_t provoking;
   uint8_t occlusion_mode;
   uint8_t viewport_count;
   bool generate_primitive_id;
   bool force_never_in_shader;
   bool in_meta;
   bool valid;
};

/** Root descriptor table. */
struct hk_root_descriptor_table {
   /* Address of this descriptor itself. Must be first for reflection. */
   uint64_t root_desc_addr;

   /* Descriptor set base addresses. Must follow root_desc_addr to match our
    * push layout.
    */
   uint64_t sets[HK_MAX_SETS];

   union {
      struct {
         uint32_t view_index;
         uint32_t ppp_multisamplectl;

         /* Vertex input state, indexed by compacted attribute slot. The
          * bases live at the end of this arm (attrib_base below): the clamp
          * and stride tables at 32 entries exactly fill the 256 bytes the
          * three 16-entry tables used to, which keeps every later field --
          * notably the pinned pipeline_stats offsets -- where it was.
          */
         uint32_t attrib_clamps[AGX_MAX_VBUFS];
         uint32_t attrib_strides[AGX_MAX_VBUFS];

         /* Mask of outputs flowing VS->TCS, VS->GS, or TES->GS . */
         uint64_t vertex_outputs;

         /* Address of vertex param buffer if geom/tess is used, else 0 */
         uint64_t vertex_params;

         /* Address of tessellation param buffer if tessellation used, else 0 */
         uint64_t tess_params;

         /* Address of geometry param buffer if GS is used, else 0 */
         uint64_t geometry_params;

         /* Pipeline statistics queries. This is a base address with flags. */
         uint64_t pipeline_stats;
         VkQueryPipelineStatisticFlags pipeline_stats_flags;

         float blend_constant[4];
         uint16_t no_epilog_discard;
         uint16_t _pad1;
         uint16_t api_sample_mask;
         uint16_t _pad2;
         uint16_t force_never_in_shader;
         uint16_t _pad3;
         uint16_t provoking;
         uint16_t _pad4;

         /* True if there is an API geometry shader. If false, there may still
          * be a geometry shader in use (notably for transform feedback) but it
          * should not contribute to pipeline statistics.
          */
         uint16_t api_gs;
         uint16_t _pad5;

         uint16_t rasterization_stream;
         uint16_t _pad6;

         /* Alpha-to-coverage enable for fragment shaders that write the
          * sample mask; they apply it before their own coverage update.
          * Sysval fields must sit at 4-byte offsets (root table loads). */
         uint16_t alpha_to_coverage;
         uint16_t _pad7;

         /* Mapping from varying slots written by the last vertex stage to UVS
          * indices. This mapping must be compatible with the fragment shader.
          */
         uint8_t uvs_index[VARYING_SLOT_MAX];

         /* GPU addresses of the immutable NIR constant-data blobs of the
          * bound vertex and fragment shaders, or zero if they have none. */
         uint64_t vs_constant_data_addr;
         uint64_t fs_constant_data_addr;

         /* Vertex attribute base addresses, indexed by compacted slot. */
         uint64_t attrib_base[AGX_MAX_VBUFS];
      } draw;
      struct {
         /* GPU address of the immutable NIR constant-data blob for the
          * currently bound compute shader. */
         uint64_t constant_data_addr;
         uint64_t group_count_addr;
         uint32_t base_group[3];

         /* Native G17P indirect dispatch only.  The grid helper reads the
          * threadgroup counts at group_count_addr, multiplies them by
          * grid_helper_local_size, and publishes the six-u32 launch geometry
          * at grid_helper_geometry_addr for the main record to consume.  For
          * an indirect-local grid, group_count_addr instead holds a launch
          * geometry, which the geometry copy helper publishes as it is.  Both
          * are zero for every direct dispatch and for every other chip. */
         uint64_t grid_helper_geometry_addr;

         /* Native G17P indirect dispatch whose argument buffer is a sparse
          * residency buffer: the base of that buffer's residency bitmap (a
          * struct hk_sparse_buffer_map_header followed by one bit per page),
          * and zero otherwise.  When it is set, group_count_addr names the
          * PRIMARY mapping rather than the read-only shadow, and the grid
          * helper reads the bitmap to decide whether the arguments are
          * resident, publishing a zero grid when they are not.  See
          * hk_CmdDispatchIndirect() for why the shadow cannot be read
          * directly here. */
         uint64_t grid_helper_sparse_map_addr;

         uint32_t grid_helper_local_size[3];
         uint32_t _pad_grid_helper;

         /* VK_KHR_ray_tracing_pipeline: device address of the
          * VkTraceRaysIndirectCommand2KHR record the megakernel reads its
          * shader binding tables and launch size from.  Zero for every
          * ordinary compute dispatch; the ray-tracing megakernel is the only
          * shader that loads it. */
         uint64_t trace_rays_addr;
      } cs;
   };

   /* Client push constants */
   uint8_t push[HK_MAX_PUSH_SIZE];

   /* Dynamic buffer bindings */
   struct hk_buffer_address dynamic_buffers[HK_MAX_DYNAMIC_BUFFERS];

   /* Start index in dynamic_buffers where each set starts */
   uint8_t set_dynamic_buffer_start[HK_MAX_SETS];

   /* Process-local on G17P; loaded through the root instead of cached code. */
   uint64_t printf_buffer_address;
};

/* The graphics arm is the larger of the two, so everything after the union --
 * push constants, dynamic buffers, the printf address -- keeps its offset when
 * a compute field is added. Growing the compute arm past it would move all of
 * them at once, which no single change should do silently. */
static_assert(sizeof(((struct hk_root_descriptor_table *)0)->cs) <=
                 sizeof(((struct hk_root_descriptor_table *)0)->draw),
              "compute root fields must not grow the root descriptor table");

/* helper macro for computing root descriptor byte offsets */
#define hk_root_descriptor_offset(member)                                      \
   offsetof(struct hk_root_descriptor_table, member)

struct hk_descriptor_state {
   bool root_dirty;
   uint64_t generation;
   struct hk_root_descriptor_table root;

   uint32_t set_sizes[HK_MAX_SETS];
   struct hk_descriptor_set *sets[HK_MAX_SETS];
   uint32_t sets_dirty;

   struct hk_push_descriptor_set *push[HK_MAX_SETS];
   uint32_t push_dirty;
   uint32_t push_active;
};

struct hk_attachment {
   VkFormat vk_format;
   struct hk_image_view *iview;

   VkResolveModeFlagBits resolve_mode;
   struct hk_image_view *resolve_iview;

   bool clear;
   uint32_t clear_colour[4];
};

struct hk_bg_eot {
   uint64_t usc;
   union {
      struct agx_counts_packed counts;
      uint64_t rsrc_spec;
   };
};

/* BG/EOT programs are self-addressed and may reference command-buffer pool
 * uploads, so reuse them only inside the command buffer which owns both. One
 * exact render-pass key owns its main/partial BG/EOT variants; hash collisions
 * remain harmless misses. */
#define HK_G17P_BG_EOT_CACHE_ENTRIES 16
#define HK_G17P_BG_EOT_CACHE_KEY_MAX 768
struct hk_g17p_bg_eot_dev_entry;

struct hk_g17p_bg_eot_cache_entry {
   uint8_t key[HK_G17P_BG_EOT_CACHE_KEY_MAX];
   struct hk_bg_eot program[4];
   /* frg-codegen v4: the device-lifetime entry this pass fills or reuses
    * (HK_G17P_BG_EOT_DEVICE_CACHE), or NULL. */
   struct hk_g17p_bg_eot_dev_entry *dev_entry;
   uint64_t hash;
   uint16_t key_size;
   uint8_t valid_variants;
   uint8_t direct_color_resolve_variants;
   bool valid;
};

struct hk_render_registers {
   uint32_t width, height, layers;
   uint32_t zls_width, zls_height;
   uint32_t isp_bgobjdepth;
   uint32_t isp_bgobjvals;
   struct agx_zls_control_packed zls_control, zls_control_partial;
   uint32_t depth_dimensions;
   bool process_empty_tiles;
   enum u_tristate dbias_is_int;

   struct {
      uint32_t dimensions;
      uint64_t buffer, meta;
      uint32_t stride, meta_stride;
   } depth;

   struct {
      uint64_t buffer, meta;
      uint32_t stride, meta_stride;
   } stencil;

   struct {
      struct hk_bg_eot main;
      struct hk_bg_eot partial;
   } bg;

   struct {
      struct hk_bg_eot main;
      struct hk_bg_eot partial;
   } eot;
};

struct hk_rendering_state {
   VkRenderingFlagBits flags;

   VkRect2D area;
   uint32_t layer_count;
   uint32_t view_mask;

   /* The main G17P EOT writes the single-sample resolve target directly. */
   bool g17p_direct_color_resolve;

   /* Exact admission result for the current physical tilebuffer layout.
    * The layout is immutable after BeginRendering except for the one
    * attachmentless late-sample transition, which refreshes this value. */
   bool g17p_rt0_layout_supported;

   uint32_t color_att_count;
   struct hk_attachment color_att[HK_MAX_RTS];
   struct hk_attachment depth_att;
   struct hk_attachment stencil_att;

   struct agx_tilebuffer_layout tilebuffer;
   struct hk_render_registers cr;

   /* HK_ZLS_PRECISE_DEPTH_STORE (v50, default off): the ZLS words for the
    * stream that ends the pass, when it is also the first stream (only) or
    * a later one (last). Depth honours storeOp there; stencil keeps the
    * forced store. zls_first_cs is the stream BeginRendering created. */
   bool zls_precise;
   struct agx_zls_control_packed zls_control_only, zls_control_last;
   const void *zls_first_cs;

   /* Preserve untouched NONE/NONE attachments. The broader
    * HK_G17P_ZLS_ELIDE_UNTOUCHED path remains opt-in. The pass may drop its
    * depth/stencil attachment from every control stream when no draw or
    * clear in it tests, writes, bounds-tests or clears depth/stencil and
    * neither aspect has a CLEAR load op or a resolve (see
    * hk_g17p_zls_elide()). */
   bool g17p_zls_elide_ok;
   bool g17p_depth_touched;
};

struct hk_index_buffer_state {
   struct hk_addr_range buffer;
   enum agx_index_size size;
   uint32_t restart;
};

/* Dirty tracking bits for state not tracked by vk_dynamic_graphics_state or
 * shaders_dirty.
 */
enum hk_dirty {
   HK_DIRTY_VB = BITFIELD_BIT(0),
   HK_DIRTY_OCCLUSION = BITFIELD_BIT(1),
   HK_DIRTY_PROVOKING = BITFIELD_BIT(2),
   HK_DIRTY_VARYINGS = BITFIELD_BIT(3),
};

#define HK_G17P_GRAPHICS_IOTO_KEY_MAX 240

#define HK_G17P_GRAPHICS_ROOT_VERSION 1u
#define HK_G17P_GRAPHICS_ROOT_MAX_RANGES 7u

enum hk_g17p_graphics_root_valid {
   HK_G17P_GRAPHICS_ROOT_RANGE0 = BITFIELD_BIT(0),
};

/* Versioned, ordinary command-buffer memory consumed by a linked shader's
 * persistent G17P setup program.  Executable code never contains these
 * draw-varying values.  Every field is immutable once its launcher is
 * published and validity is explicit rather than inferred from zero values. */
struct hk_g17p_graphics_root_record {
   uint32_t version;
   uint32_t size_B;
   uint64_t valid;
   uint32_t range_count;
   uint32_t _pad;

   /* Source VAs only.  The shader-owned setup template defines each range's
    * destination and width, loads the VA through this fixed array, then
    * dereferences it.  Direct and indirect draws therefore share one exact
    * executable shape; address changes are ordinary data changes. */
   uint64_t range_address[HK_G17P_GRAPHICS_ROOT_MAX_RANGES];
};

static_assert(offsetof(struct hk_g17p_graphics_root_record, range_address) % 8 == 0,
              "G17P root qwords must be naturally aligned");
static_assert(sizeof(struct hk_g17p_graphics_root_record) <= 512,
              "G17P graphics root must remain compact and bounded");

/* Where a G17P graphics uniform range's source address comes from. */
enum hk_g17p_range_source {
   /* Fixed for the shader and device (heap pointers, promoted constants). */
   HK_G17P_RANGE_SOURCE_CONSTANT = 0,
   /* The graphics root plus a fixed offset. */
   HK_G17P_RANGE_SOURCE_ROOT = 1,
   HK_G17P_RANGE_SOURCE_DRAW_PARAMS = 2,
   HK_G17P_RANGE_SOURCE_DRAW_ID = 3,
   /* Uploaded while collecting the ranges (combined-sampler pointers,
    * spilled render-target descriptors): not re-derivable. */
   HK_G17P_RANGE_SOURCE_UPLOADED,
};

struct hk_g17p_prepared_fragment_key {
   const struct hk_shader *shader;
   const struct hk_linked_shader *linked;
   uint64_t root;
   uint64_t dma_set_base;
   uint16_t tile_size_px;
   uint8_t sample_size_B;
   uint8_t nr_samples;
   bool spills;
   bool blend_constant;
};

struct hk_g17p_graphics_ioto_cache_entry {
   uint8_t key[HK_G17P_GRAPHICS_IOTO_KEY_MAX];
   uint64_t program;
   const void *source_shader;
   uint64_t linked_shader_va;
   uint64_t rodata_va;
   uint64_t root;
   uint64_t draw_params;
   uint64_t draw_id_ptr;
   uint16_t key_size;
   uint16_t tile_size_px;
   uint8_t sample_size_B;
   uint8_t nr_samples;
   bool root_dependent;
   bool fast_valid;
   bool valid;

   /* Launcher inputs of the last compact publication for this stage.  A
    * later draw of the same linked shader, render layout and range layout
    * whose only change is range source addresses -- a new root after a
    * descriptor push, or the next indexed-indirect record's draw parameters
    * -- would re-derive exactly the same bounds, constant-program extent and
    * setup template.  The record keeps those so such a draw rebuilds only
    * its root record and launcher. */
   struct {
      struct agx3_ioto_uniform_range ranges[HK_G17P_GRAPHICS_ROOT_MAX_RANGES];
      /* The same ranges after hk_g17p_graphics_compact_ranges(), their root
       * record, and where each source address came from (enum
       * hk_g17p_range_source) with its root-relative offset.  When no source
       * is an upload made while collecting (patchable), a draw can re-derive
       * every address from the root, draw-parameter and draw-ID pointers. */
      struct agx3_ioto_uniform_range
         compact_ranges[HK_G17P_GRAPHICS_ROOT_MAX_RANGES];
      struct hk_g17p_graphics_root_record root;
      /* Patchable address r is base[source[r]] + source_offset[r], with
       * base = {0, root, draw parameters, draw ID}; a constant's offset is
       * its address. */
      uint64_t source_offset[HK_G17P_GRAPHICS_ROOT_MAX_RANGES];
      uint8_t source[HK_G17P_GRAPHICS_ROOT_MAX_RANGES];
      bool patchable;
      bool uses_root;
      bool has_draw_params;
      bool has_draw_id;
      /* Fragment admission inputs (hk_g17p_basic_graphics_program_supported):
       * the varying-coefficient check reads the topology, provoking vertex
       * and hardware vertex shader. */
      const void *admit_hw_vs;
      uint8_t admit_topology;
      uint8_t admit_provoking;
      bool external;
      const void *source_shader;
      const void *linked;
      const void *setup;
      uint64_t linked_shader_va;
      uint64_t rodata_va;
      uint8_t range_count;
      uint16_t tile_size_px;
      uint8_t sample_size_B;
      uint8_t nr_samples;
      bool blend;
      bool spills;
      bool valid;
      /* hk_g17p_dma_set_base() when this record was made: a set-pointer
       * uniform DMA span is recorded as a constant address, valid only
       * while the same set stays bound. */
      uint64_t dma_set_base;
   } compact;
   /* AGX_G17P_UNIFORM_DMA span source when it is a descriptor-set pointer
    * (see hk_g17p_dma_set_base); 0 otherwise. */
   uint64_t dma_set_base;

   /* HK_G17P_MDI_ONE_LEVEL: the last one-level (absolute-address) vertex
    * setup published for an indexed-indirect multi-draw record.  The next
    * record of the same linked shader, root, descriptor-set DMA base and
    * render layout differs only in its draw-parameter / draw-ID addresses,
    * so it relocates the same linked-shader template (`entry`, owned by
    * `linked`) to those addresses instead of re-deriving every range. */
   struct {
      struct agx3_ioto_uniform_range ranges[HK_G17P_GRAPHICS_ROOT_MAX_RANGES];
      uint64_t source_offset[HK_G17P_GRAPHICS_ROOT_MAX_RANGES];
      uint8_t source[HK_G17P_GRAPHICS_ROOT_MAX_RANGES];
      const void *entry;
      const void *source_shader;
      const void *linked;
      uint64_t linked_shader_va;
      uint64_t rodata_va;
      uint64_t root;
      uint64_t dma_set_base;
      size_t capacity;
      uint16_t tile_size_px;
      uint8_t sample_size_B;
      uint8_t nr_samples;
      uint8_t range_count;
      bool has_draw_params;
      bool has_draw_id;
      bool valid;
      /* Patch form: the template's bytes at its published VA, valid for
       * any record because the program is position-independent (checked
       * by relocation at a second VA), with only the LDIMM pairs that
       * address the draw-parameter/draw-ID ranges rewritten per record. */
      bool patch;
      uint8_t site_count;
      uint16_t length;
      struct {
         uint16_t offset;
         uint16_t dest_half;
         uint8_t low_length;
         uint8_t high_length;
         uint8_t range;
         uint64_t delta;
      } sites[4];
      uint8_t bytes[512];
   } one_level;
};

struct hk_prepared_fast_link_vs {
   struct hk_shader *sw_vs;
   struct hk_shader *hw_vs;
   uint64_t dynamic_generation;
   enum agx_index_size index_size;
   bool indexed;
   bool restart;
   bool valid;
};

struct hk_graphics_state {
   struct hk_ordinary_fs_selection ordinary_fs;
   /* v96: per-draw selection of a vertex object's ordinary hardware variant
    * (hk_select_ordinary_vs()). Valid for `api` at descriptor `generation`;
    * UINT64_MAX marks a bind-time prediction the draw must re-prove. */
   struct {
      struct hk_api_shader *api;
      uint64_t generation;
      bool valid, ordinary;
   } ordinary_vs;
   /* v57d: set while hk_cmd_prepare_bound_graphics() runs (bind time). */
   bool g17p_bind_prepare;
   struct hk_rendering_state render;
   struct hk_descriptor_state descriptors;

   enum hk_dirty dirty;
   /* CmdBindVertexBuffers is not vk_dynamic_graphics_state and therefore does
    * not advance its generation.  Prepared G17P roots snapshot this separate
    * identity so a post-BeginRendering bind cannot publish stale attrib VAs. */
   uint64_t vertex_buffer_generation;
   /* The vertex_buffer_generation that descriptors.root.draw.attrib_* were
    * last derived from (hk_flush_dynamic_state_slow()'s vertex-buffer
    * block).  A prepared bind tail records this, not the live generation:
    * a tail rebuilt outside a full flush (vkCmdBindIndexBuffer) must not
    * vouch for a root whose attrib VAs predate the latest vertex binding. */
   uint64_t root_vertex_buffer_generation;

   uint64_t root;
   struct agx_ptr prepared_root;
   uint64_t draw_params;
   uint64_t draw_id_ptr;
   uint32_t g17p_draw_params_value[2];
   uint32_t g17p_draw_id_value;
   bool g17p_draw_params_host_valid;
   bool g17p_draw_id_host_valid;
   struct agx_ptr prepared_draw_params;
   struct agx_ptr prepared_draw_id;
   struct agx_ptr prepared_vertex_params;
   bool prepared_vertex_params_filled;
   struct agx_ptr prepared_tess_params;
   bool prepared_tess_params_filled;
   struct agx_ptr prepared_tess_out_draws;
   struct agx_ptr prepared_geometry_params;
   bool prepared_geometry_params_filled;
   uint64_t prepared_vertex_usc;
   /* HK_G17P_MDI_ONE_LEVEL: set only around the vertex-program upload of an
    * indexed-indirect record in hk_g17p_emit_indirect_record(). */
   bool g17p_vs_one_level;
   struct agx_ptr prepared_vertex_compact_root;
   const struct hk_shader *prepared_vertex_shader;
   const struct hk_linked_shader *prepared_vertex_linked;
   uint64_t prepared_vertex_root;
   uint64_t prepared_vertex_draw_params;
   uint64_t prepared_vertex_draw_id_ptr;
   uint64_t prepared_vertex_dma_set_base;
   struct agx_vdm_state_update prepared_vdm_state_update;
   bool prepared_vdm_state_update_valid;
   uint64_t prepared_fragment_usc;
   /* What prepared_fragment_usc was published for: the inputs of
    * hk_upload_usc_words() for a fragment shader. A leftover program must
    * not be consumed once any of them changed. */
   struct hk_g17p_prepared_fragment_key prepared_fragment_key;

   struct {
      struct hk_g17p_prepared_dispatch tess_vs;
      struct hk_g17p_prepared_dispatch tess_tcs;
      struct hk_g17p_prepared_dispatch gs_vs;
      struct hk_g17p_prepared_dispatch gs_main;
      struct hk_g17p_prepared_dispatch gs_count;
      struct hk_g17p_prepared_dispatch gs_pre;
   } g17p_dispatch;

   struct {
      struct hk_g17p_prepared_precomp tess_count;
      struct hk_g17p_prepared_precomp tess_prefix;
      struct hk_g17p_prepared_precomp tess_final;
      struct hk_g17p_prepared_precomp gs_setup;
   } g17p_precomp;
   struct hk_g17p_prepared_sw_tess_batch g17p_sw_tess_batch;

   struct hk_g17p_prepared_bind_tail g17p_bind_tail;
   /* Index key (width, restart) whose vertex prelink and bind tail the last
    * index-buffer bind of the current render pass prepared. */
   struct {
      bool valid;
      bool restart;
      uint8_t size;
   } g17p_index_prepared;

   /* The last successfully built HAL200 bind tail's inputs and packed
    * records.  A rebuild whose only changed inputs are the root, descriptor
    * or vertex-buffer identity (a descriptor push or vertex-buffer bind)
    * reads exactly the same state for every record except the fragment
    * record's stage program, which is recomputed; see
    * hk_g17p_build_native_bind_tail(). */
   struct {
      struct hk_g17p_prepared_bind_tail snapshot;
      struct agx_ppp_hal200_fragment_fields fs_fields;
      struct agx_ppp_viewport_payload_core viewport_record;
      uint8_t bind_group[0x80];
      const struct hk_shader *fs;
      const struct hk_shader *hw_vs;
      bool valid;
   } g17p_bind_tail_memo;

   /* Per vertex-attribute slot: the last API format seen and the pipe format
    * and interchange block size hk_flush_dynamic_state_slow() derives from
    * it, which are pure functions of the format. */
   struct {
      VkFormat vk_format;
      enum pipe_format format;
      unsigned interchange_align;
   } vb_format_memo[AGX_MAX_VBUFS];

   /* The last VertexProgramVariant state update this command buffer encoded,
    * keyed by every encoder input except the pipeline address; see
    * hk_g17p_encode_vdm_state_update(). */
   struct {
      struct agx_vdm_state_update_draw_info info;
      struct agx_vdm_state_update encoded;
      bool valid;
   } g17p_vdm_update_memo;

   uint32_t shaders_dirty;
   struct hk_api_shader *shaders[MESA_SHADER_MESH + 1];

   /* Vertex buffers */
   struct hk_addr_range vb[AGX_MAX_VBUFS];

   /* Transform feedback buffers */
   struct hk_addr_range xfb[4];

   /* Is transform feedback enabled? */
   bool xfb_enabled;

   /* Internal transform feedback offset vec4.
    *
    * TODO: Strictly could be global.
    */
   uint64_t xfb_offsets;

   /* Pointer to the GPU memory backing active transform feedback queries,
    * per-stream. Zero if no query is bound.
    */
   uint64_t xfb_query[4];

   struct hk_index_buffer_state index;
   enum agx_primitive topology;
   enum agx_object_type object_type;

   /* Provoking vertex 0, 1, or 2. Usually 0 or 2 for FIRST/LAST. 1 can only be
    * set for tri fans.
    */
   uint8_t provoking;

   struct {
      enum agx_visibility_mode mode;

      /* If enabled, index of the current occlusion query in the occlusion heap.
       * There can only be one active at a time (hardware contraint).
       */
      uint16_t index;
   } occlusion;

   /* Fast linked shader data structures */
   uint64_t varyings;
   struct agx_varyings_vs linked_varyings;
   /* hk_g17p_smooth_varyings_supported() of the last (linked FS, hardware
    * VS, topology, provoking) it evaluated against the linked_varyings
    * above. Cleared whenever linked_varyings is reassigned. */
   struct {
      const void *linked;
      const void *vs;
      uint32_t topology;
      uint32_t provoking;
      bool valid;
      bool result;
   } g17p_smooth_memo;

   uint32_t linked_dirty;
   struct hk_linked_shader *linked[MESA_SHADER_STAGES];

   /* Last complete fast-link selection for this command buffer. The shader
    * identity is part of the cache key: identical bytes owned by a different
    * shader may have a different lifetime. Graphics-state invalidation clears
    * these records with the rest of gfx state. */
   struct {
      struct hk_shader *shader;
      struct hk_fast_link_key_vs key;
      bool valid;
   } fast_link_vs;
   /* Exact inputs of the most recent successful VS prelink.  This lets the
    * authoritative draw consume bind-time preparation without rebuilding the
    * same vertex-input key merely because its pre-existing dirty bits have
    * not yet been cleared.  Any later dynamic-state mutation changes the
    * generation; stage or indexed-state changes are matched separately. */
   struct hk_prepared_fast_link_vs prepared_fast_link_vs;
   struct {
      struct hk_shader *shader;
      struct hk_fast_link_key_fs key;
      bool valid;
   } fast_link_fs;

   /* Exact last VS/FS loader identities. These are recording-only selectors;
    * the published programs and every referenced BO remain command-buffer-
    * owned. Submission never mutates or consults this cache. */
   struct hk_g17p_graphics_ioto_cache_entry g17p_ioto[2];
   /* Exact command-buffer-owned uploads for combined texture/sampler pointer
    * pairs.  Shader/material switches may revisit the same pair thousands of
    * times; a fresh upload address would otherwise force a fresh Ioto program
    * even though the GPU-visible values are identical. */
   struct {
      uint64_t pointers[2];
      uint64_t address;
      bool valid;
   } g17p_combined_sampler_pointers[32];

   /* Exact immutable eMRT descriptor-table upload used by every fragment
    * program in one spilled render configuration. */
   struct {
      uint8_t entries[HK_MAX_RTS][AGX_G17P_EMRT_DESCRIPTOR_SIZE];
      uint64_t address;
      uint64_t pointer;
      uint8_t color_att_count;
      bool valid;
   } g17p_spilled_rt_descs;

   bool generate_primitive_id;

   /* Tessellation state */
   struct {
      uint64_t out_draws;
      uint64_t grids;
      struct hk_tess_info info;
      enum mesa_prim prim;

      /* G17P hardware tessellation: the factor buffer the tessellator reads,
       * and either the unrolled patch count of a direct draw or the patch
       * draw an indirect draw's setup kernel writes. */
      const struct poly_tess_params *params;
      uint64_t factor_buffer;
      uint64_t patch_draw;
      uint32_t factor_buffer_size_B;
      uint32_t patches;
   } tess;

   /* Needed by vk_command_buffer::dynamic_graphics_state */
   struct vk_vertex_input_state _dynamic_vi;
   struct vk_sample_locations_state _dynamic_sl;
};

struct hk_compute_state {
   struct hk_descriptor_state descriptors;
   struct hk_api_shader *shader;
};

/*
 * VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR state.  Descriptor sets and push
 * constants bound to the ray-tracing bind point are separate from the compute
 * ones, even though the pipeline executes as a compute shader: an application
 * may bind a compute pipeline's sets between vkCmdBindPipeline and
 * vkCmdTraceRaysKHR and neither may disturb the other.
 */
struct hk_ray_tracing_state {
   struct hk_descriptor_state descriptors;
   struct hk_ray_tracing_pipeline *pipeline;
};

#define HK_RT_SHADER_STAGES                                                    \
   (VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |         \
    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |       \
    VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR)

struct hk_cmd_push {
   void *map;
   uint64_t addr;
   uint32_t range;
   bool no_prefetch;
};

struct hk_scratch_req {
   bool main;
   bool preamble;
};

/* struct agx_timestamp_req lives in hk_sync.h. */

/*
 * hk_cs represents a single control stream, to be enqueued either to the
 * CDM or VDM for compute/3D respectively.
 */
enum hk_cs_type {
   HK_CS_CDM,
   HK_CS_VDM,
};

struct hk_cs {
   struct list_head node;

   /* Parent command buffer. Convenience. */
   struct hk_cmd_buffer *cmd;

   /* Data master */
   enum hk_cs_type type;

   /* Address of the root control stream for the job */
   uint64_t addr;

   /* Fat pointer to the start of the current chunk of the control stream */
   struct agx_ptr chunk;

   /* Start pointer of the root control stream */
   void *start;

   /* Current pointer within the control stream */
   void *current;

   /* End pointer of the current chunk of the control stream */
   void *end;

   /* AGX3 CDM: the value of current just after a strong cache flush, so a
    * flush requested with nothing recorded since it can be elided. */
   void *strong_flush_end;

   /* Likewise for the weak ordering barrier attached to a native launch. A
    * later cache-flush request upgrades this record in place instead of
    * emitting two adjacent barriers. */
   void *weak_barrier_end;

   /* AGX3 CDM: the value of current just after a standalone barrier that is
    * the stream's *first* record. Apple's emitter only opens a stream with a
    * barrier (the strong flush that makes a preceding render visible), so a
    * stream is canonical on its own; but concatenating it after a stream that
    * already ends in a barrier leaves two adjacent barriers with no launch
    * between them, which Apple never emits. hk_cs_merge_cdm() uses this to
    * fold the leading record into the destination's trailing one. */
   void *leading_barrier_end;

   /* Whether there is more than just the root chunk */
   bool stream_linked;

   /* Active-context selector consumed by AGX3 VertexProgramVariant. */
   bool agx3_active_context_selector_valid;
   uint32_t agx3_raw_active_context_selector;

   /* Render-state publications this G17P graphics stream has recorded.
    * Every one of them overwrites the same context-0 state, so this -- not
    * the draw count and not the bytes used -- is the depth that decides
    * whether the stream faults. See hk_cmd_draw.c. */
   unsigned g17p_state_publications;

   /* Compact identity for a submitted G17P draw.  The full launch trace is
    * intentionally enormous; this is enough to name the exact fragment
    * program and draw that preceded a DM1 fragment timeout. */
   struct {
      uint64_t fragment_program;
      uint64_t emitted_indirect_buffer;
      uint32_t fragment_shader_hash;
      uint32_t fragment_main_hash;
      uint32_t fragment_ioto_hash;
      uint32_t fragment_shader_size;
      uint32_t fragment_main_size;
      uint32_t fragment_ioto_size;
      uint32_t api_count;
      uint32_t api_instances;
      uint32_t api_first;
      int32_t api_base_vertex;
      uint32_t api_first_instance;
      uint32_t color_formats[HK_MAX_RTS];
      uint32_t depth_format;
      uint8_t color_count;
      uint16_t draw_id;
      uint8_t topology;
      bool api_indexed;
      bool api_indirect;
      bool emitted_indexed;
      bool emitted_indirect;
   } g17p_draw_trace;

   /* Exact fragment launch inputs retained only for failure-only faulttrace.
    * The allocation remains empty on the production path. */
   struct util_dynarray g17p_faulttrace_snapshots;

   /* Whether the sampler heap is required. Although we always must maintain the
    * heap for correctness, it's often not necessary since we can push lots of
    * samplers (especially for GL/DX11-era engines).
    */
   bool uses_sampler_heap;

   /* Scratch requirements */
   struct {
      union {
         struct hk_scratch_req vs;
         struct hk_scratch_req cs;
      };

      struct hk_scratch_req fs;
   } scratch;

   /* Immediate writes, type libagx_imm_write. These happen at the end of the
    * control stream (parallel on AGX2, native ordered stores on G17P). This
    * accelerates queries. Implies CDM.
    */
   struct util_dynarray imm_writes;

   /* Number of successfully emitted draws whose last raster producer is
    * positively known to export only the default Layer=0, without multiview.
    * Compare with stats.cmds: untracked emitters conservatively disqualify.
    * The wider proof count does not wrap with stats.cmds.
    */
   uint64_t g17p_layer_zero_draws;
   bool g17p_layer_zero_unknown;

   /* Statistics */
   struct {
      uint32_t calls, cmds, flushes, merged;
   } stats;

   /* Timestamp writes: the start of the command (compute start, render
    * vertex start) and its end (compute end, render fragment end). The
    * kernel writes both from the firmware's native pass record.
    */
   struct {
      struct agx_timestamp_req start;
      struct agx_timestamp_req end;
   } timestamp;

   /* G17P query commands observe timestamp writes from earlier streams.
    * Firmware start timestamps reach query memory only at retirement, so a
    * producer cannot be merged across this consumer boundary.
    */
   bool g17p_timestamp_query;

   /* Recorded dependencies, see hk_sync.h. */
   struct hk_sync_stream sync;

   /* Writable destinations known while recording meta compute dispatches. */
   struct hk_drm_attachment_set compute_attachments;

   /* Remaining state is for graphics only, ignored for compute */
   struct agx_tilebuffer_layout tib;

   struct util_dynarray scissor, depth_bias;
   uint64_t uploaded_scissor, uploaded_zbias;

   /* We can only set ppp_multisamplectl once per batch. has_sample_locations
    * tracks if we've committed to a set of sample locations yet. vk_meta
    * operations do not set has_sample_locations since they don't care and it
    * would interfere with the app-provided samples.
    *
    */
   bool has_sample_locations;
   uint32_t ppp_multisamplectl;

   struct hk_render_registers cr;

   /* Snapshot of the writable image ranges for this exact VDM command. Render
    * state may change before queue submission, so the synthetic fragment
    * attachment command must travel with the control stream it describes. */
   struct hk_drm_attachment_set fragment_attachments;

   /* Active restart index if one is set. Zero if there is no restart index set
    * yet, since Vulkan does not allow zero restart indices (unlike OpenGL).
    * This is used in place of dirty tracking, because dirty tracking
    * restart indices is complicated and just checking the saved value is cheap.
    */
   uint32_t restart_index;
};

static inline unsigned
hk_sync_sq(const struct hk_cs *cs)
{
   return cs->type == HK_CS_VDM ? HK_SYNC_VDM : HK_SYNC_CDM;
}

/* The later of two streams in list order; NULL is earlier than anything. */
static inline struct hk_cs *
hk_sync_later(struct hk_cs *a, struct hk_cs *b)
{
   if (!a)
      return b;
   if (!b)
      return a;
   return b->sync.seq > a->sync.seq ? b : a;
}

void *hk_cmd_buffer_emit_cs_jump(struct hk_cmd_buffer *cmd, void *current,
                                 uint64_t target, bool vdm);

/*
 * Helper to merge two compute control streams, concatenating the second control
 * stream to the first one. Must sync with hk_cs.
 */
static inline void
hk_cs_merge_cdm(struct hk_cs *a, const struct hk_cs *b)
{
   assert(a->type == HK_CS_CDM && b->type == HK_CS_CDM);
   assert(a->cmd == b->cmd);
   assert(!a->timestamp.end.handle);

   const size_t b_size = (uint8_t *)b->current - (uint8_t *)b->start;
   const size_t remaining = (uint8_t *)a->end - (uint8_t *)a->current;
   const bool inline_b = !a->stream_linked && !b->stream_linked &&
                         hk_cs_space_fits(remaining, b_size,
                                          AGX_CDM_STREAM_LINK_LENGTH);

   if (inline_b) {
      uint8_t *dst = a->current;

      /* Two barriers with no launch between them is not a shape Apple's CDM
       * emitter produces, and the G17P stream oracle rejects it. Within one
       * stream hk_cdm_cache_flush() already strengthens the trailing record
       * in place; do the same across a concatenation. b's leading barrier is
       * the strong flush, so copying its word over a's trailing barrier is
       * exactly that upgrade -- no encoder needed, and it removes a record
       * rather than adding one. */
      const bool b_opens_with_barrier =
         b->leading_barrier_end &&
         (uint8_t *)b->leading_barrier_end ==
            (uint8_t *)b->start + AGX3_CDM_BARRIER_SIZE;
      const bool a_ends_with_barrier =
         ((uint8_t *)a->current == (uint8_t *)a->strong_flush_end ||
          (uint8_t *)a->current == (uint8_t *)a->weak_barrier_end) &&
         (uint8_t *)a->current - (uint8_t *)a->start >= AGX3_CDM_BARRIER_SIZE;
      const size_t b_skip = (b_opens_with_barrier && a_ends_with_barrier)
                               ? AGX3_CDM_BARRIER_SIZE
                               : 0;
      if (b_skip)
         memcpy(dst - AGX3_CDM_BARRIER_SIZE, b->start, AGX3_CDM_BARRIER_SIZE);

      memcpy(dst, (const uint8_t *)b->start + b_skip, b_size - b_skip);
      a->current = dst + (b_size - b_skip);

      /* These markers point just past an emitted barrier. Preserve them in
       * the destination root so another adjacent merge still has an exact
       * view of the stream tail. A marker on the folded record maps onto the
       * trailing record that absorbed it. */
      a->strong_flush_end =
         b->strong_flush_end
            ? dst + ((uint8_t *)b->strong_flush_end - (uint8_t *)b->start) -
                 b_skip
            : NULL;
      a->weak_barrier_end =
         b->weak_barrier_end
            ? dst + ((uint8_t *)b->weak_barrier_end - (uint8_t *)b->start) -
                 b_skip
            : NULL;
   } else {
      void *link_end =
         hk_cmd_buffer_emit_cs_jump(a->cmd, a->current, b->addr, false);
      const bool linked = link_end != a->current;
      a->current = b->current;
      a->end = b->end;
      a->chunk = b->chunk;
      a->stream_linked |= linked || b->stream_linked;
      a->strong_flush_end = b->strong_flush_end;
      a->weak_barrier_end = b->weak_barrier_end;
   }

   a->uses_sampler_heap |= b->uses_sampler_heap;
   a->scratch.cs.main |= b->scratch.cs.main;
   a->scratch.cs.preamble |= b->scratch.cs.preamble;

   /* Submission sees only the merged root. Keep the destinations recorded
    * by both streams without changing their inline/link batching policy. */
   for (unsigned i = 0; i < b->compute_attachments.count; ++i) {
      const struct drm_asahi_attachment *attachment =
         &b->compute_attachments.list[i];
      hk_drm_attachment_set_add(&a->compute_attachments,
                                attachment->pointer, attachment->size);
   }
   a->compute_attachments.complete &= b->compute_attachments.complete;

   /* The merged command ends where b ended. It starts where a started; a
    * start stamp b carried was a top-of-pipe write, which any earlier value
    * also satisfies. The caller refuses the merge when both have one. */
   a->timestamp.end = b->timestamp.end;
   if (!a->timestamp.start.handle)
      a->timestamp.start = b->timestamp.start;

   a->g17p_timestamp_query |= b->g17p_timestamp_query;

   a->stats.calls += b->stats.calls;
   a->stats.cmds += b->stats.cmds;
   a->stats.flushes += b->stats.flushes;
   a->stats.merged++;
}

static inline uint64_t
hk_cs_current_addr(struct hk_cs *cs)
{
   return cs->chunk.gpu + ((uint8_t *)cs->current - (uint8_t *)cs->chunk.cpu);
}

struct hk_uploader {
   /** List of hk_cmd_bo */
   struct list_head bos;

   /* Current addresses */
   uint8_t *map;
   uint64_t base;
   uint32_t offset;
   uint32_t capacity;
};

/*
 * Bump arena for the G17P Ioto programs one command buffer records. Semantic
 * program reuse happens before reservation; resource programs embed their
 * reserved address, so post-composition byte hashing cannot deduplicate them.
 * See hk_ioto_arena_core.h for why programs cannot each have a BO.
 */
struct hk_ioto_arena {
   /* Chunk currently being filled. Owned by hk_cmd_buffer::large_bos. */
   struct agx_bo *bo;
   uint8_t *map;
   uint64_t base;
   uint32_t cursor;
   uint32_t capacity;

   /* Recording statistics, reported when the arena cannot grow. */
   uint32_t programs;
   uint64_t program_bytes;
   uint64_t chunk_bytes;
};

struct hk_cmd_buffer {
   struct vk_command_buffer vk;

   /* frg-codegen v4: the per-recording BG/EOT cache (13.5 KiB) lives outside
    * `state` so the full-state memset of every reset no longer clears it;
    * hk_reset_cmd_buffer() invalidates the entries instead. Entries are only
    * read when valid, so this is the same cache contract. */
   struct hk_g17p_bg_eot_cache_entry
      g17p_bg_eot_cache[HK_G17P_BG_EOT_CACHE_ENTRIES];
   uint8_t g17p_bg_eot_cache_next;

   /* v94 HK_G17P_HOST_ENCODE bit1: set by hk_reset_cmd_buffer, cleared by
    * vkBeginCommandBuffer. */
   bool g17p_pristine;

   struct {
      struct hk_graphics_state gfx;
      struct hk_compute_state cs;
      struct hk_ray_tracing_state rt;
      struct hk_conditional_render_state conditional;
   } state;

   struct {
      struct hk_uploader main, usc, root, vdm, streams;
   } uploader;

   /* Command-pool slab spans leased by this command buffer's VDM uploader. */
   struct list_head vdm_allocations;
   /* Most recently addressed span. Recording often references the current
    * 4 KiB VDM chunk many times before moving to the next one. */
   struct hk_vdm_allocation *g17p_vdm_last_allocation;

   /* List of all recorded control streams */
   struct list_head control_streams;

   /* Current recorded control stream */
   struct {
      /* VDM stream for 3D */
      struct hk_cs *gfx;

      /* CDM stream for compute */
      struct hk_cs *cs;

      /* CDM stream that executes immediately before the current graphics
       * control stream. Used for geometry shading, tessellation, etc.
       */
      struct hk_cs *pre_gfx;

      /* CDM stream that will execute after the current graphics control stream
       * finishes. Used for queries.
       */
      struct hk_cs *post_gfx;
   } current_cs;

   /* Recorded dependencies between control streams, see hk_sync.h. */
   struct hk_sync_state sync;

   /* Tile hazards: struct hk_sync_pass_record for every recorded pass
    * instance footprint, in recording order. Cleared by hk_sync_reset. */
   struct util_dynarray sync_pass_records;

   /* Are we currently inside a vk_meta operation? This alters sample location
    * behaviour.
    */
   bool in_meta;

   /* CmdClearAttachments is the sole vk_meta operation affected by
    * conditional rendering. */
   bool conditional_meta;

   /* XXX: move me?
    *
    * Indirect draw generated by the indirect GS translator.
    */
   uint64_t geom_index_buffer;
   uint32_t geom_index_count;
   uint32_t geom_instance_count;

   /* Does the command buffer use the geometry heap? */
   bool uses_heap;

   /* Owned BOs and retained references needed by recorded commands. */
   struct util_dynarray large_bos;
   /* Direct-mapped temporal cache avoids rescanning large_bos for the linked
    * vertex/fragment shader, setup and rodata BOs repeated by every draw. */
   struct agx_bo *recent_retained_bos[16];
   /* Open-addressed pointer set of the BOs hk_cmd_buffer_retain_bo() has
    * appended to large_bos (capacity a power of two, or zero). A recording
    * retains one BO per distinct shader/setup/program it publishes, so a
    * linear scan of large_bos on every temporal-cache miss grew quadratically
    * with the number of pipelines a command buffer uses. */
   struct agx_bo **retained_set;
   uint32_t retained_set_capacity;
   uint32_t retained_set_count;

   /* Executable USC-window arena the G17P Ioto programs recorded by this
    * command buffer are bump-allocated from. Its chunks are owned by
    * large_bos, so it dies with the recording like every other program
    * reference. */
   struct hk_ioto_arena ioto;

   /* Exact command-buffer-local G17P compute programs prepared before draw.
    * Entries own no storage outside the command buffer's Ioto arena. */
   struct util_dynarray g17p_compute_programs;

   /* Exact immutable libagx argument/program pairs. Their argument bytes live
    * in the command uploader and their loaders live in the Ioto arena, so
    * both have precisely command-recording lifetime. */
   struct util_dynarray g17p_precomp_programs;
   struct hash_table *g17p_precomp_index;

   /* Opt-in recording-time cache diagnosis; never touched during submission. */
   struct {
      uint64_t calls, hits, misses, probes, comparisons, max_entries;
      uint64_t first_lookup_ns, lookup_ns;
   } g17p_precomp_lookup_stats;

   /* Opt-in shader performance diagnosis; inactive in production. */
   /* v57d HK_G17P_ORDINARY_VARIANT_TRACE=3 per-command-buffer draw index. */
   uint32_t g17p_trace_draws;
   bool shader_perf_draw_active;
   uint32_t shader_perf_draw_bo_allocs;
   uint32_t shader_perf_draw_bo_creates;
   uint64_t shader_perf_draw_bo_alloc_ns;

};

/* Transfer entry points bracket their recording so the streams they create
 * belong to the transfer stage class (VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT and
 * its sub-stages). Nesting is fine: meta helpers re-enter through the
 * dispatch table. */
static inline void
hk_sync_xfer_begin(struct hk_cmd_buffer *cmd)
{
   cmd->sync.xfer_depth++;
}

static inline void
hk_sync_xfer_end(struct hk_cmd_buffer *cmd)
{
   assert(cmd->sync.xfer_depth > 0);
   cmd->sync.xfer_depth--;
}

/*
 * One reservation inside the Ioto arena: where the program is to be composed,
 * and the address the composer must be given for it.
 */
struct hk_ioto_reservation {
   void *map;
   uint64_t va;
};

/*
 * Reserve `size` bytes of executable USC-window space for one Ioto program.
 * On failure the command buffer error is already set.
 */
bool hk_cmd_reserve_ioto_program(struct hk_cmd_buffer *cmd, size_t size,
                                 const char *label,
                                 struct hk_ioto_reservation *reservation);

/*
 * Publish the `length` bytes composed at reservation->map and return the
 * reserved address for the launch record.
 */
uint64_t hk_cmd_publish_ioto_program(struct hk_cmd_buffer *cmd,
                                     const struct hk_ioto_reservation *
                                        reservation,
                                     size_t length);

/* Release the arena's bookkeeping. The chunks themselves are in large_bos. */
void hk_ioto_arena_finish(struct hk_cmd_buffer *cmd);

VK_DEFINE_HANDLE_CASTS(hk_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

extern const struct vk_command_buffer_ops hk_cmd_buffer_ops;

static inline struct hk_cmd_pool *
hk_cmd_buffer_pool(struct hk_cmd_buffer *cmd)
{
   return (struct hk_cmd_pool *)cmd->vk.pool;
}

/* Whether the current blend state reads the blend constants. Derived from
 * the dynamic graphics state at each use rather than cached: the fragment
 * stage program (its blend-constant uniform range) is built by bind-time and
 * BeginRendering preparation as well as by the draw, and that preparation
 * runs before the draw-time dirty-state flush. A cached copy refreshed only
 * by that flush was stale there, so the first draw after a blend-equation
 * change published a fragment program without the blend-constant range and
 * the epilog read unwritten uniforms. Prepared programs are keyed by
 * dynamic_graphics_state.generation, which covers every input here. */
static inline bool
hk_cmd_uses_blend_constant(const struct hk_cmd_buffer *cmd)
{
   static_assert(VK_BLEND_FACTOR_CONSTANT_COLOR + 3 ==
                    VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
                 "constant factors are contiguous");
   const struct vk_color_blend_state *cb =
      &cmd->vk.dynamic_graphics_state.cb;

   for (unsigned i = 0; i < cb->attachment_count; ++i) {
      const unsigned factors[] = {
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

/*
 * The hardware vertex shader is supplied by the last geometry stage. The
 * geometry pipeline is vertex->tess->geometry so we search backwards.
 */
static inline struct hk_shader *
hk_bound_hw_vs(struct hk_graphics_state *gfx)
{
   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

   if (gs)
      return &gs->variants[HK_GS_VARIANT_RAST];
   else if (tes)
      return &tes->variants[HK_VS_VARIANT_HW];
   else if (vs->g17p_ordinary_variants && gfx->ordinary_vs.valid &&
            gfx->ordinary_vs.api == vs && gfx->ordinary_vs.ordinary)
      return &vs->variants[HK_VS_VARIANT_HW_ORDINARY];
   else
      return &vs->variants[HK_VS_VARIANT_HW];
}

static inline struct hk_shader *
hk_bound_sw_vs(struct hk_graphics_state *gfx)
{
   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);

   if (hw_vs == &vs->variants[HK_VS_VARIANT_HW] ||
       (vs->g17p_ordinary_variants &&
        hw_vs == &vs->variants[HK_VS_VARIANT_HW_ORDINARY]))
      return hw_vs;
   else
      return &vs->variants[HK_VS_VARIANT_SW];
}

static inline struct hk_shader *
hk_bound_sw_vs_before_gs(struct hk_graphics_state *gfx)
{
   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
   struct hk_api_shader *api = tes ?: vs;

   return &api->variants[HK_VS_VARIANT_SW];
}

struct agx_ptr hk_pool_alloc_internal(struct hk_cmd_buffer *cmd, uint32_t size,
                                      uint32_t alignment, bool usc);
struct agx_ptr hk_pool_cs_alloc(struct hk_cmd_buffer *cmd, uint32_t size,
                                uint32_t alignment);
struct agx_ptr hk_pool_vdm_primary_alloc(struct hk_cmd_buffer *cmd,
                                         uint32_t size, uint32_t alignment);

uint64_t hk_g17p_vdm_compact(struct hk_cmd_buffer *cmd, uint64_t primary);
uint64_t hk_g17p_vdm_alias(struct hk_cmd_buffer *cmd, uint64_t primary);

uint64_t hk_pool_upload(struct hk_cmd_buffer *cmd, const void *data,
                        uint32_t size, uint32_t alignment);
uint64_t hk_pool_usc_upload(struct hk_cmd_buffer *cmd, const void *data,
                            uint32_t size, uint32_t alignment);

/* Keep externally owned storage referenced by recorded GPU commands alive for
 * the command-buffer lifetime. Repeated uses of the same BO are deduplicated.
 */
void hk_cmd_buffer_retain_bo(struct hk_cmd_buffer *cmd, struct agx_bo *bo);
void hk_cmd_buffer_release_bos(struct hk_cmd_buffer *cmd,
                               struct agx_device *dev);

static inline struct agx_ptr
hk_pool_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   return hk_pool_alloc_internal(cmd, size, alignment, false);
}

static inline struct agx_ptr
hk_pool_usc_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   return hk_pool_alloc_internal(cmd, size, alignment, true);
}

/*
 * Allocate a payload that a VDM record references through its 40-bit pointer.
 * HAL200 draw-state fields are narrower than the VDM control-stream pointer,
 * so those payloads retain a device-global low alias. Control-stream roots
 * themselves use the constrained main heap directly.
 */
static inline struct agx_ptr
hk_pool_vdm_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   struct agx_ptr T = hk_pool_vdm_primary_alloc(cmd, size, alignment);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (T.gpu && hk_g17p_vdm_low_window_enabled(&dev->dev)) {
      T.gpu = hk_g17p_vdm_compact(cmd, T.gpu);
      if (!T.gpu) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_OUT_OF_DEVICE_MEMORY);
         T.cpu = NULL;
      }
   }

   return T;
}

bool hk_cs_init_graphics(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
/* v56g/v57e HK_G17P_ZLS_ELIDE_UNTOUCHED: a depth/stencil clear touched it. */
void hk_g17p_note_depth_clear(struct hk_cmd_buffer *cmd);

bool hk_g17p_split_trace_enabled(void);
void hk_g17p_note_gfx_split(struct hk_cmd_buffer *cmd, const char *reason,
                            unsigned commands, size_t remaining);
uint32_t hk_default_sample_positions(unsigned nr_samples);

static inline struct hk_cs *
hk_cmd_buffer_get_cs_general(struct hk_cmd_buffer *cmd, struct hk_cs **ptr,
                             bool compute)
{
   if (vk_command_buffer_has_error(&cmd->vk))
      return NULL;

   /* A recorded dependency may have ordered the open general compute stream
    * after work that follows it in the list. It is then ended, and the work
    * goes into a stream positioned after that requirement. Work of a new
    * stage class must also update the reused stream's kind and dependencies,
    * so later barriers include transfers and meta graphics emitted there. */
   if (*ptr && ptr == &cmd->current_cs.cs &&
       (cmd->sync.cs_check_pending ||
        (hk_sync_general_cs_kind(cmd) & ~(*ptr)->sync.kind)))
      hk_sync_check_cs(cmd);

   if ((*ptr) == NULL) {
      /* Allocate root control stream */
      struct hk_device *dev = hk_cmd_buffer_device(cmd);
      size_t initial_size = hk_cs_root_size(dev->dev.chip, compute);
      struct agx_ptr root = hk_pool_cs_alloc(cmd, initial_size, 1024);
      if (!root.cpu)
         return NULL;

      /* Allocate hk_cs for the new stream */
      struct hk_cs *cs = malloc(sizeof(*cs));
      *cs = (struct hk_cs){
         .cmd = cmd,
         .type = compute ? HK_CS_CDM : HK_CS_VDM,
         .addr = root.gpu,
         .start = root.cpu,
         .chunk = root,
         .current = root.cpu,
         .end = root.cpu + initial_size,
      };

      list_inithead(&cs->node);
      util_dynarray_init(&cs->g17p_faulttrace_snapshots, NULL);
      hk_drm_attachment_set_init(&cs->compute_attachments);
      /* Only known meta destinations are collected so far. */
      cs->compute_attachments.complete = false;

      bool before_gfx = (ptr == &cmd->current_cs.pre_gfx);

      if (!compute) {
         const bool initialized = hk_cs_init_graphics(cmd, cs);
         if (!hk_graphics_cs_publishable(
                dev->dev.chip, initialized,
                cs->agx3_active_context_selector_valid)) {
            util_dynarray_fini(&cs->g17p_faulttrace_snapshots);
            free(cs);
            return NULL;
         }
      }

      /* Insert into the command buffer. We usually append to the end of the
       * command buffer, except for pre-graphics streams which go right before
       * the graphics workload. (This implies a level of out-of-order processing
       * that's allowed by Vulkan and required for efficient
       * geometry/tessellation shaders.)
       */
      if (before_gfx && cmd->current_cs.gfx) {
         list_addtail(&cs->node, &cmd->current_cs.gfx->node);
      } else {
         list_addtail(&cs->node, &cmd->control_streams);
      }

      *ptr = cs;

      hk_sync_stream_created(cmd, cs, ptr, compute);
   }

   assert(*ptr != NULL);
   return *ptr;
}

static inline struct hk_cs *
hk_cmd_buffer_get_cs(struct hk_cmd_buffer *cmd, bool compute)
{
   struct hk_cs **ptr = compute ? &cmd->current_cs.cs : &cmd->current_cs.gfx;
   return hk_cmd_buffer_get_cs_general(cmd, ptr, compute);
}

void hk_ensure_cs_has_space(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            size_t space);

static inline uint64_t
hk_cs_alloc_for_indirect(struct hk_cs *cs, size_t size_B)
{
   hk_ensure_cs_has_space(cs->cmd, cs, size_B);

   uint64_t addr = hk_cs_current_addr(cs);
   cs->current += size_B;
   return addr;
}

static void
hk_cmd_buffer_dirty_all(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   vk_dynamic_graphics_state_dirty_all(dyn);
   gfx->dirty = ~0;
   gfx->shaders_dirty = ~0;
   gfx->linked_dirty = ~0;
   gfx->descriptors.root_dirty = true;
}

static inline void
hk_cs_destroy(struct hk_cs *cs)
{
   util_dynarray_fini(&cs->g17p_faulttrace_snapshots);
   if (cs->type == HK_CS_VDM) {
      util_dynarray_fini(&cs->scissor);
      util_dynarray_fini(&cs->depth_bias);
   } else {
      util_dynarray_fini(&cs->imm_writes);
   }

   free(cs);
}

void hk_dispatch_imm_writes(struct hk_cmd_buffer *cmd, struct hk_cs *cs);

bool hk_g17p_smooth_varyings_supported(
   struct hk_cmd_buffer *cmd, const struct hk_linked_shader *linked);
bool hk_g17p_launch_trace_enabled(void);
bool hk_g17p_draw_trace_enabled(void);
bool hk_g17p_strong_dispatch_flush_enabled(void);
bool hk_g17p_verbose_enabled(void);
void hk_g17p_trace_hexdump(const char *label, uint64_t base, const void *data,
                           size_t len);
void hk_g17p_trace_line(const char *label, uint64_t base, const void *data,
                        size_t len);

void *hk_cmd_buffer_emit_cdm_terminate(struct hk_cmd_buffer *cmd,
                                       void *current);
void *hk_cmd_buffer_emit_vdm_terminate(struct hk_cmd_buffer *cmd,
                                       void *current);

static void
hk_cmd_buffer_end_compute_internal(struct hk_cmd_buffer *cmd,
                                   struct hk_cs **ptr)
{
   if (*ptr) {
      struct hk_cs *cs = *ptr;

      /* This control stream may write immediates as it ends. Queue the writes
       * now that we're done emitting everything else.
       */
      if (cs->imm_writes.size) {
         hk_dispatch_imm_writes(cmd, cs);
      }

      /* A transformed draw may reserve a pre-graphics stream and then prove
       * that it has no vertices to process. Do not leave an empty CDM job in
       * the submission list. A stream that only carries a timestamp stays:
       * that is the one job the stamp can ride on. */
      if (cs->stats.cmds == 0 && cs->stats.flushes == 0 &&
          !cs->timestamp.end.handle && !cs->timestamp.start.handle) {
         hk_sync_forget_stream(cmd, cs);
         list_del(&cs->node);
         hk_cs_destroy(cs);
      }
   }

   *ptr = NULL;
}

static void
hk_cmd_buffer_end_compute(struct hk_cmd_buffer *cmd)
{
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.cs);
}

/* Return true when the VDM was replaced/removed, false when it needs normal
 * graphics finalization (for example, a native background clear). */
bool hk_optimize_empty_vdm(struct hk_cmd_buffer *cmd);

static void
hk_cmd_buffer_end_graphics(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.gfx;

   if (cs && cs->stats.cmds == 0 && hk_optimize_empty_vdm(cmd))
      cs = NULL;

   if (cs) {
      struct hk_device *dev = hk_cmd_buffer_device(cmd);
      const bool full_usc_state = dev->dev.chip == AGX_CHIP_G17P;
      /* Scissor and depth bias arrays are staged to dynamic arrays on the
       * CPU. When we end the control stream, they're done growing and are
       * ready for upload. G17P's ISP consumes these as full bit-40-set GART
       * addresses, matching native's USC-window allocations; a main-heap
       * address has bit 39 set and incorrectly selects the upper root.
       */
      cs->uploaded_scissor =
         full_usc_state
            ? hk_pool_usc_upload(cmd, cs->scissor.data, cs->scissor.size, 64)
            : hk_pool_upload(cmd, cs->scissor.data, cs->scissor.size, 64);

      cs->uploaded_zbias =
         full_usc_state
            ? hk_pool_usc_upload(cmd, cs->depth_bias.data,
                                 cs->depth_bias.size, 64)
            : hk_pool_upload(cmd, cs->depth_bias.data,
                             cs->depth_bias.size, 64);

      /* TODO: maybe free scissor/depth_bias now? */
      cs->current = hk_cmd_buffer_emit_vdm_terminate(cmd, cs->current);
   }

   cmd->current_cs.gfx = NULL;

   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.pre_gfx);
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);

   assert(cmd->current_cs.pre_gfx == NULL);
   assert(cmd->current_cs.post_gfx == NULL);

   /* We just flushed out the heap use. If we want to use it again, we'll need
    * to queue a free for it again.
    */
   cmd->uses_heap = false;
}

static inline uint64_t
hk_pipeline_stat_addr(struct hk_cmd_buffer *cmd,
                      VkQueryPipelineStatisticFlagBits stat)
{
   struct hk_root_descriptor_table *root = &cmd->state.gfx.descriptors.root;
   VkQueryPipelineStatisticFlags flags = root->draw.pipeline_stats_flags;

   if (flags & stat) {
      assert(!cmd->in_meta && "queries paused for meta");
      assert(util_bitcount(stat) == 1 && "by construction");

      uint32_t index;
      const bool present =
         hk_pipeline_statistics_report_index(flags, stat, &index);
      assert(present && "enabled statistic has a compact report index");

      return root->draw.pipeline_stats + (sizeof(uint64_t) * index);
   } else {
      /* Query disabled */
      return AGX_SCRATCH_PAGE_ADDRESS;
   }
}

static inline bool
hk_stat_enabled(uint64_t addr)
{
   return addr != AGX_SCRATCH_PAGE_ADDRESS;
}

void hk_cmd_buffer_begin_graphics(struct hk_cmd_buffer *cmd,
                                  const VkCommandBufferBeginInfo *pBeginInfo);
void hk_cmd_buffer_begin_compute(struct hk_cmd_buffer *cmd,
                                 const VkCommandBufferBeginInfo *pBeginInfo);

void hk_cmd_invalidate_graphics_state(struct hk_cmd_buffer *cmd);
void hk_cmd_invalidate_compute_state(struct hk_cmd_buffer *cmd);

void hk_cmd_bind_shaders(struct vk_command_buffer *vk_cmd, uint32_t stage_count,
                         const mesa_shader_stage *stages,
                         struct vk_shader **const shaders);

void hk_cmd_bind_graphics_shader(struct hk_cmd_buffer *cmd,
                                 const mesa_shader_stage stage,
                                 struct hk_api_shader *shader);

/* Prepare immutable G17P graphics-compute records once a complete shader bind
 * is visible. This is called both when rendering starts and after a batched
 * vkCmdBindShadersEXT, so no shader set pays lazy setup in vkCmdDraw*. */
void hk_cmd_prepare_bound_graphics(struct hk_cmd_buffer *cmd);

void hk_cmd_bind_compute_shader(struct hk_cmd_buffer *cmd,
                                struct hk_api_shader *shader);

void hk_cmd_bind_vertex_buffer(struct hk_cmd_buffer *cmd, uint32_t vb_idx,
                               struct hk_addr_range addr_range);

static inline struct hk_descriptor_state *
hk_get_descriptors_state(struct hk_cmd_buffer *cmd,
                         VkPipelineBindPoint bind_point)
{
   switch (bind_point) {
   case VK_PIPELINE_BIND_POINT_GRAPHICS:
      return &cmd->state.gfx.descriptors;
   case VK_PIPELINE_BIND_POINT_COMPUTE:
      return &cmd->state.cs.descriptors;
   case VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR:
      return &cmd->state.rt.descriptors;
   default:
      UNREACHABLE("Unhandled bind point");
   }
};

void hk_cmd_buffer_flush_push_descriptors(struct hk_cmd_buffer *cmd,
                                          struct hk_descriptor_state *desc);

void hk_meta_resolve_rendering(struct hk_cmd_buffer *cmd,
                               const VkRenderingInfo *pRenderingInfo);

uint64_t hk_cmd_buffer_upload_root(struct hk_cmd_buffer *cmd,
                                   VkPipelineBindPoint bind_point);
bool hk_cmd_buffer_prepare_graphics_root(struct hk_cmd_buffer *cmd);

void hk_reserve_scratch(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        struct hk_shader *s);

/* AGX2 returns its shader-base-relative USC offset.  G17P returns the absolute
 * 48-bit Ioto program VA consumed by HAL200 VDM state. */
uint64_t hk_g17p_dma_set_base(const struct hk_cmd_buffer *cmd,
                              const struct hk_shader *s);
/* HK_G17P_MDI_ONE_LEVEL (hk_cmd_buffer.c). */
bool hk_g17p_mdi_one_level_enabled(void);

uint64_t hk_upload_usc_words(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                             struct hk_linked_shader *linked);
uint64_t hk_prepare_usc_words(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                              struct hk_linked_shader *linked,
                              struct agx_ptr *compact_root);
bool hk_retarget_prepared_usc_root(struct hk_cmd_buffer *cmd,
                                   struct hk_shader *s,
                                   struct hk_linked_shader *linked,
                                   uint64_t prepared_program,
                                   struct agx_ptr compact_root);

void hk_usc_upload_spilled_rt_descs(struct agx_usc_builder *b,
                                    struct hk_cmd_buffer *cmd);

void hk_cdm_cache_flush(struct hk_device *dev, struct hk_cs *cs);

void hk_dispatch_with_usc_launch(struct hk_device *dev, struct hk_cs *cs,
                                 struct agx_cdm_launch_word_0_packed launch,
                                 uint32_t usc, struct agx_grid grid,
                                 struct agx_workgroup local_size);

void hk_dispatch_with_usc(struct hk_device *dev, struct hk_cs *cs,
                          struct agx_shader_info *info, uint32_t usc,
                          struct agx_grid grid,
                          struct agx_workgroup local_size);

void hk_dispatch_g17p_native(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                             struct hk_shader *s, struct agx_grid grid,
                             struct agx_workgroup local_size);

void hk_g17p_prewarm_linked_compute_program(struct hk_cmd_buffer *cmd,
                                             struct hk_shader *s,
                                             struct hk_linked_shader *linked,
                                             bool internal_graphics);
bool hk_g17p_prepare_graphics_dispatch(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct hk_linked_shader *linked, uint64_t indirect_local_grid,
   struct agx_workgroup indirect_local_size,
   struct hk_g17p_prepared_dispatch *out);
bool hk_dispatch_prepared_g17p_native(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, struct hk_shader *s,
   struct hk_g17p_prepared_dispatch *prepared, struct agx_grid grid,
   struct agx_workgroup local_size);
void hk_g17p_prewarm_indirect_helpers(struct hk_cmd_buffer *cmd);
void hk_g17p_prewarm_linked_compute_shader(struct hk_device *dev,
                                            struct hk_shader *s,
                                            struct hk_linked_shader *linked);
void hk_g17p_linked_compute_cache_finish(struct hk_linked_shader *linked);
void hk_g17p_linked_graphics_cache_finish(struct hk_device *dev,
                                          struct hk_linked_shader *linked);

/*
 * Saved application state around a driver-internal meta operation.  The
 * acceleration-structure builder needs this as much as a blit does: the
 * common build shaders bind their own compute pipeline and push constants,
 * so a build recorded between two application dispatches would otherwise
 * silently clobber them.
 */
struct hk_meta_save {
   struct vk_vertex_input_state _dynamic_vi;
   struct vk_sample_locations_state _dynamic_sl;
   struct vk_dynamic_graphics_state dynamic;
   struct hk_api_shader *shaders[MESA_SHADER_MESH + 1];
   struct hk_addr_range vb0;
   struct hk_descriptor_set *desc0;
   /* Nested meta must retain the outer query/conditional-rendering scope. */
   bool in_meta;
   bool has_push_desc0;
   bool push_active0;
   enum agx_visibility_mode occlusion;
   struct hk_push_descriptor_set push_desc0;
   VkQueryPipelineStatisticFlags pipeline_stats_flags;
   uint8_t push[HK_MAX_PUSH_SIZE];
};

void hk_meta_begin(struct hk_cmd_buffer *cmd, struct hk_meta_save *save,
                   VkPipelineBindPoint bind_point);
void hk_meta_end(struct hk_cmd_buffer *cmd, struct hk_meta_save *save,
                 VkPipelineBindPoint bind_point);

void hk_cmd_dispatch_rays(struct hk_cmd_buffer *cmd,
                          struct hk_api_shader *shader, uint64_t record_addr,
                          uint64_t group_count_addr, struct agx_grid grid);

void hk_cmd_dispatch_unaligned(VkCommandBuffer commandBuffer,
                               uint32_t invocations_x, uint32_t invocations_y,
                               uint32_t invocations_z);

static inline void
hk_dispatch_with_local_size(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            struct hk_shader *s, struct agx_grid grid,
                            struct agx_workgroup local_size)
{
   if (agx_is_shader_empty(&s->b))
      return;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (agx_chip_command_stream_generation(dev->dev.chip) ==
       AGX_CS_GENERATION_HAL200) {
      hk_reserve_scratch(cmd, cs, s);
      hk_dispatch_g17p_native(cmd, cs, s, grid, local_size);
      return;
   }

   uint64_t uploaded = hk_upload_usc_words(cmd, s, s->only_linked);
   if (uploaded > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint32_t usc = uploaded;

   hk_reserve_scratch(cmd, cs, s);
   hk_dispatch_with_usc(dev, cs, &s->b.info, usc, grid, local_size);
}

void hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                         enum agx_barrier barrier, enum libagx_program idx,
                         void *data, size_t data_size);
void hk_dispatch_precomp_g17p(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                              enum agx_barrier barrier,
                              enum libagx_program idx, const void *data,
                              size_t data_size);
void hk_g17p_precomp_lookup_stats_finish(struct hk_cmd_buffer *cmd);
void hk_g17p_prewarm_precomp(struct hk_cmd_buffer *cmd,
                             enum libagx_program idx, const void *data,
                             size_t data_size);
bool hk_g17p_prepare_precomp(struct hk_cmd_buffer *cmd,
                             enum libagx_program idx, const void *data,
                             size_t data_size,
                             struct hk_g17p_prepared_precomp *out);
bool hk_dispatch_prepared_precomp_g17p(
   struct hk_cmd_buffer *cmd, struct agx_grid grid, enum agx_barrier barrier,
   enum libagx_program idx, const void *data, size_t data_size,
   const struct hk_g17p_prepared_precomp *prepared);

/* Turn a three/four/five-word indirect argument record into an execution-time
 * conditional record. `barrier` selects the ordered compute stream which must
 * precede its consumer. */
uint64_t hk_apply_conditional_rendering(struct hk_cmd_buffer *cmd,
                                        uint64_t source, uint32_t words,
                                        enum agx_barrier barrier);

#define MESA_DISPATCH_PRECOMP hk_dispatch_precomp

void hk_queue_write(struct hk_cmd_buffer *cmd, uint64_t address, uint32_t value,
                    bool after_gfx);

/* Dispatch the native G17P raw-address fill meta shader into an explicitly
 * selected compute stream.  The target may be the ordinary or post-graphics
 * stream; the latter preserves query-availability ordering. */
void hk_g17p_meta_fill_addr(struct hk_cmd_buffer *cmd, struct hk_cs **target,
                            uint64_t address, uint64_t size, uint32_t value);

/* Poll a naturally aligned 32-bit device address from a native G17P compute
 * stream until it reaches value. */
void hk_g17p_meta_wait_addr(struct hk_cmd_buffer *cmd, uint64_t address,
                            uint32_t value);
void hk_native_meta_copy_query(struct hk_cmd_buffer *cmd, uint64_t destination,
                               uint64_t source, uint64_t availability,
                               VkQueryResultFlags flags);

void agx_fill_velem_keys(const struct vk_vertex_input_state *vi,
                         uint64_t attribs_read, struct agx_velem_key *keys);

struct agx_robustness hk_prolog_robustness(struct hk_device *dev);

/* A refusal during command recording reaches the application as one
 * VK_ERROR_FEATURE_NOT_PRESENT out of vkEndCommandBuffer, with nothing to say
 * which of the driver's three dozen refusal sites produced it -- so every such
 * failure costs a bisect that the driver already knows the answer to.
 * HK_DEBUG=cmderr names the file and line that set the error. */
void hk_trace_cmd_error(const char *file, unsigned line, VkResult error);

#undef vk_command_buffer_set_error
#define vk_command_buffer_set_error(command_buffer, error)                     \
   (hk_trace_cmd_error(__FILE__, __LINE__, (error)),                           \
    __vk_command_buffer_set_error((command_buffer), (error), __FILE__,          \
                                  __LINE__))
