/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once
#include "hk_ordinary_descriptor.h"

#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir.h"
#include "poly/nir/poly_nir.h"
#include "util/macros.h"
#include "agx_linker.h"
#include "agx_nir_lower_vbo.h"
#include "agx_pack.h"
#include "agx_usc.h"
#include "agx_uvs.h"
#include "agx_vdm_state_update_core.h"

#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_private.h"

#include "nir_xfb_info.h"
#include "nir.h"
#include "shader_enums.h"
#include "vk_pipeline_cache.h"

#include "nir_defines.h"

#include "vk_shader.h"

struct hk_physical_device;
struct hk_pipeline_compilation_ctx;
struct vk_descriptor_set_layout;
struct vk_graphics_pipeline_state;
struct vk_pipeline_cache;
struct vk_pipeline_layout;
struct vk_pipeline_robustness_state;
struct vk_shader_module;

struct hk_tess_info {
   enum tess_primitive_mode mode : 8;
   enum gl_tess_spacing spacing  : 8;
   bool points;
   bool ccw;
};
static_assert(sizeof(struct hk_tess_info) == 4, "packed");

static struct hk_tess_info
hk_tess_info_merge(struct hk_tess_info a, struct hk_tess_info b)
{
   static_assert(TESS_PRIMITIVE_UNSPECIFIED == 0, "zero state");
   static_assert(TESS_SPACING_UNSPECIFIED == 0, "zero state");

   /* Just merge by OR'ing the raw bits */
   uint32_t x, y;
   memcpy(&x, &a, sizeof(x));
   memcpy(&y, &b, sizeof(y));

   x |= y;

   struct hk_tess_info out;
   memcpy(&out, &x, sizeof(out));
   return out;
}

/*
 * Whether an evaluation shader's own domain is one Apple's G17P tessellation
 * state record can express. The record carries a single patch-type bit for
 * triangles and quads (docs/G17P_TESSELLATION_ABI.md 8.2); there is no
 * isoline encoding and no point-mode encoding, so those domains keep the
 * software tessellator even when the native path is selected.
 *
 * Vulkan lets either tessellation stage carry the execution modes, so a
 * control shader can still introduce point mode for an evaluation shader
 * that did not declare it. That case is caught at draw time; this predicate
 * only decides which evaluation-shader lowering to compile.
 */
static inline bool
hk_g17p_tes_domain_is_native(const struct shader_info *info)
{
   return !info->tess.point_mode &&
          (info->tess._primitive_mode == TESS_PRIMITIVE_TRIANGLES ||
           info->tess._primitive_mode == TESS_PRIMITIVE_QUADS);
}

struct hk_shader_info {
   union {
      struct {
         uint32_t attribs_read;
         BITSET_DECLARE(attrib_components_read, AGX_MAX_ATTRIBS * 4);
         bool use_prolog;
         uint8_t _pad[7];
      } vs;

      struct {
         struct agx_interp_info interp;
         struct agx_fs_epilog_link_info epilog_key;

         bool reads_sample_mask;
         bool post_depth_coverage;
         bool uses_sample_shading;
         bool early_fragment_tests;
         bool writes_memory;

         /* Rasterization samples the G17P coverage updates are compiled for */
         uint8_t g17p_samples;

         /* AGX_G17P_UNIT_RANGE_BLEND: every colour output is provably in
          * [0, 1] (computed only with the flag set). */
         bool g17p_colour_unit_range;

         uint8_t _pad[5];
      } fs;

      struct {
         uint64_t tcs_per_vertex_outputs;
         uint32_t tcs_output_stride;
         uint8_t tcs_output_patch_size;
         uint8_t tcs_nr_patch_outputs;

         struct hk_tess_info info;

         /* Evaluation shaders only: this shader's own domain is one the G17P
          * hardware tessellator can express (triangles or quads, no point
          * mode) and the device is running the native path, so the hardware
          * variant was lowered with poly_nir_lower_tes_native(). An isoline
          * or point-mode evaluation shader keeps the software lowering even
          * with HK_G17P_NATIVE_TESS=1, because Apple's tessellation state
          * record has no encoding for either. */
         bool g17p_native_domain;
      } tess;

      struct poly_gs_info gs;

      /* Used to initialize the union for other stages */
      uint8_t _pad[32];
   };

   struct agx_unlinked_uvs_layout uvs;

   /* Transform feedback buffer strides */
   uint8_t xfb_stride[MAX_XFB_BUFFERS];

   mesa_shader_stage stage : 8;
   uint8_t clip_distance_array_size;
   uint8_t cull_distance_array_size;
   uint8_t set_count;

   /* Positive proof recorded before inserting the default Layer output.
    * Only original vertex hardware variants qualify.  False is unknown.
    * Serialized with shader binary version 26 or newer.
    */
   bool g17p_default_layer_zero;

   /* The bounded G17P graphics/compute path lowers one Gen9 combined
    * image/sampler to native texture/sampler state 0. Command recording joins
    * the bound descriptor set and native sampler heap into those CProg slots.
    */
   bool agx3_combined_sampler_valid;
   bool agx3_uses_root;
   /* Diagnostic HK_G17P_APPLE_CS / HK_G17P_APPLE_FS (hk_g17p_apple_override.c):
    * this variant executes Apple's machine code. Never set otherwise. */
   bool g17p_apple_cs;
   bool g17p_apple_fs;
   /* Actual uniform address selected after stage and resource lowering. */
   unsigned agx3_root_uniform;
   bool agx3_txf_texture_valid;
   bool agx3_storage_image_valid;
   bool agx3_sampled_storage_valid;
   bool agx3_indexed_sampler_valid;
   bool agx3_indexed_storage_valid;
   struct agx_g17p_combined_sampler_binding agx3_combined_sampler;
   struct agx_g17p_txf_binding agx3_txf_texture;
   struct agx_g17p_storage_image_binding agx3_storage_image;
   struct agx_g17p_sampled_storage_binding agx3_sampled_storage;
   struct agx_g17p_indexed_storage_binding agx3_indexed_storage;

   /* XXX: is there a less goofy way to do this? I really don't want dynamic
    * allocation here.
    */
   nir_xfb_info xfb_info;
   nir_xfb_output_info xfb_outputs[64];
};

/*
 * Hash table keys for fast-linked shader variants. These contain the entire
 * prolog/epilog key so we only do 1 hash table lookup instead of 2 in the
 * general case where the linked shader is already ready.
 */
struct hk_fast_link_key_vs {
   struct agx_vs_prolog_key prolog;
};

struct hk_fast_link_key_fs {
   unsigned nr_samples_shaded;
   struct agx_fs_prolog_key prolog;
   struct agx_fs_epilog_key epilog;
};

struct hk_shader {
   struct agx_shader_part b;

   struct hk_shader_info info;
   struct agx_fragment_face_2_packed frag_face;
   struct agx_counts_packed counts;

   /* Native AGX3 VertexProgramVariant state.  G17P hardware vertex shaders
    * with position and up to two ordinary varying slots populate the exact
    * no-preamble fields; other AGX3 layouts remain zero. */
   struct agx_vdm_state_update_program_info agx3_vdm_state_update;

   const void *code_ptr;
   uint32_t code_size;

   const void *data_ptr;
   uint32_t data_size;

   /* BO for any uploaded shader part */
   struct agx_bo *bo;

   /* Cache of fast linked variants */
   struct {
      simple_mtx_t lock;
      struct hash_table *ht;
      bool lazy_ht;
   } linked;

   /* The variant was compiled (hk_compile_nir) or deserialized from a
    * compiled variant.  False only for a variant that was never compiled,
    * e.g. the software vertex variant of a shader without a tessellation or
    * geometry successor.  A compiled variant may have zero bytes of code: a
    * G17P fragment shader with no work (an empty main) is all prolog and
    * epilog, so code_size == 0 does not mean "never compiled".
    */
   bool compiled;

   /* If there's only a single possibly linked variant, direct pointer. TODO:
    * Union with the cache to save some space?
    */
   struct hk_linked_shader *only_linked;

   /* Address to the uploaded preamble section. Preambles are uploaded
    * separately from fast-linked main shaders.
    */
   uint64_t preamble_addr;

   /* Address of the start of the shader data section */
   uint64_t data_addr;

   /* v54 AGX_G17P_INLINE_BLEND: the fragment main's NIR (serialized just
    * before its backend compile) and that compile's key, so fast-link can
    * build one program with the draw's colour epilog inlined. NULL when the
    * flag is off or the shape is not eligible. */
   void *g17p_mono_nir;
   size_t g17p_mono_nir_size;
   struct agx_shader_key g17p_mono_key;

   /* v56e HK_G17P_LAZY_GENERIC: a generic ordinary-descriptor variant whose
    * backend compile is deferred (on first selection, or on a background
    * queue). NULL when the variant was compiled eagerly. Every consumer of a
    * variant's compiled state goes through hk_shader_ensure_compiled(). */
   struct hk_lazy_compile *lazy;
};

enum hk_vs_variant {
   /* Hardware vertex shader, when next stage is fragment */
   HK_VS_VARIANT_HW,

   /* Hardware compute shader, when next is geometry/tessellation */
   HK_VS_VARIANT_SW,

   HK_VS_VARIANTS,

   /* v96 (G17P, HK_G17P_ORDINARY_VS_VARIANTS): the hardware vertex shader
    * specialised for ordinary descriptors, selected per draw like the
    * fragment ordinary variants. Allocated only when the API vertex shader
    * has g17p_ordinary_variants; the two variants above stay the complete
    * generic programs. */
   HK_VS_VARIANT_HW_ORDINARY = HK_VS_VARIANTS,
};

enum hk_gs_variant {
   /* Hardware vertex shader used for rasterization */
   HK_GS_VARIANT_RAST,

   /* Main compute shader */
   HK_GS_VARIANT_MAIN,

   /* Count compute shader */
   HK_GS_VARIANT_COUNT,

   /* Pre-GS compute shader */
   HK_GS_VARIANT_PRE,

   HK_GS_VARIANTS,
};

/* clang-format off */
static const char *hk_gs_variant_name[] = {
   [HK_GS_VARIANT_RAST] = "Rasterization",
   [HK_GS_VARIANT_MAIN] = "Main",
   [HK_GS_VARIANT_COUNT] = "Count",
   [HK_GS_VARIANT_PRE] = "Pre-GS",
};
/* clang-format on */

static inline unsigned
hk_num_variants(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_TESS_EVAL:
      return HK_VS_VARIANTS;

   case MESA_SHADER_GEOMETRY:
      return HK_GS_VARIANTS;

   default:
      return 1;
   }
}

/*
 * A G17P hardware vertex program whose UVS layout does not start at component
 * zero with a four-component position group. The AGX3 packer admits a dense
 * terminal output run only from five outputs up, a floor every position-
 * writing shader clears through its four position components alone, so every
 * shape below needs the sparse component-addressed staging protocol instead.
 *
 * `info` is the hardware shader being compiled and `sw_stage` the API stage it
 * serves; `xfb_outputs` says whether the pipeline records transform feedback.
 */
static inline bool
hk_g17p_sparse_vertex_outputs(const struct shader_info *info,
                              mesa_shader_stage sw_stage, bool xfb_outputs)
{
   if (info->stage != MESA_SHADER_VERTEX)
      return false;

   return info->has_transform_feedback_varyings || xfb_outputs ||
          sw_stage == MESA_SHADER_GEOMETRY ||
          /* A raster-discard TES may write only point size, leaving the
           * four position components absent just like an XFB raster VS. */
          sw_stage == MESA_SHADER_TESS_EVAL ||
          info->next_stage == MESA_SHADER_GEOMETRY ||
          /* The hardware variant of a control-point vertex shader is never
           * bound -- the software variant runs in compute -- but it is still
           * compiled, and its UVS layout is as sparse as the geometry case
           * above. Without this the AGX3 packer refuses it at st_vary #4 and
           * takes every G17P tessellation pipeline with it. */
          info->next_stage == MESA_SHADER_TESS_CTRL ||
          /* A vertex shader may legally write no gl_Position at all. The
           * four-entry UVS position group stays reserved, so its user
           * varyings are a dense run based at AGX3_UVS_POSITION_ENTRIES --
           * the same shape as the XFB and geometry cases -- but there are
           * fewer than five of them, so the dense-run admission refuses the
           * store at st_vary #4. That is the whole
           * dEQP-VK.pipeline.*.no_position.basic.*.v0* family. */
          !(info->outputs_written & VARYING_BIT_POS);
}

/*
 * An hk_api shader maps 1:1 to a VkShader object. An hk_api_shader may contain
 * multiple hardware hk_shader's, built at shader compile time. This complexity
 * implements the legacy geometry pipeline and native sample-count variants.
 */
struct hk_api_shader {
   struct vk_shader vk;

   /* Non-NULL only for a fresh API wrapper borrowing immutable deserialized
    * code/data from the device's bounded binary cache. */
   struct hk_shader_binary_cache_entry *binary_cache_entry;

   /* Is this an internal passthrough geometry shader? */
   bool is_passthrough;

   /* Dynamic G17P sample-rate/coverage shaders have native 1x/2x/4x binaries. */
   bool g17p_sample_variants;
   /* Generic and independently allocated ordinary fragment programs. */
   bool g17p_ordinary_variants;

   /* v57d (ordinary variants only): the last draw's variant (0 none,
    * 1 generic, 2 ordinary) for bind-time prediction, and the
    * HK_G17P_ORDINARY_VARIANT_TRACE counters. */
   unsigned g17p_last_draw_ordinary;
   uint32_t g17p_trace_id;
   uint64_t g17p_trace_draws[2];
   uint64_t g17p_trace_reason_draws[16];
   uint64_t g17p_trace_wait_ns;
   uint64_t g17p_trace_last_print_ns;
   /* v57d: a region whose guard stays a runtime branch (a vertex shader
    * under HK_G17P_ORDINARY_VS, or a fragment without variants) keeps its
    * requirement list here so HK_G17P_ORDINARY_VARIANT_TRACE=3 can evaluate
    * on the CPU which arm the GPU guard should take. Not serialized. */
   struct hk_ordinary_descriptor_requirements *g17p_guard_requirements;

   struct hk_shader variants[];
};

void hk_shader_binary_cache_finish(struct hk_device *dev);
/* v57d HK_G17P_ORDINARY_VARIANT_TRACE (hk_cmd_draw.c). */
uint32_t hk_ordinary_trace_id(struct hk_api_shader *obj);
void hk_ordinary_trace_final(struct hk_api_shader *obj);

/* v56e: compile a lazily deferred variant now (waiting for its background
 * job if one is queued). VK_SUCCESS immediately for an eager variant. */
VkResult hk_shader_ensure_compiled(struct hk_device *dev, struct hk_shader *s);
bool hk_shader_is_compiled(const struct hk_shader *s);
/* Diagnostic HK_G17P_ORDINARY_FORCE_GENERIC=1: select the generic variant for
 * every draw (the lazy-path correctness gate). */
bool hk_g17p_ordinary_force_generic(void);
void hk_lazy_compile_finish(struct hk_device *dev);
bool hk_shader_ensure_link_ht_locked(struct hk_shader *shader);

static inline unsigned
hk_api_num_variants(const struct hk_api_shader *obj)
{
   if (obj->g17p_ordinary_variants) {
      if (obj->vk.stage == MESA_SHADER_VERTEX)
         return HK_VS_VARIANTS + 1;
      assert(obj->vk.stage == MESA_SHADER_FRAGMENT);
      return obj->g17p_sample_variants ? 6 : 2;
   }
   return obj->g17p_sample_variants ? 3 : hk_num_variants(obj->vk.stage);
}

static inline size_t
hk_api_shader_size(mesa_shader_stage stage, bool samples, bool ordinary)
{
   unsigned count = ordinary ? (stage == MESA_SHADER_VERTEX ? HK_VS_VARIANTS + 1 :
                                samples ? 6 : 2) :
                    samples ? 3 : hk_num_variants(stage);
   return sizeof(struct hk_api_shader) + count * sizeof(struct hk_shader) +
          (ordinary ? sizeof(struct hk_ordinary_descriptor_requirements) : 0);
}

static inline struct hk_ordinary_descriptor_requirements *
hk_api_ordinary_requirements(struct hk_api_shader *obj)
{
   assert(obj->g17p_ordinary_variants);
   return (void *)&obj->variants[hk_api_num_variants(obj)];
}

#define hk_foreach_variant(api_shader, var)                                    \
   for (struct hk_shader *var = api_shader->variants;                          \
        var < api_shader->variants + hk_api_num_variants(api_shader);          \
        ++var)

static const char *
hk_variant_name(struct hk_api_shader *obj, struct hk_shader *variant)
{
   unsigned i = variant - obj->variants;
   assert(i < hk_api_num_variants(obj));

   if (obj->g17p_ordinary_variants && obj->vk.stage == MESA_SHADER_VERTEX) {
      static const char *names[] = {"Hardware, generic descriptors", "Software",
                                    "Hardware, ordinary descriptors"};
      assert(i < ARRAY_SIZE(names));
      return names[i];
   } else if (obj->g17p_ordinary_variants) {
      if (obj->g17p_sample_variants) {
         static const char *names[] = {
            "1 sample, generic descriptors", "2 samples, generic descriptors",
            "4 samples, generic descriptors", "1 sample, ordinary descriptors",
            "2 samples, ordinary descriptors", "4 samples, ordinary descriptors",
         };
         return names[i];
      }
      return i ? "Ordinary descriptors" : "Generic descriptors";
   } else if (obj->g17p_sample_variants) {
      static const char *names[] = {"1 sample", "2 samples", "4 samples"};
      return names[i];
   } else if (hk_api_num_variants(obj) == 1) {
      return NULL;
   } else if (obj->vk.stage == MESA_SHADER_GEOMETRY) {
      assert(i < ARRAY_SIZE(hk_gs_variant_name));
      return hk_gs_variant_name[i];
   } else {
      assert(i < 2);
      return i == HK_VS_VARIANT_SW ? "Software" : "Hardware";
   }
}

static struct hk_shader *
hk_only_variant(struct hk_api_shader *obj)
{
   if (!obj)
      return NULL;

   assert(hk_api_num_variants(obj) == 1);
   return &obj->variants[0];
}

static struct hk_shader *
hk_any_variant(struct hk_api_shader *obj)
{
   if (!obj)
      return NULL;

   return &obj->variants[0];
}

static inline struct hk_shader *
hk_fs_variant(struct hk_api_shader *obj, unsigned samples)
{
   if (!obj)
      return NULL;

   assert(obj->vk.stage == MESA_SHADER_FRAGMENT);
   if (obj->g17p_sample_variants)
      assert(samples == 0 || samples == 1 || samples == 2 || samples == 4);
   if (obj->g17p_ordinary_variants)
      return &obj->variants[obj->g17p_sample_variants && samples > 1 ?
                            util_logbase2(samples) : 0];
   if (!obj->g17p_sample_variants)
      return hk_only_variant(obj);

   return &obj->variants[samples > 1 ? util_logbase2(samples) : 0];
}

static struct hk_shader *
hk_main_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_MAIN];
}

static struct hk_shader *
hk_count_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_COUNT];
}

static struct hk_shader *
hk_pre_gs_variant(struct hk_api_shader *obj)
{
   return &obj->variants[HK_GS_VARIANT_PRE];
}

#define HK_MAX_LINKED_USC_SIZE                                                 \
   (AGX_USC_PRESHADER_LENGTH + AGX_USC_FRAGMENT_PROPERTIES_LENGTH +            \
    AGX_USC_REGISTERS_LENGTH + AGX_USC_SHADER_LENGTH + AGX_USC_SHARED_LENGTH + \
    AGX_USC_SAMPLER_LENGTH + (AGX_USC_UNIFORM_LENGTH * 9))

struct hk_linked_shader {
   struct agx_linked_shader b;

   /* Immutable G17P compute-program templates keyed by uniform-range shape.
    * Entries are published lock-free while command buffers record in
    * parallel and are released with this linked executable. */
   void *g17p_compute_templates;

   /* Immutable AGX3 graphics-IOTO templates.  Per-draw command-buffer
    * programs relocate only their absolute uniform addresses. */
   void *g17p_graphics_templates;

   /* Directly owned immutable CPU setup templates for the compact-root path.
    * Vertex has one structural identity. Fragment entries are indexed by the
    * finite imageblock/blend/spill domain, never by draw-varying addresses. */
   void *g17p_graphics_vertex_setup;
   void *g17p_graphics_fragment_setups;

   /* True if the VS prolog uses software indexing, either for geom/tess or
    * adjacency primitives.
    */
   bool sw_indexing;

   /* True when sw_indexing is set purely because the topology carries
    * adjacency, i.e. the bound program is still the hardware vertex shader
    * and only fetches its own indices. The geometry/tessellation case, where
    * the software vertex shader runs as a compute kernel, is a different
    * shape and keeps sw_indexing with this clear. */
   bool sw_indexing_hw_vs;

   /* Absolute uniform-half base selected by the fragment epilog key for its
    * dynamic blend constants. Zero means the standard half4 ABI. */
   uint16_t g17p_blend_constant_base;
   uint16_t g17p_emrt_heap_base;

   /* The pipeline-statistics fragment prolog loads its counter address from
    * the descriptor root even when the application fragment shader does not. */
   bool g17p_prolog_uses_root;

   /* Distinct from hk_shader::counts due to addition of cf_binding_count, which
    * is delayed since it depends on cull distance.
    */
   struct agx_fragment_shader_word_0_packed fs_counts;

   /* Baked USC words to bind this linked shader */
   struct {
      uint8_t data[HK_MAX_LINKED_USC_SIZE];
      size_t size;
   } usc;
};

/* v54: `main_part`, when non-NULL, replaces main->b as the linked main part
 * (the inline-blend monolithic fragment program); main still supplies the
 * API shader's uniform and resource layout. */
struct hk_linked_shader *hk_fast_link_part(struct hk_device *dev, bool fragment,
                                           struct hk_shader *main,
                                           struct agx_shader_part *main_part,
                                           struct agx_shader_part *prolog,
                                           struct agx_shader_part *epilog,
                                           unsigned nr_samples_shaded);
struct hk_linked_shader *hk_fast_link(struct hk_device *dev, bool fragment,
                                      struct hk_shader *main,
                                      struct agx_shader_part *prolog,
                                      struct agx_shader_part *epilog,
                                      unsigned nr_samples_shaded);

extern const struct vk_device_shader_ops hk_device_shader_ops;

uint64_t
hk_physical_device_compiler_flags(const struct hk_physical_device *pdev);

/* The bounded native CProg shape both graphics stages admit: imported
 * uniform inputs stored in, and exactly filling, the preamble's TPS frame,
 * with a known staging footprint. */
static inline bool
hk_g17p_native_cprog_qualified(const struct agx_shader_info *info)
{
   return info->has_preamble && info->preamble_size &&
          info->agx3_preamble_staging_known && info->nr_preamble_gprs &&
          info->nr_preamble_gprs <= 256 &&
          info->preamble_scratch_size ==
             info->agx3_preamble_inputs.storage_bytes &&
          (!!info->agx3_preamble_inputs.count ==
           info->agx3_preamble_tps_inputs);
}

static inline nir_address_format
hk_buffer_addr_format(VkPipelineRobustnessBufferBehaviorEXT robustness,
                      bool null_descriptor)
{
   /* A null buffer descriptor has a zero range.  Preserve that range through
    * explicit-I/O lowering even when ordinary robustness is disabled so null
    * loads return zero and null stores are discarded before touching the
    * read-only zero page. */
   if (null_descriptor)
      return nir_address_format_64bit_bounded_global;

   switch (robustness) {
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT:
      return nir_address_format_64bit_global_32bit_offset;
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT:
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT:
      return nir_address_format_64bit_bounded_global;
   default:
      UNREACHABLE("Invalid robust buffer access behavior");
   }
}

bool hk_lower_uvs_index(nir_shader *s, mesa_shader_stage sw_stage,
                        unsigned nr_vbos, unsigned uniform_base);

bool
hk_nir_lower_descriptors(nir_shader *nir,
                         const struct vk_pipeline_robustness_state *rs,
                         uint32_t set_layout_count,
                         struct vk_descriptor_set_layout *const *set_layouts,
                         bool sparse_buffer_residency,
                         uint64_t sparse_ro_offset);

VkResult hk_compile_shader(struct hk_device *dev,
                           struct vk_shader_compile_info *info,
                           const struct vk_graphics_pipeline_state *state,
                           const struct vk_features *features,
                           const VkAllocationCallbacks *pAllocator,
                           struct hk_api_shader **shader_out);

void hk_preprocess_nir_internal(struct vk_physical_device *vk_pdev,
                                nir_shader *nir);

void hk_api_shader_destroy(struct vk_device *vk_dev,
                           struct vk_shader *vk_shader,
                           const VkAllocationCallbacks *pAllocator);

const nir_shader_compiler_options *
hk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                   UNUSED const struct vk_pipeline_robustness_state *rs);

struct spirv_to_nir_options
hk_get_spirv_options(struct vk_physical_device *vk_pdev,
                     UNUSED mesa_shader_stage stage,
                     const struct vk_pipeline_robustness_state *rs);

struct hk_api_shader *hk_meta_shader(struct hk_device *dev,
                                     hk_internal_builder_t builder, void *data,
                                     size_t data_size);

struct hk_passthrough_gs_key {
   /* Bit mask of outputs written by the VS/TES, to be passed through */
   uint64_t outputs;

   /* Clip/cull sizes, implies clip/cull written in output */
   uint8_t clip_distance_array_size;
   uint8_t cull_distance_array_size;

   /* Transform feedback buffer strides */
   uint8_t xfb_stride[MAX_XFB_BUFFERS];

   /* Decomposed primitive */
   enum mesa_prim prim;

   /* Transform feedback info. Must use hk_passthrough_gs_key_size to get the
    * key size */
   nir_xfb_info xfb_info;
};

static inline size_t
hk_passthrough_gs_key_size(uint16_t output_count)
{
   return (sizeof(struct hk_passthrough_gs_key) - sizeof(nir_xfb_info)) +
      nir_xfb_info_size(output_count);
}

void hk_nir_passthrough_gs(struct nir_builder *b, const void *key_);
