/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Honeykrisp bridge for the exact bounded AGX3 Ioto program composer.
 */

#pragma once

#include "asahi/compiler/agx3_ioto_uniform_program.h"
#include "asahi/lib/agx3_compute_program_core.h"
#include "asahi/lib/agx_ioto_program_core.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct agx_tilebuffer_layout;

struct hk_agx3_ioto_program_desc {
   uint64_t program_va;
   const void *body;
   size_t body_size;
   struct agx_ioto_state_tail tail;
   uint16_t profile_payload0;
   uint16_t profile_payload1;
};

/*
 * Byte-generation bridge only; success does not admit a physical device.
 * This exposes only the straight-line absent-preshader form and preserves
 * output and *written on every failure.
 */
bool hk_agx3_ioto_program_generate(
   enum agx_chip chip, const struct hk_agx3_ioto_program_desc *desc,
   void *output, size_t capacity, size_t *written);

/*
 * Compose an exact terminal suffix without PIXREL.  Inline/external suffixes
 * replace the absent 0e terminal; they are never appended after it.
 */
bool hk_agx3_ioto_program_generate_no_pixrel(
   enum agx_chip chip, const struct hk_agx3_ioto_program_desc *desc,
   const struct agx_ioto_preshader_desc *preshader, void *output,
   size_t capacity, size_t *written);

/* Honeykrisp bridge for G17P's default-profile program with no resource-load
 * records. */
bool hk_agx3_g17p_no_resource_program_generate(
   enum agx_chip chip, uint64_t program_va,
   const struct agx_ioto_state_tail_output *packed_tail, void *output,
   size_t capacity, size_t *written);

struct hk_agx3_g17p_graphics_program_desc {
   uint64_t program_va;
   uint64_t shader_va;
   const struct agx3_ioto_uniform_range *uniform_ranges;
   size_t uniform_range_count;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint16_t nr_gprs;
   bool compiler_d_valid;
   uint32_t compiler_d;
   bool fragment;
   bool writes_sample_mask;
};

#define HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE 6144u
#define HK_AGX3_G17P_GRAPHICS_MAX_RANGES 5u
#define HK_AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS 128u

bool hk_agx3_g17p_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size);

struct hk_agx3_g17p_graphics_program {
   uint8_t bytes[HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   uint16_t length;
};

#define HK_AGX3_G17P_BINDLESS_SAMPLER_BASE_INDEX 28u

struct hk_agx3_g17p_combined_sampler_heap {
   uint64_t descriptor_set_va;
   uint64_t sampler_heap_va;
   uint32_t texture_offset_B;
   uint16_t sampler_index;
   uint16_t sampler_count;
   uint16_t sampler_element_size_B;
};

/* Resolve two raw table pointers for the bounded texture0/sampler0 fragment
 * ABI. Upload these qwords into command-owned memory, then load its 16 bytes
 * into uniform halves 0..7. Descriptors themselves remain in their tables;
 * copying inline descriptor bytes to those uniforms is not this ABI.
 */
bool hk_agx3_g17p_combined_sampler_pointers(
   const struct hk_agx3_g17p_combined_sampler_heap *heap,
   uint64_t pointers[2]);

struct hk_agx3_g17p_sampled_storage_heap {
   struct hk_agx3_g17p_combined_sampler_heap sampled;
   uint64_t pbe_descriptor_set_va;
   uint32_t pbe_offset_B;
};

/* Resolve Apple's exact four-slot compute texture0/sample0 -> write-texture1
 * tuple: sampled texture, sampler, destination PBE, coordinate clamp. */
bool hk_agx3_g17p_sampled_storage_pointers(
   const struct hk_agx3_g17p_sampled_storage_heap *heap,
   uint64_t pointers[4]);

struct hk_agx3_g17p_indexed_storage_heap {
   uint64_t texture_heap_va;
   uint64_t sampler_heap_va;
   uint64_t pbe_descriptor_set_va;
   uint32_t pbe_offset_B;
};

bool hk_agx3_g17p_indexed_storage_pointers(
   const struct hk_agx3_g17p_indexed_storage_heap *heap,
   uint64_t pointers[4]);

/* Load one root descriptor followed by every descriptor-set pointer named by
 * the shader's pipeline layout. */
bool hk_agx3_g17p_root_range(uint16_t start_halfs, uint8_t set_count,
                            uint64_t root_va,
                            struct agx3_ioto_uniform_range *range);

/* Build the bounded G17P vertex or fragment Ioto program.  A zero-range
 * fragment uses the valid ordinary-FS PIXREL suffix; a zero-range vertex uses
 * the distinct compute/vertex profile suffix. Up to three dword-aligned
 * absolute uniform ranges add the CProg loads used by vertex attributes,
 * descriptor roots, or the bounded native texture/sampler pair above. */
bool hk_agx3_g17p_basic_graphics_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_graphics_program_desc *desc,
   struct hk_agx3_g17p_graphics_program *program);

enum hk_agx3_g17p_rt0_bg_op {
   HK_AGX3_G17P_RT0_BG_CLEAR,
   HK_AGX3_G17P_RT0_BG_RELOAD,
   HK_AGX3_G17P_RT0_BG_NONE,
};

#define HK_AGX3_G17P_RT0_CLEAR_PUSH_HALFS 12u
#define HK_AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS 14u

/* Resource spec for the bounded one-RT background shaders, not an ordinary
 * fragment shader or EOT. CLEAR/RELOAD write every live RT0 component; NONE
 * has no color output. The upper 17 bits are shader pass control, not a VA. */
bool hk_agx3_g17p_rt0_bg_resource_spec(
   enum agx_chip chip, enum hk_agx3_g17p_rt0_bg_op op, uint64_t *resource_spec);

struct hk_agx3_g17p_rt0_bg_program_desc {
   uint64_t program_va;
   uint64_t shader_va;
   uint64_t clear_value_va;
   uint64_t texture_pointer_va;
   uint64_t sampler_pointer_va;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint16_t nr_gprs;
   uint16_t staging_halfs;
   uint16_t clear_push_halfs;
   bool staging_known;
   enum hk_agx3_g17p_rt0_bg_op op;
};

struct hk_agx3_g17p_rt0_bg_program {
   uint8_t bytes[AGX_IOTO_G17P_RT0_RELOAD_PROGRAM_MAX_SIZE];
   uint16_t length;
};

/* Build the bounded clear or reload background program. Clear maps RT values
 * from uniform half4 onward; reload maps two 64-bit texture/sampler pointers
 * to halfs0..7, matching the native immediate-TEX ABI. */
bool hk_agx3_g17p_rt0_bg_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_rt0_bg_program_desc *desc,
   struct hk_agx3_g17p_rt0_bg_program *program);

bool hk_agx3_g17p_rt0_bg_shader_resources_match(
   enum hk_agx3_g17p_rt0_bg_op op, unsigned push_count, bool uses_txf);

struct hk_agx3_g17p_rt0_eot_program_desc {
   uint64_t program_va;
   uint64_t shader_va;
   /* Raw 32-byte-aligned GPU VA to the owned RT0 PBE32 descriptor. The
    * CProg publishes this pointer in uniform halves 0..3, not its pointee. */
   uint64_t pbe_argument_va;
   /* RT0 byte offset inside one imageblock sample (native kind5 argument).
    * Published as a zero-extended32-bit value in uniform halves4..5. */
   uint8_t imageblock_offset_B;
   uint8_t attachment_count;
   uint8_t imageblock_offset1_B;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint16_t nr_gprs;
   uint16_t staging_halfs;
};

struct hk_agx3_g17p_rt0_eot_program {
   uint8_t bytes[AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE];
   uint16_t length;
};

/* Build the bounded G17P EOT Ioto program. Two LDIMM records hold the
 * raw PBE table pointer in GPR halves 4..7 across the profile setup, then COPY
 * publishes it in uniform halves 0..3. A third LDIMM/COPY supplies
 * imageblock offset through CProg GPR8 into halves4..5. The exact MRT form
 * additionally publishes its second offset at halves8..9, retaining the
 * native metadata's intervening dword. No inline PBE payload is supplied. */
bool hk_agx3_g17p_rt0_eot_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_rt0_eot_program_desc *desc,
   struct hk_agx3_g17p_rt0_eot_program *program);

/* The bounded G17P render path handles one ordinary framebuffer layer. */
bool hk_agx3_g17p_single_layer_rendering(uint32_t view_mask,
                                        uint32_t layer_count);

/* Admit the one-sample, one-attachment tilebuffer used by that path. */
bool hk_agx3_g17p_rt0_tile_layout_supported(
   struct agx_tilebuffer_layout *tib);

/* Bounded Honeykrisp bridge for the exact G17P direct-add3 Ioto/CDM pair.
 * These functions expose a tested bring-up primitive only; Vulkan compute
 * capability advertisement remains unchanged. */
bool hk_agx3_g17p_direct_add3_program_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17p_direct_add3_program_desc *desc,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program);

bool hk_agx3_g17p_direct_add3_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink);

#define HK_AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE 6144u
#define HK_AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS  128u
#define HK_AGX3_G17P_MAX_ROOT_SETS            8u

struct hk_agx3_g17p_compute_program_desc {
   uint64_t program_va;
   uint64_t shader_va;
   const struct agx3_ioto_uniform_range *uniform_ranges;
   size_t uniform_range_count;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint16_t nr_gprs;
   bool compiler_d_valid;
   uint32_t compiler_d;
   uint32_t threadgroup_memory_size_B;
   struct agx3_cdm_direct_template_sources launch_sources;
};

struct hk_agx3_g17p_compute_program {
   uint8_t bytes[HK_AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE];
   uint16_t length;
   struct agx3_compute_program launch;
};

/* Build a dispatch-local Ioto program from dword-aligned root, descriptor and
 * constant ranges, then bind that exact allocation to the native A18 Pro CDM
 * launch state.  Spilling and indirect dispatch have separate contracts. */
bool hk_agx3_g17p_compute_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_compute_program_desc *desc,
   struct hk_agx3_g17p_compute_program *program);

bool hk_agx3_g17p_native_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink);

/* Byte bridge for the native main/grid-helper indirect record pair.  Helper
 * program and resource construction remain caller-owned. */
bool hk_agx3_g17p_native_indirect_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_g17p_native_indirect_dispatch *dispatch,
   struct agx3_compute_dispatch_sink *sink);

/*
 * Host-only admission bridge for the exact G17S minimal compute witness.
 * The explicit chip argument is intentionally rejected for every sibling
 * identity; this does not publish a Vulkan feature or admit a device.
 */
bool hk_agx3_g17s_minimal_compute_program_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc, void *output,
   size_t capacity, size_t *written, struct agx3_compute_program *program);

/* G17S-only mapped upload/launch-address join.  This is host-side admission
 * plumbing and does not publish a Vulkan feature or device capability. */
bool hk_agx3_g17s_minimal_compute_program_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program);

/* Exact G17S live-resource bridge.  The shared core owns all target,
 * relocation and transactional validation; this publishes no Vulkan cap. */
bool hk_agx3_g17s_minimal_compute_program_live_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx_ioto_g17s_minimal_compute_resources *resources,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program);

/* Append one live-admitted G17S launch plus its strong barrier.  This is a
 * bounded byte bridge only and does not publish a Vulkan capability. */
bool hk_agx3_g17s_minimal_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink);

#ifdef __cplusplus
}
#endif
