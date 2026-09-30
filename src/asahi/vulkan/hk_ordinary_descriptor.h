/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "util/mesa-blake3.h"

#define HK_ORDINARY_DESCRIPTOR_MAX_BINDINGS 16

struct hk_ordinary_descriptor_binding {
   uint32_t set;
   uint32_t binding;
   uint32_t offset;
   bool push;
   blake3_hash layout_hash;
};

/* Requirements belong to the separately compiled ordinary variant, not to
 * the generic shader. Every entry must be proven at command recording. */
struct hk_ordinary_descriptor_requirements {
   uint32_t count;
   uint32_t set_mask;
   struct hk_ordinary_descriptor_binding bindings[HK_ORDINARY_DESCRIPTOR_MAX_BINDINGS];
};

struct nir_shader;
struct vk_descriptor_set_layout;
struct hk_descriptor_set;
struct hk_push_descriptor_set;
struct hk_api_shader;
struct hk_shader;

/* v57d: why a draw's descriptors did not satisfy the ordinary proof (the
 * first failing check; HK_ORDINARY_OK when they did). */
enum hk_ordinary_reason {
   HK_ORDINARY_OK = 0,
   HK_ORDINARY_NO_PROOF,          /* shader has no usable requirement list */
   HK_ORDINARY_SET_RANGE,         /* required set index >= bound set count */
   HK_ORDINARY_PUSH_INACTIVE,     /* push set required but not pushed */
   HK_ORDINARY_PUSH_SHADOWED,     /* a descriptor set is bound over the push set */
   HK_ORDINARY_PUSH_LAYOUT_FLAGS, /* pushed layout not a plain push layout */
   HK_ORDINARY_PUSH_LAYOUT_HASH,  /* pushed with a different set layout */
   HK_ORDINARY_SET_UNBOUND,       /* required set not bound / not mapped */
   HK_ORDINARY_SET_ADDRESS,       /* bound address differs from the set */
   HK_ORDINARY_SET_LAYOUT,        /* layout generation/flags/binding mismatch */
   HK_ORDINARY_SET_BINDING,       /* binding type/array/offset mismatch */
   HK_ORDINARY_DESC_RANGE,        /* descriptor bytes outside the set */
   HK_ORDINARY_SCALED_FORMAT,     /* view needs scaled-format conversion */
   HK_ORDINARY_CUSTOM_BORDER,     /* sampler needs custom-border emulation */
   HK_ORDINARY_FORCED_GENERIC,    /* HK_G17P_ORDINARY_FORCE_GENERIC */
   HK_ORDINARY_PREDICTED,         /* bind-time prediction, not yet proven */
   HK_ORDINARY_REASON_COUNT,
};

const char *hk_ordinary_reason_name(enum hk_ordinary_reason reason);

struct hk_ordinary_fs_selection {
   struct hk_api_shader *api;
   struct hk_shader *selected;
   uint64_t descriptor_generation;
   unsigned samples;
   bool valid;
   /* v57d: outcome of the proof behind `selected`, and for a failed proof the
    * binding's scaled_conversion / clamp-0 values (trace only). */
   enum hk_ordinary_reason reason;
   uint16_t detail_scaled;
   int16_t detail_border;
   uint8_t detail_binding;
};

/* v57d: the first failing check for these descriptors (HK_ORDINARY_OK if the
 * proof holds). hk_ordinary_descriptor_sets_match() is reason == OK. */
enum hk_ordinary_reason hk_ordinary_descriptor_reason(
   const struct hk_ordinary_descriptor_requirements *proof,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   struct hk_push_descriptor_set *const *push_sets, uint32_t push_active,
   unsigned set_count, struct hk_ordinary_fs_selection *detail);

bool hk_ordinary_descriptor_sets_match(
   const struct hk_ordinary_descriptor_requirements *proof,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   unsigned set_count);

struct hk_shader *hk_ordinary_descriptor_select(
   struct hk_ordinary_fs_selection *cache, struct hk_api_shader *obj,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   unsigned set_count, uint64_t generation, unsigned samples, bool *changed);

struct hk_shader *hk_ordinary_descriptor_select_push_slow(
   struct hk_ordinary_fs_selection *cache, struct hk_api_shader *obj,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   struct hk_push_descriptor_set *const *push_sets, uint32_t push_active,
   unsigned set_count, uint64_t generation, unsigned samples, bool *changed);

/* The selection is re-queried several times per draw; answer the cached case
 * inline. */
static inline struct hk_shader *
hk_ordinary_descriptor_select_push(
   struct hk_ordinary_fs_selection *cache, struct hk_api_shader *obj,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   struct hk_push_descriptor_set *const *push_sets, uint32_t push_active,
   unsigned set_count, uint64_t generation, unsigned samples, bool *changed)
{
   if (cache->valid && cache->api == obj &&
       cache->descriptor_generation == generation && cache->samples == samples) {
      *changed = false;
      return cache->selected;
   }

   return hk_ordinary_descriptor_select_push_slow(
      cache, obj, sets, addresses, push_sets, push_active, set_count,
      generation, samples, changed);
}

/* v57 (neo-mc-shaders-mesa-v57), both opt-in and not hardware-qualified:
 *   HK_ORDINARY_V57_VERTEX: a hardware vertex shader forms the region too.
 *     Vertex objects have no ordinary variants, so the guard stays a runtime
 *     uniform branch (HK_ORDINARY_GUARD_DYNAMIC); a single sample is admitted.
 *   HK_ORDINARY_V57_TXF: texelFetch (txf) on a fixed combined-image binding is
 *     admitted; its guard is the same kind==0 && no-custom-border proof. */
#define HK_ORDINARY_V57_VERTEX (1u << 0)
#define HK_ORDINARY_V57_TXF (1u << 1)

bool hk_nir_ordinary_descriptor_regions(
   struct nir_shader *nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *layouts, bool allow_single_sample,
   unsigned v57_flags,
   struct hk_ordinary_descriptor_requirements *requirements);

enum hk_ordinary_guard_mode {
   HK_ORDINARY_GUARD_DYNAMIC,
   HK_ORDINARY_GUARD_TRUE,
   HK_ORDINARY_GUARD_FALSE,
};

bool hk_nir_lower_ordinary_descriptor_guard(struct nir_shader *nir,
                                           enum hk_ordinary_guard_mode mode);
