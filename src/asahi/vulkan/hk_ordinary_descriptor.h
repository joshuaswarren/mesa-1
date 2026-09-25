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

struct hk_ordinary_fs_selection {
   struct hk_api_shader *api;
   struct hk_shader *selected;
   uint64_t descriptor_generation;
   unsigned samples;
   bool valid;
};

bool hk_ordinary_descriptor_sets_match(
   const struct hk_ordinary_descriptor_requirements *proof,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   unsigned set_count);

struct hk_shader *hk_ordinary_descriptor_select(
   struct hk_ordinary_fs_selection *cache, struct hk_api_shader *obj,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   unsigned set_count, uint64_t generation, unsigned samples, bool *changed);

struct hk_shader *hk_ordinary_descriptor_select_push(
   struct hk_ordinary_fs_selection *cache, struct hk_api_shader *obj,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   struct hk_push_descriptor_set *const *push_sets, uint32_t push_active,
   unsigned set_count, uint64_t generation, unsigned samples, bool *changed);

bool hk_nir_ordinary_descriptor_regions(
   struct nir_shader *nir, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *layouts, bool allow_single_sample,
   struct hk_ordinary_descriptor_requirements *requirements);

enum hk_ordinary_guard_mode {
   HK_ORDINARY_GUARD_DYNAMIC,
   HK_ORDINARY_GUARD_TRUE,
   HK_ORDINARY_GUARD_FALSE,
};

bool hk_nir_lower_ordinary_descriptor_guard(struct nir_shader *nir,
                                           enum hk_ordinary_guard_mode mode);
