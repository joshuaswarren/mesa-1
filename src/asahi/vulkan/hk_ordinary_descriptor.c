/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_ordinary_descriptor.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_shader.h"
#include <string.h>
#include "util/u_debug.h"

/* Diagnostic HK_G17P_ORDINARY_FORCE_GENERIC=1: select the generic variant for
 * every draw (the lazy-path correctness gate). */
DEBUG_GET_ONCE_BOOL_OPTION(hk_g17p_ordinary_force_generic_opt,
                           "HK_G17P_ORDINARY_FORCE_GENERIC", false)

bool
hk_g17p_ordinary_force_generic(void)
{
   return debug_get_option_hk_g17p_ordinary_force_generic_opt();
}

const char *
hk_ordinary_reason_name(enum hk_ordinary_reason reason)
{
   static const char *const names[HK_ORDINARY_REASON_COUNT] = {
      [HK_ORDINARY_OK] = "ok",
      [HK_ORDINARY_NO_PROOF] = "no-proof",
      [HK_ORDINARY_SET_RANGE] = "set-range",
      [HK_ORDINARY_PUSH_INACTIVE] = "push-inactive",
      [HK_ORDINARY_PUSH_SHADOWED] = "push-shadowed",
      [HK_ORDINARY_PUSH_LAYOUT_FLAGS] = "push-layout-flags",
      [HK_ORDINARY_PUSH_LAYOUT_HASH] = "push-layout-hash",
      [HK_ORDINARY_SET_UNBOUND] = "set-unbound",
      [HK_ORDINARY_SET_ADDRESS] = "set-address",
      [HK_ORDINARY_SET_LAYOUT] = "set-layout",
      [HK_ORDINARY_SET_BINDING] = "set-binding",
      [HK_ORDINARY_DESC_RANGE] = "desc-range",
      [HK_ORDINARY_SCALED_FORMAT] = "scaled-format",
      [HK_ORDINARY_CUSTOM_BORDER] = "custom-border",
      [HK_ORDINARY_FORCED_GENERIC] = "forced-generic",
      [HK_ORDINARY_PREDICTED] = "predicted",
   };
   return reason < HK_ORDINARY_REASON_COUNT && names[reason] ? names[reason]
                                                           : "?";
}

enum hk_ordinary_reason
hk_ordinary_descriptor_reason(
   const struct hk_ordinary_descriptor_requirements *proof,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   struct hk_push_descriptor_set *const *push_sets, uint32_t push_active,
   unsigned set_count, struct hk_ordinary_fs_selection *detail)
{
   if (!proof || !proof->count || proof->count > HK_ORDINARY_DESCRIPTOR_MAX_BINDINGS ||
       !sets || !addresses)
      return HK_ORDINARY_NO_PROOF;
   const struct hk_sampled_image_descriptor_layout native =
      hk_sampled_image_descriptor_layout(9);
   for (unsigned i = 0; i < proof->count; ++i) {
      const struct hk_ordinary_descriptor_binding *required = &proof->bindings[i];
      if (required->set >= set_count)
         return HK_ORDINARY_SET_RANGE;
      const uint8_t *data;
      uint32_t size;
      if (required->push) {
         if (!(push_active & BITFIELD_BIT(required->set)) || !push_sets ||
             !push_sets[required->set])
            return HK_ORDINARY_PUSH_INACTIVE;
         if (sets[required->set])
            return HK_ORDINARY_PUSH_SHADOWED;
         const struct hk_push_descriptor_set *push = push_sets[required->set];
         if (!(push->layout_flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR) ||
             (push->layout_flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT))
            return HK_ORDINARY_PUSH_LAYOUT_FLAGS;
         if (memcmp(push->layout_hash, required->layout_hash, sizeof(blake3_hash)))
            return HK_ORDINARY_PUSH_LAYOUT_HASH;
         data = push->data;
         size = sizeof(push->data);
      } else {
         const struct hk_descriptor_set *set = sets[required->set];
         if (!set || !set->mapped_ptr || !set->layout)
            return HK_ORDINARY_SET_UNBOUND;
         if (addresses[required->set] != set->addr)
            return HK_ORDINARY_SET_ADDRESS;
         if (set->layout->descriptor_generation != 9 ||
             required->binding >= set->layout->binding_count ||
             (set->layout->vk.flags &
              (VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR |
               VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT)))
            return HK_ORDINARY_SET_LAYOUT;
         const struct hk_descriptor_set_binding_layout *binding =
            &set->layout->binding[required->binding];
         if (binding->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
             binding->array_size != 1 || binding->flags ||
             binding->offset != required->offset)
            return HK_ORDINARY_SET_BINDING;
         data = set->mapped_ptr;
         size = set->size;
      }

      /* Match the GPU predicate. Flags-zero descriptor contents may be
       * consumed during binding/recording; changing a binding used by a
       * recorded command invalidates that command buffer. */
      uint64_t scaled_offset = (uint64_t)required->offset + native.texture + 32 +
         offsetof(struct hk_sampled_image_descriptor_sw, scaled_conversion);
      uint64_t border_offset = (uint64_t)required->offset + native.clamp_0_sampler_index;
      if (scaled_offset + 2 > size || border_offset + 2 > size)
         return HK_ORDINARY_DESC_RANGE;
      uint16_t scaled;
      int16_t border;
      memcpy(&scaled, data + scaled_offset, sizeof(scaled));
      memcpy(&border, data + border_offset, sizeof(border));
      if (scaled != 0 || border >= 0) {
         if (detail) {
            detail->detail_scaled = scaled;
            detail->detail_border = border;
            detail->detail_binding = required->binding;
         }
         return scaled != 0 ? HK_ORDINARY_SCALED_FORMAT
                            : HK_ORDINARY_CUSTOM_BORDER;
      }
   }
   return HK_ORDINARY_OK;
}

static bool
hk_ordinary_descriptor_sets_match_impl(
   const struct hk_ordinary_descriptor_requirements *proof,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   struct hk_push_descriptor_set *const *push_sets, uint32_t push_active,
   unsigned set_count)
{
   return hk_ordinary_descriptor_reason(proof, sets, addresses, push_sets,
                                        push_active, set_count, NULL) ==
          HK_ORDINARY_OK;
}

bool
hk_ordinary_descriptor_sets_match(
   const struct hk_ordinary_descriptor_requirements *proof,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   unsigned set_count)
{
   return hk_ordinary_descriptor_sets_match_impl(
      proof, sets, addresses, NULL, 0, set_count);
}

struct hk_shader *
hk_ordinary_descriptor_select_push_slow(
   struct hk_ordinary_fs_selection *cache, struct hk_api_shader *obj,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   struct hk_push_descriptor_set *const *push_sets, uint32_t push_active,
   unsigned set_count, uint64_t generation, unsigned samples, bool *changed)
{
   assert(obj && obj->g17p_ordinary_variants && changed);
   *changed = false;
   if (cache->valid && cache->api == obj &&
       cache->descriptor_generation == generation && cache->samples == samples)
      return cache->selected;

   struct hk_ordinary_fs_selection detail = {0};
   enum hk_ordinary_reason reason = hk_g17p_ordinary_force_generic()
      ? HK_ORDINARY_FORCED_GENERIC
      : hk_ordinary_descriptor_reason(hk_api_ordinary_requirements(obj), sets,
                                      addresses, push_sets, push_active,
                                      set_count, &detail);
   bool ordinary = reason == HK_ORDINARY_OK;
   assert(!obj->g17p_sample_variants ||
          samples == 0 || samples == 1 || samples == 2 || samples == 4);
   unsigned sample_index = obj->g17p_sample_variants && samples > 1 ?
      util_logbase2(samples) : 0;
   unsigned ordinary_offset = obj->g17p_sample_variants ? 3 : 1;
   struct hk_shader *selected =
      &obj->variants[sample_index + (ordinary ? ordinary_offset : 0)];
   *changed = !cache->valid || cache->selected != selected;
   *cache = (struct hk_ordinary_fs_selection){
      .api = obj, .selected = selected, .descriptor_generation = generation,
      .samples = samples, .valid = true, .reason = reason,
      .detail_scaled = detail.detail_scaled,
      .detail_border = detail.detail_border,
      .detail_binding = detail.detail_binding,
   };
   return selected;
}

struct hk_shader *
hk_ordinary_descriptor_select(
   struct hk_ordinary_fs_selection *cache, struct hk_api_shader *obj,
   struct hk_descriptor_set *const *sets, const uint64_t *addresses,
   unsigned set_count, uint64_t generation, unsigned samples, bool *changed)
{
   return hk_ordinary_descriptor_select_push(
      cache, obj, sets, addresses, NULL, 0,
      set_count, generation, samples, changed);
}
