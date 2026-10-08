/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_descriptor_set_layout.h"
#include "hk_sampler.h"

void
hk_descriptor_set_layout_hash(struct hk_descriptor_set_layout *layout)
{
   struct mesa_blake3 blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);

#define BLAKE3_UPDATE_VALUE(x) \
   _mesa_blake3_update(&blake3_ctx, &(x), sizeof(x))
   /* Ordinary shader proofs distinguish pushed CPU snapshots from allocated
    * sets. Binding layouts alone cannot identify that proof domain. The
    * Vulkan shader cache consumes this hash, so separate both kinds here. */
   const bool push = layout->vk.flags &
      VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
   BLAKE3_UPDATE_VALUE(push);
   BLAKE3_UPDATE_VALUE(layout->descriptor_generation);
   BLAKE3_UPDATE_VALUE(layout->native_sampler_indices);
   BLAKE3_UPDATE_VALUE(layout->non_variable_descriptor_buffer_size);
   BLAKE3_UPDATE_VALUE(layout->vk.dynamic_descriptor_count);
   BLAKE3_UPDATE_VALUE(layout->binding_count);

   for (uint32_t b = 0; b < layout->binding_count; b++) {
      BLAKE3_UPDATE_VALUE(layout->binding[b].type);
      BLAKE3_UPDATE_VALUE(layout->binding[b].flags);
      BLAKE3_UPDATE_VALUE(layout->binding[b].array_size);
      BLAKE3_UPDATE_VALUE(layout->binding[b].offset);
      BLAKE3_UPDATE_VALUE(layout->binding[b].stride);
      BLAKE3_UPDATE_VALUE(layout->binding[b].dynamic_buffer_index);

      if (layout->binding[b].immutable_samplers != NULL) {
         for (uint32_t i = 0; i < layout->binding[b].array_size; i++) {
            const struct hk_sampler *sampler =
               layout->binding[b].immutable_samplers[i];
            /* These shader-visible values were initialized with the sampler. */
            if (sampler != NULL && sampler->vk.ycbcr_conversion != NULL) {
               BLAKE3_UPDATE_VALUE(sampler->ycbcr);
               BLAKE3_UPDATE_VALUE(sampler->min_filter);
               BLAKE3_UPDATE_VALUE(sampler->mag_filter);
               BLAKE3_UPDATE_VALUE(sampler->mip_lod_bias);
            }
         }
      }
   }
#undef BLAKE3_UPDATE_VALUE
   _mesa_blake3_final(&blake3_ctx, layout->vk.blake3);
}
