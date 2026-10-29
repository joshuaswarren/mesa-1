/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * Copyright 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "compiler/libcl/libcl.h"

#ifdef __OPENCL_VERSION__
#include "compiler/libcl/libcl_vk.h"
#else
#include <stdbool.h>
#include "vulkan/vulkan_core.h"
#endif

/* Availability of one query: a 32-bit word, or with availability_shift 1 a
 * 64-bit slot that the kernel may fill with the retirement timestamp of the
 * render that ended the query. Any nonzero bit of the slot means available.
 */
static inline bool
libagx_query_available(GLOBAL volatile uint32_t *availability,
                       uint32_t availability_shift, uint32_t query)
{
   uint32_t word = query << availability_shift;
   uint32_t value = availability[word];
   if (availability_shift)
      value |= availability[word + 1];
   return value != 0;
}

/* Shared with the host tests. One invocation owns a complete query tuple,
 * including its single availability slot. Keep loads volatile: WAIT must
 * reload availability, and results must be read after observing availability.
 * The caller must order the producer before launching a waiting consumer.
 */
static inline void
libagx_copy_query_g17p_result(GLOBAL volatile uint32_t *availability,
                              GLOBAL volatile uint64_t *results,
                              GLOBAL uint16_t *oq_index, uint64_t dst_addr,
                              uint64_t dst_stride, uint32_t first_query,
                              uint32_t flags, uint16_t reports_per_query,
                              uint32_t availability_shift, uint32_t invocation)
{
   uint32_t query = first_query + invocation;
   uint32_t index = oq_index ? oq_index[query] : query;
   GLOBAL volatile uint64_t *report =
      results + (uint64_t)index * reports_per_query;
   uint64_t dst = dst_addr + (uint64_t)invocation * dst_stride;

   bool available;
   do {
      available = availability
                     ? libagx_query_available(availability,
                                              availability_shift, query)
                     : report[0] != UINT64_MAX;
   } while (!available && (flags & VK_QUERY_RESULT_WAIT_BIT));

   if (available || (flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
      for (unsigned i = 0; i < reports_per_query; ++i) {
         uint64_t value = report[i];
         if (flags & VK_QUERY_RESULT_64_BIT)
            ((GLOBAL uint64_t *)dst)[i] = value;
         else
            ((GLOBAL uint32_t *)dst)[i] = value;
      }
   }

   if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
      if (flags & VK_QUERY_RESULT_64_BIT)
         ((GLOBAL uint64_t *)dst)[reports_per_query] = available;
      else
         ((GLOBAL uint32_t *)dst)[reports_per_query] = available;
   }
}
