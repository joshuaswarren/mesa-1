/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Stream entries use the native struct layout of a 32-bit tag followed by
 * the payload, with the complete entry padded to pointer alignment. Check
 * integer spans before forming pointers into an application's byte stream.
 */
static inline bool
d12_pso_stream_span_valid(uintptr_t address, size_t size)
{
   return address && !(address % sizeof(void *)) && size &&
          size <= UINTPTR_MAX - address;
}

static inline bool
d12_pso_stream_entry_size(size_t remaining, size_t payload_size,
                          size_t payload_alignment, size_t *payload_offset,
                          size_t *entry_size)
{
   if (!payload_alignment || payload_alignment > sizeof(void *) ||
       (payload_alignment & (payload_alignment - 1)))
      return false;

   size_t offset = (sizeof(uint32_t) + payload_alignment - 1) &
                   ~(payload_alignment - 1);
   if (offset > remaining || payload_size > remaining - offset)
      return false;

   size_t size = offset + payload_size;
   if (size > SIZE_MAX - (sizeof(void *) - 1))
      return false;
   size = (size + sizeof(void *) - 1) & ~(sizeof(void *) - 1);
   if (size > remaining)
      return false;

   *payload_offset = offset;
   *entry_size = size;
   return true;
}

/* VS, MS and CS each select a pipeline class. Other graphics defaults may
 * accompany a compute stream, but two selectors cannot coexist. */
static inline bool
d12_pso_stream_pipeline_class_valid(bool vertex, bool mesh, bool compute)
{
   return (unsigned)vertex + (unsigned)mesh + (unsigned)compute == 1;
}
