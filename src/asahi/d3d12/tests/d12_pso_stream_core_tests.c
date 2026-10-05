/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "../d12_pso_stream_core.h"

#include <assert.h>
#include <stdio.h>

struct scalar_entry {
   uint32_t tag;
   uint32_t value;
};

struct pointer_entry {
   uint32_t tag;
   void *value;
};

int
main(void)
{
   _Alignas(void *) unsigned char stream[64];
   assert(d12_pso_stream_span_valid((uintptr_t)stream, sizeof(stream)));
   assert(!d12_pso_stream_span_valid(0, sizeof(stream)));
   assert(!d12_pso_stream_span_valid((uintptr_t)stream + 1, sizeof(stream)));
   assert(!d12_pso_stream_span_valid((uintptr_t)stream, 0));
   assert(!d12_pso_stream_span_valid(UINTPTR_MAX & ~(uintptr_t)7, 16));

   size_t offset, size;
   assert(d12_pso_stream_entry_size(sizeof(struct scalar_entry),
                                    sizeof(uint32_t), _Alignof(uint32_t),
                                    &offset, &size));
   assert(offset == offsetof(struct scalar_entry, value));
   assert(size == sizeof(struct scalar_entry));
   assert(d12_pso_stream_entry_size(sizeof(struct pointer_entry),
                                    sizeof(void *), _Alignof(void *),
                                    &offset, &size));
   assert(offset == offsetof(struct pointer_entry, value));
   assert(size == sizeof(struct pointer_entry));

   /* Reject every truncated prefix of a pointer entry, including padding. */
   for (size_t n = 0; n < sizeof(struct pointer_entry); n++)
      assert(!d12_pso_stream_entry_size(n, sizeof(void *), _Alignof(void *),
                                        &offset, &size));
   /* A byte-sized payload fits at byte 4, but the entry needs bytes 5..7. */
   for (size_t n = 5; n < sizeof(void *); n++)
      assert(!d12_pso_stream_entry_size(n, 1, 1, &offset, &size));
   assert(!d12_pso_stream_entry_size(SIZE_MAX, SIZE_MAX, 8, &offset, &size));
   assert(!d12_pso_stream_entry_size(64, 1, 0, &offset, &size));
   assert(!d12_pso_stream_entry_size(64, 1, 3, &offset, &size));

   for (unsigned mask = 0; mask < 8; mask++) {
      bool expected = mask == 1 || mask == 2 || mask == 4;
      assert(d12_pso_stream_pipeline_class_valid(mask & 1, mask & 2,
                                                 mask & 4) == expected);
   }
   puts("d12_pso_stream_core_tests: PASS");
   return 0;
}
