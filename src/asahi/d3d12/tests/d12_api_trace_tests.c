/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */

#include "../d12_debug.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

int
main(void)
{
   /* The tables need only distinct addresses: no COM method is invoked. */
   static const char real[D12_API_TRACE_IFACE_COUNT] = {0};
   static const char foreign = 0;
   const void *wrapped[D12_API_TRACE_IFACE_COUNT];
   const bool trace = d12_api_trace_enabled() || d12_api_trace_fail_enabled();

   for (unsigned i = 0; i < D12_API_TRACE_IFACE_COUNT; ++i) {
      assert(!d12_api_trace_vtbl_matches(i, NULL, &real[i]));
      assert(!d12_api_trace_vtbl_matches(i, &foreign, &real[i]));
      wrapped[i] = d12_api_trace_vtbl(i, &real[i]);
      assert((wrapped[i] != &real[i]) == trace);
      assert(d12_api_trace_vtbl_matches(i, &real[i], &real[i]));
      assert(d12_api_trace_vtbl_matches(i, wrapped[i], &real[i]));
      assert(!d12_api_trace_vtbl_matches(i, wrapped[i], &foreign));
      assert(d12_api_trace_vtbl(i, &real[i]) == wrapped[i]);
   }
   for (unsigned i = 0; i < D12_API_TRACE_IFACE_COUNT; ++i) {
      for (unsigned j = 0; j < D12_API_TRACE_IFACE_COUNT; ++j) {
         if (i != j)
            assert(!d12_api_trace_vtbl_matches(i, wrapped[j], &real[i]));
      }
      /* A second implementation falls back to its own table. It cannot
       * acquire the first implementation's wrapper identity. */
      assert(d12_api_trace_vtbl(i, &foreign) == &foreign);
      assert(d12_api_trace_vtbl_matches(i, &foreign, &foreign));
      assert(!d12_api_trace_vtbl_matches(i, wrapped[i], &foreign));
   }
   assert(!d12_api_trace_vtbl_matches(-1, &foreign, &real[0]));
   assert(!d12_api_trace_vtbl_matches(D12_API_TRACE_IFACE_COUNT,
                                      &foreign, &real[0]));
   puts("PASS trace identity: all interfaces, foreign tables, repeated registration");
   return 0;
}
