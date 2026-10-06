/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * D3D12AGX_DEBUG=api: trace every COM method call on the driver's objects.
 *
 * The HelloWorld samples under Proton failed with a bare E_INVALIDARG after
 * device creation and before any driver log line, and finding the call took
 * several Neo cycles. With the trace each object installs a generated
 * forwarding vtable (d12_api_trace.h, from gen_api_trace.py) that logs the
 * method, its arguments and its HRESULT, so one run names the failing call.
 */

#include "d12_private.h"

#include "util/log.h"
#include "util/u_atomic.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>

#include "d12_api_trace.h"

/* Monotonic call counter so lines from interleaved threads can be
 * sequenced and a crash log can be matched to the last recorded call. */
static uint32_t d12_api_trace_seq;

static void
d12_api_trace_vlog(const char *prefix, const char *method, const void *self,
                   const char *fmt, va_list ap, const char *suffix)
{
   char args[512];
   args[0] = '\0';
   if (fmt && *fmt)
      vsnprintf(args, sizeof(args), fmt, ap);
   mesa_logi("api %s%u %s(this=%p%s%s)%s", prefix,
             p_atomic_inc_return(&d12_api_trace_seq), method, self,
             args[0] ? ", " : "", args, suffix ? suffix : "");
}

void
d12_api_trace_call(const char *method, const void *self, const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   d12_api_trace_vlog("> ", method, self, fmt, ap, NULL);
   va_end(ap);
}

static const char *
d12_hresult_name(long hr)
{
   switch ((uint32_t)hr) {
   case 0x00000000: return "S_OK";
   case 0x00000001: return "S_FALSE";
   case 0x80004001: return "E_NOTIMPL";
   case 0x80004002: return "E_NOINTERFACE";
   case 0x80004003: return "E_POINTER";
   case 0x80004005: return "E_FAIL";
   case 0x8007000E: return "E_OUTOFMEMORY";
   case 0x80070057: return "E_INVALIDARG";
   case 0x887A0001: return "DXGI_ERROR_INVALID_CALL";
   case 0x887A0005: return "DXGI_ERROR_DEVICE_REMOVED";
   case 0x887A0006: return "DXGI_ERROR_DEVICE_HUNG";
   case 0x887A0007: return "DXGI_ERROR_DEVICE_RESET";
   case 0x887A0020: return "DXGI_ERROR_DRIVER_INTERNAL_ERROR";
   case 0x887A0022: return "DXGI_ERROR_NOT_CURRENTLY_AVAILABLE";
   case 0x887E0001: return "D3D12_ERROR_ADAPTER_NOT_FOUND";
   case 0x887E0002: return "D3D12_ERROR_DRIVER_VERSION_MISMATCH";
   case 0x887E0003: return "D3D12_ERROR_INVALID_REDIST";
   default: return NULL;
   }
}

void
d12_api_trace_hresult(const char *method, const void *self, long hr,
                      const char *fmt, ...)
{
   char suffix[96];
   const char *name = d12_hresult_name(hr);
   snprintf(suffix, sizeof(suffix), " -> 0x%08" PRIx32 "%s%s%s",
            (uint32_t)hr, name ? " (" : "", name ? name : "",
            name ? ")" : "");

   /* With the full trace on, the call line already carried the arguments;
    * the result line repeats only the method. Under api-fail the failing
    * call is the only line, so it gets the arguments too. */
   va_list ap;
   va_start(ap, fmt);
   d12_api_trace_vlog(FAILED(hr) ? "! " : "< ", method, self, fmt, ap,
                      suffix);
   va_end(ap);
}

const char *
d12_api_trace_guid(const GUID *guid, char buf[D12_API_TRACE_GUID_LEN])
{
   if (!guid) {
      snprintf(buf, D12_API_TRACE_GUID_LEN, "(null)");
      return buf;
   }
   snprintf(buf, D12_API_TRACE_GUID_LEN,
            "%08" PRIx32 "-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
            (uint32_t)guid->Data1, guid->Data2, guid->Data3, guid->Data4[0],
            guid->Data4[1], guid->Data4[2], guid->Data4[3], guid->Data4[4],
            guid->Data4[5], guid->Data4[6], guid->Data4[7]);
   return buf;
}

const void *
d12_api_trace_vtbl(enum d12_api_trace_iface iface, const void *real_vtbl)
{
   if (!d12_api_trace_enabled() && !d12_api_trace_fail_enabled())
      return real_vtbl;
   if (iface >= D12_API_TRACE_IFACE_COUNT)
      return real_vtbl;

   /* Every interface has exactly one real implementation table in this
    * driver, so the slot is written once and then only re-confirmed. */
   const void **slot = d12_api_trace_table[iface].real;
   const void *previous = p_atomic_cmpxchg(slot, NULL, real_vtbl);
   if (previous && previous != real_vtbl) {
      mesa_logw("api trace: interface %u has two real vtables (%p, %p); "
                "tracing only the first", (unsigned)iface, previous,
                real_vtbl);
      return real_vtbl;
   }
   return d12_api_trace_table[iface].trace_vtbl;
}
