/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>

/* Debug and qualification knobs, read from the environment exactly once.
 * getenv() is a linear scan of the environment block; several of these were
 * evaluated on every draw, dispatch, submission and fence poll. */
struct d12_debug_flags {
   bool dump_entry_props;
   bool debug_bindings;
   bool dump_insts;
   bool dump_tex;
   bool print_nir;
   bool validate_bindings;
   bool dxbc_survey;
   bool dump_render_cmd;
   bool dump_read_set;
   bool dump_sync;
   bool dump_segs;
   bool dump_write_set;
   bool dump_scratch;
   bool dump_usc;
   bool dump_predication;
   bool validate_mrt;
   bool dump_descriptor_fixups;
   bool dump_indirect;
   bool dump_samplers;
   bool dump_gs_usc;
   bool dump_gs_vb;
   bool dump_so;
   bool dump_vb;
   bool dump_tess;
   bool dump_mesh;
   bool dump_sparse;
   bool dump_shaders;
   /* A/B control: flush the compute caches after every API dispatch as
    * the driver did before deferring the flush to the next barrier. */
   bool flush_each_dispatch;
   /* D3D12AGX_SUBMIT_SELF_WAIT: every submission carries an input sync on
    * the queue's own previous timeline point (the pre-t-1185 shape), for
    * A/B against the firmware-ordered default. */
   bool submit_self_wait;
   /* D3D12AGX_SUBMIT_INLINE_ALWAYS: submit from the calling thread even
    * while earlier submissions are in flight (the pre-t-1185 shape). */
   bool submit_inline_always;
   /* D3D12AGX_G17P_DISPATCH_NO_BARRIER: native G17P API dispatches drop
    * the trailing weak barrier word; the strong flush still follows at the
    * next barrier and at the segment end. Enabled by default; set to 0
    * to retain per-launch weak ordering for diagnostics. Neo only. */
   bool g17p_dispatch_no_barrier;

   /* Qualification opt-ins consumed by the G17P admission gates in
    * d12_cmd.c. Each is true only when the variable is exactly "1". */
   bool physical_baseline;
   bool physical_d32;
   bool shim_depth_metadata;
   bool shim_texture_metadata;

   /* D3D12AGX_DEBUG is a comma-separated list of named options:
    *   api       log every COM method call on the driver's objects with its
    *             scalar/pointer arguments, and every HRESULT result
    *   api-fail  log only the calls whose HRESULT failed
    * The trace is installed per object at creation (d12_api_trace.c), so
    * the variable has to be set before D3D12CreateDevice. */
   bool api_trace;
   bool api_trace_fail;
};

/* Filled once by d12_debug_slow(), which publishes d12_debug_ready last.
 * The accessor is inline so the hot paths (several calls per draw and
 * dispatch) pay a load and a branch, not a call into another unit. */
extern struct d12_debug_flags d12_debug_flags_storage;
extern bool d12_debug_ready;
const struct d12_debug_flags *d12_debug_slow(void);

static inline const struct d12_debug_flags *
d12_debug(void)
{
   if (__builtin_expect(__atomic_load_n(&d12_debug_ready, __ATOMIC_ACQUIRE),
                        1))
      return &d12_debug_flags_storage;
   return d12_debug_slow();
}

/* D3D12AGX_DEBUG=api support (d12_api_trace.c). Objects pass their real
 * vtable through d12_api_trace_vtbl() when they are created and install
 * whatever it returns: the real table when tracing is off, otherwise the
 * generated forwarding table for that interface. */
enum d12_api_trace_iface {
   D12_API_TRACE_DEVICE,
   D12_API_TRACE_LIST,
   D12_API_TRACE_QUEUE,
   D12_API_TRACE_RESOURCE,
   D12_API_TRACE_FENCE,
   D12_API_TRACE_DESCRIPTOR_HEAP,
   D12_API_TRACE_PIPELINE_STATE,
   D12_API_TRACE_ROOT_SIGNATURE,
   D12_API_TRACE_COMMAND_ALLOCATOR,
   D12_API_TRACE_HEAP,
   D12_API_TRACE_QUERY_HEAP,
   D12_API_TRACE_IFACE_COUNT,
};

const void *d12_api_trace_vtbl(enum d12_api_trace_iface iface,
                               const void *real_vtbl);

static inline bool
d12_api_trace_enabled(void)
{
   return d12_debug()->api_trace;
}

static inline bool
d12_api_trace_fail_enabled(void)
{
   return d12_debug()->api_trace_fail;
}

/* Logging helpers used by the generated wrappers and by the exported entry
 * points (D3D12CreateDevice and friends), which have no vtable. */
void d12_api_trace_call(const char *method, const void *self,
                        const char *fmt, ...);
void d12_api_trace_hresult(const char *method, const void *self,
                           long hr, const char *fmt, ...);

#define D12_API_TRACE_GUID_LEN 40
struct _GUID;
const char *d12_api_trace_guid(const struct _GUID *guid,
                               char buf[D12_API_TRACE_GUID_LEN]);
