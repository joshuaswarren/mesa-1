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

   /* Qualification opt-ins consumed by the G17P admission gates in
    * d12_cmd.c. Each is true only when the variable is exactly "1". */
   bool physical_baseline;
   bool physical_d32;
   bool shim_depth_metadata;
   bool shim_texture_metadata;
};

const struct d12_debug_flags *d12_debug(void);
