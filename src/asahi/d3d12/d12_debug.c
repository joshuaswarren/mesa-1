/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "d12_debug.h"

#include "c11/threads.h"

#include <stdlib.h>
#include <string.h>

static struct d12_debug_flags d12_debug_flags;
static once_flag d12_debug_once = ONCE_FLAG_INIT;

static bool
d12_env_set(const char *name)
{
   return getenv(name) != NULL;
}

static bool
d12_env_is_one(const char *name)
{
   const char *value = getenv(name);
   return value && !strcmp(value, "1");
}

static void
d12_debug_init(void)
{
   struct d12_debug_flags *f = &d12_debug_flags;
   f->dump_entry_props = d12_env_set("D3D12AGX_DUMP_ENTRY_PROPS");
   f->debug_bindings = d12_env_set("D3D12AGX_DEBUG_BINDINGS");
   f->dump_insts = d12_env_set("D3D12AGX_DUMP_INSTS");
   f->dump_tex = d12_env_set("D3D12AGX_DUMP_TEX");
   f->print_nir = d12_env_set("D3D12AGX_PRINT_NIR");
   f->validate_bindings = d12_env_set("D3D12AGX_VALIDATE_BINDINGS");
   f->dxbc_survey = d12_env_set("D3D12AGX_DXBC_SURVEY");
   f->dump_render_cmd = d12_env_set("D3D12AGX_DUMP_RENDER_CMD");
   f->dump_read_set = d12_env_set("D3D12AGX_DUMP_READ_SET");
   f->dump_sync = d12_env_set("D3D12AGX_DUMP_SYNC");
   f->dump_segs = d12_env_set("D3D12AGX_DUMP_SEGS");
   f->dump_write_set = d12_env_set("D3D12AGX_DUMP_WRITE_SET");
   f->dump_scratch = d12_env_set("D3D12AGX_DUMP_SCRATCH");
   f->dump_usc = d12_env_set("D3D12AGX_DUMP_USC");
   f->dump_predication = d12_env_set("D3D12AGX_DUMP_PREDICATION");
   f->validate_mrt = d12_env_set("D3D12AGX_VALIDATE_MRT");
   f->dump_descriptor_fixups = d12_env_set("D3D12AGX_DUMP_DESCRIPTOR_FIXUPS");
   f->dump_indirect = d12_env_set("D3D12AGX_DUMP_INDIRECT");
   f->dump_samplers = d12_env_set("D3D12AGX_DUMP_SAMPLERS");
   f->dump_gs_usc = d12_env_set("D3D12AGX_DUMP_GS_USC");
   f->dump_gs_vb = d12_env_set("D3D12AGX_DUMP_GS_VB");
   f->dump_so = d12_env_set("D3D12AGX_DUMP_SO");
   f->dump_vb = d12_env_set("D3D12AGX_DUMP_VB");
   f->dump_tess = d12_env_set("D3D12AGX_DUMP_TESS");
   f->dump_mesh = d12_env_set("D3D12AGX_DUMP_MESH");
   f->dump_sparse = d12_env_set("D3D12AGX_DUMP_SPARSE");
   f->dump_shaders = d12_env_set("D3D12AGX_DUMP_SHADERS");
   f->physical_baseline = d12_env_is_one("AURORA_D3D12_PHYSICAL_BASELINE");
   f->physical_d32 = d12_env_is_one("AURORA_D3D12_PHYSICAL_D32");
   f->shim_depth_metadata = d12_env_is_one("AURORA_D3D12_SHIM_DEPTH_METADATA");
   f->shim_texture_metadata =
      d12_env_is_one("AURORA_D3D12_SHIM_TEXTURE_METADATA");
}

const struct d12_debug_flags *
d12_debug(void)
{
   call_once(&d12_debug_once, d12_debug_init);
   return &d12_debug_flags;
}
