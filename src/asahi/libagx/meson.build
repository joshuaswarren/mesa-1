# Copyright 2024 Valve Corporation
# SPDX-License-Identifier: MIT

libagx_shader_files = files(
  'compression.cl',
  'copy.cl',
  'draws.cl',
  'geometry.cl',
  'query.cl',
  'tessellation.cl',
  'tessellator.cl',
  'texture.cl',
  'helper.cl',
)

libagx_spv = custom_target(
  input : libagx_shader_files,
  output : 'libagx.spv',
  command : [
    prog_mesa_clc, '-o', '@OUTPUT@', '--depfile', '@DEPFILE@',
    libagx_shader_files, '--',
    '-I' + join_paths(meson.project_source_root(), 'include'),
    '-I' + join_paths(meson.project_source_root(), 'src/compiler/libcl'),
    '-I' + join_paths(meson.project_source_root(), 'src/poly/cl'),
    '-I' + join_paths(meson.current_source_dir(), '.'),
    '-I' + join_paths(meson.current_source_dir(), '../../'),
    # BUILD-STATUS.md stage-1 (native mesa_clc/asahi_clc bring-up): the
    # G13/G15/G17 fail-closed chip-detection header, included as
    # "lib/agx_chip.h" from libagx_dgc.h, lives at src/asahi/lib/agx_chip.h.
    # The '../../' entry above only reaches src/ (two levels up from
    # src/asahi/libagx), which resolves every OTHER quoted include this
    # custom_target needs ("compiler/libcl/libcl.h" etc.) but not this one,
    # which needs src/asahi/ (one level up). This is a real, pre-existing
    # gap in this tree's own libagx/meson.build -- VERIFIED by hand-driving
    # mesa_clc with an extra -I: without it, `fatal error: 'lib/agx_chip.h'
    # file not found`; with it, libagx.spv builds clean. Not part of the
    # honeykrisp Windows port itself (host_machine.system() doesn't gate
    # this), just never previously exercised for this CL compile.
    '-I' + join_paths(meson.current_source_dir(), '../'),
    '-I' + join_paths(meson.current_source_dir(), 'shaders'),
    '-I' + join_paths(meson.current_build_dir(), '../genxml'),
    cl_args,
    # libagx is a shipped, precompiled, cached artifact, so build it with
    # release semantics whatever the enclosing build type is. libcl's assert()
    # expands to printf + abort, and AGX3 refuses any shader still carrying
    # load_printf_buffer_address, because the printf buffer is process-local
    # and baking its address into cached machine code would poison the cache.
    # Without this, which precompiled kernels exist on G17P depends on
    # -Db_ndebug: a debugoptimized build silently loses primitive restart and
    # the blit/fill kernels. Deliberate overflow guards that must survive a
    # release build do not use assert() and are unaffected.
    '-DNDEBUG',
    ],
  depends : [agx_pack],
  depfile : 'libagx_shaders.h.d',
)

libagx = custom_target(
  input : libagx_spv,
  output : ['libagx.cpp', 'libagx.h'],
  command : [prog_vtn_bindgen2, libagx_spv, '@OUTPUT0@', '@OUTPUT1@'],
)

idep_libagx = declare_dependency(
  sources : [libagx],
  include_directories : include_directories('.'),
)
