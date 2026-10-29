# Copyright 2018 Rob Clark
# Copyright 2019 Collabora, Ltd.
# Copyright 2017 Intel Corporation
# SPDX-License-Identifier: MIT

libasahi_lib_files = files(
  'agx3_cdm_decode_core.c',
  'agx3_cdm_decode_core.h',
  'agx3_cdm_direct_launch_semantic_core.c',
  'agx3_cdm_direct_launch_semantic_core.h',
  'agx3_compute_program_core.c',
  'agx3_compute_program_core.h',
  'agx3_g17p_grid_helper_core.c',
  'agx3_g17p_grid_helper_core.h',
  'agx3_code_container_core.c',
  'agx3_code_container_core.h',
  'agx_ioto_core.c',
  'agx_ioto_core.h',
  'agx_ioto_load_core.c',
  'agx_ioto_load_core.h',
  'agx_ioto_preshader_core.c',
  'agx_ioto_preshader_core.h',
  'agx_ioto_profile_core.c',
  'agx_ioto_profile_core.h',
  'agx_ioto_program_core.c',
  'agx_ioto_program_core.h',
  'agx_ioto_state_core.c',
  'agx_ioto_state_core.h',
  'agx_g17p_draw_state_upload.h',
  'agx_rt_result.h',
  'agx_rt_ift.h',
  'agx_scs_layout.c',
  'agx_scs_layout.h',
  'agx_bo.c',
  'agx_border.c',
  'agx_chip.c',
  'agx_g15_wire.c',
  'agx_device.c',
  'agx_bytes.h',
  'agx_linker.c',
  'agx_bg_eot.c',
  'agx_bg_eot_key.c',
  'agx_tilebuffer.c',
  'agx_nir_lower_msaa.c',
  'agx_nir_lower_sample_intrinsics.c',
  'agx_nir_lower_tilebuffer.c',
  'agx_nir_lower_uvs.c',
  'agx_nir_lower_vbo.c',
  'agx_nir_prolog_epilog.c',
  'agx_nir_legacy_epilog.c',
  'agx_ppp.h',
  'agx_ppp_hal200_graphics_core.c',
  'agx_ppp_hal200_graphics_core.h',
  'agx_ppp_hal200_raster_core.c',
  'agx_ppp_hal200_raster_core.h',
  'agx_ppp_hal300_depth_bias_scissor_core.c',
  'agx_ppp_hal300_depth_bias_scissor_core.h',
  'agx_ppp_hal300_graphics_core.c',
  'agx_ppp_hal300_graphics_core.h',
  'agx_ppp_hal300_depth_stencil_core.c',
  'agx_ppp_hal300_depth_stencil_core.h',
  'agx_ppp_decode_core.c',
  'agx_ppp_decode_core.h',
  'agx_ppp_viewport_payload_core.c',
  'agx_ppp_viewport_payload_core.h',
  'agx_ppp_w_clamp_cull2_core.c',
  'agx_ppp_w_clamp_cull2_core.h',
  'agx_pbe_agx3_core.c',
  'agx_pbe_agx3_core.h',
  'agx_scs_layout.c',
  'agx_scs_layout.h',
  'agx_scratch.c',
  'agx_texture_side_entry.c',
  'agx_texture_side_entry.h',
  'agx_vdm_barrier_core.c',
  'agx_vdm_barrier_core.h',
  'agx_vdm_hal200_bind_tail_core.c',
  'agx_vdm_hal200_bind_tail_core.h',
  'agx_vdm_hal200_direct_draw_core.c',
  'agx_vdm_hal200_direct_draw_core.h',
  'agx_vdm_decode_core.c',
  'agx_vdm_mesh_core.c',
  'agx_vdm_mesh_core.h',
  'agx_vdm_decode_core.h',
  'agx_vdm_index_list_core.c',
  'agx_vdm_index_list_core.h',
  'agx_vdm_ppp_state_core.c',
  'agx_vdm_ppp_state_core.h',
  'agx_vdm_stream_link_core.c',
  'agx_vdm_stream_link_core.h',
  'agx_vdm_stream_terminate_core.c',
  'agx_vdm_stream_terminate_core.h',
  'agx_vdm_tessellate_core.c',
  'agx_vdm_tessellate_core.h',
  'agx_vdm_state_core.c',
  'agx_vdm_state_core.h',
  'agx_vdm_state_update_core.c',
  'agx_vdm_state_update_core.h',
  'agx_va.c',
  'agx_va_runtime_hole.c',
  'agx_va_runtime_hole.h',
  'pool.c',
)

asahi_lib_has_virtio = with_vdrm
if asahi_lib_has_virtio
  libasahi_lib_files += files('agx_device_virtio.c')
endif

libasahi_decode_files = files(
  'agx3_cdm_decode_core.c',
  'agx3_cdm_direct_launch_semantic_core.c',
  'agx_ioto_core.c',
  'agx_ppp_decode_core.c',
  'agx_ppp_hal200_graphics_core.c',
  'agx_ppp_hal300_depth_bias_scissor_core.c',
  'agx_ppp_hal300_graphics_core.c',
  'agx_ppp_hal300_graphics_core.h',
  'agx_ppp_hal300_depth_stencil_core.c',
  'agx_ppp_viewport_payload_core.c',
  'agx_ppp_w_clamp_cull2_core.c',
  'agx_vdm_barrier_core.c',
  'agx_vdm_hal200_bind_tail_core.c',
  'agx_vdm_hal200_direct_draw_core.c',
  'agx_vdm_decode_core.c',
  'agx_vdm_mesh_core.c',
  'agx_vdm_mesh_core.h',
  'agx_vdm_index_list_core.c',
  'agx_vdm_ppp_state_core.c',
  'agx_vdm_state_core.c',
  'agx_vdm_state_update_core.c',
  'agx_vdm_stream_link_core.c',
  'agx_vdm_stream_terminate_core.c',
  'agx_vdm_tessellate_core.c',
  'decode.c',
)

libasahi_decode = static_library(
  'asahi_decode',
  [libasahi_decode_files, agx_pack],
  dependencies : [dep_valgrind, idep_mesautil],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  link_with : [libagx2_disasm],
  build_by_default : false,
)

libasahi_decode_shared = shared_library(
  'asahi_decode',
  [libasahi_decode_files, agx_pack],
  dependencies : [dep_valgrind, idep_mesautil],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  build_by_default : with_tools.contains('asahi'),
  link_with : [libagx2_disasm],
)

if asahi_lib_has_virtio
  libasahi_lib_includes = [inc_asahi, inc_virtio_gpu, inc_virtio_vdrm]
  libasahi_lib_links = [libasahi_decode, libvdrm, libpoly_nir]
  libasahi_lib_deps = [
    dep_libdrm, dep_valgrind, idep_nir, idep_mesautil, idep_libagx,
    idep_libpoly,
  ]
else
  libasahi_lib_includes = [inc_asahi]
  libasahi_lib_links = [libasahi_decode, libpoly_nir]
  libasahi_lib_deps = [
    dep_valgrind, idep_nir, idep_mesautil, idep_libagx, idep_libpoly,
  ]
endif

libagx_shaders = custom_target(
  input : libagx_spv,
  output : ['libagx_shaders.h', 'libagx_shaders.c'],
  command : [prog_asahi_clc, libagx_spv, '@OUTPUT0@', '@OUTPUT1@'],
)

idep_libagx_shaders_h = declare_dependency(
  sources : [libagx_shaders],
  include_directories : include_directories('.'),
)

libasahi_lib = static_library(
  'asahi_lib',
  [libasahi_lib_files, libagx_shaders, agx_pack, sha1_h],
  include_directories : libasahi_lib_includes,
  c_args : [
    no_override_init_args,
    '-Wno-c2x-extensions',
  ],
  gnu_symbol_visibility : 'hidden',
  link_with: libasahi_lib_links,
  dependencies: libasahi_lib_deps,
  build_by_default : false,
)

if with_tests
  test('asahi-g15-wire',
    executable('asahi-g15-wire',
      files('tests/test-g15-wire.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  foreach rt_test : ['rt-bvh', 'rt-ift', 'scs-layout']
    test('asahi-' + rt_test,
      executable('asahi-' + rt_test,
        ['tests/test-' + rt_test + '.c', 'agx_scs_layout.c'],
        include_directories : include_directories('.'),
        c_args : ['-UNDEBUG'],
      ),
      suite : ['asahi'],
    )
  endforeach
  test(
    'asahi-chip-admission',
    executable(
      'asahi-chip-admission',
      files('tests/test-chip-admission.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'asahi-usc-base-page',
    executable(
      'asahi-usc-base-page',
      files('tests/test-usc-base-page.c'),
      include_directories : [inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil, idep_nir],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout],
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'libasahi_tests',
    executable(
      'libasahi_tests',
      files(
        'tests/test-chip.cpp',
        'tests/test-packing.cpp',
        'tests/test-tilebuffer.cpp',
        'tests/test-g17p-msaa-epilog.cpp',
        'tests/test-m5-depth-export.cpp',
        'tests/test-primitive-varyings.cpp',
        '../vulkan/hk_nir_widen_16bit_varyings.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies: [
        idep_gtest, idep_agx_pack, idep_mesautil, idep_nir,
      ],
      include_directories: [inc_asahi],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout,
                   libagx3c_disasm],
    ),
    suite : ['asahi'],
    protocol : 'gtest',
  )
endif

if with_tests
  test('vbo_robustness', executable('vbo_robustness_test',
    files('tests/test-vbo-robustness.c'),
    include_directories : [inc_asahi, inc_include, inc_src],
    dependencies : [idep_nir, idep_mesautil, idep_agx_pack, idep_libagx_shaders_h],
    link_with : [libasahi_lib, libasahi_compiler, libasahi_layout]),
    suite : ['asahi'])
endif

if dep_iokit.found()
  libasahi_wrap = shared_library(
    'wrap',
    'wrap.c',
    include_directories : [inc_src, inc_asahi],
    dependencies : [idep_mesautil, dep_iokit],
    c_args : [no_override_init_args, '-Wno-missing-prototypes'],
    gnu_symbol_visibility : 'hidden',
    build_by_default : with_tools.contains('asahi'),
    link_with: libasahi_decode,
  )
endif

if with_tests
  test('asahi-rt-payload-gen1', executable('asahi-rt-payload-gen1',
    'tests/test-rt-payload-gen1.c',
    c_args : ['-ffp-contract=off'],
    include_directories : include_directories('.'),
    dependencies : [dep_m]), suite : ['asahi'])
  test('agx3_m5_programs', executable('agx3_m5_programs',
    '../compiler/test/test-agx3-m5-program.c',
    include_directories: [inc_include, inc_src, inc_asahi],
    dependencies: [idep_mesautil, idep_nir],
    link_with: [libasahi_compiler, libasahi_lib]), suite: ['asahi'])
endif

if with_tests
  test('agx_rt_envelope', executable('agx_rt_envelope',
    'tests/test-rt-envelope.c',
    c_args: ['-ffp-contract=off'],
    include_directories: include_directories('.')), suite: ['asahi'])
  test('agx_rt_blas_gen3', executable('agx_rt_blas_gen3',
    'tests/test-rt-blas-gen3.c',
    include_directories: include_directories('.')), suite: ['asahi'])
  test('agx_rt_bvh', executable('agx_rt_bvh',
    'tests/test-rt-bvh.c',
    include_directories: include_directories('.')), suite: ['asahi'])
  test('agx_m5_layout', executable('agx_m5_layout',
    ['tests/test-agx-m5-layout.c', 'agx3_code_container_core.c'],
    include_directories: include_directories('.')), suite: ['asahi'])
endif

if with_tests
  test('agx3_m5_instructions', executable('agx3_m5_instructions',
    '../compiler/test/test-agx3-m5-instructions.c',
    include_directories: [inc_include, inc_src, inc_asahi],
    dependencies: [idep_mesautil, idep_nir, idep_agx_opcodes_h],
    link_with: [libasahi_compiler, libasahi_lib]), suite: ['asahi'])
endif

if with_tests
  test('agx_scs_nir', executable('agx_scs_nir',
    '../compiler/test/test-scs-nir.c',
    c_args: ['-UNDEBUG'],
    include_directories: [inc_include, inc_src, inc_asahi],
    dependencies: [idep_mesautil, idep_nir, idep_agx_opcodes_h],
    link_with: [libasahi_compiler, libasahi_lib]), suite: ['asahi'])
endif

if with_tests
  test('agx3-m5-mesh-program', executable('agx3-m5-mesh-program',
    '../compiler/test/test-native-mesh-program.c',
    include_directories: [inc_include, inc_src, inc_asahi],
    dependencies: [idep_mesautil, idep_nir, idep_agx_opcodes_h],
    link_with: [libasahi_compiler, libasahi_lib]), suite: ['asahi'])
endif

if with_tests
  test('agx_scs_query', executable('agx_scs_query',
    '../compiler/test/test-scs-query.c',
    c_args: ['-UNDEBUG'],
    include_directories: [inc_include, inc_src, inc_asahi],
    dependencies: [idep_mesautil, idep_nir, idep_agx_opcodes_h],
    link_with: [libasahi_compiler, libasahi_lib]), suite: ['asahi'])
endif

if with_tests
  test('agx_scs_layout', executable('agx_scs_layout',
    ['tests/test-scs-layout.c', 'agx_scs_layout.c'],
    c_args: ['-UNDEBUG'],
    include_directories: include_directories('.')), suite: ['asahi'])
endif

if with_tests
  test('agx_rt_ift', executable('agx_rt_ift', 'tests/test-rt-ift.c',
    c_args: ['-UNDEBUG'],
    include_directories: include_directories('.')), suite: ['asahi'])
endif
