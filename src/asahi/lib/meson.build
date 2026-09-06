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
  'agx3_code_container_core.c',
  'agx3_code_container_core.h',
  'agx_ioto_core.c',
  'agx_ioto_core.h',
  'agx_ioto_imageblock_core.c',
  'agx_ioto_imageblock_core.h',
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
  'agx_rt_result.h',
  'agx_scs_layout.c',
  'agx_scs_layout.h',
  'agx_bo.c',
  'agx_border.c',
  'agx_chip.c',
  'agx_device.c',
  'agx_linker.c',
  'agx_bg_eot.c',
  'agx_tilebuffer.c',
  'agx_nir_lower_msaa.c',
  'agx_nir_lower_sample_intrinsics.c',
  'agx_nir_lower_tilebuffer.c',
  'agx_nir_lower_uvs.c',
  'agx_nir_lower_vbo.c',
  'agx_nir_prolog_epilog.c',
  'agx_ppp.h',
  'agx_ppp_hal200_graphics_core.c',
  'agx_ppp_hal200_graphics_core.h',
  'agx_ppp_hal200_raster_core.c',
  'agx_ppp_hal200_raster_core.h',
  'agx_ppp_hal300_depth_bias_scissor_core.c',
  'agx_ppp_hal300_depth_bias_scissor_core.h',
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

asahi_lib_is_windows = host_machine.system() == 'windows'
asahi_lib_has_virtio = not asahi_lib_is_windows and with_vdrm
asahi_d3d12_strict_args = []
if asahi_lib_is_windows and with_asahi_d3d12
  asahi_d3d12_strict_args = cc.get_supported_arguments(
    '-Wall',
    '-Wextra',
    '-Werror',
    '-Wpedantic',
    '-Wshadow',
    '-Wstrict-prototypes',
    '-Wmissing-prototypes',
    '-Wno-unknown-pragmas',
    '-Wno-ignored-attributes',
    '-Wno-microsoft-enum-forward-reference',
    '-Wno-language-extension-token',
    '-Wno-c23-extensions',
    # These diagnostics originate in NIR macros and generated AGX pack
    # interfaces included by the winsys, rather than in the winsys itself.
    '-Wno-gnu-statement-expression-from-macro-expansion',
    '-Wno-gnu-zero-variadic-macro-arguments',
    '-Wno-unused-parameter',
    '-Wno-gnu-empty-struct',
    '-Wno-zero-length-array',
    '-Wno-missing-field-initializers',
    '-Wno-unused-function',
  )
endif

if asahi_lib_is_windows
  libasahi_lib_files += files(
    'agx_d3dkmt.c',
    'agx_d3dkmt.h',
    'agx_d3dkmt_core.c',
    'agx_d3dkmt_core.h',
    'agx_d3dkmt_priv.h',
    'agx_d3dkmt_runtime.c',
    'agx_d3dkmt_runtime.h',
    'agx_device_d3dkmt.c',
    'agx_device_d3dkmt.h',
    'agx_sync_d3dkmt.c',
    'decode_stub.c',
  )

  agxw_contract_dir = meson.get_external_property('agx_wddm_contract_core', '')
  if agxw_contract_dir == ''
    agxw_contract_dir = meson.get_external_property('ntagx_wddm_contract_core')
  endif

  libasahi_lib_files += files(agxw_contract_dir / 'agxw_contract.c')
  libasahi_lib_inc_win = include_directories(agxw_contract_dir)
elif asahi_lib_has_virtio
  libasahi_lib_files += files('agx_device_virtio.c')
  libasahi_lib_inc_win = []
else
  libasahi_lib_inc_win = []
endif

libasahi_decode_files = files(
  'agx3_cdm_decode_core.c',
  'agx3_cdm_direct_launch_semantic_core.c',
  'agx_ioto_core.c',
  'agx_ppp_decode_core.c',
  'agx_ppp_hal200_graphics_core.c',
  'agx_ppp_hal300_depth_bias_scissor_core.c',
  'agx_ppp_hal300_depth_stencil_core.c',
  'agx_ppp_viewport_payload_core.c',
  'agx_ppp_w_clamp_cull2_core.c',
  'agx_vdm_barrier_core.c',
  'agx_vdm_hal200_bind_tail_core.c',
  'agx_vdm_hal200_direct_draw_core.c',
  'agx_vdm_decode_core.c',
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

if asahi_lib_is_windows
  libasahi_lib_includes = [inc_asahi, libasahi_lib_inc_win]
  libasahi_lib_links = [libpoly_nir]
  libasahi_lib_deps = [
    dep_valgrind, idep_nir, idep_mesautil, idep_libagx, idep_libpoly,
  ]
elif asahi_lib_has_virtio
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
    asahi_d3d12_strict_args,
  ],
  gnu_symbol_visibility : 'hidden',
  link_with: libasahi_lib_links,
  dependencies: libasahi_lib_deps,
  build_by_default : false,
)

if with_tests
  test(
    'asahi-agx-rt-bvh',
    executable(
      'asahi-agx-rt-bvh',
      files('tests/test-rt-bvh.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-chip',
    executable(
      'asahi-agx-chip',
      files(
        'agx_chip.c',
        'tests/test-chip.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-linker-container',
    executable(
      'asahi-agx-linker-container',
      files(
        'agx3_code_container_core.c',
        'tests/test-agx-linker-container.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-g17p-early-demote-epilog',
    executable(
      'asahi-g17p-early-demote-epilog',
      files('tests/test-agx-g17p-early-demote-epilog.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-g17p-advanced-blend',
    executable(
      'asahi-g17p-advanced-blend',
      files('tests/test-agx-g17p-advanced-blend.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-g17p-srgb-epilog',
    executable(
      'asahi-g17p-srgb-epilog',
      files('tests/test-agx-g17p-srgb-epilog.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_lib, libasahi_compiler, libasahi_layout],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-ppp-hal200-raster-core',
    executable(
      'asahi-ppp-hal200-raster-core',
      files(
        'agx_ppp_hal200_raster_core.c',
        'tests/test-ppp-hal200-raster-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-va-runtime-hole',
    executable(
      'asahi-agx-va-runtime-hole',
      files(
        'agx_va_runtime_hole.c',
        'tests/test-agx-va-runtime-hole.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_mesautil],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-scs-layout',
    executable(
      'asahi-agx-scs-layout',
      files(
        'agx_scs_layout.c',
        'tests/test-agx-scs-layout.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-ioto-core',
    executable(
      'asahi-agx-ioto-core',
      files(
        'agx_ioto_core.c',
        'tests/test-agx-ioto-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-ioto-profile-core',
    executable(
      'asahi-agx-ioto-profile-core',
      files(
        'agx_ioto_profile_core.c',
        'tests/test-agx-ioto-profile-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-ioto-imageblock-core',
    executable(
      'asahi-agx-ioto-imageblock-core',
      files(
        'agx_ioto_imageblock_core.c',
        'tests/test-agx-ioto-imageblock-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-ioto-state-core',
    executable(
      'asahi-agx-ioto-state-core',
      files(
        'agx_ioto_core.c',
        'agx_ioto_state_core.c',
        'tests/test-agx-ioto-state-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-ioto-load-core',
    executable(
      'asahi-agx-ioto-load-core',
      files(
        'agx_ioto_core.c',
        'agx_ioto_load_core.c',
        'tests/test-agx-ioto-load-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-ioto-preshader-core',
    executable(
      'asahi-agx-ioto-preshader-core',
      files(
        'agx_ioto_core.c',
        'agx_ioto_preshader_core.c',
        'tests/test-agx-ioto-preshader-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx-ioto-program-core',
    executable(
      'asahi-agx-ioto-program-core',
      files(
        'agx_ioto_core.c',
        'agx_ioto_preshader_core.c',
        'agx_ioto_profile_core.c',
        'agx_ioto_state_core.c',
        'agx_ioto_load_core.c',
        'agx_ioto_program_core.c',
        'tests/test-agx-ioto-program-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-decode-core',
    executable(
      'asahi-vdm-decode-core',
      files(
        'agx_vdm_barrier_core.c',
        'agx_vdm_decode_core.c',
        'agx_vdm_index_list_core.c',
        'agx_vdm_ppp_state_core.c',
        'agx_vdm_state_core.c',
        'agx_vdm_state_update_core.c',
        'agx_vdm_stream_link_core.c',
        'agx_vdm_stream_terminate_core.c',
        'agx_vdm_tessellate_core.c',
        'tests/test-vdm-decode-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx3-cdm-decode-core',
    executable(
      'asahi-agx3-cdm-decode-core',
      files(
        'agx3_cdm_decode_core.c',
        'tests/test-agx3-cdm-decode-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx3-cdm-direct-launch-semantic-core',
    executable(
      'asahi-agx3-cdm-direct-launch-semantic-core',
      files(
        'agx3_cdm_direct_launch_semantic_core.c',
        'tests/test-agx3-cdm-direct-launch-semantic-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-agx3-compute-program-core',
    executable(
      'asahi-agx3-compute-program-core',
      files(
        'agx3_cdm_direct_launch_semantic_core.c',
        'agx3_compute_program_core.c',
        'agx_ioto_core.c',
        'agx_ioto_load_core.c',
        'agx_ioto_preshader_core.c',
        'agx_ioto_profile_core.c',
        'agx_ioto_program_core.c',
        'agx_ioto_state_core.c',
        'tests/test-agx3-compute-program-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi_feature_matrix',
    prog_python,
    args : [
      files('tests/test-feature-matrix.py'),
      meson.project_source_root() / 'src/asahi/lib/agx_chip.h',
      meson.project_source_root() / 'docs/asahi/apple-gpu-feature-matrix.md',
    ],
    suite : ['asahi'],
  )

  test(
    'asahi-texture-side-entry',
    executable(
      'asahi-texture-side-entry',
      files(
        'agx_texture_side_entry.c',
        'tests/test-texture-side-entry.c',
      ),
      include_directories : [inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-barrier-core',
    executable(
      'asahi-vdm-barrier-core',
      files(
        'agx_vdm_barrier_core.c',
        'tests/test-vdm-barrier-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-index-list-core',
    executable(
      'asahi-vdm-index-list-core',
      files(
        'agx_vdm_index_list_core.c',
        'tests/test-vdm-index-list-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-pbe-agx3-core',
    executable(
      'asahi-pbe-agx3-core',
      files(
        'agx_pbe_agx3_core.c',
        'tests/test-pbe-agx3-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-ppp-hal200-graphics-core',
    executable(
      'asahi-ppp-hal200-graphics-core',
      files(
         'agx_ppp_hal200_graphics_core.c',
         'agx_vdm_state_core.c',
         'tests/test-ppp-hal200-graphics-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-ppp-hal300-depth-bias-scissor-core',
    executable(
      'asahi-ppp-hal300-depth-bias-scissor-core',
      files(
        'agx_ppp_hal300_depth_bias_scissor_core.c',
        'tests/test-ppp-hal300-depth-bias-scissor-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-ppp-hal300-depth-stencil-core',
    executable(
      'asahi-ppp-hal300-depth-stencil-core',
      files(
        'agx_ppp_hal300_depth_stencil_core.c',
        'tests/test-ppp-hal300-depth-stencil-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-ppp-viewport-payload-core',
    executable(
      'asahi-ppp-viewport-payload-core',
      files(
        'agx_ppp_viewport_payload_core.c',
        'tests/test-ppp-viewport-payload-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-ppp-w-clamp-cull2-core',
    executable(
      'asahi-ppp-w-clamp-cull2-core',
      files(
        'agx_ppp_w_clamp_cull2_core.c',
        'tests/test-ppp-w-clamp-cull2-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-ppp-decode-core',
    executable(
      'asahi-ppp-decode-core',
      files(
        'agx_ppp_decode_core.c',
        'agx_ppp_hal300_depth_bias_scissor_core.c',
        'agx_ppp_hal300_depth_stencil_core.c',
        'agx_ppp_viewport_payload_core.c',
        'agx_ppp_w_clamp_cull2_core.c',
        'tests/test-ppp-decode-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-hal200-bind-tail-core',
    executable(
      'asahi-vdm-hal200-bind-tail-core',
      files(
        'agx_vdm_hal200_bind_tail_core.c',
        'tests/test-vdm-hal200-bind-tail-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-hal200-direct-draw-core',
    executable(
      'asahi-vdm-hal200-direct-draw-core',
      files(
        'agx_vdm_hal200_direct_draw_core.c',
        'tests/test-vdm-hal200-direct-draw-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-ppp-state-core',
    executable(
      'asahi-vdm-ppp-state-core',
      files(
        'agx_vdm_ppp_state_core.c',
        'tests/test-vdm-ppp-state-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-ppp-finalizer',
    executable(
      'asahi-ppp-finalizer',
      files(
        'agx_chip.c',
        'agx_vdm_ppp_state_core.c',
        'tests/test-ppp-fini.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-stream-link-core',
    executable(
      'asahi-vdm-stream-link-core',
      files(
        'agx_vdm_stream_link_core.c',
        'tests/test-vdm-stream-link-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-stream-terminate-core',
    executable(
      'asahi-vdm-stream-terminate-core',
      files(
        'agx_vdm_stream_terminate_core.c',
        'tests/test-vdm-stream-terminate-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-tessellate-core',
    executable(
      'asahi-vdm-tessellate-core',
      files(
        'agx_vdm_tessellate_core.c',
        'tests/test-vdm-tessellate-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-state-update-core',
    executable(
      'asahi-vdm-state-update-core',
      files(
        'agx_vdm_state_core.c',
        'agx_vdm_state_update_core.c',
        'tests/test-vdm-state-update-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'asahi-vdm-state-core',
    executable(
      'asahi-vdm-state-core',
      files(
        'agx_vdm_state_core.c',
        'tests/test-vdm-state-core.c',
      ),
      include_directories : [inc_include, inc_src, inc_asahi],
    ),
    suite : ['asahi'],
  )

  test(
    'libasahi_tests',
    executable(
      'libasahi_tests',
      files(
        'tests/test-packing.cpp',
        'tests/test-tilebuffer.cpp',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies: [idep_gtest, idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib, libasahi_layout],
    ),
    suite : ['asahi'],
    protocol : 'gtest',
  )
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
