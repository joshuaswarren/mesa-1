# Copyright 2017 Intel Corporation
# SPDX-License-Identifier: MIT

files_asahi = files(
  'driinfo_asahi.h',
  'agx_batch.c',
  'agx_blit.c',
  'agx_disk_cache.c',
  'agx_fence.c',
  'agx_pipe.c',
  'agx_nir_lower_sysvals.c',
  'agx_nir_lower_bindings.c',
  'agx_nir_lower_point_size.c',
  'agx_native_descriptor.c',
  'agx_query.c',
  'agx_state.c',
  'agx_streamout.c',
  'agx_uniforms.c',
)

if with_tests
  test(
    'asahi-native-descriptor',
    executable(
      'asahi-native-descriptor',
      files(
        'agx_native_descriptor.c',
        'tests/test-agx-native-descriptor.c',
      ),
      # The AGX3 PBE encoder is the shared byte-grounded one in
      # src/asahi/lib, the same object Honeykrisp and the D3D12 UMD drive.
      # Compile it in rather than linking libasahi_lib, which would pull the
      # whole device/BO layer into a pure descriptor test.
      objects : [libasahi_lib.extract_objects('agx_pbe_agx3_core.c')],
      include_directories : [inc_src, inc_asahi],
      dependencies : [idep_agx_pack, idep_mesautil],
    ),
    suite : ['asahi'],
  )
endif

libasahi = static_library(
  'asahi',
  [files_asahi],
  include_directories : [inc_gallium_aux, inc_gallium, inc_include, inc_src, inc_asahi, inc_virtio_gpu, inc_virtio_vdrm],
  c_args : [c_msvc_compat_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  dependencies : [idep_nir, idep_mesautil, idep_agx_pack, dep_libdrm, idep_libagx, idep_libagx_shaders_h],
)

driver_asahi = declare_dependency(
  compile_args : '-DGALLIUM_ASAHI',
  link_with : [libasahi, libasahiwinsys, libasahi_compiler, libasahi_lib, libasahi_layout, libasahi_decode]
)
