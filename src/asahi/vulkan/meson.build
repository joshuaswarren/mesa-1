# Copyright © 2022 Collabora Ltd. and Red Hat Inc.
# SPDX-License-Identifier: MIT
hk_files = files(
  'hk_buffer.c',
  'hk_buffer.h',
  'hk_buffer_view.c',
  'hk_buffer_view.h',
  'hk_cmd_buffer.c',
  'hk_cmd_buffer.h',
  'hk_cmd_lifetime.c',
  'hk_cs_root_size.h',
  'hk_cmd_clear.c',
  'hk_cmd_dispatch.c',
  'hk_cmd_draw.c',
  'hk_cmd_meta.c',
  'hk_cmd_pool.c',
  'hk_cmd_pool.h',
  'hk_descriptor_set.h',
  'hk_descriptor_set.c',
  'hk_descriptor_set_layout.c',
  'hk_descriptor_set_layout.h',
  'hk_descriptor_table.c',
  'hk_descriptor_table.h',
  'hk_device.c',
  'hk_device.h',
  'hk_device_memory.c',
  'hk_device_memory.h',
  'hk_event.c',
  'hk_event.h',
  'hk_format.c',
  'hk_g17p_meta_copy_core.c',
  'hk_g17p_meta_copy_core.h',
  'hk_agx3_ioto_decode.c',
  'hk_agx3_ioto_decode.h',
  'hk_agx3_ioto_preshader_core.c',
  'hk_agx3_ioto_preshader_core.h',
  'hk_agx3_ioto_profile_core.c',
  'hk_agx3_ioto_profile_core.h',
  'hk_agx3_ioto_program_core.c',
  'hk_agx3_ioto_program_core.h',
  'hk_agx3_ioto_state_core.c',
  'hk_agx3_ioto_state_core.h',
  'hk_agx3_tile_core.c',
  'hk_agx3_tile_core.h',
  'hk_agx3_raytracing_core.c',
  'hk_agx3_raytracing_core.h',
  'hk_agx3_ppp_hal300_depth_bias_scissor.c',
  'hk_agx3_ppp_hal300_depth_bias_scissor.h',
  'hk_agx3_ppp_hal300_depth_stencil.c',
  'hk_agx3_ppp_hal300_depth_stencil.h',
  'hk_agx3_ppp_viewport_payload.c',
  'hk_agx3_ppp_viewport_payload.h',
  'hk_agx3_ppp_w_clamp_cull2.c',
  'hk_agx3_ppp_w_clamp_cull2.h',
  'hk_agx3_vdm_decode.c',
  'hk_agx3_vdm_decode.h',
  'hk_image.c',
  'hk_image.h',
  'hk_image_view.c',
  'hk_image_view.h',
  'hk_instance.c',
  'hk_instance.h',
  'hk_nir_lower_descriptors.c',
  'hk_nir_passthrough_gs.c',
  'hk_native_descriptor.h',
  'hk_native_descriptor_pack.c',
  'hk_native_descriptor_pack.h',
  'hk_native_sampler.c',
  'hk_native_sampler.h',
  'hk_physical_device.c',
  'hk_physical_device.h',
  'hk_private.h',
  'hk_query_pool.c',
  'hk_query_pool.h',
  'hk_drm_render_core.c',
  'hk_drm_render_core.h',
  'hk_queue.c',
  'hk_queue.h',
  'hk_sampler.c',
  'hk_sampler.h',
  'hk_shader.c',
  'hk_shader.h',
  'hk_wsi.c',
  'hk_wsi.h'
)

if with_tests
  test(
    'hk_cs_root_size',
    executable(
      'hk_cs_root_size_test',
      files('tests/test-hk-cs-root-size.c'),
      include_directories : [
        inc_gallium,
        inc_gallium_aux,
        inc_include,
        inc_src,
        inc_asahi,
        inc_virtio_gpu,
        inc_virtio_vdrm,
      ],
      dependencies : [
        idep_mesautil,
        idep_vulkan_runtime_headers,
        idep_vulkan_util_headers,
      ],
      c_args : [c_msvc_compat_args, '-Wno-error=pointer-arith'],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_cmd_lifetime',
    executable(
      'hk_cmd_lifetime_test',
      files('tests/test-hk-cmd-lifetime.c', 'hk_cmd_lifetime.c'),
      include_directories : [
        inc_gallium,
        inc_gallium_aux,
        inc_include,
        inc_src,
        inc_asahi,
        inc_virtio_gpu,
        inc_virtio_vdrm,
      ],
      dependencies : [
        idep_nir,
        idep_mesautil,
        idep_libagx,
        idep_vulkan_runtime_headers,
        idep_vulkan_util_headers,
      ],
      c_args : [c_msvc_compat_args, '-Wno-error=pointer-arith'],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_g17p_full_frame_scissor',
    executable(
      'hk_g17p_full_frame_scissor_test',
      files('tests/test-hk-g17p-full-frame-scissor.c',
            '../lib/agx_ppp_hal200_raster_core.c',
            '../lib/agx_ppp_hal200_graphics_core.c',
            '../lib/agx_vdm_state_core.c'),
      include_directories : [inc_include, inc_src],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_g17p_draw_state_upload',
    executable(
      'hk_g17p_draw_state_upload_test',
      files('tests/test-hk-g17p-draw-state-upload.c',
            '../lib/agx_vdm_hal200_bind_tail_core.c',
            '../lib/agx_vdm_hal200_direct_draw_core.c'),
      include_directories : [inc_include, inc_src],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_printf_root',
    executable(
      'hk_printf_root_test',
      files('tests/test-hk-printf-root.c',
            'hk_nir_lower_descriptors.c'),
      include_directories : [
        include_directories('.', '../libagx', '../../gallium/include',
                            '../../gallium/auxiliary', '../../vulkan/runtime',
                            '../../vulkan/util', '../../vulkan/wsi',
                            '../../compiler/spirv', '../../virtio/virtio-gpu',
                            '../../virtio/vdrm'),
        inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_nir, idep_mesautil],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_g17p_meta_copy_core',
    executable(
      'hk_g17p_meta_copy_core_test',
      files('tests/test-hk-g17p-meta-copy-core.c',
            'hk_g17p_meta_copy_core.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_drm_render_core',
    executable(
      'hk_drm_render_core_test',
      files('tests/test-hk-drm-render-core.c',
            'hk_drm_render_core.c'),
      include_directories : [inc_include, inc_src],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_agx3_ioto_decode',
    executable(
      'hk_agx3_ioto_decode_test',
      files('tests/test-hk-agx3-ioto-decode.c',
            'hk_agx3_ioto_decode.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_agx3_ioto_profile_core',
    executable(
      'hk_agx3_ioto_profile_core_test',
      files('tests/test-hk-agx3-ioto-profile-core.c',
            'hk_agx3_ioto_profile_core.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_agx3_ioto_preshader_core',
    executable(
      'hk_agx3_ioto_preshader_core_test',
      files('tests/test-hk-agx3-ioto-preshader-core.c',
            'hk_agx3_ioto_preshader_core.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test('hk_vbo_raw', executable('hk_vbo_raw_test',
    files('tests/test-hk-vbo-raw.c'),
    include_directories : [include_directories('.'), inc_include, inc_src, inc_asahi],
    dependencies : [idep_nir, idep_mesautil],
    link_with : [libasahi_lib, libasahi_layout, libasahi_compiler],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden'), suite : ['asahi'])

  test(
    'hk_agx3_ioto_program_core',
    executable(
      'hk_agx3_ioto_program_core_test',
      files('tests/test-hk-agx3-ioto-program-core.c',
            'hk_agx3_ioto_program_core.c',
            '../compiler/agx3_ioto_uniform_program.c',
            '../compiler/agx3_ioto_load_packer.c',
            '../compiler/agx3_mem_core.c',
            '../compiler/agx3_binder.c',
            '../compiler/agx3_compose/agx3_compose.c',
            '../compiler/agx3_compose/agx3b_tables.c',
            '../compiler/agx3_compose/agx3c_tables.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_mesautil],
      link_with : [libasahi_lib, libasahi_layout],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_agx3_ioto_state_core',
    executable(
      'hk_agx3_ioto_state_core_test',
      files('tests/test-hk-agx3-ioto-state-core.c',
            'hk_agx3_ioto_decode.c',
            'hk_agx3_ioto_state_core.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_agx3_tile_core',
    executable(
      'hk_agx3_tile_core_test',
      files('tests/test-hk-agx3-tile-core.c',
            'hk_agx3_tile_core.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_mesautil],
      link_with : [libasahi_compiler],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'hk_agx3_raytracing_core',
    executable(
      'hk_agx3_raytracing_core_test',
      files('tests/test-hk-agx3-raytracing-core.c',
            'hk_agx3_raytracing_core.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_mesautil],
      link_with : [libasahi_compiler],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'hk_agx3_ppp_hal300_depth_bias_scissor',
    executable(
      'hk_agx3_ppp_hal300_depth_bias_scissor_test',
      files('tests/test-hk-agx3-ppp-hal300-depth-bias-scissor.c',
            'hk_agx3_ppp_hal300_depth_bias_scissor.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'hk_agx3_ppp_hal300_depth_stencil',
    executable(
      'hk_agx3_ppp_hal300_depth_stencil_test',
      files('tests/test-hk-agx3-ppp-hal300-depth-stencil.c',
            'hk_agx3_ppp_hal300_depth_stencil.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'hk_agx3_ppp_viewport_payload',
    executable(
      'hk_agx3_ppp_viewport_payload_test',
      files('tests/test-hk-agx3-ppp-viewport-payload.c',
            'hk_agx3_ppp_viewport_payload.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'hk_agx3_ppp_w_clamp_cull2',
    executable(
      'hk_agx3_ppp_w_clamp_cull2_test',
      files('tests/test-hk-agx3-ppp-w-clamp-cull2.c',
            'hk_agx3_ppp_w_clamp_cull2.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'hk_agx3_vdm_decode',
    executable(
      'hk_agx3_vdm_decode_test',
      files('tests/test-hk-agx3-vdm-decode.c',
            'hk_agx3_vdm_decode.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'hk_native_descriptor',
    executable(
      'hk_native_descriptor_test',
      files('tests/test-hk-native-descriptor.c',
            'hk_native_descriptor_pack.c',
            '../lib/agx_pbe_agx3_core.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_agx_pack, idep_mesautil],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )
  test(
    'hk_native_sampler',
    executable(
      'hk_native_sampler_test',
      files('tests/test-hk-native-sampler.c', 'hk_native_sampler.c'),
      include_directories : [
        include_directories('.'), inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_agx_pack, idep_mesautil],
      link_with : [libasahi_lib],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  if with_platform_x11
    executable(
      'g17p-wsi-triangle',
      files('tests/g17p_wsi_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl, dep_xcb],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )
  endif

  if host_machine.system() == 'linux'
    executable(
      'asahi-vulkan-g17p-enumerate',
      files('tests/g17p_vulkan_enumerate.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'asahi-submit-g17p-add3',
      files('tests/asahi_submit_g17p_add3.c'),
      include_directories : [inc_include],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'asahi-submit-g17p-render',
      files('tests/asahi_submit_g17p_render.c'),
      include_directories : [inc_include],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-offscreen-triangle',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-graphics-pipeline-library',
      files('tests/g17p_graphics_pipeline_library.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-hostcopy-triangle',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_HOST_IMAGE_COPY=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-render-only',
      files('tests/g17p_render_only.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-timestamp-query',
      files('tests/g17p_vulkan_timestamp_query.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-occlusion-query',
      files('tests/g17p_render_only.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_OCCLUSION_QUERY=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-compute',
      files('tests/g17p_vulkan_compute.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # Same probe, minimal shader: one store of a computed constant, no
    # constant-data array. Isolates "does any G17P dispatch execute" from
    # "does the constant-data path work".
    # No descriptor set at all: set_count == 0, so the Ioto preamble has
    # nothing to load either. The only dispatch that touches no memory before
    # OR during the shader.
    executable(
      'g17p-vulkan-compute-noroot',
      files('tests/g17p_vulkan_compute_noroot.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # No memory access at run time: separates "dispatches execute and the
    # memory access faults" from "no dispatch has ever executed". The
    # observable is the retire duration in dmesg, not the buffer.
    executable(
      'g17p-vulkan-compute-null',
      files('tests/g17p_vulkan_compute_null.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-compute-min',
      files('tests/g17p_vulkan_compute_min.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-compute-count',
      files('tests/g17p_vulkan_compute_count.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-compute-cover',
      files('tests/g17p_vulkan_compute_cover.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-compute-lid',
      files('tests/g17p_vulkan_compute_lid.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-compute-lane',
      files('tests/g17p_vulkan_compute_lane.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-compute-stride',
      files('tests/g17p_vulkan_compute_stride.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )
  endif
endif

hk_entrypoints = custom_target(
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['hk_entrypoints.h', 'hk_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'hk',
    '--beta', with_vulkan_beta.to_string(),
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

hk_deps = [
  dep_libdrm,
  idep_nir,
  idep_vulkan_runtime,
  idep_vulkan_util,
  idep_vulkan_wsi,
  idep_vulkan_wsi_headers,
  idep_agx_pack,
  idep_libagx,
  idep_libagx_shaders_h,
]

libhk = static_library(
  'hk',
  [
    hk_entrypoints,
    hk_files,
    sha1_h,
  ],
  include_directories : [
    inc_gallium,
    inc_gallium_aux,
    inc_include,
    inc_src,
    inc_asahi,
    inc_virtio_gpu,
    inc_virtio_vdrm,
  ],
  link_with : [libasahi_lib, libasahi_layout, libasahi_compiler],
  c_args : ['-Wno-c2x-extensions'],
  dependencies : [hk_deps],
  gnu_symbol_visibility : 'hidden',
)

libvulkan_asahi = shared_library(
  'vulkan_asahi',
  link_whole : [libhk],
  link_args: [ld_args_build_id],
  gnu_symbol_visibility : 'hidden',
  install : true,
)

icd_file_name = libname_prefix + 'vulkan_asahi.' + libname_suffix

asahi_icd = custom_target(
  input : [vk_icd_gen, vk_api_xml],
  output : 'asahi_icd.' + vulkan_manifest_suffix,
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.4', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', vulkan_icd_lib_path,
    '--icd-filename', icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
  install_dir : with_vulkan_icd_dir,
  install_tag : 'runtime',
  install : true,
)

_dev_icdname = 'asahi_devenv_icd.@0@.json'.format(host_machine.cpu())
_dev_icd = custom_target(
  input : [vk_icd_gen, vk_api_xml],
  output : _dev_icdname,
  command : [
    prog_python, '@INPUT0@',
    '--api-version', '1.4', '--xml', '@INPUT1@',
    '--sizeof-pointer', sizeof_pointer,
    '--icd-lib-path', meson.current_build_dir(),
    '--icd-filename', icd_file_name,
    '--out', '@OUTPUT@',
  ],
  build_by_default : true,
)

devenv.append('VK_DRIVER_FILES', _dev_icd.full_path())
