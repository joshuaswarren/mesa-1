# Copyright © 2022 Collabora Ltd. and Red Hat Inc.
# SPDX-License-Identifier: MIT
hk_files = files(
  'hk_acceleration_structure.c',
  'hk_acceleration_structure.h',
  'hk_buffer.c',
  'hk_buffer.h',
  'hk_buffer_view.c',
  'hk_bvh_depth_core.c',
  'hk_bvh_depth_core.h',
  'hk_buffer_view.h',
  'hk_bvh.h',
  'hk_cmd_buffer.c',
  'hk_cmd_buffer.h',
  'hk_conditional_render_core.h',
  'hk_cmd_lifetime.c',
  'hk_cs_root_size.h',
  'hk_cmd_clear.c',
  'hk_cmd_dispatch.c',
  'hk_cmd_draw.c',
  'hk_cmd_meta.c',
  'hk_cmd_trace_rays.c',
  'hk_cmd_pool.c',
  'hk_cmd_pool.h',
  'hk_descriptor_set.h',
  'hk_descriptor_set.c',
  'hk_descriptor_set_layout.c',
  'hk_descriptor_set_layout_hash.c',
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
  'hk_ioto_arena.c',
  'hk_ioto_arena_core.c',
  'hk_ioto_arena_core.h',
  'hk_g17p_meta_copy_core.h',
  'hk_agx3_ioto_program_core.c',
  'hk_agx3_ioto_program_core.h',
  'hk_agx3_vdm_decode.c',
  'hk_agx3_vdm_decode.h',
  'hk_ppp_state.h',
  'hk_image.c',
  'hk_image.h',
  'hk_image_layout_core.c',
  'hk_image_layout_core.h',
  'hk_image_view.c',
  'hk_image_view.h',
  'hk_instance.c',
  'hk_instance.h',
  'hk_nir_forward_image_stores.c',
  'hk_nir_forward_image_stores.h',
  'hk_nir_lower_descriptors.c',
  'hk_nir_lower_ray_queries.c',
  'hk_nir_lower_robust_image.c',
  'hk_nir_lower_robust_image.h',
  'hk_nir_passthrough_gs.c',
  'hk_nir_ray_tracing.c',
  'hk_ray_tracing_core.c',
  'hk_ray_tracing_core.h',
  'hk_ray_tracing_pipeline.c',
  'hk_ray_tracing_pipeline.h',
  'hk_nir_ray_tracing.h',
  'hk_nir_widen_16bit_varyings.c',
  'hk_nir_widen_16bit_varyings.h',
  'hk_native_descriptor.h',
  'hk_native_descriptor_pack.c',
  'hk_native_descriptor_pack.h',
  'hk_native_sampler.c',
  'hk_native_sampler.h',
  'hk_physical_device.c',
  'hk_physical_device.h',
  'hk_pipeline_statistics_core.c',
  'hk_pipeline_statistics_core.h',
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
  'hk_nir_lower_image_view_min_lod.c',
  'hk_shader.h',
  'hk_sync.c',
  'hk_sync.h',
  'hk_wsi.c',
  'hk_wsi.h'
)

if with_tests
  # Unused recording functions require section garbage collection so the
  # test can link the production resolver without the rest of the driver.
  if ld_args_gc_sections.length() > 0
    # Exercise the real resolver; unused command-recording entry points in
    # hk_sync.c need no device implementation in this CPU-only executable.
    test(
      'hk_sync_prior',
      executable(
        'hk_sync_prior_test',
        files('tests/test-hk-sync-prior.c', 'hk_sync.c'),
        include_directories : [
          include_directories('.'),
          inc_gallium, inc_gallium_aux, inc_include, inc_src, inc_asahi,
          inc_virtio_gpu, inc_virtio_vdrm,
        ],
        dependencies : [
          idep_nir_headers, idep_mesautil, idep_libagx,
          idep_libagx_shaders_h, idep_agx_pack,
          idep_vulkan_runtime_headers, idep_vulkan_util_headers,
        ],
        c_args : [c_msvc_compat_args, '-Wno-error=pointer-arith'],
        link_args : [ld_args_gc_sections],
        gnu_symbol_visibility : 'hidden',
      ),
      suite : ['asahi'],
    )

  endif

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
    'hk_agx3_ioto_program_core',
    executable(
      'hk_agx3_ioto_program_core_test',
      files('tests/test-hk-agx3-ioto-program-core.c',
            'hk_agx3_ioto_program_core.c',
            '../compiler/agx_g18_gates.c',
            '../compiler/agx3_compute_program.c',
            '../compiler/agx3_graphics_program.c',
            '../compiler/agx3_copy_core.c',
            '../compiler/agx3_ioto_uniform_program.c',
            '../compiler/agx3_ioto_load_packer.c',
            '../compiler/agx3_cprog_input.c',
            '../compiler/agx3_static_spill_core.c',
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

    executable(
      'g17p-render-only',
      files('tests/g17p_render_only.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )
  test(
    'hk_g17p_compression',
    executable(
      'hk_g17p_compression_test',
      files('tests/test-hk-g17p-compression.c'),
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'hk_image_layout_core',
    executable(
      'hk_image_layout_core_test',
      files('tests/test-hk-image-layout-core.c',
            'hk_image_layout_core.c'),
      include_directories : [include_directories('.')],
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

subdir('bvh')

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
    hk_bvh_spv,
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
    include_directories('.'),
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

if with_tests
  executable(
    'vulkan-xfb-resume',
    files('tests/vulkan_xfb_resume.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )
  executable(
    'vulkan-mipmap-layers',
    files('tests/vulkan_mipmap_layers.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'g17p-vulkan-multiview-clear',
    files('tests/g17p_vulkan_multiview_clear.c'),
    include_directories : [inc_include],
    dependencies : [cc.find_library('vulkan')],
    c_args : [c_msvc_compat_args],
    gnu_symbol_visibility : 'hidden',
  )
endif
