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
  'hk_pass_profile.c',
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
    'hk_g17p_compression',
    executable(
      'hk_g17p_compression_test',
      files('tests/test-hk-g17p-compression.c'),
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'hk_g17p_precomp_cache',
    executable(
      'hk_g17p_precomp_cache_test',
      files('tests/test-hk-g17p-precomp-cache.c'),
      include_directories : [inc_include, inc_src],
      dependencies : [idep_mesautil],
    ),
    suite : ['asahi'],
  )

  test(
    'hk_g17p_exact_compute_cache',
    executable(
      'hk_g17p_exact_compute_cache_test',
      files('tests/test-hk-g17p-exact-compute-cache.c'),
      include_directories : [inc_include, inc_src],
      dependencies : [idep_mesautil],
    ),
    suite : ['asahi'],
  )

  test(
    'hk_shader_variants',
    executable(
      'hk_shader_variants_test',
      files('tests/test-hk-shader-variants.c',
            'hk_descriptor_set_layout_hash.c'),
      include_directories : [
        include_directories('.', '../libagx', '../../gallium/include',
                            '../../gallium/auxiliary', '../../vulkan/runtime',
                            '../../vulkan/util', '../../vulkan/wsi',
                            '../../compiler/spirv', '../../virtio/virtio-gpu',
                            '../../virtio/vdrm'),
        inc_include, inc_src, inc_asahi,
      ],
      dependencies : [idep_nir, idep_mesautil, idep_libagx,
                      idep_libagx_shaders_h, idep_agx_pack],
    ),
    suite : ['asahi'],
  )

  if host_machine.system() == 'linux'
    hk_sparse_format_properties = executable(
      'hk-sparse-format-properties',
      files('tests/hk_sparse_format_properties.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
    )
    g17p_depth_feedback_query = executable(
      'g17p-depth-feedback-query',
      files('tests/g17p_depth_feedback_query.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
    )
  endif

  test(
    'hk_ray_tracing_core',
    executable(
      'hk_ray_tracing_core_test',
      files('tests/test-hk-ray-tracing-core.c',
            'hk_ray_tracing_core.c'),
      include_directories : [include_directories('.'), inc_include, inc_src],
      dependencies : [idep_mesautil, idep_vulkan_runtime_headers],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_bvh_depth_core',
    executable(
      'hk_bvh_depth_core_test',
      files('tests/test-hk-bvh-depth-core.c',
            'hk_bvh_depth_core.c'),
      include_directories : [include_directories('.'), inc_include, inc_src],
      dependencies : [idep_mesautil, idep_vulkan_runtime_headers],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_nir_ray_traversal',
    executable(
      'hk_nir_ray_traversal_test',
      files('tests/test-hk-nir-ray-traversal.c',
            'hk_nir_ray_tracing.c'),
      include_directories : [include_directories('.'), inc_include, inc_src],
      dependencies : [idep_nir, idep_mesautil, idep_vulkan_runtime_headers],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_nir_forward_image_stores',
    executable(
      'hk_nir_forward_image_stores_test',
      files('tests/test-hk-nir-forward-image-stores.c',
            'hk_nir_forward_image_stores.c'),
      include_directories : [include_directories('.'), inc_include, inc_src],
      dependencies : [idep_nir, idep_mesautil],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_nir_widen_16bit_varyings',
    executable(
      'hk_nir_widen_16bit_varyings_test',
      files('tests/test-hk-nir-widen-16bit-varyings.c',
            'hk_nir_widen_16bit_varyings.c'),
      include_directories : [include_directories('.'), inc_include, inc_src],
      dependencies : [idep_nir, idep_mesautil],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )



  test(
    'hk_nir_image_view_min_lod',
    executable(
      'hk_nir_image_view_min_lod_test',
      files('tests/test-hk-nir-image-view-min-lod.c',
            'hk_nir_lower_image_view_min_lod.c'),
      include_directories : [include_directories('.'), inc_include, inc_src, inc_asahi],
      dependencies : [idep_nir, idep_mesautil],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_nir_robust_image',
    executable(
      'hk_nir_robust_image_test',
      files('tests/test-hk-nir-robust-image.c',
            'hk_nir_lower_robust_image.c'),
      include_directories : [include_directories('.'), inc_include, inc_src],
      dependencies : [idep_nir, idep_mesautil],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_conditional_render_core',
    executable(
      'hk_conditional_render_core_test',
      files('tests/test-hk-conditional-render-core.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_pipeline_statistics_core',
    executable(
      'hk_pipeline_statistics_core_test',
      files('tests/test-hk-pipeline-statistics-core.c',
            'hk_pipeline_statistics_core.c'),
      include_directories : [inc_include, inc_src, inc_asahi],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

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
    'hk_ioto_arena_core',
    executable(
      'hk_ioto_arena_core_test',
      files('tests/test-hk-ioto-arena-core.c', 'hk_ioto_arena_core.c'),
      include_directories : [include_directories('.'), inc_include, inc_src],
      c_args : [c_msvc_compat_args],
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
        idep_libagx_shaders_h,
        idep_vulkan_runtime_headers,
        idep_vulkan_util_headers,
      ],
      c_args : [c_msvc_compat_args, '-Wno-error=pointer-arith'],
      gnu_symbol_visibility : 'hidden',
    ),
    suite : ['asahi'],
  )

  test(
    'hk_cdm_inline_merge',
    executable(
      'hk_cdm_inline_merge_test',
      files('tests/test-hk-cdm-inline-merge.c', 'hk_drm_render_core.c'),
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
        idep_libagx_shaders_h,
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
      # hk_cmd_buffer.h pulls in agx_helpers.h, which includes the generated
      # libagx_shaders.h. Without idep_libagx this target races the generator
      # and fails to build in a Vulkan-only configuration.
      dependencies : [idep_nir, idep_mesautil, idep_libagx,
                      idep_libagx_shaders_h],
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
      dependencies : [idep_vulkan_util],
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

  test('hk_vertex_attribs', executable('hk_vertex_attribs_test',
    files('tests/test-hk-vertex-attribs.c'),
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

  test(
    'hk_memory_budget_core',
    executable(
      'hk_memory_budget_core_test',
      files('tests/test-hk-memory-budget-core.c'),
      include_directories : [include_directories('.')],
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

  if with_platform_wayland
    g17p_xdg_shell = mod_wl.scan_xml(
      mod_wl.find_protocol('xdg-shell'),
      include_core_only : false,
    )
    g17p_presentation_time_server = mod_wl.scan_xml(
      mod_wl.find_protocol('presentation-time'),
      client : false,
      server : true,
      include_core_only : false,
    )
    executable(
      'g17p-wsi-wayland-triangle',
      files('tests/g17p_wsi_triangle.c'),
      g17p_xdg_shell,
      include_directories : [inc_include],
      dependencies : [dep_dl, dep_wayland_client],
      c_args : [c_msvc_compat_args, '-DG17P_WSI_WAYLAND'],
      gnu_symbol_visibility : 'hidden',
    )
    g17p_wayland_swapchain_ownership = executable(
      'g17p-vulkan-wayland-ownership',
      files('tests/g17p_vulkan_wayland_ownership.c'),
      g17p_presentation_time_server,
      include_directories : [inc_include],
      dependencies : [dep_dl, dep_thread, dep_wayland_client, dep_wayland_server],
      c_args : [c_msvc_compat_args],
      export_dynamic : true,
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

    # Public record/execute oracle for host set/reset plus a transfer fill,
    # event wait carrying a buffer dependency, transfer copy and GPU reset.
    executable(
      'g17p-vulkan-event-buffer',
      files('tests/g17p_vulkan_event_buffer.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'asahi-vulkan-g17p-layer-requirements',
      files('tests/g17p_vulkan_layer_requirements.c'),
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

    g17p_attachmentless_record = executable(
      'g17p-attachmentless-record',
      files('tests/g17p_attachmentless_record.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    g17p_offscreen_triangle = executable(
      'g17p-offscreen-triangle',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # Chromium 151 ANGLE reserves one float4 output for transform feedback.
    # This exact image oracle qualifies the resulting 68-user/72-total UVS
    # boundary before HK_G17P_UNVERIFIED_FEATURES=vertex_output_68 is promoted.
    executable(
      'g17p-vulkan-vertex-output-68',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_VERTEX_OUTPUT_68=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-secondary-command-buffer',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_SECONDARY_COMMAND_BUFFER=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # vkQuake's R_SubmitContexts shape. The single-secondary oracle above
    # executes exactly one secondary, so "several consecutive
    # vkCmdExecuteCommands calls inside one render pass" is untested: vkQuake
    # replays six world secondaries, then six entity secondaries, then one
    # each for sky, view model, particles and GUI, one call per secondary.
    # The world is the first such group and is exactly what goes missing on
    # G17P. This closes that hole with four secondaries painting four
    # disjoint quadrants in four distinct colours, replayed one call at a
    # time, so a dropped or misordered group names itself in the readback.
    executable(
      'g17p-vulkan-secondary-command-buffer-multi',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args,
                '-DG17P_SECONDARY_COMMAND_BUFFER_MULTI=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Exact public Vulkan oracle for G17P wide Bresenham line state.
    executable(
      'g17p-vulkan-bresenham-lines',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BRESENHAM_LINES=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Two-view/layer exact image oracle.  The same binary records and submits
    # against drm-shim when G17P_MULTIVIEW_RECORD_ONLY is set.
    executable(
      'g17p-vulkan-multiview',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MULTIVIEW=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Application-authored gl_Layer image oracle, independent of multiview's
    # driver-injected ViewIndex-to-Layer lowering.
    executable(
      'g17p-vulkan-layer-output',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_LAYER_OUTPUT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Four-instance viewport-index routing oracles. The four static/dynamic
    # viewport/scissor combinations cover one exact quadrant apiece.
    executable(
      'g17p-vulkan-multi-viewport',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MULTI_VIEWPORT=1'],
      gnu_symbol_visibility : 'hidden',
    )
    executable(
      'g17p-vulkan-multi-viewport-dynamic-viewport',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MULTI_VIEWPORT=2'],
      gnu_symbol_visibility : 'hidden',
    )
    executable(
      'g17p-vulkan-multi-viewport-dynamic-scissor',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MULTI_VIEWPORT=3'],
      gnu_symbol_visibility : 'hidden',
    )
    executable(
      'g17p-vulkan-multi-viewport-dynamic-both',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MULTI_VIEWPORT=4'],
      gnu_symbol_visibility : 'hidden',
    )

    # Divergent-loop quad operations and derivatives. This records against
    # drm-shim when G17P_SHADER_QUAD_CONTROL_RECORD_ONLY is set.
    executable(
      'g17p-vulkan-shader-quad-control',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_SHADER_QUAD_CONTROL=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-implicit-lod',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_IMPLICIT_LOD=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Exact two-plane 4:2:0 upload and immutable sampler-conversion oracle.
    # Record-only mode covers both native per-plane meta uploads plus the
    # G17P fragment shader's two indexed texture fetches under drm-shim.
    executable(
      'g17p-vulkan-ycbcr-420',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_YCBCR=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Desktop-compositor path: indexed combined-image descriptors, a UBO at
    # a nonzero offset, normalized implicit derivatives, linear filtering,
    # premultiplied alpha blend, and indexed drawing in one guarded oracle.
    executable(
      'g17p-vulkan-texture-alpha-composite',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # vkQuake world-surface shape. Every oracle above indexes at most one
    # sampled texture per fragment invocation, so a second sampled texture
    # that returns zero passes all of them while turning diffuse * lightmap
    # black on every BSP surface. This closes that hole with the two samplers
    # side by side in one descriptor set, as the narrower control.
    executable(
      'g17p-vulkan-texture-two-samplers-one-set',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TWO_SAMPLERS=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # The same two samples with the second texture moved to its own
    # descriptor set: vkQuake's exact binding shape, set 0 diffuse and set 1
    # lightmap, bound together before the world draw. Paired with the one-set
    # control above, a difference isolates the second descriptor set itself.
    executable(
      'g17p-vulkan-texture-two-samplers-two-sets',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TWO_SAMPLERS=1',
                '-DG17P_TWO_SAMPLERS_SPLIT_SETS=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Descriptor allocation/type matrix for the identical compositor sample.
    # These allocated-set controls pair with vk_meta's pushed combined and
    # pushed separate blit paths without involving mip generation.
    executable(
      'g17p-vulkan-texture-alpha-composite-allocated-combined',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_DESCRIPTOR_CELL=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-allocated-separate',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_DESCRIPTOR_CELL=2'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-allocated-combined-array',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_DESCRIPTOR_CELL=1',
                '-DG17P_COMPOSITE_DESCRIPTOR_ARRAY=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-allocated-separate-array',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_DESCRIPTOR_CELL=2',
                '-DG17P_COMPOSITE_DESCRIPTOR_ARRAY=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-allocated-combined-array-exact',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_DESCRIPTOR_CELL=1',
                '-DG17P_COMPOSITE_DESCRIPTOR_ARRAY=1',
                '-DG17P_COMPOSITE_ARRAY_NEAREST_EXACT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-allocated-separate-array-exact',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_DESCRIPTOR_CELL=2',
                '-DG17P_COMPOSITE_DESCRIPTOR_ARRAY=1',
                '-DG17P_COMPOSITE_ARRAY_NEAREST_EXACT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Native sRGB texture discriminator: uncompressed source, implicit mip-1
    # minification, LINEAR filtering, premultiplied source-over, and a BGRA8
    # sRGB target. Paired source formats isolate decode/swizzle from the
    # otherwise identical draw, sampler, blend, and target state.
    executable(
      'g17p-vulkan-texture-alpha-composite-rgba8-srgb',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include, inc_src],
      dependencies : [dep_dl, idep_mesautil],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_SRGB_SOURCE=1',
                '-DG17P_COMPOSITE_SRGB_TARGET=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-bgra8-srgb',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include, inc_src],
      dependencies : [dep_dl, idep_mesautil],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_SRGB_SOURCE=1',
                '-DG17P_COMPOSITE_SRGB_SOURCE_BGRA=1',
                '-DG17P_COMPOSITE_SRGB_TARGET=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # GTK unified-GPU icon path: premultiplied BGRA8 base upload, complete
    # same-image linear-blit mip chain, implicit minified sampling and
    # compositor-style alpha blending.
    g17p_generated_mips = executable(
      'g17p-vulkan-texture-alpha-composite-bgra8p-generated-mips',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_GENERATE_MIPS=1'],
      gnu_symbol_visibility : 'hidden',
    )

    g17p_generated_mips_odd = executable(
      'g17p-vulkan-texture-alpha-composite-bgra8p-generated-mips-odd',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_GENERATE_MIPS=1',
                '-DG17P_COMPOSITE_GENERATE_MIPS_ODD=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-64',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_SPLIT=1',
                '-DIMAGE_WIDTH=64', '-DIMAGE_HEIGHT=64'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-cw',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_SPLIT=2'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-64-cw',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_SPLIT=3',
                '-DIMAGE_WIDTH=64', '-DIMAGE_HEIGHT=64'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-fractional',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_SPLIT=1',
                '-DG17P_COMPOSITE_FRACTIONAL_INSET=1',
                '-DIMAGE_WIDTH=31', '-DIMAGE_HEIGHT=29'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-no-flat',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_NO_FLAT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-lod1',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_EXPLICIT_LOD=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-single',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_SINGLE_DRAW=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-narrow-single',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_NARROW_SINGLE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-narrow-single-x1',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_NARROW_SINGLE=1',
                '-DG17P_COMPOSITE_NARROW_X=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-narrow-single-x2',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_NARROW_SINGLE=1',
                '-DG17P_COMPOSITE_NARROW_X=2'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-narrow-single-x1-w3',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_NARROW_SINGLE=1',
                '-DG17P_COMPOSITE_NARROW_X=1',
                '-DG17P_COMPOSITE_NARROW_WIDTH=3'],
      gnu_symbol_visibility : 'hidden',
    )

    # Keep the failing odd-X helper edge but move it away from either quad
    # diagonal, then remove the quad seam entirely with an oversized triangle.
    executable(
      'g17p-vulkan-texture-alpha-composite-narrow-single-x1-y8',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_NARROW_SINGLE=1',
                '-DG17P_COMPOSITE_NARROW_X=1',
                '-DG17P_COMPOSITE_NARROW_Y=8',
                '-DG17P_COMPOSITE_NARROW_HEIGHT=2'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-narrow-single-x1-triangle',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_NARROW_SINGLE=1',
                '-DG17P_COMPOSITE_NARROW_X=1',
                '-DG17P_COMPOSITE_FULLSCREEN_TRIANGLE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-repeated-full',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_REPEATED_FULL=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-narrow-two-same',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_NARROW_TWO=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-alpha-composite-narrow-two-changed',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_ALPHA_COMPOSITE=1',
                '-DG17P_COMPOSITE_NARROW_TWO_CHANGED=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-grad',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_EXPLICIT_GRADIENT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-gather',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_GATHER=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-saturate-texture',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_SATURATE_TEXTURE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-shadow-compare',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_SHADOW_COMPARE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    foreach mode, op : {'1': 'tex', '2': 'txb', '3': 'txl', '4': 'txd',
                        '5': 'txf0', '6': 'txf1'}
      executable(
        'g17p-vulkan-min-lod-' + op,
        files('tests/g17p_offscreen_triangle.c'),
        include_directories : [inc_include],
        dependencies : [dep_dl],
        c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_MIN_LOD=' + mode],
        gnu_symbol_visibility : 'hidden',
      )
    endforeach

    executable(
      'g17p-vulkan-texel-fetch-lod',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXEL_FETCH_LOD=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-cube-sample',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_CUBE_SAMPLE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-texture-3d',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TEXTURE_3D=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc1-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # Exact 3D compressed upload/sampling oracle: two BC1 slices are copied
    # into a four-slice image at z=1 through padded source rows/slices.
    g17p_vulkan_bc1_3d_decode = executable(
      'g17p-vulkan-bc1-3d-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_COMPRESSED_3D=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Raw-copy round trip for the same padded, nonzero-z compressed volume.
    g17p_vulkan_bc1_3d_roundtrip = executable(
      'g17p-vulkan-bc1-3d-roundtrip',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_COMPRESSED_3D=1',
                '-DG17P_COMPRESSED_READBACK=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc1-srgb-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC1_SRGB_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc3-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC3_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc3-srgb-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC3_SRGB_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc2-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC2_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc2-srgb-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC2_SRGB_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc5-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC5_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc7-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC7_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc7-srgb-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC7_SRGB_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc6h-ufloat-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC6H_UFLOAT_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc6h-sfloat-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC6H_SFLOAT_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc4-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC4_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc4-snorm-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC4_SNORM_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bc5-snorm-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BC5_SNORM_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-etc2-rgba8-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_ETC2_RGBA8_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-astc4x4-decode',
      files('tests/g17p_vulkan_bc1_decode.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_ASTC_DECODE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-draw-indirect',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_INDIRECT_DRAW=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-draw-indexed-indirect',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_INDIRECT_DRAW=2'],
      gnu_symbol_visibility : 'hidden',
    )

    # vkQuake's world geometry and its lightmap are both produced by compute
    # in one command buffer.  These reproduce those two kernels at their real
    # resource shape -- ten bindings, push constants, shared memory, an
    # atomicAdd whose return value addresses a dynamic store, and an rgba8
    # imageStore -- and, separately, the staging command buffer that uploads
    # the lightmap's static inputs.  Stage 1 is the brush culling that fills
    # the indirect commands, stage 2 the lightmap, stage 3 both together.
    foreach vkquake_compute : [
      ['world-gen', '1'],
      ['lightmap', '2'],
      ['world-compute', '3'],
    ]
      executable(
        'g17p-vulkan-vkquake-' + vkquake_compute[0],
        files('tests/g17p_vulkan_vkquake_world_compute.c'),
        include_directories : [inc_include],
        dependencies : [dep_dl],
        c_args : [
          c_msvc_compat_args,
          '-DG17P_VKQ_STAGE=' + vkquake_compute[1],
        ],
        gnu_symbol_visibility : 'hidden',
      )
    endforeach

    # Robust indexed indirect draws are GPU-patched before VDM consumption.
    # This variant exercises both a freshly published and a reused HAL200
    # draw-state context in one command buffer.
    executable(
      'g17p-vulkan-draw-indexed-indirect-robust',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_INDIRECT_DRAW=5'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-draw-indirect-count',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_INDIRECT_DRAW=3'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-draw-indexed-indirect-count',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_INDIRECT_DRAW=4'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-index-restart',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_INDEX_RESTART=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-instance-rate',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_INSTANCE_RATE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-vertex-rgba8-unorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_VERTEX_FORMAT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-vertex-rg16-sint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_VERTEX_FORMAT=2'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-vertex-rg16-sfloat',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_VERTEX_FORMAT=3'],
      gnu_symbol_visibility : 'hidden',
    )

    # Same data, same expected image as the static R16G16_SFLOAT oracle above;
    # the vertex input arrives through vkCmdSetVertexInputEXT instead of the
    # pipeline. Nothing else in the tree exercises
    # VK_EXT_vertex_input_dynamic_state, which is the path where a sub-32-bit
    # attribute reaches agx_nir_lower_vbo_raw's narrow branch.
    foreach dvi : [['rgba8-unorm', '1'], ['rg16-sint', '2']]
      executable(
        'g17p-vulkan-dynamic-vertex-input-' + dvi[0],
        files('tests/g17p_offscreen_triangle.c'),
        include_directories : [inc_include],
        dependencies : [dep_dl],
        c_args : [c_msvc_compat_args, '-DG17P_VERTEX_FORMAT=' + dvi[1],
                  '-DG17P_DYNAMIC_VERTEX_INPUT=1'],
        gnu_symbol_visibility : 'hidden',
      )
    endforeach

    executable(
      'g17p-vulkan-dynamic-vertex-input-rg16-sfloat',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_VERTEX_FORMAT=3',
                '-DG17P_DYNAMIC_VERTEX_INPUT=1'],
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
      'g17p-msaa2-triangle',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=2'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-msaa4-triangle',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=4'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-sample-rate-shading4',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=4',
                '-DG17P_SAMPLE_RATE_SHADING=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-dynamic-sample-mask4',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=4',
                '-DG17P_MSAA_COVERAGE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-dynamic-alpha-to-coverage4',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=4',
                '-DG17P_MSAA_COVERAGE=2'],
      gnu_symbol_visibility : 'hidden',
    )

    # Variant 1 above pairs the dynamic sample mask with the per-sample
    # fragment shader, so it also demands sampleRateShading. G17P has no AGX3
    # per-sample fragment path, so that binary cannot pass there and a failure
    # would be blamed on the sample mask. This variant qualifies the mask
    # alone, with ordinary per-pixel invocation.
    executable(
      'g17p-vulkan-dynamic-sample-mask4-no-sample-shading',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=4',
                '-DG17P_MSAA_COVERAGE=3'],
      gnu_symbol_visibility : 'hidden',
    )

    # Unresolved multisample colour store, then a per-sample texelFetch back.
    executable(
      'g17p-vulkan-msaa-store-fetch2',
      files('tests/g17p_vulkan_msaa_store_fetch.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=2'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-msaa-store-fetch4',
      files('tests/g17p_vulkan_msaa_store_fetch.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=4'],
      gnu_symbol_visibility : 'hidden',
    )

    # Explicit hardware regression; do not register it with meson test.
    executable(
      'g17p-vulkan-none-preserve',
      files('tests/g17p_vulkan_none_preserve.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
      build_by_default : false,
    )

    # Multisample depth attachment: per-sample depth testing, and the ZLS
    # store/load round trip across two render passes.
    executable(
      'g17p-vulkan-msaa-depth2',
      files('tests/g17p_vulkan_msaa_depth_test.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=2'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-msaa-depth4',
      files('tests/g17p_vulkan_msaa_depth_test.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=4'],
      gnu_symbol_visibility : 'hidden',
    )

    # Standalone reduction of the Chromium/three.js keyframes workload: a
    # large 4x-MSAA depth render with enough repeated geometry to force the
    # parameter/partial-render paths, replayed for thousands of bounded frames.
    executable(
      'g17p-vulkan-keyframes-repro',
      files('tests/g17p_vulkan_msaa_depth_test.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=4',
                '-DG17P_KEYFRAMES_REPRO=1', '-DWIDTH=1760u',
                '-DHEIGHT=1232u'],
      gnu_symbol_visibility : 'hidden',
    )

    # Isolates the multisample resolve: a clear-only pass with no draw at all,
    # plus full-screen quads at 1x and Nx.
    executable(
      'g17p-vulkan-msaa-resolve2',
      files('tests/g17p_vulkan_msaa_resolve_probe.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=2'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-msaa-resolve4',
      files('tests/g17p_vulkan_msaa_resolve_probe.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_MSAA_SAMPLES=4'],
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

    # Public Vulkan/DRM-shim oracle for VK_KHR_16bit_storage's
    # storageInputOutput16 feature. The SPIR-V modules carry the explicit
    # StorageInputOutput16 capability, so this covers feature enumeration,
    # device enablement, pipeline compilation, command recording and submit.
    executable(
      'g17p-vulkan-f16-io',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_F16_IO=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for the bounded G17P RG16 UNORM colour path.
    executable(
      'g17p-vulkan-rg16-unorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RG16_UNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for the bounded G17P RGBA16 UNORM colour path.
    executable(
      'g17p-vulkan-rgba16-unorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RGBA16_UNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R16G16B16A16 UINT storage.
    executable(
      'g17p-vulkan-rgba16-uint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RGBA16_UINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R16G16B16A16 SINT storage.
    executable(
      'g17p-vulkan-rgba16-sint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RGBA16_SINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R32G32B32A32 UINT storage.
    executable(
      'g17p-vulkan-rgba32-uint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RGBA32_UINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R32G32B32A32 SINT storage.
    executable(
      'g17p-vulkan-rgba32-sint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RGBA32_SINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R32G32B32A32 SFLOAT storage.
    executable(
      'g17p-vulkan-rgba32-sfloat',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RGBA32_SFLOAT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P A2B10G10R10 UINT storage.
    executable(
      'g17p-vulkan-a2b10g10r10-uint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_A2B10G10R10_UINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P RG16 SNORM conversion/storage.
    executable(
      'g17p-vulkan-rg16-snorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RG16_SNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for the exact G17P R32 UINT colour path.
    executable(
      'g17p-vulkan-r32-uint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_R32_UINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P scalar R32 SFLOAT output.
    executable(
      'g17p-vulkan-r32-sfloat',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_R32_SFLOAT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P two-word RG32 SFLOAT output.
    executable(
      'g17p-vulkan-rg32-sfloat',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RG32_SFLOAT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R8G8 UINT colour storage.
    executable(
      'g17p-vulkan-rg8-uint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RG8_UINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P scalar R8 UINT colour storage.
    executable(
      'g17p-vulkan-r8-uint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_R8_UINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P scalar R8 SINT colour storage.
    executable(
      'g17p-vulkan-r8-sint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_R8_SINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for the exact G17P R32 SINT colour path.
    executable(
      'g17p-vulkan-r32-sint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_R32_SINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R16G16 UINT colour storage.
    executable(
      'g17p-vulkan-rg16-uint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RG16_UINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R16G16 SINT colour storage.
    executable(
      'g17p-vulkan-rg16-sint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RG16_SINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P two-word RG32 SINT output.
    executable(
      'g17p-vulkan-rg32-sint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RG32_SINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P two-word RG32 UINT output.
    executable(
      'g17p-vulkan-rg32-uint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RG32_UINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R8G8 SINT colour storage.
    executable(
      'g17p-vulkan-rg8-sint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RG8_SINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R8G8B8A8 UINT colour storage.
    executable(
      'g17p-vulkan-rgba8-uint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RGBA8_UINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R8G8B8A8 SINT colour storage.
    executable(
      'g17p-vulkan-rgba8-sint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RGBA8_SINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R8G8B8A8 SNORM colour storage.
    executable(
      'g17p-vulkan-rgba8-snorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RGBA8_SNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R8G8 SNORM colour storage.
    executable(
      'g17p-vulkan-rg8-snorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RG8_SNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R8 SNORM colour storage.
    executable(
      'g17p-vulkan-r8-snorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_R8_SNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P RGBA16 SNORM colour storage.
    executable(
      'g17p-vulkan-rgba16-snorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RGBA16_SNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R16 SNORM colour storage.
    executable(
      'g17p-vulkan-r16-snorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_R16_SNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R8 UNORM storage.
    executable(
      'g17p-vulkan-r8-unorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_R8_UNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P B8G8R8A8 SNORM storage.
    executable(
      'g17p-vulkan-bgra8-snorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BGRA8_SNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bgra8-unorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BGRA8_UNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-bgra8-srgb',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BGRA8_SRGB=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # R06c public Vulkan oracles for destination-dependent logic operations
    # and independent constant-colour/alpha blend factors.
    executable(
      'g17p-vulkan-logic-op-xor',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_LOGIC_OP_XOR=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-blend-constant',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BLEND_CONSTANT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # R06c public Vulkan oracle for two independent RGBA8 attachments across
    # clear, store, load, and additive blend transactions.
    executable(
      'g17p-vulkan-rgba8-mrt',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RGBA8_MRT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # SuperTuxKart's default advanced pipeline renders diffuse/specular into
    # two full-resolution GL_R11F_G11F_B10F attachments.
    executable(
      'g17p-vulkan-stk-r11g11b10-mrt',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [
        c_msvc_compat_args,
        '-DG17P_RGBA8_MRT=1',
        '-DG17P_STK_R11G11B10_MRT=1',
        '-DIMAGE_WIDTH=1280',
        '-DIMAGE_HEIGHT=720',
      ],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-dynamic-blend',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_DYNAMIC_BLEND=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-advanced-blend',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_ADVANCED_BLEND=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-rasterizer-discard',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RASTER_DISCARD=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P A2R10G10B10 UNORM storage.
    executable(
      'g17p-vulkan-a2r10g10b10-unorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_A2R10G10B10_UNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R16 UNORM colour storage.
    executable(
      'g17p-vulkan-r16-unorm',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_R16_UNORM=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R16 UINT colour storage.
    executable(
      'g17p-vulkan-r16-uint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_R16_UINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R16 SINT colour storage.
    executable(
      'g17p-vulkan-r16-sint',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_R16_SINT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P R16 SFLOAT colour storage.
    executable(
      'g17p-vulkan-r16-sfloat',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_R16_SFLOAT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P RG16 SFLOAT colour storage.
    executable(
      'g17p-vulkan-rg16-sfloat',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RG16_SFLOAT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # F01b public Vulkan oracle for exact G17P RGBA16 SFLOAT colour storage.
    executable(
      'g17p-vulkan-rgba16-sfloat',
      files('tests/g17p_offscreen_triangle.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RGBA16_SFLOAT=1'],
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

    # Four identical draws in one render pass with no state recorded between
    # them. This is the control for the inter-draw VDM barrier: the barrier is
    # attached to render-state republication, so a stream of body-only draws
    # must collect none of them.
    g17p_repeat_draw = executable(
      'g17p-vulkan-repeat-draw',
      files('tests/g17p_render_only.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_REPEAT_DRAW=4'],
      gnu_symbol_visibility : 'hidden',
    )

    # Record the exact tiny render shape used by startup-sized engine targets.
    # G17P_TINY_CLEAR_ONLY=1 records the corresponding clear-only stream.
    executable(
      'g17p-vulkan-tiny-render-record',
      files('tests/g17p_render_only.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_TINY_RENDER_RECORD=1',
                '-DIMAGE_WIDTH=2', '-DIMAGE_HEIGHT=2'],
      gnu_symbol_visibility : 'hidden',
    )

    # Record more live graphics command buffers than the 192-slot G17P VDM
    # aperture. Their compact command spans must share command-pool slabs.
    executable(
      'g17p-vulkan-vdm-capacity',
      files('tests/g17p_render_only.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_VDM_CAPACITY=1'],
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
      'g17p-vulkan-queries',
      files('tests/g17p_vulkan_queries.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
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

    # v48: per-wait device-lost check through the VM-status mirror or the
    # GET_PARAMS query, against the DRM shim (see the probe's header).
    executable(
      'g17p-vm-status-mirror',
      files('tests/g17p_vm_status_mirror.c'),
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

    # Generic encodability probe: takes SPIR-V paths on the command line and
    # reports whether vkCreateComputePipelines accepts each one. This is the
    # tool that measures README-agx3-packer-limits.md's matrix.
    # VK_KHR_ray_tracing_pipeline host-side probe.  Compiles every ray
    # tracing stage into the megakernel, imports a pipeline library, and
    # records all three trace-rays forms under the no-op drm-shim.  It
    # executes nothing; see docs/G17P_RAYTRACING_PIPELINE_ABI.md for what a
    # pass does and does not establish.
    executable(
      'hk-rt-pipeline-probe',
      files('tests/hk_rt_pipeline_probe.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-graphics-probe',
      files('tests/g17p_vulkan_graphics_probe.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # Offline proof that vkCmdDispatchIndirect records a native G17P indirect
    # launch pair.  Runs under the noop drm-shim and stops at
    # vkEndCommandBuffer: it asserts the recording, never execution.
    g17p_dispatch_indirect = executable(
      'g17p-vulkan-dispatch-indirect',
      files('tests/g17p_vulkan_dispatch_indirect.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    g17p_sequential_dispatch = executable(
      'g17p-vulkan-sequential-dispatch',
      files('tests/g17p_vulkan_dispatch_indirect.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_SEQUENTIAL_DISPATCH=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Offline proof that transform feedback records on G17P through the native
    # geometry path and G17P-compiled libagx counter-copy kernel.  --execute is
    # the hardware mode with exact buffer, counter, and guard checks.
    # Firefox/Zink six-range native vertex program: six attributes, a draw
    # parameter, a vertex texture sample, three descriptor sets, and a promoted
    # constant. Recording alone reaches hk_upload_usc_words(), so the drm-shim
    # oracle in ../drm-shim/meson.build runs it without a GPU.
    g17p_vertex_range_pressure = executable(
      'g17p-vulkan-vertex-range-pressure',
      files('tests/g17p_vulkan_vertex_range_pressure.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # SuperTuxKart-shaped frame: many small clear/store render passes with
    # draws, all in one command buffer. Recording alone composes every Ioto
    # program, so the drm-shim oracle in ../drm-shim/meson.build counts the
    # executable-window mappings it takes without a GPU.
    g17p_ioto_program_pressure = executable(
      'g17p-vulkan-ioto-program-pressure',
      files('tests/g17p_vulkan_ioto_program_pressure.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    g17p_transform_feedback = executable(
      'g17p-vulkan-transform-feedback',
      files('tests/g17p_vulkan_transform_feedback.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-pipeline-probe',
      files('tests/g17p_vulkan_pipeline_probe.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # Q02 public pipeline-cache persistence oracle. Runs without submitting GPU
    # work and checks G17 cache identity, import, merge and explicit cache hit.
    g17p_pipeline_cache = executable(
      'g17p-vulkan-pipeline-cache',
      files('tests/g17p_vulkan_pipeline_cache.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # M02 public no-submit buffer-memory lifetime oracle.
    g17p_buffer_memory = executable(
      'g17p-vulkan-buffer-memory',
      files('tests/g17p_vulkan_buffer_memory.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # Public no-submit buffer-device-address capture/replay oracle.
    g17p_bda_capture = executable(
      'g17p-vulkan-bda-capture',
      files('tests/g17p_vulkan_bda_capture.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # Q01b public no-submit partial/unavailable query-result ABI oracle.
    g17p_query_results = executable(
      'g17p-vulkan-query-results',
      files('tests/g17p_vulkan_query_results.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # S04b public no-submit timeline host/object-lifetime oracle.
    g17p_timeline_host = executable(
      'g17p-vulkan-timeline-host',
      files('tests/g17p_vulkan_timeline_host.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # Public no-submit VK_EXT_memory_budget allocation-lifetime oracle.
    g17p_memory_budget = executable(
      'g17p-vulkan-memory-budget',
      files('tests/g17p_vulkan_memory_budget.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # Public no-submit VK_KHR_maintenance4 requirements-equivalence oracle.
    g17p_maintenance4 = executable(
      'g17p-vulkan-maintenance4',
      files('tests/g17p_vulkan_maintenance4.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # Public no-submit VK_KHR_maintenance5 usage2/layout oracle.
    g17p_maintenance5 = executable(
      'g17p-vulkan-maintenance5',
      files('tests/g17p_vulkan_maintenance5.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # Public no-submit VK_KHR_maintenance6 properties/bind-status oracle.
    g17p_maintenance6 = executable(
      'g17p-vulkan-maintenance6',
      files('tests/g17p_vulkan_maintenance6.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # Public record-only VK_KHR_push_descriptor allocation/push matrix.
    g17p_push_descriptor = executable(
      'g17p-vulkan-push-descriptor',
      files('tests/g17p_vulkan_push_descriptor.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # One-layer sampler2DArray NEAREST allocated-descriptor pixel control.
    g17p_push_descriptor_array_control = executable(
      'g17p-vulkan-push-descriptor-array-control',
      files('tests/g17p_vulkan_push_descriptor.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_ARRAY_CONTROL=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Raw allocated sampler2DArray implicit/explicit LOD and layer matrix.
    g17p_array_lod_control = executable(
      'g17p-vulkan-array-lod-control',
      files('tests/g17p_vulkan_push_descriptor.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_ARRAY_LOD_CONTROL=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Hardware-ready exact pipeline-statistics oracle. Each invocation accepts
    # exactly one graphics or compute counter so failures stay attributable.
    executable(
      'g17p-vulkan-pipeline-statistics',
      files('tests/g17p_vulkan_queries.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_PIPELINE_STATISTICS=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Exact vertex+fragment SSBO store/atomic oracle. Record-only runs under
    # the G17P drm-shim; execution additionally checks every result and guard.
    executable(
      'g17p-vulkan-graphics-side-effects',
      files('tests/g17p_render_only.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_GRAPHICS_SIDE_EFFECTS=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Fragment-only capability/semantics oracle. Exact guarded storage and
    # atomic results are checked while the vertex feature stays disabled.
    executable(
      'g17p-vulkan-fragment-side-effects',
      files('tests/g17p_render_only.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_FRAGMENT_SIDE_EFFECTS=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Game-feature probe: enables every advertised Vulkan 1.1/1.2/1.3
    # feature, then compiles AND records a dispatch for each SPIR-V module.
    # The record half is what catches a feature the compiler accepts but the
    # G17P native admission gates refuse at vkEndCommandBuffer.
    executable(
      'g17p-vulkan-feature-probe',
      files('tests/g17p_vulkan_feature_probe.c'),
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

    # Public Vulkan queue-progress oracle.  The shim run proves that the
    # pre-signalled bootstrap point, first real submit, and QueueWaitIdle all
    # name distinct/correct timeline points; execution additionally proves
    # that the first dispatch is complete when QueueWaitIdle returns.
    g17p_queue_progress = executable(
      'g17p-vulkan-queue-progress',
      files('tests/g17p_vulkan_compute_min.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_QUEUE_PROGRESS=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Public Vulkan WSI oracle.  The drm-shim injects acquire payload and
    # present sync-file failures, then checks exact rollback/fallback ordering,
    # image reuse, and complete lifetime balance.
    g17p_acquire_rollback = executable(
      'g17p-vulkan-acquire-rollback',
      files('tests/g17p_vulkan_acquire_rollback.c'),
      include_directories : [inc_include, inc_src],
      dependencies : [
        dep_dl,
        idep_vulkan_runtime_headers,
        idep_vulkan_util_headers,
        idep_vulkan_wsi_headers,
      ],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    # One bounded conditional-rendering case per process. Submit-only mode
    # proves direct, indirect, inverted and inherited command streams under
    # the drm-shim; execution checks every payload word and both guards.
    g17p_conditional_compute = executable(
      'g17p-vulkan-conditional-rendering',
      files('tests/g17p_vulkan_compute_min.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_CONDITIONAL_RENDERING=1'],
      gnu_symbol_visibility : 'hidden',
    )

    # Graphics half of the same qualification: direct, robust indexed,
    # indirect, count and clear-attachments paths are checked by exact RGBA8
    # pixels and precise occlusion counts.
    g17p_conditional_graphics = executable(
      'g17p-vulkan-conditional-rendering-graphics',
      files('tests/g17p_render_only.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args,
                '-DG17P_OCCLUSION_QUERY=1',
                '-DG17P_CONDITIONAL_RENDERING=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-workgroup-shared',
      files('tests/g17p_vulkan_compute_min.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_WORKGROUP_SHARED=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-workgroup-shared-1024',
      files('tests/g17p_vulkan_compute_min.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_WORKGROUP_SHARED_1024=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-compute-atomic2d',
      files('tests/g17p_vulkan_compute_min.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_COMPUTE_ATOMIC_2D=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p_vulkan_blit_scaled',
      files('tests/g17p_vulkan_blit_scaled.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-copy-rgb24',
      files('tests/g17p_vulkan_copy_rgb24.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

  executable(
      'g17p-vulkan-image-copy-subregion',
      files('tests/g17p_vulkan_image_copy_subregion.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    g17p_vulkan_image_copy_3d = executable(
      'g17p-vulkan-image-copy-3d',
      files('tests/g17p_vulkan_image_copy_subregion.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_COPY_IMAGE_3D=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-clear-image',
      files('tests/g17p_vulkan_image_copy_subregion.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_CLEAR_IMAGE'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-blit-image-nearest-flip',
      files('tests/g17p_vulkan_image_copy_subregion.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BLIT_IMAGE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-blit-image-linear-flip',
      files('tests/g17p_vulkan_image_copy_subregion.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BLIT_IMAGE=2'],
      gnu_symbol_visibility : 'hidden',
    )

    g17p_vulkan_stencil_blit = executable(
      'g17p-vulkan-stencil-blit',
      files('tests/g17p_vulkan_image_copy_subregion.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_STENCIL_BLIT=1'],
      gnu_symbol_visibility : 'hidden',
    )

    g17p_vulkan_layered_mip_blit = executable(
      'g17p-vulkan-blit-cube-layers-mip',
      files('tests/g17p_vulkan_image_copy_subregion.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_BLIT_IMAGE=3'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-resolve-image4',
      files('tests/g17p_vulkan_image_copy_subregion.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_RESOLVE_IMAGE=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-null-descriptor',
      files('tests/g17p_vulkan_compute_min.c'),
      include_directories : [inc_include],
      dependencies : [dep_dl],
      c_args : [c_msvc_compat_args, '-DG17P_NULL_DESCRIPTOR=1'],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-robust-buffer2',
      files('tests/g17p_vulkan_robust_buffer2.c'),
      include_directories : [inc_include],
      dependencies : [cc.find_library('vulkan')],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-robust-image2',
      files('tests/g17p_vulkan_robust_image2.c'),
      include_directories : [inc_include],
      dependencies : [cc.find_library('vulkan')],
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
    )

    executable(
      'g17p-vulkan-pipeline-robustness',
      files('tests/g17p_vulkan_robust_buffer2.c'),
      include_directories : [inc_include],
      dependencies : [cc.find_library('vulkan')],
      c_args : [c_msvc_compat_args, '-DG17P_PIPELINE_ROBUSTNESS=1'],
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

    # Hyprland draws a window border as one scissored draw per rectangle of
    # the damage-intersected border ring, so an ordinary border is four
    # adjacent scissor rectangles sharing their edges. Every G17P scissor
    # witness in this tree is full-frame or anchored at x=0,y=0, and none of
    # them looks at a pixel, so the seam behaviour of the HAL200 raster
    # record is unwitnessed. This oracle makes the framebuffer a coverage
    # count and reads the seams directly. --record-only is the shim gate;
    # the pixel modes need a real G17P.
    g17p_scissor_ring = executable(
      'g17p-vulkan-scissor-ring',
      files('tests/g17p_vulkan_scissor_ring.c'),
      include_directories : [inc_include],
      dependencies : [cc.find_library('vulkan')],
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
