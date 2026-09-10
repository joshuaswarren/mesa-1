# Copyright 2019 Broadcom
# SPDX-License-Identifier: MIT

libasahi_noop_drm_shim = shared_library(
  'asahi_noop_drm_shim',
  ['asahi_noop.c', 'asahi_g17p_stream_validator.c'],
  objects : libasahi_decode.extract_objects(
    'agx3_cdm_decode_core.c',
    'agx_ioto_core.c',
    'agx_ppp_hal200_graphics_core.c',
    'agx_ppp_viewport_payload_core.c',
    'agx_ppp_w_clamp_cull2_core.c',
    'agx_vdm_barrier_core.c',
    'agx_vdm_decode_core.c',
    'agx_vdm_hal200_bind_tail_core.c',
    'agx_vdm_index_list_core.c',
    'agx_vdm_ppp_state_core.c',
    'agx_vdm_state_core.c',
    'agx_vdm_state_update_core.c',
    'agx_vdm_stream_link_core.c',
    'agx_vdm_stream_terminate_core.c',
    'agx_vdm_tessellate_core.c',
  ),
  include_directories: [inc_include, inc_src, inc_asahi],
  dependencies: [dep_drm_shim, dep_valgrind],
  link_with : [libagx3b_disasm],
  gnu_symbol_visibility : 'hidden',
  install : true,
)

if with_tests and with_asahi_vk and host_machine.system() == 'linux'
  g17p_queue_progress_environment = {
    'ASAHI_SHIM_CHIP' : 'g17p',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM' : '1',
    'LD_PRELOAD' : libasahi_noop_drm_shim.full_path(),
    'MESA_SHADER_CACHE_DISABLE' : 'true',
    'VK_DRIVER_FILES' : _dev_icd.full_path(),
  }

  test(
    'hk_g17p_pipeline_cache_persistence',
    g17p_pipeline_cache,
    env : g17p_queue_progress_environment,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-pipeline-cache'],
    timeout : 30,
  )

  test(
    'hk_g17p_buffer_memory_lifetime',
    g17p_buffer_memory,
    env : g17p_queue_progress_environment,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-buffer-memory'],
    timeout : 30,
  )

  test(
    'hk_g17p_bda_capture_replay',
    g17p_bda_capture,
    env : g17p_queue_progress_environment,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-bda-capture'],
    timeout : 30,
  )

  test(
    'hk_g17p_query_result_modes',
    g17p_query_results,
    env : g17p_queue_progress_environment,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-query-results'],
    timeout : 30,
  )

  test(
    'hk_g17p_timeline_host_lifetime',
    g17p_timeline_host,
    env : g17p_queue_progress_environment + {
      'ASAHI_SHIM_TIMELINE_HOST_VALUES' : '1',
    },
    depends : _dev_icd,
    suite : ['asahi', 'g17p-timeline-host'],
    timeout : 30,
  )

  test(
    'hk_g17p_memory_budget_lifetime',
    g17p_memory_budget,
    env : g17p_queue_progress_environment + {
      'HK_SYSMEM' : '1073741824',
    },
    depends : _dev_icd,
    suite : ['asahi', 'g17p-memory-budget'],
    timeout : 30,
  )

  test(
    'hk_g17p_maintenance4_requirements',
    g17p_maintenance4,
    env : g17p_queue_progress_environment,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-maintenance4'],
    timeout : 30,
  )

  test(
    'hk_g17p_maintenance5_usage2_layout',
    g17p_maintenance5,
    env : g17p_queue_progress_environment,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-maintenance5'],
    timeout : 30,
  )

  test(
    'hk_g17p_maintenance6_bind_status',
    g17p_maintenance6,
    env : g17p_queue_progress_environment,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-maintenance6'],
    timeout : 30,
  )

  test(
    'hk_g17p_push_descriptor_record',
    g17p_push_descriptor,
    env : g17p_queue_progress_environment + {
      'G17P_PUSH_DESCRIPTOR_RECORD_ONLY' : '1',
    },
    depends : _dev_icd,
    suite : ['asahi', 'g17p-push-descriptor'],
    timeout : 30,
  )

  test(
    'hk_g17p_push_descriptor_array_control_record',
    g17p_push_descriptor_array_control,
    env : g17p_queue_progress_environment + {
      'G17P_PUSH_DESCRIPTOR_RECORD_ONLY' : '1',
    },
    depends : _dev_icd,
    suite : ['asahi', 'g17p-push-descriptor-array'],
    timeout : 30,
  )

  test(
    'hk_g17p_array_lod_control_record',
    g17p_array_lod_control,
    env : g17p_queue_progress_environment + {
      'G17P_PUSH_DESCRIPTOR_RECORD_ONLY' : '1',
    },
    depends : _dev_icd,
    suite : ['asahi', 'g17p-array-lod-control'],
    timeout : 30,
  )

  g17p_compressed_3d_environment = {
    'ASAHI_SHIM_CHIP' : 'g17p',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM' : '1',
    'G17P_COMPRESSED_DECODE_SUBMIT_ONLY' : '1',
    'LD_PRELOAD' : libasahi_noop_drm_shim.full_path(),
    'MESA_SHADER_CACHE_DISABLE' : 'true',
    'VK_DRIVER_FILES' : _dev_icd.full_path(),
  }

  g17p_queue_progress_env = environment(g17p_queue_progress_environment)
  g17p_queue_progress_env.set(
    'ASAHI_SHIM_EXPECT_QUEUE_PROGRESS_LIFECYCLE', 'success')
  g17p_queue_progress_env.set('ASAHI_SHIM_G17P_EXPECT_COMPUTE', '1')
  test(
    'hk_g17p_queue_progress_first_submit_wait_idle',
    g17p_queue_progress,
    args : ['--shim'],
    env : g17p_queue_progress_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-queue-progress'],
    timeout : 30,
  )

  # The Firefox/Zink six-range vertex program only has to be RECORDED to reach
  # hk_upload_usc_words(); a graphics range budget below the number of ranges
  # the collector emits refuses fail-closed at vkEndCommandBuffer. No submit,
  # so no stream-validator expectation is set.
  g17p_vertex_range_pressure_env = environment({
    'ASAHI_SHIM_CHIP' : 'g17p',
    'LD_PRELOAD' : libasahi_noop_drm_shim.full_path(),
    'MESA_SHADER_CACHE_DISABLE' : 'true',
    'VK_DRIVER_FILES' : _dev_icd.full_path(),
  })
  test(
    'hk_g17p_vertex_range_pressure',
    g17p_vertex_range_pressure,
    env : g17p_vertex_range_pressure_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-vertex-range-pressure'],
    timeout : 60,
  )

  g17p_xfb_draw_env = environment(g17p_queue_progress_environment)
  g17p_xfb_draw_env.set('ASAHI_SHIM_G17P_EXPECT_XFB', '1')
  test(
    'hk_g17p_transform_feedback_draw_byte_count',
    g17p_transform_feedback,
    args : ['--submit-only'],
    env : g17p_xfb_draw_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-transform-feedback'],
    timeout : 30,
  )

  g17p_queue_progress_init_failure_env = environment(
    g17p_queue_progress_environment)
  g17p_queue_progress_init_failure_env.set(
    'ASAHI_SHIM_EXPECT_QUEUE_PROGRESS_LIFECYCLE', 'init-failure')
  g17p_queue_progress_init_failure_env.set(
    'ASAHI_SHIM_FAIL_QUEUE_TIMELINE_SIGNAL_ONCE', '1')
  test(
    'hk_g17p_queue_progress_init_failure',
    g17p_queue_progress,
    args : ['--init-failure'],
    env : g17p_queue_progress_init_failure_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-queue-progress'],
    timeout : 30,
  )

  g17p_queue_progress_submit_failure_env = environment(
    g17p_queue_progress_environment)
  g17p_queue_progress_submit_failure_env.set(
    'ASAHI_SHIM_EXPECT_QUEUE_PROGRESS_LIFECYCLE', 'submit-failure')
  g17p_queue_progress_submit_failure_env.set(
    'ASAHI_SHIM_FAIL_SUBMIT_ONCE', '1')
  test(
    'hk_g17p_queue_progress_submit_failure',
    g17p_queue_progress,
    args : ['--submit-failure'],
    env : g17p_queue_progress_submit_failure_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-queue-progress'],
    timeout : 30,
  )

  g17p_acquire_rollback_environment = {
    'ASAHI_SHIM_CHIP' : 'g17p',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM' : '1',
    'LD_PRELOAD' : libasahi_noop_drm_shim.full_path(),
    'MESA_SHADER_CACHE_DISABLE' : 'true',
    'VK_DRIVER_FILES' : _dev_icd.full_path(),
  }

  foreach case : ['semaphore-failure', 'fence-failure']
    g17p_acquire_rollback_env = environment(
      g17p_acquire_rollback_environment)
    g17p_acquire_rollback_env.set(
      'ASAHI_SHIM_EXPECT_ACQUIRE_ROLLBACK', case)
    test(
      'hk_g17p_acquire_rollback_' + case,
      g17p_acquire_rollback,
      args : ['--' + case],
      env : g17p_acquire_rollback_env,
      depends : _dev_icd,
      suite : ['asahi', 'g17p-acquire-rollback'],
      timeout : 30,
    )
  endforeach

  g17p_compressed_3d_env = environment(g17p_compressed_3d_environment)
  test(
    'hk_g17p_compressed_3d_upload',
    g17p_vulkan_bc1_3d_decode,
    env : g17p_compressed_3d_env,
    suite : ['asahi', 'g17p-compressed-textures'],
    timeout : 30,
  )

  g17p_compressed_3d_fail_env = environment(g17p_compressed_3d_environment)
  g17p_compressed_3d_fail_env.set('ASAHI_SHIM_FAIL_SUBMIT_ONCE', '1')
  g17p_compressed_3d_fail_env.set(
    'G17P_COMPRESSED_DECODE_EXPECT_SUBMIT_FAILURE', '1')
  test(
    'hk_g17p_compressed_3d_submit_failure',
    g17p_vulkan_bc1_3d_decode,
    env : g17p_compressed_3d_fail_env,
    suite : ['asahi', 'g17p-compressed-textures'],
    timeout : 30,
  )

  foreach phase : ['import', 'export']
    case = 'present-' + phase + '-failure'
    g17p_present_rollback_env = environment(
      g17p_acquire_rollback_environment)
    g17p_present_rollback_env.set(
      'ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK', case)
    test(
      'hk_g17p_present_dmabuf_' + phase + '_fallback',
      g17p_acquire_rollback,
      args : ['--' + case],
      env : g17p_present_rollback_env,
      depends : _dev_icd,
      suite : ['asahi', 'g17p-present-rollback'],
      timeout : 30,
    )
  endforeach

  g17p_compressed_roundtrip_env = environment(
    g17p_compressed_3d_environment)
  test(
    'hk_g17p_compressed_3d_raw_roundtrip',
    g17p_vulkan_bc1_3d_roundtrip,
    env : g17p_compressed_roundtrip_env,
    suite : ['asahi', 'g17p-compressed-textures'],
    timeout : 30,
  )

  g17p_compressed_roundtrip_fail_env = environment(
    g17p_compressed_3d_environment)
  g17p_compressed_roundtrip_fail_env.set('ASAHI_SHIM_FAIL_SUBMIT_ONCE', '1')
  g17p_compressed_roundtrip_fail_env.set(
    'G17P_COMPRESSED_DECODE_EXPECT_SUBMIT_FAILURE', '1')
  test(
    'hk_g17p_compressed_3d_raw_roundtrip_submit_failure',
    g17p_vulkan_bc1_3d_roundtrip,
    env : g17p_compressed_roundtrip_fail_env,
    suite : ['asahi', 'g17p-compressed-textures'],
    timeout : 30,
  )

  g17p_present_submit_rollback_env = environment(
    g17p_acquire_rollback_environment)
  g17p_present_submit_rollback_env.set(
    'ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK', 'present-submit-failure')
  test(
    'hk_g17p_present_submit_rollback',
    g17p_acquire_rollback,
    args : ['--present-submit-failure'],
    env : g17p_present_submit_rollback_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-present-submit-rollback'],
    timeout : 30,
  )

  g17p_empty_submit_failure_env = environment(
    g17p_acquire_rollback_environment)
  g17p_empty_submit_failure_env.set(
    'ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK', 'empty-submit-failure')
  test(
    'hk_g17p_empty_submit_failure',
    g17p_acquire_rollback,
    args : ['--empty-submit-failure'],
    env : g17p_empty_submit_failure_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-empty-submit-failure'],
    timeout : 30,
  )

  g17p_release_rollback_env = environment(
    g17p_acquire_rollback_environment)
  g17p_release_rollback_env.set(
    'ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK', 'release-failure')
  test(
    'hk_g17p_release_rollback',
    g17p_acquire_rollback,
    args : ['--release-failure'],
    env : g17p_release_rollback_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-release-rollback'],
    timeout : 30,
  )

  g17p_present_priority_env = environment(
    g17p_acquire_rollback_environment)
  g17p_present_priority_env.set(
    'ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK', 'present-priority')
  test(
    'hk_g17p_present_priority',
    g17p_acquire_rollback,
    args : ['--present-priority'],
    env : g17p_present_priority_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-present-priority'],
    timeout : 30,
  )

  g17p_image_copy_3d_environment = {
    'ASAHI_SHIM_CHIP' : 'g17p',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM' : '1',
    'LD_PRELOAD' : libasahi_noop_drm_shim.full_path(),
    'MESA_SHADER_CACHE_DISABLE' : 'true',
    'VK_DRIVER_FILES' : _dev_icd.full_path(),
  }
  g17p_image_copy_3d_env = environment(g17p_image_copy_3d_environment)
  g17p_image_copy_3d_env.set(
    'ASAHI_SHIM_G17P_EXPECT_FRAG_CONSTANT_DRAWS', '7')
  test(
    'hk_g17p_image_copy_3d',
    g17p_vulkan_image_copy_3d,
    args : ['--submit-only'],
    env : g17p_image_copy_3d_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-image-copy-3d'],
    timeout : 30,
  )

  # Hyprland's renderBorder issues four adjacent scissored draws into one
  # render pass, changing only the scissor between them -- the exact shape
  # that faulted the GMMU on hardware and that ZINK_DEBUG=sync cured. Assert
  # the three inter-draw VDM barriers are present and that no draw
  # republishes render state without one in front of it.
  #
  # Both halves matter. Without the count, a driver that stopped emitting
  # passes; without the unbarriered-republication check, a barrier emitted in
  # the wrong place still satisfies the count. Reverting the emitter yields
  # "0/3 barriers, 3/0 unbarriered", so this test can fail.
  g17p_interdraw_barrier_env = environment(g17p_image_copy_3d_environment)
  g17p_interdraw_barrier_env.set(
    'ASAHI_SHIM_G17P_EXPECT_INTERDRAW_BARRIERS', '3:4')
  test(
    'hk_g17p_interdraw_vdm_barrier',
    g17p_scissor_ring,
    args : ['ring', '--record-only'],
    env : g17p_interdraw_barrier_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-interdraw-barrier'],
    timeout : 30,
  )

  # The same four draws with HK_G17P_SPLIT_DIRECT_DRAWS=1, which reproduces
  # what ZINK_DEBUG=sync does to the graphics side: one control stream per
  # draw. Each stream opens with its own barrier, so the barrier count is
  # identical (4) while the render passes are not (4 versus 1). That is the
  # whole performance argument, and pinning it here keeps a future change
  # from quietly turning the cheap fix back into the expensive one.
  g17p_split_draw_barrier_env = environment(g17p_image_copy_3d_environment)
  g17p_split_draw_barrier_env.set('HK_G17P_SPLIT_DIRECT_DRAWS', '1')
  g17p_split_draw_barrier_env.set(
    'ASAHI_SHIM_G17P_EXPECT_INTERDRAW_BARRIERS', '0:4')
  test(
    'hk_g17p_interdraw_vdm_barrier_split_control',
    g17p_scissor_ring,
    args : ['ring', '--record-only'],
    env : g17p_split_draw_barrier_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-interdraw-barrier'],
    timeout : 30,
  )

  # The narrowness control. Four identical draws with no state between them
  # are body-only: they reuse what the first draw latched, so nothing the VDM
  # is still dereferencing gets overwritten and no barrier is owed. Expect
  # zero, against four draws. ZINK_DEBUG=sync cannot tell this case apart and
  # would split the render pass four ways here too.
  g17p_repeat_draw_barrier_env = environment(g17p_image_copy_3d_environment)
  g17p_repeat_draw_barrier_env.set(
    'ASAHI_SHIM_G17P_EXPECT_INTERDRAW_BARRIERS', '0:4')
  test(
    'hk_g17p_interdraw_vdm_barrier_body_only',
    g17p_repeat_draw,
    env : g17p_repeat_draw_barrier_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-interdraw-barrier'],
    timeout : 30,
  )

  g17p_image_copy_3d_fail_env = environment(
    g17p_image_copy_3d_environment)
  g17p_image_copy_3d_fail_env.set('ASAHI_SHIM_FAIL_SUBMIT_ONCE', '1')
  test(
    'hk_g17p_image_copy_3d_submit_failure',
    g17p_vulkan_image_copy_3d,
    args : ['--expect-submit-failure'],
    env : g17p_image_copy_3d_fail_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-image-copy-3d'],
    timeout : 30,
  )

  g17p_stencil_blit_env = environment(g17p_queue_progress_environment)
  g17p_stencil_blit_env.set(
    'ASAHI_SHIM_G17P_EXPECT_STENCIL_BLIT', '1')
  test(
    'hk_g17p_stencil_blit_meta_fallback',
    g17p_vulkan_stencil_blit,
    args : ['--submit-only'],
    env : g17p_stencil_blit_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-stencil-blit'],
    timeout : 30,
  )

  if with_platform_wayland
    g17p_wayland_ownership_env = environment(
      g17p_acquire_rollback_environment)
    g17p_wayland_ownership_env.set('MESA_VK_WSI_DEBUG', 'sw')
    g17p_wayland_ownership_env.set(
      'ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK', 'wayland-ownership')
    test(
      'hk_g17p_wayland_swapchain_ownership',
      g17p_wayland_swapchain_ownership,
      env : g17p_wayland_ownership_env,
      depends : _dev_icd,
      suite : ['asahi', 'g17p-wayland-swapchain-ownership'],
      timeout : 30,
    )

    g17p_wayland_present_alloc_env = environment(
      g17p_acquire_rollback_environment)
    g17p_wayland_present_alloc_env.set('MESA_VK_WSI_DEBUG', 'sw')
    g17p_wayland_present_alloc_env.set(
      'ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK', 'wayland-present-allocation')
    test(
      'hk_g17p_wayland_present_allocation_failure',
      g17p_wayland_swapchain_ownership,
      args : ['--present-allocation-failure'],
      env : g17p_wayland_present_alloc_env,
      depends : _dev_icd,
      suite : ['asahi', 'g17p-wayland-present-allocation'],
      timeout : 30,
    )

    g17p_wayland_surface_retry_env = environment(
      g17p_acquire_rollback_environment)
    g17p_wayland_surface_retry_env.set('MESA_VK_WSI_DEBUG', 'sw')
    g17p_wayland_surface_retry_env.set(
      'ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK', 'wayland-surface-init-retry')
    test(
      'hk_g17p_wayland_surface_init_retry',
      g17p_wayland_swapchain_ownership,
      args : ['--surface-init-retry'],
      env : g17p_wayland_surface_retry_env,
      depends : _dev_icd,
      suite : ['asahi', 'g17p-wayland-surface-init-retry'],
      timeout : 30,
    )

    g17p_wayland_present_wrapper_env = environment(
      g17p_acquire_rollback_environment)
    g17p_wayland_present_wrapper_env.set('MESA_VK_WSI_DEBUG', 'sw')
    g17p_wayland_present_wrapper_env.set(
      'ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK', 'wayland-present-wrapper-retry')
    test(
      'hk_g17p_wayland_present_wrapper_retry',
      g17p_wayland_swapchain_ownership,
      args : ['--present-wrapper-retry'],
      env : g17p_wayland_present_wrapper_env,
      depends : _dev_icd,
      suite : ['asahi', 'g17p-wayland-present-wrapper-retry'],
      timeout : 30,
    )

    g17p_wayland_present_queue_env = environment(
      g17p_acquire_rollback_environment)
    g17p_wayland_present_queue_env.set('MESA_VK_WSI_DEBUG', 'sw')
    g17p_wayland_present_queue_env.set(
      'ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK', 'wayland-present-queue-retry')
    test(
      'hk_g17p_wayland_present_queue_retry',
      g17p_wayland_swapchain_ownership,
      args : ['--present-queue-retry'],
      env : g17p_wayland_present_queue_env,
      depends : _dev_icd,
      suite : ['asahi', 'g17p-wayland-present-queue-retry'],
      timeout : 30,
    )

    g17p_wayland_presentation_wrapper_env = environment(
      g17p_acquire_rollback_environment)
    g17p_wayland_presentation_wrapper_env.set('MESA_VK_WSI_DEBUG', 'sw')
    g17p_wayland_presentation_wrapper_env.set(
      'ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK',
      'wayland-presentation-wrapper-retry')
    test(
      'hk_g17p_wayland_presentation_wrapper_retry',
      g17p_wayland_swapchain_ownership,
      args : ['--presentation-wrapper-retry'],
      env : g17p_wayland_presentation_wrapper_env,
      depends : _dev_icd,
      suite : ['asahi', 'g17p-wayland-presentation-wrapper-retry'],
      timeout : 30,
    )
  endif

  g17p_layered_mip_blit_environment = {
    'ASAHI_SHIM_CHIP' : 'g17p',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM' : '1',
    'LD_PRELOAD' : libasahi_noop_drm_shim.full_path(),
    'MESA_SHADER_CACHE_DISABLE' : 'true',
    'VK_DRIVER_FILES' : _dev_icd.full_path(),
  }
  g17p_layered_mip_blit_env = environment(
    g17p_layered_mip_blit_environment)
  g17p_layered_mip_blit_env.set(
    'ASAHI_SHIM_G17P_EXPECT_FRAG_CONSTANT_DRAWS', '18')
  test(
    'hk_g17p_layered_mip_blit',
    g17p_vulkan_layered_mip_blit,
    args : ['--submit-only'],
    env : g17p_layered_mip_blit_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-layered-mip-blit'],
    timeout : 30,
  )

  g17p_layered_mip_blit_fail_env = environment(
    g17p_layered_mip_blit_environment)
  g17p_layered_mip_blit_fail_env.set('ASAHI_SHIM_FAIL_SUBMIT_ONCE', '1')
  test(
    'hk_g17p_layered_mip_blit_submit_failure',
    g17p_vulkan_layered_mip_blit,
    args : ['--expect-submit-failure'],
    env : g17p_layered_mip_blit_fail_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-layered-mip-blit'],
    timeout : 30,
  )

  g17p_conditional_environment = {
    'ASAHI_SHIM_CHIP' : 'g17p',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM' : '1',
    'ASAHI_SHIM_G17P_EXPECT_COMPUTE' : '1',
    'ASAHI_SHIM_G17P_EXPECT_STRONG_COMPUTE_BARRIER' : '1',
    # draw_indirect_count was retired when V12c qualified on hardware and
    # drawIndirectCount became unconditional; no code reads that token now.
    'HK_G17P_UNVERIFIED_FEATURES' : 'conditional_rendering',
    'LD_PRELOAD' : libasahi_noop_drm_shim.full_path(),
    'MESA_SHADER_CACHE_DISABLE' : 'true',
    'VK_DRIVER_FILES' : _dev_icd.full_path(),
  }
  g17p_conditional_env = environment(g17p_conditional_environment)

  foreach case : [
    'direct-zero',
    'direct-nonzero',
    'inverted-zero',
    'inverted-nonzero',
    'indirect-zero',
    'indirect-nonzero',
    'inherited-zero',
    'inherited-nonzero',
  ]
    test(
      'hk_g17p_conditional_compute_' + case,
      g17p_conditional_compute,
      args : ['--submit-only', case],
      env : g17p_conditional_env,
      suite : ['asahi', 'g17p-conditional-rendering'],
      timeout : 30,
    )
  endforeach

  g17p_conditional_truncated_env = environment(g17p_conditional_environment)
  g17p_conditional_truncated_env.set(
    'ASAHI_SHIM_G17P_TRUNCATE_COMPUTE_END', '1')
  test(
    'hk_g17p_conditional_compute_truncated_stream',
    g17p_conditional_compute,
    args : ['--submit-only', 'direct-zero'],
    env : g17p_conditional_truncated_env,
    suite : ['asahi', 'g17p-conditional-rendering'],
    should_fail : true,
    timeout : 30,
  )

  foreach case : [
    'execute',
    'discard',
    'indexed-execute',
    'indexed-discard',
    'indirect-execute',
    'indirect-discard',
    'count-execute',
    'count-discard',
    'clear-execute',
    'clear-discard',
  ]
    test(
      'hk_g17p_conditional_graphics_' + case,
      g17p_conditional_graphics,
      args : ['--submit-only', case],
      env : g17p_conditional_env,
      suite : ['asahi', 'g17p-conditional-rendering'],
      timeout : 30,
    )
  endforeach
endif
