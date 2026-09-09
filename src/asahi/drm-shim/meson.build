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

  g17p_conditional_environment = {
    'ASAHI_SHIM_CHIP' : 'g17p',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM' : '1',
    'ASAHI_SHIM_G17P_EXPECT_COMPUTE' : '1',
    'ASAHI_SHIM_G17P_EXPECT_STRONG_COMPUTE_BARRIER' : '1',
    'HK_G17P_UNVERIFIED_FEATURES' :
      'conditional_rendering,draw_indirect_count',
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
