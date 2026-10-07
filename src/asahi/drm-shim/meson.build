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
  test(
    'hk_g17p_depth_feedback_query',
    g17p_depth_feedback_query,
    args : ['compressed'],
    env : {
      'ASAHI_SHIM_CHIP' : 'g17p',
      'LD_PRELOAD' : libasahi_noop_drm_shim.full_path(),
      'VK_DRIVER_FILES' : _dev_icd.full_path(),
      'MESA_SHADER_CACHE_DISABLE' : 'true',
    },
    depends : [_dev_icd, libasahi_noop_drm_shim, libvulkan_asahi],
    suite : ['asahi'],
  )

  foreach sparse_case : [['g17p', 'no-sparse'], ['g13g', 'sparse']]
    test(
      'hk_sparse_format_properties_' + sparse_case[0],
      hk_sparse_format_properties,
      args : [sparse_case[1]],
      env : {
        'ASAHI_SHIM_CHIP' : sparse_case[0],
        'LD_PRELOAD' : libasahi_noop_drm_shim.full_path(),
        'VK_DRIVER_FILES' : _dev_icd.full_path(),
        'MESA_VK_VERSION_OVERRIDE' : '1.0',
      },
      depends : [_dev_icd, libasahi_noop_drm_shim, libvulkan_asahi],
      suite : ['asahi'],
    )
  endforeach

  g17p_queue_progress_environment = {
    'ASAHI_SHIM_CHIP' : 'g17p',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM' : '1',
    'LD_PRELOAD' : libasahi_noop_drm_shim.full_path(),
    'MESA_SHADER_CACHE_DISABLE' : 'true',
    'VK_DRIVER_FILES' : _dev_icd.full_path(),
  }

  test(
    'hk_g17p_attachmentless_record',
    g17p_attachmentless_record,
    env : g17p_queue_progress_environment,
    depends : [_dev_icd, libvulkan_asahi, libasahi_noop_drm_shim],
    suite : ['asahi', 'g17p-attachmentless'],
    timeout : 30,
  )

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

  # One command buffer holding a frame's worth of render passes and draws.
  # Each Ioto program used to take a BO of its own -- a 16KiB page plus a
  # 16KiB guard page of the 4GiB USC exec window for a couple of hundred
  # bytes of program -- and this recording alone establishes thousands of
  # executable-window mappings, which is what ran the window out under
  # SuperTuxKart. Sub-allocated out of arena chunks the same recording needs
  # a handful. The budget is what fails if programs go back to one BO each;
  # there is no submit, so no stream-validator expectation is set.
  g17p_ioto_pressure_env = environment({
    'ASAHI_SHIM_CHIP' : 'g17p',
    'ASAHI_SHIM_G17P_MAX_USC_BINDS' : '64',
    'LD_PRELOAD' : libasahi_noop_drm_shim.full_path(),
    'MESA_SHADER_CACHE_DISABLE' : 'true',
    'VK_DRIVER_FILES' : _dev_icd.full_path(),
  })
  test(
    'hk_g17p_ioto_program_pressure',
    g17p_ioto_program_pressure,
    env : g17p_ioto_pressure_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-ioto-program-pressure'],
    timeout : 120,
  )

  # STK/GTK-style full BGRA8 upload and mip generation. The base upload,
  # two-level mip chain and three readbacks must be one compute stream; only
  # the final sample remains graphics. Regressing either fast path adds one or
  # more firmware render jobs, while regressing mip ordering removes the two
  # strong compute launches. The other two strong records are the fixture's
  # two transfer pipeline barriers: since 89500458996 a barrier whose source
  # scope is compute/transfer upgrades the open stream's launch tail to the
  # strong flush in place instead of ending the stream.
  g17p_generated_mips_env = environment(g17p_queue_progress_environment)
  g17p_generated_mips_env.set('G17P_GENERATED_MIPS_SUBMIT_ONLY', '1')
  g17p_generated_mips_env.set('ASAHI_SHIM_G17P_EXPECT_COMPUTE', '1')
  g17p_generated_mips_env.set('ASAHI_SHIM_G17P_EXPECT_COMPUTE_COUNTS',
                              '7,7,0,4')
  # The raw copy and both mip filters make every fixture compute launch 2D.
  # Flattening the copy drops this to six; flattening the mip filters produces
  # four here and five in the odd fixture.
  g17p_generated_mips_env.set('ASAHI_SHIM_G17P_EXPECT_2D_COMPUTE_LAUNCHES',
                              '7')
  g17p_generated_mips_env.set('ASAHI_SHIM_G17P_EXPECT_STREAM_SHAPE',
                              '1:17:17:1:0')
  test(
    'hk_g17p_generated_mips_compute',
    g17p_generated_mips,
    env : g17p_generated_mips_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-generated-mips'],
    timeout : 30,
  )

  # A 129x129 base level exercises the Q8 coordinate-table kernel for the
  # first blit while retaining the same merged compute-stream contract.
  test(
    'hk_g17p_generated_mips_odd_compute',
    g17p_generated_mips_odd,
    env : g17p_generated_mips_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-generated-mips'],
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

  g17p_image_copy_3d_environment = {
    'ASAHI_SHIM_CHIP' : 'g17p',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM' : '1',
    'LD_PRELOAD' : libasahi_noop_drm_shim.full_path(),
    'MESA_SHADER_CACHE_DISABLE' : 'true',
    'VK_DRIVER_FILES' : _dev_icd.full_path(),
  }
  g17p_image_copy_3d_env = environment(g17p_image_copy_3d_environment)
  # The uncompressed uploads and compatible image copies use raw CDM slices,
  # so this regression should not need the fragment path.
  test(
    'hk_g17p_image_copy_3d',
    g17p_vulkan_image_copy_3d,
    args : ['--submit-only'],
    env : g17p_image_copy_3d_env,
    depends : _dev_icd,
    suite : ['asahi', 'g17p-image-copy-3d'],
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

  # Performance invariant: 128 draws in one render pass must remain in one
  # stream with NO opt-in batching flag. Alternating all four direct/indirect,
  # indexed/nonindexed forms also checks body framing after the first draw.
  foreach mode : ['direct', 'indirect', 'mixed-indexed', 'stencil', 'mixed-stencil']
    batching_env = environment(g17p_image_copy_3d_environment)
    batching_env.set('G17P_REPEAT_DRAW_COUNT', '128')
    batching_env.set('ASAHI_SHIM_G17P_EXPECT_STREAM_SHAPE',
                     mode in ['stencil', 'mixed-stencil'] ? '1:128:128:1' : '1:128:1:1')
    if mode == 'indirect'
      batching_env.set('G17P_REPEAT_INDIRECT', '1')
    elif mode in ['mixed-indexed', 'mixed-stencil']
      batching_env.set('G17P_REPEAT_INDIRECT', '2')
    endif
    if mode in ['stencil', 'mixed-stencil']
      batching_env.set('G17P_REPEAT_STENCIL_REFERENCE', '1')
    endif
    test('hk_g17p_batching_' + mode, g17p_repeat_draw,
         env : batching_env, depends : [_dev_icd, libasahi_noop_drm_shim],
         suite : ['asahi', 'g17p-batching'], timeout : 30)
  endforeach

  # Minimal reproduction for the native absolute G17P indirect-argument
  # pointer: one direct draw followed by one non-indexed indirect draw in the
  # same stream. The strict validator must check the encoded VA directly;
  # rebasing it as a compact PPP/stream-link payload rejects the owned BO.
  indirect_address_env = environment(g17p_image_copy_3d_environment)
  indirect_address_env.set('G17P_REPEAT_DRAW_COUNT', '2')
  indirect_address_env.set('G17P_REPEAT_INDIRECT', '2')
  indirect_address_env.set('G17P_REPEAT_STENCIL_REFERENCE', '1')
  indirect_address_env.set('ASAHI_SHIM_G17P_EXPECT_STREAM_SHAPE', '1:2:2:1')
  test('hk_g17p_indirect_argument_address', g17p_repeat_draw,
       env : indirect_address_env,
       depends : [_dev_icd, libasahi_noop_drm_shim],
       suite : ['asahi', 'g17p-batching', 'g17p-indirect-address'],
       timeout : 30)

  # Cross the real VDM-root capacity with all four draw forms and fresh native
  # state on every draw. The 32 KiB root and following 64 KiB chunks carry all
  # 4096 draws in one render through nine native links. A root-capacity
  # boundary must not become one of the retained renders whose firmware job
  # list is bounded independently.
  batching_capacity_env = environment(g17p_image_copy_3d_environment)
  batching_capacity_env.set('G17P_REPEAT_DRAW_COUNT', '4096')
  batching_capacity_env.set('G17P_REPEAT_INDIRECT', '2')
  batching_capacity_env.set('G17P_REPEAT_STENCIL_REFERENCE', '1')
  batching_capacity_env.set('ASAHI_SHIM_G17P_EXPECT_STREAM_SHAPE', '1:4096:4096:1:9')
  test('hk_g17p_batching_capacity', g17p_repeat_draw,
       env : batching_capacity_env, depends : [_dev_icd, libasahi_noop_drm_shim],
       suite : ['asahi', 'g17p-batching'], timeout : 30)

  batching_bias_env = environment(g17p_image_copy_3d_environment)
  batching_bias_env.set('G17P_REPEAT_DRAW_COUNT', '4096')
  batching_bias_env.set('G17P_REPEAT_INDIRECT', '2')
  batching_bias_env.set('G17P_REPEAT_STENCIL_REFERENCE', '1')
  batching_bias_env.set('G17P_REPEAT_INTEGER_DEPTH_BIAS', '1')
  batching_bias_env.set('ASAHI_SHIM_G17P_EXPECT_INTEGER_DEPTH_BIAS', '1')
  batching_bias_env.set('ASAHI_SHIM_G17P_EXPECT_STREAM_SHAPE', '1:4096:4096:1:9')
  test('hk_g17p_batching_capacity_integer_bias', g17p_repeat_draw,
       env : batching_bias_env, depends : [_dev_icd, libasahi_noop_drm_shim],
       suite : ['asahi', 'g17p-batching'], timeout : 30)

  batching_repeat_env = environment(g17p_image_copy_3d_environment)
  batching_repeat_env.set('G17P_REPEAT_DRAW_COUNT', '128')
  batching_repeat_env.set('G17P_REPEAT_INDIRECT', '2')
  batching_repeat_env.set('G17P_REPEAT_SUBMITS', '32')
  batching_repeat_env.set('ASAHI_SHIM_G17P_EXPECT_STREAM_SHAPE', '32:4096:32:32')
  test('hk_g17p_batching_retained_submits', g17p_repeat_draw,
       env : batching_repeat_env, depends : [_dev_icd, libasahi_noop_drm_shim],
       suite : ['asahi', 'g17p-batching'], timeout : 30)

  batching_scissor_env = environment(g17p_image_copy_3d_environment)
  batching_scissor_env.set('ASAHI_SHIM_G17P_EXPECT_STREAM_SHAPE', '1:4:4:1')
  test('hk_g17p_batching_scissor_ring', g17p_scissor_ring,
       args : ['ring', '--record-only'], env : batching_scissor_env,
       depends : [_dev_icd, libasahi_noop_drm_shim],
       suite : ['asahi', 'g17p-batching'], timeout : 30)

  # Whether a readback copy is ordered after the render that produced it is
  # carried entirely by drm_asahi_cmd_header::vdm_barrier and by which ioctl
  # the out-syncobjs ride on. Both are write-only from userspace, so a driver
  # that asks for the wrong thing and a kernel that ignores the right thing
  # look identical from a stale readback. Record the submissions and assert
  # the request. The script carries its own negative control.
  test(
    'hk_g17p_copy_after_render_order',
    prog_python,
    args : [
      files('../vulkan/tests/test-g17p-copy-after-render-order.py'),
      meson.current_build_dir() / '../../..',
    ],
    depends : [_dev_icd, libasahi_noop_drm_shim, g17p_offscreen_triangle],
    suite : ['asahi', 'g17p-copy-order'],
    timeout : 60,
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
  # The twelve cube-layer base uploads are raw CDM copies; the six layered
  # blits remain the general graphics path and the readback adds one CDM
  # launch.
  g17p_layered_mip_blit_env.set(
    'ASAHI_SHIM_G17P_EXPECT_FRAG_CONSTANT_DRAWS', '6')
  g17p_layered_mip_blit_env.set('ASAHI_SHIM_G17P_EXPECT_COMPUTE', '1')
  g17p_layered_mip_blit_env.set(
    # One strong barrier is the host-read cache flush (6aae441547b), which
    # strengthens the final launch's tail record in place. The other is the
    # upload-to-blit pipeline barrier: since 89500458996 a barrier whose
    # source scope is compute/transfer strengthens the last upload's tail in
    # place instead of ending the stream.
    'ASAHI_SHIM_G17P_EXPECT_COMPUTE_COUNTS', '13,13,0,2')
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

endif
