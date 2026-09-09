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
