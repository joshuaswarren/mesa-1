# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT

# Smoke tests for what loads and submits on G17P under the no-op DRM shim:
# libgallium and EGL/GLES initialisation, one multi-target draw and its
# readback through the asahi Gallium driver and through Zink on the Vulkan
# driver, and EGL/GBM swaps. The shim validates every G17P command stream the
# drivers submit; pixel values are only compared on hardware.

gl_mrt = executable(
  'asahi-g17p-gl-mrt',
  files('test-g17p-gl-mrt.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-mrt-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_mrt,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

if with_gallium_zink and with_asahi_vk
  test(
    'zink-g17p-gl-mrt-shim-submit',
    prog_python,
    args : [
      files('run-gl-shim.py'),
      libasahi_noop_drm_shim,
      libegl,
      libgallium_dri,
      gl_mrt,
    ],
    env : [
      'ASAHI_GL_SHIM_VULKAN=' + libvulkan_asahi.full_path(),
      'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
      'MESA_SHADER_CACHE_DISABLE=true',
    ],
    depends : [libvulkan_asahi],
    suite : ['asahi'],
  )
endif

egl_gbm_swap = executable(
  'asahi-g17p-egl-gbm-swap',
  files('test-g17p-egl-gbm-swap.c'),
  include_directories : inc_include,
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
    dep_gbm,
    dep_libdrm,
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-egl-gbm-swap-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    egl_gbm_swap,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_SHIM_G17P_EXPECT_SWAP_LIFECYCLE=1',
  ],
  suite : ['asahi'],
)

gl_msaa_resolve = executable(
  'asahi-g17p-gl-msaa-resolve',
  files('test-g17p-gl-msaa-resolve.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
    cc.find_library('m'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-msaa-resolve-software-control',
  gl_msaa_resolve,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-msaa-sparse-depth-software-control',
  gl_msaa_resolve,
  args : ['--sparse-depth'],
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)
