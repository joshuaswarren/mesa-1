# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT

gl_resources = executable(
  'asahi-g17p-gl-resources',
  files('test-g17p-gl-resources.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
    dep_gbm,
  ],
  build_by_default : false,
)

gl_state_matrix = executable(
  'asahi-g17p-gl-state-matrix',
  files('test-g17p-gl-state-matrix.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-state-matrix-software-control',
  gl_state_matrix,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

run_target(
  'asahi-g17p-gl-state-matrix-shim',
  command : [
    prog_python,
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_state_matrix,
  ],
  env : [
    'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
    'ASAHI_SHIM_G17P_EXPECT_MATRIX=1',
  ],
)

test(
  'asahi-g17p-gl-resources-software-control',
  gl_resources,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

run_target(
  'asahi-g17p-gl-resources-shim',
  command : [
    prog_python,
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_resources,
  ],
)

gl_buffer_sync = executable(
  'asahi-g17p-gl-buffer-sync',
  files('test-g17p-gl-buffer-sync.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-buffer-sync-software-control',
  gl_buffer_sync,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

run_target(
  'asahi-g17p-gl-buffer-sync-shim',
  command : [
    prog_python,
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_buffer_sync,
  ],
)

gbm_share = executable(
  'asahi-g17p-gbm-share',
  files('test-g17p-gbm-share.c'),
  dependencies : [dep_gbm, dep_libdrm],
  build_by_default : false,
)

run_target(
  'asahi-g17p-gbm-share-shim',
  command : [
    prog_python,
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gbm_share,
  ],
)

egl_dmabuf_image = executable(
  'asahi-g17p-egl-dmabuf-image',
  files('test-g17p-egl-dmabuf-image.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
    dep_gbm,
    dep_libdrm,
  ],
  build_by_default : false,
)

run_target(
  'asahi-g17p-egl-dmabuf-image-shim',
  command : [
    prog_python,
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    egl_dmabuf_image,
  ],
)
