# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT

gl_resources = executable(
  'asahi-g17p-gl-resources',
  files('test-g17p-gl-resources.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
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
