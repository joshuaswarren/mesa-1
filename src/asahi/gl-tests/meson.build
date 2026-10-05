# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT

# Hardware stress reproducer; the GL software path cannot exercise the
# intermittent G17P failure. Invoke explicitly with Zink -> Honeykrisp.
executable(
  'asahi-zink-discard-corruption',
  files('test-zink-discard-corruption.c'),
  dependencies : [cc.find_library('EGL'), cc.find_library('GLESv2')],
  build_by_default : false,
)

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

gl_depth_texture = executable(
  'asahi-g17p-gl-depth-texture',
  files('test-g17p-gl-depth-texture.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

gl_texture_offset = executable(
  'asahi-g17p-gl-texture-offset',
  files('test-g17p-gl-texture-offset.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

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
  'asahi-g17p-gl-mrt-software-control',
  gl_mrt,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
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

test(
  'asahi-g17p-gl-wboit-software-control',
  gl_mrt,
  args : ['--wboit'],
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-wboit-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_mrt,
    '--wboit',
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

gl_msaa = executable(
  'asahi-g17p-gl-msaa',
  files('test-g17p-gl-msaa.c'),
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
  'asahi-g17p-gl-msaa-software-control',
  gl_msaa,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-msaa-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_msaa,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_SHIM_G17P_EXPECT_MSAA_DRAWS=12',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-texture-offset-software-control',
  gl_texture_offset,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-texture-offset-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_texture_offset,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-depth-texture-software-control',
  gl_depth_texture,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

run_target(
  'asahi-g17p-gl-depth-texture-shim',
  command : [
    prog_python,
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_depth_texture,
  ],
  env : ['ASAHI_SHIM_G17P_EXPECT_DEPTH_TEXTURE=1'],
)

test(
  'asahi-g17p-gl-depth-texture-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_depth_texture,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_SHIM_G17P_EXPECT_DEPTH_TEXTURE=1',
  ],
  suite : ['asahi'],
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

gl_hypr_mixed_draw = executable(
  'f48-hypr-mixed-draw',
  files('test-g17p-gl-hypr-mixed-draw.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

foreach mode : ['tight', 'api-barrier']
  test(
    'asahi-g17p-gl-hypr-mixed-draw-software-control-' + mode,
    gl_hypr_mixed_draw,
    args : [
      files('hypr-0.56.1-tex300.vert'),
      files('hypr-0.56.1-border-rounded.frag'),
      '--' + mode,
    ],
    env : [
      'EGL_PLATFORM=surfaceless',
      'LIBGL_ALWAYS_SOFTWARE=true',
    ],
    suite : ['asahi'],
  )

  test(
    'asahi-g17p-gl-hypr-mixed-draw-shim-submit-' + mode,
    prog_python,
    args : [
      files('run-gl-shim.py'),
      libasahi_noop_drm_shim,
      libegl,
      libgallium_dri,
      gl_hypr_mixed_draw,
      files('hypr-0.56.1-tex300.vert'),
      files('hypr-0.56.1-border-rounded.frag'),
      '--' + mode,
    ],
    env : [
      'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
      'MESA_SHADER_CACHE_DISABLE=true',
    ],
    suite : ['asahi'],
  )
endforeach

gl_multi_viewport = executable(
  'asahi-g17p-gl-multi-viewport',
  files('test-g17p-gl-multi-viewport.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
  ],
  build_by_default : false,
)

gl_rgb32_fbo = executable(
  'asahi-g17p-gl-rgb32-fbo',
  files('test-g17p-gl-rgb32-fbo.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
  ],
  build_by_default : false,
)

gl_stencil_blit = executable(
  'asahi-g17p-gl-stencil-blit',
  files('test-g17p-gl-stencil-blit.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-stencil-blit-software-control',
  gl_stencil_blit,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

gl_default_framebuffer = executable(
  'asahi-g17p-gl-default-framebuffer',
  files('test-g17p-gl-default-framebuffer.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
    cc.find_library('X11'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-default-framebuffer-software-control',
  gl_default_framebuffer,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

gl_array_shadow = executable(
  'asahi-g17p-gl-array-shadow',
  files('test-g17p-gl-array-shadow.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
    cc.find_library('m'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-array-shadow-software-control',
  gl_array_shadow,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

gl_lod_bias = executable(
  'asahi-g17p-gl-lod-bias',
  files('test-g17p-gl-lod-bias.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
    cc.find_library('m'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-lod-bias-software-control',
  gl_lod_bias,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
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

gl_xfb_interleaved = executable(
  'asahi-g17p-gl-xfb-interleaved',
  files('test-g17p-gl-xfb-interleaved.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
    cc.find_library('m'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-xfb-interleaved-software-control',
  gl_xfb_interleaved,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

gl_xfb_streams = executable(
  'asahi-g17p-gl-xfb-streams',
  files('test-g17p-gl-xfb-streams.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
    cc.find_library('m'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-xfb-streams-software-control',
  gl_xfb_streams,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

gl_xfb_geometry = executable(
  'asahi-g17p-gl-xfb-geometry',
  files('test-g17p-gl-xfb-geometry.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-xfb-geometry-software-control',
  gl_xfb_geometry,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

gl_xfb_overflow_query = executable(
  'asahi-g17p-gl-xfb-overflow-query',
  files('test-g17p-gl-xfb-overflow-query.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-xfb-overflow-query-software-control',
  gl_xfb_overflow_query,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

gl_vertex_flat_varying = executable(
  'asahi-g17p-gl-vertex-flat-varying',
  files('test-g17p-gl-vertex-flat-varying.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-vertex-flat-varying-software-control',
  gl_vertex_flat_varying,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g13g-gl-vertex-flat-varying-shim-control',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_vertex_flat_varying,
  ],
  env : [
    'ASAHI_SHIM_CHIP=g13g',
    'ASAHI_GL_EXPECT_RENDERER=G13G',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

gl_texture_swizzle = executable(
  'asahi-g17p-gl-texture-swizzle',
  files('test-g17p-gl-texture-swizzle.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GL'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-texture-swizzle-software-control',
  gl_texture_swizzle,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g13g-gl-xfb-geometry-shim-control',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_xfb_geometry,
  ],
  env : [
    'ASAHI_SHIM_CHIP=g13g',
    'ASAHI_GL_EXPECT_RENDERER=G13G',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g13g-gl-xfb-overflow-query-shim-control',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_xfb_overflow_query,
  ],
  env : [
    'ASAHI_SHIM_CHIP=g13g',
    'ASAHI_GL_EXPECT_RENDERER=G13G',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-xfb-overflow-query-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_xfb_overflow_query,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-xfb-geometry-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_xfb_geometry,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-xfb-streams-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_xfb_streams,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-multi-viewport-software-control',
  gl_multi_viewport,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

run_target(
  'asahi-g17p-gl-multi-viewport-shim',
  command : [
    prog_python,
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_multi_viewport,
  ],
  env : ['ASAHI_SHIM_G17P_EXPECT_MULTI_VIEWPORT=1'],
)

test(
  'asahi-g17p-gl-multi-viewport-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_multi_viewport,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_SHIM_G17P_EXPECT_MULTI_VIEWPORT=1',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-rgb32-fbo-software-control',
  gl_rgb32_fbo,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-rgb32-fbo-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_rgb32_fbo,
  ],
  env : [
    'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
    'ASAHI_SHIM_G17P_EXPECT_RGB32_FBO_DRAWS=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
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

egl_shared_context = executable(
  'asahi-g17p-egl-shared-context',
  files('test-g17p-egl-shared-context.c'),
  include_directories : inc_include,
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

gl_texture_array = executable(
  'asahi-g17p-gl-texture-array',
  files('test-g17p-gl-texture-array.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

gl_layered_fbo = executable(
  'asahi-g17p-gl-layered-fbo',
  files('test-g17p-gl-layered-fbo.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

gl_frag_constants = executable(
  'asahi-g17p-gl-frag-constants',
  files('test-g17p-gl-frag-constants.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

gl_packed16_fbo = executable(
  'asahi-g17p-gl-packed16-fbo',
  files('test-g17p-gl-packed16-fbo.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

gl_texture_3d = executable(
  'asahi-g17p-gl-texture-3d',
  files('test-g17p-gl-texture-3d.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

gl_texture_border = executable(
  'asahi-g17p-gl-texture-border',
  files('test-g17p-gl-texture-border.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

gl_vertex_texture = executable(
  'asahi-g17p-gl-vertex-texture',
  files('test-g17p-gl-vertex-texture.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

gl_texture_cube_lod = executable(
  'asahi-g17p-gl-texture-cube-lod',
  files('test-g17p-gl-texture-cube-lod.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-texture-cube-lod-software-control',
  gl_texture_cube_lod,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-texture-cube-lod-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_texture_cube_lod,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-texture-array-software-control',
  gl_texture_array,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-layered-fbo-software-control',
  gl_layered_fbo,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-frag-constants-software-control',
  gl_frag_constants,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-frag-constants-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_frag_constants,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
    'ASAHI_SHIM_G17P_EXPECT_FRAG_CONSTANT_DRAWS=7',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-packed16-fbo-software-control',
  gl_packed16_fbo,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-packed16-fbo-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_packed16_fbo,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
    'ASAHI_SHIM_G17P_EXPECT_16BIT_FBO_DRAWS=4',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-layered-fbo-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_layered_fbo,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-texture-array-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_texture_array,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-texture-3d-software-control',
  gl_texture_3d,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-texture-3d-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_texture_3d,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-texture-border-software-control',
  gl_texture_border,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-texture-border-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_texture_border,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
    'ASAHI_SHIM_G17P_EXPECT_FRAG_CONSTANT_DRAWS=3',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-vertex-texture-software-control',
  gl_vertex_texture,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-vertex-texture-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_vertex_texture,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

# The kernel refuses a render job whose VM has not bound every T8140 native
# context view (mmu.rs install_t8140_job_context_aliases), which is how a
# driver without the fixed 0x1000000000 render-context bind fails on Neo:
# "job context view 0 source 0x1000000000+0x10000 -> 0x0 is not bound".
# The shim models that registry, so this asserts the bind contract host-side.
test(
  'asahi-g17p-gl-vertex-texture-shim-context-views',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_vertex_texture,
  ],
  env : [
    'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
    'ASAHI_SHIM_G17P_EXPECT_CONTEXT_VIEWS=1',
    'ASAHI_SHIM_G17P_ENFORCE_CONTEXT_VIEWS=1',
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_GL_EXPECT_RENDERER=A18 Pro',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

# The same draw on a shader-cache HIT, which is the path every launch after the
# first one takes and which no other shim oracle exercises, because they all
# disable the cache. A compiled-shader property that survives compilation but
# not serialisation shows up only here.
test(
  'asahi-g17p-gl-vertex-texture-shim-warm-cache',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_vertex_texture,
  ],
  env : [
    'ASAHI_SHIM_VALIDATE_G17P_STREAM=1',
    'ASAHI_SHIM_G17P_EXPECT_CONTEXT_VIEWS=1',
    'ASAHI_SHIM_G17P_ENFORCE_CONTEXT_VIEWS=1',
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_GL_EXPECT_RENDERER=A18 Pro',
    'ASAHI_GL_SHIM_WARM_CACHE=1',
  ],
  suite : ['asahi'],
)

# The AGX2 control arm: the same client and the same shim, changing only the
# reported chip. A failure that reproduces here is not G17P-specific.
test(
  'asahi-g13g-gl-vertex-texture-shim-control',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_vertex_texture,
  ],
  env : [
    'ASAHI_SHIM_CHIP=g13g',
    'ASAHI_GL_EXPECT_RENDERER=G13G',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-egl-shared-context-software-control',
  egl_shared_context,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-egl-shared-context-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    egl_shared_context,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

gl_compute = executable(
  'asahi-g17p-gl-compute',
  files('test-g17p-gl-compute.c'),
  dependencies : [
    cc.find_library('EGL'),
    cc.find_library('GLESv2'),
  ],
  build_by_default : false,
)

test(
  'asahi-g17p-gl-compute-software-control',
  gl_compute,
  env : [
    'EGL_PLATFORM=surfaceless',
    'LIBGL_ALWAYS_SOFTWARE=true',
  ],
  suite : ['asahi'],
)

run_target(
  'asahi-g17p-gl-compute-shim',
  command : [
    prog_python,
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_compute,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_COMPUTE=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
)

test(
  'asahi-g17p-gl-compute-shim-submit',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_compute,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_VALID_STREAM=1',
    'ASAHI_SHIM_G17P_EXPECT_COMPUTE=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-gl-compute-rejects-truncated-terminate',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gl_compute,
  ],
  env : [
    'ASAHI_SHIM_G17P_EXPECT_COMPUTE=1',
    'ASAHI_SHIM_G17P_TRUNCATE_COMPUTE_END=1',
    'MESA_SHADER_CACHE_DISABLE=true',
  ],
  should_fail : true,
  suite : ['asahi'],
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

bo_export_race = executable(
  'asahi-g17p-bo-export-race',
  files('test-g17p-bo-export-race.c'),
  include_directories : [
    inc_include,
    inc_src,
    inc_asahi,
    inc_virtio_gpu,
    inc_virtio_vdrm,
  ],
  dependencies : [
    dep_libdrm,
    dep_thread,
    idep_libagx,
    idep_nir,
    idep_mesautil,
  ],
  link_with : [libasahi_lib, libasahi_compiler, libasahi_layout],
  build_by_default : false,
)

test(
  'asahi-g17p-bo-export-race',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    bo_export_race,
  ],
  env : [
    'ASAHI_SHIM_DELAY_PRIME_EXPORT=1',
    'ASAHI_SHIM_EXPECT_SINGLE_PRIME_EXPORT=1',
  ],
  suite : ['asahi'],
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

test(
  'asahi-g17p-gbm-prime-export-retry',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    gbm_share,
  ],
  env : [
    'ASAHI_SHIM_FAIL_PRIME_EXPORT_ONCE=1',
    'ASAHI_SHIM_EXPECT_PRIME_EXPORT_FAILURE=1',
    'ASAHI_GL_EXPECT_PRIME_EXPORT_RETRY=1',
  ],
  suite : ['asahi'],
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

if with_platform_wayland
  gl_buffer_age_xdg_shell = mod_wl.scan_xml(
    mod_wl.find_protocol('xdg-shell'),
    include_core_only : false,
  )
  egl_buffer_age = executable(
    'asahi-g17p-egl-buffer-age',
    files('test-g17p-egl-buffer-age.c'),
    gl_buffer_age_xdg_shell,
    dependencies : [
      cc.find_library('EGL'),
      cc.find_library('GLESv2'),
      dep_gbm,
      dep_wayland_client,
      dependency('wayland-egl'),
    ],
    build_by_default : false,
  )
endif

run_target(
  'asahi-g17p-egl-gbm-swap-shim',
  command : [
    prog_python,
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    egl_gbm_swap,
  ],
  env : ['ASAHI_SHIM_G17P_EXPECT_SWAP_LIFECYCLE=1'],
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

test(
  'asahi-g17p-egl-gbm-failed-submit-reset',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    egl_gbm_swap,
  ],
  env : [
    'ASAHI_SHIM_FAIL_SUBMIT_ONCE=1',
    'ASAHI_GL_EXPECT_SUBMIT_FAILURE=1',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-egl-gbm-implicit-sync-fallback',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    egl_gbm_swap,
  ],
  env : [
    'ASAHI_SHIM_EXPECT_DMABUF_SYNC_FAILURES=1',
    'ASAHI_GL_EXPECT_IMPLICIT_SYNC_FALLBACK=1',
  ],
  suite : ['asahi'],
)

test(
  'asahi-g17p-egl-gbm-server-wait-accumulate',
  prog_python,
  args : [
    files('run-gl-shim.py'),
    libasahi_noop_drm_shim,
    libegl,
    libgallium_dri,
    egl_gbm_swap,
  ],
  env : [
    'ASAHI_GL_EXPECT_SERVER_WAIT_ACCUMULATE=1',
    'ASAHI_SHIM_EXPECT_SYNC_WAIT_TIMEOUT=1',
  ],
  suite : ['asahi'],
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
