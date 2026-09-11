# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT

hk_bvh_include_dir = dir_source_root + '/src/asahi/vulkan/bvh'

hk_bvh_includes = files(
  'hk_build_interface.h',
  '../hk_bvh.h',
)

# Reassigning bvh_spv follows the same pattern turnip uses: src/vulkan is
# subdir'd before src/asahi, so the common bvh_spv has already been consumed
# by then and vk_bvh_include_dir / vk_glsl_shader_preamble are in scope.
hk_bvh_spv = []
foreach s : [['encode.comp', 'hk_encode'], ['write.comp', 'hk_bvh_write']]
  command = [
    prog_glslang, '-V',
    '-I' + vk_bvh_include_dir,
    '-I' + hk_bvh_include_dir,
    '-I' + dir_source_root + '/src/asahi/vulkan',
    '--target-env', 'spirv1.5', '-x', '-o', '@OUTPUT@', '@INPUT@',
  ]
  command += glslang_quiet
  command += vk_glsl_shader_preamble

  hk_bvh_spv += custom_target(
    s[1] + '.spv.h',
    input : s[0],
    output : s[1] + '.spv.h',
    command : command,
    depend_files : [vk_bvh_includes, hk_bvh_includes],
  )
endforeach
