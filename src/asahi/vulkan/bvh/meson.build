# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT

hk_bvh_include_dir = dir_source_root + '/src/asahi/vulkan/bvh'

hk_bvh_includes = files(
  'agx3_node.h',
  'hk_build_interface.h',
  '../hk_bvh.h',
)

# Reassigning bvh_spv follows the same pattern turnip uses: src/vulkan is
# subdir'd before src/asahi, so the common bvh_spv has already been consumed
# by then and vk_bvh_include_dir / vk_glsl_shader_preamble are in scope.
hk_bvh_spv = []
foreach s : [['encode.comp', 'hk_encode'], ['write.comp', 'hk_bvh_write'],
             ['agx3_leaf.comp', 'hk_agx3_leaf'],
             ['agx3_leaf.comp', 'hk_agx3_leaf_pairs', '-DHK_AGX3_LEAF_PAIRS=1'],
             ['agx3_leaf.comp', 'hk_agx3_refit_encode', '-DHK_AGX3_REFIT=1'],
             ['agx3_envelope.comp', 'hk_agx3_envelope'],
             ['agx3_plan.comp', 'hk_agx3_plan'],
             ['agx3_plan.comp', 'hk_agx3_plan_pairs', '-DHK_AGX3_PLAN_PAIRS=1'],
             ['agx3_refit.comp', 'hk_agx3_refit'],
             ['agx3_refit_leaf.comp', 'hk_agx3_refit_leaf'],
             ['vertex_table.comp', 'hk_vertex_table']]
  command = [
    prog_glslang, '-V',
    '-I' + vk_bvh_include_dir,
    '-I' + hk_bvh_include_dir,
    '-I' + dir_source_root + '/src/asahi/vulkan',
    '--target-env', 'spirv1.5', '-x', '-o', '@OUTPUT@', '@INPUT@',
  ]
  command += s.length() > 2 ? [s[2]] : []
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
