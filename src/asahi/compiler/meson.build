# Copyright 2018 Rob Clark
# Copyright 2019 Collabora, Ltd.
# SPDX-License-Identifier: MIT

libasahi_agx_files = files(
  'agx_compile.c',
  'agx_dce.c',
  'agx_liveness.c',
  'agx_insert_waits.c',
  'agx_nir_lower_address.c',
  'agx_nir_lower_cull_distance.c',
  'agx_nir_lower_fminmax.c',
  'agx_nir_lower_frag_sidefx.c',
  'agx_nir_lower_sample_mask.c',
  'agx_nir_lower_discard_zs_emit.c',
  'agx_nir_lower_interpolation.c',
  'agx_nir_lower_g17p_combined_sampler.c',
  'agx_nir_ray_query.c',
  'agx_nir_lower_shared_bitsize.c',
  'agx_nir_lower_subgroups.c',
  'agx_nir_lower_texture.c',
  'agx_nir_opt_preamble.c',
  'agx_lower_64bit.c',
  'agx_lower_divergent_shuffle.c',
  'agx_lower_parallel_copy.c',
  'agx_lower_pseudo.c',
  'agx_lower_spill.c',
  'agx_lower_uniform_sources.c',
  'agx_pack.c',
  'agx3_pack.c',
  'agx3_select_core.c',
  'agx3_copy_core.c',
  'agx3_isel_core.c',
  'agx3_iadd_core.c',
  'agx3_pack_binary.c',
  'agx3_compose/agx3_compose.c',
  'agx3_compose/agx3b_tables.c',
  'agx3_compose/agx3c_tables.c',
  'agx3_binder.c',
  'agx3_atomic_core.c',
  'agx3_cf_core.c',
  'agx3_mem_core.c',
  'agx3_local_tps_core.c',
  'agx3_scs_core.c',
  'agx3_subgroup_core.c',
  'agx3_static_spill_core.c',
  'agx3_ioto_load_packer.c',
  'agx3_ioto_load_packer.h',
  'agx3_ioto_uniform_program.c',
  'agx3_ioto_uniform_program.h',
  'agx3_tex_core.c',
  'agx3_falu_mod_core.c',
  'agx3_vary_core.c',
  'agx3_tile_core.c',
  'agx3_pixel_core.c',
  'agx3_stage_core.c',
  'agx3_fltnorm_core.c',
  'agx3_writeout_core.c',
  'agx3_zs_core.c',
  'agx3_rt_core.c',
  'agx3_slot_core.c',
  'agx3_alu_core.c',
  'agx3_funop_core.c',
  'agx3_sr_core.c',
  'agx_performance.c',
  'agx_print.c',
  'agx_pressure_schedule.c',
  'agx_ir.c',
  'agx_opt_cse.c',
  'agx_opt_break_if.c',
  'agx_opt_empty_else.c',
  'agx_opt_jmp_none.c',
  'agx_opt_compact_constants.c',
  'agx_opt_promote_constants.c',
  'agx_opt_register_cache.c',
  'agx_optimizer.c',
  'agx_repair_ssa.c',
  'agx_reindex_ssa.c',
  'agx_spill.c',
  'agx_register_allocate.c',
  'agx_validate.c',
  'agx_validate_ra.c',
)

agx_nir_algebraic_c = custom_target(
  input : 'agx_nir_algebraic.py',
  output : 'agx_nir_algebraic.c',
  command : [
    prog_python, '@INPUT@', '-p', dir_compiler_nir,
  ],
  capture : true,
  depend_files : nir_algebraic_depends,
)

agx_opcodes_h = custom_target(
  input : ['agx_opcodes.h.py'],
  output : 'agx_opcodes.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('agx_opcodes.py'),
)

idep_agx_opcodes_h = declare_dependency(
  sources : [agx_opcodes_h],
  include_directories : include_directories('.'),
)

agx_opcodes_c = custom_target(
  input : ['agx_opcodes.c.py'],
  output : 'agx_opcodes.c',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('agx_opcodes.py'),
)

agx_builder_h = custom_target(
  input : ['agx_builder.h.py'],
  output : 'agx_builder.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : files('agx_opcodes.py'),
)

idep_agx_builder_h = declare_dependency(
  sources : [agx_builder_h],
  include_directories : include_directories('.'),
)

libasahi_compiler = static_library(
  'asahi_compiler',
  [libasahi_agx_files, agx_opcodes_c, agx_nir_algebraic_c],
  dependencies: [idep_nir, idep_agx_opcodes_h, idep_agx_builder_h, idep_mesautil, idep_libagx],
  link_with : [libagx2_disasm],
  c_args : [no_override_init_args, '-Wno-c2x-extensions'],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

if with_tests
  test(
    'agx3_subgroup_core',
    executable(
      'agx3_subgroup_core_test',
      files(
        'agx3_subgroup_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-subgroup-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_atomic_core',
    executable(
      'agx3_atomic_core_test',
      files(
        'agx3_atomic_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-atomic-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_pack',
    executable(
      'agx3_pack_test',
      files('test/test-agx3-pack.c', 'agx3_pack.c'),
      c_args : [c_msvc_compat_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_opcode_mapping',
    prog_python,
    args : files('test/test-agx3-opcodes.py'),
    suite : ['asahi'],
  )

  test(
    'agx3_tex_core',
    executable(
      'agx3_tex_core_test',
      files('test/test-agx3-tex-core.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      dependencies : [idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_local_tps_core',
    executable(
      'agx3_local_tps_core_test',
      files(
        'agx3_local_tps_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-local-tps-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_static_spill_core',
    executable(
      'agx3_static_spill_core_test',
      files(
        'agx3_static_spill_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-static-spill-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_falu_mod_core',
    executable(
      'agx3_falu_mod_core_test',
      files(
        'agx3_falu_mod_core.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_binder.c',
        'test/test-agx3-falu-mod-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_vary_core',
    executable(
      'agx3_vary_core_test',
      files(
        'agx3_vary_core.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_binder.c',
        'test/test-agx3-vary-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_cf_core',
    executable(
      'agx3_cf_core_test',
      files(
        'agx3_cf_core.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_binder.c',
        'test/test-agx3-cf-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_mem_core',
    executable(
      'agx3_mem_core_test',
      files(
        'agx3_mem_core.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_binder.c',
        'test/test-agx3-mem-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_alu_core',
    executable(
      'agx3_alu_core_test',
      files(
        'agx3_alu_core.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_binder.c',
        'test/test-agx3-alu-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_select_core',
    executable(
      'agx3_select_core_test',
      files(
        'agx3_select_core.c',
        'agx3_pack.c',
        'agx3_copy_core.c',
        'agx3_isel_core.c',
        'agx3_iadd_core.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_binder.c',
        'test/test-agx3-select-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_tile_core',
    executable(
      'agx3_tile_core_test',
      files(
        'agx3_tile_core.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'agx3_binder.c',
        'test/test-agx3-tile-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  # Pulls in a dozen encoder cores, so link the compiler library rather than
  # enumerating sources.
  test(
    'agx3_g17p_producers',
    executable(
      'agx3_g17p_producers_test',
      files('test/test-agx3-g17p-producers.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      dependencies : [idep_nir, idep_agx_opcodes_h, idep_agx_builder_h,
                      idep_mesautil, idep_libagx],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  # Apple's own G17P texture instructions, captured on real hardware. The
  # command-stream corpus contains no `smp` at all, so these are the only
  # ground truth for AGX3 texture addressing.
  test(
    'agx3_smp_witness',
    executable(
      'agx3_smp_witness_test',
      files(
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-smp-witness.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  # Apple's own decoded instructions, reproduced byte-exactly. This is the
  # ground truth every hand-written AGX3 encoder is validated against, so it
  # belongs in CI rather than being run by hand.
  test(
    'agx3_compose_corpus',
    executable(
      'agx3_compose_corpus_test',
      files(
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/agx3_compose_corpus_vectors.c',
        'test/test-agx3-compose-corpus.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_slot_core',
    executable(
      'agx3_slot_core_test',
      files(
        'agx3_slot_core.c',
        'test/test-agx3-slot-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_iadd_core',
    executable(
      'agx3_iadd_core_test',
      files(
        'agx3_iadd_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-iadd-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_isel_core',
    executable(
      'agx3_isel_core_test',
      files(
        'agx3_isel_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-isel-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_copy_core',
    executable(
      'agx3_copy_core_test',
      files(
        'agx3_copy_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-copy-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_scs_core',
    executable(
      'agx3_scs_core_test',
      files(
        'agx3_scs_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-scs-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_ioto_load_packer',
    executable(
      'agx3_ioto_load_packer_test',
      files(
        'agx3_ioto_load_packer.c',
        'agx3_mem_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        '../lib/agx_ioto_core.c',
        '../lib/agx_ioto_load_core.c',
        'test/test-agx3-ioto-load-packer.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_ioto_uniform_program',
    executable(
      'agx3_ioto_uniform_program_test',
      files(
        'agx3_ioto_uniform_program.c',
        'agx3_ioto_load_packer.c',
        'agx3_mem_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        '../lib/agx_ioto_core.c',
        '../lib/agx_ioto_load_core.c',
        '../lib/agx_ioto_profile_core.c',
        '../lib/agx_ioto_state_core.c',
        '../lib/agx_ioto_preshader_core.c',
        '../lib/agx_ioto_program_core.c',
        'test/test-agx3-ioto-uniform-program.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx_tests',
    executable(
      'agx_tests',
      files(
         'test/test-compact-constants.cpp',
         'test/test-minifloat.cpp',
         'test/test-optimizer.cpp',
         'test/test-lower-pseudo.cpp',
         'test/test-lower-parallel-copy.cpp',
         'test/test-lower-spill.cpp',
         'test/test-repair-ssa.cpp',
         'test/test-agx3-pack-binary.cpp',
         'test/test-agx3-sr.cpp',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies: [idep_gtest, idep_nir, idep_agx_opcodes_h, idep_agx_builder_h, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
    protocol : 'gtest',
  )

  test(
    'agx3_compile_fail_closed',
    executable(
      'agx3_compile_fail_closed_test',
      files('test/test-agx3-compile-fail-closed.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_int64',
    executable(
      'agx3_int64_test',
      files('test/test-agx3-int64.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_workgroup',
    executable(
      'agx3_workgroup_test',
      files('test/test-agx3-workgroup.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_multi_store',
    executable(
      'agx3_multi_store_test',
      files('test/test-agx3-multi-store.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_event_wait',
    executable(
      'agx3_event_wait_test',
      files('test/test-agx3-event-wait.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_pixel_core',
    executable(
      'agx3_pixel_core_test',
      files(
        'agx3_pixel_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-pixel-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_funop_core',
    executable(
      'agx3_funop_core_test',
      files(
        'agx3_funop_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-funop-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_rt_core',
    executable(
      'agx3_rt_core_test',
      files(
        'agx3_rt_core.c',
        'agx3_binder.c',
        'agx3_compose/agx3_compose.c',
        'agx3_compose/agx3b_tables.c',
        'agx3_compose/agx3c_tables.c',
        'test/test-agx3-rt-core.c',
      ),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_graphics_compile',
    executable(
      'agx3_graphics_compile_test',
      files('test/test-agx3-graphics-compile.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx_register_bank',
    executable(
      'agx_register_bank_test',
      files('test/test-register-bank.c'),
      c_args : [c_msvc_compat_args, no_override_init_args, '-UNDEBUG'],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_rgb10_memory',
    executable(
      'agx3_rgb10_memory_test',
      files('test/test-agx3-rgb10-memory.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )

  test(
    'agx3_ray_query_prerequisites',
    executable(
      'agx3_ray_query_prerequisites_test',
      files('test/test-agx3-ray-query-prerequisites.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      dependencies : [idep_nir, idep_mesautil],
      link_with : [libasahi_compiler],
    ),
    suite : ['asahi'],
  )
endif
