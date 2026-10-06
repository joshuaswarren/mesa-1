# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
#
# d3d12agx: experimental native Direct3D 12 driver on the AGX substrate.
# Campaign front door: ntasp repo, docs/d3d12agx-campaign.md.

dep_dxheaders_d12 = dependency('directx-headers', required : false)
if not dep_dxheaders_d12.found()
  dep_dxheaders_d12 = dependency(
    'DirectX-Headers',
    version : '>= 1.619.1',
    fallback : ['DirectX-Headers', 'dep_dxheaders'],
    default_options : ['build-test=false'],
    required : true,
  )
endif

libd3d12agx_files = files(
  'd12_agx3_cmdstream_core.h',
  'd12_agx3_index_list_core.c',
  'd12_agx3_graphics_bind_core.c',
  'd12_agx3_isa_core.c',
  'd12_agx3_ppp_hal300_depth_bias_scissor_core.c',
  'd12_agx3_ppp_hal300_depth_stencil_core.c',
  'd12_agx3_ppp_viewport_payload_core.c',
  'd12_agx3_ppp_w_clamp_cull2_core.c',
  'd12_agx3_vdm_barrier_core.c',
  'd12_agx3_vdm_core.c',
  'd12_agx3_vdm_decode_core.c',
  'd12_agx3_vdm_ppp_state_core.c',
  'd12_agx3_vdm_stream_link_core.c',
  'd12_agx3_vdm_stream_terminate_core.c',
  'd12_agx3_vdm_tessellate_core.c',
  'd12_agx3_vdm_state_update_core.c',
  'd12_attachment_view_core.c',
  'd12_core_caps.c',
  'd12_core_caps_query.c',
  'd12_dxbc_wrap.c',
  'd12_cmd.c',
  'd12_debug.c',
  'd12_descriptor.c',
  'd12_device.c',
  'd12_device_removal_core.c',
  'd12_fence.c',
  'd12_format.c',
  'd12_g17p_vdm_storage.c',
  'd12_g17p_txf_prefix_core.c',
  'd12_native_render_resources_core.c',
  'd12_native_compute.c',
  'd12_pipeline.c',
  'd12_nir_g17p_txf.c',
  'd12_present_blt_math.c',
  'd12_present_format_core.c',
  'd12_query.c',
  'd12_query_core.c',
  'd12_read_set_core.c',
  'd12_resolve_core.c',
  'd12_sparse_residency_core.c',
  'd12_queue.c',
  'd12_queue_timeline_core.c',
  'd12_resource.c',
  'd12_root_sig.c',
  'd12_shader_cache.c',
  'd12_texture_copy_math.c',
  'd12_viewport_math.c',
  'd12_write_set_core.c',
  'dxil/d12_dxil_bitstream.c',
  'dxil/d12_dxil_module.c',
  'dxil/d12_dxbc.c',
  'dxil/d12_dxil_nir.c',
)

d12_is_wine = host_machine.system() == 'windows' and asahi_d3d12_platform == 'wine'
d12_is_windows = host_machine.system() == 'windows' and not d12_is_wine
d12_core_strict_args = []
if d12_is_windows
  d12_core_strict_args = cc.get_supported_arguments(
    '-Wall',
    '-Wextra',
    '-Werror',
    '-Wpedantic',
    '-Wshadow',
    '-Wstrict-prototypes',
    '-Wmissing-prototypes',
    '-Wno-unknown-pragmas',
    '-Wno-ignored-attributes',
    '-Wno-microsoft-enum-forward-reference',
    '-Wno-language-extension-token',
    '-Wno-c23-extensions',
    # NIR's public helper macros intentionally use GNU statement expressions
    # and omit variadic arguments.  Generated AGX pack headers also expose
    # callbacks with unused formal parameters.  Keep the owned D3D12 sources
    # warning-clean without treating those upstream/generated interfaces as
    # build failures.
    '-Wno-gnu-statement-expression-from-macro-expansion',
    '-Wno-gnu-zero-variadic-macro-arguments',
    '-Wno-unused-parameter',
    '-Wno-gnu-empty-struct',
    '-Wno-zero-length-array',
    '-Wno-missing-field-initializers',
    # Common AGX headers intentionally provide static inline helpers that
    # are not consumed by every translation unit.
    '-Wno-unused-function',
    # Public COM vtables are initialized from generated fail-safe defaults and
    # then override the operations implemented by this frontend.
    '-Wno-initializer-overrides',
  )
endif
if d12_is_windows
  libd3d12agx_files += files(
    'd12_platform_windows.c',
    'd12_device_runtime.c',
  )
  # d12_platform_windows.c includes the D3DKMT winsys interface, whose public
  # header exposes the shared AppleAgxWddm wire contract.  libasahi_lib uses
  # this include privately, so propagate it to this direct consumer too.
  d12_platform_includes = [libasahi_lib_inc_win]
  d12_platform_deps = []
elif d12_is_wine
  # Wine/Proton: ARM64EC PE d3d12.dll. The Linux DRM driver body, kernel
  # calls proxied to d3d12.so (wine/unix) through a tiny libdrm shim; the DRM
  # syncobj sync provider is compiled in here because libmesa_util only
  # carries it when a real libdrm exists. See wine/DESIGN.md.
  libd3d12agx_files += files(
    'wine/d12_platform_wine.c',
    'wine/d12_wine_drm.c',
    'wine/d12_wine_swapchain.c',
    '../../util/u_sync_provider.c',
  )
  d12_platform_includes = [inc_asahi_d3d12_wine, include_directories('wine')]
  # DXGI interface IIDs used by the presenter come from mingw's dxguid.
  d12_platform_deps = [cc.find_library('dxguid'), cc.find_library('ntdll')]
  d12_core_strict_args += ['-DHAVE_LIBDRM', '-DAGX_NO_VIRTIO', '-DAGX_WINE_UNIXLIB', '-DD12_PLATFORM_WINE']
else
  libd3d12agx_files += files('d12_platform_linux.c')
  d12_platform_includes = [inc_virtio_gpu, inc_virtio_vdrm]
  d12_platform_deps = [dep_libdrm]
endif

libd3d12agx_core = static_library(
  'd3d12agx_core',
  libd3d12agx_files,
  include_directories : [
    inc_include,
    inc_src,
    inc_asahi,
    d12_platform_includes,
    include_directories('../isa'),
  ],
  dependencies : [
    dep_dxheaders_d12,
    d12_platform_deps,
    dep_valgrind,
    idep_nir,
    idep_mesautil,
    idep_agx_pack,
    # d12_cmd.c directly includes the generated libagx_shaders.h.  Linkage
    # against libasahi_lib is not a compile-time dependency, so an empty
    # parallel build can otherwise compile d12_cmd.c before the header exists.
    idep_libagx_shaders_h,
    idep_libagx,
    idep_libpoly,
  ],
  link_with : [
    libasahi_lib, libasahi_layout, libasahi_compiler, libagx2_disasm,
    libagx3b_disasm,
    libpoly_nir,
  ],
  c_args : [no_override_init_args, d12_core_strict_args],
  gnu_symbol_visibility : 'hidden',
  install : false,
)

# Keep the renderer implementation in one archive.  The public COM frontend
# and the WDDM DDI frontend select different exports and object-lifetime glue,
# but link the same compiler, pipeline, resource-layout and command encoder
# implementation.  link_whole is intentional: the Windows .def files are the
# roots that retain otherwise-unreferenced entrypoints from the archive.
libd3d12agx = shared_library(
  d12_is_wine ? 'd3d12' : 'd3d12agx',
  link_whole : [libd3d12agx_core],
  dependencies : [
    dep_dxheaders_d12,
    d12_platform_deps,
    dep_valgrind,
    idep_nir,
    idep_mesautil,
    idep_agx_pack,
  ],
  link_with : [
    libasahi_lib, libasahi_layout, libasahi_compiler, libagx2_disasm,
    libagx3b_disasm,
    libpoly_nir,
  ],
  vs_module_defs : files(d12_is_wine ? 'wine/d3d12.def' : 'd3d12agx.def')[0],
  name_prefix : (d12_is_windows or d12_is_wine) ? '' : [],
  install : false,
)

if d12_is_windows
  d12_windows = import('windows')
  d12_umd_wdk_include = meson.get_external_property(
    'd3d12_umd_wdk_include',
    '',
  )
  d12_umd_generated_include = meson.get_external_property(
    'd3d12_umd_generated_include',
    '',
  )

  if d12_umd_wdk_include != '' and d12_umd_generated_include != ''
    d12_umd_wddm_version = get_option('asahi-d3d12-wddm-version')
    if d12_umd_wddm_version == '3.1'
      d12_umd_interface_version = '0x10000'
      d12_umd_wddm_major = '31'
    elif d12_umd_wddm_version == '2.4'
      d12_umd_interface_version = '0x9001'
      d12_umd_wddm_major = '24'
    elif d12_umd_wddm_version == '2.3'
      d12_umd_interface_version = '0x8001'
      d12_umd_wddm_major = '23'
    elif d12_umd_wddm_version == '2.1'
      d12_umd_interface_version = '0x6003'
      d12_umd_wddm_major = '21'
    else
      d12_umd_interface_version = '0x5002'
      d12_umd_wddm_major = '20'
    endif
    d12_umd_strict_args = d12_core_strict_args + cc.get_supported_arguments(
      '-Wconversion',
      '-Wundef',
    )
    d12_umd_wdk_inc = include_directories(
      d12_umd_wdk_include,
      is_system : true,
    )
    d12_umd_generated_inc = include_directories(d12_umd_generated_include)

    libd3d12agx_umd_glue = static_library(
      'd3d12agx_umd_glue',
      files(
        'umd/d12_umd_abi.c',
        'umd/d12_umd_kmt.c',
        'umd/d12_umd_agx_transport.c',
        'umd/d12_umd_backend.c',
        'umd/d12_umd_command_core.c',
        'umd/d12_umd_lifetime_core.c',
        'umd/d12_umd_command_funcs.c',
        'umd/d12_umd_pipeline_serialization.c',
        'umd/d12_umd_pipeline_library.c',
        'umd/d12_umd_state_funcs.c',
        'umd/d12_umd_pipeline_funcs.c',
        'umd/d12_umd_queue_runtime.c',
        'umd/d12_umd_residency.c',
        'umd/d12_umd_residency_core.c',
        'umd/d12_umd_resource_private.c',
        'umd/d12_umd_device_funcs.c',
        'umd/d12_umd_descriptor_funcs.c',
        'umd/d12_umd_query_funcs.c',
        'umd/d12_umd_resource_command_funcs.c',
        'umd/d12_umd_present_funcs.c',
        'umd/d12_umd_unsupported_funcs.c',
        'umd/d12_umd_tables.c',
        'umd/d12_umd_adapter.c',
      ),
      include_directories : [
        inc_include,
        inc_src,
        inc_asahi,
        include_directories('../isa'),
        d12_platform_includes,
        d12_umd_wdk_inc,
        d12_umd_generated_inc,
      ],
      dependencies : [dep_dxheaders_d12],
      c_args : [
        no_override_init_args,
        d12_umd_strict_args,
        '-UWIN32_LEAN_AND_MEAN',
        '-DNTDDI_VERSION=0x0A000010',
        '-DD3D_UMD_INTERFACE_VERSION=' + d12_umd_interface_version,
        '-DD12_UMD_WITH_AGX_BACKEND=1',
      ],
      gnu_symbol_visibility : 'hidden',
      install : false,
    )

    libd3d12agx_umd = shared_library(
      'd3d12agx_umd',
      d12_windows.compile_resources(
        'umd/d3d12agx_umd.rc',
        args : ['-DAURORA_WDDM_MAJOR=' + d12_umd_wddm_major],
      ),
      dependencies : [
        dep_dxheaders_d12,
        d12_platform_deps,
        dep_valgrind,
        idep_nir,
        idep_mesautil,
        idep_agx_pack,
      ],
      link_whole : [libd3d12agx_core, libd3d12agx_umd_glue],
      link_with : [
        libasahi_lib, libasahi_layout, libasahi_compiler, libagx2_disasm,
        libagx3b_disasm,
        libpoly_nir,
      ],
      gnu_symbol_visibility : 'hidden',
      vs_module_defs : files('umd/d3d12agx_umd.def')[0],
      name_prefix : '',
      install : false,
    )
  else
    warning(
      'D3D12 WDDM UMD target disabled: set d3d12_umd_wdk_include and ' +
      'd3d12_umd_generated_include in the Windows cross file',
    )
  endif
endif

# Live API contract checks beyond the frozen suites: interface versions,
# Device4+/Device15 entry points, NULL-argument robustness.
executable(
  'd3d12agx_device_interfaces',
  files('tests/device_interfaces.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

# Live queue/fence contract: GPU and CPU Signal ordering, multi-fence waits,
# cross-queue Wait, and queue release with parked waiters.
executable(
  'd3d12agx_queue_fence_live',
  files('tests/queue_fence_live.c'),
  dependencies : [dep_dxheaders_d12, dependency('threads')],
  link_with : [libd3d12agx],
)

# Live: PSO lifetime under a pipeline library from several threads.
executable(
  'd3d12agx_pso_library_race',
  files('tests/pso_library_race.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

executable(
  'd3d12agx_resource_validation',
  files('tests/resource_validation.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

executable(
  'd3d12agx_resource_copies',
  files('tests/resource_copies.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

executable(
  'd3d12agx_cpu_volume_live',
  files('tests/cpu_volume_live.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

# Checked placement math can also run without a GPU or DirectX headers.
d3d12agx_allocation_math_tests = executable(
  'd3d12agx_allocation_math_tests',
  files('tests/d12_allocation_math_tests.c'),
)
test('d3d12agx_allocation_math', d3d12agx_allocation_math_tests)

# CPU-side microbenchmark: record/submit cost, descriptor writes, PSO creation.
executable(
  'd3d12agx_streamout_topology_live',
  files('tests/streamout_topology_live.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

if not d12_is_windows
  executable(
    'd3d12agx_fence_multiple_live',
    files('tests/fence_multiple_live.c'),
    dependencies : [dep_dxheaders_d12],
    link_with : [libd3d12agx],
  )
  executable(
    'd3d12agx_command_sync_live',
    files('tests/command_sync_live.c'),
    dependencies : [dep_dxheaders_d12],
    link_with : [libd3d12agx],
  )
endif

d3d12agx_microbench = executable(
  'd3d12agx_microbench',
  files('tests/d12_microbench.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

# Frame-level benchmark: dlopens the D3D12 implementation (D3D12_LIBRARY), so
# the same binary measures d3d12agx and a native vkd3d-proton build.
if host_machine.system() != 'windows'
  executable(
    'd3d12agx_framebench',
    files('tests/d12_framebench.c'),
    dependencies : [dep_dxheaders_d12, dep_dl],
  )
endif

# Concurrent PSO creation over the device shader cache.
executable(
  'd3d12agx_pso_threads',
  files('tests/pso_threads.c'),
  dependencies : [dep_dxheaders_d12, dep_thread],
  link_with : [libd3d12agx],
)

d3d12agx_g1_smoke = executable(
  'd3d12agx_g1_smoke',
  files('tests/g1_smoke.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_shared_interop = executable(
  'd3d12agx_shared_interop',
  files('tests/shared_interop.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g2_smoke = executable(
  'd3d12agx_g2_smoke',
  files('tests/g2_smoke.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g3_smoke = executable(
  'd3d12agx_g3_smoke',
  files('tests/g3_smoke.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_render_ppm = executable(
  'd3d12agx_render_ppm',
  files('tests/render_ppm.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g4_baseline_pixels = executable(
  'd3d12agx_g4_baseline_pixels',
  files('tests/g4_baseline_pixels.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g4_pso_compile = executable(
  'd3d12agx_g4_pso_compile',
  files('tests/g4_pso_compile.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g4_triangle = executable(
  'd3d12agx_g4_triangle',
  files('tests/g4_triangle.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g4_smoke = executable(
  'd3d12agx_g4_smoke',
  files('tests/g4_smoke.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g5_wave = executable(
  'd3d12agx_g5_wave',
  files('tests/g5_wave.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g5_control_flow = executable(
  'd3d12agx_g5_control_flow',
  files('tests/g5_control_flow.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g5_atomics = executable(
  'd3d12agx_g5_atomics',
  files('tests/g5_atomics.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g5_descriptors = executable(
  'd3d12agx_g5_descriptors',
  files('tests/g5_descriptors.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g5_texture = executable(
  'd3d12agx_g5_texture',
  files('tests/g5_texture.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g5_texture_compile_only = executable(
  'd3d12agx_g5_texture_compile_only',
  files('tests/g5_texture_compile_only.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

# Build-time only: generates committable probe DXIL so the suites do not
# need dxc (absent on aarch64) or the LGPL corpus. Nothing links it.
d3d12agx_make_probe = executable(
  'd3d12agx_make_probe',
  files('tests/make_probe.c'),
  include_directories : [inc_include, inc_src, inc_compiler, inc_gallium],
  dependencies : [idep_nir, idep_mesautil, idep_libdxil_compiler],
)

d3d12agx_g11_indexed_instanced = executable(
  'd3d12agx_g11_indexed_instanced',
  files('tests/g11_indexed_instanced.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

# Live: a declared render pass the driver splits keeps its first half.
executable(
  'd3d12agx_render_pass_split',
  files('tests/render_pass_split.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

# Live: state recorded inside a bundle reaches its draws and its caller.
executable(
  'd3d12agx_bundle_state',
  files('tests/bundle_state.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

# Live: ExecuteIndirect whose count buffer is written by the GPU in-list.
executable(
  'd3d12agx_execute_indirect_count',
  files('tests/execute_indirect_count.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g10_vertex_buffers = executable(
  'd3d12agx_g10_vertex_buffers',
  files('tests/g10_vertex_buffers.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g9_amplification = executable(
  'd3d12agx_g9_amplification',
  files('tests/g9_amplification.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g8_mesh = executable(
  'd3d12agx_g8_mesh',
  files('tests/g8_mesh.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g7_tessellation = executable(
  'd3d12agx_g7_tessellation',
  files('tests/g7_tessellation.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g6_geometry = executable(
  'd3d12agx_g6_geometry',
  files('tests/g6_geometry.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

d3d12agx_g5_depth = executable(
  'd3d12agx_g5_depth',
  files('tests/g5_depth.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

# DXIL frontend (G4): standalone of the COM driver for now
libd12_dxil_files = files(
  'd12_debug.c',
  'dxil/d12_dxil_bitstream.c',
  'dxil/d12_dxil_module.c',
)

d3d12agx_dxil_dump = executable(
  'd3d12agx_dxil_dump',
  [libd12_dxil_files, files('dxil/d12_dxil_dump.c')],
  include_directories : [inc_include, inc_src, inc_compiler],
  dependencies : [idep_mesautil],
)

# DXIL -> NIR -> AGX, the "no SPIR-V" path end to end
d3d12agx_dxil_compile = executable(
  'd3d12agx_dxil_compile',
  [libd12_dxil_files, files('dxil/d12_dxil_nir.c', 'dxil/d12_dxbc.c',
                            'dxil/d12_dxil_compile.c')],
  include_directories : [
    inc_include, inc_src, inc_asahi, inc_gallium, inc_gallium_aux,
    include_directories('../isa'),
  ],
  dependencies : [idep_nir, idep_mesautil, idep_agx_pack, idep_libagx],
  link_with : [
    libasahi_compiler, libasahi_lib, libasahi_layout, libagx2_disasm,
  ],
)

# DXIL parser capacity: a synthetic module past every old fixed table size,
# built with Mesa's DXIL writer, parsed and translated. CPU only.
d3d12agx_dxil_limits_tests = executable(
  'd3d12agx_dxil_limits_tests',
  [libd12_dxil_files, files('dxil/d12_dxil_nir.c', 'dxil/d12_dxbc.c',
                            'tests/d12_dxil_limits_tests.c')],
  include_directories : [
    inc_include, inc_src, inc_asahi, inc_gallium, inc_gallium_aux,
    include_directories('../isa'),
  ],
  dependencies : [idep_nir, idep_mesautil, idep_agx_pack, idep_libagx,
                  idep_libdxil_compiler],
  link_with : [
    libasahi_compiler, libasahi_lib, libasahi_layout, libagx2_disasm,
  ],
)
test('d3d12agx_dxil_limits', d3d12agx_dxil_limits_tests,
     suite : ['asahi', 'd3d12'])

d3d12agx_dxil_mesh_tests = executable(
  'd3d12agx_dxil_mesh_tests',
  [libd12_dxil_files, files('dxil/d12_dxil_nir.c', 'dxil/d12_dxbc.c',
                            'tests/d12_dxil_mesh_tests.c')],
  include_directories : [
    inc_include, inc_src, inc_asahi, inc_gallium, inc_gallium_aux,
    include_directories('../isa'),
  ],
  dependencies : [idep_nir, idep_mesautil, idep_agx_pack, idep_libagx,
                  idep_libdxil_compiler],
  link_with : [
    libasahi_compiler, libasahi_lib, libasahi_layout, libagx2_disasm,
  ],
  c_args : ['-UNDEBUG'],
)
test('d3d12agx_dxil_mesh', d3d12agx_dxil_mesh_tests,
     suite : ['asahi', 'd3d12'])

# CPU-only production lowering/ISA checks. Never creates a DRM device or queue.
d3d12agx_heap_desc_tests = executable(
  'd3d12agx_heap_desc_tests',
  files('tests/d12_heap_desc_tests.c'),
  dependencies : [dep_dxheaders_d12],
  c_args : ['-UNDEBUG'],
)
d3d12agx_submit_tests = executable(
  'd3d12agx_submit_tests',
  files('tests/d12_submit_tests.c'),
  include_directories : [inc_include, inc_src],
  c_args : ['-UNDEBUG'],
)
test('d3d12agx_heap_desc', d3d12agx_heap_desc_tests, suite : ['asahi', 'd3d12'])
test('d3d12agx_submit', d3d12agx_submit_tests, suite : ['asahi', 'd3d12'])

d3d12agx_native_direct_draw_tests = executable(
  'd3d12agx_native_direct_draw_tests',
  files('tests/d12_agx3_index_list_core_tests.c', 'd12_agx3_index_list_core.c',
        '../lib/agx_vdm_index_list_core.c',
        '../lib/agx_vdm_hal200_direct_draw_core.c'),
  include_directories : [inc_include, inc_src, inc_asahi],
)
test('d3d12agx_native_direct_draw', d3d12agx_native_direct_draw_tests,
     suite : ['asahi', 'd3d12'])

d3d12agx_nir_integer_tests = executable(
  'd3d12agx_nir_integer_tests',
  files('tests/d12_nir_integer_tests.c'),
  include_directories : [inc_include, inc_src],
  dependencies : [idep_nir, idep_mesautil],
  c_args : ['-UNDEBUG'],
)
test('d3d12agx_nir_integer', d3d12agx_nir_integer_tests,
     suite : ['asahi', 'd3d12'])

d3d12agx_texture_descriptor_depth_tests = executable(
  'd3d12agx_texture_descriptor_depth_tests',
  files('tests/d12_texture_descriptor_depth_tests.c'),
  include_directories : [inc_include, inc_src, inc_asahi,
                        d12_platform_includes, include_directories('../isa')],
  dependencies : [dep_dxheaders_d12, d12_platform_deps, idep_nir,
                  idep_mesautil, idep_agx_pack, idep_libagx, idep_libagx_shaders_h],
  link_with : [libd3d12agx_core, libasahi_lib, libasahi_layout,
               libasahi_compiler, libagx2_disasm, libagx3b_disasm, libpoly_nir],
  c_args : ['-UNDEBUG'],
)
test('d3d12agx_texture_descriptor_depth', d3d12agx_texture_descriptor_depth_tests,
     suite : ['asahi', 'd3d12'])

d3d12agx_dxbc_sparse_tests = executable(
  'd3d12agx_dxbc_sparse_tests',
  files('tests/d12_dxbc_sparse_tests.c'),
  include_directories : [inc_include, inc_src, inc_asahi],
  dependencies : [idep_nir, idep_mesautil, idep_agx_pack, idep_libagx],
  link_with : [libd3d12agx_core, libasahi_lib, libasahi_layout,
               libasahi_compiler, libagx2_disasm, libagx3b_disasm, libpoly_nir],
  c_args : ['-UNDEBUG'],
)
test('d3d12agx_dxbc_sparse', d3d12agx_dxbc_sparse_tests,
     suite : ['asahi', 'd3d12'])

d3d12agx_nir_sparse_buffer_tests = executable(
  'd3d12agx_nir_sparse_buffer_tests',
  files('tests/d12_nir_sparse_buffer_tests.c'),
  include_directories : [inc_include, inc_src],
  dependencies : [idep_nir, idep_mesautil],
  c_args : ['-UNDEBUG'],
)
test('d3d12agx_nir_sparse_buffer', d3d12agx_nir_sparse_buffer_tests,
     suite : ['asahi', 'd3d12'])

d3d12agx_nir_min_lod_tests = executable(
  'd3d12agx_nir_min_lod_tests',
  files('tests/d12_nir_min_lod_tests.c'),
  include_directories : [inc_include, inc_src],
  dependencies : [idep_nir, idep_mesautil],
  c_args : ['-UNDEBUG'],
)
test('d3d12agx_nir_min_lod', d3d12agx_nir_min_lod_tests,
     suite : ['asahi', 'd3d12'])

# Supplemental live API validation; the frozen G1 suite remains unchanged.
executable(
  'd3d12agx_createheap_validation',
  files('tests/createheap_validation.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : libd3d12agx,
)
d3d12agx_g17p_txf_lower_tests = executable(
  'd3d12agx_g17p_txf_lower_tests',
  files('d12_nir_g17p_txf.c', 'tests/d12_nir_g17p_txf_tests.c'),
  include_directories : [inc_include, inc_src, inc_asahi],
  dependencies : [idep_nir, idep_mesautil, idep_agx_pack, idep_libagx],
  link_with : [libasahi_compiler, libasahi_lib, libasahi_layout],
)

# Original first depth pair with strict full-frame checks. Separate partial
# physical qualification probe; mandatory full G5 target is unchanged.
d3d12agx_g5_d32_first_pair = executable(
  'd3d12agx_g5_d32_first_pair',
  files('tests/g5_depth.c'),
  c_args : ['-DD12_D32_FIRST_PAIR_PROBE=1'],
  include_directories : [inc_include, inc_src, inc_asahi],
  dependencies : [dep_dxheaders_d12],
  link_with : libd3d12agx,
)

# Source-only bounded metadata diagnostics; neither target admits a GPU feature.
d3d12agx_g17p_txf_prefix_core_tests = executable(
  'd3d12agx_g17p_txf_prefix_core_tests',
  files('d12_g17p_txf_prefix_core.c', 'tests/d12_g17p_txf_prefix_core_tests.c'),
)
d3d12agx_g5_texture_native_metadata = executable(
  'd3d12agx_g5_texture_native_metadata',
  files('tests/g5_texture_native_metadata.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : libd3d12agx,
)

# Cold native graphics state validation needs no GPU.
d3d12agx_graphics_bind_core_tests = executable(
  'd3d12agx_graphics_bind_core_tests',
  files('d12_agx3_graphics_bind_core.c', 'tests/d12_agx3_graphics_bind_core_tests.c',
        '../lib/agx_ppp_hal200_graphics_core.c',
        '../lib/agx_ppp_hal200_raster_core.c',
        '../lib/agx_ppp_w_clamp_cull2_core.c',
        '../lib/agx_ppp_viewport_payload_core.c',
        '../lib/agx_vdm_hal200_bind_tail_core.c', '../lib/agx_vdm_state_core.c'),
  include_directories : [inc_include, inc_src],
  c_args : ['-UNDEBUG'],
)
test('d3d12agx_graphics_bind_core', d3d12agx_graphics_bind_core_tests,
     suite : ['asahi', 'd3d12'])

# The derivation test has no COM or Mesa dependency and needs no GPU.
d3d12agx_core_caps_tests = executable(
  'd3d12agx_core_caps_tests',
  files('d12_core_caps.c', 'tests/d12_core_caps_tests.c'),
)
test('d3d12agx_core_caps', d3d12agx_core_caps_tests, suite : ['asahi', 'd3d12'])

executable(
  'd3d12agx_caps_queries',
  files('tests/caps_queries.c'),
  dependencies : [dep_dxheaders_d12],
  link_with : [libd3d12agx],
)

if host_machine.system() != 'windows'
  executable(
    'd3d12agx_caps_probe',
    files('tests/caps_probe.c'),
    dependencies : [dep_dxheaders_d12, dep_dl],
  )
endif

# Uses the same runtime archive as libd3d12agx, with access to retained private
# program metadata. Registered noop fd only; no command queue or submission.
if not d12_is_windows
  d3d12agx_native_stage_certification = executable(
    'd3d12agx_native_stage_certification',
    files('tests/native_stage_certification.c'),
    include_directories : [inc_include, inc_src, inc_asahi,
                          d12_platform_includes, include_directories('../isa')],
    dependencies : [dep_dxheaders_d12, d12_platform_deps, dep_valgrind,
                    idep_nir, idep_mesautil, idep_agx_pack, idep_libagx,
                    idep_libagx_shaders_h, idep_libpoly],
    link_with : [libd3d12agx_core, libasahi_lib, libasahi_layout,
                 libasahi_compiler, libagx2_disasm, libagx3b_disasm,
                 libpoly_nir],
  )
endif
