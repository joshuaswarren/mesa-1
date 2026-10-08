#!/bin/zsh
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT

set -euo pipefail

umd_dir=${0:A:h}
d12_dir=${umd_dir:h}
repo_root=${umd_dir:h:h:h:h}
main_contract=${repo_root:h}/AuroraSilicon/drivers/AppleAgxWddm/core
contract_compatible() {
    [[ -f $1/agxw_contract.h ]] &&
        grep -Eq '^#define[[:space:]]+AGXW_CONTRACT_VERSION[[:space:]]+([3-9]|[1-9][0-9]+)u' "$1/agxw_contract.h" &&
        grep -q 'submit_private_max_version' "$1/agxw_contract.h" &&
        grep -q 'AGXW_SUBMIT_PRIVATE_VERSION_V4' "$1/agxw_contract.h" &&
        grep -q 'AGXW_SUBMIT_PRIVATE_VERSION_V5' "$1/agxw_contract.h" &&
        grep -q 'AGXW_SUBMIT_CLEAN_SET_COMPLETE' "$1/agxw_contract.h"
}
if [[ -n ${AGX_WDDM_CONTRACT_CORE:-} ]]; then
    contract_dir=$AGX_WDDM_CONTRACT_CORE
elif contract_compatible "$main_contract"; then
    contract_dir=$main_contract
else
    contract_dir=$main_contract
fi
if ! contract_compatible "$contract_dir"; then
    print -u2 "latest merged AGX WDDM contract (v4 placement + v5 clean sets) is required: $contract_dir"
    print -u2 "set AGX_WDDM_CONTRACT_CORE to the latest AppleAgxWddm core directory"
    exit 2
fi
out=${D12_UMD_HOST_TEST_OUT:-$repo_root/_build-d3d12-umd-host}
cc=${HOST_CC:-clang}

mkdir -p "$out"

common=(
    -std=c11
    -O2
    -Wall -Wextra -Werror -Wpedantic
)

for target in 20 21 23 24 31; do
    if [[ $target == 20 ]]; then
        interface=0x5002
    elif [[ $target == 21 ]]; then
        interface=0x6003
    elif [[ $target == 23 ]]; then
        interface=0x8001
    elif [[ $target == 24 ]]; then
        interface=0x9001
    else
        interface=0x10000
    fi
    "$cc" "$common[@]" \
        -DD3D_UMD_INTERFACE_VERSION=$interface \
        -I"$contract_dir" -I"$umd_dir" \
        "$umd_dir/d12_umd_abi.c" "$umd_dir/tests/d12_umd_abi_tests.c" \
        -o "$out/d12_umd_abi_tests_wddm$target"
    "$out/d12_umd_abi_tests_wddm$target"
done

"$cc" "$common[@]" \
    "$d12_dir/d12_attachment_view_core.c" \
    "$d12_dir/tests/d12_attachment_view_core_tests.c" \
    -o "$out/d12_attachment_view_core_tests"
"$out/d12_attachment_view_core_tests"

"$cc" "$common[@]" -I"$d12_dir" \
    "$d12_dir/tests/d12_buffer_view_core_tests.c" \
    -o "$out/d12_buffer_view_core_tests"
"$out/d12_buffer_view_core_tests"

"$cc" "$common[@]" -I"$d12_dir" \
    "$d12_dir/tests/d12_blend_core_tests.c" \
    -o "$out/d12_blend_core_tests"
"$out/d12_blend_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" \
    -I"$repo_root/include" \
    "$repo_root/src/asahi/compiler/agx3_pack.c" \
    "$repo_root/src/asahi/compiler/agx3_cf_core.c" \
    "$repo_root/src/asahi/compiler/agx3_tex_core.c" \
    "$repo_root/src/asahi/compiler/agx3_tile_core.c" \
    "$repo_root/src/asahi/compiler/agx3_vary_core.c" \
    "$repo_root/src/asahi/compiler/agx3_local_tps_core.c" \
    "$repo_root/src/asahi/compiler/agx3_static_spill_core.c" \
    "$repo_root/src/asahi/compiler/agx3_subgroup_core.c" \
    "$repo_root/src/asahi/compiler/agx3_binder.c" \
    "$repo_root/src/asahi/compiler/agx3_compose/agx3_compose.c" \
    "$repo_root/src/asahi/compiler/agx3_compose/agx3b_tables.c" \
    "$repo_root/src/asahi/compiler/agx3_compose/agx3c_tables.c" \
    "$d12_dir/d12_agx3_isa_core.c" \
    "$d12_dir/tests/d12_agx3_isa_core_tests.c" \
    -o "$out/d12_agx3_isa_core_tests"
"$out/d12_agx3_isa_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_vdm_state_core.c" \
    "$d12_dir/d12_agx3_vdm_core.c" \
    "$d12_dir/tests/d12_agx3_vdm_core_tests.c" \
    -o "$out/d12_agx3_vdm_core_tests"
"$out/d12_agx3_vdm_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_vdm_barrier_core.c" \
    "$repo_root/src/asahi/lib/agx_vdm_ppp_state_core.c" \
    "$repo_root/src/asahi/lib/agx_vdm_state_core.c" \
    "$repo_root/src/asahi/lib/agx_vdm_state_update_core.c" \
    "$repo_root/src/asahi/lib/agx_vdm_index_list_core.c" \
    "$repo_root/src/asahi/lib/agx_vdm_stream_link_core.c" \
    "$repo_root/src/asahi/lib/agx_vdm_stream_terminate_core.c" \
    "$repo_root/src/asahi/lib/agx_vdm_tessellate_core.c" \
    "$repo_root/src/asahi/lib/agx_vdm_decode_core.c" \
    "$d12_dir/d12_agx3_vdm_decode_core.c" \
    "$d12_dir/tests/d12_agx3_vdm_decode_core_tests.c" \
    -o "$out/d12_agx3_vdm_decode_core_tests"
"$out/d12_agx3_vdm_decode_core_tests"

"$cc" "${common[@]}" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_vdm_barrier_core.c" \
    "$d12_dir/d12_agx3_vdm_barrier_core.c" \
    "$d12_dir/tests/d12_agx3_vdm_barrier_core_tests.c" \
    -o "$out/d12_agx3_vdm_barrier_core_tests"
"$out/d12_agx3_vdm_barrier_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_ppp_viewport_payload_core.c" \
    "$d12_dir/d12_agx3_ppp_viewport_payload_core.c" \
    "$d12_dir/tests/d12_agx3_ppp_viewport_payload_core_tests.c" \
    -o "$out/d12_agx3_ppp_viewport_payload_core_tests"
"$out/d12_agx3_ppp_viewport_payload_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_ppp_w_clamp_cull2_core.c" \
    "$d12_dir/d12_agx3_ppp_w_clamp_cull2_core.c" \
    "$d12_dir/tests/d12_agx3_ppp_w_clamp_cull2_core_tests.c" \
    -o "$out/d12_agx3_ppp_w_clamp_cull2_core_tests"
"$out/d12_agx3_ppp_w_clamp_cull2_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_ppp_hal300_depth_bias_scissor_core.c" \
    "$d12_dir/d12_agx3_ppp_hal300_depth_bias_scissor_core.c" \
    "$d12_dir/tests/d12_agx3_ppp_hal300_depth_bias_scissor_core_tests.c" \
    -o "$out/d12_agx3_ppp_hal300_depth_bias_scissor_core_tests"
"$out/d12_agx3_ppp_hal300_depth_bias_scissor_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_ppp_hal300_depth_stencil_core.c" \
    "$d12_dir/d12_agx3_ppp_hal300_depth_stencil_core.c" \
    "$d12_dir/tests/d12_agx3_ppp_hal300_depth_stencil_core_tests.c" \
    -o "$out/d12_agx3_ppp_hal300_depth_stencil_core_tests"
"$out/d12_agx3_ppp_hal300_depth_stencil_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_vdm_ppp_state_core.c" \
    "$d12_dir/d12_agx3_vdm_ppp_state_core.c" \
    "$d12_dir/tests/d12_agx3_vdm_ppp_state_core_tests.c" \
    -o "$out/d12_agx3_vdm_ppp_state_core_tests"
"$out/d12_agx3_vdm_ppp_state_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_vdm_stream_link_core.c" \
    "$d12_dir/d12_agx3_vdm_stream_link_core.c" \
    "$d12_dir/tests/d12_agx3_vdm_stream_link_core_tests.c" \
    -o "$out/d12_agx3_vdm_stream_link_core_tests"
"$out/d12_agx3_vdm_stream_link_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_vdm_stream_terminate_core.c" \
    "$d12_dir/d12_agx3_vdm_stream_terminate_core.c" \
    "$d12_dir/tests/d12_agx3_vdm_stream_terminate_core_tests.c" \
    -o "$out/d12_agx3_vdm_stream_terminate_core_tests"
"$out/d12_agx3_vdm_stream_terminate_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_vdm_tessellate_core.c" \
    "$d12_dir/d12_agx3_vdm_tessellate_core.c" \
    "$d12_dir/tests/d12_agx3_vdm_tessellate_core_tests.c" \
    -o "$out/d12_agx3_vdm_tessellate_core_tests"
"$out/d12_agx3_vdm_tessellate_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_vdm_state_core.c" \
    "$repo_root/src/asahi/lib/agx_vdm_state_update_core.c" \
    "$d12_dir/d12_agx3_vdm_state_update_core.c" \
    "$d12_dir/tests/d12_agx3_vdm_state_update_core_tests.c" \
    -o "$out/d12_agx3_vdm_state_update_core_tests"
"$out/d12_agx3_vdm_state_update_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/lib/agx_vdm_index_list_core.c" \
    "$d12_dir/d12_agx3_index_list_core.c" \
    "$d12_dir/tests/d12_agx3_index_list_core_tests.c" \
    -o "$out/d12_agx3_index_list_core_tests"
"$out/d12_agx3_index_list_core_tests"

"$cc" "$common[@]" -I"$d12_dir" \
    -I"$repo_root/src" -I"$repo_root/include" -I"$repo_root/src/compiler" \
    "$d12_dir/tests/d12_binding_map_tests.c" \
    -o "$out/d12_binding_map_tests"
"$out/d12_binding_map_tests"

"$cc" "$common[@]" -I"$d12_dir" \
    "$d12_dir/tests/d12_bundle_state_core_tests.c" \
    -o "$out/d12_bundle_state_core_tests"
"$out/d12_bundle_state_core_tests"

"$cc" "$common[@]" -I"$d12_dir" \
    "$d12_dir/tests/d12_pso_stream_core_tests.c" \
    -o "$out/d12_pso_stream_core_tests"
"$out/d12_pso_stream_core_tests"

"$cc" "$common[@]" -I"$d12_dir" \
    "$d12_dir/tests/d12_sampler_reduction_core_tests.c" \
    -o "$out/d12_sampler_reduction_core_tests"
"$out/d12_sampler_reduction_core_tests"

"$cc" "$common[@]" -I"$d12_dir" \
    "$d12_dir/tests/d12_render_pass_core_tests.c" \
    -o "$out/d12_render_pass_core_tests"
"$out/d12_render_pass_core_tests"

"$cc" "$common[@]" -I"$d12_dir" \
    "$d12_dir/tests/d12_sample_position_core_tests.c" \
    -o "$out/d12_sample_position_core_tests"
"$out/d12_sample_position_core_tests"

"$cc" "$common[@]" -Wno-c23-extensions \
    -I"$d12_dir" -I"$repo_root/src" -I"$repo_root/include" \
    "$repo_root/src/asahi/compiler/agx3_pack.c" \
    "$d12_dir/tests/d12_raytracing_core_tests.c" \
    -o "$out/d12_raytracing_core_tests"
"$out/d12_raytracing_core_tests"

"$cc" "$common[@]" -I"$d12_dir" \
    "$d12_dir/tests/d12_command_validation_core_tests.c" \
    -o "$out/d12_command_validation_core_tests"
"$out/d12_command_validation_core_tests"

"$cc" "$common[@]" -I"$d12_dir" \
    "$d12_dir/tests/d12_discard_core_tests.c" \
    -o "$out/d12_discard_core_tests"
"$out/d12_discard_core_tests"

"$cc" "$common[@]" -I"$d12_dir" \
    -I"$repo_root/include" -I"$repo_root/src" \
    "$repo_root/src/asahi/lib/agx_pbe_agx3_core.c" \
    "$repo_root/src/asahi/lib/agx_texture_side_entry.c" \
    "$d12_dir/tests/d12_native_descriptor_core_tests.c" \
    -o "$out/d12_native_descriptor_core_tests"
"$out/d12_native_descriptor_core_tests"

"$cc" "$common[@]" \
    "$d12_dir/d12_core_caps.c" "$d12_dir/tests/d12_core_caps_tests.c" \
    -o "$out/d12_core_caps_tests"
"$out/d12_core_caps_tests"

"$cc" "$common[@]" -I"$d12_dir" \
    "$d12_dir/tests/d12_format_caps_core_tests.c" \
    -o "$out/d12_format_caps_core_tests"
"$out/d12_format_caps_core_tests"

"$cc" "$common[@]" \
    -I"$repo_root/src" -I"$repo_root/include" \
    "$d12_dir/d12_device_removal_core.c" \
    "$d12_dir/tests/d12_device_removal_core_tests.c" \
    -pthread \
    -o "$out/d12_device_removal_core_tests"
"$out/d12_device_removal_core_tests"

"$cc" "$common[@]" \
    "$d12_dir/d12_present_blt_math.c" \
    "$d12_dir/tests/d12_present_blt_math_tests.c" \
    -o "$out/d12_present_blt_math_tests"
"$out/d12_present_blt_math_tests"

"$cc" "$common[@]" \
    "$d12_dir/d12_present_format_core.c" \
    "$d12_dir/tests/d12_present_format_core_tests.c" \
    -o "$out/d12_present_format_core_tests"
"$out/d12_present_format_core_tests"

"$cc" "$common[@]" \
    "$d12_dir/d12_query_core.c" \
    "$d12_dir/tests/d12_query_core_tests.c" \
    -o "$out/d12_query_core_tests"
"$out/d12_query_core_tests"

"$cc" "$common[@]" \
    "$d12_dir/d12_resolve_core.c" \
    "$d12_dir/tests/d12_resolve_core_tests.c" \
    -o "$out/d12_resolve_core_tests"
"$out/d12_resolve_core_tests"

"$cc" "$common[@]" \
    "$d12_dir/d12_texture_copy_math.c" \
    "$d12_dir/tests/d12_texture_copy_math_tests.c" \
    -o "$out/d12_texture_copy_math_tests"
"$out/d12_texture_copy_math_tests"

"$cc" "$common[@]" \
    "$d12_dir/d12_viewport_math.c" \
    "$d12_dir/tests/d12_viewport_math_tests.c" \
    -o "$out/d12_viewport_math_tests"
"$out/d12_viewport_math_tests"

"$cc" "$common[@]" \
    "$repo_root/src/poly/tests/poly_heap_math_tests.c" \
    -o "$out/poly_heap_math_tests"
"$out/poly_heap_math_tests"

"$cc" "$common[@]" \
    "$d12_dir/d12_read_set_core.c" \
    "$d12_dir/tests/d12_read_set_core_tests.c" \
    -o "$out/d12_read_set_core_tests"
"$out/d12_read_set_core_tests"

"$cc" "$common[@]" \
    "$d12_dir/d12_sparse_residency_core.c" \
    "$d12_dir/tests/d12_sparse_residency_core_tests.c" \
    -o "$out/d12_sparse_residency_core_tests"
"$out/d12_sparse_residency_core_tests"

"$cc" "$common[@]" \
    "$d12_dir/d12_write_set_core.c" \
    "$d12_dir/tests/d12_write_set_core_tests.c" \
    -o "$out/d12_write_set_core_tests"
"$out/d12_write_set_core_tests"

"$cc" "$common[@]" \
    "$d12_dir/d12_queue_timeline_core.c" \
    "$d12_dir/tests/d12_queue_timeline_core_tests.c" \
    -o "$out/d12_queue_timeline_core_tests"
"$out/d12_queue_timeline_core_tests"

"$cc" "$common[@]" \
    "$umd_dir/d12_umd_command_core.c" \
    "$umd_dir/tests/d12_umd_command_core_tests.c" \
    -o "$out/d12_umd_command_core_tests"
"$out/d12_umd_command_core_tests"

"$cc" "$common[@]" \
    "$umd_dir/d12_umd_lifetime_core.c" \
    "$umd_dir/tests/d12_umd_lifetime_core_tests.c" \
    -o "$out/d12_umd_lifetime_core_tests"
"$out/d12_umd_lifetime_core_tests"

"$cc" "$common[@]" \
    "$umd_dir/d12_umd_residency_core.c" \
    "$umd_dir/tests/d12_umd_residency_core_tests.c" \
    -o "$out/d12_umd_residency_core_tests"
"$out/d12_umd_residency_core_tests"

"$cc" "$common[@]" \
    -I"$d12_dir" -I"$umd_dir" \
    "$umd_dir/d12_umd_pipeline_serialization.c" \
    "$umd_dir/tests/d12_umd_pipeline_serialization_tests.c" \
    -o "$out/d12_umd_pipeline_serialization_tests"
"$out/d12_umd_pipeline_serialization_tests"

"$cc" "$common[@]" \
    -I"$d12_dir" -I"$umd_dir" \
    "$umd_dir/d12_umd_pipeline_serialization.c" \
    "$umd_dir/d12_umd_pipeline_library.c" \
    "$umd_dir/tests/d12_umd_pipeline_library_tests.c" \
    -o "$out/d12_umd_pipeline_library_tests"
"$out/d12_umd_pipeline_library_tests"

"$cc" "$common[@]" \
    -I"$contract_dir" -I"$d12_dir" -I"$umd_dir" \
    "$contract_dir/agxw_contract.c" \
    "$umd_dir/d12_umd_resource_private.c" \
    "$umd_dir/tests/d12_umd_resource_private_tests.c" \
    -o "$out/d12_umd_resource_private_tests"
"$out/d12_umd_resource_private_tests"

"$cc" "$common[@]" \
    -I"$repo_root/src/asahi/lib" -I"$contract_dir" \
    "$repo_root/src/asahi/lib/agx_d3dkmt_core.c" \
    "$contract_dir/agxw_contract.c" \
    "$repo_root/src/asahi/lib/agx_d3dkmt_core_tests.c" \
    -o "$out/agx_d3dkmt_core_tests"
"$out/agx_d3dkmt_core_tests"

python3 "$repo_root/src/asahi/genxml/gen_pack.py" \
    "$repo_root/src/asahi/genxml/cmdbuf.xml" \
    "$repo_root/src/asahi/genxml/agx_pack_header.h" \
    > "$out/agx_pack.h"

# agx_ppp.h uses the installed-style include path while the older standalone
# packing tests include the generated header directly.
mkdir -p "$out/asahi/genxml"
cp "$out/agx_pack.h" "$out/asahi/genxml/agx_pack.h"

"$cc" "$common[@]" \
    -Wno-c23-extensions -Wno-unused-parameter -Wno-initializer-overrides \
    -DHAVE_PTHREAD -DHAVE_STRUCT_TIMESPEC \
    -I"$out" -I"$repo_root/include" -I"$repo_root/src" \
    -I"$repo_root/src/asahi" -I"$repo_root/src/asahi/vulkan" \
    "$repo_root/src/asahi/lib/agx_pbe_agx3_core.c" \
    "$repo_root/src/asahi/vulkan/hk_native_descriptor_pack.c" \
    "$repo_root/src/asahi/vulkan/tests/test-hk-native-descriptor.c" \
    -o "$out/hk_native_descriptor_tests"
"$out/hk_native_descriptor_tests"

"$cc" "$common[@]" \
    -Wno-c23-extensions -Wno-unused-parameter \
    -Wno-gnu-statement-expression-from-macro-expansion -Wno-unused-function \
    -DHAVE_PTHREAD -DHAVE_STRUCT_TIMESPEC \
    -I"$out" -I"$repo_root/include" -I"$repo_root/src" \
    -I"$repo_root/src/asahi" \
    "$repo_root/src/asahi/genxml/tests/test-agx3-packing.c" \
    -o "$out/agx3_packing_tests"
"$out/agx3_packing_tests"

"$cc" "$common[@]" \
    -Wno-c23-extensions -Wno-unused-parameter \
    -DHAVE_PTHREAD -DHAVE_STRUCT_TIMESPEC \
    -I"$out" -I"$repo_root/include" -I"$repo_root/src" \
    -I"$repo_root/src/asahi" \
    "$repo_root/src/asahi/lib/agx3_cdm_decode_core.c" \
    "$repo_root/src/asahi/lib/agx3_cdm_direct_launch_semantic_core.c" \
    "$repo_root/src/asahi/lib/agx3_compute_program_core.c" \
    "$repo_root/src/asahi/lib/agx_ioto_core.c" \
    "$repo_root/src/asahi/lib/agx_ioto_preshader_core.c" \
    "$repo_root/src/asahi/lib/agx_ioto_profile_core.c" \
    "$repo_root/src/asahi/lib/agx_ioto_state_core.c" \
    "$repo_root/src/asahi/lib/agx_ioto_program_core.c" \
    "$repo_root/src/asahi/d3d12/tests/d12_agx3_cmdstream_core_tests.c" \
    -o "$out/d12_agx3_cmdstream_core_tests"
"$out/d12_agx3_cmdstream_core_tests"

"$cc" "$common[@]" \
    "$d12_dir/d12_dxbc_wrap.c" "$d12_dir/tests/d12_dxbc_wrap_tests.c" \
    -o "$out/d12_dxbc_wrap_tests"
"$out/d12_dxbc_wrap_tests" "$d12_dir/tests/shaders/cs.dxil"

python3 "$umd_dir/tests/check_warm_recording.py"
python3 "$umd_dir/tests/check_raster_state.py"
python3 "$umd_dir/tests/check_wave_ops.py"
python3 "$umd_dir/tests/check_exact_read_set.py"
python3 "$umd_dir/tests/check_sparse_residency.py"
python3 "$umd_dir/tests/check_resource_binding.py"
python3 "$umd_dir/tests/check_sampler_reduction.py"
python3 "$umd_dir/tests/check_feature_level_12.py"
python3 "$umd_dir/tests/check_tiler_features.py"
python3 "$umd_dir/tests/check_large_buffer_views.py"
python3 "$umd_dir/tests/check_dxbc_buffer_view_bounds.py"
python3 "$umd_dir/tests/check_ddi_submit_thread.py"
"$umd_dir/tests/test_build_mesa_windows_arm64.sh"
python3 "$umd_dir/packaging/test_stage_package.py"
python3 "$umd_dir/tests/test_check_pe.py"
