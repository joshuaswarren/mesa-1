#!/usr/bin/env python3
"""Static gate tying the advertised feature level to its implementation.

A feature level is a promise about behaviour, and the number is one line of
code.  These assertions keep each of the three requirements 12_0 adds over
11_1 attached to the code that actually satisfies it, so removing the
implementation breaks the build gate rather than the application.
"""

from pathlib import Path


d12 = Path(__file__).resolve().parents[2]
caps = (d12 / "d12_core_caps.c").read_text()
caps_h = (d12 / "d12_core_caps.h").read_text()
adapter = (d12 / "umd/d12_umd_adapter.c").read_text()
device = (d12 / "d12_device.c").read_text() + (d12 / "d12_core_caps_query.c").read_text()
resource = (d12 / "d12_resource.c").read_text()
sparse_core = (d12 / "d12_sparse_residency_core.h").read_text()
format_core = (d12 / "d12_format_caps_core.h").read_text()
fmt = (d12 / "d12_format.c").read_text()
dxil = (d12 / "dxil/d12_dxil_nir.c").read_text()
validation = (d12 / "d12_command_validation_core.h").read_text()

# The level itself, and the three requirements it adds over 11_1.
assert "caps->resource_binding_tier >= 2" in caps
assert "caps->resource_binding_tier = 1;" in caps
assert "caps->tiled_resources_tier = 0;" in caps
assert "caps->typed_uav_load_additional_formats = legacy;" in caps

# Both frontends must report one policy. The DDI speaks its own enumeration,
# so the translation is pinned to the WDK values by a static assertion.
assert "d12_core_pipeline_level(" in caps_h
assert adapter.count("d12_core_pipeline_level(") == 2
assert "D3D12DDI_3DPIPELINELEVEL_11_0;" not in adapter
assert "_Static_assert(D12_CORE_PIPELINE_LEVEL_11_0 ==" in adapter
assert "(D3D12_RESOURCE_BINDING_TIER)caps->resource_binding_tier" in device

# Resource Binding Tier 2: a shader reaches the whole heap through a rooted
# range rather than a per-register root slot, and an unwritten descriptor is
# a null descriptor rather than a fault.
assert "d12_binding_map_rooted_base(" in (d12 / "dxil/d12_dxil.h").read_text()
descriptor = (d12 / "d12_descriptor.c").read_text()
assert "heap->slots[i].address = device->null_va;" in descriptor

# Tiled Resources Tier 2, requirement by requirement.
#   nonpacked standard-tile levels
assert "d12_sparse_standard_tile_level_is_nonpacked" in sparse_core
assert "d12_sparse_standard_tile_el(block_B, &std_w_el, &std_h_el)" in resource
#   shader mapped-status and LOD clamp
assert "dxil_sparse_feedback_status" in dxil
assert "buffer_residency_status" in dxil
assert "texture_min_lod" in (d12 / "d12_cmd.c").read_text()
#   NULL-mapped tiles read as zero through the read-only alias
assert "sparse_ro_offset" in device

# Typed UAV loads: the published list, and a runtime check of the promise
# against the table that answers CheckFormatSupport.
assert "D12_FORMAT_CORE_TYPED_UAV_LOAD_ADDITIONAL_LIST" in format_core
assert "D12_FORMAT_CORE_TYPED_UAV_LOAD_ADDITIONAL_COUNT 15u" in format_core
assert "d12_format_typed_uav_load_complete" in fmt
assert "d12_format_typed_uav_load_complete(" in device

# Inherited from 11_1: target-independent rasterization is honoured where it
# is expressible and refused where it is not.
assert "d12_raster_forced_sample_count_supported" in validation
assert "d12_raster_rasterization_samples" in validation
assert "pso->raster.forced_sample_count" in (d12 / "d12_cmd.c").read_text()

# 12_1 is not claimed: it needs conservative rasterization and ROVs.
assert ".max_feature_level = D12_CORE_FEATURE_LEVEL_12_1," not in caps

# The one inherited requirement AGX cannot meet must stay recorded rather
# than quietly dropped, and the quality-level query must keep telling the
# truth about it.
assert "8x MSAA" in caps
runtime = (d12 / "d12_device_runtime.c").read_text()
assert "d12_sample_count_supported" in validation
assert "d12_sample_count_supported(sample_count)" in runtime
# An unencodable count must be refused at layout time too: it used to reach
# agx_translate_sample_count and abort the calling process.
assert resource.count("d12_sample_count_supported(") == 2
assert "samples != 1 && samples != 2 && samples != 4" not in resource

print(
    "feature-level derivation: PASS "
    "(binding tier 1 and tiled tier 0 withhold FL12; implementation routes retained)"
)
