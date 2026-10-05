#!/usr/bin/env python3
"""Static gate for D3D12 buffer-view limits and DXBC typed stores."""

from pathlib import Path


d12 = Path(__file__).resolve().parents[2]
core = (d12 / "d12_buffer_view_core.h").read_text()
cmd = (d12 / "d12_cmd.c").read_text()
device = (d12 / "d12_device.c").read_text()
dxbc = (d12 / "dxil/d12_dxbc.c").read_text()


assert "#define D12_TYPED_BUFFER_VIEW_MAX_ELEMENTS (UINT32_C(1) << 27)" in core
fill_start = device.index("d12_fill_buffer_descriptor(")
fill_end = device.index("d12_device_CreateShaderResourceView(", fill_start)
fill = device[fill_start:fill_end]
assert "typed && num_elements > D12_TYPED_BUFFER_VIEW_MAX_ELEMENTS" in fill
assert "d12_buffer_view_range(" in fill
assert "d12_buffer_view_element_count_valid" not in fill

store_start = dxbc.index("case D12_DXBC_OP_STORE_RAW:")
store_end = dxbc.index("case D12_DXBC_OP_ATOMIC_AND:", store_start)
store = dxbc[store_start:store_end]
assert "nir_ugt_imm(b, count, AGX_TEXTURE_BUFFER_MAX_SIZE)" not in store
assert "D12_UNIFORM_HEAP_PBES" in store
assert "rooted_native_index(ctx, descriptor)" in store
assert "D12_UNIFORM_PBES" in store
assert "uav_binding_slot(ctx, &o[0])" in store
typed_store = store[store.index("nir_def *count = buffer_view_count_or_bytes") :]
assert "nir_ult(b, element_or_byte, count)" in typed_store
assert typed_store.index("nir_ult(b, element_or_byte, count)") < typed_store.index(
    "nir_bindless_image_store("
)
assert "robust_store_typed_buffer(" not in store

for dead_tail_token in (
    "d12_buffer_tail_descriptor",
    "2 * (size_t)last_texture",
    "(uint64_t)last_texture << 48",
    "last_texture + slot",
):
    assert dead_tail_token not in cmd

assert "d12_dxbc_buffer_native_slot" not in dxbc
assert "d12_dxbc_buffer_native_coord" not in dxbc
assert "AGX_TEXTURE_BUFFER_MAX_SIZE" not in dxbc

print(
    "dxbc-buffer-view-bounds: PASS "
    "(typed extent + full-width byte bounds + native heaps + guarded typed PBE stores)"
)
