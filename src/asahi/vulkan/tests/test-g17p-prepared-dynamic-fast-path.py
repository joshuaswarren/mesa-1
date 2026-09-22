#!/usr/bin/env python3
"""Guard the exact unpublished HAL200 draw-state consume path."""

from pathlib import Path


vulkan = Path(__file__).resolve().parents[1]
source = (vulkan / "hk_cmd_draw.c").read_text()
header = (vulkan / "hk_cmd_buffer.h").read_text()
start = source.index("\nhk_g17p_flush_prepared_dynamic_state(")
end = source.index("\n/*\n * Does this draw republish render state", start)
flush = source[start:end]

for witness in (
    "!desc->root_dirty",
    "gfx->prepared_vertex_usc",
    "hk_prelinked_vs_matches(cmd, sw_vs, hw_vs, draw)",
    "hk_g17p_prepared_bind_tail_matches(cmd, &gfx->g17p_bind_tail)",
):
    assert witness in flush, witness

# Vertex-buffer binds are outside vk_dynamic_graphics_state.  The prepared
# root must carry a separate identity or a bind after BeginRendering can reuse
# stale attrib addresses while every old "exact" check still passes.
assert "uint64_t vertex_buffer_generation;" in header
assert (
    "prepared->vertex_buffer_generation = gfx->vertex_buffer_generation;"
    in source
)
assert "prepared->vertex_buffer_generation ==" in source
assert "cmd->state.gfx.vertex_buffer_generation++" in source

# The builder stores MAX2(count, 1), so match that exact canonical value.
assert (
    "prepared->viewport_count == MAX2(dyn->vp.viewport_count, 1)" in source
)

# Admission observes the prebuilt linkage, but the full path may replace it.
# Its consumers must reacquire gfx->linked[] after hk_update_fast_linked().
slow_start = source.index("\nhk_flush_dynamic_state_slow(")
slow_end = source.index("\nstatic __attribute__((hot)) void\nhk_flush_dynamic_state(", slow_start)
slow = source[slow_start:slow_end]
assert "prepared_linked_vs" in slow and "prepared_linked_fs" in slow
post_link_vs = slow.index(
    "struct hk_linked_shader *linked_vs = gfx->linked[MESA_SHADER_VERTEX];"
)
post_link_fs = slow.index(
    "struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];"
)
assert post_link_vs > slow.index("g17p_draw_parameters:")
assert post_link_fs > slow.index("g17p_draw_parameters:")
assert post_link_vs > slow.index("hk_update_fast_linked(cmd, fs, &key)")

assert "gfx->draw_params = draw.b.ptr" in flush
assert "VkDrawIndexedIndirectCommand, vertexOffset" in flush
assert "VkDrawIndirectCommand, firstVertex" in flush
assert flush.index("gfx->draw_params = draw.b.ptr") < flush.index(
    "hk_retarget_prepared_usc_root(")

wrapper_start = source.index("\nhk_flush_dynamic_state(")
wrapper_end = source.index("\nstatic bool\nhk_needs_index_robustness", wrapper_start)
wrapper = source[wrapper_start:wrapper_end]
assert "hk_g17p_flush_prepared_dynamic_state(" in wrapper
assert wrapper.index("hk_g17p_flush_prepared_dynamic_state(") < wrapper.index(
    "hk_flush_dynamic_state_slow(")

print("PASS exact prepared HAL200 state skips immutable rebuild only")
