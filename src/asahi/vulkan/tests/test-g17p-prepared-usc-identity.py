#!/usr/bin/env python3

from pathlib import Path

root = Path(__file__).resolve().parents[1]
header = (root / "hk_cmd_buffer.h").read_text()
draw = (root / "hk_cmd_draw.c").read_text()

for field in (
    "prepared_vertex_shader",
    "prepared_vertex_linked",
    "prepared_vertex_root",
    "prepared_vertex_compact_root",
    "prepared_vertex_draw_params",
    "prepared_vertex_draw_id_ptr",
):
    assert field in header, f"missing prepared vertex identity field: {field}"

prepare_start = draw.index("\nhk_cmd_prepare_bound_graphics(")
prepare_end = draw.index("\nstatic enum hk_g17p_draw_split", prepare_start)
prepare = draw[prepare_start:prepare_end]
for capture in (
    "gfx->prepared_vertex_shader = hw_vs",
    "gfx->prepared_vertex_linked = linked_hw_vs",
    "gfx->prepared_vertex_root = gfx->root",
    "gfx->prepared_vertex_draw_params = gfx->draw_params",
    "gfx->prepared_vertex_draw_id_ptr = gfx->draw_id_ptr",
):
    assert capture in prepare, f"prepared launcher does not capture {capture}"

flush_start = draw.index("\nhk_g17p_flush_prepared_dynamic_state(")
flush_end = draw.index("\nstatic bool\nhk_needs_index_robustness", flush_start)
flush = draw[flush_start:flush_end]
assert "bool prepared_vertex_exact" in flush
assert "hk_retarget_prepared_usc_root(" in flush
assert "gfx->prepared_vertex_compact_root = (struct agx_ptr){0}" in flush
for comparison in (
    "gfx->prepared_vertex_shader == hw_vs",
    "gfx->prepared_vertex_linked == linked_hw_vs",
    "gfx->prepared_vertex_root == gfx->root",
    "gfx->prepared_vertex_draw_params == gfx->draw_params",
    "gfx->prepared_vertex_draw_id_ptr == gfx->draw_id_ptr",
):
    assert comparison in flush, f"draw does not validate {comparison}"

assert "prepared_vertex_exact\n            ? gfx->prepared_vertex_usc" in flush

print("PASS prepared vertex launchers require exact shader and GPU-pointer identity")
