#!/usr/bin/env python3

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "hk_cmd_draw.c").read_text()
header = (root / "hk_cmd_buffer.h").read_text()

assert "bool includes_gs;" in header
assert "bool gs_emitted;" in header

prepare_start = source.index("\nhk_cmd_prepare_bound_graphics(")
prepare_end = source.index("\nstatic enum hk_g17p_draw_split", prepare_start)
prepare = source[prepare_start:prepare_end]
for witness in (
    "g17p_precomp.gs_setup.valid",
    "g17p_dispatch.gs_vs.indirect_local_valid",
    "g17p_dispatch.gs_main.indirect_local_valid",
):
    assert witness in prepare, f"preparation omits {witness}"

emit_start = source.index("\nhk_g17p_emit_prepared_sw_tess_batch(")
emit_end = source.index("\nstatic __attribute__((hot)) struct agx_draw\nhk_launch_tess", emit_start)
emit = source[emit_start:emit_end]
for check in (
    "batch->dynamic_generation !=",
    "batch->descriptor_generation != gfx->descriptors.generation",
    "batch->root != gfx->root",
    "batch->draw_params != gfx->draw_params",
    "batch->draw_id_ptr != gfx->draw_id_ptr",
    "batch->state != state",
    "batch->c_prims != c_prims",
    "batch->c_invs != c_inv",
    "batch->gs_vs == gs_vs",
    "batch->gs_main == gs_main",
    "2 * AGX3_G17P_NATIVE_INDIRECT_COMPUTE_DISPATCH_SIZE",
    "hk_g17p_materialize_prepared_indirect(pgs_vs",
    "hk_g17p_materialize_prepared_indirect(pgs_main",
    "pgs_setup->direct_template",
    "pgs_vs->indirect_local_template",
    "pgs_main->indirect_local_template",
    "gfx->g17p_sw_tess_batch.gs_emitted = true",
):
    assert check in emit, f"combined tess/GS transaction omits {check}"

assert emit.index("hk_ensure_cs_has_space(") < emit.index(
    "hk_g17p_materialize_prepared_indirect(pgs_vs"
), "all stream storage must be reserved before single-use helper roots mutate"
assert emit.index("pgs_setup->direct_template") < emit.index(
    "pgs_vs->indirect_local_template"
) < emit.index("pgs_main->indirect_local_template")

hot_start = source.index("\nhk_g17p_launch_prepared_sw_tess_gs(")
hot_end = source.index("\nstatic __attribute__((hot)) struct agx_draw\nhk_launch_tess(", hot_start)
hot = source[hot_start:hot_end]
for witness in (
    "agx_is_indirect(draw.b)",
    "!gfx->g17p_sw_tess_batch.includes_gs",
    "VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT",
    "patches * tcs->info.tess.tcs_output_patch_size",
    "hk_g17p_emit_prepared_sw_tess_batch(",
):
    assert witness in hot, f"hot prepared tess/GS consume omits {witness}"

gs_start = source.index("\nhk_launch_gs_prerast(")
gs_end = source.index("\n/*\n * G17P tessellates in hardware", gs_start)
gs = source[gs_start:gs_end]
assert "if (gfx->g17p_sw_tess_batch.gs_emitted)" in gs
assert "gfx->g17p_sw_tess_batch.gs_emitted = false" in gs
assert "goto gs_result" in gs

draw_start = source.index("\nhk_draw_impl(")
draw = source[draw_start:]
assert "hk_g17p_launch_prepared_sw_tess_gs(cmd, ccs, draw)" in draw
assert "if (tess && !prepared_tess_gs_emitted)" in draw
assert "if (geom && !prepared_tess_gs_emitted)" in draw

print("PASS G17P software tessellation and ordinary GS publish one exact transaction")
