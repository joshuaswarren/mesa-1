#!/usr/bin/env python3

from pathlib import Path

root = Path(__file__).resolve().parents[1]
buffer_source = (root / "hk_cmd_buffer.c").read_text()
draw_source = (root / "hk_cmd_draw.c").read_text()

prepare_start = buffer_source.index("\nhk_cmd_buffer_prepare_graphics_root(")
prepare_end = buffer_source.index("\nstatic ", prepare_start)
prepare = buffer_source[prepare_start:prepare_end]

assert "!cmd->state.gfx.descriptors.root_dirty && cmd->state.gfx.root" in prepare
assert prepare.index("!cmd->state.gfx.descriptors.root_dirty") < prepare.index(
    "hk_pool_root_alloc"
), "clean published roots must bypass allocation"

upload_start = buffer_source.index("\nhk_cmd_buffer_upload_root(")
upload_end = buffer_source.index("\nbool\n", upload_start)
upload = buffer_source[upload_start:upload_end]
for invalidation in (
    "prepared_vertex_usc = 0",
    "prepared_fragment_usc = 0",
    "g17p_bind_tail.valid = false",
):
    assert invalidation in upload, f"new root must invalidate {invalidation}"

tail_start = draw_source.index("\nhk_g17p_prepare_native_bind_tail(")
tail_end = draw_source.index("\n/*\n * Return the given sample positions", tail_start)
tail = draw_source[tail_start:tail_end]
assert "force_never_in_shader !=" in tail
assert "force_never_in_shader;\n      gfx->descriptors.root_dirty = true;" in tail

print("PASS G17P reuses clean immutable roots and invalidates stale prepared state")
