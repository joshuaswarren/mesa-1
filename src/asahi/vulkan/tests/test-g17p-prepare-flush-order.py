#!/usr/bin/env python3
"""Keep G17P graphics prewarm behind authoritative shader-state assembly.

The transformed (tessellation/geometry) preparation path uploads native USC
programs before the draw.  It must first assign the active HW VS -> FS UVS
layout and populate the root exactly as the draw path does.  Otherwise legal
TES/GS varyings retain stale slot zero and vkEndCommandBuffer fails with
VK_ERROR_FEATURE_NOT_PRESENT.
"""

from pathlib import Path


source = (Path(__file__).resolve().parents[1] / "hk_cmd_draw.c").read_text()

prepare_start = source.index("\nhk_cmd_prepare_bound_graphics(")
prepare_end = source.index("\nstatic ", prepare_start)
prepare = source[prepare_start:prepare_end]

prelink = prepare.index("hk_prelink_graphics(")
flush = prepare.index("hk_flush_shaders(cmd)")
upload = prepare.index("hk_upload_usc_words(")
assert prelink < flush < upload, (
    "prewarm must link, then assemble shader/root state, then upload USC"
)
assert "vk_command_buffer_has_error(&cmd->vk)" in prepare[flush:upload]

tail_start = source.index("\nhk_g17p_prepare_native_bind_tail(", prepare_end)
tail_end = source.index("\nstatic ", tail_start)
tail = source[tail_start:tail_end]
assert "hk_flush_shaders(cmd)" not in tail, (
    "shader-state assembly belongs at the common preparation boundary"
)

print("PASS G17P prewarm orders shader/root assembly before every USC upload")
