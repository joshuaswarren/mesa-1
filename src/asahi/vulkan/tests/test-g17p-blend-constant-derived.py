#!/usr/bin/env python3
"""The blend-constant uniform range must follow the current dynamic blend state.

dEQP-VK.pipeline.monolithic.blend.dual_source.multi_attachments.* (Neo,
2026-10-01): the fragment stage program built for the native bind tail at
vkCmdBindPipeline / vkCmdBeginRendering read a cached `uses_blend_constant`
that only the draw-time dirty flush refreshed. The first draw after a
blend-equation change therefore published no blend-constant uniform range and
its epilog read unwritten uniforms; every failing iteration used a constant
blend factor. The prepared tail is keyed by dynamic_graphics_state.generation,
so the flag must be derived from that state wherever it is read.
"""

import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
header = (root / "hk_cmd_buffer.h").read_text()
sources = {p.name: p.read_text() for p in root.glob("*.c")}
sources["hk_cmd_buffer.h"] = header

# No cached copy anywhere.
assert not re.search(r"\bbool\s+uses_blend_constant\s*;", header), \
    "cached uses_blend_constant field is back"
for name, text in sources.items():
    for bad in ("state.gfx.uses_blend_constant", "gfx->uses_blend_constant"):
        assert bad not in text, f"{name} reads the cached flag ({bad})"

# The helper derives it from the dynamic graphics state.
m = re.search(r"\nhk_cmd_uses_blend_constant\(const struct hk_cmd_buffer \*cmd\)\n\{(.*?)\n\}\n",
              header, re.S)
assert m, "hk_cmd_uses_blend_constant() missing"
body = m.group(1)
assert "cmd->vk.dynamic_graphics_state.cb" in body
for factor in ("src_color_blend_factor", "src_alpha_blend_factor",
               "dst_color_blend_factor", "dst_alpha_blend_factor"):
    assert factor in body, f"helper ignores {factor}"
assert "VK_BLEND_FACTOR_CONSTANT_COLOR" in body and \
       "VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA" in body

# Every consumer that shapes the fragment program uses the derived value:
# the uniform ranges, the compact-launcher identity, the legacy USC upload and
# the epilog relocation decision.
buf = sources["hk_cmd_buffer.c"]
assert buf.count("hk_cmd_uses_blend_constant(cmd)") >= 5
ranges = buf[buf.index("The non-monolithic blend epilog reads the four constants") - 400:]
assert "if (hk_cmd_uses_blend_constant(cmd)) {" in ranges[:600]
draw = sources["hk_cmd_draw.c"]
assert draw.count("hk_cmd_uses_blend_constant(cmd)") >= 2
assert not re.search(r"\nuses_blend_constant\(const struct vk_color_blend_state", draw)

print("PASS blend-constant range derived from the current dynamic blend state")
