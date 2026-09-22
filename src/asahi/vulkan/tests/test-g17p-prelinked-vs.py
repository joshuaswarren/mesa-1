#!/usr/bin/env python3

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "hk_cmd_draw.c").read_text()
header = (root / "hk_cmd_buffer.h").read_text()

assert "prepared_fast_link_vs" in header
for exact_input in (
    "sw_vs",
    "hw_vs",
    "dynamic_generation",
    "index_size",
    "indexed",
    "restart",
):
    assert exact_input in header

prelink_start = source.index("\nhk_prelink_graphics(", source.index("\nhk_fill_fast_link_key_fs("))
prelink_end = source.index("\nstatic inline bool\nhk_prelinked_vs_matches", prelink_start)
prelink = source[prelink_start:prelink_end]
assert prelink.index("hk_update_fast_linked") < prelink.index(
    "prepared_fast_link_vs ="
), "only a successful authoritative prelink may publish the witness"

match_start = prelink_end
match_end = source.index("\n/*\n * Does this draw republish", match_start)
match = source[match_start:match_end]
for exact_check in (
    "prepared->sw_vs == sw_vs",
    "prepared->hw_vs == hw_vs",
    "dynamic_graphics_state.generation",
    "prepared->indexed == draw.indexed",
    "prepared->index_size == draw.index_size",
    "prepared->restart == draw.restart",
):
    assert exact_check in match

draw_guard = source[source.index("if ((IS_DIRTY(VI)") :]
draw_guard = draw_guard[: draw_guard.index("if (perf_trace)")]
assert "!hk_prelinked_vs_matches(cmd, sw_vs, hw_vs, draw)" in draw_guard

print("PASS G17P draw consumes only an exact successful VS prelink")
