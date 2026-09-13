#!/usr/bin/env python3

from pathlib import Path
import sys


def function_body(text: str, name: str) -> str:
    start = text.index(name)
    end = text.index("\n}", start) + 2
    return text[start:end]


root = Path(sys.argv[1])
drm = function_body(
    (root / "drivers/dri2/platform_drm.c").read_text(),
    "dri2_drm_query_buffer_age(",
)
wayland = function_body(
    (root / "drivers/dri2/platform_wayland.c").read_text(),
    "dri2_wl_kopper_query_buffer_age(",
)

# Returning zero must not bypass the operations that acquire the drawable's
# back image.  These ordering assertions intentionally pin that containment.
assert drm.index("get_back_bo(") < drm.index("force_zero_buffer_age")
assert wayland.index("kopperQueryBufferAge(") < wayland.index(
    "force_zero_buffer_age"
)

setup = function_body(
    (root / "drivers/dri2/egl_dri2.c").read_text(), "dri2_setup_screen("
)
assert "dri2_force_zero_buffer_age_for_renderer" in setup

print("EGL DRI2 buffer-age acquisition ordering: passed")
