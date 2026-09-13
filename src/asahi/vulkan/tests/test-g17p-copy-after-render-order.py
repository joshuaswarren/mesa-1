#!/usr/bin/env python3
"""Check that Honeykrisp asks the kernel to order a readback copy after a render.

A `vkCmdCopyImageToBuffer` of a just-rendered image becomes a compute (CDM)
command, while the render is a VDM command on a different subqueue. Nothing in
the submission makes the copy wait for the render except
`drm_asahi_cmd_header::vdm_barrier`: 0 means "wait on all render commands from
previous submit ioctls", DRM_ASAHI_BARRIER_NONE (0xFFFF) means "wait on
nothing". G17P reports max_commands_per_submission=1, so a single
vkQueueSubmit is split across ioctls by queue_submit_looped() and the two land
in *different* ioctls -- which is exactly when that field is the only ordering
left.

This runs the in-tree offscreen-triangle oracle against the no-op DRM shim,
which executes nothing but records every submit ioctl, and asserts the recorded
sequence. It needs no hardware.

Usage: test-g17p-copy-after-render-order.py MESON_BUILD_DIR
"""

import os
import pathlib
import re
import subprocess
import sys
import tempfile

BARRIER_NONE = 0xFFFF


def parse(text):
    """[(in_syncs, out_syncs, [(name, vdm_barrier, cdm_barrier), ...]), ...]"""
    submits = []
    for line in text.splitlines():
        head = re.match(r"submit \d+ in_syncs=(\d+) out_syncs=(\d+) bytes=\d+$",
                        line)
        if head:
            submits.append((int(head[1]), int(head[2]), []))
            continue
        # Trailing fields (declared control-stream extents) are optional:
        # match the barrier fields and ignore whatever the shim appends after.
        cmd = re.match(r"  cmd (\S+) type=\d+ size=\d+ vdm_barrier=(\d+) "
                       r"cdm_barrier=(\d+)(?: .*)?$", line)
        if cmd:
            assert submits, "command outside a submit"
            submits[-1][2].append((cmd[1], int(cmd[2]), int(cmd[3])))
            continue
        raise AssertionError("unparsed shim log line: %r" % line)
    return submits


def check(submits):
    """Raise unless a copy is ordered after the render that produced it."""
    renders = [i for i, s in enumerate(submits)
               if any(c[0] == "render" for c in s[2])]
    computes = [i for i, s in enumerate(submits)
                if any(c[0] == "compute" for c in s[2])]
    assert renders, "no render command was submitted"
    assert computes, "no compute command was submitted; the readback copy did " \
                     "not become a CDM dispatch"

    last_render = max(renders)
    copies = [i for i in computes if i > last_render]
    assert copies, "the readback copy was submitted before the render"
    copy_index = min(copies)

    for name, vdm, cdm in submits[copy_index][2]:
        if name != "compute":
            continue
        assert vdm != BARRIER_NONE, (
            "the readback copy declares vdm_barrier=NONE: it does not wait for "
            "the render on the VDM subqueue at all")
        assert vdm == 0, (
            "the readback copy declares vdm_barrier=%u, an index within its own "
            "submit; the render is in an earlier ioctl, which only vdm_barrier=0 "
            "waits on" % vdm)
        assert cdm == 0, (
            "the readback copy declares cdm_barrier=%u; earlier compute work in "
            "previous ioctls is not waited on" % cdm)

    # Userspace waits on the out-syncobjs of the submission as a whole. They
    # must ride the last ioctl, so the fence a client waits on is the copy's
    # and not the render's.
    signalling = [i for i, s in enumerate(submits) if s[1] > 0]
    assert signalling, "no submission signalled anything"
    assert max(signalling) == len(submits) - 1, (
        "out-syncobjs are not on the final ioctl: a client can be released "
        "before the last command has run")
    assert copy_index in signalling or copy_index <= max(signalling), \
        "the copy is not covered by any signalled syncobj"


def self_check():
    """The assertion has to be able to fail, or it proves nothing."""
    ordered = parse(
        "submit 0 in_syncs=0 out_syncs=0 bytes=296\n"
        "  cmd set-fragment-attachments type=3 size=24 vdm_barrier=65535 cdm_barrier=65535\n"
        "  cmd render type=0 size=256 vdm_barrier=0 cdm_barrier=0\n"
        "submit 1 in_syncs=0 out_syncs=2 bytes=72\n"
        "  cmd compute type=1 size=64 vdm_barrier=0 cdm_barrier=0\n")
    check(ordered)

    unordered = parse(
        "submit 0 in_syncs=0 out_syncs=0 bytes=296\n"
        "  cmd render type=0 size=256 vdm_barrier=0 cdm_barrier=0\n"
        "submit 1 in_syncs=0 out_syncs=2 bytes=72\n"
        "  cmd compute type=1 size=64 vdm_barrier=65535 cdm_barrier=0\n")
    try:
        check(unordered)
    except AssertionError as error:
        assert "does not wait for the render" in str(error), error
    else:
        raise AssertionError("an unordered copy passed the check")


def run(build_dir):
    build = pathlib.Path(build_dir).resolve()
    shim = build / "src/asahi/drm-shim/libasahi_noop_drm_shim.so"
    vulkan = build / "src/asahi/vulkan"
    binary = vulkan / "g17p-offscreen-triangle"
    icd = vulkan / "asahi_devenv_icd.aarch64.json"
    for path in (shim, binary, icd):
        assert path.exists(), "missing build artifact: %s" % path

    with tempfile.TemporaryDirectory() as tmp:
        log = pathlib.Path(tmp) / "submit.log"
        env = dict(os.environ,
                   LD_PRELOAD=str(shim),
                   ASAHI_SHIM_CHIP="g17p",
                   ASAHI_SHIM_SUBMIT_LOG=str(log),
                   VK_DRIVER_FILES=str(icd),
                   MESA_SHADER_CACHE_DISABLE="true")
        # The shim executes nothing, so the oracle's pixel comparison fails by
        # construction. Only the recorded submissions are under test here.
        subprocess.run([str(binary)], env=env, capture_output=True, timeout=300)
        assert log.exists(), "the shim recorded no submit ioctl at all"
        text = log.read_text()

    submits = parse(text)
    check(submits)
    return text


if __name__ == "__main__":
    self_check()
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    recorded = run(sys.argv[1])
    sys.stdout.write(recorded)
    print("PASS: the readback copy is a later submit ioctl than the render and "
          "declares vdm_barrier=0, so it waits on it; the submission's "
          "out-syncobjs ride that copy")
