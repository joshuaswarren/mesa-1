#!/usr/bin/env python3
"""Check strict32 clear-only and triangle recordings from the no-op DRM shim.

Usage: test-g17p-clear-only-recording.py CLEAR_LOG TRIANGLE_LOG CONTROL_LOG
The control is the frozen pre-fix triangle+marker recording. This checks
recording, not execution: no-op shim pixels and marker must stay untouched.
"""

import pathlib
import re
import struct
import sys


def check(clear, triangle, control):
    assert "G17P_NO_DRAW set, clear-only pass" in clear
    assert "G17P clear-only graphics: 32x32 samples=1 sample_control=0x88 " \
           "process_empty_tiles=1 scissor=16 zbias=12 direct_resolve=0" in clear
    assert "command buffer recorded" in clear and "g17p-render-only: submitted" in clear
    assert "returned -" not in clear
    assert "vertex-ioto" not in clear and "fragment-ioto" not in clear
    assert "G17P graphics shader:" not in clear
    assert "G17P cdm" not in clear and "compute-ioto" not in clear
    assert "G17P BG launch: op=clear" in clear
    assert "G17P BG launch: op=reload" in clear
    assert "G17P auxiliary-shader: kind=eot" in clear
    assert "main=90 block=256 allocation=1088 gpr_halfs=6 D=0" in clear

    stream = re.search(r"G17P dump: vdm-stream base=[0-9a-f]+ len=16\n"
                       r"G17P dump:   \+0000 ([0-9a-f ]+)\n", clear)
    assert stream, "clear-only stream must be one 16-byte record sequence"
    words = struct.unpack("<4I", bytes.fromhex(stream[1]))
    assert words[0] == 0x21000960  # existing initial VDM barrier
    assert words[1] == 0x300      # three-word initial PPP state
    assert words[2] != 0 and words[2] % 4 == 0
    assert words[3] == 0xC0000000 # independently packed stream terminate

    marker = re.search(r"G17P BG entry marker: image=([0-9a-f]+) "
                       r"marker=([0-9a-f]+) magic=42474531", clear)
    assert marker and int(marker[2], 16) == int(marker[1], 16) + 0x1000
    assert "marker after-fence-wait value=00000000 " \
           "guards=13579bdf/2468ace0/a55ac33c: INCONCLUSIVE" in clear
    assert "CLEAR-ONLY FAIL - 0/1024 pixels match clear" in clear
    assert "PASS" not in clear

    assert "G17P clear-only graphics:" not in triangle
    assert "G17P graphics shader: stage=0" in triangle
    assert "G17P graphics shader: stage=4" in triangle
    assert "len=132 linked=0" in triangle
    assert triangle == control, "all logged triangle inputs must be unchanged"
    # The EOT Ioto buffer is not logged by this trace. Its intentional
    # relocation/wait changes are tested by the separate Ioto core unit test.


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    check(*(pathlib.Path(p).read_text() for p in sys.argv[1:]))
    print("G17P clear-only recording: graphics-only stream and triangle control pass")
