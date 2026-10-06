#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright 2026 Ryan Murray <ryan@aurorasilicon.org>
"""Summarize completed HK_PASS_PROFILE windows, normalized by frame count."""

import argparse
import json
import re
from pathlib import Path


HEADER = re.compile(r"(\w+)=([-\d.]+)")
ROW = re.compile(
    r"\s*([\d.]+) ms busy\s+([\d.]+) ms span\s+(\d+) x\s+"
    r"(\d+) draws\s+(.*)"
)


def windows(lines):
    current = None
    for line in lines:
        if line.startswith("== window"):
            # Yield only after collecting the rows. Yielding at the header
            # lets the consumer see an empty list for every window.
            if current is not None:
                yield current
            current = {"header": dict(HEADER.findall(line)), "rows": []}
        elif current is not None and not line.startswith("#"):
            match = ROW.fullmatch(line.rstrip("\n"))
            if match:
                busy, span, count, draws, key = match.groups()
                current["rows"].append(
                    (key, float(busy), float(span), int(count), int(draws))
                )
    if current is not None:
        yield current


def summarize(lines, *, start=None, end=None, frames=None, frame_key=None):
    if (frames is None) == (frame_key is None):
        raise ValueError("specify either frames or an exact once-per-frame key")
    totals = {}
    seconds = busy_ms = 0.0
    selected = dropped = overflowed = 0
    for window in windows(lines):
        header = window["header"]
        t, duration = float(header["t"]), float(header["s"])
        if (start is not None and t < start) or (
            end is not None and t + duration > end
        ):
            continue
        selected += 1
        seconds += duration
        busy_ms += float(header["gpu_busy_ms"])
        dropped += int(header.get("dropped", 0))
        overflowed += int(header.get("overflowed", 0))
        for key, busy, span, count, draws in window["rows"]:
            total = totals.setdefault(key, [0.0, 0.0, 0, 0])
            for i, value in enumerate((busy, span, count, draws)):
                total[i] += value
    if not selected or seconds <= 0:
        raise ValueError("no complete windows in the selected interval")
    if frame_key is not None:
        if frame_key not in totals:
            raise ValueError("frame key absent from selected windows")
        frames = totals[frame_key][2]
    if frames <= 0:
        raise ValueError("frame count must be positive")
    return {
        "windows": selected,
        "seconds": seconds,
        "frames": frames,
        "fps": frames / seconds,
        "gpu_busy_percent": 100 * busy_ms / (seconds * 1000),
        "gpu_busy_ms_per_frame": busy_ms / frames,
        "dropped": dropped,
        "overflowed": overflowed,
        "passes": [
            {
                "key": key,
                "busy_ms_per_frame": value[0] / frames,
                "span_ms_per_frame": value[1] / frames,
                "commands_per_frame": value[2] / frames,
                "draws_per_frame": value[3] / frames,
                "busy_ms_per_command": value[0] / value[2] if value[2] else 0,
            }
            for key, value in sorted(totals.items(), key=lambda item: -item[1][0])
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--window", type=Path, help="JSON with mono0 and mono1")
    normalization = parser.add_mutually_exclusive_group(required=True)
    normalization.add_argument("--frames", type=float)
    normalization.add_argument(
        "--frame-key", help="exact pass key known to occur once per frame"
    )
    args = parser.parse_args()
    interval = json.loads(args.window.read_text()) if args.window else {}
    try:
        with args.profile.open() as stream:
            result = summarize(
                stream, start=interval.get("mono0"), end=interval.get("mono1"),
                frames=args.frames, frame_key=args.frame_key,
            )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
