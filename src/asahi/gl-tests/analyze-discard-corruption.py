#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Map the offscreen discard fixture's failures using only the standard library.

Run with a directory of fixture logs/RGBA files and --output DIRECTORY. The
oracle matches test-zink-discard-corruption.c: red/green/alpha are exact; the
survivor's blue channel comes from GPU sin arithmetic. Retry reads are excluded
from the primary failure rate. Filenames retain each process, mode and frame.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct
import zlib

WIDTH = HEIGHT = 512
CHECK = re.compile(r"depth=(\w+)(?: run=(\d+))? bad_pixels=(\d+)")
GPU = re.compile(
    r"depth=(\w+) run=(\d+) gpu_elapsed_ms=([0-9.]+) disjoint=(\d+)"
)


def chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data +
            struct.pack(">I", zlib.crc32(kind + data)))


def write_png(path, mask):
    rows = b"".join(b"\0" + mask[y * WIDTH:(y + 1) * WIDTH]
                    for y in range(HEIGHT))
    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
                     chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def map_frame(image, group, mode, reported_bad, output):
    data = image.read_bytes()
    if len(data) != WIDTH * HEIGHT * 4:
        raise ValueError(f"{image}: expected {WIDTH * HEIGHT * 4} bytes")
    mask = bytearray(WIDTH * HEIGHT)
    coordinates = []
    colours = collections.Counter()
    tiles = collections.Counter()
    full = group.startswith("solid")
    depth_off = group.startswith("original-nodepth")
    for index in range(WIDTH * HEIGHT):
        rgba = tuple(data[4 * index:4 * index + 4])
        x, y = index % WIDTH, index // WIDTH
        survivor = (mode == "always" or depth_off) and (full or x >= WIDTH // 2)
        wrong = (rgba[0] != (0 if survivor else 255) or
                 rgba[1] != (255 if survivor else 0) or rgba[3] != 255 or
                 (not survivor and rgba[2] != 0))
        if wrong:
            mask[index] = 255
            coordinates.append((x, y))
            colours[str(rgba)] += 1
            tiles[f"{x // 32},{y // 32}"] += 1
    if len(coordinates) != reported_bad:
        raise ValueError(f"{image}: mapped {len(coordinates)}, log says {reported_bad}")
    write_png(output / (image.stem + "-bad.png"), mask)
    xs, ys = zip(*coordinates)
    return {
        "image": str(image), "sha256": hashlib.sha256(data).hexdigest(),
        "bad_pixels": reported_bad, "bbox": [min(xs), min(ys), max(xs), max(ys)],
        "colors": dict(colours), "tiles": dict(tiles),
        "row_mod_32": dict(collections.Counter(y % 32 for y in ys)),
        "col_mod_32": dict(collections.Counter(x % 32 for x in xs)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"root": str(args.root), "groups": {}, "bad_frames": []}
    for log in sorted(args.root.glob("*.log")):
        group = log.stem.rsplit("-", 1)[0]
        totals = summary["groups"].setdefault(group, {
            "processes": 0, "bad_processes": 0, "modes": {},
        })
        totals["processes"] += 1
        checks = CHECK.findall(log.read_text())
        timings = GPU.findall(log.read_text())
        totals["bad_processes"] += any(int(n) > 0 for mode, run, n in checks)
        for mode, run, elapsed, disjoint in timings:
            values = totals["modes"].setdefault(mode, {
                "checks": 0, "bad_checks": 0, "bad_counts": [],
            })
            values.setdefault("gpu_steady_ms", [])
            values.setdefault("gpu_disjoint", 0)
            values["gpu_disjoint"] += int(disjoint) != 0
            if int(run) != 0:
                values["gpu_steady_ms"].append(float(elapsed))
        for mode, run, count in checks:
            bad = int(count)
            counts = totals["modes"].setdefault(mode, {
                "checks": 0, "bad_checks": 0, "bad_counts": [],
            })
            counts["checks"] += 1
            counts["bad_checks"] += bad > 0
            if not bad:
                continue
            counts["bad_counts"].append(bad)
            frame = f"-{int(run):04d}" if run else ""
            image = args.root / f"{log.stem}-{mode}{frame}.rgba"
            if image.is_file():
                summary["bad_frames"].append(
                    map_frame(image, group, mode, bad, args.output))
    for totals in summary["groups"].values():
        for values in totals["modes"].values():
            samples = values.get("gpu_steady_ms", [])
            values["gpu_median_ms"] = statistics.median(samples) if samples else None
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["groups"], indent=2))
    print("mapped_bad_frames", len(summary["bad_frames"]))


if __name__ == "__main__":
    main()
