#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT

"""Bounded inventory of fail-closed AGX3 encoder refusals.

Physical register numbers vary with surrounding shader pressure, so the
canonical operand shape retains widths, immediates, modifiers, conditions and
channel counts while replacing register/uniform indices with '#'.
"""

import argparse
import collections
import glob
import json
import pathlib
import re
import sys


REFUSAL = re.compile(
    r"no grounded AGX3 encoding for ([A-Za-z0-9_]+) \[([^\]\r\n]+)\]"
)
PHYSICAL_INDEX = re.compile(r"\b(reg|uniform)([0-9]+)\b")


def canonical(opcode, operands):
    return f"{opcode} [{PHYSICAL_INDEX.sub(lambda m: m.group(1) + '#', operands)}]"


def parse_lines(lines, source, records):
    for line_number, line in enumerate(lines, 1):
        for match in REFUSAL.finditer(line):
            raw = f"{match.group(1)} [{match.group(2)}]"
            shape = canonical(match.group(1), match.group(2))
            if shape not in records:
                records[shape] = {
                    "shape": shape,
                    "count": 0,
                    "first": f"{source}:{line_number}",
                    "first_raw": raw,
                }
            records[shape]["count"] += 1


def input_files(arguments, max_files):
    files = []
    for argument in arguments:
        if argument == "-":
            files.append(None)
            continue

        path = pathlib.Path(argument)
        matches = [path] if path.exists() else map(pathlib.Path, glob.glob(argument))
        for match in matches:
            if match.is_dir():
                files.extend(sorted(match.rglob("*.log")))
            elif match.is_file():
                files.append(match)

    unique = []
    seen = set()
    for path in files:
        key = None if path is None else path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
        if len(unique) > max_files:
            raise ValueError(f"input exceeds --max-files={max_files}")
    return unique


def run_self_test():
    records = collections.OrderedDict()
    parse_lines(
        [
            "x: no grounded AGX3 encoding for texture_load "
            "[d32:reg0 s0=32:reg140.discard s1=16:imm0]",
            "x: no grounded AGX3 encoding for texture_load "
            "[d32:reg8 s0=32:reg126.discard s1=16:imm0]",
            "x: no grounded AGX3 encoding for if_fcmp "
            "[d0:none0 s0=16:imm48 s1=32:reg2 cond=1 chan=0]",
        ],
        "fixture",
        records,
    )
    values = list(records.values())
    assert len(values) == 2
    assert values[0]["count"] == 2
    assert "s0=32:reg#.discard" in values[0]["shape"]
    assert "imm48" in values[1]["shape"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="log files/directories/globs, or -")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0
    if not args.paths:
        parser.error("provide at least one path or - for stdin")
    if args.max_files < 1 or args.max_bytes < 1:
        parser.error("bounds must be positive")

    records = collections.OrderedDict()
    consumed = 0
    try:
        files = input_files(args.paths, args.max_files)
    except ValueError as error:
        parser.error(str(error))

    for path in files:
        if path is None:
            parse_lines(sys.stdin, "stdin", records)
            continue
        size = path.stat().st_size
        consumed += size
        if consumed > args.max_bytes:
            parser.error(f"input exceeds --max-bytes={args.max_bytes}")
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            parse_lines(stream, str(path), records)

    ordered = sorted(records.values(), key=lambda item: (-item["count"], item["shape"]))
    if args.json:
        json.dump(ordered, sys.stdout, indent=2)
        print()
    else:
        for item in ordered:
            print(f"{item['count']}\t{item['shape']}\t{item['first']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
