#!/usr/bin/env python3
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
"""Append a d3d12-stats record (see reports/d3d12-stats/README.md).

record-stats.py --task t-1084 --machine neo --driver 'd3d12agx@sha' --corpus frozen17 \
    --passed 8 --failed 9 --total 17 --metric fps_median=60 --evidence /path --notes '...'
"""
import argparse, datetime, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.environ.get("D12_STATS_DIR"),
                help="stats directory (default: $D12_STATS_DIR)")
ap.add_argument("--task", required=True)
ap.add_argument("--machine", required=True)
ap.add_argument("--driver", required=True)
ap.add_argument("--corpus", required=True)
for k in ("passed", "failed", "crashed", "skipped", "total"):
    ap.add_argument("--" + k, type=int, default=0)
ap.add_argument("--metric", action="append", default=[])
ap.add_argument("--evidence", action="append", default=[])
ap.add_argument("--notes", default="")
ap.add_argument("--short", required=True, help="short slug for the file name")
a = ap.parse_args()
if not a.dir:
    ap.error("--dir or D12_STATS_DIR is required")
now = datetime.datetime.now(datetime.timezone.utc)
metrics = {}
for m in a.metric:
    k, v = m.split("=", 1)
    try:
        metrics[k] = float(v)
    except ValueError:
        metrics[k] = v
rec = {"task": a.task, "time_utc": now.isoformat(timespec="seconds"), "machine": a.machine,
       "driver": a.driver, "corpus": a.corpus, "passed": a.passed, "failed": a.failed,
       "crashed": a.crashed, "skipped": a.skipped, "total": a.total, "metrics": metrics,
       "evidence": a.evidence, "notes": a.notes}
path = os.path.join(a.dir, f"{a.task}-{now.strftime('%Y%m%dT%H%M')}-{a.short}.json")
with open(path, "w") as f:
    json.dump([rec], f, indent=1)
print(path)
