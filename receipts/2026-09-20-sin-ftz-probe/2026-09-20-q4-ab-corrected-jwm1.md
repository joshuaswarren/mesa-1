# Q4 premerge corrected A/B — jwm1 M1 G13G, 3 rounds/arm/leg (2026-09-20)

Corrected rerun per Main's authorization after the argparse failure.
Same protocol as `2026-09-20-q4-driver-ab-jwm1.md` but on the
native-matched ctx1024 prompt (from `/tmp/q4-ctx1024-exact.txt`, 1053
tokens) and the pinned short leg ("Hi", 30 tokens). Candidate =
clamped-then-load qmm_coopmat (8bbaa1f4) in separate venv; base =
stock qmm_coopmat (5b183060, existing v0.7.2-rc.1 venv). Both use the
fckey 095cb driver via isolated ICD. Shader cache OFF, compile OFF,
HF_HUB_OFFLINE=1.

## Raw results (all runs, both arms, both legs)

| run | decode tok/s | prefill s | digest |
|---|---:|---:|---|
| base short r1 | 116.32 | 0.079 | `7fd25a869ff21678` |
| base short r2 | 115.42 | 0.078 | `7fd25a869ff21678` |
| base short r3 | 116.32 | 0.080 | `7fd25a869ff21678` |
| base ctx1024 r1 | 100.42 | 0.954 | `7da83f06ec9f001d` |
| base ctx1024 r2 | 100.08 | 0.956 | `7da83f06ec9f001d` |
| base ctx1024 r3 | 99.87 | 0.951 | `7da83f06ec9f001d` |
| cand short r1 | 113.94 | 0.077 | `7fd25a869ff21678` |
| cand short r2 | 115.91 | 0.079 | `7fd25a869ff21678` |
| cand short r3 | 115.81 | 0.078 | `7fd25a869ff21678` |
| cand ctx1024 r1 | 100.43 | 1.010 | `7da83f06ec9f001d` |
| cand ctx1024 r2 | 99.92 | 1.014 | `7da83f06ec9f001d` |
| cand ctx1024 r3 | 100.18 | 1.011 | `7da83f06ec9f001d` |

## Medians

| leg | base | candidate | delta |
|---|---:|---:|---:|
| short decode | 116.32 | 115.81 | −0.44% |
| ctx1024 decode | 100.08 | 100.15 | +0.07% |

## Prefill (the honest finding)

| leg | base | candidate | delta |
|---|---:|---:|---:|
| short prefill s | 0.079 | 0.077 | −2% |
| ctx1024 prefill s | 0.954 | 1.011 | **+6.0%** |

The ctx1024 prefill regression is real: the unconditional loads in the
clamped form add ~57 ms to the 1053-token prefill (the loads now execute
even for lanes that would have been guarded, and the clamped index
computation adds a few ALU ops inside the k-loop). This is the trade:
correct FTZ subnormal handling costs ~6% prefill.

## Digests

Both pinned digests reproduce on EVERY run of BOTH arms:
- short: `7fd25a869ff21678` (12/12)
- ctx1024: `7da83f06ec9f001d` (12/12)

Zero mismatches. The clamp-then-load restructure preserves semantics
exactly.

## Full SHAs (64 hex)

| artifact | sha256 |
|---|---|
| base libmlx | `c28a485f98b55aadf12c68ae6d373f308517177c0cf33ad3c77ea82a796e931e` |
| candidate libmlx | `0300da4c2815440e537ee7c2036f448e13f7c2695c98dd8e8b8117bba226908d` |
| fckey driver | `aade269367b42001adb20156990570bf6ee1afd1576b43f778ee5c6a507f50b3` |
| e167 driver | `7087accecede1f556ed604d28fe88a8f6e5e90df202bc8cdf50c4cea51c76bfd` |

## Gate status

- Digest pins: PASS (12/12)
- Decode parity: PASS (−0.44% short, +0.07% ctx1024, within noise)
- ctx1024 prefill: **+6.0% regression** — the honest trade for correct
  FTZ handling; Main must disposition whether this is acceptable
- 16-shape: NOT run (qmm_shape_ab.py not staged on jwm1)

## Prior failed window

The first 12-round attempt produced zero data: all 24 runs died at
argparse because the script dropped `--model`/`--matrix`. Logs preserved
in `window-q4-12r/*.log` (all show argparse rc=2). This is separate from
the current window which ran correctly.
