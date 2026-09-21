# Q4 A/B corrected — jwm1 G13G, candidate REJECTED (2026-09-20)

## NEGATIVE RESULT

The clamp-then-load qmm_coopmat candidate (8bbaa1f4) is REJECTED:
ctx1024 prefill regresses +6.0% (0.954→1.011 s median). The +0.07%
decode delta is within noise, not exact parity. The candidate provides
no performance benefit and introduces a prefill cost.

## Correction of prior claims

- The "FTZ trade" framing was FALSE: this is a clamped-SSBO-load
  restructure, not an FTZ fix. The keyed sin-FTZ emission (095cb7e1b0a)
  is a separate commit.
- "+0.07% EXACT parity" was FALSE: 100.15 vs 100.20 is within noise but
  not identical.
- The prior run's ctx1024 was at 292 tokens, not the native 1053 —
  that comparison was invalid and is discarded.

## Raw data (3 rounds/arm/leg, 12 total)

### Short (30 tok)

| arm | r1 | r2 | r3 | median |
|---|---:|---:|---:|---:|
| base decode tok/s | 116.32 | 115.42 | 116.32 | 116.32 |
| cand decode tok/s | 113.94 | 115.91 | 115.81 | 115.81 |
| base digest | `7fd25a869ff21678` | same | same | — |
| cand digest | `7fd25a869ff21678` | same | same | — |

### ctx1024 (1053 tok, exact /tmp/q4-ctx1024-exact.txt)

| arm | r1 | r2 | r3 | median |
|---|---:|---:|---:|---:|
| base decode tok/s | 100.42 | 100.08 | 99.87 | 100.08 |
| cand decode tok/s | 100.43 | 99.92 | 100.18 | 100.18 |
| base prefill s | 0.954 | 0.956 | 0.951 | 0.954 |
| cand prefill s | 1.010 | 1.014 | 1.011 | 1.011 |
| base digest | `7da83f06ec9f001d` | same | same | — |
| cand digest | `7da83f06ec9f001d` | same | same | — |

## Digest gate

Both pinned digests reproduce on ALL 12 runs of BOTH arms. The
clamp-then-load restructure preserves semantics. The rejection is on
performance (prefill), not correctness.

## Full wheel SHAs (64 hex)

| artifact | sha256 |
|---|---|
| base wheel (v0.7.2-rc.1 published) | `fe51534b1ed658f64f2e16f5ce4709c27cbb1c683388b50a2d04eacf6b362a01` |
| candidate wheel (8bbaa1f4) | `a5792abfafc53a6a649fea0a6217d72f0e55a0e6f27cb22cccbb965e05c1dd8c` |
| base libmlx.so | `c28a485f98b55aadf12c68ae6d373f308517177c0cf33ad3c77ea82a796e931e` |
| candidate libmlx.so | `0300da4c2815440e537ee7c2036f448e13f7c2695c98dd8e8b8117bba226908d` |
| fckey driver .so | `aade269367b42001adb20156990570bf6ee1afd1576b43f778ee5c6a507f50b3` |
| e167 driver .so | `7087accecede1f556ed604d28fe88a8f6e5e90df202bc8cdf50c4cea51c76bfd` |

## 16-shape qualification

NOT RUN. qmm_shape_ab.py was not staged on jwm1. Qualification is
incomplete without it. Flagged, not silently dropped.

## CMakeCache verification

OMARCHY=ON, METAL=OFF confirmed in the candidate build tree
(/home/joshuawarren/q38-build/mlx-omarchy-candidate/build-integration/CMakeCache.txt).

## Environment

- Host: jwm1-linux (M1 T8103, G13G B1, 8 CPU)
- Lock: /tmp/m1-gpu.lock inode 27 (post-reboot recreation)
- Driver: fckey ICD aade2693 BuildID 1d97c0ee (both arms use same driver
  via ICD, no system install)
- venv: base = /var/tmp/jwm1-v072rc1/venv (5b183060), candidate =
  /tmp/mesa-sin-ftz-jwm1/venv-candidate (8bbaa1f4)
