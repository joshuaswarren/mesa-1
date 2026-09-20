# Q4 native-matched legs — jwm1 M1 G13G, integrated candidate vs Sep6 native divisor (2026-09-20)

Native-matched measurement per Main's directive: run the SAME unchanged
Sep-6 native Q4 legs (short-decode-32 + longctx-1024-decode-32 on
Qwen2.5-0.5B-4bit @ a5339a41) unchanged on current Linux with the
INTEGRATED candidate driver (honeykrisp-omarchy @ 095cb7e1b0a via
isolated ICD), retaining the same-die T8103 native divisor from
`mlx-omarchy/receipts/native-baseline-2026-09-06/`.

## Native reference (Sep 6, macOS 14.8.9, mlx 0.32.2-era, jwm1/T8103)

| leg | native decode med | native prefill med | native digest |
|---|---:|---:|---|
| short-decode-32 | 150.57 tok/s | 294.1 tok/s | `7fd25a869ff21678` |
| longctx-1024-decode-32 | 140.38 tok/s | 1840.9 tok/s | `7da83f06ec9f001d` |

5 repeats each, AC power, ids stable across repeats.

## Candidate results (jwm1, Linux, integrated driver)

| leg | candidate decode med | % of native | candidate prefill med | % of native | candidate digest |
|---|---:|---:|---:|---:|---|
| short-decode-32 | 115.93 tok/s | 77.0 % | 384.4 tok/s (exceeds native 294.1) | — | `7fd25a869ff21678` (matches native) |
| longctx-1024-decode-32 | 110.80 tok/s | 78.9 % | 808.6 tok/s (43.9 % of native 1840.9) | — | `079171ef27ad8242` (different prompt construction than native ctx leg — throughput is the parity metric, digest equality was never the cross-OS contract) |

## Comparison caveats (recorded, not hidden)

1. Native baseline predates the current Linux wheel generation
   (macOS 14.8.9 + mlx 0.32.2-era vs Linux current v0.7.2-rc.1 lineage):
   cross-OS-generation caveat applies, but the divisor is the exact
   same-die T8103 reference Main directed to retain.
2. This is NOT a performance-parity completion claim: 77-79 % decode
   parity confirms the per-dispatch submit cost remains THE open
   bottleneck, consistent with the term-A attribution.
3. jwm1 GPU window ran clean: no concurrent CPU build (the earlier
   -j2 contamination attempt aborted at preflight with zero
   executions), exclusive lock, isolated ICD, harness sha
   `f5062d88f34b0845` verified in-window.

## Staged pins

Model snapshot a5339a41 staged on jwm1 as REAL FILES (broken symlinks
quarantined): `model.safetensors` 278 MB sha `ddffab9cbc7bf6de…` verified
jw16-source == jwm1-dest. Harness `bench_decode.py` sha `f5062d88f34b0845`
+ deps `caps_sim_guard.py` / `mlx_provenance.py` shipped and verified.
