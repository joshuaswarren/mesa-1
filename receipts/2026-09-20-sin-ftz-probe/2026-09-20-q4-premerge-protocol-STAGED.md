# Q4 decode/prefill pin + performance regression — pre-merge gate protocol (STAGED, not executed)

Purpose: before proposing `hk/fdiv-fckey` (or any Mesa fork change) for a
`omacom/mesa` merge, the merge candidate must pass the standing Q4
decode/prefill regression on BOTH consumer hosts. This document is the
prepared protocol; execution requires Main-owned GPU windows.

## Hosts and current baselines (historical, for delta judgment)

| host | silicon | driver reference | wheel | short tok/s | ctx1024 tok/s | source |
|---|---|---|---|---:|---:|---|
| jwm1 (G13G) | M1 t8103 | hkf96e090-2 | v0.6.0 `2e252962` | 116.97 | 105.78 | 2026-09-16-termA packaged battery |
| jw16 (G13C) | M1 Max t6001 | hk5deac1c-2 | v0.7.1 `50eeb29` | 191.90 | 150.76 | 2026-09-19-q4-chainbatch ladder |

Model: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` @ `a5339a41`.
Digest pins (fatal on every run): short `7fd25a869ff21678`,
ctx1024 `7da83f06ec9f001d`. Protocol: 12 interleaved rounds, rotating leg
order, fresh bench_decode subprocess per leg, `MLX_DISABLE_COMPILE=1
HF_HUB_OFFLINE=1`, one `/tmp/m1-gpu.lock` flock hold per window,
llm-inference stop/restore with active+health confirmation.

## Gate (proposed for the merge proposal)

1. Digest pins hold on both hosts, both legs (driver swap only; runtime
   wheel held fixed per host).
2. Decode medians within the host's wander band of its pre-window baseline
   (short <1.2 % spread; ctx within documented ±5-10 % wander); no leg
   regresses beyond wander.
3. Prefill legs recorded (short + ctx) and within wander.
4. CTS 1989-case float_controls subset: 0 Fail on the candidate (three
   sin_denorm_flush_to_zero fixes expected on every generation; the rest
   identical to baseline).
5. omarchy runtime suite green on the candidate driver.

## Staged artifacts

- Driver A/B arms: `/tmp/mesa-sin-ftz-jwm1/drivers/*.so` on jwm1
  (candidate `aade2693…`, baseline `7087acce…`) — ICD-switched, no install.
- CTS gate: `/tmp/mesa-sin-ftz-jwm1/bin/deqp-vk` +
  `probe/fp32-caselist.txt` (1989 cases, binary-verified).
- jwm1 gaps to close before the perf leg (owner/provisioning scope, NOT
  this lane): mlx-omarchy wheel venv + model snapshot + vulkan-tools
  (vulkaninfo) on the fresh install; benchmark harness `bench_decode.py`
  sha `f5062d88f34b0845…` to be shipped with the wheel.

## Status

STAGED ONLY. Nothing in this protocol has run on jwm1; the digests and
baselines above are quoted from the cited receipts, not re-measured.
