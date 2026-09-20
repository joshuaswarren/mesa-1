# Q4 premerge driver A/B — jwm1 M1 G13G (2026-09-20)

Same driver-only causal A/B as `2026-09-20-q4-driver-ab-jw16.md`, on the
fresh jwm1 install using the restored runtime (receipt
`2026-09-20-jwm1-gpu-v072rc1-restore`): v0.7.2-rc.1 wheel
(`fe51534b…`, venv `/var/tmp/jwm1-v072rc1/venv`), model
`SiddhJagani/Qwen3.8-2B-mlx-4Bit` @ snapshot `0867d98b…`, harness
`bench_decode_vlm.py` (`--prompt-id long --tokens 32 --warmup-tokens 4
--stop-policy ignore-eos`), compile ENABLED (per the receipt protocol),
`HF_HUB_OFFLINE=1`, `MESA_SHADER_CACHE_DISABLE=true` both arms.
Drivers: candidate `aade2693…` vs baseline `7087acce…` (jw16-built generic
armv8-a binaries, sha-verified on jwm1, isolated ICDs — no install).
Lock: inode 1384 exclusive; no inference service exists on this image
(verified: 0 unit files) — nothing to stop/restore; GUI untouched.
Coordinated with M1RuntimeRestore (clear; their lane is CPU/disk only).

## Results (3 alternating rounds per arm)

| metric | candidate (fckey) | baseline (e167) | delta |
|---|---:|---:|---:|
| decode tok/s median | 13.2173 | 13.2755 | −0.44 % |
| prefill s median | 5.7337 | 5.7583 | −0.43 % |
| generated-ID digest | `24a65754f4a0d99a` | **same** | — |
| provenance | verified=match ×3 | verified=match ×3 | — |

The digest equals the restore receipt's pin exactly on every run of BOTH
arms: the sin-FTZ keying changes no generated token on this config (the
shipped decode path never declares DFTZ32, so the default emission is
byte-identical — as designed). Performance parity within noise (round
spread ±0.1 tok/s).

## Gate

PASS: digest pin held on all 6 runs, decode/prefill parity within noise,
no regression. Combined with the jw16 window (`2026-09-20-q4-driver-ab-jw16.md`)
and the 1989-case CTS window (`2026-09-20-cts146-window-jw16.md`), the
candidate satisfies the staged pre-merge gate on both consumer hosts at
screening grade; 12-round decision batteries are the follow-up if
merge-grade evidence is wanted.

Artifacts: `/tmp/mesa-sin-ftz-jwm1/window-q4/` on jwm1
(`{fckey,e167}-r{1,2,3}.log` — each contains the full result JSON line).
