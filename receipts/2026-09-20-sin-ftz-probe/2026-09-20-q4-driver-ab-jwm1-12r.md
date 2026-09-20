# Q4 premerge driver A/B — jwm1 M1 G13G, 12 rounds/arm corrected rerun (2026-09-20)

Corrected merge-grade rerun of `2026-09-20-q4-driver-ab-jwm1.md` (the
3-round screening window that PASSED). The first 12-round attempt
produced zero data — the rewritten script dropped `--model`/`--matrix`
and all 24 runs died at argparse (reported to Main; artifacts in
`window-q4-12r/*.log` preserved as the failure record). Per Main's
authorization: exact-command smoke (rc=0, correct schema, correct
digest) validated before this batch; M1RuntimeRestore's guard re-kbuild
completed before the window (ane.ko 9875ef0, sha `99e8b8b5…`, no
concurrent ANE deploy).

## Config (identical to the 3-round screening window)

Candidate `hk/fdiv-fckey 095cb7e1b0a` (`aade2693…`) vs matched baseline
`e1677564284` (`7087acce…`); jw16-built generic armv8-a binaries
(sha-verified on jwm1); v0.7.2-rc.1 restored runtime
(`/var/tmp/jwm1-v072rc1/venv`, wheel `fe51534b…`); model
`SiddhJagani/Qwen3.8-2B-mlx-4Bit` @ snapshot `0867d98b…`; harness
`bench_decode_vlm.py` (`--prompt-id long --tokens 32 --warmup-tokens 4
--stop-policy ignore-eos`), compile ENABLED, `HF_HUB_OFFLINE=1`,
`MESA_SHADER_CACHE_DISABLE=true` both arms; isolated ICDs; lock inode 27
(post-reboot recreation of /tmp/m1-gpu.lock — the reboot wiped /tmp, so
1384 no longer exists; noted for the record); no inference service on
this image (nothing to stop/restore); GUI untouched. Coordinated with
M1RuntimeRestore before launch.

## Results (12 rounds per arm, alternating; 24/24 runs rc=0)

| metric | candidate (fckey) | baseline (e167) | delta |
|---|---:|---:|---:|
| decode tok/s median | 13.2463 | 13.2638 | −0.13 % |
| prefill s median | 5.7579 | 5.7555 | +0.04 % |
| generated-ID digest | `24a65754f4a0d99a` | **same** | — |
| provenance | verified=match ×12 | verified=match ×12 | — |

Per-run spread: candidate 13.1969–13.3442, baseline 13.2042–13.2953 —
distributions fully overlapping; parity within noise. Prefill parity
to 0.04 %.

## Gate

DECISION-GRADE PASS on jwm1/G13G: digest pin held 24/24 runs, decode
and prefill parity confirmed at 12-round depth. Combined with:

- jw16/G13C 12-round decision window (`2026-09-20-q4-driver-ab-jw16-12r.md`):
  PASS, 48/48 digests, all legs within documented wander;
- jw16/G13C + jwm1/G13G 1989-case CTS windows
  (`2026-09-20-cts146-window-jw16.md`, `…-jwm1.md`): candidate 0 Fail
  (fixes the 3 baseline FTZ-sin failures), NotSupported identical
  (declared shaderFloat64 gaps);
- M2/G14C device A/B (`2026-09-20-sin-ftz-window.md`): same 3-case fix.

…the candidate `hk/fdiv-fckey 095cb7e1b0a` satisfies the staged
pre-merge gate (`2026-09-20-q4-premerge-protocol-STAGED.md`) on all
three Apple GPU generations at decision grade for the Q4 decode path,
with the keyed DFTZ32 emission additionally CTS-verified on two of
them. Merge proposal readiness rests with Main.

Artifacts: `jwm1-q4-12r/` — 24 per-run logs (full result JSON lines),
pins-before/after. No binaries archived (pins recorded).
