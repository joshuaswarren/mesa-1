# Branch triage — legacy fork → mesa-1 (2026-09-19)

Archaeology of all 21 branches on `joshuaswarren/mesa` (legacy, frozen) against
`honeykrisp-omarchy` @ `d8d4e1c500b` (identical on both forks). Ports landed on
`joshuaswarren/mesa-1`. No hardware was touched; every number below is quoted
from existing receipts.

## Classification table

| Branch | Class | Ahead of hk trunk | Evidence |
|---|---|---|---|
| `honeykrisp-omarchy` | (trunk) | 0 | Identical tip `d8d4e1c500` on both forks. |
| `main` | SUPERSEDED | 0 | Tip `4a34ded300` is an ancestor of trunk (magma-gpu-rs bump); no unique work. |
| `honeykrisp-coopmat` | SUPERSEDED | 0 | Tip merged into trunk; was the base for all hk/* work (0x6f bit6 simd_matrix). |
| `honeykrisp-miscompile-repros` | SUPERSEDED | 0 | Verified, not assumed: tip `7302d4328` is an ancestor of trunk; the branch is repro-case hygiene plus hardware verdicts ("cases 2/3/5 reproduce on m1-test-host, 1/4 do not") already recorded in trunk history. |
| `hk/byte-extract` | SUPERSEDED (RECEIPTED-WIN content) | 1 | Merged via cherry-pick `473e74fbf3b` in trunk/`upstream/correctness` (same subject, different sha); receipt `mesa:receipts/2026-09-08-byte-extract.json` (ubfe fix, verified m1-test-host G13G). |
| `hk/precise-math` | SUPERSEDED (RECEIPTED-WIN content) | 0 | Merged; `mesa:receipts/2026-09-08-precise-math.json`. |
| `hk/coopmat-shapes` | SUPERSEDED (RECEIPTED-WIN content) | 0 | Merged; `mesa:receipts/2026-09-08-coopmat-shapes.json`. |
| `hk/queue-lifetime` | SUPERSEDED | 0 | Merged (diagnostics-only); wedge root-caused to mlx-omarchy, no driver defect (`mesa:receipts/2026-09-08-queue-lifetime.json`). |
| `hk/cdm-barrier-trim` | SUPERSEDED (RECEIPTED-WIN) | 0 | Landed 2026-09-16: G13G trim; packaged A/B ctx1053 99.49→105.78 tok/s (+6.32%), short +2.90%, digests held (termA receipt §Verdict). The G13X half was reverted in `5deac1c` after t6001-test-host ctx regression. |
| `hk/cdm-barrier-floor` | SUPERSEDED | 1 | Tip `66b5631a` is contained in `hk/app-barrier` (both forks). Per-launch barrier measured a throughput no-op but load-bearing for correctness (4/41 suite cases fail without it): `receipts/2026-09-10-dispatch-floor/verdict.json`. Instrumentation only. |
| `hk/app-barrier` | RECEIPTED-LOSS (no-effect/subsumed) | 2 | Builds on barrier-floor; adds `HK_CDMBARBITS` + `HK_APPBAR=1`. appbar mode measured +1.94% and subsumed by the trim (`appbar-ab.json`, termA §1.3). Already on mesa-1 (`33deb56de`, contains old tip). |
| `hk/cdm-chain-batch` | RECEIPTED-LOSS (NO-LAND) | 1 | `receipts/2026-09-19-q4-chainbatch-t6001-test-host.md`: emission reduction works (16 vs ~2500 barriers/2-token run) but digests corrupt nondeterministically on t6001-test-host. Barrier is load-bearing memory ordering. |
| `hk/cdm-g13x-178` | UNRECEIPTED (unmeasured probe) | 1 | Designed set {4,5,6,8} + USC inval {3,4,5,6,8}: **bench failed at instance creation; unmeasured** (termA addendum 4 §9.3 table). The only bit-trim candidate never measured. |
| `hk/cdm-g13x-578` | RECEIPTED-LOSS (correctness-broken) | 1 | {4,5,7,8}: breaks generated-ID digests at round 0 (termA addendum 4 §9.3). |
| `hk/cdm-g13x-1f0` | RECEIPTED-LOSS | 1 | G13G trim {4,5,6,7,8} on G13X: −18% short / −38% ctx, digests hold (addendum 4 §9.3, §9.4). |
| `hk/cdm-g13x-fffb` | RECEIPTED-LOSS | 1 | Sink−unk_2: −24% short / −40% ctx at round 0, digests clean (addendum 4 §9.3). |
| `hk/fma-ceiling-unroll` | RECEIPTED-LOSS | 1 | Commit message itself: "REJECTED: measured regression" — qmm_coopmat 1034→817 GFLOP/s (−21%), digest identical. |
| `termb/device-load-coh7` | RECEIPTED-LOSS | 1 | `receipts/2026-09-16-termB-kv-mechanism.md` §(c): coherency 4→7 value-clean but −8% decode (190.18→175.03 tok/s). |
| `hk/trig-invariance` | RECEIPTED-LOSS (harness retained) | 1 | `mlx-omarchy:receipts/2026-09-16-mesa-trig-invariance.md`: trig windows compile with identical opcode sequences; perf gate fails, fold does not land. Value = offline compile harness. Already on mesa-1 (`9c3d01e06`). |
| `hk/agx-wait-batching` | RECEIPTED-LOSS (NO-LAND) | 11 | `receipts/2026-09-19-agx-wait-batching-t6001-test-host.md`: digest-clean but perf-neutral on t6001-test-host; NO-LAND. Already on mesa-1 (`40905e2f2`, contains old tip `14eb6778`). |
| `upstream/correctness` | UNRECEIPTED (by design — PR prep, correctness not perf) | 6 | The landed correctness set re-sequenced for upstream; 6 commits incl. trunk-absent `7c6aa47ce96`, `aa4fee0d5`. Identical tip on both forks. |

## Ports to mesa-1

Pushed 2026-09-19, history and authorship intact (ref push, no rewrite):

`hk/cdm-chain-batch` `ae819e10`, `hk/cdm-g13x-178` `242e591a`,
`hk/cdm-g13x-1f0` `5cd0e72f`, `hk/cdm-g13x-578` `49c21f17`,
`hk/cdm-g13x-fffb` `66955760`, `hk/fma-ceiling-unroll` `c4251875`,
`termb/device-load-coh7` `f6286e41`.

Not ported, deliberately:

- Merged/superseded content already on mesa-1 via trunk or an existing branch:
  `main`, `honeykrisp-coopmat`, `honeykrisp-miscompile-repros`, `hk/byte-extract`,
  `hk/precise-math`, `hk/coopmat-shapes`, `hk/queue-lifetime`,
  `hk/cdm-barrier-trim`, `hk/cdm-barrier-floor` (contained in `hk/app-barrier`).
- Already on mesa-1 with tips containing the old tips:
  `hk/app-barrier`, `hk/trig-invariance`, `hk/agx-wait-batching`,
  `honeykrisp-omarchy`, `upstream/correctness`.

Conflict surface: every pushed branch trial-merges clean onto mesa-1 trunk
(`git merge-tree --write-tree`, no conflicts). The `hk/cdm-g13x-*` variants all
touch `libagx_dgc.h` `agx_cdm_barrier()` and are mutually exclusive
alternatives — merge at most one.

## The cdm family: how the seven branches relate

One problem — the per-launch `CDM_BARRIER` cache-maintenance op Honeykrisp
emits after every compute launch (~2.1 ms of a ~9.8 ms m1-test-host decode token) —
attacked three independent ways. They are alternative approaches, not stackable:

1. **Reduce bits per barrier** (`hk/cdm-barrier-trim` landed for G13G;
   `hk/cdm-g13x-{178,1f0,578,fffb}` are four alternative bit sets for G13X,
   mutually exclusive — exactly one `agx_cdm_barrier()` body each).
2. **Emit fewer barriers** (`hk/cdm-chain-batch`: defer the barrier across
   merged dependent control streams, one barrier per chain instead of per
   launch).
3. **Move the barrier to app-recorded barriers** (`hk/app-barrier` `HK_APPBAR=1`;
   subsumed — the mlx-omarchy graph is a serial chain so almost every pair
   already has an app barrier).
   `hk/cdm-barrier-floor` is the shared instrumentation underneath 1–3
   (nocdmbarrier/usccdmbarrier perftest knobs).

Measured outcome on t6001-test-host (termA addendum 4, packaged byte-verified probes):
the barrier bits are a coupled maintenance family. Known-good G13X sets are
exactly the full sink (ctx-protective: +3.17% ctx1053, short −13%) and the
designed set {4,5,6,8} (mirror image); every sampled subset between them is
correctness-broken or hits a −18…−40% cliff. The 2026-09-19 chain-batch screen
then showed the barrier is load-bearing memory ordering, closing the whole
amortize-the-barrier lever family. `hk/cdm-g13x-178` is the sole unmeasured
candidate.

## Ranked benchmark queue (for a later t6001-test-host lane under /tmp/gpu.lock)

1. **hk/cdm-g13x-178 — the only unmeasured branch and the only one that could
   still hold both legs.** Designed set {4,5,6,8} alone wins short +13%
   (190.66→215.47 tok/s, 12-rd) but loses ctx1024 −3.17% (142.12→137.62); 178
   adds bit 3, the one *named* maintenance bit (USC cache inval, Asahi Lina),
   hypothesised to restore the KV-stream leg. A/B: installed
   `hk5deac1c-2` (sink) vs packaged `hk242e591-2`, PKGBUILD recipe unchanged,
   emission byte-verified with `ASAHI_MESA_DEBUG=trace` + agxdecode before
   measuring. Legs: **short AND ctx1024** (≥12 interleaved rounds each —
   ctx needs n≥12 per the addendum 4 noise floor; short resolves 1%).
   Gate: pins `7fd25a869ff21678` / `7da83f06ec9f001d` held on every run.
   Prior: sink−unk_2 was −24%/−40% at round 0, so abort at round 0 on a big
   short cliff.
2. **Reconfirm the sink-vs-designed trade on the current v0.7.1 wheel**
   (both numbers predate it; two-pass KV landed since). Same packaged A/B, 12
   rounds; cheap because both packages exist. Contexts: short + ctx1024.
3. **Nothing else.** Every other branch is receipted NO-LAND/REJECTED with
   direct evidence; re-measuring them is not queued.

Note on a future G13G-context win: `hk/cdm-barrier-trim` is already landed and
packaged there; no open G13G candidate exists.

## Ready-to-PR candidate sitting unmerged

**`upstream/correctness`** (`aa4fee0d5`, present on mesa-1, deliberately
un-merged into trunk so the PR base stays upstream-clean). It carries the
landed, receipted correctness work (ubfe, correctly-rounded fdiv/frcp, faithful
log, exact sin/cos) plus two commits not yet in trunk (`7c6aa47ce96`
flush-to-zero in fp32 div/log/sin; `aa4fee0d5` `nir: pass FP_MATH_CTRL to the
builder generator`). This is the only branch on either fork that is PR-ready as
submitted work; it is correctness, not performance, so it needs upstream-style
review rather than a GPU screen.
