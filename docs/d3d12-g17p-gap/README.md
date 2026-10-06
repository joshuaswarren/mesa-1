# G17P native D3D12 gap campaign

The accepted t-1083 ledger contains 496 Neo strict B-only outcomes, not simply
the difference between the two pass totals. `CASES.tsv` lists every case, its
first observed diagnostic and plan slices. `bucket.py` reproduces that list
from the accepted raw evidence and verifies all 496 A/B pairs.

The final driver changes are linear commits on accepted i2
`3b1710ace113d4da8b56daf08076e6a0ac3626e6`:

* `54d529ce7c7`: publish Gen9 linear row pitch, native RGBA32 BG/EOT traffic
  and the existing monolithic vertex-fetch parameter pointer.
* `8f7347b64e6`: validate and publish bounded smooth coefficients/VIT through
  shared native PPP records, and reuse Honeykrisp quad/pass control encoding.

The latter stage retains flat/system/depth coefficient restrictions. Indexed,
indirect, advanced texture and software-stage execution still have separate
admission requirements. The physical qualification flags remain explicit.

The fallback-based prototype measured 139 recovered outcomes over the entire
291-case graphics-admission bucket. Its next version measured 120/142 shader
files and 10/17 frozen suites on a clean Neo boot. These results do not establish
the final i2-based branch's Neo pass count. Both final Mac checks retain 139/142
shader files and 11/17 frozen suites; the final Neo batches were stopped before
testing because the current boot journal contained a prior submit failure.

Phase 2 numbering differs from parity numbering: Phase 2 P04/P05/P06 are vertex
parameter/VIT/physical publication, P07 is indexing and P08 instancing; texture
execution is P03. Parity P14 groups vertex/index/instancing/texture execution;
P16 covers Phase 2 P10-P17 geometry/tessellation/mesh/amplification. Compute ABI
and meta programs map to Phase 2 P09 and parity P15.

The controller run report and exact hardware command ledger are at:

* `<aurora-ctl>/runs/r-t-1189-d3d12agx-on-g17p-close-the-496-vkd3d-proton-vkd3-1/report.md`
* `<aurora-ctl>/reports/d3d12-g17p-gap-20261006/HW-COMMANDS.md`
* `<aurora-ctl>/reports/d3d12-g17p-gap-20261006/BUCKETS.json`
* `<aurora-ctl>/reports/d3d12-g17p-gap-20261006/RUN-STATS.json`

Reproduce the analysis and host checks:

```sh
python3 docs/d3d12-g17p-gap/bucket.py \
  <aurora-ctl>/reports/d3d12-scoreboard-20261005 \
  <aurora-ctl>/reports/d3d12-g17p-gap-20261006
python3 docs/d3d12-g17p-gap/test_runner.py
python3 docs/d3d12-g17p-gap/summarize.py \
  <aurora-ctl>/reports/d3d12-g17p-gap-20261006 \
  <aurora-ctl>/reports/d3d12-stats
```

`run.py` derives from the accepted t-1083 MIT harness. It preserves strict
verdict accounting, records explicit physical opt-ins and prevents a batch
with kernel submit failures from acquiring a COMPLETE marker. `frozen.py`
hash-checks the 63 frozen inputs, runs the unchanged runner and saves each
child's command/output/status. `summarize.py` keeps revisions separate and
marks invalid or incomplete evidence explicitly.
