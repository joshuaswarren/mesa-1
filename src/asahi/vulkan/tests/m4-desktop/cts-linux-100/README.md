# Fixed Linux 100-case development target

User requested 100 Linux passes on 2026-09-13, explicitly excluding Windows.
Retain the applicable original sample and its order. Replace six Windows-only
cases and four requests for DISJOINT on single-plane formats with ten samples
from the original mustpass population (seed 20260918). No exclusions depend on
whether a test passes or on optional device feature support. Original case 89
is valid and retained. Exact rules, removals and additions: selection.json.

Use the separately identified CTS 1.0.2.6 diagnostic binary with four retained
corrections: input-aspect lifetime/subpass reference, YCbCr readback layout,
scaled clear union member, and the invalid destination-linear-filter precheck.
The patches preserve pixel comparisons. Source and original executable were
restored after building. Exact patch and binary hashes: cts-build-manifest.json.
This is a development target, not an unmodified CTS or Vulkan conformance claim.
The original 100-case report remains separate in ../cts-1.0-baseline/.

Initial run: 81 Pass, 19 NotSupported, no Fail or Crash. All 100 outcomes and
guarded evidence paths are retained in initial-results.json and
initial-summary.json. Every hardware batch uses the 60-second watchdog and
0.5-second thermal guard. Never replace a failing or unsupported Linux case to
improve this target's result.

Target files: /root/cts-1.0.2.6/linux-cases.txt and
/root/cts-1.0.2.6/deqp-vk-linux-corrected.

BGR8 UINT follow-up: padded BGRA storage supports sampling and raw transfers;
BLIT_DST remains unadvertised pending native BGRA integer attachment export.
Case 91 Pass in 20260913T161629985260Z-cts-padded-bgr-granularity (82 known passes).
The BGR format was added to the existing CTS sampler oracle without altering
reference checks: repeat sampling Pass in 20260913T161821590699Z-cts-bgr-sampling-control,
and opaque-white border sampling Pass in the subsequent cts-bgr-border-control
experiment. The initial extra border-control name was invalid and selected zero
tests, not a process or GPU crash; evidence is retained. These added controls
are separate from the frozen 100-case target.

D24S8 follow-up: separate Z32_FLOAT depth and S8 stencil planes with GPU packed
24-bit transfer conversion. Fused normalization preserves all depth bits in
the exhaustive host arithmetic model (build/cts-next/d24-arithmetic.log).
Exact GPU mip/array/padded-copy probes Pass in
20260913T163411496978Z-d24-exact-copy-probe. Original CTS depth image copy,
three clears and three nearest/linear shadow comparison controls Pass in
20260913T163443287457Z-cts-d24-controls-000. Depth blit destination remains
unadvertised and its control reports NotSupported, not Pass. Host transfers
and multisampling of this emulated format remain unadvertised. Three target
stencil cases pass in 20260913T162648763042Z-cts-d24-stencil, bringing known
target passes to 85; this is not yet a fresh full-target run. Watchdog and
thermal guards active, no reset.

Explicit chroma reconstruction: immutable filter state selects luma-centre
reconstruction followed by full-resolution filtering. Nearest chroma ignores
chroma location; linear retains cosited/midpoint coordinates. Target 98 Pass
in 20260913T164314522092Z-cts-explicit-chroma-flrp (86 known target passes).
Two implementation-time userspace compiler crashes are retained, diagnosed and
fixed: editing an unattached NIR clone, and stale flrp_lowered state. No GPU reset.

Ten extra controls cover nearest/linear texture and reconstruction filters,
2/3 planes, 420/422 and 8/10-bit values. Initial 6 Pass / 4 Fail in
20260913T164354617458Z-cts-explicit-controls-000 exposed a CTS oracle error:
reconstructLinearXYChromaSample reverses midpoint weights. For i=2k,
i_RB = 0.5*(i-0.5) = k-0.25, so the upper sample weight is 0.75, not 0.25;
for i=2k+1 it is 0.25. This agrees with CTS's own 422 oracle and the equations at
https://docs.vulkan.org/spec/latest/chapters/textures.html#textures-explicit-reconstruction
The same inverted XY weights were found in Khronos CTS main on this date.
chroma-midpoint-reference.patch corrects just these two expressions; no
tolerances or pixel checks are removed. All ten controls Pass using the
separately built diagnostic in 20260913T164603014781Z-cts-chroma-midpoint-oracle-000.
Exact combined patch/manifest are retained. The fixed 100 target still uses its
previous diagnostic binary: its explicit-nearest case needs no oracle change.
