# D3D12 G13/G14 compatibility validation (2026-10-05)

Driver commit `ae23ffebb9d1475c7a1d3d24aff34dc6aeb8bf44` restores the legacy
graphics path on stock Asahi kernels. Both machines ran the same M1 Pro-built
library and the unchanged 63-file reference corpus from
`ba4cfd5dfa4c4d6f5b88adcf9ac5eb233c3442a4`.

* M1 Pro, t6000/G13S: Fedora Asahi Remix 44,
  `7.0.13-400.asahi.fc44.aarch64+16k`.
* M2, t8112/G14G: Fedora Asahi Remix 44,
  `7.1.6-400.asahi.fc44.aarch64+16k`.
* Before: `a212d162f5e4728d6723a667096348c49e701c8e`.
* After: `ae23ffebb9d1475c7a1d3d24aff34dc6aeb8bf44`.

| Suite | M1 Pro before | M1 Pro after | M2 before | M2 after |
|---|---|---|---|---|
| g1_smoke | FAIL(3) | FAIL(3) | FAIL(3) | FAIL(3) |
| g2_smoke | PASS | PASS | PASS | PASS |
| g3_smoke | FAIL(1) | PASS | FAIL(1) | PASS |
| g4_smoke | PASS | PASS | PASS | PASS |
| g4_triangle | FAIL(1) | PASS | FAIL(1) | PASS |
| g5_depth | FAIL(1) | FAIL(3) | FAIL(1) | FAIL(3) |
| g5_descriptors | PASS | PASS | PASS | PASS |
| g5_atomics | PASS | PASS | PASS | PASS |
| g5_control_flow | PASS | PASS | PASS | PASS |
| g5_wave | PASS | PASS | PASS | PASS |
| g5_texture | BLOCKED | PASS | BLOCKED | PASS |
| g6_geometry | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| g7_tessellation | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| g8_mesh | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| g9_amplification | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| g10_vertex_buffers | BLOCKED | PASS | BLOCKED | PASS |
| g11_indexed_instanced | BLOCKED | PASS | BLOCKED | PASS |
| **Strict total** | **6/17** | **11/17** | **6/17** | **11/17** |

All 17 suites were attempted on each run. BLOCKED means no final suite verdict;
it counts as a strict failure. No suite crashed or timed out.

The M1 Pro debugger observed `header.size=240`, `AGX_CHIP_G13X`,
`render_rsrc_spec_hi=false`, and kernel features `0x1`. Graphics jobs now
complete instead of failing submission with EINVAL. The legacy packet omits
the four appended resource-specifier high dwords: stock Asahi's `read_up_to()`
advances by the smaller of its own struct size and the advertised size, leaving
an oversized packet's tail to be interpreted as another command header.
Native DRM version >= 0.2 or the exact-prior-barrier feature selects the
extended packet; G17P always retains its previous extended packet.

G4's full-image and half-viewport pixel models, GPU-copy pixels and canaries
all pass on both machines. G3 also completes compute after render in the same
command list. G5 now reaches its depth and cull checks: depth passes, and the
cull counts are `none=26450 cullback=0 cullfront=26450 cullfront+FCC=0`.
Its three remaining failures are the frozen inverted-facing expectations,
also seen with the prior vkd3d reference measurements. G1 still asserts an
older capability profile and stub CreateHeap. G6/G7 retain incompatible shader
signature refusals; G8/G9 retain invalid pipeline-stream subobject refusals.

Both supplemental live CreateHeap probes pass. The frozen G1 descriptor
`{.SizeInBytes=4096}` returns `E_INVALIDARG` (`0x80070057`) and clears the
output. A DEFAULT heap of that size succeeds, its zero alignment/node masks
normalize correctly, and valid validation-only calls return `S_FALSE`.

Linux Meson packet and heap tests pass. The CPU BG/EOT resource-specifier
comparison against Honeykrisp passes all 1,296 cases. G17P GPU code, packet
selection, tiling and cache behavior are preserved by diff review. The
requested CreateHeap descriptor validation is shared by all GPUs. No Neo
runtime test was performed; the orchestrator owns that regression before
publication. No kernel, packages, services or system driver installation were
changed on either test Mac.

Raw logs, exact commands, build provenance and generated results:
`<aurora-ctl>/reports/d3d12-t1081-20261005/`.
See `tools/build-t1081.sh`, `tools/run17.sh`, `tools/probes-t1081.sh`,
`TABLE.md`, `results.json`, and `raw/<machine>-t1081-{before,after}/`.
Run a suite prefix from `/home/ryan/aurora-d3d12` with
`bash run17.sh prefix-t1081 results/<fresh-result-name>`; its frozen inputs
and artifact hashes are checked before execution. Both suite-window kernel
logs contain no entries. These observations use Linux-side measurements and
existing Asahi code only.
