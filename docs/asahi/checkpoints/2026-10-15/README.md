# J873 development checkpoint — 2026-10-15

This checkpoint retains three distinct source states. They do not have
interchangeable qualification results.

| Source | Status |
|---|---|
| Current `j873` branch | AGX3 merge plus local sample-mask, operand and layout work in progress; separately compile-checked for publication |
| `candidate1005.patch` | Isolated candidate with 118 distinct qualified hardware CTS cases before the ongoing subgroup survey |
| `candidate1041-from1005.patch` | Experimental opaque RGBA8 multisample point admission; built and CPU-replayed, pending hardware validation |

Apply `candidate1005.patch` to the exact base commit
`29dd453f6690fccbfd5d86849f12d891ef844fc5`, in an isolated checkout. Its 27
source hashes are retained in `candidate1005-sources.json`. Apply the 1041
patch only after 1005; `candidate1041-sources.json` records the resulting
hashes. These are source snapshots, not automatic installation commands.
`branch-sources.json` identifies the six local files being committed separately.

Candidate1005's library SHA256 is
`df89ef595913a4af0e13eb61c9f34021e63ebd2c4abbe1384c865c9122c0f48b`.
Its 118 distinct hardware passes comprise a 13-case canary and 105 disjoint
integer cases. A fresh 13-case repeat confirmed the new Linux boot but adds
no distinct coverage. The remaining 5,935 subgroup IDs are under survey;
the first 256 reported Pass, pending the separate evidence qualification.
All 6,053 subgroup IDs also have CPU replay coverage without compiler
refusal; those outcomes do not establish numerical GPU correctness.

Candidate1041's library SHA256 is
`333c2a1ff041906c2bffc1aa9f3ebdf65a53c449ab32406a6ba93fa0bc30bdca`.
Its 80-case CPU replay removed 16 point draw-admission refusals, preserved 64
controls and changed no shader binaries. Native PPP comparisons matched
12 point probes. The exact kicked native ring remains unproven, and no
physical CTS pass is attributed to 1041.

Selected receipts, the current plan, guarded-boot tools and CTS progress
runner are published in the companion
[m1n1 checkpoint](https://github.com/aurora-silicon/m1n1-aurora/tree/j873/experiments/j873/checkpoints/2026-10-15-mesa1005).
The full 63,657-ID inventory, configuration matrix, unresolved families and
final OpenGL 4.6 conformance remain open. Neither this source publication nor
the ongoing single-configuration survey is a conformance claim.
