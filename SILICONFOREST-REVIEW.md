# Mesa source integration review

## October 24 integration candidate

The update merges Aurora `aurora-wip` at
`4911479bf988ce88ec0889ed0378352a0bb0fae2` and Scott Jones's draft
[PR71](https://github.com/aurora-silicon/mesa/pull/71) at
`0163e7fa96f4c54f2fb72e6053b2aaa22446a44d`. Both remain original ancestry,
with original author identities. There were 214 upstream-only commits and 33
downstream-only commits before integration; these counts include merges and
maintenance, not 214 missing M3 driver fixes.

- Retain downstream placed-mapping ownership, fixed-address drm-shim behavior,
  the checked configurable geometry heap, and Zink ordering fixes. Equivalent
  upstream RT fixture sequencing and NULL XFB counter handling replace the old
  implementations without dropping their regression coverage.
- Update the known-collision ABI gate for Aurora's unified high feature bits
  and timestamp-copy payloads. The checked 18 wire facts match kernel
  `7b969982306e1498a8cbe6e5da9bde743acd8fc5`'s canonical header. Its retained
  G15 backend uses a separate low-bit wire, now handled by the explicit
  converter described below. Header equality alone does not qualify a backend.
- T8122's separate runtime reports VM_STATUS and SCHEDULED_QUEUES, without
  exact-prior barriers (`m3/m3_drm.rs:217` in the pinned kernel). Give G15G its
  own baseline check instead of requiring the Pro/Max/Neo feature pair. Host
  drm-shim probes cover its actual feature baseline and reject user binds into
  all four T8122 reservations; no GPU submission is performed.
- Keep M4 admission disabled even though the upstream snapshot contains M4
  code. Preserve T6030/T6031/Neo identity checks. T8122 uses its exact supported
  firmware/runtime contract without a Mesa environment opt-in, as requested by
  the owner. The obsolete variable being unset, `0` or `1` does not change
  admission. Kernel and packaging defaults remain separate work.
- Keep all four T8122 runtime reservations and the below-2^39 heap/alias cap.
  The updated layout has host tests, but Scott's earlier hardware results used
  a different layout. J613 texture/MSAA/compute and desktop checks remain due;
  J615 and HAL200 are not qualified by those earlier results.
- GCC 16 at `-O2` exposed a nested pNext compression rejection regression.
  Copying each header's type and next pointer through untyped storage, plus
  copying the external-memory payload, preserves all 3,359 policy checks;
  neither the policy nor the failing assertion is relaxed.
- Host fixtures now follow the integrated policies: G15G wide CDM barriers
  are positive cases, while actual M4 device admission is refused without
  removing offline compiler capability coverage. The inherited image-store
  regression compared whole G17P/G16G binaries, including unrelated register
  copies and scheduling. It now compares the single image-write instruction
  and independently checks the float/integer data bit. No production compiler
  change is needed; a deliberate float-as-integer mutation must fail the test.
- Preserve this fork's assistant policy; do not import Aurora's workflow that
  rewrites it. Historical compatibility branches and private-source exclusions
  are unchanged.

The later Aurora code includes native render-pass/depth-store improvements,
query-progress and upload-reuse work, compiler fixes, and the unified kernel
UAPI. No FPS gain is claimed without a matched hardware benchmark. Full CTS,
M1/M2/M3 hardware regression tests and installing a system package remain
separate from host compilation and drm-shim validation.

## Main integration

The starting point is the current 26.3 development tree, rather than an older
26.1 source tree with every later branch applied over it.

| Work | Exact source | Decision |
| --- | --- | --- |
| DJ Neo / AGX3b | `aurora-silicon/mesa:agx3b`, `dba15302aa0554b52ec6ca5103bd0391035e0fdc` | Preserved as ancestry of the native M3 source; includes DJ's merged PR12–22 work |
| Eryk M3 Pro | [PR23](https://github.com/aurora-silicon/mesa/pull/23), `cf8ec613be202d8bfc5440e61e9aa6c60700fad7` | Main foundation; 39 commits after DJ's pinned head |
| Naeem M3 Max | [PR25](https://github.com/aurora-silicon/mesa/pull/25), `aab698d2ef34e2aef38ba4d2bfcfa3abc5ded4c5` | Merged as original ancestry; four unique commits, preserving all nine later Eryk commits missing from this fork |
| Joshua null transform-feedback counters | [omacom PR10](https://github.com/omacom/mesa/pull/10), `b87e7456cdbb7abdea2f360cdcba0e2a56316f3f` | Cherry-picked with provenance; no dereference of a NULL counter array |
| Eryk/Mike/Ryan Zink protections | [PR7](https://github.com/aurora-silicon/mesa/pull/7), `53a09d65633f1e6d6cc26bf851c16f63fcec2060`, `02c37f31ded86dd39e721dfccc8b43668aecb3fb` | Imported ordered-access history, combined barrier masks, depth-unbind transition, regression fixture and fast-path follow-up |
| Joshua placed-memory review | [omacom PR8](https://github.com/omacom/mesa/pull/8), `72c18d4deb9440c839ad2fb35b0537fd993915e0`, `f91b6f40b3c6c529fc523c3339060e0af165762f`, `ddfb4cf114a6e72b690581ff6fb2fe8447d11ffa` | Adapted missing range/result/cache checks while retaining the base's newer independent mappings for aliased imports |

The Zink integration preserves the source checkpoint's attribution to Mike
Blumenkrantz (`b448ad053f2ce190a5b6b2d27b30dcbf3d45ec7d` /
`6ff2d4aa9da581e3c5f9fdaa4b51416b116d5ace`) and Eryk
(`53a363032ed040c87e4ff1d97876bfd00c7e2424`). The full discovery snapshot retains
public PR history alongside the new integration commits.

## Adversarial findings and treatment

1. **A newer branch name was not a superset.** Naeem's branch diverged before
   nine Eryk correctness fixes: staging-bank/read hazards, cube texture indices,
   half-source square root, GL scratch word packing, multisampled window
   buffers, compute variant reuse, border-depth comparison and tile-load masks.
   Merging into Eryk's current head retained these instead of replacing the tree
   with Naeem's snapshot.
2. **Known kernel ABI collision.** The maintained kernel and the experimental
   native kernel assign different meanings to bits 3–8 and 10–15. Render and
   compute payloads also differ by timestamp-copy fields. A GPU identity check
   alone is insufficient. Main requires the native per-command-sync and exact
   prior-barrier contract. The physical integration follow-up also supports
   Omacom's retained G15S/G15C `0x3e`/`0xbe` wire through an explicit feature
   and packet converter; it never passes canonical packets to that backend.
   See [the retained-wire contract](SILICONFOREST-G15-WIRE.md).
   CPU tests exercise missing and conflicting
   feature bits. This does not prove compatibility with arbitrary kernels that
   fabricate the same feature word.
3. **Blind cherry-picking regressed newer implementations.** Thomas Glanzmann's
   [plane-offset fix](https://github.com/aurora-silicon/mesa/pull/10),
   `ba289e643e0b361747988280a459b73c34c2cad9`, is already covered by the base's
   `offset_B` handling plus overflow, row-pitch, overlap and binding validation.
   Its older implementation was not reapplied. The production image-layout
   tests are included in quick checks and the complete Asahi suite.
4. **Placed mappings needed a selective port.** The base already tracks mapped
   lengths and gives imported memory objects independent mappings. Replacing
   that with PR8 would remove working alias isolation. The adaptation instead
   adds runtime allocation/BO bounds, a runtime result-address check, protection
   against moving a shared mapping on backends without independent mappings,
   and release of a placed mapping before returning its BO to the cache.
   A NULL/already-unmapped object does no reserve mmap. Physical Vulkan CTS and
   virtio execution remain unqualified.
5. **Capture/replay changes are not interchangeable.** Joshua's
   [omacom PR9](https://github.com/omacom/mesa/pull/9),
   `11ae82d8d8c379c2ae895806b4409537c5ceb25b`, introduces a separate 4-GiB range.
   The native base already has a separate replay heap reserved after firmware
   and alias holes. Replacing the allocator with PR9 would discard those holes.
   Its AGX2-specific extension/error classification still needs a dedicated port
   and is held; this repository does not claim that M1/M2 capture/replay issue is
   resolved by the native implementation.
6. **The reported compression failure was not reproduced.** The current exact
   compression-policy source passes 3,035 host checks. No policy was weakened
   and no test disabled to turn an older reported 33/34 result green.
7. **Feature exposure stays narrow.** G15C does not gain unsupported ray-query
   advertisement. Neo sparse aliasing/strict residency restrictions remain.
   No broad hardware-admission switch, known-bad barrier simplification, or
   forced compiler generation was added.

8. **The full host run exposed inherited test and shim defects.** The compute
   wire assertion still described an older 88-byte packet; it now checks the
   actual 136-byte payload and the appended field offsets. The RT fixture read
   a length in the same C argument list that an encoder modified it; reading
   through a pointer inside the assertion sequences those operations without
   changing expected bytes. All 611 checks remain. The generic drm-shim mmap
   wrapper dropped the requested address, making placed mappings try address
   zero; it now forwards that address. All 36 Asahi suites and the four host
   Vulkan regression profiles pass, including Vulkan validation on the latter.

## Experimental and held work

| Source | Treatment and reason |
| --- | --- |
| Newer private M5 HAL300 research | Kept local at the owner's instruction. No private-source snapshot or runnable M5 claim is published here. |
| DJ public `m5-agx-g15-g17-failclosed`, `068c165a0c83cab4e4c6e9a3c89f63f7a4fa0c2c` | Preserved on `experimental/m5-recognition` and as an exact source tag. Recognition/encoder research remains fail-closed, not an active M5 driver. |
| Eryk M4, `08b19ec3a2e422a7252fdfe77c185450fccf86da` | Preserved on `experimental/m4` with original qualification evidence; no new hardware claim. |
| Personal Air/legacy/older Pro/Max sources | Separate named profiles; firmware and kernel contracts differ from main. Only policy/branch notices and removal of the obsolete inherited macOS workflow differ at their new tips. |
| Joshua omacom PR3–6 | Source snapshots retained. Latest PR6 `f144c7949b7d960f26677b671341d5a123057ddf` reports 55 regressions over 3,253,584 cases. A narrow later pass does not qualify the complete performance stack. Not merged. |
| Joshua PR11 `44841202b4fd041fc29b6ff77f585d4df6e70dd7` | Depends on the above performance stack; retained for review, not applied standalone or enabled. |
| Official MR44475 phi shuffling, `e22cd9bd50daa4671b41a81ee72c8238a0604a8f` | Examined. Main has the newer parallel-phi implementation that snapshots original destinations and whole vector ranges. The older one-line reverse-walk fix is not a drop-in replacement. |
| Official MR44041 lighter CDM barriers | Retained for review. It removes undocumented barrier bits and does not establish equivalent cache guarantees for the merged native GPU paths. |
| Official MR44932 virtio implicit-sync optimization | Exact candidate head retained. Current external-BO lifetime code differs; do not blindly apply older sync assumptions. |
| Official GL/Vulkan interop MR42872, memory-budget MR40238, SVM MR37668 | Candidate snapshots retained; these are separate features requiring targeted runtime validation, not prerequisites for native M3/Neo source integration. |
| Official Mesa main, `c823a83e8f373a5c36c2b9ed1d7dc98a4f3360ee` | Pinned as a future rebase source. It is 1,335 commits beyond this fork's October 1 base and includes cross-driver/NIR interface changes. Not merged wholesale during this chip-stack integration. |
| Official Asahi fork main, `01ac84e119f468a3267714b0b4a339316a3083f4` | Checked directly; its tip is dated April 2025, so it is not presumed newer than the current Mesa development source. |
| DJ Windows D3DKMT/WC/UMD branches | Preserved in the inventory; not mixed into Linux default behavior. |

## Review limits

This is a provenance/graph/dependency review, focused inspection of integration
boundaries and correctness fixes, host compilation and CPU tests. DJ's base alone
changes hundreds of files and includes generated tables and captured fixtures;
this is not a claim that every inherited line has been independently audited.
No M3/Neo/M4/M5 hardware, physical GPU execution, compositor session, firmware
recovery, full Vulkan/OpenGL CTS or game compatibility was tested here. Source
preservation is not a substitute for those checks.


## Configurable geometry heap follow-up

Official [MR44931](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/44931),
`a2a5472089a029c067a2e3e35adc420643603065`, is adapted in `86e11a3ee2d` with
s0ullight's author credit. The option is `hk_geometry_heap_size_mb`, default
128 MiB, valid range 16–1024 MiB. The implementation keeps the current device
callback, one-time initialization, shader-stage warm-up and BO/map failure
handling. No allocator or native kernel ABI guard is replaced.

The upstream change adds no per-game profile. An explicit setting such as
`hk_geometry_heap_size_mb=256 your-game` is needed to opt in; this integration
does not establish that a game crash is fixed or that general FPS improves.
The production callback is exercised under ASan/UBSan by the public CPU checker,
including the default, both limits, the 256 MiB override and BO/map failures.
A fresh x86_64 build from this public source tree passed all 36 registered Asahi
tests. The generated option parser rejected invalid/out-of-range environment settings
in 11 direct parser checks. Adversarial review found XML values bypass that
parser range check; `db96168675f` clamps them at allocation to 16–1024 MiB
and statically bounds publication to the GPU u32 fields. The regression uses
the real `poly_heap` declaration. Twelve real XML-parser-to-callback checks
covered negative, zero, in-range and oversized values, including 4096/5000 MiB;
all allocated and published the same bounded size. Removing the clamp fails
the maintained regression. These are host checks, without GPU qualification.

## October 26 M3 Pro qualification

The last local packed-GS-output and select-guard fixes are already covered
by `d4b6840f` and `ae035d59`; no duplicate production patches are imported.
Jonathan Goyvaerts's public `141481c` warning/build fixes are replayed with
original attribution and `-x` provenance. They resolve Clang's enabled-test
build failures in the command-stream header.

Restore the conditional-rendering, unified-image-layouts, memory-budget and
sixteen-set profile on hardware-tested G15S only. Other G15 profiles and the
FMA/rounding and kernel/device admission checks retain their existing policy.
The actual-driver shim probe rejects the previous G15S profile and verifies
both the restored profile and the unchanged profiles.

[The J514S receipt](OMACOM-M3PRO-20261026.json) pins built source `eaf69705c3`,
Arch package/library hashes and the initial kernel. GPU readbacks through
set 15, 139 host suites, the sixteen-set graphics/compute group and
Settings/theme cycles pass.

The seven-case regression group has six passes and one intermittent failure
that passes retry. Geometry-shader output sometimes remains at the clear
colour; this was also observed with the earlier Mesa and kernel. The receipt
records a `uvec3_highp_geometry` failure; the rebased kernel `8824dd6` showed
the same symptom in `uint_highp_geometry`. This dropped-draw defect remains
unresolved and is not a clean pass or an established consequence of load.

All 384 conditional-rendering draw cases and all 60 clear-attachment cases
skip. Exposing conditional rendering does not qualify these skipped paths;
the current full suite will not validate paths it also skips. Skips and
warnings remain in the receipt. Full Vulkan 1.4 CTS on the rebased stack is
pending the owner's run; the earlier full result belongs to a different Mesa.

Newer shared-native candidates remain separate review work: bank-constrained
RA recovery (`c7d849e5`), passthrough-GS OOM reporting (`89bb9b98`), uniform
atomics (`52e35626`), application then internal compute DMA (`a29a074b`,
`43901729`), and occlusion availability (`e07644ef`, `360a1c91`). The blend
series `8fc7dc70` also touches shared native paths; its new constant-program
fast path requires G17P's qualified preamble. None is qualified on M3 by this
receipt. G17P-only timestamp/cache-range/signal optimizations, RT work and
virtio-only unbind handling are not direct-DRM M3 prerequisites.
