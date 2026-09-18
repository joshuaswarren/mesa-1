# Coalesce independent coverage tests before native colour stores

The no-depth sample-rate A2C epilog interleaved four `sample_mask_agx`
updates with four colour stores inside one G17P colour transaction. The exact
alpha-to-one reproducer reported a maximum alpha error of 0.74902 against
clear alpha 1; the colour channels agreed. Coalescing only pure kills did not transform this
epilog: these updates also run fixed-function tests on surviving samples.

The intrinsic operands are TARGET and LIVE, not two kill masks. For disjoint
targets, combine TARGET by union and LIVE by union of each `TARGET & LIVE`.
This preserves the per-sample tests and kills. Unioning unmasked LIVE is wrong.
Reject overlapping testing targets, mutable coverage reads and mixed testing
updates/depth exports. Pure kills retain the existing overlapping-target union.
Only constant/exported and invariant coordinate expressions are movable.

Host tests cover 2x/4x, constant/dynamic targets and all 256 exported masks
(1,024 combinations), plus overlapping-target and mutable-input rejection.
The no-depth epilog fixture now emits one coverage update rather than four.
The existing depth-export and MSAA epilog tests remain enabled.

## Apple compiler witness

Compile the adjacent `g17p-sample-alpha-colour.metal` and `.mtlp-json` with:

```sh
xcrun metal -c g17p-sample-alpha-colour.metal -o /tmp/colour.air
xcrun metallib /tmp/colour.air -o /tmp/colour.metallib
xcrun metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
  -N g17p-sample-alpha-colour.mtlp-json /tmp/colour.metallib \
  -o /tmp/colour.nat
```

Metal toolchain v27.1.5194.15, captured 2026-09-16. Object SHA256:
`df1c64c03f2f2be6c478d7a95f1ab01eb286fcf89010db4adca553c12c93eae7`.
Both fragment texts are 400 bytes; the 2x SHA256 is
`e0bd6c71477d6b4abedebcd367de033c43633e8644afa9a19acd1627830db2d3`
and the 4x SHA256 is
`ee2f5b6b8eababd67929c3346c9a0915f619c86bbe79fed989819679555393f0`.

Both contain combined class-13 PIXWAIT `8712540d0800` at 0x5a,
MOVMSK `571454000001` at 0x124, LSTCOL
`e7065404000006ce08000008` at 0x152, and a backward branch at 0x176.
This independently witnesses native sample-dependent alpha coverage without
depth export. Apple uses a loop; this does not prove that Mesa's split,
statically unrolled epilog must have the same transaction structure. No new
instruction encoding or guessed control field is introduced here.

## Hardware qualification

Neo native Vulkan API, CTS 1.4.6.2, DRM SHA256
`c3080a3c4674bcf085927037d0869b613dd88962c3f0ca0c88a7b9a4d563f4cf`.
Candidate `k4-nodepth-mask-coalesce-r1` library SHA256
`56b9e472c8f5ee282039f71eadc3115c0154b9f78b966350dbf70cffa0f4a50b`:

- `nodepth-mask-coalesce-family21-baseline`: 9 Fail / 12 Pass on parent
  `862b38ad462`. Candidate runs `-r1` and `-r3`: 21/21 Pass each.
  Run `-r2` was interrupted by a separate reboot and is not qualification.
- `nodepth-mask-coalesce-sample-rate14-baseline`: 14 Fail on the parent.
  Candidate `-r1` and `-r2`: 14/14 Pass each.
- `nodepth-mask-coalesce-depth28-r1`: 28/28 Pass.
- `nodepth-mask-coalesce-alpha-one14-r1`: 14/14 Pass.
- Every completed gate above has zero GPU fault markers and a live desktop
  at completion. Later unrelated interruptions are not counted as passes.

This fixes 23 confirmed failures across these two families. It is not full
Vulkan qualification; pixel-centre interpolation and wider parity work remain.
