# Explicit sample interpolation needs raster-count specialization

The native CTS case
`dEQP-VK.pipeline.fast_linked_library.misc.frag_lib_varying_samples_2`
failed on Mesa `46eeb53ec49`: its independently compiled fragment contained
`get_sr AGX_SR_SAMPLES_LOG2`, which has no established G17P encoding. The
shader does explicit sample interpolation but does not request sample shading.
The library's sample count is supplied by the eventual linked pipeline.

`agx_nir_lower_sample_intrinsics` clamps explicit sample interpolation to index
zero at one sample. That creates a count dependency even in a per-pixel shader.
Select the existing 1/2/4-sample variants for this dependency and constant-fold
`load_samples_log2_agx` from the selected key. Do not unroll the per-pixel body,
invent a special-register encoding, or change the existing fail-closed test.

## Apple compiler witness

Inputs: `g17p-explicit-sample-count.metal` and
`g17p-explicit-sample-count.mtlp-json` in this directory. Metal toolchain
v27.1.5194.15 (2026-09-16), targeting the actual G17P:

```sh
xcrun metal -c g17p-explicit-sample-count.metal -o /tmp/sample.air
xcrun metallib /tmp/sample.air -o /tmp/sample.metallib
xcrun metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
  -N g17p-explicit-sample-count.mtlp-json /tmp/sample.metallib \
  -o /tmp/sample.nat
```

Captured native object SHA256:
`3ff4464b893aaeb07dd632b8588fd5912c4305b2401765f9de219d26661104c4`.
Extract the nested `__GPU*,__fragment` object's `__TEXT,__text` with the
`sections()` reader in `capture-g17p-mrt-eot.py`:

| Raster samples | Text bytes | Text SHA256 |
| --- | ---: | --- |
| 1 | 174 | `a82f9f97b25156178c077fa9e26fddbe6a0fb643345d974ca7531b85dad66198` |
| 2 | 176 | `d588ce109aebd66523a3047e627fa91633b3d1429b38a86a8147bb55cdf2ce10` |
| 4 | 176 | `d588ce109aebd66523a3047e627fa91633b3d1429b38a86a8147bb55cdf2ce10` |

The second fragment uses a buffer-sourced sample index, matching the CTS's
runtime-indexed operation. Its text hashes are:

| Raster samples | Text bytes | Text SHA256 |
| --- | ---: | --- |
| 1 | 174 | `a82f9f97b25156178c077fa9e26fddbe6a0fb643345d974ca7531b85dad66198` |
| 2 | 178 | `61bfa64cd2a07e086220a792edebbe53d2fbb5a6f0eddba799e8797cbe8dc1c0` |
| 4 | 178 | `e794cc9f00964b64c3688ae1298f5e2bb68e551f30a8a06ba3731fce2ba413c3` |

The dynamic 2×/4× main programs (after the 64-byte constant program) are
byte-identical. They begin with the buffer/uniform-sourced sample copy
`03042700820001000000`, round-tripped as
`form 36 copy ep1 size10: 9:2 10:1 11:1 17:4 23:3 24:1 46:1`.

All six contain one `af04540203000a03` interpolation-coordinate instruction.
The in-tree compose-table inverse round-trips that byte vector exactly:
`form 200 itrcoords ep0 size8: 2:2 9:2 10:2 11:1 23:2 24:1 27:1 38:2`.
It is followed by one perspective interpolation chain, not an unrolled
per-sample shader. The 2× and 4× texts are identical. This witness supports
preserving per-pixel execution; it does not establish any samples-log2 SR.
Metal's out-of-range 1× sample index is not used as a Vulkan semantic oracle:
Mesa retains its existing single-sample index-zero lowering, tested separately.

Host coverage exercises the lowering at 1/2/4 samples, verifies zero versus
one as the resulting constant sample index, preserves runtime indices for
multisampling, removes all count-register reads, and checks that sample shading
remains disabled.

## Hardware qualification (2026-09-16)

On Neo with the qualified DRM module SHA256
`c3080a3c4674bcf085927037d0869b613dd88962c3f0ca0c88a7b9a4d563f4cf`,
native Vulkan API, CTS 1.4.6.2:

- `explicit-sample-count-baseline`: all four cases fail on Mesa `46eeb53ec49`.
- `explicit-sample-count-gate-r1` and `explicit-sample-count-gate-r2`: all four
  pass on each run with the fix. Cases are `frag_lib_varying_samples_2` and
  `_4`, under both `fast_linked_library` and `pipeline_library`.
- `explicit-sample-count-controls-r1`: 16/16 related offset/sample-position
  interpolation controls pass.
- `explicit-sample-count-direct-controls-r1`: 12/12 previously passing explicit
  sample interpolation controls pass, including single-sample behavior.
- Every run has zero kernel fault markers and retains a live Hyprland desktop.

Candidate `k4-explicit-sample-count-r1` library SHA256:
`a64985b1f88527df276f9b699b9c9511ae9b0b2f45df52e1484a09cc7ef8fc57`.
Runtime sources did not change after packaging; only witness documentation and
the host test's runtime-index matrix were extended. The focused host tests
`asahi-sample-count` and `hk_shader_variants` both pass. The broader
`asahi-g17p-early-demote-epilog` test reports a separate dynamic-depth assertion
(`masks=2 waits=1 releases=1 depth=1`); this change does not fix or waive it.
This qualification closes these four library cases, not full conformance.
