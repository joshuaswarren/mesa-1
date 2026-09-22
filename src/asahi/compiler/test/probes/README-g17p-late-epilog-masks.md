# Restore native sample operands after epilog preprocessing

The native CTS cases `pipeline.fast_linked_library.multisample.alpha_to_one`
at 2/4 samples failed fragment-epilog compilation. The captured 2x epilog
ended with two stores carrying literal masks `#1` and `#2`. The G17P store
protocol requires coverage-qualified register masks for these multisample
stores; the packer's refusal was correct.

The first `agx_nir_lower_g17p_tile_samples` ran before preprocessing, while
the masks still used the loop induction value. Preprocessing then unrolled
the blend's sample loop and exposed constants. The second lowering was
restricted to `g17p_emrt`, leaving ordinary epilogs with those literals.
Run that existing lowering after preprocessing for all native fragment
epilogs. It leaves already-lowered live operands unchanged and does not add
pixel transactions, GPU waits, or alter batching.

## Native evidence

The Apple compiler capture retained in `README-g17p-explicit-sample-count.md`
was freshly compiled for `applegpu_g17p` on 2026-09-16. Its 2x/4x fragments
write alpha one and demonstrate the live-coverage store contract:

- `24c21006`: `rld`, normalized SR39, destination half 4; exact compose-table
  round-trip `form 252 rld ep0 size4: 2:4 9:2 10:1 12:1 21:39 23:7 24:1 38:1`.
- `e71654000000044e08000008`: `lstcol`, sample operand half 4; exact round-trip
  `form 211 lstcol ep0 size12: 9:2 18:1 26:15 35:9 37:7 38:1 44:4 51:2
  52:1 55:1 65:1 66:1`. Field 44 is the register operand, as documented and
  tested in `test-agx-g17p-msaa-epilog.c`.

This fix reuses the existing lowering and witnessed per-sample coverage-AND
path; it does not introduce an instruction encoding. The retained native
per-sample witnesses are documented in `README-g17p-per-sample-fragment.md`.

Host regression: `asahi-g17p-msaa-epilog alpha-one` constructs the real
source-alpha/inverse-source-alpha blend with alpha-to-one, confirms that the
post-preprocess pass is necessary, and checks every 2x/4x emitted store uses
a register mask and one shared coverage read.

## Qualification

Neo, native Vulkan API, CTS 1.4.6.2, DRM SHA256
`c3080a3c4674bcf085927037d0869b613dd88962c3f0ca0c88a7b9a4d563f4cf`:

- `late-epilog-masks-alpha-family14-baseline`: 14 Fail on the previous candidate.
- `late-epilog-masks-alpha-family14-r1` and `-r2`: 14/14 Pass each, covering
  alpha-to-one at 2/4 samples under all seven pipeline construction modes.
- `late-epilog-masks-controls-r1`: 8/8 related minimum-sample-shading controls
  pass, including previously passing per-pixel MSAA.
- All runs report zero GPU fault markers and retain the desktop.
- Full `asahi-g17p-msaa-epilog`, `asahi-sample-count`, and
  `hk_shader_variants` host tests pass, as does the isolated `alpha-one` test.

Candidate `k4-late-epilog-masks-r1` library SHA256:
`f7aa789f9923eaea054d95ea73a4ab819c8ca0e1cbf27194f7e643f0f82aac8b`.
`late-epilog-masks-reject6-r1` confirms the two original alpha-to-one failures
pass while all four sample-rate depth-export epilog rejections remain. Do not
count those separate failures or the sample-rate A2C mismatches as resolved.
