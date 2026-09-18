# Seed 20260920 image format work

The frozen selection contains all 89 image-format/operation NotSupported cases
from the final 871 Pass / 129 NotSupported report. Original indices are in
selection.json; subset indices are in the new result files. CTS and its oracles
are unchanged.

RGB/BGR storage and scaled sampling: 20 Pass, 69 NotSupported, zero failures or
crashes with ten workers. GPU transfers retain API three-component byte layout;
optimal image storage is padded and sampled alpha is one. Scaled samples use
integer descriptors and float conversion; linear scaled filtering is not yet
advertised. Signed BGRA EOT uses the existing packed I8 word path.

Exact binary identities, guarded experiment path and timing are retained in
rgb-summary.json. Experiments used the 60-second hardware watchdog and mandatory
six-channel thermal guard. An initial newly enabled signed BGRA blit failed the
tile-layout allowlist; adding its I8 EOT path made that case and the complete
89-case subset pass or remain unsupported without failures.

Padded RGB attachments: 25 Pass, 64 NotSupported, zero failures or crashes.
The four RGB blend cases and BGR sRGB granularity now pass. Epilogs use the
logical API format for blending (including absent destination alpha), then
pad stores to the physical tile format. Physical layout and blend semantics
remain separate. See blend-results.json and blend-summary.json.

Packed 5:5:5:1: 37 Pass, 52 NotSupported, zero failures or crashes.
Vulkan R5G5B5A1/B5G5R5A1 map to low-alpha Gallium formats, which use native
channel layout 0x0d. Twelve more original cases pass, covering cube/array
sampling, mip filtering, blits and blending. Native PBE packing already
supports this channel layout; retain the generic component swizzles.

X8 D24: original case 428 now passes its shadow sampling oracle. Physical
depth uses float32, with the existing packed-D24 GPU transfer conversion.
Views, host-transfer restrictions, sample counts and depth bias follow the
existing D24S8 emulation. This is one-case evidence; a regression rerun follows.

Unsigned scaled filtering and packed scaled sampling: the full frozen 1,000-case
rerun is 932 Pass / 68 NotSupported / zero failures or crashes. All 871 previous
passes remain Pass; 61 of 89 selected format skips become Pass, leaving 28
format skips. Each case ran once in this final report with twenty workers,
30-second per-case bounds and the watchdog/thermal guards. See
filtering-regression-summary.json for exact identities, timing and evidence.

USCALED 8/16-bit textures filter through native UNORM descriptors and shader
rescaling. Descriptor metadata follows view swizzles and leaves absent/constant
channels unscaled. Custom borders use the independently bound image's scale;
gathers select the original border value. Packed 10-bit scaled sampling uses
integer descriptors; unsupported packed scaled stores remain unadvertised.
The direct SNORM10 descriptor experiment failed seven pixel oracles and was
removed; no SNORM10 feature is enabled by this change.

Historical CTS correction: ImageTest, ImageViewTest and SamplerTest inferred
integer shader types from mapVkFormat's integer backing representation of
USCALED/SSCALED. The captured original shader uses uint32 tex with a USCALED
view, contrary to Vulkan's numeric format / OpTypeFloat requirement:
https://docs.vulkan.org/spec/latest/chapters/formats.html#formats-numericformat
The patch changes only generated shader type selection; CPU backing data,
lookup scale/bias and pixel comparisons are unchanged. Keep the original CTS
binary/results and identify the corrected binary by scaled-cts-build.json.
No driver workaround for invalid integer-typed scaled sampling was added.

The initial border-query marker used NIR's gather-only component field and
asserted in original case 287. Its dedicated backend flag fixes the assertion;
case 287 and the complete final regression pass. Initial experiment evidence
is retained separately under build/cts-seed20-formats and _provenance/experiments.

Packed YCbCr: all four applicable original cases pass (35,314,643,798).
Original 542,603,921 now skip specifically because they request disjoint binding
on a single-plane format; do not advertise DISJOINT to satisfy them. Their three
existing non-disjoint companion cases also pass, recorded separately and not
counted as improvements to the frozen sample. This covers packed 8/10/12/16-bit
words, both luma placements, linear/optimal storage, full-range colour conversion,
nearest/linear filtering, midpoint/cosited chroma and swapped chroma mapping.

The driver preserves the packed bytes and reconstructs full-resolution samples
in GPU shaders. It advertises explicit reconstruction, uses raw UNORM word
fetches, and retains independent luma/chroma filtering. API size queries report
luma pixels; internal descriptors/queries measure packed words. Linear layouts
accept horizontal subsampling with unchanged packed-row stride calculations;
the host Linear.Packed422Rows test verifies automatic and explicit row pitches.
A second hardware chroma sampler is unnecessary for shader reconstruction and
would overflow a single-plane descriptor binding, so it is not allocated.

These are focused hardware results, not an application/video-player claim.
