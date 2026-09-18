# Vulkan 1.0 random baseline — 2026-09-13

100 cases selected before execution from233795 must-pass entries with Python
Random(20260913). No exclusions/redraws. Original outcomes:35 Pass,19 Fail,
41 NotSupported,5 userspace crashes,0 timeouts. Results and exact names are in
results.json/cases.txt; original logs remain at the listed workspace evidence
paths and on the target. Unsupported results are not passes. This is a sample,
not conformance. Driver:Apple M4(G16G B0), native Asahi Vulkan, library
19afd1a56957886e18ecbaf88f63b473700c69a5cbb3011df0227dbc783369bb.
The four additional info tests passed, independently of the100 sample cases.

All batches completed with a60-second watchdog lease and six-channel thermal
monitoring. Highest sample temperature58.75 C. No GPU hang/watchdog reset.
The existing DCP idle fault occurred at uptime3610 before CTS(3999); the render
GPU continued to retire jobs. A guarded reboot subsequently recovered display
state; default SDDM remained enabled/active.

Observed failures include compiler refusals(imad, texture_sample, texture_load,
image_write, ld_tile), multi-plane image handling,16-bit shader inputs and one
texel-view pixel mismatch. These are observations, not a blanket kernel blame.
Case45's render-pass abort also crashes under lavapipe(control
20260913T104404951360Z-cts-1-0-old-renderpass-control). Pinned CTS returns the
input-aspect vector by value then passes .data() from that temporary after its
lifetime ends(vktRenderPassTests.cpp:680,1190). Keep this historical test bug
separate from driver defects; the original baseline test code is unchanged.

First fix:seed20260913 drawn again from the19 Fail cases selects index10,
integer cube sampling with explicit tiny LOD. Fresh hardware run
20260913T104946283939Z-cts-cube-fix passes against CTS's reference image.
See compiler/test/probes/README-g16g-cube-register-lod.md for native evidence.
Unit checks:6262 texture-core checks,33 focused optimizer/staging tests and
cube-lowering-only pass. The broader historical graphics suite has unrelated
stale expectations. ExplicitLodStagingAndLifetime fails both with this change
and at baseline HEAD3b3a3df7ce7; its baseline log is retained in
build/cts-fixes/baseline-explicit-lod-test.log. Do not claim the full unit suite
passes.

Second fix:continuing the same RNG without replacement selects index29,
`dEQP-VK.binding_model.shader_access.secondary_cmd_buf.uniform_texel_buffer.vertex.single_descriptor.offset_nonzero`.
Its vertex shader contains register multiply minus immediate1. Generalize the
existing immediate-multiplier lowering to register multipliers, emitting the
native pure multiply and subtract-immediate with modulo-2^32 semantics.
Original compiler bundle6a7a8a01caba15aa0ddfeb33673d8a69ecc0ae7c8bb4d892fd927970a2d20c70
replays successfully after the fix. End-to-end NIR regression:
`agx3_graphics_compile_test --imad-negative-only` passes. Native CTS run
20260913T105408152345Z-cts-negative-imad-fix passes, peak58 C, full guards,
libraryb1d2fbf56b331f4e9f3518648d5fdb3d533a5ca19621dc3b3cb379746dc04752.
Normal login and Foot restored after the earlier idle-display recovery:
/root/asahi-evidence/cts-restore-login-20260913T1054. Kernel unchanged.

Third selection:index42, geometry uniform texel buffers. The arithmetic fix
cleared its compiler rejection, exposing a Mesa command-recording refusal:
logical GEOMETRY was rejected despite the raster variant executing as hardware
VERTEX. The runtime also reconstructed root_uniform from logical stage alone,
losing the texture-root reservation. Retain the final compiler root address in
shader metadata and use the physical stage for indexed-sampler acceptance.
Bump shader binary version9->10; reject the preceding version in the cache test.

20260913T105823659292Z-cts-geometry-uniform-fix:all4 cases pass(indices42,1,10,29
in the original sample), peak57.5 C, full guards, no kernel changes.
Librarybc1d7cfc2b6e5d39e613cedfa42086c7379abf613469da23dfa37b72f33612e9
contains the root/stage fix before the cache-version-only bump. The version10
build was subsequently installed;2769 VDM decode/cache checks pass.

Fourth selection:index55 (vertex/fragment push-template uniform buffers) now
passes with the existing arithmetic fix; no separate source change was needed.
Evidence:20260913T105941993579Z-cts-push-template-check.

Fifth selection:index37, mutable RGBA8_UNORM/RGBA8_UINT draw/load. Add integer
RGBA8 to native bindless storage lowering and accept its U32 input lanes in
the indexed IMGWCI packer; the PBE supplies destination width. This reuses the
existing G16G integer-write witness, with no new instruction encoding.
`agx3_graphics_compile_test --integer-storage-only` passes. The historical
compiler bundle remains rejected because it was captured after the missing
NIR lowering; the fresh full-pipeline CTS run covers that transformation.
20260913T110524038553Z-cts-rgba8ui-native-store:Pass, peak57.5 C, full guards,
library1f144b9cc221f2e523edcb123b72ffd328ff16a6f3fd07ef101de44e2b031818.
Kernel unchanged.

Sixth selection:index43, three-plane 10-bit 444 to 420 optimal/linear copy.
The native GPU raw-copy path rejected plane aspect bits and linear endpoints.
Use the selected plane's format, address, dimensions and tiling/stride in
buffer/image and image/image transfers. Preserve compression and packed
depth/stencil exclusions. No CPU copy or kernel change.
20260913T110821361548Z-cts-plane-copy:original case43 Pass, peak57.75 C.
Library8ce7c7b0eebdf8664ccd0ccaaf681f2600c4f63e542a4ca45b30db94cd9afc98.

20260913T110850190124Z-cts-plane-copy-regression:original case66 also Pass,
plus linear/linear Pass. Three optimal-destination regressions exposed another
historical CTS bug:downloadImage transitions to TRANSFER_SRC_OPTIMAL but passes
TRANSFER_DST_OPTIMAL to CmdCopyImageToBuffer (vktYCbCrUtil.cpp:619). The driver
correctly rejects that invalid layout. Preserve the baseline binary and counts.
A separate diagnostic binary applies only ycbcr-read-layout.patch, SHA256
6a82de1157dfe922ab967cfa5b1404cddc0749d6b935d08321940919cbd1726d.
20260913T111025012776Z-cts-plane-copy-corrected-readback:all5 regressions Pass,
including linear/linear, linear/optimal, optimal/optimal, disjoint planes and
case66. This is corrected-test evidence, not five original-CTS passes.
Semantics checked against https://docs.vulkan.org/spec/latest/chapters/copies.html.

Seventh selection:index23, ETC2 punchthrough blocks viewed as RGBA16_SINT
attachments. Original output was uniformly black; the retained QPA/reference
and compact statistics are in 20260913T111116577746Z-cts-block-view-pixels and
build/cts-fixes/block-view-images. Diagnostic readback confirmed the upload
was byte-exact but the attachment output was zero before decompression
(20260913T111820243687Z-cts-block-view-source-readback). Successful shader bundles are retained at
20260913T111540483190Z-cts-block-view-compiler-inputs.
RGBA16_UINT and RGBA16_UNORM controls passed, isolating format-dependent EOT.

Apple G16G AIR-NT32023.917 builtin format8/components4 compiles to the ordinary
68-byte I16 EOT (selector0x15), unlike inherited G17P's84-byte special program.
Restrict the special program to G17P. Native input/archive/tool hashes:
build/cts-fixes/native-rgba16-sint-eot; tracked reproducible input and witness
are in compiler/test/probes/README-g16g-sint16-eot.md.
20260913T112057102686Z-cts-m4-sint16-eot:original case23 Pass, peak58 C,
library44d03aabcb5f82a67bffc58eb71f2cd514874355523c40a0d88804a7339db80e.
No CTS assertion, application shader, kernel or software-rendering fallback
changed. The original CTS source and binary have been restored after diagnostics.

Eighth selection:index4, RG8 sRGB blending. The backend refused two-channel
W8 tile loads despite already packing/unpacking their two bytes correctly.
Fresh Apple G16G RG8Unorm framebuffer-fetch compilation witnesses mask3/W8;
see compiler/test/probes/README-g16g-rg8-fetch.md. Admit that exact form.
All54 tile-core checks pass; original failed compiler bundle101de234... replays
successfully (412bytes). 20260913T112410307569Z-cts-rg8-tile-load:original CTS
case4 Pass (result image matches reference), peak58.5 C, full guards.
Libraryb898fc1ea3acae2dc693383e185319f4cfc63030573efc3cbf0c7f65b1880257.

Ninth selection:index48, integer-array gather with multiple negative offsets
and swizzles. Enable the witnessed array/offset combination in native resource
lowering, immediate translation, IR width validation and instruction packing.
Native evidence:compiler/test/probes/README-g16g-array-gather.md. All6264
texture-core checks and end-to-end --array-gather-only compile test pass.
The first attempt stopped in userspace IR validation (20260913T112854797486Z-
cts-array-gather-offset); it was not a GPU crash. After fixing immediate width,
20260913T113102546620Z-cts-array-gather-validated:original case48 Pass,
peak58.75 C, full guards. Library
73b5d0efd21c9db27d01c6de4ef7fe89bc0b3ced85a84c7eb38ae6609249197f.

Tenth selection:index66 already passed with the plane-copy fix, using the
original test (20260913T110850190124Z-cts-plane-copy-regression, local index4).
Continue to original index19; no extra source change needed for66.

Eleventh selection:index19, extended compressed-block texel views with a
runtime compute mip level and signed RGBA16 store. Allow the existing native
integer-LOD fetch in compute, lower signed RGBA16 storage descriptors and emit
integer U32 input lanes with the PBE's signed16 destination format. Native
witness:compiler/test/probes/README-g16g-integer-copy-lod.md.
--compute-lod-store-only passes through the full NIR/IR/native compiler path.
20260913T113435091246Z-cts-compute-lod-sint16-store:original case19 Pass,
peak58.25 C, full guards, libraryc99775a2887dc8720b8d3a920f5f53daadc44c8f0d9a4527547534345ddcf060. Kernel unchanged.

Twelfth selection:index11, signed cube-array view with base mip and restricted
array range. Preserve native cube-array descriptors on G16G and project XYZ
while retaining the cube index. Pack index+face with the native combined
auxiliary form. Witness:compiler/test/probes/README-g16g-cube-array.md.
6268 texture-core checks, --cube-array-only and existing --cube-sample-lowering-only
pass. 20260913T114027179419Z-cts-native-cube-array:original case11 Pass,
result image matches reference, peak58.5 C, full guards.
Library1f85037d07176963e0a14881d7edacf2b15768a380217351245821cf5082b59f. Kernel unchanged.

Thirteenth selection:index91, 1D R32_UINT compute image barrier with indirect
read dispatch. Both direct-dispatch controls also returned zeros. Upload and
readback alone were byte-exact; constant/unrolled diagnostic shaders still
returned zero, excluding the initial loop-immediate hypothesis. The descriptor
writer's native view-type gate omitted 1D and 1D arrays, replacing otherwise
valid native descriptors with nulls. Admit their existing height-one 2D mapping.
20260913T115520513673Z-cts-storage-1d-descriptor: original case91 Pass,
peak58.75 C, full guards. 20260913T115544783157Z-cts-storage-1d-regression:
three direct read/write controls and image.load_store.1d_array.r32_uint Pass.
Libraryaf002684817557d9a155229cfdf174e3f6f4d235f0a572ed44b4a5ad74f0dba8.
Diagnostic CTS patches/binaries are separate; original source and executable
restored to original SHA d296ca5f77238367933e88b99babefe050e841d3c7e1fbcb6e95e4d424814b06.
Kernel unchanged; no synchronization barrier or test assertion weakened.

Fourteenth selection:index6, 10-bit YCbCr reconstruction. Plane uploads now
succeed but R10X6 samples were normalized as R16 (256*64/65535 = 0.250004,
expected256/1023 = 0.250244). Scale real channels of converted 10/12-bit planes
by65535/((2^bits-1)*2^(16-bits)) before range/matrix conversion. Preserve
implicit alpha/default channels and ordinary8/16-bit planes.
20260913T115855778906Z-cts-ycbcr-plane-normalization:original case6 Pass,
peak58.75 C, full guards. Library
850bd7084452c9ee353eb37626eead4cb8826ecc2a4f24b4c82f6a826b3127b1.
20260913T115924165455Z-cts-ycbcr-scale-controls:equivalent12-bit case Pass.
The16-bit control selects a different vertex-stage path and hits a saved
texture_sample compiler refusal (bundle6cfe6852...); not a normalization pass.
Next random selection,index5, gets past its old upload failure and now reports
NotSupported:tessellation shaders unavailable. Do not count it as passing.
Standalone padded-format views without YCbCr conversion are outside this
lowering; their format emulation still needs qualification. Kernel unchanged.

Sixteenth selection:index95 already passes with the earlier register multiply
minus-byte fix:20260913T120014438276Z-cts-mutable-array-sampler. No new fix.
Seventeenth selection:index54, geometry cube-array descriptor arrays/base mip.
The physical vertex shader was barred from all sampled textures, even explicit
LOD operations without derivatives. Enable explicit-LOD resource lowering and
backend emission, apply cube projection in that stage, and include cube arrays
in fixed auxiliary staging accounting. Saved lowered shader10e026e3... compiles
offline to5922bytes. 20260913T120311060543Z-cts-vertex-level-native:original54
Pass, peak58.75 C, full guards. Library
46f0d6289b4bf348edab0a3f2fc2fba5991771f048ab7235c9ff86ee051e7f17.
20260913T120336440823Z-cts-vertex-sample-controls:16-bit YCbCr control now Pass;
next selection,index47, mutable geometry2D sampler also Pass. Kernel unchanged.

Nineteenth selection:index82, BC4 blocks viewed as RG32_FLOAT. Add the missing
RG32_FLOAT entries alongside R32_FLOAT/RGBA32_FLOAT in indexed storage lowering
and the existing F32-lane image-write path. The PBE supplies the channel count.
20260913T120507843955Z-cts-rg32f-native-store:original case82 Pass, full guards,
library0b18082c2c0056f0e63676d0c4140c909bc879bcaabd473293ad5d8a68e3a41d.
All19 original Fail cases have been visited in the recorded random order:
18 now Pass, index5 now correctly reaches the tessellation NotSupported check.
This is cumulative targeted evidence, not yet a fresh complete100-case result.
Five original userspace crashes remain to revisit separately. Kernel unchanged.

Crash revisit order (same repeated-choice method/seed):16,46,45,67,62,
recorded in build/cts-fixes/random-crash-order.json. Index16 now Pass with raw
plane copies:20260913T120621411847Z-cts-former-plane-copy-crash.
Index46 asserted on16-bit vertex input gathering. Retain the prolog's32-bit
attribute lane ABI, gather16-bit consumers and narrow after import (float
conversion for float inputs, low16bits for integers). Direct VBO lowering
already handles destination width. 20260913T120749746033Z-cts-16bit-vertex-input:
original46 Pass. 20260913T120817110382Z-cts-vs16-input-controls:signed input Pass;
the32-to16 RTZ output control reaches a separate half-register subtract refusal,
bundle d206ca5f... retained. 20260913T120840923012Z-cts-vs16-float-input:
actual16-to32 floating input Pass. Library
b964dc21b9f0ec9422f1560b8299e22cbbf30da9e8574b6faf0a3cb77c858c02.
Full watchdog/thermal guards, unchanged original CTS and kernel.

Crash67 now Pass with raw disjoint plane copies (experiment ending
cts-former-disjoint-copy-crash). Crash62 reaches the next16-bit boundary:
vertex-to-geometry memory stores asserted32 bits and loads used packed16-bit
strides. Preserve one32-bit word per component on both sides, zero-extending
raw16-bit payloads on store and truncating on load without float interpretation.
20260913T121054004036Z-cts-geometry-word-stride:original62 Pass.
20260913T121120925505Z-cts-gs16-payload-controls:half-float and integer-vector
geometry input controls Pass. Full guards, library
938a27c0badf79fae93d55e31a24be9f40ae3268258281b5dc16ad33b0d759f3.
All four original Mesa-crash cases now Pass in targeted tests. Crash45 remains
the independently reproduced CTS input-aspect lifetime bug. Kernel unchanged.

Fresh original100-case rerun at Mesa5af5a339be9, unchanged CTS binary and
kernel:57Pass,42NotSupported,0Fail,1Crash. All35 original passes retained.
No GPU timeout/hang/watchdog reset; peak59.75 C across four fully guarded batches.
Exact results and evidence IDs:fixed-results.json. The surviving crash45 is
historical CTS code:besides the dangling vector, its aspect reference names
subpass0 (no inputs) instead of subpass1. The separate tracked
input-aspect-lifetime.patch fixes both color-case issues, without changing
rendering or comparison. 20260913T121336206915Z-cts-input-aspect-subpass-fixed:
case45 Pass using diagnostic binary586df4c94f4b4a29525c9bf44500325a75c14018f96cae567c8163d3db5a3a81.
The original CTS source/binary is restored; keep this adapted result separate.
New cts-campaign.py automates bounded consecutive batches with continuity,
identity and guard checks; it does not retry tests or recover hardware.

Follow-up control:input_output_float_32_to_16.scalar0_rtz_vert. Add native
16-bit integer subtraction (Apple G16G witness at
compiler/test/probes/README-g16g-half-isub.md). Exact-byte core regression and
existing ALU checks pass; saved failed shader d206ca5f... replays to360bytes.
The next boundary was flat16-bit fragment input:recover the consumer type
from the32-bit varying ABI, matching existing UVS producer widening.
20260913T122111624150Z-cts-flat16-rtz:unmodified control Pass.
20260913T122141687798Z-cts-flat16-controls:scalar7 RTZ and scalar0 RTE Pass;
integer-vector control reaches a separate block_image_store refusal, bundle
1a8fccb9... retained. Full guards; installed library
046aa4dbd7aad5b2371ce9e5daebc42917d53ed6de57571c943c6aaffdf4c7e0.
This follow-up is outside the fixed100-case total. Kernel unchanged.

Next random sample:20 cases, seed20260914, same original233795-case population
and sampling method. Selection retained in build/cts-fixes/sample-20260914.
20260913T122235277224Z-cts-next-20-000:11Pass,6NotSupported,3Fail,0Crash.
Repeated random failure order12,19,3 retained in next-20-fix-order.json.
First selection12 was a cross-instance3D copy rejected by the2D graphics-copy
path. Use raw GPU copies for1D and3D images, preserving per-mip/per-slice layout
and validating source/destination extents. 20260913T122416700043Z-
cts-cross-instance-3d-copy:Pass, peak59.25 C. Earlier1D image-copy control also
Pass:20260913T122512368139Z-cts-raw-1d-image-copy. Full guards, library
3c8e213536fa89bf3c49df4730caee81c3721389a8b6c0c835f1180092e146d6.
Copying between different image types remains a separate unsupported path.
Kernel unchanged; copy data stays on the GPU.

Random sample selection19: admit RGBA8 signed storage writes through the
native I8 write operation, with channel layout supplied by the PBE descriptor.
20260913T122617830487Z-cts-rgba8-signed-store: original CTS Pass,
peak59.25 C, full watchdog and thermal guards. Library
534d34a2ab0705a3ef3e3daca4f291fb86927fd28febea9baf67fcb1cc5dae59.
Kernel unchanged.

Random selection3: lower storage image reads to native indexed texel fetches
in graphics stages as well as compute. The descriptor, coordinate and memory
access checks are unchanged. 20260913T123103445244Z-cts-vertex-storage-load:
original vertex3D descriptor-array test Pass. 20260913T123119010060Z-
cts-storage-graphics-controls: matching fragment and vertex+fragment tests Pass.
Full watchdog/thermal guards, library
68cc4bbe6f9e478c7428accadeb1346d1c6671bf9224af91dab9da32fdd5e76d
(before whitespace cleanup). Kernel unchanged.

The complete second sample rerun (seed 20260914) is 14 Pass, 6 NotSupported,
zero failures or crashes: 20260913T123153996441Z-cts-next20-fixed-000.

Third random sample: seed 20260915, 20 cases, same population and method.
20260913T123207376203Z-cts-third20-000: 9 Pass, 9 NotSupported, 2 Fail.
Random failure order: 9, 7. Selection and results retained under
build/cts-fixes/sample-20260915 and third20.

Selection 9: enable native indexed storage writes in physical vertex shaders
(including Poly's geometry raster shader) and the R16 UINT typed write format.
20260913T123329455139Z-cts-geometry-storage-write: original CTS Pass.
20260913T123351835629Z-cts-storage-write-stage-controls: equivalent vertex and compute tests Pass.
Full watchdog/thermal guards, library
4d360cdda4be1b14c1180256edd2b692bbabf1ba988822c407449ddd460bcf80.

Selection 7 is the already documented historical CTS read-layout typo in
vktYCbCrUtil.cpp:623. The original binary remains Fail. The separately repaired
binary passes: 20260913T123413913719Z-cts-third20-corrected-ycbcr.
Do not count this adapted result as an original CTS pass. Kernel unchanged.

Fourth sample: seed 20260916, 20 cases, same population/method.
20260913T123458321106Z-cts-fourth20-000: 13 Pass, 4 NotSupported, 3 Fail.
Failure order 2, 1, 8. Selection and results retained in sample-20260916/fourth20.

Selection 2: storage cubes and cube arrays use integer face coordinates with
2D array read/PBE descriptors. Admit those views and lower loads/stores with
that dimensionality while retaining format/access and layer bounds checks.
20260913T123638923984Z-cts-cube-storage-array: original test Pass.
20260913T123726566876Z-cts-storage-cube-control: non-array cube control Pass.
Final library edc41bae6143f237caadb9c8f033327b69cc6300883031e320c30e23bab3a381.
Full watchdog and thermal guards; kernel unchanged.

Fourth sample selection 1: use raw GPU copies for single-sample color format
reinterpretation. CopyImage need not render either source/destination format.
20260913T123808979318Z-cts-format-reinterpret-copy: original packed-format
case Pass. 20260913T123837254329Z-cts-format-reinterpret-reverse: reverse Pass.
Final library 6358680915359a449e756e3d1058fa7cd6f7b3468aa4ceb00acbb97005a69d45.
Full guards, kernel unchanged. Depth/stencil and multisample selection retained.

Fourth sample selection 8: nonzero explicit 3D mip levels use the existing
native scaled immediate field. Apple G16G witness and exact-byte test are in
compiler/test/probes/README-g16g-3d-level.md. Texture core: 6270 checks pass.
20260913T124006017131Z-cts-3d-level-signed: original CTS Pass.
Full fourth sample rerun: {'Pass': 16, 'NotSupported': 4}. Evidence: 20260913T124031783614Z-cts-fourth20-fixed-000.
Library 32e11d82830ae213a9365b393a467d47647cc885127d36a74f741bcaa18c0547.
Full guards; kernel unchanged. Runtime 3D LOD remains a separate boundary.

Fifth sample: seed 20260917, 20 cases, same population/method.
20260913T124112588589Z-cts-fifth20-000 stopped at the known CTS input-aspect
lifetime assertion (case 0, r5g6b5 input attachment). The corrected separate
binary passes (experiment ending cts-fifth20-corrected-input-aspect).
Remaining 19 cases, experiment ending cts-fifth20-rest-001: 11 Pass,
5 NotSupported, 3 Fail. Failure order 4, 2, 16. Cases 4 and 16 reproduce the
known CTS download layout typo; both corrected controls pass (experiment ending
cts-fifth-ycbcr-corrected). Keep original failures/crash in baseline results.

Selection 2: native adjacency draws were rejected before drawing (black image),
not corrupted by GPU execution. Reuse existing GPU passthrough geometry
lowering to remove adjacency vertices. Compile the required VS compute variant
for adjacency or dynamic topology and invalidate passthrough state on topology
changes. No CPU vertex processing. The first diagnostic exposed the missing
producer variant; its empty raster shader was correctly rejected.
20260913T124641378535Z-cts-adjacency-variant: original CTS Pass.
20260913T124702175898Z-cts-adjacency-controls: all eight indexed/nonindexed line/triangle list/strip
adjacency controls Pass. Full guards, library
0574966263f47c9fa3dd7350f2906e45562396c7a193434c05739e3a201f1136.
Kernel unchanged.

Regression checkpoint after bbb02bab89b: the complete original 100-case sample
still reports 57 Pass, 42 NotSupported, 1 Crash, zero Fail. Exact results and
evidence batches: regression-results.json. The surviving crash is the known
historical CTS input-aspect bug; no tests or expectations were changed.
All cases ran on boot 99ddcbd0-1a1d-4f25-9fc0-e7a213acfad8 with mandatory
watchdog/thermal guards. Peak 60.00 C, no GPU hangs or watchdog resets.
SDDM remains enabled and active. Kernel source unchanged at 0df60d539ba3.

Format capability work: native raw transfers now admit plain power-of-two
color texels (up to 16 bytes, channels up to 32 bits) independently of sampling
and attachment support. Same-format copies also select the raw GPU path when
there is no texture descriptor. Existing GPU packed clears need no format ISA.
20260913T152751670371Z-cts-new-packed-transfers: original previously skipped
copy cases 13, 27, 49, 60 now Pass. Library
72396cba265cd2a95d430a9fa3577880c4205f323e8f855a631cebba30527150
adds the final 32-bit channel limit (initial test library in experiment manifest).

The original scaled-format clear tests use the wrong VkClearColorValue union
member. Scaled types require floating-point values per
https://docs.vulkan.org/refpages/latest/refpages/source/VkClearColorValue.html .
The retained scaled-clear-union.patch converts only the command's clear value;
reference/readback comparisons remain intact. All nine 1D/2D/3D corrected
controls pass (experiment ending cts-scaled-clear-corrected). These are adapted
controls, not original CTS passes. Corrected diagnostic executable SHA256:
9e4acc7d384cc9dca547fe0968f2ffd5ea0acf3e2bf27749ced3e248784efb32.
Original CTS source/binary restored. Full watchdog/thermal guards; kernel unchanged.

Scaled sampled images: native 8/16-bit integer descriptors plus GPU conversion
to float implement signed/unsigned scaled formats. A reserved descriptor-tail
word selects conversion at runtime, preserving descriptor updates and arrays.
The shader binary ABI is now 11 so old binaries cannot omit the conversion.
Linear filtering and scaled attachments remain unadvertised. Float texture
shaders now read this descriptor metadata; performance impact is not measured.
20260913T153520366551Z-cts-scaled-image-sampling: original cases 38 and 96 Pass.
Unsigned scalar/RG and signed RGBA16 controls all Pass (experiment ending
cts-scaled-sampler-controls). Library
353adcd7c87b10be333d246d450f2816ed85936f36a42b1414dd6533d02f415a.
Fresh original 100 cases: 64 Pass, 35 NotSupported, 1 known CTS Crash, zero Fail.
Seven former skips now Pass: 13, 27, 38, 49, 56, 60, 96. Exact results/evidence:
format-capability-results.json. Full watchdog/thermal guards; kernel unchanged.
