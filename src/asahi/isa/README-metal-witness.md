# Getting Apple's own G17P encoding for any shader shape

Apple's Metal toolchain is installed on the development Mac and targets our exact
part directly.  This turns ISA questions from inference into witness: write MSL,
get Apple's byte-exact `applegpu_g17p` machine code, diff against ours.

## Required compiler-change gate

Use this workflow for every AGX3B change that can affect instruction selection,
operand packing, staging, register allocation, scheduling, or program ABI.  It
is also required before adding a G16G/G17P chip gate or accepting a changed
G17P byte golden.  A host test proves only what our compiler emitted; it is not
an architecture oracle.

The default rule is that G16G and G17P share the AGX3B compiler path.  Start
with one common implementation.  Keep a chip-specific branch only when an
Apple-compiler witness, captured hardware result, or device limit proves a
real difference.  Resource counts and command-stream plumbing may differ even
when the ISA form does not.

For each fix:

1. Reduce the failing CTS or application shader to the smallest MSL program
   that preserves the disputed operation and its live operands.
2. Compile it with `metal-nt -arch applegpu_g17p`.  Compile an M4/G16G witness
   too when its native target is available.
3. Extract the embedded Mach-O `__TEXT,__text` or stage section.  Decode every
   relevant instruction and record raw bytes plus the re-composed field vector.
4. Compile the equivalent NIR with Mesa and diff instruction form, operands,
   register domains, staging order, and waits.  Do not compare only total size
   or a hash.
5. Keep the MSL, `.mtlp-json`, toolchain version, exact command, extracted text
   hash, and conclusion under `src/asahi/compiler/test/probes/` for a permanent
   rule.  Temporary investigations may begin under `.tmp`, but the evidence
   that justifies production code must move into the tree.

The 2026-09-16 Eryk-tail integration is the model.  A G17P half-uniform clamp
emitted tied `fsel` operations whose compared and selected register fields were
identical, validating per-instruction uniform-copy reuse on both G16G and G17P.
A `texturecube_array` sample with register LOD emitted the fused `PACK_SMPIR`
layer/face/LOD form with its seven-bit register field, validating the shared
low-128 allocator constraint.  The former changed exactly one G17P background
clear golden from 442 bytes to 394; the four sibling programs stayed
byte-identical.  The native witness justified updating that golden instead of
adding a G17P exception.

## Toolchain

    MT=$(dirname $(xcrun -f metal))     # Metal.xctoolchain/usr/bin
    "$MT/metal-nt" -archs               # applegpu_g17p is in the list

`applegpu-nt` and `metal-nt` are the same AIR Native Translator driver.

## Harness

`/tmp/g17p-re/g17probe/gen.py` does MSL -> AIR -> metallib -> native pipeline and
extracts `__text`.  Copy it somewhere durable before relying on it.

    import sys; sys.path.insert(0, ".../g17probe")
    from gen import build
    build(name, msl, pipeline, outdir, arch="applegpu_g17p", kind="compute")

### The pipeline-script shape (this is the part that costs an hour to rediscover)

`metal-nt` requires `-N <script>`; without it you get
`error: cannot translate functions in '<lib>'  note: specify a pipeline script`.
The script is `.mtlp-json`.  For a COMPUTE pipeline the entry is:

    {"compute_function": "<kernel name>"}      # a bare STRING, not an object

`{"compute_function": {"library": "lib0", "name": ...}}` and adding `"library"`
or `"label"` siblings are all parse errors.  Render pipelines use the
`render_pipelines` key instead of `compute_pipelines`, with each entry a
`{"vertex_function": "<name>", "fragment_function": "<name>"}` pair of bare
strings.

#### Verified working invocation, MetalToolchain v27.1.5194.15 (2026-09-10)

The entry above is only the innermost object.  It must be wrapped in a
`pipelines` table, and three other things bite:

    printf '{"pipelines":{"compute_pipelines":[{"compute_function":"K"}]}}' \
        > p.mtlp-json                          # the .mtlp-json SUFFIX is load-bearing:
                                               # any other name -> "malformed flatbuffer magic"
    xcrun metal -c w.metal -o w.air
    xcrun metallib w.air -o w.metallib
    "$MT/metal-nt" -arch applegpu_g17p \
        -platform_version macos 27.0 27.0 \     # without this the target is AIR 2.5/2.8
        -N p.mtlp-json w.metallib -o out.bin    # and metallib is AIR 2.9 -> refused

* A bare top-level `{"compute_function": ...}` or `{"compute_pipelines": ...}`
  is `unknown field`; `{"pipelines":[...]}` is `wrong number of unnamed fields
  in table vector`.  `pipelines` is a TABLE, not a vector -- use `{`, not `[`.
  The four keys inside it are `compute_pipelines`, `render_pipelines`,
  `tile_render_pipelines`, `mesh_render_pipelines` (strings recoverable from
  `usr/metal/<n>/bin/air-pack`).
* `-std=metalN.M` does NOT lower the emitted AIR version; only
  `-platform_version` moves the target.  macos 26.0 gives AIR 2.8, which is
  still too old; 27.0 works.
* The output is a `Mach-O 64-bit GPU executable applegpu` whose code lives in
  `__TEXT,__compute` for a compute pipeline -- NOT `__text`.  Extract with
  `otool -s __TEXT __compute out.bin`.
* `xcrun metal -arch applegpu_g17p -c` looks like a shortcut but dead-ends in
  `Plugin interface not implemented: AIRNTEmitObject`.

### A render pipeline's fragment shader is dead code unless it has a side effect

`metal-nt` compiles a render pipeline with no attachment information, so a
`fragment float4` whose result goes only to a colour output is optimised away
entirely: the fragment object comes back as an 80-byte stub and every variant
you are trying to tell apart is byte-identical.  Declare the entry
`fragment void` and store the value to a `device` buffer instead.  With that
one change, `t.sample(s, uv)`, `bias(k)` and `bias(reg)` differ exactly where
they should.  Sampling with an implicit LOD is also illegal in a compute
kernel -- `metal-nt` aborts with `LLVM ERROR: Failed to encode` -- so bias
witnesses have to come from the render path.

## Turning bytes back into fields

`.tmp/g17p-fcmpsel-evidence/tools/decomp.c` is the inverse of the encoder: give
it a hex instruction and it enumerates every AGX3b/AGX3c form of that length,
walks the form's choice-group alternatives, derives the field vector the target
bytes imply (predicates pin selectors, direct deposits recover operands, LUT
deposits are inverted by enumeration), and then **re-composes and compares**.
Only exact round-trips are printed, so a reported field vector is a fact about
the encoder rather than a reading of a disassembler's field names -- which is
the failure mode the retraction in `README-agx3-packer-limits.md` describes.
`scan.py` next to it walks a whole code image that way, one instruction at a
time; `bind.c` composes a field vector back to bytes.

Build them against the in-tree tables:

    cc -O2 -I src/asahi/compiler/agx3_compose -o /tmp/decomp \
       .tmp/g17p-fcmpsel-evidence/tools/decomp.c \
       src/asahi/compiler/agx3_compose/agx3_compose.c \
       src/asahi/compiler/agx3_compose/agx3{b,c}_tables.c

The in-tree Python disassembler (`isa/test/disasm.py`) is not a substitute: on
the six-byte `fsel` above it matched the wrong form, reported the wrong length,
and asserted inside its operand mapper.

## Reading the output

Code starts after the container's leading blocks; find `__text` (gen.py does).
The first 0x40 bytes before the entry are the constant program (`0e 00 00 00`
then `06 00` filler) -- the same code-container layout the driver builds.

## Standing hazard: field positions differ per instruction FORM

The XML field *names* and bit positions are per-instruction.  Confirmed traps:

    base kind :  lddev = bit 101      stdev = bit 99
    index kind:  stdev = bit 100
    access kind: lddev = f75 @97:99   stdev = f73 @95:97
    16-bit off : lddev = f76 @77:92   stdev = f74 @75:90

`lddev.f75` and `stdev.f75` are DIFFERENT PHYSICAL FIELDS.  A decode rule pinned
on one form does not transfer to the other.  Map each form separately, by
varying one field at a time from a known-good baseline through
`agx3_mem_encode_load/store` or `agx3_compose`.

Mapped, both forms: wait mask = bits 12..17 (wait slots 1..6);
lddev signal slot = bits 70..72; lddev uniform base = byte4 = base/4.

## Corpus

- `src/asahi/compiler/test/agx3_compose_corpus_vectors.c` -- 1,494 real Apple
  vectors with field values, incl. 78 `lddev` / 16 `stdev`.  Use one as a
  baseline: an all-zero field vector does not satisfy the form's predicates.
- m1n1 `proxyclient/experiments/g17p_grid_helper_code.py` -- a captured 16 KiB
  Apple code image, 8 shaders, all of which disassemble cleanly to `end`.

## Witness: the multisample texel-fetch sample index (2026-09-10)

Question: does AGX3 accept a per-sample operand on a texture fetch, or is
multisample sampling simply absent on G17P?  Answer: it accepts one.

MSL `texture2d_ms<float, access::read> t; out[0] = t.read(gid, N);` compiled for
`applegpu_g17p`, one constant N per kernel, plus a `texture2d` single-sample
control and a register-sourced index.  Round-tripped through `decomp`:

    N=0   051700000000   form 305 smp ep0 size6 : ... 119:3 132:3 133:1
    N=1   051700000200   form 305 smp ep0 size6 : ... 119:3 126:2 132:3 133:1

The two differ in exactly ONE byte and the field vectors differ in exactly one
field.  **Field 126 carries the sample index, scaled by two** -- absent (0) at
N=0, `126:2` at N=1.  That 2x is the ordinary half-register operand scaling,
the same convention as `AGX_ABI_VIN_ATTRIB(i) = 2 * (8 + i)`.

Two things this does NOT say, deliberately:

* N=2, N=3 and the register-sourced index select a LONGER form (`smp ep1`,
  size 8; `decomp` labels it form 360).  Its sample field was not isolated --
  the standing per-form hazard above applies, so do not assume it is also 126.
* Our compiler does not use this operand at all.  `agx_emit_image_load()`
  packs the sample into the coordinate vector instead -- `agx_emit_collect_to`
  of `{ms_index, layer}` into an extra 32-bit coordinate component
  (agx_compile.c) -- and `agx_nir_lower_g17p_indexed_samplers` admits exactly
  that shape.  That path is already in production for `vkCmdResolveImage` and
  is covered by `test_indexed_multisample_fetch()`.  Both encodings evidently
  work; this witness is a second, independent way in, not a correction.

## Witness: the end-of-tile store CAN keep the samples (2026-09-10)

Question: is G17P's end-of-tile store pinned to a 2D target -- i.e. can a
multisample colour surface never be stored unresolved, so a multisample image
can be created but never populated or read back? Answer: no. Apple's own
builtin end-of-tile factory emits a second program per selector that stores the
tile unresolved.

`test/probes/g17p-native-rt0-eot.btnf.hex` is a flatbuffer whose root field 2 =
5 selects that factory. Its one attachment table lives at file offset 0x70 with
seven single-byte fields at 0x74..0x7a:

    0x74 partition   0x75 internal format   0x76 texture type
    0x77 (unused)    0x78 extra flags       0x79 components   0x7a (unused)

Field 0x76 is the `Texture dimension` enum from `genxml/cmdbuf.xml`. Poking one
byte and rerunning `metal-nt` walks it:

    0 -> 48 bytes (1D)          4 -> 68 bytes (2D MULTISAMPLE)
    2 -> 68 bytes (2D)          5,6 -> 80 bytes
    3 -> 84 bytes (2D array)    8 -> 84 bytes (2D multisample array)
    1,7 -> metal-nt SIGSEGVs (not renderable)

The 2D and 2D-multisample programs differ in **three bytes**, and `decomp`
round-trips that to exactly two encoder fields:

    2D    PACK_IMGWIR 5:4 ... | IMGWBLK ... 29:3 30:1 32:1 ...
    2DMS  PACK_IMGWIR 5:6 ... | IMGWBLK ... 29:3 30:2 32:1 ...
    2DA   PACK_IMGWIR 5:6 ... | IMGWBLK ... 29:3 30:1 31:2 32:1 ...
    2DMSA PACK_IMGWIR 5:6 ... | IMGWBLK ... 29:3 30:2 31:2 32:1 ...

**IMGWBLK field 30 is the dimension class** (2D = 1, multisample = 2) and field
31 is the array flag. No instruction is added for the multisample form: the two
grid-coordinate SR reads are unchanged and nothing computes a sample index,
because the block store covers every sample of the tile. The array form, by
contrast, really does add a PACK_IMGWIR and a `get_sr` for the layer.

For U8NORM RT0, texture type 3 produces this complete 84-byte 2D-array text:

```
1c9c100634a810060c9d102624a91426
24c310469f1054000004180050060b00
9f2054020000140050060b0097455404
00081000f0c000009705540600080000
d0401000571054000000008028750000
0e000000
```

Its SHA-256 is
`69a97b1048833dc06dbe02360d9c34c54b46cc8a07ad782eb589b0ec5a6ac307`.
The first four SR reads and two products are the same tile-origin calculation
as the qualified 2D builtin: threadgroup position multiplied by dispatched
group size for X and Y. The fifth SR supplies the layer. Native metadata field
0 is 4 GPR dwords and fields 38/42 are both 16. Per-thread X/Y is not an
equivalent EOT coordinate source: it restarts at every 32x32 tile.

Sweeping internal format (0x75) x components (0x79) shows the transform is
uniform across all five selectors the driver uses, byte for byte:

    byte 43 (PACK_IMGWIR wire width)  0x04 -> 0x06
    byte 60                           0xa8 -> 0x28
    byte 61 (selector | dim)          0x?5 -> 0x?2   (?  = 7,1,3,5,9)

The attachment table carries no sample count -- the count comes from the
tilebuffer geometry and the PBE descriptor, which is consistent with a block
store that writes the whole imageblock. The later RGBA32 audit in
`../compiler/test/probes/README-g17p-native-rt0-eot.md` identifies its old
84-byte fixture as an array program, not a signedness variant. The same holds
unresolved: internal formats 10, 11 and 17 with four components give one
68-byte type-4 program (SHA256 prefix `cbe3da3f8d30b159`) and one 84-byte
type-8 program (`25b7628e79fd827b`), so signed RGBA32 stores multisample
targets like the unsigned and float formats.

The matching write descriptor comes from the UMD rather than the compiler:
`AGX::PBEStateEncoderGen9<kAGXRevHAL200>::StateFields::StateFields` derives its
dimension nibble as `2 + ((flags >> 16) & 2)` (and 3 / 8 on the array path),
and deposits `(samples >> 2) & 3` at word0[57:56] -- the two-bit `Sample Count`
enum, 2x=0 4x=1 8x=2. Width and height stay logical. Cross-check: the same
function's `level | (levels-1) << 44` write into word2 and its word0 bit 58 /
word1 bit 63 extension markers are exactly what `agx_pbe_agx3_2d_mip_pack`
already encodes from independent evidence.

## Witness: simple AGX3 per-sample fragments are unrolled (2026-09-10)

`.tmp/g17p-msaa-grounding/` compiles the same MSL fragment function at
`raster_sample_count` 1, 2 and 4 (the key is a `render_pipelines` entry field of
that name). For a shader that reads `[[sample_id]]`, Apple's `applegpu_g17p`
text grows 234 -> 290 -> 422 bytes and the number of tile-store sequences goes
1 -> 2 -> 4, with the sample index folded in as an immediate. These shaders
are **unrolled inside the fragment program**; this is not a universal native
rule. The later `README-g17p-sample-alpha-depth.md` witness has a native sample
loop and combined class-13 transaction for alpha coverage plus depth. Neither
shape makes `agx_linker.c`'s AGX2 external-loop bytes valid AGX3 code.

Separately, the *per-pixel* program for the same shader is 210 bytes at 1x and
214 at 2x and 4x (2x and 4x byte-identical), so the ordinary fragment epilogue
also changes under multisampling. That difference is unexplained and is not a
claim about our own epilogue being wrong.

## Witness: a loop break deeper than two levels is CNDCF, not a longer CNDLT (2026-09-10)

Question: AGX3's exec-mask stack ops CNDST/CNDLT/CNDEND carry a level count,
and the packer refused any loop break escaping more than the loop itself plus
one enclosing `if`. Does Apple widen CNDLT's count, or emit a different unwind?
Answer: neither -- there is a **separate instruction**, `cndcf`, whose count
field is sixteen bits wide, and CNDLT is simply its compact count-2 form.

`.tmp/g17p-deep-break-evidence/deepbreak.metal` breaks out of a `for` loop from
inside 1..7 nested ifs. Every condition reads a distinct element of a
`device const uint *` and every level stores before the next opens, so neither
the conditions nor the blocks can be folded. `probe2.metal` adds four controls.
Decoded with `tools/decomp.c` against the in-tree compose tables:

    enclosing ifs   native break instruction        bytes
    1               (none -- loop rotated; the      -
                     exit test IS the loop's cndlt)
    2               cndcf slot1 count3              8f 05 54 03 00 01
    3               cndcf slot1 count4              8f 05 54 04 00 01
    4               cndcf slot1 count5              8f 05 54 05 00 01
    5               cndcf slot1 count6              8f 05 54 06 00 01
    6               cndcf slot1 count7              8f 05 54 07 00 01
    7               cndcf slot1 count8              8f 05 54 08 00 01

`cndcf` is `length="6"`, `f12` at bits 24:39 (the count), `f1` at 40:42 (the
flag slot), `f0` at bit 43 (invert) -- the same count/slot/invert triple CNDLT
carries in a four-byte form with `f12` pinned to 2 at bits 24:25.

Four controls pin the semantics:

* **Sense.** Negating the innermost condition (`p_inv`) leaves the cndcf bytes
  identical and changes only the compare. The break's sense is in the compare,
  not in the count.
* **Invert.** A break in an `else` arm (`p_else`) is `8f 05 54 03 00 09`: same
  count, `f0` set. Apple folds the else into the cndcf rather than emitting a
  `cndef`.
* **Scope.** The same two-if break inside an INNER loop, with an outer loop and
  an outer `if` still open (`p_inner`), is still **count 3** -- the count is the
  number of levels open inside the *innermost* loop, loop included. That is
  exactly how Mesa's per-loop `loop_nesting` counter is defined.
* **Depth 1 never needs it.** With work after the `if` so the loop cannot be
  rotated (`p_mid`), Apple still emits only CNDLT: two levels is the loop's own
  cost and the compact form covers it.

So the native level count is Mesa's `nest + 1` at every depth, because a loop
owns two architectural levels where Mesa's model counts one. `agx3_cf_cndcf()`
reproduces all eight witnessed encodings byte for byte
(`test-agx3-cf-core.c: test_deep_break_levels`).

Two smaller facts fell out and are worth not rediscovering:

* A compute pipeline's `.nat` is a **container**: `__TEXT,__compute` of the
  outer Mach-O holds one Mach-O per pipeline, and each of *those* carries the
  machine code in `__TEXT,__text`. The extractor is
  `.tmp/g17p-deep-break-evidence/extract.py`.
* `cndend`'s own count (`f10`) is sixteen bits too, at the same bit position as
  `cndcf`'s. Only 1 and 2 are witnessed and encoded.

## Witness: centroid interpolation is an explicit ITRCOORDS token (2026-09-14)

Question: G17P centroid inputs were encoded as the f32interp register-source
leaf reading an "implicit" coordinate register that nothing produced, and
single-sample centroid and centre interpolation disagreed on hardware. What
does Apple emit?

```metal
struct VCentroid { float4 pos [[position]]; float4 c [[centroid_perspective]]; };
vertex VCentroid v_centroid(uint vid [[vertex_id]]) {
  VCentroid o; o.pos = float4(float(vid), 0.0, 0.0, 1.0);
  o.c = float4(float(vid)); return o;
}
fragment float4 f_centroid(VCentroid in [[stage_in]]) { return in.c; }
```

compiled as a render pipeline (`raster_sample_count` 1, then 4, one RGBA8Unorm
colour attachment), with the same `metal`/`metallib`/`metal-nt -arch
applegpu_g17p -platform_version macos 26.0 26.0` invocation as above.
Decoded with `decomp`, the 1x fragment opens:

    04c2110601000000   rld  SR39 W16 -> h0, slot 1, f48=1 (entry marker)
    af14540603000a01   itrcoords r6 <- h0 (terminal), wait slot 1, f38=1
    2f055400030003020900  f32interp RHW, token r6 live, f54=1
    af005404030002482001  f32rcp
    2f055400030203020800  f32interp coef 1, token r6 live
    ...
    2f055406030803022000  f32interp coef 4, token r6 final (f43)

`[[centroid_no_perspective]]` is the same chain without the RHW, and 4x adds
only a later unmarked SR39 read for the tile store. So the centroid is computed
from the covered samples in SR39 by ITRCOORDS mode 1 (explicit sample
interpolation is mode 2 on a sample index) and consumed through exactly the
sample-token f32interp leaves. There is no implicit form.

The same programs show where the fragment entry marker goes when SR39 is the
first input: on the long-form rld (f48), not on the first f32interp (f63).
`uint m [[sample_mask]]` inputs and a 4x fragment with no inputs at all open
the same way (`14c2110601000000`, SR39 into h2).
