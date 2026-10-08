# Getting Apple's own G17P encoding for a shader shape

Apple's Metal toolchain targets the G17P directly, so an ISA question can be
answered with Apple's byte-exact machine code instead of inference: write the
shape in MSL, compile it for `applegpu_g17p`, and compare against ours.

Use this for any AGX3B change to instruction selection, operand packing,
staging, register allocation, scheduling or the program ABI, and before adding
a G16G/G17P chip gate.  G16G and G17P share the AGX3B compiler path; keep a
chip-specific branch only when Apple's output, a hardware result or a device
limit shows a real difference.  A host test only shows what our compiler
emitted; it is not an architecture oracle.

1. Reduce the shader to the smallest MSL that keeps the operation and its live
   operands.
2. Compile it for `applegpu_g17p` (and a G16G target when one is available).
3. Decode the relevant instructions to field vectors (see below).
4. Compare instruction form, operands, register domains, staging order and
   waits with Mesa's output for the equivalent NIR -- not just size or a hash.

## Toolchain

    MT=$(dirname $(xcrun -f metal))     # Metal.xctoolchain/usr/bin
    "$MT/metal-nt" -archs               # applegpu_g17p is in the list

`metal-nt` needs a pipeline script, and the script's `.mtlp-json` suffix is
required (any other name fails with "malformed flatbuffer magic").  `pipelines`
is a table with the keys `compute_pipelines`, `render_pipelines`,
`tile_render_pipelines` and `mesh_render_pipelines`; each entry names functions
with bare strings.  Verified with MetalToolchain 27.1.5194.15:

    printf '{"pipelines":{"compute_pipelines":[{"compute_function":"K"}]}}' \
        > p.mtlp-json
    xcrun metal -c w.metal -o w.air
    xcrun metallib w.air -o w.metallib
    "$MT/metal-nt" -arch applegpu_g17p -platform_version macos 27.0 27.0 \
        -N p.mtlp-json w.metallib -o out.bin

`-platform_version macos 27.0 27.0` is needed: older targets emit an AIR
version `metal-nt` refuses, and `-std=metalN.M` does not change it.
`xcrun metal -arch applegpu_g17p -c` is not a shortcut; it fails in
`AIRNTEmitObject`.

The output is a Mach-O GPU executable.  Compute code is in `__TEXT,__compute`
(`otool -s __TEXT __compute out.bin`).  The first 0x40 bytes before the entry
point are the constant program (`0e 00 00 00`, then `06 00` filler), the same
container layout the driver builds.

A render pipeline's fragment function is compiled without attachment
information, so a `fragment float4` whose result only reaches a colour output
is optimised away to an 80-byte stub.  Declare it `fragment void` and store to
a `device` buffer instead.  Implicit-LOD sampling is illegal in a compute
kernel (`metal-nt` aborts with "Failed to encode"), so bias witnesses must come
from the render path.

## Decoding

Decode an instruction by finding the AGX3b/AGX3c form whose field vector
re-composes to the same bytes through `agx3_compose` (in-tree tables under
`src/asahi/compiler/agx3_compose/`): enumerate the forms of that length, derive
the fields the bytes imply, re-compose and compare.  Only exact round trips are
facts about the encoding; field names from a disassembler are not.  Start from
a real instruction as the baseline -- an all-zero field vector does not satisfy
a form's predicates.

Field positions differ per instruction form.  For example, the memory forms:

    base kind :  lddev = bit 101      stdev = bit 99
    index kind:  stdev = bit 100
    access kind: lddev = f75 @97:99   stdev = f73 @95:97
    16-bit off : lddev = f76 @77:92   stdev = f74 @75:90

`lddev.f75` and `stdev.f75` are different physical fields, so a rule found on
one form does not transfer to another.  Map each form separately by varying
one field at a time from a known-good baseline.  Common to both memory forms:
wait mask = bits 12..17 (wait slots 1..6), lddev signal slot = bits 70..72,
lddev uniform base = byte 4 = base / 4.
