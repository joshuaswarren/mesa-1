# Getting Apple's own G17P encoding for any shader shape

Apple's Metal toolchain is installed on the development Mac and targets our exact
part directly.  This turns ISA questions from inference into witness: write MSL,
get Apple's byte-exact `applegpu_g17p` machine code, diff against ours.

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
