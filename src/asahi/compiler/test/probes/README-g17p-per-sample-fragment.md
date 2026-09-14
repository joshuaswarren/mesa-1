# G17P runs a fragment function per sample by unrolling it

`agx_fast_link()` splices literal AGX2 machine-code fragments around a fragment
shader to run it once per sample, and refuses to do so on G17P because those
fragments are not AGX3 code:

```
agx_fast_link: external sample loop is not AGX3 machine code
```

`capture-g17p-per-sample-fragment.py` asks Apple's own compiler what the AGX3
answer is. It compiles the same trivial fragment function twice -- once reading
`[[sample_id]]`, once not -- at one and at four rasterization samples, and
prints the native fragment object.

```
per-pixel-1x    118 bytes
per-sample-1x   118 bytes
per-pixel-4x    126 bytes
per-sample-4x   240 bytes
```

There is no loop. At four samples the per-sample program carries the body
**four times**: four staged operand groups (`97 04 54 ...`) and four tilebuffer
stores whose sample operands are `06`, `04`, `02` and `00`, against the single
store `e7 16 54 ... 02 4e` of the per-pixel program. Both end with the ordinary
`0e` terminator; neither has a loop header, a back edge, or the AGX2 `0088`
stop.

At one sample the two programs are the same length, differing only in a short
prologue: per-sample and per-pixel invocation coincide there. That is exactly
what `agx_nir_lower_g17p_1x_sample_shading()` relies on.

So the way to run a fragment shader per sample on G17P is to **unroll the body
once per sample**, substituting the sample identity in each copy and letting
`agx_nir_lower_g17p_tile_samples()` address each copy's own tilebuffer sample.
Wrapping the body is an AGX2 shape with no AGX3 equivalent.

Requires Apple's `metal`, `metallib` and `metal-nt` through `xcrun`; captured
against AIR-NT 32023.917, `-arch applegpu_g17p`.

## The depth store carries the sample as a one-hot field

`capture-g17p-per-sample-depth.py` does the same for a fragment that writes
depth, which is the shape a multisample depth blit takes.

```
depth-pixel-4x   316 bytes
depth-sample-4x  724 bytes
```

Unrolled again, and the four depth stores differ in exactly the fields that
name the sample:

```
d7 14 54 00 00 03 34
d7 24 54 04 20 03 34
d7 44 54 08 40 03 34
d7 84 54 0c 60 03 07
```

Read against the field roles `agx3_zs_core.h` already documents, these are
**four ordinary `depthf` instructions**, not a new form. The two operand bytes
walk `sample * 4` and `sample * 0x20` -- the f5 operand naming each copy's own
staged word -- and byte 1 walks `0x10, 0x20, 0x40, 0x80`, which is the
f17..f22 per-slot wait mask: one scoreboard slot per copy. No field carries a
sample number.

The sample selection lives where that header says the contract lives: in what
the staged register holds, and in the coverage the `movmsk` before each store
sets. `lower_g17p_tile_sample()` already describes the same shape for
tilebuffer stores -- coverage "ANDed with a one-hot bit for each iteration of
an unrolled per-sample shader".

So the encoders exist. What `agx3_pack_binary.c` still has to learn is the
**sequencing**: its `g17p_deferred_zs_wait` is a one-shot, so a body that emits
one `wait_pix` and N `zs_emit_agx3` is refused at the second with

```
no grounded AGX3 encoding for zs_emit_agx3
```

Each store needs its own pixel transaction on its own slot, with coverage set
per iteration.

Reproduce with `ASAHI_MSAA_DEPTH_BLIT=1` under the drm-shim
(`asahi-g17p-blit`), which builds the blit's programs without a GPU.
