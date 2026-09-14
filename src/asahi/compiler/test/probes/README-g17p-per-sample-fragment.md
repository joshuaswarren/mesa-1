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
