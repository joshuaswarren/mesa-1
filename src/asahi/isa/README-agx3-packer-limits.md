# AGX3 (G17P) shader packer: what it cannot encode yet

**Status: the SSBO limit below is GONE, and so is the `fcmpsel` one; two
texture-side gaps remain.** Every row of the original matrix compiles,
including the ones marked **no**, and so do device atomics. The measurement
that produced the matrix is preserved verbatim further down, because the
*method* is what makes it re-runnable, and because the retraction it carries is
still worth reading before theorising about a wrong value on this ISA.

Measured 2026-09-07 under the G17P drm-shim, on the integration branch plus the
commits named in "What changed". No hardware.

## Current matrix

`src/asahi/isa/probes/r*.comp` are the compute shapes; `f*.frag` plus
`common.vert` are the fragment ones. Both sets are driven by the two probes
described in "Reproducing".

| shader body (one `main`, SSBO at set 0 binding 0)                  | encodes |
| ------------------------------------------------------------------ | ------- |
| `values[0] = 0x11110000u;`                                          | yes     |
| `values[0] = 0x11110000u + gl_GlobalInvocationID.x;`                | yes     |
| `values[gl_GlobalInvocationID.x] = 0x12340000u + gid;`              | yes     |
| `values[gl_LocalInvocationID.x] = 0x44440000u + lid;`               | yes     |
| `values[gl_WorkGroupID.x] = 0x11110000u + gid;`                     | yes     |
| `if (g<16) values[g] = A; else values[g] = B;`                      | yes     |
| `values[0] = A;` **and** `values[32u + g] = B;`                     | yes     |
| `values[w] = A;` **and** `values[32u + g] = B;`                     | yes     |
| `x = values[31u - g];` **then** `values[g] = 0x12340000u + x;`      | yes     |
| two loads and a store, four stores, in-place read-modify-write      | yes     |
| `atomicAdd` / `atomicMax` / `atomicOr` / `atomicAnd` / `atomicXor`  | yes     |
| `atomicMin` / `atomicExchange` / `atomicCompSwap`, indexed or not   | yes     |
| an atomic whose return value is then stored                         | yes     |
| shared-memory atomic with barriers                                  | yes     |

Fragment shapes, same day, same build. These are what actually blocks vkcube
and every textured draw:

| fragment body (`sampler2D tex` at binding 1)                       | encodes |
| ------------------------------------------------------------------ | ------- |
| `texture(tex, uv)`                                                  | yes     |
| `texture(tex, uv) * interpolated_colour`                            | yes     |
| `textureLod(tex, uv, 0.0)`, `textureProj`, `texelFetch`             | yes     |
| `dFdx` on an interpolated value                                     | yes     |
| `texture(tex, uv) / float(textureSize(tex, 0).x)`                   | yes\*   |
| `if (x < 0.1) discard;`                                             | yes\*\*  |
| `max(dot(normalize(n), L), 0.0)` — normalize/clamp lighting         | yes\*\*  |
| that lighting multiplied by a texture fetch                         | yes\*\*  |
| `clamp(colour, 0.0, 1.0)`                                           | yes\*\*  |
| `max(a, b)` / `min(a, b)` between two interpolated values           | yes\*\*  |
| `max(0.0, x)` — the constant written **first**                      | yes\*\*\*|
| `max(0.0031308, x)` — likewise, constant not a minifloat            | yes\*\*\*|
| `x < 0.5 ? a : b` — a plain float ternary                           | yes\*\*\*|
| `x >= k ? a : b`, `x <= k ? a : b`, `x != k ? a : b`                | yes\*\*\*|
| `t.a > 0.5 ? t : colour` on a texture fetch                         | yes\*\*\*|
| `clamp(texture(tex, uv) * colour, 0.0, 1.0)`                        | **no**† |
| `texture(tex, uv, 0.5)` — LOD **bias**                              | **no**‡ |
| `textureGrad(tex, uv, dFdx(uv), dFdy(uv))`                          | **no**‡ |

\* This row was **no** until the 32-bit reciprocal was encoded; see below.
\*\* These rows were **no** until the 32-bit tied `fcmpsel` was encoded; see
below. The last three are new probes (`f14`, `f15`, `f16`) added with it.

\*\*\* These rows were **no** after the tied minimum and maximum landed: two
further gaps, described in "The untied select was a second gap" and "Which
side the constant is written on" below. New probes `f17`-`f22`.

† Not a `fcmpsel` gap at all: this shape reaches the packer as `fmul.sat`, a
saturating float multiply, and it is the *saturate modifier* that has no
grounded encoding (`agx3_pack_binary.c`, the generic ALU fallback names
`fmul`). Recorded here because the probe is new and the distinction is easy to
lose.

‡ The LOD-bias and gradient *instruction* encodings are now grounded and
implemented — see "The bias and gradient sample forms" below — but these two
shapes still fail, for a different reason found while grounding them, also
recorded there.

## The rule, as it stands now

There is no per-shader SSBO access limit, no atomic limit, and no missing
minimum/maximum. What remains is a small set of missing *instruction
encodings*, each of which fails the whole shader closed. With
`AGX_MESA_DEBUG=shaders` the packer names the one it refused:

```
agx3_pack_binary.c:4288: no grounded AGX3 encoding for texture_sample
agx3_pack_binary.c:6567: no grounded AGX3 encoding for fmul
```

Known-missing, in the order they are worth doing:

- **The float saturate modifier.** `clamp(x, 0.0, 1.0)` on a value that is
  itself a product reaches the packer as `fmul.sat`, and the generic ALU
  fallback has no encoding for the saturating form. Apple compiles
  `saturate(a[i])` on its own to the eight-byte `09 01 1c 81 01 02 00 82`,
  which decomposes as `f32add.ep2` carrying f14 = 1 and a minifloat operand;
  the plain add does not set f14. That is a witness, not a swept field
  assignment — nothing here varied f14 against a second shape — so it is
  recorded as a starting point, not as an encoding.

- **The bindless texture-heap descriptor shape.** A sample whose texture and
  sampler come from the bindless heap rather than the combined-sampler
  descriptor names a *uniform* heap base in source 2; `agx3_emit_texture_sample`
  requires that source to be the immediate zero. This is what actually blocks
  the LOD-bias and gradient probes today; see below.

- **`textureGrad`'s gradient staging.** The instruction mode is grounded
  (f139 = 5) but the four gradient scalars have to be published into operand
  banks 8/10/12/14 through the ten-byte `copy.ep1` form, and only the
  four-byte `copy.ep0` coordinate staging is implemented.

## The 32-bit tied fcmpsel, and what it cost

`max(x, 0.0)` on floats — so `normalize()`, `clamp()`, `saturate()` and the
`discard` compare — used to reach the generic ALU fallback and fail the shader
closed. It is now encoded. Apple's own bytes, from
`metal-nt -arch applegpu_g17p`:

```
max(a[i], b[i])      12 03 1e 05 00 c0     fsel.ep7, register second operand
min(a[i], b[i])      12 03 1e 05 01 c0
max(a[i], 0.0f)      12 03 0a 81 00 c2     fsel.ep2, minifloat second operand
max(a[i], 1.0f)      12 03 0a b1 00 c2
max(a[i], -1.0f)     12 03 1a b1 00 c2
max(a[i], 31.0f)     12 03 0a ff 10 c2
min(a[i], 0.0f)      12 03 0a 81 01 c2
```

It is a six-byte `fsel`, not the mixed-width select the packer was reaching
for. Structurally it is the same form the half `fmin`/`fmax` encoder already
used: only the five width selectors change (f11/f43/f60/f77/f94, 1 = one
16-bit half, 2 = a 32-bit pair). The immediate is an ordinary AGX minifloat and
is deposited **twice**, into f51 and f85, because the compared operand and the
selected operand are the same value — which is exactly what makes the
instruction a minimum or a maximum rather than a general select. Apple picks
the encoding profile by that operand: ep7 (destination in f3) for a register,
ep2 (destination in f17) for a minifloat.

Three things had to change together:

1. `agx3_alu_fmin_f32` / `agx3_alu_fmax_f32` and their `_imm` forms
   (`agx3_alu_core.c`), byte-exact against every witness above plus Apple's
   own register and lifetime assignment for eight simultaneously live maxima.
2. A packer handler for the tied 32-bit shape, placed ahead of every other
   `fcmpsel` handler so the general select paths cannot claim it.
3. The optimizer had to stop half-folding the constant. `agx_is_float_src()`
   reports source 3 of a select as an integer — correctly, a select can carry
   an integer — so the generic rule folded the minifloat into the *comparison*
   and left the *selection* a register, breaking the tie and handing the packer
   a shape it has no encoding for. The tie now folds in both slots or neither.

The same session found that two of the half-select handlers refused the shader
outright when their operand shape did not match, instead of falling through to
the later handler that did. `if (x < 0.1) discard;` is exactly that: a half
select of two byte constants, which the byte-constant encoder has always
supported and never got the chance to. Those two handlers now skip. That alone
also fixed `AGX3PackBinary.UpdateLightmapNativeIntegerAndMaskForms`.

## The untied select was a second gap

Encoding the tied minimum and maximum did not close `fcmpsel`. A plain float
ternary — `x < 0.5 ? a : b`, the commonest select there is — still failed the
shader closed, because Mesa keeps the comparison at 32 bits and carries the
constant as a minifloat, so the packer sees a select with a 32-bit register on
one side and a 16-bit immediate on the other and nothing to widen. The tied
handler does not apply: the compared operand and the selected operand are
different values, which is the whole point of a select.

The refusal message named only the opcode, which is true of the shapes that
encode as well, so it is now printed with the operand shape:

```
agx3_pack_binary.c:6622: no grounded AGX3 encoding for fcmpsel
  [d32:reg8 s0=32:reg4.discard s1=16:imm32 s2=32:reg4.discard s3=32:reg6.cache
   cond=5 chan=1]
```

Apple's bytes for `(a[i] OP 0.5f) ? c[i] : d[i]`, six kernels differing only in
the operator:

```
<    12 03 0f a1 22 06 03 02 20 0a     f110 = 1
>    12 03 0f a1 22 06 02 02 20 0a     f110 = 2
>=   12 03 0f a1 26 06 02 02 20 0a     f110 = 5
<=   12 03 0f a1 26 06 03 02 20 0a     f110 = 6
==   12 03 0f a1 26 06 00 02 20 0a     f110 = 0
!=   12 03 0f a1 26 08 00 02 20 06     f110 = 0, selected operands exchanged
```

Only the condition moves. f110 is literally `enum agx_fcond`, and `!=` is `==`
with the two selected values exchanged — which is also how an inverted
condition encodes, so `invert_cond` needs no form of its own. The same field
in the same form governs the half-register comparison the tree already had
grounded for `EQ` and `LT` only; that handler is now open to all five
witnessed conditions and to the half being on either side. The NaN-ignoring
`LTN`/`GTN` codes are *not* witnessed for this form and stay refused; they
belong to the tied minimum and maximum, which have their own grounding.

The shape space is enumerable, and now is:
`AGX3PackBinary.DISABLED_FcmpselShapeSurvey` in
`src/asahi/compiler/test/test-agx3-pack-binary.cpp` walks every combination of
destination width, four operand kinds, seven conditions and inversion, and
prints the refusals. Run it with
`agx_tests --gtest_also_run_disabled_tests --gtest_filter=*FcmpselShapeSurvey*`.
Every `d32 = fcmpsel <reg|imm|half>, <reg|imm|half>, r32, r32` now encodes
except `LTN`/`GTN`. Two families remain refused, and neither has an Apple
witness to encode from:

- **Selects that are themselves immediates** (`sel(r32, #k)` and friends).
- **A 16-bit destination whose comparison is not also 16-bit** — e.g.
  `d16 = fcmpsel r32, h16, h16, h16`. Asked for exactly this shape,
  `(a[i] OP 0.5f) ? c[i] : d[i]` with `a` a float and `c`, `d`, `o` halves,
  Apple's compiler does **not** emit it: it widens the halves, selects in 32
  bits, and narrows the result, so every witness comes back `11:2 77:2 94:2`.
  If Mesa hands the packer this shape the fix is to widen in Mesa, not to
  invent an encoding for it.

## Which side the constant is written on

Encoding the tied minimum and maximum, and then the untied select, still left
stock vkcube refusing on `fcmpsel`. The shape, from the refusal line the
hardware now prints:

```
agx3_pack_binary.c:6726: no grounded AGX3 encoding for fcmpsel
  [d32:reg2 s0=16:reg0.discard s1=32:reg2.discard s2=16:imm0
   s3=32:reg2.discard cond=7 chan=1]
```

`cond=7` is GTN, so this is a maximum, not a select: sources 0 and 2 are one
tied operand and 1 and 3 the other. But source 0 is a 16-bit *register* while
source 2 is a 16-bit *immediate zero* — the tie is broken, and broken across
widths. `max(0.0, x)` reproduces it exactly (`f21`).

It is the same half-fold as before, in the other pair. The fold had been opened
for sources 1 and 3 only, so a constant written *first* still folded into
source 2 alone and left source 0 a register. That is strictly worse than the
other order, because `agx_opt_compact_constants` then sees the surviving
reference as the constant's only float use and narrows it to 16 bits, producing
a 32-bit maximum compared against a 16-bit register — a shape Apple's own
compiler will not emit at all. Folding either pair as a unit fixes both halves
of that: the foldable case becomes the grounded minifloat form, and the
un-foldable case keeps two SSA references, one of which is not a float use, so
the constant is never narrowed and the grounded register form takes it (`f22`).

No new encoding was needed, and none was invented. The packer normalizes
`max(0.0, x)` to the `max(x, 0.0)` Apple encodes, which is legal because NIR
declares `fmin_agx` and `fmax_agx` `2src_commutative`, and a tied GTN/LTN
`fcmpsel` can only have come from one of them — every other condition a
comparison produces is ordered.

The refused shape itself stays refused, correctly: with a 16-bit register on
one side of a 32-bit tie there is nothing to encode and no witness to encode
it from. It simply is not produced any more.

## Where the fragment epilog is not the problem

`asahi-g17p-epilog-matrix` sweeps 36,864 fragment-epilog keys — twelve colour
formats, six blend modes, sRGB lowering, and the eight link flags — compiles
each, and reports the opcode of every refusal. It is a ratchet: it fails only
if the refusal count grows.

As measured, 33,136 refuse, and the opcodes are `zs_emit` (depth or stencil
written from the epilog), `ld_tile` and `st_tile` (tilebuffer formats), `log2`
(the sRGB encode's `pow`, which is also why `asahi-g17p-srgb-epilog` is red)
and a saturating `fadd`/`fmul`. **No `fcmpsel` shape appears anywhere in that
space**, which is what ruled the epilog out as the source of a reported
`fcmpsel` refusal, and is why the untied select above was found instead.

## The bias and gradient sample forms

Both instruction encodings are grounded, from fragment and compute pipelines
compiled with `metal-nt`. **Neither closes its probe**, and the reason is worth
recording because it is not where anyone was looking.

`t.sample(s, uv, bias(k))` is the ordinary implicit-LOD sample with two extra
fields: f37 = 1 (immediate operand), f139 = 1 (bias mode), and f29 = k * 64 in
the same 12-bit fixed-point field the explicit-level form uses. Swept over
k = 0.25, 0.5, 1, 1.5, 2, 4, 1/64 and 31, f29 came out 16, 32, 64, 96, 128,
256, 1 and 1984 — exactly k * 64 — and k = 63 saturated the field at 2047. A
zero bias is folded away by Apple's own compiler into the plain form. That is
`AGX3_TEX_LOD_BIAS` in `agx3_tex_core.c`, with byte-exact vectors.

`t.sample(s, uv, gradient2d(dx, dy))` selects f139 = 5 and stages the four
gradient scalars into operand banks 8/10/12/14 with six ten-byte `copy.ep1`
instructions ahead of the sample. The mode is implemented in neither direction
yet; only the mode number is recorded.

**What actually blocks both probes** is a NIR-side gate, not the ISA:
`agx_nir_lower_g17p_combined_sampler.c` accepts a shader only when *every*
texture op is `nir_texop_tex` or `nir_texop_txl`. One `txb` or `txd` disowns
the whole shader, which then falls back to the bindless texture-heap
descriptor form — source 2 a uniform heap base rather than the immediate zero
`agx3_emit_texture_sample` requires — and the sample is refused for that
instead. Measured directly: `f01` reaches the packer with sources 2/3/4 all
immediate zero, `f05` with `[2] = uniform 16, [3] = register 2, [4] = register
0`. Closing either probe means teaching that pass to keep a genuine API bias
(its existing `txb` handling only recognises and *discards* the driver's own
zero-valued software bias), or grounding the bindless-heap sample form.

## What changed earlier, and how it was measured

Two compiler fixes, both grounded byte-exactly against Apple's own machine
code (see "Getting Apple's own G17P encoding", below):

1. **`iadd` with narrow register operands.** The register-slot-A form pinned
   both operand widths to 32-bit, so `iadd d32, h_a, h_b` had no encoding at
   all and every compute shader that sums `gl_LocalInvocationID`'s three
   16-bit components was refused outright. f24 is slot A's width and f39 is
   slot B's, on both encodings; f28 and f43 are their sign-extensions.

2. **32-bit reciprocal and reciprocal square root.** `RCP` and `RSQRT`
   accepted only 16-bit operands, so a float divide or a `normalize()` — both
   of which NIR lowers to `frcp` / `frsq` on 32-bit values — refused the
   shader. The forms were already in the table; only the half widths had an
   encoder.

Neither of these is the SSBO limit. The SSBO rows were already passing before
this session's changes: the compiler-side work the previous revision of this
file described as "in flight" landed, and nobody re-measured. The lesson is in
"Two traps" below — a matrix is only true on the day it is run.

## Reproducing

Both probes are in-tree meson targets and need no hardware.

```
docker run --rm -v <repo>:/workspace -w /workspace/<mesa worktree> \
  winunix-dev:mesa-f16-deps bash -lc '
    meson setup build -Dvulkan-drivers=asahi -Dgallium-drivers= -Dplatforms= \
      -Dbuild-tests=true -Dtools=drm-shim -Dbuildtype=debugoptimized
    ninja -C build
    cd src/asahi/isa/probes
    for f in *.comp; do glslangValidator -V $f -o ${f%.comp}.spv; done
    glslangValidator -V common.vert -o common.vert.spv
    for f in *.frag; do glslangValidator -V $f -o ${f%.frag}.frag.spv; done

    V=$PWD/../../../../build/src/asahi/vulkan
    export LD_PRELOAD=$V/../drm-shim/libasahi_noop_drm_shim.so
    export ASAHI_SHIM_CHIP=g17p
    export VK_DRIVER_FILES=$V/asahi_devenv_icd.aarch64.json

    $V/g17p-vulkan-pipeline-probe *.spv                 # compute matrix
    $V/g17p-vulkan-graphics-probe common.vert.spv *.frag.spv   # fragment matrix'
```

Each line prints `ENCODES` or `REFUSED` with the `VkResult`; the exit status is
the number of refusals. Add `AGX_MESA_DEBUG=shaders` to see which instruction
the packer refused.

As of this revision the compute probe reports 24 of 24 encoding and the
fragment probe 18 of 21. The three refusals are `f05_texture_bias` and
`f12_texture_grad`, both on the bindless-heap descriptor shape, and
`f16_saturate_texture` on `fmul.sat`. Stock vkcube's own fragment SPIR-V
(`.tmp/g17p-wide-imad-evidence/vk-repro/cube.frag.spv`) encodes; it did before
this revision too, so vkcube's reported failure is not this file's matrix and
has not been reproduced here.

Two traps, both of which produced wrong answers before being caught:

- Use `asahi_devenv_icd.aarch64.json`, not `asahi_icd.aarch64.json`. The latter
  points at the *installed* `/usr/lib` driver, so the loader silently runs some
  other build and every result is about the wrong binary.
- Rebuild with a bare `ninja -C <build>`. Naming the executable target skips
  the regenerated SPIR-V header, so the previous shader keeps running and every
  variant "fails" identically. On a Docker bind mount, `touch` the file you
  edited first: ninja's mtime comparison can otherwise skip the rebuild and you
  will measure the previous binary.

## Standing defect: baseline wait and lifetime bits in `agx3_alu_compose()`

Not fixed here, because fixing it changes the bytes of every wait-carrying ALU
instruction in every shader and that cannot be validated without hardware.

`agx3_alu_table.h` was extracted by decoding one witnessed Apple instruction
per form, and each AGX3b entry ended up with that specimen's own scoreboard
wait folded in as if it were a form constant — `iadd` and `isub` pin f50,
`imadd` f65, `sll`/`srl` f60, `bit_and`/`bit_or`/`bit_xor` f49, and so on,
each of which is that form's *last wait field*. `iadd` additionally pins
f27 = 1, the slot-A last-use bit.

`agx3_alu_compose()` copies the baseline verbatim, so:

- every ALU it emits also waits scoreboard slot 6 (slot 8 on AGX3c), and a
  genuine slot-6 wait is indistinguishable from none —
  `agx3_alu_compose(IADD, wait = 0)` and `wait = 1u << 5` produce identical
  bytes; and
- every `iadd` it emits declares slot A dead, whether or not it is.

Apple varies both bits per instruction. Its dependent add after a slot-6
`lddev` is `9f 01 56 ...`; its add after a slot-1 `get_sr` is `9f 11 54 ...`;
its `iadd` last-use byte is 0x11, 0x15 or 0x17 depending on which operands are
actually dead. The over-wait is conservative and therefore harmless; the
"slot A is dead" claim on a live register is not, and is the same class of
error that `agx3_iadd_core.h`'s header warns about.

The fix is small — skip baseline entries whose field appears in the form's
`wait[]` list, which `agx3_alu_mod_*()` and `agx3_alu_bit_*()` already do —
but it also requires re-baselining `test/agx3_alu_core_vectors.c` (a generated
file) and a hardware run, so it is recorded here rather than applied.

## RETRACTED: the "positional data operand" model

**This section previously claimed that `device_store` names no source register
and stores whatever the immediately preceding instruction produced. That is
WRONG. Do not propagate it.** It is kept as a retraction because the claim was
circulated before it was checked.

### What is actually true

`device_store` **does** name its data register, at **byte 3, in half-register
units**. Confirmed two independent ways:

- A controlled encoder sweep varying only the data register, everything else
  fixed: `data=r0/r1/r2/r3/r5` produce byte 3 = `0x00/0x02/0x04/0x06/0x0a`.
- Mesa's own encoder says so in source: `agx3_mem_core.c common_overrides()`
  does `KV(2, d->data)`, and field 2 lands in byte 3.

The index register is likewise named explicitly, at byte 5.

`0x54` / `0x55` / `0x56` in byte 2 is **not** an addressing mode. It is the top
of the **scoreboard wait mask** (slots 1-4 live in byte 1; slot 5 sets byte 2 to
`0x55`, slot 6 to `0x56`). So the "undefined mode `0x55`" flagged here earlier
is simply a store waiting on slot 5 — entirely legitimate.

Byte 2 also carries one bit that is not a wait: AGX3b f79, the "last LDDEV of
the stream" marker, which *clears* bit 4 (`0x54` -> `0x44`). Apple sets it on
the final `lddev` and only the final one; a two-load kernel emits `0x54` then
`0x44`.

### The real cause of the wrong values

A missing **write-after-write hazard check** in the AGX3 slot allocator
(`agx3_slot_core.c`), found by recompiling the exact `lid` binary and reading
the IR:

```
r0...r3 = device_load ... slot 0      <- async, claims r0,r1,r2,r3
$r3     = mov_imm #0x44440000         <- writes r3 while that load is IN FLIGHT
r3      = iadd ^r3, $r2l.abs, #0x0    <- the value lives in r3
device_store ^r3, ^r0:r1, ^r4.abs     <- waits for the load, THEN reads r3
```

Only `r0:r1` of the 4-word descriptor load are used, so the allocator reused r3
for the value. The store correctly waits for the load — and that is what kills
it: the load retires and overwrites r3 with the descriptor's **fourth word**
before the store reads it. The store writes that word. Descriptor word 4 is
zero, which is the observed zero.

The pass computed waits from **sources only** (read-after-write) and never
checked destinations; worse, a synchronous write cleared the pending tag, so
the in-flight load on that register was forgotten. Mesa's AGX2 pass has an
explicit write-after-write block; the AGX3 port dropped it. Fixed by adding the
destination check, which is in `agx3_slot_assign()` today.

### Why the wrong model looked so convincing

The five-probe correlation in the original worked example is real: the probes
whose value-producing op was adjacent to the store were exactly the ones with
correct values. But adjacency was a *proxy*. What actually mattered was whether
an async load was still in flight over the register holding the value, and
"something scheduled in between" correlates with that without being it.

Three things should have lowered confidence at the time:

1. The field semantics came from a **secondary source** (the ISA-DB's prose)
   rather than from the encoder that emits the bytes. The DB's own sweep looked
   at bytes +8/+9/+11 and concluded the data register "is not in this
   instruction" — it was looking in the wrong place.
2. A correlation over five samples with a plausible mechanism is not a
   mechanism. There was no controlled experiment varying one field.
3. The model predicted that identical store bytes must mean identical
   operands. They *were* identical and the operands *were* correct — which
   should have been treated as evidence against the model rather than for it.

**When a value is wrong on this ISA, read the encoder source or sweep one field
at a time before theorising from a disassembler's field names.**

## The original measurement, kept for its method

The matrix below was measured from the runtime side (Vulkan pipeline creation)
against the tree as it stood before the compiler workstream began changing
`src/asahi/compiler/agx3_pack_binary.c` and friends.

| shader body (one `main`, SSBO at set 0 binding 0)                  | encoded |
| ------------------------------------------------------------------ | ------- |
| `values[0] = A;` **and** `values[32u + g] = B;`                     | **no**  |
| `values[w] = A;` **and** `values[32u + g] = B;`                     | **no**  |
| `x = values[31u - g];` **then** `values[g] = 0x12340000u + x;`      | **no**  |
| any `atomicAdd` / `atomicMax` / `atomicOr` on the SSBO              | **no**  |

The rule was stated as "at most one SSBO access per shader", with
`load_preamble` / uniform reads and constant-data (rodata) loads not counting,
and atomics unencodable in any position. It bounded what a diagnostic probe
could express, which is why `g17p_vulkan_compute_count.comp` has the shape it
does: every invocation stores to the *same* slot with its own ID as the
*value*, the most that one SSBO access can be made to reveal.

`src/asahi/compiler/test/test-agx3-ssbo-multi-access.c` now owns every row of
that matrix, so it cannot come back silently. Its store bytes are pinned
against Apple's own two-store kernel: two consecutive `stdev` instructions both
carry byte 1 = 0 and byte 2 = 0x54, i.e. a store signals no scoreboard slot and
a second store does not wait on the first.

## Disassembling a G17P shader binary

```
cd g17-upstream/agx-re/tools/agx-isa
python3 agxisa.py tokenize <raw hex of the shader binary>
```

> **Caution:** this disassembler's `device_store` field assignment is wrong in
> at least three places. It labels byte 3 `extmode` when byte 3 is the **data
> register** (half-register units), labels byte 2 `addr_mode` when byte 2 is the
> top of the **scoreboard wait mask**, and its prose asserts the data register
> "is not in this instruction". Instruction *boundaries* and other families
> decode fine, and it is still the fastest way to see the shape of a program —
> but for `device_store` fields trust `agx3_mem_core.c` and the encoder, not
> this tool. Believing its field names cost an entire wrong theory; see the
> retraction above.

Shader bytes come from the driver: run a probe under the drm-shim with
`HK_G17P_LAUNCH_TRACE=1` and `grep -A9 "shader-binary"`.

## Getting Apple's own G17P encoding for any shape

The development Mac's Metal toolchain contains an offline AIR-to-native
translator that targets `applegpu_g17p` directly, so an encoding question is a
witness rather than an inference. See `README-metal-witness.md` next to this
file for the harness; every byte quoted above came out of it.

The field semantics used here were each pinned by a controlled sweep over
`agx3_bind_compose()`, one field at a time, against the Apple bytes — never by
reading a field name out of a disassembler.

## Hyprland's own GLES shaders, measured (2026-09-07)

Hyprland 0.56.1 ships thirteen built-in fragment shaders, all `#version 300
es`. Running each one through Zink -> Honeykrisp under the G17P drm-shim, one
shader per process, closed two encodings and left three open. No hardware.

Reproduce with `neo-gpu-harness/scripts/hypr-shader-probe.sh <mesa worktree>`;
it fetches the pinned Hyprland tag, preprocesses the GLSL the way Hyprland's
own `CShaderLoader` does, runs a llvmpipe positive control and a
known-unencodable negative control, and refuses to report a result if either
control does not behave.

Two things about that measurement are easy to get wrong and both produced a
wrong answer here first:

- **Zink does not fail a draw whose pipeline it could not build.** It logs
  `pipeline creation failed; dropping draws that need it` and returns cleanly,
  so `glGetError()` reports success for a shader the packer refused outright.
  The authority is the packer's own refusal line, not the GL error.
- **Under the no-op drm-shim nothing executes**, so the render target is never
  written by anyone and a framebuffer readback cannot distinguish a dropped
  draw from a working one either. The readback is only meaningful on hardware
  or on llvmpipe.

State after the two encodings below:

| Hyprland fragment shader                       | G17P    |
| ---------------------------------------------- | ------- |
| passthru, quad, rgbamatte, surface             | encodes |
| blurprepare, blurfinish                        | encodes |
| blur1, blur2                                   | **no** - `hmul [d16 s0=16:reg s1=16:imm]` |
| border, inner_glow, shadow                     | **no** - `iadd [d16 s0=16:reg s1=16:imm.neg]` |
| ext, glitch                                    | **no** - `fcmpsel` with a 32-bit compare source and 16-bit immediates |

### Closed: the 16-bit control-flow compare

`if (gradientLength < 2)` in Hyprland's `gradient.glsl` reached the packer as
`if_icmp [s0=16:imm0 s1=16:reg2 cond=5]` and then, once that order was
accepted, as `if_icmp [s0=16:reg s1=16:imm2 cond=5]`. Two separate gaps:

1. **Operand order.** `agx3_emit_cf_compare()`'s 16-bit path only matched a
   constant on the right. The 32-bit path beside it already canonicalized the
   other order with `agx3_cf_swap_icond()`; the 16-bit path now does the same.
   This emits the encoding that was already grounded, with the relation
   swapped, so it admits no new bytes.
2. **Non-zero constant.** `agx3_cf_icmp_zero16_runtime()` never set f16, which
   is why it was a zero compare. Witnessed with `metal-nt -arch
   applegpu_g17p`, one kernel per immediate against the zero form as the
   control (`.tmp/g17p-hyprland-cf-evidence/icmp16imm.metal`), every witness
   the same ten-byte instruction with only f16 different:

       short x;  if (x < 0)    0a 04 23 80 02 00 07 c2 02 00   (the zero form)
       short x;  if (x < 1)    0a 04 23 80 02 00 17 c2 02 00
       short x;  if (x < 2)    0a 04 23 82 02 00 07 c2 02 00
       short x;  if (x < 3)    0a 04 23 82 02 00 17 c2 02 00
       short x;  if (x < 4)    0a 04 23 84 02 00 07 c2 02 00
       short x;  if (x < 17)   0a 04 23 90 02 00 17 c2 02 00
       short x;  if (x < 127)  0a 04 23 fe 02 00 17 c2 02 00
       short x;  if (x < 255)  0a 04 33 fe 02 00 17 c2 02 00
       short x;  if (x > 5)    0a 04 23 84 02 00 16 c2 02 00
       short x;  if (x == 7)   0a 04 23 86 06 00 17 c2 02 00
       ushort x; if (x == 2)   0a 04 23 82 06 00 07 c2 02 00

   f16 carries the constant split the way the 32-bit `agx3_cf_icmp_imm()` form
   already deposits it: bits 1..7 land in byte 3, bit 0 in byte 6 bit 4, bit 7
   in byte 2 bit 4.

   `ushort x < 2` is **not** in that list on purpose: Apple compiles it to the
   compact six-byte point (`0a 04 22 82 05 c2`) instead, so the two unsigned
   ordered relations stay refused with a non-zero constant. At 256 the
   instruction changes shape entirely, so the constant is capped at 255. Both
   refusals are asserted in `test-agx3-cf-core.c`.

### Open: 16-bit subtract-immediate

`border`, `inner_glow` and `shadow` now stop at
`iadd [d16:reg1 s0=16:reg s1=16:imm1.neg]` -- `gradientLength - 1`, a 16-bit
subtract of a small constant. `agx3_alu_add_imm16()` encodes the *addition*;
subtraction is a different form, not a flipped field. Apple's bytes, from
`.tmp/g17p-hyprland-cf-evidence/iaddimm16.metal` (`ushort` in, `ushort` out,
one kernel per constant), are recorded here so the next attempt starts from
the witness rather than from another sweep:

    ushort y = x + 2      9f 01 56 02 02 04 08 00 15 04
    ushort y = x + 17     9f 01 56 02 02 22 08 00 15 04
    ushort y = x - 1      1f 01 56 02 02 04 04 00 13 01
    ushort y = x - 2      1f 01 56 02 02 04 08 00 13 01
    ushort y = x - 17     1f 01 56 02 02 04 44 00 13 01
    ushort y = x - 255    1f 01 56 02 02 04 fc 03 13 01

The add form carries the constant at byte 5 as `imm << 1`; the subtract form
is a different opcode byte (0x1f, not 0x9f) and carries `imm * 4` across bytes
6 and 7, with a different trailing pair. Identify the form with
`.tmp/g17p-fcmpsel-evidence/tools/decomp.c` before binding any of it.

### Open: the other two

- `hmul [d16:reg1 s0=16:reg1 s1=16:imm80]` and `... s1=16:imm64` (blur1,
  blur2). The in-tree `hmul` path requires both sources to be registers
  (`agx3_pack_binary.c`, the AGX3B HMUL block); a minifloat immediate operand
  has no witness yet.
- `fcmpsel [d16:reg1 s0=32:reg14 s1=16:imm48 s2=16:imm255 s3=16:imm0]` (ext)
  and `fcmpsel [d32:reg2 s0=32:reg22.cache s1=16:imm64 s2=16:imm0
  s3=32:reg2.discard cond=1]` (glitch). These are *mixed-width* selects -- a
  32-bit compare source with 16-bit immediate operands -- which the tied and
  untied 32-bit `fcmpsel` work above does not cover.
