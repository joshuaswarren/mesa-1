# AGX3 (G17P) shader packer: what it cannot encode yet

**Status: the SSBO limit below is GONE, so is the `fcmpsel` one, and so are the
32-bit rounding modes. Two texture-side gaps and one saturate gap remain.**

> **CORRECTION (2026-09-10).** The previous revision said "the float source
> modifiers" were done, without qualification. **That was too broad, and it was
> wrong for a whole instruction family.** `f30`/`f31` were modelled on
> `f32log2`, `f32exp2` and `f32rcp`, but *not* on `f32rsqrt` or `f32sqrtstep`,
> whose encoders had no abs/neg parameters at all and whose packer handlers
> explicitly refused `src.abs || src.neg`. So `sqrt(abs(x))` and
> `inversesqrt(abs(x))` -- the standard safe forms of both idioms -- failed
> every shader that used them closed, while `inversesqrt(x + 1.0)` and
> `normalize()` passed and made the family look done. Both are now closed
> against Apple's own bytes; see "The rsqrt family's source modifiers" below.
>
> The `iadd` narrow-operand row further down was likewise recorded as fixed
> when only *half* of it was. See "The `iadd` widening fix was partial" below.
>
> The lesson is the one this file already teaches in its retraction: a claim
> phrased over a *category* ("the float source modifiers") is not supported by
> evidence gathered on *instances* of it. Name the forms you actually tested. Every row of the original matrix compiles, including
the ones marked **no**, and so do device atomics. The measurement that produced
the matrix is preserved verbatim further down, because the *method* is what
makes it re-runnable, and because the retraction it carries is still worth
reading before theorising about a wrong value on this ISA.

**Every vkQuake pipeline now compiles: 385 of 385, 248 of 248 unique
pipelines**, up from 67 of 385, with no packer refusal anywhere in the run. See
"vkQuake" below for the exact reproduction.

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
| `exp2(-(d * d * z * z))` — a negated exponential source              | yes§    |
| `exp2(-x)` with `x` read again — negate over a **live** source       | yes§    |
| `log2(abs(x))` — an absolute-value source, live or dead              | yes§    |
| `floor(vec2(16384.0) * uv)` — a **32-bit** floor                     | yes§    |
| `ceil(vec2(64.0) * uv)` — a **32-bit** ceiling                        | yes§    |
| `clamp(texture(tex, uv) * colour, 0.0, 1.0)`                        | **no**† |
| `texture(tex, uv, 0.5)` — LOD **bias**                              | yes‡    |
| `texture(tex, uv, -1.0)` — **negative** LOD bias                    | yes‡    |
| `texture(tex, uv, vcolor.x)` — per-lane LOD bias                    | yes‡    |
| `textureLod(tex, uv, 2.0)` / `textureLod(tex, uv, vcolor.x)`        | yes‡    |
| `textureGrad(tex, uv, dFdx(uv), dFdy(uv))`                          | yes‡    |
| `texture(sampler2DArray, vec3(uv, layer))`                          | see below |
| `texture(samplerCube, n)`                                           | see below |
| `textureGather(tex, uv, comp)`                                      | yes‡    |
| `texture(sampler2DShadow, ...)`                                     | **no**§ |
| `textureGrad` on a 2D **array**                                      | **no**§  |

\* This row was **no** until the 32-bit reciprocal was encoded; see below.
\*\* These rows were **no** until the 32-bit tied `fcmpsel` was encoded; see
below. The last three are new probes (`f14`, `f15`, `f16`) added with it.

\*\*\* These rows were **no** after the tied minimum and maximum landed: two
further gaps, described in "The untied select was a second gap" and "Which
side the constant is written on" below. New probes `f17`-`f22`.

§ These five rows were **no** until the float source modifiers and the 32-bit
rounding modes were encoded; see "The source modifiers and the 32-bit rounding
modes" below. New probes `f23`-`f28`. Two of them, `log2(abs(x))` and the
32-bit `ceil`, were refusing in the live GNOME session; the other three are
vkQuake's fog term and its quantized lightmap coordinate.

† Not a `fcmpsel` gap at all: this shape reaches the packer as `fmul.sat`, a
saturating float multiply, and it is the *saturate modifier* that has no
grounded encoding (`agx3_pack_binary.c`, the generic ALU fallback names
`fmul`). Recorded here because the probe is new and the distinction is easy to
lose.

‡ Closed 2026-09-07. See "The bias and gradient sample forms" below for the
byte tables and for what actually blocked them (a NIR-side software-bias
lowering, not the ISA).

§ Shadow compare (f143) and the immediate-array-layer form (f77 = 1) are now
*grounded* — the byte tables are below — but not implemented.

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

## The source modifiers and the 32-bit rounding modes

Four shapes failed the whole shader closed and turned out to be one field
family. Two were refusing on the live desktop:

```
agx3_pack_binary.c:4854: no grounded AGX3 encoding for log2
  [d32:reg8 s0=32:reg8.abs cond=0 chan=1]
agx3_pack_binary.c:4893: no grounded AGX3 encoding for ceil
  [d32:reg12 s0=32:reg12 cond=0 chan=1]
```

and two in every vkQuake fragment shader:

```
agx3_pack_binary.c:4832: no grounded AGX3 encoding for exp2
  [d32:reg2 s0=32:reg2.neg.discard cond=0 chan=1]
agx3_pack_binary.c:4874: no grounded AGX3 encoding for floor
  [d32:reg12 s0=32:reg12.discard cond=0 chan=1]
```

All four are the single-source form that `f32rcp`, `f32rsqrt`, `f32log2` and
`f32exp2` already use. Apple's own bytes, one kernel per property, each a
single-field difference from its control:

```
o[i] = exp2(a[i])         af 02 56 00 02 00 b0 40 00 00
o[i] = exp2(-a[i])        af 02 56 00 02 00 b0 40 01 00   f31
o[i] = exp2(fabs(a[i]))   af 02 56 00 02 00 b0 c0 00 00   f30
o[i] = exp2(-fabs(a[i]))  af 02 56 00 02 00 b0 c0 01 00   both
o[i] = log2(-a[i])        2f 02 56 00 02 00 b0 40 01 00
o[i] = log2(fabs(a[i]))   2f 02 56 00 02 00 b0 c0 00 00
o[i] = rsqrt(-a[i])       af 01 56 00 02 00 b0 40 01 00
o[i] = rint(a[i])         2f 00 56 00 02 00 b0 40 00 00   f43 absent
o[i] = trunc(a[i])        2f 00 56 00 02 00 b0 40 06 00   f43 = 1
o[i] = ceil(a[i])         2f 00 56 00 02 00 b0 40 04 00   f43 = 2
o[i] = floor(a[i])        2f 00 56 00 02 00 b0 40 02 00   f43 = 3
o[i] = floor(-a[i])       2f 00 56 00 02 00 b0 40 03 00   f43 = 3, f31
```

f31 negates the source, f30 takes its absolute value, and the two are
separable because `exp2(-fabs(x))` sets both. The 32-bit rounding form is the
half one with the width selectors f10 and f26 raised from 1 to 2 — the same
relationship `f32log2` has to `hlog2` — with f43 selecting the mode. Every
field vector above was recovered with `tools/decomp.c`, which re-composes and
compares, so they are facts about the encoder rather than readings of a
disassembler.

### This is what the f30 note was missing

The comment beside `agx3_alu_f32log2` recorded f30 as a bit set on exactly one
shape out of nineteen witnesses — the `f32log2` of Metal's `pow(x, y)` — and
deliberately did not model it. f30 is the **absolute-value modifier**. `pow` is
defined for a negative base with an integral exponent, so its lowering takes
`log2(|x|)`; `powr()`, which promises a non-negative base, leaves f30 clear,
and so does the hand-written `exp2(log2(x) * y)`. Witnessed directly:

```
pow(a[i], b[i])   log2 is  2f 02 56 06 03 00 92 c0 00 00   f30 set
powr(a[i], b[i])  log2 is  2f 02 56 04 03 08 b0 40 00 00   f30 clear
```

It is not a domain flag, and the shape NIR produces genuinely does not want it.

### Why floor and ceil do not fall through to the funop path

`agx3_map_funop` already claims `FLOOR`, `CEIL`, `TRUNC`, `LOG2` and `EXP2`,
and `agx3_funop_encode` already handles 32-bit operands with f30/f31 and f43 —
so the obvious minimal fix was to make the fail-closed handlers *skip* rather
than refuse, exactly as two half-select handlers were fixed earlier. **Do not
do that.** `agx3_funop_core.c`'s AGX3b baseline is

```c
static const struct agx3_field_kv b_rint_like[] = {
   KV(9, 2), KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1), KV(40, 1),
};
```

f29 and f40 are folded in as if they were form constants. Measured directly
through the encoder:

```
floor wait=0 no-mods   2f 00 56 00 02 00 b0 40 02 00
floor wait=slot6       2f 00 56 00 02 00 b0 40 02 00     <- identical
floor wait=slot1       2f 10 56 00 02 00 b0 40 02 00     <- slot 6 still set
```

So a funop can never *not* wait scoreboard slot 6, and can never describe a
live source: it always declares f29, "this was the last use". The over-wait is
conservative and harmless; claiming a live register is dead is a silent
miscompile. This is the same standing defect "Baseline wait and lifetime bits
in `agx3_alu_compose()`" describes, in a second file. The dedicated encoders
model f11, f27, f29 and all six wait slots instead, each witnessed:

```
float x = a[i]; o[i] = floor(x) + x    2f 00 56 04 03 00 92 40 02 00
o[i] = floor(a[i]) * 4 + floor(b[i])   2f 10 54 04 03 08 b0 40 02 00
```

`agx3_funop_core.c` is still reached by 32-bit `trunc`, `dFdx` and `dFdy`, and
still has this defect. It was left alone because no measured refusal points at
it and fixing it changes the bytes of instructions this session cannot test.

### The probes lied before they told the truth

`f23`, `f25`, `f26` and `f28` reported **ENCODES on the baseline** in their
first form, which would have been a false pass on a shape that was demonstrably
refusing in vkQuake. Two causes, both worth remembering:

- **`exp()` is not `exp2()` here.** NIR lowers `exp(y)` to `exp2(y * log2 e)`,
  and a leading negation folds into that constant rather than onto the
  exponential's source. The refused shape only appears when nothing follows the
  negation, so the probe has to write `exp2(-(...))` even though vkQuake writes
  `exp(-(...))` — in vkQuake the density is a push constant and the fold lands
  elsewhere.
- **`common.vert` writes constants into everything but `texcoord.xy`.** It
  emits `texcoord = vec4(p, 0.0, 1.0)` and `vcolor = vec4(p, 0.5, 1.0)`, so any
  probe built from `.z` or `.w` is link-time constant-folded down to nothing.
  The dump showed a fragment shader consisting of one texture sample and four
  exports. **Read the shader dump, not just the pass/fail line**, when a probe
  passes that should not.

## Five gaps closed 2026-09-10, measured under the G17P drm-shim

Reproduced with a two-arm harness: the same client, the same binary, the same
host, changing only `ASAHI_SHIM_CHIP`. **`g17p` selects G17P only for that
exact string** -- anything else, unset included, silently falls through to
`params_g13g` and your "test" becomes a second control. The AGX2 arm needs the
`params_g13g` repair described under "The control arm" below.

Every encoding below is Apple's own, obtained with `metal-nt -arch
applegpu_g17p`, decomposed with `tools/decomp.c` (which re-composes and
compares, so a reported field vector is a fact about the encoder), and each
field pinned by a sweep that varies **one field at a time** against a control.

### The rsqrt family's source modifiers

`f32rsqrt` and `f32sqrtstep` take f30 (absolute value) and f31 (negate) in the
same bit positions the funop family uses -- but their encoders had no
parameters for them, so `sqrt(abs(x))` and `inversesqrt(abs(x))` were refused
outright. One kernel per modifier, each a single-field difference:

```
o[i] = rsqrt(a[i])          af 01 56 00 02 00 b0 40 00 00
o[i] = rsqrt(-a[i])         af 01 56 00 02 00 b0 40 01 00   f31
o[i] = rsqrt(fabs(a[i]))    af 01 56 00 02 00 b0 c0 00 00   f30
o[i] = rsqrt(-fabs(a[i]))   af 01 56 00 02 00 b0 c0 01 00   both
o[i] = sqrt(a[i])           2f 01 56 04 03 00 92 40 00 00
o[i] = sqrt(fabs(a[i]))     2f 01 56 04 03 00 92 c0 00 00   f30
```

`-fabs` setting both is what makes the two separable. NEG is deliberately not
wired on the square-root step: `sqrt` of a negative is NaN, no witness
exercises it, and an unwitnessed bit is worse than a refusal. The half forms
have no modifier witness either and stay closed.

`f32rcp` already carried both and needed no change -- verified, not assumed.

**A trap worth naming.** The first attempt still refused, because the handler
validated its source with `agx3_vary_reg32()`, which itself returns false for
`.abs`. A helper can be the thing rejecting the modifier you just added.

### `packHalf2x16`: the multiply's destination width is independent

Refused: `fmul [d16:reg2 s0=32:reg2.discard s1=16:imm32]`. f10 selects the
**destination** width and f26 the source's, independently, exactly as on
`f32rcp`. Apple emits the mixed-width multiply directly:

```
o[i] = a[i] * 0.5f        (float o)   19 a1 15 03 80 c0   f10 = 2
o[i] = half(a[i] * 0.5f)  (half  o)   11 a1 15 03 80 c0   f10 = 1
o[i] = half(a[i] * 2.0f)              11 c1 15 03 80 c0   f17 = 64
```

So the **optimizer was right and the packer was wrong**. `agx_optimizer.c`'s
FMUL exception in `agx_optimizer_fmov_rev()` folds the `f2f16` into the
multiply and produces exactly this shape; a proposed fix to block that fold, or
to widen the 16-bit source, would have papered over a form the hardware has.

Two blockers had to come out, not one: an *earlier* `fmul_mixed_half` block
claimed any 16-bit-destination FMUL and then refused it because the operand was
an immediate, so the minifloat path below it was unreachable. It now requires
both operands to be registers.

> Beware f35 here. It sits next to f10 and it moved between the two witnesses,
> which invites reading it as a second width selector. It is the **source
> register**: `agx3_alu_fmul_minifloat_mod()` says `{35, src}`, and it differed
> only because register allocation differed. Two samples and a plausible story
> is precisely the trap the retraction below documents.

### The `iadd` widening fix was partial

Refused: `iadd [d32:reg2 s0=16:reg2.discard s1=16:imm7]` -- a 16-bit register
plus an immediate, widened to 32 bits. The register+register form encoded, so
the earlier fix looked complete; **it was partial, not regressed.** The
encoder was already right (`test-agx3-iadd-core.c` had carried the exact bytes
for a year); only the packer gate was missing.

```
int(short x) + 1     9f 01 56 00 02 02 00 08 15 0c   f15 = 1
int(short x) + 2     9f 01 56 00 02 04 00 08 15 0c   f15 = 2
int(short x) + 7     9f 01 56 00 02 0e 00 08 15 0c   f15 = 7
int(short x) + 17    9f 01 56 00 02 22 00 08 15 0c   f15 = 17
uint(ushort x) + 7   9f 01 56 00 02 0e 00 08 15 04   f43 clear
```

f15 is the slot-A immediate, f43 slot B's sign-extension.

**A bare 16-bit integer source is SIGN-extended.** That is not a guess and not
a coin flip: `agx_optimizer_signext()` fuses a SIGNEXT into the add by dropping
it and *asserting* the surviving source carries no ABS, and `agx_pack.c`
spells the same rule for AGX2 as `zext = abs || immediate`. Getting this
backwards is a silent miscompile, so it is worth stating where the rule lives.

This is the **highest-frequency codegen refusal in a real Vulkan CTS run**.

### The negated integer source was a shifted subtract

Reported as `iadd [d32:reg4 s0=32:reg0 s1=32:reg0.neg]` and read as a
degenerate `x + (-x)` that should have folded to zero. **It is not.** The
refusal printer did not show `shift`, and the instruction is `shift = 3`:
`x - (x << 3)`, the strength-reduced form of `x * -7`. There is no optimizer
bug and nothing folds to zero.

There is also no integer negate field to find. Apple compiles `a - b` and
`a + (-b)` to byte-identical `isub`; the missing piece was the **shift on slot
B**, f54, which the subtract carries exactly as the add does:

```
o[i] = a[i] - (c[i] << 0)   1f 01 56 00 02 00 10 a8 17 05   f54 absent
o[i] = a[i] - (c[i] << 1)   1f 01 56 00 02 00 10 e8 16 05   f54 = 1
o[i] = a[i] - (c[i] << 2)   1f 01 56 00 02 00 10 a8 16 15   f54 = 2
o[i] = a[i] - (c[i] << 3)   1f 01 56 00 02 00 10 e8 16 15   f54 = 3
o[i] = a[i] - (c[i] << 4)   1f 01 56 00 02 00 10 a8 16 05   f54 = 4
```

Apple stops emitting this form at N = 5, so the shift is capped at four rather
than extrapolated. The ADD keeps its existing shifted path
(`agx3_iadd_encode`) untouched: re-routing a shape that already encodes is
what cost two hardware probes before.

**`mask`, `format` and `shift` are now printed by the refusal reporter.** Two
of the five gaps here were misdiagnosed from an operand list that omitted
them, which is a full round trip each time.

### Register spilling was capped at scalars

`stack_store [d0:none0 s0=32:reg16.discard s1=16:imm0 mask=0xf fmt=2]`. AGX3's
grounded thread-private-storage form moves **one 32-bit component per
instruction** (`agx3_translate_static_spill()` admits `mask == 1` only), so any
*vector* spill or scratch access failed the whole shader closed -- regardless
of what the shader computed. That silently capped shader complexity across the
entire driver, which makes it the widest-reaching gap of the five.

Closing it needed no new encoding. `mask` is always contiguous from component
zero (both producers build it with `BITFIELD_MASK`), components live in
consecutive registers and their stack slots are consecutive, so N scalar
accesses at `offset + 4*c` are exactly equivalent. `agx3_split_vector_stack_access()`
runs after register allocation -- which covers **both** producers, real spills
from `agx_lower_spill()` and NIR scratch from `agx_emit_{load,store}_scratch()`
-- and before the last-use pass, so each component's hints are computed against
the instructions that actually survive.

A second blocker sat behind it: the gate refused `data.discard`, which is set
on essentially every real spill, since a spilled value is dead after its store.
f27, which `src_kill` drives, *is* that last-use bit -- the same field the ALU
forms set for `AGX3_ALU_HINT_DISCARD` -- so it is carried now rather than
refused. `.cache` stays refused: this form models no cache hint, and claiming a
live register is dead is a silent miscompile.

**Still open on this path:** a *dynamic*-offset vector access
(`stack_load [d32:reg16 s0=32:reg4.discard mask=0xf]`, a runtime-indexed
scratch array) is the register-indexed `ldtps`/`sttps` family, which
`agx3_local_tps_core.c` models but which nothing wires to `STACK_LOAD` /
`STACK_STORE`. Splitting it would need an add per component after RA. Not
attempted here.

### Open, and grounded enough to start: `.abs` on a mixed-width `fcmpsel`

`(abs(a) < b) ? 1.0h : 0.0h` refuses:

```
fcmpsel [d32:reg8 s0=32:reg0.abs.discard s1=16:imm32 s2=32:reg2.discard
         s3=16:imm0 cond=5]
```

It is tempting to assume this is another f30-style modifier bit, as it was for
the rsqrt family. **It is not, and that is the useful finding.** Adding `.abs`
changes the *endpoint and the instruction length*:

```
o[i] = (a[i]       < b[i]) ? 1.0h : 0.0h
   12 03 19 05 22 b0 03 c0 20 80                    form 134 fsel ep20, 10 B
o[i] = (fabs(a[i]) < b[i]) ? 1.0h : 0.0h
   12 03 19 05 23 b0 03 00 20 80 00 80 01 00        form 142 fsel ep28, 14 B
o[i] = (fabs(a[i]) > b[i]) ? 1.0h : 0.0h
   12 03 19 05 23 b0 02 00 20 80 00 80 01 00        form 142 fsel ep28, 14 B
o[i] = (fabs(a[i]) < b[i]) ? 1.0f : 0.0f
   02 01 1b 05 23 b0 03 00 20 80 00 80 01 00        form 144 fsel ep30, 14 B
o[i] = (-a[i]      < b[i]) ? 1.0h : 0.0h
   12 05 19 03 22 b0 02 c8 20 80                    form 134 fsel ep20, 10 B
```

So NEG stays on the ten-byte endpoint (f35/f52 swap, f65 and f110 move) while
ABS moves to a fourteen-byte one carrying f47. Whoever picks this up needs to
map ep28/ep30 as forms in their own right -- roughly twenty fields each, one
sweep per field -- not to add a bit to ep20. Doing the latter would produce a
plausible-looking encoding that is a different instruction.

This is the shape `textureGrad(samplerCube, ...)` reaches through cube-to-2D
lowering, and the same family as the Hyprland `ext`/`glitch` mixed-width
selects recorded further down.

### The control arm

`params_g13g` in `src/asahi/drm-shim/asahi_noop.c` was a six-field stub with
`num_clusters_total = 0`, which makes `agx_chip_from_params()` return
`AGX_CHIP_UNKNOWN` and the device refuse to open. Without the real T8103
identity there is no AGX2 arm at all, and every G17P result is unfalsifiable.
Every non-architectural field is held equal to `params_g17p` deliberately, so
the two arms differ only by generation.

### Not verified

- **No hardware.** Nothing here ran on a G17P. Every claim is an encoding
  claim, checked byte-for-byte against Apple's compiler and against the
  in-tree encoder, and nothing more.
- `EXT_vertex_input_dynamic_state` with a sub-32-bit attribute format is
  reported to hit the mixed-width `iadd` shape through the draw-time prolog.
  The encoding gap is closed, but a dynamic-vertex-input harness here compiled
  no shader at all under the shim (3 lines of `AGX_MESA_DEBUG=shaders` output
  against 204 for an ordinary graphics pipeline), and the same harness passed
  with the fix reverted. **That path was never exercised, so no claim is made
  about the extension.**
- The immediate-zero-minuend `isub` (`-a`) has no reproducer here: the obvious
  `uint(-int(x))` shader encodes on both arms already.

## vkQuake

vkQuake is patched with a `-shader-sweep` switch that creates every pipeline in
the game, records `SWEEP <kind> <name> OK|FAIL <VkResult>` per pipeline and a
`SWEEP-SUMMARY` line, and keeps going past a failure instead of calling
`Sys_Error`. The patch and the driver scripts are in
`.tmp/g17p-vkquake-evidence/`.

```sh
docker exec -e MESA_BUILD=/src/_build-full vkquake-game /vkq/run-sweep.sh
docker exec -e MESA_BUILD=/src/_build-full vkquake-game /vkq/sweep-detail.sh
```

The second adds `AGX_MESA_DEBUG=shaders`, so each refusal names its
instruction and operand shape.

| measurement                    | before  | after   |
| ------------------------------ | ------- | ------- |
| pipeline creations             | 67/385  | 385/385 |
| unique pipelines by name       | 60/248  | 248/248 |
| packer refusals in the run     | 318     | 0       |

Before, the refusals were exactly two opcodes over 318 pipelines: `exp2` with a
negated source (276) and a 32-bit `floor` (42). Both are closed above. Nothing
here says vkQuake *renders* — the sweep runs under the drm-shim and only
qualifies pipeline creation.

### Sample count and swapchain format

The sweep used to pin one sample count and one swapchain format, which is a
narrow gate: a pipeline that refuses at the configuration the game actually
selects breaks it just as completely. `VKQ_SWEEP_SAMPLES` and
`VKQ_SWEEP_FORMAT` now choose them, so one build covers the space.

| samples | format          | result  | refused                        |
| ------- | --------------- | ------- | ------------------------------ |
| 1       | B8G8R8A8_UNORM  | 385/385 | none                           |
| 1       | B8G8R8A8_SRGB   | 385/385 | none                           |
| 1       | R8G8B8A8_UNORM  | 385/385 | none                           |
| 1       | R8G8B8A8_SRGB   | 385/385 | none                           |
| 2, 4, 8 | either          | 366/385 | 17 `texture_sample`, 2 `texture_load` |

`vid_fsaa` defaults to `"0"`, so the shipping configuration is the clean one.
Under MSAA the nineteen refusals are all one shape, and it is the third gap
already listed above rather than a new one:

```
no grounded AGX3 encoding for texture_sample
  [d32:regN s0=32:regN s1=16:imm0 s2=64:uni16 s3=32:regN
   s4=16:regN.discard s5=16:null0 cond=0 chan=4]
```

Source 2 is a *uniform* heap base, not the immediate zero
`agx3_emit_texture_sample` requires — the bindless texture-heap descriptor
form. The cause is upstream of the packer:
`agx_nir_lower_g17p_combined_sampler.c` disowns a shader unless every texture
op is `nir_texop_tex` or `nir_texop_txl`, and an MSAA resolve fetches with
`txf_ms`. One multisample fetch therefore drops the whole shader onto the
bindless path. The affected pipelines are the resolves and moment composites:
`wboit_resolve`, `mboit_resolve`, `basic_blend`, and the
`world`/`alias`/`md5`/`md5_8`/`particles`/`sprite` `_mboit_composite`
variants.

Closing it is the same work as `f05_texture_bias` and `f12_texture_grad`:
either teach that pass to keep the op, or ground the bindless-heap sample
form.

## Native Gallium's texel fetch: closed without AGX2 addressing

`asahi-g17p-gallium-blit` reproduces the whole thing offline under the no-op
DRM shim -- the 4x8 `B8G8R8A8_UNORM` to `R8G8B8A8_UNORM` `util_blitter_blit`
that core 983 died on. With `AGX_MESA_DEBUG=shaders` the fragment program
refuses:

```
agx3_pack_binary.c:4335: no grounded AGX3 encoding for texture_load
  [d32:reg8 s0=32:reg8.discard s1=16:reg4.discard s2=64:uni0
   s3=32:reg0.discard s4=16:imm0 s5=16:null0 cond=0 chan=4]
```

Three of those operands were wrong at once, and they all followed from **one**
cause: u_blitter carries the mip level in a varying, so the `txf` has a
**dynamic LOD**. `agx_nir_needs_texture_crawl()` returns true for a `txf` with
a non-zero LOD, which makes `agx_nir_lower_bindings()` force the bindless path
and makes texture lowering emit a descriptor crawl to clamp the level.

1. **`s1`, the level, is a register.** It is now published through the
   byte-grounded integer `PACK_SMPIR` form and selected by the dynamic-level
   fetch form.
2. **`s2`/`s3` are a bindless heap base plus a 32-bit index.**
   G17P's indirect operand is a native global-heap index, not that AGX2 pair.
   A statically bound Gallium fetch is now restored to its immediate
   texture-state index after its descriptor crawl has been made explicit.
3. **The crawl read AGX2 descriptor bits.** The preamble did
   `load_constant_agx(desc, 1)` and pulls bits 24..27 and 28..31 out of word
   1 -- `first_level` and `levels` in the 24-byte Gen7/8 texture packet.
   Gen9's 32-byte texture descriptor has a different layout; the Gen9 helper
   now reads its level fields from word 5.

Gallium now lowers the crawl with the Gen9 descriptor helper, then restores a
statically bound fetch to its texture-state register while keeping the
descriptor address live only for the clamp. Thus the hardware fetch never
receives AGX2 base-plus-offset addressing, and the clamp is not bypassed.
`asahi-g17p-gallium-blit-encode` ratchets the exact u_blitter shape.

### The texel fetch's level is now grounded

`metal-nt -arch applegpu_g17p` on one source file, three compute kernels
differing only in the level argument to `texture2d<float>::read()`
(`.tmp/g17p-gallium2-evidence/texlod.metal`):

```text
read(gid, 0)        05800cb8b0001701000000000100   f37=1 f139=2 f29=0
read(gid, 2)        05800cb8b0001701000000080100   f37=1 f139=2 f29=128
read(gid, lod[i])   05800cb830018024000000000100   f37 absent, f139=4
```

`f29` is the level in 1/64 steps, exactly as the filtered sample's explicit
level and bias already are. The per-lane form drops the immediate operand
selector entirely and selects `f139 = 4`, with the level published into
texture operand bank 6 by a preceding `PACK_SMPIR`. The same two-field delta
appears between two *fragment* programs in `texlodfs.metal`, so it is not a
compute-only shape; those bytes also carry fragment-only `f83`/`f112`/`f124`
selectors this encoder does not model, which is why the in-tree vectors are
the compute pair.

`PACK_SMPIR`'s `f74` types the level, and it is the only field that moves
between a fragment texel fetch and a fragment filtered sample compiled from
one file (`texlodtype.metal`):

```text
t.read(uint2(uv), uint(lod))       1705540600000004f0c00c00   f74 = 3
t.sample(s, uv, level(lod))        17055406000c0004f0c00800   f74 = 2
```

So 2 is a float level and 3 an integer one. `agx3_tex_stage_typed_lod()`
carries that; `agx3_tex_stage_lod()` stays the float form every existing
caller uses.

All five vectors are pinned byte-for-byte in
`src/asahi/compiler/test/test-agx3-tex-core.c`. The integer per-lane form is
routed for `texture_load`; filtered-sample bias and gradient retain their
separate heap/staging gaps above.

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

All of this is grounded from `xcrun metal-nt -arch applegpu_g17p` (Metal
toolchain 27.1), decomposed with the production compose tables and re-composed to prove
each field vector. The harness and every `.nat` are in
`.tmp/g17p-texgrad-evidence`.

### One field, f139, selects the LOD shape

Seven fragment kernels differing only in the sample's LOD argument, all with an
immediate texture 0 / sampler 0, dest r0, coordinate bank 0:

```
t.sample(s, uv)                       05800cb8b0000000010010001200   f139 -      f156 1
t.sample(s, uv, bias(0.5))            05800cb8b0000700010010021200   f139 1 f37 1 f29 32   f156 1
t.sample(s, uv, level(2.0))           05800cb8b0000900010010081200   f139 2 f37 1 f29 128  f156 1
t.sample(s, uv, level(4.5))           05800cb8b0000900010010121200   f139 2 f37 1 f29 288  f156 1
t.sample(s, uv, bias(in.uv.x))        05800cb8b0000400010010001200   f139 3      f156 1
t.sample(s, uv, level(in.uv.x))       05800cb8b0000001010010001200   f139 4      f156 1
t.sample(s, uv, gradient2d(dx, dy))   05800cb8b0000401000010001200   f139 5      f156 -
```

So the whole family is one field. **f139 = 3 is a per-lane LOD bias**, staged
exactly like the per-lane level (f139 = 4): the same twelve-byte `PACK_SMPIR`
into operand bank 6, the same fragment marker. The two kernels above are
byte-identical apart from bytes 6 and 7. **f139 = 5 is the gradient form, and
it is the only sample shape that clears f156** — the four staged scalars are
the derivative the marker would otherwise ask for.

(Those fragment bytes carry f83 = 4 / f112 = 8. Apple's *compute* samples, and
everything Mesa emits, carry f83 = 0 / f112 = 4 instead; that difference is
orthogonal to this work and is called out under "Not closed" below.)

### f29 is a *signed* 1/64 LOD, and negative bias is what games actually use

Sweeping `t.sample(s, uv, bias(k))` over positive and negative k pins the
field completely:

```
k     1/64  0.25   0.5     1     1.5      2      4     31     32     63
f29      1    16    32    64      96    128    256   1984   2047   2047

k    -1/64  -0.5    -1    -2      -4    -31    -32    -33    -64
f29   4095  4064  4032  3968    3840   2112   2048   2048   2048
```

So f29 = k * 64 in **two's complement over 12 bits**, saturating at 2047
(+31.984375) and 2048 (-32). The positive half was already implemented; the
negative half was refused outright by an `immediate_level >= 0.0f` guard, even
though a negative mip bias is one of the most common things a real game does
(texture sharpening, and every renderer that biases against an upscaler).
`agx_minifloat_decode` spans [-31, 31], so every immediate a shader can
actually reach lands inside the representable range and no clamp is needed —
only the sign has to survive into the field. `f31_texture_bias_negative`
covers it, and `test-agx3-tex-core.c` asserts the whole sweep above.

An explicit *level* stays unsigned: `textureLod` with a negative level is
refused rather than silently encoded as a large positive LOD.

### The gradient staging

Apple's compute `t.sample(s, g[0], gradient2d(g[1], g[2]))` emits **six
`copy` instructions naming one shared staging slot**, publishing six
consecutive source registers into banks 0, 2, 8, 10, 12 and 14, and then the
sample:

```
copy 0b10210c   bank  0 <- r16      copy 5b16210c   bank 10 <- r22
copy 1b12210c   bank  2 <- r18      copy 6b18210c   bank 12 <- r24
copy 4b14210c   bank  8 <- r20      copy 7b1a210c   bank 14 <- r26
smp  05800cb8b0000401000010000100
```

`agx3_tex_encode_sample()` reproduces that sample byte for byte. The staging
copies go through the packer's existing `agx3_tex_stage_copy_hints()`, which
emits the same banks on the same shared slot but names its source with f16 /
f23 = 2 rather than the compact f17 / f23 = 3 class this particular kernel
used; that is the staging form every already-qualified G17P sample path
(plain, explicit level, array) already uses, so the gradient path inherits it
rather than introducing a new one.

Banks are half-register units, so 8/10/12/14 are four consecutive 32-bit
operand slots. The **interleaved** order (ddx.x, ddy.x, ddx.y, ddy.y) is
confirmed three more times by fragment kernels that pass literal gradients,
where the staged value is the constant itself:

```
gradient2d((0.25, 0.5), (0.75, 1.0))   bank 8 = 0.25  10 = 0.75  12 = 0.5  14 = 1.0
gradient2d((1, 2), (4, 8))             bank 8 = 1     10 = 4     12 = 2    14 = 8
gradient2d((16, 32), (64, 128))        bank 8 = 16    10 = 64    12 = 32   14 = 128
```

Mesa's `agx_collect` for `nir_tex_src_ddx`/`ddy` already lays the four scalars
out in exactly that order, so the run is copied straight through.

Every gradient witness names coordinate bank 0, so the packer rebases the
coordinates to bank 0 for a gradient sample, as it already does for an
explicit per-lane level.

### The gradient works through the bindless heap too

A compute kernel sampling through an argument-buffer `{texture2d, sampler}`
pair — the same uniform-descriptor operand class Honeykrisp's heap-indexed
path uses — emits the identical six staging copies and

```
smp  05a00cb8b8000401920390010100802000000000
     f103 = 3 / f132 = 3 (uniform), f97 = 20, f126 = 24, f139 = 5
```

which `agx3_tex_encode_sample()` reproduces byte for byte. So the gradient
mode is orthogonal to the descriptor operand class; there is no separate
"bindless-heap gradient" form to ground.

### What actually blocked both probes

Not the ISA. Honeykrisp called `agx_nir_lower_texture_early` with
`support_lod_bias = true` on every chip, so `nir_lower_tex` rewrote **every**
`texture()` into `txb(descriptor load)` and scaled every `textureGrad` by
`exp2(descriptor load)`. On Gen4 that load is the sampler's real bias; from
HAL200 onward `hk_CreateSampler` puts the bias in the hardware sampler word
(`SAMPLER_HAL200.lod_bias`) and deliberately publishes a software bias of
**zero**, so the lowering only added a descriptor crawl computing nothing —
and that crawl is what made `agx_nir_lower_g17p_combined_sampler()` disown the
shader, which then fell back to the heap-descriptor form.

`hk_compile_nir` now passes `support_lod_bias` only when
`agx_chip_sampler_generation(chip) == AGX_SAMPLER_GENERATION_GEN4`, the same
predicate `hk_CreateSampler` uses to decide whether the software field is
nonzero. A genuine API bias and explicit gradients now survive as the shader's
own operands, and `texture()` stays a plain `tex` again — which also un-blocks
the cube-to-array lowering, whose shape test only ever accepted
`nir_texop_tex`.

### textureGather

A gather is a plain implicit sample with one bit cleared and one or two set:
**f144 = 1 selects it, f145 carries the component, and the f74 mask-group bit
that every ordinary sample sets is dropped.** Five fragment kernels differing
only in the gathered component:

```
t.gather(s, uv)                          05800ca490000000010000001200
t.gather(s, uv, int2(0), component::x)   05800ca490000000010000001200   f145 -
t.gather(s, uv, int2(0), component::y)   05800cac90000000010000001200   f145 1
t.gather(s, uv, int2(0), component::z)   05800cb490000000010000001200   f145 2
t.gather(s, uv, int2(0), component::w)   05800cbc90000000010000001200   f145 3
```

`agx3_tex_encode_sample()` reproduces all four (modulo the f83 / f112 profile
byte), and `agx_nir_lower_g17p_combined_sampler()` now keeps a `tg4`. Every
witness is a plain 2D implicit-LOD gather, so an array gather, a gather with a
coordinate offset and a gather carrying any LOD argument all stay refused
rather than being encoded from an unwitnessed shape. `f29_texture_gather` and
`f32_texture_gather_alpha` cover it.

### Not closed

- **Shadow compare.** Grounded, not implemented. `depth2d`'s
  `sample_compare(s, uv, 0.5f)` is `1d800ca090002000010000001200`: f143 = 1,
  mask f75 = 1, no f74, and the reference value staged into bank 4 (an
  ordinary `ldimm` to f5 = 4). Encoding it means staging the comparator,
  which is a fifth staged operand this emitter does not have a slot plan for
  yet, so `f28_texture_shadow` still refuses.
- **The immediate array layer.** `texture2d_array::sample(s, uv, 3u)` is
  `05800cb8b0000d18010010001200`: f77 = **1** with the layer in f29 and
  f37 = 1, a different selector from the f77 = 2 register-layer form Mesa
  emits. Combined with an immediate bias the layer keeps f29 and the bias
  moves to f43 with f51 = 1 (`05800cb8b000051b010010001202`). Array plus
  gradient is `05800cb8b0000f19000010001200` (f77 = 1, f29 = 3, f139 = 5) and
  is refused today because Mesa's array path uses the register-layer form.
- **Native cube sampling.** f76 = 4 selects a cube, but the shader must first
  run `cem_calc` (three of them) and publish the face through a third
  `PACK_SMPIR` variant, f74 = 10 — distinct from f74 = 2 (level) and f74 = 4
  (array layer). Mesa lowers cube to 2D array in NIR instead.
- **f83 / f112.** Apple's *fragment* samples carry f83 = 4 and f112 = 8 where
  its compute samples, and everything Mesa emits, carry f83 = 0 and f112 = 4.
  Mesa's f112 = 4 fragment samples are hardware-proven (the desktop renders),
  so this is recorded, not changed.

## What changed earlier, and how it was measured

Two compiler fixes, both grounded byte-exactly against Apple's own machine
code (see "Getting Apple's own G17P encoding", below):

1. **`iadd` with narrow register operands.** The register-slot-A form pinned
   both operand widths to 32-bit, so `iadd d32, h_a, h_b` had no encoding at
   all and every compute shader that sums `gl_LocalInvocationID`'s three
   16-bit components was refused outright. f24 is slot A's width and f39 is
   slot B's, on both encodings; f28 and f43 are their sign-extensions.

   **This fix was PARTIAL — see "The `iadd` widening fix was partial" below.**
   It closed the register+register form only. The register+**immediate** form,
   `iadd d32, h_a, #k`, still had no path and stayed refused for another three
   days, and it is the single highest-frequency codegen refusal in a real
   Vulkan CTS run.

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
fragment probe 25 of 28. The three refusals are `f05_texture_bias` and
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
