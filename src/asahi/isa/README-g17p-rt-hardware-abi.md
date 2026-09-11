# The G17P hardware ray-traversal ABI: one complete decoded traversal

`README-g17p-scs-map.md` records what the ray-traversal scratch *addressing*
means.  This file records the thing that addressing serves: the complete
instruction sequence Apple emits for a hardware ray traversal on our part,
decoded byte for byte, with the hardware-mandated parts separated from the
compiler-chosen ones by measurement rather than assumption.

Everything here is host-side, from Apple's own translator.  See
`README-metal-witness.md` for the `metal-nt` invocation and its traps.

Raw artefacts (listings, SCS tables, the scanner, and all 40 cross-architecture
binaries) live outside the repo in
`.tmp/g17p-rt-hw-lowering-evidence/`.

## Provenance

`LISTING_rq_isect_g17p.txt` is one complete G17P traversal: **187 instructions,
1538 bytes, contiguous, zero undecoded bytes**.  `LISTING_rq_prim_g17p.txt` is
the `intersection_query` / `while (q.next())` form: 561 instructions, 4634
bytes, also zero undecoded.

Two disciplines make these facts about the encoder rather than readings of a
disassembler:

* Every reported field vector is re-composed by the production `agx3_compose`
  encoder and byte-compared before it is printed.  Only exact round-trips are
  emitted, and instruction sizes sum to exactly 1538/1538 and 4634/4634 with no
  gaps.
* The harness was **reproduced, not merely consumed**: recompiling the same MSL
  through `metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0`
  yields a file byte-identical to the pre-existing witness.

## The traversal core

```
        stscs f70=3 f68=3 f71=0     64-bit   <- acceleration-structure words
        stscs f70=3 f68=3 f71=256   64-bit   KILL
        barrier        070454480900          (f25=1 f30=2 f39=1)
        intersect_ray  5f10540000012600c74f4c00
                       f5=0 f63=96 f78=112 f84=3 f85=2 f92=126
        ldscs f70=3 f68=1 f71=263   W8  slot6 wait{6}   <- completion poll
        icmp  vs imm 64
        iadd  r1 = r2 + (r3<<3) ; iadd r1 = r1 + u64
        cndst push slot0 count2
        bn    -> exit                        (skip the loop if already done)

L:      ldscs f70=4 f68=1 f71=7     W8  base r1 slot6   <- status byte
        bit_andc  r4 &= ~0x40
        bit_or
        stscs f70=4 f68=1 f71=7     W8  base r1 KILL
        barrier        070454080900          (f25=1 f30=1 f39=1)
        continue_ray   5f1154040000          f5=4
        ldscs f70=4 f68=1 f71=7     W8  slot6 wait{6}   <- poll
        icmp  vs imm 64
        cndlt loop-top slot0
        b     -> L                           *** BACK EDGE ***

exit:   cndend pop2
        ldscs f70=3 f68=1 f71=256   W32      <- committed result word
        ldscs f70=0 f68=1 f71=24    W32      <- distance (the ray's own tmax)
        9 epilogue-mode loads (f28=1 f80=1), f70=4 f68=15, 512 stride
        stdev ; end
```

**The traversal loop is ten instructions.**  Bit 6 (`0x40`) of the status byte
is the traversal-complete flag; both polls test it with the same `icmp` against
immediate 64, and the two `icmp` encodings are byte-identical.

`rq_prim` contains three copies of this same ten-instruction loop, one per
`continue_ray` site, inside the outer `while (q.next())`.

## The hardware performs the box AND triangle tests

This is measured, not inferred, and it corrects the older reading that
`intersect_ray` is only a per-node test accelerator with software traversal.

| kernel     | total | setup | loop | after | float ops in loop |
|------------|-------|-------|------|-------|-------------------|
| `rq_isect` | 187   | 140   | 17   | 30    | **0**             |
| `rq_prim`  | 561   | 149   | 374  | 38    | 9 `fcmp` + 3 `fsel`, **0 `f32add`, 0 `f32mul`** |

All 24 float instructions in `rq_isect` (7 `f32add`, 7 `f32mul`, 10 `fsel`) are
in setup, before the first `intersect_ray` -- ray-setup precompute.  There is no
`fmin`/`fmax`/`fma`/`rcp` anywhere in either kernel.  A slab test or a
Moller-Trumbore test is tens of multiply-adds and min/max per node; there are
none.  The software `agx.triangle_test` / `agx.bounding_box_test` /
`agx.curve_test` in the RT runtime are fallbacks behind
`agc.ria.rtu_disabled()` / `agc.ria.rtt_disabled()`, plus the
intersection-function path -- not the mainline.

## The staged ray record

Two stores, `(f70=0, f68=15, f71=0)` and `(f70=0, f68=15, f71=16)`, 32 bytes.
Recovered by eight one-variable compiles in which every ray parameter is
`k + <const>` with `k` a runtime value so nothing folds.  **Each variant differs
from the baseline in exactly one byte** -- one `f32add` minifloat immediate --
so the defining register is unambiguous, and for all eight nothing rewrites that
register between its definition and the staging store.

| byte | +0 | +4 | +8 | +12 | +16 | +20 | +24 | +28 |
|------|----|----|----|-----|-----|-----|-----|-----|
|      | origin.x | direction.x | origin.y | direction.y | origin.z | direction.z | **max_distance** | not a ray field |

The record is **interleaved**: `(ox,dx) (oy,dy) (oz,dz) tmax`.

Two consequences a lowering must honour:

* **`max_distance` lands at byte +24, which is exactly the `ldscs f70=0 f71=24`
  in the epilogue.**  The committed distance is the ray's own `tmax` slot,
  written in setup and read back after the loop -- the same address, not a
  separate result field.  `agc.intersect.ray` is declared `void`, consistent
  with this.
* **`min_distance` is not staged at all.**  It is defined into a register,
  consumed by an `fcmp`, and never reaches the 32-byte block on either
  encoding.  The unit does not take `tmin`; we implement it.

Byte +28 is an integer written by an `iadd`, not a ray parameter.

## Hardware-mandated versus compiler-chosen

The discriminator is **cross-generation invariance**, not assumption.  The same
five ray-query kernels are byte-identical across G15S, G16P and G17P (AGX3b),
and byte-identical to each other but a *different program* across G17G, G17S and
G18P (AGX3c).  G14P decodes under neither AGX3 table (it is AGX2).  Those two
families also ship different RT runtimes -- see the generation map below -- so
anything invariant across the boundary is invariant across two SoC generations
*and* two different SCS layouts.

**Hardware-mandated** (invariant AGX3b <-> AGX3c):

* the instruction skeleton above, back edge and all;
* the ray record's bytes 0..24, including `tmax` at +24 (AGX3c reaches the same
  layout with a three-lane `f68=7` store instead of a four-lane one);
* the status byte at `(f70=4, f68=1, f71=7)` and its bit-6 completion flag;
* the epilogue region `f70=4, f68=15` at 512 stride.

**Compiler- or driver-chosen** (moves across the boundary):

* where the ray is staged (`f71 = 0/16` on AGX3b; `16/28/36/52` on AGX3c);
* the post-`intersect_ray` poll address (`(3,1,263)` register-indexed on AGX3b;
  `f71=51` uniform-indexed on AGX3c);
* the scoreboard slot (6 on AGX3b, 8 on AGX3c);
* ray-record byte 28 and beyond -- on AGX3c it is separately written *and read
  back*, so it is bookkeeping there too;
* instruction widths (`intersect_ray` 12->14 B, `continue_ray` 6->8 B,
  `ldscs` 14->16 B) and the barrier bytes.

This is why Apple's layout printer says an assignment "only applies to this
shader in isolation", and why that is not an obstacle: **the per-shader
variability is allocation, not semantics.**  `agc.scs.get_layout()` returns a
single driver-supplied packed `i64` (fields at 16/8/8/16/16 bits), and the
shader reaches the scratch through 27 payload-segment base pointers.  Segment
identity and each segment's internal field layout are the hardware's; placement
is ours.

## The acceleration-structure address

`intersect_ray` on G17P has **three register-shaped operands and no pointer
operand**.  A differential moving the acceleration structure from `buffer(0)` to
`buffer(5)` changed exactly two bytes, both argument-table repacking, and no
instruction encoding tracks the argument slot; declaring an extra unused
`primitive_acceleration_structure` gives a byte-identical kernel.

The address reaches the unit **through the SCS**, as the two 64-bit stores
`(f70=3, f68=3, f71=0)` and `(f70=3, f68=3, f71=256)` immediately before the
barrier.  The RT runtime's IR shows the shape: the pointer is masked to 44 bits
and mode bits are packed in at 44/48/52 before being stored to a payload
segment.  Which of the two slots takes the packed pointer and which the raw one
is **not** separated by this evidence.

## Generation map, and a correction to the evidence base

| family | encoding | RT runtime shipped | parts |
|---|---|---|---|
| AGX3b | `AGX3b.xml` | `raytracing_runtime_ria2.metallib` | G15S, G16P, **G17P** |
| AGX3c | `AGX3c.xml` | `raytracing_runtime_ria3.metallib` | G17G, G17S, G18P |

The runtime split aligns exactly with the encoding split.  **This matters
because earlier RT work used `ria3`, which is the wrong generation for our
part.**  The intrinsic's operand set differs between them:

```
ria2 (ours): agc.intersect.ray(i32, i32,            i16,i16,i16, float, i8 as(5)*, i32 x6)
ria3:        agc.intersect.ray(i32, i32 as(1)*, i16 x6,          float, i8 as(5)*, i32 x6)
```

On ria3 the acceleration structure is a **device-pointer operand**; on ria2 it
is not an operand at all.  A lowering built from the ria3 signature would be
wrong for G17P.  The correct runtime is extracted at
`.tmp/g17p-rt-bvh-evidence/ria2/ll/` (132 functions); the sibling `ll/`
directory in that folder is ria3.

## The node format the unit walks

`intersect_ray` is handed a memory address and walks the structure itself, so
the node layout is hardware-defined.  From Apple's own `AGXMetalG17P` driver
(full symbols):

* `-[AGXG17FamilyDevice buildBVHForRIA]` returns 1 unconditionally.
* `getSelectedRiaBvhGen()` defaults to **2** (`riaBvhGenOverride` initialises to
  0; `csel w0, w8, w9, gt` with `w9 = 2`).
* The version nibble is the low nibble of the byte at **acceleration-structure
  base + 0x43**, and the node-header class is selected from it: **versions 1-4
  use `RIANodeHeaderGen1`**, version 5 uses `RIANodeHeaderGen5`.  Those are the
  only two node-header classes in the binary.
* Gen 1 and gen 2 share one leaf-type table; `getCompatibilityKey` differs only
  in version stamps (`0x100`->`0x200`, `0x8007_1a15`->`0x8007_1a16`).
* It is eight-wide (`BVH8SimNode`, `buildAllocateBVH8SingleTGRia2`), with
  64-byte node headers and six 7-bit quantised components per child at bits
  57/50/43/36/29/22 -- verified instruction-for-instruction against Apple's
  `RIANodeHeaderGen1::getQuantizedChildBounds`, and already decoded exactly by
  `src/asahi/lib/agx_rt_bvh.h`.

`src/asahi/vulkan/hk_bvh.h` is two-wide, 56-byte, full-float, with type tags in
pointer low bits.  It shares nothing with the above, so **a hardware traversal
cannot be fed by the current builder**; an encoder for Apple's format is
required.  `agx_rt_bvh.h` is decode-only and supplies no encoder.

Apple's own gen-2 builder is available as Metal compute shaders with full type
names in
`/System/Library/Extensions/AGXMetalG17P.bundle/Contents/Resources/Archive_release_gen2.metallib`
(`buildBVHSingleDispatchRia2`, `encodeNodesKernelRia2`,
`assignBVHOffsetsKernelRia2`, `BVHWideNode`, `bvh8Dump`).

## The barrier rule is positional, not opcode-keyed

`f30` is the only differing field between the two forms, on both encodings:

```
AGX3b   070454480900  {f25=1, f30=2, f39=1}      070454080900  {f25=1, f30=1, f39=1}
AGX3c   171100060000  {f14=3, f15=9, f30=2, ...} 170900060000  {..., f30=1, ...}
```

The rule, identical on both encodings and audited over all five AGX3b kernels
and two AGX3c kernels (`BARRIER_AUDIT.txt`):

* **`f30=2`** -- before the **first RT op of a traversal spin loop**
  (`intersect_ray` on entry, `continue_ray` on every later re-entry), and before
  the SCS epilogue.
* **`f30=1`** -- **inside** the spin-loop body, and before `end`.

**Refuted:** the reading "`f30=2` before `intersect_ray`, `f30=1` before
`continue_ray`".  `rq_prim` has five `continue_ray` sites and only three take
`f30=1`; the other two take `f30=2`.

Two further facts.  The last two barriers of *every* kernel are not adjacent to
any RT op -- one before the 512-stride epilogue, one before `end` -- so these
are the **general memory barrier**, and `agx3_rt_update_barrier()`'s framing as
an "RT-expansion barrier" is too narrow.  And `incStk` carries **no** barrier at
any site in any kernel.

The right shape is therefore one encoder with an explicit `f30` parameter
(2 = loop-entry / epilogue scope, 1 = in-loop / end scope), selected by the
emitter from loop position -- not two encoders keyed on the RT opcode.
`agx3_rt_update_barrier()` currently hardcodes `{f25=1, f30=1, f39=1}` and can
therefore emit only the in-loop form.

## Retraction: `f68` does mean consecutive registers

An earlier revision of this work claimed the reading "`f68=15` means four
consecutive 32-bit registers starting at `f2/2`" was refuted.  **That claim was
wrong and is withdrawn.**  The eight-point ray sweep above confirms the reading
directly, and AGX3c corroborates it independently by mapping `f68=7` to exactly
three consecutive registers at bytes 16/20/24.

The apparent counter-example came from `rq_isect`, where five of the eight ray
lanes are compile-time constants that get CSE'd and their registers reused; that
reuse was misread as a broken model.  A refutation that is itself wrong is worse
than no refutation, which is why it is recorded here rather than quietly
dropped.

## Trap: a minifloat differential needs minifloat-exact immediates

The first `max_distance` sweep used 8.25 -> 8.75.  **Neither is minifloat-exact**,
Apple folded both to the same encoding, and the diff came back empty -- which
reads as "`max_distance` never reaches the instruction stream".  It does; it is
at byte +24.

Any differential that varies a minifloat immediate must use
`agx_minifloat_exact()` values, or it is silently a null test that reports a
confident negative.  This one was caught because "tmax is dead" was
semantically implausible, not because the experiment flagged anything.

## Open, with the experiment that would settle each

* **Does RIA BVH gen 2 require Quantised Triangle Blocks?**
  `GenericBackendRia<2>::supportsQtbs()` returns 1 where `<1>` returns 0, and
  `QuantizedTriangleBlock<2, indexed>` plus four `bvhRefitQtb*KernelRia2`
  kernels exist -- yet `RIANodeHeaderGen1` has no QTB addressing at all (only
  `RIANodeHeaderGen5` does) and gen-2 structures are read through the Gen1
  header.  The shader-side names `nodesWithQtbs`, `qtbEnabled`, `qtbPrefixSum`
  suggest QTBs are per-node optional and switchable at build time, but that is
  an inference from naming, not a measurement.  Settle it by reading
  `RIABuildUtils::encodeBVH`'s binding setup for a path that clears
  `qtbEnabled`, or `GenericBackendRia<2>::computeConservativeTreeSize` for a
  QTB-free branch.  Note a live Metal build **cannot** settle it on an M2 Pro
  development host: Metal would load `AGXMetalG14X` and build a G14 structure.
* **Which of `(3,3,0)` / `(3,3,256)` takes the 44-bit-packed pointer.**  The SCS
  map pins `(3,*,0)` as SEG-6 word 0, the config word, which argues for
  `(3,3,256)` as the pointer -- but that is not proven here.
* **`f70` is not a disjoint bank.**  Across every access in these kernels
  `1 << f70` equals the index-register scale, and `f70` moves with the element
  stride of the indexed record rather than the access width.  The poll at
  `(3,1,263)` and the loop byte at `(4,1,7)` are very likely the same byte
  (`263 = 256 + 7`, and the intervening `iadd`s build `(4,*,7)`'s base from the
  poll's).  Not proven; `agx3_scs_core.h`'s "role NOT established" still stands.
* The `f71` scale in the `f28:1 f80:1` epilogue mode, packed-word bits 8-17, and
  W bits 0-1 / 6-7 / 30-31 remain as recorded in `README-g17p-scs-map.md`.  The
  epilogue mode's 512 stride and `f70=4 f68=15` shape is now known to be
  invariant across AGX3b and AGX3c, which constrains but does not settle the
  scale question.
