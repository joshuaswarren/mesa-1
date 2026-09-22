# G17P: control-flow flag slots are reusable, and `sra` has no immediate value

Two Apple-compiler witnesses taken for the `dEQP-VK.graphicsfuzz.*` bucket, in
which all seven mustpass failures were *packer refusals* -- `AGX backend cannot
encode fragment shader for G17P`, `VK_ERROR_FEATURE_NOT_PRESENT` out of
`vkCreateGraphicsPipelines` -- rather than wrong pixels.

Toolchain: MetalToolchain **v27.1.5194.15**, `Apple metal version 32023.917
(metalfe-32023.917)`, target `air64-apple-darwin27.0.0`.

Invocation for every witness below:

    MT=$(dirname $(xcrun -f metal))
    xcrun metal -c X.metal -o X.air
    xcrun metallib X.air -o X.metallib
    "$MT/metal-nt" -arch applegpu_g17p -platform_version macos 27.0 27.0 \
        -N X_<kernel>.mtlp-json X.metallib -o X_<kernel>.nat
    # then __TEXT,__compute -> nested Mach-O -> __TEXT,__text

Decoded with `decomp`/`scan.py` (the structural decomposer described in
`src/asahi/isa/README-metal-witness.md`), built against the in-tree
`agx3_compose` tables. Only exact re-compose round-trips are reported, so each
field vector below is a fact about the encoder.

---

## 1. Deep loop nests reuse flag slots (fixes 5 of the 7)

Question: G17P has six control-flow flag slots (`AGX3_CF_MAX_FLAG_SLOT 5`).
`agx3_try_control_flow()` handed loop depth *d* the slot *d-1* and refused the
shader once that reached 6, so any shader with seven or more nested loops was
refused outright. Is a slot a per-depth resource?

Answer: no. Apple reuses them freely.

`g16g-nested-loops.metal` -- the probe already in this directory, seven and
eight data-dependent nested loops -- recompiled for `applegpu_g17p`
(940 and 1064 bytes of `__text`; sha256 prefixes `4ee9c8e3fd7d436d` and
`720c4c3c4dd647e6`). Every loop test is a `cndlt` (form 31), whose field 1 is
the flag slot:

    loops7   +22c 8f045406  cndlt 1:1 12:2     slot 1
             +26a 8f045406  cndlt 1:1 12:2     slot 1
             +2a8 8f045406  cndlt 1:1 12:2     slot 1
             +2e6 8f045406  cndlt 1:1 12:2     slot 1
             +324 8f045406  cndlt 1:1 12:2     slot 1
             +362 8f045406  cndlt 1:1 12:2     slot 1
             +380 8f045422  cndlt 0:1 12:2     slot 0   (outermost)

    loops8   seven `cndlt 1:1` in slot 1, then one `cndlt 0:1` in slot 0

Eight nested loops, two distinct slots. Slot 1 carries seven different loops'
exit predicates in one program.

A second shape -- eight nested loops whose body stores to memory (kernel
`loop8` of the auxiliary probe reproduced at the end of this file, 606 bytes,
sha256 prefix `a5588320bf21a89b`) -- shows Apple spreading them wider when its
scheduler hoists loop-invariant compares: slots `2,1,0,5,4,0,0,0`. Even there
slot 0 carries four of the eight. Either way the slot is an ordinary
allocatable resource, not a function of depth.

Our own emission never needs more than one live predicate at a time: in
`agx3_try_control_flow()` the compare (`agx3_emit_cf_compare`) is immediately
followed by the `cnd` that consumes it, with nothing in between, so the slot
is dead one instruction later.

Consequence in the tree: the G17P slot assignment now wraps modulo six, exactly
as the G16G branch beside it already did. Depths one through six are unchanged
arithmetically (`x % 6 == x` there), so no shader that already encoded changes
a byte; only the refusal past depth six is replaced.

## 2. `sra` takes its shifted value in a register only (fixes the other 2)

Question: `1 >> x` -- a constant value shifted by a register count -- reached
the packer as `asr [d32:reg2 s0=16:imm#1 s1=32:reg6]` and was refused
("no grounded AGX3 encoding for asr"). Is there an immediate-value form?

Answer: not for the *arithmetic* shift. `g17p-shift-immediate-value.metal`
(this directory), one kernel per shape, `__text` 86 bytes each:

    reg_reg     in[0] >> in[1]   a7015408000008d8e200
                                 form 312 sra ep0 size10
    immneg_reg  (-7)  >> in[0]   1cf922fe1e0cff0f      form 204 ldimm ep1, 14:4294967289
                                 a7015408000400d8e200  form 312 sra ep0 size10
    imm1_reg    1     >> in[0]   a700540800020000d0120400
                                 form 314 srl ep0 size12 : 15:1
    imm5_reg    5     >> in[0]   a7005408000a0000d0120400
                                 form 314 srl ep0 size12 : 15:5
    uimm1_reg   1u    >> in[0]   byte-identical to imm1_reg

`sra` (form 312) is register/register in every case. For a NEGATIVE constant --
the only case where arithmetic and logical shift differ -- Apple spends an
`ldimm` to put the constant in a register and then uses the ordinary
register/register `sra`. The immediate-value form it uses for 1 and 5 is a
different opcode, `srl` (form 314, field 15 = the value), which is only correct
because those constants are non-negative.

Consequence in the tree: `agx_optimizer_inline_imm()` no longer folds a
constant into source 0 of `AGX_OPCODE_ASR` on AGX3B, so the `MOV_IMM` survives
and instruction selection reproduces Apple's `ldimm` + `sra reg,reg` pair.
Source 1 (the count) still folds -- that one does have an immediate form,
`agx3_alu_asr_imm32`/`imm16`.

---

## Hardware result

`deqp-vk` 1.4.6.2 on G17P (A18 Pro), `MESA_SHADER_CACHE_DISABLE=1`, the seven
`dEQP-VK.graphicsfuzz.*` mustpass failures of this bucket:

    baseline                      0/7
    flag-slot wrap only           5/7   (the five nested-loop cases)
    ASR immediate-value fix only  2/7   (cov-simplify-right-shift-greater-than-zero,
                                         find-msb-from-lsb)
    both                          7/7

These tests verify an exact framebuffer colour, so the deep-loop programs are
confirmed to execute correctly under slot reuse, not merely to compile.

---

## Auxiliary probe source (second deep-loop shape)

    #include <metal_stdlib>
    using namespace metal;
    kernel void loop8(device int *out [[buffer(0)]],
                      device const int *in [[buffer(1)]])
    {
        int one = in[0]; int a = 0;
        for (int i0=0;i0<one;i0++)
         for (int i1=0;i1<one;i1++)
          for (int i2=0;i2<one;i2++)
           for (int i3=0;i3<one;i3++)
            for (int i4=0;i4<one;i4++)
             for (int i5=0;i5<one;i5++)
              for (int i6=0;i6<one;i6++)
               for (int i7=0;i7<one;i7++)
                 { out[a & 63] = i7; a++; }
        out[64] = a;
    }

The store in the body is load-bearing. Give the nest a body the optimiser can
collapse -- a bare `a++` -- and Apple folds the whole thing into arithmetic:
three, seven and ten nested loops then all compile to the same 150 bytes with
no loop left in them, which says nothing at all about flag slots.
