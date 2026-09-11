# The G17P ray-traversal SCS map: what is measured, what is open

`stscs`/`ldscs` address the ray-traversal scratch by `(bank f70 in 0..4, byte
offset f71)`.  This file records what that addressing means, how each claim was
obtained, and -- just as importantly -- which claims are *not* settled and what
experiment would settle them.  Encoders live in `compiler/agx3_scs_core.h` and
`compiler/agx3_rt_core.h`; this is the evidence behind them.

Everything below is host-side, from Apple's own translator.  See
`README-metal-witness.md` for the `metal-nt` invocation and its traps.

**Read `README-g17p-rt-hardware-abi.md` alongside this file.**  It carries one
complete traversal decoded byte for byte, the staged-ray field map, the
completion protocol, and -- the part that changes how this file should be read
-- a hardware-mandated versus compiler-chosen split derived from AGX3b/AGX3c
invariance.  It also records that earlier RT work used the `ria3` runtime, which
is the **wrong generation for G17P**: our part ships `ria2`, and the
`intersect_ray` operand set differs between them.

## The layout is per-shader.  Do not hardcode it.

Apple's own layout printer says the assignment "only applies to this shader in
isolation", and measurement agrees: curve param sits at bank 2 +896 in one
kernel and barycentrics at bank 3 +512 in another, under different tags.
Across 27 kernels **only bank 0 offset 0 (the staged ray) was invariant**.

A lowering must therefore *read* the layout rather than assume it.  A map
recorded from one kernel is a witness, not a contract.

Refined since this was written: the variability is **allocation, not
semantics**.  `agc.scs.get_layout()` returns a single driver-supplied packed
`i64`, and the parts that survive the AGX3b/AGX3c boundary -- the ray record's
bytes 0..24, the status byte's bit-6 completion flag, the epilogue's `f70=4
f68=15` 512 stride -- are hardware-mandated, while placement is ours to choose
because we supply that constant.  See `README-g17p-rt-hardware-abi.md`.

## The vocabulary: SCS is sliced per virtual ray id

Recovered from the string cluster at VA `0x4739c5f`-`0x4739db6` in the arm64
slice of `libapplegpu-nt.dylib`, with `user_data_start` / `user_data_end` /
`driver_data_start` at `0x47394b6`:

```
scs.scs_layout.ptr      global_stride           shader_stride
vrid_ext                ray_core                token_buffer
scs.num_vrids           scs.max_level_depth     global_vrid_size
vrid_level_1            vrid_level_2            segment_start
ray_user_data_as_i8     instance_leaf_start     vrid_instance_leaf_offset
scs.instance_leaf_ptrs.a
```

**`vrid` = virtual ray id.**  The SCS is sliced per-VRID with a global stride,
and `scs.num_vrids` grows with instance levels.  That is the shape of the bank
map even without running the printer, and it is why a single flat "offset of
field X" table was always going to be the wrong abstraction.

## Apple's layout printer: located, and why the env vars never worked

The text printer is gated at `libapplegpu-nt.dylib` `0xaf174`:

```
adrp x8, 0x4b85000
ldrb w8, [x8, #0x648]     ; gate byte, VA 0x4b85648
cmp  w8, #0x1
b.ne 0xaf554              ; skip the whole text printer
```

Three things fall out, all measured:

* Both gate bytes are function-local statics the optimiser folded to
  constants.  `0x4b85648` (text) is initialised to literal **1** at
  `0xafc4c`-`0xafc50` under `__cxa_guard_acquire(0x4b85650)`; `0x4b85658`
  (JSON) is initialised to literal **0** at `0xafc74`.  **The JSON variant is
  dead code in this build** -- only the text form can ever print, and it is
  unconditionally armed once reached.
* The real gate is the predicate at `0x167dc`:
  ```c
  if (!statusStreamEnabled(this->stream_at_0x6a0))  /* call 0x2c30 */
      return 0;
  if (!this->isDriverShader())                      /* vcall [vtable+0x38] */
      return 1;                                     /* user shader: log it */
  const char *e = getenv("AGC_STATUS_LOG_DRIVER_SHADERS");  /* VA 0x47244fc */
  if (!e || *e == '0') return 0;
  return atoi(e) != 0;
  ```
  For an ordinary shader the printer needs **only** a live status-log stream.
* **`AGC_ENABLE_STATUS_FILE` has zero code references in the G17 translator.**
  It lives in `libapplegpuG9G12-nt.dylib`, a different plugin.  Only
  `AGC_ENABLE_STATUS_FILE_STDERR` (`0x31ac`), `AGC_STATUS_LOG_DIRECTORY`
  (`0x40a4`), `AGC_ENABLE_PER_OBJECT_STATUS_FILE`,
  `AGC_ENABLE_STATUS_FILE_TIMING` and `AGC_STATUS_LOG_DRIVER_SHADERS` are
  referenced here.  That is why sweeping the env matrix produced nothing: the
  variable everyone reaches for is read by another library entirely.

Two ways in, for whoever picks this up:

1. `libapplegpu-nt.dylib` reads
   `/System/Library/PrivateFrameworks/AGXCompilerCore.framework/EnvVars.txt`
   (string at VA `0x473abc8`).  That file does not exist on this host and the
   path is SIP-protected.  If it can be supplied, it is the sanctioned
   injection point.
2. Otherwise the work is exactly "make `this->[0x6a0]` a live stream in the
   offline path".  Gate `0x167dc`, stream check `0x2c30`.

`-mtranslator` is a dead end: its accepted values are only `load`,
`serialize`, `deserialize`.

## SEG layout

* SEG0 is 64 bytes and holds a **second ray slot** at bytes 36..59, selected by
  packed-word bit 13.
* **SEG 6 word 0** (bank 3 offset 0 in the `<instancing, triangle_data>`
  layout; the `stscs f70=3 f68=3 f71=0` / `ldscs f70=3 f68=1 f71=0` pair), from
  `016_agx.intersect.ray.ll` lines 598-625: preserve mask `0x1F007C00`;
  bits 1-6 = `(cfg_byte << 1)`; bits 17-18 = the cull-mode enum; bit 7 = flags
  bit 5; bit 8 = flags bit 3; bit 9 = flags bit 7; bit 29 = flags bit 1;
  bit 30 = `%56` bit 0; bit 31 = `%56` bit 2.

## Open: the `f71` scale in the `f28:1 f80:1` epilogue mode

Measured: `f71` is the **identical** 16-bit field (instruction bits 79..94) in
both modes -- `fieldbits ldscs` gives the same bit map for an ordinary and an
epilogue field vector.  What moves is `f68`, which relocates into the
destination-register slot (bits 24..27), consistent with "no GPR destination".

Measured: the 512 stride is invariant across six independent levers -- tag set,
`intersector` vs `intersection_query`, `max_levels` 2/4/5/16, and user payload
4 B / 20 B / 256 B.  Only the *count* moves (5 -> 32 chunks, capping at 32).
One witnessed non-multiple, `bank 2, f71=2816` (= 5x512 + 256), proves the
field is not restricted to 512 multiples.

**Not settled, and deliberately not guessed.**  Every epilogue access in the
corpus is a multiple of 256 f71-units, so two readings are observationally
equivalent:
* `f71` is a byte offset and 512 is a real 512-byte VRID stride; or
* `f71` is byte x 32 and the chunks are contiguous 16-byte units.

The separating experiment: **produce two epilogue-mode accesses in the same
bank whose `f71` differ by less than 256.**  That needs a payload the compiler
cannot round to 16 bytes -- try a struct with a trailing `uchar` or a
`packed_float3`, or a `visible_function_table` payload -- and look for an
`f68 < 15` chunk adjacent to an `f68 = 15` one.  A gap of 16 proves byte
offsets; a gap of 512 proves the x32 scale.  A 20-byte payload was tried and
failed: the compiler rounds the copy to whole 16-byte chunks regardless.

## Open: packed-word bits 8-17

An 18-kernel sweep varying one `intersection_params` setting at a time did
**not** isolate the packed word W, for two reasons worth recording:

* The parameters that move anything move a **uniform-register flags word**, not
  W.  Baseline for `<instancing, triangle_data>` is `0x4900` (bits 8, 11, 14),
  materialised in the constant program at `0x5e`, with a second constant
  `0x100` at `0xbe`.  `geometry_cull_mode::triangle` adds bit 30 to both (a
  clean two-line diff).  `opacity_cull_mode::opaque` adds bit 17;
  `::non_opaque` adds bit 18 (both also restructure ~180 lines of culling
  code).
* **W is not compile-time constant even with all-constant inputs**: bits 26-29
  (CoreId) are runtime, so W is built by an `iadd`, not an `ldimm`.

Better experiment: the still-ambiguous W bits come from `agx.intersect.ray`'s
`i1` args `%26`/`%28` and the `%55 == 0` test.  Bind those from the MSL side by
compiling the **`intersection_query::reset`** path (not `intersector`) with
`intersection_params` whose `should_assume_identity_transforms` /
`should_accept_any_intersection` are compile-time constants, then diff the
`bit_or` immediates feeding lane 3 of `stscs f70=0 f68=15 f71=16` -- that lane
is W's low half.

## Open, and needs hardware: W bits 0-1, 6-7, 30-31

No ria3 code writes them, and `init_core`'s clear mask `0x3FFC7C00` stops at
bit 29, so 30-31 survive initialisation.  There is no host-side evidence for
their meaning; none is inferred here.

## Traps -- negative results that cost time to rediscover

* **Several `intersector::intersect` parameters produce byte-identical G17P
  code**: `set_triangle_cull_mode`, `set_triangle_front_facing_winding`,
  `accept_any_intersection(true)`, `assume_identity_transforms(true)` and
  `assume_geometry_type`.  A lowering that assumes these matter *in that path*
  is wrong.
* The instance mask *does* reach the instruction stream: masks
  `0xff/0x01/0x55/0xaa` give a single-instruction diff, `iadd ... f15:
  255/1/85/170`.  This confirms the 8-bit runtime mask path without relying on
  the metallib.
* See also `memory/g17p-rt-two-tree-claims-are-wrong.md`: the tree's canonical
  `intersect_ray` bytes are corpus vector 561, a real Apple instruction from an
  unrelated kernel, and appear in **none** of the 10 witnessed G17P RT sites.
  It is not a lowering template.
