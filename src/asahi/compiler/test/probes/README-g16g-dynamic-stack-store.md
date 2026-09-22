# G16G graphicsfuzz refusals: dynamic stack stores, deep loops, high registers

Seventeen `dEQP-VK.graphicsfuzz.*` cases failed pipeline creation on G16G
because the AGX3 packer refused one instruction shape each
(`no grounded AGX3 encoding for ...`, host shim compile with
`build/claude-fix/shimrun.sh`):

| shape | cases | refusal |
|---|---|---|
| `stack_store value, r_index` | 8 (`spv-*`, `cov-*large-array*`, `stable-triangle-array-nested-loop`) | only immediate offsets were translated (static COPY) |
| `break_if_icmp` at loop depth 7 | 6 (`cov-*nested-loops*`, `cov-optimize-phis-for`, `cov-dfdx-dfdy-after-nested-loops`) | flag slot `loop_depth - 1 >= 6` |
| `stack_load #256.abs` | 6 (`spv-*maze*`, `spv-access-chains`, `spv-copy-object`, `spv-dead-break-and-unroll`) | the abs bit that marks a 16-bit immediate was rejected |
| `imad #3, r, #0` | 1 (`cov-function-two-loops-limit-...`) | constant multiplier in the first source |
| `convert u16->f32` above half 127 | 1 (`cov-nested-loop-large-array-index-...`) | register limit 128 in the half int2flt form |

## Witnesses (AIR-NT 32023.917, applegpu_g16g)

Compiled with `build/claude-fix/witness/mcompile.py` and decoded with
`tools/shader-fields.py ... --encoding AGX3b --offset 64`.

* `g16g-dynamic-stack-store.metal` (`cs64`, `cs256`, `fs64`): a dynamically
  indexed private-array store is `sttps` frame 1 with a register index and no
  scoreboard slot. Scalar `uint` stores use W16 mask 3 with an element index
  (shift 2, byte offset 4); the following `ldtps` reads the same region.
  The `sttps` carries the wait of the preceding static `copy` writes
  (f45/f46) instead of a separate wait.
* `g16g-dynamic-stack-store-static-reads.metal` (`fsstatic`, `fsvec4`):
  static `rld` reads of the array follow the `sttps` immediately without any
  wait, so the TPS pipeline orders them; a `float4` store is W32 mask 15
  with a byte index (`ef105400220a170001406f00`), its reload consumes the
  location (`6f005400200a570104400200`). Both are unit-test vectors in
  `test-agx3-local-tps-core.c`.
* `g16g-nested-loops.metal` (`loops7`, `loops8`): every loop level, seven and
  eight deep, tests through flag slot 1 (`icmp` f31 = 1, `cndlt` f1 = 1);
  the compare always immediately precedes the `cndlt` that consumes it, so
  deeper nests wrap the six slots instead of being refused.
* `g16g-int2flt-high-registers.metal` (`hiregs`, `hiregs16`): 96 live
  conversions put `int2flt` destinations up to half 160 and sources up to
  half 189 in the same form (`a707548402b0a420`, `a70754460244a520`, the
  source's high bit in byte 6); the composer accepts halves below 192.

## Hardware checks

`tools/tps-oracle.py dynstore` (32 lanes, three lane-dependent dynamic
`uvec4` stores including a read-modify-write, then static and dynamic reads;
the host shim confirms three register-indexed `stack_store xyzw` and the
static `rld` reads) run with `vk-buffer-oracle` under `gpu-experiment.py`:
`PASS: buffer oracle bad=0/4352` (experiment `20260915T2*-claude-tps-dynstore`).
All 17 graphicsfuzz cases pass; the 1,001-case regression list is unchanged
(815 Pass / 186 NotSupported).
