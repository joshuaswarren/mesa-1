# G17P wide MRT compiler witnesses

The sources and pipeline scripts in `g17p-wide-mrt/` generate native fragment
programs without using a GPU device. They were compiled with Apple Metal
Toolchain v27.1.5194.15, targeting `applegpu_g17p` on 2026-09-13:

```sh
xcrun metal -c fragment-4.metal -o fragment-4.air
xcrun metallib fragment-4.air -o fragment-4.metallib
xcrun metal-nt -arch applegpu_g17p -platform_version macos 27.0 27.0 \
  -N fragment-4.mtlp-json fragment-4.metallib -o fragment-4.nat
```

Repeat with `fragment-8` and `fragment-8-sparse`. The sparse shader omits
output 1 while retaining all eight pipeline attachments. This is compiler
evidence; no hardware or CTS pass is claimed by these captures.

| Probe | Fragment text bytes | Fragment text SHA256 |
| --- | ---: | --- |
| four targets | 238 | ea8c19c0750688036e21f4e0b31bf1fe1f3af02e9ff12953f7b4b0f84312c25b |
| eight targets | 366 | e77ea0832fd4dafc66f982269575fc52a4f70662f702f19a14d7416e8d269d28 |
| eight slots, output 1 omitted | 322 | 3b061c40f67ab8680ebeef17c76b59a555c9207e8bc934fc8a9defecc580c0f7 |

The complete native object SHA256 values, respectively, are
`9abe6a59677ff68efd3bc4e3ad858026a8cf42515636c6d55f10a30fc9371ffa`,
`8b8a3afda2dd72da6498687afbd52fb74ea38cda84f18da90c3c5519be05c9bc`,
and `21b157ca5ea806220f746217e8b769fd45d6ccadf13be45a2d7bb9a88f3d293d`.

## Shared synchronization classes

The three-class limit is not a three-target limit. In the four-output
program's transaction at text+0x90, native code acquires class12 before
storing target3, retains it across the target2 class192 and target1 class48
transactions, then stores target0 and releases class12 with the final bit.
The eight-output program behaves identically, except that targets7..3 are
all stored under the initial class12 acquisition. Those stores must not
release class12 individually, nor reacquire it while it remains held.

`AGX3PackBinary.G17PFourRGBA8OutputsShareTheFinalPixelClass` checks the complete
native four-output transaction, byte for byte. The eight-output test checks
the independently observed synchronization sequence and the complete store
count. It does not claim identical register-allocation hints for every store.
The `asahi-g17p-mrt-epilog` test additionally traverses real NIR output
lowering and register allocation for four/eight distinct clear inputs,
checking that all descending stores survive.

Current admission covers straight-line opaque RGBA8 outputs at contiguous
four-byte tile offsets. Wide blending, other formats and sparse physical
offsets remain rejected. Existing two/three-target behavior is unchanged.

## Sparse outputs need a mask, not just a count

Fragment `__GPU_METADATA_2` root field1 is `0x203` for three, four and eight
active outputs. For the eight-slot shader omitting output1 it is `0x233`:
the loader releases the unused class48. Its fragment stores offset8 under
class192 and never acquires class48. Compacting sparse physical offsets into
ordinal target numbers would lose this distinction. The compiler and loader
now retain logical attachment identity and a consumed partition mask.
`g17p-sparse-mrt.h` records six independent native oracles, including layouts
with attachment0 absent. Its inputs live under `g17p-sparse-mrt/`. The packer
test compares native synchronization sequences and physical offsets; real
NIR tests cover epilog, background clear and reload compilation for all six
masks. Ioto tests compare the native loader release metadata, including
0x23f for target2 alone.

Gallium now packs active EOT attachments while retaining logical background
clear and texture slots. CPU layout and loader tests cover all 255 nonempty
attachment masks. Real NIR epilog, clear and reload tests cover the six sparse
oracles. These are different coverage scopes: the integrated runtime change
has not yet been tested on hardware.

## EOT physical partitions are separate from fragment classes

The native UMD framebuffer constructor computes an EOT attachment's partition
as `tile_offset / sample_stride`, and its PBE setup supplies
`tile_offset % sample_stride` as the offset argument. Every compact RGBA8
offset admitted here is below the sample stride, so all EOT attachments use
physical partition zero. The earlier EOT inputs `(0, 1, 2, 0, ...)` incorrectly
reused fragment synchronization classes and must not be used as a recipe.

The current `agx_g17p_wide_mrt_eot.h` contains the partition-zero native texts
for active counts 1 through 8. Their sizes are 68, 120, 164, 208, 252, 296, 340
and 384 bytes, with three native GPR dwords and `16 * count` staging bytes.
The driver conservatively allocates eight GPR halves. Each text's SHA256 is
recorded next to its array. No application shader name or hash selects it;
selection uses the internal layout and operation contract.

The complete offline recipe now lives beside this document in
`g17p-mrt-eot.json` and `capture-g17p-mrt-eot.py`, using the existing committed
BTNF seed. From the repository root, with MetalToolchain available:

```sh
python3 src/asahi/compiler/test/probes/capture-g17p-mrt-eot.py capture /tmp/g17p-eot-proof
python3 src/asahi/compiler/test/probes/capture-g17p-mrt-eot.py check /tmp/g17p-eot-proof
python3 src/asahi/compiler/test/probes/capture-g17p-mrt-eot.py header /tmp/g17p-eot-proof
```

`capture` requires a new output directory and retains the exact compiler
commands, version, serialized inputs, native objects, sections and receipts.
It opens no GPU device. `check` validates all nine input/text digests, GPR and
staging metadata, ESL pointer/offset ranges, compact driver arguments and
store masks. It then compares the generated RGBA8 header and both mixed-format
source/test arrays with native output. `header` emits the RGBA8 header to
stdout. The tools do not depend on a particular host workspace or campaign
artifact path. A compiler/version mismatch fails validation; updating golden
data requires reviewing the native input and encoding contract, not copying
current Mesa output into the expectation.

The ninth variant is RGBA16F/R8 at offsets0/8 within sample stride16. Its
native input uses formats8/3, components4/1 and partitions0/0. The former
partitions0/1 input reproduces the old 238-byte source exactly. Changing only
that incorrect partition byte produces the correct 120-byte witness, SHA256
`b3191cbbd673db5fb00d3e744f60ed59befad9bc19a2bf30aa9420effa63df12`.
The production builtin, its independent compiler oracle and the loader's
sample-stride checks now use that contract. This corrects an input error;
it is not a reason to change fragment synchronization classes.

All nine variants were recaptured with the repository tool. The corrected
mixed-format compiler/loader tests pass, as do the wide/sparse CPU contracts.
Hardware qualification remains pending. General compiler lowering and the
unsupported format/blending/operation combinations remain separate work;
these internal builtin templates do not establish full framebuffer support.
