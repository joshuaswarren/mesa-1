# G17P native RT0 EOT witness

This replaces the earlier compute-wrapper EOT witness for the bounded
single-layer, one-sample RGBA8/BGRA8 RT0 path. The installed Apple GPU compiler's
real builtin EOT factory produces a fragment object with 68 bytes of text.

The adjacent `g17p-native-rt0-eot.btnf.hex` is the complete 123-byte input,
not an edited shader or relabelled compute metadata. Decode it with `xxd -r -p`
and invoke the matching Apple Metal toolchain:

```sh
xxd -r -p g17p-native-rt0-eot.btnf.hex > eot_rt0.btnf
xcrun metal-nt -N eot_rt0.btnf -arch applegpu_g17p \
  -platform_version macos 26.0 26.0 -o eot_rt0.nat
```

Input SHA-256:
`6806d787abddcd75594ab0be2acd6da29c2438a91d52a333071a277df918801a`.
Original output SHA-256:
`7a75c15cd395fe6b806b1459955c156494a4d9bc1d60d09f4c2527302855fc3f`.
Toolchain versions can change object packaging; the pinned executable text is:

```text
1c9c100624a814060c9d102624a91026
9f1054000004140050060b00
9f2054020000100050060b00
9705540400080000d0401000
5710540000000080a8750000
0e000000
```

Text SHA-256:
`e641858631bd55fb938f839920beb6b2921a8bbada65bd72b21102daed9651de`.
The original outer `__TEXT,__builtin` contains the fragment Mach-O at offset
0x4d0. Its metadata has common field0=3 GPR dwords and fields38/42 D=16.
The implementation conservatively allocates eight GPR halves (native needs
six) and eight staging halves; those quantities are independent.

## Input and binding contract

BTNF vendor root field2=5 selects the genuine EOT factory. Root field3 has
EOT configuration union discriminator4, one attachment and enabled mask1.
The attachment fields produce key `0x10020300`: partition0, internal format3,
texture type/layout code2, four components, no extra flags.

The native object's ESL records are `(kind,count32,dest32,source32)`:

- `(3,2,0,0)`: qword binding0x32 supplies the raw, 32-byte-aligned PBE-table
  GPU VA to uniform halves0..3.
- `(0,1,2,0)`: the sole driver argument, kind5/index0/destination0, supplies
  the imageblock byte offset as a 32-bit value to uniform halves4..5.

Its `__GPU_LD_MD` RT store mask (field11) is1. ESL kind3 and driver-argument
kind3 are different namespaces; no inline PBE32 uniform payload is requested.

Matching UMD `AGXMetalG17P.arm64e` SHA-256
`b641761d0ecb74babe7fbce66cb895bc180fe008f4d4c533845138c4eb64217b`:
binding0x32 maps to driver qword0x36 at +0x1b0 (0x392944–0x392970).
Kind5 maps to driver dword0x72 at +0x1c8 (0x392ed8/0x3931b8).
`configurePBE` at 0x331788–0x331798 writes
`((draw_state >> 12) & 255) % sample_stride` to framebuffer+0x388+4*RT.
For the admitted unspilled single RT0, the offset is0 and sample stride8.
Mesa takes this value from `agx_tilebuffer_offset_B`, not BO/image dimensions.

The shader's four SR reads obtain group position and dispatched group size.
Two products generate staging coordinates for the block origin. PACK reads
the byte-offset uniform and IMGW reads the table pointer with RT index0.
The complete 68-byte entry is uploaded directly into the existing CODE
container at block+0x80, padded to128 bytes; LDSHDR f1 remains1.

## The multisample sibling

The attachment table's texture-type byte is at file offset 0x76 and takes the
`Texture dimension` enum. Setting it to 4 (2D multisample) instead of 2 (2D)
makes the same factory emit the program that stores the tile **unresolved**:
same length, same registers, same metadata, three bytes different. The full
sweep and the round-tripped encoder fields are in
`isa/README-metal-witness.md`; the emitted programs are the
`g17p_rt0_eot_ms` table in `agx3_pack_binary.c` and are asserted byte for byte
by `test_g17p_multisample_eot_block_store`.

## Validation scope

### RGBA32 signed store dimension correction

The former RGBA32_SINT 84-byte fixture was the array program. It reads layer
SR0xc3, packs three coordinates, and sets IMGWBLK's array field even when Mesa
has a non-array render target. This is not a signedness-dependent instruction
sequence.

A fresh native-factory cross-product changes attachment format at byte0x75 to
10, 11 or 17, signed integer, unsigned integer or float, while independently
setting texture type at byte0x76 to 2 or 3. Components at byte0x79 stay4.
All three formats produce identical text for each texture type:

- Type2, 2D: 68 bytes, SHA256
  `60beeb253d8f81af993e93617ed3fca4559e0f614c8465f4cd2bb42c0f1f5fb6`.
- Type3, 2D array: 84 bytes, SHA256
  `a2b424a85d087598fafb384d4b50e04311363bc068eefb020fac0a84df581286`.

The type2 text is the existing selector0x35 program. Type3 reproduces the old
signed fixture byte for byte. The regression requires the same type2 bytes for
signed and unsigned four-word I32 stores. Texture types 4 (multisample) and 8
(multisample array) are likewise identical across the three formats, and the
multisample regression requires the unresolved I32 bytes for the signed key.
This is native compiler evidence, not GPU pixel validation.

On J700 A18 Pro B1, native DM1, the old per-thread-coordinate program stored
only the first 32x32 tile of a 512x512 target. The matched native shader and
CProg pair passed a contained orange triangle over red clear: 73,536 strict
interior, 188,224 strict exterior and 384 edge-centre pixels, zero mismatches,
with the full image byte-identical three seconds after the fence. No shader
diagnostic override was enabled. This proves this bounded spatial path, not
MRT, MSAA, arbitrary formats, repeated submissions or Vulkan conformance.

### Selector follows the imageblock lane class

Every single-target 2D program is the 68-byte text above with one IMGWBLK
selector byte. Sweeping byte 0x75 (internal format) over 0..22 with one to
four components at texture type 2 gives a selector that depends on the format
only. Builtin background clears of the same formats (root field 2 = 4, clear
mask 1) store exactly one lane class per selector:

| Formats | Clear store | Selector |
| --- | --- | --- |
| 2, 3 | W8 through `fltnorm` | 0x75 |
| 4, 5 | W8 integer | 0x55 |
| 6, 7 | W16 through `fltnorm` | 0x95 |
| 8, 9, 18 | W16 integer | 0x15 |
| 12..16, 22 | W16 float | 0x15 |
| 10, 11, 17 | raw W32 | 0x35 |
| 19..21 | W32 through `pack` | 0x75 |

Apple's render pipelines (`metal-nt -arch applegpu_g17p` on an mtlp-json with
one colour attachment per MTLPixelFormat) store each format in one of these
classes, the same class the tilebuffer lowering uses. That fixes the selectors
the earlier internal-format guesses got wrong: B5G6R5Unorm stores W16 float
lanes (0x15, not 0x35), RG32Uint and RG32Sint raw W32 lanes (0x35, not 0x75
and 0x15), and RGBA16Unorm `fltnorm` W16 lanes (0x95, not 0x15). The former
84-byte RGBA16_SINT program is byte-identical to the texture-type-3 (2D array)
builtin of every 16-bit internal format; its 2D program is the 0x15 text.
