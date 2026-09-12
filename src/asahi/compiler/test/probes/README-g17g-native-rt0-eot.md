# G17G native RT0 EOT witness

The existing `g17p-native-rt0-eot.btnf.hex` is a hardware-independent builtin
factory request. Generate an independent M5 witness with the G17G target:

```sh
xxd -r -p g17p-native-rt0-eot.btnf.hex > eot_rt0.btnf
xcrun metal-nt -N eot_rt0.btnf -arch applegpu_g17g \
  -platform_version macos 26.0 26.0 -o eot_rt0.nat
```

Metal toolchain 27A5194o produced a 2632-byte native object with SHA-256
`bd9e36505453ed7239cc4b8fc5963bd9551a07dc6f88a7804da202e2be60be2c`.
Its nested fragment `__text` is 78 bytes at outer offset 0x770, SHA-256
`993b96ceeb6ca1622d34811dbe603fe3502768892775c7cb70805637c72db6df`:

```text
1c9c100624a814060c9d1026248101402801a00a2706040a2100a102680120000102
3708040a2100a1022801200001022700202a2100ab0a00004a301800250415002400
a02200e000550e000000
```

The builtin request remains kind5/key0x10020300, enabled RT mask1. This
proves only an unspilled, single-sample, single-layer RGBA8/BGRA8 RT0 store.
Native common metadata field0=3 GPR dwords and fields38/42 D=16 require
six GPR halves and eight staging halves. Mesa conservatively allocates eight
GPR halves. Native loader RT-store mask is1.

The native ESL matches the independently decoded G17P binding contract:
qword binding0x32 loads the PBE-table pointer into uniform halves0..3;
driver argument kind5/index0 loads the tile byte offset into halves4..5.
The admitted tile byte offset is0. These bindings still require a G17G
resource-loading Ioto program; compiling the shader alone does not establish
valid command-stream construction or a successful Vulkan render.

The compiler recognizes the complete explicit-coordinate post-RA graph,
requires an explicit G17G key, and rejects changed attachment index, offset,
dimension, format, and conflicting G17P selectors. Tests compare complete
native text and register metadata and exercise these rejection boundaries.
