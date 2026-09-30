/* AIR-NT 32023.921, G17P CProg RLDs. See agx3-cprog-inputs.md.
 * Logical halves below equal native TPS half minus 8 for these prefix inputs. */
static const struct {
   struct agx3_cprog_import desc;
   size_t length;
   uint8_t bytes[8];
} native_cprog_imports[] = {
   {{.uniform_half=0, .dest_half=0, .width=AGX3_MEM_W32, .slot=2,
      .slot_mirror=2, .wait_mask=32}, 8, {0x0c,0x88,0x19,0x21,0,0,0,0x80}},
   {{.uniform_half=2, .dest_half=2, .width=AGX3_MEM_W32, .slot=2,
      .slot_mirror=2}, 4, {0x1c,0x8a,0x18,0x21}},
   {{.uniform_half=2, .dest_half=2, .width=AGX3_MEM_W32, .slot=1,
      .wait_mask=32}, 8, {0x1c,0x8a,0x09,0x07,0,0,0,0x80}},
   {{.uniform_half=0, .dest_half=0, .width=AGX3_MEM_W16, .slot=1,
      .wait_mask=32}, 8, {0x04,0x88,0x09,0x06,0,0,0,0x80}},
   {{.uniform_half=0, .dest_half=0, .width=AGX3_MEM_W32, .slot=1,
      .wait_mask=32}, 8, {0x0c,0x88,0x09,0x07,0,0,0,0x80}},
   {{.uniform_half=0, .dest_half=4, .width=AGX3_MEM_W32, .slot=2,
      .slot_mirror=2, .wait_mask=32}, 8, {0x2c,0x88,0x19,0x21,0,0,0,0x80}},
   {{.uniform_half=2, .dest_half=6, .width=AGX3_MEM_W32, .slot=2,
      .slot_mirror=2}, 4, {0x3c,0x8a,0x18,0x21}},
   {{.uniform_half=4, .dest_half=0, .width=AGX3_MEM_W32, .slot=1},
      4, {0x0c,0x8c,0x08,0x07}},
   {{.uniform_half=0, .dest_half=0, .width=AGX3_MEM_W32, .slot=2,
      .wait_mask=32, .structural_flag=true}, 8, {0x0c,0x88,0x09,0x27,0x60,0,0,0x80}},
   {{.uniform_half=2, .dest_half=2, .width=AGX3_MEM_W32, .slot=2},
      4, {0x1c,0x8a,0x08,0x27}},
   {{.uniform_half=0, .dest_half=8, .width=AGX3_MEM_W32, .slot=2,
      .wait_mask=32, .structural_flag=true}, 8, {0x4c,0x88,0x09,0x27,0x60,0,0,0x80}},
   {{.uniform_half=2, .dest_half=10, .width=AGX3_MEM_W32, .slot=2},
      4, {0x5c,0x8a,0x08,0x27}},
};
