/* SPDX-License-Identifier: MIT */
#include "agx3_rt_m5_core.h"
#include "agx3_binder.h"
#include "agx3_pack.h"
#include "agx3_scs_core.h"
#include <string.h>

#define KV(f, v) {(f), (v)}

bool
agx3_rt_m5_encode_enter_query(enum agx_isa_encoding encoding,
                              uint16_t destination, uint16_t allocation_index,
                              uint8_t out[16], uint8_t *length)
{
   if (!out || !length)
      return false;
   memset(out, 0, 16);
   *length = 0;
   if (encoding != AGX_ISA_ENCODING_AGX3C || destination > 252 ||
       (destination & 1) || allocation_index < 4 ||
       allocation_index >= 4 + AGX3_RT_M5_MAX_LIVE_QUERIES)
      return false;
   const struct agx3_field_kv fields[] = {
      KV(2, destination), KV(9, 2),   KV(10, 2),
      KV(14, 3),          KV(15, 90), KV(24, allocation_index),
   };
   return agx3_bind_compose(encoding, "incStk", 0, fields,
                            sizeof(fields) / sizeof(fields[0]), NULL, 0, out,
                            length);
}

bool
agx3_rt_m5_encode_intersect(enum agx_isa_encoding encoding,
                           const struct agx3_rt_m5_intersect_desc *d,
                           uint8_t out[16], uint8_t *length)
{
   if (!out || !length)
      return false;
   memset(out, 0, 16);
   *length = 0;
   if (encoding != AGX_ISA_ENCODING_AGX3C || !d ||
       d->address_register > 252 || (d->address_register & 1) ||
       d->bvh_base > (d->bvh_uniform ? 508 : 252) ||
       (d->bvh_base & 1) || (d->bvh_uniform && d->bvh_kill))
      return false;

   const struct agx3_field_kv base[] = {
      KV(9, 5), KV(10, 2), KV(12, 8), KV(13, 1), KV(14, 2),
      KV(24, 2), KV(25, 3), KV(28, 1), KV(29, 92), KV(30, 1),
      KV(49, 32), KV(55, 3), KV(56, 3), KV(70, 2), KV(85, 1), KV(99, 1),
   };
   const struct agx3_field_kv operands[] = {
      KV(17, d->address_register),
      KV(d->bvh_uniform ? 63 : 62, d->bvh_base),
      KV(69, d->bvh_uniform ? 3 : 2), KV(73, d->bvh_kill),
   };
   return agx3_bind_compose(encoding, "intersect_ray", 0, base,
                           sizeof(base) / sizeof(base[0]), operands,
                           sizeof(operands) / sizeof(operands[0]), out, length);
}

bool
agx3_rt_m5_encode_continue(enum agx_isa_encoding encoding,
                          uint8_t out[16], uint8_t *length)
{
   if (!out || !length)
      return false;
   memset(out, 0, 16);
   *length = 0;
   if (encoding != AGX_ISA_ENCODING_AGX3C)
      return false;
   const struct agx3_field_kv fields[] = {
      KV(5, 8), KV(9, 5), KV(10, 2), KV(12, 8), KV(13, 1),
      KV(14, 2), KV(15, 34), KV(16, 1),
   };
   return agx3_bind_compose(encoding, "continue_ray", 0, fields,
                           sizeof(fields) / sizeof(fields[0]), NULL, 0,
                           out, length);
}

bool
agx3_rt_m5_ift_loader_generate(
   const struct agx3_rt_m5_ift_loader_desc *d,
   uint8_t out[AGX3_RT_M5_IFT_LOADER_MAX_SIZE], size_t *length)
{
   if (!d || !out || !length || !d->constants_va ||
       (d->constants_va & 3) || (d->constants_va >> 48) ||
       !d->shader_va || (d->shader_va & 63) || (d->shader_va >> 42) ||
       d->fetch_span < 1 || d->fetch_span > 3 ||
       (d->context48 ? (d->constant_dwords != 20 && d->constant_dwords != 24)
                     : (d->constant_dwords < 2 || d->constant_dwords > 34 ||
                        d->constant_dwords % 4 != 2)))
      return false;

   uint8_t staged[AGX3_RT_M5_IFT_LOADER_MAX_SIZE] = {0};
   size_t used = 0;
   const uint16_t base = d->context48 ? 28 : 24;
   const uint16_t dest[] = {8, base, base + 2};
   const uint32_t value[] = {0x6000, d->constants_va, d->constants_va >> 32};
   for (unsigned i = 0; i < 3; ++i) {
      struct agx3_encoded_ldimm imm;
      if (!agx3_pack_ldimm_u32(AGX_ISA_ENCODING_AGX3C, dest[i], value[i], &imm))
         return false;
      memcpy(staged + used, imm.bytes, imm.length);
      used += imm.length;
   }

   /* Context fields at +80 and, in the second layout, +48 are runtime
    * inputs. Keep their numeric names until the wider SCS ABI is decoded. */
   for (unsigned i = 0; i <= d->context48; ++i) {
      struct agx3_scs_desc scs = {
         .data = i ? 20 : base - 4,
         .width = AGX3_MEM_W32,
         .byte_offset = i ? 48 : 80,
         .address_kind = AGX3_SCS_ADDRESS_REGISTER_BASE,
         .base_register = 8, .component_mask = 3,
         .has_address_selector = true, .address_selector = 2, .slot = 2,
      };
      uint8_t size;
      if (!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &scs,
                               staged + used, &size))
         return false;
      used += size;
   }

   /* The native loader loads the high, possibly partial, constant chunk
    * first. The final low chunk overwrites the address pair itself. f75=2
    * is retained as an explicit Ioto load field; it is not reinterpreted as
    * the generic shader-memory instruction's address scale. */
   const unsigned chunks = (d->constant_dwords + 3) / 4;
   for (unsigned c = chunks; c-- > 0;) {
      unsigned words = d->constant_dwords - 4 * c;
      if (words > 4) words = 4;
      const struct agx3_field_kv fields[] = {
         KV(2, base + 8 * c), KV(9, 2), KV(10, 2), KV(12, 2),
         KV(23, 1), KV(24, 1), KV(31, base), KV(38, 2), KV(39, 3),
         KV(45, 15), KV(53, 1), KV(54, 1), KV(59, 104),
         KV(73, (1u << words) - 1), KV(75, 2), KV(76, 16 * c),
      };
      uint8_t size;
      if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, "lddev", 0,
                             fields, sizeof(fields) / sizeof(fields[0]),
                             NULL, 0, staged + used, &size))
         return false;
      used += size;
   }

   const struct agx3_field_kv local[] = {KV(0, 3), KV(1, 165)};
   const struct agx3_field_kv shader[] = {
      KV(0, d->shader_va), KV(1, d->fetch_span), KV(5, 3), KV(6, 107),
      KV(23, 1),
   };
   uint8_t size;
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, "specLM", 0,
                          local, sizeof(local) / sizeof(local[0]),
                          NULL, 0, staged + used, &size))
      return false;
   used += size;
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, "ldshdr", 0,
                          shader, sizeof(shader) / sizeof(shader[0]),
                          NULL, 0, staged + used, &size))
      return false;
   used += size;
   memcpy(out, staged, used);
   *length = used;
   return true;
}

bool
agx3_rt_m5_ift_entry_generate(
   const struct agx3_rt_m5_ift_entry_desc *d,
   uint8_t out[AGX3_RT_M5_IFT_ENTRY_MAX_SIZE], size_t *length)
{
   if (!d || !out || !length || !d->uniforms || d->uniform_count < 2 ||
       d->uniform_count > AGX3_RT_M5_IFT_MAX_UNIFORMS)
      return false;
   for (unsigned i = 0; i < d->uniform_count; ++i) {
      if ((d->uniforms[i].uniform_half & 1) ||
          (d->uniforms[i].register_half & 1) ||
          d->uniforms[i].uniform_half > 70 ||
          d->uniforms[i].register_half > 90)
         return false;
   }
   uint8_t staged[AGX3_RT_M5_IFT_ENTRY_MAX_SIZE] = {0};
   size_t used;
   if (!agx3_rt_m5_ift_loader_generate(&d->loader, staged, &used))
      return false;
   uint8_t n;
   const struct agx3_field_kv tpr[] = {
      KV(0, 3), KV(1, 166), KV(20, d->tpr20 ? 20 : 0),
      KV(21, d->tpr20 ? 20 : 0),
   };
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, "specTPR", 0, tpr,
                          sizeof(tpr) / sizeof(tpr[0]), NULL, 0,
                          staged + used, &n))
      return false;
   used += n;

   /* Keep the complete native call setup, including the SR14 read. LDSHDR
    * alone is not the end of this entry program. */
   for (unsigned i = 0; i < 3; ++i) {
      const unsigned dest = i == 1 ? 0 : 2;
      const unsigned value = i == 0 ? 0 : i == 1 ? 1024 : 256;
      const struct agx3_field_kv imm[] = {
         KV(2, dest), KV(9, 2), KV(10, 2), KV(14, value),
         KV(15, 3), KV(16, 105),
      };
      if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, "ldimm", 2,
                             imm, sizeof(imm) / sizeof(imm[0]), NULL, 0,
                             staged + used, &n))
         return false;
      used += n;
      if (i == 0) {
         const struct agx3_field_kv sr[] = {
            KV(2, 2), KV(9, 2), KV(10, 1), KV(12, 1), KV(21, 14),
            KV(23, 7), KV(24, 1), KV(28, 4), KV(29, 144), KV(38, 1),
         };
         if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, "rld", 1, sr,
                                sizeof(sr) / sizeof(sr[0]), NULL, 0,
                                staged + used, &n))
            return false;
         used += n;
      } else if (i == 1) {
         const struct agx3_field_kv add[] = {
            KV(9, 2), KV(10, 2), KV(23, 2), KV(24, 2), KV(31, 2),
            KV(38, 2), KV(39, 2), KV(44, 3), KV(45, 80), KV(46, 1),
            KV(55, 4),
         };
         if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, "iadd32", 0, add,
                                sizeof(add) / sizeof(add[0]), NULL, 0,
                                staged + used, &n))
            return false;
         used += n;
      }
   }
   const struct agx3_field_kv call[] = {
      KV(9, 2), KV(10, 3), KV(15, 2), KV(16, 27), KV(17, 1), KV(25, 1),
   };
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, "cndcall", 0, call,
                          sizeof(call) / sizeof(call[0]), NULL, 0,
                          staged + used, &n))
      return false;
   used += n;
   for (unsigned i = 0; i < d->uniform_count; ++i) {
      const struct agx3_field_kv copy[] = {
         KV(3, d->uniforms[i].uniform_half), KV(9, 3), KV(10, 2),
         KV(16, d->uniforms[i].register_half), KV(23, 2), KV(24, 2),
         KV(27, 1), KV(28, 1), KV(29, 35), KV(31, i == 0),
      };
      if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, "copy", 0, copy,
                             sizeof(copy) / sizeof(copy[0]), NULL, 0,
                             staged + used, &n))
         return false;
      used += n;
   }
   if (d->final_copy16) {
      const struct agx3_field_kv copy[] = {
         KV(9, 2), KV(10, 1), KV(23, 2), KV(24, 1), KV(28, 3),
         KV(29, 35), KV(42, 1),
      };
      if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, "copy", 1, copy,
                             sizeof(copy) / sizeof(copy[0]), NULL, 0,
                             staged + used, &n))
         return false;
      used += n;
   }
   struct agx3_encoded_end end;
   if (!agx3_pack_end(AGX_ISA_ENCODING_AGX3C, &end))
      return false;
   memcpy(staged + used, end.bytes, end.length);
   used += end.length;
   memcpy(out, staged, used);
   *length = used;
   return true;
}

/* The type-1 handler has no geometry fetch or triangle intersection. It
 * commits the hardware candidate in the per-lane packet, publishes continuation
 * state, and executes the shared traversal return. SR57/SR58/SR7 are kept
 * numbered: interpreting them more broadly is not needed to encode this ABI. */
bool
agx3_rt_m5_hit_commit_generate(
   const struct agx3_rt_m5_hit_commit_desc *d,
   uint8_t out[AGX3_RT_M5_HIT_COMMIT_MAX_SIZE], size_t *length)
{
   if (!d || !out || !length || d->payload_uniform > 508 ||
       d->context6_uniform > 510 || d->context8_uniform > 510 ||
       ((d->payload_uniform | d->context6_uniform | d->context8_uniform) & 1))
      return false;
   uint8_t staged[AGX3_RT_M5_HIT_COMMIT_MAX_SIZE];
   size_t used = 0;
#define OP(name, ep, ...) do { \
   const struct agx3_field_kv fields[] = {__VA_ARGS__}; \
   uint8_t n; \
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, name, ep, fields, \
                          sizeof(fields) / sizeof(fields[0]), NULL, 0, \
                          staged + used, &n)) return false; \
   used += n; \
} while (0)
#define RLD(dst, sr, slot, ep, waits) \
   OP("rld", ep, KV(2,dst), KV(9,2), KV(10,1), KV(11,waits), KV(12,slot), \
      KV(21,sr), KV(23,7), KV(24,1), KV(28,4), KV(29,144), KV(38,1))
   RLD(0,58,1,0,0);
   /* Extract the lane's 64-byte packet offset. */
   OP("bit_and",1,KV(2,6),KV(9,2),KV(10,1),KV(11,1),KV(23,2),KV(24,1),
      KV(27,1),KV(31,7),KV(37,3),KV(38,1),KV(42,3),KV(43,10),KV(44,1));
   OP("imadd32",0,KV(2,10),KV(9,2),KV(10,2),KV(16,6),KV(23,2),KV(24,1),
      KV(30,64),KV(38,1),KV(39,1),KV(53,1),KV(54,1),KV(59,3),KV(60,84));
   struct agx3_scs_desc scs = {
      .data=2,.width=AGX3_MEM_W32,.byte_offset=48,
      .address_kind=AGX3_SCS_ADDRESS_UNIFORM_INDEXED,.index=10,
      .uniform_base=d->payload_uniform,.component_mask=1,
      .has_address_selector=true,.address_selector=0,.slot=8,
      .c_load_marker=true,
   };
   uint8_t n;
   if (!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C,&scs,staged+used,&n))
      return false;
   used += n;
   RLD(4,57,1,0,0);
   RLD(8,7,2,1,1);
   OP("ldimm",2,KV(9,2),KV(10,2),KV(14,0x40000000),KV(15,3),KV(16,105));
   OP("imadd32",0,KV(2,4),KV(9,2),KV(10,2),KV(16,4),KV(23,2),KV(24,1),
      KV(27,1),KV(32,d->context6_uniform),KV(38,3),KV(39,1),
      KV(53,1),KV(54,1),KV(59,3),KV(60,84),KV(61,1));
   OP("bit_or",1,KV(2,2),KV(9,2),KV(10,2),KV(11,1),KV(16,2),KV(23,2),
      KV(24,2),KV(27,1),KV(31,10),KV(37,3),KV(38,2),KV(42,3),
      KV(43,16),KV(51,1));
   OP("clr",0,KV(2,9),KV(9,2),KV(10,1),KV(14,3),KV(15,25));
   OP("iadd32",0,KV(2,6),KV(9,2),KV(10,1),KV(16,6),KV(23,2),KV(24,1),
      KV(27,1),KV(32,d->context8_uniform),KV(38,3),KV(39,1),KV(44,3),KV(45,80));
   OP("clr",0,KV(2,7),KV(9,2),KV(10,1),KV(14,3),KV(15,25));
   scs=(struct agx3_scs_desc){
      .data=0,.width=AGX3_MEM_W32,.byte_offset=44,
      .address_kind=AGX3_SCS_ADDRESS_UNIFORM_INDEXED,.index=10,
      .uniform_base=d->payload_uniform,.component_mask=3,
      .has_address_selector=true,.address_selector=0,
      .value_kill=true,.index_kill=true,
   };
   if (!agx3_scs_encode_store(AGX_ISA_ENCODING_AGX3C,&scs,staged+used,&n))
      return false;
   used += n;
   /* This form consumes a 16-bit index, unlike the general SCS helpers. */
   OP("stscs",0,KV(2,6),KV(9,2),KV(10,2),KV(13,1),KV(16,8),KV(23,2),
      KV(24,2),KV(27,1),KV(31,4),KV(38,2),KV(39,2),KV(42,1),KV(45,15),
      KV(53,1),KV(54,1),KV(59,188),KV(61,1),KV(68,1),KV(70,2),KV(76,1));
   /* Shared return: select the per-thread continuation record at
    * 192 + SR57 * 512, issue its destination-less SCS load, then return. */
   RLD(4,57,1,0,0);
   OP("ldimm",2,KV(9,2),KV(10,2),KV(11,1),KV(14,192),KV(15,3),KV(16,105));
   OP("ldimm",2,KV(2,2),KV(9,2),KV(10,2),KV(11,1),KV(14,512),KV(15,3),KV(16,105));
   OP("imadd32",0,KV(9,2),KV(10,2),KV(16,4),KV(23,2),KV(24,1),KV(27,1),
      KV(31,2),KV(38,2),KV(39,2),KV(42,1),KV(53,2),KV(54,2),KV(57,1),
      KV(59,3),KV(60,84),KV(61,1));
   /* Immediate-zero slot-A operand; the indexed epilogue API intentionally
    * does not cover this native return form. */
   OP("ldscs",0,KV(10,2),KV(23,1),KV(24,1),KV(28,1),KV(38,2),KV(39,2),
      KV(42,1),KV(45,15),KV(53,1),KV(54,1),KV(59,106),KV(68,3),KV(76,2),KV(80,1));
   OP("barrier",0,KV(14,3),KV(15,9),KV(25,1),KV(30,1),KV(39,1));
   OP("end",0,KV(0,3),KV(1,39));
#undef RLD
#undef OP
   if (used > sizeof(staged))
      return false;
   memcpy(out,staged,used);
   *length=used;
   return true;
}
