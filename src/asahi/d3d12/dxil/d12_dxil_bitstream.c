/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * LLVM bitstream reader and DXBC container unwrapper. Written from the
 * public LLVM BitCodeFormat specification; block/abbrev numbering
 * cross-checked against Mesa's own DXIL writer (src/microsoft/compiler).
 */

#include "d12_dxil.h"

#include <stdlib.h>
#include <string.h>

#define MESA_LOG_TAG "d3d12agx"
#include "util/log.h"
#include "util/macros.h"

/*
 * Container
 */

static uint32_t
read_le32(const uint8_t *p)
{
   return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
          ((uint32_t)p[3] << 24);
}

bool
d12_dxil_find_bitcode(const void *data, size_t size_B,
                      const uint8_t **out_bitcode, size_t *out_size_B,
                      const char **why)
{
   const uint8_t *p = data;

   if (!p || size_B < 4) {
      *why = "input too small";
      return false;
   }

   /* Bare LLVM bitcode? */
   if (read_le32(p) == D12_BITCODE_MAGIC) {
      *out_bitcode = p;
      *out_size_B = size_B;
      return true;
   }

   if (read_le32(p) != D12_DXBC_MAGIC) {
      *why = "not a DXBC container or LLVM bitcode";
      return false;
   }

   /* DXBC header: magic, 16-byte digest, u16 major, u16 minor, u32 total
    * size, u32 part count, then part offsets.
    */
   if (size_B < 32) {
      *why = "truncated DXBC header";
      return false;
   }

   uint32_t total_size = read_le32(p + 24);
   uint32_t part_count = read_le32(p + 28);

   if (total_size > size_B) {
      *why = "DXBC size field exceeds input";
      return false;
   }

   if (part_count > 64 || 32 + (size_t)part_count * 4 > size_B) {
      *why = "implausible DXBC part count";
      return false;
   }

   for (uint32_t i = 0; i < part_count; i++) {
      uint32_t off = read_le32(p + 32 + i * 4);

      /* Each part: 4-byte fourcc, 4-byte size, then payload */
      if ((size_t)off + 8 > size_B) {
         *why = "DXBC part offset out of range";
         return false;
      }

      uint32_t part_size = read_le32(p + off + 4);
      if ((size_t)off + 8 + part_size > size_B) {
         *why = "DXBC part extends past input";
         return false;
      }

      /* 'DXIL' holds bitcode; 'ILDB' is the unsigned debug variant. */
      bool is_dxil = !memcmp(p + off, "DXIL", 4);
      bool is_ildb = !memcmp(p + off, "ILDB", 4);
      if (!is_dxil && !is_ildb)
         continue;

      const uint8_t *part = p + off + 8;

      /* DxilProgramHeader:
       *   +0  u32 ProgramVersion  ((kind << 16) | (major << 4) | minor)
       *   +4  u32 SizeInUint32
       *   +8  DxilBitcodeHeader:
       *         +0  char[4] "DXIL"
       *         +4  u32 DxilVersion
       *         +8  u32 BitcodeOffset  (relative to the BITCODE HEADER)
       *         +12 u32 BitcodeSize
       * Verified against real dxc output rather than assumed.
       */
      if (part_size < 24) {
         *why = "truncated DXIL part header";
         return false;
      }

      const uint8_t *bc_header = part + 8;
      if (memcmp(bc_header, "DXIL", 4)) {
         *why = "DXIL part lacks its bitcode header magic";
         return false;
      }

      uint32_t bc_offset = read_le32(bc_header + 8);
      uint32_t bc_size = read_le32(bc_header + 12);

      /* Offsets are from the bitcode header, which sits 8 bytes into the
       * part payload.
       */
      if ((size_t)8 + bc_offset + bc_size > part_size) {
         *why = "DXIL bitcode extends past its part";
         return false;
      }

      const uint8_t *bc = bc_header + bc_offset;
      if (bc_size < 4 || read_le32(bc) != D12_BITCODE_MAGIC) {
         *why = "DXIL part does not contain LLVM bitcode";
         return false;
      }

      *out_bitcode = bc;
      *out_size_B = bc_size;
      return true;
   }

   *why = "no DXIL part in DXBC container";
   return false;
}

bool
d12_dxil_container_stage(const void *data, size_t size_B,
                         mesa_shader_stage *out_stage)
{
   const uint8_t *p = data;

   if (!p || size_B < 32 || read_le32(p) != D12_DXBC_MAGIC)
      return false;

   uint32_t part_count = read_le32(p + 28);
   if (part_count > 64 || 32 + (size_t)part_count * 4 > size_B)
      return false;

   for (uint32_t i = 0; i < part_count; i++) {
      uint32_t off = read_le32(p + 32 + i * 4);
      if ((size_t)off + 12 > size_B)
         return false;

      if (memcmp(p + off, "DXIL", 4) && memcmp(p + off, "ILDB", 4))
         continue;

      /* ProgramVersion's high 16 bits are the DXIL shader kind. */
      uint32_t program_version = read_le32(p + off + 8);
      switch (program_version >> 16) {
      case 0: *out_stage = MESA_SHADER_FRAGMENT; return true;
      case 1: *out_stage = MESA_SHADER_VERTEX; return true;
      case 2: *out_stage = MESA_SHADER_GEOMETRY; return true;
      case 3: *out_stage = MESA_SHADER_TESS_CTRL; return true;
      case 4: *out_stage = MESA_SHADER_TESS_EVAL; return true;
      case 5: *out_stage = MESA_SHADER_COMPUTE; return true;
      /*
       * Mesh and amplification have no hardware stage and no in-tree
       * emulation to borrow, so they are translated directly into
       * compute -- a mesh shader IS a compute shader that writes vertex,
       * primitive and index buffers. Reporting COMPUTE here is the
       * truth about what gets compiled, not a fallback: the translator
       * keys the mesh dx.ops off the container kind, not off this.
       */
      case 13: *out_stage = MESA_SHADER_COMPUTE; return true;
      case 14: *out_stage = MESA_SHADER_COMPUTE; return true;
      /*
       * Library targets (lib_6_x): work-graph nodes and raytracing
       * shaders. Both are compute in every sense that matters here --
       * a node launches as a threadgroup, and ray-query traversal runs
       * inline in whatever stage invoked it.
       */
      case 6: *out_stage = MESA_SHADER_COMPUTE; return true;
      default: return false;
      }
   }

   return false;
}

/*
 * The raw DXIL shader kind, for callers that need to tell "no DXIL part"
 * from "a kind with no Mesa stage". Returning only a mapped stage was how
 * library and mesh shaders silently compiled as FRAGMENT: the caller could
 * not distinguish the two false returns, and its fallback default filled
 * the gap.
 */
bool
d12_dxil_container_kind(const void *data, size_t size_B, uint32_t *out_kind)
{
   const uint8_t *p = data;

   if (!p || size_B < 32 || read_le32(p) != D12_DXBC_MAGIC)
      return false;

   uint32_t part_count = read_le32(p + 28);
   if (part_count > 64 || 32 + (size_t)part_count * 4 > size_B)
      return false;

   for (uint32_t i = 0; i < part_count; i++) {
      uint32_t off = read_le32(p + 32 + i * 4);
      if ((size_t)off + 12 > size_B)
         return false;

      if (memcmp(p + off, "DXIL", 4) && memcmp(p + off, "ILDB", 4))
         continue;

      *out_kind = read_le32(p + off + 8) >> 16;
      return true;
   }

   return false;
}

const char *
d12_dxil_kind_name(uint32_t kind)
{
   switch (kind) {
   case 0: return "pixel";
   case 1: return "vertex";
   case 2: return "geometry";
   case 3: return "hull";
   case 4: return "domain";
   case 5: return "compute";
   case 6: return "library";
   case 7: return "ray generation";
   case 8: return "intersection";
   case 9: return "any hit";
   case 10: return "closest hit";
   case 11: return "miss";
   case 12: return "callable";
   case 13: return "mesh";
   case 14: return "amplification";
   case 15: return "node";
   default: return "unknown";
   }
}

/*
 * Bitstream primitives. The stream is a little-endian bit sequence; we read
 * from the low bit of each byte upward.
 */

void
d12_bitstream_init(struct d12_bitstream *bs, const uint8_t *data,
                   size_t size_B)
{
   memset(bs, 0, sizeof(*bs));
   bs->data = data;
   bs->size_B = size_B;
}

static void
bitstream_fail(struct d12_bitstream *bs, const char *msg)
{
   if (!bs->error) {
      bs->error = true;
      bs->error_msg = msg;
   }
}

bool
d12_bitstream_at_end(const struct d12_bitstream *bs)
{
   return bs->pos_bits >= (uint64_t)bs->size_B * 8;
}

uint64_t
d12_bitstream_read64(struct d12_bitstream *bs, unsigned bits)
{
   if (bits == 0)
      return 0;

   if (bits > 64) {
      bitstream_fail(bs, "read of more than 64 bits");
      return 0;
   }

   if (bs->error)
      return 0;

   if (bs->pos_bits + bits > (uint64_t)bs->size_B * 8) {
      bitstream_fail(bs, "read past end of bitstream");
      return 0;
   }

   uint64_t result = 0;
   for (unsigned i = 0; i < bits; i++) {
      uint64_t bit_pos = bs->pos_bits + i;
      uint8_t byte = bs->data[bit_pos >> 3];
      uint64_t bit = (byte >> (bit_pos & 7)) & 1;
      result |= bit << i;
   }

   bs->pos_bits += bits;
   return result;
}

uint32_t
d12_bitstream_read(struct d12_bitstream *bs, unsigned bits)
{
   if (bits > 32) {
      bitstream_fail(bs, "32-bit read of more than 32 bits");
      return 0;
   }
   return (uint32_t)d12_bitstream_read64(bs, bits);
}

uint64_t
d12_bitstream_read_vbr(struct d12_bitstream *bs, unsigned bits)
{
   if (bits < 2 || bits > 32) {
      bitstream_fail(bs, "invalid VBR width");
      return 0;
   }

   const uint64_t continue_mask = 1ull << (bits - 1);
   const uint64_t value_mask = continue_mask - 1;

   uint64_t result = 0;
   unsigned shift = 0;

   for (;;) {
      uint64_t chunk = d12_bitstream_read64(bs, bits);
      if (bs->error)
         return 0;

      if (shift >= 64) {
         bitstream_fail(bs, "VBR value overflows 64 bits");
         return 0;
      }

      result |= (chunk & value_mask) << shift;

      if (!(chunk & continue_mask))
         return result;

      shift += bits - 1;
   }
}

void
d12_bitstream_align32(struct d12_bitstream *bs)
{
   uint64_t rem = bs->pos_bits & 31;
   if (rem)
      d12_bitstream_read64(bs, 32 - (unsigned)rem);
}

/*
 * Abbreviation handling
 */

static bool
parse_define_abbrev(struct d12_bitstream *bs, struct d12_abbrev_list *list)
{
   if (list->count >= D12_MAX_ABBREVS) {
      bitstream_fail(bs, "too many abbreviations in block");
      return false;
   }

   struct d12_abbrev *ab = &list->abbrevs[list->count];
   memset(ab, 0, sizeof(*ab));

   uint64_t num_ops = d12_bitstream_read_vbr(bs, 5);
   if (bs->error)
      return false;

   if (num_ops > D12_MAX_ABBREV_OPS) {
      bitstream_fail(bs, "abbreviation has too many operands");
      return false;
   }

   for (uint64_t i = 0; i < num_ops; i++) {
      struct d12_abbrev_op *op = &ab->ops[i];

      bool is_literal = d12_bitstream_read(bs, 1) != 0;
      if (is_literal) {
         op->enc = D12_ENC_LITERAL;
         op->value = d12_bitstream_read_vbr(bs, 8);
      } else {
         op->enc = d12_bitstream_read(bs, 3);
         if (op->enc == D12_ENC_FIXED || op->enc == D12_ENC_VBR)
            op->value = d12_bitstream_read_vbr(bs, 5);
      }

      if (bs->error)
         return false;
   }

   ab->num_ops = (unsigned)num_ops;
   list->count++;
   return true;
}

static char
decode_char6(uint32_t v)
{
   if (v < 26)
      return 'a' + v;
   if (v < 52)
      return 'A' + (v - 26);
   if (v < 62)
      return '0' + (v - 52);
   return v == 62 ? '.' : '_';
}

/* Read one scalar operand of the given encoding. */
static uint64_t
read_scalar_op(struct d12_bitstream *bs, const struct d12_abbrev_op *op)
{
   switch (op->enc) {
   case D12_ENC_FIXED:
      return op->value ? d12_bitstream_read64(bs, (unsigned)op->value) : 0;
   case D12_ENC_VBR:
      return d12_bitstream_read_vbr(bs, (unsigned)op->value);
   case D12_ENC_CHAR6:
      return (uint64_t)decode_char6(d12_bitstream_read(bs, 6));
   default:
      bitstream_fail(bs, "unexpected operand encoding");
      return 0;
   }
}

/* Start decoding a record into the walk's shared operand scratch. */
static void
record_begin(struct d12_bitstream *bs, struct d12_record *rec)
{
   memset(rec, 0, sizeof(*rec));
   rec->ops = bs->record_ops;
}

static bool
record_push(struct d12_bitstream *bs, struct d12_record *rec, uint64_t v)
{
   if (rec->num_ops >= bs->record_ops_capacity) {
      if (rec->num_ops >= D12_LIMIT_RECORD_OPS) {
         bitstream_fail(bs, "record has too many operands");
         return false;
      }

      unsigned capacity = MAX2(bs->record_ops_capacity * 2, 1024u);
      uint64_t *ops = realloc(bs->record_ops, capacity * sizeof(*ops));
      if (!ops) {
         bitstream_fail(bs, "out of memory decoding a record");
         return false;
      }
      bs->record_ops = ops;
      bs->record_ops_capacity = capacity;
      rec->ops = ops;
   }
   rec->ops[rec->num_ops++] = v;
   return true;
}

static bool
read_abbreviated_record(struct d12_bitstream *bs,
                        const struct d12_abbrev *ab, struct d12_record *rec)
{
   record_begin(bs, rec);

   for (unsigned i = 0; i < ab->num_ops; i++) {
      const struct d12_abbrev_op *op = &ab->ops[i];

      if (op->enc == D12_ENC_LITERAL) {
         if (!record_push(bs, rec, op->value))
            return false;
         continue;
      }

      if (op->enc == D12_ENC_ARRAY) {
         /* Array length, then elements described by the NEXT operand. */
         if (i + 1 >= ab->num_ops) {
            bitstream_fail(bs, "array abbreviation missing element type");
            return false;
         }

         const struct d12_abbrev_op *elem = &ab->ops[++i];
         uint64_t len = d12_bitstream_read_vbr(bs, 6);
         if (bs->error)
            return false;

         for (uint64_t e = 0; e < len; e++) {
            uint64_t v = elem->enc == D12_ENC_LITERAL ? elem->value
                                                      : read_scalar_op(bs, elem);
            if (bs->error || !record_push(bs, rec, v))
               return false;
         }
         continue;
      }

      if (op->enc == D12_ENC_BLOB) {
         uint64_t len = d12_bitstream_read_vbr(bs, 6);
         if (bs->error)
            return false;

         d12_bitstream_align32(bs);

         uint64_t byte_pos = bs->pos_bits / 8;
         if (byte_pos + len > bs->size_B) {
            bitstream_fail(bs, "blob extends past end of bitstream");
            return false;
         }

         rec->blob = bs->data + byte_pos;
         rec->blob_size = (size_t)len;
         bs->pos_bits += len * 8;
         d12_bitstream_align32(bs);
         continue;
      }

      uint64_t v = read_scalar_op(bs, op);
      if (bs->error || !record_push(bs, rec, v))
         return false;
   }

   if (rec->num_ops == 0) {
      bitstream_fail(bs, "abbreviated record with no operands");
      return false;
   }

   /* First operand is the record code. */
   rec->code = (uint32_t)rec->ops[0];
   memmove(rec->ops, rec->ops + 1, (rec->num_ops - 1) * sizeof(rec->ops[0]));
   rec->num_ops--;
   return true;
}

/*
 * Block walking
 */

static const struct d12_abbrev_list *
blockinfo_for(const struct d12_blockinfo *bi, unsigned block_id)
{
   for (unsigned i = 0; i < bi->count; i++) {
      if (bi->entries[i].block_id == block_id)
         return &bi->entries[i].abbrevs;
   }
   return NULL;
}

static struct d12_abbrev_list *
blockinfo_get_or_add(struct d12_bitstream *bs, struct d12_blockinfo *bi,
                     unsigned block_id)
{
   for (unsigned i = 0; i < bi->count; i++) {
      if (bi->entries[i].block_id == block_id)
         return &bi->entries[i].abbrevs;
   }

   if (bi->count >= D12_MAX_BLOCKINFO_BLOCKS) {
      bitstream_fail(bs, "too many blocks described in BLOCKINFO");
      return NULL;
   }

   bi->entries[bi->count].block_id = block_id;
   memset(&bi->entries[bi->count].abbrevs, 0,
          sizeof(bi->entries[bi->count].abbrevs));
   return &bi->entries[bi->count++].abbrevs;
}

static bool
walk_block(struct d12_bitstream *bs, unsigned block_id, unsigned abbrev_width,
           unsigned depth, struct d12_blockinfo *blockinfo,
           const struct d12_block_visitor *visitor)
{
   if (depth > 16) {
      bitstream_fail(bs, "block nesting too deep");
      return false;
   }

   /* Abbreviations visible here: those from BLOCKINFO for this block ID,
    * plus any DEFINE_ABBREVs local to this block instance.
    */
   struct d12_abbrev_list local = {0};
   const struct d12_abbrev_list *inherited =
      blockinfo_for(blockinfo, block_id);
   unsigned inherited_count = inherited ? inherited->count : 0;

   /* BLOCKINFO's SETBID target for DEFINE_ABBREVs inside a BLOCKINFO block */
   struct d12_abbrev_list *blockinfo_target = NULL;

   if (visitor->enter_block && !visitor->enter_block(visitor->user, block_id,
                                                     depth))
      return false;

   for (;;) {
      if (bs->error)
         return false;

      if (d12_bitstream_at_end(bs)) {
         bitstream_fail(bs, "unterminated block");
         return false;
      }

      uint32_t id = d12_bitstream_read(bs, abbrev_width);
      if (bs->error)
         return false;

      if (id == D12_ABBREV_END_BLOCK) {
         d12_bitstream_align32(bs);
         if (visitor->exit_block &&
             !visitor->exit_block(visitor->user, block_id, depth))
            return false;
         return true;
      }

      if (id == D12_ABBREV_ENTER_SUBBLOCK) {
         uint32_t sub_id = (uint32_t)d12_bitstream_read_vbr(bs, 8);
         uint32_t sub_width = (uint32_t)d12_bitstream_read_vbr(bs, 4);
         d12_bitstream_align32(bs);
         uint32_t len_words = d12_bitstream_read(bs, 32);
         if (bs->error)
            return false;

         if (sub_width < 1 || sub_width > 32) {
            bitstream_fail(bs, "invalid sub-block abbreviation width");
            return false;
         }

         uint64_t body_start = bs->pos_bits;
         uint64_t body_end = body_start + (uint64_t)len_words * 32;
         if (body_end > (uint64_t)bs->size_B * 8) {
            bitstream_fail(bs, "sub-block extends past end of bitstream");
            return false;
         }

         if (sub_id == D12_BLOCK_BLOCKINFO) {
            /* Parse BLOCKINFO ourselves: it declares abbreviations for
             * other blocks rather than carrying payload of its own.
             */
            struct d12_abbrev_list ignored = {0};
            for (;;) {
               uint32_t bi_id = d12_bitstream_read(bs, sub_width);
               if (bs->error)
                  return false;

               if (bi_id == D12_ABBREV_END_BLOCK) {
                  d12_bitstream_align32(bs);
                  break;
               }

               if (bi_id == D12_ABBREV_DEFINE_ABBREV) {
                  struct d12_abbrev_list *target =
                     blockinfo_target ? blockinfo_target : &ignored;
                  if (!parse_define_abbrev(bs, target))
                     return false;
                  continue;
               }

               if (bi_id == D12_ABBREV_UNABBREV_RECORD) {
                  struct d12_record rec;
                  record_begin(bs, &rec);
                  rec.code = (uint32_t)d12_bitstream_read_vbr(bs, 6);
                  uint64_t n = d12_bitstream_read_vbr(bs, 6);
                  for (uint64_t i = 0; i < n; i++) {
                     uint64_t v = d12_bitstream_read_vbr(bs, 6);
                     if (bs->error || !record_push(bs, &rec, v))
                        return false;
                  }

                  if (rec.code == D12_BLOCKINFO_SETBID && rec.num_ops >= 1) {
                     blockinfo_target = blockinfo_get_or_add(
                        bs, blockinfo, (unsigned)rec.ops[0]);
                     if (!blockinfo_target)
                        return false;
                  }
                  continue;
               }

               bitstream_fail(bs, "unexpected abbreviated record in "
                                  "BLOCKINFO");
               return false;
            }
         } else {
            if (!walk_block(bs, sub_id, sub_width, depth + 1, blockinfo,
                            visitor))
               return false;
         }

         /* Trust the declared length: skip to the recorded end so a
          * partially-understood sub-block cannot desynchronize the parent.
          */
         bs->pos_bits = body_end;
         continue;
      }

      if (id == D12_ABBREV_DEFINE_ABBREV) {
         if (!parse_define_abbrev(bs, &local))
            return false;
         continue;
      }

      struct d12_record rec;
      record_begin(bs, &rec);

      if (id == D12_ABBREV_UNABBREV_RECORD) {
         rec.code = (uint32_t)d12_bitstream_read_vbr(bs, 6);
         uint64_t n = d12_bitstream_read_vbr(bs, 6);
         if (bs->error)
            return false;

         for (uint64_t i = 0; i < n; i++) {
            uint64_t v = d12_bitstream_read_vbr(bs, 6);
            if (bs->error || !record_push(bs, &rec, v))
               return false;
         }
      } else {
         unsigned index = id - D12_ABBREV_FIRST_APPLICATION;
         const struct d12_abbrev *ab;

         if (index < inherited_count) {
            ab = &inherited->abbrevs[index];
         } else if (index - inherited_count < local.count) {
            ab = &local.abbrevs[index - inherited_count];
         } else {
            bitstream_fail(bs, "reference to undefined abbreviation");
            return false;
         }

         if (!read_abbreviated_record(bs, ab, &rec))
            return false;
      }

      if (visitor->record &&
          !visitor->record(visitor->user, block_id, depth, &rec, id))
         return false;
   }
}

static bool walk_top_level(struct d12_bitstream *bs,
                           struct d12_blockinfo *blockinfo,
                           const struct d12_block_visitor *visitor);

bool
d12_dxil_walk(struct d12_bitstream *bs,
              const struct d12_block_visitor *visitor)
{
   uint32_t magic = d12_bitstream_read(bs, 32);
   if (bs->error || magic != D12_BITCODE_MAGIC) {
      bitstream_fail(bs, "missing LLVM bitcode magic");
      return false;
   }

   /* Top level: a sequence of ENTER_SUBBLOCK entries at abbrev width 2. */
   struct d12_blockinfo blockinfo = {0};
   bool ok = walk_top_level(bs, &blockinfo, visitor);

   /* The record scratch belongs to this walk; a visitor that kept a
    * record pointer would already be wrong before this free. */
   free(bs->record_ops);
   bs->record_ops = NULL;
   bs->record_ops_capacity = 0;
   return ok;
}

static bool
walk_top_level(struct d12_bitstream *bs, struct d12_blockinfo *blockinfo,
               const struct d12_block_visitor *visitor)
{
   while (!d12_bitstream_at_end(bs)) {
      /* Trailing padding is zero bits; stop if fewer than 32 bits remain. */
      if (bs->pos_bits + 32 > (uint64_t)bs->size_B * 8)
         break;

      uint32_t id = d12_bitstream_read(bs, 2);
      if (bs->error)
         return false;

      if (id != D12_ABBREV_ENTER_SUBBLOCK) {
         bitstream_fail(bs, "expected a top-level block");
         return false;
      }

      uint32_t block_id = (uint32_t)d12_bitstream_read_vbr(bs, 8);
      uint32_t width = (uint32_t)d12_bitstream_read_vbr(bs, 4);
      d12_bitstream_align32(bs);
      uint32_t len_words = d12_bitstream_read(bs, 32);
      if (bs->error)
         return false;

      if (width < 1 || width > 32) {
         bitstream_fail(bs, "invalid top-level abbreviation width");
         return false;
      }

      uint64_t body_end = bs->pos_bits + (uint64_t)len_words * 32;
      if (body_end > (uint64_t)bs->size_B * 8) {
         bitstream_fail(bs, "top-level block extends past end");
         return false;
      }

      if (!walk_block(bs, block_id, width, 0, blockinfo, visitor))
         return false;

      bs->pos_bits = body_end;
   }

   return true;
}
