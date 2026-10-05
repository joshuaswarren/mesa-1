/*
 * Copyright 2024 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_linker.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include "util/macros.h"
#include "util/mesa-blake3.h"
#include "util/ralloc.h"
#include "agx_abi.h"
#include "agx3_code_container_core.h"
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx3_fltnorm_core.h"
#include "agx3_pixel_core.h"
#include "agx_device.h"
#include "agx_pack.h"
#include "agx_scratch.h"

/*
 * When sample shading is used with a non-monolithic fragment shader, we
 * fast-link a program with the following structure:
 *
 *    Fragment prolog;
 *
 *    for (u16 sample_bit = 1; sample_bit < (1 << # of samples); ++sample_bit) {
 *       API fragment shader;
 *       Fragment epilog;
 *    }
 *
 * This means the prolog runs per-pixel but the fragment shader and epilog run
 * per-sample. To do this, we need to generate the loop on the fly. The
 * following binary sequences form the relevant loop.
 */

static_assert(AGX_ABI_FIN_SAMPLE_MASK == 2, "r1l known");

void
agx_fs_epilog_apply_device_policy(enum agx_chip chip,
                                  struct agx_fs_epilog_key *key)
{
   if (!key)
      return;

   const bool g17p = chip == AGX_CHIP_G17P;
   /* G18 coverage also dirties color during depth-only or fully masked draws.
    * Preserve the previous tile value while executing the other side effects.
    */
   key->preserve_zero_mask = g17p || chip == AGX_CHIP_G18G;
   /* Both AGX3 tile interfaces move packed bytes. The RGB transfer function
    * belongs around blending; alpha remains linear. R11G11B10 is explicitly
    * packed into one raw tile word, with a matching R32_UINT store PBE. */
   key->lower_srgb = g17p || agx_chip_m5_paths(chip);
   key->lower_r11g11b10 = g17p || chip == AGX_CHIP_G18G;
   if (chip == AGX_CHIP_G18G && key->nr_samples == 1) {
      /* Match the single-sample render-pass layout. Native G18 eight-target
       * renders use the complete 32-KiB imageblock with 1024 raster samples. */
      key->max_bytes_per_tile = 32768;
      key->max_samples_per_tile = 1024;
   }
}

bool
agx_fs_epilog_may_discard(const struct agx_fs_epilog_link_info *link,
                          bool alpha_to_coverage,
                          bool nontrivial_force_early)
{
   return !nontrivial_force_early &&
          (alpha_to_coverage || (link && (link->write_z || link->write_s)));
}

/* clang-format off */
static const uint8_t sample_loop_header[] = {
   /* mov_imm r0l, 0x0, 0b0 */
   0x62, 0x00, 0x00, 0x00,

   /* mov_imm r1l, 0x0, 0b0 */
   0x62, 0x04, 0x01, 0x00,
};

#define STOP                                                                   \
   /* stop */                                                                  \
   0x88, 0x00,                                                                 \
                                                                               \
   /* trap */                                                                  \
   0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00,                             \
   0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00,

static const uint8_t stop[] = {STOP};

static const uint8_t sample_loop_footer[] = {
   /* iadd r1l, 0, r1l, lsl 1 */
   0x0e, 0x04, 0x00, 0x20, 0x84, 0x00, 0x00, 0x00,

   /* while_icmp r0l, ult, r1h, 0, 1 */
   0x52, 0x2c, 0x42, 0x00, 0x00, 0x00,

   /* jmp_exec_any */
   0x00, 0xc0, 0x00, 0x00, 0x00, 0x00,

   /* pop_exec r0l, 1 */
   0x52, 0x0e, 0x00, 0x00, 0x00, 0x00,

   STOP
};

/* Offset in sample_loop_footer to the jmp_exec_any's target */
#define SAMPLE_LOOP_FOOTER_JMP_PATCH_OFFS (16)

/* Offset of the jmp_exec_any, for calculating the PC offsets */
#define SAMPLE_LOOP_FOOTER_JMP_OFFS (14)

/* Offset in sample_loop_footer to the while_icmp's sample count immediate. Bit
 * position in the byte given by the shift.
 */
#define SAMPLE_LOOP_FOOTER_COUNT_PATCH_OFFS (11)
#define SAMPLE_LOOP_FOOTER_COUNT_SHIFT (4)
/* clang-format on */


/*
 * Apple's AGX3 code container.
 *
 * On G17P the LDSHDR record names an entry point INSIDE a container, not the
 * base of an executable allocation, and the shader loader reads container
 * metadata at fixed negative offsets from that entry.  The m1n1 add3 witness
 * (proxyclient/experiments/g17p_add3_code.py) lays it out as:
 *
 *   +0x000  dword 0x340                 size of the leading block
 *   +0x040  0x0006 filler .. +0x340
 *   +0x100  helper record table
 *   +0x200  helper record table (identical)
 *   +0x340  dword block_size            == entry - 0x80, the CODE block size
 *   +0x380  constant program (0x40)     == entry - 0x40
 *   +0x3C0  main program                <- LDSHDR points here
 *
 * block_size = 0x40 (its own header) + 0x40 (constant program) +
 * align64(code).  For the witness's 56-byte main program that is
 * 0x40 + 0x40 + 0x40 = 0xC0, which is the dword the capture actually
 * contains, and the container ends at 0x340 + 0xC0 = 0x400 -- exactly the
 * 1 KiB closure the witness proved on hardware.  So this rule reproduces the
 * witness byte for byte rather than merely resembling it.
 *
 * Honeykrisp's compute path guarantees no preamble, so the constant program
 * is the witness's minimal terminating one.
 */
#define AGX3_CONTAINER_LEAD        0x340u
#define AGX3_CONTAINER_ENTRY       0x3C0u
#define AGX3_CONTAINER_BLOCK_HDR   0x40u
#define AGX3_CONTAINER_CONST_PROG  0x40u

/* "0f 00 54 <opcode> 00 x6 06 00 06 00 06 00" per record, then two terminals. */
static void
agx3_fill_helper_table(uint8_t *image, size_t base)
{
   static const uint8_t opcodes[] = {0x60, 0x50, 0x40, 0x30, 0x20, 0x10};
   static const uint8_t terminal[16] = {
      0xf7, 0x03, 0xaa, 0x00, 0x8f, 0x02, 0x54, 0x01,
      0x06, 0x00, 0x06, 0x00, 0x06, 0x00, 0x06, 0x00,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(opcodes); ++i) {
      uint8_t *record = image + base + i * 0x10;

      memset(record, 0, 0x10);
      record[0] = 0x0f;
      record[2] = 0x54;
      record[3] = opcodes[i];
      for (unsigned h = 10; h < 16; h += 2) {
         record[h] = 0x06;
         record[h + 1] = 0x00;
      }
   }
   memcpy(image + base + 0x60, terminal, sizeof(terminal));
   memcpy(image + base + 0x70, terminal, sizeof(terminal));
}

static void
agx3_put_le32(uint8_t *where, uint32_t value)
{
   where[0] = value;
   where[1] = value >> 8;
   where[2] = value >> 16;
   where[3] = value >> 24;
}

/* Total container bytes for `code_size` bytes of executable. */
static size_t
agx3_container_lead_size(enum agx_chip chip)
{
   return chip == AGX_CHIP_G18G ? AGX3_G18G_CONTAINER_LEAD_SIZE
                               : AGX3_CONTAINER_LEAD;
}

static size_t
agx3_container_size(enum agx_chip chip, size_t code_size)
{
   return agx3_container_lead_size(chip) + AGX3_CONTAINER_BLOCK_HDR +
          AGX3_CONTAINER_CONST_PROG + ALIGN_POT(code_size, 64);
}

/* The container's leading block: dword 0x340, 0x0006 filler, two helper
 * tables. Identical in both captured Apple code images, so it is a fixed
 * prologue rather than anything workload-specific. */
void
agx3_build_container_lead(uint8_t *image)
{
   memset(image, 0, AGX3_CONTAINER_LEAD);
   agx3_put_le32(image, AGX3_CONTAINER_LEAD);
   for (size_t offset = 0x40; offset < AGX3_CONTAINER_LEAD; offset += 2) {
      image[offset] = 0x06;
      image[offset + 1] = 0x00;
   }
   agx3_fill_helper_table(image, 0x100);
   agx3_fill_helper_table(image, 0x200);
}

static void
agx3_build_container(enum agx_chip chip, uint8_t *image, size_t code_size, bool fragment,
                     bool sampled_fragment)
{
   const size_t lead_size = agx3_container_lead_size(chip);
   memset(image, 0, agx3_container_size(chip, code_size));
   if (chip == AGX_CHIP_G18G) {
      bool built = agx3_build_g18g_container_lead(image, lead_size);
      assert(built);
      (void)built;
   } else if (agx_chip_m5_paths(chip)) {
      bool built = agx3_build_g17g_container_lead(image, AGX3_CONTAINER_LEAD);
      assert(built);
      (void)built;
   } else {
      agx3_build_container_lead(image);
   }

   agx3_put_le32(image + lead_size,
                 AGX3_CONTAINER_BLOCK_HDR + AGX3_CONTAINER_CONST_PROG +
                    ALIGN_POT(code_size, 64));

   uint8_t *constant = image + lead_size + AGX3_CONTAINER_BLOCK_HDR;
   if (chip == AGX_CHIP_G18G) {
      agx3_build_container_stop_program_agx3d(constant);
      return;
   }
   agx3_build_container_constant_program(constant,
                                          fragment || agx_chip_m5_paths(chip),
                                          sampled_fragment && !agx_chip_m5_paths(chip));
}

static bool
agx3_uses_code_container(const struct agx_device *dev)
{
   return dev->chip == AGX_CHIP_G17P || agx_chip_m5_paths(dev->chip);
}

bool
agx_fast_link(struct agx_linked_shader *linked, struct agx_device *dev,
              bool fragment, struct agx_shader_part *main,
              struct agx_shader_part *prolog, struct agx_shader_part *epilog,
              unsigned nr_samples_shaded)
{
   /* These literal loop fragments are AGX2 machine code. G17P exposes only
    * one sample and its Vulkan path folds sample shading before split-FS
    * lowering; fail closed if a caller ever bypasses that canonicalization. */
   if ((dev->chip == AGX_CHIP_G17P || agx_chip_m5_paths(dev->chip)) &&
       nr_samples_shaded) {
      fprintf(stderr,
              "agx_fast_link: external sample loop is not AGX3 machine code\n");
      return false;
   }

   uint8_t fused_colour[AGX3_COLOUR_LINK_SIZE];
   const bool colour_fused = dev->chip == AGX_CHIP_G17P && fragment &&
      !prolog && !nr_samples_shaded && main && epilog &&
      agx3_fuse_colour_parts(&main->info.agx3_colour_link,
                              &epilog->info.agx3_colour_link, fused_colour);
   /* A native fragment enters its pixel transaction once across all linked
    * parts.  A sample-mask prolog already opens it, so retaining the main
    * shader's second entry resets the coverage state immediately before its
    * forced-early wait.  Keep the first gate and remove every later one.  A
    * gate at offset zero is outside all of that part's branches; otherwise
    * retain the older branch-free proof that no branch spans the bytes. */
   struct agx_shader_part *parts[] = {prolog, main, epilog};
   bool drop_entry[ARRAY_SIZE(parts)] = {false};
   bool entry_seen = false;
   if (dev->chip == AGX_CHIP_G17P && fragment && !nr_samples_shaded &&
       !colour_fused) {
      int first_entry = -1;
      for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i) {
         struct agx_shader_part *part = parts[i];
         if (!part || !part->info.agx3_colour_link.entry_emitted)
            continue;

         const uint32_t off = part->info.agx3_colour_link.entry_offset;
         if (entry_seen) {
            if (off != 0 && !part->info.agx3_colour_link.branch_free)
               return false;
            drop_entry[i] = true;
         } else {
            entry_seen = true;
            first_entry = i;
         }
      }

      /* v52 AGX_G17P_EARLY_COLOUR_GATE: a gate the main part opened early on
       * behalf of a colour epilog is removed again when no later part opens
       * the protocol (for example no colour target is bound). The compiler
       * placed it at offset zero or in a branch-free part. */
      if (first_entry >= 0 &&
          parts[first_entry]->info.agx3_colour_link.entry_optional) {
         bool later = false;
         for (unsigned i = first_entry + 1; i < ARRAY_SIZE(parts); ++i)
            later |= parts[i] && parts[i]->info.agx3_colour_link.entry_emitted;
         if (!later) {
            const struct agx_shader_part *part = parts[first_entry];
            if (part->info.agx3_colour_link.entry_offset != 0 &&
                !part->info.agx3_colour_link.branch_free &&
                !part->info.agx3_colour_link.entry_in_entry_block)
               return false;
            drop_entry[first_entry] = true;
         }
      }

      for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i) {
         if (!drop_entry[i])
            continue;
         struct agx_shader_part *part = parts[i];
         const uint32_t off = part->info.agx3_colour_link.entry_offset;
         uint8_t gate[AGX3_PIXEL_MAX_LENGTH], gate_size;
         if ((off & 1) || off > part->info.main_size ||
             part->info.main_size - off < AGX3_PIXEL_MAX_LENGTH ||
             !agx3_pixel_colour_entry(AGX_ISA_ENCODING_AGX3B, gate,
                                      &gate_size) ||
             gate_size != AGX3_PIXEL_MAX_LENGTH ||
             memcmp((const uint8_t *)part->binary + part->info.main_offset +
                       off,
                    gate, gate_size))
            return false;
      }
   }
   size_t size = 0;
   unsigned nr_gprs = 0, scratch_size = 0, agx3_spectpr_scratch_size = 0;
   unsigned staging_halfs = 0;
   unsigned colour_classes = 0;
   bool staging_known = true;
   bool reads_tib = false, writes_sample_mask = false,
        disable_tri_merging = false, tag_write_disable = true;
   bool sampled_fragment = false;

   if (nr_samples_shaded) {
      size += sizeof(sample_loop_header);

      if (nr_samples_shaded > 1)
         size += sizeof(sample_loop_footer);
      else
         size += sizeof(stop);
   }

   for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i) {
      struct agx_shader_part *part = parts[i];
      if (!part)
         continue;

      size += part->info.main_size;

      nr_gprs = MAX2(nr_gprs, part->info.nr_gprs);
      staging_halfs = MAX2(staging_halfs, part->info.agx3_staging_halfs);
      staging_known &= part->info.agx3_staging_known;
      scratch_size = MAX2(scratch_size, part->info.scratch_size);
      agx3_spectpr_scratch_size =
         MAX2(agx3_spectpr_scratch_size,
              part->info.agx3_spectpr_scratch_size);
      reads_tib |= part->info.reads_tib;
      writes_sample_mask |= part->info.writes_sample_mask;
      disable_tri_merging |= part->info.disable_tri_merging;
      linked->uses_base_param |= part->info.uses_base_param;
      linked->uses_txf |= part->info.uses_txf;
      sampled_fragment |= part->info.g17p_sampled_fragment;
      tag_write_disable &= part->info.tag_write_disable;
      if (part->info.agx3_colour_link.colour_classes &
          ~BITFIELD_MASK(AGX3_G17P_MAX_COLOUR_PARTITIONS))
         return false;
      colour_classes |= part->info.agx3_colour_link.colour_classes;
   }

   if (colour_fused)
      size = sizeof(fused_colour);
   else
      for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i)
         size -= drop_entry[i] ? AGX3_PIXEL_MAX_LENGTH : 0;

   assert(size > 0 && "must stop");

   if ((dev->chip == AGX_CHIP_G17P || agx_chip_m5_paths(dev->chip)) &&
       (main->info.stage == MESA_SHADER_VERTEX ||
        main->info.stage == MESA_SHADER_FRAGMENT ||
        main->info.stage == MESA_SHADER_COMPUTE) &&
       staging_known && !linked->g17p_compiler_d_valid) {
      linked->g17p_compiler_d_valid = true;
      linked->g17p_compiler_d = staging_halfs * 2;
   }

   const bool container = agx3_uses_code_container(dev);
   const size_t alloc_size = container ? agx3_container_size(dev->chip, size) : size;

   linked->bo = agx_bo_create(dev, alloc_size, 0, AGX_BO_EXEC | AGX_BO_LOW_VA,
                              "Linked executable");
   linked->size = size;
   linked->entry_offset = container ? agx3_container_lead_size(dev->chip) +
                                        AGX3_CODE_BLOCK_ENTRY_OFFSET : 0;
   if (!linked->bo) {
      fprintf(stderr,
              "agx_fast_link: executable low-VA allocation of %zu bytes "
              "failed (shader base 0x%" PRIx64 ")\n",
              size, dev->shader_base);
      return false;
   }

   uint8_t *container_map = agx_bo_map(linked->bo);
   uint8_t *linked_map = container_map ? container_map + linked->entry_offset : NULL;
   if (!linked_map) {
      fprintf(stderr,
              "agx_fast_link: executable BO mapping failed (size %zu, "
              "VA 0x%" PRIx64 ", handle %u)\n",
              size, linked->bo->va ? linked->bo->va->addr : 0,
              (unsigned)linked->bo->handle);
      agx_bo_unreference(dev, linked->bo);
      linked->bo = NULL;
      return false;
   }

   if (container)
      agx3_build_container(dev->chip, container_map, size, fragment,
                           fragment && sampled_fragment);

   size_t offset = 0;
   if (colour_fused) {
      memcpy(linked_map, fused_colour, sizeof(fused_colour));
      offset = sizeof(fused_colour);
   }

   /* FS prolog happens per-pixel, outside the sample loop */
   if (prolog) {
      size_t sz = prolog->info.main_size;
      memcpy(linked_map + offset, prolog->binary, sz);
      if (dev->chip == AGX_CHIP_G17P && fragment &&
          !agx3_fltnorm_relocate_start(linked_map + offset, sz,
                                       prolog->info.agx3_fltnorm_start, offset))
         goto invalid_start_marker;
      offset += sz;
   }

   if (nr_samples_shaded) {
      memcpy(linked_map + offset, sample_loop_header,
             sizeof(sample_loop_header));
      offset += sizeof(sample_loop_header);
   }

   size_t sample_loop_begin = offset;

   /* Main shader and epilog happen in the sample loop, so start from i=1 */
   for (unsigned i = 1; !colour_fused && i < ARRAY_SIZE(parts); ++i) {
      struct agx_shader_part *part = parts[i];
      if (!part)
         continue;

      size_t sz = part->info.main_size;
      const uint8_t *part_code = (const uint8_t *)part->binary + part->info.main_offset;
      if (drop_entry[i]) {
         const unsigned gate = part->info.agx3_colour_link.entry_offset;
         memcpy(linked_map + offset, part_code, gate);
         memcpy(linked_map + offset + gate, part_code + gate + AGX3_PIXEL_MAX_LENGTH,
                sz - gate - AGX3_PIXEL_MAX_LENGTH);
         sz -= AGX3_PIXEL_MAX_LENGTH;
      } else {
         memcpy(linked_map + offset, part_code, sz);
      }
      if (dev->chip == AGX_CHIP_G17P && fragment &&
          !agx3_fltnorm_relocate_start(linked_map + offset, sz,
                                       part->info.agx3_fltnorm_start, offset))
         goto invalid_start_marker;

      /* v52 AGX_G17P_INPUT_MARK: a fragment carries one input entry marker,
       * on its first input. Drop this part's LLDCOL first-load marker when
       * an earlier part already marked an input, as Apple's throughput
       * fragment leaves its blend read unmarked after its marked rld. The
       * packer recorded the load's offset; check the bytes before changing
       * the one bit (byte 1 bit 3, the f78 first marker). */
      if (dev->chip == AGX_CHIP_G17P && fragment &&
          (agx_get_fs_structure_flags() & AGX_FS_STRUCTURE_INPUT_MARK) &&
          part->info.agx3_colour_link.lldcol_first_valid &&
          part->info.agx3_colour_link.lldcol_first_offset != UINT32_MAX) {
         bool earlier_marked = false;
         for (unsigned j = 0; j < i; ++j)
            earlier_marked |=
               parts[j] && parts[j]->info.agx3_colour_link.input_marked;
         if (earlier_marked) {
            uint32_t lo = part->info.agx3_colour_link.lldcol_first_offset;
            if (drop_entry[i] &&
                part->info.agx3_colour_link.entry_offset < lo)
               lo -= AGX3_PIXEL_MAX_LENGTH;
            uint8_t *ld = linked_map + offset + lo;
            if (lo + 12 > sz || ld[0] != 0x67 || (ld[1] & 0x0f) != 0x0e)
               goto invalid_start_marker;
            ld[1] &= ~0x08;
         }
      }
      offset += sz;
   }

   if (nr_samples_shaded > 1) {
      assert(sample_loop_footer[SAMPLE_LOOP_FOOTER_COUNT_PATCH_OFFS] == 0);

      /* Make a stack copy of the footer so we can efficiently patch it */
      uint8_t footer[sizeof(sample_loop_footer)];
      memcpy(footer, sample_loop_footer, sizeof(footer));

      /* Patch in sample end */
      uint8_t end = (1u << nr_samples_shaded) - 1;
      footer[SAMPLE_LOOP_FOOTER_COUNT_PATCH_OFFS] =
         end << SAMPLE_LOOP_FOOTER_COUNT_SHIFT;

      /* Patch in the branch target */
      int32_t loop_size = offset - sample_loop_begin;
      int32_t branch_offs = -(SAMPLE_LOOP_FOOTER_JMP_OFFS + loop_size);
      int32_t *target = (int32_t *)(footer + SAMPLE_LOOP_FOOTER_JMP_PATCH_OFFS);
      *target = branch_offs;

      /* Copy in the patched footer */
      memcpy(linked_map + offset, footer, sizeof(footer));
      offset += sizeof(footer);
   } else if (nr_samples_shaded) {
      /* Just end after the first sample, no need to loop for a single sample */
      memcpy(linked_map + offset, stop, sizeof(stop));
      offset += sizeof(stop);
   }

   assert(offset == size);

   linked->scratch_size = scratch_size;
   linked->agx3_spectpr_scratch_size = agx3_spectpr_scratch_size;
   linked->nr_gprs = nr_gprs;
   linked->g17p_colour_classes = colour_classes;

   if (dev->chip == AGX_CHIP_G18G && getenv("ASAHI_G18_RESOURCE_TRACE") &&
       (scratch_size || agx3_spectpr_scratch_size) && size <= (SIZE_MAX - 1) / 2) {
      /* Once per linked executable, rather than once per dispatch. Preserve
       * the final bytes, including prolog/epilog and link-time patches. */
      char *hex = malloc(size * 2 + 1);
      if (hex) {
         static const char digits[] = "0123456789abcdef";
         blake3_hash digest;
         char hash[BLAKE3_HEX_LEN];
         _mesa_blake3_compute(linked_map, size, digest);
         _mesa_blake3_format(hash, digest);
         for (size_t i = 0; i < size; ++i) {
            hex[2 * i] = digits[linked_map[i] >> 4];
            hex[2 * i + 1] = digits[linked_map[i] & 15];
         }
         hex[size * 2] = 0;
         fprintf(stderr,
                 "G18_LINKED_SHADER size=%zu scratch=%u spectpr=%u blake3=%s hex=%s\n",
                 size, scratch_size, agx3_spectpr_scratch_size, hash, hex);
         free(hex);
      }
   }

   if (agx_bo_sync_for_device(linked->bo, 0, alloc_size)) {
      agx_bo_unreference(dev, linked->bo);
      linked->bo = NULL;
      return false;
   }

   agx_pack(&linked->shader, USC_SHADER, cfg) {
      cfg.code = agx_usc_addr(dev, linked->bo->va->addr + linked->entry_offset);
      cfg.unk_2 = fragment ? 2 : 3;

      if (fragment)
         cfg.loads_varyings = linked->cf.nr_bindings > 0;
   }

   agx_pack(&linked->regs, USC_REGISTERS, cfg) {
      cfg.register_count = nr_gprs;
      cfg.unk_1 = fragment;
      cfg.spill_size = scratch_size ? agx_scratch_get_bucket(scratch_size) : 0;
      cfg.unk_4 = 1;
   }

   if (fragment) {
      agx_pack(&linked->fragment_props, USC_FRAGMENT_PROPERTIES, cfg) {
         cfg.early_z_testing = !writes_sample_mask;
         cfg.unk_2 = true;
         cfg.unk_3 = 0xf;
         cfg.unk_4 = 0x2;
         cfg.unk_5 = 0x0;
      }

      /* If we skip the fragment shader, we can optimize the pass type */
      if (tag_write_disable) {
         reads_tib = false;
         writes_sample_mask = false;
         linked->no_op = true;
      }

      agx_pack(&linked->fragment_control, FRAGMENT_CONTROL, cfg) {
         cfg.tag_write_disable = tag_write_disable;
         cfg.disable_tri_merging = disable_tri_merging;

         if (reads_tib && writes_sample_mask)
            cfg.pass_type = AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH;
         else if (reads_tib)
            cfg.pass_type = AGX_PASS_TYPE_TRANSLUCENT;
         else if (writes_sample_mask)
            cfg.pass_type = AGX_PASS_TYPE_PUNCH_THROUGH;
         else
            cfg.pass_type = AGX_PASS_TYPE_OPAQUE;
      }

      /* Merge the CF binding lists from the prolog to handle cull distance */
      memcpy(&linked->cf, &main->info.varyings.fs,
             sizeof(struct agx_varyings_fs));

      struct agx_varyings_fs *prolog_vary =
         prolog ? &prolog->info.varyings.fs : NULL;

      if (prolog_vary && prolog_vary->nr_bindings) {
         assert(!prolog_vary->reads_z);
         linked->cf.nr_cf = MAX2(linked->cf.nr_cf, prolog_vary->nr_cf);

         assert(linked->cf.nr_bindings + prolog_vary->nr_bindings <=
                   ARRAY_SIZE(linked->cf.bindings) &&
                "bounded by # of coeff registers");

         memcpy(linked->cf.bindings + linked->cf.nr_bindings,
                prolog_vary->bindings,
                sizeof(struct agx_cf_binding) * prolog_vary->nr_bindings);

         linked->cf.nr_bindings += prolog_vary->nr_bindings;
      }

      agx_pack(&linked->osel, OUTPUT_SELECT, cfg) {
         cfg.varyings = linked->cf.nr_bindings > 0;
         cfg.frag_coord_z = linked->cf.reads_z;
      }
   }

   return true;

invalid_start_marker:
   fprintf(stderr, "agx_fast_link: invalid G17P part-start FLTNORM marker\n");
   agx_bo_unreference(dev, linked->bo);
   linked->bo = NULL;
   return false;
}
