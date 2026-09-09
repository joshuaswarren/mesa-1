/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi_g17p_stream_validator.h"

#include "asahi/lib/agx_ppp_hal200_graphics_core.h"
#include "asahi/lib/agx_ioto_core.h"
#include "asahi/lib/agx_ioto_program_core.h"
#include "asahi/lib/agx_ppp_viewport_payload_core.h"
#include "asahi/lib/agx_ppp_w_clamp_cull2_core.h"
#include "asahi/lib/agx_va_runtime_hole.h"
#include "asahi/lib/agx3_cdm_decode_core.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx_vdm_decode_core.h"
#include "asahi/lib/agx_vdm_hal200_bind_tail_core.h"
#include "asahi/isa/disasm.h"
#include "drm-uapi/asahi_drm.h"

#include <stdio.h>
#include <string.h>

#define G17P_CONTEXT_BASE UINT64_C(0x1000000000)
#define G17P_USC_BASE UINT64_C(0x10000000000)
#define MAX_RECORDS 65536
#define MAX_IOTO_PROGRAM_SIZE 6144
#define G17P_CDM_DIRECT_LAUNCH_SIZE 40

static uint32_t
load_le32(const uint8_t *p)
{
   return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
          ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t
load_le16(const uint8_t *p)
{
   return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint64_t
compact_address(uint64_t address)
{
   return address < G17P_CONTEXT_BASE ? G17P_CONTEXT_BASE + address : address;
}

static uint64_t
decode_pipeline_pair(const uint8_t pair[8])
{
   const uint32_t high = load_le32(pair);
   const uint32_t low = load_le32(pair + 4);

   if (high & UINT32_C(0x003fffff))
      return 0;

   return ((uint64_t)(high >> 22) << 38) | ((uint64_t)low << 6);
}

static uint64_t
load_le64(const uint8_t *p)
{
   return (uint64_t)load_le32(p) | ((uint64_t)load_le32(p + 4) << 32);
}

static bool
fail_at(uint64_t address, const char *reason)
{
   fprintf(stderr, "shim: invalid G17P command stream at %#llx: %s\n",
           (unsigned long long)address, reason);
   return false;
}

static bool
read_owned(const struct asahi_g17p_stream_memory *memory, uint64_t address,
           void *dst, size_t size)
{
   return memory->owned(memory->data, address, size) &&
          memory->read(memory->data, address, dst, size);
}

static bool
validate_shader_blob(const struct asahi_g17p_stream_memory *memory,
                     uint64_t address)
{
   size_t available;
   if ((address & 1) || !memory->span ||
       !memory->span(memory->data, address, &available) || available < 4)
      return fail_at(address, "shader entry is unaligned or unowned");

   /* LDSHDR names +0x3c0 inside Apple's executable container. The dword at
    * entry-0x80 is the exact 0x80 framing plus 64-byte-padded main span; use
    * it instead of the page-rounded BO mapping as the executable bound. */
   uint8_t container_head[4], block_head[4];
   if (address < 0x3c0 ||
       !read_owned(memory, address - 0x3c0, container_head,
                   sizeof(container_head)) ||
       !read_owned(memory, address - 0x80, block_head, sizeof(block_head)) ||
       load_le32(container_head) != 0x340)
      return fail_at(address, "shader does not own a G17P code container");
   const uint32_t block_size = load_le32(block_head);
   if (block_size < 0x80 || (block_size & 63) || block_size - 0x80 > available)
      return fail_at(address, "shader container has an invalid code span");
   available = block_size - 0x80;

   FILE *sink = memory->verbose ? stderr : fopen("/dev/null", "w");
   if (!sink)
      return fail_at(address, "could not open disassembly sink");

   size_t offset = 0;
   bool valid = false;
   while (offset < available && offset < 16384) {
      uint8_t bytes[64] = {0};
      const size_t readable =
         available - offset < sizeof(bytes) ? available - offset
                                            : sizeof(bytes);
      if (!memory->read(memory->data, address + offset, bytes, readable) ||
          readable < 2)
         break;

      const signed decoded = agx3b_disassemble_instr(
         (BITSET_WORD *)bytes, sink, (unsigned)offset, readable, false);
      const size_t length = decoded < 0 ? (size_t)-decoded : (size_t)decoded;
      /* The generated decoder reports a negative length for an instruction
       * carrying not-yet-named bits, while still giving its authoritative
       * byte length. That is usable for a bounded walk; zero is not. */
      if (decoded == 0 || !length || length > readable)
         break;

      if (length == 4 && load_le32(bytes) == UINT32_C(0x0000000e)) {
         valid = true;
         offset += length;
         break;
      }
      offset += length;
   }

   if (!memory->verbose)
      fclose(sink);
   if (!valid)
      return fail_at(address, "shader has no decodable terminal end");
   for (; offset < available; ++offset) {
      uint8_t byte;
      if (!memory->read(memory->data, address + offset, &byte, 1) || byte)
         return fail_at(address + offset,
                        "shader container has nonzero trailing code");
   }
   return true;
}

static bool
validate_ioto_program(const struct asahi_g17p_stream_memory *memory,
                      uint64_t address, bool fragment, bool nested_tables,
                      struct asahi_g17p_stream_stats *stats)
{
   size_t available;
   if ((address & 63) || !memory->span ||
       !memory->span(memory->data, address, &available) || available < 4)
      return fail_at(address, "Ioto program is unaligned or unowned");

   available = MIN2(available, MAX_IOTO_PROGRAM_SIZE);
   uint64_t cursor = address;
   bool seen_speclm = false, seen_ldshdr = false, seen_spectpr = false;
   bool seen_pixrel = false;
   uint32_t immediates[128] = {0};
   bool immediate_valid[128] = {0};
   for (unsigned i = 0; i < 2048 && cursor - address < available; ++i) {
      uint8_t bytes[AGX_IOTO_MAX_RECORD_SIZE] = {0};
      const size_t remain = available - (cursor - address);
      const size_t readable = MIN2(remain, sizeof(bytes));
      if (!memory->read(memory->data, cursor, bytes, readable))
         return fail_at(cursor, "Ioto record is unreadable");

      struct agx_ioto_program_record record;
      size_t consumed;
      if (!agx_ioto_decode_program_record(AGX_CHIP_G17P, bytes, readable,
                                           &record, &consumed) ||
          !consumed || consumed > readable) {
         fprintf(stderr, "shim: Ioto bytes");
         for (size_t j = 0; j < readable; ++j)
            fprintf(stderr, " %02x", bytes[j]);
         fprintf(stderr, "\n");
         return fail_at(cursor, "unknown or truncated Ioto record");
      }

      if (record.kind == AGX_IOTO_PROGRAM_RECORD_TERMINATE) {
         if (!seen_speclm || !seen_ldshdr || !seen_spectpr ||
             fragment != seen_pixrel)
            return fail_at(cursor, "incomplete Ioto graphics state tail");
         return true;
      }

      if (record.kind == AGX_IOTO_PROGRAM_RECORD_IOTO) {
         switch (record.ioto.kind) {
         case AGX_IOTO_KIND_LDIMM_EP0:
         case AGX_IOTO_KIND_LDIMM_EP1: {
            const uint16_t w0 = load_le16(bytes);
            unsigned dest;
            uint32_t value;
            if (record.ioto.kind == AGX_IOTO_KIND_LDIMM_EP0) {
               dest = (w0 >> 4) & 0xf;
               value = bytes[1];
            } else {
               const uint16_t w1 = load_le16(bytes + 2);
               const uint16_t w2 = load_le16(bytes + 4);
               const uint16_t w3 = load_le16(bytes + 6);
               dest = ((w0 >> 4) & 0xf) | (((w1 >> 6) & 3) << 4) |
                      (((w1 >> 3) & 1) << 6);
               value = ((uint32_t)(w0 >> 8) & 0x7f) |
                       (((uint32_t)w2 >> 1) & 0xf) << 7 |
                       (((uint32_t)w2 >> 10) & 3) << 11 |
                       ((uint32_t)w3 & 0xf) << 13 |
                       (((uint32_t)w3 >> 4) & 0xff) << 17 |
                       (((uint32_t)w1 >> 9) & 0x7f) << 25;
            }
            if (dest >= ARRAY_SIZE(immediates))
               return fail_at(cursor, "Ioto LDIMM destination is invalid");
            immediates[dest] = value;
            immediate_valid[dest] = true;

            /* The bounded absolute-load builder always materializes a 64-bit
             * source as consecutive low/high dwords. Validate it as soon as
             * the high half arrives, independently of the producer's range
             * list. A non-fragment EOT program is the one no-nested-table
             * form: dwords 2/3 hold its PBE argument address, while 4/5 are
             * the two MRT imageblock offsets and must not be mistaken for an
             * address. */
            if (dest && immediate_valid[dest - 1]) {
               const uint64_t source =
                  immediates[dest - 1] | ((uint64_t)value << 32);
               const bool absolute_source = fragment || nested_tables ||
                                            dest == 3;
               if (absolute_source && source > UINT32_MAX) {
                  if (!memory->owned(memory->data, source, 4))
                     return fail_at(cursor, "Ioto load source is unowned");

                  uint8_t nested[16];
                  if (nested_tables &&
                      memory->read(memory->data, source, nested,
                                   sizeof(nested))) {
                     const uint64_t a = load_le64(nested);
                     const uint64_t b = load_le64(nested + 8);
                     if (a > UINT32_MAX && b > UINT32_MAX &&
                         a < (UINT64_C(1) << 48) &&
                         b < (UINT64_C(1) << 48) &&
                         (!memory->owned(memory->data, a, 32) ||
                          !memory->owned(memory->data, b, 32)))
                        return fail_at(cursor,
                                       "Ioto nested resource pointer is unowned");
                     if (memory->verbose && a > UINT32_MAX &&
                         b > UINT32_MAX && a < (UINT64_C(1) << 48) &&
                         b < (UINT64_C(1) << 48)) {
                        uint8_t texture[32] = {0}, sampler[32] = {0};
                        if (memory->read(memory->data, a, texture,
                                         sizeof(texture)) &&
                            memory->read(memory->data, b, sampler,
                                         sizeof(sampler))) {
                           fprintf(stderr,
                                   "shim: G17P resource pair texture=%#llx "
                                   "sampler=%#llx texture-bytes=",
                                   (unsigned long long)a,
                                   (unsigned long long)b);
                           for (unsigned j = 0; j < sizeof(texture); ++j)
                              fprintf(stderr, "%02x", texture[j]);
                           fprintf(stderr, " sampler-bytes=");
                           for (unsigned j = 0; j < sizeof(sampler); ++j)
                              fprintf(stderr, "%02x", sampler[j]);
                           fprintf(stderr, "\n");
                        }
                     }
                  }
               }
            }
            break;
         }
         case AGX_IOTO_KIND_SPECLM_EP0:
            if (seen_speclm || seen_ldshdr || seen_spectpr)
               return fail_at(cursor, "misordered Ioto SPECLM");
            seen_speclm = true;
            break;
         case AGX_IOTO_KIND_LDSHDR_EP0: {
            if (!seen_speclm || seen_ldshdr || seen_spectpr || consumed != 10)
               return fail_at(cursor, "misordered Ioto LDSHDR");
            const uint16_t w1 = load_le16(bytes + 2);
            const uint16_t w2 = load_le16(bytes + 4);
            const uint16_t w3 = load_le16(bytes + 6);
            const uint16_t w4 = load_le16(bytes + 8);
            const uint64_t shader =
               ((uint64_t)((w1 >> 7) & 0x1ff) << 6) |
               ((uint64_t)w2 << 15) | ((uint64_t)w3 << 31) |
               ((uint64_t)(w4 & 1) << 47);
            if ((shader & 63) || !validate_shader_blob(memory, shader))
               return fail_at(cursor, "Ioto LDSHDR shader is not owned");
            if (fragment)
               stats->fragment_code = shader;
            seen_ldshdr = true;
            break;
         }
         case AGX_IOTO_KIND_SPECTPR_EP0:
            if (!seen_ldshdr || seen_spectpr)
               return fail_at(cursor, "misordered Ioto SPECTPR");
            seen_spectpr = true;
            break;
         case AGX_IOTO_KIND_PIXREL_EP0:
            if (!fragment || !seen_spectpr || seen_pixrel)
               return fail_at(cursor, "invalid Ioto PIXREL");
            seen_pixrel = true;
            break;
         default:
            if (seen_speclm && !seen_spectpr)
               return fail_at(cursor, "foreign record inside Ioto state tail");
            break;
         }
      }
      cursor += consumed;
   }

   return fail_at(cursor, "Ioto program has no bounded terminator");
}

static bool
validate_compact_compute_helper(
   const struct asahi_g17p_stream_memory *memory, uint64_t address,
   const uint8_t bytes[AGX3_CDM_INDIRECT_LAUNCH_SIZE])
{
   struct agx3_cdm_indirect_launch_info helper;
   if (!agx3_cdm_indirect_launch_unpack(
          AGX_CHIP_G17P, bytes, AGX3_CDM_INDIRECT_LAUNCH_SIZE, &helper) ||
       helper.word0_payload != UINT32_C(0x10080000) ||
       helper.word1_payload != 0 || !helper.pipeline ||
       (helper.pipeline & 63) ||
       helper.opaque_word3 !=
          (UINT32_C(0x40000000) | ((helper.pipeline >> 40) & 0xff)) ||
       !helper.indirect_parameters || (helper.indirect_parameters & 3) ||
       !memory->owned(memory->data, helper.indirect_parameters, 6 * 4))
      return fail_at(address, "invalid compact indirect compute helper");

   struct asahi_g17p_stream_stats shader_stats = {0};
   if (!validate_ioto_program(memory, helper.pipeline, false, true,
                              &shader_stats))
      return fail_at(address, "compact compute helper Ioto program is invalid");

   return true;
}

bool
asahi_g17p_compute_validate(
   const struct asahi_g17p_stream_memory *memory,
   const struct drm_asahi_cmd_compute *compute,
   struct asahi_g17p_compute_stats *stats)
{
   struct asahi_g17p_compute_stats local = {0};
   bool need_barrier = false;
   bool expect_compact_helper = false;
   bool followed_link = false;

   if (stats)
      memset(stats, 0, sizeof(*stats));
   if (!memory || !memory->read || !memory->owned || !memory->span ||
       !compute || compute->flags || !compute->cdm_ctrl_stream_base ||
       compute->cdm_ctrl_stream_end <= compute->cdm_ctrl_stream_base ||
       compute->cdm_ctrl_stream_end - compute->cdm_ctrl_stream_base >
          SIZE_MAX ||
       !memory->owned(memory->data, compute->cdm_ctrl_stream_base,
                      (size_t)(compute->cdm_ctrl_stream_end -
                               compute->cdm_ctrl_stream_base)))
      return fail_at(compute ? compute->cdm_ctrl_stream_base : 0,
                     "compute stream bounds are invalid");

   uint64_t cursor = compute->cdm_ctrl_stream_base;
   for (unsigned iteration = 0; iteration < MAX_RECORDS; ++iteration) {
      size_t span;
      if (!memory->span(memory->data, cursor, &span) || span < 4)
         return fail_at(cursor, "compute record is not owned");

      if (!followed_link && cursor >= compute->cdm_ctrl_stream_end)
         return fail_at(cursor, "compute stream ended without terminate");

      const size_t available = followed_link
                                  ? span
                                  : MIN2(span, (size_t)(
                                                  compute->cdm_ctrl_stream_end -
                                                  cursor));
      uint8_t bytes[G17P_CDM_DIRECT_LAUNCH_SIZE] = {0};
      const size_t readable = MIN2(available, sizeof(bytes));
      if (!read_owned(memory, cursor, bytes, readable))
         return fail_at(cursor, "compute record is not owned");
      if (memory->verbose) {
         fprintf(stderr, "shim: G17P CDM %#llx:",
                 (unsigned long long)cursor);
         for (size_t i = 0; i < readable; ++i)
            fprintf(stderr, " %02x", bytes[i]);
         fprintf(stderr, "\n");
      }

      /* Native indirect dispatches are a direct-shaped 40-byte main launch,
       * a barrier, then an explicitly delimited 24-byte compact helper and
       * its barrier.  Type zero alone cannot distinguish the helper from the
       * 36/40-byte launch families, so decode it only in that exact state. */
      if (expect_compact_helper && !need_barrier) {
         if (available < AGX3_CDM_INDIRECT_LAUNCH_SIZE ||
             !validate_compact_compute_helper(memory, cursor, bytes))
            return false;

         local.launches++;
         need_barrier = true;
         expect_compact_helper = false;
         cursor += AGX3_CDM_INDIRECT_LAUNCH_SIZE;
         continue;
      }

      struct agx3_cdm_decoded_record record;
      if (!agx3_cdm_decode_one(AGX_CHIP_G17P, bytes, readable, &record) ||
          !record.length || record.length > available)
         return fail_at(cursor, "unknown or truncated AGX3 compute record");

      switch (record.kind) {
      case AGX3_CDM_RECORD_LAUNCH: {
         if (need_barrier || (record.data.launch.pipeline & 63) ||
             !record.data.launch.pipeline ||
             record.length != G17P_CDM_DIRECT_LAUNCH_SIZE ||
             record.data.launch.opaque_word3 !=
                (UINT32_C(0x40000000) |
                 ((record.data.launch.pipeline >> 40) & 0xff)))
            return fail_at(cursor, "compute launch ordering is invalid");

         for (unsigned axis = 0; axis < 3; ++axis) {
            if (!record.data.launch.global_size[axis] ||
                !record.data.launch.local_size[axis])
               return fail_at(cursor, "compute launch has a zero dimension");
         }

         struct asahi_g17p_stream_stats shader_stats = {0};
         if (!validate_ioto_program(memory, record.data.launch.pipeline,
                                    false, true, &shader_stats))
            return fail_at(cursor, "compute launch Ioto program is invalid");

         local.launches++;
         need_barrier = true;
         expect_compact_helper =
            (record.data.launch.word0_payload & UINT32_C(0x00800000)) != 0;
         break;
      }

      case AGX3_CDM_RECORD_BARRIER:
         if (record.data.barrier.flags != 0 &&
             record.data.barrier.flags != 0x5)
            return fail_at(cursor, "compute barrier is not canonical");

         if (!need_barrier) {
            /* Immediate post-graphics writes begin with the strong cache
             * flush which makes the preceding render visible.  It is an
             * ordered stream prologue, not a launch terminator. */
            if (expect_compact_helper || local.launches != 0 ||
                local.prologue_barriers != 0 ||
                record.data.barrier.flags != 0x5)
               return fail_at(cursor, "compute barrier is not canonical");

            local.prologue_barriers++;
            local.strong_barriers++;
            break;
         }

         local.barriers++;
         local.strong_barriers += record.data.barrier.flags == 0x5;
         need_barrier = false;
         break;

      case AGX3_CDM_RECORD_STREAM_TERMINATE:
         if (need_barrier || expect_compact_helper || !local.launches ||
             (!followed_link &&
              cursor + record.length != compute->cdm_ctrl_stream_end))
            return fail_at(cursor, "compute terminator is not final");
         local.terminated = true;
         if (stats)
            *stats = local;
         return true;

      case AGX3_CDM_RECORD_STREAM_LINK: {
         const uint64_t target = record.data.stream_link.target;
         if (need_barrier || expect_compact_helper ||
             record.data.stream_link.with_return || (target & 3) ||
             !memory->owned(memory->data, target, 4))
            return fail_at(cursor, "compute stream link is invalid");

         cursor = target;
         followed_link = true;
         continue;
      }

      case AGX3_CDM_RECORD_LINK_PADDING:
      case AGX3_CDM_RECORD_SUBSTREAM_BARRIER:
      case AGX3_CDM_RECORD_SECONDARY_BARRIER:
      case AGX3_CDM_RECORD_UNKNOWN:
      default:
         return fail_at(cursor, "compute record is outside native Gallium");
      }

      cursor += record.length;
   }

   return fail_at(cursor, "compute record limit reached before terminate");
}

static uint64_t
render_resource_spec(const struct drm_asahi_bg_eot *program, uint32_t high)
{
   return program->rsrc_spec | ((uint64_t)high << 32);
}

static bool
validate_render_aux_program(const struct asahi_g17p_stream_memory *memory,
                            const struct drm_asahi_bg_eot *program,
                            const char *name, bool fragment,
                            bool nested_tables)
{
   if ((program->usc & 0xf) != 4) {
      fprintf(stderr, "shim: G17P %s USC tag is not 4\n", name);
      return false;
   }

   const uint64_t address = G17P_USC_BASE | (program->usc & ~UINT32_C(0xf));
   struct asahi_g17p_stream_stats stats = {0};
   if (!validate_ioto_program(memory, address, fragment, nested_tables,
                              &stats)) {
      fprintf(stderr, "shim: G17P %s Ioto program %#llx is invalid\n", name,
              (unsigned long long)address);
      return false;
   }

   return true;
}

bool
asahi_g17p_render_aux_validate(
   const struct asahi_g17p_stream_memory *memory,
   const struct drm_asahi_cmd_render *render)
{
   if (!memory || !render)
      return false;

   const uint64_t bg =
      render_resource_spec(&render->bg, render->bg_rsrc_spec_hi);
   const uint64_t partial_bg = render_resource_spec(
      &render->partial_bg, render->bg_partial_rsrc_spec_hi);
   const uint64_t eot =
      render_resource_spec(&render->eot, render->bg_eot_rsrc_spec_hi);
   const uint64_t partial_eot = render_resource_spec(
      &render->partial_eot, render->bg_eot_partial_rsrc_spec_hi);
   const uint32_t high = render->bg_rsrc_spec_hi |
                         render->bg_partial_rsrc_spec_hi |
                         render->bg_eot_rsrc_spec_hi |
                         render->bg_eot_partial_rsrc_spec_hi;
   if (!!high != !!(render->flags & DRM_ASAHI_RENDER_RSRC_SPEC_HI))
      return fail_at(0, "render resource-spec high flag disagrees with payload");

   const uint64_t bg_resources =
      AGX_IOTO_G17P_BG_RESOURCE_SPEC & ((UINT64_C(1) << 47) - 1);
   const uint64_t bg_writes = bg_resources | (UINT64_C(0x1ffff) << 47);
   if ((bg != bg_resources && bg != bg_writes) ||
       (partial_bg != bg_resources && partial_bg != bg_writes) ||
       (eot != AGX_IOTO_G17P_EOT_RESOURCE_SPEC && eot != bg_resources) ||
       (partial_eot != AGX_IOTO_G17P_EOT_RESOURCE_SPEC &&
        partial_eot != bg_resources))
      return fail_at(0, "render auxiliary resource spec is not canonical");

   return validate_render_aux_program(memory, &render->bg, "background",
                                      true, true) &&
          validate_render_aux_program(memory, &render->partial_bg,
                                      "partial background", true, true) &&
          validate_render_aux_program(memory, &render->eot, "end-of-tile",
                                      eot != AGX_IOTO_G17P_EOT_RESOURCE_SPEC,
                                      false) &&
          validate_render_aux_program(memory, &render->partial_eot,
                                      "partial end-of-tile",
                                      partial_eot !=
                                         AGX_IOTO_G17P_EOT_RESOURCE_SPEC,
                                      false);
}

static bool
validate_slot(const struct asahi_g17p_stream_memory *memory, unsigned slot,
              uint64_t address, size_t size,
              struct asahi_g17p_stream_stats *stats)
{
   uint8_t bytes[AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH];
   if (size > sizeof(bytes) || !read_owned(memory, address, bytes, size))
      return fail_at(address, "bind slot has no owning VM mapping");

   struct agx_ppp_hal200_graphics_fields graphics;
   struct agx_ppp_viewport_payload_fields viewport;
   struct agx_ppp_w_clamp_cull2_fields w_cull2;

   switch (slot) {
   case 0: { /* Varyings */
      if (size != 20 || load_le32(bytes) != UINT32_C(0x10040000))
         return fail_at(address, "invalid 20-byte varyings slot");
      if (((load_le32(bytes + 4) | load_le32(bytes + 8)) &
           UINT32_C(0xff000000)) || load_le32(bytes + 12) ||
          load_le32(bytes + 16))
         return fail_at(address, "varyings slot has reserved bits");
      return true;
   }
   case 1: { /* Fragment program and VIT */
      if (size != 28 || load_le32(bytes) != AGX_PPP_HAL200_FS_HEADER ||
          !agx_ppp_hal200_graphics_core_unpack(
             AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_FS, bytes, size,
             &graphics))
         return fail_at(address, "invalid 28-byte fragment slot");

      const uint64_t vit =
         ((uint64_t)(load_le32(bytes + 4) & UINT32_C(0xffff)) << 32) |
         load_le32(bytes + 8);
      const uint64_t program = decode_pipeline_pair(bytes + 16);
      if (memory->verbose)
         fprintf(stderr,
                 "shim: G17P FS slot=%#llx VIT=%#llx program=%#llx\n",
                 (unsigned long long)address, (unsigned long long)vit,
                 (unsigned long long)program);
      if ((vit & 63) || (program & 63) || load_le32(bytes + 12) ||
          load_le32(bytes + 24) ||
          (vit && !memory->owned(memory->data, vit, 4)) || !program ||
          !validate_ioto_program(memory, program, true, true, stats)) {
         fprintf(stderr,
                 "shim: G17P fragment slot %#llx has unowned VIT %#llx or "
                 "program %#llx\n",
                 (unsigned long long)address, (unsigned long long)vit,
                 (unsigned long long)program);
         return false;
      }
      stats->varying_fs |= vit != 0;
      stats->no_varying_fs |= vit == 0;
      return true;
   }
   case 2: {
      if (size != 28 ||
          !agx_ppp_hal200_graphics_core_unpack(
             AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_OUTPUT, bytes, size,
             &graphics) ||
          (graphics.words[0] & ~UINT32_C(0x003d0000)) ||
          !!(graphics.words[0] & UINT32_C(0x00080000)) !=
             !!(graphics.words[0] & UINT32_C(0x00100000)) ||
          graphics.words[1] ||
          graphics.words[2] > 15 ||
          (graphics.words[3] & UINT32_C(0xff00ff00)) ||
          graphics.words[4] != AGX_PPP_HAL200_PRIMARY_FACE_HEADER ||
          (graphics.words[5] & UINT32_C(0x00040200)) !=
             UINT32_C(0x00040200) ||
          (graphics.words[5] & ~UINT32_C(0x040fc200)))
         return fail_at(address, "output slot has non-canonical semantics");
      return true;
   }
   case 3: {
      if (size != 28 ||
          !agx_ppp_hal200_graphics_core_unpack(
             AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE, bytes,
             size, &graphics))
         return fail_at(address, "invalid primary-face slot");
      stats->depth_enabled |= ((graphics.words[1] >> 24) & 7) != 7 ||
                              !(graphics.words[1] & (UINT32_C(1) << 21));
      stats->stencil_enabled |= graphics.words[2] != UINT32_C(0x0e000000) ||
                                graphics.words[4] != UINT32_C(0x0e000000);
      stats->primary_tri_merge_disable |=
         (graphics.words[0] & (UINT32_C(1) << 26)) != 0;
      return true;
   }
   case 4: {
      if (size != 20 ||
          !agx_ppp_hal200_graphics_core_unpack(
             AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE, bytes,
             size, &graphics) || graphics.words[1] != graphics.words[2] ||
          graphics.words[3] != UINT32_C(0x0001ffff) ||
          (graphics.words[0] & ~UINT32_C(0x64200200)) ||
          (graphics.words[0] & UINT32_C(0x200)) == 0)
         return fail_at(address, "secondary-face slot is non-canonical");
      stats->blend_enabled |= (graphics.words[0] >> 29) != 0;
      stats->rasterizer_discard |=
         (graphics.words[0] & (UINT32_C(1) << 21)) != 0;
      stats->secondary_tri_merge_disable |=
         (graphics.words[0] & (UINT32_C(1) << 26)) != 0;
      return true;
   }
   case 5:
      if (!agx_ppp_viewport_payload_core_unpack(
             AGX_CHIP_G17P, bytes, size, &viewport))
         return fail_at(address, "invalid viewport slot");
      stats->viewport_count = viewport.viewport_count;
      if (memory->verbose && viewport.viewport_count > 1) {
         fprintf(stderr,
                 "shim: viewport[1] tx=%#x sx=%#x ty=%#x sy=%#x\n",
                 viewport.viewports[1].translate_x,
                 viewport.viewports[1].scale_x,
                 viewport.viewports[1].translate_y,
                 viewport.viewports[1].scale_y);
      }
      stats->multi_viewport |=
         viewport.viewport_count == 16 &&
         viewport.viewports[1].translate_x == UINT32_C(0x41b80000) &&
         viewport.viewports[1].scale_x == UINT32_C(0x41200000) &&
         viewport.viewports[1].translate_y == UINT32_C(0x42180000) &&
         viewport.viewports[1].scale_y == UINT32_C(0xc1100000);
      stats->two_viewport |=
         viewport.viewport_count == 2 &&
         viewport.viewports[0].translate_x == UINT32_C(0x41000000) &&
         viewport.viewports[0].scale_x == UINT32_C(0x41000000) &&
         viewport.viewports[0].translate_y == UINT32_C(0x41800000) &&
         viewport.viewports[0].scale_y == UINT32_C(0x41800000) &&
         viewport.viewports[1].translate_x == UINT32_C(0x41c00000) &&
         viewport.viewports[1].scale_x == UINT32_C(0x41000000) &&
         viewport.viewports[1].translate_y == UINT32_C(0x41800000) &&
         viewport.viewports[1].scale_y == UINT32_C(0x41800000);
      stats->four_viewport |=
         viewport.viewport_count == 4 &&
         viewport.viewports[0].translate_x == UINT32_C(0x41000000) &&
         viewport.viewports[0].scale_x == UINT32_C(0x41000000) &&
         viewport.viewports[0].translate_y == UINT32_C(0x41000000) &&
         viewport.viewports[0].scale_y == UINT32_C(0x41000000) &&
         viewport.viewports[1].translate_x == UINT32_C(0x41c00000) &&
         viewport.viewports[1].scale_x == UINT32_C(0x41000000) &&
         viewport.viewports[1].translate_y == UINT32_C(0x41000000) &&
         viewport.viewports[1].scale_y == UINT32_C(0x41000000) &&
         viewport.viewports[2].translate_x == UINT32_C(0x41000000) &&
         viewport.viewports[2].scale_x == UINT32_C(0x41000000) &&
         viewport.viewports[2].translate_y == UINT32_C(0x41c00000) &&
         viewport.viewports[2].scale_y == UINT32_C(0x41000000) &&
         viewport.viewports[3].translate_x == UINT32_C(0x41c00000) &&
         viewport.viewports[3].scale_x == UINT32_C(0x41000000) &&
         viewport.viewports[3].translate_y == UINT32_C(0x41c00000) &&
         viewport.viewports[3].scale_y == UINT32_C(0x41000000);
      stats->viewport_changed |=
         viewport.viewports[0].translate_x == UINT32_C(0x41d80000) &&
         viewport.viewports[0].scale_x == UINT32_C(0x41c00000) &&
         viewport.viewports[0].translate_y == UINT32_C(0x41c80000) &&
         viewport.viewports[0].scale_y == UINT32_C(0x41a00000);
      return true;
   case 6:
      return size == 12 && agx_ppp_w_clamp_cull2_core_unpack(
                              AGX_CHIP_G17P, bytes, size, &w_cull2);
   case 7:
      if (size != 8 ||
          !agx_ppp_hal200_graphics_core_unpack(
             AGX_CHIP_G17P, AGX_PPP_HAL200_GRAPHICS_CULL, bytes, size,
             &graphics) || (graphics.words[0] & ~UINT32_C(0x00030d83)))
         return fail_at(address, "cull slot has reserved bits");
      stats->cull_enabled |= (graphics.words[0] & 3) != 0;
      if (!!(graphics.words[0] & (UINT32_C(1) << 17)) !=
          stats->rasterizer_discard)
         return fail_at(address, "rasterizer-discard records disagree");
      return true;
   default:
      return false;
   }
}

static bool
validate_bind_tail(const struct asahi_g17p_stream_memory *memory,
                   uint64_t address,
                   struct asahi_g17p_stream_stats *stats)
{
   static const uint32_t controls[8] = {
      0x700, 0x500, 0x700, 0x500, 0xa00, 0x300, 0x200, 0x200,
   };
   size_t lengths[8] = {20, 28, 28, 28, 20, 0, 12, 8};
   uint8_t bytes[AGX_VDM_HAL200_BIND_TAIL_LENGTH];
   struct agx_vdm_hal200_bind_pair decoded[8];

   if (!read_owned(memory, address, bytes, sizeof(bytes)) ||
       load_le32(bytes) != UINT32_C(0x00000500) ||
       !agx_vdm_hal200_bind_tail_unpack(AGX_CHIP_G17P, bytes, sizeof(bytes),
                                        decoded))
      return fail_at(address, "invalid 68-byte HAL200 bind tail");

   /* The complete viewport payload is immediately followed by the varying
    * record. Their independently addressed bind slots therefore make the
    * variable 1..16 viewport length self-describing. */
   if (decoded[0].context_offset <= decoded[5].context_offset)
      return fail_at(address, "viewport/varying slots are not ordered");
   lengths[5] = decoded[0].context_offset - decoded[5].context_offset;
   if (lengths[5] < AGX_PPP_VIEWPORT_PAYLOAD_MIN_LENGTH ||
       lengths[5] > AGX_PPP_VIEWPORT_PAYLOAD_MAX_LENGTH)
      return fail_at(address, "viewport slot length is out of range");
   const unsigned viewport_count = (lengths[5] - 8) / 32;
   uint8_t output_record[AGX_PPP_HAL200_OUTPUT_LENGTH];
   uint8_t primary_record[AGX_PPP_HAL200_PRIMARY_FACE_LENGTH];
   if (!read_owned(memory, compact_address(decoded[2].context_offset),
                   output_record, sizeof(output_record)) ||
       !read_owned(memory, compact_address(decoded[3].context_offset),
                   primary_record, sizeof(primary_record)) ||
       load_le32(output_record + 12) != viewport_count - 1)
      return fail_at(address, "output/viewport counts disagree");

   if (!!(load_le32(output_record + 24) & (UINT32_C(1) << 26)) !=
       !!(load_le32(primary_record + 4) & (UINT32_C(1) << 26)))
      return fail_at(address, "output/primary triangle-merge controls disagree");

   /* OUTPUT bits 19 and 20 are the paired viewport/layer UVS selector.  A
    * multiview stream is not complete merely because it exposes two image
    * layers: the vertex ViewIndex must also reach this native output path. */
   stats->layer_output |=
      (load_le32(output_record + 4) & UINT32_C(0x00180000)) ==
      UINT32_C(0x00180000);

   for (unsigned i = 0; i < 8; ++i) {
      /* Fixed controls and lengths are intentionally independent of the
       * producer's slot construction. */
      if (decoded[i].control != controls[i])
         return fail_at(address + 4 + i * 8, "invalid bind-slot control");

      const uint64_t slot = compact_address(decoded[i].context_offset);
      if (!validate_slot(memory, i, slot, lengths[i], stats))
         return false;
   }

   return true;
}

static bool
validate_initial_ppp(const struct asahi_g17p_stream_memory *memory,
                     uint64_t address, uint32_t size_words)
{
   uint8_t bytes[12];
   struct agx_ppp_w_clamp_cull2_fields state;

   address = compact_address(address);
   if (size_words != 3 || !read_owned(memory, address, bytes, sizeof(bytes)) ||
       !agx_ppp_w_clamp_cull2_core_unpack(AGX_CHIP_G17P, bytes, sizeof(bytes),
                                          &state))
      return fail_at(address, "invalid initial 12-byte PPP state");

   return true;
}

bool
asahi_g17p_stream_validate(const struct asahi_g17p_stream_memory *memory,
                           const struct asahi_g17p_stream_config *config,
                           struct asahi_g17p_stream_stats *stats)
{
   struct asahi_g17p_stream_stats local = {0};
   bool pending_selector = false;
   bool have_state = false;
   bool restart_seen = false;
   const uint64_t root = config ? config->root : 0;
   uint64_t cursor = root;

   if (stats)
      memset(stats, 0, sizeof(*stats));
   if (!config || !memory || !memory->read || !memory->owned ||
       !memory->owned(memory->data, root, 4))
      return fail_at(root, "root is not owned");

   for (unsigned iteration = 0; iteration < MAX_RECORDS; ++iteration) {
      uint8_t bytes[68];
      if (!read_owned(memory, cursor, bytes, 4))
         return fail_at(cursor, "record header is not owned");

      if (pending_selector) {
         const uint64_t selector = compact_address(load_le32(bytes));
         uint8_t indices[8];
         if ((load_le32(bytes) & 3) ||
             !read_owned(memory, selector, indices, sizeof(indices)))
            return fail_at(cursor, "draw-state selector is not owned");
         if (load_le32(indices) != UINT32_C(0x100))
            return fail_at(selector, "draw-state selector flags are invalid");
         if (local.viewport_count == 0 ||
             local.viewport_count > AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS)
            return fail_at(selector, "selector has no bounded viewport state");
         const uint64_t scissor_base =
            config->scissor_base + (uint64_t)load_le16(indices + 4) * 16;
         const uint64_t depth_bias_base =
            config->depth_bias_base + (uint64_t)load_le16(indices + 6) * 12;
         for (unsigned i = 0; i < local.viewport_count; ++i) {
            const uint64_t scissor = scissor_base + (uint64_t)i * 16;
            const uint64_t depth_bias = depth_bias_base + (uint64_t)i * 12;
            uint8_t scissor_record[16], depth_bias_record[12];
            if (!read_owned(memory, scissor, scissor_record,
                            sizeof(scissor_record)) ||
                !read_owned(memory, depth_bias, depth_bias_record,
                            sizeof(depth_bias_record)))
               return fail_at(selector,
                              "selected raster array is not fully owned");
            if (load_le16(scissor_record) < load_le16(scissor_record + 2) ||
                load_le16(scissor_record + 4) < load_le16(scissor_record + 6))
               return fail_at(scissor,
                              "selected scissor extents are inverted");
            if (i == 0) {
               local.scissor_changed |= load_le16(scissor_record) == 30 &&
                                        load_le16(scissor_record + 2) == 7 &&
                                        load_le16(scissor_record + 4) == 28 &&
                                        load_le16(scissor_record + 6) == 9;
            }
         }
         pending_selector = false;
         cursor += 4;
         continue;
      }

      if (!read_owned(memory, cursor, bytes, sizeof(bytes)))
         return fail_at(cursor, "record crosses its owning mapping");

      struct agx_vdm_decoded_record record;
      if (!agx_vdm_decode_one(AGX_CHIP_G17P, bytes, sizeof(bytes), &record)) {
         if (load_le32(bytes) == UINT32_C(0x40000001)) {
            if (restart_seen)
               return fail_at(cursor, "duplicate indexed restart record");
            restart_seen = true;
            cursor += 8;
            continue;
         }
         return fail_at(cursor, "unknown or non-canonical AGX3 record");
      }

      switch (record.kind) {
      case AGX_VDM_RECORD_BARRIER:
         cursor += record.length;
         break;

      case AGX_VDM_RECORD_PPP_STATE_UPDATE:
         if (!validate_initial_ppp(memory,
                                   record.data.ppp_state_update.pointer,
                                   record.data.ppp_state_update.size_words))
            return false;
         cursor += record.length;
         break;

      case AGX_VDM_RECORD_STATE_UPDATE:
         if (memory->verbose)
            fprintf(stderr, "shim: G17P VS program=%#llx\n",
                    (unsigned long long)record.data.state_update.pipeline);
         if (!validate_ioto_program(memory,
                                    record.data.state_update.pipeline, false,
                                    true,
                                    &local)) {
            fprintf(stderr,
                    "shim: G17P vertex program %#llx at VDM %#llx is "
                    "unowned\n",
                    (unsigned long long)record.data.state_update.pipeline,
                    (unsigned long long)cursor);
            return false;
         }
         cursor += record.length;
         if (!validate_bind_tail(memory, cursor, &local))
            return false;
         cursor += AGX_VDM_HAL200_BIND_TAIL_LENGTH;
         pending_selector = true;
         have_state = true;
         local.state_updates++;
         break;

      case AGX_VDM_RECORD_INDEX_LIST_NONINDEXED:
         if (!have_state || restart_seen)
            return fail_at(cursor, "non-indexed draw has invalid state order");
         local.draws++;
         if (record.data.index_list_nonindexed.instances >
             local.max_instance_count)
            local.max_instance_count =
               record.data.index_list_nonindexed.instances;
         local.primitive_mask |=
            UINT32_C(1) << record.data.index_list_nonindexed.primitive;
         cursor += record.length;
         break;

      case AGX_VDM_RECORD_INDEX_LIST_INDEXED: {
         const struct agx_vdm_index_list_direct_indexed_info *draw =
            &record.data.index_list_indexed;
         const size_t index_size =
            draw->index_size == AGX_VDM_INDEX_LIST_INDEX_SIZE_U8  ? 1
            : draw->index_size == AGX_VDM_INDEX_LIST_INDEX_SIZE_U16 ? 2
                                                                  : 4;
         const uint64_t bytes_needed = (uint64_t)draw->count * index_size;
         if (!have_state || !restart_seen || bytes_needed > SIZE_MAX ||
             !memory->owned(memory->data, draw->index_buffer,
                            (size_t)bytes_needed))
            return fail_at(cursor, "indexed draw buffer is not owned");
         restart_seen = false;
         local.draws++;
         local.indexed_draws++;
         local.primitive_mask |= UINT32_C(1) << draw->primitive;
         local.index_size_mask |= UINT8_C(1) << draw->index_size;
         local.restart_enabled |= draw->restart_enable;
         local.restart_disabled |= !draw->restart_enable;
         cursor += record.length;
         break;
      }

      case AGX_VDM_RECORD_STREAM_LINK:
         cursor = compact_address(record.data.stream_link.target);
         if (!memory->owned(memory->data, cursor, 4))
            return fail_at(cursor, "stream-link target is not owned");
         local.links++;
         break;

      case AGX_VDM_RECORD_STREAM_TERMINATE:
         /* Empty VDM streams are valid for clear-only render passes, whose
          * useful work is performed by the background/EOT programs. */
         if (pending_selector || restart_seen)
            return fail_at(cursor, "stream terminates with incomplete draw");
         local.clear_only = local.draws == 0;
         local.body_only = local.draws > local.state_updates;
         if (stats)
            *stats = local;
         return true;

      case AGX_VDM_RECORD_INDEX_LIST_INDIRECT_NONINDEXED:
         if (!have_state || restart_seen ||
             !memory->owned(
                memory->data,
                record.data.index_list_indirect_nonindexed.indirect_buffer,
                16))
            return fail_at(cursor, "indirect draw arguments are not owned");
         local.draws++;
         local.primitive_mask |= UINT32_C(1) <<
                                 record.data.index_list_indirect_nonindexed.primitive;
         local.indirect_nonindexed = true;
         cursor += record.length;
         break;

      case AGX_VDM_RECORD_INDEX_LIST_INDIRECT_INDEXED: {
         const struct agx_vdm_index_list_indirect_indexed_info *draw =
            &record.data.index_list_indirect_indexed;
         const size_t index_range = ((size_t)draw->index_buffer_size_word + 1) * 4;
         if (!have_state || restart_seen ||
             !memory->owned(memory->data, draw->indirect_buffer, 20) ||
             !memory->owned(memory->data, draw->index_buffer, index_range))
            return fail_at(cursor,
                           "indexed-indirect buffers are not owned");
         local.draws++;
         local.indexed_draws++;
         local.primitive_mask |= UINT32_C(1) << draw->primitive;
         local.index_size_mask |= UINT8_C(1) << draw->index_size;
         local.restart_enabled |= draw->restart_enable;
         local.restart_disabled |= !draw->restart_enable;
         local.indirect_indexed = true;
         cursor += record.length;
         break;
      }

      case AGX_VDM_RECORD_TESSELLATE_DIRECT:
      case AGX_VDM_RECORD_UNKNOWN:
      default:
         return fail_at(cursor, "record is outside native Gallium matrix");
      }
   }

   return fail_at(cursor, "record limit reached before terminate");
}
