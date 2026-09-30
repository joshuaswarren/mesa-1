/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * DIAGNOSTIC EXPERIMENT ONLY -- inert unless HK_G17P_APPLE_CS or
 * HK_G17P_APPLE_FS names a file. See hk_g17p_apple_override.h.
 *
 * The files are the __TEXT,__text sections that `metal-nt -arch
 * applegpu_g17p` produced for native_throughput.swift's compute_main and
 * fragment_main (neo-throughput-apple-compiler/apple-runtime/
 * runtime-compute-compute-7408.bin, 250 B, and
 * runtime-render-fragment-14176.bin, 446 B). Layout of both, from the
 * recovered compose-byte decodes (compute.fields / fragment.fields):
 *
 *   text[0, 4)     constant program: a lone END (const_calc_phases has one
 *                  phase with one temporary and no uniform inputs)
 *   text[4, 64)    NOP padding
 *   text[64, n)    main, ending in END
 *
 * The constant program writes nothing, so only the main is linked; the
 * shader runs as an ordinary no-preamble Honeykrisp G17P shader.
 *
 * Apple compile-reply metadata (apple-runtime/metadata-saved-schema.json):
 *   compute:  DMA {type 3 -> regs 0..1 (buffer-binding table: buffer(0) VA)},
 *             {type 5, buffer(1) -> regs 4..5 (Params)},
 *             {type 6 -> regs 6..7 (static constants 0x61c88647, 0)};
 *             temporary_register_count 7, special register 80, no IPR.
 *   fragment: DMA {type 6 -> regs 4..7 (static constants 0x3b808081,
 *             0x3e7cfcfe, 0x61c88647, 0)}, {type 5, buffer(0) -> regs 2..3
 *             (Params)}; temporary_register_count 8, special registers 80/81,
 *             input/output MRT mask 0xf, blend inline (LLDCOL..LSTCOL,
 *             PIXREL, END), no IPR.
 * Register numbers above are 32-bit; Honeykrisp ranges count 16-bit halves.
 * The instruction fields that consume them (uniform slot f23/f38 = 3):
 * compute icmp U10 (iterations), bit_xor U8 (seed), isub U12 (constant),
 * stdev base u0; fragment icmp U6, bit_xor U4, isub U12, f32mul U8,
 * f32add U10.
 *
 * Mutant texts (for example a re-encoded loop head) of the same size and
 * layout are accepted so the lab can bisect Apple's encoding differences;
 * every application prints the FNV-1a 64 hash of the text it loaded.
 */

#include "hk_g17p_apple_override.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/compiler/agx3_ioto_uniform_program.h"
#include "asahi/compiler/agx_compile.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_shader.h"
#include "nir.h"
#include "util/log.h"
#include "util/mesa-blake3.h"
#include "util/u_debug.h"
#include "vk_log.h"

DEBUG_GET_ONCE_OPTION(hk_g17p_apple_cs_path, "HK_G17P_APPLE_CS", NULL)
DEBUG_GET_ONCE_OPTION(hk_g17p_apple_fs_path, "HK_G17P_APPLE_FS", NULL)

#define APPLE_MAIN_OFFSET 64u
#define APPLE_CS_TEXT_SIZE 250u
#define APPLE_FS_TEXT_SIZE 446u

/* nir_shader_info::source_blake3 of throughput.comp.spv (743b1049...) and
 * throughput.frag.spv (93d39f48...), printed form. */
static const uint32_t cs_spirv_blake3[BLAKE3_OUT_LEN32] = {
   0x434b5236, 0xe3caa3a8, 0x69424c3b, 0x540ebb96,
   0x42184a27, 0xbb990875, 0x00f10a2c, 0xcd90617c,
};
static const uint32_t fs_spirv_blake3[BLAKE3_OUT_LEN32] = {
   0x328aa085, 0xef430ec8, 0x07070784, 0x7a2ff5ff,
   0x3c1c9aa6, 0x6ddee6bd, 0x7599244a, 0x68e08f23,
};

static const uint32_t cs_static_constants[4] = {0x61c88647u, 0, 0, 0};
static const uint32_t fs_static_constants[4] = {
   0x3b808081u, /* 1.0f / 255 */
   0x3e7cfcfeu, /* 63.0f / 255 */
   0x61c88647u, /* -0x9e3779b9 */
   0,
};

static const uint8_t agx3_end[4] = {0x0e, 0x00, 0x00, 0x00};

static uint64_t
fnv1a64(const uint8_t *p, size_t n)
{
   uint64_t h = UINT64_C(0xcbf29ce484222325);
   for (size_t i = 0; i < n; ++i) {
      h ^= p[i];
      h *= UINT64_C(0x100000001b3);
   }
   return h;
}

/* Read and structurally validate one Apple text. Returns a malloc'd copy of
 * the main (text[64, size)) or NULL. */
static void *
load_main(const char *path, size_t expected_size, uint64_t *hash_out,
          size_t *main_size_out)
{
   FILE *f = fopen(path, "rb");
   if (!f) {
      mesa_loge("HK_G17P_APPLE: cannot open %s", path);
      return NULL;
   }
   uint8_t text[512];
   const size_t n = fread(text, 1, sizeof(text), f);
   fclose(f);
   if (n != expected_size) {
      mesa_loge("HK_G17P_APPLE: %s is %zu bytes, expected %zu", path, n,
                expected_size);
      return NULL;
   }
   /* CProg END, NOP padding (0x0006 halfwords), main ending in END. */
   if (memcmp(text, agx3_end, sizeof(agx3_end)) ||
       memcmp(text + n - sizeof(agx3_end), agx3_end, sizeof(agx3_end))) {
      mesa_loge("HK_G17P_APPLE: %s does not have the CProg END / main END "
                "layout", path);
      return NULL;
   }
   for (unsigned i = 4; i < APPLE_MAIN_OFFSET; i += 2) {
      if (text[i] != 0x06 || text[i + 1] != 0x00) {
         mesa_loge("HK_G17P_APPLE: %s padding is not NOP at %u", path, i);
         return NULL;
      }
   }

   const size_t main_size = n - APPLE_MAIN_OFFSET;
   void *code = malloc(main_size);
   if (!code)
      return NULL;
   memcpy(code, text + APPLE_MAIN_OFFSET, main_size);
   *hash_out = fnv1a64(text, n);
   *main_size_out = main_size;
   return code;
}

static void
replace_main(struct hk_shader *shader, void *code, size_t main_size,
             uint16_t nr_gprs)
{
   struct agx_shader_info *info = &shader->b.info;
   free((void *)shader->b.binary);
   shader->b.binary = code;

   info->binary_size = main_size;
   info->has_preamble = false;
   info->preamble_offset = 0;
   info->preamble_size = 0;
   info->main_offset = 0;
   info->main_size = main_size;

   /* The G17P launch does not encode a GPR count; keep the larger of ours
    * and Apple's so no host-side consumer under-reserves. */
   info->nr_gprs = MAX2(info->nr_gprs, nr_gprs);
   info->nr_preamble_gprs = 0;
   info->scratch_size = 0;
   info->agx3_spectpr_scratch_size = 0;
   info->preamble_scratch_size = 0;
   /* SPECTPR D = Apple's largest IPR bytes, absent (0) for both kernels.
    * Mesa's own variants of both kernels also publish D = 0. */
   info->agx3_staging_halfs = 0;
   info->agx3_staging_known = true;
   info->agx3_preamble_staging_halfs = 0;
   info->agx3_preamble_staging_known = true;
   info->agx3_preamble_uniform_inputs = false;
   info->agx3_preamble_tps_inputs = false;
   info->texture_state_count = 0;
   info->sampler_state_count = 0;
   info->uses_sampler_heap = false;
   info->uses_txf = false;
   info->rodata = (struct agx_rodata){0};
   info->agx3_fltnorm_start = false;
}

VkResult
hk_g17p_apple_override_apply(struct hk_device *dev,
                             const struct nir_shader *nir,
                             struct hk_shader *shader)
{
   if (dev->dev.chip != AGX_CHIP_G17P)
      return VK_SUCCESS;

   const bool compute = nir->info.stage == MESA_SHADER_COMPUTE &&
                        shader->b.info.stage == MESA_SHADER_COMPUTE;
   const bool fragment = nir->info.stage == MESA_SHADER_FRAGMENT &&
                         shader->b.info.stage == MESA_SHADER_FRAGMENT;
   const char *path = compute    ? debug_get_option_hk_g17p_apple_cs_path()
                      : fragment ? debug_get_option_hk_g17p_apple_fs_path()
                                 : NULL;
   if (!path || !path[0])
      return VK_SUCCESS;

   if (!_mesa_printed_blake3_equal(nir->info.source_blake3,
                                   compute ? cs_spirv_blake3
                                           : fs_spirv_blake3))
      return VK_SUCCESS;

   uint64_t hash = 0;
   size_t main_size = 0;
   void *code = load_main(path,
                          compute ? APPLE_CS_TEXT_SIZE : APPLE_FS_TEXT_SIZE,
                          &hash, &main_size);
   if (!code)
      return vk_errorf(dev, VK_ERROR_UNKNOWN,
                       "HK_G17P_APPLE: Apple text refused");

   struct agx_shader_info *info = &shader->b.info;
   const struct agx_shader_info before = *info;

   if (compute) {
      /* Apple reads the global thread index directly (special register 80)
       * and no workgroup-shared memory; refuse a variant that needs more. */
      if (info->local_size != 0 || shader->info.set_count < 1) {
         free(code);
         return vk_errorf(dev, VK_ERROR_UNKNOWN,
                          "HK_G17P_APPLE_CS: compute interface mismatch");
      }
      replace_main(shader, code, main_size, 14 /* 7 x 32-bit */);
      shader->info.g17p_apple_cs = true;
   } else {
      /* Mesa's split main leaves the tile write to the epilog, so it
       * reports tag_write_disable; Apple's main writes RT0 itself. */
      if (info->writes_sample_mask || info->varyings.fs.nr_bindings != 0) {
         free(code);
         return vk_errorf(dev, VK_ERROR_UNKNOWN,
                          "HK_G17P_APPLE_FS: fragment interface mismatch "
                          "(sample_mask=%u bindings=%u)",
                          info->writes_sample_mask,
                          info->varyings.fs.nr_bindings);
      }
      replace_main(shader, code, main_size, 16 /* 8 x 32-bit */);
      /* Apple's main blends inline: it reads and writes RT0 (LLDCOL /
       * LSTCOL), so the pass is translucent, and it releases colour
       * partition 0 itself (terminal PIXREL 0x20c). No epilog is linked. */
      info->reads_tib = true;
      info->tag_write_disable = false;
      info->agx3_colour_link = (struct agx3_colour_link_info){
         .colour_classes = 1,
      };
      shader->info.g17p_apple_fs = true;
   }

   fprintf(stderr,
           "HK_G17P_APPLE_%s: APPLIED text fnv=%016" PRIx64 " main %u->%u "
           "gprs %u->%u staging %u->0 rodata %u->0 preamble %u->0\n",
           compute ? "CS" : "FS", hash, before.main_size, info->main_size,
           before.nr_gprs, info->nr_gprs, before.agx3_staging_halfs,
           before.rodata.size_16, before.has_preamble);
   return VK_SUCCESS;
}

static bool
push_ranges(struct hk_cmd_buffer *cmd, uint64_t params_va,
            uint16_t params_start, const uint32_t constants[4],
            uint16_t constants_start, uint16_t constants_halfs,
            struct agx3_ioto_uniform_range *ranges, size_t capacity,
            size_t *count)
{
   if (*count + 2 > capacity)
      return false;
   const uint64_t constants_va =
      hk_pool_upload(cmd, constants, 4 * sizeof(uint32_t), 16);
   if (!constants_va)
      return false;
   ranges[(*count)++] = (struct agx3_ioto_uniform_range){
      .start_halfs = params_start, .size_halfs = 4, .address = params_va,
   };
   ranges[(*count)++] = (struct agx3_ioto_uniform_range){
      .start_halfs = constants_start,
      .size_halfs = constants_halfs,
      .address = constants_va,
   };
   return true;
}

bool
hk_g17p_apple_cs_ranges(struct hk_cmd_buffer *cmd, const struct hk_shader *s,
                        uint64_t root_ptr,
                        struct agx3_ioto_uniform_range *ranges,
                        size_t capacity, size_t *range_count)
{
   *range_count = 0;
   const struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   const struct hk_descriptor_set *set = desc->sets[0];
   if (!s->info.g17p_apple_cs || !root_ptr || !set || !set->layout ||
       set->layout->binding_count < 1 || !desc->root.sets[0] ||
       capacity < 3)
      return false;

   const struct hk_descriptor_set_binding_layout *binding =
      &set->layout->binding[0];
   if (binding->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      return false;

   /* A G17P storage-buffer descriptor is struct hk_buffer_address; its first
    * qword is the base address Apple's STDEV takes from uniform quad 0. */
   ranges[(*range_count)++] = (struct agx3_ioto_uniform_range){
      .start_halfs = 0, .size_halfs = 4,
      .address = desc->root.sets[0] + binding->offset,
   };
   if (!push_ranges(cmd, root_ptr + hk_root_descriptor_offset(push), 8,
                    cs_static_constants, 12, 4, ranges, capacity,
                    range_count)) {
      *range_count = 0;
      return false;
   }
   return true;
}

bool
hk_g17p_apple_fs_ranges(struct hk_cmd_buffer *cmd, const struct hk_shader *s,
                        uint64_t root_ptr,
                        struct agx3_ioto_uniform_range *ranges,
                        size_t capacity, size_t *range_count)
{
   *range_count = 0;
   if (!s->info.g17p_apple_fs || !root_ptr)
      return false;
   return push_ranges(cmd, root_ptr + hk_root_descriptor_offset(push), 4,
                      fs_static_constants, 8, 8, ranges, capacity,
                      range_count);
}
