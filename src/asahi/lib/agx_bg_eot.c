/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_bg_eot.h"
#include "agx_g17p_wide_mrt_eot.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "agx_compile.h"
#include "agx3_code_container_core.h"
#include "agx_device.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "libagx_shaders.h"
#include "nir.h"
#include "nir_builder.h"
#include "pool.h"

#include <inttypes.h>
#include <stdlib.h>

static bool
agx_g17p_rt0_eot_text_supported(const struct agx_shader_part *bin)
{
   /* Admit only complete native builtin texts: the common 68-byte RT0
    * program, an exact 84-byte layered or signed-integer RT0 program, the
    * admitted 120-byte RGBA8 pair, or the
    * 238-byte different-partition RGBA16F+R8 WBOIT pair. None has a synthetic
    * compute preamble; the CODE container supplies the stop constant. */
   const bool rt0 = bin && bin->info.binary_size == 68 &&
                    bin->info.nr_gprs == 8 &&
                    bin->info.agx3_staging_halfs == 8;
   const bool native_84_rt0 = bin && bin->info.binary_size == 84 &&
                              bin->info.nr_gprs == 8 &&
                              bin->info.agx3_staging_halfs == 8;
   const bool rgba8_mrt = bin && bin->info.binary_size == 120 &&
                          bin->info.nr_gprs == 8 &&
                          bin->info.agx3_staging_halfs == 16;
   const bool wboit_mrt = bin && bin->info.binary_size == 238 &&
                          bin->info.nr_gprs == 8 &&
                          bin->info.agx3_staging_halfs == 16;
   /* The multi-target program of two to eight targets, eight staging halves
    * and one 44-byte tail each. */
   const bool raw32_mrt = bin && bin->info.agx3_staging_halfs >= 16 &&
                         bin->info.agx3_staging_halfs <=
                            8 * AGX_MAX_RENDER_TARGETS &&
                         bin->info.agx3_staging_halfs % 8 == 0 &&
                         bin->info.binary_size ==
                            32 + 44 * (bin->info.agx3_staging_halfs / 8) &&
                         bin->info.nr_gprs == 8 &&
                         bin->info.agx3_staging_known;
   /* The 2D-ARRAY multi-target program of two to eight targets. It carries the
    * layer wire in addition to the XY tile origin, so it has a longer common
    * lead and a 54-byte tail each, and Apple's metadata asks for five GPR
    * dwords rather than four. See agx3_g17p_array_eot.h. */
   const bool array_mrt = bin && bin->info.agx3_staging_halfs >= 16 &&
                          bin->info.agx3_staging_halfs <=
                             8 * AGX_MAX_RENDER_TARGETS &&
                          bin->info.agx3_staging_halfs % 8 == 0 &&
                          bin->info.binary_size ==
                             42 + 54 * (bin->info.agx3_staging_halfs / 8) &&
                          bin->info.nr_gprs == 10 &&
                          bin->info.agx3_staging_known;
   return bin && bin->binary &&
          (rt0 || native_84_rt0 || rgba8_mrt || wboit_mrt || raw32_mrt ||
           array_mrt) &&
          bin->info.main_offset == 0 &&
          bin->info.main_size == bin->info.binary_size &&
          !bin->info.scratch_size && bin->info.agx3_staging_known;
}

static bool
agx_g17p_upload_bg_eot(struct agx_bg_eot_cache *cache,
                       const struct agx_shader_part *bin,
                       struct agx_bg_eot_shader *res)
{
   /* Match the standalone allocation lead already used for G17P main
    * shaders. Native packed CODE blocks do not each own such a lead. */
   const size_t lead_size = 0x340;
   /* LDSHDR's existing auxiliary fetch-span metadata is based on compiler
    * text size. Reserve at least that span after the main entry too, so the
    * EOT text128/main64 case cannot prefetch outside this CODE block. */
   const size_t main_span = MAX2(bin->info.main_size, bin->info.binary_size);
   const size_t block_size = agx3_code_block_size(main_span);
   const unsigned main_offset = bin->info.main_offset;
   if (!bin->binary || !block_size || block_size > SIZE_MAX - lead_size ||
       bin->info.has_preamble ||
       bin->info.rodata.size_16 ||
       (main_offset != 0 &&
        main_offset != AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE) ||
       main_offset > bin->info.binary_size ||
       bin->info.main_size > bin->info.binary_size - main_offset) {
      mesa_loge("asahi: G17P %s program geometry is unproven: binary=%u "
                "main=%u@%u preamble=%u rodata=%u",
                bin->info.stage == MESA_SHADER_COMPUTE ? "end-of-tile"
                                                       : "background",
                bin->info.binary_size, bin->info.main_size, main_offset,
                bin->info.has_preamble, bin->info.rodata.size_16);
      return false;
   }

   const size_t size = lead_size + block_size;
   uint8_t *image = malloc(size);
   if (!image)
      return false;

   agx3_build_container_lead(image);
   const uint8_t *text = bin->binary;
   const bool built = agx3_build_code_block(
      image + lead_size, block_size, text + main_offset, bin->info.main_size,
      main_span, main_offset ? text : NULL, true);
   const uint64_t allocation = built
      ? agx_pool_upload_aligned_with_bo(&cache->pool, image, size, 128,
                                         &res->bo)
      : 0;
   if (!allocation) {
      free(image);
      return false;
   }

   /* Keep compiler-text metadata intact: native EOT has binary_size=68,
    * main_offset=0 and main_size=68. Its LDSHDR fetch-span calculation uses
    * the main padded to128, not this allocation's size. Rebase ptr so
    * the existing ptr + main_offset expression names CODE block +0x80.
    * Clear/reload have main_offset=0 and get a fragment constant program.
    *
    * The valid 194053 first EOT Ioto names a different, larger shader whose
    * code page was not captured. This repairs the common loader contract;
    * it is not an assertion of byte parity with that uncaptured EOT block.
    */
   res->ptr = allocation + lead_size + AGX3_CODE_BLOCK_ENTRY_OFFSET - main_offset;
   free(image);
   return true;
}

static bool
lower_tex_handle_to_u0(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_texture_handle_agx)
      return false;

   const unsigned descriptor_size = *(const unsigned *)data;
   assert(descriptor_size == 24 || descriptor_size == 32);

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def_rewrite_uses(
      &intr->def, nir_bindless_image_agx(
                     b, nir_imul_imm(b, intr->src[0].ssa, descriptor_size),
                     .desc_set = 0));

   return true;
}

static unsigned
agx_bg_eot_texture_descriptor_size(const struct agx_device *dev)
{
   switch (agx_chip_descriptor_generation(dev->chip)) {
   case AGX_DESCRIPTOR_GENERATION_7:
   case AGX_DESCRIPTOR_GENERATION_8:
      return 24;
   case AGX_DESCRIPTOR_GENERATION_9:
   case AGX_DESCRIPTOR_GENERATION_10:
      return 32;
   case AGX_DESCRIPTOR_GENERATION_UNKNOWN:
   default:
      return 0;
   }
}

static struct agx_bg_eot_shader *
agx_compile_bg_eot_shader(struct agx_bg_eot_cache *cache, nir_shader *shader,
                          struct agx_shader_key *key,
                          struct agx_tilebuffer_layout *tib)
{
   agx_nir_lower_texture(shader);
   agx_preprocess_nir(shader);
   if (tib) {
      unsigned descriptor_size =
         agx_bg_eot_texture_descriptor_size(cache->dev);

      if (!descriptor_size) {
         mesa_loge("asahi: background shader has no proven native descriptor geometry");
         ralloc_free(shader);
         return NULL;
      }

      unsigned bindless_base = 0;
      NIR_PASS(_, shader, agx_nir_lower_tilebuffer, tib, NULL, &bindless_base,
               NULL, NULL, false, cache->dev->chip == AGX_CHIP_G17P);
      NIR_PASS(_, shader, agx_nir_lower_monolithic_msaa, tib->nr_samples);
      NIR_PASS(_, shader, agx_nir_lower_multisampled_image_store);
      NIR_PASS(_, shader, agx_nir_lower_texture);
      NIR_PASS(_, shader, nir_shader_intrinsics_pass, lower_tex_handle_to_u0,
               nir_metadata_control_flow, &descriptor_size);
      if (cache->dev->chip == AGX_CHIP_G17P) {
         /* Monolithic MSAA introduces a small sample loop after the common
          * preprocessing pass has already run. Fold that loop here so each
          * local-pixel access has the static one-hot sample mask required by
          * the native G17P tile selector. */
         bool progress;
         do {
            progress = false;
            NIR_PASS(progress, shader, nir_lower_vars_to_ssa);
            NIR_PASS(progress, shader, nir_opt_copy_prop);
            NIR_PASS(progress, shader, nir_opt_constant_folding);
            NIR_PASS(progress, shader, nir_opt_loop_unroll);
            NIR_PASS(progress, shader, nir_opt_dead_cf);
            NIR_PASS(progress, shader, nir_opt_dce);
         } while (progress);

         NIR_PASS(_, shader, agx_nir_lower_g17p_tile_samples,
                  tib->nr_samples);
      }
   }

   struct agx_bg_eot_shader *res = rzalloc(cache->ht, struct agx_bg_eot_shader);
   struct agx_shader_part bin;
   if (!agx_compile_shader_nir(shader, key, &bin)) {
      mesa_loge("asahi: the %s shader did not compile for this chip",
                shader->info.stage == MESA_SHADER_COMPUTE ? "end-of-tile"
                                                          : "background");
      ralloc_free(shader);
      ralloc_free(res);
      return NULL;
   }

   if (cache->dev->chip == AGX_CHIP_G17P && tib == NULL &&
       !agx_g17p_rt0_eot_text_supported(&bin)) {
      free(bin.binary);
      ralloc_free(shader);
      ralloc_free(res);
      return NULL;
   }

   res->info = bin.info;
   if (cache->dev->chip == AGX_CHIP_G17P) {
      if (!agx_g17p_upload_bg_eot(cache, &bin, res)) {
         free(bin.binary);
         ralloc_free(shader);
         ralloc_free(res);
         return NULL;
      }
   } else {
      res->ptr = agx_pool_upload_aligned_with_bo(
         &cache->pool, bin.binary, bin.info.binary_size, 128, &res->bo);
   }
   free(bin.binary);
   ralloc_free(shader);

   return res;
}

static nir_def *
build_background_op(nir_builder *b, enum agx_bg_eot_op op, unsigned rt,
                    unsigned nr, bool source_msaa, bool layered,
                    bool wide_coords)
{
   if (op == AGX_BG_LOAD) {
      nir_def *coord = nir_load_pixel_coord(b);

      /* MSAA/array lowerings assume 32-bit coordinates, could be lifted but
       * that's going to need more work.
       */
      if (layered || source_msaa || wide_coords)
         coord = nir_u2u32(b, coord);

      if (layered) {
         coord = nir_vec3(b, nir_channel(b, coord, 0), nir_channel(b, coord, 1),
                          nir_load_layer_id(b));
      }

      b->shader->info.fs.uses_sample_shading |= source_msaa;

      nir_def *tex = nir_txf(
         b, coord, .ms_index = source_msaa ? nir_load_sample_id(b) : NULL,
         .texture_index = rt * 2,
         .dim = source_msaa ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D,
         .is_array = layered,

         /* The type doesn't matter as long as it matches the store */
         .dest_type = nir_type_uint32,

         /* Layer is necessarily already in-bounds so we do not want the
          * compiler to clamp it, which would require reading the descriptor
          */
         .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP);

      return nir_trim_vector(b, tex, nr);
   } else {
      assert(op == AGX_BG_CLEAR);

      return nir_load_preamble(b, nr, 32, 4 + (rt * 8));
   }
}

static struct agx_bg_eot_shader *
agx_build_background_shader(struct agx_bg_eot_cache *cache,
                            struct agx_bg_eot_key *key)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "agx_background");
   b.shader->info.fs.untyped_color_outputs = true;

   struct agx_shader_key compiler_key = {
      .dev = agx_gather_device_key(cache->dev),
      .fs.ignore_tib_dependencies = true,
      /* G17P multisample backgrounds store through the coverage register. */
      .fs.nr_samples =
         cache->dev->chip == AGX_CHIP_G17P ? key->tib.nr_samples : 0,
      .reserved_preamble = key->reserved_preamble,
      .no_preamble = cache->dev->chip == AGX_CHIP_G17P,
   };

   /* The multi-target background protocol is keyed, not inferred: without
    * this the packer sees a bare run of tile stores, refuses the shader and
    * the draw is dropped. Vulkan says so through mixed_mrt; GL reaches the
    * same shapes through an ordinary draw-buffer mask, and
    * KHR-GL30.draw_buffers clears eight RGBA8 targets at once. */
   unsigned background_targets = 0;
   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt)
      background_targets += key->op[rt] != AGX_BG_EOT_NONE;
   compiler_key.g17p_mrt_bg = key->mixed_mrt || background_targets > 1;

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
      if (key->op[rt] == AGX_BG_EOT_NONE)
         continue;

      unsigned nr = util_format_get_nr_components(key->tib.logical_format[rt]);
      bool msaa = key->tib.nr_samples > 1;
      bool source_msaa = msaa && !key->single_sample_reload;
      bool layered = key->tib.layered;
      assert(nr > 0);

      if (key->g17p_packed_clear && key->op[rt] == AGX_BG_CLEAR &&
          cache->dev->chip == AGX_CHIP_G17P && rt == 0 &&
          agx_g17p_packed_clear_ok(&key->tib)) {
         /* One raw tile word: the driver packed the clear colour exactly as
          * the R8G8B8A8_UNORM FLTNORM pair would, so store its 32 bits. */
         nir_def *packed = nir_load_preamble(&b, 1, 32, 4 + (rt * 8));
         nir_store_local_pixel_agx(&b, packed,
                                   nir_imm_intN_t(&b, 0xFF, 16),
                                   nir_undef(&b, 2, 16),
                                   .base = agx_tilebuffer_offset_B(&key->tib, rt),
                                   .write_mask = 0x1,
                                   .format = PIPE_FORMAT_R32_UINT, .target = rt);
         continue;
      }

      nir_store_output(
         &b, build_background_op(&b, key->op[rt], rt, nr, source_msaa, layered,
                                 cache->dev->chip == AGX_CHIP_G17P),
         nir_imm_int(&b, 0), .write_mask = BITFIELD_MASK(nr),
         .src_type = nir_type_uint32,
         .io_semantics.location = FRAG_RESULT_DATA0 + rt,
         .io_semantics.num_slots = 1);

      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + rt);
   }

   return agx_compile_bg_eot_shader(cache, b.shader, &compiler_key, &key->tib);
}

static struct agx_bg_eot_shader *
agx_build_end_of_tile_shader(struct agx_bg_eot_cache *cache,
                             struct agx_bg_eot_key *key)
{
   const unsigned rgba8_mask = agx_g17p_rgba8_mrt_mask(&key->tib);
   const unsigned active_count = util_bitcount(rgba8_mask);
   if (cache->dev->chip == AGX_CHIP_G17P && rgba8_mask) {
      /* EOT partitions are physical offset/sample-stride quotients, not
       * fragment synchronization classes. All these compact offsets stay
       * in partition zero; native PBEs and offset arguments use active order. */
      for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
         if (key->op[rt] != ((rgba8_mask & BITFIELD_BIT(rt))
                              ? AGX_EOT_STORE : AGX_BG_EOT_NONE))
            return NULL;
      }
      struct agx_g17p_wide_mrt_eot native;
      if (!agx_g17p_wide_mrt_eot_get(active_count, &native))
         return NULL;
      struct agx_shader_part bin = {
         .binary = (void *)native.code,
         .info = {
            .stage = MESA_SHADER_COMPUTE,
            .binary_size = native.size,
            .main_size = native.size,
            .nr_gprs = 8,
            .agx3_staging_known = true,
            .agx3_staging_halfs = native.staging_halfs,
         },
      };
      struct agx_bg_eot_shader *res = rzalloc(cache->ht, struct agx_bg_eot_shader);
      if (!res || !agx_g17p_upload_bg_eot(cache, &bin, res)) {
         ralloc_free(res);
         return NULL;
      }
      res->info = bin.info;
      return res;
   }

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "agx_eot");

   /* A multisample tile can be written out two ways. The default is to
    * resolve it through an ordinary single-sample PBE target, which is what
    * every G17P path except VK_ATTACHMENT_STORE_OP_STORE wants and the only
    * shape the earlier native EOT substitution covered. Storing the surface
    * unresolved needs the multisample IMGWBLK dimension, and on G17P that is
    * a separate Apple builtin rather than a field the backend may set on the
    * 2D one -- so it is only selected when the caller asked for it and the
    * target really is multisample.
    */
   const bool g17p = cache->dev->chip == AGX_CHIP_G17P;
   const bool store_multisample =
      key->tib.nr_samples > 1 && (!g17p || key->multisample_store);

   /* AGX2 carries the sample in the store's coordinate vector, so its
    * multisample block store is a three-component GLSL_SAMPLER_DIM_MS one.
    * Apple's G17P builtins do not: both the resolving and the unresolved
    * end-of-tile programs read exactly the same two grid coordinates and
    * differ only in the IMGWBLK dimension class the packer deposits. Keep
    * the two-coordinate 2D shape on G17P and let the compiler key select
    * which of the two Apple programs is meant, the same way every other
    * G17P end-of-tile selector is chosen. */
   enum glsl_sampler_dim dim = (store_multisample && !g17p)
                                  ? GLSL_SAMPLER_DIM_MS
                                  : GLSL_SAMPLER_DIM_2D;

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
      if (key->op[rt] == AGX_BG_EOT_NONE)
         continue;

      /* The end-of-tile shader is unsuitable to handle spilled render targets.
       * Skip them. If blits are needed with spilled render targets, other parts
       * of the driver need to implement them.
       */
      if (key->tib.spilled[rt])
         continue;

      assert(key->op[rt] == AGX_EOT_STORE);
      unsigned offset_B = agx_tilebuffer_offset_B(&key->tib, rt);

      nir_def *layer = nir_undef(&b, 1, 16);
      if (key->tib.layered)
         layer = nir_u2u16(&b, nir_load_layer_id(&b));

      bool explicit = false;
      nir_def *coords = layer;
      if (cache->dev->chip == AGX_CHIP_G17P) {
         /* AGX3's EOT imgwblk has no implicit pixel-coordinate form.  Apple's
          * G17P compiler reads the hardware grid X/Y SRs explicitly.  Array
          * stores append the hardware layer ID; dropping it here made every
          * layered EOT store address layer zero even though the image
          * descriptor and tilebuffer were correctly marked as arrays. */
         nir_def *xy = nir_trim_vector(
            &b, nir_load_global_invocation_id(&b, 32), 2);
         coords = key->tib.layered
                     ? nir_vec3(&b, nir_channel(&b, xy, 0),
                                nir_channel(&b, xy, 1),
                                nir_u2u32(&b, nir_load_layer_id(&b)))
                     : xy;
         explicit = true;
      }

      enum pipe_format physical =
         agx_tilebuffer_physical_format(&key->tib, rt);
      if (cache->dev->chip == AGX_CHIP_G17P &&
          (key->tib.logical_format[rt] == PIPE_FORMAT_R11G11B10_FLOAT ||
           key->tib.logical_format[rt] == PIPE_FORMAT_R9G9B9E5_FLOAT)) {
         /* G17P keeps packed-float tiles as one raw 32-bit word. The matching
          * PBE descriptor also uses R32_UINT to preserve those exact bits. */
         physical = PIPE_FORMAT_R32_UINT;
      } else if (cache->dev->chip == AGX_CHIP_G17P &&
          util_format_is_srgb(key->tib.logical_format[rt])) {
         /* G17P applies the sRGB transfer in the fragment's own packs
          * (fltnorm/normflt sRGB modes) and stores the resulting bytes in a
          * U8NORM imageblock.  EOT is a raw transfer
          * from that imageblock to the format-aware PBE descriptor; asking
          * IMGWBLK to consume SRGBA8 would perform a second conversion and has
          * no native builtin.  Match the physical format selected by the
          * fragment lowering, just as background clear/reload already do. */
         physical = (enum pipe_format)AIL_ISA_FORMAT_U8NORM;
      }

      nir_image_store_block_agx(
         &b, nir_imm_intN_t(&b, rt, 16), nir_imm_int(&b, offset_B), coords,
         .format = physical,
         .image_dim = dim, .image_array = key->tib.layered,
         .explicit_coord = explicit);
   }

   struct agx_shader_key compiler_key = {
      .dev = agx_gather_device_key(cache->dev),
      .reserved_preamble = key->reserved_preamble,
      .no_preamble = cache->dev->chip == AGX_CHIP_G17P,
   };

   if (cache->dev->chip == AGX_CHIP_G17P) {
      agx_g17p_configure_rt0_eot_key(key->tib.logical_format[0],
                                    &compiler_key);
      compiler_key.g17p_rt0_eot_multisample = store_multisample;

      /* A mixed layout storing several targets uses the multi-target
       * builtin, selecting each target's IMGWBLK tail by its format. A
       * single stored target keeps its format's own program. */
      /* Counted whatever named the layout. A GL draw-buffer mask reaches the
       * same several-target store as a Vulkan mixed layout; the selectors
       * come from each target's own logical format either way. */
      unsigned stores = 0;
      for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt)
         stores += key->op[rt] == AGX_EOT_STORE;

      if (stores > 1 &&
          stores <= ARRAY_SIZE(compiler_key.g17p_mrt_eot_selectors)) {
         for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
            if (key->op[rt] == AGX_EOT_STORE) {
               compiler_key.g17p_mrt_eot_selectors
                  [compiler_key.g17p_mrt_eot_count++] =
                  agx_g17p_mrt_eot_selector(key->tib.logical_format[rt]);
            }
         }
      }
   }

   compiler_key.g17p_r11g11b10_mrt_eot =
      cache->dev->chip == AGX_CHIP_G17P &&
      agx_g17p_classify_mrt(&key->tib) == AGX_G17P_MRT_R11G11B10;

   return agx_compile_bg_eot_shader(cache, b.shader, &compiler_key, NULL);
}

struct agx_bg_eot_shader *
agx_get_bg_eot_shader(struct agx_bg_eot_cache *cache,
                      struct agx_bg_eot_key *key)
{
   struct hash_entry *ent = _mesa_hash_table_search(cache->ht, key);
   if (ent)
      return ent->data;

   struct agx_bg_eot_shader *ret = NULL;
   bool has_store = false;

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
      if (key->op[rt] == AGX_EOT_STORE) {
         has_store = true;
         ret = agx_build_end_of_tile_shader(cache, key);
         break;
      }
   }

   if (!ret && !has_store)
      ret = agx_build_background_shader(cache, key);

   if (!ret)
      return NULL;

   ret->key = *key;
   _mesa_hash_table_insert(cache->ht, &ret->key, ret);
   return ret;
}

DERIVE_HASH_TABLE(agx_bg_eot_key)

void
agx_bg_eot_init(struct agx_bg_eot_cache *cache, struct agx_device *dev)
{
   agx_pool_init(&cache->pool, dev, "Internal programs",
                 AGX_BO_EXEC | AGX_BO_LOW_VA, true);
   simple_mtx_init(&cache->lock, mtx_plain);
   cache->ht = agx_bg_eot_key_table_create(NULL);
   cache->dev = dev;
}

void
agx_bg_eot_cleanup(struct agx_bg_eot_cache *cache)
{
   agx_pool_cleanup(&cache->pool);
   _mesa_hash_table_destroy(cache->ht, NULL);
   simple_mtx_destroy(&cache->lock);
   cache->ht = NULL;
   cache->dev = NULL;
}

static struct agx_precompiled_shader *
agx_get_precompiled_locked(struct agx_bg_eot_cache *cache, unsigned program)
{
   simple_mtx_assert_locked(&cache->lock);

   /* It is possible that, while waiting for the lock, another thread uploaded
    * the shader. Check for that so we don't double-upload.
    */
   if (cache->precomp[program])
      return cache->precomp[program];

   if (!cache->dev->libagx_programs)
      return NULL;

   /* Otherwise, we need to upload. */
   struct agx_precompiled_shader *p =
      ralloc(cache->ht, struct agx_precompiled_shader);

   const uint32_t *bin = cache->dev->libagx_programs[program];
   if (!bin) {
      mesa_loge("asahi: libagx program %u is not built for this chip", program);
      ralloc_free(p);
      return NULL;
   }
   const struct agx_precompiled_kernel_info *info = (void *)bin;
   const void *binary = (const uint8_t *)bin + sizeof(*info);

   p->info = *info;
   p->rodata_offset = 0;

   assert(info->main_offset == 0 || program != LIBAGX_HELPER);

   p->b.workgroup =
      agx_workgroup(info->workgroup_size[0], info->workgroup_size[1],
                    info->workgroup_size[2]);

   if (cache->dev->chip == AGX_CHIP_G17P) {
      const size_t lead_size = 0x340;
      const size_t main_span = MAX2(info->main_size, info->binary_size);
      const size_t block_size = agx3_code_block_size(main_span);
      /* Promoted constants live in the binary and are published as a uniform
       * range beside the kernel's arguments, the same way the legacy USC path
       * maps them with agx_usc_immediates(). They are carried in this image
       * so one allocation still holds the whole program. */
      const size_t rodata_size = (size_t)info->rodata.size_16 * 2;
      const size_t rodata_pad = rodata_size ? ALIGN_POT(rodata_size, 16) : 0;
      if (!block_size || block_size > SIZE_MAX - lead_size - rodata_pad ||
          info->main_offset > info->binary_size ||
          info->main_size > info->binary_size - info->main_offset ||
          (rodata_size &&
           (info->rodata.offset > info->binary_size ||
            rodata_size > info->binary_size - info->rodata.offset)) ||
          info->preamble_offset || info->preamble_size ||
          info->nr_preamble_gprs) {
         mesa_loge("asahi: libagx program %u has no G17P launch shape: "
                   "main %u@%u of %u, preamble %u@%u/%u, rodata %u",
                   program, info->main_size, info->main_offset,
                   info->binary_size, info->preamble_size, info->preamble_offset,
                   info->nr_preamble_gprs, info->rodata.size_16);
         ralloc_free(p);
         return NULL;
      }

      const size_t size = lead_size + block_size + rodata_pad;
      uint8_t *image = calloc(1, size);
      if (!image) {
         ralloc_free(p);
         return NULL;
      }
      agx3_build_container_lead(image);
      const bool built = agx3_build_code_block(
         image + lead_size, block_size,
         (const uint8_t *)binary + info->main_offset, info->main_size,
         main_span, NULL, false);
      if (rodata_size) {
         memcpy(image + lead_size + block_size,
                (const uint8_t *)binary + info->rodata.offset, rodata_size);
         p->rodata_offset = lead_size + block_size;
      }
      p->ptr = built
         ? agx_pool_upload_aligned_with_bo(&cache->pool, image, size, 128,
                                           &p->bo)
         : 0;
      p->entry_offset = lead_size + AGX3_CODE_BLOCK_ENTRY_OFFSET;
      free(image);
   } else {
      p->ptr = agx_pool_upload_aligned_with_bo(
         &cache->pool, binary, info->binary_size, 128, &p->bo);
      p->entry_offset = info->main_offset;
   }
   if (!p->ptr) {
      ralloc_free(p);
      return NULL;
   }

   if (cache->dev->chip == AGX_CHIP_G17P) {
      /* HAL200 consumes the CODE entry through a dispatch-local Ioto program;
       * none of the legacy USC launch words below are part of that ABI. */
      p_atomic_set(&cache->precomp[program], p);
      return p;
   }

   /* Bake launch */
   agx_pack(&p->b.launch, CDM_LAUNCH_WORD_0, cfg) {
      cfg.texture_state_register_count = 8 /* same encoding as 0 */;
      cfg.sampler_state_register_count = 1;
      cfg.uniform_register_count = info->push_count;
      cfg.preshader_register_count = info->nr_preamble_gprs;
   }

   /* Bake USC */
   struct agx_usc_builder b =
      agx_usc_builder(p->b.usc.data, sizeof(p->b.usc.data));

   agx_usc_immediates(&b, &info->rodata, p->ptr);

   if (info->uses_txf)
      agx_usc_push_packed(&b, SAMPLER, cache->dev->txf_sampler);

   agx_usc_shared(&b, info->local_size, info->imageblock_stride, 0);

   agx_usc_pack(&b, SHADER, cfg) {
      cfg.code = agx_usc_addr(cache->dev, p->ptr + info->main_offset);
      cfg.unk_2 = 3;
   }

   agx_usc_pack(&b, REGISTERS, cfg) {
      cfg.register_count = info->nr_gprs;
      cfg.spill_size = 0;
   }

   if (info->nr_preamble_gprs) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code = agx_usc_addr(cache->dev, p->ptr + info->preamble_offset);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   p->b.usc.size = b.head - p->b.usc.data;

   /* We must only write to the cache once we are done compiling, since other
    * threads may be reading the cache concurrently. Do this last.
    */
   p_atomic_set(&cache->precomp[program], p);
   return p;
}

struct agx_precompiled_shader *
agx_get_precompiled(struct agx_bg_eot_cache *cache, unsigned program)
{
   /* Shaders are immutable once written, so if we atomically read a non-NULL
    * shader, then we have a valid cached shader and are done.
    */
   struct agx_precompiled_shader *ret = p_atomic_read(cache->precomp + program);

   if (ret != NULL)
      return ret;

   /* Otherwise, take the lock and upload. */
   simple_mtx_lock(&cache->lock);
   ret = agx_get_precompiled_locked(cache, program);
   simple_mtx_unlock(&cache->lock);

   return ret;
}

uint64_t
agx_helper_program(struct agx_bg_eot_cache *cache)
{
   struct agx_precompiled_shader *pc =
      agx_get_precompiled(cache, LIBAGX_HELPER);

   return pc ? (pc->ptr | 1) : 0;
}
