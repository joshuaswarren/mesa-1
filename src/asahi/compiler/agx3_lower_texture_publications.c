/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx3_staging.h"
#include <string.h>

struct publication {
   agx_index coordinates, reference;
   uint32_t value;
   unsigned cost;
};

static bool
same_register_value(agx_index a, agx_index b)
{
   /* Cache/discard annotations describe individual reads, not values. */
   return a.type == b.type && a.value == b.value && a.size == b.size &&
          a.channels_m1 == b.channels_m1 && a.memory == b.memory &&
          a.abs == b.abs && a.neg == b.neg;
}

static bool
overlaps_register(agx_index a, agx_index b)
{
   return a.type == AGX_INDEX_REGISTER && b.type == AGX_INDEX_REGISTER &&
          a.memory == b.memory &&
          a.value < b.value + agx_index_size_16(b) &&
          b.value < a.value + agx_index_size_16(a);
}

/* Native register-only instructions preserve the sampler-private file. Keep
 * compound memory/texture operations, control flow, coverage changes and
 * arithmetic publications as region boundaries until their native resource
 * effects are represented. This classifies effects; it does not reorder them.
 */
static bool
preserves_publications(const agx_instr *I)
{
   if (agx_texture_resource(I))
      return true;
   const struct agx_opcode_info *info = &agx_opcodes_info[I->op];
   if (I->agx3_tex_coord_stage || I->nr_dests != 1 ||
       I->dest[0].type != AGX_INDEX_REGISTER || I->dest[0].memory ||
       !info->can_eliminate || !info->can_reorder ||
       info->schedule_class != AGX_SCHEDULE_CLASS_NONE)
      return false;
   switch (I->op) {
   case AGX_OPCODE_TEXTURE_SAMPLE:
   case AGX_OPCODE_TEXTURE_LOAD:
   case AGX_OPCODE_IMAGE_LOAD:
   case AGX_OPCODE_INTERP_AGX3:
   case AGX_OPCODE_ITRCOORDS_AGX3:
   case AGX_OPCODE_RHW_INTERP_AGX3:
   case AGX_OPCODE_LDCF:
      return false;
   default:
      return true;
   }
}

/* This is the relocatable coordinate ABI, independent of shader identity or
 * instruction neighbourhood. Other native operand ABIs still have compound
 * lowering. Arithmetic-to-staging retains its helper-lane provenance.
 */
static bool
has_coordinate_publication(const agx_instr *I)
{
   if (I->op != AGX_OPCODE_TEXTURE_SAMPLE || agx_texture_resource(I) ||
       I->agx3_tex_coords_staged || I->dim != AGX_DIM_2D || I->query_lod ||
       (I->offset && !I->offset_xyz) || I->nr_srcs != 6 ||
       I->src[0].type != AGX_INDEX_REGISTER || I->src[0].memory ||
       I->src[0].abs || I->src[0].neg || I->src[0].size != AGX_SIZE_32 ||
       agx_channels(I->src[0]) != 2)
      return false;

   if (I->shadow && (I->src[5].type != AGX_INDEX_REGISTER ||
                    I->src[5].memory || I->src[5].abs || I->src[5].neg ||
                    I->src[5].size != AGX_SIZE_32))
      return false;

   return I->lod_mode == AGX_LOD_MODE_AUTO_LOD ||
          ((I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS ||
            I->lod_mode == AGX_LOD_MODE_LOD_MIN) &&
           I->src[1].type == AGX_INDEX_IMMEDIATE);
}

bool
agx3_lower_texture_publications(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return false;

   uint32_t next = 0;
   agx_foreach_instr_global(ctx, I)
      next = MAX2(next, agx_texture_resource(I));

   bool progress = false;
   agx_foreach_block(ctx, block) {
      /* At most one four-half pair fits per entry. Reserving each available
       * value's aligned footprint bounds future-use overlap by the actual
       * register-file capacity, independently of the number of samples.
       * Eviction ends reuse eligibility; it never removes a required copy.
       */
      struct publication available[AGX3_STAGING_HALFS / 4];
      unsigned count = 0, reserved = 0;
      agx_foreach_instr_in_block_safe(block, I) {
         if (has_coordinate_publication(I)) {
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            agx_index reference = agx_null();
            if (I->shadow) {
               reference = I->src[5];
               reference.channels_m1 = 0;
            }
            unsigned match = count;
            for (unsigned j = 0; j < count; ++j) {
               if (same_register_value(available[j].coordinates, I->src[0]) &&
                   same_register_value(available[j].reference, reference)) {
                  match = j;
                  break;
               }
            }
            uint32_t value;
            if (match < count) {
               const struct publication hit = available[match];
               value = hit.value;
               memmove(&available[match], &available[match + 1],
                        (count - match - 1) * sizeof(*available));
               available[count - 1] = hit;
            } else {
               const unsigned cost = I->shadow ? 8 : 4;
               while (reserved + cost > AGX3_STAGING_HALFS) {
                  reserved -= available[0].cost;
                  memmove(available, available + 1, --count * sizeof(*available));
               }
               agx_publish_texture_agx3(&b, I->src[0], reference, ++next);
               value = next;
               available[count++] = (struct publication){
                  I->src[0], reference, value, cost,
               };
               reserved += cost;
            }
            /* The resource, not its original GPRs, feeds later samples. */
            I->src[0] = agx_get_vec_index(value, AGX_SIZE_32, I->shadow ? 3 : 2);
            I->src[0].type = AGX_INDEX_RESOURCE;
            if (I->shadow)
               I->src[5] = agx_null();
            progress = true;
         }

         if (!preserves_publications(I)) {
            count = reserved = 0;
            continue;
         }
         /* Any overlapping physical definition changes the input value,
          * including an asynchronous sample result or a half-register write.
          */
         for (unsigned j = 0; j < count;) {
            bool overwritten = false;
            agx_foreach_dest(I, d) {
               overwritten |= overlaps_register(I->dest[d], available[j].coordinates) ||
                              overlaps_register(I->dest[d], available[j].reference);
            }
            if (!overwritten) {
               ++j;
               continue;
            }
            reserved -= available[j].cost;
            memmove(&available[j], &available[j + 1],
                     (--count - j) * sizeof(*available));
         }
      }
   }
   return progress;
}
