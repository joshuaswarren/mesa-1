/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_cdm_decode_core.h"

#include "asahi/genxml/agx3_cmdstream.h"

#include <string.h>

bool
agx3_cdm_decode_one(enum agx_chip chip, const void *bytes, size_t available,
                    struct agx3_cdm_decoded_record *record)
{
   if (record)
      memset(record, 0, sizeof(*record));

   if (!bytes || !record || available < sizeof(uint32_t))
      return false;

   const uint32_t word = agx3_cmdstream_load_le32(bytes);
   const enum agx_cdm_block_type type = (enum agx_cdm_block_type)(word >> 29);

   switch (type) {
   case AGX_CDM_BLOCK_TYPE_LAUNCH: {
      /* Type zero has three witnessed lengths, and word 0's mode field is
       * what tells them apart: mode 2 reads its whole grid from six u32 in
       * memory and is 24 bytes, where the direct form carries its own three
       * global and three local size words and is 40.  Without keying on the
       * mode a parser cannot consume either one without risking loss of
       * synchronization on the other. */
      switch ((word >> 27) & 3) {
      case 2: {
         /* Unlike the direct unpacker, which takes the available span, the
          * compact one takes the record's exact size. */
         struct agx3_cdm_indirect_launch_info indirect;
         if (available < AGX3_CDM_INDIRECT_LAUNCH_SIZE ||
             !agx3_cdm_indirect_launch_unpack(chip, bytes,
                                              AGX3_CDM_INDIRECT_LAUNCH_SIZE,
                                              &indirect))
            return false;
         record->kind = AGX3_CDM_RECORD_INDIRECT_LAUNCH;
         record->length = AGX3_CDM_INDIRECT_LAUNCH_SIZE;
         record->chip = chip;
         record->data.indirect_launch.word0_payload = indirect.word0_payload;
         record->data.indirect_launch.word1_payload = indirect.word1_payload;
         record->data.indirect_launch.pipeline = indirect.pipeline;
         record->data.indirect_launch.opaque_word3 = indirect.opaque_word3;
         record->data.indirect_launch.word4_payload =
            indirect.indirect_word4_payload;
         record->data.indirect_launch.parameters =
            indirect.indirect_parameters;
         return true;
      }

      case 1: {
         struct agx3_cdm_indirect_launch_36_info indirect;
         if (available < AGX3_CDM_INDIRECT_LAUNCH_36_SIZE ||
             !agx3_cdm_indirect_launch_36_unpack(
                chip, bytes, AGX3_CDM_INDIRECT_LAUNCH_36_SIZE, &indirect))
            return false;
         record->kind = AGX3_CDM_RECORD_INDIRECT_LAUNCH_36;
         record->length = AGX3_CDM_INDIRECT_LAUNCH_36_SIZE;
         record->chip = chip;
         record->data.indirect_launch.word0_payload = indirect.word0_payload;
         record->data.indirect_launch.word1_payload = indirect.word1_payload;
         record->data.indirect_launch.pipeline = indirect.pipeline;
         record->data.indirect_launch.opaque_word3 = indirect.opaque_word3;
         record->data.indirect_launch.word4_payload =
            indirect.indirect_word4_payload;
         record->data.indirect_launch.parameters =
            indirect.indirect_parameters;
         memcpy(record->data.indirect_launch.trailing_words,
                indirect.trailing_words, sizeof(indirect.trailing_words));
         return true;
      }

      case 3:
         return false;

      default:
         break;
      }

      struct agx3_cdm_launch_info launch;
      if (!agx3_cdm_launch_unpack(chip, bytes, available, &launch))
         return false;
      record->kind = AGX3_CDM_RECORD_LAUNCH;
      record->length = AGX3_CDM_LAUNCH_SIZE;
      record->chip = chip;
      record->data.launch.word0_payload = launch.word0_payload;
      record->data.launch.word1_payload = launch.word1_payload;
      record->data.launch.pipeline = launch.pipeline;
      record->data.launch.opaque_word3 = launch.opaque_word3;
      memcpy(record->data.launch.global_size, launch.global_size,
             sizeof(launch.global_size));
      memcpy(record->data.launch.local_size, launch.local_size,
             sizeof(launch.local_size));
      return true;
   }

   case AGX_CDM_BLOCK_TYPE_STREAM_LINK: {
      struct agx3_cdm_stream_link_info link;
      if (!agx3_cdm_stream_link_unpack(chip, bytes, available, &link))
         return false;
      record->kind = AGX3_CDM_RECORD_STREAM_LINK;
      record->length = AGX3_CDM_STREAM_LINK_SIZE;
      record->chip = chip;
      record->data.stream_link.target = link.target;
      record->data.stream_link.with_return = link.with_return;
      return true;
   }

   case AGX_CDM_BLOCK_TYPE_STREAM_TERMINATE:
      if (!agx3_cdm_stream_terminate_validate(chip, bytes, available))
         return false;
      record->kind = AGX3_CDM_RECORD_STREAM_TERMINATE;
      record->length = AGX3_CDM_STREAM_TERMINATE_SIZE;
      record->chip = chip;
      return true;

   case AGX_CDM_BLOCK_TYPE_BARRIER:
      if (agx3_cdm_sstt_flush_invalidate_validate(chip, bytes, available)) {
         record->kind = AGX3_CDM_RECORD_SSTT_FLUSH_INVALIDATE;
         record->length = AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE;
         record->chip = chip;
         return true;
      }

      if (agx3_cdm_secondary_barrier_validate(chip, bytes, available)) {
         record->kind = AGX3_CDM_RECORD_SECONDARY_BARRIER;
         record->length = AGX3_CDM_SECONDARY_BARRIER_SIZE;
         record->chip = chip;
         return true;
      }

      struct agx3_cdm_substream_barrier_info substream;
      if (agx3_cdm_substream_barrier_unpack(chip, bytes, available,
                                            &substream)) {
         record->kind = AGX3_CDM_RECORD_SUBSTREAM_BARRIER;
         record->length = AGX3_CDM_BARRIER_SIZE;
         record->chip = chip;
         record->data.substream_barrier.flags = substream.flags;
         record->data.substream_barrier.substream = substream.substream;
         return true;
      }

      struct agx3_cdm_barrier_info barrier;
      if (!agx3_cdm_barrier_unpack(chip, bytes, available, &barrier))
         return false;
      record->kind = AGX3_CDM_RECORD_BARRIER;
      record->length = AGX3_CDM_BARRIER_SIZE;
      record->chip = chip;
      record->data.barrier.flags = barrier.flags;
      return true;

   case AGX_CDM_BLOCK_TYPE_UNKNOWN_5: {
      struct agx3_cdm_link_padding_info padding;
      if (!agx3_cdm_link_padding_unpack(chip, bytes, available, &padding))
         return false;
      record->kind = AGX3_CDM_RECORD_LINK_PADDING;
      record->length = AGX3_CDM_LINK_PADDING_SIZE;
      record->chip = chip;
      record->data.link_padding.count = padding.count;
      return true;
   }

   case AGX_CDM_BLOCK_TYPE_STREAM_RETURN:
   case AGX_CDM_BLOCK_TYPE_UNKNOWN_6:
   default:
      return false;
   }
}
