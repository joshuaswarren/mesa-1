/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"
#include "hk_device.h"

/* Flush unflushed dependency-tracked CDM work (cmd->state.cs.cdm_dirty) at a
 * visibility boundary: a vkCmdPipelineBarrier2, the end of the stream's
 * batch, or hk_EndCommandBuffer. No-op when the command buffer holds no
 * unflushed launches or the stream is not the compute stream they went to.
 */
void
hk_cdm_flush_dirty(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   if (!cs || !cmd->state.cs.cdm_dirty || cs->type != HK_CS_CDM)
      return;

   hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
   hk_cdm_cache_flush(hk_cmd_buffer_device(cmd), cs);
}

void
hk_cdm_flush_all_dirty(struct hk_cmd_buffer *cmd)
{
   if (!cmd->state.cs.cdm_dirty)
      return;

   if (cmd->current_cs.cs)
      hk_cdm_flush_dirty(cmd, cmd->current_cs.cs);

   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
      if (cs != cmd->current_cs.cs)
         hk_cdm_flush_dirty(cmd, cs);
   }

   cmd->state.cs.cdm_dirty = false;
}
