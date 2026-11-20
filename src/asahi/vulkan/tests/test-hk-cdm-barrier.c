/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"
#include "hk_entrypoints.h"

#undef NDEBUG
#include <assert.h>
#include <stdlib.h>

static uint32_t stream[4096];

static void
begin_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   *cs = (struct hk_cs){
      .cmd = cmd,
      .type = HK_CS_CDM,
      .start = stream,
      .current = stream,
      .end = stream + ARRAY_SIZE(stream),
   };
   list_inithead(&cmd->control_streams);
   list_addtail(&cs->node, &cmd->control_streams);
   cmd->current_cs.cs = cs;
}

static void
launch(struct hk_cs *cs, bool application)
{
   struct hk_cmd_buffer *cmd = cs->cmd;
   struct agx_shader_info info = {0};
   hk_dispatch_with_usc(hk_cmd_buffer_device(cmd), cs, &info, 0x40, agx_1d(4),
                        agx_workgroup(1, 1, 1),
                        application && hk_app_dispatch_elides_post_flush(cmd));
}

static bool
is_barrier(const uint32_t *word)
{
   return (*word >> 29) == AGX_CDM_BLOCK_TYPE_BARRIER;
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   assert(dev && cmd);
   dev->dev.chip = AGX_CHIP_G13X;
   cmd->vk.base.type = VK_OBJECT_TYPE_COMMAND_BUFFER;
   cmd->vk.base.device = &dev->vk;

   struct hk_cs cs;
   begin_stream(cmd, &cs);
   launch(&cs, true);
   const uint32_t *edge = cs.current;
   const size_t launch_words = edge - stream;
   launch(&cs, true);
   const uint32_t *next = cs.current;
   assert(next - edge == launch_words);
   assert(!is_barrier(edge));

   const VkMemoryBarrier2 memory = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
   };
   const VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &memory,
   };
   hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(cmd), &dependency);
   assert(cs.current == next);
   assert(cmd->current_cs.cs == &cs);
   launch(&cs, true);
   assert(is_barrier(next));
   assert((const uint32_t *)cs.current - next == launch_words + 1);

   const uint32_t *tail = cs.current;
   hk_cmd_buffer_end_compute(cmd);
   assert(is_barrier(tail));
   assert((const uint32_t *)cs.current == tail + 1);

   begin_stream(cmd, &cs);
   for (unsigned i = 0; i < 2; i++) {
      launch(&cs, false);
      assert(is_barrier((const uint32_t *)cs.current - 1));
   }

   cmd->in_meta = true;
   launch(&cs, true);
   assert(is_barrier((const uint32_t *)cs.current - 1));
   cmd->in_meta = false;
   dev->perftest = HK_PERF_ALWAYSCDMBARRIER;
   tail = cs.current;
   launch(&cs, true);
   assert(is_barrier(tail + launch_words));
   tail = cs.current;
   hk_cmd_buffer_end_compute(cmd);
   assert(cs.current == tail);

   free(cmd);
   free(dev);
   return 0;
}
