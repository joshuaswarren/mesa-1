/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"
#include "asahi/genxml/agx3_cmdstream.h"

#include <assert.h>
#include <stdio.h>

static unsigned jump_calls;

void *
hk_cmd_buffer_emit_cs_jump(UNUSED struct hk_cmd_buffer *cmd, void *current,
                           UNUSED uint64_t target, UNUSED bool vdm)
{
   jump_calls++;
   return current;
}

int
main(void)
{
   /* Adjacent compute roots are assembled independently while recording.
    * When their used prefixes fit in one root, merging must concatenate the
    * bytes instead of inserting a CDM stream link. The kernel's compute ABI
    * points at end - 4 as the terminator, so a linked small stream would make
    * that pointer name unused root padding rather than the real terminator. */
   static uint8_t a_root[HK_CS_ROOT_SIZE];
   static uint8_t b_root[HK_CS_ROOT_SIZE];
   memset(a_root, 0xa5, sizeof(a_root));
   memset(b_root, 0x5a, sizeof(b_root));

   struct hk_cmd_buffer cmd = {0};
   struct hk_cs a = {
      .cmd = &cmd,
      .type = HK_CS_CDM,
      .addr = 0x100000,
      .start = a_root,
      .current = a_root + 12,
      .end = a_root + sizeof(a_root),
      .chunk = {.cpu = a_root, .gpu = 0x100000},
      .stats = {.cmds = 1},
   };
   struct hk_cs b = {
      .cmd = &cmd,
      .type = HK_CS_CDM,
      .addr = 0x200000,
      .start = b_root,
      .current = b_root + 84,
      .end = b_root + sizeof(b_root),
      .chunk = {.cpu = b_root, .gpu = 0x200000},
      .strong_flush_end = b_root + 44,
      .weak_barrier_end = b_root + 84,
      .stats = {.cmds = 2},
   };

   hk_cs_merge_cdm(&a, &b);
   assert(jump_calls == 0);
   assert(!a.stream_linked);
   assert(a.current == a_root + 96);
   assert(a.end == a_root + sizeof(a_root));
   assert(a.chunk.cpu == a_root && a.chunk.gpu == 0x100000);
   assert(!memcmp(a_root + 12, b_root, 84));
   assert(a.strong_flush_end == a_root + 56);
   assert(a.weak_barrier_end == a_root + 96);
   assert(a.stats.cmds == 3 && a.stats.merged == 1);

   uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
   assert(agx3_cdm_stream_terminate_pack(AGX_CHIP_G17P, terminate));
   memcpy(a.current, terminate, sizeof(terminate));
   a.current = (uint8_t *)a.current + sizeof(terminate);
   const size_t submitted_len = (uint8_t *)a.current - (uint8_t *)a.start;
   assert(submitted_len == 100);
   assert(agx3_cdm_stream_terminate_validate(
      AGX_CHIP_G17P, a_root + submitted_len - sizeof(terminate),
      sizeof(terminate)));

   puts("G17P inline CDM merge and terminal pointer PASS");
   return 0;
}
