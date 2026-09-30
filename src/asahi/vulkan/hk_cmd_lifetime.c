/* SPDX-License-Identifier: MIT */
#include "hk_cmd_buffer.h"

static bool
hk_g17p_ioto_quarantine_enabled(const struct agx_device *dev)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = HK_ENV_BOOL_ONCE("HK_G17P_IOTO_QUARANTINE", false);

   return dev->chip == AGX_CHIP_G17P && enabled;
}

void
hk_cmd_buffer_retain_bo(struct hk_cmd_buffer *cmd, struct agx_bo *bo)
{
   if (!bo)
      return;

   /* Pointer identity is sufficient: the retained reference prevents reuse
    * until reset. Mix two address regions so adjacent allocator objects do not
    * collapse into one direct-mapped slot. The table size is a power of two. */
   const uintptr_t key = (uintptr_t)bo;
   const unsigned cache_index =
      ((key >> 4) ^ (key >> 12)) & (ARRAY_SIZE(cmd->recent_retained_bos) - 1);
   if (cmd->recent_retained_bos[cache_index] == bo)
      return;

   util_dynarray_foreach(&cmd->large_bos, struct agx_bo *, retained) {
      if (*retained == bo) {
         cmd->recent_retained_bos[cache_index] = bo;
         return;
      }
   }

   agx_bo_reference(bo);
   util_dynarray_append(&cmd->large_bos, bo);
   cmd->recent_retained_bos[cache_index] = bo;
}

void
hk_cmd_buffer_release_bos(struct hk_cmd_buffer *cmd, struct agx_device *dev)
{
   util_dynarray_foreach(&cmd->large_bos, struct agx_bo *, bo) {
      if (hk_g17p_ioto_quarantine_enabled(dev) && (*bo)->label &&
          !strcmp((*bo)->label, "G17P Ioto program arena")) {
         struct hk_device *hk_dev = hk_cmd_buffer_device(cmd);
         simple_mtx_lock(&hk_dev->vdm.lock);
         util_dynarray_append(&hk_dev->vdm.quarantined_ioto_bos, *bo);
         simple_mtx_unlock(&hk_dev->vdm.lock);
         continue;
      }
      agx_bo_unreference(dev, *bo);
   }

   util_dynarray_clear(&cmd->large_bos);
   memset(cmd->recent_retained_bos, 0, sizeof(cmd->recent_retained_bos));
}
