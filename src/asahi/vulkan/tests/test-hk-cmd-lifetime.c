/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"

#include <assert.h>
#include <stdio.h>

static unsigned reference_calls;
static unsigned unreference_calls;

void
agx_bo_reference(struct agx_bo *bo)
{
   assert(bo != NULL);
   bo->refcnt++;
   reference_calls++;
}

void
agx_bo_unreference(UNUSED struct agx_device *dev, struct agx_bo *bo)
{
   assert(bo != NULL);
   assert(bo->refcnt > 0);
   bo->refcnt--;
   unreference_calls++;
}

int
main(void)
{
   struct hk_cmd_buffer cmd = {0};
   struct agx_device dev = {.chip = AGX_CHIP_G17P};
   struct agx_bo shader = {.refcnt = 1};
   struct agx_bo constants = {.refcnt = 1};
   cmd.large_bos = UTIL_DYNARRAY_INIT;

   hk_cmd_buffer_retain_bo(&cmd, NULL);
   assert(reference_calls == 0);
   assert(util_dynarray_num_elements(&cmd.large_bos, struct agx_bo *) == 0);

   hk_cmd_buffer_retain_bo(&cmd, &shader);
   hk_cmd_buffer_retain_bo(&cmd, &shader);
   hk_cmd_buffer_retain_bo(&cmd, &constants);

   assert(reference_calls == 2);
   assert(shader.refcnt == 2);
   assert(constants.refcnt == 2);
   assert(util_dynarray_num_elements(&cmd.large_bos, struct agx_bo *) == 2);

   struct agx_bo **retained = util_dynarray_begin(&cmd.large_bos);
   assert(retained[0] == &shader);
   assert(retained[1] == &constants);

   /* Destroy the source shader after recording. Its constants/preamble BO
    * must remain alive on the command buffer's reference until reset. */
   agx_bo_unreference(&dev, &constants);
   assert(constants.refcnt == 1);

   /* Reset releases each distinct reference, keeps the dynarray usable, and
    * final destruction releases again before finalizing its storage. */
   hk_cmd_buffer_release_bos(&cmd, &dev);
   assert(unreference_calls == 3);
   assert(shader.refcnt == 1);
   assert(constants.refcnt == 0);
   assert(util_dynarray_num_elements(&cmd.large_bos, struct agx_bo *) == 0);

   hk_cmd_buffer_retain_bo(&cmd, &shader);
   assert(reference_calls == 3);
   assert(shader.refcnt == 2);
   hk_cmd_buffer_release_bos(&cmd, &dev);
   assert(unreference_calls == 4);
   assert(shader.refcnt == 1);

   util_dynarray_fini(&cmd.large_bos);
   puts("G17P command-buffer BO retain/reset/destroy ordering PASS");
   return 0;
}
