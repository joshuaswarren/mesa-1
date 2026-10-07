/* SPDX-License-Identifier: MIT */
#include "../hk_cmd_buffer.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned reference_calls;
static unsigned unreference_calls;
static unsigned error_calls;

/* hk_cmd_buffer_retain_bo() reports allocation failure through
 * vk_command_buffer_set_error(), which reaches the runtime's __vk_errorf().
 * It does not run on the paths below; count any call so an unexpected error
 * fails the test. */
VkResult
__vk_errorf(UNUSED const void *obj, VkResult error, UNUSED const char *file,
            UNUSED int line, UNUSED const char *format, ...)
{
   error_calls++;
   return error;
}

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

   /* Enough distinct BOs to grow the retained set several times, each
    * retained twice (the second through the set, not the one-entry cache,
    * because the BOs in between evict it): one reference per distinct BO. */
   enum { MANY = 1500 };
   static struct agx_bo many[MANY];
   const unsigned refs_before = reference_calls;
   for (unsigned round = 0; round < 2; ++round) {
      for (unsigned i = 0; i < MANY; ++i) {
         if (round == 0)
            many[i].refcnt = 1;
         hk_cmd_buffer_retain_bo(&cmd, &many[i]);
      }
   }
   assert(reference_calls == refs_before + MANY);
   assert(util_dynarray_num_elements(&cmd.large_bos, struct agx_bo *) == MANY);
   for (unsigned i = 0; i < MANY; ++i)
      assert(many[i].refcnt == 2);
   hk_cmd_buffer_release_bos(&cmd, &dev);
   for (unsigned i = 0; i < MANY; ++i)
      assert(many[i].refcnt == 1);

   /* After a reset the set is empty again: a BO retained before counts
    * once more. */
   hk_cmd_buffer_retain_bo(&cmd, &many[7]);
   hk_cmd_buffer_retain_bo(&cmd, &many[7]);
   assert(many[7].refcnt == 2);
   hk_cmd_buffer_release_bos(&cmd, &dev);
   assert(many[7].refcnt == 1);
   assert(error_calls == 0);

   free(cmd.retained_set);
   util_dynarray_fini(&cmd.large_bos);
   puts("G17P command-buffer BO retain/reset/destroy ordering PASS");
   return 0;
}
