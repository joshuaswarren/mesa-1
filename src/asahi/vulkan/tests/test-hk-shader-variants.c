/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_shader.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

int main(void)
{
   struct hk_api_shader *obj = calloc(1, sizeof(*obj) +
                                         3 * sizeof(struct hk_shader));
   CHECK(obj);
   obj->vk.stage = MESA_SHADER_FRAGMENT;
   CHECK(hk_api_num_variants(obj) == 1);
   CHECK(hk_fs_variant(obj, 4) == &obj->variants[0]);
   CHECK(hk_only_variant(obj) == &obj->variants[0]);

   obj->g17p_sample_variants = true;
   CHECK(hk_api_num_variants(obj) == 3);
   CHECK(hk_fs_variant(obj, 0) == &obj->variants[0]);
   CHECK(hk_fs_variant(obj, 1) == &obj->variants[0]);
   CHECK(hk_fs_variant(obj, 2) == &obj->variants[1]);
   CHECK(hk_fs_variant(obj, 4) == &obj->variants[2]);
   /* Changing the count must select the previous immutable binary again. */
   CHECK(hk_fs_variant(obj, 2) == &obj->variants[1]);
   CHECK(hk_fs_variant(obj, 1) == &obj->variants[0]);
   unsigned count = 0;
   hk_foreach_variant(obj, shader) {
      CHECK(shader == &obj->variants[count++]);
      CHECK(hk_variant_name(obj, shader));
   }
   CHECK(count == 3);
   CHECK(hk_fs_variant(NULL, 2) == NULL);
   free(obj);
   return 0;
}
