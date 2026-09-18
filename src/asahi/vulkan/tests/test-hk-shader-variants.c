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

   /* G17P sparse vertex output selection. A position-writing vertex shader
    * feeding a fragment shader stays on the dense terminal protocol. */
   struct shader_info info = {
      .stage = MESA_SHADER_VERTEX,
      .next_stage = MESA_SHADER_FRAGMENT,
      .outputs_written = VARYING_BIT_POS | VARYING_BIT_VAR(0),
   };
   CHECK(!hk_g17p_sparse_vertex_outputs(&info, MESA_SHADER_VERTEX, false));

   /* A vertex shader that writes no gl_Position bases its outputs at the
    * reserved UVS position group, which the dense run cannot admit. */
   info.outputs_written = VARYING_BIT_VAR(0);
   CHECK(hk_g17p_sparse_vertex_outputs(&info, MESA_SHADER_VERTEX, false));

   /* The pre-existing sparse shapes are unchanged. */
   info.outputs_written = VARYING_BIT_POS | VARYING_BIT_VAR(0);
   CHECK(hk_g17p_sparse_vertex_outputs(&info, MESA_SHADER_VERTEX, true));
   info.has_transform_feedback_varyings = true;
   CHECK(hk_g17p_sparse_vertex_outputs(&info, MESA_SHADER_VERTEX, false));
   info.has_transform_feedback_varyings = false;
   info.next_stage = MESA_SHADER_TESS_CTRL;
   CHECK(hk_g17p_sparse_vertex_outputs(&info, MESA_SHADER_VERTEX, false));
   info.next_stage = MESA_SHADER_GEOMETRY;
   CHECK(hk_g17p_sparse_vertex_outputs(&info, MESA_SHADER_VERTEX, false));
   info.next_stage = MESA_SHADER_FRAGMENT;
   CHECK(hk_g17p_sparse_vertex_outputs(&info, MESA_SHADER_GEOMETRY, false));
   CHECK(hk_g17p_sparse_vertex_outputs(&info, MESA_SHADER_TESS_EVAL, false));

   /* Only vertex programs use this protocol; a positionless fragment shader
    * must not be dragged into it. */
   info.stage = MESA_SHADER_FRAGMENT;
   info.outputs_written = 0;
   CHECK(!hk_g17p_sparse_vertex_outputs(&info, MESA_SHADER_FRAGMENT, false));
   return 0;
}
