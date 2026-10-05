/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_shader.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "util/hash_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

static uint32_t
layout_key_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(blake3_hash));
}

static bool
layout_key_equal(const void *a, const void *b)
{
   return memcmp(a, b, sizeof(blake3_hash)) == 0;
}

static void
test_push_layout_cache_identity(void)
{
   struct hk_descriptor_set_layout *layout = calloc(1,
      sizeof(*layout) + sizeof(layout->binding[0]));
   CHECK(layout);
   layout->descriptor_generation = 9;
   layout->binding_count = 1;
   layout->non_variable_descriptor_buffer_size = 64;
   layout->binding[0] = (struct hk_descriptor_set_binding_layout){
      .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .array_size = 1, .stride = 64,
   };

   /* Exercise the production layout hasher with the same binding bytes.
    * These hashes feed vk_pipeline.c's shader-cache layout key. */
   hk_descriptor_set_layout_hash(layout);
   blake3_hash allocated;
   memcpy(allocated, layout->vk.blake3, sizeof(allocated));
   layout->vk.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
   hk_descriptor_set_layout_hash(layout);
   blake3_hash pushed;
   memcpy(pushed, layout->vk.blake3, sizeof(pushed));
   CHECK(memcmp(allocated, pushed, sizeof(allocated)) != 0);

   struct hash_table *cache = _mesa_hash_table_create(
      NULL, layout_key_hash, layout_key_equal);
   CHECK(cache);
   const bool allocated_proof = false, pushed_proof = true;
   CHECK(_mesa_hash_table_insert(cache, allocated, (void *)&allocated_proof));
   CHECK(_mesa_hash_table_search(cache, pushed) == NULL);
   CHECK(_mesa_hash_table_insert(cache, pushed, (void *)&pushed_proof));
   CHECK(_mesa_hash_table_search(cache, allocated)->data == &allocated_proof);
   CHECK(_mesa_hash_table_search(cache, pushed)->data == &pushed_proof);

   layout->vk.flags = 0;
   hk_descriptor_set_layout_hash(layout);
   CHECK(memcmp(allocated, layout->vk.blake3, sizeof(allocated)) == 0);
   CHECK(_mesa_hash_table_search(cache, layout->vk.blake3)->data == &allocated_proof);
   _mesa_hash_table_destroy(cache, NULL);
   free(layout);
}

int main(void)
{
   test_push_layout_cache_identity();
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
   puts("shader variants: layout cache identity and sample/vertex variants passed");
   return 0;
}
