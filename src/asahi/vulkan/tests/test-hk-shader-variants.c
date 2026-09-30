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
   const struct hk_ordinary_descriptor_binding allocated_proof = {.push = false};
   const struct hk_ordinary_descriptor_binding pushed_proof = {.push = true};
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

static void
test_ordinary_descriptors(void)
{
   struct hk_api_shader *obj = calloc(1,
      hk_api_shader_size(MESA_SHADER_FRAGMENT, false, true));
   CHECK(obj);
   obj->vk.stage = MESA_SHADER_FRAGMENT;
   obj->g17p_ordinary_variants = true;
   CHECK(hk_api_num_variants(obj) == 2);
   CHECK(hk_fs_variant(obj, 1) == &obj->variants[0]);
   CHECK(strcmp(hk_variant_name(obj, &obj->variants[1]), "Ordinary descriptors") == 0);
   struct hk_ordinary_descriptor_requirements *proof = hk_api_ordinary_requirements(obj);
   *proof = (struct hk_ordinary_descriptor_requirements){
      .count = 1, .set_mask = 1, .bindings = {{.set = 0, .binding = 1, .offset = 16}},
   };
   struct hk_descriptor_set_layout *layout = calloc(1,
      sizeof(*layout) + 2 * sizeof(layout->binding[0]));
   CHECK(layout);
   layout->descriptor_generation = 9;
   layout->binding_count = 2;
   layout->binding[1] = (struct hk_descriptor_set_binding_layout){
      .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .array_size = 1, .offset = 16,
   };
   uint8_t data[128] = {0};
   const struct hk_sampled_image_descriptor_layout native = hk_sampled_image_descriptor_layout(9);
   unsigned border_offset = 16 + native.clamp_0_sampler_index;
   unsigned scaled_offset = 16 + 32 + offsetof(struct hk_sampled_image_descriptor_sw, scaled_conversion);
   uint16_t scaled = 0;
   int16_t border = -1;
   memcpy(data + border_offset, &border, 2);
   struct hk_descriptor_set set = {.layout = layout, .mapped_ptr = data, .addr = 0x1000, .size = sizeof(data)};
   struct hk_descriptor_set *sets[] = {&set};
   uint64_t addresses[] = {set.addr};
   CHECK(hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   struct hk_ordinary_fs_selection cache = {0};
   bool changed;
   CHECK(hk_ordinary_descriptor_select(&cache, obj, sets, addresses, 1, 1, 1, &changed) == &obj->variants[1]);
   CHECK(changed);
   /* An exact cache hit must not walk descriptors again. */
   CHECK(hk_ordinary_descriptor_select(&cache, obj, NULL, NULL, 0, 1, 1, &changed) == &obj->variants[1]);
   CHECK(!changed);
   for (unsigned kind = 1; kind < 65536; ++kind) {
      scaled = kind;
      memcpy(data + scaled_offset, &scaled, 2);
      CHECK(!hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   }
   CHECK(hk_ordinary_descriptor_select(&cache, obj, sets, addresses, 1, 2, 1, &changed) == &obj->variants[0]);
   CHECK(changed);
   scaled = 0;
   memcpy(data + scaled_offset, &scaled, 2);
   for (unsigned value = 0; value < 65536; ++value) {
      uint16_t raw = value;
      memcpy(data + border_offset, &raw, 2);
      CHECK(hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1) == (value >= 32768));
   }
   CHECK(hk_ordinary_descriptor_select(&cache, obj, sets, addresses, 1, 3, 1, &changed) == &obj->variants[1]);
   CHECK(changed);
   const VkDescriptorBindingFlags flags[] = {
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
      VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT,
      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
      VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(flags); ++i) {
      layout->binding[1].flags = flags[i];
      CHECK(!hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   }
   layout->binding[1].flags = 0;
   layout->vk.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
   CHECK(!hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   layout->vk.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
   CHECK(!hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   layout->vk.flags = 0;
   layout->descriptor_generation = 8;
   CHECK(!hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   layout->descriptor_generation = 9;
   layout->binding[1].array_size = 2;
   CHECK(!hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   layout->binding[1].array_size = 1;
   layout->binding[1].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
   CHECK(!hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   layout->binding[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   layout->binding[1].offset++;
   CHECK(!hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   layout->binding[1].offset--;
   set.size = scaled_offset + 1;
   CHECK(!hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   set.size = sizeof(data);
   addresses[0]++;
   CHECK(!hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   addresses[0]--;
   proof->bindings[0].offset = UINT32_MAX;
   layout->binding[1].offset = UINT32_MAX;
   CHECK(!hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   proof->bindings[0].offset = layout->binding[1].offset = 16;
   proof->bindings[0].binding = 2;
   CHECK(!hk_ordinary_descriptor_sets_match(proof, sets, addresses, 1));
   proof->bindings[0].binding = 1;
   sets[0] = NULL;
   CHECK(hk_ordinary_descriptor_select(&cache, obj, sets, addresses, 1, 4, 1, &changed) == &obj->variants[0]);
   CHECK(changed);
   sets[0] = &set;
   /* Restoring an earlier descriptor generation must re-evaluate it. */
   CHECK(hk_ordinary_descriptor_select(&cache, obj, sets, addresses, 1, 1, 1, &changed) == &obj->variants[1]);
   CHECK(changed);

   struct hk_api_shader *both = calloc(1,
      hk_api_shader_size(MESA_SHADER_FRAGMENT, true, true));
   CHECK(both);
   both->vk.stage = MESA_SHADER_FRAGMENT;
   both->g17p_sample_variants = true;
   both->g17p_ordinary_variants = true;
   *hk_api_ordinary_requirements(both) = *proof;
   CHECK(hk_api_num_variants(both) == 6);
   CHECK(hk_fs_variant(both, 1) == &both->variants[0]);
   CHECK(hk_fs_variant(both, 2) == &both->variants[1]);
   CHECK(hk_fs_variant(both, 4) == &both->variants[2]);
   CHECK(strcmp(hk_variant_name(both, &both->variants[4]),
                "2 samples, ordinary descriptors") == 0);
   struct hk_ordinary_fs_selection combined = {0};
   CHECK(hk_ordinary_descriptor_select(&combined, both, sets, addresses,
         1, 10, 1, &changed) == &both->variants[3]);
   CHECK(changed);
   CHECK(hk_ordinary_descriptor_select(&combined, both, sets, addresses,
         1, 10, 2, &changed) == &both->variants[4]);
   CHECK(changed);
   CHECK(hk_ordinary_descriptor_select(&combined, both, sets, addresses,
         1, 10, 4, &changed) == &both->variants[5]);
   CHECK(changed);
   CHECK(hk_ordinary_descriptor_select(&combined, both, NULL, NULL,
         0, 10, 4, &changed) == &both->variants[5]);
   CHECK(!changed);
   scaled = 1;
   memcpy(data + scaled_offset, &scaled, 2);
   CHECK(hk_ordinary_descriptor_select(&combined, both, sets, addresses,
         1, 11, 4, &changed) == &both->variants[2]);
   CHECK(changed);
   CHECK(hk_ordinary_descriptor_select(&combined, both, sets, addresses,
         1, 11, 2, &changed) == &both->variants[1]);
   CHECK(changed);
   CHECK(hk_ordinary_descriptor_select(&combined, both, sets, addresses,
         1, 11, 1, &changed) == &both->variants[0]);
   CHECK(changed);
   scaled = 0;
   memcpy(data + scaled_offset, &scaled, 2);
   CHECK(hk_ordinary_descriptor_select(&combined, both, sets, addresses,
         1, 12, 1, &changed) == &both->variants[3]);
   CHECK(changed);
   free(both);
   free(layout);
   free(obj);
}

static void
test_push_ordinary_descriptors(void)
{
   struct hk_api_shader *obj = calloc(1,
      hk_api_shader_size(MESA_SHADER_FRAGMENT, false, true));
   struct hk_push_descriptor_set *push = calloc(1, sizeof(*push));
   CHECK(obj && push);
   obj->vk.stage = MESA_SHADER_FRAGMENT;
   obj->g17p_ordinary_variants = true;
   struct hk_ordinary_descriptor_requirements *proof =
      hk_api_ordinary_requirements(obj);
   *proof = (struct hk_ordinary_descriptor_requirements){
      .count = 1, .set_mask = 1,
      .bindings = {{.set = 0, .binding = 1, .offset = 16, .push = true}},
   };
   for (unsigned i = 0; i < sizeof(blake3_hash); ++i)
      proof->bindings[0].layout_hash[i] = (uint8_t)(i + 1);
   memcpy(push->layout_hash, proof->bindings[0].layout_hash,
          sizeof(blake3_hash));
   push->layout_flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
   const struct hk_sampled_image_descriptor_layout native =
      hk_sampled_image_descriptor_layout(9);
   unsigned scaled_offset = 16 + 32 +
      offsetof(struct hk_sampled_image_descriptor_sw, scaled_conversion);
   unsigned border_offset = 16 + native.clamp_0_sampler_index;
   int16_t border = -1;
   memcpy(push->data + border_offset, &border, sizeof(border));
   struct hk_descriptor_set *sets[] = {NULL};
   uint64_t addresses[] = {0};
   struct hk_push_descriptor_set *push_sets[] = {push};
   struct hk_ordinary_fs_selection cache = {0};
   bool changed;
   CHECK(hk_ordinary_descriptor_select_push(&cache, obj, sets, addresses,
         push_sets, 1, 1, 1, 1, &changed) == &obj->variants[1]);
   CHECK(changed);
   uint16_t scaled = 1;
   memcpy(push->data + scaled_offset, &scaled, sizeof(scaled));
   CHECK(hk_ordinary_descriptor_select_push(&cache, obj, sets, addresses,
         push_sets, 1, 1, 2, 1, &changed) == &obj->variants[0]);
   CHECK(changed);
   scaled = 0;
   memcpy(push->data + scaled_offset, &scaled, sizeof(scaled));
   CHECK(hk_ordinary_descriptor_select_push(&cache, obj, sets, addresses,
         push_sets, 1, 1, 3, 1, &changed) == &obj->variants[1]);
   border = 0;
   memcpy(push->data + border_offset, &border, sizeof(border));
   CHECK(hk_ordinary_descriptor_select_push(&cache, obj, sets, addresses,
         push_sets, 1, 1, 4, 1, &changed) == &obj->variants[0]);
   border = -1;
   memcpy(push->data + border_offset, &border, sizeof(border));
   push->layout_hash[0] ^= 1;
   CHECK(hk_ordinary_descriptor_select_push(&cache, obj, sets, addresses,
         push_sets, 1, 1, 5, 1, &changed) == &obj->variants[0]);
   push->layout_hash[0] ^= 1;
   push->layout_flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
   CHECK(hk_ordinary_descriptor_select_push(&cache, obj, sets, addresses,
         push_sets, 1, 1, 6, 1, &changed) == &obj->variants[0]);
   push->layout_flags &= ~VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
   CHECK(hk_ordinary_descriptor_select_push(&cache, obj, sets, addresses,
         push_sets, 0, 1, 7, 1, &changed) == &obj->variants[0]);
   CHECK(hk_ordinary_descriptor_select_push(&cache, obj, sets, addresses,
         push_sets, 1, 1, 8, 1, &changed) == &obj->variants[1]);
   free(push);
   free(obj);
}

int main(void)
{
   test_push_layout_cache_identity();
   test_ordinary_descriptors();
   test_push_ordinary_descriptors();
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
   puts("shader variants: ordinary descriptor truth tables, fallback/restore/cache transitions and sample/vertex variants passed");
   return 0;
}
