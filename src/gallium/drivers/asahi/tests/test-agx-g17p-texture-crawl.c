/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "agx_state.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>

/*
 * Two predicates have to agree about which texel fetches need a descriptor.
 *
 * agx_nir_lower_texture() predicates a G17P integer fetch explicitly: it reads
 * the texture's size, and its level count when the level is not zero, so it can
 * select the robustness zero after the fetch. Both of those are texture crawls,
 * and a crawl is lowered through the texture's descriptor -- which only exists
 * when the fetch was made bindless first.
 *
 * Making it bindless is agx_nir_lower_bindings()' job, and it decides from its
 * own copy of the same condition. The two drifted: the fetch lowering takes a
 * two- or three-dimensional fetch, the binding lowering only a two-dimensional
 * one. A texelFetch from a sampler3D therefore reached the crawl with no
 * descriptor to crawl, and the compiler asserted "must be bindless".
 *
 * KHR-GL33.texture_size_promotion.functional is the conformance case: it
 * fetches from every sampler type in turn, sampler3D among them. The assert
 * aborts the process, so it takes the whole conformance session with it.
 *
 * Run the driver's own sequence over a fetch of each shape and check that
 * every fetch the lowering will crawl carries a handle by the time it gets
 * there.
 */

struct shape {
   const char *name;
   enum glsl_sampler_dim dim;
   bool array;
   unsigned coord_components;
};

static const struct shape shapes[] = {
   {"sampler1D", GLSL_SAMPLER_DIM_1D, false, 1},
   {"sampler2D", GLSL_SAMPLER_DIM_2D, false, 2},
   {"sampler3D", GLSL_SAMPLER_DIM_3D, false, 3},
   {"sampler2DRect", GLSL_SAMPLER_DIM_RECT, false, 2},
   {"sampler1DArray", GLSL_SAMPLER_DIM_1D, true, 2},
   {"sampler2DArray", GLSL_SAMPLER_DIM_2D, true, 3},
};

/* texelFetch(sampler, coord, 0) -- the shape the conformance case compiles. */
static nir_shader *
build(const struct shape *shape)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "texel fetch");

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = shape->dim;
   tex->is_array = shape->array;
   tex->coord_components = shape->coord_components;
   tex->dest_type = nir_type_float32;
   tex->texture_index = 0;
   tex->sampler_index = 0;

   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_imm_zero(&b, shape->coord_components, 32));
   tex->src[1] =
      nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, 0));

   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   nir_store_output(&b, &tex->def, nir_imm_int(&b, 0),
                    .io_semantics.location = FRAG_RESULT_DATA0,
                    .write_mask = 0xf, .src_type = nir_type_float32);

   nir_shader *shader = b.shader;
   shader->info.num_textures = 1;
   BITSET_SET(shader->info.textures_used, 0);
   BITSET_SET(shader->info.samplers_used, 0);
   nir_shader_gather_info(shader, nir_shader_get_entrypoint(shader));
   return shader;
}

/* The fetch lowering's own condition, kept here so the test states what it is
 * checking rather than restating a number. */
static bool
lowering_will_crawl(const nir_tex_instr *tex)
{
   return tex->op == nir_texop_txf &&
          (tex->sampler_dim == GLSL_SAMPLER_DIM_2D ||
           tex->sampler_dim == GLSL_SAMPLER_DIM_3D) &&
          nir_tex_instr_src_index((nir_tex_instr *)tex,
                                  nir_tex_src_min_lod) < 0 &&
          !(tex->backend_flags & AGX_TEXTURE_FLAG_NO_CLAMP);
}

static bool
check(const struct shape *shape)
{
   nir_shader *shader = build(shape);

   bool uses_bindless_samplers = false;
   struct agx_lower_bindings_options options = {
      .uses_bindless_samplers = &uses_bindless_samplers,
      .g17p_plain_txf_crawl = true,
      .g17p_native_sampler_handles = true,
      .bindful_sampler_count = 0,
   };
   NIR_PASS(_, shader, agx_nir_lower_bindings, &options);

   bool crawled = false, bindless = true;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (!lowering_will_crawl(tex))
               continue;

            crawled = true;
            bindless &= nir_tex_instr_src_index(
                           tex, nir_tex_src_texture_handle) >= 0;
         }
      }
   }

   /* Say what happened before going any further: the crawl lowering below
    * asserts rather than returns on a fetch with no descriptor, so running it
    * first would take the report down with it. */
   const bool ok = !crawled || bindless;
   printf("%-16s %-9s %s\n", shape->name, crawled ? "crawled" : "plain",
          ok ? "ok" : "NO DESCRIPTOR");

   /* And the sequence itself must survive. */
   if (ok) {
      NIR_PASS(_, shader, agx_nir_lower_texture_for_generation,
               AGX_DESCRIPTOR_GENERATION_9);
      NIR_PASS(_, shader, agx_nir_rebind_g17p_texture_crawls);
      nir_validate_shader(shader, "after the G17P texture sequence");
   }

   ralloc_free(shader);
   return ok;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool ok = true;

   for (unsigned i = 0; i < ARRAY_SIZE(shapes); ++i)
      ok &= check(&shapes[i]);

   glsl_type_singleton_decref();
   return ok ? 0 : 1;
}
