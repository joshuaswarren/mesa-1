/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020 Collabora Ltd.
 * Copyright 2016 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "asahi/clc/asahi_clc.h"
#include "asahi/isa/disasm.h"
#include "asahi/layout/layout.h"
#include "asahi/lib/agx_abi.h"
#include "compiler/nir/nir_builder.h"
#include "util/bitset.h"
#include "util/glheader.h"
#include "util/half_float.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/u_debug.h"
#include "util/u_dynarray.h"
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "agx_opcodes.h"
#include "agx3_pack.h"
#include "glsl_types.h"
#include "nir.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "shader_enums.h"

/* Cache-line align shader programs. This matches the prop compiler. */
#define AGX_CODE_ALIGN 0x80

/* clang-format off */
static const struct debug_named_value agx_debug_options[] = {
   {"shaders",   AGX_DBG_SHADERS,	"Dump shaders in NIR and AIR"},
   {"shaderdb",  AGX_DBG_SHADERDB,	"Print statistics"},
   {"verbose",   AGX_DBG_VERBOSE,	"Disassemble verbosely"},
   {"internal",  AGX_DBG_INTERNAL,	"Dump even internal shaders"},
   {"novalidate",AGX_DBG_NOVALIDATE,"Skip IR validation in debug builds"},
   {"noopt",     AGX_DBG_NOOPT,     "Disable backend optimizations"},
   {"wait",      AGX_DBG_WAIT,      "Wait after all async instructions"},
   {"nopreamble",AGX_DBG_NOPREAMBLE,"Do not use shader preambles"},
   {"demand",    AGX_DBG_DEMAND,    "Bound tightly to register demand"},
   {"nosched",   AGX_DBG_NOSCHED,   "Do not schedule the shader"},
   {"spill",     AGX_DBG_SPILL,     "Spill (almost) everything"},
   {"nopromote", AGX_DBG_NOPROMOTE, "Do not promote constants to uniforms"},
   {"refuse",    AGX_DBG_REFUSE,    "Name refused instructions, without dumping shaders"},
   DEBUG_NAMED_VALUE_END
};
/* clang-format on */

DEBUG_GET_ONCE_FLAGS_OPTION(agx_compiler_debug, "AGX_MESA_DEBUG",
                            agx_debug_options, 0)

int agx_compiler_debug = 0;

/*
 * Pad binary to a given alignment and return aligned offset into the binary.
 */
static unsigned
agx_pad_binary(struct util_dynarray *dyn, uint32_t align)
{
   if (dyn->size % align) {
      unsigned ngrow = align - (dyn->size % align);
      memset(util_dynarray_grow_bytes(dyn, ngrow, 1), 0, ngrow);
   }

   assert((dyn->size % align) == 0);
   return dyn->size;
}

uint64_t
agx_get_compiler_debug(void)
{
   return debug_get_option_agx_compiler_debug();
}

static agx_index
agx_cached_preload(agx_context *ctx, unsigned base, enum agx_size size)
{
   if (agx_is_null(ctx->preloaded[base])) {
      agx_block *block = agx_start_block(ctx);
      agx_builder b = agx_init_builder(ctx, agx_before_block(block));
      ctx->preloaded[base] = agx_preload(&b, agx_register(base, size));
   }

   return ctx->preloaded[base];
}

static agx_index
agx_tess_coord_x(agx_builder *b)
{
   return agx_cached_preload(b->shader, 4, AGX_SIZE_32);
}

static agx_index
agx_tess_coord_y(agx_builder *b)
{
   return agx_cached_preload(b->shader, 6, AGX_SIZE_32);
}

static agx_index
agx_vertex_id_zero_base(agx_builder *b)
{
   return agx_cached_preload(b->shader, AGX_ABI_VIN_VERTEX_ID_ZERO_BASE,
                             AGX_SIZE_32);
}

static agx_index
agx_vertex_id(agx_builder *b)
{
   return agx_cached_preload(b->shader, AGX_ABI_VIN_VERTEX_ID, AGX_SIZE_32);
}

static agx_index
agx_instance_id(agx_builder *b)
{
   return agx_cached_preload(b->shader, AGX_ABI_VIN_INSTANCE_ID, AGX_SIZE_32);
}

#define VARYING_NUM_COMPONENTS (VARYING_SLOT_MAX * 4)

struct coefficient_info {
   BITSET_DECLARE(smooth, VARYING_NUM_COMPONENTS);
   BITSET_DECLARE(flat, VARYING_NUM_COMPONENTS);
   BITSET_DECLARE(noperspective, VARYING_NUM_COMPONENTS);
};

static BITSET_WORD *
bitset_for_interp(struct coefficient_info *info, enum glsl_interp_mode mode)
{
   /* clang-format off */
   switch (mode) {
   case INTERP_MODE_NONE:
   case INTERP_MODE_SMOOTH:         return info->smooth;
   case INTERP_MODE_NOPERSPECTIVE:  return info->noperspective;
   case INTERP_MODE_FLAT:           return info->flat;
   default:                         UNREACHABLE("invalid interp mode");
   }
   /* clang-format on */
}

static bool
gather_cf(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   /* First handle frag coord loads */
   struct coefficient_info *info = data;
   if (intr->intrinsic == nir_intrinsic_load_frag_coord_z) {
      BITSET_SET(info->noperspective, VARYING_SLOT_POS + 2);
      return false;
   }
   if (intr->intrinsic == nir_intrinsic_load_frag_coord_w) {
      BITSET_SET(info->noperspective, VARYING_SLOT_POS + 3);
      return false;
   }

   /* Look for input loads and grab the instruction with the interp mode */
   nir_intrinsic_instr *bary;
   unsigned nr = 1;

   if (intr->intrinsic == nir_intrinsic_load_coefficients_agx) {
      bary = intr;
      /* Always load a scalar */
   } else if (intr->intrinsic == nir_intrinsic_load_interpolated_input) {
      bary = nir_src_as_intrinsic(intr->src[0]);
      nr = intr->num_components;

      /* Perspective interpolation internally reads W */
      if (nir_intrinsic_interp_mode(bary) != INTERP_MODE_NOPERSPECTIVE)
         BITSET_SET(info->noperspective, VARYING_SLOT_POS + 3);
   } else {
      return false;
   }

   BITSET_WORD *set = bitset_for_interp(data, nir_intrinsic_interp_mode(bary));
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_src *offset = nir_get_io_offset_src(intr);

   /* Mark the exact range for direct loads to minimize CF registers, but mark a
    * conservative bounding range for indirect array access.
    */
   if (nir_src_is_const(*offset)) {
      unsigned location = sem.location + nir_src_as_uint(*offset);
      unsigned start_comp = (location * 4) + nir_intrinsic_component(intr);

      BITSET_SET_COUNT(set, start_comp, nr);
   } else {
      unsigned start_comp = (sem.location * 4) + nir_intrinsic_component(intr);
      bool compact = sem.location == VARYING_SLOT_CLIP_DIST0 ||
                     sem.location == VARYING_SLOT_CLIP_DIST1;
      unsigned stride = compact ? 1 : 4;

      /* For now we have to assign CF for the whole vec4 to make indirect
       * indexiing work. This could be optimized later.
       */
      nr = stride;

      for (unsigned i = 0; i < sem.num_slots; ++i) {
         BITSET_SET_COUNT(set, start_comp + (i * stride), nr);
      }
   }

   return false;
}

/*
 * We assign all coefficient registers up front to ensure we have a consistent
 * layout required for indirects to work.
 */
static void
assign_coefficient_regs(nir_shader *nir, struct agx_varyings_fs *var)
{
   struct coefficient_info info = {0};
   nir_shader_intrinsics_pass(nir, gather_cf, nir_metadata_all, &info);

   /* W */
   if (BITSET_TEST(info.noperspective, VARYING_SLOT_POS + 3)) {
      var->bindings[var->nr_bindings++] = (struct agx_cf_binding){
         .cf_base = var->nr_cf++,
         .slot = VARYING_SLOT_POS,
         .offset = 3,
         .count = 1,
         .smooth = true,
      };
   }

   /* Z */
   if (BITSET_TEST(info.noperspective, VARYING_SLOT_POS + 2)) {
      var->bindings[var->nr_bindings++] = (struct agx_cf_binding){
         .cf_base = var->nr_cf++,
         .slot = VARYING_SLOT_POS,
         .offset = 2,
         .count = 1,
         .smooth = true,
      };

      var->reads_z = true;
   }

   static_assert(VARYING_SLOT_POS == 0, "special and handled first");

   for (unsigned i = VARYING_SLOT_POS + 1; i < VARYING_SLOT_MAX; ++i) {
      bool smooth = BITSET_TEST_COUNT(info.smooth, i * 4, 4);
      bool flat = BITSET_TEST_COUNT(info.flat, i * 4, 4);
      bool noperspective = BITSET_TEST_COUNT(info.noperspective, i * 4, 4);

      if (!(smooth || flat || noperspective))
         continue;

      /* From the GLSL 4.60 spec ("Input Layout Qualifiers"):
       *
       *    when location aliasing, the aliases sharing the location must have
       *    the same underlying numerical type and bit width (floating-point or
       *    integer, 32-bit versus 64-bit, etc.) and the same auxiliary storage
       *    and interpolation qualification.
       *
       * SPIR-V should obey this as well although the spec text is muddier.
       */
      assert((smooth + flat + noperspective) == 1 &&
             "slots must have consistent interpolation");

      BITSET_WORD *set = smooth ? info.smooth
                         : flat ? info.flat
                                : info.noperspective;

      /* Find the start offset */
      unsigned offset = 0;
      for (offset = 0; offset < 4 && !BITSET_TEST(set, (i * 4) + offset);
           ++offset)
         ;

      /* Find the end offset. TODO: Do we ever need to split into two bindings
       * to handle e.g. x_zw read masks?
       */
      unsigned count = 0;
      for (unsigned c = offset; c < 4; ++c) {
         if (BITSET_TEST(set, (i * 4) + c))
            count = c - offset + 1;
      }
      assert(count >= 1 && (count + offset) <= 4);

      var->bindings[var->nr_bindings++] = (struct agx_cf_binding){
         .cf_base = var->nr_cf,
         .slot = i,
         .offset = offset,
         .count = count,
         .smooth = !flat,
         .perspective = smooth,
      };

      var->nr_cf += count;
   }
}

static agx_index
agx_get_cf(agx_context *ctx, gl_varying_slot slot, unsigned offset)
{
   struct agx_varyings_fs *varyings = &ctx->out->varyings.fs;

   /* We already have an appropriate binding, find it */
   for (unsigned b = 0; b < varyings->nr_bindings; ++b) {
      if (varyings->bindings[b].slot == slot &&
          (slot != VARYING_SLOT_POS ||
           offset == varyings->bindings[b].offset)) {

         signed cf_offset = offset - varyings->bindings[b].offset;
         assert(cf_offset >= 0);

         return agx_immediate(varyings->bindings[b].cf_base + cf_offset);
      }
   }

   UNREACHABLE("all coefficient registers preassigned");
}

/* Builds a 64-bit hash table key for an index */
static uint64_t
agx_index_to_key(agx_index idx)
{
   STATIC_ASSERT(sizeof(idx) <= sizeof(uint64_t));

   uint64_t key = 0;
   memcpy(&key, &idx, sizeof(idx));
   return key;
}

/*
 * Extract a single channel out of a vector source. We split vectors with
 * p_split so we can use the split components directly, without emitting a
 * machine instruction. This has advantages of RA, as the split can usually be
 * optimized away.
 */
static agx_index
agx_emit_extract(agx_builder *b, agx_index vec, unsigned channel)
{
   agx_index *components = _mesa_hash_table_u64_search(b->shader->allocated_vec,
                                                       agx_index_to_key(vec));

   assert(components != NULL && "missing agx_emit_collect_to");

   return components[channel];
}

static agx_index
agx_extract_nir_src(agx_builder *b, nir_src src, unsigned channel)
{
   agx_index idx = agx_src_index(&src);

   /* We only deal with scalars, extract a single scalar if needed */
   if (nir_src_num_components(src) > 1)
      return agx_emit_extract(b, idx, channel);
   else
      return idx;
}

static void
agx_cache_collect(agx_builder *b, agx_index dst, unsigned nr_srcs,
                  agx_index *srcs)
{
   /* Lifetime of a hash table entry has to be at least as long as the table */
   agx_index *channels = ralloc_array(b->shader, agx_index, nr_srcs);

   for (unsigned i = 0; i < nr_srcs; ++i)
      channels[i] = srcs[i];

   _mesa_hash_table_u64_insert(b->shader->allocated_vec, agx_index_to_key(dst),
                               channels);
}

/*
 * Combine multiple scalars into a vector destination. This corresponds to
 * collect, lowered to moves (a shuffle in general) after register allocation.
 *
 * To optimize vector extractions, we record the individual channels
 */
static agx_instr *
agx_emit_collect_to(agx_builder *b, agx_index dst, unsigned nr_srcs,
                    agx_index *srcs)
{
   agx_cache_collect(b, dst, nr_srcs, srcs);

   if (nr_srcs == 1)
      return agx_mov_to(b, dst, srcs[0]);

   agx_instr *I = agx_collect_to(b, dst, nr_srcs);

   agx_foreach_src(I, s)
      I->src[s] = srcs[s];

   return I;
}

static agx_index
agx_emit_collect(agx_builder *b, unsigned nr_srcs, agx_index *srcs)
{
   agx_index dst = agx_vec_temp(b->shader, srcs[0].size, nr_srcs);
   agx_emit_collect_to(b, dst, nr_srcs, srcs);
   return dst;
}

static agx_index
agx_vec2(agx_builder *b, agx_index s0, agx_index s1)
{
   return agx_emit_collect(b, 2, (agx_index[]){s0, s1});
}

static agx_index
agx_pad_to_32(agx_builder *b, agx_index s)
{
   assert(s.size == AGX_SIZE_16);
   assert(agx_channels(s) == 1);

   agx_index srcs[2] = {s, agx_undef(AGX_SIZE_16)};
   agx_index dst = agx_vec_temp(b->shader, AGX_SIZE_32, 1);
   agx_emit_collect_to(b, dst, 2, srcs);
   return dst;
}

static agx_index
agx_recollect_vector(agx_builder *b, nir_src vec)
{
   agx_index comps[4];
   unsigned nr = nir_src_num_components(vec);

   for (unsigned i = 0; i < nr; ++i)
      comps[i] = agx_extract_nir_src(b, vec, i);

   return agx_emit_collect(b, nr, comps);
}

/*
 * Extract the lower or upper N-bits from a (2*N)-bit quantity. We use a split
 * without null destinations to let us CSE (and coalesce) the splits when both x
 * and y are split.
 */
static agx_instr *
agx_subdivide_to(agx_builder *b, agx_index dst, agx_index s0, unsigned comp)
{
   assert((s0.size == (dst.size + 1)) && "only 2x subdivide handled");
   assert((comp == 0 || comp == 1) && "too many components");

   /* Handle immediates specially so we don't have to constant fold splits. */
   if (s0.type == AGX_INDEX_IMMEDIATE) {
      unsigned bits = 16 * agx_size_align_16(dst.size);
      return agx_mov_imm_to(b, dst, (s0.value >> bits) & BITFIELD64_MASK(bits));
   }

   agx_instr *split = agx_split(b, 2, s0);
   split->dest[comp] = dst;
   split->dest[1 - comp] = agx_temp(b->shader, dst.size);
   return split;
}

void
agx_block_add_successor(agx_block *block, agx_block *successor)
{
   assert(block != NULL && successor != NULL);

   /* Cull impossible edges */
   if (block->unconditional_jumps)
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(block->successors); ++i) {
      if (block->successors[i]) {
         if (block->successors[i] == successor)
            return;
         else
            continue;
      }

      block->successors[i] = successor;
      util_dynarray_append(&successor->predecessors, block);
      return;
   }

   UNREACHABLE("Too many successors");
}

/*
 * Splits an n-component vector (vec) into n scalar destinations (dests) using a
 * split pseudo-instruction.
 *
 * Pre-condition: dests is filled with agx_null().
 */
static void
agx_emit_split(agx_builder *b, agx_index *dests, agx_index vec, unsigned n)
{
   agx_instr *I = agx_split(b, n, vec);

   agx_foreach_dest(I, d) {
      dests[d] = agx_temp(b->shader, vec.size);
      I->dest[d] = dests[d];
   }
}

static void
agx_emit_cached_split(agx_builder *b, agx_index vec, unsigned n)
{
   agx_index dests[4] = {agx_null(), agx_null(), agx_null(), agx_null()};
   agx_emit_split(b, dests, vec, n);
   agx_cache_collect(b, vec, n, dests);
}

static void
agx_emit_load_const(agx_builder *b, nir_load_const_instr *instr)
{
   /* Ensure we've been scalarized and bit size lowered */
   unsigned bit_size = instr->def.bit_size;
   assert(instr->def.num_components == 1);

   /* Emit move, later passes can inline/push if useful */
   agx_mov_imm_to(b, agx_def_index(&instr->def),
                  nir_const_value_as_uint(instr->value[0], bit_size));
}

/*
 * Implement mul_high of 32-bit sources by doing a 32x32->64-bit multiply and
 * extracting only the high word.
 */
static agx_instr *
agx_mul_high_to(agx_builder *b, agx_index dst, agx_index P, agx_index Q,
                bool is_signed)
{
   assert(P.size == Q.size && "source sizes must match");
   assert(P.size == dst.size && "dest size must match");
   assert(P.size != AGX_SIZE_64 && "64x64 multiply should have been lowered");

   static_assert(AGX_SIZE_64 == (AGX_SIZE_32 + 1), "enum wrong");
   static_assert(AGX_SIZE_32 == (AGX_SIZE_16 + 1), "enum wrong");

   if (!is_signed) {
      P = agx_abs(P);
      Q = agx_abs(Q);
   }

   agx_index product = agx_temp(b->shader, P.size + 1);
   agx_imad_to(b, product, P, Q, agx_zero(), 0);

   return agx_subdivide_to(b, dst, product, 1);
}

static enum agx_format
agx_format_for_pipe(enum pipe_format format)
{
#define CASE(x)                                                                \
   if (format == (enum pipe_format)AIL_ISA_FORMAT_##x)                         \
      return AGX_FORMAT_##x;

   CASE(I8);
   CASE(I16);
   CASE(I32);
   CASE(F16);
   CASE(U8NORM);
   CASE(S8NORM);
   CASE(U16NORM);
   CASE(S16NORM);
   CASE(RGB10A2);
   CASE(SRGBA8);
   CASE(RG11B10F);
   CASE(RGB9E5);

#undef CASE
   UNREACHABLE("Invalid format");
}

static agx_index
cf_for_intrinsic(agx_builder *b, nir_intrinsic_instr *intr)
{
   /* Determine the base location, taking into account a constant offset */
   unsigned location = nir_intrinsic_io_semantics(intr).location;
   bool compact = location == VARYING_SLOT_CLIP_DIST0 ||
                  location == VARYING_SLOT_CLIP_DIST1;

   nir_src *offset = nir_get_io_offset_src(intr);
   if (nir_src_is_const(*offset)) {
      /* XXX: NIR is broken and uses constant offsets in slots but dynamic
       * offsets in scalars for compact varyings. This needs to be fixed
       * upstream.
       */
      location += nir_src_as_uint(*offset);
   }

   agx_index I = agx_get_cf(b->shader, location, nir_intrinsic_component(intr));

   /* If we have a non-constant offset, we add it to the CF. Offsets are in
    * vec4 slots (unless we're compact) but the CF is in components, so we need
    * to shift the offset by 2 before adding.
    */
   if (!nir_src_is_const(*offset)) {
      I = agx_iadd(b, I, agx_src_index(offset), compact ? 0 : 2);
   }

   return I;
}

static enum agx_interpolation
agx_interp_for_bary(nir_intrinsic_instr *bary, agx_index *sample_index)
{
   switch (bary->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel:
      return AGX_INTERPOLATION_CENTER;

   case nir_intrinsic_load_barycentric_centroid:
      return AGX_INTERPOLATION_CENTROID;

   case nir_intrinsic_load_barycentric_at_sample:
      *sample_index = agx_src_index(&bary->src[0]);
      return AGX_INTERPOLATION_SAMPLE;

   default:
      UNREACHABLE("should have been lowered");
   }
}

/* AGX3 removed AGX2's fused projective iterator. Build Apple's witnessed
 * scalar software sequence while values are still in SSA so register
 * allocation can place the temporary RHW, reciprocal, and channels safely.
 * G17P explicit at-sample interpolation first computes one ITRCOORDS token
 * from the low half of the sample index, then shares it across RHW and every
 * scalar channel. Dynamic coefficient indices remain on the legacy IR path
 * so the AGX3 packer rejects them instead of guessing. */
static bool
agx_emit_load_vary_agx3(agx_builder *b, agx_index dest, agx_index cf,
                        agx_index sample_index, unsigned components,
                        enum agx_interpolation interp, bool perspective,
                        nir_alu_type dest_type)
{
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2 ||
       (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32) ||
       cf.type != AGX_INDEX_IMMEDIATE)
      return false;

   agx_index interp_coords = agx_zero();
   if (interp == AGX_INTERPOLATION_SAMPLE) {
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
          (sample_index.size != AGX_SIZE_16 &&
           sample_index.size != AGX_SIZE_32))
         return false;

      agx_index sample_half = sample_index;
      if (sample_index.size == AGX_SIZE_32) {
         sample_half = agx_temp(b->shader, AGX_SIZE_16);
         agx_subdivide_to(b, sample_half, sample_index, 0);
      }

      /* ITRCOORDS has only a register source. This also keeps a constant
       * sample ID out of generic immediate promotion later in the backend. */
      if (sample_half.type != AGX_INDEX_NORMAL)
         sample_half = agx_mov(b, sample_half);

      interp_coords = agx_temp(b->shader, AGX_SIZE_32);
      agx_itrcoords_agx3_to(b, interp_coords, sample_half);
   } else if ((interp != AGX_INTERPOLATION_CENTER &&
               interp != AGX_INTERPOLATION_CENTROID) ||
              sample_index.type != AGX_INDEX_IMMEDIATE ||
              sample_index.value != 0) {
      return false;
   }

   agx_index reciprocal_w = agx_null();
   if (perspective) {
      agx_index rhw_cf = agx_get_cf(b->shader, VARYING_SLOT_POS, 3);
      if (rhw_cf.type != AGX_INDEX_IMMEDIATE)
         return false;

      agx_index rhw = agx_temp(b->shader, AGX_SIZE_32);
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (interp == AGX_INTERPOLATION_CENTER ||
           interp == AGX_INTERPOLATION_SAMPLE))
         agx_rhw_interp_agx3_to(b, rhw, rhw_cf, interp_coords, interp);
      else
         agx_interp_agx3_to(b, rhw, rhw_cf, interp_coords, interp, true);
      reciprocal_w = b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B
                        ? agx_persp_rcp_agx3(b, rhw, rhw_cf.value)
                        : agx_rcp(b, rhw);
   }

   agx_index channels[4];
   for (unsigned c = 0; c < components; ++c) {
      agx_index channel_cf = cf;
      channel_cf.value += c;

      agx_index raw = agx_temp(b->shader, AGX_SIZE_32);
      agx_interp_agx3_to(b, raw, channel_cf, interp_coords, interp,
                         perspective);
      if (perspective &&
          b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         /* Native f32mul.ep4 identifies the coefficient involved in this
          * projective divide. Preserve it explicitly rather than trying to
          * reconstruct SSA provenance after physical register allocation. */
         channels[c] = agx_persp_mul_agx3(b, raw, reciprocal_w, channel_cf.value);
      } else {
         channels[c] = perspective ? agx_fmul(b, raw, reciprocal_w) : raw;
      }

      /* AGX3 interpolation is natively 32-bit. Vulkan's
       * storageInputOutput16 feature nevertheless permits a logical 16-bit
       * fragment input, so narrow only after interpolation. Vertex output
       * lowering performs the matching type-correct widening before ST_VARY.
       * This preserves integer bits for flat varyings and applies the required
       * floating-point conversion for smooth/noperspective float varyings. */
      if (dest.size == AGX_SIZE_16) {
         agx_index narrowed = agx_temp(b->shader, AGX_SIZE_16);
         if (nir_alu_type_get_base_type(dest_type) == nir_type_float)
            agx_fmov_to(b, narrowed, channels[c]);
         else
            agx_subdivide_to(b, narrowed, channels[c], 0);

         channels[c] = narrowed;
      }
   }

   agx_emit_collect_to(b, dest, components, channels);
   return true;
}

static void
agx_emit_load_vary(agx_builder *b, agx_index dest, nir_intrinsic_instr *instr)
{
   ASSERTED unsigned components = instr->num_components;
   nir_intrinsic_instr *bary = nir_src_as_intrinsic(instr->src[0]);

   assert(components >= 1 && components <= 4);

   agx_index sample_index = agx_zero();
   enum agx_interpolation interp = agx_interp_for_bary(bary, &sample_index);

   bool perspective =
      nir_intrinsic_interp_mode(bary) != INTERP_MODE_NOPERSPECTIVE;

   agx_index I = cf_for_intrinsic(b, instr);

   if (agx_emit_load_vary_agx3(b, dest, I, sample_index, components, interp,
                               perspective, nir_intrinsic_dest_type(instr))) {
      agx_emit_cached_split(b, dest, components);
      return;
   }

   /* For perspective interpolation, we project (multiply by 1/W) */
   if (perspective) {
      agx_index J = agx_get_cf(b->shader, VARYING_SLOT_POS, 3);
      agx_iterproj_to(b, dest, I, J, sample_index, components, interp);
   } else {
      agx_iter_to(b, dest, I, sample_index, components, interp);
   }

   agx_emit_cached_split(b, dest, components);
}

static void
agx_wait_pixel_mask(agx_builder *b, uint32_t mask)
{
   /* Background programs do not need to wait as they are the eldest pixels */
   if (b->shader->key->fs.ignore_tib_dependencies) {
      assert(b->shader->nir->info.internal);
      return;
   }

   /* No need to wait twice on a fence */
   mask &= ~b->shader->already_pixel_waited;
   if (mask == 0) {
      return;
   }

   agx_wait_pix(b, mask);

   /* Only mark the fence as waited if we're not in control flow. Eventually we
    * should do something smarter with a dataflow.
    */
   if (b->shader->total_nesting == 0) {
      b->shader->already_pixel_waited |= mask;
   }
}

/*
 * AGX3's `lstcol` carries only a width, so unlike AGX2's st_tile it cannot
 * normalize on the way out: an 8-bit normalized target has to arrive already
 * packed, two channels to a 16-bit register. `fltnorm` is that pack -- it
 * clamps and scales two 32-bit floats into the two halves of one 16-bit
 * register -- and Apple's own G17P fragment shaders spend exactly
 * ceil(channels / 2) of them before every 8-bit colour store.
 *
 * Returns the number of packed 16-bit registers, or 0 if this store is not one
 * this path can pack, in which case the caller keeps the unpacked value and
 * the AGX3 packer refuses the shader rather than storing raw floats through a
 * width that would truncate them.
 */
static unsigned
agx_pack_normalized_agx3(agx_builder *b, const agx_index *channels,
                         unsigned nr_channels, enum agx_format format,
                         agx_index *packed)
{
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2)
      return 0;

   if (format == AGX_FORMAT_RGB10A2) {
      if (nr_channels != 4)
         return 0;
      agx_index pack_sources[4];
      for (unsigned i = 0; i < 4; ++i) {
         if (channels[i].size == AGX_SIZE_32) {
            pack_sources[i] = channels[i];
         } else if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
                    channels[i].size == AGX_SIZE_16) {
            /* As with the normalized 8-bit epilog below, a texture result
             * can arrive through the half export ABI. The native RGB10A2
             * pack consumes f32, so widen before packing instead of handing
             * an unpacked half vector to the tile-store encoder. */
            pack_sources[i] = agx_temp(b->shader, AGX_SIZE_32);
            agx_fmov_to(b, pack_sources[i], channels[i]);
         } else {
            return 0;
         }
      }
      packed[0] = agx_pack_rgb10a2_agx3(b, pack_sources[0], pack_sources[1],
                                      pack_sources[2], pack_sources[3]);
      return 1;
   }

   const bool norm8 =
      format == AGX_FORMAT_U8NORM || format == AGX_FORMAT_S8NORM;
   const bool norm16 =
      format == AGX_FORMAT_U16NORM || format == AGX_FORMAT_S16NORM;

   /* Fresh Apple G17P R16Unorm/R16Snorm pipelines use the same conversion
    * opcode with format modes 2/3 and one instruction per 16-bit channel.
    * Keep AGX3c closed until an equivalent native witness exists. */
   if (!norm8 &&
       !(norm16 && b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B))
      return 0;

   if (nr_channels < 1 || nr_channels > 4)
      return 0;

   /* Every fltnorm witness takes 32-bit float sources. A linked fragment
    * epilog may receive normalized colour through the 16-bit export ABI, so
    * reproduce Apple's native half-to-float conversion before packing. */
   agx_index pack_sources[4];
   for (unsigned i = 0; i < nr_channels; ++i) {
      if (channels[i].size == AGX_SIZE_32) {
         pack_sources[i] = channels[i];
      } else if (norm8 &&
                 b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
                 channels[i].size == AGX_SIZE_16) {
         pack_sources[i] = agx_temp(b->shader, AGX_SIZE_32);
         agx_fmov_to(b, pack_sources[i], channels[i]);
      } else {
         return 0;
      }
   }

   unsigned n = 0;
   for (unsigned i = 0; i < nr_channels; i += 2) {
      const unsigned lanes = MIN2(2u, nr_channels - i);
      const agx_index hi =
         (lanes == 2) ? pack_sources[i + 1] : agx_null();
      /* The destination is a 16-bit half, not the sources' 32 bits, so the
       * generated temp-allocating helper cannot be used here. */
      agx_index dest = agx_temp(b->shader, AGX_SIZE_16);
      if (norm16)
         dest.channels_m1 = lanes - 1;
      agx_fltnorm_agx3_to(b, dest, pack_sources[i], hi, format);
      if (norm16) {
         agx_index halves[2] = {agx_null()};
         agx_emit_split(b, halves, dest, lanes);
         for (unsigned c = 0; c < lanes; ++c)
            packed[n++] = halves[c];
      } else {
         packed[n++] = dest;
      }
   }
   return n;
}

/* Pure 8-bit integer outputs have already been clamped and narrowed to 16-bit
 * NIR lanes. AGX3 tile traffic instead carries two adjacent bytes in each
 * 16-bit register, so explicitly reproduce that physical packing here. */
static unsigned
agx_pack_integer_8_agx3(agx_builder *b, const agx_index *channels,
                        unsigned nr_channels, enum agx_format format,
                        agx_index *packed)
{
   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       format != AGX_FORMAT_I8 || nr_channels < 1 || nr_channels > 4)
      return 0;

   for (unsigned i = 0; i < nr_channels; ++i)
      if (channels[i].size != AGX_SIZE_16)
         return 0;

   unsigned n = 0;
   for (unsigned i = 0; i < nr_channels; i += 2) {
      if (i + 1 == nr_channels) {
         packed[n++] = channels[i];
         continue;
      }

      agx_index upper = agx_pad_to_32(b, channels[i + 1]);
      agx_index shifted = agx_temp(b->shader, AGX_SIZE_32);
      agx_bfi_to(b, shifted, agx_immediate(0), upper,
                 agx_immediate(8), 0);
      agx_index lower = agx_and(b, channels[i], agx_immediate(0xff));
      agx_index pair = agx_temp(b->shader, AGX_SIZE_16);
      agx_bitop_to(b, pair, lower, shifted, 0xe /* OR */);
      packed[n++] = pair;
   }

   return n;
}

/* A multi-component W8 memory payload is byte-packed in the register file:
 * two adjacent components occupy the low/high bytes of one 16-bit register.
 * NIR keeps each 8-bit component in its own 16-bit lane, so unpack the raw
 * device-load payload before exposing it to the rest of the backend. */
static void
agx_unpack_integer_8_memory_agx3(agx_builder *b, agx_index dest,
                                 agx_index packed, unsigned nr_channels)
{
   const unsigned nr_halfs = DIV_ROUND_UP(nr_channels, 2);
   agx_index halfs[2] = {agx_null()};
   for (unsigned i = 0; i < nr_halfs; ++i) {
      halfs[i] = agx_temp(b->shader, AGX_SIZE_16);
      agx_subdivide_to(b, halfs[i], packed, i);
   }

   agx_index channels[4] = {agx_null()};
   agx_index shift = agx_mov_imm(b, 32, 8);
   for (unsigned i = 0; i < nr_channels; ++i) {
      agx_index wide = agx_pad_to_32(b, halfs[i / 2]);
      if (i & 1)
         wide = agx_ushr(b, wide, shift);

      agx_index narrow = agx_temp(b->shader, AGX_SIZE_16);
      agx_subdivide_to(b, narrow, wide, 0);
      channels[i] = agx_and(b, narrow, agx_immediate(0xff));
   }

   agx_emit_collect_to(b, dest, nr_channels, channels);
}

static agx_instr *
agx_emit_local_store_pixel(agx_builder *b, nir_intrinsic_instr *instr)
{
   bool explicit = nir_intrinsic_explicit_coord(instr);

   /* TODO: Reverse-engineer interactions with MRT */
   if (b->shader->stage == MESA_SHADER_FRAGMENT) {
      agx_wait_pixel_mask(b, 0xC);
   }

   /* Compact the registers according to the mask */
   agx_index compacted[4] = {agx_null()};

   unsigned compact_count = 0;
   u_foreach_bit(i, nir_intrinsic_write_mask(instr)) {
      compacted[compact_count++] = agx_extract_nir_src(b, instr->src[0], i);
   }

   const enum agx_format format =
      agx_format_for_pipe(nir_intrinsic_format(instr));

   agx_index packed[4] = {agx_null()};
   unsigned nr_packed =
      agx_pack_normalized_agx3(b, compacted, compact_count, format, packed);
   if (nr_packed == 0)
      nr_packed = agx_pack_integer_8_agx3(
         b, compacted, compact_count, format, packed);

   /* G17P's tile store encodes only the lane width.  Unlike AGX2 it cannot
    * narrow an f32 colour to the f16 lanes named by that width, so make the
    * conversion explicit before collecting the store payload.  Keep AGX3c
    * unchanged until its matching narrowing ALU form is grounded. */
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       format == AGX_FORMAT_F16 && nr_packed == 0) {
      nr_packed = compact_count;
      for (unsigned i = 0; i < compact_count; ++i) {
         if (compacted[i].size == AGX_SIZE_32) {
            packed[i] = agx_temp(b->shader, AGX_SIZE_16);
            agx_fmov_to(b, packed[i], compacted[i]);
         } else {
            packed[i] = compacted[i];
         }
      }
   }

   agx_index collected =
      nr_packed ? agx_emit_collect(b, nr_packed, packed)
                : agx_emit_collect(b, compact_count, compacted);
   agx_index coords = explicit ? agx_src_index(&instr->src[2]) : agx_null();

   b->shader->out->tag_write_disable = false;
   return agx_st_tile(b, collected, agx_src_index(&instr->src[1]), coords,
                      format, (nr_packed && format == AGX_FORMAT_RGB10A2) ? 1 :
                         nir_intrinsic_write_mask(instr),
                      nir_intrinsic_base(instr), explicit,
                      nir_intrinsic_target(instr));
}

/* Stage one G17P depth/stencil word, whose low nibble selects every sample
 * and whose next two nibbles come from `mid`, and write it under the class-1
 * transaction the first write of the shader opens and the last releases. */
static agx_instr *
agx_emit_zs_word_agx3(agx_builder *b, nir_intrinsic_instr *instr,
                      agx_index z, agx_index s, unsigned base, agx_index mid)
{
   const bool write_z = base & 1;
   const bool write_s = base & 2;
   agx_index coverage = agx_iadd(b, agx_immediate(15), mid, 4);

   agx_index word = agx_temp(b->shader, AGX_SIZE_32);
   if (write_s)
      agx_emit_collect_to(b, word, 2, (agx_index[]){coverage, s});
   else
      agx_mov_to(b, word, coverage);

   if (write_z) {
      agx_index depth;
      if (nir_src_is_const(instr->src[1])) {
         const float value = nir_src_as_float(instr->src[1]);
         depth = agx_mov_imm(b, 32, fui(CLAMP(value, 0.0f, 1.0f)));
      } else {
         depth = agx_temp(b->shader, AGX_SIZE_32);
         agx_fmov_to(b, depth, z)->saturate = true;
      }
      word = agx_vec2(b, word, depth);
   }

   b->shader->out->writes_sample_mask = true;
   agx_wait_pixel_mask(b, 0x1);
   agx_instr *I = agx_zs_emit_agx3(b, word, base);
   assert(b->shader->zs_writes_left > 0);
   if (--b->shader->zs_writes_left == 0)
      agx_signal_pix(b, 0x1);
   return I;
}

static agx_instr *
agx_emit_store_zs(agx_builder *b, nir_intrinsic_instr *instr)
{
   unsigned base = nir_intrinsic_base(instr);
   bool write_z = base & 1;
   bool write_s = base & 2;

   agx_index z = agx_src_index(&instr->src[1]);
   agx_index s = agx_src_index(&instr->src[2]);

   assert(!write_z || z.size == AGX_SIZE_32);
   assert(!write_s || s.size == AGX_SIZE_16);

   /* G17P depthf reads staged dwords: coverage at half 0 and float32 depth
    * at half 2. Native FSEL.ep35 publishes the clamped depth directly to
    * staging[2], followed by a separate coverage COPY to staging[0].
    * Do not narrow depth into the coverage word's high half. Build every
    * full-width value before RA so the allocator owns the payload.
    *
    * A stencil reference occupies that high half instead. Apple's G17P
    * [[stencil]] fragments copy the 16-bit reference into the upper half of
    * the register holding the coverage, then stage the whole dword; with
    * [[depth(any)]] as well, depth still follows at half 2.
    */
   const unsigned samples = MAX2(b->shader->key->fs.nr_samples, 1);
   const unsigned mask =
      nir_src_is_const(instr->src[0]) ? nir_src_as_uint(instr->src[0]) : 0;
   const bool sample_words =
      samples > 1 &&
      (mask == 0xff || util_bitcount(mask & BITFIELD_MASK(samples)) == 1);
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       !sample_words) {
      /* At several samples, only a mask known at compile time has a
       * witnessed word; any other reads the one-sample coverage, which the
       * packer refuses there. */
      agx_index sr2 = agx_temp(b->shader, AGX_SIZE_16);
      agx_get_coverage_1x_agx3_to(b, sr2);
      return agx_emit_zs_word_agx3(b, instr, z, s, base,
                                   agx_iadd(b, sr2, sr2, 4));
   } else if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
      /* Apple's 2x/4x fragments writing [[depth(any)]] or [[stencil]] per
       * [[sample_id]] write one DEPTHF per sample, each word built from SR39:
       * the API mask nibble, above the covered samples ANDed with that
       * sample's bit. A whole-pixel depth or stencil alone is written the
       * same way for every sample, but a whole-pixel depth and stencil is
       * written once, with all covered samples (metal-nt -arch applegpu_g17p,
       * msaa/native/zexp2 fs_id_{2,4}x and fs_var_4x, msaa/native/zmrt/zsten
       * fs_id_s_4x, fs_s_4x and fs_ds_4x).
       */
      const bool whole = mask == 0xff;
      const bool combined = whole && write_z && write_s;
      agx_instr *last = NULL;
      if (whole && !combined)
         b->shader->zs_writes_left += samples - 1;
      for (unsigned i = 0; i < (combined ? 1 : samples); ++i) {
         if (!whole && !(mask & BITFIELD_BIT(i)))
            continue;

         agx_index sr39 = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_to(b, sr39, AGX_SR_INPUT_SAMPLE_MASK);
         agx_index upper = agx_ushr(b, sr39, agx_immediate(4));
         agx_index covered = agx_and(
            b, sr39,
            agx_immediate(combined ? BITFIELD_MASK(4) : BITFIELD_BIT(i)));
         last = agx_emit_zs_word_agx3(b, instr, z, s, base,
                                      agx_iadd(b, upper, covered, 4));
      }
      return last;
   }

   if (write_z && write_s) {
      agx_index u2u32 = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, u2u32, s);
      s = u2u32;
   }

   agx_index zs = (write_z && write_s) ? agx_vec2(b, z, s) : write_z ? z : s;

   /* Not necessarily a sample mask but overlapping hw mechanism... Should
    * maybe rename this flag to something more general.
    */
   b->shader->out->writes_sample_mask = true;

   agx_wait_pixel_mask(b, 0x1);
   return agx_zs_emit(b, agx_src_index(&instr->src[0]), zs, base);
}

/* Unpack already-loaded normalized halves into `dest`. A byte format packs
 * two channels into each half; a 16-bit one occupies a half per channel and
 * is recollected into the pair the conversion consumes. Tile and device
 * traffic differ only in how the halves arrive, so both share this tail. */
static void
agx_unpack_normalized_halves_agx3(agx_builder *b, agx_index dest,
                                  agx_index *halves, enum agx_format format,
                                  unsigned nr_comps)
{
   const bool norm16 =
      format == AGX_FORMAT_U16NORM || format == AGX_FORMAT_S16NORM;
   agx_index channels[4] = {agx_null()};
   const unsigned conversions = DIV_ROUND_UP(nr_comps, 2);
   for (unsigned i = 0; i < conversions; ++i) {
      /* One unpack per packed register; each yields a 32-bit pair, which the
       * split below turns back into the two scalar channels. */
      agx_index pair = agx_temp(b->shader, AGX_SIZE_32);
      const unsigned lanes = MIN2(2u, nr_comps - i * 2);
      pair.channels_m1 = lanes - 1;
      agx_index source = halves[i];
      if (norm16) {
         source = agx_temp(b->shader, AGX_SIZE_16);
         source.channels_m1 = lanes - 1;
         agx_emit_collect_to(b, source, lanes, &halves[i * 2]);
      }
      agx_normflt_agx3_to(b, pair, source, format);

      agx_index unpacked[2] = {agx_null()};
      agx_emit_split(b, unpacked, pair, lanes);
      for (unsigned c = 0; c < lanes; ++c) {
         if (dest.size == AGX_SIZE_16) {
            channels[i * 2 + c] = agx_temp(b->shader, AGX_SIZE_16);
            agx_fmov_to(b, channels[i * 2 + c], unpacked[c]);
         } else {
            channels[i * 2 + c] = unpacked[c];
         }
      }
   }

   agx_emit_collect_to(b, dest, nr_comps, channels);
   agx_emit_cached_split(b, dest, nr_comps);
}

/*
 * The tilebuffer read half of the same story as agx_pack_normalized_agx3():
 * an 8-bit normalized attachment comes back packed, two channels to a 16-bit
 * register, and `normflt` is the unpack -- one 16-bit register in, two 32-bit
 * floats out. Apple's programmable-blending shaders spend exactly
 * ceil(channels / 2) of them straight after the tile load.
 *
 * Returns true if it handled the load.
 */
static bool
agx_unpack_normalized_agx3(agx_builder *b, agx_index dest,
                           nir_intrinsic_instr *instr, enum agx_format format,
                           unsigned nr_comps)
{
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2)
      return false;
   if (format == AGX_FORMAT_RGB10A2) {
      if (nr_comps != 4 || dest.size != AGX_SIZE_32)
         return false;
      agx_index packed = agx_temp(b->shader, AGX_SIZE_32);
      agx_ld_tile_to(b, packed, agx_src_index(&instr->src[0]), agx_null(),
                    format, 1, nir_intrinsic_base(instr), false,
                    nir_intrinsic_target(instr));
      agx_unpack_rgb10a2_agx3_to(b, dest, packed);
      agx_emit_cached_split(b, dest, 4);
      return true;
   }
   const bool norm8 =
      format == AGX_FORMAT_U8NORM || format == AGX_FORMAT_S8NORM;
   const bool norm16 =
      format == AGX_FORMAT_U16NORM || format == AGX_FORMAT_S16NORM;
   if (!norm8 &&
       !(norm16 && b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B))
      return false;
   if (nr_comps < 1 || nr_comps > 4 ||
       (dest.size != AGX_SIZE_16 && dest.size != AGX_SIZE_32))
      return false;

   const unsigned loaded_channels =
      norm16 ? nr_comps : DIV_ROUND_UP(nr_comps, 2);

   agx_index packed = agx_temp(b->shader, AGX_SIZE_16);
   packed.channels_m1 = loaded_channels - 1;
   agx_ld_tile_to(b, packed, agx_src_index(&instr->src[0]), agx_null(), format,
                  BITFIELD_MASK(nr_comps), nir_intrinsic_base(instr), false,
                  nir_intrinsic_target(instr));

   agx_index halves[4] = {agx_null()};
   agx_emit_split(b, halves, packed, loaded_channels);

   agx_unpack_normalized_halves_agx3(b, dest, halves, format, nr_comps);
   return true;
}

static void
agx_emit_local_load_pixel(agx_builder *b, agx_index dest,
                          nir_intrinsic_instr *instr)
{
   agx_wait_pixel_mask(b, 0x8);

   unsigned nr_comps = instr->def.num_components;
   const enum agx_format format =
      agx_format_for_pipe(nir_intrinsic_format(instr));

   if (agx_unpack_normalized_agx3(b, dest, instr, format, nr_comps))
      return;

   agx_ld_tile_to(b, dest, agx_src_index(&instr->src[0]), agx_null(), format,
                  BITFIELD_MASK(nr_comps), nir_intrinsic_base(instr), false,
                  nir_intrinsic_target(instr));
   agx_emit_cached_split(b, dest, nr_comps);
}

static bool
nir_is_coherent(nir_intrinsic_instr *instr)
{
   return nir_intrinsic_access(instr) & (ACCESS_COHERENT | ACCESS_VOLATILE);
}

static void
agx_emit_load(agx_builder *b, agx_index dest, nir_intrinsic_instr *instr)
{
   agx_index addr = agx_src_index(&instr->src[0]);
   agx_index offset = agx_src_index(&instr->src[1]);
   enum agx_format fmt = agx_format_for_pipe(nir_intrinsic_format(instr));
   unsigned shift = nir_intrinsic_base(instr);
   const bool index_bytes = shift & AGX_NIR_MEMORY_INDEX_BYTES;
   assert(!index_bytes || b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B);
   const unsigned packed_count = (shift & AGX_NIR_MEMORY_PACKED_COUNT_MASK) >>
                                 AGX_NIR_MEMORY_PACKED_COUNT_SHIFT;
   assert(packed_count <= 4);
   assert(!packed_count || b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B);
   shift &= ~(AGX_NIR_MEMORY_INDEX_BYTES | AGX_NIR_MEMORY_PACKED_COUNT_MASK);

   assert(shift <= 3);

   /* Zero-extend offset if we're not sign-extending */
   if (!nir_intrinsic_sign_extend(instr))
      offset = agx_abs(offset);

   const bool norm8 = fmt == AGX_FORMAT_U8NORM || fmt == AGX_FORMAT_S8NORM;
   const bool norm16 = fmt == AGX_FORMAT_U16NORM || fmt == AGX_FORMAT_S16NORM;
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       (norm8 || norm16)) {
      const unsigned nr = instr->def.num_components;
      assert(nr >= 1 && nr <= 4 && packed_count == 0);
      assert(dest.size == AGX_SIZE_16 || dest.size == AGX_SIZE_32);

      /* Device memory provides raw bytes/halves, with no format conversion:
       * agx3_fmt_to_width() has no normalized device witness, so a load left
       * carrying one is refused at the packer and the draw is dropped. Keep
       * the element width and index scaling intact and reuse the grounded
       * normalized unpack the tile load already uses. Byte loads reserve an
       * aligned 32-bit payload even for a single channel. */
      agx_index packed = norm8 ? agx_temp(b->shader, AGX_SIZE_32)
                              : agx_vec_temp(b->shader, AGX_SIZE_16, nr);
      agx_instr *load = agx_device_load_to(
         b, packed, addr, offset, norm8 ? AGX_FORMAT_I8 : AGX_FORMAT_I16,
         BITFIELD_MASK(nr), shift, nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;
      load->memory_is_packed = norm8 && nr > 1;

      agx_index halves[4] = {agx_null()};
      if (norm8) {
         for (unsigned i = 0; i < DIV_ROUND_UP(nr, 2); ++i) {
            halves[i] = agx_temp(b->shader, AGX_SIZE_16);
            agx_subdivide_to(b, halves[i], packed, i);
         }
      } else {
         agx_emit_split(b, halves, packed, nr);
      }
      agx_unpack_normalized_halves_agx3(b, dest, halves, fmt, nr);
      return;
   }

   /* AGX3 memory traffic is raw. Convert a normalized packed texel with
    * the same native instruction used by tile loads, after its real load. */
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       fmt == AGX_FORMAT_RGB10A2) {
      assert(dest.size == AGX_SIZE_32 && instr->def.num_components == 4);
      assert(packed_count == 0);
      agx_index packed = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *load = agx_device_load_to(b, packed, addr, offset,
         AGX_FORMAT_I32, 1, shift, nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;
      agx_unpack_rgb10a2_agx3_to(b, dest, packed);
      agx_emit_cached_split(b, dest, 4);
      return;
   }

   /* A scalar W8 load writes an aligned low half. Multi-component W8 loads
    * instead return a byte-packed payload, like G17P tile traffic. */
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       fmt == AGX_FORMAT_I8 && dest.size == AGX_SIZE_16 && packed_count == 0) {
      const unsigned nr = instr->def.num_components;
      agx_index payload = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *load = agx_device_load_to(b, payload, addr, offset, fmt,
                                           BITFIELD_MASK(nr), shift,
                                           nir_is_coherent(instr));
      load->memory_index_is_bytes = index_bytes;

      if (nr == 1) {
         agx_subdivide_to(b, dest, payload, 0);
      } else {
         load->memory_is_packed = true;
         agx_unpack_integer_8_memory_agx3(b, dest, payload, nr);
      }
      return;
   }

   agx_instr *load = agx_device_load_to(b, dest, addr, offset, fmt,
                      BITFIELD_MASK(packed_count ? packed_count : instr->def.num_components), shift,
                      nir_is_coherent(instr));
   load->memory_index_is_bytes = index_bytes;
   load->memory_is_packed = packed_count != 0;
   agx_emit_cached_split(b, dest, instr->def.num_components);
}

static void
agx_emit_store(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index addr = agx_src_index(&instr->src[1]);
   agx_index offset = agx_src_index(&instr->src[2]);
   enum agx_format fmt = agx_format_for_pipe(nir_intrinsic_format(instr));
   unsigned shift = nir_intrinsic_base(instr);
   const bool index_bytes = shift & AGX_NIR_MEMORY_INDEX_BYTES;
   assert(!index_bytes || b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B);
   shift &= ~AGX_NIR_MEMORY_INDEX_BYTES;

   assert(shift <= 3);

   /* Zero-extend offset if we're not sign-extending */
   if (!nir_intrinsic_sign_extend(instr))
      offset = agx_abs(offset);

   agx_index value;
   unsigned mask;
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       fmt == AGX_FORMAT_RGB10A2) {
      assert(nir_src_num_components(instr->src[0]) == 4 &&
             nir_src_bit_size(instr->src[0]) == 32);
      value = agx_pack_rgb10a2_agx3(b,
         agx_extract_nir_src(b, instr->src[0], 0),
         agx_extract_nir_src(b, instr->src[0], 1),
         agx_extract_nir_src(b, instr->src[0], 2),
         agx_extract_nir_src(b, instr->src[0], 3));
      fmt = AGX_FORMAT_I32;
      mask = 1;
   } else if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
              fmt == AGX_FORMAT_I8 &&
              nir_src_bit_size(instr->src[0]) == 8 &&
              nir_src_num_components(instr->src[0]) > 1) {
      /* Preserve one W8 vector transaction while packing its register payload
       * two bytes per 16-bit lane. */
      const unsigned nr = nir_src_num_components(instr->src[0]);
      agx_index channels[4] = {agx_null()};
      for (unsigned i = 0; i < nr; ++i)
         channels[i] = agx_extract_nir_src(b, instr->src[0], i);

      agx_index packed[2] = {agx_null()};
      const unsigned nr_packed =
         agx_pack_integer_8_agx3(b, channels, nr, fmt, packed);
      assert(nr_packed == DIV_ROUND_UP(nr, 2));
      value = agx_emit_collect(b, nr_packed, packed);
      mask = BITFIELD_MASK(nr);
   } else {
      value = agx_recollect_vector(b, instr->src[0]);
      mask = BITFIELD_MASK(nir_src_num_components(instr->src[0]));
   }
   agx_instr *store = agx_device_store(b, value, addr, offset, fmt, mask,
                                      shift, nir_is_coherent(instr));
   store->memory_index_is_bytes = index_bytes;
   store->memory_is_packed =
      b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
      fmt == AGX_FORMAT_I8 && nir_src_num_components(instr->src[0]) > 1;
}

/*
 * Hardware bindless texture sources are specified as a 64-bit uniform base
 * address summed with a 32-bit register index. We model in NIR with the
 * bindless_image_agx intrinsic.
 */
static agx_index
agx_translate_bindless_handle(agx_builder *b, nir_src *handle, agx_index *base)
{
   nir_intrinsic_instr *intr = nir_src_as_intrinsic(*handle);
   assert(intr->intrinsic == nir_intrinsic_bindless_image_agx ||
          intr->intrinsic == nir_intrinsic_bindless_sampler_agx);

   *base = agx_uniform(nir_intrinsic_desc_set(intr), AGX_SIZE_64);
   return agx_src_index(&intr->src[0]);
}

/* Preambles write directly to uniform registers, so move from uniform to GPR */
static agx_instr *
agx_emit_load_preamble(agx_builder *b, agx_index dst,
                       nir_intrinsic_instr *instr)
{
   agx_index srcs[4] = {agx_null()};
   unsigned dim = instr->def.num_components;
   assert(dim <= ARRAY_SIZE(srcs) && "shouldn't see larger vectors");

   unsigned base = nir_intrinsic_base(instr);
   unsigned stride = agx_size_align_16(dst.size);

   for (unsigned i = 0; i < dim; ++i)
      srcs[i] = agx_uniform(base + i * stride, dst.size);

   return agx_emit_collect_to(b, dst, dim, srcs);
}

static agx_instr *
agx_emit_store_preamble(agx_builder *b, nir_intrinsic_instr *instr)
{
   nir_preamble_class cls = nir_intrinsic_preamble_class(instr);
   unsigned base = nir_intrinsic_base(instr);

   if (cls != nir_preamble_class_general) {
      agx_index heap, offset;
      offset = agx_translate_bindless_handle(b, &instr->src[0], &heap);

      /* base is 32-bit units for images but 16-bit for samplers, hence the
       * division difference to convert into texture/sampler state units.
       */
      if (cls == nir_preamble_class_image)
         return agx_tex_state_store(b, heap, offset, base / 2);
      else
         return agx_sampler_state_store(b, heap, offset, base);
   }

   agx_index vec = agx_src_index(&instr->src[0]);
   unsigned stride = agx_size_align_16(vec.size);
   unsigned nr = nir_src_num_components(instr->src[0]);

   for (unsigned i = 0; i < nr; i += (4 / stride)) {
      agx_index data[4] = {0};
      unsigned count = MIN2(4 / stride, nr - i);

      for (unsigned c = 0; c < count; ++c) {
         data[c] = agx_extract_nir_src(b, instr->src[0], i + c);
      }

      agx_uniform_store(b, agx_emit_collect(b, count, data),
                        agx_immediate(base + i * stride), BITFIELD_MASK(count));
   }

   return NULL;
}

static enum agx_dim
agx_tex_dim(enum glsl_sampler_dim dim, bool array)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_1D:
      return array ? AGX_DIM_1D_ARRAY : AGX_DIM_1D;

   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_EXTERNAL:
      return array ? AGX_DIM_2D_ARRAY : AGX_DIM_2D;

   case GLSL_SAMPLER_DIM_MS:
      return array ? AGX_DIM_2D_MS_ARRAY : AGX_DIM_2D_MS;

   case GLSL_SAMPLER_DIM_3D:
      assert(!array && "3D arrays unsupported");
      return AGX_DIM_3D;

   case GLSL_SAMPLER_DIM_CUBE:
      return array ? AGX_DIM_CUBE_ARRAY : AGX_DIM_CUBE;

   case GLSL_SAMPLER_DIM_BUF:
      UNREACHABLE("Buffer textures should have been lowered");

   default:
      UNREACHABLE("Invalid sampler dim\n");
   }
}

static agx_instr *
agx_emit_block_image_store(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index offset = agx_src_index(&instr->src[1]);
   agx_index coords = agx_src_index(&instr->src[2]);
   enum agx_format format = agx_format_for_pipe(nir_intrinsic_format(instr));

   bool ms = nir_intrinsic_image_dim(instr) == GLSL_SAMPLER_DIM_MS;
   bool array = nir_intrinsic_image_array(instr);
   enum agx_dim dim = agx_tex_dim(nir_intrinsic_image_dim(instr), array);
   bool explicit = nir_intrinsic_explicit_coord(instr);

   /* Modified coordinate descriptor */
   if (!explicit) {
      if (array) {
         agx_index layer = coords;
         coords = agx_temp(b->shader, AGX_SIZE_32);
         agx_emit_collect_to(b, coords, 2,
                             (agx_index[]){
                                ms ? agx_mov_imm(b, 16, 0) : layer,
                                ms ? layer : agx_undef(AGX_SIZE_16),
                             });
      } else {
         coords = agx_null();
      }
   }

   agx_index base, index;
   if (instr->intrinsic == nir_intrinsic_bindless_image_store_block_agx) {
      index = agx_translate_bindless_handle(b, &instr->src[0], &base);

      assert(base.size == AGX_SIZE_64);
      assert(index.size == AGX_SIZE_32);
   } else {
      base = agx_zero();
      index = agx_src_index(&instr->src[0]);

      assert(index.size == AGX_SIZE_16);
   }

   // XXX: how does this possibly work
   if (format == AGX_FORMAT_F16)
      format = AGX_FORMAT_I16;

   return agx_block_image_store(b, base, index, offset, coords, format, dim,
                                explicit);
}

static agx_instr *
agx_get_sr_native_width_to(agx_builder *b, agx_index dst, enum agx_sr sr)
{
   bool narrow_agx3 = false;

   switch (sr) {
   case AGX_SR_THREADS_PER_THREADGROUP_X:
   case AGX_SR_THREADS_PER_THREADGROUP_Y:
   case AGX_SR_THREADS_PER_THREADGROUP_Z:
   case AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_X:
   case AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_Y:
   case AGX_SR_DISPATCH_THREADS_PER_THREADGROUP_Z:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_X:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_Y:
   case AGX_SR_THREAD_POSITION_IN_THREADGROUP_Z:
   case AGX_SR_THREAD_INDEX_IN_THREADGROUP:
   case AGX_SR_THREAD_INDEX_IN_SUBGROUP:
   case AGX_SR_SUBGROUP_INDEX_IN_THREADGROUP:
      narrow_agx3 = true;
      break;
   default:
      break;
   }

   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 && narrow_agx3 &&
       dst.size == AGX_SIZE_32) {
      /* Apple's AGX3 get_sr forms expose these builtins as witnessed 16-bit
       * values. NIR presents the Vulkan/D3D builtins as uint32_t, so read the
       * native half register and explicitly zero-extend it. This also makes
       * the async dependency visible to the scoreboard allocator. */
      agx_index native = agx_temp(b->shader, AGX_SIZE_16);
      agx_get_sr_to(b, native, sr);
      return agx_mov_to(b, dst, native);
   }

   return agx_get_sr_to(b, dst, sr);
}

static agx_instr *
agx_load_compute_dimension(agx_builder *b, agx_index dst,
                           nir_intrinsic_instr *instr, enum agx_sr base)
{
   unsigned dim = instr->def.num_components;
   unsigned size = instr->def.bit_size;
   assert(size == 16 || size == 32);

   agx_index srcs[3];
   for (unsigned i = 0; i < ARRAY_SIZE(srcs); ++i) {
      srcs[i] = agx_temp(b->shader, agx_size_for_bits(size));
      agx_get_sr_native_width_to(b, srcs[i], base + i);
   }

   return agx_emit_collect_to(b, dst, dim, srcs);
}

static enum agx_atomic_opc
translate_atomic_opcode(nir_atomic_op op)
{
   /* clang-format off */
   switch (op) {
   case nir_atomic_op_iadd:    return AGX_ATOMIC_OPC_ADD;
   case nir_atomic_op_imin:    return AGX_ATOMIC_OPC_IMIN;
   case nir_atomic_op_umin:    return AGX_ATOMIC_OPC_UMIN;
   case nir_atomic_op_imax:    return AGX_ATOMIC_OPC_IMAX;
   case nir_atomic_op_umax:    return AGX_ATOMIC_OPC_UMAX;
   case nir_atomic_op_iand:    return AGX_ATOMIC_OPC_AND;
   case nir_atomic_op_ior:     return AGX_ATOMIC_OPC_OR;
   case nir_atomic_op_ixor:    return AGX_ATOMIC_OPC_XOR;
   case nir_atomic_op_xchg:    return AGX_ATOMIC_OPC_XCHG;
   case nir_atomic_op_cmpxchg: return AGX_ATOMIC_OPC_CMPXCHG;
   default: UNREACHABLE("unknown atomic opcode");
   }
   /* clang-format on */
}

/*
 * The "base" of a local load/store/atomic can be zero but no other immediates.
 * This would be a little silly to handle when inlining immediates, so we
 * instead exclude these ops from immediate inlining and just handle 0 specially
 * when translating.
 */
static agx_index
agx_local_base(nir_src src)
{
   if (nir_src_is_const(src) && nir_src_as_uint(src) == 0)
      return agx_zero();
   else
      return agx_src_index(&src);
}

static agx_index
agx_local_atomic_base(agx_builder *b, nir_src src)
{
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       nir_src_is_const(src)) {
      const uint64_t address = nir_src_as_uint(src);

      /* G17P has a witnessed 16-bit immediate byte-address field for
       * threadgroup atomics, including returned exchanges at nonzero
       * addresses. Keep aligned constants immediate instead of needlessly
       * materializing the ungrounded returned register-address form. */
      if (address <= UINT16_MAX && (address & 3) == 0)
         return agx_immediate(address);
   }

   return agx_local_base(src);
}

static void
agx_emit_atomic(agx_builder *b, agx_index dst, nir_intrinsic_instr *instr,
                bool local)
{
   enum agx_atomic_opc op =
      translate_atomic_opcode(nir_intrinsic_atomic_op(instr));
   agx_index base =
      local ? agx_local_atomic_base(b, instr->src[0])
            : agx_src_index(&instr->src[0]);
   agx_index value = agx_src_index(&instr->src[local ? 1 : 2]);
   agx_index index = local ? agx_zero() : agx_src_index(&instr->src[1]);

   /* cmpxchg (only) takes 2 sources, passed in consecutive registers */
   if (op == AGX_ATOMIC_OPC_CMPXCHG) {
      agx_index value2 = agx_src_index(&instr->src[local ? 2 : 3]);
      value = agx_vec2(b, value2, value);
   }

   if (local) {
      assert(base.size == AGX_SIZE_16);
      agx_local_atomic_to(b, dst, value, base, index, op);
   } else {
      assert(base.size == AGX_SIZE_64);
      agx_atomic_to(b, dst, value, base, index, op);
   }
}

bool
agx_lower_agx3_device_atomic_sources(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return false;

   bool progress = false;
   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_ATOMIC || I->atomic_opc == AGX_ATOMIC_OPC_CMPXCHG ||
          I->nr_dests != 1 || I->nr_srcs != 3 || agx_is_null(I->dest[0]) ||
          I->dest[0].type != AGX_INDEX_NORMAL ||
          I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
          I->src[0].type != AGX_INDEX_NORMAL ||
          I->src[0].size != AGX_SIZE_32 || agx_channels(I->src[0]) != 1)
         continue;

      /* Native G17P atomicdev has one GPR field (field16) for both the input
       * value and returned old value.  Copy the SSA value to a last-use
       * temporary; RA then assigns the atomic destination to that temporary's
       * physical register.  The address index remains an independent live
       * operand and can never be overwritten by this legalization. */
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index tied = agx_temp(ctx, AGX_SIZE_32);
      agx_mov_to(&b, tied, I->src[0]);
      I->src[0] = tied;
      progress = true;
   }

   return progress;
}

static enum agx_format
format_for_bitsize(unsigned bitsize)
{
   switch (bitsize) {
   case 8:
      return AGX_FORMAT_I8;
   case 16:
      return AGX_FORMAT_I16;
   case 32:
      return AGX_FORMAT_I32;
   default:
      UNREACHABLE("should've been lowered");
   }
}

static void
agx_emit_local_load(agx_builder *b, agx_index dst, nir_intrinsic_instr *instr)
{
   agx_index address = agx_local_base(instr->src[0]);
   const bool agx3 = b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2;
   agx_index base = agx3 ? agx_uniform(8, AGX_SIZE_16) : address;
   agx_index index = agx3 ? address : agx_zero();
   assert(address.size == AGX_SIZE_16);

   enum agx_format format = format_for_bitsize(instr->def.bit_size);
   unsigned nr = instr->def.num_components;
   unsigned mask = BITFIELD_MASK(nr);

   agx_local_load_to(b, dst, base, index, format, mask);
   agx_emit_cached_split(b, dst, nr);
}

static void
agx_emit_local_store(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index value = agx_src_index(&instr->src[0]);
   agx_index address = agx_local_base(instr->src[1]);
   const bool agx3 = b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2;
   agx_index base = agx3 ? agx_uniform(8, AGX_SIZE_16) : address;
   agx_index index = agx3 ? address : agx_zero();
   assert(address.size == AGX_SIZE_16);

   enum agx_format format = format_for_bitsize(nir_src_bit_size(instr->src[0]));
   unsigned mask = BITFIELD_MASK(
      nir_src_num_components(instr->src[0])); /* XXX: there's a write mask */

   agx_local_store(b, value, base, index, format, mask);
}

static void
agx_emit_load_scratch(agx_builder *b, agx_index dst, nir_intrinsic_instr *instr)
{
   agx_index offset = agx_src_index(&instr->src[0]);
   enum agx_format format = format_for_bitsize(instr->def.bit_size);
   unsigned nr = instr->def.num_components;
   unsigned mask = BITFIELD_MASK(nr);

   agx_stack_load_to(b, dst, offset, format, mask);
   agx_emit_cached_split(b, dst, nr);
   b->shader->any_scratch = true;
}

static void
agx_emit_store_scratch(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index value = agx_recollect_vector(b, instr->src[0]);
   agx_index offset = agx_src_index(&instr->src[1]);
   enum agx_format format = format_for_bitsize(nir_src_bit_size(instr->src[0]));
   unsigned mask = BITFIELD_MASK(nir_src_num_components(instr->src[0]));

   agx_stack_store(b, value, offset, format, mask);
   b->shader->any_scratch = true;
}

static unsigned
agx_expand_tex_to(agx_builder *b, nir_def *def, agx_index src, bool masked)
{
   unsigned nr_channels = def->num_components;
   nir_component_mask_t mask = nir_def_components_read(def);

   if (!masked)
      mask = (nir_component_mask_t)BITFIELD_MASK(nr_channels);

   agx_index packed_channels[8] = {agx_null()};
   agx_index unpacked_channels[8] = {agx_null()};

   /* Hardware writes the masked components contiguously, expand out for NIR */
   agx_emit_split(b, packed_channels, src,
                  ALIGN_POT(nr_channels, 4) /* XXX: why not nr_channels */);

   for (unsigned i = 0; i < nr_channels; ++i) {
      unpacked_channels[i] =
         (mask & BITFIELD_BIT(i))
            ? packed_channels[util_bitcount(mask & BITFIELD_MASK(i))]
            : agx_undef(src.size);
   }

   agx_emit_collect_to(b, agx_def_index(def), nr_channels, unpacked_channels);
   return mask;
}

static agx_instr *
agx_emit_image_load(agx_builder *b, agx_index dst, nir_intrinsic_instr *intr)
{
   agx_index ms_index = agx_src_index(&intr->src[2]);
   agx_index lod = agx_src_index(&intr->src[3]);
   enum agx_lod_mode lod_mode = AGX_LOD_MODE_LOD_MIN;
   bool sparse = intr->intrinsic == nir_intrinsic_image_sparse_load ||
                 intr->intrinsic == nir_intrinsic_bindless_image_sparse_load;

   agx_index bindless = agx_immediate(0), texture;
   if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
       intr->intrinsic == nir_intrinsic_bindless_image_sparse_load) {

      texture = agx_translate_bindless_handle(b, &intr->src[0], &bindless);
   } else if (nir_src_is_const(intr->src[0]) &&
              nir_src_as_uint(intr->src[0]) < 0x100) {
      texture = agx_immediate(nir_src_as_uint(intr->src[0]));
   } else {
      texture = agx_src_index(&intr->src[0]);
   }

   assert(nir_src_num_components(intr->src[1]) == 4);
   agx_index coord[4] = {
      agx_extract_nir_src(b, intr->src[1], 0),
      agx_extract_nir_src(b, intr->src[1], 1),
      agx_extract_nir_src(b, intr->src[1], 2),
      agx_extract_nir_src(b, intr->src[1], 3),
   };

   /* Get the image dimension. Cubes are lowered to 2D, since they are logically
    * equivalent for imageLoad, but out-of-bounds behaviour for cubes on G13
    * is wrong according to Piglit's arb_shader_image_load_store-invalid.
    *
    * This requires a matching transform in the driver.
    */
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   bool is_array = nir_intrinsic_image_array(intr);

   if (dim == GLSL_SAMPLER_DIM_CUBE) {
      dim = GLSL_SAMPLER_DIM_2D;
      is_array = true;
   }

   bool is_ms = dim == GLSL_SAMPLER_DIM_MS;
   unsigned coord_comps = glsl_get_sampler_dim_coordinate_components(dim);
   if (is_array && is_ms) {
      agx_index layer = agx_temp(b->shader, AGX_SIZE_16);
      agx_subdivide_to(b, layer, coord[coord_comps], 0);

      assert(ms_index.size == AGX_SIZE_16);
      agx_index tmp = agx_temp(b->shader, AGX_SIZE_32);
      agx_emit_collect_to(b, tmp, 2, (agx_index[]){ms_index, layer});
      coord[coord_comps++] = tmp;
   } else if (is_ms) {
      agx_index tmp = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, tmp, ms_index);
      coord[coord_comps++] = tmp;
   } else if (is_array) {
      coord_comps++;
   }

   /* Multisampled images do not support mipmapping */
   if (is_ms) {
      lod_mode = AGX_LOD_MODE_AUTO_LOD;
      lod = agx_zero();
   } else if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
              dim == GLSL_SAMPLER_DIM_2D && !is_array && !sparse &&
              nir_src_is_const(intr->src[3]) &&
              nir_src_as_uint(intr->src[3]) == 0) {
      /* Match texelFetch's constant-level handling. Image reads also use
       * native SMP, whose immediate level-zero form needs no PACK_SMPIR
       * staging. This applies to heap-indexed storage images and all formats;
       * neither the resource operand nor the component type changes the LOD.
       */
      lod_mode = AGX_LOD_MODE_AUTO_LOD;
      lod = agx_zero();
   }

   /* G17P's dynamic integer-level PACK_SMPIR reads a 32-bit GPR. Apple's
    * native ushort-level image kernel widens the loaded ushort before the
    * otherwise byte-identical staging/fetch sequence. Robust bindless image
    * lowering can synthesize the same semantic uint level in a 16-bit
    * register, so make that required widening explicit before RA. */
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       lod_mode == AGX_LOD_MODE_LOD_MIN && lod.size == AGX_SIZE_16 &&
       lod.type != AGX_INDEX_IMMEDIATE) {
      agx_index lod32 = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, lod32, lod);
      lod = lod32;
   }

   agx_index coords = agx_emit_collect(b, coord_comps, coord);
   agx_index tmp = agx_vec_temp(b->shader, dst.size, sparse ? 8 : 4);

   agx_instr *I = agx_image_load_to(b, tmp, coords, lod, bindless, texture,
                                    agx_immediate(0), agx_null(),
                                    agx_tex_dim(dim, is_array), lod_mode, 0,
                                    false, sparse, nir_is_coherent(intr));
   I->mask = agx_expand_tex_to(b, &intr->def, tmp, !sparse);

   b->shader->out->uses_txf = true;
   b->shader->out->sampler_state_count =
      MAX2(b->shader->out->sampler_state_count, 1);
   return NULL;
}

static agx_instr *
agx_emit_export(agx_builder *b, unsigned base, nir_src src)
{
   agx_builder b_ = *b;
   agx_cursor after_cursor = agx_after_block(agx_exit_block(b->shader));
   b_.cursor = after_cursor;

   for (unsigned c = 0; c < nir_src_num_components(src); ++c) {
      agx_index chan = agx_extract_nir_src(b, src, c);
      unsigned stride = agx_size_align_16(chan.size);

      agx_export(&b_, chan, base + (c * stride));
   }

   if (agx_cursors_equal(b->cursor, after_cursor)) {
      b->cursor = agx_after_block_logical(b->cursor.block);
   }

   return NULL;
}

static agx_instr *
agx_load_exported_to(agx_builder *b, agx_index dst, unsigned base, unsigned nr)
{
   agx_index chans[4] = {0};
   unsigned stride = agx_size_align_16(dst.size);

   for (unsigned c = 0; c < nr; ++c) {
      chans[c] = agx_cached_preload(b->shader, base + c * stride, dst.size);
   }

   return agx_emit_collect_to(b, dst, nr, chans);
}

static agx_instr *
agx_emit_image_store(agx_builder *b, nir_intrinsic_instr *instr)
{
   /* See remarks in agx_emit_image_load */
   enum glsl_sampler_dim glsl_dim = nir_intrinsic_image_dim(instr);
   bool is_array = nir_intrinsic_image_array(instr);

   if (glsl_dim == GLSL_SAMPLER_DIM_CUBE) {
      glsl_dim = GLSL_SAMPLER_DIM_2D;
      is_array = true;
   }

   enum agx_dim dim = agx_tex_dim(glsl_dim, is_array);
   assert(glsl_dim != GLSL_SAMPLER_DIM_MS && "needs to be lowered");

   agx_index base, index;
   if (instr->intrinsic == nir_intrinsic_bindless_image_store) {
      index = agx_translate_bindless_handle(b, &instr->src[0], &base);

      assert(base.size == AGX_SIZE_64);
      assert(index.size == AGX_SIZE_32);
   } else {
      base = agx_zero();
      index = agx_src_index(&instr->src[0]);

      assert(index.size == AGX_SIZE_16);
   }

   agx_index coords4 = agx_src_index(&instr->src[1]);
   agx_index lod = agx_src_index(&instr->src[4]);
   assert(lod.size == AGX_SIZE_16);

   int coord_components = glsl_get_sampler_dim_coordinate_components(glsl_dim);
   if (is_array)
      coord_components++;

   agx_index coord_comps[4] = {};
   for (unsigned i = 0; i < coord_components; ++i)
      coord_comps[i] = agx_emit_extract(b, coords4, i);

   agx_index coords = agx_emit_collect(b, coord_components, coord_comps);
   agx_index data = agx_src_index(&instr->src[3]);

   /* If the image format has less than 4 components, nir_opt_shrink_stores can
    * shrink the store. But the IR still expects 4 components: pad with undef.
    */
   if (nir_src_num_components(instr->src[3]) < 4) {
      agx_index chan[4] = {agx_null()};

      for (unsigned i = 0; i < 4; ++i) {
         if (i < nir_src_num_components(instr->src[3]))
            chan[i] = agx_extract_nir_src(b, instr->src[3], i);
         else
            chan[i] = agx_undef(data.size);
      }

      data = agx_emit_collect(b, 4, chan);
   }

   /* Image stores act like tilebuffer stores when used for tib spilling */
   b->shader->out->tag_write_disable = false;

   agx_instr *write = agx_image_write(b, data, coords, lod, base, index, dim,
                                      nir_is_coherent(instr));
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
      /* imgwci.f70 follows the TYPE OF THE DATA the shader stores, not the
       * destination surface's format. Metal never puts a pixel format in a
       * shader -- a texture2d<float, access::write> is written the same way
       * whichever surface is bound -- so f70 cannot be encoding one. Apple's
       * own applegpu_g17p output for four kernels that differ only in the
       * texture's data type, each `t.write(v, gid)`:
       *
       *   texture2d<float> d71054000040000000040000883a0900   f70:1 f71:2
       *   texture2d<uint>  d71054000040000000040000881a0900   f70:0 f71:2
       *   texture2d<int>   d71054000040000000040000881a0900   f70:0 f71:2
       *   texture2d<half>  d71054000040000000040000882a0900   f70:1 f71:1
       *
       * int and uint are byte-identical, so f70 separates float from integer
       * data; f71 is the data width, which the packer pins to 32-bit.
       *
       * Selecting on nir_intrinsic_format() reproduced that only for the
       * three formats spelled out here and failed every other one closed --
       * including a format-less image, which is what Vulkan's
       * shaderStorageImageWriteWithoutFormat gives and what DXVK's clear and
       * copy kernels all declare. Select on the stored type instead: it
       * gives the same f70 for those three formats and is the rule Apple's
       * compiler is following. */
      nir_alu_type stored =
         nir_alu_type_get_base_type(nir_intrinsic_src_type(instr));
      if (stored == nir_type_invalid) {
         /* Not every producer of an image store fills in src_type. Fall back
          * to the declared format, which says the same thing whenever it is
          * there at all. */
         switch (nir_intrinsic_format(instr)) {
         case PIPE_FORMAT_R8G8B8A8_UINT:
         case PIPE_FORMAT_R32_UINT:
            stored = nir_type_uint;
            break;
         case PIPE_FORMAT_R8G8B8A8_SINT:
         case PIPE_FORMAT_R32_SINT:
            stored = nir_type_int;
            break;
         case PIPE_FORMAT_R32_FLOAT:
         case PIPE_FORMAT_R8G8B8A8_UNORM:
         case PIPE_FORMAT_R10G10B10A2_UNORM:
            stored = nir_type_float;
            break;
         default:
            break;
         }
      }

      switch (stored) {
      case nir_type_float:
         write->format = AGX_FORMAT_U8NORM;
         break;
      case nir_type_int:
      case nir_type_uint:
         write->format = AGX_FORMAT_I8;
         break;
      default:
         /* The AGX3 packer remains fail-closed for anything unwitnessed. */
         write->format = AGX_NUM_FORMATS;
         break;
      }
   }
   return write;
}

static enum agx_simd_op
translate_simd_op(nir_op op)
{
#define CASE(agx_, nir_)                                                       \
   case nir_op_##nir_:                                                         \
      return AGX_SIMD_OP_##agx_;

   switch (op) {
      CASE(AND, iand)
      CASE(FADD, fadd)
      CASE(OR, ior)
      CASE(FMUL, fmul)
      CASE(XOR, ixor)
      CASE(FMIN, fmin)
      CASE(FMAX, fmax)
      CASE(IADD, iadd)
      CASE(SMIN, imin)
      CASE(SMAX, imax)
      CASE(UMIN, umin)
      CASE(UMAX, umax)
   default:
      UNREACHABLE("unknown simd op");
   }
#undef CASE
}

static agx_instr *
agx_emit_intrinsic(agx_builder *b, nir_intrinsic_instr *instr)
{
   agx_index dst = nir_intrinsic_infos[instr->intrinsic].has_dest
                      ? agx_def_index(&instr->def)
                      : agx_null();
   mesa_shader_stage stage = b->shader->stage;

   switch (instr->intrinsic) {
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_at_sample:
   case nir_intrinsic_load_barycentric_at_offset:
      /* handled later via load_vary */
      return NULL;
   case nir_intrinsic_load_interpolated_input:
      assert(stage == MESA_SHADER_FRAGMENT);
      agx_emit_load_vary(b, dst, instr);
      return NULL;

   case nir_intrinsic_load_coefficients_agx:
      assert(stage == MESA_SHADER_FRAGMENT);
      agx_ldcf_to(b, dst, cf_for_intrinsic(b, instr), 1);
      agx_emit_cached_split(b, dst, 3);
      return NULL;

   case nir_intrinsic_load_agx:
   case nir_intrinsic_load_constant_agx:
      agx_emit_load(b, dst, instr);
      return NULL;

   case nir_intrinsic_store_uvs_agx:
      assert(stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_TESS_EVAL);
      return agx_st_vary(b, agx_src_index(&instr->src[1]),
                         agx_src_index(&instr->src[0]));

   case nir_intrinsic_store_agx:
      agx_emit_store(b, instr);
      return NULL;

   case nir_intrinsic_store_shared:
      agx_emit_local_store(b, instr);
      return NULL;

   case nir_intrinsic_load_shared:
      agx_emit_local_load(b, dst, instr);
      return NULL;

   case nir_intrinsic_global_atomic_agx:
   case nir_intrinsic_global_atomic_swap_agx:
      agx_emit_atomic(b, dst, instr, false);
      return NULL;

   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
      agx_emit_atomic(b, dst, instr, true);
      return NULL;

   case nir_intrinsic_store_zs_agx:
      assert(stage == MESA_SHADER_FRAGMENT);
      return agx_emit_store_zs(b, instr);

   case nir_intrinsic_store_local_pixel_agx:
      return agx_emit_local_store_pixel(b, instr);

   case nir_intrinsic_load_local_pixel_agx:
      assert(stage == MESA_SHADER_FRAGMENT);
      agx_emit_local_load_pixel(b, dst, instr);
      return NULL;

   case nir_intrinsic_load_pixel_coord:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX2) {
         return agx_emit_collect_to(
            b, dst, 2,
            (agx_index[2]){
               agx_get_sr(b, 16, AGX_SR_THREAD_POSITION_IN_GRID_X),
               agx_get_sr(b, 16, AGX_SR_THREAD_POSITION_IN_GRID_Y),
            });
      } else {
         agx_index xy[2];
         for (unsigned i = 0; i < 2; i++) {
            agx_index native = agx_temp(b->shader, AGX_SIZE_32);
            agx_get_sr_native_width_to(
               b, native, AGX_SR_THREAD_POSITION_IN_GRID_X + i);
            xy[i] = agx_temp(b->shader, AGX_SIZE_16);
            agx_subdivide_to(b, xy[i], native, 0);
         }
         return agx_emit_collect_to(b, dst, 2, xy);
      }

   case nir_intrinsic_load_frag_coord_z: {
      agx_index cf = agx_get_cf(b->shader, VARYING_SLOT_POS, 2);

      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
         return agx_interp_agx3_to(b, dst, cf, agx_zero(),
                                     AGX_INTERPOLATION_CENTER, false);

      return agx_iter_to(b, dst, cf, agx_zero(), 1, AGX_INTERPOLATION_CENTER);
   }

   case nir_intrinsic_load_frag_coord_w: {
      agx_index cf = agx_get_cf(b->shader, VARYING_SLOT_POS, 3);

      /* Native position.w uses ordinary interpolation (f54=0). The
       * projective RHW form is private to reciprocal interpolation and
       * cannot be reused as a public arithmetic value. */
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
         return agx_interp_agx3_to(b, dst, cf, agx_zero(),
                                     AGX_INTERPOLATION_CENTER, false);

      return agx_iter_to(b, dst, cf, agx_zero(), 1, AGX_INTERPOLATION_CENTER);
   }

   case nir_intrinsic_sample_mask_agx: {
      assert(stage == MESA_SHADER_FRAGMENT);
      b->shader->out->writes_sample_mask = true;

      /* We need to wait_pix before running Z/S tests, but we don't need to
       * wait_pix before merely discarding. Omit the wait_pix when the affected
       * samples are unconditionally killed.
       */
      bool no_tests =
         nir_src_is_const(instr->src[1]) && nir_src_as_uint(instr->src[1]) == 0;

      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          !b->shader->key->fs.inside_sample_loop) {
         /* MOVMSK takes one coverage word: the samples that stay alive in
          * nibble 0, above the pixel's coverage and API input mask. Native
          * early-fragment-tests code uses word zero for a no-op. Keep TARGET
          * dynamic and select that word without changing EXEC around a
          * coverage operation. */
         const unsigned samples = MAX2(b->shader->key->fs.nr_samples, 1);
         const unsigned all = BITFIELD_MASK(samples);
         agx_index target_in = agx_src_index(&instr->src[0]);
         agx_index live_in = agx_src_index(&instr->src[1]);
         agx_index target, keep, mid;

         if (samples == 1) {
            /* Apple's one-sample fragments read SR2 and repeat it in both
             * coverage nibbles. */
            target = agx_and(b, target_in, agx_immediate(1));
            keep = agx_and(b, live_in, agx_immediate(1));
            agx_index t16 = agx_temp(b->shader, AGX_SIZE_16);
            agx_get_coverage_1x_agx3_to(b, t16);
            mid = agx_iadd(b, t16, t16, 4);
         } else {
            /* At two and four samples Apple reads SR39, whose low byte holds
             * the covered samples and the API input mask, masks the coverage
             * nibble and recombines them (metal-nt -arch applegpu_g17p,
             * [[sample_mask]] fragments at raster_sample_count 2 and 4).
             * Untargeted samples keep their state, and a word targeting no
             * sample is the no-op, selected by (target + all) >> samples. */
            agx_index masked = agx_and(b, target_in, agx_immediate(all));
            agx_index untargeted =
               agx_xor(b, masked, agx_immediate(all));
            keep = agx_and(b, agx_or(b, untargeted, live_in),
                           agx_immediate(all));
            target = agx_ushr(b, agx_iadd(b, masked, agx_immediate(all), 0),
                              agx_immediate(samples));
            agx_index t16 = agx_temp(b->shader, AGX_SIZE_16);
            agx_get_sr_to(b, t16, AGX_SR_INPUT_SAMPLE_MASK);
            agx_index upper = agx_ushr(b, t16, agx_immediate(4));
            agx_index covered = agx_and(b, t16, agx_immediate(15));
            mid = agx_iadd(b, covered, upper, 4);
         }

         agx_index word = agx_iadd(b, keep, mid, 4);
         agx_index zero = agx_mov_imm(b, 16, 0);
         agx_index target32 = agx_temp(b->shader, AGX_SIZE_32);
         agx_index word32 = agx_temp(b->shader, AGX_SIZE_32);
         agx_emit_collect_to(b, target32, 2, (agx_index[]){target, zero});
         agx_emit_collect_to(b, word32, 2, (agx_index[]){word, zero});
         agx_index selected = agx_temp(b->shader, AGX_SIZE_32);
         agx_imad_to(b, selected, target32, word32, agx_zero(), 0);
         return agx_sample_mask_agx3(b, target, selected, !no_tests ? 2 : 0);
      }

      if (!no_tests)
         agx_wait_pixel_mask(b, 0x1);

      return agx_sample_mask(b, agx_src_index(&instr->src[0]),
                             agx_src_index(&instr->src[1]));
   }

   case nir_intrinsic_load_back_face_agx:
      return agx_get_sr_to(b, dst, AGX_SR_BACKFACING);

   case nir_intrinsic_load_samples_log2_agx:
      return agx_get_sr_to(b, dst, AGX_SR_SAMPLES_LOG2);

   case nir_intrinsic_load_sample_mask_in:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         /* Apple's G17P fragment ABI exposes the API-visible sample mask in
          * bits 4..7 of native W16 SR39.  Extract it before widening to NIR's
          * uint32_t result; reading SR39 as W32 selects an unwitnessed form. */
         agx_index native = agx_temp(b->shader, AGX_SIZE_16);
         agx_index shifted = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_to(b, native, AGX_SR_INPUT_SAMPLE_MASK);
         agx_ushr_to(b, shifted, native, agx_immediate(4));
         return agx_mov_to(b, dst, shifted);
      }
      return agx_get_sr_to(b, dst, AGX_SR_INPUT_SAMPLE_MASK);

   case nir_intrinsic_load_sample_mask:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         /* G17P's covered samples are bits 0..3 of native W16 SR39, the word
          * Apple's multisample fragments AND with each sample's bit before
          * that sample's tile store. */
         agx_index native = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_to(b, native, AGX_SR_INPUT_SAMPLE_MASK);
         return agx_mov_to(b, dst, agx_and(b, native, agx_immediate(15)));
      }
      return agx_get_sr_coverage_to(b, dst, AGX_SR_COVERAGE_MASK);

   case nir_intrinsic_load_active_samples_agx:
      /* Everywhere else this is a loop variable that the MSAA lowerings have
       * already substituted. It survives to the backend only on G17P, where
       * agx_nir_lower_g17p_tile_samples asks for the live coverage word as a
       * real register operand for lstcol/lldcol: Apple's own multisample
       * fragments read exactly this, native W16 SR39, and pass it straight to
       * the tile access. Bits 0..3 are the covered samples -- the per-sample
       * form ANDs this same word with 1/2/4/8. */
      assert(b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             "only the G17P tile lowering leaves this for the backend");
      assert(instr->def.bit_size == 16 && "16-bit sample mask");
      return agx_get_sr_to(b, dst, AGX_SR_INPUT_SAMPLE_MASK);

   case nir_intrinsic_load_helper_invocation:
      /* Compare special register to zero. We could lower this in NIR (letting
       * us fold in an inot) but meh?
       */
      return agx_icmp_to(b, dst,
                         agx_get_sr_coverage(
                            b, b->shader->key->dev.encoding ==
                                  AGX_ISA_ENCODING_AGX2 ? 32 : 16,
                            AGX_SR_IS_ACTIVE_THREAD),
                         agx_zero(), AGX_ICOND_UEQ, false);

   case nir_intrinsic_load_vertex_id_zero_base:
      assert(stage == MESA_SHADER_COMPUTE && "only for SW VS");
      return agx_mov_to(b, dst, agx_abs(agx_vertex_id_zero_base(b)));

   case nir_intrinsic_load_vertex_id:
      /* AGX3 hardware vertex shaders read the invocation index explicitly.
       * Apple's isolated G17P vertex-attribute shaders begin with
       * `0cdd1006`, a 32-bit rld from SR 0xdd. Retaining AGX2's preloaded r5
       * ABI made every invocation observe the same index and collapse a
       * triangle to one vertex. Software vertex shaders still use the preload
       * ABI populated by their compute launcher. */
      if (stage == MESA_SHADER_VERTEX &&
          b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2)
         return agx_get_sr_to(b, dst, AGX_SR_VERTEX_ID);

      return agx_mov_to(b, dst, agx_abs(agx_vertex_id(b)));

   case nir_intrinsic_load_primitive_id:
      /* AGX3's tessellator hands the patch index to the post-tessellation
       * vertex program in SR 0xd9, Apple's [[patch_id]]. */
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         assert(stage == MESA_SHADER_VERTEX && "post-tessellation vertex");
         return agx_get_sr_to(b, dst, AGX_SR_TESS_PATCH_ID);
      }

      assert(stage == MESA_SHADER_TESS_EVAL);
      return agx_mov_to(b, dst, agx_abs(agx_vertex_id(b)));

   case nir_intrinsic_load_instance_id:
      /* The matching Apple hardware-vertex witness is `0cd81006`, reading
       * instance ID from SR 0xd8. */
      if (stage == MESA_SHADER_VERTEX &&
          b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2)
         return agx_get_sr_to(b, dst, AGX_SR_INSTANCE_ID);

      return agx_mov_to(b, dst, agx_abs(agx_instance_id(b)));

   case nir_intrinsic_load_tess_coord_xy: {
      /* On AGX3 the tessellator's position in the patch arrives as two
       * 32-bit float special registers, Apple's [[position_in_patch]]. */
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         assert(stage == MESA_SHADER_VERTEX && "post-tessellation vertex");
         agx_index coords[] = {agx_get_sr(b, 32, AGX_SR_TESS_COORD_U),
                               agx_get_sr(b, 32, AGX_SR_TESS_COORD_V)};
         return agx_emit_collect_to(b, dst, 2, coords);
      }

      assert(stage == MESA_SHADER_TESS_EVAL);

      agx_index coords[] = {agx_tess_coord_x(b), agx_tess_coord_y(b)};
      return agx_emit_collect_to(b, dst, 2, coords);
   }

   case nir_intrinsic_load_preamble:
      return agx_emit_load_preamble(b, dst, instr);

   case nir_intrinsic_store_preamble:
      return agx_emit_store_preamble(b, instr);

   case nir_intrinsic_image_load:
   case nir_intrinsic_image_sparse_load:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
      return agx_emit_image_load(b, dst, instr);

   case nir_intrinsic_image_store:
   case nir_intrinsic_bindless_image_store:
      return agx_emit_image_store(b, instr);

   case nir_intrinsic_image_store_block_agx:
   case nir_intrinsic_bindless_image_store_block_agx:
      return agx_emit_block_image_store(b, instr);

   case nir_intrinsic_load_workgroup_id:
      return agx_load_compute_dimension(b, dst, instr,
                                        AGX_SR_THREADGROUP_POSITION_IN_GRID_X);

   case nir_intrinsic_load_workgroup_size:
      return agx_load_compute_dimension(b, dst, instr,
                                        AGX_SR_THREADS_PER_THREADGROUP_X);

   case nir_intrinsic_load_global_invocation_id:
      return agx_load_compute_dimension(b, dst, instr,
                                        AGX_SR_THREAD_POSITION_IN_GRID_X);

   case nir_intrinsic_load_local_invocation_id:
      return agx_load_compute_dimension(
         b, dst, instr, AGX_SR_THREAD_POSITION_IN_THREADGROUP_X);

   case nir_intrinsic_load_local_invocation_index:
      return agx_get_sr_native_width_to(
         b, dst, AGX_SR_THREAD_INDEX_IN_THREADGROUP);

   case nir_intrinsic_load_layer_id:
      return agx_get_sr_to(b, dst, AGX_SR_THREADGROUP_POSITION_IN_GRID_Z);

   case nir_intrinsic_barrier: {
      assert(!b->shader->is_preamble && "invalid");

      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         bool emitted = false;

         if (nir_intrinsic_memory_scope(instr) != SCOPE_NONE) {
            const nir_variable_mode modes =
               nir_intrinsic_memory_modes(instr);

            /* The AGX2 cache-operation train collapses on AGX3 into one
             * exact barrier per semantic memory scope. Keep Apple's observed
             * device -> texture -> threadgroup ordering. A tessellation
             * control shader runs as a compute kernel whose outputs poly
             * lowered to the global output buffer, so its output barrier is
             * a device-memory barrier. */
            const bool tcs_outputs =
               b->shader->nir->info.stage == MESA_SHADER_TESS_CTRL &&
               (modes & nir_var_shader_out);
            if ((modes & (nir_var_mem_global | nir_var_image)) ||
                tcs_outputs) {
               agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_DEVICE);
               emitted = true;
            }

            if (modes & nir_var_image) {
               agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE);
               emitted = true;
            }

            if (modes & nir_var_mem_shared) {
               agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_THREADGROUP);
               emitted = true;
            }
         }

         /* A workgroup execution barrier with no memory scope is the
          * recovered mem_none form. Scoped memory barriers already carry the
          * execution rendezvous, as shown by each Metal mem_flags contrast. */
         if (nir_intrinsic_execution_scope(instr) >= SCOPE_WORKGROUP &&
             !emitted) {
            assert(mesa_shader_stage_is_compute(b->shader->nir->info.stage));
            agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_EXECUTION);
         }

         return NULL;
      }

      bool needs_image_barriers = false;

      if (nir_intrinsic_memory_scope(instr) != SCOPE_NONE) {
         nir_variable_mode modes = nir_intrinsic_memory_modes(instr);

         if (modes & (nir_var_mem_global | nir_var_image)) {
            agx_memory_barrier(b);

            /* Pull out all the big hammers to make cross-workgroup memory
             * barriers work.
             */
            if (nir_intrinsic_memory_scope(instr) >= SCOPE_QUEUE_FAMILY) {
               agx_unknown_barrier_1(b);
               agx_memory_barrier_2(b);

               /* These are observed on G13D. At some point we should figure out
                * what the individual opcodes do.
                */
               agx_device_barrier_2(b);
               agx_unknown_barrier_2(b);
               agx_memory_barrier_3(b);
               agx_device_barrier_1(b);
            }
         }

         if (modes & nir_var_image) {
            agx_image_barrier_1(b);
            agx_image_barrier_2(b);
            needs_image_barriers = true;
         }
      }

      /* Nothing to do for subgroup barriers */
      if (nir_intrinsic_execution_scope(instr) >= SCOPE_WORKGROUP) {
         assert(mesa_shader_stage_is_compute(b->shader->nir->info.stage));

         agx_threadgroup_barrier(b);
      }

      if (needs_image_barriers) {
         agx_image_barrier_3(b);
         agx_image_barrier_4(b);
      }

      return NULL;
   }

   case nir_intrinsic_fence_pbe_to_tex_agx: {
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE);
         return NULL;
      }

      agx_image_barrier_1(b);
      agx_image_barrier_2(b);
      agx_image_barrier_3(b);
      agx_image_barrier_4(b);
      return NULL;
   }

   case nir_intrinsic_fence_mem_to_tex_agx: {
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_DEVICE);
         agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE);
         return NULL;
      }

      /* Flush out the atomic to main memory... Found experimentally... */
      agx_memory_barrier(b);
      agx_memory_barrier_2(b);

      /* TODO: Which ones do we actually need? */
      agx_image_barrier_1(b);
      agx_image_barrier_2(b);
      agx_image_barrier_3(b);
      agx_image_barrier_4(b);

      /* Flush out the texture cache */
      agx_flush_memory_to_texture(b);
      return NULL;
   }

   case nir_intrinsic_fence_pbe_to_tex_pixel_agx: {
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE);
         return NULL;
      }

      agx_image_barrier_1(b);
      agx_image_barrier_2(b);
      agx_flush_memory_to_texture(b);
      agx_image_barrier_3(b);
      return NULL;
   }

   case nir_intrinsic_fence_helper_exit_agx: {
      assert(b->shader->key->is_helper);
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2) {
         agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_DEVICE);
         return NULL;
      }

      agx_memory_barrier(b);
      agx_unknown_barrier_1(b);
      agx_memory_barrier_2(b);
      agx_unknown_barrier_2(b);
      agx_memory_barrier_3(b);
      return NULL;
   }

   case nir_intrinsic_begin_invocation_interlock: {
      agx_wait_pixel_mask(b, 0xC);
      return NULL;
   }

   case nir_intrinsic_ddx:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddx_fine:
      return agx_dfdx_to(b, dst, agx_src_index(&instr->src[0]));

   case nir_intrinsic_ddy:
   case nir_intrinsic_ddy_coarse:
   case nir_intrinsic_ddy_fine:
      return agx_dfdy_to(b, dst, agx_src_index(&instr->src[0]));

   case nir_intrinsic_load_subgroup_invocation:
      return agx_get_sr_native_width_to(b, dst,
                                        AGX_SR_THREAD_INDEX_IN_SUBGROUP);

   case nir_intrinsic_load_subgroup_id:
      return agx_get_sr_native_width_to(
         b, dst, AGX_SR_SUBGROUP_INDEX_IN_THREADGROUP);

   case nir_intrinsic_load_active_subgroup_invocation_agx:
      return agx_get_sr_coverage_to(b, dst,
                                    AGX_SR_ACTIVE_THREAD_INDEX_IN_SUBGROUP);

   case nir_intrinsic_load_active_subgroup_count_agx:
      /* AGX3 reads the count as a 16-bit value, as Apple's uniform atomics
       * do; zero-extend it for the 32-bit consumers NIR expects. */
      if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX2 &&
          dst.size == AGX_SIZE_32) {
         agx_index native = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_coverage_to(b, native,
                                AGX_SR_TOTAL_ACTIVE_THREADS_IN_SUBGROUP);
         return agx_mov_to(b, dst, native);
      }

      return agx_get_sr_coverage_to(b, dst,
                                    AGX_SR_TOTAL_ACTIVE_THREADS_IN_SUBGROUP);

   case nir_intrinsic_reduce: {
      assert((instr->def.bit_size == 1 || instr->def.bit_size == 16 ||
              instr->def.bit_size == 32) &&
             "should've been lowered");

      unsigned cluster_size = nir_intrinsic_cluster_size(instr);
      assert(cluster_size == 0 || cluster_size == 4 || cluster_size >= 32);

      enum agx_simd_op op =
         translate_simd_op(nir_intrinsic_reduction_op(instr));

      agx_index src0 = agx_src_index(&instr->src[0]);

      if (cluster_size == 4)
         return agx_quad_reduce_to(b, dst, src0, op);
      else
         return agx_simd_reduce_to(b, dst, src0, op);
   }

   case nir_intrinsic_exclusive_scan: {
      assert((instr->def.bit_size == 1 || instr->def.bit_size == 16 ||
              instr->def.bit_size == 32) &&
             "should've been lowered");

      return agx_simd_prefix_to(
         b, dst, agx_src_index(&instr->src[0]),
         translate_simd_op(nir_intrinsic_reduction_op(instr)));
   }

   case nir_intrinsic_read_invocation: {
      /* TODO: Check if we're actually inside divergent control flow */
      b->shader->any_quad_divergent_shuffle |= b->shader->any_cf;

      /* Lane ID guaranteed to be uniform */
      agx_index lane = agx_src_index(&instr->src[1]);

      /*
       * AGX3's immediate-lane broadcast puts the lane in a seven-bit field,
       * so lanes 0..127 -- every lane a subgroup can have -- encode directly.
       * A constant outside that range is undefined behaviour rather than a
       * program we must reject, so materialise it into the register-lane
       * form, which takes any value, instead of refusing the shader.
       */
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          lane.type == AGX_INDEX_IMMEDIATE && lane.value > 0x7f) {
         agx_index reg = agx_temp(b->shader, AGX_SIZE_32);
         agx_mov_imm_to(b, reg, lane.value);
         lane = reg;
      }

      return agx_shuffle_to(b, dst, agx_src_index(&instr->src[0]), lane);
   }

   case nir_intrinsic_quad_broadcast: {
      /* TODO: Check if we're actually inside divergent control flow */
      b->shader->any_quad_divergent_shuffle |= b->shader->any_cf;

      /* Lane ID guaranteed to be uniform */
      return agx_quad_shuffle_to(b, dst, agx_src_index(&instr->src[0]),
                                 agx_src_index(&instr->src[1]));
   }

   case nir_intrinsic_quad_swap_horizontal: {
      return agx_quad_shuffle_xor_to(b, dst, agx_src_index(&instr->src[0]),
                                     agx_immediate(1));
   }

   case nir_intrinsic_quad_swap_vertical: {
      return agx_quad_shuffle_xor_to(b, dst, agx_src_index(&instr->src[0]),
                                     agx_immediate(2));
   }

   case nir_intrinsic_quad_swap_diagonal: {
      return agx_quad_shuffle_xor_to(b, dst, agx_src_index(&instr->src[0]),
                                     agx_immediate(3));
   }

   case nir_intrinsic_ballot: {
      return agx_ballot_to(b, dst, agx_src_index(&instr->src[0]));
   }

   case nir_intrinsic_quad_ballot_agx: {
      return agx_quad_ballot_to(b, dst, agx_src_index(&instr->src[0]));
   }

   case nir_intrinsic_doorbell_agx: {
      return agx_doorbell(b, nir_src_as_uint(instr->src[0]));
   }

   case nir_intrinsic_stack_map_agx: {
      return agx_stack_map(b, agx_src_index(&instr->src[1]),
                           nir_src_as_uint(instr->src[0]));
   }

   case nir_intrinsic_stack_unmap_agx: {
      return agx_stack_unmap_to(b, dst, nir_src_as_uint(instr->src[0]));
   }

   case nir_intrinsic_load_scratch:
      agx_emit_load_scratch(b, dst, instr);
      return NULL;

   case nir_intrinsic_store_scratch:
      agx_emit_store_scratch(b, instr);
      return NULL;

   case nir_intrinsic_load_core_id:
      return agx_get_sr_to(b, dst, AGX_SR_CORE_ID);

   case nir_intrinsic_load_helper_op_id_agx:
      assert(b->shader->key->is_helper);
      return agx_get_sr_barrier_to(b, dst, AGX_SR_HELPER_OP);

   case nir_intrinsic_load_helper_arg_lo_agx:
      assert(b->shader->key->is_helper);
      return agx_get_sr_barrier_to(b, dst, AGX_SR_HELPER_ARG_L);

   case nir_intrinsic_load_helper_arg_hi_agx:
      assert(b->shader->key->is_helper);
      return agx_get_sr_barrier_to(b, dst, AGX_SR_HELPER_ARG_H);

   case nir_intrinsic_load_exported_agx:
      return agx_load_exported_to(b, dst, nir_intrinsic_base(instr),
                                  instr->def.num_components);

   case nir_intrinsic_export_agx:
      return agx_emit_export(b, nir_intrinsic_base(instr), instr->src[0]);

   case nir_intrinsic_bindless_image_agx:
   case nir_intrinsic_bindless_sampler_agx:
      /* These must always be chased */
      return NULL;

   case nir_intrinsic_load_barycentric_sample:
   case nir_intrinsic_load_sample_id:
   case nir_intrinsic_load_sample_pos:
      UNREACHABLE("Sample shading should have been lowered");

   default:
      fprintf(stderr, "Unhandled intrinsic %s\n",
              nir_intrinsic_infos[instr->intrinsic].name);
      UNREACHABLE("Unhandled intrinsic");
   }
}

static agx_index
agx_alu_src_index(agx_builder *b, nir_alu_src src)
{
   /* Check well-formedness of the input NIR */
   ASSERTED unsigned bitsize = nir_src_bit_size(src.src);
   unsigned comps = nir_src_num_components(src.src);
   unsigned channel = src.swizzle[0];

   assert(bitsize == 1 || bitsize == 8 || bitsize == 16 || bitsize == 32 ||
          bitsize == 64);
   assert(channel < comps);

   return agx_extract_nir_src(b, src.src, channel);
}

/*
 * Emit an instruction translating (s0 * s1) + (s2 << s3). Assuming s3 is
 * constant, this is an imad instruction. If s1 == 1, then this is optimized to
 * an iadd instruction, which is faster.
 */
static agx_instr *
agx_emit_imadshl_agx(agx_builder *b, nir_alu_instr *alu, agx_index dst,
                     agx_index s0, agx_index s1, agx_index s2, agx_index s3)
{
   /* The immediate iadd/imad encoding accepts only shifts 0..4.  NIR normally
    * guarantees that domain, but this backend is also a process boundary for
    * arbitrary DXIL- and SPIR-V-derived shaders.  Keep every out-of-domain
    * value on the general shift path instead of letting the packer's
    * always-live domain check take down a release process.
    */
   unsigned shift;
   bool constant_shift = nir_src_is_const(alu->src[3].src);

   if (constant_shift)
      shift = nir_alu_src_as_uint(alu->src[3]);

   if (!constant_shift || shift > 4) {
      s2 = agx_bfi(b, agx_immediate(0), s2, s3, 0);
      shift = 0;
   }

   /* Only the immediate-encoding path reaches this assertion. */
   assert(shift <= 4 && "domain restriction on the input NIR");

   /* Emit iadd if possible, else imad */
   if (nir_src_is_const(alu->src[1].src) &&
       nir_alu_src_as_uint(alu->src[1]) == 1) {

      return agx_iadd_to(b, dst, s0, s2, shift);
   } else {
      /*
       * AGX3 has no shifted imad. Apple's own compiler does not emit one:
       * across `o[i] = a[i]*b[i] + (c[i] << N)` kernels for N = 1, 2 and 3,
       * the imadd is BYTE-IDENTICAL and the shift appears on a following
       * iadd instead. So split it the same way rather than inventing a shift
       * field for an instruction that has none.
       *
       * imad(a, b, c, s) is a*b + (c << s), and integer wraparound makes the
       * split exact: the product is computed unshifted, then added with the
       * shift on the addend. Both halves are witnessed byte-exactly -- the
       * pure-multiply imadd and the shifted iadd -- in test-agx3-iadd-core.c.
       */
      const enum agx_isa_encoding enc = b->shader->key->dev.encoding;
      const bool agx3 = enc != AGX_ISA_ENCODING_AGX2;

      if (agx3 && shift != 0) {
         agx_index product = agx_temp(b->shader, dst.size);
         agx_imad_to(b, product, s0, s1, agx_immediate(0), 0);
         return agx_iadd_to(b, dst, product, s2, shift);
      }

      return agx_imad_to(b, dst, s0, s1, s2, shift);
   }
}

static bool
is_conversion_to_8bit(nir_op op)
{
   switch (op) {
   case nir_op_i2i8:
   case nir_op_u2u8:
   case nir_op_f2i8:
   case nir_op_f2u8:
   case nir_op_b2i8:
      return true;
   default:
      return false;
   }
}

static agx_instr *
agx_fminmax_to(agx_builder *b, agx_index dst, agx_index s0, agx_index s1,
               nir_alu_instr *alu)
{
   /* The hardware gtn/ltn modes are unfortunately incorrect for signed zeros */
   assert(!nir_alu_instr_is_signed_zero_preserve(alu) &&
          "should've been lowered");

   assert((alu->def.bit_size == 16) ==
             (alu->op == nir_op_fmin || alu->op == nir_op_fmax) &&
          "fp32 should be lowered");

   bool fmax = alu->op == nir_op_fmax || alu->op == nir_op_fmax_agx;
   enum agx_fcond fcond = fmax ? AGX_FCOND_GTN : AGX_FCOND_LTN;

   /* Calculate min/max with the appropriate hardware instruction. This will not
    * handle denorms, but we were already lowered for that.
    */
   return agx_fcmpsel_to(b, dst, s0, s1, s0, s1, fcond);
}

/* A fragment shader that reads a varying array at a run-time index computes
 * its coefficient index as dest16 = imm + (index << 2), because the ITERPROJ
 * coefficient operand is sixteen bits wide. AGX3B's shifted add
 * (agx3_iadd_core.h) takes a narrow *source* but always writes a 32-bit
 * register, so that shape was refused and the whole shader with it -- which
 * is KHR-GL33.shaders.indexing.varying_array's four *_dynamic_read cases.
 *
 * Give the add the wide destination it requires and narrow afterwards. The
 * low sixteen bits are unchanged by widening the sum, so this is exact.
 */
static void
agx_lower_agx3_shifted_iadd_dest(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_IADD || I->nr_dests != 1 || I->nr_srcs != 2 ||
          I->shift == 0 || I->saturate ||
          I->dest[0].size != AGX_SIZE_16 || agx_channels(I->dest[0]) != 1)
         continue;

      
/* Only the constant-addend form (agx3_iadd_core.h) is affected: it is
       * the one that always writes a 32-bit register. A shifted add of two
       * registers has its own encoding and keeps its narrow destination. */
      if (I->src[0].type != AGX_INDEX_IMMEDIATE ||
          I->src[1].type == AGX_INDEX_IMMEDIATE)
         continue;

      agx_builder b = agx_init_builder(ctx, agx_after_instr(I));
      agx_index narrow = I->dest[0];
      agx_index wide = agx_temp(ctx, AGX_SIZE_32);
      I->dest[0] = wide;
      agx_subdivide_to(&b, narrow, wide, 0);
   }
}

/* AGX3B's transcendental forms take one width: agx3_alu_hexp2_neg() for a
 * half pair, agx3_alu_f32exp2_mod() for a word pair, and no mixed form. The
 * exponent of the level-of-detail calculation textureGrad needs arrives as an
 * exp2 of a 16-bit value into a 32-bit result, which was refused -- and a
 * refusal takes the whole shader, which is what
 * KHR-GL33.texture_swizzle's eight remaining smoke cases hit.
 *
 * Widen the source. Half to float is exact, so no value changes; it is the
 * same move agx_lower_agx3_fma_sources() makes for a mixed FFMA. Only the
 * mismatched shape is touched, and that shape did not encode at all before.
 */
static void
agx_lower_agx3_mixed_width_transcendental(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if ((I->op != AGX_OPCODE_EXP2 && I->op != AGX_OPCODE_LOG2) ||
          I->nr_dests != 1 || I->nr_srcs != 1 || I->saturate)
         continue;

      if (I->dest[0].size != AGX_SIZE_32 || I->src[0].size != AGX_SIZE_16 ||
          agx_channels(I->src[0]) != 1 ||
          I->src[0].type == AGX_INDEX_IMMEDIATE ||
          I->src[0].type == AGX_INDEX_NULL)
         continue;

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_index widened = agx_temp(ctx, AGX_SIZE_32);
      agx_fmov_to(&b, widened, I->src[0]);
      I->src[0] = widened;
   }
}
static void
agx_lower_agx3_fma_sources(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if (I->op != AGX_OPCODE_FFMA || I->nr_srcs != 3 || I->saturate)
         continue;

      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
      agx_foreach_src(I, s) {
         if (I->src[s].size != AGX_SIZE_16 ||
             I->src[s].type == AGX_INDEX_IMMEDIATE ||
             I->src[s].type == AGX_INDEX_NULL || I->src[s].abs ||
             I->src[s].neg)
            continue;

         agx_index widened = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, widened, I->src[s]);
         I->src[s] = widened;
      }

      bool mixed = false, has_neg = false, unsupported_modifier = false;
      agx_foreach_src(I, s) {
         mixed |= I->src[s].type == AGX_INDEX_IMMEDIATE ||
                  I->src[s].size != AGX_SIZE_32;
         has_neg |= I->src[s].neg;
         unsupported_modifier |= I->src[s].abs ||
                                 (I->src[s].neg &&
                                  (I->src[s].type == AGX_INDEX_IMMEDIATE ||
                                   I->src[s].size != AGX_SIZE_32));
      }

      /* The native modifier form takes three full F32 registers. Preserve
       * exact fneg semantics for mixed/minifloat FMA shapes by flipping the
       * source sign bit before register allocation. The resulting FMA has no
       * modifier and stays on the already-grounded mixed-source form. */
      if (!mixed || !has_neg || unsupported_modifier)
         continue;

      agx_foreach_src(I, s) {
         if (!I->src[s].neg)
            continue;

         agx_index source = I->src[s];
         source.neg = false;
         agx_index sign = agx_mov_imm(&b, 32, UINT32_C(0x80000000));
         agx_index flipped = agx_temp(ctx, AGX_SIZE_32);
         agx_bitop_to(&b, flipped, source, sign, 0x6);
         I->src[s] = flipped;
      }
   }
}

static void
agx_materialize_agx3_alu_constants(agx_context *ctx)
{
   agx_foreach_instr_global_safe(ctx, I) {
      agx_builder b = agx_init_builder(ctx, agx_before_instr(I));

      /* A half fabs is initially represented as fadd(abs(x), -0.0h).  The
       * G17P half-add form has no absolute-source field, and leaving this
       * pseudo operation intact until after register allocation makes the
       * packer refuse frexp/modf precision shaders.  fabs on an IEEE half is
       * exactly a sign-bit clear, so lower that exact move to the grounded
       * half BITOP path while the mask can still receive an SSA register. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FADD && I->nr_dests == 1 && I->nr_srcs == 2 &&
          I->dest[0].size == AGX_SIZE_16 && !I->saturate &&
          I->src[0].size == AGX_SIZE_16 && I->src[0].abs &&
          !I->src[0].neg &&
          agx_is_equiv(I->src[1], agx_immediate(UINT8_C(0x80)))) {
         agx_index source = I->src[0];
         source.abs = false;
         agx_index mask = agx_mov_imm(&b, 16, UINT16_C(0x7fff));

         I->op = AGX_OPCODE_BITOP;
         I->src[0] = source;
         I->src[1] = mask;
         I->truth_table = 0x8; /* AND */
      }

      /* The grounded scalar-half FSEL and rounding forms do not carry float
       * source modifiers. CTS reaches both through frexp/modf, with fabs
       * folded into a comparator or floor input. Preserve every half bit by
       * materializing abs/neg as sign-bit operations before RA instead of
       * refusing the post-RA instruction. */
      const bool half_fsel =
         ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
         I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
         I->dest[0].size == AGX_SIZE_16;
      const bool half_round =
         ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
         (I->op == AGX_OPCODE_FLOOR || I->op == AGX_OPCODE_CEIL) &&
         I->nr_srcs == 1 && I->dest[0].size == AGX_SIZE_16;

      if (half_fsel || half_round) {
         const unsigned source_count = half_fsel ? 4 : 1;
         for (unsigned s = 0; s < source_count; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE ||
                I->src[s].size != AGX_SIZE_16 ||
                (!I->src[s].abs && !I->src[s].neg))
               continue;

            const bool absolute = I->src[s].abs;
            const bool negate = I->src[s].neg;
            agx_index value = I->src[s];
            value.abs = false;
            value.neg = false;

            if (absolute) {
               agx_index mask = agx_mov_imm(&b, 16, UINT16_C(0x7fff));
               agx_index result = agx_temp(ctx, AGX_SIZE_16);
               agx_bitop_to(&b, result, value, mask, 0x8 /* AND */);
               value = result;
            }

            if (negate) {
               agx_index sign = agx_mov_imm(&b, 16, UINT16_C(0x8000));
               agx_index result = agx_temp(ctx, AGX_SIZE_16);
               agx_bitop_to(&b, result, value, sign, 0x6 /* XOR */);
               value = result;
            }

            I->src[s] = value;
         }
      }

      /* frexp's full-precision path compares zero against fabs(x) and
       * selects two full floats.  G17P's zero-comparison FSEL is grounded
       * only without source modifiers, so make fabs/fneg exact with integer
       * sign-bit operations while the modified comparator is still SSA. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32) {
         const bool zero_first =
            I->src[0].type == AGX_INDEX_IMMEDIATE && I->src[0].value == 0;
         const bool zero_second =
            I->src[1].type == AGX_INDEX_IMMEDIATE && I->src[1].value == 0;

         if (zero_first != zero_second) {
            const unsigned s = zero_first ? 1 : 0;
            if (I->src[s].type != AGX_INDEX_IMMEDIATE &&
                I->src[s].size == AGX_SIZE_32 &&
                (I->src[s].abs || I->src[s].neg)) {
               const bool absolute = I->src[s].abs;
               const bool negate = I->src[s].neg;
               agx_index value = I->src[s];
               value.abs = false;
               value.neg = false;

               if (absolute) {
                  agx_index mask =
                     agx_mov_imm(&b, 32, UINT32_C(0x7fffffff));
                  agx_index result = agx_temp(ctx, AGX_SIZE_32);
                  agx_bitop_to(&b, result, value, mask, 0x8 /* AND */);
                  value = result;
               }

               if (negate) {
                  agx_index sign =
                     agx_mov_imm(&b, 32, UINT32_C(0x80000000));
                  agx_index result = agx_temp(ctx, AGX_SIZE_32);
                  agx_bitop_to(&b, result, value, sign, 0x6 /* XOR */);
                  value = result;
               }

               I->src[s] = value;
            }
         }
      }

      /* Chromium/Skia compares a negated f32 register for equality and then
       * selects two unrelated f32 registers. The only native FSEL negate bit
       * is grounded for a different half-mask endpoint, so do not infer that
       * modifier across result shapes. Materialize IEEE fneg exactly by
       * flipping the sign bit, using the already-grounded MOV-immediate and
       * BITOP XOR forms, before register allocation. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32 && !I->invert_cond && !I->saturate &&
          I->fcond == AGX_FCOND_EQ && I->src[0].size == AGX_SIZE_32 &&
          I->src[0].type != AGX_INDEX_IMMEDIATE && I->src[0].neg &&
          !I->src[0].abs && I->src[1].size == AGX_SIZE_32 &&
          I->src[1].type != AGX_INDEX_IMMEDIATE && !I->src[1].abs &&
          !I->src[1].neg && I->src[2].size == AGX_SIZE_32 &&
          I->src[2].type != AGX_INDEX_IMMEDIATE && !I->src[2].abs &&
          !I->src[2].neg && I->src[3].size == AGX_SIZE_32 &&
          I->src[3].type != AGX_INDEX_IMMEDIATE && !I->src[3].abs &&
          !I->src[3].neg) {
         agx_index source = I->src[0];
         source.neg = false;
         agx_index sign = agx_mov_imm(&b, 32, UINT32_C(0x80000000));
         agx_index flipped = agx_temp(ctx, AGX_SIZE_32);
         agx_bitop_to(&b, flipped, source, sign, 0x6 /* XOR */);
         I->src[0] = flipped;
      }

      /* A fused control-flow comparison has no native nonzero-minifloat
       * endpoint on G17P. Chromium's three.js skinning fragment keeps such a
       * threshold around a texture-sampling branch, so it cannot be flattened
       * into an FSEL. Materialize the exact IEEE value while temporaries still
       * exist; the resulting IF/WHILE/BREAK_IF_FCMP uses the already-grounded
       * register/register compare. Keep zero inline for the distinct compact
       * zero-comparison form. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          (I->op == AGX_OPCODE_IF_FCMP ||
           I->op == AGX_OPCODE_WHILE_FCMP ||
           I->op == AGX_OPCODE_BREAK_IF_FCMP) &&
          I->nr_srcs == 2) {
         const bool immediate_first =
            I->src[0].type == AGX_INDEX_IMMEDIATE;
         const unsigned immediate_src = immediate_first ? 0 : 1;
         const unsigned register_src = immediate_src ^ 1;

         if (I->src[immediate_src].type == AGX_INDEX_IMMEDIATE &&
             I->src[immediate_src].value != 0 &&
             I->src[register_src].type != AGX_INDEX_IMMEDIATE &&
             I->src[register_src].size == AGX_SIZE_32 &&
             !I->src[register_src].abs && !I->src[register_src].neg) {
            float value =
               agx_minifloat_decode(I->src[immediate_src].value);
            if (I->src[immediate_src].abs)
               value = fabsf(value);
            if (I->src[immediate_src].neg)
               value = -value;

            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            I->src[immediate_src] = agx_mov_imm(&b, 32, bits);
         }
      }

      /* Hyprland's colour-management shaders add a highp accumulator to a
       * mediump matrix term. There is no grounded plain mixed-width FADD on
       * G17P, so widen the half input explicitly and use the ordinary f32
       * register form. Keep the separately witnessed abs(f32) + f16 endpoint
       * available for its exact modifier shape. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FADD && I->nr_srcs == 2 && !I->saturate &&
          I->dest[0].size == AGX_SIZE_32) {
         for (unsigned s = 0; s < 2; ++s) {
            const unsigned other = s ^ 1;
            if (I->src[s].type == AGX_INDEX_IMMEDIATE ||
                I->src[s].size != AGX_SIZE_16 || I->src[s].abs ||
                I->src[s].neg || I->src[other].type == AGX_INDEX_IMMEDIATE ||
                I->src[other].size != AGX_SIZE_32 ||
                I->src[other].abs || I->src[other].neg)
               continue;

            agx_index widened = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, widened, I->src[s]);
            I->src[s] = widened;
         }
      }

      /* Native half FMA can carry a minifloat addend, but both multiplier
       * inputs are registers. Constant propagation commonly produces
       * clamp-like x * 0.5h + 0.5h; materialize only multiplier constants and
       * preserve the fused immediate-addend form. */
      if (I->op == AGX_OPCODE_HFMA && I->nr_srcs == 3) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type != AGX_INDEX_IMMEDIATE)
               continue;

            float value = agx_minifloat_decode(I->src[s].value);
            if (I->src[s].abs)
               value = fabsf(value);
            if (I->src[s].neg)
               value = -value;
            I->src[s] = agx_mov_imm(&b, 16, _mesa_float_to_half(value));
         }

         /* The immediate-addend form has no grounded multiplier-negate bit.
          * When either multiplicand is negated, load the addend as an exact
          * half so the general register FMA can carry the witnessed product
          * sign. This is refract's fused 1.0h - x*x operation. */
         if (I->src[2].type == AGX_INDEX_IMMEDIATE &&
             (I->src[0].neg || I->src[1].neg || I->src[2].abs ||
              I->src[2].neg)) {
            float value = agx_minifloat_decode(I->src[2].value);
            if (I->src[2].abs)
               value = fabsf(value);
            if (I->src[2].neg)
               value = -value;
            I->src[2] =
               agx_mov_imm(&b, 16, _mesa_float_to_half(value));
         }

         /* The grounded half FMA carries negate but not absolute modifiers.
          * Clear a register's sign bit explicitly, preserving a following
          * negate as -abs(x), before register allocation. */
         for (unsigned s = 0; s < 3; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE || !I->src[s].abs)
               continue;

            const bool negate = I->src[s].neg;
            agx_index source = I->src[s];
            source.abs = false;
            source.neg = false;
            agx_index mask = agx_mov_imm(&b, 16, UINT16_C(0x7fff));
            agx_index absolute = agx_temp(ctx, AGX_SIZE_16);
            agx_bitop_to(&b, absolute, source, mask, 0x8 /* AND */);
            absolute.neg = negate;
            I->src[s] = absolute;
         }
      }

      /* Only saturation through -0.0h and 1.0h - x have grounded immediate
       * half-add forms. Materialize every other minifloat operand so general
       * mediump offsets use the exact register-register add encoding. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_HADD && I->nr_srcs == 2) {
         const bool saturate_negzero =
            I->saturate && I->src[1].type == AGX_INDEX_IMMEDIATE &&
            I->src[1].value == UINT8_C(0x80);
         const bool one_minus =
            !I->saturate && I->src[0].type != AGX_INDEX_IMMEDIATE &&
            I->src[0].neg && I->src[1].type == AGX_INDEX_IMMEDIATE &&
            I->src[1].value == UINT8_C(0x30);

         if (!saturate_negzero && !one_minus) {
            for (unsigned s = 0; s < 2; ++s) {
               if (I->src[s].type != AGX_INDEX_IMMEDIATE)
                  continue;

               float value = agx_minifloat_decode(I->src[s].value);
               if (I->src[s].abs)
                  value = fabsf(value);
               if (I->src[s].neg)
                  value = -value;
               I->src[s] =
                  agx_mov_imm(&b, 16, _mesa_float_to_half(value));
            }
         }
      }

      /* Half add has no register absolute field. Materialize abs by clearing
       * the sign bit, retaining a following negate for the FMA subtraction
       * lowering below. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_HADD && I->nr_srcs == 2) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE || !I->src[s].abs)
               continue;

            const bool negate = I->src[s].neg;
            agx_index source = I->src[s];
            source.abs = false;
            source.neg = false;
            agx_index mask = agx_mov_imm(&b, 16, UINT16_C(0x7fff));
            agx_index absolute = agx_temp(ctx, AGX_SIZE_16);
            agx_bitop_to(&b, absolute, source, mask, 0x8 /* AND */);
            absolute.neg = negate;
            I->src[s] = absolute;
         }
      }

      /* The recovered register-register half add has no grounded negate
       * field. Preserve signed sums through the witnessed FMA form:
       * a - b = fma(a, 1.0h, -b), while -a - b uses both grounded FMA sign
       * bits. Do this after materializing constants so terrain's 0.5h - x
       * shape reaches the same register FMA path. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_HADD && I->nr_srcs == 2 &&
          (I->src[0].neg || I->src[1].neg) && !I->src[0].abs &&
          !I->src[1].abs && I->src[0].type != AGX_INDEX_IMMEDIATE &&
          I->src[1].type != AGX_INDEX_IMMEDIATE) {
         if (I->src[0].neg && !I->src[1].neg) {
            const agx_index tmp = I->src[0];
            I->src[0] = I->src[1];
            I->src[1] = tmp;
         }

         const bool product_neg = I->src[0].neg;
         I->src[0].neg = false;
         I->op = AGX_OPCODE_HFMA;
         I->src[2] = I->src[1];
         I->src[1] =
            agx_mov_imm(&b, 16, _mesa_float_to_half(1.0f));
         I->src[1].neg = product_neg;
         I->nr_srcs = 3;
      }

      /* G17P's grounded scalar-half min/max takes four half registers. NIR
       * represents max(x, k) as a tied compare/select, so load one exact half
       * constant and reuse it in both tied positions. In particular, this is
       * the mediump max(x, 0.0h) generated by glmark2's bump fragment. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 && !I->invert_cond &&
          !I->saturate &&
          (I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
          agx_is_equiv(I->src[0], I->src[2]) &&
          agx_is_equiv(I->src[1], I->src[3])) {
         const unsigned constant_pair =
            I->src[0].type == AGX_INDEX_IMMEDIATE ? 0 : 1;
         const unsigned register_pair = constant_pair ^ 1;

         if (I->src[constant_pair].type == AGX_INDEX_IMMEDIATE &&
             I->src[register_pair].type != AGX_INDEX_IMMEDIATE &&
             I->src[register_pair].size == AGX_SIZE_16 &&
             !I->src[constant_pair].abs && !I->src[constant_pair].neg &&
             !I->src[register_pair].abs && !I->src[register_pair].neg) {
            const float value =
               agx_minifloat_decode(I->src[constant_pair].value);
            const agx_index materialized =
               agx_mov_imm(&b, 16, _mesa_float_to_half(value));
            I->src[constant_pair] = materialized;
            I->src[constant_pair + 2] = materialized;
         }
      }

      /* The grounded mixed register/immediate FSEL can select an f32 register
       * or minifloat after comparing two f32 registers. Normalize a narrow
       * first comparator to full precision rather than guessing an unwitnessed
       * reversed-immediate or mixed-width encoding. The constant can already
       * be a compact half register or an inline minifloat; the latter is the
       * point-size clamp generated by glmark2's ideas vertex shader. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32 &&
          (I->src[0].type == AGX_INDEX_IMMEDIATE ||
           (I->src[0].size == AGX_SIZE_16 && !I->src[0].abs &&
            !I->src[0].neg)) &&
          I->src[1].type != AGX_INDEX_IMMEDIATE &&
          I->src[1].size == AGX_SIZE_32 &&
          !((I->fcond == AGX_FCOND_LTN || I->fcond == AGX_FCOND_GTN) &&
            !I->invert_cond && !I->saturate &&
            agx_is_equiv(I->src[0], I->src[2]) &&
            agx_is_equiv(I->src[1], I->src[3])) &&
          ((I->src[2].type == AGX_INDEX_IMMEDIATE &&
            I->src[3].type != AGX_INDEX_IMMEDIATE &&
            I->src[3].size == AGX_SIZE_32) ||
           (I->src[3].type == AGX_INDEX_IMMEDIATE &&
            I->src[2].type != AGX_INDEX_IMMEDIATE &&
            I->src[2].size == AGX_SIZE_32))) {
         if (I->src[0].type == AGX_INDEX_IMMEDIATE) {
            float value = agx_minifloat_decode(I->src[0].value);
            if (I->src[0].abs)
               value = fabsf(value);
            if (I->src[0].neg)
               value = -value;

            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            I->src[0] = agx_mov_imm(&b, 32, bits);
         } else {
            agx_index widened = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, widened, I->src[0]);
            I->src[0] = widened;
         }
      }

      /* Motion-blur loop bounds compare a highp value against a minifloat and
       * select a half-register boolean or a raw 16-bit immediate. Neither
       * mixed axis is grounded together, so materialize both constants and
       * use the general four-register half FSEL. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 && !I->saturate) {
         const bool compare_immediate_first =
            I->src[0].type == AGX_INDEX_IMMEDIATE;
         const unsigned compare_immediate = compare_immediate_first ? 0 : 1;
         const unsigned compare_register = compare_immediate ^ 1;
         const bool select_immediate_first =
            I->src[2].type == AGX_INDEX_IMMEDIATE;
         const unsigned select_immediate = select_immediate_first ? 2 : 3;
         const unsigned select_register = select_immediate == 2 ? 3 : 2;

         if (I->src[compare_immediate].type == AGX_INDEX_IMMEDIATE &&
             I->src[compare_register].type != AGX_INDEX_IMMEDIATE &&
             I->src[compare_register].size == AGX_SIZE_32 &&
             !I->src[compare_register].abs &&
             !I->src[compare_register].neg &&
             I->src[select_immediate].type == AGX_INDEX_IMMEDIATE &&
             I->src[select_immediate].value <= 1 &&
             !I->src[select_immediate].abs &&
             !I->src[select_immediate].neg &&
             I->src[select_register].type != AGX_INDEX_IMMEDIATE &&
             I->src[select_register].size == AGX_SIZE_16 &&
             !I->src[select_register].abs &&
             !I->src[select_register].neg) {
            float value =
               agx_minifloat_decode(I->src[compare_immediate].value);
            if (I->src[compare_immediate].abs)
               value = fabsf(value);
            if (I->src[compare_immediate].neg)
               value = -value;
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            I->src[compare_immediate] = agx_mov_imm(&b, 32, bits);
            I->src[select_immediate] =
               agx_mov_imm(&b, 16, I->src[select_immediate].value);
         }
      }

      /* A mediump zero test selecting either a highp value or zero can use
       * the grounded full-register/zero mixed-result FSEL once its comparator
       * is widened. Leave both zero immediates inline so existing compact
       * encodings, including stock vkcube, remain byte-identical. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32 && !I->saturate &&
          ((I->src[0].type == AGX_INDEX_IMMEDIATE &&
            I->src[0].value == 0 && I->src[1].type != AGX_INDEX_IMMEDIATE &&
            I->src[1].size == AGX_SIZE_16) ||
           (I->src[1].type == AGX_INDEX_IMMEDIATE &&
            I->src[1].value == 0 && I->src[0].type != AGX_INDEX_IMMEDIATE &&
            I->src[0].size == AGX_SIZE_16)) &&
          ((I->src[2].type == AGX_INDEX_IMMEDIATE &&
            I->src[2].value == 0 && I->src[3].type != AGX_INDEX_IMMEDIATE &&
            I->src[3].size == AGX_SIZE_32) ||
           (I->src[3].type == AGX_INDEX_IMMEDIATE &&
            I->src[3].value == 0 && I->src[2].type != AGX_INDEX_IMMEDIATE &&
            I->src[2].size == AGX_SIZE_32))) {
         const unsigned half_src =
            I->src[0].type == AGX_INDEX_IMMEDIATE ? 1 : 0;
         if (!I->src[half_src].abs && !I->src[half_src].neg) {
            agx_index widened = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, widened, I->src[half_src]);
            I->src[half_src] = widened;
         }
      }

      /* Hyprland colour management compares mediump values against compact
       * constants and compares highp values against abs(highp), selecting two
       * independent highp registers. Normalize those comparator-only shapes
       * to the general four-register f32 select. This avoids extending a
       * modifier or mixed-width field beyond the endpoints witnessed for it. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_32 && !I->saturate &&
          I->fcond == AGX_FCOND_GE &&
          ((I->src[0].type != AGX_INDEX_IMMEDIATE &&
            I->src[0].size == AGX_SIZE_16) ||
           (I->src[1].type != AGX_INDEX_IMMEDIATE &&
            I->src[1].size == AGX_SIZE_16) ||
           I->src[0].abs || I->src[1].abs) &&
          ((I->src[2].type == AGX_INDEX_IMMEDIATE && !I->src[2].abs &&
            !I->src[2].neg) ||
           (I->src[2].size == AGX_SIZE_32 && !I->src[2].abs &&
            !I->src[2].neg)) &&
          ((I->src[3].type == AGX_INDEX_IMMEDIATE && !I->src[3].abs &&
            !I->src[3].neg) ||
           (I->src[3].size == AGX_SIZE_32 && !I->src[3].abs &&
            !I->src[3].neg)) &&
          (I->src[2].type != AGX_INDEX_IMMEDIATE ||
           I->src[3].type != AGX_INDEX_IMMEDIATE)) {
         for (unsigned s = 0; s < 2; ++s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE) {
               float value = agx_minifloat_decode(I->src[s].value);
               if (I->src[s].abs)
                  value = fabsf(value);
               if (I->src[s].neg)
                  value = -value;

               uint32_t bits;
               memcpy(&bits, &value, sizeof(bits));
               I->src[s] = agx_mov_imm(&b, 32, bits);
            } else if (I->src[s].size == AGX_SIZE_16 &&
                       !I->src[s].abs && !I->src[s].neg) {
               agx_index widened = agx_temp(ctx, AGX_SIZE_32);
               agx_fmov_to(&b, widened, I->src[s]);
               I->src[s] = widened;
            } else if (I->src[s].size == AGX_SIZE_32 && I->src[s].abs &&
                       !I->src[s].neg) {
               agx_index source = I->src[s];
               source.abs = false;
               agx_index mask =
                  agx_mov_imm(&b, 32, UINT32_C(0x7fffffff));
               agx_index absolute = agx_temp(ctx, AGX_SIZE_32);
               agx_bitop_to(&b, absolute, source, mask, 0x8 /* AND */);
               I->src[s] = absolute;
            }
         }
      }

      /* Conditional discard lowers a standalone comparison to byte selection
       * after register allocation. Materialize a nonzero minifloat comparator
       * while a temporary can still be allocated, so Hyprland's alpha == 1.0
       * predicate reaches the grounded two-register comparison form. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMP && I->nr_srcs == 2 &&
          I->dest[0].size == AGX_SIZE_16 &&
          ctx->stage == MESA_SHADER_FRAGMENT &&
          ctx->nir->info.fs.uses_discard) {
         const bool immediate_first =
            I->src[0].type == AGX_INDEX_IMMEDIATE;
         const unsigned immediate_src = immediate_first ? 0 : 1;
         const unsigned register_src = immediate_src ^ 1;

         if (I->src[immediate_src].type == AGX_INDEX_IMMEDIATE &&
             I->src[immediate_src].value != 0 &&
             I->src[register_src].type != AGX_INDEX_IMMEDIATE &&
             I->src[register_src].size == AGX_SIZE_32 &&
             !I->src[register_src].abs && !I->src[register_src].neg) {
            float value =
               agx_minifloat_decode(I->src[immediate_src].value);
            if (I->src[immediate_src].abs)
               value = fabsf(value);
            if (I->src[immediate_src].neg)
               value = -value;

            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            I->src[immediate_src] = agx_mov_imm(&b, 32, bits);
         }
      }

      /* Standalone half comparisons lower after RA to a byte-constant FSEL.
       * Normalize their zero comparison here, while a temporary can still be
       * allocated, so late pseudo lowering reaches the same grounded form. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMP && I->nr_srcs == 2 &&
          I->dest[0].size == AGX_SIZE_16 &&
          (I->fcond == AGX_FCOND_EQ || I->fcond == AGX_FCOND_LT ||
           I->fcond == AGX_FCOND_GT || I->fcond == AGX_FCOND_GE ||
           I->fcond == AGX_FCOND_LE)) {
         const bool zero_first = I->src[0].type == AGX_INDEX_IMMEDIATE &&
                                 I->src[0].value == 0;
         const unsigned value_src = zero_first ? 1 : 0;
         const unsigned zero_src = zero_first ? 0 : 1;

         if (I->src[zero_src].type == AGX_INDEX_IMMEDIATE &&
             I->src[zero_src].value == 0 &&
             I->src[value_src].type != AGX_INDEX_IMMEDIATE &&
             I->src[value_src].size == AGX_SIZE_16 &&
             !I->src[value_src].abs && !I->src[value_src].neg) {
            agx_index widened = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, widened, I->src[value_src]);
            I->src[value_src] = widened;
         }
      }

      /* The general half compare/select takes four registers. When a half
       * comparison selects a register or a minifloat, materialize the latter;
       * terrain uses this for threshold ? value : 0.0h. Keep f32 comparisons
       * selecting byte constants on their separate grounded form. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 &&
          I->src[0].type != AGX_INDEX_IMMEDIATE &&
          I->src[0].size == AGX_SIZE_16 &&
          I->src[1].type != AGX_INDEX_IMMEDIATE &&
          I->src[1].size == AGX_SIZE_16 &&
          (I->src[2].type == AGX_INDEX_IMMEDIATE ||
           I->src[3].type == AGX_INDEX_IMMEDIATE) &&
          (I->src[2].type == AGX_INDEX_IMMEDIATE ||
           I->src[2].size == AGX_SIZE_16) &&
          (I->src[3].type == AGX_INDEX_IMMEDIATE ||
           I->src[3].size == AGX_SIZE_16)) {
         for (unsigned s = 2; s < 4; ++s) {
            if (I->src[s].type != AGX_INDEX_IMMEDIATE)
               continue;

            float value = agx_minifloat_decode(I->src[s].value);
            if (I->src[s].abs)
               value = fabsf(value);
            if (I->src[s].neg)
               value = -value;
            I->src[s] = agx_mov_imm(&b, 16, _mesa_float_to_half(value));
         }
      }

      /* A half comparison against a minifloat has grounded G17P FSEL forms
       * once the register comparator is full-width. Widen the half operand
       * explicitly for a half result selected from registers and/or
       * minifloats instead of treating it as a nonexistent mixed-width
       * immediate form. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 &&
          (I->src[2].type == AGX_INDEX_IMMEDIATE ||
           I->src[3].type == AGX_INDEX_IMMEDIATE) &&
          (I->src[2].type == AGX_INDEX_IMMEDIATE ||
           I->src[2].size == AGX_SIZE_16) &&
          (I->src[3].type == AGX_INDEX_IMMEDIATE ||
           I->src[3].size == AGX_SIZE_16)) {
         const bool immediate_first =
            I->src[0].type == AGX_INDEX_IMMEDIATE;
         const unsigned value_src = immediate_first ? 1 : 0;
         const unsigned immediate_src = immediate_first ? 0 : 1;

         if (I->src[immediate_src].type == AGX_INDEX_IMMEDIATE &&
             I->src[value_src].type != AGX_INDEX_IMMEDIATE &&
             I->src[value_src].size == AGX_SIZE_16 &&
             !I->src[value_src].abs && !I->src[value_src].neg) {
            agx_index widened = agx_temp(ctx, AGX_SIZE_32);
            agx_fmov_to(&b, widened, I->src[value_src]);
            I->src[value_src] = widened;
         }
      }

      /* A half comparison against a minifloat selecting two half registers
       * has no separately grounded immediate form. Materialize the compare
       * constant so it uses the general four-register half select whose
       * conditions and lifetime fields are already recovered. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          I->op == AGX_OPCODE_FCMPSEL && I->nr_srcs == 4 &&
          I->dest[0].size == AGX_SIZE_16 &&
          I->src[2].type != AGX_INDEX_IMMEDIATE &&
          I->src[2].size == AGX_SIZE_16 &&
          I->src[3].type != AGX_INDEX_IMMEDIATE &&
          I->src[3].size == AGX_SIZE_16) {
         for (unsigned s = 0; s < 2; ++s) {
            const unsigned other = s ^ 1;
            if (I->src[s].type != AGX_INDEX_IMMEDIATE ||
                I->src[other].type == AGX_INDEX_IMMEDIATE ||
                I->src[other].size != AGX_SIZE_16)
               continue;

            float value = agx_minifloat_decode(I->src[s].value);
            if (I->src[s].abs)
               value = fabsf(value);
            if (I->src[s].neg)
               value = -value;
            I->src[s] = agx_mov_imm(&b, 16, _mesa_float_to_half(value));
         }
      }

      /* Multiplication is commutative.  Keep a sole immediate in the second
       * multiply slot, matching Apple's native IMAD form and the existing
       * grounded G17P encoder.  Geometry/XFB arithmetic naturally arrives
       * from NIR with the constant first. */
      if (I->op == AGX_OPCODE_IMAD && I->nr_srcs == 3 &&
          I->src[0].type == AGX_INDEX_IMMEDIATE &&
          I->src[1].type != AGX_INDEX_IMMEDIATE) {
         agx_index tmp = I->src[0];
         I->src[0] = I->src[1];
         I->src[1] = tmp;
      }

      /* The grounded immediate-multiplier IMAD has either no addend or a
       * nonnegative byte addend. Split `a * byte - byte` into the witnessed
       * pure multiply followed by the witnessed subtract-immediate form. */
      if (I->op == AGX_OPCODE_IMAD && I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_32 && I->shift == 0 &&
          I->src[1].type == AGX_INDEX_IMMEDIATE && !I->src[1].abs &&
          !I->src[1].neg && I->src[1].value <= UINT8_MAX &&
          I->src[2].type == AGX_INDEX_IMMEDIATE && !I->src[2].abs &&
          I->src[2].neg && I->src[2].value <= UINT8_MAX) {
         agx_index product = agx_temp(ctx, AGX_SIZE_32);
         agx_imad_to(&b, product, I->src[0], I->src[1], agx_zero(), 0);
         I->op = AGX_OPCODE_IADD;
         I->src[0] = product;
         I->src[1] = I->src[2];
         I->nr_srcs = 2;
      }

      if ((I->op == AGX_OPCODE_BFI || I->op == AGX_OPCODE_BFEIL) &&
          I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_32 &&
          I->src[2].type == AGX_INDEX_IMMEDIATE) {
         I->src[2] = agx_mov_imm(&b, 32, I->src[2].value);
      }

      /* G17P's register-selected shift has no immediate value operand.  NIR
       * uses BFI(0, 1, shift) for 1u << shift, so materialize that value just
       * like the variable shift amount above. */
      if ((I->op == AGX_OPCODE_BFI || I->op == AGX_OPCODE_BFEIL) &&
          I->nr_dests == 1 && I->nr_srcs == 3 &&
          I->dest[0].size == AGX_SIZE_32 &&
          I->src[1].type == AGX_INDEX_IMMEDIATE) {
         I->src[1] = agx_mov_imm(&b, 32, I->src[1].value);
      } else if (I->op == AGX_OPCODE_BITOP && I->nr_dests == 1 &&
                 (I->dest[0].size == AGX_SIZE_32 ||
                  (I->dest[0].size == AGX_SIZE_16 &&
                   (I->truth_table == 0x8 || I->truth_table == 0xe ||
                    I->truth_table == 0x6)))) {
         agx_foreach_src(I, s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE)
               I->src[s] = agx_mov_imm(&b, I->dest[0].size == AGX_SIZE_16 ? 16 : 32,
                                      I->src[s].value);
         }
      }
   }
}

void
agx_materialize_agx3_cf_constants(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B)
      return;

   agx_foreach_instr_global_safe(ctx, I) {
      if ((I->op != AGX_OPCODE_IF_FCMP &&
           I->op != AGX_OPCODE_WHILE_FCMP &&
           I->op != AGX_OPCODE_BREAK_IF_FCMP) ||
          I->nr_srcs != 2)
         continue;

      /* A half operand compared against a float one reaches here once the
       * optimizer has propagated the widening into the comparison. Apple's
       * G17P compiler has no mixed-width control-flow compare: for
       * `if (f < h)` it widens h with an explicit conversion first, then
       * compares the f32 pair. Do the same.
       */
      for (unsigned s = 0; s < 2; ++s) {
         if (I->src[s].type != AGX_INDEX_NORMAL ||
             I->src[s].size != AGX_SIZE_16 ||
             I->src[s ^ 1].size != AGX_SIZE_32)
            continue;

         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_index wide = agx_temp(ctx, AGX_SIZE_32);
         agx_fmov_to(&b, wide, I->src[s]);
         I->src[s] = wide;
      }

      /* G17P's structured float-CF path is grounded for f32 register pairs
       * and for literal zero. Firefox's WebRender fragment compares a live
       * f32 against 1.0 (and later -10.0), which reaches AIR as a nonzero
       * minifloat. Materialize only that missing register/minifloat axis into
       * an exact f32 LDIMM before RA; this adds no speculative CF encoding and
       * leaves the shorter native zero form byte-identical. */
      for (unsigned s = 0; s < 2; ++s) {
         const unsigned other = s ^ 1;
         if (I->src[s].type != AGX_INDEX_IMMEDIATE ||
             I->src[s].value == 0 ||
             I->src[other].type == AGX_INDEX_IMMEDIATE ||
             I->src[other].size != AGX_SIZE_32 || I->src[other].memory ||
             I->src[other].abs || I->src[other].neg)
            continue;

         float value = agx_minifloat_decode(I->src[s].value);
         if (I->src[s].abs)
            value = fabsf(value);
         if (I->src[s].neg)
            value = -value;

         uint32_t bits;
         memcpy(&bits, &value, sizeof(bits));
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         I->src[s] = agx_mov_imm(&b, 32, bits);
      }
   }
}

/* Half-precision ALU instructions cannot encode 32-bit register sources.
 * Convert them explicitly. Immediate and null sources have no size bit. */
static agx_index
agx_coerce_f16_src(agx_builder *b, agx_index src)
{
   if (src.size == AGX_SIZE_16 || src.type == AGX_INDEX_IMMEDIATE ||
       src.type == AGX_INDEX_NULL)
      return src;

   agx_index tmp = agx_temp(b->shader, AGX_SIZE_16);
   agx_fmov_to(b, tmp, src);
   return tmp;
}

/* Ordinary G17P f32 arithmetic has no generally witnessed scalar-half
 * register source.  Widen such operands explicitly with the native
 * half-to-float add form before register allocation.  Immediate and null
 * sources have no size bit and remain directly encodable. */
static agx_index
agx_g17p_coerce_f32_src(agx_builder *b, agx_index src)
{
   if (b->shader->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       src.size == AGX_SIZE_32 || src.type == AGX_INDEX_IMMEDIATE ||
       src.type == AGX_INDEX_NULL)
      return src;

   assert(src.size == AGX_SIZE_16);
   agx_index tmp = agx_temp(b->shader, AGX_SIZE_32);
   agx_fmov_to(b, tmp, src);
   return tmp;
}

static agx_index
agx_g17p_shift_boundary(agx_builder *b, unsigned shift)
{
   assert(shift >= 1 && shift <= 31);
   agx_index positive = agx_mov_imm(b, 32, shift);
   agx_index boundary = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, boundary, agx_zero(), positive, 0);
   return boundary;
}

static agx_index
agx_g17p_ushr_imm(agx_builder *b, agx_index value, unsigned shift)
{
   return agx_ushr(b, value, agx_g17p_shift_boundary(b, shift));
}

static agx_index
agx_g17p_and_imm(agx_builder *b, agx_index value, uint32_t mask)
{
   return agx_and(b, value, agx_mov_imm(b, 32, mask));
}

static agx_index
agx_g17p_ishl_imm(agx_builder *b, agx_index value, unsigned shift)
{
   agx_index shifted = agx_temp(b->shader, AGX_SIZE_32);
   agx_bfi_to(b, shifted, agx_immediate(0), value,
              agx_immediate(shift), 0);
   return shifted;
}

static agx_instr *
agx_emit_g17p_bitrev(agx_builder *b, agx_index dst, agx_index value)
{
   /* AGX3's BITREV opcode is not grounded. Express GLSL
    * bitfieldReverse() with the ordinary G17P shifts and bitwise operations
    * instead. Every shift below is a compile-time constant and every mask is
    * exact, so this is the conventional five-stage 32-bit permutation rather
    * than an approximation of an unknown native instruction. */
   static const struct {
      uint8_t shift;
      uint32_t mask;
   } stages[] = {
      {1, UINT32_C(0x55555555)},
      {2, UINT32_C(0x33333333)},
      {4, UINT32_C(0x0f0f0f0f)},
      {8, UINT32_C(0x00ff00ff)},
   };

   agx_index x = value;
   for (unsigned i = 0; i < ARRAY_SIZE(stages); ++i) {
      agx_index lo = agx_g17p_and_imm(b, x, stages[i].mask);
      lo = agx_g17p_ishl_imm(b, lo, stages[i].shift);
      agx_index hi = agx_g17p_and_imm(
         b, agx_g17p_ushr_imm(b, x, stages[i].shift), stages[i].mask);
      x = agx_or(b, lo, hi);
   }

   return agx_or_to(b, dst, agx_g17p_ushr_imm(b, x, 16),
                    agx_g17p_ishl_imm(b, x, 16));
}

static agx_instr *
agx_emit_g17p_ufind_msb(agx_builder *b, agx_index dst, agx_index value)
{
   /* FT1 on G17P exposes a non-architectural logarithmic intermediate. Find
    * the top bit with the integer forms already qualified on hardware:
    * smear the top bit downward, popcount the resulting prefix with SWAR,
    * then subtract one. Zero smears/counts to zero and therefore returns -1
    * without a compare or select. */
   assert(value.size == AGX_SIZE_32 && dst.size == AGX_SIZE_32);

   agx_index scan = value;
   static const unsigned smear_shifts[] = {1, 2, 4, 8, 16};
   for (unsigned i = 0; i < ARRAY_SIZE(smear_shifts); ++i)
      scan = agx_or(b, scan, agx_g17p_ushr_imm(b, scan, smear_shifts[i]));

   agx_index odd = agx_g17p_and_imm(
      b, agx_g17p_ushr_imm(b, scan, 1), UINT32_C(0x55555555));
   scan = agx_iadd(b, scan, agx_neg(odd), 0);

   agx_index pairs_lo = agx_g17p_and_imm(b, scan, UINT32_C(0x33333333));
   agx_index pairs_hi = agx_g17p_and_imm(
      b, agx_g17p_ushr_imm(b, scan, 2), UINT32_C(0x33333333));
   scan = agx_iadd(b, pairs_lo, pairs_hi, 0);

   scan = agx_g17p_and_imm(
      b, agx_iadd(b, scan, agx_g17p_ushr_imm(b, scan, 4), 0),
      UINT32_C(0x0f0f0f0f));
   scan = agx_iadd(b, scan, agx_g17p_ushr_imm(b, scan, 8), 0);
   scan = agx_iadd(b, scan, agx_g17p_ushr_imm(b, scan, 16), 0);

   agx_index count = agx_g17p_and_imm(b, scan, 0x3f);
   return agx_iadd_to(b, dst, count, agx_neg(agx_immediate(1)), 0);
}

static agx_instr *
agx_emit_alu(agx_builder *b, nir_alu_instr *instr)
{
   unsigned srcs = nir_op_infos[instr->op].num_inputs;
   unsigned sz = instr->def.bit_size;
   unsigned src_sz = srcs ? nir_src_bit_size(instr->src[0].src) : 0;
   ASSERTED unsigned comps = instr->def.num_components;

   assert(comps == 1 || nir_op_is_vec_or_mov(instr->op));
   assert(sz == 1 ||
          ((nir_op_is_vec_or_mov(instr->op) ||
            is_conversion_to_8bit(instr->op) || instr->op == nir_op_bcsel) &&
           sz == 8) ||
          sz == 16 || sz == 32 || sz == 64);

   agx_index dst = agx_def_index(&instr->def);
   agx_index s0 = srcs > 0 ? agx_alu_src_index(b, instr->src[0]) : agx_null();
   agx_index s1 = srcs > 1 ? agx_alu_src_index(b, instr->src[1]) : agx_null();
   agx_index s2 = srcs > 2 ? agx_alu_src_index(b, instr->src[2]) : agx_null();
   agx_index s3 = srcs > 3 ? agx_alu_src_index(b, instr->src[3]) : agx_null();

   agx_index i0 = agx_immediate(0);
   agx_index i1 = agx_immediate(1);

#define UNOP(nop, aop)                                                         \
   case nir_op_##nop:                                                          \
      return agx_##aop##_to(b, dst, s0);
#define BINOP(nop, aop)                                                        \
   case nir_op_##nop:                                                          \
      return agx_##aop##_to(b, dst, s0, s1);

   switch (instr->op) {
      UNOP(f2f16, fmov);
      UNOP(f2f16_rtne, fmov);
      UNOP(f2f32, fmov);
      UNOP(fround_even, roundeven);
      UNOP(ftrunc, trunc);
      UNOP(ffloor, floor);
      UNOP(fceil, ceil);
      UNOP(frcp, rcp);
      UNOP(frsq, rsqrt);
      UNOP(flog2, log2);
      UNOP(fsinc_agx3, sinc_agx3);
      UNOP(fexp2, exp2);

      UNOP(mov, mov);
      UNOP(u2u32, mov);
   case nir_op_bitfield_reverse:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          src_sz == 32)
         return agx_emit_g17p_bitrev(b, dst, s0);
      return agx_bitrev_to(b, dst, s0);
   case nir_op_bit_count:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          src_sz == 16 && sz == 32) {
         /* Apple's native ushort popcount writes a ushort.  Count(16 bits)
          * is at most 16, so widen its exact result after the operation. */
         agx_index count16 = agx_temp(b->shader, AGX_SIZE_16);
         agx_popcount_to(b, count16, s0);
         return agx_mov_to(b, dst, count16);
      }
      return agx_popcount_to(b, dst, s0);
   case nir_op_ufind_msb:
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          src_sz == 32)
         return agx_emit_g17p_ufind_msb(b, dst, s0);
      return agx_ffs_to(b, dst, s0);
      BINOP(iand, and);
      BINOP(ior, or);
      BINOP(ixor, xor);
      BINOP(interleave, intl);

   case nir_op_fadd:
      if (instr->def.bit_size == 16)
         return agx_hadd_to(b, dst, agx_coerce_f16_src(b, s0),
                            agx_coerce_f16_src(b, s1));
      else
         return agx_fadd_to(b, dst, agx_g17p_coerce_f32_src(b, s0),
                            agx_g17p_coerce_f32_src(b, s1));

   case nir_op_fmul:
      if (instr->def.bit_size == 16)
         return agx_hmul_to(b, dst, agx_coerce_f16_src(b, s0),
                            agx_coerce_f16_src(b, s1));
      else
         return agx_fmul_to(b, dst, agx_g17p_coerce_f32_src(b, s0),
                            agx_g17p_coerce_f32_src(b, s1));

   case nir_op_ffma:
      if (instr->def.bit_size == 16)
         return agx_hfma_to(b, dst, agx_coerce_f16_src(b, s0),
                            agx_coerce_f16_src(b, s1),
                            agx_coerce_f16_src(b, s2));
      else
         return agx_ffma_to(b, dst, agx_g17p_coerce_f32_src(b, s0),
                            agx_g17p_coerce_f32_src(b, s1),
                            agx_g17p_coerce_f32_src(b, s2));

   case nir_op_fsat: {
      agx_instr *I = agx_fadd_to(b, dst, s0, agx_negzero());
      if (instr->def.bit_size == 16)
         I->op = AGX_OPCODE_HADD;

      I->saturate = true;
      return I;
   }

   case nir_op_feq:
      return agx_fcmp_to(b, dst, s0, s1, AGX_FCOND_EQ, false);
   case nir_op_flt:
      return agx_fcmp_to(b, dst, s0, s1, AGX_FCOND_LT, false);
   case nir_op_fge:
      return agx_fcmp_to(b, dst, s0, s1, AGX_FCOND_GE, false);
   case nir_op_fneu:
      return agx_fcmp_to(b, dst, s0, s1, AGX_FCOND_EQ, true);

   case nir_op_ieq:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_UEQ, false);
   case nir_op_ine:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_UEQ, true);
   case nir_op_ilt:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_SLT, false);
   case nir_op_ige:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_SLT, true);
   case nir_op_ult:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_ULT, false);
   case nir_op_uge:
      return agx_icmp_to(b, dst, s0, s1, AGX_ICOND_ULT, true);

   case nir_op_inot:
      if (sz == 1)
         return agx_xor_to(b, dst, s0, i1);
      else
         return agx_not_to(b, dst, s0);

   case nir_op_b2b1:
      return agx_icmp_to(b, dst, s0, i0, AGX_ICOND_UEQ, true);

   case nir_op_fsqrt:
      if (instr->def.bit_size == 16) {
         /* G17P computes a scalar half sqrt as a half-input sqrt step into a
          * float32 temporary followed by a mixed multiply rounded once to
          * the half destination.  Keeping the temporary wide matches Apple
          * and avoids treating the neighbouring half as input state. */
         agx_index step = agx_temp(b->shader, AGX_SIZE_32);
         agx_srsqrt_to(b, step, s0);
         return agx_fmul_to(b, dst, step, s0);
      }
      return agx_fmul_to(b, dst, s0, agx_srsqrt(b, s0));
   case nir_op_fabs:
      return agx_fmov_to(b, dst, agx_abs(s0));
   case nir_op_fneg:
      return agx_fmov_to(b, dst, agx_neg(s0));

   case nir_op_fmin:
   case nir_op_fmax:
   case nir_op_fmin_agx:
   case nir_op_fmax_agx:
      return agx_fminmax_to(b, dst, s0, s1, instr);

   case nir_op_iabs: {
      assert((sz == 16 || sz == 32) && "8-bit absolute values lowered");
      /* Keep G17P's absolute value independent of signed isel.  The usual
       * two's-complement identity is exact even for the most negative value,
       * and a 16-bit value uses the same arithmetic shift, XOR and subtract
       * at half width. */
      agx_index sign = agx_asr(b, s0, agx_immediate(sz - 1));
      agx_index magnitude = agx_xor(b, s0, sign);
      return agx_iadd_to(b, dst, magnitude, agx_neg(sign), 0);
   }

   case nir_op_imin:
      return agx_icmpsel_to(b, dst, s0, s1, s0, s1, AGX_ICOND_SLT);
   case nir_op_imax:
      return agx_icmpsel_to(b, dst, s0, s1, s0, s1, AGX_ICOND_SGT);
   case nir_op_umin:
      return agx_icmpsel_to(b, dst, s0, s1, s0, s1, AGX_ICOND_ULT);
   case nir_op_umax:
      return agx_icmpsel_to(b, dst, s0, s1, s0, s1, AGX_ICOND_UGT);
   case nir_op_bounds_agx:
      /* end offset > bound ? 0 : data */
      return agx_icmpsel_to(b, dst, s1, s2, agx_zero(), s0, AGX_ICOND_UGT);

   case nir_op_iadd:
      return agx_iadd_to(b, dst, s0, s1, 0);
   case nir_op_imadshl_agx:
      return agx_emit_imadshl_agx(b, instr, dst, s0, s1, s2, s3);
   case nir_op_imsubshl_agx:
      return agx_emit_imadshl_agx(b, instr, dst, s0, s1, agx_neg(s2), s3);
   case nir_op_isub:
      /* The native G17P subtract form takes two registers. Do not route an
       * immediate minuend through iadd's narrow-source sign-extension bit:
       * that bit is not arithmetic negation and would turn `c - x` into
       * `c + x`.
       */
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          sz == 32 && s0.type == AGX_INDEX_IMMEDIATE)
         s0 = agx_mov_imm(b, 32, s0.value);
      return agx_iadd_to(b, dst, s0, agx_neg(s1), 0);
   case nir_op_ineg:
      /* G17P's iadd operand modifier is a narrow-source sign extension, not
       * an arithmetic negate. Materialize the zero minuend so post-RA
       * instruction selection reaches the native register-register isub
       * form instead of silently encoding `0 + x`.
       */
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
          sz == 32) {
         agx_index zero = agx_mov_imm(b, 32, 0);
         return agx_iadd_to(b, dst, zero, agx_neg(s0), 0);
      }
      return agx_iadd_to(b, dst, i0, agx_neg(s0), 0);
   case nir_op_imul:
      return agx_imad_to(b, dst, s0, s1, i0, 0);
   case nir_op_umul_2x32_64:
      return agx_imad_to(b, dst, agx_abs(s0), agx_abs(s1), i0, 0);
   case nir_op_imul_2x32_64:
      return agx_imad_to(b, dst, s0, s1, i0, 0);
   case nir_op_umul_high:
      return agx_mul_high_to(b, dst, s0, s1, false);
   case nir_op_imul_high:
      return agx_mul_high_to(b, dst, s0, s1, true);

   case nir_op_ishl:
      return agx_bfi_to(b, dst, i0, s0, s1, 0);
   case nir_op_ushr: {
      /* G17P's register `srl` takes the ordinary positive count, as
       * applegpu_g17p's `ushort(a >> s)` (a710540002080000d0130500, after
       * its own `s & 31`) does. This used to build `0 - count`, but the
       * G17P iadd packer read NEG as a sign extension and emitted `0 +
       * count`; every hardware-proven program therefore already carries
       * the positive count through this add, which is kept. */
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         if (s1.type == AGX_INDEX_IMMEDIATE) {
            s1 = agx_g17p_shift_boundary(b, s1.value);
         } else {
            agx_index boundary = agx_temp(b->shader, AGX_SIZE_32);
            agx_iadd_to(b, boundary, i0, s1, 0);
            s1 = boundary;
         }
      }
      return agx_ushr_to(b, dst, s0, s1);
   }
   case nir_op_ishr:
      return agx_asr_to(b, dst, s0, s1);

   case nir_op_extr_agx:
      return agx_extr_to(b, dst, s0, s1, s2,
                         nir_alu_src_as_uint(instr->src[3]));

   case nir_op_ubitfield_extract: {
      unsigned m = nir_alu_src_as_uint(instr->src[2]);
      assert(m != 0 && "should've been optimized");

      /* Disable masking if the whole thing is used */
      if (m >= 32)
         m = 0;

      return agx_bfeil_to(b, dst, i0, s0, s1, m);
   }

   case nir_op_bcsel:
      return agx_icmpsel_to(b, dst, s0, i0, s2, s1, AGX_ICOND_UEQ);

   case nir_op_i2i32: {
      if (src_sz == 8) {
         if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
            /* G17P's grounded shifts operate on complete 32-bit registers.
             * Widen first and sign-extend from bit 7 as Apple does, rather
             * than inventing a half-register BFI/ASR encoding. */
            agx_index wide = agx_pad_to_32(b, s0);
            agx_index zero32 = agx_mov_imm(b, 32, 0);
            agx_index shift = agx_mov_imm(b, 32, 24);
            agx_index shifted =
               agx_bfi(b, zero32, wide, shift, 0);
            return agx_asr_to(b, dst, shifted, shift);
         }

         /* Sign extend in software, NIR likes 8-bit conversions */
         agx_index ishl16 = agx_bfi(b, i0, s0, agx_immediate(8), 0);
         return agx_asr_to(b, dst, ishl16, agx_immediate(8));
      } else {
         assert(s0.size == AGX_SIZE_16 && "other conversions lowered");
         return agx_signext_to(b, dst, s0);
      }
   }

   case nir_op_i2i16: {
      if (src_sz == 8) {
         if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
            /* Perform the same exact 32-bit sign extension and expose its
             * low half only after the shift pair has completed. */
            agx_index wide = agx_pad_to_32(b, s0);
            agx_index zero32 = agx_mov_imm(b, 32, 0);
            agx_index shift = agx_mov_imm(b, 32, 24);
            agx_index shifted =
               agx_bfi(b, zero32, wide, shift, 0);
            agx_index extended = agx_asr(b, shifted, shift);
            return agx_subdivide_to(b, dst, extended, 0);
         }

         /* Sign extend in software, NIR likes 8-bit conversions */
         agx_index ishl16 = agx_bfi(b, i0, s0, agx_immediate(8), 0);
         return agx_asr_to(b, dst, ishl16, agx_immediate(8));
      } else {
         assert(s0.size == AGX_SIZE_32 && "other conversions lowered");
         return agx_subdivide_to(b, dst, s0, 0);
      }
   }

   case nir_op_u2u16: {
      if (s0.size == AGX_SIZE_32)
         return agx_subdivide_to(b, dst, s0, 0);
      else
         return agx_mov_to(b, dst, s0);
   }

   /* It will be put into a 16-bit register, but zero out the garbage. We could
    * optimize this in the future but it ensures correctness for u2u16(u2u8(x))
    * sequences.
    */
   case nir_op_u2u8:
   case nir_op_i2i8:
      return agx_and_to(b, dst, s0, agx_immediate(0xFF));

   case nir_op_iadd_sat: {
      agx_instr *I = agx_iadd_to(b, dst, s0, s1, 0);
      I->saturate = true;
      return I;
   }

   case nir_op_isub_sat: {
      agx_instr *I = agx_iadd_to(b, dst, s0, agx_neg(s1), 0);
      I->saturate = true;
      return I;
   }

   case nir_op_uadd_sat: {
      agx_instr *I = agx_iadd_to(b, dst, agx_abs(s0), agx_abs(s1), 0);
      I->saturate = true;
      return I;
   }

   case nir_op_usub_sat: {
      agx_instr *I = agx_iadd_to(b, dst, agx_abs(s0), agx_neg(agx_abs(s1)), 0);
      I->saturate = true;
      return I;
   }

   case nir_op_fsin_agx: {
      if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B) {
         /* Compatibility for callers which already performed the AGX2 NIR
          * quadrant reduction. Fold [0,4] exactly to [-1,1], then use the
          * same native sinc operation as the G17P radians lowering. The
          * folds stay in the operand's own precision: the builder would size
          * the temporary from the leading immediate and make a 16-bit add of
          * a 32-bit quadrant, which neither keeps the precision nor packs. */
         agx_index middle = agx_temp(b->shader, s0.size);
         agx_fadd_to(b, middle, agx_immediate_f(2.0f), agx_neg(s0));
         agx_index last = agx_fadd(b, s0, agx_immediate_f(-4.0f));
         agx_index t = agx_fcmpsel(b, s0, agx_immediate_f(1.0f), s0,
                                   middle, AGX_FCOND_LE);
         t = agx_fcmpsel(b, s0, agx_immediate_f(3.0f), last, t,
                         AGX_FCOND_GT);
         return agx_fmul_to(b, dst, agx_sinc_agx3(b, t), t);
      }

      agx_index fixup = agx_sin_pt_1(b, s0);
      agx_index sinc = agx_sin_pt_2(b, fixup);
      return agx_fmul_to(b, dst, sinc, fixup);
   }

   case nir_op_bittest_select_agx3: {
      assert(nir_src_is_const(instr->src[1].src));
      const unsigned mask = nir_alu_src_as_uint(instr->src[1]);
      assert(mask == 1 || mask == 2 || mask == 4);
      return agx_bittest_select_agx3_to(b, dst, s0, s2, s3, mask);
   }

   case nir_op_f2i16:
      return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_F_TO_S16), s0,
                            AGX_ROUND_RTZ);

   case nir_op_f2i32:
      return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_F_TO_S32), s0,
                            AGX_ROUND_RTZ);

   case nir_op_f2u16:
      return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_F_TO_U16), s0,
                            AGX_ROUND_RTZ);

   case nir_op_f2u32:
      return agx_convert_to(b, dst, agx_immediate(AGX_CONVERT_F_TO_U32), s0,
                            AGX_ROUND_RTZ);

   case nir_op_u2f16:
   case nir_op_u2f32: {
      if (src_sz == 64)
         UNREACHABLE("64-bit conversions unimplemented");

      enum agx_convert mode = (src_sz == 32)   ? AGX_CONVERT_U32_TO_F
                              : (src_sz == 16) ? AGX_CONVERT_U16_TO_F
                                               : AGX_CONVERT_U8_TO_F;

      return agx_convert_to(b, dst, agx_immediate(mode), s0, AGX_ROUND_RTE);
   }

   case nir_op_i2f16:
   case nir_op_i2f32: {
      if (src_sz == 64)
         UNREACHABLE("64-bit conversions unimplemented");

      enum agx_convert mode = (src_sz == 32)   ? AGX_CONVERT_S32_TO_F
                              : (src_sz == 16) ? AGX_CONVERT_S16_TO_F
                                               : AGX_CONVERT_S8_TO_F;

      return agx_convert_to(b, dst, agx_immediate(mode), s0, AGX_ROUND_RTE);
   }

   case nir_op_pack_32_2x16_split:
   case nir_op_pack_64_2x32_split: {
      agx_index idx[] = {s0, s1};
      return agx_emit_collect_to(b, dst, 2, idx);
   }

   case nir_op_unpack_64_2x32_split_x:
   case nir_op_unpack_32_2x16_split_x:
      return agx_subdivide_to(b, dst, s0, 0);

   case nir_op_unpack_64_2x32_split_y:
   case nir_op_unpack_32_2x16_split_y:
      return agx_subdivide_to(b, dst, s0, 1);

   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4: {
      agx_index idx[] = {s0, s1, s2, s3};
      return agx_emit_collect_to(b, dst, srcs, idx);
   }

   case nir_op_vec8:
   case nir_op_vec16:
      UNREACHABLE("should've been lowered");

   default:
      fprintf(stderr, "Unhandled ALU op %s\n", nir_op_infos[instr->op].name);
      UNREACHABLE("Unhandled ALU instruction");
   }
}

static enum agx_lod_mode
agx_lod_mode_for_nir(nir_texop op, bool biased, bool min_lod, bool lod_is_zero)
{
   switch (op) {
   case nir_texop_tex:
   case nir_texop_tg4:
      /* We could support this for tex, but it's never actually seen because tex
       * is always turned into txb to implement sampler LOD bias in Vulkan.
       */
      assert(!min_lod && "unimplemented");

      return AGX_LOD_MODE_AUTO_LOD;
   case nir_texop_txb:
      return min_lod ? AGX_LOD_MODE_AUTO_LOD_BIAS_MIN
                     : AGX_LOD_MODE_AUTO_LOD_BIAS;
   case nir_texop_lod:
      assert(!min_lod);
      return biased ? AGX_LOD_MODE_AUTO_LOD_BIAS : AGX_LOD_MODE_AUTO_LOD;
   case nir_texop_txd:
      return min_lod ? AGX_LOD_MODE_LOD_GRAD_MIN : AGX_LOD_MODE_LOD_GRAD;
   case nir_texop_txl:
   case nir_texop_txf:
      assert(!min_lod);
      return lod_is_zero ? AGX_LOD_MODE_AUTO_LOD : AGX_LOD_MODE_LOD_MIN;
   case nir_texop_txf_ms:
      assert(!min_lod);
      assert(lod_is_zero && "no mipmapping");
      return AGX_LOD_MODE_AUTO_LOD;
   default:
      UNREACHABLE("Unhandled texture op");
   }
}

static enum agx_gather
agx_gather_for_nir(nir_tex_instr *tex)
{
   if (tex->op == nir_texop_tg4) {
      enum agx_gather components[] = {
         AGX_GATHER_R,
         AGX_GATHER_G,
         AGX_GATHER_B,
         AGX_GATHER_A,
      };

      assert(tex->component < ARRAY_SIZE(components));
      return components[tex->component];
   } else {
      return AGX_GATHER_NONE;
   }
}

static void
agx_emit_tex(agx_builder *b, nir_tex_instr *instr)
{
   agx_index coords = agx_null(), bindless = agx_immediate(0),
             texture = agx_immediate(instr->texture_index),
             sampler = agx_immediate(instr->sampler_index),
             lod = agx_immediate(0), compare = agx_null(),
             packed_offset = agx_null();

   /* Default to the txf sampler at ss0 */
   if (!nir_tex_instr_need_sampler(instr)) {
      sampler = agx_immediate(0);
   }

   bool lod_is_zero = true;

   for (unsigned i = 0; i < instr->num_srcs; ++i) {
      agx_index index = agx_src_index(&instr->src[i].src);

      switch (instr->src[i].src_type) {
      case nir_tex_src_backend1:
         coords = index;
         break;

      case nir_tex_src_backend2:
         /* G17P's texture offset is an immediate in SMP, with signed
          * x/y nibbles and a third z nibble for 3D. Keep the constant from
          * agx_nir_lower_texture in that form instead of materializing a GPR
          * that the native instruction does not read.
          */
         if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             nir_src_is_const(instr->src[i].src) &&
             (instr->op == nir_texop_tg4 ||
              nir_src_as_uint(instr->src[i].src) <= 0xfff)) {
            uint32_t value = nir_src_as_uint(instr->src[i].src);
            if (instr->op == nir_texop_tg4)
               value = (value & 0xf) | ((value >> 12) & 0xf0);
            /* Apple's compiler emits an all-zero offset as a plain sample. */
            packed_offset = value ? agx_immediate(value) : agx_null();
         } else {
            packed_offset = index;
         }
         if (packed_offset.type == AGX_INDEX_IMMEDIATE)
            packed_offset.size = AGX_SIZE_32;
         break;

      case nir_tex_src_lod:
      case nir_tex_src_bias:
         /* G17P has an exact immediate texel-fetch level encoding. Keep
          * compile-time levels in that form instead of promoting them into
          * the per-lane PACK_SMPIR path. Besides being smaller, this matches
          * Apple's constant read(..., level) programs and avoids introducing
          * an asynchronous staging dependency for a literal. */
         if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
             instr->op == nir_texop_txf &&
             nir_src_is_const(instr->src[i].src) &&
             nir_src_as_uint(instr->src[i].src) <= 31)
            lod = agx_immediate(nir_src_as_uint(instr->src[i].src));
         else
            lod = index;
         lod_is_zero = nir_src_is_const(instr->src[i].src) &&
                       nir_src_as_uint(instr->src[i].src) == 0;
         break;

      case nir_tex_src_lod_bias_min_agx:
         assert(index.size == AGX_SIZE_32);
         lod = index;
         break;

      case nir_tex_src_comparator:
         assert(index.size == AGX_SIZE_32);
         compare = index;
         break;

      case nir_tex_src_texture_offset:
         texture = index;
         break;
      case nir_tex_src_sampler_handle:
         nir_intrinsic_instr *intr = nir_src_as_intrinsic(instr->src[i].src);
         if (intr && intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
            b->shader->out->uses_sampler_heap = true;
            /* Heap offset */
            sampler = agx_src_index(&intr->src[1]);
         } else {
            sampler = index;
         }
         break;

      case nir_tex_src_texture_handle:
         texture =
            agx_translate_bindless_handle(b, &instr->src[i].src, &bindless);
         break;

      case nir_tex_src_min_lod:
         assert(instr->op == nir_texop_txd && "other cases lowered");
         break;

      case nir_tex_src_ddx: {
         int y_idx = nir_tex_instr_src_index(instr, nir_tex_src_ddy);
         assert(y_idx >= 0 && "we only handle gradients");

         int min_idx = nir_tex_instr_src_index(instr, nir_tex_src_min_lod);
         bool has_min = min_idx >= 0;
         agx_index min;

         unsigned n = nir_tex_instr_src_size(instr, y_idx);
         assert((n == 2 || n == 3) && "other sizes not supported");

         agx_index index2 = agx_src_index(&instr->src[y_idx].src);

         if (has_min) {
            min = agx_src_index(&instr->src[min_idx].src);

            /* Undef extend to 32-bit since our IR is iffy */
            min = agx_pad_to_32(b, min);
         }

         /* We explicitly don't cache about the split cache for this */
         unsigned chans = (2 * n) + (has_min ? 1 : 0);
         lod = agx_vec_temp(b->shader, AGX_SIZE_32, chans);
         agx_instr *I = agx_collect_to(b, lod, chans);

         for (unsigned i = 0; i < n; ++i) {
            I->src[(2 * i) + 0] = agx_emit_extract(b, index, i);
            I->src[(2 * i) + 1] = agx_emit_extract(b, index2, i);
         }

         if (has_min)
            I->src[2 * n] = min;

         break;
      }

      case nir_tex_src_ddy:
         /* handled above */
         break;

      default:
         UNREACHABLE("Unexpected texture source");
      }
   }

   bool has_min_lod =
      nir_tex_instr_src_index(instr, nir_tex_src_lod_bias_min_agx) >= 0 ||
      nir_tex_instr_src_index(instr, nir_tex_src_min_lod) >= 0;

   bool has_bias = (has_min_lod && instr->op != nir_texop_txd) ||
                   (nir_tex_instr_src_index(instr, nir_tex_src_bias) >= 0);

   enum agx_lod_mode lod_mode =
      agx_lod_mode_for_nir(instr->op, has_bias, has_min_lod, lod_is_zero);

   /* AGX2's AUTO_LOD encoding also covers txl(..., 0). G17P's AUTO form is
    * specifically the fragment derivative form, while Apple's compute and
    * vertex witnesses use the explicit immediate-level encoding. Keep every
    * non-fragment txl at level zero explicit so the packer selects that
    * native form. */
   if (b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       b->shader->stage != MESA_SHADER_FRAGMENT &&
       (instr->op == nir_texop_txl || instr->op == nir_texop_txf) &&
       lod_is_zero) {
      lod_mode = AGX_LOD_MODE_LOD_MIN;
      /* The native G17P vertex form carries level zero in the instruction.
       * Keeping NIR's fp32 zero SSA here made texture_sample source 1 two
       * half-registers wide even though the explicit-level IR operand is one,
       * so validation failed before the packer could select that form. */
      lod = agx_immediate(0);
   }

   if (lod_mode == AGX_LOD_MODE_AUTO_LOD) {
      /* Ignored logically but asserted 0 */
      lod = agx_immediate(0);
   }

   agx_index dst = agx_def_index(&instr->def);

   /* Pack the shadow reference and offset sources into the one texture IR
    * operand. G17P constants leave only the comparator here because their
    * offset rides in SMP. Runtime G17P offsets remain a two-component vector
    * for PACK_SMPIR; a comparison, when present, is its leading component. */
   agx_index compare_offset = agx_null();

   /* G17P constant offsets live in the instruction, leaving src[5] to
    * the comparator. Runtime gather offsets retain the combined vector;
    * the packer publishes each value into its own native operand bank. */
   const bool offset_present = !agx_is_null(packed_offset);
   const uint16_t offset_xyz =
      packed_offset.type == AGX_INDEX_IMMEDIATE
         ? (uint16_t)packed_offset.value : 0;

   const bool g17p_split_offset =
      b->shader->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
      packed_offset.type == AGX_INDEX_IMMEDIATE;

   if (g17p_split_offset && !agx_is_null(compare))
      packed_offset = agx_null();

   if (!agx_is_null(compare) && !agx_is_null(packed_offset)) {
      /* A 32-bit immediate inside a COLLECT is not a shape
       * agx_validate_sources() models, and it exits the process rather than
       * failing the shader closed. */
      if (packed_offset.type == AGX_INDEX_IMMEDIATE)
         packed_offset = agx_mov_imm(b, 32, packed_offset.value);

      const unsigned offset_channels = agx_channels(packed_offset);
      agx_index sources[4] = {compare};
      assert(offset_channels <= 3);
      if (offset_channels == 1) {
         sources[1] = packed_offset;
      } else {
         agx_index offset_components[3] = {
            agx_null(), agx_null(), agx_null()
         };
         /* Vector loads are contiguous but do not necessarily have an entry
          * in allocated_vec. Split this texture operand explicitly instead
          * of assuming its producer was an agx_emit_collect_to(). The split
          * remains a pseudo instruction and coalesces with a preceding
          * collect when one exists. */
         agx_emit_split(b, offset_components, packed_offset, offset_channels);
         for (unsigned c = 0; c < offset_channels; ++c)
            sources[c + 1] = offset_components[c];
      }
      compare_offset = agx_emit_collect(b, offset_channels + 1, sources);
   }
   else if (!agx_is_null(packed_offset))
      compare_offset = packed_offset;
   else if (!agx_is_null(compare))
      compare_offset = compare;

   agx_index tmp = agx_vec_temp(b->shader, dst.size, instr->is_sparse ? 8 : 4);
   agx_instr *I = agx_texture_sample_to(
      b, tmp, coords, lod, bindless, texture, sampler, compare_offset,
      agx_tex_dim(instr->sampler_dim, instr->is_array), lod_mode, 0,
      offset_present, !agx_is_null(compare),
      instr->op == nir_texop_lod, agx_gather_for_nir(instr), instr->is_sparse);

   /* The packer reads the G17P offset from the instruction, not from src[5],
    * so it survives being split away from the comparator above. */
   I->offset_xyz = offset_xyz;

   if (instr->op == nir_texop_txf || instr->op == nir_texop_txf_ms) {
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      b->shader->out->uses_txf = true;
      b->shader->out->sampler_state_count =
         MAX2(b->shader->out->sampler_state_count, 1);
   }

   /* Destination masking doesn't seem to work properly for gathers (because
    * it's mostly pointless), but it does show up in the lowering of
    * textureGatherOffsets. Don't try to mask the destination for gathers.
    *
    * TODO: Check if it works with sparse.
    */
   bool masked = (instr->op != nir_texop_tg4) && !instr->is_sparse;
   I->mask = agx_expand_tex_to(b, &instr->def, tmp, masked);
}

static void
agx_emit_jump(agx_builder *b, nir_jump_instr *instr)
{
   if (instr->type == nir_jump_halt) {
      agx_stop(b);
   } else {
      assert(instr->type == nir_jump_break);
      agx_block_add_successor(b->shader->current_block, b->shader->break_block);
      agx_break(b, b->shader->loop_nesting + 1, b->shader->break_block);
   }

   b->shader->current_block->unconditional_jumps = true;
}

static void
agx_emit_phi(agx_builder *b, nir_phi_instr *instr)
{
   agx_instr *I =
      agx_phi_to(b, agx_def_index(&instr->def), exec_list_length(&instr->srcs));

   /* Deferred */
   I->phi = instr;
}

/* Look up the AGX block corresponding to a given NIR block. Used when
 * translating phi nodes after emitting all blocks.
 */
static agx_block *
agx_from_nir_block(agx_context *ctx, nir_block *block)
{
   return ctx->indexed_nir_blocks[block->index];
}

static void
agx_emit_phi_deferred(agx_context *ctx, agx_block *block, agx_instr *I)
{
   nir_phi_instr *phi = I->phi;
   I->phi = NULL;

   /* Guaranteed by lower_phis_to_scalar */
   assert(phi->def.num_components == 1);

   nir_foreach_phi_src(src, phi) {
      agx_block *pred = agx_from_nir_block(ctx, src->pred);
      unsigned i = agx_predecessor_index(block, pred);
      assert(i < I->nr_srcs);

      I->src[i] = agx_src_index(&src->src);
   }
}

static void
agx_emit_phis_deferred(agx_context *ctx)
{
   agx_foreach_block(ctx, block) {
      agx_foreach_phi_in_block(block, I)
         agx_emit_phi_deferred(ctx, block, I);
   }
}

static void
agx_emit_undef(agx_builder *b, nir_undef_instr *instr)
{
   /* For now, just lower undefs to zero. This doesn't matter too much, since
    * the lowering happens in NIR and this just allows for late lowering passes
    * to result in undefs.
    */
   if (instr->def.num_components > 1) {
      assert(instr->def.num_components <= 4);
      agx_index zero = agx_mov_imm(b, instr->def.bit_size, 0);

      agx_emit_collect_to(b, agx_def_index(&instr->def),
                          instr->def.num_components,
                          (agx_index[4]){zero, zero, zero, zero});
   } else {
      agx_mov_imm_to(b, agx_def_index(&instr->def), 0);
   }
}

static void
agx_emit_instr(agx_builder *b, struct nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
      agx_emit_load_const(b, nir_instr_as_load_const(instr));
      break;

   case nir_instr_type_intrinsic:
      agx_emit_intrinsic(b, nir_instr_as_intrinsic(instr));
      break;

   case nir_instr_type_alu:
      agx_emit_alu(b, nir_instr_as_alu(instr));
      break;

   case nir_instr_type_tex:
      agx_emit_tex(b, nir_instr_as_tex(instr));
      break;

   case nir_instr_type_jump:
      agx_emit_jump(b, nir_instr_as_jump(instr));
      break;

   case nir_instr_type_phi:
      agx_emit_phi(b, nir_instr_as_phi(instr));
      break;

   case nir_instr_type_undef:
      agx_emit_undef(b, nir_instr_as_undef(instr));
      break;

   default:
      UNREACHABLE("should've been lowered");
   }
}

static agx_block *
agx_create_block(agx_context *ctx)
{
   agx_block *blk = rzalloc(ctx, agx_block);

   /* This is conservative, TODO: divergence analysis */
   blk->divergent = ctx->total_nesting > 0;

   util_dynarray_init(&blk->predecessors, blk);

   return blk;
}

static agx_block *
emit_block(agx_context *ctx, nir_block *block)
{
   if (ctx->after_block) {
      ctx->current_block = ctx->after_block;
      ctx->after_block = NULL;
   } else {
      ctx->current_block = agx_create_block(ctx);
   }

   agx_block *blk = ctx->current_block;
   list_addtail(&blk->link, &ctx->blocks);
   list_inithead(&blk->instructions);

   ctx->indexed_nir_blocks[block->index] = blk;

   agx_builder _b = agx_init_builder(ctx, agx_after_block(blk));

   nir_foreach_instr(instr, block) {
      agx_emit_instr(&_b, instr);
   }

   return blk;
}

static agx_block *emit_cf_list(agx_context *ctx, struct exec_list *list);

static void
emit_if(agx_context *ctx, nir_if *nif)
{
   agx_block *first_block = ctx->current_block;
   agx_builder _b = agx_init_builder(ctx, agx_after_block(first_block));
   agx_index cond = agx_src_index(&nif->condition);

   agx_instr *if_ = agx_if_icmp(&_b, cond, agx_zero(), 1, AGX_ICOND_UEQ, true,
                                NULL /* filled in later */);
   ctx->loop_nesting++;
   ctx->total_nesting++;

   /* Emit the two subblocks. */
   agx_block *if_block = emit_cf_list(ctx, &nif->then_list);
   agx_block *end_then = ctx->current_block;

   _b.cursor = agx_after_block(ctx->current_block);

   agx_block *else_block = emit_cf_list(ctx, &nif->else_list);
   agx_block *end_else = ctx->current_block;

   /* If the "if" fails, we fallthrough to the else */
   if_->target = else_block;

   /* Insert an else instruction at the beginning of the else block. We use
    * "else_fcmp 0.0, 0.0, eq" as unconditional else, matching the blob.
    *
    * If it fails, we fall through to the logical end of the last else block.
    */
   _b.cursor = agx_before_block(else_block);
   agx_else_fcmp(&_b, agx_zero(), agx_zero(), 1, AGX_FCOND_EQ, false, end_else);

   ctx->after_block = agx_create_block(ctx);

   agx_block_add_successor(first_block, if_block);
   agx_block_add_successor(first_block, else_block);
   agx_block_add_successor(end_then, ctx->after_block);
   agx_block_add_successor(end_else, ctx->after_block);

   _b.cursor = agx_after_block(ctx->current_block);
   agx_pop_exec(&_b, 1);
   ctx->loop_nesting--;
   ctx->total_nesting--;
}

static void
emit_loop(agx_context *ctx, nir_loop *nloop)
{
   assert(!nir_loop_has_continue_construct(nloop));
   /* We only track nesting within the innermost loop, so push and reset */
   unsigned pushed_nesting = ctx->loop_nesting;
   ctx->loop_nesting = 0;
   ctx->total_nesting++;

   agx_block *popped_break = ctx->break_block;

   ctx->break_block = agx_create_block(ctx);
   agx_block *loop_header = agx_create_block(ctx);

   /* AGX3 loops always open two architectural exec-mask levels before their
    * body. Apple's top-level prodloop witness starts with the same
    * unconditional CNDST(2) used by nested loops, and the shared footer pops
    * those two levels. AGX2 needs the push only inside other control flow,
    * where threads may already be masked off (TODO: divergence analysis).
    */
   agx_builder _b = agx_init_builder(ctx, agx_after_block(ctx->current_block));
   const bool agx3 = ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX2;
   const bool top_level_agx3 = agx3 && ctx->total_nesting == 1;
   if (top_level_agx3)
      agx_push_exec(&_b, 0);
   if (agx3 || ctx->total_nesting > 1)
      agx_push_exec(&_b, 1);

   /* Fallthrough to loop header */
   agx_block_add_successor(ctx->current_block, loop_header);

   /* Emit the loop */
   ctx->after_block = loop_header;
   loop_header->loop_header = true;
   agx_block *start_block = emit_cf_list(ctx, &nloop->body);

   _b.cursor = agx_after_block(ctx->current_block);
   agx_jmp_exec_any(&_b, start_block);
   agx_pop_exec(&_b, 1);
   if (top_level_agx3)
      agx_pop_exec(&_b, 1);
   agx_block_add_successor(ctx->current_block, loop_header);

   /* Pop off */
   ctx->after_block = ctx->break_block;
   ctx->break_block = popped_break;

   /* Update shader-db stats */
   ++ctx->loop_count;

   /* All nested control flow must have finished */
   assert(ctx->loop_nesting == 0);

   /* Restore loop nesting (we might be inside an if inside an outer loop) */
   ctx->loop_nesting = pushed_nesting;
   ctx->total_nesting--;
}

/* Before the first control flow structure, the nesting counter needs to be
 * zeroed for correct operation. This only happens at most once, since by
 * definition this occurs at the end of the first block, which dominates the
 * rest of the program. */

static void
emit_first_cf(agx_context *ctx)
{
   if (ctx->any_cf)
      return;

   agx_builder _b = agx_init_builder(ctx, agx_after_block(ctx->current_block));
   agx_begin_cf(&_b);
   ctx->any_cf = true;
}

static agx_block *
emit_cf_list(agx_context *ctx, struct exec_list *list)
{
   agx_block *start_block = NULL;

   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block: {
         agx_block *block = emit_block(ctx, nir_cf_node_as_block(node));

         if (!start_block)
            start_block = block;

         break;
      }

      case nir_cf_node_if:
         emit_first_cf(ctx);
         emit_if(ctx, nir_cf_node_as_if(node));
         break;

      case nir_cf_node_loop:
         emit_first_cf(ctx);
         emit_loop(ctx, nir_cf_node_as_loop(node));
         break;

      default:
         UNREACHABLE("Unknown control flow");
      }
   }

   return start_block;
}

static void
agx_set_st_vary_final(agx_context *ctx)
{
   agx_foreach_instr_global_rev(ctx, I) {
      if (I->op == AGX_OPCODE_ST_VARY) {
         I->last = true;
         return;
      }
   }

   /* If we got here, there was no varying written. We need to mark that. */
   agx_block *last_block = list_last_entry(&ctx->blocks, agx_block, link);
   agx_builder _b = agx_init_builder(ctx, agx_after_block_logical(last_block));
   agx_no_varyings(&_b);
}

static void
agx_calc_stats(agx_context *ctx, unsigned size, struct agx2_stats *stats)
{
   struct agx_cycle_estimate cycles = agx_estimate_cycles(ctx);
   uint32_t old_preamble_inst = stats->preamble_inst;

   *stats = (struct agx2_stats){
      .alu = cycles.alu,
      .fscib = cycles.f_scib,
      .ic = cycles.ic,
      .code_size = size,
      .gprs = ctx->max_reg,
      .uniforms = ctx->out->push_count,
      .scratch = ctx->scratch_size_B,
      .threads = agx_occupancy_for_register_count(ctx->max_reg).max_threads,
      .loops = ctx->loop_count,
      .preamble_inst = old_preamble_inst,
   };

   /* Count instructions */
   agx_foreach_instr_global(ctx, I) {
      stats->instrs++;

      if (I->op == AGX_OPCODE_STACK_STORE)
         stats->spills++;
      else if (I->op == AGX_OPCODE_STACK_LOAD)
         stats->fills++;
   }
}

static bool
agx_lower_sincos_filter(const nir_instr *instr, UNUSED const void *_)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   return alu->op == nir_op_fsin || alu->op == nir_op_fcos;
}

/* Sine and cosine are implemented via the sin_pt_1 and sin_pt_2 opcodes for
 * heavy lifting. sin_pt_2 implements sinc in the first quadrant, expressed in
 * turns (sin (tau x) / x), while sin_pt_1 implements a piecewise sign/offset
 * fixup to transform a quadrant angle [0, 4] to [-1, 1]. The NIR opcode
 * fsin_agx models the fixup, sinc, and multiply to obtain sine, so we just
 * need to change units from radians to quadrants modulo turns. Cosine is
 * implemented by shifting by one quadrant: cos(x) = sin(x + tau/4).
 */

static nir_def *
agx_lower_sincos_impl(struct nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *x = nir_mov_alu(b, alu->src[0], 1);

   if ((uintptr_t)data && x->bit_size <= 32) {
      /* Native G17P sin/cos use three non-overlapping pieces of 2/pi.
       * The magic rounding add retains the quadrant in its low two bits.
       * Keep the explicit FMAs and subtraction order, including low pieces.
       * Native trig-fast text hashes are retained with the focused tests. */
      const unsigned bits = x->bit_size;
      const uint32_t saved_ctrl = b->fp_math_ctrl;
      b->fp_math_ctrl = nir_fp_no_fast_math;
      x = nir_f2f32(b, x);
      nir_def *hi = nir_imm_float(b, uif(0x3f22f983));
      nir_def *magic = nir_imm_float(b, uif(0x4b400000));
      nir_def *quadrant = nir_ffma(b, x, hi, magic);
      nir_def *r = nir_ffma(b, x, hi, nir_fsub(b, magic, quadrant));
      r = nir_ffma(b, x, nir_imm_float(b, uif(0x32dc9c88)), r);
      r = nir_ffma(b, x, nir_imm_float(b, uif(0x25a94fe1)), r);
      nir_def *complement = nir_fsub(b, nir_imm_float(b, 1), nir_fabs(b, r));
      const bool cosine = alu->op == nir_op_fcos;
      r = nir_bittest_select_agx3(b, quadrant, nir_imm_int(b, 1),
                                 cosine ? nir_fneg(b, r) : complement,
                                 cosine ? complement : r);
      r = nir_bittest_select_agx3(b, quadrant, nir_imm_int(b, 2),
                                 nir_fneg(b, r), r);
      nir_def *result = nir_fmul(b, nir_fsinc_agx3(b, r), r);
      nir_def *small = cosine ? nir_imm_float(b, 1) : x;
      result = nir_bcsel(b, nir_flt_imm(b, nir_fabs(b, x), 0x1p-12f),
                        small, result);
      result = nir_bcsel(b, nir_flt(b, nir_imm_float(b, uif(0x4ac90fdb)),
                                  nir_fabs(b, x)),
                        nir_imm_float(b, 0), result);
      if (bits != 32)
         result = nir_f2fN(b, result, bits);
      b->fp_math_ctrl = saved_ctrl;
      return result;
   }

   nir_def *turns = nir_fmul_imm(b, x, M_1_PI * 0.5f);

   if (alu->op == nir_op_fcos)
      turns = nir_fadd_imm(b, turns, 0.25f);

   nir_def *quadrants = nir_fmul_imm(b, nir_ffract(b, turns), 4.0);
   return nir_fsin_agx(b, quadrants);
}

static bool
agx_lower_sincos(nir_shader *shader, bool g17p)
{
   return nir_shader_lower_instructions(shader, agx_lower_sincos_filter,
                                        agx_lower_sincos_impl,
                                        (void *)(uintptr_t)g17p);
}

static bool
agx_lower_front_face(struct nir_builder *b, nir_intrinsic_instr *intr,
                     UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_front_face)
      return false;

   nir_def *def = &intr->def;
   assert(def->bit_size == 1);

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_rewrite_uses(def, nir_inot(b, nir_load_back_face_agx(b, 1)));
   return true;
}

/*
 * Standard NIR optimization loop. This is run in agx_preprocess_nir, then once
 * again at shader variant compile time. Unless there was a complex shader key,
 * the latter run should be almost a no-op.
 */
static void
agx_optimize_loop_nir(nir_shader *nir)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = 64,
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &peephole_select_options);
      NIR_PASS(progress, nir, nir_opt_phi_precision);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);
}

static bool
agx_nir_lower_imul64_filter(const nir_instr *instr, const void *_)
{
   if (instr->type != nir_instr_type_alu)
      return false;

   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   return alu->op == nir_op_imul && alu->def.bit_size == 64;
}

static nir_def *
agx_nir_umul_high32(nir_builder *b, nir_def *x, nir_def *y)
{
   nir_def *shift = nir_imm_int(b, 16);
   nir_def *mask = nir_imm_int(b, 0xffff);

   /* Hacker's Delight unsigned 16-bit partial-product expansion. */
   nir_def *xl = nir_iand(b, x, mask);
   nir_def *yl = nir_iand(b, y, mask);
   nir_def *xh = nir_ushr(b, x, shift);
   nir_def *yh = nir_ushr(b, y, shift);
   nir_def *lo = nir_imul(b, xl, yl);
   nir_def *m1 = nir_imul(b, xl, yh);
   nir_def *m2 = nir_imul(b, xh, yl);
   nir_def *hi = nir_imul(b, xh, yh);
   nir_def *t = nir_iadd(b, m2, nir_ushr(b, lo, shift));
   nir_def *w1 = nir_iadd(b, m1, nir_iand(b, t, mask));
   nir_def *w2 = nir_ushr(b, t, shift);
   return nir_iadd(b, hi, nir_iadd(b, nir_ushr(b, w1, shift), w2));
}

static nir_def *
agx_nir_lower_imul64_instr(nir_builder *b, nir_instr *instr, void *_)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *y = nir_ssa_for_alu_src(b, alu, 1);
   nir_def *xl = nir_unpack_64_2x32_split_x(b, x);
   nir_def *xh = nir_unpack_64_2x32_split_y(b, x);
   nir_def *yl = nir_unpack_64_2x32_split_x(b, y);
   nir_def *yh = nir_unpack_64_2x32_split_y(b, y);
   nir_def *hi = nir_iadd(
      b, agx_nir_umul_high32(b, xl, yl),
      nir_iadd(b, nir_imul(b, xl, yh), nir_imul(b, xh, yl)));

   return nir_pack_64_2x32_split(b, nir_imul(b, xl, yl), hi);
}

static bool
agx_nir_lower_imul64(nir_shader *nir)
{
   return nir_shader_lower_instructions(
      nir, agx_nir_lower_imul64_filter, agx_nir_lower_imul64_instr, NULL);
}

static bool
agx_nir_has_alu64(const nir_shader *nir, nir_op opcode)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;

            const nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == opcode && alu->def.bit_size == 64)
               return true;
         }
      }
   }

   return false;
}

bool
agx_mem_vectorize_cb(unsigned align_mul, unsigned align_offset,
                     unsigned bit_size, unsigned num_components,
                     int64_t hole_size, nir_intrinsic_instr *low,
                     nir_intrinsic_instr *high, void *data)
{
   if (hole_size > 0)
      return false;

   /* Must be aligned to the size of the load */
   unsigned align = nir_combined_align(align_mul, align_offset);
   if ((bit_size / 8) > align)
      return false;

   if (num_components > 4)
      return false;

   if (bit_size > 32)
      return false;

   return true;
}

static bool
set_speculate(nir_builder *b, nir_instr *instr, UNUSED void *_)
{
   if (instr->type == nir_instr_type_intrinsic &&
       nir_intrinsic_has_access(nir_instr_as_intrinsic(instr))) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      nir_intrinsic_set_access(
         intr, ACCESS_CAN_SPECULATE | nir_intrinsic_access(intr));
      return true;
   }

   if (instr->type == nir_instr_type_tex) {
      nir_instr_as_tex(instr)->can_speculate = true;
      return true;
   }

   return false;
}

static void
agx_optimize_nir(nir_shader *nir, bool soft_fault, bool uniform_atomics,
                 bool byte_indices,
                 uint16_t *preamble_size, uint8_t *ts_count, uint8_t *ss_count)
{
   /* This runs only once up front since other optimizations don't affect it */
   NIR_PASS(_, nir, nir_opt_shrink_stores, true);

   agx_optimize_loop_nir(nir);

   /* If soft fault is enabled, we can freely speculate everything. That lets us
    * peephole select and form preambles more aggressively.
    */
   if (soft_fault) {
      NIR_PASS(_, nir, nir_shader_instructions_pass, set_speculate,
               nir_metadata_control_flow, NULL);
   }

   /* Peephole select again after setting the speculate flag but before
    * vectorizing. This cleans up short-circuit loads in unrolled loops.
    *
    * XXX: Set indirect_load_ok once we can investigate CTS flakes.
    */
   nir_opt_peephole_select_options peephole_select_options = {
      .limit = 64,
      .expensive_alu_ok = true,
   };
   NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);

   NIR_PASS(_, nir, nir_opt_load_store_vectorize,
            &(const nir_load_store_vectorize_options){
               .modes = nir_var_mem_global | nir_var_mem_constant |
                        nir_var_shader_temp,
               .callback = agx_mem_vectorize_cb,
            });
   NIR_PASS(_, nir, nir_lower_pack);
   NIR_PASS(_, nir, nir_opt_algebraic);

   nir_divergence_analysis(nir);
   bool progress = false;

   static const nir_lower_subgroups_options subgroups_options = {
      .ballot_bit_size = 32,
      .ballot_components = 1,
      .lower_elect = true,
      .lower_subgroup_masks = true,
   };

   /*
    * Rewrites a uniform-address atomic into "one lane of the subgroup performs
    * the whole operation", which needs ACTIVE_THREAD_INDEX_IN_SUBGROUP and
    * TOTAL_ACTIVE_THREADS_IN_SUBGROUP. Apple's AGX3 numbers for those two
    * special registers are not witnessed (agx3_map_sr() carries only the
    * registers we have witnesses for), so on AGX3 the optimized form cannot
    * be encoded and every shader containing a uniform-address atomic failed
    * to compile.
    *
    * Skipping the optimization keeps atomics CORRECT -- each lane performs its
    * own atomic, which is what the unoptimized program means -- at the cost of
    * one atomic per lane instead of one per subgroup on uniform addresses.
    * Re-enable once those two SR numbers are witnessed rather than guessed; a
    * wrong SR number would silently elect the wrong lane.
    */
   if (uniform_atomics)
      NIR_PASS(progress, nir, nir_opt_uniform_atomics, true);

   /* Lower addressing modes. The sooner we do this, the sooner we get rid of
    * amul/aadd instructions and can let nir_opt_algebraic do its job. But we
    * want to vectorize first since nir_opt_load_store_vectorize doesn't know
    * how to handle our loads. Likewise for uniform atomic optimization.
    */
   NIR_PASS(_, nir, agx_nir_lower_address, byte_indices);

   /* Now that loads carry an index, a select between two of them can choose
    * the index instead of the loaded data. Run it to a fixed point with the
    * cleanup below, so a chain of candidate selects collapses to one load.
    * poly's geometry rasterization copy loaded every candidate vertex per
    * varying, which overflowed the AGX3B register file and spilled.
    */
   NIR_PASS(_, nir, agx_nir_opt_select_load);
   NIR_PASS(_, nir, nir_opt_cse);
   NIR_PASS(_, nir, nir_opt_dce);

   NIR_PASS(progress, nir, nir_opt_uniform_subgroup, &subgroups_options);
   if (progress) {
      if (byte_indices)
         NIR_PASS(_, nir, agx_nir_lower_subgroups_g17p);
      else
         NIR_PASS(_, nir, agx_nir_lower_subgroups);
   }

   /* The above create operations that need lowering/optimizing */
   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);

   progress = false;

   /* If address lowering made progress, clean up before forming preambles.
    * Otherwise the optimized preambles might just be constants! Do it before
    * lowering int64 too, to avoid lowering constant int64 arithmetic.
    */
   if (progress) {
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_dce);
   }

   /* Only lower int64 after optimizing address arithmetic, so that u2u64/i2i64
    * conversions remain.
    */
   progress = false;
   NIR_PASS(progress, nir, nir_lower_int64);

   /* If we lowered actual int64 arithmetic (not folded into the address
    * calculations), then clean up after the lowering.
    */
   if (progress) {
      do {
         progress = false;

         NIR_PASS(progress, nir, nir_opt_algebraic);
         NIR_PASS(progress, nir, nir_opt_constant_folding);
         NIR_PASS(progress, nir, nir_opt_dce);
      } while (progress);
   }

   /* Reassociate before forming preambles because it makes preambles more
    * effective. Clean up after.
    */
   nir_opt_reassociate_loop(
      nir, nir_reassociate_scalar_math | nir_reassociate_cse_heuristic);

   /* Lower fmin/fmax before optimizing preambles so we can see across uniform
    * expressions. Do it after nir_opt_reassociate because nir_opt_reassociate
    * should work on unlowered fmin/fmax.
    */
   NIR_PASS(_, nir, agx_nir_lower_fminmax);

   /* Force preamble generation for internal shaders since hk's meta copies
    * depend on it for correctness with imgwblk intrinsics.
    */
   if (preamble_size &&
       (!(agx_compiler_debug & AGX_DBG_NOPREAMBLE) || nir->info.internal)) {
      unsigned sizes[] = {
         *preamble_size,
         ts_count ? *ts_count : 1000 /* large finite */,
         ss_count ? *ss_count : 1000 /* large finite */,
      };

      /* Don't clobber txf sampler */
      if (sizes[2] == 0)
         sizes[2]++;

      NIR_PASS(_, nir, agx_nir_opt_preamble, sizes);

      *preamble_size = sizes[0];

      if (ts_count)
         *ts_count = sizes[1];

      /* if something other than the txf sampler is written... */
      if (ss_count && sizes[2] > 1)
         *ss_count = sizes[2];
   }

   /* Forming preambles may dramatically reduce the instruction count
    * in certain blocks, causing some if-else statements to become
    * trivial. We want to peephole select those, given that control flow
    * prediction instructions are costly.
    *
    * We need to lower int64 again to deal with the resulting 64-bit csels.
    */
   NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);
   NIR_PASS(_, nir, nir_lower_int64);

   /* We need to lower fmin/fmax again after nir_opt_algebraic_late due to f2fmp
    * wackiness. This is usually a no-op but is required for correctness in
    * GLES.
    */
   NIR_PASS(_, nir, nir_opt_algebraic_late);
   NIR_PASS(_, nir, agx_nir_lower_fminmax);

   /* Fuse add/sub/multiplies/shifts after running opt_algebraic_late to fuse
    * isub but before shifts are lowered.
    */
   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, agx_nir_fuse_algebraic_late);
   } while (progress);

   /* Before optimizing bounds checks, we need to clean up and index defs so
    * optimize_bounds does the right thing.
    */
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);

   nir_foreach_function_impl(impl, nir) {
      nir_index_ssa_defs(impl);
   }

   /* Do remaining lowering late, since this inserts &s for shifts so we want to
    * do it after fusing constant shifts. Constant folding will clean up.
    */
   NIR_PASS(_, nir, agx_nir_lower_algebraic_late);
   NIR_PASS(_, nir, agx_nir_fuse_selects);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_combine_barriers, NULL, NULL);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_cse);
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);

   /* Cleanup optimizations */
   nir_move_options move_all = nir_move_const_undef | nir_move_load_ubo |
                               nir_move_load_input | nir_move_load_frag_coord |
                               nir_move_comparisons | nir_move_copies |
                               nir_move_load_ssbo | nir_move_alu;

   NIR_PASS(_, nir, nir_opt_sink, move_all);
   NIR_PASS(_, nir, nir_opt_move, move_all);
   NIR_PASS(progress, nir, nir_lower_all_phis_to_scalar);

   /* After lowering phis to scalar, we must copy prop/DCE for correctness. RA
    * can't deal with dead phis.
    */
   do {
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      progress = false;
   } while (progress);
}

/*
 * Varyings that are used as texture coordinates should be kept at fp32, because
 * fp16 does not have enough precision for large textures. It's technically
 * conformant not to, but every app gets this wrong.
 */
static bool
gather_texcoords(nir_builder *b, nir_tex_instr *tex, void *data)
{
   nir_def *coord = nir_get_tex_src(tex, nir_tex_src_coord);
   if (!coord)
      return false;

   nir_scalar x = nir_scalar_resolved(coord, 0);
   nir_scalar y = nir_scalar_resolved(coord, 1);
   nir_intrinsic_instr *intr = nir_scalar_as_intrinsic(x);

   if (x.def == y.def && intr &&
       intr->intrinsic == nir_intrinsic_load_interpolated_input) {

      uint64_t *mask = data;
      *mask |= BITFIELD64_BIT(nir_intrinsic_io_semantics(intr).location);
   }

   return false;
}

/*
 * Build a bit mask of varyings (by location) that are used as texture
 * coordinates. This information is needed by lower_mediump_io.
 */
uint64_t
agx_gather_texcoords(nir_shader *nir)
{
   assert(nir->info.stage == MESA_SHADER_FRAGMENT);

   uint64_t mask = 0;
   nir_shader_tex_pass(nir, gather_texcoords, nir_metadata_all, &mask);
   return mask;
}

static nir_mem_access_size_align
mem_access_size_align_cb(nir_intrinsic_op intrin, uint8_t bytes,
                         uint8_t bit_size, uint32_t align,
                         uint32_t align_offset, bool offset_is_const,
                         enum gl_access_qualifier access, const void *cb_data)
{
   align = nir_combined_align(align, align_offset);

   assert(util_is_power_of_two_nonzero(align));

   if ((bytes & 1) || (align == 1))
      bit_size = 8;
   else if ((bytes & 2) || (align == 2))
      bit_size = 16;
   else if (bit_size >= 32)
      bit_size = 32;

   return (nir_mem_access_size_align){
      .num_components = MIN2(bytes / (bit_size / 8), 4),
      .bit_size = bit_size,
      .align = bit_size / 8,
      .shift = nir_mem_access_shift_method_scalar,
   };
}

static unsigned
lower_bit_size_callback(const nir_instr *instr, UNUSED void *_)
{
   if (instr->type == nir_instr_type_intrinsic) {
      /* Handle small subgroup ops */
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

      switch (intr->intrinsic) {
      case nir_intrinsic_reduce:
      case nir_intrinsic_exclusive_scan:
      case nir_intrinsic_inclusive_scan:
         /* The identity for iand doesn't work for lowered 1-bit booleans, so
          * lower that explicitly.
          */
         if (nir_intrinsic_reduction_op(intr) == nir_op_iand &&
             intr->def.bit_size == 1)
            return 16;

         /* In general, we have 16-bit ops instead of 8-bit, so lower those. */
         return intr->def.bit_size == 8 ? 16 : 0;
      default:
         return 0;
      }
   } else if (instr->type == nir_instr_type_alu) {
      /* Lower 8-bit ALU to 16-bit. We check the destination, as we do not want
       * to lower conversions from 8-bit to larger types. Those conversions get
       * implemented natively.
       */
      nir_alu_instr *alu = nir_instr_as_alu(instr);

      switch (alu->op) {
      case nir_op_bitfield_reverse:
         return alu->def.bit_size < 32 ? 32 : 0;
      default:
         break;
      }

      if (alu->def.bit_size == 8 && !is_conversion_to_8bit(alu->op))
         return 16;
      else if (alu->def.bit_size == 1 && alu->src[0].src.ssa->bit_size == 8)
         return 16 /* comparisons */;
   }

   return 0;
}

static bool
lower_load_from_texture_handle(nir_builder *b, nir_intrinsic_instr *intr,
                               void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_from_texture_handle_agx)
      return false;

   nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def *base = nir_load_preamble(b, 1, 64, nir_intrinsic_desc_set(handle));
   nir_def *offset = nir_u2u64(b, handle->src[0].ssa);

   nir_def_rewrite_uses(&intr->def, nir_iadd(b, base, offset));
   return true;
}

static bool
lower_sink_address(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ro_sink_address_poly:
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
      return true;
   case nir_intrinsic_ro_to_rw_poly: {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(
         &intr->def,
         nir_bcsel(b, nir_ieq_imm(b, intr->src[0].ssa, AGX_ZERO_PAGE_ADDRESS),
                   nir_imm_int64(b, AGX_SCRATCH_PAGE_ADDRESS),
                   intr->src[0].ssa));
      return true;
   }
   default:
      break;
   }

   return false;
}

static void
agx_remove_unreachable_block(agx_block *block)
{
   /* Delete the edges */
   agx_foreach_successor(block, succ) {
      unsigned block_idx = agx_predecessor_index(succ, block);

      /* Remove the corresponding predecessor from the successor */
      struct util_dynarray *blocks = &succ->predecessors;
      int remaining = agx_num_predecessors(succ) - (block_idx + 1);
      assert(remaining >= 0);

      memcpy(util_dynarray_element(blocks, agx_block *, block_idx),
             util_dynarray_element(blocks, agx_block *, block_idx + 1),
             remaining * sizeof(agx_block *));
      blocks->size -= sizeof(agx_block *);

      /* Remove the corresponding source from the phis */
      agx_foreach_phi_in_block(succ, phi) {
         assert(block_idx + 1 <= phi->nr_srcs);

         memcpy(phi->src + block_idx, phi->src + block_idx + 1,
                (phi->nr_srcs - (block_idx + 1)) * sizeof(phi->src[0]));

         phi->nr_srcs--;

         /* This might cause phis to become trivial. Lower 1-source phis to
          * moves and let copyprop take it from here.
          */
         if (phi->nr_srcs == 1) {
            phi->op = AGX_OPCODE_MOV;
         }
      }
   }

   /* Remove the successor from the predecessor. */
   block->successors[0] = NULL;
   block->successors[1] = NULL;

   /* Note: we do not remove the block itself, although it is now fully orphaned
    * in the control flow graph. We still need it in source order if it has any
    * pop_exec instructions, for a loop continue block.
    *
    * TODO: Is there a better way to handle this?
    *
    * Affects: dEQP-VK.graphicsfuzz.cov-matching-if-always-true-inside-loop
    */
}

/*
 * NIR sometimes contains unreachable blocks (e.g. due to infinite loops). These
 * blocks have no predecessors, but do have successors and can contribute to
 * phis. They are dead and do not need to be here. Further, they violate the IR
 * invariant:
 *
 *    Live-in sources are live-out in all predecessors.
 *
 * ...which RA depends on when handling live range splits. The simplest solution
 * is to simply delete these dead blocks. Fortunately, because they are
 * unreachable, this does not have any ill effects. Notably, this cannot
 * introduce critical edges.
 *
 * Deleting a block may cause a successor to become unreachable, so we use a
 * fixed-point algorithm to converge.
 */
static void
agx_remove_unreachable_blocks(agx_context *ctx)
{
   agx_block *start = agx_start_block(ctx);
   bool progress;

   do {
      progress = false;

      agx_foreach_block_safe(ctx, pred) {
         if (pred != start && agx_num_predecessors(pred) == 0 &&
             agx_num_successors(pred) > 0) {

            agx_remove_unreachable_block(pred);
            progress = true;
         }
      }
   } while (progress);
}

static bool
agx_should_dump(nir_shader *nir, unsigned agx_dbg_bit)
{
   return (agx_compiler_debug & agx_dbg_bit) &&
          !(nir->info.internal && !(agx_compiler_debug & AGX_DBG_INTERNAL));
}

#define AGX_PASS(shader, pass, ...)                                            \
   do {                                                                        \
      pass(shader, ##__VA_ARGS__);                                             \
      agx_validate(ctx, #pass);                                                \
   } while (0)

/* G17P constant programs write the uniform file with 32-bit COPY records,
 * including half-valued expressions. Coalesce adjacent scalar halves before
 * RA so both values occupy one real GPR pair and one aligned uniform pair.
 * An isolated half is handled by the packer's separately guarded padding
 * path; odd starts and non-adjacent live halves remain fail-closed. */
bool
agx3_coalesce_uniform_half_stores(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       !ctx->is_preamble)
      return false;

   bool progress = false;
   agx_foreach_block(ctx, block) {
      agx_instr *pending = NULL;
      unsigned pending_base = 0;

      agx_foreach_instr_in_block_safe(block, I) {
         const bool scalar_half =
            I->op == AGX_OPCODE_UNIFORM_STORE && I->nr_srcs == 2 &&
            I->mask == 1 && I->src[0].size == AGX_SIZE_16 &&
            agx_channels(I->src[0]) == 1 &&
            I->src[1].type == AGX_INDEX_IMMEDIATE && !I->src[1].abs &&
            !I->src[1].neg;
         if (!scalar_half) {
            if (I->op == AGX_OPCODE_UNIFORM_STORE)
               pending = NULL;
            if (pending != NULL) {
               agx_foreach_src(I, s) {
                  if (I->src[s].type == AGX_INDEX_UNIFORM)
                     pending = NULL;
               }
            }
            continue;
         }

         const unsigned base = I->src[1].value;
         if (pending != NULL && !(pending_base & 1) &&
             base == pending_base + 1) {
            agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
            agx_index halves[2] = {pending->src[0], I->src[0]};
            agx_index pair = agx_vec_temp(ctx, AGX_SIZE_32, 1);
            agx_emit_collect_to(&b, pair, 2, halves);
            agx_uniform_store(&b, pair, agx_immediate(pending_base), 1);
            agx_remove_instruction(pending);
            agx_remove_instruction(I);
            pending = NULL;
            progress = true;
         } else {
            pending = I;
            pending_base = base;
         }
      }
   }

   return progress;
}

/* Native G17P fragment programs issue independent LDDEVs sharing a register
 * base in increasing address order. The row-major UBO path is the minimal
 * counterexample: NIR presents offsets 16 then 0, both become ready together,
 * and the resulting descending pair returns intermittent stale data while the
 * otherwise identical ascending column-major path is stable. Sort only an
 * adjacent run of non-coherent loads with identical operation and base; SSA
 * keeps their consumers attached to the correct result and no serialization
 * is introduced. */
static bool
agx3_order_fragment_device_loads(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT)
      return false;

   bool progress = false;
   agx_foreach_block(ctx, block) {
      agx_instr *run[16];
      unsigned count = 0;

      agx_foreach_instr_in_block(block, I) {
         const bool candidate =
            I->op == AGX_OPCODE_DEVICE_LOAD && !I->coherent &&
            I->nr_dests == 1 && I->nr_srcs == 2 &&
            I->src[1].type == AGX_INDEX_IMMEDIATE && !I->src[1].abs &&
            !I->src[1].neg && !I->src[1].memory;
         const bool joins =
            candidate && count > 0 && count < ARRAY_SIZE(run) &&
            agx_is_equiv(I->src[0], run[0]->src[0]) &&
            I->format == run[0]->format && I->mask == run[0]->mask &&
            I->memory_index_is_bytes == run[0]->memory_index_is_bytes;

         /* SPLIT only describes the preceding SSA result and emits no machine
          * instruction. It may sit between the loads in an otherwise adjacent
          * hardware run. */
         if (!candidate && I->op == AGX_OPCODE_SPLIT)
            continue;

         if (!candidate || (count > 0 && !joins)) {
            for (unsigned i = 1; i < count; ++i) {
               for (unsigned j = i; j > 0 &&
                    ((uint64_t)run[j - 1]->src[1].value << run[j - 1]->shift) >
                    ((uint64_t)run[j]->src[1].value << run[j]->shift); --j) {
                  agx_instr *lower = run[j];
                  list_del(&lower->link);
                  list_addtail(&lower->link, &run[j - 1]->link);
                  run[j] = run[j - 1];
                  run[j - 1] = lower;
                  progress = true;
               }
            }
            count = 0;
         }

         if (candidate)
            run[count++] = I;
      }

      for (unsigned i = 1; i < count; ++i) {
         for (unsigned j = i; j > 0 &&
              ((uint64_t)run[j - 1]->src[1].value << run[j - 1]->shift) >
              ((uint64_t)run[j]->src[1].value << run[j]->shift); --j) {
            agx_instr *lower = run[j];
            list_del(&lower->link);
            list_addtail(&lower->link, &run[j - 1]->link);
            run[j] = run[j - 1];
            run[j - 1] = lower;
            progress = true;
         }
      }
   }

   return progress;
}

/* Keep scalar UVS values live until a native-shaped terminal writeout.
 * Run after pressure scheduling but before RA: moving physical-register stores
 * would lose values when later instructions reuse their sources. */
bool
agx3_sink_vertex_outputs(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_VERTEX || ctx->is_preamble ||
       ctx->key->secondary)
      return false;

   /* Outputs already belonging to the final block can be bundled locally
    * even when earlier blocks contain instance transforms. NIR may also
    * leave the stores in the immediately preceding block and put only STOP
    * in a synthetic final block. Treat that single-predecessor,
    * single-successor fallthrough as one terminal region: moving uses over
    * that edge cannot change their execution mask. */
   agx_block *previous = NULL, *stop_block = NULL;
   agx_foreach_block(ctx, block) {
      previous = stop_block;
      stop_block = block;
   }
   if (stop_block == NULL)
      return false;

   agx_block *output_block = stop_block;
   bool final_has_store = false;
   agx_foreach_instr_in_block(stop_block, I)
      final_has_store |= I->op == AGX_OPCODE_ST_VARY;

   if (!final_has_store && previous &&
       agx_num_successors(previous) == 1 &&
       previous->successors[0] == stop_block &&
       agx_num_predecessors(stop_block) == 1 &&
       *util_dynarray_element(&stop_block->predecessors, agx_block *, 0) ==
          previous &&
       previous->divergent == stop_block->divergent) {
      output_block = previous;
   }

   agx_foreach_block(ctx, block) {
      if (block == output_block)
         break;
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_ST_VARY || I->op == AGX_OPCODE_STOP)
            return false;
      }
   }

   agx_instr *stores[AGX3_MAX_VERTEX_OUTPUTS] = {0}, *stop = NULL;
   if (output_block != stop_block) {
      agx_foreach_instr_in_block(stop_block, I) {
         if (I->op != AGX_OPCODE_STOP || stop)
            return false;
         stop = I;
      }
   }

   bool seen_stop = false;
   agx_foreach_instr_in_block(output_block, I) {
      if (seen_stop)
         return false;
      switch (I->op) {
      case AGX_OPCODE_STOP:
         if (output_block != stop_block || stop)
            return false;
         stop = I;
         seen_stop = true;
         break;
      case AGX_OPCODE_ST_VARY: {
         if (I->nr_dests || I->nr_srcs != 2 ||
             I->src[0].type != AGX_INDEX_IMMEDIATE || I->src[0].value >= ARRAY_SIZE(stores) ||
             I->src[1].type != AGX_INDEX_NORMAL || I->src[1].memory ||
             I->src[1].size != AGX_SIZE_32 || agx_channels(I->src[1]) != 1 ||
             stores[I->src[0].value])
            return false;
         stores[I->src[0].value] = I;
         break;
      }
      default:
         /* Pure SSA producers may move relative to vertex writeout. Their
          * actual encodings remain validated by the normal compiler path;
          * side effects and control flow require a separate ordering proof. */
         if (!agx_opcodes_info[I->op].can_eliminate)
            return false;
         break;
      }
   }
   if (!stop)
      return false;
   unsigned count = 0;
   while (count < ARRAY_SIZE(stores) && stores[count])
      ++count;
   if (count < 5)
      return false;
   for (unsigned i = count; i < ARRAY_SIZE(stores); ++i) {
      if (stores[i])
         return false;
   }
   for (unsigned i = 0; i < count; ++i) {
      list_del(&stores[i]->link);
      list_addtail(&stores[i]->link, &stop->link);
   }
   return true;
}

/* Apple's G17P multi-RT fragments finish colour partitions in descending
 * order and make partition zero the final release.  This holds for both the
 * opaque two-RGBA8 witness (offsets 4, 0) and the mixed RGBA16F+R8 witnesses
 * (offsets 8, 0), including programmable blending.  Sink the stores while
 * they are still in SSA so every payload remains live through the native
 * ordering and RA cannot reuse an earlier target's physical register.
 *
 * This is partition-driven, not format- or workload-driven.  It is restricted
 * to straight-line fragment epilogs and internal background programs, and it
 * refuses any non-colour side effect other than a leading depth export.  The
 * packer independently proves the partition graph before selecting the
 * corresponding pixel classes.
 */
static bool
agx3_order_g17p_mrt_tile_ops(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       (!ctx->key->secondary && !ctx->key->fs.ignore_tib_dependencies) ||
       list_length(&ctx->blocks) != 1)
      return false;

   /* A multisample epilog stores every target through the coverage word
    * register (agx_nir_lower_g17p_tile_samples). */
   const bool multisample_stores =
      ctx->key->fs.nr_samples > 1 && !ctx->key->fs.ignore_tib_dependencies;
   agx_instr *loads[AGX_MAX_RTS] = {0};
   agx_instr *stores[AGX_MAX_RTS] = {0};
   agx_instr *stop = NULL;
   agx_instr *first_wait = NULL;
   unsigned load_count = 0, store_count = 0;
   /* Apple's store-only MRT fragments that export depth complete the class-1
    * depth transaction before the first colour transaction, which follows
    * unchanged. Admit exactly that leading wait, depth write and signal. */
   unsigned depth_ops = 0;
   agx_foreach_block(ctx, block) {
      agx_foreach_instr_in_block(block, I) {
         const bool colour_started = first_wait || load_count || store_count;
         switch (I->op) {
         case AGX_OPCODE_PRELOAD:
         case AGX_OPCODE_COLLECT:
         case AGX_OPCODE_SPLIT:
            break;
         case AGX_OPCODE_WAIT_PIX:
            if (I->writeout == 1) {
               if (depth_ops != 0 || colour_started)
                  return false;
               depth_ops = 1;
               break;
            }
            if (depth_ops == 1 || depth_ops == 2)
               return false;
            if (!first_wait)
               first_wait = I;
            break;
         case AGX_OPCODE_ZS_EMIT_AGX3:
            if (depth_ops != 1)
               return false;
            depth_ops = 2;
            break;
         case AGX_OPCODE_SIGNAL_PIX:
            if (depth_ops != 2 || I->writeout != 1)
               return false;
            depth_ops = 3;
            break;
         case AGX_OPCODE_FLTNORM_AGX3:
            break;
         case AGX_OPCODE_ST_TILE:
            if (store_count >= ARRAY_SIZE(stores) || I->nr_dests != 0 ||
                I->nr_srcs != 3 || I->explicit_coords ||
                I->src[0].type != AGX_INDEX_NORMAL || I->src[0].memory ||
                !(multisample_stores
                     ? I->src[1].type == AGX_INDEX_NORMAL &&
                          I->src[1].size == AGX_SIZE_16
                     : I->src[1].type == AGX_INDEX_IMMEDIATE &&
                          I->src[1].value == 0xff) ||
                !agx_is_null(I->src[2]))
               return false;
            for (unsigned i = 0; i < store_count; ++i) {
               if (stores[i]->pixel_offset == I->pixel_offset)
                  return false;
            }
            stores[store_count++] = I;
            break;
         case AGX_OPCODE_LD_TILE:
            if (load_count >= ARRAY_SIZE(loads) || I->nr_dests != 1 ||
                I->nr_srcs != 2 || I->explicit_coords ||
                I->src[0].type != AGX_INDEX_IMMEDIATE ||
                I->src[0].value != 0xff || !agx_is_null(I->src[1]))
               return false;
            for (unsigned i = 0; i < load_count; ++i) {
               if (loads[i]->pixel_offset == I->pixel_offset)
                  return false;
            }
            loads[load_count++] = I;
            break;
         case AGX_OPCODE_STOP:
            if (stop)
               return false;
            stop = I;
            break;
         default:
            if (!agx_opcodes_info[I->op].can_eliminate)
               return false;
            break;
         }
      }
   }

   /* An epilog writing only some of the framebuffer's targets keeps each
    * store's partition. A lone store to partition zero is the ordinary
    * single-target protocol. */
   const uint64_t partitions = ctx->key->g17p_tile_partitions;
   uint64_t written = 0;
   for (unsigned i = 0; i < store_count; ++i) {
      if (stores[i]->pixel_offset >= 64)
         return false;
      written |= BITFIELD64_BIT(stores[i]->pixel_offset);
   }
   const bool subset = partitions && written != partitions;
   /* Without a partition key the attachment names itself, and a lone store to
    * an attachment other than zero is still a multi-target layout: it owns
    * its own class rather than class pair zero
    * (g17p-sparse-mrt.h, fragment-8-active-mask04). */
   const bool lone_sparse_store =
      !partitions && store_count == 1 && stores[0]->colour_target != 0;
   if (!stop || (partitions && (written & ~partitions)) ||
       (!lone_sparse_store &&
        store_count < (subset && written != BITFIELD64_BIT(0) ? 1 : 2)) ||
       ((subset || multisample_stores) && load_count) ||
       (depth_ops && (depth_ops != 3 || load_count)))
      return false;

   /* Native raw32 BG clear/reload keeps ascending stores and has no
    * ordinary colour transactions, including two/three RGBA32 targets. Other
    * multi-target BG programs have no transactions either and store in no
    * required partition order.
    *
    * The wide-MRT class protocol is a FRAGMENT witness
    * (test/probes/g17p-wide-mrt/), and extending it to clears is not
    * supported by anything: a background compiled with the transaction no
    * longer matches agx3_g17p_rt0_bg_program_generate()'s loader
    * descriptor, and KHR-GL30.draw_buffers.draw_buffers_1 aborts the
    * process at "background Ioto program" on real hardware. */
   if (ctx->key->fs.ignore_tib_dependencies) {
      bool rgba32 = !load_count;
      for (unsigned i = 0; i < store_count; ++i)
         rgba32 &= stores[i]->format == AGX_FORMAT_I32 &&
                   stores[i]->mask == 15 && stores[i]->pixel_offset == 16 * i;
      bool rgba8 = !load_count && store_count >= 3;
      for (unsigned i = 0; i < store_count; ++i)
         rgba8 &= stores[i]->format == AGX_FORMAT_U8NORM &&
                  stores[i]->mask == 15 && stores[i]->pixel_offset == 4 * i;
      if (ctx->key->g17p_mrt_bg || store_count >= 4 || rgba32 || rgba8)
         return false;
   }

   if (load_count && !first_wait)
      return false;

   /* Native acquires and loads higher partitions first.  Move only the tile
    * loads themselves; their SSA consumers stay in place and RA extends the
    * values' physical lifetimes safely. */
   for (unsigned i = 0; i < load_count; ++i) {
      for (unsigned j = i + 1; j < load_count; ++j) {
         if (loads[j]->pixel_offset > loads[i]->pixel_offset) {
            agx_instr *tmp = loads[i];
            loads[i] = loads[j];
            loads[j] = tmp;
         }
      }
   }
   agx_instr *load_anchor = first_wait;
   for (unsigned i = 0; i < load_count; ++i) {
      list_del(&loads[i]->link);
      list_add(&loads[i]->link, &load_anchor->link);
      load_anchor = loads[i];
   }

   /* Sort physical tile partitions by byte offset, highest first. */
   for (unsigned i = 0; i < store_count; ++i) {
      for (unsigned j = i + 1; j < store_count; ++j) {
         if (stores[j]->pixel_offset > stores[i]->pixel_offset) {
            agx_instr *tmp = stores[i];
            stores[i] = stores[j];
            stores[j] = tmp;
         }
      }
   }
   if (stores[store_count - 1]->pixel_offset != 0 && !subset)
      return false;

   for (unsigned i = 0; i < store_count; ++i) {
      list_del(&stores[i]->link);
      list_addtail(&stores[i]->link, &stop->link);
   }
   ctx->agx3_mrt_tile_ops_ordered = true;
   return true;
}

/* Scalar epilog inputs retain their fixed export ABI, and unlike a collect a
 * direct PRELOAD cannot be assigned into a tile store's register bank by RA:
 * the output of colour target n arrives at GPR half 8 + 8n. The grounded
 * stores read halves 0..31 in store-only fragments of three and four targets
 * (and multi-target BG programs), halves 0..63 with more targets, and halves
 * 0..15 otherwise, so a store of the second target alone found its input out
 * of bank and was refused.
 * Materialize only out-of-bank scalar inputs, immediately before the first
 * store, and let RA choose a legal free register. The bank mirrors RA's
 * tile_gpr_limit.
 */
static bool
agx3_bank_scalar_tile_store_inputs(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble)
      return false;

   agx_instr *first_store = NULL;
   unsigned store_count = 0;
   bool loads = false;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_ST_TILE) {
         first_store = first_store ? first_store : I;
         store_count++;
      } else if (I->op == AGX_OPCODE_LD_TILE) {
         loads = true;
      }
   }
   if (!first_store)
      return false;

   const bool wide_bank =
      !loads && store_count <= AGX_MAX_RTS &&
      store_count >= (ctx->key->g17p_mrt_bg ? 2 : 3) &&
      (ctx->key->secondary || ctx->key->g17p_mrt_bg);
   const unsigned bank = !wide_bank ? 16 : store_count > 4 ? 64 : 32;

   bool progress = false;
   agx_builder b = agx_init_builder(ctx, agx_before_instr(first_store));
   agx_foreach_instr_global(ctx, store) {
      agx_index source = store->src[0];
      if (store->op != AGX_OPCODE_ST_TILE || source.type != AGX_INDEX_NORMAL ||
          agx_channels(source) != 1)
         continue;

      agx_foreach_instr_global(ctx, I) {
         if (I->op == AGX_OPCODE_PRELOAD && I->nr_dests == 1 &&
             agx_is_equiv(I->dest[0], source)) {
            if (I->src[0].type == AGX_INDEX_REGISTER &&
                I->src[0].value >= bank) {
               agx_index copy = agx_temp(ctx, source.size);
               agx_mov_to(&b, copy, source);
               store->src[0] = copy;
               progress = true;
            }
            break;
         }
      }
   }
   return progress;
}

static bool
agx3_implicit_coord_stage_producer(const agx_instr *I)
{
   if (I == NULL || I->nr_dests != 1 || I->dest[0].type != AGX_INDEX_NORMAL ||
       I->dest[0].size != AGX_SIZE_32 || agx_channels(I->dest[0]) != 1 ||
       I->saturate)
      return false;

   /* These are the two native terminal-coordinate forms needed by ordinary
    * smooth sampling and compositor UV+offset sampling. Other arithmetic
    * stays on the established GPR plus COPY path until its staging destination
    * form is grounded independently. */
   return I->op == AGX_OPCODE_PERSP_MUL_AGX3 ||
          (I->op == AGX_OPCODE_FADD && I->nr_srcs == 2 &&
           I->src[0].type == AGX_INDEX_NORMAL &&
           I->src[1].type == AGX_INDEX_NORMAL &&
           I->src[0].size == AGX_SIZE_32 &&
           I->src[1].size == AGX_SIZE_32 &&
           agx_channels(I->src[0]) == 1 &&
           agx_channels(I->src[1]) == 1 && !I->src[0].abs &&
           !I->src[0].neg && !I->src[1].abs && !I->src[1].neg);
}

/* Mark the last two scalar coordinate operations of a straight-line implicit
 * fragment sample for direct staging destinations. Apple does not calculate
 * the UVs in GPRs and copy them afterwards: the final perspective MUL or ADD
 * itself publishes staging[0]/[2]. That distinction preserves helper-lane
 * provenance at a misaligned hardware-scissor edge, which is required for the
 * implicit derivative quad and therefore the correct mip level.
 *
 * Run after scheduling so the start marker names the actual first producer,
 * but before RA while exact SSA def/use identity is still available. */
static bool
agx3_mark_implicit_tex_coord_staging(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3B ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble ||
       ctx->key->fs.ignore_tib_dependencies || list_length(&ctx->blocks) != 1)
      return false;

   agx_instr **defs = rzalloc_array(ctx, agx_instr *, ctx->alloc);
   unsigned *uses = rzalloc_array(ctx, unsigned, ctx->alloc);
   if (defs == NULL || uses == NULL)
      return false;

   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_src(I, s) {
         assert(I->src[s].value < ctx->alloc);
         uses[I->src[s].value]++;
      }
      agx_foreach_ssa_dest(I, d) {
         assert(I->dest[d].value < ctx->alloc);
         defs[I->dest[d].value] = I;
      }
   }

   bool progress = false;
   agx_foreach_instr_global(ctx, sample) {
      const bool plain_implicit =
         sample->op == AGX_OPCODE_TEXTURE_SAMPLE && sample->nr_srcs == 6 &&
         sample->lod_mode == AGX_LOD_MODE_AUTO_LOD &&
         sample->dim == AGX_DIM_2D && !sample->shadow && !sample->offset &&
         !sample->sparse &&
         sample->gather == AGX_GATHER_NONE;
      if (!plain_implicit || sample->src[0].type != AGX_INDEX_NORMAL ||
          sample->src[0].size != AGX_SIZE_32 ||
          agx_channels(sample->src[0]) != 2 ||
          uses[sample->src[0].value] != 1)
         continue;

      agx_instr *collect = defs[sample->src[0].value];
      if (collect == NULL || collect->op != AGX_OPCODE_COLLECT ||
          collect->nr_srcs != 2 || collect->nr_dests != 1)
         continue;

      agx_instr *producer[2] = {NULL, NULL};
      bool valid = true;
      for (unsigned c = 0; c < 2; ++c) {
         const agx_index scalar = collect->src[c];
         if (scalar.type != AGX_INDEX_NORMAL || scalar.size != AGX_SIZE_32 ||
             agx_channels(scalar) != 1 || uses[scalar.value] != 1) {
            valid = false;
            break;
         }

         producer[c] = defs[scalar.value];
         if (!agx3_implicit_coord_stage_producer(producer[c]) ||
             producer[c]->agx3_tex_coord_stage) {
            valid = false;
            break;
         }
      }

      if (!valid || producer[0] == producer[1])
         continue;

      /* Apple's G17P implicit fragment emits every coefficient interpolation
       * before it opens the coordinate-staging transaction, then publishes X
       * and Y consecutively. Keep an unrelated interpolation out of that
       * bounded staging transaction, exactly as in the native witness. Both
       * terminal producers are pure, their SSA results have the single
       * COLLECT use proven above, and sinking them to the COLLECT therefore
       * preserves data dependencies.
       */
      for (unsigned c = 0; c < 2; ++c) {
         list_del(&producer[c]->link);
         list_addtail(&producer[c]->link, &collect->link);
      }

      for (unsigned c = 0; c < 2; ++c) {
         producer[c]->agx3_tex_coord_stage = true;
         producer[c]->agx3_tex_coord_component = c;
      }
      producer[0]->agx3_tex_coord_start = true;
      sample->agx3_tex_coords_staged = true;
      progress = true;
      /* One synthetic staging token is modeled by the slot planner. A later
       * sample retains the existing immediate COPY path; broad overlapping
       * staging groups need distinct native lifetime evidence. */
      break;
   }

   return progress;
}

static bool
agx_compile_function_nir(nir_shader *nir, nir_function_impl *impl,
                         struct agx_shader_key *key,
                         struct util_dynarray *binary,
                         struct agx_shader_info *out, unsigned *offset_out)
{
   const unsigned initial_binary_size = binary->size;
   nir_index_blocks(impl);
   nir_index_ssa_defs(impl);

   agx_context *ctx = rzalloc(NULL, agx_context);
   ctx->nir = nir;
   ctx->is_preamble = impl->function->is_preamble;
   ctx->out = out;
   ctx->key = key;
   ctx->stage = nir->info.stage;
   ctx->allocated_vec = _mesa_hash_table_u64_create(ctx);
   ctx->indexed_nir_blocks = rzalloc_array(ctx, agx_block *, impl->num_blocks);
   list_inithead(&ctx->blocks);

   if (ctx->stage == MESA_SHADER_FRAGMENT && !ctx->is_preamble) {
      ctx->any_cf = key->fs.inside_sample_loop;

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            ctx->zs_writes_left +=
               instr->type == nir_instr_type_intrinsic &&
               nir_instr_as_intrinsic(instr)->intrinsic ==
                  nir_intrinsic_store_zs_agx;
         }
      }
   }

   ctx->alloc = impl->ssa_alloc;
   emit_cf_list(ctx, &impl->body);
   agx_emit_phis_deferred(ctx);

   /* Index blocks now that we're done emitting so the order is consistent. Do
    * this before agx_remove_unreachable_blocks so we match NIR indexing. This
    * makes for easier debugging.
    */
   agx_foreach_block(ctx, block) {
      block->index = ctx->num_blocks++;
   }

   agx_remove_unreachable_blocks(ctx);

   /* Only allocate scratch if it's statically used, regardless of if the NIR
    * info claims otherwise.
    */
   if (ctx->any_scratch) {
      assert(!ctx->is_preamble && "preambles don't use scratch");
      ctx->scratch_size_B = align(nir->scratch_size, 16);
   }

   /* Stop the main shader or preamble shader after the exit block. For real
    * functions, we would return here.
    */
   if (!ctx->key->no_stop || ctx->is_preamble) {
      agx_block *last_block = list_last_entry(&ctx->blocks, agx_block, link);
      agx_builder _b = agx_init_builder(ctx, agx_after_block(last_block));
      agx_stop(&_b);
   }

   agx_validate(ctx, "IR translation");

   if (likely(!(agx_compiler_debug & AGX_DBG_NOOPT))) {
      /* Eliminate dead instructions before CSE to avoid silly scheduling */
      AGX_PASS(ctx, agx_dce, false);

      /* CSE before eliminating dead destinations so that subdivision is
       * optimized properly.
       */
      AGX_PASS(ctx, agx_opt_cse);

      /* After DCE, use counts are right so we can run the optimizer. */
      AGX_PASS(ctx, agx_optimizer_backward);
      AGX_PASS(ctx, agx_optimizer_forward);
      AGX_PASS(ctx, agx_opt_compact_constants);

      /* After inlining constants, promote what's left */
      if (key->promote_constants && !key->secondary &&
          !(agx_compiler_debug & AGX_DBG_NOPROMOTE)) {
         AGX_PASS(ctx, agx_opt_promote_constants);
      }
   }

   agx_lower_agx3_fma_sources(ctx);
   agx_lower_agx3_mixed_width_transcendental(ctx);
   agx_lower_agx3_shifted_iadd_dest(ctx);

   if (key->dev.encoding != AGX_ISA_ENCODING_AGX2)
      agx_materialize_agx3_alu_constants(ctx);

   agx_materialize_agx3_cf_constants(ctx);

   /* For correctness, lower uniform sources after copyprop (for correctness,
    * as copyprop creates uniform sources). To keep register pressure in
    * check, lower after CSE, since moves are cheaper than registers.
    */
   AGX_PASS(ctx, agx_lower_uniform_sources);

   AGX_PASS(ctx, agx3_coalesce_uniform_half_stores);

   /* RA correctness depends on DCE */
   AGX_PASS(ctx, agx_dce, true);

   AGX_PASS(ctx, agx_lower_agx3_device_atomic_sources);

   if (agx_should_dump(nir, AGX_DBG_SHADERS))
      agx_print_shader(ctx, stdout);

   if (likely(!(agx_compiler_debug & AGX_DBG_NOSCHED))) {
      AGX_PASS(ctx, agx_pressure_schedule);
   }

   AGX_PASS(ctx, agx3_order_fragment_device_loads);

   if (agx_should_dump(nir, AGX_DBG_SHADERS))
      agx_print_shader(ctx, stdout);

   AGX_PASS(ctx, agx3_sink_vertex_outputs);
   AGX_PASS(ctx, agx3_order_g17p_mrt_tile_ops);
   AGX_PASS(ctx, agx3_bank_scalar_tile_store_inputs);
   AGX_PASS(ctx, agx3_mark_implicit_tex_coord_staging);

   /* A returned G17P device atomic publishes its value in TPS half 0 before
    * RLD moves it to the assigned GPR. If this shader also needs compiler
    * spills, reserve the complete 16-byte native atomic frame before RA lays
    * out ordinary spill cells. This keeps both mechanisms in one static TPS
    * frame without adding a memory barrier or serializing the kernel. */
   bool g17p_returned_atomic = false;
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       (ctx->stage == MESA_SHADER_COMPUTE ||
        ctx->stage == MESA_SHADER_FRAGMENT)) {
      agx_foreach_instr_global(ctx, I) {
         if (I->op == AGX_OPCODE_ATOMIC && I->nr_dests == 1 &&
             !agx_is_null(I->dest[0])) {
            g17p_returned_atomic = true;
            break;
         }
      }
   }
   if (g17p_returned_atomic && ctx->key->has_scratch)
      ctx->scratch_size_B = MAX2(ctx->scratch_size_B, 16);

   AGX_PASS(ctx, agx_ra);

   /* Apple's G17P compiler publishes a 16-byte scratch footprint for both
    * compute and fragment shaders that consume a device atomic's return.
    * The otherwise equivalent return-unused shaders publish zero, and the
    * returned form immediately reloads the old value from TPS half 0. Keep
    * this fixed ABI reservation distinct from allocator-created spill size;
    * the packer refuses a mixed returned-atomic/compiler-scratch shader until
    * ordinary scratch offsets are rebased around these words. */
   if (g17p_returned_atomic)
      out->agx3_spectpr_scratch_size =
         MAX2(out->agx3_spectpr_scratch_size, 16);

   agx_lower_64bit_postra(ctx);

   if (ctx->scratch_size_B > 0) {
      /* Apple always allocate 40 more bytes in the entrypoint and align to 4. */
      uint64_t stack_size = align(DIV_ROUND_UP(ctx->scratch_size_B, 4) + 10, 4);

      assert(stack_size < INT16_MAX);

      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2) {
         agx_block *start_block = agx_start_block(ctx);
         agx_builder _b =
            agx_init_builder(ctx, agx_before_block(start_block));
         agx_stack_adjust(&_b, stack_size);

         /* If we're going to execute multiple times, make sure we clean up
          * after ourselves, else the hardware faults. */
         if (ctx->stage == MESA_SHADER_FRAGMENT && !ctx->is_preamble &&
             ctx->key->fs.inside_sample_loop) {
            _b = agx_init_builder(ctx, agx_after_block(agx_end_block(ctx)));
            agx_stack_adjust(&_b, -stack_size);
         }
      }

      /* AGX3 static spills are copy/rld accesses to the launch-allocated TPS
       * frame. The pinned G15/G16/G17 compute spill corpus emits no incStk;
       * that distinct instruction belongs to the ray-traversal stack. Keep
       * publishing the frame size below, but never inject an AGX2 stack
       * instruction into an AGX3 shader.
       *
       * G17P publishes that frame in bytes: applegpu_g17p's metadata for
       * kernels whose highest reloaded TPS half is 470 and 790 records 944
       * and 1584 bytes, the halves up to and including that 32-bit word.
       * SPECTPR's spill fields are those bytes in quads. */
      if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
         stack_size = align(ctx->scratch_size_B, 4);

      if (ctx->is_preamble)
         out->preamble_scratch_size = stack_size;
      else
         out->scratch_size = stack_size;
   }

   if ((ctx->stage == MESA_SHADER_VERTEX ||
        ctx->stage == MESA_SHADER_TESS_EVAL) &&
       !impl->function->is_preamble && !ctx->key->secondary)
      agx_set_st_vary_final(ctx);

   /* AGX3 has no scoreboard `wait`; dependency fences ride on the terminal
    * `end`, and every real kernel ends with mask 0. Wait insertion is
    * therefore an AGX2-only pass. */
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2)
      agx_insert_waits(ctx);
   agx_opt_empty_else(ctx);
   agx_opt_break_if(ctx);
   agx_opt_jmp_none(ctx);

   if (ctx->any_quad_divergent_shuffle)
      agx_lower_divergent_shuffle(ctx);

   agx_lower_pseudo(ctx);

   /* AGX3's thread-private storage moves one 16-bit or 32-bit component per
    * instruction, so a vector spill or scratch access has to become a run of
    * scalar ones. Do it before the last-use pass, so each component's hints
    * are computed against the instructions that actually survive. */
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3B)
      agx3_split_vector_stack_access(ctx);

   /* Set last-use bits after lowering pseudo for best results. */
   agx_opt_register_cache(ctx);

   if (agx_should_dump(nir, AGX_DBG_SHADERS))
      agx_print_shader(ctx, stdout);

   /* Upload constants before the binary instead after to reduce the chance they
    * get prefetched into the i-cache, when we want them only in the d-cache.
    * Also helps fill the padding space for small preambles.
    */
   if (ctx->out->rodata.size_16 && !impl->function->is_preamble) {
      ctx->out->rodata.offset = agx_pad_binary(binary, 4);
      unsigned size_16 = ctx->out->rodata.size_16;

      uint16_t *ro = util_dynarray_grow(binary, uint16_t, size_16);
      memcpy(ro, ctx->rodata, size_16 * 2);
   }

   unsigned offset = agx_pad_binary(binary, AGX_CODE_ALIGN);
   if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2) {
      agx_pack_binary(ctx, binary);
   } else {
      /* AGX3 (M3/M4/M5). Device open still refuses these parts
       * (agx_chip_codegen_supported), so this runs only when a caller builds an
       * AGX3 device key deliberately. agx3_pack_binary fails the shader -- never
       * aborts -- on any ungrounded op and rolls the emitted shader prefix back
       * before returning false. */
      if (!agx3_pack_binary(ctx, binary)) {
         binary->size = initial_binary_size;
         ralloc_free(ctx);
         return false;
      }
   }

   unsigned nr_gprs = ctx->max_reg + 1;

   /* If the preamble uses scratch (due to spilling), we need to set maximal
    * GPRs. Do it here so the driver doesn't have to worry about it.
    */
   if (impl->function->is_preamble)
      out->nr_preamble_gprs = ctx->scratch_size_B ? 256 : nr_gprs;
   else {
      out->nr_gprs = nr_gprs;
      out->agx3_staging_halfs = ctx->agx3_staging_halfs;
      out->agx3_staging_known = ctx->agx3_staging_known;
      out->agx3_fltnorm_start = ctx->agx3_fltnorm_start;
      out->agx3_colour_link = ctx->agx3_colour_link;
   }

   if (impl->function->is_preamble) {
      uint32_t instrs = 0;
      agx_foreach_instr_global(ctx, I) {
         instrs++;
      }

      ctx->out->stats.preamble_inst = instrs;
   } else {
      agx_calc_stats(ctx, binary->size, &ctx->out->stats);
   }

   /*
    * Disassemble if the environment variable is set. Also disassemble to
    * /dev/null in debug builds as a smoke test - a legally encoded shader
    * should not hit any disassembler errors.
    */
   bool dump_shaders = agx_should_dump(nir, AGX_DBG_SHADERS);
#ifndef NDEBUG
   bool selftest = !dump_shaders;
#else
   bool selftest = false;
#endif

#ifdef _WIN32
   /* Windows' null device is NUL; /dev/null may fail to open and feeding a
    * NULL FILE to the disassembler would turn a debug self-test into an
    * access violation during shader compilation. */
   if (selftest) {
      FILE *probe = fopen("NUL", "w");
      if (!probe) {
         fprintf(stderr, "agx_compile: unable to open NUL; skipping "
                         "disassembler self-test\n");
         selftest = false;
      } else {
         fclose(probe);
      }
   }
#endif

   if ((dump_shaders || selftest) &&
       ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX2) {
#ifdef _WIN32
      FILE *fp = selftest ? fopen("NUL", "w") : stdout;
#else
      FILE *fp = selftest ? fopen("/dev/null", "w") : stdout;
#endif
      bool errors =
         agx2_disassemble(binary->data + offset, binary->size - offset, fp);

      if (errors) {
         /* In a self-test, disassemble again to get something the user can look
          * at to figure out what went wrong.
          */
         if (selftest) {
            agx2_disassemble(binary->data + offset, binary->size - offset,
                             stderr);
         }

         UNREACHABLE("Disassembly error hit.");
      }

      if (selftest) {
         fclose(fp);
      }
   }

   *offset_out = offset;
   ralloc_free(ctx);
   return true;
}

/*
 * The hardware frcp instruction is sometimes off by 1 ULP. For correctly
 * rounded frcp, a refinement step is required. This routine has been
 * exhaustively tested with a modified math_bruteforce.
 *
 * While Khronos APIs allow 2.5 ULP error for divides, nir_lower_idiv relies on
 * correctly rounded frcp. This is therefore load bearing for integer division
 * on all APIs.
 */
static nir_def *
libagx_frcp(nir_builder *b, nir_def *x)
{
   nir_def *u = nir_frcp(b, x);

   /* Do 1 Newton-Raphson refinement step.
    *
    * Define f(u) = xu - 1. Then f(u) = 0 iff u = 1/x. Newton's method gives:
    *
    * u_2 = u - f(u) / f'(u) = u - (xu - 1) / x
    *
    * Our original guess is close, so we approximate (1 / x) by u:
    *
    * u_2 = u - u(xu - 1) = u + u(1 - xu)
    *     = fma(fma(-x, u, 1), u, u)
    */
   nir_def *one = nir_imm_float(b, 1.0);
   nir_def *u_2 = nir_ffma(b, nir_ffma(b, nir_fneg(b, x), u, one), u, u);

   /* If the original value was infinite, frcp will generate the correct zero.
    * However, the Newton-Raphson step would multiply 0 * Inf and get a NaN. So
    * skip the refinement step for infinite inputs. We do this backwards,
    * checking whether the refined result is NaN, since we can implement this
    * check in a single fcmpsel instruction. The other case where the refinement
    * is NaN is a NaN input, in which skipping refinement is acceptable.
    */
   return nir_bcsel(b, nir_fisnan(b, u_2), u, u_2);
}

static bool
agx_nir_lower_fdiv(nir_builder *b, nir_alu_instr *alu, void *_)
{
   if (alu->op != nir_op_frcp || !nir_alu_instr_is_exact(alu) ||
       alu->def.bit_size != 32)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def_replace(&alu->def, libagx_frcp(b, nir_ssa_for_alu_src(b, alu, 0)));
   return true;
}

static bool
agx_nir_lower_g17p_f16_classify(nir_builder *b, nir_alu_instr *alu,
                                void *_data)
{
   if (alu->def.num_components != 1 ||
       nir_src_bit_size(alu->src[0].src) != 16)
      return false;

   const bool is_nan =
      alu->op == nir_op_fneu && nir_alu_srcs_equal(alu, alu, 0, 1);

   bool is_inf = false;
   nir_alu_instr *absolute = NULL;
   if (alu->op == nir_op_feq) {
      for (unsigned constant = 0; constant < 2; ++constant) {
         const unsigned value = constant ^ 1;
         nir_alu_instr *candidate = nir_src_as_alu(alu->src[value].src);
         const double compare = nir_src_is_const(alu->src[constant].src)
                                   ? nir_src_as_float(alu->src[constant].src)
                                   : 0.0;

         if (candidate && candidate->op == nir_op_fabs &&
             nir_src_bit_size(candidate->src[0].src) == 16 &&
             isinf(compare) && compare > 0.0) {
            is_inf = true;
            absolute = candidate;
         }
      }
   }

   if (!is_nan && !is_inf)
      return false;

   /* Classify the IEEE-754 binary16 encoding directly. Clearing the sign
    * leaves +infinity at 0x7c00 and every NaN strictly above it. This is exact
    * for quiet and signaling NaNs, both infinities, both zeros, subnormals,
    * and finite values. Ordinary half compares stay on their native path. */
   b->cursor = nir_before_instr(&alu->instr);
   nir_def *value = is_nan ? nir_ssa_for_alu_src(b, alu, 0)
                           : nir_ssa_for_alu_src(b, absolute, 0);
   nir_def *magnitude =
      nir_iand_imm(b, nir_u2u32(b, value), UINT32_C(0x7fff));
   nir_def *result = is_nan
                        ? nir_ult(b, nir_imm_int(b, 0x7c00), magnitude)
                        : nir_ieq_imm(b, magnitude, 0x7c00);
   nir_def_replace(&alu->def, result);
   return true;
}

/* Shared preprocessing with target-bounded subgroup choices. */
static void
agx_preprocess_nir_internal(nir_shader *nir, bool g17p)
{
   if (!nir)
      return;

   /* G17P has one native helper-capable fragment entry protocol: the first
    * coefficient load leaves its input-start marker clear and the linked
    * program disables triangle merging. Fresh applegpu_g17p derivatives and
    * quad shuffles use that same protocol and carry otherwise identical
    * ProgramVariant metadata.
    *
    * nir_shader_gather_info derives these flags from individual operations,
    * but the quad-control execution modes are independently meaningful. In
    * particular, RequireFullQuadsKHR promises a full quad even when no quad
    * intrinsic survives optimization, and QuadDerivativesKHR requires the
    * derivative group to be the full quad. Preserve those source-level modes
    * explicitly instead of making the hardware protocol depend on whichever
    * intrinsics happen to remain at this point in lowering. */
   if (g17p && nir->info.stage == MESA_SHADER_FRAGMENT &&
       (nir->info.fs.require_full_quads || nir->info.fs.quad_derivatives)) {
      nir->info.fs.needs_coarse_quad_helper_invocations = true;
      nir->info.fs.needs_full_quad_helper_invocations = true;
   }

   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   /* Lower large arrays to scratch and small arrays to csel */
   NIR_PASS(_, nir, nir_lower_vars_to_scratch, AGX_SCRATCH_ARRAY_THRESHOLD_B,
            glsl_get_natural_size_align_bytes, glsl_get_word_size_align_bytes);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
            nir_var_function_temp, ~0);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_var_copies);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, agx_nir_lower_frag_sidefx, g17p);
      NIR_PASS(_, nir, nir_lower_is_helper_invocation);
   }

   /* Clean up deref gunk after lowering I/O */
   NIR_PASS(_, nir, nir_opt_dce);

   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   if (g17p) {
      NIR_PASS(_, nir, nir_shader_alu_pass,
               agx_nir_lower_g17p_f16_classify,
               nir_metadata_control_flow, NULL);
   }
   NIR_PASS(_, nir, nir_lower_frexp);
   NIR_PASS(_, nir, nir_lower_alu);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_lower_flrp, 16 | 32 | 64, false);
   NIR_PASS(_, nir, agx_lower_sincos, g17p);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, agx_lower_front_face,
            nir_metadata_control_flow, NULL);
   if (g17p)
      NIR_PASS(_, nir, agx_nir_lower_subgroups_g17p);
   else
      NIR_PASS(_, nir, agx_nir_lower_subgroups);
   NIR_PASS(_, nir, nir_lower_all_phis_to_scalar);

   /* After lowering, run through the standard suite of NIR optimizations. We
    * will run through the loop later, once we have the shader key, but if we
    * run now, that run will ideally be almost a no-op.
    */
   agx_optimize_loop_nir(nir);

   /* Lower idiv after an optimization loop so we can constant fold more before
    * nir_opt_idiv_const.
    */
   NIR_PASS(_, nir, nir_opt_idiv_const, 16);

   nir_lower_idiv_options idiv_options = {
      .allow_fp16 = true,
   };

   NIR_PASS(_, nir, nir_lower_idiv, &idiv_options);

   /* Has to run after nir_lower_idiv */
   NIR_PASS(_, nir, nir_shader_alu_pass, agx_nir_lower_fdiv,
            nir_metadata_control_flow, NULL);

   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   /* We're lowered away all variables. Remove them all for smaller shaders. */
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_all, NULL);
   nir->info.io_lowered = true;

   /* Move before lowering */
   nir_move_options move_all = nir_move_const_undef | nir_move_load_ubo |
                               nir_move_load_input | nir_move_load_frag_coord |
                               nir_move_comparisons | nir_move_copies |
                               nir_move_load_ssbo | nir_move_alu;

   NIR_PASS(_, nir, nir_opt_sink, move_all);
   NIR_PASS(_, nir, nir_opt_move, move_all);
   /* Local-memory offsets are narrowed before the key-dependent optimizer so
    * conversion lowering and cleanup remain part of the normal pipeline.
    * G17P ordinary loads/stores use scalar-element indices; atomics retain
    * their byte-addressed encoding.
    */
   NIR_PASS(_, nir, agx_nir_lower_shared_bitsize, g17p);
}

void
agx_preprocess_nir(nir_shader *nir)
{
   agx_preprocess_nir_internal(nir, false);
}

void
agx_preprocess_nir_g17p(nir_shader *nir)
{
   agx_preprocess_nir_internal(nir, true);
}

static bool
lower_printf_buffer(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct agx_device_key *dev = data;
   uint64_t value = 0;
   if (intr->intrinsic == nir_intrinsic_load_printf_buffer_address) {
      /* AGX3 frontends must publish the process-local printf address through
       * runtime state.  Baking it here would poison serialized shader caches;
       * the preflight below rejects an address intrinsic that escaped that
       * lowering.  AGX2 retains its established fixed-address ABI. */
      if (dev->encoding != AGX_ISA_ENCODING_AGX2)
         return false;
      value = LIBAGX_PRINTF_BUFFER_ADDRESS;
   } else if (intr->intrinsic == nir_intrinsic_load_printf_buffer_size) {
      value = LIBAGX_PRINTF_BUFFER_SIZE;
   } else {
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, value, intr->def.bit_size));
   return true;
}

/* A fragment shader that indexes a varying array at run time reads it through
 * one load_interpolated_input whose offset is a value rather than a constant.
 * That needs the interpolation to take its coefficient index from a register.
 * AGX2 encodes exactly that -- `iter r4_r5_r6_r7, r0l, cf0` -- but no AGX3B
 * form for it is witnessed, and Metal cannot express a dynamically indexed
 * interpolant, so metal-nt will not produce one either. Refusing the shader
 * is what KHR-GL33.shaders.indexing.varying_array's four *_dynamic_read cases
 * hit.
 *
 * Read every slot the array spans and select between them. This runs in the
 * compile rather than the preprocess because IO is lowered by then: the
 * indirection is an offset source, not a deref, so
 * nir_lower_indirect_derefs_to_if_else_trees cannot see it.
 *
 * It costs one interpolation per array element where AGX2 needs one in total.
 * That is an efficiency gap against the baseline, recorded as such -- not a
 * correctness one, and the alternative is refusing to compile at all.
 */
static bool
lower_indirect_fs_input(nir_builder *b, nir_intrinsic_instr *intr,
                        UNUSED void *data)
{
   unsigned offset_src;
   if (intr->intrinsic == nir_intrinsic_load_interpolated_input)
      offset_src = 1;
   else if (intr->intrinsic == nir_intrinsic_load_input)
      offset_src = 0;
   else
      return false;

   if (nir_src_is_const(intr->src[offset_src]))
      return false;

   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   const unsigned slots = sem.num_slots;
   if (slots <= 1 || slots > 32)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *index = intr->src[offset_src].ssa;
   nir_def *result = NULL;

   for (unsigned i = 0; i < slots; ++i) {
      nir_io_semantics one_sem = sem;
      one_sem.num_slots = 1;
      one_sem.location = sem.location + i;

      nir_def *one;
      if (intr->intrinsic == nir_intrinsic_load_interpolated_input) {
         one = nir_load_interpolated_input(
            b, intr->def.num_components, intr->def.bit_size,
            intr->src[0].ssa, nir_imm_int(b, 0),
            .base = nir_intrinsic_base(intr) + i,
            .component = nir_intrinsic_component(intr),
            .dest_type = nir_intrinsic_dest_type(intr),
            .io_semantics = one_sem);
      } else {
         one = nir_load_input(
            b, intr->def.num_components, intr->def.bit_size,
            nir_imm_int(b, 0), .base = nir_intrinsic_base(intr) + i,
            .component = nir_intrinsic_component(intr),
            .dest_type = nir_intrinsic_dest_type(intr),
            .io_semantics = one_sem);
      }

      result = result == NULL
                  ? one
                  : nir_bcsel(b, nir_ieq_imm(b, index, i), one, result);
   }

   nir_def_replace(&intr->def, result);
   return true;
}

static bool
has_unlowered_printf_address(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_load_printf_buffer_address)
               return true;
         }
      }
   }

   return false;
}

bool
agx_compile_shader_nir(nir_shader *nir, struct agx_shader_key *key,
                       struct agx_shader_part *out)
{
   agx_compiler_debug = agx_get_compiler_debug();
   struct agx_shader_info *info = &out->info;

   struct util_dynarray binary = UTIL_DYNARRAY_INIT;

   memset(out, 0, sizeof *out);

   /* Record API/source arithmetic before backend lowering introduces its own
    * 64-bit address and invocation-ID calculations. */
   const bool source_iadd64 = agx_nir_has_alu64(nir, nir_op_iadd);
   const bool source_imul64 = agx_nir_has_alu64(nir, nir_op_imul);

   if ((key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        key->dev.encoding == AGX_ISA_ENCODING_AGX3C) &&
       has_unlowered_printf_address(nir))
      return false;

   assert(nir->info.io_lowered &&
          "agx_preprocess_nir is called first, then the shader is specalized,"
          "then the specialized shader is compiled");

   /* The AGX3 backend has byte-grounded traversal/SCS codecs, but no complete
    * NIR ray-query lowering yet.  Reject before instruction selection reaches
    * its fatal unhandled-intrinsic path, and keep the output transactional.
    * This gate is removed only together with the full lowering and scheduling
    * contract; it does not prevent direct internal encoder tests. */
   if (nir->info.ray_queries != 0 &&
       (key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
        key->dev.encoding == AGX_ISA_ENCODING_AGX3C))
      return false;

   /* If required, tag writes will be enabled by instruction selection */
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      info->tag_write_disable = !nir->info.writes_memory;

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_printf_buffer,
            nir_metadata_control_flow, &key->dev);

   if (key->dev.encoding == AGX_ISA_ENCODING_AGX3B &&
       nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_indirect_fs_input,
               nir_metadata_none, NULL);
   }

   NIR_PASS(_, nir, nir_lower_frag_coord_to_pixel_coord);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);

   /* Late sysval lowering creates large loads. Load lowering creates unpacks */
   nir_lower_mem_access_bit_sizes_options lower_mem_access_options = {
      .modes = nir_var_mem_ssbo | nir_var_mem_constant |
               nir_var_mem_task_payload | nir_var_shader_temp |
               nir_var_function_temp | nir_var_mem_global | nir_var_mem_shared,
      .callback = mem_access_size_align_cb,
   };
   NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &lower_mem_access_options);

   /* Optimize scratch access for silly internal CL shaders */
   if (nir->info.internal) {
      NIR_PASS(_, nir, nir_lower_scratch_to_var);
      NIR_PASS(_, nir, nir_lower_vars_to_scratch, 256,
               glsl_get_natural_size_align_bytes,
               glsl_get_natural_size_align_bytes);
      NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
               nir_var_function_temp, ~0);
   }

   /* Cleanup 8-bit math before lowering */
   bool progress;
   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);

   NIR_PASS(_, nir, nir_lower_bit_size, lower_bit_size_callback, NULL);

   /* Late blend lowering creates vectors */
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);

   /* Late VBO lowering creates constant udiv instructions */
   NIR_PASS(_, nir, nir_opt_idiv_const, 16);

   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_load_from_texture_handle,
            nir_metadata_control_flow, NULL);

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_sink_address,
            nir_metadata_control_flow, NULL);

   info->push_count = key->reserved_preamble;
   const bool form_preamble = !key->secondary && !key->no_preamble;
   const bool is_agx3 = key->dev.encoding == AGX_ISA_ENCODING_AGX3B ||
                       key->dev.encoding == AGX_ISA_ENCODING_AGX3C;
   const bool is_g17p = key->dev.encoding == AGX_ISA_ENCODING_AGX3B;

   /* AGX2 has native 64-bit add and 32x32->64 multiply forms, so the shared
    * compiler options preserve them.  G17P has no grounded 64-bit add, so
    * lower it to low/add, carry, high/add.
    *
    * G17P's 32x32->64 multiply IS grounded (agx3_iadd_core.h, dest_64), but a
    * full 64x64->64 multiply still needs several of them plus the carry
    * arithmetic, which is what agx_nir_lower_imul64 builds out of 32-bit
    * pieces.  Keep that lowering; the only cost of leaving it alone is that
    * its inner high-word multiply is expanded in software rather than using
    * one wide imad.  Changing that is a codegen optimisation and belongs with
    * a hardware measurement, not with enabling the instruction.
    *
    * Keep the override local to this compile so the other encodings retain
    * their paths and frontends continue sharing agx_nir_options. */
   const nir_shader_compiler_options *original_options = nir->options;
   nir_shader_compiler_options agx3_options;
   if (is_g17p) {
      agx3_options = *original_options;
      agx3_options.lower_iabs = false;
      /* G17P's exact saturate form is the witnessed standalone f32add.ep2
       * operation. Preserve fsat so isel emits that form; the backend
       * optimizer keeps it separate from producers with no saturation bit. */
      agx3_options.lower_fsat = false;
      /* The G17P integer encoder has no witnessed saturating-subtract form.
       * Lower it to the canonical unsigned compare, subtract, and select;
       * each resulting operation already has an Apple-exact encoding.  XFB
       * uses this for bounds-safe counter arithmetic. */
      agx3_options.lower_usub_sat = true;
      if (source_iadd64)
         agx3_options.lower_int64_options |= nir_lower_iadd64;
      if (source_imul64)
         agx3_options.lower_int64_options &= ~nir_lower_imul64;
      nir->options = &agx3_options;

      /* Lower the 64-bit multiply BEFORE agx_optimize_nir, not only after.
       *
       * Clearing nir_lower_imul64 above leaves a 64-bit imul standing all the
       * way through agx_optimize_nir, and agx_nir_fuse_algebraic_late's
       * strength-reduction rule
       *
       *    (('imul', a, 1 + (1 << s)), iaddshl(a, a, s))
       *
       * is not width-qualified, so a 64-bit `x * 3` is rewritten into a
       * 64-bit imadshl_agx.  agx_emit_imadshl_agx() then emits a 64-bit iadd,
       * which G17P has no grounded encoding for, and the whole shader is
       * refused:
       *
       *    no grounded AGX3 encoding for iadd
       *      [d64:reg4 s0=64:reg4.discard s1=64:reg4.discard]
       *
       * That refusal is reached by ordinary shaderInt64 code -- one 64-bit
       * multiply followed by a 64-bit add -- while shaderInt64 is advertised.
       * Lowering here leaves only 32-bit multiplies and adds plus
       * pack_64_2x32_split, so no rule in the late fusion can form a 64-bit
       * add out of it, and the expansion additionally gets optimized by the
       * passes that follow rather than being emitted raw.
       *
       * The existing call after agx_optimize_nir stays: this pass is
       * idempotent, and it still covers a 64-bit multiply created during
       * optimization rather than present in the source. */
      if (source_imul64)
         NIR_PASS(_, nir, agx_nir_lower_imul64);
   }

   agx_optimize_nir(
      nir, key->dev.soft_fault, !is_agx3,
      key->dev.encoding == AGX_ISA_ENCODING_AGX3B,
      form_preamble ? &info->push_count : NULL,
      (form_preamble && key->promote_textures) ? &info->texture_state_count
                                              : NULL,
      (form_preamble && key->promote_textures) ? &info->sampler_state_count
                                              : NULL);

   if (is_g17p && source_imul64)
      NIR_PASS(_, nir, agx_nir_lower_imul64);

   if (is_g17p)
      nir->options = original_options;

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      info->varyings.fs.nr_cf = key->fs.cf_base;
      assign_coefficient_regs(nir, &info->varyings.fs);
   }

   if (agx_should_dump(nir, AGX_DBG_SHADERS))
      nir_print_shader(nir, stdout);

   info->local_size = nir->info.shared_size;
   info->uses_resource_info_query = nir->info.uses_resource_info_query;
   info->uses_ray_traversal = nir->info.ray_queries != 0;
   /* NIR ray_queries counts inline intersection_query variables, matching the
    * G17S next()/committed witness whose ShaderInfo ID32 is nonzero.  Future
    * one-shot/full traversal lowerings must set uses_ray_traversal without
    * automatically setting this committed-path property. */
   info->ray_traversal_depth_nonzero = nir->info.ray_queries != 0;

   nir_foreach_function_with_impl(func, impl, nir) {
      unsigned offset;
      if (!agx_compile_function_nir(nir, impl, key, &binary, &out->info,
                                    &offset)) {
         util_dynarray_fini(&binary);
         memset(out, 0, sizeof(*out));
         return false;
      }

      if (func->is_preamble) {
         info->preamble_offset = offset;
         info->preamble_size = binary.size - offset;
         info->has_preamble = true;
      } else if (func->is_entrypoint) {
         info->main_offset = offset;
         info->main_size = binary.size - offset;
      } else {
         UNREACHABLE("General functions not yet supported");
      }
   }

   if (agx_should_dump(nir, AGX_DBG_SHADERDB)) {
      agx2_stats_fprintf(stderr, _mesa_shader_stage_to_abbrev(nir->info.stage),
                         &out->info.stats);
   }

   info->stage = nir->info.stage;

   /* Check these outside the stage check since nir->info.stage is the hardware
    * stage and these are read in the vertex *software* stage.
    */
   info->uses_draw_id =
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_DRAW_ID);

   info->uses_base_param =
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_FIRST_VERTEX) ||
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_BASE_VERTEX) ||
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_BASE_INSTANCE);

   if (nir->info.stage == MESA_SHADER_VERTEX ||
       nir->info.stage == MESA_SHADER_TESS_EVAL) {
      info->nonzero_viewport = nir->info.outputs_written & VARYING_BIT_VIEWPORT;

      info->writes_layer_viewport =
         nir->info.outputs_written & (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT);

   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      info->disable_tri_merging =
         nir->info.uses_wide_subgroup_intrinsics ||
         nir->info.fs.needs_coarse_quad_helper_invocations ||
         nir->info.writes_memory;

      /* Writing the sample mask requires tag writes */
      info->tag_write_disable &= !info->writes_sample_mask;

      /* Report a canonical depth layout. This happens at the end because the
       * sample mask lowering affects it.
       */
      enum gl_frag_depth_layout layout = nir->info.fs.depth_layout;

      if (!(nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH)))
         info->depth_layout = FRAG_DEPTH_LAYOUT_UNCHANGED;
      else if (layout == FRAG_DEPTH_LAYOUT_NONE)
         info->depth_layout = FRAG_DEPTH_LAYOUT_ANY;
      else
         info->depth_layout = layout;

      info->reads_tib = nir->info.fs.uses_fbfetch_output;
      info->early_fragment_tests = nir->info.fs.early_fragment_tests;
   } else if (nir->info.stage == MESA_SHADER_COMPUTE) {
      info->imageblock_stride = nir->info.cs.image_block_size_per_thread_agx;

      for (unsigned i = 0; i < 3; ++i) {
         info->workgroup_size[i] = nir->info.workgroup_size[i];
      }
   }

   out->binary = binary.data;
   info->binary_size = binary.size;
   return true;
}
