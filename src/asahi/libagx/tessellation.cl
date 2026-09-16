/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "poly/geometry.h"
#include "poly/tessellator.h"

KERNEL(1)
libagx_tess_setup_indirect(
   global struct poly_tess_params *p,
   global uint32_t *grids /* output: VS then TCS then tess */,
   global struct poly_vertex_params *vp /* output */, global uint32_t *indirect,
   uint64_t in_index_buffer,
   uint32_t in_index_buffer_range_el, uint32_t in_index_size_B,
   uint64_t vertex_outputs /* bitfield */,

   /* Tess control invocation counter if active, else zero */
   global uint32_t *tcs_statistic)
{
   uint count = indirect[0], instance_count = indirect[1];
   unsigned in_patches = count / p->input_patch_size;

   /* TCS invocation counter increments once per-patch */
   if (tcs_statistic) {
      *tcs_statistic += in_patches;
   }

   size_t draw_stride = 5 * sizeof(uint32_t);
   unsigned unrolled_patches = in_patches * instance_count;

   uint32_t alloc = 0;
   uint32_t tcs_out_offs = alloc;
   alloc += unrolled_patches * p->tcs_stride_el * 4;

   uint32_t patch_coord_offs = alloc;
   alloc += unrolled_patches * 4;

   uint32_t count_offs = alloc;
   alloc += unrolled_patches * sizeof(uint32_t);

   uint vb_offs = alloc;
   uint vb_size = poly_tcs_in_size(count * instance_count, vertex_outputs);
   alloc += vb_size;

   /* Allocate all patch calculations in one go */
   global uchar *blob = poly_heap_alloc(p->heap, alloc);

   p->tcs_buffer = (global float *)(blob + tcs_out_offs);
   p->patches_per_instance = in_patches;
   p->coord_allocs = (global uint *)(blob + patch_coord_offs);
   p->nr_patches = unrolled_patches;

   vp->output_buffer = (uintptr_t)(blob + vb_offs);
   vp->outputs = vertex_outputs;
   p->counts = (global uint32_t *)(blob + count_offs);

   if (vp) {
      vp->verts_per_instance = count;
   }

   /* If indexing is enabled, the third word is the offset into the index buffer
    * in elements. Apply that offset now that we have it. For a hardware
    * indirect draw, the hardware would do this for us, but for software input
    * assembly we need to do it ourselves.
    *
    * XXX: Deduplicate?
    */
   if (in_index_size_B) {
      vp->index_buffer =
         poly_index_buffer(in_index_buffer, in_index_buffer_range_el,
                           indirect[2], in_index_size_B);

      vp->index_buffer_range_el =
         poly_index_buffer_range_el(in_index_buffer_range_el, indirect[2]);
   }

   /* VS grid size */
   grids[0] = count;
   grids[1] = instance_count;
   grids[2] = 1;

   /* VS workgroup size */
   grids[3] = 64;
   grids[4] = 1;
   grids[5] = 1;

   /* TCS grid size */
   grids[6] = in_patches * p->output_patch_size;
   grids[7] = instance_count;
   grids[8] = 1;

   /* TCS workgroup size */
   grids[9] = p->output_patch_size;
   grids[10] = 1;
   grids[11] = 1;

   /* Tess grid size */
   grids[12] = unrolled_patches;
   grids[13] = 1;
   grids[14] = 1;

   /* Tess workgroup size */
   grids[15] = 64;
   grids[16] = 1;
   grids[17] = 1;
}

/*
 * An IEEE half with the value of a float tessellation level, rounded to
 * nearest. Anything not positive, or too small for a normal half, becomes the
 * smallest positive half, since a zero culls the patch; an overflow saturates
 * to the largest finite half. The tessellator clamps both to its own range.
 */
static inline uint16_t
libagx_tess_level_half(float level, uint32_t bits)
{
   const uint32_t mantissa = bits & 0x7fffff;
   const int exponent = (int)((bits >> 23) & 0xff) - 127 + 15;

   if (!(level > 0.0f))
      return 1;
   if (exponent >= 31)
      return 0x7bff;
   if (exponent <= 0)
      return 1;

   uint32_t value = ((uint32_t)exponent << 10) | (mantissa >> 13);
   if (mantissa & (1u << 12))
      value++;

   return min(value, (uint32_t)0x7bff);
}

/*
 * Set up an indirect draw for a hardware tessellator, as
 * libagx_tess_setup_indirect() does for the software one.
 *
 * There is no software tessellator output to allocate. Instead the kernel
 * places each patch's factor record, which the tessellator reads from the
 * factor buffer by patch index: the heap is that buffer, so the draw starts
 * at the patch index of this draw's first record. The heap has no way to
 * report an overflow here, so a draw that does not fit draws nothing.
 */
KERNEL(1)
libagx_tess_setup_indirect_hw(
   global struct poly_tess_params *p,
   global uint32_t *grids /* output: VS then TCS then factors */,
   global struct poly_vertex_params *vp /* output */, global uint32_t *indirect,
   global uint32_t *patch_draw /* output: patch count, instance count,
                                  patch start, base instance */,
   uint64_t in_index_buffer, uint32_t in_index_buffer_range_el,
   uint32_t in_index_size_B, uint64_t vertex_outputs /* bitfield */,
   uint32_t factor_record_B,

   /* Tess control invocation counter if active, else zero */
   global uint32_t *tcs_statistic)
{
   global struct poly_heap *heap = p->heap;
   uint count = indirect[0], instance_count = indirect[1];
   uint in_patches = count / p->input_patch_size;
   uint unrolled_patches = in_patches * instance_count;

   uint64_t tcs_out_B = (uint64_t)unrolled_patches * p->tcs_stride_el * 4;
   uint64_t vb_B = poly_tcs_in_size(count * instance_count, vertex_outputs);

   /* One spare record lets the records start on a record boundary. */
   uint64_t factors_B = ((uint64_t)unrolled_patches + 1) * factor_record_B;
   uint64_t size_B = (tcs_out_B + vb_B + factors_B + 15) & ~(uint64_t)15;

   uint offs = 0;
   bool allocated = size_B <= heap->size;
   if (allocated) {
      offs = atomic_fetch_add((volatile global atomic_uint *)(&heap->bottom),
                              (uint)size_B);
      allocated = poly_heap_allocation_fits(offs, size_B, heap->size);
   }

   if (!allocated) {
      count = instance_count = in_patches = unrolled_patches = 0;
      tcs_out_B = vb_B = 0;
   }

   /* TCS invocation counter increments once per-patch */
   if (tcs_statistic) {
      *tcs_statistic += in_patches;
   }

   global uchar *blob = heap->base + offs;
   uint factor_offs = offs + tcs_out_B + vb_B;

   p->tcs_buffer = (global float *)blob;
   p->patches_per_instance = in_patches;
   p->nr_patches = unrolled_patches;
   p->patch_start = (factor_offs + factor_record_B - 1) / factor_record_B;

   vp->output_buffer = (uintptr_t)(blob + tcs_out_B);
   vp->outputs = vertex_outputs;
   vp->verts_per_instance = count;

   if (in_index_size_B) {
      vp->index_buffer =
         poly_index_buffer(in_index_buffer, in_index_buffer_range_el,
                           indirect[2], in_index_size_B);

      vp->index_buffer_range_el =
         poly_index_buffer_range_el(in_index_buffer_range_el, indirect[2]);
   }

   /* VS grid size and workgroup size */
   grids[0] = count;
   grids[1] = instance_count;
   grids[2] = 1;
   grids[3] = 64;
   grids[4] = 1;
   grids[5] = 1;

   /* TCS grid size and workgroup size */
   grids[6] = in_patches * p->output_patch_size;
   grids[7] = instance_count;
   grids[8] = 1;
   grids[9] = p->output_patch_size;
   grids[10] = 1;
   grids[11] = 1;

   /* Factor grid size and libagx_tess_factors_half's workgroup size */
   grids[12] = unrolled_patches;
   grids[13] = 1;
   grids[14] = 1;
   grids[15] = 32;
   grids[16] = 1;
   grids[17] = 1;

   /* The tessellator draws every instance's patches as one run, in the order
    * the TCS dispatch unrolled them. */
   patch_draw[0] = unrolled_patches;
   patch_draw[1] = 1;
   patch_draw[2] = p->patch_start;
   patch_draw[3] = 0;
}

/*
 * Fill the per-patch factor records of a hardware tessellator from the levels
 * a tessellation control shader wrote.
 *
 * Each patch's levels sit at the start of its TCS output record: outer levels
 * at elements 0-3 and inner levels at 4-5. The tessellator reads a triangle
 * patch as {edge[3], inside} and a quad patch as {edge[4], inside[2]}, all
 * IEEE halves, and edge i subdivides the same edge as outer level i. Vulkan
 * discards a patch any of whose used outer levels is not positive; the
 * tessellator culls a patch whose edge factors are zero.
 *
 * The tessellator names a record by its patch index, which counts from
 * patch_start.
 */
KERNEL(32)
libagx_tess_factors_half(constant struct poly_tess_params *p,
                         global uint16_t *factors, uint32_t quads)
{
   uint patch = cl_global_id.x;
   global float *levels = p->tcs_buffer + (patch * p->tcs_stride_el);
   global uint32_t *bits = (global uint32_t *)levels;
   const uint edges = quads ? 4 : 3;
   const uint insides = quads ? 2 : 1;
   global uint16_t *out =
      factors + ((p->patch_start + patch) * (edges + insides));

   bool culled = false;
   for (uint i = 0; i < edges; ++i)
      culled |= !(levels[i] > 0.0f);

   for (uint i = 0; i < edges; ++i)
      out[i] = culled ? 0 : libagx_tess_level_half(levels[i], bits[i]);

   for (uint i = 0; i < insides; ++i) {
      out[edges + i] =
         culled ? 0 : libagx_tess_level_half(levels[4 + i], bits[4 + i]);
   }
}
