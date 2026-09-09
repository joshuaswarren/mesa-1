/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "asahi/lib/agx_abi.h"
#include "compiler/libcl/libcl_vk.h"
#include "poly/geometry.h"
#include "agx_pack.h"
#include "d3d12_sparse.h"
#include "libagx_dgc.h"

/*
 * To implement drawIndirectCount generically, we dispatch a kernel to
 * clone-and-patch the indirect buffer, predicating out draws as appropriate.
 */
KERNEL(32)
libagx_predicate_indirect(global uint32_t *out, constant uint32_t *in,
                          constant uint32_t *draw_count, uint32_t stride_el,
                          uint indexed__2)
{
   uint draw = cl_global_id.x;
   uint words = indexed__2 ? 5 : 4;
   bool enabled = draw < *draw_count;
   out += draw * words;
   in += draw * stride_el;

   /* Copy enabled draws, zero predicated draws. */
   for (uint i = 0; i < words; ++i) {
      out[i] = enabled ? in[i] : 0;
   }
}

/*
 * VK_EXT_conditional_rendering predicates both graphics and compute commands
 * with one 32-bit value.  Honeykrisp lowers each affected command to an
 * indirect argument record and runs this kernel immediately before the
 * consumer.  This is deliberately separate from predicate_indirect: Vulkan
 * requires a 32-bit load (D3D12 uses 64-bit predication), permits inversion,
 * and compute dispatch records contain three rather than four/five words.
 */
KERNEL(1)
libagx_conditional_render(global uint32_t *out, constant uint32_t *in,
                          constant uint32_t *predicate, uint32_t words,
                          uint32_t inverted)
{
   bool execute = ((*predicate != 0) != (inverted != 0));

   for (uint i = 0; i < words; ++i)
      out[i] = execute ? in[i] : 0;
}

/*
 * D3D12 ExecuteIndirect may source both its terminal draw record and root
 * constants from buffers written earlier on the GPU. Patch one bounded
 * argument slice in the pre-graphics stream so the draw observes those
 * writes, and predicate it against the execution-time count buffer.
 */
KERNEL(1)
libagx_d3d12_indirect_patch(global uint8_t *out, constant uint8_t *in,
                            constant uint32_t *draw_count,
                            uint32_t command_index, uint32_t size_B)
{
   bool enabled = !draw_count || command_index < *draw_count;

   for (uint i = 0; i < size_B; ++i)
      out[i] = enabled ? in[i] : 0;
}

/* Root constants inherit values from the preceding executed command.  The
 * destination is seeded with that state before this kernel runs, so a record
 * rejected by the count buffer must leave it untouched rather than zero it. */
KERNEL(1)
libagx_d3d12_indirect_constant(global uint8_t *out,
                               constant uint8_t *previous,
                               constant uint8_t *in,
                               constant uint32_t *draw_count,
                               uint32_t command_index, uint32_t total_size_B,
                               uint32_t destination_offset_B,
                               uint32_t size_B)
{
   if (previous) {
      for (uint i = 0; i < total_size_B; ++i)
         out[i] = previous[i];
   }

   if (!draw_count || command_index < *draw_count) {
      for (uint i = 0; i < size_B; ++i)
         out[destination_offset_B + i] = in[i];
   }
}

/*
 * Patch one root CBV/SRV/UAV address into the compact D3D12 root table.
 *
 * The argument record is only guaranteed to be DWORD aligned, so load the
 * 64-bit GPU VA as two words.  A zero address keeps the safe null-buffer
 * address already present in the table.  Bare root SRV/UAV descriptors have
 * no declared extent; publish UINT32_MAX in BUFINFO for a live address, just
 * like the direct root-descriptor path does.
 *
 * Reserved buffers have a read-only shadow mapping on AGX.  The CPU direct
 * setter resolves that mapping from the device's sparse-resource registry;
 * ExecuteIndirect cannot inspect the GPU-produced VA on the CPU, so it gets
 * the same registry as a compact base/size array and performs the lookup here.
 */
KERNEL(1)
libagx_d3d12_indirect_root(global uint64_t *out_address,
                           global uint64_t *out_bufinfo,
                           global uint64_t *out_residency,
                           constant uint32_t *in,
                           constant uint32_t *draw_count,
                           constant uint64_t *previous_address,
                           constant uint64_t *previous_bufinfo,
                           constant uint64_t *previous_residency,
                           constant struct
                              libagx_d3d12_sparse_registry_view
                                 *sparse_registry,
                           uint64_t null_address,
                           uint64_t sparse_ro_offset,
                           uint32_t command_index,
                           uint32_t publish_unbounded,
                           uint32_t translate_read)
{
   bool enabled = !draw_count || command_index < *draw_count;
   uint64_t original_address = *out_address;
   uint64_t original_bufinfo = *out_bufinfo;
   uint64_t original_residency = *out_residency;
   if (!enabled) {
      *out_address = previous_address ? *previous_address : original_address;
      *out_bufinfo = previous_bufinfo ? *previous_bufinfo : original_bufinfo;
      *out_residency = previous_residency ? *previous_residency
                                          : original_residency;
      return;
   }

   uint64_t address = ((uint64_t)in[1] << 32) | in[0];
   uint64_t residency = 0;

   uint64_t translated = libagx_d3d12_sparse_read_address(
      address, sparse_registry, sparse_ro_offset, &residency);
   if (translate_read)
      address = translated;

   uint64_t metadata = original_bufinfo & 0xffff000000000000ul;
   *out_address = address ? address : null_address;
   *out_bufinfo = metadata |
                  ((address && publish_unbounded) ? UINT32_MAX : 0);
   *out_residency = residency;
}

/* Patch one PSO attribute fed by an indirect vertex-buffer view.  Each
 * attribute gets its own compact output triple because D3D12 supplies one
 * view per IA slot while AGX folds the input-layout byte offset into a
 * per-attribute base. */
KERNEL(1)
libagx_d3d12_indirect_vbv(global uint64_t *out_base,
                          global uint32_t *out_clamp,
                          global uint32_t *out_stride,
                          constant uint32_t *in,
                          constant uint32_t *draw_count,
                          constant uint64_t *previous_base,
                          constant uint32_t *previous_clamp,
                          constant uint32_t *previous_stride,
                          constant struct
                             libagx_d3d12_sparse_registry_view
                                *sparse_registry,
                          uint64_t sparse_ro_offset,
                          uint32_t command_index, uint32_t offset_B,
                          uint32_t element_size_B,
                          uint32_t interchange_align)
{
   bool enabled = !draw_count || command_index < *draw_count;
   if (!enabled) {
      if (previous_base)
         *out_base = *previous_base;
      if (previous_clamp)
         *out_clamp = *previous_clamp;
      if (previous_stride)
         *out_stride = *previous_stride;
      return;
   }

   uint64_t address = ((uint64_t)in[1] << 32) | in[0];
   uint64_t ignored_residency;
   address = libagx_d3d12_sparse_read_address(
      address, sparse_registry, sparse_ro_offset, &ignored_residency);
   uint32_t size_B = in[2];
   uint32_t stride_B = in[3];
   uint32_t required_B = offset_B + element_size_B;

   if (address && required_B >= offset_B && size_B >= required_B) {
      *out_base = address + offset_B;
      *out_clamp = stride_B ? (size_B - required_B) / stride_B : UINT32_MAX;
   } else {
      *out_base = AGX_ZERO_PAGE_ADDRESS;
      *out_clamp = 0;
   }
   *out_stride = interchange_align ? stride_B / interchange_align : 0;
}

/* An indirect IBV makes the VDM's index address, range and element width GPU
 * data as well as the draw counts.  Generate the final direct VDM draw words
 * in the pre-graphics stream after both records are visible, avoiding any CPU
 * readback or command-list resubmission fixup. */
KERNEL(1)
libagx_d3d12_indirect_indexed_draw(
   global uint32_t *out, constant VkDrawIndexedIndirectCommand *command,
   constant uint32_t *view,
   constant struct libagx_d3d12_sparse_registry_view *sparse_registry,
   uint64_t null_address, uint64_t sparse_ro_offset,
   uint32_t format_r16_uint, uint32_t format_r32_uint,
   uint32_t restart, enum agx_primitive topology)
{
   VkDrawIndexedIndirectCommand cmd = *command;
   uint64_t address = ((uint64_t)view[1] << 32) | view[0];
   uint64_t ignored_residency;
   address = libagx_d3d12_sparse_read_address(
      address, sparse_registry, sparse_ro_offset, &ignored_residency);
   uint32_t size_B = view[2];
   uint32_t format = view[3];
   bool valid = address && size_B &&
                (format == format_r16_uint || format == format_r32_uint);
   enum agx_index_size index_size =
      format == format_r32_uint ? AGX_INDEX_SIZE_U32 : AGX_INDEX_SIZE_U16;

   if (!valid) {
      address = null_address;
      size_B = 4;
      cmd.indexCount = 0;
   }

   struct agx_draw draw = agx_draw_indexed(
      cmd.indexCount, cmd.instanceCount, cmd.firstIndex, cmd.vertexOffset,
      cmd.firstInstance, address, size_B, index_size, restart);
   agx_vdm_draw(out, 0, draw, topology);
}

/* D3D12 indirect dispatch records contain workGROUP counts while AGX's
 * indirect CDM mode consumes global thread counts. Perform the three bounded
 * multiplies on the GPU so records written earlier in the list remain fully
 * ordered and never require CPU mapping. */
KERNEL(1)
libagx_d3d12_indirect_dispatch(global uint32_t *out,
                               constant uint32_t *in, uint32_t workgroup_x,
                               uint32_t workgroup_y, uint32_t workgroup_z)
{
   out[0] = in[0] * workgroup_x;
   out[1] = in[1] * workgroup_y;
   out[2] = in[2] * workgroup_z;
}

/*
 * D3D12 predication applies to both draws and dispatches and reads a 64-bit
 * value. The command-list front end snapshots that value at SetPredication,
 * then uses this kernel to turn an ordinary indirect argument record into a
 * zero-count record when the selected comparison says to skip the command.
 *
 * `skip_if_nonzero` names D3D12_PREDICATION_OP_NOT_EQUAL_ZERO. The API's
 * operation describes the condition which suppresses execution, so equality
 * between the snapshot's truth value and this flag means zero the record.
 */
KERNEL(1)
libagx_d3d12_predicate(global uint32_t *out, constant uint32_t *in,
                       constant uint64_t *predicate, uint32_t words,
                       uint32_t skip_if_nonzero)
{
   bool skip = ((*predicate != 0) == (skip_if_nonzero != 0));

   for (uint i = 0; i < words; ++i)
      out[i] = skip ? 0 : in[i];
}

/*
 * Indexing/offseting is in software if necessary so we strip all
 * indexing/offset information.
 */
KERNEL(1)
libagx_draw_without_adj(global VkDrawIndirectCommand *out,
                        global VkDrawIndirectCommand *in,
                        global struct poly_vertex_params *vp,
                        uint64_t index_buffer, uint64_t index_buffer_range_el,
                        int index_size_B, enum mesa_prim prim)
{
   *out = (VkDrawIndirectCommand){
      .vertexCount = libagx_remap_adj_count(in->vertexCount, prim),
      .instanceCount = in->instanceCount,
   };

   /* TODO: Deduplicate */
   if (index_size_B) {
      uint offs = in->firstVertex;

      vp->index_buffer = poly_index_buffer(index_buffer, index_buffer_range_el,
                                           offs, index_size_B);

      vp->index_buffer_range_el =
         poly_index_buffer_range_el(index_buffer_range_el, offs);
   }
}

/* Precondition: len must be < the group size */
static void
libagx_memcpy_small(global uchar *dst, constant uchar *src, uint len, uint tid)
{
   if (tid < len) {
      dst[tid] = src[tid];
   }
}

static void
libagx_memcpy_aligned_uint4(global uint *dst, constant uint *src, uint len,
                            uint tid, uint group_size)
{
   for (uint i = tid; i < len; i += group_size) {
      vstore4(vload4(i, src), i, dst);
   }
}

static void
libagx_memcpy_to_aligned(global uint *dst, constant uchar *src, uint len,
                         uint tid, uint group_size)
{
   /* Copy a few bytes at the start */
   uint start_unaligned = ((uintptr_t)src) & 3;
   if (start_unaligned) {
      uint need = 4 - start_unaligned;
      libagx_memcpy_small((global uchar *)dst, src, need, tid);
      src += need;
      len -= need;
   }

   /* Copy a few bytes at the end */
   uint end_unaligned = len & 0xf;
   len -= end_unaligned;
   libagx_memcpy_small(((global uchar *)dst) + len, src + len, end_unaligned,
                       tid);

   /* Now both src and dst are word-aligned, and len is 16-aligned */
   libagx_memcpy_aligned_uint4(dst, (constant uint *)src, len / 16, tid,
                               group_size);
}

/* Precondition: len must be < the group size */
static void
libagx_memset_small(global uchar *dst, uchar b, int len, uint tid)
{
   if (tid < len) {
      dst[tid] = b;
   }
}

/*
 * AGX does not implement robustBufferAccess2 semantics for
 * index buffers, where out-of-bounds indices read as zero. When we
 * dynamically detect index buffer overread (this if-statement), we need
 * to clone the index buffer and zero-extend it to get robustness.
 *
 * We do this dynamically (generating a VDM draw to consume the result) to avoid
 * expensive allocations & memcpys in the happy path where no out-of-bounds
 * access occurs. Otherwise we could use a hardware indirect draw, rather than
 * generating VDM words directly in shader.
 *
 * TODO: Handle multiple draws in parallel.
 */
KERNEL(32)
libagx_draw_robust_index(global uint32_t *vdm, global struct poly_heap *heap,
                         constant VkDrawIndexedIndirectCommand *cmd,
                         uint64_t in_buf_ptr, uint32_t in_buf_range_B,
                         ushort restart, enum agx_chip chip,
                         enum agx_primitive topology,
                         uint32_t draw_state_context_offset,
                         uint32_t render_state_emitted,
                         enum agx_index_size index_size__3)
{
   uint tid = get_sub_group_local_id();
   bool first = tid == 0;
   enum agx_index_size index_size = index_size__3;

   struct agx_draw draw = agx_draw_indexed(
      cmd->indexCount, cmd->instanceCount, cmd->firstIndex, cmd->vertexOffset,
      cmd->firstInstance, in_buf_ptr, in_buf_range_B, index_size, restart);

   if (agx_indices_to_B(cmd->firstIndex, index_size) >= in_buf_range_B) {
      /* If the entire draw is out-of-bounds, skip it. We handle this specially
       * for both performance and to avoid integer wrapping issues in the main
       * code path (where cmd->firstIndex could get treated as a negative).
       */
      draw.index_buffer = AGX_ZERO_PAGE_ADDRESS;
      draw.index_buffer_range_B = 4;
      draw.start = 0;
   } else if (agx_direct_draw_overreads_indices(draw)) {
      uint start_B = agx_indices_to_B(draw.start, draw.index_size);
      constant void *in_buf =
         (constant void *)(draw.index_buffer + start_B);
      uint in_size_el =
         (draw.index_buffer_range_B - start_B) >> draw.index_size;
      uint in_size_B = agx_indices_to_B(in_size_el, index_size);

      /* After a small number of zeroes at the end, extra zeroes cannot change
       * rendering since they will duplicate the same degenerate primitive many
       * times. Therefore we clamp the number of zeroes we need to extend with.
       * This makes the memset constant time.
       */
      draw.b.count[0] =
         agx_robust_index_clamped_count(draw.b.count[0], in_size_el);

      uint out_size_el = draw.b.count[0];
      uint out_size_B = 0, alloc_size_B = 0;
      bool size_valid = agx_robust_index_shadow_sizes(
         out_size_el, index_size, &out_size_B, &alloc_size_B);

      /* Allocate memory for the shadow index buffer.  The general geometry
       * allocator reports overflow through the AGX2 printf/abort ABI.  This
       * G17P-capable precompiled kernel cannot bake a process-local printf
       * address into cached machine code, so make overflow memory-safe here:
       * every lane observes the same atomic allocation and an exhausted heap
       * emits an empty indexed draw instead of touching an invalid address. */
      uint padded_offs = 0;
      if (first && size_valid) {
         padded_offs = atomic_fetch_add(
            (volatile global atomic_uint *)(&heap->bottom), alloc_size_B);
      }

      padded_offs = sub_group_broadcast(padded_offs, 0);
      bool allocated =
         size_valid &&
         poly_heap_allocation_fits(padded_offs, alloc_size_B, heap->size);
      global uchar *padded = heap->base + padded_offs;

      if (!allocated) {
         draw.b.count[0] = 0;
         draw.index_buffer = AGX_ZERO_PAGE_ADDRESS;
         draw.index_buffer_range_B = 4;
         draw.start = 0;
      } else {
         draw.index_buffer = (uintptr_t)padded;
         draw.index_buffer_range_B = out_size_B;
         draw.start = 0;

         /* Clone the index buffer. The destination is aligned as a
          * post-condition of the heap allocation above. */
         libagx_memcpy_to_aligned((global uint *)padded, in_buf, in_size_B,
                                  tid, 32);

         /* Extend with up to 32 zeroes with a small memset */
         libagx_memset_small(padded + in_size_B, 0, out_size_B - in_size_B,
                             tid);
      }
   }

   if (first) {
      agx_vdm_draw_robust_index(vdm, chip, draw, topology,
                                draw_state_context_offset,
                                render_state_emitted != 0);
   }
}
