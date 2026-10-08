/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "d12_private.h"
#include "d12_native_compute.h"
#include "asahi/lib/agx3_code_container_core.h"
#include "asahi/lib/agx_ioto_program_core.h"

/* Reuse the shared G17P CODE producer. Driver uniforms come from the
 * command-lifetime Ioto program; this container has no generated preamble. */
HRESULT
d12_native_compute_code_create(struct d12_device *device,
                            const struct agx_shader_part *part,
                            struct agx_bo **out_bo, uint64_t *out_entry, uint64_t *out_rodata)
{
   const struct agx_shader_info *info = &part->info;
   const size_t rodata_size = (size_t)info->rodata.size_16 * 2;
   const size_t block_size = agx3_code_block_size(info->main_size);
   const size_t entry_offset = AGX3_CONTAINER_LEAD_SIZE +
                              AGX3_CODE_BLOCK_ENTRY_OFFSET;

   if (!part->binary || info->stage != MESA_SHADER_COMPUTE ||
       !info->main_size || info->main_offset > info->binary_size ||
       info->main_size > info->binary_size - info->main_offset ||
       !info->nr_gprs || !info->agx3_staging_known ||
       !info->workgroup_size[0] || !info->workgroup_size[1] ||
       !info->workgroup_size[2] || !block_size ||
       info->rodata.offset > info->binary_size ||
       rodata_size > info->binary_size - info->rodata.offset ||
       block_size > SIZE_MAX - AGX3_CONTAINER_LEAD_SIZE - rodata_size)
      return E_FAIL;

   /* No legacy PRESHADER, helper scratch, or uniform-DMA producer is supplied
    * by the native direct-compute path yet. Keep those requirements explicit. */
   if (info->has_preamble || info->nr_preamble_gprs || info->scratch_size ||
       info->preamble_scratch_size || info->agx3_uniform_dma_count ||
       info->imageblock_stride)
      return E_NOTIMPL;

   const size_t container_size = AGX3_CONTAINER_LEAD_SIZE + block_size;
   const size_t upload_size = container_size + rodata_size;
   struct agx_bo *bo = agx_bo_create(&device->dev, upload_size, 0,
                                    AGX_BO_EXEC | AGX_BO_LOW_VA,
                                    "d3d12 native compute container");
   if (!bo)
      return E_OUTOFMEMORY;

   if (!bo->va || bo->va->addr > UINT64_MAX - entry_offset)
      goto fail;
   const uint64_t entry = bo->va->addr + entry_offset;
   if (!agx_va_in_usc_window(&device->dev, bo->va->addr) ||
       !agx_va_in_usc_window(&device->dev, entry) ||
       info->main_size > AGX_USC_WINDOW_SIZE -
                           (entry - device->dev.shader_base))
      goto fail;

   uint8_t *mapping = agx_bo_map(bo);
   if (!mapping)
      goto fail;
   agx3_build_container_lead(mapping);
   if (!agx3_build_code_block(mapping + AGX3_CONTAINER_LEAD_SIZE,
                              block_size,
                              (const uint8_t *)part->binary + info->main_offset,
                              info->main_size, info->main_size, NULL, false))
      goto fail;
   if (rodata_size)
      memcpy(mapping + container_size,
             (const uint8_t *)part->binary + info->rodata.offset, rodata_size);
   if (agx_bo_sync_for_device(bo, 0, upload_size))
      goto fail;

   *out_bo = bo;
   *out_entry = entry;
   *out_rodata = rodata_size ? bo->va->addr + container_size : 0;
   return S_OK;

fail:
   agx_bo_unreference(&device->dev, bo);
   return E_FAIL;
}

bool
d12_native_compute_uniforms(const struct agx_shader_info *info,
                            uint32_t *required_cells)
{
   if (!info || !required_cells || !info->agx3_uniform_halfs_valid)
      return false;

   /* Slot zero and fragment-only slot 12 have no compute producer. The
    * unused slot 8 must not become an implicit Poly pointer publication. */
   const uint32_t allowed = BITFIELD_BIT(D12_UNIFORM_ROOT / 4) |
                           BITFIELD_RANGE(D12_UNIFORM_TEXTURES / 4,
                                          D12_NATIVE_COMPUTE_CELLS -
                                             D12_UNIFORM_TEXTURES / 4);
   const unsigned ro_base = info->rodata.base_uniform;
   const unsigned ro_size = info->rodata.size_16;
   if (ro_size && ((ro_base & 1) || (ro_size & 1) ||
                   ro_base < D12_UNIFORM_RESERVED ||
                   ro_base >= AGX3_CPROG_INPUT_HALVES ||
                   ro_size > AGX3_CPROG_INPUT_HALVES - ro_base))
      return false;

   uint32_t cells = 0;
   for (unsigned h = 0; h < AGX3_CPROG_INPUT_HALVES; ++h) {
      if (!(info->agx3_uniform_halfs[h / 64] & BITFIELD64_BIT(h % 64)))
         continue;
      if (ro_size && h >= ro_base && h - ro_base < ro_size)
         continue;
      if (h >= D12_UNIFORM_RESERVED || !(allowed & BITFIELD_BIT(h / 4)))
         return false;
      cells |= BITFIELD_BIT(h / 4);
   }
   *required_cells = cells;
   return true;
}

bool
d12_native_compute_ranges(
   const struct agx_shader_info *info, const struct d12_native_compute_abi *abi,
   const uint64_t cells[D12_NATIVE_COMPUTE_CELLS],
   struct agx3_ioto_uniform_range ranges[D12_NATIVE_COMPUTE_RANGES],
   unsigned *count)
{
   uint32_t required;
   if (!abi || !abi->certified || !cells || !ranges || !count ||
       !d12_native_compute_uniforms(info, &required) ||
       required != abi->required_cells)
      return false;

   struct agx3_ioto_uniform_range staged[D12_NATIVE_COMPUTE_RANGES];
   unsigned n = 0;
   for (unsigned slot = 0; slot < D12_NATIVE_COMPUTE_CELLS; ++slot) {
      if (!(required & BITFIELD_BIT(slot)))
         continue;
      if (!cells[slot] || (cells[slot] & 7) ||
          cells[slot] >= (UINT64_C(1) << 48) - 7)
         return false;
      staged[n++] = (struct agx3_ioto_uniform_range){
         .start_halfs = slot * 4, .size_halfs = 4, .address = cells[slot],
      };
   }
   if (info->rodata.size_16) {
      size_t size = (size_t)info->rodata.size_16 * 2;
      if (!abi->rodata_va || (abi->rodata_va & 3) ||
          abi->rodata_va >= (UINT64_C(1) << 48) ||
          size > (UINT64_C(1) << 48) - abi->rodata_va)
         return false;
      staged[n++] = (struct agx3_ioto_uniform_range){
         .start_halfs = info->rodata.base_uniform,
         .size_halfs = info->rodata.size_16, .address = abi->rodata_va,
      };
   }
   memcpy(ranges, staged, n * sizeof(staged[0]));
   *count = n;
   return true;
}

bool
d12_native_compute_generate(
   const struct agx_shader_info *info, const struct d12_native_compute_abi *abi,
   uint64_t entry, uint64_t program_va,
   const uint64_t cells[D12_NATIVE_COMPUTE_CELLS],
   struct agx3_g17p_compute_program *program)
{
   struct agx3_ioto_uniform_range ranges[D12_NATIVE_COMPUTE_RANGES];
   unsigned count;
   struct agx3_cdm_direct_template_sources sources;
   if (!info || info->stage != MESA_SHADER_COMPUTE ||
       !info->main_size || !info->nr_gprs ||
       !info->agx3_staging_known || info->has_preamble ||
       info->nr_preamble_gprs || info->scratch_size ||
       info->preamble_scratch_size || info->agx3_uniform_dma_count ||
       info->imageblock_stride || info->texture_state_count ||
       info->sampler_state_count || info->uses_txf ||
       info->uses_sampler_heap || info->uses_ray_traversal ||
       !info->workgroup_size[0] || !info->workgroup_size[1] ||
       !info->workgroup_size[2] ||
       !d12_native_compute_ranges(info, abi, cells, ranges, &count) ||
       !agx3_cdm_direct_template_sources_compute(
          info->uses_resource_info_query, false, false, &sources))
      return false;

   const struct agx3_g17p_compute_program_desc desc = {
      .program_va = program_va, .shader_va = entry,
      .uniform_ranges = ranges, .uniform_range_count = count,
      .shader_binary_size = info->main_size,
      .scratch_size = info->agx3_spectpr_scratch_size,
      .nr_gprs = info->nr_gprs,
      .compiler_d_valid = info->agx3_staging_known,
      .compiler_d = (uint32_t)info->agx3_staging_halfs * 2,
      .threadgroup_memory_size_B = info->local_size
         ? MAX2(256u, ALIGN_POT(info->local_size, 64u)) : 0,
      .launch_sources = sources,
      .ldshdr_f1 = AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
   };
   return agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &desc, program);
}
