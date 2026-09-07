/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "libagx/query.h"
#include "vulkan/vulkan_core.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_pack.h"
#include "agx_scratch.h"
#include "agx_tilebuffer.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_agx3_ioto_program_core.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx3_g17p_grid_helper_core.h"
#include "util/log.h"
#include "util/u_debug.h"
#include "pool.h"

#include <inttypes.h>
#include <stdio.h>

void
hk_cmd_buffer_begin_compute(struct hk_cmd_buffer *cmd,
                            const VkCommandBufferBeginInfo *pBeginInfo)
{
}

void
hk_cmd_invalidate_compute_state(struct hk_cmd_buffer *cmd)
{
   struct hk_push_descriptor_set *push[HK_MAX_SETS];
   memcpy(push, cmd->state.cs.descriptors.push, sizeof(push));
   memset(&cmd->state.cs, 0, sizeof(cmd->state.cs));
   memcpy(cmd->state.cs.descriptors.push, push, sizeof(push));
}

void
hk_cmd_bind_compute_shader(struct hk_cmd_buffer *cmd,
                           struct hk_api_shader *shader)
{
   cmd->state.cs.shader = shader;
}

void
hk_cdm_cache_flush(struct hk_device *dev, struct hk_cs *cs)
{
   assert(cs->type == HK_CS_CDM);
   assert(cs->current + AGX_CDM_BARRIER_LENGTH < cs->end &&
          "caller must ensure space");

   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      cs->current = agx_cdm_barrier(cs->current, dev->dev.chip);
      break;

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      /* Apple's CommandEncodingNext emitter uses flags=5 for its strong
       * post-dispatch cache flush on every grounded G15--G17 identity.  Keep
       * Honeykrisp's deliberately conservative flush-after-every-launch
       * policy, but encode it in the target command-stream ISA. */
      const struct agx3_cdm_barrier_info info = {
         .chip = dev->dev.chip,
         .flags = 0x5,
      };
      uint8_t packed[AGX3_CDM_BARRIER_SIZE];
      if (!agx3_cdm_barrier_pack(&info, packed)) {
         mesa_loge("no grounded AGX3 CDM cache flush for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cs->cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      memcpy(cs->current, packed, sizeof(packed));
      cs->current = (uint32_t *)((uint8_t *)cs->current + sizeof(packed));
      break;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      mesa_loge("unknown CDM command-stream generation for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cs->cmd->vk,
                                  VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   cs->stats.flushes++;
}

void
hk_dispatch_with_usc_launch(struct hk_device *dev, struct hk_cs *cs,
                            struct agx_cdm_launch_word_0_packed launch,
                            uint32_t usc, struct agx_grid grid,
                            struct agx_workgroup wg)
{
   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      break;

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300:
      /* `usc` addresses an AGX2 USC block.  AGX3 direct launches instead
       * require a dispatch-local, 64-byte-aligned Ioto program VA.  Reject
       * this legacy entry point until the caller supplies that joined object;
       * never feed an AGX2 launch to CommandEncodingNext. */
      mesa_loge("legacy USC compute launch rejected for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cs->cmd->vk,
                                  VK_ERROR_FEATURE_NOT_PRESENT);
      return;

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      mesa_loge("unknown compute command-stream generation for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cs->cmd->vk,
                                  VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   hk_ensure_cs_has_space(cs->cmd, cs, 0x2000 /* TODO */);
   cs->stats.cmds++;
   cs->current =
      agx_cdm_launch(cs->current, dev->dev.chip, grid, wg, launch, usc);

   hk_cdm_cache_flush(dev, cs);
}

void
hk_dispatch_with_usc(struct hk_device *dev, struct hk_cs *cs,
                     struct agx_shader_info *info, uint32_t usc,
                     struct agx_grid grid, struct agx_workgroup local_size)
{
   struct agx_cdm_launch_word_0_packed launch;
   agx_pack(&launch, CDM_LAUNCH_WORD_0, cfg) {
      cfg.texture_state_register_count = info->texture_state_count;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(info->sampler_state_count, false);
      cfg.uniform_register_count = info->push_count;
      cfg.preshader_register_count = info->nr_preamble_gprs;
   }

   hk_dispatch_with_usc_launch(dev, cs, launch, usc, grid, local_size);
}

static bool
hk_g17p_compute_shader_supported(struct hk_shader *s,
                                 struct hk_linked_shader *linked)
{
   if (s && linked && hk_g17p_launch_trace_enabled())
      fprintf(stderr,"G17P compute resources: txf=%u binding=%u storage=%u sampled_storage=%u preamble=%u local=%u texture=%u sampler=%u heap=%u scratch=%u/%u\n",
         linked->b.uses_txf,s->info.agx3_txf_texture_valid,
         s->info.agx3_storage_image_valid,s->info.agx3_sampled_storage_valid,
         s->b.info.has_preamble,
         s->b.info.local_size,s->b.info.texture_state_count,s->b.info.sampler_state_count,
         s->b.info.uses_sampler_heap,s->b.info.scratch_size,linked->b.scratch_size);
   return s && linked && s->info.stage == MESA_SHADER_COMPUTE &&
          linked->b.bo && linked->b.bo->va && linked->b.size != 0 &&
          linked->b.size <= UINT32_MAX && linked->b.nr_gprs != 0 &&
          !s->b.info.has_preamble && s->b.info.preamble_scratch_size == 0 &&
          s->b.info.scratch_size == 0 && linked->b.scratch_size == 0 &&
          s->b.info.agx3_staging_known &&
          s->b.info.local_size <= 32768 &&
          s->b.info.texture_state_count == 0 &&
          (s->b.info.sampler_state_count == 0 ||
           ((s->info.agx3_txf_texture_valid ||
             s->info.agx3_storage_image_valid ||
             s->info.agx3_sampled_storage_valid ||
             s->info.agx3_indexed_storage_valid ||
             s->info.agx3_indexed_sampler_valid) &&
            s->b.info.sampler_state_count == 1)) &&
          !s->b.info.uses_sampler_heap &&
          (!linked->b.uses_txf || s->info.agx3_txf_texture_valid ||
           s->info.agx3_storage_image_valid ||
           s->info.agx3_sampled_storage_valid ||
           s->info.agx3_indexed_storage_valid ||
           s->info.agx3_indexed_sampler_valid) &&
          s->info.set_count <= HK_AGX3_G17P_MAX_ROOT_SETS;
}

static bool
hk_g17p_compute_uniform_ranges(
   struct hk_shader *s, uint64_t root_ptr, uint64_t texture_pointers,
   struct agx3_ioto_uniform_range ranges[3], size_t *range_count)
{
   if (!s || !ranges || !range_count ||
       (!!texture_pointers != (s->info.agx3_txf_texture_valid ||
                               s->info.agx3_storage_image_valid ||
                               s->info.agx3_sampled_storage_valid ||
                               s->info.agx3_indexed_storage_valid ||
                               s->info.agx3_indexed_sampler_valid)))
      return false;

   /* A shader that reads neither the root descriptor nor any descriptor set
    * needs NO preamble loads at all.
    *
    * This path used to build a root range unconditionally, so every dispatch
    * got an Ioto program that loads the root descriptor and sets[0] from
    * memory before the first shader instruction runs -- even a shader that
    * touches nothing.  That matters because the preamble's loads happen
    * whatever the shader does: a null-shader dispatch still faults with
    * reason=3 and zero retire duration, which is a fault BEFORE the first
    * instruction, and the preamble is the only thing that executes there.
    *
    * `agx3_uses_root` already records whether the compiler emitted
    * load_root_agx (hk_shader.c:1156, and hk_cmd_buffer.c:979 already gates on
    * it), and `hk_agx3_g17p_compute_program_generate` already composes a
    * no-resource program when range_count == 0.  Both halves existed; compute
    * simply never connected them.
    *
    * Zero ranges is therefore both a correctness improvement (no load the
    * shader cannot consume) and the bisect the null probe needs: a dispatch
    * whose preamble performs no memory access whatsoever.
    */
   const struct agx_rodata *rodata_probe = &s->b.info.rodata;
   if (!texture_pointers && !s->info.agx3_uses_root && s->info.set_count == 0 &&
       rodata_probe->size_16 == 0) {
      *range_count = 0;
      return true;
   }

   const unsigned root_uniform = (s->info.agx3_storage_image_valid ||
                                  s->info.agx3_sampled_storage_valid ||
                                  s->info.agx3_indexed_storage_valid)
                                    ? AGX_G17P_STORAGE_ROOT_HALFS
                                 : texture_pointers
                                    ? AGX_G17P_TXF_ROOT_HALFS
                                    : 0;
   if (!hk_agx3_g17p_root_range(root_uniform, s->info.set_count, root_ptr, &ranges[0]))
      return false;

   /* Deliver the footprint the compiler finally settled on.
    *
    * `reserved_preamble = root + 4 * (1 + set_count)` (hk_shader.c:1255) is
    * where the compiler STARTS; agx_opt_promote_constants() can raise
    * `push_count` afterwards (agx_opt_promote_constants.c:149).  Measured on
    * hardware this is currently a no-op -- push_count reads 8 for the probe,
    * exactly 4 * (1 + 1) -- so this only matters for a shader whose constants
    * were promoted.  Safe either way: push_count is in the same 16-bit
    * uniform registers as size_halfs, and the bytes come from the root
    * descriptor table, which is larger than the range.
    */
   if (s->b.info.push_count > root_uniform + ranges[0].size_halfs)
      ranges[0].size_halfs = s->b.info.push_count - root_uniform;

   size_t count = 1;
   if (texture_pointers) {
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0, .size_halfs = root_uniform,
         .address = texture_pointers,
      };
   }
   const struct agx_rodata *rodata = &s->b.info.rodata;
   if (rodata->size_16 != 0) {
      const uint64_t size_B = (uint64_t)rodata->size_16 * 2;
      if (!s->bo || !s->bo->va || (rodata->base_uniform & 1) != 0 ||
          (rodata->size_16 & 1) != 0 || rodata->size_16 > 512 ||
          rodata->offset > s->bo->size ||
          size_B > s->bo->size - rodata->offset)
         return false;

      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = rodata->base_uniform,
         .size_halfs = rodata->size_16,
         .address = s->bo->va->addr + rodata->offset,
      };
   }

   *range_count = count;
   return true;
}

/*
 * Compose and upload one dispatch-local Ioto program for `s`, bound to its
 * native launch state.  A native indirect dispatch needs two of these -- the
 * caller program and the grid helper -- built from different roots and
 * announcing different program kinds, so this is shared rather than inlined.
 *
 * `trace` names the program in the launch trace; the caller's diagnostics stay
 * with the caller.  On failure the command buffer error is already set and
 * *out is meaningless.
 */
static bool
hk_g17p_build_compute_program_raw(
   struct hk_cmd_buffer *cmd, uint64_t shader_va,
   uint32_t shader_binary_size, uint32_t scratch_size, uint16_t nr_gprs,
   bool compiler_d_valid, uint32_t compiler_d,
   uint32_t threadgroup_memory_size_B,
   const struct agx3_cdm_direct_template_sources *launch_sources,
   enum agx_ioto_g17p_compute_ldshdr_f1 kind, const char *trace,
   const struct agx3_ioto_uniform_range *ranges, size_t range_count,
   struct agx_bo **program_bo_out,
   struct hk_agx3_g17p_compute_program *out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!shader_va || !shader_binary_size || !nr_gprs || !compiler_d_valid ||
       !launch_sources || !program_bo_out || !out ||
       (range_count && !ranges)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   struct agx_bo *program_bo = agx_bo_create(
      &dev->dev, HK_AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE,
      AGX_IOTO_PROGRAM_ALIGNMENT, AGX_BO_EXEC | AGX_BO_LOW_VA,
      "G17P compute Ioto program");
   if (!program_bo) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   util_dynarray_append(&cmd->large_bos, program_bo);

   void *program_map = agx_bo_map(program_bo);
   if (!program_map) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_MEMORY_MAP_FAILED);
      return false;
   }

   /* Fail closed on a USC-window escape.
    *
    * usc_exec_base (== dev->dev.shader_base) is handed to the kernel once at
    * queue-create time and programmed into USC_EXEC_BASE_CP.  Every exec
    * pointer this record carries -- the Ioto program the CDM launch points at,
    * and the shader binary the Ioto LDSHDR record points at -- must resolve
    * inside that 4 GiB window.  A pointer outside it is a GPU page fault a few
    * microseconds later with no attribution, so refuse the dispatch here,
    * where both numbers are known, instead of letting the GPU discover it.
    */
   if (!agx_va_in_usc_window(&dev->dev, program_bo->va->addr) ||
       !agx_va_in_usc_window(&dev->dev, shader_va) ||
       shader_binary_size >
          AGX_USC_WINDOW_SIZE - (shader_va - dev->dev.shader_base)) {
      mesa_loge("G17P dispatch: exec pointer outside the USC window "
                "[%" PRIx64 ", %" PRIx64 "): program=%" PRIx64
                " shader=%" PRIx64 "\n",
                dev->dev.shader_base,
                dev->dev.shader_base + AGX_USC_WINDOW_SIZE,
                program_bo->va->addr, shader_va);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
      return false;
   }

   const struct hk_agx3_g17p_compute_program_desc desc = {
      .program_va = program_bo->va->addr,
      /* The container's entry point, not the allocation base: the
       * shader loader reads the block-size dword at entry - 0x80. */
      .shader_va = shader_va,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .shader_binary_size = shader_binary_size,
      .scratch_size = scratch_size,
      .nr_gprs = nr_gprs,
      .compiler_d_valid = compiler_d_valid,
      .compiler_d = compiler_d,
      .threadgroup_memory_size_B = threadgroup_memory_size_B,
      .launch_sources = *launch_sources,
      .ldshdr_f1 = kind,
   };
   if (!hk_agx3_g17p_compute_program_generate(dev->dev.chip, &desc, out)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   memcpy(program_map, out->bytes, out->length);
   if (agx_bo_sync_for_device(program_bo, 0, out->length)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_DEVICE_LOST);
      return false;
   }
   *program_bo_out = program_bo;
   if (trace && hk_g17p_launch_trace_enabled())
      fprintf(stderr, "G17P program[%s]: kind=%u va=%" PRIx64 " len=%u\n",
              trace, (unsigned)kind, program_bo->va->addr,
              (unsigned)out->length);
   return true;
}

static bool
hk_g17p_build_compute_program(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                              struct hk_linked_shader *linked,
                              uint64_t root_ptr, uint64_t texture_pointers,
                              enum agx_ioto_g17p_compute_ldshdr_f1 kind,
                              const char *trace,
                              struct agx3_ioto_uniform_range ranges[3],
                              size_t *range_count_out,
                              struct agx_bo **program_bo_out,
                              struct hk_agx3_g17p_compute_program *out)
{
   size_t range_count;
   if (!hk_g17p_compute_uniform_ranges(s, root_ptr, texture_pointers, ranges,
                                       &range_count)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(
          s->b.info.uses_resource_info_query,
          s->b.info.uses_ray_traversal,
          s->b.info.ray_traversal_depth_nonzero, &launch_sources)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   /* The recorded Ioto program dereferences these executable/constant BOs
    * after vkQueueSubmit returns. Their originating pipeline need not remain
    * alive after command recording, so the command buffer owns references. */
   hk_cmd_buffer_retain_bo(cmd, linked->b.bo);
   if (s->b.info.rodata.size_16 != 0)
      hk_cmd_buffer_retain_bo(cmd, s->bo);

   const bool ok = hk_g17p_build_compute_program_raw(
      cmd, linked->b.bo->va->addr + linked->b.entry_offset, linked->b.size,
      linked->b.scratch_size, linked->b.nr_gprs,
      s->b.info.agx3_staging_known,
      (uint32_t)s->b.info.agx3_staging_halfs * 2,
      s->b.info.local_size ? MAX2(256u, ALIGN_POT(s->b.info.local_size, 64u))
                           : 0,
      &launch_sources, kind, trace, ranges, range_count, program_bo_out, out);
   if (ok)
      *range_count_out = range_count;
   return ok;
}

void
hk_dispatch_precomp_g17p(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                         enum agx_barrier barrier, enum libagx_program idx,
                         const void *data, size_t data_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if ((unsigned)idx >= LIBAGX_NUM_PROGRAMS) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct agx_precompiled_shader *prog =
      agx_get_precompiled(&dev->bg_eot, idx);
   if (dev->dev.chip != AGX_CHIP_G17P || !prog || !prog->bo ||
       !prog->info.main_size || !prog->info.nr_gprs ||
       prog->info.nr_preamble_gprs || prog->info.rodata.size_16 ||
       !prog->info.agx3_staging_known ||
       !data || !data_size || data_size > UINT16_MAX * 2u ||
       prog->entry_offset > UINT64_MAX - prog->ptr ||
       prog->b.workgroup.x == 0 || prog->b.workgroup.y == 0 ||
       prog->b.workgroup.z == 0 || agx_is_indirect(grid)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (grid.count[0] == 0 || grid.count[1] == 0 || grid.count[2] == 0)
      return;

   struct hk_cs **target = (barrier & AGX_POSTGFX)  ? &cmd->current_cs.post_gfx
                           : (barrier & AGX_PREGFX) ? &cmd->current_cs.pre_gfx
                                                    : &cmd->current_cs.cs;
   struct hk_cs *cs = hk_cmd_buffer_get_cs_general(cmd, target, true);
   if (!cs)
      return;

   const uint64_t uploaded_data = hk_pool_upload(cmd, data, data_size, 4);
   if (!uploaded_data) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   const struct agx3_ioto_uniform_range range = {
      .start_halfs = 0,
      .size_halfs = DIV_ROUND_UP(data_size, 2),
      .address = uploaded_data,
   };
   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(false, false, false,
                                                  &launch_sources)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   struct agx_bo *program_bo;
   struct hk_agx3_g17p_compute_program program;
   if (!hk_g17p_build_compute_program_raw(
          cmd, prog->ptr + prog->entry_offset, prog->info.main_size, 0,
          prog->info.nr_gprs, true,
          (uint32_t)prog->info.agx3_staging_halfs * 2, 0, &launch_sources,
          AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3, "precompiled", &range,
          1, &program_bo, &program))
      return;

   const struct agx3_compute_direct_grid direct = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {grid.count[0], grid.count[1], grid.count[2]},
      .local_size = {prog->b.workgroup.x, prog->b.workgroup.y,
                     prog->b.workgroup.z},
      .dimensions = grid.count[2] != 1 || prog->b.workgroup.z != 1
                       ? 3
                    : grid.count[1] != 1 || prog->b.workgroup.y != 1 ? 2 : 1,
   };

   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   struct agx3_compute_dispatch_sink sink = {
      .data = (uint8_t *)cs->current,
      .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
   };
   if (!hk_agx3_g17p_native_compute_dispatch_append(&program.launch, &direct,
                                                     &sink)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   cs->stats.cmds++;
}

void
hk_dispatch_g17p_native(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        struct hk_shader *s, struct agx_grid grid,
                        struct agx_workgroup local_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *linked = s ? s->only_linked : NULL;
   static_assert(HK_MAX_SETS == HK_AGX3_G17P_MAX_ROOT_SETS,
                 "G17P root range must cover every Honeykrisp set");

   if (dev->dev.chip != AGX_CHIP_G17P || cs->type != HK_CS_CDM ||
       !hk_g17p_compute_shader_supported(s, linked)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   /* An indirect grid supplies its threadgroup counts at submit time; the
    * grid helper turns them into the geometry the main record consumes.
    * Without a helper program there is nothing to publish that geometry, so
    * refuse rather than record a launch whose dimensions never arrive. */
   const bool indirect = agx_is_indirect(grid);
   struct hk_shader *helper_s =
      indirect && dev->g17p_grid_helper
         ? hk_only_variant(dev->g17p_grid_helper) : NULL;
   struct hk_linked_shader *helper_linked =
      helper_s ? helper_s->only_linked : NULL;
   if (indirect) {
      if (grid.mode != AGX_CDM_MODE_INDIRECT_GLOBAL || grid.ptr == 0 ||
          (grid.ptr & 3) != 0 ||
          !hk_g17p_compute_shader_supported(helper_s, helper_linked)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
   } else if (grid.count[0] == 0 || grid.count[1] == 0 ||
              grid.count[2] == 0) {
      return;
   }

   /* The local size must be launchable at all: a zero component would make
    * the helper publish a zero global size, which is indistinguishable from a
    * geometry block it never wrote.  The product itself is the application's
    * responsibility -- the helper multiplies in wrapping 32-bit, exactly as
    * the AGX2 hardware indirect mode does, and Vulkan already bounds
    * groupCount by maxComputeWorkGroupCount. */
   if (indirect &&
       (local_size.x == 0 || local_size.y == 0 || local_size.z == 0)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   if (s->data_size != 0) {
      const uint64_t constant_data = hk_pool_upload(
         cmd, s->data_ptr, s->data_size, HK_MIN_UBO_ALIGNMENT);
      if (!constant_data) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }

      cmd->state.cs.descriptors.root.cs.constant_data_addr = constant_data;
   } else {
      cmd->state.cs.descriptors.root.cs.constant_data_addr = 0;
   }

   const uint64_t root_ptr =
      hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
   if (!root_ptr) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   uint64_t texture_pointers = 0;
   if (s->info.agx3_txf_texture_valid || s->info.agx3_storage_image_valid ||
       s->info.agx3_sampled_storage_valid ||
       s->info.agx3_indexed_storage_valid ||
       s->info.agx3_indexed_sampler_valid) {
      const struct hk_descriptor_state *descriptors = &cmd->state.cs.descriptors;

      if (s->info.agx3_indexed_sampler_valid) {
         if (!dev->sampled_images.bo || !dev->sampled_images.bo->va ||
             !dev->samplers.table.bo || !dev->samplers.table.bo->va) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         uint64_t pointers[4] = {
            dev->sampled_images.bo->va->addr,
            dev->samplers.table.bo->va->addr,
         };
         unsigned pointer_count = 2;
         if (s->info.agx3_indexed_storage_valid) {
            const struct agx_g17p_indexed_storage_binding *binding =
               &s->info.agx3_indexed_storage;
            if (binding->pbe_set >= HK_MAX_SETS) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }
            const uint64_t set_address =
               descriptors->root.sets[binding->pbe_set];
            const uint32_t set_size =
               descriptors->set_sizes[binding->pbe_set];
            if (!set_address || binding->pbe_offset_B > set_size ||
                set_size - binding->pbe_offset_B < 32) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }
            const struct hk_agx3_g17p_indexed_storage_heap heap = {
               .texture_heap_va = pointers[0],
               .sampler_heap_va = pointers[1],
               .pbe_descriptor_set_va = set_address,
               .pbe_offset_B = binding->pbe_offset_B,
            };
            if (!hk_agx3_g17p_indexed_storage_pointers(&heap, pointers)) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }
            pointer_count = 4;
         }
         texture_pointers = hk_pool_upload(
            cmd, pointers, pointer_count * sizeof(pointers[0]), 8);
         if (!texture_pointers) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }
      } else if (s->info.agx3_sampled_storage_valid) {
         const struct agx_g17p_sampled_storage_binding *binding =
            &s->info.agx3_sampled_storage;
         if (binding->texture_set >= HK_MAX_SETS ||
             binding->pbe_set >= HK_MAX_SETS) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         const uint64_t texture_set_address =
            descriptors->root.sets[binding->texture_set];
         const uint64_t pbe_set_address =
            descriptors->root.sets[binding->pbe_set];
         const uint32_t texture_set_size =
            descriptors->set_sizes[binding->texture_set];
         const uint32_t pbe_set_size =
            descriptors->set_sizes[binding->pbe_set];
         struct hk_descriptor_set *texture_set =
            descriptors->sets[binding->texture_set];
         if (!texture_set_address || !pbe_set_address || !texture_set ||
             !texture_set->mapped_ptr ||
             binding->texture_offset_B > texture_set_size ||
             texture_set_size - binding->texture_offset_B < 32 ||
             binding->sampler_index_offset_B > texture_set->size ||
             texture_set->size - binding->sampler_index_offset_B <
                sizeof(uint16_t) ||
             binding->pbe_offset_B > pbe_set_size ||
             pbe_set_size - binding->pbe_offset_B < 32 ||
             !dev->samplers.table.bo || !dev->samplers.table.bo->va) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }

         uint16_t sampler_index;
         memcpy(&sampler_index,
                (const uint8_t *)texture_set->mapped_ptr +
                   binding->sampler_index_offset_B,
                sizeof(sampler_index));
         const struct hk_agx3_g17p_sampled_storage_heap heap = {
            .sampled = {
               .descriptor_set_va = texture_set_address,
               .sampler_heap_va = dev->samplers.table.bo->va->addr,
               .texture_offset_B = binding->texture_offset_B,
               .sampler_index = sampler_index,
               .sampler_count = dev->samplers.table.alloc,
               .sampler_element_size_B =
                  agx_chip_sampler_heap_element_size(dev->dev.chip),
            },
            .pbe_descriptor_set_va = pbe_set_address,
            .pbe_offset_B = binding->pbe_offset_B,
         };
         uint64_t pointers[4] = {0};
         if (!hk_agx3_g17p_sampled_storage_pointers(&heap, pointers)) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }

         texture_pointers = hk_pool_upload(cmd, pointers, sizeof(pointers), 8);
         if (!texture_pointers) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }
      } else {
      const uint64_t set_address = descriptors->root.sets[0];
      const uint32_t set_size = descriptors->set_sizes[0];
      const uint32_t offset = s->info.agx3_storage_image_valid
         ? s->info.agx3_storage_image.texture_offset_B
         : s->info.agx3_txf_texture.texture_offset_B;
      if (!set_address || offset > set_size ||
          set_size - offset <
             (s->info.agx3_storage_image_valid ? 64 : 32) ||
          !dev->rodata.bo ||
          !dev->rodata.bo->va) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      uint64_t pointers[4] = {0};
      if (s->info.agx3_combined_sampler_valid) {
         struct hk_descriptor_set *set = descriptors->sets[0];
         const struct agx_g17p_combined_sampler_binding *binding =
            &s->info.agx3_combined_sampler;
         if (!set || !set->mapped_ptr ||
             binding->sampler_index_offset_B > set->size ||
             set->size - binding->sampler_index_offset_B < sizeof(uint16_t)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }

         uint16_t sampler_index;
         memcpy(&sampler_index,
                (const uint8_t *)set->mapped_ptr +
                   binding->sampler_index_offset_B,
                sizeof(sampler_index));
         if (!dev->samplers.table.bo || !dev->samplers.table.bo->va) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         const struct hk_agx3_g17p_combined_sampler_heap heap = {
            .descriptor_set_va = set_address,
            .sampler_heap_va = dev->samplers.table.bo->va->addr,
            .texture_offset_B = binding->texture_offset_B,
            .sampler_index = sampler_index,
            .sampler_count = dev->samplers.table.alloc,
            .sampler_element_size_B =
               agx_chip_sampler_heap_element_size(dev->dev.chip),
         };
         if (!hk_agx3_g17p_combined_sampler_pointers(&heap, pointers)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
      } else if (s->info.agx3_storage_image_valid) {
         /* Exact access::read_write control descriptor from the native G17P
          * Tier-2 argument buffer. Preserve its full slot and alignment. */
         static const uint8_t native_read_write_control[32] = {
            0x68, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x03, 0x1b,
         };
         const uint64_t access_control =
            hk_pool_upload(cmd, native_read_write_control,
                           sizeof(native_read_write_control), 32);
         if (!access_control) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }

         pointers[0] = set_address +
                       s->info.agx3_storage_image.texture_offset_B;
         pointers[1] = set_address + s->info.agx3_storage_image.pbe_offset_B;
         pointers[2] = access_control;
         /* The native read_write ISEL clamps each unsigned coordinate against
          * uniform half12. The only consumed vector word is 0xffff. */
         pointers[3] = UINT64_C(0xffff);
      } else {
         /* Native G17P compute read() uses table pointers in uniform
          * halves0..7, with ordinary buffer arguments beginning at half8. */
         pointers[0] = set_address + offset;
         pointers[1] = dev->rodata.bo->va->addr;
      }
      if (s->info.agx3_storage_image_valid && hk_g17p_launch_trace_enabled()) {
         fprintf(stderr,
                 "G17P storage pointers: set=0x%" PRIx64
                 " size=%u tex_off=%u pbe_off=%u slot0=0x%" PRIx64
                 " slot1=0x%" PRIx64 " slot2=0x%" PRIx64
                 " slot3=0x%" PRIx64 "\n",
                 set_address, set_size,
                 s->info.agx3_storage_image.texture_offset_B,
                 s->info.agx3_storage_image.pbe_offset_B,
                 pointers[0], pointers[1], pointers[2], pointers[3]);
      }
      const size_t pointer_count =
         s->info.agx3_storage_image_valid ? 4 : 2;
      texture_pointers =
         hk_pool_upload(cmd, pointers, pointer_count * sizeof(pointers[0]), 8);
      if (!texture_pointers) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
      }
   }

   struct agx3_ioto_uniform_range ranges[3];
   size_t range_count;
   struct agx_bo *program_bo;
   struct hk_agx3_g17p_compute_program program;
   if (!hk_g17p_build_compute_program(
          cmd, s, linked, root_ptr, texture_pointers,
          indirect ? AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1
                   : AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
          "main", ranges, &range_count, &program_bo, &program))
      return;

   if (hk_g17p_launch_trace_enabled()) {
      /* The Ioto program's whole job is to load these ranges into the uniform
       * file before the shader runs. The shader reads uniform quad u0
       * (halves 0..3) and u1 (halves 4..7) as 64-bit base pointers, so the
       * root range must start at half 0 and cover at least 8 halves, and the
       * 16 bytes at root_va must be those two pointers. If the uniform file is
       * not populated the shader addresses nowhere and stores silently into
       * the void -- which is indistinguishable, from the buffer side, from
       * never having run. */
      for (size_t r = 0; r < range_count; ++r)
         fprintf(stderr,
                 "G17P uniform range[%zu]: start_halfs=%u size_halfs=%u "
                 "address=%" PRIx64 "\n",
                 r, (unsigned)ranges[r].start_halfs,
                 (unsigned)ranges[r].size_halfs,
                 (uint64_t)ranges[r].address);
      fprintf(stderr,
              "G17P root: va=%" PRIx64 " set_count=%u push_count=%u "
              "reserved_preamble_formula=%u rodata(base=%u size16=%u)\n",
              root_ptr, (unsigned)s->info.set_count,
              (unsigned)s->b.info.push_count,
              (unsigned)(4 * (1 + s->info.set_count)),
              (unsigned)s->b.info.rodata.base_uniform,
              (unsigned)s->b.info.rodata.size_16);
      /* Exactly the bytes hk_cmd_buffer_upload_root pushed to root_ptr. The
       * first 16 are what the uniform range covers: root_desc_addr then
       * sets[0]. Two plausible GPU VAs there mean the descriptor chain the
       * shader dereferences is populated; sixteen zero bytes would settle the
       * question on their own. */
      hk_g17p_trace_line("rootdesc", root_ptr,
                         &cmd->state.cs.descriptors.root, 32);
      fprintf(stderr,
              "G17P rootdesc fields: root_desc_addr=%" PRIx64
              " sets[0]=%" PRIx64 " constant_data_addr=%" PRIx64 "\n",
              (uint64_t)cmd->state.cs.descriptors.root.root_desc_addr,
              (uint64_t)cmd->state.cs.descriptors.root.sets[0],
              (uint64_t)cmd->state.cs.descriptors.root.cs.constant_data_addr);
      hk_g17p_trace_hexdump("ioto-program", program_bo->va->addr,
                            program.bytes, program.length);
      fprintf(stderr,
              "G17P shader: va=%" PRIx64 " size=%u nr_gprs=%u scratch=%u "
              "uniform_ranges=%zu set_count=%u rodata(base=%u size16=%u "
              "off=%u)\n",
              linked->b.bo->va->addr + linked->b.entry_offset,
              (unsigned)linked->b.size,
              (unsigned)linked->b.nr_gprs, (unsigned)linked->b.scratch_size,
              range_count, (unsigned)s->info.set_count,
              (unsigned)s->b.info.rodata.base_uniform,
              (unsigned)s->b.info.rodata.size_16,
              (unsigned)s->b.info.rodata.offset);
      /* The shader binary is what actually has to terminate; a wave that never
       * stops is the one thing that produces a fault-free, completion-free
       * per-queue restart request. Dump enough to see its tail. */
      void *shader_map = agx_bo_map(linked->b.bo);
      if (shader_map) {
         hk_g17p_trace_hexdump("shader-binary",
                               linked->b.bo->va->addr + linked->b.entry_offset,
                               (uint8_t *)shader_map + linked->b.entry_offset,
                               linked->b.size < 512 ? linked->b.size : 512);
      }
   }

   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   struct agx3_compute_dispatch_sink sink = {
      .data = (uint8_t *)cs->current,
      .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
   };

   if (indirect) {
      /* The six-u32 block the helper publishes and the main record consumes.
       * It lives in the command buffer's own pool, so it is allocated per
       * dispatch and cannot be shared between two launches in flight. */
      const struct agx_ptr geometry =
         hk_pool_alloc(cmd, AGX3_G17P_INDIRECT_GEOMETRY_SIZE,
                       AGX3_G17P_INDIRECT_GEOMETRY_ALIGNMENT);
      if (!geometry.gpu) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }

      /* Zero it: an unwritten block must not read back as a launchable
       * geometry if the helper never ran. */
      memset(geometry.cpu, 0, AGX3_G17P_INDIRECT_GEOMETRY_SIZE);

      /* The helper gets its own root.  It shares the caller's descriptor
       * state -- it touches none of it -- but it needs the argument address,
       * the geometry address and the local size, and the caller's root must
       * keep describing the caller's dispatch. */
      struct hk_root_descriptor_table helper_root =
         cmd->state.cs.descriptors.root;
      helper_root.cs.group_count_addr = grid.ptr;
      helper_root.cs.grid_helper_geometry_addr = geometry.gpu;
      helper_root.cs.grid_helper_local_size[0] = local_size.x;
      helper_root.cs.grid_helper_local_size[1] = local_size.y;
      helper_root.cs.grid_helper_local_size[2] = local_size.z;

      const struct agx_ptr helper_root_ptr =
         hk_pool_alloc(cmd, sizeof(helper_root), 8);
      if (!helper_root_ptr.gpu) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
      helper_root.root_desc_addr = helper_root_ptr.gpu;
      memcpy(helper_root_ptr.cpu, &helper_root, sizeof(helper_root));

      struct agx3_ioto_uniform_range helper_ranges[3];
      size_t helper_range_count;
      struct agx_bo *helper_program_bo;
      struct hk_agx3_g17p_compute_program helper_program;
      if (!hk_g17p_build_compute_program(
             cmd, helper_s, helper_linked, helper_root_ptr.gpu, 0,
             AGX3_G17P_GRID_HELPER_LDSHDR_F1, "grid-helper", helper_ranges,
             &helper_range_count, &helper_program_bo, &helper_program))
         return;

      const struct agx3_g17p_native_indirect_dispatch dispatch = {
         .helper_program_va = helper_program_bo->va->addr,
         .geometry_va = geometry.gpu,
         /* The main record's inline dimensions are ignored in this mode; the
          * encoder still rejects zero, so publish the local size as a
          * self-consistent placeholder rather than an invented grid. */
         .inline_global_size = {local_size.x, local_size.y, local_size.z},
         .local_size = {local_size.x, local_size.y, local_size.z},
      };
      if (!hk_agx3_g17p_native_indirect_compute_dispatch_append(
             &program.launch, &dispatch, &sink)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      if (hk_g17p_launch_trace_enabled()) {
         fprintf(stderr,
                 "G17P indirect launch: args=%" PRIx64 " geometry=%" PRIx64
                 " helper_program=%" PRIx64 " local=%ux%ux%u bytes=%zu\n",
                 grid.ptr, (uint64_t)geometry.gpu,
                 helper_program_bo->va->addr, local_size.x, local_size.y,
                 local_size.z, sink.length);
         hk_g17p_trace_line("indirect-pair", (uint64_t)(uintptr_t)cs->current,
                            (const uint8_t *)sink.data, sink.length);
      }

      cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
      cs->stats.cmds++;
      return;
   }

   const uint8_t dimensions =
      grid.count[2] != 1 || local_size.z != 1
         ? 3
         : grid.count[1] != 1 || local_size.y != 1 ? 2 : 1;
   const struct agx3_compute_direct_grid direct = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {grid.count[0], grid.count[1], grid.count[2]},
      .local_size = {local_size.x, local_size.y, local_size.z},
      .dimensions = dimensions,
   };

   if (!hk_agx3_g17p_native_compute_dispatch_append(&program.launch, &direct,
                                                     &sink)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   /* One-shot ground truth for the GMMU fault forensics: the fault bank only
    * reports a 64-byte-granular address, and reconstructing it by hand from
    * dmesg has already produced two mutually inconsistent readings.  Print the
    * pointers and the exact packed words so a single run decides whether a
    * reported fault address is this dispatch's program pointer, its shader
    * pointer, or neither (i.e. a stale/phantom latch).  Off unless asked for.
    */
   if (hk_g17p_launch_trace_enabled()) {
      const uint8_t *packed = (const uint8_t *)sink.data;
      uint32_t w[4];
      for (unsigned i = 0; i < 4; ++i)
         memcpy(&w[i], packed + i * 4, 4);
      const uint64_t reconstructed =
         ((uint64_t)(w[1] >> 22) << 38) | ((uint64_t)w[2] << 6);
      /* Words 1 and 3 carry OVERLAPPING high address bits (47:38 and 47:40)
       * and the add3 witness cannot say which the hardware consumes, because
       * on its VA both encode the same 2^40.  Print the OR of both: it is the
       * worst case the GPU can compute, so `pipeline_or == program_va` is the
       * only state in which no model of the redundancy can fault. */
      const uint64_t reconstructed_or =
         reconstructed | ((uint64_t)(w[3] & 0xff) << 40);
      fprintf(stderr,
              "G17P launch: usc_window=[%" PRIx64 ",%" PRIx64 ") "
              "program_va=%" PRIx64 " shader_va=%" PRIx64
              " words=%08x %08x %08x %08x pipeline=%" PRIx64
              " pipeline_or=%" PRIx64 "%s\n",
              dev->dev.shader_base,
              dev->dev.shader_base + AGX_USC_WINDOW_SIZE,
              program_bo->va->addr, linked->b.bo->va->addr, w[0], w[1], w[2],
              w[3], reconstructed, reconstructed_or,
              reconstructed_or == program_bo->va->addr ? "" : " MISMATCH");
   }

   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   cs->stats.cmds++;
}

static void
dispatch(struct hk_cmd_buffer *cmd, struct agx_grid grid)
{
   struct hk_shader *s = hk_only_variant(cmd->state.cs.shader);
   if (agx_is_shader_empty(&s->b))
      return;

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true /* compute */);
   if (!cs)
      return;

   struct agx_workgroup local_size =
      agx_workgroup(s->b.info.workgroup_size[0], s->b.info.workgroup_size[1],
                    s->b.info.workgroup_size[2]);

   uint64_t stat = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT);

   if (hk_stat_enabled(stat)) {
      perf_debug(cmd, "CS invocation statistic");
      uint64_t grid = cmd->state.cs.descriptors.root.cs.group_count_addr;

      libagx_increment_cs_invocations(cmd, agx_1d(1), AGX_BARRIER_ALL, grid,
                                      stat, agx_workgroup_threads(local_size));
   }

   if (!agx_is_indirect(grid)) {
      grid.count[0] *= local_size.x;
      grid.count[1] *= local_size.y;
      grid.count[2] *= local_size.z;
   }

   hk_dispatch_with_local_size(cmd, cs, s, grid, local_size);
   cs->stats.calls++;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX,
                   uint32_t baseGroupY, uint32_t baseGroupZ,
                   uint32_t groupCountX, uint32_t groupCountY,
                   uint32_t groupCountZ)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   desc->root.cs.base_group[0] = baseGroupX;
   desc->root.cs.base_group[1] = baseGroupY;
   desc->root.cs.base_group[2] = baseGroupZ;

   /* We don't want to key the shader to whether we're indirectly dispatching,
    * so treat everything as indirect.
    */
   VkDispatchIndirectCommand group_count = {
      .x = groupCountX,
      .y = groupCountY,
      .z = groupCountZ,
   };

   desc->root.cs.group_count_addr =
      hk_pool_upload(cmd, &group_count, sizeof(group_count), 8);

   dispatch(cmd, agx_3d(groupCountX, groupCountY, groupCountZ));
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                       VkDeviceSize offset)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   desc->root.cs.base_group[0] = 0;
   desc->root.cs.base_group[1] = 0;
   desc->root.cs.base_group[2] = 0;

   uint64_t dispatch_addr = hk_buffer_address(buffer, offset, true);
   assert(dispatch_addr != 0);

   desc->root.cs.group_count_addr = dispatch_addr;
   dispatch(cmd, agx_grid_indirect(dispatch_addr));
}
