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
#include "hk_g17p_precomp_cache.h"
#include "hk_g17p_exact_compute_cache.h"
#include "hk_descriptor_set.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_agx3_ioto_program_core.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx3_g17p_grid_helper_core.h"
#include "util/log.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "pool.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

/* EXP-G1b's native access::write argument is a PBE pointer followed by a
 * pointer to this control descriptor, including when it follows a sampled
 * texture in the argument table. It is not the read_write clamp scalar. */
static const uint8_t hk_g17p_native_write_control[32] = {
   0x08, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
};

struct hk_g17p_compute_program_cache_key {
   uint32_t libagx_program;
   uint32_t data_size;
   uint64_t shader_va;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint32_t compiler_d;
   uint32_t threadgroup_memory_size_B;
   uint16_t nr_gprs;
   uint16_t range_start_halfs;
   uint16_t range_size_halfs;
   bool compiler_d_valid;
   enum agx_ioto_g17p_compute_ldshdr_f1 kind;
   struct agx3_cdm_direct_template_sources launch_sources;
};

struct hk_g17p_compute_program_cache_entry {
   struct hk_g17p_compute_program_cache_entry *next;
   struct hk_g17p_compute_program_cache_key key;
   struct agx3_ioto_uniform_range range;
   struct hk_agx3_g17p_compute_program program;
};

/* Hot recording needs only the published loader bytes for tracing and the
 * compact launch record. Keep the maximum-size composer object off the draw
 * stack; one such frame crosses a host page boundary and faults on the first
 * software tessellation/geometry draw. */
struct hk_g17p_compute_program_ref {
   const uint8_t *bytes;
   uint16_t length;
   struct agx3_compute_program launch;
};

struct hk_g17p_precomp_program {
   struct hk_g17p_precomp_cache_key key;
   struct agx_workgroup local_size;
   struct hk_g17p_compute_program_ref program;
};

static __attribute__((hot)) struct hk_g17p_precomp_program *
hk_g17p_find_precomp_program(struct hk_cmd_buffer *cmd,
                             enum libagx_program idx, const void *data,
                             size_t data_size)
{
   const uint32_t hash = hk_g17p_precomp_cache_hash(idx, data, data_size);
   const uint32_t entry = hk_g17p_precomp_cache_search(
      cmd->g17p_precomp_index, util_dynarray_begin(&cmd->g17p_precomp_programs),
      sizeof(struct hk_g17p_precomp_program), idx, data, data_size, hash);
   return entry ? util_dynarray_element(&cmd->g17p_precomp_programs,
                                        struct hk_g17p_precomp_program,
                                        entry - 1) : NULL;
}

static __attribute__((hot)) bool
hk_g17p_append_precomp_direct(
   struct hk_cmd_buffer *cmd, struct agx_grid grid, enum agx_barrier barrier,
   struct agx_workgroup local_size,
   const struct agx3_compute_program *program)
{
   struct hk_cs **target = (barrier & AGX_POSTGFX)  ? &cmd->current_cs.post_gfx
                           : (barrier & AGX_PREGFX) ? &cmd->current_cs.pre_gfx
                                                    : &cmd->current_cs.cs;
   struct hk_cs *cs = hk_cmd_buffer_get_cs_general(cmd, target, true);
   if (!cs)
      return true;

   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   struct agx3_compute_dispatch_sink sink = {
      .data = (uint8_t *)cs->current,
      .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
   };
   const struct agx3_compute_direct_grid direct = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {grid.count[0], grid.count[1], grid.count[2]},
      .local_size = {local_size.x, local_size.y, local_size.z},
      .dimensions = grid.count[2] != 1 || local_size.z != 1
                       ? 3
                    : grid.count[1] != 1 || local_size.y != 1 ? 2 : 1,
   };
   const uint8_t barrier_flags = (barrier & AGX_BARRIER_ALL) ? 0x5 : 0;
   if (!hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
          program, &direct, barrier_flags, &sink)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return true;
   }

   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   if (barrier_flags == 0x5) {
      cs->strong_flush_end = cs->current;
      cs->weak_barrier_end = NULL;
   } else {
      cs->weak_barrier_end = cs->current;
   }
   cs->stats.cmds++;
   return true;
}

static __attribute__((hot)) bool
hk_g17p_append_prepared_precomp_direct(
   struct hk_cmd_buffer *cmd, struct agx_grid grid, enum agx_barrier barrier,
   enum libagx_program idx, const void *data, size_t data_size)
{
   struct hk_g17p_precomp_program *cached =
      hk_g17p_find_precomp_program(cmd, idx, data, data_size);
   return cached &&
          hk_g17p_append_precomp_direct(cmd, grid, barrier,
                                         cached->local_size,
                                         &cached->program.launch);
}

void
hk_g17p_compute_program_cache_finish(struct hk_device *dev)
{
   for (unsigned i = 0; i < LIBAGX_NUM_PROGRAMS; ++i) {
      struct hk_g17p_compute_program_cache_entry *entry =
         p_atomic_read(&dev->g17p_compute_program_cache.templates[i]);
      while (entry) {
         struct hk_g17p_compute_program_cache_entry *next = entry->next;
         free(entry);
         entry = next;
      }
      dev->g17p_compute_program_cache.templates[i] = NULL;
   }
}

static bool
hk_g17p_compute_program_cache_key_equal(
   const struct hk_g17p_compute_program_cache_key *a,
   const struct hk_g17p_compute_program_cache_key *b)
{
   return a->libagx_program == b->libagx_program &&
          a->data_size == b->data_size && a->shader_va == b->shader_va &&
          a->shader_binary_size == b->shader_binary_size &&
          a->scratch_size == b->scratch_size &&
          a->compiler_d == b->compiler_d &&
          a->threadgroup_memory_size_B == b->threadgroup_memory_size_B &&
          a->nr_gprs == b->nr_gprs &&
          a->range_start_halfs == b->range_start_halfs &&
          a->range_size_halfs == b->range_size_halfs &&
          a->compiler_d_valid == b->compiler_d_valid && a->kind == b->kind &&
          a->launch_sources.ti_spill_bytes ==
             b->launch_sources.ti_spill_bytes &&
          a->launch_sources.ti_largest_ipr_bytes ==
             b->launch_sources.ti_largest_ipr_bytes &&
          a->launch_sources.ray_traversal_depth_nonzero ==
             b->launch_sources.ray_traversal_depth_nonzero;
}

static void **
hk_g17p_compute_program_cache_head(
   struct hk_device *dev, struct hk_linked_shader *linked,
   const struct hk_g17p_compute_program_cache_key *key)
{
   if (linked)
      return &linked->g17p_compute_templates;

   if (key->libagx_program >= LIBAGX_NUM_PROGRAMS)
      return NULL;

   return &dev->g17p_compute_program_cache.templates[key->libagx_program];
}

void
hk_g17p_linked_compute_cache_finish(struct hk_linked_shader *linked)
{
   struct hk_g17p_compute_program_cache_entry *entry =
      p_atomic_read(&linked->g17p_compute_templates);
   while (entry) {
      struct hk_g17p_compute_program_cache_entry *next = entry->next;
      free(entry);
      entry = next;
   }
   linked->g17p_compute_templates = NULL;
}

static bool
hk_g17p_compute_program_cache_lookup(
   struct hk_device *dev, struct hk_linked_shader *linked,
   const struct hk_g17p_compute_program_cache_key *key,
   const struct hk_agx3_g17p_compute_program_desc *desc,
   void *program_map, size_t program_capacity,
   struct hk_g17p_compute_program_ref *program)
{
   void **head = hk_g17p_compute_program_cache_head(dev, linked, key);
   if (!head)
      return false;

   struct hk_g17p_compute_program_cache_entry *entry = p_atomic_read(head);
   for (; entry; entry = entry->next) {
      if (hk_g17p_compute_program_cache_key_equal(&entry->key, key) &&
          hk_agx3_g17p_compute_program_relocate_single_range_to(
             AGX_CHIP_G17P, desc, &entry->range, &entry->program,
             program_map, program_capacity, &program->length,
             &program->launch)) {
         program->bytes = program_map;
         return true;
      }
   }
   return false;
}

static void
hk_g17p_compute_program_cache_store(
   struct hk_device *dev, struct hk_linked_shader *linked,
   const struct hk_g17p_compute_program_cache_key *key,
   const struct hk_agx3_g17p_compute_program_desc *desc,
   const struct hk_agx3_g17p_compute_program *program);

static ATTRIBUTE_NOINLINE bool
hk_g17p_compute_program_generate_slow(
   struct hk_device *dev, struct hk_linked_shader *linked,
   const struct hk_g17p_compute_program_cache_key *key, bool cacheable,
   const struct hk_agx3_g17p_compute_program_desc *desc, void *program_map,
   struct hk_g17p_compute_program_ref *program)
{
   struct hk_agx3_g17p_compute_program generated;
   if (!hk_agx3_g17p_compute_program_generate(dev->dev.chip, desc,
                                               &generated))
      return false;

   if (cacheable)
      hk_g17p_compute_program_cache_store(dev, linked, key, desc, &generated);

   memcpy(program_map, generated.bytes, generated.length);
   program->bytes = program_map;
   program->length = generated.length;
   program->launch = generated.launch;
   return true;
}

static void
hk_g17p_compute_program_cache_store(
   struct hk_device *dev, struct hk_linked_shader *linked,
   const struct hk_g17p_compute_program_cache_key *key,
   const struct hk_agx3_g17p_compute_program_desc *desc,
   const struct hk_agx3_g17p_compute_program *program)
{
   void **head = hk_g17p_compute_program_cache_head(dev, linked, key);
   if (!head)
      return;

   struct hk_g17p_compute_program_cache_entry *candidate =
      malloc(sizeof(*candidate));
   if (!candidate)
      return;

   candidate->key = *key;
   candidate->range = desc->uniform_ranges[0];
   candidate->program = *program;

   while (true) {
      struct hk_g17p_compute_program_cache_entry *entry = p_atomic_read(head);
      for (struct hk_g17p_compute_program_cache_entry *it = entry; it;
           it = it->next) {
         struct hk_agx3_g17p_compute_program relocated;
         if (hk_g17p_compute_program_cache_key_equal(&it->key, key) &&
             hk_agx3_g17p_compute_program_relocate_single_range(
                AGX_CHIP_G17P, desc, &it->range, &it->program, &relocated)) {
            free(candidate);
            return;
         }
      }

      candidate->next = entry;
      if (p_atomic_cmpxchg_ptr(head, entry, candidate) == entry)
         return;
   }
}

static bool
hk_g17p_precompiled_uniform_data_size(
   const struct agx_precompiled_shader *prog, size_t argument_size,
   size_t *uniform_data_size)
{
   if (!prog->info.printf_uniform) {
      *uniform_data_size = argument_size;
      return true;
   }

   if (prog->info.printf_uniform < AGX_ABI_AGX3_PRECOMP_ARG_HALFS)
      return false;

   const size_t printf_offset_B =
      (size_t)(prog->info.printf_uniform - AGX_ABI_AGX3_PRECOMP_ARG_HALFS) * 2;
   *uniform_data_size =
      MAX2(argument_size, printf_offset_B + sizeof(uint64_t));
   return true;
}

static bool
hk_g17p_compute_program_cache_prewarm_kind(
   struct hk_device *dev, enum libagx_program idx, size_t data_size,
   enum agx_ioto_g17p_compute_ldshdr_f1 kind)
{
   struct agx_precompiled_shader *prog =
      agx_get_precompiled(&dev->bg_eot, idx);
   if (!prog || prog->info.rodata.size_16)
      return prog != NULL;

   size_t uniform_data_size;
   if (!hk_g17p_precompiled_uniform_data_size(prog, data_size,
                                               &uniform_data_size))
      return false;

   uint32_t uploaded_size;
   uint16_t uniform_size_halfs;
   if (!hk_agx3_g17p_precompiled_data_layout(
          uniform_data_size, &uploaded_size, &uniform_size_halfs))
      return false;

   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(false, false, false,
                                                  &launch_sources))
      return false;

   const uint32_t threadgroup_memory_size_B =
      prog->info.local_size
         ? MAX2(256u, ALIGN_POT(prog->info.local_size, 64u))
         : 0;
   const struct agx3_ioto_uniform_range range = {
      .start_halfs = AGX_ABI_AGX3_PRECOMP_ARG_HALFS,
      .size_halfs = uniform_size_halfs,
      /* Only the layout is cached. Both this address and program_va are
       * relocated to command-buffer storage on every lookup. */
      /* Use the full-width encoding shape of a normal G17P main-heap upload.
       * Relocation requires the two LDIMM instruction lengths to stay fixed;
       * an all-zero low dword would select a shorter template that cannot be
       * patched to real command-buffer addresses. */
      .address = UINT64_C(0x5a89abcde0),
   };
   const struct hk_agx3_g17p_compute_program_desc desc = {
      .program_va = dev->dev.shader_base + 0x10000,
      .shader_va = prog->ptr + prog->entry_offset,
      .uniform_ranges = &range,
      .uniform_range_count = 1,
      .shader_binary_size = prog->info.main_size,
      .scratch_size = prog->info.agx3_tps_size,
      .nr_gprs = prog->info.nr_gprs,
      .compiler_d_valid = true,
      .compiler_d = (uint32_t)prog->info.agx3_staging_halfs * 2,
      .threadgroup_memory_size_B = threadgroup_memory_size_B,
      .launch_sources = launch_sources,
      .ldshdr_f1 = kind,
   };
   const struct hk_g17p_compute_program_cache_key key = {
      .libagx_program = idx,
      .data_size = uploaded_size,
      .shader_va = desc.shader_va,
      .shader_binary_size = desc.shader_binary_size,
      .scratch_size = desc.scratch_size,
      .compiler_d = desc.compiler_d,
      .threadgroup_memory_size_B = desc.threadgroup_memory_size_B,
      .nr_gprs = desc.nr_gprs,
      .range_start_halfs = range.start_halfs,
      .range_size_halfs = range.size_halfs,
      .compiler_d_valid = desc.compiler_d_valid,
      .kind = kind,
      .launch_sources = launch_sources,
   };

   uint8_t relocated[HK_AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE];
   struct hk_g17p_compute_program_ref relocated_program;
   struct hk_agx3_g17p_compute_program generated;
   if (hk_g17p_compute_program_cache_lookup(
          dev, NULL, &key, &desc, relocated, sizeof(relocated),
          &relocated_program))
      return true;

   if (!hk_agx3_g17p_compute_program_generate(dev->dev.chip, &desc,
                                               &generated))
      return false;

   hk_g17p_compute_program_cache_store(dev, NULL, &key, &desc, &generated);
   return true;
}

bool
hk_g17p_compute_program_cache_prewarm(struct hk_device *dev,
                                      enum libagx_program program,
                                      size_t data_size)
{
   return hk_g17p_compute_program_cache_prewarm_kind(
             dev, program, data_size,
             AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3) &&
          hk_g17p_compute_program_cache_prewarm_kind(
             dev, program, data_size,
             AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1);
}

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
      /* A strong flush already ends the stream: every launch has one in its
       * tail, and a second barrier with no launch between them is not a
       * shape Apple's CDM emitter produces. */
      if (cs->current == cs->strong_flush_end)
         return;

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

      /* Native AGX3 launches carry a weak ordering barrier in their tail.
       * When an API operation immediately needs a cache flush, strengthen
       * that record in place. Apple's emitter does not produce adjacent weak
       * and strong barriers, and retaining the weak record buys no ordering
       * while making the control stream larger. */
      if (cs->current == cs->weak_barrier_end) {
         memcpy((uint8_t *)cs->current - sizeof(packed), packed,
                sizeof(packed));
         cs->strong_flush_end = cs->current;
         cs->weak_barrier_end = NULL;
         cs->stats.flushes++;
         return;
      }

      const bool at_stream_start = cs->current == cs->start;
      memcpy(cs->current, packed, sizeof(packed));
      cs->current = (uint32_t *)((uint8_t *)cs->current + sizeof(packed));
      cs->strong_flush_end = cs->current;
      cs->weak_barrier_end = NULL;
      /* A barrier that opens the stream is the post-graphics strong flush.
       * It is canonical here and only here; record it so a later inline
       * concatenation can fold it into the preceding stream's trailing
       * barrier instead of leaving the two adjacent. */
      if (at_stream_start)
         cs->leading_barrier_end = cs->current;
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

/* A kernel that samples (indexed sampler, texel fetch or storage image) takes
 * a texture pointer table built from the draw's descriptors at dispatch time
 * (hk_dispatch_g17p_native()). Bind-time preparation cannot know that table,
 * so it must not build launch programs or dispatch templates for such a
 * kernel: with no table the uniform-range builder refuses the kernel, and a
 * template without the table would be wrong if it were ever consumed. */
static bool
hk_g17p_compute_uses_texture_pointers(const struct hk_shader *s)
{
   return s && (s->info.agx3_txf_texture_valid ||
                s->info.agx3_storage_image_valid ||
                s->info.agx3_sampled_storage_valid ||
                s->info.agx3_indexed_storage_valid ||
                s->info.agx3_indexed_sampler_valid);
}

static bool
hk_g17p_compute_shader_supported(struct hk_shader *s,
                                 struct hk_linked_shader *linked)
{
   if (s && linked && hk_g17p_launch_trace_enabled())
      fprintf(stderr,"G17P compute resources: txf=%u binding=%u storage=%u sampled_storage=%u preamble=%u local=%u texture=%u sampler=%u heap=%u scratch=%u/%u idx_samp=%u idx_store=%u combined=%u stage=%u/%u bo=%u va=%u size=%u gprs=%u pre_scratch=%u staging=%u sets=%u\n",
         linked->b.uses_txf,s->info.agx3_txf_texture_valid,
         s->info.agx3_storage_image_valid,s->info.agx3_sampled_storage_valid,
         s->b.info.has_preamble,
         s->b.info.local_size,s->b.info.texture_state_count,s->b.info.sampler_state_count,
         s->b.info.uses_sampler_heap,s->b.info.scratch_size,linked->b.scratch_size,
         s->info.agx3_indexed_sampler_valid,s->info.agx3_indexed_storage_valid,
         s->info.agx3_combined_sampler_valid,
         (unsigned)s->b.info.stage,(unsigned)s->info.stage,
         linked->b.bo != NULL, linked->b.bo && linked->b.bo->va != NULL,
         (unsigned)linked->b.size, linked->b.nr_gprs,
         s->b.info.preamble_scratch_size, s->b.info.agx3_staging_known,
         s->info.set_count);
   /* poly_nir_lower_gs preserves the API geometry stage on the main kernel,
    * even though Honeykrisp has always launched that binary through CDM. */
   return s && linked &&
          (s->b.info.stage == MESA_SHADER_COMPUTE ||
           s->info.stage == MESA_SHADER_GEOMETRY ||
           s->info.stage == MESA_SHADER_TESS_CTRL) &&
          linked->b.bo && linked->b.bo->va && linked->b.size != 0 &&
          linked->b.size <= UINT32_MAX && linked->b.nr_gprs != 0 &&
          !s->b.info.has_preamble && s->b.info.preamble_scratch_size == 0 &&
          s->b.info.scratch_size <= AGX3_G17P_MAX_SPILL_B &&
          linked->b.scratch_size <= AGX3_G17P_MAX_SPILL_B &&
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

/* Software VS adds an indexing-parameter range to the seven hardware-VS
 * ranges. Keep its eight-range collectors and exact-program cache together;
 * the compute composer independently bounds the generated load program. */
#define HK_G17P_COMPUTE_MAX_RANGES 8u

/* Uniform ABI for compute-emulated vertex work.  Geometry variants use the
 * ordinary root-at-zero compute ABI; the software vertex variant additionally
 * consumes the same VBO, draw-parameter and vertex-parameter ranges that the
 * legacy USC builder uploads.  Ioto copies these byte-identical ranges into
 * the uniform file before launching the compute-stage binary. */
static bool
hk_g17p_compute_uniform_ranges(
   struct hk_shader *s, uint64_t root_ptr, uint64_t texture_pointers,
   struct agx3_ioto_uniform_range ranges[3], size_t *range_count);

static bool
hk_g17p_internal_graphics_uniform_ranges(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct hk_linked_shader *linked, uint64_t root_ptr,
   uint64_t texture_pointers,
   struct agx3_ioto_uniform_range
      ranges[HK_G17P_COMPUTE_MAX_RANGES],
   size_t *range_count)
{
   /* Uniform DMA (v51a/b) is only enabled for hardware graphics stages;
    * this internal path cannot publish its span, so fail closed. */
   if (s->b.info.agx3_uniform_dma_valid)
      return false;

   if (s->info.stage != MESA_SHADER_VERTEX)
      return hk_g17p_compute_uniform_ranges(s, root_ptr, texture_pointers,
                                             ranges, range_count);

   if (!!texture_pointers != s->info.agx3_indexed_sampler_valid)
      return false;
   const unsigned uniform_base = s->info.agx3_indexed_sampler_valid
      ? AGX_ABI_G17P_TEXTURE_POINTER_HALFS : 0;

   const unsigned nr_vbos = DIV_ROUND_UP(
      BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);
   size_t count = 0;
   if (texture_pointers) {
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0,
         .size_halfs = uniform_base,
         .address = texture_pointers,
      };
   }
   if (nr_vbos) {
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = uniform_base + AGX_ABI_VUNI_VBO_BASE(0),
         .size_halfs = 4 * nr_vbos,
         .address = root_ptr + hk_root_descriptor_offset(draw.attrib_base),
      };
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = uniform_base + AGX_ABI_VUNI_VBO_CLAMP(nr_vbos, 0),
         .size_halfs = 2 * nr_vbos,
         .address = root_ptr + hk_root_descriptor_offset(draw.attrib_clamps),
      };
   }
   if (cmd->state.gfx.draw_params) {
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = uniform_base + AGX_ABI_VUNI_FIRST_VERTEX(nr_vbos),
         .size_halfs = 4,
         .address = cmd->state.gfx.draw_params,
      };
   }
   if (cmd->state.gfx.draw_id_ptr) {
      /* A uniform range is loaded in 32-bit dwords and an odd size_halfs is
       * rejected outright, so publish the containing dword. See the same
       * range in hk_cmd_buffer.c. */
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = uniform_base + AGX_ABI_VUNI_DRAW_ID(nr_vbos),
         .size_halfs = 2,
         .address = cmd->state.gfx.draw_id_ptr,
      };
   }
   if (linked->sw_indexing) {
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = uniform_base + AGX_ABI_VUNI_VERTEX_PARAMS(nr_vbos),
         .size_halfs = 4,
         .address = root_ptr + hk_root_descriptor_offset(draw.vertex_params),
      };
   }

   const unsigned root_uniform =
      uniform_base + AGX_ABI_VUNI_COUNT_VK(nr_vbos);
   if (s->info.agx3_uses_root || s->info.set_count) {
      if (count >= HK_G17P_COMPUTE_MAX_RANGES ||
          !hk_agx3_g17p_root_range(root_uniform, s->info.set_count, root_ptr,
                                   &ranges[count]))
         return false;
      count++;
   }

   if (s->b.info.rodata.size_16) {
      const struct agx_rodata *rodata = &s->b.info.rodata;
      if (count >= HK_G17P_COMPUTE_MAX_RANGES || !s->bo || !s->bo->va ||
          !hk_agx3_g17p_rodata_range(
             rodata->base_uniform, rodata->size_16, s->bo->va->addr,
             rodata->offset, s->bo->size, &ranges[count]))
         return false;
      count++;
   }

   *range_count = count;
   return true;
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
   /* Uniform DMA is never enabled for compute (see hk_shader.c). */
   if (s->b.info.agx3_uniform_dma_valid)
      return false;

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

   size_t count = 1;
   if (texture_pointers) {
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0, .size_halfs = root_uniform,
         .address = texture_pointers,
      };
   }
   const struct agx_rodata *rodata = &s->b.info.rodata;
   if (rodata->size_16 != 0) {
      if (!s->bo || !s->bo->va ||
          !hk_agx3_g17p_rodata_range(
             rodata->base_uniform, rodata->size_16, s->bo->va->addr,
             rodata->offset, s->bo->size, &ranges[count]))
         return false;
      count++;
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
   struct hk_cmd_buffer *cmd, uint32_t cache_program, uint32_t cache_data_size,
   struct hk_linked_shader *cache_linked,
   uint64_t shader_va,
   uint32_t shader_binary_size, uint32_t scratch_size, uint16_t nr_gprs,
   bool compiler_d_valid, uint32_t compiler_d,
   uint32_t threadgroup_memory_size_B,
   const struct agx3_cdm_direct_template_sources *launch_sources,
   enum agx_ioto_g17p_compute_ldshdr_f1 kind, const char *trace,
   const struct agx3_ioto_uniform_range *ranges, size_t range_count,
   uint64_t *program_va_out,
   struct hk_g17p_compute_program_ref *out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!shader_va || !shader_binary_size || !nr_gprs || !compiler_d_valid ||
       !launch_sources || !program_va_out || !out ||
       (range_count && !ranges)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   struct hk_ioto_reservation reservation;
   if (!hk_cmd_reserve_ioto_program(cmd,
                                    HK_AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE,
                                    "G17P compute Ioto program",
                                    &reservation))
      return false;

   void *program_map = reservation.map;
   const uint64_t program_va = reservation.va;

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
   if (!agx_va_in_usc_window(&dev->dev, program_va) ||
       !agx_va_in_usc_window(&dev->dev, shader_va) ||
       shader_binary_size >
          AGX_USC_WINDOW_SIZE - (shader_va - dev->dev.shader_base)) {
      mesa_loge("G17P dispatch: exec pointer outside the USC window "
                "[%" PRIx64 ", %" PRIx64 "): program=%" PRIx64
                " shader=%" PRIx64 "\n",
                dev->dev.shader_base,
                dev->dev.shader_base + AGX_USC_WINDOW_SIZE,
                program_va, shader_va);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
      return false;
   }

   const struct hk_agx3_g17p_compute_program_desc desc = {
      .program_va = program_va,
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

   /* STK reaches this through hk_dispatch_precomp_g17p: vk_meta reuses the
    * libagx shader, but every upload has a fresh argument allocation. Cache
    * the one-range structural program on the device (the same lifetime as
    * bg_eot) and patch only that address and the local launch pointer. */
   const bool cacheable =
      (cache_program != UINT32_MAX || cache_linked) && range_count == 1 &&
      ranges[0].size_halfs > 0 && ranges[0].size_halfs <= 128 &&
      !(ranges[0].start_halfs & 1) && !(ranges[0].size_halfs & 1) &&
      !(ranges[0].address & 3);
   struct hk_g17p_compute_program_cache_key cache_key = {0};
   bool cache_hit = false;
   if (cacheable) {
      cache_key = (struct hk_g17p_compute_program_cache_key){
         .libagx_program = cache_program,
         .data_size = cache_data_size,
         .shader_va = shader_va,
         .shader_binary_size = shader_binary_size,
         .scratch_size = scratch_size,
         .compiler_d = compiler_d,
         .threadgroup_memory_size_B = threadgroup_memory_size_B,
         .nr_gprs = nr_gprs,
         .range_start_halfs = ranges[0].start_halfs,
         .range_size_halfs = ranges[0].size_halfs,
         .compiler_d_valid = compiler_d_valid,
         .kind = kind,
         .launch_sources = *launch_sources,
      };
      cache_hit = hk_g17p_compute_program_cache_lookup(
         dev, cache_linked, &cache_key, &desc, program_map,
         HK_AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE, out);
   }

   if (!cache_hit) {
      if (!hk_g17p_compute_program_generate_slow(
             dev, cache_linked, &cache_key, cacheable, &desc, program_map,
             out)) {
         mesa_loge("G17P compute program rejected (%s): binary=%u scratch=%u "
                   "gprs=%u compiler_d_valid=%u compiler_d=%u shared=%u "
                   "kind=%u ranges=%zu program=%" PRIx64 " shader=%" PRIx64,
                   trace ? trace : "unknown", shader_binary_size, scratch_size,
                   nr_gprs, compiler_d_valid, compiler_d,
                   threadgroup_memory_size_B, (unsigned)kind, range_count,
                   program_va, shader_va);
         for (size_t i = 0; i < range_count; ++i)
            mesa_loge("G17P compute range[%zu]: start=%u size=%u address=%" PRIx64,
                      i, ranges[i].start_halfs, ranges[i].size_halfs,
                      ranges[i].address);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }
   }

   const uint64_t published =
      hk_cmd_publish_ioto_program(cmd, &reservation, out->length);
   if (!published)
      return false;

   /* Publication can deduplicate against an earlier loader and give the
    * reservation back to the arena. Both direct and indirect launches must
    * name the retained loader before that reservation is overwritten. */
   if (!hk_agx3_g17p_compute_program_rebind_published(&out->launch, published)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
      return false;
   }


   *program_va_out = published;
   if (trace && hk_g17p_launch_trace_enabled())
      fprintf(stderr, "G17P program[%s]: kind=%u va=%" PRIx64 " len=%u\n",
              trace, (unsigned)kind, published, (unsigned)out->length);
   return true;
}

struct hk_g17p_exact_compute_program {
   struct hk_g17p_exact_compute_cache_entry cache;
   size_t range_count;
   struct agx3_ioto_uniform_range
      ranges[HK_G17P_COMPUTE_MAX_RANGES];
   uint64_t program_va;
   const uint8_t *program_bytes;
   uint16_t program_length;
   struct agx3_compute_program launch;
};

static struct hk_g17p_exact_compute_key
hk_g17p_exact_compute_key(struct hk_cmd_buffer *cmd, const struct hk_shader *s,
                          const struct hk_linked_shader *linked,
                          uint64_t root_ptr, uint64_t texture_pointers,
                          bool internal_graphics,
                          enum agx_ioto_g17p_compute_ldshdr_f1 kind)
{
   return (struct hk_g17p_exact_compute_key){.words = {
      (uintptr_t)s, (uintptr_t)linked, root_ptr, texture_pointers,
      s->info.stage == MESA_SHADER_VERTEX ? cmd->state.gfx.draw_params : 0,
      s->info.stage == MESA_SHADER_VERTEX ? cmd->state.gfx.draw_id_ptr : 0,
      kind, internal_graphics,
   }};
}

static struct hk_g17p_exact_compute_program *
hk_g17p_find_exact_compute_program(
   struct hk_cmd_buffer *cmd, const struct hk_shader *s,
   const struct hk_linked_shader *linked, uint64_t root_ptr,
   uint64_t texture_pointers, bool internal_graphics,
   enum agx_ioto_g17p_compute_ldshdr_f1 kind)
{
   const struct hk_g17p_exact_compute_key key = hk_g17p_exact_compute_key(
      cmd, s, linked, root_ptr, texture_pointers, internal_graphics, kind);
   const uint32_t hash = hk_g17p_exact_compute_cache_hash(&key);
   const uint32_t hit = hk_g17p_exact_compute_cache_search(
      cmd->g17p_compute_index, util_dynarray_begin(&cmd->g17p_compute_programs),
      sizeof(struct hk_g17p_exact_compute_program), &key, hash);
   return hit ? util_dynarray_element(&cmd->g17p_compute_programs,
                                      struct hk_g17p_exact_compute_program,
                                      hit - 1) : NULL;
}

static bool
hk_g17p_build_compute_program(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                              struct hk_linked_shader *linked,
                              uint64_t root_ptr, uint64_t texture_pointers,
                              bool internal_graphics,
                              enum agx_ioto_g17p_compute_ldshdr_f1 kind,
                              const char *trace,
                              struct agx3_ioto_uniform_range
                                 ranges[HK_G17P_COMPUTE_MAX_RANGES],
                              size_t *range_count_out,
                              uint64_t *program_va_out,
                              struct hk_g17p_compute_program_ref *out)
{
   const struct hk_g17p_exact_compute_key key = hk_g17p_exact_compute_key(
      cmd, s, linked, root_ptr, texture_pointers, internal_graphics, kind);
   const uint32_t hash = hk_g17p_exact_compute_cache_hash(&key);
   const uint32_t hit = hk_g17p_exact_compute_cache_search(
      cmd->g17p_compute_index, util_dynarray_begin(&cmd->g17p_compute_programs),
      sizeof(struct hk_g17p_exact_compute_program), &key, hash);
   struct hk_g17p_exact_compute_program *cached = hit
      ? util_dynarray_element(&cmd->g17p_compute_programs,
                              struct hk_g17p_exact_compute_program, hit - 1)
      : NULL;
   if (cached) {
      memcpy(ranges, cached->ranges,
             cached->range_count * sizeof(cached->ranges[0]));
      *range_count_out = cached->range_count;
      *program_va_out = cached->program_va;
      out->bytes = cached->program_bytes;
      out->length = cached->program_length;
      out->launch = cached->launch;
      return true;
   }

   size_t range_count;
   const bool ranges_ok = internal_graphics
      ? hk_g17p_internal_graphics_uniform_ranges(
           cmd, s, linked, root_ptr, texture_pointers, ranges, &range_count)
      : hk_g17p_compute_uniform_ranges(s, root_ptr, texture_pointers, ranges,
                                        &range_count);
   if (!ranges_ok) {
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
      cmd, UINT32_MAX, 0, linked,
      linked->b.bo->va->addr + linked->b.entry_offset, linked->b.size,
      MAX2(linked->b.scratch_size, linked->b.agx3_spectpr_scratch_size),
      linked->b.nr_gprs,
      s->b.info.agx3_staging_known,
      (uint32_t)s->b.info.agx3_staging_halfs * 2,
      s->b.info.local_size ? MAX2(256u, ALIGN_POT(s->b.info.local_size, 64u))
                           : 0,
      &launch_sources, kind, trace, ranges, range_count, program_va_out, out);
   if (ok) {
      *range_count_out = range_count;
      const struct hk_g17p_exact_compute_program exact = {
         .cache.key = key,
         .range_count = range_count,
         .program_va = *program_va_out,
      };
      struct hk_g17p_exact_compute_program *stored =
         util_dynarray_grow(&cmd->g17p_compute_programs,
                            struct hk_g17p_exact_compute_program, 1);
      if (!stored) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return false;
      }
      *stored = exact;
      stored->program_bytes = out->bytes;
      stored->program_length = out->length;
      stored->launch = out->launch;
      memcpy(stored->ranges, ranges,
             range_count * sizeof(stored->ranges[0]));
      const uint32_t position = util_dynarray_num_elements(
         &cmd->g17p_compute_programs, struct hk_g17p_exact_compute_program);
      if (!hk_g17p_exact_compute_cache_insert(
             &cmd->g17p_compute_index, &stored->cache, position, hash)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return false;
      }
   }
   return ok;
}

static void
hk_g17p_prewarm_linked_compute_program_kind(
   struct hk_device *dev, struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct hk_linked_shader *linked, bool internal_graphics,
   enum agx_ioto_g17p_compute_ldshdr_f1 kind)
{
   if (dev->dev.chip != AGX_CHIP_G17P ||
       !hk_g17p_compute_shader_supported(s, linked))
      return;

   /* Use ordinary full-width, aligned addresses so the cached template has
    * the same LDIMM encoding shape as command-buffer uploads. Only the range
    * layout is retained; every real address is relocated at lookup. */
   const uint64_t dummy_root = UINT64_C(0x5a89abcde0);
   const bool uses_texture_pointers =
      s->info.agx3_txf_texture_valid ||
      s->info.agx3_storage_image_valid ||
      s->info.agx3_sampled_storage_valid ||
      s->info.agx3_indexed_storage_valid ||
      s->info.agx3_indexed_sampler_valid;
   const uint64_t dummy_texture_pointers =
      uses_texture_pointers ? dummy_root + 0x2000 : 0;

   struct agx3_ioto_uniform_range
      ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t range_count;
   const bool vertex_graphics =
      internal_graphics && s->info.stage == MESA_SHADER_VERTEX;
   if (vertex_graphics && !cmd)
      return;

   const bool ranges_ok = vertex_graphics
      ? hk_g17p_internal_graphics_uniform_ranges(
           cmd, s, linked, dummy_root, dummy_texture_pointers, ranges,
           &range_count)
      : hk_g17p_compute_uniform_ranges(s, dummy_root,
                                        dummy_texture_pointers, ranges,
                                        &range_count);
   if (!ranges_ok || range_count != 1 || ranges[0].size_halfs == 0 ||
       ranges[0].size_halfs > 128 || (ranges[0].start_halfs & 1) ||
       (ranges[0].size_halfs & 1) || (ranges[0].address & 3))
      return;

   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(
          s->b.info.uses_resource_info_query,
          s->b.info.uses_ray_traversal,
          s->b.info.ray_traversal_depth_nonzero, &launch_sources))
      return;

   const uint32_t threadgroup_memory_size_B =
      s->b.info.local_size
         ? MAX2(256u, ALIGN_POT(s->b.info.local_size, 64u))
         : 0;
   const struct hk_agx3_g17p_compute_program_desc desc = {
      .program_va = dev->dev.shader_base + 0x10000,
      .shader_va = linked->b.bo->va->addr + linked->b.entry_offset,
      .uniform_ranges = ranges,
      .uniform_range_count = 1,
      .shader_binary_size = linked->b.size,
      .scratch_size =
         MAX2(linked->b.scratch_size, linked->b.agx3_spectpr_scratch_size),
      .nr_gprs = linked->b.nr_gprs,
      .compiler_d_valid = s->b.info.agx3_staging_known,
      .compiler_d = (uint32_t)s->b.info.agx3_staging_halfs * 2,
      .threadgroup_memory_size_B = threadgroup_memory_size_B,
      .launch_sources = launch_sources,
      .ldshdr_f1 = kind,
   };
   const struct hk_g17p_compute_program_cache_key key = {
      .libagx_program = UINT32_MAX,
      .shader_va = desc.shader_va,
      .shader_binary_size = desc.shader_binary_size,
      .scratch_size = desc.scratch_size,
      .compiler_d = desc.compiler_d,
      .threadgroup_memory_size_B = desc.threadgroup_memory_size_B,
      .nr_gprs = desc.nr_gprs,
      .range_start_halfs = ranges[0].start_halfs,
      .range_size_halfs = ranges[0].size_halfs,
      .compiler_d_valid = desc.compiler_d_valid,
      .kind = kind,
      .launch_sources = launch_sources,
   };

   uint8_t relocated[HK_AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE];
   struct hk_g17p_compute_program_ref relocated_program;
   struct hk_agx3_g17p_compute_program generated;
   if (hk_g17p_compute_program_cache_lookup(
          dev, linked, &key, &desc, relocated, sizeof(relocated),
          &relocated_program))
      return;

   if (hk_agx3_g17p_compute_program_generate(dev->dev.chip, &desc,
                                              &generated))
      hk_g17p_compute_program_cache_store(dev, linked, &key, &desc,
                                           &generated);
}

void
hk_g17p_prewarm_linked_compute_program(struct hk_cmd_buffer *cmd,
                                        struct hk_shader *s,
                                        struct hk_linked_shader *linked,
                                        bool internal_graphics)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (hk_g17p_compute_shader_supported(s, linked)) {
      /* The recorded Ioto program will dereference these BOs. Take the
       * command-buffer references alongside prewarming, outside the first
       * vkCmdDraw, and let the normal retain calls deduplicate. */
      hk_cmd_buffer_retain_bo(cmd, linked->b.bo);
      if (s->b.info.rodata.size_16 != 0)
         hk_cmd_buffer_retain_bo(cmd, s->bo);
   }

   hk_g17p_prewarm_linked_compute_program_kind(
      dev, cmd, s, linked, internal_graphics,
      AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3);
   hk_g17p_prewarm_linked_compute_program_kind(
      dev, cmd, s, linked, internal_graphics,
      AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1);

   /* Once the graphics root is final, publish the exact direct and indirect
    * caller loaders into this command buffer's executable arena. The grid
    * helper remains keyed to its draw-local argument address. A sampling
    * kernel's loader depends on the draw's texture table; the draw builds it. */
   if (internal_graphics && cmd->state.gfx.root &&
       !hk_g17p_compute_uses_texture_pointers(s) &&
       hk_g17p_compute_shader_supported(s, linked)) {
      const enum agx_ioto_g17p_compute_ldshdr_f1 kinds[] = {
         AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
         AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1,
      };
      for (unsigned i = 0; i < ARRAY_SIZE(kinds); ++i) {
         struct agx3_ioto_uniform_range
            ranges[HK_G17P_COMPUTE_MAX_RANGES];
         size_t range_count;
         uint64_t program_va;
         struct hk_g17p_compute_program_ref program;
         hk_g17p_build_compute_program(
            cmd, s, linked, cmd->state.gfx.root, 0, true, kinds[i],
            "prewarm", ranges, &range_count, &program_va, &program);
      }
   }
}

bool
hk_g17p_prepare_graphics_dispatch(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct hk_linked_shader *linked, uint64_t indirect_local_grid,
   struct agx_workgroup indirect_local_size,
   struct hk_g17p_prepared_dispatch *out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   *out = (struct hk_g17p_prepared_dispatch){0};

   if (dev->dev.chip != AGX_CHIP_G17P || !cmd->state.gfx.root ||
       hk_g17p_compute_uses_texture_pointers(s) ||
       !hk_g17p_compute_shader_supported(s, linked))
      return false;

   struct agx3_ioto_uniform_range ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t range_count;
   uint64_t program_va;
   struct hk_g17p_compute_program_ref program;
   if (!hk_g17p_build_compute_program(
          cmd, s, linked, cmd->state.gfx.root, 0, true,
          AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3, "prepare-draw",
          ranges, &range_count, &program_va, &program))
      return false;

   out->program = program.launch;
   out->shader = s;
   out->linked = linked;
   out->root = cmd->state.gfx.root;
   out->draw_params = cmd->state.gfx.draw_params;
   out->draw_id_ptr = cmd->state.gfx.draw_id_ptr;

   const struct agx3_compute_direct_grid template_grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {1, 1, 1},
      .local_size = {1, 1, 1},
      .dimensions = 3,
   };
   struct agx3_compute_dispatch_sink direct_sink = {
      .data = out->direct_template,
      .capacity = sizeof(out->direct_template),
   };
   if (!hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
          &out->program, &template_grid, 0, &direct_sink) ||
       direct_sink.length != sizeof(out->direct_template))
      return false;

   out->valid = true;

   if (!indirect_local_grid)
      return true;

   struct hk_g17p_compute_program_ref indirect_main;
   if (!hk_g17p_build_compute_program(
          cmd, s, linked, cmd->state.gfx.root, 0, true,
          AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1, "prepare-indirect-local", ranges,
          &range_count, &program_va, &indirect_main))
      return true;

   out->indirect_local_grid = indirect_local_grid;
   out->indirect_local_size[0] = indirect_local_size.x;
   out->indirect_local_size[1] = indirect_local_size.y;
   out->indirect_local_size[2] = indirect_local_size.z;

   struct hk_shader *helper_s = dev->g17p_geometry_copy_helper
      ? hk_only_variant(dev->g17p_geometry_copy_helper)
      : NULL;
   if (!helper_s ||
       !hk_g17p_compute_shader_supported(helper_s, helper_s->only_linked))
      return true;

   out->indirect_local_geometry =
      hk_pool_alloc(cmd, AGX3_G17P_INDIRECT_GEOMETRY_SIZE,
                    AGX3_G17P_INDIRECT_GEOMETRY_ALIGNMENT);
   out->indirect_local_helper_root =
      hk_pool_alloc(cmd, sizeof(struct hk_root_descriptor_table), 8);
   if (!out->indirect_local_geometry.gpu ||
       !out->indirect_local_helper_root.gpu)
      return true;

   struct hk_root_descriptor_table *helper_root =
      out->indirect_local_helper_root.cpu;
   *helper_root = cmd->state.cs.descriptors.root;
   helper_root->root_desc_addr = out->indirect_local_helper_root.gpu;
   helper_root->cs.grid_helper_geometry_addr =
      out->indirect_local_geometry.gpu;
   helper_root->cs.grid_helper_sparse_map_addr = 0;

   struct agx3_ioto_uniform_range
      helper_ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t helper_range_count;
   uint64_t helper_program_va;
   struct hk_g17p_compute_program_ref helper_program;
   if (!hk_g17p_build_compute_program(
          cmd, helper_s, helper_s->only_linked,
          out->indirect_local_helper_root.gpu, 0, false,
          AGX3_G17P_GRID_HELPER_LDSHDR_F1, "prepare-indirect-local-helper",
          helper_ranges, &helper_range_count, &helper_program_va,
          &helper_program))
      return true;

   const struct agx3_g17p_native_indirect_dispatch indirect_dispatch = {
      .helper_program = &helper_program.launch,
      .geometry_va = out->indirect_local_geometry.gpu,
   };
   struct agx3_compute_dispatch_sink indirect_sink = {
      .data = out->indirect_local_template,
      .capacity = sizeof(out->indirect_local_template),
   };
   if (!hk_agx3_g17p_native_indirect_compute_dispatch_append(
          &indirect_main.launch, &indirect_dispatch, &indirect_sink) ||
       indirect_sink.length != sizeof(out->indirect_local_template))
      return true;

   out->indirect_local_valid = true;
   return true;
}

void
hk_g17p_prewarm_linked_compute_shader(struct hk_device *dev,
                                       struct hk_shader *s,
                                       struct hk_linked_shader *linked)
{
   if (!s || s->info.stage == MESA_SHADER_VERTEX)
      return;

   const enum agx_ioto_g17p_compute_ldshdr_f1 kinds[] = {
      AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
      AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1,
      AGX3_G17P_GRID_HELPER_LDSHDR_F1,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(kinds); ++i)
      hk_g17p_prewarm_linked_compute_program_kind(
         dev, NULL, s, linked, s->info.stage != MESA_SHADER_COMPUTE, kinds[i]);
}

void
hk_g17p_prewarm_indirect_helpers(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_api_shader *helpers[] = {
      dev->g17p_grid_helper,
      dev->g17p_grid_helper_sparse,
      dev->g17p_geometry_copy_helper,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(helpers); ++i) {
      if (!helpers[i])
         continue;

      struct hk_shader *s = hk_only_variant(helpers[i]);
      hk_g17p_prewarm_linked_compute_program_kind(
         dev, cmd, s, s->only_linked, false,
         AGX3_G17P_GRID_HELPER_LDSHDR_F1);
   }
}

/*
 * The helper that publishes an indirect grid's launch geometry. An
 * indirect-global grid holds threadgroup counts, which the grid helper
 * multiplies by the local size; an indirect-local grid already holds the
 * geometry, written by an earlier kernel, which the copy helper publishes.
 */
static struct hk_shader *
hk_g17p_indirect_helper(struct hk_device *dev, struct agx_grid grid,
                        uint64_t sparse_map_addr)
{
   struct hk_api_shader *helper =
      grid.mode == AGX_CDM_MODE_INDIRECT_GLOBAL
         ? (sparse_map_addr ? dev->g17p_grid_helper_sparse
                            : dev->g17p_grid_helper)
      : grid.mode == AGX_CDM_MODE_INDIRECT_LOCAL
         ? dev->g17p_geometry_copy_helper
         : NULL;

   return helper ? hk_only_variant(helper) : NULL;
}

static bool
hk_g17p_indirect_launch_supported(struct hk_device *dev, struct agx_grid grid,
                                  struct agx_workgroup local_size,
                                  uint64_t sparse_map_addr)
{
   /* Without a helper program there is nothing to publish the geometry, so
    * refuse rather than record a launch whose dimensions never arrive. */
   struct hk_shader *helper =
      hk_g17p_indirect_helper(dev, grid, sparse_map_addr);
   if (grid.ptr == 0 || (grid.ptr & 3) != 0 ||
       !hk_g17p_compute_shader_supported(helper,
                                         helper ? helper->only_linked : NULL))
      return false;

   /* The local size must be launchable at all: a zero component would make
    * the helper publish a zero global size, which is indistinguishable from a
    * geometry block it never wrote.  The product itself is the application's
    * responsibility -- the helper multiplies in wrapping 32-bit, exactly as
    * the AGX2 hardware indirect mode does, and Vulkan already bounds
    * groupCount by maxComputeWorkGroupCount.  An indirect-local grid brings
    * its own local size. */
   return grid.mode == AGX_CDM_MODE_INDIRECT_LOCAL ||
          (local_size.x != 0 && local_size.y != 0 && local_size.z != 0);
}

/*
 * Append an indirect launch of a built main program: the helper publishes
 * the grid's launch geometry, and the main record consumes it.
 */
static __attribute__((hot)) bool
hk_g17p_append_indirect_launch(
   struct hk_cmd_buffer *cmd, struct agx_grid grid,
   struct agx_workgroup local_size, uint64_t sparse_map_addr,
   const struct hk_g17p_compute_program_ref *program,
   struct agx3_compute_dispatch_sink *sink)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_shader *helper_s =
      hk_g17p_indirect_helper(dev, grid, sparse_map_addr);

   /* The six-u32 block the helper publishes and the main record consumes.
    * It lives in the command buffer's own pool, so it is allocated per
    * dispatch and cannot be shared between two launches in flight. */
   const struct agx_ptr geometry =
      hk_pool_alloc(cmd, AGX3_G17P_INDIRECT_GEOMETRY_SIZE,
                    AGX3_G17P_INDIRECT_GEOMETRY_ALIGNMENT);
   if (!geometry.gpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }

   /* Zero it: an unwritten block must not read back as a launchable
    * geometry if the helper never ran. */
   memset(geometry.cpu, 0, AGX3_G17P_INDIRECT_GEOMETRY_SIZE);

   /* The helper gets its own root.  It shares the caller's descriptor
    * state -- it touches none of it -- but it needs the grid address, the
    * geometry address and the local size, and the caller's root must keep
    * describing the caller's dispatch. */
   struct hk_root_descriptor_table helper_root =
      cmd->state.cs.descriptors.root;
   helper_root.cs.group_count_addr = grid.ptr;
   helper_root.cs.grid_helper_geometry_addr = geometry.gpu;
   /* Authoritative here rather than whatever the copied root carries: an
    * internal or precompiled launch must never inherit an application
    * dispatch's residency bitmap. */
   helper_root.cs.grid_helper_sparse_map_addr = sparse_map_addr;
   helper_root.cs.grid_helper_local_size[0] = local_size.x;
   helper_root.cs.grid_helper_local_size[1] = local_size.y;
   helper_root.cs.grid_helper_local_size[2] = local_size.z;

   const struct agx_ptr helper_root_ptr =
      hk_pool_alloc(cmd, sizeof(helper_root), 8);
   if (!helper_root_ptr.gpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   helper_root.root_desc_addr = helper_root_ptr.gpu;
   memcpy(helper_root_ptr.cpu, &helper_root, sizeof(helper_root));

   struct agx3_ioto_uniform_range
      helper_ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t helper_range_count;
   uint64_t helper_program_va;
   struct hk_g17p_compute_program_ref helper_program;
   if (!hk_g17p_build_compute_program(
          cmd, helper_s, helper_s->only_linked, helper_root_ptr.gpu, 0, false,
          AGX3_G17P_GRID_HELPER_LDSHDR_F1, "grid-helper", helper_ranges,
          &helper_range_count, &helper_program_va, &helper_program))
      return false;

   const struct agx3_g17p_native_indirect_dispatch dispatch = {
      .helper_program = &helper_program.launch,
      .geometry_va = geometry.gpu,
   };
   const size_t start = sink->length;
   if (!hk_agx3_g17p_native_indirect_compute_dispatch_append(
          &program->launch, &dispatch, sink)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   if (hk_g17p_launch_trace_enabled()) {
      fprintf(stderr,
              "G17P indirect launch: args=%" PRIx64 " geometry=%" PRIx64
              " helper_program=%" PRIx64 " local=%ux%ux%u bytes=%zu\n",
              grid.ptr, (uint64_t)geometry.gpu, helper_program_va,
              local_size.x, local_size.y, local_size.z, sink->length - start);
      hk_g17p_trace_line("indirect-pair",
                         (uint64_t)(uintptr_t)(sink->data + start),
                         (const uint8_t *)sink->data + start,
                         sink->length - start);
   }

   return true;
}

static bool
hk_g17p_get_precomp_program(struct hk_cmd_buffer *cmd,
                             struct agx_precompiled_shader *prog,
                             enum libagx_program idx, const void *data,
                             size_t data_size, bool indirect,
                             bool cache_lookup_missed,
                             struct hk_g17p_compute_program_ref *program)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!indirect && !cache_lookup_missed) {
      struct hk_g17p_precomp_program *cached =
         hk_g17p_find_precomp_program(cmd, idx, data, data_size);
      if (cached) {
         *program = cached->program;
         return true;
      }
   }

   size_t uniform_data_size;
   if (!hk_g17p_precompiled_uniform_data_size(prog, data_size,
                                               &uniform_data_size)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   const size_t printf_offset_B = prog->info.printf_uniform
      ? (size_t)(prog->info.printf_uniform - AGX_ABI_AGX3_PRECOMP_ARG_HALFS) * 2
      : 0;
   uint32_t uploaded_size;
   uint16_t uniform_size_halfs;
   if (!hk_agx3_g17p_precompiled_data_layout(
          uniform_data_size, &uploaded_size, &uniform_size_halfs)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   /* G17P Ioto resource loads operate on whole dwords. Packed argument
    * structs may end in a 16-bit member, so zero the unused padding half. */
   struct agx_ptr uploaded = hk_pool_alloc(cmd, uploaded_size, 4);
   if (!uploaded.cpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   memcpy(uploaded.cpu, data, data_size);
   memset((uint8_t *)uploaded.cpu + data_size, 0, uploaded_size - data_size);
   const void *cache_data = uploaded.cpu;
   if (prog->info.printf_uniform) {
      struct agx_ptr key = hk_pool_alloc(cmd, data_size, 1);
      if (!key.cpu) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
      }
      memcpy(key.cpu, data, data_size);
      cache_data = key.cpu;
      memcpy((uint8_t *)uploaded.cpu + printf_offset_B,
             &dev->dev.printf_buffer_address, sizeof(uint64_t));
   }

   struct agx3_ioto_uniform_range ranges[2] = {{
      .start_halfs = AGX_ABI_AGX3_PRECOMP_ARG_HALFS,
      .size_halfs = uniform_size_halfs,
      .address = uploaded.gpu,
   }};
   unsigned range_count = 1;
   if (prog->info.rodata.size_16) {
      ranges[range_count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = prog->info.rodata.base_uniform,
         .size_halfs = prog->info.rodata.size_16,
         .address = prog->ptr + prog->rodata_offset,
      };
   }

   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(false, false, false,
                                                  &launch_sources)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   const uint32_t threadgroup_memory_size_B =
      prog->info.local_size
         ? MAX2(256u, ALIGN_POT(prog->info.local_size, 64u))
         : 0;
   if (hk_g17p_launch_trace_enabled())
      fprintf(stderr,
              "G17P precompiled resources: program=%u local=%u allocated=%u\n",
              (unsigned)idx, (unsigned)prog->info.local_size,
              threadgroup_memory_size_B);

   uint64_t program_va;
   if (!hk_g17p_build_compute_program_raw(
          cmd, (uint32_t)idx, uploaded_size, NULL,
          prog->ptr + prog->entry_offset, prog->info.main_size,
          prog->info.agx3_tps_size, prog->info.nr_gprs, true,
          (uint32_t)prog->info.agx3_staging_halfs * 2,
          threadgroup_memory_size_B, &launch_sources,
          indirect ? AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1
                   : AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
          "precompiled", ranges, range_count, &program_va, program))
      return false;

   if (!indirect) {
      struct hk_g17p_precomp_program *stored =
         util_dynarray_grow(&cmd->g17p_precomp_programs,
                            struct hk_g17p_precomp_program, 1);
      if (!stored) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return false;
      }
      *stored = (struct hk_g17p_precomp_program){
         .key = {
            .idx = idx,
            .data_size = data_size,
            .data = cache_data,
         },
         .local_size = prog->b.workgroup,
         .program = *program,
      };
      const uint32_t entry = util_dynarray_num_elements(
         &cmd->g17p_precomp_programs, struct hk_g17p_precomp_program);
      const uint32_t hash = hk_g17p_precomp_cache_hash(idx, data, data_size);
      if (!hk_g17p_precomp_cache_insert(&cmd->g17p_precomp_index,
                                        &stored->key, entry, hash)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return false;
      }
   }

   return true;
}

bool
hk_g17p_prepare_precomp(struct hk_cmd_buffer *cmd, enum libagx_program idx,
                        const void *data, size_t data_size,
                        struct hk_g17p_prepared_precomp *out)
{
   *out = (struct hk_g17p_prepared_precomp){0};
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct agx_precompiled_shader *prog =
      (unsigned)idx < LIBAGX_NUM_PROGRAMS
         ? agx_get_precompiled(&dev->bg_eot, idx)
         : NULL;
   if (dev->dev.chip != AGX_CHIP_G17P || !prog || !prog->bo ||
       !prog->info.main_size || !prog->info.nr_gprs ||
       prog->info.nr_preamble_gprs ||
       (prog->info.rodata.size_16 && !prog->rodata_offset) ||
       !prog->info.agx3_staging_known || !data || !data_size ||
       data_size > HK_G17P_PREPARED_PRECOMP_DATA_MAX ||
       prog->entry_offset > UINT64_MAX - prog->ptr)
      return false;

   struct hk_g17p_compute_program_ref program;
   if (!hk_g17p_get_precomp_program(cmd, prog, idx, data, data_size, false, false,
                                    &program))
      return false;

   out->program = program.launch;
   out->local_size = prog->b.workgroup;
   out->idx = idx;
   out->data_size = data_size;
   memcpy(out->data, data, data_size);

   const struct agx3_compute_direct_grid template_grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {1, 1, 1},
      .local_size = {out->local_size.x, out->local_size.y,
                     out->local_size.z},
      .dimensions = 3,
   };
   struct agx3_compute_dispatch_sink template_sink = {
      .data = out->direct_template,
      .capacity = sizeof(out->direct_template),
   };
   if (!hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
          &out->program, &template_grid, 0, &template_sink) ||
       template_sink.length != sizeof(out->direct_template))
      return false;

   out->valid = true;
   return true;
}

void
hk_g17p_prewarm_precomp(struct hk_cmd_buffer *cmd, enum libagx_program idx,
                         const void *data, size_t data_size)
{
   struct hk_g17p_prepared_precomp prepared;
   (void)hk_g17p_prepare_precomp(cmd, idx, data, data_size, &prepared);
}

__attribute__((hot)) bool
hk_dispatch_prepared_precomp_g17p(
   struct hk_cmd_buffer *cmd, struct agx_grid grid, enum agx_barrier barrier,
   enum libagx_program idx, const void *data, size_t data_size,
   const struct hk_g17p_prepared_precomp *prepared)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (dev->dev.chip != AGX_CHIP_G17P || !prepared->valid ||
       prepared->idx != idx || prepared->data_size != data_size ||
       data_size > sizeof(prepared->data) ||
       memcmp(prepared->data, data, data_size) != 0 ||
       agx_is_indirect(grid) || !grid.count[0] || !grid.count[1] ||
       !grid.count[2] || hk_g17p_launch_trace_enabled())
      return false;

   struct hk_cs **target = (barrier & AGX_POSTGFX)  ? &cmd->current_cs.post_gfx
                           : (barrier & AGX_PREGFX) ? &cmd->current_cs.pre_gfx
                                                    : &cmd->current_cs.cs;
   struct hk_cs *cs = hk_cmd_buffer_get_cs_general(cmd, target, true);
   if (!cs)
      return true;

   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   uint8_t *out = (uint8_t *)cs->current;
   memcpy(out, prepared->direct_template, sizeof(prepared->direct_template));
   agx3_cmdstream_store_le32(out + 16, grid.count[0]);
   agx3_cmdstream_store_le32(out + 20, grid.count[1]);
   agx3_cmdstream_store_le32(out + 24, grid.count[2]);
   const uint8_t barrier_flags = (barrier & AGX_BARRIER_ALL) ? 0x5 : 0;
   agx3_cmdstream_store_le32(
      out + AGX3_CDM_DIRECT_LAUNCH_SIZE,
      agx3_cdm_barrier_word(barrier_flags));
   cs->current = out + sizeof(prepared->direct_template);
   if (barrier_flags == 0x5) {
      cs->strong_flush_end = cs->current;
      cs->weak_barrier_end = NULL;
   } else {
      cs->weak_barrier_end = cs->current;
   }
   cs->stats.cmds++;
   return true;
}

__attribute__((hot)) void
hk_dispatch_precomp_g17p(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                         enum agx_barrier barrier, enum libagx_program idx,
                         const void *data, size_t data_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if ((unsigned)idx >= LIBAGX_NUM_PROGRAMS) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   const bool checked_cache =
      data && data_size && !agx_is_indirect(grid) && grid.count[0] &&
      grid.count[1] && grid.count[2] && !hk_g17p_launch_trace_enabled();
   const unsigned checked_cache_size = cmd->g17p_precomp_programs.size;
   if (checked_cache &&
       hk_g17p_append_prepared_precomp_direct(cmd, grid, barrier, idx, data,
                                               data_size))
      return;

   struct agx_precompiled_shader *prog =
      agx_get_precompiled(&dev->bg_eot, idx);
   if (!prog) {
      mesa_loge("no G17P precompiled program %u", (unsigned)idx);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (dev->dev.chip != AGX_CHIP_G17P || !prog->bo ||
       !prog->info.main_size || !prog->info.nr_gprs ||
       prog->info.nr_preamble_gprs ||
       (prog->info.rodata.size_16 && !prog->rodata_offset) ||
       (prog->info.printf_uniform &&
        prog->info.printf_uniform < AGX_ABI_AGX3_PRECOMP_ARG_HALFS) ||
       !prog->info.agx3_staging_known ||
       !data || !data_size ||
       prog->entry_offset > UINT64_MAX - prog->ptr ||
       prog->b.workgroup.x == 0 || prog->b.workgroup.y == 0 ||
       prog->b.workgroup.z == 0) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   const bool indirect = agx_is_indirect(grid);
   const struct agx_workgroup local_size = {
      .x = prog->b.workgroup.x,
      .y = prog->b.workgroup.y,
      .z = prog->b.workgroup.z,
   };
   if (indirect &&
       !hk_g17p_indirect_launch_supported(dev, grid, local_size, 0)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   if (!indirect &&
       (grid.count[0] == 0 || grid.count[1] == 0 || grid.count[2] == 0))
      return;

   struct hk_cs **target = (barrier & AGX_POSTGFX)  ? &cmd->current_cs.post_gfx
                           : (barrier & AGX_PREGFX) ? &cmd->current_cs.pre_gfx
                                                    : &cmd->current_cs.cs;
   struct hk_cs *cs = hk_cmd_buffer_get_cs_general(cmd, target, true);
   if (!cs)
      return;

   struct hk_g17p_compute_program_ref program;
   /* The fast path already searched these exact arguments. Cache entries
    * are immutable until command-buffer reset. Preserve a fresh lookup if
    * intervening stream setup ever appends a precompiled program. */
   const bool cache_lookup_missed = checked_cache &&
      checked_cache_size == cmd->g17p_precomp_programs.size;
   if (!hk_g17p_get_precomp_program(cmd, prog, idx, data, data_size,
                                    indirect, cache_lookup_missed, &program))
      return;

   if (indirect) {
      hk_ensure_cs_has_space(cmd, cs, 0x2000);
      struct agx3_compute_dispatch_sink sink = {
         .data = (uint8_t *)cs->current,
         .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
      };
      if (!hk_g17p_append_indirect_launch(cmd, grid, local_size, 0, &program,
                                          &sink))
         return;

      cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
      /* The indirect pair ends in the deferred launch's own weak ordering
       * barrier (agx3_g17p_native_indirect_compute_dispatch_append passes
       * flags 0).  Record it exactly as the direct path does, so a following
       * hk_cdm_cache_flush() strengthens that record in place instead of
       * appending a second barrier after it: two adjacent barriers with no
       * launch between them is a shape Apple's CDM emitter never produces and
       * the G17P stream oracle rejects. */
      cs->weak_barrier_end = cs->current;
      cs->stats.cmds++;
      return;
   }

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
   /* Internal helpers communicate through memory with their consumer. Match
    * the AGX2 flush-after-launch contract when the caller requests the common
    * all-memory barrier; HAL200 encodes Apple's witnessed strong value 5 in
    * the four-byte tail of this dispatch. */
   const uint8_t barrier_flags = (barrier & AGX_BARRIER_ALL) ? 0x5 : 0;
   if (!hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
          &program.launch, &direct, barrier_flags, &sink)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   if (barrier_flags == 0x5) {
      cs->strong_flush_end = cs->current;
      cs->weak_barrier_end = NULL;
   } else {
      cs->weak_barrier_end = cs->current;
   }
   cs->stats.cmds++;
}

static __attribute__((hot)) bool
hk_g17p_append_graphics_direct(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs,
   const struct agx3_compute_program *program, struct agx_grid grid,
   struct agx_workgroup local_size)
{
   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   struct agx3_compute_dispatch_sink sink = {
      .data = (uint8_t *)cs->current,
      .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
   };
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
   const uint8_t barrier_flags = 0;
   if (!hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
          program, &direct, barrier_flags, &sink)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return true;
   }

   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   if (barrier_flags == 0x5) {
      cs->strong_flush_end = cs->current;
      cs->weak_barrier_end = NULL;
   } else {
      cs->weak_barrier_end = cs->current;
   }
   cs->stats.cmds++;
   return true;
}

static bool
hk_g17p_append_prepared_graphics_direct(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, struct hk_shader *s,
   struct hk_linked_shader *linked, struct agx_grid grid,
   struct agx_workgroup local_size)
{
   struct hk_g17p_exact_compute_program *cached =
      hk_g17p_find_exact_compute_program(
         cmd, s, linked, cmd->state.gfx.root, 0, true,
         AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3);
   if (!cached)
      return false;

   return hk_g17p_append_graphics_direct(cmd, cs, &cached->launch, grid,
                                          local_size);
}

static __attribute__((hot)) bool
hk_dispatch_prepared_g17p_native_impl(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, struct hk_shader *s,
   struct hk_g17p_prepared_dispatch *prepared, struct agx_grid grid,
   struct agx_workgroup local_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *linked = s->only_linked;
   if (!linked && s->info.stage == MESA_SHADER_VERTEX)
      linked = cmd->state.gfx.linked[MESA_SHADER_VERTEX];
   if (!prepared->valid ||
       prepared->shader != s || prepared->linked != linked ||
       prepared->root != cmd->state.gfx.root ||
       prepared->draw_params != cmd->state.gfx.draw_params ||
       prepared->draw_id_ptr != cmd->state.gfx.draw_id_ptr ||
       dev->dev.chip != AGX_CHIP_G17P || cs->type != HK_CS_CDM)
      return false;

   if (hk_g17p_launch_trace_enabled())
      return false;

   if (agx_is_indirect(grid)) {
      if (!prepared->indirect_local_valid ||
          prepared->indirect_local_consumed ||
          grid.mode != AGX_CDM_MODE_INDIRECT_LOCAL || !grid.ptr ||
          (grid.ptr & 3) != 0 || grid.ptr != prepared->indirect_local_grid ||
          local_size.x != prepared->indirect_local_size[0] ||
          local_size.y != prepared->indirect_local_size[1] ||
          local_size.z != prepared->indirect_local_size[2])
         return false;

      hk_reserve_scratch(cmd, cs, s);

      memset(prepared->indirect_local_geometry.cpu, 0,
             AGX3_G17P_INDIRECT_GEOMETRY_SIZE);
      struct hk_root_descriptor_table *helper_root =
         prepared->indirect_local_helper_root.cpu;
      helper_root->cs.group_count_addr = grid.ptr;
      helper_root->cs.grid_helper_geometry_addr =
         prepared->indirect_local_geometry.gpu;
      helper_root->cs.grid_helper_sparse_map_addr = 0;
      helper_root->cs.grid_helper_local_size[0] = local_size.x;
      helper_root->cs.grid_helper_local_size[1] = local_size.y;
      helper_root->cs.grid_helper_local_size[2] = local_size.z;

      hk_ensure_cs_has_space(cmd, cs, 0x2000);
      memcpy(cs->current, prepared->indirect_local_template,
             sizeof(prepared->indirect_local_template));
      cs->current = (uint32_t *)((uint8_t *)cs->current +
                                 sizeof(prepared->indirect_local_template));
      cs->weak_barrier_end = cs->current;
      cs->stats.cmds++;
      prepared->indirect_local_consumed = true;
      return true;
   }

   if (agx_is_shader_empty(&s->b) || grid.count[0] == 0 ||
       grid.count[1] == 0 || grid.count[2] == 0 || !local_size.x ||
       !local_size.y || !local_size.z)
      return true;

   hk_reserve_scratch(cmd, cs, s);
   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   uint8_t *out = (uint8_t *)cs->current;
   memcpy(out, prepared->direct_template, sizeof(prepared->direct_template));
   const uint32_t local[3] = {local_size.x, local_size.y, local_size.z};
   for (unsigned axis = 0; axis < 3; ++axis) {
      agx3_cmdstream_store_le32(out + 16 + axis * 4, grid.count[axis]);
      agx3_cmdstream_store_le32(out + 28 + axis * 4, local[axis]);
   }
   const uint8_t barrier_flags = 0;
   agx3_cmdstream_store_le32(
      out + AGX3_CDM_DIRECT_LAUNCH_SIZE,
      agx3_cdm_barrier_word(barrier_flags));
   cs->current = (uint32_t *)(out + sizeof(prepared->direct_template));
   if (barrier_flags == 0x5) {
      cs->strong_flush_end = cs->current;
      cs->weak_barrier_end = NULL;
   } else {
      cs->weak_barrier_end = cs->current;
   }
   cs->stats.cmds++;
   return true;
}

__attribute__((hot)) bool
hk_dispatch_prepared_g17p_native(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, struct hk_shader *s,
   struct hk_g17p_prepared_dispatch *prepared, struct agx_grid grid,
   struct agx_workgroup local_size)
{
   if (!hk_dispatch_prepared_g17p_native_impl(cmd, cs, s, prepared, grid,
                                              local_size))
      return false;
   return true;
}

static bool
hk_g17p_append_prepared_graphics_indirect(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, struct hk_shader *s,
   struct hk_linked_shader *linked, struct agx_grid grid,
   struct agx_workgroup local_size, uint64_t sparse_map_addr)
{
   struct hk_g17p_exact_compute_program *cached =
      hk_g17p_find_exact_compute_program(
         cmd, s, linked, cmd->state.gfx.root, 0, true,
         AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1);
   if (!cached)
      return false;

   const struct hk_g17p_compute_program_ref program = {
      .bytes = cached->program_bytes,
      .length = cached->program_length,
      .launch = cached->launch,
   };
   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   struct agx3_compute_dispatch_sink sink = {
      .data = (uint8_t *)cs->current,
      .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
   };
   if (!hk_g17p_append_indirect_launch(cmd, grid, local_size, sparse_map_addr,
                                        &program, &sink))
      return true;

   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   cs->weak_barrier_end = cs->current;
   cs->stats.cmds++;
   return true;
}

__attribute__((hot)) void
hk_dispatch_g17p_native(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        struct hk_shader *s, struct agx_grid grid,
                        struct agx_workgroup local_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *linked = s ? s->only_linked : NULL;
   if (s && !linked && s->info.stage == MESA_SHADER_VERTEX)
      linked = cmd->state.gfx.linked[MESA_SHADER_VERTEX];
   static_assert(HK_MAX_SETS == HK_AGX3_G17P_MAX_ROOT_SETS,
                 "G17P root range must cover every Honeykrisp set");

   if (dev->dev.chip != AGX_CHIP_G17P || cs->type != HK_CS_CDM ||
       !hk_g17p_compute_shader_supported(s, linked)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   const bool internal_graphics = s->info.stage != MESA_SHADER_COMPUTE;

   /* Only an application compute dispatch can name a sparse argument buffer;
    * an internal graphics kernel's indirect grid is a driver record. */
   const uint64_t sparse_map_addr =
      internal_graphics
         ? 0
         : cmd->state.cs.descriptors.root.cs.grid_helper_sparse_map_addr;

   const bool indirect = agx_is_indirect(grid);
   if (indirect) {
      if (!hk_g17p_indirect_launch_supported(dev, grid, local_size,
                                             sparse_map_addr)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
   } else if (grid.count[0] == 0 || grid.count[1] == 0 ||
              grid.count[2] == 0) {
      return;
   }

   /* BeginRendering publishes exact internal-graphics loaders after the root
    * address is fixed. A direct draw can append that launch immediately,
    * avoiding the general compute resource-validation and composition path.
    * Misses and launch tracing retain the authoritative slow path below. */
   if (internal_graphics && !indirect && !hk_g17p_launch_trace_enabled() &&
       hk_g17p_append_prepared_graphics_direct(cmd, cs, s, linked, grid,
                                               local_size))
      return;
   if (internal_graphics && indirect && !hk_g17p_launch_trace_enabled() &&
       hk_g17p_append_prepared_graphics_indirect(
          cmd, cs, s, linked, grid, local_size, sparse_map_addr))
      return;

   if (!internal_graphics && s->data_size != 0) {
      const uint64_t constant_data = hk_pool_upload(
         cmd, s->data_ptr, s->data_size, HK_MIN_UBO_ALIGNMENT);
      if (!constant_data) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }

      cmd->state.cs.descriptors.root.cs.constant_data_addr = constant_data;
   } else if (!internal_graphics) {
      cmd->state.cs.descriptors.root.cs.constant_data_addr = 0;
   }

   const uint64_t root_ptr = internal_graphics
      ? cmd->state.gfx.root
      : hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
   if (!root_ptr) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   uint64_t texture_pointers = 0;
   /* Poly geometry kernels use graphics descriptor roots but execute through
    * CDM. Their indexed samples consume the same device-global texture and
    * sampler heap pair as compute. Other internal-graphics resource profiles
    * still need their own root contracts. */
   if (internal_graphics &&
       (s->info.agx3_txf_texture_valid || s->info.agx3_storage_image_valid ||
        s->info.agx3_sampled_storage_valid ||
        s->info.agx3_indexed_storage_valid)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (s->info.agx3_txf_texture_valid || s->info.agx3_storage_image_valid ||
       s->info.agx3_sampled_storage_valid ||
       s->info.agx3_indexed_storage_valid ||
       s->info.agx3_indexed_sampler_valid) {
      const struct hk_descriptor_state *descriptors = internal_graphics
         ? &cmd->state.gfx.descriptors
         : &cmd->state.cs.descriptors;

      if (s->info.agx3_indexed_sampler_valid) {
         if (!dev->sampled_images.base_va ||
             !dev->samplers.table.bo || !dev->samplers.table.bo->va) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         uint64_t pointers[4] = {
            dev->sampled_images.base_va,
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
            const uint64_t write_control = hk_pool_upload(
               cmd, hk_g17p_native_write_control,
               sizeof(hk_g17p_native_write_control), 32);
            if (!write_control) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_OUT_OF_DEVICE_MEMORY);
               return;
            }
            const struct hk_agx3_g17p_indexed_storage_heap heap = {
               .texture_heap_va = pointers[0],
               .sampler_heap_va = pointers[1],
               .pbe_descriptor_set_va = set_address,
               .write_control_va = write_control,
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
         const uint64_t write_control = hk_pool_upload(
            cmd, hk_g17p_native_write_control,
            sizeof(hk_g17p_native_write_control), 32);
         if (!write_control) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }
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
            .write_control_va = write_control,
            .pbe_offset_B = binding->pbe_offset_B,
         };
         uint64_t pointers[4] = {0};
         if (!hk_agx3_g17p_sampled_storage_pointers(&heap, pointers)) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         /* See hk_g17p_texture_state_pointer(). */
         pointers[0] = hk_g17p_texture_state_pointer(
            cmd, pointers[0],
            (const uint8_t *)texture_set->mapped_ptr +
               binding->texture_offset_B,
            texture_set->size > binding->texture_offset_B
               ? texture_set->size - binding->texture_offset_B
               : 0);
         if (!pointers[0]) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
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
         /* See hk_g17p_texture_state_pointer(). */
         pointers[0] = hk_g17p_texture_state_pointer(
            cmd, pointers[0],
            (const uint8_t *)set->mapped_ptr + binding->texture_offset_B,
            set->size > binding->texture_offset_B
               ? set->size - binding->texture_offset_B
               : 0);
         if (!pointers[0]) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }
      } else if (s->info.agx3_storage_image_valid) {
         /* Exact access controls from native G17P Tier-2 argument buffers.
          * A write-only resource is a PBE pointer followed by write-control;
          * read_write instead starts with texture/PBE and points at the
          * paired read/write controls in its third slot. */
         static const uint8_t native_read_write_control[32] = {
            0x68, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x03, 0x1b,
            0x08, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
         };
         const bool write_only = s->info.agx3_storage_image.write_only;
         const uint64_t access_control =
            hk_pool_upload(cmd, write_only ? hk_g17p_native_write_control
                                           : native_read_write_control,
                           sizeof(native_read_write_control), 32);
         if (!access_control) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }

         if (write_only) {
            pointers[0] =
               set_address + s->info.agx3_storage_image.pbe_offset_B;
            pointers[1] = access_control;
         } else {
            pointers[0] = set_address +
                          s->info.agx3_storage_image.texture_offset_B;
            pointers[1] =
               set_address + s->info.agx3_storage_image.pbe_offset_B;
            pointers[2] = access_control;
            /* The native read_write ISEL clamps each unsigned coordinate
             * against uniform half12. The consumed vector word is 0xffff. */
            pointers[3] = UINT64_C(0xffff);
         }
      } else {
         /* Native G17P compute read() uses table pointers in uniform
          * halves0..7, with ordinary buffer arguments beginning at half8. */
         pointers[0] = set_address + offset;
         pointers[1] = dev->rodata.bo->va->addr;

         /* Unlike the indexed ABI, direct txf publishes the descriptor-set
          * texture word itself.  G17P ignores the low five pointer bits, so
          * snapshot an unaligned record just as the combined-sampler paths do.
          * The shader selector admits such a snapshot only when sampled-image
          * update-after-bind is disabled; an aligned live set pointer is safe
          * with either feature state. */
         if (pointers[0] & (HK_G17P_TEXTURE_STATE_ALIGN - 1)) {
            struct hk_descriptor_set *set = descriptors->sets[0];
            const void *set_map = set ? set->mapped_ptr : NULL;
            uint32_t mapped_size = set ? set->size : 0;
            if (!set && descriptors->push[0]) {
               set_map = descriptors->push[0]->data;
               mapped_size = sizeof(descriptors->push[0]->data);
            }

            if (dev->vk.enabled_features
                   .descriptorBindingSampledImageUpdateAfterBind ||
                !set_map || offset > mapped_size) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }

            pointers[0] = hk_g17p_texture_state_pointer(
               cmd, pointers[0], (const uint8_t *)set_map + offset,
               mapped_size - offset);
            if (!pointers[0]) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_OUT_OF_DEVICE_MEMORY);
               return;
            }
         }
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

   struct agx3_ioto_uniform_range
      ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t range_count;
   uint64_t program_va;
   struct hk_g17p_compute_program_ref program;
   if (!hk_g17p_build_compute_program(
          cmd, s, linked, root_ptr, texture_pointers, internal_graphics,
          indirect ? AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1
                   : AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
          "main", ranges, &range_count, &program_va, &program))
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
      hk_g17p_trace_hexdump("ioto-program", program_va,
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
      if (!hk_g17p_append_indirect_launch(cmd, grid, local_size,
                                          sparse_map_addr, &program, &sink))
         return;

      cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
      /* The indirect pair ends in the deferred launch's own weak ordering
       * barrier (agx3_g17p_native_indirect_compute_dispatch_append passes
       * flags 0).  Record it exactly as the direct path does, so a following
       * hk_cdm_cache_flush() strengthens that record in place instead of
       * appending a second barrier after it: two adjacent barriers with no
       * launch between them is a shape Apple's CDM emitter never produces and
       * the G17P stream oracle rejects. */
      cs->weak_barrier_end = cs->current;
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

   /* Every launch already ends in a barrier record; the only open question is
    * its flags. This path passes 0, a barrier that orders the launches but
    * does not make the producer's writes visible to the consumer. Every other
    * compute path in the driver flushes: hk_dispatch_with_usc_launch() calls
    * hk_cdm_cache_flush() after each AGX2 launch, and hk_dispatch_precomp_g17p()
    * passes Apple's witnessed strong value 5 for AGX_BARRIER_ALL. So the
    * "flush after every launch" policy hk_cdm_cache_flush() documents is not
    * actually held on the one path every API dispatch and every vk_meta copy,
    * fill and buffer update on G17P goes through.
    *
    * A missing write->read dependency here is invisible until two dispatches
    * in one CDM stream are producer and consumer, which is exactly the shape
    * of the G17P meta copy paths: the compressed copies emit one dispatch per
    * block with nothing in between, and merge_control_streams() concatenates
    * adjacent CDM streams at hk_EndCommandBuffer, so independent copies share
    * a stream too.
    *
    * Whether this weak shape explains the Hyprland GMMU read fault that
    * ZINK_DEBUG=sync cures is unproven.
    */
   const uint8_t barrier_flags = 0;
   if (!hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
          &program.launch, &direct, barrier_flags, &sink)) {
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
              program_va, linked->b.bo->va->addr, w[0], w[1], w[2],
              w[3], reconstructed, reconstructed_or,
              reconstructed_or == program_va ? "" : " MISMATCH");
   }

   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   if (barrier_flags == 0x5) {
      cs->strong_flush_end = cs->current;
      cs->weak_barrier_end = NULL;
   } else {
      cs->weak_barrier_end = cs->current;
   }
   cs->stats.cmds++;
}

static void
dispatch_grid(struct hk_cmd_buffer *cmd, struct agx_grid grid, bool in_threads)
{
   struct hk_shader *s = hk_only_variant(cmd->state.cs.shader);
   if (agx_is_shader_empty(&s->b))
      return;

   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (hk_conditional_render_is_active(&cmd->state.conditional, cmd->in_meta,
                                        cmd->conditional_meta)) {
      const uint64_t patched = hk_apply_conditional_rendering(
         cmd, desc->root.cs.group_count_addr, 3, AGX_BARRIER_ALL);
      if (!patched)
         return;

      /* All API dispatch forms already publish their three group counts in
       * the root. Point both the shader-visible count and the command-stream
       * indirect launch at the predicated copy, so skipped commands also
       * contribute zero pipeline-stat invocations. */
      desc->root.cs.group_count_addr = patched;
      desc->root_dirty = true;
      grid = agx_grid_indirect(patched);
   }

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true /* compute */);
   if (!cs)
      return;

   struct agx_workgroup local_size =
      agx_workgroup(s->b.info.workgroup_size[0], s->b.info.workgroup_size[1],
                    s->b.info.workgroup_size[2]);

   /* Internal compute work must not contribute to the application's active
    * pipeline query. This also honours hk_pipeline_stat_addr's !in_meta
    * contract in debug builds; graphics meta pauses its own counters. */
   if (!cmd->in_meta) {
      uint64_t stat = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT);

      if (hk_stat_enabled(stat)) {
         perf_debug(cmd, "CS invocation statistic");
         uint64_t grid = cmd->state.cs.descriptors.root.cs.group_count_addr;

         libagx_increment_cs_invocations(cmd, agx_1d(1), AGX_BARRIER_ALL, grid,
                                         stat, agx_workgroup_threads(local_size));
      }
   }

   if (!agx_is_indirect(grid) && !in_threads) {
      grid.count[0] *= local_size.x;
      grid.count[1] *= local_size.y;
      grid.count[2] *= local_size.z;
   }

   hk_dispatch_with_local_size(cmd, cs, s, grid, local_size);
   cs->stats.calls++;
}

static void
dispatch(struct hk_cmd_buffer *cmd, struct agx_grid grid)
{
   dispatch_grid(cmd, grid, false);
}

/*
 * Launch an exact number of invocations, allowing a partial trailing
 * workgroup.  The common acceleration-structure build shaders have no bounds
 * guard of their own and rely on this; the AGX command stream takes a thread
 * count natively, so "unaligned" costs nothing here beyond skipping the
 * workgroup-to-thread multiply that the API entry points do.
 *
 * The group_count the root still advertises is rounded up, because that is
 * what gl_NumWorkGroups and the compute-invocation statistic mean.
 */
void
hk_cmd_dispatch_unaligned(VkCommandBuffer commandBuffer, uint32_t invocations_x,
                          uint32_t invocations_y, uint32_t invocations_z)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;

   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   struct hk_shader *s = hk_only_variant(cmd->state.cs.shader);
   if (!s || agx_is_shader_empty(&s->b))
      return;

   desc->root.cs.base_group[0] = 0;
   desc->root.cs.base_group[1] = 0;
   desc->root.cs.base_group[2] = 0;

   VkDispatchIndirectCommand group_count = {
      .x = DIV_ROUND_UP(invocations_x, s->b.info.workgroup_size[0]),
      .y = DIV_ROUND_UP(invocations_y, s->b.info.workgroup_size[1]),
      .z = DIV_ROUND_UP(invocations_z, s->b.info.workgroup_size[2]),
   };
   desc->root.cs.group_count_addr =
      hk_pool_upload(cmd, &group_count, sizeof(group_count), 8);
   desc->root.cs.grid_helper_sparse_map_addr = 0;

   dispatch_grid(cmd, agx_3d(invocations_x, invocations_y, invocations_z),
                 true);
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
   desc->root.cs.grid_helper_sparse_map_addr = 0;

   dispatch(cmd, agx_3d(groupCountX, groupCountY, groupCountZ));
}

/*
 * Dispatch a ray-tracing megakernel.
 *
 * VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR has its own descriptor state, but the
 * megakernel executes on the ordinary compute path, so the ray-tracing state is
 * installed into the compute slots for the duration of one dispatch and put
 * back afterwards.  Anything the dispatch lazily allocates inside the
 * descriptor state (a pushed descriptor snapshot, the constant-data address)
 * is carried back rather than dropped, or it would leak.
 */
void
hk_cmd_dispatch_rays(struct hk_cmd_buffer *cmd, struct hk_api_shader *shader,
                     uint64_t record_addr, uint64_t group_count_addr,
                     struct agx_grid grid)
{
   struct hk_descriptor_state *rt = &cmd->state.rt.descriptors;

   if (rt->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, rt);

   const struct hk_compute_state saved = cmd->state.cs;

   cmd->state.cs.shader = shader;
   cmd->state.cs.descriptors = *rt;

   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   desc->root.cs.base_group[0] = 0;
   desc->root.cs.base_group[1] = 0;
   desc->root.cs.base_group[2] = 0;
   desc->root.cs.group_count_addr = group_count_addr;
   desc->root.cs.grid_helper_sparse_map_addr = 0;
   desc->root.cs.trace_rays_addr = record_addr;
   desc->root_dirty = true;

   dispatch(cmd, grid);

   /* Carry the state back before restoring, so a pushed descriptor set the
    * flush above allocated stays owned by the ray-tracing state. */
   *rt = cmd->state.cs.descriptors;
   rt->root.cs.trace_rays_addr = 0;

   cmd->state.cs = saved;
   cmd->state.cs.descriptors.root_dirty = true;
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

   /* Arguments are normally read through the sparse read-only shadow, so that
    * a sparse resident buffer honours sparseResidencyNonResidentStrict: an
    * unbound page reads as zero there, and a zero grid launches nothing.
    *
    * On G17P that shadow cannot be read here.  A read through the shadow does
    * not reliably return data written through the primary mapping in the same
    * submission.  Measured by forcing hk_g17p_sparse_buffer_copy() to read
    * resident pages through the shadow as well, over five runs of
    * dEQP-VK.sparse_resources.buffer.ssbo.sparse_residency: buffer_size_2_10
    * and _2_16 failed 5/5, and _2_20 failed 1/5.  The same build passes 3/3
    * unforced, and an added strong CDM cache-flush record does not cure it.
    * The mechanism is not established; what is established is that the value
    * read back is not dependable.
    *
    * Indirect dispatch is where this was measured, not the only place it can
    * bite.  The storage-buffer lowering and the sparse buffer copy both read
    * resident pages through the primary mapping and consult the residency
    * bitmap, so neither is exposed.  hk_CmdDrawIndirect(),
    * hk_CmdDrawIndexedIndirect(), hk_CmdDrawIndirectCount() and conditional
    * rendering do still hand the shadow address to the hardware, which reads
    * the record from it, so the same unreliable read is latent on those paths.
    * They have not been probed: a draw whose indirect arguments come from an
    * unbound page is the same unbounded-work hazard this change exists to
    * remove, and is not something to demonstrate on a shared machine.
    *
    * So do what they do: name the primary mapping and hand the grid helper the
    * bitmap, which substitutes a zero grid for a non-resident page and keeps
    * the strictness promise.  A sparse buffer without residency has no bitmap;
    * every page an application may legally read from it is bound.
    */
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* A residency bitmap is only useful with the helper that reads it. That
    * helper is built for devices that enabled sparseResidencyBuffer, which is
    * the only way to create a buffer carrying a bitmap; if it is somehow
    * absent, keep the old behaviour rather than refusing the dispatch. */
   const bool have_sparse_helper = dev->g17p_grid_helper_sparse != NULL;
   const bool use_primary =
      dev->dev.chip == AGX_CHIP_G17P && buffer->va &&
      (!buffer->sparse_map || have_sparse_helper);

   uint64_t dispatch_addr = hk_buffer_address(buffer, offset, !use_primary);
   assert(dispatch_addr != 0);

   desc->root.cs.group_count_addr = dispatch_addr;
   desc->root.cs.grid_helper_sparse_map_addr =
      (use_primary && buffer->sparse_map) ? buffer->sparse_map->va->addr : 0;
   dispatch(cmd, agx_grid_indirect(dispatch_addr));
}
