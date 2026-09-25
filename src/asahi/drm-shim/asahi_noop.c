/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2018 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "asahi_g17p_stream_validator.h"
#include "util/macros.h"
#include "drm-shim/drm_shim.h"
#include "drm-uapi/asahi_drm.h"

bool drm_shim_driver_prefers_first_render_node = true;

/*
 * T8103 / M1, the AGX2 identity Honeykrisp is already conformant on.  This is
 * the control arm for G17P work: run the same client twice, changing only
 * ASAHI_SHIM_CHIP, and any difference is attributable to the AGX3 paths.
 *
 * Only the architectural identity differs from params_g17p below - generation,
 * variant, chip id, ISA, HAL, core topology, soft faults and the 39-bit user
 * VA range.  Everything else is held equal deliberately so the diff isolates
 * the generation.  A stub here (the previous zero num_clusters_total) makes
 * agx_chip_from_params return AGX_CHIP_UNKNOWN and the device fails to open,
 * which is why the fields are spelled out.
 */
static const struct drm_asahi_params_global params_g13g = {
   .features = 0,
   .gpu_generation = 13,
   .gpu_variant = 'G',
   .gpu_revision = 0,
   .chip_id = 0x8103,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 8,
   .core_masks = {0xff},
   .vm_start = 0x4000,
   .vm_end = 0x7fffff8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 1,
   .max_attachments = 16,
   .max_frequency_khz = 1278000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 2,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_LEGACY,
};

/*
 * T8140 / A18 Pro, exactly as our G17P kernel adapter reports it
 * (drivers/gpu/drm/asahi/g17_drm.rs:24-27,118-140 and file.rs:295-324): a
 * 42-bit user root two pages below the top, one die, one cluster, USC
 * generation 3, HAL 200, and soft faults on because fault_control defaults to
 * 0xb.  Select with ASAHI_SHIM_CHIP=g17p.  This exists so the Honeykrisp G17P
 * shader and command-stream paths can be exercised without hardware.
 */
static const struct drm_asahi_params_global params_g17p = {
   .features = DRM_ASAHI_FEATURE_SOFT_FAULTS,
   .gpu_generation = 17,
   .gpu_variant = 'P',
   .gpu_revision = 0,
   .chip_id = 0x8140,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 5,
   .core_masks = {0x1f},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 1,
   .max_attachments = 16,
   .max_frequency_khz = 1500000,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_200,
};

/* Compiler diagnostics only: J813's observed G17G C0/HAL 300 topology.
 * This profile rejects every submission; it cannot validate GPU execution or
 * produce a conformance result. Keep its explicit name distinct from hardware.
 */
static const struct drm_asahi_params_global params_g17g_compile_only = {
   .gpu_generation = 17,
   .gpu_variant = 'G',
   .gpu_revision = 0x20,
   .chip_id = 0x8142,
   .num_dies = 1,
   .num_clusters_total = 1,
   .num_cores_per_cluster = 10,
   .core_masks = {0x3ff},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 1,
   .max_attachments = 16,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_300,
};

/*
 * T8152 / M6 (J873), selected with ASAHI_SHIM_CHIP=g18g. There is no G18G
 * kernel driver yet, so like the G17G profile above this one rejects every
 * submission: it exists to drive the G18G identity, record packers and
 * compiler gates offline, not to validate execution.
 *
 * Grounded fields: generation 18, variant 'G' and chip 0x8152 (the J873 is
 * Mac18,5 / T8152, its ADT GPU node is gpu,t8152 and its kext
 * com.apple.AGXG18G); USC generation 3 (the applegpu_g18g Mach-O
 * cpusubtype 0x1c3); a 42-bit user VA two pages below the top, as on M5 (the
 * UAT is 43 bits wide with root selection at bit 42, per the SPTM GET_INFO
 * replies in j873/hv/G18G-BOOTSTRAP.md). Revision and topology are measured
 * on the J873: the Linux driver's SGX ID probe decodes revision B1 and two
 * clusters of six cores (mask 0xfff), and the native ioreg reports
 * num_cores=12, num_mgpus=2, core_mask_list=(63,63). (The kext's class
 * name, AGX::PI_400::G::A0, ends in A0; the ID register reads B1.)
 *
 * gpu_hal_generation stays DRM_ASAHI_GPU_HAL_300 until the kernel settles on a
 * value: Apple's G18G kext is PI_400, but no Linux G18G driver publishes one
 * and the uAPI defines no DRM_ASAHI_GPU_HAL_400. Mesa selects the G18G record
 * layouts from agx_chip_command_stream_generation, never from this number.
 */
static const struct drm_asahi_params_global params_g18g = {
   .gpu_generation = 18,
   .gpu_variant = 'G',
   .gpu_revision = 0x11,
   .chip_id = 0x8152,
   .num_dies = 1,
   .num_clusters_total = 2,
   .num_cores_per_cluster = 6,
   .core_masks = {0x3f, 0x3f},
   .vm_start = 0x4000,
   .vm_end = (1ull << 42) - 0x8000ull,
   .vm_kernel_min_size = 0x20000000,
   .max_commands_per_submission = 1,
   .max_attachments = 16,
   .command_timestamp_frequency_hz = 1000000000,
   .usc_generation = 3,
   .gpu_hal_generation = DRM_ASAHI_GPU_HAL_300,
};

static bool
g17g_compile_only(void)
{
   const char *chip = getenv("ASAHI_SHIM_CHIP");
   return chip && !strcmp(chip, "g17g-compile-only");
}

static bool
g18g_profile(void)
{
   const char *chip = getenv("ASAHI_SHIM_CHIP");
   return chip && !strcmp(chip, "g18g");
}

static const struct drm_asahi_params_global *
shim_params(void)
{
   const char *chip = getenv("ASAHI_SHIM_CHIP");

   if (g17g_compile_only())
      return &params_g17g_compile_only;

   if (g18g_profile())
      return &params_g18g;

   if (chip && !strcmp(chip, "g17p"))
      return &params_g17p;

   return &params_g13g;
}

struct asahi_bo {
   struct shim_bo base;
   uint32_t offset;
   uint32_t gem_flags;
};

static struct asahi_bo *
asahi_bo(struct shim_bo *bo)
{
   return (struct asahi_bo *)bo;
}

struct asahi_device {
   uint64_t next_offset;
};

static struct asahi_device asahi = {
   .next_offset = 0x1000,
};

struct asahi_vm_mapping {
   int fd;
   uint64_t address;
   uint64_t size;
   uint64_t object_offset;
   bool single_page;
   struct shim_bo *bo;
};

static struct {
   struct asahi_vm_mapping *entries;
   size_t count;
   size_t capacity;
} shim_mappings;

/*
 * The kernel's T8140 native render-context view registry, mirrored.
 *
 * drivers/gpu/drm/asahi/mmu.rs keeps a seven-slot registry of the fixed
 * process resources native AGXSecureGart presents twice: once at a canonical
 * high "source" address userspace owns, and once at a low "context" address
 * the driver materializes.  A slot is filled ONLY by a VM_BIND map whose range
 * is a superset of that view's source interval
 * (Vm::track_t8140_native_context_binding), and is cleared by any map or unmap
 * that merely overlaps it (the same function, plus
 * untrack_t8140_native_context_range).
 *
 * Every render submission then walks all seven slots
 * (Uat::install_t8140_job_context_aliases) and refuses the job with ENOENT if
 * one is empty, which is the exact hardware failure this models:
 *
 *   MMU: T8140 job context view 0 source 0x1000000000+0x10000 -> 0x0 is not
 *   bound
 *   G17P render: STAGE FAIL stage=new-user-render-storage errno=-2
 *
 * A no-op VM_BIND has no such registry, so a driver that never establishes
 * these bindings -- or that later takes one away -- looks perfectly healthy
 * under the shim and only fails on the target.
 *
 * The addresses are the kernel's, not a restatement of Mesa's: they come from
 * T8140_NATIVE_CONTEXT_VIEWS, so a driver that moves its context BO without
 * the kernel moving with it fails here rather than on Neo.
 */
struct asahi_t8140_context_view {
   uint64_t source_address;
   uint64_t size;
   uint64_t context_address;
};

static const struct asahi_t8140_context_view t8140_context_views[] = {
   {UINT64_C(0x1000000000), UINT64_C(0x10000), UINT64_C(0)},
   {UINT64_C(0x1000018000), UINT64_C(0x8000), UINT64_C(0x18000)},
   {UINT64_C(0x1000028000), UINT64_C(0x8000), UINT64_C(0x28000)},
   {UINT64_C(0x1000038000), UINT64_C(0x8000), UINT64_C(0x38000)},
   {UINT64_C(0x1000048000), UINT64_C(0x8000), UINT64_C(0x48000)},
   {UINT64_C(0x1000058000), UINT64_C(0x8000), UINT64_C(0x58000)},
   {UINT64_C(0x1000068000), UINT64_C(0xc000), UINT64_C(0x68000)},
};

/*
 * The window the kernel reserves for the client null view for a VM's whole
 * lifetime (T8140_CLIENT_NULL_VIEW_START/SIZE).  file.rs rejects any user bind
 * that overlaps a driver range with EINVAL, so Mesa must bind its low
 * presentations from 0x18000 upwards and never below.
 */
#define T8140_CLIENT_NULL_VIEW_START UINT64_C(0)
#define T8140_CLIENT_NULL_VIEW_SIZE  UINT64_C(0x10000)

struct asahi_t8140_context_slot {
   bool bound;
   uint32_t handle;
   uint64_t object_offset;
};

static struct asahi_t8140_context_slot
   t8140_context_slots[ARRAY_SIZE(t8140_context_views)];
static unsigned t8140_render_submits_checked;
static unsigned t8140_render_submits_unbound;

static bool
t8140_registry_active(void)
{
   return shim_params() == &params_g17p;
}

static bool
range_overlaps(uint64_t a_start, uint64_t a_size, uint64_t b_start,
               uint64_t b_size)
{
   return a_size && b_size && a_start < b_start + b_size &&
          b_start < a_start + a_size;
}

static bool
range_is_superset(uint64_t outer_start, uint64_t outer_size,
                  uint64_t inner_start, uint64_t inner_size)
{
   return outer_size && inner_start >= outer_start &&
          inner_start + inner_size <= outer_start + outer_size;
}

static void
t8140_track_bind(const struct drm_asahi_gem_bind_op *op)
{
   if (!t8140_registry_active())
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(t8140_context_views); ++i) {
      const struct asahi_t8140_context_view *view = &t8140_context_views[i];

      /* Order matters and is the kernel's: a bind that merely overlaps the
       * view invalidates the slot, and only a bind that fully contains it
       * re-establishes one. */
      if (range_overlaps(op->addr, op->range, view->source_address,
                         view->size))
         t8140_context_slots[i].bound = false;

      if (range_is_superset(op->addr, op->range, view->source_address,
                            view->size)) {
         t8140_context_slots[i] = (struct asahi_t8140_context_slot){
            .bound = true,
            .handle = op->handle,
            .object_offset = op->offset + (view->source_address - op->addr),
         };
      }
   }
}

static void
t8140_untrack_range(uint64_t address, uint64_t size)
{
   if (!t8140_registry_active())
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(t8140_context_views); ++i) {
      const struct asahi_t8140_context_view *view = &t8140_context_views[i];
      if (range_overlaps(address, size, view->source_address, view->size))
         t8140_context_slots[i].bound = false;
   }
}

/*
 * The submission-time gate, reported in the kernel's own words so a shim log
 * and a Neo dmesg can be compared line for line.
 *
 * Reporting is unconditional; failing the submit is not, because a driver that
 * trips this still records a valid command stream and the other oracles should
 * keep testing it.  ASAHI_SHIM_G17P_ENFORCE_CONTEXT_VIEWS reproduces the
 * kernel's ENOENT, and ASAHI_SHIM_G17P_EXPECT_CONTEXT_VIEWS asserts at exit
 * that no render submission was ever missing a view.
 */
static bool
t8140_check_context_views(void)
{
   if (!t8140_registry_active())
      return true;

   t8140_render_submits_checked++;

   bool complete = true;
   for (unsigned i = 0; i < ARRAY_SIZE(t8140_context_views); ++i) {
      if (t8140_context_slots[i].bound)
         continue;

      const struct asahi_t8140_context_view *view = &t8140_context_views[i];
      fprintf(stderr,
              "shim: MMU: T8140 job context view %u source %#llx+%#llx -> "
              "%#llx is not bound\n",
              i, (unsigned long long)view->source_address,
              (unsigned long long)view->size,
              (unsigned long long)view->context_address);
      complete = false;
   }

   if (!complete) {
      for (unsigned i = 0; i < ARRAY_SIZE(t8140_context_views); ++i) {
         const struct asahi_t8140_context_slot *slot = &t8140_context_slots[i];
         fprintf(stderr,
                 "shim: T8140 context view %u: %s handle=%u object_offset=%#llx\n",
                 i, slot->bound ? "bound" : "UNBOUND", slot->handle,
                 (unsigned long long)slot->object_offset);
      }
   }

   if (!complete)
      t8140_render_submits_unbound++;

   return complete;
}

/*
 * A submit the stream oracle refused is only reported today: the ioctl returns
 * EINVAL, Mesa prints it, and the client carries on and exits 0, so `meson
 * test` stays green over a command stream the kernel would never have run.
 * This knob makes a run fail closed on that.
 */
static unsigned g17p_streams_validated;
static unsigned g17p_streams_rejected;

static void __attribute__((destructor))
check_g17p_stream_validity(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_VALID_STREAM"))
      return;

   if (!g17p_streams_validated || g17p_streams_rejected) {
      fprintf(stderr,
              "shim: G17P command streams accepted by the oracle: %u, "
              "rejected: %u\n",
              g17p_streams_validated, g17p_streams_rejected);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr, "shim: all %u G17P command streams were accepted\n",
           g17p_streams_validated);
}

static void __attribute__((destructor))
check_t8140_context_views(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_CONTEXT_VIEWS"))
      return;

   if (!t8140_render_submits_checked || t8140_render_submits_unbound) {
      fprintf(stderr,
              "shim: G17P render submissions with a complete native context "
              "view set: %u of %u\n",
              t8140_render_submits_checked - t8140_render_submits_unbound,
              t8140_render_submits_checked);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: all %u G17P render submissions carried the complete native "
           "context view set\n",
           t8140_render_submits_checked);
}

static unsigned invalid_vm_binds;

/* Executable-window accounting.
 *
 * A G17P Ioto program is a couple of hundred bytes, but the smallest BO that
 * can hold one costs a 16KiB page plus a 16KiB guard page of the 4GiB USC
 * exec window. Counting the binds that land in that window is the only
 * host-side measure of how fast a recording eats it, and it is exactly what
 * ran out on hardware, so a test can hold the driver to a budget here rather
 * than having to drive a laptop to 4GiB of address space. */
#define SHIM_G17P_USC_WINDOW_BASE UINT64_C(0x10000000000)
#define SHIM_G17P_USC_WINDOW_SIZE UINT64_C(0x100000000)

/* Mirrors AGX_G17P_RENDER_AUX_ALIAS_* in agx_va_runtime_hole.h. */
#define SHIM_G17P_RENDER_AUX_ALIAS_START UINT64_C(0x10002000000)
#define SHIM_G17P_RENDER_AUX_ALIAS_END   UINT64_C(0x10002400000)

static unsigned usc_window_binds;
static uint64_t usc_window_bytes;
static bool failed_submit_once;
static unsigned submit_calls;
static unsigned queue_progress_queue_creates;
static unsigned queue_progress_queue_destroys;
static unsigned queue_progress_submits;
static uint32_t queue_progress_private_handle;
static uint64_t queue_progress_private_point;
static bool queue_progress_private_sync_valid;
static unsigned wsi_rollback_queue_creates;
static unsigned wsi_rollback_queue_destroys;
static unsigned wsi_rollback_submits;

static struct asahi_g17p_stream_stats matrix_stats;

/*
 * How many graphics control streams the run submitted.
 *
 * The barrier accounting above sums every stream together, so it cannot tell
 * one stream carrying sixteen draws apart from sixteen streams carrying one
 * each -- which is the entire question a draws-per-stream policy is about.
 * Counting validated streams separates them.
 */
static unsigned validated_graphics_streams;
static struct asahi_g17p_compute_stats compute_stats;
static bool multiview_layers_two;
static unsigned synchronized_render_submits;

static const char *
queue_progress_lifecycle_case(void)
{
   return getenv("ASAHI_SHIM_EXPECT_QUEUE_PROGRESS_LIFECYCLE");
}

static const char *
wsi_rollback_case(void)
{
   const char *test_case = getenv("ASAHI_SHIM_EXPECT_ACQUIRE_ROLLBACK");
   return test_case ? test_case :
      getenv("ASAHI_SHIM_EXPECT_PRESENT_ROLLBACK");
}

static void __attribute__((destructor))
check_vm_bind_lifetime(void)
{
   if (!getenv("ASAHI_SHIM_EXPECT_VALID_VM_BINDS"))
      return;

   if (invalid_vm_binds) {
      fprintf(stderr, "shim: observed %u invalid VM_BIND operations\n",
              invalid_vm_binds);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr, "shim: all VM_BIND operations had nonzero ranges\n");
}

/*
 * ASAHI_SHIM_G17P_MAX_USC_BINDS caps how many distinct executable-window
 * mappings the run may establish. Recording an Ioto program per draw out of
 * its own BO blows any useful budget immediately; sub-allocating them out of
 * arena chunks costs a handful of binds for a whole frame.
 */
static void __attribute__((destructor))
check_usc_window_pressure(void)
{
   const char *budget = getenv("ASAHI_SHIM_G17P_MAX_USC_BINDS");
   if (!budget)
      return;

   fprintf(stderr,
           "shim: G17P USC window binds=%u bytes=%llu (budget %s)\n",
           usc_window_binds, (unsigned long long)usc_window_bytes, budget);

   char *end = NULL;
   const unsigned long limit = strtoul(budget, &end, 0);
   if (end == budget || (end && *end != '\0')) {
      fprintf(stderr, "shim: bad ASAHI_SHIM_G17P_MAX_USC_BINDS '%s'\n",
              budget);
      _Exit(EXIT_FAILURE);
   }

   if (usc_window_binds > limit) {
      fprintf(stderr,
              "shim: G17P executable window pressure: %u binds exceeds the "
              "budget of %lu\n", usc_window_binds, limit);
      _Exit(EXIT_FAILURE);
   }
}

static void __attribute__((destructor))
check_queue_progress_lifecycle(void)
{
   const char *test_case = queue_progress_lifecycle_case();
   if (!test_case)
      return;

   const bool success = !strcmp(test_case, "success");
   const bool init_failure = !strcmp(test_case, "init-failure");
   const bool submit_failure = !strcmp(test_case, "submit-failure");
   const unsigned expected_submits = init_failure ? 0 : 1;
   const bool exact =
      (success || init_failure || submit_failure) &&
      queue_progress_queue_creates == 1 &&
      queue_progress_queue_destroys == 1 &&
      queue_progress_submits == expected_submits &&
      failed_submit_once == submit_failure &&
      (init_failure ||
       (queue_progress_private_sync_valid &&
        queue_progress_private_handle == 2 &&
        queue_progress_private_point == 2));

   if (!exact) {
      fprintf(stderr,
              "shim: bad Asahi queue progress lifecycle '%s': queue=%u/%u "
              "submits=%u private-shape=%u private=%u:%llu "
              "submit-failed=%u\n",
              test_case, queue_progress_queue_creates,
              queue_progress_queue_destroys, queue_progress_submits,
              queue_progress_private_sync_valid,
              queue_progress_private_handle,
              (unsigned long long)queue_progress_private_point,
              failed_submit_once);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: exact Asahi queue progress lifecycle '%s': queue=%u/%u "
           "submits=%u private=%u:%llu\n",
           test_case, queue_progress_queue_creates,
           queue_progress_queue_destroys, queue_progress_submits,
           queue_progress_private_handle,
           (unsigned long long)queue_progress_private_point);
}

static void __attribute__((destructor))
check_wsi_rollback_lifecycle(void)
{
   const char *test_case = wsi_rollback_case();
   if (!test_case)
      return;

   const unsigned expected_submits =
      !strcmp(test_case, "wayland-present-allocation") ? 5 : 0;

   if (wsi_rollback_queue_creates != 1 ||
       wsi_rollback_queue_destroys != 1 ||
       wsi_rollback_submits != expected_submits) {
      fprintf(stderr,
              "shim: bad Asahi WSI rollback '%s': queue=%u/%u "
              "hardware-submits=%u\n",
              test_case, wsi_rollback_queue_creates,
              wsi_rollback_queue_destroys,
              wsi_rollback_submits);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: exact Asahi WSI rollback '%s': queue=%u/%u "
           "hardware-submits=%u\n",
           test_case, wsi_rollback_queue_creates,
           wsi_rollback_queue_destroys, wsi_rollback_submits);
}

static bool
matrix_case_matches(const char *name,
                    const struct asahi_g17p_stream_stats *stats)
{
   if (!name)
      return true;
   if (!strcmp(name, "clear-only"))
      return stats->clear_only;
   if (!strcmp(name, "nonindexed unchanged-state body"))
      return stats->draws == 2 && stats->state_updates == 1 &&
             stats->body_only && !stats->primary_tri_merge_disable &&
             !stats->secondary_tri_merge_disable;
   if (!strcmp(name, "nonindexed line topology"))
      return stats->draws == 1 &&
             (stats->primitive_mask & (UINT32_C(1) << 1)) &&
             stats->primary_tri_merge_disable &&
             stats->secondary_tri_merge_disable;
   if (!strcmp(name, "indexed u8 without restart"))
      return stats->indexed_draws == 1 && stats->index_size_mask == 1 &&
             stats->restart_disabled;
   if (!strcmp(name, "indexed u16 topology and restart"))
      return stats->indexed_draws == 1 && stats->index_size_mask == 2 &&
             stats->restart_enabled;
   if (!strcmp(name, "indexed u32 without restart"))
      return stats->indexed_draws == 1 && stats->index_size_mask == 4 &&
             stats->restart_disabled;
   if (!strcmp(name, "indirect nonindexed"))
      return stats->indirect_nonindexed && !stats->indirect_indexed;
   if (!strcmp(name, "indirect indexed"))
      return stats->indirect_indexed;
   if (!strcmp(name, "viewport and scissor"))
      return stats->viewport_changed && stats->scissor_changed;
   if (!strcmp(name, "cull and front face"))
      return stats->cull_enabled;
   if (!strcmp(name, "depth and stencil"))
      return stats->depth_enabled && stats->stencil_enabled;
   if (!strcmp(name, "blend"))
      return stats->draws == 1 && stats->varying_fs;
   if (!strcmp(name, "fragment program without varyings"))
      return stats->no_varying_fs && stats->primary_tri_merge_disable &&
             stats->secondary_tri_merge_disable;
   if (!strcmp(name, "rasterizer discard"))
      return stats->rasterizer_discard;
   if (!strcmp(name, "linked stream with state re-emission"))
      return stats->draws == 5000 && stats->state_updates == 5000 &&
             stats->links > 0;
   return true;
}

static void
merge_stream_stats(const struct asahi_g17p_stream_stats *stats)
{
   if (stats->varying_fs && stats->fragment_code) {
      if (matrix_stats.fragment_code &&
          matrix_stats.fragment_code != stats->fragment_code)
         matrix_stats.blend_enabled = true;
      else if (!matrix_stats.fragment_code)
         matrix_stats.fragment_code = stats->fragment_code;
   }
   validated_graphics_streams++;
   matrix_stats.draws += stats->draws;
   matrix_stats.indexed_draws += stats->indexed_draws;
   matrix_stats.state_updates += stats->state_updates;
   matrix_stats.barriers += stats->barriers;
   matrix_stats.inter_draw_barriers += stats->inter_draw_barriers;
   matrix_stats.unbarriered_state_republications +=
      stats->unbarriered_state_republications;
   matrix_stats.stencil_state_updates += stats->stencil_state_updates;
   matrix_stats.links += stats->links;
   matrix_stats.viewport_count =
      MAX2(matrix_stats.viewport_count, stats->viewport_count);
   matrix_stats.max_instance_count =
      MAX2(matrix_stats.max_instance_count, stats->max_instance_count);
   matrix_stats.primitive_mask |= stats->primitive_mask;
   matrix_stats.index_size_mask |= stats->index_size_mask;
#define MERGE_FLAG(name) matrix_stats.name |= stats->name
   MERGE_FLAG(clear_only);
   MERGE_FLAG(body_only);
   MERGE_FLAG(restart_enabled);
   MERGE_FLAG(restart_disabled);
   MERGE_FLAG(indirect_nonindexed);
   MERGE_FLAG(indirect_indexed);
   MERGE_FLAG(varying_fs);
   MERGE_FLAG(no_varying_fs);
   MERGE_FLAG(viewport_changed);
   MERGE_FLAG(multi_viewport);
   MERGE_FLAG(two_viewport);
   MERGE_FLAG(four_viewport);
   MERGE_FLAG(scissor_changed);
   MERGE_FLAG(cull_enabled);
   MERGE_FLAG(depth_enabled);
   MERGE_FLAG(stencil_enabled);
   MERGE_FLAG(blend_enabled);
   MERGE_FLAG(rasterizer_discard);
   MERGE_FLAG(primary_tri_merge_disable);
   MERGE_FLAG(secondary_tri_merge_disable);
   MERGE_FLAG(layer_output);
#undef MERGE_FLAG
}

static void __attribute__((destructor))
check_matrix_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_MATRIX"))
      return;

   const uint32_t required_primitives =
      (UINT32_C(1) << 0) | (UINT32_C(1) << 1) | (UINT32_C(1) << 6) |
      (UINT32_C(1) << 9) | (UINT32_C(1) << 11);
   const bool complete =
      matrix_stats.clear_only && matrix_stats.body_only &&
      (matrix_stats.primitive_mask & required_primitives) ==
         required_primitives &&
      (matrix_stats.index_size_mask & 7) == 7 &&
      matrix_stats.restart_enabled && matrix_stats.restart_disabled &&
      matrix_stats.indirect_nonindexed && matrix_stats.indirect_indexed &&
      matrix_stats.varying_fs && matrix_stats.no_varying_fs &&
      matrix_stats.viewport_changed && matrix_stats.scissor_changed &&
      matrix_stats.cull_enabled && matrix_stats.depth_enabled &&
      matrix_stats.stencil_enabled && matrix_stats.blend_enabled &&
      matrix_stats.rasterizer_discard && matrix_stats.links &&
      matrix_stats.state_updates >= 5000;
   if (!complete) {
      fprintf(stderr,
              "shim: incomplete G17P matrix coverage: prim=%#x index=%#x "
              "draws=%u state=%u links=%u flags=%u%u%u%u%u%u%u%u%u%u%u%u%u%u\n",
              matrix_stats.primitive_mask, matrix_stats.index_size_mask,
              matrix_stats.draws, matrix_stats.state_updates,
              matrix_stats.links, matrix_stats.clear_only,
              matrix_stats.body_only, matrix_stats.restart_enabled,
              matrix_stats.restart_disabled, matrix_stats.indirect_nonindexed,
              matrix_stats.indirect_indexed, matrix_stats.varying_fs,
              matrix_stats.no_varying_fs, matrix_stats.viewport_changed,
              matrix_stats.scissor_changed, matrix_stats.cull_enabled,
              matrix_stats.depth_enabled, matrix_stats.stencil_enabled,
              matrix_stats.blend_enabled);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr, "shim: complete G17P production stream matrix coverage\n");
}

/*
 * Assert where the inter-draw VDM barrier landed.
 *
 * ASAHI_SHIM_G17P_EXPECT_INTERDRAW_BARRIERS is "<barriers>" or
 * "<barriers>:<draws>".  It checks both directions at once:
 *
 *  - the expected number of barriers that separate two draws actually
 *    appeared, so a fix that silently stopped emitting is caught; and
 *  - unbarriered_state_republications is zero, so a barrier emitted in the
 *    wrong place cannot satisfy the count.  That second number is computed
 *    from the stream itself rather than from the expectation, which is what
 *    makes this an ordering check and not a tally.
 */
static void __attribute__((destructor))
check_inter_draw_barriers(void)
{
   const char *value = getenv("ASAHI_SHIM_G17P_EXPECT_INTERDRAW_BARRIERS");
   if (!value)
      return;

   char *rest = NULL;
   const unsigned expected = strtoul(value, &rest, 10);
   const bool have_draws = rest && *rest == ':';
   const unsigned expected_draws =
      have_draws ? strtoul(rest + 1, NULL, 10) : 0;

   if (matrix_stats.inter_draw_barriers != expected ||
       matrix_stats.unbarriered_state_republications != 0 ||
       (have_draws && matrix_stats.draws != expected_draws)) {
      fprintf(stderr,
              "shim: G17P inter-draw barriers %u/%u, unbarriered state "
              "republications %u/0, draws %u/%s\n",
              matrix_stats.inter_draw_barriers, expected,
              matrix_stats.unbarriered_state_republications,
              matrix_stats.draws, have_draws ? rest + 1 : "any");
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: G17P emitted %u inter-draw VDM barriers across %u draws "
           "with no unbarriered state republication\n",
           matrix_stats.inter_draw_barriers, matrix_stats.draws);
}

/*
 * Assert the shape a draws-per-stream policy produced.
 *
 * ASAHI_SHIM_G17P_EXPECT_STREAM_SHAPE is
 * "<graphics streams>:<draws>:<state updates>[:<barriers>[:<links>]]". Draw
 * counts prevent dropped draws from passing; state counts distinguish changed
 * and reused state. The optional barrier and link counts reject per-draw cache
 * invalidation or retained renders masquerading as efficient batching.
 */
static void __attribute__((destructor))
check_stream_shape(void)
{
   const char *value = getenv("ASAHI_SHIM_G17P_EXPECT_STREAM_SHAPE");
   if (!value)
      return;

   unsigned expected_streams = 0, expected_draws = 0, expected_states = 0;
   unsigned expected_barriers = 0, expected_links = 0;
   const int fields = sscanf(value, "%u:%u:%u:%u:%u", &expected_streams,
                             &expected_draws, &expected_states,
                             &expected_barriers, &expected_links);
   if (fields < 3) {
      fprintf(stderr,
              "shim: ASAHI_SHIM_G17P_EXPECT_STREAM_SHAPE wants "
              "<streams>:<draws>:<state updates>, got '%s'\n",
              value);
      _Exit(EXIT_FAILURE);
   }

   if (validated_graphics_streams != expected_streams ||
       matrix_stats.draws != expected_draws ||
       matrix_stats.state_updates != expected_states ||
       (fields >= 4 && matrix_stats.barriers != expected_barriers) ||
       (fields == 5 && matrix_stats.links != expected_links)) {
      fprintf(stderr,
              "shim: G17P stream shape %u/%u streams, %u/%u draws, %u/%u "
              "state updates\n",
              validated_graphics_streams, expected_streams, matrix_stats.draws,
              expected_draws, matrix_stats.state_updates, expected_states);
      if (fields >= 4)
         fprintf(stderr, "shim: G17P barriers %u/%u\n",
                 matrix_stats.barriers, expected_barriers);
      if (fields == 5)
         fprintf(stderr, "shim: G17P links %u/%u\n", matrix_stats.links,
                 expected_links);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: G17P recorded %u graphics streams carrying %u draws, %u "
           "state updates and %u links\n",
           validated_graphics_streams, matrix_stats.draws,
           matrix_stats.state_updates, matrix_stats.links);
}

static void __attribute__((destructor))
check_fragment_constant_draws(void)
{
   const char *value = getenv("ASAHI_SHIM_G17P_EXPECT_FRAG_CONSTANT_DRAWS");
   if (!value)
      return;

   const unsigned expected = strtoul(value, NULL, 10);
   if (matrix_stats.draws != expected) {
      fprintf(stderr,
              "shim: G17P fragment-constant test submitted %u/%u draws\n",
              matrix_stats.draws, expected);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr, "shim: G17P fragment-constant test submitted %u draws\n",
           expected);
}

static void __attribute__((destructor))
check_msaa_draws(void)
{
   const char *value = getenv("ASAHI_SHIM_G17P_EXPECT_MSAA_DRAWS");
   if (!value)
      return;

   const unsigned expected = strtoul(value, NULL, 10);
   if (matrix_stats.draws != expected ||
       matrix_stats.state_updates != expected) {
      fprintf(stderr,
              "shim: G17P MSAA test submitted draws=%u/%u state=%u/%u\n",
              matrix_stats.draws, expected, matrix_stats.state_updates,
              expected);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr, "shim: G17P MSAA test submitted %u exact draws\n",
           expected);
}

static void __attribute__((destructor))
check_16bit_fbo_draws(void)
{
   const char *value = getenv("ASAHI_SHIM_G17P_EXPECT_16BIT_FBO_DRAWS");
   if (!value)
      return;

   const unsigned expected = strtoul(value, NULL, 10);
   if (matrix_stats.draws != expected ||
       matrix_stats.state_updates != expected) {
      fprintf(stderr,
              "shim: G17P 16-bit FBO test submitted draws=%u/%u state=%u/%u\n",
              matrix_stats.draws, expected, matrix_stats.state_updates,
              expected);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr, "shim: G17P 16-bit FBO test submitted %u exact draws\n",
           expected);
}

static void __attribute__((destructor))
check_rgb32_fbo_draws(void)
{
   const char *value = getenv("ASAHI_SHIM_G17P_EXPECT_RGB32_FBO_DRAWS");
   if (!value)
      return;

   const unsigned expected = strtoul(value, NULL, 10);
   if (matrix_stats.draws != expected ||
       matrix_stats.state_updates != expected) {
      fprintf(stderr,
              "shim: G17P RGB32 FBO test submitted draws=%u/%u state=%u/%u\n",
              matrix_stats.draws, expected, matrix_stats.state_updates,
              expected);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr, "shim: G17P RGB32 FBO test submitted %u exact draws\n",
           expected);
}

static void __attribute__((destructor))
check_multi_viewport_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_MULTI_VIEWPORT"))
      return;

   if (!matrix_stats.multi_viewport || matrix_stats.viewport_count != 16 ||
       matrix_stats.draws != 1 || matrix_stats.state_updates != 1) {
      fprintf(stderr,
              "shim: incomplete G17P multi-viewport coverage: count=%u "
              "draws=%u state=%u exact=%u\n",
              matrix_stats.viewport_count, matrix_stats.draws,
              matrix_stats.state_updates, matrix_stats.multi_viewport);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: complete G17P 16-viewport index-1 stream coverage\n");
}

static void __attribute__((destructor))
check_multiview_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_MULTIVIEW"))
      return;

   /* Honeykrisp implements multiview by emitting one native draw for each
    * view while changing the root-table ViewIndex.  The shader lowering
    * routes that value through the layer/viewport UVS output, and the render
    * command must expose both array layers to BG/EOT. */
   if (!multiview_layers_two || matrix_stats.draws != 2 ||
       matrix_stats.state_updates != 2 || !matrix_stats.layer_output) {
      fprintf(stderr,
              "shim: incomplete G17P multiview coverage: layers=%u draws=%u "
              "state=%u varying=%u layer-output=%u\n",
              multiview_layers_two, matrix_stats.draws,
              matrix_stats.state_updates, matrix_stats.varying_fs,
              matrix_stats.layer_output);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: complete G17P two-view/layer render stream coverage\n");
}

static void __attribute__((destructor))
check_vk_multi_viewport_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_VK_MULTI_VIEWPORT"))
      return;

   if (!matrix_stats.four_viewport || matrix_stats.viewport_count != 4 ||
       matrix_stats.draws != 1 || matrix_stats.state_updates != 1 ||
       matrix_stats.max_instance_count != 4 || !matrix_stats.layer_output) {
      fprintf(stderr,
              "shim: incomplete G17P Vulkan multi-viewport coverage: "
              "count=%u draws=%u state=%u instances=%u output=%u exact=%u\n",
              matrix_stats.viewport_count, matrix_stats.draws,
              matrix_stats.state_updates, matrix_stats.max_instance_count,
              matrix_stats.layer_output, matrix_stats.four_viewport);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: complete G17P Vulkan four-viewport output coverage\n");
}

static void __attribute__((destructor))
check_layer_output_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_LAYER_OUTPUT"))
      return;

   if (!multiview_layers_two || matrix_stats.draws != 1 ||
       matrix_stats.state_updates != 1 ||
       matrix_stats.max_instance_count != 2 || !matrix_stats.layer_output) {
      fprintf(stderr,
              "shim: incomplete G17P layer-output coverage: layers=%u "
              "draws=%u state=%u instances=%u layer-output=%u\n",
              multiview_layers_two, matrix_stats.draws,
              matrix_stats.state_updates, matrix_stats.max_instance_count,
              matrix_stats.layer_output);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: complete G17P application-authored layer-output coverage\n");
}

static void __attribute__((destructor))
check_depth_texture_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_DEPTH_TEXTURE"))
      return;

   /* One API depth draw, the lowered depth and physical S8 plane blits, one
    * colour draw which consumes raw depth/shadow depth/copied stencil, then
    * the standalone S8 plane blit and its stencil-tested observer. A refused
    * standalone blit omits one of the final two draws. */
   if (matrix_stats.draws != 6 || matrix_stats.state_updates != 6 ||
       !matrix_stats.depth_enabled || !matrix_stats.stencil_enabled ||
       !matrix_stats.varying_fs) {
      fprintf(stderr,
              "shim: incomplete G17P depth-texture coverage: draws=%u "
              "state=%u depth=%u stencil=%u varying=%u\n",
              matrix_stats.draws, matrix_stats.state_updates,
              matrix_stats.depth_enabled, matrix_stats.stencil_enabled,
              matrix_stats.varying_fs);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: complete G17P depth/stencil render/blit/sample coverage\n");
}

static void __attribute__((destructor))
check_stencil_blit_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_STENCIL_BLIT"))
      return;

   /* Without fragment stencil export, common meta emits one bit-selective
    * graphics pass for each stencil bit. Require all eight fixed-function
    * stencil updates plus the stencil-plane readback draw so merely accepting
    * the API command cannot pass. G17P's fragment layer input is an ordinary
    * flat varying, while the observer remains a no-varying draw, so this
    * sequence must exercise both paths. The two clears and final transfer also
    * have an exact three-launch compute envelope. All three launch tail
    * barriers are strong here: two request AGX_BARRIER_ALL and the third is
    * strengthened in place by the host-read flush (6aae441547b). */
   if (matrix_stats.draws != 9 || matrix_stats.state_updates != 9 ||
       matrix_stats.stencil_state_updates != 9 ||
       !matrix_stats.stencil_enabled || !matrix_stats.varying_fs ||
       !matrix_stats.no_varying_fs || compute_stats.launches != 3 ||
       compute_stats.barriers != 3 || compute_stats.strong_barriers != 3) {
      fprintf(stderr,
              "shim: incomplete G17P stencil blit coverage: draws=%u "
              "state=%u stencil-state=%u stencil=%u varying=%u/%u "
              "compute=%u/%u/%u\n",
              matrix_stats.draws, matrix_stats.state_updates,
              matrix_stats.stencil_state_updates,
              matrix_stats.stencil_enabled, matrix_stats.varying_fs,
              matrix_stats.no_varying_fs, compute_stats.launches,
              compute_stats.barriers, compute_stats.strong_barriers);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: complete G17P eight-pass stencil blit and readback "
           "coverage\n");
}

static void __attribute__((destructor))
check_swap_lifecycle_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_SWAP_LIFECYCLE"))
      return;

   /* Eight swaps at each of two window geometries. Each swap must retire one
    * real draw; a lost back buffer, stale context binding, or resize failure
    * makes the count short even when EGL itself reports success. */
   if (matrix_stats.draws != 16 || matrix_stats.state_updates != 16 ||
       synchronized_render_submits != 16 || !matrix_stats.no_varying_fs) {
      fprintf(stderr,
              "shim: incomplete G17P swap lifecycle: draws=%u state=%u "
              "synchronized=%u no-varying=%u\n",
              matrix_stats.draws, matrix_stats.state_updates,
              synchronized_render_submits, matrix_stats.no_varying_fs);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: complete G17P 16-frame synchronized swap/resize lifecycle "
           "coverage\n");
}

static void __attribute__((destructor))
check_compute_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_COMPUTE"))
      return;

   bool exact = true;
   const char *expected =
      getenv("ASAHI_SHIM_G17P_EXPECT_COMPUTE_COUNTS");
   const char *expected_2d =
      getenv("ASAHI_SHIM_G17P_EXPECT_2D_COMPUTE_LAUNCHES");
   if (expected) {
      unsigned launches, barriers, prologues, strong;
      char trailing;
      exact = sscanf(expected, "%u,%u,%u,%u%c", &launches, &barriers,
                     &prologues, &strong, &trailing) == 4 &&
              compute_stats.launches == launches &&
              compute_stats.barriers == barriers &&
              compute_stats.prologue_barriers == prologues &&
              compute_stats.strong_barriers == strong;
   }

   if (expected_2d) {
      unsigned launches_2d;
      char trailing;
      exact &= sscanf(expected_2d, "%u%c", &launches_2d, &trailing) == 1 &&
               compute_stats.two_dimensional_launches == launches_2d;
   }

   if (!exact || !compute_stats.launches ||
       compute_stats.launches != compute_stats.barriers ||
       !compute_stats.terminated ||
       (getenv("ASAHI_SHIM_G17P_EXPECT_STRONG_COMPUTE_BARRIER") &&
        !compute_stats.strong_barriers)) {
      fprintf(stderr,
              "shim: incomplete G17P compute coverage: launches=%u "
              "2d=%u barriers=%u prologue=%u strong=%u terminated=%u "
              "expected=%s expected-2d=%s\n",
              compute_stats.launches, compute_stats.two_dimensional_launches,
              compute_stats.barriers,
              compute_stats.prologue_barriers, compute_stats.strong_barriers,
              compute_stats.terminated, expected ? expected : "<structural>",
              expected_2d ? expected_2d : "<structural>");
      _Exit(EXIT_FAILURE);
   }
}

static void __attribute__((destructor))
check_xfb_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_XFB"))
      return;

   /* The public oracle records two XFB triangle-list draws, then consumes the
    * final byte counter through vkCmdDrawIndirectByteCountEXT. XFB lowers each
    * producer to a synthetic indexed geometry draw and brackets it with five
    * libagx counter launches; the consumer adds one GPU-side patch launch and
    * one native non-indexed indirect draw. This catches a CPU-readback path,
    * silently dropped passthrough GS, duplicate restart state, or counter work
    * that never reaches the UAPI. */
   if (matrix_stats.draws != 3 || matrix_stats.indexed_draws != 2 ||
       matrix_stats.state_updates != 3 || !matrix_stats.restart_enabled ||
       !matrix_stats.rasterizer_discard || !matrix_stats.indirect_nonindexed ||
       compute_stats.launches != 11 || compute_stats.barriers != 11 ||
       !compute_stats.terminated) {
      fprintf(stderr,
              "shim: incomplete G17P XFB coverage: draws=%u indexed=%u "
              "state=%u restart-on=%u discard=%u indirect=%u launches=%u "
              "barriers=%u terminated=%u\n",
              matrix_stats.draws, matrix_stats.indexed_draws,
              matrix_stats.state_updates, matrix_stats.restart_enabled,
              matrix_stats.rasterizer_discard,
              matrix_stats.indirect_nonindexed, compute_stats.launches,
              compute_stats.barriers, compute_stats.terminated);
      _Exit(EXIT_FAILURE);
   }

   fprintf(stderr,
           "shim: complete G17P XFB geometry/counter stream coverage\n");
}

static bool
range_contains(uint64_t base, uint64_t size, uint64_t address,
               size_t access_size)
{
   return address >= base && access_size <= size &&
          address - base <= size - access_size;
}

static const struct asahi_vm_mapping *
find_mapping(int fd, uint64_t address, size_t size)
{
   for (size_t i = shim_mappings.count; i > 0; --i) {
      const struct asahi_vm_mapping *mapping = &shim_mappings.entries[i - 1];
      if (mapping->fd == fd &&
          range_contains(mapping->address, mapping->size, address, size))
         return mapping;
   }

   return NULL;
}

static void
remove_mappings(int fd, uint64_t address, uint64_t size)
{
   const uint64_t end = address + size;
   for (size_t i = 0; i < shim_mappings.count;) {
      struct asahi_vm_mapping *mapping = &shim_mappings.entries[i];
      const uint64_t mapping_end = mapping->address + mapping->size;
      if (mapping->fd != fd || address >= mapping_end ||
          mapping->address >= end) {
         ++i;
         continue;
      }

      if (getenv("ASAHI_SHIM_TRACE_G17P_STREAM"))
         fprintf(stderr, "shim: unmap %#llx..%#llx (bound %#llx..%#llx)\n",
                 (unsigned long long)address, (unsigned long long)end,
                 (unsigned long long)mapping->address,
                 (unsigned long long)mapping_end);

      drm_shim_bo_put(mapping->bo);
      *mapping = shim_mappings.entries[--shim_mappings.count];
   }
}

static bool
add_mapping(int fd, const struct drm_asahi_gem_bind_op *op)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, op->handle);
   const bool single_page = op->flags & DRM_ASAHI_BIND_SINGLE_PAGE;
   const uint64_t object_range = single_page ? UINT64_C(0x4000) : op->range;
   if (!bo || op->offset > bo->size ||
       object_range > bo->size - op->offset) {
      if (bo)
         drm_shim_bo_put(bo);
      return false;
   }

   remove_mappings(fd, op->addr, op->range);
   if (shim_mappings.count == shim_mappings.capacity) {
      const size_t capacity = MAX2(shim_mappings.capacity * 2, 64);
      void *entries = realloc(shim_mappings.entries,
                              capacity * sizeof(*shim_mappings.entries));
      if (!entries) {
         drm_shim_bo_put(bo);
         return false;
      }
      shim_mappings.entries = entries;
      shim_mappings.capacity = capacity;
   }

   shim_mappings.entries[shim_mappings.count++] = (struct asahi_vm_mapping){
      .fd = fd,
      .address = op->addr,
      .size = op->range,
      .object_offset = op->offset,
      .single_page = single_page,
      .bo = bo,
   };
   if (op->addr >= SHIM_G17P_USC_WINDOW_BASE &&
       op->addr - SHIM_G17P_USC_WINDOW_BASE < SHIM_G17P_USC_WINDOW_SIZE) {
      usc_window_binds++;
      usc_window_bytes += op->range;
   }
   if (getenv("ASAHI_SHIM_TRACE_G17P_STREAM"))
      fprintf(stderr, "shim: map %#llx..%#llx handle %u offset %#llx\n",
              (unsigned long long)op->addr,
              (unsigned long long)(op->addr + op->range), op->handle,
              (unsigned long long)op->offset);
   return true;
}

static bool
shim_gpu_owned(void *data, uint64_t address, size_t size)
{
   return find_mapping((int)(intptr_t)data, address, size) != NULL;
}

static bool
shim_gpu_span(void *data, uint64_t address, size_t *available)
{
   const struct asahi_vm_mapping *mapping =
      find_mapping((int)(intptr_t)data, address, 1);
   if (!mapping || !available)
      return false;

   *available = mapping->size - (address - mapping->address);
   return true;
}

static bool
shim_gpu_read(void *data, uint64_t address, void *dst, size_t size)
{
   const struct asahi_vm_mapping *mapping =
      find_mapping((int)(intptr_t)data, address, size);
   if (!mapping)
      return false;

   size_t copied = 0;
   while (copied < size) {
      const uint64_t mapping_offset = address - mapping->address + copied;
      const uint64_t page_offset = mapping_offset & UINT64_C(0x3fff);
      const size_t chunk = mapping->single_page
                              ? MIN2(size - copied,
                                     (size_t)(UINT64_C(0x4000) - page_offset))
                              : size - copied;
      const uint64_t bo_offset =
         mapping->object_offset +
         (mapping->single_page ? page_offset : mapping_offset);
      const uint64_t file_offset = mapping->bo->mem_addr + bo_offset;
      if (pread(shim_device.mem_fd, (uint8_t *)dst + copied, chunk,
                (off_t)file_offset) != (ssize_t)chunk)
         return false;
      copied += chunk;
   }

   return true;
}

static int
asahi_ioctl_noop(int fd, unsigned long request, void *arg)
{
   return 0;
}

static int
asahi_ioctl_queue_create(int fd, unsigned long request, void *arg)
{
   if (queue_progress_lifecycle_case())
      queue_progress_queue_creates++;
   if (wsi_rollback_case())
      wsi_rollback_queue_creates++;

   return 0;
}

static int
asahi_ioctl_queue_destroy(int fd, unsigned long request, void *arg)
{
   if (queue_progress_lifecycle_case())
      queue_progress_queue_destroys++;
   if (wsi_rollback_case())
      wsi_rollback_queue_destroys++;

   return 0;
}

static int
asahi_ioctl_gem_bind_object(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_gem_bind_object *bind = arg;

   switch (bind->op) {
   case DRM_ASAHI_BIND_OBJECT_OP_BIND:
      if (!bind->handle ||
          bind->flags != DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS) {
         errno = EINVAL;
         return -1;
      }

      /* The no-op winsys has no firmware object table.  Preserve a stable,
       * nonzero identity so timestamp-pool recording exercises the same
       * driver lifetime path without pretending an object was submitted. */
      bind->object_handle = bind->handle;
      return 0;

   case DRM_ASAHI_BIND_OBJECT_OP_UNBIND:
      if (!bind->object_handle) {
         errno = EINVAL;
         return -1;
      }
      return 0;

   default:
      errno = EINVAL;
      return -1;
   }
}

static int
asahi_ioctl_gem_create(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_gem_create *create = arg;
   struct asahi_bo *bo = calloc(1, sizeof(*bo));

   drm_shim_bo_init(&bo->base, create->size);

   assert(UINT64_MAX - asahi.next_offset > create->size);
   bo->offset = asahi.next_offset;
   bo->gem_flags = create->flags;
   asahi.next_offset += create->size;

   create->handle = drm_shim_bo_get_handle(shim_fd, &bo->base);

   if (getenv("ASAHI_SHIM_TRACE_G17P_STREAM"))
      fprintf(stderr,
              "shim: gem_create size=%#llx flags=%#x%s vm_id=%u -> handle %u\n",
              (unsigned long long)create->size, create->flags,
              (create->flags & DRM_ASAHI_GEM_VM_PRIVATE) ? " (VM_PRIVATE)"
                                                         : "",
              create->vm_id, create->handle);

   drm_shim_bo_put(&bo->base);

   return 0;
}

static int
asahi_ioctl_gem_mmap_offset(int fd, unsigned long request, void *arg)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct drm_asahi_gem_mmap_offset *map = arg;
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, map->handle);

   map->offset = drm_shim_bo_get_mmap_offset(shim_fd, bo);

   drm_shim_bo_put(bo);

   return 0;
}

static int
asahi_ioctl_get_param(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_get_params *gp = arg;

   switch (gp->param_group) {
   case 0: {
      /* Mirror the kernel, which copies min(sizeof(params), data.size) and
       * reports back what it wrote (file.rs get_params).  Asserting equality
       * instead made the shim unusable against any caller whose header
       * differs by a field. */
      size_t size = MIN2((size_t)gp->size, sizeof(struct drm_asahi_params_global));
      memcpy((void *)gp->pointer, shim_params(), size);
      gp->size = size;
      return 0;
   }

   default:
      fprintf(stderr, "Unknown DRM_IOCTL_ASAHI_GET_PARAMS %d\n",
              gp->param_group);
      return -1;
   }
}

/* The kernel's VA rules, mirrored so a bad address layout fails here instead
 * of costing a boot on hardware.
 *
 * drivers/gpu/drm/asahi/file.rs rejects a bind that is not contained in the
 * device user range, or that overlaps the VM's kernel range, both with EINVAL.
 * A no-op VM_BIND accepts every address, so an allocator geometry that the
 * real driver refuses looks perfectly healthy under the shim.  These two
 * checks are the whole difference between reproducing that on a laptop and
 * discovering it on the target. */
static struct {
   uint64_t kernel_start, kernel_end;
   bool valid;
} shim_vm;

static int
asahi_ioctl_vm_create(int fd, unsigned long request, void *arg)
{
   struct drm_asahi_vm_create *create = arg;

   shim_vm.kernel_start = create->kernel_start;
   shim_vm.kernel_end = create->kernel_end;
   shim_vm.valid = true;
   create->vm_id = 1;
   return 0;
}

static int
asahi_ioctl_vm_bind(int fd, unsigned long request, void *arg)
{
   const struct drm_asahi_vm_bind *bind = arg;
   const struct drm_asahi_params_global *params = shim_params();

   for (uint32_t i = 0; i < bind->num_binds; ++i) {
      const struct drm_asahi_gem_bind_op *op =
         (const struct drm_asahi_gem_bind_op *)(uintptr_t)(bind->userptr +
                                                          (uint64_t)i *
                                                             bind->stride);
      uint64_t start = op->addr;
      uint64_t end = start + op->range;

      if (!op->range) {
         fprintf(stderr, "shim: gem_bind: zero-sized range at %#llx\n",
                 (unsigned long long)start);
         invalid_vm_binds++;
         return -EINVAL;
      }

      if (end < start || start < params->vm_start || end > params->vm_end) {
         fprintf(stderr,
                 "shim: gem_bind: Invalid map range %#llx..%#llx "
                 "(not contained in user range %#llx..%#llx)\n",
                 (unsigned long long)start, (unsigned long long)end,
                 (unsigned long long)params->vm_start,
                 (unsigned long long)params->vm_end);
         return -EINVAL;
      }

      /* T8140 installs four pair/generation AXFB aliases in every
       * render-capable client root, and that range lies inside G17P's USC
       * heap. The kernel refuses a bind that lands on them; a run that gets
       * its allocator geometry wrong here produces thousands of rejected
       * binds on hardware and nothing at all under a no-op shim. */
      if (params == &params_g17p &&
          start < SHIM_G17P_RENDER_AUX_ALIAS_END &&
          SHIM_G17P_RENDER_AUX_ALIAS_START < end) {
         fprintf(stderr,
                 "shim: gem_bind: Invalid map range %#llx..%#llx "
                 "(overlaps the kernel render-aux alias %#llx..%#llx)\n",
                 (unsigned long long)start, (unsigned long long)end,
                 (unsigned long long)SHIM_G17P_RENDER_AUX_ALIAS_START,
                 (unsigned long long)SHIM_G17P_RENDER_AUX_ALIAS_END);
         return -EINVAL;
      }

      if (shim_vm.valid && start < shim_vm.kernel_end &&
          shim_vm.kernel_start < end) {
         fprintf(stderr,
                 "shim: gem_bind: Invalid map range %#llx..%#llx "
                 "(intrudes in kernel range %#llx..%#llx)\n",
                 (unsigned long long)start, (unsigned long long)end,
                 (unsigned long long)shim_vm.kernel_start,
                 (unsigned long long)shim_vm.kernel_end);
         return -EINVAL;
      }

      /* file.rs rejects any user bind overlapping a driver-reserved range,
       * and the client null view is reserved for the VM's whole lifetime. */
      if (t8140_registry_active() &&
          range_overlaps(start, op->range, T8140_CLIENT_NULL_VIEW_START,
                         T8140_CLIENT_NULL_VIEW_SIZE)) {
         fprintf(stderr,
                 "shim: gem_bind: Invalid map range %#llx..%#llx (overlaps "
                 "the driver-reserved T8140 client null view %#llx..%#llx)\n",
                 (unsigned long long)start, (unsigned long long)end,
                 (unsigned long long)T8140_CLIENT_NULL_VIEW_START,
                 (unsigned long long)(T8140_CLIENT_NULL_VIEW_START +
                                      T8140_CLIENT_NULL_VIEW_SIZE));
         return -EINVAL;
      }

      if (op->flags & DRM_ASAHI_BIND_UNBIND) {
         remove_mappings(fd, op->addr, op->range);
         t8140_untrack_range(op->addr, op->range);
      } else if (!add_mapping(fd, op)) {
         fprintf(stderr, "shim: gem_bind: invalid BO mapping\n");
         return -EINVAL;
      } else {
         /* A repeated single page is not the same physical layout as any
          * native restricted resource; the kernel refuses the overlap outright
          * (Vm::validate_t8140_native_context_binding). */
         if (t8140_registry_active() &&
             (op->flags & DRM_ASAHI_BIND_SINGLE_PAGE)) {
            for (unsigned v = 0; v < ARRAY_SIZE(t8140_context_views); ++v) {
               if (range_overlaps(op->addr, op->range,
                                  t8140_context_views[v].source_address,
                                  t8140_context_views[v].size)) {
                  fprintf(stderr,
                          "shim: gem_bind: single-page map %#llx..%#llx "
                          "overlaps native context view %u\n",
                          (unsigned long long)start, (unsigned long long)end,
                          v);
                  return -EINVAL;
               }
            }
         }
         t8140_track_bind(op);
      }
   }

   return 0;
}

/* A submission's ordering is carried entirely by the barrier fields in each
 * command header and by which ioctl the in/out syncobjs ride on. Both are
 * write-only from userspace: the kernel is the only thing that has ever read
 * them, so a mistake in what Honeykrisp asks for and a kernel that does not
 * honour the request look identical from a readback that comes back stale.
 * Set ASAHI_SHIM_SUBMIT_LOG to a path and every submit ioctl is appended to
 * it, one line per command, exactly as the kernel would parse it. */
static void
asahi_log_submit(const struct drm_asahi_submit *submit)
{
   const char *path = getenv("ASAHI_SHIM_SUBMIT_LOG");
   if (!path)
      return;

   FILE *log = fopen(path, "a");
   if (!log)
      return;

   static unsigned ioctl_index;
   fprintf(log, "submit %u in_syncs=%u out_syncs=%u bytes=%u\n", ioctl_index++,
           submit->in_sync_count, submit->out_sync_count, submit->cmdbuf_size);

   const uint8_t *cmdbuf = (const uint8_t *)(uintptr_t)submit->cmdbuf;
   uint32_t offset = 0;
   while (offset + sizeof(struct drm_asahi_cmd_header) <= submit->cmdbuf_size) {
      struct drm_asahi_cmd_header header;
      memcpy(&header, cmdbuf + offset, sizeof(header));

      const char *name = header.cmd_type == DRM_ASAHI_CMD_RENDER    ? "render"
                         : header.cmd_type == DRM_ASAHI_CMD_COMPUTE ? "compute"
                         : header.cmd_type == DRM_ASAHI_SET_VERTEX_ATTACHMENTS
                            ? "set-vertex-attachments"
                         : header.cmd_type == DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS
                            ? "set-fragment-attachments"
                         : header.cmd_type == DRM_ASAHI_SET_COMPUTE_ATTACHMENTS
                            ? "set-compute-attachments"
                            : "unknown";

      fprintf(log, "  cmd %s type=%u size=%u vdm_barrier=%u cdm_barrier=%u",
              name, header.cmd_type, header.size, header.vdm_barrier,
              header.cdm_barrier);

      /* The declared control-stream extent is the other write-only field the
       * kernel acts on. Print it: a stream whose declared end is far past the
       * records it actually contains is indistinguishable from a correct one
       * until the hardware reads the gap. */
      if (header.cmd_type == DRM_ASAHI_CMD_COMPUTE &&
          header.size >= sizeof(struct drm_asahi_cmd_compute)) {
         struct drm_asahi_cmd_compute compute;
         memcpy(&compute, cmdbuf + offset + sizeof(header), sizeof(compute));
         fprintf(log, " cdm_base=0x%llx cdm_end=0x%llx cdm_len=%llu",
                 (unsigned long long)compute.cdm_ctrl_stream_base,
                 (unsigned long long)compute.cdm_ctrl_stream_end,
                 (unsigned long long)(compute.cdm_ctrl_stream_end -
                                      compute.cdm_ctrl_stream_base));
      } else if (header.cmd_type == DRM_ASAHI_CMD_RENDER &&
                 header.size >= sizeof(struct drm_asahi_cmd_render)) {
         struct drm_asahi_cmd_render render;
         memcpy(&render, cmdbuf + offset + sizeof(header), sizeof(render));
         fprintf(log, " vdm_base=0x%llx",
                 (unsigned long long)render.vdm_ctrl_stream_base);
      }
      fprintf(log, "\n");

      offset += sizeof(header) + header.size;
   }
   if (offset != submit->cmdbuf_size)
      fprintf(log, "  ragged walk ended at %u of %u\n", offset,
              submit->cmdbuf_size);

   fclose(log);
}

/* 0 (default) silent, 1 report, 2 reject. See asahi_g17p_stream_validator.h. */
static unsigned
g17p_strict_descriptors(void)
{
   const char *value = getenv("ASAHI_SHIM_G17P_STRICT_DESCRIPTORS");

   if (value == NULL || value[0] == '\0')
      return 0;

   return (unsigned)strtoul(value, NULL, 10);
}

/*
 * Validation normally rejects the submit, which stops the application at the
 * first complaint. A long-running client -- a game walking a level -- then
 * validates only the streams it emitted before that point. Warn-only mode
 * reports every failure and lets the run continue, so one session covers the
 * whole stream population instead of its prefix.
 */
static bool
g17p_stream_warn_only(void)
{
   const char *value = getenv("ASAHI_SHIM_G17P_STREAM_WARN_ONLY");

   return value != NULL && value[0] != '\0' && strcmp(value, "0");
}

static unsigned g17p_stream_failures;

static int
g17p_stream_reject(const char *what)
{
   g17p_stream_failures++;

   if (!g17p_stream_warn_only())
      return -EINVAL;

   fprintf(stderr, "shim: continuing after %s failure %u (warn-only)\n", what,
           g17p_stream_failures);
   return 0;
}

static int
asahi_ioctl_submit(int fd, unsigned long request, void *arg)
{
   if (g17g_compile_only()) {
      fprintf(stderr, "shim: G17G compiler-only profile rejects GPU submission\n");
      errno = EOPNOTSUPP;
      return -1;
   }

   if (g18g_profile()) {
      fprintf(stderr, "shim: G18G profile has no kernel contract and rejects "
                      "GPU submission\n");
      errno = EOPNOTSUPP;
      return -1;
   }

   const struct drm_asahi_submit *submit = arg;
   submit_calls++;
   asahi_log_submit(submit);
   if (queue_progress_lifecycle_case()) {
      const struct drm_asahi_sync *syncs =
         (const void *)(uintptr_t)submit->syncs;
      queue_progress_submits++;
      if (syncs && submit->out_sync_count) {
         const struct drm_asahi_sync *private_sync =
            &syncs[submit->in_sync_count + submit->out_sync_count - 1];
         queue_progress_private_sync_valid =
            private_sync->sync_type == DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ &&
            private_sync->handle != 0;
         queue_progress_private_handle = private_sync->handle;
         queue_progress_private_point = private_sync->timeline_value;
      }
   }
   if (wsi_rollback_case())
      wsi_rollback_submits++;

   if (getenv("ASAHI_SHIM_FAIL_SUBMIT_ONCE") && !failed_submit_once) {
      failed_submit_once = true;
      errno = EIO;
      return -1;
   }

   const char *fail_submit_at = getenv("ASAHI_SHIM_FAIL_SUBMIT_AT");
   if (fail_submit_at && strtoul(fail_submit_at, NULL, 10) == submit_calls) {
      errno = EIO;
      return -1;
   }

   /* The kernel's native-context gate runs for every render submission, not
    * only when the stream oracle is enabled, so check it first. */
   if (t8140_registry_active()) {
      const uint8_t *scan = (const void *)(uintptr_t)submit->cmdbuf;
      size_t at = 0;
      bool complete = true;

      while (at + sizeof(struct drm_asahi_cmd_header) <= submit->cmdbuf_size) {
         const struct drm_asahi_cmd_header *header = (const void *)(scan + at);
         const size_t command_size = sizeof(*header) + header->size;

         if (command_size > submit->cmdbuf_size - at)
            break;

         if (header->cmd_type == DRM_ASAHI_CMD_RENDER)
            complete &= t8140_check_context_views();

         at += command_size;
      }

      if (!complete && getenv("ASAHI_SHIM_G17P_ENFORCE_CONTEXT_VIEWS")) {
         fprintf(stderr,
                 "shim: G17P render: STAGE FAIL stage=new-user-render-storage "
                 "errno=-2\n");
         errno = ENOENT;
         return -1;
      }
   }

   const char *validate = getenv("ASAHI_SHIM_VALIDATE_G17P_STREAM");
   if (!validate || !strcmp(validate, "0") || shim_params() != &params_g17p)
      return 0;

   const uint8_t *cmdbuf = (const void *)(uintptr_t)submit->cmdbuf;
   size_t cursor = 0;
   while (cursor < submit->cmdbuf_size) {
      if (submit->cmdbuf_size - cursor < sizeof(struct drm_asahi_cmd_header))
         return -EINVAL;

      const struct drm_asahi_cmd_header *header = (const void *)(cmdbuf + cursor);
      const size_t command_size = sizeof(*header) + header->size;
      if (command_size > submit->cmdbuf_size - cursor)
         return -EINVAL;

      if (header->cmd_type == DRM_ASAHI_CMD_RENDER) {
         if (header->size < offsetof(struct drm_asahi_cmd_render,
                                     vdm_ctrl_stream_base) +
                               sizeof(uint64_t))
            return -EINVAL;

         struct drm_asahi_cmd_render render = {0};
         memcpy(&render, header + 1, MIN2(sizeof(render), header->size));

         if (getenv("ASAHI_SHIM_G17P_EXPECT_INTEGER_DEPTH_BIAS") &&
             !(render.flags & DRM_ASAHI_RENDER_DBIAS_IS_INT)) {
            fprintf(stderr, "shim: G17P render lost integer depth-bias mode\n");
            return -EINVAL;
         }

         /* G17P creates a per-job render-storage view from the fixed HAL200
          * context aperture.  Unlike ordinary client-VM shader and resource
          * mappings, its source GEM must therefore be visible outside the
          * private VM.  Mirror that kernel requirement so a private context
          * allocation fails on the host with the same ENOENT class instead
          * of reaching hardware first. */
         const struct asahi_vm_mapping *context =
            find_mapping(fd, UINT64_C(0x1000000000), UINT64_C(0x10000));
         if (!context ||
             (asahi_bo(context->bo)->gem_flags & DRM_ASAHI_GEM_VM_PRIVATE)) {
            fprintf(stderr,
                    "shim: G17P render context is absent from shared job view\n");
            return -ENOENT;
         }

         if (submit->in_sync_count > 0 && submit->out_sync_count == 2)
            synchronized_render_submits++;
         if (getenv("ASAHI_SHIM_G17P_EXPECT_MULTIVIEW") ||
             getenv("ASAHI_SHIM_G17P_EXPECT_LAYER_OUTPUT"))
            multiview_layers_two |= render.layers == 2;
         const struct asahi_g17p_stream_memory memory = {
            .data = (void *)(intptr_t)fd,
            .read = shim_gpu_read,
            .owned = shim_gpu_owned,
            .span = shim_gpu_span,
            .verbose = getenv("ASAHI_SHIM_TRACE_G17P_STREAM") != NULL,
            .strict_descriptors = g17p_strict_descriptors(),
         };
         struct asahi_g17p_stream_stats stats;
         const struct asahi_g17p_stream_config config = {
            .root = render.vdm_ctrl_stream_base,
            .scissor_base = render.isp_scissor_base,
            .depth_bias_base = render.isp_dbias_base,
         };
         if (!asahi_g17p_render_aux_validate(&memory, &render) ||
             !asahi_g17p_stream_validate(&memory, &config, &stats)) {
            g17p_streams_rejected++;
            const int rejected = g17p_stream_reject("VDM");
            if (rejected)
               return rejected;
            cursor += command_size;
            continue;
         }
         const char *matrix_case = getenv("ASAHI_SHIM_G17P_CASE");
         const bool auxiliary_clear = stats.clear_only && matrix_case &&
                                      strcmp(matrix_case, "clear-only");
         if (!auxiliary_clear && !matrix_case_matches(matrix_case, &stats)) {
            fprintf(stderr, "shim: G17P matrix case '%s' semantic mismatch\n",
                    matrix_case);
            g17p_streams_rejected++;
            return -EINVAL;
         }
         if (!auxiliary_clear && matrix_case && !strcmp(matrix_case, "blend"))
            stats.blend_enabled = true;
         merge_stream_stats(&stats);
         g17p_streams_validated++;
         fprintf(stderr,
                 "shim: G17P VDM validated: %u draws, %u indexed, %u state "
                 "updates, %u links, %u barriers (%u inter-draw, %u "
                 "unbarriered republications)\n",
                 stats.draws, stats.indexed_draws, stats.state_updates,
                 stats.links, stats.barriers, stats.inter_draw_barriers,
                 stats.unbarriered_state_republications);
      } else if (header->cmd_type == DRM_ASAHI_CMD_COMPUTE) {
         if (header->size < offsetof(struct drm_asahi_cmd_compute,
                                     cdm_ctrl_stream_end) +
                               sizeof(uint64_t))
            return -EINVAL;

         struct drm_asahi_cmd_compute compute = {0};
         memcpy(&compute, header + 1, MIN2(sizeof(compute), header->size));
         if (getenv("ASAHI_SHIM_G17P_TRUNCATE_COMPUTE_END") &&
             compute.cdm_ctrl_stream_end >= 4)
            compute.cdm_ctrl_stream_end -= 4;
         const struct asahi_g17p_stream_memory memory = {
            .data = (void *)(intptr_t)fd,
            .read = shim_gpu_read,
            .owned = shim_gpu_owned,
            .span = shim_gpu_span,
            .verbose = getenv("ASAHI_SHIM_TRACE_G17P_STREAM") != NULL,
            .strict_descriptors = g17p_strict_descriptors(),
         };
         struct asahi_g17p_compute_stats stats;
         if (!asahi_g17p_compute_validate(&memory, &compute, &stats)) {
            g17p_streams_rejected++;
            const int rejected = g17p_stream_reject("CDM");
            if (rejected)
               return rejected;
            cursor += command_size;
            continue;
         }
         compute_stats.launches += stats.launches;
         compute_stats.two_dimensional_launches +=
            stats.two_dimensional_launches;
         compute_stats.barriers += stats.barriers;
         compute_stats.prologue_barriers += stats.prologue_barriers;
         compute_stats.strong_barriers += stats.strong_barriers;
         compute_stats.sstt_flushes += stats.sstt_flushes;
         compute_stats.terminated |= stats.terminated;
         g17p_streams_validated++;
         fprintf(stderr,
                 "shim: G17P CDM validated: %u launches (%u 2D), %u barriers, "
                 "%u prologue (%u strong), %u SSTT flushes\n",
                 stats.launches, stats.two_dimensional_launches,
                 stats.barriers, stats.prologue_barriers,
                 stats.strong_barriers, stats.sstt_flushes);
      }

      cursor += command_size;
   }

   return cursor == submit->cmdbuf_size ? 0 : -EINVAL;
}

static ioctl_fn_t driver_ioctls[] = {
   [DRM_ASAHI_GET_PARAMS] = asahi_ioctl_get_param,
   [DRM_ASAHI_VM_CREATE] = asahi_ioctl_vm_create,
   [DRM_ASAHI_VM_DESTROY] = asahi_ioctl_noop,
   [DRM_ASAHI_VM_BIND] = asahi_ioctl_vm_bind,
   [DRM_ASAHI_GEM_CREATE] = asahi_ioctl_gem_create,
   [DRM_ASAHI_GEM_MMAP_OFFSET] = asahi_ioctl_gem_mmap_offset,
   [DRM_ASAHI_QUEUE_CREATE] = asahi_ioctl_queue_create,
   [DRM_ASAHI_QUEUE_DESTROY] = asahi_ioctl_queue_destroy,
   [DRM_ASAHI_GEM_BIND_OBJECT] = asahi_ioctl_gem_bind_object,
   [DRM_ASAHI_SUBMIT] = asahi_ioctl_submit,
};

void
drm_shim_driver_init(void)
{
   shim_device.bus_type = DRM_BUS_PLATFORM;
   shim_device.driver_name = "asahi";
   shim_device.driver_ioctls = driver_ioctls;
   shim_device.driver_ioctl_count = ARRAY_SIZE(driver_ioctls);

   const char *chip = getenv("ASAHI_SHIM_CHIP");
   const bool g17p = chip && !strcmp(chip, "g17p");

   drm_shim_override_file(g17g_compile_only() ? "DRIVER=asahi\n"
                                 "OF_FULLNAME=/soc/agx\n"
                                 "OF_COMPATIBLE_0=apple,gpu-g17g\n"
                                 "OF_COMPATIBLE_N=1\n"
                               : g18g_profile() ? "DRIVER=asahi\n"
                                 "OF_FULLNAME=/soc/agx\n"
                                 "OF_COMPATIBLE_0=apple,gpu-g18g\n"
                                 "OF_COMPATIBLE_N=1\n"
                               : g17p ? "DRIVER=asahi\n"
                                 "OF_FULLNAME=/soc/agx\n"
                                 "OF_COMPATIBLE_0=apple,gpu-g17p\n"
                                 "OF_COMPATIBLE_N=1\n"
                               : "DRIVER=asahi\n"
                                 "OF_FULLNAME=/soc/agx\n"
                                 "OF_COMPATIBLE_0=apple,gpu-g13g\n"
                                 "OF_COMPATIBLE_N=1\n",
                          "/sys/dev/char/%d:%d/device/uevent", DRM_MAJOR,
                          render_node_minor);
}
