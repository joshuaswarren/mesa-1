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

static const struct drm_asahi_params_global params_g13g = {
   .gpu_generation = 13,
   .gpu_variant = 'G',
   .gpu_revision = 0,
   .vm_start = 0x4000,
   .vm_end = 0x7fffff8000ull,
   .command_timestamp_frequency_hz = 1000000000,
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

static const struct drm_asahi_params_global *
shim_params(void)
{
   const char *chip = getenv("ASAHI_SHIM_CHIP");

   if (chip && !strcmp(chip, "g17p"))
      return &params_g17p;

   return &params_g13g;
}

struct asahi_bo {
   struct shim_bo base;
   uint32_t offset;
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
   struct shim_bo *bo;
};

static struct {
   struct asahi_vm_mapping *entries;
   size_t count;
   size_t capacity;
} shim_mappings;

static struct asahi_g17p_stream_stats matrix_stats;
static struct asahi_g17p_compute_stats compute_stats;

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
             stats->body_only;
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
      return stats->no_varying_fs;
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
   matrix_stats.draws += stats->draws;
   matrix_stats.indexed_draws += stats->indexed_draws;
   matrix_stats.state_updates += stats->state_updates;
   matrix_stats.links += stats->links;
   matrix_stats.viewport_count =
      MAX2(matrix_stats.viewport_count, stats->viewport_count);
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
   MERGE_FLAG(scissor_changed);
   MERGE_FLAG(cull_enabled);
   MERGE_FLAG(depth_enabled);
   MERGE_FLAG(stencil_enabled);
   MERGE_FLAG(blend_enabled);
   MERGE_FLAG(rasterizer_discard);
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
check_compute_coverage(void)
{
   if (!getenv("ASAHI_SHIM_G17P_EXPECT_COMPUTE"))
      return;

   if (!compute_stats.launches ||
       compute_stats.launches != compute_stats.barriers ||
       !compute_stats.terminated ||
       (getenv("ASAHI_SHIM_G17P_EXPECT_STRONG_COMPUTE_BARRIER") &&
        !compute_stats.strong_barriers)) {
      fprintf(stderr,
              "shim: incomplete G17P compute coverage: launches=%u "
              "barriers=%u strong=%u terminated=%u\n",
              compute_stats.launches, compute_stats.barriers,
              compute_stats.strong_barriers, compute_stats.terminated);
      _Exit(EXIT_FAILURE);
   }
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

      drm_shim_bo_put(mapping->bo);
      *mapping = shim_mappings.entries[--shim_mappings.count];
   }
}

static bool
add_mapping(int fd, const struct drm_asahi_gem_bind_op *op)
{
   struct shim_fd *shim_fd = drm_shim_fd_lookup(fd);
   struct shim_bo *bo = drm_shim_bo_lookup(shim_fd, op->handle);
   if (!bo || op->offset > bo->size || op->range > bo->size - op->offset) {
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
      .bo = bo,
   };
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

   const uint64_t bo_offset =
      mapping->object_offset + (address - mapping->address);
   const uint64_t file_offset = mapping->bo->mem_addr + bo_offset;
   return pread(shim_device.mem_fd, dst, size, (off_t)file_offset) ==
          (ssize_t)size;
}

static int
asahi_ioctl_noop(int fd, unsigned long request, void *arg)
{
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
   asahi.next_offset += create->size;

   create->handle = drm_shim_bo_get_handle(shim_fd, &bo->base);

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

      if (end < start || start < params->vm_start || end > params->vm_end) {
         fprintf(stderr,
                 "shim: gem_bind: Invalid map range %#llx..%#llx "
                 "(not contained in user range %#llx..%#llx)\n",
                 (unsigned long long)start, (unsigned long long)end,
                 (unsigned long long)params->vm_start,
                 (unsigned long long)params->vm_end);
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

      if (op->flags & DRM_ASAHI_BIND_UNBIND) {
         remove_mappings(fd, op->addr, op->range);
      } else if (!add_mapping(fd, op)) {
         fprintf(stderr, "shim: gem_bind: invalid BO mapping\n");
         return -EINVAL;
      }
   }

   return 0;
}

static int
asahi_ioctl_submit(int fd, unsigned long request, void *arg)
{
   const struct drm_asahi_submit *submit = arg;
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
         const struct asahi_g17p_stream_memory memory = {
            .data = (void *)(intptr_t)fd,
            .read = shim_gpu_read,
            .owned = shim_gpu_owned,
            .span = shim_gpu_span,
            .verbose = getenv("ASAHI_SHIM_TRACE_G17P_STREAM") != NULL,
         };
         struct asahi_g17p_stream_stats stats;
         const struct asahi_g17p_stream_config config = {
            .root = render.vdm_ctrl_stream_base,
            .scissor_base = render.isp_scissor_base,
            .depth_bias_base = render.isp_dbias_base,
         };
         if (!asahi_g17p_render_aux_validate(&memory, &render) ||
             !asahi_g17p_stream_validate(&memory, &config, &stats))
            return -EINVAL;
         const char *matrix_case = getenv("ASAHI_SHIM_G17P_CASE");
         const bool auxiliary_clear = stats.clear_only && matrix_case &&
                                      strcmp(matrix_case, "clear-only");
         if (!auxiliary_clear && !matrix_case_matches(matrix_case, &stats)) {
            fprintf(stderr, "shim: G17P matrix case '%s' semantic mismatch\n",
                    matrix_case);
            return -EINVAL;
         }
         if (!auxiliary_clear && matrix_case && !strcmp(matrix_case, "blend"))
            stats.blend_enabled = true;
         merge_stream_stats(&stats);
         fprintf(stderr,
                 "shim: G17P VDM validated: %u draws, %u indexed, %u state "
                 "updates, %u links\n",
                 stats.draws, stats.indexed_draws, stats.state_updates,
                 stats.links);
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
         };
         struct asahi_g17p_compute_stats stats;
         if (!asahi_g17p_compute_validate(&memory, &compute, &stats))
            return -EINVAL;
         compute_stats.launches += stats.launches;
         compute_stats.barriers += stats.barriers;
         compute_stats.strong_barriers += stats.strong_barriers;
         compute_stats.terminated |= stats.terminated;
         fprintf(stderr,
                 "shim: G17P CDM validated: %u launches, %u barriers "
                 "(%u strong)\n",
                 stats.launches, stats.barriers, stats.strong_barriers);
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
   [DRM_ASAHI_QUEUE_CREATE] = asahi_ioctl_noop,
   [DRM_ASAHI_QUEUE_DESTROY] = asahi_ioctl_noop,
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

   drm_shim_override_file(g17p ? "DRIVER=asahi\n"
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
