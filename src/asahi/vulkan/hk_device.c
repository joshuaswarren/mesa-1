/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_device.h"

#include "agx_bg_eot.h"
#include "agx_helpers.h"
#include "agx_scratch.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_table.h"
#include "hk_entrypoints.h"
#include "hk_instance.h"
#include "hk_native_sampler.h"
#include "hk_physical_device.h"
#include "hk_shader.h"

#include "asahi/genxml/agx_pack.h"
#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_device.h"
#include "asahi/lib/agx3_g17p_grid_helper_core.h"
#include "compiler/nir/nir_builder.h"
#include "poly/geometry.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/simple_mtx.h"
#include "vulkan/vulkan_core.h"
#include "vulkan/wsi/wsi_common.h"
#include "layout.h"
#include "vk_cmd_enqueue_entrypoints.h"
#include "vk_common_entrypoints.h"
#include "vk_debug_utils.h"
#include "vk_device.h"
#include "vk_drm_syncobj.h"
#include "vk_pipeline_cache.h"
#include "vk_shader.h"

#include <fcntl.h>
#include <xf86drm.h>

/* clang-format off */
static const struct debug_named_value hk_perf_options[] = {
   {"notess",    HK_PERF_NOTESS,   "Skip draws with tessellation"},
   {"noborder",  HK_PERF_NOBORDER, "Disable custom border colour emulation"},
   {"nobarrier", HK_PERF_NOBARRIER,"Ignore pipeline barriers"},
   {"batch",     HK_PERF_BATCH,    "Batch submissions"},
   {"norobust",  HK_PERF_NOROBUST, "Disable robustness"},
   DEBUG_NAMED_VALUE_END
};
/* clang-format on */

/*
 * We preupload some constants so we can cheaply reference later without extra
 * allocation and copying.
 *
 * TODO: This is small, don't waste a whole BO.
 */
static VkResult
hk_upload_rodata(struct hk_device *dev)
{
   const unsigned sampler_size =
      agx_chip_sampler_heap_element_size(dev->dev.chip);
   if (!sampler_size)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   const size_t rodata_size =
      align(sampler_size, sizeof(uint64_t)) + sizeof(struct poly_heap);
   dev->rodata.bo =
      agx_bo_create(&dev->dev, rodata_size, 0, 0, "Read only data");

   if (!dev->rodata.bo)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   uint8_t *map = agx_bo_map(dev->rodata.bo);
   uint32_t offs = 0;

   offs = align(offs, 8);
   if (agx_chip_command_stream_generation(dev->dev.chip) ==
       AGX_CS_GENERATION_AGX2) {
      agx_pack(&dev->dev.txf_sampler, USC_SAMPLER, cfg) {
         cfg.start = 0;
         cfg.count = 1;
         cfg.buffer = dev->rodata.bo->va->addr + offs;
      }
   } else {
      memset(&dev->dev.txf_sampler, 0, sizeof(dev->dev.txf_sampler));
   }

   struct hk_native_sampler_descriptor txf;
   if (!hk_pack_native_txf_sampler(dev->dev.chip, &txf)) {
      agx_bo_unreference(&dev->dev, dev->rodata.bo);
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   memcpy(map + offs, &txf, sampler_size);
   offs += sampler_size;

   /* The heap descriptor isn't strictly readonly data, but we only have a
    * single instance of it device-wide and -- after initializing at heap
    * allocate time -- it is read-only from the CPU perspective. The GPU uses it
    * for scratch, but is required to reset it after use to ensure resubmitting
    * the same command buffer works.
    *
    * So, we allocate it here for convenience.
    */
   offs = align(offs, sizeof(uint64_t));
   dev->rodata.heap = dev->rodata.bo->va->addr + offs;
   offs += sizeof(struct poly_heap);

   return VK_SUCCESS;
}

static uint32_t
internal_key_hash(const void *key_)
{
   const struct hk_internal_key *key = key_;

   return _mesa_hash_data(key, sizeof(struct hk_internal_key) + key->key_size);
}

static bool
internal_key_equal(const void *a_, const void *b_)
{
   const struct hk_internal_key *a = a_;
   const struct hk_internal_key *b = b_;

   return a->builder == b->builder && a->key_size == b->key_size &&
          memcmp(a->key, b->key, a->key_size) == 0;
}

static VkResult
hk_init_internal_shaders(struct hk_internal_shaders *s)
{
   s->ht = _mesa_hash_table_create(NULL, internal_key_hash, internal_key_equal);
   if (!s->ht)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   simple_mtx_init(&s->lock, mtx_plain);
   return VK_SUCCESS;
}

static void
hk_destroy_internal_shaders(struct hk_device *dev,
                            struct hk_internal_shaders *s, bool part)
{
   hash_table_foreach(s->ht, ent) {
      if (part) {
         struct agx_shader_part *part = ent->data;
         free(part->binary);

         /* The agx_shader_part itself is ralloc'd against the hash table so
          * will be freed.
          */
      } else {
         struct hk_api_shader *obj = ent->data;
         hk_api_shader_destroy(&dev->vk, &obj->vk, NULL);
      }
   }

   _mesa_hash_table_destroy(s->ht, NULL);
   simple_mtx_destroy(&s->lock);
}

DERIVE_HASH_TABLE(agx_sampler_packed);

static VkResult
hk_init_sampler_heap(struct hk_device *dev, struct hk_sampler_heap *h)
{
   const unsigned sampler_size =
      agx_chip_sampler_heap_element_size(dev->dev.chip);
   if (!sampler_size ||
       agx_chip_sampler_generation(dev->dev.chip) ==
          AGX_SAMPLER_GENERATION_UNKNOWN)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   h->ht = agx_sampler_packed_table_create(NULL);
   if (!h->ht)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   VkResult result =
      hk_descriptor_table_init(dev, &h->table, sampler_size, 1024,
                               dev->dev.chip == AGX_CHIP_G17P ? 2048 : 1024);

   if (result != VK_SUCCESS) {
      ralloc_free(h->ht);
      return result;
   }

   if (dev->dev.chip == AGX_CHIP_G17P) {
      /* Native indexed read() and sample() share the sampler table prefix.
       * Reserve texel-fetch sampler zero once, so mixed shaders need no
       * per-fetch pointer switch or descriptor copy. Preserve at least the
       * previous 1024 application slots in the enlarged table. */
      struct hk_native_sampler_descriptor txf;
      uint32_t index = UINT32_MAX;
      result = hk_pack_native_txf_sampler(dev->dev.chip, &txf)
         ? hk_descriptor_table_add(dev, &h->table, &txf, sampler_size, &index)
         : VK_ERROR_FEATURE_NOT_PRESENT;
      if (result != VK_SUCCESS) {
         hk_descriptor_table_finish(dev, &h->table);
         ralloc_free(h->ht);
         return result;
      }
      assert(index == 0);
   }
   simple_mtx_init(&h->lock, mtx_plain);
   return VK_SUCCESS;
}

static void
hk_destroy_sampler_heap(struct hk_device *dev, struct hk_sampler_heap *h)
{
   hk_descriptor_table_finish(dev, &h->table);
   ralloc_free(h->ht);
   simple_mtx_destroy(&h->lock);
}

static VkResult
hk_sampler_heap_add_locked(struct hk_device *dev, struct hk_sampler_heap *h,
                           const struct hk_native_sampler_descriptor *desc,
                           struct hk_rc_sampler **out)
{
   /* Only the first eight bytes carry recovered sampler state.  The sparse
    * tail is required to be zero, so the packed word is a complete dedup key;
    * assert that invariant before an upload can silently alias future data. */
   const uint8_t *bytes = (const uint8_t *)desc;
   for (unsigned i = sizeof(struct agx_sampler_packed);
        i < h->table.desc_size; ++i) {
      assert(bytes[i] == 0);
      if (bytes[i] != 0)
         return VK_ERROR_FEATURE_NOT_PRESENT;
   }

   struct agx_sampler_packed key;
   memcpy(&key, desc, sizeof(key));
   struct hash_entry *ent = _mesa_hash_table_search(h->ht, &key);
   if (ent != NULL) {
      *out = ent->data;

      assert((*out)->refcount != 0);
      (*out)->refcount++;

      return VK_SUCCESS;
   }

   struct hk_rc_sampler *rc = ralloc(h->ht, struct hk_rc_sampler);
   if (!rc)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   uint32_t index;
   VkResult result =
      hk_descriptor_table_add(dev, &h->table, desc, h->table.desc_size, &index);
   if (result != VK_SUCCESS) {
      ralloc_free(rc);
      return result;
   }

   *rc = (struct hk_rc_sampler){
      .key = key,
      .refcount = 1,
      .index = index,
   };

   _mesa_hash_table_insert(h->ht, &rc->key, rc);
   *out = rc;

   return VK_SUCCESS;
}

VkResult
hk_sampler_heap_add(struct hk_device *dev,
                    const struct hk_native_sampler_descriptor *desc,
                    struct hk_rc_sampler **out)
{
   struct hk_sampler_heap *h = &dev->samplers;

   simple_mtx_lock(&h->lock);
   VkResult result = hk_sampler_heap_add_locked(dev, h, desc, out);
   simple_mtx_unlock(&h->lock);

   return result;
}

static void
hk_sampler_heap_remove_locked(struct hk_device *dev, struct hk_sampler_heap *h,
                              struct hk_rc_sampler *rc)
{
   assert(rc->refcount != 0);
   rc->refcount--;

   if (rc->refcount == 0) {
      hk_descriptor_table_remove(dev, &h->table, rc->index);
      _mesa_hash_table_remove_key(h->ht, &rc->key);
      ralloc_free(rc);
   }
}

void
hk_sampler_heap_remove(struct hk_device *dev, struct hk_rc_sampler *rc)
{
   struct hk_sampler_heap *h = &dev->samplers;

   simple_mtx_lock(&h->lock);
   hk_sampler_heap_remove_locked(dev, h, rc);
   simple_mtx_unlock(&h->lock);
}

static VkResult
hk_check_status(struct vk_device *device)
{
   struct hk_device *dev = container_of(device, struct hk_device, vk);
   if (dev->dev.params.features & DRM_ASAHI_FEATURE_VM_STATUS) {
      struct drm_asahi_vm_status status = {.vm_id = dev->dev.vm_id};
      struct drm_asahi_get_params query = {
         .param_group = DRM_ASAHI_PARAM_GROUP_VM_STATUS,
         .pointer = (uintptr_t)&status,
         .size = sizeof(status),
      };
      /* A DRM syncobj can signal an error fence successfully. The kernel
       * publishes this sticky VM error before signaling such a fence. */
      if (drmIoctl(dev->dev.fd, DRM_IOCTL_ASAHI_GET_PARAMS, &query))
         return vk_device_set_lost(&dev->vk, "VM status query failed: %m");
      if (status.error)
         return vk_device_set_lost(&dev->vk, "GPU VM %u failed: %s",
                                    status.vm_id, strerror(-status.error));
   }
   return vk_check_printf_status(&dev->vk, &dev->dev.printf);
}

/* The common wait entrypoints already invoke check_status. Their polling
 * counterparts only inspect the syncobj, so cover those two paths here too.
 * Older kernels perform no additional ioctl; known-lost devices short circuit.
 */
VKAPI_ATTR VkResult VKAPI_CALL
hk_GetFenceStatus(VkDevice device, VkFence fence)
{
   VkResult result = vk_common_GetFenceStatus(device, fence);
   if (result < 0)
      return result;

   VK_FROM_HANDLE(hk_device, dev, device);
   VkResult status = vk_device_check_status(&dev->vk);
   return status == VK_SUCCESS ? result : status;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetSemaphoreCounterValue(VkDevice device, VkSemaphore semaphore,
                            uint64_t *value)
{
   VkResult result = vk_common_GetSemaphoreCounterValue(device, semaphore, value);
   if (result < 0)
      return result;

   VK_FROM_HANDLE(hk_device, dev, device);
   VkResult status = vk_device_check_status(&dev->vk);
   return status == VK_SUCCESS ? result : status;
}

static VkResult
hk_get_timestamp(struct vk_device *device, uint64_t *timestamp)
{
   struct hk_device *dev = container_of(device, struct hk_device, vk);
   *timestamp = agx_get_gpu_timestamp(&dev->dev);
   return VK_SUCCESS;
}

/*
 * The native G17P indirect-compute grid helper.
 *
 * A native indirect dispatch is two CDM records: a direct-shaped main launch
 * with bit 23 set, whose inline dimension words the hardware ignores, and a
 * compact helper launch naming a program plus a six-u32 geometry block.  The
 * compact record carries no dimensions and no argument table, so nothing in
 * the command stream can turn a runtime VkDispatchIndirectCommand into a
 * launch geometry except a program that reads it and writes the block.
 *
 * The contract this implements is agx3_g17p_grid_helper_core.h's, which was
 * grounded byte-exactly against Apple's own compiler: geometry[0..2] are
 * GLOBAL THREAD counts (threadgroups * local size), geometry[3..5] are the
 * local size.  The kernel is written in NIR rather than reusing Apple's
 * compiled image because that image's uniform ABI is not established, while
 * Honeykrisp's own root ABI is the one every other G17P compute dispatch
 * already uses.
 *
 * Every invocation writes the same six values, so the helper needs no
 * invocation guard and therefore no system values: the compact record does not
 * say how many threads it launches, and an idempotent kernel is correct for
 * any answer.
 */
static void
hk_init_g17p_grid_helper(struct hk_device *dev,
                         const VkAllocationCallbacks *pAllocator)
{
   if (dev->dev.chip != AGX_CHIP_G17P)
      return;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P indirect grid helper");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *root = nir_load_root_agx(&b);
   nir_def *groups_addr = nir_load_global(
      &b, 1, 64,
      nir_iadd_imm(&b, root, hk_root_descriptor_offset(cs.group_count_addr)),
      .align_mul = 8);
   nir_def *geometry_addr = nir_load_global(
      &b, 1, 64,
      nir_iadd_imm(&b, root,
                   hk_root_descriptor_offset(cs.grid_helper_geometry_addr)),
      .align_mul = 8);
   nir_def *local = nir_load_global(
      &b, 3, 32,
      nir_iadd_imm(&b, root,
                   hk_root_descriptor_offset(cs.grid_helper_local_size)),
      .align_mul = 4);

   /* VkDispatchIndirectCommand is exactly three u32 threadgroup counts. */
   nir_def *groups = nir_load_global(&b, 3, 32, groups_addr, .align_mul = 4);

   nir_store_global(&b, nir_imul(&b, groups, local), geometry_addr,
                    .align_mul = 4);
   nir_store_global(&b, local,
                    nir_iadd_imm(&b, geometry_addr,
                                 AGX3_G17P_INDIRECT_GEOMETRY_LOCAL_OFFSET),
                    .align_mul = 4);

   struct vk_shader_compile_info info = {
      .nir = b.shader,
      .robustness = &vk_robustness_disabled,
      .stage = MESA_SHADER_COMPUTE,
   };
   hk_compile_shader(dev, &info, NULL, NULL, pAllocator,
                     &dev->g17p_grid_helper);

   /* A refusal is not fatal: it costs indirect dispatch, not the device. */
   if (!dev->g17p_grid_helper)
      mesa_logw("G17P indirect compute unavailable: grid helper refused");
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateDevice(VkPhysicalDevice physicalDevice,
                const VkDeviceCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator, VkDevice *pDevice)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   VkResult result = VK_ERROR_OUT_OF_HOST_MEMORY;
   struct hk_device *dev;
   struct hk_instance *instance = (struct hk_instance *)pdev->vk.instance;

   dev = vk_zalloc2(&instance->vk.alloc, pAllocator, sizeof(*dev), 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!dev)
      return vk_error(pdev, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_device_dispatch_table dispatch_table;

   /* For secondary command buffer support, overwrite any command entrypoints
    * in the main device-level dispatch table with
    * vk_cmd_enqueue_unless_primary_Cmd*.
    */
   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &vk_cmd_enqueue_unless_primary_device_entrypoints, true);

   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &hk_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &wsi_device_entrypoints, false);

   /* Populate primary cmd_dispatch table */
   vk_device_dispatch_table_from_entrypoints(&dev->cmd_dispatch,
                                             &hk_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(&dev->cmd_dispatch,
                                             &wsi_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(
      &dev->cmd_dispatch, &vk_common_device_entrypoints, false);

   result = vk_device_init(&dev->vk, &pdev->vk, &dispatch_table, pCreateInfo,
                           pAllocator);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   dev->vk.shader_ops = &hk_device_shader_ops;
   dev->vk.command_dispatch_table = &dev->cmd_dispatch;

   drmDevicePtr drm_device = NULL;
   int ret = drmGetDeviceFromDevId(pdev->render_dev, 0, &drm_device);
   if (ret != 0) {
      result = vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                         "Failed to get DRM device: %m");
      goto fail_init;
   }

   const char *path = drm_device->nodes[DRM_NODE_RENDER];
   dev->dev.fd = open(path, O_RDWR | O_CLOEXEC);
   if (dev->dev.fd < 0) {
      drmFreeDevice(&drm_device);
      result = vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                         "failed to open device %s", path);
      goto fail_init;
   }

   dev->perftest = debug_get_flags_option("HK_PERFTEST", hk_perf_options, 0);

   if (instance->no_border) {
      dev->perftest |= HK_PERF_NOBORDER;
   }

   if (HK_PERF(dev, NOROBUST)) {
      dev->vk.enabled_features.robustBufferAccess = false;
      dev->vk.enabled_features.robustBufferAccess2 = false;
      dev->vk.enabled_features.robustImageAccess = false;
      dev->vk.enabled_features.robustImageAccess2 = false;
      dev->vk.enabled_features.pipelineRobustness = false;
   }

   bool succ = agx_open_device(NULL, &dev->dev);
   drmFreeDevice(&drm_device);
   if (!succ) {
      result = vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                         "Failed to get DRM device: %m");
      goto fail_fd;
   }

   vk_device_set_drm_fd(&dev->vk, dev->dev.fd);
   dev->vk.copy_sync_payloads = vk_drm_syncobj_copy_payloads;
   dev->vk.command_buffer_ops = &hk_cmd_buffer_ops;
   dev->vk.check_status = hk_check_status;
   dev->vk.get_timestamp = hk_get_timestamp;

   /* This holds for current platforms. We do not currently implement
    * timestamp scaling, this would require changes in the query copy kernel
    * as well. Calibrated timestamps depends on this.
    */
   assert(dev->dev.user_timestamp_to_ns.num ==
             dev->dev.user_timestamp_to_ns.den &&
          "user timestamps are in ns");

   result = hk_init_sampler_heap(dev, &dev->samplers);
   if (result != VK_SUCCESS)
      goto fail_dev;

   if (dev->dev.chip == AGX_CHIP_G17P || dev->dev.chip == AGX_CHIP_G17G) {
      result = hk_descriptor_table_init(dev, &dev->sampled_images, 32,
                                        HK_MAX_DESCRIPTORS, HK_MAX_DESCRIPTORS);
      if (result != VK_SUCCESS)
         goto fail_samplers;

      struct hk_native_texture_descriptor null_texture;
      hk_set_null_native_texture(agx_chip_descriptor_generation(dev->dev.chip),
                                 &null_texture);
      uint32_t null_index;
      result = hk_descriptor_table_add(dev, &dev->sampled_images,
                                       &null_texture, 32, &null_index);
      if (result != VK_SUCCESS)
         goto fail_sampled_images;
      assert(null_index == 0);
   }

   result = hk_descriptor_table_init(
      dev, &dev->occlusion_queries, sizeof(uint64_t), AGX_MAX_OCCLUSION_QUERIES,
      AGX_MAX_OCCLUSION_QUERIES);
   if (result != VK_SUCCESS)
      goto fail_sampled_images;

   result = hk_upload_rodata(dev);
   if (result != VK_SUCCESS)
      goto fail_queries;

   /* XXX: error handling, and should this even go on the device? */
   agx_bg_eot_init(&dev->bg_eot, &dev->dev);
   if (!dev->bg_eot.ht) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail_rodata;
   }

   result = hk_init_internal_shaders(&dev->prolog_epilog);
   if (result != VK_SUCCESS)
      goto fail_bg_eot;

   result = hk_init_internal_shaders(&dev->kernels);
   if (result != VK_SUCCESS)
      goto fail_internal_shaders;

   for (unsigned i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
      for (unsigned q = 0; q < pCreateInfo->pQueueCreateInfos[i].queueCount;
           q++) {
         result = hk_queue_init(dev, &pCreateInfo->pQueueCreateInfos[i], q);
         if (result != VK_SUCCESS)
            goto fail_queues;
      }
   }

   struct vk_pipeline_cache_create_info cache_info = {
      .weak_ref = true,
   };
   dev->vk.mem_cache = vk_pipeline_cache_create(&dev->vk, &cache_info, NULL);
   if (dev->vk.mem_cache == NULL) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail_queues;
   }

   result = hk_device_init_meta(dev);
   if (result != VK_SUCCESS)
      goto fail_mem_cache;

   /* Precompile an empty fragment shader that can be used to handle API-level
    * null fragment shaders. We do this at device-time to make binds cheap.
    * Regardless, compiling this shader should be fast.
    */
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  &agx_nir_options, "empty FS");
   struct vk_shader_compile_info info = {
      .nir = b.shader,
      .robustness = &vk_robustness_disabled,
      .stage = MESA_SHADER_FRAGMENT,
   };
   hk_compile_shader(dev, &info, NULL, NULL, pAllocator, &dev->null_fs);
   if (!dev->null_fs) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail_meta;
   }

   hk_init_g17p_grid_helper(dev, pAllocator);

   *pDevice = hk_device_to_handle(dev);

   simple_mtx_init(&dev->vdm.lock, mtx_plain);
   list_inithead(&dev->vdm.slabs);
   simple_mtx_init(&dev->scratch.lock, mtx_plain);
   agx_scratch_init(&dev->dev, &dev->scratch.vs);
   agx_scratch_init(&dev->dev, &dev->scratch.fs);
   agx_scratch_init(&dev->dev, &dev->scratch.cs);

   u_rwlock_init(&dev->external_bos.lock);
   dev->external_bos.counts = UTIL_DYNARRAY_INIT;
   dev->external_bos.list = UTIL_DYNARRAY_INIT;

   return VK_SUCCESS;

fail_meta:
   hk_device_finish_meta(dev);
fail_mem_cache:
   vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);
fail_queues:
   vk_foreach_queue_safe(iter, &dev->vk) {
      struct hk_queue *queue = container_of(iter, struct hk_queue, vk);
      hk_queue_finish(dev, queue);
   }
   hk_destroy_internal_shaders(dev, &dev->kernels, false);
fail_internal_shaders:
   hk_destroy_internal_shaders(dev, &dev->prolog_epilog, true);
fail_bg_eot:
   agx_bg_eot_cleanup(&dev->bg_eot);
fail_rodata:
   agx_bo_unreference(&dev->dev, dev->rodata.bo);
fail_queries:
   hk_descriptor_table_finish(dev, &dev->occlusion_queries);
fail_sampled_images:
   if (dev->dev.chip == AGX_CHIP_G17P || dev->dev.chip == AGX_CHIP_G17G)
      hk_descriptor_table_finish(dev, &dev->sampled_images);
fail_samplers:
   hk_destroy_sampler_heap(dev, &dev->samplers);
fail_dev:
   agx_close_device(&dev->dev);
fail_fd:
   close(dev->dev.fd);
fail_init:
   vk_device_finish(&dev->vk);
fail_alloc:
   vk_free(&dev->vk.alloc, dev);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, _device);

   if (!dev)
      return;

   util_dynarray_fini(&dev->external_bos.counts);
   util_dynarray_fini(&dev->external_bos.list);
   u_rwlock_destroy(&dev->external_bos.lock);

   hk_device_finish_meta(dev);
   hk_destroy_internal_shaders(dev, &dev->kernels, false);
   hk_destroy_internal_shaders(dev, &dev->prolog_epilog, true);

   vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);

   vk_foreach_queue_safe(iter, &dev->vk) {
      struct hk_queue *queue = container_of(iter, struct hk_queue, vk);
      hk_queue_finish(dev, queue);
   }

   vk_device_finish(&dev->vk);

   hk_device_finish_vdm_slabs(dev);
   simple_mtx_destroy(&dev->vdm.lock);

   agx_scratch_fini(&dev->scratch.vs);
   agx_scratch_fini(&dev->scratch.fs);
   agx_scratch_fini(&dev->scratch.cs);
   simple_mtx_destroy(&dev->scratch.lock);

   if (dev->null_fs) {
      hk_api_shader_destroy(&dev->vk, &dev->null_fs->vk, pAllocator);
   }

   if (dev->g17p_grid_helper) {
      hk_api_shader_destroy(&dev->vk, &dev->g17p_grid_helper->vk, pAllocator);
   }

   hk_destroy_sampler_heap(dev, &dev->samplers);
   if (dev->dev.chip == AGX_CHIP_G17P || dev->dev.chip == AGX_CHIP_G17G)
      hk_descriptor_table_finish(dev, &dev->sampled_images);
   hk_descriptor_table_finish(dev, &dev->occlusion_queries);
   agx_bo_unreference(&dev->dev, dev->rodata.bo);
   agx_bo_unreference(&dev->dev, dev->heap);
   agx_bg_eot_cleanup(&dev->bg_eot);
   agx_close_device(&dev->dev);
   vk_free(&dev->vk.alloc, dev);
}
