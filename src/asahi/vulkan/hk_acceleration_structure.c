/*
 * Copyright © 2021 Google
 * Copyright © 2023 Valve Corporation
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * VK_KHR_acceleration_structure for Honeykrisp.
 *
 * The heavy lifting -- Morton codes, the radix sort, PLOC/HPLOC or LBVH
 * construction -- is Mesa's common builder, which produces a driver-neutral
 * binary tree in scratch memory.  All this file adds is the conversion of
 * that tree into the node format the software traversal reads (hk_bvh.h),
 * which happens in one compute dispatch per structure, plus the entry points
 * and the handful of vk_device hooks the common builder calls back into.
 */

#include "hk_acceleration_structure.h"

#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"

#include "radix_sort/radix_sort_u64.h"
#include "vk_acceleration_structure.h"
#include "vk_common_entrypoints.h"
#include "vk_meta.h"

#include "bvh/hk_build_interface.h"

#include "util/u_math.h"

static const uint32_t encode_spv[] = {
#include "bvh/hk_encode.spv.h"
};

static const uint32_t write_spv[] = {
#include "bvh/hk_bvh_write.spv.h"
};

static_assert(sizeof(struct hk_bvh_triangle_node) == 48, "node format");
static_assert(sizeof(struct hk_bvh_aabb_node) == 32, "node format");
static_assert(sizeof(struct hk_bvh_instance_node) == 120, "node format");
static_assert(sizeof(struct hk_bvh_box_node) == 56, "node format");
static_assert(sizeof(struct hk_bvh_box_node) == HK_BVH_NODE_PREFETCH_SIZE,
              "the traversal prefetches a whole box node before it knows the "
              "node type, so no node may be larger than one");

/*
 * The G17P subgroup is 32 wide and both minSubgroupSize and maxSubgroupSize
 * report 32, so every log2 below is fixed rather than tuned.
 */
static const struct radix_sort_vk_target_config hk_radix_sort_config = {
   .keyval_dwords = 2,
   .init = {.workgroup_size_log2 = 8},
   .fill = {.workgroup_size_log2 = 8, .block_rows = 8},
   .histogram =
      {
         .workgroup_size_log2 = 8,
         .subgroup_size_log2 = 5,
         .block_rows = 14,
      },
   .prefix =
      {
         .workgroup_size_log2 = 8,
         .subgroup_size_log2 = 5,
      },
   .scatter =
      {
         .workgroup_size_log2 = 8,
         .subgroup_size_log2 = 5,
         .block_rows = 14,
      },
   .nonsequential_dispatch = false,
};

static VkResult
get_pipeline_spv(struct hk_device *dev, const char *name, const uint32_t *spv,
                 uint32_t spv_size, unsigned push_constant_size,
                 VkPipeline *pipeline, VkPipelineLayout *layout)
{
   size_t key_size = strlen(name);

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = push_constant_size,
   };

   VkResult result = vk_meta_get_pipeline_layout(
      &dev->vk, &dev->meta, NULL, &pc_range, name, key_size, layout);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline cached = vk_meta_lookup_pipeline(&dev->meta, name, key_size);
   if (cached != VK_NULL_HANDLE) {
      *pipeline = cached;
      return VK_SUCCESS;
   }

   const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size,
      .pCode = spv,
   };
   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &module_info,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "main",
         },
      .layout = *layout,
   };

   return vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &pipeline_info,
                                          name, key_size, pipeline);
}

/* ---------------------------------------------------------------- */
/* Layout                                                            */

static void
hk_leaf_node_sizes(VkGeometryTypeKHR geometry_type, uint32_t *ir_size,
                   uint32_t *out_size)
{
   switch (geometry_type) {
   case VK_GEOMETRY_TYPE_TRIANGLES_KHR:
      *ir_size = sizeof(struct vk_ir_triangle_node);
      *out_size = sizeof(struct hk_bvh_triangle_node);
      return;
   case VK_GEOMETRY_TYPE_AABBS_KHR:
      *ir_size = sizeof(struct vk_ir_aabb_node);
      *out_size = sizeof(struct hk_bvh_aabb_node);
      return;
   default:
      *ir_size = sizeof(struct vk_ir_instance_node);
      *out_size = sizeof(struct hk_bvh_instance_node);
      return;
   }
}

static VkDeviceSize
hk_as_size(VkGeometryTypeKHR geometry_type, uint32_t leaf_node_count)
{
   /* The common builder always produces MAX2(leaves, 2) - 1 internal nodes,
    * so a one-primitive structure still has a root to point at. */
   uint32_t internal_node_count = MAX2(leaf_node_count, 2) - 1;

   uint32_t ir_size, out_size;
   hk_leaf_node_sizes(geometry_type, &ir_size, &out_size);

   uint32_t nodes_size = internal_node_count * sizeof(struct hk_bvh_box_node) +
                         leaf_node_count * out_size;

   /* Pad so the traversal's unconditional whole-node prefetch of the last
    * node cannot read past the allocation. */
   nodes_size = util_align_npot(nodes_size, HK_BVH_NODE_PREFETCH_SIZE);

   return sizeof(struct hk_bvh_header) + nodes_size;
}

static VkDeviceSize
hk_get_as_size(VkDevice _device,
               const struct vk_acceleration_structure_build_state *state)
{
   return hk_as_size(vk_get_as_geometry_type(state->build_info),
                     state->leaf_node_count);
}

/* ---------------------------------------------------------------- */
/* Encode                                                            */

static VkResult
hk_encode_prepare(VkCommandBuffer commandBuffer,
                  const struct vk_acceleration_structure_build_state *state)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result =
      get_pipeline_spv(dev, "hk-bvh-encode", encode_spv, sizeof(encode_spv),
                       sizeof(struct hk_encode_args), &pipeline, &layout);
   if (result != VK_SUCCESS)
      return result;

   vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline);
   return VK_SUCCESS;
}

static void
hk_encode_as(VkCommandBuffer commandBuffer,
             const struct vk_acceleration_structure_build_state *state)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, dst,
                  state->build_info->dstAccelerationStructure);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   VkGeometryTypeKHR geometry_type =
      vk_get_as_geometry_type(state->build_info);

   uint32_t ir_size, out_size;
   hk_leaf_node_sizes(geometry_type, &ir_size, &out_size);

   VkDeviceAddress scratch = state->build_info->scratchData.deviceAddress;
   uint32_t internal_bound = MAX2(state->leaf_node_count, 2) - 1;

   VkPipeline pipeline;
   VkPipelineLayout layout;
   if (get_pipeline_spv(dev, "hk-bvh-encode", encode_spv, sizeof(encode_spv),
                        sizeof(struct hk_encode_args), &pipeline,
                        &layout) != VK_SUCCESS)
      return;

   const struct hk_encode_args args = {
      .intermediate_bvh = scratch + state->scratch.ir_offset,
      .output_bvh = vk_acceleration_structure_get_va(dst),
      .header = scratch + state->scratch.header_offset,
      .root_offset = state->leaf_node_count * ir_size,
      .internal_node_bound = internal_bound,
      .leaf_node_bound = state->leaf_node_count,
      .geometry_type = geometry_type,
      .as_size = hk_as_size(geometry_type, state->leaf_node_count),
   };
   vk_common_CmdPushConstants(commandBuffer, layout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(args),
                              &args);

   /* One invocation per output node.  The shader clamps against the counts
    * the builder wrote into the header, which can be smaller than this bound
    * when leaves turn out inactive. */
   hk_cmd_dispatch_unaligned(commandBuffer,
                             internal_bound + state->leaf_node_count, 1, 1);
}

static const struct vk_acceleration_structure_build_ops hk_as_build_ops = {
   .get_as_size = hk_get_as_size,
   .encode_prepare[0] = hk_encode_prepare,
   .encode_as[0] = hk_encode_as,
};

/* ---------------------------------------------------------------- */
/* vk_device hooks the common builder calls                          */

static void
hk_write_buffer_cp(VkCommandBuffer commandBuffer, VkDeviceAddress addr,
                   void *data, uint32_t size)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   struct {
      uint64_t dst;
      uint32_t count;
      uint32_t data[28];
   } args = {
      .dst = addr,
      .count = DIV_ROUND_UP(size, 4),
   };

   /* The only caller writes a vk_ir_header, which is 104 bytes.  Anything
    * larger is a change in the common builder, not a bigger buffer: refuse
    * rather than write a truncated header that would fail obscurely later. */
   assert(size <= sizeof(args.data));
   if (size > sizeof(args.data))
      return;
   memcpy(args.data, data, size);

   VkPipeline pipeline;
   VkPipelineLayout layout;
   if (get_pipeline_spv(dev, "hk-bvh-write", write_spv, sizeof(write_spv),
                        sizeof(args), &pipeline, &layout) != VK_SUCCESS)
      return;

   vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline);
   vk_common_CmdPushConstants(commandBuffer, layout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(args),
                              &args);
   hk_cmd_dispatch_unaligned(commandBuffer, args.count, 1, 1);
}

static void
hk_flush_buffer_write_cp(VkCommandBuffer commandBuffer)
{
   /* hk_write_buffer_cp is an ordinary dispatch, so the barrier the common
    * builder already emits between phases covers it. */
}

static void
hk_cmd_fill_buffer_addr(VkCommandBuffer commandBuffer, VkDeviceAddress addr,
                        VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   vk_meta_fill_buffer_addr(&cmd->vk, &dev->meta, addr, size, data);
}

/* ---------------------------------------------------------------- */
/* Device state                                                      */

VkResult
hk_device_init_accel_struct_state(struct hk_device *dev)
{
   dev->accel_struct_args = (struct vk_acceleration_structure_build_args){
      .subgroup_size = 32,
      .bvh_bounds_offset = offsetof(struct hk_bvh_header, bounds),
      /* The common builder's spec_map gives BUILD_FLAGS_ID and
       * ROOT_FLAGS_OFFSET_ID the same push-constant offset, so root_flags
       * would receive the build flags.  It is only read when cull-flag
       * propagation is on, so leave that off until that is fixed upstream. */
      .root_flags_offset = 0,
      .propagate_cull_flags = false,
      .emit_markers = false,
   };

   dev->vk.as_build_ops = &hk_as_build_ops;
   dev->vk.write_buffer_cp = hk_write_buffer_cp;
   dev->vk.flush_buffer_write_cp = hk_flush_buffer_write_cp;
   dev->vk.cmd_dispatch_unaligned = hk_cmd_dispatch_unaligned;
   dev->vk.cmd_fill_buffer_addr = hk_cmd_fill_buffer_addr;

   simple_mtx_init(&dev->radix_sort_lock, mtx_plain);
   return VK_SUCCESS;
}

void
hk_device_finish_accel_struct_state(struct hk_device *dev)
{
   if (dev->radix_sort) {
      radix_sort_vk_destroy(dev->radix_sort, hk_device_to_handle(dev),
                            &dev->vk.alloc);
      dev->radix_sort = NULL;
   }
   simple_mtx_destroy(&dev->radix_sort_lock);
}

/*
 * Created on first use rather than at device creation: it compiles a dozen
 * compute pipelines, and the overwhelming majority of devices never build an
 * acceleration structure.
 */
static bool
hk_init_radix_sort(struct hk_device *dev)
{
   simple_mtx_lock(&dev->radix_sort_lock);
   if (!dev->radix_sort && !dev->radix_sort_failed) {
      dev->radix_sort = vk_create_radix_sort_u64(
         hk_device_to_handle(dev), &dev->vk.alloc, VK_NULL_HANDLE,
         hk_radix_sort_config);
      dev->accel_struct_args.radix_sort = dev->radix_sort;

      if (!dev->radix_sort) {
         /* The sort is a dozen ordinary compute pipelines, so the realistic
          * failure is the AGX3 packer refusing one of their instructions --
          * which it does today, on the unconditional break nested deeper than
          * two that the scan kernel emits.  The common builder dereferences
          * this pointer without checking, so every entry point below has to
          * bail out rather than hand it a NULL.  Say so once: the alternative
          * is an application that segfaults inside Mesa with no clue why. */
         mesa_loge("hk: acceleration structure builds are unavailable -- the "
                   "radix sort pipelines did not compile for this chip");
         dev->radix_sort_failed = true;
      }
   }
   bool ok = dev->radix_sort != NULL;
   simple_mtx_unlock(&dev->radix_sort_lock);
   return ok;
}

/* ---------------------------------------------------------------- */
/* Entry points                                                      */

VKAPI_ATTR void VKAPI_CALL
hk_GetAccelerationStructureBuildSizesKHR(
   VkDevice _device, VkAccelerationStructureBuildTypeKHR buildType,
   const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
   const uint32_t *pMaxPrimitiveCounts,
   VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
   VK_FROM_HANDLE(hk_device, dev, _device);

   /*
    * Do not "fix" this by sizing without the sort.
    *
    * It looks fixable: radix_sort_vk_get_memory_requirements() reads only
    * rs->config -- keyval dwords and the workgroup/subgroup log2s -- so the
    * scratch size needs no compiled pipeline, and only vk_create_radix_sort_
    * u64() returning NULL wholesale on a pipeline failure takes sizing down
    * with the build.  Keeping a config-only object alive would make sizing
    * succeed.
    *
    * It would not help.  hk_CmdBuildAccelerationStructuresKHR still records
    * VK_ERROR_FEATURE_NOT_PRESENT, so the application trades a specific
    * failure -- a zero size, which says the build path is not functional --
    * for a generic one at vkEndCommandBuffer.  Reaching a *working* build
    * needs the pipelines, and faking one is worse than either: an untouched
    * structure leaves a TLAS instance node holding a garbage bvh_ptr that the
    * traversal dereferences, and a hand-written empty BVH hides the real
    * blocker behind "every ray misses".
    *
    * So the zero is deliberate and it is the honest answer.  Delete it when
    * the pipelines compile, not before.
    */
   if (!hk_init_radix_sort(dev)) {
      *pSizeInfo = (VkAccelerationStructureBuildSizesInfoKHR){
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
         .pNext = pSizeInfo->pNext,
      };
      return;
   }

   vk_get_as_build_sizes(_device, buildType, pBuildInfo, pMaxPrimitiveCounts,
                         pSizeInfo, &dev->accel_struct_args);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (!hk_init_radix_sort(dev)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   /* A build binds its own compute pipelines and push constants, so the
    * application's compute state has to survive it. */
   struct hk_meta_save save;
   hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);

   vk_cmd_build_acceleration_structures(commandBuffer, &dev->vk, &dev->meta,
                                        infoCount, pInfos, ppBuildRangeInfos,
                                        &dev->accel_struct_args);

   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDeviceAccelerationStructureCompatibilityKHR(
   VkDevice _device, const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
   VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   bool compat = memcmp(pVersionInfo->pVersionData,
                        pdev->vk.properties.driverUUID, VK_UUID_SIZE) == 0;
   *pCompatibility = compat
                        ? VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR
                        : VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

/*
 * Host-side builds and the copy/serialize family are unimplemented and, in
 * the case of the host builds, unadvertised: accelerationStructureHostCommands
 * stays false, so a conformant application never reaches them.  They fail
 * loudly rather than silently returning success on an untouched structure.
 */
VKAPI_ATTR VkResult VKAPI_CALL
hk_BuildAccelerationStructuresKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyAccelerationStructureKHR(VkDevice _device,
                                VkDeferredOperationKHR deferredOperation,
                                const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyMemoryToAccelerationStructureKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyAccelerationStructureToMemoryKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_WriteAccelerationStructuresPropertiesKHR(
   VkDevice _device, uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType, size_t dataSize, void *pData, size_t stride)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}
