/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * vkCmdTraceRaysKHR and its indirect forms.
 *
 * Every form publishes the same record, VkTraceRaysIndirectCommand2KHR, and
 * dispatches the pipeline's megakernel with the record's address in the compute
 * root descriptor table.  The direct form fills the record on the host; the
 * indirect forms let one GPU invocation fill in the launch size, because that
 * is the only part that is not known until the command executes.
 */

#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_ray_tracing_pipeline.h"
#include "hk_shader.h"

#include "asahi/lib/agx_bo.h"
#include "libagx_shaders.h"

static VkTraceRaysIndirectCommand2KHR
hk_trace_rays_record(const VkStridedDeviceAddressRegionKHR *raygen,
                     const VkStridedDeviceAddressRegionKHR *miss,
                     const VkStridedDeviceAddressRegionKHR *hit,
                     const VkStridedDeviceAddressRegionKHR *callable,
                     uint32_t width, uint32_t height, uint32_t depth)
{
   return (VkTraceRaysIndirectCommand2KHR){
      .raygenShaderRecordAddress = raygen->deviceAddress,
      .raygenShaderRecordSize = raygen->size,

      .missShaderBindingTableAddress = miss->deviceAddress,
      .missShaderBindingTableSize = miss->size,
      .missShaderBindingTableStride = miss->stride,

      .hitShaderBindingTableAddress = hit->deviceAddress,
      .hitShaderBindingTableSize = hit->size,
      .hitShaderBindingTableStride = hit->stride,

      .callableShaderBindingTableAddress = callable->deviceAddress,
      .callableShaderBindingTableSize = callable->size,
      .callableShaderBindingTableStride = callable->stride,

      .width = width,
      .height = height,
      .depth = depth,
   };
}

static struct hk_api_shader *
hk_bound_rt_shader(struct hk_cmd_buffer *cmd)
{
   struct hk_ray_tracing_pipeline *pipeline = cmd->state.rt.pipeline;

   if (pipeline == NULL || pipeline->shader == NULL) {
      /* A pipeline library is never executable and a failed create returns
       * VK_NULL_HANDLE.  Recording a trace against either is application
       * error; report it rather than dispatching nothing silently. */
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
      return NULL;
   }

   return pipeline->shader;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdTraceRaysKHR(VkCommandBuffer commandBuffer,
                   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
                   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
                   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
                   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
                   uint32_t width, uint32_t height, uint32_t depth)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   struct hk_api_shader *shader = hk_bound_rt_shader(cmd);
   if (!shader)
      return;

   if (width == 0 || height == 0 || depth == 0)
      return;

   const VkTraceRaysIndirectCommand2KHR record = hk_trace_rays_record(
      pRaygenShaderBindingTable, pMissShaderBindingTable,
      pHitShaderBindingTable, pCallableShaderBindingTable, width, height,
      depth);

   const uint64_t record_addr =
      hk_pool_upload(cmd, &record, sizeof(record), 8);
   if (!record_addr)
      return;

   uint32_t groups[3];
   hk_rt_launch_group_count(width, height, depth, groups);

   const VkDispatchIndirectCommand group_count = {
      .x = groups[0],
      .y = groups[1],
      .z = groups[2],
   };
   const uint64_t group_count_addr =
      hk_pool_upload(cmd, &group_count, sizeof(group_count), 8);
   if (!group_count_addr)
      return;

   /* hk_cmd_dispatch_rays() takes the same grid every API dispatch takes:
    * threadgroups, which the compute path multiplies by the workgroup size. */
   hk_cmd_dispatch_rays(cmd, shader, record_addr, group_count_addr,
                        agx_3d(groups[0], groups[1], groups[2]));
}

/*
 * The shared tail of both indirect forms.  `record_addr` is the record the
 * megakernel will read; `dims_addr` is where the three launch dimensions live
 * right now; `record_dims_addr` is where they must be copied to, or zero when
 * the record already contains them.
 */
static void
hk_trace_rays_indirect(struct hk_cmd_buffer *cmd, struct hk_api_shader *shader,
                       uint64_t record_addr, uint64_t record_dims_addr,
                       uint64_t dims_addr)
{
   const uint64_t group_count_addr =
      hk_pool_alloc(cmd, sizeof(VkDispatchIndirectCommand), 8).gpu;
   if (!group_count_addr)
      return;

   libagx_trace_rays_indirect_setup(cmd, agx_1d(1), AGX_BARRIER_ALL,
                                    record_dims_addr, group_count_addr,
                                    dims_addr, HK_RT_WORKGROUP_SIZE_X_LOG2);

   hk_cmd_dispatch_rays(cmd, shader, record_addr, group_count_addr,
                        agx_grid_indirect(group_count_addr));
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdTraceRaysIndirectKHR(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   VkDeviceAddress indirectDeviceAddress)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   struct hk_api_shader *shader = hk_bound_rt_shader(cmd);
   if (!shader)
      return;

   /* The application's buffer holds only a VkTraceRaysIndirectCommandKHR, so
    * the shader binding tables are assembled here and the three dimensions are
    * copied in by the setup kernel. */
   const VkTraceRaysIndirectCommand2KHR record = hk_trace_rays_record(
      pRaygenShaderBindingTable, pMissShaderBindingTable,
      pHitShaderBindingTable, pCallableShaderBindingTable, 0, 0, 0);

   const uint64_t record_addr =
      hk_pool_upload(cmd, &record, sizeof(record), 8);
   if (!record_addr)
      return;

   hk_trace_rays_indirect(cmd, shader, record_addr,
                          record_addr + HK_RT_CMD_WIDTH, indirectDeviceAddress);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdTraceRaysIndirect2KHR(VkCommandBuffer commandBuffer,
                            VkDeviceAddress indirectDeviceAddress)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   struct hk_api_shader *shader = hk_bound_rt_shader(cmd);
   if (!shader)
      return;

   /* The application supplied the whole record, so the megakernel can read it
    * where it is; only the threadgroup counts still have to be derived. */
   hk_trace_rays_indirect(cmd, shader, indirectDeviceAddress, 0,
                          indirectDeviceAddress + HK_RT_CMD_WIDTH);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdSetRayTracingPipelineStackSizeKHR(VkCommandBuffer commandBuffer,
                                        uint32_t pipelineStackSize)
{
   /* The megakernel's stack is static scratch inside its compiled binary, sized
    * at pipeline creation for maxPipelineRayRecursionDepth.  A dynamic size
    * cannot shrink it and must not grow it, so this is deliberately a no-op
    * rather than a partial implementation.  Documented in
    * docs/G17P_RAYTRACING_PIPELINE_ABI.md. */
}
