/* SPDX-License-Identifier: MIT */
/*
 * Host-side checks on the pure parts of VK_KHR_ray_tracing_pipeline: the
 * shader-group table, the group handle layout, the shader binding table index
 * arithmetic, the launch decomposition, and the stack-size bound.
 *
 * Every check here is a *contract* the megakernel builder and the trace-rays
 * entry points both depend on, so a change to either that breaks the contract
 * fails at `meson test --suite asahi` instead of on hardware.
 */

#include "../hk_ray_tracing_core.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void
test_group_table(void)
{
   struct hk_ray_tracing_group g;

   /* A general group names exactly one shader, and it is the recursive one. */
   assert(hk_rt_group_from_create_info(
      &g, VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 7,
      VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR));
   assert(g.recursive_index == 7);
   assert(g.ahit_index == VK_SHADER_UNUSED_KHR);
   assert(g.isec_index == VK_SHADER_UNUSED_KHR);

   /* A triangle hit group has a closest hit and optionally an any hit, and
    * never an intersection shader even if one is supplied. */
   assert(hk_rt_group_from_create_info(
      &g, VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
      VK_SHADER_UNUSED_KHR, 3, 4, 5));
   assert(g.recursive_index == 3);
   assert(g.ahit_index == 4);
   assert(g.isec_index == VK_SHADER_UNUSED_KHR);

   /* A procedural hit group keeps its intersection shader, and its any hit is
    * only meaningful alongside one. */
   assert(hk_rt_group_from_create_info(
      &g, VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
      VK_SHADER_UNUSED_KHR, 3, 4, 5));
   assert(g.recursive_index == 3);
   assert(g.ahit_index == 4);
   assert(g.isec_index == 5);

   assert(hk_rt_group_from_create_info(
      &g, VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
      VK_SHADER_UNUSED_KHR, 3, 4, VK_SHADER_UNUSED_KHR));
   assert(g.recursive_index == 3);
   assert(g.ahit_index == VK_SHADER_UNUSED_KHR);
   assert(g.isec_index == VK_SHADER_UNUSED_KHR);

   /* An unknown group type is refused rather than silently dropped. */
   struct hk_ray_tracing_group untouched = {.recursive_index = 0xdeadbeef};
   g = untouched;
   assert(!hk_rt_group_from_create_info(&g, (VkRayTracingShaderGroupTypeKHR)99,
                                        0, 0, 0, 0));
   assert(g.recursive_index == untouched.recursive_index);
}

static void
test_group_rebase(void)
{
   const struct hk_ray_tracing_group src = {
      .handle = {.index = 11},
      .recursive_index = 0,
      .ahit_index = VK_SHADER_UNUSED_KHR,
      .isec_index = 2,
   };

   const struct hk_ray_tracing_group dst = hk_rt_group_rebase(src, 5);

   assert(dst.handle.index == 11); /* a handle is device-unique, it never moves */
   assert(dst.recursive_index == 5);
   assert(dst.isec_index == 7);

   /* VK_SHADER_UNUSED_KHR is not an index and must not be offset -- rebasing it
    * would turn "no any-hit shader" into a reference to a real stage. */
   assert(dst.ahit_index == VK_SHADER_UNUSED_KHR);
}

static void
test_group_handle(void)
{
   uint8_t buffer[HK_RAY_TRACING_GROUP_HANDLE_SIZE + 8];
   memset(buffer, 0xAA, sizeof(buffer));

   hk_rt_write_group_handle(buffer,
                            (struct hk_ray_tracing_group_handle){.index =
                                                                   0x12345678});

   /* Little-endian index, then zeroes to the full handle size: an application
    * may memcmp two whole handles, so the padding must be deterministic. */
   assert(buffer[0] == 0x78 && buffer[1] == 0x56 && buffer[2] == 0x34 &&
          buffer[3] == 0x12);
   for (unsigned i = 4; i < HK_RAY_TRACING_GROUP_HANDLE_SIZE; ++i)
      assert(buffer[i] == 0);

   /* Nothing past the handle is touched. */
   for (unsigned i = HK_RAY_TRACING_GROUP_HANDLE_SIZE; i < sizeof(buffer); ++i)
      assert(buffer[i] == 0xAA);

   /* The specification fixes the handle size at 32 bytes. */
   assert(HK_RAY_TRACING_GROUP_HANDLE_SIZE == 32);
}

static void
test_record_layout(void)
{
   /* The megakernel reads the record with these offsets baked in, and
    * vkCmdTraceRaysIndirect2KHR hands it the application's own buffer, so the
    * layout is the extension's, not ours. */
   assert(HK_RT_CMD_RAYGEN_ADDRESS == 0);
   assert(HK_RT_CMD_MISS_ADDRESS == 16);
   assert(HK_RT_CMD_HIT_ADDRESS == 40);
   assert(HK_RT_CMD_CALLABLE_ADDRESS == 64);
   assert(HK_RT_CMD_WIDTH == 88);

   /* Each region's stride is two VkDeviceSize past its address. */
   assert(hk_rt_sbt_stride_offset(HK_RT_CMD_MISS_ADDRESS) ==
          offsetof(VkTraceRaysIndirectCommand2KHR,
                   missShaderBindingTableStride));
   assert(hk_rt_sbt_stride_offset(HK_RT_CMD_HIT_ADDRESS) ==
          offsetof(VkTraceRaysIndirectCommand2KHR,
                   hitShaderBindingTableStride));
   assert(hk_rt_sbt_stride_offset(HK_RT_CMD_CALLABLE_ADDRESS) ==
          offsetof(VkTraceRaysIndirectCommand2KHR,
                   callableShaderBindingTableStride));

   /* The three launch dimensions are contiguous: the indirect setup kernel
    * writes them as three consecutive uint32_t. */
   assert(offsetof(VkTraceRaysIndirectCommand2KHR, height) ==
          HK_RT_CMD_WIDTH + 4);
   assert(offsetof(VkTraceRaysIndirectCommand2KHR, depth) ==
          HK_RT_CMD_WIDTH + 8);
}

static void
test_hit_sbt_index(void)
{
   /* The specification's formula, with the instance contribution included. */
   assert(hk_rt_hit_sbt_index(0, 0, 0, 0) == 0);
   assert(hk_rt_hit_sbt_index(2, 3, 0, 4) == 2 + 12);
   assert(hk_rt_hit_sbt_index(2, 3, 100, 4) == 2 + 100 + 12);

   /* A zero stride collapses every geometry of an instance onto one record,
    * which is what a single-hit-group scene does. */
   assert(hk_rt_hit_sbt_index(1, 0, 7, 12345) == 8);

   /* The shader computes this in 32 bits and so does this helper. */
   assert(hk_rt_hit_sbt_index(0xFFFFFFFFu, 1, 1, 1) == 1);
}

static void
test_launch_decomposition(void)
{
   uint32_t g[3];

   hk_rt_launch_group_count(1, 1, 1, g);
   assert(g[0] == 1 && g[1] == 1 && g[2] == 1);

   hk_rt_launch_group_count(HK_RT_WORKGROUP_SIZE_X, 4, 2, g);
   assert(g[0] == 1 && g[1] == 4 && g[2] == 2);

   hk_rt_launch_group_count(HK_RT_WORKGROUP_SIZE_X + 1, 1, 1, g);
   assert(g[0] == 2);

   /* Must round up, never down: a launch that loses its trailing rays is a
    * silently wrong image. */
   for (uint32_t w = 1; w <= 4 * HK_RT_WORKGROUP_SIZE_X; ++w) {
      hk_rt_launch_group_count(w, 1, 1, g);
      assert(g[0] * HK_RT_WORKGROUP_SIZE_X >= w);
      assert((g[0] - 1) * HK_RT_WORKGROUP_SIZE_X < w);
   }

   /* And must not overflow on the largest representable width. */
   hk_rt_launch_group_count(0xFFFFFFFFu, 1, 1, g);
   assert(g[0] == (0xFFFFFFFFu / HK_RT_WORKGROUP_SIZE_X) + 1);

   /* The indirect path computes the same number on the GPU, with a shift
    * instead of a divide (libagx_trace_rays_indirect_setup in
    * src/asahi/libagx/draws.cl).  The two must agree everywhere, or a direct
    * and an indirect trace of the same size launch different grids.  This is
    * the host oracle for that kernel's arithmetic; it says nothing about
    * whether its machine code executes. */
   assert((1u << HK_RT_WORKGROUP_SIZE_X_LOG2) == HK_RT_WORKGROUP_SIZE_X);
   const uint32_t shift_mask = (1u << HK_RT_WORKGROUP_SIZE_X_LOG2) - 1u;
#define KERNEL_FORM(w)                                                         \
   (((w) >> HK_RT_WORKGROUP_SIZE_X_LOG2) + (((w) & shift_mask) ? 1u : 0u))

   for (uint32_t w = 0; w <= 8 * HK_RT_WORKGROUP_SIZE_X; ++w) {
      hk_rt_launch_group_count(w, 3, 2, g);
      assert(g[0] == KERNEL_FORM(w));
      assert(g[1] == 3 && g[2] == 2);
   }

   /* The top of the range is the half that matters: the obvious
    * (width + mask) >> log2 form WRAPS here and launches zero workgroups. */
   for (uint32_t w = 0xFFFFFFFFu; w > 0xFFFFFF00u; --w) {
      hk_rt_launch_group_count(w, 1, 1, g);
      assert(g[0] == KERNEL_FORM(w));
      assert(g[0] != 0);
   }
#undef KERNEL_FORM
}

static void
test_stack_size(void)
{
   const struct hk_rt_stack_sizes sizes = {
      .raygen = 16,
      .any_hit = 8,
      .closest_hit = 32,
      .miss = 24,
      .intersection = 40,
      .callable = 4,
   };

   /* Depth 0: a raygen shader that never traces still needs its own frame. */
   assert(hk_rt_pipeline_stack_size(&sizes, 0) == 16 + 31 * 4);

   /* Depth 1: raygen, then the worst first-level frame.  An intersection
    * shader may have an any-hit stacked on it, so isec + ahit = 48 beats the
    * 32-byte closest hit. */
   assert(hk_rt_pipeline_stack_size(&sizes, 1) == 16 + 48 + 31 * 4);

   /* Deeper levels cannot be intersection shaders: an intersection shader
    * cannot trace, so every level past the first is chit or miss. */
   assert(hk_rt_pipeline_stack_size(&sizes, 3) == 16 + 48 + 2 * 32 + 31 * 4);

   /* The bound is monotone in the recursion depth. */
   uint32_t prev = 0;
   for (uint32_t d = 0; d <= 31; ++d) {
      const uint32_t s = hk_rt_pipeline_stack_size(&sizes, d);
      assert(s >= prev);
      prev = s;
   }

   /* A pipeline with no callables reserves nothing for them. */
   struct hk_rt_stack_sizes no_callable = sizes;
   no_callable.callable = 0;
   assert(hk_rt_pipeline_stack_size(&no_callable, 1) == 16 + 48);
}

static void
test_group_stack_size(void)
{
   const struct hk_rt_stack_sizes sizes = {
      .raygen = 16,
      .any_hit = 8,
      .closest_hit = 32,
      .miss = 24,
      .intersection = 40,
      .callable = 4,
   };

   const struct hk_ray_tracing_group full = {
      .recursive_index = 1,
      .ahit_index = 2,
      .isec_index = 3,
   };
   const struct hk_ray_tracing_group bare = {
      .recursive_index = VK_SHADER_UNUSED_KHR,
      .ahit_index = VK_SHADER_UNUSED_KHR,
      .isec_index = VK_SHADER_UNUSED_KHR,
   };

   assert(hk_rt_group_stack_size(&sizes, &full,
                                 VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR,
                                 sizes.closest_hit) == 32);
   assert(hk_rt_group_stack_size(&sizes, &full,
                                 VK_SHADER_GROUP_SHADER_ANY_HIT_KHR, 0) == 8);
   assert(hk_rt_group_stack_size(&sizes, &full,
                                 VK_SHADER_GROUP_SHADER_INTERSECTION_KHR,
                                 0) == 40);
   assert(hk_rt_group_stack_size(&sizes, &full,
                                 VK_SHADER_GROUP_SHADER_GENERAL_KHR,
                                 sizes.raygen) == 16);

   /* A shader the group does not have reports zero, which is what
    * vkGetRayTracingShaderGroupStackSizeKHR requires. */
   assert(hk_rt_group_stack_size(&sizes, &bare,
                                 VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR,
                                 sizes.closest_hit) == 0);
   assert(hk_rt_group_stack_size(&sizes, &bare,
                                 VK_SHADER_GROUP_SHADER_ANY_HIT_KHR, 0) == 0);
   assert(hk_rt_group_stack_size(&sizes, &bare,
                                 VK_SHADER_GROUP_SHADER_INTERSECTION_KHR,
                                 0) == 0);
}

int
main(void)
{
   test_group_table();
   test_group_rebase();
   test_group_handle();
   test_record_layout();
   test_hit_sbt_index();
   test_launch_decomposition();
   test_stack_size();
   test_group_stack_size();

   puts("hk ray tracing pipeline core PASS");
   return 0;
}
