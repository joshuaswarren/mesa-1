/*
 * Does Apple's own G17P compiler use the quad subgroup forms outside the
 * fragment and compute stages?  Three entry points, each paired with a
 * SIMD-group control built from an identical body, so the only thing that can
 * move between the two native objects is the quad/SIMD selector itself.
 *
 *   vertex_quad / vertex_simd            the hardware vertex program
 *   tess_eval_quad / tess_eval_simd      the post-tessellation vertex program,
 *                                        which is Metal's form of a Vulkan
 *                                        tessellation evaluation shader
 *   kernel_quad / kernel_simd            the CDM kernel, which is what
 *                                        hk_launch_tess() dispatches a
 *                                        tessellation control shader as
 */
#include <metal_stdlib>
using namespace metal;

struct CP {
   float4 position [[attribute(0)]];
};

struct VOut {
   float4 position [[position]];
   float4 value [[user(locn0)]];
};

vertex VOut vertex_quad(uint vid [[vertex_id]],
                        constant float4 *src [[buffer(0)]])
{
   float4 a = src[vid];
   float4 b = quad_broadcast(a, 3u);
   float4 c = quad_shuffle_xor(a, 1u);
   float4 d = quad_shuffle_xor(a, 2u);
   float4 e = quad_shuffle_xor(a, 3u);
   VOut out;
   out.position = float4(float(vid), 0.0, 0.0, 1.0);
   out.value = b + c + d + e;
   return out;
}

vertex VOut vertex_simd(uint vid [[vertex_id]],
                        constant float4 *src [[buffer(0)]])
{
   float4 a = src[vid];
   float4 b = simd_broadcast(a, 3u);
   float4 c = simd_shuffle_xor(a, 1u);
   float4 d = simd_shuffle_xor(a, 2u);
   float4 e = simd_shuffle_xor(a, 3u);
   VOut out;
   out.position = float4(float(vid), 0.0, 0.0, 1.0);
   out.value = b + c + d + e;
   return out;
}

[[patch(quad, 4)]]
vertex VOut tess_eval_quad(patch_control_point<CP> cps [[stage_in]],
                           float2 uv [[position_in_patch]],
                           uint pid [[patch_id]])
{
   float4 a = cps[0].position * uv.x + cps[1].position * uv.y;
   float4 b = quad_broadcast(a, 3u);
   float4 c = quad_shuffle_xor(a, 1u);
   float4 d = quad_shuffle_xor(a, 2u);
   float4 e = quad_shuffle_xor(a, 3u);
   VOut out;
   out.position = float4(float(pid), 0.0, 0.0, 1.0);
   out.value = b + c + d + e;
   return out;
}

[[patch(quad, 4)]]
vertex VOut tess_eval_simd(patch_control_point<CP> cps [[stage_in]],
                           float2 uv [[position_in_patch]],
                           uint pid [[patch_id]])
{
   float4 a = cps[0].position * uv.x + cps[1].position * uv.y;
   float4 b = simd_broadcast(a, 3u);
   float4 c = simd_shuffle_xor(a, 1u);
   float4 d = simd_shuffle_xor(a, 2u);
   float4 e = simd_shuffle_xor(a, 3u);
   VOut out;
   out.position = float4(float(pid), 0.0, 0.0, 1.0);
   out.value = b + c + d + e;
   return out;
}

fragment float4 frag_pass(VOut in [[stage_in]])
{
   return in.value;
}

kernel void kernel_quad(device float4 *dst [[buffer(0)]],
                        constant float4 *src [[buffer(1)]],
                        uint tid [[thread_position_in_grid]])
{
   float4 a = src[tid];
   float4 b = quad_broadcast(a, 3u);
   float4 c = quad_shuffle_xor(a, 1u);
   float4 d = quad_shuffle_xor(a, 2u);
   float4 e = quad_shuffle_xor(a, 3u);
   dst[tid] = b + c + d + e;
}

kernel void kernel_simd(device float4 *dst [[buffer(0)]],
                        constant float4 *src [[buffer(1)]],
                        uint tid [[thread_position_in_grid]])
{
   float4 a = src[tid];
   float4 b = simd_broadcast(a, 3u);
   float4 c = simd_shuffle_xor(a, 1u);
   float4 d = simd_shuffle_xor(a, 2u);
   float4 e = simd_shuffle_xor(a, 3u);
   dst[tid] = b + c + d + e;
}

/*
 * The 16-bit pairs.  Vulkan's subgroupQuadBroadcast/subgroupQuadSwap on f16,
 * i16 and bool data reach the packer as 16-bit quad shuffles; the SIMD twin of
 * each isolates the quad selector from the narrowed width pair.
 */
#define HALF_PAIR(name, expr)                                                  \
   kernel void kernel_##name(device ushort *dst [[buffer(0)]],                 \
                             constant ushort *src [[buffer(1)]],               \
                             uint tid [[thread_position_in_grid]])             \
   {                                                                           \
      ushort a = src[tid];                                                     \
      dst[tid] = expr;                                                         \
   }

HALF_PAIR(quad_half_bcast, quad_broadcast(a, 3u))
HALF_PAIR(simd_half_bcast, simd_broadcast(a, 3u))
HALF_PAIR(quad_half_xor,   quad_shuffle_xor(a, 1u))
HALF_PAIR(simd_half_xor,   simd_shuffle_xor(a, 1u))
HALF_PAIR(simd_half_up,    simd_shuffle_up(a, 1u))
HALF_PAIR(simd_half_down,  simd_shuffle_down(a, 1u))

/*
 * The register-lane attempt.  Apple folds a `constant` buffer lane into the
 * immediate-lane form, so this does NOT witness the register-lane quad word
 * Mesa emits for subgroupQuadBroadcast with a non-constant lane.  Kept so the
 * next person does not repeat it -- see the README.
 */
kernel void kernel_quad_half_reg(device ushort *dst [[buffer(0)]],
                                 constant ushort *src [[buffer(1)]],
                                 constant ushort &lane [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]])
{
   ushort a = src[tid];
   dst[tid] = quad_broadcast(a, lane);
}

kernel void kernel_simd_half_reg(device ushort *dst [[buffer(0)]],
                                 constant ushort *src [[buffer(1)]],
                                 constant ushort &lane [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]])
{
   ushort a = src[tid];
   dst[tid] = simd_broadcast(a, lane);
}
