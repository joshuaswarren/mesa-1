/* SPDX-License-Identifier: MIT
 * Paired arithmetic and uniform-materialization witnesses. Keep each kernel's
 * output observable and use explicit fma where contraction is part of the case.
 */
#include <metal_stdlib>
using namespace metal;

#define FLOAT_CASE(name, expression)                                           \
kernel void name(device const float4 *input [[buffer(0)]],                      \
                 constant float4 &p [[buffer(1)]],                            \
                 device float4 *output [[buffer(2)]],                          \
                 uint id [[thread_position_in_grid]])                         \
{ float4 x = input[id]; output[id] = (expression); }
#define HALF_CASE(name, expression)                                            \
kernel void name(device const half4 *input [[buffer(0)]],                       \
                 constant half4 &p [[buffer(1)]],                             \
                 device half4 *output [[buffer(2)]],                           \
                 uint id [[thread_position_in_grid]])                         \
{ half4 x = input[id]; output[id] = (expression); }
#define UINT_CASE(name, expression)                                            \
kernel void name(device const uint4 *input [[buffer(0)]],                       \
                 constant uint4 &p [[buffer(1)]],                             \
                 device uint4 *output [[buffer(2)]],                           \
                 uint id [[thread_position_in_grid]])                         \
{ uint4 x = input[id]; output[id] = (expression); }

FLOAT_CASE(uniform_repeat, x * p.x + p.y)
FLOAT_CASE(uniform_chain, fma(fma(x, p.x, p.y), p.x, p.z))
FLOAT_CASE(uniform_swizzle, fma(x, p.xyxy, p.zwzw))
FLOAT_CASE(uniform_dot, float4(dot(x, p)))
FLOAT_CASE(uniform_add, x + p)
FLOAT_CASE(uniform_mul, x * p)
FLOAT_CASE(uniform_add_absneg, -abs(x) + abs(p))
FLOAT_CASE(uniform_mul_absneg, -abs(x) * abs(p))
FLOAT_CASE(uniform_fma_absneg, fma(-abs(x), abs(p), -abs(p.wzyx)))
FLOAT_CASE(uniform_fma_addend, fma(x, x.wzyx, p))
FLOAT_CASE(uniform_fma_immediate, fma(x, float4(0.5f), p))
FLOAT_CASE(uniform_fma_uniform_imm, fma(x, p, float4(0.5f)))
FLOAT_CASE(float_fma, fma(x, p, x.wzyx))
FLOAT_CASE(float_clamp, clamp(x, p.x, p.y))
FLOAT_CASE(float_absneg, fma(-abs(x), p, abs(x.wzyx)))
FLOAT_CASE(float_minmax, min(max(x, p.x), p.y))
FLOAT_CASE(float_recip, x / p)
FLOAT_CASE(float_rsqrt, rsqrt(abs(x) + p))
FLOAT_CASE(float_round, floor(x) + ceil(p))
FLOAT_CASE(float_select, select(x, -x, x < p))
HALF_CASE(half_repeat, x * p.x + p.y)
HALF_CASE(half_chain, fma(fma(x, p.x, p.y), p.x, p.z))
HALF_CASE(half_siblings, fma(x, p.xyxy, p.zwzw))
HALF_CASE(half_add, x + p)
HALF_CASE(half_mul, x * p)
HALF_CASE(half_dot, half4(dot(x, p)))
HALF_CASE(half_clamp, clamp(x, p.x, p.y))
HALF_CASE(half_absneg, fma(-abs(x), p, abs(x.wzyx)))
UINT_CASE(integer_mad, x * p.x + p.y)
UINT_CASE(integer_shift, (x << 3) + p)
UINT_CASE(integer_rotate, rotate(x, p))
UINT_CASE(integer_extract, (x >> p.x) & 255u)
UINT_CASE(integer_logic, (x & p.x) ^ p.y)
UINT_CASE(integer_select, select(x, p, x < p))
