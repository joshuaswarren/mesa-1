/* SPDX-License-Identifier: MIT
 * Integer operand-domain witnesses. These compile probes do not establish
 * runtime correctness or permission to enable unrelated encodings.
 */
#include <metal_stdlib>
using namespace metal;

#define CASE(T, name, expression) \
kernel void name(device const T *input [[buffer(0)]], \
                 constant T &p [[buffer(1)]], \
                 device T *output [[buffer(2)]], \
                 uint id [[thread_position_in_grid]]) \
{ T x = input[id]; output[id] = T(expression); }

#define GROUP(T, prefix) \
CASE(T, prefix##_add, x + p) \
CASE(T, prefix##_sub, x - p) \
CASE(T, prefix##_sub_reverse, p - x) \
CASE(T, prefix##_mul, x * p) \
CASE(T, prefix##_mad_uniform, x * p.x + p.y) \
CASE(T, prefix##_mad_register, x * p + x.wzyx) \
CASE(T, prefix##_and, x & p) \
CASE(T, prefix##_or, x | p) \
CASE(T, prefix##_xor, x ^ p) \
CASE(T, prefix##_nand, ~(x & p)) \
CASE(T, prefix##_nor, ~(x | p)) \
CASE(T, prefix##_xnor, ~(x ^ p)) \
CASE(T, prefix##_andnot, x & ~p) \
CASE(T, prefix##_notand, ~x & p) \
CASE(T, prefix##_shift_left, x << p) \
CASE(T, prefix##_shift_right, x >> p) \
CASE(T, prefix##_shift_add, (x << 3) + p) \
CASE(T, prefix##_extract, (x >> p.x) & T(255)) \
CASE(T, prefix##_min, min(x, p)) \
CASE(T, prefix##_max, max(x, p)) \
CASE(T, prefix##_select, select(x, p, x < p)) \
CASE(T, prefix##_rotate, rotate(x, p))

GROUP(uint4, u32)
GROUP(int4, i32)
GROUP(ushort4, u16)
GROUP(short4, i16)
