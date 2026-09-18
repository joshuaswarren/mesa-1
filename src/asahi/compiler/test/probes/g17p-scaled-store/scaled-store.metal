#include <metal_stdlib>
using namespace metal;

kernel void store_uscaled8(device uchar4 *dst [[buffer(0)]],
                           device const float4 *src [[buffer(1)]],
                           uint gid [[thread_position_in_grid]])
{
   dst[gid] = uchar4(clamp(rint(src[gid]), 0.0f, 255.0f));
}

kernel void store_sscaled8(device char4 *dst [[buffer(0)]],
                           device const float4 *src [[buffer(1)]],
                           uint gid [[thread_position_in_grid]])
{
   dst[gid] = char4(clamp(rint(src[gid]), -128.0f, 127.0f));
}
