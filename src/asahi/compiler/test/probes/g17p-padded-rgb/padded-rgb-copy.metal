#include <metal_stdlib>
using namespace metal;

kernel void copy_rgb_to_padded(device uchar *dst [[buffer(0)]],
                               device const uchar *src [[buffer(1)]],
                               uint gid [[thread_position_in_grid]])
{
   const uint src_offset = gid * 3;
   const uint dst_offset = gid * 4;
   dst[dst_offset + 0] = src[src_offset + 0];
   dst[dst_offset + 1] = src[src_offset + 1];
   dst[dst_offset + 2] = src[src_offset + 2];
}

kernel void copy_padded_to_rgb(device uchar *dst [[buffer(0)]],
                               device const uchar *src [[buffer(1)]],
                               uint gid [[thread_position_in_grid]])
{
   const uint src_offset = gid * 4;
   const uint dst_offset = gid * 3;
   dst[dst_offset + 0] = src[src_offset + 0];
   dst[dst_offset + 1] = src[src_offset + 1];
   dst[dst_offset + 2] = src[src_offset + 2];
}
