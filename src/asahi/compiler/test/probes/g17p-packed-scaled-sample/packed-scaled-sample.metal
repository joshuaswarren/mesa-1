#include <metal_stdlib>
using namespace metal;

kernel void
unpack_rgb10a2(texture2d<float, access::read> source [[texture(0)]],
               device float4 *destination [[buffer(0)]],
               uint2 gid [[thread_position_in_grid]])
{
   const float4 normalized = source.read(gid);
   destination[gid.y * source.get_width() + gid.x] =
      normalized * float4(1023.0f, 1023.0f, 1023.0f, 3.0f);
}
