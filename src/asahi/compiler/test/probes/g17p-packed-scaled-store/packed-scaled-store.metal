#include <metal_stdlib>
using namespace metal;

kernel void
store_rgb10a2(texture2d<uint, access::write> destination [[texture(0)]],
              device const float4 *source [[buffer(0)]],
              uint2 gid [[thread_position_in_grid]])
{
   const float4 rounded = rint(source[gid.y * destination.get_width() + gid.x]);
   const float4 clamped =
      clamp(rounded, float4(0.0f), float4(1023.0f, 1023.0f, 1023.0f, 3.0f));
   destination.write(uint4(clamped), gid);
}
