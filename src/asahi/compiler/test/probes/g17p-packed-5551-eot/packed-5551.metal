#include <metal_stdlib>
using namespace metal;

vertex float4
packed_5551_vertex(uint id [[vertex_id]])
{
   return float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.0, 1.0);
}

fragment float4
packed_5551_fragment(constant float4 *color [[buffer(0)]])
{
   return color[0];
}
