#include <metal_stdlib>
using namespace metal;
vertex float4 colour_vertex(uint id [[vertex_id]]) {
   return float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.5, 1);
}
fragment float4 colour_fragment(float4 pos [[position]], uint sample [[sample_id]],
                                constant float4 &colour [[buffer(0)]]) {
   float4 out = colour;
   out.a *= ((sample + uint(pos.x)) & 1) ? 0.5 : 1.0;
   return out;
}
