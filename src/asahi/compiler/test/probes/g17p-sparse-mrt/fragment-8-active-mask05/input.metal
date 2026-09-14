#include <metal_stdlib>
using namespace metal;
vertex float4 mrt_vertex(uint id [[vertex_id]]) { return float4(float2((id<<1)&2,id&2)*2.0-1.0,0.0,1.0); }
struct Outputs {
float4 c0 [[color(0)]];
float4 c2 [[color(2)]];
};
fragment Outputs mrt_fragment(constant float4 *colors [[buffer(0)]]) { Outputs o;
o.c0 = colors[0];
o.c2 = colors[2];
return o; }
