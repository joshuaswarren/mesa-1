#include <metal_stdlib>
using namespace metal;
vertex float4 mrt_vertex(uint id [[vertex_id]]) { return float4(float2((id<<1)&2,id&2)*2.0-1.0,0.0,1.0); }
struct Outputs {
float4 c1 [[color(1)]];
float4 c3 [[color(3)]];
};
fragment Outputs mrt_fragment(constant float4 *colors [[buffer(0)]]) { Outputs o;
o.c1 = colors[1];
o.c3 = colors[3];
return o; }
