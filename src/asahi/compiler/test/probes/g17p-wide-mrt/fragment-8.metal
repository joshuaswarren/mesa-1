#include <metal_stdlib>
using namespace metal;
vertex float4 mrt_vertex(uint id [[vertex_id]]) { return float4(float2((id<<1)&2,id&2)*2.0-1.0,0.0,1.0); }
struct Outputs {
float4 c0 [[color(0)]];
float4 c1 [[color(1)]];
float4 c2 [[color(2)]];
float4 c3 [[color(3)]];
float4 c4 [[color(4)]];
float4 c5 [[color(5)]];
float4 c6 [[color(6)]];
float4 c7 [[color(7)]];
};
fragment Outputs mrt_fragment(constant float4 *colors [[buffer(0)]]) { Outputs o;
o.c0 = colors[0];
o.c1 = colors[1];
o.c2 = colors[2];
o.c3 = colors[3];
o.c4 = colors[4];
o.c5 = colors[5];
o.c6 = colors[6];
o.c7 = colors[7];
return o; }
