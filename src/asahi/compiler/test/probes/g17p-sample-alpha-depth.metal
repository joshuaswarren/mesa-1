#include <metal_stdlib>
using namespace metal;
vertex float4 depth_vertex(uint id [[vertex_id]]) {
   return float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0.5, 1);
}
struct SampleDepth {
   float4 color [[color(0)]];
   float depth [[depth(any)]];
};
fragment SampleDepth depth_fragment(float4 pos [[position]],
                                    uint sample [[sample_id]]) {
   float alpha = (int(pos.x) & 1) ? 0.2 + 0.2 * float(sample) : 0.4;
   return {float4(0.25, 0.5, 0.75, alpha), 0.2 + 0.1 * float(sample)};
}
