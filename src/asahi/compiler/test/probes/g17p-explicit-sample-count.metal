#include <metal_stdlib>
using namespace metal;

struct VertexOut {
   float4 position [[position]];
   float value [[user(locn0)]];
};
struct FragmentIn {
   float4 position [[position]];
   interpolant<float, interpolation::perspective> value [[user(locn0)]];
};
vertex VertexOut sample_vertex(uint id [[vertex_id]]) {
   VertexOut out;
   out.position = float4(float2((id << 1) & 2, id & 2) * 2.0 - 1.0, 0, 1);
   out.value = float(id);
   return out;
}
fragment float4 sample_fragment(FragmentIn in [[stage_in]]) {
   return float4(in.value.interpolate_at_sample(1), 0, 0, 1);
}
fragment float4 sample_dynamic_fragment(FragmentIn in [[stage_in]],
                                       constant uint &sample [[buffer(0)]]) {
   return float4(in.value.interpolate_at_sample(sample), 0, 0, 1);
}
