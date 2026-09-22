#include <metal_stdlib>
using namespace metal;

struct FsOut {
   half4 color [[color(0)]];
};

struct VertexOut {
   float4 position [[position]];
   float4 coordinate [[user(locn0)]];
   float lod [[user(locn1)]];
};

vertex VertexOut vertex_main(uint vertex_id [[vertex_id]])
{
   constexpr float2 positions[] = {
      float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0),
   };
   VertexOut out;
   out.position = float4(positions[vertex_id], 0.0, 1.0);
   out.coordinate = float4(0.25, 0.5, 0.75, 1.0);
   out.lod = float(vertex_id);
   return out;
}

fragment FsOut half_uniform_clamp(constant half4 &value [[buffer(0)]])
{
   FsOut out;
   out.color = clamp(value, half4(-1.0h), half4(1.0h));
   return out;
}

struct CubeIn {
   float4 coordinate [[user(locn0)]];
   float lod [[user(locn1)]];
};

fragment float4 cube_array_register_lod(
   texturecube_array<float> tex [[texture(0)]],
   sampler samp [[sampler(0)]],
   CubeIn in [[stage_in]])
{
   return tex.sample(samp, in.coordinate.xyz, uint(in.coordinate.w),
                     level(in.lod));
}
