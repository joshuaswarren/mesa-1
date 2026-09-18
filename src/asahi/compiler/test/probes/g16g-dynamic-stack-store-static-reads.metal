#include <metal_stdlib>
using namespace metal;
struct V { float4 pos [[position]]; float4 c; };
vertex V vs(uint vid [[vertex_id]], device const float4 *p [[buffer(0)]]) { V o; o.pos = p[vid]; o.c = p[vid+3]; return o; }
/* dynamic store followed by static reads of the same array */
fragment float4 fsstatic(V v [[stage_in]], constant uint &n [[buffer(1)]]) {
  float a[64];
  for (uint i = 0; i < 64u; i++) a[i] = float(i) * v.c.z;
  uint k = uint(v.pos.x) & 63u;
  a[(k * 7u + n) & 63u] = v.c.x;
  return float4(a[5], a[9], a[63], 1.0);
}
/* dynamic vec4 store and load */
fragment float4 fsvec4(V v [[stage_in]], constant uint &n [[buffer(1)]]) {
  float4 b[16];
  for (uint i = 0; i < 16u; i++) b[i] = float4(i) * v.c;
  uint k = uint(v.pos.x) & 15u;
  b[(k * 7u + n) & 15u] = v.c;
  return b[(k + n) & 15u];
}
