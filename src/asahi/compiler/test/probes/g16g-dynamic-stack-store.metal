#include <metal_stdlib>
using namespace metal;
/* dynamically indexed private-array stores and loads (compute) */
kernel void cs64(device uint *out [[buffer(0)]], constant uint &n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
  uint a[64];
  for (uint i = 0; i < 64u; i++) a[i] = i * 3u;
  a[(tid * 7u + n) & 63u] = n;
  a[(n * 5u) & 63u] += tid;
  out[tid] = a[(tid + n) & 63u] + a[(n * 3u) & 63u];
}
kernel void cs256(device uint *out [[buffer(0)]], constant uint &n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
  uint a[256];
  for (uint i = 0; i < n; i++) a[i & 255u] = i * 3u;
  for (uint i = 0; i < n; i++) a[(i * 7u + tid) & 255u] += a[(i * 5u) & 255u];
  out[tid] = a[(tid + n) & 255u];
}
/* fragment: dynamic float array store then load, like the graphicsfuzz maze shaders */
struct V { float4 pos [[position]]; float4 c; };
vertex V vs(uint vid [[vertex_id]], device const float4 *p [[buffer(0)]]) { V o; o.pos = p[vid]; o.c = p[vid+3]; return o; }
fragment float4 fs64(V v [[stage_in]], constant uint &n [[buffer(1)]]) {
  float a[64];
  for (uint i = 0; i < 64u; i++) a[i] = float(i);
  uint k = uint(v.pos.x) & 63u;
  a[(k * 7u + n) & 63u] = v.c.x;
  a[(n * 5u) & 63u] += v.c.y;
  return float4(a[(k + n) & 63u], a[(n * 3u) & 63u], 0.0, 1.0);
}
