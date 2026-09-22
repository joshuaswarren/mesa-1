#include <metal_stdlib>
using namespace metal;
/* many live floats produced by ushort/short/uint -> float conversions: which register forms does int2flt take above r63? */
kernel void hiregs(device float *out [[buffer(0)]], constant ushort *in [[buffer(1)]], constant int *sin [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  float v[96];
  for (uint i = 0; i < 96u; i++) v[i] = float(in[tid + i]) + float(sin[tid + i]) * 0.5f;
  float acc = 0.0f;
  for (uint i = 0; i < 96u; i++) acc += v[i] * v[(i * 37u + 11u) % 96u];
  out[tid] = acc;
}
kernel void hiregs16(device float *out [[buffer(0)]], constant ushort *in [[buffer(1)]], constant short *sin [[buffer(2)]], uint tid [[thread_position_in_grid]]) {
  float v[96];
  for (uint i = 0; i < 96u; i++) v[i] = float(in[tid + i]) + float(sin[tid + i]) * 0.5f;
  float acc = 0.0f;
  for (uint i = 0; i < 96u; i++) acc += v[i] * v[(i * 37u + 11u) % 96u];
  out[tid] = acc;
}
