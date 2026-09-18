#include <metal_stdlib>
using namespace metal;
/* seven and eight nested data-dependent loops: which flag slots do the inner loop tests use? */
kernel void loops7(device uint *out [[buffer(0)]], constant uint *n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
  uint acc = tid;
  for (uint a = 0; a < n[0]; a++)
   for (uint b = 0; b < n[1] + (acc & 1u); b++)
    for (uint c = 0; c < n[2] + (acc & 2u); c++)
     for (uint d = 0; d < n[3] + (acc & 4u); d++)
      for (uint e = 0; e < n[4] + (acc & 8u); e++)
       for (uint f = 0; f < n[5] + (acc & 16u); f++)
        for (uint g = 0; g < n[6] + (acc & 32u); g++)
          acc = acc * 3u + a + b + c + d + e + f + g;
  out[tid] = acc;
}
kernel void loops8(device uint *out [[buffer(0)]], constant uint *n [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
  uint acc = tid;
  for (uint a = 0; a < n[0]; a++)
   for (uint b = 0; b < n[1] + (acc & 1u); b++)
    for (uint c = 0; c < n[2] + (acc & 2u); c++)
     for (uint d = 0; d < n[3] + (acc & 4u); d++)
      for (uint e = 0; e < n[4] + (acc & 8u); e++)
       for (uint f = 0; f < n[5] + (acc & 16u); f++)
        for (uint g = 0; g < n[6] + (acc & 32u); g++)
         for (uint h = 0; h < n[7] + (acc & 64u); h++)
          acc = acc * 3u + a + b + c + d + e + f + g + h;
  out[tid] = acc;
}
