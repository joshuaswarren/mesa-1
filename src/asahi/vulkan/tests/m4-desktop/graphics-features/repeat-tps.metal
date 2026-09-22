#include <metal_stdlib>
using namespace metal;
kernel void cs(device uint *p [[buffer(0)]], uint t [[thread_position_in_grid]]) {
 volatile uint4 a[16];
 for(uint i=0;i<16;i++) a[i]=((device uint4 *)p)[t*17+i];
 uint index=p[t*68+64]&15;
 uint4 x=a[index]; uint4 y=a[0]; uint4 z=a[index]; ((device uint4 *)p)[t*17+16]=x^y^z;
}
