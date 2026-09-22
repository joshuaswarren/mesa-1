#include <metal_stdlib>
using namespace metal;

/* value in a register, count in a register -- the shape our packer handles */
kernel void reg_reg(device int *out [[buffer(0)]], device const int *in [[buffer(1)]])
{
    out[0] = in[0] >> in[1];
}

/* IMMEDIATE value, register count -- the shape our packer refuses */
kernel void imm1_reg(device int *out [[buffer(0)]], device const int *in [[buffer(1)]])
{
    out[0] = 1 >> in[0];
}

kernel void imm5_reg(device int *out [[buffer(0)]], device const int *in [[buffer(1)]])
{
    out[0] = 5 >> in[0];
}

kernel void immneg_reg(device int *out [[buffer(0)]], device const int *in [[buffer(1)]])
{
    out[0] = (-7) >> in[0];
}

/* unsigned immediate value, register count */
kernel void uimm1_reg(device uint *out [[buffer(0)]], device const uint *in [[buffer(1)]])
{
    out[0] = 1u >> in[0];
}
