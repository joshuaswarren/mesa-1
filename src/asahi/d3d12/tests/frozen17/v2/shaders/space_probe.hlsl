// Copyright 2026 Aurora Silicon
// SPDX-License-Identifier: MIT
//
// The same register number in three different spaces, plus a fourth in a
// different class. Under the old fixed slot formula t0/space0 and
// t0/space2 collapsed onto the same slot and one silently won.
StructuredBuffer<uint>   a : register(t0, space0);
StructuredBuffer<uint>   b : register(t0, space2);
StructuredBuffer<uint>   c : register(t0, space5);
RWStructuredBuffer<uint> o : register(u0, space3);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
   o[tid.x] = a[tid.x] + b[tid.x] * 2 + c[tid.x] * 4;
}
