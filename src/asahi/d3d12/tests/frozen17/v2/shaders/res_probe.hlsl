// Copyright 2026 Aurora Silicon
// SPDX-License-Identifier: MIT
//
// Resources at deliberately non-zero, non-matching registers in three
// different classes, plus a structured buffer whose stride is not 4.
// Everything the binding model needs to disambiguate is distinct here:
// if t2/u3/b1 all still looked like "slot 0", nothing below would.
struct Wide { uint a; uint b; uint c; };   // stride 12, not 4

StructuredBuffer<Wide>     src : register(t2);
RWStructuredBuffer<uint>   dst : register(u3);
cbuffer Params : register(b1) { uint scale; uint bias; };

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
   Wide w = src[tid.x];
   dst[tid.x] = w.a * scale + w.c + bias;
}
