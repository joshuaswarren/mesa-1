// Copyright 2026 Aurora Silicon
// SPDX-License-Identifier: MIT
//
// Texture.Load: an unfiltered, unsampled fetch at integer coordinates.
// No sampler is involved, so this isolates the texture DESCRIPTOR path --
// AGX reaches a texture through a hardware descriptor rather than an
// address, and this proves that descriptor is built and found correctly.
//
// Each thread reads one texel of a 16x16 RGBA8 texture and writes the four
// channels back as bytes, so a channel swizzle or a wrong row stride shows
// up as specific wrong values rather than as nothing at all.
Texture2D<float4>        tex : register(t0);
RWStructuredBuffer<uint> o   : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
   uint x = tid.x % 16;
   uint y = tid.x / 16;

   float4 c = tex.Load(int3(x, y, 0));

   o[tid.x] = (uint(c.r * 255.0 + 0.5))
            | (uint(c.g * 255.0 + 0.5) << 8)
            | (uint(c.b * 255.0 + 0.5) << 16)
            | (uint(c.a * 255.0 + 0.5) << 24);
}
