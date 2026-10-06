// Copyright 2026 Aurora Silicon
// SPDX-License-Identifier: MIT
//
// A second solid-colour pixel shader, in a colour distinct from solid.hlsl.
// Depth tests need two draws to be told apart in the colour buffer; with one
// shader colour, "the near triangle won" and "the far triangle won" produce
// identical images and the test proves nothing.
float4 main() : SV_Target
{
   return float4(0.0, 1.0, 0.0, 1.0);
}
