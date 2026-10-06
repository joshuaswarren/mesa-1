// Copyright 2026 Aurora Silicon
// SPDX-License-Identifier: MIT
//
// A triangle whose depth varies per vertex, so the rasterizer has to
// interpolate Z across the primitive rather than take one value per draw.
//
// Geometry is stated explicitly rather than derived from the vertex index,
// because the test grades specific screen regions and needs to know exactly
// where the primitive is:
//
//   v0 = (-0.9, -0.9)  z = 0.9   bottom-left, far
//   v1 = ( 0.9, -0.9)  z = 0.9   bottom-right, far
//   v2 = ( 0.0,  0.9)  z = 0.1   top-centre, near
//
// NDC +y is up, so in the rendered image the triangle is near at the top and
// far at the bottom. Written with ternaries, not branches: they lower to
// selects, which our DXIL front end handles today.
float4 main(uint vid : SV_VertexID) : SV_Position
{
   float x = (vid == 0) ? -0.9 : ((vid == 1) ? 0.9 : 0.0);
   float y = (vid == 2) ? 0.9 : -0.9;
   float z = (vid == 2) ? 0.1 : 0.9;
   return float4(x, y, z, 1.0);
}
