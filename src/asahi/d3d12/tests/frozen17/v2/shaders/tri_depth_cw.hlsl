// Copyright 2026 Aurora Silicon
// SPDX-License-Identifier: MIT
//
// tri_depth.hlsl with vertices 1 and 2 swapped: the same triangle in the
// same place, wound the other way round. The hardware keeps separate
// front-face and back-face state, so this is the only way to find out
// whether the back half is being programmed at all.
float4 main(uint vid : SV_VertexID) : SV_Position
{
   float x = (vid == 0) ? -0.9 : ((vid == 2) ? 0.9 : 0.0);
   float y = (vid == 1) ? 0.9 : -0.9;
   float z = (vid == 1) ? 0.1 : 0.9;
   return float4(x, y, z, 1.0);
}
