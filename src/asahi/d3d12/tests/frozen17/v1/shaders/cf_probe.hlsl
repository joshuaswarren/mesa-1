// Copyright 2026 Aurora Silicon
// SPDX-License-Identifier: MIT
//
// Control flow that cannot be flattened into selects: a loop with a
// data-dependent trip count, an early-out branch, and a value that is
// live across the merge (a phi).
RWStructuredBuffer<uint> o : register(u0);
StructuredBuffer<uint>   n : register(t0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
   uint count = n[tid.x];

   // Early out: the store must not happen for these lanes.
   if (count == 0xFFFF) {
      o[tid.x] = 0xDEAD;
      return;
   }

   // Data-dependent loop; `acc` is a phi across the back edge.
   uint acc = 1;
   for (uint i = 0; i < count; i++)
      acc = acc * 3 + i;

   // A second phi: which value survives the merge depends on the branch.
   uint tag = (acc & 1) ? 0x100 : 0x200;
   o[tid.x] = acc + tag;
}
