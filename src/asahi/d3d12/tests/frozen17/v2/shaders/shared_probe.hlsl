// Copyright 2026 Aurora Silicon
// SPDX-License-Identifier: MIT
//
// Groupshared memory and a local array.
//
//   `scratch` is groupshared: every thread writes its own slot, a barrier
//   makes the group's writes visible, then each thread reads a DIFFERENT
//   thread's slot. That read only returns the right value if the memory is
//   genuinely shared across the group and the barrier is honoured -- a
//   per-thread private array gives each lane its own value back.
//
//   `local` is a function-local array indexed dynamically, which is the
//   alloca/getelementptr path.
groupshared uint scratch[64];
RWStructuredBuffer<uint> o : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
   scratch[lane] = lane * 7 + 1;

   GroupMemoryBarrierWithGroupSync();

   // Read the slot 13 lanes along, wrapping: a value this thread never wrote.
   uint peer = (lane + 13) % 64;
   uint from_shared = scratch[peer];

   // A local array, indexed by a value the compiler cannot fold away.
   uint local[4];
   local[0] = from_shared;
   local[1] = from_shared * 2;
   local[2] = from_shared * 3;
   local[3] = from_shared * 4;

   o[tid.x] = local[lane % 4];
}
