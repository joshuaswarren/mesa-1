// Copyright 2026 Aurora Silicon
// SPDX-License-Identifier: MIT
//
// Atomics on a UAV. Every thread contends for the same few addresses, so
// the result is only correct if the operations really are atomic on the
// GPU -- a non-atomic read-modify-write loses updates and lands short.
//
//   slot 0: InterlockedAdd of 1, once per thread   -> N
//   slot 1: InterlockedMax over the thread index   -> N-1
//   slot 2: InterlockedAdd of the thread index     -> sum(0..N-1)
//   slot 3: compare-exchange chain                 -> counted separately
RWStructuredBuffer<uint> o : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
   uint old;
   InterlockedAdd(o[0], 1, old);
   InterlockedMax(o[1], tid.x);
   InterlockedAdd(o[2], tid.x, old);

   // Only one thread can win the 0 -> 0xABC transition. Counted with a
   // select rather than an if, so this shader stays branch-free and tests
   // atomics alone.
   uint prev;
   InterlockedCompareExchange(o[3], 0, 0xABC, prev);
   InterlockedAdd(o[4], prev == 0 ? 1 : 0);
}
