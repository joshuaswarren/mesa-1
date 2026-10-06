// Copyright 2026 Aurora Silicon
// SPDX-License-Identifier: MIT
//
// Wave (subgroup) operations. Each thread writes results whose correctness
// depends on what OTHER lanes in its wave contributed, so a lane-local
// implementation produces different answers rather than plausible ones.
//
// Laid out per thread as a small record so a single dispatch covers the
// whole family and the CPU can check each field independently.
struct Result {
   uint lane;        // WaveGetLaneIndex
   uint width;       // WaveGetLaneCount
   uint sum;         // WaveActiveSum of 1        -> the wave's active count
   uint prefix;      // WavePrefixSum of 1        -> this lane's rank
   uint maxidx;      // WaveActiveMax of the lane index
   uint first;       // WaveReadLaneFirst of the lane index
   uint anytrue;     // WaveActiveAnyTrue(lane == 0)
   uint alltrue;     // WaveActiveAllTrue(lane < 1024)
};

RWStructuredBuffer<Result> o : register(u0);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
   Result r;
   r.lane    = WaveGetLaneIndex();
   r.width   = WaveGetLaneCount();
   r.sum     = WaveActiveSum(1u);
   r.prefix  = WavePrefixSum(1u);
   r.maxidx  = WaveActiveMax(WaveGetLaneIndex());
   r.first   = WaveReadLaneFirst(WaveGetLaneIndex());
   r.anytrue = WaveActiveAnyTrue(WaveGetLaneIndex() == 0) ? 1u : 0u;
   r.alltrue = WaveActiveAllTrue(WaveGetLaneIndex() < 1024u) ? 1u : 0u;
   o[tid.x] = r;
}
