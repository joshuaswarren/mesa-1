# Allocation-flag buckets

The J713 0 A.D. CPU profile `20260911T114803313970Z` attributed 1.23% of
all-core CPU-clock samples to the driver thread's `agx_bo_cache_fetch`.
Sampled instruction offsets locate almost all of that cost in list traversal.

Each size class now has sixteen flag-hash buckets. Exact flag, size and
alignment checks remain mandatory. Hash collisions only affect search cost;
the global LRU, cache budget and eviction policy are unchanged.

Both ARM64 libraries build. The workspace `tools/bo-cache-test.py` exercises
the actual internal cache with synthetic BOs and no DRM device. It passes
mixed flags (including collisions and future flag bits), size/alignment
rejection, LRU trimming, shared-object rejection and complete eviction.

Guarded actual-game run `20260911T122040990231Z` passed 180 seconds and
34 health checks, peaking at 52 C. The final scene retains buildings/shadows.
Buffer-cache lookup falls below the profile's displayed 0.05% threshold;
driver-thread all-core samples fall from 3.04% to 2.12%. Game profiling records
2.07 FPS over the last120 complete frames: 452.5 ms/frame in swap buffers out
of 482.8 ms/frame overall. This reduces CPU overhead without establishing a
substantial gameplay improvement. Main-thread wall times include waits.

Full, unmodified trace replay `20260911T122543031078Z` passes without API or
shader errors, peaking at 52 C. Frames172530,179895,187257 are pixel-identical
to the previous M4 baseline. This does not remove the existing small llvmpipe
differences or cold GPU-readback defect. No other GPU platform is qualified
by these J713 results.
