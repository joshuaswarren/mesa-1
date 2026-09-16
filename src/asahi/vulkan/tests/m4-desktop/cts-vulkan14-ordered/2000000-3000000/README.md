# CTS catalog range 2,000,000-3,000,000 on the M4 desktop (2026-09-15/16)

Ordered Vulkan 1.4 CTS catalog (vulkan-cts-1.4.6.2 binary `deqp-vk-seed30-scaled-blit`), cases 2,000,000 to 2,999,999 in 100 segments of 10,000, 20 workers (6 for the synchronization segments 85-93 after a system OOM kill), 60 s per-case bound, 900 s runner bound, under the 60 s watchdog and the thermal guard. Batch `build/cts-vulkan14-ordered-perf/run6/2000000-3000000`, series `cts-3m-r6`, tally `build/claude-fix/r6-tally.py`.

## Result

| Status | Cases |
|---|---|
| Pass | 434,763 |
| NotSupported | 565,236 |
| Fail | 1 |

Total counted: 1,000,000. A retained recovery attempt (`run-recover*`) overrides the Unrun/Interrupted/Timeout/Crash outcome of the same case, and the targeted reruns with the final library (`build/claude-fix/r6-val{3,4,5,6}-run1`, `r6-remaining-run1`) override the first-attempt Fail outcomes they cover. First-attempt totals over the 100 segment runs: NotSupported 553,204, Pass 417,589, Unrun 28,862, Fail 188, Interrupted 155, Crash 1, Timeout 1.

## Remaining

- `dEQP-VK.subgroups.basic.graphics.subgroupmemorybarrierimage` (Fail, 0 of 27 values; also failed in the 1.2M-2.0M range). Compiles cleanly; image writes from vertex/tessellation/geometry stages under subgroup barriers give wrong values and need a hardware experiment.

## Stops and recoveries

- `2270000-2280000`: run gpu_timeout {'Pass': 2565, 'Unrun': 6302, 'NotSupported': 1112, 'Interrupted': 21}; run-recover1 pass {'Pass': 4492, 'NotSupported': 1830, 'Fail': 1}
- `2290000-2300000`: run gpu_timeout {'Pass': 5007, 'Unrun': 3944, 'Interrupted': 20, 'NotSupported': 1029}; run-recover1 pass {'Pass': 3286, 'NotSupported': 678}
- `2730000-2740000`: run pass {'Pass': 5380, 'NotSupported': 4620}; run-attempt1-thermal-refusal no summary {}
- `2750000-2760000`: run process_failure {'Pass': 7660, 'Interrupted': 9, 'NotSupported': 2331}; run-attempt1-thermal-refusal no summary {}; run-recover1 pass {'Pass': 9}
- `2760000-2770000`: run process_failure {'Pass': 5673, 'Unrun': 3569, 'Interrupted': 55, 'NotSupported': 702, 'Crash': 1}; run-recover1 pass {'Pass': 3191, 'NotSupported': 434}
- `2840000-2850000`: run process_failure {'Pass': 300, 'Unrun': 9039, 'NotSupported': 637, 'Interrupted': 24}; run-recover1 pass {'NotSupported': 6029, 'Pass': 3034}
- `2920000-2930000`: run process_failure {'Pass': 9893, 'Unrun': 103, 'Interrupted': 4}; run-recover1 pass {'Pass': 107}
- `2930000-2940000`: run process_failure {'Pass': 1957, 'Unrun': 5905, 'Interrupted': 22, 'NotSupported': 2115, 'Timeout': 1}; run-recover1 pass {'Pass': 2867, 'NotSupported': 3061}

Segments 28 and 30: GPU timeouts with the lost-follower signature (job status all-ones, queue consumed, no timestamps); the in-flight cases pass alone and together, the kernel now keeps the compute depth at one. Segments 74 and 76: thermal refusals at the segment start (moved aside as `run-attempt1-thermal-refusal`). Segments 76 and 93: QPA records lost for cases the runner reported as Pass (reran individually). Segment 77: compiler abort in the spiller (fixed). Segment 85: 20 workers of multi-queue tests exhausted the 15 GB of RAM (OOM kill; recovered and continued at 6 workers). Segment 94: one contention timeout (`tessellation.invariance.outer_triangle_set.triangles_fractional_even_spacing`, passes alone).

## Fixes (in order)

- `fc01aaa3889` hk: keep the first-fan B/C flat select and admit fans natively (`rasterization.flatshading.triangle_fan`, `provoking_vertex.draw.*.triangle_fan`); apply alpha-to-coverage in the main shader when it writes gl_SampleMask (`query_pool.discard.normal.*.alpha_to_coverage`); map B10G10R10A2_UNORM spilled stores (`dynamic_rendering.*.multisample_resolve.a2r10g10b10_unorm_pack32.max_attachments_8_samples_4`)
- `87418be6759` agx: ground SR 33/34/36 (SIMD-group active-thread index/count) from an Apple witness (`reconvergence.terminate_invocation.bit_count`)
- `cd3a288c9df` hk: native depth words pass everything under emulated NEVER (`rasterization.frag_side_effects.*.depth_never`); native cull bit 15 for Bresenham lines (`primitives_multisample_{2,4}_bit.no_stipple.bresenham_lines*`)
- `94a17b28046` hk/agx: robust index overreads reach the robustness kernel (`robustness.bind_index_buffer2.*.oo_*`, `index_access.draw*_indexed_2`); -|x| moves through the sign-bit route (`robustness2.*.vertex_attribute_fetch.*.len_25{2,6}.*.vert`); tessellation winding in the native cull word (`shader_object.misc.state.*.cull.*`, `shader_object.tessellation.*.orientation_ccw*`); default point size for hardware TES (`shader_object.tessellation.hlsl.point_mode*`); int8->float via the grounded half operations (`spirv_assembly.*.convertstof.int8_to_float*`)
- `518c9e5c833` agx: remat-only spiller falls back to scratch spilling (`ssbo.phys.layout.random.8bit.nested_structs_arrays.20` compiler abort)
- `f1531c5cf9e` agx: native fp32 min/max with IEEE NaN semantics from 34 Apple witnesses (`float_controls2.fp32.*.nmin/nmax/nclamp*`, `opnclamp.all`, `denorm_nclamp_nan_flush_to_zero`); whole-word scratch access on AGX3 (`subgroups.ballot_broadcast.*.subgroupbroadcast_{f,i,u}16vec*`, `{i,u}8vec*`)
- `9ee08342e0a` hk: preserve write-disabled attachments by reloading the tile (`dynamic_rendering.*.local_read.interaction_with_color_write_enable`)
- `7773a806872` agx: split 64-bit adds on AGX3 (`float_controls.fp32.input_args.rounding_rte_conv_from_{int,uint}64_*_vert`)
- `d689bc84f85e` drm/asahi (kernel `eryk_m4_gpu`): hold the M4 compute pipelining depth at one after two lost-follower GPU timeouts without timestamps (segments 28 and 30)

Each Mesa fix was validated on the M4 at 20 workers with the accumulated list of fixed cases (`build/claude-fix/r6-val*-run1`), the kernel change with the 1,001-case regression list (`build/claude-perf/reg1000/cases.txt`, 815 Pass / 186 NotSupported). Witness studies: `build/claude-fix/witness-sr59/REPORT.md`, `build/claude-fix/witness-fminmax-nan/REPORT.md`.

## Files

- `segments.json`: every attempt per segment with verdict, counts, runner seconds and peak temperature.
- `remaining.txt`: the remaining non-pass case with its first-attempt status, segment and message.
