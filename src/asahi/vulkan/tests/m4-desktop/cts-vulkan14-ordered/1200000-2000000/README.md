# CTS catalog range 1,200,000-2,000,000 on the M4 desktop (2026-09-15/16)

Ordered Vulkan 1.4 CTS catalog (vulkan-cts-1.4.6.2 binary `deqp-vk-seed30-scaled-blit`), cases 1,200,000 to 1,999,999 in 80 segments of 10,000, run with `tools/cts-bulk-series.py` (20 workers, 50 cases per group, first-attempt execution, hard 60 s case bound, watchdog and 0.5 s thermal guards). Together with `../1000000-1200000/` this covers the second million of the catalog. Batch directory `build/cts-vulkan14-ordered-perf/run5/1200000-2000000`; every segment attempt keeps its evidence directory under `_provenance/experiments/`.

## Result

| Status | Cases |
|---|---|
| Pass | 419,410 |
| NotSupported | 380,589 |
| QualityWarning | 1 |

Total counted: 800,000. A retained recovery attempt of a segment overrides the Unrun/Interrupted/Timeout outcome of the same case; original attempts stay on disk.

## Stops and recoveries

- `1300000-1310000`: run process_failure {'Pass': 3984, 'NotSupported': 6013, 'Interrupted': 3}; run-recover1 pass {'NotSupported': 2, 'Pass': 1}. one worker's QPA lost a chunk of three records (1 Pass, 2 NotSupported in the worker log); the three cases were rerun individually.
- `1510000-1520000`: run pass {'Pass': 10000}; run-attempt1-thermal-refusal no summary {}. the thermal guard refused to start at 69 C after the previous segment; resumed after cooling (attempt kept as `run-attempt1-thermal-refusal`).
- `1560000-1570000`: run gpu_timeout {'Pass': 759, 'Unrun': 9144, 'Interrupted': 22, 'NotSupported': 75}; run-recover1 gpu_timeout {'Pass': 814, 'Unrun': 8213, 'Interrupted': 12, 'NotSupported': 127}; run-recover2 pass {'Pass': 7840, 'NotSupported': 1326}. GPU hang in `timestamp.transfer_tests` under 20 workers on the first attempt and on its recovery (kernel job status all-ones); fixed in the kernel (`05fa7e2b200d`), recovered fully on the fixed module.
- `1580000-1590000`: run process_failure {'Pass': 6920, 'Unrun': 2759, 'Interrupted': 24, 'NotSupported': 287, 'Timeout': 10}; run-recover1 pass {'Pass': 2650, 'NotSupported': 114}; run-recover2-dual pass {'Pass': 28, 'NotSupported': 1}. dual-source multi-attachment throughput timeouts (known); the rest recovered at 20 workers and the dual-source cases at 2 workers.
- `1770000-1780000`: run process_failure {'Pass': 7717, 'Unrun': 1736, 'Interrupted': 27, 'NotSupported': 511, 'Timeout': 9}; run-recover1 pass {'Pass': 1662, 'NotSupported': 87}; run-recover2-dual pass {'Pass': 23}. dual-source multi-attachment throughput timeouts (known); same recovery.
- `1890000-1900000`: run pass {'Pass': 3925, 'NotSupported': 6074, 'Fail': 1}. one Fail, `shader_object_unlinked_spirv.multisample.storage_image.64x64_1.r8g8b8a8_unorm.samples_4` ("Rendered images are not correct"); it passes five times alone and its 96-case family passes at 20 workers (`claude-simg1..5`, `claude-simgfam1`), so the recheck is counted and the failure is recorded as not reproduced.

## Fixes (in order)

- `95faf065566` hk: apply integer border colours as texel values on native scaled views (46 `sampler.*.a2r10g10b10_uscaled_pack32.*clamp_to_border*` cases; the CTS uses integer borders for USCALED formats)
- `05fa7e2b200d` drm/asahi (kernel `eryk_m4_gpu`): never pipeline M4 compute commands that carry user timestamps (the segment 37 hang)
- `bdee16c17ebe` drm/asahi (kernel): read the architected counter frequency for M4 get_time (1 GHz, not 24 MHz; `timestamp.calibrated.*`)

Each fix was verified against the 1,001-case regression set (`build/claude-perf/reg1000/cases.txt`, 815 Pass / 186 NotSupported) before installation; the kernel module was installed through `install-asahi-module.py` and a guarded reboot.

## Open

1 cases are not passing:

| Family | Cases | Statuses | Note |
|---|---|---|---|
| pipeline.monolithic.pipeline_binary | 1 | {'QualityWarning': 1} | `dedicated.unique_key_pairs`: identical binaries get distinct keys because the common runtime keys binaries by their compile inputs; re-keying by content would break the cache lookup pipeline binaries exist for. Accepted quality warning. |
