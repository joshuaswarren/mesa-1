# CTS catalog range 3,000,000-3,206,030 on the M4 desktop (2026-09-16)

Ordered Vulkan 1.4 CTS catalog (vulkan-cts-1.4.6.2 binary `deqp-vk-seed30-scaled-blit`), cases 3,000,000 to 3,206,029 (the end of the 3,206,030-case catalog) in 21 segments of 10,000, 20 workers, 60 s per-case bound, 900 s runner bound, under the 60 s watchdog and the thermal guard. Batch `build/cts-vulkan14-ordered-perf/run7/3000000-3206030`, series `cts-3m2-r7`, tally `build/claude-fix/r7-tally.py`. This closes the catalog: every range from 0 to 3,206,030 has now been run.

## Result

| Status | Cases |
|---|---|
| Pass | 53,670 |
| NotSupported | 152,358 |
| Fail | 0 |
| Timeout | 2 |

Total counted: 206,030. Retained recovery attempts (`run-recover*`, chronological) override the Unrun/Interrupted outcome of the same case, and the targeted reruns with the final library (`build/claude-fix/r7-ycc-val2`, `r7-wsi-final*`, `r7-wsi-cs-*`) override the first-attempt outcomes they cover. First-attempt totals over the 21 segment runs: NotSupported 149,155, Pass 53,506, Unrun 3,244, Fail 72, Interrupted 46, Timeout 7.

## Remaining: `wsi.wayland.colorspace.{basic,hdr}` (Timeout at the 60 s bound)

The WSI segment (3,130,000-3,140,000) runs against the login greeter's Hyprland on `wayland-1` with no user session. At 20 and 4 workers the concurrent WSI cases contend for that compositor: six failed (`maintenance1.present_modes.heterogenous.*` acquire timeouts, `present_timing.*` "Failed to retrieve all timing results") and six timed out. Rerun one case per process with the final library (`build/claude-fix/r7-wsi-final*`), ten of the twelve pass, so those were contention outcomes, not driver failures. The two that remain, `colorspace.basic` and `colorspace.hdr`, render 60 FIFO-paced frames for every supported surface format and colour space with an unbounded acquire wait and log nothing before the 60 s per-case bound (`build/claude-fix/r7-wsi-cs-{basic,hdr}`); they need a WSI run inside a user compositor session with a mapped surface. The per-case bound stays at 60 s.

## Stops and recoveries

- `3110000-3120000`: run (20 workers) pass {'NotSupported': 9998, 'Pass': 2}; run-attempt1-display-recovery (None workers) no summary {}
- `3130000-3140000`: run (20 workers) process_failure {'NotSupported': 6590, 'Unrun': 3244, 'Pass': 105, 'Timeout': 7, 'Interrupted': 46, 'Fail': 8}; run-recover1 (4 workers) process_failure {'Unrun': 1442, 'NotSupported': 1802, 'Timeout': 2, 'Pass': 54, 'Interrupted': 5}; run-recover2 (1 workers) pass {'NotSupported': 1401, 'Pass': 36}

Segment 3,110,000-3,120,000: the WSI guard reported that the internal display needed recovery before the first WSI cases; the attempt was moved aside (`run-attempt1-display-recovery`, no summary), the display manager was stopped and the target rebooted through `/root/guarded-reboot.py`, then the segment ran clean. Segment 3,130,000-3,140,000 (WSI): compositor-bound cases stop the runner as process failures; after the 20-worker and 4-worker attempts the remaining cases ran in single-worker rounds (`build/claude-fix/r7-wsi-loop.sh`, `run-recover2`) excluding the cases that had already ended as Fail or Timeout.

## Fixes

- `e32a14a1e8b` hk: blend 16-bit YCbCr chroma planes in the shader. The texture unit filters 16-bit UNORM planes at 16-bit precision, so a linear blend of two texels rounds to a whole code (the CTS bounds the reconstruction to a 2^-16 step); the 48 `ycbcr.conversion.g16_b16_r16_3plane_{420,422}` and `g16_b16r16_2plane_{420,422}` `chroma_reconstruction.nearest.explicit_linear_*` cases returned e.g. 1725/65535 for an exact 1724.5/65535. After the common YCbCr pass splits the sample into plane accesses, a linear-filtered access to a 16-bit chroma plane becomes four nearest fetches and an fp32 blend whose position is first converted to fixed point with the advertised eight sub-texel bits, like the hardware (without that, samples on texel boundaries returned a 2^-26 residue of a neighbour where the reference demands zero). Validated with all 7,144 `ycbcr.conversion`/`ycbcr.filtering` cases of the 16-bit, 10x6 and 12x4 multi-planar formats (`build/claude-fix/r7-ycc-val2`: 6,100 Pass / 1,044 NotSupported) and the 1,001-case regression list (`build/claude-fix/r7-reg1000`: 815 Pass / 186 NotSupported).

## Files

- `segments.json`: every attempt per segment with verdict, worker count, counts, runner seconds and peak temperature.
- `remaining.txt`: the remaining non-pass cases with status, segment and message.
