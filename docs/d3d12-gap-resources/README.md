# Resource and descriptor validation follow-up (t-1109)

The validation change builds on Fable's refactor at `1d2ee857bd4`. It shares
AIL sizing between `GetResourceAllocationInfo` and methods 1/2/3, including
individual size padding, final heap padding, empty queries and overflow
refusals. Placed resources must fit the declared heap size as well as the BO;
rounded BO storage does not enlarge an application's heap. Constructor error
paths clear output pointers. Sampler descriptor heaps support validation-only
calls and reject invalid flags, node masks and shader-visible counts above
2,048. The existing native sampler-state limit remains 1,024 distinct states.

The arithmetic regression can run without a GPU:

```sh
clang -std=c11 -Wall -Wextra -Werror -fsanitize=address,undefined \
  src/asahi/d3d12/tests/d12_allocation_math_tests.c -o <aurora-ctl>/scratch/t1109/allocation-math-tests
<aurora-ctl>/scratch/t1109/allocation-math-tests
```

After a remote Mesa build, run `d3d12agx_resource_validation` against its matching
`libd3d12agx.so`. It checks all four allocation-info methods, mixed buffer/small
texture alignment, failure outputs, sampler heap capability queries and heap
reuse after rejected placements. Also run `d3d12agx_device_interfaces`, all
`tests/run-suites.sh` suites, and the sorted shader passset. Compare exact
passsets with `comm -23 before.passset after.passset`; a count alone cannot
establish absence of regressions. These hardware checks remain outstanding.

Use the t-1083 native corpus adapters without changing upstream test assertions.
Select these inventory categories: `d3d12_descriptors`, `d3d12_copy`,
`d3d12_resource`, `d3d12_clear` (136 registrations at Proton revision
`31d1f89ca5b3f2fd3b6f025c8e6f73ed3eaa852b`). Each registration must run in its own
process so one crash cannot suppress the rest of a file. The task-private
runner is the t-1083 `run-v6.py`, with strict TODO/known-bug failures preserved.
Use the same memory ceiling and timeouts for A and B. Keep cgroup OOM events,
process crashes, timeouts, missing verdicts and unmeasured tests distinguishable.

The report and request snapshots are under
`<aurora-ctl>/reports/d3d12-t1109-20261006/`. The human report is
`<aurora-ctl>/runs/r-t-1109-d3d12agx-gap-closing-b-resources-descriptors-cop-1/report.md`.
At the initial checkpoint no candidate remote build or current before/after
score is available: the controller rejects M1 Pro/M2 as unregistered and rejects
`execute --device buildbox` as Neo-only. Do not bypass the hardware queue.

This change does not establish zero crashes in the four corpus files. Remaining
work includes crash stack traces on a qualified runtime, block-compressed and
hardware-compressed copy coverage, CopyResource dimensional coverage, aliasing
barriers, descriptor-copy shader observations for every heap type, and the
required two-machine reruns. Existing sampler heaps, CopyDescriptors and raw
linear/twiddled copy paths were already present in the refactor base.

Public references: [CreateDescriptorHeap capability queries](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-createdescriptorheap),
[allocation-info array packing](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device4-getresourceallocationinfo1%28uint_uint_constd3d12_resource_desc_d3d12_resource_allocation_info1%29).
The new code uses Linux-side source and public API contracts, with no private
macOS reverse-engineering material.
