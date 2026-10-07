# G17P Vulkan 1.4 CTS regression cases

`tools/vk-cts/g17p-known-failures-20261007.txt` preserves 35 historically
failing, crashing, or flaky cases from the Vulkan 1.4.6.2 campaign. It includes
both additional shader-object image mismatches and the later SNORM blend
flake. It is a regression caselist, with no skips or expected failures.

On 2026-10-07, all 35 cases passed individually with Mesa
`651f448cf0d219e7d5e65ade8f71b95875de18e6` on Linux/G17P. The case log preserves
images and shader sources. The 24 former compiler crashes include 18
`glsl.builtin_var.fragdepth` cases and six dynamic sample-shading cases.

Existing root fixes and their checks:

* `22ed2d7dab1`: sink conditional depth/stencil stores before Z/S export
  lowering. Regression: `asahi-zs-emit-control-flow` and
  `asahi-g17p-conditional-exports`.
* `afbd9f42da5`: preserve sampled values needed by derivatives in G17P helper
  lanes. Regression: all six `glsl.helper_invocations` cases and
  `asahi-g17p-helper-samples`, including ALU chains and loop phis.
* `82075087d0c`: back the sampled-image descriptor table on demand. The
  advertised capacity stays at one million descriptors. Initial backing is
  256 KiB rather than 32 MiB per logical device. Checks include
  `api.object_management.max_concurrent.device` and
  `api.info.vulkan1p2.feature_bits_influence` under the original memory budget.

The full multisample built-in, dynamic-state miscellaneous, and helper groups
contain 317 cases: 124 Pass and 193 NotSupported on this snapshot. Full object
management contains 457 cases: 453 Pass and four allocation-callback quality
warnings, identical to the historical warnings.

The original causes of intermittent sample-ID/position and the two
interrupted-pack image mismatches are not established by their current Pass
results. Preserve the original observations when comparing candidates. An
isolated Pass also does not resolve a prior runner deadline failure under
concurrent workload and thermal duty.

A normal CTS caselist invocation is:

```sh
VK_DRIVER_FILES="$ICD" "$DEQP" \
  --deqp-caselist-file=tools/vk-cts/g17p-known-failures-20261007.txt \
  --deqp-log-filename=g17p-known-failures.qpa \
  --deqp-log-images=enable --deqp-log-shader-sources=enable
```

On hardware managed by Aurora, execute tests through the approved hardware
queue and GPU guard. Record the kernel, boot ID, library hash, queue slots,
ENOSPC and submit-failed counters. Stop at the first new quarantine or poisoned
VM event. This caselist does not replace the complete mustpass run or the
stride-50 regression bar.
