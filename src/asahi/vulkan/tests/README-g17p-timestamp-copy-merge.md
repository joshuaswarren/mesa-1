# G17P timestamp-copy merge regression

`tests/g17p_timestamp_copy_merge.c` exercises the earlier-subqueue prefix that must survive `hk_sync_merge`. It is a standalone manual probe; the default mode requires the loaded G17P no-op DRM shim and does not check GPU results.

The sequence is deliberately narrow:

1. Enable `hostQueryReset` and host-reset one timestamp query. A GPU query reset introduces extra ordering and can hide this regression.
2. Record `BOTTOM_OF_PIPE` timestamp, a 1,261,568-byte buffer copy, then `WAIT_BIT | 64_BIT` query-result copy with a 24-byte stride.
3. Submit and wait idle. This must retain two CDMs, three dispatches and one merge. The timestamp producer is command index 1; the merged consumer must have `cdm_barrier=1`. The unfixed implementation records `-1` (none).
4. Only after that wait, submit a separate transfer-write → host-read barrier and wait again. Moving this barrier into the first command buffer can conceal the missing dependency.
5. In explicitly enabled hardware mode, compare the copied timestamp to `vkGetQueryPoolResults`, check all bulk-copy bytes against a nonconstant source pattern, and check the 16 guard bytes after the timestamp.

The cleanup removes unused image/rendering/query-type helpers and enables only the host-query-reset feature. All three buffers are released on successful completion.

## CPU recording

From src/asahi/vulkan, build with Vulkan headers and loader installed:

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror \
  tests/g17p_timestamp_copy_merge.c -lvulkan -ldl \
  -o g17p_timestamp_copy_merge
```

For an in-tree Mesa header checkout, add `-I/path/to/mesa/include`; systems with only the loader runtime may use `-l:libvulkan.so.1` instead of `-lvulkan`.

Run using an ICD JSON that points at the driver under test:

```sh
LD_PRELOAD=/absolute/path/libasahi_noop_drm_shim.so \
ASAHI_SHIM_CHIP=g17p VK_DRIVER_FILES=/absolute/path/icd.json \
HK_SYNC_TRACE=1 HK_G17P_SUBMIT_TRACE=1 \
timeout 30 ./g17p_timestamp_copy_merge
```

The program checks that `ioctl` actually resolves to the loaded no-op shim before creating a Vulkan instance. It prints `TIMESTAMP_COPY_MERGE_RECORDED results_not_checked=1`; this is recording completion, not a GPU correctness pass. Inspect the first submission's dependency and counts to distinguish fixed from unfixed drivers.

## Explicit hardware qualification

Build separately with `-DG17P_TIMESTAMP_COPY_MERGE_HARDWARE` and pass `--hardware`; both are required. Hardware mode rejects `LD_PRELOAD`. Run only the fixed, identified driver with the normal external timeout/health guard:

```sh
cc -std=c11 -O2 -DG17P_TIMESTAMP_COPY_MERGE_HARDWARE \
  tests/g17p_timestamp_copy_merge.c -lvulkan -ldl \
  -o g17p_timestamp_copy_merge_hardware
VK_DRIVER_FILES=/absolute/path/fixed-icd.json \
HK_SYNC_TRACE=1 HK_G17P_SUBMIT_TRACE=1 \
timeout --kill-after=3 35 ./g17p_timestamp_copy_merge_hardware --hardware
```

The unfixed dependency is tested with the no-op shim. Successful hardware readback prints `TIMESTAMP_COPY_MERGE_READBACK_PASS`.
