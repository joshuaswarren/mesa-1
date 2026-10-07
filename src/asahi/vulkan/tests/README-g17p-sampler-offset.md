# Texture-state pointer alignment on G17P

`g17p_vulkan_sampler_offset.c` reproduces the stock-vkcube GPU fault without a
window system. The G17P combined-sampler and texel-fetch ABIs give the texture
unit a pointer to 32-byte texture state, and the hardware ignores the low five
bits of that pointer. Sampled-image descriptors are only 8-byte aligned with a
72-byte stride, so a texture binding that follows a 16-byte buffer descriptor
starts at set offset 16. vkcube's set 0 has exactly this shape: a uniform
buffer at binding 0 and one combined image sampler at binding 1.

The probe samples a 4x4 RGBA8 texture with NEAREST filtering and checks every
texel exactly. The default graphics mode uses vkcube's layout. Each mode has a
`-control` variant that places a dynamic buffer at binding 0. A dynamic buffer
takes no set space, so the texture word moves to offset 0 while the shaders
stay the same. The compute modes repeat the layout with a storage buffer and an
unsigned sampled image read by `texelFetch`. They set `HK_PERFTEST=txf` before
device creation, selecting the direct texture-state txf ABI instead of the
indexed heap ABI that would otherwise win for this shader. The diagnostic marks
the fetch in bounds before texture lowering so Vulkan's out-of-bounds predicate
does not retain the bindless handle that makes the deliberately narrower direct
ABI decline. Run one mode per process:

```sh
cc -O2 -Wall -Wextra -Werror g17p_vulkan_sampler_offset.c -o sampler-offset -lvulkan
VK_DRIVER_FILES=/path/to/asahi.json ./sampler-offset graphics-control
VK_DRIVER_FILES=/path/to/asahi.json ./sampler-offset graphics
```

Exit status 0 means the result is exact. Status 1 means wrong texels or device
loss, and status 2 means a setup failure. `G17P_PROBE_RT_DIM` sets the size of
the render target (4..64; hk does not compress targets below 16x16).
`G17P_PROBE_STATIC_VIEWPORT=1` bakes the viewport into the pipeline.
`G17P_PROBE_ANY_DEVICE=1` runs the probe on other GPUs as a self-check. It
passes on an M2 (G14G) under stock Mesa 26.2.3.

On MacBook Neo, daily kernel 7.1.12-aurora-j700-mixed-t1271-daily-20261007,
`graphics` built from `82852e1e346` triggers a firmware stamp-blamed recovery
on the probe's own 3D queue and poisons its VM. This is the same signature as
stock vkcube on that build (t-1411). With `2e8f5f5987c` ("hk: publish
32-byte-aligned texture state for G17P combined samplers") applied to the same
tree, and with `651f448cf0d`, every mode passes exactly on both target sizes
and with either viewport mode. Only the 82852 build was measured failing.
Other bases that publish the unaligned set-relative pointer (anything before
`2e8f5f5987c`, including `6895114e17a`) are expected to fail the same way.

Without the diagnostic `txf` selector, hk lowers this shader to the indexed
sampler ABI (`idx_samp=1`). That ABI addresses the device-wide texture heap,
which is aligned by construction. The explicit selector is needed because the
direct txf lowering accepts a strict subset of the indexed lowering's shapes;
there is no GLSL shader that naturally makes only the indexed pass decline.
With the selector, `HK_G17P_LAUNCH_TRACE=1` reports `txf=1 idx_samp=0`, and the
probe covers the set-relative pointer in `hk_cmd_dispatch.c`.

A failing run costs a firmware queue slot until reboot. Never loop the probe
against a driver that is known to fail. The probe is deliberately not
registered as an automatic host unit test.
