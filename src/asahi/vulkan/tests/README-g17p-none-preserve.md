# Full-area untouched depth/stencil preservation

Build the explicit `g17p-vulkan-none-preserve` target with Mesa tests enabled.
It is deliberately not registered with `meson test` and is not built by default.
Run it with the intended Vulkan ICD selected and the normal hardware health guard.
An optional integer argument (1..4096) repeats the middle pass's draw.

Sixteen cases test full 32x32 and partial 27x19 render areas, primary and
secondary draws, and four middle-pass modes: untouched NONE/NONE, active
depth writes, active stencil writes, and CLEAR load operations. The source
initializes known depth/stencil values on each half of the image, checks
exact readback of both aspects, then verifies a depth-dependent color draw.
Guard bytes around each readback allocation must remain untouched.

The original bug loses all 1024 depth/stencil values in the two full-area
NONE/NONE cases. The standard load-store-none CTS uses a 27x19 area within a
32x32 image, which forces a reload and masks that path. The partial-area and
active-use cases are negative controls for that distinction.

For a bounded split-stream test, run the same binary with argument `3` and
`HK_G17P_DRAWS_PER_STREAM=1`. This is a diagnostic setting for that run only;
it is not a driver performance configuration. Use the ordinary driver path
for performance measurements. Check the driver's stream trace when claiming
that the split occurred.

Regenerate the checked-in SPIR-V headers with
`python3 gen_g17p_none_preserve.py`. This needs glslangValidator and spirv-val.
The hardware fixture itself needs only a Vulkan loader at runtime.
