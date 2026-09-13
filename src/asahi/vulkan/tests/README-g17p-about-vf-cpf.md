# G17P About VF_CPF oracle

`g17p_vulkan_about_vf_cpf.c` is a public-Vulkan reproduction of the
fragment-program shape retained from the Omarchy About fault. It embeds the
adjacent generated SPIR-V headers and draws one fullscreen triangle with:

- smooth `vec2` linkage plus `gl_PointSize` (HAL200 output size 7/select 5);
- set 0/binding 0 as a dynamically indexed `sampler2D[2]`;
- set 0/binding 1 as the 16-byte composite-parameter UBO; and
- one RGBA8 target, with guarded vertex, uniform, and readback buffers.

Build and validate the sources, headers, and ARM64 runner without GPU hardware
from the ARM64 Linux Mesa build environment:

```sh
timeout 90s src/asahi/isa/probes/run-f39-about-vf-cpf.sh _build-full
```

## Isolated hardware bundle

Transfer these runtime files together:

1. `_build-full/src/asahi/vulkan/f39-about-vf-cpf`
2. the exact candidate `libvulkan_asahi.so`
3. a target-local Vulkan ICD JSON whose `library_path` names that exact library

The two `.spv` files are audit artifacts; the executable embeds them and does
not read them during hardware execution. The target also needs its normal
ARM64 Vulkan loader and the shared libraries reported by `ldd` for the selected
driver. Do not transfer `asahi_devenv_icd.aarch64.json` unchanged when its
`library_path` still begins with the build-container path `/src`.

When running from a full build tree, select a relocated manifest with
`F39_VK_DRIVER_FILES` and use one cell per recovered GPU session:

```sh
F39_VK_DRIVER_FILES=/absolute/path/f39-asahi.json \
  timeout 90s src/asahi/isa/probes/run-f39-about-vf-cpf.sh \
  _build-full --hardware-baseline

F39_VK_DRIVER_FILES=/absolute/path/f39-asahi.json \
  timeout 90s src/asahi/isa/probes/run-f39-about-vf-cpf.sh \
  _build-full --hardware-flags1
```

Each cell has an inner 20-second timeout. The script requires the final traced
FragmentProgram word to be zero for baseline and one for `flags1`; the latter
must also log diagnostic activation. A GPU fault or timeout is a failed cell,
even if the expected word was traced. Run the second cell only after the first
session has recovered from any device loss.

This A/B test can establish whether the final record word is causal for this
shader. It cannot derive a production value: Metal sources the low nibble from
private compiler metadata field 32, for which Mesa has no proven semantic
equivalent.
