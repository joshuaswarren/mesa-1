# CPU replay of Asahi compiler inputs

`AGX_SHADER_DUMP_PATH` writes shader inputs for every stage at the entry to the AGX
backend, including the exact backend key and serialized driver-lowered NIR.
Create the directory first. Files are named by stage and a BLAKE3 hash of the
complete input; identical inputs are retained once. Binary exclusive creation
works with the Windows CRT as well as native Linux. `AGX_MESA_DEBUG=refuse` names an instruction and
its operand shape when the AGX3 packer declines it.

For Vulkan applications, also set `MESA_SPIRV_DUMP_PATH` to a separate existing
directory. The compiler-input log includes the SPIR-V source BLAKE3 hash so
the lowered input can be matched to that dump. Applications may have their
own shader-dump environment variable; vkd3d-proton uses
`VKD3D_SHADER_DUMP_PATH`.

Build with `-Dbuild-tests=true -Dvulkan-drivers=asahi`, then run on any build
machine, without a GPU:

```sh
ninja -C build src/asahi/compiler/agx_replay
AGX_MESA_DEBUG=refuse build/src/asahi/compiler/agx_replay dumps/compute-*
build/src/asahi/compiler/agx_replay --print-nir dumps/compute-example
```

The tool reports PASS or REFUSED for each input, with a total refusal count.
The dump stores a backend key and Mesa's NIR serialization, so
capture and replay must use the same compiler revision, architecture and
code-generation settings. It is a diagnostic format, not a persistent shader
cache or interchange format. Preserve the original dumps when testing a
compiler fix; minimal regression witnesses should express the relevant
operation and operand shape in NIR rather than importing application assets.

For raw DXIL, use the same vkd3d-proton and dxil-spirv revisions and the original
root signature. Device features and descriptor layouts change conversion and
NIR lowering; a generic reflected layout or another Vulkan device may not
reproduce the same refusal. Asahi's existing DRM shim, selected with
`ASAHI_SHIM_CHIP=g17p`, can run the full stock conversion and pipeline compiler
without physical hardware. Preserve those backend inputs and use `agx_replay`
for before/after comparisons. Never submit a shader under the shim and claim
a physical GPU execution result.
