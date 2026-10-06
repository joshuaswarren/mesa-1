# Frozen17 v2

This is a separately versioned correction of the 17-suite reference from
`ba4cfd5dfa4c4d6f5b88adcf9ac5eb233c3442a4` (t-1070).
`v1/` contains the original **63 byte-identical inputs**. `v2/` contains
64 inputs: the same set plus one constant-blue pixel shader. The original
unversioned tests, shaders and runner are unchanged. `manifest.json` records
suite intent, arguments, hashes and the exact changed-input set; `v1.sha256`
and `v2.sha256` are independently usable checksum lists.

Only G1, G5 depth, G6, G7, G8 and G9 change. The other eleven fixtures, their
arguments and all their shared inputs retain their v1 bytes. The score always
uses **17** required attempts. Refusal, crash, timeout, missing input, no
verdict, or a zero exit without ALL PASS is a failure. No stage is skipped or
substituted with a weaker acceptance test.

## Rationale by fixture

### G1: device, capabilities, queue/fence and heap ABI

V1 hard-codes an early driver's capability profile and expects heap APIs to
remain stubs. V2 preserves device creation, queue/fence ordering, timestamp,
allocator and COM checks, while checking public query contracts:

* Architecture flags are reported and cache-coherent UMA implies UMA. They
  describe the hardware **and driver**, rather than an immutable chip profile.
  See [ARCHITECTURE1](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_architecture1).
* The maximum feature level must be in the requested 11_0/12_0 set and must
  pass device probing. See [FEATURE_LEVELS](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_feature_levels).
* Shader model queries descend if the runtime does not recognize a version;
  successful output is bounded by the input and round-trips. Exact 6_2 is not
  mandated. See [SHADER_MODEL](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_shader_model).
* Both defined [resource heap tiers](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_resource_heap_tier)
  are valid. OPTIONS3 must succeed; it need not advertise optional timestamp
  or immediate-write support. [CheckFeatureSupport](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-checkfeaturesupport)
  queries support rather than promising every queried feature exists.
* PREDICATION and HARDWARE_COPY allow S_OK, E_INVALIDARG or E_NOTIMPL.
  The pinned [vkd3d-proton device implementation](https://github.com/HansKristian-Work/vkd3d-proton/blob/31d1f89ca5b3f2fd3b6f025c8e6f73ed3eaa852b/libs/vkd3d/device.c#L5495)
  documents Windows E_NOTIMPL for these undocumented queries. Native Linux can
  have a zero DXGI LUID; check stability rather than nonzero. The same source
  copies the LUID supplied at device creation; its Linux path supplies zero.
* A valid 4096-byte buffer's allocation info must have sufficient size,
  default 64KB alignment and an aligned total. A DEFAULT, buffer-only heap
  uses that info and must be created successfully with a matching GetDesc.
  See [allocation info](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-getresourceallocationinfo%28uint_uint_constd3d12_resource_desc%29),
  [heap descriptors](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_heap_desc),
  and [CreateHeap](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-createheap).
  This replaces an invalid zero-property descriptor/E_NOTIMPL expectation and
  a zero-resource query used to test an obsolete aggregate-return stub.

### G5 depth: depth behaviour and cull winding

[RASTERIZER_DESC](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_rasterizer_desc)
defines FrontCounterClockwise using winding on the render target. The frozen
`tri_depth` triangle is counter-clockwise when viewed on that target. With
FCC=FALSE it is back-facing: NONE draws, BACK removes, FRONT keeps, and
FRONT+FCC=TRUE removes. V1 expects the opposite three outcomes. V2 changes
those three assertions, retaining depth, write-mask, interpolation, reversed
winding, and no-depth-pass checks.

Reference execution exposed another defect **inside this same fixture**:
creating a DSV from an RGBA8 resource and expecting subsequent Close to fail
crashes the reference before the cull block. [CreateDepthStencilView](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-createdepthstencilview)
is void, takes a depth-stencil resource, and cannot infer a default DSV for
every resource. V2 replaces this invalid API use and implementation-specific
Close assumption with a descriptor check on the actual valid D32 resource.
Both implementations then execute and pass the entire depth/cull suite.

### G6: geometry pass-through, four points and exact pixels

V1's VS exports POSITION at register 1, while the declared PS input expects
register 0; the GS exports register 0. A single position-dependent PS cannot
match both layouts. V2 adds `solid_link.dxil`, a constant-blue PS with **no
inputs**, retaining the original VS, GS, four-point count and complete image
comparison. The [graphics PSO contract](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_graphics_pipeline_state_desc#remarks)
requires correct stage linkage. The pinned [reference linkage validator](https://github.com/HansKristian-Work/vkd3d-proton/blob/31d1f89ca5b3f2fd3b6f025c8e6f73ed3eaa852b/libs/vkd3d/state.c#L5192)
rejects the original declaration even though the PS does not load POSITION.

The test also omitted [OMSetRenderTargets](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-omsetrendertargets).
Clearing an RTV does not bind it. V2 binds the RTV on every draw. This is a
setup correction: the original nonzero-point and exact-pixel oracles remain.

### G7: quad tessellation, exactly 4096 pixels

The HS declares four outer and two inner patch factors; the v1 DS declares
none. V2 declares the matching six factors and the **four input control
points** in the DS's shader metadata. Its bilinear position calculation,
constant-factor HS and 4096-pixel quad oracle remain unchanged. See the same
[stage-linkage contract](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_graphics_pipeline_state_desc#remarks),
[OutputPatch control-point count](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-outputpatch),
and [reference validator](https://github.com/HansKristian-Work/vkd3d-proton/blob/31d1f89ca5b3f2fd3b6f025c8e6f73ed3eaa852b/libs/vkd3d/state.c#L5192).
It also binds the previously unbound RTV, as required for colour output.
The unused patch declarations survive generation: verify.py compares the six
binary PSG1 elements, including semantic, index, register, mask and type.

### G8: mesh quad through a legal PSO stream

[Microsoft's stream ABI](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_pipeline_state_stream_desc#remarks)
requires individually pointer-aligned **enum/payload pairs** with natural
member alignment. It explicitly warns against padding every tag to a pointer.
V1 over-aligns every payload: rasterizer and RT-format data start at offset 8
instead of 4, shifting the next parsed tag into zero padding and creating an
apparent duplicate ROOT_SIGNATURE. V2 aligns each whole struct and the stream
buffer, leaving payload alignment natural. On these aarch64 machines the
rasterizer pair is 48 bytes, the RT-format pair 40, and the G8 stream 152.
The original MS/PS and full-screen coverage oracle remain. V2 binds the RTV.

### G9: amplification launches the mesh quad

The same [stream contract](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_pipeline_state_stream_desc#remarks)
applies with the additional AS pair: v1 192 bytes becomes v2 176. V2 fixes
pair/storage alignment and binds the target. The AS/MS/PS bytes, dispatch and
full-target pixel oracle remain unchanged. Both corrected streams reach
reference Vulkan pipeline compilation, rather than duplicate-subobject
rejection. This does **not** claim successful mesh execution on these runtimes.

## Reproduction

Run on a Linux build/reference machine, never the Neo. From this directory:

```sh
python3 verify.py --git-origin
python3 generate.py /path/to/mesa-build /new/shader-build
cmp /new/shader-build/solid_link.dxil v2/shaders/solid_link.dxil
cmp /new/shader-build/tess_ds.dxil v2/shaders/tess_ds.dxil
python3 build.py /new/binaries --mesa /path/to/libd3d12agx.so --proton /path/to/proton-build
bash run.sh v1 /new/binaries/mesa/v1/bin /path/to/mesa-library-dir /new/mesa-v1-results
bash run.sh v2 /new/binaries/mesa/v2/bin /path/to/mesa-library-dir /new/mesa-v2-results
bash run.sh v1 /new/binaries/proton/v1/bin /path/to/proton-d3d12:/path/to/proton-d3d12core /new/proton-v1-results
bash run.sh v2 /new/binaries/proton/v2/bin /path/to/proton-d3d12:/path/to/proton-d3d12core /new/proton-v2-results
```

`generate.py` only compiles the existing generator against an already built
Mesa's libraries, writing to a new directory. It generates only solid_link
and tess_ds, not other shaders. Generation was reproduced with Mesa base
`6a05a2d89225355d666d00b9f61152f8b9892240` on M1 Pro, GCC 16.2.1.
The two emitted blobs are byte-identical to v2's committed inputs.

`build.py` uses DirectX-Headers and records every compile/link command.
`proton-abi.h` is a compile-only binding for WIDL's explicit aggregate return
pointer, used equally for v1 and v2. Mesa binaries use the frozen native ABI
header. Neither runtime nor fixture assertions are patched by that binding.
run.sh returns the mandatory runner's exit code and also retains it in RC.txt.
The pinned native Proton d3d12 and d3d12core libraries are unchanged t-1092
artifacts from commit `31d1f89ca5b3f2fd3b6f025c8e6f73ed3eaa852b`.
The read-only MIT signature dumper is reused from t-1092, commit
`4de95ba4e76d6c04cf5f3dc75eca8239097d811a`, with attribution retained here.
No LGPL/GPL implementation was copied into these MIT fixtures.

## Measured validation and limits (2026-10-06)

| Implementation | M1 Pro v1 | M1 Pro v2 | M2 v1 | M2 v2 |
|---|---:|---:|---:|---:|
| d3d12agx, assigned Mesa base | 11/17 | 15/17 | 11/17 | 15/17 |
| vkd3d-proton, pinned reference | 9/17 | 13/17 | 9/17 | 13/17 |

The eight qualification runs attempted all 136 required cells. G1, G5 depth, G6 and G7 pass v2 on both implementations
and machines. G6 draws four points and compares all pixels; G7 covers 4096
pixels with zero inside/outside mismatches. G8/G9 fail in d3d12agx after stream
parsing with a shader-stage refusal; the reference reaches stock Honeykrisp's
AGX compiler and crashes. GDB confirms that location for both reference
stages on M1 Pro. These remain failures out of 17.

The unchanged G5 descriptor and G10 fixtures fail on the reference. Their
assertions and shader inputs are outside the six-fixture correction scope;
audit/classification is follow-up work. These observations prevent treating
this corpus as a complete universal D3D12 conformance certificate. No native
Windows run was performed. The corrected linkage/stream setup is supported
by public contracts and unmodified reference behaviour, with successful
execution proved for G1/G5/G6/G7.

Full commands, per-suite stdout/stderr/rc, hashes, kernel windows, build
receipts and GDB traces are in the t-1218 run report and evidence directory.
Only clean-room public-source and Linux measurements were used.
