# Native AGX D3D12 capability policy

`d12_core_caps_init` derives a separate immutable policy for each opened
adapter. `d12_core_caps_init_agx` copies Linux UAPI generation, active core masks,
resource allocator bounds and VM identity through existing AGX helpers, and describes the
connected compiler paths. `d12_core_caps_query` translates that policy into the
public structures. Device creation and feature negotiation use the same policy;
a null output pointer still probes the actual adapter.

The source-only derivation test accepts synthetic G13/G14/G17P adapters, missing
compiler paths and invalid VM ranges. It verifies isolation between devices and
separates device-creation enum validation from FEATURE_LEVELS request/reply
behavior. The live `d3d12agx_caps_queries` test checks all 50 feature structures
for wrong sizes and null pointers, as well as VA consistency, version negotiation,
node selection and queue priority. `d3d12agx_caps_probe LIBRARY` dumps all 50
structures, SM/root/FL requests, formats, MSAA counts and device interfaces as
JSON lines. Failed HRESULTs mean output fields are undefined.

FORMAT_INFO also reports public two/three-plane video layouts accurately even
when allocation is unsupported; the values were independently checked against
Microsoft's MIT DirectX-Headers property-format table on Linux.

## Contracts and evidence

| Policy | Implementation / qualification |
| --- | --- |
| VM bits | Bit length of the highest resource VA in the UAPI-derived main VMA heap; OPTIONS and dedicated VA query agree. Internal USC/kernel ranges are excluded. This is the public allocation window, not a guessed silicon address width. |
| UMA / coherence | One BO allocation kind. Native Linux DRM coherent mappings have a no-op cache maintenance hook; virtio/Windows do not inherit that proof. The D3D12 TileBasedRenderer efficiency promise remains false, independently of the physical tile buffer. |
| Wave width | Connected AGX2 wave lowering, 32 lanes. G17P full WaveOps is not qualified. TotalLaneCount is one executable wave per active core, a conservative lower bound; Proton's 1024 fallback is an explicitly guessed occupancy, not a UAPI fact. |
| Shader model | G13/G14 SM6.0; G17P SM5.1. A G17P SM6.0 convergence shader reaches a refused native container, so complete SM6.0 is withheld. Individual native f16/i64 operations do not justify a complete SM6.2 contract. AGX2 compiler paths provide those operations; G17P full contracts remain off. |
| Binding / FL | Tier 1, FL11_0. The 1024 distinct native sampler limit cannot guarantee Tier 2's 2048 samplers. FL12_0/12_1/12_2 are withheld. The inherited missing 8x MSAA baseline requirement remains an unresolved conformance gap even at FL11_0. |
| Sparse | Tier 0. SM5 texture feedback cannot compile; feedback/default-mapping/lifetime witnesses fail. Existing bind/mapping code does not establish a tier. |
| Mesh / RT / VRS / feedback / work graphs / enhanced barriers | Off. Compiler acceptance or exposed COM methods are not complete execution contracts. BARRIER_LAYOUT reports unsupported consistently with OPTIONS12. |
| Root signature | 1.2 requests succeed and clamp to proven 1.1. Integer static samplers are not qualified; the non-normalized static-sampler witness fails. |
| Modern stencil / dynamic depth bias / predication | Off. Modern stencil PSO stream support is missing; ExecuteIndirect predication fails; dynamic bias is not fully qualified. |
| Render pass | G13/G14 Tier 1; upstream suspend/resume variants execute successfully. G17P is not qualified. |
| Sample positions | Off on every generation. The G13/G14 PPP_MULTISAMPLECTL writer and host encoder tests need a dedicated per-device pixel/depth-resolve witness before advertisement. |
| Descriptor limits | 1024 distinct samplers, 1000000 rooted views. The view limit is an implementation bound, not a new million-entry stress result. |
| Formats / MSAA | No-output sample mask 1 until its separate coverage contract is qualified ([public specification](https://microsoft.github.io/DirectX-Specs/d3d/VulkanOn12.html)). Existing attachment format table, filtered by the connected device path. G17P withholds full texture sampling, additional typed UAV loads and multisample support; its bounded Texture.Load path is not a general sampler contract. |
| Queue / cache / protection | Normal DIRECT/COMPUTE/COPY priority only; single-PSO cache path only; no protection, cross-node, existing-address heaps, displayable textures, meta commands or tight placement. OPTIONS21 uses the official ExecuteIndirect 1.0 value 10. |

The WDDM adapter currently lacks an opened-device discovery seam and retains a
conservative allocation policy through `d12_core_get_caps`. Native COM answers do
not use that fallback. WDDM per-adapter qualification is a separate integration
requirement; this change does not establish Windows runtime conformance.

The full run report contains exact commands, library/source hashes, frozen input
receipts, paired outcomes and failures. Skips, TODOs and crashes are not passes.
