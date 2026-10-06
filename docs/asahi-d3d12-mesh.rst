D3D12 mesh and amplification lowering
====================================

The D3D12 bridge implements mesh shaders inside the existing Asahi pipeline
and command recording paths. Mesh and amplification DXIL containers retain
their shader kinds (13 and 14) for API validation. Their application programs
are translated to compute NIR; they are never passed to vertex-bytecode
validation.

A mesh pipeline shares the ordinary graphics pipeline's pixel shader,
tilebuffer and fixed-function state. Its layout placeholder vertex shader is
unused by draws. A separate rasterization vertex program reads the mesh's
index, vertex and primitive records. Mesh/pixel linking uses signature
semantics and component declarations, including primitive attributes, rather
than assuming matching signature row numbers.

Recording owns and retains the output BOs. Each mesh workgroup has separate
count and output records sized from the shader declarations. Setup writes a
non-indexed indirect draw descriptor; the rasterization program expands the
mesh indices and suppresses primitives outside the workgroup's actual count.
An amplification program publishes a payload and child dimensions. A setup
program constructs a child-to-parent map and a six-word indirect-local grid.
The child mesh program uses that map to select its payload and local group ID.
The payload handoff includes workgroup synchronization and inter-dispatch
barriers.

G13/G14 use the existing legacy launch path. G17P uses the existing native
CODE/Ioto producers, retained metadata, recording-owned uniform cells and
native indirect draw/state encoders. Its mesh raster path admits the bounded
root-only triangle subset; texture and other unsupported source-cell ABIs
remain closed. Small function-local payload temporaries are promoted to SSA;
programs still requiring an unsupported scratch ABI are refused. Compute
quad derivatives use existing quad broadcasts and subtraction on G17P. The
2D invocation remapping runs before that lowering; 1D derivative groups retain
linear quad order. Legacy derivative instructions are unchanged on G13/G14.

Qualification boundary
----------------------

The public mesh tier stays withdrawn. The current storage model bounds direct
and per-parent child mesh groups to 256, amplification groups to 64, and total
children to 16,384. Payload declarations are bounded to 16 KiB and output
declarations to 256 vertices/primitives. This is not the full D3D12 dispatch
range contract. Advertising Tier 1 also requires a coherent maximum dispatch
query; forcing the tier alone is not sufficient.

Diagnostic qualification can override OPTIONS7's mesh tier and OPTIONS9's
mesh/amplification derivative bit in a private library to exercise all six
unmodified upstream registrations. Such an override is not production
capability exposure. Full strict regression runs use the public queries.
Frozen17 v2 G8/G9 additionally check complete pixel coverage and valid pipeline
streams with bound render targets. No shim result is evidence of GPU pixels.

Provenance
----------

The initial lowering was adapted from Aurora Silicon's MIT August work
(``af56cd3f963``, ``b446904d360``, ``bfd3c9266cc``), with independent fixes for
output bounds, semantic linking, owned buffers, payload lifetime, invocation
layout and indirect launch shape. The external vkd3d-proton tests are run
unchanged; their implementation is not incorporated into the MIT driver.
The mesh/payload and derivative contracts follow the public DirectX
specifications:

* https://github.com/microsoft/DirectX-Specs/blob/master/d3d/MeshShader.md
* https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_Derivatives.html
