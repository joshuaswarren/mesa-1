# P10 registered native preraster recording

The recorder reuses Poly's parameter blocks and Honeykrisp's GS setup order:
setup (indirect only), software VS, GS main, then the render segment. Each
compute launch has the shared native flags=5 barrier. Indirect launches consume
Poly's six-word global-threads/local-size grid directly. No CPU count readback
or replacement is performed.

Compiler declarations select source cells within the recording's uploaded data.
The recorder verifies the exact CODE owner, certification, workgroup, pointer
cell alignment and Poly parameter spans. Direct and indirect draws retain
separate cells and Ioto programs. Descriptor/tess/mesh/node cells remain refused
until their pointee ownership has its own certificate. Unknown GPU heap pointers
use the existing conservative read/write/residency path. Queue submission
rechecks registered-fd admission, exact-prior-barriers and a render successor.

`d12_cmd_certify_native_gs` is a private registered-only diagnostic that creates
ordinary command-list storage and invokes the actual preraster recorder twice.
It inserts a **render placeholder**, creates no queue and submits no GPU work.
Its uploads, Ioto and CDM artifacts can be independently checked with:

```
python3 docs/d3d12-native-stage/P10/verify-recording.py CERTIFICATE/gs
```

This certificate does not prove raster context, retirement, GPU-written counts
or pixels. Physical GS admission remains closed. The native raster producer
still refuses point/generated UVS and indexed/indirect draws. Removing those
guards before their certification would bypass the accepted execution gate.

Build with `docs/d3d12-native-stage/build-native-preraster-kit.py` on a native
Linux/aarch64/16K builder, using the exported P00 pins/frozen blobs and a separately
supplied Neo kernel identity receipt. It checks compiler, options and dependency
versions, and stages all 17 exact reference executables, one library, the shim
and the certificate. It neither accesses nor compiles on the Neo. The supplied
kernel receipt must subsequently pass the Neo's `verify-reference.sh` checks.

`gs-inventory.json` contains 14 registrations in pinned vkd3d-proton
31d1f89ca5b3, including all of `d3d12_geometry_shader.c` and callers of GS-using
functions in the clip/cull, PSO and root-signature tests. `run-gs.py` is the
versioned t-1083 v8 process-isolated harness with one explicit option for the
existing physical baseline/D32 flags. It changes no shader, assertion, verdict
classification or filtering. It supplies no GS admission override. Use the same
inventory and binary adapters for A and B, with a new output directory per run.

`g6-legal-v1.json` describes a **separate input variant**: unchanged VS/GS and
frozen executable/assertions, with vkd3d-proton's input-free green PS in place of
the signature-incompatible frozen PS. Its empty ISG1 avoids the declared
SV_Position register mismatch. Materialize it from the pinned public header:

```
python3 docs/d3d12-native-stage/P10/materialize-g6-legal-v1.py \
  PROTON/tests FROZEN/shaders NEW/g6-legal-v1
```

The extracted upstream bytecode remains a private corpus artifact with its
upstream licence/provenance. It is not vendored here. This variant's results
never count as a frozen17 pass. Actual GPU results remain necessary.
