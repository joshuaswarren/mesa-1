# Bounded physical G17P Texture.Load

The native compute path accepts indexed, nonarray, nonshadow 2D `txf` on
G17P with native32 descriptors. Application samplers, images, indirect or
predicated dispatch, preambles and unsupported scratch contracts retain their
existing refusals. This is bounded producer qualification, not a feature-level
conformance claim.

Physical admission requires `AURORA_D3D12_PHYSICAL_BASELINE=1`, a resident native
context, `DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS`, and an actual Asahi render fd.
The existing fd check rejects registered noop fds before checking the emulated
DRM identity. Metadata admission still requires the registered noop callback
and `AURORA_D3D12_SHIM_TEXTURE_METADATA=1`. Each segment records which admission
it used, and queue submission rechecks that same admission before ioctl.

The existing producers supply the descriptor table, real 32-byte txf sampler,
texture pointer halves 0–3, sampler 4–7, root 8–11, and query pointer 16–19.
Compiler-owned CODE, staging and SPECTPR metadata go through the shared native
Ioto/CDM composer. Stream ownership, bounds, descriptor refresh, read tracking,
BO retention and cache publication remain mandatory.

## Reproduction

All Neo operations must use `aurora-ctl hw request` with the assigned task ID.
Build only in a new task directory, in a transient unit with
`AllowedCPUs=1-4`, `CPUQuota=200%`, `Nice=10`, and Ninja `-j2`. Replay the P00
pinned configuration, overlay the 63 frozen ba4c blobs, preserve a complete
source/build receipt, and publish a new immutable private prefix. Never modify
the P00 prefix, the desktop Mesa prefix or Steam.

With a sealed candidate and matched kernel receipt, the exact strict test is:

```sh
aurora-ctl hw request execute --device neo \
  --task t-1087-d3d12-phase-2-p03-physical-texture-load-on-g17p- \
  --mode exclusive --timeout 60m -- \
  "D3D12AGX_DUMP_TEX=1 bash /var/tmp/aurora-d3d12-phase2-kit/verify-reference.sh /opt/aurora-d3d12-phase2/CANDIDATE CANDIDATE /var/tmp/aurora-d3d12-phase2/UNIQUE-P03-g5_texture g5_texture"
```

Replace both `CANDIDATE` tokens with the full driver commit and `UNIQUE` with a
single-use run name. Run the same harness with selection `all` to retain all
17 original results. Append dashboard stats after each test batch and export
its raw evidence through separate queued packing/download requests.

`g5_texture_native_metadata` has four supplemental modes. The original
`--record-only` and `--shim-submit-only` modes still require the validated noop
shim. `--shim-record-revoke` records with metadata admission, unsets its flag,
and requires `DXGI_ERROR_DEVICE_REMOVED` at queue admission. Under real-fd
baseline admission, `--physical-record-revoke` performs the corresponding
physical-flag revocation. Both revocation modes deliberately submit no GPU
work and grade no pixels. They do not contribute to the frozen17 numerator.

Also run the existing `g17p_txf_lower_tests` and
`g17p_txf_prefix_core_tests`. Negative recording controls must reject a physical
flag on the registered noop fd and a real fd without physical admission.
Keep their expected failures separate from the original acceptance verdict.
