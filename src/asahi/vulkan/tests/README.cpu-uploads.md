# Borealis CPU upload benchmark

Build Mesa with `-Dvulkan-drivers=asahi -Dbuild-tests=true`. The `asahi` test
suite includes `hk_cpu_upload`, a host-only correctness test that also runs a
small upload benchmark. It links the production descriptor-binding and upload
code. No Vulkan or DRM device is created and nothing is submitted.

```
meson test -C build --suite asahi --print-errorlogs
ASAHI_MESA_DEBUG=perf build/src/asahi/vulkan/hk-cpu-upload-bench 1000
```

The numeric argument is the number of simulated command buffers, each with
16 draws. The fixture enables the perf counters itself. Three workloads keep
descriptor offsets, draw parameters and draw IDs identical, change them every
four draws, or change them every draw. Both shader constant payloads stay
fixed. The fixture checks changed dynamic-layout indexing, push-descriptor
replacement, direct/indirect transitions, immutable earlier uploads, shader
payload identity, conservative descriptor paths and cache reset.

Preallocated host storage replaces the command-buffer uploader's mapped BOs.
Small batches avoid its allocation slow path. This measures the targeted
recording helpers, including a state reset per batch; it does not measure
full draw-state construction, shader compilation, submission or GPU work.
Elapsed `ns_per_draw` is not a game performance prediction.

On a real driver, `ASAHI_MESA_DEBUG=perf` prints a `CPU uploads:` line at the
end of each command buffer:

- `draws`: recorded draw calls, including internal meta and expanded multiview draws.
- `descriptor_binds` / `descriptor_noops`: calls per bind point and unchanged
- `pipe_bind` / `desc_bind_ns` / `pool_alloc`: compute-path cost counters (ns,
  AGX_DBG_PERF only) for pipeline bind, compute descriptor bind, main-pool alloc
  binds on the admitted native graphics path. Combined sampler, sample-variant,
  geometry/tessellation and compute descriptor paths stay conservative.
- `root`, `draw_params`, `draw_id`, `constants`: successful upload count and
  copied payload bytes. Root bytes exclude unused dynamic-buffer slots;
  allocator padding and cache-line clean traffic are not included.
- Corresponding `*_reuse` counters: successful cached returns.

Only these paths are counted; total driver allocation and copy traffic can be
higher. Sum counters across command buffers before dividing by draws. Perf
logging can affect CPU-bound games, so collect counters separately from fps.

Cached storage belongs to the command buffer and is immutable until reset.
Root reuse preserves descriptor-generation invalidation and compares only
previously uploaded bytes, direct draw parameters
require a valid host value, and indirect parameters remain execution-time GPU
addresses. Constants use a process-unique immutable payload identity instead
of a recyclable object address. The last payload per shader stage is cached;
a change of payload uploads a new copy.
