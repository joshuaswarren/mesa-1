# Deadzone startup: placed heaps and capability negotiation

## Placed-buffer rejection

The Linux/Wine startup witness requests a DEFAULT heap with flags `0x10c0`
and a 512-byte UAV buffer. The flags combine ALLOW_ONLY_BUFFERS and
CREATE_NOT_ZEROED. Those are compatible: the latter permits skipping a clear,
and does not require dirty memory. See Microsoft's
[heap flags](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_heap_flags).

The rejection occurs before heap/resource validation when `D3D12AGX_DEBUG`
contains `api` or `api-fail`. Tracing installs a forwarding vtable on each
object; the heap ownership check recognized only the original vtable. The
trace-aware identity check accepts that registered wrapper and its exact
original implementation. It still rejects foreign objects before downcasting.
Resources and shared fences use the same check.

CREATE_NOT_ZEROED, buffer-only heap flags and heap tier 2 already work on the
base. No flag is stripped, no alignment requirement is relaxed and tracing
does not need to be disabled. Explicit buffer Alignment is restricted to 0
or 64 KiB, as specified by Microsoft's
[resource description](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_resource_desc).

## Tests

The CPU unit suite tests all traced interface identities and rejects shader
model enum holes. Heap descriptor tests cover 480 combinations: seven usable
resource-class masks, every subset of CREATE_NOT_ZEROED, CREATE_NOT_RESIDENT,
ALLOW_SHADER_ATOMICS and TOOLS_USE_MANUAL_WRITE_TRACKING, process-sharing
variants, and the cross-adapter combination on DEFAULT and non-CPU-accessible
CUSTOM heaps. Display, protection and write-watch are explicitly unsupported.
Cross-adapter combinations follow Microsoft's
[shared heap restrictions](https://learn.microsoft.com/en-us/windows/win32/direct3d12/shared-heaps).

The manual `d3d12agx_placed_resource_probe` target creates no queue and submits
no GPU command. It tests 240 DEFAULT heap combinations, buffer placements at
0 and 64 KiB, misalignment, bounds/overflow, texture class restrictions,
foreign objects, feature queries and cross-adapter buffer allocation. Allocation
success does not qualify shared-handle export/import.

Run the CPU suite and the allocation probe separately:

```sh
meson test -C build --suite d3d12 --print-errorlogs
ninja -C build src/asahi/d3d12/d3d12agx_placed_resource_probe
for mode in '' api api-fail; do
    D3D12AGX_DEBUG="$mode" build/src/asahi/d3d12/d3d12agx_placed_resource_probe
done
```

The `--repro` option checks just the 512-byte UAV in a `0x10c0` heap. With the
original driver it fails under tracing; the corrected driver passes.

## SM6 decline

G17P reports shader model 5.1, no wave ops, resource binding tier 1 and resource
heap tier 2. The game consequently falls back to SM5. These answers follow the
implemented backend, rather than the hardware's age. The G17P DXIL producer
has not qualified the complete SM6 compiler/wave contract; the native sampler
table supports 1024 distinct states, below the 2048 required by binding tier 2.
Deduplicating sampler states cannot guarantee 2048 distinct states.

Valid shader model requests clamp to the implemented version. Unknown enum
values, including the gap between 5.1 and 6.0, return E_INVALIDARG without
mutating the caller's query. G13/G14/G17P query tests exercise all byte-sized
version values and malformed query sizes. Higher capabilities require backend
implementation and qualification before changing these answers.

## Blocker chain

1. Windows ordinal loading: experimental tool dependency; repaired separately.
2. Wine Unix fd teardown: fixed in the fd-ownership prerequisite.
3. Traced placed-heap identity: fixed here, with native and x64 Wine allocation
   witnesses. The game advances to shader-library and graphics PSO creation.
4. Graphics PSO creation returns E_INVALIDARG. A startup attempt also reaches
   the 500 MB available-memory guard. That stopped attempt does not establish
   rendering, a main menu, or a successful GPU session.

Keep game attempts bounded and retain complete kernel journals and queue-slot
snapshots. Stop at the first new quarantine, poison, submit failure or memory
guard condition. Wine `+debugstr` captures Mesa's OutputDebugString diagnostics;
console redirection alone can lose them when the game changes its console.
