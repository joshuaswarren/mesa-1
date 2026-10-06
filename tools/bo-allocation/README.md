# Vulkan BO allocation diagnostics

Build on an aarch64 Linux builder, using the Mesa headers for the ioctl shim:

```sh
cc -O2 -Wall -Wextra tools/bo-allocation/vk-alloc.c -lvulkan -o vk-alloc
cc -shared -fPIC -O2 -Wall -Wextra -Iinclude/drm-uapi \
  tools/bo-allocation/ioctl-timing.c -ldl -pthread -o ioctl-timing.so
```

Select one Asahi ICD with `VK_DRIVER_FILES`. The probe never submits GPU work.
Its CSV reports allocation/free wall time and process CPU time separately.
Arguments are repetition count, memory type index, byte size, and optional
`pressure` mode:

```sh
# Each iteration creates a fresh device, outside the measured interval.
./vk-alloc 20 0 268435456
./vk-alloc 20 1 268435456

# One device, with a small BO cached before twenty allocate/free cycles.
# This catches a capacity policy that permanently refuses the larger BO.
LD_PRELOAD="$PWD/ioctl-timing.so" ./vk-alloc 20 0 268435456 pressure
LD_PRELOAD="$PWD/ioctl-timing.so" ./vk-alloc 20 1 268435456 pressure
```

`pressure` intentionally differs from fresh-device mode: it measures whether
repeated allocations reuse backing storage when an older cached BO competes
for the cache's byte budget. Report the first (cold) iteration separately
from the remaining iterations. Without pressure, a repeated allocation can
look fast even with the defective capacity policy, because an otherwise
empty cache admits the large BO.

The shim records large GEM_CREATE and single-operation VM_BIND requests:
`bo-ioctl,kind,bytes,address,handle,flags,result,wall_seconds,thread_cpu_seconds`.
READ|WRITE binds identify the ordinary mapping; READ-only binds identify the
shadow. GEM_CREATE is lazy: backing-page allocation/pinning happens under
the first VM_BIND. Preserve that distinction in time breakdowns.

For a Neo session, run the commands through the hardware request queue and
wrap the measurement in `guard.py OUT COMMAND...`. It captures full kernel
logs, boot identity and queue slots, rejects an ENOSPC boot or fewer than 40
free slots, terminates the process group on the first quarantine, poisoned
queue or submission failure, and verifies unchanged submission-failure counts.
The wrapper bounds one session to 550 seconds. Use a systemd service with a
matching runtime bound to also constrain a hung wrapper or child.

Profile a full CTS case as well as a single allocation. In CTS 1.4.6.2,
`memory.mapping.*.full.variable.implicit_unmap[_map2]` performs 128
allocate/map/free iterations for each host-visible memory type, plus the
initial large-size probe. A multi-second aggregate cost is not the cost of
one `vkAllocateMemory` call. Capture perf with kernel symbols and ioctl
counts, and rerun the complete `memory.mapping` caselist with the normal
60-second per-case deadline.
