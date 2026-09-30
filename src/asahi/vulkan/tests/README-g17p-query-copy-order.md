# Waiting query-copy ordering on G17P

`g17p_query_copy_order.c` exercises one command buffer containing a GPU query
reset, an empty fragment-invocation query, and a waiting GPU copy of its result
and availability. It expects zero invocations and availability equal to one.
The transfer-to-host barrier and fence make the coherent readback observable.
This is a hardware regression probe, not a no-submit test. Select a single ICD:

```sh
cc -O2 -Wall -Wextra -Werror g17p_query_copy_order.c -o query-copy-order -lvulkan
VK_DRIVER_FILES=/path/to/asahi.json timeout -k 2 15 ./query-copy-order
```

On MacBook Neo, Mesa `10081cd` loses the device during the five-second fence
wait; the kernel reports an asynchronous compute timeout. The general compute
stream contains a waiting query-copy shader before the deferred post-graphics
availability write can execute. Ending the producing graphics and compute
streams before recording a waiting copy prevents this cycle.

The same probe passes with the ordering fix and on host lavapipe, returning
`result=0 available=1`. The full frozen Factorio Linux X11 replay also completes
with three 3840x2160 readbacks, without a GPU fault; each differs from the M4
reference by at most one RGB code value. This does not qualify unrelated query
features, other GPU generations, or the Proton/DX11 capture's missing features.

The broken driver can leave the GPU session unusable; reboot before subsequent
hardware tests. The probe is deliberately not registered as an automatic host
unit test.

When integrating with the explicit dependency tracking introduced in `8950045`,
finish the producers before calling `hk_query_command_cs`. That helper attaches
dependencies to the consumer stream; ending that stream afterwards would discard
the intended association. The hardware observations above describe the original
`10081cd`-based build, not a rerun of the full corpus on the newer upstream base.
