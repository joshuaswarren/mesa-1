# Helper texture and multisample readback regressions

The compiler test `asahi-g17p-helper-samples` checks integer samples whose
values reach each derivative intrinsic, including long ALU chains and
loop-carried phis. Coordinate derivatives alone leave ordinary samples on
the implicit path. The G17P lowering uses explicit coordinate gradients for
the sampled values required in helper lanes.

For runtime coverage, use the Vulkan CTS repeat runner:

```sh
python3 src/asahi/vulkan/tests/run-helper-msaa-cts.py \
  --deqp /path/to/deqp-vk --icd /path/to/asahi_icd.json \
  --output /path/to/new/results --iterations 200
```

It starts a fresh process on every iteration, disables Mesa shader caching,
requires a complete result for each of the five cases, and returns success
only when every result is Pass. It preserves QPA logs and a per-case count in
`summary.json`. For a shared test machine, split the repetitions into bounded
sessions and add their counts only when each session passes the hardware
health checks. Run the affected `glsl.helper_invocations`, discard/demote,
and `pipeline.*.multisample_shader_builtin` groups as well.

On the Neo, invoke this runner through the approved hardware-request queue,
under the session's thermal and kernel-journal guard. Record queue slots
before and after, require at least 40 free slots, stop on the first queue
quarantine or poison, and reject results from a boot with submit ENOSPC or
an increased submit-failure count. Build the driver elsewhere.
