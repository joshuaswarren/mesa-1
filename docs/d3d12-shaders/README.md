# Shader measurement runner

`run-v5.py` derives from the MIT t-1083 scoreboard runner. It preserves its
per-registration accounting and adds repeatable `--env NAME=VALUE` options.
Every supplied value is recorded in `identity.json` and the individual
command files; ambient driver opt-ins are still discarded.

Extract the shader selection from the pinned t-1083 inventory:

```python
import json
rows = json.load(open("inventory.json"))
json.dump([r for r in rows if r["corpus"] == "vkd3d-proton-tests"
           and r["category"] == "d3d12_shaders"],
          open("shaders.json", "w"), indent=2)
```

Build the native A adapter and B runner from the same Proton checkout with
the t-1083 adapter. A uses Mesa's native aggregate-return COM ABI; B uses its
unmodified generated ABI. Do not change assertions, shaders or capability
responses. Preserve the adapter's ABI audit and build commands.

Example on an authorized reference Mac (replace paths and source revision):

```sh
python3 run-v5.py --inventory shaders.json --corpus vkd3d-proton-tests \
  --machine m1pro --driver d3d12agx@REV --binary ./bin/d3d12-agx \
  --src ./src --out ./raw/A --timeout 12 --library-path ./lib
python3 run-v5.py --inventory shaders.json --corpus vkd3d-proton-tests \
  --machine m1pro --driver vkd3d-proton@REV --binary ./bin/d3d12 \
  --src ./src --out ./raw/B --timeout 12 --library-path ./lib \
  --icd /usr/share/vulkan/icd.d/asahi_icd.aarch64.json
```

Package **both** Proton's `libvkd3d-proton-d3d12.so` and its dynamically
loaded `libvkd3d-proton-d3d12core.so`. Inspect stderr for failed device
discovery before interpreting corpus-wide skips as feature coverage.

The Neo only runs through `aurora-ctl hw request` with `--task`, short
exclusive GPU requests, and off-target builds. Its currently qualified
native graphics path requires:

```text
--env AURORA_D3D12_PHYSICAL_BASELINE=1
--env AURORA_D3D12_PHYSICAL_D32=1
```

Use those controls for both baseline and candidate A runs. Record B's exact
Honeykrisp ICD/prefix separately; a distro reference that cannot create a
device is a blocked comparison, not 63 hardware-feature skips. Output
directories cannot be reused. Budget interruptions are not test verdicts.
