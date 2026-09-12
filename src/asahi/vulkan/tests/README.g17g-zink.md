# M5 Zink rendering probe

`g17g_zink_draw.c` exercises a 256x256 surfaceless OpenGL triangle through
Zink and Honeykrisp. It rejects other renderer names and uses no depth,
stencil, textures, or fragment varyings. Build it against EGL and OpenGL:

```sh
cc -Wall -Wextra -O2 g17g_zink_draw.c -lEGL -lGL -o g17g_zink_draw
```

Select the experimental M5 ICD and matching Mesa libraries when running it.
Set `ASAHI_M5_BRINGUP=1`, `MESA_LOADER_DRIVER_OVERRIDE=zink`, and
`GALLIUM_DRIVER=zink`. The default path checks all 65,536 RGBA pixels returned
by `glReadPixels` against `ff470dff`. `--submit-only` deliberately skips that
check and must not be interpreted as rendering proof.

`AGX_GL_DEFER_READBACK=1` retains the live context after `glFinish`; send SIGUSR1
to the reported PID to request readback. `AGX_GL_HOLD=1` also retains resources
after that check. These modes let an external diagnostic verify queue
completion and read the actual render target while its allocations are live.
Do not access GPU MMIO as part of that RAM inspection.

On M5 hardware on 2026-09-12, both graphics queues retired and all 65,536
render-target pixels independently matched `0d47ffff` in the selected BGRA
surface. This exercised Mesa-generated shaders, the AGX3c robust-fetch integer
select, position plus point-size UVS layout, and the inverted full-frame Zink
viewport. No native reference shader was substituted.

OpenGL readback did **not** pass: the tile-load and meta image-copy shader
paths were rejected before submission. Zink can report these Vulkan failures
while `glFinish` and `glGetError` appear successful, so always inspect the
compiler/submission log and verify the pixels independently. The rendering
result does not establish general OpenGL support, glxgears, or display-server
presentation. Depth, fragment varyings, transfers, sustained queue reuse, and
the combined GPU/DCP desktop remain separate validation work.
