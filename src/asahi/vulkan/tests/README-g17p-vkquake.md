# Driving vkQuake as a G17P test client

vkQuake is a useful G17P workload because it is Vulkan-native -- it does not go
through Zink -- and because it exercises indirect draws, compute-generated
index buffers, lightmap compute, secondary command buffers and a multi-subpass
main render pass in one process.

Almost every knob worth turning is a cvar. Read this before running one, or
your experiment will silently measure nothing.

## `+commands` are silently ignored on the shareware data

**vkQuake only populates the `cmdline` cvar on the registered path.** The
`Cvar_SetROM ("cmdline", ...)` call lives inside `COM_CheckRegistered()`,
immediately before `Con_Printf ("Playing registered version.\n")`. With the
shareware `pak0.pak` the function takes the other branch, prints

```
Playing shareware version.
```

and never sets `cmdline` at all. `Cmd_StuffCmds_f()` is "rewritten to read the
`cmdline` cvar", so it then reads an empty string and executes nothing.

The consequence is that **every `+command` on the command line is dropped, with
no error and no log line**. The command line itself still looks right --
vkQuake echoes it at startup:

```
Command line: .../vkquake -basedir ... +r_fullbright 1 +map start
```

-- so `+r_fullbright 1`, `+r_indirect 0`, `+r_showtris 1`, `+map start` and the
rest all appear to have been passed, and all do nothing. Runs configured this
way produce clean-looking negative results that mean nothing whatsoever. This
cost a full round of G17P world-surface experiments before it was noticed.

### Use `autoexec.cfg`

`quake.rc` execs `autoexec.cfg`, and a loose `autoexec.cfg` in the game
directory does run. Write the cvars there instead:

```sh
cat > "$BASEDIR/id1/autoexec.cfg" <<'CFG'
r_indirect 0
scr_showfps 1
map e1m1
echo AUTOEXEC_RAN
CFG
```

**Always include a marker line like that `echo`**, and confirm it reaches
`qconsole.log` (run with `-condebug`; the file lands in
`~/.local/share/vkQuake/qconsole.log`). Without the marker there is no way to
distinguish "the cvar applied and changed nothing" from "the cvar never ran",
and those two readings support opposite conclusions.

Prefer a cvar whose effect is visible when validating that the config applied.
`r_fullbright 1` and `r_lightmap 1` both change how entity models are shaded,
so a screenshot shows immediately whether they took. `viewsize` is a poor
choice: `viewsize_allow_shrinking` defaults to `0`, so a smaller value is
ignored and looks exactly like a config that never ran.

### `map` in `autoexec.cfg` beats `startdemos`

`quake.rc` ends with `startdemos demo1 demo2 demo3`, which is why a run that
asked for a map ends up playing the demo loop instead. `Cmd_Exec_f` inserts the
exec'd file at the *front* of the command buffer, so an `autoexec.cfg` that
issues `map ...` runs first and leaves `sv.active` true; `startdemos` then
declines to start a demo. Putting the map there is therefore enough, and no
`-nodemos` equivalent is needed.

Demo playback is still fine for rendering work -- `demo1` walks e1m3 -- but the
map has to be chosen deliberately rather than assumed.

## Cvars worth knowing for G17P work

| cvar | effect | why it matters |
| --- | --- | --- |
| `r_indirect 0` | world drawn with ordinary indexed draws instead of `vkCmdDrawIndexedIndirect` with compute-generated counts | isolates the indirect path, which is how the world is drawn by default |
| `r_gpulightmapupdate 0` | lightmaps uploaded from the CPU instead of built by compute | separates lightmap *content* from lightmap *sampling* |
| `r_fullbright 1` | binds a grey lightmap to set 1 | world still black under it means the failure is not the lightmap |
| `r_lightmap 1` | binds a grey diffuse to set 0 | world still black under it means the failure is not the world textures |
| `r_showtris 1` | wireframe over world and entities | distinguishes "drawn and shaded black" from "never rasterized" |
| `r_tasks 0` | single-threaded command-buffer recording | isolates concurrent recording |
| `scr_showfps 1` | on-screen frame rate | note the kernel logs heavily per command, so absolute numbers are depressed; compare relatively |

`r_fullbright` and `r_lightmap` are cheat-protected through
`r_fullbright_cheatsafe` / `r_lightmap_cheatsafe`, which require
`cl.maxclients == 1`. Single-player maps and the shipped single-player demos
both satisfy that.

Note that `indirect` is computed per frame as

```c
indirect = r_indirect.value && indirect_ready && r_gpulightmapupdate.value &&
           !scr_speeds.value;
```

so `r_gpulightmapupdate 0` and `scr_speeds 1` *also* disable indirect drawing.
An experiment that means to vary only the lightmap source, or only to read the
poly counters, changes the draw path at the same time.

## What the frame looks like from the driver's side

- Everything is recorded into **secondary** command buffers and replayed with
  one `vkCmdExecuteCommands` call each, consecutively, inside a render pass
  begun with `VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS`
  (`R_SubmitContexts`). The world is six of them, entities another six, then
  one each for sky, view model, particles, water, the GUI and post-process.
- World surfaces come from one large device-local vertex buffer with a 28-byte
  stride and three attributes at +0 (`vec3` position), +12 and +20 (two `vec2`
  texture coordinates); the second set is the lightmap coordinate.
- The world fragment shader samples three textures from three descriptor sets
  -- set 0 diffuse, set 1 lightmap, set 2 fullbright -- and multiplies the
  first two. A second sampled texture that returned zero would black every
  world surface while leaving models correct, because a model's second texture
  is only added.
- Large buffers are uploaded by `R_StagingUploadBuffer`, which chunks by
  `staging_buffer_size` and issues a `vkCmdCopyBuffer` per chunk with non-zero
  `srcOffset` and `dstOffset`. Alias models use the same path at smaller sizes.
