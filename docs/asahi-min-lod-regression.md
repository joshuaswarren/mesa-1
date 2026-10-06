# Asahi texture minimum-LOD regression

Image-view minimum LOD can be added after early texture lowering. In
Honeykrisp, newer sampler descriptors carry sampler LOD bias themselves,
so an implicit `tex` need not have been converted to `txb` by that early pass.
The backend accepts a minimum with packed bias or gradients, but previously
asserted when the late pass left `tex(min_lod)` intact.

The common late Asahi pass now converts this case to `txb(0, min_lod)` and
packs it immediately. This must happen after constant folding: folding a
zero bias before packing converts the instruction back to `tex`. An
explicit `txl` with a minimum is clamped with `fmax` before fp16 source
legalization. Existing `txb` and `txd` minimum operands keep their packing.

Build Mesa with `-Dbuild-tests=true -Dvulkan-drivers=asahi`, then run:

```sh
meson test -C build agx_texture_min_lod --print-errorlogs
build/src/asahi/compiler/agx_texture_min_lod_test --reproduce
```

The CPU test compiles all four operations for AGX2/G13 and AGX3B/G17P, with
no minimum, a constant minimum, and a dynamic minimum (24 cases). It also
checks packed bias/minimum values and the explicit-LOD clamp. `--reproduce`
prints NIR and compiles the original failing shape without stopping at the
structural check, so an unfixed compiler reaches the backend assertion.

Four Vulkan binaries extend the existing offscreen triangle/readback
harness, with precompiled SPIR-V:

```sh
meson compile -C build g17p-vulkan-min-lod-tex g17p-vulkan-min-lod-txb \
  g17p-vulkan-min-lod-txl g17p-vulkan-min-lod-txd
bash src/asahi/vulkan/tests/run-texture-min-lod.sh "$MESA_PREFIX" "$PROBE_DIR"
```

The runner requires an installed prefix and the four prebuilt binaries.
On the Neo it must run inside a bounded GPU-session guard through an approved
`aurora-ctl hw request execute`; build on another machine. The runner stops
at the first failing application. The outer guard must save QID counts and
kernel journals, refuse insufficient free QIDs or an ENOSPC-contaminated
boot, and stop on a new quarantine, poison, or submit failure.

The texture contains three solid mips: red, green, blue. A 64-texel texture
across 32 pixels gives implicit/gradient LOD one (green). A bias of minus
one and explicit LOD zero select red. An image-view minimum of two must
select blue for every operation. Both zero and two are tested, with exact
per-pixel comparison, upload/readback guards, and readback checksums.

`hk_image_view_min_lod=true` enables Honeykrisp's existing conformant
image-view implementation, normally enabled by the vkd3d application
profile. The test queries and requires its real feature. It changes neither
reported device identity nor driver capability implementation. Use the same
option for both ICDs when comparing the complete `dEQP-VK.texture.mipmap.*`
group, which includes sampler and image-view minimum-LOD cases.

The fragment source contains the exact header regeneration command. Run
`glslangValidator` on the build machine; no on-target C or shader-tool build
is needed.

Integer fetches use a different rule. Vulkan's
[Integer Texel Coordinate Operations](https://docs.vulkan.org/spec/latest/chapters/textures.html#textures-integer-coordinate-operations)
select `baseMipLevel + Lod`. A level below `floor(view.minLod)` is out of
bounds and returns all zero components with robustImageAccess2. The minimum
is a bounds threshold, not an offset or a clamp on the requested fetch level.
Storage-image operations use zero for this sampling parameter.

Honeykrisp now handles integer view minima alongside its gather bounds
predicate, without passing an integer value as the floating-point NIR
`min_lod` sampling source. This also preserves the G17P common texture
lowering's coordinate and level robustness path.

```sh
meson test -C build hk_nir_image_view_min_lod agx_texture_min_lod \
  agx3_graphics_compile --print-errorlogs
meson compile -C build g17p-vulkan-min-lod-txf0 g17p-vulkan-min-lod-txf1
```

The new CPU test covers 882 combinations of sampled fetch, array fetch, 3D
fetch, gather and multisample fetch, three result types, integer and
fractional minima, rebased views, and levels on both sides of the threshold.
It checks all four zero components and preservation of the requested level.
The runner also requires the two new prebuilt fetch binaries: level zero
and level one, each tested at minima 0, 0.5, 1, 1.5 and 2. Red and green
identify valid fetches; zero including alpha identifies the excluded ones.
