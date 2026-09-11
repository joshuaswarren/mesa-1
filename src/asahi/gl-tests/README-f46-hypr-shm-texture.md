# F46 Hyprland SHM texture oracle

F46 reproduces the GL work Hyprland performs when it receives a large
`WL_SHM_FORMAT_ARGB8888` client buffer. It is a surfaceless EGL test, so it does
not open a window or display an image. Success is reported through exact pixel
checks and the process exit status.

The client image is 1840x960. It is uploaded with `GL_BGRA_EXT`, sampled as a
2D texture, and composited over an opaque background at opacity 0.985. This
matches the byte order of little-endian Wayland ARGB8888 and Omarchy's default
window opacity. Software runs check four asymmetric pixels immediately and
again after three seconds, allowing a tolerance of two in each 8-bit channel.

## Modes

`--static`

- Allocates and fills one 1840x960 BGRA texture.
- Draws and submits it once at opacity 0.985.
- Establishes the baseline upload, sampling, blending, and retained-image
  behavior without incremental updates or resource replacement.

`--updates`

- Starts with the same 1840x960 texture.
- Performs 120 uploads at 40 Hz, for three seconds total.
- Each upload changes one 32-row band with `glTexSubImage2D` while setting
  `GL_UNPACK_ROW_LENGTH` and `GL_UNPACK_SKIP_ROWS` exactly as Hyprland's SHM
  texture update does.
- Draws and submits after every band update. The texture allocation remains
  unchanged throughout this mode.

`--resize`

- Draws the initial 1840x960 texture.
- Deletes it after submission, allocates and draws a 1760x920 replacement,
  then deletes that and returns to an 1840x960 replacement.
- Uses solid but asymmetric colors for each allocation so scaling cannot make
  the expected result ambiguous.
- Isolates two draw/delete/reallocate transitions without incremental
  `glTexSubImage2D` updates.

## Build and local validation

From an AArch64 Linux builder with this repository mounted at `/src`:

```sh
cd /src
./src/asahi/gl-tests/run-f46-hypr-shm-texture.sh \
  /src/_build-hypr-desktop /src/_build-full --software --static
```

The last two arguments select the backend and mode:

```text
backend: --software | --shim | --hardware
mode:    --static | --updates | --resize
```

Run the local controls for every mode:

```sh
for mode in --static --updates --resize; do
  ./src/asahi/gl-tests/run-f46-hypr-shm-texture.sh \
    /src/_build-hypr-desktop /src/_build-full --software "$mode"
  ./src/asahi/gl-tests/run-f46-hypr-shm-texture.sh \
    /src/_build-hypr-desktop /src/_build-full --shim "$mode"
done
```

The software control performs real rendering and should end with:

```text
GL46 immediate: exact BGRA/0.985 composite (+/-2)
GL46 retained +3s: exact BGRA/0.985 composite (+/-2)
GL46 Hypr wl_shm BGRA texture <mode> exact immediately/3s
```

The G17P no-op control validates command-stream submission but deliberately
does not execute GPU work or inspect pixels. It may print one VDM validation
line per submitted draw and should end with:

```text
GL46 Hypr wl_shm BGRA texture <mode> submitted (pixels skipped)
```

Any nonzero exit status, GL error, pixel mismatch, shader failure, EGL failure,
or missing final mode summary is a failure. Exit status 77 means the selected
GL stack does not expose `GL_EXT_texture_format_BGRA8888`, so that stack cannot
run this oracle.

## Packaged binary

The standalone binary accepts exactly one mode argument:

```sh
f46-hypr-shm-texture --static
f46-hypr-shm-texture --updates
f46-hypr-shm-texture --resize
```

It dynamically links EGL and GLESv2 and therefore must run with the matching
Mesa libraries and driver environment. `file` should identify the packaged
artifact as an AArch64 ELF executable.
