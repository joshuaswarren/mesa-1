# G17P element-buffer offset oracle

Qt image nodes issue indexed triangle strips starting at the second U16 in an
element buffer. G17P's index fetch rounds that address down to a dword boundary:
the preceding restart value replaces one vertex and half of the image disappears.
Gallium now uploads misaligned direct element-buffer ranges to aligned storage
before acquiring the render batch. A synchronized read map preserves GPU-written
indices. Aligned draws, indirect draws, uploaded user indices, and other chips
retain their existing paths.

Build and run against the selected native Mesa on Neo:

```sh
cc -O2 test-g17p-gl-index-offset.c -o index-offset -lEGL -lGLESv2
./index-offset
```

The surfaceless GLES 3.1 test checks all 4096 pixels in each of 48 cases: U8/U16/U32,
four element offsets, restart enabled/disabled, and CPU uploads/GPU compute
writes. It prints the actual renderer and exits nonzero on any mismatch or GL
error. U32 offsets provide aligned controls. An unpatched Neo fails sub-dword
offsets; the patched driver must report `48/48 cases passed`. A Qt Quick image
probe and the real Omarchy wallpaper are also checked on the running desktop.
