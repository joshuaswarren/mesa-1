# Honeykrisp Q4_ROWS pack_64_4x16 reproducer

This is the exact compute shader used to reproduce the AGX compiler failure on jw16 (M1 Max, T6001), Mesa fork commit `1432df0196b`. The shader came from the GemvRepack bench source, SHA256 `9697a3c9ff9c8e575d482cd047900a0ce59a2891e99193cb4ef6db395bdebcb7`.

Build the shader with `QMM_VEC_PIPE=1`, then create its Vulkan compute pipeline using `ROWS_PER_SLOT=2`, `SLOTS_PER_GROUP=4` (also reproduces at `ROWS_PER_SLOT=4`, `SLOTS_PER_GROUP=4`). Pipeline creation repeatedly prints `Unhandled ALU op pack_64_4x16` and does not complete. This is the AGX compiler's unsupported-ALU path in `src/asahi/compiler/agx_compile.c`; the unbounded diagnostic spam is part of the observed failure. The same pipelined shader at `ROWS_PER_SLOT=1`, `SLOTS_PER_GROUP=8` compiles and runs. Use glslangValidator-compatible SPIR-V; jw16's glslc output has a separate, pre-existing Mesa incompatibility.

The standalone source needs the Q4 bench's descriptor layout and push-constant setup to create the pipeline. No AsahiLinux repository is required; the repro source and AGX failure location are both in this mesa-1 fork.
