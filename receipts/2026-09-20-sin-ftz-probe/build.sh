#!/bin/sh
# Assembles both probe modules (spirv-as from ~/.local/bin on this box).
set -e
cd "$(dirname "$0")"
spirv-as --target-env vulkan1.1 sin-denorm-ftz.spvasm -o sin-denorm-ftz.spv
spirv-as --target-env vulkan1.1 sin-denorm-default.spvasm -o sin-denorm-default.spv
# guard: the ftz module must carry the execution mode, the default must not
spirv-dis sin-denorm-ftz.spv | grep -q "OpExecutionMode .* DenormFlushToZero 32"
! spirv-dis sin-denorm-default.spv | grep -q "DenormFlushToZero"
echo "probe modules OK (ftz keyed, default clean)"
