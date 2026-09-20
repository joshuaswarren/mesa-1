#!/bin/sh
# Assembles both probe modules and MANDATORILY validates them.
# v1 modules failed vtn: "Variables in StorageBuffer storage class must have
# struct type with Block decoration" - originals kept as *.invalid.spvasm/.spv.
set -e
cd "$(dirname "$0")"
AS=spirv-as
VAL="spirv-val --target-env vulkan1.2"
$AS --target-env vulkan1.2 sin-denorm-ftz-v2.spvasm -o sin-denorm-ftz-v2.spv
$AS --target-env vulkan1.2 sin-denorm-default-v2.spvasm -o sin-denorm-default-v2.spv
$VAL sin-denorm-ftz-v2.spv
$VAL sin-denorm-default-v2.spv
# execution-mode guard: ftz carries DenormFlushToZero 32, default does not
spirv-dis sin-denorm-ftz-v2.spv | grep -q "OpExecutionMode .* DenormFlushToZero 32"
! spirv-dis sin-denorm-default-v2.spv | grep -q "DenormFlushToZero"
echo "probe modules OK: assembled, spirv-val clean, ftz keyed, default clean"
