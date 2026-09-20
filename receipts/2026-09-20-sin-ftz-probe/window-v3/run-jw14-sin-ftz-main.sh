#!/bin/bash
set -euo pipefail
cd /tmp/jw14-sin-ftz-main-v3
exec 9</tmp/jw14-bench/gpu.lock
flock -n 9
sha256sum -c drivers.sha256
sha256sum sin-denorm-probe.py ./*.spv ./*.json > inputs.before.sha256
VK_DRIVER_FILES="$PWD/jw14m2-fckey-icd.json" timeout 30s vulkaninfo > candidate-vulkaninfo.txt 2>candidate-vulkaninfo.stderr
for arm in e167 fckey; do
  env -u VK_DRIVER_FILES -u VK_ICD_FILENAMES MESA_SHADER_CACHE_DISABLE=true \
    MATHREPRO=/tmp/mesa-fckey-095cb/bin/mathrepro \
    timeout --signal=TERM --kill-after=10s 150s \
    /tmp/v072rc1-main-m2/venv/bin/python sin-denorm-probe.py \
      "$PWD/jw14m2-$arm-icd.json" "$arm" > "$arm.log" 2>&1
done
sha256sum -c drivers.sha256
sha256sum -c inputs.before.sha256
printf 'Completed both GPU probe arms; lock released on exit.\n'
