#!/bin/sh
set -u
cd /tmp/mesa-sin-ftz-jwm1
V=/var/tmp/jwm1-v072rc1/venv/bin/python
H=/var/tmp/jwm1-v072rc1/harness
W=/tmp/mesa-sin-ftz-jwm1/window-q4
mkdir -p "$W"
exec 9>/tmp/m1-gpu.lock
flock -x 9 || { echo BLOCKER-FLOCK; exit 11; }
echo "lock inode: $(stat -c %i /tmp/m1-gpu.lock) (expect 1384)"
for round in 1 2 3; do
  for arm in fckey e167; do
    echo "== r$round $arm start $(date -u +%H:%M:%S)"
    MESA_SHADER_CACHE_DISABLE=true HF_HUB_OFFLINE=1 \
    VK_DRIVER_FILES=/tmp/mesa-sin-ftz-jwm1/jwm1-$arm-icd.json \
    $V $H/bench_decode_vlm.py --prompt-id long --tokens 32 --warmup-tokens 4 \
      --stop-policy ignore-eos > "$W/$arm-r$round.log" 2>&1
    echo "== r$round $arm rc=$? $(date -u +%H:%M:%S)"
  done
done
echo "== window done"
