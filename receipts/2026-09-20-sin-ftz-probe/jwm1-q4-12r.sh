#!/bin/sh
set -u
cd /tmp/mesa-sin-ftz-jwm1
V=/var/tmp/jwm1-v072rc1/venv/bin/python
H=/var/tmp/jwm1-v072rc1/harness
SNAP=/home/joshuawarren/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381
W=/tmp/mesa-sin-ftz-jwm1/window-q4-12r
mkdir -p "$W"
exec 9>/tmp/m1-gpu.lock
flock -x 9 || { echo BLOCKER-FLOCK; exit 11; }
echo "lock inode: $(stat -c %i /tmp/m1-gpu.lock) (post-reboot recreation)"
echo "cpus: $(nproc); ane module: $(test -d /sys/module/ane && echo present || echo absent)"
sha256sum -c PINS.sha256 > "$W/pins-before.txt" 2>&1 && echo "pins before: PASS" || { echo PINS-FAIL; exit 9; }
for round in $(seq 1 12); do
  for arm in fckey e167; do
    echo "== r$round $arm start $(date -u +%H:%M:%S)"
    MESA_SHADER_CACHE_DISABLE=true HF_HUB_OFFLINE=1 \
    VK_DRIVER_FILES=/tmp/mesa-sin-ftz-jwm1/jwm1-$arm-icd.json \
    $V $H/bench_decode_vlm.py --model "$SNAP" \
      --matrix $H/bench_matrix.json \
      --prompt-id long --tokens 32 --warmup-tokens 4 \
      --stop-policy ignore-eos > "$W/$arm-r$round.log" 2>&1
    echo "== r$round $arm rc=$? $(date -u +%H:%M:%S)"
  done
done
sha256sum -c PINS.sha256 > "$W/pins-after.txt" 2>&1 && echo "pins after: PASS" || echo PINS-AFTER-FAIL
echo "== window done"
