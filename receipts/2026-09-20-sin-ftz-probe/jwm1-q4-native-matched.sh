#!/bin/sh
set -u
cd /tmp/mesa-sin-ftz-jwm1
V=/var/tmp/jwm1-v072rc1/venv/bin/python
SNAP=/home/joshuawarren/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a41-real
W=/tmp/mesa-sin-ftz-jwm1/window-q4-clean
mkdir -p "$W"
echo "== model snapshot check: $(ls $SNAP/model.safetensors 2>/dev/null || echo MISSING)"
ls "$SNAP/model.safetensors" >/dev/null 2>&1 || { echo "BLOCKER: snapshot missing"; exit 8; }
exec 9>/tmp/m1-gpu.lock
flock -x 9 || { echo BLOCKER-FLOCK; exit 11; }
echo "lock inode: $(stat -c %i /tmp/m1-gpu.lock)"
echo "== candidate arm (fckey ICD), 5 rounds x 2 legs, unprofiled"
for round in 1 2 3 4 5; do
  MESA_SHADER_CACHE_DISABLE=false HF_HUB_OFFLINE=1 \
  VK_DRIVER_FILES=/tmp/mesa-sin-ftz-jwm1/jwm1-fckey-icd.json \
  $V /var/tmp/jwm1-v072rc1/harness/bench_decode.py \
    --model "$SNAP" \
    --prompt "Hi" --tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4 \
    > "$W/fckey-short-r$round.log" 2>&1
  echo "fckey short r$round rc=$?"
  MESA_SHADER_CACHE_DISABLE=false HF_HUB_OFFLINE=1 \
  VK_DRIVER_FILES=/tmp/mesa-sin-ftz-jwm1/jwm1-fckey-icd.json \
  $V /var/tmp/jwm1-v072rc1/harness/bench_decode.py \
    --model "$SNAP" \
    --prompt "You are reviewing a measurement protocol for a small language-model benchmark. The benchmark runs the same workload matrix on two operating systems and compares prefill and decode throughput. Each leg loads one pinned model snapshot, applies the model chat template to the prompt, runs greedy decoding with a fixed random seed, and records the prompt phase, the per-token phase, the generated token count, a hash of the generated token ids, and peak memory. A leg is valid only when the model snapshot revision matches the manifest pin, the prompt applies the chat template, and the generated ids match the reference digest for the leg. Entry 20 of 20." \
    --tokens 32 --temp 0.0 --seed 0 --warmup-tokens 4 \
    > "$W/fckey-ctx1024-r$round.log" 2>&1
  echo "fckey ctx1024 r$round rc=$?"
done
echo "== window done"
