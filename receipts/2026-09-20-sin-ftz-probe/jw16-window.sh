#!/bin/sh
# jw16 pinned-CTS window (1989 float_controls cases) - Main-delegated.
# Lock: existing /tmp/m1-gpu.lock inode 12, exclusive flock, never replaced.
# Service: stopped before lock, restarted + confirmed active AFTER release.
set -u
cd /tmp/mesa-sin-ftz-ab
DEQP=/tmp/cts-146/build/external/vulkancts/modules/vulkan/deqp-vk
LIST=/tmp/mesa-sin-ftz-ab/probe/fp32-caselist-vk146-binary-verified.txt
W=/tmp/mesa-sin-ftz-ab/window-cts146
mkdir -p "$W"

systemctl is-active llm-inference.service > "$W/service-before.txt" 2>&1
echo "service before: $(cat $W/service-before.txt)"
sha256sum -c PINS.sha256 > "$W/pins-before.txt" 2>&1 && echo "pins before: PASS" || { echo "BLOCKER: pins"; cat "$W/pins-before.txt"; exit 9; }
sha256sum "$DEQP" | head -1

sudo -n systemctl stop llm-inference.service || { echo "BLOCKER: sudo stop"; exit 10; }
restore() {
  sudo -n systemctl start llm-inference.service
  sleep 4
  systemctl is-active llm-inference.service > "$W/service-after.txt" 2>&1
  echo "service after: $(cat $W/service-after.txt)"
}
trap 'restore' EXIT

exec 9>/tmp/m1-gpu.lock
flock -x 9 || { echo "BLOCKER: flock"; exit 11; }
echo "lock inode: $(stat -c %i /tmp/m1-gpu.lock)"

for arm in fckey e167; do
  mkdir -p "$W/$arm"
  cd "$W/$arm"
  echo "== leg $arm start $(date -u +%H:%M:%S)"
  MESA_SHADER_CACHE_DISABLE=true \
  VK_DRIVER_FILES=/tmp/mesa-sin-ftz-ab/jw16-$arm-icd.json \
  timeout 600 $DEQP --deqp-runmode=execute \
    --deqp-caselist-file=$LIST \
    --deqp-log-images=disable \
    > "$W/$arm/run.log" 2> "$W/$arm/run.stderr"
  echo "== leg $arm rc=$? end $(date -u +%H:%M:%S)"
  cp TestResults.qpa "$W/$arm/results.qpa" 2>/dev/null || true
  gzip -f "$W/$arm/results.qpa" 2>/dev/null || true
done
echo "== window done"
