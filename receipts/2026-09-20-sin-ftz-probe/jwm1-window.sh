#!/bin/sh
# jwm1 G13G pinned-CTS window (1989 cases) - Main-delegated.
# Lock: existing /tmp/m1-gpu.lock (inode 1384 expected), exclusive flock.
# Fresh install: no inference service exists - verified, nothing to stop/restore.
set -u
cd /tmp/mesa-sin-ftz-jwm1
DEQP=/tmp/mesa-sin-ftz-jwm1/bin/deqp-vk
LIST=/tmp/mesa-sin-ftz-jwm1/probe/fp32-caselist.txt
W=/tmp/mesa-sin-ftz-jwm1/window-cts146
mkdir -p "$W"

echo "service unit check: $(systemctl list-unit-files llm-inference.service 2>/dev/null | tail -1)"
systemctl is-active llm-inference.service > "$W/service-before.txt" 2>&1
echo "service before: $(cat $W/service-before.txt)"
sha256sum -c PINS.sha256 > "$W/pins-before.txt" 2>&1 && echo "pins before: PASS" || { echo "BLOCKER: pins"; cat "$W/pins-before.txt"; exit 9; }

exec 9>/tmp/m1-gpu.lock
flock -x 9 || { echo "BLOCKER: flock"; exit 11; }
echo "lock inode: $(stat -c %i /tmp/m1-gpu.lock) (expect 1384)"

for arm in fckey e167; do
  mkdir -p "$W/$arm"
  cd "$W/$arm"
  echo "== leg $arm start $(date -u +%H:%M:%S)"
  MESA_SHADER_CACHE_DISABLE=true \
  VK_DRIVER_FILES=/tmp/mesa-sin-ftz-jwm1/jwm1-$arm-icd.json \
  timeout 600 $DEQP --deqp-runmode=execute \
    --deqp-caselist-file=$LIST \
    --deqp-log-images=disable \
    > "$W/$arm/run.log" 2> "$W/$arm/run.stderr"
  echo "== leg $arm rc=$? end $(date -u +%H:%M:%S)"
  cp TestResults.qpa "$W/$arm/results.qpa" 2>/dev/null || true
  gzip -f "$W/$arm/results.qpa" 2>/dev/null || true
done
cd /tmp/mesa-sin-ftz-jwm1
sha256sum -c PINS.sha256 > "$W/pins-after.txt" 2>&1 && echo "pins after: PASS" || { echo "PINS-AFTER-FAIL"; cat "$W/pins-after.txt"; }
echo "== window done; no service to restore (none exists on this host)"
