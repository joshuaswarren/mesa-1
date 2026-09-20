#!/bin/sh
set -u
cd /tmp/mesa-sin-ftz-ab
W=/tmp/mesa-sin-ftz-ab/window-q4-12r
mkdir -p "$W"
systemctl is-active llm-inference.service > "$W/service-before.txt" 2>&1
echo "service before: $(cat $W/service-before.txt)"
sudo -n systemctl stop llm-inference.service || { echo BLOCKER-SUDO; exit 10; }
restore() {
  sudo -n systemctl start llm-inference.service
  sleep 4
  systemctl is-active llm-inference.service > "$W/service-after.txt" 2>&1
  echo "service after: $(cat $W/service-after.txt)"
}
trap 'restore' EXIT
exec 9>/tmp/m1-gpu.lock
flock -x 9 || { echo BLOCKER-FLOCK; exit 11; }
echo "lock inode: $(stat -c %i /tmp/m1-gpu.lock)"
echo "== fckey 12r start $(date -u +%H:%M:%S)"
VK_DRIVER_FILES=/tmp/mesa-sin-ftz-ab/jw16-fckey-icd.json \
  python3 /var/tmp/cb/gap_ladder.py /var/tmp/v071perf-venv /var/tmp/cb/scripts "$W/out-fckey-12r.json" 12 \
  > "$W/fckey-12r.log" 2>&1
echo "== fckey 12r rc=$? $(date -u +%H:%M:%S)"
echo "== e167 12r start $(date -u +%H:%M:%S)"
VK_DRIVER_FILES=/tmp/mesa-sin-ftz-ab/jw16-e167-icd.json \
  python3 /var/tmp/cb/gap_ladder.py /var/tmp/v071perf-venv /var/tmp/cb/scripts "$W/out-e167-12r.json" 12 \
  > "$W/e167-12r.log" 2>&1
echo "== e167 12r rc=$? $(date -u +%H:%M:%S)"
