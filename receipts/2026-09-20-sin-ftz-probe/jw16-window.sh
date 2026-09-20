#!/bin/sh
# jw16 sin-FTZ A/B window - Main-delegated. Lock: existing /tmp/m1-gpu.lock
# (inode 12, exclusive flock, never replaced). Service: stopped before the
# lock, restarted + confirmed active AFTER release, even on failure (trap).
set -u
cd /tmp/mesa-sin-ftz-ab
W=/tmp/mesa-sin-ftz-ab/window-jw16
mkdir -p "$W"

systemctl is-active llm-inference.service > "$W/service-before.txt" 2>&1
echo "== service before: $(cat $W/service-before.txt)"

if ! sudo -n systemctl stop llm-inference.service; then
  echo "BLOCKER: cannot stop llm-inference (sudo)"; exit 10
fi
stopped=1
restore() {
  sudo -n systemctl start llm-inference.service
  sleep 2
  systemctl is-active llm-inference.service > "$W/service-after.txt" 2>&1
  echo "== service after: $(cat $W/service-after.txt)"
}
trap 'restore' EXIT

exec 9>/tmp/m1-gpu.lock
if ! flock -x 9; then echo "BLOCKER: flock -x failed"; exit 11; fi
echo "== lock inode: $(stat -c %i /tmp/m1-gpu.lock) (must be 12)"

VK_DRIVER_FILES=/tmp/mesa-sin-ftz-ab/jw16-fckey-icd.json \
  vulkaninfo > "$W/caps-fckey.txt" 2> "$W/caps-fckey.stderr"
grep -A14 VkPhysicalDeviceVulkan12Properties "$W/caps-fckey.txt" \
  > "$W/caps-fckey-props.txt" || echo "no Vulkan12Properties block found"
grep -E "shaderDenorm(Preserve|FlushToZero)Float32" "$W/caps-fckey-props.txt" || true

MATHREPRO=/tmp/mesa-sin-ftz-ab/bin/mathrepro \
  python3 probe/sin-denorm-probe.py jw16-fckey-icd.json jw16-fckey \
  --env MESA_SHADER_CACHE_DISABLE=true > "$W/fckey-run.txt" 2>&1
echo "== fckey arm rc=$?"

MATHREPRO=/tmp/mesa-sin-ftz-ab/bin/mathrepro \
  python3 probe/sin-denorm-probe.py jw16-e167-icd.json jw16-e167 \
  --env MESA_SHADER_CACHE_DISABLE=true > "$W/e167-run.txt" 2>&1
echo "== e167 arm rc=$?"

sha256sum -c PINS.sha256 > "$W/pins-after.txt" 2>&1 && echo "== pins after: PASS"

python3 - <<'PY'
import struct
def words(p):
    d = open(p, "rb").read()
    assert len(d) == 256, (p, len(d))
    return list(struct.unpack("<64I", d))
SUB = {0x00000001, 0x007fffff, 0x80000001, 0x807fffff}
ZER = {0x00000000, 0x80000000}
inp = words("probe/in-jw16-fckey.bin")
legs = {("e167","default"): words("probe/out-jw16-e167-default.bin"),
        ("e167","ftz"):     words("probe/out-jw16-e167-ftz.bin"),
        ("fckey","default"):words("probe/out-jw16-fckey-default.bin"),
        ("fckey","ftz"):    words("probe/out-jw16-fckey-ftz.bin")}
assert inp == words("probe/in-jw16-e167.bin"), "inputs differ"
fails = []
for i, x in enumerate(inp):
    d_e, f_e = legs[("e167","default")][i], legs[("e167","ftz")][i]
    d_c, f_c = legs[("fckey","default")][i], legs[("fckey","ftz")][i]
    if d_e != d_c: fails.append((i, hex(x), "default arms differ", d_e, d_c))
    if x in SUB:
        if f_e != x: fails.append((i, hex(x), "baseline FTZ leak expected", f_e, x))
        if f_c != x & 0x80000000:
            fails.append((i, hex(x), "candidate FTZ signed zero expected", f_c, x & 0x80000000))
    elif x in ZER:
        if len({d_e, f_e, d_c, f_c}) != 1: fails.append((i, hex(x), "zero rows differ", d_e, f_e, d_c, f_c))
    else:
        if len({d_e, f_e, d_c, f_c}) != 1: fails.append((i, hex(x), "other rows differ", d_e, f_e, d_c, f_c))
print("semantic rows: %d words x 4 legs, %d failures" % (len(inp) * 4, len(fails)))
for f in fails: print("FAIL:", f)
open("window-jw16/semantic-assert.txt", "w").write(
    "%d failures\n" % len(fails) + "\n".join(map(str, fails)))
sys_exit = 1 if fails else 0
raise SystemExit(sys_exit)
PY
echo "== semantic assert rc=$?"
