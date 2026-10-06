#!/bin/bash
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
# run.sh VERSION BIN_DIR LIBRARY_PATH OUTPUT
# Both versions use their frozen mandatory runner. A compile-only Proton ABI
# binding is selected at build time, never by modifying fixture assertions.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
version=$1
[[ $version == v1 || $version == v2 ]]
bin=$(realpath "$2")
[[ ! -e $4 ]]
mkdir -p "$4/wrappers"
out=$(realpath "$4")
(cd "$here/$version" && sha256sum -c "../$version.sha256") > "$out/INPUT-CHECK.txt"
cp "$here/$version.sha256" "$out/INPUTS.sha256"
{
 hostname; uname -a; cat /proc/cmdline
 tr '\0' ' ' < /proc/device-tree/compatible; echo
 rpm -q mesa-vulkan-drivers kernel-16k || true
 systemctl is-active aurora-kbd-off.service
 DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus" busctl --user set-property org.gnome.Mutter.DisplayConfig /org/gnome/Mutter/DisplayConfig org.gnome.Mutter.DisplayConfig PowerSaveMode i 3
 DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus" busctl --user get-property org.gnome.Mutter.DisplayConfig /org/gnome/Mutter/DisplayConfig org.gnome.Mutter.DisplayConfig PowerSaveMode
} > "$out/MACHINE.txt" 2>&1
sha256sum "$bin"/* > "$out/BINARIES.sha256"
# Colon-separated runtime search directories are deliberately supported.
IFS=: read -ra dirs <<< "$3"
for i in "${!dirs[@]}"; do dirs[$i]=$(realpath "${dirs[$i]}"); done
for dir in "${dirs[@]}"; do
 find "$dir" -maxdepth 1 -type f -name '*.so*' -exec sha256sum {} +
done > "$out/LIBRARIES.sha256"
export FROZEN17_BIN=$bin FROZEN17_OUT=$out
for binary in "$bin"/d3d12agx_*; do
 name=${binary##*/}
 cat > "$out/wrappers/$name" <<'WRAPPER'
#!/bin/bash
set -u
name=${0##*/d3d12agx_}
stem=$FROZEN17_OUT/$name
printf '%q ' "$FROZEN17_BIN/d3d12agx_$name" "$@" > "$stem.command"
printf '\n' >> "$stem.command"
date -u +%FT%TZ > "$stem.start"
timeout --kill-after=5s 170s "$FROZEN17_BIN/d3d12agx_$name" "$@" > "$stem.stdout" 2> "$stem.stderr"
rc=$?
printf '%s\n' "$rc" > "$stem.rc"
cat "$stem.stdout"
cat "$stem.stderr" >&2
exit "$rc"
WRAPPER
 chmod +x "$out/wrappers/$name"
done
runtime=$(IFS=:; echo "${dirs[*]}")
export LD_LIBRARY_PATH="$runtime"
unset LD_PRELOAD ASAHI_SHIM_CHIP
# Do not inherit opt-in diagnostic/emulation knobs from another campaign.
while IFS= read -r variable; do unset "$variable"; done < <(compgen -A variable | /usr/bin/grep '^AURORA_D3D12_' || true)
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/asahi_icd.aarch64.json
export VKD3D_DEBUG=warn
export VKD3D_SHADER_CACHE_PATH="$out"
ldd "$bin/d3d12agx_g1_smoke" > "$out/LIBRARY-RESOLUTION.txt"
printf '%q ' bash "$here/$version/run-suites.sh" "$out/wrappers" > "$out/COMMAND.txt"
printf '\n' >> "$out/COMMAND.txt"
env | sort > "$out/ENV.txt"
start=$(date +%s)
# Existing reference-machine advisory GPU lock, bounded queue wait.
exec 9> "$HOME/aurora-d3d12/gpu.lock"
flock -w 600 9
cd "$out"
set +e
bash "$here/$version/run-suites.sh" "$out/wrappers" > "$out/TEST.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$out/RC.txt"
sudo -n journalctl -k --since="@$start" --no-pager > "$out/KERNEL.log" 2>&1
(cd "$here/$version" && sha256sum -c "../$version.sha256") > "$out/INPUT-AFTER.txt"
cat "$out/TEST.log"
exit "$rc"
