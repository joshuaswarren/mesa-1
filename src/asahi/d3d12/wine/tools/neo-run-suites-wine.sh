#!/bin/bash
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
#
# Run the 17 frozen suites, cross-built as x86-64 Windows exes, through a
# Proton tool on the Neo. The frozen run-suites.sh is executed unchanged; the
# "binaries" it runs are wrappers that launch the .exe with neo-run-proton.sh,
# so every child's stdout/stderr/rc is kept.
#
# Usage: neo-run-suites-wine.sh <tool-name> <prefix-dir> <suites-dir> <results-dir>
#   suites-dir holds d3d12agx_<name>.exe, shaders/, run-suites.sh (from build-windows-tests.sh)
set -u
tool=$1; prefix=$2; suites=$(realpath "$3"); results=$4
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ ! -e $results ]] || { echo "results dir exists: $results" >&2; exit 2; }
mkdir -p "$results/wrappers" "$results/logs"
results=$(realpath "$results")
names=(g1_smoke g2_smoke g3_smoke g4_smoke g4_triangle g5_depth g5_descriptors
       g5_atomics g5_control_flow g5_wave g5_texture g6_geometry g7_tessellation
       g8_mesh g9_amplification g10_vertex_buffers g11_indexed_instanced)
for name in "${names[@]}"; do
   cat > "$results/wrappers/d3d12agx_$name" <<WRAPPER
#!/bin/bash
set -u
out=$results/$name
printf '%q ' "$suites/d3d12agx_$name.exe" "\$@" > "\$out.command"; printf '\\n' >> "\$out.command"
date -u +%FT%TZ > "\$out.start"
log=\$(STEAM_ROOT=\${STEAM_ROOT:-} bash "$here/neo-run-proton.sh" "$tool" "$prefix" "$results/logs" 170 "$suites/d3d12agx_$name.exe" "\$@")
rc=\$?
printf '%s\\n' "\$rc" > "\$out.rc"
cp "\$log" "\$out.stdout"
# the suites print their verdicts on stdout; proton merges streams into the log
cat "\$out.stdout"
exit "\$rc"
WRAPPER
   chmod +x "$results/wrappers/d3d12agx_$name"
done
{
   uname -a; getconf PAGESIZE; cat "/home/user/.local/share/Aurora-Steam-ARM64-20261005/compatibilitytools.d/$tool/AURORA-D3D12AGX-BUILD.json" 2>/dev/null
} > "$results/MACHINE.txt" 2>&1
start=$(date +%s)
(cd "$suites" && bash "$suites/run-suites.sh" "$results/wrappers") > "$results/TEST.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$results/RC.txt"
journalctl -k --since="@$start" --no-pager > "$results/KERNEL.log" 2>&1 || true
cat "$results/TEST.log"
exit $rc
