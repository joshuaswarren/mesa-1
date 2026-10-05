#!/bin/bash
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
#
# Create the experimental Proton tool next to an existing one, swapping in the
# d3d12agx Wine pair. The source tool is never modified.
#
# Usage: install-proton-tool.sh [--refresh] <compatibilitytools.d> <source-tool-name> <new-tool-name> <artifact-dir>
#   artifact-dir holds d3d12.dll (Wine-builtin marked, ARM64EC) and d3d12.so
#   --refresh: the new tool already exists; replace only the d3d12 pair, the
#              wrapper and the build record (prefixes keep working).
set -euo pipefail
refresh=0
[[ ${1:-} == --refresh ]] && { refresh=1; shift; }
root=$1; src_name=$2; new_name=$3; art=$(realpath "$4")
src=$root/$src_name; dst=$root/$new_name
[[ -d $src ]] || { echo "missing source tool $src" >&2; exit 2; }
[[ -f $art/d3d12.dll && -f $art/d3d12.so ]] || { echo "missing artifacts in $art" >&2; exit 2; }
if (( refresh )); then
   [[ -d $dst && -f $dst/AURORA-D3D12AGX-BUILD.json ]] || { echo "no experimental tool to refresh at $dst" >&2; exit 2; }
else
   [[ -e $dst ]] && { echo "refusing to overwrite $dst" >&2; exit 2; }
   # Hard-link copy: the tool is >1 GiB, the files we change are replaced, not edited.
   cp -al "$src" "$dst"
fi
wine=$dst/files/lib/wine

# Replace the PE (Wine's ARM64X d3d12 kept for reference) and add the unixlib.
rm -f "$wine/aarch64-windows/d3d12.dll"
[[ -f $wine/aarch64-windows/d3d12.dll.wine-orig ]] ||
   cp "$src/files/lib/wine/aarch64-windows/d3d12.dll" "$wine/aarch64-windows/d3d12.dll.wine-orig"
install -m 0644 "$art/d3d12.dll" "$wine/aarch64-windows/d3d12.dll"
install -m 0755 "$art/d3d12.so" "$wine/aarch64-unix/d3d12.so"
# Stop Proton from copying vkd3d-proton's d3d12 into prefixes for this tool.
for arch in aarch64-windows x86_64-windows i386-windows; do
   for f in d3d12.dll d3d12core.dll; do
      if [[ -e $wine/vkd3d-proton/$arch/$f ]]; then
         rm -f "$wine/vkd3d-proton/$arch/$f"
      fi
   done
done

# Tool identity.
rm -f "$dst/compatibilitytool.vdf"
sed "s/$src_name/$new_name/g" "$src/compatibilitytool.vdf" > "$dst/compatibilitytool.vdf"

# Wrapper: force the builtin d3d12 and keep per-game logs in this tool unless
# the caller already chose a log directory (the test runners do).
rm -f "$dst/aurora-proton"
python3 - "$src/aurora-proton" "$dst/aurora-proton" <<'PY'
import sys
src, dst = sys.argv[1:3]
text = open(src).read()
marker = 'export AURORA_TOOL_ROOT PROTON_USE_ARM64=1\n'
inject = ('export AURORA_TOOL_ROOT PROTON_USE_ARM64=1\n'
          '# d3d12agx experimental tool: builtin d3d12 (ARM64EC PE + unixlib), vkd3d-proton bypassed.\n'
          'export WINEDLLOVERRIDES="d3d12=b;d3d12core=b${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"\n'
          'export PROTON_LOG=${PROTON_LOG:-1}\n'
          'export PROTON_LOG_DIR="${PROTON_LOG_DIR:-$AURORA_TOOL_ROOT/aurora-game-logs}"\n')
if marker not in text:
    sys.exit('wrapper layout changed; adjust install-proton-tool.sh')
open(dst, 'w').write(text.replace(marker, inject, 1))
PY
chmod 0755 "$dst/aurora-proton"
mkdir -p "$dst/aurora-game-logs"
cat > "$dst/AURORA-D3D12AGX-BUILD.json" <<JSON
{"tool": "$new_name", "based_on": "$src_name", "artifacts": "$art",
 "source_commit": "$(cat "$art/SOURCE-COMMIT" 2>/dev/null || echo unknown)",
 "sha256": {"d3d12.dll": "$(sha256sum "$art/d3d12.dll" | cut -d' ' -f1)",
            "d3d12.so": "$(sha256sum "$art/d3d12.so" | cut -d' ' -f1)"}}
JSON
echo "$( (( refresh )) && echo refreshed || echo installed) $dst"
