#!/usr/bin/env bash
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
# Supplementary P09 compiler/CODE/Ioto certificate, never frozen-suite parity.
set -euo pipefail
if [[ ${1:-} == --help ]]; then
  echo "Usage: $0 [PREFIX CANDIDATE_SHA OUTPUT_DIR]"
  echo 'Without arguments, run from PREFIX/diagnostics/P09. Neo only through the hardware queue.'
  exit 0
fi
if (( $# == 0 )); then
  prefix=$(cd -- "$(dirname -- "$0")/../.." && pwd -P)
  sha=$(cat "$prefix/SOURCE-COMMIT")
  out=${AURORA_CERTIFICATION_OUT:-/var/tmp/aurora-d3d12-P09-$sha}
elif (( $# == 3 )); then
  prefix=$(realpath "$1")
  sha=$2
  out=$3
else
  echo 'Expected no arguments or PREFIX CANDIDATE_SHA OUTPUT_DIR' >&2
  exit 2
fi
[[ $sha =~ ^[0-9a-f]{40}$ ]]
[[ $(uname -s) == Linux && $(uname -m) == aarch64 && $(getconf PAGESIZE) == 16384 ]]
[[ $(cat "$prefix/SOURCE-COMMIT") == "$sha" ]]
[[ $(cat "$prefix/REFERENCE-COMMIT") == ba4cfd5dfa4c4d6f5b88adcf9ac5eb233c3442a4 ]]
# The staging manifest is generated from reviewed build output, not recreated
# from installed files. This also pins this helper and the actual shim.
(cd "$prefix" && sha256sum --strict -c ARTIFACTS.sha256) >/dev/null
[[ ! -e $out ]]
mkdir -p "$out"
out=$(realpath "$out")
cp "$prefix/SOURCE-COMMIT" "$prefix/REFERENCE-COMMIT" "$prefix/REFERENCE-BUILD.json" "$out/"
printf '%q ' "$0" "$prefix" "$sha" "$out" > "$out/COMMAND.txt"
printf '\n' >> "$out/COMMAND.txt"
uname -a > "$out/KERNEL.txt"
getconf PAGESIZE > "$out/PAGESIZE.txt"
S=$prefix/reference/src/asahi/d3d12/tests/shaders
(cd "$prefix/reference" && sha256sum --strict -c ../reference-inputs.sha256) > "$out/INPUTS.txt"
mkdir "$out/gs"
# Both the callback and the actual registered fd are checked by the helper.
# No physical flag or preload inherited from the caller can widen admission.
set +e
env -u AURORA_D3D12_PHYSICAL_BASELINE -u AURORA_D3D12_PHYSICAL_D32 \
    -u AURORA_D3D12_SHIM_TEXTURE_METADATA -u AURORA_D3D12_SHIM_DEPTH_METADATA \
    ASAHI_SHIM_CHIP=g17p AURORA_D3D12_SHIM_GRAPHICS=1 \
    ASAHI_SHIM_VALIDATE_G17P_STREAM=1 \
    LD_PRELOAD="$prefix/diagnostics/P09/libasahi_noop_drm_shim.so" \
    LD_LIBRARY_PATH="$prefix/lib" \
    D3D12AGX_NATIVE_STAGE_ARTIFACTS="$out/gs" AGX_MESA_DEBUG=shaders \
    timeout 170s "$prefix/diagnostics/P09/d3d12agx_native_stage_certification" \
    "$S/gs_input_vs.dxil" "$S/solid.dxil" "$S/gs_point.dxil" \
    > "$out/stdout" 2> "$out/stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$out/RC.txt"
python3 - "$out" "$rc" <<'PY'
import hashlib, json, pathlib, re, sys
out = pathlib.Path(sys.argv[1])
stages = []
for p in sorted((out / 'gs').glob('stage-*.json')):
    record = json.loads(p.read_text())
    record['name'] = p.with_suffix('.name').read_text().strip()
    record['refusal'] = p.with_suffix('.refusal').read_text().strip()
    stages.append(record)
abis = [json.loads(p.read_text()) for p in sorted((out / 'gs').glob('*.abi.json'))]
stdout = (out / 'stdout').read_text()
ok = (sys.argv[2] == '0' and 'NATIVE STAGE CERTIFICATION: PASS' in stdout
      and len(stages) == 4 and all(s['compiled'] and s['refusal'] == 'none' for s in stages)
      and len(abis) == 3 and all(a['certified'] and not a['physical_admission'] for a in abis))
files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(out.rglob('*')) if p.is_file()}
result = dict(status='PASS' if ok else 'FAIL', corpus='P09-registered-no-submit',
              strict_suite_parity=False, api_pso_creation=re.findall(r'API_G6_PSO_CREATION:.*', stdout), stages=stages, abis=abis, sha256=files)
(out / 'CERTIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
print('P09 REGISTERED NO-SUBMIT:', result['status'], 'stages=', len(stages), 'native_compute=', len(abis))
sys.exit(0 if ok else 1)
PY
