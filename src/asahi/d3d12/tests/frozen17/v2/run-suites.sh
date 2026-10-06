#!/bin/bash
# Run every d3d12agx GPU acceptance suite with its correct shader arguments.
#
# The argument lists are the part that is easy to get wrong -- each test takes
# a different set of shaders, and a wrong name makes a test print usage and
# exit 0, which reads as a pass in a loop that only checks the last line.
# Keep this file as the single source of truth. A suite that produces no
# verdict line at all is a FAILURE here, not a silent pass.
#
# Usage: gpusuite.sh [path-to-mesa-build]
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
S=$HERE/shaders
B=${1:-$HERE/../../../../_build/src/asahi/d3d12}

run() {
   local name=$1; shift
   attempts=$((attempts+1))
   local bin=$B/d3d12agx_$name
   [ -x "$bin" ] || { printf "%-18s MISSING BINARY\n" "$name"; return 1; }
   for f in "$@"; do
      [ -r "$f" ] || { printf "%-18s MISSING SHADER %s\n" "$name" "$(basename "$f")"; return 1; }
   done
   local out rc last
   out=$(timeout 180 "$bin" "$@" 2>&1); rc=$?
   last=$(echo "$out" | grep -E "ALL PASS|FAILURES" | tail -1)
   if [ -z "$last" ]; then
      printf "%-18s NO VERDICT (exit %d)\n" "$name" "$rc"
      echo "$out" | tail -3 | sed 's/^/                   | /'
      return 1
   fi
   printf "%-18s %s\n" "$name" "$last"
   echo "$out" | grep "^FAIL" | sed 's/^/                   | /'
   if [ $rc -ne 0 ] || [[ "$last" != *"ALL PASS"* ]] ||
      echo "$out" | grep -Eq "^FAIL|BLOCKED"; then
      return 1
   fi
   passes=$((passes+1))
   return 0
}

fails=0
attempts=0
passes=0
required=17

# The G1-G4 milestone smokes. They were NOT in this runner, so when the
# descriptor grew from 32 to 40 bytes g3_smoke failed unnoticed for a
# day. A suite that nothing runs is a suite that is not green.
run g1_smoke                                                                       || fails=$((fails+1))
run g2_smoke                                                                       || fails=$((fails+1))
run g3_smoke                                                                       || fails=$((fails+1))
run g4_smoke        "$S/cs.dxil"                                                   || fails=$((fails+1))

run g4_triangle     "$S/tri.dxil" "$S/solid.dxil"                                  || fails=$((fails+1))
run g5_depth        "$S/tri.dxil" "$S/solid.dxil" "$S/tri_depth.dxil" \
                    "$S/solid_g.dxil" "$S/tri_depth_cw.dxil"                       || fails=$((fails+1))
run g5_descriptors  "$S/res_probe.dxil" "$S/rs_table.rs" "$S/rs_mixed.rs" \
                    "$S/rs_descriptors.rs" "$S/space_probe.dxil"                   || fails=$((fails+1))
run g5_atomics      "$S/atomic_probe.dxil"                                         || fails=$((fails+1))
run g5_control_flow "$S/cf_probe.dxil"                                             || fails=$((fails+1))
run g5_wave         "$S/wave_probe.dxil"                                           || fails=$((fails+1))
run g5_texture      "$S/tex_probe.dxil"                                            || fails=$((fails+1))

# G6 runs against probe shaders GENERATED from NIR by d3d12agx_make_probe
# and committed next to it, so it needs neither dxc (absent on aarch64)
# nor the LGPL corpus.
run g6_geometry "$S/gs_input_vs.dxil" "$S/solid_link.dxil" "$S/gs_point.dxil" \
   || fails=$((fails+1))

run g7_tessellation "$S/gs_input_vs.dxil" "$S/solid.dxil" "$S/tess_hs.dxil" \
   "$S/tess_ds.dxil" || fails=$((fails+1))

# G8 and G9 run against generated probes too. Mesh and amplification
# support in nir_to_dxil was written for exactly this: before it, no mesh
# probe could be built and both suites were gated on the LGPL corpus.
run g8_mesh "$S/mesh_quad.dxil" "$S/solid.dxil" || fails=$((fails+1))

run g9_amplification "$S/as_dispatch.dxil" "$S/mesh_quad.dxil" \
   "$S/solid.dxil" || fails=$((fails+1))

# G10 is the first suite whose vertex data comes from a BOUND BUFFER
# rather than from SV_VertexID. It draws the same triangle twice -- once
# each way -- and compares.
run g10_vertex_buffers "$S/tri.dxil" "$S/solid.dxil" "$S/vb_vs.dxil" \
   "$S/vb_ps.dxil" || fails=$((fails+1))

# G11 closes what G10 left unproven: indexed draws through a bound index
# buffer, and per-instance attributes.
run g11_indexed_instanced "$S/tri.dxil" "$S/solid.dxil" "$S/vb_vs.dxil" \
   "$S/vb_ps.dxil" "$S/vb_inst_vs.dxil" || fails=$((fails+1))

echo
# All seventeen suites are required, including texture. Missing input,
# blocked execution and a zero exit code without ALL PASS remain failures.
if [ $attempts -ne $required ] || [ $passes -ne $required ] || [ $fails -ne 0 ]; then
   echo "GPU SUITES: $passes/$required passed, $attempts attempted; not a full pass"
   [ $fails -ne 0 ] || fails=1
else
   echo "GPU SUITES: all green ($passes/$required required suites)"
fi
exit $fails
