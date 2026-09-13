#!/bin/sh
# SPDX-License-Identifier: MIT
#
# A/B acceptance wrapper for f38's exact Hypr/Zink sampled-texture failure.
# The shader-visible image remains an ordinary descriptor (set 2, binding
# 128).  The diagnostic changes only Honeykrisp's backing for Zink's pushed
# driver-uniform set 0.

set -u

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$test_dir/../../.." && pwd)
gl_build=$(CDPATH= cd -- "${1:-$repo_dir/_build-hypr-desktop}" && pwd)
vk_build=$(CDPATH= cd -- "${2:-$repo_dir/_build-full}" && pwd)
run_hardware=${3:-}
log_dir=${4:-$gl_build/src/asahi/gl-tests/f39-push-backing-logs}
f38=$test_dir/run-f38-hypr-egl-dmabuf.sh

test -x "$f38" || {
   echo "missing executable f38 runner: $f38" >&2
   exit 2
}

case "$run_hardware" in
   "") hardware_arg= ;;
   --hardware) hardware_arg=--hardware ;;
   *)
      echo "unknown option: $run_hardware (expected --hardware)" >&2
      exit 2
      ;;
esac

mkdir -p "$log_dir"

run_one()
{
   name=$1
   dedicated=$2
   log=$log_dir/$name.log

   echo "f39 $name: HK_G17P_PUSH_DESCRIPTOR_DEDICATED_BO=$dedicated"
   env \
      MESA_SHADER_CACHE_DISABLE=true \
      GALLIUM_THREAD=0 \
      ZINK_DESCRIPTORS=lazy \
      ZINK_DEBUG=noreorder \
      HK_G17P_LAUNCH_TRACE=1 \
      HK_G17P_DUMP_WRITE_RANGES=1 \
      HK_G17P_PUSH_DESCRIPTOR_DEDICATED_BO="$dedicated" \
      "$f38" "$gl_build" "$vk_build" $hardware_arg >"$log" 2>&1
   rc=$?
   sed -n \
      -e '/^GL08 /p' \
      -e '/^G17P push-descriptor-set:/s/ bytes=.*//p' \
      -e '/label=G17P pushed descriptor diagnostic/p' "$log"
   echo "f39 $name rc=$rc log=$log"
   return "$rc"
}

verify_candidate_trace()
{
   log=$1
   bases=$(sed -n 's/^G17P push-descriptor-set: base=\([^ ]*\) .*/\1/p' \
      "$log")
   test -n "$bases" || {
      echo "f39 FAIL: candidate has no pushed-descriptor trace" >&2
      return 1
   }

   for base in $bases; do
      grep -q "label=G17P pushed descriptor diagnostic va=0x$base " \
         "$log" || {
         echo "f39 FAIL: pushed set $base is not rooted in a dedicated BO" >&2
         return 1
      }
   done
}

if [ "$run_hardware" != --hardware ]; then
   # The no-op shim validates command construction but skips pixel truth.  It
   # cannot provide the control failure, so only smoke-test the candidate.
   run_one candidate 1
   candidate_rc=$?
   if [ "$candidate_rc" -ne 0 ]; then
      echo "f39 FAIL: host candidate smoke test returned $candidate_rc" >&2
      exit 1
   fi
   verify_candidate_trace "$log_dir/candidate.log" || exit 1
   echo "f39 PASS: host command-state smoke test; hardware pixels skipped"
   exit 0
fi

run_one control 0
control_rc=$?
run_one candidate 1
candidate_rc=$?

verify_candidate_trace "$log_dir/candidate.log" || exit 1

if [ "$candidate_rc" -ne 0 ]; then
   echo "f39 FAIL: dedicated backing did not repair all f38 sample phases" >&2
   exit 1
fi

if [ "$control_rc" -eq 0 ]; then
   echo "f39 INCONCLUSIVE: control no longer reproduces; candidate is non-regressing"
   exit 3
fi

echo "f39 PASS: control reproduced and dedicated backing repaired exact f38 pixels"
