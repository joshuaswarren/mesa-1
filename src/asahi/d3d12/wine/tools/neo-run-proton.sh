#!/bin/bash
# Copyright 2026 Aurora Silicon
# SPDX-License-Identifier: MIT
#
# Run a Windows executable under a Proton compatibility tool in the logged-in
# user's graphical session (outside Steam), with a dedicated prefix and a
# bounded timeout. Meant to be invoked as root through aurora-ctl.
#
# Usage: neo-run-proton.sh <tool-name> <prefix-dir> <log-dir> <timeout-s> <exe> [args...]
# Env:   STEAM_ROOT (default /home/user/.local/share/Aurora-Steam-ARM64-20261005)
#        RUN_USER (default user), EXTRA_ENV (space separated VAR=VALUE list)
#        PROTON_VERB (default run): "run" is what Steam does (prefix setup incl.
#        the tool's dxvk/vkd3d-proton install, then the steam.exe shim);
#        "runinprefix" skips the prefix setup, so under the vkd3d-proton tool
#        it silently used Wine's own d3d12 instead of vkd3d-proton.
set -u
tool=$1; prefix=$2; logdir=$3; timeout_s=$4; shift 4
steam_root=${STEAM_ROOT:-/home/user/.local/share/Aurora-Steam-ARM64-20261005}
run_user=${RUN_USER:-user}
tooldir=$steam_root/compatibilitytools.d/$tool
[[ -x $tooldir/aurora-proton ]] || { echo "no tool $tooldir" >&2; exit 2; }
mkdir -p "$logdir" "$prefix"
chown "$run_user" "$logdir" "$prefix"

# Session environment of the user's compositor (gnome-shell or sway).
sess_pid=$(pgrep -u "$run_user" -x gnome-shell | head -1)
[[ -n $sess_pid ]] || sess_pid=$(pgrep -u "$run_user" -x sway | head -1)
[[ -n $sess_pid ]] || { echo "no graphical session for $run_user" >&2; exit 3; }
sess_env=()
declare -A have=()
for v in DISPLAY WAYLAND_DISPLAY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS XAUTHORITY XDG_SESSION_TYPE; do
   val=$(tr '\0' '\n' < "/proc/$sess_pid/environ" | sed -n "s/^$v=//p" | head -1)
   [[ -n $val ]] && { sess_env+=("$v=$val"); have[$v]=$val; }
done
# gnome-shell's own environment names neither its Wayland socket nor its
# Xwayland display (it creates both); derive them from the runtime dir and
# the Xwayland command line so Wine's x11 driver can open a window.
rt=${have[XDG_RUNTIME_DIR]:-/run/user/$(id -u "$run_user")}
if [[ -z ${have[WAYLAND_DISPLAY]:-} && -S $rt/wayland-0 ]]; then
   sess_env+=("WAYLAND_DISPLAY=wayland-0")
fi
if [[ -z ${have[DISPLAY]:-} ]]; then
   xw=$(pgrep -u "$run_user" -a -x Xwayland | head -1)
   disp=$(sed -n 's/.*Xwayland \(:[0-9]*\).*/\1/p' <<< "$xw")
   auth=$(sed -n 's/.*-auth \([^ ]*\).*/\1/p' <<< "$xw")
   [[ -n $disp ]] && sess_env+=("DISPLAY=$disp")
   [[ -n $auth && -z ${have[XAUTHORITY]:-} ]] && sess_env+=("XAUTHORITY=$auth")
fi

exe_dir=$(dirname "$1")
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log=$logdir/run-$stamp.log
{
   echo "tool=$tool prefix=$prefix exe=$1 args=${*:2} timeout=$timeout_s verb=${PROTON_VERB:-run}"
   echo "session env: ${sess_env[*]}"
} > "$log"

# Everything after the verb is the command line; the exe directory is only
# the cwd (an extra leading path made wine open an Explorer window on it).
timeout --signal=TERM --kill-after=10s "$timeout_s" \
   runuser -u "$run_user" -- env -i HOME=/home/$run_user USER=$run_user LOGNAME=$run_user \
      PATH=/usr/local/bin:/usr/bin:/bin \
      "${sess_env[@]}" \
      STEAM_COMPAT_CLIENT_INSTALL_PATH="$steam_root" \
      STEAM_COMPAT_DATA_PATH="$prefix" \
      STEAM_COMPAT_INSTALL_PATH="$exe_dir" \
      STEAM_COMPAT_APP_ID=0 SteamAppId=0 SteamGameId=0 PROTON_VERB="${PROTON_VERB:-run}" \
      PROTON_LOG=1 PROTON_LOG_DIR="$logdir" \
      WINEDEBUG=${WINEDEBUG:-err+all,fixme-all,+loaddll} \
      ${EXTRA_ENV:-} \
      bash -c 'cd "$1" && shift && exec "$0" "$PROTON_VERB" "$@"' "$tooldir/aurora-proton" "$exe_dir" "$@" \
      >> "$log" 2>&1
rc=$?
echo "exit=$rc" >> "$log"
echo "$log"
exit $rc
