#!/usr/bin/env bash
# Run a command with no network access. Linux only (CI).
#
# The command runs in a fresh network namespace holding only a loopback
# interface, as the invoking user, with the invoking environment. A step that
# tries to reach the network fails loudly instead of quietly depending on it.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "offline.sh: needs Linux network namespaces; run the command directly" >&2
  exit 2
fi

# `sudo --preserve-env` keeps the caller's environment except PATH (secure_path)
# and possibly HOME, so both are passed through explicitly. setpriv drops back
# to the caller's uid/gid inside the namespace.
exec sudo --preserve-env unshare --net -- bash -c '
  set -euo pipefail
  ip link set lo up
  uid=$1 gid=$2 path=$3 home=$4
  shift 4
  exec setpriv --reuid="$uid" --regid="$gid" --init-groups \
    env PATH="$path" HOME="$home" "$@"
' offline.sh "$(id -u)" "$(id -g)" "$PATH" "$HOME" "$@"
