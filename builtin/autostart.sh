#!/usr/bin/env bash
# Autostart: user-owned components launched by the core at boot.
#
# This script runs OUTSIDE the execution sandbox with the user's
# authority. That is deliberate: builtin is read-only inside the
# sandbox, and builtin services own their data directory.
#
# The core owns the lifetime of everything started here: it stops
# this script's process group on exit, so nothing needs to detach.
#
# The agent never writes this file; agents add their own components
# to workspace/autostart.sh instead.

set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Caches stay inside the repository, next to the tree that owns them;
# a boot never writes into $HOME.
export UV_CACHE_DIR="$HERE/.cache/uv"
export PIP_CACHE_DIR="$HERE/.cache/pip"

launch() {
    local dir="$1"
    local name
    name="$(basename "$dir")"

    [ -d "$dir" ] || {
        echo "[builtin-autostart] missing service dir: $dir" >&2
        return
    }

    mkdir -p "$dir/data"

    uv run --project "$dir" python "$dir/main.py" \
        >> "$dir/data/$name.log" 2>&1 &
}

# --- builtin components ------------------------------------------------
launch "$HERE/services/record"
launch "$HERE/services/vision"
