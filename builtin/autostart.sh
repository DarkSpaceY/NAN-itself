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

LOG="$HERE/autostart.log"

note() {
    echo "[autostart] $*" >> "$LOG"
}

note "boot at $(date)"

# The launcher creates only logs/: <name>.boot.log holds what it
# printed before loguru took over -- uv failures, import errors,
# uvicorn's warnings -- and <service_id>.log is the log the service
# itself rotates. Any state directory is the service's own choice.
launch() {
    local dir="$1"
    local name
    name="$(basename "$dir")"

    [ -d "$dir" ] || {
        note "missing service dir: $dir"
        return
    }

    mkdir -p "$dir/logs"

    local boot_log="$dir/logs/$name.boot.log"

    # Only startup output lands here, but a crash loop can pile it up;
    # keep one previous generation.
    if [ -f "$boot_log" ] && [ "$(wc -c < "$boot_log")" -gt 1048576 ]; then
        mv -f "$boot_log" "$boot_log.1"
    fi

    uv run --project "$dir" python "$dir/main.py" >> "$boot_log" 2>&1 &

    note "launched $name (pid $!)"
}

# --- builtin components ------------------------------------------------
launch "$HERE/services/record"
launch "$HERE/services/vision"
