#!/usr/bin/env bash
# Autostart: default components launched by the core at boot.
#
# The core stops everything started here when it exits, so launch
# services in the background without detaching, e.g.:
#   python "$NAN_WORKSPACE/services/echo_service.py" \
#       >> "$NAN_WORKSPACE/echo.log" 2>&1 &

set -u

: "${NAN_WORKSPACE:?NAN_WORKSPACE must be set by the core}"

# This script runs inside the sandbox: writes are limited to the
# workspace, so the common package-manager and library caches are
# redirected here for every launch below (a cold sync or download
# would otherwise write under $HOME and fail).
export UV_CACHE_DIR="$NAN_WORKSPACE/.cache/uv"
export PIP_CACHE_DIR="$NAN_WORKSPACE/.cache/pip"
export npm_config_cache="$NAN_WORKSPACE/.cache/npm"
export PLAYWRIGHT_BROWSERS_PATH="$NAN_WORKSPACE/.cache/ms-playwright"
export XDG_CACHE_HOME="$NAN_WORKSPACE/.cache"

LOG="$NAN_WORKSPACE/autostart.log"

note() {
    echo "[autostart] $*" >> "$LOG"
}

note "boot at $(date)"

# --- default components -------------------------------------------------
# Add your always-on services below (memory, search, ...).
# Registration target: http://127.0.0.1:${NAN_INJECTION_PORT:-8901}/register

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

launch "$NAN_WORKSPACE/services/system"
launch "$NAN_WORKSPACE/services/network"
launch "$NAN_WORKSPACE/services/mcp"
