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

echo "[autostart] boot at $(date)" >> "$NAN_WORKSPACE/autostart.log"

# --- default components -------------------------------------------------
# Add your always-on services below (memory, search, ...).
# Registration target: http://127.0.0.1:${NAN_INJECTION_PORT:-8901}/register

launch_service() {
    local name="$1"
    local dir="$NAN_WORKSPACE/services/$name"
    [ -d "$dir" ] || { echo "[autostart] missing service dir: $dir" >> "$NAN_WORKSPACE/autostart.log"; return; }
    mkdir -p "$dir/data"
    uv run --project "$dir" python "$dir/main.py" \
        >> "$dir/data/$name.log" 2>&1 &
    echo "[autostart] launched $name (pid $!)" >> "$NAN_WORKSPACE/autostart.log"
}

launch_service system
launch_service network

MCP_DIR="$NAN_WORKSPACE/services/mcp"
if [ -d "$MCP_DIR" ]; then
    mkdir -p "$MCP_DIR/data"
    uv run --project "$MCP_DIR" python "$MCP_DIR/main.py" \
        >> "$MCP_DIR/data/mcp.log" 2>&1 &
    echo "[autostart] launched mcp (pid $!)" >> "$NAN_WORKSPACE/autostart.log"
fi
