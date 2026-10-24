#!/usr/bin/env bash
# Autostart: default components launched by the core at boot.
# This script lives OUTSIDE the workspace and is NOT for agents
# to edit: it runs with the user's authority, so changing it is
# a privilege escalation vector. It is USER-OWNED -- the agent
# must never write it (an agent-writable autostart is a
# privilege escalation vector).
#
# Each line should launch a service in the background, e.g.:
#   nohup python "$NAN_WORKSPACE/services/echo_service.py" \
#       >> "$NAN_WORKSPACE/echo.log" 2>&1 &

set -u

: "${NAN_CORE_HOME:?NAN_CORE_HOME must be set by the core}"
: "${NAN_WORKSPACE:?NAN_WORKSPACE must be set by the core}"

echo "[autostart] boot at $(date)" >> "$NAN_WORKSPACE/autostart.log"

# --- default components -------------------------------------------------
# Add your always-on services below (memory, search, ...).
# Registration target: http://127.0.0.1:${NAN_INJECTION_PORT:-8901}/register

launch_service() {
    local name="$1"
    local dir="$NAN_WORKSPACE/services/$name"
    [ -d "$dir" ] || { echo "[autostart] missing service dir: $dir" >> "$NAN_WORKSPACE/autostart.log"; return; }
    nohup uv run --project "$dir" python "$dir/main.py" \
        >> "$NAN_WORKSPACE/$name.log" 2>&1 &
    echo "[autostart] launched $name (pid $!)" >> "$NAN_WORKSPACE/autostart.log"
}

launch_service system
launch_service network
