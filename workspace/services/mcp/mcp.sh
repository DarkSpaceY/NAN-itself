#!/bin/sh
# Thin launcher for the agent-facing CLI. The implementation is
# mcp_cli.py; the service's port is read from the sibling config.yaml.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)

exec python3 "$HERE/mcp_cli.py" "$@"
