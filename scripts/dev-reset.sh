#!/bin/sh
# Dev reset: return this checkout to the state of a fresh clone.
#
# Removes runtime state only -- no source, no config, no workspace
# home, no package caches. The next boot recreates what it needs.
#
#   workspace/.nan        job logs and status files
#   workspace/.tmp        scratch files
#   workspace/logs        autostart records
#   snapshots/            per-agent recovery state
#   services/*/logs       service and boot logs
#   services/plan/data    the goal tree
#   services/segv/data    the memory archive
#
# Usage:
#   sh scripts/dev-reset.sh          # dry run: list what would go
#   sh scripts/dev-reset.sh --yes    # delete it
#
# Refuses to run while the core is up: deleting snapshots or the segv
# archive under a live agent loses state that nothing can rebuild.

set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)

WORKSPACE="$ROOT/workspace"

# Keep in sync with config/nan.yaml (gateway.port / injection.port).
GATEWAY_PORT=8765
INJECTION_PORT=8901

log() { printf '%s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
    "") ;;
    --yes|-y) assume_yes=1 ;;
    -h|--help)
        sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    *) die "unknown argument: ${1} (use --yes to delete)" ;;
esac

[ -d "$ROOT" ] || die "cannot locate the checkout"
[ -d "$WORKSPACE" ] || die "missing workspace directory: $WORKSPACE"

listening() {
    command -v lsof >/dev/null 2>&1 || return 1

    lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

if listening "$GATEWAY_PORT" || listening "$INJECTION_PORT"; then
    die "NAN is running (gateway :$GATEWAY_PORT / injection :$INJECTION_PORT); stop it first"
fi

set -- "$WORKSPACE/.nan" "$WORKSPACE/.tmp" "$WORKSPACE/logs" "$ROOT/snapshots"

for dir in "$WORKSPACE"/services/*/logs "$WORKSPACE"/services/plan/data "$WORKSPACE"/services/segv/data; do
    [ -e "$dir" ] || continue

    set -- "$@" "$dir"
done

# rm -rf has no undo; keep every target inside the checkout.
for dir in "$@"; do
    case "$dir" in
        "$ROOT"/*) ;;
        *) die "refusing to touch a path outside the checkout: $dir" ;;
    esac
done

if [ "${assume_yes:-0}" -eq 0 ]; then
    log "Would remove:"

    found=0

    for dir in "$@"; do
        [ -e "$dir" ] || continue

        found=1

        log "  ${dir#"$ROOT"/}  ($(du -sh "$dir" | cut -f1))"
    done

    [ "$found" -eq 1 ] || log "  (nothing; already clean)"

    log ""
    log "Nothing was deleted. Re-run with --yes to apply."

    exit 0
fi

for dir in "$@"; do
    [ -e "$dir" ] || continue

    rm -rf "$dir"

    log "removed ${dir#"$ROOT"/}"
done

log "done: runtime state cleared"
