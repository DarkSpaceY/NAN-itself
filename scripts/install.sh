#!/bin/sh
# NAN-itself bootstrap installer: clones the repository, installs
# dependencies, builds the web UI and prepares the directory layout.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/DarkSpaceY/NAN-itself/main/scripts/install.sh | sh
#   or: sh install.sh
#
# Environment overrides:
#   NAN_HOME  install directory (default: ~/nan)
#   NAN_REPO  repository URL    (default: the GitHub repository)
#   NAN_REF   branch or tag     (default: main)

set -eu

REPO_URL="${NAN_REPO:-https://github.com/DarkSpaceY/NAN-itself.git}"
REPO_REF="${NAN_REF:-main}"
INSTALL_DIR="${NAN_HOME:-$HOME/nan}"

log() { printf '\n==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- platform ---------------------------------------------------------

case "$(uname -s)" in
    Darwin) OS=macos ;;
    Linux) OS=linux ;;
    MINGW*|MSYS*|CYGWIN*)
        die "native Windows is not supported; install WSL2 and run this inside it
       (guide: https://learn.microsoft.com/en-us/windows/wsl/install)" ;;
    *) die "unsupported OS: only macOS and Linux are supported
       (on Windows, install WSL2: https://learn.microsoft.com/en-us/windows/wsl/install)" ;;
esac

command -v git >/dev/null 2>&1 || die "git is required but not installed"

# --- uv ---------------------------------------------------------------

if command -v uv >/dev/null 2>&1; then
    log "uv $(uv --version | awk '{print $2}') found"
else
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv installation failed"
fi

# --- clone ------------------------------------------------------------

# An existing checkout is used as it is: this installer never pulls,
# resets or otherwise overwrites local work (config/nan.yaml holds the
# user's key and builtin/ is user-owned). Updating is a deliberate git
# operation the user performs.
if [ -d "$INSTALL_DIR/.git" ]; then
    log "using the existing checkout at $INSTALL_DIR (left untouched)"
else
    log "cloning NAN-itself ($REPO_REF) into $INSTALL_DIR"
    git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# --- configuration ----------------------------------------------------

# The live config is git-ignored -- it holds the user's key -- so a
# fresh checkout only carries the template. An existing config is
# never overwritten.
if [ ! -f config/nan.yaml ]; then
    log "creating config/nan.yaml from config/nan.yaml.example"
    cp config/nan.yaml.example config/nan.yaml
fi

# --- python dependencies ---------------------------------------------

log "syncing python dependencies (uv fetches Python 3.12+ if needed)"
uv sync

# --- directory layout -------------------------------------------------

# builtin/services/ (record, vision) and workspace/services/ come with
# the checkout; the mkdirs only make sure the two territories exist if
# a user removed them.
mkdir -p builtin/services workspace/services

# --- web UI -----------------------------------------------------------

if command -v npm >/dev/null 2>&1; then
    log "building the web UI (frontend/dist)"
    (cd frontend && npm install --no-fund --no-audit && npm run build)
else
    warn "npm not found: skipping the web UI build"
    warn "the gateway starts without it, serving only /ws and /surfaces"
    warn "install Node.js 18+ and run: cd frontend && npm install && npm run build"
fi

# --- done -------------------------------------------------------------

CONFIG="$INSTALL_DIR/config/nan.yaml"

log "installed at $INSTALL_DIR"
printf '
Next steps:
  1. Fill in the active model in %s (llm: provider / api_key / model).
  2. Review %s/nan/persona.md.
  3. Run:
       cd %s && uv run nan
  The web UI comes up on http://127.0.0.1:8765 once frontend/dist exists.
' "$CONFIG" "$INSTALL_DIR" "$INSTALL_DIR"
