#!/bin/sh
# NAN-itself installer: prepares python dependencies, installs the
# desktop shell and builds the renderer. Run it inside a checkout, or
# pipe it to sh to clone one into ~/nan.
#
# Usage:
#   sh scripts/install.sh
#   curl -fsSL https://raw.githubusercontent.com/DarkSpaceY/NAN-itself/main/scripts/install.sh | sh
#
# Environment overrides:
#   NAN_HOME  clone directory when run outside a checkout (default: ~/nan)
#   NAN_REPO  repository URL    (default: the GitHub repository)
#   NAN_REF   branch or tag     (default: main)

set -eu

REPO_URL="${NAN_REPO:-https://github.com/DarkSpaceY/NAN-itself.git}"
REPO_REF="${NAN_REF:-main}"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

# --- platform ---------------------------------------------------------

case "$(uname -s)" in
    Darwin) ;;
    Linux) ;;
    MINGW*|MSYS*|CYGWIN*)
        die "native Windows is not supported; install WSL2 and run this inside it
       (guide: https://learn.microsoft.com/en-us/windows/wsl/install)" ;;
    *) die "unsupported OS: only macOS and Linux are supported
       (on Windows, install WSL2: https://learn.microsoft.com/en-us/windows/wsl/install)" ;;
esac

command -v git >/dev/null 2>&1 || die "git is required but not installed"
command -v npm >/dev/null 2>&1 || die "Node.js 18+ (npm) is required for the desktop UI"

# --- locate the tree --------------------------------------------------

# Running from a checkout (sh scripts/install.sh) prepares that tree;
# a piped install has no script path and clones into NAN_HOME.
SCRIPT_DIR=""

if [ -f "$0" ]; then
    SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fi

if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/../pyproject.toml" ]; then
    INSTALL_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

    log "using the checkout at $INSTALL_DIR"
else
    INSTALL_DIR="${NAN_HOME:-$HOME/nan}"

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
fi

cd "$INSTALL_DIR"

# --- uv ---------------------------------------------------------------

if command -v uv >/dev/null 2>&1; then
    log "uv $(uv --version | awk '{print $2}') found"
else
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv installation failed"
fi

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

# workspace/services/ ships with the checkout; builtin/services/ is
# user-owned. The mkdirs only make sure both exist if a user removed them.
mkdir -p builtin/services workspace/services

# --- desktop ----------------------------------------------------------

log "installing the desktop shell and building the renderer (electron/)"
(cd electron && npm install --no-fund --no-audit && npm run build)

# --- done -------------------------------------------------------------

CONFIG="$INSTALL_DIR/config/nan.yaml"

log "installed at $INSTALL_DIR"
printf '
Next steps:
  1. Fill in the active model in %s (llm: provider / api_key / model).
  2. Review %s/nan/persona.md.
  3. Run:
       cd %s && uv run nan
  The NAN window opens once the gateway is up; services reach it on
  http://127.0.0.1:8765.
' "$CONFIG" "$INSTALL_DIR" "$INSTALL_DIR"
