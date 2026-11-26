"""
The agent's own home directory.

Everything a tool resolves through ``~`` lands under
``<workspace>/home`` instead of the user's account: caches, configs
and dotfiles stay inside the agent's territory, and the agent starts
with no inherited identity (no git identity, no SSH keys, no package
registry credentials). Moving the workspace therefore moves the
whole environment. Temporary files are excluded: ``TMPDIR`` keeps
pointing at the system location, so they expire instead of becoming
part of the workspace.
"""

from __future__ import annotations

from pathlib import Path


def home_environment(workspace: Path) -> dict[str, str]:
    """
    The variables that point ``~`` and the XDG directories at
    ``<workspace>/home``. The home directory is created on demand.
    """
    home = (workspace / "home").resolve()

    return {
        "HOME": str(home),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(home / ".local" / "share"),
        "XDG_STATE_HOME": str(home / ".local" / "state"),
    }
