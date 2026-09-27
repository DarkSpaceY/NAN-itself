
from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent

# Belt-and-suspenders: make the project parent importable when pytest
# is launched from elsewhere. `nan_itself` itself resolves through the
# editable install, not through this entry (src layout).
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


@pytest.fixture(autouse=True)
def _isolated_persona(tmp_path, monkeypatch):
    """
    Redirect the Agent's persona path to a per-test temp file, so no
    test reads or writes the repository's workspace/persona.md.
    """
    from nan_itself.utils import paths

    path = tmp_path / "persona.md"
    path.write_text("persona", encoding="utf-8")
    monkeypatch.setattr(paths, "persona_path", lambda: path)
    return path