"""The shipped boot script only names services that are in the tree."""

from __future__ import annotations

import re
from pathlib import Path


def test_builtin_autostart_only_launches_services_that_exist() -> None:
    root = Path(__file__).resolve().parents[2]
    script = (root / "builtin" / "autostart.sh").read_text(encoding="utf-8")
    names = re.findall(r'launch "\$HERE/services/([^"]+)"', script)

    missing = [name for name in names if not (root / "builtin" / "services" / name).is_dir()]

    assert missing == []
