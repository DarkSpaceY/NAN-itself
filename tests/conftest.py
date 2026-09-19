
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = PROJECT_ROOT.parent

# Belt-and-suspenders: make the project parent importable when pytest
# is launched from elsewhere. `nan_itself` itself resolves through the
# editable install, not through this entry (src layout).
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))