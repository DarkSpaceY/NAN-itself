"""
Make the service directory importable (`main`) while keeping the
tests in their own folder, matching the other services.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
