"""
Make the service directory importable (`main`) while keeping the
tests in their own folder, matching the other services.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from nan_service_kit.config import HUB_ENV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The core exports its injection port to autostart; tests run the
# service directly, so they supply the value the config expects.
os.environ.setdefault(HUB_ENV, "8901")
