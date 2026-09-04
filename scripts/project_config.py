"""Make the shared project configuration importable by directly-run scripts."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.project_config import *  # noqa: F401,F403,E402
