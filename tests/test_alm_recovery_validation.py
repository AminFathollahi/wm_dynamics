import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_alm_recovery_validation.py"
SPEC = importlib.util.spec_from_file_location("run_alm_recovery_validation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_recovery_fit_recovers_planted_rate():
    time = np.arange(20) * 0.1 + 0.05
    deficit = 0.1 + 1.2 * np.exp(-3.0 * np.maximum(time - 1.0, 0.0))
    estimate = MODULE.fit_recovery_rate(time, deficit)
    assert estimate["status"] == "complete"
    assert estimate["lambda_rate"] == pytest.approx(3.0, rel=0.05)
