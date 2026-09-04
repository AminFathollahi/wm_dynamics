import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_wolff_corrected_analysis.py"
SPEC = importlib.util.spec_from_file_location("run_wolff_corrected_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_circular_mahalanobis_score_recovers_planted_orientation_code():
    rng = np.random.default_rng(9)
    labels = np.repeat(np.arange(12), 8)
    theta = 2.0 * np.pi * labels / 12.0 - np.pi
    patterns = np.column_stack((np.cos(theta), np.sin(theta), np.cos(2 * theta), np.sin(2 * theta)))
    data = np.repeat(patterns[:, :, None], 5, axis=2)
    data += rng.normal(scale=0.35, size=data.shape)
    score, null = MODULE.circular_mahalanobis_scores(data, theta, seed=12)
    assert float(np.mean(score)) > float(np.mean(null)) + 0.25
