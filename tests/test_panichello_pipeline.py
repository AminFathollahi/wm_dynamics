from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_panichello_pipeline import cue_selective_units_two_stage  # noqa: E402


def test_cue_selective_units_two_stage_detects_planted_tuning():
    rng = np.random.default_rng(1)
    n_trials, n_units, n_bins = 120, 5, 10
    labels = rng.integers(0, 5, size=n_trials)
    counts = rng.poisson(2.0, size=(n_trials, n_units, n_bins)).astype(float)
    counts[labels == 1, 0, :] += 6.0
    out = cue_selective_units_two_stage(counts, labels, rng)
    assert out["procedure"] == "two_stage_trial_block_permutation"
    assert out["n_units"] == n_units
    assert out["n_cue_selective_units"] >= 1


if __name__ == "__main__":
    test_cue_selective_units_two_stage_detects_planted_tuning()
    print("ok")
