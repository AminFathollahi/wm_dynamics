"""Checks on writing the artifact and on resuming an interrupted run.

The run this module drives takes hours, and both of the ways it can lose that
work are silent until they happen: an estimator that returns a whole
permutation null distribution cannot be written by a per-object numeric
fallback, and a fit read back from an incomplete record would be carried
forward as if it had finished. These assert both directly.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_dominant_latent_identity_and_behaviour_breadth as module  # noqa: E402


@pytest.fixture()
def isolated_record(tmp_path, monkeypatch):
    """The module's completed-fit record redirected to a temporary file."""
    monkeypatch.setattr(module, "CHECKPOINT_PATH", tmp_path / "fits.json")
    monkeypatch.setattr(module, "_COMPLETED_FITS", {})
    return tmp_path / "fits.json"


def test_a_permutation_null_distribution_can_be_written(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "OUTPUT_PATH", tmp_path / "artifact.json")
    module._flush({"unit_count_vs_per_session_correlation": {
        "rho": np.float64(-0.3), "n": np.int64(11), "null": np.arange(10000, dtype=float)}})
    written = json.loads((tmp_path / "artifact.json").read_text())
    assert written["unit_count_vs_per_session_correlation"]["rho"] == -0.3
    assert written["unit_count_vs_per_session_correlation"]["n"] == 11
    assert len(written["unit_count_vs_per_session_correlation"]["null"]) == 10000


def test_a_non_finite_estimate_is_written_as_null_rather_than_an_invalid_token(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "OUTPUT_PATH", tmp_path / "artifact.json")
    module._flush({"slope": float("nan"), "ci_upper": float("inf")})
    assert json.loads((tmp_path / "artifact.json").read_text()) == {"slope": None, "ci_upper": None}


def test_a_fit_already_recorded_as_complete_is_not_recomputed(isolated_record):
    calls = []

    def fit():
        calls.append(1)
        return {"status": "tested", "leading_latent_fractional_rank": 0.5}

    first = module._fit("current_item|210921", fit)
    module._COMPLETED_FITS.clear()
    module._COMPLETED_FITS.update(module._load_completed_fits())
    second = module._fit("current_item|210921", fit)

    assert len(calls) == 1
    assert first == second


def test_every_fit_is_recorded_under_its_own_key(isolated_record):
    module._fit("item_at_trial_minus_1|210921", lambda: {"rank": 0.1})
    module._fit("item_at_trial_minus_2|210921", lambda: {"rank": 0.2})
    module._fit("item_at_trial_minus_1|210927", lambda: {"rank": 0.3})
    recorded = module._load_completed_fits()
    assert set(recorded) == {"item_at_trial_minus_1|210921", "item_at_trial_minus_2|210921",
                             "item_at_trial_minus_1|210927"}
    assert recorded["item_at_trial_minus_2|210921"]["value"] == {"rank": 0.2}


def test_a_truncated_record_is_treated_as_absent_rather_than_trusted(isolated_record):
    module._fit("current_item|210921", lambda: {"rank": 0.5})
    text = isolated_record.read_text()
    isolated_record.write_text(text[: len(text) // 2])
    assert module._load_completed_fits() == {}


def test_a_fit_without_its_completion_flag_is_not_served(isolated_record):
    isolated_record.write_text(json.dumps({
        "current_item|210921": {"complete": True, "value": {"rank": 0.5}},
        "current_item|210927": {"value": {"rank": 0.9}},
    }))
    assert set(module._load_completed_fits()) == {"current_item|210921"}


def test_a_missing_record_is_an_empty_record(isolated_record):
    assert module._load_completed_fits() == {}
