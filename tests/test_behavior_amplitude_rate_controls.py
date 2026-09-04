"""Tests for scripts/run_behavior_amplitude_rate_controls.py's pooling
helper and its pre-declared branch: does the leading-component gain's
correlation with trial outcome survive controlling for total spike count?"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_behavior_amplitude_rate_controls import _pool  # noqa: E402


def _session(session_id: str, r: float) -> dict:
    return {"session": session_id, "status": "computed",
            "analysis": {"control_spike_count": {"gain_given_spike_count": {"status": "computed", "r": r}}}}


def test_pool_reuses_slope_across_sessions_test_and_reports_significance():
    rng = np.random.default_rng(0)
    sessions = [_session(f"s{i}", r=-0.2 + rng.normal(scale=0.02)) for i in range(11)]
    result = _pool(sessions, ("control_spike_count", "gain_given_spike_count"))
    assert result["status"] == "tested"
    assert result["n_sessions"] == 11
    assert result["mean_value"] < 0
    assert result["significant"] is True


def test_pool_skips_sessions_missing_the_requested_field():
    # _pool is called (in main()) only on already-status=='computed' sessions, so every entry here has
    # an "analysis" key; s1 is missing the specific nested field this call asks for (e.g. a session where
    # trial_amplitude_covariates itself did not compute). Only s0 is usable, which is below
    # slope_across_sessions_test's own power floor -- the point of this test is that the unusable session
    # is not silently counted in as a zero or dropped some other way that changes n_sessions.
    sessions = [
        _session("s0", r=-0.2),
        {"session": "s1", "status": "computed", "analysis": {"control_spike_count": {}}},
    ]
    result = _pool(sessions, ("control_spike_count", "gain_given_spike_count"))
    assert result["status"] == "underpowered_by_construction"
    assert result["n_sessions"] == 1


def test_pool_with_no_matching_sessions_is_not_computed():
    result = _pool([], ("control_spike_count", "gain_given_spike_count"))
    assert result["status"] == "not_computed"
