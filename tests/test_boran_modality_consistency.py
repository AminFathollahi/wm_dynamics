"""Focused regression tests for the Boran spike-vs-iEEG modality-consistency arm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_boran_modality_consistency import (  # noqa: E402
    _modality_pair,
    lfp_maintenance_tensor,
    session_seed,
)

BORAN_NWB = Path(
    "/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory"
    "/data/000574/sub-01/sub-01_ses-01.nwb"
)


def _identifiable_estimate(lambda_rate: float, ci=(0.5, 2.0)):
    return {"status": "identifiable", "lambda_rate": lambda_rate, "lambda_ci": list(ci)}


def _not_estimable():
    return {"status": "not_estimable", "lambda_rate": None, "lambda_ci": None}


class TestSessionSeed:
    def test_deterministic(self):
        assert session_seed("sub-01_ses-01") == session_seed("sub-01_ses-01")

    def test_distinguishes_subject_and_session(self):
        a = session_seed("sub-01_ses-01")
        b = session_seed("sub-01_ses-02")
        c = session_seed("sub-02_ses-01")
        assert len({a, b, c}) == 3


class TestModalityPair:
    def test_both_identifiable_gives_diff_and_log_ratio(self):
        spike_fit = {"moment": _identifiable_estimate(2.0, ci=(1.5, 2.5))}
        lfp_fit = {"moment": _identifiable_estimate(1.0, ci=(0.6, 1.4))}
        pair = _modality_pair(spike_fit, lfp_fit, "moment")
        assert pair is not None
        assert pair["spike_lambda"] == 2.0 and pair["lfp_lambda"] == 1.0
        assert pair["diff_spike_minus_lfp"] == pytest.approx(1.0)
        assert pair["log_ratio_spike_over_lfp"] == pytest.approx(np.log(2.0))
        # [1.5,2.5] and [0.6,1.4] do not overlap
        assert pair["ci_overlap"] is False

    def test_overlapping_ci_flagged_true(self):
        spike_fit = {"moment": _identifiable_estimate(2.0, ci=(1.0, 3.0))}
        lfp_fit = {"moment": _identifiable_estimate(1.5, ci=(1.2, 2.2))}
        pair = _modality_pair(spike_fit, lfp_fit, "moment")
        assert pair["ci_overlap"] is True

    def test_either_not_identifiable_returns_none(self):
        spike_fit = {"moment": _identifiable_estimate(2.0)}
        lfp_fit = {"moment": _not_estimable()}
        assert _modality_pair(spike_fit, lfp_fit, "moment") is None
        assert _modality_pair(lfp_fit, spike_fit, "moment") is None


@pytest.mark.skipif(not BORAN_NWB.exists(), reason="DANDI 000574 data not on external drive")
class TestLfpMaintenanceTensor:
    def test_output_shape_and_finite(self):
        from preprocessing import load_boran_nwb

        ieeg = load_boran_nwb(str(BORAN_NWB), signal="ieeg", reject_channels=True, mains_hz=50.0)
        n_trials = ieeg["epochs"].shape[0]
        trial_mask = np.zeros(n_trials, dtype=bool)
        trial_mask[:6] = True  # small subset -- shape/finiteness check, not a scientific run
        n_bins = 30
        tensor = lfp_maintenance_tensor(ieeg, trial_mask, n_bins=n_bins)
        assert tensor.shape[0] == 6
        assert tensor.shape[2] == n_bins
        assert tensor.shape[1] >= 1  # at least one bipolar/CAR-orphan channel survived
        assert np.all(np.isfinite(tensor))


@pytest.mark.skipif(not BORAN_NWB.exists(), reason="DANDI 000574 data not on external drive")
class TestAnalyzeSessionSmoke:
    def test_one_real_session_runs_end_to_end(self):
        from run_boran_modality_consistency import analyze_session

        result = analyze_session(BORAN_NWB, seed=20260801)
        assert result["status"] in ("complete", "excluded")
        if result["status"] == "complete":
            assert result["n_units_qc"] >= 1
            assert result["n_channels_bipolar"] >= 1
            assert len(result["folds"]) == 5
            for fold in result["folds"]:
                assert fold["spike"]["state_space"]["status"] in (
                    "identifiable", "not_identifiable", "unconfined", "nonconverged", "not_estimable",
                )
                assert fold["lfp"]["state_space"]["status"] in (
                    "identifiable", "not_identifiable", "unconfined", "nonconverged", "not_estimable",
                )
