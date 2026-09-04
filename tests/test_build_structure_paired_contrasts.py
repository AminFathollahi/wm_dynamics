"""Regression tests for scripts/build_structure_paired_contrasts.py.
Pure-arithmetic checks on synthetic per-patient
lambda/diffusion dicts and synthetic session artifacts -- no real data.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import build_structure_paired_contrasts as mod  # noqa: E402


class TestBootstrapPairedDifference:
    def test_non_identified_below_minimum_patients(self):
        rng = np.random.default_rng(0)
        result = mod.bootstrap_paired_difference(np.array([1.0]), ["sub-1"], rng)
        assert result["status"] == "non_identified"
        assert result["n_patients_both_structures"] == 1

    def test_recovers_known_constant_difference(self):
        rng = np.random.default_rng(0)
        diffs = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        patient_ids = [f"sub-{i}" for i in range(len(diffs))]
        result = mod.bootstrap_paired_difference(diffs, patient_ids, rng)
        assert result["status"] == "estimable"
        assert result["mean_difference"] == pytest.approx(2.0)
        assert result["a_greater_than_b"] is True
        assert result["patient_bootstrap_ci95"][0] <= 2.0 <= result["patient_bootstrap_ci95"][1]


class TestPairedContrastOnePair:
    def test_restricts_to_patients_present_in_both_regions_per_metric(self):
        rng = np.random.default_rng(0)
        lambda_a = {"sub-1": 4.0, "sub-2": 6.0, "sub-3": 8.0, "sub-4": 5.0}
        diffusion_a = {"sub-1": 8.0, "sub-2": 12.0, "sub-4": 10.0}
        lambda_b = {"sub-1": 2.0, "sub-2": 2.0, "sub-4": 2.0}
        diffusion_b = {"sub-1": 4.0, "sub-2": 4.0, "sub-4": 4.0}
        result = mod.paired_contrast_one_pair(
            "region_a", "region_b", lambda_a, diffusion_a, lambda_b, diffusion_b, rng,
        )
        # lambda: sub-1,sub-2,sub-4 shared -> diffs [2.0, 4.0, 3.0]
        assert result["lambda_s_per_s"]["n_patients_both_structures"] == 3
        assert result["lambda_s_per_s"]["mean_difference"] == pytest.approx(3.0)
        # stationary variance = D/lambda: sub-1 -> 8/4 - 4/2 = 0; sub-2 -> 12/6 - 4/2 = 0; sub-4 -> 10/5 - 4/2 = 0
        assert result["stationary_variance_D_over_lambda"]["n_patients_both_structures"] == 3
        assert result["stationary_variance_D_over_lambda"]["mean_difference"] == pytest.approx(0.0)
        # sub-3 has no diffusion_a entry and is absent from b entirely
        assert "sub-3" not in str(result["diffusion_state_units_sq_per_s"])


class TestPerPatientLambdaDiffusionFromSessions:
    def test_000469_style_session_is_its_own_patient(self):
        sessions = {
            "sub-1": {
                "status": "complete",
                "folds": [
                    {"state_space": {"status": "identifiable", "lambda_rate": 1.0, "diffusion": 2.0}},
                    {"state_space": {"status": "not_identifiable", "lambda_rate": None, "diffusion": None}},
                ],
            },
            "sub-2": {"status": "excluded"},
        }
        lam, dif = mod.per_patient_lambda_diffusion_from_sessions(sessions, mod._session_to_patient_identity)
        assert lam == {"sub-1": 1.0}
        assert dif == {"sub-1": 2.0}

    def test_multi_session_patient_averages_across_sessions(self):
        sessions = {
            "P1_ses-1": {
                "status": "complete",
                "folds": [{"state_space": {"status": "identifiable", "lambda_rate": 2.0, "diffusion": 4.0}}],
            },
            "P1_ses-2": {
                "status": "complete",
                "folds": [{"state_space": {"status": "identifiable", "lambda_rate": 4.0, "diffusion": 8.0}}],
            },
        }
        lam, dif = mod.per_patient_lambda_diffusion_from_sessions(sessions, mod._session_to_patient_by_ses_suffix)
        assert lam == {"P1": pytest.approx(3.0)}
        assert dif == {"P1": pytest.approx(6.0)}

    def test_content_axis_battery_style_row(self):
        sessions = {
            "P1_ses-1": {
                "content_axis_fit": {
                    "status": "complete",
                    "folds": [{"state_space": {"status": "identifiable", "lambda_rate": 5.0, "diffusion": 10.0}}],
                },
            },
        }
        lam, dif = mod.per_patient_lambda_diffusion_from_sessions(sessions, mod._session_to_patient_by_ses_suffix)
        assert lam == {"P1": 5.0}
        assert dif == {"P1": 10.0}
