"""Tests for scripts/run_ctg_offdiagonal_temporal_stability.py -- the pieces
that could silently break this analysis without erroring: subject-id
extraction (session suffixes must not merge into the cluster key), the
cluster-bootstrap CI/p-value/mdd arithmetic, and the numbers reported for
each cell."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_ctg_offdiagonal_temporal_stability import (  # noqa: E402
    _subject_id, _cluster_bootstrap, _apply_fdr, _independent_group_summary,
    INDEPENDENT_PATIENT_GROUPS, DIAG_INTERPRETABLE_MIN,
)


def test_subject_id_drops_session_suffix():
    assert _subject_id(Path("dandi000673_ctg_sub-10_ses-1_ecephys+image.npz")) == "sub-10"
    assert _subject_id(Path("dandi000574_units_ctg_sub-01_ses-01.npz")) == "sub-01"
    assert _subject_id(Path("boran_ctg_sub-01.npz")) == "sub-01"


def test_subject_id_falls_back_to_trailing_token():
    assert _subject_id(Path("miller_ctg_corrected_al.npz")) == "al"


def test_cluster_bootstrap_recovers_known_positive_effect():
    rng = np.random.default_rng(0)
    values = np.full(20, 0.1) + rng.normal(0, 0.01, 20)
    out = _cluster_bootstrap(values, rng)
    assert abs(out["mean_offdiag_effect"] - 0.1) < 0.01
    assert out["ci95_lower"] > 0
    assert out["mdd"] > 0
    assert out["n_subjects"] == 20


def test_cluster_bootstrap_ci_straddles_zero_for_null_data():
    rng = np.random.default_rng(1)
    values = rng.normal(0, 0.1, 15)
    out = _cluster_bootstrap(values, rng)
    assert out["ci95_lower"] < 0 < out["ci95_upper"]


def _computed_cell(offdiag, p, diag_auc):
    stats = {
        "status": "computed", "mean_offdiag_effect": offdiag, "ci95_lower": offdiag - 0.02,
        "ci95_upper": offdiag + 0.02, "p_value_bootstrap": p, "mdd": 0.03,
        "mean_diag_auc": diag_auc, "n_subjects": 9,
        "diagonal_auc_minus_reference": diag_auc - DIAG_INTERPRETABLE_MIN,
    }
    return {"stats": stats}


def test_cells_carry_numbers_and_no_threshold_label():
    cells = {
        "a": _computed_cell(0.05, 0.001, 0.70),
        "b": _computed_cell(0.01, 0.40, 0.52),
        "c": {"stats": {"status": "infeasible"}},
    }
    _apply_fdr(cells)
    assert "verdict" not in cells["a"] and "verdict" not in cells["c"]
    reading = cells["b"]["reading"]
    assert reading["q_value_fdr"] >= reading["p_value_bootstrap"]
    assert reading["diagonal_auc_minus_reference"] == 0.52 - DIAG_INTERPRETABLE_MIN
    assert set(reading) >= {"offdiag_effect", "ci95_lower", "ci95_upper", "p_value_bootstrap",
                            "q_value_fdr", "mdd", "mean_diag_auc", "n_subjects"}
    assert "reading" not in cells["c"]


def test_group_summary_reports_estimate_difference_between_views():
    primary = {}
    for group in INDEPENDENT_PATIENT_GROUPS.values():
        for i, cell in enumerate(group["cells"]):
            primary[cell] = {"reading": {"offdiag_effect": 0.1 * (i + 1)}}
    summary = {row["group"]: row for row in _independent_group_summary(primary)}
    two_views = summary["dandi_000574"]["offdiag_effect_difference_between_views"]
    assert two_views["first_minus_second"] == pytest.approx(
        primary[two_views["first_view"]]["reading"]["offdiag_effect"]
        - primary[two_views["second_view"]]["reading"]["offdiag_effect"])
    assert "offdiag_effect_difference_between_views" not in summary["dandi000469"]
    assert summary["dandi000673_dandi001187"]["canonical_view"] == "dandi001187"


def test_diag_interpretable_min_matches_project_default():
    assert DIAG_INTERPRETABLE_MIN == 0.55
