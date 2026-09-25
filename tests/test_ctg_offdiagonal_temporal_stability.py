"""Tests for scripts/run_ctg_offdiagonal_temporal_stability.py -- the pieces
that could silently break this analysis without erroring: subject-id
extraction (session suffixes must not merge into the cluster key), the
cluster-bootstrap CI/p-value/mdd arithmetic, and the clear/inconclusive
decision rule."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_ctg_offdiagonal_temporal_stability import (  # noqa: E402
    _subject_id, _cluster_bootstrap, _verdict, DIAG_INTERPRETABLE_MIN,
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


def test_verdict_requires_both_fdr_and_ci():
    interpretable_stats = {
        "status": "computed", "diag_interpretable": True,
        "ci95_lower": 0.02, "ci95_upper": 0.08, "q_value_fdr": 0.01,
    }
    assert _verdict(interpretable_stats) == "clear_stable"

    ci_only = dict(interpretable_stats, q_value_fdr=0.2)
    assert _verdict(ci_only) == "inconclusive"

    q_only = dict(interpretable_stats, ci95_lower=-0.01)
    assert _verdict(q_only) == "inconclusive"

    undecodable = dict(interpretable_stats, diag_interpretable=False)
    assert _verdict(undecodable) == "inconclusive_undecodable_diagonal"

    infeasible = {"status": "infeasible"}
    assert _verdict(infeasible) == "infeasible"


def test_diag_interpretable_min_matches_project_default():
    assert DIAG_INTERPRETABLE_MIN == 0.55
