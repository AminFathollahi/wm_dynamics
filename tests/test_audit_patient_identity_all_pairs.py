"""Regression tests for scripts/audit_patient_identity_all_pairs.py.

The core risk this module guards against actually happened once during
development: comparing raw cumulative trial-onset time (rather than
inter-trial intervals) gives a near-monotonic ramp whose Pearson correlation
is close to 1 between ANY two same-task sessions regardless of patient
identity, producing thousands of spurious "candidate" matches. These tests
pin the fix (compare on ITI) and the duplicate-vs-independent thresholds.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from audit_patient_identity_all_pairs import compare, classify  # noqa: E402


def _fp(onset_rel, release="X", patient="P1"):
    onset_rel = np.asarray(onset_rel, dtype=float)
    return {
        "release": release,
        "patient": patient,
        "n_trials": len(onset_rel),
        "session_span_s": float(onset_rel[-1] - onset_rel[0]),
        "onset_rel": onset_rel,
        "iti": np.diff(onset_rel),
    }


def test_exact_duplicate_session_is_high_confidence():
    rng = np.random.default_rng(0)
    onset = np.cumsum(np.r_[0, rng.uniform(1.0, 3.0, 49)])
    a = _fp(onset, release="A")
    b = _fp(onset.copy(), release="B")
    cmp = compare(a, b)
    assert classify(cmp) == "high_confidence_duplicate"


def test_independent_sessions_with_similar_trend_are_not_flagged():
    """Two unrelated sessions built from the SAME jitter distribution (same
    task template, different patients) must not classify as a duplicate or
    even a candidate -- this is the case that raw cumulative-onset
    correlation got wrong (~0.999 for any two same-length ramps).
    """
    rng = np.random.default_rng(1)
    onset_a = np.cumsum(np.r_[0, rng.uniform(1.0, 3.0, 49)])
    onset_b = np.cumsum(np.r_[0, rng.uniform(1.0, 3.0, 49)])
    cmp = compare(_fp(onset_a), _fp(onset_b))
    assert classify(cmp) == "independent"


def test_mismatched_trial_count_has_no_max_abs_diff_and_is_not_duplicate():
    rng = np.random.default_rng(2)
    onset_a = np.cumsum(np.r_[0, rng.uniform(1.0, 3.0, 49)])
    onset_b = np.cumsum(np.r_[0, rng.uniform(1.0, 3.0, 39)])
    cmp = compare(_fp(onset_a), _fp(onset_b))
    assert cmp is not None
    assert cmp["max_abs_iti_diff_s"] is None
    assert classify(cmp) != "high_confidence_duplicate"


def test_grossly_different_session_span_is_not_comparable():
    onset_a = np.arange(50, dtype=float) * 1.5
    onset_b = np.arange(50, dtype=float) * 15.0
    assert compare(_fp(onset_a), _fp(onset_b)) is None
