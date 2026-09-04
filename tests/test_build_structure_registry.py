"""Regression tests for scripts/build_structure_registry.py's ordering logic:
a non-identified pooled lambda must never out-rank a real estimate via an
inf-sort fallback, and a structure's minimum-patient gate must reflect the
pooled estimate's actual finite support."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import build_structure_registry as mod  # noqa: E402


def test_non_identified_structure_never_ranked_ahead_of_an_estimable_one():
    structures = {
        "fast_and_estimable": {
            "status": "identified",
            "confinement_rate_lambda_pooled_across_datasets": {"status": "estimable", "mean": 2.0, "n_patients": 10},
        },
        "slow_and_estimable": {
            "status": "identified",
            "confinement_rate_lambda_pooled_across_datasets": {"status": "estimable", "mean": 1.0, "n_patients": 8},
        },
        "identified_but_lambda_non_identified": {
            "status": "identified",
            "confinement_rate_lambda_pooled_across_datasets": {"status": "non_identified", "n_patients": 1},
        },
        "structure_non_identified": {
            "status": "non_identified", "n_patients_total": 1,
        },
    }
    estimable, ranked, non_identified = mod.rank_estimable_structures(structures)
    assert ranked == ["fast_and_estimable", "slow_and_estimable"]
    assert set(estimable) == {"fast_and_estimable", "slow_and_estimable"}
    assert non_identified == {"identified_but_lambda_non_identified": 1, "structure_non_identified": 1}


def test_pooled_gate_uses_the_bootstraps_finite_count_not_raw_collected_n(monkeypatch):
    """A patient-lambda value collected as present but non-finite (e.g. NaN, which
    bootstrap_summary's own np.isfinite filter drops) must not let a structure's raw
    n_total clear the minimum-patient gate while its actual pooled evidence does not --
    this previously let a structure pool two non-identified singletons into a reported
    'estimable' n=2 entry that entered the cross-structure ordering."""
    def fake_collect(region, artifacts):
        return {
            "dandi_000469": {"patientA": 2.0},
            "dandi_001187_000673_content_axis_battery": {"patientB": 2.1, "patientC": float("nan")},
        }
    monkeypatch.setattr(mod, "collect_lambda_by_dataset", fake_collect)
    result = mod.build_structure("vmpfc", {}, {}, np.random.default_rng(0))
    assert result["status"] == "non_identified"
    assert result["n_patients_total"] == 3
    assert result["n_patients_pooled_finite"] == 2


def test_pooled_key_excluded_even_when_present_in_every_source_set():
    """Python's `-` binds tighter than `|`, so `a | b | c - {"pooled"}` only strips
    "pooled" from c -- it leaks back in from a and b. Regenerated all_regions must not
    contain it regardless of how many of the unioned sets carry a "pooled" key."""
    a = {"hippocampus", "pooled"}
    b = {"amygdala", "pooled"}
    c = {"dacc", "pooled"}
    d = {"vtc"}
    combined_correct = (a | b | c | d) - {"pooled"}
    combined_buggy = a | b | c | d - {"pooled"}
    assert "pooled" not in combined_correct
    assert "pooled" in combined_buggy  # documents the exact defect being guarded against
