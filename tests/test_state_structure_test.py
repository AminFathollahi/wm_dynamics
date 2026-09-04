from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_state_structure_test import (  # noqa: E402
    omnibus_group_permutation_test,
    structure_effect_conditioned_on_nugget_share,
    unpaired_matched_structure_test,
    within_patient_paired_structure_test,
)


def _session(patient, session, structure, d_perm, white, n_units=20, dataset="dandi_000469"):
    return {"dataset": dataset, "patient": patient, "session": session, "structure": structure,
            "d_perm": d_perm, "white_fraction_median": white, "n_units": n_units, "n_trials": 40, "n_lags_used": 5}


def test_omnibus_permutation_detects_real_group_difference_and_not_noise():
    rng = np.random.default_rng(1)
    labels = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)
    separated = np.concatenate([rng.normal(0.0, 0.1, 20), rng.normal(0.5, 0.1, 20), rng.normal(1.0, 0.1, 20)])
    result = omnibus_group_permutation_test(labels, separated, seed_name="test_separated")
    assert result["status"] == "tested"
    assert result["significant"]

    flat = rng.normal(0.0, 0.1, 60)
    result_flat = omnibus_group_permutation_test(labels, flat, seed_name="test_flat")
    assert not result_flat["significant"]


def test_conditioning_on_nugget_share_removes_a_confounded_structure_effect():
    """Structure is entirely a proxy for nugget share here (each structure's
    white_fraction_median is a fixed offset with no independent d_perm
    signal): the raw structure effect must be significant, but both the
    residualized and matched-band tests must fail to find anything, giving
    the recording_quality_explains_the_pattern verdict."""
    rng = np.random.default_rng(2)
    sessions = []
    structure_white_offset = {"amygdala": 0.3, "hippocampus": 0.6, "dacc": 0.9}
    for structure, offset in structure_white_offset.items():
        for i in range(10):
            white = offset + rng.normal(0, 0.02)
            d_perm = 0.4 * white + rng.normal(0, 0.01)  # d_perm depends on white share ONLY
            sessions.append(_session(f"{structure}_p{i}", f"{structure}_s{i}", structure, d_perm, white))
    result = structure_effect_conditioned_on_nugget_share(sessions)
    assert result["raw_structure_effect_on_d_perm"]["significant"]
    assert result["verdict"] == "recording_quality_explains_the_pattern"
    assert "statement" in result


def test_conditioning_on_nugget_share_does_not_remove_a_genuine_structure_effect():
    """Structure has its own effect on d_perm independent of nugget share
    (nugget share is unrelated noise here): the residualized test must
    still find the structure effect, so the verdict must NOT claim
    recording quality explains it."""
    rng = np.random.default_rng(3)
    sessions = []
    structure_d_perm_offset = {"amygdala": 0.0, "hippocampus": 0.3, "dacc": 0.6}
    for structure, offset in structure_d_perm_offset.items():
        for i in range(10):
            white = rng.uniform(0.5, 0.9)  # unrelated to structure or d_perm
            d_perm = offset + rng.normal(0, 0.02)
            sessions.append(_session(f"{structure}_p{i}", f"{structure}_s{i}", structure, d_perm, white))
    result = structure_effect_conditioned_on_nugget_share(sessions)
    assert result["raw_structure_effect_on_d_perm"]["significant"]
    assert result["verdict"] == "structure_differences_persist_after_conditioning"


def test_within_patient_paired_test_detects_separation_with_enough_shared_patients():
    rng = np.random.default_rng(4)
    sessions = []
    for i in range(10):
        p = f"patient{i}"
        sessions.append(_session(p, f"{p}_a", "amygdala", 0.0 + rng.normal(0, 0.02), 0.9))
        sessions.append(_session(p, f"{p}_h", "hippocampus", 0.3 + rng.normal(0, 0.02), 0.9))
    result = within_patient_paired_structure_test(sessions)
    assert result["branch"] == "structures_separate"
    assert result["n_patients_with_2_or_more_structures"] == 10


def test_within_patient_paired_test_flags_too_few_shared_patients():
    sessions = [
        _session("p1", "p1_a", "amygdala", 0.1, 0.9),
        _session("p2", "p2_h", "hippocampus", 0.2, 0.9),
        _session("p3", "p3_a", "amygdala", 0.15, 0.9),
    ]
    result = within_patient_paired_structure_test(sessions)
    assert result["branch"] == "not_enough_within_patient_pairs"
    assert result["n_patients_with_2_or_more_structures"] == 0


def test_unpaired_matched_structure_test_requires_a_common_band():
    """Unit count and nugget share held constant (and equal) across both
    groups so every session trivially falls in the common band -- isolates
    whether the matched-band machinery itself works from whether a given
    random draw happens to produce overlapping bands (it often won't, by
    design: the band is the INTERSECTION of each group's own middle-50%
    range, which shrinks fast with real per-group spread)."""
    rng = np.random.default_rng(5)
    sessions = []
    for i in range(8):
        sessions.append(_session(f"a{i}", f"a{i}_s", "amygdala", rng.normal(0, 0.02), 0.9, n_units=20))
    for i in range(8):
        sessions.append(_session(f"h{i}", f"h{i}_s", "hippocampus", rng.normal(0, 0.02), 0.9, n_units=20))
    result = unpaired_matched_structure_test(sessions)
    tested = [p for p in result["pairs"] if p["status"] == "tested"]
    assert len(tested) == 1
    assert tested[0]["n_sessions_a"] >= 4 and tested[0]["n_sessions_b"] >= 4
