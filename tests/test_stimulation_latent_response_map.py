"""Tests for scripts/run_stimulation_latent_response_map.py.

Covers the two estimator properties the analysis depends on (recovery of a
planted displacement's magnitude and direction; scale-invariance of the
spontaneous-variability normalisation) plus the crash-proofing primitives
(atomic checkpointing, the reproduction-gate float-tolerant diff).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from control import subspace_alignment  # noqa: E402
from run_stimulation_latent_response_map import (  # noqa: E402
    deep_diff,
    displacement_and_normalisation,
    load_checkpoint,
    per_trial_state_norm,
    pool_sessions,
    rotation_null_alignment,
    save_checkpoint,
)


def _planted_displacement(rng, k=4, n_unstim=300, n_stim=200, magnitude=10.0, noise_sd=1.0):
    """Unstimulated trials centred on the origin; stimulated trials shifted
    by `magnitude` along axis 0 only -- a known displacement magnitude along
    a known direction, everything else pure noise."""
    unstim = rng.normal(0.0, noise_sd, size=(n_unstim, 1, k))
    stim = rng.normal(0.0, noise_sd, size=(n_stim, 1, k))
    stim[:, :, 0] += magnitude
    scores = np.concatenate([unstim, stim], axis=0)
    unstim_mask = np.zeros(n_unstim + n_stim, dtype=bool)
    unstim_mask[:n_unstim] = True
    stim_mask = ~unstim_mask
    return scores, unstim_mask, stim_mask


def test_estimator_recovers_planted_magnitude():
    rng = np.random.default_rng(0)
    magnitude, noise_sd = 10.0, 1.0
    scores, unstim_mask, stim_mask = _planted_displacement(rng, magnitude=magnitude, noise_sd=noise_sd)
    centroid = scores[unstim_mask].mean(axis=(0, 1))
    state = per_trial_state_norm(scores, centroid)
    result = displacement_and_normalisation(state, unstim_mask, stim_mask)

    assert result["status"] == "complete"
    # raw_displacement = E[||shift+noise||] - E[||noise||]; the noise floor
    # (a k=4-dimensional chi distribution here) pulls this systematically
    # below the pure shift, so "recovers" means "in the right ballpark and
    # on the expected (low) side of it", not an exact match.
    assert magnitude * 0.7 < result["raw_displacement"] < magnitude
    # spontaneous_sd is set by the noise alone (unstimulated trials carry no
    # shift), so the normalised ratio should be large and unambiguous.
    assert result["normalized_displacement"] > 5.0


def test_estimator_recovers_planted_alignment():
    rng = np.random.default_rng(1)
    k = 4
    scores, unstim_mask, stim_mask = _planted_displacement(rng, k=k, magnitude=10.0, noise_sd=1.0)
    direction = scores[stim_mask].mean(axis=(0, 1)) - scores[unstim_mask].mean(axis=(0, 1))
    direction_unit = direction / np.linalg.norm(direction)

    aligned_basis = np.eye(k)[:, [0]]      # axis 0: the planted direction
    orthogonal_basis = np.eye(k)[:, [1]]   # axis 1: pure noise, no shift

    aligned = rotation_null_alignment(aligned_basis, direction_unit, 500, np.random.default_rng(2))
    orthogonal = rotation_null_alignment(orthogonal_basis, direction_unit, 500, np.random.default_rng(3))

    assert aligned["alignment"] > 0.95
    assert aligned["p_value"] < 0.01
    assert orthogonal["alignment"] < aligned["alignment"]
    # An unaligned direction should look like a typical random direction,
    # i.e. not resolvable from the rotation null at conventional alpha.
    assert orthogonal["p_value"] > 0.05


def test_normalisation_is_scale_invariant():
    """Two synthetic sessions with identical TRUE displacement in different
    score-unit scales must report the same normalised value -- the entire
    point of dividing by the unstimulated trial-to-trial standard deviation
    in the same frame, rather than reporting the raw score-unit number."""
    rng = np.random.default_rng(4)
    scores_a, unstim_mask, stim_mask = _planted_displacement(rng, magnitude=6.0, noise_sd=1.0)
    for scale in (0.003, 1.0, 850.0):
        scores_b = scores_a * scale
        centroid_a = scores_a[unstim_mask].mean(axis=(0, 1))
        centroid_b = scores_b[unstim_mask].mean(axis=(0, 1))
        result_a = displacement_and_normalisation(per_trial_state_norm(scores_a, centroid_a), unstim_mask, stim_mask)
        result_b = displacement_and_normalisation(per_trial_state_norm(scores_b, centroid_b), unstim_mask, stim_mask)

        if scale != 1.0:
            assert result_a["raw_displacement"] != pytest.approx(result_b["raw_displacement"])  # raw is NOT comparable
        assert result_a["normalized_displacement"] == pytest.approx(result_b["normalized_displacement"], rel=1e-6)


def test_pool_sessions_sign_flip_and_mdd():
    values = {"s1": 1.0, "s2": 1.2, "s3": 0.8, "s4": 1.1, "s5": 0.9}
    pooled = pool_sessions(values, 1000)
    assert pooled["status"] == "complete"
    assert pooled["n"] == 5
    assert pooled["mean_diff"] == pytest.approx(1.0, abs=1e-9)
    assert pooled["significant"]  # tight cluster of positive values, clearly away from 0
    assert pooled["mdd_80pct_power"] is not None and pooled["mdd_80pct_power"] > 0


def test_checkpoint_round_trip_and_atomicity(tmp_path, monkeypatch):
    import run_stimulation_latent_response_map as mod

    monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    assert load_checkpoint("unit_a") is None  # nothing written yet -> absent, not an error

    record = {"value": 1.5, "nested": {"a": [1, 2, 3]}}
    save_checkpoint("unit_a", record)
    assert load_checkpoint("unit_a") == record

    # An unparseable checkpoint (e.g. a truncated write) is treated as absent.
    bad_path = mod._checkpoint_path("unit_b")
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not valid json")
    assert load_checkpoint("unit_b") is None

    # No stray temp files survive a normal save.
    leftovers = [p for p in (tmp_path / "checkpoints").glob("._tmp_*")]
    assert leftovers == []


def test_deep_diff_tolerance_and_sensitivity():
    a = {"x": 1.0000001, "y": [1, 2, {"z": 3.0}], "s": "same"}
    b = {"x": 1.0000002, "y": [1, 2, {"z": 3.0}], "s": "same"}
    assert deep_diff(a, b) == []  # within 1e-6 relative tolerance

    c = {"x": 1.5, "y": [1, 2, {"z": 3.0}], "s": "same"}
    diffs = deep_diff(a, c)
    assert len(diffs) == 1 and "$.x" in diffs[0]

    d = {"x": 1.0000001, "y": [1, 2, {"z": 3.0}], "s": "different"}
    diffs2 = deep_diff(a, d)
    assert len(diffs2) == 1 and "$.s" in diffs2[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
