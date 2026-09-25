from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_microstimulation_selectivity_mechanism import (  # noqa: E402
    READINGS, _analyze_arrays, _negate_pooled, _pool_across_sessions, _pool_multiple_imputation, _split_mask,
)


def _make_session_arrays(rng, n_angle=3, n_channels=6, n_per_group=24, nonpref_boost=3.0, pref_boost=0.5):
    """Planted mechanism: post-stim rate rises more for non-preferred targets than for the
    unit's own preferred target; no pre-stimulation offset is planted. Tuning is present in
    both pre- and post-window control rates, so both selection-window readings should agree."""
    pre, post, angle, is_stim = [], [], [], []
    base = rng.uniform(2.0, 4.0, size=n_channels)
    tuning = rng.uniform(1.0, 2.0, size=n_channels)
    preferred_angle = rng.integers(0, n_angle, size=n_channels)
    for a in range(n_angle):
        is_pref = (preferred_angle == a).astype(float)
        tuned_rate = base + tuning * is_pref
        for flag in (False, True):
            for _ in range(n_per_group):
                pre.append(rng.normal(tuned_rate, 0.3))
                boost = is_pref * pref_boost + (1 - is_pref) * nonpref_boost if flag else 0.0
                post.append(rng.normal(tuned_rate + boost, 0.3))
                angle.append(a)
                is_stim.append(flag)
    return (np.clip(np.array(pre), 0, None), np.clip(np.array(post), 0, None),
            np.array(angle), np.array(is_stim, dtype=bool), np.arange(n_channels))


def test_split_mask_is_stratified_and_deterministic():
    rng = np.random.default_rng(0)
    angle = rng.integers(0, 4, size=200)
    is_stim = rng.integers(0, 2, size=200).astype(bool)
    mask_a = _split_mask(angle, is_stim, "seed_x")
    mask_b = _split_mask(angle, is_stim, "seed_x")
    assert np.array_equal(mask_a, mask_b)
    for a in range(4):
        for flag in (False, True):
            group = (angle == a) & (is_stim == flag)
            train_frac = mask_a[group].mean()
            assert 0.3 < train_frac < 0.7


def test_multiple_imputation_pooling_matches_hand_computation():
    pairs = [(1.0, 0.5), (2.0, 0.7), (0.0, 0.4)]
    result = _pool_multiple_imputation(pairs)
    estimates = np.array([1.0, 2.0, 0.0])
    variances = np.array([0.5, 0.7, 0.4])
    expected_estimate = estimates.mean()
    expected_variance = variances.mean() + (1 + 1 / 3) * estimates.var(ddof=1)
    assert result["m"] == 3
    assert abs(result["estimate"] - expected_estimate) < 1e-12
    assert abs(result["variance"] - expected_variance) < 1e-12


def test_pool_multiple_imputation_single_draw_has_no_between_imputation_term():
    result = _pool_multiple_imputation([(3.0, 0.2)])
    assert result["m"] == 1
    assert result["estimate"] == 3.0
    assert result["variance"] == 0.2


def test_analyze_arrays_detects_larger_nonpreferred_stimulation_response():
    rng = np.random.default_rng(7)
    pre, post, angle, is_stim, channel_ids = _make_session_arrays(rng)
    result = _analyze_arrays(pre, post, angle, is_stim, channel_ids, n_angle=3, prefix="synthetic_session")
    assert result["n_splits_usable"] > 0
    for reading in READINGS:
        summary = result["summary"][reading]["all_units"]
        assert summary["preferred"]["raw"]["n_units"] > 0
        assert summary["nonpreferred_pooled"]["raw"]["n_units"] > 0
        assert summary["difference_pooled"]["raw"]["session_mean"] > 0
        # corrected barely moves the raw estimate since no pre-stimulation offset was planted
        raw_mean = summary["difference_pooled"]["raw"]["session_mean"]
        corrected_mean = summary["difference_pooled"]["corrected"]["session_mean"]
        assert abs(raw_mean - corrected_mean) < 1.0


def test_planted_effect_recovered_and_index_is_negated_difference():
    sessions = []
    for seed in (1, 2, 3, 4):
        rng = np.random.default_rng(seed)
        pre, post, angle, is_stim, channel_ids = _make_session_arrays(rng)
        result = _analyze_arrays(pre, post, angle, is_stim, channel_ids, n_angle=3, prefix=f"synthetic_{seed}")
        sessions.append({"animal": "Zz" if seed < 3 else "Yy", "summary": result["summary"]})

    for reading in READINGS:
        difference = _pool_across_sessions(sessions, reading, "all_units", "difference_pooled", "raw")
        assert difference["status"] == "computed"
        assert difference["n_sessions"] == 4
        assert difference["estimate"] > 0
        assert difference["p_value"] < 0.2

        selectivity_index = _negate_pooled(difference)
        assert selectivity_index["estimate"] == -difference["estimate"]
        assert selectivity_index["ci"] == [-difference["ci"][1], -difference["ci"][0]]
        assert selectivity_index["p_value"] == difference["p_value"]


if __name__ == "__main__":
    test_split_mask_is_stratified_and_deterministic()
    test_multiple_imputation_pooling_matches_hand_computation()
    test_pool_multiple_imputation_single_draw_has_no_between_imputation_term()
    test_analyze_arrays_detects_larger_nonpreferred_stimulation_response()
    test_planted_effect_recovered_and_index_is_negated_difference()
    print("ok")
