from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_selective_persistence_and_attractor_geometry import CATEGORY_SUBSPACE_DIMENSIONALITY, CORPUS_SPECS, _category_neuron_test, _da_session, _build_pseudopopulation, _fit_pca_2d, _fit_selective_persistence_glm, _part1_region_summary, _project_pca_2d, _project_pca_3d, _pseudopopulation_decoding, _pseudopopulation_geometry, _pseudopopulation_region_report, _pseudopopulation_region_report_all_arms, _selective_neuron_keys, _region_association, _select_ridge_penalty_pseudopop, _split_trial_pool, _session_correlation, _part1_reaction_time_cells, _session_latency, MIN_UNITS_PER_REGION
from info_decoding import _category_469, _category_divided
from memorandum_decoding import fit_demixed_axes
import run_selective_persistence_and_attractor_geometry as target_module  # noqa: E402


def test_category_469_is_identity():
    pic = np.array([1, 2, 3, 4, 5])
    assert _category_469(pic).tolist() == [1, 2, 3, 4, 5]


def test_category_divided_maps_three_digit_codes_to_five_categories():
    pic = np.array([104, 205, 312, 441, 555])
    assert _category_divided(pic).tolist() == [1, 2, 3, 4, 5]


def test_fit_pca_2d_and_project_pca_2d_round_trip_on_encoding_window_rates():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6))
    mu, V = _fit_pca_2d(X, n_comp=3)
    Z = _project_pca_2d(X, mu, V)
    assert Z.shape == (20, 3)
    assert Z == pytest.approx((X - mu) @ V)


def test_project_pca_3d_on_out_of_sample_maintenance_trajectory():
    rng = np.random.default_rng(1)
    enc = rng.standard_normal((15, 6))
    maint = rng.standard_normal((15, 6, 20))
    mu, V = _fit_pca_2d(enc, n_comp=3)
    Z_maint = _project_pca_3d(maint, mu, V)
    assert Z_maint.shape == (15, 20, 3)
    X = maint.transpose(0, 2, 1).reshape(-1, 6) - mu
    assert Z_maint.reshape(-1, 3) == pytest.approx(X @ V)


def _synthetic_rates_with_category_tuning(rng, n_trials=60, preferred=2, n_categories=5, effect=3.0):
    category = rng.integers(1, n_categories + 1, size=n_trials)
    rates = rng.normal(loc=2.0, scale=0.5, size=n_trials)
    rates[category == preferred] += effect
    return rates, category


def test_category_neuron_test_detects_planted_tuning():
    rng = np.random.default_rng(2)
    rates, category = _synthetic_rates_with_category_tuning(rng)
    trial_id = np.arange(len(rates))
    out = _category_neuron_test(rates, category, trial_id, rng)
    assert out["status"] == "computed"
    assert out["preferred_category"] == 2
    assert out["omnibus_p"] < 0.05
    assert out["meets_published_category_neuron_criterion"] is True


def test_category_neuron_test_null_data_rarely_meets_criterion():
    rng = np.random.default_rng(3)
    n_trials = 60
    category = rng.integers(1, 6, size=n_trials)
    rates = rng.normal(loc=2.0, scale=0.5, size=n_trials)
    trial_id = np.arange(n_trials)
    out = _category_neuron_test(rates, category, trial_id, rng)
    assert out["status"] == "computed"
    assert "meets_published_category_neuron_criterion" in out


def test_category_neuron_test_underpowered_below_trial_floor():
    rng = np.random.default_rng(4)
    rates = rng.standard_normal(5)
    category = np.array([1, 1, 2, 2, 3])
    trial_id = np.arange(5)
    out = _category_neuron_test(rates, category, trial_id, rng)
    assert out["status"] == "underpowered"


def test_category_neuron_test_widened_window_doubles_trial_rows():
    rng = np.random.default_rng(41)
    rates, category = _synthetic_rates_with_category_tuning(rng)
    n = len(rates)
    rates_widened = np.concatenate([rates, rates])
    category_widened = np.concatenate([category, category])
    trial_id_widened = np.concatenate([np.arange(n), np.arange(n)])
    out = _category_neuron_test(rates_widened, category_widened, trial_id_widened, rng)
    assert out["status"] == "computed"
    assert out["n_trials"] == n
    assert out["meets_published_category_neuron_criterion"] is True


def test_fit_selective_persistence_glm_underpowered_below_two_neurons():
    rows = [{"baseline_corrected_maintenance_rate_correct_trials": [1.0] * 10,
             "is_preferred_correct_trials": [1] * 5 + [0] * 5, "patient": "sub-1", "neuron_id": "s1__u0"}]
    out = _fit_selective_persistence_glm(rows)
    assert out["status"] == "underpowered"


def test_fit_selective_persistence_glm_recovers_planted_effect():
    rng = np.random.default_rng(5)
    rows = []
    for i in range(20):
        n = 30
        condition = rng.integers(0, 2, size=n)
        metric = rng.normal(0.0, 1.0, size=n) + condition * 2.0
        rows.append({"baseline_corrected_maintenance_rate_correct_trials": metric.tolist(),
                     "is_preferred_correct_trials": condition.tolist(),
                     "patient": f"sub-{i % 6}", "neuron_id": f"s1__u{i}"})
    out = _fit_selective_persistence_glm(rows)
    assert out["status"] == "computed"
    assert out["estimate"] == pytest.approx(2.0, abs=0.5)
    assert out["ci_lower"] < out["estimate"] < out["ci_upper"]
    for key in ("mdd", "p_value", "n_patients"):
        assert key in out


def test_part1_region_summary_selection_matched_below_floor_reports_reason():
    region_sessions = {
        "s1": {"status": "computed", "patient": "sub-1", "category_all": [1, 2, 1], "accuracy_all": [1, 1, 1],
               "neurons": [
            {"neuron_index": 0, "test": {"status": "computed", "meets_published_category_neuron_criterion": False},
             "test_first_encoding_only": {"status": "computed",
                                           "meets_published_category_neuron_criterion": False},
             "preferred_category": 1,
             "rate_cat_window_all_trials": [1.0, 2.0, 3.0], "rate_probe_window_all_trials": [1.0, 2.0, 3.0],
             "baseline_corrected_maint_all_trials": [1.0, 2.0, 3.0],
             "baseline_corrected_maintenance_rate_correct_trials": [1.0, 2.0, 3.0],
             "is_preferred_correct_trials": [1, 0, 1]},
        ]},
    }
    out = _part1_region_summary(region_sessions)
    assert out["selection_matched_arm"]["status"] == "below_published_minimum_category_neuron_count"
    assert out["full_population_arm"]["n_neurons_tested"] == 1
    assert out["trial_split_matched_arm"]["status"] == "not_computable"


def test_session_latency_masks_nonfinite_and_nonpositive_as_nan():
    raw = np.array([1.5, -1.0, np.nan, 0.0, 2.0, 3.0])
    keep = np.array([True, True, True, True, True, False])
    out = _session_latency(raw, keep)
    assert out.shape == (5,)
    assert np.isfinite(out).tolist() == [True, False, False, False, True]
    assert out[0] == pytest.approx(1.5)
    assert out[4] == pytest.approx(2.0)


def test_part1_reaction_time_cells_recovers_planted_correlation():
    rng = np.random.default_rng(60)
    n = 200
    is_preferred = rng.integers(0, 2, size=n).astype(bool)
    latency = rng.uniform(0.3, 2.0, size=n)
    rate = np.where(is_preferred, 3.0, 1.0) - 0.8 * latency + rng.normal(0, 0.05, n)
    region_sessions = {
        "s1": {
            "status": "computed", "patient": "sub-1", "latency_correct_trials": latency.tolist(),
            "neurons": [{
                "neuron_index": 0, "test": {"status": "computed"},
                "baseline_corrected_maintenance_rate_correct_trials": rate.tolist(),
                "is_preferred_correct_trials": is_preferred.astype(int).tolist(),
            }],
        },
    }
    nonpref, signed = _part1_reaction_time_cells(region_sessions)
    assert set(nonpref) == {"s1__u0"}
    assert set(signed) == {"s1__u0"}
    nonpref_cell = nonpref["s1__u0"]
    assert nonpref_cell["patient"] == "sub-1"
    assert len(nonpref_cell["nonpreferred_rate"]) == len(nonpref_cell["latency"])
    assert len(nonpref_cell["nonpreferred_rate"]) == int((~is_preferred).sum())
    signed_cell = signed["s1__u0"]
    assert len(signed_cell["signed_preferred_rate"]) == len(signed_cell["latency"]) == n
    r = np.corrcoef(nonpref_cell["nonpreferred_rate"], nonpref_cell["latency"])[0, 1]
    assert r < -0.5


def test_part1_trial_split_arm_recovers_planted_effect_and_pools_across_splits():
    rng = np.random.default_rng(40)
    region_sessions = {}
    n_trials = 200
    for s in range(3):
        category_all = rng.integers(1, 6, size=n_trials)
        accuracy_all = np.ones(n_trials, dtype=int)
        neurons = []
        for u in range(6):
            preferred = int(rng.integers(1, 6))
            rate_cat = np.where(category_all == preferred, 6.0, 1.0) + rng.normal(0, 0.3, n_trials)
            bcm = np.where(category_all == preferred, 3.0, 0.5) + rng.normal(0, 0.5, n_trials)
            neurons.append({"neuron_index": u, "test": {"status": "computed"},
                             "rate_cat_window_all_trials": rate_cat.tolist(),
                             "rate_probe_window_all_trials": rate_cat.tolist(),
                             "baseline_corrected_maint_all_trials": bcm.tolist()})
        region_sessions[f"s{s}"] = {"status": "computed", "patient": f"sub-{s}",
                                     "category_all": category_all.tolist(), "accuracy_all": accuracy_all.tolist(),
                                     "neurons": neurons}
    out, out_floor_restricted = target_module._part1_trial_split_arm(region_sessions)
    assert out["status"] == "computed"
    assert out["n_splits_used"] >= 1
    assert out["estimate"] > 0
    assert len(out["per_split"]) == target_module.N_TRIAL_SPLIT_REPEATS
    assert out_floor_restricted["per_split"] == out["per_split"]


def _synthetic_session_for_da(rng, n_units=20, n_trials_per_category=10, n_categories=4, window_s=2.3):
    unit_regions = np.array(["hippocampus"] * n_units)
    n_trials = n_trials_per_category * n_categories
    category = np.repeat(np.arange(1, n_categories + 1), n_trials_per_category)
    rng.shuffle(category)
    t_enc1 = np.arange(n_trials, dtype=float) * (window_s + 3.0)
    t_maint = t_enc1 + 1.0
    spike_lists_all = []
    for _ in range(n_units):
        spikes = []
        for t0 in t_enc1:
            n_spk = rng.poisson(2.0 * (window_s + 2.0))
            spikes.append(t0 + rng.uniform(0, window_s + 2.0, size=n_spk))
        spike_lists_all.append(np.sort(np.concatenate(spikes)) if spikes else np.array([]))
    accuracy = rng.uniform(size=n_trials) > 0.3
    latency = rng.uniform(0.3, 2.0, size=n_trials)
    return dict(patient="sub-synthetic", spike_lists_all=spike_lists_all, unit_regions=unit_regions,
                category=category, category_basis="synthetic", t_enc1=t_enc1, t_baseline=t_enc1 - 0.5,
                t_maint=t_maint, accuracy=accuracy, latency=latency, enc_win=0.5, baseline_win=0.5,
                maint_win=window_s)


def test_da_session_end_to_end_on_synthetic_session_above_floor():
    rng = np.random.default_rng(6)
    session = _synthetic_session_for_da(rng)
    out = _da_session(session, "hippocampus")
    assert out["status"] == "computed"
    assert out["n_units"] == 20
    assert out["category_subspace_dimensionality"] == min(CATEGORY_SUBSPACE_DIMENSIONALITY, out["n_pc"])
    assert -1e-6 <= out["demixed_variance_captured"] <= 1.0 + 1e-6
    assert out["ridge_penalty"] in target_module.DPCA_RIDGE_LAMBDA_GRID
    assert len(out["da"]) == out["n_trials_with_defined_da"]
    assert len(out["da"]) == len(out["da_pca_only"])
    assert len(out["da"]) == len(out["accuracy"]) == len(out["population_spike_count"]) == len(
        out["rate_free_deviation"])
    assert all(np.isfinite(v) for v in out["da"])
    assert all(np.isfinite(v) for v in out["rate_free_deviation"])


def test_da_session_refused_below_unit_floor():
    rng = np.random.default_rng(7)
    session = _synthetic_session_for_da(rng, n_units=3)
    out = _da_session(session, "hippocampus")
    assert out["status"] == "refused"
    assert f"{MIN_UNITS_PER_REGION}" in out["reason"]


def test_da_session_refused_below_category_floor():
    rng = np.random.default_rng(8)
    session = _synthetic_session_for_da(rng, n_trials_per_category=2, n_categories=4)
    out = _da_session(session, "hippocampus")
    assert out["status"] == "refused"


def test_session_correlation_recovers_planted_relationship():
    rng = np.random.default_rng(9)
    x = rng.standard_normal(40)
    y = (x > 0).astype(float)
    cell = {"da": x.tolist(), "accuracy": y.tolist()}
    r = _session_correlation("da", "accuracy", cell, "test_tag")
    assert r is not None
    assert r > 0.3


def test_session_correlation_none_below_trial_floor():
    cell = {"da": [1.0, 2.0], "accuracy": [1.0, 0.0]}
    r = _session_correlation("da", "accuracy", cell, "test_tag2")
    assert r is None


def test_region_association_underpowered_below_four_patients(monkeypatch):
    monkeypatch.setattr(target_module, "stable_seed", lambda s: 0)
    rng = np.random.default_rng(10)
    cells = {}
    for i in range(2):
        x = rng.standard_normal(20)
        y = rng.standard_normal(20)
        cells[f"s{i}"] = {"status": "computed", "patient": f"sub-{i}", "da": x.tolist(),
                          "accuracy": y.tolist()}
    out = _region_association(cells, "da", "accuracy", "test_region_assoc")
    assert out["status"] == "underpowered"


def test_corpus_specs_regions_match_region_resolved_rate_stability_behaviour():
    from run_region_resolved_rate_stability_behaviour import CORPUS_SPECS as REGION_CORPUS_SPECS
    for corpus in CORPUS_SPECS:
        assert CORPUS_SPECS[corpus]["regions"] == REGION_CORPUS_SPECS[corpus]["regions"]


def test_fit_demixed_axes_axis_tracks_whichever_category_labels_it_is_given():
    # two orthogonal signals live in the same data: dim 0 separates category_a,
    # dim 1 separates an unrelated category_b -- a supervised fit must pick the
    # axis matching whichever label vector it is handed, an unsupervised PCA
    # fit (which never sees labels) could not distinguish the two.
    rng = np.random.default_rng(11)
    n = 40
    Z = rng.standard_normal((n, 5)) * 0.3
    category_a = np.array([1] * (n // 2) + [2] * (n // 2))
    Z[category_a == 1, 0] += 5.0
    Z[category_a == 2, 0] -= 5.0
    category_b = np.tile([1, 2], n // 2)
    Z[category_b == 1, 1] += 5.0
    Z[category_b == 2, 1] -= 5.0

    fit_a = fit_demixed_axes(Z, category_a, np.random.default_rng(0), d=1)
    fit_b = fit_demixed_axes(Z, category_b, np.random.default_rng(0), d=1)
    v_a, v_b = np.abs(fit_a["decoder"][:, 0]), np.abs(fit_b["decoder"][:, 0])
    assert v_a[0] > 0.9 and v_a[1] < 0.2
    assert v_b[1] > 0.9 and v_b[0] < 0.2


def test_build_pseudopopulation_pools_across_neurons_and_categories():
    rng = np.random.default_rng(13)
    neurons_enc, neurons_maint = {}, {}
    for key in ("s1__u0", "s1__u1"):
        neurons_enc[key] = {1: rng.standard_normal(8), 2: rng.standard_normal(8), 3: rng.standard_normal(8)}
        neurons_maint[key] = {1: rng.standard_normal((8, 3)), 2: rng.standard_normal((8, 3)),
                               3: rng.standard_normal((8, 3))}
    pseudopop, common_cats = _build_pseudopopulation(neurons_enc, neurons_maint, n_pseudo=5,
                                                       rng=np.random.default_rng(2))
    assert common_cats == [1, 2, 3]
    assert pseudopop["enc"].shape == (15, 2)
    assert pseudopop["maint"].shape == (15, 2, 3)
    assert pseudopop["category"].tolist().count(1) == 5
    assert pseudopop["category"].tolist().count(2) == 5
    assert pseudopop["category"].tolist().count(3) == 5


def test_pseudopopulation_region_report_strips_raw_arrays(monkeypatch):
    fake_geometry = {
        "status": "computed", "n_sessions_used": 2, "n_patients": 2, "n_units_pooled": 4,
        "ridge_penalty": 0.5, "variance_captured_category_marginal": 0.8,
        "pseudopopulation": {"maint": np.zeros((4, 4, 3)), "category": np.array([1, 1, 2, 2]),
                              "enc": np.zeros((4, 4)), "n_units": 4},
        "axes": np.zeros((4, 2)), "encoder_axes": np.zeros((4, 2)),
        "collected_enc": {"s1__u0": {1: np.zeros(5)}}, "collected_maint": {"s1__u0": {1: np.zeros((5, 3))}},
    }
    monkeypatch.setattr(target_module, "_pseudopopulation_geometry",
                         lambda corpus, region, selected_keys=None, seed_tag=None: fake_geometry)
    monkeypatch.setattr(target_module, "_pseudopopulation_decoding",
                         lambda collected_enc, collected_maint, rng: {"status": "computed",
                                                                       "observed_accuracy": 0.9})
    out = _pseudopopulation_region_report("dandi_000469", "hippocampus")
    assert out["status"] == "computed"
    assert out["decoding"]["observed_accuracy"] == 0.9
    for leaked_key in ("pseudopopulation", "axes", "encoder_axes", "collected_enc", "collected_maint"):
        assert leaked_key not in out
    assert out["variance_captured_category_marginal"] == 0.8


def test_selective_neuron_keys_reads_part1_checkpoint_criterion(monkeypatch):
    fake_checkpoint = {
        "s1": {"status": "computed", "schema_version": target_module.PART1_SCHEMA_VERSION, "neurons": [
            {"neuron_index": 0, "test": {"status": "computed", "meets_published_category_neuron_criterion": True}},
            {"neuron_index": 1, "test": {"status": "computed", "meets_published_category_neuron_criterion": False}},
        ]},
        "s2": {"status": "refused", "schema_version": target_module.PART1_SCHEMA_VERSION},
    }
    monkeypatch.setattr(target_module, "_load_checkpoint", lambda name: fake_checkpoint)
    keys = _selective_neuron_keys("dandi_000469", "hippocampus")
    assert keys == {"s1__u0"}


def test_count_matched_pseudopopulation_split_not_computable_below_candidate_floor():
    out = target_module._count_matched_pseudopopulation_split(
        "dandi_000469", "hippocampus", {}, [], ["s1__u0", "s1__u1"], {}, 5, 0)
    assert out["status"] == "not_computable"
    assert out["n_candidate_units"] == 2


def test_pseudopopulation_region_report_all_arms_orders_selective_first(monkeypatch):
    monkeypatch.setattr(target_module, "_selective_neuron_keys",
                         lambda corpus, region, field="test": {"s1__u0", "s1__u1"})
    monkeypatch.setattr(target_module, "_collect_region_neurons",
                         lambda corpus, region, step, selected_keys=None: {
                             "enc": {f"s1__u{i}": {} for i in range(20)}})

    def fake_report(corpus, region, selected_keys=None, seed_tag=None):
        n = len(selected_keys) if selected_keys is not None else 20
        return {"status": "computed", "n_units_pooled": n}

    monkeypatch.setattr(target_module, "_pseudopopulation_region_report", fake_report)
    monkeypatch.setattr(
        target_module, "_pseudopopulation_selective_holdout_arm",
        lambda corpus, region: ({"status": "computed", "estimate": 0.4,
                                  "selective_minus_random_difference": {"mean": 0.15}}, {}, [], []))
    out = _pseudopopulation_region_report_all_arms("dandi_000469", "hippocampus")
    assert list(out.keys())[:2] == ["selective_arm", "selective_arm_holdout"]
    assert out["selective_arm"]["n_selective_units"] == 2
    assert out["selective_arm"]["n_units_pooled"] == 2
    assert out["all_unit_arm"]["n_units_pooled"] == 20
    assert out["selective_arm"]["description"]
    assert out["all_unit_arm"]["description"]
    assert out["selective_arm_holdout"]["estimate"] == 0.4
    assert out["selective_arm_holdout"]["selective_minus_random_difference"]["mean"] == pytest.approx(0.15)


def _synthetic_collected_for_holdout(rng, n_units=30, n_per_cat=40, n_cat=5, n_time=4):
    enc, maint = {}, {}
    for u in range(n_units):
        preferred = int(rng.integers(1, n_cat + 1))
        enc[f"s1__u{u}"] = {}
        maint[f"s1__u{u}"] = {}
        for c in range(1, n_cat + 1):
            base = 6.0 if c == preferred else 1.0
            enc[f"s1__u{u}"][c] = base + rng.normal(0, 0.4, n_per_cat)
            maint[f"s1__u{u}"][c] = rng.normal(0, 1.0, (n_per_cat, n_time))
    return enc, maint


def test_pseudopopulation_selective_holdout_arm_selection_uses_only_train_half(monkeypatch):
    rng = np.random.default_rng(50)
    enc, maint = _synthetic_collected_for_holdout(rng)
    monkeypatch.setattr(target_module, "_collect_region_neurons",
                         lambda corpus, region, step, selected_keys=None: {"enc": enc, "maint": maint})
    combined, collected, common_cats, neuron_keys = (
        target_module._pseudopopulation_selective_holdout_arm("dandi_000469", "hippocampus"))
    assert combined["status"] == "computed"
    assert len(combined["per_split"]) == target_module.N_TRIAL_SPLIT_REPEATS
    assert all(row["n_selected_units"] <= len(neuron_keys) for row in combined["per_split"])
    computed_splits = [row for row in combined["per_split"] if row["status"] == "computed"]
    assert computed_splits
    for row in computed_splits:
        assert "count_matched_all_unit_subsample" in row


def test_split_trial_pool_is_disjoint_per_neuron_and_category():
    rng = np.random.default_rng(14)
    neurons_enc = {"s1__u0": {1: rng.standard_normal(9), 2: rng.standard_normal(11)},
                   "s1__u1": {1: rng.standard_normal(9), 2: rng.standard_normal(11)}}
    train_pool, test_pool = _split_trial_pool(neurons_enc, [1, 2], rng)
    for key in neurons_enc:
        for c in (1, 2):
            train_idx, test_idx = set(train_pool[key][c].tolist()), set(test_pool[key][c].tolist())
            assert not (train_idx & test_idx)
            assert train_idx | test_idx == set(range(len(neurons_enc[key][c])))
            assert len(train_idx) > 0 and len(test_idx) > 0


def test_build_pseudopopulation_respects_trial_pool_restriction():
    rng = np.random.default_rng(15)
    neurons_enc = {"s1__u0": {1: np.arange(10.0), 2: np.arange(10.0, 20.0), 3: np.arange(20.0, 30.0)}}
    neurons_maint = {"s1__u0": {c: v[:, None].repeat(2, axis=1) for c, v in
                                 {1: np.arange(10.0), 2: np.arange(10.0, 20.0), 3: np.arange(20.0, 30.0)}.items()}}
    trial_pool = {"s1__u0": {1: np.array([0, 1]), 2: np.array([5, 6, 7]), 3: np.array([9])}}
    pseudopop, _ = _build_pseudopopulation(neurons_enc, neurons_maint, n_pseudo=20, rng=rng,
                                            trial_pool=trial_pool)
    for row, c in zip(pseudopop["enc"][:, 0], pseudopop["category"]):
        assert row in neurons_enc["s1__u0"][int(c)][trial_pool["s1__u0"][int(c)]]


def test_pseudopopulation_decoding_no_real_trial_crosses_split(monkeypatch):
    # build_pseudopopulation is called twice per repeat (train, test); assert the trial_pool argument
    # passed each time is drawn from disjoint index sets for every neuron/category.
    rng = np.random.default_rng(16)
    n_per_cat = 12
    neurons_enc, neurons_maint = {}, {}
    for key in ("s1__u0", "s1__u1", "s1__u2"):
        neurons_enc[key] = {c: rng.standard_normal(n_per_cat) + (5.0 if c == 1 else -5.0)
                             for c in (1, 2, 3)}
        neurons_maint[key] = {c: rng.standard_normal((n_per_cat, 4)) for c in (1, 2, 3)}

    seen_pools = []
    real_build = target_module._build_pseudopopulation

    def _spy_build(neurons_enc, neurons_maint, n_pseudo, rng_arg, trial_pool=None):
        if trial_pool is not None:
            seen_pools.append(trial_pool)
        return real_build(neurons_enc, neurons_maint, n_pseudo, rng_arg, trial_pool)

    monkeypatch.setattr(target_module, "_build_pseudopopulation", _spy_build)
    out = _pseudopopulation_decoding(neurons_enc, neurons_maint, rng)
    assert out["status"] == "computed"
    assert len(seen_pools) >= 2
    train_pool, test_pool = seen_pools[0], seen_pools[1]
    for key in neurons_enc:
        for c in (1, 2, 3):
            assert not (set(train_pool[key][c].tolist()) & set(test_pool[key][c].tolist()))


def _synthetic_pooled_neurons(rng, n_units=6, n_per_cat=10):
    neurons_enc, neurons_maint = {}, {}
    for u in range(n_units):
        key = f"s1__u{u}"
        neurons_enc[key] = {c: rng.poisson(8.0 if c == 1 else 4.0, n_per_cat).astype(float) for c in (1, 2, 3)}
        neurons_maint[key] = {c: rng.poisson(3.0, (n_per_cat, 4)).astype(float) for c in (1, 2, 3)}
    return neurons_enc, neurons_maint


def test_select_ridge_penalty_pseudopop_no_real_trial_crosses_split(monkeypatch):
    rng = np.random.default_rng(18)
    neurons_enc, neurons_maint = _synthetic_pooled_neurons(rng)
    common_cats = [1, 2, 3]

    seen_pools = []
    real_build = target_module._build_pseudopopulation

    def _spy_build(enc, maint, n_pseudo, rng_arg, trial_pool=None):
        if trial_pool is not None:
            seen_pools.append(trial_pool)
        return real_build(enc, maint, n_pseudo, rng_arg, trial_pool)

    monkeypatch.setattr(target_module, "_build_pseudopopulation", _spy_build)
    best, score_table = _select_ridge_penalty_pseudopop(neurons_enc, neurons_maint, common_cats,
                                                          n_pseudo=5, n_folds=3, grid=(0.0, 0.5, 0.9), rng=rng, d=2)
    assert best in (0.0, 0.5, 0.9)
    assert set(score_table) == {0.0, 0.5, 0.9}
    assert len(seen_pools) >= 2
    for i in range(0, len(seen_pools) - 1, 2):
        train_pool, test_pool = seen_pools[i], seen_pools[i + 1]
        for key in neurons_enc:
            for c in common_cats:
                assert not (set(train_pool[key][c].tolist()) & set(test_pool[key][c].tolist()))


def test_pseudopopulation_demixed_arm_fits_on_encoding_training_pseudo_trials_only(monkeypatch):
    n_units, n_per_cat, n_pseudo = 6, 12, 8
    neurons_enc = {f"s1__u{u}": {c: 1.0 + u * 1000 + c * 100 + np.arange(n_per_cat) for c in (1, 2, 3)}
                   for u in range(n_units)}
    rng = np.random.default_rng(40)
    neurons_maint = {key: {c: rng.poisson(3.0, (n_per_cat, 4)).astype(float) for c in (1, 2, 3)}
                     for key in neurons_enc}
    collected = {"enc": neurons_enc, "maint": neurons_maint, "n_sessions_used": 1, "n_patients": 1}
    monkeypatch.setattr(target_module, "_collect_region_neurons", lambda *args, **kwargs: collected)
    pools, fits = [], []
    real_split, real_axes = target_module._split_trial_pool, target_module.memorandum_ridge_axes
    monkeypatch.setattr(target_module, "N_PSEUDO_TRIALS_PER_CATEGORY", n_pseudo)

    def spy_split(*args):
        pools.append(real_split(*args))
        return pools[-1]

    def spy_axes(Z, category, lam, d):
        fits.append(Z)
        return real_axes(Z, category, lam, d)

    monkeypatch.setattr(target_module, "_split_trial_pool", spy_split)
    monkeypatch.setattr(target_module, "memorandum_ridge_axes", spy_axes)
    out = _pseudopopulation_geometry("dandi_000469", "hippocampus")
    assert out["status"] == "computed" and out["category_subspace_dimensionality"] == 4
    fit = fits[0]
    assert fit.shape == (3 * n_pseudo, 1, n_units)
    train_pool = pools[0][0]
    rates = (fit[:, 0, :] / 2.0) ** 2 - 3.0 / 8.0
    for u, key in enumerate(sorted(neurons_enc)):
        allowed = {neurons_enc[key][c][i] for c in (1, 2, 3) for i in train_pool[key][c]}
        assert set(np.round(rates[:, u], 6)) <= {round(v, 6) for v in allowed}
