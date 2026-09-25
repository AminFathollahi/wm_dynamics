import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_within_animal_component_identity as analysis  # noqa: E402


def _cell(alignment, draws, status="computed"):
    draws = np.asarray(draws, dtype=float)
    return {"status": status, "alignment": alignment, "null_draws": draws}


def test_pool_within_animal_requires_the_session_floor():
    cells = [_cell(0.5, [0.1, 0.1, 0.1]) for _ in range(3)]
    pooled = analysis._pool_within_animal(cells, "seed")

    assert pooled["status"] == "not_computable"
    assert pooled["n_sessions"] == 3
    assert str(analysis.MIN_SESSIONS_PER_ANIMAL) in pooled["reason"]


def test_pool_within_animal_averages_sessions_equally():
    cells = [
        _cell(0.9, [0.1] * 20),
        _cell(0.7, [0.1] * 20),
        _cell(0.2, [0.1] * 20),
        _cell(0.4, [0.1] * 20),
    ]
    pooled = analysis._pool_within_animal(cells, "seed")

    assert pooled["status"] == "computed"
    assert pooled["n_sessions"] == 4
    assert pooled["mean_alignment"] == np.mean([0.9, 0.7, 0.2, 0.4])
    assert pooled["mean_alignment_above_null"] == np.mean([0.8, 0.6, 0.1, 0.3])
    assert pooled["minimum_detectable_difference_80pct_power"] > 0


def test_pool_within_animal_ignores_not_computable_cells_and_none():
    cells = [
        _cell(0.9, [0.1] * 20),
        _cell(0.7, [0.1] * 20),
        _cell(0.2, [0.1] * 20),
        _cell(0.4, [0.1] * 20),
        None,
        _cell(0.0, [], status="not_computable"),
    ]
    pooled = analysis._pool_within_animal(cells, "seed")

    assert pooled["status"] == "computed"
    assert pooled["n_sessions"] == 4


def _computed_candidate(p_value, ci):
    return {
        "status": "computed",
        "p_value": p_value,
        "mean_alignment_above_null": (ci[0] + ci[1]) / 2,
        "session_cluster_bootstrap_interval_95pct": list(ci),
        "minimum_detectable_difference_80pct_power": 0.05,
    }


def test_finalize_clears_only_when_fdr_and_ci_both_agree():
    per_corpus = {
        "panichello_2024_macaque_lPFC": {
            "gain_total_spike_count": {"monkey_A": _computed_candidate(0.001, (0.1, 0.3))},
            "memorandum_content": {"monkey_A": _computed_candidate(0.02, (-0.05, 0.2))},
        },
        "watters_2026_macaque_multi_object": {
            "gain_total_spike_count": {"Elgar": _computed_candidate(0.8, (-0.1, 0.1))},
            "memorandum_content": {"Elgar": _computed_candidate(0.9, (-0.2, -0.15))},
        },
    }

    result = analysis._finalize(per_corpus)

    gain_a = result["panichello_2024_macaque_lPFC"]["gain_total_spike_count"]["monkey_A"]
    content_a = result["panichello_2024_macaque_lPFC"]["memorandum_content"]["monkey_A"]
    gain_elgar = result["watters_2026_macaque_multi_object"]["gain_total_spike_count"]["Elgar"]
    content_elgar = result["watters_2026_macaque_multi_object"]["memorandum_content"]["Elgar"]

    assert gain_a["verdict"] == "cleared"
    assert content_a["verdict"] == "clears_neither"
    assert gain_elgar["verdict"] == "not_cleared_no_internal_reference"
    assert content_elgar["verdict"] == "clears_neither"


def test_finalize_bounds_a_non_clearing_content_cell_against_its_own_animals_gain():
    per_corpus = {
        "panichello_2024_macaque_lPFC": {
            "gain_total_spike_count": {"monkey_A": _computed_candidate(0.001, (0.1, 0.3))},
            "memorandum_content": {"monkey_A": _computed_candidate(0.9, (-0.02, 0.02))},
        },
        "watters_2026_macaque_multi_object": {
            "gain_total_spike_count": {},
            "memorandum_content": {},
        },
    }
    per_corpus["panichello_2024_macaque_lPFC"]["memorandum_content"]["monkey_A"]["minimum_detectable_difference_80pct_power"] = 0.05

    result = analysis._finalize(per_corpus)
    content_a = result["panichello_2024_macaque_lPFC"]["memorandum_content"]["monkey_A"]

    assert content_a["bound"] == "bounded_negative"
    assert content_a["internal_reference_gain_effect"] == 0.2


def test_finalize_reports_inconclusive_when_mdd_exceeds_the_gain_reference():
    per_corpus = {
        "panichello_2024_macaque_lPFC": {
            "gain_total_spike_count": {"monkey_A": _computed_candidate(0.001, (0.01, 0.03))},
            "memorandum_content": {"monkey_A": _computed_candidate(0.9, (-0.02, 0.02))},
        },
        "watters_2026_macaque_multi_object": {"gain_total_spike_count": {}, "memorandum_content": {}},
    }
    per_corpus["panichello_2024_macaque_lPFC"]["memorandum_content"]["monkey_A"]["minimum_detectable_difference_80pct_power"] = 0.5

    result = analysis._finalize(per_corpus)
    content_a = result["panichello_2024_macaque_lPFC"]["memorandum_content"]["monkey_A"]

    assert content_a["bound"] == "inconclusive"


def test_finalize_marks_bound_unavailable_when_the_animal_has_no_gain_cell():
    per_corpus = {
        "panichello_2024_macaque_lPFC": {
            "gain_total_spike_count": {"monkey_J": {"status": "not_computable"}},
            "memorandum_content": {"monkey_J": _computed_candidate(0.9, (-0.02, 0.02))},
        },
        "watters_2026_macaque_multi_object": {"gain_total_spike_count": {}, "memorandum_content": {}},
    }

    result = analysis._finalize(per_corpus)
    content_j = result["panichello_2024_macaque_lPFC"]["memorandum_content"]["monkey_J"]

    assert content_j["bound"] == "inconclusive_no_internal_reference"


def test_fit_cell_reports_not_computable_below_the_trial_floor():
    activity = np.abs(np.random.default_rng(0).normal(size=(4, 5))) + 1.0
    target = np.array([0.0, 1.0, 0.0, 1.0])

    result = analysis._fit_cell(activity, target, "categorical", 10, "seed")

    assert result["status"] == "not_computable"
    assert result["n_trials"] == 4


def test_fit_cell_recovers_a_planted_continuous_target():
    rng = np.random.default_rng(3)
    n = 200
    labels = rng.normal(size=n)
    values = rng.normal(0.0, 0.15, (n, 8))
    values[:, 0] += 3.0
    values[:, 1] += labels
    activity = values / np.linalg.norm(values, axis=1, keepdims=True)

    result = analysis._fit_cell(activity, labels, "continuous", 199, "seed")

    assert result["status"] == "computed"
    assert result["target_kind"] == "continuous"
    assert result["alignment_above_null"] > 0.1
    assert result["p_value"] <= 0.05


def test_compare_flags_numeric_and_structural_mismatches():
    reference = {"a": {"b": 0.5, "c": [1, 2]}}
    identical = {"a": {"b": 0.5 + 1e-12, "c": [1, 2]}}
    different_value = {"a": {"b": 0.6, "c": [1, 2]}}
    different_keys = {"a": {"b": 0.5}}

    same_diffs, same_max = [], [0.0]
    analysis._compare(reference, identical, "root", same_diffs, same_max)
    assert same_diffs == []

    value_diffs, value_max = [], [0.0]
    analysis._compare(reference, different_value, "root", value_diffs, value_max)
    assert value_diffs
    assert value_max[0] > 0

    key_diffs, key_max = [], [0.0]
    analysis._compare(reference, different_keys, "root", key_diffs, key_max)
    assert key_diffs
