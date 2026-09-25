import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_pfc4_content_cell_checks as checks  # noqa: E402


def test_r_squared_above_null_recovers_a_planted_signal():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(scale=0.2, size=n)
    result = checks._r_squared_above_null(x, y, 100, np.random.default_rng(1))

    assert result["n_trials"] == n
    assert result["r_squared"] > 0.9
    assert result["r_squared_above_null"] > 0.8


def test_r_squared_above_null_corrects_dimensionality_inflation_on_pure_noise():
    rng = np.random.default_rng(2)
    n = 60
    y = rng.normal(size=n)
    low_dim = rng.normal(size=n)
    high_dim_labels = rng.integers(0, 8, size=n)
    high_dim = (high_dim_labels[:, None] == np.unique(high_dim_labels)[None, :]).astype(float)

    low = checks._r_squared_above_null(low_dim, y, 200, np.random.default_rng(3))
    high = checks._r_squared_above_null(high_dim, y, 200, np.random.default_rng(4))

    assert high["r_squared"] > low["r_squared"]
    assert abs(low["r_squared_above_null"]) < 0.15
    assert abs(high["r_squared_above_null"]) < 0.15


def test_r_squared_above_null_returns_none_below_the_trial_floor():
    result = checks._r_squared_above_null([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 50, np.random.default_rng(0))

    assert result is None


def test_unit_counts_builds_histogram_per_animal():
    class _FakeIter:
        def __call__(self, root):
            entries = [
                {"patient": "rr014", "session": "rr014_a", "n_units": 3},
                {"patient": "rr014", "session": "rr014_b", "n_units": 5},
                {"patient": "rr015", "session": "rr015_a", "n_units": 7},
            ]
            yield from entries

    original = checks.iter_pfc4
    checks.iter_pfc4 = _FakeIter()
    try:
        per_session, histogram = checks._unit_counts(Path("."))
    finally:
        checks.iter_pfc4 = original

    assert per_session == {"rr014_a": 3, "rr014_b": 5, "rr015_a": 7}
    assert histogram["rr014"]["n_sessions"] == 2
    assert histogram["rr014"]["unit_count_histogram"]["3"] == 1
    assert histogram["rr014"]["unit_count_histogram"]["5"] == 1
    assert histogram["rr015"]["unit_count_histogram"]["7"] == 1


def _cell(mean_above_null, p_value, ci, mdd, n_sessions=10):
    return {
        "status": "computed",
        "n_sessions": n_sessions,
        "mean_alignment_above_null": mean_above_null,
        "p_value": p_value,
        "session_cluster_bootstrap_interval_95pct": list(ci),
        "minimum_detectable_difference_80pct_power": mdd,
    }


def _delivered_per_corpus():
    return {
        "panichello_2024_macaque_lPFC": {
            "gain_total_spike_count": {"monkey_A": _cell(0.09, 0.001, (0.04, 0.15), 0.08)},
            "memorandum_content": {"monkey_A": _cell(-0.03, 0.9, (-0.07, 0.01), 0.06)},
        },
        checks.PFC4_CORPUS: {
            "gain_total_spike_count": {"rr014": _cell(0.09, 0.001, (0.05, 0.14), 0.06)},
            "memorandum_content": {"rr014": _cell(0.05, 0.02, (0.001, 0.09), 0.07)},
        },
    }


def test_finalize_with_pfc4_reports_unstable_when_verdict_moves():
    delivered_per_corpus = _delivered_per_corpus()
    full_content = {"rr014": _cell(0.05, 0.02, (0.001, 0.09), 0.07)}
    full_gain = {"rr014": _cell(0.09, 0.001, (0.05, 0.14), 0.06)}
    restricted_content = {"rr014": _cell(0.03, 0.5, (-0.02, 0.08), 0.07)}
    restricted_gain = {"rr014": _cell(0.10, 0.001, (0.06, 0.15), 0.06)}

    full_result = checks._finalize_with_pfc4(delivered_per_corpus, full_gain, full_content)
    restricted_result = checks._finalize_with_pfc4(delivered_per_corpus, restricted_gain, restricted_content)

    full_status = checks._status_or_verdict(full_result["memorandum_content"]["rr014"])
    restricted_status = checks._status_or_verdict(restricted_result["memorandum_content"]["rr014"])

    assert full_status != restricted_status


def test_status_or_verdict_reports_not_computable_without_a_verdict_key():
    cell = {"status": "not_computable", "n_sessions": 2, "reason": "fewer than the session floor"}

    assert checks._status_or_verdict(cell) == "not_computable"
