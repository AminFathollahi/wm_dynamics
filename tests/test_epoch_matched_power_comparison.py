"""Guards on the two things the epoch-matched-power comparison cannot get wrong:
the within-session pairing across epochs, and the restriction to the lags both
epochs reach.

A cross-epoch paired contrast that silently pairs the wrong sessions, or that
tests a lag one epoch never reaches, produces a number that looks like a result
and is not one. Both failures are silent under any check that only inspects the
output's shape, so they are checked here against the mechanism.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _directory in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from run_epoch_matched_power_comparison import (  # noqa: E402
    DECIDING_WIDTH_BINS, LAG_PATH, apply_single_fdr_pass, d_perm_series,
    existence_pvalues_by_lag, fitted_rows_by_key, matched_lag_range,
    paired_epoch_contrast, paired_epoch_series, shared_breakpoint_both_epochs,
)
from state_persistence import paired_vs_null_contrast  # noqa: E402


def _row(dataset, session, structure, epoch, lags, observed, null, status="fitted"):
    return {
        "dataset": dataset, "patient": session, "session": session, "structure": structure,
        "epoch": epoch, "bin_ms": 100.0, "window_mode": "native", "window_s": 0.1 * (max(lags) + 3),
        "n_trials": 30, "n_units": 20, "n_bins": max(lags) + 3, "width_bins": DECIDING_WIDTH_BINS,
        "profile": {"status": status,
                    "lags": {str(lag): {"r_median": observed[i]} for i, lag in enumerate(lags)}},
        "null_poisson": None,
        "null_permutation": {"lags": {str(lag): {"r_null_median": null[i]} for i, lag in enumerate(lags)}},
    }


@pytest.fixture
def two_epoch_rows():
    """Three sessions, encoding reaching lags 3-5 and delay reaching lags 3-7, with
    a per-session encoding-minus-delay difference that is distinct per session so
    a mispairing changes the paired values."""
    rows = []
    for index, session in enumerate(("s1", "s2", "s3")):
        rows.append(_row("corpus_a", session, "pooled", "encoding", [3, 4, 5],
                         [0.30 + index, 0.20 + index, 0.10 + index], [0.05, 0.05, 0.05]))
        rows.append(_row("corpus_a", session, "pooled", "delay", [3, 4, 5, 6, 7],
                         [0.28 + 2 * index, 0.18 + 2 * index, 0.08 + 2 * index, 0.05, 0.02],
                         [0.05, 0.05, 0.05, 0.05, 0.05]))
    return rows


def test_pairing_is_a_key_join_and_survives_row_reordering(two_epoch_rows):
    encoding = fitted_rows_by_key(two_epoch_rows, "encoding", DECIDING_WIDTH_BINS)
    delay = fitted_rows_by_key(two_epoch_rows, "delay", DECIDING_WIDTH_BINS)
    keys, encoding_series, delay_series = paired_epoch_series(encoding, delay)

    shuffled = list(reversed(two_epoch_rows))
    keys_shuffled, encoding_shuffled, delay_shuffled = paired_epoch_series(
        fitted_rows_by_key(shuffled, "encoding", DECIDING_WIDTH_BINS),
        fitted_rows_by_key(shuffled, "delay", DECIDING_WIDTH_BINS))

    assert keys == keys_shuffled
    assert encoding_series == encoding_shuffled
    assert delay_series == delay_shuffled

    for key, encoding_one, delay_one in zip(keys, encoding_series, delay_series):
        assert encoding_one[3] == encoding[key]["profile"]["lags"]["3"]["r_median"] - 0.05
        assert delay_one[3] == delay[key]["profile"]["lags"]["3"]["r_median"] - \
            delay[key]["null_permutation"]["lags"]["3"]["r_null_median"]


def test_a_mispaired_call_is_rejected_rather_than_silently_averaged(two_epoch_rows):
    encoding = fitted_rows_by_key(two_epoch_rows, "encoding", DECIDING_WIDTH_BINS)
    delay = fitted_rows_by_key(two_epoch_rows, "delay", DECIDING_WIDTH_BINS)
    keys, encoding_series, delay_series = paired_epoch_series(encoding, delay)
    with pytest.raises(ValueError):
        paired_epoch_contrast(keys, encoding_series, delay_series[:-1], [3, 4, 5])


def test_mispairing_the_sessions_changes_the_paired_values(two_epoch_rows):
    """A rotated pairing must not produce the same per-lag differences as the
    correct one, or the guard above would be guarding nothing."""
    encoding = fitted_rows_by_key(two_epoch_rows, "encoding", DECIDING_WIDTH_BINS)
    delay = fitted_rows_by_key(two_epoch_rows, "delay", DECIDING_WIDTH_BINS)
    keys, encoding_series, delay_series = paired_epoch_series(encoding, delay)
    correct = [e[3] - d[3] for e, d in zip(encoding_series, delay_series)]
    rotated = [e[3] - d[3] for e, d in zip(encoding_series, delay_series[1:] + delay_series[:1])]
    assert correct != rotated


def test_matched_lag_range_is_the_intersection_both_epochs_reach(two_epoch_rows):
    encoding = fitted_rows_by_key(two_epoch_rows, "encoding", DECIDING_WIDTH_BINS)
    delay = fitted_rows_by_key(two_epoch_rows, "delay", DECIDING_WIDTH_BINS)
    _, encoding_series, delay_series = paired_epoch_series(encoding, delay)
    assert matched_lag_range(encoding_series, delay_series) == [3, 4, 5]
    assert matched_lag_range(delay_series) == [3, 4, 5, 6, 7]


def test_a_lag_one_epoch_cannot_reach_is_rejected(two_epoch_rows):
    encoding = fitted_rows_by_key(two_epoch_rows, "encoding", DECIDING_WIDTH_BINS)
    delay = fitted_rows_by_key(two_epoch_rows, "delay", DECIDING_WIDTH_BINS)
    keys, encoding_series, delay_series = paired_epoch_series(encoding, delay)
    with pytest.raises(ValueError):
        paired_epoch_contrast(keys, encoding_series, delay_series, [3, 4, 5, 6])


def test_rows_without_a_permutation_null_never_enter_the_pairing(two_epoch_rows):
    two_epoch_rows[0]["null_permutation"] = None
    two_epoch_rows[2]["profile"]["status"] = "width_exceeds_epoch"
    encoding = fitted_rows_by_key(two_epoch_rows, "encoding", DECIDING_WIDTH_BINS)
    delay = fitted_rows_by_key(two_epoch_rows, "delay", DECIDING_WIDTH_BINS)
    keys, _, _ = paired_epoch_series(encoding, delay)
    assert [key[2] for key in keys] == ["s3"]


def test_shared_breakpoint_is_one_location_for_both_epochs():
    """A hinge planted at the same lag in both epochs, with different slopes on
    either side, must be recovered once rather than twice."""
    lags = list(range(3, 18))
    def hinge(early, late, floor):
        return {lag: floor + early * min(lag, 10) * 0.1 + late * max(0, lag - 10) * 0.1 for lag in lags}
    fit = shared_breakpoint_both_epochs(
        {"encoding": hinge(-0.5, 0.0, 0.4), "delay": hinge(-0.2, 0.0, 0.2)}, lags)
    assert fit["breakpoint_bins"] == 10
    assert set(fit["by_epoch"]) == {"encoding", "delay"}
    assert fit["by_epoch"]["encoding"]["early_slope_r_per_s"] < \
        fit["by_epoch"]["delay"]["early_slope_r_per_s"]


def test_resampling_existence_path_reproduces_the_project_contrast():
    """The resampling arms skip the bootstrap interval and share sign draws across
    lags for speed. The p-value they consume must still be the project's own
    existence p-value, to Monte Carlo error."""
    rng = np.random.default_rng(7)
    series = [{3: float(v3), 4: float(v4)} for v3, v4 in
              zip(rng.normal(0.05, 0.08, 40), rng.normal(0.01, 0.08, 40))]
    fast = existence_pvalues_by_lag(series, [3, 4], np.random.default_rng(11))
    for lag in (3, 4):
        reference = paired_vs_null_contrast([s[lag] for s in series], [0.0] * len(series),
                                             direction="greater")
        assert fast[lag]["mean_diff"] == pytest.approx(reference["mean_diff"], abs=1e-12)
        assert fast[lag]["p_value"] == pytest.approx(reference["p_value"], abs=0.01)


def test_single_correction_pass_covers_exactly_the_lags_tested():
    by_lag = {3: {"status": "tested", "p_value": 0.001}, 4: {"status": "tested", "p_value": 0.5},
              5: {"status": "underpowered_by_construction"}}
    summary = apply_single_fdr_pass(by_lag)
    assert summary["lags_bins_tested"] == [3, 4]
    assert summary["n_lags_tested"] == 2
    assert "fdr_q_value" not in summary["by_lag"]["5"]
    assert summary["by_lag"]["3"]["fdr_q_value"] == pytest.approx(0.002)


@pytest.mark.skipif(not LAG_PATH.exists(), reason="lag artifact not present")
def test_delivered_artifact_pairs_exactly_and_restricts_to_the_shorter_epoch():
    """The comparison's two load-bearing facts, read off the delivered artifact:
    the encoding arm's fitted rows are a strict subset of the delay arm's, and the
    lags both epochs reach stop where the shorter encoding window stops."""
    rows = json.loads(LAG_PATH.read_text())["human_lag_rows"]
    encoding = fitted_rows_by_key(rows, "encoding", DECIDING_WIDTH_BINS)
    delay = fitted_rows_by_key(rows, "delay", DECIDING_WIDTH_BINS)
    assert set(encoding) <= set(delay)
    keys, encoding_series, delay_series = paired_epoch_series(encoding, delay)
    assert len(keys) == len(encoding)
    lags = matched_lag_range(encoding_series, delay_series)
    assert lags == list(range(3, 18))
    assert max(lags) < max(matched_lag_range(delay_series))
    for key, series in zip(keys, encoding_series):
        assert d_perm_series(encoding[key]) == series
