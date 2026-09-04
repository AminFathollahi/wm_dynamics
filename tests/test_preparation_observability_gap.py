"""Fails if the two preparation arms stop being matched on the things that make the
comparison a comparison: the delay window length, the number of bins in it, the
estimator width, the bin width, the anatomical pooling level, the epoch, and the
firing-rate thinning that puts the animal arm on the human arm's rate. Also pins the
amplitude statistic itself, so a change in how the shared-component amplitude is
formed cannot pass silently.

These read the delivered persistence artifacts and never write to them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from run_preparation_observability_gap import (  # noqa: E402
    BIN_MS, EPOCH, EXPECTED_N_BINS, LAG_PATH, MATCHED_CELLS, PERSISTENCE_PATH, STRUCTURE,
    WIDTH_BINS, WINDOW_S, probability_of_superiority, reachable_lags, row_amplitude,
    select_arm_rows,
)


@pytest.fixture(scope="module")
def arms() -> tuple[list[dict], list[dict]]:
    if not LAG_PATH.exists():
        pytest.skip("the persistence lag artifact is not on disk")
    artifact = json.loads(LAG_PATH.read_text())
    human, _ = select_arm_rows(artifact["human_lag_rows"])
    mouse, _ = select_arm_rows(artifact["alm_lag_rows"])
    return human, mouse


def test_both_arms_are_non_empty(arms):
    human, mouse = arms
    assert human, "the human arm selected no rows at the shared estimator cell"
    assert mouse, "the animal arm selected no rows at the shared estimator cell"


def test_arms_share_window_bin_count_and_estimator_width(arms):
    for rows in arms:
        assert {row["window_s"] for row in rows} == {WINDOW_S}
        assert {row["n_bins"] for row in rows} == {EXPECTED_N_BINS}
        assert {row["width_bins"] for row in rows} == {WIDTH_BINS}
        assert {row["bin_ms"] for row in rows} == {BIN_MS}
        assert {row["structure"] for row in rows} == {STRUCTURE}
        assert {row["epoch"] for row in rows} == {EPOCH}


def test_arms_share_a_contiguous_reachable_lag_range(arms):
    human, mouse = arms
    common = sorted(set.intersection(*(reachable_lags(row) for row in human + mouse)))
    assert common, "the two arms share no reachable lag"
    assert common == list(range(common[0], common[-1] + 1))
    assert common[0] == WIDTH_BINS, "the shortest reachable lag is the estimator width"
    assert common[-1] == EXPECTED_N_BINS - WIDTH_BINS


def test_every_selected_row_carries_a_fitted_profile_and_a_permutation_null(arms):
    for rows in arms:
        for row in rows:
            assert row["profile"]["status"] == "fitted"
            assert row["null_permutation"] is not None


def test_animal_arm_is_thinned_to_the_human_firing_rate():
    if not PERSISTENCE_PATH.exists():
        pytest.skip("the persistence artifact is not on disk")
    matched = json.loads(PERSISTENCE_PATH.read_text())["matched_sensitivity_alm"]
    keep = matched["rate_matched_keep_probability"]
    assert 0.0 < keep < 1.0, "the animal arm is not thinned at all"
    achieved = keep * matched["alm_median_rate_hz"]
    assert abs(achieved - matched["human_median_rate_hz"]) < 1e-6, (
        "the thinning no longer lands the animal arm on the human median firing rate"
    )


def test_matched_cells_are_feasible_in_both_arms(arms):
    human, mouse = arms
    for units, trials in MATCHED_CELLS:
        for rows, name in ((human, "human"), (mouse, "animal")):
            eligible = sum(1 for row in rows if row["n_units"] >= units and row["n_trials"] >= trials)
            assert eligible > 0, f"no {name} session can supply the matched cell {units}x{trials}"


def test_amplitude_is_the_observed_minus_null_mean_over_the_requested_lags():
    row = {
        "profile": {"status": "fitted", "lags": {"3": {"r_median": 0.5}, "4": {"r_median": 0.3}}},
        "null_permutation": {"lags": {"3": {"r_null_median": 0.1}, "4": {"r_null_median": 0.1}}},
    }
    assert row_amplitude(row, [3, 4]) == pytest.approx(0.3)
    assert row_amplitude(row, [3]) == pytest.approx(0.4)
    assert row_amplitude(row, [3, 4, 5]) is None, "a lag the row does not reach must exclude the row"


def test_amplitude_rejects_rows_without_a_permutation_null():
    assert row_amplitude({"profile": {"status": "fitted", "lags": {}}, "null_permutation": None}, [3]) is None
    assert row_amplitude({"profile": {"status": "width_exceeds_epoch"},
                          "null_permutation": {"lags": {}}}, [3]) is None


def test_probability_of_superiority_is_a_proper_overlap_statistic():
    assert probability_of_superiority(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert probability_of_superiority(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert probability_of_superiority(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
