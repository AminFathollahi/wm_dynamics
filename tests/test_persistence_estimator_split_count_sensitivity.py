"""Tests for scripts/run_persistence_estimator_split_count_sensitivity.py's
pre-declared branch logic: does starving the estimator's split/replicate
counts move a species' d_perm slope enough to matter for the gap it is
compared against?"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_persistence_estimator_split_count_sensitivity import (  # noqa: E402
    _apply_resume_state, _classify, _fraction_closed, _panichello_branch, _per_session_slope_triplet,
)


def _tested_change(mean_diff: float, significant: bool) -> dict:
    return {"status": "tested", "mean_diff_down_minus_reference": mean_diff, "significant": significant,
            "ci_lower": mean_diff - 0.02, "ci_upper": mean_diff + 0.02}


def test_null_change_with_small_fraction_clears_the_species_claim():
    change = _tested_change(mean_diff=0.005, significant=False)
    fraction = _fraction_closed(change, gap=0.178)
    assert _classify(change, fraction) == "estimator_settings_do_not_account_for_the_cross_species_difference"


def test_significant_positive_change_closing_most_of_the_gap_convicts_the_settings():
    change = _tested_change(mean_diff=0.10, significant=True)
    fraction = _fraction_closed(change, gap=0.178)
    assert fraction["fraction"] > 1.0 / 3.0
    assert _classify(change, fraction) == "cross_species_difference_is_substantially_a_settings_difference"


def test_intermediate_fraction_is_partially_attributable_not_forced_to_either_branch():
    change = _tested_change(mean_diff=0.04, significant=True)
    fraction = _fraction_closed(change, gap=0.178)
    assert 0.10 < fraction["fraction"] < 1.0 / 3.0
    assert _classify(change, fraction) == "partially_attributable"


def test_significant_negative_change_is_partially_attributable_not_cleared():
    # A change that is statistically significant but moves AWAY from the macaque (negative fraction)
    # closes none of the gap, yet the pre-declared clearance branch requires "not significant" literally
    # -- so this lands in partially_attributable rather than the clearance branch, even though it is, if
    # anything, stronger evidence against a settings account than a null would be. This is a real
    # ambiguity in the pre-declared rule's wording (see the implementation report), and this test pins
    # down what the code as written actually does with it rather than what might seem more intuitive.
    change = _tested_change(mean_diff=-0.03, significant=True)
    fraction = _fraction_closed(change, gap=0.178)
    assert fraction["fraction"] < 0.0
    assert _classify(change, fraction) == "partially_attributable"


def _default_test(significant_positive: bool, significant_negative: bool, status: str = "tested") -> dict:
    return {"status": status, "significant_positive": significant_positive, "significant_negative": significant_negative}


def _paired(significant: bool, status: str = "tested") -> dict:
    return {"status": status, "significant": significant}


def test_macaque_branch_default_not_significant_is_withdrawn():
    # m_default is not distinguishable from zero at all (neither significantly positive nor negative):
    # the cross-species difference is withdrawn regardless of what the paired change itself did.
    branch = _panichello_branch(_paired(significant=False), _default_test(False, False))
    assert branch == "macaque_positive_d_perm_slope_does_not_survive_matched_estimator_settings"
    # Also withdrawn even if the paired change happened to be significant -- the withdrawal condition is
    # about m_default, not about the paired change.
    branch2 = _panichello_branch(_paired(significant=True), _default_test(False, False))
    assert branch2 == "macaque_positive_d_perm_slope_does_not_survive_matched_estimator_settings"


def test_macaque_branch_default_significantly_negative_reverses_and_is_never_folded_into_withdrawn():
    # Direction-aware branching (0.29): a significant NEGATIVE m_default is stronger and different
    # evidence than a null -- it must land in its own branch, never in the withdrawal branch above, and
    # never in an ambiguous middle branch, regardless of the paired change's own significance.
    branch = _panichello_branch(_paired(significant=False), _default_test(False, True))
    assert branch == "macaque_d_perm_slope_reverses_sign_under_matched_estimator_settings"
    branch2 = _panichello_branch(_paired(significant=True), _default_test(False, True))
    assert branch2 == "macaque_d_perm_slope_reverses_sign_under_matched_estimator_settings"


def test_macaque_branch_default_positive_and_paired_change_null_is_not_an_artifact():
    branch = _panichello_branch(_paired(significant=False), _default_test(True, False))
    assert branch == "macaque_positive_d_perm_slope_is_not_an_estimator_setting_artifact"


def test_macaque_branch_default_positive_and_paired_change_significant_is_magnitude_dependent():
    branch = _panichello_branch(_paired(significant=True), _default_test(True, False))
    assert branch == "macaque_positive_d_perm_slope_survives_but_its_magnitude_is_estimator_setting_dependent"


def test_macaque_branch_not_computable_when_either_input_is_untested():
    assert _panichello_branch(_paired(significant=True, status="underpowered_by_construction"),
                               _default_test(True, False)) == "not_computable"
    assert _panichello_branch(_paired(significant=True),
                               _default_test(True, False, status="not_computed")) == "not_computable"


def test_per_session_slope_triplet_normalises_string_lag_keys_from_a_json_round_trip():
    # A row loaded back from this module's own on-disk session cache has been through json.dumps then
    # json.loads, which turns every int dict key into a string (JSON has no integer keys) -- the exact
    # shape a resumed run's cached rows arrive in. Without normalising back to int keys, the lag-range
    # comparison in per_session_slopes_in_range (int <= key <= int) raises a TypeError only a resumed run
    # would ever hit. This constructs that string-keyed shape directly, the way json.loads would produce
    # it, rather than round-tripping through json.dumps/json.loads for the same effect.
    profile = {
        "status": "fitted",
        "lags": {str(lag): {"r_median": 0.5 - 0.01 * lag} for lag in range(2, 10)},
    }
    null_permutation = {
        "lags": {str(lag): {"r_null_median": 0.3} for lag in range(2, 10)},
    }
    triplet = _per_session_slope_triplet(profile, null_permutation)
    assert triplet is not None
    assert triplet["d_perm_slope"] == pytest.approx(triplet["r_obs_slope"] - triplet["r_null_slope"])


def test_resume_carries_panichello_session_cache_even_when_human_and_alm_are_not_yet_complete():
    # The regression this test exists for: a prior version of the resume logic only copied
    # panichello_session_rows inside the human/ALM branch, so a resume that found human/ALM NOT yet
    # complete on disk (this case) started the panichello cache from an empty dict regardless of what
    # was already cached -- silently discarding real, expensive-to-recompute work.
    prior_output = {
        "panichello_session_rows": {"210921": {"session": "210921", "reference": {}, "down": {}}},
    }
    output, human_and_alm_done = _apply_resume_state({}, prior_output)
    assert human_and_alm_done is False
    assert output["panichello_session_rows"] == prior_output["panichello_session_rows"]


def test_resume_carries_panichello_session_cache_when_human_and_alm_are_complete():
    prior_output = {
        "human_delay_arm": {"branch": "x"}, "alm_arm": {"branch": "y"},
        "corpora_completed": ["human_delay", "alm"],
        "panichello_session_rows": {
            "210921": {"session": "210921", "reference": {}, "down": {}},
            "210927": {"session": "210927", "reference": {}, "down": {}},
        },
    }
    output, human_and_alm_done = _apply_resume_state({}, prior_output)
    assert human_and_alm_done is True
    assert output["human_delay_arm"] == {"branch": "x"}
    assert output["alm_arm"] == {"branch": "y"}
    # The exact defect this test pins down: every cached session must survive the resume, not just the
    # ones that happen to share a branch with the human/ALM copy-over.
    assert set(output["panichello_session_rows"].keys()) == {"210921", "210927"}


def test_resume_from_empty_prior_output_starts_clean():
    output, human_and_alm_done = _apply_resume_state({}, {})
    assert human_and_alm_done is False
    assert "panichello_session_rows" not in output
