#!/usr/bin/env python3
"""Runnable, assert-based check for scripts/run_ram_randomised_prestimulation_moderation.py.

No test framework: run directly with
    /home/amin/miniconda3/envs/wm_dynamics/bin/python tests/test_run_ram_randomised_prestimulation_moderation.py

Exercises the bias-only construction (_add_bias_only_moderator, the exact function
scripts/run_ram_randomised_prestimulation_moderation.py's main() calls) on two synthetic
datasets, each with several subjects who each contribute two sessions (so the bias-only
moderator -- a session mean -- genuinely varies within a subject, exactly as it does for any
open-loop RAM subject with more than one usable session):

  scenario "between_session": the interaction effect on the outcome is driven entirely by each
    session's own mean moderator (a between-session quantity); the native, trial-level moderator
    is that same session mean plus tiny noise. The bias-only arm must REPRODUCE the native arm's
    sign and significance.

  scenario "within_session": each session still has its own mean moderator (so the bias-only
    construction is not trivially constant), but the outcome depends only on each word's own
    DEVIATION from its session's mean -- a purely within-session quantity the bias-only
    construction throws away by design. The native arm must still detect the effect; the
    bias-only arm must NULL IT OUT (not significant).
"""
from __future__ import annotations

import sys
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_ram_randomised_prestimulation_moderation as mod  # noqa: E402

# Small, fast settings for a unit test -- the production run uses larger values.
mod.N_WITHIN_SUBJECT_NULL_DRAWS = 200
mod.N_BOOT = 500
mod.MIN_WORDS_PER_SUBJECT_FOR_EFFECT = 20

N_SUBJECTS = 14
N_SESSIONS_PER_SUBJECT = 2
N_WORDS_PER_SESSION = 70
EFFECT_SIZE = 1.4


def _make_rows(rng: np.random.Generator, signal: str) -> list[dict]:
    """signal='between_session': outcome ~ stim * session_mean(moderator).
    signal='within_session': outcome ~ stim * (moderator - session_mean(moderator))."""
    rows = []
    for si in range(N_SUBJECTS):
        subject = f"sub-synthetic{si:03d}"
        for se in range(N_SESSIONS_PER_SUBJECT):
            session = f"{subject}/ses-{se}"
            session_mean = rng.normal()
            for w in range(N_WORDS_PER_SESSION):
                stim = float(rng.integers(0, 2))
                within_dev = rng.normal(scale=1.0)
                moderator_native = session_mean + (within_dev if signal == "within_session"
                                                     else within_dev * 0.05)
                driver = session_mean if signal == "between_session" else within_dev
                outcome = EFFECT_SIZE * stim * driver + rng.normal(scale=0.3)
                rows.append({
                    "session": session, "subject": subject,
                    "outcome": float(outcome), "stim": stim, "preceding_stim": float(rng.integers(0, 2)),
                    "moderator_native": float(moderator_native),
                    "serialpos": float(rng.integers(1, 13)), "list_number": float(rng.integers(1, 6)),
                    "phase": rng.choice(["phase_a", "phase_b", "no_stimulation_list"]),
                })
    return rows


def _run_scenario(signal: str, seed: int) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    rows = _make_rows(rng, signal)
    mod._add_bias_only_moderator(rows)
    native = mod._fit_arm(rows, "outcome", "moderator_native", True, f"test|{signal}|native")
    bias_only = mod._fit_arm(rows, "outcome", "moderator_bias_only", True, f"test|{signal}|bias")
    return native, bias_only


def check_bias_only_construction() -> None:
    native_b, bias_b = _run_scenario("between_session", seed=1)
    assert native_b["status"] == "computed", native_b
    assert bias_b["status"] == "computed", bias_b
    assert native_b["significant"], f"between-session scenario: native should be significant, got {native_b}"
    assert bias_b["significant"], f"between-session scenario: bias-only should REPRODUCE native, got {bias_b}"
    assert np.sign(native_b["subject_clustered_mean"]) == np.sign(bias_b["subject_clustered_mean"]), (
        "between-session scenario: bias-only reproduced significance but with the wrong sign")

    native_w, bias_w = _run_scenario("within_session", seed=2)
    assert native_w["status"] == "computed", native_w
    assert native_w["significant"], f"within-session scenario: native should still detect the effect, got {native_w}"
    bias_w_nulled = (bias_w["status"] != "computed") or (not bias_w["significant"])
    assert bias_w_nulled, f"within-session scenario: bias-only should NULL OUT the effect, got {bias_w}"


def check_delivered_artifact() -> None:
    artifact = json.loads((ROOT / "results/randomised_prestimulation_moderation_open_loop.json").read_text())
    assert artifact["scope"]["corpus"].startswith("OpenNeuro ds005489")
    for key in ("session_zero_drop_accounting", "list_zero_drop_accounting",
                "word_zero_drop_accounting", "subject_zero_drop_accounting"):
        assert artifact[key]["reconciles"], key
    sessions = artifact["session_zero_drop_accounting"]
    assert sessions["sessions_seen"] == sessions["sessions_analysed"] + sessions["sessions_refused"]
    assert len(sessions["per_session_status"]) == sessions["sessions_seen"]
    lists = artifact["list_zero_drop_accounting"]
    assert lists["lists_seen"] == lists["lists_analysed"] + lists["lists_refused"]
    assert len(lists["per_list_status"]) == lists["lists_seen"]
    words = artifact["word_zero_drop_accounting"]
    assert words["words_seen"] == words["words_analysed"] + words["words_refused"]
    assert artifact["subject_zero_drop_accounting"]["subjects_with_more_than_one_analysed_session"] > 0

    valid_branches = {
        "pre_stimulation_state_moderates_the_stimulation_effect",
        "moderation_is_between_session_only",
        "restricted_arm_significant_with_its_own_controls_despite_native_null",
        "no_moderation_above_the_reported_bound",
        "underpowered_to_ask",
        "native_significant_but_restricted_arm_disagrees_in_sign",
        "not_computable",
        "branch_criteria_not_matched_see_full_object",
    }
    for outcome in artifact["outcomes"].values():
        assert outcome["branch"] in valid_branches, outcome["branch"]
        for arm_key in ("native", "bias_only", "restricted", "restricted_bias_only",
                        "restricted_nuisance_partialled"):
            arm = outcome[arm_key]
            if arm.get("status") == "computed":
                assert "subject_clustered_mean" in arm and "n_words" in arm
        # a bias-only arm's moderator is session-constant, so its significance flag must be driven
        # by the between-subject null rather than the near-untestable within-subject one
        for arm_key in ("bias_only", "restricted_bias_only"):
            assert outcome[arm_key]["p_value_source"] == "between_subject_permutation"

    # the two currently-known determinations -- pinned so a silent regression in either arm's
    # fit or in _classify_branch's priority ordering is caught rather than passing quietly
    assert artifact["outcomes"]["displacement"]["branch"] == (
        "restricted_arm_significant_with_its_own_controls_despite_native_null")
    assert artifact["outcomes"]["recalled"]["branch"] == "moderation_is_between_session_only"

    composition = artifact["restriction_composition_and_contamination_check"]
    assert composition["n_native_arm_words"] == (
        composition["n_retained_restricted_arm_words"] + composition["n_excluded_preceding_word_stimulated_words"])
    assert 0.0 <= composition["native_arm_contaminated_moderator_fraction"] <= 1.0
    for outcome in artifact["outcomes"].values():
        assert "nuisance_partialling_ladder" in outcome
        assert "restricted_vs_its_bias_only_sign_relationship" in outcome

    # the restricted arm is the one pinned above as significant with its own mdd above its own
    # observed magnitude -- the winner's-curse caveat must therefore be present there
    disp_restricted = artifact["outcomes"]["displacement"]["restricted"]
    assert disp_restricted["effect_magnitude_below_its_own_minimum_detectable_difference"] is True
    assert disp_restricted["winners_curse_caveat"] == mod.WINNERS_CURSE_CAVEAT


def check_restricted_arm_construction() -> None:
    rows = [
        {"preceding_stim": 0.0, "stim": 1.0, "moderator_native": 0.1,
         "serialpos": 2.0, "list_number": 1.0, "phase": "phase_a"},
        {"preceding_stim": 1.0, "stim": 0.0, "moderator_native": -0.1,
         "serialpos": 3.0, "list_number": 1.0, "phase": "phase_b"},
    ]
    restricted = mod._restrict_to_unstimulated_preceding_words(rows)
    assert len(restricted) == 1
    assert all(row["preceding_stim"] == 0.0 for row in restricted)
    assert "preceding_stim" not in mod._covariates(restricted, "moderator_native", False)


def _make_single_session_rows(rng: np.random.Generator, n_subjects: int, effect_size: float) -> list[dict]:
    rows = []
    for si in range(n_subjects):
        subject = f"sub-single{si:03d}"
        session = f"{subject}/ses-0"
        subject_mean = rng.normal()
        for _ in range(60):
            stim = float(rng.integers(0, 2))
            rows.append({
                "session": session, "subject": subject,
                "outcome": float(effect_size * stim * subject_mean + rng.normal(scale=0.3)),
                "stim": stim, "preceding_stim": float(rng.integers(0, 2)),
                "moderator_native": float(subject_mean + rng.normal(scale=0.01)),
                "serialpos": float(rng.integers(1, 13)), "list_number": float(rng.integers(1, 6)),
                "phase": rng.choice(["phase_a", "phase_b", "no_stimulation_list"]),
            })
    return rows


def check_between_subject_null_for_single_session_subjects() -> None:
    """Scenario A: every subject contributes exactly ONE session, so the bias-only moderator (a
    session mean) is fully subject-constant for every one of them -- proves the within-subject
    shuffle is the identity (degenerate, p near 1.0) in exactly this case, per
    BETWEEN_SUBJECT_NULL_REASON. (The subject-clustered partial-correlation POINT ESTIMATE is
    itself uninformative when every subject is single-session -- the moderator column is then
    collinear with that subject's own intercept, so no null, within- or between-subject, can
    recover a planted effect from an all-single-session population; that is a property of the
    point estimate, not of which null is used to test it.)

    Scenario B: a MIX of single- and multi-session subjects, matching this corpus's own structure
    (roughly half single-session) -- proves the between-subject shuffle still correctly detects a
    real planted effect once at least some subjects contribute genuine cross-session variation."""
    rng_a = np.random.default_rng(7)
    rows_a = _make_single_session_rows(rng_a, n_subjects=20, effect_size=2.0)
    mod._add_bias_only_moderator(rows_a)
    bias_a = mod._fit_arm(rows_a, "outcome", "moderator_bias_only", True, "test|single_session|bias",
                           significance_null="between_subject")
    assert bias_a["status"] == "computed", bias_a
    assert bias_a["n_admissible_subjects_with_more_than_one_session"] == 0
    assert bias_a["within_subject_null_is_degenerate_zero_spread"], (
        f"expected a degenerate within-subject null for single-session subjects, got {bias_a}")
    assert bias_a["within_subject_permutation_p"] > 0.9, (
        f"expected a near-1.0 within-subject p from an identity null, got {bias_a}")

    rng_b = np.random.default_rng(11)
    rows_b = _make_single_session_rows(rng_b, n_subjects=10, effect_size=2.0)
    for si in range(14):
        subject = f"sub-multi{si:03d}"
        for se in range(2):
            session = f"{subject}/ses-{se}"
            session_mean = rng_b.normal()
            for _ in range(60):
                stim = float(rng_b.integers(0, 2))
                rows_b.append({
                    "session": session, "subject": subject,
                    "outcome": float(2.0 * stim * session_mean + rng_b.normal(scale=0.3)),
                    "stim": stim, "preceding_stim": float(rng_b.integers(0, 2)),
                    "moderator_native": float(session_mean + rng_b.normal(scale=0.01)),
                    "serialpos": float(rng_b.integers(1, 13)), "list_number": float(rng_b.integers(1, 6)),
                    "phase": rng_b.choice(["phase_a", "phase_b", "no_stimulation_list"]),
                })
    mod._add_bias_only_moderator(rows_b)
    bias_b = mod._fit_arm(rows_b, "outcome", "moderator_bias_only", True, "test|mixed_session|bias",
                           significance_null="between_subject")
    assert bias_b["status"] == "computed", bias_b
    assert bias_b["n_admissible_subjects_with_more_than_one_session"] == 14
    assert bias_b["between_subject_permutation_p"] < 0.05, (
        f"expected the between-subject null to detect the planted effect in a mixed-session "
        f"population, got {bias_b}")
    assert bias_b["significant"], "the arm's significance flag must follow the between-subject null"


def check_restriction_composition_and_contamination_check() -> None:
    """Materiality fires when retained and excluded populations differ on composition, and does not
    fire when they are identical -- distinct dict objects per row (not shared references) so the
    function's own id()-based excluded-rows computation exercises a real retained/excluded split."""
    material_all = (
        [{"serialpos": 1, "stim": 1.0, "phase": "phase_a"} for _ in range(10)]
        + [{"serialpos": 12, "stim": 0.0, "phase": "no_stimulation_list"} for _ in range(10)]
    )
    material_restricted = [r for r in material_all if r["serialpos"] == 12]
    material = mod._restriction_composition_and_contamination_check(
        material_all, material_restricted, mod.COMPOSITION_MATERIALITY_THRESHOLD)
    assert material["any_comparison_material"], material
    assert material["n_native_arm_words"] == 20 and material["n_retained_restricted_arm_words"] == 10
    assert material["n_excluded_preceding_word_stimulated_words"] == 10
    assert material["native_arm_contaminated_moderator_fraction"] == 0.5

    base = [{"serialpos": (i % 12) + 1, "stim": float(i % 2), "phase": "phase_a"} for i in range(24)]
    other = [{"serialpos": (i % 12) + 1, "stim": float(i % 2), "phase": "phase_a"} for i in range(24)]
    identical_all = base + other
    identical = mod._restriction_composition_and_contamination_check(
        identical_all, base, mod.COMPOSITION_MATERIALITY_THRESHOLD)
    assert not identical["any_comparison_material"], identical
    assert identical["native_arm_contaminated_moderator_fraction"] == 0.5


def check_restricted_vs_its_bias_only_sign_relationship() -> None:
    def _arm(status="computed", significant=True, mean=0.0):
        return {"status": status, "significant": significant, "subject_clustered_mean": mean}

    same_sign = mod._restricted_vs_its_bias_only_sign_relationship(_arm(mean=0.05), _arm(mean=0.02))
    assert "SAME direction" in same_sign, same_sign

    reversal = mod._restricted_vs_its_bias_only_sign_relationship(_arm(mean=0.05), _arm(mean=-0.02))
    assert "SIGN REVERSAL" in reversal, reversal

    restricted_not_sig = mod._restricted_vs_its_bias_only_sign_relationship(_arm(significant=False), _arm())
    assert restricted_not_sig.startswith("not applicable"), restricted_not_sig

    bias_not_sig = mod._restricted_vs_its_bias_only_sign_relationship(_arm(), _arm(significant=False))
    assert bias_not_sig.startswith("not applicable"), bias_not_sig


def check_nuisance_partialling_ladder() -> None:
    restricted = {"status": "computed", "subject_clustered_mean": -0.05, "within_subject_permutation_p": 0.01,
                  "significant": True, "n_words": 100, "bootstrap_ci95_lo": -0.08, "bootstrap_ci95_hi": -0.02}
    partialled = {"status": "computed", "subject_clustered_mean": -0.03, "within_subject_permutation_p": 0.02,
                  "significant": True, "n_words": 100, "bootstrap_ci95_lo": -0.06, "bootstrap_ci95_hi": -0.01}

    ladder = mod._nuisance_partialling_ladder(restricted, partialled, ("time_on_task",))
    delta = ladder["change_in_subject_clustered_mean_attributable_to_the_added_covariates"]
    assert abs(delta - 0.02) < 1e-12, ladder
    assert not any("directional-deviation" in c for c in ladder["four_covariate_partialled"]["covariates"])

    ladder_with_amplitude = mod._nuisance_partialling_ladder(
        restricted, partialled, ("time_on_task", "current_word_component_amplitude"))
    assert any("directional-deviation" in c for c in ladder_with_amplitude["four_covariate_partialled"]["covariates"])

    not_computable = {"status": "not_computable", "reason": "fewer_than_min_trials"}
    ladder_absent = mod._nuisance_partialling_ladder(restricted, not_computable, ("time_on_task",))
    assert ladder_absent["change_in_subject_clustered_mean_attributable_to_the_added_covariates"] is None
    assert ladder_absent["four_covariate_partialled"]["status"] == "not_computable"


def check_winners_curse_caveat_logic() -> None:
    """winners_curse_caveat depends on two quantities _fit_arm derives itself (the observed
    subject-clustered mean's position relative to its own 80%-power mdd, and the significance flag)
    -- both are monkeypatched here to exact, hand-picked values so the boolean is exercised
    deterministically rather than by hunting for a naturally-occurring case in synthetic data. The
    `rows` passed to _fit_arm are otherwise real (only the per-subject effects and the null draws
    are faked), so the pooled mixed-effects side check still runs against real row data."""
    rows = _make_rows(np.random.default_rng(99), "between_session")
    subjects = sorted({r["subject"] for r in rows})

    def _fixed_effects(mean: float, sd: float, n: int, seed: int):
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal(n)
        raw = (raw - raw.mean()) / raw.std(ddof=1)  # exact ddof=1 std of 1, exact mean of 0
        values = mean + sd * raw
        return {s: float(v) for s, v in zip(subjects[:n], values)}, {}

    orig_subject_effects, orig_within_null = mod._subject_partial_effects, mod._within_subject_null
    try:
        mod._within_subject_null = lambda *a, **k: np.zeros(50)  # forces p ~ 1/51 < 0.05 whenever mean != 0

        # mean 0.05 with sd 0.35 over n=25 -> mdd = z*sd/sqrt(n) ~ 0.196, mean sits below it
        mod._subject_partial_effects = lambda *a, **k: _fixed_effects(0.05, 0.35, 25, 1)
        below = mod._fit_arm(rows, "outcome", "moderator_native", True, "wc_test|below")
        assert below["status"] == "computed", below
        assert below["significant"], below
        assert below["effect_magnitude_below_its_own_minimum_detectable_difference"], below
        assert below["winners_curse_caveat"] == mod.WINNERS_CURSE_CAVEAT, below

        # mean 0.3 with sd 0.1 over n=25 -> mdd ~ 0.056, mean sits well ABOVE it despite same
        # forced significance, so the caveat must not fire
        mod._subject_partial_effects = lambda *a, **k: _fixed_effects(0.3, 0.1, 25, 2)
        above = mod._fit_arm(rows, "outcome", "moderator_native", True, "wc_test|above")
        assert above["status"] == "computed", above
        assert above["significant"], above
        assert not above["effect_magnitude_below_its_own_minimum_detectable_difference"], above
        assert above["winners_curse_caveat"] is None, above
    finally:
        mod._subject_partial_effects, mod._within_subject_null = orig_subject_effects, orig_within_null


def check_new_branch_classification() -> None:
    """Directly exercises _classify_branch's new rule with hand-built arm summaries: (1) fires when
    native is null but the restricted arm is significant and survives its own restricted-scope
    bias-only control (different sign, so it does not explain the restricted result) and its own
    nuisance-partialled re-fit (still significant, same sign); (2) does NOT fire, and falls back to
    the pre-existing powered-null rule, when the restricted arm's own bias-only control reproduces
    it in the SAME sign."""
    def _arm(significant, mean):
        return {"status": "computed", "significant": significant, "subject_clustered_mean": mean,
                "minimum_detectable_difference_80pct_power": {"status": "computed", "mdd": 0.05}}

    native = _arm(False, -0.01)
    bias_only = _arm(False, 0.01)
    restricted = _arm(True, -0.04)
    restricted_partialled = _arm(True, -0.04)

    restricted_bias_only_different_sign = _arm(True, +0.01)
    branch = mod._classify_branch(native, bias_only, restricted, restricted_bias_only_different_sign,
                                   restricted_partialled)
    assert branch == "restricted_arm_significant_with_its_own_controls_despite_native_null", branch

    restricted_bias_only_same_sign = _arm(True, -0.03)
    branch2 = mod._classify_branch(native, bias_only, restricted, restricted_bias_only_same_sign,
                                    restricted_partialled)
    assert branch2 == "no_moderation_above_the_reported_bound", branch2


def check_apply_list_labels_reproduces_observed_assignment() -> None:
    """_apply_list_labels_to_rows, given the OBSERVED label for each of a subject's own lists,
    must reproduce each row's own already-recorded current-word and preceding-word stimulation
    status exactly -- this is the same recomputation the design-based null runs on a PERMUTED
    label assignment, so if it cannot even reproduce the observed one, nothing it computes under
    a permutation can be trusted either."""
    subject = "sub-reproduce"
    list_labels = {1: "no_stimulation_list", 2: "phase_a", 3: "phase_b"}
    rows = []
    expected_stim, expected_preceding = [], []
    for list_number, label in list_labels.items():
        stim_positions = mod.STIMULATED_SERIALPOS_BY_LABEL[label]
        for sp in range(1, 13):
            rows.append({"session": "ses0", "list_number": float(list_number), "serialpos": float(sp)})
            expected_stim.append(1.0 if sp in stim_positions else 0.0)
            expected_preceding.append(1.0 if (sp - 1) in stim_positions else 0.0)

    label_of = {(r["session"], int(r["list_number"])): list_labels[int(r["list_number"])] for r in rows}
    recomputed = mod._apply_list_labels_to_rows(rows, label_of)
    assert list(recomputed["stim"]) == expected_stim, "recomputed current-word stim disagrees with the design"
    assert list(recomputed["preceding_stim"]) == expected_preceding, (
        "recomputed preceding-word stim disagrees with the design")
    assert list(recomputed["is_phase_b"]) == [1.0 if list_labels[int(r["list_number"])] == "phase_b" else 0.0
                                                for r in rows]
    assert list(recomputed["is_no_stim_list"]) == [
        1.0 if list_labels[int(r["list_number"])] == "no_stimulation_list" else 0.0 for r in rows]


def check_list_arrangement_diagnostics() -> None:
    """_list_arrangement_diagnostics's log10-space multinomial count against two hand-computable
    cases: a small 4-list subject (1 no_stim, 1 phase_a, 2 phase_b -> 4!/(1!1!2!) = 12 distinct
    arrangements, so the smallest attainable two-sided p if that subject alone were exhaustively
    enumerated is 1/12 =~ 0.0833, above 0.05 -- correctly flagged insufficient) against a subject
    with this corpus's own realistic list counts (12 lists, split 3/5/4 -- comfortably enough
    arrangements to clear 0.05, correctly NOT flagged)."""
    small = {"sub-small": [{"session": "s", "list_number": 1, "phase": "no_stimulation_list"},
                            {"session": "s", "list_number": 2, "phase": "phase_a"},
                            {"session": "s", "list_number": 3, "phase": "phase_b"},
                            {"session": "s", "list_number": 4, "phase": "phase_b"}]}
    diag = mod._list_arrangement_diagnostics(small)["sub-small"]
    assert diag["n_lists"] == 4
    assert diag["list_label_counts"] == {"no_stimulation_list": 1, "phase_a": 1, "phase_b": 2}
    assert abs(diag["smallest_attainable_two_sided_p_if_exhaustively_enumerated_for_this_subject_alone"]
               - 1.0 / 12.0) < 1e-9, diag
    assert diag["insufficient_arrangements_for_p_below_0_05"] is True, diag

    realistic_labels = (["no_stimulation_list"] * 3 + ["phase_a"] * 5 + ["phase_b"] * 4)
    realistic = {"sub-realistic": [{"session": "s", "list_number": i + 1, "phase": lbl}
                                    for i, lbl in enumerate(realistic_labels)]}
    diag_r = mod._list_arrangement_diagnostics(realistic)["sub-realistic"]
    assert diag_r["insufficient_arrangements_for_p_below_0_05"] is False, diag_r
    assert diag_r["smallest_attainable_two_sided_p_if_exhaustively_enumerated_for_this_subject_alone"] < 0.05


_STRUCTURE_TEST_EXPECTED_POSITIONS = {
    "no_stimulation_list": frozenset(),
    "phase_a": frozenset({1, 2, 5, 6, 9, 10}),
    "phase_b": frozenset({3, 4, 7, 8, 11, 12}),
}


def check_design_based_null_preserves_list_structure() -> None:
    """Runs the real _design_based_null end to end, capturing every per-draw, per-subject label
    assignment it actually constructs (by wrapping _apply_list_labels_to_rows, the function that
    consumes that assignment), and checks two structural invariants against each one: (1) the
    count of lists carrying each label is exactly what this subject's own data has on every draw,
    so the number of stimulated lists never changes; (2) every list's own recomputed
    current-word-stimulated serial positions exactly match its assigned label's own fixed
    template, checked against a hard-coded expectation independent of the module's own
    STIMULATED_SERIALPOS_BY_LABEL dict (so that dict being wrong would not go uncaught)."""
    rng = np.random.default_rng(123)
    subject = "sub-structtest"
    list_labels = ["no_stimulation_list", "no_stimulation_list", "phase_a", "phase_a", "phase_b", "phase_b"]
    inventory = {subject: [{"session": "sesX", "list_number": i + 1, "phase": lbl}
                            for i, lbl in enumerate(list_labels)]}
    observed_counts = Counter(list_labels)

    rows = []
    for list_number, label in enumerate(list_labels, start=1):
        stim_positions = _STRUCTURE_TEST_EXPECTED_POSITIONS[label]
        for sp in range(1, 13):
            rows.append({
                "session": "sesX", "subject": subject, "list_number": float(list_number),
                "serialpos": float(sp), "word_index": (list_number - 1) * 12 + sp,
                "outcome": float(rng.normal()), "moderator_native": float(rng.normal()),
                "displacement": float(rng.normal()), "phase": label,
                "stim": 1.0 if sp in stim_positions else 0.0,
                "preceding_stim": 1.0 if (sp - 1) in stim_positions else 0.0,
            })

    captured = []
    real_apply = mod._apply_list_labels_to_rows

    def _spy(srows, label_of):
        captured.append(dict(label_of))
        return real_apply(srows, label_of)

    mod._apply_list_labels_to_rows = _spy
    try:
        mod._design_based_null(rows, "outcome", "moderator_native", True, [subject], inventory,
                                n_draws=25, seed_tag="structure_test")
    finally:
        mod._apply_list_labels_to_rows = real_apply

    assert len(captured) == 25, f"expected one label assignment captured per draw, got {len(captured)}"
    for label_of in captured:
        assigned_counts = Counter(label_of.values())
        assert assigned_counts == observed_counts, (
            f"a design-based draw changed the number of lists carrying each label: "
            f"{assigned_counts} != {observed_counts}")
        for (session, list_number), label in label_of.items():
            expected_positions = _STRUCTURE_TEST_EXPECTED_POSITIONS[label]
            list_rows = [r for r in rows if r["list_number"] == list_number]
            recomputed = mod._apply_list_labels_to_rows(list_rows, {(session, list_number): label})
            actual_positions = {int(r["serialpos"]) for r, s in zip(list_rows, recomputed["stim"]) if s == 1.0}
            assert actual_positions == expected_positions, (
                f"list {list_number} assigned label {label} has a broken within-list pattern: "
                f"{actual_positions} != {expected_positions}")


def _mutate_stimulated_serialpos_by_label_to_break() -> None:
    """Deliberately corrupts the phase_a template to a wrong, partial set of positions (as if a
    future edit accidentally mis-typed it) -- used only to demonstrate
    check_design_based_null_preserves_list_structure can fail; reverted immediately after."""
    mod.STIMULATED_SERIALPOS_BY_LABEL["phase_a"] = frozenset({1, 2, 3})


def _restore_stimulated_serialpos_by_label() -> None:
    mod.STIMULATED_SERIALPOS_BY_LABEL["phase_a"] = frozenset({1, 2, 5, 6, 9, 10})


def _mutate_between_subject_null_to_break() -> None:
    """Deliberately breaks the between-subject null: returns all-NaN draws (as if the shuffle were
    never run), so no arm can ever be called significant by it. Used only to demonstrate
    check_between_subject_null_for_single_session_subjects can fail; reverted immediately after."""
    def _broken(*_args, **_kwargs):
        return np.full(mod.N_WITHIN_SUBJECT_NULL_DRAWS, np.nan)
    mod._between_subject_null = _broken


def _mutate_bias_only_to_break() -> None:
    """Deliberately breaks the bias-only construction: instead of each row's own session's mean,
    every row gets the same GLOBAL mean across all rows -- collapsing all between-session
    structure this control depends on. Used only to demonstrate check_bias_only_construction can
    fail; reverted by _restore_bias_only immediately after."""
    def _broken(rows: list[dict]) -> None:
        global_mean = float(np.mean([r["moderator_native"] for r in rows]))
        for r in rows:
            r["moderator_bias_only"] = global_mean
    mod._add_bias_only_moderator = _broken


def _restore_bias_only() -> None:
    import importlib
    importlib.reload(mod)
    mod.N_WITHIN_SUBJECT_NULL_DRAWS = 200
    mod.N_BOOT = 500
    mod.MIN_WORDS_PER_SUBJECT_FOR_EFFECT = 20


# pytest discovers test_* functions by default; the check_* functions above are the real bodies
# (also runnable standalone, see module docstring) and these just give pytest an entry point.
def test_delivered_artifact() -> None:
    check_delivered_artifact()


def test_bias_only_construction() -> None:
    check_bias_only_construction()


def test_restricted_arm_construction() -> None:
    check_restricted_arm_construction()


def test_between_subject_null_for_single_session_subjects() -> None:
    check_between_subject_null_for_single_session_subjects()


def test_new_branch_classification() -> None:
    check_new_branch_classification()


def test_restriction_composition_and_contamination_check() -> None:
    check_restriction_composition_and_contamination_check()


def test_restricted_vs_its_bias_only_sign_relationship() -> None:
    check_restricted_vs_its_bias_only_sign_relationship()


def test_nuisance_partialling_ladder() -> None:
    check_nuisance_partialling_ladder()


def test_winners_curse_caveat_logic() -> None:
    check_winners_curse_caveat_logic()


def test_apply_list_labels_reproduces_observed_assignment() -> None:
    check_apply_list_labels_reproduces_observed_assignment()


def test_list_arrangement_diagnostics() -> None:
    check_list_arrangement_diagnostics()


def test_design_based_null_preserves_list_structure() -> None:
    check_design_based_null_preserves_list_structure()


if __name__ == "__main__":
    check_delivered_artifact()
    print("PASS: delivered artifact branches and zero-drop ledgers reconcile")
    check_bias_only_construction()
    print("PASS (unbroken): bias-only reproduces the planted between-session association and "
          "nulls out the planted within-session one")

    _mutate_bias_only_to_break()
    try:
        check_bias_only_construction()
    except AssertionError as e:
        print(f"PASS (deliberate break correctly caught): {e}")
    else:
        raise SystemExit("FAIL: the deliberately broken bias-only construction was NOT caught by the check")
    finally:
        _restore_bias_only()

    check_bias_only_construction()
    print("PASS (restored): bias-only construction is back to its real behaviour and the check passes again")
    check_restricted_arm_construction()
    print("PASS: restricted arm excludes preceding-word stimulation and its redundant covariate")

    check_between_subject_null_for_single_session_subjects()
    print("PASS (unbroken): between-subject null detects a between-subject effect that the "
          "within-subject null cannot see for single-session subjects")

    _mutate_between_subject_null_to_break()
    try:
        check_between_subject_null_for_single_session_subjects()
    except AssertionError as e:
        print(f"PASS (deliberate break correctly caught): {e}")
    else:
        raise SystemExit("FAIL: the deliberately broken between-subject null was NOT caught by the check")
    finally:
        _restore_bias_only()

    check_between_subject_null_for_single_session_subjects()
    print("PASS (restored): between-subject null is back to its real behaviour and the check passes again")

    check_new_branch_classification()
    print("PASS: the new forward-only branch fires only when the restricted arm's own bias-only "
          "control does not reproduce it in the same sign")

    check_restriction_composition_and_contamination_check()
    print("PASS: composition/contamination check flags a material split and clears an identical one")
    check_restricted_vs_its_bias_only_sign_relationship()
    print("PASS: restricted-vs-bias-only sign relationship text covers same-sign, reversal and both "
          "not-applicable cases")
    check_nuisance_partialling_ladder()
    print("PASS: nuisance partialling ladder reports the right delta and covariate list per outcome")
    check_winners_curse_caveat_logic()
    print("PASS: winners_curse_caveat fires only when the observed effect sits below its own mdd "
          "AND the arm is significant")

    check_apply_list_labels_reproduces_observed_assignment()
    print("PASS: recomputing stim/preceding_stim/phase indicators from a label assignment "
          "reproduces the observed per-word stimulation status exactly")
    check_list_arrangement_diagnostics()
    print("PASS: list-level arrangement counts match hand-computed multinomial coefficients")

    check_design_based_null_preserves_list_structure()
    print("PASS (unbroken): the design-based null's own label permutations preserve each "
          "subject's list-label counts and each list's own within-list stimulated-position pattern")

    _mutate_stimulated_serialpos_by_label_to_break()
    try:
        check_design_based_null_preserves_list_structure()
    except AssertionError as e:
        print(f"PASS (deliberate break correctly caught): {e}")
    else:
        raise SystemExit("FAIL: the deliberately corrupted phase_a template was NOT caught by the check")
    finally:
        _restore_stimulated_serialpos_by_label()

    check_design_based_null_preserves_list_structure()
    print("PASS (restored): STIMULATED_SERIALPOS_BY_LABEL is back to its real values and the "
          "check passes again")
