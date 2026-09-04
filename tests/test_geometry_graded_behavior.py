"""Self-check for scripts/run_geometry_graded_behavior.py's LME wrapper.

Not a general statistics.py test (that's tests/test_statistics.py) -- this
checks the specific pattern this script relies on: recovering a planted
drift -> RT slope from multi-subject synthetic data via
statistics.linear_mixed_effects_test plus the bootstrap CI helper.
"""
import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_geometry_graded_behavior as graded
from run_geometry_graded_behavior import (
    _lme_report, bootstrap_ci_beta, _bias_only_predictor, _within_subject_predictor,
    _require_complete_corpus_admission, _within_subject_shuffle, _subject_level_table,
    _ols_slope, _between_subject_shuffle_test, _within_subject_identity_note,
    _nuisance_partialled_verdict, _within_subject_correlation,
    REFERENCE_R_UNITS,
)
from statistics import permutation_pvalue


def _synthetic_drift_rt(true_beta: float, n_subjects: int = 6, n_trials_per_subj: int = 80,
                        rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng(0)
    drift, rt, subj = [], [], []
    for s in range(n_subjects):
        subj_intercept = rng.normal(0, 2.0)  # random-effect nuisance the LME must partial out
        d = rng.normal(1.0, 0.5, n_trials_per_subj)
        y = subj_intercept + true_beta * d + rng.normal(0, 0.3, n_trials_per_subj)
        drift.append(d); rt.append(y); subj.append([f"s{s}"] * n_trials_per_subj)
    return np.concatenate(drift), np.concatenate(rt), np.concatenate(subj)


def test_recovers_positive_slope():
    drift, rt, subj = _synthetic_drift_rt(true_beta=1.5, rng=np.random.default_rng(1))
    res = _lme_report(drift, rt, subj, "test_pos")
    assert "beta" in res
    assert res["beta"] == pytest.approx(1.5, abs=0.3)
    assert res["p_value"] < 0.01
    assert res["ci_lo"] < res["beta"] < res["ci_hi"]


def test_recovers_negative_slope():
    drift, rt, subj = _synthetic_drift_rt(true_beta=-1.2, rng=np.random.default_rng(2))
    res = _lme_report(drift, rt, subj, "test_neg")
    assert res["beta"] == pytest.approx(-1.2, abs=0.3)
    assert res["p_value"] < 0.01


def test_null_slope_not_significant():
    drift, rt, subj = _synthetic_drift_rt(true_beta=0.0, rng=np.random.default_rng(3))
    res = _lme_report(drift, rt, subj, "test_null")
    assert abs(res["beta"]) < 0.3
    assert res["p_value"] > 0.05


def test_underpowered_flag():
    drift, rt, subj = _synthetic_drift_rt(true_beta=1.0, n_subjects=1, n_trials_per_subj=5,
                                          rng=np.random.default_rng(4))
    res = _lme_report(drift, rt, subj, "test_underpowered")
    assert res["underpowered"] is True


def _synthetic_between_subject_only(n_subjects: int = 30, n_trials_per_subj: int = 40,
                                    rng: np.random.Generator | None = None):
    """Plant a drift/RT association that is PURELY between-subject: each subject's mean drift
    predicts that subject's mean RT, but within a subject drift and RT are independent noise. The
    bias-only transform should recover this; the within-subject transform should null it out.
    (n_subjects=30 rather than a handful -- fitting a fixed slope on a per-subject-constant
    predictor alongside a subject random intercept is a low-effective-n design, same as this
    project's real bias-only arms; too few subjects leaves the MixedLM fit on the boundary of its
    parameter space and the test flaky rather than a clean check of the construction.)"""
    if rng is None:
        rng = np.random.default_rng(0)
    drift, rt, subj = [], [], []
    for s in range(n_subjects):
        subj_mean_drift = rng.normal(1.0, 1.0)
        subj_mean_rt = 2.0 * subj_mean_drift  # the only real signal: between-subject slope 2.0
        d = subj_mean_drift + rng.normal(0, 0.05, n_trials_per_subj)  # tight around its own mean
        y = subj_mean_rt + rng.normal(0, 0.3, n_trials_per_subj)  # independent of d within-subject
        drift.append(d); rt.append(y); subj.append([f"s{s}"] * n_trials_per_subj)
    return np.concatenate(drift), np.concatenate(rt), np.concatenate(subj)


def test_bias_only_recovers_between_subject_association():
    drift, rt, subj = _synthetic_between_subject_only(rng=np.random.default_rng(5))
    bias_res = _lme_report(_bias_only_predictor(drift, subj), rt, subj, "test_bias_recovers")
    assert bias_res["p_value"] < 0.05
    assert bias_res["beta"] > 0  # planted slope was positive (2.0)


def test_within_subject_nulls_out_purely_between_subject_association():
    drift, rt, subj = _synthetic_between_subject_only(rng=np.random.default_rng(6))
    within_res = _lme_report(_within_subject_predictor(drift, subj), rt, subj, "test_within_nulls")
    assert within_res["p_value"] > 0.05


def test_incomplete_corpus_admission_refuses_artifact_replacement():
    with pytest.raises(RuntimeError, match="missing scored corpus"):
        _require_complete_corpus_admission({"boran_ieeg": {"beta": 0.01}})


# ── Defect one: bias-only arm needs a between-subject, not within-subject, null ────────────────

def test_within_subject_shuffle_is_exactly_invariant_for_subject_constant_predictor():
    """The numeric proof behind BIAS_ONLY_NULL_REASON: a within-subject shuffle can never change
    a subject-constant predictor, so a permutation test built on it returns p ~= 1 by
    construction, regardless of the data -- not an assertion in prose, a literal permutation
    test run below."""
    rng = np.random.default_rng(7)
    drift, rt, subj = _synthetic_between_subject_only(rng=rng)
    predictor = _bias_only_predictor(drift, subj)  # constant within every subject, by construction
    observed = _ols_slope(predictor, rt)

    n_perm = 200
    exceed = np.empty(n_perm, dtype=bool)
    for i in range(n_perm):
        shuffled = _within_subject_shuffle(predictor, subj, rng)
        assert np.array_equal(shuffled, predictor), (
            "a within-subject shuffle changed a subject-constant predictor -- it should be a "
            "no-op by construction"
        )
        exceed[i] = abs(_ols_slope(shuffled, rt)) >= abs(observed)
    p_within_subject_shuffle = permutation_pvalue(exceed)
    assert p_within_subject_shuffle == 1.0, (
        f"expected exactly p=1.0 under a within-subject shuffle of a subject-constant "
        f"predictor, got {p_within_subject_shuffle}"
    )


def test_between_subject_shuffle_is_not_invariant_and_detects_the_planted_effect():
    """The complement to the invariance proof above: the between-subject shuffle DOES change a
    subject-constant predictor's pairing with the outcome, and correctly detects a planted
    purely-between-subject association that the within-subject shuffle cannot."""
    rng = np.random.default_rng(8)
    drift, rt, subj = _synthetic_between_subject_only(n_subjects=40, rng=rng)
    predictor = _bias_only_predictor(drift, subj)
    x_i, y_i = _subject_level_table(predictor, rt, subj)

    shuffled_once = rng.permutation(x_i)
    assert not np.array_equal(shuffled_once, x_i), (
        "a between-subject shuffle should generally change the predictor-to-subject pairing"
    )

    result = _between_subject_shuffle_test(x_i, y_i, n_perm=2000, rng=np.random.default_rng(9))
    assert result["status"] == "computed"
    assert result["p_value_between_subject"] < 0.05, (
        "the between-subject shuffle should detect the planted between-subject association "
        f"(got p={result['p_value_between_subject']})"
    )
    assert result["beta_between_subject"] > 0  # planted slope was positive


# ── Defect two: within-subject beta is arithmetically identical to native beta ─────────────────

def test_within_subject_beta_identical_to_native_and_disclosed():
    drift, rt, subj = _synthetic_drift_rt(true_beta=0.8, rng=np.random.default_rng(10))
    native = _lme_report(drift, rt, subj, "test_identity_native")
    within = _lme_report(_within_subject_predictor(drift, subj), rt, subj, "test_identity_within")
    note = _within_subject_identity_note(within, native)
    assert note["arithmetically_identical_to_native"] is True
    assert within["beta"] == pytest.approx(native["beta"], rel=1e-3)


# ── Defect (leading arm): nuisance-partialled verdict is powered-null-or-inconclusive ──────────

def test_nuisance_partialled_verdict_powered_null_below_reference():
    arm = {"beta": 0.001, "p_value": 0.90, "mdc_80_r": REFERENCE_R_UNITS - 0.05,
          "within_subject_r": 0.001}
    verdict, detail = _nuisance_partialled_verdict(arm)
    assert verdict == "powered_null"
    assert "reference" in detail


def test_nuisance_partialled_verdict_inconclusive_above_reference():
    arm = {"beta": 0.001, "p_value": 0.90, "mdc_80_r": REFERENCE_R_UNITS + 0.05,
          "within_subject_r": 0.001}
    verdict, _ = _nuisance_partialled_verdict(arm)
    assert verdict == "inconclusive"


def test_nuisance_partialled_verdict_pooled_powered_null():
    """Pooled verdict fires correctly when mdc is below reference (powered null).
    Mimics a meta-analytically pooled arm with n_trials large enough to detect."""
    arm = {
        "beta": 0.02, "p_value": 0.75,
        "mdc_80_r": REFERENCE_R_UNITS - 0.02,
        "n_trials": 10000,
        "forest": {"pooled": 0.02},
    }
    verdict, detail = _nuisance_partialled_verdict(arm)
    assert verdict == "powered_null"
    assert "reference" in detail
    assert "minimum detectable" in detail


def test_nuisance_partialled_verdict_pooled_inconclusive():
    """Pooled verdict fires correctly when mdc is at or above reference (inconclusive).
    Mimics a meta-analytically pooled arm with insufficient n_trials."""
    arm = {
        "beta": 0.03, "p_value": 0.60,
        "mdc_80_r": REFERENCE_R_UNITS + 0.01,
        "n_trials": 2000,
        "forest": {"pooled": 0.03},
    }
    verdict, detail = _nuisance_partialled_verdict(arm)
    assert verdict == "inconclusive"
    assert "underpowered" in detail
    assert "reference" in detail


def test_subject_adjusted_r_removes_pure_between_subject_association():
    """The reported r must estimate the within-subject link, not subject offsets."""
    drift, rt, subj = _synthetic_between_subject_only(rng=np.random.default_rng(19))
    raw_r = np.corrcoef(drift, rt)[0, 1]
    within_r = _within_subject_correlation(drift, rt, subj)
    assert raw_r > 0.8
    assert abs(within_r) < 0.1


# ── Defect: pooled bias-only arm silently dropped its own detection bound ──────────────────────

def test_pool_arm_bias_only_populates_mdc_80_r_on_between_subject_scale():
    """The bias-only arm has no within-subject variation to bootstrap (its predictor is
    subject-constant), so the generic within_subject_r/within_subject_r_se pooling used by the
    other arms silently found nothing to pool and left mdc_80_r absent -- not null, missing --
    from the pooled bias-only arm. It must instead pool on its own between_subject_r scale using
    the analytic Fisher-z SE (1/sqrt(n_subjects-3)) each corpus's own mdc_80_r already used."""
    scored = {
        "dandi000469": {"bias_only": {
            "beta": 0.1, "se": 0.2, "p_value": 0.5, "n_trials": 100, "n_subjects": 10,
            "between_subject_r": 0.3,
        }},
        "dandi001187": {"bias_only": {
            "beta": 0.15, "se": 0.25, "p_value": 0.5, "n_trials": 120, "n_subjects": 15,
            "between_subject_r": 0.2,
        }},
    }
    pooled = graded._pool_arm(scored, "bias_only", ["dandi000469", "dandi001187"])
    assert "mdc_80_r" in pooled, "mdc_80_r must never be silently missing from a pooled arm"
    assert np.isfinite(pooled["mdc_80_r"])
    assert "between_subject_r" in pooled
    assert "mdc_80_r_correction_note" in pooled


def test_main_uses_canonical_independent_pool_and_current_helper_contract(
        monkeypatch, tmp_path):
    """Cheap integration test for the main dispatcher.

    This catches argument-contract drift between ``main`` and the clustered-
    power/control helpers without running any bootstrap fits, and protects the
    rule that the linked 000673 release is analysed but not double-counted in
    the primary meta-analysis.
    """
    corpus_names = ("dandi000469", "dandi001187", "dandi000673", "boran_ieeg")
    helper_calls = {"controls": 0}

    def corpus_result(beta):
        result = {
            "beta": beta, "se": 0.1, "p_value": 0.4,
            "r_squared": 0.01, "n_trials": 8, "n_subjects": 2,
            "ci_lo": beta - 0.2, "ci_hi": beta + 0.2,
            "within_subject_r": 0.01, "within_subject_r_se": 0.1,
            "mdc_80_r": 0.28,
        }
        trials = {
            "drift": np.linspace(0, 1, 8),
            "rt": np.repeat([1.0, 2.0], 4),
            "subj": np.repeat(["s1", "s2"], 4),
            "load": np.ones(8), "outcome": np.ones(8),
            "amp": np.ones(8), "trial_idx": np.tile(np.arange(4), 2),
            "sessions_seen": 2, "sessions_analysed": 2, "sessions_refused": [],
        }
        return result, trials

    for index, (name, function_name) in enumerate(zip(corpus_names, (
            "run_dandi000469", "run_dandi001187", "run_dandi000673", "run_boran"))):
        monkeypatch.setattr(
            graded, function_name,
            lambda beta=0.01 * (index + 1): corpus_result(beta),
        )

    def fake_controls(trials, seed_prefix, native):
        helper_calls["controls"] += 1
        base = {
            "beta": native["beta"], "se": 0.1, "p_value": 0.4,
            "n_trials": 8, "n_subjects": 2, "r_squared": 0.01,
            "within_subject_r": 0.01, "within_subject_r_se": 0.1,
            "mdc_80_r": 0.5,
        }
        return {
            "bias_only": dict(base), "within_subject": dict(base),
            "nuisance_partialled": dict(base, verdict="inconclusive"),
            "nuisance_partialled_zero_drop": {
                "trials_seen": 8, "trials_analysed": 8,
                "trials_refused": 0, "zero_drop_holds": True,
            },
        }

    monkeypatch.setattr(graded, "_run_control_arms", fake_controls)
    monkeypatch.setattr(graded, "RESULTS", tmp_path)
    graded.main()

    import json
    artifact = json.loads((tmp_path / "geometry_graded_behavior.json").read_text())
    assert helper_calls == {"controls": 4}
    assert artifact["_meta"]["datasets_pooled"] == [
        "dandi000469", "dandi001187", "boran_ieeg"]
    assert artifact["_meta"]["linked_sensitivity_views_excluded"] == ["dandi000673"]
    assert artifact["_all_release_views_sensitivity"]["datasets_combined"] == list(corpus_names)


if __name__ == "__main__":
    test_recovers_positive_slope()
    test_recovers_negative_slope()
    test_null_slope_not_significant()
    test_underpowered_flag()
    test_bias_only_recovers_between_subject_association()
    test_within_subject_nulls_out_purely_between_subject_association()
    test_incomplete_corpus_admission_refuses_artifact_replacement()
    test_within_subject_shuffle_is_exactly_invariant_for_subject_constant_predictor()
    test_between_subject_shuffle_is_not_invariant_and_detects_the_planted_effect()
    test_within_subject_beta_identical_to_native_and_disclosed()
    test_nuisance_partialled_verdict_powered_null_below_reference()
    test_nuisance_partialled_verdict_inconclusive_above_reference()
    test_nuisance_partialled_verdict_pooled_powered_null()
    test_nuisance_partialled_verdict_pooled_inconclusive()
    test_pool_arm_bias_only_populates_mdc_80_r_on_between_subject_scale()
    print("All geometry_graded_behavior self-checks passed.")
