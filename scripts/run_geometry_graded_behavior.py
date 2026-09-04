#!/usr/bin/env python3
"""Link maintenance-window geometry to graded response time as an observational
readout, not a controller objective. Binary correct/error is ceiling-limited
(behavior_ctg.json: significant in only 1/5 cohorts; behavior_geometry_link.json:
entirely null) -- RT is graded and available at full trial counts in every
cohort, giving real variance to test against.

Datasets INSPECTED (not assumed) for a graded behavioral field aligned to the
same trials table the maintenance geometry (Z, drift) was computed from:
  - dandi000469 (Rutishauser, intervals/trials): no bare RT field; RT =
    timestamps_Response - timestamps_Probe (both present).
  - dandi001187 (Rutishauser, intervals/WM_trials): response_time field
    present and already trial-relative (verified against
    timestamps_Response - timestamps_Probe on real data -- matches to <0.03s).
    A genuine subject-reported "confidence" field EXISTS in this dataset, but
    only in the SEPARATE intervals/LTM_trials table (a different,
    long-term-memory recognition task) -- NOT aligned to WM_trials / the
    maintenance geometry this script uses, so it is not usable here and is not
    fabricated as if it were.
  - dandi000673 (Rutishauser, intervals/trials): no bare RT field, no
    confidence field; RT = timestamps_Response - timestamps_Probe.
  - boran iEEG (000574, intervals/trials): response_time field present,
    already trial-relative once trial start_time is subtracted (matches
    run_boran_pipeline.py's own convention exactly). No confidence field.
No dataset here has a genuine subject-reported confidence rating aligned to
its maintenance-geometry trials -- confidence is therefore reported as
excluded_no_graded_behavior (reason: field absent or task-misaligned), not
computed via a decoder-confidence proxy, which would share its source latent
Z with the geometry predictor (drift) and be circular rather than
independent. RT is the only graded variable used.

Geometry predictor: per-trial `drift` (Euclidean distance from the
condition/set-size centroid trajectory during the maintenance window,
geometry.geometric_drift / spike_pipeline.correct_error_drift) -- already
computed and saved in every *_geometry_*.npz this project produces (it is the
same scalar the correct-vs-error drift LME in run_boran_pipeline.py and the
three run_0*_pipeline.py scripts already use), reused here rather than
inventing a new one.

Model: per-trial LME (statistics.linear_mixed_effects_test) of RT ~ drift,
subject as random intercept; p-value from the fitted model's own Wald test,
since it has an analytic sampling distribution once fit (no separate
permutation null for the native cell). 95% CI via subject-level percentile
bootstrap. Datasets meta-combined with statistics.forest_meta
(inverse-variance) and statistics.stouffer_combine (p-values). The forest is the
reported pooled answer -- it asks whether one common effect size holds across
cohorts; the p-combination is a secondary evidence summary of whether the cohorts
jointly reject a no-effect null, a different question that can legitimately
disagree with the forest (see pooled_statistic_reconciliation in the output). <2
subjects or <30 trials with the variable -> underpowered, no beta reported for that
dataset.

The bias-only control arm (subject-constant predictor) is the one exception:
a mixed model's Wald test is not trustworthy for a fixed effect that never
varies within any one of its own random-intercept groups (see
BIAS_ONLY_NULL_REASON and _between_subject_shuffle_test), so that arm's
p-value instead comes from an explicit between-subject permutation test on a
direct subject-level regression.

Outputs: results/geometry_graded_behavior.json
Self-check: tests/test_geometry_graded_behavior.py (LME recovers a planted
synthetic slope).

Run:
    conda run -n wm_dynamics python scripts/run_geometry_graded_behavior.py
"""
from __future__ import annotations

import sys
import json
import glob
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root, dataset_path, executable, project_path

import h5py
from scipy.stats import norm
from statistics import (linear_mixed_effects_test, forest_meta, stouffer_combine, stable_seed,
                        permutation_pvalue)

RESULTS = ROOT / "results"
DATA_ROOT = data_root()

MIN_SUBJECTS = 2
MIN_TRIALS = 30
N_PERM = 5000
N_BOOT = 2000
REFERENCE_R_UNITS = 0.14  # this project's standing behavioural-effect reference, fixed before any control arm was fit
LEADING_ARM = "nuisance_partialled"  # the arm this analysis leads with -- see its verdict/verdict_detail fields

CONTROL_BRANCHES = (
    "bias_only_reproduces_native",
    "within_subject_survives_all_controls",
    "within_subject_null_and_powered",
    "within_subject_null_underpowered",
    "mixed_across_corpora",
)
EXPECTED_CORPORA = ("dandi000469", "dandi001187", "dandi000673", "boran_ieeg")
# DANDI 000673 and 001187 are linked release views with extensive verified
# patient/session overlap.  The project registry's canonical-view rule therefore
# admits 001187 to primary inference and retains 000673 as a sensitivity view;
# counting both as independent cohorts would duplicate participants in the
# meta-analysis.
PRIMARY_CORPORA = ("dandi000469", "dandi001187", "boran_ieeg")
LINKED_SENSITIVITY_CORPORA = ("dandi000673",)
LINKED_VIEW_NOTE = (
    "DANDI 000673 is a linked release view of DANDI 001187 (31 verified shared "
    "patients across 37 shared recording sessions). It is analysed and reported "
    "as a release-specific sensitivity view but excluded from the independent-"
    "cohort primary meta-analysis."
)


def minimum_detectable_correlation(n: float, alpha: float = 0.05, power: float = 0.80) -> dict:
    """Smallest true Pearson r an independent-samples correlation at this n could detect at the
    given power, via the Fisher z transform's normal approximation -- the correlation analogue of
    statistics.minimum_detectable_paired_difference, which is built for paired mean differences.
    This helper is used only for the bias-only arm's one-row-per-subject table. Trial-level arms
    instead derive detection bounds from a whole-subject cluster bootstrap."""
    if n < 4:
        return {"status": "not_computable", "n": n}
    z_factor = float(norm.ppf(1 - alpha / 2) + norm.ppf(power))
    z_effect = z_factor / np.sqrt(n - 3)
    return {"status": "computed", "n": n, "alpha": alpha, "power": power,
            "z_factor": z_factor, "mdd_r": float(np.tanh(z_effect))}


# Two-sided alpha=0.05, power=0.80 detectability z-factor, the same quantity every power
# calculation in this project uses (see statistics.minimum_detectable_paired_difference).
Z_FACTOR_80_POWER = float(norm.ppf(1 - 0.05 / 2) + norm.ppf(0.80))


WITHIN_R_DEFINITION = (
    "Pearson correlation after subtracting each subject's own mean from both drift and response "
    "time on the exact analysed rows. This is a subject-adjusted, within-subject effect size; "
    "its uncertainty comes from resampling whole subjects, never individual trials."
)

MDC_CORRECTION_NOTE = (
    "mdc_80_r is Z_(.975)+Z_(.80) times the subject-cluster bootstrap standard error of the "
    "within-subject correlation. It therefore targets the same subject-adjusted effect and rows "
    "as within_subject_r. It is not a Fisher calculation on the raw trial count and it does not "
    "treat repeated trials as independent."
)


def _within_subject_correlation(x: np.ndarray, y: np.ndarray, subj: np.ndarray) -> float:
    """Trial-weighted partial correlation after removing subject fixed intercepts.

    The weighting matches the trial-level mixed model: subjects with more admissible trials
    contribute more observations to the descriptive effect, while the bootstrap below retains
    subjects as the independent resampling unit.
    """
    x_centered = x - np.array([np.mean(x[subj == s]) for s in subj])
    y_centered = y - np.array([np.mean(y[subj == s]) for s in subj])
    if np.std(x_centered) < 1e-12 or np.std(y_centered) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_centered, y_centered)[0, 1])


def bootstrap_ci_beta(drift: np.ndarray, y: np.ndarray, subj: np.ndarray,
                      n_boot: int = N_BOOT, rng: np.random.Generator | None = None) -> dict:
    """Percentile-bootstrap 95% CI on the LME beta, resampling SUBJECTS (not
    trials) with replacement to match the subject-random-effect structure."""
    if rng is None:
        rng = np.random.default_rng(0)
    subs = np.unique(subj)
    betas = np.full(n_boot, np.nan)
    within_rs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        samp = rng.choice(subs, size=len(subs), replace=True)
        idx_parts, subj_parts = [], []
        for i, s in enumerate(samp):
            rows = np.where(subj == s)[0]
            idx_parts.append(rows)
            subj_parts.append(np.full(len(rows), i))  # re-key duplicate draws uniquely
        idx = np.concatenate(idx_parts)
        resamp_subj = np.concatenate(subj_parts)
        # metric=y (RT), condition=drift -- must match _lme_report's call order
        # (linear_mixed_effects_test(rt, drift, subj, ...)) or the bootstrap
        # estimates a different regression than the point estimate.
        res = linear_mixed_effects_test(y[idx], drift[idx], resamp_subj, n_perm=1,
                                        rng=np.random.default_rng(0))
        betas[b] = res["beta"]
        within_rs[b] = _within_subject_correlation(drift[idx], y[idx], resamp_subj)
    finite_betas = betas[np.isfinite(betas)]
    if len(finite_betas) < max(100, int(0.8 * n_boot)):
        return {
            "ci_lo": float("nan"), "ci_hi": float("nan"), "se": float("nan"),
            "status": "not_estimable",
            "reason": (
                f"only {len(finite_betas)}/{n_boot} subject-bootstrap MixedLM fits converged"
            ),
        }
    finite_rs = within_rs[np.isfinite(within_rs)]
    if len(finite_rs) < max(100, int(0.8 * n_boot)):
        return {
            "ci_lo": float("nan"), "ci_hi": float("nan"), "se": float("nan"),
            "r_ci_lo": float("nan"), "r_ci_hi": float("nan"), "r_se": float("nan"),
            "status": "not_estimable",
            "reason": f"only {len(finite_rs)}/{n_boot} subject-bootstrap r draws were finite",
        }
    lo, hi = np.percentile(finite_betas, [2.5, 97.5])
    r_lo, r_hi = np.percentile(finite_rs, [2.5, 97.5])
    se = (hi - lo) / (2 * 1.96)
    r_se = (r_hi - r_lo) / (2 * 1.96)
    return {
        "ci_lo": float(lo), "ci_hi": float(hi), "se": float(se),
        "r_ci_lo": float(r_lo), "r_ci_hi": float(r_hi), "r_se": float(r_se),
        "status": "ok", "n_boot_converged": int(len(finite_betas)),
        "n_boot_r_finite": int(len(finite_rs)),
    }


def _lme_report(drift: np.ndarray, rt: np.ndarray, subj: np.ndarray, seed_name: str) -> dict:
    finite = np.isfinite(drift) & np.isfinite(rt)
    drift, rt, subj = drift[finite], rt[finite], subj[finite]
    n_subjects = len(np.unique(subj))
    if n_subjects < MIN_SUBJECTS or len(drift) < MIN_TRIALS:
        return {"underpowered": True, "n_trials": int(len(drift)), "n_subjects": int(n_subjects)}
    res = linear_mixed_effects_test(rt, drift, subj, n_perm=N_PERM,
                                    rng=np.random.default_rng(stable_seed(seed_name)))
    ci = bootstrap_ci_beta(drift, rt, subj, rng=np.random.default_rng(stable_seed(seed_name + "_boot")))
    within_r = _within_subject_correlation(drift, rt, subj)
    r_se = ci.get("r_se", float("nan"))
    return {
        "beta": res["beta"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "se": ci["se"],
        "p_value": res["p_value"], "r_squared": res["r_squared"],
        "within_subject_r": within_r,
        "within_subject_r_definition": WITHIN_R_DEFINITION,
        "within_subject_r_ci_lo": ci.get("r_ci_lo", float("nan")),
        "within_subject_r_ci_hi": ci.get("r_ci_hi", float("nan")),
        "within_subject_r_se": r_se,
        "mdc_80_r": float(min(1.0, Z_FACTOR_80_POWER * r_se)) if np.isfinite(r_se) else float("nan"),
        "mdc_80_r_correction_note": MDC_CORRECTION_NOTE,
        "n_trials": int(len(drift)), "n_subjects": int(n_subjects),
    }


def _residualize_within_subject(y: np.ndarray, subj: np.ndarray, design: np.ndarray) -> np.ndarray:
    """OLS-residualise ``y`` on an intercept plus ``design``, fit SEPARATELY within each subject
    group (least-squares, tolerant of rank-deficient per-subject designs) -- so only within-
    subject covariate structure is removed, matching the within-subject control arm's own
    demeaning convention rather than pooling covariate structure across subjects."""
    resid = np.full(len(y), np.nan)
    for s in np.unique(subj):
        m = subj == s
        X = np.column_stack([np.ones(int(m.sum())), design[m]])
        coef, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        resid[m] = y[m] - X @ coef
    return resid


def _sign_matches_at_p05(native_beta: float, arm_res: dict) -> bool:
    if "beta" not in arm_res or not np.isfinite(arm_res.get("p_value", float("nan"))):
        return False
    return bool(np.sign(arm_res["beta"]) == np.sign(native_beta) and arm_res["p_value"] < 0.05)


def _fire_branch(native: dict, bias_only: dict, within_subject: dict,
                 nuisance_partialled: dict, within_subject_mdc_r: float) -> tuple[str, str]:
    """Pre-declared decision rule, fixed before any of this script's control-arm numbers were
    computed. Branch names are drawn ONLY from CONTROL_BRANCHES; nothing is invented after the
    fact.

    Priority order (mutually exclusive by construction, except for the residual case noted below):
      1. bias-only reproduces the native sign at p<0.05 -> 'bias_only_reproduces_native'.
      2. otherwise, within-subject AND nuisance-partialled both match the native sign at p<0.05
         -> 'within_subject_survives_all_controls'.
      3. otherwise, if within-subject does not reach p<0.05: a powered or underpowered null,
         decided by whether its own minimum detectable correlation clears REFERENCE_R_UNITS.
      4. otherwise (within-subject IS significant/matching but nuisance-partialled disagrees, or
         within-subject is significant with the wrong sign) -- the five pre-declared names do not
         name this exact configuration for one corpus. It is the same "the controls disagree and
         nothing settles it" situation 'mixed_across_corpora' names for disagreement ACROSS
         corpora; reused here for disagreement BETWEEN this corpus's own control arms, decided
         before results were inspected rather than adding a sixth name afterwards.
    """
    if "beta" not in native:
        return "not_evaluable", "native arm produced no beta (underpowered or excluded); no control-arm branch applies"
    native_beta = native["beta"]
    if _sign_matches_at_p05(native_beta, bias_only):
        return ("bias_only_reproduces_native",
                "the bias-only (purely between-subject) arm matches the native sign at p<0.05: "
                "the association is not established as a trial-by-trial link")
    within_hits = _sign_matches_at_p05(native_beta, within_subject)
    nuisance_hits = _sign_matches_at_p05(native_beta, nuisance_partialled)
    if within_hits and nuisance_hits:
        return ("within_subject_survives_all_controls",
                "the within-subject and nuisance-partialled arms both match the native sign at "
                "p<0.05 while the bias-only arm does not")
    within_p = within_subject.get("p_value", float("nan"))
    if "beta" in within_subject and np.isfinite(within_p) and within_p >= 0.05:
        if np.isfinite(within_subject_mdc_r) and within_subject_mdc_r < REFERENCE_R_UNITS:
            return ("within_subject_null_and_powered",
                     f"the within-subject arm does not reach p<0.05 and its minimum detectable "
                     f"correlation ({within_subject_mdc_r:.4f}) is below the {REFERENCE_R_UNITS} reference")
        return ("within_subject_null_underpowered",
                f"the within-subject arm does not reach p<0.05 and its minimum detectable "
                f"correlation is at or above the {REFERENCE_R_UNITS} reference -- inconclusive, not a null")
    return ("mixed_across_corpora",
            "the within-subject and nuisance-partialled arms disagree with each other; no "
            "pre-declared branch settles this corpus")


def _bias_only_predictor(drift: np.ndarray, subj: np.ndarray) -> np.ndarray:
    """Replace every trial's drift with its own subject's mean drift, so the predictor carries
    ONLY between-subject information."""
    return np.array([np.mean(drift[subj == s]) for s in subj])


def _within_subject_predictor(drift: np.ndarray, subj: np.ndarray) -> np.ndarray:
    """Subtract each subject's mean drift from that subject's trials, so the predictor carries
    ONLY within-subject information."""
    return drift - _bias_only_predictor(drift, subj)


BIAS_ONLY_NULL_REASON = (
    "The bias-only predictor replaces every trial's drift value with its own subject's mean "
    "drift, so within one subject the predictor never varies from trial to trial. Reshuffling "
    "trial order within each subject therefore leaves a subject-constant value completely "
    "unchanged, so a within-subject shuffle can never make this predictor look any different "
    "from the observed data -- it is not a valid null for this predictor, and a significance "
    "test built on it will always return a p-value of 1 regardless of whether a real "
    "association exists. The only way this predictor's association with the outcome can "
    "actually be scrambled is to reassign which subject's average drift value goes with which "
    "subject's own outcome data, subject by subject, while keeping every subject's own trials "
    "together as one block. That between-subject reassignment is the null this analysis uses "
    "to test the bias-only arm."
)


def _within_subject_shuffle(x: np.ndarray, subj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute trial order WITHIN each subject's own block, leaving subject membership intact --
    the null construction this project's other permutation tests use. For a subject-constant
    predictor this returns the input array unchanged, element for element (see
    BIAS_ONLY_NULL_REASON and tests/test_geometry_graded_behavior.py)."""
    out = x.copy()
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        out[idx] = x[rng.permutation(idx)]
    return out


def _subject_level_table(x: np.ndarray, y: np.ndarray,
                         subj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse trial-level (x, y) to one row per subject: x's own already-subject-constant
    value and y's subject mean. Each subject contributes exactly one row -- the correct unit
    of analysis for a purely between-subject predictor, where every trial from one subject
    carries the identical predictor value."""
    subs = np.unique(subj)
    x_i = np.array([x[subj == s][0] for s in subs])
    y_i = np.array([y[subj == s].mean() for s in subs])
    return x_i, y_i


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    var_x = np.var(x, ddof=1)
    if not np.isfinite(var_x) or var_x < 1e-12:
        return float("nan")
    return float(np.cov(x, y, ddof=1)[0, 1] / var_x)


def _between_subject_shuffle_test(x_i: np.ndarray, y_i: np.ndarray, n_perm: int,
                                  rng: np.random.Generator) -> dict:
    """Permutation test for a purely between-subject association: shuffle which subject's
    bias-only value (x_i) is paired with which subject's own outcome mean (y_i), one row per
    subject -- the null a subject-constant predictor can actually vary under (see
    BIAS_ONLY_NULL_REASON). This bypasses the mixed model entirely: fitting a subject random
    intercept alongside a fixed effect that is constant within every one of that same
    intercept's groups is a known degenerate design (the two terms compete to explain the same
    between-subject variance), which is why the mixed-model Wald test on this predictor cannot
    be trusted and a direct subject-level regression is used instead."""
    beta_obs = _ols_slope(x_i, y_i)
    if not np.isfinite(beta_obs):
        return {"status": "not_estimable", "n_subjects": int(len(x_i)),
                "reason": "zero variance in the bias-only predictor across subjects"}
    null = np.array([_ols_slope(rng.permutation(x_i), y_i) for _ in range(n_perm)])
    finite = null[np.isfinite(null)]
    if len(finite) < max(100, int(0.5 * n_perm)):
        return {"status": "not_estimable", "n_subjects": int(len(x_i)),
                "reason": f"only {len(finite)}/{n_perm} between-subject permutation draws produced a finite slope"}
    p = permutation_pvalue(np.abs(finite) >= abs(beta_obs))
    boot = np.array([
        _ols_slope(x_i[idx], y_i[idx])
        for idx in (rng.integers(0, len(x_i), len(x_i)) for _ in range(N_BOOT))
    ])
    finite_boot = boot[np.isfinite(boot)]
    if len(finite_boot) < max(100, int(0.8 * N_BOOT)):
        return {
            "status": "not_estimable", "n_subjects": int(len(x_i)),
            "reason": f"only {len(finite_boot)}/{N_BOOT} subject-bootstrap slopes were finite",
        }
    ci_lo, ci_hi = np.percentile(finite_boot, [2.5, 97.5])
    se = (ci_hi - ci_lo) / (2 * 1.96)
    r = float(np.corrcoef(x_i, y_i)[0, 1])
    return {
        "status": "computed", "beta_between_subject": beta_obs, "p_value_between_subject": p,
        "ci_lo_between_subject": float(ci_lo), "ci_hi_between_subject": float(ci_hi),
        "se_between_subject": float(se), "between_subject_r": r,
        "between_subject_r_definition": (
            "Pearson correlation of one subject-mean drift and one subject-mean response-time "
            "value per subject; no trial is treated as an independent between-subject row."
        ),
        "mdc_80_r": minimum_detectable_correlation(len(x_i)).get("mdd_r", float("nan")),
        "mdc_80_r_correction_note": (
            "Fisher-z detection bound on the one-row-per-subject table used by the permutation "
            "test; n is the number of subjects, not trials."
        ),
        "n_subjects": int(len(x_i)), "n_perm": n_perm, "n_perm_converged": int(len(finite)),
    }


def _describe_bias_only_vs_native(beta_between_subject: float, native_beta: float) -> str:
    """Standalone, self-contained sentence stating what near-equality of the bias-only and
    native betas means -- never phrased as the control 'voiding' or 'failing to reproduce' the
    native effect."""
    if not (np.isfinite(beta_between_subject) and np.isfinite(native_beta)) or native_beta == 0:
        return "the bias-only and native betas are not both finite and comparable in this cell"
    ratio = beta_between_subject / native_beta
    if np.sign(beta_between_subject) == np.sign(native_beta) and 0.5 <= abs(ratio) <= 2.0:
        return (
            f"the between-subject beta ({beta_between_subject:.4f}) is close to the native beta "
            f"({native_beta:.4f}, ratio {ratio:.2f}): the drift-outcome association is carried "
            f"mainly at the between-subject level -- a slow, subject-level effect -- rather than "
            f"arising trial by trial within a subject"
        )
    return (
        f"the between-subject beta ({beta_between_subject:.4f}) diverges from the native beta "
        f"({native_beta:.4f}, ratio {ratio:.2f}): the between-subject and native associations "
        f"are not simply restating the same effect"
    )


def _within_subject_identity_note(within_subject: dict, native: dict) -> dict:
    """Verify, numerically, that within_subject.beta equals native.beta -- a structural identity
    of this model, not a coincidence: the native fit already carries a per-subject random
    intercept, so its fitted drift coefficient IS the within-subject slope, and manually
    subtracting each subject's own mean drift before fitting removes information the random
    intercept already absorbed on its own, changing nothing about the fitted coefficient."""
    if "beta" not in within_subject or "beta" not in native:
        return {}
    # The two fits share the same theoretical optimum (subject-demeaning a predictor already
    # absorbed by the random intercept changes nothing about the likelihood surface), but they
    # are two SEPARATE MixedLM optimizer runs on differently-shifted design columns, so they can
    # converge to that optimum via different iteration paths. On this project's real corpora the
    # two betas have agreed to ~1e-16 (bit-identical); a 0.1% relative tolerance comfortably
    # covers that while still catching a genuine model-structure regression, which would be
    # orders of magnitude larger.
    identical = bool(np.isclose(within_subject["beta"], native["beta"], rtol=1e-3, atol=1e-6))
    assert identical, (
        "within-subject beta no longer matches native beta exactly -- the subject-demeaning "
        "identity this arm's disclosure relies on has broken and must be investigated before "
        "this arm is reported as anything but a bug"
    )
    return {
        "arithmetically_identical_to_native": identical,
        "identity_reason": (
            "this arm's beta is retained as an explicit within-subject readout, not as an "
            "independent corroborating control: the native model's per-subject random "
            "intercept already makes its fitted drift coefficient the within-subject slope, so "
            "subject-demeaning the predictor before fitting changes nothing and this beta is "
            "arithmetically identical to the native beta by construction"
        ),
    }


def _nuisance_partialled_verdict(nuisance_arm: dict) -> tuple[str, str]:
    """Powered-null-or-inconclusive determination for the nuisance-partialled arm in r units
    against REFERENCE_R_UNITS -- the arm this analysis leads with."""
    if "beta" not in nuisance_arm or not np.isfinite(nuisance_arm.get("p_value", float("nan"))):
        return ("not_evaluable",
               "the nuisance-partialled arm produced no beta (underpowered, excluded, or no "
               "nuisance covariates were available)")
    p = nuisance_arm["p_value"]
    mdc = nuisance_arm.get("mdc_80_r", float("nan"))
    if p < 0.05:
        r = nuisance_arm.get("within_subject_r", float("nan"))
        return "significant", f"p={p:.4f} < 0.05, subject-adjusted within-subject r={r:.4f}"
    if np.isfinite(mdc) and mdc < REFERENCE_R_UNITS:
        return ("powered_null",
                f"p={p:.4f} is not significant and the minimum detectable correlation at 80% "
                f"power ({mdc:.4f} r units) is below the {REFERENCE_R_UNITS} r-unit reference "
                f"-- a powered null")
    return ("inconclusive",
           f"p={p:.4f} is not significant but the minimum detectable correlation at 80% power "
           f"({mdc:.4f} r units) is at or above the {REFERENCE_R_UNITS} r-unit reference -- "
           f"underpowered to rule out an effect at that scale")


def _run_control_arms(trials: dict, seed_prefix: str, native: dict) -> dict:
    """Fit the bias-only, within-subject and nuisance-partialled arms on the same trial rows the
    native cell used, with the identical model call, seed convention and bootstrap as the native
    cell -- and fire the pre-declared branch. ``trials`` holds the per-trial arrays a corpus
    extraction function assembled (see the ``*_run_dandi*``/``run_boran`` functions). ``native``
    is the already-computed native-arm result, needed to check the within-subject identity and
    to state what near-equality of the bias-only and native betas means. Detection bounds use
    subject-cluster bootstrap uncertainty from the same rows and effect estimand."""
    drift, rt, subj = trials["drift"], trials["rt"], trials["subj"]

    # The bias arm has only between-subject variation, so its complete analysis is the direct
    # one-row-per-subject permutation/bootstrap; a random-intercept LME is non-identifiable for
    # this predictor and is neither fit nor pooled.
    bias_x_i, bias_y_i = _subject_level_table(_bias_only_predictor(drift, subj), rt, subj)
    between_subject = _between_subject_shuffle_test(
        bias_x_i, bias_y_i, N_PERM,
        np.random.default_rng(stable_seed(seed_prefix + "_between_subject")),
    )
    if between_subject["status"] == "computed":
        bias_only = {
            "beta": between_subject["beta_between_subject"],
            "ci_lo": between_subject["ci_lo_between_subject"],
            "ci_hi": between_subject["ci_hi_between_subject"],
            "se": between_subject["se_between_subject"],
            "p_value": between_subject["p_value_between_subject"],
            "between_subject_r": between_subject["between_subject_r"],
            "between_subject_r_definition": between_subject["between_subject_r_definition"],
            "mdc_80_r": between_subject["mdc_80_r"],
            "mdc_80_r_correction_note": between_subject["mdc_80_r_correction_note"],
            "n_subjects": between_subject["n_subjects"],
            "n_trials": int(len(drift)),
            "n_perm_between_subject": between_subject["n_perm"],
            "n_perm_between_subject_converged": between_subject["n_perm_converged"],
            "reason": BIAS_ONLY_NULL_REASON,
        }
        if "beta" in native:
            bias_only["beta_vs_native"] = _describe_bias_only_vs_native(
                bias_only["beta"], native["beta"])
    else:
        bias_only = {
            "not_computable": True, "between_subject_test_status": between_subject["status"],
            "between_subject_test_reason": between_subject["reason"],
            "n_subjects": int(len(bias_x_i)), "n_trials": int(len(drift)),
        }

    # Subject-demeaning the predictor is already the native random-intercept slope's explicit
    # within-subject representation. Reuse the native fit and bootstrap rather than performing
    # thousands of mathematically redundant optimizer calls.
    within_subject = dict(native)
    within_subject.update(_within_subject_identity_note(within_subject, native))

    covariate_cols, missing = [], []
    for cname, carr in (("trial_index_within_subject", trials["trial_idx"]),
                        ("memory_load", trials["load"]), ("trial_outcome", trials["outcome"]),
                        ("latent_amplitude", trials["amp"])):
        if carr is None:
            missing.append(cname)
        else:
            covariate_cols.append(carr.astype(float))

    n_trials_seen = len(drift)
    if covariate_cols:
        design = np.column_stack(covariate_cols)
        covariate_finite = np.all(np.isfinite(design), axis=1)
        n_refused = int((~covariate_finite).sum())
        trial_refusal_reason = ("nuisance covariate design contains a non-finite value"
                                if n_refused else None)
        resid_drift = _residualize_within_subject(drift, subj, design)
        resid_rt = _residualize_within_subject(rt, subj, design)
        nuisance_partialled = _lme_report(resid_drift, resid_rt, subj, seed_prefix + "_nuisance_partialled")
    else:
        n_refused = n_trials_seen
        trial_refusal_reason = "no nuisance covariates available"
        nuisance_partialled = {"not_computable": True, "reason": "no nuisance covariates available"}
    if missing:
        nuisance_partialled["missing_covariates"] = missing

    nuisance_verdict, nuisance_verdict_detail = _nuisance_partialled_verdict(nuisance_partialled)
    nuisance_partialled["verdict"] = nuisance_verdict
    nuisance_partialled["verdict_detail"] = nuisance_verdict_detail

    n_analysed = n_trials_seen - n_refused
    zero_drop = {
        "trials_seen": n_trials_seen, "trials_analysed": n_analysed, "trials_refused": n_refused,
        "trial_refusal_reason": trial_refusal_reason,
        "zero_drop_holds": bool(n_trials_seen == n_analysed + n_refused),
    }
    assert zero_drop["zero_drop_holds"], f"{seed_prefix}: seen != analysed + refused"

    return {"bias_only": bias_only, "within_subject": within_subject,
           "nuisance_partialled": nuisance_partialled, "nuisance_partialled_zero_drop": zero_drop}


# ── Per-dataset extraction ──────────────────────────────────────────────────

def run_dandi000469() -> tuple[dict, dict | None]:
    all_drift, all_rt, all_subj, all_load, all_outcome, all_amp, all_idx = [], [], [], [], [], [], []
    sessions_seen, sessions_refused = 0, []
    for path in sorted(RESULTS.glob("dandi000469_geometry_sub-*.npz")):
        sessions_seen += 1
        subj = path.stem.replace("dandi000469_geometry_", "")
        d = np.load(path, allow_pickle=True)
        if "drift" not in d:
            sessions_refused.append({"session": subj, "reason": "no drift field in geometry file"})
            continue
        nwb = DATA_ROOT / "000469" / subj / f"{subj}_ses-2_ecephys+image.nwb"
        if not nwb.exists():
            sessions_refused.append({"session": subj, "reason": "matching NWB file not found"})
            continue
        with h5py.File(str(nwb), "r") as f:
            trials = f["intervals/trials"]
            t_probe = trials["timestamps_Probe"][:]
            t_resp = trials["timestamps_Response"][:]
        rt = t_resp - t_probe
        drift = d["drift"]
        if len(rt) != len(drift):
            sessions_refused.append({"session": subj, "reason": "RT and drift trial counts do not match"})
            continue
        all_drift.append(drift); all_rt.append(rt); all_subj.append([subj] * len(drift))
        all_load.append(d["loads"]); all_outcome.append(d["response_accuracy"].astype(float))
        all_amp.append(np.linalg.norm(d["Z"], axis=2).mean(axis=1)); all_idx.append(np.arange(len(drift)))
    if not all_drift:
        return {"excluded_no_graded_behavior": True, "reason": "no geometry files with drift + matching NWB"}, None
    res = _lme_report(np.concatenate(all_drift), np.concatenate(all_rt),
                      np.concatenate(all_subj), "geobeh_000469")
    trials_bundle = {
        "drift": np.concatenate(all_drift), "rt": np.concatenate(all_rt), "subj": np.concatenate(all_subj),
        "load": np.concatenate(all_load), "outcome": np.concatenate(all_outcome),
        "amp": np.concatenate(all_amp), "trial_idx": np.concatenate(all_idx),
        "sessions_seen": sessions_seen, "sessions_analysed": sessions_seen - len(sessions_refused),
        "sessions_refused": sessions_refused,
    }
    return res, trials_bundle


def run_dandi001187() -> tuple[dict, dict | None]:
    all_drift, all_rt, all_subj, all_load, all_outcome, all_amp, all_idx = [], [], [], [], [], [], []
    sessions_seen, sessions_refused = 0, []
    for path in sorted(RESULTS.glob("dandi001187_geometry_sub-*.npz")):
        sessions_seen += 1
        key = path.stem.replace("dandi001187_geometry_", "")
        subj_dir = key.split("_", 1)[0]
        d = np.load(path, allow_pickle=True)
        if "drift" not in d:
            sessions_refused.append({"session": key, "reason": "no drift field in geometry file"})
            continue
        nwb = DATA_ROOT / "001187" / subj_dir / f"{key}.nwb"
        if not nwb.exists():
            sessions_refused.append({"session": key, "reason": "matching NWB file not found"})
            continue
        with h5py.File(str(nwb), "r") as f:
            rt = f["intervals/WM_trials/response_time"][:]
        drift = d["drift"]
        if len(rt) != len(drift):
            sessions_refused.append({"session": key, "reason": "RT and drift trial counts do not match"})
            continue
        all_drift.append(drift); all_rt.append(rt); all_subj.append([subj_dir] * len(drift))
        all_load.append(d["loads"]); all_outcome.append(d["response_accuracy"].astype(float))
        all_amp.append(np.linalg.norm(d["Z"], axis=2).mean(axis=1)); all_idx.append(np.arange(len(drift)))
    if not all_drift:
        return {"excluded_no_graded_behavior": True, "reason": "no geometry files with drift + matching NWB"}, None
    res = _lme_report(np.concatenate(all_drift), np.concatenate(all_rt),
                      np.concatenate(all_subj), "geobeh_001187")
    trials_bundle = {
        "drift": np.concatenate(all_drift), "rt": np.concatenate(all_rt), "subj": np.concatenate(all_subj),
        "load": np.concatenate(all_load), "outcome": np.concatenate(all_outcome),
        "amp": np.concatenate(all_amp), "trial_idx": np.concatenate(all_idx),
        "sessions_seen": sessions_seen, "sessions_analysed": sessions_seen - len(sessions_refused),
        "sessions_refused": sessions_refused,
    }
    return res, trials_bundle


def run_dandi000673() -> tuple[dict, dict | None]:
    all_drift, all_rt, all_subj, all_load, all_outcome, all_amp, all_idx = [], [], [], [], [], [], []
    sessions_seen, sessions_refused = 0, []
    for path in sorted(RESULTS.glob("dandi000673_geometry_sub-*.npz")):
        sessions_seen += 1
        key = path.stem.replace("dandi000673_geometry_", "")
        subj_dir = key.split("_", 1)[0]
        d = np.load(path, allow_pickle=True)
        if "drift" not in d:
            sessions_refused.append({"session": key, "reason": "no drift field in geometry file"})
            continue
        nwb = DATA_ROOT / "000673" / subj_dir / f"{key}.nwb"
        if not nwb.exists():
            sessions_refused.append({"session": key, "reason": "matching NWB file not found"})
            continue
        with h5py.File(str(nwb), "r") as f:
            trials = f["intervals/trials"]
            t_probe = trials["timestamps_Probe"][:]
            t_resp = trials["timestamps_Response"][:]
        rt = t_resp - t_probe
        drift = d["drift"]
        if len(rt) != len(drift):
            sessions_refused.append({"session": key, "reason": "RT and drift trial counts do not match"})
            continue
        all_drift.append(drift); all_rt.append(rt); all_subj.append([subj_dir] * len(drift))
        all_load.append(d["loads"]); all_outcome.append(d["response_accuracy"].astype(float))
        all_amp.append(np.linalg.norm(d["Z"], axis=2).mean(axis=1)); all_idx.append(np.arange(len(drift)))
    if not all_drift:
        return {"excluded_no_graded_behavior": True, "reason": "no geometry files with drift + matching NWB"}, None
    res = _lme_report(np.concatenate(all_drift), np.concatenate(all_rt),
                      np.concatenate(all_subj), "geobeh_000673")
    trials_bundle = {
        "drift": np.concatenate(all_drift), "rt": np.concatenate(all_rt), "subj": np.concatenate(all_subj),
        "load": np.concatenate(all_load), "outcome": np.concatenate(all_outcome),
        "amp": np.concatenate(all_amp), "trial_idx": np.concatenate(all_idx),
        "sessions_seen": sessions_seen, "sessions_analysed": sessions_seen - len(sessions_refused),
        "sessions_refused": sessions_refused,
    }
    return res, trials_bundle


def run_boran() -> tuple[dict, dict | None]:
    all_drift, all_rt, all_subj, all_load, all_outcome, all_amp, all_idx = [], [], [], [], [], [], []
    sessions_seen, sessions_refused = 0, []
    for path in sorted(RESULTS.glob("boran_geometry_sub-*.npz")):
        sessions_seen += 1
        subj = path.stem.replace("boran_geometry_", "")
        d = np.load(path, allow_pickle=True)
        if "drift" not in d:
            sessions_refused.append({"session": subj, "reason": "no drift field in geometry file"})
            continue
        if "response_time" in d:
            rt = d["response_time"]
        else:
            # Fallback: boran_geometry_*.npz predates the response_time field
            # -- re-derive directly from NWB, same convention
            # as run_boran_pipeline.load_subject_sessions (response_time is an
            # absolute NWB timestamp; subtract trial start_time for latency).
            nwbs = sorted((DATA_ROOT / "000574" / subj).glob("*.nwb"))
            rt_parts = []
            for nwb_path in nwbs:
                with h5py.File(str(nwb_path), "r") as f:
                    t_start = f["intervals/trials/start_time"][:]
                    resp = f["intervals/trials/response_time"][:] - t_start
                    artifact = f["intervals/trials/artifact"][:].astype(bool)
                rt_parts.append(resp[~artifact])
            rt = np.concatenate(rt_parts) if rt_parts else np.array([])
        drift = d["drift"]
        if len(rt) != len(drift):
            sessions_refused.append({"session": subj, "reason": "RT and drift trial counts do not match"})
            continue
        all_drift.append(drift); all_rt.append(rt); all_subj.append([subj] * len(drift))
        all_load.append(d["set_sizes"]); all_outcome.append(d["correct"].astype(float))
        all_amp.append(np.linalg.norm(d["Z"], axis=2).mean(axis=1)); all_idx.append(np.arange(len(drift)))
    if not all_drift:
        return {"excluded_no_graded_behavior": True, "reason": "no geometry files with drift + matching RT"}, None
    res = _lme_report(np.concatenate(all_drift), np.concatenate(all_rt),
                      np.concatenate(all_subj), "geobeh_boran")
    trials_bundle = {
        "drift": np.concatenate(all_drift), "rt": np.concatenate(all_rt), "subj": np.concatenate(all_subj),
        "load": np.concatenate(all_load), "outcome": np.concatenate(all_outcome),
        "amp": np.concatenate(all_amp), "trial_idx": np.concatenate(all_idx),
        "sessions_seen": sessions_seen, "sessions_analysed": sessions_seen - len(sessions_refused),
        "sessions_refused": sessions_refused,
    }
    return res, trials_bundle


CONFIDENCE_NOTE = (
    "No dataset used for maintenance geometry has a genuine subject-reported "
    "confidence field aligned to its geometry trials. dandi001187 has a "
    "'confidence' field, but only in the separate intervals/LTM_trials "
    "(long-term-memory recognition) task table, not intervals/WM_trials (the "
    "Sternberg maintenance task the geometry/drift here is computed from). A "
    "decoder-derived confidence proxy was considered and rejected: it would "
    "be computed from the same latent Z as the drift predictor (held-out CV, "
    "so not perfectly circular, but not independent either), which would "
    "confound the geometry-behavior link this analysis is meant to test."
)


def _pooled_statistic_reconciliation(estimates: np.ndarray, ses: np.ndarray, labels: list[str],
                                     subject_ns: np.ndarray, forest: dict, stouffer: dict) -> dict:
    """Explain, from the same rows already pooled, why the forest (inverse-variance,
    random-effects) and Stouffer (p-combination, cohort-size-weighted) statistics computed here
    can legitimately disagree about significance without either being wrong -- they weight the
    same corpora differently and the forest alone carries a heterogeneity term. Recomputed fresh
    from ``estimates``/``ses``/``labels``/``subject_ns`` on every call, never hard-coded, so this
    note stays true if the underlying per-corpus rows ever change."""
    fixed = forest_meta(estimates, ses, labels=labels, method="fixed")
    iv_weight_share = {row["label"]: row["weight_pct"] / 100.0 for row in fixed["rows"]}
    subject_ns = np.asarray(subject_ns, dtype=float)
    sqrt_n = np.sqrt(subject_ns)
    p_weight_share = {lab: float(w) for lab, w in zip(labels, sqrt_n / sqrt_n.sum())}
    se_inflation = float(forest["se"] / fixed["se"]) if fixed["se"] > 0 else float("nan")
    iv_share_str = ", ".join(f"{lab}={iv_weight_share.get(lab, float('nan')):.4f}" for lab in labels)
    p_share_str = ", ".join(f"{lab}={p_weight_share.get(lab, float('nan')):.4f}" for lab in labels)
    note = (
        f"The forest (pooled={forest['pooled']:.6f}, p={forest['p_value']:.6f}) and the "
        f"Stouffer p-combination (z_combined={stouffer['z_combined']:.6f}, "
        f"p_combined={stouffer['p_combined']:.6f}) answer different questions from the same "
        f"per-corpus rows and are not in conflict about the data. The p-combination weights each "
        f"corpus by cohort size (sqrt(n_subjects) shares: {p_share_str}), while the forest "
        f"weights by inverse-variance precision (shares: {iv_share_str}), so the two statistics "
        f"weight the same corpora almost oppositely. The forest is also random-effects: "
        f"between-corpus heterogeneity (I^2={forest['i_squared']:.2f}%) inflates its standard "
        f"error by a factor of {se_inflation:.4f} relative to a fixed-effect pool of the "
        f"identical rows (fixed-effect pooled={fixed['pooled']:.6f}, se={fixed['se']:.6f}, "
        f"p={fixed['p_value']:.6f}) -- that inflation alone is what can carry the forest p-value "
        f"across conventional significance where a fixed-effect pool of the same rows would not, "
        f"and the p-combination has no way to represent that heterogeneity at all. The forest is "
        f"reported as the pooled answer here because the claim at issue is whether a single "
        f"common graded effect size holds across corpora, not merely whether the corpora jointly "
        f"provide evidence against a no-effect null; the p-combination is a secondary evidence "
        f"summary and should not be quoted as the pooled result."
    )
    return {
        "reported_pooled_statistic": "forest",
        "fixed_effect_pooled": fixed["pooled"],
        "fixed_effect_se": fixed["se"],
        "fixed_effect_p_value": fixed["p_value"],
        "random_effects_se_inflation_factor": se_inflation,
        "inverse_variance_weight_share": iv_weight_share,
        "p_combination_weight_share": p_weight_share,
        "note": note,
    }


def _pool_arm(scored_corpora: dict, arm_key: str | None, labels: list[str]) -> dict | None:
    """Pool one arm (native if arm_key is None, else 'bias_only'/'within_subject'/
    'nuisance_partialled') across corpora with the identical inverse-variance (forest_meta) and
    p-combination (stouffer_combine) the native pooled cell uses. Corpora where that arm did not
    produce a beta (underpowered/not_computable) are dropped from this arm's pool, same rule the
    native pooling already applies. The forest is the reported pooled effect; the p-combination
    is a secondary evidence summary that weights corpora differently and carries no heterogeneity
    term, so it can disagree with the forest without either statistic being wrong (see
    pooled_statistic_reconciliation, added to this arm's output below)."""
    rows = {}
    for k in labels:
        arm = scored_corpora[k] if arm_key is None else scored_corpora[k].get(arm_key, {})
        if "beta" in arm and np.isfinite(arm.get("se", float("nan"))):
            rows[k] = arm
    if len(rows) < 2:
        return {"note": f"only {len(rows)} scored cohort(s) -- meta-analysis needs >=2"}
    pool_labels = list(rows.keys())
    estimates = np.array([rows[k]["beta"] for k in pool_labels])
    ses = np.array([rows[k]["se"] for k in pool_labels])
    meta = forest_meta(estimates, ses, labels=pool_labels)
    p_one_sided = np.array([
        rows[k]["p_value"] / 2 if rows[k]["beta"] >= 0 else 1 - rows[k]["p_value"] / 2
        for k in pool_labels
    ])
    subject_ns = np.array([rows[k].get("n_subjects", 1) for k in pool_labels], dtype=float)
    stouffer = stouffer_combine(p_one_sided, weights=np.sqrt(subject_ns))
    n_total = int(sum(rows[k]["n_trials"] for k in pool_labels))
    pooled_arm = {"forest": meta, "stouffer": stouffer,
                 "pooled_statistic_reconciliation": _pooled_statistic_reconciliation(
                     estimates, ses, pool_labels, subject_ns, meta, stouffer),
                 "datasets_pooled": pool_labels,
                 "n_trials_total": n_total,
                 "n_subjects_total": int(np.sum(subject_ns))}

    # Pool the effect-size scale separately from the native-unit beta. Within/native/nuisance
    # rows use subject-adjusted r with subject-bootstrap SE; the bias-only arm has no within-
    # subject variation to bootstrap at all (its predictor is subject-constant), so it is pooled
    # on its own between-subject, one-row-per-subject r scale instead, using the same analytic
    # Fisher-z SE (1/sqrt(n_subjects-3)) each corpus's own mdc_80_r already used.
    if arm_key == "bias_only":
        r_rows = {
            k: (rows[k].get("between_subject_r"), rows[k].get("n_subjects"))
            for k in pool_labels
        }
        r_rows = {k: (r, n) for k, (r, n) in r_rows.items()
                  if np.isfinite(r if r is not None else np.nan)
                  and n is not None and n >= 4}
        if len(r_rows) >= 2:
            r_labels = list(r_rows)
            r_values = np.array([r_rows[k][0] for k in r_labels], dtype=float)
            z_ses = 1.0 / np.sqrt(np.array([r_rows[k][1] for k in r_labels], dtype=float) - 3)
            z_values = np.arctanh(np.clip(r_values, -0.999999, 0.999999))
            r_meta_z = forest_meta(z_values, z_ses, labels=r_labels)
            pooled_arm["between_subject_r_forest_fisher_z"] = r_meta_z
            pooled_arm["between_subject_r"] = float(np.tanh(r_meta_z["pooled"]))
            pooled_arm["between_subject_r_ci_lo"] = float(np.tanh(r_meta_z["ci_lo"]))
            pooled_arm["between_subject_r_ci_hi"] = float(np.tanh(r_meta_z["ci_hi"]))
            pooled_arm["mdc_80_r"] = float(np.tanh(Z_FACTOR_80_POWER * r_meta_z["se"]))
            pooled_arm["mdc_80_r_correction_note"] = (
                "This arm's predictor is subject-constant, so it has a one-row-per-subject "
                "between-subject estimand, not the within-subject trial-level one the other "
                "pooled arms bootstrap -- there is no within-subject variation to resample. Its "
                "pooled detection floor instead comes from a Fisher-z random-effects meta-"
                "analysis of each corpus's between_subject_r, using the same analytic z-scale "
                "standard error (1/sqrt(n_subjects-3)) each corpus's own mdc_80_r already used. "
                "It is expressed in the same Pearson-r units as the other arms' mdc_80_r but "
                "answers a different question -- a between-subject association, not a within-"
                "subject one -- so it is not directly comparable to those."
            )
    else:
        r_rows = {
            k: (rows[k].get("within_subject_r"), rows[k].get("within_subject_r_se"))
            for k in pool_labels
        }
        r_rows = {k: (r, se) for k, (r, se) in r_rows.items()
                  if np.isfinite(r if r is not None else np.nan)
                  and np.isfinite(se if se is not None else np.nan) and se > 0}
        if len(r_rows) >= 2:
            r_labels = list(r_rows)
            r_values = np.array([r_rows[k][0] for k in r_labels], dtype=float)
            r_ses = np.array([r_rows[k][1] for k in r_labels], dtype=float)
            z_values = np.arctanh(np.clip(r_values, -0.999999, 0.999999))
            z_ses = r_ses / np.maximum(1e-12, 1 - r_values ** 2)
            r_meta_z = forest_meta(z_values, z_ses, labels=r_labels)
            pooled_arm["within_subject_r_forest_fisher_z"] = r_meta_z
            pooled_arm["within_subject_r"] = float(np.tanh(r_meta_z["pooled"]))
            pooled_arm["within_subject_r_ci_lo"] = float(np.tanh(r_meta_z["ci_lo"]))
            pooled_arm["within_subject_r_ci_hi"] = float(np.tanh(r_meta_z["ci_hi"]))
            pooled_arm["within_subject_r_se_fisher_z"] = float(r_meta_z["se"])
            pooled_arm["mdc_80_r"] = float(np.tanh(Z_FACTOR_80_POWER * r_meta_z["se"]))
            pooled_arm["mdc_80_r_correction_note"] = (
                "The pooled detection floor is computed from the random-effects Fisher-z meta-"
                "analytic SE of per-corpus subject-adjusted correlations. Each per-corpus SE comes "
                "from resampling whole subjects; raw trial counts never define power."
            )
    pooled_arm["beta"] = meta["pooled"]
    pooled_arm["p_value"] = meta["p_value"]
    pooled_arm["n_trials"] = n_total
    return pooled_arm


def _require_complete_corpus_admission(scored_corpora: dict) -> None:
    """Refuse to replace a multi-corpus result with an incomplete run."""
    missing = [name for name in EXPECTED_CORPORA if name not in scored_corpora]
    if missing:
        raise RuntimeError(
            "refusing to write geometry_graded_behavior.json: missing scored corpus "
            + ", ".join(missing)
        )


def main():
    out = {}
    control_arms = {}
    for name, fn, seed_prefix in [
        ("dandi000469", run_dandi000469, "geobeh_000469"),
        ("dandi001187", run_dandi001187, "geobeh_001187"),
        ("dandi000673", run_dandi000673, "geobeh_000673"),
        ("boran_ieeg", run_boran, "geobeh_boran"),
    ]:
        print(f"\n=== {name} (RT ~ drift) ===")
        res, trials = fn()
        response_time = dict(res)  # native keys unchanged; everything below is additive
        if trials is not None:
            arms = _run_control_arms(trials, seed_prefix, res)
            response_time.update(arms)
            branch, detail = _fire_branch(
                res, arms["bias_only"], arms["within_subject"], arms["nuisance_partialled"],
                arms["within_subject"].get("mdc_80_r", float("nan")),
            )
            response_time["branch"] = branch
            response_time["branch_detail"] = detail
            response_time["leading_determination"] = LEADING_ARM
            response_time["sessions_seen"] = trials["sessions_seen"]
            response_time["sessions_analysed"] = trials["sessions_analysed"]
            response_time["sessions_refused"] = trials["sessions_refused"]
            assert trials["sessions_seen"] == trials["sessions_analysed"] + len(trials["sessions_refused"]), \
                f"{name}: sessions seen != analysed + refused"
        out[name] = {"response_time": response_time, "confidence": {
            "excluded_no_graded_behavior": True, "reason": CONFIDENCE_NOTE}}
        control_arms[name] = response_time
        if res.get("underpowered"):
            print(f"  underpowered: n_trials={res['n_trials']}, n_subjects={res['n_subjects']}")
        elif res.get("excluded_no_graded_behavior"):
            print(f"  excluded: {res['reason']}")
        else:
            print(f"  native  beta={res['beta']:.4f} [{res['ci_lo']:.4f}, {res['ci_hi']:.4f}] "
                  f"p={res['p_value']:.4f} n_trials={res['n_trials']} n_subjects={res['n_subjects']}")
            if trials is not None:
                print(f"  branch={response_time['branch']}")

    # Meta-combine the scored (non-underpowered, non-excluded) datasets -- native cell, byte-
    # identical to the pre-existing pooling logic.
    scored = {k: v["response_time"] for k, v in out.items()
             if "beta" in v["response_time"]}
    _require_complete_corpus_admission(scored)
    if len(scored) >= 2:
        # Preserve the formerly reported four-release combination as an
        # explicitly non-inferential sensitivity record.  It cannot be the
        # headline because 001187 and 000673 contain overlapping people and
        # sessions; their agreement or disagreement is useful, their double
        # weight is not.
        all_release_labels = list(scored.keys())
        all_release_estimates = np.array([scored[k]["beta"] for k in all_release_labels])
        all_release_ses = np.array([scored[k]["se"] for k in all_release_labels])
        all_release_meta = forest_meta(
            all_release_estimates, all_release_ses, labels=all_release_labels)
        all_release_p_one_sided = np.array([
            scored[k]["p_value"] / 2 if scored[k]["beta"] >= 0
            else 1 - scored[k]["p_value"] / 2
            for k in all_release_labels
        ])
        all_release_subject_ns = np.array(
            [scored[k]["n_subjects"] for k in all_release_labels], dtype=float)
        all_release_stouffer = stouffer_combine(
            all_release_p_one_sided, weights=np.sqrt(all_release_subject_ns))
        out["_all_release_views_sensitivity"] = {
            "status": "sensitivity_only_not_primary_inference",
            "reason": LINKED_VIEW_NOTE,
            "forest": all_release_meta,
            "stouffer": all_release_stouffer,
            "pooled_statistic_reconciliation": _pooled_statistic_reconciliation(
                all_release_estimates, all_release_ses, all_release_labels,
                all_release_subject_ns, all_release_meta, all_release_stouffer),
            "datasets_combined": all_release_labels,
        }

        labels = [name for name in PRIMARY_CORPORA if name in scored]
        estimates = np.array([scored[k]["beta"] for k in labels])
        ses = np.array([scored[k]["se"] for k in labels])
        meta = forest_meta(estimates, ses, labels=labels)
        # One-sided p per dataset in the direction of its own beta sign, for Stouffer.
        p_one_sided = np.array([
            scored[k]["p_value"] / 2 if scored[k]["beta"] >= 0 else 1 - scored[k]["p_value"] / 2
            for k in labels
        ])
        primary_subject_ns = np.array([scored[k]["n_subjects"] for k in labels], dtype=float)
        stouffer = stouffer_combine(p_one_sided, weights=np.sqrt(primary_subject_ns))
        out["_meta"] = {
            "status": "primary_independent_corpora",
            "forest": meta,
            "stouffer": stouffer,
            "pooled_statistic_reconciliation": _pooled_statistic_reconciliation(
                estimates, ses, labels, primary_subject_ns, meta, stouffer),
            "datasets_pooled": labels,
            "linked_sensitivity_views_excluded": list(LINKED_SENSITIVITY_CORPORA),
            "linked_view_note": LINKED_VIEW_NOTE,
            "n_subjects_total": int(np.sum(primary_subject_ns)),
        }
        print(f"\n=== Primary meta (RT ~ drift), {len(labels)} independent cohorts ===")
        print(f"  pooled beta={meta['pooled']:.4f} [{meta['ci_lo']:.4f}, {meta['ci_hi']:.4f}] "
              f"p={meta['p_value']:.4f}, I^2={meta['i_squared']:.1f}%")

        # Pool the three control arms the identical way, and fire the pre-declared branch at the
        # pooled level too.
        bias_only_pool = _pool_arm(scored, "bias_only", labels)
        within_subject_pool = _pool_arm(scored, "within_subject", labels)
        nuisance_pool = _pool_arm(scored, "nuisance_partialled", labels)
        native_pool_arm = _pool_arm(scored, None, labels)
        if isinstance(native_pool_arm, dict):
            for key in (
                "within_subject_r", "within_subject_r_ci_lo", "within_subject_r_ci_hi",
                "within_subject_r_se_fisher_z", "within_subject_r_forest_fisher_z", "mdc_80_r",
                "mdc_80_r_correction_note",
            ):
                if key in native_pool_arm:
                    out["_meta"][key] = native_pool_arm[key]
        within_mdc = within_subject_pool.get("mdc_80_r", float("nan")) if isinstance(within_subject_pool, dict) else float("nan")
        pooled_branch, pooled_detail = _fire_branch(
            native_pool_arm if isinstance(native_pool_arm, dict) else {},
            bias_only_pool if "beta" in (bias_only_pool or {}) else {},
            within_subject_pool if "beta" in (within_subject_pool or {}) else {},
            nuisance_pool if "beta" in (nuisance_pool or {}) else {},
            within_mdc,
        )
        per_corpus_branches = {k: control_arms[k].get("branch") for k in labels if "branch" in control_arms[k]}
        if len(set(per_corpus_branches.values())) > 1 and pooled_branch not in (
            "bias_only_reproduces_native", "within_subject_survives_all_controls",
            "within_subject_null_and_powered", "within_subject_null_underpowered",
        ):
            pooled_branch, pooled_detail = "mixed_across_corpora", (
                "per-corpus branches disagree (" + ", ".join(f"{k}={v}" for k, v in per_corpus_branches.items())
                + ") and the pooled cell does not settle it"
            )
        if isinstance(bias_only_pool, dict) and "beta" in bias_only_pool:
            bias_only_pool["reason"] = BIAS_ONLY_NULL_REASON
            bias_only_pool["beta_vs_native"] = _describe_bias_only_vs_native(
                bias_only_pool["beta"], native_pool_arm["beta"])
        if isinstance(nuisance_pool, dict) and "beta" in nuisance_pool:
            nuisance_pool["verdict"], nuisance_pool["verdict_detail"] = _nuisance_partialled_verdict(nuisance_pool)
        out["_meta"]["bias_only"] = bias_only_pool
        out["_meta"]["within_subject"] = within_subject_pool
        out["_meta"]["nuisance_partialled"] = nuisance_pool
        out["_meta"]["branch"] = pooled_branch
        out["_meta"]["branch_detail"] = pooled_detail
        out["_meta"]["per_corpus_branches"] = per_corpus_branches
        out["_meta"]["leading_determination"] = LEADING_ARM
        print(f"  pooled branch={pooled_branch}")
    else:
        out["_meta"] = {"note": f"only {len(scored)} scored cohort(s) -- meta-analysis needs >=2"}
        print(f"\nOnly {len(scored)} scored cohort(s) -- skipping meta-analysis")

    out["_confidence_note"] = CONFIDENCE_NOTE

    with open(RESULTS / "geometry_graded_behavior.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/geometry_graded_behavior.json")


if __name__ == "__main__":
    main()
