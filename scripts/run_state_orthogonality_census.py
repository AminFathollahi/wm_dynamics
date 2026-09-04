"""run_state_orthogonality_census.py -- turn this project's "the state is not
X" nulls into bounded exclusions, and check that the observable those bounds
are about is reliable enough to be orthogonal to anything.

A non-significant test says nothing on its own about how large an association
it could have ruled out. Every null in this project that bears on what the
cross-unit maintenance state IS gets three things here: its point estimate and
confidence interval as already computed, the minimum effect it could have
detected at 80% power (statistics.minimum_detectable_paired_difference, the
same primitive results/rate_free_state_geometry_behavior_link.json already
reports for its behavioural contrast), and a plain-language bound saying what
the test does and does not exclude. Nulls are sorted most-constraining first.
A null whose minimum detectable effect is larger than the association it was
meant to rule out is reported as excluding nothing, not carried forward as
evidence of orthogonality.

Two further things the bound arithmetic alone cannot deliver, both run here:

  Sign-invariance audit. The leading-component gain is the first left singular
  vector of a per-session trials x windows matrix, scaled by its singular
  value. The overall sign of a singular vector is set by the decomposition,
  not by the data, and is not comparable across sessions -- state_persistence.
  temporal_profile_sign_crossings already says so for the matching right
  singular vector. A pooled test that averages SIGNED per-session correlations
  of such an observable against a covariate therefore has a mean near zero
  whether or not the per-session association is strong, and its apparent null
  is a property of the estimator. This module tests that possibility directly
  (are the per-session signs distinguishable from coin flips, and how large is
  the per-session association once its sign is discarded) rather than assuming
  either way, and recomputes each candidate with a sign-invariant statistic.

  Same-observable census. The nulls on disk are spread over three different
  observables: the leading-component gain, the pairwise window correlation
  used by the lag census, and the rate-free direction deviation that carries
  the behavioural prediction. "The component that is orthogonal to everything
  is the component that predicts behaviour" is only a claim about one
  component if one observable carries both halves. Every candidate identity is
  therefore recomputed here on the rate-free deviation observable -- the one
  the behavioural result is about -- across all 25 sessions rather than the 11
  that the behavioural reachability floor admits.

Split-half reliability of both observables is estimated by splitting units,
not trials: an observable computed from a random half of the units is
correlated across trials with the same observable from the disjoint other
half, and Spearman-Brown corrects the result back to full-population length.
An observable that does not survive that split cannot be orthogonal to
anything and cannot predict anything either.

The residual-existence question -- whether anything is left in the state once
the dominant rank-1 gain is removed -- is answered here with a null that has
the same component removed from every permutation replicate as from the
observed data. The version stored in results/state_latent_identity.json
subtracts the gain from the observed side only and is marked withdrawn as
mis-specified in results/rank1_gain_temporal_profile_closure.json; it is not
read, quoted or extended by this module.

Corpus: macaque lPFC (Panichello et al. 2024), all 25 sessions. Every
correlation is computed within one session; only per-session coefficients are
pooled, by the paired sign-flip test.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import binomtest, kruskal, norm

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from corpus_sessions import data_root  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402
from run_state_behavior_link import _counts_from_spikes, trial_amplitude_covariates  # noqa: E402
from run_state_latent_identity import session_rank1_and_residual  # noqa: E402
from state_persistence import (  # noqa: E402
    _permute_counts_independently_per_unit, residual_pair_correlations, slope_across_sessions_test,
)
from statistics import (  # noqa: E402
    bootstrap_ci, fdr_bh, minimum_detectable_paired_difference, spearman_permutation_test, stable_seed,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUTPUT_PATH = RESULTS_DIR / "state_orthogonality_census.json"
# The residual-structure arm's 100 same-pipeline permutation replicates per session dominate wall
# clock (order an hour pooled over 25 sessions, unit-count-dependent). Checkpointed per session so a
# kill mid-run -- this mandate has already been killed once by an environment quota reset -- resumes
# rather than restarts; a session present in the checkpoint is read back, never refit.
CHECKPOINT_PATH = RESULTS_DIR / ".checkpoints" / "state_orthogonality_census_checkpoint.json"

DECIDING_WIDTH_BINS = 3
BIN_WIDTH_S = 0.1
COMMON_RANGE_BINS = (3, 8)
N_UNIT_SPLIT_REPEATS = 10
N_RESIDUAL_NULL_REPLICATES = 100
N_PERM_CORRELATE = 2000
N_BOOT = 2000
SEED_TAG = "state_orthogonality_census"

# Two thresholds on the correlation scale, both fixed before any bound was computed.
# The tighter one is this project's existing commensurable behavioural scale: the minimum
# detectable paired difference of the persistence contrast in results/state_behavior_link.json
# (0.139 r units, reported there as ~0.14), already reused unchanged by
# results/rate_free_state_geometry_behavior_link.json. The looser one is the conventional
# boundary of a large correlation. A null whose minimum detectable effect exceeds the looser
# threshold could not have seen even a large association and excludes nothing.
MEANINGFUL_EFFECT_THRESHOLD_R_UNITS = 0.14
LARGE_ASSOCIATION_THRESHOLD_R_UNITS = 0.30

CONSTRAINS_AT_MEANINGFUL_SCALE = "constrains_at_the_meaningful_effect_scale"
CONSTRAINS_LARGE_ONLY = "constrains_only_large_associations"
EXCLUDES_NOTHING = "does_not_exclude_the_candidate"
NOT_ON_CORRELATION_SCALE = "reported_in_its_own_units_not_on_the_correlation_scale"

PRESENT_AND_DOMINANT = "present_and_dominant"
PRESENT_SMALL_AND_BOUNDED = "present_small_and_bounded"
NOT_DETECTABLY_PRESENT = "not_detectably_present_above_the_no_association_reference"

# Split-half reliability floor, fixed before fitting. An observable whose correlation with a
# disjoint half of the same population is below 0.5 after Spearman-Brown carries less than half
# its variance as signal; every correlation it enters is attenuated by more than sqrt(0.5), so
# neither an exclusion nor a positive prediction computed on it can be read at face value.
RELIABILITY_FLOOR = 0.50

BOUND_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "For each null bearing on the identity of the cross-unit maintenance state: take its point "
    "estimate and confidence interval as already computed, compute the minimum effect detectable at "
    "80% power from the observed between-session spread and the number of sessions actually "
    "available, and classify the exclusion on the correlation scale. Minimum detectable effect at or "
    f"below {MEANINGFUL_EFFECT_THRESHOLD_R_UNITS} r units: the null excludes any association above "
    "that size and is reported as constraining at the meaningful effect scale. Above that but at or "
    f"below {LARGE_ASSOCIATION_THRESHOLD_R_UNITS}: the null excludes only large associations and is "
    "reported as such, with the range it leaves open stated explicitly. Above "
    f"{LARGE_ASSOCIATION_THRESHOLD_R_UNITS}: the null could not have detected even a large "
    "association and is reported as excluding nothing, whatever its p-value. Tests whose statistic "
    "is not a correlation are classified separately and their bound is stated in their own units. "
    "Ordering is by minimum detectable effect ascending, most-constraining first; that ordering is "
    "part of the result and is not re-sorted by p-value or by point estimate."
)

SIGN_INVARIANCE_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "An observable defined as a singular vector has no fixed sign across sessions unless a "
    "convention imposes one. For every pooled test that averages SIGNED per-session associations of "
    "such an observable, two checks run before its bound is read. First, the per-session signs are "
    "tested against a fair coin by an exact binomial test: a split indistinguishable from chance "
    "means the pooled mean is near zero by construction and its confidence interval bounds nothing "
    "about the association's size. Second, the association is recomputed with a sign-invariant "
    "statistic (the absolute per-session correlation, against the value expected for an unrelated "
    "pair at that session's own trial count) and tested by the same paired sign-flip test. If the "
    "sign-invariant statistic is significantly above its reference, the candidate is NOT excluded -- "
    "it is present, and the signed pooled null was an artifact of the estimator. This is recorded as "
    "'candidate_association_is_present_not_null', named here as a possible outcome before fitting "
    "because it is neither an exclusion nor a power failure and must not be forced onto either."
)

RELIABILITY_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Split-half reliability is estimated by splitting UNITS, not trials: the observable is computed "
    "from a random half of the session's units and from the disjoint other half, over the same "
    "trials, and the two are correlated across trials. Spearman-Brown corrects the half-length "
    "correlation to full-population length. The per-session value is the median over "
    f"{N_UNIT_SPLIT_REPEATS} independent splits; the pooled value is the median across sessions with "
    "a session-resampled percentile interval. For an observable whose sign is set by a "
    "decomposition, the half-to-half correlation's own sign is arbitrary too, so its magnitude is "
    "used and the value expected for two unrelated halves at that trial count is reported beside it "
    "rather than left implicit. If the lower end of the pooled interval falls below "
    f"{RELIABILITY_FLOOR}, the observable is too unreliable to support either an exclusion or a "
    "prediction and that finding takes precedence over every bound in this census."
)

RESIDUAL_STRUCTURE_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Whether anything survives removal of the dominant rank-1 gain is tested with a null built by "
    "the same procedure as the observed value. Each permutation replicate permutes the trial axis "
    "independently per unit, then goes through the identical pipeline the observed side uses -- same "
    "trial half-split, leading latent fitted on the same half, window means at the same width, "
    "rank-1 component removed by the same decomposition -- and its residual pairwise correlation is "
    "scored at each lag. The contrast at each lag is the observed residual correlation minus the "
    "median over replicates, pooled across sessions by the paired sign-flip test and corrected for "
    "multiplicity across lags. A null that removes the component from only one side of the "
    "comparison is negative by construction and is not used, quoted or extended here."
)

MAGNITUDE_LADDER_RULE_DECLARED_BEFORE_FITTING = (
    "A candidate that is present is not thereby dominant, and the difference matters more than the "
    "presence: an association that accounts for most of an observable's variance identifies it, "
    "while one that accounts for a few percent leaves it nearly free of the candidate. Every "
    "association is therefore placed on one magnitude scale -- the median absolute per-session "
    "association, read against the value unrelated variables give at the same trial counts -- and "
    f"classified against the same large-association threshold of {LARGE_ASSOCIATION_THRESHOLD_R_UNITS} "
    "r units the bounds already use. At or above it the candidate is reported as present and "
    "dominant in that observable. Below it, and above the no-association reference, the candidate is "
    "reported as present, small and bounded, with the shared-variance ceiling its own interval "
    "leaves open stated in the same object; that is a quantitative near-orthogonality result with a "
    "number attached, not a null and not an orthogonality claim."
)

CENSUS_BRANCH_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Evaluated in this order. (1) If the split-half reliability interval of the behaviour-predicting "
    "observable has a lower end below the reliability floor, the branch is "
    "'observable_reliability_insufficient' regardless of every other result. (2) Otherwise, if any "
    "candidate identity turns out to be PRESENT rather than absent once tested with a sign-invariant "
    "statistic on the same observable, the branch is "
    "'candidate_association_is_present_not_null' and the orthogonality framing is not available for "
    "that candidate. (3) Otherwise, if no candidate's null reaches even the large-association "
    "threshold, the branch is 'nulls_do_not_constrain': the state's identity is untested, not "
    "orthogonal. (4) Otherwise, if every candidate tested on the behaviour-predicting observable is "
    "excluded at the meaningful effect scale and the behavioural prediction survives partialling, "
    "the branch is 'predictive_component_orthogonal_within_bounds'. (5) Otherwise the branch is "
    "'inconclusive_below_detection_floor'."
)


# ============================================================================
# Bound arithmetic
# ============================================================================

def _expected_absolute_correlation_under_no_association(n: int) -> float:
    """Mean |r| for two unrelated variables at ``n`` observations. Under no
    association r is approximately Normal(0, 1/(n-1)) for moderate n, so the
    folded mean is sqrt(2 / (pi * (n - 1))) -- the reference an absolute
    correlation must be compared against, since |r| is bounded below by zero
    and its expectation is therefore positive whatever the truth."""
    if n < 3:
        return float("nan")
    return float(np.sqrt(2.0 / (np.pi * (n - 1))))


def bounded_exclusion(
    name: str, candidate: str, observable: str, per_session_values, point_estimate: float,
    ci_lower: float | None, ci_upper: float | None, p_value: float | None, source: str,
    units: str = "correlation", scale_factor: float = 1.0, note: str | None = None,
) -> dict:
    """One null converted to a bounded exclusion: its estimate and interval as
    given, the minimum effect its own between-session spread and session count
    could have detected at 80% power, and the plain-language statement of what
    that does and does not rule out.

    ``scale_factor`` converts the test's own units to the correlation scale
    where a standard conversion exists (a macro one-versus-rest area under the
    curve maps to a rank-biserial correlation as 2 * (auc - 0.5), so its
    minimum detectable difference doubles). ``units`` other than "correlation"
    are classified separately and bounded only in their own units."""
    mdd = minimum_detectable_paired_difference(np.asarray(per_session_values, dtype=float))
    result = {
        "name": name, "candidate_identity": candidate, "state_observable": observable,
        "source_artifact": source, "n_sessions": int(len(per_session_values)),
        "point_estimate": float(point_estimate), "confidence_interval": [ci_lower, ci_upper],
        "p_value": p_value, "statistic_units": units,
        "minimum_detectable_effect_80pct_power": mdd,
    }
    if mdd.get("status") != "computed":
        result["constraining_power"] = "not_computable"
        result["bound_in_plain_language"] = (
            "The minimum detectable effect could not be computed for this test, so it carries no bound."
        )
        # None, not float("inf"): a bare Infinity token is not valid JSON.
        # Callers that need these to sort last use _sort_last_if_none() below.
        result["sort_key_minimum_detectable_effect_correlation_units"] = None
        return result

    mdd_own_units = float(mdd["mdd"])
    result["minimum_detectable_effect_in_own_units"] = mdd_own_units

    if units != "correlation":
        result["constraining_power"] = NOT_ON_CORRELATION_SCALE
        # None, not float("inf"): see the not_computable branch above.
        result["sort_key_minimum_detectable_effect_correlation_units"] = None
        result["bound_in_plain_language"] = (
            f"This test's statistic is {units}, which does not convert to a shared-variance fraction. "
            f"At {len(per_session_values)} sessions it could have detected a mean value of "
            f"{mdd_own_units:.4f} in those units with 80% probability; the observed value is "
            f"{point_estimate:.4f}. It bounds the effect only in its own units and is not comparable "
            "with the correlation-scale exclusions above it."
        )
        if note:
            result["note"] = note
        return result

    mdd_r = mdd_own_units * scale_factor
    result["minimum_detectable_effect_correlation_units"] = mdd_r
    result["sort_key_minimum_detectable_effect_correlation_units"] = mdd_r
    result["maximum_shared_variance_not_excluded"] = float(min(1.0, mdd_r ** 2))

    if mdd_r <= MEANINGFUL_EFFECT_THRESHOLD_R_UNITS:
        result["constraining_power"] = CONSTRAINS_AT_MEANINGFUL_SCALE
        result["bound_in_plain_language"] = (
            f"At {len(per_session_values)} sessions this test would have detected a mean per-session "
            f"association of {mdd_r:.3f} or larger with 80% probability. It therefore excludes "
            f"{candidate} as an identity for {observable} at any strength above {mdd_r:.3f}, i.e. "
            f"above {100 * mdd_r ** 2:.1f}% shared variance on average across sessions. Associations "
            f"weaker than {mdd_r:.3f} remain compatible with it."
        )
    elif mdd_r <= LARGE_ASSOCIATION_THRESHOLD_R_UNITS:
        result["constraining_power"] = CONSTRAINS_LARGE_ONLY
        result["bound_in_plain_language"] = (
            f"At {len(per_session_values)} sessions this test could only have detected a mean "
            f"per-session association of {mdd_r:.3f} or larger. It excludes {candidate} as a LARGE "
            f"identity for {observable} but leaves the whole range up to {mdd_r:.3f} "
            f"({100 * mdd_r ** 2:.1f}% shared variance) open. It is not evidence that the two are "
            "unrelated."
        )
    else:
        result["constraining_power"] = EXCLUDES_NOTHING
        result["bound_in_plain_language"] = (
            f"At {len(per_session_values)} sessions this test could not have detected a mean "
            f"per-session association below {mdd_r:.3f} -- larger than a conventionally large "
            f"correlation. It does not exclude {candidate} as an identity for {observable} at any "
            "strength the data could distinguish, and must not be cited as evidence of orthogonality."
        )
    if note:
        result["note"] = note
    return result


def _sort_last_if_none(entry: dict) -> tuple[bool, float]:
    """Sort key for a list of ``bounded_exclusion()`` results that puts the
    ones with no sort value (not computable, or not on the correlation
    scale) after every one that has a real value, without ever needing
    float("inf") -- which is not valid JSON -- as a stored field."""
    value = entry["sort_key_minimum_detectable_effect_correlation_units"]
    return (value is None, value if value is not None else 0.0)


# ============================================================================
# Sign-invariance audit
# ============================================================================

def sign_invariance_audit(per_session_rho, per_session_n, candidate: str, observable: str) -> dict:
    """Whether a pooled test that averages SIGNED per-session associations of a
    sign-arbitrary observable is measuring the association or measuring the
    sign convention: the per-session signs against a fair coin, and the same
    association recomputed without its sign against the value two unrelated
    variables would give at each session's own trial count."""
    rho = np.asarray(per_session_rho, dtype=float)
    n_obs = np.asarray(per_session_n, dtype=float)
    n_positive = int(np.sum(rho > 0))
    n_sessions = int(len(rho))
    sign_test = binomtest(n_positive, n_sessions, 0.5, alternative="two-sided")

    reference = np.array([_expected_absolute_correlation_under_no_association(int(n)) for n in n_obs])
    magnitude_test = slope_across_sessions_test((np.abs(rho) - reference).tolist(), "two-sided")

    signs_uninformative = bool(sign_test.pvalue > 0.05)
    magnitude_present = bool(magnitude_test.get("significant_positive", False))
    if magnitude_present:
        verdict = "candidate_association_is_present_not_null"
    elif signs_uninformative:
        verdict = "signed_pooled_test_uninterpretable_and_magnitude_not_above_reference"
    else:
        verdict = "signs_consistent_across_sessions_signed_pooled_test_is_interpretable"

    return {
        "candidate_identity": candidate, "state_observable": observable, "n_sessions": n_sessions,
        "n_sessions_positive": n_positive, "n_sessions_negative": n_sessions - n_positive,
        "sign_split_vs_fair_coin_p_value": float(sign_test.pvalue),
        "signed_pooled_mean": float(np.mean(rho)),
        "median_absolute_association": float(np.median(np.abs(rho))),
        "mean_absolute_association": float(np.mean(np.abs(rho))),
        "median_shared_variance_per_session": float(np.median(rho ** 2)),
        "median_reference_absolute_association_under_no_association": float(np.median(reference)),
        "magnitude_above_reference_test": magnitude_test,
        "verdict": verdict,
        "verdict_in_plain_language": (
            f"The per-session association between {observable} and {candidate} has median magnitude "
            f"{np.median(np.abs(rho)):.3f} ({100 * np.median(rho ** 2):.1f}% shared variance in the "
            f"median session) against {np.median(reference):.3f} expected for unrelated variables at "
            f"these trial counts, while the signed pooled mean is {np.mean(rho):+.4f}. "
            + (
                f"The signs split {n_positive} positive to {n_sessions - n_positive} negative, "
                f"indistinguishable from coin flips (p={sign_test.pvalue:.3f}), so the signed pooled "
                "mean is near zero by construction and bounds nothing. "
                if signs_uninformative else
                f"The signs split {n_positive} positive to {n_sessions - n_positive} negative, which "
                f"is not a coin flip (p={sign_test.pvalue:.3f}), so the signed pooled mean is "
                "interpretable. "
            )
            + (
                f"{candidate} is PRESENT in {observable}, not excluded from it."
                if magnitude_present else
                f"The sign-invariant magnitude is not above its reference either, so {candidate} is "
                f"not detectably present in {observable}."
            )
        ),
    }


# ============================================================================
# Per-session observables and their identity candidates
# ============================================================================

def session_observables(path: Path) -> dict | None:
    """Both state observables for one session, over the same trials, with the
    per-trial covariates every identity candidate needs.

    ``gain`` is the leading-component score of the centred trials x windows
    matrix -- the observable this project's identity nulls were computed on.
    ``deviation`` is one minus the cosine between the trial's L2-normalised
    per-unit activity vector and the leave-one-out mean direction of every
    other trial -- the observable the behavioural prediction is about. Both
    are computed on every trial of the session, correct and error alike, so
    the two are on identical trial sets and the covariates are defined for
    every trial."""
    raw = loadmat(str(path), simplify_cells=True)
    spikes = np.asarray(raw["spks"], dtype=float)
    time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
    counts = _counts_from_spikes(spikes, time_ms)
    if counts.shape[0] < 16 or counts.shape[1] < 4:
        return None
    covariates = trial_amplitude_covariates(counts)
    if covariates.get("status") != "computed":
        return None
    deviation = rate_free_state_deviation(counts.sum(axis=2))
    finite = np.isfinite(deviation)
    if finite.sum() < 16:
        return None
    return {
        "counts": counts,
        "gain": np.asarray(covariates["leading_component_score_gain"], dtype=float),
        "deviation": deviation,
        "finite_deviation": finite,
        "total_spike_count": np.asarray(covariates["total_spike_count"], dtype=float),
        "trial_index": np.asarray(covariates["trial_index"], dtype=float),
        "item_label": np.asarray(raw["cueAngIdx"]).reshape(-1),
        "is_correct": np.asarray(raw["isCorr"]).astype(bool).reshape(-1),
        "n_trials": int(counts.shape[0]), "n_units": int(counts.shape[1]),
    }


def _label_variance_share(values: np.ndarray, labels: np.ndarray) -> dict:
    """Share of the observable's rank variance attributable to the item label,
    as the epsilon-squared of a Kruskal-Wallis test: (H - k + 1) / (n - k).
    Sign-invariant by construction (it is an omnibus spread-between-classes
    statistic, not a directional score), centred on zero when the label is
    unrelated to the observable, and directly readable as a shared-variance
    fraction -- unlike an area under the curve computed from a scalar score,
    whose value flips to its complement when the score's sign flips."""
    classes = [values[labels == c] for c in np.unique(labels)]
    classes = [c for c in classes if len(c) >= 2]
    k, n = len(classes), int(sum(len(c) for c in classes))
    if k < 2 or n <= k:
        return {"status": "not_computable", "n_classes": k, "n_trials": n}
    statistic, p_value = kruskal(*classes)
    return {
        "status": "computed", "n_classes": k, "n_trials": n,
        "kruskal_h": float(statistic), "p_value": float(p_value),
        "variance_share": float((statistic - k + 1) / (n - k)),
    }


def session_identity_candidates(observables: dict, seed: int) -> dict:
    """Each candidate identity tested against each state observable within one
    session. Rank correlations throughout, so a monotone but non-linear
    relation is not missed and no candidate is advantaged by the choice."""
    rng = np.random.default_rng(seed)
    finite = observables["finite_deviation"]
    out = {}
    for observable_name, values, mask in (
        ("leading_component_gain", observables["gain"], np.ones(observables["n_trials"], dtype=bool)),
        ("rate_free_direction_deviation", observables["deviation"], finite),
    ):
        v = values[mask]
        row = {"n_trials": int(mask.sum())}
        for candidate, covariate in (
            ("total_spike_count", observables["total_spike_count"][mask]),
            ("trial_index", observables["trial_index"][mask]),
        ):
            test = spearman_permutation_test(v, covariate, n_perm=N_PERM_CORRELATE, rng=rng)
            row[candidate] = {k: val for k, val in test.items() if k != "null"}
        row["item_label"] = _label_variance_share(v, observables["item_label"][mask])
        out[observable_name] = row
    return out


# ============================================================================
# Split-half reliability, by unit split
# ============================================================================

def _split_half_reliability(observable_fn, counts: np.ndarray, rng: np.random.Generator,
                             sign_arbitrary: bool) -> dict:
    n_units = counts.shape[1]
    half_correlations, sign_agreements = [], []
    for _ in range(N_UNIT_SPLIT_REPEATS):
        order = rng.permutation(n_units)
        first, second = order[: n_units // 2], order[n_units // 2:]
        a, b = observable_fn(counts[:, first, :]), observable_fn(counts[:, second, :])
        if a is None or b is None:
            continue
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 16 or np.std(a[valid]) == 0 or np.std(b[valid]) == 0:
            continue
        r = float(np.corrcoef(a[valid], b[valid])[0, 1])
        sign_agreements.append(r > 0)
        half_correlations.append(abs(r) if sign_arbitrary else r)
    if not half_correlations:
        return {"status": "not_computable"}
    half = float(np.median(half_correlations))
    return {
        "status": "computed", "n_splits": len(half_correlations),
        "n_units": int(n_units),
        "median_half_to_half_correlation": half,
        "spearman_brown_full_length_reliability": float(2.0 * half / (1.0 + half)) if half > -1 else float("nan"),
        "expected_half_to_half_correlation_if_unrelated": _expected_absolute_correlation_under_no_association(
            int(counts.shape[0])) if sign_arbitrary else 0.0,
        "fraction_of_splits_with_positive_correlation": float(np.mean(sign_agreements)),
        "sign_treated_as_arbitrary": bool(sign_arbitrary),
    }


def session_reliability(observables: dict, seed: int) -> dict:
    counts = observables["counts"]

    def _gain(subset: np.ndarray):
        result = trial_amplitude_covariates(subset)
        if result.get("status") != "computed":
            return None
        return np.asarray(result["leading_component_score_gain"], dtype=float)

    def _deviation(subset: np.ndarray):
        return rate_free_state_deviation(subset.sum(axis=2))

    return {
        "leading_component_gain": _split_half_reliability(
            _gain, counts, np.random.default_rng(seed), sign_arbitrary=True),
        "rate_free_direction_deviation": _split_half_reliability(
            _deviation, counts, np.random.default_rng(seed + 1), sign_arbitrary=False),
    }


# ============================================================================
# Residual structure after the dominant gain, with a symmetric null
# ============================================================================

def session_residual_structure(counts: np.ndarray, seed: int) -> dict | None:
    """Residual pairwise correlation at each lag once the rank-1 gain is
    removed, minus the same quantity computed on permutation replicates that
    have been through the identical removal. The observed side and every
    replicate share one trial half-split (the same integer seed), so the only
    difference between them is the cross-unit covariance the permutation
    destroys."""
    observed = session_rank1_and_residual(counts, DECIDING_WIDTH_BINS, seed)
    if observed is None:
        return None
    observed_by_lag = residual_pair_correlations(observed["residual"], DECIDING_WIDTH_BINS)
    rng = np.random.default_rng(seed)
    null_by_lag: dict[int, list[float]] = {}
    n_fitted = 0
    for _ in range(N_RESIDUAL_NULL_REPLICATES):
        permuted = _permute_counts_independently_per_unit(counts, rng)
        replicate = session_rank1_and_residual(permuted, DECIDING_WIDTH_BINS, seed)
        if replicate is None:
            continue
        n_fitted += 1
        for lag, value in residual_pair_correlations(replicate["residual"], DECIDING_WIDTH_BINS).items():
            null_by_lag.setdefault(lag, []).append(value)
    low, high = COMMON_RANGE_BINS
    contrast = {
        lag: float(observed_by_lag[lag] - np.median(null_by_lag[lag]))
        for lag in observed_by_lag if lag in null_by_lag and low <= lag <= high
    }
    return {
        "n_replicates_requested": N_RESIDUAL_NULL_REPLICATES, "n_replicates_fitted": n_fitted,
        "observed_residual_correlation_by_lag": {str(k): float(v) for k, v in observed_by_lag.items()},
        "null_median_residual_correlation_by_lag": {
            str(k): float(np.median(v)) for k, v in null_by_lag.items()},
        "contrast_by_lag": {str(k): v for k, v in contrast.items()},
        "rank1_variance_share": observed["rank1"]["observed_share"],
    }


def pool_residual_structure(per_session: list[dict]) -> dict:
    lags = sorted({int(lag) for row in per_session for lag in row["contrast_by_lag"]})
    by_lag, p_values, order = {}, [], []
    for lag in lags:
        values = [row["contrast_by_lag"][str(lag)] for row in per_session if str(lag) in row["contrast_by_lag"]]
        test = slope_across_sessions_test(values, "two-sided")
        test["minimum_detectable_effect_80pct_power"] = minimum_detectable_paired_difference(values)
        by_lag[str(lag)] = test
        if test.get("status") == "tested":
            p_values.append(test["two_sided_p_value"])
            order.append(lag)
    if p_values:
        fdr = fdr_bh(np.array(p_values), alpha=0.05)
        for lag, q, reject in zip(order, fdr["q_values"], fdr["reject"]):
            by_lag[str(lag)]["fdr_q_value"] = float(q)
            by_lag[str(lag)]["fdr_significant"] = bool(reject)
    n_clearing = sum(1 for lag in order if by_lag[str(lag)].get("fdr_significant"))
    n_positive = sum(
        1 for lag in order
        if by_lag[str(lag)].get("fdr_significant") and by_lag[str(lag)]["mean_value"] > 0)
    return {
        "n_sessions": len(per_session), "by_lag": by_lag,
        "n_lags_tested": len(order), "n_lags_clearing_multiplicity_correction": n_clearing,
        "n_lags_clearing_with_a_positive_mean": n_positive,
        "verdict": (
            "structure_survives_removal_of_the_dominant_gain" if n_positive > 0 else
            "nothing_detectable_survives_removal_of_the_dominant_gain"
        ),
    }


# ============================================================================
# Assembly
# ============================================================================

def _pooled_from_values(values: list[float]) -> dict:
    test = slope_across_sessions_test(values, "two-sided")
    return test


def exclusions_from_stored_artifacts() -> list[dict]:
    """Every null already on disk that bears on what the state is, converted
    to a bounded exclusion. Values are read as stored; nothing is refitted."""
    identity = json.loads((RESULTS_DIR / "state_latent_identity.json").read_text())
    macaque = identity["per_corpus"]["panichello_lpfc"]
    rows = macaque["session_rows"]
    rate_free = json.loads((RESULTS_DIR / "rate_free_state_geometry_behavior_link.json").read_text())
    amplitude = json.loads((RESULTS_DIR / "behavior_amplitude_rate_controls.json").read_text())
    behavior = json.loads((RESULTS_DIR / "state_behavior_link.json").read_text())
    content = json.loads((RESULTS_DIR / "state_content_link.json").read_text())

    spike_rho = [r["gain_correlates"]["total_spike_count"]["rho"] for r in rows]
    trial_rho = [r["gain_correlates"]["trial_index"]["rho"] for r in rows]
    label_auc = [r["gain_correlates"]["item_label"]["macro_auc"] for r in rows]
    position_coefficient = [r["position_vs_lag"]["joint_regression"]["coef_x2"] for r in rows]

    spike_test = macaque["gain_correlates"]["total_spike_count"]["pooled_rho_test"]
    trial_test = macaque["gain_correlates"]["trial_index"]["pooled_rho_test"]
    position_test = macaque["position_vs_lag_decomposition"]["position_coefficient_test"]

    gate_values = [s["analysis"]["orthogonality_gate"]["r"] for s in rate_free["sessions"]
                   if s.get("status") == "computed"]
    gate_test = rate_free["pooled"]["orthogonality_gate_deviation_vs_spike_count"]

    gain_given_rate = [s["analysis"]["control_spike_count"]["gain_given_spike_count"]["r"]
                       for s in amplitude["sessions"] if s.get("status") == "computed"]
    gain_given_rate_test = amplitude["pooled"]["gain_given_spike_count"]

    persistence_values = behavior["deciding_contrast"]["differences_matched_correct_minus_error"]
    persistence_test = behavior["deciding_contrast"]["test"]

    content_rows = [r for r in content["session_rows"] if r["dataset"] == "panichello_2024"]
    content_values = [
        r["trial_resolved"]["high_margin_d_perm"]["median"] - r["trial_resolved"]["low_margin_d_perm"]["median"]
        for r in content_rows
        if r.get("trial_resolved", {}).get("status") == "tested"
        and r["trial_resolved"]["high_margin_d_perm"].get("status") == "tested"
        and r["trial_resolved"]["low_margin_d_perm"].get("status") == "tested"
    ]
    content_test = content["per_corpus"]["panichello_2024"]["trial_resolved_diff_test"]

    return [
        bounded_exclusion(
            "leading_component_gain_versus_total_spike_count", "total spike count",
            "the leading-component gain", spike_rho, spike_test["mean_value"],
            spike_test["ci_lower"], spike_test["ci_upper"], spike_test["two_sided_p_value"],
            "results/state_latent_identity.json",
            note="Pools SIGNED per-session rank correlations of a singular-vector observable; see the "
                  "sign-invariance audit before reading this bound."),
        bounded_exclusion(
            "leading_component_gain_versus_trial_index", "position in the session",
            "the leading-component gain", trial_rho, trial_test["mean_value"],
            trial_test["ci_lower"], trial_test["ci_upper"], trial_test["two_sided_p_value"],
            "results/state_latent_identity.json",
            note="Pools SIGNED per-session rank correlations of a singular-vector observable; see the "
                  "sign-invariance audit before reading this bound."),
        bounded_exclusion(
            "leading_component_gain_versus_item_label", "the remembered item",
            "the leading-component gain", [a - 0.5 for a in label_auc],
            float(np.median(label_auc)) - 0.5, None, None, None,
            "results/state_latent_identity.json", scale_factor=2.0,
            note="Stated as a macro one-versus-rest area under the curve minus chance; the minimum "
                  "detectable effect is converted to the correlation scale as twice that difference, "
                  "the rank-biserial equivalent. An area under the curve computed from a signed scalar "
                  "score inherits that score's sign ambiguity, so the sign-invariant recomputation is "
                  "the one to read."),
        bounded_exclusion(
            "window_pair_correlation_versus_position_in_epoch", "where in the delay a window pair sits",
            "the pairwise window correlation of the lag census", position_coefficient,
            position_test["mean_value"], position_test["ci_lower"], position_test["ci_upper"],
            position_test["two_sided_p_value"], "results/state_latent_identity.json",
            units="correlation per second of position within the epoch"),
        bounded_exclusion(
            "rate_free_direction_deviation_versus_total_spike_count", "total spike count",
            "the rate-free direction deviation", gate_values, gate_test["mean_value"],
            gate_test["ci_lower"], gate_test["ci_upper"], gate_test["two_sided_p_value"],
            "results/rate_free_state_geometry_behavior_link.json",
            note="Restricted to the sessions the behavioural reachability floor admits; the same test "
                  "at every session of the corpus is in the same-observable census below."),
        bounded_exclusion(
            "leading_component_gain_versus_outcome_given_spike_count", "trial outcome beyond firing rate",
            "the leading-component gain", gain_given_rate, gain_given_rate_test["mean_value"],
            gain_given_rate_test["ci_lower"], gain_given_rate_test["ci_upper"],
            gain_given_rate_test["two_sided_p_value"], "results/behavior_amplitude_rate_controls.json"),
        bounded_exclusion(
            "state_persistence_magnitude_correct_versus_error", "trial outcome",
            "the state's persistence magnitude across lags", persistence_values,
            persistence_test["mean_value"], persistence_test["ci_lower"], persistence_test["ci_upper"],
            persistence_test["two_sided_p_value"], "results/state_behavior_link.json"),
        bounded_exclusion(
            "state_persistence_versus_content_decodability", "how well the item is decodable on the trial",
            "the state's persistence magnitude across lags", content_values,
            content_test["mean_value"], None, None, content_test["two_sided_p_value"],
            "results/state_content_link.json",
            units="difference in permutation-referenced persistence between high- and low-margin trials"),
    ]


def same_observable_census(per_session_candidates: list[dict], per_session_n: list[int]) -> dict:
    """Every candidate identity, on both observables, over every session of the
    corpus -- the arm that puts the exclusions and the behavioural prediction
    on one observable instead of three."""
    out = {}
    for observable in ("leading_component_gain", "rate_free_direction_deviation"):
        sign_arbitrary = observable == "leading_component_gain"
        rows = [c[observable] for c in per_session_candidates]
        block = {}
        for candidate, label in (("total_spike_count", "total spike count"),
                                  ("trial_index", "position in the session")):
            values = [r[candidate]["rho"] for r in rows]
            pooled = _pooled_from_values(values)
            entry = {
                "pooled_signed_test": pooled,
                "bounded_exclusion": bounded_exclusion(
                    f"{observable}_versus_{candidate}", label,
                    f"the {observable.replace('_', ' ')}", values, pooled["mean_value"],
                    pooled["ci_lower"], pooled["ci_upper"], pooled["two_sided_p_value"],
                    "results/state_orthogonality_census.json"),
                "sign_invariance_audit": sign_invariance_audit(
                    values, per_session_n, label, f"the {observable.replace('_', ' ')}"),
            }
            block[candidate] = entry
        label_shares = [r["item_label"]["variance_share"] for r in rows
                        if r["item_label"].get("status") == "computed"]
        label_pooled = _pooled_from_values(label_shares)
        block["item_label"] = {
            "pooled_variance_share_test": label_pooled,
            "median_variance_share": float(np.median(label_shares)),
            "bounded_exclusion": bounded_exclusion(
                f"{observable}_versus_item_label", "the remembered item",
                f"the {observable.replace('_', ' ')}", label_shares, label_pooled["mean_value"],
                label_pooled["ci_lower"], label_pooled["ci_upper"], label_pooled["two_sided_p_value"],
                "results/state_orthogonality_census.json",
                units="share of the observable's rank variance attributable to the item label"),
            "statistic_note": (
                "Sign-invariant by construction and centred on zero when the label is unrelated to "
                "the observable, so no sign-invariance audit is needed and the pooled mean is "
                "readable directly as a shared-variance fraction."
            ),
        }
        out[observable] = block
        out[observable]["sign_of_this_observable_is_set_by_a_decomposition"] = sign_arbitrary
    return out


def association_magnitude_ladder(census: dict) -> dict:
    """Every association in the census on one magnitude scale, largest first.

    Presence and dominance are different findings and the census's own verdict
    field records only presence, so the two observables end up carrying the
    same label for associations that differ by more than an order of magnitude
    in shared variance. This ladder separates them and attaches, to each small
    association, the shared-variance ceiling its own interval leaves open.
    """
    rungs = []
    for observable, block in census.items():
        for candidate, entry in block.items():
            if not isinstance(entry, dict) or "sign_invariance_audit" not in entry:
                continue
            audit = entry["sign_invariance_audit"]
            magnitude = audit["magnitude_above_reference_test"]
            median_abs = audit["median_absolute_association"]
            reference = audit["median_reference_absolute_association_under_no_association"]
            present = audit["verdict"] == "candidate_association_is_present_not_null"
            if not present:
                classification = NOT_DETECTABLY_PRESENT
            elif median_abs >= LARGE_ASSOCIATION_THRESHOLD_R_UNITS:
                classification = PRESENT_AND_DOMINANT
            else:
                classification = PRESENT_SMALL_AND_BOUNDED
            ceiling_upper = magnitude.get("ci_upper")
            ceiling = (float(min(1.0, ceiling_upper + reference) ** 2)
                       if ceiling_upper is not None else None)
            if classification == PRESENT_AND_DOMINANT:
                plain = (
                    f"{audit['candidate_identity'].capitalize()} accounts for "
                    f"{100 * audit['median_shared_variance_per_session']:.1f}% of "
                    f"{audit['state_observable']}'s variance in the median session, at median "
                    f"magnitude {median_abs:.3f} against {reference:.3f} for unrelated variables. "
                    "That is an identification, not a residual association: the observable is "
                    "largely a restatement of the candidate and cannot be described as orthogonal "
                    "to it in any useful sense."
                )
            elif classification == PRESENT_SMALL_AND_BOUNDED:
                plain = (
                    f"{audit['candidate_identity'].capitalize()} is detectable in "
                    f"{audit['state_observable']} but small: median magnitude {median_abs:.3f} "
                    f"against {reference:.3f} for unrelated variables, "
                    f"{100 * audit['median_shared_variance_per_session']:.1f}% of the observable's "
                    "variance in the median session"
                    + (f", with the upper end of the pooled magnitude interval leaving at most "
                       f"{100 * ceiling:.1f}% shared. " if ceiling is not None else ". ")
                    + "The observable is therefore close to free of the candidate, quantitatively "
                    "and with a ceiling attached, rather than either orthogonal to it or "
                    "unrelated to it."
                )
            else:
                plain = (
                    f"{audit['candidate_identity'].capitalize()} is not detectable above the "
                    f"no-association reference in {audit['state_observable']} at this session count."
                )
            rungs.append({
                "state_observable": audit["state_observable"],
                "candidate_identity": audit["candidate_identity"],
                "observable_key": observable, "candidate_key": candidate,
                "median_absolute_association": median_abs,
                "median_shared_variance_per_session": audit["median_shared_variance_per_session"],
                "reference_absolute_association_under_no_association": reference,
                "pooled_magnitude_above_reference": magnitude.get("mean_value"),
                "pooled_magnitude_ci95": [magnitude.get("ci_lower"), magnitude.get("ci_upper")],
                "pooled_magnitude_p_value": magnitude.get("two_sided_p_value"),
                "signed_pooled_mean": audit["signed_pooled_mean"],
                "shared_variance_ceiling_left_open": ceiling,
                "classification": classification,
                "in_plain_language": plain,
            })
    rungs.sort(key=lambda r: -r["median_absolute_association"])
    dominant = [r for r in rungs if r["classification"] == PRESENT_AND_DOMINANT]
    small = [r for r in rungs if r["classification"] == PRESENT_SMALL_AND_BOUNDED]
    return {
        "rule_declared_before_fitting": MAGNITUDE_LADDER_RULE_DECLARED_BEFORE_FITTING,
        "large_association_threshold_r_units": LARGE_ASSOCIATION_THRESHOLD_R_UNITS,
        "rungs_sorted_largest_first": rungs,
        "n_present_and_dominant": len(dominant),
        "n_present_small_and_bounded": len(small),
        "reading": (
            "Presence is not one finding. "
            + "; ".join(f"{r['candidate_identity']} in {r['state_observable']} at median magnitude "
                        f"{r['median_absolute_association']:.3f} "
                        f"({100 * r['median_shared_variance_per_session']:.1f}% shared)"
                        for r in rungs if r["classification"] != NOT_DETECTABLY_PRESENT)
            + f". The gap between the largest and the smallest of these is a factor of "
              f"{rungs[0]['median_absolute_association'] / max(min(r['median_absolute_association'] for r in rungs if r['classification'] != NOT_DETECTABLY_PRESENT), 1e-12):.1f} "
              "in magnitude, so a single presence label across both observables would report the "
              "same finding for an identification and for a few percent of shared variance."
        ),
    }


def dominant_mode_identity(census: dict, ladder: dict) -> dict:
    """What the dominant cross-unit mode is, stated positively.

    Every earlier test of this observable's identity pooled SIGNED per-session
    correlations. The leading component's sign is set by a decomposition and is
    arbitrary per session, so that pooling averages a near-deterministic
    relationship toward zero and returns a null. Discarding the sign returns
    the relationship.
    """
    gain = census["leading_component_gain"]
    rate = gain["total_spike_count"]["sign_invariance_audit"]
    drift = gain["trial_index"]["sign_invariance_audit"]
    drift_signed = gain["trial_index"]["pooled_signed_test"]
    return {
        "state_observable": "the leading component gain",
        "identified_as": (
            "the trial's total spike count, up to an arbitrary per-session sign"),
        "association_with_total_spike_count": {
            "median_absolute_association": rate["median_absolute_association"],
            "median_shared_variance_per_session": rate["median_shared_variance_per_session"],
            "reference_absolute_association_under_no_association":
                rate["median_reference_absolute_association_under_no_association"],
            "pooled_magnitude_above_reference": rate["magnitude_above_reference_test"]["mean_value"],
            "pooled_magnitude_ci95": [rate["magnitude_above_reference_test"]["ci_lower"],
                                      rate["magnitude_above_reference_test"]["ci_upper"]],
            "pooled_magnitude_p_value": rate["magnitude_above_reference_test"]["two_sided_p_value"],
            "signed_pooled_mean": rate["signed_pooled_mean"],
            "n_sessions_positive": rate["n_sessions_positive"],
            "n_sessions_negative": rate["n_sessions_negative"],
            "sign_split_vs_fair_coin_p_value": rate["sign_split_vs_fair_coin_p_value"],
            "n_sessions": rate["n_sessions"],
        },
        "association_with_position_in_the_session": {
            "median_absolute_association": drift["median_absolute_association"],
            "median_shared_variance_per_session": drift["median_shared_variance_per_session"],
            "signed_pooled_mean": drift_signed["mean_value"],
            "signed_pooled_ci95": [drift_signed["ci_lower"], drift_signed["ci_upper"]],
            "signed_pooled_p_value": drift_signed["two_sided_p_value"],
            "n_sessions_positive": drift["n_sessions_positive"],
            "n_sessions_negative": drift["n_sessions_negative"],
            "sign_split_vs_fair_coin_p_value": drift["sign_split_vs_fair_coin_p_value"],
            "significant_on_the_signed_test_as_well": bool(drift_signed.get("significant", False)),
        },
        "why_this_was_previously_reported_as_a_null": (
            "The leading component is a singular vector and its sign is whatever the decomposition "
            "returns, independently in each session. Pooling SIGNED per-session correlations of such "
            "an observable averages a near-deterministic relationship toward zero: here the signs "
            f"split {rate['n_sessions_positive']} positive to {rate['n_sessions_negative']} negative "
            f"(p={rate['sign_split_vs_fair_coin_p_value']:.3f} against a fair coin), which turns a "
            f"median magnitude of {rate['median_absolute_association']:.3f} into a signed pooled mean "
            f"of {rate['signed_pooled_mean']:+.4f} with an interval that spans zero. The null was a "
            "property of the pooling, not of the data. Position in the session is not sign-arbitrary "
            "in the same way and survives even the signed test."
        ),
        "in_plain_language": (
            f"Up to an arbitrary per-session sign, the leading component gain is essentially the "
            f"trial's total spike count: median absolute association "
            f"{rate['median_absolute_association']:.3f}, "
            f"{100 * rate['median_shared_variance_per_session']:.0f}% of the observable's variance "
            f"shared in the median session, against "
            f"{rate['median_reference_absolute_association_under_no_association']:.3f} for unrelated "
            f"variables at these trial counts. It also drifts with position in the session "
            f"(median magnitude {drift['median_absolute_association']:.3f}, "
            f"{100 * drift['median_shared_variance_per_session']:.0f}% shared, and "
            f"{drift_signed['mean_value']:+.3f} with p={drift_signed['two_sided_p_value']:.4f} even on "
            "the signed test). The dominant cross-unit mode of this corpus therefore has a measured "
            "identity rather than an unknown one, and it is a rate mode."
        ),
        "what_this_does_not_say": (
            "It says nothing about the rate-free direction deviation, which is a different "
            "observable, shares a few percent of its variance with the same two variables, and is "
            "the one the behavioural prediction is computed on. The magnitude ladder holds both."
        ),
        "magnitude_ladder_position": next(
            (r for r in ladder["rungs_sorted_largest_first"]
             if r["observable_key"] == "leading_component_gain"
             and r["candidate_key"] == "total_spike_count"), None),
    }


def rate_free_observable_corroboration(census: dict) -> dict:
    """The same rate-free construction, measured in a second primate corpus.

    The multi-object corpus fits the identical observable and tests it against
    total spike count as a pre-declared gate before its behavioural arm runs,
    so its gate is a second estimate of the same quantity in independent data.
    """
    audit = census["rate_free_direction_deviation"]["total_spike_count"]["sign_invariance_audit"]
    path = RESULTS_DIR / "watters_state_geometry.json"
    entry = {
        "corpus": (
            "Multi-object spatial working memory in macaque frontal cortex, two animals, fixed "
            "1.0 s maintenance delay, continuous saccadic report: 'Working Memory of Multi-Object "
            "Scenes in Primate Frontal Cortex', Watters, Gabel, Tenenbaum and Jazayeri, bioRxiv "
            "preprint posted 2026-01-27, DOI 10.64898/2026.01.27.702062, data DANDI 000620. An "
            "unreviewed preprint, cited on that basis."
        ),
        "source_artifact": "results/watters_state_geometry.json",
        "shared_construction": (
            "The same rate-free direction deviation: each trial's per-unit spike counts over the "
            "delay epoch, L2-normalised to unit length so that overall rate is divided out, and the "
            "trial's angular deviation from the session's mean direction. Tested against total spike "
            "count by the same paired sign-flip test over per-session correlations."
        ),
        "this_corpus": {
            "corpus": "macaque lPFC (Panichello et al. 2024)",
            "n_sessions": audit["n_sessions"],
            "median_absolute_association": audit["median_absolute_association"],
            "median_shared_variance_per_session": audit["median_shared_variance_per_session"],
            "signed_pooled_mean": audit["signed_pooled_mean"],
        },
    }
    if not path.exists():
        entry["status"] = "corroborating_artifact_not_on_disk"
        return entry
    other = json.loads(path.read_text())
    arms = {}
    for arm in ("single_and_multi_unit", "good_single_units_only"):
        gate = (other.get("results", {}).get(arm, {}).get("pooled", {})
                .get("behaviour", {}).get("orthogonality_gate_state_deviation_vs_spike_count"))
        if gate is None:
            continue
        arms[arm] = {
            "n_sessions": gate.get("n_sessions"),
            "pooled_signed_mean": gate.get("mean_value"),
            "ci95": [gate.get("ci_lower"), gate.get("ci_upper")],
            "p_value": gate.get("two_sided_p_value"),
            "significant": gate.get("significant"),
            "minimum_detectable_paired_difference_at_80pct_power":
                gate.get("minimum_detectable_paired_difference_at_80pct_power", {}).get("mdd"),
        }
    entry["status"] = "read"
    entry["pooled_single_and_multi_unit_arm"] = arms.get("single_and_multi_unit")
    entry["good_single_units_only_arm"] = arms.get("good_single_units_only")
    pooled = arms.get("single_and_multi_unit") or {}
    single = arms.get("good_single_units_only") or {}
    entry["agreement"] = (
        f"In its pooled single-and-multi-unit arm that corpus obtains "
        f"{pooled.get('pooled_signed_mean', float('nan')):+.4f} with p="
        f"{pooled.get('p_value', float('nan')):.3f} over {pooled.get('n_sessions')} sessions, and "
        f"could have detected a difference of {pooled.get('minimum_detectable_paired_difference_at_80pct_power', float('nan')):.3f} "
        "at 80% power, so its null is a powered one. Two corpora, two estimates of the same "
        "construction, agreeing that the rate-free observable is close to rate-free."
    )
    entry["where_the_agreement_fails"] = (
        f"The same gate fails in that corpus's good-single-units-only arm: "
        f"{single.get('pooled_signed_mean', float('nan')):+.4f} with p="
        f"{single.get('p_value', float('nan')):.4f} over {single.get('n_sessions')} sessions, a "
        "significant negative association. The rate-free construction is therefore not robust to "
        "unit-quality selection in that corpus, and the corroboration must be quoted with that arm "
        "attached rather than from the pooled arm alone."
    )
    return entry


def candidate_dispositions(watters_available: bool) -> list[dict]:
    """Candidate identities that no test in this project has yet addressed,
    each routed to the corpus that could test it or recorded as untestable
    with the reason."""
    return [
        {
            "candidate": "the previous trial's remembered item",
            "status": "under_test_elsewhere",
            "corpus_that_supports_it": "macaque lPFC (Panichello et al. 2024)",
            "disposition": (
                "The per-trial item label needed for it is on disk in the same files this census "
                "reads. It is being fitted as its own analysis and is deliberately not duplicated "
                "here; its estimate, minimum detectable effect and bound belong with that fit."
            ),
        },
        {
            "candidate": "reaction time",
            "status": "available_but_not_yet_run" if not watters_available else "runnable_now",
            "corpus_that_supports_it": (
                "the multi-object primate corpus (Watters et al.), whose per-trial event timestamps "
                "give a response time directly"),
            "disposition": (
                "Not derivable in the macaque lPFC corpus this census runs on: its session files carry "
                "only the cued angle, its index, trial correctness, spike counts and the within-trial "
                "time axis, with no response-time field and no trial-onset timestamps. The multi-object "
                "primate corpus carries explicit per-trial fixation, stimulus, delay, cue, response, "
                "feedback and inter-trial-interval timestamps, so both reaction time and the "
                "inter-trial interval are derivable there. That corpus is being staged as a separate "
                "arm; until it is, this candidate is untested rather than excluded."
            ),
        },
        {
            "candidate": "pupil diameter and eye position, as a proxy for arousal",
            "status": "two_session_probe_only",
            "corpus_that_supports_it": (
                "the multi-object primate corpus, standardized recordings only -- two sessions, one "
                "animal"),
            "disposition": (
                "The most plausible remaining identity for a slow content-free gain, and the one this "
                "census cannot bound at all: the macaque lPFC corpus has no eye or pupil channel, and "
                "the only recordings that do carry one amount to two sessions from a single animal. "
                "Two sessions cannot support a pooled per-session test at any power -- the minimum "
                "detectable effect at that session count is larger than the correlation scale allows -- "
                "so any result there is a lead to follow, never an exclusion and never a finding."
            ),
        },
        {
            "candidate": "time since trial start, and the inter-trial interval",
            "status": "not_testable_in_this_corpus",
            "corpus_that_supports_it": "the multi-object primate corpus",
            "disposition": (
                "Distinct from position in the session, and not excluded by it: position in the session "
                "is slow drift over tens of minutes, while the inter-trial interval is a fast recovery "
                "variable that changes from one trial to the next. The macaque lPFC session files carry "
                "a within-trial time axis only -- every trial's samples are expressed relative to that "
                "trial's own alignment -- so no trial-to-trial gap can be reconstructed from them at "
                "any effort. The multi-object primate corpus carries an explicit inter-trial-interval "
                "onset timestamp per trial and is where this candidate must be tested."
            ),
        },
        {
            "candidate": "the number of items held",
            "status": "not_testable_in_this_corpus",
            "corpus_that_supports_it": "the multi-object primate corpus (one to three items)",
            "disposition": (
                "The macaque lPFC task holds a single cued item on every trial, so load is constant "
                "and has no variance to correlate against -- the statistic cannot reach a value that "
                "would falsify the hypothesis, and no null should be reported for it here. Load is a "
                "live candidate for a gain-like signal and the multi-object primate corpus is the only "
                "one in this project with the manipulation."
            ),
        },
    ]


def _decide_branch(reliability: dict, stored_exclusions: list[dict], census: dict) -> dict:
    behaviour_observable = reliability["rate_free_direction_deviation"]["pooled"]
    reliability_lower = behaviour_observable.get("ci_lower")
    if reliability_lower is None or reliability_lower < RELIABILITY_FLOOR:
        return {
            "branch": "observable_reliability_insufficient",
            "reason": (
                "The observable carrying the behavioural prediction has a split-half reliability "
                f"interval whose lower end is {reliability_lower}, below the pre-declared floor of "
                f"{RELIABILITY_FLOOR}. Neither an exclusion nor a prediction computed on it can be "
                "read at face value."
            ),
        }

    present = [
        f"{observable}: {candidate}"
        for observable, block in census.items()
        for candidate, entry in block.items()
        if isinstance(entry, dict) and entry.get("sign_invariance_audit", {}).get("verdict")
        == "candidate_association_is_present_not_null"
    ]
    if present:
        return {
            "branch": "candidate_association_is_present_not_null",
            "reason": (
                "At least one candidate identity is present in a state observable once the observable's "
                "arbitrary per-session sign is discarded, so the corresponding null was a property of "
                "the estimator rather than of the data: " + "; ".join(sorted(present)) + ". The "
                "orthogonality framing is not available for those candidates."
            ),
            "candidates_present": sorted(present),
        }

    correlation_scale = [e for e in stored_exclusions if e.get("constraining_power") in
                          (CONSTRAINS_AT_MEANINGFUL_SCALE, CONSTRAINS_LARGE_ONLY, EXCLUDES_NOTHING)]
    if all(e["constraining_power"] == EXCLUDES_NOTHING for e in correlation_scale):
        return {
            "branch": "nulls_do_not_constrain",
            "reason": (
                "No null on the correlation scale could have detected even a large association at its "
                "own session count. The honest statement is that the state's identity is untested, not "
                "that it is orthogonal."
            ),
        }

    deviation_block = census["rate_free_direction_deviation"]
    all_meaningful = all(
        entry["bounded_exclusion"].get("constraining_power") == CONSTRAINS_AT_MEANINGFUL_SCALE
        for candidate, entry in deviation_block.items()
        if isinstance(entry, dict) and "bounded_exclusion" in entry
    )
    if all_meaningful:
        return {
            "branch": "predictive_component_orthogonal_within_bounds",
            "reason": (
                "Every candidate identity tested on the behaviour-predicting observable is excluded at "
                "the meaningful effect scale, and that observable's split-half reliability clears the "
                "floor."
            ),
        }
    return {
        "branch": "inconclusive_below_detection_floor",
        "reason": (
            "Some exclusions bind and some do not: at least one candidate's minimum detectable effect "
            "on the behaviour-predicting observable is above the meaningful effect scale, so the set "
            "of nulls does not yet support a blanket orthogonality claim."
        ),
    }


def _pool_reliability(per_session: list[dict], key: str, rng: np.random.Generator) -> dict:
    values = [s[key]["spearman_brown_full_length_reliability"] for s in per_session
              if s[key].get("status") == "computed"]
    if len(values) < 2:
        return {"status": "not_computable", "n_sessions": len(values)}
    array = np.asarray(values, dtype=float)
    median, lower, upper = bootstrap_ci(array, lambda x: float(np.median(x)), n_boot=N_BOOT, rng=rng)
    return {
        "status": "computed", "n_sessions": int(len(values)),
        "median_reliability": float(median), "ci_lower": float(lower), "ci_upper": float(upper),
        "min_reliability": float(array.min()), "max_reliability": float(array.max()),
        "attenuation_factor_at_the_median": float(np.sqrt(max(median, 0.0))),
        "attenuation_note": (
            "Any correlation this observable enters is attenuated by the square root of its "
            "reliability, so a true association of size r appears at about "
            f"{np.sqrt(max(median, 0.0)):.3f} r. Every bound in this census is a bound on the "
            "attenuated association, and the corresponding bound on the true association is looser by "
            "that factor."
        ),
    }


def main() -> None:
    start = time.time()
    root = data_root()
    config = json.loads((Path(__file__).resolve().parents[1] / "config" / "datasets.json").read_text())
    directory = root / config["datasets"]["panichello_2024"]["local_path"]
    paths = [Path(p) for p in sorted(glob.glob(str(directory / "*.mat")))]

    watters_root = root / "Watters"
    watters_available = False  # this census does not stage that corpus; see the candidate dispositions

    output = {
        "version": "2026-08-14",
        "scope": (
            "Macaque lPFC (Panichello et al. 2024), all 25 deposited sessions, delay epoch, 100 ms "
            "bins, deciding window width 3 bins. Every correlation is computed within one session and "
            "only per-session coefficients are pooled across sessions, by the paired sign-flip test. "
            "Nulls already on disk are read as stored and are not refitted; the same-observable census "
            "and the reliability and residual-structure arms are fitted here. The stored nulls were "
            "computed on correct trials only for the leading-component gain and on all trials for the "
            "rate-free direction deviation; every quantity fitted here uses all trials of the session, "
            "correct and error alike, so both observables and every covariate sit on one trial set."
        ),
        "purpose": (
            "A non-significant test bounds nothing on its own. Each null bearing on the identity of "
            "the cross-unit maintenance state is reported here with the minimum effect it could have "
            "detected at 80% power and a plain-language statement of what it does and does not "
            "exclude, ordered most-constraining first."
        ),
        "meaningful_effect_threshold_r_units": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
        "meaningful_effect_threshold_source": (
            "the minimum detectable paired difference of the persistence contrast in "
            "results/state_behavior_link.json (0.139 r units), already reused unchanged by "
            "results/rate_free_state_geometry_behavior_link.json so that every bound in this project's "
            "behavioural and identity lines sits on one scale"
        ),
        "large_association_threshold_r_units": LARGE_ASSOCIATION_THRESHOLD_R_UNITS,
        "reliability_floor": RELIABILITY_FLOOR,
        "bound_decision_rule_declared_before_fitting": BOUND_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "sign_invariance_decision_rule_declared_before_fitting":
            SIGN_INVARIANCE_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "reliability_decision_rule_declared_before_fitting":
            RELIABILITY_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "residual_structure_decision_rule_declared_before_fitting":
            RESIDUAL_STRUCTURE_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "census_branch_decision_rule_declared_before_fitting":
            CENSUS_BRANCH_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "withdrawn_input_not_used": (
            "The per-lag residual-existence values stored under the identity artifact's rank-1 gain "
            "test remove the rank-1 component from the observed matrix but not from the permutation "
            "replicates they are compared against, so every value is negative by construction. They "
            "are not read, quoted or extended anywhere in this census. The residual-structure arm here "
            "removes the component from both sides and refits the null."
        ),
        "n_sessions_seen": len(paths),
        "sessions": [], "excluded_sessions": [],
        "parameters": {
            "n_unit_split_repeats": N_UNIT_SPLIT_REPEATS,
            "n_residual_null_replicates": N_RESIDUAL_NULL_REPLICATES,
            "n_permutations_per_correlation": N_PERM_CORRELATE,
            "n_bootstrap_resamples": N_BOOT,
            "deciding_width_bins": DECIDING_WIDTH_BINS,
            "lag_range_bins": list(COMMON_RANGE_BINS),
            "bin_width_s": BIN_WIDTH_S,
            "seed_tag": SEED_TAG,
        },
    }

    with locked_json_update(CHECKPOINT_PATH) as checkpoint:
        cached_sessions = dict(checkpoint)

    per_session_candidates, per_session_n, per_session_reliability, per_session_residual = [], [], [], []
    for i, path in enumerate(paths):
        session_id = path.stem
        seed = stable_seed(f"{SEED_TAG}|{session_id}")
        entry = cached_sessions.get(session_id)
        if entry is None:
            observables = session_observables(path)
            if observables is None:
                entry = {
                    "kind": "excluded", "session": session_id,
                    "reason": "fewer than 16 usable trials or fewer than 4 units",
                }
            else:
                candidates = session_identity_candidates(observables, seed)
                reliability = session_reliability(observables, seed + 1)
                residual = session_residual_structure(observables["counts"], seed + 2)
                entry = {
                    "kind": "tested", "session": session_id,
                    "n_trials": observables["n_trials"], "n_units": observables["n_units"],
                    "n_correct": int(observables["is_correct"].sum()),
                    "identity_candidates": candidates, "split_half_reliability": reliability,
                    "residual_structure": residual,
                }
            with locked_json_update(CHECKPOINT_PATH) as checkpoint:
                checkpoint[session_id] = entry
            print(f"{session_id} done ({i + 1}/{len(paths)}) {time.time() - start:.0f}s", flush=True)
        else:
            print(f"{session_id} read from checkpoint ({i + 1}/{len(paths)})", flush=True)

        if entry["kind"] == "excluded":
            output["excluded_sessions"].append({"session": entry["session"], "reason": entry["reason"]})
            continue
        per_session_candidates.append(entry["identity_candidates"])
        per_session_n.append(entry["n_trials"])
        per_session_reliability.append(entry["split_half_reliability"])
        if entry["residual_structure"] is not None:
            per_session_residual.append(entry["residual_structure"])
        output["sessions"].append({
            "session": entry["session"], "status": "tested",
            "n_trials": entry["n_trials"], "n_units": entry["n_units"], "n_correct": entry["n_correct"],
            "identity_candidates": entry["identity_candidates"],
            "split_half_reliability": entry["split_half_reliability"],
            "residual_structure": entry["residual_structure"],
        })

    stored = exclusions_from_stored_artifacts()
    stored.sort(key=_sort_last_if_none)
    output["bounded_exclusions_sorted_most_constraining_first"] = stored

    census = same_observable_census(per_session_candidates, per_session_n)
    output["same_observable_census"] = census

    rng = np.random.default_rng(stable_seed(f"{SEED_TAG}|pooled"))
    output["split_half_reliability"] = {
        "leading_component_gain": {
            "pooled": _pool_reliability(per_session_reliability, "leading_component_gain", rng),
        },
        "rate_free_direction_deviation": {
            "pooled": _pool_reliability(per_session_reliability, "rate_free_direction_deviation", rng),
        },
    }
    output["residual_structure_after_the_dominant_gain"] = (
        pool_residual_structure(per_session_residual) if per_session_residual
        else {"status": "not_computable", "n_sessions": 0}
    )

    output["candidate_dispositions"] = candidate_dispositions(watters_available)
    output["candidate_disposition_note"] = (
        f"The multi-object primate corpus is present on disk at {watters_root.name}/ but is not read "
        "by this census; the candidates routed to it are recorded as untested with the corpus that "
        "supports them named, not as excluded."
    )

    reliability_pooled = {
        "leading_component_gain": output["split_half_reliability"]["leading_component_gain"],
        "rate_free_direction_deviation": output["split_half_reliability"]["rate_free_direction_deviation"],
    }
    decision = _decide_branch(reliability_pooled, stored, census)
    output.update(decision)

    output["n_sessions_tested"] = len(output["sessions"])
    output["n_sessions_excluded"] = len(output["excluded_sessions"])
    output["session_accounting"] = (
        f"{output['n_sessions_seen']} sessions seen = {output['n_sessions_tested']} tested + "
        f"{output['n_sessions_excluded']} excluded"
    )
    output["wall_clock_s"] = time.time() - start
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False turns any future stray non-finite value into an
    # immediate ValueError at generation time instead of a silently invalid
    # `Infinity`/`NaN` token in the artifact.
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=float, allow_nan=False))
    print(f"wrote {OUTPUT_PATH} in {output['wall_clock_s']:.0f}s -- branch {output['branch']}")


if __name__ == "__main__":
    main()
