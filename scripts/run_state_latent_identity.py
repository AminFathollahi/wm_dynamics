"""run_state_latent_identity.py -- what is the dominant shared latent that
r_lag_profile has been measuring for five rounds?

The estimator's own construction already rules out the naive reading: each
window's across-trial mean is subtracted before anything is correlated
(state_persistence._window_means_at_width), so a condition-locked ramp or
any temporal profile common to every trial is gone before d_perm is
computed. What is left to test is trial-to-trial covariation of the leading
latent's window means, and two sharp hypotheses survive that removal:

  H-A, the pair-position confound: at fixed width the set of (start, lag)
  pairs is not sampled uniformly over the epoch -- short lags reach the
  whole epoch, long lags are forced toward the centre -- so if r depends on
  WHERE a pair sits rather than on how far apart its two windows are, a
  position effect can masquerade as a lag effect. This module stores r per
  individual (start, lag) pair rather than only its mean at each lag, and
  regresses d_perm on lag and on position jointly.

  H-B, the state is one per-trial scalar: if the centred trial x window
  matrix is close to rank 1, "the state" is a single per-trial gain on a
  shared temporal profile, with no autocorrelation structure of its own to
  measure. This module tests the rank-1 share of that matrix against a
  matched-noise reference, and asks whether d_perm survives on the residual
  after the rank-1 component is removed.

Scope: delay epoch, the deciding window width (3 bins = 300 ms) only, the
same sessions reachable through the lag census (src/corpus_sessions.py's
iter_all_corpora for the three human corpora restricted to the pooled
structure, src/corpus_sessions.py's iter_alm for mouse ALM, and the
Panichello 2024 macaque .mat files read directly, exactly as
run_state_persistence.py and run_state_content_link.py already do). The
macaque arm is run first and in full: it has the strongest d_perm of any
corpus, and it is the only one where item content is both strongly decodable
and worth asking whether the leading latent carries it. The human and mouse
arms follow only if time remains, and are declared not-done rather than
delivered shallow if it does not.

Two documented simplifications, both for compute-budget reasons. (1) The
per-pair position-versus-lag census reuses the full repeated half-split
estimator exactly as delivered elsewhere in this project, so it costs what
one width of the existing lag census already cost -- the per-pair store is
more output, not more compute. The rank-1 gain test and its correlates, by
contrast, use a SINGLE representative half-split per session (fixed seed)
rather than the repeated-split median the rest of this project uses: a
per-trial gain is only meaningful for a specific
held-out set of trials, and repeated random splits would each hold out a
different, unaligned set of trials, which the lag-wise-averaged profile
estimator does not need to worry about but a per-trial quantity does.
(2) Reaction time and trial accuracy are not usable per-trial covariates in
any of the three corpora this module reads: every human corpus's session
loader already restricts to correct trials before this module ever sees
them (accuracy is constant, not a variable), the Panichello .mat files carry
no response-time field among the keys this project has ever found in them,
and the ALM loader does not expose one. Extracting a genuine RT covariate
would need new loader code that does not exist yet; the gap is recorded
once here rather than re-derived per corpus.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from corpus_sessions import ALM_WINDOW_S, data_root, iter_alm, iter_all_corpora  # noqa: E402
from geometry import content_decoding_single_latent  # noqa: E402
from observability import _leading_latent_projection  # noqa: E402
from spike_pipeline import FrozenPSTHTransform, build_psth  # noqa: E402
from state_persistence import (  # noqa: E402
    _ols_slope, _window_means_at_width, ols_with_two_predictors, pair_midpoint_seconds, paired_vs_null_contrast,
    per_unit_permutation_null_r_lag_profile_per_pair, r_lag_profile_per_pair, rank1_gain_and_residual,
    rank1_gain_share, residual_pair_correlations, slope_across_sessions_test, temporal_profile_sign_crossings,
)
from statistics import auroc, fdr_bh, spearman_permutation_test, stable_seed  # noqa: E402

BIN_MS = 100.0
BIN_WIDTH_S = BIN_MS / 1000.0
DECIDING_WIDTH_BINS = 3
HUMAN_DATASETS = ("dandi_000469", "dandi_001187", "dandi_000574")
COMMON_RANGE_BINS = (3, 8)  # 0.3-0.8 s, the deciding early-segment range used across this project's lag contrasts
PANICHELLO_DELAY_WINDOW_MS = (300.0, 1450.0)
N_SHUFFLES_RANK1 = 200
N_PERM_LABEL = 300
CONTENT_K_MAX = 8

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "state_latent_identity.json"


def _stable_seed(*parts) -> int:
    return stable_seed("|".join(str(p) for p in parts))


def _no_null(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "null"}


def _counts_from_spikes(spike_lists, onset, window_s: float, bin_ms: float = BIN_MS) -> np.ndarray:
    rate = build_psth(spike_lists, onset, bin_ms=bin_ms, smooth_ms=0.0, window_s=window_s)
    return rate * (bin_ms / 1000.0)


# ============================================================================
# Session loaders -- same sessions and epoch/width conventions as the lag
# census (run_state_persistence.py's human_lag_rows/alm_lag_rows/
# panichello_lag_rows), yielding raw counts plus whatever trial covariates
# each corpus actually carries.
# ============================================================================

def _panichello_directory(root: Path) -> Path | None:
    config = json.loads((Path(__file__).resolve().parents[1] / "config" / "datasets.json").read_text())
    entry = config["datasets"]["panichello_2024"]  # raise if the registry key is missing/mistyped, not a silent skip
    path = root / entry["local_path"]
    return path if path.is_dir() else None


def macaque_sessions(root: Path):
    directory = _panichello_directory(root)
    if directory is None:
        return
    for path in sorted(glob.glob(str(directory / "*.mat"))):
        raw = loadmat(path, squeeze_me=True)
        spikes = np.asarray(raw["spks"], dtype=float)
        time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
        correct = np.asarray(raw["isCorr"], dtype=bool).reshape(-1)
        cue_idx = np.asarray(raw["cueAngIdx"]).reshape(-1)
        spikes, cue_idx = spikes[correct], cue_idx[correct]
        starts = np.arange(PANICHELLO_DELAY_WINDOW_MS[0], PANICHELLO_DELAY_WINDOW_MS[1], BIN_MS)
        binned = [spikes[:, (time_ms >= s) & (time_ms < s + BIN_MS), :].sum(axis=1) for s in starts]
        counts = np.stack(binned, axis=2)
        yield {
            "corpus": "panichello_lpfc", "session": Path(path).stem, "counts": counts,
            "label": cue_idx.astype(int), "label_field": "cueAngIdx (8-way cued colour/location bin)",
            "n_splits": 10, "n_null_replicates": 10,
        }


def human_sessions(root: Path):
    for meta in iter_all_corpora(root):
        if meta["structure"] != "pooled":
            continue
        onset = meta["epoch_onsets"]["delay"]
        window_s = meta["epoch_windows"]["delay"]
        counts = _counts_from_spikes(meta["spike_lists"], onset, window_s)
        yield {
            "corpus": "human_delay", "dataset": meta["dataset"], "session": meta["session"], "counts": counts,
            "label": meta.get("item_ids"), "label_field": meta.get("item_id_field"),
            "n_splits": 12, "n_null_replicates": 20,
        }


def alm_sessions(root: Path):
    for meta in iter_alm(root, bin_ms=BIN_MS, window_s=ALM_WINDOW_S):
        yield {
            "corpus": "alm", "session": meta["session"], "counts": meta["counts"],
            "label": meta.get("condition"), "label_field": meta.get("item_id_field"),
            "n_splits": 12, "n_null_replicates": 20,
        }


# ============================================================================
# Position-versus-lag decomposition
# ============================================================================

def per_pair_d_perm_records(counts: np.ndarray, width_bins: int, n_splits: int, n_null_replicates: int,
                             rng: np.random.Generator) -> list[dict] | None:
    profile = r_lag_profile_per_pair(counts, width_bins, n_splits=n_splits, rng=rng)
    if profile["status"] != "fitted":
        return None
    null = per_unit_permutation_null_r_lag_profile_per_pair(counts, width_bins, n_replicates=n_null_replicates, rng=rng)
    null_lookup = {(p["s"], p["lag"]): p["r_null_median"] for p in null["pairs"]}
    records = []
    for p in profile["pairs"]:
        key = (p["s"], p["lag"])
        if key not in null_lookup:
            continue
        records.append({
            "s": p["s"], "lag": p["lag"], "lag_s": p["lag"] * BIN_WIDTH_S,
            "position_s": pair_midpoint_seconds(p["s"], p["lag"], width_bins, BIN_WIDTH_S),
            "r": p["r_median"], "r_null": null_lookup[key], "d_perm": p["r_median"] - null_lookup[key],
        })
    return records if records else None


def session_position_vs_lag(records: list[dict]) -> dict:
    """The position-versus-lag decomposition for one session: the joint
    regression of d_perm on lag and position, the position-matched subset
    within the common range, and the r-vs-position slope at the lags with
    the most pairs."""
    joint = ols_with_two_predictors([r["lag_s"] for r in records], [r["position_s"] for r in records],
                                     [r["d_perm"] for r in records])
    marginal_slope = _ols_slope([r["lag_s"] for r in records], [r["d_perm"] for r in records])

    lo, hi = COMMON_RANGE_BINS
    in_range = [r for r in records if lo <= r["lag"] <= hi and r["lag"] != DECIDING_WIDTH_BINS]
    matched_slope, n_matched_pairs, n_matched_lags = None, 0, 0
    if in_range:
        lag_at_max = max(r["lag"] for r in in_range)
        positions_at_max = [r["position_s"] for r in in_range if r["lag"] == lag_at_max]
        pos_lo, pos_hi = min(positions_at_max), max(positions_at_max)
        matched = [r for r in in_range if pos_lo <= r["position_s"] <= pos_hi]
        n_matched_pairs = len(matched)
        n_matched_lags = len(set(r["lag"] for r in matched))
        if n_matched_lags >= 2:
            matched_slope = _ols_slope([r["lag_s"] for r in matched], [r["d_perm"] for r in matched])

    lag_counts = Counter(r["lag"] for r in records)
    fixed_lag_slopes = {}
    for lag, _count in lag_counts.most_common(3):
        pts = [r for r in records if r["lag"] == lag]
        if len(pts) >= 3:
            slope = _ols_slope([r["position_s"] for r in pts], [r["d_perm"] for r in pts])
            fixed_lag_slopes[str(lag)] = {"slope": slope, "n_pairs": len(pts)}

    return {
        "joint_regression": joint, "marginal_lag_slope": marginal_slope,
        "n_matched_pairs": n_matched_pairs, "n_matched_lags": n_matched_lags, "matched_range_slope": matched_slope,
        "fixed_lag_position_slopes": fixed_lag_slopes,
    }


def pool_position_vs_lag(per_session: list[dict]) -> dict:
    coef_l = [s["joint_regression"]["coef_x1"] for s in per_session if s.get("joint_regression")]
    coef_pos = [s["joint_regression"]["coef_x2"] for s in per_session if s.get("joint_regression")]
    marginal = [s["marginal_lag_slope"] for s in per_session if s.get("marginal_lag_slope") is not None]
    matched = [s["matched_range_slope"] for s in per_session if s.get("matched_range_slope") is not None]

    l_test = slope_across_sessions_test(coef_l, "two-sided") if coef_l else {"status": "not_computed"}
    pos_test = slope_across_sessions_test(coef_pos, "two-sided") if coef_pos else {"status": "not_computed"}
    marginal_test = slope_across_sessions_test(marginal, "two-sided") if marginal else {"status": "not_computed"}
    matched_test = slope_across_sessions_test(matched, "two-sided") if matched else {"status": "not_computed"}

    n_joint, n_matched = len(coef_l), len(matched)
    l_significant = bool(l_test.get("significant", False))
    pos_significant = bool(pos_test.get("significant", False))
    marginal_sign = float(np.sign(marginal_test.get("mean_value", 0.0))) if marginal_test.get("status") == "tested" else 0.0
    l_sign = float(np.sign(l_test.get("mean_value", 0.0))) if l_test.get("status") == "tested" else 0.0

    if n_joint < 4 or n_matched < 4:
        branch = "inseparable_by_construction"
    elif l_significant and pos_significant:
        branch = "both_significant"
    elif l_significant and l_sign != marginal_sign and marginal_sign != 0.0:
        branch = "off_branch_list_lag_significant_sign_reversed_from_marginal"
    elif l_significant:
        branch = "lag_effect_survives_position"
    elif pos_significant:
        branch = "effect_is_position_not_lag"
    else:
        branch = "off_branch_list_neither_significant"

    return {
        "n_sessions_with_joint_regression": n_joint, "n_sessions_with_position_matched_subset": n_matched,
        "lag_coefficient_test": l_test, "position_coefficient_test": pos_test,
        "marginal_lag_slope_test": marginal_test, "position_matched_subset_slope_test": matched_test,
        "median_n_matched_pairs": float(np.median([s["n_matched_pairs"] for s in per_session])) if per_session else None,
        "median_n_matched_lags": float(np.median([s["n_matched_lags"] for s in per_session])) if per_session else None,
        "branch": branch,
        "branch_priority_note": "both_significant checked first (report both, state the observable conflates "
                                 "them); then whether L is significant with a sign reversed from the marginal "
                                 "slope (off-list, named rather than forced); then lag_effect_survives_position "
                                 "(L significant, position not); then effect_is_position_not_lag (position "
                                 "significant, L not); then the named off-list bucket for neither significant. "
                                 "inseparable_by_construction is checked before any of these if too few sessions "
                                 "produced a fittable joint regression or position-matched subset.",
    }


# ============================================================================
# The rank-1 gain test: does one per-trial scalar multiplying a shared
# temporal profile explain the leading latent, and does anything survive
# once that component is removed?
# ============================================================================

def session_rank1_and_residual(counts: np.ndarray, width_bins: int, seed: int) -> dict | None:
    rng = np.random.default_rng(seed)
    n_trials = counts.shape[0]
    perm = rng.permutation(n_trials)
    half = n_trials // 2
    train_idx, test_idx = perm[:half], perm[half:]
    if len(train_idx) < 6 or len(test_idx) < 6:
        return None
    transform = FrozenPSTHTransform().fit(counts[train_idx])
    z_train, z_test = transform.transform(counts[train_idx]), transform.transform(counts[test_idx])
    latent = _leading_latent_projection(z_train, z_test)
    window_means = _window_means_at_width(latent, width_bins)
    if window_means is None or window_means.shape[1] < 4:
        return None
    rank1 = rank1_gain_share(window_means, n_shuffles=N_SHUFFLES_RANK1, rng=rng)
    gain, h_profile, residual = rank1_gain_and_residual(window_means)
    crossings = temporal_profile_sign_crossings(h_profile)
    position_corr = [float(np.corrcoef(gain, window_means[:, j])[0, 1]) if window_means[:, j].std() > 0 else None
                      for j in range(window_means.shape[1])]
    return {"test_idx": test_idx, "gain": gain, "rank1": rank1, "residual": residual,
            "temporal_profile_sign_crossings": crossings,
            "position_corr": [c for c in position_corr if c is not None]}


def residual_existence_by_lag(residual: np.ndarray, width_bins: int, null_lookup: dict) -> dict[int, float]:
    """d_perm recomputed on the residual after the rank-1 gain component is
    removed, one value per lag: this session's residual correlation at each
    lag minus the SAME per-unit permutation null already computed for this
    session's full (non-residualised) census, reused rather than re-derived
    on the residual, for compute-budget reasons documented at module level."""
    r_by_lag = residual_pair_correlations(residual, width_bins)
    null_per_lag: dict[int, list[float]] = {}
    for (_s, lag), value in null_lookup.items():
        null_per_lag.setdefault(lag, []).append(value)
    null_per_lag_mean = {lag: float(np.mean(values)) for lag, values in null_per_lag.items()}
    lo, hi = COMMON_RANGE_BINS
    return {lag: r_by_lag[lag] - null_per_lag_mean[lag] for lag in r_by_lag
            if lag in null_per_lag_mean and lo <= lag <= hi}


def common_range_marginal_slope(records: list[dict]) -> float | None:
    """The same early-segment slope this project's other lag contrasts
    report (common range, lag in seconds, adjacency lag dropped), computed
    here on the RAW, non-residualised d_perm so it can be compared directly
    against :func:`residual_early_slope`'s value on the same range after
    the rank-1 component is removed."""
    lo, hi = COMMON_RANGE_BINS
    by_lag: dict[int, list[float]] = {}
    for r in records:
        if lo <= r["lag"] <= hi and r["lag"] != DECIDING_WIDTH_BINS:
            by_lag.setdefault(r["lag"], []).append(r["d_perm"])
    if len(by_lag) < 2:
        return None
    lags = sorted(by_lag)
    return _ols_slope([lag * BIN_WIDTH_S for lag in lags], [float(np.mean(by_lag[lag])) for lag in lags])


def residual_early_slope(residual_d_perm_by_lag: dict[int, float]) -> float | None:
    """The early-segment slope (lag in seconds) of the residual d_perm
    values within the common range, adjacency lag dropped -- the same shape
    of contrast this project runs on the un-residualised d_perm elsewhere,
    run here on what is left after the dominant rank-1 component is
    subtracted out. If this slope is not significant once pooled across
    sessions while the un-residualised early slope is, the fast component
    is carried entirely by the leading mode and is not itself evidence of a
    second, independent decaying component."""
    lags = [lag for lag in residual_d_perm_by_lag if lag != DECIDING_WIDTH_BINS]
    if len(lags) < 2:
        return None
    return _ols_slope([lag * BIN_WIDTH_S for lag in lags], [residual_d_perm_by_lag[lag] for lag in lags])


def pool_residual_existence(per_session_residual: list[dict[int, float]]) -> dict:
    lags = sorted({lag for d in per_session_residual for lag in d})
    by_lag, pvalues, lag_order = {}, [], []
    for lag in lags:
        values = [d[lag] for d in per_session_residual if lag in d]
        result = paired_vs_null_contrast(values, [0.0] * len(values), direction="greater")
        by_lag[str(lag)] = result
        if result.get("status") == "tested":
            pvalues.append(result["p_value"])
            lag_order.append(lag)
    if pvalues:
        fdr = fdr_bh(np.array(pvalues), alpha=0.05)
        for lag, q, rej in zip(lag_order, fdr["q_values"], fdr["reject"]):
            by_lag[str(lag)]["fdr_q_value"] = float(q)
            by_lag[str(lag)]["fdr_significant"] = bool(rej)
    n_tested, n_clearing = len(lag_order), sum(1 for lag in lag_order if by_lag[str(lag)].get("fdr_significant"))
    return {"by_lag": by_lag, "n_lags_tested": n_tested, "n_lags_clearing_fdr": n_clearing}


def pool_rank1(per_session_rank1: list[dict]) -> dict:
    shares = [r["observed_share"] for r in per_session_rank1]
    null_medians = [r["null_share_median"] for r in per_session_rank1]
    share_test = paired_vs_null_contrast(shares, null_medians, direction="greater")
    n_sig = sum(1 for r in per_session_rank1 if r["significantly_above_reference"])
    return {
        "n_sessions": len(per_session_rank1), "n_sessions_significantly_above_reference": n_sig,
        "median_observed_share": float(np.median(shares)) if shares else None,
        "median_null_share_reference": float(np.median(null_medians)) if null_medians else None,
        "pooled_share_vs_matched_noise_reference_test": share_test,
    }


def pool_sign_crossings(per_session_crossings: list[dict]) -> dict:
    counts = [c["n_sign_crossings"] for c in per_session_crossings]
    return {
        "n_sessions": len(counts),
        "median_n_sign_crossings": float(np.median(counts)) if counts else None,
        "n_sessions_with_at_least_one_crossing": sum(1 for c in counts if c >= 1),
        "fraction_sessions_with_at_least_one_crossing": (sum(1 for c in counts if c >= 1) / len(counts)) if counts else None,
    }


def pool_residual_early_slope(per_session_slopes: list[float]) -> dict:
    slopes = [s for s in per_session_slopes if s is not None]
    return slope_across_sessions_test(slopes, "two-sided") if len(slopes) >= 4 else \
        {"status": "underpowered_by_construction", "n_sessions": len(slopes)}


def sign_crossing_interpretation(pooled_crossings: dict, pooled_residual_slope: dict, marginal_early_slope_test: dict) -> dict:
    """Combines the three quantities a lag-dependent slope on a near-rank-1
    matrix needs before it can be trusted: whether the shared temporal
    profile h(t) changes sign inside the epoch, and whether a lag-dependent
    slope survives once the rank-1 component driving that profile is removed.
    Under exact rank 1 with a sign-changing h(t), pairs on the same side of
    the crossing correlate +1 and pairs straddling it correlate -1; since
    longer lags straddle a mid-epoch crossing more often than short ones,
    this manufactures an apparent monotone decay out of a structure with no
    per-lag dynamics in it at all -- a risk that does not exist when h(t)
    never changes sign."""
    crosses = bool(pooled_crossings.get("fraction_sessions_with_at_least_one_crossing") is not None
                    and pooled_crossings["fraction_sessions_with_at_least_one_crossing"] >= 0.5)
    residual_slope_significant_negative = bool(pooled_residual_slope.get("significant_negative", False))
    marginal_slope_significant_negative = bool(marginal_early_slope_test.get("significant_negative", False))

    if not crosses:
        verdict = "profile_does_not_change_sign_no_artifact_risk"
    elif not marginal_slope_significant_negative:
        verdict = "no_lag_dependent_slope_to_explain_sign_crossing_moot"
    elif residual_slope_significant_negative:
        verdict = "lag_dependent_slope_survives_removal_of_the_rank1_component"
    else:
        verdict = "lag_dependent_slope_does_not_survive_removal_of_the_rank1_component_consistent_with_a_sign_crossing_artifact"

    return {
        "majority_of_sessions_have_a_sign_changing_temporal_profile": crosses,
        "marginal_early_slope_significant_negative_before_removing_rank1_component": marginal_slope_significant_negative,
        "residual_early_slope_significant_negative_after_removing_rank1_component": residual_slope_significant_negative,
        "verdict": verdict,
    }


def pool_gain_position(per_session_position_corr: list[list[float]]) -> dict:
    medians = [float(np.median(np.abs(pc))) for pc in per_session_position_corr if pc]
    ranges = [float(np.max(pc) - np.min(pc)) for pc in per_session_position_corr if pc]
    return {
        "n_sessions": len(medians),
        "median_abs_correlation_across_sessions": float(np.median(medians)) if medians else None,
        "median_within_session_range_across_sessions": float(np.median(ranges)) if ranges else None,
        "interpretation": "high median correlation with a small within-session range is what a rank-1 gain "
                           "predicts (the same per-trial scalar explains every window's values about equally "
                           "well); a small or highly variable correlation argues against it.",
    }


def rank1_branch(pooled_rank1: dict, pooled_residual: dict) -> str:
    rank1_dominant = bool(pooled_rank1["pooled_share_vs_matched_noise_reference_test"].get("significant", False))
    n_tested, n_clear = pooled_residual["n_lags_tested"], pooled_residual["n_lags_clearing_fdr"]
    if n_tested == 0:
        return "off_branch_list_residual_not_testable"
    if rank1_dominant and n_clear == 0:
        return "state_is_a_per_trial_gain"
    if rank1_dominant and n_clear > 0:
        return "mixed"
    if (not rank1_dominant) and n_clear > 0:
        return "state_has_structure_beyond_a_gain"
    return "off_branch_list_no_rank1_dominance_and_no_residual_structure"


# ============================================================================
# What the rank-1 gain correlates with
# ============================================================================

def macro_auc_of_scalar(scores: np.ndarray, labels: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    classes = np.unique(labels)

    def _macro(lab: np.ndarray) -> float:
        aucs = []
        for c in classes:
            y = (lab == c).astype(int)
            if 0 < y.sum() < len(y):
                aucs.append(auroc(y, scores))
        return float(np.mean(aucs)) if aucs else float("nan")

    observed = _macro(labels)
    null = np.array([_macro(rng.permutation(labels)) for _ in range(n_perm)])
    valid = null[~np.isnan(null)]
    p_value = float((np.sum(valid >= observed) + 1) / (len(valid) + 1)) if len(valid) >= 10 and not np.isnan(observed) else None
    return {"macro_auc": observed, "p_value": p_value, "n_classes": int(len(classes)), "chance_level": 0.5}


def session_gain_correlates(gain: np.ndarray, test_idx: np.ndarray, counts: np.ndarray, label, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    total_spike_count = counts.sum(axis=(1, 2))[test_idx]
    trial_index = test_idx.astype(float)
    out = {
        "total_spike_count": _no_null(spearman_permutation_test(gain, total_spike_count, rng=rng)),
        "trial_index": _no_null(spearman_permutation_test(gain, trial_index, rng=rng)),
    }
    if label is not None:
        lab = np.asarray(label)[test_idx]
        classes, counts_per_class = np.unique(lab, return_counts=True)
        if len(classes[counts_per_class >= 2]) >= 2 and len(gain) >= 8:
            out["item_label"] = macro_auc_of_scalar(gain, lab, N_PERM_LABEL, rng)
        else:
            out["item_label"] = {"status": "too_few_classes_or_trials_after_split"}
    else:
        out["item_label"] = {"status": "no_label_available"}
    return out


def pool_gain_correlates(per_session: list[dict]) -> dict:
    rho_spike = [c["total_spike_count"]["rho"] for c in per_session if "rho" in c.get("total_spike_count", {})]
    rho_trial = [c["trial_index"]["rho"] for c in per_session if "rho" in c.get("trial_index", {})]
    auc_label = [c["item_label"]["macro_auc"] for c in per_session
                 if isinstance(c.get("item_label", {}).get("macro_auc"), float) and not np.isnan(c["item_label"]["macro_auc"])]
    p_label = [c["item_label"]["p_value"] for c in per_session if c.get("item_label", {}).get("p_value") is not None]
    n_label_sig = sum(1 for p in p_label if p <= 0.05)

    spike_test = slope_across_sessions_test(rho_spike, "two-sided") if rho_spike else {"status": "not_computed"}
    trial_test = slope_across_sessions_test(rho_trial, "two-sided") if rho_trial else {"status": "not_computed"}
    return {
        "total_spike_count": {"n_sessions": len(rho_spike), "pooled_rho_test": spike_test},
        "trial_index": {"n_sessions": len(rho_trial), "pooled_rho_test": trial_test},
        "item_label": {"n_sessions_tested": len(auc_label), "n_sessions_significant": n_label_sig,
                        "median_macro_auc": float(np.median(auc_label)) if auc_label else None},
        "reaction_time_and_accuracy": {
            "status": "not_available_in_any_corpus_this_module_reads",
            "reason": "every human corpus's session loader (src/corpus_sessions.py) already restricts to "
                      "correct trials before this module sees them, so accuracy is constant, not a usable "
                      "covariate; the Panichello .mat files carry no response-time field among the keys this "
                      "project has found in them (cueAng, cueAngIdx, isCorr, spks, tc); the ALM loader does not "
                      "expose a per-trial response time. Extracting a genuine RT covariate needs new loader "
                      "code that does not exist yet. (Panichello trial accuracy, specifically, IS usable as a "
                      "per-trial covariate for a within-session behavioural contrast -- see "
                      "scripts/run_state_behavior_link.py, which reads isCorr directly rather than through this "
                      "module's session loader.)",
        },
    }


# ============================================================================
# Macaque-specific: leading latent's variance share, content by latent
# ============================================================================

def macaque_specific_fields(counts: np.ndarray, label: np.ndarray, seed: int) -> dict:
    n_trials, n_units, n_bins = counts.shape
    window_mean = FrozenPSTHTransform().fit(counts).transform(counts).mean(axis=2)  # (trials, units), in-sample
    centred = window_mean - window_mean.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centred, compute_uv=False)
    total = float(np.sum(singular_values ** 2))
    leading_share = float(singular_values[0] ** 2 / total) if total > 0 else None

    k = min(CONTENT_K_MAX, n_units - 2, max(2, n_trials // 8))
    by_latent = {}
    if k >= 2:
        x = window_mean[:, :, None]
        for j in range(k):
            rng = np.random.default_rng(seed + 1000 * (j + 1))
            result = content_decoding_single_latent(x, label, np.array([0]), k, j, n_splits=3, n_perm=100, rng=rng)
            by_latent[f"latent_{j + 1}"] = {"auc": float(result["auc_per_t"][0]), "p_value": float(result["p_per_t"][0])}
    return {
        "leading_latent_variance_share_of_population_in_sample": leading_share,
        "k_latents": k, "content_auc_by_latent": by_latent,
        "scope_note": "the variance share is computed IN-SAMPLE (basis fit and scored on the same trials, "
                       "unlike every cross-validated statistic elsewhere in this artifact) because it describes "
                       "the population's own geometry rather than a generalisation claim; label_field identifies "
                       "the content label used for content_auc_by_latent.",
    }


# ============================================================================
# Per-session driver and per-corpus aggregation
# ============================================================================

def analyze_session(entry: dict) -> dict | None:
    counts = entry["counts"]
    n_trials = counts.shape[0]
    if n_trials < 16:
        return {"status": "too_few_trials", "n_trials": int(n_trials)}
    seed = _stable_seed(entry["corpus"], entry.get("dataset", ""), entry["session"])
    rng = np.random.default_rng(seed)

    records = per_pair_d_perm_records(counts, DECIDING_WIDTH_BINS, entry["n_splits"], entry["n_null_replicates"], rng)
    if records is None:
        return {"status": "profile_not_fitted", "n_trials": int(n_trials)}
    position_vs_lag = session_position_vs_lag(records)

    rank1_seed = seed + 7
    rank1_result = session_rank1_and_residual(counts, DECIDING_WIDTH_BINS, rank1_seed)
    if rank1_result is None:
        return {"status": "rank1_split_not_fitted", "n_trials": int(n_trials),
                "position_vs_lag": position_vs_lag, "n_pairs_censused": len(records)}

    # The same null this session's per-pair census already computed, reused (not
    # re-derived on the residual) so the residual d_perm test costs nothing new.
    null_lookup = {(r["s"], r["lag"]): r["r_null"] for r in records}
    residual_existence = residual_existence_by_lag(rank1_result["residual"], DECIDING_WIDTH_BINS, null_lookup)
    residual_slope = residual_early_slope(residual_existence)
    marginal_slope = common_range_marginal_slope(records)

    gain_correlates = session_gain_correlates(rank1_result["gain"], rank1_result["test_idx"], counts,
                                               entry.get("label"), seed + 13)

    out = {
        "status": "tested", "n_trials": int(n_trials), "n_units": int(counts.shape[1]),
        "n_pairs_censused": len(records), "position_vs_lag": position_vs_lag,
        "rank1_share": rank1_result["rank1"], "position_correlation_by_window": rank1_result["position_corr"],
        "temporal_profile_sign_crossings": rank1_result["temporal_profile_sign_crossings"],
        "residual_existence_by_lag": residual_existence, "residual_early_slope": residual_slope,
        "marginal_early_slope_before_residualising": marginal_slope,
        "gain_correlates": gain_correlates,
        "records": records,
    }
    return out


def run_corpus(sessions_iter, corpus_label: str, max_sessions: int | None = None) -> dict:
    t0 = time.time()
    per_session, status_rows = [], []
    n_seen = 0
    for entry in sessions_iter:
        n_seen += 1
        if max_sessions is not None and n_seen > max_sessions:
            break
        result = analyze_session(entry)
        base = {"session": entry["session"], "dataset": entry.get("dataset", corpus_label)}
        if result is None or result.get("status") != "tested":
            status_rows.append({**base, "status": (result or {}).get("status", "no_result")})
            continue
        per_session.append({**base, **result})
        print(f"    {corpus_label}/{entry['session']}: n_trials={result['n_trials']} "
              f"n_pairs={result['n_pairs_censused']} rank1_share={result['rank1_share']['observed_share']:.3f} "
              f"elapsed={time.time() - t0:.1f}s", file=sys.stderr)

    pooled_position_vs_lag = pool_position_vs_lag([s["position_vs_lag"] for s in per_session])
    pooled_rank1 = pool_rank1([s["rank1_share"] for s in per_session]) if per_session else {"n_sessions": 0}
    pooled_residual = pool_residual_existence([s["residual_existence_by_lag"] for s in per_session]) \
        if per_session else {"n_lags_tested": 0, "n_lags_clearing_fdr": 0, "by_lag": {}}
    pooled_gain_position = pool_gain_position([s["position_correlation_by_window"] for s in per_session])
    pooled_gain_correlates = pool_gain_correlates([s["gain_correlates"] for s in per_session])
    pooled_crossings = pool_sign_crossings([s["temporal_profile_sign_crossings"] for s in per_session])
    pooled_residual_slope = pool_residual_early_slope([s["residual_early_slope"] for s in per_session])
    pooled_marginal_slope = pool_residual_early_slope([s["marginal_early_slope_before_residualising"] for s in per_session])
    crossing_interpretation = sign_crossing_interpretation(pooled_crossings, pooled_residual_slope, pooled_marginal_slope)
    branch = rank1_branch(pooled_rank1, pooled_residual) if per_session else "not_computed"

    return {
        "n_sessions_seen": n_seen, "n_sessions_tested": len(per_session), "n_sessions_status_row": len(status_rows),
        "status_rows": status_rows,
        "position_vs_lag_decomposition": pooled_position_vs_lag,
        "rank1_gain_test": {
            "pooled_rank1_share": pooled_rank1, "pooled_residual_existence": pooled_residual,
            "pooled_gain_position_correlation": pooled_gain_position,
            "pooled_temporal_profile_sign_crossings": pooled_crossings,
            "pooled_residual_early_slope_test": pooled_residual_slope,
            "pooled_marginal_early_slope_test_same_range": pooled_marginal_slope,
            "sign_crossing_interpretation": crossing_interpretation,
            "branch": branch,
        },
        "gain_correlates": pooled_gain_correlates,
        "session_rows": per_session,
        "wall_clock_s": time.time() - t0,
    }


def main() -> None:
    t0 = time.time()
    root = data_root()
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    output_path = OUTPUT_PATH
    output = json.loads(output_path.read_text()) if output_path.exists() else {
        "version": "2026-08-16",
        "scope": (
            "Delay epoch, deciding window width (3 bins = 300 ms), the same sessions reachable through the lag "
            "census. Per-corpus arms are computed and written independently and incrementally -- the macaque arm "
            "first, because it has the strongest cross-unit correlation of any corpus and is the only one where "
            "item content is both strongly decodable and worth asking whether the leading latent carries it -- "
            "so a partial delivery still has every completed arm on disk even if a later arm is cut for time."
        ),
        "gain_analysis_scope_note": (
            "session_rows[*].rank1_share, temporal_profile_sign_crossings, residual_existence_by_lag, "
            "residual_early_slope and gain_correlates are all computed from ONE fixed-seed held-out half-split "
            "per session, not the repeated-split median every profile statistic elsewhere in this project uses. "
            "A per-trial gain is only meaningful for a specific set of held-out trials; repeated random splits "
            "would each hold out a different, unaligned set of trials, which a lag-wise-averaged profile does "
            "not need to worry about but a per-trial quantity does. session_rows[*].n_pairs_censused and "
            "position_vs_lag, by contrast, are the repeated-split median exactly as delivered elsewhere."
        ),
        "reaction_time_and_accuracy_scope_note": (
            "Reaction time and trial accuracy were wanted as gain correlates (see each corpus's "
            "gain_correlates.reaction_time_and_accuracy field below) but are not usable through any session "
            "loader this module reads: every human corpus's session loader already restricts to correct "
            "trials before this module ever sees them, so accuracy is constant; the Panichello .mat files "
            "carry no response-time field among the keys this project has found in them; the ALM loader does "
            "not expose one. Panichello trial accuracy specifically IS usable and is tested directly against "
            "isCorr in scripts/run_state_behavior_link.py, which bypasses this module's session loader for "
            "exactly that reason."
        ),
        "bin_width_s": BIN_WIDTH_S, "deciding_width_bins": DECIDING_WIDTH_BINS,
        "common_range_bins_for_matched_subset_and_residual_test": COMMON_RANGE_BINS,
        "per_corpus": {},
    }

    if arg in ("all", "panichello"):
        print("Macaque lPFC (priority arm)...", file=sys.stderr)
        panichello_result = run_corpus(macaque_sessions(root), "panichello_lpfc")
        macaque_extra = []
        for entry in macaque_sessions(root):
            seed = _stable_seed("macaque_specific", entry["session"])
            macaque_extra.append({"session": entry["session"], **macaque_specific_fields(entry["counts"], entry["label"], seed)})
        panichello_result["macaque_specific_fields"] = macaque_extra
        output["per_corpus"]["panichello_lpfc"] = panichello_result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))
        print(f"  wrote panichello arm, n_tested={panichello_result['n_sessions_tested']}, "
              f"branch={panichello_result['rank1_gain_test']['branch']}, "
              f"position_branch={panichello_result['position_vs_lag_decomposition']['branch']}, "
              f"{panichello_result['wall_clock_s']:.1f}s", file=sys.stderr)

    if arg in ("all", "human"):
        print("Human delay...", file=sys.stderr)
        human_result = run_corpus(human_sessions(root), "human_delay")
        output["per_corpus"]["human_delay"] = human_result
        output_path.write_text(json.dumps(output, indent=2))
        print(f"  wrote human arm, n_tested={human_result['n_sessions_tested']}, "
              f"branch={human_result['rank1_gain_test']['branch']}, {human_result['wall_clock_s']:.1f}s", file=sys.stderr)

    if arg in ("all", "alm"):
        print("Mouse ALM...", file=sys.stderr)
        alm_result = run_corpus(alm_sessions(root), "alm")
        output["per_corpus"]["alm"] = alm_result
        output_path.write_text(json.dumps(output, indent=2))
        print(f"  wrote alm arm, n_tested={alm_result['n_sessions_tested']}, "
              f"branch={alm_result['rank1_gain_test']['branch']}, {alm_result['wall_clock_s']:.1f}s", file=sys.stderr)

    output["wall_clock_s_this_invocation"] = time.time() - t0
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {output_path} in {time.time() - t0:.1f}s total this invocation", file=sys.stderr)


if __name__ == "__main__":
    main()
