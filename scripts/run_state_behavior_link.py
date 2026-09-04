"""run_state_behavior_link.py -- does the cross-unit state predict the
trial's outcome?

Within a single Panichello 2024 macaque lPFC recording session, is the
cross-unit state -- d_perm, the fixed-width across-trial correlation of the
leading population latent's window means, measured against its per-unit
permutation null -- stronger on trials the animal answered correctly than on
trials it got wrong? A positive answer converts this project's existence
claim ("a trial-specific cross-unit state exists") into a behavioural one
("and it matters"). A null answer is reported as a two-sided confidence
interval and a minimum-detectable-difference bound rather than as a bare
p-value, because the honest end of an existence argument is a bound on
relevance, not a shrug.

The deciding statistic is a LEVEL, not a slope: the mean of d_perm(lag) over
the declared early range (lags 3-8 bins = 0.3-0.8 s at width 3 bins = 0.3 s)
-- the widest range every delay-length arm in this project reaches without
extrapolating past its own native window, and the natural reading of "is the
state STRONGER", a magnitude question, as opposed to the decay-shape
questions this project's other artifacts already answer. Both are
pre-declared before any correct-versus-error contrast is computed.

Direction is NOT pre-declared anywhere in this module: the sign found is
reported as found. Every difference is computed WITHIN one session; only the
per-session differences are combined across sessions (paired sign-flip
test), never trials from two sessions pooled together, because accuracy
stratifies hard by cohort year and the corpus has two animals.

Trial count is matched by subsampling correct trials down to the
error-trial count, at least 200 independent draws per session, the full
statistic (profile AND its permutation null -- both move with trial count)
recomputed on every draw, median across draws reported as the session's
matched correct-trial value with the across-draw spread alongside it. The
unmatched (all correct trials vs all error trials) contrast is also
reported, clearly labelled as the confounded diagnostic this design guards
against, never as the deciding number.

Reuses src/state_persistence.py's r_lag_profile and
per_unit_permutation_null_r_lag_profile unchanged -- the identical estimator
scripts/run_state_persistence.py's panichello_lag_rows uses for this corpus,
pointed at trial subsets instead of at the full correct-only population that
function restricts itself to.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import pointbiserialr

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from corpus_sessions import data_root  # noqa: E402
from observability import _leading_latent_projection  # noqa: E402
from spike_pipeline import FrozenPSTHTransform  # noqa: E402
from state_persistence import (  # noqa: E402
    _window_means_at_width, per_unit_permutation_null_r_lag_profile, r_lag_profile,
    rank1_gain_and_residual, slope_across_sessions_test,
)
from statistics import minimum_detectable_paired_difference, stable_seed  # noqa: E402

BIN_MS = 100.0
BIN_WIDTH_S = BIN_MS / 1000.0
DECIDING_WIDTH_BINS = 3
PANICHELLO_DELAY_WINDOW_MS = (300.0, 1450.0)  # matches scripts/run_state_persistence.py's macaque delay window

# The declared "0.3-0.85 s" early range at the deciding fixed window width:
# bins 3-8 (0.3-0.8 s). This is the SAME range already established elsewhere
# in this project as the widest common range every delay-length arm (human,
# mouse ALM, macaque) can reach without any arm extrapolating past its own
# native window (see scripts/run_state_shape_common_range.py's
# COMMON_RANGES_BINS["0.3_to_0.8s"]), and it is confirmed here by an exact
# independent reproduction: recomputing the round's own r_obs/r_null/d_perm
# slope table for all four delay-length/encoding arms at bins (3, 8) with no
# lag excluded matches the round's quoted numbers to four decimal places for
# every arm and every one of the three statistics (see
# results/state_shape_common_range.json's four_row_headline_table field) --
# bins (3, 9) does not. Declared here, before any correct-versus-error
# contrast is computed, and never adjusted afterward.
LAG_RANGE_BINS = (3, 8)

# Pre-declared reachability floor: a session needs at least this many error
# trials for the matched-count correlation contrast to have usable sampling
# error. Computed per session, not assumed from the accuracy figures alone.
MIN_ERROR_TRIALS_FOR_REACHABILITY = 60
MIN_RESULT_SESSIONS_TO_PROCEED = 6

N_MATCHED_DRAWS = 200

# Full-statistic parameters: used once per session for the error-trial set
# and for the confounded unmatched diagnostic -- the same convention
# scripts/run_state_persistence.py's panichello_lag_rows uses for this
# corpus (n_splits=10, n_null_replicates=10, default splits-per-replicate).
FULL_N_SPLITS = 10
FULL_N_NULL_REPLICATES = 10
FULL_N_NULL_SPLITS_PER_REPLICATE = 6

# Reduced parameters for the >=200-draw matched-subsampling loop.
# Recomputing the full statistic (profile + permutation null) at
# FULL_*-parameter cost on every one of >=200 draws, for sessions with up to
# ~800 simultaneously recorded units, is not tractable at this project's
# wall-clock budget: a single FULL_*-parameter draw on the heaviest
# reachable session (716 units) measured ~13 s, which would put 200 draws at
# over 40 minutes for that ONE session. These are the smallest split and
# replicate counts at which r_lag_profile and its permutation null can still
# report a "fitted" status at all (status requires >=3 successful
# half-splits, and the permutation null's own inner r_lag_profile call needs
# the same floor). The draw COUNT itself is never reduced below the
# pre-declared 200 -- only these internal split/replicate counts, disclosed
# here and again in every session's own "matched_draw_parameters" field and
# in results/implementation_report.md.
MATCHED_DRAW_N_SPLITS = 4
MATCHED_DRAW_N_NULL_REPLICATES = 4
MATCHED_DRAW_N_NULL_SPLITS_PER_REPLICATE = 3

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "state_behavior_link.json"


def _panichello_directory(root: Path) -> Path | None:
    config = json.loads((Path(__file__).resolve().parents[1] / "config" / "datasets.json").read_text())
    entry = config["datasets"]["panichello_2024"]  # raise if the registry key is missing/mistyped, not a silent skip
    path = root / entry["local_path"]
    return path if path.is_dir() else None


def _counts_from_spikes(spikes: np.ndarray, time_ms: np.ndarray) -> np.ndarray:
    """(trials, units, bins) delay-epoch spike counts at the deciding bin
    width, from the raw (trials, time, units) spike-count array and its
    per-sample time axis -- the same binning
    scripts/run_state_persistence.py's panichello_lag_rows uses (that
    function restricts to correct trials before binning; this module needs
    both classes intact, so the restriction happens later, per trial
    subset, not here)."""
    starts = np.arange(PANICHELLO_DELAY_WINDOW_MS[0], PANICHELLO_DELAY_WINDOW_MS[1], BIN_MS)
    binned = [spikes[:, (time_ms >= s) & (time_ms < s + BIN_MS), :].sum(axis=1) for s in starts]
    return np.stack(binned, axis=2)


def full_statistic(counts: np.ndarray, n_splits: int, n_null_replicates: int,
                    n_null_splits_per_replicate: int, seed: int) -> dict:
    """d_perm(L) at every lag in LAG_RANGE_BINS, its two components r_obs(L)
    and r_null(L) reported alongside it in every case (a d_perm field
    without them is an incomplete result, per this project's own standing
    rule on the contrast), and the single scalar this module's deciding
    contrast tests: the mean of d_perm over that range. Profile and
    permutation null use independently seeded generators (seed, seed + 1),
    the same pattern scripts/run_state_persistence.py's _lag_run_row uses."""
    profile = r_lag_profile(counts, DECIDING_WIDTH_BINS, n_splits=n_splits, rng=np.random.default_rng(seed))
    if profile["status"] != "fitted":
        return {"status": profile["status"]}
    null = per_unit_permutation_null_r_lag_profile(
        counts, DECIDING_WIDTH_BINS, n_replicates=n_null_replicates,
        n_splits_per_replicate=n_null_splits_per_replicate, rng=np.random.default_rng(seed + 1))
    lo, hi = LAG_RANGE_BINS
    lags_in_range = sorted(lag for lag in profile["lags"] if lo <= lag <= hi and lag in null["lags"])
    if not lags_in_range:
        return {"status": "no_lags_in_declared_range"}
    per_lag = {
        str(lag): {
            "r_obs": profile["lags"][lag]["r_median"],
            "r_null": null["lags"][lag]["r_null_median"],
            "d_perm": profile["lags"][lag]["r_median"] - null["lags"][lag]["r_null_median"],
        }
        for lag in lags_in_range
    }
    r_obs_mean = float(np.mean([v["r_obs"] for v in per_lag.values()]))
    r_null_mean = float(np.mean([v["r_null"] for v in per_lag.values()]))
    d_perm_level = float(np.mean([v["d_perm"] for v in per_lag.values()]))
    return {
        "status": "fitted", "n_lags_in_range": len(lags_in_range), "per_lag": per_lag,
        "r_obs_mean": r_obs_mean, "r_null_mean": r_null_mean, "d_perm_level": d_perm_level,
        "identity_check_r_obs_minus_r_null_minus_d_perm": r_obs_mean - r_null_mean - d_perm_level,
    }


def matched_correct_draws(counts_all: np.ndarray, correct_idx: np.ndarray, n_error: int,
                           session_seed_tag: str, n_draws: int = N_MATCHED_DRAWS) -> dict:
    """Trial-count match: n_draws independent subsamples of the correct
    trials down to n_error, the full statistic recomputed on each (both
    r_obs and r_null move with trial count, so both must be recomputed, not
    only r_obs), median across draws reported as the session's matched
    correct-trial value with the across-draw spread alongside it so a
    reader can see the matching worked, rather than only its summary."""
    d_perm_levels, r_obs_means, r_null_means = [], [], []
    n_fitted = 0
    for draw in range(n_draws):
        draw_rng = np.random.default_rng(stable_seed(f"{session_seed_tag}|draw{draw}|subsample"))
        sub_idx = draw_rng.choice(correct_idx, size=n_error, replace=False)
        counts_sub = counts_all[sub_idx]
        fit_seed = stable_seed(f"{session_seed_tag}|draw{draw}|fit")
        result = full_statistic(counts_sub, MATCHED_DRAW_N_SPLITS, MATCHED_DRAW_N_NULL_REPLICATES,
                                 MATCHED_DRAW_N_NULL_SPLITS_PER_REPLICATE, fit_seed)
        if result["status"] != "fitted":
            continue
        n_fitted += 1
        d_perm_levels.append(result["d_perm_level"])
        r_obs_means.append(result["r_obs_mean"])
        r_null_means.append(result["r_null_mean"])
    if n_fitted < max(3, n_draws // 2):
        return {"status": "too_few_fitted_draws", "n_draws_requested": n_draws, "n_draws_fitted": n_fitted}
    return {
        "status": "fitted", "n_draws_requested": n_draws, "n_draws_fitted": n_fitted,
        "d_perm_level_median": float(np.median(d_perm_levels)),
        "d_perm_level_iqr": [float(np.percentile(d_perm_levels, 25)), float(np.percentile(d_perm_levels, 75))],
        "r_obs_mean_median": float(np.median(r_obs_means)),
        "r_null_mean_median": float(np.median(r_null_means)),
        "d_perm_level_all_draws": d_perm_levels,
        "r_obs_mean_all_draws": r_obs_means,
        "r_null_mean_all_draws": r_null_means,
    }


def trial_amplitude_covariates(counts_all: np.ndarray) -> dict:
    """The three per-trial covariates :func:`cheap_first_look` correlates
    against trial outcome, computed once so a caller needing the raw
    per-trial arrays (rather than only their correlation with outcome) does
    not re-derive them: (a) the per-trial leading-component score of the
    centred trial x window matrix -- the per-trial gain a rank-1
    decomposition assigns, via rank1_gain_and_residual exactly as this
    project's rank-1 gain audit does -- (b) total per-trial spike count in
    the delay epoch, and (c) trial index within the session. Fit in-sample
    on the same trials, the same convention macaque_specific_fields uses in
    scripts/run_state_latent_identity.py."""
    if counts_all.shape[0] < 16:
        return {"status": "not_computable", "reason": "fewer than 16 trials"}
    transform = FrozenPSTHTransform().fit(counts_all)
    z = transform.transform(counts_all)
    latent = _leading_latent_projection(z, z)
    window_means = _window_means_at_width(latent, DECIDING_WIDTH_BINS)
    if window_means is None or window_means.shape[1] < 2:
        return {"status": "not_computable", "reason": "fewer than 2 windows at the deciding width"}
    gain, _h_profile, _residual = rank1_gain_and_residual(window_means)
    total_spike_count = counts_all.sum(axis=(1, 2))
    trial_index = np.arange(counts_all.shape[0], dtype=float)
    return {
        "status": "computed", "leading_component_score_gain": gain,
        "total_spike_count": total_spike_count, "trial_index": trial_index,
    }


def cheap_first_look(counts_all: np.ndarray, is_corr: np.ndarray) -> dict:
    """The cheap first look this project's gain-correlates field should have
    contained: point-biserial correlation between trial outcome and each of
    :func:`trial_amplitude_covariates`'s three per-trial covariates.
    Answers a different question from the matched d_perm contrast: whether
    the trial's overall amplitude or coherence predicts the outcome, as
    opposed to whether the cross-unit state's persistence across time does
    -- report both, one is not a substitute for the other."""
    covariates = trial_amplitude_covariates(counts_all)
    if covariates["status"] != "computed":
        return covariates
    is_corr_float = is_corr.astype(float)

    def _pointbiserial(x: np.ndarray) -> dict:
        if np.std(x) == 0.0 or np.std(is_corr_float) == 0.0:
            return {"status": "not_computable", "reason": "zero variance in outcome or covariate"}
        result = pointbiserialr(is_corr_float, x)
        return {"status": "computed", "r": float(result.statistic), "p_value": float(result.pvalue)}

    return {
        "status": "computed",
        "leading_component_score_gain": _pointbiserial(covariates["leading_component_score_gain"]),
        "total_spike_count": _pointbiserial(covariates["total_spike_count"]),
        "trial_index": _pointbiserial(covariates["trial_index"]),
    }


def analyze_session(path: Path) -> dict:
    session_id = path.stem
    raw = loadmat(str(path), simplify_cells=True)
    spikes = np.asarray(raw["spks"], dtype=float)
    time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
    is_corr = np.asarray(raw["isCorr"]).astype(bool).reshape(-1)
    n_trials = int(len(is_corr))
    n_correct = int(is_corr.sum())
    n_error = n_trials - n_correct
    accuracy = (n_correct / n_trials) if n_trials else None
    n_units = int(spikes.shape[2])
    reachable = n_error >= MIN_ERROR_TRIALS_FOR_REACHABILITY

    base = {
        "session": session_id, "n_trials": n_trials, "n_correct": n_correct, "n_error": n_error,
        "accuracy": accuracy, "n_units": n_units, "reachable": bool(reachable),
        "reachability_reason": (
            f"n_error={n_error} >= {MIN_ERROR_TRIALS_FOR_REACHABILITY} (pre-declared floor)" if reachable else
            f"n_error={n_error} < {MIN_ERROR_TRIALS_FOR_REACHABILITY} (pre-declared floor) -- too few error "
            "trials for the matched-count correlation contrast to have usable sampling error"
        ),
    }

    counts_all = _counts_from_spikes(spikes, time_ms)
    base["cheap_first_look"] = cheap_first_look(counts_all, is_corr)

    if not reachable:
        base["status"] = "not_reachable_for_matched_contrast"
        return base

    correct_idx = np.where(is_corr)[0]
    error_idx = np.where(~is_corr)[0]
    counts_error = counts_all[error_idx]
    counts_correct_full = counts_all[correct_idx]
    session_seed_tag = f"panichello_behavior_link|{session_id}"
    session_seed = stable_seed(session_seed_tag)

    error_full = full_statistic(counts_error, FULL_N_SPLITS, FULL_N_NULL_REPLICATES,
                                 FULL_N_NULL_SPLITS_PER_REPLICATE, session_seed + 1)
    confounded_unmatched_correct_full = full_statistic(
        counts_correct_full, FULL_N_SPLITS, FULL_N_NULL_REPLICATES,
        FULL_N_NULL_SPLITS_PER_REPLICATE, session_seed + 2)
    matched_correct = matched_correct_draws(counts_all, correct_idx, n_error, session_seed_tag)
    # Same error trials as error_full, but at the SAME reduced estimator settings
    # matched_correct_draws uses -- the trial-count-matched cell of the deciding
    # contrast must also be estimator-setting-matched to its partner cell, or a
    # d_perm difference between the two is inseparable from a difference in how
    # many half-splits each side's median was taken over (both r_obs and r_null
    # move with split/replicate count, not only with trial count).
    error_matched_params = full_statistic(counts_error, MATCHED_DRAW_N_SPLITS, MATCHED_DRAW_N_NULL_REPLICATES,
                                           MATCHED_DRAW_N_NULL_SPLITS_PER_REPLICATE, session_seed + 4)

    return finalize_tested_session(base, error_full, error_matched_params, matched_correct,
                                    confounded_unmatched_correct_full)


def finalize_tested_session(base: dict, error_full: dict, error_matched_params: dict,
                             matched_correct: dict, confounded_unmatched_correct_full: dict) -> dict:
    """The derived fields for a reachable session, given its four already-
    computed cells -- shared by :func:`analyze_session` (which computes all
    four fresh) and by :func:`patch_deciding_contrast_parity` (which reuses
    three already-computed, expensive cells from a prior run's output and
    supplies only a freshly recomputed ``error_matched_params``), so the two
    call sites cannot derive the deciding contrast differently."""
    base["error_full_statistic"] = error_full
    base["error_full_statistic_matched_params"] = error_matched_params
    base["matched_correct_draws"] = matched_correct
    base["confounded_unmatched_correct_full_statistic"] = confounded_unmatched_correct_full
    base["matched_draw_parameters"] = {
        "n_splits": MATCHED_DRAW_N_SPLITS, "n_null_replicates": MATCHED_DRAW_N_NULL_REPLICATES,
        "n_null_splits_per_replicate": MATCHED_DRAW_N_NULL_SPLITS_PER_REPLICATE,
        "reduced_from_full_statistic_parameters_for_runtime": True,
        "also_applied_to": "error_full_statistic_matched_params, so the deciding contrast's two cells share "
                            "the same estimator settings and only trial-count matching (not estimator setting) "
                            "distinguishes the deciding pairing from the confounded-unmatched one.",
        "full_statistic_parameters": {
            "n_splits": FULL_N_SPLITS, "n_null_replicates": FULL_N_NULL_REPLICATES,
            "n_null_splits_per_replicate": FULL_N_NULL_SPLITS_PER_REPLICATE,
        },
    }

    if error_full["status"] != "fitted" or matched_correct["status"] != "fitted" \
            or error_matched_params["status"] != "fitted":
        base["status"] = "full_statistic_not_fitted"
        return base

    base["status"] = "tested"
    base["correct_minus_error_d_perm_level_matched"] = \
        matched_correct["d_perm_level_median"] - error_matched_params["d_perm_level"]
    base["deciding_pairing_note"] = (
        "correct_minus_error_d_perm_level_matched IS THE DECIDING NUMBER. It pairs matched_correct_draws "
        "(trial count matched to n_error, estimator settings n_splits=%d/n_null_replicates=%d/"
        "n_null_splits_per_replicate=%d) against error_full_statistic_matched_params (the SAME error trials, "
        "the SAME reduced estimator settings) -- both trial count AND estimator setting are matched between "
        "the two cells this difference compares, so the difference cannot be an artifact of either."
    ) % (MATCHED_DRAW_N_SPLITS, MATCHED_DRAW_N_NULL_REPLICATES, MATCHED_DRAW_N_NULL_SPLITS_PER_REPLICATE)
    base["correct_minus_error_d_perm_level_confounded_unmatched"] = (
        confounded_unmatched_correct_full["d_perm_level"] - error_full["d_perm_level"]
        if confounded_unmatched_correct_full["status"] == "fitted" else None
    )
    base["confounded_unmatched_diagnostic_note"] = (
        "correct_minus_error_d_perm_level_confounded_unmatched uses ALL correct trials against ALL error "
        "trials, at unequal trial counts, both sides at the FULL estimator settings -- both r_obs and r_null "
        "move with trial count, so this diagnostic is confounded with the correct/error trial-count "
        "imbalance and is reported only to show what the unmatched contrast would have given, never as the "
        "deciding number. See deciding_pairing_note for the deciding number."
    )
    base["parameter_setting_effect_on_error_trials"] = (
        error_full["d_perm_level"] - error_matched_params["d_perm_level"]
        if error_full["status"] == "fitted" and error_matched_params["status"] == "fitted" else None
    )
    base["parameter_setting_effect_note"] = (
        "error_full_statistic (full settings) minus error_full_statistic_matched_params (reduced settings), "
        "computed on the IDENTICAL error trials -- isolates how much d_perm_level moves from the "
        "split/replicate-count reduction alone, holding both the trial set and the correct/error label fixed. "
        "Reported beside the deciding contrast so a reader can see whether the reduction itself could account "
        "for the deciding difference's size."
    )
    return base


def _pool_cheap_first_look(sessions: list[dict], field: str) -> dict:
    values = [
        s["cheap_first_look"][field]["r"] for s in sessions
        if s.get("cheap_first_look", {}).get(field, {}).get("status") == "computed"
    ]
    test = slope_across_sessions_test(values, "two-sided") if values else {"status": "not_computed"}
    return {"n_sessions": len(values), "pooled_r_test": test}


def patch_deciding_contrast_parity() -> None:
    """One-time fix for an estimator-parity defect caught after the full run
    landed: the deciding contrast's two cells were computed at different
    internal split/replicate settings (matched_correct_draws at the reduced
    settings, error_full_statistic at the full settings), so a d_perm
    difference between them was inseparable from a difference in how many
    half-splits each side's median was taken over. Reuses every
    already-computed, expensive field in the existing output (matched_
    correct_draws' >=200 draws are NOT recomputed) and adds only the one
    new cheap computation (error_full_statistic_matched_params, a single
    full_statistic call at the reduced settings on the same error trials),
    then re-derives every field that depends on it via
    :func:`finalize_tested_session`, the same function :func:`analyze_
    session` uses, so the two code paths cannot disagree on how the
    deciding number is built."""
    if not OUTPUT_PATH.exists():
        raise SystemExit("no existing results/state_behavior_link.json to patch")
    output = json.loads(OUTPUT_PATH.read_text())
    root = data_root()
    directory = _panichello_directory(root)
    n_patched = 0
    for session in output["sessions"]:
        if session.get("status") != "tested":
            continue
        session_id = session["session"]
        path = directory / f"{session_id}.mat"
        raw = loadmat(str(path), simplify_cells=True)
        spikes = np.asarray(raw["spks"], dtype=float)
        time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
        is_corr = np.asarray(raw["isCorr"]).astype(bool).reshape(-1)
        counts_all = _counts_from_spikes(spikes, time_ms)
        counts_error = counts_all[np.where(~is_corr)[0]]
        session_seed = stable_seed(f"panichello_behavior_link|{session_id}")

        error_matched_params = full_statistic(counts_error, MATCHED_DRAW_N_SPLITS, MATCHED_DRAW_N_NULL_REPLICATES,
                                               MATCHED_DRAW_N_NULL_SPLITS_PER_REPLICATE, session_seed + 4)
        finalize_tested_session(
            session, session["error_full_statistic"], error_matched_params, session["matched_correct_draws"],
            session["confounded_unmatched_correct_full_statistic"])
        n_patched += 1
        print(f"  patched {session_id} ({n_patched} done)", file=sys.stderr, flush=True)
        OUTPUT_PATH.write_text(json.dumps(output, indent=2))

    _recompute_pooled_fields(output)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Patched {n_patched} tested sessions and recomputed the pooled deciding contrast.", file=sys.stderr)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "patch":
        patch_deciding_contrast_parity()
        return

    t0 = time.time()
    root = data_root()
    directory = _panichello_directory(root)
    if directory is None:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(
            {"status": "not_available", "reason": "Panichello_2024 not staged"}, indent=2))
        return
    files = sorted(glob.glob(str(directory / "*.mat")))
    session_ids = [Path(p).stem for p in files]

    output = {
        "version": "2026-08-17",
        "scope": (
            "All 25 Panichello 2024 macaque lPFC sessions read directly from the deposited .mat files "
            "(cueAng, cueAngIdx, isCorr, spks, tc). Within each session reachable at the pre-declared floor, "
            "the deciding contrast is the cross-unit state's magnitude (mean d_perm over the macaque's own "
            "full reachable lag range at the deciding fixed window width) on correct trials at matched trial "
            "count minus the same quantity on error trials, tested across sessions by a paired sign-flip "
            "test. Sessions below the reachability floor are reported for accuracy and reachability only."
        ),
        "bin_width_s": BIN_WIDTH_S, "deciding_width_bins": DECIDING_WIDTH_BINS,
        "lag_range_bins": list(LAG_RANGE_BINS),
        "lag_range_seconds": [LAG_RANGE_BINS[0] * BIN_WIDTH_S, LAG_RANGE_BINS[1] * BIN_WIDTH_S],
        "lag_range_reason": (
            "Bins 3-8 (0.3-0.8 s): the widest range every delay-length arm in this project (human, mouse ALM, "
            "macaque) reaches without extrapolating past its own native window, already established as "
            "COMMON_RANGES_BINS['0.3_to_0.8s'] in scripts/run_state_shape_common_range.py, and confirmed as the "
            "literal range behind this project's own quoted r_obs/r_null/d_perm reference numbers by an exact "
            "four-decimal-place independent reproduction (results/state_shape_common_range.json's "
            "four_row_headline_table field). n_bins=12 for every macaque session (the delay-epoch window and "
            "bin width are fixed by construction), so bins 3-8 are always reachable whenever the estimator "
            "fits at all. Fixed in seconds and bins before any correct-versus-error contrast was computed."
        ),
        "deciding_statistic_note": (
            "A LEVEL (the mean of d_perm over lag_range_bins), not a slope: the question this artifact "
            "answers is whether the state is STRONGER on correct trials, a magnitude comparison, not whether "
            "its decay shape differs -- shape questions are answered elsewhere in this project."
        ),
        "reachability_floor_min_error_trials": MIN_ERROR_TRIALS_FOR_REACHABILITY,
        "n_matched_draws_per_session": N_MATCHED_DRAWS,
        "sessions": [],
        "sessions_completed": [],
        "sessions_pending": list(session_ids),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for i, path in enumerate(files):
        session_id = Path(path).stem
        print(f"[{i + 1}/{len(files)}] {session_id}...", file=sys.stderr, flush=True)
        t_s = time.time()
        result = analyze_session(Path(path))
        print(f"  status={result.get('status')} reachable={result.get('reachable')} "
              f"n_error={result.get('n_error')} n_units={result.get('n_units')} "
              f"elapsed={time.time() - t_s:.1f}s total_elapsed={time.time() - t0:.1f}s",
              file=sys.stderr, flush=True)
        output["sessions"].append(result)
        output["sessions_completed"].append(session_id)
        output["sessions_pending"] = session_ids[i + 1:]
        OUTPUT_PATH.write_text(json.dumps(output, indent=2))  # flush per session

    n_reachable = sum(1 for s in output["sessions"] if s.get("reachable"))
    output["n_sessions_reachable"] = n_reachable
    if n_reachable < MIN_RESULT_SESSIONS_TO_PROCEED:
        output["status"] = "underpowered_stopped_before_deciding_contrast"
        output["stop_reason"] = (
            f"Only {n_reachable} of 25 sessions reach the pre-declared reachability floor of "
            f"{MIN_ERROR_TRIALS_FOR_REACHABILITY} error trials; fewer than "
            f"{MIN_RESULT_SESSIONS_TO_PROCEED} means the deciding contrast is not run, per the pre-declared "
            "stopping rule -- an underpowered behavioural contrast is exactly the kind of result this "
            "project has spent several rounds withdrawing."
        )
        OUTPUT_PATH.write_text(json.dumps(output, indent=2))
        print(output["stop_reason"], file=sys.stderr)
        return

    branch = _recompute_pooled_fields(output)
    output["wall_clock_s"] = time.time() - t0
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Wrote {OUTPUT_PATH} in {time.time() - t0:.1f}s -- branch={branch}", file=sys.stderr)
    print(json.dumps({"branch": branch, "n_sessions_reachable": n_reachable,
                       "n_sessions_tested": sum(1 for s in output['sessions'] if s.get('status') == 'tested')},
                      indent=2))


def _recompute_pooled_fields(output: dict) -> str:
    """The cross-session pooling (deciding_contrast, confounded_unmatched_
    diagnostic, parameter_setting_effect_diagnostic, cheap_first_look_
    pooled), factored out of :func:`main` so :func:`patch_deciding_
    contrast_parity` can rebuild these fields from the per-session data
    without duplicating the pooling logic. Returns the deciding branch."""
    tested = [s for s in output["sessions"] if s.get("status") == "tested"]
    differences_matched = [s["correct_minus_error_d_perm_level_matched"] for s in tested]
    differences_confounded = [
        s["correct_minus_error_d_perm_level_confounded_unmatched"] for s in tested
        if s.get("correct_minus_error_d_perm_level_confounded_unmatched") is not None
    ]
    parameter_effects = [
        s["parameter_setting_effect_on_error_trials"] for s in tested
        if s.get("parameter_setting_effect_on_error_trials") is not None
    ]

    deciding_test = slope_across_sessions_test(differences_matched, "two-sided") if differences_matched else \
        {"status": "not_computed"}
    confounded_test = slope_across_sessions_test(differences_confounded, "two-sided") if differences_confounded else \
        {"status": "not_computed"}
    parameter_effect_test = slope_across_sessions_test(parameter_effects, "two-sided") if parameter_effects else \
        {"status": "not_computed"}
    mdd = minimum_detectable_paired_difference(differences_matched) if len(differences_matched) >= 2 else \
        {"status": "not_computable"}

    if deciding_test.get("significant_positive"):
        branch = "state_is_stronger_on_correct_trials_than_error_trials"
    elif deciding_test.get("significant_negative"):
        branch = "state_is_weaker_on_correct_trials_than_error_trials"
    elif deciding_test.get("status") == "tested":
        branch = "no_significant_behavioral_link_within_the_reported_bound"
    else:
        branch = "underpowered_by_construction"

    output["deciding_contrast"] = {
        "n_sessions_tested": len(tested),
        "differences_matched_correct_minus_error": differences_matched,
        "mean_difference": float(np.mean(differences_matched)) if differences_matched else None,
        "median_difference": float(np.median(differences_matched)) if differences_matched else None,
        "mean_vs_median_note": (
            "The mean can be carried by a minority of sessions in a small-n paired design; the median is "
            "reported alongside it for exactly that reason, and both are reported rather than only the mean."
        ),
        "test": deciding_test,
        "minimum_detectable_paired_difference_at_80pct_power": mdd,
        "branch": branch,
        "pairing_note": (
            "This is matched_correct_draws vs error_full_statistic_matched_params per session -- trial count "
            "AND estimator settings (n_splits/n_null_replicates/n_null_splits_per_replicate) both matched "
            "between the two cells. See confounded_unmatched_diagnostic for the trial-count-unmatched, "
            "estimator-setting-matched comparison, and parameter_setting_effect for the trial-count-matched, "
            "estimator-setting-UNmatched comparison -- the two ways this pairing could have been confounded, "
            "both ruled out."
        ),
    }
    output["confounded_unmatched_diagnostic"] = {
        "n_sessions": len(differences_confounded),
        "differences_confounded_unmatched": differences_confounded,
        "test": confounded_test,
        "note": (
            "Both cells at the FULL estimator settings, but ALL correct trials against ALL error trials -- "
            "unequal trial counts. Both r_obs and r_null move with trial count, so this diagnostic is "
            "confounded with the correct/error trial-count imbalance and is never the deciding result."
        ),
    }
    output["parameter_setting_effect_diagnostic"] = {
        "n_sessions": len(parameter_effects),
        "differences_full_minus_reduced_settings_same_error_trials": parameter_effects,
        "test": parameter_effect_test,
        "note": (
            "error_full_statistic (n_splits=%d/n_null_replicates=%d/n_null_splits_per_replicate=%d) minus "
            "error_full_statistic_matched_params (n_splits=%d/n_null_replicates=%d/"
            "n_null_splits_per_replicate=%d), on the IDENTICAL error trials in every session -- isolates how "
            "much d_perm_level moves from the split/replicate-count reduction alone, with trial identity and "
            "the correct/error label both held fixed. Answers 'how much does the parameter setting move "
            "d_perm at all' directly."
        ) % (FULL_N_SPLITS, FULL_N_NULL_REPLICATES, FULL_N_NULL_SPLITS_PER_REPLICATE,
             MATCHED_DRAW_N_SPLITS, MATCHED_DRAW_N_NULL_REPLICATES, MATCHED_DRAW_N_NULL_SPLITS_PER_REPLICATE),
    }
    output["cheap_first_look_pooled"] = {
        "leading_component_score_gain": _pool_cheap_first_look(tested, "leading_component_score_gain"),
        "total_spike_count": _pool_cheap_first_look(tested, "total_spike_count"),
        "trial_index": _pool_cheap_first_look(tested, "trial_index"),
        "scope_note": "Pooled across the same reachable, tested sessions the deciding contrast uses; "
                      "per-session values for ALL 25 sessions (including those below the reachability floor) "
                      "are in sessions[*].cheap_first_look.",
    }
    return branch


if __name__ == "__main__":
    main()
