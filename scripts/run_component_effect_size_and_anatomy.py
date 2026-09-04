"""run_component_effect_size_and_anatomy.py -- two questions about the
rate-free direction-deviation component that predicts trial outcome in
macaque prefrontal cortex, neither previously asked.

BLOCK A. The component's association with behaviour is a bare correlation
of roughly 0.10 in the single-item macaque corpus and roughly 0.02 in the
multi-object macaque corpus (DANDI 000620). A correlation coefficient in
those units is not
interpretable on its own -- this block translates it into behavioural units
a reader can picture: the decile contrast (top-decile-deviation trials
versus bottom-decile-deviation trials, in the corpus's own behavioural
units), the same contrast after matching the two deciles on total spike
count, and a cross-validated held-out single-trial discrimination measure,
run identically for the deviation, the dominant population latent's
per-trial amplitude, and total spike count, in the same folds -- because
the project's claim is that the amplitude's raw association is an artifact
of spike count, and a predictor comparison in held-out data is the
strongest available form of that claim.

BLOCK B. The single-item corpus spans three animals and, by the source
study's own account, more than one recording area. This block asks whether
the component's association with behaviour localises to one area or one
animal, or is present throughout the corpus. Every restricted cell is
gated against total spike count separately, and a cell that fails its own
gate, or that has no session reaching the reachability floor, is void by
name rather than silently dropped.

Scope: the two macaque corpora whose rate-free deviation observable passes
its own orthogonality gate against total spike count --
results/dissociation_replication_and_counting_noise.json's five-corpus
census (Block B there) establishes that this gate passes at macaque lPFC
(0.0377, p=0.589) and at the multi-object macaque corpus (-0.0016, p=0.955)
and fails at mouse ALM (-0.2057, p=0.0005) and both human corpora (-0.1704,
p=0.0285; -0.1912, p=0.0015). The mouse and human corpora are excluded here
on that already-measured precondition, not left pending.

No estimator is forked. rate_free_state_deviation, trial_amplitude_
covariates, partial_correlation_permutation_test, pearson_permutation_test,
permutation_test_twosample, auroc, minimum_detectable_paired_difference,
fdr_bh, stable_seed (src/statistics.py and the modules below), slope_across_
sessions_test (src/state_persistence.py), _analyze_session / _pool (run_
rate_free_state_geometry_behavior_link.py), reproduction_gate (run_
dissociation_cross_preparation_test.py), _observable_arrays /
_session_observable_arm / _pool_cell (run_dissociation_replication_and_
counting_noise.py), _behaviour_observables / _pool_values /
PRIMARY_QUALITY_TIER / MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION (run_watters_
state_geometry.py), _subsets (run_watters_load_decomposition.py),
_load_session / _session_paths / ANIMAL_ASSIGNMENT_SOURCE (run_dominant_
latent_identity_and_behaviour_breadth.py), iter_watters / data_root
(src/corpus_sessions.py) are all imported unchanged (_behaviour_observables
is one layer further in: it is what _observable_arrays itself calls). The
only new code this module
introduces is glue that assembles per-trial arrays already produced by the
functions above into the decile contrast, the spike-count-matched decile
contrast, the contiguous-fold cross-validated discrimination (reported both
pooled across folds and within fold), and the direct paired comparison
between predictors' cross-validated discrimination values that any ordering
claim between them requires -- none of which any existing module computes.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import data_root, iter_watters  # noqa: E402
from provenance import _json_safe, checkpoint_safe, restore_checkpoint  # noqa: E402
from run_dissociation_cross_preparation_test import reproduction_gate  # noqa: E402
from run_dissociation_replication_and_counting_noise import (  # noqa: E402
    _observable_arrays, _pool_cell, _session_observable_arm,
)
from run_dominant_latent_identity_and_behaviour_breadth import (  # noqa: E402
    ANIMAL_ASSIGNMENT_SOURCE,
    _load_session as _macaque_load_session, _session_paths as _macaque_session_paths,
)
from run_rate_free_state_geometry_behavior_link import (  # noqa: E402
    _analyze_session as _macaque_analyze_session, _pool as _macaque_pool, rate_free_state_deviation,
)
from run_state_behavior_link import (  # noqa: E402
    MIN_ERROR_TRIALS_FOR_REACHABILITY, _counts_from_spikes, _panichello_directory, trial_amplitude_covariates,
)
from run_watters_load_decomposition import _subsets  # noqa: E402
from run_watters_state_geometry import (  # noqa: E402
    MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION, PRIMARY_QUALITY_TIER, _pool_values,
)
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import (  # noqa: E402
    auroc, fdr_bh, minimum_detectable_paired_difference, partial_correlation_permutation_test,
    pearson_permutation_test, permutation_test_twosample, stable_seed,
)

from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "component_effect_size_and_anatomy.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "component_effect_size_and_anatomy_checkpoint.json"
ANALYSIS_VERSION = "2026-08-19"

N_CV_FOLDS = 5
DECILE_FRACTION = 0.10
QUINTILE_FRACTION = 0.20
PREDICTORS = ("deviation", "amplitude", "spike_count")
REPRODUCTION_TOLERANCE = 1e-6

# The multi-object corpus's own delivered primary-cell deviation values (results/
# dissociation_replication_and_counting_noise.json, block_a.results.single_and_multi_unit.pooled.deviation.
# within_item_count_level), read live below and compared to these; read once here so the comparison values
# are visible beside the code that checks them rather than only inside a read call.
WATTERS_DEVIATION_SOURCE_ARTIFACT = ROOT / "results" / "dissociation_replication_and_counting_noise.json"

WORSE_BEHAVIOUR_SIGN_MAP = {
    "macaque_lPFC_single_item": (
        "worse_behaviour = 1.0 - is_correct, i.e. the trial's own error indicator (1 = error, 0 = correct). "
        "Applied once, immediately after loading each session's raw arrays, before any decile, matching or "
        "cross-validation step -- every downstream quantity in this corpus is already in the 'higher = worse' "
        "direction."
    ),
    "macaque_multi_object": (
        "worse_behaviour = report_error, the corpus's own continuous graded deviation between the saccadic "
        "report and the cued position, already scaled so that a larger value is a worse trial. No sign flip "
        "is needed; the raw column is used unchanged, and this is stated so a reader can see that the choice "
        "was checked rather than assumed."
    ),
}

BLOCK_A_TRANSLATION_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per corpus, per session (multi-object corpus: per item-count level within session, combined into one "
    "session value by the trial-count-weighted estimator run_dissociation_replication_and_counting_noise."
    "_session_observable_arm already uses for this corpus's within-load statistics -- pooling this corpus's "
    "behavioural association across item count is already established to reverse its sign, so no quantity "
    "in this corpus's Block A output is ever pooled across item count):\n"
    "  1. Decile contrast: rank trials by the deviation observable, contrast worse_behaviour in the top "
    "decile against the bottom decile, session value pooled across sessions by the paired sign-flip test "
    "with a 95% CI. Quintile contrast reported beside it as a sensitivity check on the decile choice, not a "
    "separate decision.\n"
    "  2. Spike-count-matched decile contrast: within the same top/bottom decile trials, keep only the "
    "overlapping range of total spike count between the two groups (common support) and pair each kept "
    "top-decile trial to its nearest kept bottom-decile trial by spike count without replacement; trials "
    "outside common support or left unpaired are discarded and counted. The contrast is recomputed on the "
    "matched pairs only.\n"
    "  3. Cross-validated single-trial discrimination: 5 contiguous folds in the session's own trial order "
    "(neither corpus carries a run or block field -- verified directly: the single-item corpus's .mat files "
    "carry only {cueAng, cueAngIdx, isCorr, spks, tc}, and the multi-object corpus's behavioural CSV carries "
    "no run/block column -- so a random split is not used; contiguous chronological blocks are used instead, "
    "the specified fallback for a corpus with no run structure). A single-feature model (logistic regression for the corpus scored "
    "correct/incorrect, linear regression for the corpus with a continuous graded report) is fit on the "
    "training folds and evaluated out-of-fold; the discrimination statistic is held-out AUROC minus 0.5 for "
    "the binary corpus and the held-out Pearson correlation for the continuous corpus, so both are centred "
    "on a chance value of zero. Run identically, in the identical folds, for the deviation, the dominant "
    "latent's per-trial amplitude (trial_amplitude_covariates), and total spike count.\n"
    "  4. Within-session and between-session associations are reported as two separate numbers and never "
    "pooled: within-session is the same raw deviation-versus-worse_behaviour correlation used above; "
    "between-session is a single correlation of each session's own mean deviation against its own mean "
    "worse_behaviour, across sessions (per item-count level for the multi-object corpus, never combined "
    "across levels).\n"
    "Branches, checked in this order on the cross-validated discrimination result:\n"
    "  - Held-out discrimination above chance (pooled sign-flip test positive and significant) for the "
    "deviation and NOT for the amplitude -> "
    "'accuracy_predicting_component_carries_held_out_single_trial_information_and_the_dominant_amplitude_does_not'.\n"
    "  - Above chance for both -> 'both_observables_carry_held_out_information', reported with both "
    "intervals and the statement that this alone does not separate them, because the amplitude's association "
    "is diagnosed by its behaviour under a spike-count control, not by its magnitude.\n"
    "  - Above chance for neither -> 'no_observable_reaches_held_out_single_trial_discrimination_at_this_power', "
    "reported with the minimum discrimination detectable at 80% power for each observable.\n"
    "  - Above chance for the amplitude only -> 'dominant_amplitude_outpredicts_the_component_in_held_out_data', "
    "reported prominently and without reinterpretation.\n"
    "  - If the within-session and between-session associations disagree in sign, both are reported, the "
    "within-session one is named the commensurable one, and neither is pooled with the other."
)

BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Only where the corpus carries an area label. The multi-object corpus's shared electrode table gives "
    "every recorded channel the literal location 'unknown' (verified against two independent statements "
    "already on disk: src/corpus_sessions.py's own _watters_unit_index docstring and scripts/build_"
    "anatomical_census.py's electrode-level census note), so it supports one pooled population and no "
    "area-resolved split at all -- refused by name, not attempted. Its animal split is required and is run.\n"
    "The single-item corpus's area assignment is per animal, not per unit or per channel (verified against "
    "run_dominant_latent_identity_and_behaviour_breadth.py's own ANIMAL_ASSIGNMENT_SOURCE and against "
    "scripts/build_structure_registry.py's single-item-corpus census note: no staged file carries a channel-level "
    "area field). Monkey A is area 8, monkey J is area 9/46, and monkey H is undetermined between the two "
    "areas -- monkey H's sessions are therefore excluded from the area split by name (not voided by a gate) "
    "and remain in the animal split, which covers all three animals.\n"
    "Per restricted group: recompute the deviation observable's orthogonality gate against total spike "
    "count on every session in the group with a defined direction, then recompute the raw deviation-versus-"
    "worse_behaviour association restricted further to the group's sessions reaching the reachability floor "
    "(the same floor -- at least "
    f"{MIN_ERROR_TRIALS_FOR_REACHABILITY} error trials -- used everywhere else in this corpus's behavioural "
    "line; the multi-object corpus needs no such floor, its report is continuous). A group whose pooled gate "
    "is significant is void by the gate; a group with zero sessions reaching the reachability floor is void "
    "by reachability; a group with 1-3 reachable sessions is reported with its own point estimate but its "
    "pooled sign-flip test is underpowered by construction (fewer than 4 units) and is reported as such, "
    "distinct from a gate void. No threshold spike count is asserted as a cutoff for trustworthiness; each "
    "group's median total spike count per trial is reported beside its gate so a reader can see how close it "
    "sits to the corpus's own overall value.\n"
    "  - Every restricted cell void (by gate or by zero reachable sessions, none merely underpowered) -> "
    "'anatomical_localisation_unreachable_at_this_unit_count_per_area', with the counts and gates that made "
    "it so.\n"
    "  - No cell reaches a computed status, but at least one is 'underpowered_by_construction' (reachable, "
    "too few sessions for the pooled test) rather than void -> 'no_group_reaches_a_powered_behavioural_"
    "association_some_cells_reachable_but_underpowered', reported with every cell's own numbers, distinct "
    "from the unreachable branch above because a group here genuinely has reachable sessions.\n"
    "  - Some cells void or underpowered, at least one computed -> report the surviving (computed) cells, "
    "name the void or underpowered ones and why, never compare a surviving cell against a void or "
    "underpowered one.\n"
    "  - Exactly one cell survives (reaches a computed, non-underpowered pooled behaviour association) -> no "
    "cross-group comparison is possible; this is reported as its own outcome, "
    "'only_one_group_survives_no_cross_group_comparison_possible', not forced onto either of the two "
    "remaining named branches below.\n"
    "  - Two or more cells survive, association present in every one, no significant direct two-sample "
    "permutation test between any pair of them (permutation_test_twosample on the groups' own per-session "
    "raw correlation values) -> 'accuracy_predicting_component_is_not_localised_to_one_recorded_area' (or "
    "'..._one_animal' for the animal split), reported with the minimum difference detectable at 80% power.\n"
    "  - A significant direct two-sample test between two surviving cells -> "
    "'accuracy_predicting_component_is_stronger_in_one_area' (or '..._one_animal'), naming the cell, with "
    "the observed difference and its permutation p-value.\n"
    "The animal split is never quoted without the session count behind each animal."
)


# =======================================================================================================
# Checkpointing (fit-level; temp file + os.replace; completion flag written only after the fit returns)
# =======================================================================================================

_COMPLETED_FITS: dict[str, dict] = {}
_FITS_SERVED_FROM_CHECKPOINT = 0
_FITS_COMPUTED_HERE = 0


def _load_completed_fits() -> dict[str, dict]:
    try:
        entries = json.loads(CHECKPOINT_PATH.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(entries, dict):
        return {}
    return {key: {**entry, "value": restore_checkpoint(entry["value"])} for key, entry in entries.items()
            if isinstance(entry, dict) and entry.get("complete") is True}


def _fit(key: str, compute) -> dict:
    global _FITS_SERVED_FROM_CHECKPOINT, _FITS_COMPUTED_HERE
    entry = _COMPLETED_FITS.get(key)
    if entry is not None:
        _FITS_SERVED_FROM_CHECKPOINT += 1
        return entry["value"]
    value = compute()
    _COMPLETED_FITS[key] = {"complete": True, "value": value}
    _FITS_COMPUTED_HERE += 1
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = CHECKPOINT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(checkpoint_safe(_COMPLETED_FITS), allow_nan=False, default=float))
    os.replace(scratch, CHECKPOINT_PATH)
    return value


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


# =======================================================================================================
# Generic decile / matched-decile / cross-validated-discrimination machinery (new: no existing estimator
# in this project computes any of the three)
# =======================================================================================================

def _decile_contrast(deviation: np.ndarray, worse: np.ndarray, fraction: float) -> dict:
    n = len(deviation)
    k = int(round(n * fraction))
    if k < 2 or 2 * k > n:
        return {"status": "too_few_trials", "n_trials": n, "fraction": fraction}
    order = np.argsort(deviation)
    bottom_idx, top_idx = order[:k], order[-k:]
    return {
        "status": "computed", "fraction": fraction, "n_per_group": k,
        "contrast": float(worse[top_idx].mean() - worse[bottom_idx].mean()),
        "top_idx": top_idx, "bottom_idx": bottom_idx,
    }


def _spike_count_matched_contrast(deviation: np.ndarray, worse: np.ndarray, spike_count: np.ndarray,
                                   fraction: float) -> dict:
    base = _decile_contrast(deviation, worse, fraction)
    if base["status"] != "computed":
        return {"status": base["status"], "n_trials": base["n_trials"], "fraction": fraction}
    top_idx, bottom_idx = base["top_idx"], base["bottom_idx"]
    top_sc, bot_sc = spike_count[top_idx], spike_count[bottom_idx]
    lo, hi = max(top_sc.min(), bot_sc.min()), min(top_sc.max(), bot_sc.max())
    if lo > hi:
        return {"status": "no_common_support", "fraction": fraction,
                "n_top_before_matching": len(top_idx), "n_bottom_before_matching": len(bottom_idx)}
    top_pool = [int(i) for i in top_idx if lo <= spike_count[i] <= hi]
    bot_pool = [int(i) for i in bottom_idx if lo <= spike_count[i] <= hi]
    remaining = list(bot_pool)
    pairs: list[tuple[int, int]] = []
    for i in sorted(top_pool, key=lambda j: spike_count[j]):
        if not remaining:
            break
        j = min(remaining, key=lambda b: abs(spike_count[b] - spike_count[i]))
        pairs.append((i, j))
        remaining.remove(j)
    if len(pairs) < 2:
        return {"status": "too_few_matched_pairs", "fraction": fraction, "n_matched_pairs": len(pairs),
                "n_top_before_matching": len(top_idx), "n_bottom_before_matching": len(bottom_idx)}
    matched_top = np.array([p[0] for p in pairs])
    matched_bottom = np.array([p[1] for p in pairs])
    return {
        "status": "computed", "fraction": fraction, "n_matched_pairs": len(pairs),
        "n_top_before_matching": len(top_idx), "n_bottom_before_matching": len(bottom_idx),
        "n_top_discarded": len(top_idx) - len(pairs), "n_bottom_discarded": len(bottom_idx) - len(pairs),
        "matched_top_spike_count_mean": float(spike_count[matched_top].mean()),
        "matched_bottom_spike_count_mean": float(spike_count[matched_bottom].mean()),
        "contrast": float(worse[matched_top].mean() - worse[matched_bottom].mean()),
    }


def _contiguous_folds(n: int, k: int) -> np.ndarray:
    edges = np.linspace(0, n, k + 1).astype(int)
    fold = np.empty(n, dtype=int)
    for f in range(k):
        fold[edges[f]:edges[f + 1]] = f
    return fold


def _cv_discrimination_binary(feature: np.ndarray, y: np.ndarray, k: int = N_CV_FOLDS) -> dict:
    n = len(feature)
    if n < 2 * k:
        return {"status": "too_few_trials", "n_trials": n}
    folds = _contiguous_folds(n, k)
    oof = np.full(n, np.nan)
    per_fold_pairs: list[tuple[int, float]] = []
    for f in range(k):
        test, train = folds == f, folds != f
        if len(np.unique(y[train])) < 2 or not test.any():
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(feature[train].reshape(-1, 1), y[train])
        oof[test] = clf.predict_proba(feature[test].reshape(-1, 1))[:, 1]
        if int(test.sum()) >= 4 and len(np.unique(y[test])) >= 2:
            per_fold_pairs.append((int(test.sum()), auroc(y[test], oof[test])))
    valid = np.isfinite(oof)
    if valid.sum() < 4 or len(np.unique(y[valid])) < 2:
        return {"status": "not_computable", "n_trials": n}
    return {"status": "computed", "n_trials": n, "n_folds": k, "n_valid": int(valid.sum()),
            "auc": auroc(y[valid], oof[valid]),
            "auc_per_fold_not_pooled_across_folds": _trial_count_weighted(per_fold_pairs),
            "n_folds_with_a_computable_per_fold_statistic": len(per_fold_pairs)}


def _cv_discrimination_continuous(feature: np.ndarray, target: np.ndarray, k: int = N_CV_FOLDS) -> dict:
    n = len(feature)
    if n < 2 * k:
        return {"status": "too_few_trials", "n_trials": n}
    folds = _contiguous_folds(n, k)
    oof = np.full(n, np.nan)
    per_fold_pairs: list[tuple[int, float]] = []
    for f in range(k):
        test, train = folds == f, folds != f
        if train.sum() < 2 or not test.any():
            continue
        reg = LinearRegression()
        reg.fit(feature[train].reshape(-1, 1), target[train])
        oof[test] = reg.predict(feature[test].reshape(-1, 1))
        if int(test.sum()) >= 4 and np.std(oof[test]) > 0.0 and np.std(target[test]) > 0.0:
            per_fold_pairs.append((int(test.sum()), float(np.corrcoef(oof[test], target[test])[0, 1])))
    valid = np.isfinite(oof)
    if valid.sum() < 4 or np.std(oof[valid]) == 0.0 or np.std(target[valid]) == 0.0:
        return {"status": "not_computable", "n_trials": n}
    r = float(np.corrcoef(oof[valid], target[valid])[0, 1])
    return {"status": "computed", "n_trials": n, "n_folds": k, "n_valid": int(valid.sum()), "r": r,
            "r_per_fold_not_pooled_across_folds": _trial_count_weighted(per_fold_pairs),
            "n_folds_with_a_computable_per_fold_statistic": len(per_fold_pairs)}


def _trial_count_weighted(entries: list[tuple[int, float | None]]) -> float | None:
    tested = [(n, v) for n, v in entries if v is not None]
    if not tested:
        return None
    n_arr = np.array([n for n, _ in tested], dtype=float)
    v_arr = np.array([v for _, v in tested], dtype=float)
    return float(np.sum((n_arr / n_arr.sum()) * v_arr))


# =======================================================================================================
# Block A -- one corpus's per-session translation record
# =======================================================================================================

def _session_block_a(session_id: str, arrays: dict, is_binary_outcome: bool) -> dict:
    """arrays carries deviation, amplitude, spike_count, worse_behaviour, and (multi-object corpus only)
    item_count. Every quantity is computed per item-count level when item_count is present and combined
    across levels by the trial-count-weighted estimator; never pooled across level any other way."""
    if "item_count" in arrays:
        item_count = arrays["item_count"]
        levels = sorted({int(v) for v in item_count.tolist()})
        groups: list[tuple[str, int, dict | None]] = []
        for lv in levels:
            mask = item_count == float(lv)
            n = int(mask.sum())
            sub = ({k: v[mask] for k, v in arrays.items() if k != "item_count"}
                   if n >= MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION else None)
            groups.append((str(lv), n, sub))
    else:
        groups = [("all", len(arrays["deviation"]), arrays)]

    per_level: dict[str, dict] = {}
    decile_pairs, quintile_pairs, matched_pairs = [], [], []
    cv_pairs: dict[str, list[tuple[int, float]]] = {p: [] for p in PREDICTORS}
    cv_raw_pairs: dict[str, list[tuple[int, float]]] = {p: [] for p in PREDICTORS}
    cv_per_fold_pairs: dict[str, list[tuple[int, float]]] = {p: [] for p in PREDICTORS}

    for label, n, sub in groups:
        if sub is None:
            per_level[label] = {"status": "too_few_trials", "n_trials": n}
            continue
        deviation, worse, spike_count = sub["deviation"], sub["worse_behaviour"], sub["spike_count"]
        decile = _decile_contrast(deviation, worse, DECILE_FRACTION)
        quintile = _decile_contrast(deviation, worse, QUINTILE_FRACTION)
        matched = _spike_count_matched_contrast(deviation, worse, spike_count, DECILE_FRACTION)
        cv: dict[str, dict] = {}
        for predictor in PREDICTORS:
            feature = sub[predictor]
            if is_binary_outcome:
                cv[predictor] = _cv_discrimination_binary(feature, worse.astype(int))
            else:
                cv[predictor] = _cv_discrimination_continuous(feature, worse)

        per_level[label] = {
            "status": "computed", "n_trials": n,
            "decile_contrast": {k: v for k, v in decile.items() if k not in ("top_idx", "bottom_idx")},
            "quintile_contrast": {k: v for k, v in quintile.items() if k not in ("top_idx", "bottom_idx")},
            "spike_count_matched_decile_contrast": matched,
            "cross_validated_discrimination": cv,
        }
        if decile["status"] == "computed":
            decile_pairs.append((n, decile["contrast"]))
        if quintile["status"] == "computed":
            quintile_pairs.append((n, quintile["contrast"]))
        if matched["status"] == "computed":
            matched_pairs.append((n, matched["contrast"]))
        for predictor in PREDICTORS:
            c = cv[predictor]
            if c["status"] != "computed":
                continue
            raw_value = c["auc"] if is_binary_outcome else c["r"]
            centred_value = (c["auc"] - 0.5) if is_binary_outcome else c["r"]
            cv_pairs[predictor].append((n, centred_value))
            cv_raw_pairs[predictor].append((n, raw_value))
            per_fold_raw = c["auc_per_fold_not_pooled_across_folds"] if is_binary_outcome \
                else c["r_per_fold_not_pooled_across_folds"]
            if per_fold_raw is not None:
                per_fold_centred = (per_fold_raw - 0.5) if is_binary_outcome else per_fold_raw
                cv_per_fold_pairs[predictor].append((n, per_fold_centred))

    combined = {
        "decile_contrast": _trial_count_weighted(decile_pairs),
        "quintile_contrast": _trial_count_weighted(quintile_pairs),
        "spike_count_matched_decile_contrast": _trial_count_weighted(matched_pairs),
        "cross_validated_discrimination_centred_on_chance": {
            p: _trial_count_weighted(cv_pairs[p]) for p in PREDICTORS},
        "cross_validated_discrimination_raw": {p: _trial_count_weighted(cv_raw_pairs[p]) for p in PREDICTORS},
        "cross_validated_discrimination_per_fold_not_pooled_across_folds": {
            p: _trial_count_weighted(cv_per_fold_pairs[p]) for p in PREDICTORS},
    }
    return {"session": session_id, "n_levels": len(groups), "per_level": per_level, "combined": combined}


def _pool_block_a_combined(session_records: list[dict], key: str, sub: str | None = None) -> dict:
    values = []
    for r in session_records:
        v = r["combined"][key] if sub is None else r["combined"][key][sub]
        if v is not None:
            values.append(v)
    return _pool_values(values)


def _above_chance(pooled: dict) -> bool:
    return bool(pooled.get("status") == "tested" and pooled.get("significant_positive"))


CROSS_VALIDATED_DISCRIMINATION_POOLED_VS_PER_FOLD_NOTE = (
    "Two versions of the held-out discrimination statistic are reported for every predictor in every "
    "corpus. The across-block version pools every fold's out-of-fold prediction into one vector before "
    "computing AUROC or Pearson r; because each fold's model carries its own intercept and slope, this "
    "pooled statistic can absorb between-fold (between-chronological-block) variation on top of whatever "
    "within-fold signal the predictor carries, and a session-time drift can dominate or invert it. The "
    "within-block version computes the statistic separately inside each fold, on that fold's own held-out "
    "trials only, then combines the per-fold values into one session value by trial-count weighting -- it "
    "cannot pick up any between-fold offset because no fold's statistic is ever compared against another "
    "fold's predictions. Neither version is treated as the corrected one; they answer different questions "
    "and are reported side by side. The branch decision below is fixed to the across-block version, "
    "matching the decision rule declared before fitting; the within-block version is reported as its own "
    "named sensitivity and never substituted into that decision."
)


def _matched_contrast_reachability(decile_pooled: dict, matched_pooled: dict) -> dict:
    """Whether the spike-count-matched decile contrast was ever powered to detect the effect size the
    unmatched decile contrast shows. A non-significant matched contrast whose own minimum detectable
    difference at 80% power exceeds the unmatched effect it would have to detect sits below its own
    detection floor -- that is a reachability statement, not evidence the association fails to survive
    spike-count matching, and the two must not be conflated."""
    if decile_pooled.get("status") != "tested" or matched_pooled.get("status") != "tested":
        return {"status": "not_computable"}
    mdd_block = matched_pooled.get("minimum_detectable_paired_difference_at_80pct_power", {})
    if mdd_block.get("status") != "computed":
        return {"status": "not_computable"}
    unmatched_value = decile_pooled["mean_value"]
    matched_value = matched_pooled["mean_value"]
    mdd = mdd_block["mdd"]
    powered_to_detect_unmatched_effect = bool(abs(unmatched_value) >= mdd)
    return {
        "status": "computed",
        "unmatched_decile_contrast": unmatched_value,
        "matched_decile_contrast": matched_value,
        "matched_minimum_detectable_difference_at_80pct_power": mdd,
        "matched_comparison_is_powered_to_detect_the_unmatched_effect_size": powered_to_detect_unmatched_effect,
        "reading": (
            "the matched comparison is powered to detect an effect at least as small as the unmatched "
            "contrast, so its own significance speaks to whether the association survives spike-count "
            "matching"
            if powered_to_detect_unmatched_effect else
            "the matched comparison's minimum detectable difference at 80% power exceeds the unmatched "
            "contrast it would have to detect; a non-significant matched contrast here is below its own "
            "detection floor, not evidence that the association fails to survive spike-count matching"
        ),
    }


def _block_a_cv_branch(deviation_pooled: dict, amplitude_pooled: dict) -> str:
    dev_above, amp_above = _above_chance(deviation_pooled), _above_chance(amplitude_pooled)
    if dev_above and not amp_above:
        return ("accuracy_predicting_component_carries_held_out_single_trial_information_and_the_dominant_"
                "amplitude_does_not")
    if dev_above and amp_above:
        return "both_observables_carry_held_out_information"
    if amp_above and not dev_above:
        return "dominant_amplitude_outpredicts_the_component_in_held_out_data"
    return "no_observable_reaches_held_out_single_trial_discrimination_at_this_power"


PAIRWISE_PREDICTOR_COMPARISONS = (
    ("deviation", "amplitude"), ("deviation", "spike_count"), ("amplitude", "spike_count"),
)

# The two fired branches that state an ordering between two predictors' held-out discrimination, and
# which pair of predictors that ordering is about. The other two possible branches (both above chance;
# neither above chance) make no claim about which predictor beats which, so no pair is listed for them --
# a direct paired test is required to support an ordering claim, not to support the absence of one.
ORDERING_BRANCHES = {
    "dominant_amplitude_outpredicts_the_component_in_held_out_data": ("deviation", "amplitude"),
    "accuracy_predicting_component_carries_held_out_single_trial_information_and_the_dominant_"
    "amplitude_does_not": ("deviation", "amplitude"),
}


def _paired_predictor_diffs(session_records: list[dict], key: str, a: str, b: str) -> list[float]:
    """Per-session (a - b) on the sessions where both predictors' combined value is defined under the
    named cross-validated-discrimination key -- the same per-session values already pooled separately for
    each predictor, paired within session rather than compared across separately pooled means."""
    diffs = []
    for r in session_records:
        va, vb = r["combined"][key].get(a), r["combined"][key].get(b)
        if va is not None and vb is not None:
            diffs.append(va - vb)
    return diffs


def _pairwise_cell_reachability(cell: dict) -> dict:
    """Whether a non-significant pairwise difference was ever powered to detect the very difference it
    observed. A non-significant cell whose own minimum detectable difference at 80% power exceeds the
    absolute observed mean difference sits below its own detection floor, not a confirmed absence of
    ordering; a non-significant cell whose minimum detectable difference is at or below the observed
    difference is a powered null."""
    if cell.get("status") != "tested" or cell.get("significant"):
        return {"status": "not_applicable"}
    mdd_block = cell.get("minimum_detectable_paired_difference_at_80pct_power", {})
    if mdd_block.get("status") != "computed":
        return {"status": "not_computable"}
    observed = abs(cell["mean_value"])
    mdd = mdd_block["mdd"]
    powered = bool(mdd <= observed)
    return {
        "status": "computed",
        "observed_absolute_mean_difference": observed,
        "minimum_detectable_difference_at_80pct_power": mdd,
        "label": "powered_null" if powered else "below_its_own_detection_floor",
        "reading": (
            "the minimum detectable difference at 80% power is at or below the observed difference, so "
            "the non-significant result is a powered null"
            if powered else
            "the minimum detectable difference at 80% power exceeds the observed difference, so the "
            "non-significant result is below its own detection floor, not a confirmed absence of ordering"
        ),
    }


def _pairwise_predictor_tests(session_records: list[dict], key: str) -> dict:
    """Direct paired session-level tests between every pair of the three predictors' cross-validated
    discrimination values, under one estimator version (identified by `key`, either the pooled-across-fold
    or the within-fold combined value already carried per session), by the same two-sided paired sign-flip
    estimator used for every other pooled quantity in this artifact. Corrected across the three comparisons
    by Benjamini-Hochberg at alpha 0.05 on the raw two-sided p-values; raw and corrected are both reported.
    Two significance verdicts computed separately for two predictors are never treated as a difference
    between them -- only this direct test is."""
    cells = {f"{a}_minus_{b}": _pool_values(_paired_predictor_diffs(session_records, key, a, b))
             for a, b in PAIRWISE_PREDICTOR_COMPARISONS}
    tested_labels = [label for label, cell in cells.items() if cell.get("status") == "tested"]
    if tested_labels:
        correction = fdr_bh(np.array([cells[label]["two_sided_p_value"] for label in tested_labels]))
        for label, q, reject in zip(tested_labels, correction["q_values"], correction["reject"]):
            cells[label]["benjamini_hochberg_q_value"] = float(q)
            cells[label]["significant_after_benjamini_hochberg_correction"] = bool(reject)
    for cell in cells.values():
        cell["reachability"] = _pairwise_cell_reachability(cell)
    return cells


def _pairwise_comparison_note(branch: str, across_block: dict, within_block: dict) -> str:
    base = (
        "The branch decision above was fixed, before any number here was computed, to the across-block "
        "(pooled-across-fold) cross-validated discrimination version, and it stays exactly as fired "
        "regardless of what the pairwise tests below show; adding them changes no pre-declared decision "
        "rule and moves no branch."
    )
    pair = ORDERING_BRANCHES.get(branch)
    if pair is None:
        return base + (
            " This corpus's fired branch does not itself state which predictor outpredicts which, so it "
            "makes no ordering claim for a direct paired test to confirm or contradict; the pairwise "
            "comparisons below are reported for completeness, not because the branch requires them."
        )
    a, b = pair
    label = f"{a}_minus_{b}"
    across_cell, within_cell = across_block.get(label, {}), within_block.get(label, {})
    across_sig, within_sig = bool(across_cell.get("significant")), bool(within_cell.get("significant"))
    return base + (
        f" The fired branch states an ordering between {a} and {b}. A direct paired session-level test "
        f"between them ({label}) is "
        f"{'significant' if across_sig else 'not significant'} under the across-block version (mean "
        f"difference {across_cell.get('mean_value')}, two-sided p={across_cell.get('two_sided_p_value')}) "
        f"and is {'significant' if within_sig else 'not significant'} under the within-block version (mean "
        f"difference {within_cell.get('mean_value')}, two-sided p={within_cell.get('two_sided_p_value')}). "
        f"The ordering is therefore "
        f"{'established' if across_sig else 'not established'} by a direct paired test under the "
        f"across-block version, which is the version the fired branch decision is fixed to, and is "
        f"{'established' if within_sig else 'not established'} by a direct paired test under the "
        f"within-block version."
    )


def _sign(value: float | None) -> int | None:
    if value is None:
        return None
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def run_block_a(corpus_key: str, session_records: list[dict], within_session: dict, between_session,
                 reference: dict) -> dict:
    pooled = {
        "decile_contrast": _pool_block_a_combined(session_records, "decile_contrast"),
        "quintile_contrast_sensitivity": _pool_block_a_combined(session_records, "quintile_contrast"),
        "spike_count_matched_decile_contrast": _pool_block_a_combined(
            session_records, "spike_count_matched_decile_contrast"),
        "cross_validated_discrimination": {
            p: {
                "centred_on_chance": _pool_block_a_combined(
                    session_records, "cross_validated_discrimination_centred_on_chance", p),
                "raw": _pool_block_a_combined(session_records, "cross_validated_discrimination_raw", p),
                "per_fold_not_pooled_across_folds": _pool_block_a_combined(
                    session_records, "cross_validated_discrimination_per_fold_not_pooled_across_folds", p),
            } for p in PREDICTORS
        },
        "cross_validated_discrimination_pooled_vs_per_fold_note": CROSS_VALIDATED_DISCRIMINATION_POOLED_VS_PER_FOLD_NOTE,
    }
    branch = _block_a_cv_branch(
        pooled["cross_validated_discrimination"]["deviation"]["centred_on_chance"],
        pooled["cross_validated_discrimination"]["amplitude"]["centred_on_chance"],
    )
    pooled["spike_count_matched_decile_contrast_reachability"] = _matched_contrast_reachability(
        pooled["decile_contrast"], pooled["spike_count_matched_decile_contrast"])

    pairwise_across = _pairwise_predictor_tests(session_records, "cross_validated_discrimination_centred_on_chance")
    pairwise_within = _pairwise_predictor_tests(
        session_records, "cross_validated_discrimination_per_fold_not_pooled_across_folds")
    pooled["pairwise_predictor_comparisons"] = {
        "across_block": pairwise_across,
        "within_block": pairwise_within,
        "significant_under_both_estimator_versions": [
            label for label in pairwise_across
            if pairwise_across[label].get("significant") and pairwise_within.get(label, {}).get("significant")
        ],
        "note": _pairwise_comparison_note(branch, pairwise_across, pairwise_within),
    }

    within_sign = _sign(within_session.get("mean_value"))
    between_sign = None
    between_disagreement = None
    if isinstance(between_session, dict):
        between_sign = _sign(between_session.get("r"))
        if within_sign is not None and between_sign is not None and within_sign != between_sign and \
                within_sign != 0 and between_sign != 0:
            between_disagreement = (
                "within-session and between-session associations disagree in sign; the within-session "
                "association is the commensurable one, and the two are never pooled with each other."
            )
    elif isinstance(between_session, list):
        signs = [_sign(row.get("between_session", {}).get("r")) for row in between_session
                 if row.get("between_session", {}).get("status") == "computed"]
        signs = [s for s in signs if s not in (None, 0)]
        if within_sign not in (None, 0) and any(s != within_sign for s in signs):
            between_disagreement = (
                "at least one item-count level's between-session association disagrees in sign with the "
                "within-session association; the within-session association is the commensurable one, and "
                "levels are never pooled with each other or with the within-session number."
            )

    return {
        "corpus": corpus_key,
        "n_sessions": len(session_records),
        "reference_effect_size": reference,
        "per_session": session_records,
        "pooled": pooled,
        "within_session_association": within_session,
        "between_session_association": between_session,
        "within_versus_between_sign_disagreement": between_disagreement,
        "branch": branch,
    }


# =======================================================================================================
# Single-item macaque corpus -- loading, reproduction, Block A, Block B
# =======================================================================================================

def _macaque_full_session(session: dict) -> dict | None:
    """session: run_dominant_latent_identity_and_behaviour_breadth._load_session's own dict (counts,
    is_corr, animal, date_block, n_error, session). Adds the deviation observable, the dominant latent's
    per-trial amplitude, and the worse_behaviour transform, restricted to the identical finite-direction
    mask rate_free_state_deviation implies -- rate_free_state_deviation and trial_amplitude_covariates are
    both called unchanged; only the field assembly is new."""
    counts = session["counts"]
    if counts.shape[0] < 16:
        return None
    activity = counts.sum(axis=2)
    deviation = rate_free_state_deviation(activity)
    finite = np.isfinite(deviation)
    if int(finite.sum()) < 16:
        return None
    covariates = trial_amplitude_covariates(counts)
    if covariates["status"] != "computed":
        return None
    amplitude = np.asarray(covariates["leading_component_score_gain"], dtype=float)
    is_corr = session["is_corr"].astype(float)
    return {
        "session": session["session"], "animal": session["animal"], "date_block": session["date_block"],
        "n_error": session["n_error"], "n_trials_total": session["n_trials"],
        "n_trials_with_defined_direction": int(finite.sum()),
        "is_corr": is_corr[finite],
        "arrays": {
            "deviation": deviation[finite], "amplitude": amplitude[finite],
            "spike_count": activity.sum(axis=1).astype(float)[finite],
            "worse_behaviour": 1.0 - is_corr[finite],
        },
    }


def _macaque_analyze(full: dict) -> dict:
    """The reused gate/raw/partial estimator (_macaque_analyze_session), called on every session with a
    defined direction regardless of reachability -- reachability only gates which sessions enter the
    BEHAVIOUR pooling, not the gate pooling."""
    arrays_for_estimator = {
        "is_corr": full["is_corr"], "deviation": full["arrays"]["deviation"],
        "spike_count": full["arrays"]["spike_count"],
        "trial_index": np.arange(full["n_trials_with_defined_direction"], dtype=float),
        "n_trials_total": full["n_trials_total"],
        "n_trials_with_defined_direction": full["n_trials_with_defined_direction"],
    }
    return _macaque_analyze_session(full["session"], arrays_for_estimator)


def load_macaque_corpus(root) -> list[dict]:
    rows = []
    for path in _macaque_session_paths(root):
        session = _macaque_load_session(path)
        full = _macaque_full_session(session)
        if full is None:
            rows.append({"session": session["session"], "animal": session["animal"], "status": "not_computable"})
            continue
        full["analysis"] = _fit(f"macaque_session|{full['session']}", lambda f=full: _macaque_analyze(f))
        full["status"] = "computed"
        rows.append(full)
    return rows


def macaque_reproduction(root) -> dict:
    return _fit("macaque_reproduction_gate", lambda: reproduction_gate(root))


def macaque_block_a(rows: list[dict]) -> dict:
    computed = [r for r in rows if r["status"] == "computed"]
    reachable = [r for r in computed if r["n_error"] >= MIN_ERROR_TRIALS_FOR_REACHABILITY]
    session_records = [_session_block_a(r["session"], r["arrays"], is_binary_outcome=True) for r in reachable]

    # within-session: the same raw deviation-vs-is_corr correlation _macaque_analyze_session already
    # produced, sign-flipped once to the worse_behaviour convention (worse_behaviour = 1 - is_corr, so its
    # Pearson r with deviation is exactly the negative of deviation's r with is_corr; the p-value and CI
    # magnitudes are unchanged, only the sign and the CI bounds flip).
    flipped = []
    for r in reachable:
        raw = r["analysis"]["raw_outcome_vs_deviation"]
        if raw.get("status") != "computed":
            continue
        flipped.append(-raw["r"])
    within_session = _pool_values(flipped)

    session_means = []
    for r in reachable:
        session_means.append((float(np.mean(r["arrays"]["deviation"])), float(np.mean(r["arrays"]["worse_behaviour"]))))
    between_session: dict
    if len(session_means) >= 4:
        dev_means = np.array([m[0] for m in session_means])
        worse_means = np.array([m[1] for m in session_means])
        rng = np.random.default_rng(stable_seed("component_effect_size_and_anatomy|macaque|between_session"))
        between_session = pearson_permutation_test(dev_means, worse_means, n_perm=10000, rng=rng)
        between_session["status"] = "computed"
    else:
        between_session = {"status": "underpowered_by_construction", "n_sessions": len(session_means)}

    reference = {
        "raw_association_r_units": None,
        "note": (
            "The single-item corpus's own already-delivered raw association (results/rate_free_state_"
            "geometry_behavior_link.json) is reproduced above as part of the reproduction gate rather than "
            "re-quoted here; see the top-level reproduction block for its exact value."
        ),
    }
    return run_block_a("macaque_lPFC_single_item", session_records, within_session, between_session, reference)


# =======================================================================================================
# Multi-object macaque corpus -- loading, reproduction, Block A, Block B
# =======================================================================================================

def load_watters_corpus(root) -> tuple[int, list[dict], list[dict]]:
    seen, loaded, refused = 0, [], []
    for session in iter_watters(root, bin_ms=100.0):
        seen += 1
        if session["status"] != "loaded":
            refused.append({"session": session["session"], "animal": session.get("animal"), "status": session["status"]})
            continue
        loaded.append(session)
    return seen, loaded, refused


def _watters_full_session(session: dict) -> dict | None:
    counts = session["counts"]
    arrays, excluded, usable = _observable_arrays(counts, session)
    if arrays is None:
        return None
    worse = arrays["report_error"]
    full_arrays = {
        "deviation": arrays["deviation"], "amplitude": arrays["amplitude"], "spike_count": arrays["spike_count"],
        "worse_behaviour": worse, "item_count": arrays["item_count"],
    }
    return {
        "session": session["session"], "animal": session["animal"], "task_variant": session["task_variant"],
        "n_trials_usable": int(usable.sum()), "trials_excluded_by_reason": excluded, "arrays": full_arrays,
    }


def _watters_deviation_arm(session: dict) -> dict:
    """The one already-existing correlation family this module needs from run_dissociation_replication_and_
    counting_noise.py, called unchanged, restricted to the primary unit-quality tier only (a unit-quality-
    tier sensitivity sweep is out of scope here) -- gives the gate, the raw association and every
    partial, pooled within item-count level exactly as the delivered artifact computes them, and is reused
    for the reproduction check, Block A's within-session number and Block B's gate/behaviour pooling all at
    once rather than recomputed three times."""
    counts = session["counts"]
    arrays, _excluded, usable = _observable_arrays(counts, session)
    if arrays is None:
        return {"status": "too_few_trials_with_a_defined_state_direction_and_report", "n_trials_usable": int(usable.sum())}
    # This exact tag string -- not a name of this module's own choosing -- is required for a bit-exact
    # reproduction: _session_observable_arm's own permutation p-values depend on the seed derived from this
    # string (the observed r does not), and results/dissociation_replication_and_counting_noise.json's
    # _analyse_watters_session_for_block_a built its per-session tag from its OWN module name, not this
    # one's. Reusing that literal tag is what makes the reproduction gate below exact rather than merely
    # close.
    tag = f"dissociation_replication_and_counting_noise|watters|{session['session']}|{PRIMARY_QUALITY_TIER}"
    return {"status": "computed", "arm": _session_observable_arm(arrays, "deviation", tag)}


def load_watters_deviation_arms(loaded: list[dict]) -> list[dict]:
    rows = []
    for session in loaded:
        arm = _fit(f"watters_deviation_arm|{session['session']}", lambda s=session: _watters_deviation_arm(s))
        rows.append({"session": session["session"], "animal": session["animal"],
                      "task_variant": session["task_variant"], **arm})
    return rows


def watters_reproduction(deviation_arms: list[dict]) -> dict:
    rows = [{"by_tier": {PRIMARY_QUALITY_TIER: {"status": "computed", "deviation": r["arm"]}}}
            for r in deviation_arms if r["status"] == "computed"]
    gate = _pool_cell(rows, PRIMARY_QUALITY_TIER, "deviation", "within_load", "orthogonality_gate_vs_spike_count")
    raw = _pool_cell(rows, PRIMARY_QUALITY_TIER, "deviation", "within_load", "raw_vs_report_error")
    delivered = json.loads(WATTERS_DEVIATION_SOURCE_ARTIFACT.read_text())
    delivered_node = delivered["block_a"]["results"][PRIMARY_QUALITY_TIER]["pooled"]["deviation"]["within_item_count_level"]
    delivered_gate = delivered_node["orthogonality_gate_vs_spike_count"]
    delivered_raw = delivered_node["raw_vs_report_error"]
    checks = {
        "gate_r": abs((gate.get("mean_value") or float("nan")) - delivered_gate["mean_value"]) <= REPRODUCTION_TOLERANCE,
        "gate_p": abs((gate.get("two_sided_p_value") or float("nan")) - delivered_gate["two_sided_p_value"]) <= REPRODUCTION_TOLERANCE,
        "raw_r": abs((raw.get("mean_value") or float("nan")) - delivered_raw["mean_value"]) <= REPRODUCTION_TOLERANCE,
        "raw_p": abs((raw.get("two_sided_p_value") or float("nan")) - delivered_raw["two_sided_p_value"]) <= REPRODUCTION_TOLERANCE,
    }
    return {
        "status": "reproduced_exactly" if all(checks.values()) else "not_reproduced",
        "tolerance": REPRODUCTION_TOLERANCE, "checks": checks,
        "recomputed": {"gate": gate, "raw": raw},
        "delivered_source": "results/dissociation_replication_and_counting_noise.json:block_a.results."
                             f"{PRIMARY_QUALITY_TIER}.pooled.deviation.within_item_count_level",
        "delivered": {"gate": delivered_gate, "raw": delivered_raw},
    }


def watters_block_a(full_sessions: list[dict], deviation_arms: list[dict]) -> dict:
    session_records = [_session_block_a(f["session"], f["arrays"], is_binary_outcome=False) for f in full_sessions]

    arm_by_session = {r["session"]: r["arm"] for r in deviation_arms if r["status"] == "computed"}
    within_values = [arm_by_session[s].get("within_load_trial_count_weighted", {}).get("raw_vs_report_error")
                      for s in arm_by_session if arm_by_session[s].get("within_load_trial_count_weighted", {}).get(
                          "raw_vs_report_error") is not None]
    within_session = _pool_values(within_values)

    levels = sorted({int(v) for f in full_sessions for v in f["arrays"]["item_count"].tolist()})
    between_session_rows = []
    for level in levels:
        dev_means, worse_means = [], []
        for f in full_sessions:
            mask = f["arrays"]["item_count"] == float(level)
            n = int(mask.sum())
            if n < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
                continue
            dev_means.append(float(np.mean(f["arrays"]["deviation"][mask])))
            worse_means.append(float(np.mean(f["arrays"]["worse_behaviour"][mask])))
        if len(dev_means) >= 4:
            rng = np.random.default_rng(stable_seed(f"component_effect_size_and_anatomy|watters|between_session|{level}"))
            test = pearson_permutation_test(np.array(dev_means), np.array(worse_means), n_perm=10000, rng=rng)
            test["status"] = "computed"
        else:
            test = {"status": "underpowered_by_construction", "n_sessions": len(dev_means)}
        between_session_rows.append({"item_count": level, "n_sessions": len(dev_means), "between_session": test})

    reference = {
        "raw_association_r_units": None,
        "note": "The multi-object corpus's own already-delivered raw within-item-count-level association "
                "(results/dissociation_replication_and_counting_noise.json) is reproduced above as part of "
                "the reproduction gate rather than re-quoted here; see the top-level reproduction block for "
                "its exact value.",
    }
    return run_block_a("macaque_multi_object", session_records, within_session, between_session_rows, reference)


# =======================================================================================================
# Block B -- localisation by area (single-item corpus only) and by animal (both corpora)
# =======================================================================================================

def _macaque_group_cell(group_rows: list[dict]) -> dict:
    computed_rows = [r for r in group_rows if r["status"] == "computed"]
    gate = _macaque_pool(computed_rows, "orthogonality_gate")
    reachable_rows = [r for r in computed_rows if r["n_error"] >= MIN_ERROR_TRIALS_FOR_REACHABILITY]
    flipped_r = [-r["analysis"]["raw_outcome_vs_deviation"]["r"] for r in reachable_rows
                 if r["analysis"]["raw_outcome_vs_deviation"].get("status") == "computed"]
    flipped_rows = [{"status": "computed", "analysis": {"raw_outcome_vs_deviation": {"status": "computed", "r": v}}}
                     for v in flipped_r]
    behaviour = _macaque_pool(flipped_rows, "raw_outcome_vs_deviation")
    gate_void = bool(gate.get("significant") is True)
    if gate_void:
        status = "void_due_to_gate_failure"
    elif len(flipped_r) == 0:
        status = "void_no_reachable_sessions"
    elif behaviour.get("status") == "underpowered_by_construction":
        status = "underpowered_by_construction"
    elif behaviour.get("status") == "tested":
        status = "computed"
    else:
        status = "not_computable"
    spike_medians = [float(np.median(r["arrays"]["spike_count"])) for r in computed_rows]
    return {
        "n_sessions_in_group": len(group_rows), "n_sessions_gate_computed": len(computed_rows),
        "n_sessions_reachable_for_behaviour": len(flipped_r),
        "median_total_spike_count_per_trial": float(np.median(spike_medians)) if spike_medians else None,
        "gate": gate, "behaviour_association": behaviour, "status": status,
        "per_session_worse_behaviour_r": flipped_r,
    }


def _two_sample_minimum_detectable_difference(xa: np.ndarray, xb: np.ndarray) -> dict:
    """Smallest true between-group mean difference a two-sample two-sided test could have detected at 80%
    power, from the two groups' own observed spread and sizes -- the two-sample analogue of
    minimum_detectable_paired_difference (src/statistics.py), reusing its same alpha=0.05/power=0.80
    z-factor rather than a new one, applied to the standard two-sample standard error in place of the
    one-sample sqrt(n) it uses for a paired design."""
    if len(xa) < 2 or len(xb) < 2:
        return {"status": "not_computable", "n_a": len(xa), "n_b": len(xb),
                "reason": "fewer than 2 sessions in one of the two groups -- no spread to estimate"}
    one_sample_reference = minimum_detectable_paired_difference(np.concatenate([xa, xb]))
    z = one_sample_reference["z_factor"]
    se = float(np.sqrt(np.var(xa, ddof=1) / len(xa) + np.var(xb, ddof=1) / len(xb)))
    return {"status": "computed", "n_a": len(xa), "n_b": len(xb), "alpha": 0.05, "power": 0.80,
            "z_factor": z, "mdd": float(z * se)}


def _localisation_branch(cells: dict[str, dict], label: str) -> dict:
    computed = {k: v for k, v in cells.items() if v["status"] == "computed"}
    void = {k: v for k, v in cells.items()
            if v["status"] in ("void_due_to_gate_failure", "void_no_reachable_sessions")}
    underpowered = {k: v for k, v in cells.items() if v["status"] == "underpowered_by_construction"}

    if not computed and len(void) == len(cells):
        return {"branch": "anatomical_localisation_unreachable_at_this_unit_count_per_area",
                "surviving_groups": [], "void_groups": list(void), "underpowered_groups": [], "pairwise_tests": {}}
    if not computed:
        return {"branch": "no_group_reaches_a_powered_behavioural_association_some_cells_reachable_but_underpowered",
                "surviving_groups": [], "void_groups": list(void), "underpowered_groups": list(underpowered),
                "pairwise_tests": {}}
    if len(computed) == 1:
        return {"branch": "only_one_group_survives_no_cross_group_comparison_possible",
                "surviving_groups": list(computed), "void_groups": list(void),
                "underpowered_groups": list(underpowered), "pairwise_tests": {}}

    names = list(computed)
    pairwise: dict[str, dict] = {}
    any_significant = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            xa = np.array(computed[a]["per_session_worse_behaviour_r"])
            xb = np.array(computed[b]["per_session_worse_behaviour_r"])
            rng = np.random.default_rng(stable_seed(f"component_effect_size_and_anatomy|localisation|{label}|{a}|{b}"))
            diff, p = permutation_test_twosample(xa, xb, n_perm=10000, rng=rng)
            sig = bool(p <= 0.05)
            any_significant = any_significant or sig
            pairwise[f"{a}_vs_{b}"] = {
                "observed_difference": diff, "p_value": p, "significant": sig,
                "n_a": len(xa), "n_b": len(xb),
                "minimum_detectable_paired_difference_at_80pct_power": _two_sample_minimum_detectable_difference(xa, xb),
            }
    if any_significant:
        branch = f"accuracy_predicting_component_is_stronger_in_one_{label}"
    else:
        branch = f"accuracy_predicting_component_is_not_localised_to_one_recorded_{label}"
    return {"branch": branch, "surviving_groups": names, "void_groups": list(void),
            "underpowered_groups": list(underpowered), "pairwise_tests": pairwise}


def macaque_block_b(rows: list[dict]) -> dict:
    by_animal = {"monkey_A": [], "monkey_H": [], "monkey_J": []}
    for r in rows:
        if r.get("animal") in by_animal:
            by_animal[r["animal"]].append(r)

    animal_cells = {a: _macaque_group_cell(group) for a, group in by_animal.items()}
    animal_result = _localisation_branch(animal_cells, "animal")

    area_cells = {"area_8": _macaque_group_cell(by_animal["monkey_A"]),
                  "area_9_46": _macaque_group_cell(by_animal["monkey_J"])}
    area_result = _localisation_branch(area_cells, "area")

    return {
        "area_split": {
            "cells": area_cells, "result": area_result,
            "excluded_from_area_split": {
                "monkey_H": "recorded area is undetermined between area 8 and area 9/46 for this animal "
                            "(ANIMAL_ASSIGNMENT_SOURCE); excluded from the area split by name, not voided "
                            "by a gate or by reachability, and remains in the animal split below.",
            },
            "assignment_source": ANIMAL_ASSIGNMENT_SOURCE,
        },
        "animal_split": {
            "cells": animal_cells, "result": animal_result,
            "session_counts_by_animal": {a: len(group) for a, group in by_animal.items()},
        },
    }


def _watters_group_cell(subset_rows: list[dict], full_by_session: dict[str, dict]) -> dict:
    computed_rows = [r for r in subset_rows if r["status"] == "computed"]
    pool_rows = [{"by_tier": {PRIMARY_QUALITY_TIER: {"status": "computed", "deviation": r["arm"]}}}
                 for r in computed_rows]
    gate = _pool_cell(pool_rows, PRIMARY_QUALITY_TIER, "deviation", "within_load", "orthogonality_gate_vs_spike_count")
    behaviour = _pool_cell(pool_rows, PRIMARY_QUALITY_TIER, "deviation", "within_load", "raw_vs_report_error")
    gate_void = bool(gate.get("significant") is True)
    per_session_r = [r["arm"]["within_load_trial_count_weighted"]["raw_vs_report_error"] for r in computed_rows
                      if r["arm"]["within_load_trial_count_weighted"].get("raw_vs_report_error") is not None]
    if gate_void:
        status = "void_due_to_gate_failure"
    elif len(per_session_r) == 0:
        status = "void_no_reachable_sessions"
    elif behaviour.get("status") == "underpowered_by_construction":
        status = "underpowered_by_construction"
    elif behaviour.get("status") == "tested":
        status = "computed"
    else:
        status = "not_computable"
    spike_medians = [float(np.median(full_by_session[r["session"]]["arrays"]["spike_count"]))
                      for r in computed_rows if r["session"] in full_by_session]
    return {
        "n_sessions_in_group": len(subset_rows), "n_sessions_gate_computed": len(computed_rows),
        "n_sessions_with_a_within_load_association": len(per_session_r),
        "median_total_spike_count_per_trial": float(np.median(spike_medians)) if spike_medians else None,
        "gate": gate, "behaviour_association": behaviour, "status": status,
        "per_session_worse_behaviour_r": per_session_r,
    }


def watters_block_b(deviation_arms: list[dict], full_sessions: list[dict]) -> dict:
    full_by_session = {f["session"]: f for f in full_sessions}
    subsets = _subsets(deviation_arms)
    animal_groups = {k[len("animal_"):]: v for k, v in subsets.items() if k.startswith("animal_")}
    animal_cells = {a: _watters_group_cell(group, full_by_session) for a, group in animal_groups.items()}
    animal_result = _localisation_branch(animal_cells, "animal")
    return {
        "area_split": {
            "status": "refused_by_corpus_property",
            "reason": "Every recorded channel's NWB location field is the literal string 'unknown' in every "
                      "staged spike-sorting file for this corpus, so no area-resolved split is possible -- "
                      "verified directly against two independent statements already on disk (src/corpus_"
                      "sessions.py's own _watters_unit_index docstring, line 379, and scripts/build_"
                      "anatomical_census.py's per-corpus electrode census note). Refused by name, not "
                      "attempted and not reported as a null result.",
        },
        "animal_split": {
            "cells": animal_cells, "result": animal_result,
            "session_counts_by_animal": {a: len(group) for a, group in animal_groups.items()},
        },
    }


# =======================================================================================================
# Zero-drop accounting
# =======================================================================================================

def macaque_zero_drop(rows: list[dict]) -> dict:
    n_seen = len(rows)
    n_computed = sum(1 for r in rows if r["status"] == "computed")
    n_not_computable = n_seen - n_computed
    n_reachable = sum(1 for r in rows if r["status"] == "computed"
                       and r["n_error"] >= MIN_ERROR_TRIALS_FOR_REACHABILITY)
    return {
        "n_sessions_on_disk": n_seen, "n_sessions_with_a_defined_direction": n_computed,
        "n_sessions_not_computable_too_few_trials": n_not_computable,
        "n_sessions_reaching_the_reachability_floor": n_reachable,
        "n_sessions_below_the_reachability_floor": n_computed - n_reachable,
        "reconciles": bool(n_seen == n_computed + n_not_computable),
    }


def watters_zero_drop(seen: int, loaded: list[dict], refused: list[dict], full_sessions: list[dict]) -> dict:
    n_usable = len(full_sessions)
    return {
        "n_behavioural_session_dates_seen": seen, "n_sessions_loaded": len(loaded),
        "n_sessions_refused_by_the_shared_loader": len(refused),
        "refusals_by_reason": {reason: sum(1 for r in refused if r["status"] == reason)
                                for reason in sorted({r["status"] for r in refused})},
        "n_sessions_reaching_a_usable_behavioural_arm": n_usable,
        "n_loaded_sessions_without_a_usable_behavioural_arm": len(loaded) - n_usable,
        "reconciles": bool(seen == len(loaded) + len(refused)),
    }


# =======================================================================================================
# Driver
# =======================================================================================================

def main() -> None:
    t0 = time.time()
    _COMPLETED_FITS.update(_load_completed_fits())
    _log(f"model fits already recorded as complete: {len(_COMPLETED_FITS)}")
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "scope": (
            "Two macaque corpora whose rate-free direction-deviation observable passes its own "
            "orthogonality gate against total spike count: the single-item macaque lateral prefrontal "
            "cortex corpus and the multi-object macaque corpus (DANDI 000620). The "
            "gate values that set this scope, read from results/dissociation_replication_and_counting_noise."
            "json's five-corpus census: macaque lPFC 0.0377 (p=0.589, passes), multi-object macaque -0.0016 "
            "(p=0.955, passes), mouse ALM -0.2057 (p=0.0005, fails), human dandi_000469 -0.1704 (p=0.0285, "
            "fails), human dandi_001187 -0.1912 (p=0.0015, fails). The mouse and both human corpora are "
            "excluded here on that already-measured precondition and are not treated as pending."
        ),
        "sign_convention": WORSE_BEHAVIOUR_SIGN_MAP,
        "block_a_decision_rule_declared_before_fitting": BLOCK_A_TRANSLATION_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "block_b_decision_rule_declared_before_fitting": BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "status": "running",
    }
    _flush(output)

    _log("reproduction gate: single-item macaque corpus")
    macaque_repro = macaque_reproduction(root)
    output["reproduction"] = {"macaque_lPFC_single_item": {"status": macaque_repro["status"],
                                                             "checks": macaque_repro["checks"]}}
    _flush(output)
    _log(f"  -> {macaque_repro['status']}")

    _log("loading multi-object macaque corpus (one pass)")
    watters_seen, watters_loaded, watters_refused = load_watters_corpus(root)
    _log(f"  -> seen={watters_seen} loaded={len(watters_loaded)} refused={len(watters_refused)} "
         f"elapsed={time.time() - t0:.0f}s")

    _log("computing the multi-object corpus's own deviation arm per session (reused estimator)")
    watters_deviation_arms = load_watters_deviation_arms(watters_loaded)
    output["_progress"] = {"watters_deviation_arms_done": len(watters_deviation_arms)}
    _flush(output)
    _log(f"  -> {len(watters_deviation_arms)} sessions, elapsed={time.time() - t0:.0f}s")

    _log("reproduction gate: multi-object macaque corpus")
    watters_repro = watters_reproduction(watters_deviation_arms)
    output["reproduction"]["macaque_multi_object"] = {"status": watters_repro["status"],
                                                        "checks": watters_repro["checks"],
                                                        "recomputed": watters_repro["recomputed"],
                                                        "delivered": watters_repro["delivered"]}
    output.pop("_progress", None)
    _flush(output)
    _log(f"  -> {watters_repro['status']}")

    both_reproduced = (macaque_repro["status"] == "reproduced_exactly"
                        and watters_repro["status"] == "reproduced_exactly")
    output["reproduction"]["both_corpora_reproduced_exactly"] = both_reproduced
    if not both_reproduced:
        output["status"] = "stopped_reproduction_gate_failed"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate did not reproduce exactly on at least one corpus; no new number computed")
        print(json.dumps({"status": output["status"]}, indent=2))
        return

    _log("loading single-item macaque corpus (all 25 sessions, gate not restricted to reachable sessions)")
    macaque_rows = load_macaque_corpus(root)
    output["_progress"] = {"macaque_sessions_done": len(macaque_rows)}
    _flush(output)
    _log(f"  -> {len(macaque_rows)} sessions, elapsed={time.time() - t0:.0f}s")

    _log("loading multi-object corpus per-trial arrays for Block A")
    watters_full_sessions = []
    for session in watters_loaded:
        full = _fit(f"watters_full_session|{session['session']}", lambda s=session: _watters_full_session(s))
        if full is not None:
            # A fit served from the on-disk checkpoint has round-tripped through JSON, which turns every
            # array field into a plain list; the array masking and .tolist() calls below require real
            # ndarrays regardless of whether this session's fit was just computed or was restored.
            full = {**full, "arrays": {k: np.asarray(v) for k, v in full["arrays"].items()}}
            watters_full_sessions.append(full)
    output.pop("_progress", None)
    _flush(output)
    _log(f"  -> {len(watters_full_sessions)} sessions with a usable behavioural arm, elapsed={time.time() - t0:.0f}s")

    _log("Block A: macaque translation")
    output["block_a"] = {"macaque_lPFC_single_item": macaque_block_a(macaque_rows)}
    _flush(output)
    _log(f"  -> branch={output['block_a']['macaque_lPFC_single_item']['branch']}")

    _log("Block A: multi-object macaque translation")
    output["block_a"]["macaque_multi_object"] = watters_block_a(watters_full_sessions, watters_deviation_arms)
    _flush(output)
    _log(f"  -> branch={output['block_a']['macaque_multi_object']['branch']}")

    _log("Block B: single-item macaque localisation")
    output["block_b"] = {"macaque_lPFC_single_item": macaque_block_b(macaque_rows)}
    _flush(output)
    _log(f"  -> area={output['block_b']['macaque_lPFC_single_item']['area_split']['result']['branch']} "
         f"animal={output['block_b']['macaque_lPFC_single_item']['animal_split']['result']['branch']}")

    _log("Block B: multi-object macaque localisation")
    output["block_b"]["macaque_multi_object"] = watters_block_b(watters_deviation_arms, watters_full_sessions)
    _flush(output)
    _log(f"  -> animal={output['block_b']['macaque_multi_object']['animal_split']['result']['branch']}")

    output["zero_drop_accounting"] = {
        "macaque_lPFC_single_item": macaque_zero_drop(macaque_rows),
        "macaque_multi_object": watters_zero_drop(watters_seen, watters_loaded, watters_refused, watters_full_sessions),
    }

    output["how_this_artifact_was_assembled"] = {
        "n_model_fits_served_from_an_earlier_invocation": _FITS_SERVED_FROM_CHECKPOINT,
        "n_model_fits_computed_in_this_invocation": _FITS_COMPUTED_HERE,
        "completed_fit_record": "results/.checkpoints/component_effect_size_and_anatomy_checkpoint.json",
    }
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({
        "reproduction": {k: v["status"] for k, v in output["reproduction"].items() if isinstance(v, dict)},
        "block_a_branches": {k: v["branch"] for k, v in output["block_a"].items()},
        "block_b_area_branch_macaque": output["block_b"]["macaque_lPFC_single_item"]["area_split"]["result"]["branch"],
        "block_b_animal_branch_macaque": output["block_b"]["macaque_lPFC_single_item"]["animal_split"]["result"]["branch"],
        "block_b_animal_branch_watters": output["block_b"]["macaque_multi_object"]["animal_split"]["result"]["branch"],
        "wall_clock_s": output["wall_clock_s"],
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
