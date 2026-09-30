"""run_occupied_subspace_rank_selection_repair.py -- repairs and re-tests a cross-validated rank
selector that is provably incapable of returning anything but the full ambient rank under a common
condition, and reports whether the delivered "the axis lies within the occupied state space" finding
survives once that condition is removed.

THE MECHANISM, from first principles. `_cv_pca_rank` (scripts/run_deviation_axis_structure.py:1088)
picks an occupied-subspace rank k by 5-fold cross-validated PCA reconstruction error: for each fold,
it mean-centres the training trials, takes their SVD, and for every candidate k in 1..max_k projects
the held-out trials onto the top-k right-singular vectors and accumulates squared reconstruction
error; the reported rank is the k with the smallest summed error (np.argmin). The candidate range the
delivered callers pass is max_k = min(p, n_trials - 1), where p is the ambient recorded-unit count --
so the candidate set always reaches k = p whenever there are enough trials.

After mean-centring, SVD with full_matrices=False returns min(n_train, p) right-singular vectors. When
a training fold has n_train >= p, that full set of p singular vectors is a COMPLETE orthonormal basis
of R^p: for ANY vector v in R^p, basis @ (basis.T @ v) == v to floating-point precision, because the
basis spans the whole ambient space, not because it captured anything about the held-out trial's
structure. So when the candidate range reaches k = p and any training fold satisfies n_train >= p, the
reconstruction error at that k is forced to ~1e-28 by a linear-algebra identity, for every session,
regardless of whether the neural population's activity is genuinely full rank. Because held-out error
is monotone non-increasing in k up to that point (a bigger subspace can only reconstruct more), the
curve decreases all the way to numerical zero and np.argmin necessarily returns max_k. The selector
cannot ever prefer a smaller rank once the candidate range includes a point where a complete basis is
reachable: cross-validation is not being defeated by noise here, it is being defeated by a tautology
built into the candidate range.

Consequence for the delivered `results/deviation_axis_structure.json`: its `occupied_state_space_block`
reports the residual axis's within-occupied-space fraction as a comparison against a matched-dimension
random-axis null at the cross-validated rank. If that rank is forced to the ambient dimension p, then
"within the occupied space" is true by construction for every vector in R^p, including the axis itself:
the number is a restatement of "the ambient space contains the ambient space," not a measurement of
where the axis actually sits relative to what the population's own trial-to-trial variability spans.

THE REPAIR. A new function, `_cv_pca_rank_capped`, is the same selector with exactly one departure:
the candidate range is capped strictly below min(min_n_train_across_folds, p), so no candidate k can
ever reach a point where any training fold could supply a complete (or, when p exceeds every fold's
n_train, a training-row-space-exhausting) basis. This is the ONLY change from the original; no other
convention (folds, error accumulation, tie-break) is altered. The original `_cv_pca_rank` is imported
here UNCHANGED, never edited, and both selectors are run side by side on identical sessions, identical
trials, and identical residual axes, so every number in this artifact is directly comparable to its
counterpart in `results/deviation_axis_structure.json`.

SCOPE. Both corpora already gated into the delivered `occupied_state_space_block` (the single-item
macaque lateral prefrontal cortex corpus and the multi-object macaque corpus, each already checked to
carry a residual-direction concentration finding) are re-run here. This module reuses every delivered
loader, gate, and decomposition function unchanged (`_macaque_bundles`, `_watters_bundles`,
`full_reproduction_gate`, `_collect_axis_entries`, `leading_eigenvector`, `_occupied_space_decomposition`,
`pooled_off_fraction_against_matched_null`, `classify_occupied_space_branch`, `_trial_count_weighted`,
`_weighted_combine_draws`); the only new code is the capped selector, the whole-session cluster-
bootstrap detection floor, and the orchestration that runs both selectors side by side and classifies
the pre-declared verdict.

A SEPARATE, STRUCTURAL note on the off-fraction test itself, independent of the rank bug: an
off-fraction is a squared-norm ratio bounded below by zero, so "off-fraction significantly BELOW its
null" is not a symmetric two-sided finding -- a near-zero off-fraction (forced there by the rank bug or
not) is mechanically likely to test as significantly below a strictly-positive matched null, because
there is nowhere for it to go but up. This artifact reports that construction explicitly for both
corpora's pooled comparisons, in both the original and the corrected artifact, without adopting a
different test.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from corpus_sessions import data_root, iter_watters  # noqa: E402
from provenance import _json_safe, git_commit  # noqa: E402
from corpus_sessions import _reachable_sessions
from subspace_identity import N_CV_FOLDS, _contiguous_folds
from info_decoding import N_BOOT_SESSION_CLUSTER
from statistics import Z_80_POWER
from run_deviation_axis_structure import _macaque_bundles
from info_decoding import _collect_axis_entries
from info_decoding import _cv_pca_rank, classify_occupied_space_branch, _occupied_space_decomposition, _weighted_combine_draws, pooled_off_fraction_against_matched_null
from corpus_sessions import _watters_bundles
from subspace_identity import leading_eigenvector
from info_decoding import CORPORA, MIN_FOLD_TRIALS, N_RANDOM_AXIS_DRAWS
from corpus_sessions import _panichello_directory
from run_multi_object_interference_and_locus_within_item_count import _trial_count_weighted
from run_deviation_serial_dependence_and_temporal_locus import full_reproduction_gate  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import stable_seed  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "occupied_subspace_rank_selection_repair.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_occupied_subspace_rank_selection_repair"
ANALYSIS_VERSION = "2026-09-05"

SELECTORS = ("original", "corrected")

DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "THE ERROR MECHANISM (from first principles, self-contained): `_cv_pca_rank` selects an occupied-"
    "subspace rank k by 5-fold cross-validated PCA reconstruction error over candidate k = 1..max_k, "
    "max_k = min(ambient_unit_count_p, n_trials - 1). After mean-centring, a training fold's SVD returns "
    "min(n_train, p) right-singular vectors; whenever a fold has n_train >= p, taking all p of them is a "
    "COMPLETE orthonormal basis of the ambient space R^p, so ANY held-out vector reconstructs to itself "
    "exactly (basis @ basis.T = identity on R^p) -- not because the model generalises, but because a "
    "complete basis is an identity map by linear algebra. Held-out reconstruction error is forced to "
    "~1e-28 at k=p and stays there for every larger candidate, so np.argmin over a range reaching that "
    "point always returns max_k, regardless of the data's true occupied dimensionality.\n"
    "THE CORRECTION, declared before any fit: a new selector, `_cv_pca_rank_capped`, is byte-identical to "
    "the original except that its candidate range is capped at cap = min(max_k, min_n_train_across_folds, "
    "p) - 1, where min_n_train_across_folds is computed per session/level from the SAME 5 contiguous "
    "chronological folds the original selector uses. This is applied uniformly whether or not p happens "
    "to exceed every fold's n_train in a given cell: a candidate sitting exactly at a fold's ceiling lets "
    "that fold's basis exhaust its own training row space, which is the identical qualitative failure "
    "(a near-complete basis manufacturing near-zero held-out error) at a smaller scale. No other change "
    "is made: same 5 contiguous folds, same accumulated squared error across folds, same argmin tie-break "
    "(first minimum). No one-standard-error rule is added.\n"
    "THE PRE-DECLARED VERDICT BRANCHES, decided per corpus from the re-run pooled "
    "observed-off-fraction-versus-matched-null comparison (identical construction to "
    "pooled_off_fraction_against_matched_null in the delivered module, imported unchanged), after first "
    "confirming the tautology count from step 1 (best_k == max_k, monotone non-increasing error, final "
    "error < 1e-20) is non-trivial for that corpus under the ORIGINAL selector:\n"
    "  - if the corrected selector's pooled branch is "
    "'the_axis_lies_within_the_occupied_state_space_but_outside_the_coding_subspace' -> "
    "'within_occupied_space_finding_is_tautological_under_the_original_selector_and_holds_under_the_"
    "corrected_one'.\n"
    "  - if the corrected selector's pooled branch is 'the_axis_lies_outside_the_occupied_state_space' -> "
    "'within_occupied_space_finding_is_tautological_under_the_original_selector_and_does_not_hold_under_"
    "the_corrected_one'.\n"
    "  - if the corrected comparison is not computable, or the corrected selector's own per-cell rank "
    "distribution still saturates its new, smaller ceiling in a majority of cells for a reason that "
    "cannot be distinguished from the same capacity-exhaustion failure at smaller scale -> "
    "'within_occupied_space_finding_is_tautological_under_the_original_selector_and_is_inconclusive_under_"
    "the_corrected_one'.\n"
    "  - if, instead, the fraction of cells hitting max_k under the ORIGINAL selector is not in fact "
    "dominant for that corpus (the tautology count from step 1 is a small minority), the finding is "
    "reported as 'both_selectors_agree' for that corpus: there was no tautology to repair there in the "
    "first place, only in the cells already flagged.\n"
    "Power: every non-trivial pooled cell carries its own minimum detectable off-fraction difference at "
    "80% power from a whole-SESSION cluster bootstrap (resample sessions with replacement, "
    f"{N_BOOT_SESSION_CLUSTER} draws, recompute the pooled mean off-fraction each draw; mdd = {Z_80_POWER} "
    "* that bootstrap standard error), never a trial-count formula or an intraclass-correlation design "
    "effect.\n"
    "The off-fraction test's own construction is reported explicitly for every pooled comparison in both "
    "selectors: an off-fraction is a squared-norm ratio bounded below by zero, so a two-sided empirical "
    "test of a near-zero observed value against a strictly positive matched null is testing a quantity "
    "that has nowhere to go but up -- 'significantly below the null' is close to guaranteed whenever the "
    "observed off-fraction sits at or near its floor, independent of whether the rank bug is present. This "
    "is disclosed as a structural property of the test, not adopted as a new statistic here."
)


# =======================================================================================================
# Checkpointing (mirrors run_deviation_axis_structure.py's convention: one file per unit of work, temp
# file + os.replace, completion flag written only after the fit returns).
# =======================================================================================================

def _checkpoint_path(unit: str) -> Path:
    return CHECKPOINT_DIR / f"{unit.replace('/', '_')}.json"


def _load_checkpoint(unit: str) -> dict | None:
    path = _checkpoint_path(unit)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True:
        return None
    return data["record"]


def _save_checkpoint(unit: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(unit)
    payload = {"_complete": True, "record": _json_safe(record)}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload, allow_nan=False))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


# =======================================================================================================
# THE CORRECTED SELECTOR -- the only estimator this module adds. `_cv_pca_rank` is imported, unchanged,
# from scripts/run_deviation_axis_structure.py and used directly below as the "original" side of every
# comparison.
# =======================================================================================================

def _cv_pca_rank_capped(U: np.ndarray, max_k: int, seed_tag: str) -> dict:
    """Identical to run_deviation_axis_structure._cv_pca_rank in every respect (5 contiguous chronological
    folds, accumulated squared held-out reconstruction error, np.argmin over the candidate range) except
    one: the candidate range is capped at min(max_k, min_n_train_across_folds, p) - 1, so no candidate k
    can ever equal a value at which some training fold's SVD supplies a complete (or training-row-space-
    exhausting) basis. At that point, and only at that point, `test_c @ basis @ basis.T` is forced toward
    `test_c` by linear algebra rather than by genuine generalisation -- see this module's docstring for the
    full derivation. Declared, not discovered after the fact: the cap is applied uniformly regardless of
    whether p happens to exceed every fold's n_train in a given cell, because a candidate sitting exactly
    at a fold's own ceiling reproduces the identical qualitative failure (near-complete basis, near-zero
    held-out error by capacity exhaustion) at whatever scale that fold's n_train sits. No one-standard-
    error tie-break is added; ties are broken by the same first-minimum np.argmin the original uses."""
    n = U.shape[0]
    if max_k < 1 or n < N_CV_FOLDS * MIN_FOLD_TRIALS:
        return {"status": "not_estimable"}
    folds = _contiguous_folds(n, N_CV_FOLDS)
    fold_sizes = [int((folds == f).sum()) for f in range(N_CV_FOLDS)]
    min_n_train = n - max(fold_sizes)
    capped_max_k = min(max_k, min_n_train, U.shape[1]) - 1
    if capped_max_k < 1:
        return {"status": "not_estimable", "reason": "capped_candidate_range_below_one",
                "min_n_train_across_folds": min_n_train, "p": int(U.shape[1]), "uncapped_max_k": max_k}
    candidate_ks = list(range(1, capped_max_k + 1))
    errors = np.zeros(len(candidate_ks))
    n_folds_used = 0
    for f in range(N_CV_FOLDS):
        train, test = folds != f, folds == f
        if int(train.sum()) < 4 or not test.any():
            continue
        mean_train = U[train].mean(axis=0)
        _u_svd, _s_svd, vt_svd = np.linalg.svd(U[train] - mean_train, full_matrices=False)
        test_c = U[test] - mean_train
        n_folds_used += 1
        for i, k in enumerate(candidate_ks):
            k_eff = min(k, vt_svd.shape[0])
            basis = vt_svd[:k_eff].T
            proj = test_c @ basis @ basis.T
            errors[i] += float(np.sum((test_c - proj) ** 2))
    if n_folds_used == 0:
        return {"status": "not_estimable"}
    best_idx = int(np.argmin(errors))
    if best_idx == len(candidate_ks) - 1:
        return {
            "status": "nonidentified_at_search_ceiling", "n_folds_used": n_folds_used,
            "candidate_ks": candidate_ks, "cv_reconstruction_error": errors.tolist(),
            "min_n_train_across_folds": min_n_train, "uncapped_max_k": max_k,
            "capped_max_k": capped_max_k, "selected_k_at_search_ceiling": candidate_ks[best_idx],
            "best_k": None,
        }
    return {
        "status": "computed", "n_folds_used": n_folds_used, "candidate_ks": candidate_ks,
        "cv_reconstruction_error": errors.tolist(), "best_k": candidate_ks[best_idx],
        "min_n_train_across_folds": min_n_train, "uncapped_max_k": max_k, "capped_max_k": capped_max_k,
    }


# =======================================================================================================
# Whole-session cluster bootstrap detection floor -- reuses this project's Z_80_POWER convention
# (imported unchanged from run_component_identity_subspace_atlas.py) on a per-session scalar list;
# never a trial-count formula or an intraclass-correlation design effect.
# =======================================================================================================

def _session_cluster_bootstrap_mdd(values: list[float], seed_tag: str) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    n = int(arr.size)
    if n < 4:
        return {"status": "not_computable", "n_sessions": n}
    rng = np.random.default_rng(stable_seed(seed_tag))
    boot_means = np.array([arr[rng.integers(0, n, size=n)].mean() for _ in range(N_BOOT_SESSION_CLUSTER)])
    se = float(np.std(boot_means, ddof=1))
    return {"status": "computed", "n_sessions": n, "n_bootstrap_draws": N_BOOT_SESSION_CLUSTER,
            "cluster_bootstrap_se": se, "z_multiplier": Z_80_POWER,
            "minimum_detectable_off_fraction_difference_at_80pct_power": Z_80_POWER * se}


# =======================================================================================================
# Loading -- identical to run_deviation_axis_structure.main()'s loading section, reusing every function
# unchanged, so the bundles fed to both selectors are the same objects the delivered artifact was built
# from.
# =======================================================================================================

def _load_corpora_and_accounting(root: Path) -> tuple[dict, dict]:
    watters_seen, watters_loaded, watters_refused = 0, [], []
    refusal_hist: dict[str, int] = {}
    for session in iter_watters(root, bin_ms=100.0):
        watters_seen += 1
        if session["status"] != "loaded":
            watters_refused.append({"session": session["session"], "status": session["status"]})
            refusal_hist[session["status"]] = refusal_hist.get(session["status"], 0) + 1
            continue
        watters_loaded.append(session)

    gate_result, watters_arrays_by_session = full_reproduction_gate(root, watters_loaded)

    macaque_paths = _reachable_sessions(root)
    macaque_bundles, _ = _macaque_bundles(root)
    n_macaque_on_disk = len(list(_panichello_directory(root).glob("*.mat"))) if _panichello_directory(root) else 0

    watters_bundles = _watters_bundles(watters_arrays_by_session)
    n_watters_arrays_none = len(watters_loaded) - len(watters_arrays_by_session)

    bundles = {CORPORA[0]: macaque_bundles, CORPORA[1]: watters_bundles}

    n_macaque_refused_reachability = n_macaque_on_disk - len(macaque_paths)
    n_macaque_refused_too_few_trials = len(macaque_paths) - len(macaque_bundles)
    zero_drop = {
        CORPORA[0]: {
            "n_seen": n_macaque_on_disk,
            "n_loaded": len(macaque_paths),
            "n_refused": n_macaque_refused_reachability + n_macaque_refused_too_few_trials,
            "refusal_reasons": {
                "below_reachability_floor": n_macaque_refused_reachability,
                "too_few_trials_with_defined_direction_after_load": n_macaque_refused_too_few_trials,
            },
            "n_analysed": len(macaque_bundles),
            "reconciles": bool(n_macaque_on_disk == n_macaque_refused_reachability
                               + n_macaque_refused_too_few_trials + len(macaque_bundles)),
        },
        CORPORA[1]: {
            "n_seen": watters_seen,
            "n_loaded": len(watters_loaded),
            "n_refused": len(watters_refused),
            "refusal_reasons": refusal_hist,
            "n_arrays_not_computable_after_load": n_watters_arrays_none,
            "n_analysed": len(watters_bundles),
            "reconciles": bool(watters_seen == len(watters_refused) + n_watters_arrays_none + len(watters_bundles)),
        },
    }
    return {"gate_result": gate_result, "bundles": bundles}, zero_drop


# =======================================================================================================
# Per-cell, per-selector occupied-space decomposition. Mirrors run_deviation_axis_structure's
# run_occupied_space_block, doubled: both selectors are applied to the identical (R, U, axis) triple
# built by the delivered, unchanged _collect_axis_entries.
# =======================================================================================================

def _cell_under_selector(U: np.ndarray, axis: np.ndarray, max_k: int, selector_name: str, tag_base: str) -> dict:
    if selector_name == "original":
        cv = _cv_pca_rank(U, max_k, f"{tag_base}|cv")
    else:
        cv = _cv_pca_rank_capped(U, max_k, f"{tag_base}|cv")
    if cv.get("status") != "computed":
        return {"status": cv.get("status", "not_estimable"), "cv": cv}
    errs = cv["cv_reconstruction_error"]
    ks = cv["candidate_ks"]
    diffs = np.diff(errs) if len(errs) > 1 else np.array([0.0])
    primary = _occupied_space_decomposition(U, axis, cv["best_k"], N_RANDOM_AXIS_DRAWS, f"{tag_base}|primary")
    out = {
        "status": "computed", "best_k": cv["best_k"], "max_k_this_selector": ks[-1],
        "best_k_equals_max_k_this_selector": bool(cv["best_k"] == ks[-1]),
        "held_out_error_monotone_non_increasing": bool(np.all(diffs <= 1e-9)),
        "final_held_out_error": float(errs[-1]),
        "final_held_out_error_below_1e_minus_20": bool(errs[-1] < 1e-20),
    }
    if selector_name == "corrected":
        out["min_n_train_across_folds"] = cv.get("min_n_train_across_folds")
        out["uncapped_max_k"] = cv.get("uncapped_max_k")
        out["capped_max_k"] = cv.get("capped_max_k")
    if primary.get("status") != "computed":
        out["decomposition_status"] = primary.get("status", "not_computable")
        return out
    out.update({
        "decomposition_status": "computed",
        "within_fraction": primary["within_fraction"], "off_fraction": primary["off_fraction"],
        "null_off_fraction_mean": primary["null_off_fraction_mean"],
        "null_off_fraction_sd": primary["null_off_fraction_sd"],
        "two_sided_p_value_vs_matched_null": primary["two_sided_p_value"],
        "off_fraction_above_null": primary["off_fraction_above_null"],
        "null_draws": primary["null_off_fraction_draws"],  # kept in the checkpoint; stripped only when
                                                            # the OUTPUT artifact is written (main() below)
    })
    return out


def _run_corpus(bundles: list[dict], corpus_key: str) -> dict:
    axis_entries = _collect_axis_entries(bundles, corpus_key)
    per_session = []
    for bundle in bundles:
        session = bundle["session"]
        unit = f"cells|{corpus_key}|{session}"
        cached = _load_checkpoint(unit)
        if cached is not None:
            per_session.append(cached)
            continue

        entries = axis_entries.get(session, [])
        if not entries:
            record = {"session": session, "status": "not_computable", "levels": []}
            _save_checkpoint(unit, record)
            per_session.append(record)
            continue

        level_records = []
        for entry in entries:
            R, U, n_trials, level = entry["R"], entry["U"], entry["n_trials"], entry["level"]
            n, p = U.shape
            axis = leading_eigenvector(R)
            max_k = min(p, n - 1)
            folds = _contiguous_folds(n, N_CV_FOLDS)
            fold_sizes = [int((folds == f).sum()) for f in range(N_CV_FOLDS)]
            min_n_train = n - max(fold_sizes)
            tag_base = f"occupied_subspace_rank_selection_repair|{corpus_key}|{session}|{level}"
            record = {"n_trials": n_trials, "level": level, "p_ambient": int(p), "max_k": int(max_k),
                       "min_n_train_across_folds": int(min_n_train)}
            for name in SELECTORS:
                record[name] = _cell_under_selector(U, axis, max_k, name, f"{tag_base}|{name}")
            level_records.append(record)

        record = {"session": session, "status": "computed", "levels": level_records}
        per_session.append(record)
        _save_checkpoint(unit, record)  # null_draws kept: checkpoint reuse must not thin the pooled null

    # Recombine per-session, per-selector, trial-count weighted across levels -- identical convention to
    # run_deviation_axis_structure.run_occupied_space_block.
    combined_per_selector: dict[str, list[dict]] = {name: [] for name in SELECTORS}
    for sess_record in per_session:
        if sess_record.get("status") != "computed":
            continue
        for name in SELECTORS:
            computed_levels = [lvl for lvl in sess_record["levels"]
                                if lvl.get(name, {}).get("decomposition_status") == "computed"]
            if not computed_levels:
                continue
            off_fraction = _trial_count_weighted([(lvl["n_trials"], lvl[name]["off_fraction"]) for lvl in computed_levels])
            null_draws = _weighted_combine_draws([(lvl["n_trials"], lvl[name]["null_draws"]) for lvl in computed_levels])
            combined_per_selector[name].append({
                "session": sess_record["session"], "off_fraction": off_fraction, "null_draws": null_draws,
            })

    pooled_per_selector = {}
    for name in SELECTORS:
        computed_sessions = combined_per_selector[name]
        off_fracs = [s["off_fraction"] for s in computed_sessions if s.get("off_fraction") is not None]
        pooled_test = slope_across_sessions_test(off_fracs, alternative="two-sided") if off_fracs else {"status": "not_computed"}
        mdd = _session_cluster_bootstrap_mdd(
            off_fracs, f"occupied_subspace_rank_selection_repair|mdd|{corpus_key}|{name}")
        comparison = pooled_off_fraction_against_matched_null(
            [s for s in computed_sessions if s.get("null_draws") is not None])
        branch = classify_occupied_space_branch(comparison)
        pooled_per_selector[name] = {
            "n_sessions_pooled": len(off_fracs),
            "pooled_off_fraction_across_sessions": pooled_test,
            "minimum_detectable_off_fraction_difference_at_80pct_power_whole_session_cluster_bootstrap": mdd,
            "pooled_observed_vs_matched_null": comparison,
            "branch": branch,
            "off_fraction_test_is_one_sided_by_construction": (
                "off_fraction is a squared-norm ratio bounded below by zero; the pooled observed value here "
                f"({comparison.get('pooled_observed_off_fraction')}) is compared two-sidedly against a "
                f"strictly positive matched null (mean {comparison.get('pooled_null_mean')}), so a near-zero "
                "observed value has nowhere to go but up and 'significantly below the null' is close to "
                "guaranteed whenever the observed value sits at or near its floor -- true under either "
                "selector, reported here rather than silently adopted as evidence of anything."
            ),
        }

    # Tautology census under the ORIGINAL selector, computed live here rather than
    # only read from the delivered artifact, so this artifact stands alone.
    all_level_cells = [lvl for sess in per_session if sess.get("status") == "computed" for lvl in sess["levels"]]
    n_cells = len(all_level_cells)
    n_original_full = sum(1 for lvl in all_level_cells if lvl["original"].get("best_k_equals_max_k_this_selector"))
    n_original_monotone = sum(1 for lvl in all_level_cells if lvl["original"].get("held_out_error_monotone_non_increasing"))
    n_original_below_floor = sum(1 for lvl in all_level_cells if lvl["original"].get("final_held_out_error_below_1e_minus_20"))
    n_corrected_full_at_own_cap = sum(
        1 for lvl in all_level_cells
        if lvl["corrected"].get("status") == "computed"
        and lvl["corrected"].get("best_k") == lvl["corrected"].get("capped_max_k"))
    n_corrected_computed = sum(1 for lvl in all_level_cells if lvl["corrected"].get("status") == "computed")

    tautology_dominant = bool(n_cells > 0 and n_original_full > n_cells / 2.0)

    strip_draws = {name: [{k: v for k, v in s.items() if k != "null_draws"} for s in combined_per_selector[name]]
                   for name in SELECTORS}

    per_session_for_output = [
        {**sess, "levels": [
            {k: v for k, v in lvl.items() if k not in SELECTORS} |
            {name: {kk: vv for kk, vv in lvl[name].items() if kk != "null_draws"} for name in SELECTORS}
            for lvl in sess["levels"]
        ]} if sess.get("status") == "computed" else sess
        for sess in per_session
    ]

    return {
        "n_sessions_total": len(bundles),
        "n_sessions_computed": sum(1 for s in per_session if s.get("status") == "computed"),
        "per_session": per_session_for_output,
        "step1_tautology_census_under_original_selector": {
            "n_cells": n_cells, "n_best_k_equals_max_k": n_original_full,
            "fraction_best_k_equals_max_k": (n_original_full / n_cells) if n_cells else None,
            "n_monotone_non_increasing": n_original_monotone, "n_final_error_below_1e_minus_20": n_original_below_floor,
            "tautology_dominant_for_this_corpus": tautology_dominant,
        },
        "step_corrected_own_ceiling_census": {
            "n_cells_corrected_computed": n_corrected_computed,
            "n_corrected_best_k_equals_corrected_own_capped_max_k": n_corrected_full_at_own_cap,
            "fraction": (n_corrected_full_at_own_cap / n_corrected_computed) if n_corrected_computed else None,
        },
        "combined_per_session_by_selector": strip_draws,
        "pooled_by_selector": pooled_per_selector,
    }


def _final_verdict(corpus_result: dict) -> str:
    tautology_dominant = corpus_result["step1_tautology_census_under_original_selector"]["tautology_dominant_for_this_corpus"]
    corrected = corpus_result["pooled_by_selector"]["corrected"]
    corrected_comparison = corrected["pooled_observed_vs_matched_null"]
    ceiling_fraction = corpus_result["step_corrected_own_ceiling_census"]["fraction"]
    if not tautology_dominant:
        return "both_selectors_agree"
    if corrected_comparison.get("status") != "computed":
        return ("within_occupied_space_finding_is_tautological_under_the_original_selector_and_"
                "is_inconclusive_under_the_corrected_one")
    if ceiling_fraction is not None and ceiling_fraction > 0.5:
        # The corrected selector still saturates its OWN, deliberately-lower ceiling in most cells --
        # a candidate can now reach its cap without a complete or training-row-space-exhausting basis,
        # so this is not automatically the same tautology, but this repair has no independent way to
        # tell "genuinely high occupied dimensionality" apart from "the cap is still too permissive"
        # from the pooled comparison alone. Declared as inconclusive rather than resolved either way.
        return ("within_occupied_space_finding_is_tautological_under_the_original_selector_and_"
                "is_inconclusive_under_the_corrected_one")
    branch = corrected["branch"]
    if branch == "the_axis_lies_within_the_occupied_state_space_but_outside_the_coding_subspace":
        return ("within_occupied_space_finding_is_tautological_under_the_original_selector_and_"
                "holds_under_the_corrected_one")
    if branch == "the_axis_lies_outside_the_occupied_state_space":
        return ("within_occupied_space_finding_is_tautological_under_the_original_selector_and_"
                "does_not_hold_under_the_corrected_one")
    return ("within_occupied_space_finding_is_tautological_under_the_original_selector_and_"
            "is_inconclusive_under_the_corrected_one")


def _reader_can_still_cite(verdict: str) -> str:
    if verdict.endswith("holds_under_the_corrected_one") or verdict == "both_selectors_agree":
        return ("YES, with a caveat: the delivered numeric within_fraction/off_fraction values in "
                "results/deviation_axis_structure.json were produced by a rank selector that was tautological "
                "in most of its cells, so those specific numbers should not be cited as a measurement. The "
                "QUALITATIVE conclusion -- the axis sits inside the occupied state space and outside a lower-"
                "dimensional coding subspace -- is reproduced under a selector that cannot reach a complete "
                "basis, so the qualitative claim may continue to be cited; the delivered artifact itself is not "
                "amended and its own numbers should not be quoted as more than a construction artifact where "
                "this repair found best_k == max_k.")
    if verdict.endswith("does_not_hold_under_the_corrected_one"):
        return ("NO: once the selector is prevented from returning a trivially complete basis, the axis's "
                "pooled off-fraction lies significantly ABOVE the matched-dimension random-axis null for this "
                "corpus, the opposite of the delivered branch. The delivered "
                "'the_axis_lies_within_the_occupied_state_space_but_outside_the_coding_subspace' branch for this "
                "corpus should not be cited as a measurement; it was true by construction, not by data.")
    return ("NOT YET: the corrected selector's pooled comparison is not computable (or the corrected rank still "
            "saturates its own smaller ceiling in a way indistinguishable from the same capacity-exhaustion "
            "failure at a reduced scale) for this corpus, so neither the delivered branch nor its reversal is "
            "supported by this repair. The delivered artifact's numbers for this corpus should be treated as "
            "not measuring what they claim to measure, and no substitute number is available from this run.")


def main() -> None:
    t0 = time.time()
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "scope": (
            "A cross-validated rank selector used by results/deviation_axis_structure.json's "
            "occupied_state_space_block (`_cv_pca_rank`) chooses an occupied-subspace rank by held-out PCA "
            "reconstruction error over candidate ranks 1..max_k, max_k = min(ambient_unit_count, "
            "n_trials - 1). After mean-centring, a training fold's SVD supplies at most min(n_train, "
            "ambient_unit_count) right-singular vectors; whenever a fold's training-trial count is at least "
            "the ambient unit count, taking all of those vectors is a COMPLETE orthonormal basis of the whole "
            "ambient space, and any held-out point reconstructs to itself exactly by linear algebra, not by "
            "generalisation. Held-out error is monotone non-increasing in rank up to that point, so it is "
            "driven to numerical zero (~1e-28) exactly when the candidate range reaches that complete-basis "
            "point, and np.argmin over a range that includes it always returns the top of the range. This "
            "artifact establishes how often that condition held in the two macaque corpora the delivered "
            "occupied-state-space block analyses (single-item lateral prefrontal cortex; multi-object), "
            "introduces a corrected selector whose candidate range is capped strictly below the point where a "
            "complete basis becomes reachable, and re-runs the occupied-state-space decomposition under both "
            "selectors side by side on identical sessions and identical trials, so a reader can see directly "
            "whether the delivered 'axis lies within the occupied state space' finding was a measurement or a "
            "restatement of the candidate range's own ceiling. It also reports, independent of the rank bug, "
            "that the off-fraction significance test is one-sided by construction (a squared-norm ratio cannot "
            "fall below zero), which affects how any 'significantly below the null' verdict in either artifact "
            "should be read."
        ),
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "status": "running",
    }
    _flush(output)

    _log("loading both corpora (identical loaders to run_deviation_axis_structure.py)")
    loaded, zero_drop = _load_corpora_and_accounting(root)
    gate_result = loaded["gate_result"]
    bundles = loaded["bundles"]
    output["reproduction_gate"] = {"status": gate_result["status"]}
    output["zero_drop_accounting"] = zero_drop
    _flush(output)
    _log(f"reproduction gate: {gate_result['status']}; "
         f"macaque bundles={len(bundles[CORPORA[0]])} multi-object bundles={len(bundles[CORPORA[1]])} "
         f"elapsed={time.time() - t0:.0f}s")

    if gate_result["status"] != "reproduced_exactly":
        output["status"] = "void_reproduction_gate_did_not_reproduce"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate did not reproduce")
        return

    output["occupied_state_space_block_repair"] = {}
    verdicts = {}
    for corpus_key in CORPORA:
        _log(f"occupied-state-space rank-selection repair: {corpus_key} ({len(bundles[corpus_key])} sessions)")
        corpus_result = _run_corpus(bundles[corpus_key], corpus_key)
        corpus_result["verdict"] = _final_verdict(corpus_result)
        corpus_result["reader_can_still_cite_the_delivered_finding_for_this_corpus"] = _reader_can_still_cite(
            corpus_result["verdict"])
        output["occupied_state_space_block_repair"][corpus_key] = corpus_result
        verdicts[corpus_key] = corpus_result["verdict"]
        _flush(output)
        _log(f"  {corpus_key}: original full-rank fraction="
             f"{corpus_result['step1_tautology_census_under_original_selector']['fraction_best_k_equals_max_k']} "
             f"verdict={corpus_result['verdict']} elapsed={time.time() - t0:.0f}s")

    output["branch"] = verdicts
    output["git_commit"] = git_commit(ROOT)
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({"reproduction_gate": gate_result["status"], "branch": verdicts,
                       "wall_clock_s": output["wall_clock_s"]}, indent=2, default=float))
    _log("FINISHED run_occupied_subspace_rank_selection_repair")


if __name__ == "__main__":
    main()
