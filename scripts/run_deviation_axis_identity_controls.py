"""run_deviation_axis_identity_controls.py -- results/deviation_axis_structure.json measured, for the
single-item and multi-object macaque corpora, the direction of the rate-free deviation's residual (its
leading eigenvector, "the axis") and its absolute cosine against four reference directions per session, but
never pooled those four alignments into a judged number and never asked whether the strongest of them (the
axis's near-identity alignment with the session's slow linear drift direction, |cos| ~0.99 in the single-
item corpus) is a discovery or a construction artifact. This module does three things, reusing every
estimator that artifact already delivers unchanged and never rerunning or overwriting it:

Part one pools the four already-computed alignments properly (session-clustered, multi-object corpus
combined within item-count level by trial-count weighting before pooling across sessions) and fires one
pre-declared branch per reference.

Part two asks whether the slow-drift alignment is mechanical: the residual r_i is, by construction, the
component of trial i's direction orthogonal to the session's leave-one-out mean direction, so if that mean
itself translates linearly over the session, every residual is pushed toward the SAME direction the mean
moved along, with no deviation structure required at all. This is tested three ways: (1) an analytic/
synthetic control -- trials generated from isotropic noise around a mean that translates at each session's
own measured drift rate, with NO deviation structure, pushed through the identical axis-estimation code
path; (2) a detrended control -- the linear trend removed from the input before the axis is refit, and
separately a temporally-local leave-one-out reference in place of the whole-session one; (3) a mandatory
fold-based bias-only control on every behavioural cell this module recomputes.

Part three tests, rather than asserts, whether the axis's substantial alignment with the total-spike-count
direction is in tension with the delivered claim that the deviation's MAGNITUDE is rate-free: the
correlation between the deviation magnitude and total spike count, the correlation between the axis's own
signed projection and total spike count, and whether the deviation-behaviour link survives partialling
total spike count, recomputed here rather than quoted.

No estimator is forked. _residual_rows, _unit_residual_matrix, participation_ratio_and_leading_fraction,
leading_eigenvector, _regression_direction, _class_mean_subspace_basis, _regression_subspace_basis,
_class_mean_dict, unit_direction_vectors, _worse_behaviour, _trial_count_weighted, _weighted_combine_draws,
_pool_rotation_statistic, _fold_combined_and_pooled, _contiguous_folds, _bias_only_reproduces,
_macaque_bundles, _watters_bundles, full_reproduction_gate, _reachable_sessions, _panichello_directory,
rate_free_state_deviation, SIGN_TO_WORSE_BEHAVIOUR, MIN_TRIALS_WITH_DEFINED_DIRECTION, N_CV_FOLDS,
MIN_FOLD_TRIALS, N_ROTATION_DRAWS, RESIDUAL_NORM_FLOOR, WATTERS_REGRESSION_DIM, CONTENT_LABEL_K_CLASSES,
SHARP_TEST_MIN_PER_CLASS, stable_seed, permutation_pvalue, partial_correlation_permutation_test,
minimum_detectable_paired_difference, slope_across_sessions_test, fdr_bh and _restore_arrays are every one
imported unchanged from where this project already defines them (the deviation-axis-structure module and
its own upstream dependencies). The only new functions this module introduces are the mechanical-drift
synthetic generator, the linear and local-window detrending controls, the four draws-exposing mirrors of
the delivered alignment-null functions needed to pool a rotation-null test across sessions (identical
formula, only the raw draws returned), and the rate-reconciliation correlations of part three.
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
from provenance import _json_safe  # noqa: E402
from run_deviation_axis_structure import (  # noqa: E402
    CORPORA, MIN_FOLD_TRIALS, N_CV_FOLDS, N_PERM, N_ROTATION_DRAWS, RESIDUAL_NORM_FLOOR, SHARP_TEST_MIN_PER_CLASS,
    SIGN_TO_WORSE_BEHAVIOUR, WATTERS_REGRESSION_DIM, _bias_only_reproduces, _class_mean_dict,
    _class_mean_subspace_basis, _contiguous_folds, _fold_combined_and_pooled, _macaque_bundles,
    _panichello_directory, _pool_rotation_statistic, _reachable_sessions, _regression_direction,
    _regression_subspace_basis, _residual_rows, _trial_count_weighted, _unit_residual_matrix, _watters_bundles,
    _weighted_combine_draws, _worse_behaviour, full_reproduction_gate, leading_eigenvector,
    unit_direction_vectors,
)
from run_deviation_axis_structure import MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402
from run_deviation_axis_structure import _MACAQUE_DELIVERED_RAW_R as MACAQUE_REFERENCE_R  # noqa: E402
from run_deviation_axis_structure import _WATTERS_DELIVERED_RAW_R as WATTERS_REFERENCE_R  # noqa: E402
from run_rate_free_state_geometry_behavior_link import (  # noqa: E402
    MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, rate_free_state_deviation,
)
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference, partial_correlation_permutation_test, permutation_pvalue,
    stable_seed,
)

OUTPUT_PATH = ROOT / "results" / "deviation_axis_identity_controls.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_deviation_axis_identity_controls"
SOURCE_ARTIFACT_PATH = ROOT / "results" / "deviation_axis_structure.json"
ANALYSIS_VERSION = "2026-08-26"

REPRODUCTION_TOLERANCE = 1e-6
N_SYNTHETIC_DRIFT_REPS = 20
LOCAL_WINDOW_HALF_TRIALS = 15
# Every reference cosine in this artifact lives on a [0, 1] scale; a gap smaller than this is smaller than
# the smallest gap distinguished anywhere else in either this artifact or the one it reads from, so it is
# the pre-declared floor below which a null result is reported as underpowered rather than clean.
MEANINGFUL_COSINE_DIFFERENCE = 0.05
MAX_SESSIONS_ENV_VAR = "WM_DYNAMICS_AXIS_IDENTITY_MAX_SESSIONS"

REFERENCE_NAMES = (
    "total_spike_count_direction", "slow_drift_direction", "memorandum_coding_subspace",
    "preceding_trial_class_mean_direction",
)

SLOW_DRIFT_DIRECTION_DEFINITION = (
    "slow_drift_direction is computed, unchanged, by _regression_direction(U, trial_index): an ordinary-"
    "least-squares fit of every unit's L2-normalised per-trial direction vector U (unit_direction_vectors "
    "applied to the session's per-unit activity) against the trial's acquisition-order index (0, 1, 2, ... "
    "within the session -- the same trial_index field every session bundle in this project's deviation "
    "modules carries), keeping only the trial_index regression coefficient (dropping the intercept) and "
    "normalising it to unit length. It is, by construction, the single linear direction along which the "
    "session's average unit-activity direction moves most, in a least-squares sense, over the course of "
    "the session -- i.e. a literal linear-drift-over-time direction, not a physiologically motivated one."
)

POOLED_ALIGNMENT_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "For each of the four reference directions, per session (single-item corpus) or per session combining "
    "item-count levels by trial-count weighting (multi-object corpus), the axis's absolute cosine and a "
    "freshly regenerated >=1000-draw rotation null (the identical formula the source artifact's own "
    "alignment functions use, only the raw draws additionally returned so they can be pooled) are combined "
    "across sessions by the paired sign-flip test, clustered on session, carrying its own confidence "
    "interval and minimum detectable difference at 80% power, and compared against the pooled null by a "
    "two-sided empirical percentile test (the same construction this project's anisotropy pooling already "
    "uses). The fraction of cells clearing their OWN rotation null is reported separately at the finer "
    "session x item-count-level granularity the per-cell significance test actually runs at. Branches, one "
    "per reference:\n"
    "  - the pooled test is not computable -> 'not_computable'.\n"
    "  - the pooled cosine sits significantly ABOVE the pooled null (p<=0.05, mean above the null centre) "
    "-> 'axis_aligns_with_this_reference'.\n"
    "  - the pooled cosine is NOT significantly above the pooled null AND the minimum detectable difference "
    "is below the pre-declared 0.05 absolute-cosine-unit floor -> 'axis_does_not_align_with_this_reference', "
    "a powered clean negative.\n"
    "  - otherwise -> 'inconclusive_below_detection_floor', never quoted alone."
)

DRIFT_MECHANICAL_CONTROL_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per session (single-item corpus) or per session combining item-count levels by trial-count weighting "
    "(multi-object corpus), 20 synthetic trial sets are generated with NO deviation structure at all: "
    "isotropic Gaussian noise (standard deviation matched to the residual scale of the session's own real "
    "linear fit of U against trial_index) around a mean that translates linearly along the session's own "
    "measured linear-trend direction in U, at the session's own measured rate (that OLS fit's slope vector "
    "norm). The identical residual-axis code path and the identical slow_drift_direction construction are "
    "applied to the synthetic trials; the resulting synthetic axis-to-drift-direction cosine is averaged "
    "over the 20 repetitions to give one session-level synthetic alignment, paired against the real "
    "observed alignment from part one for the SAME session. The paired difference (real minus synthetic) "
    "is pooled across sessions by the paired sign-flip test, clustered on session.\n"
    "  - the paired difference is NOT significant (two-sided, 0.05) AND its own minimum detectable "
    "difference is below the pre-declared 0.05 absolute-cosine-unit floor -> "
    "'drift_alignment_is_reproduced_by_a_pure_mechanical_drift_generator_with_no_deviation_structure', "
    "reported as the mechanical explanation.\n"
    "  - the paired difference IS significant and positive (real exceeds the pure-drift baseline) -> "
    "'real_alignment_exceeds_the_pure_drift_baseline'.\n"
    "  - the paired difference IS significant and negative -> 'synthetic_control_exceeds_the_real_"
    "alignment', reported as an anomaly.\n"
    "  - neither significant nor powered -> 'inconclusive_below_detection_floor'."
)

DETRENDED_CONTROL_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Two detrending variants, both applied session-by-session (multi-object corpus: per item-count level, "
    "combined by trial-count weighting): (a) linear_detrend -- each unit's raw per-trial activity has its "
    "own OLS linear trend against trial_index subtracted (intercept kept, only the trial_index-proportional "
    "component removed), then the standard residual-axis code path runs unchanged on the detrended input; "
    "(b) local_window -- the leave-one-out reference direction for trial i is recomputed from only the "
    "+/-15 chronologically nearest OTHER trials rather than every other trial in the session, on the "
    "UNMODIFIED raw activity. For each variant the resulting axis is realigned (fresh >=1000-draw rotation "
    "nulls) against the SAME four reference directions part one used (computed from the original, non-"
    "detrended data, since those references describe properties of the real recording, not of the "
    "detrending choice), and the change in each pooled alignment (variant minus part-one original, paired "
    "per session, paired sign-flip test) is reported with its own minimum detectable difference. Branch per "
    "reference per variant:\n"
    "  - the paired change is NOT significant AND its minimum detectable difference is below the 0.05 "
    "absolute-cosine-unit floor -> 'alignment_survives_detrending_unchanged'.\n"
    "  - the paired change IS significant and negative (detrending reduces the alignment) -> "
    "'detrending_reduces_this_alignment', reported with the size of the reduction.\n"
    "  - the paired change IS significant and positive -> 'detrending_increases_this_alignment', reported "
    "as an anomaly.\n"
    "  - otherwise -> 'inconclusive_below_detection_floor'.\n"
    "Separately, only for the linear_detrend variant, the delivered deviation observable is recomputed on "
    "the detrended activity (rate_free_state_deviation applied unchanged to the detrended input) and its "
    "association with worse behaviour is retested (partial_correlation_permutation_test, pooled across "
    "sessions by the paired sign-flip test) against the standing 0.14 r-unit reference, with its own "
    "minimum detectable difference and a MANDATORY fold-based bias-only control (replacing each held-out "
    "trial's detrended deviation with its own fold's training-trial mean, this project's established bias-"
    "only construction) that VOIDS the cell -- named 'detrended_deviation_behaviour_link_not_separable_"
    "from_a_session_level_offset', never silently dropped -- if it reproduces the real result's sign and "
    "significance."
)

RATE_RECONCILIATION_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per session (single-item corpus) or per session combining item-count levels by trial-count weighting "
    "(multi-object corpus): (1) the correlation between the deviation magnitude d_i (equivalently ||r_i||) "
    "and the trial's total spike count; (2) the correlation between the signed projection of r_i onto the "
    "axis and the trial's total spike count; (3) the deviation's raw association with worse behaviour, "
    "recomputed here rather than quoted; (4) that association after partialling total spike count, also "
    "recomputed here, with a MANDATORY fold-based bias-only control on the partialled residual. Each pooled "
    "across sessions by the paired sign-flip test, clustered on session, with its own confidence interval "
    "and minimum detectable difference at 80% power.\n"
    "  - the partialled cell's bias-only control reproduces its sign and significance -> "
    "'partialled_behavioural_association_not_separable_from_a_session_level_offset', void regardless of "
    "the tests below.\n"
    "  - (1) is NOT significant AND (4) remains significant, same sign as (3), with a direct paired "
    "session-level test of |raw r| minus |partial r| NOT significant -> "
    "'axis_rate_alignment_does_not_compromise_the_rate_free_magnitude'.\n"
    "  - (4) is NOT significant AND its own minimum detectable difference is below the smaller of the "
    "standing 0.14 r-unit reference and this corpus's own delivered raw effect size -> 'axis_rate_"
    "alignment_compromises_the_rate_free_status', a powered finding against the standing claim, reported "
    "plainly.\n"
    "  - (1) IS itself significant -> 'orthogonality_gate_not_reproduced_here', a precondition failure "
    "reported rather than forced onto either branch above.\n"
    "  - none of the above -> 'inconclusive_below_detection_floor'."
)


# =======================================================================================================
# Checkpointing -- identical pattern to run_deviation_axis_structure.py, with a self-contained inverse of
# provenance._json_safe applied on load (this project's established fix for the hazard where a numpy array
# served from a checkpoint comes back as a plain, un-indexable list -- kept local rather than imported
# since the sibling module that historically carried this helper is not a stable dependency to import from).
# =======================================================================================================

def _is_bool_leaf_list(value: list) -> bool:
    return len(value) > 0 and all(isinstance(v, bool) for v in value)


def _is_numeric_leaf_list(value: list) -> bool:
    return len(value) > 0 and all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) for v in value)


def _restore_arrays(value):
    """Inverse of provenance._json_safe for a checkpointed record: JSON has no array type, so every ndarray
    a fit returned was flattened to nested lists (None standing in for a nonfinite entry) before being
    written to the checkpoint file, and comes back as a plain list on reload -- restored here rather than
    trusted to arrive as an array."""
    if isinstance(value, dict):
        return {k: _restore_arrays(v) for k, v in value.items()}
    if isinstance(value, list):
        if _is_bool_leaf_list(value):
            return np.array(value, dtype=bool)
        if _is_numeric_leaf_list(value):
            return np.array([np.nan if v is None else v for v in value], dtype=float)
        restored = [_restore_arrays(v) for v in value]
        if restored and all(isinstance(r, np.ndarray) for r in restored):
            try:
                return np.array(restored)
            except ValueError:
                return restored
        return restored
    return value


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
    return _restore_arrays(data["record"])


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


def _run_checkpointed(unit: str, fit_fn):
    cached = _load_checkpoint(unit)
    if cached is not None:
        return cached
    record = _json_safe(fit_fn())
    _save_checkpoint(unit, record)
    return record


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


# =======================================================================================================
# Draws-exposing mirrors of the delivered alignment-null functions -- identical formula, only the raw
# rotation-null draws are additionally returned (the delivered functions summarise and discard them),
# needed here to pool a rotation-null test across sessions via _pool_rotation_statistic.
# =======================================================================================================

def _vector_reference_alignment_with_draws(a: np.ndarray, ref: np.ndarray | None, n_draws: int, seed_tag: str) -> dict | None:
    if ref is None:
        return None
    n_units = a.shape[0]
    observed = abs(float(np.dot(a, ref)))
    rng = np.random.default_rng(stable_seed(seed_tag))
    g = rng.standard_normal((n_draws, n_units))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    draws = np.abs(g @ ref)
    return {"observed": observed, "draws": draws}


def _subspace_reference_alignment_with_draws(a: np.ndarray, basis: np.ndarray | None, n_draws: int, seed_tag: str) -> dict | None:
    if basis is None:
        return None
    n_units = a.shape[0]
    observed = min(float(np.linalg.norm(basis.T @ a)), 1.0)
    rng = np.random.default_rng(stable_seed(seed_tag))
    g = rng.standard_normal((n_draws, n_units))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    draws = np.minimum(np.linalg.norm(g @ basis, axis=1), 1.0)
    return {"observed": observed, "draws": draws}


def _preceding_reference_alignment_with_draws(a: np.ndarray, U: np.ndarray, labels: np.ndarray, class_mean: dict | None,
                                               n_draws: int, seed_tag: str) -> dict | None:
    if class_mean is None:
        return None
    n_units = U.shape[1]
    n = U.shape[0]
    cosines, refs = [], []
    for t in range(1, n):
        if not (np.isfinite(labels[t]) and np.isfinite(labels[t - 1])):
            continue
        own, prev = labels[t], labels[t - 1]
        if own == prev or prev not in class_mean:
            continue
        m = class_mean[prev]
        mn = np.linalg.norm(m)
        if mn <= 0.0:
            continue
        ref = m / mn
        refs.append(ref)
        cosines.append(abs(float(np.dot(a, ref))))
    if len(cosines) < SHARP_TEST_MIN_PER_CLASS:
        return None
    observed = float(np.mean(cosines))
    refs_arr = np.stack(refs)
    rng = np.random.default_rng(stable_seed(seed_tag))
    g = rng.standard_normal((n_draws, n_units))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    draws = np.mean(np.abs(refs_arr @ g.T), axis=0)
    return {"observed": observed, "draws": draws, "n_qualifying_trials": len(cosines)}


def _alignment_suite(axis: np.ndarray, U: np.ndarray, spike_count: np.ndarray, trial_index: np.ndarray,
                      labels: np.ndarray, cued_theta: np.ndarray | None, seed_tag: str, is_watters: bool) -> dict:
    """The same four reference constructions the source artifact's axis-alignment block uses, applied here
    to whatever (axis, U) pair the caller hands in -- the real axis in part one, a detrended-variant axis
    in part two -- so the identical alignment machinery serves all three parts without being duplicated
    three times."""
    out: dict[str, dict | None] = {}
    ref = _regression_direction(U, spike_count)
    out["total_spike_count_direction"] = _vector_reference_alignment_with_draws(axis, ref, N_ROTATION_DRAWS, f"{seed_tag}|spike")
    ref = _regression_direction(U, trial_index)
    out["slow_drift_direction"] = _vector_reference_alignment_with_draws(axis, ref, N_ROTATION_DRAWS, f"{seed_tag}|drift")
    if is_watters:
        theta = np.mod(cued_theta, 2.0 * np.pi)
        target_2d = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        basis = _regression_subspace_basis(U, target_2d, WATTERS_REGRESSION_DIM)
    else:
        dim = len(np.unique(labels)) - 1
        basis = _class_mean_subspace_basis(U, labels, dim) if dim >= 1 else None
    out["memorandum_coding_subspace"] = _subspace_reference_alignment_with_draws(axis, basis, N_ROTATION_DRAWS, f"{seed_tag}|memorandum")
    class_mean = _class_mean_dict(U, labels)
    out["preceding_trial_class_mean_direction"] = _preceding_reference_alignment_with_draws(
        axis, U, labels, class_mean, N_ROTATION_DRAWS, f"{seed_tag}|preceding")
    return out


# =======================================================================================================
# PART ONE -- pooled alignment
# =======================================================================================================

def _pooled_alignment_macaque_session(bundle: dict) -> dict:
    rows = _residual_rows(bundle["activity_by_unit"])
    if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "too_few_trials", "n_kept": rows["n_kept"]}
    R, idx = _unit_residual_matrix(rows)
    U = unit_direction_vectors(bundle["activity_by_unit"])[idx]
    axis = leading_eigenvector(R)
    tag = f"deviation_axis_identity_controls|pooled_alignment|panichello|{bundle['session']}"
    alignments = _alignment_suite(axis, U, bundle["spike_count"][idx], bundle["trial_index"][idx],
                                   bundle["memorandum_label"][idx], None, tag, False)
    return {"status": "computed", "alignments": alignments, "n_trials_kept": int(R.shape[0]), "n_units": int(R.shape[1])}


def _pooled_alignment_watters_session(bundle: dict) -> dict:
    item_count = bundle["item_count"]
    levels = sorted({int(v) for v in item_count.tolist()})
    per_level: dict[str, dict] = {}
    entries_by_ref: dict[str, list[tuple[int, dict]]] = {k: [] for k in REFERENCE_NAMES}
    for level in levels:
        mask = item_count == float(level)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            per_level[str(level)] = {"status": "too_few_trials", "n_trials": n_level}
            continue
        rows = _residual_rows(bundle["activity_by_unit"][mask])
        if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            per_level[str(level)] = {"status": "too_few_trials_after_residual_floor", "n_trials": n_level}
            continue
        R, idx = _unit_residual_matrix(rows)
        U = unit_direction_vectors(bundle["activity_by_unit"][mask])[idx]
        axis = leading_eigenvector(R)
        tag = f"deviation_axis_identity_controls|pooled_alignment|watters|{bundle['session']}|level{level}"
        alignments = _alignment_suite(axis, U, bundle["spike_count"][mask][idx], bundle["trial_index"][mask][idx],
                                       bundle["memorandum_label"][mask][idx], bundle["cued_theta"][mask][idx], tag, True)
        per_level[str(level)] = {"status": "computed", "n_trials": n_level, "alignments": alignments}
        for ref_name, entry in alignments.items():
            if entry is not None:
                entries_by_ref[ref_name].append((n_level, entry))
    n_levels_tested = sum(1 for v in per_level.values() if v.get("status") == "computed")
    if n_levels_tested == 0:
        return {"status": "no_item_count_level_reaches_the_floor", "per_level": per_level}
    combined = {}
    for ref_name, entries in entries_by_ref.items():
        if not entries:
            combined[ref_name] = None
            continue
        observed = _trial_count_weighted([(n, e["observed"]) for n, e in entries])
        draws = _weighted_combine_draws([(n, e["draws"]) for n, e in entries])
        combined[ref_name] = {"observed": observed, "draws": draws}
    return {"status": "computed", "n_levels_tested": n_levels_tested, "per_level": per_level, "alignments": combined}


def run_part_one(bundles_by_corpus: dict[str, list[dict]]) -> dict:
    out = {}
    for corpus_key, bundles in bundles_by_corpus.items():
        fitter = _pooled_alignment_macaque_session if corpus_key == CORPORA[0] else _pooled_alignment_watters_session
        per_session = []
        for bundle in bundles:
            unit = f"pooled_alignment|{corpus_key}|{bundle['session']}"
            record = _run_checkpointed(unit, lambda b=bundle: fitter(b))
            per_session.append({"session": bundle["session"], **record})

        by_reference = {}
        for ref_name in REFERENCE_NAMES:
            session_records = []
            n_cells_total, n_cells_significant = 0, 0
            for s in per_session:
                if s.get("status") != "computed":
                    continue
                if corpus_key == CORPORA[0]:
                    cells = [s["alignments"].get(ref_name)]
                else:
                    cells = [lv.get("alignments", {}).get(ref_name) for lv in s.get("per_level", {}).values()
                             if lv.get("status") == "computed"]
                for cell in cells:
                    if cell is None or cell.get("observed") is None or cell.get("draws") is None:
                        continue
                    n_cells_total += 1
                    draws_arr = np.asarray(cell["draws"], dtype=float)
                    finite_draws = draws_arr[np.isfinite(draws_arr)]
                    if finite_draws.size >= 10 and cell["observed"] > float(np.mean(finite_draws)):
                        p = float(permutation_pvalue(np.abs(finite_draws - float(np.mean(finite_draws)))
                                                      >= abs(cell["observed"] - float(np.mean(finite_draws)))))
                        if p <= 0.05:
                            n_cells_significant += 1
                combined = s["alignments"].get(ref_name) if corpus_key == CORPORA[0] else s.get("alignments", {}).get(ref_name)
                if combined is not None and combined.get("observed") is not None and combined.get("draws") is not None:
                    session_records.append({"observed": combined["observed"], "null_draws": combined["draws"]})
            pooled = _pool_rotation_statistic(session_records)
            branch = _classify_alignment_branch(pooled)
            by_reference[ref_name] = {
                "pooled": pooled, "branch": branch,
                "n_cells_total": n_cells_total, "n_cells_clearing_own_null": n_cells_significant,
                "fraction_cells_clearing_own_null": (n_cells_significant / n_cells_total) if n_cells_total else None,
            }
        out[corpus_key] = {
            "n_sessions_total": len(bundles), "n_sessions_computed": sum(1 for s in per_session if s.get("status") == "computed"),
            "per_session": per_session, "by_reference": by_reference,
        }
    return out


def _classify_alignment_branch(pooled: dict) -> str:
    if pooled.get("real_pooled", {}).get("status") != "tested":
        return "not_computable"
    if pooled["significant"] and pooled["below_null"] is False:
        return "axis_aligns_with_this_reference"
    mdd = pooled["minimum_detectable_difference_80pct_power"]
    mdd_value = mdd.get("mdd") if isinstance(mdd, dict) and mdd.get("status") == "computed" else None
    if mdd_value is not None and mdd_value < MEANINGFUL_COSINE_DIFFERENCE:
        return "axis_does_not_align_with_this_reference"
    return "inconclusive_below_detection_floor"


# =======================================================================================================
# PART TWO, CONTROL 1 -- analytic/synthetic pure-drift control
# =======================================================================================================

def _linear_trend_vector(U: np.ndarray, covariate: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    n = U.shape[0]
    design = np.column_stack([np.ones(n), covariate])
    coef, *_ = np.linalg.lstsq(design, U, rcond=None)
    fitted = design @ coef
    resid_sd = float(np.std(U - fitted))
    return coef[0], coef[1], resid_sd


def _synthetic_pure_drift_axis_to_drift_alignment(n_units: int, trial_index: np.ndarray, intercept: np.ndarray,
                                                    beta: np.ndarray, noise_sigma: float, seed: int) -> float | None:
    rng = np.random.default_rng(seed)
    mean_traj = intercept[None, :] + trial_index[:, None] * beta[None, :]
    synth_activity = mean_traj + rng.standard_normal((len(trial_index), n_units)) * max(noise_sigma, 1e-12)
    rows = _residual_rows(synth_activity)
    if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    R, idx = _unit_residual_matrix(rows)
    U_synth = unit_direction_vectors(synth_activity)[idx]
    axis_synth = leading_eigenvector(R)
    ref_synth = _regression_direction(U_synth, trial_index[idx])
    if ref_synth is None:
        return None
    return abs(float(np.dot(axis_synth, ref_synth)))


def _drift_mechanical_level(activity: np.ndarray, trial_index: np.ndarray, seed_tag: str) -> dict | None:
    rows = _residual_rows(activity)
    if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    R, idx = _unit_residual_matrix(rows)
    U = unit_direction_vectors(activity)[idx]
    axis = leading_eigenvector(R)
    t = trial_index[idx]
    ref_real = _regression_direction(U, t)
    if ref_real is None:
        return None
    real_observed = abs(float(np.dot(axis, ref_real)))
    intercept, beta, noise_sigma = _linear_trend_vector(U, t)
    n_units = U.shape[1]
    reps = []
    for rep in range(N_SYNTHETIC_DRIFT_REPS):
        seed = stable_seed(f"{seed_tag}|rep{rep}")
        val = _synthetic_pure_drift_axis_to_drift_alignment(n_units, t, intercept, beta, noise_sigma, seed)
        if val is not None:
            reps.append(val)
    if not reps:
        return None
    return {
        "real_observed": real_observed, "synthetic_mean_over_reps": float(np.mean(reps)),
        "n_synthetic_reps_computed": len(reps), "measured_drift_rate": float(np.linalg.norm(beta)),
        "residual_noise_sd_used": noise_sigma, "n_trials": int(R.shape[0]),
    }


def _drift_mechanical_macaque_session(bundle: dict) -> dict:
    tag = f"deviation_axis_identity_controls|drift_mechanical|panichello|{bundle['session']}"
    level = _drift_mechanical_level(bundle["activity_by_unit"], bundle["trial_index"], tag)
    if level is None:
        return {"status": "too_few_trials"}
    return {"status": "computed", **level}


def _drift_mechanical_watters_session(bundle: dict) -> dict:
    item_count = bundle["item_count"]
    levels = sorted({int(v) for v in item_count.tolist()})
    per_level, real_entries, synth_entries = {}, [], []
    for level_val in levels:
        mask = item_count == float(level_val)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            per_level[str(level_val)] = {"status": "too_few_trials", "n_trials": n_level}
            continue
        tag = f"deviation_axis_identity_controls|drift_mechanical|watters|{bundle['session']}|level{level_val}"
        result = _drift_mechanical_level(bundle["activity_by_unit"][mask], bundle["trial_index"][mask], tag)
        if result is None:
            per_level[str(level_val)] = {"status": "not_computable", "n_trials": n_level}
            continue
        per_level[str(level_val)] = {"status": "computed", "n_trials": n_level, **result}
        real_entries.append((n_level, result["real_observed"]))
        synth_entries.append((n_level, result["synthetic_mean_over_reps"]))
    n_levels_tested = sum(1 for v in per_level.values() if v.get("status") == "computed")
    if n_levels_tested == 0:
        return {"status": "no_item_count_level_reaches_the_floor", "per_level": per_level}
    return {
        "status": "computed", "n_levels_tested": n_levels_tested, "per_level": per_level,
        "real_observed": _trial_count_weighted(real_entries), "synthetic_mean_over_reps": _trial_count_weighted(synth_entries),
    }


def run_drift_mechanical_control(bundles: list[dict], corpus_key: str) -> dict:
    fitter = _drift_mechanical_macaque_session if corpus_key == CORPORA[0] else _drift_mechanical_watters_session
    per_session = []
    for bundle in bundles:
        unit = f"drift_mechanical|{corpus_key}|{bundle['session']}"
        record = _run_checkpointed(unit, lambda b=bundle: fitter(b))
        per_session.append({"session": bundle["session"], **record})

    diffs = [s["real_observed"] - s["synthetic_mean_over_reps"] for s in per_session if s.get("status") == "computed"]
    pooled_diff = slope_across_sessions_test(diffs, alternative="two-sided") if diffs else {"status": "not_computed"}
    mdd = minimum_detectable_paired_difference(diffs) if len(diffs) >= 2 else {"status": "not_computable"}
    if pooled_diff.get("status") == "tested" and not pooled_diff["significant"]:
        mdd_value = mdd.get("mdd") if mdd.get("status") == "computed" else None
        branch = ("drift_alignment_is_reproduced_by_a_pure_mechanical_drift_generator_with_no_deviation_structure"
                  if mdd_value is not None and mdd_value < MEANINGFUL_COSINE_DIFFERENCE
                  else "inconclusive_below_detection_floor")
    elif pooled_diff.get("status") == "tested" and pooled_diff["significant"]:
        branch = ("real_alignment_exceeds_the_pure_drift_baseline" if pooled_diff["mean_value"] > 0.0
                  else "synthetic_control_exceeds_the_real_alignment")
    else:
        branch = "inconclusive_below_detection_floor"

    return {
        "decision_rule_declared_before_fitting": DRIFT_MECHANICAL_CONTROL_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "n_sessions_total": len(bundles), "n_sessions_computed": sum(1 for s in per_session if s.get("status") == "computed"),
        "per_session": per_session, "real_minus_synthetic_pooled": pooled_diff,
        "minimum_detectable_paired_difference_at_80pct_power": mdd, "branch": branch,
    }


# =======================================================================================================
# PART TWO, CONTROL 2 -- detrended controls
# =======================================================================================================

def _linear_detrend_activity(activity_by_unit: np.ndarray, trial_index: np.ndarray) -> np.ndarray:
    n = activity_by_unit.shape[0]
    t_centered = trial_index - trial_index.mean()
    design = np.column_stack([np.ones(n), t_centered])
    coef, *_ = np.linalg.lstsq(design, activity_by_unit, rcond=None)
    slope = coef[1]
    return activity_by_unit - np.outer(t_centered, slope)


def _local_window_residual_rows(activity_by_unit: np.ndarray, half_window: int, floor: float = RESIDUAL_NORM_FLOOR) -> dict:
    """Same algebra _residual_rows applies (r_i = u_i - (u_i . m_i) m_i), with m_i recomputed from only the
    +/- half_window chronologically NEAREST other trials instead of the whole-session leave-one-out mean --
    a temporally-local reference that cannot be dominated by a session-wide linear translation the way the
    whole-session mean can."""
    activity = np.asarray(activity_by_unit, dtype=float)
    n = activity.shape[0]
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)
    valid = ~np.isnan(unit_vectors).any(axis=1)
    loo_mean_normalized = np.full_like(unit_vectors, np.nan)
    for i in range(n):
        if not valid[i]:
            continue
        lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
        neighbours = [j for j in range(lo, hi) if j != i and valid[j]]
        if not neighbours:
            continue
        m = unit_vectors[neighbours].mean(axis=0)
        mn = np.linalg.norm(m)
        if mn == 0.0:
            continue
        loo_mean_normalized[i] = m / mn
    with np.errstate(invalid="ignore"):
        cosine = np.einsum("ij,ij->i", unit_vectors, loo_mean_normalized)
        residual = unit_vectors - cosine[:, None] * loo_mean_normalized
    finite = np.isfinite(cosine)
    residual_norm = np.linalg.norm(residual, axis=1)
    with np.errstate(invalid="ignore"):
        above_floor = finite & (residual_norm >= floor)
    return {
        "residual": residual, "residual_norm": residual_norm, "keep": above_floor,
        "n_trials_with_defined_direction": int(finite.sum()), "n_kept": int(above_floor.sum()),
    }


def _detrend_variant_level(activity: np.ndarray, trial_index: np.ndarray, labels: np.ndarray, spike_count: np.ndarray,
                            cued_theta: np.ndarray | None, variant: str, seed_tag: str, is_watters: bool) -> dict | None:
    if variant == "linear_detrend":
        used_activity = _linear_detrend_activity(activity, trial_index)
        rows = _residual_rows(used_activity)
        n_kept = rows["n_kept"]
    else:
        used_activity = activity
        rows = _local_window_residual_rows(activity, LOCAL_WINDOW_HALF_TRIALS)
        n_kept = rows["n_kept"]
    if n_kept < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    R, idx = _unit_residual_matrix(rows)
    U = unit_direction_vectors(used_activity)[idx]
    axis = leading_eigenvector(R)
    alignments = _alignment_suite(axis, U, spike_count[idx], trial_index[idx], labels[idx],
                                   cued_theta[idx] if cued_theta is not None else None, seed_tag, is_watters)
    return {"alignments": alignments, "n_trials_kept": int(R.shape[0])}


def _detrend_variant_macaque_session(bundle: dict, variant: str) -> dict:
    tag = f"deviation_axis_identity_controls|detrend|{variant}|panichello|{bundle['session']}"
    level = _detrend_variant_level(bundle["activity_by_unit"], bundle["trial_index"], bundle["memorandum_label"],
                                    bundle["spike_count"], None, variant, tag, False)
    if level is None:
        return {"status": "too_few_trials"}
    return {"status": "computed", **level}


def _detrend_variant_watters_session(bundle: dict, variant: str) -> dict:
    item_count = bundle["item_count"]
    levels = sorted({int(v) for v in item_count.tolist()})
    per_level: dict[str, dict] = {}
    entries_by_ref: dict[str, list[tuple[int, dict]]] = {k: [] for k in REFERENCE_NAMES}
    for level_val in levels:
        mask = item_count == float(level_val)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            per_level[str(level_val)] = {"status": "too_few_trials", "n_trials": n_level}
            continue
        tag = f"deviation_axis_identity_controls|detrend|{variant}|watters|{bundle['session']}|level{level_val}"
        level_result = _detrend_variant_level(bundle["activity_by_unit"][mask], bundle["trial_index"][mask],
                                               bundle["memorandum_label"][mask], bundle["spike_count"][mask],
                                               bundle["cued_theta"][mask], variant, tag, True)
        if level_result is None:
            per_level[str(level_val)] = {"status": "not_computable", "n_trials": n_level}
            continue
        per_level[str(level_val)] = {"status": "computed", "n_trials": n_level, **level_result}
        for ref_name, entry in level_result["alignments"].items():
            if entry is not None:
                entries_by_ref[ref_name].append((n_level, entry))
    n_levels_tested = sum(1 for v in per_level.values() if v.get("status") == "computed")
    if n_levels_tested == 0:
        return {"status": "no_item_count_level_reaches_the_floor", "per_level": per_level}
    combined = {}
    for ref_name, entries in entries_by_ref.items():
        if not entries:
            combined[ref_name] = None
            continue
        observed = _trial_count_weighted([(n, e["observed"]) for n, e in entries])
        draws = _weighted_combine_draws([(n, e["draws"]) for n, e in entries])
        combined[ref_name] = {"observed": observed, "draws": draws}
    return {"status": "computed", "n_levels_tested": n_levels_tested, "per_level": per_level, "alignments": combined}


def _session_alignment_dict(record: dict, corpus_key: str) -> dict | None:
    if record.get("status") != "computed":
        return None
    return record.get("alignments")


def run_detrend_variant(bundles: list[dict], corpus_key: str, variant: str, original_by_session: dict[str, dict]) -> dict:
    fitter = _detrend_variant_macaque_session if corpus_key == CORPORA[0] else _detrend_variant_watters_session
    per_session = []
    for bundle in bundles:
        unit = f"detrend|{variant}|{corpus_key}|{bundle['session']}"
        record = _run_checkpointed(unit, lambda b=bundle: fitter(b, variant))
        per_session.append({"session": bundle["session"], **record})

    by_reference = {}
    for ref_name in REFERENCE_NAMES:
        diffs = []
        for s in per_session:
            variant_alignment = _session_alignment_dict(s, corpus_key)
            if variant_alignment is None:
                continue
            variant_entry = variant_alignment.get(ref_name)
            original_entry = (original_by_session.get(s["session"]) or {}).get(ref_name)
            if variant_entry is None or original_entry is None:
                continue
            if variant_entry.get("observed") is None or original_entry.get("observed") is None:
                continue
            diffs.append(variant_entry["observed"] - original_entry["observed"])
        pooled = slope_across_sessions_test(diffs, alternative="two-sided") if diffs else {"status": "not_computed"}
        mdd = minimum_detectable_paired_difference(diffs) if len(diffs) >= 2 else {"status": "not_computable"}
        if pooled.get("status") == "tested" and not pooled["significant"]:
            mdd_value = mdd.get("mdd") if mdd.get("status") == "computed" else None
            branch = ("alignment_survives_detrending_unchanged" if mdd_value is not None and mdd_value < MEANINGFUL_COSINE_DIFFERENCE
                      else "inconclusive_below_detection_floor")
        elif pooled.get("status") == "tested" and pooled["significant"]:
            branch = "detrending_reduces_this_alignment" if pooled["mean_value"] < 0.0 else "detrending_increases_this_alignment"
        else:
            branch = "inconclusive_below_detection_floor"
        by_reference[ref_name] = {"paired_change_pooled": pooled, "minimum_detectable_difference_80pct_power": mdd,
                                   "branch": branch, "n_paired_sessions": len(diffs)}

    return {
        "variant": variant,
        "n_sessions_total": len(bundles), "n_sessions_computed": sum(1 for s in per_session if s.get("status") == "computed"),
        "per_session": per_session, "by_reference": by_reference,
    }


def _detrended_deviation_behaviour_macaque_session(bundle: dict, sign: float) -> dict:
    trial_index = bundle["trial_index"]
    detrended = _linear_detrend_activity(bundle["activity_by_unit"], trial_index)
    d_detrended = rate_free_state_deviation(detrended)
    finite = np.isfinite(d_detrended)
    n = int(finite.sum())
    if n < N_CV_FOLDS * MIN_FOLD_TRIALS:
        return {"status": "too_few_trials", "n_finite": n}
    feature = d_detrended[finite]
    worse = _worse_behaviour(bundle, sign)[finite]
    folds = _contiguous_folds(n, N_CV_FOLDS)
    bias_only = np.full(n, np.nan)
    for f in range(N_CV_FOLDS):
        train, test = folds != f, folds == f
        if int(train.sum()) < MIN_FOLD_TRIALS or not test.any():
            continue
        bias_only[test] = float(np.mean(feature[train]))
    tag = f"deviation_axis_identity_controls|detrend_behaviour|panichello|{bundle['session']}"
    real = _fold_combined_and_pooled(feature, worse, folds, f"{tag}|real")
    bias = _fold_combined_and_pooled(bias_only, worse, folds, f"{tag}|bias")
    return {"status": "computed", "n_trials": n, "real": real, "bias_only": bias}


def _detrended_deviation_behaviour_watters_session(bundle: dict, sign: float) -> dict:
    item_count = bundle["item_count"]
    levels = sorted({int(v) for v in item_count.tolist()})
    worse_full = _worse_behaviour(bundle, sign)
    per_level: dict[str, dict] = {}
    real_entries, bias_entries = [], []
    for level_val in levels:
        mask = item_count == float(level_val)
        n_level = int(mask.sum())
        if n_level < N_CV_FOLDS * MIN_FOLD_TRIALS:
            per_level[str(level_val)] = {"status": "too_few_trials", "n_trials": n_level}
            continue
        activity = bundle["activity_by_unit"][mask]
        trial_index = bundle["trial_index"][mask]
        detrended = _linear_detrend_activity(activity, trial_index)
        d_detrended = rate_free_state_deviation(detrended)
        finite = np.isfinite(d_detrended)
        n = int(finite.sum())
        if n < N_CV_FOLDS * MIN_FOLD_TRIALS:
            per_level[str(level_val)] = {"status": "too_few_trials_after_finite_filter", "n_trials": n_level}
            continue
        feature = d_detrended[finite]
        worse = worse_full[mask][finite]
        folds = _contiguous_folds(n, N_CV_FOLDS)
        bias_only = np.full(n, np.nan)
        for f in range(N_CV_FOLDS):
            train, test = folds != f, folds == f
            if int(train.sum()) < MIN_FOLD_TRIALS or not test.any():
                continue
            bias_only[test] = float(np.mean(feature[train]))
        tag = f"deviation_axis_identity_controls|detrend_behaviour|watters|{bundle['session']}|level{level_val}"
        real = _fold_combined_and_pooled(feature, worse, folds, f"{tag}|real")
        bias = _fold_combined_and_pooled(bias_only, worse, folds, f"{tag}|bias")
        per_level[str(level_val)] = {"status": "computed", "n_trials": n_level, "real": real, "bias_only": bias}
        if real.get("status") == "computed" and real["pooled_across_fold"].get("status") == "computed":
            real_entries.append((n_level, real["pooled_across_fold"]["r"]))
        if bias.get("status") == "computed" and bias["pooled_across_fold"].get("status") == "computed":
            bias_entries.append((n_level, bias["pooled_across_fold"]["r"]))
    n_levels_tested = sum(1 for v in per_level.values() if v.get("status") == "computed")
    if n_levels_tested == 0:
        return {"status": "no_item_count_level_reaches_the_floor", "per_level": per_level}
    real_r = _trial_count_weighted(real_entries) if real_entries else None
    bias_r = _trial_count_weighted(bias_entries) if bias_entries else None
    return {
        "status": "computed", "per_level": per_level,
        "real": {"pooled_across_fold": {"status": "computed" if real_r is not None else "not_computable", "r": real_r}},
        "bias_only": {"pooled_across_fold": {"status": "computed" if bias_r is not None else "not_computable", "r": bias_r}},
    }


def _pool_behaviour_key(per_session: list[dict], key: str) -> dict:
    vals = []
    for s in per_session:
        entry = (s.get(key) or {})
        pooled = entry.get("pooled_across_fold", {})
        if pooled.get("status") == "computed":
            vals.append(pooled["r"])
    pooled_test = slope_across_sessions_test(vals, alternative="two-sided") if vals else {"status": "not_computed"}
    if len(vals) >= 2:
        pooled_test["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(vals)
    return pooled_test


def run_detrended_deviation_behaviour(bundles: list[dict], corpus_key: str) -> dict:
    fitter = _detrended_deviation_behaviour_macaque_session if corpus_key == CORPORA[0] else _detrended_deviation_behaviour_watters_session
    sign = SIGN_TO_WORSE_BEHAVIOUR[corpus_key]
    per_session = []
    for bundle in bundles:
        unit = f"detrend_behaviour|{corpus_key}|{bundle['session']}"
        record = _run_checkpointed(unit, lambda b=bundle: fitter(b, sign))
        per_session.append({"session": bundle["session"], **record})

    pooled_real = _pool_behaviour_key(per_session, "real")
    pooled_bias = _pool_behaviour_key(per_session, "bias_only")
    voids = _bias_only_reproduces(pooled_real, pooled_bias)
    mdd = pooled_real.get("minimum_detectable_paired_difference_at_80pct_power", {})
    mdd_value = mdd.get("mdd") if isinstance(mdd, dict) and mdd.get("status") == "computed" else None
    reference_r = MEANINGFUL_EFFECT_THRESHOLD_R_UNITS
    if voids:
        branch = "detrended_deviation_behaviour_link_not_separable_from_a_session_level_offset"
    elif pooled_real.get("status") == "tested" and pooled_real["significant"]:
        branch = "detrended_deviation_still_predicts_behaviour"
    elif pooled_real.get("status") == "tested" and mdd_value is not None and mdd_value < reference_r:
        branch = "detrended_deviation_does_not_predict_behaviour_powered_null"
    else:
        branch = "inconclusive_below_detection_floor"

    return {
        "n_sessions_total": len(bundles), "n_sessions_computed": sum(1 for s in per_session if s.get("status") == "computed"),
        "per_session": per_session, "pooled_real": pooled_real, "pooled_bias_only": pooled_bias,
        "bias_only_reproduces_real": voids, "reference_effect_r_units": reference_r, "branch": branch,
    }


# =======================================================================================================
# PART THREE -- rate reconciliation
# =======================================================================================================

def _rate_reconciliation_level(activity: np.ndarray, deviation: np.ndarray, spike_count: np.ndarray,
                                worse: np.ndarray, seed_tag: str) -> dict | None:
    rows = _residual_rows(activity)
    if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    R, idx = _unit_residual_matrix(rows)
    axis = leading_eigenvector(R)
    d = deviation[idx]
    sc = spike_count[idx]
    proj = rows["residual"][idx] @ axis
    w = worse[idx]
    n = int(R.shape[0])

    def _rng(name: str) -> np.random.Generator:
        return np.random.default_rng(stable_seed(f"{seed_tag}|{name}"))

    corr_d_spike = partial_correlation_permutation_test(d, sc, [], N_PERM, _rng("d_spike"))
    corr_proj_spike = partial_correlation_permutation_test(proj, sc, [], N_PERM, _rng("proj_spike"))
    raw_behaviour = partial_correlation_permutation_test(w, d, [], N_PERM, _rng("raw_behaviour"))
    partial_behaviour = partial_correlation_permutation_test(w, d, [sc], N_PERM, _rng("partial_behaviour"))

    real_fold, bias_fold = {"status": "too_few_trials"}, {"status": "too_few_trials"}
    if n >= N_CV_FOLDS * MIN_FOLD_TRIALS:
        design_full = np.column_stack([np.ones(n), sc])
        d_resid = d - design_full @ np.linalg.lstsq(design_full, d, rcond=None)[0]
        folds = _contiguous_folds(n, N_CV_FOLDS)
        bias_only = np.full(n, np.nan)
        for f in range(N_CV_FOLDS):
            train, test = folds != f, folds == f
            if int(train.sum()) < MIN_FOLD_TRIALS or not test.any():
                continue
            bias_only[test] = float(np.mean(d_resid[train]))
        real_fold = _fold_combined_and_pooled(d_resid, w, folds, f"{seed_tag}|partial_real")
        bias_fold = _fold_combined_and_pooled(bias_only, w, folds, f"{seed_tag}|partial_bias")

    return {
        "n_trials": n, "corr_deviation_vs_spike_count": corr_d_spike, "corr_axis_projection_vs_spike_count": corr_proj_spike,
        "raw_behaviour_association": raw_behaviour, "partial_behaviour_association_controlling_spike_count": partial_behaviour,
        "partial_behaviour_fold_based": {"real": real_fold, "bias_only": bias_fold},
    }


def _rate_reconciliation_macaque_session(bundle: dict) -> dict:
    sign = SIGN_TO_WORSE_BEHAVIOUR[CORPORA[0]]
    worse = _worse_behaviour(bundle, sign)
    tag = f"deviation_axis_identity_controls|rate_reconciliation|panichello|{bundle['session']}"
    level = _rate_reconciliation_level(bundle["activity_by_unit"], bundle["deviation"], bundle["spike_count"], worse, tag)
    if level is None:
        return {"status": "too_few_trials"}
    return {"status": "computed", **level}


def _rate_reconciliation_watters_session(bundle: dict) -> dict:
    sign = SIGN_TO_WORSE_BEHAVIOUR[CORPORA[1]]
    worse = _worse_behaviour(bundle, sign)
    item_count = bundle["item_count"]
    levels = sorted({int(v) for v in item_count.tolist()})
    per_level: dict[str, dict] = {}
    keys = ("corr_deviation_vs_spike_count", "corr_axis_projection_vs_spike_count",
            "raw_behaviour_association", "partial_behaviour_association_controlling_spike_count")
    entries: dict[str, list[tuple[int, float]]] = {k: [] for k in keys}
    fold_entries: dict[str, list[tuple[int, float]]] = {"real": [], "bias_only": []}
    for level_val in levels:
        mask = item_count == float(level_val)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            per_level[str(level_val)] = {"status": "too_few_trials", "n_trials": n_level}
            continue
        tag = f"deviation_axis_identity_controls|rate_reconciliation|watters|{bundle['session']}|level{level_val}"
        level_result = _rate_reconciliation_level(bundle["activity_by_unit"][mask], bundle["deviation"][mask],
                                                    bundle["spike_count"][mask], worse[mask], tag)
        if level_result is None:
            per_level[str(level_val)] = {"status": "not_computable", "n_trials": n_level}
            continue
        per_level[str(level_val)] = {"status": "computed", "n_trials": n_level, **level_result}
        for key in keys:
            entry = level_result[key]
            if entry.get("status") == "computed":
                entries[key].append((n_level, entry["r"]))
        for fk in fold_entries:
            entry = level_result["partial_behaviour_fold_based"][fk]
            if entry.get("status") == "computed" and entry["pooled_across_fold"].get("status") == "computed":
                fold_entries[fk].append((n_level, entry["pooled_across_fold"]["r"]))
    n_levels_tested = sum(1 for v in per_level.values() if v.get("status") == "computed")
    if n_levels_tested == 0:
        return {"status": "no_item_count_level_reaches_the_floor", "per_level": per_level}
    combined = {key: (_trial_count_weighted(entries[key]) if entries[key] else None) for key in keys}
    fold_combined = {fk: (_trial_count_weighted(fold_entries[fk]) if fold_entries[fk] else None) for fk in fold_entries}
    return {"status": "computed", "n_levels_tested": n_levels_tested, "per_level": per_level,
            "combined_r": combined, "combined_partial_fold_r": fold_combined}


def run_part_three(bundles: list[dict], corpus_key: str, delivered_reference_effect: float) -> dict:
    fitter = _rate_reconciliation_macaque_session if corpus_key == CORPORA[0] else _rate_reconciliation_watters_session
    per_session = []
    for bundle in bundles:
        unit = f"rate_reconciliation|{corpus_key}|{bundle['session']}"
        record = _run_checkpointed(unit, lambda b=bundle: fitter(b))
        per_session.append({"session": bundle["session"], **record})

    def _pool_scalar(extract) -> dict:
        vals = [extract(s) for s in per_session if extract(s) is not None]
        test = slope_across_sessions_test(vals, alternative="two-sided") if vals else {"status": "not_computed"}
        if len(vals) >= 2:
            test["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(vals)
        return test

    if corpus_key == CORPORA[0]:
        pooled_d_spike = _pool_scalar(lambda s: s.get("corr_deviation_vs_spike_count", {}).get("r")
                                       if s.get("status") == "computed" and s["corr_deviation_vs_spike_count"].get("status") == "computed" else None)
        pooled_proj_spike = _pool_scalar(lambda s: s.get("corr_axis_projection_vs_spike_count", {}).get("r")
                                          if s.get("status") == "computed" and s["corr_axis_projection_vs_spike_count"].get("status") == "computed" else None)
        pooled_raw = _pool_scalar(lambda s: s.get("raw_behaviour_association", {}).get("r")
                                   if s.get("status") == "computed" and s["raw_behaviour_association"].get("status") == "computed" else None)
        pooled_partial = _pool_scalar(lambda s: s.get("partial_behaviour_association_controlling_spike_count", {}).get("r")
                                       if s.get("status") == "computed" and s["partial_behaviour_association_controlling_spike_count"].get("status") == "computed" else None)
        pooled_partial_real_fold = _pool_scalar(
            lambda s: s.get("partial_behaviour_fold_based", {}).get("real", {}).get("pooled_across_fold", {}).get("r")
            if s.get("status") == "computed" and s["partial_behaviour_fold_based"]["real"].get("status") == "computed"
            and s["partial_behaviour_fold_based"]["real"]["pooled_across_fold"].get("status") == "computed" else None)
        pooled_partial_bias_fold = _pool_scalar(
            lambda s: s.get("partial_behaviour_fold_based", {}).get("bias_only", {}).get("pooled_across_fold", {}).get("r")
            if s.get("status") == "computed" and s["partial_behaviour_fold_based"]["bias_only"].get("status") == "computed"
            and s["partial_behaviour_fold_based"]["bias_only"]["pooled_across_fold"].get("status") == "computed" else None)
        raw_minus_partial = [
            abs(s["raw_behaviour_association"]["r"]) - abs(s["partial_behaviour_association_controlling_spike_count"]["r"])
            for s in per_session if s.get("status") == "computed"
            and s["raw_behaviour_association"].get("status") == "computed"
            and s["partial_behaviour_association_controlling_spike_count"].get("status") == "computed"
        ]
    else:
        pooled_d_spike = _pool_scalar(lambda s: s.get("combined_r", {}).get("corr_deviation_vs_spike_count") if s.get("status") == "computed" else None)
        pooled_proj_spike = _pool_scalar(lambda s: s.get("combined_r", {}).get("corr_axis_projection_vs_spike_count") if s.get("status") == "computed" else None)
        pooled_raw = _pool_scalar(lambda s: s.get("combined_r", {}).get("raw_behaviour_association") if s.get("status") == "computed" else None)
        pooled_partial = _pool_scalar(lambda s: s.get("combined_r", {}).get("partial_behaviour_association_controlling_spike_count") if s.get("status") == "computed" else None)
        pooled_partial_real_fold = _pool_scalar(lambda s: s.get("combined_partial_fold_r", {}).get("real") if s.get("status") == "computed" else None)
        pooled_partial_bias_fold = _pool_scalar(lambda s: s.get("combined_partial_fold_r", {}).get("bias_only") if s.get("status") == "computed" else None)
        raw_minus_partial = [
            abs(s["combined_r"]["raw_behaviour_association"]) - abs(s["combined_r"]["partial_behaviour_association_controlling_spike_count"])
            for s in per_session if s.get("status") == "computed"
            and s["combined_r"].get("raw_behaviour_association") is not None
            and s["combined_r"].get("partial_behaviour_association_controlling_spike_count") is not None
        ]

    paired_raw_vs_partial = slope_across_sessions_test(raw_minus_partial, alternative="two-sided") if raw_minus_partial else {"status": "not_computed"}
    if len(raw_minus_partial) >= 2:
        paired_raw_vs_partial["minimum_detectable_paired_difference_at_80pct_power"] = minimum_detectable_paired_difference(raw_minus_partial)

    bias_reproduces = _bias_only_reproduces(pooled_partial_real_fold, pooled_partial_bias_fold)
    branch = _classify_rate_reconciliation_branch(
        pooled_d_spike, pooled_raw, pooled_partial, bias_reproduces, paired_raw_vs_partial,
        min(MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, delivered_reference_effect))

    return {
        "decision_rule_declared_before_fitting": RATE_RECONCILIATION_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "n_sessions_total": len(bundles), "n_sessions_computed": sum(1 for s in per_session if s.get("status") == "computed"),
        "per_session": per_session,
        "pooled_deviation_vs_spike_count": pooled_d_spike, "pooled_axis_projection_vs_spike_count": pooled_proj_spike,
        "pooled_raw_behaviour_association": pooled_raw, "pooled_partial_behaviour_association": pooled_partial,
        "pooled_partial_fold_based_real": pooled_partial_real_fold, "pooled_partial_fold_based_bias_only": pooled_partial_bias_fold,
        "bias_only_reproduces_partial_association": bias_reproduces,
        "paired_raw_minus_partial_effect_size": paired_raw_vs_partial,
        "reference_effect_r_units_used": min(MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, delivered_reference_effect),
        "branch": branch,
    }


def _classify_rate_reconciliation_branch(pooled_d_spike: dict, pooled_raw: dict, pooled_partial: dict,
                                          bias_reproduces: bool, paired_raw_vs_partial: dict, reference_effect: float) -> str:
    if bias_reproduces:
        return "partialled_behavioural_association_not_separable_from_a_session_level_offset"
    d_spike_significant = bool(pooled_d_spike.get("status") == "tested" and pooled_d_spike.get("significant"))
    if d_spike_significant:
        return "orthogonality_gate_not_reproduced_here"
    partial_significant = bool(pooled_partial.get("status") == "tested" and pooled_partial.get("significant"))
    raw_significant = bool(pooled_raw.get("status") == "tested" and pooled_raw.get("significant"))
    same_sign = bool(raw_significant and partial_significant
                      and (pooled_raw["mean_value"] > 0.0) == (pooled_partial["mean_value"] > 0.0))
    paired_not_significant = bool(paired_raw_vs_partial.get("status") == "tested" and not paired_raw_vs_partial.get("significant"))
    if partial_significant and same_sign and paired_not_significant:
        return "axis_rate_alignment_does_not_compromise_the_rate_free_magnitude"
    mdd = pooled_partial.get("minimum_detectable_paired_difference_at_80pct_power")
    mdd_value = mdd.get("mdd") if isinstance(mdd, dict) and mdd.get("status") == "computed" else None
    if (pooled_partial.get("status") == "tested") and (not partial_significant) and mdd_value is not None and mdd_value < reference_effect:
        return "axis_rate_alignment_compromises_the_rate_free_status"
    return "inconclusive_below_detection_floor"


# =======================================================================================================
# Reproduction gate against the source artifact, and zero-drop accounting
# =======================================================================================================

def _reproduction_gate_against_source(part_one: dict, session_limited: bool) -> dict:
    if not SOURCE_ARTIFACT_PATH.exists():
        return {"status": "source_artifact_missing"}
    source = json.loads(SOURCE_ARTIFACT_PATH.read_text())
    worst_diff, n_compared, detail = 0.0, 0, []
    for corpus_key in CORPORA:
        src_block = source.get("axis_alignment_block", {}).get(corpus_key, {})
        src_by_session = {s["session"]: s for s in src_block.get("per_session", [])}
        for rec in part_one.get(corpus_key, {}).get("per_session", []):
            src = src_by_session.get(rec["session"])
            if src is None or src.get("status") != "computed" or rec.get("status") != "computed":
                continue
            for ref_name, entry in rec["alignments"].items():
                if entry is None or entry.get("observed") is None:
                    continue
                src_entry = src.get("alignments", {}).get(ref_name)
                if src_entry is None or src_entry.get("observed") is None:
                    continue
                diff = abs(entry["observed"] - src_entry["observed"])
                worst_diff = max(worst_diff, diff)
                n_compared += 1
                if diff >= REPRODUCTION_TOLERANCE:
                    detail.append({"corpus": corpus_key, "session": rec["session"], "reference": ref_name, "abs_diff": diff})
    passed = bool(worst_diff < REPRODUCTION_TOLERANCE and n_compared > 0)
    status = "reproduced_exactly" if passed else "not_reproduced"
    if session_limited and n_compared == 0:
        status = "resource_limited_not_evaluated"
    return {
        "status": status, "tolerance": REPRODUCTION_TOLERANCE, "n_values_compared": n_compared,
        "max_abs_diff": worst_diff, "passed": passed, "session_limited_note": session_limited,
        "mismatches": detail[:20],
    }


def _zero_drop_accounting(n_macaque_on_disk: int, n_macaque_reachable: int, n_macaque_analysed: int,
                           watters_seen: int, watters_refused_n: int, watters_arrays_none: int,
                           watters_analysed: int) -> dict:
    return {
        CORPORA[0]: {
            "n_seen": n_macaque_on_disk, "n_excluded_below_reachability_floor": n_macaque_on_disk - n_macaque_reachable,
            "n_reaching_floor": n_macaque_reachable, "n_excluded_too_few_trials_with_defined_direction": n_macaque_reachable - n_macaque_analysed,
            "n_analysed": n_macaque_analysed,
            "reconciles": bool(n_macaque_on_disk == (n_macaque_on_disk - n_macaque_reachable)
                               + (n_macaque_reachable - n_macaque_analysed) + n_macaque_analysed),
        },
        CORPORA[1]: {
            "n_seen": watters_seen, "n_refused_by_shared_loader": watters_refused_n, "n_arrays_not_computable": watters_arrays_none,
            "n_analysed": watters_analysed,
            "reconciles": bool(watters_seen == watters_refused_n + watters_arrays_none + watters_analysed),
        },
    }


# =======================================================================================================
# Driver
# =======================================================================================================

def _session_limit() -> int | None:
    raw = os.environ.get(MAX_SESSIONS_ENV_VAR)
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def main() -> None:
    t0 = time.time()
    root = data_root()
    limit = _session_limit()
    session_limited = limit is not None

    output: dict = {
        "version": ANALYSIS_VERSION,
        "reads_as_input": "results/deviation_axis_structure.json (never modified, never rerun)",
        "scope": (
            "The same two macaque corpora, on the same sessions, the source artifact used, analysed within "
            "item-count level in the multi-object corpus and combined across levels by trial-count "
            "weighting; a pooled-across-item-count number is never reported as this corpus's effect size."
        ),
        "decision_rules_declared_before_fitting": {
            "part_one_pooled_alignment": POOLED_ALIGNMENT_DECISION_RULE_DECLARED_BEFORE_FITTING,
            "part_two_drift_mechanical_control": DRIFT_MECHANICAL_CONTROL_DECISION_RULE_DECLARED_BEFORE_FITTING,
            "part_two_detrended_control": DETRENDED_CONTROL_DECISION_RULE_DECLARED_BEFORE_FITTING,
            "part_three_rate_reconciliation": RATE_RECONCILIATION_DECISION_RULE_DECLARED_BEFORE_FITTING,
        },
        "slow_drift_direction_definition": SLOW_DRIFT_DIRECTION_DEFINITION,
        "sign_map": SIGN_TO_WORSE_BEHAVIOUR,
        "session_limited": session_limited, "session_limit_env_var": MAX_SESSIONS_ENV_VAR, "session_limit_value": limit,
        "status": "running",
    }
    _flush(output)

    _log("loading the multi-object macaque corpus")
    watters_seen, watters_loaded, watters_refused = 0, [], []
    for session in iter_watters(root, bin_ms=100.0):
        watters_seen += 1
        if session["status"] != "loaded":
            watters_refused.append({"session": session["session"], "status": session["status"]})
            continue
        watters_loaded.append(session)
        # smoke-test convenience only: loading every session of the multi-object corpus from the raw
        # per-session feature cache dominates wall clock, so under a session limit this loop stops early
        # once enough LOADED sessions exist to exercise every downstream block; the reproduction gate below
        # is then expected to fail on the resulting truncated corpus and is softened accordingly, never
        # treated as a non-reproduction verdict.
        if session_limited and len(watters_loaded) >= limit:
            break
    _log(f"multi-object macaque corpus: {watters_seen} seen, {len(watters_loaded)} loaded, "
         f"{len(watters_refused)} refused, elapsed={time.time() - t0:.0f}s")

    _log("running the reproduction gate the source module ships (estimator + residual-identity clauses)")
    gate_result, watters_arrays_by_session = full_reproduction_gate(root, watters_loaded)
    output["source_estimator_reproduction_gate"] = {"status": gate_result["status"], "session_limited_note": session_limited}
    _flush(output)
    if gate_result["status"] != "reproduced_exactly" and not session_limited:
        output["status"] = "void_source_estimator_gate_did_not_reproduce"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: the source module's own reproduction gate did not reproduce; no new number was read")
        return
    if gate_result["status"] != "reproduced_exactly" and session_limited:
        _log("NOTE: source estimator gate did not reproduce under a session limit (expected on a truncated "
             "corpus) -- treated as a resource-limit note, not a non-reproduction verdict; continuing.")

    macaque_paths = _reachable_sessions(root)
    macaque_bundles, _ = _macaque_bundles(root)
    n_macaque_on_disk = len(list(_panichello_directory(root).glob("*.mat"))) if _panichello_directory(root) else 0

    watters_bundles = _watters_bundles(watters_arrays_by_session)
    n_watters_arrays_none = len(watters_loaded) - len(watters_arrays_by_session)

    full_macaque_n, full_watters_n = len(macaque_bundles), len(watters_bundles)
    if limit is not None:
        macaque_bundles = macaque_bundles[:limit]
        watters_bundles = watters_bundles[:limit]
    _log(f"bundles: single-item {len(macaque_bundles)}/{full_macaque_n}, multi-object {len(watters_bundles)}/{full_watters_n} "
         f"(session_limited={session_limited}), elapsed={time.time() - t0:.0f}s")

    output["zero_drop_accounting"] = _zero_drop_accounting(
        n_macaque_on_disk, len(macaque_paths), full_macaque_n,
        watters_seen, len(watters_refused), n_watters_arrays_none, full_watters_n)
    output["reachability_analysed_this_run"] = {
        CORPORA[0]: len(macaque_bundles), CORPORA[1]: len(watters_bundles),
    }
    _flush(output)

    corpora = {CORPORA[0]: macaque_bundles, CORPORA[1]: watters_bundles}
    delivered_reference_effect = {CORPORA[0]: abs(float(MACAQUE_REFERENCE_R)), CORPORA[1]: abs(float(WATTERS_REFERENCE_R))}

    _log("part one: pooling axis alignments")
    part_one = run_part_one(corpora)
    output["part_one_pooled_alignment"] = part_one
    _flush(output)
    for corpus_key in CORPORA:
        for ref_name, entry in part_one[corpus_key]["by_reference"].items():
            _log(f"  part one [{corpus_key}][{ref_name}]: {entry['branch']}")

    output["reproduction_gate_against_axis_structure"] = _reproduction_gate_against_source(part_one, session_limited)
    _flush(output)
    gate_here = output["reproduction_gate_against_axis_structure"]
    _log(f"reproduction gate against results/deviation_axis_structure.json: {gate_here['status']} "
         f"(max abs diff {gate_here.get('max_abs_diff')}, n_compared={gate_here.get('n_values_compared')})")
    if gate_here["status"] == "not_reproduced":
        output["status"] = "void_reproduction_gate_against_axis_structure_did_not_reproduce"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: recomputed alignment values did not match the source artifact to 1e-6")
        return

    _log("part two: drift-mechanical control (synthetic pure-drift generator)")
    part_two = {"slow_drift_direction_definition": SLOW_DRIFT_DIRECTION_DEFINITION, "drift_mechanical_control": {},
                "detrended_control": {}, "detrended_deviation_behaviour": {}}
    for corpus_key, bundles in corpora.items():
        result = run_drift_mechanical_control(bundles, corpus_key)
        part_two["drift_mechanical_control"][corpus_key] = result
        _log(f"  [{corpus_key}]: {result['branch']}")
    output["part_two_drift_mechanical_control"] = part_two
    _flush(output)

    original_alignment_by_session = {
        corpus_key: {s["session"]: (s.get("alignments") if corpus_key == CORPORA[0] else s.get("alignments"))
                     for s in part_one[corpus_key]["per_session"] if s.get("status") == "computed"}
        for corpus_key in CORPORA
    }

    _log("part two: detrended controls (linear + local-window)")
    for variant in ("linear_detrend", "local_window"):
        part_two["detrended_control"][variant] = {}
        for corpus_key, bundles in corpora.items():
            result = run_detrend_variant(bundles, corpus_key, variant, original_alignment_by_session[corpus_key])
            part_two["detrended_control"][variant][corpus_key] = result
            for ref_name, entry in result["by_reference"].items():
                _log(f"  [{variant}][{corpus_key}][{ref_name}]: {entry['branch']}")
    output["part_two_drift_mechanical_control"] = part_two
    _flush(output)

    _log("part two: detrended deviation vs behaviour (linear detrend only), with mandatory bias-only control")
    for corpus_key, bundles in corpora.items():
        result = run_detrended_deviation_behaviour(bundles, corpus_key)
        part_two["detrended_deviation_behaviour"][corpus_key] = result
        _log(f"  [{corpus_key}]: {result['branch']}")
    output["part_two_drift_mechanical_control"] = part_two
    output["status"] = "part_one_and_two_complete"
    output["wall_clock_s_after_part_two"] = time.time() - t0
    _flush(output)
    _log(f"PART ONE AND TWO COMPLETE, elapsed={time.time() - t0:.0f}s -- proceeding to part three")

    _log("part three: rate reconciliation")
    part_three = {}
    for corpus_key, bundles in corpora.items():
        result = run_part_three(bundles, corpus_key, delivered_reference_effect[corpus_key])
        part_three[corpus_key] = result
        _log(f"  [{corpus_key}]: {result['branch']}")
    output["part_three_rate_reconciliation"] = part_three

    output["how_this_artifact_was_assembled"] = {"per_unit_checkpoint_directory": str(CHECKPOINT_DIR.relative_to(ROOT))}
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({
        "part_one_branches": {c: {r: e["branch"] for r, e in part_one[c]["by_reference"].items()} for c in CORPORA},
        "drift_mechanical_branches": {c: part_two["drift_mechanical_control"][c]["branch"] for c in CORPORA},
        "part_three_branches": {c: part_three[c]["branch"] for c in CORPORA},
        "wall_clock_s": output["wall_clock_s"],
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
