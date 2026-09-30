"""run_component_identity_subspace_atlas.py -- one exhaustive projection atlas for the accuracy-
predicting population-state deviation component, replacing a series of one-candidate-at-a-time
identity tests with a single pre-declared sweep.

THE COMPONENT. For a trial with population state vector v and m the leave-one-out mean state
direction across that session's other trials, the deviation is d = 1 - cos(v, m), with residual
r = v - (v . m) m. This is exactly run_rate_free_state_geometry_behavior_link.rate_free_state_
deviation, imported unchanged below and quoted verbatim in RATE_FREE_STATE_DEVIATION_SOURCE_QUOTE.
The per-session AXIS this module aligns against candidate subspaces is the leading eigenvector of
R^T R, R the stacked, unit-normalised residual rows -- run_deviation_axis_structure.leading_
eigenvector applied to run_deviation_axis_structure._unit_residual_matrix's output, both imported
unchanged (SOURCE_QUOTE constants below). No estimator here is a fork of an existing one; every
piece of already-delivered machinery this module depends on is imported, never re-typed.

CORPUS SCOPE. "Already computable" is read as: the project has already applied rate_free_state_
deviation to this corpus's trials in a delivered script. That holds for six corpora, not the two
already characterised as having a genuinely rate-free deviation (results/rate_free_state_geometry_
behavior_link.json restricts its own claim to macaque lPFC; the multi-object corpus is characterised
in results/deviation_axis_structure.json). The deviation is COMPUTABLE more broadly:
  - panichello_2024_macaque_lPFC        -- rate_free_state_geometry_behavior_link.py (native corpus)
  - watters_2026_macaque_multi_object   -- run_deviation_axis_structure.py (native corpus)
  - inagaki_alm5_mouse_ALM              -- run_dissociation_cross_preparation_test.py applies rate_
                                            free_state_deviation to this corpus's control-trial delay
                                            counts; that module's own orthogonality gate found the
                                            mouse ALM deviation MORE rate-contaminated than the macaque
                                            reference in that specific comparison. That finding is
                                            about whether this corpus's deviation is validated as the
                                            established rate-free component; it does not mean the
                                            deviation is uncomputable here, and the mouse anterior
                                            lateral motor cortex corpus is EXPLICITLY commissioned for
                                            the upcoming-response subspace test below, so it is
                                            included, with this caveat carried in its every record.
  - dandi_000469, dandi_001187, dandi_000574 (human) -- run_human_maintenance_behaviour_link.py
                                            applies the identical estimator to these three corpora's
                                            delay-epoch spike counts.
Human region-resolved sub-structures (e.g. "pre_sma", "hippocampus") are not separately analysed
here; only the "pooled" structure per session enters this atlas, matching the convention scripts/
run_state_space_estimation_admissibility.py already uses for these same three corpora.

CANDIDATE SUBSPACES, per corpus, are built ONLY where that corpus's own data supports them; absence
is recorded with a reason rather than fabricated (see CANDIDATE_SUPPORT_MATRIX below, filled in as
each corpus is analysed). Two multi-item task corpora (macaque multi-object working-memory corpus (doi:10.64898/2026.01.27.702062), and the multi-object cardinality
inside it) use the project's own trial-count-weighted combination across item-count levels
(_trial_count_weighted, run_component_effect_size_and_anatomy.py, imported unchanged) exactly as
results/deviation_axis_structure.json does for this corpus; no other corpus in this atlas has more
than one memorandum cardinality.

NULL CONSTRUCTION -- THE ONE GENUINELY NEW PIECE OF MATH IN THIS MODULE. Every existing alignment-
null in this project (run_deviation_axis_structure._alignment_null_subspace, run_deviation_axis_
identity_controls._subspace_reference_alignment_with_draws) draws its random comparison subspace
from the full AMBIENT unit space. That is the wrong null for a high-dimensional population: a random
ambient subspace is (with overwhelming probability) already close to orthogonal to a session's
occupied subspace, so a candidate direction's occupied-space alignment to it looks "significant"
purely because random ambient subspaces almost never fall inside the occupied space at all. This
module instead draws every null subspace UNIFORMLY AT RANDOM INSIDE THE SESSION'S OWN OCCUPIED
SUBSPACE (estimated by the project's own cross-validated PCA rank selector, run_deviation_axis_
structure._cv_pca_rank, imported unchanged), of the SAME dimension as the candidate subspace being
tested, at least N_RANDOM_SUBSPACE_DRAWS = 1000 draws per cell. See _alignment_against_occupied_null
below -- the only null-drawing function this module defines; every other piece of null-drawing
machinery in this module reuses run_deviation_axis_structure._empirical_two_sided unchanged for the
percentile test and p-value.

DECISION RULE -- declared in full, before any number is computed, under decision_rule_declared_
before_fitting in the output artifact, and never edited after seeing a result. See that string for
the exact branches. The z multiplier this project uses for 80%-power minimum-detectable-effect
bounds is statistics.Z_80_POWER; the detection floor for the
"no candidate aligned" branch is a whole-SESSION cluster bootstrap standard error (resampling
sessions with replacement, never a trial-count formula or an intraclass-correlation design effect),
multiplied by that same constant -- see _session_cluster_bootstrap_mdd.

REACTION TIME (step 5) is branched separately from the subspace-identity branch, per trial-time
outcomes where they exist in this project's registry: watters_2026 (reaction_time_ms, native column)
and inagaki_alm5 (First_lick minus Cue_start, in the corpus's own raw behaviour struct, not exposed
by src/corpus_sessions.py's iter_alm). The ALM RT number is NOT reported here: this corpus's raw
response code (Trial_types_of_response_vector) needs a verified mapping to a binary correct/error
label before it can serve as the accuracy control this analysis requires alongside spike count, and
that mapping was not verified against the original publication within this task's scope. Reporting
an RT number without a checked accuracy control would be silently substituting a weaker analysis for
the one asked for, so ALM's RT cell is 'not_computed' with this reason stated, not a fabricated
number. Neither macaque prefrontal spatial working-memory corpus (Dryad doi:10.5061/dryad.kkwh70sct) (no timestamp field: its raw .mat carries only cueAng, cueAngIdx, isCorr,
spks, tc) nor any of the three human corpora (no response-time field in the trial tables
src/corpus_sessions.py already reads) can ask this question at all.

BIAS-ONLY CONTROL (step 6) accompanies every RT association reported, following the algorithm
_bias_only_between_session (run_swap_versus_imprecision_by_item_count.py) already documents and this
module quotes verbatim in BIAS_ONLY_ALGORITHM_SOURCE_QUOTE: collapse each session's trials to that
session's own mean deviation and own mean outcome, correlate BETWEEN sessions (statistics.pearson_
permutation_test, unchanged), and void the real association only if this control is ALSO significant
in the SAME direction -- sign and significance only, the two statistics are on different scales and
are never compared by magnitude. This module does not import that function directly: its signature
is bound to a swap/item-count-level data shape this atlas does not share, so the between-session
correlation is recomputed here from session-mean arrays using the identical algorithm and the
identical primitive statistic, exactly as this project's own component_binding_bias_only_control.py
did before it for the same reason ("this module's own generalisation ... changes exactly two things
relative to the quoted function").

ENVIRONMENT NOTE: this run requires WM_DYNAMICS_DATA_ROOT to point at the currently mounted copy of
the external data drive; the checked-in default in this session's shell profile pointed at a drive
label that was not mounted, so every invocation of this module (including the tests) must export the
correct value first (see the module's own __main__ guard, which does not silently fall back).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import norm, pearsonr, theilslopes

Z_95 = float(norm.ppf(0.975))

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from corpus_sessions import (  # noqa: E402
    alm_data_directory, data_root, iter_dandi_000469, iter_dandi_001187, iter_dandi_000574,
    load_watters_session, watters_behaviour, watters_session_dates, watters_directories,
)
from provenance import _json_safe, git_commit  # noqa: E402
from spike_pipeline import build_psth  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import (  # noqa: E402
    Z_80_POWER, fdr_bh, partial_correlation_permutation_test, pearson_permutation_test, permutation_pvalue,
    stable_seed,
)
from preprocessing import load_boran_nwb  # noqa: E402
from corpus_sessions import _reachable_sessions
from statistics import _trial_count_weighted
from subspace_identity import _class_mean_subspace_basis, _regression_subspace_basis
from info_decoding import _residual_rows
from info_decoding import _cv_pca_rank, _empirical_two_sided, _unit_residual_matrix, _weighted_combine_draws
from subspace_identity import leading_eigenvector
from info_decoding import _leave_one_out_unit_directions, residual_decomposition_and_identity_check
from statistics import MIN_TRIALS_WITH_DEFINED_DIRECTION
from corpus_sessions import _session_arrays as _panichello_reproduction_gate_session_arrays
from stimulation_response_estimator import rate_free_state_deviation
from spike_pipeline import _counts_from_spikes
from corpus_sessions import _panichello_directory
from run_state_content_link import MIN_TRIALS_PER_CLASS, usable_label
from spike_pipeline import BIN_MS
from info_decoding import MIN_CLASSES
from spike_pipeline import delay_counts
from info_decoding import FIELD_MAINTENANCE_WINDOW_S
from corpus_sessions import _boran_field_potential_session
from info_decoding import FIELD_BAND_HI_HZ, FIELD_BAND_LO_HZ
from info_decoding import CANDIDATE_KEYS, CANDIDATE_SUPPORT_MATRIX, MAX_SESSIONS_ENV_VAR, N_BOOT_SESSION_CLUSTER, _previous_label  # noqa: E402
from info_decoding import _session_core  # noqa: E402
from corpus_sessions import _session_limit, _panichello_session_inputs  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "component_identity_subspace_atlas.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_component_identity_subspace_atlas"
ANALYSIS_VERSION = "2026-09-05"

N_RANDOM_SUBSPACE_DRAWS = 1000
N_PERM = 2000
FDR_ALPHA = 0.05

CORPORA = (
    "panichello_2024_macaque_lPFC", "watters_2026_macaque_multi_object", "inagaki_alm5_mouse_ALM",
    "dandi_000469_human", "dandi_001187_human", "dandi_000574_human",
)

RATE_FREE_STATE_DEVIATION_SOURCE_QUOTE = (
    "run_rate_free_state_geometry_behavior_link.rate_free_state_deviation, quoted verbatim from its "
    "own source:\n\n"
    'def rate_free_state_deviation(activity_by_unit: np.ndarray) -> np.ndarray:\n'
    '    """Per trial, deviation_i = 1 - cosine(unit_vector_i, renormalised\n'
    '    leave-one-out mean of every OTHER trial\'s own unit-normalised\n'
    '    direction), from a (n_trials, n_units) per-unit activity array..."""\n'
    '    activity = np.asarray(activity_by_unit, dtype=float)\n'
    '    n_trials = activity.shape[0]\n'
    '    norms = np.linalg.norm(activity, axis=1, keepdims=True)\n'
    '    with np.errstate(invalid="ignore", divide="ignore"):\n'
    '        unit_vectors = np.where(norms > 0, activity / np.where(norms > 0, norms, 1.0), np.nan)\n'
    '    valid = ~np.isnan(unit_vectors).any(axis=1)\n'
    '    total = np.nansum(unit_vectors, axis=0)\n'
    '    n_valid = int(valid.sum())\n'
    '    deviation = np.full(n_trials, np.nan)\n'
    '    for i in range(n_trials):\n'
    '        if not valid[i]:\n'
    '            continue\n'
    '        n_other = n_valid - 1\n'
    '        if n_other < 1:\n'
    '            continue\n'
    '        loo_mean = (total - unit_vectors[i]) / n_other\n'
    '        loo_norm = np.linalg.norm(loo_mean)\n'
    '        if loo_norm == 0.0:\n'
    '            continue\n'
    '        cosine = float(np.dot(unit_vectors[i], loo_mean / loo_norm))\n'
    '        deviation[i] = 1.0 - cosine\n'
    '    return deviation\n\n'
    "This module imports it unchanged (from run_rate_free_state_geometry_behavior_link import "
    "rate_free_state_deviation) and never re-implements it."
)

DEVIATION_AXIS_SOURCE_QUOTE = (
    "The per-session deviation AXIS (a single unit-space vector, needed to compute a principal angle "
    "against a candidate subspace) is run_deviation_axis_structure.leading_eigenvector applied to "
    "run_deviation_axis_structure._unit_residual_matrix's output, both imported unchanged:\n\n"
    "def leading_eigenvector(R: np.ndarray) -> np.ndarray:\n"
    "    _w, v = np.linalg.eigh(R.T @ R)\n"
    "    return v[:, -1]\n\n"
    "R is the stacked, unit-normalised residual rows r_i / ||r_i|| for every trial whose residual norm "
    "clears the numerical floor (run_deviation_axis_structure._residual_rows /"
    " _unit_residual_matrix), r_i itself built by run_deviation_subspace_decomposition."
    "residual_decomposition_and_identity_check, which PROVES on every session that its own "
    "leave-one-out arithmetic reproduces rate_free_state_deviation's delivered scalar to floating-"
    "point tolerance before anything downstream is trusted."
)

BIAS_ONLY_ALGORITHM_SOURCE_QUOTE = (
    "Verbatim from run_swap_versus_imprecision_by_item_count.py's _bias_only_between_session "
    "docstring (the algorithm this module's own _bias_only_between_session_atlas below implements "
    "against this atlas's own reaction-time arrays, since the quoted function's signature is bound to "
    "a swap/item-count-level data shape this atlas does not share):\n\n"
    '"""The session-level bias-only statistic: collapses every session\'s real per-trial deviation at '
    "this level to that session's own mean (the bias-only substitution the control specifies), one "
    "number per session, paired against that same session's own mean outcome at this level -- session "
    "is the unit of analysis, N sessions, matching the real statistic's own unit of analysis.\n\n"
    "This is NOT the same estimator as the real one. The real statistic is a per-session WITHIN-session "
    "correlation (many trials, one r per session) pooled across sessions by a sign-flip test. Once "
    "every trial in a session is replaced by that session's own mean, the within-session predictor has "
    "zero variance by construction, so a per-session correlation is undefined -- there is no "
    'well-defined way to recompute "the same" per-session statistic under this substitution. The '
    "closest well-defined, session-is-the-unit-of-analysis alternative is a single BETWEEN-session "
    "correlation over the N (session mean deviation, session mean outcome) points, computed here. Its "
    "r, p-value and n are on a different scale from the real within-session, sign-flip-pooled effect "
    "size (different estimator, different degrees of freedom) and are never compared to the real "
    "effect size by magnitude; only this test's own sign and significance enter the voiding rule, "
    'which does not depend on magnitude."""'
)

DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "0. Reproduction gate: reproduce results/rate_free_state_geometry_behavior_link.json's own pooled "
    "raw_outcome_vs_deviation.mean_value on the identical 11 reachable macaque spatial working-memory corpus sessions, to tolerance "
    "1e-06, using rate_free_state_deviation unchanged. Failure -> branch "
    "'void_reproduction_gate_failed', no further number read.\n"
    "1. A candidate identity is ALIGNED in a given corpus if its pooled (session-level sign-flip test "
    "on each session's own z = (observed_alignment - null_mean) / null_sd against that session's "
    "matched occupied-space random-subspace null) two-sided p-value clears Benjamini-Hochberg FDR "
    "(alpha=0.05) across every candidate that corpus actually supports. The sign-flip test's own "
    "significant_positive/significant_negative fields ARE the 'pooled session-level test agrees in "
    "sign' requirement -- a pooled test cannot reject a two-sided null without the per-session z's "
    "agreeing enough in sign to move the mean away from zero, so no separate sign-agreement check is "
    "computed on top of it.\n"
    "2. Read across the whole atlas (not per corpus): for each candidate TYPE, aligned_anywhere = true "
    "if it is ALIGNED (rule 1) in at least one corpus that supports it. If exactly one candidate type "
    "has aligned_anywhere = true, branch 'component_identity_resolved_to_a_single_labelled_subspace', "
    "naming it and every corpus it aligned in.\n"
    "3. If more than one candidate type has aligned_anywhere = true, branch "
    "'component_identity_is_multiply_aligned', and within each corpus where two or more of ITS OWN "
    "aligned candidates co-occur, report the alignment of the deviation axis to each candidate after "
    "mutually orthogonalising those candidates' bases against each other (Gram-Schmidt, in the order "
    "the CANDIDATE_KEYS tuple lists them), so shared variance is not counted twice.\n"
    "4. If no candidate type has aligned_anywhere = true, branch "
    "'component_is_not_aligned_with_any_labelled_subspace_tested' -- a positive, publishable result -- "
    "PROVIDED every computed cell also reports its own minimum alignment detectable at 80% power, from "
    "a whole-session cluster bootstrap standard error (resample sessions with replacement, "
    f"{N_BOOT_SESSION_CLUSTER} draws, recompute the pooled mean z each draw; mdd = {Z_80_POWER} * "
    "that bootstrap SE), never from a trial-count formula or an intraclass-correlation design effect. "
    "Any cell missing this floor is 'inconclusive_below_detection_floor' instead of the null branch.\n"
    "5. Reaction time (step 5) is branched separately as "
    "'reaction_time_association' / 'reaction_time_not_askable_in_this_corpus', per corpus, never "
    "merged into the subspace-identity branch above, per every reported association's own bias-only "
    "control (step 6, BIAS_ONLY_ALGORITHM_SOURCE_QUOTE)."
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ============================================================================================
# Checkpointing -- identical atomic-replace pattern to run_deviation_axis_structure.py's own.
# ============================================================================================

def _checkpoint_path(unit: str) -> Path:
    safe = unit.replace("/", "_").replace(" ", "_")
    return CHECKPOINT_DIR / f"{safe}.json"


def _load_checkpoint(unit: str) -> dict | None:
    path = _checkpoint_path(unit)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _save_checkpoint(unit: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(unit)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_json_safe(record), indent=2, allow_nan=False, default=float))
    os.replace(tmp, path)


def _run_checkpointed(unit: str, fit_fn):
    cached = _load_checkpoint(unit)
    if cached is not None:
        return cached
    record = _json_safe(fit_fn())
    _save_checkpoint(unit, record)
    return record


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)




# ============================================================================================
# Core session geometry: axis recovery (step 1) + occupied-space basis + the ONE new null.
# ============================================================================================



def _occupied_basis(U: np.ndarray, seed_tag: str) -> tuple[np.ndarray | None, dict]:
    n, p = U.shape
    max_k = min(n - 1, p)
    cvr = _cv_pca_rank(U, max_k, seed_tag)
    if cvr.get("status") != "computed":
        return None, cvr
    mean_u = U.mean(axis=0)
    _u_svd, _s_svd, vt = np.linalg.svd(U - mean_u, full_matrices=False)
    k = min(cvr["best_k"], vt.shape[0])
    return vt[:k].T, cvr


def _alignment_against_occupied_null(axis: np.ndarray, candidate_basis: np.ndarray, occupied_basis: np.ndarray,
                                      n_draws: int, seed_tag: str) -> dict:
    """The one new null in this module: a matched-dimension random subspace drawn UNIFORMLY INSIDE
    the session's own occupied subspace (occupied_basis, already orthonormal, k_occ columns), not the
    ambient unit space -- see module docstring, NULL CONSTRUCTION. dim = candidate_basis.shape[1]
    columns of occupied_basis are combined by a random orthonormal (k_occ, dim) matrix, which maps to
    a uniformly random dim-dimensional subspace of the occupied span (QR of an i.i.d. Gaussian is
    Haar-distributed on the Stiefel manifold, the same fact run_deviation_subspace_decomposition.
    _random_orthonormal_basis already relies on for its own, ambient-space, null)."""
    dim = candidate_basis.shape[1]
    k_occ = occupied_basis.shape[1]
    if k_occ < dim or dim < 1:
        return {"status": "not_computable", "reason": "occupied-space rank below the candidate subspace dimension"}
    observed = float(min(np.linalg.norm(candidate_basis.T @ axis), 1.0))
    rng = np.random.default_rng(stable_seed(seed_tag))
    draws = np.empty(n_draws)
    for d in range(n_draws):
        g_local = rng.standard_normal((k_occ, dim))
        q, _ = np.linalg.qr(g_local)
        basis_ambient = occupied_basis @ q[:, :dim]
        draws[d] = min(float(np.linalg.norm(basis_ambient.T @ axis)), 1.0)
    result = _empirical_two_sided(observed, draws)
    if result.get("status") != "computed":
        return result
    result["principal_angle_deg"] = float(np.degrees(np.arccos(min(max(observed, -1.0), 1.0))))
    result["candidate_dim"] = dim
    result["occupied_k"] = k_occ
    result["_null_draws"] = draws  # transient: consumed only by _combine_levels_by_trial_count, stripped before saving
    return result


def _combine_levels_by_trial_count(level_cells: list[tuple[int, dict]]) -> dict:
    """Combines one candidate's per-item-count-level cells into one session-level cell, by
    trial-count weighting, exactly the convention run_deviation_axis_structure.py's own
    _weighted_combine_draws / _trial_count_weighted apply to this same corpus's other per-level
    statistics -- both imported unchanged and used here, not re-derived. Used only for the
    multi-object corpus (macaque multi-object corpus), the only corpus in this atlas with more than one memorandum
    cardinality."""
    usable = [(n, c) for n, c in level_cells if c is not None and c.get("status") == "computed"]
    if not usable:
        return {"status": "not_computable", "reason": "no item-count level reached a computed cell"}
    observed = _trial_count_weighted([(n, c["observed"]) for n, c in usable])
    combined_draws = _weighted_combine_draws([(n, c["_null_draws"]) for n, c in usable])
    if combined_draws is None:
        return {"status": "not_computable", "reason": "no finite null draws to combine"}
    result = _empirical_two_sided(observed, combined_draws)
    if result.get("status") == "computed":
        result["principal_angle_deg"] = float(np.degrees(np.arccos(min(max(observed, -1.0), 1.0))))
        result["candidate_dim"] = usable[0][1]["candidate_dim"]
        result["occupied_k"] = _trial_count_weighted([(n, c["occupied_k"]) for n, c in usable])
        result["_null_draws"] = combined_draws
    result["n_levels_combined"] = len(usable)
    result["levels_used_n_trials"] = [n for n, _ in usable]
    return result


def _strip_draws(cell: dict | None) -> dict | None:
    if cell is None or "_null_draws" not in cell:
        return cell
    return {k: v for k, v in cell.items() if k != "_null_draws"}


def _cell_z(cell: dict) -> float | None:
    if cell.get("status") != "computed" or cell.get("null_sd", 0.0) in (0.0, None):
        return None
    return (cell["observed"] - cell["null_mean"]) / cell["null_sd"]


def _pool_candidate_cells(per_session_cells: list[dict | None]) -> dict:
    zs = [z for z in (_cell_z(c) for c in per_session_cells if c is not None) if z is not None]
    if len(zs) < 4:
        return {"status": "not_computable", "n_sessions": len(zs), "reason": "fewer than 4 sessions carry a computed cell"}
    pooled = slope_across_sessions_test(zs, alternative="two-sided")
    pooled["n_sessions"] = len(zs)
    observed_vals = [c["observed"] for c in per_session_cells if c is not None and c.get("status") == "computed"]
    null_mean_vals = [c["null_mean"] for c in per_session_cells if c is not None and c.get("status") == "computed"]
    pooled["mean_observed_alignment"] = float(np.mean(observed_vals)) if observed_vals else None
    pooled["mean_null_mean"] = float(np.mean(null_mean_vals)) if null_mean_vals else None
    return pooled


def _session_cluster_bootstrap_mdd(per_session_cells: list[dict | None], seed_tag: str) -> dict:
    """Detection floor for rule 4: whole-SESSION cluster bootstrap standard error of the pooled mean
    z (resample sessions with replacement, not trials; never an ICC design effect or a trial-count
    formula), times the task-fixed Z_80_POWER multiplier."""
    zs = np.array([z for z in (_cell_z(c) for c in per_session_cells if c is not None) if z is not None])
    if len(zs) < 4:
        return {"status": "not_computable", "n_sessions": int(len(zs))}
    rng = np.random.default_rng(stable_seed(seed_tag))
    n = len(zs)
    boot_means = np.array([zs[rng.integers(0, n, size=n)].mean() for _ in range(N_BOOT_SESSION_CLUSTER)])
    se = float(np.std(boot_means, ddof=1))
    return {"status": "computed", "n_sessions": int(n), "cluster_bootstrap_se": se,
            "z_multiplier": Z_80_POWER, "minimum_detectable_z_at_80pct_power": Z_80_POWER * se}


# ============================================================================================
# Candidate-subspace builders, one per label TYPE, all reusing the two session-level basis
# fitters already delivered (run_deviation_axis_structure._class_mean_subspace_basis /
# _regression_subspace_basis) -- categorical labels go through the first, every continuous or
# scalar covariate (including every field-potential and gain/time covariate) through the second
# with dim=1.
# ============================================================================================

def _categorical_candidate(U: np.ndarray, labels: np.ndarray, min_classes: int = MIN_CLASSES,
                            min_per_class: int = MIN_TRIALS_PER_CLASS) -> tuple[np.ndarray | None, dict]:
    ok, reason, mask = usable_label(labels, min_classes=min_classes, min_per_class=min_per_class)
    if not ok:
        return None, {"status": "absent", "reason": reason}
    classes = np.unique(labels[mask])
    dim = max(1, len(classes) - 1)
    basis = _class_mean_subspace_basis(U[mask], labels[mask], dim)
    if basis is None:
        return None, {"status": "absent", "reason": "class-mean subspace fit failed (degenerate)"}
    return basis, {"status": "present", "n_classes": int(len(classes)), "n_trials_used": int(mask.sum())}


def _continuous_candidate(U: np.ndarray, covariate: np.ndarray, dim: int = 1) -> tuple[np.ndarray | None, dict]:
    finite = np.all(np.isfinite(np.atleast_2d(covariate.T).T), axis=1) if covariate.ndim > 1 else np.isfinite(covariate)
    if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None, {"status": "absent", "reason": "fewer than the trial floor have a finite covariate"}
    target = covariate[finite] if covariate.ndim > 1 else covariate[finite][:, None]
    if np.std(target, axis=0).min() <= 0.0:
        return None, {"status": "absent", "reason": "covariate has zero variance among finite trials"}
    basis = _regression_subspace_basis(U[finite], target, dim)
    if basis is None:
        return None, {"status": "absent", "reason": "regression subspace fit failed (degenerate)"}
    return basis, {"status": "present", "n_trials_used": int(finite.sum())}


# ============================================================================================
# Field-potential candidates (000574 only) -- band power reuses field_potential_sessions
# unchanged; the aperiodic-slope fit is new (no existing implementation in this project; see
# module docstring). Both align by TRIAL POSITION with the spike-based session because both use
# the identical admission mask ((~artifact) & correct) applied to the same NWB trial table, the
# same way run_state_space_estimation_admissibility.py's own _boran_field_potential_session
# already documents doing independently of src/corpus_sessions.py's spike iterator.
# ============================================================================================

FIELD_APERIODIC_FIT_RANGES_HZ = ((20.0, 45.0), (55.0, 95.0))
FIELD_APERIODIC_FIT_RANGES_RATIONALE = (
    "Two flanking windows, chosen to exclude the canonical oscillatory peaks this project's own "
    "field-potential band analysis already targets (FIELD_BAND_LO_HZ=1, FIELD_BAND_HI_HZ=40 Hz -- "
    "theta/alpha/beta) and the 50 Hz mains line (the DANDI 000574 corpus, doi 10.1038/s41597-020-0364-3, is a European recording; load_boran_nwb's "
    "own default mains_hz=50.0) plus its first harmonic region, while keeping two contiguous decades of "
    "bandwidth-per-window for a robust slope: 20-45 Hz (upper beta/low gamma, below line noise) and "
    "55-95 Hz (gamma, above line noise, below where amplifier/anti-alias roll-off typically starts "
    "distorting the spectrum at these sampling rates). This is a disclosed reading of a genuinely "
    "underspecified instruction, exactly as run_rate_free_state_geometry_behavior_link.py's own "
    "construction_operationalisation makes for its analogous choice."
)


def _welch_slope(trace: np.ndarray, srate: float) -> float | None:
    nperseg = int(min(len(trace), max(64, round(srate * 0.5))))
    if nperseg < 32:
        return None
    freqs, psd = welch(trace, fs=srate, nperseg=nperseg)
    mask = np.zeros_like(freqs, dtype=bool)
    for lo, hi in FIELD_APERIODIC_FIT_RANGES_HZ:
        mask |= (freqs >= lo) & (freqs < hi)
    mask &= psd > 0
    if int(mask.sum()) < 8:
        return None
    slope, *_ = theilslopes(np.log10(psd[mask]), np.log10(freqs[mask]))
    return float(slope)


def _boran_aperiodic_slope_session(nwb_path: Path, signal: str) -> dict | None:
    """New: per-trial, channel-averaged robust log-log aperiodic slope over the maintenance window,
    admission mask identical to run_state_space_estimation_admissibility._boran_field_potential_session
    (duplicated rather than imported since that function returns band POWER, not raw epochs, and
    reads the same file independently of src/corpus_sessions.py's own spike iterator, matching this
    project's own precedent for reading a second signal group from the same NWB file)."""
    import h5py

    with h5py.File(str(nwb_path), "r") as handle:
        if "intervals/trials" not in handle:
            return None
        trials = handle["intervals/trials"]
        artifact = trials["artifact"][:].astype(bool)
        correct = trials["correct"][:].astype(bool)
    keep = (~artifact) & correct
    if int(keep.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    loaded = load_boran_nwb(str(nwb_path), signal=signal, epoch_win=(-3.2, 0.3))
    epochs = loaded["epochs"][keep]  # (N, C, T)
    times = loaded["times"]
    srate = loaded["srate"]
    win_mask = (times >= FIELD_MAINTENANCE_WINDOW_S[0]) & (times < FIELD_MAINTENANCE_WINDOW_S[1])
    n_trials, n_ch, _ = epochs.shape
    slope = np.full(n_trials, np.nan)
    for i in range(n_trials):
        per_channel = [_welch_slope(epochs[i, c, win_mask], srate) for c in range(n_ch)]
        finite = [v for v in per_channel if v is not None]
        if finite:
            slope[i] = float(np.mean(finite))
    return {"slope_mean_across_channels": slope, "n_trials": n_trials, "n_channels": n_ch}


def _boran_field_scalars(nwb_path: Path, signal: str) -> dict | None:
    """(band_power, aperiodic_slope) per-trial scalars for one 000574 session/signal, channel- and
    (for band power) bin-averaged, aligned by trial position to the spike-based session (see module
    docstring): _boran_field_potential_session is imported unchanged for the band-power half; the
    aperiodic-slope half is this module's own new function, sharing the identical admission mask."""
    band_record = _boran_field_potential_session(nwb_path, signal)
    slope_record = _boran_aperiodic_slope_session(nwb_path, signal)
    if band_record is None or slope_record is None:
        return None
    band_power = band_record["X"].mean(axis=(1, 2))  # (trials,), channel- and bin-averaged
    slope = slope_record["slope_mean_across_channels"]  # (trials,), channel-averaged
    if band_power.shape[0] != slope.shape[0]:
        return None
    return {"band_power": band_power, "aperiodic_slope": slope, "n_trials": int(band_power.shape[0])}


# ============================================================================================
# ALM: raw-mat re-derivation of the same control-trial index set src/corpus_sessions.py's
# load_alm_raw_session already builds, needed here only to attach the response condition, the
# response code, and (for completeness, though not tested -- see module docstring) the raw
# fields a verified reaction-time control would need. This duplicates a few lines of that
# function's filtering rather than modifying it, per this task's explicit constraint against
# editing existing corpus iterators in place.
# ============================================================================================

def _alm_session_arrays(path: Path, window_s: float, bin_ms: float) -> dict | None:
    from corpus_sessions import _alm_build_counts, _alm_trial_condition, ALM_MIN_TRIALS_PER_ARM, ALM_MIN_UNITS, ALM_MIN_UNIT_RATE_HZ

    units = np.atleast_1d(loadmat(path, struct_as_record=False, squeeze_me=True)["unit"])
    behavior = units[0].Behavior
    trial_info = units[0].Trial_info
    trial_type = np.asarray(behavior.Trial_types_of_response_vector, dtype=int).reshape(-1)
    stimulation = np.asarray(behavior.stim_trial_vector, dtype=int).reshape(-1)
    delay_duration = np.asarray(behavior.delay_dur, dtype=float).reshape(-1)
    delay_start = np.asarray(behavior.Delay_start, dtype=float).reshape(-1)
    condition = _alm_trial_condition(np.asarray(trial_info.Trial_types).reshape(-1))
    start, stop = np.asarray(trial_info.Trial_range_to_analyze, dtype=int).reshape(-1) - 1
    eligible = np.arange(start, stop + 1)
    eligible = eligible[(trial_type[eligible] < 5) & (delay_duration[eligible] >= window_s)]
    control_trials = eligible[stimulation[eligible] == 0]
    if len(control_trials) < ALM_MIN_TRIALS_PER_ARM or len(units) < 1:
        return None
    counts = _alm_build_counts(units, delay_start, control_trials, bin_ms, window_s)
    rates = counts.sum(axis=(0, 2)) / (len(control_trials) * window_s)
    unit_mask = rates >= ALM_MIN_UNIT_RATE_HZ
    counts = counts[:, unit_mask]
    if int(np.sum(unit_mask)) < ALM_MIN_UNITS:
        return None
    return {
        "counts": counts, "condition": condition[control_trials],
        "response_code": trial_type[control_trials], "n_trials": len(control_trials),
    }


# ============================================================================================
# Generic helpers shared by every corpus's candidate assembly.
# ============================================================================================


def _time_in_trial_basis(counts_tensor_kept: np.ndarray) -> np.ndarray | None:
    """Direction along which trial-averaged per-unit activity ramps across the delay's own bins --
    the same regression-direction construction run_deviation_axis_structure._regression_direction
    applies to a per-trial covariate, applied here to elapsed BIN time instead, because 'time in
    trial' is a within-trial temporal structure the fixed delay-epoch-summed activity vector this
    module's other candidates all use (activity_by_unit) has already collapsed away."""
    if counts_tensor_kept.ndim != 3 or counts_tensor_kept.shape[0] < MIN_TRIALS_WITH_DEFINED_DIRECTION or counts_tensor_kept.shape[2] < 3:
        return None
    bin_mean = counts_tensor_kept.mean(axis=0)  # (units, bins)
    bin_time = np.arange(bin_mean.shape[1], dtype=float)
    bin_time = bin_time - bin_time.mean()
    design = np.column_stack([np.ones_like(bin_time), bin_time])
    coef, *_ = np.linalg.lstsq(design, bin_mean.T, rcond=None)
    slope = coef[1]
    norm = np.linalg.norm(slope)
    return (slope / norm)[:, None] if norm > 0 else None


def _build_candidate_cells(core: dict, seed_prefix: str, categorical_spec: dict, continuous_spec: dict,
                            time_in_trial_counts: np.ndarray | None) -> tuple[dict, dict, dict]:
    """categorical_spec: key -> full-length label array (pre-kept-trial-subsetting).
    continuous_spec: key -> full-length covariate array (1-D) or None to skip.
    time_in_trial_counts: (n_trials_total, n_units, n_bins) raw counts, or None if unavailable.
    Returns (cells, support, occupied_cv_rank_result)."""
    U, idx, axis = core["U"], core["idx"], core["axis"]
    occupied_basis, cvr = _occupied_basis(U, f"{seed_prefix}|occupied")
    cells: dict[str, dict] = {}
    support: dict[str, dict] = {}
    if occupied_basis is None:
        for key in (*categorical_spec, *continuous_spec, "time_in_trial_within_delay"):
            cells[key] = {"status": "not_computable", "reason": "occupied-space rank not estimable this session"}
            support[key] = {"status": "not_computable"}
        return cells, support, cvr

    for key, full_labels in categorical_spec.items():
        labels = full_labels[idx].astype(float)
        basis, supp = _categorical_candidate(U, labels)
        support[key] = supp
        cells[key] = ({"status": "absent", "reason": supp.get("reason")} if basis is None else
                       _alignment_against_occupied_null(axis, basis, occupied_basis, N_RANDOM_SUBSPACE_DRAWS, f"{seed_prefix}|{key}"))

    for key, full_covariate in continuous_spec.items():
        covariate = full_covariate[idx]
        basis, supp = _continuous_candidate(U, covariate, dim=1 if covariate.ndim == 1 else covariate.shape[1])
        support[key] = supp
        cells[key] = ({"status": "absent", "reason": supp.get("reason")} if basis is None else
                       _alignment_against_occupied_null(axis, basis, occupied_basis, N_RANDOM_SUBSPACE_DRAWS, f"{seed_prefix}|{key}"))

    key = "time_in_trial_within_delay"
    if time_in_trial_counts is None:
        cells[key] = {"status": "absent", "reason": "no bin-resolved delay-epoch tensor available for this corpus"}
        support[key] = {"status": "absent", "reason": "no bin-resolved tensor"}
    else:
        basis = _time_in_trial_basis(time_in_trial_counts[idx])
        support[key] = {"status": "present"} if basis is not None else {"status": "absent", "reason": "regression fit failed (degenerate)"}
        cells[key] = ({"status": "absent", "reason": "regression fit failed (degenerate)"} if basis is None else
                       _alignment_against_occupied_null(axis, basis, occupied_basis, N_RANDOM_SUBSPACE_DRAWS, f"{seed_prefix}|{key}"))
    return cells, support, cvr


def _assemble_record(session_id: str, core: dict, cvr: dict, cells: dict, support: dict, extra: dict | None = None) -> dict:
    record = {
        "session": session_id, "status": "computed",
        "n_kept_trials": core["n_kept"], "n_trials_total": core["n_trials_total"], "n_units": core["n_units"],
        "identity_passed": core["identity_passed"], "identity_max_abs_diff": core["identity_max_abs_diff"],
        "occupied_space": cvr, "candidate_support": support,
        "candidates": {k: _strip_draws(v) for k, v in cells.items()},
    }
    if extra:
        record.update(extra)
    return record


for _human_key in ("dandi_000469_human", "dandi_001187_human"):
    CANDIDATE_SUPPORT_MATRIX[_human_key] = {
        "memorandum_content": ("present", "the encoded picture identity (item_ids), native to this corpus's trial table"),
        "upcoming_response": ("absent", "the same/different probe response depends on a probe stimulus not yet presented during the maintenance-delay window this atlas's activity vector is built from; no population direction can encode an undetermined future response"),
        "gain_total_spike_count": ("present", None), "time_in_trial_within_delay": ("present", None),
        "previous_trial_content": ("present", "positional shift of item_ids, over the admitted-trial sequence"),
        "field_potential_low_frequency_band_power_ieeg": ("absent", "no field-potential loader exists in this project's registry for this corpus (only DANDI 000574 ships one)"),
        "field_potential_low_frequency_band_power_eeg": ("absent", "no field-potential loader exists in this project's registry for this corpus"),
        "field_potential_aperiodic_slope_ieeg": ("absent", "no field-potential loader exists in this project's registry for this corpus"),
        "field_potential_aperiodic_slope_eeg": ("absent", "no field-potential loader exists in this project's registry for this corpus"),
    }
CANDIDATE_SUPPORT_MATRIX["dandi_000574_human"] = {
    "memorandum_content": ("absent", "set_letters (item identity) is 'not available' on every trial in this public NWB release, per src/corpus_sessions.py's own iter_dandi_000574 item_id_unavailable_reason"),
    "upcoming_response": ("absent", "same reasoning as the other two human corpora: the probe-driven response is undetermined during the maintenance window"),
    "gain_total_spike_count": ("present", None), "time_in_trial_within_delay": ("present", None),
    "previous_trial_content": ("absent", "no item identity label exists in this corpus, so no previous-trial content label exists either"),
    "field_potential_low_frequency_band_power_ieeg": ("present", "depth macro-contact band power, run_state_space_estimation_admissibility.field_potential_sessions"),
    "field_potential_low_frequency_band_power_eeg": ("present", "scalp montage band power, same loader"),
    "field_potential_aperiodic_slope_ieeg": ("present", "depth macro-contact robust log-log aperiodic slope, this module's own new fit"),
    "field_potential_aperiodic_slope_eeg": ("present", "scalp montage robust log-log aperiodic slope, this module's own new fit"),
}
CANDIDATE_SUPPORT_MATRIX["ds006848_human_scalp"] = {
    "memorandum_content": ("present", "the first digit of the presented 7-digit sequence, native to this corpus's trial table"),
    "upcoming_response": ("absent", "the typed digit-by-digit response is produced only after the retention window this atlas's activity vector is built from; no population direction can encode a not-yet-produced response"),
    "gain_total_spike_count": ("present", "summed broadband band power, not spike counts; see run_rank_free_component_identity.GAIN_QUANTITY_BY_CORPUS"),
    "time_in_trial_within_delay": ("present", None),
    "previous_trial_content": ("present", "positional shift of the first-digit content label"),
    "field_potential_low_frequency_band_power_ieeg": ("absent", "no intracranial recording in this corpus"),
    "field_potential_low_frequency_band_power_eeg": ("absent", "this corpus's own channel-power activity vector IS the field potential; no separate co-registered signal exists to test as a candidate distinct from the session's own base activity"),
    "field_potential_aperiodic_slope_ieeg": ("absent", "no intracranial recording in this corpus"),
    "field_potential_aperiodic_slope_eeg": ("absent", "this corpus's own channel-power activity vector IS the field potential; no separate co-registered signal exists to test as a candidate distinct from the session's own base activity"),
}
CANDIDATE_SUPPORT_MATRIX["ds005034_human_scalp"] = {
    "memorandum_content": ("absent", "no per-trial item identity is recorded in this corpus's public BIDS release (config/datasets.json)"),
    "upcoming_response": ("absent", "no per-trial response is recorded in this corpus's public BIDS release"),
    "gain_total_spike_count": ("present", "summed broadband band power, not spike counts; see run_rank_free_component_identity.GAIN_QUANTITY_BY_CORPUS"),
    "time_in_trial_within_delay": ("present", None),
    "previous_trial_content": ("absent", "no item identity label exists in this corpus, so no previous-trial content label exists either"),
    "field_potential_low_frequency_band_power_ieeg": ("absent", "no intracranial recording in this corpus"),
    "field_potential_low_frequency_band_power_eeg": ("absent", "this corpus's own channel-power activity vector IS the field potential; no separate co-registered signal exists to test as a candidate distinct from the session's own base activity"),
    "field_potential_aperiodic_slope_ieeg": ("absent", "no intracranial recording in this corpus"),
    "field_potential_aperiodic_slope_eeg": ("absent", "this corpus's own channel-power activity vector IS the field potential; no separate co-registered signal exists to test as a candidate distinct from the session's own base activity"),
}
for _dandi_000004_key in ("dandi_000004_human_hippocampus", "dandi_000004_human_amygdala"):
    CANDIDATE_SUPPORT_MATRIX[_dandi_000004_key] = {
        "memorandum_content": ("present", "the trial's stimCategory code (session-local picture category, 5 categories per session), native to this corpus's trial table"),
        "upcoming_response": ("absent", "this project's human-corpus covariate pipeline does not extract a response label for any single-unit corpus; out of scope for the candidates this run computes"),
        "gain_total_spike_count": ("present", None), "time_in_trial_within_delay": ("present", None),
        "previous_trial_content": ("present", "positional shift of stimCategory, over the admitted-trial sequence"),
        "field_potential_low_frequency_band_power_ieeg": ("absent", "this corpus ships single-unit spikes only, no co-registered field potential"),
        "field_potential_low_frequency_band_power_eeg": ("absent", "this corpus ships single-unit spikes only, no co-registered field potential"),
        "field_potential_aperiodic_slope_ieeg": ("absent", "this corpus ships single-unit spikes only, no co-registered field potential"),
        "field_potential_aperiodic_slope_eeg": ("absent", "this corpus ships single-unit spikes only, no co-registered field potential"),
    }


# ============================================================================================
# Per-corpus session loops.
# ============================================================================================



def run_panichello(root: Path) -> list[dict]:
    records = []
    for session_id, reason, core, categorical, continuous, counts_all in _panichello_session_inputs(root):

        def _compute(session_id=session_id, reason=reason, core=core, categorical=categorical,
                     continuous=continuous, counts_all=counts_all):
            if core is None:
                return {"session": session_id, "status": "not_computable", "reason": reason}
            cells, support, cvr = _build_candidate_cells(core, f"panichello|{session_id}", categorical, continuous, counts_all)
            return _assemble_record(session_id, core, cvr, cells, support)

        records.append(_run_checkpointed(f"panichello|{session_id}", _compute))
    return records


def run_watters(root: Path) -> list[dict]:
    behaviour = watters_behaviour(root)
    dates = watters_session_dates(root)
    limit = _session_limit()
    if limit:
        dates = dates[:limit]
    records = []
    for animal, session_date, variant in dates:
        session_id = f"{animal}_{session_date}_{variant}"
        records.append(_run_checkpointed(
            f"watters|{session_id}",
            lambda animal=animal, session_date=session_date, variant=variant, session_id=session_id:
            _watters_session_record(root, behaviour, animal, session_date, variant, session_id),
        ))
    return records


def _watters_session_record(root: Path, behaviour, animal: str, session_date: str, variant: str, session_id: str) -> dict:
    """One macaque multi-object corpus session's checkpointed candidate-cell record -- the body run_watters used to run
    inline per session, factored out only so it can be wrapped in _run_checkpointed (rule: checkpoint
    per session) without touching any of its own computation."""
    session = load_watters_session(root, animal, session_date, behaviour)
    if session.get("status") != "loaded":
        return {"session": session_id, "status": "not_computable", "reason": session.get("status")}
    counts = session["counts"]  # (trials, units, bins)
    num_objects = session["num_objects"]
    cued = np.stack([np.cos(session["cued_theta"]), np.sin(session["cued_theta"])], axis=1)
    prev_cued = np.stack([_previous_label(cued[:, 0]), _previous_label(cued[:, 1])], axis=1)
    level_cells: dict[str, list[tuple[int, dict]]] = {}
    level_support: list[dict] = []
    level_core_summaries = []
    for level in sorted({int(v) for v in num_objects.tolist()}):
        mask = num_objects == level
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            continue
        activity_by_unit = counts[mask].sum(axis=2)
        core = _session_core(activity_by_unit)
        if core is None:
            continue
        cells, support, cvr = _build_candidate_cells(
            core, f"watters|{session_id}|level{level}",
            {}, {"gain_total_spike_count": core["spike_count"]}, counts[mask],
        )
        # memorandum_content and previous_trial_content are 2-D (cos, sin) continuous targets --
        # built separately since _build_candidate_cells' continuous_spec assumes one covariate
        # array per key with matching ndim handling (it already supports covariate.shape[1]>1).
        mem_basis, mem_supp = _continuous_candidate(core["U"], cued[mask][core["idx"]], dim=2)
        prev_basis, prev_supp = _continuous_candidate(core["U"], prev_cued[mask][core["idx"]], dim=2)
        occupied_basis, _ = _occupied_basis(core["U"], f"watters|{session_id}|level{level}|occupied")
        if occupied_basis is not None:
            cells["memorandum_content"] = ({"status": "absent", "reason": mem_supp.get("reason")} if mem_basis is None else
                _alignment_against_occupied_null(core["axis"], mem_basis, occupied_basis, N_RANDOM_SUBSPACE_DRAWS, f"watters|{session_id}|level{level}|memorandum_content"))
            cells["previous_trial_content"] = ({"status": "absent", "reason": prev_supp.get("reason")} if prev_basis is None else
                _alignment_against_occupied_null(core["axis"], prev_basis, occupied_basis, N_RANDOM_SUBSPACE_DRAWS, f"watters|{session_id}|level{level}|previous_trial_content"))
        else:
            cells["memorandum_content"] = {"status": "not_computable"}
            cells["previous_trial_content"] = {"status": "not_computable"}
        support["memorandum_content"] = mem_supp
        support["previous_trial_content"] = prev_supp
        for key, cell in cells.items():
            level_cells.setdefault(key, []).append((n_level, cell))
        level_support.append(support)
        level_core_summaries.append((n_level, core, cvr))
    if not level_core_summaries:
        return {"session": session_id, "status": "not_computable",
                "reason": "no item-count level reached the trial floor"}
    combined_cells = {key: _combine_levels_by_trial_count(entries) for key, entries in level_cells.items()}
    combined_support = {key: level_support[0].get(key, {"status": "absent"}) for key in combined_cells}
    n_kept_total = sum(n for n, _, _ in level_core_summaries)
    return {
        "session": session_id, "status": "computed",
        "n_levels": len(level_core_summaries), "n_kept_trials_total": n_kept_total,
        "identity_passed_every_level": all(c["identity_passed"] for _, c, _ in level_core_summaries),
        "candidate_support": combined_support,
        "candidates": {k: _strip_draws(v) for k, v in combined_cells.items()},
    }


def _alm_session_inputs(root: Path):
    """Yields (session_id, reason_or_None, core_or_None, categorical_spec, continuous_spec,
    counts_or_None) for every ALM session -- factored out of run_alm for the same reason as
    _panichello_session_inputs (rule-3 recomputation reuse)."""
    directory = alm_data_directory(root)
    paths = sorted(glob.glob(str(directory / "*.mat"))) if directory.is_dir() else []
    limit = _session_limit()
    if limit:
        paths = paths[:limit]
    for path in paths:
        session_id = Path(path).stem
        arrays = _alm_session_arrays(path, window_s=2.0, bin_ms=100.0)
        if arrays is None:
            yield session_id, "failed ALM eligibility floors", None, {}, {}, None
            continue
        counts = arrays["counts"]
        activity_by_unit = counts.sum(axis=2)
        core = _session_core(activity_by_unit)
        if core is None:
            yield (session_id, "fewer than the trial floor have a defined leave-one-out direction",
                   None, {}, {}, None)
            continue
        condition = arrays["condition"].astype(float)
        categorical = {"upcoming_response": condition, "previous_trial_content": _previous_label(condition)}
        continuous = {"gain_total_spike_count": core["spike_count"]}
        yield session_id, None, core, categorical, continuous, counts


def run_alm(root: Path) -> list[dict]:
    records = []
    for session_id, reason, core, categorical, continuous, counts in _alm_session_inputs(root):

        def _compute(session_id=session_id, reason=reason, core=core, categorical=categorical,
                     continuous=continuous, counts=counts):
            if core is None:
                return {"session": session_id, "status": "not_computable", "reason": reason}
            cells, support, cvr = _build_candidate_cells(core, f"alm|{session_id}", categorical, continuous, counts)
            return _assemble_record(session_id, core, cvr, cells, support)

        records.append(_run_checkpointed(f"alm|{session_id}", _compute))
    return records


def _human_session_generic(entry: dict, bin_ms: float = BIN_MS) -> dict | None:
    """Shared loader body for the three human corpora: builds the bin-resolved delay tensor with
    run_state_content_link.delay_counts (unchanged), sums it for activity_by_unit, and recovers this
    session's core geometry -- identical for all three corpora, only the trial-table fields differ,
    which the caller already resolved into `entry`."""
    spike_lists = entry["spike_lists"]
    onset = entry["epoch_onsets"]["delay"]
    window_s = entry["epoch_windows"]["delay"]
    counts_all = delay_counts(spike_lists, onset, window_s, bin_ms=bin_ms)
    activity_by_unit = counts_all.sum(axis=2)
    core = _session_core(activity_by_unit)
    if core is None:
        return None
    return {"core": core, "counts_all": counts_all}


def _human_corpus_session_record(root: Path, entry: dict, corpus_key: str, session_id: str) -> dict:
    """One human-corpus session's checkpointed candidate-cell record -- the body run_human_corpus
    used to run inline per session, factored out only so it can be wrapped in _run_checkpointed
    (rule: checkpoint per session) without touching any of its own computation."""
    loaded = _human_session_generic(entry)
    if loaded is None:
        return {"session": session_id, "status": "not_computable",
                "reason": "fewer than the trial floor have a defined leave-one-out direction"}
    core, counts_all = loaded["core"], loaded["counts_all"]
    categorical, continuous = {}, {"gain_total_spike_count": core["spike_count"]}
    item_ids = entry.get("item_ids")
    if item_ids is not None:
        item_ids = np.asarray(item_ids, dtype=float)
        categorical["memorandum_content"] = item_ids
        categorical["previous_trial_content"] = _previous_label(item_ids)
    cells, support, cvr = _build_candidate_cells(core, f"{corpus_key}|{session_id}", categorical, continuous, counts_all)

    if corpus_key == "dandi_000574_human":
        nwb_path = root / "000574" / entry["patient"] / f"{entry['session']}.nwb"
        occupied_basis, _ = _occupied_basis(core["U"], f"{corpus_key}|{session_id}|occupied")
        for signal, tag in (("ieeg", "ieeg"), ("eeg", "eeg")):
            scalars = None
            try:
                scalars = _boran_field_scalars(nwb_path, signal)
            except Exception as exc:  # field-potential preprocessing failure is a per-session event, not fatal
                support[f"field_potential_low_frequency_band_power_{tag}"] = {"status": "not_computable", "reason": str(exc)}
                support[f"field_potential_aperiodic_slope_{tag}"] = {"status": "not_computable", "reason": str(exc)}
                cells[f"field_potential_low_frequency_band_power_{tag}"] = {"status": "not_computable", "reason": str(exc)}
                cells[f"field_potential_aperiodic_slope_{tag}"] = {"status": "not_computable", "reason": str(exc)}
                continue
            if scalars is None or scalars["n_trials"] != core["n_trials_total"]:
                reason = "field-potential trial count does not match the spike-based session's own trial admission"
                for cand in (f"field_potential_low_frequency_band_power_{tag}", f"field_potential_aperiodic_slope_{tag}"):
                    support[cand] = {"status": "not_computable", "reason": reason}
                    cells[cand] = {"status": "not_computable", "reason": reason}
                continue
            for cand, covariate in (
                (f"field_potential_low_frequency_band_power_{tag}", scalars["band_power"]),
                (f"field_potential_aperiodic_slope_{tag}", scalars["aperiodic_slope"]),
            ):
                basis, supp = _continuous_candidate(core["U"], covariate[core["idx"]], dim=1)
                support[cand] = supp
                if basis is None or occupied_basis is None:
                    cells[cand] = {"status": "absent", "reason": supp.get("reason", "occupied basis not estimable")}
                    continue
                cells[cand] = _alignment_against_occupied_null(core["axis"], basis, occupied_basis, N_RANDOM_SUBSPACE_DRAWS, f"{corpus_key}|{session_id}|{cand}")

    return _assemble_record(session_id, core, cvr, cells, support)


def run_human_corpus(root: Path, iterator, corpus_key: str) -> list[dict]:
    limit = _session_limit()
    records = []
    n_seen = 0
    for entry in iterator(root):
        if entry.get("structure") != "pooled":
            continue
        n_seen += 1
        if limit and n_seen > limit:
            break
        session_id = f"{entry['patient']}__{entry['session']}"
        records.append(_run_checkpointed(
            f"{corpus_key}|{session_id}",
            lambda entry=entry, corpus_key=corpus_key, session_id=session_id:
            _human_corpus_session_record(root, entry, corpus_key, session_id),
        ))
    return records


# ============================================================================================
# Rule 0 -- reproduction gate. Reproduces results/rate_free_state_geometry_behavior_link.json's
# own pooled raw_outcome_vs_deviation.mean_value on the identical 11 reachable macaque spatial working-memory corpus sessions,
# to tolerance 1e-06, using rate_free_state_deviation unchanged (via the imported, unmodified
# _panichello_reproduction_gate_session_arrays). That pooled mean_value is
# state_persistence.slope_across_sessions_test's mean_diff over the per-session DETERMINISTIC
# Pearson r (deviation vs trial outcome, zero controls) -- deterministic because it does not depend
# on the delivered module's own permutation seed at all, only on the data -- so this gate recomputes
# each session's r directly with scipy.stats.pearsonr rather than re-running the delivered module's
# permutation-based significance test, which would add compute without changing whether the gate
# passes.
# ============================================================================================

def _panichello_reproduction_gate(root: Path) -> dict:
    delivered_path = ROOT / "results" / "rate_free_state_geometry_behavior_link.json"
    if not delivered_path.exists():
        return {"status": "not_computable",
                "reason": "results/rate_free_state_geometry_behavior_link.json does not exist"}
    delivered = json.loads(delivered_path.read_text())
    target = delivered.get("pooled", {}).get("raw_outcome_vs_deviation", {}).get("mean_value")
    delivered_session_ids = delivered.get("sessions_completed", [])
    if target is None:
        return {"status": "not_computable",
                "reason": "delivered artifact has no pooled.raw_outcome_vs_deviation.mean_value"}

    paths = _reachable_sessions(root)
    session_ids = [p.stem for p in paths]
    r_values = []
    for path in paths:
        arrays = _panichello_reproduction_gate_session_arrays(path)
        if arrays is None:
            continue
        r, _p_analytic = pearsonr(arrays["is_corr"], arrays["deviation"])
        r_values.append(float(r))
    pooled = slope_across_sessions_test(r_values, alternative="two-sided")
    recomputed = pooled.get("mean_value")
    session_ids_match = session_ids == delivered_session_ids
    passed = recomputed is not None and abs(recomputed - target) <= 1e-6 and session_ids_match
    return {
        "status": "reproduced_exactly" if passed else "reproduction_failed",
        "tolerance": 1e-6,
        "target_mean_value": target,
        "recomputed_mean_value": recomputed,
        "absolute_difference": (abs(recomputed - target) if recomputed is not None else None),
        "n_sessions_reachable": len(paths),
        "session_ids_match_delivered_artifact": session_ids_match,
        "delivered_session_ids": delivered_session_ids,
        "recomputed_session_ids": session_ids,
    }


# ============================================================================================
# Step 6 -- bias-only between-session control, this module's own recomputation of
# BIAS_ONLY_ALGORITHM_SOURCE_QUOTE's algorithm against a reaction-time association (see module
# docstring, BIAS-ONLY CONTROL): collapses each session to its own mean deviation and own mean
# outcome (already computed over the identical finite-trial mask the real per-session association
# used), one point per session, correlated BETWEEN sessions with pearson_permutation_test
# unchanged.
# ============================================================================================

def _bias_only_between_session_atlas(session_records: list[dict], seed_tag: str) -> dict:
    means_x = [r["mean_deviation"] for r in session_records if r.get("status") == "computed"]
    means_y = [r["mean_outcome"] for r in session_records if r.get("status") == "computed"]
    n_sessions = len(means_x)
    if n_sessions < 4:
        return {"status": "not_computable", "n_sessions": n_sessions,
                "reason": "fewer than 4 sessions reach a computed real association"}
    rng = np.random.default_rng(stable_seed(seed_tag))
    result = pearson_permutation_test(np.array(means_x), np.array(means_y), n_perm=N_PERM, rng=rng)
    result["status"] = "computed"
    result["n_sessions"] = n_sessions
    return result


def _real_minus_control(real: dict, bias_only: dict) -> dict:
    """real.mean_value (within-session, sign-flip-pooled r) and bias_only.r (a between-session
    correlation) are both on the correlation scale. Interval combines the real statistic's own
    sign-flip-test confidence interval (converted to a standard error) with a Fisher-z delta-method
    standard error for the bias-only correlation."""
    real_value, control_value = real.get("mean_value"), bias_only.get("r")
    if real.get("status") != "tested" or bias_only.get("status") != "computed" \
            or real_value is None or control_value is None:
        return {"status": "not_applicable"}
    estimate = real_value - control_value
    n_bias = bias_only.get("n_sessions")
    if not n_bias or n_bias < 4:
        return {"status": "not_computable", "estimate": estimate, "real_value": real_value,
                "control_value": control_value}
    real_se = (real["ci_upper"] - real["ci_lower"]) / (2.0 * Z_95)
    bias_se = float((1.0 - control_value ** 2) / np.sqrt(n_bias - 3))
    se = float(np.sqrt(real_se ** 2 + bias_se ** 2))
    return {
        "status": "computed", "estimate": estimate, "real_value": real_value,
        "control_value": control_value, "standard_error": se,
        "interval_95pct": [estimate - Z_95 * se, estimate + Z_95 * se],
    }


def _reaction_time_branch(session_records: list[dict], seed_tag: str) -> dict:
    """Real association (pooled across sessions by the paired sign-flip test on each session's own
    deterministic r) plus its bias-only between-session control -- voided only if the control is
    ALSO significant (two-sided p<0.05) in the SAME direction as the real association; sign and
    significance only, never a magnitude comparison (BIAS_ONLY_ALGORITHM_SOURCE_QUOTE)."""
    computed = [r for r in session_records if r.get("status") == "computed"]
    r_values = [r["r"] for r in computed]
    real = slope_across_sessions_test(r_values, alternative="two-sided") if r_values else {
        "status": "not_computable", "n_sessions": 0,
        "reason": "no session reached a computed reaction-time association",
    }
    bias_only = _bias_only_between_session_atlas(computed, f"{seed_tag}|bias_only")
    real_significant = bool(real.get("status") == "tested" and real.get("significant"))
    bias_significant = bool(bias_only.get("status") == "computed" and bias_only.get("p_value", 1.0) < 0.05)
    same_sign = bool(
        real_significant and bias_significant
        and (real.get("mean_value", 0.0) > 0.0) == (bias_only.get("r", 0.0) > 0.0)
    )
    voided = bool(real_significant and bias_significant and same_sign)
    return {
        "status": "reaction_time_association",
        "n_sessions_computed": len(computed),
        "real_association": real,
        "bias_only_between_session_control": bias_only,
        "voided_by_bias_only_control": voided,
        "real_minus_control": _real_minus_control(real, bias_only),
        "voiding_rule": (
            "voids the real association only if the bias-only between-session control is ALSO "
            "significant (two-sided p<0.05) in the SAME direction; sign and significance only, "
            "never a magnitude comparison -- see BIAS_ONLY_ALGORITHM_SOURCE_QUOTE"
        ),
    }


def run_watters_reaction_time(root: Path) -> list[dict]:
    """Per macaque multi-object corpus session: whole-session (every item-count level pooled together) rate-free
    deviation vs the corpus's native reaction_time_ms column, zero-control Pearson correlation
    (partial_correlation_permutation_test). Pooled across item-count levels rather than combined by
    level as the candidate-subspace cells above are -- reaction time is a single per-trial scalar
    outcome, not a level-conditioned subspace fit, so there is no shared-basis reason to keep levels
    separate here; a disclosed simplification relative to the per-level candidate machinery."""
    behaviour = watters_behaviour(root)
    dates = watters_session_dates(root)
    limit = _session_limit()
    if limit:
        dates = dates[:limit]
    records = []
    for animal, session_date, variant in dates:
        session_id = f"{animal}_{session_date}_{variant}"

        def _compute(animal=animal, session_date=session_date, session_id=session_id):
            session = load_watters_session(root, animal, session_date, behaviour)
            if session.get("status") != "loaded":
                return {"session": session_id, "status": "not_computable", "reason": session.get("status")}
            counts = session["counts"]
            activity_by_unit = counts.sum(axis=2)
            core = _session_core(activity_by_unit)
            if core is None:
                return {"session": session_id, "status": "not_computable",
                        "reason": "fewer than the trial floor have a defined leave-one-out direction"}
            deviation = core["deviation"][core["idx"]]
            rt = np.asarray(session["reaction_time_ms"], dtype=float)[core["idx"]]
            finite = np.isfinite(deviation) & np.isfinite(rt)
            if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
                return {"session": session_id, "status": "not_computable",
                        "reason": "fewer than the trial floor have both a defined deviation and a finite reaction time"}
            deviation, rt = deviation[finite], rt[finite]
            tag = f"watters_reaction_time|{session_id}"
            cell = partial_correlation_permutation_test(
                rt, deviation, [], n_perm=N_PERM, rng=np.random.default_rng(stable_seed(tag)))
            cell["status"] = "computed"
            cell["session"] = session_id
            cell["n_trials"] = int(finite.sum())
            cell["mean_deviation"] = float(np.mean(deviation))
            cell["mean_outcome"] = float(np.mean(rt))
            return cell

        records.append(_run_checkpointed(f"watters_reaction_time|{session_id}", _compute))
    return records


REACTION_TIME_NOT_ASKABLE_REASONS = {
    "panichello_2024_macaque_lPFC": (
        "no timestamp field in this corpus's raw .mat: it carries only cueAng, cueAngIdx, isCorr, spks, tc"
    ),
    "dandi_000469_human": (
        "no response-time field in the trial tables src/corpus_sessions.py already reads for this corpus"
    ),
    "dandi_001187_human": (
        "no response-time field in the trial tables src/corpus_sessions.py already reads for this corpus"
    ),
    "dandi_000574_human": (
        "no response-time field in the trial tables src/corpus_sessions.py already reads for this corpus"
    ),
}

INAGAKI_ALM5_REACTION_TIME_NOT_COMPUTED_REASON = (
    "First_lick minus Cue_start is present in this corpus's raw behaviour struct (Behavior.First_lick, "
    "Behavior.Cue_start) but reporting it as an accuracy-controlled reaction-time association requires a "
    "verified mapping of Trial_types_of_response_vector to a binary correct/error label to serve as the "
    "accuracy control this analysis requires alongside spike count; that mapping was not verified against "
    "the source publication within this task's scope. Reporting an RT number without a checked accuracy "
    "control would be silently substituting a weaker analysis for the one asked for, so this corpus's "
    "reaction-time cell is 'not_computed' with this reason stated, not a fabricated number."
)


def _reaction_time_block(root: Path) -> dict:
    block: dict[str, dict] = {}
    watters_records = run_watters_reaction_time(root)
    block["watters_2026_macaque_multi_object"] = {
        "per_session": watters_records,
        **_reaction_time_branch(watters_records, "reaction_time|watters_2026_macaque_multi_object"),
    }
    block["inagaki_alm5_mouse_ALM"] = {
        "status": "not_computed", "reason": INAGAKI_ALM5_REACTION_TIME_NOT_COMPUTED_REASON,
    }
    for corpus_key, reason in REACTION_TIME_NOT_ASKABLE_REASONS.items():
        block[corpus_key] = {"status": "reaction_time_not_askable_in_this_corpus", "reason": reason}
    return block


# ============================================================================================
# Rule 3 -- mutual Gram-Schmidt orthogonalisation of a corpus's own co-aligned candidates, and the
# per-corpus session-input sources it (and only it) needs to rebuild candidate BASIS matrices --
# the primary pass's own cells carry alignment statistics, not the raw bases, since orthogonalising
# a candidate against another is only meaningful once we already know which candidates aligned
# (rule 1), a fact not available during the primary per-session pass. Recomputes core session
# geometry from the raw source rather than reusing the primary pass's checkpointed records for
# exactly that reason.
# ============================================================================================

def _human_session_covariate_inputs(root: Path, iterator, corpus_key: str):
    """Rule-3-only: yields (session_id, core_or_None, categorical_spec, continuous_spec,
    counts_all_or_None) for every 'pooled' session of one human corpus, duplicating the minimal
    loading _human_session_generic and (for dandi_000574_human) _boran_field_scalars already do in
    run_human_corpus's own primary pass, so the orthogonalisation recomputation below can rebuild
    every candidate's basis without threading state out of that pass."""
    limit = _session_limit()
    n_seen = 0
    for entry in iterator(root):
        if entry.get("structure") != "pooled":
            continue
        n_seen += 1
        if limit and n_seen > limit:
            break
        session_id = f"{entry['patient']}__{entry['session']}"
        loaded = _human_session_generic(entry)
        if loaded is None:
            yield session_id, None, {}, {}, None
            continue
        core, counts_all = loaded["core"], loaded["counts_all"]
        categorical, continuous = {}, {"gain_total_spike_count": core["spike_count"]}
        item_ids = entry.get("item_ids")
        if item_ids is not None:
            item_ids = np.asarray(item_ids, dtype=float)
            categorical["memorandum_content"] = item_ids
            categorical["previous_trial_content"] = _previous_label(item_ids)
        if corpus_key == "dandi_000574_human":
            nwb_path = root / "000574" / entry["patient"] / f"{entry['session']}.nwb"
            for signal, tag in (("ieeg", "ieeg"), ("eeg", "eeg")):
                try:
                    scalars = _boran_field_scalars(nwb_path, signal)
                except Exception:
                    scalars = None
                if scalars is not None and scalars["n_trials"] == core["n_trials_total"]:
                    continuous[f"field_potential_low_frequency_band_power_{tag}"] = scalars["band_power"]
                    continuous[f"field_potential_aperiodic_slope_{tag}"] = scalars["aperiodic_slope"]
        yield session_id, core, categorical, continuous, counts_all


def _gram_schmidt_bases(bases_in_order: list[tuple[str, np.ndarray]]) -> dict[str, np.ndarray | None]:
    """Mutually orthogonalises a list of (key, basis) column-space matrices against each other, in
    the given order (rule 3: CANDIDATE_KEYS tuple order): each candidate's own basis is
    re-orthonormalised against the span of every EARLIER candidate in the order, so shared variance
    is counted once, by whichever candidate comes first. A candidate basis fully absorbed by the
    span of earlier ones (numerically zero residual rank) maps to None."""
    orthogonalized: dict[str, np.ndarray | None] = {}
    accumulated_basis = None  # (p, k_so_far) orthonormal columns spanning every earlier candidate
    for key, basis in bases_in_order:
        b = basis.copy()
        if accumulated_basis is not None:
            b = b - accumulated_basis @ (accumulated_basis.T @ b)
        q, r = np.linalg.qr(b)
        keep = np.abs(np.diag(r)) > 1e-10
        q = q[:, keep]
        if q.shape[1] == 0:
            orthogonalized[key] = None
            continue
        orthogonalized[key] = q
        accumulated_basis = q if accumulated_basis is None else np.column_stack([accumulated_basis, q])
    return orthogonalized


def _orthogonalized_alignment_cell(core: dict, aligned_keys: list[str], categorical_spec: dict,
                                    continuous_spec: dict, time_in_trial_counts, seed_prefix: str) -> dict:
    """Rebuilds each of aligned_keys' own basis with the identical builders _build_candidate_cells
    uses (_categorical_candidate / _continuous_candidate / _time_in_trial_basis), mutually
    Gram-Schmidt-orthogonalises them, and realigns the deviation axis against each orthogonalised
    basis, against the same occupied-space null as the primary pass."""
    U, idx, axis = core["U"], core["idx"], core["axis"]
    occupied_basis, _cvr = _occupied_basis(U, f"{seed_prefix}|occupied")
    if occupied_basis is None:
        return {key: {"status": "not_computable", "reason": "occupied-space rank not estimable this session"}
                for key in aligned_keys}
    raw_bases: dict[str, np.ndarray] = {}
    for key in aligned_keys:
        basis = None
        if key in categorical_spec:
            basis, _supp = _categorical_candidate(U, categorical_spec[key][idx].astype(float))
        elif key in continuous_spec:
            covariate = continuous_spec[key][idx]
            basis, _supp = _continuous_candidate(U, covariate, dim=1 if covariate.ndim == 1 else covariate.shape[1])
        elif key == "time_in_trial_within_delay" and time_in_trial_counts is not None:
            basis = _time_in_trial_basis(time_in_trial_counts[idx])
        if basis is not None:
            raw_bases[key] = basis
    ordered_keys = [k for k in CANDIDATE_KEYS if k in raw_bases]
    orthogonalized = _gram_schmidt_bases([(k, raw_bases[k]) for k in ordered_keys])
    cells: dict[str, dict] = {}
    for key in aligned_keys:
        basis = orthogonalized.get(key)
        if basis is None:
            cells[key] = {"status": "not_computable",
                          "reason": "basis not available this session, or fully absorbed by an earlier "
                                    "candidate's span under Gram-Schmidt orthogonalisation"}
            continue
        cells[key] = _alignment_against_occupied_null(
            axis, basis, occupied_basis, N_RANDOM_SUBSPACE_DRAWS, f"{seed_prefix}|orthogonalized|{key}")
    return cells


def _orthogonalized_alignment_for_watters(root: Path, aligned_keys: list[str]) -> dict:
    """Rule 3 for the multi-object corpus: per session, per item-count level (matching the primary
    pass's own per-level subspace-fitting granularity), rebuilds and mutually orthogonalises this
    corpus's own aligned candidates' bases, realigns the deviation axis against each, and combines
    levels by trial count (_combine_levels_by_trial_count) exactly as the primary pass does, before
    pooling across sessions."""
    behaviour = watters_behaviour(root)
    dates = watters_session_dates(root)
    limit = _session_limit()
    if limit:
        dates = dates[:limit]
    per_key_sessions: dict[str, list] = {key: [] for key in aligned_keys}
    for animal, session_date, variant in dates:
        session = load_watters_session(root, animal, session_date, behaviour)
        session_id = f"{animal}_{session_date}_{variant}"
        if session.get("status") != "loaded":
            for key in aligned_keys:
                per_key_sessions[key].append(None)
            continue
        counts = session["counts"]
        num_objects = session["num_objects"]
        cued = np.stack([np.cos(session["cued_theta"]), np.sin(session["cued_theta"])], axis=1)
        prev_cued = np.stack([_previous_label(cued[:, 0]), _previous_label(cued[:, 1])], axis=1)
        level_entries: dict[str, list] = {key: [] for key in aligned_keys}
        for level in sorted({int(v) for v in num_objects.tolist()}):
            mask = num_objects == level
            n_level = int(mask.sum())
            if n_level < MIN_TRIALS_WITH_DEFINED_DIRECTION:
                continue
            activity_by_unit = counts[mask].sum(axis=2)
            core = _session_core(activity_by_unit)
            if core is None:
                continue
            continuous = {
                "gain_total_spike_count": core["spike_count"],
                "memorandum_content": cued[mask],
                "previous_trial_content": prev_cued[mask],
            }
            cells = _orthogonalized_alignment_cell(
                core, aligned_keys, {}, continuous, counts[mask],
                f"watters|{session_id}|level{level}|orthogonalized")
            for key in aligned_keys:
                cell = cells.get(key)
                if cell is not None and cell.get("status") == "computed":
                    level_entries[key].append((n_level, cell))
        for key in aligned_keys:
            if level_entries[key]:
                per_key_sessions[key].append(_strip_draws(_combine_levels_by_trial_count(level_entries[key])))
            else:
                per_key_sessions[key].append(None)
    return {key: _pool_candidate_cells(sessions) for key, sessions in per_key_sessions.items()}


def _orthogonalized_alignment_for_corpus(corpus_key: str, root: Path, aligned_keys: list[str]) -> dict:
    if corpus_key == "watters_2026_macaque_multi_object":
        return _orthogonalized_alignment_for_watters(root, aligned_keys)
    if corpus_key == "panichello_2024_macaque_lPFC":
        raw_iter = ((sid, core, cat, cont, cnt) for sid, _reason, core, cat, cont, cnt in _panichello_session_inputs(root))
    elif corpus_key == "inagaki_alm5_mouse_ALM":
        raw_iter = ((sid, core, cat, cont, cnt) for sid, _reason, core, cat, cont, cnt in _alm_session_inputs(root))
    elif corpus_key in ("dandi_000469_human", "dandi_001187_human", "dandi_000574_human"):
        iterator = {"dandi_000469_human": iter_dandi_000469, "dandi_001187_human": iter_dandi_001187,
                    "dandi_000574_human": iter_dandi_000574}[corpus_key]
        raw_iter = _human_session_covariate_inputs(root, iterator, corpus_key)
    else:
        return {key: {"status": "not_computable", "reason": f"no session-input source registered for {corpus_key}"}
                for key in aligned_keys}

    per_key_sessions: dict[str, list] = {key: [] for key in aligned_keys}
    for session_id, core, categorical, continuous, counts in raw_iter:
        if core is None:
            for key in aligned_keys:
                per_key_sessions[key].append(None)
            continue
        cells = _orthogonalized_alignment_cell(
            core, aligned_keys, categorical, continuous, counts, f"{corpus_key}|{session_id}|orthogonalized")
        for key in aligned_keys:
            per_key_sessions[key].append(_strip_draws(cells.get(key)))
    return {key: _pool_candidate_cells(sessions) for key, sessions in per_key_sessions.items()}


# ============================================================================================
# Rules 1-4: per-corpus BH-FDR alignment, cross-corpus aligned_anywhere read, and final branch.
# ============================================================================================

def _corpus_candidate_pooled(atlas: dict, corpus_key: str, candidate_key: str) -> dict:
    """Session-level cells for one (corpus, candidate) pair, pooled with _pool_candidate_cells, plus
    its own 80%-power detection floor (_session_cluster_bootstrap_mdd) -- both computed from the
    identical per-session cell list."""
    records = atlas[corpus_key]
    per_session = [r.get("candidates", {}).get(candidate_key) for r in records if r.get("status") == "computed"]
    pooled = _pool_candidate_cells(per_session)
    mdd = _session_cluster_bootstrap_mdd(per_session, f"{corpus_key}|{candidate_key}|mdd_bootstrap")
    return {"pooled": pooled, "detection_floor_80pct_power": mdd}


def _build_alignment_atlas(atlas: dict) -> dict:
    """Rule 1: within each corpus, BH-FDR (alpha=FDR_ALPHA) across every candidate that corpus's own
    CANDIDATE_SUPPORT_MATRIX declares 'present' and that reached a pooled, tested cell -- one test
    family per corpus (the corpus's own declared candidate set), not one pooled across corpora."""
    result: dict[str, dict] = {}
    for corpus_key in CORPORA:
        supported = [k for k in CANDIDATE_KEYS if CANDIDATE_SUPPORT_MATRIX[corpus_key][k][0] == "present"]
        cells = {k: _corpus_candidate_pooled(atlas, corpus_key, k) for k in supported}
        testable = [k for k in supported if cells[k]["pooled"].get("status") == "tested"]
        p_values = [cells[k]["pooled"]["p_value"] for k in testable]
        if p_values:
            fdr = fdr_bh(np.array(p_values), alpha=FDR_ALPHA)
            reject_by_key = dict(zip(testable, [bool(v) for v in fdr["reject"]]))
            q_by_key = dict(zip(testable, [float(v) for v in fdr["q_values"]]))
        else:
            reject_by_key, q_by_key = {}, {}
        candidate_summaries: dict[str, dict] = {}
        for k in supported:
            candidate_summaries[k] = {
                **cells[k], "fdr_q_value": q_by_key.get(k), "aligned": bool(reject_by_key.get(k, False)),
                "n_candidates_in_this_corpus_fdr_family": len(testable),
            }
        for k in CANDIDATE_KEYS:
            if k not in supported:
                candidate_summaries[k] = {"status": "absent_from_this_corpus",
                                           "reason": CANDIDATE_SUPPORT_MATRIX[corpus_key][k][1]}
        result[corpus_key] = candidate_summaries
    return result


def _aligned_anywhere(alignment_atlas: dict) -> dict[str, dict]:
    """Rule 2: for each candidate TYPE, aligned_anywhere = true if it is ALIGNED (rule 1) in at
    least one corpus that supports it -- read across the whole atlas, not per corpus."""
    summary = {}
    for candidate_key in CANDIDATE_KEYS:
        corpora_aligned_in = [ck for ck in CORPORA if alignment_atlas[ck].get(candidate_key, {}).get("aligned")]
        summary[candidate_key] = {"aligned_anywhere": bool(corpora_aligned_in), "corpora_aligned_in": corpora_aligned_in}
    return summary


def _decide_branch(alignment_atlas: dict, aligned_anywhere: dict, root: Path) -> dict:
    resolved_types = [k for k in CANDIDATE_KEYS if aligned_anywhere[k]["aligned_anywhere"]]

    if len(resolved_types) == 1:
        key = resolved_types[0]
        return {"branch": "component_identity_resolved_to_a_single_labelled_subspace",
                "resolved_candidate": key, "corpora_aligned_in": aligned_anywhere[key]["corpora_aligned_in"]}

    if len(resolved_types) >= 2:
        per_corpus_orthogonalisation = {}
        for corpus_key in CORPORA:
            own_aligned = [k for k in CANDIDATE_KEYS if alignment_atlas[corpus_key].get(k, {}).get("aligned")]
            if len(own_aligned) >= 2:
                per_corpus_orthogonalisation[corpus_key] = {
                    "own_aligned_candidates": own_aligned,
                    "gram_schmidt_order": [k for k in CANDIDATE_KEYS if k in own_aligned],
                    "orthogonalized_alignment": _orthogonalized_alignment_for_corpus(corpus_key, root, own_aligned),
                }
        return {"branch": "component_identity_is_multiply_aligned",
                "aligned_candidate_types": resolved_types,
                "per_corpus_mutual_orthogonalisation": per_corpus_orthogonalisation}

    # No candidate type aligned anywhere: rule 4 requires every supported cell to carry its own
    # 80%-power detection floor before this can be reported as a positive null.
    missing_floor = []
    for corpus_key in CORPORA:
        for k in CANDIDATE_KEYS:
            if CANDIDATE_SUPPORT_MATRIX[corpus_key][k][0] != "present":
                continue
            mdd = alignment_atlas[corpus_key][k]["detection_floor_80pct_power"]
            if mdd.get("status") != "computed":
                missing_floor.append(f"{corpus_key}|{k}")
    if not missing_floor:
        return {"branch": "component_is_not_aligned_with_any_labelled_subspace_tested",
                "note": "every supported candidate cell in this atlas carries its own 80%-power detection floor"}
    return {"branch": "inconclusive_below_detection_floor", "cells_missing_a_detection_floor": missing_floor}


# ============================================================================================
# Zero-drop accounting (project non-negotiable): every corpus's own runner already guarantees one
# record per session it iterates (either 'computed' or 'not_computable' with a stated reason), so
# n_seen = len(records) for every corpus in this atlas -- this is what that guarantee reconciles to.
# ============================================================================================

def _zero_drop_summary(records: list[dict]) -> dict:
    n_seen = len(records)
    computed = [r for r in records if r.get("status") == "computed"]
    refused = [r for r in records if r.get("status") != "computed"]
    refusals_by_reason: dict[str, int] = {}
    for r in refused:
        reason = str(r.get("reason", r.get("status", "unknown")))
        refusals_by_reason[reason] = refusals_by_reason.get(reason, 0) + 1
    n_analysed, n_refused = len(computed), len(refused)
    reconciles = (n_seen == n_analysed + n_refused)
    return {"n_seen": n_seen, "n_loaded": n_analysed, "n_refused": n_refused,
            "refusals_by_reason": refusals_by_reason, "n_analysed": n_analysed, "reconciles": reconciles}


def _rank_dependency_gate(path: Path) -> dict:
    if not path.is_file():
        return {"status": "failed", "reason": "rank_artifact_missing"}
    artifact = json.loads(path.read_text())
    branches = artifact.get("branch", {})
    cleared = artifact.get("status") == "complete" and bool(branches) and all(
        not str(branch).startswith("not_determinable") for branch in branches.values()
    )
    return {
        "status": "cleared" if cleared else "failed",
        "source": str(path),
        "rank_artifact_status": artifact.get("status"),
        "rank_branches": branches,
        "reason": None if cleared else "occupied_rank_is_not_identified_for_every_required_corpus",
    }


SCOPE = (
    "Six corpora sharing one already-delivered, unmodified per-trial estimator "
    "(run_rate_free_state_geometry_behavior_link.rate_free_state_deviation) and one already-"
    "delivered axis recovery (run_deviation_axis_structure.leading_eigenvector applied to its own "
    "_unit_residual_matrix): panichello_2024_macaque_lPFC (single-item delayed saccade, up to 25 "
    "sessions), watters_2026_macaque_multi_object (multi-object continuous report, up to 47 session "
    "dates, 1-3 item-count levels combined by trial-count weighting), inagaki_alm5_mouse_ALM "
    "(motor-planning delayed response, up to 23 sessions -- this corpus's own orthogonality gate in "
    "run_dissociation_cross_preparation_test.py found its deviation observable MORE rate-"
    "contaminated than the macaque reference in that comparison, so every candidate-alignment number "
    "from this corpus carries that caveat and is never read as validating the rate-free construction "
    "here, only as the upcoming-response subspace test this corpus was explicitly commissioned for), "
    "and three human intracranial corpora restricted to their 'pooled' region structure only "
    "(dandi_000469_human, dandi_001187_human, dandi_000574_human; up to 18, 28 and 26 sessions "
    "respectively). Up to nine candidate labelled subspaces (CANDIDATE_KEYS) are tested only where "
    "each corpus's own data supports them (CANDIDATE_SUPPORT_MATRIX records absence with a reason "
    "rather than fabricating a cell). Alignment is against a null of subspaces drawn uniformly at "
    "random INSIDE each session's own occupied subspace (a cross-validated-PCA-rank estimate), not "
    f"the ambient unit space, {N_RANDOM_SUBSPACE_DRAWS} draws per cell, matched in dimension to the "
    "candidate under test -- the one new piece of null construction this module adds (see module "
    "docstring, NULL CONSTRUCTION). Per corpus, candidate alignment is Benjamini-Hochberg FDR "
    f"corrected at alpha={FDR_ALPHA} across every candidate that corpus supports; a candidate type "
    "is read as aligned_anywhere if it clears that correction in at least one corpus, across the "
    "whole atlas. Reaction time is tested only in watters_2026_macaque_multi_object (native "
    "reaction_time_ms, whole-session pooled across item-count levels -- a disclosed simplification "
    "relative to the per-level candidate-subspace machinery); inagaki_alm5's raw response code lacks "
    "a verified accuracy mapping in this task's scope so its reaction-time cell is not_computed "
    "rather than fabricated; panichello_2024 and the three human corpora have no response-time field "
    "at all and are reaction_time_not_askable_in_this_corpus. Every reported reaction-time "
    "association carries a between-session bias-only control (collapsing each session to its own "
    "mean deviation and mean outcome, correlated BETWEEN sessions) and is voided only if that "
    "control is ALSO significant in the same direction -- sign and significance only, never a "
    "magnitude comparison. Every session this atlas's corpus iterators enumerate is counted by "
    "machine-readable outcome (computed, or refused with a stated reason) in zero_drop_accounting; "
    "no session is silently dropped."
)


def main() -> None:
    root = data_root()
    t0 = time.time()
    output: dict = {
        "version": ANALYSIS_VERSION, "scope": SCOPE,
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "status": "running",
    }
    _flush(output)
    rank_gate = _rank_dependency_gate(ROOT / "results" / "occupied_subspace_rank_estimation.json")
    output["rank_dependency_gate"] = rank_gate
    if rank_gate["status"] != "cleared":
        output["status"] = "complete"
        output["branch"] = {"branch": "unresolved_because_occupied_rank_is_not_identified"}
        output["zero_drop_accounting"] = {
            "reconciles": None,
            "reason": "run stopped before corpus loading because the required occupied-rank dependency failed",
        }
        output["git_commit"] = git_commit(ROOT)
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        return
    _log("reproduction_gate: starting")
    gate = _panichello_reproduction_gate(root)
    output["reproduction_gate"] = gate
    _flush(output)
    _log(f"reproduction_gate: {gate.get('status')}")

    if gate.get("status") != "reproduced_exactly":
        output["status"] = "complete"
        output["branch"] = {"branch": "void_reproduction_gate_failed"}
        output["zero_drop_accounting"] = {
            "reconciles": None,
            "reason": "run stopped at the reproduction gate, before any corpus was analysed",
        }
        output["git_commit"] = git_commit(ROOT)
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("FINISHED run_component_identity_subspace_atlas: void_reproduction_gate_failed")
        return

    corpus_runners = {
        "panichello_2024_macaque_lPFC": run_panichello,
        "watters_2026_macaque_multi_object": run_watters,
        "inagaki_alm5_mouse_ALM": run_alm,
        "dandi_000469_human": lambda r: run_human_corpus(r, iter_dandi_000469, "dandi_000469_human"),
        "dandi_001187_human": lambda r: run_human_corpus(r, iter_dandi_001187, "dandi_001187_human"),
        "dandi_000574_human": lambda r: run_human_corpus(r, iter_dandi_000574, "dandi_000574_human"),
    }

    atlas: dict[str, list[dict]] = {}
    output["atlas"] = atlas
    for corpus_key in CORPORA:
        _log(f"corpus: starting {corpus_key}")
        atlas[corpus_key] = corpus_runners[corpus_key](root)
        _flush(output)
        _log(f"corpus: finished {corpus_key} ({len(atlas[corpus_key])} sessions)")

    per_corpus_zero_drop = {ck: _zero_drop_summary(atlas[ck]) for ck in CORPORA}
    output["zero_drop_accounting"] = {
        **per_corpus_zero_drop,
        "reconciles": all(v["reconciles"] for v in per_corpus_zero_drop.values()),
    }
    _flush(output)

    _log("reaction_time: starting")
    output["reaction_time"] = _reaction_time_block(root)
    _flush(output)
    _log("reaction_time: finished")

    _log("candidate_alignment: pooling")
    alignment_atlas = _build_alignment_atlas(atlas)
    output["candidate_alignment_by_corpus"] = alignment_atlas
    aligned_anywhere = _aligned_anywhere(alignment_atlas)
    output["aligned_anywhere_by_candidate_type"] = aligned_anywhere
    _flush(output)

    branch = _decide_branch(alignment_atlas, aligned_anywhere, root)
    output["branch"] = branch
    output["status"] = "complete"
    output["git_commit"] = git_commit(ROOT)
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    _log(f"FINISHED run_component_identity_subspace_atlas: {branch['branch']}")


if __name__ == "__main__":
    _data_root_env = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not _data_root_env or not Path(_data_root_env).is_dir():
        raise SystemExit(
            f"WM_DYNAMICS_DATA_ROOT is not set to an existing, currently-mounted directory "
            f"(got {_data_root_env!r}). This run requires it to point at the mounted external data "
            "drive -- see this module's own docstring, ENVIRONMENT NOTE. Refusing to silently fall "
            "back to any other default."
        )
    main()
