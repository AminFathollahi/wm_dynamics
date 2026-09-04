"""Estimation-choice robustness ladder for the five rank-withdrawn state-space claims.

The dimensionality sweep withdrew five latent-dependent claims because their verdicts flip
somewhere in the swept rank grid relative to their native full-rank anchors. This module asks,
claim by claim and estimator by estimator, whether that withdrawal reflects the projection step
or the data: each claim is recomputed at one fixed operating point under every admissible
estimation choice from the project's admissibility roster, and the verdicts are compared.

This is a robustness audit and never a comparison between estimators. No sentence in this module,
its tests, or the artifact it writes may say that one estimator ranked above, beat, or was better
than another on any claim. The only question asked is whether the project's conclusions depend on
how the state space was estimated.

Claims, in delivery priority order:

  1. cross-temporal generalisation area-under-curve (human intracranial corpus);
  2. memorandum decoding defining the coding subspace, against the trial-wise deviation score
     (same corpus);
  3. occupied-manifold-versus-deviation decomposition (same corpus);
  4. fitted linear dynamics (dominant eigenmode class and spectral decay radius; causal
     microstimulation macaque corpus);
  5. linear control model (stimulation-input targeting alignment; same corpus).

Two standing guards are enforced in code, not only in prose:

  - the shared fitting entry point refuses a non-null label argument. Positive pairs for any
    embedding trained here are defined by temporal adjacency and nothing else; no trial label, no
    accuracy, no response, no reaction time and no epoch identity enters any fitting objective;
  - the subspace-angle path refuses any representation without a canonical orthonormal basis.
    Every linear-geometry claim reaches a nonlinear estimator only through the estimator-invariant
    restatement -- the cross-validated predictable fraction of the deviation component's variance,
    defined identically for a projection and for an embedding -- reported beside the delivered
    linear-geometry quantity wherever that quantity exists.

Execution tiers: tier one (native full rank, principal components, factor analysis, Gaussian-
process factor analysis where the data type admits it, trial-level variational autoencoder,
temporal diffusion embedding) and tier two (time-contrastive embedding) are freshly fitted per
session -- the admissibility checkpoints hold scores, not fitted representations, so nothing at
these tiers is reusable from cache. The hours-per-session sequential autoencoder runs only where
tier one and tier two disagree (per claim, never project-wide), on a seeded random sample sized so
its minimum detectable difference falls below the claim's tier-one effect size, declared before
any tier-three number exists; a sample larger than this pass's budget cap is reported as requiring
a human budget decision rather than silently truncated.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_state_space_estimation_robustness.py \
        [--phase all|human|microstim] [--candidates ...] [--sessions-limit N] \
        [--n-perm-ctg N] [--n-perm-restatement N] [--run-rung-three]
"""
from __future__ import annotations

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    import os as _os
    _os.environ[_var] = "1"

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import data_root, iter_dandi_000469  # noqa: E402
from geometry import _ctg_score_fold_multiclass, temporal_stability_tau  # noqa: E402
from dynamics import ensemble_dmd  # noqa: E402
from control import canonicalize_eigenvector_phase, dominant_eigenmode  # noqa: E402
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference, permutation_pvalue, stable_seed,
)
from provenance import canonical_json, git_commit  # noqa: E402
from run_state_space_dimensionality_sweep import (  # noqa: E402
    CAUSAL_MICROSTIM_BIN_S, CAUSAL_MICROSTIM_GRAMIAN_HORIZON, CTG_N_SPLITS, CTG_STEP,
    DELIVERED_RANK, HUMAN_BIN_MS,
    claim_memorandum_subspace, claim_occupied_manifold,
    cross_validated_predictable_fraction, in_sample_linear_fraction,
    leave_one_out_cosine_deviation, component_direction,
    load_causal_microstim_dynamics_inputs, load_human_session_arrays,
    restatement_reduction_synthetic_check,
)
from run_state_space_estimation_admissibility import CANDIDATES, _flatten  # noqa: E402
from run_latent_model_comparison import anscombe_counts, counts_to_spiketrains  # noqa: E402
from run_macaque_pfc_microstimulation_pipeline import SESSIONS as CAUSAL_MICROSTIM_SESSIONS  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "state_space_estimation_robustness.json"
SWEEP_PATH = RESULTS / "state_space_dimensionality_sweep.json"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_state_space_estimation_robustness"
RUNG_THREE_CACHE_DIR = CHECKPOINT_DIR / "rung_three_latents"

# Operating point of every reduced candidate. The rank dimension of the question was settled by
# the dimensionality sweep; this leg holds rank at the project's delivered convention so the only
# thing varying across cells is the estimation choice itself.
OPERATING_RANK = DELIVERED_RANK

CTG_N_PERM_DEFAULT = 100
RESTATEMENT_N_PERM_DEFAULT = 100
SUBSPACE_COMPANION_N_PERM = 50

MAJORITY_SIGNIFICANCE_THRESHOLD = 0.5

RUNG_THREE_Z_FACTOR = 2.8016015201700604  # z(0.975) + z(0.8), the project's MDD convention
# Resource guard expressed in sessions, not a rationing cap: at the measured per-fit median below,
# 25 sessions cost about 118 s of wall clock -- negligible against the 900 s subprocess timeout.
RUNG_THREE_MAX_SESSIONS = 25
RUNG_THREE_CACHE_CEILING_BYTES = 1 << 30
RUNG_THREE_FREE_SPACE_FLOOR_BYTES = 5 << 30
RUNG_THREE_SEED_TAG = "stable_hash_sample_of_applicable_sessions"
# Measured median wall clock of 8 real fits through the actual subprocess worker, one per corpus
# this project uses; observed range 2.9-53.3 s.
MEDIAN_SEQUENTIAL_AUTOENCODER_FIT_COST_S = 4.7

LINEAR_KINDS = {"native_full_rank", "principal_components", "factor_analysis",
                "gaussian_process_factor_analysis"}
NO_TIME_AXIS_CANDIDATES = {"trial_level_variational_autoencoder"}
NONLINEAR_EMBEDDINGS = {"temporal_diffusion_embedding", "time_contrastive_embedding"}

STATUS_VOCABULARY = {
    "settled_robust": "verdict agreed across every admissible estimator run; the claim stands "
                      "robust to the estimation choice, with the range of effect sizes stated",
    "escalated": "verdict disagreed across estimators and the claim completed its escalation run "
                 "under the pre-declared session budget",
    "requires_refit_budget_decision": "verdict disagreed and the sizing rule's required sample "
                                      "exceeds this pass's session budget cap; a human decides "
                                      "whether to raise the cap and spend it",
    "pending_cached_fit_reuse": "waiting on a cached fit covering the needed split (none existed "
                                "on this leg: admissibility caches hold scores, not latents)",
    "inconclusive_below_detection_floor": "an escalation sample whose minimum detectable "
                                          "difference exceeds the effect it audits; both numbers "
                                          "reported, never read as agreement or disagreement",
    "escalation_attempt_failed_resource_limit": "the escalation run was triggered and started but "
                                                "its fits did not complete on this machine; "
                                                "recorded as a resource limit, never as a "
                                                "statement about the sessions",
    "escalation_undefined_for_claim_quantity": "the claim's quantity has no definition under the "
                                               "tier-three estimator's representation, for every "
                                               "session; no sample size is computed and no budget "
                                               "cap, however large, would make this claim's "
                                               "escalation computable",
    "escalation_exceeds_available_sessions": "the sizing rule's required sample size is within "
                                             "this pass's session budget cap but exceeds the "
                                             "number of sessions in the corpus that have a "
                                             "computable native reference cell to draw from; no "
                                             "larger sample without replacement is possible, so "
                                             "raising the cap further would not help",
}

HUMAN_CANDIDATES = ("native_full_rank", "principal_components", "factor_analysis",
                    "gaussian_process_factor_analysis", "trial_level_variational_autoencoder",
                    "temporal_diffusion_embedding", "time_contrastive_embedding")
MICROSTIM_CANDIDATES = ("native_full_rank", "principal_components", "factor_analysis",
                        "gaussian_process_factor_analysis", "temporal_diffusion_embedding",
                        "time_contrastive_embedding")


# ── Guards ────────────────────────────────────────────────────────────────────────

def fit_representation(candidate: str, X_3d: np.ndarray, k: int, rng: np.random.Generator,
                       is_spiking: bool, bin_ms: float, labels: np.ndarray | None = None) -> dict:
    """Single fitting entry point for every estimator this leg runs.

    Refuses a non-null ``labels`` argument outright: positive pairs for any embedding are defined
    by temporal adjacency and nothing else, so a label- or outcome-conditioned fit can never reach
    a claim cell through this path."""
    if labels is not None:
        raise ValueError(
            "fit_representation refuses a non-null label argument: no trial label, behavioural "
            "outcome, response or epoch identity may enter any representation-fitting objective "
            "on this leg; embeddings are time-contrastive only")
    out = CANDIDATES[candidate](X_3d, X_3d, k, rng, is_spiking, bin_ms)
    out["positive_pair_definition"] = (
        "temporal_adjacency_only_no_label_of_any_kind"
        if candidate == "time_contrastive_embedding"
        else "not_a_contrastive_model_unsupervised_fit")
    out["candidate"] = candidate
    return out


def require_linear_representation(candidate: str) -> None:
    """Refuses to build a subspace-angle projector on anything without a canonical orthonormal
    basis: a subspace angle inside a nonlinear embedding has no basis in which it is the same
    quantity, and the estimator-invariant restatement is the only route a linear-geometry claim
    may take through a nonlinear estimator."""
    if candidate in LINEAR_KINDS:
        return
    raise ValueError(
        f"subspace-angle projector requested on a non-linear representation (candidate="
        f"{candidate!r}); a subspace angle has no canonical basis outside a linear projection -- "
        "use the estimator-invariant restatement instead")


# ── Claim cells on precomputed representations ────────────────────────────────────

def ctg_content_permutation_null_on_latents(latent: np.ndarray, labels: np.ndarray,
                                            t_idx: np.ndarray, n_splits: int, n_perm: int,
                                            rng: np.random.Generator) -> dict:
    """Cross-temporal multiclass content decoding with a label-shuffle null, computed directly on
    precomputed per-timepoint coordinates. Identical fold structure, scorer and null to the
    delivered content-CTG routine, minus its internal per-fold projection step: the representation
    here is whatever estimator the cell audits, held fixed while only decoder labels permute."""
    from sklearn.model_selection import StratifiedKFold

    n_trials = latent.shape[0]
    labels = np.asarray(labels)
    all_classes = np.unique(labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=int(rng.integers(0, 1_000_000)))
    fold_data = [(latent[tr], labels[tr], latent[te], labels[te])
                 for tr, te in skf.split(np.zeros(n_trials), labels)]
    fold_mats = [_ctg_score_fold_multiclass(z_tr, y_tr, z_te, y_te, t_idx, all_classes)
                 for z_tr, y_tr, z_te, y_te in fold_data]
    auc_obs = np.nanmean(np.stack(fold_mats), axis=0)

    def _stat(mat: np.ndarray) -> float:
        n_t = mat.shape[0]
        return float(np.nanmean(mat[~np.eye(n_t, dtype=bool)] - 0.5))

    off_obs = _stat(auc_obs)
    null = np.zeros(n_perm)
    for p in range(n_perm):
        mats_p = [_ctg_score_fold_multiclass(z_tr, rng.permutation(y_tr), z_te,
                                             rng.permutation(y_te), t_idx, all_classes)
                  for z_tr, y_tr, z_te, y_te in fold_data]
        null[p] = _stat(np.nanmean(np.stack(mats_p), axis=0))
    diag = np.diag(auc_obs)
    tau_info = temporal_stability_tau(auc_obs)
    null_mean = float(np.mean(null))
    return {"status": "computed", "offdiag_auc_minus_chance": off_obs,
            "diag_auc_peak_minus_chance": float(np.nanmax(diag)),
            "tau": tau_info["tau"], "tau_interpretable": tau_info["interpretable"],
            "p_value": permutation_pvalue(null >= off_obs),
            "null_mean_offdiag_minus_chance": null_mean,
            "effect_size": float(off_obs - null_mean),
            "n_perm": int(n_perm), "n_trials": int(n_trials)}


def class_mean_coordinates(latent_trial: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
    """Orthonormal basis of the centred between-class mean structure inside any representation's
    coordinate space -- the coding-subspace structure expressed as regression coordinates rather
    than an ambient-angle projector, so one definition serves a linear projection and a nonlinear
    embedding alike."""
    classes = np.unique(labels)
    if len(classes) < 2:
        return None
    means = np.stack([latent_trial[labels == c].mean(axis=0) for c in classes])
    means_c = means - means.mean(axis=0)
    _, s, vt = np.linalg.svd(means_c, full_matrices=False)
    rank = int(np.sum(s > 1e-10))
    if rank < 1:
        return None
    return vt[:rank].T


def restated_claim_cell(y: np.ndarray, coords: np.ndarray | None, labels: np.ndarray,
                        latent_trial: np.ndarray, null_kind: str,
                        rng: np.random.Generator, n_perm: int) -> dict:
    """Estimator-invariant restatement cell for one linear-geometry claim: the cross-validated
    predictable fraction of the deviation component's variance from ``latent_trial @ coords``.

    ``null_kind='label_permutation'`` mirrors the coding-subspace claim's delivered null: the
    class-mean coordinate structure is rebuilt under permuted item labels while the deviation
    target stays fixed. ``null_kind='y_shuffle'`` mirrors the occupied-manifold claim's
    random-direction baseline: the trial-to-state correspondence is broken while the
    representation stays fixed."""
    if coords is None or coords.shape[1] < 1:
        return {"status": "not_computable", "reason": "no usable coordinate structure"}
    base_z = latent_trial @ coords
    observed = cross_validated_predictable_fraction(y, base_z, rng=rng)
    if observed.get("status") != "computed":
        return {"status": observed.get("status", "not_computable"),
                "reason": observed.get("reason")}
    null_vals = []
    for _ in range(n_perm):
        if null_kind == "label_permutation":
            null_coords = class_mean_coordinates(latent_trial, rng.permutation(labels))
            if null_coords is None:
                continue
            null_z, null_y = latent_trial @ null_coords, y
        elif null_kind == "y_shuffle":
            null_z, null_y = base_z, rng.permutation(y)
        else:
            raise ValueError(f"unknown null_kind {null_kind!r}")
        entry = cross_validated_predictable_fraction(null_y, null_z, rng=rng)
        if entry.get("status") == "computed":
            null_vals.append(entry["predictable_fraction"])
    if not null_vals:
        return {"status": "not_computable", "reason": "empty null"}
    frac = observed["predictable_fraction"]
    null_mean = float(np.mean(null_vals))
    return {"status": "computed", "predictable_fraction": frac, "null_mean": null_mean,
            "effect_size": float(frac - null_mean),
            "p_value": permutation_pvalue(np.asarray(null_vals) >= frac),
            "null_kind": null_kind, "n_null": len(null_vals),
            "null_values": [float(v) for v in null_vals],  # raw draws, for cross-level pooling only
            "n_dims": int(coords.shape[1]), "n_trials": int(len(y)),
            "in_sample_linear_beside": in_sample_linear_fraction(y, base_z)}


def subspace_angle_companion(X: np.ndarray, y_deviation: np.ndarray, w: np.ndarray,
                             labels: np.ndarray, rng: np.random.Generator) -> dict:
    """Delivered linear-geometry quantities for the two subspace claims, computed once per session
    at the operating rank, reported beside the restatement cells. The guard is called first so the
    angle path structurally cannot receive a nonlinear representation."""
    require_linear_representation("companion_check_native_basis")
    memo = claim_memorandum_subspace(X, labels, y_deviation, w, OPERATING_RANK, rng,
                                     SUBSPACE_COMPANION_N_PERM)
    occ = claim_occupied_manifold(X, w, OPERATING_RANK, rng, SUBSPACE_COMPANION_N_PERM)
    return {"memorandum_within_vs_null": memo, "occupied_within_vs_random": occ}


# ── Dynamics and control on precomputed latents (causal microstimulation corpus) ──

def fit_dynamics_from_latents(latent: np.ndarray, dt: float, rng: np.random.Generator,
                              r2_margin: float = 0.02) -> dict:
    """Ensemble-DMD plant fit directly on precomputed latent trajectories (n_trials, n_bins, k):
    the delivered retention-dynamics pipeline with its internal projection step replaced by the
    audited estimator's coordinates. Everything downstream -- cross-validated one-step R^2, the
    circular-shift null, the dominant-eigenmode classification -- is the delivered machinery."""
    n_trials, n_bins, k = latent.shape
    r_use = min(k, n_trials * (n_bins - 1) - 1)
    if r_use < 1:
        return {"status": "not_computable", "reason": "fewer snapshot pairs than latent dims"}
    ens = ensemble_dmd(latent, r=r_use, dt=dt, rng=rng)
    mode = dominant_eigenmode(ens["A"])
    eigs, vecs = np.linalg.eig(ens["A"])
    v_stable = canonicalize_eigenvector_phase(vecs[:, int(np.argmin(np.abs(eigs)))])
    identifiable = bool((ens["r2_cv"] - ens["r2_null"]) > r2_margin)
    return {"status": "computed", "rho": mode.rho, "theta": mode.theta,
            "classification": mode.classification, "r2_cv": ens["r2_cv"],
            "r2_null": ens["r2_null"], "identifiable": identifiable,
            "r_used": int(r_use), "n_trials": int(n_trials),
            "A": ens["A"], "v_star": mode.v_star, "v_stable": v_stable}


def gaussian_process_factor_analysis_with_loading(counts_3d: np.ndarray, k: int, bin_ms: float,
                                                  rng: np.random.Generator) -> dict:
    """Gaussian-process factor analysis fit returning BOTH the latent trajectories and the fitted
    channel loading of the same fitted model -- the loading is what expresses a channel-space
    input direction in these coordinates, and the alignment scoring is invariant to the latent
    basis rotation an EM refit would introduce, so latents and loading must come from one fit."""
    import quantities as pq
    from elephant.gpfa import GPFA
    n_features = counts_3d.shape[2]
    k_use = int(np.clip(k, 1, n_features - 1))
    try:
        spiketrains = counts_to_spiketrains(
            np.transpose(counts_3d, (0, 2, 1)).astype(int), bin_ms)
        gpfa = GPFA(bin_size=bin_ms * pq.ms, x_dim=k_use, em_max_iters=30, verbose=False)
        gpfa.fit(spiketrains)
        latent = np.stack(gpfa.transform(spiketrains), axis=0).transpose(0, 2, 1)
        loading = np.asarray(gpfa.params_estimated["C"])  # (n_features, k)
        return {"status": "fitted", "k_used": k_use, "latent_train": latent,
                "channel_loading": loading}
    except Exception as exc:
        return {"status": "failed_to_train", "reason": f"gaussian_process_factor_analysis "
                f"raised: {exc}"}


# ── Checkpointing (atomic temp-file-then-replace, the idiom every sibling uses) ───

def _checkpoint_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return CHECKPOINT_DIR / f"{safe}.json"


def load_my_checkpoint(key: str) -> dict | None:
    path = _checkpoint_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True:
        return None
    return data["record"]


def save_my_checkpoint(key: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(key)
    payload = {"_complete": True, "record": record}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def run_checkpointed(key: str, fit_fn) -> tuple[dict, bool]:
    cached = load_my_checkpoint(key)
    if cached is not None:
        return cached, True
    record = fit_fn()
    save_my_checkpoint(key, record)
    return record, False


def _free_disk_bytes(path: Path) -> int:
    return shutil.disk_usage(str(path)).free


def rung_three_cache_save(tag: str, latent: np.ndarray) -> dict:
    """Tier-three latents only, single precision, under the declared ceiling with a free-space
    floor checked before every write: a missing cache entry is a slow refit, whereas a full disk
    is a lost run. Tier-one and tier-two latents are never cached; they refit in seconds."""
    RUNG_THREE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(f.stat().st_size for f in RUNG_THREE_CACHE_DIR.glob("*.npz"))
    free = _free_disk_bytes(RUNG_THREE_CACHE_DIR)
    if total > RUNG_THREE_CACHE_CEILING_BYTES or free < RUNG_THREE_FREE_SPACE_FLOOR_BYTES:
        return {"cached": False, "reason": "cache_ceiling_or_free_space_floor_hit",
                "cache_bytes_so_far": int(total), "free_bytes": int(free)}
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
    path = RUNG_THREE_CACHE_DIR / f"{safe}.npz"
    fd, tmp_name = tempfile.mkstemp(dir=str(RUNG_THREE_CACHE_DIR), prefix="._tmp_", suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, latent=latent.astype(np.float32))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return {"cached": True, "path": str(path)}


# ── Human corpus runner (claims 1-3) ──────────────────────────────────────────────

def human_candidate_cells(entry: dict, candidate: str, args, cache_latents: bool = False) -> dict:
    key = f"{entry['dataset']}__{entry['session']}__{entry['structure']}"
    common = {"session_key": key, "candidate": candidate}
    arrays = load_human_session_arrays(entry)
    if arrays is None:
        return {**common, "status": "excluded",
                "reason": "fewer than 20 trials or 8 units after region filtering"}
    psth_rates = arrays["psth"]  # (n_trials, n_units, n_bins) spikes/s
    # Candidate inputs are integer spike counts: firing rate rescaled by the bin width recovers
    # exactly the histogram counts each rate encodes, restored here before any estimator sees them.
    counts = np.rint(np.transpose(psth_rates, (0, 2, 1)).astype(float)
                     * (HUMAN_BIN_MS / 1000.0))
    labels = np.asarray(arrays["item_ids"])
    X_rates = arrays["X_flat"]
    y_dev = leave_one_out_cosine_deviation(X_rates, labels)
    w = component_direction(X_rates, y_dev)
    rng = np.random.default_rng(stable_seed(f"robustness_human_{key}_{candidate}"))

    fit = fit_representation(candidate, counts, OPERATING_RANK, rng,
                             is_spiking=True, bin_ms=HUMAN_BIN_MS)
    if fit.get("status") != "fitted":
        return {**common, "status": "fit_failed",
                "reason": fit.get("reason", fit.get("status"))}
    latent = np.asarray(fit["latent_train"], dtype=float)
    cells = {**common, "status": "computed", "k_used": int(fit["k_used"]),
             "positive_pair_definition": fit["positive_pair_definition"],
             "n_trials": int(counts.shape[0]), "n_units": int(counts.shape[2])}
    if cache_latents:
        cells["tier_three_latent_cache"] = rung_three_cache_save(f"{key}__{candidate}", latent)

    if candidate in NO_TIME_AXIS_CANDIDATES:
        cells["cross_temporal_generalization"] = {
            "status": "not_applicable", "reason": "candidate pools over time, so "
            "cross-temporal generalisation is undefined in its coordinates"}
    else:
        t_idx = np.arange(0, latent.shape[1], CTG_STEP)
        cells["cross_temporal_generalization"] = ctg_content_permutation_null_on_latents(
            latent, labels, t_idx, CTG_N_SPLITS, args.n_perm_ctg, rng)

    latent_trial = latent.mean(axis=1)
    cells["memorandum_coding_subspace_vs_deviation"] = restated_claim_cell(
        y_dev, class_mean_coordinates(latent_trial, labels), labels, latent_trial,
        "label_permutation", rng, args.n_perm_restatement)
    cells["occupied_manifold_vs_deviation"] = restated_claim_cell(
        y_dev, np.eye(latent_trial.shape[1]), labels, latent_trial, "y_shuffle", rng,
        args.n_perm_restatement)

    if candidate in LINEAR_KINDS:
        try:
            require_linear_representation(candidate)
            cells["linear_geometry_companion"] = subspace_angle_companion(
                X_rates, y_dev, w, labels, rng)
        except ValueError as exc:
            cells["linear_geometry_companion"] = {"status": "refused_by_angle_guard",
                                                  "reason": str(exc)}
    else:
        cells["linear_geometry_companion"] = {
            "status": "refused_by_angle_guard",
            "reason": "nonlinear representation: a subspace angle has no canonical basis here; "
                      "the estimator-invariant restatement above is the only route"}
    return cells


# ── Causal microstimulation runner (claims 4-5) ───────────────────────────────────

def _exact_channel_loading(candidate: str, fit: dict, counts_3d: np.ndarray) -> np.ndarray | None:
    """(n_channels, k) matrix whose pseudo-inverse maps a channel-space direction into these
    coordinates, rebuilt exactly for the linear candidates whose delivered fitters expose the
    pieces."""
    k_used = int(fit["k_used"])
    flat, _, _ = _flatten(anscombe_counts(counts_3d))
    n_channels = counts_3d.shape[2]
    if candidate == "native_full_rank":
        return np.eye(n_channels)
    mu = flat.mean(axis=0)
    if candidate == "principal_components":
        _, _, vt = np.linalg.svd(flat - mu, full_matrices=False)
        return vt[:k_used].T
    if candidate == "factor_analysis":
        from sklearn.decomposition import FactorAnalysis
        fa = FactorAnalysis(n_components=k_used, random_state=0).fit(flat)
        return fa.components_.T
    if candidate == "gaussian_process_factor_analysis":
        loading = fit.get("channel_loading")
        if loading is None:
            return None
        loading = np.asarray(loading, dtype=float)
        if loading.shape[0] != n_channels and loading.shape[1] == n_channels:
            loading = loading.T
        return loading
    return None


def microstim_candidate_cells(prefix: str, candidate: str, args, cache_latents: bool = False) -> dict:
    key = f"causal_microstim__{prefix}"
    common = {"session_key": key, "candidate": candidate}
    inputs = load_causal_microstim_dynamics_inputs(prefix)
    if inputs is None:
        return {**common, "status": "excluded",
                "reason": "fewer than 10 control-condition correct trials, or no control "
                          "condition identified"}
    if not inputs["cond_info"]:
        return {**common, "status": "excluded",
                "reason": "no stimulation condition survived the channel filter"}
    counts_int = inputs["trials"].astype(float)          # (N, C, T) spike counts per bin
    counts_3d = np.transpose(counts_int, (0, 2, 1))      # (N, T, C) fitter interface
    if candidate in NO_TIME_AXIS_CANDIDATES:
        return {**common, "status": "excluded",
                "reason": "candidate pools over time, so it cannot support a trajectory fit"}
    rng = np.random.default_rng(stable_seed(f"robustness_causal_microstim_{key}_{candidate}"))
    srate = 1.0 / CAUSAL_MICROSTIM_BIN_S
    dt = max(1, int(round(srate / 50.0))) / srate

    if candidate == "gaussian_process_factor_analysis":
        fit = gaussian_process_factor_analysis_with_loading(
            counts_3d, OPERATING_RANK, CAUSAL_MICROSTIM_BIN_S * 1000.0, rng)
    else:
        fit = fit_representation(candidate, counts_3d, OPERATING_RANK, rng,
                                 is_spiking=True, bin_ms=CAUSAL_MICROSTIM_BIN_S * 1000.0)
    if fit.get("status") != "fitted":
        return {**common, "status": "fit_failed",
                "reason": fit.get("reason", fit.get("status"))}
    latent = np.asarray(fit["latent_train"], dtype=float)
    if cache_latents:
        cells_cache = rung_three_cache_save(f"{key}__{candidate}", latent)
    else:
        cells_cache = None

    dyn_rng = np.random.default_rng(stable_seed(f"{key}_{candidate}_dynamics"))
    dyn = fit_dynamics_from_latents(latent, dt, dyn_rng)
    cells = {**common, "status": "computed", "k_used": int(fit["k_used"]),
             "positive_pair_definition": fit.get("positive_pair_definition",
                                                 "unsupervised_generative_fit"),
             "n_trials": int(counts_3d.shape[0]), "n_channels": int(counts_3d.shape[2])}
    if dyn.get("status") == "computed":
        cells["fitted_dynamics"] = {k2: v for k2, v in dyn.items()
                                    if k2 not in ("A", "v_star", "v_stable")}
    else:
        cells["fitted_dynamics"] = dyn

    # Control-model cell: the input direction reaches these coordinates through the estimator's
    # own channel-to-latent map. Linear candidates contribute an exact pseudo-inverse loading.
    # The nonlinear embeddings expose no such map -- their coordinates are a nonlinear function
    # of channel space, the delivered fitters do not return a probe-able transform, and refitting
    # them outside their delivered entry point would put the input image in different coordinates
    # than the plant -- so their control cells are recorded with that reason rather than approximated.
    control_cell: dict
    if dyn.get("status") != "computed":
        control_cell = {"status": "not_computable", "reason": "plant fit failed for this cell"}
    elif candidate in NONLINEAR_EMBEDDINGS:
        control_cell = {"status": "not_computable",
                        "reason": "nonlinear embedding: no channel-to-latent map exists to "
                                  "express the stimulation-input direction in these coordinates; "
                                  "the delivered alignment scoring is defined on a linear map"}
    else:
        loading = _exact_channel_loading(candidate, fit, counts_3d)
        if loading is None:
            control_cell = {"status": "not_computable",
                            "reason": "channel loading unavailable for this fit"}
        else:
            # The delivered alignment scoring projects through components.T; handing it the
            # pseudo-inverse transpose makes that projection exactly the estimator's own
            # features-to-latents map applied to the input direction.
            fit_for_control = {"A": dyn["A"], "components": np.linalg.pinv(loading).T,
                               "v_star": dyn["v_star"], "v_stable": dyn["v_stable"]}
            ctrl_rng = np.random.default_rng(stable_seed(f"{key}_{candidate}_control"))
            control_cell = claim_control_model_delivered(fit_for_control, inputs["cond_info"],
                                                         ctrl_rng)
    cells["control_model"] = control_cell
    if cells_cache is not None:
        cells["tier_three_latent_cache"] = cells_cache
    return cells


def claim_control_model_delivered(fit: dict, cond_info: dict, rng: np.random.Generator) -> dict:
    """The delivered control-model scoring (controllability, stimulation-input alignment excess
    over random directions, energy-accuracy trade-off), imported unchanged from the sweep module
    and handed the audited estimator's operator and input-direction images."""
    from run_state_space_dimensionality_sweep import (
        CAUSAL_MICROSTIM_ENERGY_ACCURACY_Q, CAUSAL_MICROSTIM_GRAMIAN_HORIZON, claim_control_model,
    )
    return claim_control_model(fit, cond_info, rng)


# ── Aggregation, verdict keys, and the pre-declared escalation rules ──────────────

def aggregate_effect_cells(per_session: list[dict]) -> dict:
    effects = np.array([r["effect_size"] for r in per_session
                        if np.isfinite(r.get("effect_size", np.nan))])
    pvals = np.array([r["p_value"] for r in per_session
                      if np.isfinite(r.get("p_value", np.nan))])
    if len(effects) == 0:
        return {"status": "not_computable", "n_sessions": 0}
    agg = {"status": "computed", "n_sessions": int(len(effects)),
           "mean_effect_size": float(np.mean(effects)),
           "std_effect_size": float(np.std(effects, ddof=1)) if len(effects) > 1 else None,
           "effect_range": [float(np.min(effects)), float(np.max(effects))],
           "minimum_detectable_difference": (minimum_detectable_paired_difference(effects)
                                             if len(effects) >= 2
                                             else {"status": "not_computable"})}
    if len(pvals) > 0:
        agg["fraction_significant_p_below_0p05"] = float(np.mean(pvals < 0.05))
    else:
        agg["fraction_significant_p_below_0p05"] = None
        if len(effects) > 0:
            agg["fraction_significant_not_computable_reason"] = "no finite p-values in the cell data"
    return agg


def effect_cell_verdict_key(aggregate: dict) -> str:
    """Pre-declared verdict key for an effect-size cell: whether a majority of the cell's sessions
    clear the permutation null, and on which side the mean effect sits -- but a sign is only ever
    written into the key when it is resolvable. A cell that fails the majority-significance test AND
    whose mean effect does not clear its own minimum detectable difference has declared its mean
    unresolvable by both of the tests this leg runs on it; encoding the sign of that unresolved mean
    would make two estimators that both concluded "no effect" carry different verdict keys purely
    because an undetectable quantity happened to average out on opposite sides of zero. The sign is
    still reported on the cell -- it is simply excluded from the identity the escalation trigger
    compares."""
    if aggregate.get("status") != "computed":
        return "not_computable"
    frac = aggregate.get("fraction_significant_p_below_0p05")
    mean_effect = aggregate.get("mean_effect_size")
    side = "positive" if (mean_effect or 0.0) > 0 else "negative"
    if frac is not None and frac >= MAJORITY_SIGNIFICANCE_THRESHOLD:
        return f"majority_significant_{side}"
    mdd = aggregate.get("minimum_detectable_difference") or {}
    if (mdd.get("status") == "computed" and mean_effect is not None
            and abs(mean_effect) > mdd["mdd"]):
        return f"no_majority_effect_above_detection_floor_{side}"
    return "no_majority_effect"


def dynamics_cell_verdict_key(aggregate: dict) -> str:
    """Unchanged classification-vote key: the eigenmode class with the most session votes, or
    ``dominant_classification_mixed`` on a tie. A vote's decidability is a separate question from
    its key -- see ``dynamics_vote_resolvability`` -- because a 5/8 majority and a 4/4 tie both
    currently pass through here as distinguishable strings, but only the tie is caught by this
    function; a coin-flip 5/8 or 5/3 majority reads as confidently as a unanimous 8/8 vote unless a
    reader also checks the resolvability field."""
    counts = aggregate.get("classification_counts") or {}
    if not counts:
        return "not_computable"
    top = max(counts.values())
    winners = sorted(c for c, n in counts.items() if n == top)
    if len(winners) > 1:
        return "dominant_classification_mixed"
    return f"dominant_classification_{winners[0]}"


def dynamics_vote_resolvability(counts: dict[str, int]) -> dict:
    """Reports how decidable a classification vote is, beside (never instead of) its verdict key:
    the winning class's vote share and whether that share clears a strict majority. A 4/4 split on
    n=8 is a coin flip (winning_fraction 0.5, not decidable) even though the key above already
    separates it into ``dominant_classification_mixed``; a 5/8 majority is decidable by this
    criterion but only barely, and this field is what lets a reader see that distinction instead of
    reading every single-winner vote as equally confident."""
    n_total = sum(counts.values())
    if not counts or n_total == 0:
        return {"status": "not_computable"}
    top = max(counts.values())
    winners = sorted(c for c, n in counts.items() if n == top)
    winning_fraction = top / n_total
    return {"status": "computed", "n_votes": int(n_total), "winning_classes": winners,
            "winning_fraction": float(winning_fraction),
            "decidable": bool(len(winners) == 1 and winning_fraction > MAJORITY_SIGNIFICANCE_THRESHOLD)}


def control_cell_verdict_key(aggregate: dict) -> str:
    """Pre-declared verdict key for the targeting-alignment control cell. The delivered scoring
    runs no permutation null on this quantity (the alignment excess is a mean over stimulation
    conditions, not a per-session test statistic), so majority-significance is not an available
    resolvability test here the way it is for the effect cells above. The resolvability criterion
    this leg uses instead is the same paired minimum-detectable-difference already computed for
    this aggregate (``excess_minimum_detectable_difference``): a sign is written into the key only
    when the mean alignment excess exceeds its own across-session spread-derived detection floor.
    Below that floor the excess's sign is not distinguishable from zero at this design's power, and
    the key must not encode it -- otherwise two estimators that both failed to clear the floor would
    disagree purely on which side of zero their undetectable mean happened to fall."""
    excess = aggregate.get("alignment_excess_over_random")
    if excess is None:
        return "not_computable"
    mdd = aggregate.get("excess_minimum_detectable_difference") or {}
    if mdd.get("status") != "computed":
        return "targeting_alignment_excess_unresolved_no_spread_estimate"
    side = "positive" if excess > 0 else "negative"
    if abs(excess) > mdd["mdd"]:
        return f"targeting_alignment_excess_above_detection_floor_{side}"
    return "targeting_alignment_excess_unresolved"


def _pre_repair_effect_cell_verdict_key_for_audit_only(aggregate: dict) -> str:
    """Reproduces the verdict key this leg computed before the sign-of-an-unresolved-mean defect
    was repaired -- an unconditional sign suffix on every ``no_majority_effect`` cell. Used only to
    build the before/after audit trail recorded in the artifact; the live decision path never calls
    this function."""
    if aggregate.get("status") != "computed":
        return "not_computable"
    frac = aggregate.get("fraction_significant_p_below_0p05")
    side = "positive" if (aggregate.get("mean_effect_size") or 0.0) > 0 else "negative"
    if frac is not None and frac >= MAJORITY_SIGNIFICANCE_THRESHOLD:
        return f"majority_significant_{side}"
    return f"no_majority_effect_{side}"


def _pre_repair_control_cell_verdict_key_for_audit_only(aggregate: dict) -> str:
    """Reproduces the pre-repair control-cell key: an unconditional sign split on the alignment
    excess with no resolvability test at all. Audit trail only; not called by the live decision
    path."""
    excess = aggregate.get("alignment_excess_over_random")
    if excess is None:
        return "not_computable"
    return ("targeting_alignment_excess_positive" if excess > 0
            else "targeting_alignment_excess_negative")


def verdict_key_repair_audit(cell_key: str, estimator_cells: dict[str, dict],
                             repaired_branch: str) -> dict:
    """Recomputes what the pre-repair verdict keys and escalation decision would have been for one
    claim's already-built estimator aggregates, beside what the repaired keys actually decided, so
    the effect of the repair is auditable directly from the artifact rather than asserted in prose.
    Only the verdict-key function differs between the two columns; every effect size, p-value,
    significance fraction and minimum detectable difference feeding both is identical, because both
    read the same aggregate dicts already built under the repaired code."""
    if cell_key == "fitted_dynamics":
        return {"applicable": False,
                "reason": "the dynamics cell's key is a classification vote, not a sign split; "
                          "the defect this audit tracks never touched it"}
    pre_fn = (_pre_repair_control_cell_verdict_key_for_audit_only if cell_key == "control_model"
              else _pre_repair_effect_cell_verdict_key_for_audit_only)
    pre_keys = {name: pre_fn(agg) for name, agg in estimator_cells.items()}
    pre_decision = decide_claim_standing(pre_keys)
    return {"applicable": True, "pre_repair_verdict_keys": pre_keys,
            "pre_repair_branch": pre_decision["branch"],
            "repaired_verdict_keys": {name: agg.get("verdict_key")
                                      for name, agg in estimator_cells.items()},
            "repaired_branch": repaired_branch,
            "branch_changed_by_repair": pre_decision["branch"] != repaired_branch}


def decide_claim_standing(verdict_keys: dict[str, str]) -> dict:
    """Pre-declared escalation trigger: a claim whose verdict is stable across every admissible
    estimator is settled and is not escalated; a claim whose verdict differs between estimators
    escalates, that claim alone. Every disagreement names the estimators on each side, and each
    estimator cell carries its own effect size beside its verdict."""
    computable = {name: key for name, key in verdict_keys.items()
                  if key not in (None, "not_applicable", "not_computable")}
    if not computable:
        return {"branch": "no_computable_estimator_cells", "unique_verdict_keys": [],
                "estimators_by_side": {}}
    unique = sorted(set(computable.values()))
    if len(unique) == 1:
        return {"branch": "verdict_confirmed_across_estimators",
                "unique_verdict_keys": unique, "estimators_by_side": {}}
    sides: dict[str, list[str]] = {}
    for name, key in sorted(computable.items()):
        sides.setdefault(key, []).append(name)
    return {"branch": "estimation_dependent_rung_three_escalation",
            "unique_verdict_keys": unique, "estimators_by_side": sides}


def rung_three_sample_size(sd: float, effect: float) -> dict:
    """Smallest paired-sample size whose minimum detectable difference falls below the absolute
    tier-one effect size, at 80 percent power and alpha 0.05. Declared before any tier-three
    number exists; a requirement beyond this pass's budget cap is a human budget decision."""
    if not np.isfinite(sd) or sd <= 0 or not np.isfinite(effect) or abs(effect) <= 0:
        return {"status": "not_computable", "reason": "non-positive effect or non-finite sd"}
    n_needed = int(np.ceil((RUNG_THREE_Z_FACTOR * sd / abs(effect)) ** 2))
    feasible = n_needed <= RUNG_THREE_MAX_SESSIONS
    n_capped = min(max(n_needed, 1), RUNG_THREE_MAX_SESSIONS)
    return {"status": "computed", "n_required": n_needed, "feasible_within_budget": feasible,
            "budget_cap_sessions": RUNG_THREE_MAX_SESSIONS,
            "mdd_at_cap": float(RUNG_THREE_Z_FACTOR * sd / np.sqrt(n_capped)),
            "seed_tag": RUNG_THREE_SEED_TAG,
            "median_sequential_autoencoder_fit_cost_s": MEDIAN_SEQUENTIAL_AUTOENCODER_FIT_COST_S,
            "estimated_wall_clock_cost_s": float(n_needed * MEDIAN_SEQUENTIAL_AUTOENCODER_FIT_COST_S)}


def aggregate_claim(records: list[dict], cell_key: str) -> tuple[dict, str]:
    """Aggregate one claim's cells for one estimator across sessions, and return the aggregate
    beside its pre-declared verdict key. Each claim family aggregates its own primary quantity;
    every aggregate carries n, the effect summary and its spread."""
    nodes, dropped = [], []
    for rec in records:
        node = rec.get(cell_key, {})
        if isinstance(node, dict) and node.get("status") == "computed":
            nodes.append(node)
        elif isinstance(node, dict) and node.get("status") is not None:
            dropped.append({"session_key": rec.get("session_key"), "status": node.get("status"),
                            "reason": node.get("reason")})
        else:
            # No cell for this key at all (e.g. a record-level fit failure recorded before any
            # per-claim cell existed): report it dropped using the record's own status/reason
            # rather than let it vanish from both nodes and dropped.
            dropped.append({"session_key": rec.get("session_key"), "status": rec.get("status"),
                            "reason": rec.get("reason")})

    if cell_key == "fitted_dynamics":
        classification_counts: dict[str, int] = {}
        rhos, identifiable = [], []
        for node in nodes:
            classification_counts[node["classification"]] = (
                classification_counts.get(node["classification"], 0) + 1)
            rhos.append(float(node["rho"]))
            identifiable.append(bool(node["identifiable"]))
        agg = {"status": "computed" if nodes else "not_computable", "n_sessions": len(nodes),
               "classification_counts": classification_counts,
               "vote_resolvability": dynamics_vote_resolvability(classification_counts),
               "mean_rho": float(np.mean(rhos)) if rhos else None,
               "std_rho": float(np.std(rhos, ddof=1)) if len(rhos) > 1 else None,
               "fraction_identifiable": float(np.mean(identifiable)) if identifiable else None,
               "rho_minimum_detectable_difference": (
                   minimum_detectable_paired_difference(np.array(rhos)) if len(rhos) >= 2
                   else {"status": "not_computable"}),
               "per_session": [{"session_key": rec.get("session_key"), "rho": float(node["rho"]),
                                "classification": node["classification"]}
                               for rec, node in zip(records, [r.get(cell_key, {}) for r in records])
                               if node.get("status") == "computed"],
               "cells_not_computed": dropped}
        return agg, dynamics_cell_verdict_key(agg)

    if cell_key == "control_model":
        excesses, excess_pairs, slopes = [], [], []
        align_v, align_r = [], []
        for rec in records:
            node = rec.get(cell_key, {})
            if not isinstance(node, dict) or node.get("status") != "computed":
                continue
            ex_v, ex_r = [], []
            for cond in node.get("per_condition", {}).values():
                ex_v.append(cond["alignment_to_vstar"])
                ex_r.append(cond["random_direction_alignment"])
            if ex_v:
                exc = float(np.mean(ex_v) - np.mean(ex_r))
                excesses.append(exc)
                excess_pairs.append({"session_key": rec.get("session_key"),
                                     "alignment_excess": exc})
            align_v.extend(ex_v)
            align_r.extend(ex_r)
            ea = node.get("energy_accuracy") or {}
            if ea.get("energy_error_slope") is not None:
                slopes.append(ea["energy_error_slope"])
        agg = {"status": "computed" if excesses else "not_computable", "n_sessions": len(excesses),
               "alignment_excess_over_random": float(np.mean(excesses)) if excesses else None,
               "std_alignment_excess": (float(np.std(excesses, ddof=1)) if len(excesses) > 1
                                        else None),
               "alignment_excess_per_session": excesses,
               "excess_minimum_detectable_difference": (
                   minimum_detectable_paired_difference(np.array(excesses)) if len(excesses) >= 2
                   else {"status": "not_computable"}),
               "mean_alignment_to_vstar": float(np.mean(align_v)) if align_v else None,
               "mean_random_direction_alignment": (float(np.mean(align_r)) if align_r else None),
               "mean_energy_error_slope": float(np.mean(slopes)) if slopes else None,
               "per_session": excess_pairs,
               "cells_not_computed": dropped}
        return agg, control_cell_verdict_key(agg)

    computed_pairs = [(rec.get("session_key"), rec.get(cell_key, {})) for rec in records
                      if isinstance(rec.get(cell_key, {}), dict)
                      and rec.get(cell_key, {}).get("status") == "computed"]
    agg = aggregate_effect_cells([{"effect_size": node.get("effect_size", np.nan),
                                   "p_value": node.get("p_value", np.nan)}
                                  for _, node in computed_pairs])
    agg["per_session"] = [{"session_key": skey, "effect_size": node.get("effect_size"),
                           "predictable_fraction": node.get("predictable_fraction"),
                           "p_value": node.get("p_value"), "n_trials": node.get("n_trials")}
                          for skey, node in computed_pairs]
    agg["cells_not_computed"] = dropped
    return agg, effect_cell_verdict_key(agg)


def build_claim_block(records_by_candidate: dict[str, list[dict]], cell_key: str,
                      priority: int, metric_description: str) -> dict:
    estimators, verdict_keys = {}, {}
    for candidate, records in sorted(records_by_candidate.items()):
        agg, vkey = aggregate_claim(records, cell_key)
        agg["verdict_key"] = vkey
        estimators[candidate] = agg
        verdict_keys[candidate] = vkey
    standing = decide_claim_standing(verdict_keys)
    return {"priority": priority, "metric": metric_description,
            "estimator_cells": estimators, "agreement": standing}


def rung_three_sizing_preview(claims: dict) -> dict:
    """Sizes the tier-three sample for every claim the repaired keys leave escalated, without
    drawing a sample or running a single fit -- the same sizing rule ``rung_three_for_claims`` uses,
    computed here so its cost (a human budget decision on whether to spend ~5 hours per session) is
    visible in every artifact even when ``--run-rung-three`` was never passed."""
    preview: dict = {}
    for name, block in claims.items():
        if block["agreement"]["branch"] != "estimation_dependent_rung_three_escalation":
            continue
        if name == "control_model":
            preview[name] = {"corpus": "causal_microstimulation",
                             "reason": ("the claim's quantity has no definition under a nonlinear "
                                        "embedding -- there is no channel-to-latent map to express "
                                        "the stimulation-input direction in the tier-three "
                                        "estimator's coordinates -- so no sample size is computed; "
                                        "escalation here can never be resolved by tier three")}
            continue
        native_cell = block["estimator_cells"].get("native_full_rank", {})
        if native_cell.get("status") != "computed":
            preview[name] = {"reason": "no computable native reference cell to size the sample from"}
            continue
        if name == "fitted_dynamics":
            sd, effect, corpus = (native_cell.get("std_rho"), native_cell.get("mean_rho"),
                                  "causal_microstimulation")
        else:
            sd, effect, corpus = (native_cell.get("std_effect_size"),
                                  native_cell.get("mean_effect_size"), "human")
        sizing = rung_three_sample_size(sd if sd is not None else float("nan"),
                                        effect if effect is not None else float("nan"))
        preview[name] = {"corpus": corpus, "sample_sizing": sizing,
                         "n_applicable_sessions": len(native_cell.get("per_session", [])),
                         "would_require_a_budget_decision_if_run": not sizing.get(
                             "feasible_within_budget", False)}
    return preview


# ── Tier three (sequential autoencoder), only where tiers one and two disagree ────

def rung_three_for_claims(claims: dict, args, root: Path) -> dict:
    """For each estimation-dependent claim: size the sample from the native reference cell's
    spread and effect (declared before anything tier three is computed), draw it by stable hash,
    rerun the claim's cells under the sequential autoencoder, and classify the outcome. A claim
    whose required sample exceeds the budget cap is left for a human decision untouched; a claim
    whose required sample exceeds the corpus's own session count can never be drawn regardless of
    the cap; a claim whose quantity has no definition under the tier-three estimator is never
    computable at any cap. Each of these gets its own status string and reason, never the same
    label for different situations."""

    sa_name = "sequential_autoencoder"
    report: dict = {}
    human_entries = None
    for name, block in claims.items():
        if block["agreement"]["branch"] != "estimation_dependent_rung_three_escalation":
            continue
        native_cell = block["estimator_cells"].get("native_full_rank", {})
        if native_cell.get("status") != "computed":
            report[name] = {"status": "requires_refit_budget_decision",
                            "reason": "no computable native reference cell to size the sample from"}
            continue
        if name in ("cross_temporal_generalization", "memorandum_coding_subspace_vs_deviation",
                    "occupied_manifold_vs_deviation"):
            sd, effect = native_cell.get("std_effect_size"), native_cell.get("mean_effect_size")
            applicable = [row["session_key"] for row in native_cell.get("per_session", [])]
            corpus = "human"
        elif name == "control_model":
            report[name] = {"corpus": "causal_microstimulation", "status":
                            "escalation_undefined_for_claim_quantity",
                            "reason": ("the claim's quantity has no definition under a nonlinear "
                                       "embedding -- there is no channel-to-latent map to express "
                                       "the stimulation-input direction in the tier-three "
                                       "estimator's coordinates -- so spending the tier-three "
                                       "budget here cannot produce a computable cell")}
            continue
        else:
            sd = native_cell.get("std_rho")
            effect = native_cell.get("mean_rho")
            applicable = [row["session_key"] for row in native_cell.get("per_session", [])]
            corpus = "causal_microstimulation"
        sizing = rung_three_sample_size(sd if sd is not None else float("nan"),
                                        effect if effect is not None else float("nan"))
        entry: dict = {"corpus": corpus, "sample_sizing": sizing,
                       "n_applicable_sessions": len(applicable)}
        if not sizing.get("feasible_within_budget", False):
            entry["status"] = "requires_refit_budget_decision"
            entry["reason"] = (f"the sizing rule requires {sizing.get('n_required')} sessions, "
                               f"above this pass's budget cap of {RUNG_THREE_MAX_SESSIONS}; "
                               "sample_sizing carries the full numbers")
            report[name] = entry
            continue
        if sizing["n_required"] > len(applicable):
            entry["status"] = "escalation_exceeds_available_sessions"
            entry["reason"] = (f"the sizing rule requires {sizing['n_required']} sessions but "
                               f"only {len(applicable)} sessions in this corpus have a computable "
                               "native reference cell to draw from; no larger sample without "
                               "replacement is possible from this corpus")
            report[name] = entry
            continue
        order = sorted(applicable)
        pick_rng = np.random.default_rng(stable_seed(f"rung_three_{name}_"
                                                     + RUNG_THREE_SEED_TAG))
        sample = sorted(pick_rng.choice(order, size=sizing["n_required"], replace=False)
                        .tolist())
        entry["sampled_sessions"] = sample
        cells = []
        for sess in sample:
            ck = f"robustness__rungthree__{name}__{sess}__{sa_name}"
            if corpus == "human":
                if human_entries is None:
                    human_entries = {}
                    for e in iter_dandi_000469(root):
                        if e["structure"] == "pooled":
                            human_entries[f"{e['dataset']}__{e['session']}__{e['structure']}"] = e
                if sess not in human_entries:
                    cells.append({"session_key": sess, "status": "excluded",
                                  "reason": "session not produced by the shared loader"})
                    continue
                rec, _ = run_checkpointed(
                    ck, lambda e=human_entries[sess]: _human_cells_with_latent(e, sa_name, args))
            else:
                prefix = sess.split("causal_microstim__")[-1]
                rec, _ = run_checkpointed(
                    ck, lambda p=prefix: microstim_candidate_cells(p, sa_name, args,
                                                                   cache_latents=True))
            cells.append(rec)
        entry["cells"] = cells
        n_failed = sum(1 for c in cells if c.get("status") != "computed")
        if n_failed == len(cells) and cells:
            entry["status"] = "escalation_attempt_failed_resource_limit"
            entry["note"] = ("every sampled fit failed on this machine; reasons are recorded "
                             "verbatim per session above and read as a resource limit")
        elif name in ("cross_temporal_generalization", "memorandum_coding_subspace_vs_deviation",
                      "occupied_manifold_vs_deviation"):
            cell_data = [{"effect_size": c[name]["effect_size"],
                          "p_value": c[name].get("p_value", np.nan)}
                         for c in cells
                         if c.get("status") == "computed"
                         and np.isfinite(c.get(name, {}).get("effect_size", np.nan))]
            effects = np.array([d["effect_size"] for d in cell_data])
            mdd = (minimum_detectable_paired_difference(effects) if len(effects) >= 2
                   else {"status": "not_computable"})
            mdd_val = mdd.get("mdd") if isinstance(mdd, dict) else None
            if mdd_val is None or not np.isfinite(mdd_val) or mdd_val > abs(effect):
                entry["status"] = "inconclusive_below_detection_floor"
                entry["minimum_detectable_difference"] = mdd
                entry["tier_one_absolute_effect"] = abs(effect)
            else:
                entry["status"] = "escalated"
                entry["aggregate"] = aggregate_effect_cells(cell_data)
                entry["minimum_detectable_difference"] = mdd
        else:
            entry["aggregate"], _ = aggregate_claim(cells, name)
            if entry["aggregate"].get("status") != "computed":
                entry["status"] = "requires_refit_budget_decision"
                entry["reason"] = ("no computable sequential-autoencoder cell survived on the "
                                   "sampled sessions; per-session reasons above are verbatim")
            else:
                entry["status"] = "escalated"
        report[name] = entry
    return report


def _human_cells_with_latent(entry: dict, candidate: str, args) -> dict:
    """Tier-three variant of the human cell builder that also caches the fitted representation --
    the only tier whose latents cost enough to be worth caching."""
    return human_candidate_cells(entry, candidate, args, cache_latents=True)


# ── Main ──────────────────────────────────────────────────────────────────────────

HUMAN_CLAIM_SPECS = [
    ("cross_temporal_generalization", 1,
     "mean cross-temporal off-diagonal one-vs-rest AUC minus its 0.5 chance level, "
     "label-permutation null per session"),
    ("memorandum_coding_subspace_vs_deviation", 2,
     "cross-validated predictable fraction of the deviation component's variance from the "
     "representation's class-mean coordinate structure, minus its label-permutation null"),
    ("occupied_manifold_vs_deviation", 3,
     "cross-validated predictable fraction of the deviation component's variance from the full "
     "representation, minus its trial-shuffle null"),
]
MICRO_CLAIM_SPECS = [
    ("fitted_dynamics", 4,
     "dominant eigenmode classification of the fitted plant and its spectral decay radius, per "
     "estimator"),
    ("control_model", 5,
     "stimulation-input targeting alignment excess over the random-direction baseline, per "
     "estimator"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "human", "microstim"), default="all")
    parser.add_argument("--candidates", nargs="*", default=None,
                        help="restrict to these candidates (smoke tests)")
    parser.add_argument("--human-sessions-limit", type=int, default=None)
    parser.add_argument("--causal-microstim-sessions-limit", type=int, default=None)
    parser.add_argument("--n-perm-ctg", type=int, default=CTG_N_PERM_DEFAULT)
    parser.add_argument("--n-perm-restatement", type=int, default=RESTATEMENT_N_PERM_DEFAULT)
    parser.add_argument("--run-rung-three", action="store_true",
                        help="execute tier three for claims whose verdicts disagree, under the "
                             "pre-declared session budget")
    args = parser.parse_args()

    t0 = time.time()
    root = data_root()
    human_candidates = tuple(args.candidates) if args.candidates else HUMAN_CANDIDATES
    micro_candidates = tuple(args.candidates) if args.candidates else MICROSTIM_CANDIDATES

    sweep_ref = json.loads(SWEEP_PATH.read_text())
    anchors = {name: summ.get("full_rank_anchor")
               for name, summ in sweep_ref["claim_summaries"].items()}
    withdrawals = {name: bool(summ.get("withdrawn"))
                   for name, summ in sweep_ref["claim_summaries"].items()}

    # ---- Phase 1: human corpus -------------------------------------------------
    human_records: dict[str, list[dict]] = {}
    cache_hits, refits = 0, 0
    n_human_seen = 0
    if args.phase in ("all", "human"):
        for entry in iter_dandi_000469(root):
            if entry["structure"] != "pooled":
                continue
            if args.human_sessions_limit is not None and n_human_seen >= args.human_sessions_limit:
                break
            n_human_seen += 1
            key = f"{entry['dataset']}__{entry['session']}__{entry['structure']}"
            for candidate in human_candidates:
                rec, hit = run_checkpointed(
                    f"robustness__human__{key}__{candidate}",
                    lambda entry=entry, candidate=candidate: human_candidate_cells(
                        entry, candidate, args))
                cache_hits += int(hit)
                refits += int(not hit)
                human_records.setdefault(candidate, []).append(rec)
                print(f"[human] {key} {candidate}: {rec['status']}", file=sys.stderr)

    # ---- Phase 2: causal microstimulation corpus -------------------------------
    micro_records: dict[str, list[dict]] = {}
    prefixes = CAUSAL_MICROSTIM_SESSIONS
    micro_prefixes_used = prefixes
    if args.phase in ("all", "microstim"):
        micro_prefixes_used = (prefixes if args.causal_microstim_sessions_limit is None
                               else prefixes[: args.causal_microstim_sessions_limit])
        for prefix in micro_prefixes_used:
            for candidate in micro_candidates:
                rec, hit = run_checkpointed(
                    f"robustness__microstim__{prefix}__{candidate}",
                    lambda prefix=prefix, candidate=candidate: microstim_candidate_cells(
                        prefix, candidate, args))
                cache_hits += int(hit)
                refits += int(not hit)
                micro_records.setdefault(candidate, []).append(rec)
                print(f"[microstim] {prefix} {candidate}: {rec['status']}", file=sys.stderr)

    # ---- Phase 3: claim blocks, agreement branches, staged statuses ------------
    claims: dict[str, dict] = {}
    for cell_key, priority, metric in HUMAN_CLAIM_SPECS:
        if not human_records:
            continue
        claims[cell_key] = build_claim_block(human_records, cell_key, priority, metric)
    for cell_key, priority, metric in MICRO_CLAIM_SPECS:
        if not micro_records:
            continue
        claims[cell_key] = build_claim_block(micro_records, cell_key, priority, metric)

    for name, block in claims.items():
        branch = block["agreement"]["branch"]
        if branch == "verdict_confirmed_across_estimators":
            block["status"] = "settled_robust"
        elif branch == "estimation_dependent_rung_three_escalation":
            block["status"] = "requires_refit_budget_decision"
            block["status_note"] = ("disagreement across estimators triggers this claim alone; "
                                    "see rung_three for the pre-declared sizing and the budget "
                                    "decision it asks for")
        else:
            block["status"] = "inconclusive_below_detection_floor"
        # every HUMAN_CLAIM_SPECS / MICRO_CLAIM_SPECS entry uses its claim name as its cell_key
        block["verdict_key_repair_audit"] = verdict_key_repair_audit(
            name, block["estimator_cells"], branch)

    rung_three_sizing = rung_three_sizing_preview(claims)
    for name, preview in rung_three_sizing.items():
        claims[name]["rung_three_sizing_preview"] = preview

    # Sizing-only by default -- no fit is spent to populate this key; --run-rung-three below is the
    # only path that replaces it with an actually-executed escalation report.
    rung_three_report: dict = dict(rung_three_sizing)
    if args.run_rung_three:
        rung_three_report = rung_three_for_claims(claims, args, root)
        for name, entry in rung_three_report.items():
            if entry.get("status") == "escalated":
                claims[name]["status"] = "escalated"
            elif entry.get("status") in STATUS_VOCABULARY:
                claims[name]["status"] = entry["status"]
            claims[name]["rung_three"] = entry

    verification_rng = np.random.default_rng(stable_seed("robustness_restatement_reduction_check"))

    def zero_drop(records_by_candidate: dict[str, list[dict]]) -> dict:
        per_candidate = {}
        for candidate, recs in sorted(records_by_candidate.items()):
            statuses: dict[str, int] = {}
            exclusions = []
            for rec in recs:
                st = rec.get("status", "missing")
                statuses[st] = statuses.get(st, 0) + 1
                if st in ("excluded", "fit_failed"):
                    exclusions.append({"session_key": rec.get("session_key"),
                                       "reason": rec.get("reason")})
            per_candidate[candidate] = {
                "n_seen": len(recs), "statuses": statuses,
                "seen_equals_tested_plus_excluded": len(recs) == sum(statuses.values()),
                "exclusions_with_reasons": exclusions}
        return per_candidate

    output = {
        "version": "state_space_estimation_robustness_v1",
        "code_commit": git_commit(ROOT),
        "question": ("whether each rank-withdrawn claim's verdict reflects the projection step or "
                     "the data: every claim recomputed at one operating point under each "
                     "admissible estimation choice"),
        "audit_character": ("a robustness audit, never a comparison between estimators: the only "
                            "question is whether the project's conclusions depend on how the "
                            "state space was estimated"),
        "pre_declared_rules": {
            "operating_rank": OPERATING_RANK,
            "verdict_key_rule": ("an effect cell's verdict key is majority-significance at p<0.05 "
                                 f"across that cell's sessions, threshold "
                                 f"{MAJORITY_SIGNIFICANCE_THRESHOLD}; the sign of the mean effect "
                                 "enters the key only where it is resolvable -- majority-"
                                 "significant, or the mean effect exceeds this same aggregate's own "
                                 "paired minimum detectable difference -- otherwise the key reads "
                                 "'no_majority_effect' with no sign encoded, because a sign the "
                                 "cell's own tests could not distinguish from zero is not a verdict "
                                 "the escalation trigger may treat as disagreeing with another such "
                                 "cell; the dynamics cell's key is its dominant eigenmode "
                                 "classification, unchanged, with a separate vote_resolvability "
                                 "field reporting the winning class's vote share so a coin-flip "
                                 "majority is distinguishable from a lopsided one; the control "
                                 "cell's key is the sign of the targeting-alignment excess over its "
                                 "random-direction baseline, entered only where that excess exceeds "
                                 "this same aggregate's own paired minimum detectable difference "
                                 "(this cell carries no permutation null of its own, so the paired "
                                 "spread-derived floor is the only resolvability test available to "
                                 "it), otherwise the key reads "
                                 "'targeting_alignment_excess_unresolved'"),
            "verdict_key_defect_repaired": (
                "the delivered verdict keys for effect cells and the control cell encoded the sign "
                "of the mean effect unconditionally, so two estimators that both concluded the "
                "effect was statistically indistinguishable from zero could receive different "
                "verdict keys purely because the sign of that indistinguishable mean differed, "
                "which the escalation trigger then read as a genuine disagreement; the repair above "
                "makes the sign part of the key only when a resolvability test says the sign is "
                "distinguishable from zero. This did not touch the escalation trigger itself, the "
                "operating rank, the fitted effect sizes, p-values, significance fractions or "
                "minimum detectable differences, which are identical before and after the repair; "
                "only the verdict key built from them, and the branch decided from that key, "
                "changed"),
            "agreement_branches": ["verdict_confirmed_across_estimators",
                                   "estimation_dependent_rung_three_escalation",
                                   "no_computable_estimator_cells"],
            "escalation_semantics": ("a claim whose verdict holds across every estimator is "
                                     "settled and is not escalated; a claim whose verdict differs "
                                     "escalates, that claim alone, to the sequential autoencoder "
                                     "on a seeded sample sized by the native reference cell's "
                                     "spread and effect size"),
            "circularity_guard": ("the shared fitting entry point refuses a non-null label "
                                  "argument; embeddings on this leg are time-contrastive only, "
                                  "positive pairs temporal adjacency and nothing else"),
            "subspace_angle_guard": ("subspace angles are refused on every representation without "
                                     "a canonical orthonormal basis; the linear-geometry claims "
                                     "reach nonlinear estimators only through the "
                                     "estimator-invariant restatement"),
            "restatement_definition": ("cross-validated predictable fraction of the deviation "
                                       "component's variance, ridge read-out, held-out "
                                       "predictions concatenated across five folds, identical "
                                       "for projection and embedding; coding-subspace null "
                                       "rebuilds class-mean coordinates under permuted item "
                                       "labels, occupied-manifold null shuffles the "
                                       "trial-to-state correspondence"),
            "status_vocabulary": STATUS_VOCABULARY,
            "rung_three_budget": {
                "sample_size_rule": ("smallest n whose minimum detectable paired difference "
                                     f"({RUNG_THREE_Z_FACTOR:.4f} * sd / sqrt(n)) falls below the "
                                     "absolute tier-one effect size, declared before any "
                                     "tier-three number exists"),
                "max_sessions_this_pass": RUNG_THREE_MAX_SESSIONS,
                "cache_ceiling_bytes": RUNG_THREE_CACHE_CEILING_BYTES,
                "free_space_floor_bytes": RUNG_THREE_FREE_SPACE_FLOOR_BYTES,
                "latents_cached": "tier three only, single precision, ceiling and free-space "
                                  "floor checked before every write",
            },
        },
        "scope": {
            "human_corpus": ("dandi_000469, structure='pooled', memorandum content label = "
                             "encoded item identity, delay epoch only"),
            "causal_microstimulation_corpus": ("control-condition correct trials for the plant "
                                               "fit, every stimulation condition surviving the "
                                               "channel filter for the control model"),
            "n_human_sessions_seen": int(n_human_seen),
            "n_causal_microstimulation_sessions_seen": int(len(micro_prefixes_used)),
            "candidates_human": list(human_candidates),
            "candidates_causal_microstimulation": list(micro_candidates),
            "n_perm_ctg": args.n_perm_ctg,
            "n_perm_restatement": args.n_perm_restatement,
            "subspace_companion_n_perm": SUBSPACE_COMPANION_N_PERM,
            "seed_scheme": ("stable_seed('robustness_<corpus>_<session_key>_<candidate>[_role]'); "
                            "tier-three samples stable_seed('rung_three_<claim>_"
                            + RUNG_THREE_SEED_TAG + "')"),
            "cache_hits_vs_refits": {"checkpoint_hits": int(cache_hits), "fresh_fits": int(refits)},
            "budget_decisions": {
                "tier_one_and_two_refits": ("every fit on this leg is a fresh fit: the "
                                            "admissibility checkpoints hold scores, not fitted "
                                            "representations, so no latent-level reuse is "
                                            "possible at these tiers; nothing was refit "
                                            "unnecessarily because nothing could be reused"),
                "temporal_diffusion_embedding_cost": ("an earlier round carried an estimate near "
                                                      "94 percent of a 57-minute-per-session fit "
                                                      "for this candidate; the measured median "
                                                      "across the seven-corporum admissibility "
                                                      "run is seconds per fit, so it runs as an "
                                                      "ordinary affordable tier-one fit with no "
                                                      "special handling"),
                "sequential_autoencoder": ("hours per session fit on this machine through its "
                                           "isolated-environment bridge; reserved for escalated "
                                           "claims only, under the pre-declared sample-size "
                                           "rule and this pass's session cap"),
                "nonlinear_control_cells": ("the control-model claim's cells under the two "
                                            "nonlinear embeddings are recorded as not computable: "
                                            "their coordinates have no channel-to-latent map, "
                                            "which the delivered alignment scoring requires, and "
                                            "approximating one would place the input direction "
                                            "in coordinates other than the fitted plant's"),
            },
            "wall_clock_s": None,
        },
        "withdrawn_claims_under_audit": withdrawals,
        "full_rank_anchors_read_live_from_the_sweep_artifact": anchors,
        "estimator_invariant_restatement_verification": {
            "function": "cross_validated_predictable_fraction(y, Z, n_splits=5, alpha=1.0)",
            "reduction_property": ("in the noiseless linear case the restatement reduces to the "
                                   "delivered subspace-projection quantity; verified numerically "
                                   "below and asserted in the test suite"),
            "synthetic_linear_reduction": restatement_reduction_synthetic_check(verification_rng),
        },
        "zero_drop_accounting": {
            "human": zero_drop(human_records),
            "causal_microstimulation": zero_drop(micro_records),
        },
        "claims": claims,
        "rung_three": rung_three_report,
        "resume_state": {
            "checkpoint_dir": str(CHECKPOINT_DIR),
            "resume_instruction": ("rerun the same command; every (corpus, session, candidate) "
                                   "cell is checkpointed atomically and completed cells are "
                                   "skipped, so a partial pass resumes where it stopped"),
            "phases_completed": {"human": bool(human_records),
                                 "causal_microstimulation": bool(micro_records)},
        },
        "wall_clock_s": None,
    }
    output["scope"]["wall_clock_s"] = time.time() - t0
    output["wall_clock_s"] = time.time() - t0
    OUTPUT_PATH.write_text(canonical_json(output))
    print(f"Wrote {OUTPUT_PATH} in {output['wall_clock_s']:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
