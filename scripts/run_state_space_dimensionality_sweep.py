"""Dimensionality-sweep robustness audit for latent-rank-dependent geometry and dynamics claims.

Several of this project's structural conclusions pass through a chosen linear latent rank before
they are ever tested: the memorandum's coding subspace and whether an accuracy-relevant deviation
direction sits outside it, the occupied neural manifold and the same deviation direction's
decomposition against it, cross-temporal generalisation decoding, the fitted linear dynamics
(dominant eigenmode, decay rate), and the linear control model (controllability, stimulation-input
alignment, the energy-accuracy trade-off). This module recomputes each of those five claim families
across a declared range of ranks, on real recordings, and reports the verdict AS A FUNCTION OF RANK
rather than at one convention value -- so that a conclusion which only holds at a single rank is
visible as such rather than silently reported as if it were rank-independent.

This is a robustness audit, not a comparison between latent-estimation methods: nothing here ranks
estimators, and every rank of the same linear estimator is reported on equal footing.

Two corpora, chosen because each already carries the raw ingredients a rank sweep needs without
requiring a new loader:

  - the single-item human intracranial corpus (structure "pooled", one session per patient,
    ``corpus_sessions.iter_dandi_000469``) supplies real single-trial spike counts with a memorandum
    content label (the encoded item identity), used for the coding-subspace, occupied-manifold and
    cross-temporal-generalisation claims;
  - the causal microstimulation macaque corpus (loaded via scripts/run_macaque_pfc_microstimulation_pipeline.py's
    per-session trial loader, the same eleven sessions and channel/trial loading the delivered
    manifold-constraint analysis reuses) supplies a fitted linear plant with a genuine stimulation
    input direction, used for the fitted-dynamics and control-model claims.

A single trial-specific deviation score anchors the first two claim families: each trial's L2-
normalised feature vector against the L2-normalised leave-one-out mean of its own condition, one
minus their cosine similarity. This scalar needs no latent rank at all -- it is computed on the
native trial-by-feature array -- and stands in as a concrete instance of an accuracy-relevant
deviation whose relationship to the coding subspace and the occupied manifold is under audit; a
ridge-regression direction fit to explain it in feature space gives the two subspace claims a
concrete axis to test.

This sweep covers linear latent estimation only (principal-component representations at every rank
in the grid, plus the native full-rank reference). Recomputing the same claims under nonlinear
latent estimators is a separate, later piece of work and is deliberately not attempted here.

Every rank-dependent quantity below is produced by one fitted-representation function,
``fit_linear_representation``, that takes a trial-by-feature array and a rank and returns PCA scores
and components. Every claim's verdict function consumes only that returned dict's ``scores`` and
``components`` fields, never the fitting call itself -- so a future round can swap in a differently
fitted (including nonlinear) representation of the same shape without touching a single claim
function.

Part (b), the estimator-invariant restatement required before any nonlinear latent estimator may be
used on these claims: ``cross_validated_predictable_fraction`` computes the cross-validated fraction
of a scalar's variance that a linear or nonlinear representation's coordinates can predict. It is
computed here only in the linear case (PCA coordinates), beside the same quantity's in-sample
(non-cross-validated) counterpart -- the naive quantity this project's existing subspace-projection
metrics are instances of. ``tests/test_state_space_dimensionality_sweep.py`` proves numerically that
the two coincide in a noiseless synthetic linear scenario, which is the sense in which the
cross-validated restatement reduces to the delivered linear quantity.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_state_space_dimensionality_sweep.py [--n-perm N] [--sessions-limit N]
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import data_root, iter_dandi_000469  # noqa: E402
from spike_pipeline import build_psth  # noqa: E402
from geometry import (  # noqa: E402
    ctg_content_permutation_null, parallel_analysis, participation_ratio,
    pca_decompose, temporal_stability_tau,
)
from dynamics import fit_retention_dynamics  # noqa: E402
from control import energy_accuracy_pareto, stimulation_input_alignment  # noqa: E402
from statistics import minimum_detectable_paired_difference, permutation_pvalue, stable_seed  # noqa: E402
from provenance import canonical_json  # noqa: E402
from run_macaque_pfc_microstimulation_pipeline import SESSIONS as CAUSAL_MICROSTIM_SESSIONS  # noqa: E402
from run_macaque_pfc_microstimulation_pipeline import BIN_S as CAUSAL_MICROSTIM_BIN_S  # noqa: E402
from run_macaque_pfc_microstimulation_pipeline import crop_trial  # noqa: E402
from run_macaque_pfc_microstimulation_pipeline import load_macaque_pfc_microstimulation_session as load_causal_microstim_session  # noqa: E402

from sklearn.decomposition import FactorAnalysis  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "state_space_dimensionality_sweep.json"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_state_space_dimensionality_sweep"
SHARD_VARIABLE = "WM_DYNAMICS_SESSION_SHARD"

# The rank grid every claim is swept across. 8 is the rank this project's linear pipeline ships
# with (ctg_nested_cv's default n_components, run_macaque_pfc_microstimulation_pipeline.N_PC, dpca_condition_subspace_
# projection's typical use) -- called out explicitly below as DELIVERED_RANK so every table can
# mark where it sits relative to the principled selectors.
RANK_GRID = (2, 3, 4, 6, 8, 12, 16, 24)
DELIVERED_RANK = 8

HUMAN_BIN_MS = 200.0  # coarser than the project's publication convention (100 ms) so the CTG
                       # permutation null, refit at every rank in the grid, stays within budget;
                       # a robustness sweep needs relative comparability across rank, not
                       # publication time resolution.
HUMAN_DELAY_WINDOW_S = 2.3
CTG_STEP = 2
CTG_N_SPLITS = 4
CTG_N_PERM_DEFAULT = 100
SUBSPACE_N_PERM_DEFAULT = 200
CV_FOLDS = 5

CAUSAL_MICROSTIM_ENERGY_ACCURACY_Q = (0.01, 0.1, 1.0, 10.0, 100.0)
CAUSAL_MICROSTIM_GRAMIAN_HORIZON = 20


# ── Checkpointing (atomic, per-session, temp-file-then-replace; same idiom as
#    run_state_space_estimation_admissibility.py's checkpoint helpers) ──────────

def _checkpoint_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return CHECKPOINT_DIR / f"{safe}.json"


def load_checkpoint(key: str) -> dict | None:
    path = _checkpoint_path(key)
    if not path.exists():
        return None
    try:
        import json
        data = json.loads(path.read_text())
    except (ValueError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True:
        return None
    return data["record"]


def save_checkpoint(key: str, record: dict) -> None:
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


def run_checkpointed(key: str, fit_fn) -> dict:
    cached = load_checkpoint(key)
    if cached is not None:
        return cached
    record = fit_fn()
    save_checkpoint(key, record)
    return record


# ── Shared representation, deviation score, and predictable-fraction machinery ──

def fit_linear_representation(X: np.ndarray, rank: int, labels: np.ndarray | None = None) -> dict:
    """PCA representation of an (n_trials, n_features) array at the given rank.

    The one function every claim's verdict routine consumes: swapping this for a differently fitted
    (including nonlinear) representation of the same ``scores``/``components`` shape changes no
    downstream claim code. That is exactly why the label refusal belongs here rather than only in
    prose -- this is the single fitting entry point a later nonlinear embedding is reached through,
    and any embedding trained with a label or behavioural outcome as its fitting objective is
    circular for every behaviour claim that consumes it. ``labels`` exists only so that passing one
    raises; this project forbids label- or behaviour-conditioned representation fitting everywhere.
    """
    if labels is not None:
        raise ValueError(
            "fit_linear_representation refuses a non-null label argument -- label- or "
            "behaviour-conditioned representation fitting is forbidden project-wide for any "
            "behaviour claim")
    rank_eff = int(max(1, min(rank, X.shape[0] - 1, X.shape[1])))
    scores, components, var_ratio = pca_decompose(X, rank_eff)
    return {"kind": "pca", "rank": rank_eff, "scores": scores, "components": components,
            "var_ratio": var_ratio, "fitting_objective": "unsupervised_pca_reconstruction_no_labels"}


def _require_linear_representation(representation: dict) -> None:
    """A subspace-angle projector (components @ components.T) has no meaning without a canonical
    orthonormal basis, which a nonlinear embedding does not have. Every code path that builds such a
    projector must call this first, so a future nonlinear representation cannot be fed to it by
    accident -- the estimator-invariant restatement (``cross_validated_predictable_fraction``) is the
    only route to a claim's verdict once the representation is nonlinear."""
    if representation.get("kind") != "pca":
        raise ValueError(
            f"subspace-angle projector requested on a non-linear representation (kind="
            f"{representation.get('kind')!r}); a subspace angle has no canonical basis in a "
            "nonlinear embedding -- use cross_validated_predictable_fraction instead")


def cross_validated_predictable_fraction(y: np.ndarray, Z: np.ndarray, n_splits: int = CV_FOLDS,
                                          alpha: float = 1.0, rng: np.random.Generator | None = None) -> dict:
    """Cross-validated fraction of scalar ``y``'s variance predictable by ridge regression on
    representation coordinates ``Z``.

    Defined identically whether ``Z`` came from a linear projection or a nonlinear embedding -- ridge
    regression from a fixed set of per-trial coordinates onto a scalar target makes no reference to
    how those coordinates were produced. Held-out predictions are concatenated across folds before
    the single R^2 is computed, rather than averaging per-fold R^2, so a fold with little residual
    variance cannot dominate the summary.
    """
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    y, Z = y[mask], Z[mask]
    n = len(y)
    if n < max(6, n_splits + 1):
        return {"status": "not_computable", "reason": "fewer trials than folds require", "n_trials": n}
    n_splits_eff = min(n_splits, n)
    kf = KFold(n_splits=n_splits_eff, shuffle=True,
               random_state=int(rng.integers(0, 2**31 - 1)) if rng is not None else 0)
    y_true_held, y_pred_held = [], []
    for train_idx, test_idx in kf.split(Z):
        model = Ridge(alpha=alpha)
        model.fit(Z[train_idx], y[train_idx])
        y_pred_held.append(model.predict(Z[test_idx]))
        y_true_held.append(y[test_idx])
    y_true_held = np.concatenate(y_true_held)
    y_pred_held = np.concatenate(y_pred_held)
    ss_res = float(np.sum((y_true_held - y_pred_held) ** 2))
    ss_tot = float(np.sum((y_true_held - y_true_held.mean()) ** 2))
    fraction = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"status": "computed", "predictable_fraction": fraction, "n_trials": n,
            "n_dims": int(Z.shape[1]), "alpha": alpha, "n_splits": n_splits_eff}


def predictable_fraction_restatement(y: np.ndarray, X: np.ndarray, representations: dict[str, np.ndarray | None],
                                      rng: np.random.Generator) -> dict:
    """The estimator-invariant restatement of every "X lies outside subspace Y" claim in this module,
    as one function: for each named representation's coordinate basis, the cross-validated fraction
    of scalar ``y``'s variance predictable from ``X @ basis``, reported beside its in-sample
    (non-cross-validated) counterpart. ``representations`` maps a claim name to an (n_features, k)
    coordinate basis or ``None`` if that representation is not computable for this session; the
    basis here is always a PCA loading matrix, but the function makes no reference to how it was
    fitted -- swapping in a differently fitted (including nonlinear per-trial) coordinate matrix for
    any entry is a matter of what gets passed in, not a rewrite of this function."""
    out = {}
    for name, basis in representations.items():
        if basis is None:
            out[name] = {"status": "not_computable", "reason": "representation not available for this session"}
            continue
        Z = X @ basis
        out[name] = {"cross_validated": cross_validated_predictable_fraction(y, Z, rng=rng),
                      "in_sample_linear": in_sample_linear_fraction(y, Z)}
    return out


def in_sample_linear_fraction(y: np.ndarray, Z: np.ndarray) -> dict:
    """Non-cross-validated (in-sample, ordinary-least-squares) fraction of ``y``'s variance
    explained by ``Z`` -- the naive linear quantity this project's existing subspace-projection
    metrics are instances of, reported beside the cross-validated version for comparison."""
    y = np.asarray(y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    y, Z = y[mask], Z[mask]
    n = len(y)
    if n < Z.shape[1] + 2:
        return {"status": "not_computable", "n_trials": n}
    Zc = Z - Z.mean(axis=0)
    yc = y - y.mean()
    coef, *_ = np.linalg.lstsq(Zc, yc, rcond=None)
    fitted = Zc @ coef
    ss_res = float(np.sum((yc - fitted) ** 2))
    ss_tot = float(np.sum(yc ** 2))
    fraction = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return {"status": "computed", "linear_fraction": fraction, "n_trials": n, "n_dims": int(Z.shape[1])}


def restatement_reduction_synthetic_check(rng: np.random.Generator) -> dict:
    """Numeric verification, run as part of this artifact rather than only in the test suite, that
    ``cross_validated_predictable_fraction`` and ``in_sample_linear_fraction`` reduce to the delivered
    subspace-projection quantity ``(||P_S w|| / ||w||) ** 2`` in a noiseless linear scenario: ``y = X @
    w`` exactly, ``X`` isotropic, ``S`` an orthonormal basis unrelated to ``w``. Reported with the
    exact discrepancy numbers, not only a pass/fail assertion."""
    n, d, k = 4000, 10, 3
    X = rng.standard_normal((n, d))
    S, _ = np.linalg.qr(rng.standard_normal((d, k)))
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    y = X @ w
    Z = X @ S
    cv = cross_validated_predictable_fraction(y, Z, alpha=1e-6, rng=rng)
    naive = in_sample_linear_fraction(y, Z)
    P = S @ S.T
    delivered = float((np.linalg.norm(P @ w) / np.linalg.norm(w)) ** 2)
    return {
        "scenario": "noiseless linear generator y = X @ w, isotropic X (n=4000, d=10), rank-3 "
                    "orthonormal basis S unrelated to w, delivered quantity = (||P_S w||/||w||)**2",
        "delivered_subspace_quantity": delivered,
        "cross_validated_predictable_fraction": cv["predictable_fraction"],
        "in_sample_linear_fraction": naive["linear_fraction"],
        "discrepancy_cross_validated_vs_subspace": abs(cv["predictable_fraction"] - delivered),
        "discrepancy_in_sample_vs_subspace": abs(naive["linear_fraction"] - delivered),
    }


def restatement_reduction_on_delivered_human_data(human_sessions: list[dict]) -> dict:
    """The same comparison as ``restatement_reduction_synthetic_check``, computed on the real
    per-session numbers ``run_human_session`` already produced for this corpus, at the delivered rank,
    rather than on synthetic data. Real recordings satisfy neither the isotropic-design nor the
    noiseless-generator precondition the synthetic reduction proof above relies on, so exact numeric
    coincidence is not expected here -- this records the actual computed numbers per session rather
    than assuming agreement."""
    rows = []
    for session in human_sessions:
        if session.get("status") != "tested":
            continue
        rank_key = str(min(DELIVERED_RANK, session["full_rank"]))
        claim2 = session.get("per_rank", {}).get(rank_key, {}).get("occupied_manifold_vs_deviation", {})
        restatement = session.get("predictable_fraction_restatement", {}).get("state_representation", {})
        if claim2.get("status") != "computed" or restatement.get("in_sample_linear", {}).get("status") != "computed":
            continue
        within_frac_sq = claim2["within_frac"] ** 2
        in_sample = restatement["in_sample_linear"]["linear_fraction"]
        cv_entry = restatement.get("cross_validated", {})
        cv_frac = cv_entry.get("predictable_fraction") if cv_entry.get("status") == "computed" else None
        rows.append({
            "session_key": session["session_key"], "rank": int(rank_key),
            "delivered_subspace_quantity_within_frac_squared": within_frac_sq,
            "in_sample_linear_fraction": in_sample,
            "cross_validated_predictable_fraction": cv_frac,
            "discrepancy_in_sample_vs_subspace": abs(in_sample - within_frac_sq),
            "discrepancy_cross_validated_vs_subspace": (
                abs(cv_frac - within_frac_sq) if cv_frac is not None else None),
        })
    discrepancies = [r["discrepancy_in_sample_vs_subspace"] for r in rows]
    return {
        "rank": DELIVERED_RANK,
        "per_session": rows,
        "n_sessions": len(rows),
        "mean_discrepancy_in_sample_vs_subspace": float(np.mean(discrepancies)) if discrepancies else None,
        "max_discrepancy_in_sample_vs_subspace": float(np.max(discrepancies)) if discrepancies else None,
        "note": "real recordings are neither isotropic in feature space nor exactly generated by a "
                "single direction w, so the exact reduction proved on synthetic data above is not "
                "expected to reproduce as a near-zero discrepancy here; this block records the actual "
                "computed numbers on delivered data rather than assuming agreement",
    }


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    return X / norm


def leave_one_out_cosine_deviation(X: np.ndarray, condition: np.ndarray) -> np.ndarray:
    """Per-trial deviation score needing no dimensionality reduction: one minus the cosine
    similarity between a trial's L2-normalised feature vector and the L2-normalised leave-one-out
    mean of its own condition. NaN for trials whose condition has fewer than two members."""
    Xn = l2_normalize_rows(np.asarray(X, dtype=float))
    labels = np.asarray(condition)
    deviation = np.full(Xn.shape[0], np.nan)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 2:
            continue
        total = Xn[idx].sum(axis=0)
        for i in idx:
            loo_mean = (total - Xn[i]) / (len(idx) - 1)
            loo_norm = np.linalg.norm(loo_mean)
            if loo_norm < 1e-12:
                continue
            deviation[i] = 1.0 - float(Xn[i] @ loo_mean) / loo_norm
    return deviation


def component_direction(X: np.ndarray, deviation: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Unit-norm ridge-regression direction in feature space along which trial-to-trial variation
    best predicts the leave-one-out cosine deviation -- a concrete axis for the two subspace-
    projection claims (the predictable-fraction restatement needs no such direction, only the
    scalar deviation itself)."""
    mask = np.isfinite(deviation)
    Xc = X[mask] - X[mask].mean(axis=0)
    yc = deviation[mask] - deviation[mask].mean()
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(Xc, yc)
    w = model.coef_
    n = np.linalg.norm(w)
    return w / n if n > 1e-12 else w


# ── Rank-selection criteria ──────────────────────────────────────────────────────

def cv_reconstruction_selected_rank(X: np.ndarray, rank_grid, n_splits: int = CV_FOLDS,
                                     rng: np.random.Generator | None = None) -> dict:
    """Rank in ``rank_grid`` maximising a held-out-ENTRY PCA reconstruction R^2.

    Projecting a whole held-out trial onto train-fit components and reconstructing that same whole
    trial (``recon = (Xte_c @ comps) @ comps.T``) is monotonically non-decreasing in rank and becomes
    exact once the rank reaches the ambient feature dimension, so it can never select anything but the
    largest rank in the grid -- not a rank selector at all. Instead, half the features are withheld
    from every held-out trial (a fixed random split, shared across ranks and folds); the latent score
    for a held-out trial is estimated by least squares from its OBSERVED half against the train-fit
    loadings restricted to those features, and R^2 is scored only on the reconstruction of the WITHHELD
    half. A rank that overfits train-fold noise degrades this held-out-half prediction rather than
    trivially improving it, so the criterion can genuinely peak below the grid's largest rank.
    """
    rng = rng or np.random.default_rng(0)
    n, d = X.shape
    if d < 4:
        return {"status": "not_computable", "reason": "fewer than 4 features"}
    n_splits_eff = min(n_splits, n)
    if n_splits_eff < 2:
        return {"status": "not_computable"}
    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
    feat_perm = rng.permutation(d)
    obs_idx, held_idx = feat_perm[: d // 2], feat_perm[d // 2:]
    scores_by_rank = {}
    for r in rank_grid:
        r_eff = min(r, len(obs_idx) - 1, n - 2)
        if r_eff < 1:
            continue
        fold_r2 = []
        for train_idx, test_idx in kf.split(X):
            mu = X[train_idx].mean(axis=0)
            _, comps, _ = pca_decompose(X[train_idx], r_eff)
            Xte_c = X[test_idx] - mu
            scores_te, *_ = np.linalg.lstsq(comps[obs_idx], Xte_c[:, obs_idx].T, rcond=None)
            recon_held = (comps[held_idx] @ scores_te).T
            resid = Xte_c[:, held_idx] - recon_held
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum(Xte_c[:, held_idx] ** 2)
            fold_r2.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan"))
        scores_by_rank[int(r)] = float(np.nanmean(fold_r2))
    if not scores_by_rank:
        return {"status": "not_computable"}
    best = max(scores_by_rank, key=scores_by_rank.get)
    return {"status": "computed", "selected_rank": int(best), "scores_by_rank": scores_by_rank}


def cv_factor_analysis_selected_rank(X: np.ndarray, rank_grid, n_splits: int = CV_FOLDS,
                                      rng: np.random.Generator | None = None) -> dict:
    """Rank in ``rank_grid`` maximising held-out-trial factor-analysis log-likelihood -- the
    established rank selector when per-feature noise is not shared."""
    rng = rng or np.random.default_rng(0)
    n = X.shape[0]
    n_splits_eff = min(n_splits, n)
    if n_splits_eff < 2:
        return {"status": "not_computable"}
    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
    scores_by_rank = {}
    for r in rank_grid:
        r_eff = min(r, X.shape[1] - 1, n - 2)
        if r_eff < 1:
            continue
        fold_ll = []
        try:
            for train_idx, test_idx in kf.split(X):
                fa = FactorAnalysis(n_components=r_eff, random_state=0)
                fa.fit(X[train_idx])
                fold_ll.append(fa.score(X[test_idx]))
        except (ValueError, np.linalg.LinAlgError):
            continue
        if fold_ll:
            scores_by_rank[int(r)] = float(np.mean(fold_ll))
    if not scores_by_rank:
        return {"status": "not_computable"}
    best = max(scores_by_rank, key=scores_by_rank.get)
    return {"status": "computed", "selected_rank": int(best), "scores_by_rank": scores_by_rank}


def rank_selection_report(X: np.ndarray, rank_grid, rng: np.random.Generator) -> dict:
    """Every principled rank selector computed for one session's trial-by-feature array, reported
    side by side with no ranking between them: cross-validated reconstruction, cross-validated
    factor-analysis likelihood, participation ratio, and parallel analysis."""
    cv_recon = cv_reconstruction_selected_rank(X, rank_grid, rng=rng)
    cv_fa = cv_factor_analysis_selected_rank(X, rank_grid, rng=rng)
    Xc = X - X.mean(axis=0)
    eigenvalues = np.linalg.svd(Xc, compute_uv=False) ** 2
    pr = float(participation_ratio(eigenvalues))
    pa = int(parallel_analysis(X, rng=rng))
    return {
        "cv_reconstruction": cv_recon,
        "cv_factor_analysis_likelihood": cv_fa,
        "participation_ratio": pr,
        "participation_ratio_rounded": int(np.clip(round(pr), 1, X.shape[1])),
        "parallel_analysis_selected_rank": pa,
        "delivered_rank": DELIVERED_RANK,
    }


# ── Claim 1: memorandum coding subspace vs. the deviation direction ─────────────

def condition_mean_subspace(X: np.ndarray, labels: np.ndarray, rank: int) -> np.ndarray | None:
    """Top ``rank`` PCA axes of the between-condition mean matrix -- the linear subspace a
    memorandum-content label occupies."""
    classes = np.unique(labels)
    if len(classes) < 2:
        return None
    means = np.stack([X[labels == c].mean(axis=0) for c in classes])
    means_c = means - means.mean(axis=0)
    d = int(min(rank, means_c.shape[0] - 1, means_c.shape[1]))
    if d < 1:
        return None
    _, _, Vt = np.linalg.svd(means_c, full_matrices=False)
    return Vt[:d].T


def claim_memorandum_subspace(X: np.ndarray, labels: np.ndarray, deviation: np.ndarray, w: np.ndarray,
                               rank: int, rng: np.random.Generator, n_perm: int) -> dict:
    """Does the deviation direction lie inside the memorandum's coding subspace at this rank?
    Effect size: the observed projection fraction minus its label-permutation null."""
    S = condition_mean_subspace(X, labels, rank)
    if S is None:
        return {"status": "not_computable", "rank": rank}
    P = S @ S.T
    w_norm = np.linalg.norm(w) + 1e-12
    within = float(np.linalg.norm(P @ w) / w_norm)
    null_vals = []
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        Sp = condition_mean_subspace(X, perm_labels, rank)
        if Sp is None:
            continue
        null_vals.append(float(np.linalg.norm((Sp @ Sp.T) @ w) / w_norm))
    null_vals = np.asarray(null_vals)
    p_value = permutation_pvalue(null_vals >= within) if len(null_vals) else float("nan")
    null_mean = float(np.mean(null_vals)) if len(null_vals) else float("nan")
    return {"status": "computed", "rank": int(S.shape[1]), "within_frac": within,
            "null_mean_within_frac": null_mean, "effect_size": within - null_mean,
            "p_value": p_value, "n_null": int(len(null_vals)), "n_trials": int(X.shape[0])}


# ── Claim 2: occupied manifold decomposition ─────────────────────────────────────

def claim_occupied_manifold(X: np.ndarray, w: np.ndarray, rank: int, rng: np.random.Generator,
                             n_perm: int) -> dict:
    """Decomposition of the deviation direction against the top-``rank`` occupied manifold, against
    a random-direction null in the same ambient feature space -- the same within/outside-fraction
    construction the delivered causal-microstimulation manifold-constraint analysis uses, generalised
    from a fixed rank to a swept one."""
    representation = fit_linear_representation(X, rank)
    _require_linear_representation(representation)
    comps = representation["components"]
    r_eff = comps.shape[1]
    C = X.shape[1]
    P = comps @ comps.T
    w_norm = np.linalg.norm(w) + 1e-12
    within = float(np.linalg.norm(P @ w) / w_norm)
    outside = float(np.linalg.norm((np.eye(C) - P) @ w) / w_norm)
    rand = rng.standard_normal((n_perm, C))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True) + 1e-12
    null_within = np.linalg.norm(rand @ P, axis=1)
    p_value = permutation_pvalue(null_within >= within)
    null_mean = float(np.mean(null_within))
    return {"status": "computed", "rank": int(r_eff), "within_frac": within, "outside_frac": outside,
            "null_mean_within_frac": null_mean, "effect_size": within - null_mean,
            "p_value": p_value, "n_null": int(n_perm), "n_trials": int(X.shape[0]),
            "fitting_objective": representation["fitting_objective"]}


# ── Claim 3: cross-temporal generalisation ───────────────────────────────────────

def claim_cross_temporal_generalization(psth_z: np.ndarray, labels: np.ndarray, t_idx: np.ndarray,
                                         rank: int, rng: np.random.Generator, n_perm: int) -> dict:
    """The delivered multiclass content-CTG machinery (geometry.ctg_content_permutation_null),
    unchanged except that ``n_components`` is the swept rank rather than the fixed convention value.
    The memorandum content label here (item identity) is multiclass, not binary, which is exactly
    what ctg_content_permutation_null (macro one-vs-rest AUC, chance 0.5 regardless of class count)
    is built for -- geometry.ctg_label_permutation_null assumes a two-class label throughout its own
    scoring fold and raises on a >2-class fit, so it is the wrong entry point for this claim."""
    n_comp = int(min(rank, psth_z.shape[1] - 2))
    if n_comp < 2:
        return {"status": "not_computable", "rank": rank}
    res = ctg_content_permutation_null(psth_z, labels, t_idx, n_components=n_comp,
                                        n_splits=CTG_N_SPLITS, n_perm=n_perm, rng=rng)
    tau_info = temporal_stability_tau(res["auc_mat"])
    diag = np.diag(res["auc_mat"])
    return {"status": "computed", "rank": n_comp, "diag_auc_peak": float(np.nanmax(diag)),
            "diag_auc_mean": tau_info["mean_diag_auc"], "offdiag_auc": tau_info["mean_offdiag_auc"],
            "offdiag_effect": tau_info["offdiag_effect"], "tau": tau_info["tau"],
            "tau_interpretable": tau_info["interpretable"], "p_value": float(res["p_value"]),
            "n_trials": int(psth_z.shape[0])}


# ── Human corpus pipeline (claims 1-3) ───────────────────────────────────────────

def load_human_session_arrays(entry: dict) -> dict | None:
    """From one corpus_sessions.iter_dandi_000469 entry, build the delay-epoch trial-by-feature
    array (trial-mean firing rate per unit, for the deviation score and the two subspace claims) and
    the full delay-epoch PSTH tensor (for cross-temporal generalisation)."""
    spike_lists = entry["spike_lists"]
    onsets = entry["epoch_onsets"]["delay"]
    n_trials = len(onsets)
    if n_trials < 20 or len(spike_lists) < 8:
        return None
    psth = build_psth(spike_lists, onsets, bin_ms=HUMAN_BIN_MS, smooth_ms=0.0,
                       window_s=HUMAN_DELAY_WINDOW_S)  # (n_trials, n_units, n_bins)
    X_flat = psth.mean(axis=2)  # (n_trials, n_units) -- native trial-by-feature array
    item_ids = entry["item_ids"]
    times = np.arange(psth.shape[2]) * (HUMAN_BIN_MS / 1000.0)
    return {"psth": psth, "X_flat": X_flat, "item_ids": item_ids, "times": times}


def run_human_session(entry: dict, rank_grid, n_perm_subspace: int, n_perm_ctg: int) -> dict:
    key = f"{entry['dataset']}__{entry['session']}__{entry['structure']}"
    arrays = load_human_session_arrays(entry)
    if arrays is None:
        return {"status": "excluded", "reason": "fewer than 20 trials or 8 units after region filtering",
                "session_key": key}
    X = arrays["X_flat"]
    labels = arrays["item_ids"]
    if len(np.unique(labels)) < 2:
        return {"status": "excluded", "reason": "fewer than 2 distinct content labels", "session_key": key}
    n_trials, n_units = X.shape
    rng = np.random.default_rng(stable_seed(f"dimensionality_sweep_human_{key}"))

    deviation = leave_one_out_cosine_deviation(X, labels)
    w = component_direction(X, deviation)

    full_rank = min(n_trials - 1, n_units)
    ranks_use = sorted(set(min(r, full_rank) for r in rank_grid) | {full_rank})

    selection = rank_selection_report(X, rank_grid, rng)

    per_rank = {}
    t_idx = np.arange(0, arrays["psth"].shape[2], CTG_STEP)
    for r in ranks_use:
        claim1 = claim_memorandum_subspace(X, labels, deviation, w, r, rng, n_perm_subspace)
        claim2 = claim_occupied_manifold(X, w, r, rng, n_perm_subspace)
        claim3 = claim_cross_temporal_generalization(arrays["psth"], labels, t_idx, r, rng, n_perm_ctg)
        per_rank[str(r)] = {
            "is_full_rank": bool(r == full_rank),
            "memorandum_subspace_vs_deviation": claim1,
            "occupied_manifold_vs_deviation": claim2,
            "cross_temporal_generalization": claim3,
        }

    # Estimator-invariant restatement, linear case, at the delivered rank.
    rep_memorandum = condition_mean_subspace(X, labels, min(DELIVERED_RANK, full_rank))
    state_fit = fit_linear_representation(X, min(DELIVERED_RANK, full_rank))
    rep_state = state_fit["components"]
    predictable_fraction = predictable_fraction_restatement(
        deviation, X, {"memorandum_representation": rep_memorandum, "state_representation": rep_state}, rng)
    predictable_fraction["memorandum_representation"]["fitting_objective"] = (
        "class_mean_subspace_conditioned_on_content_label_not_a_behaviour_outcome")
    predictable_fraction["state_representation"]["fitting_objective"] = state_fit["fitting_objective"]

    return {"status": "tested", "session_key": key, "dataset": entry["dataset"],
            "patient": entry["patient"], "structure": entry["structure"],
            "n_trials": int(n_trials), "n_units": int(n_units), "n_content_labels": int(len(np.unique(labels))),
            "full_rank": int(full_rank), "rank_grid_used": ranks_use,
            "rank_selection_criteria": selection, "per_rank": per_rank,
            "predictable_fraction_restatement": predictable_fraction}


# ── Causal microstimulation corpus pipeline (claims 4-5) ─────────────────────────

def load_causal_microstim_dynamics_inputs(prefix: str) -> dict | None:
    corr = load_causal_microstim_session(prefix, correct=True)
    if corr is None or corr["control_idx"] is None:
        return None
    err = load_causal_microstim_session(prefix, correct=False)
    control_idx = corr["control_idx"]
    channel_ids = corr["channel_ids"]
    C = len(channel_ids)

    ctrl_epochs = [crop_trial(tr["spikerate"]) for tr in corr["trials"] if tr["stim_cond"] == control_idx]
    ctrl_epochs = [e for e in ctrl_epochs if e is not None]
    if len(ctrl_epochs) < 10:
        return None
    trials = np.stack(ctrl_epochs, axis=0).transpose(0, 2, 1)  # (n_trials, C, n_bins)

    n_correct_control = len(ctrl_epochs)
    n_error_control = sum(1 for tr in err["trials"] if tr["stim_cond"] == control_idx) if err else 0

    cond_info = {}
    for c in range(len(corr["stim_channels"])):
        if c == control_idx:
            continue
        chan_ids = corr["stim_channels"][c]
        idx = [i for i, cid in enumerate(channel_ids) if cid in chan_ids]
        if len(idx) != len(chan_ids):
            continue
        b_chan = np.zeros(C)
        b_chan[idx] = 1.0 / len(idx)
        n_correct = sum(1 for tr in corr["trials"] if tr["stim_cond"] == c)
        n_error = sum(1 for tr in err["trials"] if tr["stim_cond"] == c) if err else 0
        accuracy = n_correct / (n_correct + n_error) if (n_correct + n_error) > 0 else None
        cond_info[c] = {"b_chan": b_chan, "n_correct": n_correct, "n_error": n_error, "accuracy": accuracy}

    return {"trials": trials, "n_channels": int(C), "cond_info": cond_info,
            "control_accuracy": {
                "n_correct": n_correct_control, "n_error": n_error_control,
                "accuracy": n_correct_control / (n_correct_control + n_error_control)
                if (n_correct_control + n_error_control) > 0 else None}}


def _energy_error_slope(energies: np.ndarray, errors: np.ndarray) -> float | None:
    if len(energies) < 2 or np.std(energies) < 1e-12:
        return None
    return float(np.polyfit(energies, errors, 1)[0])


def claim_control_model(fit: dict, cond_info: dict, rng: np.random.Generator) -> dict:
    """Controllability, the stimulation-input-alignment targeting quantity, and the energy-accuracy
    trade-off, at the rank ``fit`` was estimated at. The energy-accuracy sweep targets each real
    stimulation direction scaled to unit displacement in the fitted latent space with a fully
    actuatable (identity) input matrix -- this asks how the trade-off itself shifts with rank, not a
    reproduction of any single delivered causal-displacement measurement."""
    A, components, v_star, v_stable = fit["A"], fit["components"], fit["v_star"], fit["v_stable"]
    k = A.shape[0]
    per_condition = {}
    x0_list, xf_list = [], []
    for c, info in cond_info.items():
        alignment = stimulation_input_alignment(A, components, info["b_chan"], v_star, v_stable, rng,
                                                  gramian_horizon=CAUSAL_MICROSTIM_GRAMIAN_HORIZON)
        per_condition[str(c)] = {**alignment, "accuracy": info["accuracy"],
                                  "n_correct": info["n_correct"], "n_error": info["n_error"]}
        b_lat = components.T @ info["b_chan"]
        b_hat = b_lat / (np.linalg.norm(b_lat) + 1e-12)
        x0_list.append(np.zeros(k))
        xf_list.append(b_hat)

    energy_accuracy = None
    if x0_list:
        B_identity = np.eye(k)
        pareto = energy_accuracy_pareto(A, B_identity, x0_list, xf_list,
                                         np.array(CAUSAL_MICROSTIM_ENERGY_ACCURACY_Q), T=20)
        energy_accuracy = {
            "q_values": pareto["q_values"].tolist(), "energies": pareto["energies"].tolist(),
            "errors": pareto["errors"].tolist(),
            "energy_error_slope": _energy_error_slope(pareto["energies"], pareto["errors"]),
        }
    return {"status": "computed", "rank": int(k), "n_conditions": int(len(cond_info)),
            "per_condition": per_condition, "energy_accuracy": energy_accuracy}


def run_causal_microstim_session(prefix: str, rank_grid) -> dict:
    key = f"causal_microstim__{prefix}"
    inputs = load_causal_microstim_dynamics_inputs(prefix)
    if inputs is None:
        return {"status": "excluded", "reason": "fewer than 10 control-condition correct trials, "
                "or no control condition identified", "session_key": key}
    trials = inputs["trials"]
    n_trials, C, n_bins = trials.shape
    if not inputs["cond_info"]:
        return {"status": "excluded", "reason": "no stimulation condition survived the channel filter",
                "session_key": key}
    srate = 1.0 / CAUSAL_MICROSTIM_BIN_S
    rng = np.random.default_rng(stable_seed(f"dimensionality_sweep_causal_microstim_{key}"))

    full_rank = min(C, n_trials - 2)
    ranks_use = sorted(set(min(r, full_rank) for r in rank_grid) | {full_rank})

    selection = rank_selection_report(trials.mean(axis=2), rank_grid, rng)

    per_rank = {}
    for r in ranks_use:
        fit_rng = np.random.default_rng(stable_seed(f"{key}_rank_{r}"))
        try:
            fit = fit_retention_dynamics(trials, srate, k=r, rng=fit_rng)
        except (np.linalg.LinAlgError, ValueError) as exc:
            per_rank[str(r)] = {"status": "fit_failed", "reason": str(exc), "is_full_rank": bool(r == full_rank)}
            continue
        dynamics_claim = {"status": "computed", "rank": fit["r_used"], "rho": fit["rho"],
                           "theta": fit["theta"], "classification": fit["classification"],
                           "r2_cv": fit["r2_cv"], "r2_null": fit["r2_null"],
                           "identifiable": fit["identifiable"], "n_trials": fit["n_trials"]}
        control_claim = claim_control_model(fit, inputs["cond_info"], rng)
        per_rank[str(r)] = {"is_full_rank": bool(r == full_rank), "fitted_dynamics": dynamics_claim,
                             "control_model": control_claim}

    # Part (b) is defined per TRIAL (the memorandum/state representation predicts a per-trial
    # scalar); this corpus's natural targets -- alignment, accuracy -- are per STIMULATION
    # CONDITION, of which a session has at most a handful, far short of what cross-validation
    # needs. The predictable-fraction restatement is reported as computed only in the human corpus,
    # which has a genuine per-trial target (the leave-one-out cosine deviation).
    predictable_fraction = {"status": "not_computable",
                             "reason": "this corpus's control-model targets (alignment, accuracy) are "
                             "per stimulation condition, of which a session has too few to "
                             "cross-validate; see the human-corpus restatement for the computed case"}

    return {"status": "tested", "session_key": key, "n_trials": int(n_trials), "n_channels": int(C),
            "n_conditions": int(len(inputs["cond_info"])), "control_accuracy": inputs["control_accuracy"],
            "full_rank": int(full_rank), "rank_grid_used": ranks_use,
            "rank_selection_criteria": selection, "per_rank": per_rank,
            "predictable_fraction_restatement": predictable_fraction}


# ── Cross-session aggregation ────────────────────────────────────────────────────

def _collect_claim_nodes(sessions: list[dict], claim_path: tuple[str, ...]):
    """Yield (rank, node, is_full_rank) for every tested session's computed claim at claim_path."""
    for session in sessions:
        if session.get("status") != "tested":
            continue
        for rank_str, entry in session.get("per_rank", {}).items():
            node = entry
            for key in claim_path:
                node = node.get(key, {}) if isinstance(node, dict) else {}
            if node.get("status") != "computed":
                continue
            yield int(rank_str), node, bool(entry.get("is_full_rank"))


def _effect_and_significance_stats(nodes: list[dict]) -> dict:
    effects = np.array([n["effect_size"] for n in nodes if "effect_size" in n and np.isfinite(n["effect_size"])])
    pvals = np.array([n["p_value"] for n in nodes if "p_value" in n and np.isfinite(n["p_value"])])
    mdd = minimum_detectable_paired_difference(effects) if len(effects) >= 2 else {"status": "not_computable"}
    return {"n_sessions": int(len(nodes)),
            "mean_effect_size": float(np.mean(effects)) if len(effects) else None,
            "std_effect_size": float(np.std(effects, ddof=1)) if len(effects) > 1 else None,
            "fraction_significant_p_below_0p05": float(np.mean(pvals < 0.05)) if len(pvals) else None,
            "minimum_detectable_difference": mdd}


def _sign_and_significance_summary(sessions: list[dict], claim_path: tuple[str, ...]) -> dict:
    """Across sessions and ranks, whether a claim's effect-size sign or its significance (p<0.05)
    changes anywhere in the swept range, including at the full-rank anchor. A claim whose sign or
    significance is not stable across that range is withdrawn from the argument rather than reported
    as if it held generally.

    The full-rank anchor is pooled separately from the rank grid: each session's own full rank
    (native, no reduction) is a different number, so pooling those entries by their literal rank
    value would scatter them into per-session singleton buckets instead of a single anchor
    comparison. ``full_rank_anchor`` aggregates them under one label regardless of the numeric rank
    that produced each one."""
    by_rank: dict[int, list[dict]] = {}
    anchor_nodes: list[dict] = []
    for rank, node, is_full in _collect_claim_nodes(sessions, claim_path):
        by_rank.setdefault(rank, []).append(node)
        if is_full:
            anchor_nodes.append(node)

    per_rank_summary = {str(rank): _effect_and_significance_stats(nodes) for rank, nodes in sorted(by_rank.items())}
    full_rank_anchor = _effect_and_significance_stats(anchor_nodes) if anchor_nodes else {"status": "not_computable"}

    mean_effects = [v["mean_effect_size"] for v in per_rank_summary.values() if v["mean_effect_size"] is not None]
    if full_rank_anchor.get("mean_effect_size") is not None:
        mean_effects = mean_effects + [full_rank_anchor["mean_effect_size"]]
    sign_changes = len(mean_effects) >= 2 and (min(mean_effects) < 0 < max(mean_effects))
    sig_flags = [v["fraction_significant_p_below_0p05"] for v in per_rank_summary.values()
                 if v["fraction_significant_p_below_0p05"] is not None]
    if full_rank_anchor.get("fraction_significant_p_below_0p05") is not None:
        sig_flags = sig_flags + [full_rank_anchor["fraction_significant_p_below_0p05"]]
    # "significance changes" = at least one rank (grid or full-rank anchor) has a majority of
    # sessions significant and at least one other does not -- the coarsest defensible read of "the
    # null flips" across the swept range.
    significance_changes = len(sig_flags) >= 2 and (max(sig_flags) >= 0.5) and (min(sig_flags) < 0.5)
    return {"per_rank": per_rank_summary, "full_rank_anchor": full_rank_anchor,
            "sign_changes_anywhere_in_range": bool(sign_changes),
            "significance_changes_anywhere_in_range": bool(significance_changes),
            "withdrawn": bool(sign_changes or significance_changes)}


def _ctg_stats(nodes: list[dict]) -> dict:
    offdiag = np.array([n["offdiag_effect"] for n in nodes])
    diag = np.array([n["diag_auc_peak"] - 0.5 for n in nodes])
    pvals = np.array([n["p_value"] for n in nodes])
    mdd = minimum_detectable_paired_difference(offdiag) if len(offdiag) >= 2 else {"status": "not_computable"}
    return {"n_sessions": int(len(nodes)),
            "mean_offdiag_auc_minus_chance": float(np.mean(offdiag)) if len(offdiag) else None,
            "mean_diag_auc_peak_minus_chance": float(np.mean(diag)) if len(diag) else None,
            "fraction_significant_p_below_0p05": float(np.mean(pvals < 0.05)) if len(pvals) else None,
            "minimum_detectable_difference_offdiag": mdd}


def _ctg_summary(sessions: list[dict]) -> dict:
    """As ``_sign_and_significance_summary``, specialised to the cross-temporal-generalisation
    claim's own node shape (offdiag_effect / diag_auc_peak / p_value rather than effect_size), with
    the same full-rank anchor treatment."""
    by_rank: dict[int, list[dict]] = {}
    anchor_nodes: list[dict] = []
    for session in sessions:
        if session.get("status") != "tested":
            continue
        for rank_str, entry in session.get("per_rank", {}).items():
            node = entry.get("cross_temporal_generalization", {})
            if node.get("status") != "computed":
                continue
            by_rank.setdefault(int(rank_str), []).append(node)
            if entry.get("is_full_rank"):
                anchor_nodes.append(node)
    per_rank_summary = {str(rank): _ctg_stats(nodes) for rank, nodes in sorted(by_rank.items())}
    full_rank_anchor = _ctg_stats(anchor_nodes) if anchor_nodes else {"status": "not_computable"}

    means = [v["mean_offdiag_auc_minus_chance"] for v in per_rank_summary.values() if v["mean_offdiag_auc_minus_chance"] is not None]
    if full_rank_anchor.get("mean_offdiag_auc_minus_chance") is not None:
        means = means + [full_rank_anchor["mean_offdiag_auc_minus_chance"]]
    sign_changes = len(means) >= 2 and (min(means) < 0 < max(means))
    sig = [v["fraction_significant_p_below_0p05"] for v in per_rank_summary.values() if v["fraction_significant_p_below_0p05"] is not None]
    if full_rank_anchor.get("fraction_significant_p_below_0p05") is not None:
        sig = sig + [full_rank_anchor["fraction_significant_p_below_0p05"]]
    significance_changes = len(sig) >= 2 and (max(sig) >= 0.5) and (min(sig) < 0.5)
    return {"per_rank": per_rank_summary, "full_rank_anchor": full_rank_anchor,
            "sign_changes_anywhere_in_range": bool(sign_changes),
            "significance_changes_anywhere_in_range": bool(significance_changes),
            "withdrawn": bool(sign_changes or significance_changes)}


def _dynamics_stats(nodes: list[dict]) -> dict:
    rho = np.array([n["rho"] for n in nodes])
    classifications = [n["classification"] for n in nodes]
    return {"n_sessions": int(len(nodes)),
            "mean_rho": float(np.mean(rho)) if len(rho) else None,
            "std_rho": float(np.std(rho, ddof=1)) if len(rho) > 1 else None,
            "fraction_identifiable": float(np.mean([n["identifiable"] for n in nodes])) if nodes else None,
            "classification_counts": {c: classifications.count(c) for c in set(classifications)}}


def _dynamics_summary(sessions: list[dict]) -> dict:
    """As ``_sign_and_significance_summary``, specialised to the fitted-dynamics claim (dominant
    eigenmode classification and decay rate rather than a permutation effect size), with the same
    full-rank anchor treatment."""
    by_rank: dict[int, list[dict]] = {}
    anchor_nodes: list[dict] = []
    for session in sessions:
        if session.get("status") != "tested":
            continue
        for rank_str, entry in session.get("per_rank", {}).items():
            node = entry.get("fitted_dynamics", {})
            if node.get("status") != "computed":
                continue
            by_rank.setdefault(int(rank_str), []).append(node)
            if entry.get("is_full_rank"):
                anchor_nodes.append(node)
    per_rank_summary = {str(rank): _dynamics_stats(nodes) for rank, nodes in sorted(by_rank.items())}
    full_rank_anchor = _dynamics_stats(anchor_nodes) if anchor_nodes else {"status": "not_computable"}

    rhos = [v["mean_rho"] for v in per_rank_summary.values() if v["mean_rho"] is not None]
    classification_sets = [set(v["classification_counts"]) for v in per_rank_summary.values()]
    if full_rank_anchor.get("mean_rho") is not None:
        rhos = rhos + [full_rank_anchor["mean_rho"]]
        classification_sets = classification_sets + [set(full_rank_anchor["classification_counts"])]
    classification_changes = len(set.union(*classification_sets)) > 1 if classification_sets else False
    stability_sign = [(r - 1.0) for r in rhos]
    sign_changes = len(stability_sign) >= 2 and (min(stability_sign) < 0 < max(stability_sign))
    return {"per_rank": per_rank_summary, "full_rank_anchor": full_rank_anchor,
            "dominant_classification_changes_anywhere_in_range": bool(classification_changes),
            "stability_side_changes_anywhere_in_range": bool(sign_changes),
            "withdrawn": bool(classification_changes or sign_changes)}


def _control_stats(nodes: list[dict]) -> dict:
    align_vstar, align_random, slopes = [], [], []
    for n in nodes:
        for cond in n["per_condition"].values():
            align_vstar.append(cond["alignment_to_vstar"])
            align_random.append(cond["random_direction_alignment"])
        if n.get("energy_accuracy") and n["energy_accuracy"].get("energy_error_slope") is not None:
            slopes.append(n["energy_accuracy"]["energy_error_slope"])
    return {"n_sessions": int(len(nodes)), "n_conditions": int(len(align_vstar)),
            "mean_alignment_to_vstar": float(np.mean(align_vstar)) if align_vstar else None,
            "mean_random_direction_alignment": float(np.mean(align_random)) if align_random else None,
            "alignment_excess_over_random": float(np.mean(align_vstar) - np.mean(align_random)) if align_vstar else None,
            "mean_energy_error_slope": float(np.mean(slopes)) if slopes else None}


def _control_summary(sessions: list[dict]) -> dict:
    """As ``_sign_and_significance_summary``, specialised to the control-model claim (targeting
    alignment excess over a random-direction baseline), with the same full-rank anchor treatment."""
    by_rank: dict[int, list[dict]] = {}
    anchor_nodes: list[dict] = []
    for session in sessions:
        if session.get("status") != "tested":
            continue
        for rank_str, entry in session.get("per_rank", {}).items():
            node = entry.get("control_model", {})
            if node.get("status") != "computed":
                continue
            by_rank.setdefault(int(rank_str), []).append(node)
            if entry.get("is_full_rank"):
                anchor_nodes.append(node)
    per_rank_summary = {str(rank): _control_stats(nodes) for rank, nodes in sorted(by_rank.items())}
    full_rank_anchor = _control_stats(anchor_nodes) if anchor_nodes else {"status": "not_computable"}

    excess = [v["alignment_excess_over_random"] for v in per_rank_summary.values() if v["alignment_excess_over_random"] is not None]
    if full_rank_anchor.get("alignment_excess_over_random") is not None:
        excess = excess + [full_rank_anchor["alignment_excess_over_random"]]
    sign_changes = len(excess) >= 2 and (min(excess) < 0 < max(excess))
    return {"per_rank": per_rank_summary, "full_rank_anchor": full_rank_anchor,
            "alignment_excess_sign_changes_anywhere_in_range": bool(sign_changes),
            "withdrawn": bool(sign_changes)}


# ── Main ──────────────────────────────────────────────────────────────────────────

def session_shard() -> tuple[int, int]:
    """Which slice of the session list this process is responsible for, as (index, count).

    Workers share one checkpoint directory and every record is written by atomic rename, so
    concurrent workers on disjoint slices cannot collide. A sharded process deliberately writes no
    pooled artifact: its view of the corpora is partial, and the cross-session claim summaries are
    only meaningful over the whole session list. Run once unsharded afterwards (the default,
    ``0/1``) to aggregate -- every record is cached by then, so that pass only reloads sessions and
    writes the artifact.
    """
    index, count = (int(part) for part in os.environ.get(SHARD_VARIABLE, "0/1").split("/"))
    if not 0 <= index < count:
        raise SystemExit(f"{SHARD_VARIABLE} must be i/n with 0 <= i < n; got {index}/{count}")
    return index, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-perm-subspace", type=int, default=SUBSPACE_N_PERM_DEFAULT)
    parser.add_argument("--n-perm-ctg", type=int, default=CTG_N_PERM_DEFAULT)
    parser.add_argument("--human-sessions-limit", type=int, default=None)
    parser.add_argument("--causal-microstim-sessions-limit", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    shard_index, shard_count = session_shard()
    root = data_root()

    human_sessions = []
    n_seen_human = 0
    for i, entry in enumerate(iter_dandi_000469(root)):
        if entry["structure"] != "pooled":
            continue
        if args.human_sessions_limit is not None and len(human_sessions) >= args.human_sessions_limit:
            break
        n_seen_human += 1
        if i % shard_count != shard_index:
            continue
        key = f"{entry['dataset']}__{entry['session']}__{entry['structure']}"
        record = run_checkpointed(
            f"human__{key}",
            lambda entry=entry: run_human_session(entry, RANK_GRID, args.n_perm_subspace, args.n_perm_ctg),
        )
        human_sessions.append(record)
        print(f"[human] {key}: {record['status']}", file=sys.stderr)

    causal_microstim_sessions = []
    prefixes = (CAUSAL_MICROSTIM_SESSIONS if args.causal_microstim_sessions_limit is None
                else CAUSAL_MICROSTIM_SESSIONS[:args.causal_microstim_sessions_limit])
    for i, prefix in enumerate(prefixes):
        if i % shard_count != shard_index:
            continue
        # Checkpoint key carries a schema tag ("_v2") distinct from this corpus's earlier checkpoint
        # key: the records cached under the earlier key were written by a prior version of
        # run_causal_microstim_session's channel-matching logic and uniformly reported an exclusion
        # reason this file's current code cannot reproduce for any of these eleven sessions (verified
        # by calling the loader directly) -- an implementation/checkpoint-staleness artifact, not a
        # statement about the data, recorded below in causal_microstimulation_checkpoint_note. The
        # schema tag forces a fresh fit under the current code rather than silently replaying the
        # stale cached exclusion.
        record = run_checkpointed(
            f"causal_microstim_v2__{prefix}",
            lambda prefix=prefix: run_causal_microstim_session(prefix, RANK_GRID))
        causal_microstim_sessions.append(record)
        print(f"[causal_microstim] {prefix}: {record['status']}", file=sys.stderr)

    if shard_count > 1:
        print(f"Shard {shard_index} of {shard_count} finished its sessions in "
              f"{time.time() - t0:.0f}s; checkpoints written, no artifact aggregated. Run once more "
              f"with {SHARD_VARIABLE} unset to aggregate every shard's checkpoints into the "
              "artifact.", file=sys.stderr)
        return

    n_human_tested = sum(1 for s in human_sessions if s["status"] == "tested")
    n_human_excluded = sum(1 for s in human_sessions if s["status"] == "excluded")
    n_causal_microstim_tested = sum(1 for s in causal_microstim_sessions if s["status"] == "tested")
    n_causal_microstim_excluded = sum(1 for s in causal_microstim_sessions if s["status"] == "excluded")
    causal_microstim_exclusion_reasons = sorted({
        s["reason"] for s in causal_microstim_sessions if s["status"] == "excluded"})

    # Build zero_drop_accounting: sessions seen, analysed, refused, with per-reason counts
    def _build_zero_drop_accounting() -> dict:
        """Machine-checkable reconciliation: sessions seen == sessions analysed + sessions refused."""
        # Human corpus: count refusal reasons
        human_refusal_reasons = {}
        for s in human_sessions:
            if s.get("status") == "excluded":
                reason = s.get("reason", "unknown")
                human_refusal_reasons[reason] = human_refusal_reasons.get(reason, 0) + 1
        # Causal microstim corpus: count refusal reasons
        causal_refusal_reasons = {}
        for s in causal_microstim_sessions:
            if s.get("status") == "excluded":
                reason = s.get("reason", "unknown")
                causal_refusal_reasons[reason] = causal_refusal_reasons.get(reason, 0) + 1
        return {
            "human_corpus_dandi_000469": {
                "sessions_seen": int(n_seen_human),
                "sessions_analysed": int(n_human_tested),
                "sessions_refused": int(n_human_excluded),
                "refusal_reasons": human_refusal_reasons,
            },
            "causal_microstimulation_corpus": {
                "sessions_seen": int(len(prefixes)),
                "sessions_analysed": int(n_causal_microstim_tested),
                "sessions_refused": int(n_causal_microstim_excluded),
                "refusal_reasons": causal_refusal_reasons,
            },
            "summary": {
                "total_sessions_seen": int(n_seen_human) + int(len(prefixes)),
                "total_sessions_analysed": int(n_human_tested) + int(n_causal_microstim_tested),
                "total_sessions_refused": int(n_human_excluded) + int(n_causal_microstim_excluded),
                "all_seen_sessions_accounted_for": (
                    int(n_seen_human) == int(n_human_tested) + int(n_human_excluded) and
                    int(len(prefixes)) == int(n_causal_microstim_tested) + int(n_causal_microstim_excluded)
                ),
            },
        }

    verification_rng = np.random.default_rng(stable_seed("dimensionality_sweep_restatement_reduction_check"))

    output = {
        "version": "dimensionality_sweep_v1",
        "scope": {
            "human_corpus": "dandi_000469, structure='pooled', memorandum content label = encoded "
                             "item identity, delay epoch only",
            "causal_microstimulation_corpus": "causal microstimulation macaque corpus, "
                               "control-condition correct trials for the linear-plant fit, every "
                               "stimulation condition surviving the channel filter for the "
                               "control-model claims",
            "n_human_sessions_seen": int(n_seen_human), "n_human_sessions_tested": int(n_human_tested),
            "n_human_sessions_excluded": int(n_human_excluded),
            "n_causal_microstimulation_sessions_seen": int(len(prefixes)),
            "n_causal_microstimulation_sessions_tested": int(n_causal_microstim_tested),
            "n_causal_microstimulation_sessions_excluded": int(n_causal_microstim_excluded),
            "causal_microstimulation_exclusion_reasons": causal_microstim_exclusion_reasons,
            "rank_grid": list(RANK_GRID), "delivered_rank": DELIVERED_RANK,
            "human_bin_ms": HUMAN_BIN_MS, "ctg_step_bins": CTG_STEP, "ctg_n_splits": CTG_N_SPLITS,
            "n_perm_subspace": args.n_perm_subspace, "n_perm_ctg": args.n_perm_ctg,
            "seed_scheme": "stable_seed(f'dimensionality_sweep_<corpus>_<session_key>[_rank_<r>]')",
        },
        "causal_microstimulation_checkpoint_note": {
            "finding": "the per-session records cached under this corpus's earlier checkpoint key "
                       "uniformly reported the exclusion reason 'fewer than 10 control-condition "
                       "correct trials, or no control condition identified' for all eleven sessions; "
                       "calling this file's current loader directly reproduces that reason for none "
                       "of the eleven -- eight sessions fit successfully and three are excluded for a "
                       "different, session-specific reason (their recorded stimulation-channel ids do "
                       "not fully match this session's recorded channel list after the shorted-channel "
                       "filter, a genuine per-session data limitation, listed under "
                       "scope.causal_microstimulation_exclusion_reasons), so the cached records "
                       "reflected an earlier, since-corrected version of the loader rather than the "
                       "code that produced this artifact",
            "category": "implementation_and_checkpoint_staleness_artifact_not_a_statement_about_the_data",
            "remedy": "recomputed every session under a schema-tagged checkpoint key so every cached "
                      "record in this artifact reflects the loader exactly as it exists in this file",
        },
        "human_sessions": human_sessions,
        "causal_microstimulation_sessions": causal_microstim_sessions,
        "claim_summaries": {
            "memorandum_coding_subspace_vs_deviation": _sign_and_significance_summary(
                human_sessions, ("memorandum_subspace_vs_deviation",)),
            "occupied_manifold_vs_deviation": _sign_and_significance_summary(
                human_sessions, ("occupied_manifold_vs_deviation",)),
            "cross_temporal_generalization": _ctg_summary(human_sessions),
            "fitted_dynamics": _dynamics_summary(causal_microstim_sessions),
            "control_model": _control_summary(causal_microstim_sessions),
        },
        "estimator_invariant_restatement_verification": {
            "function": "cross_validated_predictable_fraction(y, Z, n_splits=5, alpha=1.0, rng=None)",
            "synthetic_linear_reduction": restatement_reduction_synthetic_check(verification_rng),
            "delivered_human_data_comparison": restatement_reduction_on_delivered_human_data(human_sessions),
        },
        "zero_drop_accounting": _build_zero_drop_accounting(),
        "wall_clock_s": time.time() - t0,
    }
    OUTPUT_PATH.write_text(canonical_json(output))
    print(f"Wrote {OUTPUT_PATH} in {output['wall_clock_s']:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
