#!/usr/bin/env python3
"""T-PHATE vs. PCA embedding of the delay-period latent trajectory, with
mandatory temporal-structure nulls.

T-PHATE is built to expose temporal autocorrelation structure, so it will
render a smooth curved trajectory from almost any temporally smoothed input
-- including smoothed noise. A curved or closed embedding with no null is a
figure that cannot be wrong; this script never reports one without one.

For each cohort, every session's trial-averaged latent trajectory is
coarsened to TARGET_T timepoints (block-averaging in time, matching this
project's existing epoch-level resolution -- macaque PFC microstimulation already uses 30 bins,
the Sternberg-lineage cohorts 23) and concatenated across sessions (session
order preserved as embedding segments). No trial or session is discarded;
only the time axis is resampled to a coarser, comparable grid across
cohorts, which is required for embedding tractability at the two
cohorts (miller, boran) whose native per-trial timepoint count is in the
thousands.

The resulting cohort trajectory is embedded two ways (PCA, T-PHATE) and one
scalar closure/curvature statistic is computed on each embedding: the
fraction of total embedded path length not explained by the embedding's own
leading axis. A straight line scores near zero; a curved or closed loop
scores higher. The observed statistic is compared against two null
distributions built the same way as the project's existing CTG
phase-scramble null and a within-session timepoint-shuffle
null: (i) each session's coarsened trajectory with its row order permuted
(destroys temporal order, keeps the marginal distribution of states), and
(ii) each session's coarsened trajectory with its per-channel Fourier phases
randomised (destroys condition-aligned structure, keeps the autocorrelation
spectrum -- reusing geometry.phase_scramble_trials).

This is a corroborating panel: it enters no claim in the paper's Results or
Abstract. An honest null result is kept and reported as such, not treated
as a reason to drop the panel.

Output: results/tphate_embedding.json (per-cohort statistics, null
percentiles, p-values, verdict) and figures/tphate_embedding_supplement.png.

Run (needs the external data mount):
    conda run -n wm_dynamics python scripts/run_tphate_embedding.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import tphate  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from geometry import phase_scramble_trials  # noqa: E402
from statistics import stable_seed  # noqa: E402

import run_vstar_eigen_audit as vea  # noqa: E402  (ALL_ITERS -- per-session Z_trials/dt/r)

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TARGET_T = 30  # coarsened timepoints per session, matching this project's existing epoch
               # resolution (macaque PFC microstimulation N_BINS=30, Sternberg-lineage cohorts T=23)
N_PERMUTATIONS = 200  # matches geometry.ctg_phase_scramble_null's existing convention
N_COMPONENTS = 2


def _coarsen(traj: np.ndarray, target_t: int) -> np.ndarray:
    """Block-average a (T, d) trajectory down to at most (target_t, d).

    Sessions whose native T is already <= target_t (the three Sternberg-
    lineage cohorts at T=23, macaque PFC microstimulation/boran_units at T=30) are left
    unchanged -- np.array_split would otherwise produce empty chunks (and
    NaN means) once the number of splits requested exceeds the array length."""
    t = min(target_t, len(traj))
    chunks = np.array_split(traj, t, axis=0)
    return np.stack([c.mean(axis=0) for c in chunks], axis=0)


def _cohort_segments(sessions: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """Concatenate coarsened per-session mean trajectories; return the pooled
    matrix and each segment's length (session boundaries)."""
    lengths = [len(s) for s in sessions]
    return np.concatenate(sessions, axis=0), lengths


def _curvature_stat(embedding: np.ndarray) -> float:
    """Fraction of total embedded path length not explained by the
    embedding's own leading axis. ~0 for a straight line, higher for a
    curved or closed path."""
    diffs = np.diff(embedding, axis=0)
    total_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    if total_len < 1e-10:
        return float("nan")
    pc1 = PCA(n_components=1).fit(embedding).components_[0]
    proj = embedding @ pc1
    explained_len = float(np.sum(np.abs(np.diff(proj))))
    return 1.0 - explained_len / total_len


def _embed(X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pca_emb = PCA(n_components=N_COMPONENTS, random_state=seed).fit_transform(X)
    tp = tphate.TPHATE(n_components=N_COMPONENTS, random_state=seed, verbose=0, n_jobs=1)
    tphate_emb = tp.fit_transform(X)
    return pca_emb, tphate_emb


def _timepoint_shuffle(segments: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    shuffled = [seg[rng.permutation(len(seg))] for seg in segments]
    return np.concatenate(shuffled, axis=0)


def _phase_scramble(segments: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    scrambled = [phase_scramble_trials(seg[None, :, :], rng)[0] for seg in segments]
    return np.concatenate(scrambled, axis=0)


def _p_value(obs: float, null: np.ndarray) -> float:
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


def run_cohort(dataset: str, segments: list[np.ndarray], rng: np.random.Generator) -> dict:
    X, lengths = _cohort_segments(segments)
    pca_obs, tphate_obs = _embed(X, seed=0)
    stat_pca_obs = _curvature_stat(pca_obs)
    stat_tphate_obs = _curvature_stat(tphate_obs)

    null_pca_shuffle = np.empty(N_PERMUTATIONS)
    null_tphate_shuffle = np.empty(N_PERMUTATIONS)
    null_pca_scramble = np.empty(N_PERMUTATIONS)
    null_tphate_scramble = np.empty(N_PERMUTATIONS)
    example_shuffle_tphate = example_scramble_tphate = None

    for p in range(N_PERMUTATIONS):
        X_shuf = _timepoint_shuffle(segments, rng)
        pca_e, tphate_e = _embed(X_shuf, seed=p + 1)
        null_pca_shuffle[p] = _curvature_stat(pca_e)
        null_tphate_shuffle[p] = _curvature_stat(tphate_e)
        if p == 0:
            example_shuffle_tphate = tphate_e

        X_scr = _phase_scramble(segments, rng)
        pca_e, tphate_e = _embed(X_scr, seed=p + 1)
        null_pca_scramble[p] = _curvature_stat(pca_e)
        null_tphate_scramble[p] = _curvature_stat(tphate_e)
        if p == 0:
            example_scramble_tphate = tphate_e

    result = {
        "n_sessions": len(segments), "target_t": TARGET_T, "n_pooled_points": len(X),
        "stat_pca_obs": stat_pca_obs, "stat_tphate_obs": stat_tphate_obs,
        "p_pca_vs_timepoint_shuffle": _p_value(stat_pca_obs, null_pca_shuffle),
        "p_pca_vs_phase_scramble": _p_value(stat_pca_obs, null_pca_scramble),
        "p_tphate_vs_timepoint_shuffle": _p_value(stat_tphate_obs, null_tphate_shuffle),
        "p_tphate_vs_phase_scramble": _p_value(stat_tphate_obs, null_tphate_scramble),
        "null_tphate_vs_timepoint_shuffle_median": float(np.median(null_tphate_shuffle)),
        "null_tphate_vs_phase_scramble_median": float(np.median(null_tphate_scramble)),
    }
    significant = (result["p_tphate_vs_timepoint_shuffle"] < 0.05
                   and result["p_tphate_vs_phase_scramble"] < 0.05)
    result["verdict"] = (
        "T-PHATE embedding is more curved/closed than both temporal-structure nulls"
        if significant else
        "indistinguishable from at least one temporal-structure null -- reported as an "
        "honest negative, not evidence against rotation (this panel is corroborating only)")
    print(f"  {dataset:14s} (N={len(segments):3d} sessions, {len(X)} pooled points): "
          f"pca_stat={stat_pca_obs:.3f} tphate_stat={stat_tphate_obs:.3f} "
          f"p(shuffle)={result['p_tphate_vs_timepoint_shuffle']:.3f} "
          f"p(scramble)={result['p_tphate_vs_phase_scramble']:.3f} -> {result['verdict']}")
    return result, pca_obs, tphate_obs, example_shuffle_tphate, example_scramble_tphate


def main():
    per_cohort_sessions: dict[str, list[np.ndarray]] = {}
    for it in vea.ALL_ITERS:
        for dataset, session, Z_trials, dt, r in it():
            mean_traj = Z_trials.mean(axis=0)  # (T, d)
            per_cohort_sessions.setdefault(dataset, []).append(_coarsen(mean_traj, TARGET_T))

    out: dict[str, dict] = {}
    figure_rows = []
    for dataset, segments in per_cohort_sessions.items():
        rng = np.random.default_rng(stable_seed(f"tphate_embedding_{dataset}"))
        result, pca_obs, tphate_obs, ex_shuffle, ex_scramble = run_cohort(dataset, segments, rng)
        out[dataset] = result
        figure_rows.append((dataset, pca_obs, tphate_obs, ex_shuffle, ex_scramble))

    with open(RESULTS / "tphate_embedding.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/tphate_embedding.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_rows = len(figure_rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3 * n_rows))
    col_titles = ["PCA (real)", "T-PHATE (real)", "T-PHATE (timepoint-shuffle null)",
                  "T-PHATE (phase-scramble null)"]
    for i, (dataset, pca_obs, tphate_obs, ex_shuffle, ex_scramble) in enumerate(figure_rows):
        for j, emb in enumerate((pca_obs, tphate_obs, ex_shuffle, ex_scramble)):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            ax.plot(emb[:, 0], emb[:, 1], "-", lw=0.7, alpha=0.7)
            ax.scatter(emb[:, 0], emb[:, 1], c=np.arange(len(emb)), cmap="viridis", s=4)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=9)
            if j == 0:
                ax.set_ylabel(dataset, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("T-PHATE vs PCA, delay-period latent trajectory (corroborating panel)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "tphate_embedding_supplement.png", dpi=150)
    print("Saved figures/tphate_embedding_supplement.png")


def _self_check():
    """A straight line should score near-zero curvature; a closed loop should
    score high, under both embeddings."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 100)
    line = np.stack([t, t, t, t, t, t, t, t], axis=1) + 0.01 * rng.standard_normal((100, 8))
    loop = np.stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)] + [t * 0] * 6, axis=1)
    stat_line = _curvature_stat(PCA(n_components=2).fit_transform(line))
    stat_loop = _curvature_stat(PCA(n_components=2).fit_transform(loop))
    # A perfect circle's 1-D projection onto any diameter has total variation 4 against a
    # circumference of 2*pi, so its theoretical curvature stat is 1 - 4/(2*pi) ~= 0.363.
    assert stat_line < 0.15, f"straight line should score low curvature, got {stat_line:.3f}"
    assert stat_loop > 0.3, f"closed loop should score high curvature, got {stat_loop:.3f}"
    assert stat_loop > stat_line
    print("Self-check passed: curvature statistic separates a line from a loop "
          f"({stat_line:.3f} vs {stat_loop:.3f}).")


if __name__ == "__main__":
    _self_check()
    main()
