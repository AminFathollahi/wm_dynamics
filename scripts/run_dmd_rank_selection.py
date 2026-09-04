#!/usr/bin/env python3
"""DMD operator-rank selection and v*-stability for the macaque PFC microstimulation causal benchmark.

The operator rank r is a separate choice from the latent dimension: it fixes
the identified leading direction v* on which the causal benchmark rests, so
it is the highest-stakes dimensionality choice in the pipeline. This script
selects r on data rather than asserting it, and checks whether v* is stable
to that choice.

  (i)  Cross-validated one-step R^2 of the DMD map for r in 4..8 (reuses
       dynamics.ensemble_dmd on the macaque PFC microstimulation control latent trajectories — the
       same pooled single-trial snapshot-pair fit used everywhere else). Report
       the curve and the data-selected r (R^2 elbow within 0.01 of the best).
  (ii) v*(r) stability: cosine similarity of v*(r) to the r=6 solution (kept as
       a fixed reference direction) and the sign/magnitude of max Re(lambda),
       across r in {5,6,7,8}. If v* rotates materially or the leading
       eigenvalue flips stability, the benchmark is rank-dependent, and that
       is reported plainly rather than papered over.

  Plus the causal-benchmark v*-alignment slope re-fit at each r (reuses the
  macaque PFC microstimulation pipeline's own build_session_features + benchmark_modifiers with
  DMD_RANK overridden), so run_dim_robustness.py can report the headline vs rank.

results/dmd_rank_selection.json — {r: {cv_r2_onestep, vstar_cos_to_r6,
    max_re_lambda, benchmark_slope, benchmark_p}}.

Run (needs the macaque PFC microstimulation data mount):
    conda run -n wm_dynamics python scripts/run_dmd_rank_selection.py
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

import run_macaque_pfc_microstimulation_pipeline as macaque_pfc_microstimulation  # noqa: E402  (reuse its loaders + feature build)
from dynamics import ensemble_dmd, dmd_reconstruction_error  # noqa: E402
from causal import benchmark_modifiers  # noqa: E402
from statistics import stable_seed  # noqa: E402
from geometry import pca_decompose  # noqa: E402
from control import canonicalize_eigenvector_phase  # noqa: E402
from provenance import _json_safe

RESULTS = ROOT / "results"
RANKS = (4, 5, 6, 7, 8)
BIN_S = macaque_pfc_microstimulation.BIN_S
N_BINS = macaque_pfc_microstimulation.N_BINS
N_PC = macaque_pfc_microstimulation.N_PC


def _control_latents() -> list[np.ndarray]:
    """Per-session control-trial latent trajectories (N, N_BINS, k), PCA fit on
    the same control pool the pipeline uses. Reused across all ranks."""
    lat = []
    for prefix in macaque_pfc_microstimulation.SESSIONS:
        corr = macaque_pfc_microstimulation.load_macaque_pfc_microstimulation_session(prefix, correct=True)
        if corr is None or corr["control_idx"] is None:
            continue
        C = len(corr["channel_ids"])
        ctrl = [macaque_pfc_microstimulation.crop_trial(tr["spikerate"]) for tr in corr["trials"]
                if tr["stim_cond"] == corr["control_idx"]]
        ctrl = [e for e in ctrl if e is not None]
        if len(ctrl) < 10:
            continue
        Zc = np.stack(ctrl, axis=0)               # (N, N_BINS, C)
        Xf = Zc.reshape(-1, C)
        _, V, _ = pca_decompose(Xf, N_PC)
        Z = ((Xf - Xf.mean(0)) @ V).reshape(Zc.shape[0], N_BINS, V.shape[1])
        lat.append(Z)
    return lat


def _vstar_and_eig(Z_mean: np.ndarray, r: int) -> tuple[np.ndarray, float]:
    """v* (phase-canonicalized direction of the top-eigenvalue eigenvector)
    and max Re(lambda) of the mean-trajectory DMD operator — matching the
    construction in run_macaque_pfc_microstimulation_pipeline."""
    r_use = min(r, Z_mean.shape[1], N_BINS - 2)
    A = dmd_reconstruction_error(Z_mean, r=r_use, dt=BIN_S)["A"]
    eigs, vecs = np.linalg.eig(A)
    top = np.argsort(eigs.real)[::-1][0]
    v = canonicalize_eigenvector_phase(vecs[:, top])
    return v, float(eigs[top].real)


def cv_r2_and_vstar(latents: list[np.ndarray]) -> dict:
    """Per-rank held-out one-step R^2 (ensemble_dmd, averaged over sessions) and
    v*/eigenvalue stats (averaged over sessions; cosine to each session's own r=6 v*)."""
    per_rank = {r: {"cv_r2": [], "cos_r6": [], "max_re": []} for r in RANKS}
    for Z in latents:
        Z_mean = Z.mean(0)
        v6, _ = _vstar_and_eig(Z_mean, 6)
        for r in RANKS:
            ens = ensemble_dmd(Z, r=r, dt=BIN_S, n_splits=5, n_null=10,
                               rng=np.random.default_rng(stable_seed(f"dmdrank_{r}")))
            per_rank[r]["cv_r2"].append(ens["r2_cv"])
            v, mre = _vstar_and_eig(Z_mean, r)
            per_rank[r]["cos_r6"].append(abs(float(v @ v6)))
            per_rank[r]["max_re"].append(mre)
    return per_rank


def benchmark_slope_at_rank(r: int) -> dict:
    """Re-run the macaque PFC microstimulation v*-alignment benchmark with the operator rank set to r,
    reusing build_session_features + benchmark_modifiers unchanged."""
    old = macaque_pfc_microstimulation.DMD_RANK
    macaque_pfc_microstimulation.DMD_RANK = r
    try:
        all_rows, session_order = [], []
        for prefix in macaque_pfc_microstimulation.SESSIONS:
            try:
                feat = macaque_pfc_microstimulation.build_session_features(prefix, structural_ctrl=None)
            except Exception:
                feat = None
            if feat is None:
                continue
            si = len(session_order)
            session_order.append(prefix)
            for row in feat["rows"]:
                row["session_idx"] = si
            all_rows.extend(feat["rows"])
        if not all_rows:
            return {"benchmark_slope": float("nan"), "benchmark_p": float("nan"), "n": 0}
        y = np.array([r_["y"] for r_ in all_rows], float)
        t = np.array([r_["t"] for r_ in all_rows], int)
        modifier = np.array([r_["modifier"] for r_ in all_rows], float)
        propensity = np.array([r_["propensity"] for r_ in all_rows], float)
        angle_idx = np.array([r_["angle_idx"] for r_ in all_rows], int)
        session_idx = np.array([r_["session_idx"] for r_ in all_rows], int)
        X = np.hstack([np.eye(angle_idx.max() + 1)[angle_idx],
                       np.eye(len(session_order))[session_idx]])

        def _col(key):
            return np.array([r_.get(key, np.nan) for r_ in all_rows], float)

        modifiers = {"vstar_alignment": modifier, "gramian_trace": _col("gramian_trace"),
                     "stable_alignment": _col("stable_alignment"),
                     "random_alignment": _col("random_alignment"), "input_norm": _col("input_norm")}
        bench = benchmark_modifiers(y, t, X, modifiers=modifiers, propensity=propensity,
                                    n_perm=2000, rng=np.random.default_rng(stable_seed(f"bench_r{r}")))
        row = bench["leaderboard"]["vstar_alignment"]
        return {"benchmark_slope": float(row["slope"]), "benchmark_p": float(row["p_value"]),
                "winner": bench["winner"], "n": int(bench["n"])}
    finally:
        macaque_pfc_microstimulation.DMD_RANK = old


def main():
    print("Caching macaque PFC microstimulation control latents once ...")
    latents = _control_latents()
    print(f"  {len(latents)} sessions")
    if not latents:
        print("No usable macaque PFC microstimulation sessions — cannot run rank selection. STOP.")
        return

    print("Per-rank cv one-step R^2 + v* stability (ensemble_dmd) ...")
    pr = cv_r2_and_vstar(latents)

    out = {}
    for r in RANKS:
        cv = float(np.nanmean(pr[r]["cv_r2"]))
        cos = float(np.nanmean(pr[r]["cos_r6"]))
        mre = float(np.nanmean(pr[r]["max_re"]))
        print(f"  r={r}: cv_R2_onestep={cv:+.4f}  |v*·v*(r6)|={cos:.4f}  mean max Re(λ)={mre:+.4f}")
        out[str(r)] = {"cv_r2_onestep": cv, "vstar_cos_to_r6": cos, "max_re_lambda": mre}

    print("\nRe-fitting the causal-benchmark v*-slope at each rank ...")
    for r in RANKS:
        b = benchmark_slope_at_rank(r)
        out[str(r)].update(benchmark_slope=b["benchmark_slope"], benchmark_p=b["benchmark_p"],
                           benchmark_winner=b.get("winner"), benchmark_n=b.get("n"))
        print(f"  r={r}: benchmark v*-slope={b['benchmark_slope']:+.4f} p={b['benchmark_p']:.4f} "
              f"winner={b.get('winner')} (N={b.get('n')})")

    # honest elbow pick: smallest r within 0.01 R^2 of the best cv-R^2
    cvs = {r: out[str(r)]["cv_r2_onestep"] for r in RANKS}
    best = max(cvs.values())
    elbow = min(r for r in RANKS if cvs[r] >= best - 0.01)
    print(f"\nData-selected r (R^2 elbow, within 0.01 of best): r={elbow}  "
          f"(paper uses r=6 -> {'MATCH' if elbow == 6 else 'DIFFERS, report'})")

    cos_range = [out[str(r)]["vstar_cos_to_r6"] for r in (5, 6, 7, 8)]
    re_signs = [np.sign(out[str(r)]["max_re_lambda"]) for r in (5, 6, 7, 8)]
    stable = min(cos_range) > 0.9 and len(set(re_signs)) == 1
    print(f"v*-STABILITY across r in {{5,6,7,8}}: min|cos to r6|={min(cos_range):.3f}, "
          f"max Re(λ) signs={re_signs} -> {'STABLE' if stable else 'RANK-FRAGILE — STOP AND REPORT'}")

    out["_meta"] = {"selected_r": int(elbow), "vstar_rank_stable": bool(stable),
                    "paper_r": 7, "ranks": list(RANKS)}
    json.dump(_json_safe(out), open(RESULTS / "dmd_rank_selection.json", "w"), indent=2, allow_nan=False)
    stats = json.load(open(RESULTS / "all_statistics.json"))
    stats["dmd_rank_selection"] = out
    json.dump(_json_safe(stats), open(RESULTS / "all_statistics.json", "w"), indent=2, allow_nan=False)
    print("\nwrote results/dmd_rank_selection.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
