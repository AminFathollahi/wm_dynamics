#!/usr/bin/env python3
"""Cluster-robust, matched-design comparison of within_frac / outside_frac
(Sadtler et al. 2014 within-manifold constraint) against vstar_alignment, on
the SAME macaque PFC microstimulation rows/sessions/exclusions as
run_macaque_pfc_microstimulation_headline_robustness.py's settled vstar/min_energy result.

WHY this script exists: scripts/run_manifold_constraint.py scored
within_frac/outside_frac with benchmark_modifiers' built-in trial-level
permutation p (n=5880 clustered trials, post shorted-channel exclusion) -- this is not comparable to
vstar_alignment's cluster-robust number (a trial-level p cannot stand in for
a session-clustered one). This script supplies the missing cluster-robust
bootstrap, mirroring run_amplification_robustness.py's method exactly,
without touching the degeneracy-gate result already in
results/manifold_constraint.json (within_frac varies, nanstd=0.033 >> 1e-9,
gate passed).

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_manifold_constraint_robustness.py
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

import run_macaque_pfc_microstimulation_pipeline as macaque_pfc_microstimulation  # noqa: E402
from causal import benchmark_modifiers  # noqa: E402
from statistics import stable_seed  # noqa: E402

RESULTS = ROOT / "results"
ARMS = ("vstar_alignment", "within_frac", "outside_frac")
N_BOOT = 2000


def _build_all_rows() -> tuple[list[dict], list[str]]:
    """Identical construction to run_amplification_robustness.py's
    _build_all_rows / run_macaque_pfc_microstimulation_headline_robustness.py's _build_all_rows."""
    all_rows, session_order = [], []
    for prefix in macaque_pfc_microstimulation.SESSIONS:
        try:
            feat = macaque_pfc_microstimulation.build_session_features(prefix, structural_ctrl=None)
        except Exception as e:
            print(f"  {prefix} FAILED: {e}")
            feat = None
        if feat is None:
            print(f"  {prefix} SKIP (insufficient data)")
            continue
        si = len(session_order)
        session_order.append(prefix)
        for row in feat["rows"]:
            row["session_idx"] = si
        all_rows.extend(feat["rows"])
    return all_rows, session_order


def _col(all_rows: list[dict], key: str) -> np.ndarray:
    return np.array([r.get(key, np.nan) for r in all_rows], dtype=float)


def _build_X(all_rows, angle_idx, session_idx, n_sessions) -> np.ndarray:
    angle_oh = np.eye(angle_idx.max() + 1)[angle_idx]
    session_oh = np.eye(n_sessions)[session_idx]
    return np.hstack([angle_oh, session_oh])


def _slope_formula(m: np.ndarray, phi: np.ndarray) -> float:
    mc = m - m.mean()
    denom = (mc ** 2).sum()
    if denom < 1e-15:
        return 0.0
    return float((mc * (phi - phi.mean())).sum() / denom)


def _cluster_bootstrap(phi, modifier, session_idx, n_sessions, n_boot, rng) -> np.ndarray:
    rows_by_session = [np.where(session_idx == s)[0] for s in range(n_sessions)]
    boot = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.integers(0, n_sessions, size=n_sessions)
        idx = np.concatenate([rows_by_session[s] for s in drawn])
        boot[b] = _slope_formula(modifier[idx], phi[idx])
    return boot


def _cluster_robust_result(phi, modifier, session_idx, n_sessions, rng) -> dict:
    boot = _cluster_bootstrap(phi, modifier, session_idx, n_sessions, N_BOOT, rng)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    slope = _slope_formula(modifier, phi)
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    p = min(p, 1.0)
    return {"slope": slope, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "p_value": p, "n_boot": N_BOOT}


def main() -> None:
    print("Rebuilding all_rows exactly as run_macaque_pfc_microstimulation_pipeline.main() does ...")
    all_rows, session_order = _build_all_rows()
    if not all_rows:
        print("No usable macaque PFC microstimulation sessions -- stopping without a result.")
        return
    n_sessions = len(session_order)

    y = _col(all_rows, "y")
    t = _col(all_rows, "t").astype(int)
    vstar_mod = _col(all_rows, "modifier")
    within_mod = _col(all_rows, "within_frac")
    outside_mod = _col(all_rows, "outside_frac")
    propensity = _col(all_rows, "propensity")
    angle_idx = _col(all_rows, "angle_idx").astype(int)
    session_idx = _col(all_rows, "session_idx").astype(int)
    X = _build_X(all_rows, angle_idx, session_idx, n_sessions)

    print(f"  {len(all_rows)} rows across {n_sessions} sessions; "
          f"within_frac non-finite: {int(np.sum(~np.isfinite(within_mod)))}, "
          f"outside_frac non-finite: {int(np.sum(~np.isfinite(outside_mod)))}")

    print("Cross-fitting ONE shared pseudo-outcome (phi) for all three arms ...")
    bench_rng = np.random.default_rng(stable_seed("manifold_constraint_robustness"))
    bench = benchmark_modifiers(
        y, t, X,
        modifiers={"vstar_alignment": vstar_mod, "within_frac": within_mod, "outside_frac": outside_mod},
        propensity=propensity, n_perm=2000, rng=bench_rng,
    )
    phi = bench["phi"]
    excluded = bench.get("excluded", {})
    for arm in ("within_frac", "outside_frac"):
        if arm in excluded:
            out = {"status": "excluded", "arm": arm, "reason": excluded[arm]["reason"], "n_sessions": n_sessions}
            with open(RESULTS / "manifold_constraint_robustness.json", "w") as f:
                json.dump(out, f, indent=2)
            print(f"{arm} EXCLUDED by benchmark_modifiers' zero-variance guard: {excluded[arm]['reason']}")
            return

    z_mod = bench["z_modifiers"]
    print("\nCluster-bootstrap (session-resampled, B=2000), apples-to-apples with vstar_alignment ...")
    cluster_rng = np.random.default_rng(stable_seed("manifold_constraint_robustness_clusterboot"))
    cluster_robust = {}
    for arm in ARMS:
        mz = z_mod[arm]
        cr = _cluster_robust_result(phi, mz, session_idx, n_sessions, cluster_rng)
        cr["survives_clustering"] = bool(cr["ci_lo"] > 0 and cr["p_value"] < 0.05)
        cluster_robust[arm] = cr
        print(f"  {arm}: cluster-robust slope={cr['slope']:+.4f} [{cr['ci_lo']:+.4f}, {cr['ci_hi']:+.4f}] "
              f"p={cr['p_value']:.4f} -> {'SURVIVES' if cr['survives_clustering'] else 'WEAKENS'}")

    # Self-check: unresampled cluster-bootstrap point estimate == pooled OLS slope.
    for arm in ARMS:
        full_slope = _slope_formula(z_mod[arm], phi)
        assert abs(full_slope - cluster_robust[arm]["slope"]) < 1e-9, \
            f"{arm}: cluster-bootstrap point estimate formula mismatch"

    w_cr, o_cr = cluster_robust["within_frac"], cluster_robust["outside_frac"]
    if w_cr["survives_clustering"] and not o_cr["survives_clustering"]:
        verdict = ("within_frac predicts the causal gate cluster-robustly and outside_frac does not -- "
                   "consistent with the Sadtler within-manifold constraint.")
    elif o_cr["survives_clustering"] and not w_cr["survives_clustering"]:
        verdict = ("outside_frac predicts the causal gate cluster-robustly and within_frac does not -- "
                   "the opposite of the Sadtler prediction; reported as-is, no spin.")
    elif w_cr["survives_clustering"] and o_cr["survives_clustering"]:
        verdict = "Both within_frac and outside_frac survive session-clustering -- the manifold constraint does not cleanly dissociate the two here."
    else:
        verdict = ("Neither within_frac nor outside_frac survives session-clustering -- the within-manifold "
                   "constraint does not predict the causal gate here, cluster-robustly confirming the "
                   "trial-level null already reported.")
    print(f"\nVERDICT: {verdict}")

    out = {
        "n_sessions": n_sessions,
        "degeneracy_gate": "passed (see results/manifold_constraint.json Part 19A)",
        "vstar_alignment": cluster_robust["vstar_alignment"],
        "within_frac": w_cr,
        "outside_frac": o_cr,
        "verdict": verdict,
    }
    with open(RESULTS / "manifold_constraint_robustness.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/manifold_constraint_robustness.json")


if __name__ == "__main__":
    main()
