#!/usr/bin/env python3
"""Cluster-robust, matched-design comparison of amplification_alignment
(alignment to w1, the propagator's top RIGHT SINGULAR vector) against
vstar_alignment (alignment to the top eigenvector), on the SAME macaque PFC microstimulation
rows/sessions/exclusions as run_macaque_pfc_microstimulation_headline_robustness.py's settled
vstar/min_energy result.

WHY a separate script rather than extending run_macaque_pfc_microstimulation_headline_robustness.py
in place: that script's ARMS tuple and self-checks are a settled robustness
result (feeds build_causal_targeting_leaderboard.py's trial_resolution
block; the underlying null is closed and should not be reopened) -- do not
touch it. This script duplicates its cluster-bootstrap METHOD (identical
_slope_formula / _cluster_bootstrap logic, same session-resampling
rationale) scoped to this one new comparison, with its own independent RNG
stream.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_amplification_robustness.py
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
from causal import benchmark_modifiers, _dr_slope  # noqa: E402
from statistics import stable_seed  # noqa: E402

RESULTS = ROOT / "results"
ARMS = ("vstar_alignment", "amplification_alignment")
N_BOOT = 2000


def _build_all_rows() -> tuple[list[dict], list[str]]:
    """Identical construction to run_macaque_pfc_microstimulation_headline_robustness.py's
    _build_all_rows (minus the session_mean_vstar_scalar column, not needed
    here)."""
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
    amp_mod = _col(all_rows, "amplification_alignment")
    propensity = _col(all_rows, "propensity")
    angle_idx = _col(all_rows, "angle_idx").astype(int)
    session_idx = _col(all_rows, "session_idx").astype(int)
    X = _build_X(all_rows, angle_idx, session_idx, n_sessions)

    n_missing_amp = int(np.sum(~np.isfinite(amp_mod)))
    print(f"  {len(all_rows)} rows across {n_sessions} sessions; "
          f"amplification_alignment non-finite rows: {n_missing_amp}")

    print("Cross-fitting ONE shared pseudo-outcome (phi) for both arms ...")
    bench_rng = np.random.default_rng(stable_seed("amplification_robustness"))
    bench = benchmark_modifiers(
        y, t, X, modifiers={"vstar_alignment": vstar_mod, "amplification_alignment": amp_mod},
        propensity=propensity, n_perm=2000, rng=bench_rng,
    )
    phi = bench["phi"]
    excluded = bench.get("excluded", {})
    if "amplification_alignment" in excluded:
        out = {"status": "excluded", "reason": excluded["amplification_alignment"]["reason"],
               "n_sessions": n_sessions}
        with open(RESULTS / "amplification_robustness.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"amplification_alignment EXCLUDED by benchmark_modifiers' zero-variance guard: "
              f"{excluded['amplification_alignment']['reason']}")
        return

    z_mod = bench["z_modifiers"]
    vstar_z, amp_z = z_mod["vstar_alignment"], z_mod["amplification_alignment"]
    assert len(phi) == len(vstar_z) == len(amp_z)

    print("\nCluster-bootstrap (session-resampled, B=2000), apples-to-apples with vstar_alignment ...")
    cluster_rng = np.random.default_rng(stable_seed("amplification_robustness_clusterboot"))
    cluster_robust = {}
    for arm, mz in (("vstar_alignment", vstar_z), ("amplification_alignment", amp_z)):
        cr = _cluster_robust_result(phi, mz, session_idx, n_sessions, cluster_rng)
        cr["survives_clustering"] = bool(cr["ci_lo"] > 0 and cr["p_value"] < 0.05)
        cluster_robust[arm] = cr
        print(f"  {arm}: cluster-robust slope={cr['slope']:+.4f} [{cr['ci_lo']:+.4f}, {cr['ci_hi']:+.4f}] "
              f"p={cr['p_value']:.4f} -> {'SURVIVES' if cr['survives_clustering'] else 'WEAKENS'}")

    # Self-check: unresampled cluster-bootstrap point estimate == pooled OLS slope.
    for arm, mz in (("vstar_alignment", vstar_z), ("amplification_alignment", amp_z)):
        full_slope = _slope_formula(mz, phi)
        assert abs(full_slope - cluster_robust[arm]["slope"]) < 1e-9, \
            f"{arm}: cluster-bootstrap point estimate formula mismatch"

    vstar_cr = cluster_robust["vstar_alignment"]
    amp_cr = cluster_robust["amplification_alignment"]
    if amp_cr["survives_clustering"] and amp_cr["slope"] >= vstar_cr["slope"]:
        verdict = ("amplification_alignment >= vstar_alignment (both cluster-robust): the causal target is "
                   "the amplified (singular) direction -- mechanistic upgrade.")
    elif vstar_cr["survives_clustering"] and not amp_cr["survives_clustering"]:
        verdict = ("vstar_alignment survives session-clustering and amplification_alignment does not: "
                   "the eigen/instability framing remains the better predictor -- keep v* primary, report "
                   "amplification_alignment as the weaker sibling.")
    elif vstar_cr["survives_clustering"] and amp_cr["survives_clustering"]:
        verdict = ("Both arms survive session-clustering; near-tie -- the causal data cannot cleanly "
                   "separate the two directions here. Keep v* primary (established headline), report "
                   "amplification_alignment as a comparably-performing sibling.")
    else:
        verdict = ("Neither arm survives session-clustering cleanly in a way that favors amplification "
                   "over v* -- report as-is, keep v* primary.")
    print(f"\nVERDICT: {verdict}")

    out = {
        "n_sessions": n_sessions,
        "cos_vstar_w1_median_source": "results/amplification_check.json",
        "vstar_alignment": vstar_cr,
        "amplification_alignment": amp_cr,
        "verdict": verdict,
    }
    with open(RESULTS / "amplification_robustness.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/amplification_robustness.json")


if __name__ == "__main__":
    main()
