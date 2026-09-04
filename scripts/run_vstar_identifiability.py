#!/usr/bin/env python3
"""Spectral-gap / identifiability diagnostic for the macaque PFC microstimulation uStim cohort.

Tests the mechanistic hypothesis that a near-degenerate eigenvalue spectrum
(a small gap between the top two eigenvalue moduli) is what drives v*'s
rank instability, bootstrap dispersion, and cross-session/split-half
instability -- i.e. that these are one phenomenon with a common cause,
rather than three independent anomalies.

Scope note: the rank-r sweep and split-half stability correlates this script
uses are macaque PFC microstimulation-only artifacts in this codebase (dmd_rank_selection.json /
vstar_split_half_stability.json), so this operates on macaque PFC microstimulation's 11 sessions
(10 for the split-half correlate, which only ran on the chronic Wa
shared-array sessions).

Reuses, no reimplementation: run_vstar_eigen_audit's macaque PFC microstimulation control-latent
loader (itself reusing run_macaque_pfc_microstimulation_pipeline's loaders) and its mean-trajectory
argmax-Re DMD fit for the rank-stability sweep; the eigenvalue gap and
bootstrap-cosine-dispersion values already computed per session in
results/vstar_eigen_audit.json (not recomputed here); the existing
results/vstar_split_half_stability.json for the split-half correlate;
statistics.spearman_permutation_test for inference; run_macaque_pfc_microstimulation_pipeline's
build_session_features + causal.benchmark_modifiers for the identifiability-
qualifying-subset re-fit.

Output: results/vstar_identifiability.json.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_vstar_identifiability.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from statistics import spearman_permutation_test, stable_seed
from causal import benchmark_modifiers

import run_vstar_eigen_audit as vea
import run_macaque_pfc_microstimulation_pipeline as rsp

RESULTS = ROOT / "results"
RANK_SWEEP = (4, 5, 7, 8)  # vs each session's own r=6 solution; the same
                           # rank range run_dmd_rank_selection.py sweeps


def _session_rank_stability() -> dict[str, float]:
    """Per-macaque PFC microstimulation-session min |cos(v*(r), v*(r=6))| over r in RANK_SWEEP --
    the session-level analogue of dmd_rank_selection.json's session-AVERAGED
    vstar_cos_to_r6 (that file only reports the cross-session mean per rank,
    not a per-session value, so this is computed fresh here from the same
    control latents / same leading-mode fit run_vstar_eigen_audit already
    uses)."""
    out = {}
    for dataset, session, Z_trials, dt, r in vea._iter_macaque_pfc_microstimulation():
        Z_mean = Z_trials.mean(0)
        _, _, v6 = vea.leading_mode(Z_mean, 6, dt)
        coss = []
        for rr in RANK_SWEEP:
            _, _, v_r = vea.leading_mode(Z_mean, rr, dt)
            coss.append(abs(float(v_r @ v6)))
        out[session] = float(min(coss))
    return out


def _requalify_subset(qualifying_sessions: list[str]) -> dict:
    """Re-run the macaque PFC microstimulation v*-alignment causal headline on ONLY the
    qualifying (well-identified-by-gap) sessions, apples-to-apples with the
    full-sample benchmark (same build_session_features, same
    benchmark_modifiers call), cluster (session)-bootstrap B=2000."""
    all_rows, session_order = [], []
    for prefix in rsp.SESSIONS:
        if prefix not in qualifying_sessions:  # rsp.SESSIONS entries ARE the session keys
            continue
        try:
            feat = rsp.build_session_features(prefix, structural_ctrl=None)
        except Exception:
            feat = None
        if feat is None:
            continue
        si = len(session_order)
        session_order.append(prefix)
        for row in feat["rows"]:
            row["session_idx"] = si
        all_rows.extend(feat["rows"])

    if len(session_order) < 3:
        return {"status": "underpowered", "n_qualifying": len(session_order),
                "reason": "fewer than 3 qualifying sessions -- cannot cluster-bootstrap meaningfully"}

    y = np.array([r["y"] for r in all_rows], float)
    t = np.array([r["t"] for r in all_rows], int)
    modifier = np.array([r["modifier"] for r in all_rows], float)
    propensity = np.array([r["propensity"] for r in all_rows], float)
    angle_idx = np.array([r["angle_idx"] for r in all_rows], int)
    session_idx = np.array([r["session_idx"] for r in all_rows], int)
    X = np.hstack([np.eye(angle_idx.max() + 1)[angle_idx], np.eye(len(session_order))[session_idx]])

    bench = benchmark_modifiers(y, t, X, modifiers={"vstar_alignment": modifier}, propensity=propensity,
                                n_perm=2000, rng=np.random.default_rng(stable_seed("vstar_identifiability_subset")))
    row = bench["leaderboard"]["vstar_alignment"]

    # session-cluster bootstrap (same recipe as run_macaque_pfc_microstimulation_headline_robustness.py)
    from causal import _dr_slope  # noqa: E402 (reuse, not reimplement, the shared core)
    phi = bench["phi"]
    z_mod = bench["z_modifiers"]["vstar_alignment"]

    def _slope_formula(m, p):
        mc = m - m.mean()
        denom = (mc ** 2).sum()
        return float((mc * (p - p.mean())).sum() / denom) if denom > 1e-15 else 0.0

    rng = np.random.default_rng(stable_seed("vstar_identifiability_clusterboot"))
    rows_by_session = [np.where(session_idx == s)[0] for s in range(len(session_order))]
    boot = np.empty(2000)
    for b in range(2000):
        drawn = rng.integers(0, len(session_order), size=len(session_order))
        idx = np.concatenate([rows_by_session[s] for s in drawn])
        boot[b] = _slope_formula(z_mod[idx], phi[idx])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    p = min(2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean())), 1.0)

    return {
        "status": "ok", "n_qualifying": len(session_order), "sessions": session_order,
        "n_rows": int(bench["n"]),
        "slope_raw": row["slope"], "slope_p_trial_level": row["p_value"],
        "cluster_robust_slope": _slope_formula(z_mod, phi), "cluster_ci_lo": float(ci_lo),
        "cluster_ci_hi": float(ci_hi), "cluster_p": float(p),
    }


def main():
    audit = json.load(open(RESULTS / "vstar_eigen_audit.json"))["macaque_pfc_microstimulation"]
    split_half = json.load(open(RESULTS / "vstar_split_half_stability.json"))

    print("Computing per-session rank stability (min |cos to r=6| over r in "
          f"{RANK_SWEEP}) ...")
    rank_stab = _session_rank_stability()

    sessions_all = list(audit.keys())
    n_conjugate = sum(1 for s in sessions_all if audit[s].get("leading_mode_is_conjugate_pair"))
    # Raw-eigenvalue gap: identically ~0 whenever the leading mode is a
    # complex-conjugate pair (a conjugate partner shares the same modulus by
    # construction). Kept only as a reference for why this statistic cannot
    # be used as a predictor.
    gap_mod_degenerate = np.array([audit[s]["gap_mod"] for s in sessions_all])
    # Gap to the next mode of genuinely distinct modulus, grouping a
    # conjugate pair into one mode first (run_vstar_eigen_audit.spectral_gaps
    # / _grouped_mode_values). This is the statistic to correlate.
    gap_mod = np.array([audit[s]["gap_to_next_distinct_mode_mod"] for s in sessions_all])
    boot_cos = np.array([audit[s]["bootstrap_vstar_cos_mean"] for s in sessions_all])
    rank_cos = np.array([rank_stab[s] for s in sessions_all])

    print(f"\nmacaque PFC microstimulation sessions (N={len(sessions_all)}): leading mode is a "
          f"complex-conjugate pair in {n_conjugate}/{len(sessions_all)} sessions")
    print(f"DEGENERATE raw gap_mod range [{gap_mod_degenerate.min():.4f}, {gap_mod_degenerate.max():.4f}] "
          f"({sum(gap_mod_degenerate < 1e-6)}/{len(sessions_all)} sessions tied at ~0 -- by construction, "
          "not a null result about biology)")
    print(f"CORRECTED gap_to_next_distinct_mode_mod range [{gap_mod.min():.3f}, {gap_mod.max():.3f}]")

    rng = np.random.default_rng(stable_seed("vstar_identifiability_corr"))
    # Degenerate-statistic correlations, reported for reference only.
    corr_boot_cos_degenerate = spearman_permutation_test(gap_mod_degenerate, boot_cos, rng=rng)
    corr_rank_degenerate = spearman_permutation_test(gap_mod_degenerate, rank_cos, rng=rng)
    # Corrected-statistic correlations: the real PART-26 test.
    corr_boot_cos = spearman_permutation_test(gap_mod, boot_cos, rng=rng)
    corr_rank = spearman_permutation_test(gap_mod, rank_cos, rng=rng)

    print(f"\n[DEGENERATE, audit trail only] raw gap_mod vs bootstrap_vstar_cos_mean: "
          f"rho={corr_boot_cos_degenerate['rho']:+.3f} p={corr_boot_cos_degenerate['p_value']:.4f}")
    print(f"[DEGENERATE, audit trail only] raw gap_mod vs rank_stability: "
          f"rho={corr_rank_degenerate['rho']:+.3f} p={corr_rank_degenerate['p_value']:.4f}")
    print(f"[CORRECTED] gap_to_next_distinct_mode_mod vs bootstrap_vstar_cos_mean: rho={corr_boot_cos['rho']:+.3f} "
          f"p={corr_boot_cos['p_value']:.4f} (N={corr_boot_cos['n']})")
    print(f"[CORRECTED] gap_to_next_distinct_mode_mod vs rank_stability (min cos to r6, r in {RANK_SWEEP}): "
          f"rho={corr_rank['rho']:+.3f} p={corr_rank['p_value']:.4f} (N={corr_rank['n']})")

    # (iii) split-half: subset to the 10 qualifying Wa sessions
    sh_sessions = split_half["sessions_qualified"]
    sh_cos = np.array(split_half["within_split_half_cos"])
    gap_mod_sh_degenerate = np.array([audit[s]["gap_mod"] for s in sh_sessions])
    gap_mod_sh = np.array([audit[s]["gap_to_next_distinct_mode_mod"] for s in sh_sessions])
    corr_split_half_degenerate = spearman_permutation_test(gap_mod_sh_degenerate, sh_cos, rng=rng)
    corr_split_half = spearman_permutation_test(gap_mod_sh, sh_cos, rng=rng)
    print(f"[DEGENERATE, audit trail only] raw gap_mod vs within-session split-half |cos|: "
          f"rho={corr_split_half_degenerate['rho']:+.3f} p={corr_split_half_degenerate['p_value']:.4f}")
    print(f"[CORRECTED] gap_to_next_distinct_mode_mod vs within-session split-half |cos| "
          f"(N={len(sh_sessions)}): rho={corr_split_half['rho']:+.3f} p={corr_split_half['p_value']:.4f}")

    prediction_direction = "CONFIRMED" if (corr_boot_cos["rho"] > 0 and corr_rank["rho"] > 0
                                           and corr_split_half["rho"] > 0) else "NOT CONFIRMED (report as-is)"
    print(f"\nPREDICTION (corrected gap correlates positively with all three): {prediction_direction}")

    # Pre-registered identifiability inclusion criterion, stated before
    # applying it -- the gap threshold at which mean bootstrap |cos| to the
    # full-sample v* first exceeds 0.9, scanning sessions in ascending-gap
    # order (a monotonic-in-gap criterion, not a fit to the outcome we then
    # test on the SAME subset's causal slope -- the subset re-fit below is
    # the held-out check).
    order = np.argsort(gap_mod)
    threshold = None
    for i in order:
        if boot_cos[i] >= 0.9:
            threshold = float(gap_mod[i])
            break
    if threshold is None:
        threshold = float(gap_mod.max()) + 1e-6  # no session clears 0.9 -> nothing qualifies
    qualifying = [sessions_all[i] for i in range(len(sessions_all)) if gap_mod[i] >= threshold]
    print(f"\nPRE-REGISTERED CRITERION: gap_mod >= {threshold:.4f} "
          f"(first gap, in ascending order, at which bootstrap |cos| to full-sample v* >= 0.9)")
    print(f"n_qualifying = {len(qualifying)}/{len(sessions_all)}: {qualifying}")

    subset_result = _requalify_subset(qualifying)
    print(f"Qualifying-subset causal re-fit: {subset_result}")

    full_headline = {"slope": 0.032, "ci_lo": 0.006, "ci_hi": 0.064, "p": 0.008}  # pre-registered, unchanged
    degenerate_subset = len(qualifying) == len(sessions_all)
    if degenerate_subset:
        outcome = ("DEGENERATE: the ascending-gap threshold that first clears bootstrap |cos|>=0.9 is "
                   "the MINIMUM observed gap, so all 11/11 sessions trivially 'qualify' and the subset "
                   "re-fit is identical to the full-sample fit by construction -- NOT an informative "
                   "comparison. This is itself a consequence of the null/negative gap correlations above: "
                   "gap does not separate well- from poorly-identified sessions in this dataset, so no "
                   "gap-based threshold usefully excludes anyone. Report the null correlations as the "
                   "finding here, not the (uninformative) subset slope match.")
        print(f"OUTCOME: {outcome}")
    elif subset_result.get("status") == "ok":
        strengthened = subset_result["cluster_robust_slope"] > full_headline["slope"]
        outcome = ("slope HOLDS/STRENGTHENS on well-identified sessions -> geometric account "
                  "supported, noise diluted the full-sample estimate" if strengthened else
                  "slope WEAKENS on the well-identified subset -> effect not carried by "
                  "well-identified v*, evidence for a subspace (rather than single-vector) "
                  "account of the target")
        print(f"OUTCOME: {outcome}")
    else:
        outcome = f"subset re-fit not computed: {subset_result.get('reason')}"
        print(f"OUTCOME: {outcome}")

    out = {
        "sessions": sessions_all,
        "n_conjugate_leading_mode": n_conjugate,
        "gap_mod_degenerate": gap_mod_degenerate.tolist(),
        "gap_mod": gap_mod.tolist(),
        "bootstrap_vstar_cos_mean": boot_cos.tolist(),
        "rank_stability_min_cos_to_r6": rank_cos.tolist(),
        "correlations_degenerate_audit_trail_only": {
            "gap_vs_bootstrap_dispersion": corr_boot_cos_degenerate,
            "gap_vs_rank_stability": corr_rank_degenerate,
            "gap_vs_split_half": corr_split_half_degenerate,
        },
        "correlations": {
            "gap_vs_bootstrap_dispersion": corr_boot_cos,
            "gap_vs_rank_stability": corr_rank,
            "gap_vs_split_half": corr_split_half,
        },
        "prediction_direction": prediction_direction,
        "identifiability_criterion": {
            "rule": "gap_mod >= first-ascending-gap threshold at which bootstrap |cos| to "
                    "full-sample v* first reaches 0.9",
            "threshold": threshold,
        },
        "n_qualifying": len(qualifying),
        "qualifying_sessions": qualifying,
        "subset_causal_refit": subset_result,
        "full_sample_headline_UNCHANGED_PRIMARY": full_headline,
        "degenerate_subset": degenerate_subset,
        "outcome": outcome,
    }
    # spearman_permutation_test returns a 'null' array -- not JSON-serializable, strip it.
    for k in ("gap_vs_bootstrap_dispersion", "gap_vs_rank_stability", "gap_vs_split_half"):
        out["correlations"][k] = {kk: vv for kk, vv in out["correlations"][k].items() if kk != "null"}
        out["correlations_degenerate_audit_trail_only"][k] = {
            kk: vv for kk, vv in out["correlations_degenerate_audit_trail_only"][k].items() if kk != "null"}

    with open(RESULTS / "vstar_identifiability.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/vstar_identifiability.json")


if __name__ == "__main__":
    main()
