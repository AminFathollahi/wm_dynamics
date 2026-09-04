#!/usr/bin/env python3
"""Does the v* target transfer across sessions/days?

Feasibility gate (verified): the Wa monkey's 10 sessions
(Wa220801_s549 .. Wa220812_s558, consecutive days in Aug 2022) all report
IDENTICAL channel_ids = {1..96} from run_macaque_pfc_microstimulation_pipeline.load_macaque_pfc_microstimulation_session
-- a single chronic 96-channel Utah array (consistent with the dataset
docstring's "single-channel stim, 96 ch (1 array)"), not per-day re-sorted
units. The recording BASIS is shared -- gate PASSES. Sa (1 session, a
different, incompatible-format array) is excluded: it cannot be held out
(a single cluster) and is not part of the transfer pool.

18B: work in the SHARED CHANNEL space (not each session's own latent
rotation). For each Wa session, lift its own control-epoch-fit v* back to
channel space: v*_chan = V @ v*_latent (V: (C,k) PCA loadings, exactly what
build_session_features fits internally -- NOT redefined here, this script
duplicates only the same PCA+DMD+eigenvector fit already reused by
run_amplification_check.py). Leave-one-session-out: for held-out session s,
the transfer target is the unit-norm mean of v*_chan over the OTHER 9 Wa
sessions; project it into s's own latent space (V_s.T @ transfer_chan) and
score alignment(delivered stim direction, transfer target) in s's own
latent space -- consistent with how alignment_to_vstar is scored elsewhere
in this project (always in the stimulating session's own latent).

Score with the SAME cate_vs_modifier_slope/benchmark_modifiers machinery as
every other arm (src/causal.py untouched); compare, apples-to-apples on the
10 held-out Wa sessions, against each session's OWN-session vstar_alignment
slope already in results/macaque_pfc_microstimulation_headline_robustness.json's per_session
block. Aggregate with a cluster/bootstrap over the 10 held-out sessions
(not a trial-level p).

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_target_transfer.py
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

from geometry import pca_decompose
from dynamics import dmd_reconstruction_error
from control import dominant_eigenmode
from causal import benchmark_modifiers, _dr_slope
from statistics import stable_seed

sys.path.insert(0, str(ROOT / "scripts"))
from run_macaque_pfc_microstimulation_pipeline import (
    load_macaque_pfc_microstimulation_session, crop_trial, SESSIONS, N_PC, DMD_RANK, N_BINS, BIN_S,
    build_session_features,
)

RESULTS = ROOT / "results"
WA_SESSIONS = [s for s in SESSIONS if s.startswith("Wa")]
N_BOOT = 2000


def _channel_basis_check() -> bool:
    """18A gate: do the Wa sessions share channel_ids?"""
    chan_sets = {}
    for s in WA_SESSIONS:
        d = load_macaque_pfc_microstimulation_session(s, correct=True)
        if d is None:
            return False
        chan_sets[s] = tuple(sorted(d["channel_ids"].tolist()))
    ref = chan_sets[WA_SESSIONS[0]]
    return all(chan_sets[s] == ref for s in WA_SESSIONS)


def _fit_session_v_and_V(prefix: str) -> dict | None:
    """Same PCA+DMD+eigenvector fit as build_session_features (Part 1), plus
    V and v_star returned (not exposed by build_session_features itself)."""
    corr = load_macaque_pfc_microstimulation_session(prefix, correct=True)
    if corr is None or corr["control_idx"] is None:
        return None
    control_idx = corr["control_idx"]
    channel_ids = corr["channel_ids"]
    C = len(channel_ids)

    ctrl_epochs = [
        crop_trial(tr["spikerate"]) for tr in corr["trials"] if tr["stim_cond"] == control_idx
    ]
    ctrl_epochs = [e for e in ctrl_epochs if e is not None]
    if len(ctrl_epochs) < 10:
        return None

    Z_ctrl = np.stack(ctrl_epochs, axis=0)
    X_flat = Z_ctrl.reshape(-1, C)
    _, V, _ = pca_decompose(X_flat, N_PC)  # V: (C, k)
    k = V.shape[1]
    Z_ctrl_mean = ((Z_ctrl.reshape(-1, C) - X_flat.mean(0)) @ V).reshape(
        Z_ctrl.shape[0], N_BINS, k).mean(0)
    r_use = min(DMD_RANK, k, N_BINS - 2)
    dmd = dmd_reconstruction_error(Z_ctrl_mean, r=r_use, dt=BIN_S)
    A = dmd["A"]
    v_star = dominant_eigenmode(A).v_star

    return {"V": V, "v_star": v_star, "channel_ids": channel_ids}


def _slope_formula(m: np.ndarray, phi: np.ndarray) -> float:
    mc = m - m.mean()
    denom = (mc ** 2).sum()
    if denom < 1e-15:
        return 0.0
    return float((mc * (phi - phi.mean())).sum() / denom)


def _cluster_bootstrap_over_sessions(per_session_slope: np.ndarray, n_boot: int,
                                      rng: np.random.Generator) -> dict:
    """Cluster/bootstrap over the 10 held-out sessions (spec: not a trial-
    level p) -- resample sessions with replacement, recompute the MEAN
    transfer slope each draw (a session-level statistic, since each held-out
    session already contributes exactly one transfer-modifier slope)."""
    n = len(per_session_slope)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = per_session_slope[idx].mean()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    mean_slope = float(per_session_slope.mean())
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    p = min(p, 1.0)
    return {"mean": mean_slope, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "p_value": p, "n_boot": n_boot}


def main() -> None:
    print("18A feasibility gate: do the 10 Wa sessions share channel_ids?")
    basis_shared = _channel_basis_check()
    print(f"  basis_shared = {basis_shared}")
    if not basis_shared:
        out = {"status": "infeasible", "reason": "non-shared basis across Wa sessions",
               "basis_shared": False}
        with open(RESULTS / "target_transfer.json", "w") as f:
            json.dump(out, f, indent=2)
        print("GATE FAILED -- wrote infeasible status, stopping.")
        return
    print("GATE PASSED -- all 10 Wa sessions report identical channel_ids "
          "(chronic 96-ch array). Proceeding to 18B.\n")

    print("Fitting per-session (V, v_star) for all Wa sessions ...")
    fits = {}
    for s in WA_SESSIONS:
        r = _fit_session_v_and_V(s)
        if r is None:
            print(f"  {s} SKIP (insufficient data)")
            continue
        fits[s] = r
        print(f"  {s}: k={r['V'].shape[1]}, C={r['V'].shape[0]}")

    if len(fits) < 3:
        out = {"status": "infeasible", "reason": f"only {len(fits)} usable Wa sessions, too few to leave-one-out",
               "basis_shared": True}
        with open(RESULTS / "target_transfer.json", "w") as f:
            json.dump(out, f, indent=2)
        print("Too few usable sessions -- stopping.")
        return

    v_star_chan = {s: fits[s]["V"] @ fits[s]["v_star"] for s in fits}  # (C,) each, C shared=96

    with open(RESULTS / "macaque_pfc_microstimulation_headline_robustness.json") as f:
        robustness = json.load(f)
    own_per_session = robustness["per_session"]

    print("\nLeave-one-session-out transfer scoring ...")
    per_session_result = {}
    own_slopes, transfer_slopes = [], []
    boot_rng = np.random.default_rng(stable_seed("target_transfer"))
    for held_out in fits:
        others = [s for s in fits if s != held_out]
        mean_chan = np.mean([v_star_chan[s] for s in others], axis=0)
        transfer_target_chan = mean_chan / (np.linalg.norm(mean_chan) + 1e-12)
        # project into held_out's own latent space, then score alignment there
        # (consistent with how alignment_to_vstar is scored everywhere else
        # in this project: in the stimulating session's own latent).
        V_ho = fits[held_out]["V"]
        transfer_target_latent = V_ho.T @ transfer_target_chan
        transfer_target_latent /= (np.linalg.norm(transfer_target_latent) + 1e-12)

        feat = build_session_features(held_out, structural_ctrl=None)
        if feat is None:
            print(f"  {held_out} SKIP (build_session_features returned None)")
            continue

        # Recompute alignment-to-transfer-target per condition using the SAME
        # B_hat_unit construction build_session_features already used
        # internally for alignment_to_vstar (re-derive B_hat_unit from the
        # session's own cond_features via B_lat recovered geometrically is
        # not exposed -- instead reload the raw stim-channel one-hot exactly
        # as build_session_features does, in the SAME latent space V_ho).
        corr = load_macaque_pfc_microstimulation_session(held_out, correct=True)
        control_idx = corr["control_idx"]
        channel_ids = corr["channel_ids"]
        C = len(channel_ids)
        n_cond = len(corr["stim_channels"])
        transfer_align_by_cond = {}
        for c in range(n_cond):
            if c == control_idx or str(c) not in feat["cond_features"]:
                continue
            chan_ids = corr["stim_channels"][c]
            idx = [i for i, cid in enumerate(channel_ids) if cid in chan_ids]
            if not idx:
                continue
            B_chan = np.zeros((C, 1))
            B_chan[idx, 0] = 1.0 / len(idx)
            B_lat = V_ho.T @ B_chan
            B_hat_unit = B_lat[:, 0] / (np.linalg.norm(B_lat) + 1e-12)
            transfer_align_by_cond[c] = float(np.abs(B_hat_unit @ transfer_target_latent))

        # Build the same row table build_session_features's main() loop
        # builds, but with the transfer-alignment modifier substituted for
        # alignment_to_vstar.
        # feat["rows"] rows are tagged with own-session alignment_to_vstar as
        # "modifier" but not the condition id itself, so re-derive rows
        # directly the same way build_session_features does (Part 3 of that
        # function), substituting transfer_align_by_cond for alignment_to_vstar.
        rows = []
        def _epochs_for(cond_source, cond, label_correct):
            out = []
            for tr in cond_source["trials"]:
                if tr["stim_cond"] != cond:
                    continue
                cropped = crop_trial(tr["spikerate"])
                if cropped is not None:
                    out.append((cropped, label_correct, tr["angle_idx"]))
            return out

        err = load_macaque_pfc_microstimulation_session(held_out, correct=False)
        ctrl_all = _epochs_for(corr, control_idx, 1) + (
            _epochs_for(err, control_idx, 0) if err is not None else [])
        for c, talign in transfer_align_by_cond.items():
            stim_all = _epochs_for(corr, c, 1) + (_epochs_for(err, c, 0) if err is not None else [])
            if len(stim_all) < 5 or len(ctrl_all) < 5:
                continue
            n_stim, n_ctrl = len(stim_all), len(ctrl_all)
            propensity = n_stim / (n_stim + n_ctrl)
            for _, y, angle_idx in stim_all:
                rows.append({"y": y, "t": 1, "modifier": talign, "propensity": propensity, "angle_idx": angle_idx})
            for _, y, angle_idx in ctrl_all:
                rows.append({"y": y, "t": 0, "modifier": talign, "propensity": propensity, "angle_idx": angle_idx})

        if len(rows) < 20:
            print(f"  {held_out} SKIP (only {len(rows)} rows)")
            continue

        y = np.array([r["y"] for r in rows], dtype=float)
        t = np.array([r["t"] for r in rows], dtype=int)
        modifier = np.array([r["modifier"] for r in rows], dtype=float)
        propensity = np.array([r["propensity"] for r in rows], dtype=float)
        angle_idx = np.array([r["angle_idx"] for r in rows], dtype=int)
        X = np.eye(angle_idx.max() + 1)[angle_idx]

        sess_rng = np.random.default_rng(stable_seed(f"target_transfer_{held_out}"))
        bench = benchmark_modifiers(y, t, X, modifiers={"transfer_alignment": modifier},
                                    propensity=propensity, n_perm=2000, rng=sess_rng)
        if "transfer_alignment" in bench.get("excluded", {}):
            print(f"  {held_out} EXCLUDED: {bench['excluded']['transfer_alignment']['reason']}")
            continue
        row = bench["leaderboard"]["transfer_alignment"]
        own_slope = own_per_session[held_out]["vstar_alignment"]["slope"]
        per_session_result[held_out] = {
            "transfer_slope": row["slope"], "transfer_ci_lo": row["slope_ci_lo"],
            "transfer_ci_hi": row["slope_ci_hi"], "transfer_p": row["p_value"], "n": row["n"],
            "own_session_vstar_slope": own_slope,
        }
        own_slopes.append(own_slope)
        transfer_slopes.append(row["slope"])
        print(f"  {held_out}: transfer_slope={row['slope']:+.4f} p={row['p_value']:.4f}  "
              f"vs own_slope={own_slope:+.4f}")

    if len(transfer_slopes) < 3:
        out = {"status": "infeasible", "reason": f"only {len(transfer_slopes)} held-out sessions scored, too few for a cluster bootstrap",
               "basis_shared": True, "per_session": per_session_result}
        with open(RESULTS / "target_transfer.json", "w") as f:
            json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
        print("Too few scored held-out sessions -- stopping.")
        return

    transfer_arr = np.array(transfer_slopes)
    own_arr = np.array(own_slopes)
    print("\nCluster/bootstrap over the held-out sessions (B=2000) ...")
    transfer_cluster = _cluster_bootstrap_over_sessions(transfer_arr, N_BOOT, boot_rng)
    own_cluster = _cluster_bootstrap_over_sessions(own_arr, N_BOOT, boot_rng)
    retention = (transfer_cluster["mean"] / own_cluster["mean"]) if abs(own_cluster["mean"]) > 1e-9 else float("nan")

    print(f"  transfer: mean={transfer_cluster['mean']:+.4f} CI[{transfer_cluster['ci_lo']:+.4f},"
          f"{transfer_cluster['ci_hi']:+.4f}] p={transfer_cluster['p_value']:.4f}")
    print(f"  own:      mean={own_cluster['mean']:+.4f} CI[{own_cluster['ci_lo']:+.4f},"
          f"{own_cluster['ci_hi']:+.4f}] p={own_cluster['p_value']:.4f}")
    print(f"  retention (transfer/own) = {retention:.3f}")

    transfer_positive_and_sig = transfer_cluster["ci_lo"] > 0 and transfer_cluster["p_value"] < 0.05
    if transfer_positive_and_sig and retention > 0.5:
        verdict = ("Transfer slope is significantly positive and comparable to own-session slope "
                   f"(retention={retention:.2f}) -- the target is stable/pre-computable across days "
                   "(translational positive).")
    elif transfer_positive_and_sig:
        verdict = (f"Transfer slope is significantly positive but small relative to own-session "
                   f"(retention={retention:.2f}) -- partial transfer.")
    else:
        verdict = ("Transfer slope is not reliably positive under session-level bootstrap "
                   "(does not survive) -- the target is session-idiosyncratic; a fixed, "
                   "pre-computed cross-session target does not reproduce the own-session effect "
                   "(honest limitation).")
    print(f"\nVERDICT: {verdict}")

    out = {
        "basis_shared": True,
        "n_sessions_wa": len(fits),
        "n_sessions_scored": len(transfer_slopes),
        "own_vs_transfer": per_session_result,
        "transfer_slope_mean": transfer_cluster["mean"],
        "transfer_ci": [transfer_cluster["ci_lo"], transfer_cluster["ci_hi"]],
        "transfer_p": transfer_cluster["p_value"],
        "own_slope_mean": own_cluster["mean"],
        "own_ci": [own_cluster["ci_lo"], own_cluster["ci_hi"]],
        "retention": retention,
        "alignment_space_note": ("transfer target built as unit-norm mean of v*_chan (V @ v*_latent) over "
                                 "the other 9 Wa sessions in SHARED CHANNEL space, then projected into the "
                                 "held-out session's own latent space (V_ho.T @ transfer_chan) for scoring -- "
                                 "consistent with how alignment_to_vstar is scored everywhere else in this "
                                 "project (in the stimulating session's own latent)."),
        "verdict": verdict,
    }
    with open(RESULTS / "target_transfer.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/target_transfer.json")


if __name__ == "__main__":
    main()
