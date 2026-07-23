#!/usr/bin/env python3
"""Round-11 PART 19 (optional): Sadtler et al. 2014-style within- vs
outside-manifold constraint, tested causally on Soldado microstimulation.

WHY: Sadtler 2014 (Nature) + Golub 2018 + Oby 2019 -- BCI perturbations
WITHIN the intrinsic neural manifold are effective, OUTSIDE are not -- has
never been tested with delivered microstim in a WM setting. v* is
within-manifold BY CONSTRUCTION (it lives in the same k-dim PCA subspace the
DMD plant is fit in), so this could explain v*'s causal leverage at a
deeper level.

19A (degeneracy gate): the delivered stim must be scored in the FULL CHANNEL
space, not the already-latent-projected B (B_lat = V.T @ B_chan is
within-manifold by construction -> testing within_frac on B_lat would be
degenerate/circular). e_hat = B_chan (the raw one-hot stimulated-electrode
direction in channel space, exactly what run_soldado_pipeline.build_session_
features constructs BEFORE projecting into the latent space -- reused here,
not redefined). P_k = V @ V.T (V: (C,k) PCA loadings of control-epoch
maintenance activity, orthonormal columns from pca_decompose's full-SVD
construction -- src/geometry.py) is the k-dim WM-manifold projector.
    within_frac  = ||P_k @ e_hat|| / ||e_hat||
    outside_frac = ||(I-P_k) @ e_hat|| / ||e_hat||
DEGENERACY GATE: if within_frac has ~zero variance across conditions
(nanstd < 1e-9, pooled across all Soldado conditions/sessions), the arm is
uninformative -- STOP, report honestly, do not force an alternative.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_manifold_constraint.py
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
from causal import benchmark_modifiers
from statistics import stable_seed

sys.path.insert(0, str(ROOT / "scripts"))
from run_soldado_pipeline import load_soldado_session, crop_trial, SESSIONS, N_PC, N_BINS

RESULTS = ROOT / "results"


def _session_within_outside(prefix: str) -> dict | None:
    """Re-fit ONLY the PCA projector (P_k) -- the DMD/v* fit is irrelevant to
    this arm -- and score within_frac/outside_frac for every stim condition's
    RAW channel-space direction e_hat, exactly as build_session_features
    constructs B_chan (Part 2) before it ever projects into the latent
    space."""
    corr = load_soldado_session(prefix, correct=True)
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
    _, V, _ = pca_decompose(X_flat, N_PC)  # V: (C, k), orthonormal columns
    Pk = V @ V.T  # (C, C) projector onto the k-dim WM manifold

    n_cond = len(corr["stim_channels"])
    cond_fracs = {}
    for c in range(n_cond):
        if c == control_idx:
            continue
        chan_ids = corr["stim_channels"][c]
        idx = [i for i, cid in enumerate(channel_ids) if cid in chan_ids]
        if not idx:
            continue
        e_hat = np.zeros(C)
        e_hat[idx] = 1.0 / len(idx)
        e_norm = np.linalg.norm(e_hat)
        if e_norm < 1e-12:
            continue
        within = float(np.linalg.norm(Pk @ e_hat) / e_norm)
        outside = float(np.linalg.norm(e_hat - Pk @ e_hat) / e_norm)
        cond_fracs[c] = {"within_frac": within, "outside_frac": outside}

    return {"cond_fracs": cond_fracs, "n_pc": V.shape[1]}


def main() -> None:
    print("Fitting P_k (k-dim WM-manifold projector) and scoring within_frac/outside_frac "
          "per Soldado session/condition (raw channel-space e_hat) ...")
    per_session = {}
    all_within, all_outside = [], []
    for prefix in SESSIONS:
        r = _session_within_outside(prefix)
        if r is None:
            print(f"  {prefix} SKIP (insufficient data)")
            continue
        per_session[prefix] = r
        for c, f in r["cond_fracs"].items():
            all_within.append(f["within_frac"])
            all_outside.append(f["outside_frac"])
        print(f"  {prefix}: {len(r['cond_fracs'])} conditions, "
              f"within_frac={np.mean([f['within_frac'] for f in r['cond_fracs'].values()]):.4f}")

    all_within = np.array(all_within)
    all_outside = np.array(all_outside)
    within_std = float(np.nanstd(all_within))
    print(f"\nPooled across {len(all_within)} conditions: "
          f"within_frac mean={all_within.mean():.4f} std={within_std:.6g}, "
          f"outside_frac mean={all_outside.mean():.4f}")

    if within_std < 1e-9:
        out = {
            "status": "excluded_degenerate",
            "reason": (f"within_frac has ~zero variance across the {len(all_within)} scored Soldado "
                      f"conditions (nanstd={within_std:.3g} < 1e-9) -- the delivered stim set does not "
                      "vary in manifold-overlap on this dataset (every stimulated electrode direction "
                      "projects onto the k-dim control-epoch PCA manifold by a nearly-fixed fraction), "
                      "so the arm is uninformative. This is a property of the STIMULATION SET (fixed "
                      "single-electrode one-hot directions on a shared array), not a coding error. "
                      "STOPPING per the pre-registered degeneracy gate -- reported honestly, no "
                      "alternative arm substituted."),
            "n_sessions": len(per_session),
            "n_conditions_scored": int(len(all_within)),
            "within_frac_mean": float(all_within.mean()),
            "within_frac_std": within_std,
            "per_session": per_session,
        }
        with open(RESULTS / "manifold_constraint.json", "w") as f:
            json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
        print("\nDEGENERACY GATE TRIPPED -- wrote excluded_degenerate status, stopping (no 19B).")
        return

    print("\nDegeneracy gate PASSED (within_frac varies across conditions) -- proceeding to 19B.")
    # 19B: score within_frac/outside_frac as causal-targeting modifiers,
    # apples-to-apples with vstar_alignment (same rows/sessions/exclusions).
    # Rows carry alignment_to_vstar as "modifier" but not the raw condition id,
    # so within_frac/outside_frac are substituted in condition-by-condition,
    # mirroring build_session_features's own Part 3 row construction exactly.
    session_order = [p for p in SESSIONS if p in per_session]
    def _epochs_for(cond_source, cond, label_correct):
        out = []
        for tr in cond_source["trials"]:
            if tr["stim_cond"] != cond:
                continue
            cropped = crop_trial(tr["spikerate"])
            if cropped is not None:
                out.append((cropped, label_correct, tr["angle_idx"]))
        return out

    rows = []
    session_idx_ctr = 0
    for prefix in session_order:
        corr = load_soldado_session(prefix, correct=True)
        err = load_soldado_session(prefix, correct=False)
        control_idx = corr["control_idx"]
        fracs = per_session[prefix]["cond_fracs"]
        ctrl_all = _epochs_for(corr, control_idx, 1) + (_epochs_for(err, control_idx, 0) if err is not None else [])
        for c, f in fracs.items():
            stim_all = _epochs_for(corr, c, 1) + (_epochs_for(err, c, 0) if err is not None else [])
            if len(stim_all) < 5 or len(ctrl_all) < 5:
                continue
            n_stim, n_ctrl = len(stim_all), len(ctrl_all)
            propensity = n_stim / (n_stim + n_ctrl)
            for _, y, angle_idx in stim_all:
                rows.append({"y": y, "t": 1, "within_frac": f["within_frac"], "outside_frac": f["outside_frac"],
                            "propensity": propensity, "angle_idx": angle_idx, "session_idx": session_idx_ctr})
            for _, y, angle_idx in ctrl_all:
                rows.append({"y": y, "t": 0, "within_frac": f["within_frac"], "outside_frac": f["outside_frac"],
                            "propensity": propensity, "angle_idx": angle_idx, "session_idx": session_idx_ctr})
        session_idx_ctr += 1

    y = np.array([r["y"] for r in rows], dtype=float)
    t = np.array([r["t"] for r in rows], dtype=int)
    within_mod = np.array([r["within_frac"] for r in rows], dtype=float)
    outside_mod = np.array([r["outside_frac"] for r in rows], dtype=float)
    propensity = np.array([r["propensity"] for r in rows], dtype=float)
    angle_idx = np.array([r["angle_idx"] for r in rows], dtype=int)
    session_idx = np.array([r["session_idx"] for r in rows], dtype=int)
    angle_oh = np.eye(angle_idx.max() + 1)[angle_idx]
    session_oh = np.eye(session_idx.max() + 1)[session_idx]
    X = np.hstack([angle_oh, session_oh])

    print(f"\nScoring within_frac/outside_frac on N={len(rows)} rows, "
          f"{len(session_order)} sessions (same construction as vstar_alignment) ...")
    bench_rng = np.random.default_rng(stable_seed("manifold_constraint"))
    bench = benchmark_modifiers(y, t, X, modifiers={"within_frac": within_mod, "outside_frac": outside_mod},
                                propensity=propensity, n_perm=5000, rng=bench_rng)
    excluded = bench.get("excluded", {})
    result = {"n": len(rows), "n_sessions": len(session_order)}
    for arm in ("within_frac", "outside_frac"):
        if arm in excluded:
            result[arm] = {"eligible": False, "reason": excluded[arm]["reason"]}
        else:
            result[arm] = bench["leaderboard"][arm]
    print(json.dumps(result, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o))

    verdict = "excluded" if ("within_frac" in excluded or "outside_frac" in excluded) else None
    if verdict is None:
        w_sig = result["within_frac"]["p_value"] < 0.05 and result["within_frac"]["slope"] > 0
        o_sig = result["outside_frac"]["p_value"] < 0.05 and result["outside_frac"]["slope"] > 0
        if w_sig and not o_sig:
            verdict = ("within_frac predicts the causal gate and outside_frac does not -- consistent with "
                      "the Sadtler within-manifold constraint.")
        elif o_sig and not w_sig:
            verdict = ("outside_frac predicts the causal gate and within_frac does not -- the opposite of "
                      "the Sadtler prediction; reported as-is, no spin.")
        elif w_sig and o_sig:
            verdict = "Both within_frac and outside_frac predict the causal gate -- the manifold constraint does not cleanly dissociate the two here."
        else:
            verdict = "Neither within_frac nor outside_frac predicts the causal gate at trial level."
    print(f"\nVERDICT: {verdict}")

    out = {
        "status": "scored",
        "degeneracy_gate": "passed",
        "within_frac_pooled_mean": float(all_within.mean()),
        "within_frac_pooled_std": within_std,
        "n_sessions": len(session_order),
        "result": result,
        "verdict": verdict,
    }
    with open(RESULTS / "manifold_constraint.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/manifold_constraint.json")


if __name__ == "__main__":
    main()
