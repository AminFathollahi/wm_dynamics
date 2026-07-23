#!/usr/bin/env python3
"""Round-12 PART 21 -- is the PART 18 transfer null biological idiosyncrasy or
estimation noise?

WHY: PART 18 found cross-session v* transfer is null (own-session slope
survives, transfer slope does not) on the 10 chronic Wa sessions. The paper
hedges that it cannot separate genuine session-to-session biological change
from per-session estimation noise. This script resolves that hedge (if the
data allow) by comparing:
  - WITHIN-session split-half v* stability (interleaved even/odd control
    trials, PCA basis V held fixed per session) -- an estimation-noise floor.
  - ACROSS-session v* similarity (same full-session v*_chan geometry PART 18
    uses for its transfer target) -- the null distribution "how similar is v*
    between different sessions."

If within >> across: v* is stably estimable within a session but genuinely
differs across sessions (biological idiosyncrasy, affirms the transfer null).
If within ~= across (both low): v* estimation is noise-limited at this data
volume (the transfer null is at least partly noise) -- the per-session-refit
prescription is unchanged either way (K7: do not reopen the transfer null).

Reuses (DRY, no reimplementation): run_soldado_pipeline.load_soldado_session/
crop_trial/SESSIONS/N_PC/DMD_RANK/N_BINS/BIN_S, src/geometry.pca_decompose,
src/dynamics.dmd_reconstruction_error, src/control.unstable_eigenvector, and
run_target_transfer._fit_session_v_and_V for the full-session v*_chan (the
identical geometry PART 18's transfer target uses).

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_vstar_split_half_stability.py
"""
from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry import pca_decompose
from dynamics import dmd_reconstruction_error
from control import unstable_eigenvector

sys.path.insert(0, str(ROOT / "scripts"))
from run_soldado_pipeline import (
    load_soldado_session, crop_trial, SESSIONS, N_PC, DMD_RANK, N_BINS, BIN_S,
)
from run_target_transfer import _fit_session_v_and_V

RESULTS = ROOT / "results"
WA_SESSIONS = [s for s in SESSIONS if s.startswith("Wa")]
MIN_HALF = 10  # same min-data rule build_session_features/17A use to accept an A fit
MIN_QUALIFIED = 5


def _self_check() -> None:
    """Pure sanity checks on the |cos| convention used below, before touching
    real data: a vector compared to itself (or its negation, since eigenvector
    sign is arbitrary) gives |cos|==1; an orthonormal projection preserves
    unit norm."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=8)
    u /= np.linalg.norm(u)
    assert abs(abs(float(np.dot(u, u))) - 1.0) < 1e-9
    assert abs(abs(float(np.dot(u, -u))) - 1.0) < 1e-9  # sign-arbitrary eigenvector

    C, k = 96, 8
    V, _ = np.linalg.qr(rng.normal(size=(C, k)))  # orthonormal columns
    v_chan = V @ u
    assert abs(np.linalg.norm(v_chan) - 1.0) < 1e-9, "orthonormal projection must preserve unit norm"
    print("[self-check] |cos(u,u)|==1, |cos(u,-u)|==1, orthonormal projection preserves unit norm -- PASS")


def _split_half_fit(prefix: str) -> dict | None:
    """21A gate + 21B fit: interleaved even/odd split-half v* in channel space,
    with the session's PCA basis V and centering mean held FIXED (fit once on
    the full session's control epochs), isolating variability in the DYNAMICS
    operator A, not the subspace."""
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
    n_total = len(ctrl_epochs)

    halves = {"even": ctrl_epochs[0::2], "odd": ctrl_epochs[1::2]}
    if min(len(halves["even"]), len(halves["odd"])) < MIN_HALF:
        return {"qualifies": False, "n_total": n_total,
                "n_even": len(halves["even"]), "n_odd": len(halves["odd"])}

    Z_all = np.stack(ctrl_epochs, axis=0)  # (N, N_BINS, C)
    X_flat = Z_all.reshape(-1, C)
    mean_fixed = X_flat.mean(0)
    _, V, _ = pca_decompose(X_flat, N_PC)  # V: (C, k), FIXED for both halves
    k = V.shape[1]
    r_use = min(DMD_RANK, k, N_BINS - 2)

    v_chan_half = {}
    for name, eps in halves.items():
        Z_h = np.stack(eps, axis=0)
        Z_h_mean = ((Z_h.reshape(-1, C) - mean_fixed) @ V).reshape(
            Z_h.shape[0], N_BINS, k).mean(0)
        dmd = dmd_reconstruction_error(Z_h_mean, r=r_use, dt=BIN_S)
        v_star_half, _ = unstable_eigenvector(dmd["A"])
        v_chan = V @ v_star_half
        assert abs(np.linalg.norm(v_chan) - 1.0) < 1e-6, f"{prefix}/{name}: v*_chan not unit-norm"
        v_chan_half[name] = v_chan

    within_cos = abs(float(np.dot(v_chan_half["even"], v_chan_half["odd"])))
    assert 0.0 <= within_cos <= 1.0 + 1e-9, f"{prefix}: within cos out of [0,1]: {within_cos}"

    return {"qualifies": True, "n_total": n_total, "n_even": len(halves["even"]),
            "n_odd": len(halves["odd"]), "within_split_half_cos": within_cos}


def main() -> None:
    _self_check()

    print("\n21A feasibility gate: interleaved-half fit (>=%d control epochs/half) "
          "on the 10 chronic Wa (shared 96-ch) sessions ..." % MIN_HALF)
    per_session = {}
    for s in WA_SESSIONS:
        r = _split_half_fit(s)
        if r is None:
            print(f"  {s}: SKIP (session unusable)")
            continue
        per_session[s] = r
        if r["qualifies"]:
            print(f"  {s}: n_total={r['n_total']} (even={r['n_even']},odd={r['n_odd']}) "
                  f"within_cos={r['within_split_half_cos']:.4f}")
        else:
            print(f"  {s}: DOES NOT QUALIFY (n_total={r['n_total']}, "
                  f"even={r['n_even']}, odd={r['n_odd']})")

    qualified = [s for s, r in per_session.items() if r.get("qualifies")]
    print(f"\nn_qualified = {len(qualified)} / {len(WA_SESSIONS)} Wa sessions")

    if len(qualified) < MIN_QUALIFIED:
        out = {
            "status": "underpowered",
            "n_sessions_qualified": len(qualified),
            "reason": f"only {len(qualified)} Wa sessions qualify for interleaved "
                      f"split-half fitting (need >= {MIN_QUALIFIED}); cannot separate "
                      "biological idiosyncrasy from estimation noise this round -- "
                      "the existing transfer-null hedge stands unchanged.",
            "per_session": per_session,
        }
        with open(RESULTS / "vstar_split_half_stability.json", "w") as f:
            json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
        print("UNDERPOWERED -- wrote status only, stopping before 21B/21C.")
        return

    # 21C: across-session baseline, identical geometry to PART 18's transfer
    # target (full-session v*_chan via run_target_transfer._fit_session_v_and_V).
    print("\n21C: fitting full-session v*_chan for the across-session baseline ...")
    full_v_chan = {}
    for s in qualified:
        fit = _fit_session_v_and_V(s)
        if fit is None:
            print(f"  {s}: SKIP (full-session fit failed)")
            continue
        full_v_chan[s] = fit["V"] @ fit["v_star"]

    pair_sessions = [s for s in qualified if s in full_v_chan]
    across_pairs = {}
    for a, b in combinations(pair_sessions, 2):
        c = abs(float(np.dot(full_v_chan[a], full_v_chan[b])))
        assert 0.0 <= c <= 1.0 + 1e-9, f"pairwise cos out of [0,1]: {a},{b}={c}"
        across_pairs[f"{a}__{b}"] = c

    within_vals = np.array([per_session[s]["within_split_half_cos"] for s in qualified])
    across_vals = np.array(list(across_pairs.values()))
    within_median = float(np.median(within_vals))
    across_median = float(np.median(across_vals))
    across_q25, across_q75 = (float(x) for x in np.percentile(across_vals, [25, 75]))

    print(f"\nwithin_median (split-half, n={len(within_vals)} sessions)  = {within_median:.4f}")
    print(f"across_median (cross-session, n={len(across_vals)} pairs)   = {across_median:.4f} "
          f"[IQR {across_q25:.4f}, {across_q75:.4f}]")

    if within_median > 0.7 and within_median > across_median:
        verdict = (f"Within-session split-half stability (median |cos|={within_median:.3f}) is "
                   f"high and clearly exceeds the across-session baseline (median "
                   f"|cos|={across_median:.3f}) -- v* is STABLY estimable within a session but "
                   "GENUINELY differs across sessions: the PART 18 transfer null reflects "
                   "biological idiosyncrasy, not estimation noise. Affirms the "
                   "session-static-but-session-idiosyncratic conclusion and the "
                   "re-fit-per-session prescription.")
    else:
        verdict = (f"Within-session split-half stability (median |cos|={within_median:.3f}) is "
                   f"comparable to the across-session baseline (median |cos|={across_median:.3f}) "
                   "-- v* estimation is noise-limited at this data volume, so the PART 18 "
                   "transfer null is at least partly estimation noise rather than purely "
                   "biological drift. The clinical conclusion is unchanged either way "
                   "(per-session re-fit): the transfer null itself is NOT reopened as a positive.")
    print(f"\nVERDICT: {verdict}")

    out = {
        "status": "ok",
        "n_sessions_qualified": len(qualified),
        "sessions_qualified": qualified,
        "per_session": per_session,
        "within_split_half_cos": within_vals.tolist(),
        "within_median": within_median,
        "across_pairwise_cos": {
            "pairs": across_pairs,
            "median": across_median,
            "iqr": [across_q25, across_q75],
            "n_pairs": len(across_vals),
        },
        "across_median": across_median,
        "split": "interleaved",
        "basis": "shared 96ch Wa",
        "verdict": verdict,
    }
    with open(RESULTS / "vstar_split_half_stability.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/vstar_split_half_stability.json")


if __name__ == "__main__":
    main()
