#!/usr/bin/env python3
"""Rotation speed of the decoding axis, without a DMD operator.

Five independent lines already show the leading DMD mode does not rotate at
a usable rate in this data (near-zero ensemble rotation frequencies, theta
confidence intervals straddling zero, concentrated rather than uniform
phases, no clean 2-D ring attractor, and a null link between axis-rotation
and DMD-rotation frequency). But the DECODING AXIS itself -- the population
direction a linear classifier uses to separate conditions at each moment --
is already known to rotate (the content-vs-context axis-rotation-index
dissociation). This script defines and measures that rotation directly,
without requiring a linear operator, an invariant plane, or an identifiable
eigenvector:

  41A. omega_axis(t) = arccos(|<w(t), w(t+dt)>|) / dt [rad/s], from
       geometry.axis_angular_velocity, for every session in every cohort
       that has a context and/or content axis contrast (masks identical to
       scripts/run_axis_rotation_analysis.py, so the numbers are directly
       comparable to the existing axis-rotation-index results). Reported
       with a trial-level bootstrap CI.
  41B. The sharpest surviving version of the recency hypothesis: for DANDI
       000469 load-3 trials (three sequentially-encoded items per trial),
       regress the observed angular separation between per-item decoding
       axes on (a) the raw inter-item encoding lag and (b) the lag scaled by
       the session's own omega_axis. Session-clustered inference via
       statistics.linear_mixed_effects_test. Per-item encoding onset times
       come directly from the NWB trials table (timestamps_Encoding1/2/3);
       if a subject's file lacks them, that subject is excluded and recorded
       as such rather than approximated.
  41C. Re-tests the existing null correlation between content-axis rotation
       index and DMD rotation frequency (rho=-0.28, p=0.25) against all four
       fit-type x eigenvalue-selection-rule cells from
       results/vstar_fit_selection_factorial.json, with the four-test
       multiplicity corrected (statistics.fdr_bh).

Output: results/rotation_speed_axis.json.

Run (needs the external data mount):
    conda run -n wm_dynamics python scripts/run_rotation_speed_axis.py
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

import h5py  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from geometry import axis_angular_velocity  # noqa: E402
from statistics import (  # noqa: E402
    fdr_bh, linear_mixed_effects_test, spearman_permutation_test, stable_seed,
)

import run_axis_rotation_analysis as raa  # noqa: E402  (STEP_* constants)
import run_divergence_analysis as rda  # noqa: E402  (subject lists)
from run_multiitem_ctg_000469 import DATA_DIR as D469_DATA_DIR  # noqa: E402
from run_multiitem_ctg_000469 import ITEM_FIELDS, _class_counts_ok  # noqa: E402

RESULTS = ROOT / "results"
N_BOOTSTRAP = 200


def _dt_from_times(times: np.ndarray) -> float:
    return float(np.median(np.diff(times)))


# ── 41A: omega_axis per session, per axis type ──────────────────────────────

def _bootstrap_median_omega(Z, labels, dt, step, rng) -> dict | None:
    omega_obs, _ = axis_angular_velocity(Z, labels, dt, step)
    if len(omega_obs) == 0:
        return None
    n = Z.shape[0]
    boots = np.full(N_BOOTSTRAP, np.nan)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        om_b, _ = axis_angular_velocity(Z[idx], labels[idx], dt, step)
        if len(om_b):
            boots[b] = np.median(om_b)
    boots = boots[~np.isnan(boots)]
    return {
        "median_omega_rad_s": float(np.median(omega_obs)),
        "ci_lo": float(np.percentile(boots, 2.5)) if len(boots) else None,
        "ci_hi": float(np.percentile(boots, 97.5)) if len(boots) else None,
        "n_trials": int(n), "n_bootstrap_ok": int(len(boots)),
    }


def sessions_41a():
    """Yields (dataset, session, axis_type, Z_masked, labels, dt, step),
    with masks identical to run_axis_rotation_analysis.py's contrasts."""
    for path in sorted(RESULTS.glob("dandi000469_geometry_sub-*.npz")):
        subj = path.stem.replace("dandi000469_geometry_", "")
        d = np.load(path, allow_pickle=True)
        Z, loads, pic_id = d["Z"], d["loads"], d["pic_id_enc1"]
        dt = _dt_from_times(d["times"])
        ctx_mask = (loads == 1) | (loads == 3)
        if ctx_mask.sum() >= 10:
            yield ("dandi000469", subj, "context", Z[ctx_mask],
                   (loads[ctx_mask] == 3).astype(int), dt, raa.STEP_000469)
        load1_mask = loads == 1
        content_labels = pic_id[load1_mask]
        if load1_mask.sum() >= 15 and len(np.unique(content_labels)) >= 2:
            yield ("dandi000469", subj, "content", Z[load1_mask], content_labels, dt, raa.STEP_000469)

    d = np.load(RESULTS / "pfc3_content_ctg.npz", allow_pickle=True)
    from spike_pipeline import fit_pca_psth
    Z, _, _ = fit_pca_psth(d["X"], n_comp=8)
    yield "pfc3", "pooled", "content", Z, d["y"], 0.1, raa.STEP_PFC3

    for subj in rda.MILLER_SUBJECTS:
        d = np.load(RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
        Z, task_id = d["Z"], d["task_id"]
        mask = (task_id == 0) | (task_id == 2)
        dt = _dt_from_times(d["times"])
        yield "miller", subj, "context", Z[mask], (task_id[mask] == 2).astype(int), dt, raa.STEP_MILLER

    for path in sorted(RESULTS.glob("boran_geometry_sub-*.npz")):
        subj = path.stem.replace("boran_geometry_", "")
        d = np.load(path, allow_pickle=True)
        Z, set_sizes = d["Z"], d["set_sizes"]
        mask = (set_sizes == 4) | (set_sizes == 8)
        if mask.sum() < 10:
            continue
        dt = _dt_from_times(d["times"])
        yield "boran", subj, "context", Z[mask], (set_sizes[mask] == 8).astype(int), dt, raa.STEP_BORAN

    for path in sorted(RESULTS.glob("dandi000574_units_geometry_sub-*.npz")):
        key = path.stem.replace("dandi000574_units_geometry_", "")
        d = np.load(path, allow_pickle=True)
        Z, set_size = d["Z"], d["set_size"]
        mask = (set_size == 4) | (set_size == 8)
        if mask.sum() < 10:
            continue
        dt = _dt_from_times(d["times"])
        yield "boran_units", key, "context", Z[mask], (set_size[mask] == 8).astype(int), dt, raa.STEP_BINNED

    for dataset, glob_pat, prefix in (
        ("dandi001187", "dandi001187_geometry_sub-*.npz", "dandi001187_geometry_"),
        ("dandi000673", "dandi000673_geometry_sub-*.npz", "dandi000673_geometry_"),
    ):
        for path in sorted(RESULTS.glob(glob_pat)):
            key = path.stem.replace(prefix, "")
            d = np.load(path, allow_pickle=True)
            Z, loads = d["Z"], d["loads"]
            mask = (loads == 1) | (loads == 3)
            if mask.sum() < 10:
                continue
            dt = _dt_from_times(d["times"])
            yield dataset, key, "context", Z[mask], (loads[mask] == 3).astype(int), dt, raa.STEP_BINNED


def run_41a() -> dict:
    out: dict[str, dict] = {}
    for dataset, session, axis_type, Z, labels, dt, step in sessions_41a():
        rng = np.random.default_rng(stable_seed(f"rotation_speed_axis_{dataset}_{session}_{axis_type}"))
        res = _bootstrap_median_omega(Z, labels, dt, step, rng)
        if res is None:
            continue
        out.setdefault(dataset, {}).setdefault(session, {})[axis_type] = res

    content_all, context_all = [], []
    for sessions in out.values():
        for row in sessions.values():
            if "content" in row:
                content_all.append(row["content"]["median_omega_rad_s"])
            if "context" in row:
                context_all.append(row["context"]["median_omega_rad_s"])
    summary = {
        "content_median_omega_rad_s_mean": float(np.mean(content_all)) if content_all else None,
        "n_content_sessions": len(content_all),
        "context_median_omega_rad_s_mean": float(np.mean(context_all)) if context_all else None,
        "n_context_sessions": len(context_all),
    }
    print(f"  41A content omega: mean={summary['content_median_omega_rad_s_mean']} "
          f"(N={summary['n_content_sessions']})")
    print(f"  41A context omega: mean={summary['context_median_omega_rad_s_mean']} "
          f"(N={summary['n_context_sessions']})")
    return {"per_session": out, "summary": summary}


# ── 41B: recency test on omega_axis, DANDI 000469 load-3 ───────────────────

def _item_axis(Z_masked: np.ndarray, labels: np.ndarray, t_idx: np.ndarray) -> np.ndarray | None:
    """A single representative decoding direction for one item's identity,
    fit on all (trial, subsampled-timepoint) pairs pooled together (item
    identity is constant across the whole maintenance period within a
    trial), then collapsed to the leading right-singular vector of the
    per-class weight matrix."""
    X_pool = Z_masked[:, t_idx, :].reshape(-1, Z_masked.shape[2])
    y_pool = np.repeat(labels, len(t_idx))
    if len(np.unique(y_pool)) < 2:
        return None
    Xs = StandardScaler().fit_transform(X_pool)
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(Xs, y_pool)
    _, _, Vt = np.linalg.svd(clf.coef_, full_matrices=False)
    return Vt[0]


def run_41b() -> dict:
    rows = []
    excluded_no_timestamps = []
    for path in sorted(RESULTS.glob("dandi000469_geometry_sub-*.npz")):
        subj = path.stem.replace("dandi000469_geometry_", "")
        nwb_path = D469_DATA_DIR / subj / f"{subj}_ses-2_ecephys+image.nwb"
        if not nwb_path.exists():
            continue
        d = np.load(path, allow_pickle=True)
        Z, loads = d["Z"], d["loads"]
        mask = loads == 3
        if mask.sum() < 12:
            continue

        try:
            with h5py.File(str(nwb_path), "r") as f:
                item_labels = {name: f[f"intervals/trials/{field}"][:].astype(int)
                               for name, field in ITEM_FIELDS.items()}
                enc_times = {i: f[f"intervals/trials/timestamps_Encoding{i}"][:] for i in (1, 2, 3)}
        except KeyError:
            excluded_no_timestamps.append(subj)
            continue

        labels_masked = {name: item_labels[name][mask] for name in ITEM_FIELDS}
        if not all(_class_counts_ok(labels_masked[name]) for name in ITEM_FIELDS):
            continue

        dt = _dt_from_times(d["times"])
        t_idx = np.arange(0, Z.shape[1], raa.STEP_000469)
        Zm = Z[mask]
        item_axes = {name: _item_axis(Zm, labels_masked[name], t_idx) for name in ITEM_FIELDS}
        if any(v is None for v in item_axes.values()):
            continue

        omega_axis, _ = axis_angular_velocity(Zm, labels_masked["item1"], dt, raa.STEP_000469)
        if len(omega_axis) == 0:
            continue
        session_omega = float(np.median(omega_axis))
        mean_t = {i: float(np.mean(enc_times[i][mask])) for i in (1, 2, 3)}

        for name_i, name_j, i, j in (("item1", "item2", 1, 2), ("item1", "item3", 1, 3),
                                      ("item2", "item3", 2, 3)):
            angle = float(np.arccos(np.clip(np.abs(np.dot(item_axes[name_i], item_axes[name_j])), -1, 1)))
            lag = mean_t[j] - mean_t[i]
            rows.append({"session": subj, "pair": f"{name_i}-{name_j}", "angle_rad": angle,
                         "lag_s": lag, "omega_axis_rad_s": session_omega,
                         "predicted_angle_rad": session_omega * lag})

    if len(rows) < 4 * 3:  # fewer than 4 qualifying sessions (3 pairs each)
        return {"untestable": True, "reason": "fewer than 4 qualifying sessions with recoverable "
                "per-item encoding times and sufficient per-item class counts",
                "n_qualifying_sessions": len({r["session"] for r in rows}),
                "excluded_no_timestamps": excluded_no_timestamps}

    angle = np.array([r["angle_rad"] for r in rows])
    lag = np.array([r["lag_s"] for r in rows])
    pred = np.array([r["predicted_angle_rad"] for r in rows])
    subject = np.array([r["session"] for r in rows])

    raw_lag_test = linear_mixed_effects_test(angle, lag, subject, n_perm=10000,
                                              rng=np.random.default_rng(stable_seed("41b_raw_lag")))
    scaled_lag_test = linear_mixed_effects_test(angle, pred, subject, n_perm=10000,
                                                 rng=np.random.default_rng(stable_seed("41b_scaled_lag")))
    n_sessions = len(set(subject))
    print(f"  41B (N={n_sessions} sessions, {len(rows)} item-pairs): "
          f"raw-lag beta={raw_lag_test['beta']:.4f} p={raw_lag_test['p_value']:.4f}; "
          f"omega-scaled-lag beta={scaled_lag_test['beta']:.4f} p={scaled_lag_test['p_value']:.4f}")
    return {"untestable": False, "n_sessions": n_sessions, "n_item_pairs": len(rows),
            "raw_lag_test": raw_lag_test, "omega_scaled_lag_test": scaled_lag_test,
            "excluded_no_timestamps": excluded_no_timestamps, "rows": rows}


# ── 41C: re-test the axis-rotation vs DMD-rotation-frequency null ─────────

def run_41c() -> dict:
    ari = json.load(open(RESULTS / "axis_rotation_dandi000469.json"))
    factorial = json.load(open(RESULTS / "vstar_fit_selection_factorial.json"))["dandi000469"]
    cells = [("mean_traj", "re"), ("mean_traj", "mod"), ("ensemble", "re"), ("ensemble", "mod")]

    out = {}
    p_values = []
    for fit, rule in cells:
        pairs = [(row["content_axis_rotation_index"], factorial[subj][fit][rule]["f_hz"])
                 for subj, row in ari.items()
                 if "content_axis_rotation_index" in row and subj in factorial]
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        res = spearman_permutation_test(x, y, n_perm=10000,
                                         rng=np.random.default_rng(stable_seed(f"41c_{fit}_{rule}")))
        cell_name = f"{fit}+{rule}"
        out[cell_name] = {"rho": res["rho"], "p_value": res["p_value"], "n": len(pairs)}
        p_values.append(res["p_value"])
        print(f"  41C {cell_name}: rho={res['rho']:.3f} p={res['p_value']:.4f} (N={len(pairs)})")

    fdr = fdr_bh(np.array(p_values))
    out["_multiplicity_correction"] = {
        "cells": [f"{f}+{r}" for f, r in cells], "n_tests": 4,
        "q_values": fdr["q_values"].tolist(), "n_reject_at_fdr_0.05": fdr["n_reject"],
    }
    return out


def main():
    print("41A -- omega_axis per session (context and content axes) ...")
    out_41a = run_41a()

    print("\n41B -- recency test on omega_axis (DANDI 000469 load-3) ...")
    out_41b = run_41b()

    print("\n41C -- re-test axis-rotation vs DMD-rotation-frequency null (4 cells) ...")
    out_41c = run_41c()

    out = {"41A_omega_axis": out_41a, "41B_recency_test": out_41b, "41C_null_retest": out_41c}
    with open(RESULTS / "rotation_speed_axis.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/rotation_speed_axis.json")


def _self_check():
    """A rotating decoding axis should give a much larger omega_axis than a
    fixed one, and the recovered rate should match the true one to within a
    factor of ~2 (already validated more tightly ad hoc; this is the
    regression-guarding version)."""
    rng = np.random.default_rng(0)
    N, T, k = 200, 40, 4
    labels = rng.integers(0, 2, N)

    Z_fixed = rng.standard_normal((N, T, k)) * 0.1
    true_w = rng.standard_normal(k)
    true_w /= np.linalg.norm(true_w)
    for t in range(T):
        Z_fixed[:, t, :] += np.outer((labels * 2 - 1) * 1.5, true_w)
    om_fixed, _ = axis_angular_velocity(Z_fixed, labels, dt=0.1, step=5)

    Z_rot = rng.standard_normal((N, T, k)) * 0.1
    true_omega = 2 * np.pi / (T * 0.1)
    for t in range(T):
        theta = 2 * np.pi * t / T
        w_t = np.zeros(k)
        w_t[0], w_t[1] = np.cos(theta), np.sin(theta)
        Z_rot[:, t, :] += np.outer((labels * 2 - 1) * 1.5, w_t)
    om_rot, _ = axis_angular_velocity(Z_rot, labels, dt=0.1, step=5)

    assert np.median(om_fixed) < 0.2, f"fixed axis should have near-zero omega, got {np.median(om_fixed):.3f}"
    assert abs(np.median(om_rot) - true_omega) / true_omega < 0.5, (
        f"rotating axis omega {np.median(om_rot):.3f} far from true {true_omega:.3f}")
    print(f"Self-check passed: fixed-axis omega={np.median(om_fixed):.3f} rad/s, "
          f"rotating-axis omega={np.median(om_rot):.3f} rad/s (true={true_omega:.3f}).")


if __name__ == "__main__":
    _self_check()
    main()
