#!/usr/bin/env python3
"""Behavioral performance-predictability -> geometry link (Round-7, STEP C).

Anchor dataset: DANDI 000469 (content + context + outcome from the same
sessions). Add Boran iEEG (context + outcome; content is N/A -- no repeated
items, same exclusion as axis-rotation content). Per session, computes the
geometry metrics already defined elsewhere in the paper --
content/context CTG temporal-stability tau (geometry.temporal_stability_tau,
already stored per-session in dandi000469_ctg / consumed here, not
recomputed), content/context axis-rotation index (axis_rotation_*.json,
already computed by run_axis_rotation_analysis.py), and DMD rotation
magnitude (imag part of the leading eigenvalue of geometry.exact_dmd fit on
the session's mean high-load trajectory) -- and asks whether behavioral
outcome (session accuracy; matched trial-level correct-vs-error DMD-rotation
contrast where session N supports it) tracks them.

Every metric here is a SESSION-LEVEL scalar (tau/axis-rotation/dmd-rotation
are each fit once per session on that session's whole trial pool, not
per-trial), so the natural test is the SAME between-session design already
used by run_contraction_behavior_analysis_000469.py (spearman_permutation_test
across sessions) -- not a trial-level mixed-effects regression, which would
require a metric that genuinely varies trial-to-trial. Where a trial-level
paired contrast IS available (splitting a session's own trials into its
correct/error subsets and re-fitting DMD separately on each -- the same
design run_contraction_behavior_analysis_000469.py already uses for
contraction rate), paired_sign_flip_test provides the paired within-session
test spec C2 asks for. rho is reported with a bootstrap 95% CI (bootstrap_ci
on the same Spearman statistic) alongside the permutation p, matching spec
C3's {beta, ci_lo, ci_hi, p} schema (beta := rho here, since these are rank
correlations of session-level scalars, not a fitted linear-model beta).

HONEST NOTE (spec C3): the existing drift-vs-accuracy link (PAPER_REPORT.tex
L1458-1712) is already weak and mixed in direction; modest/null effects here
are expected and reported as they come out, not inflated.

Outputs: results/behavior_geometry_link.json
Updates: results/all_statistics.json -- "behavior_geometry_link" key

Run (after run_000469_pipeline.py, run_boran_pipeline.py,
run_axis_rotation_analysis.py):
    conda run -n wm_dynamics python scripts/run_behavior_geometry_link.py
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

from dynamics import exact_dmd
from statistics import spearman_permutation_test, bootstrap_ci, paired_sign_flip_test, stable_seed
from io_utils import locked_json_update

RESULTS = ROOT / "results"
MIN_SESSIONS = 4
MIN_TRIALS_PER_OUTCOME = 8   # matches PR_MIN_TRIALS_PER_GROUP convention elsewhere
DT = 0.1   # 100 ms bins, both 000469 and Boran iEEG geometry files share this via
           # their own `times` arrays (Boran iEEG's is native-rate; see per-dataset code)
DMD_RANK = 8   # module's full-latent-rank convention, matching run_divergence_analysis.py


def _dmd_rotation_hz(Z_mean: np.ndarray, dt: float) -> float:
    """Imag part (Hz) of the dominant (largest-|Re|) eigenvalue's continuous-time
    log -- same "DMD rotation magnitude" convention as run_axis_rotation_analysis's
    rotation_frequency_hz, reused here directly via exact_dmd (spec C1 instruction)."""
    T, d = Z_mean.shape
    r = min(DMD_RANK, d, T - 2)
    res = exact_dmd(Z_mean.T, r=r, dt=dt)
    lam = res["eigenvalues"]
    dominant = lam[np.argmax(np.abs(lam.real))]
    omega = np.log(dominant + 1e-300) / dt
    return float(np.abs(omega.imag) / (2 * np.pi))


def _rho_ci_p(x: np.ndarray, y: np.ndarray, seed_tag: str) -> dict:
    """Spearman rho with a bootstrap 95% CI and a permutation p -- spec C3's
    {beta, ci_lo, ci_hi, p} schema, beta:=rho (see module docstring)."""
    from scipy.stats import spearmanr

    def _rho_stat(idx_pairs: np.ndarray) -> float:
        return float(spearmanr(idx_pairs[:, 0], idx_pairs[:, 1]).statistic)

    xy = np.column_stack([x, y])
    _, ci_lo, ci_hi = bootstrap_ci(xy, _rho_stat, n_boot=5000,
                                   rng=np.random.default_rng(stable_seed(seed_tag)))
    res = spearman_permutation_test(x, y, n_perm=10000, rng=np.random.default_rng(stable_seed(seed_tag + "_perm")))
    return {"beta": res["rho"], "ci_lo": ci_lo, "ci_hi": ci_hi, "p": res["p_value"], "n": res["n"]}


def run_dandi000469() -> dict:
    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)
    ctg = stats.get("dandi000469_ctg", {})
    try:
        with open(RESULTS / "axis_rotation_dandi000469.json") as f:
            axis_rot = json.load(f)
    except FileNotFoundError:
        axis_rot = {}

    rows = {}
    for path in sorted(RESULTS.glob("dandi000469_geometry_sub-*.npz")):
        subj = path.stem.replace("dandi000469_geometry_", "")
        if subj not in ctg:
            continue
        geo = np.load(path, allow_pickle=True)
        Z, loads, resp = geo["Z"], geo["loads"], geo["response_accuracy"].astype(bool)
        times = geo["times"]
        dt = float(np.median(np.diff(times)))

        mask3 = loads == 3
        if mask3.sum() < 5:
            continue
        Z_mean3 = Z[mask3].mean(0)
        dmd_rot = _dmd_rotation_hz(Z_mean3, dt)

        row = {
            "accuracy": float(resp.mean()),
            "tau_context": ctg[subj].get("tau"),
            "tau_content": (ctg[subj].get("content_ctg") or {}).get("tau"),
            "axis_rot_context": axis_rot.get(subj, {}).get("context_axis_rotation_index"),
            "axis_rot_content": axis_rot.get(subj, {}).get("content_axis_rotation_index"),
            "dmd_rot": dmd_rot,
            "n_trials": int(len(loads)),
        }

        # Paired within-session correct-vs-error DMD-rotation contrast (load-3
        # trials only, matching the context-mask used for dmd_rot above).
        resp3 = resp[mask3]
        if resp3.sum() >= MIN_TRIALS_PER_OUTCOME and (~resp3).sum() >= MIN_TRIALS_PER_OUTCOME:
            dmd_rot_correct = _dmd_rotation_hz(Z[mask3][resp3].mean(0), dt)
            dmd_rot_error = _dmd_rotation_hz(Z[mask3][~resp3].mean(0), dt)
            row["dmd_rot_correct"] = dmd_rot_correct
            row["dmd_rot_error"] = dmd_rot_error
        rows[subj] = row

    return _build_dataset_result(rows, "DANDI 000469", has_content=True)


def run_boran_ieeg() -> dict:
    try:
        with open(RESULTS / "axis_rotation_boran.json") as f:
            axis_rot = json.load(f)
    except FileNotFoundError:
        axis_rot = {}
    # dandi000469_ctg's "tau"/pooled-tau is context-CTG derived from
    # load_vs_load_ctg's tau_info; the direct Boran-iEEG analogue is boran_ctg's
    # own set4v8 tau, read from all_statistics.json once (not per session).
    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)
    boran_ctg_all = stats.get("boran_ctg", {})

    rows = {}
    for path in sorted(RESULTS.glob("boran_geometry_sub-*.npz")):
        subj = path.stem.replace("boran_geometry_", "")
        geo = np.load(path, allow_pickle=True)
        Z, set_sizes = geo["Z"], geo["set_sizes"]
        correct = geo.get("correct", np.ones(Z.shape[0], dtype=bool)).astype(bool)
        times = geo["times"]
        dt = float(np.median(np.diff(times)))

        mask8 = set_sizes == 8
        if mask8.sum() < 5:
            continue
        # Boran iEEG is native-rate (T~4194); a full-length exact_dmd at that many
        # samples is impractical here purely for cost reasons -- downsample the
        # mean trajectory the same ~0.2s stride run_axis_rotation_analysis/
        # run_boran_pipeline already use for their own Boran CTG time axis.
        step_native = 280
        Z_mean8 = Z[mask8].mean(0)[::step_native]
        dt_ds = dt * step_native
        if Z_mean8.shape[0] < 6:
            continue
        dmd_rot = _dmd_rotation_hz(Z_mean8, dt_ds)
        boran_ctg = boran_ctg_all.get(subj, {})

        row = {
            "accuracy": float(correct.mean()),
            "tau_context": boran_ctg.get("tau"),
            "tau_content": None,       # N/A -- no repeated items (exclusion, not fabricated)
            "axis_rot_context": axis_rot.get(subj, {}).get("context_axis_rotation_index"),
            "axis_rot_content": None,  # N/A -- same exclusion
            "dmd_rot": dmd_rot,
            "n_trials": int(len(set_sizes)),
        }

        correct8 = correct[mask8]
        if correct8.sum() >= MIN_TRIALS_PER_OUTCOME and (~correct8).sum() >= MIN_TRIALS_PER_OUTCOME:
            dmd_rot_correct = _dmd_rotation_hz(Z[mask8][correct8].mean(0)[::step_native], dt_ds)
            dmd_rot_error = _dmd_rotation_hz(Z[mask8][~correct8].mean(0)[::step_native], dt_ds)
            row["dmd_rot_correct"] = dmd_rot_correct
            row["dmd_rot_error"] = dmd_rot_error
        rows[subj] = row

    return _build_dataset_result(rows, "Boran iEEG", has_content=False)


def _build_dataset_result(rows: dict, label: str, has_content: bool) -> dict:
    n_sessions = len(rows)
    result = {"n_sessions": n_sessions, "label": label, "per_session": rows}
    if n_sessions < MIN_SESSIONS:
        result["underpowered"] = True
        return result
    result["underpowered"] = False

    acc = np.array([v["accuracy"] for v in rows.values()])

    def _metric_test(field: str) -> dict | None:
        vals = np.array([v.get(field) for v in rows.values()], dtype=float)
        finite = np.isfinite(vals) & np.isfinite(acc)
        if finite.sum() < MIN_SESSIONS:
            return None
        return _rho_ci_p(vals[finite], acc[finite], f"behgeo_{label}_{field}")

    result["tau_context"] = _metric_test("tau_context")
    result["axis_rot_context"] = _metric_test("axis_rot_context")
    result["dmd_rot"] = _metric_test("dmd_rot")
    if has_content:
        result["tau_content"] = _metric_test("tau_content")
        result["axis_rot_content"] = _metric_test("axis_rot_content")

    # Paired within-session correct-vs-error DMD-rotation contrast (spec C2's
    # paired_sign_flip_test instruction), pooled across sessions with both groups.
    ce_pairs = [(v["dmd_rot_correct"], v["dmd_rot_error"]) for v in rows.values()
               if "dmd_rot_correct" in v]
    if len(ce_pairs) >= MIN_SESSIONS:
        dc = np.array([p[0] for p in ce_pairs])
        de = np.array([p[1] for p in ce_pairs])
        res = paired_sign_flip_test(de, dc, n_perm=10000, alternative="two-sided",
                                    rng=np.random.default_rng(stable_seed(f"behgeo_{label}_ce_dmdrot")))
        result["dmd_rot_correct_vs_error"] = {
            "mean_diff_error_minus_correct": res["mean_diff"], "ci_lower": res["ci_lower"],
            "ci_upper": res["ci_upper"], "p_value": res["p_value"], "n_sessions": len(ce_pairs),
        }
    else:
        result["dmd_rot_correct_vs_error"] = {"note": f"only {len(ce_pairs)} sessions with "
                                              f">= {MIN_TRIALS_PER_OUTCOME} trials in both outcome groups"}
    return result


def main():
    print("DANDI 000469 (content + context + outcome, anchor dataset)...")
    d469 = run_dandi000469()
    if d469["underpowered"]:
        print(f"  UNDERPOWERED: only {d469['n_sessions']} sessions (<{MIN_SESSIONS})")
    else:
        for metric in ("tau_context", "tau_content", "axis_rot_context", "axis_rot_content", "dmd_rot"):
            r = d469.get(metric)
            if r:
                print(f"  {metric} vs accuracy (N={r['n']}): rho={r['beta']:+.3f} "
                      f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p']:.4f}")
        ce = d469["dmd_rot_correct_vs_error"]
        if "p_value" in ce:
            print(f"  dmd_rot error-vs-correct (N={ce['n_sessions']}): "
                  f"diff={ce['mean_diff_error_minus_correct']:+.4f} p={ce['p_value']:.4f}")

    print("\nBoran iEEG (context + outcome; content N/A)...")
    boran = run_boran_ieeg()
    if boran["underpowered"]:
        print(f"  UNDERPOWERED: only {boran['n_sessions']} sessions (<{MIN_SESSIONS})")
    else:
        for metric in ("tau_context", "axis_rot_context", "dmd_rot"):
            r = boran.get(metric)
            if r:
                print(f"  {metric} vs accuracy (N={r['n']}): rho={r['beta']:+.3f} "
                      f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p={r['p']:.4f}")
        ce = boran["dmd_rot_correct_vs_error"]
        if "p_value" in ce:
            print(f"  dmd_rot error-vs-correct (N={ce['n_sessions']}): "
                  f"diff={ce['mean_diff_error_minus_correct']:+.4f} p={ce['p_value']:.4f}")

    out = {"dandi000469": d469, "boran_ieeg": boran}
    with open(RESULTS / "behavior_geometry_link.json", "w") as f:
        json.dump(out, f, indent=2)
    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["behavior_geometry_link"] = out
    print("\nSaved results/behavior_geometry_link.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
