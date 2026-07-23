#!/usr/bin/env python3
"""Round-8 Part 6: cross-subject/-session targeting heterogeneity -- a POOLED
moderator regression, not n-of-1 debugging of a single failing subject.

EFFECT CHOICE (pre-specified here, before fitting -- see comments.txt 6A
"choose one, justify"): per-session behavior-CTG diagonal AUC
(results/behavior_ctg_per_session.json), NOT the v*-alignment targeting-
benchmark CATE. Reasons:
  1. The targeting-benchmark CATE (results/targeting_benchmark_boran.json)
     only exists for the 9 Boran iEEG subjects (TES1/DLPFC coverage is a
     principled exclusion for every other cohort -- DATASET_ANALYSIS_MATRIX.md
     #2), which alone is below this analysis's own n>=8 floor once any
     covariate has missing data, and gives no cross-cohort heterogeneity to
     explain.
  2. Behavior-CTG diagonal AUC is computed uniformly, per session, across
     FOUR cohorts (Boran iEEG, Boran units, DANDI 000469/001187/000673) --
     n=1+9+25+30+34-ish sessions, clearing the floor with real power, and it
     is the SAME quantity Part 1A already reports as significant in only
     1/5 cohorts (Boran iEEG). This heterogeneity analysis therefore directly
     explains the pattern Part 1A reports as a bound, rather than opening a
     new, disconnected question.
  Miller ECoG is excluded from all of behavior-CTG (no behavioral-accuracy
  field in the public release -- see run_context_confidence_timecourse.py's
  identical exclusion), so it cannot be pooled here either way.

PRE-SPECIFIED COVARIATES (fixed here before looking at the fit; the two
listed-but-dropped items are stated up front, not cut after seeing p-values):
  - n_signal_channels : good iEEG channel count (boran_ieeg) or unit count
    (all single-unit cohorts) -- "electrode/target coverage" proxy.
  - is_lfp             : 1 for boran_ieeg (continuous LFP/iEEG), 0 for the
    four single-unit cohorts -- "signal type" (categorical, dummy-coded).
  - n_trials           : trial count feeding the CTG fit.
  - accuracy           : session behavioral accuracy (n_correct/n_trials) --
    ceiling proxy (directly explains why Part 1A's bound exists).
  - var_ratio          : fraction of variance in the top-8 PCA latent --
    "latent PR/dimensionality" proxy.
  - max_abs_lambda     : plant stability, max|eigenvalue(A)| at the shared
    r=7 rank convention (boran_ieeg/dandi000469: results/divergence_analysis.npz;
    boran_units/dandi001187/dandi000673: results/dmd_extension_<cohort>.json,
    Round-8 Part 4).
  DROPPED (pre-specified, not cherry-picked): "SNR or mean firing rate" and
  "decoder accuracy" (a load/context decoder, distinct from the outcome
  decoder whose AUC is the effect here) -- neither has a session-level
  number already computed uniformly across all four cohorts without a new
  per-session decoder-fitting pass; adding one would risk a rushed, unaudited
  covariate under this round's time budget. Noted as a gap, not fabricated.

Model: per-covariate univariate OLS slope (effect ~ covariate), permutation
p-value (shuffle covariate across sessions, 5000 perms), BH-FDR across the 6
covariates. <8 sessions with a computable effect -> underpowered, STOP
(comments.txt 6A). The single most extreme session (boran_ieeg, the only
cohort clearing the behavior-CTG bound) may illustrate a covariate's story in
prose, but conclusions are drawn ONLY from the pooled fit, never from it alone.

Outputs: results/targeting_heterogeneity.json
Self-check: tests/test_targeting_heterogeneity.py (recovers a planted
covariate->effect relationship on synthetic data).

Run (after run_behavior_ctg.py, run_dmd_extension_gap_cohorts.py,
run_divergence_analysis.py):
    conda run -n wm_dynamics python scripts/run_targeting_heterogeneity.py
"""
from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from statistics import fdr_bh, stable_seed

RESULTS = ROOT / "results"
MIN_SESSIONS = 8
N_PERM = 5000
COVARIATES = ["n_signal_channels", "is_lfp", "n_trials", "accuracy", "var_ratio", "max_abs_lambda"]


def _geometry_npz_path(cohort: str, key: str) -> Path:
    return {
        "boran_ieeg": RESULTS / f"boran_geometry_{key}.npz",
        "boran_units": RESULTS / f"dandi000574_units_geometry_{key}.npz",
        "dandi000469": RESULTS / f"dandi000469_geometry_{key.split('_ses')[0]}.npz",
        "dandi001187": RESULTS / f"dandi001187_geometry_{key}.npz",
        "dandi000673": RESULTS / f"dandi000673_geometry_{key}.npz",
    }[cohort]


def _max_abs_lambda(cohort: str, key: str, div_npz, dmd_ext: dict) -> float | None:
    if cohort == "boran_ieeg":
        eig_key = f"boran_{key}_eigenvalues"
        if eig_key in div_npz:
            return float(np.max(np.abs(div_npz[eig_key])))
        return None
    if cohort == "dandi000469":
        subj = key.split("_ses")[0]
        eig_key = f"dandi000469_{subj}_eigenvalues"
        if eig_key in div_npz:
            return float(np.max(np.abs(div_npz[eig_key])))
        return None
    # boran_units / dandi001187 / dandi000673 -> dmd_extension_<cohort>.json, rank 7
    per_subj = dmd_ext.get(cohort, {}).get("per_subject", {})
    row = per_subj.get(key)
    if row is None or "7" not in row.get("by_rank", {}):
        return None
    return row["by_rank"]["7"]["max_abs_lambda"]


def build_dataset() -> dict:
    with open(RESULTS / "behavior_ctg_per_session.json") as f:
        beh = json.load(f)
    div_npz = np.load(RESULTS / "divergence_analysis.npz", allow_pickle=True)
    dmd_ext = {}
    for cohort in ["boran_units", "dandi001187", "dandi000673"]:
        p = RESULTS / f"dmd_extension_{cohort}.json"
        dmd_ext[cohort] = json.load(open(p)) if p.exists() else {}

    rows = []
    for cohort, sessions in beh.items():
        is_lfp = 1.0 if cohort == "boran_ieeg" else 0.0
        for key, row in sessions.items():
            if row.get("underpowered", True) or "diag_auc_peak" not in row:
                continue
            geo_path = _geometry_npz_path(cohort, key)
            if not geo_path.exists():
                continue
            geo = np.load(geo_path, allow_pickle=True)
            n_chan = geo["n_channels_good"] if "n_channels_good" in geo else geo.get("n_units")
            if n_chan is None:
                continue
            var_ratio = geo["var_ratio"] if "var_ratio" in geo else None
            max_lam = _max_abs_lambda(cohort, key, div_npz, dmd_ext)
            n_trials = row["n_trials"]
            n_correct = row.get("n_correct")
            if n_correct is None or var_ratio is None or max_lam is None:
                continue
            rows.append({
                "cohort": cohort, "session": key,
                "effect": row["diag_auc_peak"],
                "n_signal_channels": float(n_chan),
                "is_lfp": is_lfp,
                "n_trials": float(n_trials),
                "accuracy": float(n_correct) / float(n_trials),
                "var_ratio": float(var_ratio),
                "max_abs_lambda": float(max_lam),
            })
    return rows


def _ols_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope + intercept of y ~ x (simple OLS)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom < 1e-15:
        return 0.0, ym
    beta = float(((x - xm) * (y - ym)).sum() / denom)
    intercept = float(ym - beta * xm)
    return beta, intercept


def _bootstrap_ci_slope(x: np.ndarray, y: np.ndarray, n_boot: int = 2000,
                        rng: np.random.Generator | None = None) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(x)
    betas = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        betas[b], _ = _ols_slope(x[idx], y[idx])
    return float(np.percentile(betas, 2.5)), float(np.percentile(betas, 97.5))


def univariate_moderator_test(effect: np.ndarray, covariate: np.ndarray,
                              n_perm: int = N_PERM, rng: np.random.Generator | None = None) -> dict:
    if rng is None:
        rng = np.random.default_rng(0)
    finite = np.isfinite(effect) & np.isfinite(covariate)
    effect, covariate = effect[finite], covariate[finite]
    n = len(effect)
    if n < MIN_SESSIONS:
        return {"underpowered": True, "n": int(n)}
    beta, _ = _ols_slope(covariate, effect)
    null = np.zeros(n_perm)
    for i in range(n_perm):
        perm_cov = rng.permutation(covariate)
        null[i], _ = _ols_slope(perm_cov, effect)
    p = float((np.sum(np.abs(null) >= np.abs(beta)) + 1) / (n_perm + 1))
    ci_lo, ci_hi = _bootstrap_ci_slope(covariate, effect, rng=rng)
    spearman = float(np.corrcoef(np.argsort(np.argsort(covariate)), np.argsort(np.argsort(effect)))[0, 1])
    return {"beta": beta, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": p, "n": int(n),
           "spearman_rho": spearman}


def main():
    rows = build_dataset()
    n_total = len(rows)
    print(f"Sessions with a computable behavior-CTG effect + all covariates: {n_total}")
    if n_total < MIN_SESSIONS:
        out = {"underpowered": True, "n_sessions": n_total,
              "reason": f"only {n_total} sessions with a computable effect (<{MIN_SESSIONS})"}
        with open(RESULTS / "targeting_heterogeneity.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"UNDERPOWERED (n={n_total} < {MIN_SESSIONS}) -- STOP, wrote underpowered marker.")
        return

    effect = np.array([r["effect"] for r in rows])
    per_covariate = {}
    for cov in COVARIATES:
        cov_vals = np.array([r[cov] for r in rows])
        rng = np.random.default_rng(stable_seed(f"heterogeneity_{cov}"))
        res = univariate_moderator_test(effect, cov_vals, rng=rng)
        per_covariate[cov] = res
        if res.get("underpowered"):
            print(f"  {cov}: underpowered (n={res['n']})")
        else:
            print(f"  {cov}: beta={res['beta']:+.4f} [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}] "
                  f"p={res['p_value']:.4f} rho={res['spearman_rho']:+.3f} n={res['n']}")

    scored = {k: v for k, v in per_covariate.items() if not v.get("underpowered")}
    if scored:
        p_vals = np.array([scored[k]["p_value"] for k in scored])
        fdr = fdr_bh(p_vals, alpha=0.05)
        for k, q in zip(scored.keys(), fdr["q_values"]):
            per_covariate[k]["q_value_fdr"] = float(q)
            per_covariate[k]["survives_fdr_0.05"] = bool(q < 0.05)
        n_survive = int(fdr["n_reject"])
        print(f"\nBH-FDR across {len(scored)} scored covariates: {n_survive} survive q<0.05")
    else:
        n_survive = 0

    # Illustrative extreme session (Boran iEEG -- the only cohort clearing the
    # behavior-CTG bound) -- ILLUSTRATION ONLY, not a basis for the conclusion.
    boran_rows = [r for r in rows if r["cohort"] == "boran_ieeg"]
    extreme = max(boran_rows, key=lambda r: r["effect"]) if boran_rows else None

    out = {
        "n_sessions": n_total,
        "cohorts_pooled": sorted(set(r["cohort"] for r in rows)),
        "n_per_cohort": {c: sum(1 for r in rows if r["cohort"] == c) for c in set(r["cohort"] for r in rows)},
        "effect_definition": "behavior_ctg diag_auc_peak (per session)",
        "per_covariate": per_covariate,
        "n_covariates_survive_fdr": n_survive,
        "illustrative_extreme_session": extreme,
        "dropped_covariates_note": (
            "SNR/mean firing rate and (context/load) decoder accuracy were pre-specified "
            "candidates but dropped before fitting -- no session-level number for either "
            "already exists uniformly across all four cohorts; see module docstring."
        ),
    }
    with open(RESULTS / "targeting_heterogeneity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/targeting_heterogeneity.json ({n_total} sessions, "
          f"{len(set(r['cohort'] for r in rows))} cohorts)")


if __name__ == "__main__":
    main()
