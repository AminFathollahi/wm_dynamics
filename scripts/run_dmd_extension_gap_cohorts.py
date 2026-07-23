#!/usr/bin/env python3
"""Round-8 Part 4: DMD/dynamics rank extension on the three dataset-coverage
gap cohorts (boran_units, dandi001187, dandi000673).

Round 7 (run_divergence_analysis.py process_boran_units / process_dandi001187
/ process_dandi000673, "STEP K2") already ran axis-rotation and a SINGLE-rank
(full latent dim, matching their siblings) DMD divergence/tangling fit on
these three cohorts. What was NOT done for them (unlike the primary Soldado
fit in run_dmd_rank_selection.py) is the cv-R^2-by-rank sweep this script
adds, reusing dynamics.ensemble_dmd exactly as run_dmd_rank_selection.py
does, per subject/session, at r in {4,5,6,7,8}.

No targeting/LQR benchmark is computed here for any of the three cohorts:
none has TES1 stimulation-field coverage (Boran/Rutishauser MTL implants have
no DLPFC electrodes TES1 maps to) -- this is the SAME principled exclusion
already recorded in DATASET_ANALYSIS_MATRIX.md, restated in each cohort's
output rather than silently omitted.

High-load trial selection matches the existing K2 convention exactly
(process_boran_units: set_size==8; _process_load1v3_dynamics: loads==3),
so results are directly comparable to the Round-7 single-rank numbers already
in results/divergence_analysis.npz.

Outputs: results/dmd_extension_boran_units.json,
         results/dmd_extension_dandi001187.json,
         results/dmd_extension_dandi000673.json

Run:
    conda run -n wm_dynamics python scripts/run_dmd_extension_gap_cohorts.py
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

from dynamics import ensemble_dmd
from statistics import stable_seed

RESULTS = ROOT / "results"
RANKS = (4, 5, 6, 7, 8)
MIN_TRIALS = 10
NO_BENCHMARK_NOTE = (
    "No LQR/TES1 targeting benchmark: this cohort has no TES1 stimulation-field "
    "coverage (MTL/hippocampal-amygdala implants; TES1 maps DLPFC-region "
    "electrodes only) -- dynamics-only metrics reported, matching the "
    "principled exclusion already recorded in DATASET_ANALYSIS_MATRIX.md."
)


def _rank_sweep(Z_trials: np.ndarray, dt: float, seed_name: str) -> dict:
    d = Z_trials.shape[2]
    n_trials = Z_trials.shape[0]
    ranks_use = sorted(set(min(r, d) for r in RANKS))
    n_splits = min(5, max(2, n_trials // 3))
    out = {}
    for r in ranks_use:
        rng = np.random.default_rng(stable_seed(f"{seed_name}_r{r}"))
        res = ensemble_dmd(Z_trials, r=r, dt=dt, n_splits=n_splits, n_null=30, rng=rng)
        out[str(r)] = {
            "cv_r2_onestep": res["r2_cv"], "cv_r2_onestep_std": res["r2_cv_std"],
            "r2_insample": res["r2_insample"],
            "r2_null": res["r2_null"], "r2_null_std": res["r2_null_std"],
            "div_scalar": res["div_scalar"],
            "max_re_lambda": float(np.max(res["eigenvalues"].real)),
            "max_abs_lambda": float(np.max(np.abs(res["eigenvalues"]))),
        }
    return out


def process_cohort(glob_pattern: str, prefix: str, condition_field: str,
                   condition_value, cohort_name: str) -> dict:
    per_subject = {}
    for path in sorted(RESULTS.glob(glob_pattern)):
        key = path.stem.replace(prefix, "")
        d = np.load(path, allow_pickle=True)
        if "Z" not in d or condition_field not in d or "times" not in d:
            continue
        Z, cond, times = d["Z"], d[condition_field], d["times"]
        Z_trials = Z[cond == condition_value]
        if Z_trials.shape[0] < MIN_TRIALS:
            print(f"    SKIP {key} -- only {Z_trials.shape[0]} qualifying trials (<{MIN_TRIALS})")
            continue
        dt = float(np.median(np.diff(times)))
        print(f"  {key}: {Z_trials.shape[0]} trials, d={Z_trials.shape[2]}, dt={dt:.4f}s")
        per_subject[key] = {
            "n_trials": int(Z_trials.shape[0]),
            "d_latent": int(Z_trials.shape[2]),
            "by_rank": _rank_sweep(Z_trials, dt, f"{cohort_name}_{key}"),
        }
    return per_subject


def _cohort_summary(per_subject: dict) -> dict:
    if not per_subject:
        return {}
    all_ranks = sorted({int(r) for v in per_subject.values() for r in v["by_rank"]})
    summary = {}
    for r in all_ranks:
        cv = [v["by_rank"][str(r)]["cv_r2_onestep"] for v in per_subject.values() if str(r) in v["by_rank"]]
        maxlam = [v["by_rank"][str(r)]["max_abs_lambda"] for v in per_subject.values() if str(r) in v["by_rank"]]
        summary[str(r)] = {
            "mean_cv_r2_onestep": float(np.mean(cv)), "std_cv_r2_onestep": float(np.std(cv)),
            "mean_max_abs_lambda": float(np.mean(maxlam)), "n_subjects": len(cv),
        }
    return summary


def main():
    cohorts = [
        ("boran_units", "dandi000574_units_geometry_sub-*.npz", "dandi000574_units_geometry_",
         "set_size", 8),
        ("dandi001187", "dandi001187_geometry_sub-*.npz", "dandi001187_geometry_", "loads", 3),
        ("dandi000673", "dandi000673_geometry_sub-*.npz", "dandi000673_geometry_", "loads", 3),
    ]
    for cohort_name, glob_pat, prefix, cond_field, cond_val in cohorts:
        print(f"\n=== {cohort_name} (DMD rank sweep, r in {RANKS}) ===")
        per_subject = process_cohort(glob_pat, prefix, cond_field, cond_val, cohort_name)
        out = {
            "per_subject": per_subject,
            "cohort_summary": _cohort_summary(per_subject),
            "n_subjects": len(per_subject),
            "targeting_benchmark": "N/A",
            "note": NO_BENCHMARK_NOTE,
        }
        if not per_subject:
            out["underpowered"] = True
            out["reason"] = f"no session with >={MIN_TRIALS} qualifying trials"
            print(f"  underpowered -- no session with >={MIN_TRIALS} qualifying trials")
        out_path = RESULTS / f"dmd_extension_{cohort_name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
