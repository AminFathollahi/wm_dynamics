#!/usr/bin/env python3
"""Round-8 Part 2: link maintenance-window geometry to GRADED behavior (response
time), as an OBSERVATIONAL READOUT ONLY -- never a controller objective (see
comments.txt SCIENTIFIC GUARDRAIL). Binary correct/error is ceiling-limited
(behavior_ctg.json: significant in only 1/5 cohorts; behavior_geometry_link.json:
entirely null) -- RT is graded and available at full trial counts in every
cohort, giving real variance to test against.

Datasets INSPECTED (not assumed) for a graded behavioral field aligned to the
same trials table the maintenance geometry (Z, drift) was computed from:
  - dandi000469 (Rutishauser, intervals/trials): no bare RT field; RT =
    timestamps_Response - timestamps_Probe (both present).
  - dandi001187 (Rutishauser, intervals/WM_trials): response_time field
    present and already trial-relative (verified against
    timestamps_Response - timestamps_Probe on real data -- matches to <0.03s).
    A genuine subject-reported "confidence" field EXISTS in this dataset, but
    only in the SEPARATE intervals/LTM_trials table (a different,
    long-term-memory recognition task) -- NOT aligned to WM_trials / the
    maintenance geometry this script uses, so it is not usable here and is not
    fabricated as if it were.
  - dandi000673 (Rutishauser, intervals/trials): no bare RT field, no
    confidence field; RT = timestamps_Response - timestamps_Probe.
  - boran iEEG (000574, intervals/trials): response_time field present,
    already trial-relative once trial start_time is subtracted (matches
    run_boran_pipeline.py's own convention exactly). No confidence field.
No dataset here has a genuine subject-reported confidence rating aligned to
its maintenance-geometry trials -- confidence is therefore reported as
excluded_no_graded_behavior (reason: field absent or task-misaligned), not
computed via a decoder-confidence proxy, which would share its source latent
Z with the geometry predictor (drift) and be circular rather than
independent. RT is the only graded variable used.

Geometry predictor: per-trial `drift` (Euclidean distance from the
condition/set-size centroid trajectory during the maintenance window,
geometry.geometric_drift / spike_pipeline.correct_error_drift) -- already
computed and saved in every *_geometry_*.npz this project produces (it is the
same scalar the correct-vs-error drift LME in run_boran_pipeline.py and the
three run_0*_pipeline.py scripts already use), reused here rather than
inventing a new one.

Model: per-trial LME (statistics.linear_mixed_effects_test) of RT ~ drift,
subject as random intercept (permutation null, within-subject label shuffle).
95% CI via subject-level percentile bootstrap. Datasets meta-combined with
statistics.forest_meta (inverse-variance) and statistics.stouffer_combine
(p-values). <2 subjects or <30 trials with the variable -> underpowered, no
beta reported for that dataset.

Outputs: results/geometry_graded_behavior.json
Self-check: tests/test_geometry_graded_behavior.py (LME recovers a planted
synthetic slope).

Run:
    conda run -n wm_dynamics python scripts/run_geometry_graded_behavior.py
"""
from __future__ import annotations

import sys
import json
import glob
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import h5py
from statistics import linear_mixed_effects_test, forest_meta, stouffer_combine, stable_seed

RESULTS = ROOT / "results"
DATA_ROOT = Path("/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory/data")

MIN_SUBJECTS = 2
MIN_TRIALS = 30
N_PERM = 5000
N_BOOT = 2000


def bootstrap_ci_beta(drift: np.ndarray, y: np.ndarray, subj: np.ndarray,
                      n_boot: int = N_BOOT, rng: np.random.Generator | None = None) -> dict:
    """Percentile-bootstrap 95% CI on the LME beta, resampling SUBJECTS (not
    trials) with replacement to match the subject-random-effect structure."""
    if rng is None:
        rng = np.random.default_rng(0)
    subs = np.unique(subj)
    betas = np.zeros(n_boot)
    for b in range(n_boot):
        samp = rng.choice(subs, size=len(subs), replace=True)
        idx_parts, subj_parts = [], []
        for i, s in enumerate(samp):
            rows = np.where(subj == s)[0]
            idx_parts.append(rows)
            subj_parts.append(np.full(len(rows), i))  # re-key duplicate draws uniquely
        idx = np.concatenate(idx_parts)
        resamp_subj = np.concatenate(subj_parts)
        # metric=y (RT), condition=drift -- must match _lme_report's call order
        # (linear_mixed_effects_test(rt, drift, subj, ...)) or the bootstrap
        # estimates a different regression than the point estimate.
        res = linear_mixed_effects_test(y[idx], drift[idx], resamp_subj, n_perm=1,
                                        rng=np.random.default_rng(0))
        betas[b] = res["beta"]
    lo, hi = np.percentile(betas, [2.5, 97.5])
    se = (hi - lo) / (2 * 1.96)
    return {"ci_lo": float(lo), "ci_hi": float(hi), "se": float(se)}


def _lme_report(drift: np.ndarray, rt: np.ndarray, subj: np.ndarray, seed_name: str) -> dict:
    finite = np.isfinite(drift) & np.isfinite(rt)
    drift, rt, subj = drift[finite], rt[finite], subj[finite]
    n_subjects = len(np.unique(subj))
    if n_subjects < MIN_SUBJECTS or len(drift) < MIN_TRIALS:
        return {"underpowered": True, "n_trials": int(len(drift)), "n_subjects": int(n_subjects)}
    res = linear_mixed_effects_test(rt, drift, subj, n_perm=N_PERM,
                                    rng=np.random.default_rng(stable_seed(seed_name)))
    ci = bootstrap_ci_beta(drift, rt, subj, rng=np.random.default_rng(stable_seed(seed_name + "_boot")))
    return {
        "beta": res["beta"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"], "se": ci["se"],
        "p_value": res["p_value"], "r_squared": res["r_squared"],
        "n_trials": int(len(drift)), "n_subjects": int(n_subjects),
    }


# ── Per-dataset extraction ──────────────────────────────────────────────────

def run_dandi000469() -> dict:
    all_drift, all_rt, all_subj = [], [], []
    for path in sorted(RESULTS.glob("dandi000469_geometry_sub-*.npz")):
        subj = path.stem.replace("dandi000469_geometry_", "")
        d = np.load(path, allow_pickle=True)
        if "drift" not in d:
            continue
        nwb = DATA_ROOT / "000469" / subj / f"{subj}_ses-2_ecephys+image.nwb"
        if not nwb.exists():
            continue
        with h5py.File(str(nwb), "r") as f:
            trials = f["intervals/trials"]
            t_probe = trials["timestamps_Probe"][:]
            t_resp = trials["timestamps_Response"][:]
        rt = t_resp - t_probe
        drift = d["drift"]
        if len(rt) != len(drift):
            continue
        all_drift.append(drift); all_rt.append(rt); all_subj.append([subj] * len(drift))
    if not all_drift:
        return {"excluded_no_graded_behavior": True, "reason": "no geometry files with drift + matching NWB"}
    return _lme_report(np.concatenate(all_drift), np.concatenate(all_rt),
                       np.concatenate(all_subj), "geobeh_000469")


def run_dandi001187() -> dict:
    all_drift, all_rt, all_subj = [], [], []
    for path in sorted(RESULTS.glob("dandi001187_geometry_sub-*.npz")):
        key = path.stem.replace("dandi001187_geometry_", "")
        subj_dir = key.split("_", 1)[0]
        d = np.load(path, allow_pickle=True)
        if "drift" not in d:
            continue
        nwb = DATA_ROOT / "001187" / subj_dir / f"{key}.nwb"
        if not nwb.exists():
            continue
        with h5py.File(str(nwb), "r") as f:
            rt = f["intervals/WM_trials/response_time"][:]
        drift = d["drift"]
        if len(rt) != len(drift):
            continue
        all_drift.append(drift); all_rt.append(rt); all_subj.append([subj_dir] * len(drift))
    if not all_drift:
        return {"excluded_no_graded_behavior": True, "reason": "no geometry files with drift + matching NWB"}
    return _lme_report(np.concatenate(all_drift), np.concatenate(all_rt),
                       np.concatenate(all_subj), "geobeh_001187")


def run_dandi000673() -> dict:
    all_drift, all_rt, all_subj = [], [], []
    for path in sorted(RESULTS.glob("dandi000673_geometry_sub-*.npz")):
        key = path.stem.replace("dandi000673_geometry_", "")
        subj_dir = key.split("_", 1)[0]
        d = np.load(path, allow_pickle=True)
        if "drift" not in d:
            continue
        nwb = DATA_ROOT / "000673" / subj_dir / f"{key}.nwb"
        if not nwb.exists():
            continue
        with h5py.File(str(nwb), "r") as f:
            trials = f["intervals/trials"]
            t_probe = trials["timestamps_Probe"][:]
            t_resp = trials["timestamps_Response"][:]
        rt = t_resp - t_probe
        drift = d["drift"]
        if len(rt) != len(drift):
            continue
        all_drift.append(drift); all_rt.append(rt); all_subj.append([subj_dir] * len(drift))
    if not all_drift:
        return {"excluded_no_graded_behavior": True, "reason": "no geometry files with drift + matching NWB"}
    return _lme_report(np.concatenate(all_drift), np.concatenate(all_rt),
                       np.concatenate(all_subj), "geobeh_000673")


def run_boran() -> dict:
    all_drift, all_rt, all_subj = [], [], []
    for path in sorted(RESULTS.glob("boran_geometry_sub-*.npz")):
        subj = path.stem.replace("boran_geometry_", "")
        d = np.load(path, allow_pickle=True)
        if "drift" not in d:
            continue
        if "response_time" in d:
            rt = d["response_time"]
        else:
            # Fallback: boran_geometry_*.npz predates the response_time field
            # (added Round-8) -- re-derive directly from NWB, same convention
            # as run_boran_pipeline.load_subject_sessions (response_time is an
            # absolute NWB timestamp; subtract trial start_time for latency).
            nwbs = sorted((DATA_ROOT / "000574" / subj).glob("*.nwb"))
            rt_parts = []
            for nwb_path in nwbs:
                with h5py.File(str(nwb_path), "r") as f:
                    t_start = f["intervals/trials/start_time"][:]
                    resp = f["intervals/trials/response_time"][:] - t_start
                    artifact = f["intervals/trials/artifact"][:].astype(bool)
                rt_parts.append(resp[~artifact])
            rt = np.concatenate(rt_parts) if rt_parts else np.array([])
        drift = d["drift"]
        if len(rt) != len(drift):
            continue
        all_drift.append(drift); all_rt.append(rt); all_subj.append([subj] * len(drift))
    if not all_drift:
        return {"excluded_no_graded_behavior": True, "reason": "no geometry files with drift + matching RT"}
    return _lme_report(np.concatenate(all_drift), np.concatenate(all_rt),
                       np.concatenate(all_subj), "geobeh_boran")


CONFIDENCE_NOTE = (
    "No dataset used for maintenance geometry has a genuine subject-reported "
    "confidence field aligned to its geometry trials. dandi001187 has a "
    "'confidence' field, but only in the separate intervals/LTM_trials "
    "(long-term-memory recognition) task table, not intervals/WM_trials (the "
    "Sternberg maintenance task the geometry/drift here is computed from). A "
    "decoder-derived confidence proxy was considered and rejected: it would "
    "be computed from the same latent Z as the drift predictor (held-out CV, "
    "so not perfectly circular, but not independent either), which would "
    "confound the geometry-behavior link this analysis is meant to test."
)


def main():
    out = {}
    for name, fn in [("dandi000469", run_dandi000469), ("dandi001187", run_dandi001187),
                     ("dandi000673", run_dandi000673), ("boran_ieeg", run_boran)]:
        print(f"\n=== {name} (RT ~ drift) ===")
        res = fn()
        out[name] = {"response_time": res, "confidence": {
            "excluded_no_graded_behavior": True, "reason": CONFIDENCE_NOTE}}
        if res.get("underpowered"):
            print(f"  underpowered: n_trials={res['n_trials']}, n_subjects={res['n_subjects']}")
        elif res.get("excluded_no_graded_behavior"):
            print(f"  excluded: {res['reason']}")
        else:
            print(f"  beta={res['beta']:.4f} [{res['ci_lo']:.4f}, {res['ci_hi']:.4f}] "
                  f"p={res['p_value']:.4f} n_trials={res['n_trials']} n_subjects={res['n_subjects']}")

    # Meta-combine the scored (non-underpowered, non-excluded) datasets.
    scored = {k: v["response_time"] for k, v in out.items()
             if "beta" in v["response_time"]}
    if len(scored) >= 2:
        labels = list(scored.keys())
        estimates = np.array([scored[k]["beta"] for k in labels])
        ses = np.array([scored[k]["se"] for k in labels])
        meta = forest_meta(estimates, ses, labels=labels)
        # One-sided p per dataset in the direction of its own beta sign, for Stouffer.
        p_one_sided = np.array([
            scored[k]["p_value"] / 2 if scored[k]["beta"] >= 0 else 1 - scored[k]["p_value"] / 2
            for k in labels
        ])
        stouffer = stouffer_combine(p_one_sided, weights=np.sqrt([scored[k]["n_trials"] for k in labels]))
        out["_meta"] = {"forest": meta, "stouffer": stouffer, "datasets_pooled": labels}
        print(f"\n=== Meta (RT ~ drift), {len(labels)} cohorts ===")
        print(f"  pooled beta={meta['pooled']:.4f} [{meta['ci_lo']:.4f}, {meta['ci_hi']:.4f}] "
              f"p={meta['p_value']:.4f}, I^2={meta['i_squared']:.1f}%")
    else:
        out["_meta"] = {"note": f"only {len(scored)} scored cohort(s) -- meta-analysis needs >=2"}
        print(f"\nOnly {len(scored)} scored cohort(s) -- skipping meta-analysis")

    out["_confidence_note"] = CONFIDENCE_NOTE

    with open(RESULTS / "geometry_graded_behavior.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/geometry_graded_behavior.json")


if __name__ == "__main__":
    main()
