#!/usr/bin/env python3
"""Participation-ratio estimator positive control and minimum-detectable-effect
power analysis.

(1) Positive control: injects a known load-dependent dimensionality increase
    into real Miller channel data and evaluates whether
    spatiotemporal_participation_ratio recovers it, via the same
    linear-mixed-effects permutation pipeline used throughout this project
    (src/statistics.linear_mixed_effects_test).
(2) Minimum detectable effect: for each of the six applicable datasets, using
    its own observed between-session participation-ratio residual standard
    deviation (after subject-demeaning) and its own subject count and
    load-level design, simulates the smallest true slope that the same
    permutation test detects at approximately 80% power.

Outputs: results/pr_positive_control.json
Does not write to all_statistics.json here; merged in separately to avoid a
read-modify-write race with concurrent per-dataset pipeline runs.

Run:
    conda run -n wm_dynamics python scripts/run_pr_positive_control.py
"""
import sys, json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry import spatiotemporal_participation_ratio
from statistics import linear_mixed_effects_test

RESULTS = ROOT / "results"
MILLER_SUBJECTS = ["al", "ca", "cc", "ug"]
MAINT_WINDOW = (0.3, 1.4)


# ── (1) Positive control: inject a known PR-vs-load slope into real Miller data ──

def inject_isotropic_noise_by_load(
    X: np.ndarray, load_centered: np.ndarray, gain: float, rng: np.random.Generator,
) -> np.ndarray:
    """Add independent per-channel Gaussian noise, with variance scaling in
    (centered) load, to the real channel data.

    Independent per-channel noise with variance alpha adds alpha*I to the
    channel covariance, which flattens the eigenvalue spectrum toward
    isotropic and therefore increases the participation ratio monotonically
    in alpha.

    Parameters
    ----------
    X : (N, C, T) real per-trial channel data
    load_centered : (N,) load value per trial, mean-subtracted
    gain : dose-response parameter; gain=0 adds load-independent noise (a null
        calibration check), gain>0 adds a load-dependent participation-ratio increase
    """
    N, C, T = X.shape
    per_channel_sd = X.std(axis=(0, 2), keepdims=True)   # (1, C, 1) — natural per-channel scale
    alpha = np.clip(1.0 + gain * load_centered, 0.05, None)   # (N,)
    noise = rng.standard_normal(X.shape).astype(X.dtype) * per_channel_sd
    noise *= alpha[:, None, None]
    return X + noise


def positive_control_miller(gains: list[float], rng: np.random.Generator) -> dict:
    """Dose-response sweep over injection gain: returns the recovered
    participation-ratio-versus-load slope and its p-value at each gain."""
    epochs_by_subj, loads_by_subj = {}, {}
    for subj in MILLER_SUBJECTS:
        dat = np.load(RESULTS / f"01_epochs_{subj}.npz", allow_pickle=True)
        epochs, times, task_id = dat["epochs"], dat["times"], dat["task_id"]
        maint = (times >= MAINT_WINDOW[0]) & (times <= MAINT_WINDOW[1])
        epochs_by_subj[subj] = epochs[:, maint, :].transpose(0, 2, 1)   # (N, C, T)
        loads_by_subj[subj] = task_id.astype(float)

    sweep = []
    for gain in gains:
        records = []
        for subj in MILLER_SUBJECTS:
            X, loads = epochs_by_subj[subj], loads_by_subj[subj]
            load_centered = loads - loads.mean()
            X_aug = inject_isotropic_noise_by_load(X, load_centered, gain, rng)
            for ld in np.unique(loads):
                mask = loads == ld
                res = spatiotemporal_participation_ratio(X_aug[mask], n_splits=2, rng=rng)
                records.append((subj, float(ld), res["pr_cv"]))
        lme = linear_mixed_effects_test(
            np.array([r[2] for r in records]), np.array([r[1] for r in records]),
            np.array([r[0] for r in records]), n_perm=2000, rng=rng,
        )
        sweep.append({"gain": gain, "beta_recovered": lme["beta"], "p_value": lme["p_value"]})
        print(f"    gain={gain:.2f}: beta_recovered={lme['beta']:.4f}, p={lme['p_value']:.4f}")

    return {"sweep": sweep}


# ── (2) Minimum detectable effect, per dataset, from its own observed noise ──

def _residual_sd(records: list[tuple]) -> float:
    """SD of the metric after subject-demeaning — the noise scale the LME
    permutation test actually operates on (matches linear_mixed_effects_test's
    internal residualisation)."""
    by_subj = defaultdict(list)
    for s, l, v in records:
        by_subj[s].append(v)
    resid = []
    for vals in by_subj.values():
        vals = np.array(vals)
        resid.extend((vals - vals.mean()).tolist())
    return float(np.std(resid))


def minimum_detectable_effect(
    subjects: list[str], load_values: list[float], sigma: float,
    rng: np.random.Generator, target_power: float = 0.8,
    n_sim: int = 200, n_perm: int = 500,
) -> dict:
    """Bisection search over true slope beta for the smallest value at which
    the permutation LME test rejects (p<0.05) in >=target_power of simulated
    datasets with this dataset's own subject count, load design, and noise SD.
    """
    subj_arr = np.array([s for s in subjects for _ in load_values])
    load_arr = np.array([l for _ in subjects for l in load_values], dtype=float)
    load_centered = load_arr - load_arr.mean()

    def _power(beta: float) -> float:
        n_sig = 0
        for _ in range(n_sim):
            metric = beta * load_centered + rng.normal(0, sigma, size=len(load_arr))
            res = linear_mixed_effects_test(metric, load_arr, subj_arr, n_perm=n_perm, rng=rng)
            n_sig += int(res["p_value"] < 0.05)
        return n_sig / n_sim

    lo, hi = 0.0, 10.0 * sigma
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if _power(mid) < target_power:
            lo = mid
        else:
            hi = mid
    return {"mde_beta": hi, "sigma_residual": sigma, "n_subjects": len(subjects),
            "load_values": load_values, "achieved_power_at_mde": _power(hi)}


def _build_all_records(stats: dict) -> dict:
    """Per-subject/session (subj, load, pr_cv) records for all six datasets,
    following the extraction in scripts/aggregate_pr_across_datasets.py."""
    out = {}

    records = []
    for subj in MILLER_SUBJECTS:
        d = np.load(RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
        pr, tid = d["pr_per_trial"], d["task_id"]
        for load in [0, 1, 2]:
            records.append((subj, load, float(np.mean(pr[tid == load]))))
    out["miller"] = records

    records = []
    for subj, v in stats["boran_ctg"].items():
        for ss, prd in v["pr_per_set"].items():
            if isinstance(prd, dict) and np.isfinite(prd.get("pr_cv", np.nan)):
                records.append((subj, float(ss), prd["pr_cv"]))
    out["boran_ieeg"] = records

    records = []
    for key, v in stats.get("dandi000574_units_ctg", {}).items():
        for ss, prd in v["pr_per_set"].items():
            if isinstance(prd, dict) and np.isfinite(prd.get("pr_cv", np.nan)):
                records.append((v["subject"], float(ss), prd["pr_cv"]))
    out["boran_units"] = records

    for dset_key, out_key in [("dandi000469_ctg", "dandi000469"),
                              ("dandi001187_ctg", "dandi001187"),
                              ("dandi000673_ctg", "dandi000673")]:
        records = []
        for key, v in stats.get(dset_key, {}).items():
            for ld, prd in v["pr_per_load"].items():
                if isinstance(prd, dict) and np.isfinite(prd.get("pr_cv", np.nan)):
                    records.append((key, float(ld), prd["pr_cv"]))
        out[out_key] = records

    return out


def main():
    rng = np.random.default_rng(0)

    print("(1) Positive control: dose-response sweep, injecting isotropic")
    print("    load-scaled noise into real Miller channel data")
    gains = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    pc = positive_control_miller(gains, rng)
    monotonic = all(pc["sweep"][i]["beta_recovered"] <= pc["sweep"][i + 1]["beta_recovered"] + 1e-6
                    for i in range(len(pc["sweep"]) - 1))
    min_significant_gain = next((s["gain"] for s in pc["sweep"] if s["p_value"] < 0.05), None)
    pc["monotonic"] = monotonic
    pc["min_significant_gain"] = min_significant_gain
    print(f"    monotonic beta-vs-gain: {monotonic}; "
          f"smallest significant gain: {min_significant_gain}")

    print("\n(2) Minimum detectable effect per dataset")
    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)
    all_records = _build_all_records(stats)

    mde_results = {}
    for dset, records in all_records.items():
        subjects = sorted(set(r[0] for r in records))
        load_values = sorted(set(r[1] for r in records))
        sigma = _residual_sd(records)
        mde = minimum_detectable_effect(subjects, load_values, sigma, rng)
        mde_results[dset] = mde
        print(f"    {dset}: N={len(subjects)} subjects, sigma_resid={sigma:.3f}, "
              f"MDE beta={mde['mde_beta']:.3f} (achieved power={mde['achieved_power_at_mde']:.2f})")

    out = {"positive_control": pc, "mde_by_dataset": mde_results}
    with open(RESULTS / "pr_positive_control.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/pr_positive_control.json")


if __name__ == "__main__":
    main()
