#!/usr/bin/env python3
"""Falsification battery: checks that could have contradicted this project's
rotation/geometry claims, run and reported even where they do not.

32A -- WITHIN-DELAY TIMEPOINT SHUFFLE, a stronger null than the existing
  circular-shift null already built into dynamics.ensemble_dmd (a circular
  shift preserves local temporal order, just rotated; a full timepoint
  shuffle destroys temporal order altogether). Applied to (i) the ensemble
  DMD one-step fit, every cohort (reusing run_vstar_eigen_audit.ALL_ITERS,
  no new session loading), and (ii) the two cross-temporal-generalisation
  (CTG) results that already have per-item/content decoding infrastructure
  (DANDI 000469 per-item identity, CRCNS pfc-3 cue location) via
  geometry.ctg_content_permutation_null, called directly on time-shuffled
  input -- no new CTG machinery. If genuine structure survives a full
  temporal shuffle, that is a stop-the-presses finding; it is not expected
  to, and reporting that expectation is met is still the point of the check.
32B -- SPECIFICITY: does the causal alignment target beat BOTH a random
  direction/plane and the stable context direction, not merely nothing? This
  panel does not recompute anything -- it cites the existing joint
  alignment-vs-random-vs-stable regression already in
  results/causal_benchmark.json's "joint"/"nested" keys (built when the
  S_m subspace arms and the top-m-PCA / channel-random-direction controls
  were added to the same macaque PFC microstimulation causal call site).
32C -- ROTATION VS. DISCRETE REGIMES. A dependency gate was checked first
  (pip dry-run): ssm fails to build at all (missing Cython in its build
  backend); dynamax would install but pulls in JAX, tensorflow-probability,
  and an unrelated web-app stack (~30 packages) for one robustness panel --
  both fail this project's "installs cleanly, no disruption" bar. A minimal,
  explicitly-simplified 2-state switching one-step map
  (dynamics.switching_ar_em) is used instead: held-out log-likelihood is
  compared against a matched single-regime null (same function, n_states=1),
  with parameter counts and AIC/BIC reported beside the raw likelihood so
  the larger model cannot "win" by default. Scoped to macaque PFC microstimulation -- the only
  cohort with an actual rotation claim, and the cohort whose
  SINDy "not a clean two-dimensional ring attractor" finding already
  motivates a discrete/higher-dimensional alternative for.

Output: results/falsification_battery.json.

Run (needs the external data mount):
    conda run -n wm_dynamics python scripts/run_falsification_battery.py
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

from dynamics import ensemble_dmd, switching_ar_em, switching_ar_score  # noqa: E402
from geometry import ctg_content_permutation_null  # noqa: E402
from statistics import stable_seed  # noqa: E402

import run_vstar_eigen_audit as vea  # noqa: E402  (ALL_ITERS, _iter_macaque_pfc_microstimulation)
from run_multiitem_ctg_000469 import DATA_DIR as D469_DATA_DIR  # noqa: E402
from run_multiitem_ctg_000469 import ITEM_FIELDS, _class_counts_ok  # noqa: E402

RESULTS = ROOT / "results"
N_NULL_DMD = 20
N_PC_CTG = 8
N_PERM_CTG = 5  # this check only uses ctg_content_permutation_null's observed tau/offdiag
                # (computed before its internal null loop runs), never its label-shuffle
                # p-value, so a minimal permutation count is used deliberately for speed


def _timepoint_shuffle_trials(Z_trials: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Z_out = np.empty_like(Z_trials)
    for i in range(Z_trials.shape[0]):
        Z_out[i] = Z_trials[i][rng.permutation(Z_trials.shape[1])]
    return Z_out


def _shuffle_ctg_input(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """X: (N, C, T) -- shuffle the time axis independently per trial."""
    X_out = np.empty_like(X)
    for i in range(X.shape[0]):
        X_out[i] = X[i][:, rng.permutation(X.shape[2])]
    return X_out


# ── 32A(i): DMD one-step fit under a full timepoint shuffle ────────────────

def run_32a_dmd() -> dict:
    out: dict[str, dict] = {}
    for it in vea.ALL_ITERS:
        for dataset, session, Z_trials, dt, r in it():
            out.setdefault(dataset, {})
            N, T, d = Z_trials.shape
            r_use = min(r, d, N * (T - 1) - 1)
            rng = np.random.default_rng(stable_seed(f"falsification_32a_dmd_{dataset}_{session}"))
            ens_real = ensemble_dmd(Z_trials, r=r_use, dt=dt, n_splits=5, n_null=N_NULL_DMD, rng=rng)
            Z_shuf = _timepoint_shuffle_trials(Z_trials, rng)
            ens_shuf = ensemble_dmd(Z_shuf, r=r_use, dt=dt, n_splits=5, n_null=N_NULL_DMD, rng=rng)
            out[dataset][session] = {
                "r2_cv_real": ens_real["r2_cv"], "r2_null_circular_shift": ens_real["r2_null"],
                "r2_cv_timepoint_shuffle": ens_shuf["r2_cv"], "n_trials": int(N),
            }
    summary = {}
    for dataset, sessions in out.items():
        real = np.array([v["r2_cv_real"] for v in sessions.values()])
        circ = np.array([v["r2_null_circular_shift"] for v in sessions.values()])
        shuf = np.array([v["r2_cv_timepoint_shuffle"] for v in sessions.values()])
        summary[dataset] = {"n_sessions": len(sessions), "mean_r2_cv_real": float(np.mean(real)),
                            "mean_r2_null_circular_shift": float(np.mean(circ)),
                            "mean_r2_cv_timepoint_shuffle": float(np.mean(shuf))}
        print(f"  {dataset:14s} (N={len(sessions):3d}): real={summary[dataset]['mean_r2_cv_real']:.3f} "
              f"circular-shift-null={summary[dataset]['mean_r2_null_circular_shift']:.3f} "
              f"timepoint-shuffle={summary[dataset]['mean_r2_cv_timepoint_shuffle']:.3f}")
    return {"per_session": out, "summary": summary}


# ── 32A(ii): CTG structure under a full timepoint shuffle ──────────────────

def run_32a_ctg() -> dict:
    results: dict = {"dandi000469": {}}
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
        with h5py.File(str(nwb_path), "r") as f:
            item_labels = {name: f[f"intervals/trials/{field}"][:].astype(int)
                           for name, field in ITEM_FIELDS.items()}
        X = Z[mask].transpose(0, 2, 1)  # (N, C, T)
        t_idx = np.arange(0, X.shape[2], 3)
        rng = np.random.default_rng(stable_seed(f"falsification_32a_ctg_469_{subj}"))
        for name in ITEM_FIELDS:
            labels = item_labels[name][mask]
            if not _class_counts_ok(labels):
                continue
            res_real = ctg_content_permutation_null(X, labels, t_idx, n_components=N_PC_CTG,
                                                     n_splits=3, n_perm=N_PERM_CTG, rng=rng)
            X_shuf = _shuffle_ctg_input(X, rng)
            res_shuf = ctg_content_permutation_null(X_shuf, labels, t_idx, n_components=N_PC_CTG,
                                                     n_splits=3, n_perm=N_PERM_CTG, rng=rng)
            results["dandi000469"].setdefault(subj, {})[name] = {
                "tau_real": res_real["tau"], "offdiag_real": res_real["mean_offdiag_auc_minus_chance"],
                "tau_timepoint_shuffle": res_shuf["tau"],
                "offdiag_timepoint_shuffle": res_shuf["mean_offdiag_auc_minus_chance"],
            }

    d = np.load(RESULTS / "pfc3_content_ctg.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    t_idx = np.arange(0, X.shape[2], 2)
    rng = np.random.default_rng(stable_seed("falsification_32a_ctg_pfc3"))
    res_real = ctg_content_permutation_null(X, y, t_idx, n_components=N_PC_CTG, n_splits=3,
                                             n_perm=N_PERM_CTG, rng=rng)
    X_shuf = _shuffle_ctg_input(X, rng)
    res_shuf = ctg_content_permutation_null(X_shuf, y, t_idx, n_components=N_PC_CTG, n_splits=3,
                                            n_perm=N_PERM_CTG, rng=rng)
    results["pfc3"] = {"tau_real": res_real["tau"], "offdiag_real": res_real["mean_offdiag_auc_minus_chance"],
                       "tau_timepoint_shuffle": res_shuf["tau"],
                       "offdiag_timepoint_shuffle": res_shuf["mean_offdiag_auc_minus_chance"]}

    n_469 = sum(len(v) for v in results["dandi000469"].values())
    print(f"  DANDI 000469 ({n_469} session x item cells): see per-cell tau collapse in JSON")
    print(f"  pfc-3: tau_real={results['pfc3']['tau_real']:.4f} "
          f"tau_timepoint_shuffle={results['pfc3']['tau_timepoint_shuffle']:.4f}")
    return results


# ── 32B: cite the existing specificity regression, do not recompute ────────

def run_32b() -> dict:
    bench = json.load(open(RESULTS / "causal_benchmark.json"))
    joint, nested = bench["joint"], bench["nested"]
    controls = ["stable_alignment", "random_alignment", "topm_pca_alignment_m2",
               "topm_pca_alignment_m3", "channel_random_alignment"]
    out = {
        "vstar_alignment_joint_coef": joint["vstar_alignment"]["coef"],
        "vstar_alignment_joint_p": joint["vstar_alignment"]["p_value"],
        "align_s_m2_joint_coef": joint.get("align_s_m2", {}).get("coef"),
        "align_s_m2_joint_p": joint.get("align_s_m2", {}).get("p_value"),
        "controls_in_same_joint_model": {c: joint.get(c) for c in controls if c in joint},
        "controls_nested_dR2": {c: nested.get(c) for c in controls if c in nested},
        "source": "results/causal_benchmark.json ('joint'/'nested' keys) -- not recomputed",
    }
    beats_both = (out["vstar_alignment_joint_p"] < 0.05
                 and out["controls_in_same_joint_model"]["stable_alignment"]["p_value"] >= 0.05
                 and out["controls_in_same_joint_model"]["random_alignment"]["p_value"] >= 0.05)
    out["verdict"] = ("v* alignment remains significant in the SAME joint model where both the "
                      "random-direction and stable-context controls do not -- specificity holds"
                      if beats_both else
                      "v* alignment does not clearly dominate both controls in the joint model -- "
                      "see numbers, not overstated")
    print(f"  joint model: vstar_alignment coef={out['vstar_alignment_joint_coef']:+.4f} "
          f"p={out['vstar_alignment_joint_p']:.4g}; stable_alignment p="
          f"{out['controls_in_same_joint_model']['stable_alignment']['p_value']:.4g}; "
          f"random_alignment p={out['controls_in_same_joint_model']['random_alignment']['p_value']:.4g}")
    print(f"  -> {out['verdict']}")
    return out


# ── 32C: rotation vs. a minimal discrete-regime alternative (macaque PFC microstimulation) ──────

def run_32c() -> dict:
    out = {}
    for dataset, session, Z_trials, dt, r in vea._iter_macaque_pfc_microstimulation():
        N, T, d = Z_trials.shape
        if N < 20:
            continue
        X1 = Z_trials[:, :-1, :].reshape(-1, d)
        X2 = Z_trials[:, 1:, :].reshape(-1, d)
        n = X1.shape[0]
        split_rng = np.random.default_rng(stable_seed(f"falsification_32c_split_{session}"))
        idx = split_rng.permutation(n)
        n_tr = int(0.7 * n)
        tr, te = idx[:n_tr], idx[n_tr:]

        fit_rng = np.random.default_rng(stable_seed(f"falsification_32c_fit_{session}"))
        fit1 = switching_ar_em(X1[tr], X2[tr], n_states=1, rng=fit_rng)
        fit2 = switching_ar_em(X1[tr], X2[tr], n_states=2, rng=fit_rng)
        ll1 = switching_ar_score(fit1, X1[te], X2[te])
        ll2 = switching_ar_score(fit2, X1[te], X2[te])
        n_te = len(te)
        aic1, aic2 = -2 * ll1 + 2 * fit1["n_params"], -2 * ll2 + 2 * fit2["n_params"]
        bic1 = -2 * ll1 + fit1["n_params"] * np.log(n_te)
        bic2 = -2 * ll2 + fit2["n_params"] * np.log(n_te)
        out[session] = {
            "held_out_loglik_1state": ll1, "held_out_loglik_2state": ll2,
            "n_params_1state": fit1["n_params"], "n_params_2state": fit2["n_params"],
            "aic_1state": aic1, "aic_2state": aic2, "bic_1state": bic1, "bic_2state": bic2,
            "n_test_pairs": int(n_te), "winner_by_bic": "2-state" if bic2 < bic1 else "1-state",
        }
        print(f"  {session}: loglik 1-state={ll1:.1f} (p={fit1['n_params']}) 2-state={ll2:.1f} "
              f"(p={fit2['n_params']}) BIC 1-state={bic1:.1f} 2-state={bic2:.1f} "
              f"-> {out[session]['winner_by_bic']} wins by BIC")

    n_2state_wins = sum(1 for v in out.values() if v["winner_by_bic"] == "2-state")
    verdict = (f"{n_2state_wins}/{len(out)} sessions favour the 2-state model by BIC (parameter-count "
              f"penalised) -- {'a genuine, prior-consistent finding' if n_2state_wins > len(out) / 2 else 'not a systematic preference for discrete regimes'}")
    print(f"\n  {verdict}")
    return {"per_session": out, "n_sessions": len(out), "n_2state_wins_by_bic": n_2state_wins,
            "verdict": verdict}


def main():
    print("32A(i) -- DMD one-step fit under a within-delay timepoint shuffle, all cohorts ...")
    out_32a_dmd = run_32a_dmd()

    print("\n32A(ii) -- CTG structure under a within-delay timepoint shuffle ...")
    out_32a_ctg = run_32a_ctg()

    print("\n32B -- specificity: v* vs. random and stable controls, same joint model ...")
    out_32b = run_32b()

    print("\n32C -- rotation vs. a minimal 2-state discrete-regime alternative (macaque PFC microstimulation) ...")
    out_32c = run_32c()

    out = {"32A_dmd_timepoint_shuffle": out_32a_dmd, "32A_ctg_timepoint_shuffle": out_32a_ctg,
          "32B_specificity": out_32b, "32C_discrete_regimes": out_32c}
    with open(RESULTS / "falsification_battery.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/falsification_battery.json")


if __name__ == "__main__":
    main()
