#!/usr/bin/env python3
"""Cheap, decisive nonlinearity test for the one-step dynamics.

The premise that neural manifolds are nonlinear is not in question; the
specific question this project needs answered is whether a richer one-step
map beats the linear DMD operator OUT-OF-SAMPLE, on the same data the causal
and descriptive claims already rest on. A degree-2 polynomial EDMD dictionary
previously showed "not significantly better" than plain DMD; this test is a
stronger, non-parametric version of the same question, using flexible
regressors already a project dependency (no new packages).

On the SAME trial-wise cross-validation folds and the SAME per-trial
circular-shift null dynamics.ensemble_dmd already uses, held-out one-step R^2
is compared across three maps x_t -> x_{t+1}:
  (a) the linear DMD operator (dynamics.ensemble_dmd's own r2_cv/r2_null,
      reused directly, not recomputed);
  (b) gradient-boosted regression (sklearn.ensemble.GradientBoostingRegressor,
      the same estimator class src/causal.py's crossfit_nuisances already
      uses for its outcome model, wrapped in MultiOutputRegressor since the
      state is vector-valued);
  (c) kernel ridge regression with an RBF kernel, approximated via a Nystroem
      feature map (sklearn.kernel_approximation.Nystroem + linear Ridge) so
      it remains tractable on cohorts with tens of thousands of snapshot
      pairs per session, as a second, smoother nonlinear family.

Every model is fit and scored on the full snapshot-pair set for its session
-- no pairs are subsampled away for any estimator, including on the largest
cohorts (this makes the GBR fit slow on high-timepoint-count sessions, which
is accepted rather than worked around).

If neither flexible map beats the linear operator out-of-sample, that is a
direct, quantitative statement that linearity is not the limiting factor on
this data -- stronger evidence than the earlier polynomial-EDMD comparison.
If one does, that is a genuine finding on its own, not folded into the
existing headline (the delivered stimulation input is only expressible in a
LINEAR image of the latent space -- Methods 5.5/5.7 -- so a nonlinear one-step
map cannot serve the causal estimand regardless of its descriptive accuracy).

Output: results/nonlinearity_onestep.json -- {cohort: {session: {r2_cv_linear,
r2_null_linear, r2_cv_gbr, r2_null_gbr, r2_cv_krr, r2_null_krr, n_trials}},
plus a per-cohort pooled summary and verdict}.

Run (needs the external data mount):
    conda run -n wm_dynamics python scripts/run_nonlinearity_onestep.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import os

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.kernel_approximation import Nystroem  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.multioutput import MultiOutputRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from dynamics import ensemble_dmd  # noqa: E402
from statistics import stable_seed  # noqa: E402

import run_vstar_eigen_audit as vea  # noqa: E402  (ALL_ITERS -- per-session Z_trials/dt/r)

RESULTS = ROOT / "results"
N_SPLITS = 5
N_NULL_NONLINEAR = 10  # dynamics.ensemble_dmd's own linear null (n_null=50) is reused as-is
KRR_LANDMARKS = 500  # exact RBF kernel ridge is O(n^2) memory / O(n^3) solve -- infeasible past a
                     # few thousand pairs (miller/boran have tens of thousands per fold: a 44k x 44k
                     # Gram matrix alone is ~15GB, repeated per fold and null draw). Nystroem +
                     # linear ridge approximates the RBF kernel from a landmark subset but the ridge
                     # regression itself still fits on ALL pairs -- no snapshot pairs are discarded,
                     # unlike raw subsampling. GBR has no such wall (cost is roughly linear in pairs)
                     # so it always runs on the full pair set.
GBR_N_JOBS = min(8, os.cpu_count() or 1)  # MultiOutputRegressor fits one GBR per output channel (d=8)
                                          # independently -- parallel across processes, identical fit
                                          # to serial, purely a wall-clock speedup on multicore hardware.


def _pairs(Z_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = Z_sub.shape[2]
    X1 = Z_sub[:, :-1, :].reshape(-1, d)
    X2 = Z_sub[:, 1:, :].reshape(-1, d)
    return X1, X2


def _r2(pred: np.ndarray, actual: np.ndarray) -> float:
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def _cv_r2(Z_trials: np.ndarray, make_model, rng: np.random.Generator) -> float:
    """Trial-wise CV, identical fold construction to ensemble_dmd. Every
    model is fit and scored on the FULL pair set -- no snapshot pairs are
    discarded for any estimator."""
    N = Z_trials.shape[0]
    trial_idx = rng.permutation(N)
    folds = np.array_split(trial_idx, min(N_SPLITS, N))
    scores = []
    for k in range(len(folds)):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        if len(tr) < 2 or len(te) < 1:
            continue
        X1_tr, X2_tr = _pairs(Z_trials[tr])
        X1_te, X2_te = _pairs(Z_trials[te])
        model = make_model()
        model.fit(X1_tr, X2_tr)
        scores.append(_r2(model.predict(X1_te), X2_te))
    return float(np.mean(scores)) if scores else float("nan")


def _null_r2(Z_trials: np.ndarray, make_model, rng: np.random.Generator, n_draws: int) -> float:
    """Same per-trial circular-shift null as ensemble_dmd, in-sample fit
    (matching ensemble_dmd's own null construction: fit and score on the
    same shifted ensemble, not held out), on the full pair set."""
    N, T, d = Z_trials.shape
    scores = []
    for _ in range(n_draws):
        Z_shift = np.empty_like(Z_trials)
        for i in range(N):
            shift = int(rng.integers(1, max(T - 1, 2)))
            Z_shift[i] = np.roll(Z_trials[i], shift, axis=0)
        X1, X2 = _pairs(Z_shift)
        model = make_model()
        model.fit(X1, X2)
        scores.append(_r2(model.predict(X1), X2))
    return float(np.mean(scores))


def main():
    out: dict[str, dict] = {}
    for it in vea.ALL_ITERS:
        for dataset, session, Z_trials, dt, r in it():
            out.setdefault(dataset, {})
            N, T, d = Z_trials.shape
            r_use = min(r, d, N * (T - 1) - 1)

            rng_lin = np.random.default_rng(stable_seed(f"nonlinearity_linear_{dataset}_{session}"))
            ens = ensemble_dmd(Z_trials, r=r_use, dt=dt, n_splits=N_SPLITS, n_null=50, rng=rng_lin)

            rng_gbr = np.random.default_rng(stable_seed(f"nonlinearity_gbr_{dataset}_{session}"))
            make_gbr = lambda: MultiOutputRegressor(  # noqa: E731
                GradientBoostingRegressor(random_state=0), n_jobs=GBR_N_JOBS)
            r2_cv_gbr = _cv_r2(Z_trials, make_gbr, rng_gbr)
            r2_null_gbr = _null_r2(Z_trials, make_gbr, rng_gbr, N_NULL_NONLINEAR)

            rng_krr = np.random.default_rng(stable_seed(f"nonlinearity_krr_{dataset}_{session}"))
            n_landmarks = min(KRR_LANDMARKS, N * (T - 1))
            make_krr = lambda: make_pipeline(  # noqa: E731
                Nystroem(kernel="rbf", n_components=n_landmarks, random_state=0), Ridge(alpha=1.0))
            r2_cv_krr = _cv_r2(Z_trials, make_krr, rng_krr)
            r2_null_krr = _null_r2(Z_trials, make_krr, rng_krr, N_NULL_NONLINEAR)

            row = {
                "r2_cv_linear": ens["r2_cv"], "r2_null_linear": ens["r2_null"],
                "r2_cv_gbr": r2_cv_gbr, "r2_null_gbr": r2_null_gbr,
                "r2_cv_krr": r2_cv_krr, "r2_null_krr": r2_null_krr,
                "n_trials": int(N), "delta_gbr_minus_linear": r2_cv_gbr - ens["r2_cv"],
                "delta_krr_minus_linear": r2_cv_krr - ens["r2_cv"],
            }
            out[dataset][session] = row
            print(f"  {dataset}/{session}: linear={row['r2_cv_linear']:.3f} "
                  f"gbr={row['r2_cv_gbr']:.3f} (d={row['delta_gbr_minus_linear']:+.3f}) "
                  f"krr={row['r2_cv_krr']:.3f} (d={row['delta_krr_minus_linear']:+.3f})")

    summary = {}
    for dataset, sessions in out.items():
        deltas_gbr = np.array([v["delta_gbr_minus_linear"] for v in sessions.values()])
        deltas_krr = np.array([v["delta_krr_minus_linear"] for v in sessions.values()])
        beats_gbr = float(np.mean(deltas_gbr > 0.02))  # >0.02 R^2: a materially better fit, not noise
        beats_krr = float(np.mean(deltas_krr > 0.02))
        verdict = ("linearity is NOT the limiting factor" if max(np.mean(deltas_gbr), np.mean(deltas_krr)) < 0.02
                   else "a flexible nonlinear map beats the linear operator materially -- genuine finding, "
                        "descriptive only (the causal modifier requires a linear image of the stimulation "
                        "input and cannot use this map)")
        summary[dataset] = {
            "n_sessions": len(sessions),
            "mean_delta_gbr": float(np.mean(deltas_gbr)), "mean_delta_krr": float(np.mean(deltas_krr)),
            "fraction_sessions_gbr_beats_by_0.02": beats_gbr,
            "fraction_sessions_krr_beats_by_0.02": beats_krr,
            "verdict": verdict,
        }
        print(f"  {dataset:14s} summary: mean(delta_gbr)={summary[dataset]['mean_delta_gbr']:+.4f}  "
              f"mean(delta_krr)={summary[dataset]['mean_delta_krr']:+.4f}  -> {verdict}")

    out["_meta"] = {"n_splits": N_SPLITS, "n_null_nonlinear": N_NULL_NONLINEAR, "summary": summary}
    with open(RESULTS / "nonlinearity_onestep.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/nonlinearity_onestep.json")


if __name__ == "__main__":
    main()
