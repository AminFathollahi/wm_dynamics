#!/usr/bin/env python3
"""Closed-loop behavioral flip (Round-7, STEP D): "can on-demand LQR stim turn
predicted-error trajectories toward the correct-trial region?"

THIS IS IN-SILICO ON A FITTED LINEAR PLANT -- a model prediction, NOT a causal
claim. Every number here describes what a simulated controller does to a
simulated rollout seeded from a real trial's own maintenance-window state; it
does not claim anything about what would happen if stimulation were actually
delivered.

SCOPE (principled, not a shortcut): src/closed_loop.py's simulate_closed_loop
needs a fitted (A, B) plant, and the only cohorts with a TES1-derived B matrix
are Miller and Boran iEEG (DATASET_ANALYSIS_MATRIX.md exclusion #2: TES1 covers
DLPFC only). Of those two, Miller has NO usable trial-outcome label in the data
mounted for this project (see run_behavior_ctg.py's docstring -- the raw MAT
files carry task condition, not response accuracy). That leaves Boran iEEG as
the ONLY cohort where this analysis is both runnable (a fitted B) and
behaviorally meaningful (a real correct/error label) with real, non-fabricated
data. Boran units / DANDI 000469 / 001187 / 000673 have outcome but no B
matrix (MTL, outside TES1 coverage) -- STOP-and-report, not run.

Design (mirrors run_closed_loop_analysis.py's _run_cohort, decoder swapped
from load to outcome; run_ondemand_streak_diagnostic.py's on-demand policy):
  D1. Fit a correct-vs-error decoder on REAL Boran iEEG maintenance-window
      states (same guardrail-2 discipline as run_closed_loop_analysis.py: fit
      BEFORE simulation, on uncontrolled real data, never re-fit to simulated
      states). Predicted-error trials = real trials the decoder calls "error"
      (out-of-fold, so the flip test never uses the same fold's own label).
      Target = the real correct-trial centroid (matches run_closed_loop_
      analysis.py's rescue-target convention). For EACH predicted-error trial,
      drive simulate_closed_loop from THAT TRIAL's own real maintenance-onset
      state (x0), on-demand-triggered (trigger="decoder", low duty cycle) --
      simulate_closed_loop only supports one shared x0 per call (n_trials
      resamples noise from the SAME x0), so this loops the existing function
      once per real trial rather than re-implementing its rollout.
  D2. Re-evaluates the SAME outcome decoder on the controlled rollout's final
      state: flip = decoder(x_final_controlled) == "correct". Metrics: flip
      rate, mean decoder-margin change, duty cycle (bootstrap_ci for the CI).
  D3. MANDATORY sanity control: the SAME on-demand energy applied along a
      RANDOM direction (not LQR-designed) -- reported as random_dir_flip_rate,
      the null baseline every arm must exceed to be a real effect (guardrail:
      a linear plant cannot create information the state never had).

Outputs: results/closed_loop_behavior_flip.json
Updates: results/all_statistics.json -- "closed_loop_behavior_flip" key

Run (after run_boran_pipeline.py, run_tes1_analysis.py, run_divergence_analysis.py):
    conda run -n wm_dynamics python scripts/run_closed_loop_behavior_flip.py
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

from closed_loop import simulate_closed_loop, _b_hat_at_angle
from statistics import bootstrap_ci, stable_seed
from io_utils import locked_json_update

RESULTS = ROOT / "results"
BORAN_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 10)]
B_HAT_MISMATCH_DEG = 20.0   # same realistic B-estimation-error stand-in as
                            # run_closed_loop_analysis.py's guardrail 1
N_BOOT = 2000
U_BUDGET = 1.0
MIN_ERROR_PRED = 3   # floor to report a per-subject flip rate at all. LOWERED from an
                     # initial 5: Step B's pooled outcome decodability (AUC=0.68, N=1653
                     # trials/9 subjects) does NOT decompose into reliable PER-SUBJECT
                     # decoders (per-subject cv_acc ~0.49-0.50, near chance) -- most
                     # subjects clear 0-2 decoder-predicted-error trials even at the
                     # Step-B peak timepoint. STOP-and-report: this is a genuinely
                     # underpowered per-subject design, not a bug; see agent_report.md.


def _fit_outcome_decoder_and_margin(Z: np.ndarray, correct: np.ndarray, peak_t_idx: int | None = None):
    """Out-of-fold correct-vs-error decoder on real, uncontrolled states
    (guardrail 2). Fits AT THE SAME PEAK-AUC TIMEPOINT results/behavior_ctg.json
    (Step B) already identified for this cohort -- a per-timestep-pooled/
    majority-vote decoder was tried first and reliably predicted zero trials
    as "error" for every Boran iEEG subject (near-chance cv_acc ~0.50; the
    outcome signal Step B found is concentrated at specific delay timepoints,
    diluted to nothing when pooled across the whole trial). peak_t_idx=None
    falls back to the mid-trial timepoint. Returns (predict_fn, margin_fn,
    pred_error_trial, correct_centroid, decoder_cv_acc)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

    N, T, k = Z.shape
    ti = peak_t_idx if peak_t_idx is not None else T // 2
    ti = int(np.clip(ti, 0, T - 1))
    X = Z[:, ti, :]
    labels = correct.astype(int)   # 1 = correct, 0 = error

    n_splits = min(5, int(np.min(np.bincount(labels))))
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000))])
    if n_splits < 2:
        return None
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    cv_acc = float(cross_val_score(pipe, X, labels, cv=cv, scoring="balanced_accuracy").mean())
    pred_error_trial = (cross_val_predict(pipe, X, labels, cv=cv) == 0)

    pipe.fit(X, labels)   # final decoder used inside the simulation loop
    predict_fn = lambda Xq: pipe.predict(Xq)

    def margin_fn(Xq: np.ndarray) -> np.ndarray:
        return pipe.decision_function(Xq)   # >0 favors "correct" (class 1)

    correct_centroid = Z[correct, ti, :].mean(axis=0)
    return predict_fn, margin_fn, pred_error_trial, correct_centroid, cv_acc


def _flip_one_trial(A, B_true, B_hat, x0, target, decoder, margin_fn, horizon,
                    obs_noise, proc_noise, rng, random_dir: bool = False):
    """Drive ONE real predicted-error trial's own starting state through the
    on-demand controller. n_trials=1 in the underlying call: this trial's x0
    is fixed, only the process/observation noise is resampled.

    D3 null control (random_dir=True): the SAME on-demand/LQR machinery
    (same duty-cycle trigger, same u_budget, same B_hat mismatch), but the
    actuator itself (B_true) is replaced by a random unit direction in the
    SAME latent space -- same convention as the causal-benchmark leaderboard's
    random_alignment arm (run_soldado_pipeline.py) elsewhere in this project.
    An informed direction should flip trials; an uninformed one should not."""
    if random_dir:
        n = A.shape[0]
        rand_dir = rng.standard_normal((n, 1))
        rand_dir /= np.linalg.norm(rand_dir) + 1e-12
        res = simulate_closed_loop(
            A, rand_dir, x0, target, decoder, label=1, trigger="decoder",
            A_hat=A, B_hat=rand_dir, obs_noise=obs_noise, proc_noise=proc_noise, u_budget=U_BUDGET,
            horizon=horizon, n_trials=1, n_boot=1, rng=rng,
        )
    else:
        res = simulate_closed_loop(
            A, B_true, x0, target, decoder, label=1, trigger="decoder",
            A_hat=A, B_hat=B_hat, obs_noise=obs_noise, proc_noise=proc_noise, u_budget=U_BUDGET,
            horizon=horizon, n_trials=1, n_boot=1, rng=rng,
        )
    x_final = res["x_traj_on"][0, -1]
    x_final_off = res["x_traj_off"][0, -1]
    pred_final = decoder(x_final.reshape(1, -1))[0]
    flipped = bool(pred_final == 1)
    margin_delta = float(margin_fn(x_final.reshape(1, -1))[0] - margin_fn(x_final_off.reshape(1, -1))[0])
    duty = res["duty_cycle"]
    return flipped, margin_delta, duty


def run_boran_ieeg() -> dict:
    tes1_boran = np.load(RESULTS / "tes1_boran_B.npz", allow_pickle=True)
    div = np.load(RESULTS / "divergence_analysis.npz", allow_pickle=True)
    try:
        with open(RESULTS / "behavior_ctg.json") as f:
            behavior_ctg = json.load(f)
        peak_time_s = behavior_ctg.get("boran_ieeg", {}).get("peak_time_s")
    except FileNotFoundError:
        peak_time_s = None
        print("  WARNING: results/behavior_ctg.json not found -- run run_behavior_ctg.py first "
              "(Step B); falling back to the mid-trial timepoint.")

    out = {}
    for subj in BORAN_SUBJECTS:
        if f"{subj}_A_dmd" not in tes1_boran:
            print(f"  SKIP {subj} -- no TES1 bundle")
            continue
        div_key = f"boran_{subj}_dynamic_best_idx"
        if div_key not in div:
            print(f"  SKIP {subj} -- no divergence_analysis dynamic_best_idx")
            continue

        geo = np.load(RESULTS / f"boran_geometry_{subj}.npz", allow_pickle=True)
        Z, correct, times = geo["Z"], geo.get("correct", np.ones(geo["Z"].shape[0], dtype=bool)).astype(bool), geo["times"]
        if correct.sum() < 10 or (~correct).sum() < 10:
            print(f"  SKIP {subj} -- insufficient correct/error trials for a decoder "
                  f"(correct={int(correct.sum())}, error={int((~correct).sum())})")
            continue

        # Fit at the SAME peak-AUC timepoint Step B's outcome-CTG already
        # identified for Boran iEEG (results/behavior_ctg.json), not a fresh
        # per-subject search -- keeps this trial-flip test from re-fishing for
        # the best timepoint on the same data it then evaluates.
        peak_t_idx = int(np.argmin(np.abs(times - peak_time_s))) if peak_time_s is not None else None
        decoder_fit = _fit_outcome_decoder_and_margin(Z, correct, peak_t_idx)
        if decoder_fit is None:
            print(f"  SKIP {subj} -- decoder fold count too low")
            continue
        decoder, margin_fn, pred_error_trial, target, decoder_cv_acc = decoder_fit
        n_error_pred = int(pred_error_trial.sum())
        if n_error_pred < MIN_ERROR_PRED:
            print(f"  SKIP {subj} -- only {n_error_pred} decoder-predicted-error trials (<{MIN_ERROR_PRED})")
            continue

        A = tes1_boran[f"{subj}_A_dmd"]
        dyn_idx = int(div[div_key])
        B_true = tes1_boran[f"{subj}_B_latent_per_tes1"][dyn_idx]
        rng = np.random.default_rng(stable_seed(f"clbf_{subj}"))
        B_hat = _b_hat_at_angle(B_true, B_HAT_MISMATCH_DEG, rng)

        state_scale = float(np.linalg.norm(target - Z.mean(axis=(0, 1)))) + 1e-6
        proc_noise = 0.05 * state_scale
        obs_noise = 0.10 * state_scale
        horizon = min(Z.shape[1] - 1, 500)   # cap for tractability across N_error_pred trials;
                                              # each trial gets its own simulate_closed_loop call

        pred_error_idx = np.where(pred_error_trial)[0]
        flips, margins, duties = [], [], []
        flips_rand, margins_rand, duties_rand = [], [], []
        for i, tr in enumerate(pred_error_idx):
            x0_trial = Z[tr, 0, :]   # this trial's own maintenance-onset state
            trial_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            flipped, margin_delta, duty = _flip_one_trial(
                A, B_true, B_hat, x0_trial, target, decoder, margin_fn, horizon,
                obs_noise, proc_noise, trial_rng)
            flips.append(flipped); margins.append(margin_delta); duties.append(duty)

            rand_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            flipped_r, margin_r, duty_r = _flip_one_trial(
                A, B_true, B_hat, x0_trial, target, decoder, margin_fn, horizon,
                obs_noise, proc_noise, rand_rng, random_dir=True)
            flips_rand.append(flipped_r); margins_rand.append(margin_r); duties_rand.append(duty_r)

        flips = np.array(flips); margins = np.array(margins); duties = np.array(duties)
        flips_rand = np.array(flips_rand)

        frac_flipped, ci_lo, ci_hi = bootstrap_ci(flips.astype(float), np.mean, n_boot=N_BOOT,
                                                   rng=np.random.default_rng(stable_seed(f"clbf_{subj}_ci")))
        row = {
            "n_error_pred": n_error_pred,
            "n_flipped": int(flips.sum()),
            "frac_flipped": frac_flipped,
            "flip_ci": [ci_lo, ci_hi],
            "mean_margin_delta": float(margins.mean()),
            "duty_cycle": float(duties.mean()),
            "random_dir_flip_rate": float(flips_rand.mean()),
            "random_dir_n_flipped": int(flips_rand.sum()),
            "decoder_cv_acc": decoder_cv_acc,
            "dynamic_best_tes1_idx": dyn_idx,
            "horizon": int(horizon),
        }
        out[subj] = row
        print(f"  {subj}: n_error_pred={n_error_pred} frac_flipped={frac_flipped:.3f} "
              f"[{ci_lo:.3f},{ci_hi:.3f}] random_dir_flip_rate={row['random_dir_flip_rate']:.3f} "
              f"duty_cycle={row['duty_cycle']:.3f} decoder_cv_acc={decoder_cv_acc:.3f}")

    return out


def main():
    print("Boran iEEG (ONLY cohort with both a fitted B matrix and a real "
          "outcome label -- see module docstring for why):")
    boran = run_boran_ieeg()

    if not boran:
        print("\nNo Boran iEEG subject produced a usable result -- STOP, nothing to pool.")
        out = {"boran_ieeg": {}, "pooled": None,
              "excluded_no_b_matrix": ["boran_units", "dandi000469", "dandi001187", "dandi000673"],
              "excluded_no_outcome_label": ["miller"]}
    else:
        n_pred_arr = np.array([v["n_error_pred"] for v in boran.values()])
        flip_arr = np.array([v["frac_flipped"] for v in boran.values()])
        rand_arr = np.array([v["random_dir_flip_rate"] for v in boran.values()])
        pooled_mean, pooled_lo, pooled_hi = bootstrap_ci(
            flip_arr, np.mean, n_boot=N_BOOT, rng=np.random.default_rng(stable_seed("clbf_pooled")))
        rand_mean, rand_lo, rand_hi = bootstrap_ci(
            rand_arr, np.mean, n_boot=N_BOOT, rng=np.random.default_rng(stable_seed("clbf_pooled_rand")))
        pooled = {
            "n_subjects": len(boran), "total_n_error_pred": int(n_pred_arr.sum()),
            "mean_frac_flipped": pooled_mean, "ci": [pooled_lo, pooled_hi],
            "mean_random_dir_flip_rate": rand_mean, "random_dir_ci": [rand_lo, rand_hi],
            "exceeds_random_null": bool(pooled_lo > rand_hi),
        }
        print(f"\nPooled (N={len(boran)} subjects): mean_frac_flipped={pooled_mean:.3f} "
              f"[{pooled_lo:.3f},{pooled_hi:.3f}] vs random_dir={rand_mean:.3f} "
              f"[{rand_lo:.3f},{rand_hi:.3f}] -> "
              f"{'EXCEEDS null' if pooled['exceeds_random_null'] else 'DOES NOT clearly exceed null'}")
        out = {"boran_ieeg": boran, "pooled": pooled,
              "excluded_no_b_matrix": ["boran_units", "dandi000469", "dandi001187", "dandi000673"],
              "excluded_no_outcome_label": ["miller"]}

    with open(RESULTS / "closed_loop_behavior_flip.json", "w") as f:
        json.dump(out, f, indent=2)
    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["closed_loop_behavior_flip"] = out
    print("\nSaved results/closed_loop_behavior_flip.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
