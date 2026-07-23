#!/usr/bin/env python3
"""Round-8 Part 5: RL policy-alignment arm -- a CONVERGENCE check, not a horse
race (see comments.txt SCIENTIFIC FRAMING). On a linear plant (A from DMD, B
from TES1) with quadratic cost, LQR is the closed-form optimum; a learned
policy can at best recover it. This script does NOT claim RL beats LQR. It
asks a narrower, useful question: does a model-free policy-gradient
controller that never sees A's eigendecomposition (unlike every other arm,
which is built from an explicit eigenvector/Gramian computation)
INDEPENDENTLY discover the same steering direction v* that those closed-form
arms use? Convergence validates v* as the analytical target from an
independent method and directly answers "why not RL?" -- divergence would
question the plant/cost, and is reported as such, not papered over.

True feedback RL (closed-loop stim -> observed neural response -> reward) is
NOT possible here: DANDI/Boran are recording-only, no delivered stimulation
with an observed response exists in this data. What IS possible, and what
this script does: train the policy against the FITTED linear plant
(x_{t+1} = A x_t + b u_t) exactly as an RL agent would train against a
simulator, then evaluate the converged direction against the same
closed-loop drift/flip-rate benchmark (src/closed_loop.simulate_closed_loop,
src/causal.dml_partial_linear) every other arm is scored on.

Policy: a Gaussian evolutionary/score-function policy-gradient (REINFORCE)
directly over the steering DIRECTION b (unit vector in the k-dim latent
space) -- a bandit-style policy-gradient method (pure numpy, no RL
framework; each "episode" is one direction sampled from a Gaussian around
the current mean, scored by rolling the plant forward under a SIMPLE
proportional feedback law along that direction, reward-weighted toward
directions with higher return -- the standard REINFORCE score-function
update). The plant rollout used for TRAINING (below) does NOT use A's
eigendecomposition or an LQR gain -- only A itself, exactly as an RL agent
interacting with a simulator would.

Once trained, b is mapped to its nearest real TES1 donor (cosine similarity)
for the ACTUAL drift-reduction/flip-rate benchmark rollout (the same
physical-realism constraint every other arm respects: only TES1 donors are
deliverable stimulation directions), then scored via the SAME
dml_partial_linear gate-slope machinery as the other 9 arms.

Outputs: results/rl_policy_arm.json (per-subject b_learned, alignment to v*,
reward curve, chosen donor, drift/flip)
Updates: results/targeting_benchmark_boran.json ("rl_policy_alignment" arm
         added to per_subject/leaderboard), results/causal_benchmark.json
         (leaderboard gains a 10th arm; n_arms updated; winner re-checked).

Self-check: tests/test_rl_policy_arm.py (on a synthetic unstable 2D plant,
the learned direction aligns cos>0.8 with the known dominant eigenvector).

Run (after run_targeting_benchmark.py, which this depends on for TES1
bundles + the outcome decoder / context decoder construction):
    conda run -n wm_dynamics python scripts/run_rl_policy_arm.py
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
sys.path.insert(0, str(ROOT / "scripts"))

from closed_loop import simulate_closed_loop, _b_hat_at_angle
from statistics import stable_seed
from io_utils import locked_json_update
from run_targeting_benchmark import (
    RESULTS, BORAN_SUBJECTS, B_HAT_MISMATCH_DEG,
    _stability_horizon, _pool_arm, _near_tie_candidates,
)
from run_closed_loop_behavior_flip import _fit_outcome_decoder_and_margin, _flip_one_trial

ARM_NAME = "rl_policy_alignment"


# ── Policy-gradient direction search (pure numpy REINFORCE) ────────────────

def _rollout_reward(A: np.ndarray, b: np.ndarray, kappa: float, horizon: int,
                    amp_penalty: float, x0: np.ndarray, proc_noise: float,
                    u_budget: float, rng: np.random.Generator) -> float:
    """One closed-loop rollout under a SIMPLE proportional feedback law along
    direction b (u_t = -kappa * (x_t . b), clipped to u_budget) -- no A
    eigendecomposition, no LQR gain. Reward = -(mean state energy) -
    amp_penalty * (mean control energy), i.e. distance-to-correct-region
    (here: the origin/stable point every arm's rollout is ultimately driving
    toward) plus a stimulation-amplitude penalty, per comments.txt 5A."""
    x = x0.copy()
    state_cost = 0.0
    ctrl_cost = 0.0
    for _ in range(horizon):
        u = float(np.clip(-kappa * (x @ b), -u_budget, u_budget))
        x = A @ x + b * u + rng.normal(0, proc_noise, size=b.shape[0])
        state_cost += float(x @ x)
        ctrl_cost += u ** 2
    return -(state_cost / horizon) - amp_penalty * (ctrl_cost / horizon)


def train_policy_direction(
    A: np.ndarray, n_episodes: int = 400, n_rollouts_per_ep: int = 4,
    sigma: float = 0.35, lr: float = 0.2, kappa: float = 0.5, horizon: int = 25,
    amp_penalty: float = 0.02, proc_noise: float = 0.05, u_budget: float = 2.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """REINFORCE on a Gaussian policy over the steering direction: each
    episode samples theta ~ N(mu, sigma^2 I), scores b=theta/||theta|| by
    average reward over several rollouts, and moves mu via the score-function
    gradient (reward-baseline) * (theta-mu), renormalizing mu to stay a unit
    direction (the plant response to a fixed direction is otherwise scale-free
    -- only the direction, not theta's magnitude, is meaningful here)."""
    if rng is None:
        rng = np.random.default_rng(0)
    k = A.shape[0]
    mu = rng.standard_normal(k)
    mu /= np.linalg.norm(mu) + 1e-12
    baseline = 0.0
    reward_history = []
    for ep in range(n_episodes):
        theta = mu + sigma * rng.standard_normal(k)
        b = theta / (np.linalg.norm(theta) + 1e-12)
        rewards = [
            _rollout_reward(A, b, kappa, horizon, amp_penalty,
                            rng.standard_normal(k) * 0.5, proc_noise, u_budget, rng)
            for _ in range(n_rollouts_per_ep)
        ]
        reward = float(np.mean(rewards))
        reward_history.append(reward)
        baseline = 0.9 * baseline + 0.1 * reward
        grad = (reward - baseline) * (theta - mu)
        mu = mu + lr * grad
        mu /= np.linalg.norm(mu) + 1e-12
    b_final = mu / (np.linalg.norm(mu) + 1e-12)
    return {"b": b_final, "reward_history": reward_history,
           "reward_final_mean": float(np.mean(reward_history[-20:]))}


def dominant_eigvec(A: np.ndarray) -> np.ndarray:
    """Same construction as run_targeting_benchmark._build_arm_directions's
    v_star: top eigenvector of A by real part of eigenvalue."""
    eigs, vecs = np.linalg.eig(A)
    order = np.argsort(eigs.real)[::-1]
    v_star = vecs[:, order[0]].real
    return v_star / (np.linalg.norm(v_star) + 1e-12)


# ── Boran per-subject integration (mirrors run_boran_targeting_benchmark) ──

def run_rl_arm_on_boran() -> dict:
    tes1_boran = np.load(RESULTS / "tes1_boran_B.npz", allow_pickle=True)
    try:
        with open(RESULTS / "behavior_ctg.json") as f:
            peak_time_s = json.load(f).get("boran_ieeg", {}).get("peak_time_s")
    except FileNotFoundError:
        peak_time_s = None

    per_subject = {}
    for subj in BORAN_SUBJECTS:
        if f"{subj}_A_dmd" not in tes1_boran:
            print(f"  SKIP {subj} -- no TES1 bundle")
            continue
        geo_path = RESULTS / f"boran_geometry_{subj}.npz"
        if not geo_path.exists():
            continue
        geo = np.load(geo_path, allow_pickle=True)
        Z = geo["Z"]
        correct = geo.get("correct", np.ones(Z.shape[0], dtype=bool)).astype(bool)
        if correct.sum() < 10 or (~correct).sum() < 10:
            print(f"  SKIP {subj} -- insufficient trials for a decoder")
            continue

        A = tes1_boran[f"{subj}_A_dmd"]
        B_bank = tes1_boran[f"{subj}_B_latent_per_tes1"]
        x0, xf = tes1_boran[f"{subj}_x0"], tes1_boran[f"{subj}_xf"]

        rng = np.random.default_rng(stable_seed(f"rlarm_{subj}"))
        train_res = train_policy_direction(A, rng=np.random.default_rng(stable_seed(f"rlarm_train_{subj}")))
        b_learned = train_res["b"]
        v_star = dominant_eigvec(A)
        align_to_vstar = float(np.abs(b_learned @ v_star))

        B_units = B_bank[:, :, 0] / (np.linalg.norm(B_bank[:, :, 0], axis=1, keepdims=True) + 1e-12)
        donor_sim = np.abs(B_units @ b_learned)

        peak_t_idx = (int(np.argmin(np.abs(geo["times"] - peak_time_s)))
                     if peak_time_s is not None else None)
        decoder_fit = _fit_outcome_decoder_and_margin(Z, correct, peak_t_idx)
        decoder, margin_fn, pred_error_trial, target, decoder_cv_acc = (
            decoder_fit if decoder_fit else (None,) * 5)
        n_error_pred = int(pred_error_trial.sum()) if decoder_fit else 0

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        set_sizes = geo["set_sizes"]
        mask_ctx = (set_sizes == 4) | (set_sizes == 8)
        Z_ctx = Z[mask_ctx]
        labels_ctx = np.repeat((set_sizes[mask_ctx] == 8).astype(int), Z_ctx.shape[1])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(Z_ctx.reshape(-1, Z_ctx.shape[-1]), labels_ctx)
        context_decoder = lambda X: pipe.predict(X)

        state_scale = float(np.linalg.norm(xf - x0)) + 1e-6
        proc_noise, obs_noise = 0.05 * state_scale, 0.10 * state_scale
        horizon_real = min(Z.shape[1] - 1, 300)
        stab_horizon = _stability_horizon(A, n_time_constants=3.0)
        horizon = int(min(horizon_real, np.ceil(stab_horizon))) if np.isfinite(stab_horizon) else horizon_real

        def _rollout(cand_idx: int) -> dict:
            B_true = B_bank[cand_idx]
            B_hat = _b_hat_at_angle(B_true, B_HAT_MISMATCH_DEG, rng)
            res = simulate_closed_loop(
                A, B_true, x0, xf, context_decoder, label=1, trigger="decoder",
                A_hat=A, B_hat=B_hat, obs_noise=obs_noise, proc_noise=proc_noise, u_budget=1.0,
                horizon=horizon, n_trials=30, n_boot=500,
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )
            flip_frac = float("nan")
            if decoder_fit is not None and n_error_pred >= 3:
                flips = []
                for tr in np.where(pred_error_trial)[0]:
                    trial_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
                    flipped, _, _ = _flip_one_trial(A, B_true, B_hat, Z[tr, 0, :], target,
                                                    decoder, margin_fn, horizon, obs_noise,
                                                    proc_noise, trial_rng)
                    flips.append(flipped)
                flip_frac = float(np.mean(flips))
            return {"donor_idx": cand_idx, "drift_reduction": res["drift_reduction"],
                   "flip_frac": flip_frac, "destabilized": bool(res["rho_closed"] > res["rho_open"]),
                   "rho_open": res["rho_open"], "rho_closed": res["rho_closed"]}

        # Part 8B, applied uniformly (comments.txt): same near-tie non-destabilizing
        # donor preference as every other arm -- RL's own physically-realizable donor
        # is picked by cosine-to-b_learned, not exempted from the rescue rule.
        candidates = _near_tie_candidates(donor_sim)
        tried = [_rollout(c) for c in candidates]
        chosen = next((t for t in tried if not t["destabilized"]), tried[0])
        donor_selection = ("argmax" if chosen is tried[0] and not tried[0]["destabilized"] else
                           "near_tie_rescue" if not chosen["destabilized"] else "argmax_no_rescue")
        donor_idx = chosen["donor_idx"]

        per_subject[subj] = {
            "arms": {ARM_NAME: {
                "donor_idx": donor_idx, "align_to_vstar": align_to_vstar,
                "drift_reduction": chosen["drift_reduction"], "flip_frac": chosen["flip_frac"],
                "destabilized": chosen["destabilized"], "rho_open": chosen["rho_open"],
                "rho_closed": chosen["rho_closed"], "donor_selection": donor_selection,
                "n_candidates_tried": len(tried),
                "b_learned": b_learned.tolist(), "donor_cos_to_learned": float(donor_sim[donor_idx]),
                "reward_final_mean": train_res["reward_final_mean"],
            }}
        }
        print(f"  {subj}: align_to_vstar={align_to_vstar:.3f} donor_cos={donor_sim[donor_idx]:.3f} "
              f"drift_reduction={chosen['drift_reduction']:+.3f} destabilized={chosen['destabilized']} "
              f"({donor_selection}, {len(tried)} tried)")

    return per_subject


def main():
    print(f"Training RL policy-direction arm ({ARM_NAME}) per Boran subject...")
    per_subject = run_rl_arm_on_boran()
    if not per_subject:
        print("No usable Boran subject -- STOP, cannot build the RL arm.")
        return

    dr = _pool_arm(per_subject, ARM_NAME, "drift_reduction")
    flip = _pool_arm(per_subject, ARM_NAME, "flip_frac")
    stable_rows = [v["arms"][ARM_NAME] for v in per_subject.values() if not v["arms"][ARM_NAME]["destabilized"]]
    n_destabilized = sum(1 for v in per_subject.values() if v["arms"][ARM_NAME]["destabilized"])
    mean_drift = float(np.nanmean([r["drift_reduction"] for r in stable_rows])) if stable_rows else float("nan")
    flip_vals = [r["flip_frac"] for r in stable_rows if np.isfinite(r["flip_frac"])]
    mean_flip = float(np.nanmean(flip_vals)) if flip_vals else float("nan")
    mean_align_to_vstar = float(np.mean([v["arms"][ARM_NAME]["align_to_vstar"] for v in per_subject.values()]))

    leaderboard_entry = {
        "mean_drift_reduction": mean_drift, "mean_flip_rate": mean_flip,
        "drift_reduction_dml": dr, "flip_rate_dml": flip,
        "n_subjects": len(stable_rows), "n_destabilized_excluded": n_destabilized,
        "mean_align_to_vstar": mean_align_to_vstar,
    }
    print(f"\n{ARM_NAME}: mean_drift_reduction={mean_drift:+.3f} mean_flip_rate={mean_flip:.3f} "
          f"mean_align_to_vstar={mean_align_to_vstar:.3f} (n={len(stable_rows)}, "
          f"{n_destabilized} destabilized-excluded)")
    convergence_verdict = (
        "CONVERGES to v*" if mean_align_to_vstar > 0.7 else
        "PARTIALLY aligns with v*" if mean_align_to_vstar > 0.4 else
        "does NOT converge to v* -- report honestly, this would question the plant/cost"
    )
    print(f"Convergence check (prediction: rl_policy_alignment ~ vstar_alignment): {convergence_verdict}")

    with open(RESULTS / "rl_policy_arm.json", "w") as f:
        json.dump({"per_subject": per_subject, "leaderboard": leaderboard_entry,
                   "convergence_verdict": convergence_verdict}, f, indent=2)
    print("Saved results/rl_policy_arm.json")

    # Extend targeting_benchmark_boran.json with the new arm (ADD, don't shrink).
    with open(RESULTS / "targeting_benchmark_boran.json") as f:
        tb = json.load(f)
    for subj, row in per_subject.items():
        if subj in tb["per_subject"]:
            tb["per_subject"][subj]["arms"][ARM_NAME] = row["arms"][ARM_NAME]
    tb["leaderboard"][ARM_NAME] = leaderboard_entry
    with open(RESULTS / "targeting_benchmark_boran.json", "w") as f:
        json.dump(tb, f, indent=2)

    # Extend causal_benchmark.json leaderboard with a 10th arm.
    slope = dr["theta"] if dr is not None else float("nan")
    slope_lo = dr["ci_lo"] if dr is not None else float("nan")
    slope_hi = dr["ci_hi"] if dr is not None else float("nan")
    p_val = dr["p_value"] if dr is not None else float("nan")
    n_val = dr["n"] if dr is not None else 0
    with locked_json_update(RESULTS / "causal_benchmark.json") as bench:
        bench["leaderboard"][ARM_NAME] = {
            "slope": slope, "slope_ci_lo": slope_lo, "slope_ci_hi": slope_hi,
            "p_value": p_val, "n": n_val, "flip_rate": mean_flip,
            "flip_rate_dml": flip, "scored_on": "boran_ieeg_targeting_benchmark",
            "mean_align_to_vstar": mean_align_to_vstar, "convergence_verdict": convergence_verdict,
        }
        bench["n_arms"] = len(bench["leaderboard"])
        scored = {k: v for k, v in bench["leaderboard"].items() if np.isfinite(v.get("slope", np.nan))}
        if scored:
            bench["winner"] = max(scored, key=lambda k: scored[k]["slope"])
        n_arms_final = bench["n_arms"]
    print(f"\nSaved results/rl_policy_arm.json; extended targeting_benchmark_boran.json "
          f"and causal_benchmark.json ({ARM_NAME} added, now {n_arms_final} arms total).")


if __name__ == "__main__":
    main()
