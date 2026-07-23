"""
closed_loop.py — In-silico closed-loop control of a fitted WM plant (R4/R5).

Two public functions:
  simulate_closed_loop — run loop-ON vs loop-OFF on a fitted plant and score
    the benefit on a held-out decoder.
  robustness_sweep      — stress-test that benefit against plant mismatch,
    observation noise, and unmodeled nonlinearity.

ANTI-CIRCULARITY GUARDRAILS (the point of this module; violating any one
invalidates the result — see comments.txt STEP 3):
  (1) The controller is designed on an ESTIMATED plant (A_hat, B_hat) and
      evaluated on a DIFFERENT "true" plant (A, B) that generates the
      simulated trajectories. Passing A_hat=A, B_hat=B (the defaults) is the
      circular case; callers demonstrating the guardrail must pass a
      perturbed A_hat/B_hat.
  (2) `decoder` must be trained on UNCONTROLLED (loop-off) states with known
      labels BEFORE calling this function — training happens entirely
      outside this module. Decodability is scored on states this module
      simulates, never re-fit to them.
  (3) A realistic (non-collapsed) effect is expected. Decodability pinned
      near 100% (or drift_reduction implausibly large) signals a circularity
      bug, not a success.

Reuses src/control.py (lqr_design -> dare_solve, controllability_gramian via
is_controllable/unstable_eigenvector) for the LQR gain; does not
re-implement Riccati-equation solving.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from control import lqr_design
from statistics import bootstrap_ci

_STATE_NORM_CAP = 1e4   # numerical safety valve; see _rollout


def _drift(X: NDArray, target: NDArray, manifold_basis: NDArray | None) -> NDArray:
    """Per-state-vector drift: distance off the coding manifold if a basis is
    given, else Euclidean distance to the target. X: (..., n)."""
    if manifold_basis is None:
        return np.linalg.norm(X - target, axis=-1)
    coeffs = X @ manifold_basis          # (..., k)
    X_proj = coeffs @ manifold_basis.T   # (..., n)
    return np.linalg.norm(X - X_proj, axis=-1)


def _paired_bootstrap_ci(
    off_vals: NDArray, on_vals: NDArray, n_boot: int, rng: np.random.Generator,
    ci: float = 0.95,
) -> tuple[float, float]:
    """CI on mean(off) - mean(on), resampling trial indices jointly."""
    n = len(off_vals)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = off_vals[idx].mean() - on_vals[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def simulate_closed_loop(
    A: NDArray,
    B: NDArray,
    x0: NDArray,
    target: NDArray,
    decoder: Callable[[NDArray], NDArray] | None,
    label: int | None = None,
    *,
    K: NDArray | None = None,
    A_hat: NDArray | None = None,
    B_hat: NDArray | None = None,
    obs_noise: float = 0.05,
    proc_noise: float = 0.02,
    u_budget: float = 1.0,
    horizon: int = 50,
    q_state: float = 1.0,
    r_control: float = 1.0,
    n_trials: int = 50,
    n_boot: int = 1000,
    manifold_basis: NDArray | None = None,
    nonlinearity_scale: float = 0.0,
    trigger: str | None = None,
    trigger_threshold: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate a fitted plant with the loop closed vs open, from the same x0.

    Dynamics (the "true" plant the trajectories are drawn from):
        x_{t+1} = A x_t + c*||x_t||*x_t + B u_t + w_t,   w_t ~ N(0, proc_noise^2 I)
    (the nonlinear term is 0 unless `nonlinearity_scale` c is set — used by
    robustness_sweep, not by nominal calls).
    Observation fed to the controller only: x_hat_t = x_t + v_t, v_t ~ N(0, obs_noise^2 I).
    Control law: u_t = -K (x_hat_t - target), clipped to ||u_t|| <= u_budget.
    Loop-OFF is identical but u_t = 0 throughout, from the same noise draws.

    `trigger` gates whether u_t is applied at all, at every step of the
    loop-on rollout (loop-off is always u_t=0, regardless of trigger):
      None       — continuous control, u_t applied every step (default;
                   bit-for-bit identical to pre-trigger behaviour: the same
                   x_hat_t/w_t draws happen in the same order either way).
      "decoder"  — engaged only when `decoder(x_hat_t) != label` (guardrail
                   2: gate reads the NOISY observation, never x_t).
      "manifold" — engaged only when x_hat_t's distance off `manifold_basis`
                   exceeds `trigger_threshold`.

    K is designed by LQR on (A_hat, B_hat) — which default to (A, B), i.e.
    the circular case; pass a perturbed A_hat/B_hat to exercise guardrail (1).

    `decoder` (if given) must already be fit on independent loop-off states
    with known labels (guardrail 2); it is called here only to predict, on
    the states this function simulates, never to fit.

    Returns
    -------
    dict: drift_on/off (+ CIs), decodability_on/off (+ CIs, None if no
    decoder), drift_reduction (+ CI), decodability_lift (+ CI, None if no
    decoder), x_traj_on/off, K, rho_open, rho_closed, control_energy,
    duty_cycle.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    target = np.asarray(target, dtype=float)
    n = A.shape[0]

    if K is None:
        A_hat = A if A_hat is None else np.asarray(A_hat, dtype=float)
        B_hat = B if B_hat is None else np.asarray(B_hat, dtype=float)
        try:
            K = lqr_design(A_hat, B_hat, q_state=q_state, r_control=r_control)["K"]
        except (np.linalg.LinAlgError, ValueError):
            # (A_hat, B_hat) is not stabilizable (e.g. B_hat orthogonal to an
            # unstable mode) — no stabilizing gain exists, so the best
            # available control is none. On real (as opposed to synthetic)
            # fitted plants this can also surface as scipy's ValueError from
            # solve_discrete_are's Schur reordering ("too far from
            # generalized Schur form") when the DARE pencil is severely
            # ill-conditioned, rather than a LinAlgError — same underlying
            # failure mode (no reliable stabilizing gain), so it gets the
            # same fallback; this is the negative-control case, not a bug,
            # and should degrade to ~zero benefit, not crash.
            K = np.zeros((B.shape[1], n))

    rho_open = float(np.max(np.abs(np.linalg.eigvals(A))))
    rho_closed = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))

    if trigger not in (None, "decoder", "manifold"):
        raise ValueError(f"trigger must be None, 'decoder', or 'manifold', got {trigger!r}")

    def _engaged(x_hat: NDArray) -> bool:
        if trigger is None:
            return True
        if trigger == "decoder":
            pred = decoder(x_hat.reshape(1, -1))[0]
            return bool(pred != label)
        off_dist = _drift(x_hat, target, manifold_basis)
        return bool(off_dist > trigger_threshold)

    def _rollout(control_on: bool, trial_rng: np.random.Generator) -> tuple[NDArray, float, int]:
        x = np.empty((horizon + 1, n))
        x[0] = x0
        energy = 0.0
        engaged_steps = 0
        for t in range(horizon):
            xt = x[t]
            if control_on:
                x_hat = xt + trial_rng.normal(0.0, obs_noise, size=n)
                if _engaged(x_hat):
                    u = -K @ (x_hat - target)
                    u_norm = np.linalg.norm(u)
                    if u_norm > u_budget:
                        u = u * (u_budget / u_norm)
                    engaged_steps += 1
                    energy += float(np.linalg.norm(u))
                else:
                    u = np.zeros(B.shape[1])
            else:
                u = np.zeros(B.shape[1])
            w = trial_rng.normal(0.0, proc_noise, size=n)
            nonlin = nonlinearity_scale * np.linalg.norm(xt) * xt
            x_next = A @ xt + nonlin + B @ u + w
            # A genuinely unstable/nonlinear regime (extreme B-mismatch or
            # nonlinearity_scale) can diverge in a handful of steps; cap the
            # state norm rather than let it overflow to inf/nan, so drift and
            # the robustness sweep's retained_benefit stay finite (a very
            # large but finite drift is itself the "control failed" signal).
            x_next_norm = np.linalg.norm(x_next)
            if x_next_norm > _STATE_NORM_CAP:
                x_next = x_next * (_STATE_NORM_CAP / x_next_norm)
            x[t + 1] = x_next
        return x, energy, engaged_steps

    x_traj_on = np.empty((n_trials, horizon + 1, n))
    x_traj_off = np.empty((n_trials, horizon + 1, n))
    energy_per_trial = np.empty(n_trials)
    engaged_per_trial = np.empty(n_trials)
    for i in range(n_trials):
        seed = int(rng.integers(0, 2**31 - 1))
        x_traj_on[i], energy_per_trial[i], engaged_per_trial[i] = _rollout(True, np.random.default_rng(seed))
        x_traj_off[i], _, _ = _rollout(False, np.random.default_rng(seed))

    drift_on_per_trial = _drift(x_traj_on, target, manifold_basis).mean(axis=1)
    drift_off_per_trial = _drift(x_traj_off, target, manifold_basis).mean(axis=1)

    drift_on, drift_on_lo, drift_on_hi = bootstrap_ci(
        drift_on_per_trial, np.mean, n_boot=n_boot, rng=rng)
    drift_off, drift_off_lo, drift_off_hi = bootstrap_ci(
        drift_off_per_trial, np.mean, n_boot=n_boot, rng=rng)
    drift_reduction = float(drift_off - drift_on)
    dr_lo, dr_hi = _paired_bootstrap_ci(drift_off_per_trial, drift_on_per_trial, n_boot, rng)

    out = {
        "drift_on": drift_on, "drift_on_ci": (drift_on_lo, drift_on_hi),
        "drift_off": drift_off, "drift_off_ci": (drift_off_lo, drift_off_hi),
        "drift_reduction": drift_reduction, "drift_reduction_ci": (dr_lo, dr_hi),
        "decodability_on": None, "decodability_on_ci": None,
        "decodability_off": None, "decodability_off_ci": None,
        "decodability_lift": None, "decodability_lift_ci": None,
        "x_traj_on": x_traj_on, "x_traj_off": x_traj_off,
        "K": K, "n_trials": n_trials,
        "rho_open": rho_open, "rho_closed": rho_closed,
        "control_energy": float(energy_per_trial.mean()),
        "duty_cycle": float(engaged_per_trial.mean() / horizon),
    }

    if decoder is not None and label is not None:
        pred_on = decoder(x_traj_on.reshape(-1, n)).reshape(n_trials, horizon + 1)
        pred_off = decoder(x_traj_off.reshape(-1, n)).reshape(n_trials, horizon + 1)
        acc_on_per_trial = (pred_on == label).mean(axis=1)
        acc_off_per_trial = (pred_off == label).mean(axis=1)

        dec_on, dec_on_lo, dec_on_hi = bootstrap_ci(acc_on_per_trial, np.mean, n_boot=n_boot, rng=rng)
        dec_off, dec_off_lo, dec_off_hi = bootstrap_ci(acc_off_per_trial, np.mean, n_boot=n_boot, rng=rng)
        lift = float(dec_on - dec_off)
        lift_lo, lift_hi = _paired_bootstrap_ci(acc_on_per_trial, acc_off_per_trial, n_boot, rng)

        out.update({
            "decodability_on": dec_on, "decodability_on_ci": (dec_on_lo, dec_on_hi),
            "decodability_off": dec_off, "decodability_off_ci": (dec_off_lo, dec_off_hi),
            "decodability_lift": lift, "decodability_lift_ci": (lift_lo, lift_hi),
        })

    return out


def _b_hat_at_angle(B: NDArray, angle_deg: float, rng: np.random.Generator) -> NDArray:
    """Construct B_hat whose j-th column is B[:,j] rotated `angle_deg` toward
    a random direction orthogonal to it, preserving column norm."""
    theta = np.deg2rad(angle_deg)
    n, m = B.shape
    B_hat = np.zeros_like(B)
    for j in range(m):
        b = B[:, j]
        bn_norm = np.linalg.norm(b) + 1e-12
        bn = b / bn_norm
        r = rng.standard_normal(n)
        r = r - (r @ bn) * bn
        r_norm = np.linalg.norm(r)
        r = r / r_norm if r_norm > 1e-12 else rng.standard_normal(n)
        b_rot = np.cos(theta) * bn + np.sin(theta) * r
        B_hat[:, j] = b_rot * bn_norm
    return B_hat


def robustness_sweep(
    A: NDArray,
    B: NDArray,
    x0: NDArray,
    target: NDArray,
    decoder: Callable[[NDArray], NDArray] | None,
    label: int | None = None,
    *,
    mismatch_angles_deg: NDArray | None = None,
    noise_levels: NDArray | None = None,
    nonlinearity_scales: NDArray | None = None,
    obs_noise: float = 0.05,
    proc_noise: float = 0.02,
    u_budget: float = 1.0,
    horizon: int = 50,
    q_state: float = 1.0,
    r_control: float = 1.0,
    n_trials: int = 30,
    n_boot: int = 500,
    failure_threshold: float = 0.1,
    rng: np.random.Generator | None = None,
) -> dict:
    """Stress-test simulate_closed_loop's drift-reduction benefit.

    Sweeps, one dimension at a time (holding the others at their nominal,
    well-specified value): (i) the angle between B_hat (used to design K)
    and the true B, (ii) observation-noise level, (iii) an unmodeled
    quadratic nonlinearity scale added only to the true (simulated) plant.

    retained_benefit(condition) = drift_reduction(condition) / drift_reduction(nominal)

    Returns the three curves plus the failure boundary — the first swept
    value at which retained_benefit drops below `failure_threshold` (None if
    it never does within the sweep).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if mismatch_angles_deg is None:
        mismatch_angles_deg = np.linspace(0.0, 90.0, 10)
    if noise_levels is None:
        noise_levels = np.linspace(obs_noise, obs_noise * 10, 10)
    if nonlinearity_scales is None:
        nonlinearity_scales = np.linspace(0.0, 1.0, 10)

    def _run(**overrides) -> dict:
        kwargs = dict(A_hat=A, B_hat=B, obs_noise=obs_noise, proc_noise=proc_noise,
                      u_budget=u_budget, horizon=horizon, q_state=q_state, r_control=r_control,
                      n_trials=n_trials, n_boot=n_boot,
                      rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))))
        kwargs.update(overrides)
        return simulate_closed_loop(A, B, x0, target, decoder, label, **kwargs)

    nominal_dr = _run()["drift_reduction"]

    def _retained(dr: float) -> float:
        return float(dr / nominal_dr) if abs(nominal_dr) > 1e-12 else float("nan")

    angle_sweep = []
    for ang in mismatch_angles_deg:
        B_hat = _b_hat_at_angle(B, float(ang), rng)
        dr = _run(B_hat=B_hat)["drift_reduction"]
        angle_sweep.append({"angle_deg": float(ang), "drift_reduction": dr,
                            "retained_benefit": _retained(dr)})

    noise_sweep = []
    for nl in noise_levels:
        dr = _run(obs_noise=float(nl))["drift_reduction"]
        noise_sweep.append({"obs_noise": float(nl), "drift_reduction": dr,
                            "retained_benefit": _retained(dr)})

    nonlinearity_sweep = []
    for c in nonlinearity_scales:
        dr = _run(nonlinearity_scale=float(c))["drift_reduction"]
        nonlinearity_sweep.append({"nonlinearity_scale": float(c), "drift_reduction": dr,
                                   "retained_benefit": _retained(dr)})

    def _boundary(sweep: list[dict], key: str) -> float | None:
        for row in sweep:
            if row["retained_benefit"] < failure_threshold:
                return row[key]
        return None

    return {
        "nominal_drift_reduction": nominal_dr,
        "angle_sweep": angle_sweep,
        "noise_sweep": noise_sweep,
        "nonlinearity_sweep": nonlinearity_sweep,
        "failure_boundary_angle_deg": _boundary(angle_sweep, "angle_deg"),
        "failure_boundary_obs_noise": _boundary(noise_sweep, "obs_noise"),
        "failure_boundary_nonlinearity_scale": _boundary(nonlinearity_sweep, "nonlinearity_scale"),
        "failure_threshold": failure_threshold,
    }
