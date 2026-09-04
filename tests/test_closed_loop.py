"""Tests for src/closed_loop.py — synthetic, deterministic, known-answer.

Each test targets one anti-circularity guardrail from the module docstring:
  (1) controller designed on an estimated plant, evaluated on a different
      true plant — exercised by the mismatch-monotonicity and negative-
      control tests.
  (2) benefit scored on a held-out decoder trained on independent loop-off
      data — exercised by TestGuardrail2HeldOutDecoder.
  (3) a realistic (non-collapsed) effect — asserted alongside (2).
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from closed_loop import simulate_closed_loop, robustness_sweep, _b_hat_at_angle
from control import lqr_design


def _unstable_system():
    """2D LDS: x0 unstable (|1.08| > 1), x1 stable (0.6); B aligned to the
    unstable eigenvector (e0) — control has real authority over the failure mode."""
    A = np.array([[1.08, 0.0], [0.0, 0.6]])
    B = np.array([[1.0], [0.0]])
    x0 = np.array([0.6, 0.0])
    target = np.array([0.0, 0.0])
    return A, B, x0, target


def _calibrated_threshold_decoder(A, B, x0, target, rng):
    """A decoder 'trained' on independent loop-off (uncontrolled) trials: the
    decision threshold is the median distance-to-target over an independent
    calibration batch of loop-off rollouts, pooled across the whole horizon
    (not the trials/seeds simulate_closed_loop will later score)."""
    calib = simulate_closed_loop(A, B, x0, target, None, n_trials=150, horizon=30, rng=rng)
    off_dists = np.linalg.norm(calib["x_traj_off"] - target, axis=-1)
    threshold = float(np.median(off_dists))

    def decoder(X):
        d = np.linalg.norm(X - target, axis=-1)
        return (d < threshold).astype(int)

    return decoder, threshold


class TestSimulateClosedLoopBenefit:
    def test_closed_loop_reduces_drift_vs_open_loop(self, rng):
        A, B, x0, target = _unstable_system()
        res = simulate_closed_loop(A, B, x0, target, None, n_trials=60, horizon=30, rng=rng)
        assert res["drift_on"] < res["drift_off"]
        assert res["drift_reduction"] > 0
        lo, hi = res["drift_reduction_ci"]
        assert lo > 0   # benefit CI excludes zero

    def test_default_K_matches_manual_lqr_design(self, rng):
        # A_hat/B_hat default to A/B (the circular case) unless overridden —
        # confirm the gain used is exactly what control.lqr_design returns.
        A, B, x0, target = _unstable_system()
        res = simulate_closed_loop(A, B, x0, target, None, n_trials=5, horizon=5, rng=rng)
        expected_K = lqr_design(A, B, q_state=1.0, r_control=1.0)["K"]
        np.testing.assert_allclose(res["K"], expected_K)


class TestGuardrail2HeldOutDecoder:
    def test_decodability_lift_is_real_and_not_collapsed(self, rng):
        A, B, x0, target = _unstable_system()
        decoder, threshold = _calibrated_threshold_decoder(A, B, x0, target, rng)

        res = simulate_closed_loop(A, B, x0, target, decoder, label=1,
                                   n_trials=60, horizon=30, rng=np.random.default_rng(1))

        # Guardrail 2: benefit survives scoring on a decoder trained on
        # independent loop-off data.
        assert res["decodability_on"] > res["decodability_off"]
        lo, hi = res["decodability_lift_ci"]
        assert lo > 0

        # Guardrail 3: decodability is a realistic effect, not the
        # circularity signature of ~100% on both arms (the threshold was
        # calibrated to bisect loop-off data, so loop-off decodability
        # should sit near chance, not near ceiling).
        assert res["decodability_off"] < 0.85
        assert res["decodability_on"] <= 1.0


class TestNegativeControlOrthogonalB:
    def test_orthogonal_b_gives_no_benefit(self, rng):
        # B_orth has zero authority over the unstable mode (e0) — control
        # cannot help, so drift_reduction should be ~0, not a false positive.
        A = np.array([[1.08, 0.0], [0.0, 0.6]])
        B_orth = np.array([[0.0], [1.0]])
        x0 = np.array([0.6, 0.0])
        target = np.array([0.0, 0.0])

        res = simulate_closed_loop(A, B_orth, x0, target, None,
                                   n_trials=60, horizon=30, rng=rng)
        assert abs(res["drift_reduction"]) < 0.1 * res["drift_off"]
        lo, hi = res["drift_reduction_ci"]
        assert lo < 0 < hi   # CI should include zero — no reliable benefit


class TestMismatchMonotonicity:
    def test_benefit_decreases_monotonically_with_mismatch_angle(self, rng):
        # Guardrail 1: a controller designed from a B_hat that is
        # increasingly misaligned with the true B should help less and
        # less. K is a simple proportional gain along B_hat (rather than
        # the module's default full LQR design) purely so the degradation
        # is numerically well-conditioned across the full 0-90 deg sweep —
        # DARE's gain can blow up near-singularly as B_hat's controllable
        # component vanishes, which would make a strict monotonicity check
        # numerically fragile rather than false.
        A, B, x0, target = _unstable_system()
        gain = 0.8
        angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]

        drift_reductions = []
        for ang in angles:
            B_hat = _b_hat_at_angle(B, ang, rng) if ang > 0 else B
            K = gain * B_hat.T
            res = simulate_closed_loop(A, B, x0, target, None, K=K,
                                       n_trials=40, horizon=30, n_boot=200,
                                       rng=np.random.default_rng(1))
            drift_reductions.append(res["drift_reduction"])

        # Monotonically non-increasing (allow tiny numerical slack).
        diffs = np.diff(drift_reductions)
        assert np.all(diffs <= 1e-6)
        assert drift_reductions[0] > drift_reductions[-1]


class TestRhoOpenClosed:
    def test_aligned_b_stabilizes_below_open_loop(self, rng):
        A, B, x0, target = _unstable_system()
        res = simulate_closed_loop(A, B, x0, target, None, n_trials=5, horizon=5, rng=rng)
        assert res["rho_open"] == pytest.approx(1.08, abs=1e-9)
        assert res["rho_closed"] < res["rho_open"]

    def test_orthogonal_b_does_not_stabilize(self, rng):
        A = np.array([[1.08, 0.0], [0.0, 0.6]])
        B_orth = np.array([[0.0], [1.0]])
        x0 = np.array([0.6, 0.0])
        target = np.array([0.0, 0.0])
        res = simulate_closed_loop(A, B_orth, x0, target, None, n_trials=5, horizon=5, rng=rng)
        assert res["rho_closed"] >= res["rho_open"]


def _early_warning_decoder(A, B, x0, target, rng, percentile=20):
    """Like `_calibrated_threshold_decoder` but calibrated to fire earlier
    (a low percentile of independent loop-off distances, not the median) —
    the point of an on-demand trigger is to catch drift before it
    accumulates into a costly correction, not to bisect the distribution."""
    calib = simulate_closed_loop(A, B, x0, target, None, n_trials=150, horizon=30, rng=rng)
    off_dists = np.linalg.norm(calib["x_traj_off"] - target, axis=-1)
    threshold = float(np.percentile(off_dists, percentile))

    def decoder(X):
        d = np.linalg.norm(X - target, axis=-1)
        return (d < threshold).astype(int)

    return decoder, threshold


class TestOnDemandTrigger:
    def test_decoder_trigger_reduces_drift_with_lower_energy_than_continuous(self, rng):
        A, B, x0, target = _unstable_system()
        decoder, threshold = _early_warning_decoder(A, B, x0, target, rng)

        res_continuous = simulate_closed_loop(A, B, x0, target, decoder, label=1,
                                              n_trials=60, horizon=30, rng=np.random.default_rng(2))
        res_ondemand = simulate_closed_loop(A, B, x0, target, decoder, label=1, trigger="decoder",
                                            n_trials=60, horizon=30, rng=np.random.default_rng(2))

        assert res_ondemand["drift_reduction"] > 0
        assert res_ondemand["control_energy"] < res_continuous["control_energy"]
        assert res_ondemand["duty_cycle"] < 1.0
        assert res_continuous["duty_cycle"] == pytest.approx(1.0)

    def test_unreachable_decoder_trigger_never_engages(self, rng):
        A, B, x0, target = _unstable_system()

        def never_trigger(X):
            return np.full(X.shape[0], 1, dtype=int)   # always predicts `label` -> never engaged

        res = simulate_closed_loop(A, B, x0, target, never_trigger, label=1, trigger="decoder",
                                   n_trials=30, horizon=20, rng=rng)
        assert res["duty_cycle"] == 0.0
        assert res["control_energy"] == 0.0
        assert abs(res["drift_reduction"]) < 0.1 * res["drift_off"]

    def test_manifold_trigger_gates_on_threshold(self, rng):
        A, B, x0, target = _unstable_system()
        manifold_basis = np.array([[1.0], [0.0]])
        res_loose = simulate_closed_loop(A, B, x0, target, None, trigger="manifold",
                                         trigger_threshold=0.0, manifold_basis=manifold_basis,
                                         n_trials=20, horizon=20, rng=np.random.default_rng(3))
        res_tight = simulate_closed_loop(A, B, x0, target, None, trigger="manifold",
                                         trigger_threshold=10.0, manifold_basis=manifold_basis,
                                         n_trials=20, horizon=20, rng=np.random.default_rng(3))
        assert res_loose["duty_cycle"] > res_tight["duty_cycle"]
        assert res_tight["duty_cycle"] == 0.0

    def test_invalid_trigger_raises(self, rng):
        A, B, x0, target = _unstable_system()
        with pytest.raises(ValueError):
            simulate_closed_loop(A, B, x0, target, None, trigger="bogus", n_trials=2, horizon=2, rng=rng)


class TestBehavioralFlip:
    """Does on-demand control, driven from a real trial's OWN starting
    state, flip a decoder's binary call on that trial? Mirrors
    scripts/run_closed_loop_behavior_flip.py's per-trial design (n_trials=1,
    trigger="decoder", x0 = that trial's own state) directly on
    simulate_closed_loop (this project's test boundary is src/, not scripts/
    -- no existing test imports scripts/, so this exercises the same reused
    machinery the script drives rather than re-testing a script-private
    helper)."""

    def test_steerable_trial_flips_with_aligned_b(self, rng):
        # x0 starts on the "error" side of a threshold decoder at x[0]=0;
        # B is aligned with the unstable/steerable mode -> on-demand control
        # should push x[0] positive and flip the decoder's call.
        A, B, _, target = _unstable_system()
        x0_error_side = np.array([-0.6, 0.0])   # mirror image of the "correct" x0

        def threshold_decoder(X):
            return (X[:, 0] > 0.0).astype(int)   # 1 = "correct" (positive side)

        res = simulate_closed_loop(A, B, x0_error_side, target, threshold_decoder, label=1,
                                   trigger="decoder", n_trials=1, horizon=30,
                                   rng=np.random.default_rng(0))
        x_final_on = res["x_traj_on"][0, -1]
        assert threshold_decoder(x_final_on.reshape(1, -1))[0] == 1   # flipped to "correct"

    def test_random_direction_control_does_not_reliably_flip(self, rng):
        # D3's mandatory null control: the SAME on-demand machinery, but B
        # replaced by a direction ORTHOGONAL to the steerable mode (the
        # existing negative-control B from TestNegativeControlOrthogonalB) --
        # applying "the same energy in a random/uninformed direction" must
        # NOT reliably flip trials that the aligned-B case above does flip.
        A = np.array([[1.08, 0.0], [0.0, 0.6]])
        B_orth = np.array([[0.0], [1.0]])   # zero authority over the unstable mode
        x0_error_side = np.array([-0.6, 0.0])
        target = np.array([0.0, 0.0])

        def threshold_decoder(X):
            return (X[:, 0] > 0.0).astype(int)

        n_flipped = 0
        n_trials_test = 20
        for i in range(n_trials_test):
            res = simulate_closed_loop(A, B_orth, x0_error_side, target, threshold_decoder, label=1,
                                       trigger="decoder", n_trials=1, horizon=30,
                                       rng=np.random.default_rng(100 + i))
            x_final = res["x_traj_on"][0, -1]
            n_flipped += int(threshold_decoder(x_final.reshape(1, -1))[0] == 1)
        # B_orth has no authority over dim 0 (the decoder's axis); dim-0
        # dynamics are governed purely by the unstable A[0,0]=1.08 mode with
        # process noise, which should not reliably push x[0] positive.
        assert n_flipped / n_trials_test < 0.5


class TestRobustnessSweep:
    def test_output_structure(self, rng):
        A, B, x0, target = _unstable_system()
        res = robustness_sweep(A, B, x0, target, None, n_trials=15, horizon=20, n_boot=100,
                               mismatch_angles_deg=np.array([0.0, 45.0]),
                               noise_levels=np.array([0.05, 1.0]),
                               nonlinearity_scales=np.array([0.0, 0.2]),
                               rng=rng)
        for key in ["nominal_drift_reduction", "angle_sweep", "noise_sweep",
                    "nonlinearity_sweep", "failure_boundary_angle_deg",
                    "failure_boundary_obs_noise", "failure_boundary_nonlinearity_scale"]:
            assert key in res
        assert len(res["angle_sweep"]) == 2
        assert len(res["noise_sweep"]) == 2

    def test_retained_benefit_is_one_at_nominal(self, rng):
        A, B, x0, target = _unstable_system()
        res = robustness_sweep(A, B, x0, target, None, n_trials=30, horizon=25, n_boot=150,
                               mismatch_angles_deg=np.array([0.0]),
                               noise_levels=np.array([0.05]),
                               nonlinearity_scales=np.array([0.0]),
                               rng=rng)
        assert res["angle_sweep"][0]["retained_benefit"] == pytest.approx(1.0, abs=0.05)

    def test_retained_benefit_degrades_with_observation_noise(self, rng):
        A, B, x0, target = _unstable_system()
        res = robustness_sweep(A, B, x0, target, None, n_trials=40, horizon=30, n_boot=200,
                               mismatch_angles_deg=np.array([0.0]),
                               noise_levels=np.array([0.05, 1.0, 3.0, 6.0]),
                               nonlinearity_scales=np.array([0.0]),
                               rng=rng)
        retained = [row["retained_benefit"] for row in res["noise_sweep"]]
        assert np.all(np.diff(retained) < 0)   # strictly decreasing with more noise

    def test_failure_boundary_detected_when_crossed(self, rng):
        A, B, x0, target = _unstable_system()
        res = robustness_sweep(A, B, x0, target, None, n_trials=40, horizon=30, n_boot=200,
                               mismatch_angles_deg=np.array([0.0]),
                               noise_levels=np.array([0.05, 1.0, 3.0, 6.0, 10.0]),
                               nonlinearity_scales=np.array([0.0]),
                               failure_threshold=0.3,
                               rng=rng)
        assert res["failure_boundary_obs_noise"] is not None
        assert res["failure_boundary_obs_noise"] in [0.05, 1.0, 3.0, 6.0, 10.0]
