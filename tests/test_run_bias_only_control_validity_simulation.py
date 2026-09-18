"""Tests for run_bias_only_control_validity_simulation.py -- checks the three simulation regimes
produce the qualitative firing behaviour the module's own docstring and decision rule declare, at a
single small, fast sweep cell (not the full production sweep, which this test would make too slow to
run routinely)."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_bias_only_control_validity_simulation as sim  # noqa: E402


def _run_regime(regime: str, n_sessions: int = 22, trials_per_session: int = 60, n_replications: int = 150):
    rng = np.random.default_rng(sim.stable_seed(f"test|{regime}|{n_sessions}|{trials_per_session}"))
    draws = [sim._simulate_one(regime, n_sessions, trials_per_session, rng) for _ in range(n_replications)]
    tested = [d for d in draws if d["real_status"] == "tested"]
    assert tested, "slope_across_sessions_test should be powered at n_sessions=22"
    return tested


def test_regime_one_zero_within_effect_control_fires_and_real_reads_near_zero():
    """Regime 1: no true within-session effect, strong between-session association. The control
    (the between-session diagnostic) should fire far more often than the nominal alpha, and the real
    within-session statistic's own point estimate should average near zero."""
    tested = _run_regime(sim.REGIME_ZERO_WITHIN_STRONG_BETWEEN)
    control_firing_rate = np.mean([d["bias_significant"] for d in tested])
    mean_real_effect = np.mean([d["real_mean_value"] for d in tested])
    assert control_firing_rate > 0.5, f"control should fire often in regime 1, got {control_firing_rate}"
    assert abs(mean_real_effect) < 0.05, f"real effect should read ~0 in regime 1, got {mean_real_effect}"


def test_regime_two_true_within_effect_control_does_not_fire():
    """Regime 2: true within-session effect, zero between-session association. The control should fire
    at close to its nominal false-positive rate, not systematically."""
    tested = _run_regime(sim.REGIME_TRUE_WITHIN_ZERO_BETWEEN)
    control_firing_rate = np.mean([d["bias_significant"] for d in tested])
    real_significant_rate = np.mean([d["real_significant"] for d in tested])
    assert control_firing_rate < 0.20, f"control should rarely fire in regime 2, got {control_firing_rate}"
    assert real_significant_rate > 0.5, "the true within-session effect should be detectable most of the time"


def test_regime_three_independent_between_session_association_produces_false_voids():
    """Regime 3: true within-session effect AND an independently generated strong between-session
    association. The real effect is present by construction in every replication, so any voided
    replication is a false void. This is the headline number the module reports, and it should be
    materially larger than the nominal alpha -- demonstrating the voiding rule is unsound, not merely
    imperfect."""
    tested = _run_regime(sim.REGIME_TRUE_WITHIN_AND_INDEPENDENT_BETWEEN)
    real_significant_rate = np.mean([d["real_significant"] for d in tested])
    voiding_rate = np.mean([d["voided"] for d in tested])
    assert real_significant_rate > 0.5, "the true within-session effect should be detectable most of the time"
    assert voiding_rate > 0.10, (
        f"regime 3 false-void rate should be materially above nominal alpha, got {voiding_rate}")


def test_voiding_rule_never_fires_when_real_statistic_is_not_significant():
    """The delivered voiding rule (and this simulation's own replica of it) requires the real statistic
    to be significant before a bias-only control can void anything -- a non-significant real statistic
    was never a positive to begin with, so it cannot be voided."""
    for regime in sim.REGIMES:
        tested = _run_regime(regime, n_replications=80)
        for d in tested:
            if not d["real_significant"]:
                assert not d["voided"]


def test_magnitude_diagnostic_runs_and_returns_a_correlation_near_zero_when_generative_processes_are_independent():
    """Sanity check on the magnitude-information diagnostic itself: built directly (not through the
    full sweep, for speed), pooling regime-2/3-style replications, the correlation between the
    bias-only control's own r and the real effect's own point estimate should be small in magnitude --
    the two are computed from independent components of total covariance by construction."""
    rng = np.random.default_rng(0)
    bias_vals, real_vals = [], []
    for _ in range(400):
        d = sim._simulate_one(sim.REGIME_TRUE_WITHIN_AND_INDEPENDENT_BETWEEN, 22, 60, rng)
        bias_vals.append(d["bias_r"])
        real_vals.append(d["real_mean_value"])
    from scipy.stats import pearsonr
    r, _p = pearsonr(bias_vals, real_vals)
    assert abs(r) < 0.35, f"bias-only magnitude should carry little information about real effect size, got r={r}"


if __name__ == "__main__":
    test_regime_one_zero_within_effect_control_fires_and_real_reads_near_zero()
    test_regime_two_true_within_effect_control_does_not_fire()
    test_regime_three_independent_between_session_association_produces_false_voids()
    test_voiding_rule_never_fires_when_real_statistic_is_not_significant()
    test_magnitude_diagnostic_runs_and_returns_a_correlation_near_zero_when_generative_processes_are_independent()
    print("all checks passed")
