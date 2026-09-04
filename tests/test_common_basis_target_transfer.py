"""Tests for scripts/run_common_basis_target_transfer.py.

No data-mount dependency: every test either exercises a pure function
directly, or builds fully synthetic sessions and monkeypatches
run_macaque_pfc_microstimulation_pipeline.load_macaque_pfc_microstimulation_session so the real leave-one-session-out
transfer machinery (_leave_one_out_transfer, _rows_from_modifier,
benchmark_modifiers, _cluster_bootstrap_over_sessions -- the same functions
the production script calls) runs unmodified on synthetic data.

Covers three cases:
  1. a planted, shared-across-sessions target direction -- the transfer test
     must recover it (a significantly positive, session-clustered slope).
  2. an independent, unrelated-across-sessions target direction -- the
     transfer test must return null (no reliably positive slope).
  3. the common-channel-basis construction returns exactly the intersection
     of the supplied per-session channel identifier sets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_common_basis_target_transfer as m  # noqa: E402
import run_macaque_pfc_microstimulation_pipeline as macaque_pfc_microstimulation  # noqa: E402
from run_target_transfer import _cluster_bootstrap_over_sessions  # noqa: E402

N_BINS = macaque_pfc_microstimulation.N_BINS
CANONICAL_IDS = list(range(1, 11))  # 10 channels; only ids 1 and 2 carry any signal
GOOD_CHANNEL, BAD_CHANNEL = 1, 2


# ---------------------------------------------------------------------------------------------------
# 3. Common-basis construction is exactly the intersection of the supplied channel id sets
# ---------------------------------------------------------------------------------------------------

def test_common_basis_is_exactly_the_intersection():
    chan_sets = {
        "s1": {1, 2, 3, 4, 5},
        "s2": {2, 3, 4, 5, 6},
        "s3": {3, 4, 5, 6, 7},
    }
    basis = m._common_channel_basis(chan_sets)
    assert basis == {3, 4, 5}


def test_common_basis_empty_when_no_shared_channel():
    chan_sets = {"s1": {1, 2}, "s2": {3, 4}}
    assert m._common_channel_basis(chan_sets) == set()


def test_common_basis_of_no_sessions_is_empty():
    assert m._common_channel_basis({}) == set()


def test_common_basis_matches_python_set_intersection_on_a_larger_random_case():
    rng = np.random.default_rng(0)
    universe = np.arange(1, 100)
    chan_sets = {f"s{i}": set(rng.choice(universe, size=60, replace=False).tolist()) for i in range(12)}
    expected = set(universe.tolist())
    for s in chan_sets.values():
        expected &= s
    assert m._common_channel_basis(chan_sets) == expected


# ---------------------------------------------------------------------------------------------------
# Synthetic session builder shared by the recovery and null tests
# ---------------------------------------------------------------------------------------------------

def _session_fit(theta: float, k: int = 2) -> dict:
    """V (10, k): channel GOOD_CHANNEL's row is [cos theta, sin theta], channel
    BAD_CHANNEL's row is the 90-degree rotation of that (so it is always
    near-orthogonal to the good channel's own direction); every other
    channel's row is exactly zero (irrelevant filler, kept out of the
    average). v_star is fixed at [1, 0] for every session, so the entire
    per-session direction is carried by theta alone -- the quantity the
    two test scenarios below control directly.
    """
    V = np.zeros((len(CANONICAL_IDS), k))
    V[GOOD_CHANNEL - 1, :] = [np.cos(theta), np.sin(theta)]
    V[BAD_CHANNEL - 1, :] = [-np.sin(theta), np.cos(theta)]
    v_star = np.array([1.0, 0.0])
    return {"V": V, "v_star": v_star, "n_present": len(CANONICAL_IDS),
            "present_canonical_ids": list(CANONICAL_IDS)}


def _synthetic_corr_and_err(n_per_arm: int, good_accuracy: float, bad_accuracy: float,
                            control_accuracy: float = 0.5) -> tuple[dict, dict]:
    """control_idx=0 (stim_channels[0] empty marks it); condition 1 stims
    GOOD_CHANNEL, condition 2 stims BAD_CHANNEL. Accuracy is realised by
    splitting each condition's trials between the 'correct' and 'error'
    session dicts in the given proportion -- exactly the two files
    load_macaque_pfc_microstimulation_session itself returns, and the only way _rows_from_modifier
    (via _epochs_for) learns a trial's outcome label."""
    stim_channels = [[], [GOOD_CHANNEL], [BAD_CHANNEL]]
    control_idx = 0
    spec = {0: control_accuracy, 1: good_accuracy, 2: bad_accuracy}

    def _trials_for(is_correct_file: bool) -> list[dict]:
        trials = []
        for cond, acc in spec.items():
            n = int(round(n_per_arm * acc)) if is_correct_file else int(round(n_per_arm * (1 - acc)))
            for _ in range(n):
                trials.append({"spikerate": np.zeros((N_BINS + 5, 1), dtype=np.float32),
                              "stim_cond": cond, "angle_idx": 0})
        return trials

    base = {"stim_channels": stim_channels, "channel_ids": np.array(CANONICAL_IDS), "control_idx": control_idx}
    corr = {**base, "trials": _trials_for(True)}
    err = {**base, "trials": _trials_for(False)}
    return corr, err


def _install_fake_loader(monkeypatch, sessions: dict[str, tuple[dict, dict]]) -> None:
    def _fake_load(prefix, correct, neural_field="spikerate"):
        corr, err = sessions[prefix]
        return corr if correct else err
    monkeypatch.setattr(macaque_pfc_microstimulation, "load_macaque_pfc_microstimulation_session", _fake_load)


# ---------------------------------------------------------------------------------------------------
# 1. Planted, shared-across-sessions target -- must be recovered
# ---------------------------------------------------------------------------------------------------

def test_transfer_recovers_a_planted_shared_target(monkeypatch):
    rng = np.random.default_rng(1)
    session_names = [f"Wa_syn_{i}" for i in range(10)]
    thetas = rng.uniform(-0.1, 0.1, size=len(session_names))  # small shared-direction jitter

    fits = {s: _session_fit(theta) for s, theta in zip(session_names, thetas)}
    sessions = {s: _synthetic_corr_and_err(n_per_arm=60, good_accuracy=0.95, bad_accuracy=0.5)
               for s in session_names}
    _install_fake_loader(monkeypatch, sessions)

    per_session, _skip = m._leave_one_out_transfer(fits, CANONICAL_IDS, "test_recover", full_inference=True)
    assert len(per_session) >= 8, "expected almost every synthetic session to score"

    slopes = np.array([v["transfer_slope"] for v in per_session.values()])
    cluster = _cluster_bootstrap_over_sessions(slopes, n_boot=2000,
                                               rng=np.random.default_rng(2))
    assert cluster["ci_lo"] > 0, f"expected a significantly positive transfer slope, got CI {cluster}"
    assert cluster["p_value"] < 0.05


# ---------------------------------------------------------------------------------------------------
# 2. Independent, unrelated-across-sessions target -- must return null
# ---------------------------------------------------------------------------------------------------

def test_transfer_returns_null_for_independent_targets(monkeypatch):
    rng = np.random.default_rng(3)
    session_names = [f"Wa_syn_{i}" for i in range(10)]
    thetas = rng.uniform(0.0, 2 * np.pi, size=len(session_names))  # fully independent per session

    fits = {s: _session_fit(theta) for s, theta in zip(session_names, thetas)}
    # The within-session contrast (good channel really is more accurate than bad) still exists --
    # only the DIRECTION that identifies "good" is independent across sessions, so a transfer target
    # built from other sessions carries no systematic information about which channel is good here.
    sessions = {s: _synthetic_corr_and_err(n_per_arm=60, good_accuracy=0.95, bad_accuracy=0.5)
               for s in session_names}
    _install_fake_loader(monkeypatch, sessions)

    per_session, _skip = m._leave_one_out_transfer(fits, CANONICAL_IDS, "test_null", full_inference=True)
    assert len(per_session) >= 8

    slopes = np.array([v["transfer_slope"] for v in per_session.values()])
    cluster = _cluster_bootstrap_over_sessions(slopes, n_boot=2000,
                                               rng=np.random.default_rng(4))
    transfers_significantly = cluster["ci_lo"] > 0 and cluster["p_value"] < 0.05
    assert not transfers_significantly, f"expected no reliable transfer under independent targets, got {cluster}"


# ---------------------------------------------------------------------------------------------------
# A degenerate (zero within-session variance) modifier subset must be flagged as not fitted, a
# genuinely varying one must not -- the distinction the neutral-restriction check's paired test
# depends on to exclude sessions with no real within-session contrast instead of scoring them as 0.0.
# ---------------------------------------------------------------------------------------------------

def test_degenerate_modifier_subset_is_flagged_not_fitted():
    assert m._modifier_subset_is_not_fitted(np.array([]))
    assert m._modifier_subset_is_not_fitted(np.array([0.42]))
    assert m._modifier_subset_is_not_fitted(np.array([0.7, 0.7, 0.7, 0.7]))


def test_varying_modifier_subset_is_not_flagged():
    assert not m._modifier_subset_is_not_fitted(np.array([0.1, 0.5, 0.9, 0.3]))


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
