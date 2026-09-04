from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_haslacher_phase_diffusion import (  # noqa: E402
    bin_analog_trials,
    group_vector_test,
    harmonic_coefficients,
)
from run_haslacher_stimulation_geometry import _sass_sanity  # noqa: E402


def test_bin_analog_trials_is_nonoverlapping():
    trials = np.arange(2 * 3 * 40, dtype=float).reshape(2, 3, 40)
    binned = bin_analog_trials(trials, sampling_rate=100.0, bin_ms=100)
    assert binned.shape == (2, 3, 4)
    assert binned[0, 0, 0] == np.mean(np.arange(10))
    assert binned[0, 0, 1] == np.mean(np.arange(10, 20))


def test_harmonic_coefficients_recovers_phase_vector():
    phases = {3: 30, 4: 90, 5: 150, 6: 210, 1: 270, 2: 330}
    values = {code: 2.0 + 0.4 * np.cos(np.deg2rad(phase)) - 0.2 * np.sin(np.deg2rad(phase))
              for code, phase in phases.items()}
    result = harmonic_coefficients(values)
    assert result is not None
    assert abs(result["cosine"] - 0.4) < 1e-10
    assert abs(result["sine"] + 0.2) < 1e-10


def test_group_vector_test_uses_participants_as_units():
    rows = [{"harmonic": {"cosine": 0.5, "sine": 0.0}} for _ in range(6)]
    result = group_vector_test(rows, ("harmonic",), "fixture", n_perm=200)
    assert result is not None
    assert result["n_participants"] == 6
    assert result["population_amplitude"] == 0.5
    assert result["circular_rotation_p_value"] < 0.1


def test_sass_sanity_requires_improvement_and_bounded_power_ratio():
    assert _sass_sanity(1.0, 100.0, 2.0)["passed"]
    assert not _sass_sanity(1.0, 2.0, 3.0)["passed"]
    assert not _sass_sanity(1.0, 100.0, 5.0)["passed"]
