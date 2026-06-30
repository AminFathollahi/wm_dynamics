"""Shared fixtures for all test modules."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_ieeg(rng):
    """Minimal synthetic iEEG dataset: 4 subjects, 3 conditions."""
    SRATE = 1200
    n_channels = 16
    n_seconds = 60
    T = n_seconds * SRATE

    data = rng.standard_normal((T, n_channels)) * 50.0  # µV scale

    # Build stim, task, target vectors mimicking Miller N-back
    stim = np.zeros(T, dtype=np.uint8)
    task = np.zeros(T, dtype=np.int8)
    target = np.zeros(T, dtype=np.uint8)

    # Place stimuli: one per 1.5 s, alternating conditions
    for i, onset in enumerate(range(300, T - 1800, 1800)):
        stim[onset : onset + 240] = (i % 26) + 1  # letter 1-26
        task[onset : onset + 1800] = i % 3  # 0/1/2-back
        target[onset : onset + 1800] = 1 + (i % 2)  # 1=non-target, 2=target

    return {
        "data": data,
        "stim": stim,
        "task": task,
        "target": target,
        "srate": SRATE,
        "subject": "test",
    }


@pytest.fixture
def synthetic_epochs(rng):
    """Synthetic epoch tensor (N, T, C) with known load structure."""
    N, T, C = 60, 200, 12
    epochs = rng.standard_normal((N, T, C)).astype(np.float32)

    # Add load-dependent signal: 2-back has 2x variance
    task_id = np.array([0] * 20 + [1] * 20 + [2] * 20, dtype=np.int8)
    tgt_id = np.array([1] * 10 + [2] * 10 + [1] * 10 + [2] * 10 + [1] * 10 + [2] * 10, dtype=np.uint8)

    for c in [0, 1, 2]:
        epochs[task_id == c] *= (c + 1) * 0.5

    times = np.linspace(-0.2, 1.5, T)
    return epochs, times, task_id, tgt_id


@pytest.fixture
def synthetic_system(rng):
    """2D stable linear system for control tests."""
    n, m = 4, 2
    # Stable rotation matrix (eigenvalues inside unit circle)
    theta = 0.1
    A_block = np.array([
        [0.95 * np.cos(theta), -0.95 * np.sin(theta)],
        [0.95 * np.sin(theta),  0.95 * np.cos(theta)],
    ])
    A = np.block([[A_block, np.zeros((2, 2))],
                  [np.zeros((2, 2)), 0.9 * np.eye(2)]])
    B = rng.standard_normal((n, m)) * 0.1
    return A, B
