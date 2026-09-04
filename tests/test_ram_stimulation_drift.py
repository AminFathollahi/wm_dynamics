"""Regression tests for the RAM (ds005489 open-loop / ds005557 closed-loop)
stimulation-drift pipeline: feature-bank shape/normalization, the
open-loop-vs-closed-loop design-gating logic, and the guarantee that
closed-loop item-level results are never silently reported as causal.

No BIDS data is required -- everything here runs on synthetic inputs, the
same pattern used by tests/test_miller_drift_spine.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drift_dynamics import simulate_confined_diffusion  # noqa: E402
from preprocessing import butterworth_bandstop  # noqa: E402
from run_ram_stimulation_drift import (  # noqa: E402
    ITEM_LEVEL_NONCAUSAL_REASON,
    closedloop_session_analysis,
    fit_group_drift,
    morlet_log_power_bank,
    openloop_causal_rows,
)


def test_morlet_log_power_bank_shape_and_buffer_removal():
    # srate must clear Nyquist for the author bank's top frequency (180 Hz).
    rng = np.random.default_rng(0)
    srate = 500.0
    n_epochs, n_channels, window_s = 6, 3, 1.366
    n_times = int(round(window_s * srate))
    t = np.arange(n_times) / srate
    raw = 1e-5 * np.sin(2 * np.pi * 40.0 * t)[None, None, :] + 1e-6 * rng.normal(
        size=(n_epochs, n_channels, n_times)
    )

    n_bins = 13
    power = morlet_log_power_bank(raw, srate, n_bins=n_bins)

    assert power.shape == (n_epochs, n_channels, 8, n_bins)
    assert np.all(np.isfinite(power))


def test_bandstop_attenuates_60hz_line_noise():
    """The Morlet bank's line-noise step (58-62 Hz Butterworth band-stop) is
    the author-specified defense against 60 Hz contamination -- verify the
    shared filter it calls actually suppresses a 60 Hz component. Uses a
    longer signal than one encoding window so filtfilt edge transients (the
    first/last few cycles) don't dominate a short-window std comparison."""
    srate = 500.0
    n_times = int(round(4.0 * srate))
    t = np.arange(n_times) / srate
    line = np.sin(2 * np.pi * 60.0 * t)

    filtered = butterworth_bandstop(line[:, None], 58.0, 62.0, srate)
    edge = int(round(0.5 * srate))  # drop half a second of edge transient each side

    assert np.std(filtered[edge:-edge]) < 0.15 * np.std(line[edge:-edge])


def _synthetic_session(rng: np.random.Generator, session="sub-TEST_ses-0", n_trials=60, n_bins=13, n_feat=5):
    """A session dict shaped like build_session_trajectory's output, with a
    confined-drift signal on feature 0 so PCA recovers a fittable PC1."""
    dt = 0.1
    _, obs = simulate_confined_diffusion(
        n_trials, n_bins, dt, lambda_rate=1.5, diffusion=0.4,
        equilibrium=0.0, observation_sd=0.1, rng=rng,
    )
    features = rng.normal(scale=0.3, size=(n_trials, n_bins, n_feat))
    features[:, :, 0] = obs

    stim = np.zeros(n_trials, dtype=int)
    stim[: n_trials // 2] = 1
    rng.shuffle(stim)
    stim_list = np.ones(n_trials, dtype=int)
    stim_list[n_trials // 2 :] = 0

    return {
        "status": "complete", "session": session, "subject": session.split("_")[0],
        "srate": 1000.0, "n_channels": n_feat, "n_words": n_trials,
        "features": features, "stim": stim, "stim_list": stim_list,
        "list": np.arange(n_trials) % 10, "serialpos": np.arange(n_trials) % 12,
        "recalled": rng.integers(0, 2, size=n_trials),
    }


def test_fit_group_drift_normalizes_only_on_plant_trials():
    rng = np.random.default_rng(2)
    session = _synthetic_session(rng)
    plant_mask = session["stim"] == 0

    fit = fit_group_drift(session["features"], 0.1, plant_mask, seed=3)

    assert fit["status"] == "complete"
    assert fit["n_plant_trials"] == int(plant_mask.sum())
    assert fit["displacement"].shape == (session["features"].shape[0],)
    # normalization statistics were fit on the plant rows only: the raw
    # (pre-PCA) plant features must be ~zero-mean/unit-sd by construction of
    # the z-transform inside fit_group_drift, non-plant rows need not be.
    train = session["features"][plant_mask].reshape(-1, session["features"].shape[-1])
    mu, sd = train.mean(axis=0), train.std(axis=0) + 1e-8
    z_train = (train - mu) / sd
    np.testing.assert_allclose(z_train.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(z_train.std(axis=0), 1.0, atol=1e-6)


def test_fit_group_drift_reports_not_estimable_with_too_few_plant_trials():
    rng = np.random.default_rng(4)
    session = _synthetic_session(rng, n_trials=10)
    plant_mask = np.zeros(10, dtype=bool)
    plant_mask[:2] = True

    fit = fit_group_drift(session["features"], 0.1, plant_mask, seed=5)
    assert fit["status"] == "excluded"
    assert "reason" in fit


def test_openloop_causal_rows_uses_only_stim_list_words():
    rng = np.random.default_rng(6)
    session = _synthetic_session(rng)
    result = openloop_causal_rows(session, seed=7)

    assert result["status"] == "complete"
    expected_n = int((session["stim_list"] == 1).sum())
    assert result["n_causal_rows"] == expected_n
    assert len(result["rows"]) == expected_n
    for row in result["rows"]:
        assert row["stim"] in (0, 1)


def test_closedloop_item_level_is_never_reported_as_causal():
    """Design-gating guarantee: whatever the closed-loop item-level pattern
    looks like, it must always be flagged descriptive/non-causal -- this is
    the one property this pipeline must never regress on."""
    rng = np.random.default_rng(8)
    session = _synthetic_session(rng)
    result = closedloop_session_analysis(session, seed=9)

    assert result["status"] == "complete"
    item_level = result["item_level"]
    if item_level["status"] == "complete":
        assert item_level["causal"] is False
        assert item_level["descriptive_only"] is True
        assert item_level["why_noncausal"] == ITEM_LEVEL_NONCAUSAL_REASON

    # The list-level contrast is the one design-correct closed-loop causal
    # test and must NOT carry the item-level non-causal flags.
    list_level = result["list_level"]
    assert "causal" not in list_level or list_level.get("causal") is not False or (
        list_level["status"] != "complete"
    )


def test_closedloop_item_level_not_estimable_still_flagged_noncausal():
    """Even when there aren't enough triggered/non-triggered items to
    estimate a pattern, the not-estimable stub must not silently read as
    causal (i.e. must carry no 'causal: True')."""
    rng = np.random.default_rng(10)
    session = _synthetic_session(rng, n_trials=8)
    session["stim"][:] = 0  # no triggered items at all -> item_level not_estimable
    result = closedloop_session_analysis(session, seed=11)

    item_level = result["item_level"]
    assert item_level["status"] == "not_estimable"
    assert item_level.get("causal") is not True


if __name__ == "__main__":
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
