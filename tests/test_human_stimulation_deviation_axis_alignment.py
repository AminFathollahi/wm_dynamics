"""Tests for scripts/run_human_stimulation_deviation_axis_alignment.py -- the pieces this module adds
on top of the axis/displacement primitives it reuses unchanged from
scripts/run_macaque_pfc_microstimulation_stimulation_deviation_axis_alignment.py (already covered by
tests/test_macaque_pfc_microstimulation_stimulation_deviation_axis_alignment.py): (1) the anatomical channel-tier
classifier, and (2) the per-session-per-tier fit wrapper -- too few tier channels is refused, and a
planted along-axis displacement is still recovered once routed through this module's own tier-masking
and control/stimulated-trial split."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_human_stimulation_deviation_axis_alignment import (  # noqa: E402
    MIN_CHANNELS_PER_TIER, _fit_session_tier, classify_channel_tier,
)


# ---------------------------------------------------------------------------------------------------
# Channel-tier classification
# ---------------------------------------------------------------------------------------------------

_LABELS = {
    "LEC1": {"ind.region": "entorhinal"},
    "LEC2": {"ind.region": "n/a"},
    "LTS1": {"ind.region": "superiortemporal"},
    "LTS2": {"ind.region": "superiortemporal"},
    "LUN1": {"ind.region": "n/a"},
    "LUN2": {"ind.region": "misc"},
}


def test_mtl_labelled_contact_classifies_depth_mtl():
    assert classify_channel_tier("LEC1-LEC2", _LABELS) == "depth_mtl"


def test_cortical_labelled_contact_classifies_depth_cortical():
    assert classify_channel_tier("LTS1-LTS2", _LABELS) == "depth_cortical"


def test_both_contacts_unlabelled_is_unclassified():
    assert classify_channel_tier("LUN1-LUN2", _LABELS) is None


def test_first_contact_unlabelled_falls_back_to_second():
    assert classify_channel_tier("LEC2-LEC1", _LABELS) == "depth_mtl"


def test_malformed_channel_name_is_unclassified():
    assert classify_channel_tier("SINGLECONTACT", _LABELS) is None


# ---------------------------------------------------------------------------------------------------
# _fit_session_tier wrapper
# ---------------------------------------------------------------------------------------------------

def test_fit_session_tier_refuses_fewer_than_three_channels():
    arrays = {"epochs_log": np.zeros((40, 3, MIN_CHANNELS_PER_TIER - 1)),
              "stim_flag": np.array([0] * 20 + [1] * 20)}
    tier_mask = np.ones(MIN_CHANNELS_PER_TIER - 1, dtype=bool)
    out = _fit_session_tier(arrays, tier_mask, "test|too_few_channels")
    assert out["status"] == "excluded"
    assert out["reason"] == "fewer_than_3_admitted_channels_in_tier"


def test_fit_session_tier_recovers_planted_along_axis_displacement():
    rng = np.random.default_rng(0)
    n_units = 24
    axis_direction = rng.standard_normal(n_units)
    axis_direction /= np.linalg.norm(axis_direction)
    base = rng.standard_normal(n_units)
    base /= np.linalg.norm(base)

    n_ctrl, n_stim = 200, 150
    ctrl = base[None, :] * 8.0 + (rng.standard_normal(n_ctrl) * 1.0)[:, None] * axis_direction[None, :]
    stim = (base[None, :] * 8.0 + (rng.standard_normal(n_stim) * 1.0)[:, None] * axis_direction[None, :]
            + 3.0 * axis_direction[None, :])
    activity = np.concatenate([ctrl, stim], axis=0)
    epochs_log = activity[:, None, :]  # a single bin, so _bin_averaged's mean over bins is a no-op
    stim_flag = np.array([0] * n_ctrl + [1] * n_stim)
    arrays = {"epochs_log": epochs_log, "stim_flag": stim_flag}
    tier_mask = np.ones(n_units, dtype=bool)

    out = _fit_session_tier(arrays, tier_mask, "test|planted_along_axis")
    assert out["status"] == "computed"
    assert out["n_control_trials"] == n_ctrl and out["n_stim_trials"] == n_stim
    for arm in ("raw_axis", "detrended_axis"):
        cell = out["by_arm"][arm]
        assert cell["status"] == "computed"
        assert cell["real_alignment"]["observed_abs_cosine"] > 0.7


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
