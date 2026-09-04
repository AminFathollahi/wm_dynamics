"""Checks on the multi-object macaque state-geometry script's two facts most
likely to break silently and go unnoticed: the reachable-lag arithmetic (a
plain range computation, but every downstream branch trusts it) and the
label-cardinality reachability bound (an empirical fact about the corpus,
not a code path -- a loader or binning change could silently make it look
reachable when the task design does not support it).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SCRIPT = ROOT / "scripts" / "run_watters_state_geometry.py"
SPEC = importlib.util.spec_from_file_location("run_watters_state_geometry", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reachable_lag_range_excludes_lags_whose_window_pair_does_not_fit():
    # 10 bins, 3-bin windows: lag must be >= width and leave room for both
    # windows, so bins 3..10 are candidates and _lag_pairs prunes any with no
    # valid (s, s+lag) pair -- exercised directly rather than re-derived here.
    result = MODULE.reachable_lag_range(n_bins=10, width_bins=3)
    assert result["reachable_lags_bins"] == [lag for lag in range(3, 11)
                                              if MODULE._lag_pairs(10, 3, lag)]
    assert result["n_reachable_lags"] == len(result["reachable_lags_bins"])
    # An epoch shorter than twice the window width reaches no lag at all.
    empty = MODULE.reachable_lag_range(n_bins=4, width_bins=3)
    assert empty["reachable_lags_bins"] == []


def test_triangle_single_object_cued_angle_caps_the_ladder_at_three_classes():
    """The finding that must survive into the artifact: the triangle task
    variant places a single object at one of exactly 3 fixed screen
    positions, so label-cardinality rungs K=4, 6 and 8 are unreachable by
    construction on triangle single-object trials, while the ring variant's
    continuous placement reaches every rung. A loader or binning regression
    that silently widened or narrowed this would misstate a bound this
    project reports explicitly rather than reproducing a null it could not
    have reached."""
    root_env = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root_env:
        pytest.skip("WM_DYNAMICS_DATA_ROOT is not set")
    root = Path(root_env)
    spikes_dir, behavior_dir = MODULE.watters_directories(root)
    if not behavior_dir.is_dir():
        pytest.skip("multi-object macaque behaviour tables are not staged locally")

    block = MODULE.cardinality_reachability_block(root)

    triangle = block["variants"]["triangle"]
    assert triangle["n_distinct_cued_positions_corpus_wide"] == 3
    assert triangle["reachable_rungs"] == [2, 3]
    assert triangle["unreachable_rungs"] == [4, 6, 8]

    ring = block["variants"]["ring"]
    assert ring["n_distinct_cued_positions_corpus_wide"] > 8
    assert ring["reachable_rungs"] == list(MODULE.CARDINALITY_RUNGS)
    assert ring["unreachable_rungs"] == []


if __name__ == "__main__":
    test_reachable_lag_range_excludes_lags_whose_window_pair_does_not_fit()
    print("reachable-lag-range check OK (no data root needed)")
    if os.environ.get("WM_DYNAMICS_DATA_ROOT"):
        test_triangle_single_object_cued_angle_caps_the_ladder_at_three_classes()
        print("triangle/ring cardinality-reachability check OK")
    else:
        print("WM_DYNAMICS_DATA_ROOT not set; skipped the data-dependent check")
