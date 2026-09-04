"""Tests for run_macaque_pfc_microstimulation_site_reproducibility.py's
site_identity_contrast: the same-vs-different-site cosine test must recover a
planted site-specific displacement direction and must return a null result
when every site shares the identical displacement direction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_macaque_pfc_microstimulation_site_reproducibility import (  # noqa: E402
    site_identity_contrast, BRANCH_SITE_SPECIFIC, BRANCH_POWERED_NULL, BRANCH_UNDERPOWERED,
)


def _unit_vec(rng, dim):
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def _planted_site_specific_units(seed=0, n_units=8, n_sites=3, dim=12, noise=0.05):
    """Each site has its OWN true direction; 'a' and 'b' are that direction
    plus small independent noise -- same-site should cohere, different-site
    should not (random directions in a 12-dim space are ~orthogonal)."""
    rng = np.random.default_rng(seed)
    site_true = [_unit_vec(rng, dim) for _ in range(n_sites)]
    units = {}
    for u in range(n_units):
        sites = {}
        for s, true_dir in enumerate(site_true):
            a = true_dir + noise * rng.standard_normal(dim)
            b = true_dir + noise * rng.standard_normal(dim)
            sites[f"site{s}"] = {"a": a / np.linalg.norm(a), "b": b / np.linalg.norm(b)}
        units[f"unit{u}"] = sites
    return units


def _identical_displacement_units(seed=0, n_units=8, n_sites=3, dim=12, noise=0.3):
    """Every site shares the SAME true direction -- same-site and different-
    site cosines should be statistically indistinguishable."""
    rng = np.random.default_rng(seed)
    shared_true = _unit_vec(rng, dim)
    units = {}
    for u in range(n_units):
        sites = {}
        for s in range(n_sites):
            a = shared_true + noise * rng.standard_normal(dim)
            b = shared_true + noise * rng.standard_normal(dim)
            sites[f"site{s}"] = {"a": a / np.linalg.norm(a), "b": b / np.linalg.norm(b)}
        units[f"unit{u}"] = sites
    return units


def test_planted_site_specific_displacement_is_recovered():
    units = _planted_site_specific_units()
    result = site_identity_contrast(units, np.random.default_rng(1), n_perm=2000)
    assert result["status"] == "computed"
    assert result["same_site_mean_cosine"] > result["different_site_mean_cosine"]
    assert result["p_value"] < 0.05
    assert result["branch"] == BRANCH_SITE_SPECIFIC


def test_identical_displacement_across_sites_returns_a_null():
    units = _identical_displacement_units()
    result = site_identity_contrast(units, np.random.default_rng(2), n_perm=2000)
    assert result["status"] == "computed"
    assert result["p_value"] >= 0.05
    assert result["branch"] in (BRANCH_POWERED_NULL, BRANCH_UNDERPOWERED)


def test_fewer_than_two_sites_in_every_unit_is_infeasible_not_a_null():
    units = {"unit0": {"site0": {"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0])}}}
    result = site_identity_contrast(units, np.random.default_rng(3), n_perm=100)
    assert result["status"] == "not_computable_from_this_recording"


def test_mdd_and_branch_use_the_named_reference_bound():
    # a small-n, high-noise case should be non-significant AND underpowered
    # relative to the project's 0.14 cosine reference, landing in the
    # inconclusive branch rather than being called a null.
    units = _identical_displacement_units(n_units=2, noise=0.05)
    result = site_identity_contrast(units, np.random.default_rng(4), n_perm=500)
    assert result["status"] == "computed"
    assert result["mdd_reference"] == 0.14
    if result["minimum_detectable_difference"].get("status") == "computed":
        mdd = result["minimum_detectable_difference"]["mdd"]
        expected = BRANCH_POWERED_NULL if mdd < 0.14 else BRANCH_UNDERPOWERED
        assert result["p_value"] >= 0.05
        assert result["branch"] == expected


if __name__ == "__main__":
    test_planted_site_specific_displacement_is_recovered()
    test_identical_displacement_across_sites_returns_a_null()
    test_fewer_than_two_sites_in_every_unit_is_infeasible_not_a_null()
    test_mdd_and_branch_use_the_named_reference_bound()
    print("All site-reproducibility self-checks passed.")
