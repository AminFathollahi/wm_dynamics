"""Tests for scripts/run_transcranial_current_requirement.py -- the three
pieces this leg lives or dies on: the spatial join between a human
intracranial site and its nearest transcranial-corpus electrode obeys the
pre-declared matching radius exactly at the boundary, the zero-drop counts
for both corpora reconcile to the row count actually read (mutually
exclusive exclusion reasons, no double-count and no silent drop), and the
verdict rule sorts a required-current number against the pre-declared
threshold without ever softening or re-deriving that threshold from the
result."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_transcranial_current_requirement import (  # noqa: E402
    MATCHING_RADIUS_MM,
    UNREACHABLE_CURRENT_THRESHOLD_MA,
    coverage_by_region,
    reference_field_v_per_m,
    required_current_by_region,
    spatial_join,
)


def _tes1(name, x, group="A", voltage=1.0):
    return {
        "name": name, "group": group, "mni": (x, 0.0, 0.0),
        "voltage_mv_per_ma": voltage, "valid_coords": True, "valid_voltage": True,
        "field_mv_per_mm_per_ma": 0.5,
    }


def _ram(name, x, region="MTL"):
    return {
        "subject": "S1", "name": name, "mni": (x, 0.0, 0.0),
        "valid_coords": True, "coarse_region": region, "is_stim_site": False,
    }


def test_spatial_join_boundary_is_inclusive_and_exclusive_correctly():
    tes1 = [_tes1("T0", 0.0)]
    # exactly at the radius: matched. Just past it: not matched.
    at_radius = _ram("R_at", MATCHING_RADIUS_MM)
    past_radius = _ram("R_past", MATCHING_RADIUS_MM + 0.001)
    sites = spatial_join([at_radius, past_radius], tes1, MATCHING_RADIUS_MM)
    assert sites[0]["matched"] is True
    assert sites[0]["nearest_distance_mm"] == MATCHING_RADIUS_MM
    assert sites[1]["matched"] is False


def test_spatial_join_picks_the_nearest_not_the_first():
    tes1 = [_tes1("far", 9.0), _tes1("near", 1.0)]
    sites = spatial_join([_ram("R", 0.0)], tes1, MATCHING_RADIUS_MM)
    assert sites[0]["matched_tes1"]["name"] == "near"
    assert sites[0]["nearest_distance_mm"] == 1.0


def test_coverage_by_region_zero_drop_reconciles():
    tes1 = [_tes1("T0", 0.0)]
    sites = [
        _ram("R1", 1.0, "MTL"), _ram("R2", 50.0, "MTL"),  # 1 matched, 1 not
        _ram("R3", 1.0, "parietal"),  # matched
    ]
    sites = spatial_join(sites, tes1, MATCHING_RADIUS_MM)
    cov = coverage_by_region(sites)
    assert cov["MTL"]["n_sites"] == 2
    assert cov["MTL"]["n_matched"] == 1
    assert cov["MTL"]["n_uncovered"] == 1
    # seen = matched + uncovered, exactly, with no site unaccounted for.
    total_seen = sum(r["n_sites"] for r in cov.values())
    total_matched = sum(r["n_matched"] for r in cov.values())
    total_uncovered = sum(r["n_uncovered"] for r in cov.values())
    assert total_seen == len(sites) == total_matched + total_uncovered


def test_uncovered_site_is_reported_not_dropped():
    """A site with no match inside the radius must still appear in the
    coverage table -- as uncovered, never silently absorbed."""
    tes1 = [_tes1("T0", 0.0)]
    sites = spatial_join([_ram("R_far", 1000.0, "occipital")], tes1, MATCHING_RADIUS_MM)
    cov = coverage_by_region(sites)
    assert cov["occipital"]["n_sites"] == 1
    assert cov["occipital"]["n_matched"] == 0
    assert cov["occipital"]["n_uncovered"] == 1


def test_reference_field_scales_as_current_over_separation_squared():
    # E ~ I / d^2: doubling separation must quarter the field, doubling
    # current must double it -- the two knobs the formula is built from.
    base = reference_field_v_per_m(current_ma=1.0, separation_mm=1.0)
    assert reference_field_v_per_m(1.0, 2.0) == base / 4
    assert reference_field_v_per_m(2.0, 1.0) == base * 2


def test_verdict_rule_uses_the_predeclared_threshold_not_a_derived_one():
    coverage = {"MTL": {"n_matched": 2}}
    # field-per-mA of 1.0 mV/mm/mA and a reference field just below/above
    # threshold * field, chosen so required current lands on each side.
    below = {"MTL": {"reference_field_v_per_m": {"median": UNREACHABLE_CURRENT_THRESHOLD_MA - 0.5}}}
    above = {"MTL": {"reference_field_v_per_m": {"median": UNREACHABLE_CURRENT_THRESHOLD_MA + 0.5}}}
    field_lookup = {"MTL": [1.0, 1.0]}
    r_below = required_current_by_region(coverage, field_lookup, below)["MTL"]
    r_above = required_current_by_region(coverage, field_lookup, above)["MTL"]
    assert r_below["verdict"] == "within_the_declared_threshold"
    assert r_above["verdict"] == "beyond_the_declared_threshold"
    # the shortfall factor is reported whatever it is, never clipped to 1.0
    assert r_above["shortfall_factor_vs_threshold"] > 1.0
    assert r_below["shortfall_factor_vs_threshold"] <= 1.0


def test_region_with_no_matched_site_is_reported_not_computed():
    coverage = {"insula": {"n_matched": 0}}
    reference_field = {"insula": {"reference_field_v_per_m": {"median": 10.0}}}
    result = required_current_by_region(coverage, {}, reference_field)["insula"]
    assert result["status"] == "unmatched_no_transfer_estimate_in_this_region"
    assert "required_transcranial_current_ma" not in result


if __name__ == "__main__":
    import subprocess
    subprocess.check_call(["/home/amin/miniconda3/envs/wm_dynamics/bin/python", "-m", "pytest", "-q", __file__])


def test_distribution_summary_reports_the_actual_spread():
    from run_transcranial_current_requirement import _distribution_summary
    d = _distribution_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert d["n"] == 5
    assert d["min"] == 1.0
    assert d["max"] == 5.0
    assert d["median"] == 3.0


def test_distribution_summary_empty_is_explicit_not_a_fabricated_zero():
    from run_transcranial_current_requirement import _distribution_summary
    d = _distribution_summary([])
    assert d == {"n": 0}
