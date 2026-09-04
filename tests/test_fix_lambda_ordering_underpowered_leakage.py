"""Regression test for scripts/fix_lambda_ordering_underpowered_leakage.py:
a `lambda_regional_ordering` block must never rank a
structure whose median came from a single identified patient, and must
never treat 'pooled' (the superseded chimeric baseline) as a peer structure.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import fix_lambda_ordering_underpowered_leakage as mod  # noqa: E402


def _region(median, n_patients):
    return {"group": {"state_space_lambda_identified_mean": {"median": median, "n_patients": n_patients}}}


class TestRebuildOrdering:
    def test_single_patient_structure_excluded_from_ranking(self):
        regions = {
            "hippocampus": _region(1.5, 5),
            "amygdala": _region(2.0, 1),  # underpowered -- must not rank first
            "dacc": _region(1.8, 3),
        }
        result = mod._rebuild_ordering(regions, list(regions))
        assert result["status"] == "estimable"
        assert "amygdala" not in result["region_lambda_state_space_median"]
        assert result["region_order_fastest_to_slowest"] == ["dacc", "hippocampus"]
        assert result["excluded_underpowered"]["amygdala"]["n_patients"] == 1

    def test_pooled_never_enters_the_ranking(self):
        regions = {
            "pooled": _region(9.0, 100),  # would rank first if not excluded by name
            "hippocampus": _region(1.5, 5),
            "amygdala": _region(2.0, 3),
        }
        result = mod._rebuild_ordering(regions, list(regions))
        assert "pooled" not in result["region_lambda_state_space_median"]
        assert "pooled" not in result["region_order_fastest_to_slowest"]
        assert "pooled" not in result["excluded_underpowered"]

    def test_non_identified_when_fewer_than_two_qualifying_structures(self):
        regions = {"hippocampus": _region(1.5, 1), "amygdala": _region(2.0, 1)}
        result = mod._rebuild_ordering(regions, list(regions))
        assert result["status"] == "non_identified"
