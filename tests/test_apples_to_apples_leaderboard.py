"""Self-check for scripts/build_apples_to_apples_leaderboard.py's pure logic:
the DML-availability formatter and the K2/K4 self-check (Soldado-arena
primary leaderboard, all-trustworthy, breadth arena never aliased as
primary, every replication cell independently dataset-tagged).
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_apples_to_apples_leaderboard import (
    _fmt_dml, self_check, DML_MIN_TRUSTWORTHY_N,
)


def _valid_primary():
    return {
        "arena": "soldado_macaque_ustim",
        "unit_level": "trial",
        "metric": "gate-slope",
        "n_arms_scored": 6,
        "cluster_robust_significant_arms": ["vstar_alignment"],
        "arms": {
            "vstar_alignment": {
                "eligible": True, "n": 15670, "p_value_trustworthy": True,
                "slope_ci_lo": 0.019, "slope_ci_hi": 0.046,
            },
        },
    }


def _valid_breadth():
    return {"role": "secondary_breadth_arena", "arena": "boran_ieeg", "arms": {}}


def test_fmt_dml_none_is_unavailable():
    out = _fmt_dml(None)
    assert out == {"available": False}


def test_fmt_dml_flags_small_n_untrustworthy():
    out = _fmt_dml({"theta": 1.0, "se": 0.1, "ci_lo": 0.8, "ci_hi": 1.2, "p_value": 1e-9, "n": 6})
    assert out["available"] is True
    assert out["p_value_trustworthy"] is False


def test_fmt_dml_flags_large_n_trustworthy():
    n = DML_MIN_TRUSTWORTHY_N + 5
    out = _fmt_dml({"theta": 1.0, "se": 0.1, "ci_lo": 0.8, "ci_hi": 1.2, "p_value": 0.01, "n": n})
    assert out["p_value_trustworthy"] is True


def test_self_check_passes_on_valid_structure():
    replication = {"vstar_alignment": {"soldado": {"dataset": "soldado_macaque_ustim", "n": 100}}}
    self_check(_valid_primary(), _valid_breadth(), replication)  # must not raise


def test_self_check_rejects_cell_missing_dataset_tag():
    replication = {"vstar_alignment": {"soldado": {"n": 100}}}  # missing "dataset" key
    with pytest.raises(AssertionError):
        self_check(_valid_primary(), _valid_breadth(), replication)


def test_self_check_rejects_non_soldado_primary():
    bad_primary = dict(_valid_primary(), arena="boran_ieeg")
    with pytest.raises(AssertionError):
        self_check(bad_primary, _valid_breadth(), {})


def test_self_check_rejects_untrustworthy_eligible_primary_arm():
    bad_primary = _valid_primary()
    bad_primary["arms"]["vstar_alignment"]["p_value_trustworthy"] = False
    with pytest.raises(AssertionError):
        self_check(bad_primary, _valid_breadth(), {})


def test_self_check_rejects_breadth_aliased_as_primary():
    bad_breadth = dict(_valid_breadth(), role="primary")
    with pytest.raises(AssertionError):
        self_check(_valid_primary(), bad_breadth, {})


def test_self_check_rejects_degenerate_zero_width_ci():
    # Round-8.1 Part 11D: catches the anat_avg_ctrl/anat_modal_ctrl bug class --
    # an eligible arm with slope_ci_lo == slope_ci_hi (e.g. a constant modifier
    # silently regressed to a fake slope=0/CI=[0,0]) must be rejected.
    bad_primary = _valid_primary()
    bad_primary["arms"]["vstar_alignment"]["slope_ci_lo"] = 0.0
    bad_primary["arms"]["vstar_alignment"]["slope_ci_hi"] = 0.0
    with pytest.raises(AssertionError):
        self_check(bad_primary, _valid_breadth(), {})


def test_self_check_rejects_wrong_n_arms_scored():
    bad_primary = _valid_primary()
    bad_primary["n_arms_scored"] = 8
    with pytest.raises(AssertionError):
        self_check(bad_primary, _valid_breadth(), {})


def test_self_check_rejects_min_energy_as_cluster_robust():
    # Round-9 Part 13B: min_energy_dir_alignment does not survive session-clustering
    # (p=0.40) and must never be reported as cluster-robust-significant.
    bad_primary = _valid_primary()
    bad_primary["cluster_robust_significant_arms"] = ["vstar_alignment", "min_energy_dir_alignment"]
    with pytest.raises(AssertionError):
        self_check(bad_primary, _valid_breadth(), {})


if __name__ == "__main__":
    test_fmt_dml_none_is_unavailable()
    test_fmt_dml_flags_small_n_untrustworthy()
    test_fmt_dml_flags_large_n_trustworthy()
    test_self_check_passes_on_valid_structure()
    test_self_check_rejects_cell_missing_dataset_tag()
    test_self_check_rejects_non_soldado_primary()
    test_self_check_rejects_untrustworthy_eligible_primary_arm()
    test_self_check_rejects_breadth_aliased_as_primary()
    test_self_check_rejects_degenerate_zero_width_ci()
    test_self_check_rejects_wrong_n_arms_scored()
    test_self_check_rejects_min_energy_as_cluster_robust()
    print("All apples-to-apples leaderboard self-checks passed.")
