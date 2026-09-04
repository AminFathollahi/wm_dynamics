"""Tests for scripts/run_observability_and_power_census.py's two hardest
correctness guarantees: a carried-forward row is never silently refitted,
and an excluded admission-matrix cell never ships without its three numbers
(the excluding quantity, its measured value, and the required value)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_observability_and_power_census as census  # noqa: E402


@pytest.fixture
def fake_existing_census(tmp_path, monkeypatch):
    payload = {
        "schema_version": "1.0.0",
        "code_commit": "deadbeef",
        "rows": [
            {"dataset": "dandi_000469", "patient": "sub-1", "session": "sub-1_ses-2", "structure": "pooled",
             "epoch": "delay", "bin_ms": 100, "modality": "single_unit", "status": "fitted",
             "median_nugget_fraction": 0.42, "n_trials": 31, "n_units": 40, "n_splits_fitted": 30},
        ],
    }
    path = tmp_path / "fake_observability_census.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(census, "EXISTING_CENSUS_PATH", path)
    return payload


class TestCarriedRowsAreNeverRefit:
    def test_carry_forward_never_calls_the_estimator(self, fake_existing_census, monkeypatch):
        calls = []

        def _poison(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("nugget_fraction must never be called while carrying forward existing rows")

        monkeypatch.setattr(census, "nugget_fraction", _poison)
        rows, verification = census.carry_forward_existing_rows()
        assert calls == []
        assert len(rows) == 1
        assert rows[0]["carried_forward"] is True
        assert rows[0]["median_nugget_fraction"] == 0.42
        assert verification["existing_rows_are_what_they_claim"] is True

    def test_carried_row_values_are_verbatim_not_recomputed(self, fake_existing_census):
        rows, _ = census.carry_forward_existing_rows()
        original = fake_existing_census["rows"][0]
        for key in ("median_nugget_fraction", "n_trials", "n_units", "n_splits_fitted"):
            assert rows[0][key] == original[key]
        assert rows[0]["source_file"] == "results/observability_census.json"
        assert rows[0]["source_code_commit"] == "deadbeef"


def _synthetic_row(**overrides) -> dict:
    row = {
        "dataset": "synthetic_corpus", "modality": "single_unit", "structure": "pooled", "bin_ms": 100,
        "status": "fitted", "n_trials": 1, "n_units": 2, "n_splits_fitted": 1,
    }
    row.update(overrides)
    return row


class TestExcludedCellsCarryThreeNumbers:
    def test_below_floor_cell_ships_excluding_quantity_measured_and_required(self):
        rows = [_synthetic_row()]
        floors = census.minimum_measurable_configuration()
        matrix = census.build_admission_matrix(rows, floors)
        assert len(matrix) == 1
        cell = next(iter(matrix.values()))
        excluded = {name: v for name, v in cell["observables"].items() if v.get("status") == "excluded"}
        assert excluded, "this synthetic cell (n_trials=1, n_units=2, n_splits_fitted=1) should exclude every trial/node-gated observable"
        for name, v in excluded.items():
            assert "excluding_quantity" in v and v["excluding_quantity"], name
            assert "measured_value" in v and v["measured_value"] is not None, name
            assert "required_value" in v and v["required_value"] is not None, name

    def test_above_floor_cell_is_admitted_with_its_measured_value(self):
        rows = [_synthetic_row(n_trials=50, n_units=50, n_splits_fitted=20)]
        floors = census.minimum_measurable_configuration()
        matrix = census.build_admission_matrix(rows, floors)
        cell = next(iter(matrix.values()))
        admitted = {name: v for name, v in cell["observables"].items() if v.get("status") == "admitted"}
        assert admitted
        for name, v in admitted.items():
            assert v.get("measured_value") is not None, name
            assert v.get("required_value") is not None, name

    def test_no_cell_is_left_without_every_declared_observable(self):
        rows = [_synthetic_row()]
        floors = census.minimum_measurable_configuration()
        matrix = census.build_admission_matrix(rows, floors)
        cell = next(iter(matrix.values()))
        assert set(cell["observables"].keys()) == set(floors.keys())
        for v in cell["observables"].values():
            assert v.get("status") is not None


class TestCensusHeadlineFailsLoudOnMissingKey:
    """The defect this class of test guards against: a pre-declared branch
    resolved by a dict lookup that silently returns nothing for a missing
    admission-matrix key, so an empty set gets compared as though it were a
    measurement. See census.admission_matrix_key_lookup_defect for the full
    account of the incident this closes."""

    def test_missing_lfp_grain_key_raises_rather_than_returns_empty_set(self):
        matrix = {
            census.CENSUS_HEADLINE_UNIT_GRAIN_KEY: {
                "n_sessions": 10, "fraction_sessions_fitted": 1.0, "median_n_splits_fitted": 20,
                "median_n_units_or_channels": 50, "median_n_trials": 40, "structure": "pooled",
                "observables": {"cross_validated_nugget_fraction": {"status": "admitted"}},
            },
            # deliberately no dandi_000574::depth_contact_lfp::depth_contact_lfp::bin100 key
        }
        with pytest.raises(census.AdmissionMatrixKeyMissing):
            census.evaluate_census_headline(matrix, census.predeclared_census_headline_rule())

    def test_identical_admitted_sets_resolve_to_the_same_set_branch_not_an_empty_lfp_set(self):
        def _cell(structure):
            return {
                "n_sessions": 10, "fraction_sessions_fitted": 1.0, "median_n_splits_fitted": 20,
                "median_n_units_or_channels": 50, "median_n_trials": 40, "structure": structure,
                "observables": {"cross_validated_nugget_fraction": {"status": "admitted"}},
            }
        matrix = {
            census.CENSUS_HEADLINE_UNIT_GRAIN_KEY: _cell("pooled"),
            census.CENSUS_HEADLINE_LFP_GRAIN_KEY: _cell("depth_contact_lfp"),
        }
        result = census.evaluate_census_headline(matrix, census.predeclared_census_headline_rule())
        assert result["branch"] == "grains_admit_the_same_set"
        assert result["lfp_grain_admitted_observables"] == ["cross_validated_nugget_fraction"]


class TestAdmissionMargins:
    """Every admitted or excluded entry must gain a ratio and a band; a
    ratio must never be invented against an observable whose floor is
    unknown (provisionally_admitted_flagged)."""

    def test_admitted_entry_gains_ratio_and_band(self):
        rows = [_synthetic_row(n_trials=50, n_units=50, n_splits_fitted=20)]
        floors = census.minimum_measurable_configuration()
        matrix = census.build_admission_matrix(rows, floors)
        census.add_admission_margins(matrix)
        cell = next(iter(matrix.values()))
        admitted = [v for v in cell["observables"].values() if v["status"] == "admitted"]
        assert admitted
        for entry in admitted:
            assert entry["margin"]["ratio"] is not None
            assert entry["margin"]["ratio"] == pytest.approx(entry["measured_value"] / entry["required_value"])
            assert entry["margin"]["band"] in census.ADMISSION_MARGIN_BAND_ORDER

    def test_excluded_entry_gains_ratio_and_band(self):
        rows = [_synthetic_row()]
        floors = census.minimum_measurable_configuration()
        matrix = census.build_admission_matrix(rows, floors)
        census.add_admission_margins(matrix)
        cell = next(iter(matrix.values()))
        excluded = [v for v in cell["observables"].values() if v["status"] == "excluded"]
        assert excluded
        for entry in excluded:
            assert entry["margin"]["band"] is not None
            assert entry["margin"]["band"] == "below_floor"

    def test_no_ratio_invented_against_an_unknown_floor(self):
        rows = [_synthetic_row(n_trials=50, n_units=50, n_splits_fitted=20)]
        floors = census.minimum_measurable_configuration()
        matrix = census.build_admission_matrix(rows, floors)
        census.add_admission_margins(matrix)
        cell = next(iter(matrix.values()))
        flagged = [v for v in cell["observables"].values() if v["status"] == "provisionally_admitted_flagged"]
        assert flagged, "this synthetic cell should still carry the always-flagged observables (betti_numbers etc.)"
        for entry in flagged:
            assert "margin" not in entry

    def test_a_cell_admitted_at_exactly_the_floor_lands_in_the_at_floor_band(self):
        matrix = {
            "synthetic::cell": {
                "observables": {
                    "watts_strogatz_sigma_modularity_q_degree_assortativity": {
                        "status": "admitted", "measured_value": 10.0, "required_value": 10,
                    },
                },
            },
        }
        census.add_admission_margins(matrix)
        entry = matrix["synthetic::cell"]["observables"]["watts_strogatz_sigma_modularity_q_degree_assortativity"]
        assert entry["margin"]["ratio"] == 1.0
        assert entry["margin"]["band"] == "at_floor"


class TestMarginAwareModalityRuleFailsLoudOnMissingObservable:
    def test_observable_scored_in_one_grain_but_absent_from_the_other_raises(self):
        matrix = {
            "dandi_000574::single_unit::pooled::bin100": {
                "observables": {
                    "cross_validated_nugget_fraction": {
                        "status": "admitted", "measured_value": 3.0, "required_value": 3,
                        "margin": {"ratio": 1.0, "band": "at_floor"},
                    },
                },
            },
            "dandi_000574::depth_contact_lfp::depth_contact_lfp::bin100": {"observables": {}},
        }
        with pytest.raises(census.AdmissionMatrixKeyMissing):
            census.evaluate_margin_aware_modality_rule(matrix, census.predeclare_margin_aware_modality_rule())


class TestScalpRowCarriesBandAndGainsAMarginRatio:
    """The scalp grain is only runnable at the low band (high gamma is not
    representable at its sampling rate); a scalp row written without its own
    band label, or an admission-matrix cell built from it without a margin
    ratio, would silently let a reader assume it is a high-gamma cell like
    every other row in this artifact. Both must be present."""

    def _synthetic_scalp_row(self, n_channels: int, **overrides) -> dict:
        row = {
            "dataset": "dandi_000574", "patient": "sub-01", "session": "sub-01_ses-01",
            "structure": "scalp_eeg", "epoch": "delay", "bin_ms": 100, "modality": "scalp_eeg",
            "population_axis": "scalp_channel", "status": "fitted",
            "n_trials": 40, "n_units": n_channels, "n_splits_fitted": 20,
            "band": "low_band_8_45hz", "n_channels": n_channels,
            "n_channels_before_mastoid_exclusion": n_channels + 2,
            "reference_scheme": "common_average_excluding_mastoids_A1_A2",
        }
        row.update(overrides)
        return row

    def test_fitted_scalp_row_carries_its_own_band_field(self):
        row = self._synthetic_scalp_row(n_channels=17)
        assert row["band"] == "low_band_8_45hz", "a scalp row must name the band it was fit at -- it cannot be high gamma"

    def test_scalp_cell_gains_a_margin_ratio_after_admission_margins_are_added(self):
        rows = [self._synthetic_scalp_row(n_channels=17)]
        floors = census.minimum_measurable_configuration()
        matrix = census.build_admission_matrix(rows, floors)
        census.add_admission_margins(matrix)
        cell = next(iter(matrix.values()))
        graph_entry = cell["observables"]["watts_strogatz_sigma_modularity_q_degree_assortativity"]
        assert graph_entry["status"] == "admitted"
        assert graph_entry["margin"]["ratio"] == pytest.approx(17.0 / 10.0)
        assert graph_entry["margin"]["band"] == "adequate"

    def test_reduced_montage_scalp_cell_lands_below_floor_not_silently_admitted(self):
        # A patient with a reduced scalp montage (verified on real DANDI 000574
        # sessions: as few as 6-8 channels after mastoid exclusion) must not be
        # reported as clearing the graph observable's floor of 10.
        rows = [self._synthetic_scalp_row(n_channels=6)]
        floors = census.minimum_measurable_configuration()
        matrix = census.build_admission_matrix(rows, floors)
        census.add_admission_margins(matrix)
        cell = next(iter(matrix.values()))
        graph_entry = cell["observables"]["watts_strogatz_sigma_modularity_q_degree_assortativity"]
        assert graph_entry["status"] == "excluded"
        assert graph_entry["margin"]["band"] == "below_floor"
