"""Tests for scripts/compute_admission_matrix_margins.py: it must refuse to
run twice against the same artifact (so a re-run cannot silently double up
or mis-detect state), and it must never call any fitting code -- every
number it writes has to already be on disk."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import compute_admission_matrix_margins as margins  # noqa: E402


def _minimal_pre_margin_payload() -> dict:
    return {
        "schema_version": "2.0.0",
        "admission_matrix": {
            "dandi_000574::single_unit::pooled::bin100": {
                "observables": {
                    "cross_validated_nugget_fraction": {
                        "status": "admitted", "measured_value": 3.0, "required_value": 3,
                    },
                    "watts_strogatz_sigma_modularity_q_degree_assortativity": {
                        "status": "excluded", "excluding_quantity": "median units/channels per session",
                        "measured_value": 4.0, "required_value": 10,
                    },
                },
            },
            "dandi_000574::depth_contact_lfp::depth_contact_lfp::bin100": {
                "observables": {
                    "cross_validated_nugget_fraction": {
                        "status": "admitted", "measured_value": 20.0, "required_value": 3,
                    },
                    "watts_strogatz_sigma_modularity_q_degree_assortativity": {
                        "status": "admitted", "measured_value": 38.0, "required_value": 10,
                    },
                },
            },
            "dandi_000574::single_unit::pooled::bin200": {
                "observables": {
                    "cross_validated_nugget_fraction": {
                        "status": "excluded", "measured_value": 0.0, "required_value": 3,
                    },
                    "watts_strogatz_sigma_modularity_q_degree_assortativity": {
                        "status": "admitted", "measured_value": 53.0, "required_value": 10,
                    },
                },
            },
            "dandi_000574::depth_contact_lfp::depth_contact_lfp::bin200": {
                "observables": {
                    "cross_validated_nugget_fraction": {
                        "status": "admitted", "measured_value": 5.0, "required_value": 3,
                    },
                    "watts_strogatz_sigma_modularity_q_degree_assortativity": {
                        "status": "admitted", "measured_value": 38.0, "required_value": 10,
                    },
                },
            },
        },
    }


def test_refuses_to_run_against_an_artifact_that_is_not_the_expected_pre_margin_schema(tmp_path):
    path = tmp_path / "census.json"
    payload = _minimal_pre_margin_payload()
    payload["schema_version"] = "2.1.0"  # already patched
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        margins.add_margins_to_existing_census(path)


def test_patches_in_place_and_bumps_schema_version(tmp_path):
    path = tmp_path / "census.json"
    path.write_text(json.dumps(_minimal_pre_margin_payload()))
    result = margins.add_margins_to_existing_census(path)
    assert result["schema_version"] == margins.SCHEMA_VERSION_WITH_MARGINS
    on_disk = json.loads(path.read_text())
    assert on_disk["schema_version"] == margins.SCHEMA_VERSION_WITH_MARGINS
    unit_entry = on_disk["admission_matrix"]["dandi_000574::single_unit::pooled::bin100"]["observables"]["cross_validated_nugget_fraction"]
    assert unit_entry["margin"]["band"] == "at_floor"
    assert unit_entry["status"] == "admitted" and unit_entry["measured_value"] == 3.0  # unchanged beside the new field
    assert "admission_margin_recount" in on_disk
    assert "small_worldness_margin_test_case" in on_disk
    assert "margin_aware_modality_result" in on_disk
