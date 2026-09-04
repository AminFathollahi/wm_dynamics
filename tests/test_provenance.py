import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from provenance import (
    ArtifactMetadata,
    canonical_json,
    checkpoint_safe,
    restore_checkpoint,
    sha256_file,
    validate_ledger,
    write_immutable_artifact,
)


def test_canonical_json_encodes_nonfinite_failed_estimates_as_null():
    encoded = canonical_json({"failed_score": float("nan"), "divergent": float("inf")})
    assert json.loads(encoded) == {"divergent": None, "failed_score": None}
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_canonical_json_handles_numpy_scalars_and_nonfinite_numpy_floats():
    encoded = canonical_json({
        "count": np.int64(3), "rate": np.float32(1.5), "failed": np.float32("nan"),
        "array": np.array([1.0, 2.0, float("inf")]),
    })
    assert json.loads(encoded) == {"count": 3, "rate": 1.5, "failed": None, "array": [1.0, 2.0, None]}
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_immutable_artifact_is_content_addressed(tmp_path):
    meta = ArtifactMetadata(
        analysis_id="unit_test", schema_version="1.0.0", status="current_exploratory",
        gate="G1", dataset_view="fixture", construct="fixture", estimand="mean",
        independent_unit="participant", seed=7, code_commit="abc", input_hashes={}, configuration={},
    )
    first = write_immutable_artifact(tmp_path, meta, {"estimate": 1.0})
    second = write_immutable_artifact(tmp_path, meta, {"estimate": 1.0})
    assert first == second
    assert (first / "artifact.json").is_file()


def test_ledger_rejects_missing_artifact_and_duplicate_claim(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    row = {
        "claim_id": "C1", "analysis_id": "A1", "dataset_view": "d", "construct": "c",
        "eligibility_rule": "all", "independent_unit": "patient", "estimand": "mean",
        "preprocessing_version": "1", "inferential_method": "sign flip", "correction_family": "none",
        "artifact_path": "artifact.json", "artifact_hash": sha256_file(artifact),
        "manuscript_location": "Appendix", "status": "current_exploratory", "gate": "G1", "caveat": "",
        "result_id": "R1", "model_prediction": "pending fitted model",
        "prediction_match_status": "pending_real_data_fit",
    }
    assert validate_ledger([row], tmp_path) == []
    errors = validate_ledger([row, row], tmp_path)
    assert any("duplicate claim_id" in error for error in errors)


def _round_trip(value):
    """The exact write-then-read path a checkpoint file goes through: encode,
    dump to a JSON string, parse it back, then restore. A test that only calls
    checkpoint_safe/restore_checkpoint directly, skipping the json.dumps/loads
    in between, would not catch a defect that only shows up once every numpy
    and Python-native type has actually been forced through a JSON string."""
    return restore_checkpoint(json.loads(json.dumps(checkpoint_safe(value), allow_nan=False)))


def test_checkpoint_round_trip_restores_array_dtype_and_shape():
    bundle = {
        "floats": np.array([1.5, -2.25, 0.0]),
        "ints": np.array([1, 2, 3], dtype=np.int64),
        "matrix": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        "mask": np.array([True, False, True, False]),
    }
    restored = _round_trip(bundle)
    for key, original in bundle.items():
        assert isinstance(restored[key], np.ndarray), key
        assert restored[key].dtype == original.dtype, key
        assert restored[key].shape == original.shape, key
        assert np.array_equal(restored[key], original), key


def test_checkpoint_round_trip_boolean_mask_indexes_identically():
    values = np.arange(10)
    mask = np.array([True, False, True, False, True, False, True, False, True, False])
    restored_mask = _round_trip(mask)
    assert restored_mask.dtype == bool
    assert np.array_equal(values[restored_mask], values[mask])
    # The naive reload (json.loads of checkpoint_safe's own "data" field, skipping
    # restore_checkpoint) is a plain Python list of True/False. numpy happens to
    # infer a boolean dtype from a pure bool list and index it correctly here --
    # but only because every element is a real Python bool with nothing else mixed
    # in. The arithmetic ops below are where the type difference is unconditional:
    naive_reload = json.loads(json.dumps(checkpoint_safe(mask), allow_nan=False))["data"]
    assert isinstance(naive_reload, list) and all(isinstance(v, bool) for v in naive_reload)
    assert naive_reload * 2 != list(mask * 2)  # list repeat vs elementwise scale-by-2
    assert (naive_reload + naive_reload) != list(mask + mask)  # concatenation vs elementwise add


def test_checkpoint_round_trip_nested_dict_and_list_of_arrays():
    # The shape a real checkpoint actually stores: _COMPLETED_FITS maps a fit key to
    # {"complete": bool, "value": <fit result>}, and a fit result nests arrays inside
    # dicts (per-session eigenvalues, weights) and inside lists (bootstrap replicates
    # stacked across draws). Only the tagged path (not the legacy heuristic fallback)
    # can recover an int array's exact dtype, so this is also the one case that would
    # catch a regression to float-only restoration.
    bundle = {
        "fits": {
            "session_a": {"eigenvalue_rank": np.array([1, 2, 3], dtype=np.int64),
                          "weights": np.array([0.1, 0.2, 0.7])},
            "session_b": {"eigenvalue_rank": np.array([4, 5], dtype=np.int64),
                          "weights": np.array([0.4, 0.6])},
        },
        "bootstrap_replicates": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
    }
    restored = _round_trip(bundle)
    for session in ("session_a", "session_b"):
        original = bundle["fits"][session]
        got = restored["fits"][session]
        for field in ("eigenvalue_rank", "weights"):
            assert isinstance(got[field], np.ndarray), (session, field)
            assert got[field].dtype == original[field].dtype, (session, field)
            assert np.array_equal(got[field], original[field]), (session, field)
    assert isinstance(restored["bootstrap_replicates"], np.ndarray)
    assert restored["bootstrap_replicates"].shape == (2, 2)
    assert np.array_equal(restored["bootstrap_replicates"], np.array(bundle["bootstrap_replicates"]))


def test_checkpoint_round_trip_falls_back_on_legacy_untagged_lists():
    # A checkpoint written before this tagging existed (plain _json_safe output,
    # no dtype/shape tag) must still come back as an array, not a list.
    legacy = json.loads(json.dumps({"mask": [True, False, True]}))
    restored = restore_checkpoint(legacy)
    assert isinstance(restored["mask"], np.ndarray)
    assert restored["mask"].dtype == bool
