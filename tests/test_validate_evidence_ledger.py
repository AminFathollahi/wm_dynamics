"""Regression tests for artifact-level inference diagnostics."""

import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from provenance import ArtifactMetadata, VALID_STATUSES, write_immutable_artifact


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_evidence_ledger.py"
SPEC = importlib.util.spec_from_file_location("validate_evidence_ledger", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_rejects_identified_group_with_failed_optimizer():
    artifact = {
        "status": "complete", "group_identified": True,
        "negative_log_likelihood": 1e30,
    }
    errors = MODULE.validate_inference_diagnostics(artifact)
    assert any("optimizer failure" in error for error in errors)


def test_rejects_comparison_verdict_without_entity_uncertainty():
    artifact = {"entity_medians": {"m4_minus_m2": 0.02}, "interpretation": "supported"}
    errors = MODULE.validate_inference_diagnostics(artifact)
    assert any("lacks uncertainty" in error for error in errors)


def test_pipeline_provenance_is_a_valid_ledger_status():
    assert "pipeline_provenance" in VALID_STATUSES


def test_pipeline_provenance_writes_an_artifact(tmp_path):
    metadata = ArtifactMetadata(
        analysis_id="pipeline_inventory", schema_version="1.0.0",
        status="pipeline_provenance", gate="G0", dataset_view="fixture",
        construct="inventory", estimand="not_applicable",
        independent_unit="not_applicable", seed=0, code_commit=None,
        input_hashes={}, configuration={},
    )
    output = write_immutable_artifact(tmp_path, metadata, {"files": 3})
    assert (output / "artifact.json").is_file()
