import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_manuscript.py"
SPEC = importlib.util.spec_from_file_location("validate_manuscript", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _ledger():
    # These three carried the project's earlier confinement-rate/diffusion spine, which is
    # retired (results/human_drift_spine_000469.json and its two companions are now reclassified
    # to "superseded" in provenance/evidence_ledger.json, matching validate_manuscript.py's own
    # required_current dict, since the permutation-based cross-unit-state result superseded them).
    return [
        {"artifact_path": "results/human_drift_spine_000469.json", "status": "superseded"},
        {"artifact_path": "results/human_drift_behavior_000469.json", "status": "superseded"},
        {"artifact_path": "results/drift_control_payload_000469.json", "status": "superseded"},
    ]


def _paper():
    return r"""
550 tests
001187 and 000673 are NOT independent
\citep{wolff2017} \citep{barbosa2021}
supports neither reading
as settled fact
results/human_drift_spine_000469.json
results/human_drift_behavior_000469.json
results/drift_control_payload_000469.json
\part{Retained exploratory results archive}
"""


def test_accepts_current_claim_boundary():
    assert MODULE.validate_text(_paper(), "550 tests", _ledger()) == []


def test_rejects_stale_counts_and_forbidden_language():
    errors = MODULE.validate_text(_paper() + "\n417 tests\ntrending", "550 tests", _ledger())
    assert any("stale test count" in error for error in errors)
    assert any("forbidden" in error for error in errors)


def test_rejects_positive_control_claim_before_archive():
    paper = _paper().replace(
        r"\part{Retained exploratory results archive}",
        "We demonstrate control.\n" + r"\part{Retained exploratory results archive}",
    )
    assert any("control claim" in error for error in MODULE.validate_text(paper, "550 tests", _ledger()))


def test_rejects_uncited_current_artifact_but_not_superseded_intermediate():
    ledger = _ledger() + [
        {"artifact_path": "results/uncited_current.json", "status": "current_exploratory"},
        {"artifact_path": "results/smoke_intermediate.json", "status": "superseded"},
    ]
    errors = MODULE.validate_text(_paper(), "550 tests", ledger)
    assert errors == ["current ledger artifact has no manuscript citation: uncited_current.json"]


def test_pipeline_provenance_status_exempt_but_evidence_status_still_required():
    ledger = _ledger() + [
        {"artifact_path": "results/uncited_provenance.json", "status": "pipeline_provenance"},
        {"artifact_path": "results/uncited_current.json", "status": "current_exploratory"},
    ]
    errors = MODULE.validate_text(_paper(), "550 tests", ledger)
    assert errors == ["current ledger artifact has no manuscript citation: uncited_current.json"]
