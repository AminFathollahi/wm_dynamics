import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    sys.path.insert(0, str(ROOT / _subdir))

import run_context_code_cross_dataset_rsa as rsa  # noqa: E402
import run_context_confidence_timecourse as confidence  # noqa: E402
import run_divergence_analysis as divergence  # noqa: E402
from provenance import (  # noqa: E402
    linked_duplicate_000673_session_keys,
    linked_duplicate_000673_sessions,
    load_overlap_report,
)

SHARED_STEM = "sub-1_ses-1_ecephys+image"
OWN_STEM = "sub-9_ses-1_ecephys+image"


def _write_report(directory):
    rows = lambda release, stem: {"release": release, "path": f"{release}/sub-1/{stem}.nwb"}  # noqa: E731
    report = {"overlap_groups": [
        [rows("000673", SHARED_STEM), rows("001187", "sub-2_ses-1_ecephys+image")],
        [rows("000469", "sub-3_ses-2_ecephys+image"), rows("000673", "sub-4_ses-1_ecephys+image")],
    ]}
    (directory / "dataset_overlap_report.json").write_text(json.dumps(report))


def test_only_000673_recordings_shared_with_001187_are_listed(tmp_path):
    _write_report(tmp_path)
    assert linked_duplicate_000673_sessions(tmp_path) == {SHARED_STEM}


def test_missing_report_raises_when_required_and_warns_otherwise(tmp_path):
    with pytest.raises(FileNotFoundError):
        linked_duplicate_000673_sessions(tmp_path)
    with pytest.warns(UserWarning):
        assert linked_duplicate_000673_sessions(tmp_path, required=False) == set()


def test_matches_the_selection_the_000673_pipeline_made_from_the_delivered_report():
    provenance = ROOT / "provenance"
    report = load_overlap_report(provenance)
    if report is None:
        pytest.skip("provenance/dataset_overlap_report.json not built")
    assert linked_duplicate_000673_sessions(provenance) == linked_duplicate_000673_session_keys(report)


def _geometry_file(directory, stem, seed):
    rng = np.random.default_rng(seed)
    n_trials, n_times = 30, 24
    np.savez(
        directory / f"dandi000673_geometry_{stem}.npz",
        Z=rng.normal(size=(n_trials, n_times, 8)),
        loads=np.tile([1, 3], n_trials // 2),
        times=np.arange(n_times) * 0.1,
    )


def test_cross_dataset_rsa_drops_linked_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(rsa, "RESULTS", tmp_path)
    _geometry_file(tmp_path, SHARED_STEM, 0)
    _geometry_file(tmp_path, OWN_STEM, 1)
    pattern = "dandi000673_geometry_sub-*.npz"
    assert len(rsa.dandi_rdms(pattern)) == 2
    assert len(rsa.dandi_rdms(pattern, exclude_keys={SHARED_STEM})) == 1


def test_confidence_timecourse_records_linked_sessions_without_testing_them(tmp_path, monkeypatch):
    monkeypatch.setattr(confidence, "RESULTS", tmp_path)
    _geometry_file(tmp_path, SHARED_STEM, 0)
    result = confidence._run_dataset(
        "dandi000673_geometry_sub-*.npz", "loads", "response_accuracy", 1, 3,
        len("dandi000673_geometry_"), exclude_keys={SHARED_STEM},
    )
    assert result[SHARED_STEM]["outcome_test"] is None
    assert "canonical" in result[SHARED_STEM]["exclusion_reason"]


def test_divergence_skips_linked_sessions_and_returns_them(tmp_path, monkeypatch):
    monkeypatch.setattr(divergence, "RESULTS", tmp_path)
    _geometry_file(tmp_path, SHARED_STEM, 0)
    out, rows = {}, {"dandi000673": {}}
    skipped = divergence._process_load1v3_dynamics(
        out, rows, "dandi000673", "dandi000673_geometry", exclude_keys={SHARED_STEM}
    )
    assert skipped == [SHARED_STEM] and out == {} and rows["dandi000673"] == {}


@pytest.mark.parametrize("producer", [
    "run_000673_pipeline", "run_behavior_ctg", "run_dim_robustness",
    "run_context_code_cross_dataset_rsa", "run_context_confidence_timecourse",
    "run_divergence_analysis",
])
def test_producer_reads_the_shared_exclusion_function(producer):
    tree = ast.parse((ROOT / "scripts" / f"{producer}.py").read_text())
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "linked_duplicate_000673_sessions" in called
