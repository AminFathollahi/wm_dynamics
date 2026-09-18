"""Tests for results/repeated_participant_inflation_census.json and the
participant_level_estimator_audit block appended to it (classifying the
artifacts named in summary.largest_exposure_artifacts). Reads the delivered
artifact only -- no analysis is re-run."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CENSUS_PATH = ROOT / "results" / "repeated_participant_inflation_census.json"

PERMITTED_LABELS = {
    "participant_level_estimator_present",
    "no_participant_level_estimator_found",
    "undetermined_from_artifact_contents",
}


@pytest.fixture(scope="module")
def census() -> dict:
    with open(CENSUS_PATH) as f:
        return json.load(f)


def test_census_loads_and_carries_documented_top_level_keys(census):
    for key in ("scope", "corpus_registry", "summary", "per_artifact", "participant_level_estimator_audit"):
        assert key in census


def test_every_corpus_has_sessions_gte_participants_and_ratio_matches(census):
    registry = census["corpus_registry"]
    assert len(registry) > 0
    for corpus, entry in registry.items():
        if not entry.get("enumerable", False):
            continue
        n_sessions = entry["n_sessions"]
        n_participants = entry["n_participants"]
        assert n_sessions >= n_participants, f"{corpus}: sessions < participants"
        expected_ratio = n_sessions / n_participants
        assert math.isclose(
            entry["ratio_sessions_per_participant"], expected_ratio, rel_tol=1e-9
        ), f"{corpus}: recorded ratio does not match sessions/participants"


def test_repeat_heavy_corpora_iff_ratio_above_one(census):
    registry = census["corpus_registry"]
    repeat_heavy_key = "repeat_heavy_corpora"
    assert repeat_heavy_key in census["summary"], "expected key not found in summary"
    repeat_heavy = set(census["summary"][repeat_heavy_key])

    for corpus, entry in registry.items():
        if not entry.get("enumerable", False):
            continue
        ratio = entry["ratio_sessions_per_participant"]
        if ratio > 1.0:
            assert corpus in repeat_heavy, f"{corpus} has ratio {ratio} > 1 but is not in {repeat_heavy_key}"
        else:
            assert corpus not in repeat_heavy, f"{corpus} has ratio {ratio} but is listed in {repeat_heavy_key}"

    # every entry in the list must correspond to a real, actually repeat-heavy corpus
    for corpus in repeat_heavy:
        assert corpus in registry
        assert registry[corpus]["ratio_sessions_per_participant"] > 1.0


def test_participant_level_estimator_audit_covers_exactly_flagged_artifacts(census):
    flagged_names = {entry["file"] for entry in census["summary"]["largest_exposure_artifacts"]}
    audit = census["participant_level_estimator_audit"]
    audited_names = set(audit["artifacts"].keys())

    assert audited_names == flagged_names

    for name, record in audit["artifacts"].items():
        assert record["label"] in PERMITTED_LABELS, f"{name} has an unpermitted label: {record['label']}"
