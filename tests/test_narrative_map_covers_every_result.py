"""tests/test_narrative_map_covers_every_result.py -- the zero-drop check:
every file in results/ must appear exactly once in results/narrative_map.json,
with a valid home, a non-null claim and evidence_gate, and (for superseded
entries) a non-null retired_by. This is the intended alarm when a later round
adds an artifact without giving it a home."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from provenance import VALID_GATES

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
HOME_VALUES = {"main_text", "supplementary", "methods_only", "superseded_recorded"}
DISPOSITION_VALUES = {"evidence", "plumbing", "superseded", "excluded"}


def _load_map() -> dict:
    return json.loads((RESULTS / "narrative_map.json").read_text())


def test_every_result_file_appears_exactly_once():
    narrative = _load_map()
    mapped_artifacts = [e["artifact"] for e in narrative["entries"]]
    assert len(mapped_artifacts) == len(set(mapped_artifacts)), "duplicate artifact entries in narrative_map.json"

    on_disk = {f"results/{p.name}" for p in RESULTS.iterdir() if p.is_file()}
    mapped = set(mapped_artifacts)
    missing = on_disk - mapped
    assert not missing, f"files on disk with no narrative_map entry: {sorted(missing)[:10]}"
    stale = mapped - on_disk
    assert not stale, f"narrative_map entries with no file on disk: {sorted(stale)[:10]}"


def test_home_is_one_of_the_four_permitted_values():
    narrative = _load_map()
    for entry in narrative["entries"]:
        assert entry["home"] in HOME_VALUES, f"{entry['artifact']} has invalid home {entry['home']!r}"


def test_superseded_entries_carry_a_retired_by():
    narrative = _load_map()
    for entry in narrative["entries"]:
        if entry["home"] == "superseded_recorded":
            assert entry.get("retired_by") is not None, f"{entry['artifact']} is superseded_recorded but has no retired_by"


def test_no_entry_missing_claim_or_evidence_gate():
    narrative = _load_map()
    for entry in narrative["entries"]:
        assert entry.get("claim"), f"{entry['artifact']} has no claim"
        assert entry.get("evidence_gate") in VALID_GATES, f"{entry['artifact']} has invalid evidence_gate {entry.get('evidence_gate')!r}"


def test_every_entry_has_a_defensible_disposition():
    narrative = _load_map()
    for entry in narrative["entries"]:
        assert entry.get("disposition") in DISPOSITION_VALUES
        assert entry.get("disposition_reason")


def test_reduced_settings_outputs_are_explicitly_excluded():
    narrative = _load_map()
    shakedowns = [
        entry for entry in narrative["entries"]
        if entry["artifact"].endswith(".shakedown.json")
    ]
    assert shakedowns
    assert all(entry["disposition"] == "excluded" for entry in shakedowns)
