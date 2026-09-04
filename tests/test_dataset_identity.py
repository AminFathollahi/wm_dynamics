"""Regression tests for the DANDI 001187/000673 canonical-identity registry.

001187 and 000673 share 31 patients across 37 recording sessions (verified by
direct NWB identity checks -- see scripts/audit_dataset_identity.py and
provenance/dataset_overlap_report.json). This was corrected 2026-08-03 from a
prior 16-patient/19-session count: the audit originally grouped overlaps on
the raw NWB ``identifier`` string, which embeds a release-local upload-order
folder number ahead of the patient code (e.g. 000673's "sub-1_ses-1_P55CS" vs
001187's "sub-2_ses-1_P55CS" for the identical patient/session), so any
overlap where the two releases numbered the same patient differently was
silently missed. `scripts/audit_patient_identity_all_pairs.py`
independently confirmed the 31/37 count via a
content-based fingerprint (exact inter-trial-interval match), and
`audit_dataset_identity.py` was then fixed to group on (patient, session)
instead of the raw string. These tests guard the three ways that dedup can
silently regress: a known overlap disappearing from the audit output, a
duplicate patient-session-task row surviving into the canonical primary
table, and a downstream pooling site double-counting a linked 000673 view of
an already-canonical 001187 session.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from provenance import linked_duplicate_000673_session_keys  # noqa: E402

PROVENANCE = ROOT / "provenance"

# Verbatim ground truth from the verified 001187/000673 identity audit --
# independent of scripts/audit_dataset_identity.py's own EXPECTED_SHARED
# constant, so a test can't be fooled by someone quietly shrinking that set.
KNOWN_SHARED_PATIENTS = {
    "P68CS", "P69CS", "P70CS", "P71CS", "P76CS", "P77CS", "P78CS", "P79CS",
    "P1802JHU", "P1809JHU", "P1811JHU", "P61CS", "P62CS", "P64CS", "P65CS",
    "P67CS", "P55CS", "P58CS", "P73CS", "P088TWH", "P089TWH", "P090TWH",
    "P093TWH", "P096TWH", "P101TWH", "P103TWH", "P106TWH", "P109TWH",
    "P110TWH", "P113TWH", "P116TWH",
}
N_KNOWN_SHARED_SESSIONS = 37  # 31 patients; 6 patients each have 2 sessions


def _load_provenance_json(name: str) -> dict | list:
    path = PROVENANCE / name
    if not path.exists():
        pytest.skip(f"provenance/{name} not built -- run scripts/audit_dataset_identity.py")
    return json.loads(path.read_text())


def test_overlap_report_covers_every_known_shared_patient():
    report = _load_provenance_json("dataset_overlap_report.json")
    missed = KNOWN_SHARED_PATIENTS - set(report["verified_001187_000673_shared_patients"])
    assert not missed, f"overlap audit missed known shared patients: {sorted(missed)}"


def test_overlap_report_has_exactly_19_shared_sessions():
    report = _load_provenance_json("dataset_overlap_report.json")
    shared_groups = [g for g in report["overlap_groups"]
                      if {row["release"] for row in g} == {"001187", "000673"}]
    assert len(shared_groups) == N_KNOWN_SHARED_SESSIONS


def test_overlap_logic_rederived_from_raw_registry_finds_every_known_overlap():
    """Re-run the audit's own (patient, session) grouping + EXPECTED_SHARED
    comparison over the persisted recording registry (not the finished report),
    so a future regression in the audit script's grouping logic itself -- not
    just a stale output file -- fails this test. Grouping on the raw
    ``native_identifier`` string instead (the pre-2026-08-03 method) would
    fail this test: it recovers only the 16 patients whose release-local
    upload-order folder number happened to agree between releases.
    """
    records = _load_provenance_json("canonical_recording_registry.json")
    native_groups = defaultdict(list)
    for row in records:
        native_groups[(row["patient"], row["session"])].append(row)
    overlaps = [g for g in native_groups.values() if len({r["release"] for r in g}) > 1]
    shared_1187_673 = {
        row["patient"] for group in overlaps for row in group
        if {r["release"] for r in group} == {"001187", "000673"}
    }
    missed = KNOWN_SHARED_PATIENTS - shared_1187_673
    assert not missed, f"re-derived grouping missed known shared patients: {sorted(missed)}"


def test_raw_identifier_string_grouping_undercounts_the_overlap():
    """Documents a fixed defect: grouping on the raw identifier
    string alone recovers fewer than the true 31 shared patients, because it
    embeds a release-local upload-order folder number ahead of the patient
    code. If this test starts failing, the fixture data changed in a way that
    no longer demonstrates the original bug -- it is not a license to revert
    the grouping key in audit_dataset_identity.py.
    """
    records = _load_provenance_json("canonical_recording_registry.json")
    native_groups = defaultdict(list)
    for row in records:
        native_groups[row["native_identifier"]].append(row)
    overlaps = [g for g in native_groups.values() if len({r["release"] for r in g}) > 1]
    shared_1187_673 = {
        row["patient"] for group in overlaps for row in group
        if {r["release"] for r in group} == {"001187", "000673"}
    }
    assert shared_1187_673 < KNOWN_SHARED_PATIENTS


def test_canonical_primary_records_has_no_duplicate_patient_session_task():
    primary = _load_provenance_json("canonical_primary_records.json")
    seen = set()
    dupes = []
    for row in primary:
        for task in (row.get("task_groups") or [None]):
            key = (row["patient"], row["session"], task)
            if key in seen:
                dupes.append(key)
            seen.add(key)
    assert not dupes, f"duplicate (patient, session, task) rows in canonical primary table: {dupes}"


def test_canonical_primary_records_excludes_linked_000673_duplicates():
    primary = _load_provenance_json("canonical_primary_records.json")
    overlap_report = _load_provenance_json("dataset_overlap_report.json")
    linked_keys = linked_duplicate_000673_session_keys(overlap_report)
    leaked = [row["path"] for row in primary
              if row["release"] == "000673" and Path(row["path"]).stem in linked_keys]
    assert not leaked, f"linked-duplicate 000673 sessions leaked into primary table: {leaked}"


def test_linked_duplicate_keys_identify_only_the_000673_side_of_a_shared_session():
    fake_report = {"overlap_groups": [[
        {"release": "000673", "path": "000673/sub-1/sub-1_ses-1_ecephys+image.nwb"},
        {"release": "001187", "path": "001187/sub-1/sub-1_ses-1_ecephys+image.nwb"},
    ]]}
    assert linked_duplicate_000673_session_keys(fake_report) == {"sub-1_ses-1_ecephys+image"}


def test_pooling_drops_linked_duplicate_view_before_stouffer_combine():
    """Synthetic stand-in for the dedup wired into run_000673_pipeline.py /
    aggregate_pr_across_datasets.py / aggregate_forest_syntheses.py: a
    duplicated session must contribute one p-value to the pool, not two.
    """
    fake_report = {"overlap_groups": [[
        {"release": "000673", "path": "000673/sub-1/sub-1_ses-1_ecephys+image.nwb"},
        {"release": "001187", "path": "001187/sub-1/sub-1_ses-1_ecephys+image.nwb"},
    ]]}
    linked_keys = linked_duplicate_000673_session_keys(fake_report)

    dandi001187_ctg = {"sub-1_ses-1_ecephys+image": {"content_ctg": {"p_value": 0.01}}}
    dandi000673_ctg = {"sub-1_ses-1_ecephys+image": {"content_ctg": {"p_value": 0.02}}}

    p_pool_naive = ([v["content_ctg"]["p_value"] for v in dandi001187_ctg.values()]
                     + [v["content_ctg"]["p_value"] for v in dandi000673_ctg.values()])
    p_pool_dedup = ([v["content_ctg"]["p_value"] for v in dandi001187_ctg.values()]
                     + [v["content_ctg"]["p_value"] for k, v in dandi000673_ctg.items()
                        if k not in linked_keys])

    assert len(p_pool_naive) == 2, "sanity check on the fixture, not the fix"
    assert len(p_pool_dedup) == 1, "the linked 000673 view must be dropped before pooling"
