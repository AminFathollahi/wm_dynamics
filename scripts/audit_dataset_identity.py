#!/usr/bin/env python3
"""Generate a canonical NWB recording registry and release-overlap report.

The audit reads metadata and compact trial/unit table hashes only.  It does not
modify source data and intentionally does not use filenames as identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from provenance import canonical_json, sha256_file  # noqa: E402

RELEASES = ("000469", "000574", "000673", "001187")

# Historical value (superseded 2026-08-03): overlap grouping originally keyed
# on the full ``native_identifier`` string, which embeds the release-local
# upload-order folder ("sub-N") ahead of the patient code, e.g. 000673's
# "sub-1_ses-1_P55CS" vs 001187's "sub-2_ses-1_P55CS" for the SAME patient
# and session -- two different strings that never grouped together, so this
# set silently missed every overlap where the two releases assigned
# different sub-N numbers to the same patient. `scripts/audit_patient_identity_all_pairs.py`
# demonstrated the true count is 31 shared patients via a
# content fingerprint (exact inter-trial-interval match, not identifier
# string), a strict superset of this list. Retained only so the discrepancy
# stays visible; do not use this set to gate anything.
EXPECTED_SHARED_SUPERSEDED_2026_08_02 = {
    "P68CS", "P69CS", "P70CS", "P71CS", "P76CS", "P77CS", "P78CS", "P79CS",
    "P1802JHU", "P1809JHU", "P1811JHU", "P61CS", "P62CS", "P64CS", "P65CS",
    "P67CS",
}
EXPECTED_SHARED = EXPECTED_SHARED_SUPERSEDED_2026_08_02 | {
    "P55CS", "P58CS", "P73CS", "P088TWH", "P089TWH", "P090TWH", "P093TWH",
    "P096TWH", "P101TWH", "P103TWH", "P106TWH", "P109TWH", "P110TWH",
    "P113TWH", "P116TWH",
}


def text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return text(value.item())
    return str(value)


def compact_hash(group: h5py.Group | None) -> str | None:
    if group is None:
        return None
    digest = hashlib.sha256()
    for name in sorted(group.keys()):
        node = group[name]
        if isinstance(node, h5py.Dataset):
            digest.update(name.encode())
            digest.update(str(node.shape).encode())
            digest.update(str(node.dtype).encode())
            # The first/last small slices capture table identity while avoiding
            # a costly raw-data checksum for every NWB file.
            try:
                if node.shape == ():
                    sample = np.asarray(node[()]).reshape(-1)
                elif node.shape[0] == 0:
                    sample = np.empty(0, dtype=node.dtype)
                else:
                    # Slice the HDF5 dataset directly. ``node[()]`` would load
                    # the complete voltage/spike table and turns a metadata
                    # audit into an accidental multi-hundred-GB read.
                    head = np.asarray(node[: min(64, node.shape[0])]).reshape(-1)
                    tail = np.asarray(node[max(0, node.shape[0] - 64) :]).reshape(-1)
                    sample = np.concatenate((head, tail))
                digest.update(np.ascontiguousarray(sample).tobytes())
            except (TypeError, ValueError, OSError):
                pass
    return digest.hexdigest()


def patient_session_from_identifier(identifier: str, path: Path) -> tuple[str, str]:
    parts = identifier.split("_")
    patient = next((p for p in parts if p.startswith("P")), path.parent.name)
    session = next((p for p in parts if p.startswith("ses-")), path.stem.split("_ecephys")[0])
    return patient, session


def record(path: Path, release: str, root: Path, compute_asset_checksums: bool) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        identifier = text(handle["identifier"][()]) if "identifier" in handle else ""
        patient, session = patient_session_from_identifier(identifier, path)
        intervals = handle.get("intervals")
        trial_groups = [] if intervals is None else [intervals[name] for name in sorted(intervals)]
        trial_hashes = [compact_hash(group) for group in trial_groups]
        unit_hash = compact_hash(handle.get("units"))
        return {
            "release": release,
            "path": str(path.relative_to(root)),
            "asset_sha256": sha256_file(path) if compute_asset_checksums else None,
            "asset_checksum_status": "computed" if compute_asset_checksums else "pending_full_hash",
            "asset_size_bytes": path.stat().st_size,
            "native_identifier": identifier,
            "patient": patient,
            "session": session,
            "task_groups": [] if intervals is None else sorted(intervals.keys()),
            "trial_table_hashes": trial_hashes,
            "unit_metadata_hash": unit_hash,
            "acquisition_time": text(handle["session_start_time"][()]) if "session_start_time" in handle else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=os.environ.get("WM_DYNAMICS_DATA_ROOT", ""))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "provenance")
    parser.add_argument("--compute-asset-checksums", action="store_true",
                        help="Read every source NWB to calculate SHA-256; this is intentionally opt-in.")
    args = parser.parse_args()
    if not args.data_root or not args.data_root.is_dir():
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT or pass --data-root to the external data directory.")
    records = []
    for release in RELEASES:
        for path in sorted((args.data_root / release).glob("**/*.nwb")):
            records.append(record(path, release, args.data_root, args.compute_asset_checksums))
    # Group by (patient, session), not the raw identifier string: the string
    # embeds a release-local upload-order folder number ahead of the patient
    # code (e.g. 000673 "sub-1_ses-1_P55CS" vs 001187 "sub-2_ses-1_P55CS" for
    # the identical recording), so two releases' own numbering for the same
    # patient/session never matched under a raw-string key. See
    # EXPECTED_SHARED's docstring and results/patient_identity_audit.json.
    native_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        native_groups[(row["patient"], row["session"])].append(row)
    overlaps = [group for group in native_groups.values() if len({r["release"] for r in group}) > 1]
    shared_1187_673 = {
        row["patient"] for group in overlaps for row in group
        if {r["release"] for r in group} == {"001187", "000673"}
    }
    missed = sorted(EXPECTED_SHARED - shared_1187_673)
    unexpected = sorted(shared_1187_673 - EXPECTED_SHARED)
    primary = []
    seen_identity: set[tuple[str, str, str]] = set()
    for row in sorted(records, key=lambda x: (x["patient"], x["session"], x["release"], x["path"])):
        # Prefer 001187 for the identified overlapping pair; every other view
        # remains in the registry and is explicitly linked.
        key = (row["patient"], row["session"], row["native_identifier"])
        if key in seen_identity or (row["release"] == "000673" and row["patient"] in EXPECTED_SHARED):
            continue
        seen_identity.add(key)
        primary.append(row)
    if missed:
        raise RuntimeError(f"verified 001187/000673 overlaps missed: {missed}")
    report = {
        "schema_version": "1.0.0",
        "data_root": str(args.data_root),
        "n_records": len(records),
        "n_primary_records": len(primary),
        "verified_001187_000673_shared_patients": sorted(shared_1187_673),
        "unexpected_001187_000673_shared_patients": unexpected,
        "overlap_groups": overlaps,
        "canonical_view_rule": "Prefer 001187 for a verified 001187/000673 shared recording; retain 000673 as a linked sensitivity view.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "canonical_recording_registry.json").write_text(canonical_json(records))
    (args.output_dir / "canonical_primary_records.json").write_text(canonical_json(primary))
    (args.output_dir / "dataset_overlap_report.json").write_text(canonical_json(report))
    print(f"audited {len(records)} NWB recordings; {len(overlaps)} linked native-identifier groups")


if __name__ == "__main__":
    main()
