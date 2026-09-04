"""Immutable analysis artifacts and evidence-ledger validation.

This module is intentionally independent of the legacy ``all_statistics.json``
file.  That file is retained as historical evidence only: new analyses write a
complete directory first and a deterministic aggregator may subsequently read
those directories.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "1.0.0"
VALID_STATUSES = {
    "current_confirmatory",
    "current_exploratory",
    "pipeline_provenance",
    "replication_or_informative_null",
    "superseded",
    "invalidated_do_not_interpret",
    "skipped",
    "nonidentified",
}
VALID_GATES = {"G0", "G1", "G2", "G3", "G4"}


def _json_safe(value: Any) -> Any:
    """Return strict-JSON data, representing failed/nonfinite estimates as null.

    Also converts numpy scalar types to native Python ones, since json.dump has
    no encoder for them and a nonfinite numpy float would otherwise reach the
    encoder unconverted and either raise or, with allow_nan=True, write an
    invalid NaN/Infinity token.
    """
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


_CHECKPOINT_ARRAY_TAG = "__provenance_ndarray__"


def checkpoint_safe(value: Any) -> Any:
    """JSON-safe encoding for fit-level checkpoint caches, tagging every numpy array
    with its dtype and shape so a later reload can rebuild the exact array instead
    of leaving a plain Python list in its place.

    A checkpoint written with plain _json_safe loses the distinction between an
    array and a list: reading it back with json.loads hands every array back as an
    untagged nested list, and a boolean mask read back that way indexes as integer
    positions rather than a boolean selection, silently. This function is a strict
    superset of _json_safe's output shape (a dict wraps each array instead of a bare
    nested list) so it is meant only for a checkpoint file an analysis script reads
    back within the same kind of run, never for a delivered result artifact:
    canonical_json and write_immutable_artifact keep calling _json_safe unchanged,
    so no existing artifact hash or byte layout is affected by this function's
    existence. See restore_checkpoint for the paired reader.
    """
    if isinstance(value, np.ndarray):
        return {
            _CHECKPOINT_ARRAY_TAG: True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _json_safe(value.tolist()),
        }
    if isinstance(value, dict):
        return {key: checkpoint_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [checkpoint_safe(item) for item in value]
    return _json_safe(value)


def _replace_none_with_nan(data: Any) -> Any:
    if isinstance(data, list):
        return [_replace_none_with_nan(item) for item in data]
    return np.nan if data is None else data


def _is_bool_leaf_list(value: list) -> bool:
    return len(value) > 0 and all(isinstance(v, bool) for v in value)


def _is_numeric_leaf_list(value: list) -> bool:
    return len(value) > 0 and all(
        v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) for v in value
    )


def restore_checkpoint(value: Any) -> Any:
    """Inverse of checkpoint_safe: rebuild every tagged array with its recorded
    dtype and shape.

    A checkpoint file written before this tagging existed holds untagged plain
    lists where an array used to be (the historical, still-broken _json_safe
    output). For that legacy shape only, this function falls back to a heuristic
    restoration -- a list whose leaves are all bool becomes a bool array, a list
    whose leaves are all int/float/None becomes a float array (None standing in
    for a nonfinite entry), and a list of already-restored equal-shaped arrays is
    stacked into one array -- the identical heuristic this project's first,
    hand-written checkpoint-array fix used, kept here so every caller gets it
    without repeating it. The heuristic cannot recover an integer array's exact
    dtype (it always reconstructs floats), which is why any newly written
    checkpoint should go through checkpoint_safe instead: a tagged array always
    comes back with its original dtype and shape exactly, an untagged one only
    approximately.
    """
    if isinstance(value, dict):
        if value.get(_CHECKPOINT_ARRAY_TAG) is True and "dtype" in value and "data" in value:
            dtype = np.dtype(value["dtype"])
            data = _replace_none_with_nan(value["data"]) if dtype.kind == "f" else value["data"]
            array = np.array(data, dtype=dtype)
            shape = tuple(value.get("shape", array.shape))
            return array.reshape(shape) if array.shape != shape and array.size == np.prod(shape) else array
        return {key: restore_checkpoint(item) for key, item in value.items()}
    if isinstance(value, list):
        if _is_bool_leaf_list(value):
            return np.array(value, dtype=bool)
        if _is_numeric_leaf_list(value):
            return np.array([np.nan if v is None else v for v in value], dtype=float)
        restored = [restore_checkpoint(item) for item in value]
        if restored and all(isinstance(item, np.ndarray) for item in restored):
            try:
                return np.array(restored)
            except ValueError:
                return restored
        return restored
    return value


def canonical_json(value: Any) -> str:
    """Serialize strict JSON deterministically for hashes and byte-stable outputs."""
    return json.dumps(
        _json_safe(value), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class ArtifactMetadata:
    analysis_id: str
    schema_version: str
    status: str
    gate: str
    dataset_view: str
    construct: str
    estimand: str
    independent_unit: str
    seed: int
    code_commit: str | None
    input_hashes: dict[str, str]
    configuration: dict[str, Any]
    completion_status: str = "complete"


def write_immutable_artifact(
    root: str | Path,
    metadata: ArtifactMetadata,
    result: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
    exclusions: list[dict[str, Any]] | None = None,
) -> Path:
    """Write an immutable content-addressed artifact directory.

    The identifier includes all substantive metadata/result bytes.  Repeating
    exactly the same run returns the existing path; a conflicting directory is
    an error rather than a silent overwrite.
    """
    if metadata.status not in VALID_STATUSES:
        raise ValueError(f"unknown evidence status: {metadata.status}")
    if metadata.gate not in VALID_GATES:
        raise ValueError(f"unknown gate: {metadata.gate}")
    payload = {
        "metadata": asdict(metadata),
        "result": result,
        "diagnostics": diagnostics or {},
        "exclusions": exclusions or [],
    }
    artifact_hash = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    destination = Path(root) / metadata.analysis_id / artifact_hash[:16]
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "artifact.json"
    rendered = canonical_json({"artifact_hash": artifact_hash, **payload})
    if output.exists() and output.read_text() != rendered:
        raise RuntimeError(f"immutable artifact collision at {destination}")
    output.write_text(rendered)
    return destination


def environment_snapshot() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
    }


def load_overlap_report(provenance_dir: str | Path) -> dict[str, Any] | None:
    """Load dataset_overlap_report.json, or None if the identity audit hasn't run yet."""
    path = Path(provenance_dir) / "dataset_overlap_report.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def linked_duplicate_000673_session_keys(overlap_report: dict[str, Any]) -> set[str]:
    """NWB filename stems of 000673 recordings that duplicate a canonical 001187 view.

    Per the audit's canonical_view_rule, 001187 is preferred for every verified
    001187/000673 overlap. Any cross-dataset pooling that already includes the
    001187 session must drop the matching 000673 key here first, or the same
    patient-session is counted twice.
    """
    stems = set()
    for group in overlap_report.get("overlap_groups", []):
        if {row["release"] for row in group} != {"001187", "000673"}:
            continue
        for row in group:
            if row["release"] == "000673":
                stems.add(Path(row["path"]).stem)
    return stems


REQUIRED_LEDGER_FIELDS = {
    "result_id", "claim_id", "analysis_id", "dataset_view", "construct", "eligibility_rule",
    "independent_unit", "estimand", "preprocessing_version", "inferential_method",
    "correction_family", "artifact_path", "artifact_hash", "manuscript_location",
    "status", "gate", "caveat", "model_prediction", "prediction_match_status",
}


def validate_ledger(rows: Iterable[dict[str, Any]], repo_root: str | Path) -> list[str]:
    """Return all ledger violations; callers fail rather than omit them."""
    root = Path(repo_root)
    errors: list[str] = []
    seen_claims: set[str] = set()
    for index, row in enumerate(rows):
        label = f"row {index}"
        missing = sorted(REQUIRED_LEDGER_FIELDS - set(row))
        if missing:
            errors.append(f"{label}: missing fields {', '.join(missing)}")
            continue
        claim_id = str(row["claim_id"])
        if claim_id in seen_claims:
            errors.append(f"{label}: duplicate claim_id {claim_id}")
        seen_claims.add(claim_id)
        if row["status"] not in VALID_STATUSES:
            errors.append(f"{label}: invalid status {row['status']}")
        if row["gate"] not in VALID_GATES:
            errors.append(f"{label}: invalid gate {row['gate']}")
        artifact = root / row["artifact_path"]
        if not artifact.is_file():
            errors.append(f"{label}: missing artifact {row['artifact_path']}")
        elif sha256_file(artifact) != row["artifact_hash"]:
            errors.append(f"{label}: artifact hash mismatch for {row['artifact_path']}")
        if row["status"].startswith("current_") and not row["manuscript_location"]:
            errors.append(f"{label}: current claim lacks manuscript location")
    return errors
