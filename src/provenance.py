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
import tempfile
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
_CHECKPOINT_STRING_KEY_PREFIX = "\u0000str:"


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

    JSON has no int dict keys, so a dict keyed by the string "2" and a dict keyed
    by the int 2 both become the JSON key "2" and are indistinguishable on
    reload. Every digit-only string key is escaped here to "\\u0000str:2", a form
    no producer in this codebase writes on its own (confirmed by grep), so
    restore_checkpoint can tell the two apart and only convert the unescaped
    (genuinely int-keyed) case back to int.
    """
    if isinstance(value, np.ndarray):
        return {
            _CHECKPOINT_ARRAY_TAG: True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": _json_safe(value.tolist()),
        }
    if isinstance(value, dict):
        return {
            (_CHECKPOINT_STRING_KEY_PREFIX + key if isinstance(key, str) and key.isdigit() else key):
                checkpoint_safe(item)
            for key, item in value.items()
        }
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

    JSON also has no integer dict keys: a dict keyed by int in memory comes back
    from json.loads keyed by its str(). Every digit-only key is restored to int
    here, so a caller that indexes a restored dict with the same ints it wrote
    (e.g. a per-chunk-size lookup table) does not have to convert at every call site.
    A digit-only key that checkpoint_safe escaped (it was a string in memory, e.g.
    str(level)) is un-escaped back to that same string instead, so it round-trips
    as the string it started as.
    """
    if isinstance(value, dict):
        if value.get(_CHECKPOINT_ARRAY_TAG) is True and "dtype" in value and "data" in value:
            dtype = np.dtype(value["dtype"])
            data = _replace_none_with_nan(value["data"]) if dtype.kind == "f" else value["data"]
            array = np.array(data, dtype=dtype)
            shape = tuple(value.get("shape", array.shape))
            return array.reshape(shape) if array.shape != shape and array.size == np.prod(shape) else array
        restored = {}
        for key, item in value.items():
            if isinstance(key, str) and key.startswith(_CHECKPOINT_STRING_KEY_PREFIX):
                restored_key = key[len(_CHECKPOINT_STRING_KEY_PREFIX):]
            elif isinstance(key, str) and key.isdigit():
                restored_key = int(key)
            else:
                restored_key = key
            restored[restored_key] = restore_checkpoint(item)
        return restored
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


def checkpoint_load(path: str | Path) -> dict | None:
    """Decoded checkpoint object, or None if the file is missing, unreadable or not a JSON object."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def checkpoint_store(path: str | Path, payload: dict) -> None:
    """Write a checkpoint atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


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


def git_commit(repo_root: str | Path, script_path: str | Path | None = None) -> str | dict | None:
    """The current HEAD sha, unchanged for every existing caller passing one argument.

    With ``script_path``, returns a dict instead: the sha, whether the working tree carries
    uncommitted changes (so a sha alone is not mistaken for identifying the exact code that ran),
    and the producing script's own sha256 (so an untracked or since-edited script's artifact still
    names the file that actually produced it, independent of git).
    """
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    if script_path is None:
        return sha
    try:
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return {
        "sha": sha,
        "working_tree_dirty": dirty,
        "producing_script_sha256": sha256_file(script_path),
    }


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


def canonical_patient_by_relative_path(provenance_dir: str | Path) -> dict[str, str]:
    """Map each primary recording's ``release/sub-N/file.nwb`` path to its canonical patient id.

    Built from canonical_primary_records.json: for a recording shared between releases (e.g.
    001187 and 000673), only the primary release's path is a key here, so filtering session paths
    against this mapping both resolves the shared-patient identity and drops the duplicate.
    """
    path = Path(provenance_dir) / "canonical_primary_records.json"
    if not path.exists():
        return {}
    records = json.loads(path.read_text())
    return {record["path"]: record["patient"] for record in records}


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
