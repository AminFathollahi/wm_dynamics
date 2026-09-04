#!/usr/bin/env python3
"""Fail if the machine-readable claim/evidence ledger is incomplete."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from provenance import validate_ledger  # noqa: E402


def validate_inference_diagnostics(artifact: object, location: str = "root") -> list[str]:
    """Reject identified failed fits and interval-free comparison verdicts."""
    errors: list[str] = []
    if isinstance(artifact, dict):
        if artifact.get("group_identified") is True:
            failed = (
                artifact.get("status") in {"nonconverged", "failed"}
                or artifact.get("optimizer_success") is False
                or float(artifact.get("negative_log_likelihood", 0.0)) >= 1e29
            )
            if failed:
                errors.append(f"{location}: group estimate is identified despite optimizer failure")
        if "entity_medians" in artifact and "interpretation" in artifact:
            inference = artifact.get("entity_median_inference")
            if not isinstance(inference, dict) or any(
                not isinstance(row, dict)
                or row.get("patient_or_session_bootstrap_interval_95") is None
                for row in inference.values()
            ):
                errors.append(f"{location}: comparison verdict lacks uncertainty on a deciding statistic")
        for key, value in artifact.items():
            errors.extend(validate_inference_diagnostics(value, f"{location}.{key}"))
    elif isinstance(artifact, list):
        for index, value in enumerate(artifact):
            errors.extend(validate_inference_diagnostics(value, f"{location}[{index}]"))
    return errors


def main() -> None:
    ledger_path = ROOT / "provenance" / "evidence_ledger.json"
    rows = json.loads(ledger_path.read_text())
    errors = validate_ledger(rows, ROOT)
    artifact_paths = sorted({row["artifact_path"] for row in rows if row.get("artifact_path")})
    for relative_path in artifact_paths:
        path = ROOT / relative_path
        if path.suffix == ".json" and path.exists():
            errors.extend(validate_inference_diagnostics(json.loads(path.read_text()), relative_path))
    if errors:
        raise SystemExit("Evidence ledger validation failed:\n- " + "\n- ".join(errors))
    print(f"Evidence ledger valid: {len(rows)} claims")


if __name__ == "__main__":
    main()
