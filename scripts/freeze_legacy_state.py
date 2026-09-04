#!/usr/bin/env python3
"""Create a read-only manifest of the pre-correction repository state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from provenance import canonical_json, environment_snapshot, sha256_file  # noqa: E402


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def headline_numbers(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"present": False}
    stats = json.loads(path.read_text())
    return {
        "present": True,
        "sha256": sha256_file(path),
        "top_level_keys": sorted(stats),
        "causal_macaque_pfc_microstimulation": stats.get("causal_macaque_pfc_microstimulation"),
        "closed_loop": stats.get("closed_loop"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "provenance" / "legacy_manifest.json")
    args = parser.parse_args()
    tracked = command("git", "ls-files")
    dirty = command("git", "status", "--short")
    file_hashes = {}
    for relative in ("PAPER_REPORT.tex", "README.md", "DATASET_ANALYSIS_MATRIX.md", "environment.yml"):
        path = ROOT / relative
        if path.exists():
            file_hashes[relative] = sha256_file(path)
    manifest = {
        "manifest_schema_version": "1.0.0",
        "purpose": "Read-only legacy snapshot before correction; not current evidence.",
        "git_commit": command("git", "rev-parse", "HEAD"),
        "dirty_inventory": dirty.splitlines(),
        "tracked_file_count": len(tracked.splitlines()),
        "environment": environment_snapshot(),
        "file_hashes": file_hashes,
        "legacy_results": headline_numbers(ROOT / "results" / "all_statistics.json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(manifest))
    print(args.output.relative_to(ROOT))


if __name__ == "__main__":
    main()
