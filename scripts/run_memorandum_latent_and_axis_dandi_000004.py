#!/usr/bin/env python3
"""Pooled latent-state decoding and memorandum axis decoding of the picture category in the DANDI 000004 new/old
recognition corpus.

Runs the pooled latent comparison (every representation, the demixed fit with the decoder projection) and the
memorandum axis study on this corpus as its own dataset group, with the read-out fixed by the memorandum decoding
study, and writes both sections to one artifact.

Run:
    WM_DYNAMICS_DATA_ROOT=<data root> python scripts/run_memorandum_latent_and_axis_dandi_000004.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    if str(ROOT / _subdir) not in sys.path:
        sys.path.insert(0, str(ROOT / _subdir))

import memorandum_decoding as md  # noqa: E402
import run_memorandum_axis_decoding as axis  # noqa: E402
import run_memorandum_decoding_dandi_000004 as recognition  # noqa: E402
import run_pooled_latent_memorandum_decoding as pooled  # noqa: E402
from provenance import canonical_json, code_identity  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "memorandum_latent_and_axis_dandi_000004.json"
SUMMARY_PATH = RESULTS / "memorandum_latent_and_axis_dandi_000004.md"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_memorandum_latent_and_axis_dandi_000004"
SCHEMA = "memorandum_latent_and_axis_dandi_000004_v1"
SHARED_SECTION_KEYS = ("session_accounting", "admission", "version", "status", "wall_clock_s")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pooled latent and memorandum axis decoding in DANDI 000004.")
    p.add_argument("--output", type=Path)
    p.add_argument("--summary", type=Path)
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--search-artifact", type=Path, default=pooled.SEARCH_ARTIFACT)
    p.add_argument("--max-sessions", type=int)
    p.add_argument("--long-pictures", action="store_true",
                   help="use only trials whose picture lasted at least as long as the encoding segment (secondary group)")
    p.add_argument("--n-resamples", type=int, default=20)
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--n-bootstrap-time-course", type=int, default=50)
    p.add_argument("--n-shuffles", type=int, default=100)
    p.add_argument("--n-shuffles-time-course", type=int, default=20)
    p.add_argument("--n-overlap-permutations", type=int, default=50)
    p.add_argument("--selection-permutations", type=int, default=md.SELECTION_PERMUTATIONS)
    p.add_argument("--regions", nargs="+", choices=md.REGIONS, default=list(md.REGIONS))
    p.add_argument("--representations", nargs="+", choices=pooled.REPRESENTATIONS, default=list(pooled.REPRESENTATIONS))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p


def shared_arguments(args, directory: Path) -> list[str]:
    out = ["--search-artifact", str(args.search_artifact), "--checkpoint-dir", str(args.checkpoint_dir / directory.stem),
           "--output", str(directory), "--summary", str(directory.with_suffix(".md")), "--loads", "1",
           "--regions", *args.regions, "--workers", str(args.workers), "--seed", str(args.seed),
           "--selection-permutations", str(args.selection_permutations), "--n-resamples", str(args.n_resamples),
           "--n-bootstrap", str(args.n_bootstrap), "--n-bootstrap-time-course", str(args.n_bootstrap_time_course),
           "--n-shuffles", str(args.n_shuffles), "--n-shuffles-time-course", str(args.n_shuffles_time_course)]
    if args.max_sessions is not None:
        out += ["--max-sessions-per-group", str(args.max_sessions)]
    return out


def run_section(main, arguments: list[str]) -> None:
    sys.argv = [sys.argv[0], *arguments]
    main()


def section(artifact: dict, group: str) -> dict:
    out = {k: v for k, v in artifact.items() if k not in SHARED_SECTION_KEYS}
    out["task_note"] = recognition.TASK_NOTE
    for key, value in out.items():
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            out[key] = recognition.stamp(value, group)
    return out


def main() -> None:
    args = parser().parse_args()
    suffix = "_picture_at_least_encoding_segment" if args.long_pictures else ""
    args.output = args.output or OUTPUT_PATH.with_name(OUTPUT_PATH.stem + suffix + ".json")
    args.summary = args.summary or SUMMARY_PATH.with_name(SUMMARY_PATH.stem + suffix + ".md")
    args.checkpoint_dir = args.checkpoint_dir or CHECKPOINT_DIR.with_name(CHECKPOINT_DIR.name + suffix)
    started = time.time()
    group = recognition.LONG_GROUP if args.long_pictures else recognition.GROUP
    recognition.configure((group,))
    pooled.build_entries = lambda max_per_group: recognition.build_entries(max_per_group, False)
    identity = code_identity(ROOT, Path(__file__))
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pooled_path = args.checkpoint_dir / "pooled_latent.json"
    axis_path = args.checkpoint_dir / "memorandum_axis.json"
    run_section(pooled.main, shared_arguments(args, pooled_path) + [
        "--representations", *args.representations])
    run_section(axis.main, shared_arguments(args, axis_path) + [
        "--n-overlap-permutations", str(args.n_overlap_permutations), "--reference-artifact", str(pooled_path)])
    pooled_artifact, axis_artifact = json.loads(pooled_path.read_text()), json.loads(axis_path.read_text())
    for part in (pooled_artifact, axis_artifact):
        part["scope"]["datasets"] = ["dandi_000004"]
    accounting = pooled_artifact["session_accounting"]
    artifact = {
        "version": SCHEMA, "status": "complete" if pooled_artifact["status"] == axis_artifact["status"] == "complete"
        else "incomplete", "code_identity": identity, "dataset_group": group,
        "task_note": recognition.TASK_NOTE,
        "scope": {
            "dataset": "dandi_000004", "dataset_group": group, "task_note": recognition.TASK_NOTE,
            "load": recognition.LOAD, "condition_note": "one condition; the corpus has no memory-load manipulation",
            "maintenance_window_s": md.MAINTENANCE_S, "encoding_segment_s": md.ENCODING_SEGMENT_S,
            "pictures_restricted_to_at_least_the_encoding_segment": args.long_pictures,
            "maintenance_window_note": "first 0.5 s of the blank between picture offset and the question screen",
            "composition": "the pooled latent comparison and the memorandum axis study, each run unchanged on this "
                           "corpus with the read-out fixed by the memorandum decoding study; the axis study's "
                           "reproduction check compares its whole-window native rows with the pooled latent section "
                           "of this artifact",
            "pooled_latent": pooled_artifact["scope"], "memorandum_axis": axis_artifact["scope"],
            "max_sessions": args.max_sessions, "seed": args.seed},
        "session_accounting": accounting, "session_reconciliation": recognition.reconciliation(accounting),
        "admission": pooled_artifact["admission"], "pooled_latent": section(pooled_artifact, group),
        "memorandum_axis": section(axis_artifact, group), "wall_clock_s": time.time() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(artifact))
    args.summary.write_text("\n".join([
        "# Pooled latent and memorandum axis decoding, DANDI 000004", "", f"status: {artifact['status']}", "",
        recognition.TASK_NOTE, "", pooled.render_summary(pooled_artifact), axis.render_summary(axis_artifact)]))


if __name__ == "__main__":
    main()
