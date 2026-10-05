"""Writes the superseding version of a finished benchmark artifact from a demixed-only refit.

Each artifact kind names where its demixed rows sit and which summary blocks its producer computes from them;
the merge itself is `demixed_refit_merge.merge_refit`.

Run:
    python scripts/merge_demixed_refit.py KIND --delivered FINISHED.json --refit REFIT.json --output SUPERSEDING.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    if str(ROOT / _subdir) not in sys.path:
        sys.path.insert(0, str(ROOT / _subdir))

from demixed_refit_merge import DEMIXED, RefusedMerge, list_rows, merge_refit, nested_rows  # noqa: E402
from provenance import canonical_json, code_identity  # noqa: E402

SNAPSHOT_ROOT = ROOT.parent if ROOT.parent.name == "benchmark_snapshots" else ROOT.parent / "benchmark_snapshots"
UNIT_LOAD1 = "human_field_potential_benchmark_2026_09_30/results/human_representation_benchmark_seeds/seed_0.json"
UNIT_LOAD3 = "human_field_potential_benchmark_2026_09_30/results/human_representation_benchmark_seeds/seed_0_load3.json"


def rng() -> np.random.Generator:
    return np.random.default_rng(1)


def without(scope: dict, *names: str) -> dict:
    return {k: v for k, v in scope.items() if k not in names}


def patient_map(artifact: dict) -> dict:
    out = {}
    for row in artifact["sessions"]:
        if "patient" in row and "session" in row:
            out[f"{row['patient']}_{row['session']}"] = row["patient"]
            out.setdefault(row["session"], row["patient"])
    return out


# --- single-unit decoding benchmark ---------------------------------------------------------------

def decoding_adapter() -> dict:
    import run_human_representation_benchmark as benchmark
    from info_decoding import render_summary

    return dict(
        locators={"records": list_rows(("records",), ("corpus", "session", "level", "decoder"))},
        settings=lambda a: {"version": a["version"], **without(a["scope"], "candidates", "sessions_limit")},
        relaxed=(),
        recompute=lambda a, new: {("improvement_summary",): benchmark.improvement_summary(
            a["records"], {f"{s['patient']}_{s['session']}": s["patient"] for s in a["sessions"] if "patient" in s}, rng())},
        markdown=lambda a: render_summary(a["records"], True, a["scope"]["time_step"]),
    )


# --- held-out-neuron co-smoothing ----------------------------------------------------------------

def cosmoothing_adapter() -> dict:
    import run_human_cosmoothing_benchmark as cosmoothing

    return dict(
        locators={"records": list_rows(("records",), ("region", "session", "level"))},
        settings=lambda a: {"version": a["version"], **without(a["scope"], "candidates", "sessions_limit")},
        relaxed=(),
        recompute=lambda a, new: {("paired_summary",): cosmoothing.paired_summary(a["records"], patient_map(a), rng())},
        markdown=lambda a: cosmoothing.render_summary(a["records"], True),
    )


# --- field potentials ----------------------------------------------------------------------------

def field_adapter(args, calibration_only: bool) -> dict:
    import run_human_field_potential_benchmark as field

    def recompute(a, new):
        if calibration_only:
            return {("calibration_summary",): field.calibration_summary(a["cross_session_calibration"], rng())}
        field.UNIT_BENCHMARK_RESULTS = {
            "load1_maintenance": Path(args.superseding_unit_load1 if new else args.delivered_unit_load1),
            "load3_first_item_maintenance": Path(args.superseding_unit_load3 if new else args.delivered_unit_load3),
        }
        patients = patient_map(a)
        content = field.content_summary(a["content_records"], patients, rng())
        reconstruction = field.reconstruction_summary(a["reconstruction_records"], patients, rng())
        return {
            ("content_summary",): content, ("reconstruction_summary",): reconstruction,
            ("co_primary_summary",): field.co_primary_summary(content, reconstruction),
            ("unit_benchmark_comparison",): field.unit_benchmark_comparison(a["content_records"]),
        }

    calibration_key = ("patient", "session_a", "session_b", "mode", "metric", "decoder", "region")
    locators = {}
    if calibration_only:
        locators["cross_session_calibration"] = list_rows(("cross_session_calibration",), calibration_key)
    else:
        locators["content_records"] = list_rows(
            ("content_records",), ("corpus", "region", "session", "level", "decoder"))
        locators["reconstruction_records"] = list_rows(
            ("reconstruction_records",), ("corpus", "region", "session", "level", "metric"))

    def markdown(a):
        if calibration_only:
            return ""
        return field.render_summary(a["sessions"], a["content_records"], a["reconstruction_records"],
                                    a["cross_session_calibration"], a["task_failures"], True, a["content_summary"],
                                    a["reconstruction_summary"])

    return dict(
        locators=locators,
        settings=lambda a: {"version": a["version"], "inference_mode": a["inference_mode"],
                            **without(a["scope"], "candidates")},
        relaxed=("corpora", "entries_limit", "entries_limit_000673"),
        recompute=recompute,
        markdown=None if calibration_only else markdown,
        notes=() if calibration_only else (
            "cross_session_calibration and calibration_summary are the finished file's own and were not refitted; "
            "the standalone calibration file holds the refitted calibration",),
    )


# --- read-out timing pilot -----------------------------------------------------------------------

def pilot_cells(artifact: dict):
    for entry in artifact["entries"]:
        e = entry["entry"]
        key = f"{e['region']}::{e['patient']}::{e['session']}"
        for split, info in entry.get("splits", {}).items():
            if DEMIXED in info.get("candidates", {}):
                yield ("pilot", key, split), info["candidates"], DEMIXED


def pilot_adapter(args) -> dict:
    import run_readout_timing_pilot as pilot

    def recompute(a, new):
        seed0 = json.loads(Path(args.superseding_seed0 if new else args.delivered_seed0).read_text())
        seed0_index = {(r["corpus"], r["session"], r["level"], r["candidate"], r["decoder"]): r
                       for r in seed0.get("records", []) if r.get("status") == "computed"}
        scores, consistency_input, entry_keys, patient_by_entry = {}, {}, [], {}
        for entry in a["entries"]:
            e = entry["entry"]
            key = f"{e['region']}::{e['patient']}::{e['session']}"
            entry_keys.append(key)
            patient_by_entry[key] = e["patient"]
            for split in pilot.SPLITS:
                cells = entry.get("splits", {}).get(split, {}).get("candidates", {})
                for candidate in pilot.CANDIDATE_KEYS:
                    record = cells.get(candidate)
                    if record is None:
                        continue
                    if record.get("status") == "computed":
                        scores[(key, split, candidate)] = record
                    if split == "stratified_folds" and candidate.removesuffix("__whole_trial") in pilot.CONSISTENCY_CANDIDATES:
                        consistency_input[(key, e["region"], f"{e['patient']}_{e['session']}", candidate)] = record
        return {("contrasts",): pilot.build_contrasts(scores, entry_keys, patient_by_entry),
                ("consistency_check_against_seed_0",): pilot.consistency_check(consistency_input, seed0_index)}

    return dict(
        locators={"entries": nested_rows(pilot_cells)},
        settings=lambda a: {"schema_version": a["schema_version"], "decoder": a["decoder"],
                            "n_permutations": a["n_permutations"], "time_step": a["time_step"],
                            "operating_rank": a["operating_rank"], "drawn_entries": a["drawn_entries"]},
        relaxed=("drawn_entries",), recompute=recompute, markdown=None,
    )


# --- read-out timing follow-up -------------------------------------------------------------------

def followup_cells(artifact: dict):
    for entry in artifact["entries"]:
        if entry.get("status") != "loaded":
            continue
        for score, cells in entry["cells"].items():
            for cid, record in cells.items():
                if record.get("representation") == DEMIXED:
                    yield ("followup", entry["level"], entry["region"], entry["patient"], entry["session"], score, cid), cells, cid


def followup_adapter(args) -> dict:
    import run_readout_timing_followup as followup

    def ordered(a):
        """Entries with each cell table in the producer's insertion order (the stored order is alphabetical)."""
        scope = a["scope"]
        splits = [followup.STRATIFIED] + [followup.chronological_name(f) for f in scope["train_fractions"]]
        plan = [(r, followup.WHOLE) for r in scope["references"]] + [
            (m, f) for m in scope["candidates"] for f in followup.INFERENCE_FORMS]
        out = []
        for entry in a["entries"]:
            if entry.get("status") != "loaded":
                out.append(entry)
                continue
            cells = {}
            for score, table in entry["cells"].items():
                cells[score] = {cid: table[cid] for split in splits for rep, inf in plan
                                if (cid := followup.cell_id(rep, inf, split)) in table}
            out.append({**entry, "cells": cells})
        return out

    def recompute(a, new):
        scope, entries = a["scope"], ordered(a)
        splits = [followup.STRATIFIED] + [followup.chronological_name(f) for f in scope["train_fractions"]]
        absolute, contrasts = followup.summarise(entries, tuple(scope["candidates"]), splits, tuple(scope["references"]))
        root = Path(args.superseding_root if new else args.delivered_root)
        reproduction = followup.reproduction(entries, root, scope["scores"], scope["seed"])
        blocks = {("absolute",): absolute, ("contrasts",): contrasts, ("accounting",): followup.accounting(a["entries"]),
                  ("timing",): followup.timing(a["entries"])}
        blocks.update({("reproduction", key): value for key, value in reproduction.items()})
        return blocks

    return dict(
        locators={"entries": nested_rows(followup_cells)},
        settings=lambda a: {"version": a["version"], **without(
            a["scope"], "candidates", "references", "wall_clock_s", "workers", "entries_limit")},
        relaxed=("levels", "scores"), recompute=recompute, markdown=followup.render_markdown,
    )


# --- memorandum decoding study -------------------------------------------------------------------

def study_adapter(partial: bool) -> dict:
    import run_memorandum_decoding_study as study

    sections = ("absolute", "latent_minus_native", "not_estimable")
    key_fields = {"absolute": ("level", "load", "region", "representation", "metric"),
                  "latent_minus_native": ("level", "load", "region", "representation", "metric"),
                  "not_estimable": ("level", "region", "representation")}

    def checks(delivered, refit):
        if not all(refit["refit"]["generator_state_reproduced"].values()):
            raise RefusedMerge("the refit could not reproduce the finished artifact's generator state")
        for name in ("decoder", "decoder_note", "unit_set_mode", "time_feature"):
            if canonical_json(delivered["latent_state"][name]) != canonical_json(refit["latent_state"][name]):
                raise RefusedMerge(f"latent_state.{name} differs between the two files")
        for name in () if partial else sections:
            others = lambda a: canonical_json([r for r in a["latent_state"][name] if r["representation"] != DEMIXED])
            if others(delivered) != others(refit):
                raise RefusedMerge(f"the refit changed rows of latent_state.{name} that are not demixed")

    return dict(
        locators={f"latent_state.{name}": list_rows(("latent_state", name), key_fields[name], flag="representation")
                  for name in sections},
        settings=lambda a: {"version": a["version"], "scope": a["scope"], "selection_rule": a["selection_rule"]},
        relaxed=(), recompute=lambda a, new: {}, markdown=study.render_summary, extra_checks=checks,
    )


# --- pooled latent memorandum decoding -----------------------------------------------------------

def pooled_latent_adapter(args) -> dict:
    import run_pooled_latent_memorandum_decoding as pooled

    delivered, refit = (json.loads(path.read_text()) for path in (args.delivered, args.refit))
    cell_key = ("group", "load", "region", "readout")
    by_cell = lambda a: {tuple(r[f] for f in cell_key): r for r in a["cells"] if r["representation"] == DEMIXED}
    old_rows, new_rows = by_cell(delivered), by_cell(refit)
    resized = sorted(k for k, row in new_rows.items() if k in old_rows and row["session_subset"] != old_rows[k]["session_subset"])

    def recompute(a, new):
        sessions, timing = a["representation_sessions"], a["fit_timing"]
        if new:
            sessions = {cell: {**entry, DEMIXED: refit["representation_sessions"][cell][DEMIXED]}
                        if cell in refit["representation_sessions"] else entry for cell, entry in sessions.items()}
            timing = timing if args.partial else {**timing, DEMIXED: refit["fit_timing"][DEMIXED]}
        return {("representation_sessions",): sessions, ("fit_timing",): timing}

    def checks(delivered, refit):
        covered = {k: v for k, v in delivered["admission"].items() if not args.partial or k in refit["admission"]}
        for name, before, after in (("session_accounting", delivered["session_accounting"], refit["session_accounting"]),
                                    ("admission", covered, refit["admission"])):
            if canonical_json(before) != canonical_json(after):
                raise RefusedMerge(f"{name} differs between the two files")
        if not args.partial and set(refit["representation_sessions"]) != set(delivered["representation_sessions"]):
            raise RefusedMerge("the refit does not cover every session-region-load cell")
        for key, row in new_rows.items():
            if key in old_rows and key not in resized and canonical_json(row["native_matched"]) != canonical_json(
                    old_rows[key]["native_matched"]):
                raise RefusedMerge(f"the native-matched scores of {key} differ from the finished run on the same sessions")

    return dict(
        locators={"cells": list_rows(("cells",), cell_key, flag="representation")},
        settings=lambda a: {"version": a["version"], "chosen_read_out": a["chosen_read_out"],
                            **without(a["scope"], "representations")},
        relaxed=("loads", "regions", "readouts"), recompute=recompute, markdown=pooled.render_summary,
        extra_checks=checks,
        notes=(f"{len(resized)} refitted rows are scored on a different session set than the finished run; "
               f"their native-matched scores were recomputed on that set: {[list(k) for k in resized]}",) if resized else (),
    )


KINDS = ("decoding", "cosmoothing", "field_potential", "field_potential_calibration", "readout_timing_pilot",
         "readout_timing_followup", "memorandum_decoding_study", "pooled_latent_memorandum_decoding")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Writes the superseding artifact from a demixed-only refit.")
    p.add_argument("kind", choices=KINDS)
    p.add_argument("--delivered", type=Path, required=True)
    p.add_argument("--refit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--partial", action="store_true",
                   help="the refit covers a subset of the demixed rows; only those are replaced")
    p.add_argument("--accept-current-summary-rule", action="store_true",
                   help="the finished file's summaries came from an older summary function; write the summaries "
                        "of the current one and record which blocks differ from the finished file")
    p.add_argument("--delivered-unit-load1", type=Path, default=SNAPSHOT_ROOT / UNIT_LOAD1)
    p.add_argument("--delivered-unit-load3", type=Path, default=SNAPSHOT_ROOT / UNIT_LOAD3)
    p.add_argument("--superseding-unit-load1", type=Path)
    p.add_argument("--superseding-unit-load3", type=Path)
    p.add_argument("--delivered-seed0", type=Path, default=SNAPSHOT_ROOT / UNIT_LOAD1)
    p.add_argument("--superseding-seed0", type=Path)
    p.add_argument("--delivered-root", type=Path, default=SNAPSHOT_ROOT,
                   help="directory holding the finished benchmark files the follow-up compares with")
    p.add_argument("--superseding-root", type=Path)
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    adapters = {
        "decoding": decoding_adapter, "cosmoothing": cosmoothing_adapter,
        "field_potential": lambda: field_adapter(args, False),
        "field_potential_calibration": lambda: field_adapter(args, True),
        "readout_timing_pilot": lambda: pilot_adapter(args),
        "readout_timing_followup": lambda: followup_adapter(args),
        "memorandum_decoding_study": lambda: study_adapter(args.partial),
        "pooled_latent_memorandum_decoding": lambda: pooled_latent_adapter(args),
    }
    needs = {"field_potential": ("superseding_unit_load1", "superseding_unit_load3"),
             "readout_timing_pilot": ("superseding_seed0",), "readout_timing_followup": ("superseding_root",)}
    for name in needs.get(args.kind, ()):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required for {args.kind}")
    adapter = adapters[args.kind]()
    account = merge_refit(
        args.delivered, args.refit, args.output, locators=adapter["locators"], settings=adapter["settings"],
        recompute=adapter["recompute"], code_identity=code_identity(ROOT, Path(__file__)), partial=args.partial,
        accept_summary_rule_change=args.accept_current_summary_rule, relaxed_when_partial=adapter["relaxed"], extra_checks=adapter.get("extra_checks"),
        notes=adapter.get("notes", ()),
        markdown=adapter["markdown"],
    )
    print(canonical_json({k: account[k] for k in ("refitted_row_count", "partial", "summary_entries_unchanged",
                                                    "summary_entries_changed", "summary_fields_added", "recomputed_blocks")}), end="")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
