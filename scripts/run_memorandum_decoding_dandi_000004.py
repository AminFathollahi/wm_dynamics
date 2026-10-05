#!/usr/bin/env python3
"""Memorandum (picture category) decoding from human medial temporal single units in the DANDI 000004 new/old
recognition corpus, with the read-out fixed by the memorandum decoding study.

The corpus is its own dataset group and is never pooled with the working-memory corpora. Trials are the correct
recognition-phase trials; the label is the picture category; encoding is the segment from picture onset and
maintenance is the first part of the short blank between picture offset and the question screen. A secondary dataset
group repeats the analysis on the trials whose picture lasted at least as long as the encoding segment.

Run:
    WM_DYNAMICS_DATA_ROOT=<data root> python scripts/run_memorandum_decoding_dandi_000004.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    if str(ROOT / _subdir) not in sys.path:
        sys.path.insert(0, str(ROOT / _subdir))

import memorandum_decoding as md  # noqa: E402
import run_memorandum_decoding_study as study  # noqa: E402
import run_pooled_latent_memorandum_decoding as pooled  # noqa: E402
from provenance import canonical_json, code_identity  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "memorandum_decoding_dandi_000004.json"
SUMMARY_PATH = RESULTS / "memorandum_decoding_dandi_000004.md"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_memorandum_decoding_dandi_000004"
SCHEMA = "memorandum_decoding_dandi_000004_v1"

GROUP = "dandi_000004"
LONG_GROUP = "dandi_000004_picture_at_least_encoding_segment"
GROUP_RELEASE = {GROUP: md.RECOGNITION_RELEASE, LONG_GROUP: md.RECOGNITION_RELEASE + "_long"}
RELEASE_GROUP = {release: group for group, release in GROUP_RELEASE.items()}
LOAD = 1
LONG_LOAD_KEY = "1_picture_at_least_encoding_segment"
ENCODING_SEGMENT_S = 1.2
TASK_NOTE = (
    "New/old recognition task, not a report-later maintenance task. The maintenance window is the first 0.5 s of the "
    "short blank between picture offset (stim_off_time) and the question screen (delay1_time), a post-picture blank "
    "before the question, not a delay held until a later report."
)
ENTRIES = study.ENTRIES
_BUILT: dict = {}


def configure(groups=(GROUP, LONG_GROUP)) -> None:
    """Window geometry, dataset groups and condition set of this corpus, set in place on the shared modules."""
    md.configure_window_geometry(md.RECOGNITION_MAINTENANCE_S, ENCODING_SEGMENT_S)
    study.PSEUDO_GROUPS.clear()
    study.PSEUDO_GROUPS.update({group: (GROUP_RELEASE[group],) for group in groups})
    study.LOADS = pooled.LOADS = (LOAD,)


def entry_info(entry: dict | None, study_class_floor: bool) -> tuple[dict, bool]:
    info = {}
    if entry is None:
        return {"status": "no_correct_trials_or_no_admitted_units"}, False
    folds = md.feasible_fold_count(entry["labels"])
    classes, counts = np.unique(entry["labels"], return_counts=True)
    kept = folds is not None or (not study_class_floor and len(classes) >= 2)
    info.update(n_trials=len(entry["labels"]), n_units=len(entry["unit_region"]), n_classes=len(classes),
                trials_per_class=counts.tolist(), n_folds=folds,
                status="entry_built" if kept else "class_below_fold_count")
    return info, kept


def build_entries(max_admitted: int | None = None, study_class_floor: bool = False) -> list[dict]:
    """Loads sessions, builds one entry each (and a second from the trials whose picture lasted at least as long as
    the encoding segment) and returns the accounting of every file seen. Pseudo-populations have no per-session
    minimum, so an entry needs only two categories unless ``study_class_floor`` asks for the per-session
    cross-validation floor (every category in at least 5 trials) to apply to all entries. With a limit, loading stops
    once that many sessions can enter a pooled assembly (five categories, enough units in a region)."""
    cache_key = (max_admitted, study_class_floor)
    if cache_key in _BUILT:
        return _BUILT[cache_key]
    accounting, admitted = [], 0
    for row in md.recognition_session_manifest():
        record = {key: row[key] for key in ("release", "patient", "session", "status")}
        if max_admitted is not None and admitted >= max_admitted:
            record["status"] = "not_selected_by_session_limit"
        else:
            session, record["status"] = md.load_recognition_session(row["path"], row["patient"])
            if session is not None:
                record["trial_accounting"] = session["timing"]
                record["loads"] = {}
                variants = {str(LOAD): (session, row), LONG_LOAD_KEY: (
                    {**session, "release": GROUP_RELEASE[LONG_GROUP],
                     "correct": session["correct"] & (session["picture_s"] >= ENCODING_SEGMENT_S)},
                    {**row, "release": GROUP_RELEASE[LONG_GROUP]})}
                for label, (variant, variant_row) in variants.items():
                    entry = md.build_entry(variant, LOAD)
                    info, kept = entry_info(entry, study_class_floor)
                    info["unit_inclusion"] = md.unit_inclusion_counts(variant, LOAD)
                    if kept:
                        ENTRIES[study.entry_key(variant_row, LOAD)] = {
                            "entry": entry, "n_folds": info["n_folds"], "n_classes": info["n_classes"]}
                        if label == str(LOAD):
                            enough = max(len(md.region_columns(entry, r)) for r in md.REGIONS) >= md.MIN_UNITS_PER_REGION
                            admitted += int(info["n_classes"] == 5 and enough)
                    record["loads"][label] = info
        accounting.append(record)
    _BUILT[cache_key] = accounting
    return accounting


def blank_distribution(timing: list[dict]) -> dict:
    values = np.array([v for t in timing for v in t["blank_s_of_correct_trials_before_the_blank_rule"]])
    if not len(values):
        return {}
    return {"n_correct_trials_with_finite_times": int(len(values)),
            "quantiles_s": dict(zip(("0", "1", "5", "10", "25", "50", "75", "95", "100"),
                                    np.percentile(values, [0, 1, 5, 10, 25, 50, 75, 95, 100]).round(4).tolist())),
            "fraction_at_least_window": {str(w): float((values >= w).mean()) for w in (0.3, 0.4, 0.5, 0.52, 0.55)},
            "n_negative_blank": int((values < 0).sum()),
            "n_removed_by_window_rule": int((values < md.RECOGNITION_MAINTENANCE_S).sum()),
            "histogram_0p4_to_0p7_by_0p02": np.histogram(values, bins=np.arange(0.4, 0.7001, 0.02))[0].tolist()}


def reconciliation(accounting: list[dict]) -> dict:
    by_status = {s: sum(1 for a in accounting if a["status"] == s) for s in sorted({a["status"] for a in accounting})}
    entry_status = {}
    for label in (str(LOAD), LONG_LOAD_KEY):
        counts = {}
        for a in accounting:
            status = a.get("loads", {}).get(label, {}).get("status")
            if status:
                counts[status] = counts.get(status, 0) + 1
        entry_status[label] = counts
    timing = [a["trial_accounting"] for a in accounting if "trial_accounting" in a]
    removed = {}
    for t in timing:
        for step, n in t["removed_by_step"].items():
            removed[step] = removed.get(step, 0) + n
    return {"files_seen": len(accounting), "by_status": by_status, "entry_status": entry_status,
            "patients": len({a["patient"] for a in accounting}), "trials_removed_by_step": removed,
            "correct_recognition_trials": sum(t["n_correct_recognition_trials"] for t in timing),
            "correct_trials_with_picture_at_least_encoding_segment": sum(
                t["n_correct_trials_with_picture_at_least_encoding_segment"] for t in timing),
            "post_picture_blank_length": blank_distribution(timing)}


def group_of(unit: str) -> str:
    first = unit.split("|")[0]
    return first if first in GROUP_RELEASE else RELEASE_GROUP[first]


def stamp(rows: list[dict], group: str | None = None) -> list[dict]:
    return [{**row, "dataset_group": group or group_of(row["unit"]), "task_note": TASK_NOTE} for row in rows]


def read_out_tasks(chosen: dict, decoder: str, args) -> list[dict]:
    """The chosen read-out on every unit set, its radial-kernel supplement, and the follow-up curves of the study.
    The secondary group (pictures at least as long as the encoding segment) gets the read-outs and time courses only."""
    kind = chosen["time_feature"]
    tasks = study.grid_tasks(LOAD, (kind,), (decoder,), args.regions, tuple(study.UNIT_SETS), args.n_resamples,
                             args.n_shuffles, "readout")
    tasks += study.grid_tasks(LOAD, (kind,), ("radial_svm",), args.regions, (chosen["unit_set"],), args.n_resamples,
                              args.n_shuffles, "readout_radial")
    follow_up = [t for t in study.follow_up_tasks({**chosen, "decoder": decoder}, args.n_resamples, args)
                 if t["tag"] != "load3"]
    radial_courses = [{**t, "id": t["id"].replace("timecourse|", "timecourse_radial_svm|", 1), "decoders": ("radial_svm",)}
                      for t in follow_up if t["tag"] == "timecourse"]
    tasks += follow_up + radial_courses
    tasks = [t for t in tasks if t["level"] == "pseudo" or ENTRIES[t["keys"][0]]["n_folds"]]
    return [t for t in tasks if group_of(t["unit"]) == GROUP or t["tag"] in ("readout", "readout_radial", "timecourse")]


def render_summary(artifact: dict) -> str:
    lines = ["# Memorandum decoding, DANDI 000004", "", f"status: {artifact['status']}", "", artifact["task_note"], ""]
    chosen = artifact.get("chosen_read_out")
    if chosen:
        lines += [f"Read-out fixed by the memorandum decoding study: unit set {chosen['unit_set']}, time feature "
                  f"{chosen['time_feature']}, decoder {chosen['decoder']}.", ""]
    lines += ["| dataset group | unit set | region | decoder | time feature | balanced accuracy above null | "
              "interval of the median | area under curve above null | role |", "|---|---|---|---|---|---|---|---|---|"]
    for s in artifact.get("read_outs", {}).get("summary", []):
        low, high = s["balanced_accuracy"]["interval_95_of_median"]
        lines.append(f"| {s['dataset_group']} | {s['unit_set']} | {s['region']} | {s['decoder']} | {s['time_feature']} | "
                     f"{s['balanced_accuracy']['median_above_null']:.4f} | {low:.4f} to {high:.4f} | "
                     f"{s['auc']['median_above_null']:.4f} | {s['role']} |")
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Memorandum decoding in the DANDI 000004 recognition corpus.")
    p.add_argument("--output", type=Path)
    p.add_argument("--summary", type=Path)
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--search-artifact", type=Path, default=pooled.SEARCH_ARTIFACT)
    p.add_argument("--max-sessions", type=int)
    p.add_argument("--study-class-floor", action="store_true",
                   help="apply the per-session cross-validation floor (every category in at least 5 trials) to the "
                        "pseudo-population entries as well; writes artifacts with the suffix _study_class_floor")
    p.add_argument("--n-resamples", type=int, default=20)
    p.add_argument("--n-shuffles", type=int, default=100)
    p.add_argument("--selection-permutations", type=int, default=md.SELECTION_PERMUTATIONS)
    p.add_argument("--regions", nargs="+", choices=md.REGIONS, default=list(md.REGIONS))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.n_shuffles < 1 or args.n_resamples < 1:
        raise SystemExit("--n-shuffles and --n-resamples must be positive")
    suffix = "_study_class_floor" if args.study_class_floor else ""
    args.output = args.output or RESULTS / f"memorandum_decoding_dandi_000004{suffix}.json"
    args.summary = args.summary or RESULTS / f"memorandum_decoding_dandi_000004{suffix}.md"
    args.checkpoint_dir = args.checkpoint_dir or CHECKPOINT_DIR.with_name(CHECKPOINT_DIR.name + suffix)
    started = time.time()
    chosen, decoder, decoder_note = pooled.read_search(args.search_artifact)
    configure()
    md.SELECTION_PERMUTATIONS = args.selection_permutations
    identity = code_identity(ROOT, Path(__file__))
    study.CONFIG.update(
        checkpoint_dir=args.checkpoint_dir, n_shuffles=args.n_shuffles, n_resamples=args.n_resamples,
        identity=hashlib.sha256(canonical_json({
            "code": identity, "selection_permutations": args.selection_permutations, "n_shuffles": args.n_shuffles,
            "n_resamples": args.n_resamples, "seed": args.seed, "chosen": chosen, "decoder": decoder,
            "study_class_floor": args.study_class_floor, "max_sessions": args.max_sessions}).encode()).hexdigest())
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accounting = build_entries(args.max_sessions, args.study_class_floor)
    totals = study.inclusion_totals(accounting)
    totals["load1"]["sessions_tested"].pop("session_unit_sets_by_region")
    artifact = {
        "version": SCHEMA, "status": "running", "code_identity": identity,
        "checkpoint_identity": study.CONFIG["identity"], "dataset_group": GROUP, "task_note": TASK_NOTE,
        "search_artifact": str(args.search_artifact), "chosen_read_out": chosen, "decoder": decoder,
        "decoder_note": decoder_note,
        "scope": {
            "dataset": "dandi_000004", "dataset_groups": {
                GROUP: "all correct recognition trials with a long enough blank",
                LONG_GROUP: "the same trials restricted to pictures that lasted at least as long as the encoding segment "
                            "(secondary; read-outs and time courses only)"},
            "task_note": TASK_NOTE, "load": LOAD,
            "condition_note": "one condition; the corpus has no memory-load manipulation",
            "regions": args.regions, "task_rate_floor_hz": md.TASK_RATE_FLOOR_HZ,
            "task_span": "first trial start to last trial stop of the whole table (learning and recognition phases)",
            "trials": "recognition-phase trials with a truth label, a button response (values 31 to 36), finite picture "
                      "onset, picture offset and question-screen times, and a blank between picture offset and the "
                      "question screen of at least the maintenance window; correct = response side matches the truth label",
            "label": "stimCategory (five picture categories)",
            "picture_onset_field": "stim_on_time", "picture_offset_field": "stim_off_time",
            "maintenance_onset_field": "stim_off_time", "maintenance_limit_field": "delay1_time (question screen onset)",
            "maintenance_window_s": md.MAINTENANCE_S,
            "maintenance_window_choice": "the blank is nominally 0.5 s; the window is the largest multiple of the "
                                         "0.1 s step that keeps more than 90 percent of the correct recognition "
                                         "trials (distribution in session_reconciliation.post_picture_blank_length)",
            "encoding_segment_s": md.ENCODING_SEGMENT_S, "selection_window_s": list(md.SELECTION_WINDOW_S),
            "encoding_segment_note": "for trials with a 1.0 s picture the last 0.2 s of the encoding segment follows "
                                     "picture offset and the first 0.2 s of the blank follows the selection window",
            "selection_permutations": args.selection_permutations, "selection_level": md.SELECTION_LEVEL,
            "study_class_floor_applied_to_pseudo_populations": args.study_class_floor,
            "minimum_units_in_region_per_session_unit_set": md.MIN_UNITS_PER_REGION,
            "pseudo_population_scheme": {
                "outer_folds": md.PSEUDO_FOLDS, "pseudo_trials_per_category_training": md.PSEUDO_TRAIN_PER_CATEGORY,
                "pseudo_trials_per_category_test": md.PSEUDO_TEST_PER_CATEGORY,
                "unit_dropped_from_a_fold_when": "its session has fewer than 1 test trial or fewer than 2 training "
                                                 "trials in some category",
                "labels_shuffled": "within each session before splitting and assembly"},
            "nested_cross_validation": {"inner_folds": md.INNER_FOLDS, "grids": {
                k: [list(v) if isinstance(v, tuple) else v for v in g] for k, g in md.HYPER_GRIDS.items()}},
            "shuffled_label_hyperparameters": "chosen on the observed labels, reused in the shuffled-label runs",
            "n_resamples": args.n_resamples, "n_shuffles": args.n_shuffles, "curve_shuffles": study.CURVE_SHUFFLES,
            "curves": {"population_size": list(study.POPULATION_SIZES), "training_trials": list(study.TRAINING_TRIALS),
                       "time_features": list(study.CURVE_KINDS)},
            "left_out": {"load_three_first_item": "the corpus has one condition"},
            "seed": args.seed, "max_sessions": args.max_sessions},
        "session_accounting": accounting, "session_reconciliation": reconciliation(accounting),
        "unit_inclusion_totals": totals}

    def flush(**extra):
        artifact.update(extra)
        artifact["wall_clock_s"] = time.time() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_json(artifact))
        args.summary.write_text(render_summary(artifact))

    flush()
    tasks = read_out_tasks(chosen, decoder, args)
    study.execute(tasks, args.workers, "read-outs")
    readout_tasks = [t for t in tasks if t["tag"] in ("readout", "readout_radial")]
    flush(session_unit_sets_run={g: {r: sum(1 for t in readout_tasks if t["level"] == "session" and t["region"] == r
                                           and t["unit_set"] == "session_all_units" and group_of(t["unit"]) == g)
                                     for r in args.regions} for g in GROUP_RELEASE})
    rng = np.random.default_rng(args.seed)
    rows = study.read_out_rows(readout_tasks)
    summary = []
    for group in GROUP_RELEASE:
        members = [r for r in rows if group_of(r["unit"]) == group]
        for s in study.summarise_read_outs(members, rng):
            s["role"] = ("chosen" if (s["unit_set"], s["time_feature"], s["decoder"]) ==
                         (chosen["unit_set"], chosen["time_feature"], decoder) else "supplementary")
            summary.append({**s, "dataset_group": group, "task_note": TASK_NOTE})
    follow_up = [t for t in tasks if t["tag"] not in ("readout", "readout_radial")]
    flush(read_outs={"summary": summary, "per_unit_rows": stamp(rows)}, curves={
        **study.curve_summary(follow_up), "dataset_groups": list(GROUP_RELEASE), "task_note": TASK_NOTE},
          time_course=stamp(study.time_course(follow_up, rng)), status="complete")


if __name__ == "__main__":
    main()
