"""Causal-inference and chronological-split follow-up of the single-unit representation benchmark.

The benchmark infers every latent from the whole trial and scores it on stratified cross-validation
folds. A closed-loop read-out has only bins 1..t at bin t and is calibrated on earlier trials and used
on later ones. For every region-session entry of the benchmark, each candidate representation is
therefore scored in four cells: whole-trial or causal inference, crossed with stratified folds or a
chronological split (the first fraction of trials fits the representation and the read-out, the rest
is scored). Principal components and demixed principal components use only bin t at bin t, so they
are scored whole-trial under both splits.

Two scores share the cells: same-time area under the curve above the shuffled-label null (linear
decoder) and held-out-neuron bits per spike (latent from held-in neurons, read-out fitted on the
training trials only). The comparisons are patient-clustered, as in the benchmark.

Run:
    python scripts/run_readout_timing_followup.py
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
import multiprocessing
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    _path = str(ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import info_decoding  # noqa: E402
import run_human_cosmoothing_benchmark as cosmoothing  # noqa: E402
import run_human_representation_benchmark as benchmark  # noqa: E402
import run_info_benchmark as benchmark_core  # noqa: E402
import run_readout_timing_pilot as pilot  # noqa: E402
from build_representation_benchmark_co_primary import estimate  # noqa: E402
from info_decoding import CTG_STEP, OPERATING_RANK, _seeded_folds, atomic_write  # noqa: E402
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from run_info_benchmark import block_identity, implementation_identity  # noqa: E402
from statistics import stable_seed  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "readout_timing_followup.json"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_readout_timing_followup"
SNAPSHOT_ROOT = ROOT.parent if ROOT.parent.name == "benchmark_snapshots" else ROOT.parent / "benchmark_snapshots"

LEVELS = tuple(name for name, _ in benchmark.LEVELS)
REFERENCES = pilot.REFERENCE_MODELS
CANDIDATE_CHOICES = pilot.FUTURE_MODELS
SCORES = ("decoding", "cosmoothing")
DEFAULT_TRAIN_FRACTIONS = (0.667, 0.75)
DEFAULT_N_PERM = 100
DECODER = pilot.DECODER
N_SPLITS = pilot.N_SPLITS

STRATIFIED = "stratified_folds"
WHOLE, CAUSAL = "whole_trial", "causal"
INFERENCE_FORMS = (WHOLE, CAUSAL)
POOLED = "regions_pooled_by_patient"
REGION_CELLS = tuple(benchmark.REGIONS) + (POOLED,)

DELIVERED_FILES = {
    "decoding": {
        "load1_maintenance": "human_field_potential_benchmark_2026_09_30/results/human_representation_benchmark_seeds/seed_0.json",
        "load3_first_item_maintenance": "human_field_potential_benchmark_2026_09_30/results/human_representation_benchmark_seeds/seed_0_load3.json"},
    "cosmoothing": {
        "load1_maintenance": "human_cosmoothing_benchmark_2026_09_28/results/human_cosmoothing_benchmark/human_cosmoothing_benchmark_load1.json",
        "load3_first_item_maintenance": "human_cosmoothing_benchmark_2026_09_28/results/human_cosmoothing_benchmark/human_cosmoothing_benchmark_load3.json"},
}
EXACT_BOTH_SCORES = ("principal_components", "gaussian_process_factor_analysis")

STATEMENT = (
    "Secondary analyses of representations that were selected on the normal form (whole-trial "
    "inference, stratified five-fold cross-validation). No decision rule is attached to any quantity "
    "here. Every row reports a mean over patients of the patient-averaged entry values, a 95% interval "
    "from resampling patients, the median over patients with its own 95% interval from resampling "
    "patients, a two-sided sign-flip p over patients, a minimum detectable paired "
    "difference, and entry and patient counts. The inference form and the split of every cell are named "
    "in the cell and in the row."
)


def chronological_name(fraction: float) -> str:
    return f"chronological_{fraction:g}"


def cell_id(representation: str, inference: str, split: str) -> str:
    return f"{representation}|{inference}|{split}"


def chronological_split(labels: np.ndarray, train_fraction: float):
    """First `train_fraction` of the trials in recorded order calibrate, the rest test. The decoder
    needs every class in both sets, so a set missing a class cannot be scored."""
    n = len(labels)
    calibration, test = np.arange(int(round(n * train_fraction))), np.arange(int(round(n * train_fraction)), n)
    classes = np.unique(labels)
    test_classes, test_counts = np.unique(labels[test], return_counts=True)
    calibration_classes, calibration_counts = np.unique(labels[calibration], return_counts=True)
    info = {"train_fraction": train_fraction, "n_calibration": int(len(calibration)), "n_test": int(len(test)),
            "calibration_class_counts": {str(c): int(k) for c, k in zip(calibration_classes, calibration_counts)},
            "test_class_counts": {str(c): int(k) for c, k in zip(test_classes, test_counts)}}
    if len(test_classes) < 2:
        return None, {**info, "reason": "test set contains fewer than two classes"}
    if not (np.array_equal(calibration_classes, classes) and np.array_equal(test_classes, classes)):
        return None, {**info, "reason": "calibration and test sets do not contain the same classes"}
    return ((calibration, test),), info


def build_splits(region: str, session_key: str, level: str, labels: np.ndarray, fractions, seed: int) -> dict:
    stratified = _seeded_folds(
        labels, N_SPLITS, stable_seed(f"info_benchmark|seed{seed}|{region}|{session_key}|{level}|folds"))
    splits = {STRATIFIED: (stratified, {} if stratified is not None
                           else {"reason": "fewer than two trials in one or more classes"})}
    for fraction in fractions:
        splits[chronological_name(fraction)] = chronological_split(labels, fraction)
    return splits


def same_time_split_auc(train_activity, train_labels, test_activity, test_labels, *, decoder="linear", seed=0):
    """Same-time scores on the diagonal of a time-by-time matrix whose other entries are NaN: the
    quantity scored here is the diagonal, and the full matrix costs one decoder evaluation per pair
    of bins more than the diagonal for the same numbers."""
    inputs = info_decoding._split_inputs(train_activity, train_labels, test_activity, test_labels)
    diagonal = info_decoding._split_score(*inputs, decoder, seed, True)
    scores = np.full((len(diagonal), len(diagonal)), np.nan)
    scores[np.diag_indices_from(scores)] = diagonal
    return scores


def use_same_time_scoring() -> None:
    """Routes the benchmark's observed and shuffled-label scoring through the diagonal-only scorer."""
    original = info_decoding._decoder_api
    info_decoding._decoder_api = lambda: SimpleNamespace(**{**vars(original()), "split_auc": same_time_split_auc})
    pilot.split_auc = same_time_split_auc


def estimate_values(values: list[float], patients: list[str]) -> dict:
    if not values:
        return {"status": "not_estimable", "n_entries": 0, "n_patients": 0, "mean": None, "ci_95": None,
                "sign_flip_p": None, "minimum_detectable_difference": None, "reason": "no entries"}
    return estimate(values, patients)


# Scoring one cell ----------------------------------------------------------------------------------

_GPU_LOCK = None


def _causal_fit(model: str, gpu_lock):
    fit = pilot.FUTURE_MODEL_FITS[model]

    def causal_fit(train, test, k, rng, is_spiking, bin_ms):
        return fit(train, test, k, rng, bin_ms, True, None, gpu_lock)
    return causal_fit


def _init_worker(gpu_lock, models) -> None:
    """Registers each causal form under its own name in the fit table that the held-out-neuron
    scorer resolves candidates through."""
    global _GPU_LOCK
    _GPU_LOCK = gpu_lock
    cosmoothing._init_worker(gpu_lock)
    benchmark_core.REPRESENTATION_FITS = {
        **benchmark_core.REPRESENTATION_FITS,
        **{f"{model}__{CAUSAL}": _causal_fit(model, gpu_lock) for model in models},
    }


def _decoding_record(record: dict) -> dict:
    keep = ("status", "reason", "k_used", "n_splits", "inference_retried_bins", "inference_carried_bins")
    out = {k: record[k] for k in keep if k in record}
    if "same_time" in record:
        out["same_time"] = record["same_time"]
    return out


def score_decoding(entry: dict, activity, labels, folds, split: str, representation: str, inference: str,
                   n_perm: int, checkpoint_dir: Path, identity: dict) -> dict:
    key = representation if representation in REFERENCES else f"{representation}__{inference}"
    record = pilot.evaluate_cell(
        entry["region"], entry["session_key"], entry["level"], activity, labels, folds,
        "chronological" if split != STRATIFIED else STRATIFIED, key, n_perm, checkpoint_dir, identity, _GPU_LOCK)
    return _decoding_record(record)


def held_out_neurons(entry: dict, n_neurons: int, n_folds: int) -> list:
    return [cosmoothing.neuron_holdout(
        n_neurons, stable_seed(f"cosmoothing|{entry['region']}|{entry['session_key']}|{entry['level']}|fold{i}|neurons"))
        for i in range(n_folds)]


def score_cosmoothing(entry: dict, activity, labels, folds, split: str, representation: str, inference: str,
                      checkpoint_dir: Path, identity: dict) -> dict:
    neurons = held_out_neurons(entry, activity.shape[2], len(folds))
    if any(len(held_in) < 2 for held_in, _ in neurons):
        return {"status": "excluded", "reason": "fewer than 2 held-in neurons in a fold"}
    candidate = representation if inference == WHOLE else f"{representation}__{CAUSAL}"
    record = cosmoothing.evaluate_candidate(
        entry["region"], entry["session_key"], entry["level"], activity, labels, folds, neurons, candidate,
        identity, checkpoint_dir / f"cosmoothing_{split}")
    keep = ("status", "reason", "bits_per_spike", "bits_per_spike_training_null", "n_spikes", "n_folds", "k_used")
    return {k: record[k] for k in keep if k in record}


def evaluate_entry(entry: dict, activity: np.ndarray, labels: np.ndarray, config: dict) -> dict:
    classes, counts = np.unique(labels, return_counts=True)
    out = {**entry, "status": "loaded", "n_trials": int(activity.shape[0]), "n_units": int(activity.shape[2]),
           "class_counts": {str(c): int(k) for c, k in zip(classes, counts)}, "splits": {},
           "cells": {score: {} for score in config["scores"]}}
    splits = build_splits(entry["region"], entry["session_key"], entry["level"], labels,
                          config["fractions"], config["seed"])
    started = time.perf_counter()
    for split, (folds, info) in splits.items():
        out["splits"][split] = {"status": "estimable" if folds is not None else "not_estimable", **info}
        identity = block_identity(activity, labels, folds, config["runtime"]) if folds is not None else None
        plan = [(r, WHOLE) for r in config["references"]] + [(m, f) for m in config["models"] for f in INFERENCE_FORMS]
        for representation, inference in plan:
            for score in config["scores"]:
                cid = cell_id(representation, inference, split)
                if folds is None:
                    record = {"status": "not_estimable", **info}
                else:
                    cell_started = time.perf_counter()
                    if score == "decoding":
                        record = score_decoding(entry, activity, labels, folds, split, representation, inference,
                                                config["n_perm"], config["checkpoint_dir"], identity)
                    else:
                        record = score_cosmoothing(entry, activity, labels, folds, split, representation,
                                                   inference, config["checkpoint_dir"], identity)
                    record["wall_seconds"] = float(time.perf_counter() - cell_started)
                    print(f"readout_timing_followup: {entry['level']} {entry['region']} {entry['session_key']} "
                          f"{score} {cid} status={record.get('status')} seconds={record['wall_seconds']:.1f}",
                          flush=True)
                out["cells"][score][cid] = {"representation": representation, "inference": inference,
                                            "split": split, **record}
    out["wall_seconds"] = float(time.perf_counter() - started)
    return out


def process_session(row: dict, level: str, target_load: int, config: dict, max_entries: int | None) -> list[dict]:
    base = {"level": level, "patient": row["patient"], "session": row["session"], "release": row["release"],
            "session_key": f"{row['patient']}_{row['session']}"}
    session_data = benchmark.load_session_for_load(row["path"], row["release"], target_load)
    if session_data is None:
        return [{**base, "status": "not_loaded"}]
    entries, n_loaded = [], 0
    for region in benchmark.REGIONS:
        if max_entries is not None and n_loaded >= max_entries:
            break
        cell = benchmark.build_cell(session_data, region)
        if cell is None:
            entries.append({**base, "region": region, "status": "insufficient_units"})
            continue
        entries.append(evaluate_entry({**base, "region": region}, cell[0], cell[1], config))
        n_loaded += 1
    return entries


# Summaries -------------------------------------------------------------------------------------------

def entry_id(entry: dict) -> str:
    return f"{entry['region']}::{entry['patient']}::{entry['session']}"


def score_value(score: str, record: dict) -> float | None:
    if record.get("status") != "computed":
        return None
    if score == "decoding":
        return record["same_time"].get("auc_above_null")
    return record.get("bits_per_spike")


def collect_values(entries: list[dict]) -> dict:
    """{(score, level): {cell_id: {entry_id: (patient, region, value)}}} over computed cells."""
    values: dict = {}
    for entry in entries:
        if entry["status"] != "loaded":
            continue
        for score, cells in entry["cells"].items():
            target = values.setdefault((score, entry["level"]), {})
            for cid, record in cells.items():
                value = score_value(score, record)
                if value is not None:
                    target.setdefault(cid, {})[entry_id(entry)] = (entry["patient"], entry["region"], float(value))
    return values


def gather(cells: dict, minuend: str, subtrahend: str | None, region_cell: str) -> tuple[list, list]:
    values, patients = [], []
    for eid, (patient, region, value) in cells.get(minuend, {}).items():
        if region_cell != POOLED and region != region_cell:
            continue
        if subtrahend is not None:
            if eid not in cells.get(subtrahend, {}):
                continue
            value -= cells[subtrahend][eid][2]
        values.append(value)
        patients.append(patient)
    return values, patients


def cell_label(representation: str, inference: str, split: str) -> dict:
    return {"representation": representation, "inference": inference, "split": split}


def contrast_specs(models, splits, references=REFERENCES) -> list[dict]:
    chronological = [s for s in splits if s != STRATIFIED]
    specs = []
    for model in models:
        for split in splits:
            for inference in INFERENCE_FORMS:
                for reference in references:
                    specs.append({"kind": "candidate_minus_reference",
                                  "minuend": cell_label(model, inference, split),
                                  "subtrahend": cell_label(reference, WHOLE, split)})
            specs.append({"kind": "causal_minus_whole_trial", "minuend": cell_label(model, CAUSAL, split),
                          "subtrahend": cell_label(model, WHOLE, split)})
        for split in chronological:
            for inference in INFERENCE_FORMS:
                specs.append({"kind": "chronological_minus_stratified",
                              "minuend": cell_label(model, inference, split),
                              "subtrahend": cell_label(model, inference, STRATIFIED)})
    for reference in references:
        for split in chronological:
            specs.append({"kind": "chronological_minus_stratified",
                          "minuend": cell_label(reference, WHOLE, split),
                          "subtrahend": cell_label(reference, WHOLE, STRATIFIED)})
    return specs


def summarise(entries: list[dict], models, splits, references=REFERENCES) -> tuple[list, list]:
    absolute, contrasts = [], []
    specs = contrast_specs(models, splits, references)
    for (score, level), cells in sorted(collect_values(entries).items()):
        for region_cell in REGION_CELLS:
            for cid in sorted(cells):
                representation, inference, split = cid.split("|")
                absolute.append({"score": score, "level": level, "region_cell": region_cell,
                                 "cell": cell_label(representation, inference, split),
                                 **estimate_values(*gather(cells, cid, None, region_cell))})
            for spec in specs:
                a, b = spec["minuend"], spec["subtrahend"]
                contrasts.append({"score": score, "level": level, "region_cell": region_cell, **spec,
                                  **estimate_values(*gather(cells, cell_id(*a.values()), cell_id(*b.values()),
                                                            region_cell))})
    return absolute, contrasts


def accounting(entries: list[dict]) -> list[dict]:
    rows = []
    for level in LEVELS:
        subset = [e for e in entries if e["level"] == level]
        if not subset:
            continue
        cells: dict = {}
        for entry in subset:
            for score, records in entry.get("cells", {}).items():
                for cid, record in records.items():
                    cells.setdefault(f"{score}|{cid}", Counter())[record.get("status")] += 1
        rows.append({
            "level": level,
            "sessions_seen": len({e["session_key"] for e in subset}),
            "entries_by_status": dict(Counter(e["status"] for e in subset)),
            "cells_by_status": {k: dict(v) for k, v in sorted(cells.items())},
        })
    return rows


def timing(entries: list[dict]) -> dict:
    seconds: dict = {}
    for entry in entries:
        for score, records in entry.get("cells", {}).items():
            for cid, record in records.items():
                if "wall_seconds" in record:
                    seconds.setdefault(f"{score}|{cid}", []).append(record["wall_seconds"])
    return {k: {"mean_seconds": float(np.mean(v)), "n_cells": len(v)} for k, v in sorted(seconds.items())}


# Reproduction of the delivered benchmark values by the whole-trial stratified cell ------------------------

def load_delivered(delivered_root: Path, score: str, level: str) -> dict | None:
    path = Path(delivered_root) / DELIVERED_FILES[score][level]
    if not path.exists():
        return None
    index = {}
    for record in json.loads(path.read_text()).get("records", []):
        if record.get("status") != "computed":
            continue
        if score == "decoding":
            if record.get("decoder") == DECODER:
                index[(record["corpus"], record["session"], record["candidate"])] = record["same_time"]["auc"]
        else:
            index[(record["region"], record["session"], record["candidate"])] = record["bits_per_spike"]
    return index


def reproduction(entries: list[dict], delivered_root: Path, scores, seed: int) -> dict:
    if seed != 0:
        return {"status": "not_compared", "reason": "delivered values were produced with fold seed 0"}
    rows, missing = [], []
    for score in scores:
        for level in LEVELS:
            index = load_delivered(delivered_root, score, level)
            if index is None:
                missing.append({"score": score, "level": level})
                continue
            for entry in entries:
                if entry["status"] != "loaded" or entry["level"] != level:
                    continue
                for cid, record in entry["cells"][score].items():
                    if record["split"] != STRATIFIED or record["inference"] != WHOLE:
                        continue
                    ours = score_value(score, record) if score == "cosmoothing" else (
                        record["same_time"]["auc"] if record.get("status") == "computed" else None)
                    delivered = index.get((entry["region"], entry["session_key"], record["representation"]))
                    if ours is None or delivered is None:
                        continue
                    exact = (record["representation"] in EXACT_BOTH_SCORES
                             or (score == "cosmoothing" and record["representation"] == "demixed_principal_components"))
                    rows.append({"score": score, "level": level, "entry": entry_id(entry),
                                 "representation": record["representation"], "followup": ours,
                                 "delivered": delivered, "difference": ours - delivered,
                                 "expected_exact": bool(exact)})
    exact_rows = [abs(r["difference"]) for r in rows if r["expected_exact"]]
    return {"status": "compared", "quantity": "decoding: same-time area under the curve; "
            "cosmoothing: bits per spike", "delivered_files_missing": missing, "rows": rows,
            "max_abs_difference_expected_exact": float(max(exact_rows)) if exact_rows else None,
            "n_expected_exact": len(exact_rows),
            "max_abs_difference_all": float(max(abs(r["difference"]) for r in rows)) if rows else None}


# Output ---------------------------------------------------------------------------------------------------

def _fmt(row: dict) -> str:
    if row["status"] != "estimable":
        return f"not_estimable (n_entries={row['n_entries']}, n_patients={row['n_patients']})"
    lo, hi = row["ci_95"]
    mdd = row["minimum_detectable_difference"]
    median = row.get("median")
    median_text = "" if median is None else (
        f"median {median:.4f} [{row['median_ci_95'][0]:.4f}, {row['median_ci_95'][1]:.4f}] ")
    return (f"{row['mean']:.4f} [{lo:.4f}, {hi:.4f}] {median_text}p={row['sign_flip_p']:.4f} "
            f"mdd={'NA' if mdd is None else f'{mdd:.4f}'} entries={row['n_entries']} patients={row['n_patients']}")


def render_markdown(artifact: dict) -> str:
    lines = ["# Read-out timing follow-up", "", artifact["statement"], "", f"status: {artifact['status']}", ""]
    repro = artifact.get("reproduction", {})
    if repro.get("status") == "compared":
        lines += ["## Whole-trial stratified cell against the delivered benchmark", "",
                  f"largest absolute difference over {repro['n_expected_exact']} comparisons expected to be exact: "
                  f"{repro['max_abs_difference_expected_exact']}", ""]
    lines += ["## Counts by level", ""]
    for row in artifact.get("accounting", []):
        lines.append(f"- {row['level']}: sessions seen {row['sessions_seen']}, entries {row['entries_by_status']}")
    for title, rows, describe in (
            ("Absolute scores", artifact.get("absolute", []), lambda r: "|".join(r["cell"].values())),
            ("Contrasts", artifact.get("contrasts", []),
             lambda r: f"{r['kind']}: " + "|".join(r["minuend"].values()) + " minus " + "|".join(r["subtrahend"].values()))):
        lines += ["", f"## {title}, regions pooled by patient", ""]
        lines += [f"- {r['score']} {r['level']} {describe(r)}: {_fmt(r)}" for r in rows if r["region_cell"] == POOLED]
    return "\n".join(lines) + "\n"


def recompute_summaries(existing: Path, output: Path) -> None:
    """Rewrites the absolute and contrast summaries of a finished file from its stored entries."""
    if existing.resolve() == output.resolve():
        raise SystemExit("--output must differ from the file named by --summaries-only")
    artifact = json.loads(existing.read_text())
    scope = artifact["scope"]
    split_names = [STRATIFIED] + [chronological_name(f) for f in scope["train_fractions"]]
    artifact["absolute"], artifact["contrasts"] = summarise(
        artifact["entries"], tuple(scope["candidates"]), split_names, tuple(scope.get("references", REFERENCES)))
    artifact["summaries_recomputed"] = {"source_file": existing.name, "source_sha256": sha256_file(existing),
                                        "code_commit": git_commit(ROOT)}
    atomic_write(output, canonical_json(artifact))
    atomic_write(output.with_suffix(".md"), render_markdown(artifact))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Causal-inference and chronological-split follow-up of the "
                                            "single-unit representation benchmark.")
    p.add_argument("--candidates", nargs="*", choices=CANDIDATE_CHOICES, default=["gaussian_process_factor_analysis"])
    p.add_argument("--references", nargs="+", choices=REFERENCES, default=list(REFERENCES),
                   help="reference representations to score; the comparisons that need an unscored cell are not estimable")
    p.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    p.add_argument("--scores", nargs="+", choices=SCORES, default=list(SCORES))
    p.add_argument("--train-fractions", nargs="+", type=float, default=list(DEFAULT_TRAIN_FRACTIONS))
    p.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--entries-limit", type=int, help="loaded region-session entries to score per level")
    p.add_argument("--summaries-only", type=Path, metavar="EXISTING",
                   help="recompute the summaries from the stored entries of an existing output file and write "
                        "them to --output, without refitting")
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--delivered-root", type=Path, default=SNAPSHOT_ROOT,
                   help="directory holding the delivered benchmark snapshots the first cell is compared with")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.summaries_only is not None:
        recompute_summaries(args.summaries_only, args.output)
        print(f"wrote {args.output}", flush=True)
        return
    if args.n_perm < 1 or not all(0.0 < f < 1.0 for f in args.train_fractions):
        raise SystemExit("--n-perm must be positive and --train-fractions must lie strictly between 0 and 1")
    use_same_time_scoring()
    primary, supplement = benchmark.build_manifest()
    manifest = primary + supplement
    levels = dict(benchmark.LEVELS)
    models = tuple(args.candidates)
    split_names = [STRATIFIED] + [chronological_name(f) for f in args.train_fractions]
    references = tuple(args.references)
    config = {"models": models, "references": references, "scores": tuple(args.scores), "fractions": tuple(args.train_fractions),
              "n_perm": args.n_perm, "seed": args.seed, "checkpoint_dir": args.checkpoint_dir,
              "runtime": implementation_identity()}
    started = time.time()
    entries: list[dict] = []

    def flush(complete: bool) -> None:
        ordered = sorted(entries, key=lambda e: (e["level"], e.get("region", ""), e["patient"], e["session"]))
        artifact = {
            "version": "readout_timing_followup_v1", "code_commit": git_commit(ROOT),
            "status": "complete" if complete else "running", "statement": STATEMENT,
            "scope": {
                "corpus": "human single-unit recordings, hippocampus and amygdala region-session entries "
                          "of the representation benchmark",
                "levels": args.levels, "candidates": list(models), "references": list(references),
                "scores": args.scores, "decoder": DECODER, "operating_rank": OPERATING_RANK, "time_step": CTG_STEP,
                "n_stratified_folds": N_SPLITS, "train_fractions": args.train_fractions,
                "n_permutations": args.n_perm, "seed": args.seed, "entries_limit": args.entries_limit,
                "workers": args.workers, "wall_clock_s": float(time.time() - started),
                "inference_forms": {WHOLE: "latent at every bin inferred from the whole trial",
                                    CAUSAL: "latent at bin t inferred from bins 1..t only"},
                "splits": {STRATIFIED: "stratified folds with the benchmark's own fold draw",
                           **{chronological_name(f): f"first {f:g} of the trials in recorded order fit the "
                              "representation and the read-out, the rest are scored"
                              for f in args.train_fractions}},
            },
            "accounting": accounting(ordered), "timing": timing(ordered), "entries": ordered,
        }
        if complete:
            artifact["reproduction"] = reproduction(ordered, args.delivered_root, args.scores, args.seed)
            artifact["absolute"], artifact["contrasts"] = summarise(ordered, models, split_names, references)
        atomic_write(args.output, canonical_json(artifact))
        if complete:
            atomic_write(args.output.with_suffix(".md"), render_markdown(artifact))

    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=context, initializer=_init_worker,
                             initargs=(context.Lock(), models)) as executor:
        for level in args.levels:
            if args.entries_limit is None:
                futures = [executor.submit(process_session, row, level, levels[level], config, None)
                           for row in manifest]
                for future in as_completed(futures):
                    entries.extend(future.result())
                    flush(complete=False)
                continue
            n_loaded = 0
            for row in manifest:
                if n_loaded >= args.entries_limit:
                    break
                result = executor.submit(process_session, row, level, levels[level], config,
                                         args.entries_limit - n_loaded).result()
                entries.extend(result)
                n_loaded += sum(e["status"] == "loaded" for e in result)
                flush(complete=False)
    flush(complete=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
