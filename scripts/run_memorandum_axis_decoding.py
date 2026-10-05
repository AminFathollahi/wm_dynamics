#!/usr/bin/env python3
"""Memorandum axis decoding in human medial temporal single units (DANDI 000469, 001187 and 000673).

Demixed axes are fitted to the memorandum marginal (category effects with the condition-independent time course
removed) and to the time marginal of each session's real training trials, pooled across sessions into
session-blocked pseudo-trials and decoded with the read-out chosen by the memorandum decoding study. Reported: how
much of the picture category a few memorandum axes per session carry beside native units,
the time course of that decoding, whether axes fitted during encoding still carry the category during maintenance,
and the overlap between the fitted subspaces.

Run:
    WM_DYNAMICS_DATA_ROOT=<data root> python scripts/run_memorandum_axis_decoding.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
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
from info_decoding import DPCA_RIDGE_LAMBDA_GRID  # noqa: E402
from provenance import canonical_json, code_identity  # noqa: E402
from statistics import stable_seed  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "memorandum_axis_decoding.json"
SUMMARY_PATH = RESULTS / "memorandum_axis_decoding.md"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_memorandum_axis_decoding"
SCHEMA = "memorandum_axis_decoding_v1"

CONFIG = pooled.CONFIG
ENTRIES = pooled.ENTRIES
N_FOLDS = md.PSEUDO_FOLDS
LOADS = study.LOADS
PRIMARY_LOAD = study.PRIMARY_LOAD
METRICS = study.METRICS
CLASSES = np.arange(1, 6)
OVERLAP_BOOTSTRAP = 2000

NATIVE_SELECTED = "native_selected"
AXIS_REPRESENTATIONS = {
    "memorandum_axes_8": {"marginal": "memorandum", "d": 8, "fit_window": "maintenance"},
    "memorandum_axes_4": {"marginal": "memorandum", "d": 4, "fit_window": "maintenance"},
    "time_axes_8": {"marginal": "time", "d": 8, "fit_window": "maintenance"},
    "memorandum_axes_8_encoding_fitted": {"marginal": "memorandum", "d": 8, "fit_window": "encoding"},
    "memorandum_axes_4_encoding_fitted": {"marginal": "memorandum", "d": 4, "fit_window": "encoding"},
}
MAINTENANCE_FITTED = ("memorandum_axes_8", "memorandum_axes_4", "time_axes_8")
ENCODING_FITTED = ("memorandum_axes_8_encoding_fitted", "memorandum_axes_4_encoding_fitted")
ASSEMBLERS = {"blocked": md.session_blocked_folds, "transfer": md.session_blocked_transfer_folds}

# Windowed read-outs run in the primary cell only. Transfer read-outs share the trial draws of the time-course
# read-out of the same window width so that they are paired with it.
READOUTS = {
    "primary": {"kind": None, "assembly": "blocked", "course": False, "seed_name": "primary", "everywhere": True,
                "sources": ("native", NATIVE_SELECTED) + MAINTENANCE_FITTED},
    **{f"time_course_{ms}ms": {
        "kind": f"maintenance_{ms}ms", "assembly": "blocked", "course": True, "seed_name": f"time_course_{ms}ms",
        "everywhere": False, "sources": (NATIVE_SELECTED, "memorandum_axes_8", "memorandum_axes_4") + ENCODING_FITTED}
       for ms in (100, 250)},
    **{f"encoding_transfer_{ms}ms": {
        "kind": f"transfer_{ms}ms", "assembly": "transfer", "course": True, "seed_name": f"time_course_{ms}ms",
        "everywhere": False, "sources": ENCODING_FITTED} for ms in (100, 250)},
}
COMPARISONS = {
    "primary": [
        ("memorandum_axes_8_minus_native_selected", ("primary", "memorandum_axes_8"), ("primary", NATIVE_SELECTED)),
        ("memorandum_axes_8_minus_native_all_units", ("primary", "memorandum_axes_8"), ("primary", "native")),
        ("memorandum_axes_4_minus_native_selected", ("primary", "memorandum_axes_4"), ("primary", NATIVE_SELECTED)),
    ],
    **{f"time_course_{ms}ms": [
        ("memorandum_axes_8_minus_native_selected", (f"time_course_{ms}ms", "memorandum_axes_8"),
         (f"time_course_{ms}ms", NATIVE_SELECTED)),
        ("memorandum_axes_8_encoding_fitted_minus_maintenance_fitted",
         (f"time_course_{ms}ms", "memorandum_axes_8_encoding_fitted"), (f"time_course_{ms}ms", "memorandum_axes_8")),
        ("memorandum_axes_4_encoding_fitted_minus_maintenance_fitted",
         (f"time_course_{ms}ms", "memorandum_axes_4_encoding_fitted"), (f"time_course_{ms}ms", "memorandum_axes_4")),
    ] for ms in (100, 250)},
    **{f"encoding_transfer_{ms}ms": [
        (f"{source}_transfer_minus_within_window", (f"encoding_transfer_{ms}ms", source),
         (f"time_course_{ms}ms", source)) for source in ENCODING_FITTED] for ms in (100, 250)},
}


def needed_representations(load: int, region: str) -> tuple[str, ...]:
    return MAINTENANCE_FITTED + (ENCODING_FITTED if (load, region) == (PRIMARY_LOAD, "pooled") else ())


def fit_axes(Z, labels, rng, d, marginal) -> dict:
    return md.fit_demixed_axes(Z, labels, rng, d,
                               fit_fn=lambda z, c, lam, k: md.ridge_marginal_axes(z, c, lam, k, marginal))


def encoding_counts(key: str, region: str) -> np.ndarray:
    entry = ENTRIES[key]["entry"]
    windows = md.encoding_fit_bins()
    return entry["features"]["encoding_100ms"][:, md.region_columns(entry, region), :][:, :, windows].transpose(0, 2, 1)


def save_array(path: Path, array: np.ndarray) -> None:
    with open(str(path) + ".tmp", "wb") as handle:
        np.save(handle, array.astype(np.float32))
    os.replace(str(path) + ".tmp", path)


def run_axes_fit_task(task: dict) -> None:
    key, region, representation = task["key"], task["region"], task["representation"]
    spec = AXIS_REPRESENTATIONS[representation]
    counts, labels = pooled.region_counts(key, region), ENTRIES[key]["entry"]["labels"]
    fit_counts = encoding_counts(key, region) if spec["fit_window"] == "encoding" else None
    fold_ids = pooled.session_fold_ids(key)
    seed = stable_seed(f"pooled_latent|fit|{key}|{region}|{representation}")
    for fold in range(N_FOLDS):
        name = pooled.fit_key(key, region, representation, fold)

        def compute():
            result = md.fold_axes_series(counts, labels, fold_ids, fold, fit_axes, spec["marginal"], spec["d"], seed,
                                         fit_counts)
            path = pooled.checkpoint_file(name).with_suffix(".npy")
            series = result.pop("series", None)
            if series is not None:
                save_array(path, series)
                result.update(file=path.name, n_dimensions=int(series.shape[2]))
            encoding = result.pop("encoding_series", None)
            if encoding is not None:
                save_array(path.with_name(path.stem + "_encoding.npy"), encoding)
                result.update(encoding_file=path.stem + "_encoding.npy")
            return result

        if pooled.stored(name, CONFIG["fit_identity"], compute)["status"] == "not_estimable":
            break


def encoding_series(key: str, region: str, representation: str) -> list:
    out = []
    for f, record in enumerate(pooled.fold_records(key, region, representation)):
        out.append(None if record["status"] == "fold_dropped" else np.load(
            pooled.checkpoint_file(pooled.fit_key(key, region, representation, f)).with_name(
                record["encoding_file"])).astype(float))
    return out


def cell_pieces(task: dict) -> list[tuple[str, callable]]:
    assemble = ASSEMBLERS[task["assembly"]]
    pieces = [(task["id"] + "|observed", lambda b: md.pooled_observed_assembled(
        b, task["kind"], (task["decoder"],), CLASSES, N_FOLDS, CONFIG["n_resamples"], task["n_shuffles"], task["seed"],
        assemble))]
    for start in range(0, task["n_bootstrap"], pooled.BOOTSTRAP_CHUNK):
        indices = range(start, min(start + pooled.BOOTSTRAP_CHUNK, task["n_bootstrap"]))
        pieces.append((f"{task['id']}|boot{start}", lambda b, i=indices: md.pooled_bootstrap_assembled(
            b, task["kind"], (task["decoder"],), CLASSES, N_FOLDS, i, task["seed"], assemble)))
    return pieces


def make_blocks(task: dict) -> list[dict]:
    source = "native" if task["source"] == NATIVE_SELECTED else task["source"]
    blocks = pooled.make_blocks(task["keys"], task["region"], source, task["source"] == NATIVE_SELECTED)
    if task["assembly"] == "transfer":
        for block, key in zip(blocks, task["keys"]):
            block["encoding_series"] = encoding_series(key, task["region"], source)
    return blocks


def run_cell_task(task: dict) -> None:
    pieces = cell_pieces(task)
    if all(pooled.stored_record(name, CONFIG["cell_identity"]) is not None for name, _ in pieces):
        return
    blocks = make_blocks(task)
    for name, compute in pieces:
        pooled.stored(name, CONFIG["cell_identity"], lambda: compute(blocks))


SPAN_SUFFIX = {"axes": "", "encoder_axes": "_encoder_spans"}


def overlap_axes(records: list[dict], field: str) -> dict[int, np.ndarray]:
    return {f: np.asarray(r[field]) for f, r in enumerate(records) if r["status"] == "fitted"}


def run_overlap_task(task: dict) -> None:
    key, region = task["key"], task["region"]
    labels, fold_ids = ENTRIES[key]["entry"]["labels"], pooled.session_fold_ids(key)
    maintenance, encoding = pooled.region_counts(key, region), encoding_counts(key, region)

    def compute():
        names = ("memorandum_axes_8", "memorandum_axes_4", "time_axes_8") + ENCODING_FITTED
        spans = {field: {name: overlap_axes(pooled.fold_records(key, region, name), field) for name in names}
                 for field in SPAN_SUFFIX}
        out = {"units": int(maintenance.shape[2]), "n_permutations": CONFIG["n_overlap_permutations"], "by_rank": {}}
        for d in (8, 4):
            first = spans["axes"][f"memorandum_axes_{d}"]
            folds = sorted(set(first) & set(spans["axes"][f"memorandum_axes_{d}_encoding_fitted"]))
            seed = stable_seed(f"pooled_latent|overlap|{key}|{region}|{d}")
            null = {f: md.permuted_label_overlaps(
                encoding[fold_ids != f], maintenance[fold_ids != f], labels[fold_ids != f], fit_axes, d,
                CONFIG["n_overlap_permutations"], seed + f) for f in folds}
            row = {"folds": folds, "dimensions": int(next(iter(first.values())).shape[1]) if first else None}
            for field, suffix in SPAN_SUFFIX.items():
                mine, fitted = spans[field][f"memorandum_axes_{d}"], spans[field][f"memorandum_axes_{d}_encoding_fitted"]
                pairs = [(f, g) for f in sorted(mine) for g in sorted(mine) if f < g]
                row[f"encoding_vs_maintenance{suffix}"] = [md.subspace_overlap(fitted[f], mine[f]) for f in folds]
                row[f"permuted_label_encoding_vs_maintenance{suffix}"] = [null[f][field] for f in folds]
                row[f"maintenance_across_folds{suffix}"] = [md.subspace_overlap(mine[f], mine[g]) for f, g in pairs]
                if d == 8:
                    time_axes = spans[field]["time_axes_8"]
                    row[f"memorandum_vs_time{suffix}"] = [md.subspace_overlap(mine[f], time_axes[f])
                                                          for f in sorted(set(mine) & set(time_axes))]
            out["by_rank"][str(d)] = row
        return out

    pooled.stored(f"overlap|{key}|{region}", CONFIG["cell_identity"], compute)


def run_task(task: dict) -> str:
    if task["level"] == "fit":
        run_axes_fit_task(task)
    elif task["level"] == "cell":
        run_cell_task(task)
    else:
        run_overlap_task(task)
    return task["id"]


def execute(tasks: list[dict], workers: int, label: str) -> None:
    print(f"{label}: {len(tasks)} tasks", flush=True)
    started = time.time()
    pool = multiprocessing.get_context("fork").Pool(workers) if workers > 1 else None
    for done, task_id in enumerate(map(run_task, tasks) if pool is None else pool.imap_unordered(run_task, tasks), 1):
        print(f"{label} {done}/{len(tasks)} {time.time() - started:.0f}s {task_id}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()


def read_cell(task: dict):
    pieces = cell_pieces(task)
    records = [pooled.stored_record(name, CONFIG["cell_identity"]) for name, _ in pieces]
    if any(r is None for r in records):
        return None
    d, k = task["decoder"], task["kind"]
    observed, boots = records[0], records[1:]
    null = None if observed["null"] is None else {m: observed["null"][d][k][m] for m in METRICS}
    boot = {m: np.concatenate([b["scores"][d][k][m] for b in boots]) for m in METRICS} if boots else None
    return {"observed": {m: observed["scores"][d][k][m] for m in METRICS}, "null": null, "bootstrap": boot,
            "record": observed, "boot_records": boots}


def task_row(task: dict, cell: dict) -> dict:
    spec = md.FEATURES[md.KINDS[task["kind"]][1]]
    observed = cell["record"]
    return {"group": task["group"], "load": task["load"], "region": task["region"], "readout": task["readout"],
            "source": task["source"], "time_feature": task["kind"], "decoder": task["decoder"],
            "window_start_s": spec["starts"].tolist(), "window_width_s": spec["width"],
            "n_resamples": CONFIG["n_resamples"], "n_shuffles": task["n_shuffles"], "n_bootstrap": task["n_bootstrap"],
            "n_sessions": len(task["keys"]), "n_patients": len({ENTRIES[k]["entry"]["patient"] for k in task["keys"]}),
            "session_subset": task["subset_id"], "sessions": sorted(task["keys"]), **pooled.score_block(cell),
            "seconds": {"observed_resample": float(np.mean(observed["resample_seconds"])),
                        "null_total": float(observed["null_seconds"]),
                        "bootstrap_resample": float(np.mean(np.concatenate(
                            [b["resample_seconds"] for b in cell["boot_records"]]))) if cell["boot_records"] else None,
                        "task_total": float(observed["seconds"] + sum(b["seconds"] for b in cell["boot_records"]))}}


def difference_row(name: str, a: dict, b: dict, cell_a: dict, cell_b: dict) -> dict:
    row = {"group": a["group"], "load": a["load"], "region": a["region"], "name": name,
           "minuend": {"readout": a["readout"], "source": a["source"]},
           "subtrahend": {"readout": b["readout"], "source": b["source"]}, "n_sessions": a["n_sessions"],
           "n_patients": a["n_patients"], "window_start_s": a["window_start_s"], "window_width_s": a["window_width_s"]}
    for m in METRICS:
        if cell_a["bootstrap"] is not None and cell_b["bootstrap"] is not None:
            diff = md.paired_bootstrap_difference(cell_a["observed"][m], cell_b["observed"][m],
                                                  cell_a["bootstrap"][m], cell_b["bootstrap"][m])
        else:
            diff = {"estimate": (cell_a["observed"][m].mean(axis=0) - cell_b["observed"][m].mean(axis=0)).tolist()}
        if cell_a["null"] is not None and cell_b["null"] is not None:
            diff["estimate_above_null"] = (
                (cell_a["observed"][m].mean(axis=0) - cell_a["null"][m].mean(axis=0))
                - (cell_b["observed"][m].mean(axis=0) - cell_b["null"][m].mean(axis=0))).tolist()
        row[m] = diff
    return row


def cell_rows(tasks: list[dict], sets: dict) -> tuple[list[dict], list[dict]]:
    cells = {t["id"]: (t, read_cell(t)) for t in tasks}
    rows = [task_row(t, c) for t, c in cells.values() if c is not None]
    lookup = {(r["group"], r["load"], r["region"], r["readout"], r["source"], r["session_subset"]): r for r in rows}
    cell_of = {(t["group"], t["load"], t["region"], t["readout"], t["source"], t["subset_id"]): c
               for t, c in cells.values() if c is not None}
    differences = []
    for (load, region, group), info in sets.items():
        for comparisons in COMPARISONS.values():
            for name, (ra, sa), (rb, sb) in comparisons:
                ka, kb = [(group, load, region, r, s, info["common_subset_id"]) for r, s in ((ra, sa), (rb, sb))]
                if ka in cell_of and kb in cell_of:
                    differences.append(difference_row(name, lookup[ka], lookup[kb], cell_of[ka], cell_of[kb]))
    return rows, differences


def reproduction_check(rows: list[dict], reference_path: Path) -> list[dict]:
    """Whole-window rows of the primary cell that the pooled latent study also computed, with identical sessions and
    seeds: native counts on the union of sessions."""
    reference = json.loads(reference_path.read_text())
    theirs = {(r["group"], "native"): r for r in reference.get("native_matched_union", [])
              if (r["load"], r["region"], r["readout"]) == (PRIMARY_LOAD, "pooled", "primary")}
    sessions = {(r["group"], "native"): r for r in reference.get("native_matched_union", [])}
    out = []
    for r in rows:
        key = (r["group"], r["source"])
        if (r["load"], r["region"], r["readout"]) != (PRIMARY_LOAD, "pooled", "primary") or key not in theirs:
            continue
        other = theirs[key]
        other_sessions = sessions[key]["sessions"]
        for m in METRICS:
            mine, ref = float(np.mean(r[m]["observed"])), float(np.mean(other[m]["observed"]))
            out.append({"group": r["group"], "source": r["source"], "metric": m, "this_study": mine, "reference": ref,
                        "difference": mine - ref, "same_sessions": sorted(r["sessions"]) == sorted(other_sessions),
                        "sessions_this_study": r["n_sessions"], "sessions_reference": len(other_sessions)})
    return out


def describe(values: dict[str, float], patient_of: dict[str, str], rng: np.random.Generator) -> dict:
    """Median and interquartile range over sessions, and the mean over patients of each patient's mean with a
    patient-bootstrap interval."""
    if not values:
        return {"n_sessions": 0}
    array = np.array(list(values.values()))
    by_patient = {}
    for key, value in values.items():
        by_patient.setdefault(patient_of[key], []).append(value)
    patient_means = np.array([np.mean(v) for v in by_patient.values()])
    draws = np.array([patient_means[rng.integers(len(patient_means), size=len(patient_means))].mean()
                      for _ in range(OVERLAP_BOOTSTRAP)])
    q1, q3 = np.percentile(array, [25, 75])
    return {"n_sessions": len(array), "n_patients": len(patient_means), "median": float(np.median(array)),
            "interquartile_range": [float(q1), float(q3)], "patient_mean": float(patient_means.mean()),
            "patient_bootstrap_interval_95": np.percentile(draws, [2.5, 97.5]).tolist()}


def overlap_summary(records: dict[str, dict], groups: dict[str, str], seed: int) -> dict:
    out = {}
    rng = np.random.default_rng([seed, 7])
    patient_of = {k: ENTRIES[k]["entry"]["patient"] for k in records}
    for group in sorted(set(groups.values())):
        members = {k: r for k, r in records.items() if groups[k] == group}
        out[group] = {}
        for d in ("8", "4"):
            per = {k: r["by_rank"][d] for k, r in members.items() if len(r["by_rank"][d]["folds"])}
            quantities = {"random_subspace_expectation": {k: v["dimensions"] / members[k]["units"]
                                                          for k, v in per.items()}}
            for suffix in SPAN_SUFFIX.values():
                quantities.update({
                    f"encoding_vs_maintenance{suffix}": {
                        k: np.mean(v[f"encoding_vs_maintenance{suffix}"]) for k, v in per.items()},
                    f"permuted_label_encoding_vs_maintenance{suffix}": {
                        k: np.mean(v[f"permuted_label_encoding_vs_maintenance{suffix}"]) for k, v in per.items()},
                    f"maintenance_across_folds{suffix}": {
                        k: np.mean(v[f"maintenance_across_folds{suffix}"]) for k, v in per.items()
                        if len(v[f"maintenance_across_folds{suffix}"])}})
                quantities[f"encoding_vs_maintenance_minus_permuted_label{suffix}"] = {
                    k: quantities[f"encoding_vs_maintenance{suffix}"][k]
                    - quantities[f"permuted_label_encoding_vs_maintenance{suffix}"][k] for k in per}
                if d == "8":
                    quantities[f"memorandum_vs_time{suffix}"] = {
                        k: np.mean(v[f"memorandum_vs_time{suffix}"]) for k, v in per.items()
                        if len(v.get(f"memorandum_vs_time{suffix}", []))}
            out[group][f"rank{d}"] = {name: describe(values, patient_of, rng) for name, values in quantities.items()}
    return out


def render_summary(artifact: dict) -> str:
    lines = ["# Memorandum axis decoding", "", f"status: {artifact['status']}", "",
             "Balanced accuracy and one-versus-rest area under the curve, mean over windows; above-null values are "
             "observed minus the label-shuffle mean. Differences carry the patient-bootstrap 2.5 and 97.5 percentiles "
             "and p as numbers.", "",
             "## Scores", "", "| group | load | region | read-out | source | metric | observed | above null | sessions |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in artifact.get("scores", []):
        for m in METRICS:
            lines.append(f"| {r['group']} | {r['load']} | {r['region']} | {r['readout']} | {r['source']} | {m} | "
                         f"{np.mean(r[m]['observed']):.4f} | {np.mean(r[m].get('above_null', [float('nan')])):.4f} | "
                         f"{r['n_sessions']} |")
    lines += ["", "## Paired differences", "", "| group | load | region | comparison | metric | estimate | interval | p |",
              "|---|---|---|---|---|---|---|---|"]
    for r in artifact.get("differences", []):
        for m in METRICS:
            d = r[m]
            interval = "" if "interval_95" not in d else (
                f"{np.mean(d['interval_95'][0]):.4f} to {np.mean(d['interval_95'][1]):.4f}")
            p = "" if "p" not in d else f"{np.mean(d['p']):.4f}"
            lines.append(f"| {r['group']} | {r['load']} | {r['region']} | {r['name']} | {m} | "
                         f"{np.mean(d['estimate']):.4f} | {interval} | {p} |")
    lines += ["", "## Subspace overlap", "", "| group | rank | quantity | sessions | median | interquartile range | "
              "patient mean | patient bootstrap interval |", "|---|---|---|---|---|---|---|---|"]
    for group, ranks in artifact.get("subspace_overlap", {}).items():
        for rank, quantities in ranks.items():
            for name, s in quantities.items():
                if s["n_sessions"]:
                    lines.append(f"| {group} | {rank} | {name} | {s['n_sessions']} | {s['median']:.4f} | "
                                 f"{s['interquartile_range'][0]:.4f} to {s['interquartile_range'][1]:.4f} | "
                                 f"{s['patient_mean']:.4f} | {s['patient_bootstrap_interval_95'][0]:.4f} to "
                                 f"{s['patient_bootstrap_interval_95'][1]:.4f} |")
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Memorandum axis decoding.")
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--search-artifact", type=Path, default=pooled.SEARCH_ARTIFACT)
    p.add_argument("--reference-artifact", type=Path, help="pooled latent study artifact to reproduce")
    p.add_argument("--max-sessions-per-group", type=int)
    p.add_argument("--n-resamples", type=int, default=20)
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--n-bootstrap-time-course", type=int, default=50)
    p.add_argument("--n-shuffles", type=int, default=100)
    p.add_argument("--n-shuffles-time-course", type=int, default=20)
    p.add_argument("--n-overlap-permutations", type=int, default=50)
    p.add_argument("--selection-permutations", type=int, default=md.SELECTION_PERMUTATIONS)
    p.add_argument("--regions", nargs="+", choices=md.REGIONS, default=list(md.REGIONS))
    p.add_argument("--loads", nargs="+", type=int, default=list(LOADS))
    p.add_argument("--readouts", nargs="+", choices=tuple(READOUTS), default=list(READOUTS))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p


def common_sessions(fit_sets: dict, exclusions: dict, union: list[str]) -> dict:
    """Sessions fitted by every representation of the cell, so that every paired comparison shares one session
    set, and the first reason each other session was left out."""
    common = set(union)
    for kept in fit_sets.values():
        common &= set(kept)
    reasons = {k: next(f"{r}: {exclusions[r][k]}" for r in exclusions if k in exclusions[r]) for k in union
               if k not in common}
    subset_id = pooled.subset_id(sorted(common), union)
    return {"keys": sorted(common), "common_subset_id": subset_id, "left_out": reasons}


def build_cell_tasks(args, chosen: dict, decoder: str, sets: dict) -> list[dict]:
    tasks = []
    for (load, region, group), info in sets.items():
        for name in args.readouts:
            spec = READOUTS[name]
            if not spec["everywhere"] and (load, region) != (PRIMARY_LOAD, "pooled"):
                continue
            kind = spec["kind"] or chosen["time_feature"]
            wanted = [(source, info["common"]) for source in spec["sources"]
                      if load == PRIMARY_LOAD and region == "pooled" or source not in ENCODING_FITTED]
            if name == "primary":
                wanted.append(("native", info["union"]))
            seen = set()
            for source, keys in wanted:
                sid = pooled.subset_id(keys, info["union"])
                task_id = f"cell|{group}|load{load}|{region}|{name}|{source}|{sid}"
                if task_id in seen or not keys:
                    continue
                seen.add(task_id)
                tasks.append({
                    "id": task_id, "level": "cell", "group": group, "load": load, "region": region, "readout": name,
                    "source": source, "keys": keys, "subset_id": sid, "kind": kind, "decoder": decoder,
                    "assembly": spec["assembly"], "n_bootstrap": args.n_bootstrap_time_course if spec["course"]
                    else args.n_bootstrap, "n_shuffles": args.n_shuffles_time_course if spec["course"]
                    else args.n_shuffles, "seed": stable_seed(
                        f"pooled_latent|{group}|load{load}|{region}|{spec['seed_name']}|{sid}|{args.seed}")})
    return tasks


def main() -> None:
    args = parser().parse_args()
    if min(args.n_resamples, args.n_shuffles, args.n_shuffles_time_course, args.n_overlap_permutations) < 1 or min(
            args.n_bootstrap, args.n_bootstrap_time_course) < 0:
        raise SystemExit("resample, shuffle and permutation counts must be positive; bootstrap counts must not be negative")
    if args.reference_artifact and not args.reference_artifact.exists():
        raise SystemExit(f"reference artifact not found: {args.reference_artifact}")
    started = time.time()
    chosen, decoder, decoder_note = pooled.read_search(args.search_artifact)
    md.SELECTION_PERMUTATIONS = args.selection_permutations
    identity = code_identity(ROOT, Path(__file__))
    shared = {"code": identity, "seed": args.seed, "folds": N_FOLDS}
    CONFIG.update(
        checkpoint_dir=args.checkpoint_dir, n_resamples=args.n_resamples,
        n_overlap_permutations=args.n_overlap_permutations,
        fit_identity=hashlib.sha256(canonical_json(shared).encode()).hexdigest(),
        cell_identity=hashlib.sha256(canonical_json({**shared, "chosen": chosen, "decoder": decoder, "args": {
            k: getattr(args, k) for k in ("n_resamples", "n_bootstrap", "n_bootstrap_time_course", "n_shuffles",
                                          "n_shuffles_time_course", "n_overlap_permutations",
                                          "selection_permutations", "max_sessions_per_group")}}).encode()).hexdigest())
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accounting = pooled.build_entries(args.max_sessions_per_group)
    admitted, excluded = {}, {}
    for load in args.loads:
        for region in args.regions:
            for group, releases in study.PSEUDO_GROUPS.items():
                keys = pooled.admitted_keys(load, region, releases)
                kept, dropped = md.representation_session_set({k: pooled.fold_records(k, region, "native") for k in keys})
                admitted[(load, region, group)] = sorted(kept)
                excluded[(load, region, group)] = dropped
    artifact = {
        "version": SCHEMA, "status": "running", "code_identity": identity, "checkpoint_identities": {
            "fits": CONFIG["fit_identity"], "cells": CONFIG["cell_identity"]},
        "search_artifact": str(args.search_artifact), "chosen_read_out": chosen,
        "scope": {
            "datasets": list(md.RELEASES), "loads": args.loads, "primary_load": PRIMARY_LOAD, "regions": args.regions,
            "task_rate_floor_hz": md.TASK_RATE_FLOOR_HZ, "minimum_units_in_region_per_session": md.MIN_UNITS_PER_REGION,
            "pseudo_population_groups": {k: list(v) for k, v in study.PSEUDO_GROUPS.items()},
            "outer_folds_per_session": N_FOLDS, "pseudo_trials_per_category_training": md.PSEUDO_TRAIN_PER_CATEGORY,
            "pseudo_trials_per_category_test": md.PSEUDO_TEST_PER_CATEGORY,
            "assembly": "session-blocked: within a fold each pseudo-trial draws, with replacement, one real training "
                        "(or test) trial of its category per session, and that trial supplies every unit or axis "
                        "dimension of the session",
            "decoder": decoder, "decoder_note": decoder_note, "time_feature": chosen["time_feature"],
            "nested_cross_validation": {"inner_folds": md.INNER_FOLDS, "grids": {
                k: [list(v) if isinstance(v, tuple) else v for v in g] for k, g in md.HYPER_GRIDS.items()}},
            "axes": {"fit": "ridge reduced-rank regression B of the anscombe-transformed counts of every (trial, bin) "
                            "row on the marginal target; the encoder is the top eigenvectors of B' Sxx B and the "
                            "decoder is B times the encoder; components are the counts minus their mean, "
                            "projected on the decoder",
                     "ridge_penalty_grid": list(DPCA_RIDGE_LAMBDA_GRID),
                     "ridge_penalty_selection": "held-out reconstruction of the marginal target through the decoder "
                                                "and encoder inside the fold's training trials",
                     "memorandum_target": "category mean at each bin minus the mean over categories at that bin",
                     "time_target": "mean over categories at each bin minus the grand mean",
                     "ranks_capped_at_unit_count": True,
                     "representations": {k: dict(v) for k, v in AXIS_REPRESENTATIONS.items()},
                     "encoding_fit_window_s": list(md.SELECTION_WINDOW_S),
                     "encoding_fit_bins": md.encoding_fit_bins().tolist()},
            "native_selection": "category_selective_mask on the training trials' whole-maintenance counts, inside folds",
            "n_resamples": args.n_resamples, "n_bootstrap": args.n_bootstrap,
            "n_bootstrap_time_course": args.n_bootstrap_time_course, "n_shuffles": args.n_shuffles,
            "n_shuffles_time_course": args.n_shuffles_time_course, "n_overlap_permutations": args.n_overlap_permutations,
            "overlap_patient_bootstrap_resamples": OVERLAP_BOOTSTRAP, "readouts": args.readouts,
            "readout_cells": {"primary": "every region and load", "others": "regions pooled, load 1"},
            "estimate": "mean over trial-draw resamples without patient redraw",
            "interval": "2.5 and 97.5 percentiles of patient-bootstrap resamples (patients drawn with replacement "
                        "within the dataset group, all sessions of a drawn patient)",
            "p": "twice the smaller share of patient-bootstrap differences at or beyond zero; a number only",
            "shuffle_null": "labels of the assembled pseudo-trials permuted (training and test separately) for the "
                            "first resample's assembly, reusing fits and the observed hyperparameters; slightly "
                            "optimistic because the hyperparameters were chosen on the observed labels",
            "paired_rows": "all rows of one cell share one session set and one seed, so trial draws and patient "
                           "bootstrap draws are identical across rows",
            "transfer": "decoder trained on the window-averaged encoding-window projection and tested on each "
                        "maintenance window; maintenance values are multiplied by the window width so the decoder's "
                        "exposure normalisation leaves projections unscaled",
            "overlap": "sum of squared cosines of the principal angles divided by the smaller dimension, between "
                       "decoder spans and, with the suffix _encoder_spans, between encoder spans; maintenance "
                       "fits of different outer folds share three quarters of their training trials",
            "left_out": {"neural_data_transformer": "needs the accelerator, which was in use"},
            "seed": args.seed, "max_sessions_per_group": args.max_sessions_per_group},
        "session_accounting": accounting,
        "admission": {f"{g}|load{l}|{r}": {
            **pooled.admission_reasons(accounting, l, r, study.PSEUDO_GROUPS[g]), "admitted": len(k),
            "excluded_by_trial_count_rule": excluded[(l, r, g)]} for (l, r, g), k in admitted.items()}}

    def flush(**extra):
        artifact.update(extra)
        artifact["wall_clock_s"] = time.time() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_json(artifact))
        args.summary.write_text(render_summary(artifact))

    flush()
    pooled.FITS.update(study.latent_fits())
    fit_tasks = {}
    for (load, region, group), keys in admitted.items():
        for key in keys:
            for representation in needed_representations(load, region):
                fit_tasks.setdefault(f"fit|{key}|{region}|{representation}", {
                    "id": f"fit|{key}|{region}|{representation}", "level": "fit", "key": key, "region": region,
                    "representation": representation})
    execute(list(fit_tasks.values()), args.workers, "fits")
    sets, representation_sessions = {}, {}
    for (load, region, group), keys in admitted.items():
        fit_sets, exclusions = {}, {}
        for representation in needed_representations(load, region):
            fit_sets[representation], exclusions[representation] = md.representation_session_set(
                {k: pooled.fold_records(k, region, representation) for k in keys})
        common = common_sessions(fit_sets, exclusions, keys)
        sets[(load, region, group)] = {"union": keys, "common": common["keys"],
                                       "common_subset_id": common["common_subset_id"]}
        representation_sessions[f"{group}|load{load}|{region}"] = {
            "native_kept_sessions": len(keys), "common_session_set": len(common["keys"]),
            "left_out_of_common_set": common["left_out"],
            "by_representation": {r: {"kept": len(fit_sets[r]), "excluded": exclusions[r]} for r in fit_sets}}
    tasks = build_cell_tasks(args, chosen, decoder, sets)
    overlap_tasks = [{"id": f"overlap|{key}|pooled", "level": "overlap", "key": key, "region": "pooled"}
                     for (load, region, group), info in sets.items() if (load, region) == (PRIMARY_LOAD, "pooled")
                     for key in info["common"]]
    flush(representation_sessions=representation_sessions)
    execute(tasks + overlap_tasks, args.workers, "cells")
    rows, differences = cell_rows(tasks, sets)
    group_of = {t["key"]: next(g for (l, r, g), info in sets.items() if (l, r) == (PRIMARY_LOAD, "pooled")
                               and t["key"] in info["common"]) for t in overlap_tasks}
    overlap_records = {t["key"]: pooled.stored_record(f"overlap|{t['key']}|pooled", CONFIG["cell_identity"])
                       for t in overlap_tasks}
    overlap_records = {k: v for k, v in overlap_records.items() if v is not None}
    all_representations = sorted({r for (l, reg, g) in admitted for r in needed_representations(l, reg)})
    dimensions = {f"{g}|load{l}|{r}": {rep: {k: [rec.get("n_dimensions") for rec in pooled.fold_records(k, r, rep)]
                                           for k in keys} for rep in needed_representations(l, r)}
                  for (l, r, g), keys in admitted.items()}
    flush(scores=rows, differences=differences, dimensions_per_session_and_fold=dimensions,
          fit_timing=pooled.fit_timing(admitted, all_representations),
          subspace_overlap=overlap_summary(overlap_records, group_of, args.seed),
          subspace_overlap_sessions={k: v for k, v in overlap_records.items()},
          reproduction_check=reproduction_check(rows, args.reference_artifact) if args.reference_artifact else None,
          status="complete")


if __name__ == "__main__":
    main()
