#!/usr/bin/env python3
"""Pooled latent-state decoding of the memorandum (picture category) from human medial temporal single units in
DANDI 000469, 001187 and 000673.

Each latent model is fitted on one session's simultaneously recorded training trials, the latent dimensions of
many sessions are pooled into session-blocked pseudo-trials, and the read-out chosen by the memorandum decoding
study decodes them. Every latent row is paired draw by draw with native counts of the same sessions.

Run:
    WM_DYNAMICS_DATA_ROOT=<data root> python scripts/run_pooled_latent_memorandum_decoding.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
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
from provenance import (  # noqa: E402
    canonical_json, checkpoint_load, checkpoint_safe, checkpoint_store, code_identity, restore_checkpoint, sha256_file,
)
from statistics import stable_seed  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "pooled_latent_memorandum_decoding.json"
SUMMARY_PATH = RESULTS / "pooled_latent_memorandum_decoding.md"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_pooled_latent_memorandum_decoding"
SEARCH_ARTIFACT = RESULTS / "memorandum_decoding_study.json"
SCHEMA = "pooled_latent_memorandum_decoding_v1"

LOADS = study.LOADS
PRIMARY_LOAD = study.PRIMARY_LOAD
N_FOLDS = md.PSEUDO_FOLDS
BOOTSTRAP_CHUNK = 25
METRICS = study.METRICS
REPRESENTATIONS = tuple(r for r in study.REPRESENTATIONS if r != study.SLOWEST_REPRESENTATION) + (
    study.SLOWEST_REPRESENTATION,)
READOUTS = {
    "primary": {"kind": None, "decoder": None, "select": False, "course": False, "everywhere": True},
    "selected_dimensions": {"kind": None, "decoder": None, "select": True, "course": False, "everywhere": False},
    "time_course_100ms": {"kind": "maintenance_100ms", "decoder": None, "select": False, "course": True,
                          "everywhere": False},
    "time_course_250ms": {"kind": "maintenance_250ms", "decoder": None, "select": False, "course": True,
                          "everywhere": False},
    "radial_kernel": {"kind": None, "decoder": "radial_svm", "select": False, "course": False, "everywhere": False},
}

ENTRIES = study.ENTRIES
FITS: dict = {}
CONFIG: dict = {}


def checkpoint_file(key: str, directory: Path | None = None) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:80]
    return (directory or CONFIG["checkpoint_dir"]) / f"{stem}_{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def stored_record(key: str, identity: str, directory: Path | None = None):
    payload = checkpoint_load(checkpoint_file(key, directory))
    if payload and payload.get("complete") and payload.get("identity") == identity:
        return restore_checkpoint(payload["record"])
    return None


def stored(key: str, identity: str, compute):
    record = stored_record(key, identity)
    if record is not None:
        return record
    started = time.time()
    record = compute()
    record["seconds"] = time.time() - started
    checkpoint_store(checkpoint_file(key), {"identity": identity, "complete": True, "record": checkpoint_safe(record)})
    return record


def build_entries(max_per_group: int | None) -> list[dict]:
    """Load sessions and build per-load entries. With a limit, loading stops for a dataset group once that many
    sessions hold an entry that can enter the pooled assembly at the primary load."""
    group_of = {release: group for group, releases in study.PSEUDO_GROUPS.items() for release in releases}
    admitted_count = dict.fromkeys(study.PSEUDO_GROUPS, 0)
    accounting = []
    for row in md.session_manifest(ROOT / "provenance"):
        record = {key: row[key] for key in ("release", "patient", "session", "status")}
        group = group_of[row["release"]]
        if row["status"] == "listed" and max_per_group is not None and admitted_count[group] >= max_per_group:
            record["status"] = "not_selected_by_session_limit"
        elif row["status"] == "listed":
            session, record["status"] = md.load_session(row["path"], row["release"], row["patient"])
            record["loads"] = {}
            for load in (LOADS if session is not None else ()):
                entry = md.build_entry(session, load)
                info = {"unit_inclusion": md.unit_inclusion_counts(session, load)}
                if entry is None:
                    info["status"] = "no_correct_trials_or_no_admitted_units"
                else:
                    folds = md.feasible_fold_count(entry["labels"])
                    classes, counts = np.unique(entry["labels"], return_counts=True)
                    info.update(n_trials=len(entry["labels"]), n_units=len(entry["unit_region"]),
                                n_classes=len(classes), trials_per_class=counts.tolist(), n_folds=folds,
                                status="entry_built" if folds else "class_below_fold_count")
                    if folds:
                        ENTRIES[study.entry_key(row, load)] = {"entry": entry, "n_folds": folds,
                                                               "n_classes": len(classes)}
                record["loads"][str(load)] = info
            if session is not None and any(
                    admitted_keys(PRIMARY_LOAD, region, study.PSEUDO_GROUPS[group], only=study.entry_key(row, PRIMARY_LOAD))
                    for region in md.REGIONS):
                admitted_count[group] += 1
        accounting.append(record)
    return accounting


def admitted_keys(load: int, region: str, releases, only: str | None = None) -> list[str]:
    """Entries with all five categories and at least the frozen minimum of units in the region."""
    keys = study.entry_keys(load, region, releases, pseudo=True)
    return [k for k in keys if (only is None or k == only)
            and len(md.region_columns(ENTRIES[k]["entry"], region)) >= md.MIN_UNITS_PER_REGION]


def admission_reasons(accounting: list[dict], load: int, region: str, releases) -> dict:
    """Every file seen for the dataset group, as admitted or excluded with the reason."""
    out = {"files_seen": 0, "entries_admitted_before_trial_rule": 0, "excluded": {}}
    for row in accounting:
        if row["release"] not in releases:
            continue
        out["files_seen"] += 1
        if row["status"] != "loaded":
            reason = row["status"]
        else:
            info = row["loads"][str(load)]
            key = study.entry_key(row, load)
            if key not in ENTRIES:
                reason = info["status"]
            elif ENTRIES[key]["n_classes"] != 5:
                reason = "not_five_categories"
            elif len(md.region_columns(ENTRIES[key]["entry"], region)) < md.MIN_UNITS_PER_REGION:
                reason = "fewer_than_8_units_in_region"
            else:
                out["entries_admitted_before_trial_rule"] += 1
                continue
        out["excluded"][reason] = out["excluded"].get(reason, 0) + 1
    return out


def session_fold_ids(key: str) -> np.ndarray:
    labels = ENTRIES[key]["entry"]["labels"]
    return md.stratified_fold_ids(labels, N_FOLDS, np.random.default_rng(stable_seed(f"pooled_latent|folds|{key}")))


def region_counts(key: str, region: str) -> np.ndarray:
    entry = ENTRIES[key]["entry"]
    return entry["features"]["maintenance_100ms"][:, md.region_columns(entry, region), :].transpose(0, 2, 1)


def fit_key(key: str, region: str, representation: str, fold: int) -> str:
    return f"fit|{key}|{region}|{representation}|fold{fold}"


def run_fit_task(task: dict) -> None:
    key, region, representation = task["key"], task["region"], task["representation"]
    counts, labels = region_counts(key, region), ENTRIES[key]["entry"]["labels"]
    fold_ids = session_fold_ids(key)
    seed = stable_seed(f"pooled_latent|fit|{key}|{region}|{representation}")
    for fold in range(N_FOLDS):
        name = fit_key(key, region, representation, fold)

        def compute():
            result = md.fold_latent_series(counts, labels, fold_ids, fold, representation, FITS, seed)
            series = result.pop("series", None)
            if series is not None:
                path = checkpoint_file(name).with_suffix(".npy")
                with open(str(path) + ".tmp", "wb") as handle:
                    np.save(handle, series.astype(np.float32))
                os.replace(str(path) + ".tmp", path)
                result.update(file=path.name, n_dimensions=int(series.shape[2]))
            return result

        if stored(name, CONFIG["fit_identity"], compute)["status"] == "not_estimable":
            break


def fold_records(key: str, region: str, representation: str) -> list[dict]:
    """Per-fold fit records. Native counts need no fit; their records only mark folds dropped by the trial rule."""
    if representation == "native":
        labels, fold_ids = ENTRIES[key]["entry"]["labels"], session_fold_ids(key)
        classes = np.unique(labels)
        return [{"status": "fold_dropped" if md.fold_row_pools(labels, fold_ids, f, classes)[1] else "fitted"}
                for f in range(N_FOLDS)]
    out = []
    for f in range(N_FOLDS):
        record = stored_record(fit_key(key, region, representation, f), CONFIG["fit_identity"])
        if record is None:
            return [{"status": "not_estimable", "reason": "fit_missing"}]
        out.append(record)
        if record["status"] == "not_estimable":
            break
    return out


def load_series(key: str, region: str, representation: str) -> list:
    if representation == "native":
        counts = region_counts(key, region)
        return [counts] * N_FOLDS
    series = []
    for f, record in enumerate(fold_records(key, region, representation)):
        series.append(None if record["status"] == "fold_dropped" else np.load(
            checkpoint_file(fit_key(key, region, representation, f)).with_name(record["file"])).astype(float))
    return series


def make_blocks(keys: list[str], region: str, source: str, select: bool) -> list[dict]:
    blocks = []
    for key in keys:
        entry = ENTRIES[key]["entry"]
        labels, fold_ids = entry["labels"], session_fold_ids(key)
        series = load_series(key, region, source)
        block = {"key": stable_seed(key), "copy": 0, "patient": entry["patient"], "labels": labels,
                 "fold_ids": fold_ids, "series": series}
        if select:
            seed = stable_seed(f"pooled_latent|selection|{key}|{region}|{source}")
            block["columns"] = [None if series[f] is None else md.selected_dimension_columns(
                series[f], labels, fold_ids, f, np.random.default_rng([seed, f])) for f in range(N_FOLDS)]
        blocks.append(block)
    return blocks


def cell_pieces(task: dict) -> list[tuple[str, callable]]:
    pieces = [(task["id"] + "|observed", lambda b: md.pooled_observed(
        b, task["kind"], (task["decoder"],), task["classes"], N_FOLDS, CONFIG["n_resamples"], task["n_shuffles"],
        task["seed"]))]
    for start in range(0, task["n_bootstrap"], BOOTSTRAP_CHUNK):
        indices = range(start, min(start + BOOTSTRAP_CHUNK, task["n_bootstrap"]))
        pieces.append((f"{task['id']}|boot{start}", lambda b, i=indices: md.pooled_bootstrap(
            b, task["kind"], (task["decoder"],), task["classes"], N_FOLDS, i, task["seed"])))
    return pieces


def cell_source(task: dict) -> tuple[Path, str]:
    """Checkpoint directory and identity holding a cell. A native cell the finished run computed is read from the
    finished run's checkpoints, so latent-minus-native differences stay paired with the same draws."""
    if CONFIG.get("finished") and task["source"] == "native":
        directory, identity = CONFIG["finished"]
        if all(stored_record(name, identity, directory) is not None
               for name, _ in cell_pieces({**task, "classes": np.arange(1, 6)})):
            return directory, identity
    return CONFIG["checkpoint_dir"], CONFIG["cell_identity"]


def run_cell_task(task: dict) -> None:
    task = {**task, "classes": np.arange(1, 6)}
    pieces = cell_pieces(task)
    directory, identity = cell_source(task)
    if all(stored_record(name, identity, directory) is not None for name, _ in pieces):
        return
    blocks = make_blocks(task["keys"], task["region"], task["source"], task["select"])
    for name, compute in pieces:
        stored(name, CONFIG["cell_identity"], lambda: compute(blocks))


def run_task(task: dict) -> str:
    if task["level"] == "fit":
        run_fit_task(task)
    elif task["level"] == "cell":
        run_cell_task(task)
    else:
        study.run_read_out_task(task)
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


def read_search(path: Path) -> tuple[dict, str, str | None]:
    if not path.exists():
        raise SystemExit(f"search artifact not found: {path}")
    artifact = json.loads(path.read_text())
    chosen = artifact.get("chosen_read_out")
    if not chosen or not {"decoder", "time_feature", "unit_set"} <= set(chosen):
        raise SystemExit(f"search artifact {path} holds no chosen read-out with decoder, time feature and unit set")
    if study.UNIT_SETS[chosen["unit_set"]][0] != "pseudo":
        raise SystemExit(f"chosen unit set {chosen['unit_set']} is not a pseudo-population; nothing to pool")
    decoder, note = study.latent_decoder(chosen, artifact.get("read_out_search", {}).get("summary", []))
    return chosen, decoder, note


def subset_id(keys: list[str], union: list[str]) -> str:
    return "all" if sorted(keys) == sorted(union) else hashlib.sha256(",".join(sorted(keys)).encode()).hexdigest()[:10]


def readout_settings(name: str, chosen: dict, decoder: str, args) -> dict:
    spec = READOUTS[name]
    return {"kind": spec["kind"] or chosen["time_feature"], "decoder": spec["decoder"] or decoder,
            "select": spec["select"], "n_bootstrap": args.n_bootstrap_time_course if spec["course"] else args.n_bootstrap,
            "n_shuffles": args.n_shuffles_time_course if spec["course"] else args.n_shuffles}


def build_cell_tasks(args, chosen: dict, decoder: str, admitted: dict, fit_sets: dict, representations) -> list[dict]:
    tasks = []
    for (load, region, group), union in admitted.items():
        for name in args.readouts:
            if not READOUTS[name]["everywhere"] and (load, region) != (PRIMARY_LOAD, "pooled"):
                continue
            settings = readout_settings(name, chosen, decoder, args)
            if name == "radial_kernel" and decoder == "radial_svm":
                continue
            subsets = {r: fit_sets[(load, region, group)][r] for r in representations}
            wanted = [("native", union)] + [(r, k) for r, k in subsets.items() if k]
            wanted += [("native", k) for r, k in subsets.items() if k and sorted(k) != sorted(union)]
            seen = set()
            for source, keys in wanted:
                sid = subset_id(keys, union)
                task_id = f"cell|{group}|load{load}|{region}|{name}|{source}|{sid}"
                if task_id in seen:
                    continue
                seen.add(task_id)
                cell_group = f"{group}|load{load}|{region}|{name}|{sid}|{args.seed}"
                tasks.append({"id": task_id, "level": "cell", "group": group, "load": load, "region": region,
                              "readout": name, "source": source, "keys": keys, "subset_id": sid,
                              "seed": stable_seed("pooled_latent|" + cell_group), **settings})
    return sorted(tasks, key=lambda t: t["source"] == study.SLOWEST_REPRESENTATION)


def read_cell(task: dict):
    pieces = cell_pieces({**task, "classes": np.arange(1, 6)})
    directory, identity = cell_source(task)
    records = [stored_record(name, identity, directory) for name, _ in pieces]
    if any(r is None for r in records):
        return None
    d, k = task["decoder"], task["kind"]
    observed, boots = records[0], records[1:]
    scores = {m: observed["scores"][d][k][m] for m in METRICS}
    null = None if observed["null"] is None else {m: observed["null"][d][k][m] for m in METRICS}
    boot = {m: np.concatenate([b["scores"][d][k][m] for b in boots]) for m in METRICS} if boots else None
    return {"observed": scores, "null": null, "bootstrap": boot, "record": observed, "boot_records": boots}


def score_block(cell: dict) -> dict:
    out = {"mean_dimensions": float(np.mean(cell["record"]["mean_dimensions"])),
           "mean_dropped_sessions_per_fold": {r: float(np.mean([d[r] for d in cell["record"]["mean_dropped_sessions"]]))
                                              for r in cell["record"]["mean_dropped_sessions"][0]}}
    for m in METRICS:
        row = {"observed": cell["observed"][m].mean(axis=0).tolist()}
        if cell["null"] is not None:
            row["null_mean"] = cell["null"][m].mean(axis=0).tolist()
            row["above_null"] = (cell["observed"][m].mean(axis=0) - cell["null"][m].mean(axis=0)).tolist()
        out[m] = row
    return out


def cell_rows(tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    by_id = {t["id"]: t for t in tasks}
    rows, union_rows = [], []
    for task in tasks:
        cell = read_cell(task)
        if cell is None:
            continue
        spec = md.FEATURES[md.KINDS[task["kind"]][1]]
        base = {"group": task["group"], "load": task["load"], "region": task["region"], "readout": task["readout"],
                "time_feature": task["kind"], "decoder": task["decoder"], "selected_dimensions": task["select"],
                "window_start_s": spec["starts"].tolist(), "window_width_s": spec["width"],
                "n_resamples": CONFIG["n_resamples"], "n_shuffles": task["n_shuffles"],
                "n_bootstrap": task["n_bootstrap"], "n_sessions": len(task["keys"]),
                "n_patients": len({ENTRIES[k]["entry"]["patient"] for k in task["keys"]}),
                "session_subset": task["subset_id"], "sessions": sorted(task["keys"])}
        observed = cell["record"]
        seconds = {"observed_resample": float(np.mean(observed["resample_seconds"])),
                   "null_total": float(observed["null_seconds"]),
                   "bootstrap_resample": float(np.mean(np.concatenate([b["resample_seconds"] for b in cell["boot_records"]])))
                   if cell["boot_records"] else None, "task_total": float(observed["seconds"] + sum(
                       b["seconds"] for b in cell["boot_records"]))}
        if task["source"] == "native":
            if task["subset_id"] == "all":
                union_rows.append({**base, "representation": "native", **score_block(cell), "seconds": seconds})
            continue
        matched = by_id[task["id"].rsplit("|", 2)[0] + f"|native|{task['subset_id']}"]
        native = read_cell(matched)
        if native is None:
            continue
        row = {**base, "representation": task["source"], "latent": score_block(cell),
               "native_matched": score_block(native), "seconds": seconds}
        row["latent_minus_native"] = {}
        for m in METRICS:
            diff = md.paired_bootstrap_difference(cell["observed"][m], native["observed"][m],
                                                  cell["bootstrap"][m], native["bootstrap"][m]) if cell["bootstrap"] else {
                "estimate": (cell["observed"][m].mean(axis=0) - native["observed"][m].mean(axis=0)).tolist()}
            if cell["null"] is not None and native["null"] is not None:
                diff["estimate_above_null"] = (
                    (cell["observed"][m].mean(axis=0) - cell["null"][m].mean(axis=0))
                    - (native["observed"][m].mean(axis=0) - native["null"][m].mean(axis=0))).tolist()
            row["latent_minus_native"][m] = diff
        rows.append(row)
    return rows, union_rows


def fit_timing(admitted: dict, representations) -> dict:
    out = {}
    for representation in representations:
        seconds, status = [], {}
        for (load, region, group), keys in admitted.items():
            for key in keys:
                for f in range(N_FOLDS):
                    record = stored_record(fit_key(key, region, representation, f), CONFIG["fit_identity"])
                    if record is None:
                        continue
                    status[record["status"]] = status.get(record["status"], 0) + 1
                    if record["status"] == "fitted":
                        seconds.append(record["seconds"])
        out[representation] = {"fits": len(seconds), "mean_seconds": float(np.mean(seconds)) if seconds else None,
                               "total_seconds": float(np.sum(seconds)), "fold_status_counts": status}
    return out


def render_summary(artifact: dict) -> str:
    """One row per cell and metric; windowed read-outs show the mean over windows (per-window values are in the
    artifact)."""
    lines = ["# Pooled latent memorandum decoding", "", f"status: {artifact['status']}", "",
             "Latent minus native-matched, estimate (mean over trial draws) with patient-bootstrap 2.5 and 97.5 "
             "percentiles and p, as numbers. Above-null values are observed minus the label-shuffle mean.", "",
             "| group | load | region | read-out | representation | metric | latent above null | native-matched above "
             "null | latent minus native | interval | p | sessions | patients |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in artifact.get("cells", []):
        for m in METRICS:
            d = r["latent_minus_native"][m]
            interval = "" if "interval_95" not in d else (
                f"{np.mean(d['interval_95'][0]):.4f} to {np.mean(d['interval_95'][1]):.4f}")
            p = "" if "p" not in d else f"{np.mean(d['p']):.4f}"
            above = [np.mean(r[side][m].get("above_null", [float("nan")])) for side in ("latent", "native_matched")]
            lines.append(f"| {r['group']} | {r['load']} | {r['region']} | {r['readout']} | {r['representation']} | {m} | "
                         f"{above[0]:.4f} | {above[1]:.4f} | {np.mean(d['estimate']):.4f} | {interval} | {p} | "
                         f"{r['n_sessions']} | {r['n_patients']} |")
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pooled latent memorandum decoding.")
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--search-artifact", type=Path, default=SEARCH_ARTIFACT)
    p.add_argument("--max-sessions-per-group", type=int)
    p.add_argument("--n-resamples", type=int, default=20)
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--n-bootstrap-time-course", type=int, default=50)
    p.add_argument("--n-shuffles", type=int, default=100)
    p.add_argument("--n-shuffles-time-course", type=int, default=20)
    p.add_argument("--selection-permutations", type=int, default=md.SELECTION_PERMUTATIONS)
    p.add_argument("--regions", nargs="+", choices=md.REGIONS, default=list(md.REGIONS))
    p.add_argument("--loads", nargs="+", type=int, default=list(LOADS))
    p.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    p.add_argument("--readouts", nargs="+", choices=tuple(READOUTS), default=list(READOUTS))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demixed-refit-of", type=Path, metavar="FINISHED_JSON",
                   help="refit only the demixed principal component rows of this finished artifact; native cells it "
                        "computed are read from --finished-checkpoint-dir")
    p.add_argument("--finished-checkpoint-dir", type=Path)
    return p


def finished_run_settings(args) -> dict:
    """Checks that the demixed-only refit shares every setting of the finished run and returns that artifact."""
    finished = json.loads(args.demixed_refit_of.read_text())
    if args.finished_checkpoint_dir is None or args.finished_checkpoint_dir.resolve() == args.checkpoint_dir.resolve():
        raise SystemExit("--finished-checkpoint-dir is required and must differ from --checkpoint-dir")
    if finished.get("status") != "complete":
        raise SystemExit("the finished artifact is not complete")
    for name in ("n_resamples", "n_bootstrap", "n_bootstrap_time_course", "n_shuffles", "n_shuffles_time_course", "seed",
                 "max_sessions_per_group"):
        if finished["scope"][name] != getattr(args, name):
            raise SystemExit(f"--{name.replace('_', '-')} differs from the finished artifact ({finished['scope'][name]})")
    args.representations = [study.DEMIXED]
    return finished


def main() -> None:
    args = parser().parse_args()
    if min(args.n_resamples, args.n_shuffles, args.n_shuffles_time_course) < 1 or min(
            args.n_bootstrap, args.n_bootstrap_time_course) < 0:
        raise SystemExit("resample and shuffle counts must be positive; bootstrap counts must not be negative")
    started = time.time()
    finished = finished_run_settings(args) if args.demixed_refit_of else None
    chosen, decoder, decoder_note = read_search(args.search_artifact)
    if finished and (canonical_json(chosen) != canonical_json(finished["chosen_read_out"])
                     or decoder != finished["scope"]["decoder"]):
        raise SystemExit("the search artifact does not hold the finished run's chosen read-out")
    md.SELECTION_PERMUTATIONS = args.selection_permutations
    identity = code_identity(ROOT, Path(__file__))
    shared = {"code": identity, "seed": args.seed, "folds": N_FOLDS}
    CONFIG.update(
        checkpoint_dir=args.checkpoint_dir, n_resamples=args.n_resamples,
        fit_identity=hashlib.sha256(canonical_json(shared).encode()).hexdigest(),
        cell_identity=hashlib.sha256(canonical_json({**shared, "chosen": chosen, "decoder": decoder, "args": {
            k: getattr(args, k) for k in ("n_resamples", "n_bootstrap", "n_bootstrap_time_course", "n_shuffles",
                                          "n_shuffles_time_course", "selection_permutations",
                                          "max_sessions_per_group")}}).encode()).hexdigest())
    if finished:
        CONFIG["finished"] = (args.finished_checkpoint_dir, finished["checkpoint_identities"]["cells"])
    study.CONFIG.update(checkpoint_dir=args.checkpoint_dir / "native_curves", n_shuffles=args.n_shuffles,
                        n_resamples=args.n_resamples, identity=CONFIG["cell_identity"])
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accounting = build_entries(args.max_sessions_per_group)
    admitted, excluded = {}, {}
    for load in args.loads:
        for region in args.regions:
            for group, releases in study.PSEUDO_GROUPS.items():
                keys = admitted_keys(load, region, releases)
                kept, dropped = md.representation_session_set({k: fold_records(k, region, "native") for k in keys})
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
                        "(or test) trial of its category per session, and that trial supplies every unit or latent "
                        "dimension of the session; within-session covariance is kept, across sessions it is not",
            "drop_rule": "a session is dropped from a fold when some category has no test trial or fewer than 2 "
                         "training trials",
            "decoder": decoder, "decoder_note": decoder_note, "time_feature": chosen["time_feature"],
            "nested_cross_validation": {"inner_folds": md.INNER_FOLDS, "grids": {
                k: [list(v) if isinstance(v, tuple) else v for v in g] for k, g in md.HYPER_GRIDS.items()}},
            "n_resamples": args.n_resamples, "n_bootstrap": args.n_bootstrap,
            "n_bootstrap_time_course": args.n_bootstrap_time_course, "n_shuffles": args.n_shuffles,
            "n_shuffles_time_course": args.n_shuffles_time_course, "readouts": args.readouts,
            "readout_cells": {"primary": "every region and load", "others": "regions pooled, load 1"},
            "estimate": "mean over trial-draw resamples without patient redraw",
            "interval": "2.5 and 97.5 percentiles of patient-bootstrap resamples (patients drawn with replacement "
                        "within the dataset group, all sessions of a drawn patient)",
            "p": "twice the smaller share of patient-bootstrap differences at or beyond zero; a number only",
            "native_matched": "the same sessions, folds and drawn trials, native 100 ms counts window-averaged "
                              "the same way; reported per representation on its own session set and once on the union",
            "shuffle_null": "labels of the assembled pseudo-trials permuted (training and test separately) for the "
                            "first resample's assembly, reusing fits and the observed hyperparameters; this differs from "
                            "the search null (labels shuffled within session before assembly) and is slightly optimistic "
                            "because the hyperparameters were chosen on the observed labels",
            "selected_dimensions_note": "category-selective dimensions (units for the native-matched row) are chosen "
                                        "inside each training fold on the session's whole-maintenance values; the "
                                        "search selected units on 200-1200 ms after picture onset, which latent "
                                        "states do not cover",
            "left_out": {"neural_data_transformer": "needs the accelerator, which was in use"},
            "representations": ["native"] + list(args.representations),
            "seed": args.seed, "max_sessions_per_group": args.max_sessions_per_group},
        "session_accounting": accounting,
        "admission": {f"{g}|load{l}|{r}": {
            **admission_reasons(accounting, l, r, study.PSEUDO_GROUPS[g]), "admitted": len(k),
            "excluded_by_trial_count_rule": excluded[(l, r, g)]} for (l, r, g), k in admitted.items()}}

    def flush(**extra):
        artifact.update(extra)
        artifact["wall_clock_s"] = time.time() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_json(artifact))
        args.summary.write_text(render_summary(artifact))

    if finished:
        artifact["refit"] = {"of_artifact": str(args.demixed_refit_of), "of_artifact_sha256": sha256_file(args.demixed_refit_of),
                             "finished_checkpoint_dir": str(args.finished_checkpoint_dir)}
    flush()
    FITS.update(study.latent_fits())
    fast = tuple(r for r in args.representations if r != study.SLOWEST_REPRESENTATION)
    phases = [phase for phase in (fast, tuple(r for r in args.representations if r not in fast)) if phase]
    fit_sets, exclusions, cells, curves, done = {}, {}, {}, [], []
    for phase in phases:
        fit_tasks = [{"id": f"fit|{key}|{region}|{r}", "level": "fit", "key": key, "region": region,
                      "representation": r} for (load, region, group), keys in admitted.items() for key in keys
                     for r in phase]
        execute(fit_tasks, args.workers, "fits")
        done += phase
        for cell_key, keys in admitted.items():
            fit_sets.setdefault(cell_key, {})
            exclusions.setdefault(cell_key, {})
            for r in phase:
                fit_sets[cell_key][r], exclusions[cell_key][r] = md.representation_session_set(
                    {k: fold_records(k, cell_key[1], r) for k in keys})
        for task in build_cell_tasks(args, chosen, decoder, admitted, fit_sets, phase):
            cells.setdefault(task["id"], task)
        execute([t for t in cells.values() if t["source"] in ("native",) + phase], args.workers, "cells")
        if not finished and phase is phases[0] and "pooled" in args.regions and PRIMARY_LOAD in args.loads:
            curves = study.grid_tasks(PRIMARY_LOAD, study.CURVE_KINDS, ("radial_svm",), ("pooled",),
                                      (chosen["unit_set"],), args.n_resamples, args.n_shuffles, "timecourse")
            execute(curves, args.workers, "native-curves")
        rows, union_rows = cell_rows(list(cells.values()))
        extra = {}
        if finished:
            from_finished = [cell_source(t)[0] == args.finished_checkpoint_dir
                             for t in cells.values() if t["source"] == "native"]
            artifact["refit"].update(native_cells_read_from_finished_run=sum(from_finished),
                                     native_cells_computed=len(from_finished) - sum(from_finished))
        else:
            extra = {"native_matched_union": union_rows,
                     "native_radial_kernel_curves": study.time_course(curves, np.random.default_rng(args.seed))}
        flush(representation_sessions={f"{g}|load{l}|{reg}": {r: {"kept": len(fit_sets[(l, reg, g)][r]),
                                                                  "excluded": exclusions[(l, reg, g)][r]}
                                                               for r in done} for (l, reg, g) in admitted},
              fit_timing=fit_timing(admitted, done), cells=rows, **extra)
    flush(status="complete")


if __name__ == "__main__":
    main()
