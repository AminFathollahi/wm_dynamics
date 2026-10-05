#!/usr/bin/env python3
"""Memorandum (picture category) decoding from human medial temporal single units in DANDI 000469,
001187 and 000673.

Searches read-outs (unit set x time feature x decoder) with nested cross-validation on all patients,
then asks which latent state keeps the content of the best read-out. Content only; no behaviour enters.

Run:
    WM_DYNAMICS_DATA_ROOT=<data root> python scripts/run_memorandum_decoding_study.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
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
from provenance import (  # noqa: E402
    canonical_json, checkpoint_load, checkpoint_safe, checkpoint_store, code_identity, restore_checkpoint, sha256_file,
)
from statistics import (  # noqa: E402
    bootstrap_ci, bootstrap_ci_timecourse, minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed,
)

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "memorandum_decoding_study.json"
SUMMARY_PATH = RESULTS / "memorandum_decoding_study.md"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_memorandum_decoding_study"
SCHEMA = "memorandum_decoding_study_v1"

LOADS = (1, 3)
PRIMARY_LOAD = 1
TIE_TOLERANCE = 0.01
SHUFFLE_CHUNK = 25
LATENT_FOLDS = 5
CURVE_SHUFFLES = 20
POPULATION_SIZES = (10, 25, 50, 100, 200, None)
TRAINING_TRIALS = (4, 6, 8, 10, None)
UNIT_SETS = {
    "pseudo_population_all_units": ("pseudo", "all"),
    "pseudo_population_selective_units": ("pseudo", "selective"),
    "session_selective_units": ("session", "selective"),
    "session_all_units": ("session", "all"),
}
DECODER_SIMPLICITY = {"linear_svm": 0, "poisson_naive_bayes": 1, "radial_svm": 2}
PSEUDO_GROUPS = {"dandi_000469": ("000469",), "dandi_001187_and_000673": ("001187", "000673")}
REPRESENTATIONS = (
    "principal_components", "demixed_principal_components", "factor_analysis", "gaussian_process_factor_analysis",
    "recurrent_switching_linear_dynamics", "temporal_diffusion_embedding", "time_contrastive_embedding",
    "sequential_autoencoder",
)
DEMIXED = "demixed_principal_components"
SLOWEST_REPRESENTATION = "recurrent_switching_linear_dynamics"
PSEUDO_REPRESENTATIONS = ("principal_components", "demixed_principal_components")
CURVE_KINDS = ("maintenance_100ms", "maintenance_250ms", "encoding_100ms", "encoding_250ms",
               "transfer_100ms", "transfer_250ms")
METRICS = ("balanced_accuracy", "auc")
SELECTION_RULE = (
    "The chosen read-out is the combination of unit set, time feature and decoder with the highest median "
    f"balanced accuracy above the label-shuffle null at load {PRIMARY_LOAD}, regions pooled. Per-session unit "
    "sets: median over patients of the patient's mean over sessions. Pseudo-population unit sets: median over "
    "dataset-group x resample values. A time feature made of sliding windows is scored as the mean over its "
    f"windows. Read-outs within {TIE_TOLERANCE} of the best go to the simplest decoder (linear support vector "
    "machine, then Poisson naive Bayes, then radial-kernel support vector machine), and among those the highest "
    "median. Every other read-out is supplementary."
)

ENTRIES: dict[str, dict] = {}
FITS: dict = {}
CONFIG: dict = {}


def checkpoint_path(key: str, directory: Path | None = None) -> Path:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:80]
    return (directory or CONFIG["checkpoint_dir"]) / f"{stem}_{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def read_record(key: str, source: tuple[Path, str] | None = None):
    directory, identity = source or (None, CONFIG["identity"])
    payload = checkpoint_load(checkpoint_path(key, directory))
    if payload and payload.get("complete") and payload.get("identity") == identity:
        return restore_checkpoint(payload["record"])
    return None


def checkpointed(key: str, compute):
    record = read_record(key)
    if record is not None:
        return record
    started = time.time()
    record = compute()
    record["seconds"] = time.time() - started
    checkpoint_store(checkpoint_path(key), {"identity": CONFIG["identity"], "complete": True,
                                            "record": checkpoint_safe(record)})
    return record


def entry_key(row: dict, load: int) -> str:
    return f"{row['release']}|{row['patient']}|{row['session']}|load{load}"


def patient_of(key: str) -> str:
    return key.split("|")[1]


def build_entries(max_sessions: int | None) -> tuple[list[dict], dict]:
    manifest = md.session_manifest(ROOT / "provenance")
    listed = [row for row in manifest if row["status"] == "listed"]
    if max_sessions is not None:
        listed = listed[:: max(1, len(listed) // max_sessions)][:max_sessions]
    chosen_paths = {row["path"] for row in listed}
    accounting = []
    for row in manifest:
        record = {key: row[key] for key in ("release", "patient", "session", "status")}
        if row["status"] == "listed" and row["path"] not in chosen_paths:
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
                        ENTRIES[entry_key(row, load)] = {"entry": entry, "n_folds": folds, "n_classes": len(classes)}
                record["loads"][str(load)] = info
        accounting.append(record)
    by_status = {s: sum(1 for a in accounting if a["status"] == s) for s in sorted({a["status"] for a in accounting})}
    return accounting, {"files_seen": len(accounting), "by_status": by_status}


def inclusion_totals(accounting: list[dict]) -> dict:
    """Units admitted by the whole-task rate floor beside the count under the earlier rule (0.1 Hz in the
    maintenance window, with a minimum number of units per region-session)."""
    out = {}
    for load in LOADS:
        rows = [a["loads"][str(load)] for a in accounting if a["status"] == "loaded" and str(load) in a["loads"]]
        out[f"load{load}"] = {region: {
            "units_in_region": sum(r["unit_inclusion"][region]["units_in_region"] for r in rows),
            "units_task_rate_floor": sum(r["unit_inclusion"][region]["units_task_rate_floor"] for r in rows),
            "units_previous_rule": sum(r["unit_inclusion"][region]["units_previous_rule"] for r in rows),
            "sessions_with_at_least_8_units_task_rate_floor": sum(
                r["unit_inclusion"][region]["units_task_rate_floor"] >= md.MIN_UNITS_PER_REGION for r in rows),
            "sessions_with_previous_rule_units": sum(r["unit_inclusion"][region]["units_previous_rule"] > 0 for r in rows),
        } for region in md.REGIONS}
        out[f"load{load}"]["entry_status"] = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
        out[f"load{load}"]["sessions_tested"] = {
            "session_unit_sets_by_region": {region: len(entry_keys(load, region)) for region in md.REGIONS},
            "pseudo_populations_by_group_and_region": {group: {
                region: len(entry_keys(load, region, releases, pseudo=True)) for region in md.REGIONS}
                for group, releases in PSEUDO_GROUPS.items()}}
    return out


def entry_keys(load: int, region: str, releases: tuple[str, ...] | None = None, pseudo: bool = False) -> list[str]:
    floor = 1 if pseudo else md.MIN_UNITS_PER_REGION
    return [key for key, item in ENTRIES.items()
            if key.endswith(f"|load{load}") and (releases is None or key.split("|")[0] in releases)
            and len(md.region_columns(item["entry"], region)) >= floor and (not pseudo or item["n_classes"] == 5)]


def folds_for(keys: list[str], pseudo: bool) -> int:
    return md.PSEUDO_FOLDS if pseudo else ENTRIES[keys[0]]["n_folds"]


def grid_tasks(load: int, kinds, decoders, regions, unit_set_names, n_runs: int, n_shuffles: int, tag: str,
               **options) -> list[dict]:
    tasks = []
    for name in unit_set_names:
        level, mode = UNIT_SETS[name]
        for region in regions:
            if level == "session":
                units = [(key, [key]) for key in entry_keys(load, region)]
            else:
                units = [(f"{group}|load{load}", keys) for group, releases in PSEUDO_GROUPS.items()
                         if (keys := entry_keys(load, region, releases, pseudo=True))]
            for unit, keys in units:
                tasks.append({"id": f"{tag}|{name}|{unit}|{region}", "tag": tag, "unit_set": name, "level": level,
                              "mode": mode, "unit": unit, "keys": keys, "region": region, "kinds": kinds,
                              "decoders": decoders, "n_runs": 1 if level == "session" else n_runs,
                              "n_shuffles": n_shuffles, "options": options})
    return tasks


def run_read_out_task(task: dict) -> None:
    pseudo = task["level"] == "pseudo"
    entries = [ENTRIES[k]["entry"] for k in task["keys"]]
    n_folds = folds_for(task["keys"], pseudo)
    classes = np.unique(np.concatenate([e["labels"] for e in entries]))
    seed = stable_seed(f"memorandum_decoding|{task['id']}")

    def make(shuffled):
        if pseudo:
            return md.pseudo_make_folds(entries, task["region"], task["mode"], task["kinds"], n_folds, shuffled,
                                        **task["options"])
        return md.session_make_folds(entries[0], task["region"], task["mode"], task["kinds"], n_folds, shuffled)

    observed = checkpointed(task["id"] + "|observed", lambda: md.observed_runs(
        make(False), task["kinds"], task["decoders"], classes, task["n_runs"], seed))
    for start in range(0, task["n_shuffles"], SHUFFLE_CHUNK):
        indices = range(start, min(start + SHUFFLE_CHUNK, task["n_shuffles"]))
        checkpointed(f"{task['id']}|shuffled{start}", lambda: {"scores": md.shuffled_runs(
            make(True), task["kinds"], task["decoders"], classes, observed["tuned"], indices, seed)})


def latent_fits() -> dict:
    import run_human_representation_benchmark as human
    import run_info_benchmark as benchmark_core

    def candidate(name):
        return lambda train, test, labels, seed: benchmark_core.fit_candidate(train, test, name, seed)

    fits = {name: candidate(name) for name in REPRESENTATIONS if name != "demixed_principal_components"}
    fits["demixed_principal_components"] = lambda train, test, labels, seed: human.fit_demixed_principal_components(
        train, labels, test, human.OPERATING_RANK, np.random.default_rng(seed))
    return fits


def run_latent_task(task: dict) -> None:
    entries = [ENTRIES[k]["entry"] for k in task["keys"]]
    seed = stable_seed(f"memorandum_decoding|{task['pair_id']}")

    def compute():
        if task["level"] == "latent":
            out = md.latent_cell(entries[0], task["region"], task["mode"], (task["representation"],), FITS,
                                 task["kind"], task["decoder"], LATENT_FOLDS, CONFIG["n_shuffles"], seed)
        else:
            out = md.pseudo_latent_cell(entries, task["region"], task["mode"], (task["representation"],), FITS,
                                        task["kind"], task["decoder"], folds_for(task["keys"], True),
                                        CONFIG["n_resamples"], CONFIG["n_shuffles"], seed)
        return out[task["representation"]]

    checkpointed(task["id"], compute)


def run_task(task: dict) -> str:
    (run_latent_task if task["level"].startswith("latent") else run_read_out_task)(task)
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


def load_task_scores(task: dict) -> tuple[dict, dict] | None:
    observed = read_record(task["id"] + "|observed")
    chunks = [read_record(f"{task['id']}|shuffled{s}") for s in range(0, task["n_shuffles"], SHUFFLE_CHUNK)]
    if observed is None or any(c is None for c in chunks):
        return None
    null = {d: {k: {m: np.concatenate([c["scores"][d][k][m] for c in chunks]) for m in METRICS}
                for k in observed["scores"][d]} for d in observed["scores"]}
    return observed, null


def above_null(observed: dict, null: dict, decoder: str, kind: str, metric: str) -> dict:
    obs = observed["scores"][decoder][kind][metric]
    null_per_window = null[decoder][kind][metric].mean(axis=0)
    return {"observed_runs": obs, "null_per_window": null_per_window, "per_window": obs.mean(axis=0) - null_per_window,
            "per_run": obs.mean(axis=1) - null_per_window.mean()}


def read_out_rows(tasks: list[dict]) -> list[dict]:
    rows = []
    for task in tasks:
        loaded = load_task_scores(task)
        if loaded is None:
            continue
        observed, null = loaded
        for decoder in task["decoders"]:
            for kind in task["kinds"]:
                row = {"unit_set": task["unit_set"], "level": task["level"], "unit": task["unit"],
                       "region": task["region"], "decoder": decoder, "time_feature": kind,
                       "n_runs": task["n_runs"], "n_shuffles": task["n_shuffles"],
                       "mean_units_per_fold": float(np.mean(observed["mean_units"])),
                       "mean_dropped_units_per_fold": {r: float(np.mean([d[r] for d in observed["mean_dropped_units"]]))
                                                       for r in observed["mean_dropped_units"][0]},
                       "task_seconds": observed["seconds"]}
                for metric in METRICS:
                    a = above_null(observed, null, decoder, kind, metric)
                    row[metric] = {"observed": float(a["observed_runs"].mean()),
                                   "null_mean": float(a["null_per_window"].mean()),
                                   "above_null": float(a["per_run"].mean()), "per_run_above_null": a["per_run"].tolist()}
                rows.append(row)
    return rows


def summarise_read_outs(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["unit_set"], row["region"], row["decoder"], row["time_feature"]), []).append(row)
    summary = []
    for (name, region, decoder, kind), members in sorted(groups.items()):
        out = {"unit_set": name, "region": region, "decoder": decoder, "time_feature": kind, "load": PRIMARY_LOAD}
        per_session = UNIT_SETS[name][0] == "session"
        if per_session:
            out.update(n_sessions=len(members), n_patients=len({patient_of(m["unit"]) for m in members}))
        else:
            out.update(n_groups=len(members), n_resamples=int(members[0]["n_runs"]),
                       group_means={m["unit"]: m["balanced_accuracy"]["above_null"] for m in members})
        for metric in METRICS:
            if per_session:
                by_patient = {}
                for m in members:
                    by_patient.setdefault(patient_of(m["unit"]), []).append(m[metric]["above_null"])
                values = np.array([np.mean(v) for v in by_patient.values()])
            else:
                values = np.concatenate([m[metric]["per_run_above_null"] for m in members])
            _, low, high = bootstrap_ci(values, np.median, n_boot=2000, rng=rng)
            out[metric] = {"median_above_null": float(np.median(values)), "mean_above_null": float(values.mean()),
                           "interval_95_of_median": [low, high],
                           "values_are": "patient means" if per_session else "dataset-group x resample values"}
        summary.append(out)
    return summary


def choose_read_out(summary: list[dict]) -> dict | None:
    pooled = [s for s in summary if s["region"] == "pooled"]
    if not pooled:
        return None
    best = max(s["balanced_accuracy"]["median_above_null"] for s in pooled)
    eligible = [s for s in pooled if s["balanced_accuracy"]["median_above_null"] >= best - TIE_TOLERANCE]
    return min(eligible, key=lambda s: (DECODER_SIMPLICITY[s["decoder"]], -s["balanced_accuracy"]["median_above_null"]))


def latent_decoder(chosen: dict, summary: list[dict]) -> tuple[str, str | None]:
    if chosen["decoder"] != "poisson_naive_bayes":
        return chosen["decoder"], None
    peers = [s for s in summary if s["unit_set"] == chosen["unit_set"] and s["region"] == "pooled"
             and s["time_feature"] == chosen["time_feature"] and s["decoder"] != "poisson_naive_bayes"]
    best = max(peers, key=lambda s: s["balanced_accuracy"]["median_above_null"])
    return best["decoder"], ("Poisson naive Bayes needs counts and cannot read a continuous latent state; the better "
                             "of the two support vector machines at the same unit set and time feature is used")


def follow_up_tasks(chosen: dict, n_resamples: int, args) -> list[dict]:
    kind, decoder = chosen["time_feature"], chosen["decoder"]
    tasks = []
    for name in ("pseudo_population_all_units", "pseudo_population_selective_units"):
        for size in POPULATION_SIZES:
            tasks += grid_tasks(PRIMARY_LOAD, (kind,), (decoder,), ("pooled",), (name,), n_resamples, CURVE_SHUFFLES,
                                f"size{size}", max_units=size)
        for count in TRAINING_TRIALS:
            tasks += grid_tasks(PRIMARY_LOAD, (kind,), (decoder,), ("pooled",), (name,), n_resamples, CURVE_SHUFFLES,
                                f"training{count}", max_train_per_class=count)
    tasks += grid_tasks(PRIMARY_LOAD, CURVE_KINDS, (decoder,), args.regions, (chosen["unit_set"],), n_resamples,
                        args.n_shuffles, "timecourse")
    tasks += grid_tasks(3, (kind,), (decoder,), args.regions, (chosen["unit_set"],), n_resamples, args.n_shuffles,
                        "load3")
    return tasks


def latent_tasks(chosen: dict, decoder: str, args) -> list[dict]:
    mode = UNIT_SETS[chosen["unit_set"]][1]
    tasks = []
    for load in args.loads:
        for region in args.regions:
            groups = [(key, "latent", [key]) for key in entry_keys(load, region)]
            groups += [(f"{group}|load{load}", "latent_pseudo", keys) for group, releases in PSEUDO_GROUPS.items()
                       if (keys := entry_keys(load, region, releases, pseudo=True))]
            for unit, level, keys in groups:
                allowed = ("native",) + (tuple(args.representations) if level == "latent" else
                                         tuple(r for r in PSEUDO_REPRESENTATIONS if r in args.representations))
                for representation in allowed:
                    pair_id = f"{level}|{mode}|{unit}|{region}"
                    tasks.append({"id": f"{pair_id}|{representation}", "pair_id": pair_id, "level": level,
                                  "mode": mode, "unit": unit, "keys": keys, "region": region, "load": load,
                                  "representation": representation, "kind": chosen["time_feature"],
                                  "decoder": decoder})
    return sorted(tasks, key=lambda t: t["representation"] == SLOWEST_REPRESENTATION)


def curve_summary(tasks: list[dict]) -> dict:
    out = {"population_size": {}, "training_trials_per_category": {}}
    for task in tasks:
        loaded = load_task_scores(task) if task["tag"].startswith(("size", "training")) else None
        if loaded is None:
            continue
        observed, null = loaded
        decoder, kind = task["decoders"][0], task["kinds"][0]
        family, label = ("population_size", task["tag"][4:]) if task["tag"].startswith("size") else (
            "training_trials_per_category", task["tag"][8:])
        cell = {}
        for metric in METRICS:
            a = above_null(observed, null, decoder, kind, metric)
            cell[metric] = {"observed_mean": float(a["observed_runs"].mean()), "null_mean": float(a["null_per_window"].mean()),
                            "above_null": float(a["per_run"].mean()),
                            "draw_interval_95": np.percentile(a["per_run"], [2.5, 97.5]).tolist()}
        cell.update(n_draws=task["n_runs"], n_shuffles=task["n_shuffles"], mean_units=float(np.mean(observed["mean_units"])))
        out[family].setdefault(task["unit_set"], {}).setdefault(task["unit"], {})["all" if label == "None" else label] = cell
    return out


def time_course(tasks: list[dict], rng: np.random.Generator) -> list[dict]:
    rows, per_patient = [], {}
    for task in tasks:
        loaded = load_task_scores(task) if task["tag"] in ("timecourse", "load3") else None
        if loaded is None:
            continue
        observed, null = loaded
        for kind in task["kinds"]:
            spec = md.FEATURES[md.KINDS[kind][1]]
            base = {"tag": task["tag"], "unit_set": task["unit_set"], "region": task["region"],
                    "decoder": task["decoders"][0], "time_feature": kind, "window_start_s": spec["starts"].tolist(),
                    "window_width_s": spec["width"], "aligned_to": f"{spec['align']} onset"}
            curves = {m: above_null(observed, null, task["decoders"][0], kind, m) for m in METRICS}
            if task["level"] == "pseudo":
                rows.append({**base, "unit": task["unit"], "n_resamples": task["n_runs"], **{m: {
                    "above_null_per_window": c["per_window"].tolist(),
                    "resample_interval_95": np.percentile(c["observed_runs"] - c["null_per_window"][None, :],
                                                          [2.5, 97.5], axis=0).tolist()} for m, c in curves.items()}})
            else:
                per_patient.setdefault((task["tag"], task["unit_set"], task["region"], task["decoders"][0], kind, base["window_width_s"]), {}).setdefault(
                    patient_of(task["unit"]), []).append({m: c["per_window"] for m, c in curves.items()} | {"base": base})
    for patients in per_patient.values():
        base = next(iter(patients.values()))[0]["base"]
        row = {**base, "unit": "patient_mean", "n_patients": len(patients)}
        for metric in METRICS:
            data = np.array([np.mean([s[metric] for s in sessions], axis=0) for sessions in patients.values()])
            mean, low, high = bootstrap_ci_timecourse(data, lambda x: x.mean(axis=0), n_boot=1000, rng=rng)
            row[metric] = {"above_null_per_window": mean.tolist(), "patient_interval_95": [low.tolist(), high.tolist()]}
        rows.append(row)
    return rows


def latent_summary(tasks: list[dict], rng: np.random.Generator, decoder: str, reader=read_record) -> dict:
    scores, failures = {}, {}
    for task in tasks:
        record = reader(task["id"])
        if record is None:
            continue
        if record["status"] != "computed":
            counts = failures.setdefault((task["level"], task["region"], task["representation"]), {})
            reason = str(record["reason"])[:200]
            counts[reason] = counts.get(reason, 0) + 1
            continue
        for metric in METRICS:
            observed = record["observed"][decoder][task["kind"]][metric]
            null = record["null"][decoder][task["kind"]][metric]
            scores[(task["level"], task["load"], task["region"], task["representation"], metric, task["unit"])] = float(
                observed.mean() - null.mean(axis=0).mean())

    def by_cluster(level, values):
        clusters = {}
        for unit, value in values.items():
            clusters.setdefault(patient_of(unit) if level == "latent" else unit, []).append(value)
        return np.array([np.mean(v) for v in clusters.values()])

    out = {"absolute": [], "latent_minus_native": [], "not_estimable": [
        {"level": k[0], "region": k[1], "representation": k[2], "reasons": v} for k, v in sorted(failures.items())]}
    for level, load, region, representation, metric in sorted({k[:5] for k in scores}):
        mine = {k[5]: v for k, v in scores.items() if k[:5] == (level, load, region, representation, metric)}
        native = {k[5]: v for k, v in scores.items() if k[:5] == (level, load, region, "native", metric)}
        clusters = by_cluster(level, mine)
        out["absolute"].append({"level": level, "load": load, "region": region, "representation": representation,
                                "metric": metric, "mean_above_null": float(clusters.mean()),
                                "median_above_null": float(np.median(clusters)), "n_entries": len(mine),
                                "n_patients_or_groups": len(clusters)})
        shared = sorted(set(mine) & set(native))
        if representation == "native" or not shared:
            continue
        diffs = by_cluster(level, {u: mine[u] - native[u] for u in shared})
        row = {"level": level, "load": load, "region": region, "representation": representation, "metric": metric,
               "n_entries": len(shared), "n_patients_or_groups": len(diffs), "mean_difference": float(diffs.mean())}
        if len(diffs) >= 3:
            _, low, high = bootstrap_ci(diffs, np.mean, n_boot=5000, rng=rng)
            flip = paired_sign_flip_test(diffs, np.zeros_like(diffs), alternative="two-sided", rng=rng)
            row.update(interval_95_patient_bootstrap=[low, high], sign_flip_p=float(flip["p_value"]),
                       minimum_detectable_difference=minimum_detectable_paired_difference(diffs).get("mdd"))
        out["latent_minus_native"].append(row)
    return out


def render_summary(artifact: dict) -> str:
    lines = ["# Memorandum decoding study", "", f"status: {artifact['status']}", "",
             f"Selection rule: {artifact['selection_rule']}", ""]
    chosen = artifact.get("chosen_read_out")
    if chosen:
        low, high = chosen["balanced_accuracy"]["interval_95_of_median"]
        lines += ["## Chosen read-out", "",
                  f"unit set {chosen['unit_set']}, time feature {chosen['time_feature']}, decoder {chosen['decoder']}: "
                  f"median balanced accuracy above the label-shuffle null "
                  f"{chosen['balanced_accuracy']['median_above_null']:.4f} (interval of the median {low:.4f} to "
                  f"{high:.4f}); median one-versus-rest area under the curve above null "
                  f"{chosen['auc']['median_above_null']:.4f}", ""]
    lines += ["## Read-outs at load 1, regions pooled", "",
              "| unit set | time feature | decoder | balanced accuracy above null | area under curve above null | role |",
              "|---|---|---|---|---|---|"]
    for s in artifact.get("read_out_search", {}).get("summary", []):
        if s["region"] == "pooled":
            lines.append(f"| {s['unit_set']} | {s['time_feature']} | {s['decoder']} | "
                         f"{s['balanced_accuracy']['median_above_null']:.4f} | {s['auc']['median_above_null']:.4f} | {s['role']} |")
    lines += ["", "## Latent minus native", "",
              "| level | load | region | representation | metric | mean difference | interval | p | patients or groups | entries |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in artifact.get("latent_state", {}).get("latent_minus_native", []):
        interval = r.get("interval_95_patient_bootstrap")
        text = "" if interval is None else f"{interval[0]:.4f} to {interval[1]:.4f}"
        lines.append(f"| {r['level']} | {r['load']} | {r['region']} | {r['representation']} | {r['metric']} | "
                     f"{r['mean_difference']:.4f} | {text} | {r.get('sign_flip_p', '')} | "
                     f"{r['n_patients_or_groups']} | {r['n_entries']} |")
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Memorandum decoding study.")
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--max-sessions", type=int)
    p.add_argument("--n-resamples", type=int, default=20)
    p.add_argument("--n-shuffles", type=int, default=100)
    p.add_argument("--selection-permutations", type=int, default=md.SELECTION_PERMUTATIONS)
    p.add_argument("--regions", nargs="+", choices=md.REGIONS, default=list(md.REGIONS))
    p.add_argument("--loads", nargs="+", type=int, default=list(LOADS))
    p.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demixed-refit-of", type=Path, metavar="DELIVERED_JSON",
                   help="refit only the demixed principal component latent tasks of this finished artifact; every "
                        "other task is read from --delivered-checkpoint-dir under the artifact's own checkpoint identity")
    p.add_argument("--delivered-checkpoint-dir", type=Path)
    p.add_argument("--demixed-tasks-limit", type=int, help="with --demixed-refit-of: score only the first N demixed tasks")
    p.add_argument("--demixed-task-level", choices=("latent", "latent_pseudo"),
                   help="with --demixed-refit-of: score only the per-session (latent) or the pooled (latent_pseudo) tasks")
    p.add_argument("--check-reproduction", action="store_true",
                   help="with --demixed-refit-of: score nothing and read the demixed tasks from the finished "
                        "checkpoints too, to confirm the finished latent summary is reproduced exactly")
    return p


def refit_demixed_latent_tasks(args, rng: np.random.Generator, flush) -> None:
    """Refits only the demixed latent tasks of a finished artifact. The finished artifact's search summary and
    time course are recomputed from its own checkpoints first, so the generator reaches the latent summary in the
    state it had there, and every other representation's task is read from those checkpoints."""
    delivered = json.loads(args.demixed_refit_of.read_text())
    for name, value in (("n_resamples", args.n_resamples), ("n_shuffles", args.n_shuffles), ("seed", args.seed),
                        ("selection_permutations", args.selection_permutations), ("regions", args.regions)):
        if delivered["scope"][name] != value:
            raise SystemExit(f"--{name.replace('_', '-')} differs from the finished artifact ({delivered['scope'][name]})")
    source = (args.delivered_checkpoint_dir, delivered["checkpoint_identity"])
    summary = summarise_read_outs(delivered["read_out_search"]["per_unit_rows"], rng)
    chosen = choose_read_out(summary)
    for s in summary:
        s["role"] = "chosen" if s is chosen else "supplementary"
    read_delivered = lambda key: read_record(key, source)
    CONFIG["identity"], fresh_identity = delivered["checkpoint_identity"], CONFIG["identity"]
    CONFIG["checkpoint_dir"], fresh_dir = args.delivered_checkpoint_dir, CONFIG["checkpoint_dir"]
    follow_up = follow_up_tasks(chosen, args.n_resamples, args)
    course = time_course(follow_up, rng)
    CONFIG["identity"], CONFIG["checkpoint_dir"] = fresh_identity, fresh_dir
    stream_reproduced = {
        "read_out_summary": canonical_json(summary) == canonical_json(delivered["read_out_search"]["summary"]),
        "time_course": canonical_json(course) == canonical_json(delivered["time_course"]),
    }
    decoder, _ = latent_decoder(chosen, summary)
    args.representations = [r for r in delivered["scope"]["representations"] if r != "native"]
    latent = latent_tasks(chosen, decoder, args)
    demixed = [t for t in latent if t["representation"] == DEMIXED
               and args.demixed_task_level in (None, t["level"])]
    if args.demixed_tasks_limit is not None:
        demixed = demixed[: args.demixed_tasks_limit]
    if args.check_reproduction:
        demixed = []
    FITS.update(latent_fits())
    execute(demixed, args.workers, "demixed latent")
    scored = {t["id"] for t in demixed}
    reader = lambda key: read_record(key) if key in scored else read_delivered(key)
    done = {t["id"] for t in latent} if args.check_reproduction else scored
    flush(refit={"of_artifact": str(args.demixed_refit_of), "of_artifact_sha256": sha256_file(args.demixed_refit_of),
                 "tasks_scored": len(demixed), "tasks_limit": args.demixed_tasks_limit,
                 "check_reproduction": args.check_reproduction,
                 "generator_state_reproduced": stream_reproduced},
          latent_state={"decoder": decoder, "decoder_note": delivered["latent_state"]["decoder_note"],
                        "unit_set_mode": UNIT_SETS[chosen["unit_set"]][1], "time_feature": chosen["time_feature"],
                        **latent_summary([t for t in latent if t["id"] in done or t["representation"] != DEMIXED],
                                         rng, decoder, reader)},
          status="complete")


def main() -> None:
    args = parser().parse_args()
    if args.n_shuffles < 1 or args.n_resamples < 1:
        raise SystemExit("--n-shuffles and --n-resamples must be positive")
    started = time.time()
    md.SELECTION_PERMUTATIONS = args.selection_permutations
    identity = code_identity(ROOT, Path(__file__))
    CONFIG.update(checkpoint_dir=args.checkpoint_dir, n_shuffles=args.n_shuffles, n_resamples=args.n_resamples,
                  identity=hashlib.sha256(canonical_json({
                      "code": identity, "selection_permutations": args.selection_permutations,
                      "n_shuffles": args.n_shuffles, "n_resamples": args.n_resamples, "seed": args.seed}).encode()).hexdigest())
    accounting, reconciliation = build_entries(args.max_sessions)
    rng = np.random.default_rng(args.seed)
    artifact = {
        "version": SCHEMA, "status": "running", "code_identity": identity, "checkpoint_identity": CONFIG["identity"],
        "selection_rule": SELECTION_RULE,
        "scope": {"datasets": list(md.RELEASES), "loads": list(LOADS), "primary_load": PRIMARY_LOAD,
                  "regions": args.regions, "task_rate_floor_hz": md.TASK_RATE_FLOOR_HZ,
                  "selection_window_s": list(md.SELECTION_WINDOW_S),
                  "selection_permutations": args.selection_permutations, "selection_level": md.SELECTION_LEVEL,
                  "n_resamples": args.n_resamples, "n_shuffles": args.n_shuffles, "curve_shuffles": CURVE_SHUFFLES,
                  "pseudo_population_scheme": {
                      "outer_folds": md.PSEUDO_FOLDS, "pseudo_trials_per_category_training": md.PSEUDO_TRAIN_PER_CATEGORY,
                      "pseudo_trials_per_category_test": md.PSEUDO_TEST_PER_CATEGORY,
                      "drawn": "with replacement, category-matched, independently for every unit, from that unit's own trials",
                      "unit_dropped_from_a_fold_when": "its session has fewer than 1 test trial or fewer than 2 training trials in some category",
                      "labels_shuffled": "within each session before splitting and assembly"},
                  "patient_split": "none; every read-out is scored on all patients with nested cross-validation",
                  "nested_cross_validation": {"inner_folds": md.INNER_FOLDS, "grids": {
                      k: [list(v) if isinstance(v, tuple) else v for v in g] for k, g in md.HYPER_GRIDS.items()}},
                  "shuffled_label_hyperparameters": "chosen on the observed labels, reused in the shuffled-label runs",
                  "latent_folds": LATENT_FOLDS, "latent_dimension_rule": "benchmark operating rank, clipped by unit count",
                  "representations": ["native"] + list(args.representations),
                  "left_out": {"neural_data_transformer": "needs the accelerator, which was in use"},
                  "pseudo_population_groups": {k: list(v) for k, v in PSEUDO_GROUPS.items()},
                  "seed": args.seed, "max_sessions": args.max_sessions},
        "session_accounting": accounting, "session_reconciliation": reconciliation,
        "unit_inclusion_totals": inclusion_totals(accounting)}

    def flush(**extra):
        artifact.update(extra)
        artifact["wall_clock_s"] = time.time() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_json(artifact))
        args.summary.write_text(render_summary(artifact))

    flush()
    if args.demixed_refit_of is not None:
        refit_demixed_latent_tasks(args, rng, flush)
        return
    search = grid_tasks(PRIMARY_LOAD, md.GRID_KINDS, md.DECODERS, args.regions, tuple(UNIT_SETS), args.n_resamples,
                        args.n_shuffles, "search")
    execute(search, args.workers, "search")
    rows = read_out_rows(search)
    summary = summarise_read_outs(rows, rng)
    chosen = choose_read_out(summary)
    for s in summary:
        s["role"] = "chosen" if s is chosen else "supplementary"
    flush(chosen_read_out=chosen, read_out_search={"summary": summary, "per_unit_rows": rows})
    if chosen is None:
        flush(status="no_read_out_scored")
        return
    decoder, note = latent_decoder(chosen, summary)
    follow_up = follow_up_tasks(chosen, args.n_resamples, args)
    execute(follow_up, args.workers, "follow-up")
    latent = latent_tasks(chosen, decoder, args)
    FITS.update(latent_fits())
    execute(latent, args.workers, "latent")
    flush(curves=curve_summary(follow_up), time_course=time_course(follow_up, rng),
          latent_state={"decoder": decoder, "decoder_note": note, "unit_set_mode": UNIT_SETS[chosen["unit_set"]][1],
                        "time_feature": chosen["time_feature"], **latent_summary(latent, rng, decoder)},
          status="complete")


if __name__ == "__main__":
    main()
