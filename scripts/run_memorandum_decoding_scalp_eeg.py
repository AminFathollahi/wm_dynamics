#!/usr/bin/env python3
"""Per-subject decoding of the remembered orientation and the cued item from scalp EEG in the two impulse-perturbation
experiments (dataset ids wolff_eeg_impulse experiment_1 and experiment_2), each experiment its own dataset group.

Features are channel means over windows of the corpus loader's epochs (channel-mean-centred, bad trials removed by
the loader's mask). The decoder is the memorandum decoding study's linear support vector machine with nested
cross-validation inside each subject; the across-subject summary is the mean over subjects with a subject-bootstrap
interval and a sign-flip p, both as numbers.

Run:
    WM_DYNAMICS_DATA_ROOT=<data root> python scripts/run_memorandum_decoding_scalp_eeg.py
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing
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
import run_wolff_impulse_pipeline as impulse  # noqa: E402
import scalp_memorandum_decoding as scalp  # noqa: E402
from project_config import dataset_path  # noqa: E402
from provenance import canonical_json, code_identity  # noqa: E402
from statistics import (  # noqa: E402
    bootstrap_ci_timecourse, minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed,
)

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "memorandum_decoding_scalp_eeg.json"
SUMMARY_PATH = RESULTS / "memorandum_decoding_scalp_eeg.md"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_memorandum_decoding_scalp_eeg"
SCHEMA = "memorandum_decoding_scalp_eeg_v1"

GROUPS = ("window_means", "course_100ms", "course_250ms", "transfer_100ms", "transfer_250ms")
METRICS = ("balanced_accuracy", "auc")
CHANCE_AUC = 0.5
CHANNEL_NOTE = ("Scalp channels are mixtures of cortical sources, so a selective channel is not a selective neuron "
                "and a channel-level read-out is not a population of units.")
BOTH_REPRESENTATIONS = ("native", "native_selected", "principal_components", "demixed_principal_components")
EXPERIMENTS = {
    "experiment_1": {
        "dataset_group": "wolff_eeg_impulse_experiment_1", "glob": "Dynamic_hidden_states_exp1_*.mat",
        "variable": "exp1_data",
        "label_definitions": {
            "orientation": "remembered orientation: angle of the cued item (cue 1 = left, 2 = right), binned into six "
                           "30-degree bins over [-pi/2, pi/2)",
            "cued_item": "cue field: 1 = left item, 2 = right item"},
        "epochs": {"encoding": "EEG_mem_items", "cue": "EEG_cue", "impulse": "EEG_impulse"},
        "windows_s": {"encoding": impulse.ENCODING_WINDOW_S, "pre_perturbation": impulse.PRE_PERTURBATION_WINDOW_S,
                      "post_perturbation": impulse.POST_PERTURBATION_WINDOW_S},
        "spec": {
            "encoding_window": ("encoding", "encoding"),
            "fixed": {"encoding": ("encoding", "encoding"), "maintenance_pre_perturbation": ("cue", "pre_perturbation"),
                      "impulse_evoked": ("impulse", "post_perturbation")},
            "course_epochs": ["encoding", "cue", "impulse"],
            "label_vectors": {"orientation": dict.fromkeys(("encoding", "cue", "impulse"), "orientation"),
                              "cued_item": dict.fromkeys(("encoding", "cue", "impulse"), "cued_item")},
            "representations": {"orientation": BOTH_REPRESENTATIONS,
                                "cued_item": ("native", "native_selected", "principal_components")}}},
    "experiment_2": {
        "dataset_group": "wolff_eeg_impulse_experiment_2", "glob": "Dynamic_hidden_states_exp2_*.mat",
        "variable": "exp2_data",
        "label_definitions": {
            "orientation": "orientation of the early-tested item (encoding and first impulse) or of the late-tested "
                           "item (second impulse), binned into six 30-degree bins over [-pi/2, pi/2)",
            "cued_item": "not defined in this experiment: no retro-cue is presented"},
        "epochs": {"encoding": ["EEG_mem_items_sess1", "EEG_mem_items_sess2"],
                   "impulse_early": ["EEG_impulse1_sess1", "EEG_impulse1_sess2"],
                   "impulse_late": ["EEG_impulse2_sess1", "EEG_impulse2_sess2"]},
        "windows_s": {"encoding": impulse.ENCODING_WINDOW_S, "post_perturbation": impulse.POST_PERTURBATION_WINDOW_S},
        "spec": {
            "encoding_window": ("encoding", "encoding"),
            "fixed": {"encoding": ("encoding", "encoding"), "impulse_evoked_early": ("impulse_early", "post_perturbation"),
                      "impulse_evoked_late": ("impulse_late", "post_perturbation")},
            "course_epochs": ["encoding", "impulse_early", "impulse_late"],
            "label_vectors": {"orientation": {"encoding": "orientation_early", "impulse_early": "orientation_early",
                                              "impulse_late": "orientation_late"}},
            "representations": {"orientation": BOTH_REPRESENTATIONS}}},
}
CONFIG = study.CONFIG


def epoch_features(epoch, n_trials: int, window_keys: dict) -> dict:
    valid = impulse.valid_trial_mask(epoch, n_trials)
    n_channels = np.asarray(epoch.trial).shape[1]
    fixed = {}
    for key, window in window_keys.items():
        values = np.full((n_trials, n_channels), np.nan)
        values[valid] = impulse.window_features(epoch, window, valid)[:, :, 0]
        fixed[key] = values
    trial = np.asarray(epoch.trial, dtype=float)
    trial = trial - trial.mean(axis=1, keepdims=True)
    time_axis = np.asarray(epoch.time, dtype=float)
    duration = time_axis[-1] + float(np.median(np.diff(time_axis)))
    starts = {w: scalp.course_starts(duration, w / 1000) for w in scalp.COURSE_WIDTHS_MS}
    course = {w: scalp.window_means(trial, time_axis, starts[w], w / 1000) for w in scalp.COURSE_WIDTHS_MS}
    return {"valid": valid, "fixed": fixed, "course": course, "course_starts": starts}


def join_sessions(parts: list[dict]) -> dict:
    return {"valid": np.concatenate([p["valid"] for p in parts]),
            "fixed": {k: np.concatenate([p["fixed"][k] for p in parts]) for k in parts[0]["fixed"]},
            "course": {w: np.concatenate([p["course"][w] for p in parts]) for w in parts[0]["course"]},
            "course_starts": parts[0]["course_starts"]}


def load_bundle(path: Path, experiment: str) -> dict:
    definition = EXPERIMENTS[experiment]
    data = impulse._load_mat(path)[definition["variable"]]
    keys = {"encoding": {"encoding": impulse.ENCODING_WINDOW_S}, "cue": {"pre_perturbation": impulse.PRE_PERTURBATION_WINDOW_S},
            "impulse": {"post_perturbation": impulse.POST_PERTURBATION_WINDOW_S},
            "impulse_early": {"post_perturbation": impulse.POST_PERTURBATION_WINDOW_S},
            "impulse_late": {"post_perturbation": impulse.POST_PERTURBATION_WINDOW_S}}
    if experiment == "experiment_1":
        results = np.asarray(data.Results, dtype=float)
        n_trials = len(results)
        labels = {"orientation": impulse.bin_orientation(np.where(results[:, 2] == 1, results[:, 0], results[:, 1])),
                  "cued_item": results[:, 2].astype(int)}
        epochs = {name: epoch_features(getattr(data, field), n_trials, keys[name])
                  for name, field in definition["epochs"].items()}
    else:
        sessions = [np.asarray(getattr(data, f"Results_sess{s}"), dtype=float) for s in (1, 2)]
        labels = {"orientation_early": np.concatenate([impulse.bin_orientation(r[:, 0]) for r in sessions]),
                  "orientation_late": np.concatenate([impulse.bin_orientation(r[:, 1]) for r in sessions])}
        epochs = {name: join_sessions([epoch_features(getattr(data, field), len(r), keys[name])
                                       for field, r in zip(fields, sessions)])
                  for name, fields in definition["epochs"].items()}
    return {"labels": labels, "epochs": epochs, "n_trials": int(len(next(iter(labels.values()))))}


def subject_status(bundle: dict, spec: dict) -> dict:
    """Trial counts and the reason a subject cannot be cross-validated, if any."""
    enc = bundle["epochs"][spec["encoding_window"][0]]["valid"]
    info = {"n_trials": bundle["n_trials"],
            "valid_trials_by_epoch": {k: int(e["valid"].sum()) for k, e in bundle["epochs"].items()},
            "trials_per_class_in_encoding_valid": {}}
    reasons = []
    for name, y in bundle["labels"].items():
        classes, counts = np.unique(y[enc], return_counts=True)
        info["trials_per_class_in_encoding_valid"][name] = {str(c): n for c, n in zip(classes.tolist(), counts.tolist())}
        if md.feasible_fold_count(y[enc]) is None:
            reasons.append(f"{name}: a class has fewer than {md.MIN_CLASS_TRIALS} valid encoding trials")
    return {**info, "status": "excluded" if reasons else "computed", "reasons": reasons}


def subject_files(experiment: str, max_subjects: int | None) -> list[Path]:
    files = sorted(dataset_path("wolff_eeg_impulse").glob(EXPERIMENTS[experiment]["glob"]),
                   key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    return files if max_subjects is None else files[:max_subjects]


def group_key(experiment: str, subject: str, group: str) -> str:
    return f"scalp|{experiment}|{subject}|{group}"


def run_subject_task(task: dict) -> str:
    experiment, subject = task["experiment"], task["subject"]
    spec = EXPERIMENTS[experiment]["spec"]
    seed = stable_seed(f"scalp_memorandum|{experiment}|{subject}")
    keys = {g: group_key(experiment, subject, g) for g in task["groups"]}
    status_key = group_key(experiment, subject, "status")
    status = study.read_record(status_key)
    if status is not None and (status["status"] == "excluded" or all(study.read_record(k) is not None for k in keys.values())):
        return task["id"]
    last_error = None
    for _ in range(3):
        try:
            bundle = load_bundle(task["path"], experiment)
            break
        except OSError as error:
            last_error = error
    else:
        study.checkpointed(status_key, lambda: {"status": "excluded", "reasons": [f"load_error: {last_error}"[:200]]})
        return task["id"]
    status = study.checkpointed(status_key, lambda: subject_status(bundle, spec))
    if status["status"] == "excluded":
        return task["id"]
    for group, key in keys.items():
        shuffles = CONFIG["n_shuffles"] if group == "window_means" else CONFIG["n_shuffles_time_course"]
        study.checkpointed(key, lambda: {"readouts": scalp.decode_group(bundle, spec, group, shuffles, seed,
                                                                          md.SELECTION_PERMUTATIONS)})
    return task["id"]


def execute(tasks: list[dict], workers: int) -> None:
    print(f"subjects: {len(tasks)} tasks", flush=True)
    started = time.time()
    pool = multiprocessing.get_context("fork").Pool(workers) if workers > 1 else None
    for done, task_id in enumerate(map(run_subject_task, tasks) if pool is None else pool.imap_unordered(run_subject_task, tasks), 1):
        print(f"subjects {done}/{len(tasks)} {time.time() - started:.0f}s {task_id}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()


def across_subjects(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """Mean over subjects per window with a subject-bootstrap 95% interval, the two-sided sign-flip p and the
    minimum detectable difference of the subject values."""
    mean, low, high = bootstrap_ci_timecourse(values, lambda x: x.mean(axis=0), n_boot=n_boot, rng=rng)
    p = [float(paired_sign_flip_test(values[:, w], np.zeros(len(values)), alternative="two-sided", n_boot=1,
                                     rng=rng)["p_value"]) for w in range(values.shape[1])]
    return {"mean": mean.tolist(), "interval_95": [low.tolist(), high.tolist()], "p": p,
            "minimum_detectable_difference": [minimum_detectable_paired_difference(values[:, w]).get("mdd")
                                              for w in range(values.shape[1])]}


def summarise_experiment(experiment: str, subjects: list[str], rng: np.random.Generator, n_boot: int) -> list[dict]:
    rows, by_cell = [], {}
    for subject in subjects:
        for group in GROUPS:
            record = study.read_record(group_key(experiment, subject, group))
            for readout, cell in (record or {"readouts": {}})["readouts"].items():
                by_cell.setdefault(readout, {"cell": cell, "subjects": {}})["subjects"][subject] = cell["representations"]
    for readout, item in sorted(by_cell.items()):
        cell, per_subject = item["cell"], item["subjects"]
        for rep in cell["representations"]:
            having = sorted(s for s, r in per_subject.items() if rep in r)
            for metric in METRICS:
                chance = (1.0 / cell["n_classes"]) if metric == "balanced_accuracy" else CHANCE_AUC
                observed = np.stack([per_subject[s][rep]["observed"][metric] for s in having])
                null = np.stack([per_subject[s][rep]["null"][metric].mean(axis=0) for s in having])
                base = {"dataset_group": EXPERIMENTS[experiment]["dataset_group"], "channel_note": CHANNEL_NOTE,
                        "readout": readout, "group": cell["group"], "kind": cell["kind"], "epoch": cell["epoch"],
                        "label_family": cell["family"], "label_vector": cell["label_vector"],
                        "window_start_s": cell["window_start_s"], "window_width_s": cell["window_width_s"],
                        "representation": rep, "metric": metric, "chance": chance, "n_subjects": len(having),
                        "subjects": having}
                rows.append({**base, "quantity": "above_label_shuffle_null", "mean_observed": observed.mean(axis=0).tolist(),
                             "mean_null": null.mean(axis=0).tolist(), **across_subjects(observed - null, rng, n_boot)})
                if rep != "native":
                    shared = [s for s in having if "native" in per_subject[s]]
                    if shared:
                        diff = np.stack([
                            (per_subject[s][rep]["observed"][metric] - per_subject[s][rep]["null"][metric].mean(axis=0))
                            - (per_subject[s]["native"]["observed"][metric]
                               - per_subject[s]["native"]["null"][metric].mean(axis=0)) for s in shared])
                        rows.append({**base, "quantity": "representation_minus_native_above_null", "n_subjects": len(shared),
                                     "subjects": shared, **across_subjects(diff, rng, n_boot)})
    return rows


def render_summary(artifact: dict) -> str:
    lines = ["# Memorandum decoding, scalp EEG", "", f"status: {artifact['status']}", "", CHANNEL_NOTE, "",
             "Fixed-window rows (mean over subjects, subject-bootstrap 2.5 and 97.5 percentiles, sign-flip p); time "
             "courses are in the artifact.", "",
             "| dataset group | read-out | representation | metric | quantity | mean | interval | p | subjects |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in artifact.get("results", []):
        if r["kind"] in ("fixed", "transfer_fixed"):
            lines.append(f"| {r['dataset_group']} | {r['readout']} | {r['representation']} | {r['metric']} | {r['quantity']} | "
                         f"{r['mean'][0]:.4f} | {r['interval_95'][0][0]:.4f} to {r['interval_95'][1][0]:.4f} | "
                         f"{r['p'][0]:.4f} | {r['n_subjects']} |")
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Memorandum decoding from scalp EEG, per subject.")
    p.add_argument("--output", type=Path, default=OUTPUT_PATH)
    p.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--experiments", nargs="+", choices=tuple(EXPERIMENTS), default=list(EXPERIMENTS))
    p.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    p.add_argument("--max-subjects", type=int)
    p.add_argument("--n-shuffles", type=int, default=100)
    p.add_argument("--n-shuffles-time-course", type=int, default=20)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--selection-permutations", type=int, default=md.SELECTION_PERMUTATIONS)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = parser().parse_args()
    if min(args.n_shuffles, args.n_shuffles_time_course, args.n_bootstrap) < 1:
        raise SystemExit("shuffle and bootstrap counts must be positive")
    started = time.time()
    md.SELECTION_PERMUTATIONS = args.selection_permutations
    identity = code_identity(ROOT, Path(__file__))
    CONFIG.update(checkpoint_dir=args.checkpoint_dir, n_shuffles=args.n_shuffles,
                  n_shuffles_time_course=args.n_shuffles_time_course,
                  identity=hashlib.sha256(canonical_json({
                      "code": identity, "selection_permutations": args.selection_permutations,
                      "n_shuffles": args.n_shuffles, "n_shuffles_time_course": args.n_shuffles_time_course,
                      "seed": args.seed}).encode()).hexdigest())
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tasks = [{"id": f"{e}|{p.stem.rsplit('_', 1)[1]}", "experiment": e, "subject": p.stem.rsplit("_", 1)[1], "path": p,
              "groups": args.groups} for e in args.experiments for p in subject_files(e, args.max_subjects)]
    artifact = {
        "version": SCHEMA, "status": "running", "code_identity": identity, "checkpoint_identity": CONFIG["identity"],
        "scope": {
            "datasets": {e: EXPERIMENTS[e]["dataset_group"] for e in args.experiments}, "channel_note": CHANNEL_NOTE,
            "unit": "one subject is one recording; no pseudo-population across subjects",
            "features": "channel means over windows of the loader's epochs, each trial channel-mean-centred; bad trials "
                        "removed with the loader's mask; no decimation or filtering",
            "label_definitions": {e: EXPERIMENTS[e]["label_definitions"] for e in args.experiments},
            "orientation_bins": impulse.N_ORIENTATION_BINS, "orientation_range_rad": list(impulse.ORIENTATION_RANGE),
            "fixed_windows_s": {e: EXPERIMENTS[e]["windows_s"] for e in args.experiments},
            "time_course": {"widths_ms": list(scalp.COURSE_WIDTHS_MS), "step_ms": int(md.WINDOW_STEP_S * 1000),
                            "aligned_to": "each epoch's own time zero (the epochs are locked to the memory items, the "
                                          "retro-cue and the impulse; the time between them is not recorded per trial)"},
            "transfer": "decoder trained on the encoding-window channel means and tested on every window of the other epochs",
            "readout_groups": args.groups, "decoder": scalp.DECODER, "latent_rank": scalp.LATENT_RANK,
            "representations": {"native": "all channels",
                                "native_selected": "channels passing the one-way analysis of variance plus right-tailed "
                                                   "permutation test (category_selective_mask) on the encoding-window "
                                                   "means of the training trials, inside each fold",
                                "principal_components": "top components of the 100 ms encoding-epoch bins of the training "
                                                        "trials, projected onto every window",
                                "demixed_principal_components": "four axes of the orientation-marginal ridge reduced-rank "
                                                                "fit to the encoding epoch of the training trials, "
                                                                "projected with the decoder; penalty chosen by held-out "
                                                                "reconstruction of the marginal target"},
            "latent_families": "principal components and demixed axes are fitted on the encoding epoch of each fold's "
                               "training trials and applied to every window; demixed axes are run for the orientation "
                               "label only",
            "outer_folds": "10 when every class has at least 10 valid encoding trials, else 5",
            "nested_cross_validation": {"inner_folds": md.INNER_FOLDS, "grid": md.HYPER_GRIDS[scalp.DECODER]},
            "shuffle_null": "labels permuted within subject across all trials; fits and hyperparameters of the observed "
                            "labels are reused, so the null is slightly optimistic",
            "n_shuffles": args.n_shuffles, "n_shuffles_time_course": args.n_shuffles_time_course,
            "selection_permutations": args.selection_permutations, "selection_level": md.SELECTION_LEVEL,
            "across_subjects": {"estimate": "mean over subjects of observed minus the shuffle-null mean",
                                "interval": "2.5 and 97.5 percentiles of subject-bootstrap means",
                                "p": "two-sided sign-flip test of the subject values against zero; a number only",
                                "n_bootstrap": args.n_bootstrap},
            "seed": args.seed, "max_subjects": args.max_subjects}}

    def flush(**extra):
        artifact.update(extra)
        artifact["wall_clock_s"] = time.time() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_json(artifact))
        args.summary.write_text(render_summary(artifact))

    flush()
    execute(tasks, args.workers)
    accounting, results = [], []
    rng = np.random.default_rng(args.seed)
    for experiment in args.experiments:
        computed = []
        for task in (t for t in tasks if t["experiment"] == experiment):
            status = study.read_record(group_key(experiment, task["subject"], "status"))
            accounting.append({"dataset_group": EXPERIMENTS[experiment]["dataset_group"], "subject": task["subject"],
                               "file": task["path"].name, **(status or {"status": "not_run"})})
            if status and status["status"] == "computed":
                computed.append(task["subject"])
        results += summarise_experiment(experiment, computed, rng, args.n_bootstrap)
    reasons = {}
    for row in accounting:
        key = f"{row['dataset_group']}|{row['status']}"
        reasons[key] = reasons.get(key, 0) + 1
    flush(subject_accounting=accounting, subject_reconciliation={"files_seen": len(accounting), "by_status": reasons},
          results=results, status="complete")


if __name__ == "__main__":
    main()
