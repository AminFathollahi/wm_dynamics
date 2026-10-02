"""Co-primary comparison of latent representations against two reference representations.

Combines delivered records only; nothing is recomputed from data. For each corpus, region cell and
level, every candidate representation is compared with principal components and with demixed
principal components on two scores that share the same region-session entries:

  decoding        same-time area under the curve above the shuffled-label null (linear decoder,
                  whole-trial inference)
  reconstruction  held-out bits per spike (single units) or bits per observation (field potentials)

Differences (candidate minus reference) are averaged within patient, then summarised across patients
with a patient-resampled 95% interval and a two-sided sign-flip p over patients. A level meets the
interval rule when one score's interval lies above zero and the other score's upper end is at or
above zero; a candidate is co-primary against a reference when the rule is met at every level of the
corpus. An input that is missing or not complete marks every cell that depends on it as
inputs_incomplete.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    if str(ROOT / _subdir) not in sys.path:
        sys.path.insert(0, str(ROOT / _subdir))

from info_decoding import atomic_write  # noqa: E402
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from statistics import bootstrap_ci, minimum_detectable_paired_difference, paired_sign_flip_test  # noqa: E402

SNAPSHOT_ROOT = ROOT.parent / "benchmark_snapshots"
OUTPUT_PATH = ROOT / "results" / "representation_benchmark_co_primary.json"

SEED = 1
N_BOOT = 5000
# A patient-resampled interval and a sign-flip over patients have nothing to vary below three patients.
MIN_PATIENTS = 3
REFERENCES = ("principal_components", "demixed_principal_components")
POOLED_REGION = "regions_pooled_by_patient"
POOLED_SOURCES = ("hippocampus", "amygdala")

LOAD1, LOAD3 = "load1_maintenance", "load3_first_item_maintenance"
LEVELS = {
    "human_single_units": (LOAD1, LOAD3),
    "dandi_000574": ("set_size",),
    "dandi_000673": (LOAD1, LOAD3),
}
CORPUS_SIGNAL = {"human_single_units": "single_units", "dandi_000574": "field_potentials",
                 "dandi_000673": "field_potentials"}

TIME_CONTRASTIVE = ("time_contrastive_embedding",)
SWITCHING_AND_TRANSFORMER = ("recurrent_switching_linear_dynamics", "neural_data_transformer")
BASE = ("native_full_rank", "principal_components", "factor_analysis", "gaussian_process_factor_analysis",
        "temporal_diffusion_embedding", "sequential_autoencoder", "demixed_principal_components")
UNIT_DECODING_CANDIDATES = BASE + TIME_CONTRASTIVE
EXCEPT_TIME_CONTRASTIVE = BASE + SWITCHING_AND_TRANSFORMER

DEMIXED_FOLD_NOTE = ("demixed principal components was scored on fold draws that differ from those of the "
                     "other representations in the single-unit decoding benchmark")

# kind: which score an input supplies; level None means every level the signal has.
INPUT_SPECS = (
    {"name": "unit_decoding_load1", "kind": "decoding", "level": LOAD1, "candidates": UNIT_DECODING_CANDIDATES,
     "default": "human_field_potential_benchmark_2026_09_30/results/human_representation_benchmark_seeds/seed_0.json"},
    {"name": "unit_decoding_load3", "kind": "decoding", "level": LOAD3, "candidates": UNIT_DECODING_CANDIDATES,
     "default": "human_field_potential_benchmark_2026_09_30/results/human_representation_benchmark_seeds/seed_0_load3.json"},
    {"name": "unit_decoding_switching_and_transformer_load1", "kind": "decoding", "level": LOAD1,
     "candidates": SWITCHING_AND_TRANSFORMER,
     "default": "readout_and_new_models_2026_09_28/results/human_representation_benchmark_seeds/seed_0_new_models.json"},
    {"name": "unit_decoding_switching_and_transformer_load3", "kind": "decoding", "level": LOAD3,
     "candidates": SWITCHING_AND_TRANSFORMER,
     "default": "readout_and_new_models_2026_09_28/results/human_representation_benchmark_seeds/seed_0_load3_new_models.json"},
    {"name": "unit_cosmoothing_load1", "kind": "reconstruction", "level": LOAD1, "candidates": EXCEPT_TIME_CONTRASTIVE,
     "default": "human_cosmoothing_benchmark_2026_09_28/results/human_cosmoothing_benchmark/human_cosmoothing_benchmark_load1.json"},
    {"name": "unit_cosmoothing_load3", "kind": "reconstruction", "level": LOAD3, "candidates": EXCEPT_TIME_CONTRASTIVE,
     "default": "human_cosmoothing_benchmark_2026_09_28/results/human_cosmoothing_benchmark/human_cosmoothing_benchmark_load3.json"},
    {"name": "unit_cosmoothing_time_contrastive_load1", "kind": "reconstruction", "level": LOAD1,
     "candidates": TIME_CONTRASTIVE,
     "default": "human_cosmoothing_time_contrastive_2026_10_02/results/human_cosmoothing_benchmark/time_contrastive_load1_maintenance.json"},
    {"name": "unit_cosmoothing_time_contrastive_load3", "kind": "reconstruction", "level": LOAD3,
     "candidates": TIME_CONTRASTIVE,
     "default": "human_cosmoothing_time_contrastive_2026_10_02/results/human_cosmoothing_benchmark/time_contrastive_load3_first_item_maintenance.json"},
    {"name": "field_potentials", "kind": "both", "level": None, "candidates": EXCEPT_TIME_CONTRASTIVE,
     "default": "human_field_potential_benchmark_2026_09_30/results/human_field_potential_benchmark.json"},
    {"name": "field_potentials_time_contrastive", "kind": "both", "level": None, "candidates": TIME_CONTRASTIVE,
     "default": "human_field_potential_time_contrastive_2026_10_01/results/human_field_potential_time_contrastive.json"},
)
UNIT_SPEC_PREFIXES = ("unit_decoding", "unit_cosmoothing")


def option_name(spec: dict) -> str:
    return "--" + spec["name"].replace("_", "-")


def load_input(spec: dict, path: Path) -> dict:
    payload, status = None, "missing"
    if Path(path).exists():
        try:
            payload = json.loads(Path(path).read_text())
            status = str(payload.get("status", "unknown"))
        except (OSError, ValueError):
            status = "unreadable"
    return {"spec": spec, "path": Path(path), "payload": payload, "status": status}


def is_complete(loaded: dict) -> bool:
    return loaded["status"] == "complete"


def is_unit_input(loaded: dict) -> bool:
    return loaded["spec"]["name"].startswith(UNIT_SPEC_PREFIXES)


def patient_map(payload: dict) -> dict[str, str]:
    mapping = {}
    for session in payload.get("sessions", []):
        mapping[session["session"]] = session["patient"]
        mapping[f"{session['patient']}_{session['session']}"] = session["patient"]
    return mapping


def collect_records(loaded_inputs: list[dict]) -> tuple[dict, dict, list[dict]]:
    """Scores keyed by ((corpus, region, level), entry, candidate) -> (patient, value), one dict per score,
    plus per-input accounting of every record seen: used, or excluded with its reason."""
    decoding, reconstruction, accounting = {}, {}, []
    for loaded in loaded_inputs:
        spec, payload = loaded["spec"], loaded["payload"]
        reasons = Counter()
        if payload is None:
            accounting.append({"input": spec["name"], "n_records_seen": 0, "n_used": 0, "excluded_by_reason": {}})
            continue
        patients = patient_map(payload)
        if is_unit_input(loaded):
            streams = [("decoding" if spec["kind"] == "decoding" else "reconstruction", payload.get("records", []))]
        else:
            streams = [("decoding", payload.get("content_records", [])),
                       ("reconstruction", payload.get("reconstruction_records", []))]
        n_seen = n_used = 0
        for kind, records in streams:
            for record in records:
                n_seen += 1
                reason, cell_corpus, region, value = classify_record(record, kind, loaded, patients)
                if reason is not None:
                    reasons[reason] += 1
                    continue
                n_used += 1
                key = ((cell_corpus, region, record["level"]), record["session"], record["candidate"])
                target = decoding if kind == "decoding" else reconstruction
                target[key] = (patients[record["session"]], value)
        accounting.append({"input": spec["name"], "n_records_seen": n_seen, "n_used": n_used,
                           "excluded_by_reason": dict(sorted(reasons.items()))})
    return add_pooled_regions(decoding), add_pooled_regions(reconstruction), accounting


def classify_record(record: dict, kind: str, loaded: dict, patients: dict) -> tuple:
    """(exclusion reason or None, corpus, region, score)."""
    spec = loaded["spec"]
    none = (None, None, None)
    if record.get("candidate") not in spec["candidates"]:
        return ("candidate_supplied_by_another_input", *none)
    if record.get("status") != "computed":
        return (f"record_status_{record.get('status')}", *none)
    if spec["level"] is not None and record.get("level") != spec["level"]:
        return ("level_not_supplied_by_this_input", *none)
    if record.get("session") not in patients:
        return ("session_without_patient", *none)
    if is_unit_input(loaded):
        corpus = "human_single_units"
        # decoding records name the brain region in their corpus field
        region = record.get("corpus") if kind == "decoding" else record.get("region")
        if kind == "decoding":
            if record.get("decoder") != "linear":
                return ("decoder_not_linear", *none)
            value = record["same_time"].get("auc_above_null")
        else:
            value = record.get("bits_per_spike")
    else:
        corpus, region = record["corpus"], record["region"]
        if record.get("mode") != "whole_trial":
            return ("inference_mode_not_whole_trial", *none)
        if kind == "decoding":
            if record.get("decoder") != "linear":
                return ("decoder_not_linear", *none)
            value = record["same_time"].get("auc_above_null")
        else:
            value = record.get("bits_per_observation")
    if corpus not in LEVELS or record.get("level") not in LEVELS[corpus]:
        return ("level_not_in_corpus", *none)
    if value is None:
        return ("score_absent", *none)
    return (None, corpus, region, float(value))


def add_pooled_regions(scores: dict) -> dict:
    """Each hippocampal and amygdalar entry also enters a pooled cell, but only for corpora that have both."""
    present = {}
    for (corpus, region, _level), _entry, _candidate in scores:
        present.setdefault(corpus, set()).add(region)
    out = dict(scores)
    for ((corpus, region, level), entry, candidate), value in scores.items():
        if region in POOLED_SOURCES and set(POOLED_SOURCES) <= present[corpus]:
            out[((corpus, POOLED_REGION, level), f"{entry}|{region}", candidate)] = value
    return out


def estimate(differences: list[float], patients: list[str]) -> dict:
    values, labels = np.asarray(differences, dtype=float), np.asarray(patients)
    unique = np.unique(labels)
    patient_means = np.array([values[labels == u].mean() for u in unique])
    base = {"n_entries": int(len(values)), "n_patients": int(len(unique))}
    if len(unique) < MIN_PATIENTS:
        return {"status": "not_estimable", **base, "mean": float(patient_means.mean()), "ci_95": None,
                "median": float(np.median(patient_means)), "median_ci_95": None, "sign_flip_p": None, "minimum_detectable_difference": None,
                "reason": f"fewer than {MIN_PATIENTS} patients"}
    rng = np.random.default_rng(SEED)
    mean, lo, hi = bootstrap_ci(patient_means, np.mean, n_boot=N_BOOT, rng=rng)
    p_value = paired_sign_flip_test(patient_means, np.zeros_like(patient_means), alternative="two-sided",
                                    rng=rng)["p_value"]
    mdd = minimum_detectable_paired_difference(patient_means).get("mdd")
    median, median_lo, median_hi = bootstrap_ci(patient_means, np.median, n_boot=N_BOOT,
                                                rng=np.random.default_rng(SEED))
    return {"status": "estimable", **base, "mean": float(mean), "ci_95": [float(lo), float(hi)],
            "median": float(median), "median_ci_95": [float(median_lo), float(median_hi)],
            "sign_flip_p": float(p_value), "minimum_detectable_difference": mdd}


def pairs(scores: dict, cell: tuple, candidate: str, reference: str | None) -> tuple[list, list]:
    values, patients = [], []
    for (c, entry, cand), (patient, value) in scores.items():
        if c != cell or cand != candidate:
            continue
        if reference is not None:
            if (cell, entry, reference) not in scores:
                continue
            value -= scores[(cell, entry, reference)][1]
        values.append(value)
        patients.append(patient)
    return values, patients


def summarise(scores: dict, cell: tuple, candidate: str, reference: str | None) -> dict | None:
    values, patients = pairs(scores, cell, candidate, reference)
    return estimate(values, patients) if values else None


def depends_on_incomplete(loaded_inputs: list[dict], corpus: str, level: str, candidate: str) -> bool:
    for loaded in loaded_inputs:
        if is_complete(loaded) or candidate not in loaded["spec"]["candidates"]:
            continue
        if is_unit_input(loaded) != (CORPUS_SIGNAL[corpus] == "single_units"):
            continue
        if loaded["spec"]["level"] in (None, level):
            return True
    return False


def interval_rule(decoding: dict | None, reconstruction: dict | None) -> bool | None:
    if not decoding or not reconstruction or "estimable" != decoding["status"] or "estimable" != reconstruction["status"]:
        return None
    (d_lo, d_hi), (r_lo, r_hi) = decoding["ci_95"], reconstruction["ci_95"]
    return bool((d_lo > 0 and r_hi >= 0) or (r_lo > 0 and d_hi >= 0))


def cell_status(decoding, reconstruction, incomplete: bool) -> str:
    if incomplete:
        return "inputs_incomplete"
    if decoding is None or reconstruction is None:
        return "no_paired_entries"
    if "estimable" != decoding["status"] or "estimable" != reconstruction["status"]:
        return "not_estimable"
    return "estimable"


def summarise_all(decoding: dict, reconstruction: dict, loaded_inputs: list[dict]) -> dict:
    cells = sorted({cell for cell, _e, _c in list(decoding) + list(reconstruction)})
    absolute, paired = [], []
    for cell in cells:
        corpus, _region, level = cell
        candidates = sorted({cand for c, _e, cand in list(decoding) + list(reconstruction) if c == cell}
                            | pending_candidates(loaded_inputs, corpus, level))
        for candidate in candidates:
            d, r = summarise(decoding, cell, candidate, None), summarise(reconstruction, cell, candidate, None)
            incomplete = depends_on_incomplete(loaded_inputs, corpus, level, candidate)
            if d is None and r is None and not incomplete:
                continue
            absolute.append({"corpus": corpus, "region_cell": cell[1], "level": level, "candidate": candidate,
                             "status": cell_status(d, r, incomplete), "decoding": d, "reconstruction": r})
            for reference in REFERENCES:
                if candidate == reference:
                    continue
                d = summarise(decoding, cell, candidate, reference)
                r = summarise(reconstruction, cell, candidate, reference)
                incomplete = (depends_on_incomplete(loaded_inputs, corpus, level, candidate)
                              or depends_on_incomplete(loaded_inputs, corpus, level, reference))
                status = cell_status(d, r, incomplete)
                record = {"corpus": corpus, "region_cell": cell[1], "level": level, "candidate": candidate,
                          "reference": reference, "status": status, "decoding_difference": d,
                          "reconstruction_difference": r,
                          "meets_interval_rule": interval_rule(d, r) if status == "estimable" else None}
                if CORPUS_SIGNAL[corpus] == "single_units" and "demixed_principal_components" in (candidate, reference):
                    record["note"] = DEMIXED_FOLD_NOTE
                paired.append(record)
    return {"absolute_scores": absolute, "paired_differences": paired, "co_primary": co_primary_rows(paired)}


def pending_candidates(loaded_inputs: list[dict], corpus: str, level: str) -> set:
    return {c for l in loaded_inputs if not is_complete(l) for c in l["spec"]["candidates"]
            if depends_on_incomplete([l], corpus, level, c)}


def co_primary_rows(paired: list[dict]) -> list[dict]:
    groups = {}
    for record in paired:
        key = (record["corpus"], record["region_cell"], record["candidate"], record["reference"])
        groups.setdefault(key, {})[record["level"]] = record
    rows = []
    for (corpus, region, candidate, reference), by_level in sorted(groups.items()):
        levels = []
        for level in LEVELS[corpus]:
            record = by_level.get(level)
            counts = {k: (record[f"{k}_difference"] or {}) if record else {} for k in ("decoding", "reconstruction")}
            levels.append({"level": level, "status": record["status"] if record else "no_paired_entries",
                           "meets_interval_rule": record["meets_interval_rule"] if record else None,
                           "n_patients": {k: v.get("n_patients") for k, v in counts.items()},
                           "n_entries": {k: v.get("n_entries") for k, v in counts.items()}})
        statuses = [lv["status"] for lv in levels]
        overall = next((s for s in ("inputs_incomplete", "not_estimable", "no_paired_entries") if s in statuses), "estimable")
        rows.append({
            "corpus": corpus, "region_cell": region, "candidate": candidate, "reference": reference,
            "status": overall, "levels": levels,
            "co_primary": bool(all(lv["meets_interval_rule"] for lv in levels)) if overall == "estimable" else None,
            "levels_required": list(LEVELS[corpus]),
            "level_note": ("the single set-size level is the only level this corpus has" if len(LEVELS[corpus]) == 1
                           else "loads 1 and 3 are both required"),
        })
    return rows


def build_artifact(loaded_inputs: list[dict], started: float | None = None) -> dict:
    started = started if started is not None else time.time()
    decoding, reconstruction, accounting = collect_records(loaded_inputs)
    summary = summarise_all(decoding, reconstruction, loaded_inputs)
    incomplete = [l["spec"]["name"] for l in loaded_inputs if not is_complete(l)]
    path = Path(__file__)
    return {
        "version": "representation_benchmark_co_primary_v1",
        "code_commit": git_commit(ROOT, path),
        "status": "complete" if not incomplete else "inputs_incomplete",
        "inputs": [{"name": l["spec"]["name"], "path": str(l["path"]), "status": l["status"],
                    "complete": is_complete(l), "score": l["spec"]["kind"], "level": l["spec"]["level"],
                    "candidates": list(l["spec"]["candidates"]),
                    "sha256": sha256_file(l["path"]) if l["path"].is_file() else None} for l in loaded_inputs],
        "scope": {
            "references": list(REFERENCES), "levels": {k: list(v) for k, v in LEVELS.items()},
            "scores": {"decoding": "same-time area under the curve above the shuffled-label null, linear decoder, "
                                   "whole-trial inference",
                       "reconstruction": "held-out bits per spike (single units); bits per observation (field potentials)"},
            "unit_of_inference": "patient; differences are averaged within patient across region-sessions",
            "n_boot": N_BOOT, "interval": "95% percentile bootstrap over patients", "seed": SEED,
            "minimum_patients": MIN_PATIENTS, "p_value": "two-sided sign-flip over patient means",
            "interval_rule": "one score's interval lies above 0 and the other score's upper end is at or above 0",
            "region_cells": f"each region; {POOLED_REGION} where the corpus has both hippocampus and amygdala; pooled for dandi_000574",
            "incomplete_inputs": incomplete, "input_accounting": accounting,
            "notes": [DEMIXED_FOLD_NOTE,
                      "cells whose candidate or reference depends on an incomplete input are reported with their "
                      "provisional estimates and the status inputs_incomplete, and carry no interval-rule flag"],
            "wall_clock_s": float(time.time() - started),
        },
        **summary,
    }


def fmt(estimate_: dict | None) -> str:
    if estimate_ is None:
        return "no paired entries"
    n = f"n={estimate_['n_patients']}pat/{estimate_['n_entries']}ent"
    if estimate_["status"] != "estimable":
        return f"{estimate_['mean']:+.4f} (no interval, {n})"
    lo, hi = estimate_["ci_95"]
    return f"{estimate_['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}] p={estimate_['sign_flip_p']:.4f} {n}"


def render_markdown(artifact: dict) -> str:
    lines = ["# Representation benchmark: co-primary comparison", "",
             f"Status: {artifact['status']}. Incomplete inputs: {', '.join(artifact['scope']['incomplete_inputs']) or 'none'}.", "",
             "Each row: candidate minus reference, mean over patients [95% interval] p, patients and entries paired. "
             "Interval rule: one score's interval above 0 and the other's upper end at or above 0.", ""]
    index = {(r["corpus"], r["region_cell"], r["candidate"], r["reference"], r["level"]): r for r in artifact["paired_differences"]}
    group = None
    for row in artifact["co_primary"]:
        head = (row["corpus"], row["region_cell"])
        if head != group:
            group = head
            lines += ["", f"## {row['corpus']} / {row['region_cell']}", ""]
        lines.append(f"- {row['candidate']} vs {row['reference']}: co_primary={row['co_primary']} ({row['status']})")
        for level in row["levels"]:
            rec = index.get((row["corpus"], row["region_cell"], row["candidate"], row["reference"], level["level"]))
            if rec:
                lines.append(f"    - {level['level']}: decoding {fmt(rec['decoding_difference'])} | "
                             f"reconstruction {fmt(rec['reconstruction_difference'])} | rule met: {rec['meets_interval_rule']}")
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Co-primary comparison of representations from delivered benchmark records.")
    for spec in INPUT_SPECS:
        parser.add_argument(option_name(spec), type=Path, default=SNAPSHOT_ROOT / spec["default"], dest=spec["name"])
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser


def main() -> None:
    args = _parser().parse_args()
    started = time.time()
    loaded = [load_input(spec, getattr(args, spec["name"])) for spec in INPUT_SPECS]
    artifact = build_artifact(loaded, started)
    atomic_write(args.output, canonical_json(artifact))
    atomic_write(args.output.with_suffix(".md"), render_markdown(artifact))
    print(f"wrote {args.output} (status {artifact['status']})", flush=True)


if __name__ == "__main__":
    main()
