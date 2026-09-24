"""build_session_admission_accuracy_census.py -- measures, per human single-unit
Sternberg corpus (dandi_000469, dandi_001187, dandi_000673, dandi_000574), the
mean behavioural accuracy of every recording session on disk before
MIN_SESSION_ACCURACY (src/spike_pipeline.py, 0.55) is applied anywhere.

MIN_SESSION_ACCURACY excludes whole sessions at eleven call sites across this
repository, but no delivered artifact records how many sessions or patients
that exclusion actually removes -- only how many trials. This script closes
that gap by reading the same accuracy field each site already reads
(``response_accuracy`` for the three Rutishauser-lineage releases,
``correct`` for dandi_000574) directly from every session file registered in
provenance/canonical_recording_registry.json, which already carries the
verified cross-release patient identity this project uses everywhere else
(scripts/run_human_drift_spine_001187_000673.py's canonical_sessions, built
by scripts/audit_dataset_identity.py). No accuracy computation or identity
resolution is re-derived here.

dandi_000469's registry rows include both its ses-1 and ses-2 recordings;
only ses-2 carries the working-memory Sternberg task (verified directly: all
20 ses-1 trial tables in this data root lack a response_accuracy field
entirely), matching the ses-2-only glob every ``MIN_SESSION_ACCURACY`` call
site for this corpus already uses. ses-1 sessions are reported separately,
not counted toward this corpus's WM-session totals, and every 000469
registry row still reconciles into one of the two groups.

MIN_SESSION_ACCURACY is never applied to dandi_000574 anywhere in this
repository (verified by exhaustive source sweep); its census is reported for
completeness and comparability only, and is marked as such.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spike_pipeline import MIN_SESSION_ACCURACY  # noqa: E402

REGISTRY_PATH = ROOT / "provenance" / "canonical_recording_registry.json"
ARTIFACT_PATH = ROOT / "results" / "session_admission_accuracy_census.json"

CORPUS_ACCURACY_FIELD = {
    "dandi_000469": "response_accuracy",
    "dandi_001187": "response_accuracy",
    "dandi_000673": "response_accuracy",
    "dandi_000574": "correct",
}
CORPUS_MIN_SESSION_ACCURACY_APPLIED = {
    "dandi_000469": True,
    "dandi_001187": True,
    "dandi_000673": True,
    "dandi_000574": False,
}
RELEASE_FOR_CORPUS = {
    "dandi_000469": "000469",
    "dandi_001187": "001187",
    "dandi_000673": "000673",
    "dandi_000574": "000574",
}
CORPUS_TRIAL_GROUP = {
    "dandi_000469": "trials",
    "dandi_001187": "WM_trials",
    "dandi_000673": "trials",
    "dandi_000574": "trials",
}


def data_root() -> Path:
    root = os.environ.get("WM_DYNAMICS_DATA_ROOT")
    if not root:
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT to the configured external data root.")
    path = Path(root)
    if not path.is_dir():
        raise SystemExit(f"WM_DYNAMICS_DATA_ROOT is not a directory: {path}")
    return path


def _session_accuracy(root: Path, row: dict, field: str, trial_group: str) -> dict:
    path = root / row["path"]
    with h5py.File(path, "r") as handle:
        if trial_group not in handle.get("intervals", {}):
            return {"status": "no_trial_group", "trial_group": trial_group}
        trials = handle[f"intervals/{trial_group}"]
        if field not in trials:
            return {"status": "no_accuracy_field", "field": field}
        values = trials[field][:].astype(bool)
    if len(values) == 0:
        return {"status": "zero_trials"}
    return {"status": "computed", "n_trials": int(len(values)), "accuracy": float(values.mean())}


def _census_corpus(root: Path, corpus: str, registry: list[dict]) -> dict:
    release = RELEASE_FOR_CORPUS[corpus]
    field = CORPUS_ACCURACY_FIELD[corpus]
    trial_group = CORPUS_TRIAL_GROUP[corpus]
    rows = [r for r in registry if r["release"] == release]

    excluded_non_wm = []
    wm_rows = rows
    if corpus == "dandi_000469":
        wm_rows = [r for r in rows if r["session"].endswith("_ses-2")]
        excluded_non_wm = [r for r in rows if not r["session"].endswith("_ses-2")]

    sessions = []
    load_errors = []
    for row in wm_rows:
        result = _session_accuracy(root, row, field, trial_group)
        record = {"patient": row["patient"], "session": row["session"], "path": row["path"], **result}
        sessions.append(record)
        if result["status"] != "computed":
            load_errors.append(record)

    computed = [s for s in sessions if s["status"] == "computed"]
    below_floor = [s for s in computed if s["accuracy"] < MIN_SESSION_ACCURACY]
    at_or_above_floor = [s for s in computed if s["accuracy"] >= MIN_SESSION_ACCURACY]

    accuracies_sorted = sorted(s["accuracy"] for s in computed)
    patients_all = sorted(set(s["patient"] for s in computed))
    patients_below_floor = sorted(set(s["patient"] for s in below_floor))

    n_seen = len(rows)
    n_reconciled = len(excluded_non_wm) + len(sessions)

    return {
        "min_session_accuracy_applied_to_this_corpus_in_repository": CORPUS_MIN_SESSION_ACCURACY_APPLIED[corpus],
        "accuracy_field": field,
        "accuracy_field_source": (
            f"provenance/canonical_recording_registry.json row -> intervals/trials/{field}, the same "
            "field the corpus's own MIN_SESSION_ACCURACY call sites read"
            if CORPUS_MIN_SESSION_ACCURACY_APPLIED[corpus] else
            f"provenance/canonical_recording_registry.json row -> intervals/trials/{field}, this "
            "corpus's own per-trial correctness field; no MIN_SESSION_ACCURACY call site reads this "
            "corpus in this repository"
        ),
        "n_registry_rows_this_release": n_seen,
        "n_sessions_excluded_as_non_working_memory_task": len(excluded_non_wm),
        "non_working_memory_sessions_excluded": [
            {"patient": r["patient"], "session": r["session"]} for r in excluded_non_wm
        ],
        "n_sessions_total": len(sessions),
        "n_sessions_load_error": len(load_errors),
        "load_errors": load_errors,
        "n_sessions_computed": len(computed),
        "n_patients_total": len(patients_all),
        "n_sessions_below_floor": len(below_floor),
        "n_patients_with_at_least_one_session_below_floor": len(patients_below_floor),
        "patients_with_at_least_one_session_below_floor": patients_below_floor,
        "sessions_below_floor": [
            {"patient": s["patient"], "session": s["session"], "accuracy": s["accuracy"], "n_trials": s["n_trials"]}
            for s in sorted(below_floor, key=lambda s: s["accuracy"])
        ],
        "n_sessions_at_or_above_floor": len(at_or_above_floor),
        "accuracy_sorted_all_sessions": accuracies_sorted,
        "accuracy_min": accuracies_sorted[0] if accuracies_sorted else None,
        "accuracy_median": float(np.median(accuracies_sorted)) if accuracies_sorted else None,
        "accuracy_max": accuracies_sorted[-1] if accuracies_sorted else None,
        "zero_drop_reconciliation": {
            "n_registry_rows": n_seen,
            "n_excluded_non_wm_plus_n_sessions_censused": n_reconciled,
            "reconciles": bool(n_seen == n_reconciled),
        },
        "per_session_accuracy": [
            {"patient": s["patient"], "session": s["session"], "status": s["status"],
             "accuracy": s.get("accuracy"), "n_trials": s.get("n_trials")}
            for s in sessions
        ],
    }


def main() -> None:
    t0 = time.time()
    root = data_root()
    registry = json.loads(REGISTRY_PATH.read_text())

    per_corpus = {corpus: _census_corpus(root, corpus, registry) for corpus in CORPUS_ACCURACY_FIELD}

    total_sessions_below_floor = sum(c["n_sessions_below_floor"] for c in per_corpus.values()
                                      if c["min_session_accuracy_applied_to_this_corpus_in_repository"])
    total_sessions_gated = sum(c["n_sessions_computed"] for c in per_corpus.values()
                                if c["min_session_accuracy_applied_to_this_corpus_in_repository"])
    if total_sessions_below_floor == 0:
        verdict = "floor_excludes_zero_sessions_in_every_gated_corpus"
    elif total_sessions_below_floor <= 1:
        verdict = "floor_excludes_at_most_one_session_across_gated_corpora"
    else:
        verdict = "floor_excludes_multiple_sessions"

    artifact = {
        "analysis_id": "session_admission_accuracy_census",
        "schema_version": "1.0.0",
        "scope": {
            "wall_clock_seconds": time.time() - t0,
            "min_session_accuracy_value": MIN_SESSION_ACCURACY,
            "min_session_accuracy_source": "src/spike_pipeline.py MIN_SESSION_ACCURACY, unchanged",
            "corpora_censused": list(CORPUS_ACCURACY_FIELD),
            "data_root_used": str(root),
        },
        "verdict": verdict,
        "verdict_definition": "counts sessions below the floor only in corpora where "
                               "min_session_accuracy_applied_to_this_corpus_in_repository is true; "
                               "dandi_000574's own below-floor count is informational and excluded "
                               "from this verdict because the floor is never applied to it in this "
                               "repository",
        "n_sessions_below_floor_across_gated_corpora": total_sessions_below_floor,
        "n_sessions_computed_across_gated_corpora": total_sessions_gated,
        "per_corpus": per_corpus,
    }
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=1))
    print(f"wrote {ARTIFACT_PATH}")
    for corpus, c in per_corpus.items():
        print(f"{corpus}: {c['n_sessions_computed']} sessions, {c['n_patients_total']} patients, "
              f"{c['n_sessions_below_floor']} below {MIN_SESSION_ACCURACY} "
              f"({c['n_patients_with_at_least_one_session_below_floor']} patients), "
              f"floor_applied={c['min_session_accuracy_applied_to_this_corpus_in_repository']}")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
