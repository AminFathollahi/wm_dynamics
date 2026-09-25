from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from project_config import data_root, load_dataset_registry  # noqa: E402
from provenance import _json_safe, git_commit  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from response_latency import LATENCY_EXTRACTORS, LATENCY_UNAVAILABLE  # noqa: E402

RESULTS = ROOT / "results"
SCHEMA_VERSION = "1"


def summarize(records: list[dict]) -> dict:
    latency = np.concatenate([r["latency_s"] for r in records])
    return {
        "n_sessions": len(records),
        "n_trials": int(len(latency)),
        "median_s": float(np.median(latency)),
        "iqr_s": [float(np.percentile(latency, 25)), float(np.percentile(latency, 75))],
        "derivation": records[0]["derivation"],
        "per_session": [
            {"patient": r["patient"], "session": r["session"], "n_trials": r["n_trials"],
             "median_s": float(np.median(r["latency_s"]))}
            for r in records
        ],
    }


def main() -> None:
    root = data_root()
    registry = load_dataset_registry()
    entries = {}
    for dataset in registry["datasets"]:
        if dataset in LATENCY_EXTRACTORS:
            records = list(LATENCY_EXTRACTORS[dataset](root))
            entries[dataset] = ({"status": "available", **summarize(records)} if records
                                 else {"status": "no_recoverable_trials", "reason": "extractor ran but every "
                                       "session yielded zero finite positive latencies"})
        elif dataset in LATENCY_UNAVAILABLE:
            entries[dataset] = {"status": "unavailable", "reason": LATENCY_UNAVAILABLE[dataset]}
        else:
            entries[dataset] = {"status": "not_surveyed", "reason": "no extractor registered and no "
                                 "unavailability reason recorded -- gap in this census, not a verified absence"}

    dandi_000469_note = {
        "total_files": 41, "files_with_full_trials_table": 21, "files_without_full_trials_table": 20,
        "finding": "the 20 files without a full trials table are exactly the ses-1 recordings "
                   "(20 sub-*_ses-1_ecephys+image.nwb files); the 21 with one are exactly the ses-2 "
                   "recordings. corpus_sessions.iter_dandi_000469 already restricts its glob to "
                   "*_ses-2_ecephys+image.nwb only, for a reason unrelated to trials-table completeness "
                   "(session selection), so the exclusion this census needed already exists in the "
                   "project's primary loader and the two selections coincide by construction, not by "
                   "having been designed against each other",
    }

    output = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(ROOT),
        "entries": entries,
        "dandi_000469_trials_table_coverage": dandi_000469_note,
        "recall_vs_recognition_latency_warning": "ram_ds005489_openloop and ram_ds005557_closedloop report "
            "a verbal free-recall latency (median several seconds, event-onset-relative within the recall "
            "period); dandi_000469, dandi_000673, dandi_001187 and dandi_000574 report a probe-locked "
            "recognition latency (median ~1.1-1.3 s). These are different behavioural constructs and are "
            "never pooled into one estimate anywhere in this project.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "response_latency_census.json").write_text(json.dumps(_json_safe(output), indent=2))
    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["response_latency_census"] = output
    print(json.dumps({k: v.get("status") for k, v in entries.items()}, indent=2))


if __name__ == "__main__":
    main()
