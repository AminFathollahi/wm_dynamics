from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_info_benchmark_seed_sweep import aggregate_results, load_records, parse_args


def record(seed: int, corpus: str, session: str, candidate: str, value: float) -> dict:
    metric = {"auc": value, "null_auc": 0.5}
    return {
        "status": "computed",
        "seed": seed,
        "corpus": corpus,
        "session": session,
        "level": "2",
        "candidate": candidate,
        "decoder": "linear",
        "same_time": metric,
        "cross_temporal": metric,
    }


def test_parse_args_forwards_benchmark_options():
    args = parse_args([
        "--seeds", "0", "1", "--benchmark-args", "--n-splits", "5", "--n-perm", "100",
    ])
    assert args.seeds == [0, 1]
    assert args.benchmark_args == ["--n-splits", "5", "--n-perm", "100"]


def test_load_records_rejects_incomplete_seed(tmp_path):
    path = tmp_path / "seed_0.json"
    path.write_text(json.dumps({"status": "running", "scope": {"seed": 0}, "records": []}))
    with pytest.raises(ValueError, match="incomplete"):
        load_records([path])


def test_aggregate_rejects_missing_seed_cell():
    rows = [
        record(0, "a", "s", "native_full_rank", 0.5),
        record(0, "a", "s", "principal_components", 0.6),
        record(1, "a", "s", "native_full_rank", 0.5),
    ]
    with pytest.raises(ValueError, match="identical benchmark cells"):
        aggregate_results(rows, ["native_full_rank", "principal_components"], "native_full_rank", 10, 0)


def test_aggregate_clusters_equal_session_names_by_corpus():
    rows = []
    for seed in (0, 1):
        for corpus, difference in (("a", 0.1), ("b", 0.3)):
            rows.extend([
                record(seed, corpus, "session_1", "native_full_rank", 0.5),
                record(seed, corpus, "session_1", "principal_components", 0.5 + difference),
            ])
    result = aggregate_results(rows, ["native_full_rank", "principal_components"], "native_full_rank", 20, 0)
    contrast = result["contrasts"]["principal_components|linear|cross_temporal"]
    assert result["uncertainty_unit"] == "recording_session"
    assert contrast["n_sessions"] == 2
    assert contrast["session_keys"] == [["a", "session_1"], ["b", "session_1"]]
