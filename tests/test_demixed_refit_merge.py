import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "scripts"):
    sys.path.insert(0, str(ROOT / sub))

from demixed_refit_merge import DEMIXED, RefusedMerge, list_rows, merge_refit  # noqa: E402

KEYS = ("session", "decoder")


def artifact(demixed_value, sessions=("a", "b", "c")):
    records = [{"session": s, "decoder": "linear", "candidate": c, "value": v, "n_trials": 10, "status": "computed"}
               for s in sessions for c, v in (("principal_components", 1.0 + len(s)), (DEMIXED, demixed_value))]
    return {"status": "complete", "scope": {"n_splits": 5, "candidates": ["x"]}, "records": records,
            "summary": summary(records)}


def summary(records):
    rows = {(r["session"], r["candidate"]): r["value"] for r in records}
    return [{"session": s, "candidate": DEMIXED, "reference": "principal_components",
             "difference": rows[(s, DEMIXED)] - rows[(s, "principal_components")]}
            for s in sorted({r["session"] for r in records})] + [{"session": "all", "candidate": "native"}]


def run(tmp_path, delivered, refit, **kwargs):
    paths = {}
    for name, value in (("delivered", delivered), ("refit", refit)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(value))
    merge_refit(
        paths["delivered"], paths["refit"], tmp_path / "out.json",
        locators={"records": list_rows(("records",), KEYS)},
        settings=lambda a: {k: v for k, v in a["scope"].items() if k != "candidates"},
        recompute=lambda a, new: {("summary",): summary(a["records"])}, code_identity={}, **kwargs)
    return json.loads((tmp_path / "out.json").read_text())


def test_merge_replaces_only_demixed_rows_and_recomputes_the_summary(tmp_path):
    delivered, refit = artifact(2.0), artifact(5.0)
    merged = run(tmp_path, delivered, refit)
    assert [r for r in merged["records"] if r["candidate"] != DEMIXED] == [
        r for r in delivered["records"] if r["candidate"] != DEMIXED]
    assert all(r["value"] == 5.0 for r in merged["records"] if r["candidate"] == DEMIXED)
    assert merged["summary"] == summary(merged["records"])
    assert merged["summary"][-1] == delivered["summary"][-1]
    assert merged["demixed_refit"]["refitted_row_count"] == 3


def test_merge_refuses_mismatched_rows_settings_and_irreproducible_summaries(tmp_path):
    with pytest.raises(RefusedMerge, match="row keys differ"):
        run(tmp_path, artifact(2.0), artifact(5.0, sessions=("a", "b")))
    other_design = artifact(5.0)
    other_design["records"][1]["n_trials"] = 11
    with pytest.raises(RefusedMerge, match="different design"):
        run(tmp_path, artifact(2.0), other_design)
    other_settings = artifact(5.0)
    other_settings["scope"]["n_splits"] = 3
    with pytest.raises(RefusedMerge, match="different settings"):
        run(tmp_path, artifact(2.0), other_settings)
    stale = artifact(2.0)
    stale["summary"][0]["difference"] = 99.0
    with pytest.raises(RefusedMerge, match="cannot be reproduced"):
        run(tmp_path, stale, artifact(5.0))


def test_timing_producers_score_only_the_requested_representation(monkeypatch):
    import run_readout_timing_followup as followup
    import run_readout_timing_pilot as pilot

    labels = np.tile([0, 1], 6)
    activity = np.zeros((12, 4, 3))
    scored = []
    monkeypatch.setattr(pilot, "load_entry", lambda entry, lookup: (activity, labels))
    monkeypatch.setattr(pilot, "evaluate_cell", lambda *a, **k: scored.append(a[6:8]) or {"status": "computed"})
    entry = {"region": "hippocampus", "patient": "p", "session": "s", "release": "000469"}
    pilot.process_entry(entry, {}, Path("."), 1, {}, (DEMIXED,))
    assert {candidate for _, candidate in scored} == {DEMIXED} and len(scored) == len(pilot.SPLITS)

    seen = []
    monkeypatch.setattr(followup, "score_decoding", lambda *a, **k: seen.append(a[5]) or {"status": "computed"})
    config = {"models": (), "references": (DEMIXED,), "scores": ("decoding",), "fractions": (0.75,), "seed": 0,
              "n_perm": 1, "checkpoint_dir": Path("."), "runtime": {}}
    followup.evaluate_entry({"region": "hippocampus", "patient": "p", "session": "s", "session_key": "p_s",
                             "level": "load1_maintenance"}, activity, labels, config)
    assert set(seen) == {DEMIXED}


def pooled_artifact(demixed_score, native_score=0.5, kept=2):
    metrics = ("auc", "balanced_accuracy")
    cell = "g|load1|pooled"

    def row(representation, score):
        return {"group": "g", "load": 1, "region": "pooled", "readout": "primary", "representation": representation,
                "session_subset": "all", "n_sessions": 2, "n_patients": 2,
                "latent": {m: {"observed": [score], "above_null": [score]} for m in metrics},
                "native_matched": {m: {"observed": [native_score], "above_null": [native_score]} for m in metrics},
                "latent_minus_native": {m: {"estimate": [score - native_score]} for m in metrics}}

    return {"version": "v", "status": "complete", "chosen_read_out": {"decoder": "linear_svm"},
            "scope": {"seed": 0, "loads": [1], "regions": ["pooled"], "readouts": ["primary"],
                      "representations": ["native", "principal_components", DEMIXED]},
            "session_accounting": [{"session": "a"}], "admission": {cell: {"admitted": 2}},
            "cells": [row("principal_components", 0.6), row(DEMIXED, demixed_score)],
            "representation_sessions": {cell: {"principal_components": {"kept": 2, "excluded": {}},
                                               DEMIXED: {"kept": kept, "excluded": {}}}},
            "fit_timing": {"principal_components": {"fits": 10}, DEMIXED: {"fits": 10 if kept == 2 else 5}}}


def test_pooled_latent_merge_replaces_demixed_rows_and_guards_native_matched_scores(tmp_path):
    import merge_demixed_refit

    def merge(refit):
        (tmp_path / "delivered.json").write_text(json.dumps(pooled_artifact(0.7)))
        refit["cells"] = [r for r in refit["cells"] if r["representation"] == DEMIXED]
        refit["representation_sessions"] = {k: {DEMIXED: v[DEMIXED]} for k, v in refit["representation_sessions"].items()}
        refit["fit_timing"] = {DEMIXED: refit["fit_timing"][DEMIXED]}
        (tmp_path / "refit.json").write_text(json.dumps(refit))
        merge_demixed_refit.main(["pooled_latent_memorandum_decoding", "--delivered", str(tmp_path / "delivered.json"),
                                  "--refit", str(tmp_path / "refit.json"), "--output", str(tmp_path / "out.json")])
        return json.loads((tmp_path / "out.json").read_text())

    delivered, merged = pooled_artifact(0.7), merge(pooled_artifact(0.9, kept=1))
    assert merged["cells"][0] == delivered["cells"][0] and merged["cells"][1]["latent"]["auc"]["observed"] == [0.9]
    sessions = merged["representation_sessions"]["g|load1|pooled"]
    assert sessions["principal_components"]["kept"] == 2 and sessions[DEMIXED]["kept"] == 1
    assert merged["fit_timing"]["principal_components"] == {"fits": 10} and merged["fit_timing"][DEMIXED] == {"fits": 5}
    assert merged["demixed_refit"]["refitted_row_count"] == 1
    with pytest.raises(RefusedMerge, match="native-matched"):
        merge(pooled_artifact(0.9, native_score=0.4))
