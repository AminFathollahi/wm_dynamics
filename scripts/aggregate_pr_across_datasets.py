#!/usr/bin/env python3
"""Consistent PR-vs-load null test across all 6 applicable neural datasets.

All datasets now use the same PR method (spatiotemporal_participation_ratio,
cross-validated, native channel/unit space — src/geometry.py); this script
just aggregates the already-computed per-subject/session pr_cv values into
one LME-style permutation slope per dataset, so the null result can be
reported and plotted on equal footing everywhere rather than only for
Miller/Boran as before.

Updates: results/all_statistics.json — "pr_lme_by_dataset" key
"""
import sys, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from statistics import linear_mixed_effects_test
from provenance import load_overlap_report, linked_duplicate_000673_session_keys, _json_safe

RESULTS = ROOT / "results"
PROVENANCE = ROOT / "provenance"


def pr_lme(pr_records: list[tuple[str, float, float]], rng) -> dict:
    """pr_records: list of (subject_key, load_value, pr_cv)."""
    subj = np.array([r[0] for r in pr_records])
    load = np.array([r[1] for r in pr_records], dtype=float)
    pr = np.array([r[2] for r in pr_records], dtype=float)
    valid = np.isfinite(pr)
    subj, load, pr = subj[valid], load[valid], pr[valid]
    return linear_mixed_effects_test(pr, load, subj, n_perm=5000, rng=rng)


def main():
    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)

    rng = np.random.default_rng(0)
    out = {}

    # Miller: pr_per_trial in 02_geometry files, grouped by task_id (0/1/2)
    records = []
    for subj in ["al", "ca", "cc", "ug"]:
        d = np.load(RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
        pr, tid = d["pr_per_trial"], d["task_id"]
        for load in [0, 1, 2]:
            records.append((subj, load, float(np.mean(pr[tid == load]))))
    out["miller"] = pr_lme(records, rng)

    # Boran iEEG: boran_ctg[subj]["pr_per_set"][{4,6,8}]["pr_cv"]
    records = []
    for subj, v in stats["boran_ctg"].items():
        for ss, prd in v["pr_per_set"].items():
            if isinstance(prd, dict) and np.isfinite(prd.get("pr_cv", np.nan)):
                records.append((subj, float(ss), prd["pr_cv"]))
    out["boran_ieeg"] = pr_lme(records, rng) if records else None

    # Boran single-units
    records = []
    for key, v in stats.get("dandi000574_units_ctg", {}).items():
        for ss, prd in v["pr_per_set"].items():
            if isinstance(prd, dict) and np.isfinite(prd.get("pr_cv", np.nan)):
                records.append((v["subject"], float(ss), prd["pr_cv"]))
    out["boran_units"] = pr_lme(records, rng) if records else None

    # 001187 and 000673 share 31 patients/37 sessions (corrected 2026-08-03 from a prior 16/19 undercount; see
    # provenance/dataset_overlap_report.json); drop 000673's linked-duplicate
    # sessions here so its LME reflects only patients not already counted via 001187.
    overlap_report = load_overlap_report(PROVENANCE)
    linked_keys = (linked_duplicate_000673_session_keys(overlap_report)
                   if overlap_report is not None else set())

    # 000469 / 001187 / 000673: [key]["pr_per_load"][{1,2,3}]["pr_cv"]
    for dset_key, out_key in [("dandi000469_ctg", "dandi000469"),
                              ("dandi001187_ctg", "dandi001187"),
                              ("dandi000673_ctg", "dandi000673")]:
        records = []
        for key, v in stats.get(dset_key, {}).items():
            if dset_key == "dandi000673_ctg" and key in linked_keys:
                continue
            for ld, prd in v["pr_per_load"].items():
                if isinstance(prd, dict) and np.isfinite(prd.get("pr_cv", np.nan)):
                    records.append((key, float(ld), prd["pr_cv"]))
        out[out_key] = pr_lme(records, rng) if records else None

    for k, v in out.items():
        if v is not None:
            print(f"  {k}: beta={v['beta']:.4f}, p={v['p_value']:.4f}, n={len(v) if False else ''}")
        else:
            print(f"  {k}: insufficient data")

    stats["pr_lme_by_dataset"] = out
    with open(RESULTS / "all_statistics.json", "w") as f:
        json.dump(_json_safe(stats), f, indent=2, allow_nan=False)
    print("Updated all_statistics.json[pr_lme_by_dataset]")


if __name__ == "__main__":
    main()
