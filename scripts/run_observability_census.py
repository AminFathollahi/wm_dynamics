#!/usr/bin/env python3
"""Observability census: for every (dataset, structure, session, epoch) the
project's human single-unit and mouse-ALM loaders can produce, at 100 and
200 ms bins, measure the cross-validated share of the leading latent's
held-out variance that is white per-trial observation noise
(src/observability.py's nugget_fraction).

Deciding contrast: human single-unit corpora versus mouse ALM (the positive
control this estimator recovers a near-zero nugget on), delay epoch, pooled
structure, 100 ms bins, Mann-Whitney U with sessions as the unit -- reported
both unmatched and with ALM subsampled to the human median trial count.

A secondary contrast asks whether the human floor is a property of the
recording regime or of how much task signal a given epoch carries: paired
within-session nugget difference, encoding minus delay, human corpora only.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus_sessions import (  # noqa: E402
    BORAN_EPOCH_WINDOWS_S,
    EPOCH_WINDOWS_S,
    data_root,
    iter_alm,
    iter_all_corpora,
)
from observability import nugget_fraction  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from spike_pipeline import build_psth  # noqa: E402
from statistics import paired_sign_flip_test, stable_seed  # noqa: E402

SEED = 20260809
BIN_WIDTHS_MS = (100, 200)
N_SPLITS = 32
OUTPUT_PATH = ROOT / "results" / "observability_census.json"

EPOCH_WINDOWS_BY_DATASET = {
    "dandi_000469": EPOCH_WINDOWS_S,
    "dandi_001187": EPOCH_WINDOWS_S,
    "dandi_000574": BORAN_EPOCH_WINDOWS_S,
}


def _row_seed(*parts: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed("|".join(str(p) for p in parts)) ^ SEED)


def census_row(dataset: str, patient: str, session: str, structure: str, epoch: str,
                bin_ms: float, counts: np.ndarray) -> dict:
    rng = _row_seed(dataset, session, structure, epoch, str(bin_ms))
    result = nugget_fraction(counts, n_splits=N_SPLITS, rng=rng, bin_width_s=bin_ms / 1000.0)
    result.pop("per_split")
    row = {
        "dataset": dataset, "patient": patient, "session": session, "structure": structure,
        "epoch": epoch, "bin_ms": bin_ms, "modality": "single_unit",
    }
    row.update(result)
    return row


def build_human_and_alm_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    n_sessions_seen = 0
    t0 = time.time()
    for entry in iter_all_corpora(root):
        n_sessions_seen += 1
        epoch_windows = EPOCH_WINDOWS_BY_DATASET[entry["dataset"]]
        for epoch, onsets in entry["epoch_onsets"].items():
            window_s = epoch_windows[epoch]
            for bin_ms in BIN_WIDTHS_MS:
                counts = build_psth(entry["spike_lists"], onsets, bin_ms=bin_ms, smooth_ms=0, window_s=window_s)
                rows.append(census_row(entry["dataset"], entry["patient"], entry["session"],
                                        entry["structure"], epoch, bin_ms, counts))
        if n_sessions_seen % 10 == 0:
            print(f"  ...{n_sessions_seen} (dataset, structure) session rows in {time.time() - t0:.0f}s", file=sys.stderr)

    for bin_ms in BIN_WIDTHS_MS:
        for entry in iter_alm(root, bin_ms=bin_ms, window_s=2.0):
            rows.append(census_row(entry["dataset"], entry["patient"], entry["session"],
                                    entry["structure"], entry["epoch"], bin_ms, entry["counts"]))
    return rows


ALM_TRIAL_MATCHED_STRUCTURE = "pooled_trial_matched_to_human_median"


def build_alm_trial_matched_rows(root: Path, target_trials: int, bin_ms: int = 100) -> list[dict]:
    """ALM sessions with their control trials
    randomly subsampled down to the human median trial count, so the
    calibration comparison also holds trial count fixed."""
    rows = []
    for entry in iter_alm(root, bin_ms=bin_ms, window_s=2.0):
        counts = entry["counts"]
        if counts.shape[0] < target_trials:
            continue
        rng = _row_seed(entry["dataset"], entry["session"], "trial_matched", str(bin_ms))
        idx = rng.choice(counts.shape[0], size=target_trials, replace=False)
        rows.append(census_row(entry["dataset"], entry["patient"], entry["session"],
                                ALM_TRIAL_MATCHED_STRUCTURE, entry["epoch"], bin_ms, counts[idx]))
    return rows


def deciding_contrast(rows: list[dict]) -> dict:
    """Human single-unit pooled-structure delay-epoch 100ms nugget
    versus mouse ALM, unmatched and trial-count-matched."""
    human = [r for r in rows if r["modality"] == "single_unit" and r["dataset"] != "inagaki_alm5"
             and r["structure"] == "pooled" and r["epoch"] == "delay" and r["bin_ms"] == 100
             and r["status"] == "fitted"]
    alm = [r for r in rows if r["dataset"] == "inagaki_alm5" and r["structure"] == "pooled"
           and r["epoch"] == "delay" and r["bin_ms"] == 100 and r["status"] == "fitted"]
    human_vals = np.array([r["median_nugget_fraction"] for r in human])
    alm_vals = np.array([r["median_nugget_fraction"] for r in alm])
    if len(human_vals) == 0 or len(alm_vals) == 0:
        return {
            "status": "not_estimable", "reason": "no fitted session on one arm",
            "n_human": int(len(human_vals)), "n_alm": int(len(alm_vals)),
        }
    u_stat, u_p = mannwhitneyu(human_vals, alm_vals, alternative="two-sided")

    human_trial_counts = [r["n_trials"] for r in human]
    median_human_trials = int(np.median(human_trial_counts)) if human_trial_counts else None
    matched_alm = [r for r in rows if r["dataset"] == "inagaki_alm5" and r["structure"] == ALM_TRIAL_MATCHED_STRUCTURE
                   and r["epoch"] == "delay" and r["bin_ms"] == 100 and r["status"] == "fitted"]
    matched_alm_vals = np.array([r["median_nugget_fraction"] for r in matched_alm])
    if len(matched_alm_vals) >= 2:
        u_stat_matched, u_p_matched = mannwhitneyu(human_vals, matched_alm_vals, alternative="two-sided")
        matched = {
            "median_human_trial_count": median_human_trials,
            "n_alm_matched": int(len(matched_alm_vals)),
            "alm_matched_median": float(np.median(matched_alm_vals)),
            "u_statistic": float(u_stat_matched), "p_value": float(u_p_matched),
        }
    else:
        matched = {"status": "not_estimable", "reason": "fewer than 2 ALM sessions at the human median trial count",
                   "median_human_trial_count": median_human_trials}

    human_median = float(np.median(human_vals))
    alm_median = float(np.median(alm_vals))
    if human_median >= 0.80 and alm_median <= 0.20 and matched.get("alm_matched_median", 1.0) <= 0.20:
        branch = "human_observability_floor_confirmed"
    elif human_median < 0.50:
        branch = "no_observability_gap"
    else:
        branch = "observability_structure_dependent"

    per_structure_low = []
    for structure in sorted({r["structure"] for r in rows if r["dataset"] != "inagaki_alm5"}):
        vals = [r["median_nugget_fraction"] for r in rows
                if r["structure"] == structure and r["epoch"] == "delay" and r["bin_ms"] == 100
                and r["status"] == "fitted" and r["dataset"] != "inagaki_alm5"]
        n_patients = len({r["patient"] for r in rows
                           if r["structure"] == structure and r["epoch"] == "delay" and r["bin_ms"] == 100
                           and r["status"] == "fitted" and r["dataset"] != "inagaki_alm5"})
        if vals and n_patients >= 3 and float(np.median(vals)) < 0.50:
            per_structure_low.append({"structure": structure, "median": float(np.median(vals)), "n_patients": n_patients})
    if per_structure_low and branch == "observability_structure_dependent":
        branch = "observability_structure_dependent"

    return {
        "human_median_nugget_fraction": human_median,
        "alm_median_nugget_fraction": alm_median,
        "n_human_sessions": int(len(human_vals)),
        "n_alm_sessions": int(len(alm_vals)),
        "unmatched": {"u_statistic": float(u_stat), "p_value": float(u_p)},
        "trial_count_matched": matched,
        "structures_with_low_median_n_ge_3_patients": per_structure_low,
        "branch": branch,
    }


def encoding_vs_delay_contrast(rows: list[dict]) -> dict:
    """Paired within-session nugget difference, encoding minus delay,
    human corpora, 100 ms bins, exact paired sign-flip test."""
    by_key: dict[tuple, dict[str, float]] = {}
    for r in rows:
        if r["dataset"] == "inagaki_alm5" or r["bin_ms"] != 100 or r["status"] != "fitted":
            continue
        if r["epoch"] not in ("encoding", "delay"):
            continue
        key = (r["dataset"], r["session"], r["structure"])
        by_key.setdefault(key, {})[r["epoch"]] = r["median_nugget_fraction"]
    paired = [(v["encoding"], v["delay"]) for v in by_key.values() if "encoding" in v and "delay" in v]
    n_pairs = len(paired)
    # An exact sign-flip null over n paired differences has only 2**n
    # distinct sign patterns; no Monte Carlo resolution can attain a
    # smaller p than that combinatorial floor.
    min_attainable_p = 1.0 / (2 ** n_pairs) if n_pairs > 0 else 1.0
    if n_pairs < 4 or min_attainable_p > 0.05:
        return {"status": "underpowered_by_construction", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p}
    encoding_vals = np.array([p[0] for p in paired])
    delay_vals = np.array([p[1] for p in paired])
    test = paired_sign_flip_test(encoding_vals, delay_vals, alternative="less", rng=np.random.default_rng(SEED))
    reduction = -test["mean_diff"]  # delay minus encoding: positive means encoding is lower
    if test["p_value"] < 0.05 and reduction > 0.20:
        branch = "signal_strength_limited"
    else:
        branch = "regime_limited"
    return {
        "n_pairs": n_pairs, "mean_encoding_minus_delay": test["mean_diff"],
        "mean_reduction_delay_minus_encoding": reduction, "p_value": test["p_value"],
        "min_attainable_p": min_attainable_p, "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        "branch": branch,
    }


def main() -> None:
    root = data_root()
    print("Building human single-unit + ALM census rows...", file=sys.stderr)
    rows = build_human_and_alm_rows(root)
    print(f"  {len(rows)} rows total", file=sys.stderr)

    human_delay_pooled = [r["n_trials"] for r in rows if r["dataset"] != "inagaki_alm5"
                           and r["structure"] == "pooled" and r["epoch"] == "delay"
                           and r["bin_ms"] == 100 and r["status"] == "fitted"]
    if human_delay_pooled:
        median_human_trials = int(np.median(human_delay_pooled))
        print(f"Subsampling ALM to the human median trial count ({median_human_trials})...", file=sys.stderr)
        rows.extend(build_alm_trial_matched_rows(root, median_human_trials))

    payload = {
        "schema_version": "1.0.0",
        "seed": SEED,
        "code_commit": git_commit(ROOT),
        "n_splits": N_SPLITS,
        "bin_widths_ms": list(BIN_WIDTHS_MS),
        "rows": rows,
        "deciding_contrast_human_vs_alm": deciding_contrast(rows),
        "encoding_vs_delay": encoding_vs_delay_contrast(rows),
        "lfp_corpora": {
            "status": "not_run",
            "reason": ("This analysis extended the census to the human single-unit corpora and "
                       "the mouse ALM positive control only. The LFP corpora (ds004752, "
                       "ds005489, ds005557, the tACS corpus) support the same estimator in "
                       "principle -- it needs only a binned multi-channel population matrix -- "
                       "but building region-resolved multi-channel loaders for four additional "
                       "recording pipelines did not fit in the time budget available here. Declared "
                       "explicitly rather than silently omitted; still outstanding."),
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(payload))
    print(f"Wrote {OUTPUT_PATH}", file=sys.stderr)
    print(json.dumps(payload["deciding_contrast_human_vs_alm"], indent=2))
    print(json.dumps(payload["encoding_vs_delay"], indent=2))


if __name__ == "__main__":
    main()
