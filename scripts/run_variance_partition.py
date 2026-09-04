"""run_variance_partition.py -- census of the four-way single-trial variance
split (across-trial-average time course, per-trial static offset, per-trial
slow decay, per-trial white noise) and the held-position correlation, across
every (dataset, structure, session, epoch, bin width) available through
src/corpus_sessions.py, plus a matched-power comparison against mouse ALM.

Each row gets its own per-session Poisson negative control (src/variance_
partition.poisson_null_from_counts): a session whose four shares cannot be
told apart from a rate-matched population with no trial-specific structure
is exactly as informative as one that was never fitted, so the null is
computed and stored for every row rather than assumed away.

The deciding contrast is delay-epoch, 100 ms bins, one row per session (the
region-pooled population): whether a trial's static offset and its slow
decay each clear their own session's null, tested with an exact paired
sign-flip test across sessions. Mouse ALM, degraded to the human recording
regime (unit count, trial count, and delay length), is the sensitivity
reference for the decay contrast -- the arm that tells "no decay detected"
apart from "no decay resolvable at this sample size".
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from corpus_sessions import (  # noqa: E402
    ALM_WINDOW_S, data_root, iter_alm, iter_all_corpora, load_alm_raw_session, alm_data_directory,
)
from spike_pipeline import build_psth  # noqa: E402
from statistics import paired_sign_flip_test, stable_seed  # noqa: E402
from variance_partition import partition_single_trial_variance, poisson_null_from_counts  # noqa: E402

EPOCHS = ("baseline", "encoding", "delay", "probe")
BIN_WIDTHS_MS = (100.0, 200.0)
N_SPLITS = 8
N_NULL_REPLICATES = 20
MATCHED_POWER_DRAWS = 12
MATCHED_POWER_N_SPLITS = 6
HUMAN_DATASETS = ("dandi_000469", "dandi_001187", "dandi_000574")
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "variance_partition.json"


def _seed(*parts) -> int:
    return stable_seed("|".join(str(p) for p in parts))


def _counts_from_spikes(spike_lists, onset, window_s: float, bin_ms: float) -> np.ndarray:
    rate = build_psth(spike_lists, onset, bin_ms=bin_ms, smooth_ms=0.0, window_s=window_s)
    return rate * (bin_ms / 1000.0)


def _run_row(counts: np.ndarray, bin_ms: float, seed: int) -> dict | None:
    if counts.shape[2] < 2 or counts.shape[0] < 6:
        return None
    partition = partition_single_trial_variance(
        counts, n_splits=N_SPLITS, rng=np.random.default_rng(seed), bin_width_s=bin_ms / 1000.0)
    null = poisson_null_from_counts(
        counts, n_replicates=N_NULL_REPLICATES, n_splits=N_SPLITS,
        rng=np.random.default_rng(seed + 1), bin_width_s=bin_ms / 1000.0)
    return {"partition": partition, "null": null, "n_trials": int(counts.shape[0]),
            "n_units": int(counts.shape[1]), "n_bins": int(counts.shape[2])}


def human_census_rows(root: Path) -> list[dict]:
    rows = []
    t0 = time.time()
    for i, meta in enumerate(iter_all_corpora(root)):
        for epoch in EPOCHS:
            window_s = meta["epoch_windows"][epoch]
            onset = meta["epoch_onsets"][epoch]
            for bin_ms in BIN_WIDTHS_MS:
                counts = _counts_from_spikes(meta["spike_lists"], onset, window_s, bin_ms)
                seed = _seed(meta["dataset"], meta["session"], meta["structure"], epoch, bin_ms)
                run = _run_row(counts, bin_ms, seed)
                if run is None:
                    continue
                rows.append({
                    "dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
                    "structure": meta["structure"], "epoch": epoch, "bin_ms": bin_ms, **run,
                })
        print(f"  session {i + 1}: {meta['dataset']} {meta['session']} {meta['structure']} "
              f"-- {len(rows)} rows so far, {time.time() - t0:.1f}s elapsed", file=sys.stderr, flush=True)
    return rows


def alm_census_rows(root: Path) -> list[dict]:
    rows = []
    for bin_ms in BIN_WIDTHS_MS:
        for meta in iter_alm(root, bin_ms=bin_ms, window_s=ALM_WINDOW_S):
            seed = _seed(meta["dataset"], meta["session"], "delay", bin_ms)
            run = _run_row(meta["counts"], bin_ms, seed)
            if run is None:
                continue
            rows.append({
                "dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
                "structure": "pooled", "epoch": "delay", "bin_ms": bin_ms, **run,
            })
    return rows


def matched_power_alm(root: Path, human_median_units: int, human_median_trials: int, human_window_s: float) -> list[dict]:
    """Four regimes per ALM session: full; unit-matched; unit+trial-matched;
    unit+trial+delay-length-matched -- the arm that turns the human `slow`
    bound from a statement about sample size into a statement about the
    recording. Median over MATCHED_POWER_DRAWS random subsampling draws."""
    directory = alm_data_directory(root)
    results = []
    if not directory.is_dir():
        return results
    for path in sorted(directory.glob("*.mat")):
        session_full = load_alm_raw_session(path, bin_ms=100.0, window_s=ALM_WINDOW_S)
        session_matched_window = load_alm_raw_session(path, bin_ms=100.0, window_s=human_window_s, require_both_arms=False)
        if session_full is None or session_matched_window is None:
            continue
        rng = np.random.default_rng(_seed("matched_power", path.stem))
        regimes = {
            "full": (session_full["control_counts"], None, None),
            "unit_matched": (session_full["control_counts"], human_median_units, None),
            "unit_trial_matched": (session_full["control_counts"], human_median_units, human_median_trials),
            "unit_trial_delay_matched": (session_matched_window["control_counts"], human_median_units, human_median_trials),
        }
        session_result = {"mouse": session_full["mouse"], "session": path.stem, "regimes": {}}
        for regime_name, (counts, n_units_target, n_trials_target) in regimes.items():
            draws = []
            for _ in range(MATCHED_POWER_DRAWS):
                draw_counts = counts
                if n_units_target is not None and n_units_target < draw_counts.shape[1]:
                    unit_idx = rng.choice(draw_counts.shape[1], size=n_units_target, replace=False)
                    draw_counts = draw_counts[:, unit_idx, :]
                if n_trials_target is not None and n_trials_target < draw_counts.shape[0]:
                    trial_idx = rng.choice(draw_counts.shape[0], size=n_trials_target, replace=False)
                    draw_counts = draw_counts[trial_idx]
                result = partition_single_trial_variance(
                    draw_counts, n_splits=MATCHED_POWER_N_SPLITS, rng=rng, bin_width_s=0.1)
                draws.append(result)
            static_values = [d["static_fraction_median"] for d in draws if d["static_fraction_median"] is not None]
            slow_values = [d["slow_fraction_median"] for d in draws if d["slow_fraction_median"] is not None]
            session_result["regimes"][regime_name] = {
                "n_draws_fitted": len(draws),
                "median_static_fraction": float(np.median(static_values)) if static_values else None,
                "median_slow_fraction": float(np.median(slow_values)) if slow_values else None,
                "n_slow_above_0p05": int(sum(1 for v in slow_values if v > 0.05)),
                "n_draws_with_slow_estimate": len(slow_values),
            }
        results.append(session_result)
    return results


def _paired_contrast(rows: list[dict], key: str) -> dict:
    observed, null = [], []
    for row in rows:
        obs = row["partition"].get(f"{key}_fraction_median")
        nul = row["null"].get(f"{key}_null_median")
        if obs is None or nul is None:
            continue
        observed.append(obs)
        null.append(nul)
    n_pairs = len(observed)
    min_attainable_p = 1.0 / (2 ** n_pairs) if n_pairs > 0 else 1.0
    if n_pairs < 4 or min_attainable_p > 0.05:
        return {"status": "underpowered_by_construction", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p}
    test = paired_sign_flip_test(np.array(observed), np.array(null), alternative="greater")
    significant = bool(test["p_value"] <= 0.05)
    return {
        "status": "tested", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p,
        "mean_diff": test["mean_diff"], "p_value": test["p_value"],
        "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"], "significant": significant,
    }


def deciding_contrast(delay_100ms_pooled_human_rows: list[dict], matched_power: list[dict]) -> dict:
    contrast_a = _paired_contrast(delay_100ms_pooled_human_rows, "static")
    contrast_b = _paired_contrast(delay_100ms_pooled_human_rows, "slow")

    matched_slow = [r["regimes"]["unit_trial_delay_matched"]["median_slow_fraction"]
                    for r in matched_power if r["regimes"]["unit_trial_delay_matched"]["median_slow_fraction"] is not None]
    matched_power_alm_slow_median = float(np.median(matched_slow)) if matched_slow else None

    a_significant = contrast_a.get("significant", False)
    b_significant = contrast_b.get("significant", False)
    alm_exceeds_human_ci = (
        matched_power_alm_slow_median is not None
        and contrast_b.get("status") == "tested"
        and matched_power_alm_slow_median > contrast_b["ci_upper"]
    )

    if b_significant:
        branch = "flow_present"
    elif not a_significant:
        branch = "neither_above_noise"
    elif alm_exceeds_human_ci:
        branch = "position_without_flow"
    else:
        # Off the predeclared branch list per house rule 0.9: position clears its
        # null and flow does not, but the matched-power sensitivity condition
        # that would confirm the absence is a floor (rather than a non-result)
        # was not met. Reported explicitly rather than forced onto the nearest
        # of the three predeclared branches.
        branch = "position_without_flow_sensitivity_unconfirmed"

    return {
        "contrast_a_position_static_minus_null": contrast_a,
        "contrast_b_flow_slow_minus_null": contrast_b,
        "matched_power_alm_slow_median_unit_trial_delay_matched": matched_power_alm_slow_median,
        "alm_matched_power_slow_exceeds_human_slow_ci_upper": alm_exceeds_human_ci,
        "branch": branch,
    }


def encoding_vs_delay_discriminator(rows: list[dict]) -> dict:
    """Cheapest discriminator: does `slow` clear its null during
    encoding (a stimulus-driven epoch) in the same sessions where it does not
    during the delay? If so, the absence of flow is a property of the
    maintenance period rather than of human intracranial recording generally."""
    by_epoch = {epoch: [r for r in rows if r["epoch"] == epoch and r["structure"] == "pooled" and r["bin_ms"] == 100.0]
                for epoch in ("encoding", "delay")}
    return {
        "encoding": _paired_contrast(by_epoch["encoding"], "slow"),
        "delay": _paired_contrast(by_epoch["delay"], "slow"),
        "n_sessions_encoding": len(by_epoch["encoding"]),
        "n_sessions_delay": len(by_epoch["delay"]),
    }


def main() -> None:
    root = data_root()
    t_start = time.time()

    print("Loading human corpora...", file=sys.stderr)
    human_rows = human_census_rows(root)
    print(f"  {len(human_rows)} human rows in {time.time() - t_start:.1f}s", file=sys.stderr)

    print("Loading ALM comparison rows...", file=sys.stderr)
    alm_rows = alm_census_rows(root)
    print(f"  {len(alm_rows)} ALM rows in {time.time() - t_start:.1f}s", file=sys.stderr)

    delay_100_pooled_human = [r for r in human_rows
                               if r["epoch"] == "delay" and r["bin_ms"] == 100.0
                               and r["structure"] == "pooled" and r["dataset"] in HUMAN_DATASETS]
    human_median_units = int(np.median([r["n_units"] for r in delay_100_pooled_human])) if delay_100_pooled_human else 0
    human_median_trials = int(np.median([r["n_trials"] for r in delay_100_pooled_human])) if delay_100_pooled_human else 0
    human_window_s = 2.3

    print("Matched-power ALM arm...", file=sys.stderr)
    matched_power = matched_power_alm(root, human_median_units, human_median_trials, human_window_s)
    print(f"  {len(matched_power)} ALM sessions in {time.time() - t_start:.1f}s", file=sys.stderr)

    decision = deciding_contrast(delay_100_pooled_human, matched_power)
    encoding_vs_delay = encoding_vs_delay_discriminator(human_rows)

    per_structure = {}
    structures = sorted({r["structure"] for r in human_rows if r["epoch"] == "delay" and r["bin_ms"] == 100.0})
    for structure in structures:
        rows_s = [r for r in human_rows if r["epoch"] == "delay" and r["bin_ms"] == 100.0 and r["structure"] == structure]
        per_structure[structure] = {
            "n_sessions": len(rows_s),
            "contrast_a_position": _paired_contrast(rows_s, "static"),
            "contrast_b_flow": _paired_contrast(rows_s, "slow"),
        }

    output = {
        "version": "2026-08-10",
        "scope": (
            "Human corpora: DANDI 000469, 001187, 000574 via src/corpus_sessions.iter_all_corpora, "
            "all four epochs, 100 and 200 ms bins, per-session Poisson null (20 replicates). "
            "Mouse ALM (Inagaki) registered as a comparison row plus a four-regime matched-power arm "
            "(unit count, trial count, delay length in bins) against the human median at 100 ms delay. "
            "LFP corpora (ds004752, ds005489, ds005557, tACS) and the Panichello/Watters macaque "
            "attribution arms are NOT included in this run -- see lfp_corpora and macaque_attribution "
            "fields for the explicit reason -- so the deciding contrast covers human single-unit "
            "corpora only, exactly as the predeclared rule specifies."
        ),
        "n_splits": N_SPLITS, "n_null_replicates": N_NULL_REPLICATES,
        "matched_power_draws": MATCHED_POWER_DRAWS,
        "human_median_units_delay_100ms": human_median_units,
        "human_median_trials_delay_100ms": human_median_trials,
        "human_window_s_for_matched_power": human_window_s,
        "deciding_contrast": decision,
        "attribution": {
            "encoding_vs_delay": encoding_vs_delay,
            "macaque_prefrontal_panichello": {
                "status": "not_run",
                "reason": "Deferred under a tight compute-time budget; requires reading area "
                          "labels out of the Panichello .mat files, which this run did not attempt.",
            },
            "macaque_maintenance_watters": {
                "status": "not_run",
                "reason": "Predeclared as a fallback only if the first two discriminators do not "
                          "resolve; not reached here.",
            },
        },
        "lfp_corpora": {
            "status": "not_run",
            "reason": "ds004752, ds005489, ds005557 and the tACS corpus are not yet wired into "
                      "src/corpus_sessions.py's iteration surface; this module covers the three "
                      "registered single-unit corpora plus ALM. Deferred, not dropped.",
        },
        "per_structure_delay_100ms": per_structure,
        "matched_power_alm_sessions": matched_power,
        "human_rows": human_rows,
        "alm_rows": alm_rows,
        "wall_clock_s": time.time() - t_start,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {OUTPUT_PATH} in {time.time() - t_start:.1f}s", file=sys.stderr)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
