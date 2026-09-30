#!/usr/bin/env python3
"""Does the ram_ds005489_openloop pre-stimulation-window displacement
(results/stimulation_response_gate_and_panel.json's specificity_check --
0.0673 control-trial SD, p=0.045, a window that closes before that word's
own stimulation begins) come from a mislabelled intra-stimulation effect, or
from a difference between stimulated-block and control-block words that
exists before any stimulation is delivered?

results/train_overlap_decontamination.json already measured, directly from
the release's own event tables, that a NEIGHBOURING word's train is never
still running during a scored word's pre-window (zero contaminated words in
19,624 checked, minimum distance 0.369 s against a 0.3 s window) -- that
account is refuted and is not retested here.

A second account survives: the release delivers one physical stimulation
train per TWO-word stimulated block (verified below, corpus-wide, not
assumed), and the second word of that pair has its own block's train
already running during the interval scored as its pre-stimulation window --
the window is labelled pre-stimulation but the physical condition is
intra-stimulation. The first word of a pair has no such excuse: nothing is
running yet when its own block's train has not started, so a pre-window
displacement in first-in-pair words is a difference stimulation cannot have
caused.

This module partitions every scored stimulated word by within-block
position -- first-in-pair vs second-in-pair, derived from which pulse
train's interval encloses the word's own on-screen presentation window,
ranked by onset within that train's own group of words -- and re-estimates
the pre- and post-stimulation-window displacement separately for each
position, against the SAME control words and the SAME feature path,
windows, admission set and inference unit as the delivered panel.

Nothing here is re-implemented: corpus loading, channel masking and window
construction come from scripts/run_human_stimulation_component_response.py
and scripts/run_stimulation_response_gate_and_panel.py; train/word
event-table geometry (build_trains_openloop, overlaps, read_events) and the
event-table path derivation come from
scripts/run_stimulation_timing_and_parameter_structure.py and
scripts/run_train_overlap_decontamination.py (the sibling test that already
refuted the neighbouring-train account); pooling and the cluster bootstrap
come from src/stimulation_response_estimator.py.

Outputs:
  results/within_block_stimulation_position.json

Run:
    python \
        scripts/run_within_block_stimulation_position.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import canonical_json, git_commit  # noqa: E402
from stimulation_response_estimator import cluster_bootstrap_pooled_effect  # noqa: E402

from run_human_stimulation_component_response import load_corpus as load_human_corpus
from stimulation_events import channel_condition_masks
from corpus_sessions import DATA as OPENLOOP_DATA
from run_ram_openloop_pipeline import BIN_S as HUMAN_BIN_S, PRE_S as HUMAN_PRE_S  # noqa: E402
from run_stimulation_response_gate_and_panel import _read_census_row
from stimulation_events import _human_session_windows, _specificity_check
from stimulation_response_estimator import _pool_arm
from run_stimulation_timing_and_parameter_structure import build_trains_openloop, read_events
from stimulation_events import overlaps
from statistics import _annotate_effect_vs_mdd

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "within_block_stimulation_position.json"

SEED = 20260911  # arbitrary fixed seed; every stochastic step is separately seeded via stable_seed

CORPUS_ID = "ram_ds005489_openloop"

# This project's own internal comparison reference (control-trial SD units), not a
# biological or clinical threshold.
INTERNAL_REFERENCE_SD = 1.0

COUNTERFACTUAL_LABEL = (
    "randomized_contemporaneous_controls: unstimulated WORD presentations within the same "
    "session; ds005489's own assignment is an experimenter-scheduled alternating block design "
    "(census treatment_assignment_classification = experimenter_randomized)."
)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(output))


# ══════════════════════════════════════════════════════════════════════════
# Decision rule -- declared before any session is loaded or any estimate is fit.
# ══════════════════════════════════════════════════════════════════════════

def _build_decision_rule() -> dict:
    return {
        "hypothesis_under_test": (
            "whether the pre-stimulation-window displacement reported for ram_ds005489_openloop "
            "reflects a mislabelled intra-stimulation effect confined to the second word of each "
            "two-word stimulated block (account A), or a stimulated-block-vs-control-block "
            "difference already present before any stimulation is delivered, visible in "
            "first-in-pair words too (account B)."
        ),
        "partition_variable": (
            "within_block_position, derived from the release's own event tables: for every "
            "stimulated WORD, the pulse train whose interval encloses that word's on-screen "
            "presentation window ([onset, onset+duration]) is its own train; stimulated words "
            "sharing the same own train are ranked by onset within that train's own group -- "
            "rank 0 = first_in_pair, rank 1 = second_in_pair, rank >= 2 = "
            "third_or_later_in_group (reported, not assumed away). Control (unstimulated) words "
            "are unchanged and identical across every arm below."
        ),
        "alpha": 0.05, "power": 0.80,
        "internal_reference_control_trial_sd_units": INTERNAL_REFERENCE_SD,
        "branches": {
            "account_b_confirmed": (
                "first_in_pair pre-window pooled p_value <= 0.05 -- a pre-stimulation displacement "
                "present in words whose own block has not started stimulating yet. Checked first "
                "and regardless of the second_in_pair result: if first-in-pair words already show "
                "it, the displacement cannot be attributed to the block's own train."
            ),
            "account_a_confirmed": (
                "second_in_pair pre-window pooled p_value <= 0.05 with a positive mean, AND "
                "first_in_pair pre-window pooled p_value > 0.05 with its minimum detectable "
                f"difference (80% power, whole-participant cluster bootstrap) below the internal "
                f"reference ({INTERNAL_REFERENCE_SD} control-trial SD) -- a genuine, powered null "
                "in first-in-pair words, not an underpowered non-result."
            ),
            "inconclusive_underpowered": (
                "neither position's pre-window pooled p_value is <= 0.05, and at least one "
                "position's minimum detectable difference does not clear the internal reference "
                "above -- the split sample is too small to have been able to detect an effect at "
                "the scale being explained."
            ),
            "neither_position_significant_despite_adequate_power": (
                "neither position's pre-window pooled p_value is <= 0.05, and both positions' "
                "minimum detectable differences clear the internal reference -- an outcome not "
                "anticipated by either named account, reported as such rather than forced into one."
            ),
        },
        "evaluation_order": [
            "account_b_confirmed", "account_a_confirmed",
            "inconclusive_underpowered", "neither_position_significant_despite_adequate_power",
        ],
        "declared_before_fitting": True,
    }


def apply_decision_rule(first_result: dict, second_result: dict) -> dict:
    def _pre_pooled(result: dict):
        w = result.get("windows", {}).get("pre_stimulation_window", {})
        if w.get("status") != "computed":
            return None
        pooled = w.get("rate_free_deviation_biomarker", {}).get("pooled", {})
        return pooled if pooled.get("status") == "computed" else None

    first_pre, second_pre = _pre_pooled(first_result), _pre_pooled(second_result)
    if first_pre is None or second_pre is None:
        return {
            "verdict": "inconclusive_not_computable",
            "reason": "one or both positions' pre-window pooled deviation is not computed",
        }

    first_p, first_mean = first_pre["p_value"], first_pre["mean_value"]
    second_p, second_mean = second_pre["p_value"], second_pre["mean_value"]
    first_mdd_obj, second_mdd_obj = first_pre.get("mdd", {}), second_pre.get("mdd", {})
    first_mdd = first_mdd_obj.get("mdd") if first_mdd_obj.get("status") == "computed" else None
    second_mdd = second_mdd_obj.get("mdd") if second_mdd_obj.get("status") == "computed" else None
    first_powered = first_mdd is not None and first_mdd < INTERNAL_REFERENCE_SD
    second_powered = second_mdd is not None and second_mdd < INTERNAL_REFERENCE_SD

    if first_p <= 0.05:
        verdict = "account_b_confirmed"
    elif second_p <= 0.05 and second_mean is not None and second_mean > 0 and first_powered:
        verdict = "account_a_confirmed"
    elif not first_powered or not second_powered:
        verdict = "inconclusive_underpowered"
    else:
        verdict = "neither_position_significant_despite_adequate_power"

    return {
        "verdict": verdict,
        "first_in_pair_pre_window_p_value": first_p, "first_in_pair_pre_window_mean": first_mean,
        "first_in_pair_pre_window_mdd": first_mdd,
        "first_in_pair_pre_window_mdd_clears_internal_reference": first_powered,
        "second_in_pair_pre_window_p_value": second_p, "second_in_pair_pre_window_mean": second_mean,
        "second_in_pair_pre_window_mdd": second_mdd,
        "second_in_pair_pre_window_mdd_clears_internal_reference": second_powered,
        "internal_reference_control_trial_sd_units": INTERNAL_REFERENCE_SD,
    }


# ══════════════════════════════════════════════════════════════════════════
# Per-word within-block position -- event tables only, no recording signal read.
# ══════════════════════════════════════════════════════════════════════════

def _events_path_for_session(session_key: str) -> Path:
    """Mirrors run_ram_openloop_pipeline.build_session_features's own path
    derivation exactly (same convention scripts/run_train_overlap_decontamination.py
    already uses), from the same session_key the panel and corpus loader use."""
    ieeg_json = OPENLOOP_DATA / session_key
    stem = str(ieeg_json).replace("_ieeg.json", "")
    return Path(stem.replace("_acq-bipolar", "") + "_events.tsv")


def _session_word_positions(session_key: str, pre_window_s: float) -> dict:
    """Per stimulated WORD event in this session, keyed by (list, serialpos)
    -- the same key the trial arrays carry as list_number/serialpos -- its
    within-block position and, as a direct measurement rather than an
    assumption, how much of its own 0.3 s pre-stimulation window overlaps
    the interval of the pulse train that belongs to its own block.

    'Own train' for a word = the train whose [start, end] interval overlaps
    that word's ON-SCREEN presentation window [onset, onset+duration] --
    the identical convention scripts/run_train_overlap_decontamination.py
    and scripts/run_stimulation_timing_and_parameter_structure.py already
    use to identify a word's own train. Stimulated words that share the
    same own train are grouped and ranked by onset within that group; the
    group-size distribution is measured corpus-wide, not assumed to be 2."""
    events_path = _events_path_for_session(session_key)
    if not events_path.exists():
        return {"status": "events_file_missing"}
    events = read_events(events_path)
    words = [e for e in events if e["trial_type"] == "WORD"]
    trains = build_trains_openloop(events)
    stim_words = [w for w in words if w.get("stimulation") == "1"]

    groups: dict[int, list[dict]] = {}
    n_no_own_train = 0
    for w in stim_words:
        onset, on_screen_end = w["_onset"], w["_onset"] + w["_duration"]
        enclosing = [i for i, t in enumerate(trains) if overlaps(t["start"], t["end"], onset, on_screen_end)]
        if not enclosing:
            n_no_own_train += 1
            continue
        groups.setdefault(enclosing[0], []).append(w)

    group_size_histogram = Counter(len(members) for members in groups.values())

    by_key: dict[tuple[int, int], dict] = {}
    for train_idx, members in groups.items():
        members_sorted = sorted(members, key=lambda w: w["_onset"])
        t = trains[train_idx]
        for rank, w in enumerate(members_sorted):
            list_raw = w.get("list")
            try:
                list_id = int(list_raw) if list_raw not in (None, "", "n/a", "-1") else None
            except ValueError:
                list_id = None
            try:
                serialpos = int(w.get("serialpos"))
            except (TypeError, ValueError):
                serialpos = None
            if list_id is None or serialpos is None:
                continue
            onset = w["_onset"]
            pre0, pre1 = onset - pre_window_s, onset
            coverage_s = max(0.0, min(pre1, t["end"]) - max(pre0, t["start"]))
            position = (
                "first_in_pair" if rank == 0 else
                "second_in_pair" if rank == 1 else
                "third_or_later_in_group"
            )
            by_key[(list_id, serialpos)] = {
                "within_block_position": position,
                "rank_in_train_group": rank,
                "train_group_size": len(members_sorted),
                "own_train_pre_window_coverage_s": coverage_s,
                "own_train_pre_window_coverage_fraction": (coverage_s / pre_window_s) if pre_window_s > 0 else None,
            }

    return {
        "status": "computed",
        "by_list_serialpos": by_key,
        "n_trains_in_session": len(trains),
        "n_stim_words_in_session": len(stim_words),
        "n_stim_words_no_own_train_identified": n_no_own_train,
        "train_group_size_histogram": {str(k): v for k, v in group_size_histogram.items()},
    }


# ══════════════════════════════════════════════════════════════════════════
# Corpus pass: build overall / first-in-pair / second-in-pair per-session rows
# ══════════════════════════════════════════════════════════════════════════

def build_arm_rows(pre_bins: int, pre_window_s: float) -> dict:
    human_corpus = load_human_corpus("open_loop_ds005489", OPENLOOP_DATA, derive_stim_from_stim_on=False, smoke=None)

    overall_rows, first_rows, second_rows = [], [], []
    session_diagnostics: dict[str, dict] = {}
    extra_excluded: dict[str, list[str]] = {}
    pooled_train_hist: Counter = Counter()
    coverage_by_position: dict[str, list[float]] = {"first_in_pair": [], "second_in_pair": []}
    totals = dict(n_trials=0, n_stim_trials=0, n_control_trials=0,
                  n_first_in_pair=0, n_second_in_pair=0, n_third_or_later_in_group=0,
                  n_stim_unmatched_to_event_table=0)

    for rec in human_corpus["records"]:
        arrays = rec["arrays"]
        ch_names = arrays["ch_names"].tolist()
        mask = channel_condition_masks(
            ch_names, str(arrays["anode"]), str(arrays["cathode"]), str(arrays["stim_channel"])
        )["excluding_stimulated_shank"]
        if mask.sum() < 1:
            extra_excluded.setdefault("no_channels_survive_shank_exclusion", []).append(rec["session_key"])
            continue

        info = _session_word_positions(rec["session_key"], pre_window_s)
        if info["status"] != "computed":
            extra_excluded.setdefault(info["status"], []).append(rec["session_key"])
            continue

        for size_key, count in info["train_group_size_histogram"].items():
            pooled_train_hist[size_key] += count

        list_arr = np.asarray(arrays["list_number"])
        serial_arr = np.asarray(arrays["serialpos"])
        stim_arr = np.asarray(arrays["stim_flag"])
        n_trials = int(len(list_arr))
        position_labels = np.array(["not_stimulated"] * n_trials, dtype=object)

        for i in range(n_trials):
            if stim_arr[i] != 1:
                continue
            entry = info["by_list_serialpos"].get((int(list_arr[i]), int(serial_arr[i])))
            if entry is None:
                position_labels[i] = "stim_word_unmatched_to_event_table"
                continue
            position_labels[i] = entry["within_block_position"]
            if entry["within_block_position"] in coverage_by_position:
                coverage_by_position[entry["within_block_position"]].append(
                    entry["own_train_pre_window_coverage_fraction"])

        n_stim_trials = int((stim_arr == 1).sum())
        n_control_trials = int((stim_arr == 0).sum())
        n_first = int((position_labels == "first_in_pair").sum())
        n_second = int((position_labels == "second_in_pair").sum())
        n_third = int((position_labels == "third_or_later_in_group").sum())
        n_unmatched = int((position_labels == "stim_word_unmatched_to_event_table").sum())

        session_diagnostics[rec["session_key"]] = {
            "n_trials": n_trials, "n_stim_trials": n_stim_trials, "n_control_trials": n_control_trials,
            "n_first_in_pair": n_first, "n_second_in_pair": n_second,
            "n_third_or_later_in_group": n_third, "n_stim_unmatched_to_event_table": n_unmatched,
            "n_trains_in_session": info["n_trains_in_session"],
            "n_stim_words_in_session_event_table": info["n_stim_words_in_session"],
            "n_stim_words_no_own_train_identified_event_table": info["n_stim_words_no_own_train_identified"],
            "train_group_size_histogram": info["train_group_size_histogram"],
        }
        totals["n_trials"] += n_trials
        totals["n_stim_trials"] += n_stim_trials
        totals["n_control_trials"] += n_control_trials
        totals["n_first_in_pair"] += n_first
        totals["n_second_in_pair"] += n_second
        totals["n_third_or_later_in_group"] += n_third
        totals["n_stim_unmatched_to_event_table"] += n_unmatched

        overall_windows = _human_session_windows(rec, pre_bins, mask)
        overall_rows.append({"session_key": rec["session_key"], "cluster_id": rec["subject_id"], "windows": overall_windows})

        control_mask_bool = stim_arr == 0
        for target_label, target_rows in (("first_in_pair", first_rows), ("second_in_pair", second_rows)):
            target_mask = position_labels == target_label
            if int(target_mask.sum()) < 1:
                continue
            keep = control_mask_bool | target_mask
            sub_arrays = {"epochs_log": arrays["epochs_log"][keep], "stim_flag": arrays["stim_flag"][keep]}
            sub_windows = _human_session_windows({"arrays": sub_arrays}, pre_bins, mask)
            target_rows.append({"session_key": rec["session_key"], "cluster_id": rec["subject_id"], "windows": sub_windows})

    coverage_summary = {}
    for position, values in coverage_by_position.items():
        arr = np.asarray([v for v in values if v is not None], dtype=float)
        coverage_summary[position] = {
            "n_words": int(len(arr)),
            "mean_fraction": float(arr.mean()) if len(arr) else None,
            "median_fraction": float(np.median(arr)) if len(arr) else None,
            "fraction_of_words_fully_covering_own_pre_window": float(np.mean(arr >= 0.999)) if len(arr) else None,
            "fraction_of_words_with_zero_coverage": float(np.mean(arr <= 0.001)) if len(arr) else None,
        }

    return {
        "human_corpus_admission": {
            "n_sessions_seen": human_corpus["n_sessions_total"], "n_sessions_used": len(overall_rows),
            "excluded_by_reason": {**human_corpus["exclusions"], **extra_excluded},
        },
        "overall_rows": overall_rows, "first_in_pair_rows": first_rows, "second_in_pair_rows": second_rows,
        "session_diagnostics": session_diagnostics,
        "trial_totals": totals,
        "trial_reconciliation": (
            "n_trials == n_stim_trials + n_control_trials; n_stim_trials == n_first_in_pair + "
            "n_second_in_pair + n_third_or_later_in_group + n_stim_unmatched_to_event_table"
        ),
        "pooled_train_group_size_histogram": dict(pooled_train_hist),
        "own_train_pre_window_coverage_by_position": coverage_summary,
    }


# ══════════════════════════════════════════════════════════════════════════
# Post-minus-pre difference, per arm -- an interval this quantity never had.
# ══════════════════════════════════════════════════════════════════════════

def _post_minus_pre_difference(result: dict, seed_tag: str) -> dict:
    windows = result.get("windows", {})
    pre, post = windows.get("pre_stimulation_window", {}), windows.get("post_stimulation_window", {})
    if pre.get("status") != "computed" or post.get("status") != "computed":
        return {"status": "not_computable", "reason": "one or both windows are a structural void or not computed"}
    pre_sessions = pre.get("per_session", {})
    post_sessions = post.get("per_session", {})
    diffs, cluster_ids = [], []
    for session_key, pre_row in pre_sessions.items():
        post_row = post_sessions.get(session_key)
        if post_row is None:
            continue
        pv = pre_row.get("rate_free_deviation_normalised_displacement")
        qv = post_row.get("rate_free_deviation_normalised_displacement")
        if pv is None or qv is None:
            continue
        diffs.append(qv - pv)
        cluster_ids.append(pre_row["cluster_id"])
    if not diffs:
        return {"status": "not_computable", "reason": "no session has both a computed pre-window and post-window displacement"}
    pooled = cluster_bootstrap_pooled_effect(np.array(diffs), cluster_ids, seed_tag)
    pooled["n_sessions_contributing"] = len(diffs)
    if pooled.get("status") == "computed" and pooled.get("mdd", {}).get("status") == "computed":
        pooled["effect_below_own_mdd"] = bool(abs(pooled["mean_value"]) < pooled["mdd"]["mdd"])
    return pooled


def main() -> None:
    t0 = time.time()
    output = {
        "version": "2026-09-11", "seed": SEED, "status": "running",
        "decision_rule_declared_before_fitting": _build_decision_rule(),
    }
    _flush(output)

    census_row = _read_census_row(CORPUS_ID)
    windows_measured = {
        "pre_stimulation_window": census_row["pre_stimulation_window_s"],
        "post_stimulation_window": census_row["post_stimulation_windows_through_probe_s"],
    }
    pre_window_s = census_row["pre_stimulation_window_s"]["value"]
    pre_bins = int(round(HUMAN_PRE_S / HUMAN_BIN_S))

    built = build_arm_rows(pre_bins, pre_window_s)
    output["scope"] = {
        "corpus": CORPUS_ID,
        "human_corpus_admission": built["human_corpus_admission"],
        "trial_totals": built["trial_totals"],
        "trial_reconciliation": built["trial_reconciliation"],
        "pooled_train_group_size_histogram": built["pooled_train_group_size_histogram"],
        "own_train_pre_window_coverage_by_position": built["own_train_pre_window_coverage_by_position"],
        "pre_window_seconds": pre_window_s,
        "attribution_limit": census_row["item_overlap"]["value"]["reading"],
        "n_perm": 10000, "n_boot": 5000, "alpha": 0.05, "power": 0.80,
    }
    output["session_diagnostics"] = built["session_diagnostics"]
    output["status"] = "arms_pending"
    _flush(output)

    arms = {}
    for arm_name, rows in (
        ("overall_reproduction", built["overall_rows"]),
        ("first_in_pair", built["first_in_pair_rows"]),
        ("second_in_pair", built["second_in_pair_rows"]),
    ):
        result = _pool_arm(f"{CORPUS_ID}__{arm_name}", rows, COUNTERFACTUAL_LABEL, windows_measured)
        _annotate_effect_vs_mdd(result)
        result["specificity_check"] = _specificity_check(result)
        result["post_minus_pre_difference"] = _post_minus_pre_difference(
            result, f"within_block|{arm_name}|post_minus_pre")
        result["n_sessions_contributing_rows"] = len(rows)
        arms[arm_name] = result
        output["arms"] = arms
        _flush(output)

    output["decision"] = apply_decision_rule(arms["first_in_pair"], arms["second_in_pair"])
    output["wall_clock_s"] = time.time() - t0
    output["code_commit"] = git_commit(ROOT)
    output["status"] = "complete"
    _flush(output)
    print(json.dumps({"verdict": output["decision"]["verdict"], "status": output["status"]}, indent=2))


if __name__ == "__main__":
    main()
