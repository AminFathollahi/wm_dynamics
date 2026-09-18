#!/usr/bin/env python3
"""Does neighbouring-train contamination explain why the ram_ds005489_openloop
pre-stimulation-window displacement of the rate-free deviation biomarker
(results/stimulation_response_gate_and_panel.json) is itself significant --
a window that ends before the scored word's own stimulation begins and so
cannot carry that word's own treatment effect?

The corpus's own design census (results/stimulation_design_census.json,
arm encoding_alternating_block_open_loop) already measured that a
stimulation train (median 4.6 s) outlasts the median item-presentation
spacing (2.5 s), spanning a mean of 2.00 item windows -- so some words
labelled unstimulated may still be recorded while a NEIGHBOURING word's
train is running. This module tests that account directly: for every
scored WORD event in the 73 admitted sessions, it determines from the raw
BIDS event tables (no recording signal) whether that word's own
pre-stimulation window overlaps a stimulation train OTHER than its own (a
stimulated word's own train is explicitly excluded from that check --
STIM_ON is measured here to precede its own WORD onset by a small real
trigger-latency offset, which otherwise puts a word's own train inside
its own pre-window and has nothing to do with the neighbouring-train
account under test), then reruns the same pre/post displacement estimator
(src/stimulation_response_estimator.py, unchanged) separately on the
uncontaminated subset (no active neighbouring train in the pre-window) and
the contaminated complement.

Everything reused, nothing re-implemented: corpus loading and channel
masking from scripts/run_human_stimulation_component_response.py; window
construction and cluster-bootstrap pooling from
scripts/run_stimulation_response_gate_and_panel.py; train/word event-table
geometry (build_trains_openloop, overlaps, read_events) from
scripts/run_stimulation_timing_and_parameter_structure.py.

Outputs:
  results/train_overlap_decontamination.json

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_train_overlap_decontamination.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import canonical_json, git_commit  # noqa: E402

from run_human_stimulation_component_response import (  # noqa: E402
    OPENLOOP_DATA, channel_condition_masks, load_corpus as load_human_corpus,
)
from run_ram_openloop_pipeline import BIN_S as HUMAN_BIN_S, PRE_S as HUMAN_PRE_S  # noqa: E402
from run_stimulation_response_gate_and_panel import (  # noqa: E402
    _human_session_windows, _pool_arm, _read_census_row, _specificity_check,
)
from run_stimulation_timing_and_parameter_structure import (  # noqa: E402
    build_trains_openloop, overlaps, read_events,
)

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "train_overlap_decontamination.json"

SEED = 20260911  # arbitrary fixed seed; every stochastic step is separately seeded via stable_seed

CORPUS_ID = "ram_ds005489_openloop"

# Reference values read from the already-delivered panel (never recomputed here as a gate --
# used only as the target effect size a power check must be able to reach).
DELIVERED_PANEL_PATH = RESULTS / "stimulation_response_gate_and_panel.json"

# This project's own internal comparison reference (control-trial SD units), not a
# biological or clinical threshold -- see docs/mandates/STANDING_CONSTRAINTS.md.
INTERNAL_REFERENCE_SD = 1.0

COUNTERFACTUAL_LABEL = (
    "randomized_contemporaneous_controls: unstimulated WORD presentations within the same "
    "session; ds005489's own assignment is an experimenter-scheduled alternating block design "
    "(census treatment_assignment_classification = experimenter_randomized)."
)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(output))


def _load_delivered_reference() -> dict:
    if not DELIVERED_PANEL_PATH.exists():
        return {"status": "not_computable", "reason": "results/stimulation_response_gate_and_panel.json is absent"}
    delivered = json.loads(DELIVERED_PANEL_PATH.read_text())
    spec = delivered.get("stimulation_response_panel", {}).get(CORPUS_ID, {}).get("specificity_check", {})
    if spec.get("status") != "computed":
        return {"status": "not_computable", "reason": "delivered specificity_check is not status=computed"}
    return {
        "status": "computed",
        "source": (
            f"results/stimulation_response_gate_and_panel.json -> stimulation_response_panel.{CORPUS_ID}.specificity_check"
        ),
        "pre_window_mean": spec["pre_window_mean"], "pre_window_p_value": spec["pre_window_p_value"],
        "post_window_mean": spec["post_window_mean"], "post_window_p_value": spec["post_window_p_value"],
    }


# ══════════════════════════════════════════════════════════════════════════
# Decision rule -- declared before any session is loaded or any estimate is fit.
# ══════════════════════════════════════════════════════════════════════════

def _build_decision_rule(reference_pre_window_displacement: float | None) -> dict:
    return {
        "hypothesis_under_test": (
            "a stimulation train from a NEIGHBOURING word is still active during a word's own "
            "pre-stimulation window, and this -- not the word's own stimulation acting backward in "
            "time -- explains why the pre-stimulation-window displacement is itself significant."
        ),
        "partition_variable": (
            "for every scored WORD event, whether its own pre-stimulation window "
            "[onset - pre_stimulation_window_s, onset) overlaps at least one stimulation train "
            "OTHER than its own (a stimulated word's own train, identified as the train "
            "overlapping that word's ON-SCREEN presentation window, is excluded from this check). "
            "'uncontaminated subset' = no such neighbouring-train overlap. 'contaminated subset' = "
            "the complement."
        ),
        "alpha": 0.05, "power": 0.80,
        "reference_pre_window_displacement_to_explain_control_trial_sd_units": reference_pre_window_displacement,
        "reference_pre_window_displacement_source": (
            "results/stimulation_response_gate_and_panel.json's own delivered "
            f"{CORPUS_ID} specificity_check.pre_window_mean"
        ),
        "internal_reference_control_trial_sd_units": INTERNAL_REFERENCE_SD,
        "supported_if": (
            "in the uncontaminated subset, the pre-window pooled p-value is > 0.05 (a null) AND its "
            "minimum detectable difference (80% power, whole-participant cluster bootstrap) is BELOW "
            "the reference pre-window displacement above (so the null is a genuine, powered null, not "
            "an underpowered non-result) AND the post-window pooled p-value is <= 0.05 with a "
            "positive mean (the same sign as the delivered post-window effect) -- i.e. the "
            "null-before/effect-after pattern is restored once contaminated words are removed."
        ),
        "refuted_if": (
            "in the uncontaminated subset, the pre-window pooled p-value is <= 0.05 -- a "
            "pre-stimulation displacement remains detectable even with every word whose pre-window "
            "overlaps a neighbouring train removed, which means something other than train overlap "
            "(e.g. serial-position structure, session-level drift, or a stimulation-block-vs-"
            "control-block difference that is not stimulation itself) separates stimulated from "
            "control words. This is reported as a more serious result than confirmation, not a "
            "disappointing one, per this project's standing rule."
        ),
        "inconclusive_if": (
            "the pre-window p-value is > 0.05 in the uncontaminated subset but its minimum detectable "
            "difference does not clear the reference pre-window displacement above -- the subset is "
            "too small to have been ABLE to detect an effect of the size being explained, so a null "
            "reading would be an infeasibility dressed as a result, not a genuine null."
        ),
        "declared_before_fitting": True,
    }


# ══════════════════════════════════════════════════════════════════════════
# Per-word train-overlap geometry -- event tables only, no recording signal read.
# ══════════════════════════════════════════════════════════════════════════

def _events_path_for_session(session_key: str) -> Path:
    """Mirrors run_ram_openloop_pipeline.build_session_features's own path
    derivation exactly, from the same session_key the panel and corpus
    loader already use (str(ieeg_json.relative_to(OPENLOOP_DATA)))."""
    ieeg_json = OPENLOOP_DATA / session_key
    stem = str(ieeg_json).replace("_ieeg.json", "")
    return Path(stem.replace("_acq-bipolar", "") + "_events.tsv")


def _session_word_contamination(session_key: str, pre_window_s: float) -> dict:
    """Per WORD event in this session, keyed by (list, serialpos) -- the
    same key build_session_features's own kept-trial arrays carry as
    list_number/serialpos -- whether the word's pre-stimulation window
    overlaps a NEIGHBOURING word's stimulation train, and the signed
    distance from the word's onset to the offset of the most recent
    neighbouring train that started at or before that onset (negative =
    that train is still active at onset).

    A stimulated word's OWN train is excluded from both computations. The
    corpus's STIM_ON onset is measured here to precede its own WORD onset
    by a small, real trigger-latency offset (about 0.2 s in a spot check),
    which puts the word's own train inside its own 0.3 s pre-window under a
    naive all-trains overlap check -- that is an artefact of onset
    alignment, not the neighbouring-train contamination this test is
    for. 'Own train' is identified the same way
    run_stimulation_timing_and_parameter_structure.py's own
    process_openloop_session already does: the train overlapping the
    word's ON-SCREEN presentation window [onset, onset + duration], not
    the pre-window."""
    events_path = _events_path_for_session(session_key)
    if not events_path.exists():
        return {"status": "events_file_missing"}
    events = read_events(events_path)
    words = [e for e in events if e["trial_type"] == "WORD"]
    trains = build_trains_openloop(events)

    by_key = {}
    for w in words:
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
        onset, on_screen_end = w["_onset"], w["_onset"] + w["_duration"]
        own_train_enclosing = [t for t in trains if overlaps(t["start"], t["end"], onset, on_screen_end)]
        own_train = own_train_enclosing[0] if own_train_enclosing else None
        other_trains = [t for t in trains if t is not own_train]

        pre0, pre1 = onset - pre_window_s, onset
        active_trains = [t for t in other_trains if overlaps(pre0, pre1, t["start"], t["end"])]
        preceding = [t for t in other_trains if t["start"] <= onset]
        governing = max(preceding, key=lambda t: t["start"]) if preceding else None
        by_key[(list_id, serialpos)] = {
            "pre_window_contaminated": len(active_trains) > 0,
            "n_trains_active_in_pre_window": len(active_trains),
            "distance_to_preceding_train_offset_s": (onset - governing["end"]) if governing is not None else None,
            "has_own_train": own_train is not None,
        }
    return {"status": "computed", "by_list_serialpos": by_key, "n_trains_in_session": len(trains)}


# ══════════════════════════════════════════════════════════════════════════
# Corpus pass: build full / uncontaminated / contaminated per-session rows
# ══════════════════════════════════════════════════════════════════════════

def build_arm_rows(pre_bins: int, pre_window_s: float) -> dict:
    human_corpus = load_human_corpus("open_loop_ds005489", OPENLOOP_DATA, derive_stim_from_stim_on=False, smoke=None)

    full_rows, uncontam_rows, contam_rows = [], [], []
    session_diagnostics = {}
    extra_excluded: dict[str, list[str]] = {}
    distances = []
    totals = dict(n_trials=0, n_matched=0, n_unmatched=0, n_uncontaminated=0, n_contaminated=0)

    for rec in human_corpus["records"]:
        arrays = rec["arrays"]
        ch_names = arrays["ch_names"].tolist()
        mask = channel_condition_masks(
            ch_names, str(arrays["anode"]), str(arrays["cathode"]), str(arrays["stim_channel"])
        )["excluding_stimulated_shank"]
        if mask.sum() < 1:
            extra_excluded.setdefault("no_channels_survive_shank_exclusion", []).append(rec["session_key"])
            continue

        contam = _session_word_contamination(rec["session_key"], pre_window_s)
        if contam["status"] != "computed":
            extra_excluded.setdefault(contam["status"], []).append(rec["session_key"])
            continue

        list_arr, serial_arr = np.asarray(arrays["list_number"]), np.asarray(arrays["serialpos"])
        n_trials = int(len(list_arr))
        uncontaminated_mask = np.zeros(n_trials, dtype=bool)
        matched_mask = np.zeros(n_trials, dtype=bool)
        for i in range(n_trials):
            info = contam["by_list_serialpos"].get((int(list_arr[i]), int(serial_arr[i])))
            if info is None:
                continue
            matched_mask[i] = True
            uncontaminated_mask[i] = not info["pre_window_contaminated"]
            if info["distance_to_preceding_train_offset_s"] is not None:
                distances.append(info["distance_to_preceding_train_offset_s"])

        n_unc = int((uncontaminated_mask & matched_mask).sum())
        n_con = int(matched_mask.sum()) - n_unc
        session_diagnostics[rec["session_key"]] = {
            "n_trials": n_trials, "n_matched": int(matched_mask.sum()),
            "n_unmatched": int((~matched_mask).sum()), "n_uncontaminated": n_unc, "n_contaminated": n_con,
        }
        totals["n_trials"] += n_trials
        totals["n_matched"] += int(matched_mask.sum())
        totals["n_unmatched"] += int((~matched_mask).sum())
        totals["n_uncontaminated"] += n_unc
        totals["n_contaminated"] += n_con

        full_windows = _human_session_windows(rec, pre_bins, mask)
        full_rows.append({"session_key": rec["session_key"], "cluster_id": rec["subject_id"], "windows": full_windows})

        for target_mask, target_rows in (
            (uncontaminated_mask & matched_mask, uncontam_rows),
            ((~uncontaminated_mask) & matched_mask, contam_rows),
        ):
            if int(target_mask.sum()) < 1:
                continue
            sub_arrays = {"epochs_log": arrays["epochs_log"][target_mask], "stim_flag": arrays["stim_flag"][target_mask]}
            sub_windows = _human_session_windows({"arrays": sub_arrays}, pre_bins, mask)
            target_rows.append({"session_key": rec["session_key"], "cluster_id": rec["subject_id"], "windows": sub_windows})

    distances_arr = np.asarray(distances, dtype=float)
    distance_summary = {
        "n_words_with_a_preceding_train": int(len(distances_arr)),
        "median_s": float(np.median(distances_arr)) if len(distances_arr) else None,
        "mean_s": float(np.mean(distances_arr)) if len(distances_arr) else None,
        "sd_s": float(np.std(distances_arr, ddof=1)) if len(distances_arr) > 1 else None,
        "min_s": float(np.min(distances_arr)) if len(distances_arr) else None,
        "max_s": float(np.max(distances_arr)) if len(distances_arr) else None,
        "fraction_negative_ie_a_preceding_train_still_active_at_word_onset": (
            float(np.mean(distances_arr < 0)) if len(distances_arr) else None
        ),
    }

    return {
        "human_corpus_admission": {
            "n_sessions_seen": human_corpus["n_sessions_total"], "n_sessions_used": len(full_rows),
            "excluded_by_reason": {**human_corpus["exclusions"], **extra_excluded},
        },
        "full_rows": full_rows, "uncontaminated_rows": uncontam_rows, "contaminated_rows": contam_rows,
        "session_diagnostics": session_diagnostics,
        "trial_totals": totals,
        "distance_to_preceding_train_offset_s_summary": distance_summary,
    }


# ══════════════════════════════════════════════════════════════════════════
# Pooling, effect-vs-mdd flagging, and the pre-declared verdict
# ══════════════════════════════════════════════════════════════════════════

def _annotate_effect_vs_mdd(pool_arm_result: dict) -> None:
    """Flags, in place, any pooled cell whose observed effect magnitude
    sits below its own minimum detectable difference -- such an estimate is
    upward-biased conditional on having reached significance at all."""
    for window in pool_arm_result.get("windows", {}).values():
        if window.get("status") != "computed":
            continue
        for biomarker_key in ("rate_free_deviation_biomarker", "nuisance_total_activity_biomarker"):
            pooled = window.get(biomarker_key, {}).get("pooled", {})
            mdd_obj = pooled.get("mdd", {})
            if pooled.get("status") == "computed" and mdd_obj.get("status") == "computed":
                pooled["effect_below_own_mdd"] = bool(abs(pooled["mean_value"]) < mdd_obj["mdd"])


def apply_decision_rule(uncontam_result: dict, reference_effect: float | None) -> dict:
    windows = uncontam_result.get("windows", {})
    pre = windows.get("pre_stimulation_window", {})
    post = windows.get("post_stimulation_window", {})
    if pre.get("status") != "computed" or post.get("status") != "computed":
        return {"verdict": "inconclusive_not_computable", "reason": "one or both uncontaminated-subset windows are a structural void or not computed"}
    pre_pooled = pre["rate_free_deviation_biomarker"]["pooled"]
    post_pooled = post["rate_free_deviation_biomarker"]["pooled"]
    if pre_pooled.get("status") != "computed":
        return {"verdict": "inconclusive_not_computable", "reason": "pre-window pooled deviation not computed in the uncontaminated subset (too few clusters or too few sessions)"}

    pre_p, pre_mean = pre_pooled["p_value"], pre_pooled["mean_value"]
    mdd_obj = pre_pooled.get("mdd", {})
    pre_mdd = mdd_obj.get("mdd") if mdd_obj.get("status") == "computed" else None
    post_p = post_pooled.get("p_value") if post_pooled.get("status") == "computed" else None
    post_mean = post_pooled.get("mean_value") if post_pooled.get("status") == "computed" else None

    if pre_p <= 0.05:
        verdict = "contamination_account_refuted"
    elif pre_mdd is None:
        verdict = "inconclusive_mdd_not_computable"
    elif reference_effect is not None and pre_mdd >= reference_effect:
        verdict = "inconclusive_underpowered"
    elif post_p is not None and post_p <= 0.05 and post_mean is not None and post_mean > 0:
        verdict = "contamination_account_supported"
    else:
        verdict = "contamination_account_partially_supported_post_window_not_significant_in_reduced_subset"

    return {
        "verdict": verdict,
        "pre_window_p_value": pre_p, "pre_window_mean": pre_mean,
        "pre_window_mdd": pre_mdd,
        "pre_window_mdd_clears_internal_reference_1p0": (pre_mdd < INTERNAL_REFERENCE_SD) if pre_mdd is not None else None,
        "pre_window_mdd_clears_reference_effect_to_explain": (pre_mdd < reference_effect) if (pre_mdd is not None and reference_effect is not None) else None,
        "reference_effect_to_explain": reference_effect,
        "post_window_p_value": post_p, "post_window_mean": post_mean,
    }


def main() -> None:
    t0 = time.time()
    delivered_reference = _load_delivered_reference()
    reference_effect = delivered_reference.get("pre_window_mean") if delivered_reference.get("status") == "computed" else None

    output = {
        "version": "2026-09-11", "seed": SEED, "status": "running",
        "decision_rule_declared_before_fitting": _build_decision_rule(reference_effect),
        "delivered_panel_reference_values": delivered_reference,
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
        "trial_reconciliation": "n_trials == n_matched + n_unmatched; n_matched == n_uncontaminated + n_contaminated",
        "distance_to_preceding_train_offset_s_summary": built["distance_to_preceding_train_offset_s_summary"],
        "attribution_limit": census_row["item_overlap"]["value"]["reading"],
        "n_perm": 10000, "n_boot": 5000, "alpha": 0.05, "power": 0.80,
    }
    output["session_diagnostics"] = built["session_diagnostics"]
    output["status"] = "arms_pending"
    _flush(output)

    arms = {}
    for arm_name, rows in (
        ("full_reproduction", built["full_rows"]),
        ("uncontaminated_subset", built["uncontaminated_rows"]),
        ("contaminated_subset", built["contaminated_rows"]),
    ):
        result = _pool_arm(f"{CORPUS_ID}__{arm_name}", rows, COUNTERFACTUAL_LABEL, windows_measured)
        _annotate_effect_vs_mdd(result)
        result["specificity_check"] = _specificity_check(result)
        result["n_sessions_contributing_rows"] = len(rows)
        arms[arm_name] = result
        output["arms"] = arms
        _flush(output)

    output["decision"] = apply_decision_rule(arms["uncontaminated_subset"], reference_effect)
    output["wall_clock_s"] = time.time() - t0
    output["code_commit"] = git_commit(ROOT)
    output["status"] = "complete"
    _flush(output)
    print(json.dumps({"verdict": output["decision"]["verdict"], "status": output["status"]}, indent=2))


if __name__ == "__main__":
    main()
