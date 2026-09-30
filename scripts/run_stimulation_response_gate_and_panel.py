#!/usr/bin/env python3
"""Two deliverables in one script, run in order.

The first deliverable is a reproduction gate. results/human_stimulation_component_response.json's
block_b reports a stimulation-induced displacement of the rate-normalized
angular-deviation biomarker (rate_free_state_deviation), pooled across
sessions, for three channel conditions (full channel set; excluding the
stimulated bipolar pair; excluding the stimulated shank). This module
reruns that same computation from the producer script's own functions
(scripts/run_human_stimulation_component_response.py's load_corpus and
run_stimulation_displacement, imported unchanged) and compares the result
against the delivered artifact under a tolerance declared before the run.
This project has previously found delivered stimulation artifacts that no
longer reproduce because shared analysis code drifted underneath them, so
this gate is expected to be capable of failing; a failure is reported and
the dependent human-arm panel work below is skipped, not repaired.

The second deliverable is a common time-resolved stimulation-response interface
(src/stimulation_response_estimator.py) applied to the two stimulation arms
results/stimulation_design_census.json certifies have all three of a
randomized/experimenter-scheduled stimulation action, a measured
pre-stimulation window, and a measured post-stimulation window:
ram_ds005489_openloop (human intracranial, episodic encoding) and
macaque_pfc_microstimulation (macaque dlPFC, working-memory delay). Two
biomarkers only: the rate-free angular deviation (candidate) and total
band power / total spike count (explicit nuisance control, never scored as
a candidate). Windows, per-arm, come only from the census row; a window the
census marks structural_void is never computed and is carried into the
artifact as a void with the census's own reason. Every pooled cell reports
which no-stimulation counterfactual was used, and uncertainty is a
whole-participant (human) or whole-animal (macaque) cluster bootstrap,
never a trial-count formula.

Outputs:
  results/stimulation_response_gate_and_panel.json

Run:
    python \
        scripts/run_stimulation_response_gate_and_panel.py
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
from statistics import stable_seed  # noqa: E402
from stimulation_response_estimator import (  # noqa: E402
    cluster_bootstrap_pooled_effect,
    nuisance_treatment_effect,
    rate_free_state_deviation,
    window_treatment_effect,
)
from stimulation_events import _human_session_windows, _specificity_check  # noqa: E402
from stimulation_response_estimator import _pool_arm  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "stimulation_response_gate_and_panel.json"
CENSUS_PATH = RESULTS / "stimulation_design_census.json"
DELIVERED_DISPLACEMENT_PATH = RESULTS / "human_stimulation_component_response.json"

SEED = 20260911  # arbitrary fixed seed for this module's own bootstrap RNG streams (stable_seed keys carry the rest)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(output))


# ══════════════════════════════════════════════════════════════════════════
# Reproduction gate on results/human_stimulation_component_response.json's block_b
# ══════════════════════════════════════════════════════════════════════════

# Tolerance declared BEFORE the reproduction run, per the three different quantities being compared:
#   - mean_value and the two CI bounds: the producer's per-session displacement values are read back
#     from an existing, non-deleted checkpoint cache (never recomputed from raw iEEG here -- see
#     "checkpoint_scope_disclosure" below) and the pooling (subject_clustered_mean_test) is
#     deterministic given those cached per-session numbers and a seed keyed only on the sorted subject
#     tuple and sample size (stable_seed). Both runs therefore execute the identical deterministic
#     arithmetic on the identical inputs, so exact reproduction (float equality up to accumulation
#     order) is the correct standard, not a loose one: relative tolerance 1e-9.
#   - p_value: paired_sign_flip_test's p is a Monte Carlo permutation p-value at n_perm=10000 sign
#     flips, i.e. a value on the lattice k/(n_perm+1) for integer k -- it is not continuous and cannot
#     be held to a float tolerance the way a mean can. Because the draws are RNG-seeded identically
#     (same stable_seed key), the reproduction should land on the exact same lattice point; the
#     declared tolerance is therefore also exact equality, with the SEPARATE, explicit acknowledgement
#     that if it does not land exactly, the correct diagnostic is "which lattice point", not "how far
#     off in absolute p", because the two are not comparable as real numbers.
#   - n_sessions / n_subjects (admission counts): integers, zero tolerance. A differing admission set
#     is a gate failure on its own even when every mean is inside tolerance, because it means the two
#     runs analysed different data, not the same data computed twice.
GATE_TOLERANCE = {
    "mean_value_relative_tolerance": 1e-9,
    "ci_bound_relative_tolerance": 1e-9,
    "p_value_rule": "exact lattice match required (k/(n_perm+1)); any mismatch is reported as a lattice-point diagnostic, never absorbed into a numeric tolerance",
    "admission_count_rule": "n_sessions and n_subjects must match exactly; any difference fails the gate regardless of how close the pooled mean is",
}


def _relclose(observed: float, delivered: float, rtol: float) -> bool:
    if delivered == 0.0:
        return abs(observed) <= rtol
    return abs(observed - delivered) <= rtol * abs(delivered)


def run_reproduction_gate() -> dict:
    from run_human_stimulation_component_response import load_corpus, run_stimulation_displacement, CHECKPOINT_DIR, load_checkpoint
    from corpus_sessions import _stimulation_displacement_session
    from stimulation_events import N_PERM
    from project_config import dataset_path
    from corpus_sessions import DATA as OPENLOOP_DATA
    CLOSEDLOOP_DATA = dataset_path("ram_ds005557_closedloop")

    if not DELIVERED_DISPLACEMENT_PATH.exists():
        return {"status": "not_computable", "reason": "results/human_stimulation_component_response.json is absent"}
    delivered = json.loads(DELIVERED_DISPLACEMENT_PATH.read_text())
    delivered_conditions = delivered.get("block_b", {}).get("pooled_normalised_displacement_by_channel_condition", {})
    delivered_zero_drop = delivered.get("zero_drop_accounting", {})

    t0 = time.time()
    openloop = load_corpus("open_loop_ds005489", OPENLOOP_DATA, derive_stim_from_stim_on=False, smoke=None)
    closedloop = load_corpus("closed_loop_ds005557", CLOSEDLOOP_DATA, derive_stim_from_stim_on=True, smoke=None)
    all_records = openloop["records"] + closedloop["records"]
    reproduced_zero_drop = {
        "open_loop_ds005489": {"n_sessions_total": openloop["n_sessions_total"], "n_sessions_used": openloop["n_sessions_used"]},
        "closed_loop_ds005557": {"n_sessions_total": closedloop["n_sessions_total"], "n_sessions_used": closedloop["n_sessions_used"]},
    }
    admission_set_matches = all(
        reproduced_zero_drop[k]["n_sessions_total"] == delivered_zero_drop.get(k, {}).get("n_sessions_total")
        and reproduced_zero_drop[k]["n_sessions_used"] == delivered_zero_drop.get(k, {}).get("n_sessions_used")
        for k in ("open_loop_ds005489", "closed_loop_ds005557")
    )

    reproduced_block_b = run_stimulation_displacement(all_records)
    reproduced_conditions = reproduced_block_b["pooled_normalised_displacement_by_channel_condition"]

    per_condition = {}
    all_pass = True
    for cond in ("full_channel_set", "excluding_stimulated_pair", "excluding_stimulated_shank"):
        d = delivered_conditions.get(cond, {})
        r = reproduced_conditions.get(cond, {})
        if d.get("status") != "computed" or r.get("status") != "computed":
            per_condition[cond] = {"status": "not_computable", "delivered": d, "reproduced": r}
            all_pass = False
            continue
        counts_match = (d.get("n_sessions") == r.get("n_sessions")) and (d.get("n_subjects") == r.get("n_subjects"))
        mean_ok = _relclose(r["mean_value"], d["mean_value"], GATE_TOLERANCE["mean_value_relative_tolerance"])
        ci_lo_ok = _relclose(r["ci_lower"], d["ci_lower"], GATE_TOLERANCE["ci_bound_relative_tolerance"])
        ci_hi_ok = _relclose(r["ci_upper"], d["ci_upper"], GATE_TOLERANCE["ci_bound_relative_tolerance"])
        lattice_unit = 1.0 / (N_PERM + 1)
        p_ok = abs(r["p_value"] - d["p_value"]) <= 1e-12  # exact lattice match, per the declared rule
        p_lattice_points_off = round(abs(r["p_value"] - d["p_value"]) / lattice_unit)
        passed = counts_match and mean_ok and ci_lo_ok and ci_hi_ok and p_ok
        all_pass = all_pass and passed
        per_condition[cond] = {
            "status": "computed", "passed": passed,
            "delivered": {"mean_value": d["mean_value"], "ci_lower": d["ci_lower"], "ci_upper": d["ci_upper"],
                          "p_value": d["p_value"], "n_sessions": d["n_sessions"], "n_subjects": d["n_subjects"]},
            "reproduced": {"mean_value": r["mean_value"], "ci_lower": r["ci_lower"], "ci_upper": r["ci_upper"],
                           "p_value": r["p_value"], "n_sessions": r["n_sessions"], "n_subjects": r["n_subjects"]},
            "checks": {"admission_counts_match": counts_match, "mean_within_tolerance": mean_ok,
                       "ci_lower_within_tolerance": ci_lo_ok, "ci_upper_within_tolerance": ci_hi_ok,
                       "p_value_exact_lattice_match": p_ok, "p_value_lattice_points_off": p_lattice_points_off},
        }

    gate_status = "reproduced" if (all_pass and admission_set_matches) else "failed_to_reproduce"

    # Supplementary spot check: run_stimulation_displacement's per-session numeric path (band-pass,
    # epoching, rate_free_state_deviation, the fixed-reference scoring) is normally read back from a
    # non-deleted checkpoint cache keyed only by session identity, not by code version -- a real
    # per-session code drift could be silently masked by a stale cache hit and never show up in the
    # pooled comparison above. Checkpoints under results/.checkpoints/ are kept as immutable provenance
    # and are never deleted, which forecloses forcing a full fresh recomputation here, so this spot check
    # instead recomputes a handful of sessions'
    # per-session displacement directly (bypassing the checkpoint) and diffs them against what the
    # cache is currently returning, to catch drift in that per-session path specifically.
    spot_check_sessions = all_records[:5] + all_records[-5:]
    spot_check = []
    for rec in spot_check_sessions:
        fresh = _stimulation_displacement_session(rec)
        cached = load_checkpoint(f"blockB__{rec['corpus']}__{rec['session_key']}")
        if cached is None:
            spot_check.append({"session": rec["session_key"], "status": "no_cached_checkpoint_to_compare"})
            continue
        agree = True
        for cond in ("full_channel_set", "excluding_stimulated_pair", "excluding_stimulated_shank"):
            fc, cc = fresh.get("conditions", {}).get(cond, {}), cached.get("conditions", {}).get(cond, {})
            if fc.get("status") != cc.get("status"):
                agree = False
                continue
            if fc.get("status") == "computed":
                agree = agree and _relclose(fc["displacement"], cc["displacement"], 1e-9)
        spot_check.append({"session": rec["session_key"], "status": "agrees" if agree else "DRIFT_DETECTED"})
    spot_check_drift = any(s["status"] == "DRIFT_DETECTED" for s in spot_check)

    return {
        "status": "computed",
        "decision_rule_declared_before_fitting": GATE_TOLERANCE,
        "gate_status": gate_status,
        "admission_set_matches_delivered": admission_set_matches,
        "reproduced_zero_drop_accounting": reproduced_zero_drop,
        "delivered_zero_drop_accounting": delivered_zero_drop,
        "per_condition": per_condition,
        "checkpoint_scope_disclosure": (
            "Every one of the 103 sessions block_b pools already has a results/.checkpoints/"
            "run_human_stimulation_component_response/blockB__*.json checkpoint from the original "
            "delivery run. Checkpoints under results/.checkpoints/ are kept as immutable provenance of that "
            "original run and are never deleted, so this reproduction cannot force the per-session feature-extraction code (band-pass "
            "filtering, epoching, the fixed-reference scoring) to re-execute from raw iEEG for every "
            "session -- run_checkpointed reuses the cached per-session record whenever the checkpoint "
            "key (session identity only, not a code hash) is present. What this reproduction genuinely "
            "re-executes fresh, for all 103 sessions, is the POOLING arithmetic (subject_clustered_mean_test: "
            "collapsing to subject-level means, the seeded sign-flip permutation test, and the seeded "
            "bootstrap CI) plus corpus loading and session admission. The 10-session spot check below "
            "additionally re-executes the per-session numeric path itself, bypassing the checkpoint, "
            "for 5 first and 5 last records, to give the drift concern raised above some "
            "direct coverage beyond the pooling layer."
        ),
        "per_session_spot_check": {"sessions": spot_check, "drift_detected": spot_check_drift},
        "seed_and_determinism": (
            "No new RNG seed is introduced by this gate: every stochastic step (the sign-flip null, the "
            "bootstrap CI) is seeded inside run_stimulation_displacement/subject_clustered_mean_test via "
            "stable_seed(f'subject_clustered|{tuple(unique_subjects)}|{n}'), a deterministic CRC32 of a "
            "string built only from the subject set and sample size -- identical inputs necessarily "
            "produce an identical seed and therefore an identical draw sequence."
        ),
        "wall_clock_s": time.time() - t0,
    }



# ══════════════════════════════════════════════════════════════════════════
# The shared time-resolved interface, applied to the two admissible arms
# ══════════════════════════════════════════════════════════════════════════

def _read_census_row(corpus_id: str) -> dict:
    census = json.loads(CENSUS_PATH.read_text())
    for row in census["rows"]:
        if row["corpus_id"] == corpus_id:
            return row
    raise KeyError(f"{corpus_id} not found in {CENSUS_PATH}")




def _macaque_session_windows(prefix: str, pre_bins: int) -> tuple[dict | None, str | None]:
    """Trials are read from the correct-trial file only, matching the
    condition-token indexing (control_idx, stim_channels) that file's own
    control_idx is computed from -- the error-trial file enumerates its own
    conditions separately and is not merged in here, to avoid reconciling
    two files' potentially different condition orderings."""
    from run_macaque_pfc_microstimulation_pipeline import load_macaque_pfc_microstimulation_session
    from spike_pipeline import crop_trial
    from spike_pipeline import BIN_S as MACAQUE_BIN_S
    corr = load_macaque_pfc_microstimulation_session(prefix, correct=True)
    if corr is None or corr["control_idx"] is None:
        return None, "no_usable_correct_trial_file_or_no_control_condition"
    control_idx = corr["control_idx"]
    n_trials_seen = len(corr["trials"])
    pre_rows, post_rows, treated_flags = [], [], []
    for tr in corr["trials"]:
        cropped = crop_trial(tr["spikerate"])
        if cropped is None:
            continue
        pre_rows.append(cropped[:pre_bins, :].sum(axis=0) * MACAQUE_BIN_S)   # estimated spike count per channel
        post_rows.append(cropped[pre_bins:, :].sum(axis=0) * MACAQUE_BIN_S)
        treated_flags.append(tr["stim_cond"] != control_idx)
    if not pre_rows:
        return None, "no_trial_survives_the_common_crop_window"
    pre_mat, post_mat = np.array(pre_rows), np.array(post_rows)
    treated = np.array(treated_flags, dtype=bool)
    windows = {
        "pre_stimulation_window": {
            "control_activity": pre_mat[~treated], "treated_activity": pre_mat[treated],
            "control_total": pre_mat[~treated].sum(axis=1), "treated_total": pre_mat[treated].sum(axis=1),
        },
        "post_stimulation_window": {
            "control_activity": post_mat[~treated], "treated_activity": post_mat[treated],
            "control_total": post_mat[~treated].sum(axis=1), "treated_total": post_mat[treated].sum(axis=1),
        },
    }
    return {"windows": windows, "n_trials_seen": n_trials_seen, "n_trials_used": len(pre_rows)}, None






def run_stimulation_response_panel() -> dict:
    from stimulation_events import channel_condition_masks
    from corpus_sessions import DATA as OPENLOOP_DATA
    from run_human_stimulation_component_response import load_corpus as load_human_corpus
    from run_ram_openloop_pipeline import BIN_S as HUMAN_BIN_S, PRE_S as HUMAN_PRE_S
    from spike_pipeline import PRE_S as MACAQUE_PRE_S
    from run_macaque_pfc_microstimulation_pipeline import SESSIONS as MACAQUE_SESSIONS
    from spike_pipeline import BIN_S as MACAQUE_BIN_S

    panel = {}

    # ---- human arm: ram_ds005489_openloop (open-loop corpus only -- the classifier-triggered
    # closed-loop corpus is foreclosed by the census's own randomization-classification rule and is
    # never used to build this panel, even though it is pooled into block_b's own headline numbers). ----
    row = _read_census_row("ram_ds005489_openloop")
    windows_measured = {"pre_stimulation_window": row["pre_stimulation_window_s"],
                        "post_stimulation_window": row["post_stimulation_windows_through_probe_s"]}
    human_corpus = load_human_corpus("open_loop_ds005489", OPENLOOP_DATA, derive_stim_from_stim_on=False, smoke=None)
    pre_bins = int(round(HUMAN_PRE_S / HUMAN_BIN_S))
    per_session_rows, extra_excluded = [], {}
    for rec in human_corpus["records"]:
        arrays = rec["arrays"]
        mask = channel_condition_masks(arrays["ch_names"].tolist(), str(arrays["anode"]), str(arrays["cathode"]),
                                       str(arrays["stim_channel"]))["excluding_stimulated_shank"]
        if mask.sum() < 1:
            extra_excluded.setdefault("no_channels_survive_shank_exclusion", []).append(rec["session_key"])
            continue
        per_session_rows.append({"session_key": rec["session_key"], "cluster_id": rec["subject_id"],
                                 "windows": _human_session_windows(rec, pre_bins, mask)})

    human_result = _pool_arm(
        "ram_ds005489_openloop", per_session_rows,
        counterfactual_label=(
            "randomized_contemporaneous_controls: unstimulated WORD presentations within the same "
            "session; ds005489's own assignment is an experimenter-scheduled alternating block design "
            "(census treatment_assignment_classification = experimenter_randomized)."
        ),
        windows_measured=windows_measured,
    )
    human_result["channel_condition"] = "excluding_stimulated_shank"
    human_result["session_admission"] = {
        "n_sessions_seen": human_corpus["n_sessions_total"], "n_sessions_used": len(per_session_rows),
        "excluded_by_reason": {**human_corpus["exclusions"], **extra_excluded},
    }
    human_result["attribution_limit"] = row["item_overlap"]["value"]["reading"]
    human_result["specificity_check"] = _specificity_check(human_result)
    panel["ram_ds005489_openloop"] = human_result

    # ---- macaque arm: macaque_pfc_microstimulation ----
    row_m = _read_census_row("macaque_pfc_microstimulation")
    windows_measured_m = {"pre_stimulation_window": row_m["pre_stimulation_window_s"],
                          "post_stimulation_window": row_m["post_stimulation_windows_through_probe_s"]}
    pre_bins_m = int(round(MACAQUE_PRE_S / MACAQUE_BIN_S))
    per_session_rows_m, excluded_m = [], {}
    for prefix in MACAQUE_SESSIONS:
        res, reason = _macaque_session_windows(prefix, pre_bins_m)
        if res is None:
            excluded_m.setdefault(reason, []).append(prefix)
            continue
        per_session_rows_m.append({"session_key": prefix, "cluster_id": prefix[:2], "windows": res["windows"],
                                   "n_trials_seen": res["n_trials_seen"], "n_trials_used": res["n_trials_used"]})

    macaque_result = _pool_arm(
        "macaque_pfc_microstimulation", per_session_rows_m,
        counterfactual_label=(
            "randomized_contemporaneous_controls: no-stimulation trials interleaved in the same session "
            "(census control_conditions); trial-to-condition sequence within a session is an "
            "experimenter-set pseudo-randomised order (census treatment_assignment_classification = "
            "experimenter_randomized), read from the correct-trial file only per session."
        ),
        windows_measured=windows_measured_m,
    )
    macaque_result["channel_condition"] = (
        "full_channel_set -- no per-trial stimulated-channel exclusion is applied. Unlike the human "
        "bipolar-pair corpus, stimulation site varies within every session (11/11 sessions, per the "
        "census's own stimulation_site field), so a single fixed channel mask does not generalise the "
        "same way; excluding each trial's own condition-specific stimulated channel(s) was out of scope "
        "for this analysis and is disclosed here rather than silently applied or silently skipped."
    )
    macaque_result["session_admission"] = {
        "n_sessions_seen": len(MACAQUE_SESSIONS), "n_sessions_used": len(per_session_rows_m),
        "excluded_by_reason": excluded_m,
    }
    macaque_result["inference_unit_note"] = (
        "11 sessions from 2 animals (Sa: 1 session, Wa: 10 sessions); cluster_bootstrap_pooled_effect "
        "clusters on animal id, collapsing each animal's own sessions to one mean before the sign-flip "
        "test and the bootstrap CI run over the 2 animal-level values. With only 2 clusters the sign-flip "
        "null has exactly 2^2=4 sign patterns, so the minimum attainable two-sided p-value is bounded "
        "well above the conventional 0.05 threshold by construction -- this is the honest consequence of "
        "the animal count, reported as such rather than adjusted or hidden."
    )
    macaque_result["specificity_check"] = _specificity_check(macaque_result)
    panel["macaque_pfc_microstimulation"] = macaque_result

    return panel


def main() -> None:
    t0 = time.time()
    output = {"version": "2026-09-11", "seed": SEED, "status": "running"}
    _flush(output)

    output["reproduction_gate"] = run_reproduction_gate()
    output["status"] = "gate_complete_panel_pending"
    _flush(output)

    gate_status = output["reproduction_gate"].get("gate_status")
    if gate_status != "reproduced":
        output["stimulation_response_panel"] = {
            "status": "not_run",
            "reason": (
                f"reproduction_gate.gate_status = {gate_status!r}. The human arm of this panel reuses "
                "the same corpus, the same rate_free_state_deviation estimator, and the same "
                "fixed-reference scoring the failed reproduction covers, so the dependent panel work is "
                "not built on top of a number that failed to reproduce; the gate failure is reported at "
                "the top of the report instead."
            ),
        }
    else:
        output["stimulation_response_panel"] = run_stimulation_response_panel()
    _flush(output)

    output["wall_clock_s"] = time.time() - t0
    output["code_commit"] = git_commit(ROOT)
    output["status"] = "complete"
    _flush(output)
    print(json.dumps({"gate_status": gate_status, "status": output["status"]}, indent=2))


if __name__ == "__main__":
    main()
