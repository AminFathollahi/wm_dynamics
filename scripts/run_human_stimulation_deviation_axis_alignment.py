#!/usr/bin/env python3
"""Does electrical stimulation delivered to human intracranial recordings displace the population
state along the same low-dimensional direction that carries the rate-free behavioural-deviation
signal, or off it?

Two prior artifacts already establish the two halves this question needs in human data:
results/recording_tier_component_transfer.json shows the deviation component survives at every
recording tier this project has built (sorted units, medial-temporal depth, cortical depth, scalp,
and beamformed cortical sources), and results/human_stimulation_component_response.json shows
stimulation measurably displaces the same component in two human intracranial free-recall corpora
(OpenNeuro ds005489, experimenter-scheduled; OpenNeuro ds005557, classifier-triggered). Neither
artifact asks whether that displacement lands along the behaviourally-linked axis or off it -- this
module is the one place both a control-trial-only axis and a stimulation displacement can be built in
the same sessions, because both corpora carry stimulation and non-stimulated trials side by side.

CONSTRUCTION. Every recording channel's per-trial log high-gamma power, averaged over the analysis
window, is L2-normalised to a unit direction -- the identical rate-free construction this project's
deviation estimator uses internally. The axis is the leading eigenvector of the residual-direction
covariance built from non-stimulated (control) trials ONLY; a stimulated trial is never permitted into
that fit (estimate_axis raises immediately on anything but source="control_only"). The displacement
direction is the difference of the mean unit-normalised direction over admitted stimulated trials and
the mean over the same control trials the axis was fit on. Both the axis-estimation and the
displacement-vector primitives are reused unchanged from the sibling analysis that asked this same
question in a macaque intracortical microstimulation corpus
(scripts/run_macaque_pfc_microstimulation_stimulation_deviation_axis_alignment.py) -- the arithmetic is corpus-agnostic and
is not touched anywhere in this module.

RECORDING TIERS. Both corpora record only bipolar intracranial macro-contacts (depth SEEG and
subdural ECOG grid/strip electrodes); neither ships sorted single units, a scalp montage, or a
beamformed source reconstruction, so those three tiers are recorded here as absent with their reason,
never as a null. The two tiers this corpus DOES support -- medial-temporal depth and cortical depth --
are built by classifying each admitted channel's own anatomical location (the ind.region
Desikan-Killiany label this project's dataset registry already declares as the label_convention for
both corpora) into a medial-temporal-lobe structure (hippocampus, amygdala, entorhinal or
parahippocampal cortex) or a non-medial-temporal cortical structure. A depth SEEG contact and a
subdural ECOG contact are pooled into the same "cortical depth" tier when both are classified
non-medial-temporal, since this project's established five-tier vocabulary has no separate ECOG tier
and an ECOG contact is, in this release, never sited in medial-temporal tissue; this pooling is
disclosed here rather than silently assumed. Because ind.region is a cortical-surface atlas, it does
not directly label hippocampus or amygdala proper -- the medial-temporal tier built from it is
therefore dominated by parahippocampal and entorhinal contacts, a scope limit of the corpus's own
declared label_convention, not a choice made for this analysis.

CHANNEL-ARTIFACT CONTROL. Stimulation produces a large deflection on channels at and near the driven
electrode pair. Every fit here uses only the channel set that already excludes the driven bipolar pair
and every other channel sharing a physical lead with it (channel_condition_masks'
excluding_stimulated_shank condition, reused unchanged from the delivered displacement module) --
the same artifact-cleaned channel set that module's own mediation and site-targeting analyses treat as
their primary channel condition.

TWO ARMS, MANDATORY AGREEMENT. Exactly as in the macaque sibling, every decomposition here is run
twice: once against the raw control-trial axis, and once against the axis re-estimated after each
control trial's own linear time-trend is regressed out first. Only agreement between the two arms
licenses a directional sentence about the component; disagreement is reported as its own finding
(stimulation moved the drift, not the component), never silently resolved toward whichever arm looks
more interesting.

BIAS-ONLY CONTROL. A "bias-only axis" -- the plain normalised mean unit-direction over the same
control trials, the one direction the residual axis is by construction orthogonal to -- is scored
against the identical alignment test. If it reproduces the real alignment's significance and direction
relative to its own null, the branch displacement_direction_not_separable_from_a_unit_level_offset
fires for that arm, voided on sign and significance only, never on a magnitude comparison.

SIGNED-VERSUS-MAGNITUDE DISCIPLINE. The axis is a leading eigenvector with no session-comparable sign
convention established anywhere in this project. Every alignment statistic here is unsigned (absolute
cosine and its square); no signed-displacement language is used anywhere in this artifact.

Every trial's analysis window is -0.3 to +1.6 s relative to word onset (word on-screen for 1.6 s), the
same window build_session_features uses; the stimulation pulse train in both corpora outlasts a single
word presentation, so the analysed epoch overlaps current delivery by design for a stimulated trial,
and neither corpus carries an isolated working-memory maintenance delay separate from encoding. This is
disclosed everywhere it matters and is never treated as a matching requirement or a reason to drop a
trial.

Outputs:
  results/human_stimulation_deviation_axis_alignment.json

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \\
    scripts/run_human_stimulation_deviation_axis_alignment.py
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from provenance import _json_safe, checkpoint_safe, git_commit, restore_checkpoint  # noqa: E402
from spike_pipeline import normalize_region_label  # noqa: E402
from run_human_stimulation_component_response import (  # noqa: E402
    CLOSEDLOOP_DATA, OPENLOOP_DATA, _bin_averaged, channel_condition_masks, compute_block_b_displacement,
    load_corpus,
)
from run_stimulation_site_targeting_map import load_electrode_table  # noqa: E402
from run_macaque_pfc_microstimulation_stimulation_deviation_axis_alignment import (  # noqa: E402
    BEHAVIOURAL_REFERENCE_R_UNITS, CONTINUITY_ALIGNMENT_FLOOR_ABS_COSINE, N_ROTATION_DRAWS,
    _alignment_summary, _bias_only_axis, _bias_only_voids, _classify_arm, _pool_group, displacement_vector,
    estimate_axis,
)
from run_deviation_axis_identity_controls import _vector_reference_alignment_with_draws  # noqa: E402
from run_dissociation_cross_preparation_test import MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402
from run_recording_tier_component_transfer import MTL_STRUCTURES, UNLABELLED_STRUCTURES  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "human_stimulation_deviation_axis_alignment.json"
COMPONENT_RESPONSE_PATH = RESULTS / "human_stimulation_component_response.json"
RECORDING_TIER_TRANSFER_PATH = RESULTS / "recording_tier_component_transfer.json"

CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_human_stimulation_deviation_axis_alignment"
SCHEMA_TAG = "v1_control_only_axis_mtl_cortical_tier_split_2026_08_27"

CORPUS_DATA_DIR = {"open_loop_ds005489": OPENLOOP_DATA, "closed_loop_ds005557": CLOSEDLOOP_DATA}
LABEL_CONVENTION = "bids_desikan_killiany_ind_region"
DEPTH_TIERS = ("depth_mtl", "depth_cortical")
ARMS = ("raw_axis", "detrended_axis")
MIN_CHANNELS_PER_TIER = 3

ABSENT_TIERS = {
    "single_unit": "neither ds005489 nor ds005557 records sorted single units; every channel in both "
                   "corpora is a bipolar intracranial macro-contact (SEEG depth or ECOG grid/strip), "
                   "verified against every session's own channels.tsv 'type' column (values SEEG/ECOG "
                   "only, no EEG or unit-derived channel)",
    "scalp_eeg": "neither corpus carries a scalp montage; every channel's type is SEEG or ECOG",
    "beamformed_cortical": "neither corpus carries a source-reconstructed/beamformed release; both are "
                            "raw bipolar intracranial recordings only",
}


# =======================================================================================================
# Checkpointing: one file per session, temp file + os.replace, completion flag written only after the
# fit returns; every checkpoint key is tagged with a schema string so a checkpoint written by an earlier
# version of this loader is a MISS rather than a silent, wrongly-shaped hit.
# =======================================================================================================

def _checkpoint_path(unit: str) -> Path:
    import re
    return CHECKPOINT_DIR / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', unit)}.json"


def _load_checkpoint(unit: str) -> dict | None:
    path = _checkpoint_path(unit)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True or data.get("schema_tag") != SCHEMA_TAG:
        return None
    return restore_checkpoint(data["record"])


def _save_checkpoint(unit: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(unit)
    payload = {"_complete": True, "schema_tag": SCHEMA_TAG, "record": checkpoint_safe(record)}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload, allow_nan=False))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _run_checkpointed(unit: str, fit_fn):
    cached = _load_checkpoint(unit)
    if cached is not None:
        return cached
    record = fit_fn()
    _save_checkpoint(unit, record)
    return record


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# =======================================================================================================
# Channel-tier classification.
# =======================================================================================================

def classify_channel_tier(channel_name: str, labels: dict) -> str | None:
    """Classifies one bipolar recording channel ("A-B") as depth_mtl, depth_cortical, or unclassified,
    from each contact's own ind.region Desikan-Killiany label -- the label_convention this project's
    dataset registry already declares for both corpora this module reads. Pre-declared rule: the
    first-listed contact's label is used; the second is tried only when the first does not resolve to a
    known structure, since a bipolar pair's two contacts sit on the same lead and normally share one."""
    contacts = channel_name.split("-")
    if len(contacts) != 2:
        return None
    for contact in contacts:
        raw = labels.get(contact, {}).get("ind.region", "n/a")
        structure, _ = normalize_region_label(raw, LABEL_CONVENTION)
        if structure not in UNLABELLED_STRUCTURES:
            return "depth_mtl" if structure in MTL_STRUCTURES else "depth_cortical"
    return None


# =======================================================================================================
# Per-session, per-tier fit -- axis estimation and displacement decomposition, raw and detrended arms.
# =======================================================================================================

def _fit_session_tier(arrays: dict, tier_mask: np.ndarray, seed_base: str) -> dict:
    n_tier_channels = int(tier_mask.sum())
    if n_tier_channels < MIN_CHANNELS_PER_TIER:
        return {"status": "excluded", "reason": "fewer_than_3_admitted_channels_in_tier",
                "n_channels": n_tier_channels}

    activity = _bin_averaged(arrays, tier_mask)
    stim_flag = arrays["stim_flag"].astype(bool)
    control_activity, stim_activity = activity[~stim_flag], activity[stim_flag]
    n_ctrl, n_stim = int(control_activity.shape[0]), int(stim_activity.shape[0])
    if n_ctrl < MIN_TRIALS_WITH_DEFINED_DIRECTION or n_stim < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "excluded", "reason": "fewer_than_16_control_or_stimulated_trials",
                "n_control_trials": n_ctrl, "n_stim_trials": n_stim, "n_channels": n_tier_channels}

    disp = displacement_vector(control_activity, stim_activity)
    if disp is None or disp["direction_unit"] is None:
        return {"status": "excluded", "reason": "displacement_direction_undefined_or_too_few_trials",
                "n_channels": n_tier_channels}

    trial_index = np.arange(n_ctrl, dtype=float)
    bias_axis = _bias_only_axis(control_activity)

    by_arm = {}
    for arm_name, detrend in zip(ARMS, (False, True)):
        axis_fit = estimate_axis(control_activity, trial_index, source="control_only", detrend=detrend,
                                  seed_tag=f"{seed_base}|{arm_name}|stability")
        if axis_fit["status"] != "computed":
            by_arm[arm_name] = {"status": axis_fit["status"], "detail": axis_fit}
            continue
        stability = axis_fit["axis_stability"]
        stable = bool(stability.get("stable")) if stability.get("status") == "computed" else None
        if stable is False:
            by_arm[arm_name] = {"status": "axis_not_stable", "axis_stability": stability}
            continue

        axis = axis_fit["axis"]
        real_alignment = _vector_reference_alignment_with_draws(
            disp["direction_unit"], axis, N_ROTATION_DRAWS, f"{seed_base}|{arm_name}|alignment")
        bias_alignment = (
            _vector_reference_alignment_with_draws(
                disp["direction_unit"], bias_axis, N_ROTATION_DRAWS, f"{seed_base}|{arm_name}|bias_alignment")
            if bias_axis is not None else None)

        by_arm[arm_name] = {
            "status": "computed",
            "n_trials_kept_for_axis": axis_fit["n_trials_kept"],
            "axis_stability": stability,
            "real_alignment": {**_alignment_summary(real_alignment["observed"], real_alignment["draws"]),
                                "draws": real_alignment["draws"]},
            "bias_only_alignment": (
                {**_alignment_summary(bias_alignment["observed"], bias_alignment["draws"]),
                 "draws": bias_alignment["draws"]} if bias_alignment is not None else None),
        }

    return {
        "status": "computed",
        "n_channels": n_tier_channels,
        "n_control_trials": n_ctrl, "n_stim_trials": n_stim,
        "n_control_trials_used_in_displacement": disp["n_control_trials_used"],
        "n_stim_trials_used_in_displacement": disp["n_stim_trials_used"],
        "displacement_norm_unit_direction_space": disp["norm"],
        "by_arm": by_arm,
    }


def _fit_session(rec: dict) -> dict:
    arrays = rec["arrays"]
    ch_names = [str(c) for c in arrays["ch_names"].tolist()]
    anode, cathode, stim_ch = str(arrays["anode"]), str(arrays["cathode"]), str(arrays["stim_channel"])
    masks = channel_condition_masks(ch_names, anode, cathode, stim_ch)
    trusted_mask = masks["excluding_stimulated_shank"]

    electrode = load_electrode_table(CORPUS_DATA_DIR[rec["corpus"]], rec["session_key"])
    if electrode["space_for_labels"] is None:
        return {"status": "excluded", "reason": "no_electrode_anatomy_table_of_either_space"}
    labels = electrode["labels"]

    tier_of_channel = [
        classify_channel_tier(name, labels) if trusted_mask[i] else None for i, name in enumerate(ch_names)
    ]

    # Reproduction-gate probe: the SAME artifact-cleaned, untiered channel set and the SAME displacement
    # primitive the delivered response module's own Block B uses, so this module's own channel masking and
    # control/stimulated split can be checked exactly against that already-delivered artifact.
    full_activity = _bin_averaged(arrays, trusted_mask)
    repro = compute_block_b_displacement(full_activity, arrays["stim_flag"])
    reproduction_probe = {
        "n_control_trials_finite": int(np.isfinite(repro["control_deviation"]).sum()),
        "n_stim_trials_finite": int(np.isfinite(repro["stim_deviation"]).sum()),
    }

    seed_base = f"human_stimulation_deviation_axis_alignment|{rec['corpus']}|{rec['session_key']}"
    tiers = {}
    for tier in DEPTH_TIERS:
        tier_mask = np.array([t == tier for t in tier_of_channel])
        tiers[tier] = _fit_session_tier(arrays, tier_mask, f"{seed_base}|{tier}")

    return {
        "status": "computed",
        "n_trusted_channels": int(trusted_mask.sum()),
        "n_channels_per_tier": {t: int(sum(1 for x in tier_of_channel if x == t)) for t in DEPTH_TIERS},
        "n_trusted_channels_unclassified": int(sum(
            1 for i, t in enumerate(tier_of_channel) if trusted_mask[i] and t is None)),
        "electrode_table_space_for_labels": electrode["space_for_labels"],
        "reproduction_probe_excluding_stimulated_shank_full_channel_set": reproduction_probe,
        "tiers": tiers,
    }


# =======================================================================================================
# Reproduction gates against the two already-delivered human artifacts this module builds on.
# =======================================================================================================

def _reproduce_zero_drop(openloop: dict, closedloop: dict) -> dict:
    delivered = json.loads(COMPONENT_RESPONSE_PATH.read_text())["zero_drop_accounting"]
    checks = {}
    for name, fresh in (("open_loop_ds005489", openloop), ("closed_loop_ds005557", closedloop)):
        d = delivered[name]
        checks[name] = {
            "delivered_n_sessions_total": d["n_sessions_total"], "fresh_n_sessions_total": fresh["n_sessions_total"],
            "delivered_n_sessions_used": d["n_sessions_used"], "fresh_n_sessions_used": fresh["n_sessions_used"],
            "matches": bool(d["n_sessions_total"] == fresh["n_sessions_total"]
                            and d["n_sessions_used"] == fresh["n_sessions_used"]),
        }
    return {
        "status": "reproduced_exactly" if all(c["matches"] for c in checks.values()) else "not_reproduced",
        "rule": "per-corpus session totals and usable-session counts, recomputed here via the same "
                "load_corpus function the delivered component-response artifact's own producer script "
                "uses, must exactly match that delivered artifact's own zero_drop_accounting",
        "checks": checks,
    }


def _reproduce_block_b_trial_counts(per_session: dict) -> dict:
    delivered_block_b = json.loads(COMPONENT_RESPONSE_PATH.read_text())["block_b"]["per_session"]
    checks = {}
    for session_key, rec in per_session.items():
        if rec.get("status") != "computed":
            continue
        delivered_cond = delivered_block_b.get(session_key, {}).get("conditions", {}).get(
            "excluding_stimulated_shank", {})
        if delivered_cond.get("status") != "computed":
            continue
        fresh = rec["reproduction_probe_excluding_stimulated_shank_full_channel_set"]
        matches = (fresh["n_control_trials_finite"] == delivered_cond.get("n_control_trials")
                   and fresh["n_stim_trials_finite"] == delivered_cond.get("n_stim_trials"))
        checks[session_key] = {
            "delivered_n_control_trials": delivered_cond.get("n_control_trials"),
            "fresh_n_control_trials": fresh["n_control_trials_finite"],
            "delivered_n_stim_trials": delivered_cond.get("n_stim_trials"),
            "fresh_n_stim_trials": fresh["n_stim_trials_finite"],
            "matches": bool(matches),
        }
    return {
        "status": ("reproduced_exactly" if checks and all(c["matches"] for c in checks.values())
                   else "not_reproduced"),
        "rule": "every session's finite control- and stimulated-trial counts under the "
                "excluding_stimulated_shank channel condition, recomputed here with the identical "
                "displacement primitive (compute_block_b_displacement) the delivered artifact's own "
                "producer script uses on the identical channel set, must exactly match that delivered "
                "artifact's per-session values for the same condition",
        "n_sessions_checked": len(checks), "checks": checks,
    }


# =======================================================================================================
# Main
# =======================================================================================================

def main() -> None:
    t0 = time.time()

    openloop = load_corpus("open_loop_ds005489", OPENLOOP_DATA, derive_stim_from_stim_on=False, smoke=None)
    closedloop = load_corpus("closed_loop_ds005557", CLOSEDLOOP_DATA, derive_stim_from_stim_on=True, smoke=None)
    all_records = openloop["records"] + closedloop["records"]

    zero_drop_reproduction = _reproduce_zero_drop(openloop, closedloop)
    if zero_drop_reproduction["status"] != "reproduced_exactly":
        raise AssertionError("freshly computed session totals do not reproduce "
                              "results/human_stimulation_component_response.json's zero_drop_accounting "
                              "-- refusing to report")

    per_session = {}
    for rec in all_records:
        record = _run_checkpointed(f"session|{rec['corpus']}|{rec['session_key']}", lambda r=rec: _fit_session(r))
        per_session[rec["session_key"]] = {"corpus": rec["corpus"], "subject": rec["subject_id"], **record}

    n_analysed = sum(1 for r in per_session.values() if r.get("status") == "computed")
    excluded = [{"session": k, "corpus": v["corpus"], "reason": v.get("reason", v.get("status"))}
                for k, v in per_session.items() if v.get("status") != "computed"]
    zero_drop_accounting = {
        "sessions_seen": len(all_records), "sessions_analysed": n_analysed, "sessions_excluded": len(excluded),
        "exclusions": excluded, "reconciles": bool(len(all_records) == n_analysed + len(excluded)),
        "n_subjects_seen": len({r["subject_id"] for r in all_records}),
        "n_subjects_analysed": len({per_session[k]["subject"] for k in per_session
                                    if per_session[k].get("status") == "computed"}),
    }

    block_b_reproduction = _reproduce_block_b_trial_counts(per_session)
    if block_b_reproduction["status"] != "reproduced_exactly":
        raise AssertionError("freshly computed control/stimulated trial counts do not reproduce "
                              "results/human_stimulation_component_response.json's block_b per-session "
                              "values -- refusing to report")

    computed_sessions = sorted(s for s, r in per_session.items() if r.get("status") == "computed")
    corpus_of = {s: per_session[s]["corpus"] for s in computed_sessions}
    pooling_groups_sessions = {
        "all_admissible_sessions": {
            "sessions": computed_sessions,
            "criterion": "every session with an admitted electrode anatomy table and at least one usable "
                         "tier fit, pooling both the experimenter-scheduled and classifier-triggered "
                         "stimulation corpora",
        },
        "open_loop_ds005489_only": {
            "sessions": [s for s in computed_sessions if corpus_of[s] == "open_loop_ds005489"],
            "criterion": "restricted to the experimenter-scheduled arm, whose stimulation assignment is a "
                         "design property (alternating word-list blocks), not read from the subject's own "
                         "state",
        },
        "closed_loop_ds005557_only": {
            "sessions": [s for s in computed_sessions if corpus_of[s] == "closed_loop_ds005557"],
            "criterion": "restricted to the classifier-triggered arm, whose stimulation assignment is NOT "
                         "randomised -- every number in this group is descriptive/associational, not causal",
        },
    }

    per_tier_output = {}
    for tier in DEPTH_TIERS:
        per_session_tier = {s: per_session[s]["tiers"][tier] for s in computed_sessions}
        decomposition = {}
        for group_name, group in pooling_groups_sessions.items():
            sessions = group["sessions"]
            by_arm = {}
            for arm in ARMS:
                real_pooled = _pool_group(per_session_tier, sessions, arm, "real_alignment")
                bias_pooled = _pool_group(per_session_tier, sessions, arm, "bias_only_alignment")
                classification = _classify_arm(real_pooled, bias_pooled)
                mdd = classification["mdd"]
                by_arm[arm] = {
                    "n_sessions_pooled": real_pooled["n_sessions_pooled"],
                    "real_pooled_alignment": real_pooled,
                    "bias_only_pooled_alignment": bias_pooled,
                    "bias_only_reproduces_real": _bias_only_voids(real_pooled, bias_pooled),
                    "branch": classification["branch"],
                    "effect_abs_cosine": classification["effect"],
                    "minimum_detectable_difference_80pct_power": mdd,
                    "clears_behavioural_reference_0_14": (mdd < BEHAVIOURAL_REFERENCE_R_UNITS
                                                            if mdd is not None else None),
                    "clears_continuity_alignment_floor_0_05": (mdd < CONTINUITY_ALIGNMENT_FLOOR_ABS_COSINE
                                                                if mdd is not None else None),
                }
            raw_branch, detrended_branch = by_arm["raw_axis"]["branch"], by_arm["detrended_axis"]["branch"]
            positive_branches = {"stimulation_pushes_along_the_deviation_axis",
                                  "stimulation_pushes_off_the_deviation_axis"}
            if raw_branch in positive_branches and detrended_branch in positive_branches:
                agreement = "agree" if raw_branch == detrended_branch else "disagree"
            else:
                agreement = "not_applicable_one_or_both_arms_not_decided"
            decomposition[group_name] = {
                "criterion": group["criterion"],
                "n_sessions_in_group": len(sessions),
                "by_arm": by_arm,
                "raw_versus_detrended_agreement": agreement,
                "licensed_directional_verdict": (
                    raw_branch if agreement == "agree" else
                    "stimulation_moved_the_drift_not_the_component_raw_and_detrended_arms_disagree"
                    if agreement == "disagree" else
                    "not_licensed_one_or_both_arms_not_decided"
                ),
            }
        per_tier_output[tier] = {
            "status": "present",
            "n_channels_median": float(np.median([per_session[s]["n_channels_per_tier"][tier]
                                                    for s in computed_sessions
                                                    if per_session[s]["tiers"][tier].get("status") == "computed"]))
            if any(per_session[s]["tiers"][tier].get("status") == "computed" for s in computed_sessions) else None,
            "pooling_groups": decomposition,
        }
    for tier, reason in ABSENT_TIERS.items():
        per_tier_output[tier] = {"status": "absent", "reason": reason}

    per_session_output = {}
    for key, rec in per_session.items():
        out = {k: v for k, v in rec.items() if k != "tiers"}
        if rec.get("status") == "computed":
            out["tiers"] = {}
            for tier, cell in rec["tiers"].items():
                cell_out = dict(cell)
                if cell_out.get("status") == "computed":
                    cell_out["by_arm"] = {}
                    for arm, arm_cell in cell["by_arm"].items():
                        arm_out = dict(arm_cell)
                        for field in ("real_alignment", "bias_only_alignment"):
                            if arm_out.get(field) is not None:
                                arm_out[field] = {k: v for k, v in arm_out[field].items() if k != "draws"}
                        cell_out["by_arm"][arm] = arm_out
                out["tiers"][tier] = cell_out
        per_session_output[key] = out

    primary_mtl = per_tier_output["depth_mtl"]["pooling_groups"]["all_admissible_sessions"]
    primary_cortical = per_tier_output["depth_cortical"]["pooling_groups"]["all_admissible_sessions"]

    non_human_comparison = None
    macaque_path = RESULTS / "macaque_pfc_microstimulation_stimulation_deviation_axis_alignment.json"
    if macaque_path.exists():
        macaque = json.loads(macaque_path.read_text())
        macaque_primary = macaque["pooling_groups"]["all_admissible_sessions"]
        non_human_comparison = (
            "The nearest delivered non-human equivalent (a macaque intracortical microstimulation corpus "
            "asking the identical axis-alignment question with the identical estimator) is VOID at its own "
            "primary pooling group: raw-axis arm branch="
            f"{macaque_primary['by_arm']['raw_axis']['branch']}, detrended-axis arm branch="
            f"{macaque_primary['by_arm']['detrended_axis']['branch']}, licensed verdict="
            f"{macaque_primary['licensed_directional_verdict']}. That preparation delivers stimulation at "
            "single-unit resolution strictly inside a working-memory delay window that overlaps current "
            "delivery by design, exactly the same epoch-overlap property both human corpora carry here; the "
            "two preparations differ in species, recording modality and channel count, not in epoch design."
        )

    verdict = (
        f"Medial-temporal depth, primary group (all {primary_mtl['n_sessions_in_group']} admissible "
        f"sessions): raw-axis arm fires {primary_mtl['by_arm']['raw_axis']['branch']} (effect "
        f"{primary_mtl['by_arm']['raw_axis']['effect_abs_cosine']}, mdd "
        f"{primary_mtl['by_arm']['raw_axis']['minimum_detectable_difference_80pct_power']}); detrended-axis "
        f"arm fires {primary_mtl['by_arm']['detrended_axis']['branch']} (effect "
        f"{primary_mtl['by_arm']['detrended_axis']['effect_abs_cosine']}, mdd "
        f"{primary_mtl['by_arm']['detrended_axis']['minimum_detectable_difference_80pct_power']}); "
        f"raw-versus-detrended agreement: {primary_mtl['raw_versus_detrended_agreement']}; licensed "
        f"verdict: {primary_mtl['licensed_directional_verdict']}. "
        f"Cortical depth, primary group (all {primary_cortical['n_sessions_in_group']} admissible "
        f"sessions): raw-axis arm fires {primary_cortical['by_arm']['raw_axis']['branch']} (effect "
        f"{primary_cortical['by_arm']['raw_axis']['effect_abs_cosine']}, mdd "
        f"{primary_cortical['by_arm']['raw_axis']['minimum_detectable_difference_80pct_power']}); "
        f"detrended-axis arm fires {primary_cortical['by_arm']['detrended_axis']['branch']} (effect "
        f"{primary_cortical['by_arm']['detrended_axis']['effect_abs_cosine']}, mdd "
        f"{primary_cortical['by_arm']['detrended_axis']['minimum_detectable_difference_80pct_power']}); "
        f"raw-versus-detrended agreement: {primary_cortical['raw_versus_detrended_agreement']}; licensed "
        f"verdict: {primary_cortical['licensed_directional_verdict']}. Every stimulated trial analysed here "
        "is cropped to -0.3 to +1.6 s relative to word onset, so the analysed epoch overlaps current "
        "delivery by design and neither corpus has an isolated maintenance delay separate from encoding -- "
        "this travels with any positive branch above as part of the result, not a footnote."
    )

    output = {
        "analysis_id": "human_stimulation_deviation_axis_alignment",
        "schema_version": "1.0.0",
        "trigger": (
            "results/recording_tier_component_transfer.json shows the deviation component present at "
            "every recording tier this project has built in human data; results/"
            "human_stimulation_component_response.json shows human intracranial stimulation measurably "
            "displaces the same component. This module asks whether that displacement lands along the "
            "component's own control-trial-only axis or off it, in the one pair of human corpora where "
            "both a stimulation displacement and enough non-stimulated trials to fit an axis exist in the "
            "same sessions."
        ),
        "code_commit": git_commit(ROOT),
        "corpora": ["open_loop_ds005489", "closed_loop_ds005557"],
        "epoch_disclosure": (
            "every trial's window is -0.3 to +1.6 s relative to word onset (word on-screen for 1.6 s), "
            "identically for control and stimulated trials; the stimulation pulse train in both corpora "
            "outlasts a single word presentation, so the analysed epoch overlaps current delivery by design "
            "for a stimulated trial; neither corpus has an isolated working-memory maintenance delay "
            "separate from encoding"
        ),
        "stimulation_kind": {
            "open_loop_ds005489": "experimenter-scheduled electrical stimulation in alternating word-list "
                                   "blocks during list encoding; assignment is a design property",
            "closed_loop_ds005557": "online-classifier-triggered electrical stimulation during list "
                                     "encoding; assignment is NOT randomised, so every number from this "
                                     "corpus is descriptive/associational, never causal",
        },
        "feature_space": "per-channel log high-gamma power, averaged over the -0.3 to +1.6 s peri-word "
                          "window, L2-normalised to a unit direction per trial before either the axis or "
                          "the displacement is estimated -- the identical rate-free construction the "
                          "deviation estimator uses internally, kept common between the axis and the "
                          "displacement so their cosine is meaningful",
        "channel_artifact_control": "every fit uses only channel_condition_masks' "
                                     "excluding_stimulated_shank condition -- the driven bipolar pair and "
                                     "every channel sharing a physical lead with either of its two contacts "
                                     "excluded -- the same artifact-cleaned channel set the delivered "
                                     "displacement module treats as its own primary condition",
        "tier_construction": {
            "label_convention": LABEL_CONVENTION,
            "label_field": "ind.region",
            "mtl_structures": sorted(MTL_STRUCTURES),
            "rule": "a channel is depth_mtl if either bipolar contact's ind.region label resolves to a "
                    "medial-temporal-lobe structure, depth_cortical if it resolves to any other named "
                    "structure, and unclassified (excluded from both tiers) if neither contact resolves; "
                    "the first-listed contact is tried first, the second only as a fallback",
            "scope_limitation": "ind.region is a cortical-surface (Desikan-Killiany aparc) atlas and does "
                                "not directly label hippocampus or amygdala proper; the depth_mtl tier "
                                "built from it is dominated by parahippocampal and entorhinal contacts, a "
                                "limit of the corpus's own declared label_convention",
        },
        "circularity_guard": "estimate_axis raises ValueError unless called with source='control_only'; "
                              "no stimulated trial is ever passed to it in this pipeline",
        "signed_versus_magnitude_discipline": (
            "the axis is a leading eigenvector with no session-comparable sign convention established "
            "anywhere in this project; every alignment statistic here is therefore unsigned (absolute "
            "cosine and its square), and no signed-displacement language is used"
        ),
        "recording_tiers": per_tier_output,
        "per_session": per_session_output,
        "reproduction_gate": {
            "zero_drop_against_component_response_artifact": zero_drop_reproduction,
            "control_and_stimulated_trial_counts_against_component_response_block_b": block_b_reproduction,
        },
        "zero_drop_accounting": zero_drop_accounting,
        "non_human_comparison": non_human_comparison,
        "verdict": verdict,
        "wall_clock_s": round(time.time() - t0, 3),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)
    _log(f"wrote {OUTPUT_PATH} in {output['wall_clock_s']:.1f}s -- mtl verdict: "
         f"{primary_mtl['licensed_directional_verdict']}, cortical verdict: "
         f"{primary_cortical['licensed_directional_verdict']}")


if __name__ == "__main__":
    main()
