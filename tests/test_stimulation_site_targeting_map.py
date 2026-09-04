"""Tests for scripts/run_stimulation_site_targeting_map.py.

Five things are checked directly: (1) the distance computation (channel
midpoint + Euclidean distance in MNI space), (2) the missing-value-sentinel
guard on the electrode table's numeric fields, (3) the three-way trusted-
channel control -- a synthetic near/artifactual channel that dominates the
carriage-weighted predictor when included is fully removed once its shank is
excluded, recovering the true distance, (4) the session join between the
delivered displacement artifact and a corpus's delivered behavioural
artifact, including its named refusal path when no behavioural row matches,
and (5) the subject-clustered pooling this module's own Block A performs,
including how a single channel-condition-specific refusal for one session
propagates into that condition's own subject/session counts without
affecting the other two conditions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_human_stimulation_component_response import channel_condition_masks  # noqa: E402
import run_stimulation_site_targeting_map as targeting  # noqa: E402
from run_stimulation_site_targeting_map import (  # noqa: E402
    _causal_key,
    _lobe_for_label,
    _numeric_or_nan,
    channel_midpoint,
    classify_targeting_relationship,
    euclidean,
    load_admitted_sessions,
    load_electrode_table,
    predictor_for_channel_condition,
    run_block_a,
)


# ---------------------------------------------------------------------------------------------------
# (1) Distance computation
# ---------------------------------------------------------------------------------------------------

def test_channel_midpoint_averages_the_two_contacts():
    coords = {"LAH1": (0.0, 0.0, 0.0), "LAH2": (2.0, 4.0, 6.0)}
    assert channel_midpoint("LAH1-LAH2", coords) == pytest.approx((1.0, 2.0, 3.0))


def test_channel_midpoint_returns_none_for_unresolvable_channel():
    coords = {"LAH1": (0.0, 0.0, 0.0)}
    assert channel_midpoint("LAH1-LAH2", coords) is None       # LAH2 unknown
    assert channel_midpoint("LAH1", coords) is None             # not a bipolar name


def test_euclidean_matches_known_3_4_5_triangle():
    assert euclidean((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------------------------------
# (2) Missing-value-sentinel guard
# ---------------------------------------------------------------------------------------------------

def test_numeric_or_nan_treats_known_sentinels_as_missing():
    assert np.isnan(_numeric_or_nan("-999"))
    assert np.isnan(_numeric_or_nan("n/a"))
    assert np.isnan(_numeric_or_nan("N/A"))
    assert np.isnan(_numeric_or_nan(""))
    assert np.isnan(_numeric_or_nan(None))
    assert _numeric_or_nan("12.5") == pytest.approx(12.5)


def test_load_electrode_table_excludes_sentinel_coordinates_from_the_coords_dict(tmp_path):
    # size=-999 everywhere (this release's usual sentinel column) plus one contact whose x is the
    # literal missing token "n/a" -- neither may ever produce a usable midpoint.
    table = tmp_path / "sub-X_ses-0_task-FR2_space-MNI152NLin6ASym_electrodes.tsv"
    table.write_text(
        "name\tx\ty\tz\tsize\tgroup\themisphere\ttype\tind.region\tdas.region\tstein.region\n"
        "LAH1\t1.0\t2.0\t3.0\t-999\tLAH\tL\tdepth\tn/a\tCA1\tLeft CA1\n"
        "LAH2\tn/a\tn/a\tn/a\t-999\tLAH\tL\tdepth\tn/a\tn/a\tn/a\n"
    )
    ieeg_json = tmp_path / "sub-X_ses-0_task-FR2_acq-bipolar_ieeg.json"
    ieeg_json.write_text("{}")

    out = load_electrode_table(tmp_path, ieeg_json.name)
    assert out["mni_table_found"] is True
    assert "LAH1" in out["coords"]
    assert "LAH2" not in out["coords"]          # the n/a-coordinate contact never enters coords
    # The label columns are still read for a contact with unresolvable coordinates.
    assert out["labels"]["LAH2"]["das.region"] == "n/a"
    assert out["labels"]["LAH1"]["das.region"] == "CA1"
    # A midpoint built from LAH1-LAH2 is therefore unresolvable -- the sentinel never reaches an average.
    assert channel_midpoint("LAH1-LAH2", out["coords"]) is None


def test_lobe_mapping_ignores_stein_region_hemisphere_prefix():
    assert _lobe_for_label("stein.region", "Left CA1") == "temporal"
    assert _lobe_for_label("stein.region", "Right CA1") == "temporal"
    assert _lobe_for_label("ind.region", "rostralmiddlefrontal") == "frontal"
    assert _lobe_for_label("ind.region", "some_unknown_label") == "unmapped_lobe"


# ---------------------------------------------------------------------------------------------------
# (3) Three-way trusted-channel control
# ---------------------------------------------------------------------------------------------------

def test_artifact_dominated_predictor_is_corrected_by_shank_exclusion():
    # A2 sits on the same shank as the driven pair (A1, A2) and is the loudest channel by a wide
    # margin -- exactly the stimulation-artifact profile this control exists to catch. B1-B2, B2-B3
    # and C1-C2 sit on two distant shanks and carry comparable, much smaller signal (three of them,
    # so the whole-shank-excluded condition still clears the minimum-channel-geometry floor).
    ch_names = ["A1-A2", "A2-A3", "B1-B2", "B2-B3", "C1-C2"]
    anode, cathode, stim_channel = "A1", "A2", "A1-A2"
    coords = {
        "A1": (0.0, 0.0, 0.0), "A2": (1.0, 0.0, 0.0), "A3": (2.0, 0.0, 0.0),
        "B1": (100.0, 0.0, 0.0), "B2": (101.0, 0.0, 0.0), "B3": (102.0, 0.0, 0.0),
        "C1": (200.0, 0.0, 0.0), "C2": (201.0, 0.0, 0.0),
    }
    stim_site = (0.5, 0.0, 0.0)
    # Distances from stim_site to each channel's own midpoint: A1-A2=0, A2-A3=1, B1-B2=100,
    # B2-B3=101, C1-C2=200.
    # Identical rows -> the reference direction is exactly base / norm(base), no estimation noise.
    base = np.array([50.0, 1.0, 1.0, 1.0, 1.0])
    ctrl_activity_all = np.tile(base, (20, 1))

    masks = channel_condition_masks(ch_names, anode, cathode, stim_channel)

    results = {}
    for cond, mask in masks.items():
        kept = [n for n, keep in zip(ch_names, mask) if keep]
        activity = ctrl_activity_all[:, mask]
        results[cond] = predictor_for_channel_condition(activity, kept, coords, stim_site, int(mask.sum()))

    # full_channel_set: dominated by the artifact channel A1-A2, itself AT the stimulated site
    # (distance 0), so the carriage-weighted mean distance is pulled far below the true tissue distance.
    assert results["full_channel_set"]["status"] == "computed"
    assert results["full_channel_set"]["distance_weighted_carriage_predictor_mm"] == \
        pytest.approx((50 * 0 + 1 * 1 + 1 * 100 + 1 * 101 + 1 * 200) / 54)

    # excluding_stimulated_pair drops only A1-A2, leaving the still-near A2-A3 channel in the mix.
    assert results["excluding_stimulated_pair"]["status"] == "computed"
    assert results["excluding_stimulated_pair"]["distance_weighted_carriage_predictor_mm"] == \
        pytest.approx((1 * 1 + 1 * 100 + 1 * 101 + 1 * 200) / 4)

    # excluding_stimulated_shank drops the whole A shank, leaving only the three distant channels --
    # a much larger distance than the artifact-contaminated full-channel-set read.
    assert results["excluding_stimulated_shank"]["status"] == "computed"
    assert results["excluding_stimulated_shank"]["distance_weighted_carriage_predictor_mm"] == \
        pytest.approx((1 * 100 + 1 * 101 + 1 * 200) / 3)
    assert (results["excluding_stimulated_shank"]["distance_weighted_carriage_predictor_mm"]
           > results["full_channel_set"]["distance_weighted_carriage_predictor_mm"])


def test_predictor_for_channel_condition_refuses_too_few_control_trials():
    out = predictor_for_channel_condition(np.zeros((3, 2)), ["A-B", "C-D"], {}, (0.0, 0.0, 0.0), 2)
    assert out["status"] == "too_few_control_trials"


def test_predictor_for_channel_condition_refuses_when_geometry_is_unresolvable():
    activity = np.tile(np.array([1.0, 1.0]), (20, 1))
    # Neither channel's contacts are in `coords`, so no distance can ever be resolved.
    out = predictor_for_channel_condition(activity, ["A-B", "C-D"], {}, (0.0, 0.0, 0.0), 2)
    assert out["status"] == "insufficient_channel_geometry"
    assert out["n_channels_with_known_geometry"] == 0


# ---------------------------------------------------------------------------------------------------
# (4) Session join and its refusal path
# ---------------------------------------------------------------------------------------------------

def test_causal_key_swaps_json_sidecar_suffix_for_edf():
    key = "sub-R1001P/ses-0/ieeg/sub-R1001P_ses-0_task-FR2_acq-bipolar_ieeg.json"
    assert _causal_key(key) == "sub-R1001P/ses-0/ieeg/sub-R1001P_ses-0_task-FR2_acq-bipolar_ieeg.edf"


def test_load_admitted_sessions_joins_behavior_and_names_the_refusal(tmp_path, monkeypatch):
    matched_key = "sub-A/ses-0/ieeg/sub-A_ses-0_task-FR2_acq-bipolar_ieeg.json"
    unmatched_key = "sub-B/ses-0/ieeg/sub-B_ses-0_task-FR2_acq-bipolar_ieeg.json"
    component = {
        "block_b": {"per_session": {
            matched_key: {"corpus": "open_loop_ds005489", "subject": "sub-A", "anode": "A1",
                         "cathode": "A2", "stim_channel": "A1-A2", "conditions": {}},
            unmatched_key: {"corpus": "open_loop_ds005489", "subject": "sub-B", "anode": "B1",
                           "cathode": "B2", "stim_channel": "B1-B2", "conditions": {}},
        }}
    }
    causal_ram = {"per_session": {
        "sub-A/ses-0/ieeg/sub-A_ses-0_task-FR2_acq-bipolar_ieeg.edf": {
            "recall_rate_stim": 0.3, "recall_rate_ctrl": 0.5, "n_words": 300,
        },
    }}
    component_path = tmp_path / "component.json"
    component_path.write_text(json.dumps(component))
    causal_path = tmp_path / "causal_ram.json"
    causal_path.write_text(json.dumps(causal_ram))
    closed_path = tmp_path / "causal_ram_closedloop.json"
    closed_path.write_text(json.dumps({"per_session": {}}))

    monkeypatch.setattr(targeting, "COMPONENT_RESPONSE_PATH", component_path)
    monkeypatch.setattr(targeting, "CAUSAL_ARTIFACT", {
        "open_loop_ds005489": causal_path, "closed_loop_ds005557": closed_path,
    })

    sessions = load_admitted_sessions()
    by_key = {s["session_key"]: s for s in sessions}

    matched = by_key[matched_key]["behavior"]
    assert matched["status"] == "computed"
    assert matched["recall_diff"] == pytest.approx(0.3 - 0.5)

    unmatched = by_key[unmatched_key]["behavior"]
    assert unmatched["status"] == "excluded"
    assert unmatched["reason"] == "no_behavioral_outcome_in_delivered_causal_artifact"


# ---------------------------------------------------------------------------------------------------
# (5) Subject-clustered pooling and its pre-declared branch classifier
# ---------------------------------------------------------------------------------------------------

def _session(key, subject, predictor, displacement, condition="full_channel_set"):
    conditions = {c: {"status": "not_computable"} for c in
                 ("full_channel_set", "excluding_stimulated_pair", "excluding_stimulated_shank")}
    conditions[condition] = {"status": "computed", "distance_weighted_carriage_predictor_mm": predictor}
    disp_conditions = {c: {"status": "not_computable"} for c in
                      ("full_channel_set", "excluding_stimulated_pair", "excluding_stimulated_shank")}
    disp_conditions[condition] = {"status": "computed", "normalised_displacement": displacement}
    return (
        {"session_key": key, "corpus": "open_loop_ds005489", "subject": subject,
         "displacement_conditions": disp_conditions, "behavior": {"status": "excluded", "reason": "n/a"}},
        {key: {"status": "computed", "conditions": conditions}},
    )


def test_run_block_a_per_condition_refusal_does_not_shrink_the_other_conditions():
    # Four subjects, one session each. Every session has geometry computed in ALL THREE conditions
    # except subject S4's own excluding_stimulated_shank cell, which this session's own geometry
    # marks not_computable -- exactly what a real too-few-control-trials refusal on one channel
    # condition looks like once it reaches Block A.
    sessions, geometry = [], {}
    for subj, predictor, disp in (("S1", 10.0, 0.9), ("S2", 20.0, 0.5), ("S3", 30.0, 0.3), ("S4", 40.0, 0.1)):
        for cond in ("full_channel_set", "excluding_stimulated_pair", "excluding_stimulated_shank"):
            if subj == "S4" and cond == "excluding_stimulated_shank":
                continue
            key = f"{subj}__{cond}"
            s, g = _session(key, subj, predictor, disp, condition=cond)
            sessions.append(s)
            geometry.update(g)

    out = run_block_a(sessions, geometry)

    assert out["displacement_relationship"]["n_subjects_by_condition"]["full_channel_set"] == 4
    assert out["displacement_relationship"]["n_subjects_by_condition"]["excluding_stimulated_pair"] == 4
    # S4 contributes no excluding_stimulated_shank row anywhere -- that condition sees only 3 subjects.
    assert out["displacement_relationship"]["n_subjects_by_condition"]["excluding_stimulated_shank"] == 3

    # subject_aggregated_correlation refuses to compute below 4 subjects, so the shank-excluded
    # condition -- the headline condition -- must fall back to underpowered_to_ask, never a null.
    shank = out["displacement_relationship"]["by_channel_condition"]["excluding_stimulated_shank"]
    assert shank["status"] == "not_computable"
    assert out["displacement_relationship"]["branch"] == "underpowered_to_ask"


def test_classify_targeting_relationship_fires_artifact_branch_when_shank_does_not_replicate():
    corr = {
        "full_channel_set": {"status": "computed", "r": -0.6, "p_value": 0.01,
                             "mdd": {"status": "computed", "mdd": 0.2}},
        "excluding_stimulated_shank": {"status": "computed", "r": 0.05, "p_value": 0.8,
                                       "mdd": {"status": "computed", "mdd": 0.2}},
    }
    assert classify_targeting_relationship(corr) == "site_distance_effect_not_separable_from_recording_artifact"


def test_classify_targeting_relationship_fires_positive_branch_when_shank_replicates_same_sign():
    corr = {
        "full_channel_set": {"status": "computed", "r": -0.6, "p_value": 0.01,
                             "mdd": {"status": "computed", "mdd": 0.2}},
        "excluding_stimulated_shank": {"status": "computed", "r": -0.55, "p_value": 0.02,
                                       "mdd": {"status": "computed", "mdd": 0.2}},
    }
    assert classify_targeting_relationship(corr) == "stimulating_closer_to_the_carrying_contacts_moves_the_component_more"


def test_classify_targeting_relationship_fires_null_only_when_mdd_clears_the_reference_effect():
    well_powered = {
        "full_channel_set": {"status": "computed", "r": 0.02, "p_value": 0.9,
                             "mdd": {"status": "computed", "mdd": 0.10}},
        "excluding_stimulated_shank": {"status": "computed", "r": 0.01, "p_value": 0.95,
                                       "mdd": {"status": "computed", "mdd": 0.10}},
    }
    assert classify_targeting_relationship(well_powered) == "no_site_distance_relationship_above_the_reported_bound"

    underpowered = {
        "full_channel_set": {"status": "computed", "r": 0.02, "p_value": 0.9,
                             "mdd": {"status": "computed", "mdd": 0.45}},
        "excluding_stimulated_shank": {"status": "computed", "r": 0.01, "p_value": 0.95,
                                       "mdd": {"status": "computed", "mdd": 0.45}},
    }
    assert classify_targeting_relationship(underpowered) == "underpowered_to_ask"
