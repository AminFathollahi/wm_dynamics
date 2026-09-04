#!/usr/bin/env python3
"""Does intracortical microstimulation displace the macaque dorsolateral prefrontal population state
along the same low-dimensional direction that carries the behavioural deviation signal, or off it?

This is the only corpus in this project's registry where a rate-free deviation axis and a delivered
stimulation displacement can both be measured in the SAME sessions (results/stimulation_axis_same_
session_census.json censused every corpus; every other pairing shares zero sessions between the corpus
that carries the axis and the corpus that carries stimulation). It is also the corpus whose control
trials sit inside a spike-count range this project had never sampled before that census; results/
macaque_pfc_microstimulation_deviation_orthogonality_gate.json already ran the deviation observable's own orthogonality gate
here and found it passes pooled across the corpus's 11 sessions as a powered null (r=-0.0936, p=0.1001,
minimum detectable difference 0.1450 below the 0.1704 reference), with two individual sessions failing
their own gate (Wa220802_s550 at median 1190 spikes/trial, r=-0.4323, p=0.0001; Wa220803_s551 at median
753 spikes/trial, r=-0.3947, p=0.0003).

CONSTRUCTION. The axis and the displacement are estimated in the identical feature space so that a
cosine between them is meaningful: per session, every trial's (n_channels,) total-spike-count vector
(30 bins x 50 ms, -0.8 to +0.7 s relative to stimulation onset, summed and rescaled to a per-trial
count -- the identical construction results/macaque_pfc_microstimulation_deviation_orthogonality_gate.json already verified
against results/stimulation_axis_same_session_census.json's own fresh measurement) is L2-normalised to
a unit direction (unit_direction_vectors, the same primitive rate_free_state_deviation computes
internally). The axis is the leading eigenvector of the residual-direction covariance built from
control (non-stimulation) trials ONLY -- r_i = u_i - (u_i . m_-i) m_-i, m_-i the leave-one-out mean
direction over the session's other control trials, exactly the construction results/
deviation_axis_structure.json and results/deviation_axis_identity_controls.json already established and
verified stable within session. A stimulated trial is never permitted into that fit: the axis-estimation
entry point below (estimate_axis) takes an explicit source argument and raises if it is not
"control_only", so a caller that tried to slip a stimulated trial into the fit would fail immediately
rather than silently contaminate the estimate.

The displacement direction is the difference of the mean unit-normalised direction over every admitted
stimulated trial (any non-control condition, correct and error trials both, the identical trial pool
results/stimulation_latent_response_map.json's own macaque_pfc_microstimulation arm uses as its baseline-vs-condition
contrast) and the mean unit-normalised direction over the same control trials the axis was fit on. Using
unit-normalised directions on both sides of the difference, rather than raw per-channel spike counts,
keeps the displacement rate-free in the same sense the axis itself already is, since this project's own
identity-controls artifact shows the axis points substantially along the total-spike-count direction by
construction (deviation vs count 0.0247 / 0.0043, both non-significant) -- a raw-count displacement would
otherwise be dominated by any overall firing-rate change stimulation produces, which is not the question
being asked.

TWO ARMS, MANDATORY AGREEMENT. The axis's alignment with a session's slow linear drift has already been
shown mechanical in this project (a synthetic pure-drift trial set with no deviation structure at all
produces a HIGHER axis-drift alignment than the real data), so a displacement landing along the RAW axis
is not by itself evidence that stimulation moves the behaviourally-linked component -- stimulation could
simply be moving the drift, which the raw axis partly is. Every decomposition here is therefore run
twice: once against the raw axis, and once against the axis re-estimated after each control trial's own
linear time-trend is regressed out first (_linear_detrend_activity, the identical detrending function
results/deviation_axis_identity_controls.json already uses and verified the behaviourally-live deviation
survives). Only agreement between the two arms licenses a directional sentence about the component;
disagreement is reported as its own finding (stimulation moved the drift, not the component), never
silently resolved toward whichever arm looks more interesting.

CHANNEL-FILTER CROSS-CHECK. A separate analysis in this project (results/state_space_dimensionality_
sweep.json, key causal_microstimulation_sessions) excludes three of these eleven sessions -- Sa210311_
s224, Wa220803_s551, Wa220811_s557 -- with the reason "no stimulation condition survived the channel
filter". That filter exists to locate the exact recorded-channel index of the stimulating electrode
itself, needed to build a one-hot input-direction vector for a controllability/targeting question. This
module recomputes that filter directly from the raw per-session channel_ids and stim_channels tables
(reproduce_channel_filter_classification below) and confirms, session by session, that in every one of
the three excluded sessions the stimulating electrode's own channel was removed from the recorded array
(most likely the electrically-shorted-channel exclusion this project's loader already applies) -- a real,
session-specific recording property, not a filter artifact. That property is irrelevant to the
construction here: this module never needs to know which recorded channel the stimulating electrode
occupies, only the recorded population activity vector on stimulated versus control trials, which is
fully defined for all eleven sessions. All eleven therefore remain admissible for the axis/displacement
question; the three-session channel-filter exclusion is reported as a documented cross-check, and the
eight-session subset that survives it elsewhere in this project is reported as an additional sensitivity
group so the interaction between the two exclusion criteria is visible rather than hidden. One session,
Wa220803_s551, is flagged by BOTH criteria (its own spike-count gate fails, and its stimulating
electrode's channel is absent from the recorded array) and this is stated explicitly wherever that
session's membership matters.

BIAS-ONLY CONTROL. Every prior directional claim in this project that has been checked against a plain
per-unit offset has died to exactly that check, so the same discipline applies here: a "bias-only axis"
is built by replacing the true residual-eigenvector axis with the plain normalised MEAN unit-direction
over the same control trials (the one direction the residual construction is, by definition, orthogonal
to) and the identical cosine-vs-random-direction test is rerun against it. If the bias-only alignment
reproduces the real alignment's significance and its direction relative to its own null, the observed
alignment is not separable from stimulation simply landing near the session's trivial average activity
direction, and the branch displacement_direction_not_separable_from_a_unit_level_offset fires for that
arm.

SIGNED-VERSUS-MAGNITUDE DISCIPLINE. The axis is a leading eigenvector; its sign is not fixed by the
residual construction (either sign is an equally valid leading eigenvector of the same real symmetric
matrix) and no session-comparable sign convention for it exists anywhere in this project's delivered
work. Every alignment statistic here is therefore reported unsigned (absolute cosine, and its square as
the fraction of squared displacement magnitude the one-dimensional axis projection captures), exactly as
results/deviation_axis_identity_controls.json already reports its own axis-alignment tests. No signed
displacement language is used anywhere in this artifact, and this paragraph is that disclosure.

POWER. A pooled result only licenses a null (or a positive) claim if its minimum detectable difference
at 80% power clears this project's standing 0.14-r-unit behavioural reference; a cell whose floor exceeds
the effect it tests is reported as inconclusive_below_detection_floor with both numbers in the same
field, never quoted alone.

Run:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \\
    scripts/run_macaque_pfc_microstimulation_stimulation_deviation_axis_alignment.py
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
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
from statistics import minimum_detectable_paired_difference, stable_seed  # noqa: E402
from run_macaque_pfc_microstimulation_pipeline import BIN_S, DATA, crop_trial, load_macaque_pfc_microstimulation_session  # noqa: E402
from run_deviation_serial_dependence_and_temporal_locus import unit_direction_vectors  # noqa: E402
from run_deviation_axis_structure import (  # noqa: E402
    N_ROTATION_DRAWS, _axis_stability, _pool_rotation_statistic, _residual_rows, _unit_residual_matrix,
    leading_eigenvector,
)
from run_deviation_axis_identity_controls import (  # noqa: E402
    _linear_detrend_activity, _vector_reference_alignment_with_draws,
)
from run_dissociation_cross_preparation_test import MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "macaque_pfc_microstimulation_stimulation_deviation_axis_alignment.json"
CENSUS_PATH = RESULTS / "stimulation_axis_same_session_census.json"
GATE_PATH = RESULTS / "macaque_pfc_microstimulation_deviation_orthogonality_gate.json"
DIMENSIONALITY_SWEEP_PATH = RESULTS / "state_space_dimensionality_sweep.json"

CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_macaque_pfc_microstimulation_stimulation_deviation_axis_alignment"
SCHEMA_TAG = "v1_control_only_axis_rawchannel_unitdirection_displacement_2026_08_27"

CORPUS_KEY = "macaque_pfc_microstimulation"
BEHAVIOURAL_REFERENCE_R_UNITS = 0.14  # this project's standing minimum-detectable-difference reference
CONTINUITY_ALIGNMENT_FLOOR_ABS_COSINE = 0.05  # the pre-declared absolute-cosine floor this project's
                                               # earlier axis-alignment tests used; reported for
                                               # continuity with that prior work, not the deciding
                                               # reference for the branches fired here


# =======================================================================================================
# Checkpointing: one file per session, temp file + os.replace, completion flag written only after the
# fit returns; every checkpoint key is tagged with a schema string so a checkpoint written by an earlier
# version of this loader is a MISS rather than a silent, wrongly-shaped hit.
# =======================================================================================================

def _checkpoint_path(unit: str) -> Path:
    return CHECKPOINT_DIR / f"{unit.replace('/', '_')}.json"


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
# Loading: control and stimulated per-trial per-channel spike-count rows, identical construction to
# results/macaque_pfc_microstimulation_deviation_orthogonality_gate.json's own control-only loader for the control side, and
# to results/stimulation_latent_response_map.json's own macaque_pfc_microstimulation arm for which trials count as stimulated.
# =======================================================================================================

def _trial_rows(trials: list[dict], keep_cond) -> list[np.ndarray]:
    rows = []
    for tr in trials:
        if not keep_cond(tr["stim_cond"]):
            continue
        cropped = crop_trial(tr["spikerate"])
        if cropped is None:
            continue
        rows.append(cropped.sum(axis=0) * BIN_S)
    return rows


def _load_session_activity(prefix: str) -> dict:
    corr = load_macaque_pfc_microstimulation_session(prefix, correct=True)
    if corr is None or corr.get("control_idx") is None:
        return {"status": "excluded", "reason": "not_loadable_or_no_control_condition"}
    control_idx = corr["control_idx"]
    err = load_macaque_pfc_microstimulation_session(prefix, correct=False)

    ctrl_rows = _trial_rows(corr["trials"], lambda c: c == control_idx)
    if not ctrl_rows:
        return {"status": "excluded", "reason": "no_usable_control_trials"}
    control_activity = np.asarray(ctrl_rows, dtype=float)

    stim_rows = _trial_rows(corr["trials"], lambda c: c != control_idx)
    if err is not None:
        stim_rows += _trial_rows(err["trials"], lambda c: c != control_idx)
    if not stim_rows:
        return {"status": "excluded", "reason": "no_usable_stimulated_trials"}
    stim_activity = np.asarray(stim_rows, dtype=float)

    if control_activity.shape[1] != stim_activity.shape[1]:
        return {"status": "excluded", "reason": "control_and_stimulated_channel_counts_disagree"}

    return {
        "status": "loaded", "control_activity": control_activity, "stim_activity": stim_activity,
        "n_channels": int(control_activity.shape[1]), "channel_ids": corr["channel_ids"],
        "control_idx": control_idx, "stim_channels": corr["stim_channels"],
    }


def _parse_repeated_ascii_number(text: str) -> float:
    """The HDF5-generation session's ASCII-decoded amplitude field repeats the same number twice,
    whitespace-separated (e.g. "125  125"), the identical char-array convention run_macaque_pfc_microstimulation_pipeline.py's
    own _parse_chan_token already handles for the stimulating-channel field. Every repeat must agree; a
    disagreement would mean this field is not the constant-per-session scalar it is assumed to be here."""
    import re

    numbers = [float(x) for x in re.findall(r"\d+", text)]
    if not numbers:
        raise ValueError(f"no numeric amplitude found in {text!r}")
    if len(set(numbers)) != 1:
        raise ValueError(f"cond_uStimAmp repeats disagree: {text!r}")
    return numbers[0]


def _session_stimulation_amplitude(prefix: str) -> dict:
    """Reads params.cond_uStimAmp directly from the raw correct-trials file -- the field
    load_macaque_pfc_microstimulation_session does not surface -- decoding the same ASCII-char-array convention this
    project's own loader already uses for cond_uStimChan when the file is the HDF5 (v7.3) generation."""
    import h5py

    from run_macaque_pfc_microstimulation_pipeline import _ascii_to_str

    mat_path = DATA / "correct" / f"{prefix}.mat"
    if not mat_path.exists():
        return {"status": "not_loadable"}
    try:
        with h5py.File(str(mat_path), "r") as f:
            amp_ref = f["params"]["cond_uStimAmp"][()]
            raw = f[amp_ref[0, 0]][()]
            text = _ascii_to_str(raw).strip()
            return {"status": "computed", "value": _parse_repeated_ascii_number(text),
                    "source_field": "params.cond_uStimAmp",
                    "unit": "as recorded in the raw file; no physical unit is given in the metadata"}
    except OSError:
        pass
    import scipy.io as sio

    d = sio.loadmat(str(mat_path), simplify_cells=True)
    return {"status": "computed", "value": float(d["params"]["cond_uStimAmp"]),
            "source_field": "params.cond_uStimAmp",
            "unit": "as recorded in the raw file; no physical unit is given in the metadata"}


def reproduce_channel_filter_classification(prefix: str) -> dict:
    """Recomputes, straight from the raw channel_ids and stim_channels tables, whether at least one
    non-control stimulation condition has every one of its stimulating-electrode channel IDs present in
    this session's own (already shorted-channel-filtered) recorded channel_ids array -- the identical
    condition results/state_space_dimensionality_sweep.json's causal-microstimulation loader already
    tests when it decides whether a session's stimulation input direction is locatable in the recorded
    array. Used only as a cross-check that that sibling exclusion is a real per-session recording
    property (the stimulating electrode's own channel absent from the array, most likely already dropped
    as electrically shorted) and not a filter artifact -- never to exclude a session here, since this
    module's own construction never needs to locate the stimulating channel."""
    corr = load_macaque_pfc_microstimulation_session(prefix, correct=True)
    if corr is None or corr.get("control_idx") is None:
        return {"status": "not_loadable"}
    channel_id_set = set(int(c) for c in corr["channel_ids"])
    control_idx = corr["control_idx"]
    per_condition_survives = {}
    for c, chans in enumerate(corr["stim_channels"]):
        if c == control_idx:
            continue
        per_condition_survives[c] = bool(chans) and all(int(ch) in channel_id_set for ch in chans)
    any_survives = any(per_condition_survives.values())
    return {
        "status": "computed",
        "any_stimulation_condition_survives_channel_filter": any_survives,
        "per_condition_survives_channel_filter": {str(k): v for k, v in per_condition_survives.items()},
        "classification": "tested" if any_survives else "excluded_no_stimulation_condition_survived_channel_filter",
    }


# =======================================================================================================
# Axis estimation -- control trials only, raw and detrended arms, with the circularity guard.
# =======================================================================================================

def estimate_axis(activity: np.ndarray, trial_index: np.ndarray, source: str, detrend: bool,
                   seed_tag: str) -> dict:
    """The residual-eigenvector axis, fit on `activity` alone. `source` must be the literal string
    "control_only" -- anything else raises immediately, because no stimulated trial may ever reach this
    fit, and a caller passing e.g. "includes_stimulated_trials" is exactly the mistake this guard exists
    to catch before it can silently contaminate an axis estimate."""
    if source != "control_only":
        raise ValueError(
            "estimate_axis refuses any source other than 'control_only' -- a stimulated trial must never "
            f"enter the axis fit; got source={source!r}")
    used_activity = _linear_detrend_activity(activity, trial_index) if detrend else activity
    rows = _residual_rows(used_activity)
    if rows["n_kept"] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "too_few_trials_with_defined_direction", "n_kept": rows["n_kept"]}
    R, idx = _unit_residual_matrix(rows)
    axis = leading_eigenvector(R)
    stability = _axis_stability(R, seed_tag)
    return {"status": "computed", "axis": axis, "n_trials_kept": int(R.shape[0]), "axis_stability": stability}


def displacement_vector(control_activity: np.ndarray, stim_activity: np.ndarray) -> dict | None:
    """Mean unit-normalised direction over admitted stimulated trials minus the same over the matched
    (same-session) control trials -- rate-free on both sides, in the identical feature space the axis is
    estimated in."""
    u_ctrl = unit_direction_vectors(control_activity)
    u_stim = unit_direction_vectors(stim_activity)
    valid_ctrl = ~np.isnan(u_ctrl).any(axis=1)
    valid_stim = ~np.isnan(u_stim).any(axis=1)
    if int(valid_ctrl.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION or int(valid_stim.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return None
    mean_ctrl = u_ctrl[valid_ctrl].mean(axis=0)
    mean_stim = u_stim[valid_stim].mean(axis=0)
    direction = mean_stim - mean_ctrl
    norm = float(np.linalg.norm(direction))
    unit = (direction / norm).astype(float) if norm > 1e-12 else None
    return {
        "direction_unit": unit, "norm": norm,
        "n_control_trials_used": int(valid_ctrl.sum()), "n_stim_trials_used": int(valid_stim.sum()),
    }


def _bias_only_axis(control_activity: np.ndarray) -> np.ndarray | None:
    """The plain normalised mean unit-direction over control trials -- the one direction the residual
    axis is, by construction, orthogonal to. Standing in here for "every trial's value replaced by its
    session's own mean", this project's established bias-only pattern, applied to the axis itself rather
    than to a per-trial scalar since the primary statistic here is a single per-session direction, not a
    per-trial correlation."""
    u = unit_direction_vectors(control_activity)
    valid = ~np.isnan(u).any(axis=1)
    if int(valid.sum()) < 2:
        return None
    mean_dir = u[valid].mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_dir))
    return (mean_dir / mean_norm).astype(float) if mean_norm > 1e-12 else None


def _alignment_summary(observed: float, draws: np.ndarray) -> dict:
    finite = draws[np.isfinite(draws)]
    return {
        "observed_abs_cosine": observed,
        "observed_squared_fraction": observed ** 2,
        "null_mean_abs_cosine": float(np.mean(finite)) if finite.size else None,
        "null_sd_abs_cosine": float(np.std(finite)) if finite.size else None,
        "n_null_draws": int(finite.size),
    }


def _fit_session(prefix: str) -> dict:
    loaded = _load_session_activity(prefix)
    if loaded["status"] != "loaded":
        return loaded

    control_activity, stim_activity = loaded["control_activity"], loaded["stim_activity"]
    n_ctrl = control_activity.shape[0]
    trial_index = np.arange(n_ctrl, dtype=float)

    disp = displacement_vector(control_activity, stim_activity)
    if disp is None or disp["direction_unit"] is None:
        return {"status": "excluded", "reason": "displacement_direction_undefined_or_too_few_trials"}

    seed_base = f"macaque_pfc_microstimulation_stimulation_deviation_axis_alignment|{prefix}"
    bias_axis = _bias_only_axis(control_activity)

    by_arm = {}
    for arm_name, detrend in (("raw_axis", False), ("detrended_axis", True)):
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
        "n_channels": loaded["n_channels"],
        "n_control_trials": int(control_activity.shape[0]),
        "n_stim_trials": int(stim_activity.shape[0]),
        "n_control_trials_used_in_displacement": disp["n_control_trials_used"],
        "n_stim_trials_used_in_displacement": disp["n_stim_trials_used"],
        "displacement_norm_unit_direction_space": disp["norm"],
        "median_control_total_spike_count_per_trial": float(np.median(control_activity.sum(axis=1))),
        "by_arm": by_arm,
    }


# =======================================================================================================
# Pooling and branch classification
# =======================================================================================================

def _pool_group(per_session: dict, sessions: list[str], arm: str, field: str) -> dict:
    records = []
    for s in sessions:
        cell = per_session.get(s, {})
        arm_cell = cell.get("by_arm", {}).get(arm, {}) if cell.get("status") == "computed" else {}
        target = arm_cell.get(field)
        if target is None or target.get("draws") is None:
            continue
        records.append({"observed": target["observed_abs_cosine"], "null_draws": target["draws"]})
    pooled = _pool_rotation_statistic(records)
    return {"n_sessions_pooled": len(records), **pooled}


def _bias_only_voids(real_pooled: dict, bias_pooled: dict) -> bool:
    """Sign-and-significance-only voiding, never a magnitude comparison: the bias-only control reproduces
    the real result exactly when both are non-significant, or both are significant with the same
    above-/below-null direction."""
    if real_pooled.get("real_pooled", {}).get("status") != "tested" or bias_pooled.get("real_pooled", {}).get("status") != "tested":
        return False
    if bool(real_pooled.get("significant")) != bool(bias_pooled.get("significant")):
        return False
    if real_pooled.get("significant") and (real_pooled.get("below_null") != bias_pooled.get("below_null")):
        return False
    return True


def _classify_arm(pooled: dict, bias_pooled: dict) -> dict:
    mdd_block = pooled.get("minimum_detectable_difference_80pct_power", {})
    mdd = mdd_block.get("mdd") if isinstance(mdd_block, dict) and mdd_block.get("status") == "computed" else None
    effect = pooled.get("real_pooled", {}).get("mean_value")
    if pooled.get("real_pooled", {}).get("status") != "tested" or mdd is None or effect is None:
        return {"branch": "not_computable", "mdd": mdd, "effect": effect}
    voids = _bias_only_voids(pooled, bias_pooled)
    if voids:
        return {"branch": "displacement_direction_not_separable_from_a_unit_level_offset",
                "mdd": mdd, "effect": effect}
    if mdd >= BEHAVIOURAL_REFERENCE_R_UNITS:
        return {"branch": "inconclusive_below_detection_floor", "mdd": mdd, "effect": effect}
    if pooled.get("significant") and pooled.get("below_null") is False:
        return {"branch": "stimulation_pushes_along_the_deviation_axis", "mdd": mdd, "effect": effect}
    return {"branch": "stimulation_pushes_off_the_deviation_axis", "mdd": mdd, "effect": effect}


# =======================================================================================================
# Reproduction gates
# =======================================================================================================

def _reproduce_spike_count_medians(per_session: dict) -> dict:
    delivered = json.loads(GATE_PATH.read_text())
    delivered_medians = {r["session"]: r.get("median_total_spike_count_per_trial")
                          for r in delivered["per_session"] if r.get("status") == "computed"}
    checks = {}
    for session, delivered_median in delivered_medians.items():
        fresh = per_session.get(session, {}).get("median_control_total_spike_count_per_trial")
        matches = fresh is not None and np.isclose(fresh, delivered_median, rtol=1e-9, atol=1e-9)
        checks[session] = {"delivered": delivered_median, "fresh": fresh, "matches": bool(matches)}
    return {"status": "reproduced_exactly" if all(c["matches"] for c in checks.values()) else "not_reproduced",
            "rule": "every session's median control-trial total spike count computed here must exactly "
                    "reproduce results/macaque_pfc_microstimulation_deviation_orthogonality_gate.json's own already-delivered "
                    "value for the same session",
            "checks": checks}


def _reproduce_channel_filter(prefixes: list[str]) -> dict:
    delivered = json.loads(DIMENSIONALITY_SWEEP_PATH.read_text())
    delivered_status = {}
    for row in delivered["causal_microstimulation_sessions"]:
        session = row["session_key"].removeprefix("causal_microstim__")
        delivered_status[session] = row["status"]
    checks = {}
    for prefix in prefixes:
        fresh = reproduce_channel_filter_classification(prefix)
        delivered_here = delivered_status.get(prefix)
        fresh_status = "tested" if fresh.get("any_stimulation_condition_survives_channel_filter") else "excluded"
        checks[prefix] = {"delivered_status": delivered_here, "freshly_computed_status": fresh_status,
                           "matches": delivered_here == fresh_status, "detail": fresh}
    return {"status": "reproduced_exactly" if all(c["matches"] for c in checks.values()) else "not_reproduced",
            "rule": "every session's channel-filter tested/excluded classification recomputed here from "
                    "the raw channel_ids and stim_channels tables must exactly match results/state_space_"
                    "dimensionality_sweep.json's own causal_microstimulation_sessions classification for "
                    "the same session",
            "checks": checks}


def main() -> None:
    t0 = time.time()
    correct_dir = DATA / "correct"
    prefixes = sorted(p.stem for p in correct_dir.glob("*.mat")) if correct_dir.is_dir() else []
    n_seen = len(prefixes)

    gate = json.loads(GATE_PATH.read_text())
    gate_outcome_by_session = {r["session"]: r.get("gate_outcome") for r in gate["per_session"]}
    gate_failing_sessions = sorted(s for s, o in gate_outcome_by_session.items()
                                    if o == "fails_the_deviation_observable_does_not_separate_from_spike_count")

    per_session, excluded = {}, []
    amplitude_by_session = {}
    for prefix in prefixes:
        record = _run_checkpointed(f"session|{prefix}", lambda p=prefix: _fit_session(p))
        per_session[prefix] = record
        if record.get("status") != "computed":
            excluded.append({"session": prefix, "reason": record.get("reason", record.get("status"))})
        amplitude_by_session[prefix] = _run_checkpointed(
            f"amplitude|{prefix}", lambda p=prefix: _session_stimulation_amplitude(p))

    n_analysed = sum(1 for r in per_session.values() if r.get("status") == "computed")
    zero_drop_accounting = {
        "sessions_seen": n_seen, "sessions_analysed": n_analysed, "sessions_excluded": len(excluded),
        "exclusions": excluded, "reconciles": bool(n_seen == n_analysed + len(excluded)),
    }

    channel_filter_reproduction = _reproduce_channel_filter(prefixes)
    channel_filter_excluded_sessions = sorted(
        s for s, c in channel_filter_reproduction["checks"].items()
        if c["delivered_status"] == "excluded")
    channel_filter_tested_sessions = sorted(
        s for s, c in channel_filter_reproduction["checks"].items()
        if c["delivered_status"] == "tested")

    computed_sessions = sorted(s for s, r in per_session.items() if r.get("status") == "computed")
    pooling_groups = {
        "all_admissible_sessions": {
            "sessions": computed_sessions,
            "criterion": "every session this module can compute a control-only axis and a stimulated-"
                         "versus-control displacement on; this module never needs to locate the "
                         "stimulating electrode's own recorded channel, so the channel-filter exclusion a "
                         "sibling controllability analysis applies does not bear on this construction",
        },
        "excluding_own_gate_failing_sessions": {
            "sessions": [s for s in computed_sessions if s not in gate_failing_sessions],
            "criterion": "excludes the sessions whose own deviation-vs-spike-count orthogonality gate "
                         "failed in results/macaque_pfc_microstimulation_deviation_orthogonality_gate.json",
            "excluded_for_this_reason": [s for s in gate_failing_sessions if s in computed_sessions],
        },
        "matching_sibling_channel_filter_tested_subset": {
            "sessions": [s for s in computed_sessions if s in channel_filter_tested_sessions],
            "criterion": "restricts to the eight sessions results/state_space_dimensionality_sweep.json's "
                         "causal-microstimulation controllability analysis could also locate a stimulating "
                         "electrode's recorded channel in, reported only as a cross-check triangulating "
                         "the two exclusion criteria, not because this construction requires it",
            "excluded_for_this_reason": [s for s in channel_filter_excluded_sessions if s in computed_sessions],
        },
    }
    session_flagged_by_both_criteria = sorted(
        set(gate_failing_sessions) & set(channel_filter_excluded_sessions) & set(computed_sessions))

    arms = ("raw_axis", "detrended_axis")
    decomposition = {}
    for group_name, group in pooling_groups.items():
        sessions = group["sessions"]
        by_arm = {}
        for arm in arms:
            real_pooled = _pool_group(per_session, sessions, arm, "real_alignment")
            bias_pooled = _pool_group(per_session, sessions, arm, "bias_only_alignment")
            classification = _classify_arm(real_pooled, bias_pooled)
            secondary_squared_fraction = _pool_values_from_field(per_session, sessions, arm)
            by_arm[arm] = {
                "n_sessions_pooled": real_pooled["n_sessions_pooled"],
                "real_pooled_alignment": real_pooled,
                "bias_only_pooled_alignment": bias_pooled,
                "bias_only_reproduces_real": _bias_only_voids(real_pooled, bias_pooled),
                "branch": classification["branch"],
                "effect_abs_cosine": classification["effect"],
                "minimum_detectable_difference_80pct_power": classification["mdd"],
                "clears_behavioural_reference_0_14": (
                    classification["mdd"] < BEHAVIOURAL_REFERENCE_R_UNITS if classification["mdd"] is not None else None),
                "clears_continuity_alignment_floor_0_05": (
                    classification["mdd"] < CONTINUITY_ALIGNMENT_FLOOR_ABS_COSINE if classification["mdd"] is not None else None),
                "secondary_squared_fraction_of_displacement_captured": secondary_squared_fraction,
            }
        raw_branch = by_arm["raw_axis"]["branch"]
        detrended_branch = by_arm["detrended_axis"]["branch"]
        positive_branches = {"stimulation_pushes_along_the_deviation_axis", "stimulation_pushes_off_the_deviation_axis"}
        if raw_branch in positive_branches and detrended_branch in positive_branches:
            agreement = "agree" if raw_branch == detrended_branch else "disagree"
        else:
            agreement = "not_applicable_one_or_both_arms_not_decided"
        decomposition[group_name] = {
            "criterion": group["criterion"],
            "excluded_for_this_reason": group.get("excluded_for_this_reason", []),
            "by_arm": by_arm,
            "raw_versus_detrended_agreement": agreement,
            "licensed_directional_verdict": (
                raw_branch if agreement == "agree" else
                "stimulation_moved_the_drift_not_the_component_raw_and_detrended_arms_disagree"
                if agreement == "disagree" else
                "not_licensed_one_or_both_arms_not_decided"
            ),
        }

    reproduction_gate = {
        "spike_count_medians_against_orthogonality_gate": _reproduce_spike_count_medians(per_session),
        "channel_filter_classification_against_dimensionality_sweep": channel_filter_reproduction,
    }
    if reproduction_gate["spike_count_medians_against_orthogonality_gate"]["status"] != "reproduced_exactly":
        raise AssertionError("freshly computed control-trial spike-count medians do not reproduce "
                              "results/macaque_pfc_microstimulation_deviation_orthogonality_gate.json -- refusing to report")
    if reproduction_gate["channel_filter_classification_against_dimensionality_sweep"]["status"] != "reproduced_exactly":
        raise AssertionError("freshly computed channel-filter classification does not reproduce "
                              "results/state_space_dimensionality_sweep.json -- refusing to report")

    census = json.loads(CENSUS_PATH.read_text())["corpora"][CORPUS_KEY]

    per_session_output = {}
    for prefix, record in per_session.items():
        out = {k: v for k, v in record.items() if k != "by_arm"}
        if record.get("status") == "computed":
            out["by_arm"] = {}
            for arm, cell in record["by_arm"].items():
                cell_out = {k: v for k, v in cell.items()}
                for field in ("real_alignment", "bias_only_alignment"):
                    if cell_out.get(field) is not None:
                        cell_out[field] = {k: v for k, v in cell_out[field].items() if k != "draws"}
                out["by_arm"][arm] = cell_out
        out["gate_outcome"] = gate_outcome_by_session.get(prefix)
        out["fails_own_orthogonality_gate"] = prefix in gate_failing_sessions
        out["channel_filter_status_in_sibling_controllability_analysis"] = (
            "tested" if prefix in channel_filter_tested_sessions
            else "excluded_no_stimulation_condition_survived_channel_filter"
            if prefix in channel_filter_excluded_sessions else None)
        out["flagged_by_both_criteria"] = prefix in session_flagged_by_both_criteria
        out["stimulation_amplitude"] = amplitude_by_session.get(prefix)
        per_session_output[prefix] = out

    primary = decomposition["all_admissible_sessions"]
    verdict = (
        "Primary group (all eleven admissible sessions): raw-axis arm fires "
        f"{primary['by_arm']['raw_axis']['branch']} (effect {primary['by_arm']['raw_axis']['effect_abs_cosine']}, "
        f"mdd {primary['by_arm']['raw_axis']['minimum_detectable_difference_80pct_power']}); detrended-axis arm "
        f"fires {primary['by_arm']['detrended_axis']['branch']} (effect "
        f"{primary['by_arm']['detrended_axis']['effect_abs_cosine']}, mdd "
        f"{primary['by_arm']['detrended_axis']['minimum_detectable_difference_80pct_power']}); "
        f"raw-versus-detrended agreement: {primary['raw_versus_detrended_agreement']}; licensed verdict: "
        f"{primary['licensed_directional_verdict']}. Every trial analysed here is cropped to -0.8 to +0.7 s "
        "relative to stimulation onset, so the analysed epoch overlaps current delivery by design for the "
        "stimulated arm -- there is no isolated maintenance delay in this corpus, and this travels with any "
        "positive branch above as part of the result, not a footnote (join_askable_only_outside_a_"
        "maintenance_delay)."
    )

    output = {
        "analysis_id": "macaque_pfc_microstimulation_stimulation_deviation_axis_alignment",
        "schema_version": "1.0.0",
        "trigger": (
            "results/stimulation_axis_same_session_census.json found exactly one corpus in this project's "
            "registry -- the macaque dorsolateral prefrontal microstimulation corpus -- where a rate-free "
            "deviation axis and a delivered stimulation displacement can both be measured in the same "
            "sessions; results/macaque_pfc_microstimulation_deviation_orthogonality_gate.json confirmed the deviation observable "
            "itself passes its own spike-count orthogonality gate here as a powered null. This module asks "
            "whether the stimulation-induced displacement lands along that axis or off it, decomposed "
            "against both a raw and a detrended version of the axis because the raw version's alignment "
            "with the session's own slow drift has separately been shown mechanical rather than descriptive "
            "of the axis's identity."
        ),
        "code_commit": git_commit(ROOT),
        "corpus": CORPUS_KEY,
        "epoch_disclosure": census.get("epoch_disclosure"),
        "stimulation_kind": census.get("stimulation_kind"),
        "feature_space": "per-channel total spike count over the 30-bin (-0.8 to +0.7 s relative to "
                          "stimulation onset), 50 ms-bin window, L2-normalised to a unit direction per "
                          "trial before either the axis or the displacement is estimated -- the identical "
                          "rate-free construction rate_free_state_deviation uses internally, kept common "
                          "between the axis and the displacement so their cosine is meaningful",
        "circularity_guard": "estimate_axis raises ValueError unless called with source='control_only'; "
                              "no stimulated trial is ever passed to it in this pipeline",
        "signed_versus_magnitude_discipline": (
            "the axis is a leading eigenvector with no session-comparable sign convention established "
            "anywhere in this project; every alignment statistic here is therefore unsigned (absolute "
            "cosine and its square), and no signed-displacement language is used"
        ),
        "channel_filter_cross_check": {
            "sibling_artifact": "results/state_space_dimensionality_sweep.json, key "
                                 "causal_microstimulation_sessions",
            "sibling_exclusion_reason": "no stimulation condition survived the channel filter",
            "finding": "recomputed directly from each session's raw channel_ids and stim_channels tables: "
                       "in every one of the three sibling-excluded sessions (Sa210311_s224, Wa220803_s551, "
                       "Wa220811_s557), every non-control stimulation condition's designated electrode "
                       "channel is absent from that session's own recorded channel_ids array -- a real, "
                       "session-specific recording property (most likely the electrically-shorted-channel "
                       "exclusion this project's loader already applies removed the stimulating electrode's "
                       "own recording channel), not a filter artifact. That property is irrelevant to this "
                       "module's construction, which never needs to locate the stimulating electrode's "
                       "recorded channel index -- only the recorded population activity vector on "
                       "stimulated versus control trials, defined for all eleven sessions -- so all eleven "
                       "remain admissible here; the eight-session sibling-tested subset is reported as an "
                       "additional sensitivity group, not adopted as this module's own admissibility rule",
            "sessions_flagged_by_both_this_sibling_criterion_and_this_module's_own_spike_count_gate": session_flagged_by_both_criteria,
        },
        "own_orthogonality_gate_failing_sessions": gate_failing_sessions,
        "per_session": per_session_output,
        "pooling_groups": decomposition,
        "reproduction_gate": reproduction_gate,
        "zero_drop_accounting": zero_drop_accounting,
        "verdict": verdict,
        "wall_clock_s": round(time.time() - t0, 3),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)
    _log(f"wrote {OUTPUT_PATH} in {output['wall_clock_s']:.1f}s -- primary licensed verdict: "
         f"{primary['licensed_directional_verdict']}")


def _pool_values_from_field(per_session: dict, sessions: list[str], arm: str) -> dict:
    values = []
    for s in sessions:
        cell = per_session.get(s, {})
        arm_cell = cell.get("by_arm", {}).get(arm, {}) if cell.get("status") == "computed" else {}
        real = arm_cell.get("real_alignment")
        if real is None:
            continue
        values.append(real["observed_squared_fraction"])
    if len(values) < 2:
        return {"status": "not_computable", "n": len(values)}
    return {
        "status": "computed", "n": len(values), "mean_value": float(np.mean(values)),
        "median_value": float(np.median(values)),
        "minimum_detectable_paired_difference_at_80pct_power": minimum_detectable_paired_difference(values),
    }


if __name__ == "__main__":
    main()
