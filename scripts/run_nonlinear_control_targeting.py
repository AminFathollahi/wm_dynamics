#!/usr/bin/env python3
"""Does a nonlinear forward model of the stimulation-response dynamics recover a
targeting quantity the linear one could not?

The delivered linear control model (scripts/run_ram_openloop_pipeline.py's
`build_session_features`) fits a linear one-step operator on each session's own
control-trial latent trajectory and reports, per session, the alignment between the
stimulation input direction and that operator's leading eigenvector
("alignment_to_vstar", read fresh in results/human_stimulation_component_response.json's
block_f). That alignment is unrelated to the measured displacement of the
stimulation-response component this project tracks (results/human_stimulation_component_response.json,
block_f: correlation_displacement_vs_alignment_to_vstar). This module asks whether a
nonlinear forward model, admissible on its own held-out one-step prediction, does any
better -- using the same admitted sessions, the same displacement measurement, and the
same subject-level clustering unit block_f already uses, so the comparison is matched
at the level of data and not only of method.

Session admission is imported unchanged from
scripts/run_stimulation_site_targeting_map.py's `load_admitted_sessions`, which itself
reads results/human_stimulation_component_response.json's block_b (never recomputed
here). The outcome is that same block_b's "excluding_stimulated_shank" displacement
value, read exactly as block_f reads it -- signed, not normalised, no absolute value.

The nonlinear targeting quantity: for each admitted session, a one-step forward map of
the session's own control-trial-only latent trajectory (top eight principal components
of the full, unrestricted channel set -- the same channel set the delivered linear
alignment quantity was computed on) is fit by two flexible, non-parametric candidate
families (gradient-boosted regression and Nystroem-approximated kernel ridge
regression, the same two families and the same held-out-fold / same-session-shuffle
admissibility test this project already applies to its one-step dynamics elsewhere).
Whichever family clears its own same-session shuffle on a majority of subjects is kept;
if neither does, no targeting test is run and that is reported as the result on its own
terms. For the kept family, the targeting quantity is that model's own directional
derivative: the norm of the predicted one-step change in state produced by adding a
unit input at the driven channel (projected into the same latent space, before any
cosine normalisation -- the same input vector the linear model's own alignment
quantity is built from) to the session's own pre-stimulation state, averaged over that
session's stimulated trials' pre-word-onset bins -- never a regression fit to the
displacement outcome itself.

Outputs:
  results/nonlinear_control_targeting.json

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_nonlinear_control_targeting.py [--smoke N]
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import json
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.kernel_approximation import Nystroem  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.multioutput import MultiOutputRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from geometry import pca_decompose  # noqa: E402
from provenance import canonical_json, checkpoint_safe, git_commit, restore_checkpoint  # noqa: E402
from statistics import stable_seed  # noqa: E402
import run_nonlinearity_onestep as nlo  # noqa: E402  (admissibility helpers: _pairs, _cv_r2, _null_r2)
from run_human_stimulation_component_response import (  # noqa: E402
    ALPHA,
    MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
    load_raw_features,
    subject_aggregated_correlation,
)
from run_ram_openloop_pipeline import BIN_S, N_PC, PRE_S  # noqa: E402
from run_stimulation_site_targeting_map import load_admitted_sessions  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "nonlinear_control_targeting.json"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_nonlinear_control_targeting"
COMPONENT_RESPONSE_PATH = RESULTS / "human_stimulation_component_response.json"

SCHEMA = "nonlinear_control_targeting_v2"  # v2: held-out shuffle null (v1's in-sample null was unusable)

# Pre-declared before any session's fit is looked at.
MIN_CONTROL_TRIALS_FOR_DYNAMICS_FIT = 20  # need enough (trial, bin) pairs for a 5-fold CV nonlinear fit
CHANNEL_CONDITION_FOR_DISPLACEMENT = "excluding_stimulated_shank"  # unchanged from block_f's own choice
N_PRE_BINS = max(1, int(round(PRE_S / BIN_S)))  # pre-word-onset bins -- before any stimulation pulse
GATE_MAJORITY_THRESHOLD = 0.5  # same convention run_state_space_estimation_admissibility.py uses


# ── Checkpointing (fit-level, crash-proof; schema-tagged so a stale entry is a miss) ──

def _checkpoint_path(unit: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", unit)
    return CHECKPOINT_DIR / f"{safe}.json"


def load_checkpoint(unit: str) -> dict | None:
    path = _checkpoint_path(unit)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True or data.get("_schema") != SCHEMA:
        return None
    return restore_checkpoint(data["record"])


def save_checkpoint(unit: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(unit)
    payload = {"_complete": True, "_schema": SCHEMA, "record": checkpoint_safe(record)}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def run_checkpointed(unit: str, fit_fn):
    cached = load_checkpoint(unit)
    if cached is not None:
        return cached
    record = fit_fn()
    save_checkpoint(unit, record)
    return record


# ── Per-session nonlinear dynamics fit, admissibility, and targeting quantity ──

def _make_gbr():
    return MultiOutputRegressor(GradientBoostingRegressor(random_state=0), n_jobs=nlo.GBR_N_JOBS)


def _make_krr(n_landmarks: int):
    return make_pipeline(Nystroem(kernel="rbf", n_components=n_landmarks, random_state=0), Ridge(alpha=1.0))


def stimulation_input_latent_direction(V: np.ndarray, ch_names: list[str], stim_channel: str) -> np.ndarray | None:
    """The unit input at the driven bipolar channel, in the z-scored full-channel-set
    observation space (a one-hot vector, value 1.0 at that channel), projected into the
    same latent space the forward model operates in -- the same input vector the linear
    model's own alignment quantity is built from, before that quantity's final cosine
    normalisation. None if the channel is not in this session's own channel set."""
    if stim_channel not in ch_names:
        return None
    b_chan = np.zeros(len(ch_names))
    b_chan[ch_names.index(stim_channel)] = 1.0
    return V.T @ b_chan


def _cv_r2_shuffle_null(Z_trials: np.ndarray, make_model, rng: np.random.Generator) -> float:
    """The admissibility comparison the fit needs: the SAME trial-wise CV fold structure as
    nlo._cv_r2, but each fold's training pairs are built from independently, per-trial
    circularly-shifted trajectories (the same shift convention nlo._null_r2 uses to destroy
    cross-trial structure) before fitting, then scored against the REAL held-out test pairs.
    nlo._null_r2 itself fits and scores its shuffled ensemble in-sample, which inflates a
    flexible regressor's null score above its own genuinely held-out real score for reasons that
    have nothing to do with same-session shuffle structure -- that comparison is not usable as an
    admissibility gate here, so this fold-matched, held-out version is used instead."""
    N, T, d = Z_trials.shape
    trial_idx = rng.permutation(N)
    folds = np.array_split(trial_idx, min(nlo.N_SPLITS, N))
    scores = []
    for k in range(len(folds)):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        if len(tr) < 2 or len(te) < 1:
            continue
        Z_shift_tr = np.empty((len(tr), T, d))
        for i, idx in enumerate(tr):
            shift = int(rng.integers(1, max(T - 1, 2)))
            Z_shift_tr[i] = np.roll(Z_trials[idx], shift, axis=0)
        X1_tr, X2_tr = nlo._pairs(Z_shift_tr)
        X1_te, X2_te = nlo._pairs(Z_trials[te])
        model = make_model()
        model.fit(X1_tr, X2_tr)
        scores.append(nlo._r2(model.predict(X1_te), X2_te))
    return float(np.mean(scores)) if scores else float("nan")


def _fit_session_dynamics(corpus: str, session_key: str, stim_channel: str) -> dict:
    raw = load_raw_features(f"{corpus}__{session_key}")
    if raw is None:
        return {"status": "excluded", "reason": "raw_epoched_features_not_cached"}
    ch_names = [str(c) for c in raw["ch_names"].tolist()]
    stim_flag = np.asarray(raw["stim_flag"])
    epochs_log = np.asarray(raw["epochs_log"], dtype=float)
    n_trials, n_bins, n_ch = epochs_log.shape

    if stim_channel not in ch_names:
        return {"status": "excluded", "reason": "stim_channel_not_in_full_channel_set"}

    ctrl_mask = stim_flag == 0
    n_ctrl = int(ctrl_mask.sum())
    if n_ctrl < MIN_CONTROL_TRIALS_FOR_DYNAMICS_FIT:
        return {"status": "excluded", "reason": "too_few_control_trials_for_dynamics_fit",
                "n_control_trials": n_ctrl}

    stim_mask = stim_flag == 1
    if int(stim_mask.sum()) < 1:
        return {"status": "excluded", "reason": "no_stimulated_trials"}

    mu = epochs_log[ctrl_mask].mean(axis=(0, 1))
    sd = epochs_log[ctrl_mask].std(axis=(0, 1)) + 1e-8
    epochs_z = (epochs_log - mu) / sd
    X_ctrl_flat = epochs_z[ctrl_mask].reshape(-1, n_ch)
    if not np.all(np.isfinite(X_ctrl_flat)) or X_ctrl_flat.std() < 1e-8:
        return {"status": "excluded", "reason": "degenerate_control_activity"}

    _, V, var_ratio = pca_decompose(X_ctrl_flat, N_PC)
    k = V.shape[1]
    x_mean = X_ctrl_flat.mean(axis=0)
    Z_ctrl_trials = ((X_ctrl_flat - x_mean) @ V).reshape(n_ctrl, n_bins, k)
    Z_stim_trials = ((epochs_z[stim_mask].reshape(-1, n_ch) - x_mean) @ V).reshape(int(stim_mask.sum()), n_bins, k)

    b_lat = stimulation_input_latent_direction(V, ch_names, stim_channel)
    x_pre = Z_stim_trials[:, :N_PRE_BINS, :].mean(axis=(0, 1))

    n_pairs = n_ctrl * (n_bins - 1)
    rng_gbr = np.random.default_rng(stable_seed(f"nonlinear_control_targeting_gbr|{corpus}|{session_key}"))
    r2_cv_gbr = nlo._cv_r2(Z_ctrl_trials, _make_gbr, rng_gbr)
    r2_null_gbr = _cv_r2_shuffle_null(Z_ctrl_trials, _make_gbr, rng_gbr)

    n_landmarks = min(nlo.KRR_LANDMARKS, n_pairs)
    rng_krr = np.random.default_rng(stable_seed(f"nonlinear_control_targeting_krr|{corpus}|{session_key}"))
    make_krr = lambda: _make_krr(n_landmarks)  # noqa: E731
    r2_cv_krr = nlo._cv_r2(Z_ctrl_trials, make_krr, rng_krr)
    r2_null_krr = _cv_r2_shuffle_null(Z_ctrl_trials, make_krr, rng_krr)

    admissibility = {
        "gbr": {"r2_cv": r2_cv_gbr, "r2_null": r2_null_gbr,
               "clears_shuffle": bool(np.isfinite(r2_cv_gbr) and np.isfinite(r2_null_gbr) and r2_cv_gbr > r2_null_gbr)},
        "krr": {"r2_cv": r2_cv_krr, "r2_null": r2_null_krr,
               "clears_shuffle": bool(np.isfinite(r2_cv_krr) and np.isfinite(r2_null_krr) and r2_cv_krr > r2_null_krr)},
    }

    targeting = {}
    X1, X2 = nlo._pairs(Z_ctrl_trials)
    for name, make_model in (("gbr", _make_gbr), ("krr", make_krr)):
        model = make_model()
        model.fit(X1, X2)
        f_pre = model.predict(x_pre.reshape(1, -1))[0]
        f_pre_plus = model.predict((x_pre + b_lat).reshape(1, -1))[0]
        targeting[name] = float(np.linalg.norm(f_pre_plus - f_pre))

    return {
        "status": "computed", "n_control_trials": n_ctrl, "n_stim_trials": int(stim_mask.sum()),
        "n_channels": n_ch, "n_latent_dims": k, "var_explained": float(var_ratio.sum()),
        "admissibility": admissibility,
        "predicted_state_shift_for_unit_input": targeting,
    }


# ── Admissibility gate: majority of subjects clear each family's own shuffle ──

def admissibility_gate(per_session: dict, subject_of: dict, family: str) -> dict:
    by_subject = defaultdict(list)
    for session_key, rec in per_session.items():
        if rec.get("status") != "computed":
            continue
        by_subject[subject_of[session_key]].append(bool(rec["admissibility"][family]["clears_shuffle"]))
    if not by_subject:
        return {"status": "not_computable"}
    unit_outcomes = {s: (sum(v) / len(v)) > 0.5 for s, v in by_subject.items()}
    n_total = len(unit_outcomes)
    n_passed = sum(unit_outcomes.values())
    fraction = n_passed / n_total if n_total else 0.0
    return {"status": "computed", "n_subjects": n_total, "n_subjects_passed": n_passed,
            "fraction_passed": fraction, "admissible": fraction > GATE_MAJORITY_THRESHOLD}


def choose_family(gate_gbr: dict, gate_krr: dict, per_session: dict) -> dict:
    def mean_delta(family):
        # r2_cv/r2_null may come back as None after a checkpoint round trip (a non-finite float
        # is serialised as JSON null, not NaN) -- treat None exactly like a non-finite float here.
        deltas = []
        for rec in per_session.values():
            if rec.get("status") != "computed":
                continue
            cv, null = rec["admissibility"][family]["r2_cv"], rec["admissibility"][family]["r2_null"]
            if cv is None or null is None or not (np.isfinite(cv) and np.isfinite(null)):
                continue
            deltas.append(cv - null)
        return float(np.mean(deltas)) if deltas else float("nan")

    candidates = [(name, gate) for name, gate in (("gbr", gate_gbr), ("krr", gate_krr))
                 if gate.get("status") == "computed" and gate.get("admissible")]
    if not candidates:
        return {"chosen": None, "reason": "no_admissible_nonlinear_forward_model"}
    ranked = sorted(candidates, key=lambda item: mean_delta(item[0]), reverse=True)
    chosen = ranked[0][0]
    return {"chosen": chosen, "reason": "highest_mean_held_out_minus_shuffle_r2_among_admissible_families",
            "mean_delta_r2_by_family": {name: mean_delta(name) for name, _ in candidates}}


# ── Bias-only control: leave-one-subject-out mean substitution ──

def leave_one_subject_out_bias_only(values: np.ndarray, subjects: list) -> np.ndarray:
    """Replaces every session's targeting quantity with the mean of every OTHER
    subject's own sessions -- that subject's own value contributes nothing to its own
    substitute, so a real association surviving this replacement cannot be an artifact
    of one subject's own scale."""
    values = np.asarray(values, dtype=float)
    subjects = np.asarray(subjects)
    out = np.full_like(values, np.nan)
    for s in set(subjects.tolist()):
        others = values[subjects != s]
        finite = others[np.isfinite(others)]
        out[subjects == s] = float(np.mean(finite)) if len(finite) else np.nan
    return out


# ── Branch classification, pre-declared ──

def classify_branch(corr: dict, mdd_reference: float = MEANINGFUL_EFFECT_THRESHOLD_R_UNITS) -> str:
    if corr.get("status") != "computed":
        return "not_computable"
    significant = corr["p_value"] <= ALPHA and not (corr["ci_lower"] <= 0.0 <= corr["ci_upper"])
    if significant:
        return "nonlinear_controller_recovers_a_targeting_quantity"
    mdd = corr.get("mdd", {})
    if mdd.get("status") == "computed" and mdd["mdd"] < mdd_reference:
        return "targeting_claim_retired_no_controller_predicts_its_own_intervention"
    return "inconclusive_below_detection_floor"


def bias_only_voids(real_corr: dict, bias_corr: dict) -> bool:
    """Voiding is on sign and significance only, never a magnitude comparison."""
    if real_corr.get("status") != "computed" or bias_corr.get("status") != "computed":
        return False
    real_sig = real_corr["p_value"] <= ALPHA
    bias_sig = bias_corr["p_value"] <= ALPHA
    same_sign = (real_corr["r"] < 0) == (bias_corr["r"] < 0)
    return bool(real_sig and bias_sig and same_sign)


# ── Reproduction gate against the delivered linear artifact ──

def reproduction_gate(admitted_sessions: list[dict], component_response: dict) -> dict:
    block_b = component_response.get("block_b", {}).get("per_session", {})
    exact, mismatched, missing = 0, [], []
    for rec in admitted_sessions:
        cond = rec["displacement_conditions"].get(CHANNEL_CONDITION_FOR_DISPLACEMENT, {})
        if cond.get("status") != "computed":
            continue
        source = block_b.get(rec["session_key"], {}).get("conditions", {}).get(CHANNEL_CONDITION_FOR_DISPLACEMENT, {})
        if source.get("status") != "computed":
            missing.append(rec["session_key"])
            continue
        if cond["displacement"] == source["displacement"]:
            exact += 1
        else:
            mismatched.append(rec["session_key"])
    return {"n_exact": exact, "n_mismatched": len(mismatched), "mismatched_sessions": mismatched,
            "n_missing_in_source": len(missing), "outcome": "exact" if not mismatched and not missing else "not_exact"}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=None,
                        help="limit each corpus to the first N admitted sessions, for a fast dev-time check")
    args = parser.parse_args()
    t0 = time.time()

    all_sessions = load_admitted_sessions()
    open_sessions = [s for s in all_sessions if s["corpus"] == "open_loop_ds005489"]
    closed_sessions = [s for s in all_sessions if s["corpus"] == "closed_loop_ds005557"]
    if args.smoke is not None:
        open_sessions = open_sessions[:args.smoke]
        closed_sessions = closed_sessions[:args.smoke]
    admitted_sessions = open_sessions + closed_sessions
    n_seen = len(admitted_sessions)

    per_session_dynamics = {}
    exclusion_reasons = defaultdict(int)
    subject_of = {}
    for rec in admitted_sessions:
        subject_of[rec["session_key"]] = rec["subject"]
        key = f"dynamics__{rec['corpus']}__{rec['session_key']}"
        result = run_checkpointed(
            key, lambda rec=rec: _fit_session_dynamics(rec["corpus"], rec["session_key"], rec["stim_channel"]))
        per_session_dynamics[rec["session_key"]] = result
        if result["status"] != "computed":
            exclusion_reasons[result["reason"]] += 1

    gate_gbr = admissibility_gate(per_session_dynamics, subject_of, "gbr")
    gate_krr = admissibility_gate(per_session_dynamics, subject_of, "krr")
    choice = choose_family(gate_gbr, gate_krr, per_session_dynamics)

    component_response = json.loads(COMPONENT_RESPONSE_PATH.read_text())
    repro = reproduction_gate(admitted_sessions, component_response)

    output = {
        "version": "2026-08-27",
        "scope": (
            "Human intracranial free-recall stimulation, both corpora pooled -- open-loop "
            "(experimenter-scheduled) and closed-loop (classifier-triggered, causal:false) -- exactly "
            "the session set the delivered linear alignment-to-leading-eigenvector targeting quantity "
            "was tested against. Session is the unit of analysis; subject is the clustering unit for "
            "every correlation. The outcome is the delivered, unmodified stimulation-response "
            "component displacement (excluding_stimulated_shank channel condition, signed, not "
            "normalised) -- the same value the delivered linear result was tested against."
        ),
        "zero_drop_accounting": {
            "n_sessions_seen_total": n_seen,
            "n_sessions_open_loop": len(open_sessions), "n_sessions_closed_loop": len(closed_sessions),
            "n_subjects_seen": len({s["subject"] for s in admitted_sessions}),
            "n_subjects_open_loop": len({s["subject"] for s in open_sessions}),
            "n_subjects_closed_loop": len({s["subject"] for s in closed_sessions}),
            "dynamics_fit_exclusions_by_reason": dict(exclusion_reasons),
        },
        "reproduction_gate_against_delivered_linear_displacement": repro,
        "admissibility": {"gbr": gate_gbr, "krr": gate_krr, "family_choice": choice},
        "parameters": {
            "n_latent_dims_requested": N_PC,
            "min_control_trials_for_dynamics_fit": MIN_CONTROL_TRIALS_FOR_DYNAMICS_FIT,
            "channel_condition_for_displacement_outcome": CHANNEL_CONDITION_FOR_DISPLACEMENT,
            "n_pre_stimulus_bins": N_PRE_BINS, "pre_stimulus_window_s": PRE_S, "bin_width_s": BIN_S,
            "admissibility_cv_splits": nlo.N_SPLITS, "admissibility_gate_majority_threshold": GATE_MAJORITY_THRESHOLD,
            "alpha": ALPHA, "meaningful_effect_threshold_r_units": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
            "seeding": "deterministic per-session via statistics.stable_seed on corpus|session_key strings",
        },
    }

    if choice["chosen"] is None:
        output["branch"] = "no_admissible_nonlinear_forward_model"
        output["targeting_test"] = "not_run_no_admissible_forward_model"
        output["wall_clock_s"] = time.time() - t0
        output["code_commit"] = git_commit(ROOT)
        _write(output)
        print(f"Wrote {OUTPUT_PATH} ({output['wall_clock_s']:.1f}s) -- branch: {output['branch']}")
        return

    family = choice["chosen"]
    targeting_vals, outcome_vals, subjects, session_keys, per_session_out = [], [], [], [], {}
    for rec in admitted_sessions:
        dyn = per_session_dynamics[rec["session_key"]]
        disp_cond = rec["displacement_conditions"].get(CHANNEL_CONDITION_FOR_DISPLACEMENT, {})
        row = {"subject": rec["subject"], "corpus": rec["corpus"], "dynamics": dyn}
        if dyn.get("status") == "computed" and disp_cond.get("status") == "computed":
            targeting_vals.append(dyn["predicted_state_shift_for_unit_input"][family])
            outcome_vals.append(disp_cond["displacement"])
            subjects.append(rec["subject"])
            session_keys.append(rec["session_key"])
            row["targeting_quantity"] = dyn["predicted_state_shift_for_unit_input"][family]
            row["displacement"] = disp_cond["displacement"]
        per_session_out[rec["session_key"]] = row

    targeting_vals = np.array(targeting_vals)
    outcome_vals = np.array(outcome_vals)
    real_corr = subject_aggregated_correlation(targeting_vals, outcome_vals, subjects)
    branch = classify_branch(real_corr)

    bias_only_block = None
    if branch == "nonlinear_controller_recovers_a_targeting_quantity":
        bias_vals = leave_one_subject_out_bias_only(targeting_vals, subjects)
        bias_corr = subject_aggregated_correlation(bias_vals, outcome_vals, subjects)
        voided = bias_only_voids(real_corr, bias_corr)
        bias_only_block = {
            "correlation": bias_corr, "voids_the_real_result": voided,
            "control_power_check": {
                "real_r": real_corr.get("r"), "bias_only_r": bias_corr.get("r"),
                "effect_moved": (abs(bias_corr.get("r", 0.0)) < abs(real_corr.get("r", 0.0)) * 0.5
                                if bias_corr.get("status") == "computed" and real_corr.get("status") == "computed"
                                else None),
            },
        }
        if voided:
            branch = "targeting_claim_retired_no_controller_predicts_its_own_intervention"

    output["family_used"] = family
    output["targeting_predictor_definition"] = (
        "The norm of the predicted one-step latent-state change produced by adding a unit input "
        "vector at the driven bipolar channel (the same one-hot-in-channel-space, PCA-projected input "
        "direction the delivered linear alignment quantity is built from) to the session's own "
        "pre-stimulation state (the mean latent state over stimulated trials' pre-word-onset bins), "
        "evaluated through the session's own control-trial-fitted, admissible nonlinear one-step "
        "forward map. Never fit to the displacement outcome it is tested against."
    )
    output["n_sessions_in_correlation"] = len(session_keys)
    output["n_subjects_in_correlation"] = len(set(subjects))
    output["correlation_targeting_quantity_vs_displacement"] = real_corr
    output["bias_only_control"] = bias_only_block
    output["branch"] = branch
    output["meaningful_effect_threshold_r_units"] = MEANINGFUL_EFFECT_THRESHOLD_R_UNITS
    output["per_session"] = per_session_out
    output["wall_clock_s"] = time.time() - t0
    output["code_commit"] = git_commit(ROOT)
    _write(output)
    print(f"Wrote {OUTPUT_PATH} ({output['wall_clock_s']:.1f}s) -- branch: {branch}, "
          f"r={real_corr.get('r')}, p={real_corr.get('p_value')}, n_subjects={output['n_subjects_in_correlation']}")


def _write(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rendered = canonical_json(output)
    if "Infinity" in rendered or "NaN" in rendered:
        raise RuntimeError("non-finite token leaked into JSON output -- fix the offending field before writing")
    fd, tmp_name = tempfile.mkstemp(dir=str(OUTPUT_PATH.parent), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(rendered)
        os.replace(tmp_name, OUTPUT_PATH)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


if __name__ == "__main__":
    main()
